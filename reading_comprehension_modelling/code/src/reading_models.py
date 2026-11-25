"""Fixed classical and sequence candidates; fit only on Training, score held-out rows."""
import copy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from reading_features import GAZE, TEXT

MODALITIES = {"gaze": GAZE, "text": TEXT, "fusion": GAZE + TEXT}
CLASSICAL = ["prior"] + [f"{model}_{mode}" for model in ("logistic", "forest", "xgb")
                         for mode in MODALITIES] + ["tfidf_text", "tfidf_fusion"]
NEURAL = ["mlp_text"] + [f"{model}_{mode}" for model in ("cnn", "gru", "transformer")
                        for mode in ("gaze", "fusion")]


def role_indices(table, split):
    role = table[["participant_id"]].merge(split[["participant_id", "role"]],
                                          on="participant_id", validate="many_to_one").role
    if role.isna().any():
        raise ValueError("Unassigned participant")
    return {r: np.flatnonzero(role.eq(r)) for r in ("train", "validation", "test")}


def predictions(table, protocol, name, roles, probs):
    return pd.DataFrame([dict(protocol=protocol, model=name, role=role,
                              trial_id=table.trial_id.iloc[i], p_correct=float(p))
                         for role in ("validation", "test")
                         for i, p in zip(roles[role], probs[role])])


def classical_predictions(table, splits, paragraph_text, seed=17):
    from xgboost import XGBClassifier
    result, importance = [], []
    scorer = make_scorer(f1_score, labels=[0,1], average="macro", zero_division=0)
    y = table.is_correct.to_numpy(int)
    text = table[["trial_id"]].merge(paragraph_text, on="trial_id",
                                     validate="one_to_one", how="left")
    if text.paragraph.isna().any() or len(text) != len(table):
        raise ValueError("Missing paragraph text for TF-IDF baseline")
    with_text = table.assign(paragraph=text.paragraph.to_numpy())
    for protocol, split in splits.groupby("protocol"):
        roles = role_indices(table, split)
        tr = roles["train"]
        for name in CLASSICAL:
            if name == "prior":
                probs = {r: np.full(len(roles[r]), y[tr].mean())
                         for r in ("validation", "test")}
            elif name.startswith("tfidf_"):
                mode = name.split("_")[1]
                numeric = TEXT if mode == "text" else GAZE + TEXT
                transform = ColumnTransformer([
                    ("text", TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                             max_features=5000), "paragraph"),
                    ("numeric", make_pipeline(SimpleImputer(strategy="median",
                                                             keep_empty_features=True),
                                              StandardScaler()), numeric),
                ])
                pipe = make_pipeline(transform, LogisticRegression(
                    max_iter=1000, C=1, random_state=seed))
                pipe.fit(with_text.iloc[tr], y[tr],
                         logisticregression__sample_weight=compute_sample_weight("balanced", y[tr]))
                probs = {r: pipe.predict_proba(with_text.iloc[roles[r]])[:,1]
                         for r in ("validation","test")}
            else:
                family, mode = name.split("_")
                estimator = {
                    "logistic": LogisticRegression(max_iter=1000, C=1, random_state=seed),
                    "forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                       n_jobs=2, random_state=seed),
                    "xgb": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.05,
                                         n_jobs=2, random_state=seed, eval_metric="logloss"),
                }[family]
                x = table[MODALITIES[mode]].to_numpy(float)
                pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                     StandardScaler(), estimator)
                weights = compute_sample_weight("balanced", y[tr])
                pipe.fit(x[tr], y[tr], **{pipe.steps[-1][0]+"__sample_weight": weights})
                probs = {r: pipe.predict_proba(x[roles[r]])[:, 1]
                         for r in ("validation", "test")}
                imp = permutation_importance(pipe, x[roles["validation"]],
                                             y[roles["validation"]], scoring=scorer,
                                             n_repeats=3, random_state=seed, n_jobs=1)
                importance.extend(dict(protocol=protocol, model=name, feature=feature,
                                       macro_f1_decrease=float(mean), repeat_sd=float(sd))
                                  for feature, mean, sd in zip(MODALITIES[mode],
                                                                imp.importances_mean,
                                                                imp.importances_std))
            result.append(predictions(table, protocol, name, roles, probs))
    return pd.concat(result, ignore_index=True), pd.DataFrame(importance)


def neural_predictions(table, splits, x, mask, seed=17, epochs=15, patience=3):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class Reader(nn.Module):
        def __init__(self, family, mode, n_seq, n_text):
            super().__init__()
            self.family, self.mode = family, mode
            if family == "cnn":
                self.encoder = nn.Sequential(nn.Conv1d(n_seq, 32, 5, padding=2), nn.ReLU(),
                                             nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
            elif family == "gru":
                self.encoder = nn.GRU(n_seq, 32, batch_first=True)
            elif family == "transformer":
                self.proj = nn.Linear(n_seq, 32)
                self.position = nn.Parameter(torch.zeros(1, x.shape[1], 32))
                nn.init.normal_(self.position, std=.02)
                layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, 2)
            dim = (0 if mode == "text" else 32) + (n_text if mode != "gaze" else 0)
            self.head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, 1))

        def forward(self, seq, valid, text):
            if self.mode == "text":
                return self.head(text).squeeze(-1)
            if self.family == "cnn":
                # Mask after each convolution so padding cannot enter a subsequent layer.
                z = seq.transpose(1, 2)
                for layer in self.encoder:
                    z = layer(z) * valid[:, None, :]
                z = z.transpose(1, 2)
            elif self.family == "gru":
                z, _ = self.encoder(seq)
            else:
                z = self.encoder(self.proj(seq) + self.position[:, :seq.size(1)],
                                 src_key_padding_mask=~valid)
            z = (z * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
            if self.mode == "fusion":
                z = torch.cat([z, text], dim=1)
            return self.head(z).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    y = table.is_correct.to_numpy(np.float32)
    result, history, configs = [], [], []
    for protocol, split in splits.groupby("protocol"):
        roles = role_indices(table, split)
        tr = roles["train"]
        # Fit numeric transformations exclusively on valid Training observations.
        train_values = x[tr][mask[tr]]
        centre, scale = train_values.mean(0), train_values.std(0)
        scale[scale < 1e-6] = 1.
        xx = np.where(mask[..., None], (x-centre)/scale, 0).astype(np.float32)
        text_pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler())
        text = text_pipe.fit(table.iloc[tr][TEXT]).transform(table[TEXT]).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(xx), torch.from_numpy(mask),
                           torch.from_numpy(text), torch.from_numpy(y))

        def loader(ids, shuffle=False):
            return DataLoader(torch.utils.data.Subset(ds, ids.tolist()), batch_size=64,
                              shuffle=shuffle, generator=torch.Generator().manual_seed(seed))

        for name in NEURAL:
            family, mode = name.split("_")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = Reader(family, mode, x.shape[-1], text.shape[-1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            ratio = float((y[tr] == 0).sum()/(y[tr] == 1).sum())
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))
            best, wait, state = -np.inf, 0, None

            def predict(ids):
                model.eval()
                with torch.no_grad():
                    return np.concatenate([torch.sigmoid(model(a.to(device), b.to(device),
                                                               c.to(device))).cpu().numpy()
                                           for a, b, c, _ in loader(ids)])

            for epoch in range(1, epochs+1):
                model.train()
                total, n = 0., 0
                for a, b, c, target in loader(tr, True):
                    opt.zero_grad()
                    loss = loss_fn(model(a.to(device), b.to(device), c.to(device)), target.to(device))
                    loss.backward()
                    opt.step()
                    total += loss.item()*len(target)
                    n += len(target)
                pv = predict(roles["validation"])
                score = f1_score(y[roles["validation"]], pv >= .5, labels=[0,1],
                                 average="macro", zero_division=0)
                history.append(dict(protocol=protocol, model=name, epoch=epoch,
                                    train_loss=total/n, validation_macro_f1=score))
                if not np.isfinite(total/n) or not np.isfinite(pv).all():
                    raise ValueError("Non-finite neural training or prediction")
                if score > best:
                    best, wait, state = score, 0, copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                else:
                    wait += 1
                    if wait >= patience:
                        break
            if state is None:
                raise ValueError("No finite Validation score")
            model.load_state_dict(state)
            probs = {r: predict(roles[r]) for r in ("validation", "test")}
            result.append(predictions(table, protocol, name, roles, probs))
            configs.append(dict(protocol=protocol, model=name, best_epoch=best_epoch,
                                epochs_run=epoch, validation_macro_f1=best,
                                parameters=sum(p.numel() for p in model.parameters()
                                               if p.requires_grad),
                                seed=seed, batch_size=64, learning_rate=.001,
                                weight_decay=.0001, max_epochs=epochs, patience=patience,
                                pos_weight=ratio, device=str(device),
                                n_train=len(tr), n_validation=len(roles["validation"]),
                                n_test=len(roles["test"])))
            print(protocol, name, "Validation macro F1:", round(best, 4))
    return pd.concat(result, ignore_index=True), pd.DataFrame(history), pd.DataFrame(configs)
