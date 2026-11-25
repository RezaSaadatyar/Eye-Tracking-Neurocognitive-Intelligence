"""Training-only exploratory associations with crossed participant/paragraph effects."""
import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from reading_features import GAZE


def reading_associations(table):
    """Report approximate VB posterior summaries, not confirmatory p-values."""
    rows = []
    for feature in GAZE:
        data = table[["participant_id", "paragraph_key", "difficulty_level",
                      "is_correct", feature]].dropna().copy()
        if data.empty or data[feature].std() <= 0 or data.is_correct.nunique() < 2:
            continue
        data["z"] = (data[feature] - data[feature].mean()) / data[feature].std()
        model = BinomialBayesMixedGLM.from_formula(
            "is_correct ~ z + C(difficulty_level)",
            {"person": "0 + C(participant_id)", "paragraph": "0 + C(paragraph_key)"},
            data, vcp_p=0.5, fe_p=2)
        fit = model.fit_vb()
        i = model.exog_names.index("z")
        estimate, sd = float(fit.fe_mean[i]), float(fit.fe_sd[i])
        rows.append(dict(feature=feature, n_trials=len(data),
                         n_people=data.participant_id.nunique(),
                         n_paragraphs=data.paragraph_key.nunique(),
                         log_odds_per_sd=estimate, posterior_sd=sd,
                         approx_low=estimate-1.96*sd, approx_high=estimate+1.96*sd,
                         converged=bool(fit.optim_retvals.get("success", False))))
    return pd.DataFrame(rows)
