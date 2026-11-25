"""Small deterministic synthetic checks; no original OneStop data required."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from onestop_io import make_cohort, make_splits, validate_splits
from reading_features import build_features, TEXT
from reading_models import CLASSICAL, NEURAL
from reading_evaluation import evaluate, participant_bootstrap, metric


def synthetic_reports():
    fix, ia = [], []
    for b in range(1,4):
        for p in range(5):
            for trial in range(2):
                base=dict(participant_id=f"b{b}p{p}", trial_index=str(trial),
                          article_batch=str(b), article_id="1", paragraph_id=str(trial+1),
                          difficulty_level="Adv", onestopqa_question_id="0",
                          question_preview="False", practice_trial="False",
                          repeated_reading_trial="False", selected_answer="A" if trial else "B",
                          is_correct="True" if trial else "False", paragraph="Some text")
                for j in range(2):
                    fix.append(dict(**base,CURRENT_FIX_START=j*100,CURRENT_FIX_DURATION=100,
                                    CURRENT_FIX_X=j*20,CURRENT_FIX_Y=10,
                                    CURRENT_FIX_INTEREST_AREA_INDEX=j+1))
                    ia.append(dict(**base,IA_ID=j+1,IA_LABEL=["Some","text"][j],
                                   IA_DWELL_TIME=100,IA_FIXATION_COUNT=1,word_length=4,
                                   wordfreq_frequency=3,gpt2_surprisal=2))
    return pd.DataFrame(fix),pd.DataFrame(ia)


class ReadingContracts(unittest.TestCase):
    def test_cohort_and_sequences(self):
        fix,ia=synthetic_reports()
        trial,exclusions=make_cohort(fix,ia)
        self.assertEqual(len(trial),30)
        split=make_splits(trial)
        validate_splits(trial,split)
        table,x,mask=build_features(trial,fix,ia,max_fixations=4)
        self.assertEqual(x.shape,(30,4,4))
        self.assertTrue((mask.sum(1)==2).all())
        self.assertTrue((table.fix_total_ms==200).all())
        self.assertTrue((table.visited_word_fraction==1).all())
        self.assertNotIn("selected_answer",table.columns)
        self.assertNotIn("is_correct",TEXT)
        bad=split.copy()
        p=bad[(bad.protocol=="B_unseen_batch")&(bad.role=="test")].index[0]
        bad.loc[p,"role"]="train"
        with self.assertRaises(ValueError):
            validate_splits(trial,bad)

    def test_missing_answer_is_excluded(self):
        fix,ia=synthetic_reports()
        for frame in [fix,ia]:
            frame.loc[(frame.participant_id=="b1p0") & (frame.trial_index=="0"),
                      "selected_answer"]=None
        trial,exclusions=make_cohort(fix,ia)
        self.assertEqual(len(trial),29)
        self.assertEqual(int(exclusions.loc[exclusions.disposition.eq("missing_answer"),"n_trials"].iloc[0]),1)

    def test_matched_evaluation(self):
        fix,ia=synthetic_reports()
        table,_=make_cohort(fix,ia)
        splits=make_splits(table)
        rows=[]
        for protocol,split in splits.groupby("protocol"):
            expected=table.merge(split[["participant_id","role"]],on="participant_id")
            for name in CLASSICAL+NEURAL:
                for t in expected[expected.role.ne("train")].itertuples():
                    rows.append(dict(protocol=protocol,model=name,role=t.role,
                                     trial_id=t.trial_id,p_correct=.9 if t.is_correct else .1))
        pred=pd.DataFrame(rows)
        data,val,selected,test,people,strata,calib=evaluate(table,splits,pred)
        self.assertTrue(test.macro_f1.eq(1).all())
        intervals,paired=participant_bootstrap(data,draws=20)
        self.assertTrue(intervals.macro_f1_low.eq(1).all())
        self.assertTrue(paired.difference.eq(0).all())
        with self.assertRaises(ValueError):
            evaluate(table,splits,pred.iloc[:-1])
        self.assertTrue(np.isnan(metric([1,1],[.8,.9])["auroc"]))


if __name__=="__main__":
    unittest.main()
