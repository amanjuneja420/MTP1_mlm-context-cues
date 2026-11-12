"""
Significance of the time-phrase effect in the subject x time-phrase grid (x1 part C).

For each model and each of the 13 subjects that name a person or an animal (the grid's
`none` and `machine` rows are excluded), the medical mass is averaged over the duration
phrases and over the momentary phrases, and compared with the sentence without a time
phrase. A paired Wilcoxon signed-rank test is run across the 13 subjects.

Usage (from the repository root):
    python analysis/grid_stats.py results/x1_crossmodel results/x1_crossmodel/analysis
"""
import os, sys
import pandas as pd
from scipy.stats import wilcoxon

DUR = ["for two days", "for years", "since the last 2 years", "for the last two years"]
MOM = ["right now", "at the moment", "since the last 2 minutes"]
EXCLUDE = ["none", "machine"]
MODELS = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large",
          "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large"}

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
for m, name in MODELS.items():
    g = pd.read_csv(f"{src}/x1_grid_{m}.csv")
    g = g[~g.subject.isin(EXCLUDE)]
    none = g[g["tail"] == "(none)"].set_index("subject").medical_mass
    dur = g[g["tail"].isin(DUR)].groupby("subject").medical_mass.mean()
    mom = g[g["tail"].isin(MOM)].groupby("subject").medical_mass.mean()
    none, mom = none.loc[dur.index], mom.loc[dur.index]
    rows.append({"model": name, "n_subjects": len(dur),
                 "duration > none": int((dur > none).sum()), "p (duration vs none)": wilcoxon(dur, none).pvalue,
                 "duration > momentary": int((dur > mom).sum()), "p (duration vs momentary)": wilcoxon(dur, mom).pvalue,
                 "mean mass: none": none.mean(), "momentary": mom.mean(), "duration": dur.mean()})
t = pd.DataFrame(rows)
t.to_csv(f"{out}/t5b_grid_wilcoxon.csv", index=False)
print(t.round(4).to_string(index=False))
