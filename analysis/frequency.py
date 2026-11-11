"""
E9 - word-frequency analysis of the 500-sentence predictions (Zipf scale, wordfreq).
Usage (from the repository root):
    python analysis/frequency.py results/x1_crossmodel results/x1_crossmodel/analysis
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from wordfreq import zipf_frequency as zf
D, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
SHORT = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large", "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large"}
rows, per = [], []
for m, s in SHORT.items():
    f = pd.read_csv(f"{D}/x1_final_{m}.csv")
    f["zipf_target"] = f.target.map(lambda w: zf(str(w).lower(), "en"))
    f["zipf_top1"] = f.top1.map(lambda w: zf(str(w).replace("##", "").lower(), "en"))
    w = f[~f.hit1]
    rho, p = spearmanr(f.zipf_target, np.log10(f.target_rank))
    st = wilcoxon(w.zipf_top1, w.zipf_target)
    q = pd.qcut(f.zipf_target, 4, labels=["Q1 rare", "Q2", "Q3", "Q4 common"])
    byq = f.groupby(q, observed=True).agg(hit10=("hit10", "mean"), median_rank=("target_rank", "median"))
    rows.append({"model": s, "spearman(zipf_target, log rank)": rho, "p": p,
                 "wrong: zipf top1 (median)": w.zipf_top1.median(), "wrong: zipf target (median)": w.zipf_target.median(),
                 "wrong: top1 more frequent (%)": (w.zipf_top1 > w.zipf_target).mean() * 100, "wilcoxon p": st.pvalue,
                 **{f"hit@10 {k}": v for k, v in byq.hit10.items()}})
    f.assign(model=s).to_csv(f"{OUT}/e9_freq_{m}.csv", index=False)
t = pd.DataFrame(rows); t.to_csv(f"{OUT}/t9_frequency.csv", index=False)
pd.set_option("display.width", 250); print(t.round(3).T.to_string())
