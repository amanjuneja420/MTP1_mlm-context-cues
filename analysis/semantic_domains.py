"""
Table t6: are wrong top-1 predictions still from the sentence's domain?
Each predicted word and each domain label is embedded with all-MiniLM-L6-v2;
a word is assigned to the most similar domain label. Baseline: shuffled domains.
Usage (from the repository root):
    python analysis/semantic_domains.py results/x1_crossmodel results/x1_crossmodel/analysis
"""
import sys, numpy as np, pandas as pd, semantic as S
D, OUT = sys.argv[1], sys.argv[2]
SHORT = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large", "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large"}
import os
rows = []
rng = np.random.default_rng(0)
for m, s in SHORT.items():
    p = f"{D}/x1_final_{m}.csv"
    if not os.path.exists(p): continue
    a = S.analyse(p)
    f = pd.read_csv(p)
    keys = list(S.DOM); Dm = S.emb([S.DOM[k] for k in keys])
    top1 = [str(w).replace('##', '').strip().lower() for w in f.top1]
    pred_dom = np.array([keys[i] for i in (S.emb(top1) @ Dm.T).argmax(1)])
    wrong = ~f.hit1.values
    # permutation baseline: same predicted words, compared against shuffled sentence domains
    base = np.mean([(pred_dom[wrong] == rng.permutation(f.domain.values[wrong])).mean() for _ in range(200)])
    rows.append({"model": s, "n_wrong": int(wrong.sum()),
                 "wrong_top1_in_sentence_domain": (pred_dom[wrong] == f.domain.values[wrong]).mean(),
                 "shuffled_baseline": base,
                 "true_target_in_domain": a.target_in_domain.mean(),
                 "cos(top1,target)_wrong": a[~a.hit1].rel_top1.mean()})
t = pd.DataFrame(rows); t.to_csv(f"{OUT}/t6_semantic.csv", index=False); print(t.round(3).to_string(index=False))
