"""
X2 analysis: tables (t_e*.csv) and figures (f_e*.png) from x2_extended.py outputs.
Usage (from the repository root):
    python analysis/analyze_x2.py results/x2_extended results/x1_crossmodel results/x2_extended/analysis
"""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X2, X1, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
SHORT = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large",
         "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large",
         "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "BiomedBERT",
         "legal-bert-base-uncased": "Legal-BERT", "Bio_ClinicalBERT": "Bio-ClinicalBERT",
         "gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium"}
BASE = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
DOM = ["BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "legal-bert-base-uncased", "Bio_ClinicalBERT"]
DUR = ["for two days", "for years", "since the last 2 years", "for the last two years"]
MOM = ["right now", "at the moment", "since the last 2 minutes"]
pd.set_option("display.width", 250)
summary = {}


def have(p):
    return os.path.exists(p)


def show(name, df):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"\n=== {name}\n{df.round(3).to_string(index=False)}")


# ------------------------------------------------------------------ E1 domain models
rows = []
for m in ["bert-base-uncased"] + DOM:
    p = f"{X2}/x1_final_{m}.csv" if m in DOM else f"{X1}/x1_final_{m}.csv"
    if not have(p):
        continue
    f = pd.read_csv(p)
    s = f[f.single_token]
    for d, g in s.groupby("domain"):
        rows.append({"model": SHORT[m], "domain": d, "n_single": len(g), "hit@10": g.hit10.mean(), "hit@1": g.hit1.mean()})
    rows.append({"model": SHORT[m], "domain": "ALL", "n_single": len(s), "hit@10": s.hit10.mean(), "hit@1": s.hit1.mean()})
if rows:
    t = pd.DataFrame(rows).pivot(index="domain", columns="model", values="hit@10")
    t = t.reset_index()
    show("t_e1_domain_hit10", t)

grid = []
for m in BASE + DOM:
    p = f"{X2}/x1_grid_{m}.csv" if m in DOM else f"{X1}/x1_grid_{m}.csv"
    if have(p):
        g = pd.read_csv(p); g["model"] = SHORT[m]; grid.append(g)
if grid:
    G = pd.concat(grid)
    G["cond"] = G["tail"].map(lambda x: "duration" if x in DUR else ("momentary" if x in MOM else ("none" if x == "(none)" else "other")))
    t = (G[G.cond != "other"].groupby(["model", "subject", "cond"]).medical_mass.mean().unstack("cond")
         [["none", "momentary", "duration"]].reset_index())
    t = t[t.subject.isin(["patient", "person", "man", "someone", "dog", "machine"])]
    show("t_e1_grid_by_condition", t)

# ------------------------------------------------------------------ E2 autoregressive
rows = []
for m in ["gpt2", "gpt2-medium"] + BASE:
    p = f"{X2}/e2_sheet_{m}.csv"
    if not have(p):
        continue
    e = pd.read_csv(p)
    r = {"model": SHORT[m], "pool_size": e.pool_size.mean(), "pool_acc@1": (e.pool_rank == 1).mean(),
         "pool_MRR": (1 / e.pool_rank).mean()}
    if "left_only_rank" in e:
        r["left_only_hit@1"] = (e.left_only_rank == 1).mean()
        r["left_only_hit@10"] = (e.left_only_rank <= 10).mean()
    rows.append(r)
if rows:
    show("t_e2_ar_vs_mlm", pd.DataFrame(rows))

eg = []
for m in ["gpt2", "gpt2-medium"] + BASE:
    p = f"{X2}/e2_grid_{m}.csv"
    if have(p):
        g = pd.read_csv(p); g["model"] = SHORT[m]; eg.append(g)
if eg:
    E = pd.concat(eg)
    E["cond"] = E["tail"].map(lambda x: "duration" if x in DUR else ("momentary" if x in MOM else ("none" if x == "(none)" else "other")))
    t = (E[E.cond != "other"].groupby(["model", "subject", "cond"]).medical_share.mean().unstack("cond")
         [["none", "momentary", "duration"]].reset_index())
    show("t_e2_grid_share", t[t.subject.isin(["patient", "person", "someone", "machine"])])

# ------------------------------------------------------------------ E3 aoyama
def exp_layer(scores):
    d = np.diff(scores); L = np.arange(1, len(scores))
    return float((L * d).sum() / d.sum()) if d.sum() != 0 else np.nan

rows, upos_rows, curves = [], [], {}
for m in BASE:
    p = f"{X2}/e3_aoyama_{m}.csv"
    if not have(p):
        continue
    a = pd.read_csv(p); nL = a.layer.max()
    w = a.groupby("layer").word_match.mean().values; po = a.groupby("layer").pos_match.mean().values
    curves[SHORT[m]] = (np.arange(nL + 1) / nL, w, po)
    rows.append({"model": SHORT[m], "layers": nL, "positions": a[a.layer == 0].shape[0],
                 "E[layer] POS": exp_layer(po), "E[layer] word": exp_layer(w),
                 "E[rel depth] POS": exp_layer(po) / nL, "E[rel depth] word": exp_layer(w) / nL,
                 "final POS acc": po[-1], "final word acc": w[-1]})
    for u, g in a.groupby("orig_upos"):
        n = (g.layer == 0).sum()
        if n >= 15:
            upos_rows.append({"model": SHORT[m], "upos": u, "n": n,
                              "E[rel depth] word": exp_layer(g.groupby("layer").word_match.mean().values) / nL,
                              "final word acc": g[g.layer == nL].word_match.mean()})
if rows:
    show("t_e3_aoyama", pd.DataFrame(rows))
    show("t_e3_aoyama_by_upos", pd.DataFrame(upos_rows).pivot(index="upos", columns="model", values="E[rel depth] word").reset_index())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (x, w, po) in curves.items():
        ax[0].plot(x, w, marker="o", ms=2.5, label=k); ax[1].plot(x, po, marker="o", ms=2.5, label=k)
    ax[0].set_title("word match (top-1 = original word)"); ax[1].set_title("POS match (top-1 has original POS)")
    for a_ in ax:
        a_.set_xlabel("relative depth"); a_.grid(alpha=.3)
    ax[0].set_ylabel("share of masked positions"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/f_e3_aoyama.png", dpi=160)

# ------------------------------------------------------------------ E4 attention
att = []
for m in BASE:
    p = f"{X2}/e4_attention_{m}.csv"
    if have(p):
        a = pd.read_csv(p); a["model"] = SHORT[m]; att.append(a)
if att:
    A = pd.concat(att)
    cols = ["to_cls", "to_sep", "to_self", "to_prev", "to_next", "to_punct", "entropy"]
    agg = A.groupby(["model", "layer"])[cols + ["rel_depth"]].mean().reset_index()
    agg.to_csv(f"{OUT}/t_e4_attention_by_layer.csv", index=False)
    summ = A.groupby("model")[cols].mean().reset_index()
    peak = agg.loc[agg.groupby("model").to_sep.idxmax()][["model", "rel_depth", "to_sep"]].rename(columns={"rel_depth": "SEP peak depth", "to_sep": "SEP peak"})
    show("t_e4_attention_summary", summ.merge(peak, on="model"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharex=True)
    for ax, (c, ttl) in zip(axes, [("to_sep", "to [SEP] / </s>"), ("to_cls", "to [CLS] / <s>"),
                                    ("to_prev", "to previous token"), ("entropy", "attention entropy (nats)")]):
        for mname, g in agg.groupby("model"):
            ax.plot(g.rel_depth, g[c], marker="o", ms=2.5, label=mname)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("relative depth"); ax.grid(alpha=.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/f_e4_attention.png", dpi=160)

# ------------------------------------------------------------------ E5 conformal
def conformal(npz, alphas=(0.05, 0.10, 0.25), reps=200, seed=0):
    z = np.load(npz); P = z["sorted_probs"]; tp = z["target_p"]; dom = z["domain"]
    rng = np.random.default_rng(seed); n = len(tp); out = []
    for a in alphas:
        cov, size, dsize = [], [], {d: [] for d in np.unique(dom)}
        for _ in range(reps):
            perm = rng.permutation(n); cal, te = perm[: n // 2], perm[n // 2:]
            s = 1 - tp[cal]; k = int(np.ceil((len(cal) + 1) * (1 - a))) - 1
            q = np.sort(s)[min(k, len(s) - 1)]
            thr = 1 - q
            cov.append((tp[te] >= thr).mean())
            sz = (P[te] >= thr).sum(1)
            size.append(np.mean(sz))
            for d in dsize:
                msk = dom[te] == d
                if msk.any():
                    dsize[d].append(np.median(sz[msk]))
        out.append({"alpha": a, "coverage": np.mean(cov), "mean_set_size": np.mean(size),
                    "median_set_size_by_domain": {d: float(np.mean(v)) for d, v in dsize.items()}})
    return out

rows, drows = [], []
for m in BASE + DOM:
    p = f"{X2}/e5_dist_{m}.npz"
    if not have(p):
        continue
    for r in conformal(p):
        rows.append({"model": SHORT[m], "1-alpha": 1 - r["alpha"], "empirical coverage": r["coverage"],
                     "mean set size": r["mean_set_size"]})
        if abs(r["alpha"] - 0.10) < 1e-9:
            for d, v in r["median_set_size_by_domain"].items():
                drows.append({"model": SHORT[m], "domain": d, "median set size @90%": v})
if rows:
    show("t_e5_conformal", pd.DataFrame(rows))
    show("t_e5_conformal_by_domain", pd.DataFrame(drows).pivot(index="domain", columns="model", values="median set size @90%").reset_index())

# ------------------------------------------------------------------ E6 multi-token
rows = []
for m in BASE:
    p = f"{X2}/e6_multitoken_{m}.csv"
    if not have(p):
        continue
    e = pd.read_csv(p)
    rows.append({"model": SHORT[m], "n multi-token targets": len(e), "mean pieces": e.k.mean(),
                 "exact: parallel": e.exact_parallel.mean(), "exact: left-to-right": e.exact_l2r.mean(),
                 "exact: confidence-first": e.exact_conf.mean(), "first piece correct": e.first_piece_parallel.mean()})
if rows:
    show("t_e6_multitoken", pd.DataFrame(rows))

# ------------------------------------------------------------------ E7 patching
pp = []
for m in BASE:
    p = f"{X2}/e7_patching_{m}.csv"
    if have(p):
        a = pd.read_csv(p); a["model"] = SHORT[m]; pp.append(a)
if pp:
    P = pd.concat(pp)
    rows = []
    for (m, s), g in P.groupby(["model", "setup"]):
        dpos = g[g.differs]; mpos = g[g.is_mask]
        dcurve = dpos.groupby("rel_depth").restoration.mean(); mcurve = mpos.groupby("rel_depth").restoration.mean()
        cross = mcurve[mcurve >= 0.5].index.min() if (mcurve >= 0.5).any() else np.nan
        last_d = dcurve[dcurve >= 0.5].index.max() if (dcurve >= 0.5).any() else np.nan
        rows.append({"model": m, "setup": s, "mass clean": g.mass_clean.iloc[0], "mass corrupt": g.mass_corrupt.iloc[0],
                     "changed word still restores >=0.5 until depth": last_d,
                     "mask restores >=0.5 from depth": cross})
    show("t_e7_patching", pd.DataFrame(rows))
    setups = list(dict.fromkeys(P.setup))
    fig, axes = plt.subplots(1, len(setups), figsize=(5 * len(setups), 3.6), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), setups):
        for m, g in P[P.setup == s].groupby("model"):
            d = g[g.differs].groupby("rel_depth").restoration.mean()
            k = g[g.is_mask].groupby("rel_depth").restoration.mean()
            l, = ax.plot(d.index, d.values, "-", label=f"{m}: changed word")
            ax.plot(k.index, k.values, "--", color=l.get_color(), label=f"{m}: mask")
        ax.set_title(s.replace("_", " "), fontsize=9); ax.set_xlabel("relative depth of patched layer"); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("restoration of medical mass")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=6, ncol=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/f_e7_patching.png", dpi=160)

# ------------------------------------------------------------------ E8 grid2
g2 = []
for m in BASE + DOM:
    p = f"{X2}/e8_grid2_{m}.csv"
    if have(p):
        a = pd.read_csv(p); a["model"] = SHORT[m]; g2.append(a)
if g2:
    Q = pd.concat(g2)
    Q["cond"] = Q["tail"].map(lambda x: "duration" if x in DUR else ("momentary" if x in MOM else ("none" if x == "(none)" else "other")))
    t = (Q[(Q.subject == "person") & (Q.cond != "other")].groupby(["frame", "model", "cond"]).medical_mass.mean()
         .unstack("cond")[["none", "momentary", "duration"]].reset_index())
    show("t_e8_grid2_person", t)
    neg = (Q[Q.frame.isin(["is taking", "is not taking"])].groupby(["model", "frame"]).medical_mass.mean().unstack("frame").reset_index())
    show("t_e8_negation", neg)
print("\nWrote tables/figures to", OUT)
