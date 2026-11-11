"""
X1 analysis: tables t1-t6 and figures f1-f2 from the x1_crossmodel.py outputs.
Usage (from the repository root):
    python analysis/analyze_x1.py results/x1_crossmodel results/x1_crossmodel/analysis
    python analysis/semantic_domains.py results/x1_crossmodel results/x1_crossmodel/analysis   # table t6
"""
import os, sys, json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MODELS = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
SHORT = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large",
         "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large"}
have = [m for m in MODELS if os.path.exists(f"{D}/x1_final_{m}.csv")]
STOP = set("""a an the of in on at to for from by with into onto about over under beside during before after
and or but if than as is are was were be been being has have had do does did it its this that these those
her his their our my your its he she they we i you him them us who which what when where while not no""".split())
report = {}

# ------------------------------------------------------------------ 1. behaviour
rows = []
for m in have:
    f = pd.read_csv(f"{D}/x1_final_{m}.csv")
    for scope, g in (("all", f), ("single-token", f[f.single_token])):
        rows.append({"model": SHORT[m], "scope": scope, "n": len(g),
                     "hit@1": g.hit1.mean(), "hit@10": g.hit10.mean(),
                     "median_rank": g.target_rank.median(),
                     "mean_target_p": g.target_p.mean()})
beh = pd.DataFrame(rows)
beh.to_csv(f"{OUT}/t1_behaviour.csv", index=False)

dom = []
for m in have:
    f = pd.read_csv(f"{D}/x1_final_{m}.csv")
    g = f.groupby("domain").agg(n=("idx", "size"), single_token=("single_token", "mean"),
                                hit1=("hit1", "mean"), hit10=("hit10", "mean"),
                                median_rank=("target_rank", "median"))
    st = f[f.single_token].groupby("domain").agg(hit10_single=("hit10", "mean"))
    g = g.join(st); g["model"] = SHORT[m]
    dom.append(g.reset_index())
dom = pd.concat(dom)
dom.to_csv(f"{OUT}/t2_by_domain.csv", index=False)

# ------------------------------------------------------------------ 2. layer-wise (Aoyama-style)
def expected_layer(scores):
    d = np.diff(scores)                     # D(L) = S(L) - S(L-1), L = 1..N
    L = np.arange(1, len(scores))
    return float((L * d).sum() / d.sum()) if d.sum() != 0 else np.nan

lay_rows, curves = [], {}
for m in have:
    l = pd.read_csv(f"{D}/x1_layers_{m}.csv")
    f = pd.read_csv(f"{D}/x1_final_{m}.csv")[["domain", "idx", "single_token", "hit1"]]
    l = l.merge(f, on=["domain", "idx"])
    nL = l.layer.max()
    s = l[l.single_token].groupby("layer").top1_is_target.mean().values
    curves[SHORT[m]] = (np.arange(nL + 1) / nL, s)
    # emergence: first layer from which the target stays top-1 to the end (only sentences that end correct)
    em = []
    for (_, _, ), g in l[l.hit1].groupby(["domain", "idx"]):
        t = g.sort_values("layer").top1_is_target.values
        k = len(t) - 1
        while k > 0 and t[k - 1]:
            k -= 1
        em.append(k / nL)
    E = expected_layer(s)
    lay_rows.append({"model": SHORT[m], "layers": nL,
                     "expected_layer_word": E, "expected_rel_depth": E / nL,
                     "median_emergence_rel_depth": float(np.median(em)) if em else np.nan,
                     "n_correct": len(em)})
lay = pd.DataFrame(lay_rows)
lay.to_csv(f"{OUT}/t3_layerwise.csv", index=False)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
for k, (x, y) in curves.items():
    ax.plot(x, y, marker="o", ms=3, label=k)
ax.set_xlabel("relative depth (layer / number of layers)")
ax.set_ylabel("share of sentences where target is top-1")
ax.set_title("Logit lens on the 500-sentence sheet (single-token targets)")
ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/f1_layerwise.png", dpi=160)

# ------------------------------------------------------------------ 3. context attribution
ctx_rows = []
for m in have:
    p = f"{D}/x1_context_{m}.csv"
    if not os.path.exists(p):
        continue
    c = pd.read_csv(p)
    f = pd.read_csv(f"{D}/x1_final_{m}.csv")[["domain", "idx", "masked"]]
    c = c.merge(f, on=["domain", "idx"])
    mw = c.masked.str.split(" ").apply(lambda ws: next(i for i, w in enumerate(ws) if "<mask>" in w))
    c["distance"] = (c.word_index - mw).abs()
    c["function_word"] = c.word.str.lower().isin(STOP)
    c["support"] = -c.delta_logp            # > 0: removing the word hurt the target
    piv = c.pivot_table(index=["domain", "idx", "word_index", "word", "distance", "function_word"],
                        columns="method", values="support").reset_index()
    # agreement of the two methods on the single most important word per sentence
    top_d = piv.loc[piv.groupby(["domain", "idx"])["delete"].idxmax()][["domain", "idx", "word"]]
    top_m = piv.loc[piv.groupby(["domain", "idx"])["mask"].idxmax()][["domain", "idx", "word"]]
    agree = (top_d.set_index(["domain", "idx"]).word == top_m.set_index(["domain", "idx"]).word).mean()
    rho = piv[["delete", "mask"]].corr(method="spearman").iloc[0, 1]
    ctx_rows.append({
        "model": SHORT[m],
        "content_support_delete": piv[~piv.function_word]["delete"].mean(),
        "function_support_delete": piv[piv.function_word]["delete"].mean(),
        "adjacent_support_delete": piv[piv.distance == 1]["delete"].mean(),
        "far_support_delete(d>=3)": piv[piv.distance >= 3]["delete"].mean(),
        "share_top_word_adjacent": (piv.loc[piv.groupby(["domain", "idx"])["delete"].idxmax()].distance == 1).mean(),
        "top_word_agree_delete_vs_mask": agree,
        "spearman_delete_vs_mask": rho,
        "mean_|support|_delete": piv["delete"].abs().mean(),
        "mean_|support|_mask": piv["mask"].abs().mean(),
    })
    piv.to_csv(f"{OUT}/ctx_support_{m}.csv", index=False)
ctx = pd.DataFrame(ctx_rows)
ctx.to_csv(f"{OUT}/t4_context.csv", index=False)

# ------------------------------------------------------------------ 4. grid
grids = []
for m in have:
    g = pd.read_csv(f"{D}/x1_grid_{m}.csv"); g["model"] = SHORT[m]; grids.append(g)
if grids:
    G = pd.concat(grids)
    G.to_csv(f"{OUT}/t5_grid_all.csv", index=False)
    tails = list(dict.fromkeys(G["tail"]))
    subs = list(dict.fromkeys(G.subject))
    fig, axes = plt.subplots(1, len(have), figsize=(4.3 * len(have), 5.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), have):
        g = G[G.model == SHORT[m]].pivot(index="subject", columns="tail", values="medical_mass").loc[subs, tails]
        im = ax.imshow(g.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tails))); ax.set_xticklabels(tails, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs, fontsize=8)
        ax.set_title(SHORT[m], fontsize=10)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                v = g.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > .55 else "black")
    fig.suptitle("Probability mass on medical words at the mask: '<subject> is taking <mask> <tail>.'", y=1.0)
    fig.colorbar(im, ax=axes, fraction=0.015)
    fig.savefig(f"{OUT}/f2_grid.png", dpi=160, bbox_inches="tight")

pd.set_option("display.width", 200)
for name, t in [("BEHAVIOUR", beh), ("LAYER-WISE", lay), ("CONTEXT", ctx)]:
    print(f"\n=== {name}\n{t.round(3).to_string(index=False)}")
