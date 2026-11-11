"""
Redraws the data figures used in the MTP-1 report from the saved result CSVs.

Run the analysis scripts first (see README), then, from the repository root:
    python analysis/make_figures.py
Figures are written to figures/.

Three figures are produced directly by the analysis scripts and only copied here:
    x1_layerwise.png  <- analyze_x1.py  (f1_layerwise.png)
    x1_grid.png       <- analyze_x1.py  (f2_grid.png)
    e3_aoyama.png     <- analyze_x2.py  (f_e3_aoyama.png)
"""
import os, shutil, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CS = f"{ROOT}/results/case_study"
X1, X1A = f"{ROOT}/results/x1_crossmodel", f"{ROOT}/results/x1_crossmodel/analysis"
X2, X2A = f"{ROOT}/results/x2_extended", f"{ROOT}/results/x2_extended/analysis"
FIG = f"{ROOT}/figures"
warnings.filterwarnings("ignore", message="Tight layout not applied")
os.makedirs(FIG, exist_ok=True)

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

BASE = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large",
        "roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large"}
COL = dict(zip(BASE.values(), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}", bbox_inches="tight"); plt.close(fig)
    print("wrote figures/" + name)


# --- copies of figures drawn by the analysis scripts
for src, dst in [(f"{X1A}/f1_layerwise.png", "x1_layerwise.png"), (f"{X1A}/f2_grid.png", "x1_grid.png"),
                 (f"{X2A}/f_e3_aoyama.png", "e3_aoyama.png")]:
    shutil.copy(src, f"{FIG}/{dst}"); print("copied figures/" + dst)

# --- C8: activation patching on the case-study sentence (patient -> athlete), BERT-base
d = pd.read_csv(f"{CS}/c8_patching_patient_to_athlete.csv")
d = d[d.layer.between(1, 11)]
subj = d[d.token_clean == "patient"].set_index("layer").restoration
mask = d[d.token_clean == "[MASK]"].set_index("layer").restoration
fig, ax = plt.subplots(figsize=(6.2, 2.7))
ax.plot(subj.index, subj.values, "-o", ms=4, color="#1f77b4", label="patch at subject position")
ax.plot(mask.index, mask.values, "-s", ms=4, color="#d62728", label="patch at mask position")
ax.set_xticks(range(1, 12)); ax.set_ylim(-0.02, 1.0); ax.grid(alpha=.3)
ax.set_xlabel("layer patched"); ax.set_ylabel("restoration of $p$(medication)")
ax.legend(frameon=False)
save(fig, "c8_patching.png")

# --- X1 part B: delete vs mask, by distance and by word class
cs = []
for m, s in BASE.items():
    c = pd.read_csv(f"{X1A}/ctx_support_{m}.csv"); c["model"] = s; cs.append(c)
cs = pd.concat(cs); cs["dist"] = cs.distance.clip(upper=6)
fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3), sharey=True)
for s in BASE.values():
    g = cs[cs.model == s].groupby("dist")[["delete", "mask"]].mean()
    a.plot(g.index, g["delete"], "-o", color=COL[s], label=s)
    a.plot(g.index, g["mask"], "--x", color=COL[s])
a.set_xlabel("distance from the mask (words; 6 = 6 or more)")
a.set_ylabel("support = drop in log p(target) (nats)")
a.set_title("solid: delete the word   dashed: mask the word", fontsize=9)
a.legend(frameon=False, fontsize=8); a.grid(alpha=.3)
g = cs.groupby(["model", "function_word"])[["delete", "mask"]].mean()
x = np.arange(len(BASE)); w = 0.2
bars = [(False, "delete", "#1f6fb4", "content word, delete"), (False, "mask", "#9ecae1", "content word, mask"),
        (True, "delete", "#ff7f0e", "function word, delete"), (True, "mask", "#fdc896", "function word, mask")]
for i, (fw, meth, c, lab) in enumerate(bars):
    b.bar(x + (i - 1.5) * w, [g.loc[(s, fw), meth] for s in BASE.values()], w, color=c, label=lab)
b.set_xticks(x); b.set_xticklabels(BASE.values(), fontsize=8); b.legend(frameon=False, fontsize=7); b.grid(alpha=.3, axis="y")
save(fig, "x1_context.png")

# --- X1 part A: hit@10 on single-token targets, by domain
t = pd.read_csv(f"{X1A}/t2_by_domain.csv").pivot(index="domain", columns="model", values="hit10_single")
t = t[list(BASE.values())]; t = t.loc[t.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(4.7, 3.5))
im = ax.imshow(t.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        v = t.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
ax.set_xticks(range(t.shape[1])); ax.set_xticklabels(t.columns, rotation=20)
ax.set_yticks(range(t.shape[0])); ax.set_yticklabels([d.title() for d in t.index])
ax.set_title("hit@10 on single-token targets", fontsize=9)
fig.colorbar(im, ax=ax)
save(fig, "x1_domain_hit10.png")

# --- E8: medical mass by verb frame, subject "person", no time expression vs a duration
t = pd.read_csv(f"{X2A}/t_e8_grid2_person.csv")
frames = ["is taking", "is not taking", "has been taking", "takes", "took", "is using", "is on"]
fig, axes = plt.subplots(1, 4, figsize=(9, 3.2), sharey=True)
for ax, s in zip(axes, BASE.values()):
    g = t[t.model == s].set_index("frame").reindex(frames)
    y = np.arange(len(frames))
    ax.barh(y - 0.19, g["none"], 0.38, color="#c8c8c8", label="no time expression")
    ax.barh(y + 0.19, g["duration"], 0.38, color="#d62728", label="with a duration")
    ax.set_yticks(y); ax.set_yticklabels(frames)
    ax.set_xlim(0, 1); ax.set_title(s, fontsize=10); ax.set_xlabel("medical mass"); ax.grid(alpha=.3, axis="x")
axes[0].invert_yaxis()   # shared y axis: invert once
axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(2.2, 1.12))
save(fig, "e8_frames.png")

# --- E9: hit@10 by target-word frequency quartile
fig, ax = plt.subplots(figsize=(5.2, 2.7))
labels = ["rarest 25%", "2nd", "3rd", "most common 25%"]
for i, (m, s) in enumerate(BASE.items()):
    f = pd.read_csv(f"{X1A}/e9_freq_{m}.csv")
    q = pd.qcut(f.zipf_target, 4, labels=labels)
    v = f.groupby(q, observed=True).hit10.mean()
    ax.bar(np.arange(4) + (i - 1.5) * 0.2, v.values, 0.2, color=COL[s], label=s)
ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
ax.set_xlabel("target word frequency (Zipf quartile)"); ax.set_ylabel("hit@10")
ax.legend(frameon=False, ncol=2, fontsize=8); ax.grid(alpha=.3, axis="y")
save(fig, "e9_frequency.png")

# --- E4: where the mask attends, by relative depth
t = pd.read_csv(f"{X2A}/t_e4_attention_by_layer.csv")
fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5), sharex=True)
for ax, (col, ttl) in zip(axes.ravel(), [("to_sep", "to [SEP] / </s>"), ("to_cls", "to [CLS] / <s>"),
                                         ("to_punct", "to punctuation"), ("entropy", "attention entropy (nats)")]):
    for s in BASE.values():
        g = t[t.model == s]
        ax.plot(g.rel_depth, g[col], "-o", ms=2.5, color=COL[s], label=s)
    ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
for ax in axes[1]:
    ax.set_xlabel("relative depth")
axes[0, 0].legend(frameon=False, fontsize=8)
save(fig, "e4_attention.png")

# --- E7: activation patching on all four base models
P = []
for m, s in BASE.items():
    a = pd.read_csv(f"{X2}/e7_patching_{m}.csv"); a["model"] = s; P.append(a)
P = pd.concat(P)
setups = [("subject_patient_to_machine", "subject: patient → machine"),
          ("duration_years_to_minutes", "duration: years → minutes"),
          ("verb_taking_to_eating", "verb: taking → eating")]
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
for ax, (s, ttl) in zip(axes, setups):
    for m in BASE.values():
        # years -> minutes barely changes BERT-large's output (tiny denominator), so it is not plotted
        if s.startswith("duration") and m == "BERT-large":
            continue
        g = P[(P.setup == s) & (P.model == m)]
        d = g[g.differs].groupby("rel_depth").restoration.mean()
        k = g[g.is_mask].groupby("rel_depth").restoration.mean()
        ax.plot(d.index, d.values, "-", color=COL[m], label=m)
        ax.plot(k.index, k.values, "--", color=COL[m])
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("relative depth of the patched layer"); ax.grid(alpha=.3)
axes[0].set_ylabel("restoration of medical mass")
axes[0].legend(frameon=False, fontsize=7, title="solid: changed word\ndashed: mask", title_fontsize=7)
save(fig, "e7_patching.png")
