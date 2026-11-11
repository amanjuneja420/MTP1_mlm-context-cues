"""
C3 — LAYERWISE PROPAGATION OF THE FINAL PREDICTION WORD  [task 1: layer-wise propagation of the prediction]

Method: logit lens. Apply the MLM head (model.cls) to the hidden state at the
mask position after EVERY layer, not just the last. Because bert-base-uncased
ties its output decoder to the input embedding matrix, a logit at any layer is
<transformed hidden state, word embedding> - i.e. "how aligned is this partial
representation with word v's meaning?".

Verified precondition: model.cls(hidden_states[12]) reproduces model(...).logits.

Outputs:
  c3_layerwise_topk.csv        top-5 at each of 13 layers, per sentence
  c3_winner_trajectory.csv     rank+prob of the FINAL winner at every layer
  c3_probe_trajectory.csv      rank+prob of chosen probe tokens at every layer
  c3_emergence_layer.csv       the layer at which the final winner first tops the list
  c3_trajectory_*.png          plots
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common as C


def winner_trajectory(sentence: str) -> pd.DataFrame:
    """Rank/prob of the final-layer winner, tracked back through every layer."""
    w = C.final_winner(sentence)
    df = C.track_tokens(sentence, [w])
    df["is_final_winner"] = True
    return df


def emergence_layer(traj: pd.DataFrame) -> int:
    """First layer at which the final winner is already rank 1. -1 if never before 12."""
    hits = traj[traj["rank"] == 1]["layer"]
    return int(hits.min()) if len(hits) else -1


def main():
    C.load()

    # ---------------------------------------------------------------- 3.1
    C.banner("C3.1 — layer-by-layer top-5, the patient/person sentences")

    focus = {
        "patient": C.DOMAIN_SET["medical"][0],
        "person":  C.DOMAIN_SET["generic"][0],
    }

    topk_frames = []
    for name, sent in focus.items():
        df = C.layerwise_topk(sent, k=5)
        df.insert(0, "case", name)
        topk_frames.append(df)
        print(f"\n[{name}] {sent}")
        for L in range(13):
            row = df[df.layer == L]
            words = ", ".join(f"{t}" for t in row.token)
            print(f"  layer {L:2d}: {words}")

    # ---------------------------------------------------------------- 3.2
    C.banner("C3.2 — trajectory of the FINAL prediction word through the layers")

    traj_frames = []
    for name, sent in focus.items():
        w = C.final_winner(sent)
        t = C.track_tokens(sent, [w])
        t.insert(0, "case", name)
        traj_frames.append(t)
        print(f"\n[{name}] final winner = '{w}'")
        print("  layer :  " + " ".join(f"{L:>7d}" for L in t.layer))
        print("  rank  :  " + " ".join(f"{r:>7d}" for r in t["rank"]))
        print("  prob  :  " + " ".join(f"{p:>7.4f}" for p in t["prob"]))
        print(f"  --> first reaches rank 1 at layer {emergence_layer(t)}")

    # ---------------------------------------------------------------- 3.3
    C.banner("C3.3 — competing probe tokens, medical vs light-verb readings")

    probes = ["medication", "drugs", "care", "part", "classes", "over", "up", "them"]
    probe_frames = []
    for name, sent in focus.items():
        p = C.track_tokens(sent, probes)
        p.insert(0, "case", name)
        probe_frames.append(p)

    probe_df = pd.concat(probe_frames, ignore_index=True)
    for name in focus:
        print(f"\n[{name}] rank of each probe token by layer (lower = better)")
        piv = probe_df[probe_df.case == name].pivot(
            index="token", columns="layer", values="rank")
        piv = piv.reindex(probes)
        print(piv.to_string())

    # ---------------------------------------------------------------- 3.4
    C.banner("C3.4 — emergence layer across all domains")

    rows = []
    for dom, (sent, _) in C.DOMAIN_SET.items():
        w = C.final_winner(sent)
        t = C.track_tokens(sent, [w])
        el = emergence_layer(t)
        rows.append({
            "domain": dom,
            "sentence": sent,
            "final_winner": w,
            "emergence_layer": el,
            "prob_at_L6": float(t[t.layer == 6]["prob"].iloc[0]),
            "prob_at_L9": float(t[t.layer == 9]["prob"].iloc[0]),
            "prob_final": float(t[t.layer == 12]["prob"].iloc[0]),
            "rank_at_L6": int(t[t.layer == 6]["rank"].iloc[0]),
            "rank_at_L9": int(t[t.layer == 9]["rank"].iloc[0]),
        })
        traj = t.copy()
        traj.insert(0, "case", dom)
        traj_frames.append(traj)

    emg = pd.DataFrame(rows).sort_values("emergence_layer")
    print(C.df_to_md(emg.drop(columns=["sentence"])))

    # ---------------------------------------------------------------- save
    pd.concat(topk_frames, ignore_index=True).to_csv(
        C.out_path("c3_layerwise_topk.csv"), index=False)
    pd.concat(traj_frames, ignore_index=True).to_csv(
        C.out_path("c3_winner_trajectory.csv"), index=False)
    probe_df.to_csv(C.out_path("c3_probe_trajectory.csv"), index=False)
    emg.to_csv(C.out_path("c3_emergence_layer.csv"), index=False)

    # ---------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, name in zip(axes, focus):
        sub = probe_df[probe_df.case == name]
        for t in probes:
            s = sub[sub.token == t]
            if len(s):
                ax.plot(s.layer, s["rank"], marker="o", ms=3.5, label=t, lw=1.6)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_title(f'"the {name} is taking [MASK] ..."')
        ax.set_xlabel("layer")
        ax.set_ylabel("rank of token (log, inverted)")
        ax.grid(alpha=0.3)
        ax.axhline(1, color="k", lw=0.8, ls="--")
    axes[1].legend(fontsize=8, ncol=1, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("C3 — layer-wise rank of competing candidates at the mask", y=1.02)
    fig.tight_layout()
    fig.savefig(C.out_path("c3_trajectory_ranks.png"), dpi=150, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(8, 4.6))
    allt = pd.concat(traj_frames, ignore_index=True)
    for case in allt.case.unique():
        s = allt[allt.case == case]
        ax.plot(s.layer, s["prob"], marker="o", ms=3.5, lw=1.6,
                label=f"{case}: {s.token.iloc[0]}")
    ax.set_xlabel("layer")
    ax.set_ylabel("probability of the final winner")
    ax.set_title("C3 — when does the final answer actually form?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    fig2.savefig(C.out_path("c3_winner_prob.png"), dpi=150, bbox_inches="tight")

    print("\nWrote c3_*.csv and c3_*.png")


if __name__ == "__main__":
    main()
