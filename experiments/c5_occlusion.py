"""
C5 — OCCLUSION / LEAVE-ONE-OUT  [task 2: effect of neighbouring words, part 2]

C2 substitutes a neighbour; C5 DELETES it (replaces with a second [MASK]) and
measures how far the prediction distribution at the original mask moves.
This gives a per-token causal importance score rather than a correlation.

Methodologically this follows Perturbed Masking (Wu et al., ACL 2020,
arXiv:2004.14786), but scored in the OUTPUT distribution (KL divergence and
probe-token rank shift) rather than in representation space for tree induction.

Outputs:
  c5_occlusion_<case>.csv    per-token KL and probe deltas
  c5_occlusion.png           heatmap of context-word importance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


CASES = {
    "medical": ("the patient is taking [MASK] since the last 2 years .",
                ["medication", "drugs", "care"]),
    "generic": ("the person is taking [MASK] since the last 2 years .",
                ["medication", "drugs", "care"]),
    "music":   ("the musician is playing [MASK] since the last 2 years .",
                ["guitar", "music", "piano"]),
    "sport":   ("the athlete is playing [MASK] since the last 2 years .",
                ["football", "tennis", "basketball"]),
}


def main():
    C.load()

    all_frames = []
    for name, (sent, probes) in CASES.items():
        C.banner(f"C5 — occlusion, [{name}]  {sent}")
        base_winner = C.final_winner(sent)
        print(f"unperturbed winner: '{base_winner}'\n")

        df = C.occlusion(sent, probes)
        df.insert(0, "case", name)
        all_frames.append(df)

        cols = ["occluded", "new_winner", "winner_changed", "kl_from_base"]
        cols += [f"{probes[0]}_rank", f"{probes[0]}_prob_delta"]
        print(C.df_to_md(df[cols]))

        top = df.iloc[0]
        print(f"\n  most influential neighbour: '{top.occluded}' "
              f"(KL {top.kl_from_base:.3f}, winner -> '{top.new_winner}')")

        df.to_csv(C.out_path(f"c5_occlusion_{name}.csv"), index=False)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(C.out_path("c5_occlusion_all.csv"), index=False)

    # -------------------------------------------------------------- heatmap
    fig, axes = plt.subplots(len(CASES), 1, figsize=(9, 2.3 * len(CASES)))
    for ax, (name, _) in zip(axes, CASES.items()):
        d = combined[combined.case == name].sort_values("position")
        ax.bar(range(len(d)), d.kl_from_base,
               color=["#c0392b" if w else "#4a6fd4" for w in d.winner_changed])
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d.occluded, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("KL")
        ax.set_title(f"{name} — red = occluding this word flips the top prediction",
                     fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("C5 — causal importance of each neighbouring word", y=1.0)
    fig.tight_layout()
    fig.savefig(C.out_path("c5_occlusion.png"), dpi=150, bbox_inches="tight")

    print("\nWrote c5_*.csv and c5_occlusion.png")


if __name__ == "__main__":
    main()
