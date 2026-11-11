"""
C4 — ATTENTION FROM THE MASK POSITION, per layer and head

Which context tokens is the mask actually reading, and does that change between
'patient' and 'person'? Analysis design follows Clark et al. 2019 (W19-4828).

Note the interpretive caveat: raw attention weight is not the same as causal
influence (Jain & Wallace 2019). C5 (occlusion) is the causal measure; C4 shows
the routing. Where they agree, the story is strong; where they disagree, that
disagreement is itself a finding.

Outputs:
  c4_attention_<case>.csv    mean-over-heads attention, layer x token
  c4_attention_heads.csv     full layer x head x token, long format
  c4_attention.png           heatmaps
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


CASES = {
    "medical": "the patient is taking [MASK] since the last 2 years .",
    "generic": "the person is taking [MASK] since the last 2 years .",
}


def main():
    C.load()

    long_rows = []
    heat = {}

    for name, sent in CASES.items():
        C.banner(f"C4 — attention from [MASK], [{name}]")
        toks = C.tokens_of(sent)
        att = C.attention_at_mask(sent)          # (12 layers, 12 heads, seq)
        mean_over_heads = att.mean(axis=1)       # (12, seq)
        heat[name] = (toks, mean_over_heads)

        df = pd.DataFrame(mean_over_heads, columns=toks)
        df.index.name = "layer"
        df.index = range(1, 13)
        df.to_csv(C.out_path(f"c4_attention_{name}.csv"))

        # Content words only - [CLS]/[SEP] soak up most attention as "no-ops"
        content = [i for i, t in enumerate(toks) if t not in ("[CLS]", "[SEP]", ".")]
        print("mean attention from [MASK] (averaged over 12 heads), content tokens:")
        hdr = "layer | " + " | ".join(f"{toks[i]:>10s}" for i in content)
        print(hdr)
        print("-" * len(hdr))
        for L in range(12):
            vals = " | ".join(f"{mean_over_heads[L, i]:10.3f}" for i in content)
            print(f"{L+1:5d} | {vals}")

        # Where does the mask look most, per layer?
        print("\n  argmax context token per layer (excluding [CLS]/[SEP]):")
        for L in range(12):
            j = max(content, key=lambda i: mean_over_heads[L, i])
            print(f"    layer {L+1:2d}: '{toks[j]}' ({mean_over_heads[L, j]:.3f})")

        for L in range(12):
            for h in range(12):
                for i, t in enumerate(toks):
                    long_rows.append({"case": name, "layer": L + 1, "head": h + 1,
                                      "token": t, "position": i,
                                      "attention": float(att[L, h, i])})

    pd.DataFrame(long_rows).to_csv(C.out_path("c4_attention_heads.csv"), index=False)

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, len(CASES), figsize=(7.5 * len(CASES), 4.6))
    for ax, name in zip(np.atleast_1d(axes), CASES):
        toks, m = heat[name]
        im = ax.imshow(m, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(12))
        ax.set_yticklabels(range(1, 13), fontsize=8)
        ax.set_ylabel("layer")
        ax.set_title(f"{name} — attention from [MASK]")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("C4 — what the mask reads, by layer (mean over 12 heads)", y=1.02)
    fig.tight_layout()
    fig.savefig(C.out_path("c4_attention.png"), dpi=150, bbox_inches="tight")

    print("\nWrote c4_*.csv and c4_attention.png")


if __name__ == "__main__":
    main()
