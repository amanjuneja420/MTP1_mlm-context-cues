"""
C6 — LAYER-WISE x CONTEXT: the two tasks joined

The two project tasks were (1) layer-wise propagation of the final prediction
and (2) the impact of neighbouring words. C3 and C5 answer them separately.
C6 crosses them: it re-runs the logit lens under occlusion, to see at which
layer the effect of removing a neighbour becomes readable.

Limitation: occlusion changes the input, so every layer of the occluded run
differs from the intact run. C6 therefore shows when a divergence becomes
readable, not where the influence originates; C8 (activation patching) is the
experiment that answers the second question.

Outputs:
  c6_layerwise_occlusion.csv    winner trajectory under each occlusion
  c6_divergence_by_layer.csv    per-layer KL between intact and occluded runs
  c6_layer_of_influence.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common as C


def layerwise_logits_from_ids(ids: torch.Tensor, mask_index: int) -> torch.Tensor:
    """Logit lens for an already-tokenized (possibly perturbed) input."""
    _, model = C.load()
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
        stack = torch.stack([h[0, mask_index] for h in out.hidden_states])
        return model.cls(stack)


def main():
    C.load()
    tok, _ = C.load()

    CASES = {
        "medical": "the patient is taking [MASK] since the last 2 years .",
        "generic": "the person is taking [MASK] since the last 2 years .",
    }

    traj_rows, div_rows = [], []

    for case, sent in CASES.items():
        C.banner(f"C6 — [{case}] {sent}")

        inputs = tok(sent, return_tensors="pt")
        ids = inputs["input_ids"][0]
        mi = int(torch.where(ids == tok.mask_token_id)[0][0])
        toks = tok.convert_ids_to_tokens(ids)

        base_lg = layerwise_logits_from_ids(ids, mi)          # (13, vocab)
        base_winner_id = int(base_lg[12].argmax())
        base_winner = tok.convert_ids_to_tokens(base_winner_id)
        print(f"intact winner: '{base_winner}'\n")

        print("top-1 per layer under each occlusion:")
        header = "occluded  | " + " | ".join(f"L{L:<2d}" for L in range(13))
        print(header)
        print("-" * len(header))

        intact_top = [tok.convert_ids_to_tokens(int(base_lg[L].argmax()))
                      for L in range(13)]
        print("(intact)  | " + " | ".join(f"{w[:3]:<3s}" for w in intact_top))

        for pos, t in enumerate(toks):
            if pos == mi or t in ("[CLS]", "[SEP]"):
                continue
            pert = ids.clone()
            pert[pos] = tok.mask_token_id
            lg = layerwise_logits_from_ids(pert, mi)

            tops = [tok.convert_ids_to_tokens(int(lg[L].argmax())) for L in range(13)]
            print(f"{t:<9s} | " + " | ".join(f"{w[:3]:<3s}" for w in tops))

            for L in range(13):
                p = torch.softmax(base_lg[L], dim=-1)
                q = torch.softmax(lg[L], dim=-1)
                kl = float(torch.sum(p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))))
                order = torch.argsort(lg[L], descending=True)
                rank_of = torch.empty_like(order)
                rank_of[order] = torch.arange(len(order))
                div_rows.append({
                    "case": case, "occluded": t, "position": pos, "layer": L,
                    "kl": kl,
                    "winner_top1": tops[L],
                    "base_winner_rank": int(rank_of[base_winner_id]) + 1,
                    "base_winner_prob": float(q[base_winner_id]),
                })
                traj_rows.append({
                    "case": case, "occluded": t, "layer": L,
                    "top1": tops[L],
                })

        # Where does each neighbour start to matter?
        d = pd.DataFrame([r for r in div_rows if r["case"] == case])
        print("\n  layer at which occluding each token first pushes KL above 0.5:")
        for t in d.occluded.unique():
            s = d[d.occluded == t].sort_values("layer")
            hit = s[s.kl > 0.5]
            first = int(hit.layer.iloc[0]) if len(hit) else None
            peak = int(s.loc[s.kl.idxmax(), "layer"])
            print(f"    {t:<9s} first>0.5: {str(first):>4s}   peak-KL layer: {peak:2d}"
                  f"   maxKL {s.kl.max():.3f}")

    div = pd.DataFrame(div_rows)
    div.to_csv(C.out_path("c6_divergence_by_layer.csv"), index=False)
    pd.DataFrame(traj_rows).to_csv(C.out_path("c6_layerwise_occlusion.csv"), index=False)

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, case in zip(axes, CASES):
        d = div[div.case == case]
        # only the tokens that actually matter, to keep the plot readable
        keep = (d.groupby("occluded").kl.max()
                  .sort_values(ascending=False).head(5).index)
        for t in keep:
            s = d[d.occluded == t].sort_values("layer")
            ax.plot(s.layer, s.kl, marker="o", ms=3.5, lw=1.6, label=t)
        ax.set_xlabel("layer")
        ax.set_ylabel("KL(intact || occluded) at the mask")
        ax.set_title(f"{case} — where each neighbour exerts its influence")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("C6 — layer-wise influence of neighbouring words", y=1.02)
    fig.tight_layout()
    fig.savefig(C.out_path("c6_layer_of_influence.png"), dpi=150, bbox_inches="tight")

    print("\nWrote c6_*.csv and c6_layer_of_influence.png")


if __name__ == "__main__":
    main()
