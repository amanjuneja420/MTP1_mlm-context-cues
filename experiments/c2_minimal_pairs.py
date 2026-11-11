"""
C2 — MINIMAL PAIRS: quantifying the impact of neighbouring words  [task 2: effect of neighbouring words, part 1]

Changes exactly one element of the frame at a time and measures how far the
prediction moves. Complements C5 (occlusion), which deletes rather than substitutes.

Outputs:
  c2_subject_effect.csv    probe-token rank/prob for every subject
  c2_verb_effect.csv       probe-token rank/prob for every verb
  c2_mask_position.csv     top-10 when a different slot is masked
  c2_context_ablation.csv  removing/altering the duration phrase
  c2_subject_effect.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import common as C


PROBES = ["medication", "drugs", "care", "part", "classes", "lessons", "over"]


def probe_row(sentence: str, label_key: str, label_val: str) -> dict:
    tok, _ = C.load()
    out, mi = C.forward(sentence)
    logits = out.logits[0, mi]
    probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(logits, descending=True)
    rank_of = torch.empty_like(order)
    rank_of[order] = torch.arange(len(order))

    row = {label_key: label_val,
           "winner": tok.convert_ids_to_tokens(int(logits.argmax())),
           "winner_prob": float(probs.max())}
    for t in PROBES:
        i = tok.convert_tokens_to_ids(t)
        row[f"{t}_rank"] = int(rank_of[i]) + 1
        row[f"{t}_prob"] = float(probs[i])
    return row


def main():
    C.load()

    # ---------------------------------------------------------------- 2.1
    C.banner("C2.1 — SUBJECT effect (verb fixed = 'taking')")
    rows = [probe_row(s, "subject", n) for n, s in C.subject_sweep("taking").items()]
    sub = pd.DataFrame(rows)
    sub.to_csv(C.out_path("c2_subject_effect.csv"), index=False)
    print(C.df_to_md(sub[["subject", "winner", "winner_prob",
                          "medication_rank", "medication_prob",
                          "care_rank", "part_rank"]]))

    base = sub[sub.subject == "patient"]["medication_prob"].iloc[0]
    print(f"\n  'medication' probability relative to 'patient' (= {base:.4f}):")
    for _, r in sub.iterrows():
        print(f"    {r.subject:10s} {r.medication_prob:.4f}  "
              f"({r.medication_prob / base:5.1%} of patient)  rank {r.medication_rank}")

    # ---------------------------------------------------------------- 2.2
    C.banner("C2.2 — VERB effect (subject fixed = 'person')")
    rows = [probe_row(s, "verb", n) for n, s in C.verb_sweep("person").items()]
    vb = pd.DataFrame(rows)
    vb.to_csv(C.out_path("c2_verb_effect.csv"), index=False)
    print(C.df_to_md(vb[["verb", "winner", "winner_prob",
                         "medication_rank", "medication_prob", "care_rank"]]))

    # ---------------------------------------------------------------- 2.3
    C.banner("C2.3 — MASK POSITION: which slot is hidden changes everything")
    frames = []
    for slot, sent in C.MASK_POSITION_SET.items():
        df = C.top_k(sent, k=10)
        df.insert(0, "slot", slot)
        df.insert(1, "sentence", sent)
        frames.append(df)
        print(f"\n[{slot}] {sent}")
        print("   " + ", ".join(f"{t} ({p:.3f})"
                                for t, p in zip(df.token[:8], df.prob[:8])))
    pd.concat(frames, ignore_index=True).to_csv(
        C.out_path("c2_mask_position.csv"), index=False)

    # ---------------------------------------------------------------- 2.4
    C.banner("C2.4 — CONTEXT ABLATION: strip or alter the duration phrase")
    variants = {
        "full":            "the patient is taking [MASK] since the last 2 years .",
        "no_duration":     "the patient is taking [MASK] .",
        "short_duration":  "the patient is taking [MASK] since yesterday .",
        "long_duration":   "the patient is taking [MASK] since the last 20 years .",
        "no_subject_det":  "patient is taking [MASK] since the last 2 years .",
        "past_tense":      "the patient took [MASK] since the last 2 years .",
        "negated":         "the patient is not taking [MASK] since the last 2 years .",
        "plural":          "the patients are taking [MASK] since the last 2 years .",
    }
    rows = [probe_row(s, "variant", n) for n, s in variants.items()]
    ab = pd.DataFrame(rows)
    ab["sentence"] = [variants[v] for v in ab.variant]
    ab.to_csv(C.out_path("c2_context_ablation.csv"), index=False)
    print(C.df_to_md(ab[["variant", "winner", "winner_prob",
                         "medication_rank", "medication_prob", "care_rank"]]))

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = range(len(sub))
    ax.bar([i - 0.2 for i in x], sub.medication_prob, width=0.4, label="medication")
    ax.bar([i + 0.2 for i in x], sub.care_prob, width=0.4, label="care")
    ax.set_xticks(list(x))
    ax.set_xticklabels(sub.subject, rotation=40, ha="right")
    ax.set_ylabel("probability at the mask")
    ax.set_title("C2 — the subject selects which 'take X' collocation wins")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(C.out_path("c2_subject_effect.png"), dpi=150)

    print("\nWrote c2_*.csv and c2_subject_effect.png")


if __name__ == "__main__":
    main()
