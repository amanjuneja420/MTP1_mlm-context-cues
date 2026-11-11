"""
C7 — Is 'medication' SELECTED by the subject, or is it the DEFAULT for 'taking'?

The light-verb account says the subject picks the collocation. But 'person' is
maximally generic and still yields 'medication' at rank 1. Two competing readings:

  (a) SELECTION - 'person' positively cues the medical sense, the way 'athlete'
      cues 'part'.
  (b) DEFAULT   - 'medication' is what 'taking' produces absent a strong subject
      cue, and specific subjects OVERRIDE it.

These make different predictions. Under (b), stripping the human subject entirely
should leave 'medication' on top. Under (a), it should not.

This also directly explains the C5 KL-flip if (b) holds: the subject's KL measures
how far it MOVES the prediction from the verb's default, so a strong cue scores high
and a generic one scores low.
"""

import pandas as pd
import common as C


# Progressively strip the human subject while keeping the verb frame intact.
LADDER = {
    "patient (strong medical cue)": "the patient is taking [MASK] since the last 2 years .",
    "person (generic human)":       "the person is taking [MASK] since the last 2 years .",
    "someone (indefinite human)":   "someone is taking [MASK] since the last 2 years .",
    "they (pronoun, no content)":   "they are taking [MASK] since the last 2 years .",
    "it (non-human)":               "it is taking [MASK] since the last 2 years .",
    "no subject at all":            "taking [MASK] since the last 2 years .",
    "machine (non-human agent)":    "the machine is taking [MASK] since the last 2 years .",
    "company (institution)":        "the company is taking [MASK] since the last 2 years .",
    "dog (animate non-human)":      "the dog is taking [MASK] since the last 2 years .",
}

# Control: does the same pattern hold for a DIFFERENT light verb?
CONTROL = {
    "playing / person":  "the person is playing [MASK] since the last 2 years .",
    "playing / no subj": "playing [MASK] since the last 2 years .",
    "growing / person":  "the person is growing [MASK] since the last 2 years .",
    "growing / no subj": "growing [MASK] since the last 2 years .",
}


def report(cases: dict[str, str], title: str) -> pd.DataFrame:
    C.banner(title)
    rows = []
    for label, sent in cases.items():
        df = C.top_k(sent, k=6)
        med = C.track_tokens(sent, ["medication"])
        med_final = med[med.layer == 12].iloc[0]
        rows.append({
            "case": label,
            "winner": df.token.iloc[0],
            "winner_p": df.prob.iloc[0],
            "medication_rank": int(med_final["rank"]),
            "medication_p": float(med_final["prob"]),
            "top6": ", ".join(df.token),
        })
        print(f"  {label:30s} -> {df.token.iloc[0]:12s} ({df.prob.iloc[0]:.3f})")
        print(f"  {'':30s}    {', '.join(df.token)}")
        print(f"  {'':30s}    medication: rank {int(med_final['rank'])}, "
              f"p {float(med_final['prob']):.4f}\n")
    return pd.DataFrame(rows)


def main():
    C.load()

    lad = report(LADDER, "C7.1 — strip the human subject, keep 'taking'")
    ctl = report(CONTROL, "C7.2 — control: other verbs, with and without a subject")

    lad.to_csv(C.out_path("c7_subject_ladder.csv"), index=False)
    ctl.to_csv(C.out_path("c7_verb_control.csv"), index=False)

    C.banner("C7 — verdict")
    no_subj = lad[lad.case == "no subject at all"].iloc[0]
    print(f"With NO subject at all: winner = '{no_subj.winner}' "
          f"({no_subj.winner_p:.3f}), medication rank {no_subj.medication_rank}")
    if no_subj.medication_rank <= 3:
        print("\n  => (b) DEFAULT. 'medication' survives removal of the subject entirely.")
        print("     The VERB sets a default; specific subjects OVERRIDE it.")
        print("     This also explains the C5 KL-flip: subject KL measures how far the")
        print("     subject MOVES the prediction away from the verb's default.")
    else:
        print("\n  => (a) SELECTION. Removing the subject removes the medical reading,")
        print("     so a human subject is actively required to cue it.")

    print("\nWrote c7_*.csv")


if __name__ == "__main__":
    main()
