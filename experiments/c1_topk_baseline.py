"""
C1 — Top-k baseline across the full sentence set.

Establishes the final-layer behaviour that C3-C6 explain. Nothing layer-wise here.

Outputs:
  results/case_study/c1_topk_domains.csv     top-10 for each domain frame
  results/case_study/c1_topk_subjects.csv    top-10 for the subject sweep
  results/case_study/c1_topk_verbs.csv       top-10 for the verb sweep
"""

import pandas as pd
import common as C


def collect(cases: dict[str, str], label: str) -> pd.DataFrame:
    frames = []
    for name, sent in cases.items():
        df = C.top_k(sent, k=10)
        df.insert(0, label, name)
        df.insert(1, "sentence", sent)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    C.load()

    C.banner("C1.1 — domain frames, final-layer top-10")
    domains = {k: v[0] for k, v in C.DOMAIN_SET.items()}
    d_dom = collect(domains, "domain")
    d_dom.to_csv(C.out_path("c1_topk_domains.csv"), index=False)
    for name, sent in domains.items():
        top = d_dom[d_dom.domain == name].head(10)
        words = ", ".join(f"{t}" for t in top.token)
        print(f"\n[{name}] {sent}")
        print(f"   {words}")

    C.banner("C1.2 — subject sweep (verb fixed = 'taking')")
    d_sub = collect(C.subject_sweep("taking"), "subject")
    d_sub.to_csv(C.out_path("c1_topk_subjects.csv"), index=False)
    for name in C.SUBJECTS:
        top = d_sub[d_sub.subject == name].head(5)
        print(f"  {name:10s} -> " + ", ".join(f"{t} ({p:.3f})"
                                              for t, p in zip(top.token, top.prob)))

    C.banner("C1.3 — verb sweep (subject fixed = 'person')")
    d_vb = collect(C.verb_sweep("person"), "verb")
    d_vb.to_csv(C.out_path("c1_topk_verbs.csv"), index=False)
    for name in C.VERBS:
        top = d_vb[d_vb.verb == name].head(5)
        print(f"  {name:10s} -> " + ", ".join(f"{t} ({p:.3f})"
                                              for t, p in zip(top.token, top.prob)))

    print("\nWrote c1_topk_{domains,subjects,verbs}.csv")


if __name__ == "__main__":
    main()
