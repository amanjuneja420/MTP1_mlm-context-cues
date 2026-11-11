"""
Replication of Aoyama & Schneider (2022), NAACL SRW
"Probe-Less Probing of BERT's Layer-Wise Linguistic Knowledge with Masked Word Prediction"
https://aclanthology.org/2022.naacl-srw.25/

THEIR METHOD, from paper section 3:
  1. For each sentence, create n variants (n = number of tokens) by replacing one
     token with [MASK].
  2. For each variant, run vanilla BERT to generate a prediction FROM EACH LAYER.
  3. Substitute the predicted token back in and POS-tag it to get its category.

The readout mechanism is stated in their Limitations section:
  "the layer-wise masked word prediction essentially feeds intermediate layers
   directly to the classification layer"
i.e. the LOGIT LENS - hidden state at layer L -> the MLM head. That is exactly what
common.layerwise_logits() already does, so we reuse it.

SCORES (per layer):
  word match = predicted token == original token
  POS match  = UPOS(predicted, in context) == UPOS(original, in context)
  only_pos   = POS match AND NOT word match

SUMMARY STATISTICS (Tenney et al. 2019a, eqs 3 and 4):
  differential score  D(L)   = Score(L) - Score(L-1)
  expected layer      E_D[L] = sum(L * D(L)) / sum(D(L))
Uniform contribution would give expected layer 6.
Their published values on STREUSLE: POS match 3.68, word match 5.22.

Model: bert-base-uncased, no fine-tuning. POS tagger: Stanza, UD v2 17-tag UPOS.

Usage:
    python experiments/a1_aoyama_replication.py          # 10 sentences from the project dataset
    python experiments/a1_aoyama_replication.py --demo   # the paper's own example sentence
"""

import argparse
import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# BERT's vocabulary contains characters (subscripts, CJK, ...) that some console
# encodings cannot print; force UTF-8 so that printing predictions cannot crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import common as C

N_LAYERS = 13          # 0 = embeddings, 1..12 = encoder layers

_nlp = None


def tagger():
    """Stanza pipeline, pretokenized so our token indices stay aligned."""
    global _nlp
    if _nlp is None:
        import stanza
        _nlp = stanza.Pipeline(lang="en", processors="tokenize,pos",
                               tokenize_pretokenized=True, verbose=False)
    return _nlp


def upos_of_sentence(words):
    """UPOS tag for every word of an already-tokenized sentence."""
    doc = tagger()([words])
    return [w.upos for w in doc.sentences[0].words]


# --------------------------------------------------------------------- core

def predictions_by_layer(words, mask_idx):
    """
    Mask words[mask_idx], then read the top-1 prediction out of EVERY layer via the
    MLM head (the paper's "feed intermediate layers to the classification layer").
    Returns a list of 13 predicted token strings.
    """
    tok, _ = C.load()
    masked = list(words)
    masked[mask_idx] = tok.mask_token
    lg = C.layerwise_logits(" ".join(masked))          # (13, vocab)
    return [tok.convert_ids_to_tokens(int(lg[L].argmax())) for L in range(N_LAYERS)]


def topk_by_layer(words, mask_idx, k=10):
    tok, _ = C.load()
    masked = list(words)
    masked[mask_idx] = tok.mask_token
    lg = C.layerwise_logits(" ".join(masked))
    return {L: [tok.convert_ids_to_tokens(int(i)) for i in lg[L].topk(k).indices]
            for L in range(N_LAYERS)}


def score_sentence(words, gold_upos):
    """
    Full Aoyama & Schneider procedure over one sentence.
    Returns long format: one row per (masked position, layer).
    """
    n = len(words)
    preds = {i: predictions_by_layer(words, i) for i in range(n)}

    # Substitute each prediction back in and tag IN CONTEXT.
    # One tagger call per layer, batching all n substitutions.
    tagged = {}
    for L in range(N_LAYERS):
        batch = []
        for i in range(n):
            variant = list(words)
            p = preds[i][L]
            variant[i] = p if p.strip() else "unk"
            batch.append(variant)
        doc = tagger()(batch)
        for i, sent in enumerate(doc.sentences):
            tags = [w.upos for w in sent.words]
            tagged[(i, L)] = tags[i] if i < len(tags) else None

    rows = []
    for i in range(n):
        for L in range(N_LAYERS):
            p = preds[i][L]
            rows.append({
                "position": i,
                "orig_token": words[i],
                "orig_upos": gold_upos[i],
                "layer": L,
                "pred_token": p,
                "pred_upos": tagged[(i, L)],
                "word_match": p.lower() == words[i].lower(),
                "pos_match": tagged[(i, L)] == gold_upos[i],
            })

    df = pd.DataFrame(rows)
    df["only_pos"] = df.pos_match & ~df.word_match
    return df


# ------------------------------------------------------- Tenney statistics

def differential(scores):
    """D(L) = Score(L) - Score(L-1), for L = 1..12."""
    return {L: scores[L] - scores[L - 1] for L in range(1, N_LAYERS)}


def expected_layer(scores):
    """E_D[L] = sum(L * D(L)) / sum(D(L)). Uniform contribution -> 6.0."""
    d = differential(scores)
    denom = sum(d.values())
    if abs(denom) < 1e-12:
        return float("nan")
    return sum(L * v for L, v in d.items()) / denom


def layer_scores(df, col):
    return {L: float(df[df.layer == L][col].mean()) for L in range(N_LAYERS)}


# ---------------------------------------------------------------- reporting

def report(df, label):
    C.banner(f"{label}\n   {df.position.nunique()} masked positions x {N_LAYERS} layers "
             f"= {len(df)} predictions")

    pos = layer_scores(df, "pos_match")
    word = layer_scores(df, "word_match")
    only = layer_scores(df, "only_pos")
    dpos, dword = differential(pos), differential(word)

    print(f"{'layer':>5} {'POS match':>10} {'word match':>11} {'POS only':>9}"
          f" {'dPOS':>8} {'dword':>8}")
    for L in range(N_LAYERS):
        dp = f"{dpos[L]:+.3f}" if L else "     -"
        dw = f"{dword[L]:+.3f}" if L else "     -"
        print(f"{L:>5} {pos[L]:>10.3f} {word[L]:>11.3f} {only[L]:>9.3f} {dp:>8} {dw:>8}")

    e_pos, e_word = expected_layer(pos), expected_layer(word)
    print(f"\n  EXPECTED LAYER:  POS match {e_pos:.2f}   word match {e_word:.2f}")
    print(f"  paper (STREUSLE): POS 3.68, word 5.22   |   uniform would be 6.00")
    return {"label": label, "expected_layer_pos": e_pos, "expected_layer_word": e_word,
            "final_pos_acc": pos[12], "final_word_acc": word[12],
            "pos_by_layer": pos, "word_by_layer": word}


def by_upos(df):
    """Their section 4.1.2: expected layer per UPOS category."""
    rows = []
    for tag, g in df.groupby("orig_upos"):
        n = len(g[g.layer == 0])
        if n < 2:
            continue
        rows.append({
            "upos": tag, "n_tokens": n,
            "exp_layer_word": expected_layer(layer_scores(g, "word_match")),
            "exp_layer_pos": expected_layer(layer_scores(g, "pos_match")),
            "final_word_acc": float(g[g.layer == 12].word_match.mean()),
        })
    return pd.DataFrame(rows).sort_values("exp_layer_word")


def qualitative(words, mask_idx, gold, layers=(0, 3, 6, 9, 11, 12), k=10):
    """Their Table 2 format: top-k at selected layers for one masked token."""
    tk = topk_by_layer(words, mask_idx, k)
    shown = list(words)
    shown[mask_idx] = "[MASK]"
    print(f"\n  Context: {' '.join(shown)}")
    print(f"  Original: '{gold}'")
    for L in layers:
        mark = "   <-- correct at top-1" if tk[L][0].lower() == gold.lower() else ""
        print(f"    L{L:<2} {', '.join(tk[L])}{mark}")
    first = next((L for L in range(N_LAYERS) if tk[L][0].lower() == gold.lower()), None)
    print(f"    -> '{gold}' first tops the list at layer {first}")


# -------------------------------------------------------------------- data

def load_examples(n=10):
    """n sentences from the project dataset, spread across domains."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import load_samples
    rows = [r for r in load_samples() if r["bert_single_token"]]
    doms = sorted({r["domain"] for r in rows})
    per = max(1, n // len(doms))
    out = []
    for dom in doms:
        out.extend([r for r in rows if r["domain"] == dom][:per])
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run the paper's own example sentence instead")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    C.load()
    tagger()

    if args.demo:
        sents = [{
            "domain": "PAPER EXAMPLE",
            "words": ("stephanie 's knowledge of the market and properties in our price "
                      "range , made us feel secure in our decision to buy when we did .").split(),
            "target": "us",
        }]
    else:
        sents = [{"domain": r["domain"], "words": r["sentence_words"],
                  "target": r["target"]} for r in load_examples(args.n)]

    all_df, summaries = [], []
    for s in sents:
        words = s["words"]
        gold_upos = upos_of_sentence(words)
        df = score_sentence(words, gold_upos)
        df.insert(0, "domain", s["domain"])
        df.insert(1, "sentence", " ".join(words))
        all_df.append(df)
        summaries.append(report(df, f"[{s['domain']}] {' '.join(words)}"))

        tl = [w.lower() for w in words]
        if s["target"].lower() in tl:
            qualitative(words, tl.index(s["target"].lower()), s["target"])

    full = pd.concat(all_df, ignore_index=True)
    full.to_csv(C.out_path("a1_aoyama_predictions.csv"), index=False)

    C.banner("AGGREGATE over all sentences")
    agg = report(full, "ALL SENTENCES")

    C.banner("BY UPOS  (their section 4.1.2)")
    bu = by_upos(full)
    print(C.df_to_md(bu))
    bu.to_csv(C.out_path("a1_aoyama_by_upos.csv"), index=False)

    with open(C.out_path("a1_aoyama_summary.json"), "w") as f:
        json.dump({"aggregate": agg, "per_sentence": summaries}, f, indent=2)

    print("\nWrote a1_aoyama_predictions.csv, a1_aoyama_by_upos.csv, a1_aoyama_summary.json")


if __name__ == "__main__":
    main()
