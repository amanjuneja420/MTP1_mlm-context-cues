"""
Loader for the 500-sentence masked-word dataset (data/sentences.csv).

500 sentences, 10 domains x 50. Columns: domain | no | original_sentence | target | masked_sentence.
The masked sentences use RoBERTa-style <mask>; BERT needs [MASK].

Each returned row:
    domain              e.g. "MEDICINE"
    original            original sentence, with [TARGET: x] markup stripped
    target              the masked word
    masked_bert         sentence with [MASK]
    masked_raw          sentence as given, with <mask>
    sentence_words      whitespace tokens of `original` (target restored in place)
    bert_single_token   target is one WordPiece in bert-base-uncased
    roberta_single_token
"""

import csv
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(_HERE, "..", "data", "sentences.csv"))

# curly quotes and other mojibake seen in the sheet
_FIXES = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "�": "'",
}


def _clean(s):
    if s is None:
        return ""
    for a, b in _FIXES.items():
        s = s.replace(a, b)
    return s.strip()


def word_tokens(text):
    """
    Simple word tokenizer that splits punctuation off, so POS tagging and
    per-token masking line up (STREUSLE, which the paper uses, is pre-tokenized).
    """
    return re.findall(r"[\w'-]+|[^\w\s]", text)


def load_samples(path=DATA, tokenizers=True):
    """Return all 500 rows, in file order."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    bt = rt = None
    if tokenizers:
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import BertTokenizer, RobertaTokenizer
        bt = BertTokenizer.from_pretrained("bert-base-uncased")
        rt = RobertaTokenizer.from_pretrained("roberta-base")

    out = []
    for cells in rows:
        domain = cells["domain"]
        target = _clean(cells["target"])
        masked_raw = _clean(cells["masked_sentence"])
        original = _clean(cells["original_sentence"])
        if not target or not masked_raw:
            continue

        # strip the [TARGET: x] markup from the original sentence
        original = re.sub(r"\[TARGET:\s*([^\]]+)\]", r"\1", original)

        masked_bert = masked_raw.replace("<mask>", "[MASK]")

        rec = {
            "domain": domain,
            "original": original,
            "target": target,
            "masked_raw": masked_raw,
            "masked_bert": masked_bert,
            "sentence_words": word_tokens(original),
        }
        if tokenizers:
            rec["bert_single_token"] = len(bt.tokenize(target)) == 1
            rec["roberta_single_token"] = len(rt.tokenize(" " + target)) == 1
        out.append(rec)
    return out


def single_token_subset(rows=None, model="bert"):
    rows = rows if rows is not None else load_samples()
    key = "bert_single_token" if model == "bert" else "roberta_single_token"
    return [r for r in rows if r[key]]


if __name__ == "__main__":
    rows = load_samples()
    from collections import Counter
    c = Counter(r["domain"] for r in rows)
    print(f"total rows: {len(rows)}")
    for d, n in c.items():
        sub = [r for r in rows if r["domain"] == d]
        b = sum(r["bert_single_token"] for r in sub)
        print(f"  {d:20s} n={n:3d}  bert single-token {b:3d} ({b/n:.0%})")
    st = single_token_subset(rows)
    print(f"\nBERT single-token subset: {len(st)} / {len(rows)}")
    print("\nexample:")
    r = st[0]
    for k, v in r.items():
        print(f"  {k:22s} {v}")
