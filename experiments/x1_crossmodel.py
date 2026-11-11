"""
X1 - CROSS-MODEL RUN: the 500-sentence dataset + the "taking [MASK]" duration grid

Models (all vanilla, no fine-tuning):
    bert-base-uncased, bert-large-uncased, roberta-base, roberta-large

Three parts, all written to  results/x1_crossmodel/

PART A  behaviour + layer-wise (Aoyama & Schneider 2022 style, logit lens)
    For every sentence: one forward pass with the target masked, then read the
    prediction out of EVERY layer through the model's own MLM head.
      x1_final_<model>.csv    one row per sentence (target rank/prob, top-10, entropy)
      x1_layers_<model>.csv   one row per (sentence, layer): top-1, target rank/prob

PART B  context attribution (Zhang & Hashimoto 2021 style: how much does word j
        change the distribution at the mask?)
    Every context word is removed in TWO ways, because they disagree:
      "delete" : the word is removed from the sentence
      "mask"   : the word is replaced by the mask token (what C5 did)
    "mask" lets the model re-infer the removed word (e.g. "years" -> days/months),
    so it UNDER-states the effect of predictable words. Reporting both is the fix.
      x1_context_<model>.csv  one row per (sentence, word, method)

PART C  the original research question, across models
    "<subject> is taking <mask><duration tail>." over subjects x tails.
      x1_grid_<model>.csv

Usage (from the repository root):
    python experiments/x1_crossmodel.py                  # all 4 models, all parts
    python experiments/x1_crossmodel.py --models roberta-base # one model
    python experiments/x1_crossmodel.py --parts A C      # skip the slow part B
    python experiments/x1_crossmodel.py --limit 5        # 5 sentences per domain (smoke test)
Already-written CSVs are skipped unless --force, so an interrupted run can resume.
"""

import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import dataset

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_grad_enabled(False)

MODELS = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "results", "x1_crossmodel"))

WORD = re.compile(r"[A-Za-z0-9][\w'\-]*")


# ----------------------------------------------------------------------------- model

class MLM:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForMaskedLM.from_pretrained(name)
        self.model.eval()
        self.roberta = self.model.config.model_type in ("roberta", "xlm-roberta", "camembert")
        # The full MLM head (dense -> GELU -> LayerNorm -> decoder), not just the decoder.
        self.head = self.model.cls if hasattr(self.model, "cls") else self.model.lm_head
        self.mask = self.tok.mask_token
        self.mask_id = self.tok.mask_token_id
        self.n_layers = self.model.config.num_hidden_layers
        self._checked = False

    def text(self, masked_raw):
        return masked_raw.replace("<mask>", self.mask)

    def word_ids(self, word, after_space=True):
        """Token ids of a word as it would appear in running text."""
        if self.roberta:
            toks = self.tok.tokenize((" " if after_space else "") + word)
        else:
            toks = self.tok.tokenize(word)
        return self.tok.convert_tokens_to_ids(toks), toks

    def target_ids(self, masked_raw, target):
        i = masked_raw.index("<mask>")
        return self.word_ids(target, after_space=(i > 0 and masked_raw[i - 1] == " "))

    def batch_logits(self, texts, which):
        """
        Final-layer logits at one mask per text. which[k] = index (0-based) of the
        mask token to read in text k (texts can contain two masks in part B).
        """
        enc = self.tok(texts, return_tensors="pt", padding=True)
        out = self.model(**enc)
        rows = []
        for k in range(len(texts)):
            pos = (enc["input_ids"][k] == self.mask_id).nonzero().flatten().tolist()
            rows.append(out.logits[k, pos[which[k]]])
        return torch.stack(rows)

    def layer_logits(self, text):
        """Logit lens at the (single) mask: (n_layers+1, vocab), last row == model output."""
        enc = self.tok(text, return_tensors="pt")
        out = self.model(**enc, output_hidden_states=True)
        mi = (enc["input_ids"][0] == self.mask_id).nonzero().flatten().tolist()
        assert len(mi) == 1, f"expected one mask in {text!r}"
        H = torch.stack([h[0, mi[0]] for h in out.hidden_states])
        lg = self.head(H)
        if not self._checked:   # the lens at the top layer must reproduce the real output
            assert torch.allclose(lg[-1], out.logits[0, mi[0]], atol=1e-3), "logit-lens wiring is wrong"
            self._checked = True
        return lg

    def decode(self, i):
        return self.tok.convert_ids_to_tokens(int(i)).replace("Ġ", "")


def rank_of(logits, idx):
    return int((logits > logits[idx]).sum()) + 1


def kl(p_logits, q_logits):
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum())


# ----------------------------------------------------------------------------- part A

def part_a(m, rows, tag):
    finals, layers = [], []
    for n, r in enumerate(rows):
        tids, ttoks = m.target_ids(r["masked_raw"], r["target"])
        t0 = tids[0]
        lg = m.layer_logits(m.text(r["masked_raw"]))
        lp = torch.log_softmax(lg, -1)
        for L in range(lg.shape[0]):
            top = int(lg[L].argmax())
            layers.append({
                "domain": r["domain"], "idx": r["idx"], "layer": L,
                "rel_depth": L / m.n_layers,
                "top1": m.decode(top),
                "top1_is_target": top == t0,
                "target_rank": rank_of(lg[L], t0),
                "target_p": float(lp[L, t0].exp()),
            })
        fin = lg[-1]
        p = torch.softmax(fin, -1)
        top10 = fin.topk(10).indices.tolist()
        finals.append({
            "domain": r["domain"], "idx": r["idx"], "target": r["target"],
            "masked": r["masked_raw"],
            "target_tokens": " ".join(ttoks), "n_target_tokens": len(tids),
            "single_token": len(tids) == 1,
            "target_rank": rank_of(fin, t0), "target_p": float(p[t0]),
            "hit1": top10[0] == t0, "hit10": t0 in top10,
            "entropy": float(-(p * torch.log(p + 1e-12)).sum()),
            "top1": m.decode(top10[0]), "top1_p": float(p[top10[0]]),
            "top10": ", ".join(m.decode(i) for i in top10),
            "top10_p": ", ".join(f"{float(p[i]):.4f}" for i in top10),
        })
        if (n + 1) % 50 == 0:
            print(f"    A {n + 1}/{len(rows)}", flush=True)
    pd.DataFrame(finals).to_csv(os.path.join(OUT, f"x1_final_{tag}.csv"), index=False)
    pd.DataFrame(layers).to_csv(os.path.join(OUT, f"x1_layers_{tag}.csv"), index=False)


# ----------------------------------------------------------------------------- part B

def variants(masked_raw, mask_tok):
    """Yield (word_index, word, deleted_text, masked_text, mask_to_read) per context word."""
    words = masked_raw.split(" ")
    mw = next(i for i, w in enumerate(words) if "<mask>" in w)
    for j, w in enumerate(words):
        if j == mw or not WORD.search(w):
            continue
        core = WORD.search(w).group()
        deleted = words[:j] + ([w.replace(core, "", 1)] if w.replace(core, "", 1) else []) + words[j + 1:]
        del_text = " ".join(deleted).replace(" .", ".").replace(" ,", ",").strip()
        if del_text and del_text[0].islower() and j == 0:
            del_text = del_text[0].upper() + del_text[1:]
        masked = words[:j] + [w.replace(core, "<mask>", 1)] + words[j + 1:]
        yield j, core, del_text, " ".join(masked), (1 if j < mw else 0)


def part_b(m, rows, tag):
    out = []
    for n, r in enumerate(rows):
        t0 = m.target_ids(r["masked_raw"], r["target"])[0][0]
        V = list(variants(r["masked_raw"], m.mask))
        texts = [m.text(r["masked_raw"])]
        which = [0]
        for _, _, d, k, wk in V:
            texts += [m.text(d), m.text(k)]
            which += [0, wk]
        lg = m.batch_logits(texts, which)
        base = lg[0]
        base_lp = float(torch.log_softmax(base, -1)[t0])
        base_top = int(base.argmax())
        for v, (j, word, _, _, _) in enumerate(V):
            for meth, row in (("delete", lg[1 + 2 * v]), ("mask", lg[2 + 2 * v])):
                lp = float(torch.log_softmax(row, -1)[t0])
                top = int(row.argmax())
                out.append({
                    "domain": r["domain"], "idx": r["idx"], "target": r["target"],
                    "word_index": j, "word": word, "method": meth,
                    "target_logp_base": base_lp, "target_logp_new": lp,
                    "delta_logp": lp - base_lp,          # < 0: the word was supporting the target
                    "target_rank_new": rank_of(row, t0),
                    "kl_base_new": kl(base, row),
                    "new_top1": m.decode(top), "top1_changed": top != base_top,
                })
        if (n + 1) % 50 == 0:
            print(f"    B {n + 1}/{len(rows)}", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(OUT, f"x1_context_{tag}.csv"), index=False)


# ----------------------------------------------------------------------------- part C

SUBJECTS = {
    "patient": ("The patient", "is"), "person": ("The person", "is"),
    "man": ("The man", "is"), "woman": ("The woman", "is"),
    "someone": ("Someone", "is"), "he": ("He", "is"), "she": ("She", "is"),
    "people": ("People", "are"), "doctor": ("The doctor", "is"),
    "child": ("The child", "is"), "student": ("The student", "is"),
    "athlete": ("The athlete", "is"), "dog": ("The dog", "is"),
    "machine": ("The machine", "is"), "none": ("", ""),
}
TAILS = ["", " since the last 2 years", " for the last two years", " for years",
         " for two days", " since the last 2 minutes", " since yesterday",
         " every day", " right now", " at the moment"]
MEDICAL = ["medication", "medications", "drugs", "medicine", "medicines", "pills",
           "antibiotics", "insulin", "treatment", "antidepressants", "painkillers", "steroids"]


def part_c(m, tag):
    med_ids = []
    for w in MEDICAL:
        ids, _ = m.word_ids(w)
        if len(ids) == 1:
            med_ids.append(ids[0])
    med_id = m.word_ids("medication")[0][0]
    rows = []
    for s, (np_, be) in SUBJECTS.items():
        for tail in TAILS:
            sent = (f"{np_} {be} taking <mask>{tail}." if np_ else f"Taking <mask>{tail}.")
            lg = m.batch_logits([m.text(sent)], [0])[0]
            p = torch.softmax(lg, -1)
            top5 = lg.topk(5).indices.tolist()
            rows.append({
                "subject": s, "tail": tail.strip() or "(none)", "sentence": sent,
                "medication_rank": rank_of(lg, med_id), "medication_p": float(p[med_id]),
                "medical_mass": float(p[med_ids].sum()),
                "top1": m.decode(top5[0]), "top5": ", ".join(m.decode(i) for i in top5),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"x1_grid_{tag}.csv"), index=False)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--limit", type=int, default=0, help="sentences per domain (0 = all 50)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = dataset.load_samples(tokenizers=False)
    seen = {}
    for r in rows:
        seen[r["domain"]] = seen.get(r["domain"], 0) + 1
        r["idx"] = seen[r["domain"]]
    if a.limit:
        rows = [r for r in rows if r["idx"] <= a.limit]
    print(f"{len(rows)} sentences -> {OUT}")

    files = {"A": "x1_final_{}.csv", "B": "x1_context_{}.csv", "C": "x1_grid_{}.csv"}
    meta = {}
    for name in a.models:
        tag = name.split("/")[-1]
        todo = [p for p in a.parts
                if a.force or not os.path.exists(os.path.join(OUT, files[p].format(tag)))]
        if not todo:
            print(f"\n{name}: all requested parts exist, skipping (use --force)")
            continue
        t = time.time()
        print(f"\n=== {name} (loading)", flush=True)
        m = MLM(name)
        print(f"    {m.n_layers} layers, parts {todo}", flush=True)
        if "A" in todo:
            part_a(m, rows, tag)
        if "C" in todo:
            part_c(m, tag)
        if "B" in todo:
            part_b(m, rows, tag)
        meta[tag] = {"n_layers": m.n_layers, "seconds": round(time.time() - t, 1),
                     "n_sentences": len(rows), "parts": todo}
        print(f"    done in {time.time() - t:.0f}s", flush=True)
        del m

    mp = os.path.join(OUT, "x1_meta.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    old.update(meta)
    old["versions"] = {"torch": torch.__version__,
                       "transformers": __import__("transformers").__version__}
    json.dump(old, open(mp, "w"), indent=2)
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
