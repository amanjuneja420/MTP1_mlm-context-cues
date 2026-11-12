"""
X2 - EXTENDED EXPERIMENTS

Every part is tied to a paper from the project reading list. All outputs go to
    results/x2_extended/

  E1  domain   Domain-adapted BERTs (BiomedBERT, Legal-BERT, Bio_ClinicalBERT) on the
               500-sentence sheet + the "taking [MASK]" grid.          [Vakili 2022, Nozza 2020]
               -> reuses parts A and C of x1_crossmodel.py, written to x2_extended/.
  E2  ar       GPT-2 / GPT-2-medium as an autoregressive comparison:  [MARIA 2026, FIM 2022]
               (a) candidate re-scoring on the sheet, (b) left-context-only next-word
               prediction, (c) the medical-vs-other share on the grid, for GPT-2 AND the
               MLMs over the SAME candidate set.
  E3  aoyama   Aoyama & Schneider (2022) layer-wise word/POS prediction on 50 sheet
               sentences (5 per domain) for the 4 base models.        [Aoyama & Schneider 2022]
  E4  attn     Where the mask attends, per layer (CLS / SEP / self / neighbours / punct),
               attention entropy, on the 500 sentences.                [Clark et al. 2019]
  E5  conf     Saves each model's full sorted output distribution at the mask, for
               conformal prediction sets (computed in analysis).       [Dey et al. 2022]
  E6  multi    Multi-token targets: k masks for a k-piece target; parallel, left-to-right
               and confidence-first decoding.                          [Kalinsky et al. 2023]
  E7  patch    Activation patching on all 4 base models: subject (patient->machine),
               duration (years->minutes) and verb (taking->eating).    [Meng et al. 2022 / C8]
  E8  grid2    Extended minimal-pair grid: negation, tense, other verbs. [Ettinger 2020]

Usage (from the repository root):
    python experiments/x2_extended.py                       # everything (approx. 1.5-2 h on CPU)
    python experiments/x2_extended.py --parts patch grid2   # chosen parts only
    python experiments/x2_extended.py --limit 3             # smoke test: 3 sentences per domain
Each part writes its own CSVs and is skipped if they already exist (use --force to redo).
A failing part is logged and the run continues with the next one.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoTokenizer)

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_grad_enabled(False)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x1_crossmodel as cm                         # MLM wrapper, part_a, part_c
import dataset

OUT = os.path.abspath(os.path.join(HERE, "..", "results", "x2_extended"))
X1 = os.path.abspath(os.path.join(HERE, "..", "results", "x1_crossmodel"))

BASE4 = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large"]
DOMAIN_MODELS = ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 "nlpaueb/legal-bert-base-uncased",
                 "emilyalsentzer/Bio_ClinicalBERT"]
AR_MODELS = ["gpt2", "gpt2-medium"]

MEDICAL = ["medication", "medications", "drugs", "medicine", "medicines", "pills",
           "antibiotics", "insulin", "treatment", "antidepressants", "painkillers", "steroids"]
OTHER = ["care", "part", "place", "classes", "lessons", "notes", "pictures", "photos",
         "over", "off", "it", "revenge", "bribes", "selfies", "vacations", "turns", "action",
         "risks", "time", "steps", "advantage", "responsibility", "charge", "notice",
         "shelter", "orders", "courses", "exams", "breaks", "money"]


def tag(name):
    return name.split("/")[-1]


def path(name):
    return os.path.join(OUT, name)


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data

def load_rows(limit=0):
    rows = dataset.load_samples(tokenizers=False)
    seen = {}
    for r in rows:
        seen[r["domain"]] = seen.get(r["domain"], 0) + 1
        r["idx"] = seen[r["domain"]]
    if limit:
        rows = [r for r in rows if r["idx"] <= limit]
    return rows


def check_mlm_head(name):
    """Refuse a checkpoint whose MLM head would be randomly initialised."""
    _, info = AutoModelForMaskedLM.from_pretrained(name, output_loading_info=True)
    bad = [k for k in info.get("missing_keys", [])
           if ("predictions" in k or "lm_head" in k) and "decoder.weight" not in k]
    return bad


# ============================================================================ E1 domain

def e1_domain(rows, models, force):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"x1_final_{t}.csv")) and os.path.exists(path(f"x1_grid_{t}.csv")):
            log(f"  [E1] {t}: exists, skip"); continue
        bad = check_mlm_head(name)
        if bad:
            log(f"  [E1] {t}: MLM head missing ({bad[:3]}) - SKIPPED"); continue
        m = cm.MLM(name)
        old = cm.OUT
        cm.OUT = OUT                       # write into x2_extended/
        try:
            cm.part_a(m, rows, t)
            cm.part_c(m, t)
        finally:
            cm.OUT = old
        log(f"  [E1] {t}: done")


# ============================================================================ E2 autoregressive

def clean_word(w):
    w = str(w).strip()
    if not w or w.startswith("##") or not re.match(r"^[A-Za-z][A-Za-z'\-]*$", w):
        return None
    return w


def candidate_pool(r, finals):
    pool = {r["target"]}
    for f in finals:
        row = f.get((r["domain"], r["idx"]))
        if row is None:
            continue
        for w in str(row).split(", "):
            w = clean_word(w)
            if w:
                pool.add(w.lower() if w.lower() != r["target"].lower() else r["target"])
    return sorted(pool)


class AR:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def sentence_logprob(self, sents, bs=32):
        """Total log P(sentence) under the LM, with BOS prepended."""
        out = []
        for i in range(0, len(sents), bs):
            enc = [[self.bos] + self.tok(s)["input_ids"] for s in sents[i:i + bs]]
            L = max(map(len, enc))
            ids = torch.full((len(enc), L), self.tok.pad_token_id)
            att = torch.zeros((len(enc), L), dtype=torch.long)
            for k, e in enumerate(enc):
                ids[k, :len(e)] = torch.tensor(e); att[k, :len(e)] = 1
            lp = torch.log_softmax(self.model(input_ids=ids, attention_mask=att).logits[:, :-1], -1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * att[:, 1:]
            out += tok_lp.sum(-1).tolist()
        return out

    def next_word(self, prefix, k=10):
        ids = torch.tensor([[self.bos] + self.tok(prefix)["input_ids"]])
        lg = self.model(input_ids=ids).logits[0, -1]
        return lg

    def first_id(self, word, after_space=True):
        return self.tok((" " if after_space else "") + word)["input_ids"][0]


def mlm_scores(m, masked_raw, words):
    """log p(first sub-token of each word) at the mask under an MLM."""
    lg = m.batch_logits([m.text(masked_raw)], [0])[0]
    lp = torch.log_softmax(lg, -1)
    i = masked_raw.index("<mask>")
    sp = i > 0 and masked_raw[i - 1] == " "
    return [float(lp[m.word_ids(w, after_space=sp)[0][0]]) for w in words]


def e2_ar(rows, ar_models, mlm_models, force):
    finals = []
    for mname in BASE4:
        p = os.path.join(X1, f"x1_final_{mname}.csv")
        if os.path.exists(p):
            f = pd.read_csv(p)
            finals.append({(d, i): t for d, i, t in zip(f.domain, f.idx, f.top10)})
    pools = {(r["domain"], r["idx"]): candidate_pool(r, finals) for r in rows}
    log(f"  [E2] candidate pools: mean size {np.mean([len(v) for v in pools.values()]):.1f}")

    # ---- (a)+(b) on the sheet, AR models
    for name in ar_models:
        t = tag(name)
        if not force and os.path.exists(path(f"e2_sheet_{t}.csv")):
            log(f"  [E2] {t}: exists, skip"); continue
        ar = AR(name)
        out = []
        for n, r in enumerate(rows):
            pool = pools[(r["domain"], r["idx"])]
            sents = [r["masked_raw"].replace("<mask>", w) for w in pool]
            s = np.array(ar.sentence_logprob(sents))
            order = np.argsort(-s)
            tgt = pool.index(r["target"])
            # (b) left context only: next-token prediction at the mask position
            prefix = r["masked_raw"].split("<mask>")[0]
            sp = prefix.endswith(" ")
            lg = ar.next_word(prefix.rstrip(" ")) if prefix.strip() else None
            if lg is not None:
                tid = ar.first_id(r["target"], after_space=sp)
                lr = int((lg > lg[tid]).sum()) + 1
                top1 = ar.tok.decode([int(lg.argmax())]).strip()
            else:
                lr, top1 = np.nan, ""
            out.append({"domain": r["domain"], "idx": r["idx"], "target": r["target"],
                        "pool_size": len(pool), "pool_rank": int(np.where(order == tgt)[0][0]) + 1,
                        "pool_top1": pool[order[0]], "left_only_rank": lr, "left_only_top1": top1})
            if (n + 1) % 100 == 0:
                log(f"    E2 {t} {n + 1}/{len(rows)}")
        pd.DataFrame(out).to_csv(path(f"e2_sheet_{t}.csv"), index=False)
        # (c) grid share
        e2_grid(ar=ar, t=t)
        del ar

    # ---- MLMs over the same pools (fair comparison) + their grid share
    for name in mlm_models:
        t = tag(name)
        if not force and os.path.exists(path(f"e2_sheet_{t}.csv")):
            log(f"  [E2] {t}: exists, skip"); continue
        m = cm.MLM(name)
        out = []
        for r in rows:
            pool = pools[(r["domain"], r["idx"])]
            s = np.array(mlm_scores(m, r["masked_raw"], pool))
            order = np.argsort(-s)
            tgt = pool.index(r["target"])
            out.append({"domain": r["domain"], "idx": r["idx"], "target": r["target"],
                        "pool_size": len(pool), "pool_rank": int(np.where(order == tgt)[0][0]) + 1,
                        "pool_top1": pool[order[0]]})
        pd.DataFrame(out).to_csv(path(f"e2_sheet_{t}.csv"), index=False)
        e2_grid(m=m, t=t)
        del m


GRID_SUBJ = {"patient": ("The patient", "is"), "person": ("The person", "is"),
             "woman": ("The woman", "is"), "man": ("The man", "is"),
             "someone": ("Someone", "is"), "child": ("The child", "is"),
             "dog": ("The dog", "is"), "machine": ("The machine", "is")}
GRID_TAILS = ["", " since the last 2 years", " for the last two years", " for years",
              " for two days", " since the last 2 minutes", " since yesterday",
              " every day", " right now", " at the moment"]


def e2_grid(ar=None, m=None, t=""):
    words = MEDICAL + OTHER
    rows = []
    for s, (np_, be) in GRID_SUBJ.items():
        for tail in GRID_TAILS:
            frame = f"{np_} {be} taking <mask>{tail}."
            if ar is not None:
                sc = np.array(ar.sentence_logprob([frame.replace("<mask>", w) for w in words]))
            else:
                sc = np.array(mlm_scores(m, frame, words))
            p = np.exp(sc - sc.max()); p /= p.sum()
            rows.append({"subject": s, "tail": tail.strip() or "(none)",
                         "medical_share": float(p[:len(MEDICAL)].sum()),
                         "top1": words[int(p.argmax())]})
    pd.DataFrame(rows).to_csv(path(f"e2_grid_{t}.csv"), index=False)


# ============================================================================ E3 aoyama

_nlp = None


def tagger():
    global _nlp
    if _nlp is None:
        import stanza
        _nlp = stanza.Pipeline(lang="en", processors="tokenize,pos",
                               tokenize_pretokenized=True, verbose=False)
    return _nlp


def e3_aoyama(rows, models, per_domain, force):
    sub = [r for r in rows if r["idx"] <= per_domain]
    nlp = tagger()
    gold = {}
    for r in sub:
        w = r["sentence_words"]
        gold[(r["domain"], r["idx"])] = [x.upos for x in nlp([w]).sentences[0].words]
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e3_aoyama_{t}.csv")):
            log(f"  [E3] {t}: exists, skip"); continue
        m = cm.MLM(name)
        preds = {}           # (dom, idx, pos) -> list of tokens per layer
        for r in sub:
            words = r["sentence_words"]
            for i in range(len(words)):
                masked = list(words); masked[i] = m.mask
                lg = m.layer_logits(" ".join(masked))
                preds[(r["domain"], r["idx"], i)] = [m.decode(int(lg[L].argmax())) for L in range(lg.shape[0])]
        # tag every distinct substituted sentence once
        cache, todo = {}, []
        for (d, k, i), pl in preds.items():
            words = next(r["sentence_words"] for r in sub if r["domain"] == d and r["idx"] == k)
            for p in set(pl):
                v = list(words); v[i] = p if p.strip() else "unk"
                key = (d, k, i, p)
                if key not in cache:
                    cache[key] = None; todo.append((key, v))
        B = 256
        for s in range(0, len(todo), B):
            chunk = todo[s:s + B]
            doc = nlp([v for _, v in chunk])
            for (key, _), sent in zip(chunk, doc.sentences):
                tags = [w.upos for w in sent.words]
                cache[key] = tags[key[2]] if key[2] < len(tags) else None
        out = []
        for (d, k, i), pl in preds.items():
            words = next(r["sentence_words"] for r in sub if r["domain"] == d and r["idx"] == k)
            g = gold[(d, k)][i]
            for L, p in enumerate(pl):
                out.append({"domain": d, "idx": k, "position": i, "orig_token": words[i],
                            "orig_upos": g, "layer": L, "rel_depth": L / m.n_layers,
                            "pred_token": p, "pred_upos": cache[(d, k, i, p)],
                            "word_match": p.lower() == words[i].lower(),
                            "pos_match": cache[(d, k, i, p)] == g})
        pd.DataFrame(out).to_csv(path(f"e3_aoyama_{t}.csv"), index=False)
        log(f"  [E3] {t}: {len(preds)} masked positions")
        del m


# ============================================================================ E4 attention

PUNCT = set(".,;:!?'\"-()")


def e4_attention(rows, models, force):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e4_attention_{t}.csv")):
            log(f"  [E4] {t}: exists, skip"); continue
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name, attn_implementation="eager")
        model.eval()
        out = []
        for r in rows:
            text = r["masked_raw"].replace("<mask>", tok.mask_token)
            enc = tok(text, return_tensors="pt")
            ids = enc["input_ids"][0]
            att = model(**enc, output_attentions=True).attentions      # L x (1,H,T,T)
            if not att or att[0] is None:
                raise RuntimeError("no attentions returned (need eager attention)")
            mi = int((ids == tok.mask_token_id).nonzero()[0])
            toks = tok.convert_ids_to_tokens(ids)
            n = len(ids)
            is_p = torch.tensor([t_.replace("Ġ", "") in PUNCT for t_ in toks])
            for L, a in enumerate(att):
                A = a[0, :, mi, :]                                      # H x T
                ent = -(A * torch.log(A + 1e-12)).sum(-1)               # per head
                out.append({
                    "domain": r["domain"], "idx": r["idx"], "layer": L + 1,
                    "rel_depth": (L + 1) / len(att),
                    "to_cls": float(A[:, 0].mean()), "to_sep": float(A[:, n - 1].mean()),
                    "to_self": float(A[:, mi].mean()),
                    "to_prev": float(A[:, mi - 1].mean()) if mi - 1 > 0 else 0.0,
                    "to_next": float(A[:, mi + 1].mean()) if mi + 1 < n - 1 else 0.0,
                    "to_punct": float(A[:, is_p].sum(-1).mean()),
                    "entropy": float(ent.mean()),
                    "heads_sep_gt_half": float((A[:, n - 1] > 0.5).float().mean()),
                    "heads_cls_gt_half": float((A[:, 0] > 0.5).float().mean()),
                })
        pd.DataFrame(out).to_csv(path(f"e4_attention_{t}.csv"), index=False)
        log(f"  [E4] {t}: done")
        del model


# ============================================================================ E5 conformal

def e5_conformal(rows, models, force, keep=5000):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e5_dist_{t}.npz")):
            log(f"  [E5] {t}: exists, skip"); continue
        m = cm.MLM(name)
        P, tp, meta = [], [], []
        for r in rows:
            lg = m.batch_logits([m.text(r["masked_raw"])], [0])[0]
            p = torch.softmax(lg, -1)
            tids, _ = m.target_ids(r["masked_raw"], r["target"])
            P.append(torch.sort(p, descending=True).values[:keep].numpy().astype(np.float32))
            tp.append(float(p[tids[0]]))
            meta.append((r["domain"], r["idx"], len(tids) == 1))
        np.savez_compressed(path(f"e5_dist_{t}.npz"), sorted_probs=np.stack(P),
                            target_p=np.array(tp), domain=np.array([x[0] for x in meta]),
                            idx=np.array([x[1] for x in meta]), single=np.array([x[2] for x in meta]))
        log(f"  [E5] {t}: done")
        del m


# ============================================================================ E6 multi-token

def e6_multitoken(rows, models, force):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e6_multitoken_{t}.csv")):
            log(f"  [E6] {t}: exists, skip"); continue
        m = cm.MLM(name)
        out = []
        for r in rows:
            tids, ttoks = m.target_ids(r["masked_raw"], r["target"])
            k = len(tids)
            if k < 2:
                continue
            sep = "" if m.roberta else " "
            text = m.text(r["masked_raw"].replace("<mask>", sep.join(["<mask>"] * k)))
            enc = m.tok(text, return_tensors="pt")
            ids = enc["input_ids"].clone()
            pos = (ids[0] == m.mask_id).nonzero().flatten().tolist()
            if len(pos) != k:
                continue

            def run(x):
                return m.model(input_ids=x, attention_mask=enc["attention_mask"]).logits[0]

            lg = run(ids)
            par = [int(lg[p].argmax()) for p in pos]
            # left-to-right
            x = ids.clone(); l2r = []
            for p in pos:
                g = int(run(x)[p].argmax()); x[0, p] = g; l2r.append(g)
            # confidence-first
            x = ids.clone(); left = list(pos); conf = {}
            while left:
                L_ = run(x)
                pr = {p: torch.softmax(L_[p], -1).max(0) for p in left}
                best = max(left, key=lambda p: float(pr[p][0]))
                g = int(pr[best][1]); x[0, best] = g; conf[best] = g; left.remove(best)
            cf = [conf[p] for p in pos]
            # gold sequence log-prob under left-to-right teacher forcing
            x = ids.clone(); glp = 0.0
            for p, g in zip(pos, tids):
                glp += float(torch.log_softmax(run(x)[p], -1)[g]); x[0, p] = g
            dec = lambda s: m.tok.convert_tokens_to_string(m.tok.convert_ids_to_tokens(s)).strip()
            out.append({"domain": r["domain"], "idx": r["idx"], "target": r["target"], "k": k,
                        "pieces": " ".join(ttoks),
                        "parallel": dec(par), "l2r": dec(l2r), "conf_first": dec(cf),
                        "exact_parallel": par == tids, "exact_l2r": l2r == tids, "exact_conf": cf == tids,
                        "first_piece_parallel": par[0] == tids[0],
                        "gold_logprob_l2r": glp})
        pd.DataFrame(out).to_csv(path(f"e6_multitoken_{t}.csv"), index=False)
        log(f"  [E6] {t}: {len(out)} multi-token targets")
        del m


# ============================================================================ E7 patching

PATCH_SETUPS = {
    "subject_patient_to_machine": ("The patient is taking <mask> since the last 2 years.",
                                   "The machine is taking <mask> since the last 2 years."),
    "duration_years_to_minutes": ("The person is taking <mask> since the last 2 years.",
                                  "The person is taking <mask> since the last 2 minutes."),
    "verb_taking_to_eating": ("The patient is taking <mask> since the last 2 years.",
                              "The patient is eating <mask> since the last 2 years."),
}


def layers_of(model):
    base = model.base_model
    return base.embeddings, base.encoder.layer


def medical_mass(m, logits):
    ids = [m.word_ids(w)[0][0] for w in MEDICAL if len(m.word_ids(w)[0]) == 1]
    return float(torch.softmax(logits, -1)[ids].sum())


def e7_patching(models, force):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e7_patching_{t}.csv")):
            log(f"  [E7] {t}: exists, skip"); continue
        m = cm.MLM(name)
        emb, enc_layers = layers_of(m.model)
        out = []
        for setup, (clean, corrupt) in PATCH_SETUPS.items():
            ce = m.tok(m.text(clean), return_tensors="pt")
            xe = m.tok(m.text(corrupt), return_tensors="pt")
            if ce["input_ids"].shape != xe["input_ids"].shape:
                log(f"  [E7] {t} {setup}: length mismatch, skipped"); continue
            mi = int((ce["input_ids"][0] == m.mask_id).nonzero()[0])
            co = m.model(**ce, output_hidden_states=True)
            hs = co.hidden_states
            m_clean = medical_mass(m, co.logits[0, mi])
            m_corr = medical_mass(m, m.model(**xe).logits[0, mi])
            T = ce["input_ids"].shape[1]
            ctoks = [x.replace("Ġ", "") for x in m.tok.convert_ids_to_tokens(ce["input_ids"][0])]
            xtoks = [x.replace("Ġ", "") for x in m.tok.convert_ids_to_tokens(xe["input_ids"][0])]
            ids_b = xe["input_ids"].repeat(T, 1)
            att_b = xe["attention_mask"].repeat(T, 1)
            for L in range(len(hs)):
                clean_L = hs[L][0]

                def hook(mod, args, output, clean_L=clean_L):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    for i in range(T):
                        h[i, i] = clean_L[i]
                    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

                mod = emb if L == 0 else enc_layers[L - 1]
                hd = mod.register_forward_hook(hook)
                try:
                    lg = m.model(input_ids=ids_b, attention_mask=att_b).logits
                finally:
                    hd.remove()
                for i in range(T):
                    mm = medical_mass(m, lg[i, mi])
                    out.append({"setup": setup, "layer": L, "rel_depth": L / m.n_layers,
                                "position": i, "token_clean": ctoks[i], "token_corrupt": xtoks[i],
                                "is_mask": i == mi, "differs": ctoks[i] != xtoks[i],
                                "mass_clean": m_clean, "mass_corrupt": m_corr, "mass_patched": mm,
                                "restoration": (mm - m_corr) / (m_clean - m_corr) if abs(m_clean - m_corr) > 1e-9 else np.nan})
        pd.DataFrame(out).to_csv(path(f"e7_patching_{t}.csv"), index=False)
        log(f"  [E7] {t}: done")
        del m


# ============================================================================ E8 grid2

FRAMES = {
    "is taking":      "{np} {be} taking <mask>{tail}.",
    "is not taking":  "{np} {be} not taking <mask>{tail}.",
    "has been taking": "{np} {has} been taking <mask>{tail}.",
    "takes":          "{np} {takes} <mask>{tail}.",
    "took":           "{np} took <mask>{tail}.",
    "is using":       "{np} {be} using <mask>{tail}.",
    "is on":          "{np} {be} on <mask>{tail}.",
}
G2_SUBJ = {"patient": ("The patient", "is", "has", "takes"), "person": ("The person", "is", "has", "takes"),
           "woman": ("The woman", "is", "has", "takes"), "man": ("The man", "is", "has", "takes"),
           "someone": ("Someone", "is", "has", "takes"), "child": ("The child", "is", "has", "takes"),
           "people": ("People", "are", "have", "take"), "machine": ("The machine", "is", "has", "takes")}
G2_TAILS = ["", " for two days", " for years", " since the last 2 years", " right now"]


def e8_grid2(models, force):
    for name in models:
        t = tag(name)
        if not force and os.path.exists(path(f"e8_grid2_{t}.csv")):
            log(f"  [E8] {t}: exists, skip"); continue
        bad = check_mlm_head(name)
        if bad:
            log(f"  [E8] {t}: MLM head missing - skipped"); continue
        m = cm.MLM(name)
        med_ids = [m.word_ids(w)[0][0] for w in MEDICAL if len(m.word_ids(w)[0]) == 1]
        med_id = m.word_ids("medication")[0][0]
        out = []
        for fr, tmpl in FRAMES.items():
            for s, (np_, be, has, takes) in G2_SUBJ.items():
                for tail in G2_TAILS:
                    sent = tmpl.format(np=np_, be=be, has=has, takes=takes, tail=tail)
                    lg = m.batch_logits([m.text(sent)], [0])[0]
                    p = torch.softmax(lg, -1)
                    top5 = lg.topk(5).indices.tolist()
                    out.append({"frame": fr, "subject": s, "tail": tail.strip() or "(none)",
                                "sentence": sent, "medical_mass": float(p[med_ids].sum()),
                                "medication_rank": cm.rank_of(lg, med_id), "medication_p": float(p[med_id]),
                                "top1": m.decode(top5[0]), "top5": ", ".join(m.decode(i) for i in top5)})
        pd.DataFrame(out).to_csv(path(f"e8_grid2_{t}.csv"), index=False)
        log(f"  [E8] {t}: done")
        del m


# ============================================================================ main

PARTS = ["grid2", "patch", "domain", "conf", "multi", "ar", "attn", "aoyama"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=PARTS, choices=PARTS)
    ap.add_argument("--limit", type=int, default=0, help="sentences per domain (0 = all 50)")
    ap.add_argument("--aoyama-per-domain", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--base", nargs="+", default=BASE4)
    ap.add_argument("--domain-models", nargs="+", default=DOMAIN_MODELS)
    ap.add_argument("--ar-models", nargs="+", default=AR_MODELS)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = load_rows(a.limit)
    log(f"{len(rows)} sentences -> {OUT}")
    status = {}
    jobs = {
        "grid2":  lambda: e8_grid2(a.base + a.domain_models, a.force),
        "patch":  lambda: e7_patching(a.base, a.force),
        "domain": lambda: e1_domain(rows, a.domain_models, a.force),
        "conf":   lambda: e5_conformal(rows, a.base + a.domain_models, a.force),
        "multi":  lambda: e6_multitoken(rows, a.base, a.force),
        "ar":     lambda: e2_ar(rows, a.ar_models, a.base, a.force),
        "attn":   lambda: e4_attention(rows, a.base, a.force),
        "aoyama": lambda: e3_aoyama(rows, a.base, min(a.aoyama_per_domain, a.limit or 50), a.force),
    }
    for p in a.parts:
        t0 = time.time()
        log(f"\n===== {p} =====")
        try:
            jobs[p]()
            status[p] = f"ok ({time.time() - t0:.0f}s)"
        except Exception as e:
            traceback.print_exc()
            status[p] = f"FAILED: {type(e).__name__}: {e}"
        log(f"===== {p}: {status[p]}")

    mp = path("x2_status.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    old.update({k: v for k, v in status.items()})
    old["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    json.dump(old, open(mp, "w"), indent=2)
    log("\nSTATUS:")
    for k, v in status.items():
        log(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
