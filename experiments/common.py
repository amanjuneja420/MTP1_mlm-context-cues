"""
Shared library for the MTP-1 layer-wise MLM analysis.

Model: bert-base-uncased.

Everything downstream imports from here so that the measurement is defined once.

Key facts about bert-base-uncased used throughout:
  - 12 layers, 12 heads, hidden size 768, vocab 30,522
  - output_hidden_states=True yields 13 tensors: index 0 = embedding output
    (post positional + LayerNorm, pre layer 1), indices 1..12 = layer outputs
  - the MLM head is model.cls (BertOnlyMLMHead):
        dense(768->768) -> GELU -> LayerNorm -> decoder(768->30522) + bias
    The decoder weight is TIED to the input embedding matrix, which is what makes
    the logit lens meaningful: a logit is <transformed hidden state, word embedding>.
  - VERIFIED: model.cls(hidden_states[12]) == model(...).logits  (atol 1e-4)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = "bert-base-uncased"
N_LAYERS = 12

# All case-study outputs (C1-C8, A1) are written to results/case_study/.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "results", "case_study")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

_tok = None
_model = None


def load():
    """
    Load tokenizer + model once and cache.

    attn_implementation="eager" is REQUIRED: the default SDPA kernel fuses the
    softmax and silently returns no attention tensors under
    output_attentions=True, which surfaces as an empty tuple rather than an error.
    Eager is slower but these are 13-token forward passes on CPU.
    """
    global _tok, _model
    if _model is None:
        _tok = BertTokenizer.from_pretrained(MODEL_NAME)
        _model = BertForMaskedLM.from_pretrained(
            MODEL_NAME, attn_implementation="eager")
        _model.eval()
    return _tok, _model


def out_path(name: str) -> str:
    return os.path.join(OUT_DIR, name)


# --------------------------------------------------------------------------- core

def _encode(sentence: str):
    """Tokenize and locate the single [MASK]. Returns (inputs, mask_index)."""
    tok, _ = load()
    if "[MASK]" not in sentence:
        raise ValueError(f"no [MASK] in: {sentence!r}")
    inputs = tok(sentence, return_tensors="pt")
    idx = torch.where(inputs["input_ids"][0] == tok.mask_token_id)[0]
    if len(idx) != 1:
        raise ValueError(f"expected exactly 1 [MASK], found {len(idx)} in {sentence!r}")
    return inputs, int(idx[0])


def tokens_of(sentence: str) -> list[str]:
    tok, _ = load()
    inputs, _ = _encode(sentence)
    return tok.convert_ids_to_tokens(inputs["input_ids"][0])


def forward(sentence: str, attentions: bool = False):
    """Run the model. Returns (outputs, mask_index)."""
    _, model = load()
    inputs, mi = _encode(sentence)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=attentions)
    return out, mi


def top_k(sentence: str, k: int = 10) -> pd.DataFrame:
    """Final-layer top-k predictions at the mask. The C1 instrument."""
    tok, _ = load()
    out, mi = forward(sentence)
    logits = out.logits[0, mi]
    probs = torch.softmax(logits, dim=-1)
    top = logits.topk(k)
    return pd.DataFrame({
        "rank": range(1, k + 1),
        "token": [tok.convert_ids_to_tokens(int(i)) for i in top.indices],
        "token_id": [int(i) for i in top.indices],
        "logit": [float(v) for v in top.values],
        "prob": [float(probs[i]) for i in top.indices],
    })


# ---------------------------------------------------------------------- logit lens

def layerwise_logits(sentence: str) -> torch.Tensor:
    """
    Apply the MLM head to EVERY hidden state at the mask position.

    Returns a tensor of shape (13, vocab_size): row L is the vocabulary
    distribution the model "would have produced" had it stopped after layer L.
    Row 0 is the embedding layer; row 12 reproduces the model's real output.
    """
    _, model = load()
    out, mi = forward(sentence)
    stack = torch.stack([h[0, mi] for h in out.hidden_states])   # (13, 768)
    with torch.no_grad():
        return model.cls(stack)                                   # (13, vocab)


def layerwise_topk(sentence: str, k: int = 5) -> pd.DataFrame:
    """Top-k at every layer. Long format: layer, rank, token, logit, prob."""
    tok, _ = load()
    lg = layerwise_logits(sentence)
    rows = []
    for L in range(lg.shape[0]):
        probs = torch.softmax(lg[L], dim=-1)
        top = lg[L].topk(k)
        for r, (v, i) in enumerate(zip(top.values, top.indices), start=1):
            rows.append({
                "layer": L,
                "rank": r,
                "token": tok.convert_ids_to_tokens(int(i)),
                "logit": float(v),
                "prob": float(probs[i]),
            })
    return pd.DataFrame(rows)


def track_tokens(sentence: str, probe_tokens: list[str]) -> pd.DataFrame:
    """
    Rank and probability of specific tokens at every layer.

    This is the direct answer to "layerwise propagation of the final prediction
    word in place of MASK": pass the final-layer winner as a probe and watch its
    rank climb.
    """
    tok, _ = load()
    lg = layerwise_logits(sentence)
    ids = {t: tok.convert_tokens_to_ids(t) for t in probe_tokens}

    rows = []
    for L in range(lg.shape[0]):
        probs = torch.softmax(lg[L], dim=-1)
        order = torch.argsort(lg[L], descending=True)
        # position of each vocab id in the sorted order
        rank_of = torch.empty_like(order)
        rank_of[order] = torch.arange(len(order))
        for t, i in ids.items():
            if i == tok.unk_token_id:
                continue
            rows.append({
                "layer": L,
                "token": t,
                "rank": int(rank_of[i]) + 1,
                "logit": float(lg[L][i]),
                "prob": float(probs[i]),
            })
    return pd.DataFrame(rows)


def final_winner(sentence: str) -> str:
    tok, _ = load()
    out, mi = forward(sentence)
    return tok.convert_ids_to_tokens(int(out.logits[0, mi].argmax()))


# ----------------------------------------------------------------------- attention

def attention_at_mask(sentence: str) -> np.ndarray:
    """
    Attention paid BY the mask position TO every token.
    Returns array of shape (layers=12, heads=12, seq_len).
    """
    out, mi = forward(sentence, attentions=True)
    # out.attentions: tuple of 12 tensors, each (batch, heads, q_pos, k_pos)
    return np.stack([a[0, :, mi, :].numpy() for a in out.attentions])


# ----------------------------------------------------------------------- occlusion

def occlusion(sentence: str, probe_tokens: list[str] | None = None) -> pd.DataFrame:
    """
    Leave-one-out over context words: replace each context token with [MASK] in
    turn and measure how the prediction at the ORIGINAL mask changes.

    This is the causal version of "impact of neighbouring words". Methodologically
    it follows Perturbed Masking (Wu et al., ACL 2020), applied to the output
    distribution rather than to a syntax tree.

    Columns: occluded token, its position, the new winner, KL from the
    unperturbed distribution, and rank/prob deltas for each probe token.
    """
    tok, _ = load()
    inputs, mi = _encode(sentence)
    ids = inputs["input_ids"][0]
    toks = tok.convert_ids_to_tokens(ids)

    base_lg = layerwise_logits(sentence)[N_LAYERS]
    base_p = torch.softmax(base_lg, dim=-1)
    base_winner = tok.convert_ids_to_tokens(int(base_lg.argmax()))

    probe_tokens = probe_tokens or []
    probe_ids = {t: tok.convert_tokens_to_ids(t) for t in probe_tokens}

    def rank_prob(logits, vid):
        order = torch.argsort(logits, descending=True)
        rank_of = torch.empty_like(order)
        rank_of[order] = torch.arange(len(order))
        return int(rank_of[vid]) + 1, float(torch.softmax(logits, dim=-1)[vid])

    base_rp = {t: rank_prob(base_lg, i) for t, i in probe_ids.items()}

    _, model = load()
    rows = []
    for pos, t in enumerate(toks):
        if pos == mi or t in ("[CLS]", "[SEP]"):
            continue
        pert = ids.clone()
        pert[pos] = tok.mask_token_id
        with torch.no_grad():
            lg = model(input_ids=pert.unsqueeze(0)).logits[0, mi]
        p = torch.softmax(lg, dim=-1)
        # KL(base || perturbed) - how much the distribution moved
        kl = float(torch.sum(base_p * (torch.log(base_p + 1e-12) - torch.log(p + 1e-12))))

        row = {
            "occluded": t,
            "position": pos,
            "new_winner": tok.convert_ids_to_tokens(int(lg.argmax())),
            "winner_changed": tok.convert_ids_to_tokens(int(lg.argmax())) != base_winner,
            "kl_from_base": kl,
        }
        for pt, vid in probe_ids.items():
            r, pr = rank_prob(lg, vid)
            row[f"{pt}_rank"] = r
            row[f"{pt}_prob"] = pr
            row[f"{pt}_rank_delta"] = r - base_rp[pt][0]
            row[f"{pt}_prob_delta"] = pr - base_rp[pt][1]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("kl_from_base", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ sentence sets

FRAME = "the {subj} is {verb} [MASK] since the last 2 years ."

# The original patient/person pair, plus a spread of subjects that vary how strongly
# they cue a domain.
SUBJECTS = ["patient", "person", "man", "woman", "child", "student",
            "doctor", "athlete", "farmer", "musician"]

VERBS = ["taking", "eating", "using", "drinking", "buying", "selling",
         "studying", "playing", "wearing", "reading"]

# Same syntactic frame across domains, so layer-wise behaviour is comparable.
DOMAIN_SET = {
    "medical":   ("the patient is taking [MASK] since the last 2 years .",   ["medication", "drugs", "medicine"]),
    "generic":   ("the person is taking [MASK] since the last 2 years .",    ["medication", "drugs", "care"]),
    "food":      ("the person is eating [MASK] since the last 2 years .",    ["meat", "food", "less"]),
    "music":     ("the musician is playing [MASK] since the last 2 years .", ["guitar", "piano", "music"]),
    "sport":     ("the athlete is playing [MASK] since the last 2 years .",  ["football", "soccer", "tennis"]),
    "education": ("the student is studying [MASK] since the last 2 years .", ["english", "law", "medicine"]),
    "finance":   ("the investor is buying [MASK] since the last 2 years .",  ["stocks", "shares", "gold"]),
    "tech":      ("the programmer is writing [MASK] since the last 2 years .", ["code", "software", "books"]),
    "farming":   ("the farmer is growing [MASK] since the last 2 years .",   ["corn", "wheat", "vegetables"]),
    "legal":     ("the lawyer is filing [MASK] since the last 2 years .",    ["lawsuits", "suits", "papers"]),
}

# Mask-position variants of the original sentence: which slot is hidden matters.
MASK_POSITION_SET = {
    "object":  "the patient is taking [MASK] since the last 2 years .",
    "subject": "the [MASK] is taking medication since the last 2 years .",
    "verb":    "the patient is [MASK] medication since the last 2 years .",
    "duration": "the patient is taking medication since the last [MASK] years .",
}


def subject_sweep(verb: str = "taking") -> dict[str, str]:
    return {s: FRAME.format(subj=s, verb=verb) for s in SUBJECTS}


def verb_sweep(subj: str = "person") -> dict[str, str]:
    return {v: FRAME.format(subj=subj, verb=v) for v in VERBS}


# --------------------------------------------------------------------- formatting

def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Markdown table without needing the `tabulate` dependency."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v))
        else:
            d[c] = d[c].astype(str)
    header = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(r) + " |" for r in d.values.tolist()]
    return "\n".join([header, sep] + body)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
