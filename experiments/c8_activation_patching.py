"""
C8 — ACTIVATION PATCHING / CAUSAL TRACING

Fixes the interpretive limit of C6. Occlusion changes the INPUT, so every layer of
the corrupted run differs from the clean run — C6 could only say when a divergence
becomes *readable*, not where influence *originates*.

Patching fixes that by intervening on ONE (layer, position) at a time:

  1. Run the CLEAN input, cache all 13 hidden states.
  2. Run the CORRUPTED input, but at layer L, position p, overwrite the hidden
     state with the CLEAN one. Let the rest of the forward pass proceed normally.
  3. Measure how much of the clean prediction is restored.

Because only one site is altered, a recovery is attributable to THAT site.
Method follows Meng et al. 2022 (ROME, arXiv:2202.05262), adapted to a masked LM.

Indexing note (easy to get wrong):
    hidden_states[0]  = embedding output      -> patch via model.bert.embeddings
    hidden_states[L]  = output of encoder layer L-1 (L = 1..12)
                                              -> patch via model.bert.encoder.layer[L-1]

Metric — normalised restoration:
    r(L,p) = (p_patched - p_corrupt) / (p_clean - p_corrupt)
    r = 0  -> patching changed nothing
    r = 1  -> patching fully restored the clean prediction

Outputs:
  c8_patching_<setup>.csv    restoration for every (layer, position)
  c8_patching.png            heatmaps
  c8_summary.csv             where restoration peaks, per setup
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common as C


# Each setup: (clean sentence, corrupted sentence, probe token).
# The two sentences MUST tokenize to the same length so positions align.
#
# PRIMARY setups are COUNTERFACTUAL: both runs have a real content word at the
# patched position, so patching swaps one meaning for another. These are the ones
# the causal claims rest on.
#
# The [MASK]-corruption setups are kept for COMPARISON only. There the corrupted
# run has no lexical content at the patched position, so patching injects content
# the run never had rather than swapping it. Their layers 1-3 values are therefore
# not cleanly interpretable as "restoring what the model already computed".
SETUPS = {
    # --- primary (counterfactual) ---
    "patient_to_athlete": (
        "the patient is taking [MASK] since the last 2 years .",
        "the athlete is taking [MASK] since the last 2 years .",
        "medication",
    ),
    "patient_to_machine": (
        "the patient is taking [MASK] since the last 2 years .",
        "the machine is taking [MASK] since the last 2 years .",
        "medication",
    ),
    "taking_to_eating": (
        "the patient is taking [MASK] since the last 2 years .",
        "the patient is eating [MASK] since the last 2 years .",
        "medication",
    ),
    # --- comparison only (mask-corruption; see note above) ---
    "occlude_subject_COMPARISON": (
        "the patient is taking [MASK] since the last 2 years .",
        "the [MASK] is taking [MASK] since the last 2 years .",
        "medication",
    ),
    "occlude_verb_COMPARISON": (
        "the patient is taking [MASK] since the last 2 years .",
        "the patient is [MASK] [MASK] since the last 2 years .",
        "medication",
    ),
}


def encode(sentence, tok):
    return tok(sentence, return_tensors="pt")["input_ids"]


def target_mask_index(ids, tok, sentence):
    """
    The mask we READ from is the one in the object slot. In corrupted inputs there
    may be two [MASK] tokens, so take the position of the [MASK] that also appears
    in the clean sentence (the object), identified by its index in the clean input.
    """
    positions = torch.where(ids[0] == tok.mask_token_id)[0].tolist()
    return positions


def run_clean(ids):
    _, model = C.load()
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    return out


def patched_prob(clean_hs, corrupt_ids, layer, position, read_index, probe_id):
    """
    Run the corrupted input with hidden_states[layer][position] replaced by the
    clean run's value. Returns p(probe) at read_index.
    """
    _, model = C.load()

    def emb_hook(module, args, output):
        output = output.clone()
        output[0, position, :] = clean_hs[0][0, position, :]
        return output

    def layer_hook(module, args, output):
        # transformers 5.x BertLayer returns a bare tensor; older versions a tuple.
        # Handle both, or the patch silently writes into the wrong axis.
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[0, position, :] = clean_hs[layer][0, position, :]
            return (hs,) + tuple(output[1:])
        hs = output.clone()
        hs[0, position, :] = clean_hs[layer][0, position, :]
        return hs

    if layer == 0:
        handle = model.bert.embeddings.register_forward_hook(emb_hook)
    else:
        handle = model.bert.encoder.layer[layer - 1].register_forward_hook(layer_hook)

    try:
        with torch.no_grad():
            logits = model(input_ids=corrupt_ids).logits[0, read_index]
    finally:
        handle.remove()

    return float(torch.softmax(logits, dim=-1)[probe_id])


def trace(name, clean_sent, corrupt_sent, probe):
    tok, model = C.load()

    clean_ids = encode(clean_sent, tok)
    corrupt_ids = encode(corrupt_sent, tok)

    clean_toks = tok.convert_ids_to_tokens(clean_ids[0])
    corrupt_toks = tok.convert_ids_to_tokens(corrupt_ids[0])

    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(f"[{name}] length mismatch: "
                         f"{len(clean_toks)} vs {len(corrupt_toks)} — "
                         f"positions would not align.\n  {clean_toks}\n  {corrupt_toks}")

    # The object mask is the LAST [MASK] in both (subject mask, if any, comes first).
    read_index = max(torch.where(clean_ids[0] == tok.mask_token_id)[0].tolist())
    probe_id = tok.convert_tokens_to_ids(probe)

    clean_out = run_clean(clean_ids)
    clean_hs = clean_out.hidden_states
    p_clean = float(torch.softmax(clean_out.logits[0, read_index], dim=-1)[probe_id])

    with torch.no_grad():
        corr_logits = model(input_ids=corrupt_ids).logits[0, read_index]
    p_corrupt = float(torch.softmax(corr_logits, dim=-1)[probe_id])

    C.banner(f"C8 — {name}")
    print(f"clean    : {clean_sent}")
    print(f"corrupt  : {corrupt_sent}")
    print(f"probe    : '{probe}'   p_clean = {p_clean:.4f}   p_corrupt = {p_corrupt:.4f}")
    print(f"read at position {read_index} ('{clean_toks[read_index]}')\n")

    if abs(p_clean - p_corrupt) < 1e-6:
        print("  corruption had no effect — nothing to restore. Skipping.")
        return pd.DataFrame()

    rows = []
    for layer in range(13):
        for pos, t in enumerate(corrupt_toks):
            p = patched_prob(clean_hs, corrupt_ids, layer, pos, read_index, probe_id)
            rows.append({
                "setup": name, "layer": layer, "position": pos,
                "token_clean": clean_toks[pos], "token_corrupt": corrupt_toks[pos],
                "p_patched": p,
                "restoration": (p - p_corrupt) / (p_clean - p_corrupt),
            })

    df = pd.DataFrame(rows)

    # Print the restoration grid over content positions
    content = [i for i, t in enumerate(corrupt_toks) if t not in ("[CLS]", "[SEP]")]
    hdr = "layer | " + " | ".join(f"{corrupt_toks[i][:8]:>8s}" for i in content)
    print(hdr)
    print("-" * len(hdr))
    for L in range(13):
        s = df[df.layer == L].set_index("position")
        print(f"{L:5d} | " + " | ".join(f"{s.loc[i,'restoration']:8.2f}" for i in content))

    nz = df[df.layer >= 1]
    best = nz.loc[nz.restoration.idxmax()]
    print(f"\n  peak restoration (layers>=1) {best.restoration:.2f} at layer "
          f"{int(best.layer)}, position {int(best.position)} ('{best.token_clean}')")
    print(f"  layer-0 value {df[df.layer==0].restoration.max():.2f} "
          f"(trivial: patching the embedding == un-corrupting the input)")

    # Latest layer at which patching the differing position still restores >= 0.5
    diff_pos = [i for i in range(len(clean_toks))
                if clean_toks[i] != corrupt_toks[i]]
    for dp in diff_pos:
        s = df[(df.position == dp)].sort_values("layer")
        good = s[s.restoration >= 0.5]
        if len(good):
            print(f"  patching '{clean_toks[dp]}' (pos {dp}) restores >=0.5 "
                  f"at layers {int(good.layer.min())}-{int(good.layer.max())}"
                  f"  -> its information must be READ by layer ~{int(good.layer.max())}")
        else:
            print(f"  patching '{clean_toks[dp]}' (pos {dp}) never restores >=0.5")

    return df


def main():
    C.load()
    frames, summary = [], []

    for name, (clean, corrupt, probe) in SETUPS.items():
        try:
            df = trace(name, clean, corrupt, probe)
        except ValueError as e:
            print(f"\nSKIPPED {name}: {e}")
            continue
        if df.empty:
            continue
        frames.append(df)
        df.to_csv(C.out_path(f"c8_patching_{name}.csv"), index=False)

        # Layer 0 is the EMBEDDING output: patching it is identical to un-corrupting
        # the input, so it always yields r=1.0 and is not a finding. Report the peak
        # over layers >= 1, and keep the layer-0 value separately as a wiring check.
        nz = df[df.layer >= 1]
        best = nz.loc[nz.restoration.idxmax()]
        l0 = df[df.layer == 0].restoration.max()
        summary.append({
            "setup": name,
            "peak_restoration_excl_L0": float(best.restoration),
            "peak_layer": int(best.layer),
            "peak_position": int(best.position),
            "peak_token": best.token_clean,
            "L0_sanity_check": float(l0),   # expected ~1.0; trivial, not a result
        })

    if not frames:
        print("\nNo setups produced usable traces.")
        return

    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(C.out_path("c8_patching_all.csv"), index=False)
    pd.DataFrame(summary).to_csv(C.out_path("c8_summary.csv"), index=False)

    # ---------------------------------------------------------------- heatmaps
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    for ax, df in zip(np.atleast_1d(axes), frames):
        name = df.setup.iloc[0]
        toks = df[df.layer == 0].sort_values("position").token_corrupt.tolist()
        grid = df.pivot(index="layer", columns="position", values="restoration").values
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(13))
        ax.set_yticklabels(range(13), fontsize=7)
        ax.set_ylabel("layer patched")
        ax.set_title(name, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, label="restoration")
    fig.suptitle("C8 — activation patching: restoring one (layer, position) at a time",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(C.out_path("c8_patching.png"), dpi=150, bbox_inches="tight")

    C.banner("C8 — summary")
    print(C.df_to_md(pd.DataFrame(summary)))
    print("\nWrote c8_*.csv and c8_patching.png")


if __name__ == "__main__":
    main()
