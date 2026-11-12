# Context and Domain Cues in Masked-Word Prediction

An interpretability study of BERT and RoBERTa: which words in a sentence make a masked
language model predict a domain-specific word, where in the network that prediction is
formed, and why.

Masters Thesis Project (MTP-1), Department of Artificial Intelligence, IIT Kharagpur, 2026.
**Author:** Aman Juneja  **Supervisor:** Prof. Koustav Rudra

---

## Contents

1. [The question](#the-question)
2. [Main findings](#main-findings)
3. [Quick start](#quick-start)
4. [Repository layout](#repository-layout)
5. [How files are named](#how-files-are-named)
6. [Data](#data)
7. [Models](#models)
8. [Experiments](#experiments)
9. [Analysis scripts](#analysis-scripts)
10. [Key terms](#key-terms)
11. [Main output files](#main-output-files)
12. [Reproducing everything](#reproducing-everything)
13. [Limitations](#limitations)
14. [Acknowledgements, citation, license](#acknowledgements)

---

## The question

Give vanilla RoBERTa-base (no fine-tuning) the sentence

> *The **patient** is taking `<mask>` since the last 2 years.*

and its top predictions are *medication* (p = 0.236), *drugs*, *insulin*, … all medical, as
expected. Change one word:

> *The **person** is taking `<mask>` since the last 2 years.*

Nothing in this sentence is about medicine, yet the model still ranks *drugs* (0.148) and
*medication* (0.117) first. The project asks four questions:

| | Question |
|---|---|
| **RQ1** | Which context words drive the prediction, and how should a word's influence be measured? |
| **RQ2** | Where in the network is the prediction formed? |
| **RQ3** | Is the behaviour general across BERT/RoBERTa, base/large, and a 500-sentence, 10-domain collection? |
| **RQ4** | Why does it happen: the masked-LM architecture, or the text the model was trained on? |

## Main findings

Numbers are for BERT-base unless stated otherwise; every number can be traced to a file in
`results/` (see [Experiments](#experiments)).

- **"Taking" is a light verb whose object is chosen by the subject** (`c1`, `c7`).
  *athlete* → *part*, *student* → *classes*, *child* → *care*, *patient* → *medication* (0.43).
  With no subject at all the model predicts *place* (0.80) and *medication* falls to rank 376,
  so *medication* is selected, not a default.
- **The time phrase turns "taking X" into a dosage frame** (`x1` part C).
  For *person* with no time phrase, *medication* is at rank 20 (p = 0.008); adding
  *for two days* moves it to rank 1 (p = 0.52), while *right now* leaves it at rank 27.
  A duration raises the medical probability for 12–13 of the 13 person/animal subjects in all
  four models (Wilcoxon signed-rank, p < 0.001; `grid_stats.py`). The effect needs the progressive/perfect form (*is / has been taking*);
  *takes* and *took* barely show it, and adding *not* does not remove it (`e8`).
- **The answer is assembled late, at the mask position** (`c3`, `c8`, `e7`, `x1` part A).
  The correct word becomes the top prediction only at 81–86% of the network's depth in all four
  models. Activation patching shows that the subject's information leaves its own position early
  and that patching the mask position restores the prediction only from about 71–92% of the depth
  onward (range over the four models and three corruptions).
- **On 500 sentences, wrong guesses follow the domain and word frequency** (`x1`, `t6`, `t9`).
  The target is in the top 10 for 50.4% (BERT-base) to 69.6% (RoBERTa-large) of sentences.
  36–42% of wrong top-1 guesses are still from the sentence's domain (about 10% expected by
  chance), and 78–82% of them are more frequent words than the correct answer.
- **Masking a word to measure its influence hides part of its effect** (`x1` part B).
  Replacing a word by `[MASK]` lets the model guess it back, so masking shows roughly half the
  effect of deleting the word, and the two methods agree on the most important word in only
  about 40% of sentences.
- **The preference comes from the training text** (`e1`).
  Same architecture, different training data: BiomedBERT reads even *dog is taking* medically
  and reaches hit@10 = 0.80 on the medicine sentences (BERT-base: 0.51); Legal-BERT reaches
  0.91 on law (BERT-base: 0.84).

## Quick start

```bash
git clone https://github.com/amanjuneja420/MTP1_mlm-context-cues.git
cd MTP1_mlm-context-cues
pip install -r requirements.txt

# Rebuild the tables and figures from the saved results (no models needed):
python analysis/analyze_x1.py results/x1_crossmodel results/x1_crossmodel/analysis
python analysis/analyze_x2.py results/x2_extended results/x1_crossmodel results/x2_extended/analysis
python analysis/make_figures.py

# Re-run the starting observation (downloads roberta-base once, about 500 MB):
python experiments/c0b_observe_case_sentences.py
```

## Repository layout

```
data/          sentences.csv     the 500-sentence collection (10 domains x 50)
experiments/   scripts that run the models and write raw results to results/
  common.py        shared helpers for the case study (model loading, top-k, logit lens, output paths)
  dataset.py       loads data/sentences.csv and marks single-token targets
analysis/      scripts that turn raw results into tables and figures
  semantic.py      helper for semantic_domains.py (sentence embeddings)
results/       raw outputs and derived tables, exactly as used in the project
  case_study/      outputs of c1-c8 and a1
  x1_crossmodel/   outputs of x1_crossmodel.py   (+ analysis/: tables t1-t9, figures f1-f2)
  x2_extended/     outputs of x2_extended.py     (+ analysis/: tables t_e1-t_e8, figures f_e*)
figures/       the main figures, redrawn by analysis/make_figures.py
```

## How files are named

Every script and every output file starts with a short prefix that says which experiment it
belongs to. The prefixes follow the order in which the work was done.

| Prefix | Meaning | Where |
|---|---|---|
| `c0`–`c8` | **Case study.** Experiments on one sentence family, *"The patient/person is taking [MASK] since the last 2 years"*, numbered in the order they were run (`c0` = the starting observation). BERT-base unless stated. | `experiments/c*.py` → `results/case_study/c*_…` |
| `a1` | **Aoyama replication.** Re-implementation of Aoyama & Schneider (2022), *probe-less probing*, checked on the paper's own example. | `experiments/a1_…py` → `results/case_study/a1_…` |
| `x1` | **Cross-model run.** The 500-sentence collection and the time-phrase grid on four models. | `experiments/x1_crossmodel.py` → `results/x1_crossmodel/x1_…` |
| `x2` / `e1`–`e8` | **Extended suite.** `x2_extended.py` has eight parts, E1–E8, one per paper from the reading list. Part E*n* writes `e`*n*`_…` files. | `results/x2_extended/e*_…` |
| `t…`, `t_e…` | **Tables** written by the analysis scripts (`t1`–`t9` from x1, `t_e1`–`t_e8` from x2). | `results/*/analysis/` |
| `f…` | **Figures** written by the analysis scripts. | `results/*/analysis/` |

Two details that can look odd:

- **E1 writes `x1_…` files, not `e1_…` files.** E1 runs the domain-adapted BERTs through the same
  code as `x1_crossmodel.py`, so its outputs keep that format and are saved in
  `results/x2_extended/` (for example `x1_final_legal-bert-base-uncased.csv`). Its summary table
  is `t_e1_…`.
- **E9 has no script in `experiments/`.** The word-frequency analysis needs no model run; it is
  `analysis/frequency.py`, which writes `e9_freq_<model>.csv` and `t9_frequency.csv`.

Model names in file names are the Hugging Face names without the organisation, e.g.
`x1_final_roberta-large.csv` or `e8_grid2_Bio_ClinicalBERT.csv`.

## Data

`data/sentences.csv` holds 500 English sentences, 50 from each of 10 domains: General English,
Medicine, Law, Finance, Climate Science, Cybersecurity, Education, Biology, Technology and
Geography. Each sentence has one target word, usually a domain term, replaced by `<mask>`:

| domain | no | original_sentence | target | masked_sentence |
|---|---|---|---|---|
| GENERAL ENGLISH | 1 | Maya left her [TARGET: umbrella] beside the apartment door. | umbrella | Maya left her `<mask>` beside the apartment door. |

`experiments/dataset.py` loads the file, converts `<mask>` to `[MASK]` for BERT, and records
whether the target is a single token for each tokenizer (403 targets for BERT, 419 for RoBERTa).
Several results are reported separately for single-token targets, because a model can only
predict one token per mask.

The case-study sentences (`c*` experiments) and the time-phrase grid are written in the scripts
themselves (`experiments/common.py`, `experiments/x1_crossmodel.py`).

## Models

All models are used exactly as released on the Hugging Face Hub, without fine-tuning.

| Role | Models |
|---|---|
| Main subjects | `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large` |
| Domain-adapted controls (same architecture as BERT-base, different training text) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, `emilyalsentzer/Bio_ClinicalBERT`, `nlpaueb/legal-bert-base-uncased` |
| Exploratory left-to-right control | `gpt2`, `gpt2-medium` (part E2, see below) |

## Experiments

### Case study → `results/case_study/`

| Script | Question it answers | Main outputs |
|---|---|---|
| `c0_reference_snippet.py` | Reference top-10 snippet for roberta-base, kept verbatim | printed only |
| `c0b_observe_case_sentences.py` | The starting observation: patient vs person (RoBERTa-base) | printed only |
| `c1_topk_baseline.py` | What does BERT-base predict across 10 domain frames, and as the subject or verb changes? | `c1_topk_*.csv` |
| `c2_minimal_pairs.py` | How much does changing one word (subject, verb, masked slot, time phrase, negation) move the prediction? | `c2_*.csv` |
| `c3_logit_lens.py` | At which layer does each candidate word appear? (logit lens) | `c3_*.csv` |
| `c4_attention_at_mask.py` | Which words does the mask attend to, per layer and head? *Exploratory: attention barely differed between patient and person (0.091 vs 0.087); the four-model attention analysis is E4.* | `c4_*.csv` |
| `c5_occlusion.py` | How much does the output change when each word is replaced by `[MASK]`? *This first attribution method rated "years" as unimportant; x1 part B shows why masking under-states such words.* | `c5_*.csv` |
| `c6_layerwise_occlusion.py` | At which layer does the effect of removing a word become readable? *Exploratory, superseded by C8: occlusion changes the input, so it cannot show where influence originates.* | `c6_*.csv` |
| `c7_subject_ladder.py` | Is *medication* selected by the subject, or the default object of *taking*? | `c7_*.csv` |
| `c8_activation_patching.py` | Where is the information used? Clean states are copied into a corrupted run one (layer, position) at a time. | `c8_patching_*.csv`, `c8_summary.csv` |

The counterfactual patching setups (`patient_to_athlete`, `patient_to_machine`,
`taking_to_eating`) are the ones interpreted. The `c8_patching_occlude_*_COMPARISON` files use a
`[MASK]`-corrupted run and are kept for comparison only: at early layers, patching into a masked
position injects content the corrupted run never had, so they are not read causally.

### Aoyama & Schneider (2022) replication → `results/case_study/a1_*`

`a1_aoyama_replication.py` masks every word of a sentence, reads a prediction from every layer,
and scores whether the predicted word (*word match*) or its part of speech (*POS match*, tagged
with Stanza) equals the original. `--demo` runs the paper's own example sentence: the layer-12
top-10 list matches the paper's Table 2 token for token. Without `--demo` it runs a 10-sentence
pilot. The four-model version on 50 sentences (5 per domain) is part E3 of `x2_extended.py`.

### 500 sentences across four models → `results/x1_crossmodel/`

| Part of `x1_crossmodel.py` | What it does |
|---|---|
| A | For every sentence: target rank and probability, top-10, and the logit-lens readout at every layer |
| B | Context attribution: each context word is removed by **deleting** it and, separately, by **masking** it |
| C | Time-phrase grid: 15 subjects × 10 time phrases in *"The ___ is taking [MASK] …"* (the 15th subject, `none`, has no subject at all) |

### Extended suite → `results/x2_extended/`

| Part (`--parts`) | Id | What it does | Based on |
|---|---|---|---|
| `domain` | E1 | Parts A and C of `x1_crossmodel.py` on the three domain-adapted BERTs | Vakili et al. 2022; Nozza et al. 2020 |
| `ar` | E2 | GPT-2 / GPT-2-medium as a left-to-right control: each candidate is put into the blank and the whole sentence is scored. **Exploratory; not part of the MTP-1 results; planned for MTP-2.** | Bavarian et al. 2022 (FIM); MARIA 2026 |
| `aoyama` | E3 | Probe-less probing (word and POS match by layer) on 50 sentences (5 per domain, 514 masked positions), four models | Aoyama & Schneider 2022 |
| `attn` | E4 | Where the mask attends at each layer ([CLS], [SEP], itself, neighbours, punctuation) and attention entropy | Clark et al. 2019 |
| `conf` | E5 | Saves each model's sorted output distribution for conformal prediction sets | Dey et al. 2022 |
| `multi` | E6 | Multi-token targets: k masks for a k-piece word; parallel, left-to-right and confidence-first filling | Kalinsky et al. 2023 |
| `patch` | E7 | Activation patching on all four models: subject (patient → machine), time (years → minutes), verb (taking → eating) | Meng et al. 2022 |
| `grid2` | E8 | Verb frames (*is taking, is not taking, has been taking, takes, took, is using, is on*) × subjects × time phrases | Ettinger 2020 |

## Analysis scripts

| Script | Produces | Content |
|---|---|---|
| `analyze_x1.py` | `t1`–`t5`, `ctx_support_*`, `f1_layerwise.png`, `f2_grid.png` | accuracy, per-domain results, layer-wise emergence, delete-vs-mask attribution, time-phrase grid |
| `grid_stats.py` | `t5b_grid_wilcoxon.csv` | is the time-phrase effect significant? Duration vs no time phrase and vs momentary phrases, Wilcoxon signed-rank over the 13 person/animal subjects (`none` and `machine` excluded) |
| `semantic_domains.py` (uses `semantic.py`) | `t6_semantic.csv` | are wrong guesses from the sentence's domain? Words and domain names are embedded with all-MiniLM-L6-v2; baseline: shuffled domains |
| `frequency.py` | `e9_freq_*`, `t9_frequency.csv` | are wrong guesses more frequent words? Zipf frequency from `wordfreq` |
| `analyze_x2.py` | `t_e1`–`t_e8`, `f_e*.png` | summary tables for E1–E8, including the conformal prediction sets |
| `make_figures.py` | `figures/*.png` | the ten main figures |

## Key terms

| Term | Meaning here |
|---|---|
| **Masked language model (MLM)** | A model trained to fill in a hidden (`[MASK]`) word from the words on both sides; BERT and RoBERTa are MLMs. |
| **hit@k** | The correct word is among the model's top k predictions. |
| **Medical mass** | Total probability the model gives to a fixed list of 12 medical words (*medication, medications, drugs, medicine, medicines, pills, antibiotics, insulin, treatment, antidepressants, painkillers, steroids*). |
| **Logit lens** | Applying the model's own output layer to an intermediate layer, to read what the model would predict if it stopped there. |
| **Relative depth** | Layer number divided by the number of layers, so 12- and 24-layer models can be compared (1.0 = last layer). |
| **Expected layer** | Aoyama & Schneider's summary of *when* a score is gained: the average layer, weighted by how much the score improves at each layer. |
| **Activation patching** | Run the model on a clean and a corrupted sentence; copy one hidden state from the clean run into the corrupted run and measure how much of the clean prediction comes back (**restoration**: 0 = nothing, 1 = fully restored; single-site patches can overshoot above 1). |
| **Delete vs mask attribution** | Two ways to measure a word's influence: remove it from the sentence, or replace it with `[MASK]`. Influence (**support**) = drop in the log-probability of the correct word. |
| **Zipf frequency** | log10 of a word's frequency per billion words (about 7.7 for *the*; below 3 for rare words). |
| **Conformal prediction set** | The smallest set of candidate words that contains the correct word with a chosen probability (e.g. 90%), calibrated on held-out sentences. |

## Main output files

| File | One row per | Key columns |
|---|---|---|
| `x1_final_<model>.csv` | sentence | `target`, `target_rank`, `target_p`, `hit1`, `hit10`, `top1`, `top10`, `single_token` |
| `x1_layers_<model>.csv` | sentence × layer | `layer`, `rel_depth`, `top1`, `top1_is_target`, `target_rank` |
| `x1_context_<model>.csv` | sentence × context word × method | `word`, `method` (`delete`/`mask`), `delta_logp`, `kl_base_new` |
| `x1_grid_<model>.csv` | subject × time phrase | `medication_rank`, `medication_p`, `medical_mass`, `top5` |
| `e7_patching_<model>.csv` | setup × layer × position | `rel_depth`, `is_mask`, `differs`, `restoration` |
| `e8_grid2_<model>.csv` | frame × subject × time phrase | `frame`, `medical_mass`, `top5` |
| `e5_dist_<model>.npz` | sentence | `sorted_probs` (top-5000 output probabilities), `target_p`, `domain`, `single` |

The `.npz` files (about 9 MB each) let the conformal-prediction tables be recomputed without
running the models.

## Reproducing everything

Python 3.12 or newer. The saved results were produced on CPU with `torch 2.13.0` and
`transformers 5.14.1`. Approximate CPU times from the original runs are given.

```bash
pip install -r requirements.txt

# 1. Models (downloads each model once from the Hugging Face Hub)
python experiments/c0_reference_snippet.py         # c0, c0b, c1-c7: under a minute each
python experiments/c8_activation_patching.py       # about 1-2 min
python experiments/a1_aoyama_replication.py --demo # needs the Stanza English model
python experiments/x1_crossmodel.py                # about 45 min for the four models
python experiments/x2_extended.py                  # about 1.5 h for all eight parts

# 2. Tables and figures
python analysis/analyze_x1.py results/x1_crossmodel results/x1_crossmodel/analysis
python analysis/grid_stats.py results/x1_crossmodel results/x1_crossmodel/analysis
python analysis/semantic_domains.py results/x1_crossmodel results/x1_crossmodel/analysis
python analysis/frequency.py results/x1_crossmodel results/x1_crossmodel/analysis
python analysis/analyze_x2.py results/x2_extended results/x1_crossmodel results/x2_extended/analysis
python analysis/make_figures.py
```

- Scripts skip outputs that already exist; add `--force` to `x1_crossmodel.py` or
  `x2_extended.py` to recompute. `--limit N` runs N sentences per domain as a quick test, and
  `--models` / `--parts` select a subset.
- Forward passes are deterministic. Re-running on another machine reproduces the saved numbers
  up to floating-point noise (differences below 1e-4; ranks of near-tied, very unlikely words can
  shift by one). The only random step, the calibration/test split for conformal prediction, uses
  a fixed seed.
- Attention weights require `attn_implementation="eager"` in recent `transformers` versions; the
  scripts set it.

## Limitations

- The case study is built on one construction (*"is taking [MASK]"*); other light verbs
  (*have, make, give*) are not yet tested.
- Each patching setup uses one sentence pair, not an average over many.
- The logit lens is unreliable in early layers, especially in the 24-layer models, so early-layer
  readouts are not interpreted on their own; patching is used for causal claims.
- Word frequency comes from a general resource (`wordfreq`), not from the models' own training data.

## Acknowledgements

The 500-sentence collection and the reference snippet (`c0_reference_snippet.py`) were provided
by the project mentor, Subham Kumar.

## Citation

If you use this code, please cite it as described in [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)).
