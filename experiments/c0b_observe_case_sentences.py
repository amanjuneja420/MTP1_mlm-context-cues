"""
C0b - Observation run on the case-study sentences (no analysis).

Same logic as c0_reference_snippet.py; only the input sentences change:
"The patient/person is taking <mask> since the last 2 years." and a food control.
This reproduces the observation the project starts from. The reusable top-k
tool used by the later experiments is c1_topk_baseline.py.

Run (from the repository root):
    python experiments/c0b_observe_case_sentences.py
"""

import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.eval()

CASES = [
    ("The patient is taking medicine since the last 2 years.", "medicine"),
    ("The person is taking medicine since the last 2 years.",  "medicine"),
    ("The person is eating food since the last 2 years.",      "food"),
]

for sentence, target in CASES:
    masked_sentence = sentence.replace(target, tokenizer.mask_token)
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        logits = model(input_ids).logits
        mask_logits = logits[0, mask_token_index].squeeze().cpu().numpy()

    top_k = 10
    top_k_indices = mask_logits.argsort()[-top_k:][::-1]
    probs = torch.nn.functional.softmax(torch.tensor(mask_logits), dim=0)

    print("=" * 68)
    print(f"Sentence: {masked_sentence}")
    print(f"Top {top_k} candidates:")
    for i, idx in enumerate(top_k_indices):
        token_str = tokenizer.decode([idx])
        print(f"{i+1:2}. '{token_str}'".ljust(24)
              + f"logit {mask_logits[idx]:7.4f}   p {probs[idx].item():.4f}")
    print()
