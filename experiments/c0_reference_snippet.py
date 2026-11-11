"""
C0 - Reference snippet provided by the project mentor, kept verbatim.

Prints roberta-base's top-10 fills for one masked sentence. It is the known-good
baseline: if a later script disagrees with it, the later script has the bug.

Run (from the repository root):
    python experiments/c0_reference_snippet.py

The first run downloads roberta-base (about 500 MB) from the Hugging Face Hub.
"""

# Reference code for getting the top 10 predictions from roberta-base:
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM

# 1. Setup Model and Tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.eval()

sentence = "The person is eating food since the last 2 years."
target = "food"

# 2. Replace target with mask and tokenize
masked_sentence = sentence.replace(target, tokenizer.mask_token)
# masked_sentence = sentence.replace(target, "<mask>")
inputs = tokenizer(masked_sentence, return_tensors="pt")
input_ids = inputs["input_ids"]

# 3. Find mask position
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 4. Get logits using the requested approach
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits
    mask_logits = logits[0, mask_token_index].squeeze().cpu().numpy()

# 5. Extract top 10 candidates
top_k = 10
top_k_indices = mask_logits.argsort()[-top_k:][::-1]

print(f"Sentence: {masked_sentence}")
print(f"\nTop {top_k} candidates for the mask (using manual logit extraction):")
for i, idx in enumerate(top_k_indices):
    token_str = tokenizer.decode([idx])
    score = torch.nn.functional.softmax(torch.tensor(mask_logits), dim=0)[idx].item()
    print(f"{i+1}. Word: '{token_str}' | Logit: {mask_logits[idx]:.4f} | Softmax Score: {score:.4f}")
