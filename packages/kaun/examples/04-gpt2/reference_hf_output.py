#!/usr/bin/env python3
"""Get reference GPT-2 output from transformers for comparison."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = "Hello world"
print(f"Test input: {repr(text)}")

inputs = tokenizer(text, return_tensors="pt")
print(f"Input IDs: {inputs.input_ids.tolist()[0]}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"\nLogits shape: {list(logits.shape)}")
print(f"First 10 logit values at position 0: {logits[0, 0, :10].tolist()}")

# Top 5 predictions for next token (after "world")
last_logits = logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top_probs, top_indices = torch.topk(probs, 5)

print("\nTop 5 predicted next tokens:")
for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
    token = tokenizer.decode([idx])
    print(f"  Token {idx} ({repr(token)}): {prob:.4f}")

# Greedy generation
generated = model.generate(
    inputs.input_ids, max_new_tokens=20, do_sample=False
)
print(f"\nGreedy generation: {tokenizer.decode(generated[0])}")
print(f"Token IDs: {generated[0].tolist()}")
