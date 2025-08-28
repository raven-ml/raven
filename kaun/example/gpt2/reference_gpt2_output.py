#!/usr/bin/env python3
"""Get reference GPT-2 output from transformers for comparison"""

import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# Load model and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set to eval mode
model.eval()
lm_model.eval()

# Test input
test_text = "Hello world"
print(f"Test input: {repr(test_text)}")

# Tokenize
inputs = tokenizer(test_text, return_tensors="pt")
print(f"Input IDs: {inputs.input_ids.tolist()[0]}")
print(f"Input shape: {list(inputs.input_ids.shape)}")

# Forward pass (base model)
print("\n=== Base Model Output ===")
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    
print(f"Output shape: {list(last_hidden_state.shape)}")
print(f"First 10 values: {last_hidden_state.flatten()[:10].tolist()}")

# Save the full output for detailed comparison
np.save("gpt2_reference_output.npy", last_hidden_state.numpy())
print("Saved full output to gpt2_reference_output.npy")

# Forward pass (LM model)
print("\n=== Language Model Output ===")
with torch.no_grad():
    lm_outputs = lm_model(**inputs)
    logits = lm_outputs.logits
    
print(f"Logits shape: {list(logits.shape)}")

# Get predictions for next token (after "world")
last_logits = logits[0, -1, :]  # Last position
probs = torch.softmax(last_logits, dim=-1)

# Top 5 predictions
top_probs, top_indices = torch.topk(probs, 5)
print("\nTop 5 predicted next tokens:")
for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
    token = tokenizer.decode([idx])
    print(f"  Token {idx} ({repr(token)}): {prob:.4f}")

# Save logits for comparison
np.save("gpt2_reference_logits.npy", logits.numpy())
print("\nSaved logits to gpt2_reference_logits.npy")

# Also test a few more examples
print("\n=== Additional Test Cases ===")
test_cases = [
    "The",
    "GPT-2 is",
    "Hello",
]

for text in test_cases:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"{repr(text)}: shape={list(outputs.last_hidden_state.shape)}, "
          f"first_val={outputs.last_hidden_state[0,0,0].item():.6f}")