#!/usr/bin/env python3
"""Generate reference BERT outputs from HuggingFace transformers for comparison"""

from transformers import BertModel, BertTokenizer
import torch

def test_bert_output():
    print("Generating HuggingFace BERT reference outputs")
    print("=" * 45)
    
    # Load model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Test input
    text = "Hello world"
    print(f'\nInput text: "{text}"')
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get outputs
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    
    print(f"\nOutput shape: {list(last_hidden_state.shape)}")
    
    # CLS token embeddings (first token)
    cls_token = last_hidden_state[0, 0]  # [batch=0, token=0]
    
    print("\nCLS token embeddings (first 10 values):")
    for i in range(10):
        print(f"  [{i}]: {cls_token[i].item():.6f}")
    
    # Pooler output
    print("\nPooler output (first 10 values):")
    for i in range(10):
        print(f"  [{i}]: {pooler_output[0, i].item():.6f}")
    
    # Print for easy copy-paste into OCaml test
    print("\n" + "=" * 45)
    print("OCaml test expected values:")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
    print(f"CLS values: [{'; '.join(f'{cls_token[i].item():.6f}' for i in range(5))}]")
    print(f"Pooler values: [{'; '.join(f'{pooler_output[0, i].item():.6f}' for i in range(5))}]")

if __name__ == "__main__":
    test_bert_output()