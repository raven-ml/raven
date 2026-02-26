#!/usr/bin/env python3
"""Generate reference BERT outputs from HuggingFace transformers."""

from transformers import BertModel, BertTokenizer
import torch


def main():
    print("Generating HuggingFace BERT reference outputs")
    print("=" * 50)

    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text = "Hello world"
    print(f'\nInput text: "{text}"')

    inputs = tokenizer(text, return_tensors="pt")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

    print(f"Output shape: {list(last_hidden_state.shape)}")

    cls_token = last_hidden_state[0, 0]
    print("\nCLS token (first 5 values):")
    for i in range(5):
        print(f"  [{i}]: {cls_token[i].item():.6f}")

    print("\nPooler output (first 5 values):")
    for i in range(5):
        print(f"  [{i}]: {pooler_output[0, i].item():.6f}")

    print("\n" + "=" * 50)
    print("OCaml expected values:")
    cls_vals = "; ".join(f"{cls_token[i].item():.6f}" for i in range(5))
    pool_vals = "; ".join(f"{pooler_output[0, i].item():.6f}" for i in range(5))
    print(f"expected_cls = [| {cls_vals} |]")
    print(f"expected_pooler = [| {pool_vals} |]")


if __name__ == "__main__":
    main()
