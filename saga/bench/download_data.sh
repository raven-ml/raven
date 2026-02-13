#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/data" && pwd)"

echo "Downloading real-world tokenizer models to $DATA_DIR..."

curl -sL -o "$DATA_DIR/gpt2.json" \
  "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
echo "  GPT-2 (BPE, 50K vocab)"

curl -sL -o "$DATA_DIR/bert_base.json" \
  "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json"
echo "  BERT-base (WordPiece, 30K vocab)"

curl -sL -o "$DATA_DIR/llama.json" \
  "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.json"
echo "  LLaMA (BPE, 32K vocab)"

echo "Done."
