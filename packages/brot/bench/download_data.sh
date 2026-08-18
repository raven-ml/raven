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

curl -sL -o "$DATA_DIR/roberta_base.json" \
  "https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer.json"
echo "  RoBERTa-base (byte-level BPE, 50K vocab, RobertaProcessing)"

# T5's only normalizer is a Precompiled charsmap, which brot does not implement,
# so what is saved here is T5 with that normalizer dropped. The parity fixtures
# are generated from this same file, so the reference and brot read one
# tokenizer; what it loses is the SentencePiece NFKC-like character folding.
curl -sL "https://huggingface.co/google-t5/t5-base/resolve/main/tokenizer.json" \
  | python3 -c 'import json,sys; t=json.load(sys.stdin); t["normalizer"]=None; json.dump(t,sys.stdout)' \
  > "$DATA_DIR/t5_base_nonorm.json"
echo "  T5-base without its Precompiled normalizer (Unigram, 32K vocab, Metaspace)"

echo "Done."
