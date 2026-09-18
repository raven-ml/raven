#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/data" && pwd)"
# Under a Cygwin bash the native curl cannot open a /cygdrive path.
if command -v cygpath >/dev/null 2>&1; then
  DATA_DIR="$(cygpath -m "$DATA_DIR")"
fi

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
  | python3 -c 'import json,sys; t=json.load(sys.stdin.buffer); t["normalizer"]=None; json.dump(t,sys.stdout)' \
  > "$DATA_DIR/t5_base_nonorm.json"
echo "  T5-base without its Precompiled normalizer (Unigram, 32K vocab, Metaspace)"

curl -sL -o "$DATA_DIR/mistral.json" \
  "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json"
echo "  Mistral-7B-v0.1 (BPE with byte fallback, 32K vocab, Metaspace)"

curl -sL -o "$DATA_DIR/llama3.json" \
  "https://huggingface.co/NousResearch/Llama-3.2-1B/resolve/main/tokenizer.json"
echo "  Llama 3.2 (byte-level BPE, 128K vocab, regular-expression Split)"

curl -sL -o "$DATA_DIR/qwen2_5.json" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json"
echo "  Qwen2.5 (byte-level BPE, 151K vocab, regular-expression Split, NFC)"

curl -sL -o "$DATA_DIR/deepseek_v3.json" \
  "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json"
echo "  DeepSeek-V3 (byte-level BPE, 128K vocab, three regular-expression Splits)"

curl -sL -o "$DATA_DIR/gpt_oss.json" \
  "https://huggingface.co/openai/gpt-oss-20b/resolve/main/tokenizer.json"
echo "  gpt-oss (byte-level BPE, 200K vocab, the o200k regular-expression Split)"

curl -sL -o "$DATA_DIR/phi4.json" \
  "https://huggingface.co/microsoft/phi-4/resolve/main/tokenizer.json"
echo "  Phi-4 (byte-level BPE, 100K vocab, the cl100k Split with its matches kept)"

echo "Done."
