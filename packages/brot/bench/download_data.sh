#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/data" && pwd)"
# Under a Cygwin bash the native curl cannot open a /cygdrive path.
if command -v cygpath >/dev/null 2>&1; then
  DATA_DIR="$(cygpath -m "$DATA_DIR")"
fi

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Each file comes from a fixed commit of its repository and is checked against
# its SHA-256, so the parity fixtures generated from it can be regenerated.
# fetch <name> <repository> <commit> <sha256> <description>
fetch() {
  local name="$1" repo="$2" commit="$3" digest="$4" description="$5"
  local out="$DATA_DIR/$name.json"
  if [ -f "$out" ] && [ "$(sha256 "$out")" = "$digest" ]; then
    echo "  $description: present"
    return
  fi
  curl -fsSL -o "$out.part" \
    "https://huggingface.co/$repo/resolve/$commit/tokenizer.json"
  local actual
  actual="$(sha256 "$out.part")"
  if [ "$actual" != "$digest" ]; then
    rm -f "$out.part"
    echo "$repo@$commit tokenizer.json: SHA-256 $actual, expected $digest" >&2
    exit 1
  fi
  mv "$out.part" "$out"
  echo "  $description"
}

echo "Downloading real-world tokenizer models to $DATA_DIR..."

fetch gpt2 openai-community/gpt2 \
  607a30d783dfa663caf39e06633721c8d4cfcd7e \
  8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6 \
  "GPT-2 (BPE, 50K vocab)"

fetch bert_base google-bert/bert-base-uncased \
  86b5e0934494bd15c9632b12f734a8a67f723594 \
  ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98 \
  "BERT-base (WordPiece, 30K vocab)"

fetch llama hf-internal-testing/llama-tokenizer \
  d02ad6cb9dd2c2296a6332199fa2fdca5938fef0 \
  8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1 \
  "LLaMA (BPE, 32K vocab)"

fetch roberta_base FacebookAI/roberta-base \
  e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e \
  "RoBERTa-base (byte-level BPE, 50K vocab, RobertaProcessing)"

fetch t5_base google-t5/t5-base \
  a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 \
  d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066 \
  "T5-base (Unigram, 32K vocab, Precompiled normalizer, Metaspace)"

fetch mistral mistralai/Mistral-7B-v0.1 \
  27d67f1b5f57dc0953326b2601d68371d40ea8da \
  11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720 \
  "Mistral-7B-v0.1 (BPE with byte fallback, 32K vocab, Metaspace)"

fetch llama3 NousResearch/Llama-3.2-1B \
  a9745ffc3556f145a830ac0c203509ba860582a3 \
  79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4 \
  "Llama 3.2 (byte-level BPE, 128K vocab, regular-expression Split)"

fetch qwen2_5 Qwen/Qwen2.5-0.5B \
  060db6499f32faf8b98477b0a26969ef7d8b9987 \
  c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
  "Qwen2.5 (byte-level BPE, 151K vocab, regular-expression Split, NFC)"

# DeepSeek-V3-0324's tokenizer.json is DeepSeek-V3's, under the MIT licence.
fetch deepseek_v3 deepseek-ai/DeepSeek-V3-0324 \
  e9b33add76883f293d6bf61f6bd89b497e80e335 \
  621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41 \
  "DeepSeek-V3 (byte-level BPE, 128K vocab, three regular-expression Splits)"

fetch gpt_oss openai/gpt-oss-20b \
  6cee5e81ee83917806bbde320786a8fb61efebee \
  0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3 \
  "gpt-oss (byte-level BPE, 200K vocab, the o200k regular-expression Split)"

fetch phi4 microsoft/phi-4 \
  2db69c1c3e91a05d2c64a3185acfbaf36f744e25 \
  9f38d05d9d25756bb2f181ab5a0cebcd59e638df10336fc7ed1010f7296d0298 \
  "Phi-4 (byte-level BPE, 100K vocab, the cl100k Split with its matches kept)"

fetch mistral_nemo mistralai/Mistral-Nemo-Instruct-2407 \
  04d8a90549d23fc6bd7f642064003592df51e9b3 \
  e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db \
  "Mistral NeMo (byte-level BPE, 131K vocab, the Tekken regular-expression Split)"

echo "Done."
