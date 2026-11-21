#!/usr/bin/env bash
# llama.cpp's side of PERFORMANCE.md: bench_gpt_oss.ml's rows, run by
# llama-bench on the GGUF file of gpt-oss-20b that llama.cpp's README points
# to, ggml-org/gpt-oss-20b-GGUF. That file packs the experts as MXFP4 and the
# attention projections and the head as Q8_0, as the bench's rows do.
#
#   apps/gpt-oss/bench/llama_cpp.sh [cuda|vulkan|rocm|cpu]
#
# The first run downloads llama.cpp's release binaries for the backend (cuda
# by default; cpu runs a thread on each core the process may use) and the
# 12.1 GB file into raven's cache. Each row prints its median time, as thumper
# does, and the rate a user reads:
#
#   decode-q8_0/512   one token at position 512, 30 calls (10 on the CPU)
#   prefill-q8_0/512  a 512-token prompt, 10 calls (3 on the CPU)
#   first-token/12    a new process from its start to the end of its first
#                     12-token prompt, loading the weights included, 3 processes
#
# and, on CUDA, the most device memory in use during decode-q8_0/512.
set -euo pipefail

release=b12541
releases=https://github.com/ggml-org/llama.cpp/releases/download
repo=ggml-org/gpt-oss-20b-GGUF
revision=41e8a94d15d62070b36cf4553a0250717ecf46b8
file=gpt-oss-20b-MXFP4.gguf
sha256=27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901

backend=${1:-cuda}
case $backend in
cuda) assets="llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz
cudart-llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz" ;;
vulkan) assets="llama-$release-bin-ubuntu-vulkan-x64.tar.gz" ;;
rocm) assets="llama-$release-bin-ubuntu-rocm-10.0-x64.tar.gz" ;;
cpu) assets="llama-$release-bin-ubuntu-x64.tar.gz" ;;
*) echo "usage: $0 [cuda|vulkan|rocm|cpu]" >&2; exit 2 ;;
esac
# On the CPU a thread a core it may use, and fewer calls: a prefill takes
# seconds there.
threads=() decode=30 prefill=10
if [ "$backend" = cpu ]; then threads=(-t "$(nproc)") decode=10 prefill=3; fi

cache=${RAVEN_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/raven}
bin=$cache/llama.cpp/$release-$backend
model=$cache/huggingface/$(echo $repo | tr / -)/$revision/$file

if [ ! -x "$bin/llama-bench" ]; then
  mkdir -p "$bin"
  for a in $assets; do
    curl -fL "$releases/$release/$a" | tar xz -C "$bin" --strip-components=1
  done
fi
if [ ! -f "$model" ]; then
  mkdir -p "$(dirname "$model")"
  curl -fL -o "$model.part" \
    "https://huggingface.co/$repo/resolve/$revision/$file"
  echo "$sha256  $model.part" | sha256sum -c --quiet
  mv "$model.part" "$model"
fi

# The median, least and most of the numbers on standard input, one a line.
stats() {
  sort -g | awk '{ v[NR] = $1 } END {
    m = NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
    print m, v[1], v[NR], NR }'
}

# llama-bench's time of each call, in seconds, one a line.
samples() {
  "$bin/llama-bench" -m "$model" "${threads[@]}" -o jsonl "$@" |
    sed -n 's/.*"samples_ns": \[\([^]]*\)\].*/\1/p' | tr ',' '\n' |
    awk 'NF { print $1 / 1e9 }'
}

# [row name tokens]: a row of seconds per call, and tokens a second.
row() {
  stats | awk -v name="$1" -v tokens="$2" '{
    printf "%-16s %8.2f ms  %7.1f tokens/s  (%.2f-%.2f ms, %d calls)\n",
      name, $1 * 1e3, tokens / $1, $2 * 1e3, $3 * 1e3, $4 }'
}

used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

echo "llama.cpp $release ($backend${threads[*]:+, ${threads[1]} threads})," \
  "$repo@$revision"

if [ "$backend" = cuda ]; then
  peak=$(mktemp)
  (while :; do used; sleep 0.1; done) >"$peak" &
  sampler=$!
fi
samples -p 0 -n 1 -d 512 -r $decode | row decode-q8_0/512 1
if [ "$backend" = cuda ]; then
  kill $sampler
  sort -n "$peak" | tail -1 |
    awk '{ printf "%-16s %8.2f GB in use, at most, during decode-q8_0/512\n",
      "device memory", $1 * 1048576 / 1e9 }'
  rm -f "$peak"
fi

samples -p 512 -n 0 -r $prefill | row prefill-q8_0/512 512

# The first process brings the file into the page cache, as the other rows
# did for the bench's.
for i in 0 1 2 3; do
  start=$(date +%s.%N)
  "$bin/llama-bench" -m "$model" "${threads[@]}" -p 12 -n 0 -r 1 --no-warmup \
    -o jsonl >/dev/null
  [ $i = 0 ] || echo "$start $(date +%s.%N)" | awk '{ print $2 - $1 }'
done | stats | awk '{
  printf "%-16s %8.2f s   (%.2f-%.2f s, %d processes)\n",
    "first-token/12", $1, $2, $3, $4 }'
