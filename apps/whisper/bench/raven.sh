#!/usr/bin/env bash
# raven's rows of PERFORMANCE.md that a thumper suite cannot hold, in the
# reference scripts' format: each takes a process or a whole recording.
#
#   apps/whisper/bench/raven.sh cuda|cpu [MODEL...]
#
# from the repository's root, after `dune build apps/whisper`. It reads
# fetch.py's files.
#   recording/MODEL             the whole recording by bench_whisper.exe
#                               --recording, once, after a call on the clip;
#                               its text written beside it for wer.py
#   first-transcript/MODEL      main.exe from its start to its exit on
#                               fetch.py's clip, the recording's first
#                               minute, over tolk's kernel cache that a first
#                               process filled, the median of 3
#   first-transcript-cold/MODEL the same over empty caches of its own, so
#                               that it compiles every kernel, the median of 3
# On the host main.exe runs eagerly, which compiles nothing, so the cold row
# is the GPU's alone.
set -euo pipefail

device=${1:?usage: $0 cuda|cpu [MODEL...]}
shift
models=${*:-tiny small large-v3-turbo large-v3}
build=_build/default/apps/whisper
cache=${RAVEN_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/raven}
data=$cache/whisper
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

# The median, least and most of the numbers on standard input, one a line.
stats() {
  sort -g | awk '{ v[NR] = $1 } END {
    m = NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2
    print m, v[1], v[NR], NR }'
}

# row NAME AUDIO_SECONDS: the row of the milliseconds on standard input.
row() {
  stats | awk -v name="$1" -v s="$2" '{
    printf "%-28s %10.1f ms  %7.1fx real time  (%.1f-%.1f ms, %d calls)\n",
      name, $1, s * 1000 / $1, $2, $3, $4 }'
}

# Milliseconds of main.exe on the clip, with ENV... before its environment.
first() {
  local t0 t1
  t0=$(date +%s%N)
  env "$@" "$build/bin/main.exe" --model "$data/$m.pt" \
    --tokenizer "$data/ranks" --device "$device" --language en \
    --beam_size None --best_of None --verbose false -f txt -o "$out" \
    "$data/clip.wav"
  t1=$(date +%s%N)
  echo $(( (t1 - t0) / 1000000 ))
}

echo "raven $(git rev-parse --short HEAD), $device, float32"
"$build/bench/bench_whisper.exe" --recording "$device" $models
for m in $models; do
  warm=$out/warm
  mkdir -p "$warm"
  caches="CACHEDB=$warm/db XDG_CACHE_HOME=$warm CUDA_CACHE_PATH=$warm/cuda"
  first $caches > /dev/null
  for i in 1 2 3; do first $caches; done | row "first-transcript/$m" 60
  if [ "$device" != cpu ]; then
    for i in 1 2 3; do
      cold=$out/cold-$i
      mkdir -p "$cold"
      first CACHEDB="$cold/cache.db" XDG_CACHE_HOME="$cold" \
        CUDA_CACHE_PATH="$cold/cuda"
    done | row "first-transcript-cold/$m" 60
  fi
done
