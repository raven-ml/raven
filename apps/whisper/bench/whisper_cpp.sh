#!/usr/bin/env bash
# whisper.cpp's side of PERFORMANCE.md: bench_whisper.ml's rows, by
# whisper-cli on the ggml files of each release that whisper.cpp's README
# points to (ggerganov/whisper.cpp on Hugging Face), which hold float16
# weights.
#
#   apps/whisper/bench/whisper_cpp.sh cuda|cpu [MODEL...]
#
# The first run builds whisper.cpp's release from source (it publishes no
# Linux binary with CUDA; CUDA_ARCH, default 89, picks the GPU) and downloads
# the files into raven's cache; CUDA's libraries must be on the loader's path.
# It reads fetch.py's clip and recording.
# whisper-cli decodes in English with one beam and one draw (-bs 1 -bo 1), as
# raven's rows do, and at its defaults otherwise: the fallback ladder,
# conditioned on the text before, timestamps on. The rows, each in
# milliseconds and times real time:
#   minute/MODEL            fetch.py's clip, the recording's first minute:
#                           whisper-cli's total time less its load time, the
#                           median of 5 processes after 1
#   recording/MODEL         the whole recording, likewise, once
#   first-transcript/MODEL  a new process from its start to its exit on the
#                           clip, the median of 3 after 1
# and memory/MODEL, the buffers whisper.cpp states it allocates: weights,
# key-value caches and compute buffers. cpu runs the same build with -ng.
set -euo pipefail

release=v1.9.5
repo=ggerganov/whisper.cpp
revision=5359861c739e955e79d9a303bcbc70fb988958b1
declare -A sha256=(
  [tiny]=be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21
  [small]=1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b
  [large-v3-turbo]=1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69
)

device=${1:?usage: $0 cuda|cpu [MODEL...]}
shift
models=${*:-tiny small large-v3-turbo}
case $device in
cuda) flags= ;;
cpu) flags=-ng ;;
*) echo "usage: $0 cuda|cpu [MODEL...]" >&2; exit 2 ;;
esac

cache=${RAVEN_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/raven}
data=$cache/whisper
src=$cache/whisper.cpp/$release
cli=$src/build/bin/whisper-cli

if [ ! -x "$cli" ]; then
  git clone --quiet --depth 1 --branch $release https://github.com/$repo "$src"
  cmake -S "$src" -B "$src/build" -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-89} -DCMAKE_BUILD_TYPE=Release
  cmake --build "$src/build" -j "${JOBS:-4}" --target whisper-cli
fi
for m in $models; do
  file=$cache/huggingface/$(echo $repo | tr / -)/$revision/ggml-$m.bin
  if [ ! -f "$file" ]; then
    mkdir -p "$(dirname "$file")"
    curl -fL -o "$file.part" \
      "https://huggingface.co/$repo/resolve/$revision/ggml-$m.bin"
    echo "${sha256[$m]}  $file.part" | sha256sum -c --quiet
    mv "$file.part" "$file"
  fi
done

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

threads=${THREADS:-4}
log=$(mktemp)
trap 'rm -f "$log"' EXIT

# cli MODEL AUDIO [ARGS...]: whisper-cli's transcription, its log in $log.
cli() {
  local file=$cache/huggingface/$(echo $repo | tr / -)/$revision/ggml-$1.bin
  "$cli" -m "$file" -f "$2" -l en -bs 1 -bo 1 -t "$threads" $flags "${@:3}" \
    > /dev/null 2> "$log"
}

# Milliseconds of whisper-cli's last transcription, its load time left out.
transcribing() {
  awk '/load time =/ { load = $(NF-1) } /total time =/ { total = $(NF-1) }
    END { print total - load }' "$log"
}

echo "whisper.cpp $release, $device, float16 weights, $threads threads"
for m in $models; do
  cli "$m" "$data/clip.wav"
  for i in 1 2 3 4 5; do cli "$m" "$data/clip.wav"; transcribing; done |
    row "minute/$m" 60
  awk -v name="memory/$m" '/total size =|kv [a-z]+ *size =|compute buffer/ {
      for (i = 1; i <= NF; i++) if ($i == "MB") mb += $(i-1) }
    END { printf "%-28s %10.2f GB stated\n", name, mb / 1000 }' "$log"
  cli "$m" "$data/recording.wav"
  # The recording's seconds: 16-bit samples after a 44-byte header.
  seconds=$(( ($(stat -c %s "$data/recording.wav") - 44) / 32 ))e-3
  transcribing | row "recording/$m" "$seconds"
  for i in 0 1 2 3; do
    t0=$(date +%s%N); cli "$m" "$data/clip.wav"; t1=$(date +%s%N)
    [ $i -gt 0 ] && echo $(( (t1 - t0) / 1000000 ))
  done | row "first-transcript/$m" 60
done
