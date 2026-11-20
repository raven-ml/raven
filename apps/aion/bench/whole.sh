#!/bin/sh
# whole.sh DATA DIR DEVICE [MODEL]: main.exe's three commands and upstream.py's,
# each a new process from its start to its exit, on fetch.py's files in DATA,
# with the released weights of MODEL (base by default), which the first runs
# download to DIR. DEVICE is main.exe's --device: on a GPU, embed and
# posterior take 64 objects and generate 64 stars; on the host (cpu,
# cpu-jit) 16, 16 and 4; in batches of main.exe's default 16. PyTorch runs
# on cuda for a GPU and on the cpu for the host. Three rounds, raven then
# PyTorch, each printing "whole/COMMAND/MODEL SIDE SECONDS"; raven's kernels
# come from a disk cache that a first run of each command filled, and
# "first/embed/MODEL raven" runs embed over empty caches of its own, which
# compiles every kernel. SIDES="raven" leaves PyTorch out, and ROUNDS=N runs N
# rounds (0 runs the first runs alone, as a check). A thumper suite
# cannot hold these: each run is a new process that loads 1.3 to 3.5 GB of
# weights. Run from the repository's root after building main.exe.
set -eu
data=$1 dir=$2 device=$3 model=${4:-base}
sides=${SIDES:-raven pytorch}
case $device in
  cpu|cpu-jit) n=16 g=4 b=16 torch=cpu ;;
  *) n=64 g=64 b=16 torch=cuda ;;
esac
main=$PWD/_build/default/apps/aion/bin/main.exe
up=$PWD/apps/aion/bench/upstream.py
mkdir -p "$dir"
# PyTorch's own environment in DIR: uv's --with environment is gone once its
# command exits.
py=$dir/venv/bin/python
[ -x "$py" ] || { uv venv -q -p 3.12 "$dir/venv" &&
  uv pip install -q --python "$py" torch==2.15.0 numpy==2.5.5 h5py==3.16.0 \
    huggingface-hub==2.4.0 safetensors==0.8.0 \
    "polymathic-aion @ git+https://github.com/PolymathicAI/AION@db7fba6df53b605e3a1a58b7f52fc33bfaaa96d3"; }
export RAVEN_CACHE_ROOT="$dir/raven" HF_HOME="$dir/hf"
t=$data/tutorial.h5
photometry=tok_image,tok_flux_g,tok_flux_r,tok_flux_i,tok_flux_z
# [timed NAME SIDE CACHE CMD...] runs CMD with tolk's and CUDA's caches in CACHE.
timed() {
  name=$1 side=$2 cache=$3; shift 3
  s=$(date +%s.%N)
  (export CACHEDB=$cache/cache.db XDG_CACHE_HOME=$cache CUDA_CACHE_PATH=$cache/cuda
   "$@") > /dev/null
  echo "$name $side $(awk "BEGIN { printf \"%.2f\", $(date +%s.%N) - $s }")"
}
raven() {
  case $1 in
    embed) set -- embed --legacysurvey "$t:/legacysurvey" --desi "$t:/desi" \
      --rows "0:$n" --batch "$b" ;;
    posterior) set -- posterior --legacysurvey "$t:/legacysurvey" --target tok_z \
      --inputs "$photometry" --rows "0:$n" --batch "$b" ;;
    generate) set -- generate --gaia "$data/gaia.h5" --target tok_spectrum_desi \
      --inputs tok_xp_bp,tok_xp_rp --steps 12 --rows "0:$g" --batch "$b" ;;
  esac
  "$main" "$@" --model "$model" --device "$device" --out "$dir/raven.npz"
}
pytorch() {
  case $1 in
    generate) set -- generate "$data/gaia.h5" --rows "$g" --batch "$b" ;;
    *) set -- "$1" "$t" --rows "$n" --batch "$b" ;;
  esac
  "$py" "$up" "$@" --device "$torch" --model "$model" --out "$dir/pytorch.npz"
}
commands="embed posterior generate"
for c in $commands; do
  for side in $sides; do timed fill "$side" "$dir/warm" "$side" "$c" > /dev/null; done
done
for round in $(seq "${ROUNDS:-3}"); do
  for c in $commands; do
    for side in $sides; do timed "whole/$c/$model" "$side" "$dir/warm" "$side" "$c"; done
  done
  if [ "$device" != cpu ]; then
    rm -rf "$dir/cold"
    timed "first/embed/$model" raven "$dir/cold" raven embed
  fi
done
