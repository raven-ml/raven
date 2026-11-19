#!/bin/sh
# whole.sh UPSTREAM TILE DIR [MODEL...]: the tile embedded by each side's
# command, in a new process, from its start to its exit: main.exe encode, and
# TESSERA's infer_v2.py (UPSTREAM at e368539f), on cuda, by the released
# students (nano small medium large by default), downloaded to DIR. Three
# rounds, raven then PyTorch, each printing "tile/M SIDE SECONDS"; raven's
# kernels come from a disk cache that its first run filled, and medium's
# "first-tile/medium-cold" row has a cache of its own, empty, each run.
# A thumper suite cannot hold these: a run is a new process of up to two
# minutes. Run from the repository's root after building main.exe.
set -eu
upstream=$1 tile=$2 dir=$3; shift 3
models=${*:-nano small medium large}
main=$PWD/_build/default/apps/tessera/main.exe
py=$(uv run -q --no-project --with torch==2.15.0 --with numpy==2.5.4 \
  python -c 'import sys; print(sys.executable)')
mkdir -p "$dir"
for m in $models; do
  case $m in nano) r=N:bd08c90bf56ca6a1e7609a395cd47115fa12f1d8 ;;
    small) r=S:9e1a725f1ac1a3a8ab93fd85b08633754cacb508 ;;
    medium) r=M:8c80b926f67cb0b44c2cd41d0226e61741503b9a ;;
    large) r=L:24e186649bdd544e7c770f277444f3f88c71571c ;; esac
  [ -f "$dir/student_$m.pt" ] || curl -sfL -o "$dir/student_$m.pt" \
    "https://huggingface.co/geotessera/TESSERA-V-2.0-2B-${r%%:*}/resolve/${r#*:}/ckpt/student_$m.pt"
done
# [timed NAME SIDE CACHE CMD...] runs CMD with tolk's and CUDA's caches in CACHE.
timed() {
  name=$1 side=$2 c=$3; shift 3
  s=$(date +%s.%N)
  (export CACHEDB=$c/cache.db XDG_CACHE_HOME=$c CUDA_CACHE_PATH=$c/cuda; "$@") > /dev/null
  echo "$name $side $(awk "BEGIN { printf \"%.2f\", $(date +%s.%N) - $s }")"
}
raven() { "$main" encode --model "$dir/student_$1.pt" --tile "$tile" --device cuda --out "$dir/e"; }
torch() { "$py" "$upstream/tessera_infer_v2/infer_v2.py" --model "$1" --ckpt "$dir/student_$1.pt" \
  --tile "$tile" --out-dir "$dir" --device cuda; }
for m in $models; do timed fill raven "$dir/warm" raven "$m" > /dev/null; done
for round in 1 2 3; do
  for m in $models; do
    timed "tile/$m" raven "$dir/warm" raven "$m"
    timed "tile/$m" pytorch "$dir/warm" torch "$m"
  done
  case " $models " in *" medium "*)
    rm -rf "$dir/cold"; timed first-tile/medium-cold raven "$dir/cold" raven medium ;; esac
done
