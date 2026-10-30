#!/bin/sh
# Measurement set of bench/measurements.md, "Snappy": lz's kernels (whole.c),
# the same decoder without its pattern expansion, and Google's snappy 1.2.2
# in its portable and AVX2 builds (the harness test/gen/snappy_ref.cc built
# against ~/wt/lz-vendors/snappy-1.2.2/build-{default,avx2}, as
# test/gen/google_snappy.py says), each decoding data/<corpus>.snappy. Run
# from the tree's root after `dune build packages/lz/lib/liblz_stubs.a`; builds
# into OUT, then times under the lock:
#
#   packages/lz/bench/measure/snappy/compare.sh OUT
set -e
out=$1
[ -n "$out" ] || { echo "usage: compare.sh OUT" >&2; exit 2; }
mkdir -p "$out"
here=packages/lz/bench/measure/snappy
lib=packages/lz/lib
data=packages/lz/bench/data
vendor=$HOME/wt/lz-vendors/snappy-1.2.2
gcc -std=c11 -O2 -I $lib -o "$out/whole" $here/whole.c \
  _build/default/$lib/liblz_stubs.a
# Without the expansion: a copy under 16 bytes back, byte by byte.
sed 's|^        expand(op, off, len);$|        for (size_t k = 0; k < len; k++) op[k] = (op - off)[k];|' \
  $lib/snappy.c > "$out/noexp.c"
grep -q '(op - off)\[k\]' "$out/noexp.c"
gcc -std=c11 -O3 -Wno-unused-function -I $lib -o "$out/noexp" $here/whole.c \
  "$out/noexp.c"
flock -w 1800 "$HOME/wt/TIMING.lock" taskset -c 0-5 sh -c '
  out=$1 data=$2 vendor=$3
  echo "before: $(uptime)"
  (sleep 3; top -b -n 1 | head -12) &
  for c in text columns random silesia; do
    r=10; [ $c = silesia ] && r=3
    for x in "lz $out/whole" "lz-noexp $out/noexp" \
             "google $vendor/build-default/snappy_ref" \
             "google-avx2 $vendor/build-avx2/snappy_ref"; do
      set -- $x
      echo "$1 $($2 bench $data/$c.raw $data/$c.snappy $r)"
    done
  done
  wait
  echo "after: $(uptime)"' sh "$out" "$data" "$vendor"
