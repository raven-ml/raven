#!/bin/sh
# bench/README.md's table: lz (vendor_table.exe, built by dune) and the
# vendors (vendor_table.c, built here against ~/wt/lz-vendors/: libdeflate
# 1.26, lz4 1.10.0 and snappy 1.2.2 as built there, zstd 1.5.7 compiled
# from its sources with its Makefile's -O3, and the system's zlib). Each
# row runs lz then the vendors, so the two sides of a row are timed
# minutes apart at most. Run from the tree's root:
#
#   dune build packages/lz/bench/measure/vendor_table.exe
#   packages/lz/bench/measure/vendor_table.sh OUT [CORPUS...]
#
# OUT receives the vendor builds and table.tsv. CORPUS defaults to
# silesia. Decoders take the fastest of 10 runs, encoders of 3, CRC-32 of
# 50, on cores 0-5 under the timing lock, once the 1-minute load is under
# 1.5 (an hour at most, else it stops).
set -e
out=$1
[ -n "$out" ] || { echo "usage: vendor_table.sh OUT [CORPUS...]" >&2; exit 2; }
shift
corpora=${*:-silesia}
mkdir -p "$out/zstd"
here=packages/lz/bench/measure
data=packages/lz/bench/data
lz=_build/default/$here/vendor_table.exe
v=$HOME/wt/lz-vendors
z=$v/zstd-1.5.7/lib

# zstd's library: its lib/ sources with libzstd.mk's flags, single-threaded.
for f in $z/common/*.c $z/compress/*.c $z/decompress/*.c \
         $z/decompress/huf_decompress_amd64.S; do
  o=$out/zstd/$(basename "$f").o
  [ -f "$o" ] || gcc -O3 -DXXH_NAMESPACE=ZSTD_ -DZSTD_LEGACY_SUPPORT=0 \
    -I $z -I $z/common -c "$f" -o "$o"
done
for s in avx2 default; do
  g++ -O2 -I $v/libdeflate-1.26 -I $v/lz4-1.10.0/lib -I $v/snappy-1.2.2 \
    -I $v/snappy-1.2.2/build-$s -I $z -x c $here/vendor_table.c -x none \
    $out/zstd/*.o $v/libdeflate-1.26/build/libdeflate.a \
    $v/lz4-1.10.0/lib/liblz4.a $v/snappy-1.2.2/build-$s/libsnappy.a -lz \
    -o "$out/vendor_table-$s"
done

rows="decode deflate|decode zlib|decode gzip|decode zstd 1|decode zstd 3|\
decode zstd 12|decode zstd 19|decode lz4 block|decode lz4 frame|decode snappy"
for f in deflate zlib gzip; do
  for l in 1 2 3 4 5 6 7 8 9; do rows="$rows|encode $f $l"; done
done
for l in 1 2 3 4 5 6 7 8 9 10 11 12; do rows="$rows|encode zstd $l"; done
rows="$rows|encode lz4 block|encode lz4 frame|encode snappy|crc32"

flock -w 1800 "$HOME/wt/TIMING.lock" taskset -c 0-5 sh -c '
  out=$1 lz=$2 data=$3 rows=$4; shift 4
  i=0
  until awk "{exit !(\$1 < 1.5)}" /proc/loadavg; do
    i=$((i + 1))
    [ $i -le 60 ] || { echo "the host is not quiet" >&2; exit 1; }
    sleep 60
  done
  echo "# before: $(uptime)" >&2
  IFS="|"
  for r in $rows; do
    for c in "$@"; do
      $lz -r "$r" $data 10 3 $c
      "$out/vendor_table-avx2" -r "$r" $data 10 3 $c
      case $r in *snappy*)
        "$out/vendor_table-default" -r "$r" $data 10 3 $c |
          sed "s/\tsnappy\t/\tsnappy portable\t/";;
      esac
    done
  done
  echo "# after: $(uptime)" >&2' sh "$out" "$lz" "$data" "$rows" $corpora \
  | tee "$out/table.tsv"
