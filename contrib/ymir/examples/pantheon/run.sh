#!/bin/sh
# The 1999 supernova cosmology result redone on Pantheon+, in one command:
#
#   contrib/ymir/examples/pantheon/run.sh
#
# Fetches the two files of the Pantheon+ data release, pinned to a commit
# and checked by SHA-256, into $PANTHEON_DIR (default
# ${XDG_CACHE_HOME:-~/.cache}/raven/pantheon+), then fits them.
set -eu

dir=${PANTHEON_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/raven/pantheon+}
release=https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/c447f0fea703fcd0fff57de5000947b5ca81286b/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi |
    cut -d ' ' -f 1
}

fetch() {
  file=$1 url=$2 digest=$3
  if [ ! -f "$dir/$file" ] || [ "$(sha256 "$dir/$file")" != "$digest" ]; then
    mkdir -p "$dir"
    curl -sfL -o "$dir/$file.part" "$release/$url"
    if [ "$(sha256 "$dir/$file.part")" != "$digest" ]; then
      echo "$file: the download's SHA-256 is not $digest" >&2
      exit 1
    fi
    mv "$dir/$file.part" "$dir/$file"
  fi
}

fetch Pantheon+SH0ES.dat Pantheon%2BSH0ES.dat \
  1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8
fetch Pantheon+SH0ES_STAT+SYS.cov Pantheon%2BSH0ES_STAT%2BSYS.cov \
  abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc

root=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$root"
exec dune exec contrib/ymir/examples/pantheon/pantheon.exe -- "$dir"
