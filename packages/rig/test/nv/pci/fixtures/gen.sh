#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Makes each FILE of this directory from its source, or every file without
# one. Dune never runs it. After a change to a source, run it and commit the
# bytes. Needs uv on PATH, which runs vbios.py.
#
# usage: gen.sh [FILE...]

set -eu
cd "$(dirname "$0")"

fixture() {
  case $1 in
  vbios.rom | vbios_no_bit.rom | vbios_debug.rom | vbios_no_mapper.rom)
    uv run -q vbios.py ;;
  *) echo "gen.sh: no recipe for $1" >&2; exit 2 ;;
  esac
}

[ $# -gt 0 ] || set -- \
  vbios.rom vbios_no_bit.rom vbios_debug.rom vbios_no_mapper.rom
for f in "$@"; do fixture "$f"; done
