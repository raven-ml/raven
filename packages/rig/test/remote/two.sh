#!/bin/sh
#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# A job on two machines with GPUs: rig run starts support/two.exe here, on
# CTRL, whose GPU is AMD or NV, and an agent on HOST, whose GPU sends three
# frames over a rail into CTRL's. Both machines run this tree's rig: HOST
# holds the tree at the same path, built at the same commit. The caller
# holds both machines' GPU locks.
#
# usage: two.sh CTRL HOST

set -eu
[ $# -eq 2 ] || { echo "usage: two.sh CTRL HOST" >&2; exit 2; }
cd "$(dirname "$0")"
rig=$(cd ../../../.. && pwd)/_build/default/packages/rig/bin/main.exe
exec "$rig" run --rig "$rig" --on "$1,$2" -- \
  ../../../../_build/default/packages/rig/test/remote/support/two.exe
