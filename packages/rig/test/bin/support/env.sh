#---------------------------------------------------------------------------
#  Copyright (c) 2026 The Raven authors. All rights reserved.
#  SPDX-License-Identifier: ISC
#---------------------------------------------------------------------------

# Sourced by every test. It puts support/ssh and support/curl first on
# PATH, turns off the banner the debug runtime of the sanitize profile
# prints on every process's standard error, and defines [host M ADDRESS],
# which makes machine M of support/ssh with ADDRESS as its HostName.

export PATH=$PWD/support:$PATH
export OCAMLRUNPARAM="${OCAMLRUNPARAM:+$OCAMLRUNPARAM,}v=0"
host() { mkdir -p machines/$1 && echo $2 >machines/$1/hostname; }
