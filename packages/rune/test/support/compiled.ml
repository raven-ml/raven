(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let here () =
  if not (Rig.Image.links Rig.host) then
    Windtrap.skip
      ~reason:
        ("rig links no host code for "
        ^ Rig.instruction_set Rig.host
        ^ ": Rune.jit evaluates on the host")
      ()
