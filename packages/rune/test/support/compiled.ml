(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let here ?(on = Rig.host) () =
  let host_code = Rig.equal (Rig.host_of on) on || Rig.runs_on_host on in
  if host_code && not (Rig.Image.links on) then
    Windtrap.skip
      ~reason:
        ("rig links no host code for "
        ^ Rig.instruction_set (Rig.host_of on)
        ^ ": Rune.jit evaluates on " ^ Rig.name on)
      ()
