(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let device = Nx_support.memory 1

module On = (val Nx.devices [ device ])

let rounds = 4

let settled () =
  for _ = 1 to rounds do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

let held f =
  f ();
  let before = settled () in
  f ();
  settled () - before
