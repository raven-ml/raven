(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type launch = {
  kernel : Cubin.kernel;
  gpu : Gpu.t;
  shared_bytes : int;
  shared_config : int;
  max_shared_config : int;
}

(* Built once, so a lookup allocates nothing. *)
let v3 = Some Defs.qmd_v3
let v5 = Some Defs.qmd_v5

let layout (g : Gpu.t) =
  if g.compute_class = Defs.blackwell_compute_b then v5
  else if
    g.compute_class = Defs.ampere_compute_b
    || g.compute_class = Defs.ada_compute_a
  then v3
  else None
