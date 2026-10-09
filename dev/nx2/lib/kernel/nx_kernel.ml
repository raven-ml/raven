(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Prog = Prog

module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t

  val name : string
  val computes_on : Rig.t -> bool
  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> int
end

(* NX_NOT_COMPUTED of nx_spec.h. *)
let not_computed = -1
