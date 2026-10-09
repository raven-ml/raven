(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Prog = Prog
module Spec = Spec

module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  val computes_on : Rig.t -> bool
  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
  val contract : Spec.contract Spec.t -> dst:any -> any array -> answer
end
