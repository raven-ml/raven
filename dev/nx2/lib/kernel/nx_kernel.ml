(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Prog = Prog

module type S = sig
  type ('v, 's) a := ('v, 's) Nx_array.t
  type answer := Nx_array.answer

  val name : string
  val computes_on : Rig.t -> bool
  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
end
