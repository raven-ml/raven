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
  type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

  val name : string
  val computes_on : Rig.t -> bool
  val apply0 : Prog.op0 -> dst:('v, 's) a -> answer
  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
  val apply2 : Prog.op2 -> dst:('v, 's) a -> ('a, 'b) a -> ('a, 'b) a -> answer

  val apply3 :
    Prog.op3 -> dst:('v, 's) a -> ('c, 'e) a -> ('a, 'b) a -> ('a, 'b) a -> answer

  val map : Spec.map Spec.t -> dsts:any array -> any array -> answer
  val reduce : Spec.reduce Spec.t -> dsts:any array -> any array -> answer
  val scan : Spec.scan Spec.t -> dsts:any array -> any array -> answer
  val gather : Spec.gather Spec.t -> dst:('v, 's) a -> index -> ('v, 's) a -> answer

  val scatter :
    Spec.scatter Spec.t ->
    dst:('v, 's) a ->
    into:('v, 's) a ->
    index ->
    ('v, 's) a ->
    answer

  val sort :
    Spec.sort Spec.t -> values:('v, 's) a -> positions:index -> ('v, 's) a -> answer

  val assemble : Spec.assemble Spec.t -> dst:('v, 's) a -> ('v, 's) a array -> answer
  val fold : Spec.fold Spec.t -> dst:('v, 's) a -> ('v, 's) a -> answer
  val contract : Spec.contract Spec.t -> dst:any -> any array -> answer
end
