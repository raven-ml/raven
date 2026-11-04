(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's operations as tolk tensor graphs.

    {!op} takes the node of each operand of an operation and gives the node of
    each result. A node holds its value's elements as {!Lower_arith} states. The
    graph computes on one device, and the caller must hold every value on it.

    Each operation lowers as follows:
    - A [Map] is its program's expression over its loads ({!Lower_arith.prog}).
      A padded load reads its operand padded as {!Nx_kernel.load} states, and
      gives the fill where it reads no element.
    - A [Reduce] and a [Scan] reduce their program's outputs as {!Lower_reduce}
      does, and a [Contract] sums its products in its accumulator as
      {!Lower_reduce.contract} does.
    - A [Copy] and a [Place] are their operand, a [Move] its operand moved, and
      a [Bitcast] its operand's bits as {!Lower_arith.bitcast} reads them.
    - A [Gather], a [Scatter], a [Fold], an [Assemble], a [Sort] and a
      [Unique] compute as {!Lower_index} states, a [Linalg] as
      {!Lower_linalg} states, an [Fft] transforms as {!Lower_fft} does, and
      an [Attention] computes as {!Lower_attention} states.

    {b Numerics.} A result is what {!Nx.Prim.t} states of its operation, within
    the bounds the modules above state. Those bounds implement
    {!Rune.section-numerics}. *)

open Tolk

val draws : Nx_kernel.Prog.t -> bool
(** [draws prog] is whether [prog] computes Threefry blocks, as a draw and a key
    split do. *)

val op :
  ?dst:Lower_index.dst ->
  renderer:Renderer.t Lazy.t ->
  device:Ops.device ->
  host:bool ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  'r Nx.Prim.t ->
  Ops.t array
(** [op ~dst ~renderer ~device ~host node o] is the node of each result of
    [o], in position order, computed on [device], whose kernels compile for
    the host's processors where [host] holds ({!Tolk_engine.target}) and are
    rendered by [renderer]. It reads each operand [x] from [node x]. A
    [Scatter] takes its storage from [dst] where {!Lower_index.scatter} does;
    another operation ignores it.

    Raises {!Lower_arith.Unlowered}, naming the operation, for one it does not
    lower: [Check], which its caller lowers itself, a [Linalg] that
    {!Lower_linalg.op} refuses, a boolean [Add] [Scatter] or [Fold], a [Sort]
    along an axis of more than [2{^31}] elements ({!Lower_index.sort}), a
    [Unique] of more than [2{^31}] rows, an [Fft] of a precision other than
    [float32] and [float64] or whose result's precision is not its operand's,
    a [Contract] scaled in an accumulator other than [float32] or [float64], a
    reduction or a scan of a complex [Prod], and a scan of a [complex128]
    [Max], [Min] or [Arg]; [Invalid_argument] for a [Gather_ranges], which a
    program holds as a [Gather] at its positions. *)

(**/**)

val scatter_plan :
  host:bool -> Renderer.t -> p:int -> n:int -> q:int -> k:int -> int * int
(** [scatter_plan] is {!Lower_index.plan}: the runs of positions and the
    chunks of updates a [Scatter] takes. *)
