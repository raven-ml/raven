(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations as data, and values' facts: the functions over {!Value.prim} and
    {!Value.t} that the engine and [Nx.Prim] share.

    Every function is total over the constructors. {!results} states each
    operation's rule once: the shapes, dtypes, axes and attributes its operands
    must have, and the form of each result, its placement from {!Route}. The
    eager engine allocates destinations through it, and interpretations make
    traced results through it, so the two agree in dtype, layout and placement.
    A sharded value's shape is the whole's: each shard's, times the number of
    tiles along each cut axis. *)

open Value

(** {1:facts Values' facts} *)

val form : ('v, 's, 'd) t -> ('v, 's, 'd) form
val dtype : ('v, 's, 'd) t -> ('v, 's) dtype

val at : ('v, 's, 'd) t -> 'd Devices.placement option
(** [at x] is where [x]'s bytes are, or will be for a traced value: [None] for a
    value of every set. *)

val placement : ('v, 's, 'd) t -> 'd Devices.placement
(** [placement x] is [x]'s placement, as {!at} gives it. Raises
    [Invalid_argument] for a value of every set. *)

val of_arrays :
  'd Devices.placement -> ('v, 's) Nx_array.t iarray -> ('v, 's, 'd) t
(** [of_arrays p arrays] is the live value over [arrays], one per device of [p]:
    [Array] for one, [Shards] otherwise. *)

val live : string
(** [live] is the death word of a live value: values are made with it, so that a
    death claims it by physical comparison. *)

val death : ('v, 's, 'd) t -> string
(** [death x] is [""] while [x] lives, and the function it was donated to, or
    that read its handle, once it died. It reads one atomic word. *)

val alive : by:string -> int -> ('v, 's, 'd) t -> unit
(** [alive ~by i x] raises [Invalid_argument] naming [by], operand [i + 1] and
    the function [x] was donated to, if [x] died. *)

val rank : ('v, 's, 'd) t -> int

val dim : ('v, 's, 'd) t -> int -> int
(** Raises [Invalid_argument] if the axis is not below {!rank}. *)

val same_shape : ('v, 's, 'd) t -> ('w, 'r, 'd) t -> bool
(** [same_shape x y] is [shape x = shape y]. It allocates nothing. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is a fresh array. [dtype], [placement], [rank] and [dim] of a
    value on one device allocate nothing. *)

val has_shape : ('v, 's, 'd) t -> int array -> bool
(** [has_shape x s] is [shape x = s]. It allocates nothing. *)

val merge : int array -> int array -> (int array, int * int * int) result
(** [merge s s'] is the shape [s] and [s'] broadcast to, aligned at their last
    axes, each extent equal or [1]; [Error (a, e, e')] at the first axis [a] of
    that shape where [s] has [e] and [s'] has [e'], neither [1] nor the other.
*)

val broadcast_shape : by:string -> int array -> int array -> int array
(** [broadcast_shape ~by s s'] is [merge s s']'s shape. Raises
    [Invalid_argument] naming [by] and both shapes where they do not broadcast.
*)

val expect : ('v, 's) dtype -> 'd any -> ('v, 's, 'd) t
(** [expect dt (Any x)] is [x] at [dt]'s type. Raises [Invalid_argument] naming
    both dtypes if [x]'s dtype is another. *)

(** {1:operations Operations} *)

type operands =
  | Operands : 'd any list -> operands
      (** An operation's operands, all of one brand. *)

val name : 'r prim -> string
(** [name op] is [op]'s constructor, as ["Map"]. *)

val pp : Format.formatter -> 'r prim -> unit
(** [pp] formats an operation: its name, its programs as expressions over its
    operands [x0], [x1], …, and each operand's dtype, shape and placement. A
    program node read more than once prints once, as [nK = …] before the
    outputs, and as [nK] where it is read. *)

val kind : Nx_kernel.Prog.node -> string
(** [kind n] names [n]'s kind in messages, as ["Exp"] or ["Less"]. *)

val operands : 'r prim -> operands
(** [operands op] is [op]'s operands in order: a map's loads, a contraction's
    [a], [b] then [init], [Check]'s [ok] then its data, the one operand of the
    others. *)

val iteri : ('v 's 'd. int -> ('v, 's, 'd) t -> unit) -> 'r prim -> unit
(** [iteri f op] is [f i x] for each operand [x] of [op], at its position [i] in
    {!operands}. *)

val exists : ('v 's 'd. ('v, 's, 'd) t -> bool) -> 'r prim -> bool
(** [exists f op] is whether [f x] for some operand [x] of [op]. Neither
    allocates for an operation other than [Check]. *)

val map : ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> 'r prim -> 'r prim
(** [map m op] is [op] with each operand [x] replaced by [m x]. *)

val results :
  by:string ->
  ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
  'r prim ->
  'r
(** [results ~by m op] is [op]'s result, its value at position [k] made by
    [m k f], [f] the form eager execution gives it; [()] for [Check].

    Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
    operands break its rule. *)

val program :
  Nx_kernel.Prog.node -> Nx_array.Dtype.any array -> Nx_kernel.Prog.t
(** [program n ins] is {!Nx_kernel.Prog.of_node}[ ~ins n]. Each domain keeps the
    programs it made: a call equal to an earlier one on the domain makes none.

    Raises [Invalid_argument] as {!Nx_kernel.Prog.v} does. A node with a literal
    other than its dtype's zero is not kept. *)

val programs_kept : unit -> int
(** [programs_kept ()] is the number of programs the calling domain keeps. *)

val reductions_list :
  ('d, 'r) reductions ->
  (Nx_kernel.Spec.reduction * int * Nx_array.Dtype.any) list
(** [reductions_list rs] is [rs] as {!Nx_kernel.Spec.reduce} takes them. *)

val reduced : int array -> int array -> int array
(** [reduced s axes] is [s] without [axes]. *)

val op1 :
  by:string ->
  Nx_kernel.Prog.op1 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op2 :
  by:string ->
  Nx_kernel.Prog.op2 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op3 :
  by:string ->
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('v, 's, 'd) t * unit) prim
(** [opN ~by k dt x …] is the map of the one node [k] over [x …], of [x]'s
    shape, with result dtype [dt] ([op3]'s: its second operand's).

    Raises [Invalid_argument] naming [by] where [k] does not take the operands'
    dtypes ({!Nx_kernel.Prog.accepts2} and its siblings). *)

val prepare :
  by:string ->
  ('v 's 'd. 'd Devices.placement option -> ('v, 's, 'd) t -> ('v, 's, 'd) t) ->
  'r prim ->
  'r prim
(** [prepare ~by place op] is [op] with each operand [x] replaced by
    [place p x], [p] where [op]'s route reads [x] ([None] where every operand is
    of every set); [op] itself where none changes. [Place] and [Check] are [op]
    itself.

    Raises [Invalid_argument] naming [by], before [place] is called, where
    [op]'s operands break its rule. *)

(** {1:loads Loads} *)

val load_any : 'd load -> 'd any
(** [load_any l] is the operand [l] reads. *)

val load_dtype : 'd load -> Nx_array.Dtype.any
val is_padded : 'd load -> bool

val spec_load : 'd load -> Nx_kernel.Spec.load
(** [spec_load l] is how a kernel reads [l]. *)

(** {1:families Transforms and factorisations} *)

val transform : ('d, 'r) fft -> Nx_kernel.Spec.transform
val fft_axes : ('d, 'r) fft -> int array
val fft_operand : ('d, 'r) fft -> 'd any

val routine : ('d, 'r) linalg -> Nx_kernel.Spec.routine
(** [routine l] is the routine [l] computes. *)

val routine_name : ('d, 'r) linalg -> string
(** [routine_name l] names [l]'s routine in messages, as ["Cholesky"]. *)

val linalg_operands : ('d, 'r) linalg -> 'd any list
(** [linalg_operands l] is [a], then [b] for a triangular solve. *)

val is_constant : ('v, 's, 'd) t -> bool

val arrays : 'r prim -> 'r -> Nx_array.any array array
(** [arrays op r] is, per result of [op] in position order, the arrays of [r]'s
    value there: one per device of its placement. Raises [Invalid_argument] if a
    result is a constant. *)
