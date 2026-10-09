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
(** [at x] is where [x]'s bytes are, or will be for a traced value: [None]
    for a value of every set. *)

val placement : ('v, 's, 'd) t -> 'd Devices.placement
(** [placement x] is [x]'s placement, as {!at} gives it. Raises
    [Invalid_argument] for a value of every set. *)

val rank : ('v, 's, 'd) t -> int

val dim : ('v, 's, 'd) t -> int -> int
(** Raises [Invalid_argument] if the axis is not below {!rank}. *)

val same_shape : ('v, 's, 'd) t -> ('w, 'r, 'd) t -> bool
(** [same_shape x y] is [shape x = shape y]. It allocates nothing. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is a fresh array. [dtype], [placement], [rank] and [dim] of a
    value on one device allocate nothing. *)

val expect : ('v, 's) dtype -> 'd any -> ('v, 's, 'd) t
(** [expect dt (Any x)] is [x] at [dt]'s type. Raises [Invalid_argument] naming
    both dtypes if [x]'s dtype is another. *)

(** {1:operations Operations} *)

type operands =
  | Operands : 'd any list -> operands
      (** An operation's operands, all of one brand. *)

type mapper = { map : 'v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t }
type maker = { make : 'v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t }

val name : 'r prim -> string
(** [name op] is [op]'s constructor, as ["Map"]. *)

val pp : Format.formatter -> 'r prim -> unit
(** [pp] formats an operation: its name, its programs as expressions over its
    operands [x0], [x1], …, and each operand's dtype, shape and placement. *)

val kind : Nx_kernel.Prog.node -> string
(** [kind n] names [n]'s kind in messages, as ["Exp"] or ["Less"]. *)

val operands : 'r prim -> operands
(** [operands op] is [op]'s operands in order: a map's loads, [Check]'s [ok]
    then its data, the one operand of the others. *)

val map : mapper -> 'r prim -> 'r prim
(** [map m op] is [op] with each operand [x] replaced by [m.map x]. *)

val results : by:string -> maker -> 'r prim -> 'r
(** [results ~by m op] is [op]'s result, its value at position [k] made by
    [m.make k f], [f] the form eager execution gives it; [()] for [Check].

    Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
    operands break its rule. *)

val program : Nx_kernel.Prog.node -> Nx_array.Dtype.any array -> Nx_kernel.Prog.t
(** [program n ins] is {!Nx_kernel.Prog.of_node}[ ~ins n]. Each domain keeps
    the programs it made: a call equal to an earlier one on the domain makes
    none.

    Raises [Invalid_argument] as {!Nx_kernel.Prog.v} does. *)

val op1 :
  Nx_kernel.Prog.op1 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op2 :
  Nx_kernel.Prog.op2 ->
  ('w, 'r) dtype ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('w, 'r, 'd) t * unit) prim

val op3 :
  Nx_kernel.Prog.op3 ->
  ('a, 'b, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  (('v, 's, 'd) t * unit) prim
(** [opN k dt x …] is the map of the one node [k] over [x …], of [x]'s shape,
    with result dtype [dt] ([op3]'s: its second operand's). *)

type placer = {
  place :
    'v 's 'd.
    'd Devices.placement option -> ('v, 's, 'd) t -> ('v, 's, 'd) t;
}
(** How the engine makes an operand readable at a placement of its set; [None]
    where every operand of the operation is of every set. *)

val prepare : by:string -> placer -> 'r prim -> 'r prim
(** [prepare ~by pl op] is [op] with each operand [x] replaced by
    [pl.place p x], [p] where [op]'s route reads [x]; [op] itself where none
    changes. [Place] and [Check] are [op] itself.

    Raises [Invalid_argument] naming [by], before [pl] is called, where [op]'s
    operands break its rule. *)

val is_constant : ('v, 's, 'd) t -> bool

val arrays : 'r prim -> 'r -> Nx_array.any array array
(** [arrays op r] is, per result of [op] in position order, the arrays of [r]'s
    value there: one per device of its placement. Raises [Invalid_argument] if a
    result is a constant. *)
