(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What a plan reads of a call's arrays.

    A call's {e operands} are its destinations, then its loads, then its carry,
    numbered from [0] in that order. A plan reads them only through these facts,
    never a buffer: it names an operand's buffer by the operand's number
    ({!Plan.Operand}), which the door binds. So two calls of one program whose
    facts are equal ({!val-key}) plan alike, and the plan of one is the plan of
    the other. *)

type pad = {
  pad : Nx_kernel.pad;
  fill : string;  (** The fill's bits, {!Nx_kernel.Prog.bits}. *)
}
(** The type for a padded load's padding. *)

type operand = {
  dtype : Nx_array.Dtype.any;
  layout : Nx_array.Layout.t;
  buffer : int;
      (** The number of the call's first operand of this operand's buffer, by
          physical equality: its own where none before it shares it. *)
  skew : int;
      (** The bytes of the buffer's memory before its first byte, modulo 4:
          where in a 32-bit word of its allocation it starts. *)
  residue : int;
      (** The address of the buffer's first byte in the addresses of its
          device's work, modulo 32. *)
  pad : pad option;  (** A padded load's padding. *)
}
(** The type for an operand's facts. *)

type t = private {
  dsts : operand array;  (** Operands [0] to [n - 1]. *)
  loads : operand array;  (** Operands [n] to [n + m - 1]. *)
  carry : operand option;  (** Operand [n + m]. *)
  overlaps : bool array;
      (** Load [j]'s buffer and destination [k]'s overlap
          ({!Rig.Buffer.overlaps}) at [j * n + k]. *)
}
(** The type for a call's facts. *)

val residue : Rig.Buffer.t -> int
(** [residue b] is the address of [b]'s first byte in the addresses of its
    device's work, modulo 32, as {!operand}'s [residue] holds it.

    Raises [Invalid_argument] if [b] is dead, as {!Rig.Buffer.aligned} does. *)

val v :
  ?carry:Nx_array.any -> dsts:Nx_array.any iarray -> Nx_kernel.load iarray -> t
(** [v ~carry ~dsts loads] is the facts of a call of [dsts], [loads] and
    [carry].

    Raises [Invalid_argument] if a buffer is dead, as {!Rig.Buffer.aligned}
    does: the door refuses dead buffers first. *)

val key : t -> int array
(** [key t] is every fact of [t], as ints: two calls' facts are equal if and
    only if their keys are. *)

val hash : int array -> int
(** [hash k] is a hash of the key [k], non-negative, whose low bits depend on
    every field of [k]: keys that differ fall in [2{^b}] buckets by their low
    [b] bits about as uniform draws do. *)

val overlaps : t -> load:int -> dst:int -> bool
(** [overlaps t ~load ~dst] is [t.overlaps]'s entry of [load] and [dst]. *)

val ones : t -> t
(** [ones t] is the facts of [t]'s operands at their ranks with one element, at
    stride [0], each of a buffer of its own that starts on 16 bytes, and no load
    padded: the call of one element a call of no element answers as. *)
