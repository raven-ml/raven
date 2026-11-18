(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Launches of the cubin's kernels and the calls that submit them.

    A plan lowers a call nx.cuda found nothing to refuse in to calls, each
    one submission of its launches in order: launches that share the
    device's workspace or tickets are one call's, so that no other call's
    work comes between them. A launch names its kernel, its geometry, its
    parameters and its refs, the parameter bytes that hold a byte offset
    into a slot's buffer. A call names the buffer of each slot, the
    stand-ins {!workspace} and {!tickets} where it uses the device's, and
    the bytes it needs of them.

    Calls whose launches have the same kernels, parameter sizes and refs
    take parts of one submission ({!key}), so a device keeps one submission
    for each, which a later call submits with its launches' values. *)

type t = {
  kernel : int;  (** The kernel's index in {!Kernels.kernels}. *)
  size : int;  (** The parameter bytes the kernel's part takes. *)
  base : int;  (** The slot [refs]' slots count from. *)
  refs : int array;
      (** Triples of a parameter's byte, a slot from [base] and [1] if the
          launch writes the slot, else [0]. *)
  args : Rig.Submission.args;
      (** Its groups along x, y and z, its threads along x, its dynamic
          shared bytes, and its parameters, little-endian, as rig reads
          them. *)
}
(** The type for launches. *)

type call = {
  launches : t array;
  buffers : Rig.Buffer.t array;  (** The buffer of each slot. *)
  workspace : int;
  tickets : int;
      (** The bytes of the device's workspace and tickets that the slots of
          {!val-workspace} and {!val-tickets} take. *)
  kept : bool;
      (** Whether the device keeps the call's submission. A plan whose
          launches grow with its extents makes calls of its own. *)
  frees : Rig.Buffer.t list;
      (** Buffers the plan made for its own work, freed once all its calls
          are submitted, so that their memory returns when the work is
          done. *)
}
(** The type for calls, one submission each. *)

type plan = (call list, Nx_array.answer) result
(** The type for plans: their calls in order, none for no work, or the
    answer of a call that does not compute. *)

val workspace : Rig.Buffer.t
(** [workspace] stands for the device's workspace in a call's [buffers]:
    bytes of no value at the call's start, which no other call writes while
    the call's submission runs. *)

val tickets : Rig.Buffer.t
(** [tickets] stands for the device's tickets in a call's [buffers]: zero
    words, which every launch leaves zero. *)

val kernel : string -> int
(** [kernel name] is [name]'s index in {!Kernels.kernels}.

    Raises [Invalid_argument] if there is none. *)

val slot_refs : header:int -> reads:int -> writes:int -> int array
(** [slot_refs ~header ~reads ~writes] is the refs of a launch over [reads]
    read slots then [writes] written ones, whose parameters hold slot [s]'s
    address in word [header + s]. *)

val args : call -> Rig.Submission.args array
(** [args c] is the values of [c]'s launches, in order, as a submit of
    [c]'s submission passes them. *)

val key : call -> int array
(** [key c] names [c]'s submission: equal for calls whose launches have the
    same kernels, parameter sizes and refs, in the same order. *)

val matches : int array -> call -> bool
(** [matches k c] is [key c = k], and allocates nothing. *)

val parts : Rig.Image.t -> call -> Rig.Submission.part array
(** [parts i c] is the parts of [c]'s submission, of [i]'s kernels.

    Raises [Invalid_argument] if [i] lacks one. *)
