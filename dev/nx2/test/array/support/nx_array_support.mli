(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx2's suites and benches share. *)

val row : int -> (string * int * int) option
(** [row code] is the row of the dtype [code] in [nx_dtype.h], as its name, bits
    and kind (the index of the kind in [enum nx_kind]), or [None] if no dtype
    has the code. *)

val decode : int -> int -> int
(** [decode code c] is the binary32 bits of the code [c] of the narrow float
    dtype [code], read by [nx_dtype.h]'s decoder, which kernels call. *)

val of_int64 : int -> int64 -> int
(** [of_int64 code v] is the bits a store of [v] writes into an element of the
    narrow float dtype [code], through [nx_dtype.h]'s conversions from 64-bit
    integers. *)

val of_uint64 : int -> int64 -> int
(** [of_uint64 code v] is {!of_int64} for the uint64 whose bits [v] holds. *)

val answers : (string * Nx_array.answer) list
(** [answers] is every answer, in its type's order, beside its name. *)

val pp_answer : Format.formatter -> Nx_array.answer -> unit
(** [pp_answer] formats an answer's name. *)

val codes : unit -> int array
(** [codes ()] is [nx_array.h]'s codes in the order of {!Nx_array.answer}'s
    constructors: [NX_OK], [NX_DECLINED], [NX_DTYPE] and on. *)

val code : Nx_array.answer -> int
(** [code a] is the integer OCaml represents [a] by, which C reads. *)

val layout : Nx_array.Layout.t -> int array
(** [layout l] is [l]'s fields as C reads them through [nx_layout.h]'s field
    order: rank, flags, offset, lo, hi, then the extents and the strides. *)

val add : 'z -> 'x -> 'y -> Nx_array.answer
(** [add z x y] is a float32 kernel through [nx_read] and [nx_coalesce]: it
    stores [x + y] into [z] and answers as a kernel does. Its operands are
    untyped, as an array built from parts can be. *)

val read_all : Nx_array.any array -> Nx_array.answer
(** [read_all arrays] reads [arrays] through [nx_read], the first for writing
    and the others for reading, and ends the read it admits: [nx_read]'s
    answer. At most 16 arrays. *)

val copy_into :
  ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> Nx_array.answer
(** [copy_into dst src] is the gather {!Nx_array.copy} runs, into [dst], any
    written operand of [src]'s dtype and shape: it copies [src]'s elements into
    [dst] through [nx_read] and answers as a kernel does. *)

val of_array_into : ('v, 's) Nx_array.t -> 'v array -> Nx_array.answer
(** [of_array_into a xs] is the store {!Nx_array.of_array} runs, into [a], any
    array of [Array.length xs] elements: it writes [xs] in C order of indices
    through [nx_read] and answers as a kernel does. *)

val collect : ('v, 's) Nx_array.t -> Nx_array.answer
(** [collect a] reads [a] through [nx_read], empties the minor heap and compacts
    the major one while it holds the read, then calls [nx_done]. *)

val int16_at :
  int ->
  int ->
  (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Genarray.t
(** [int16_at k n] is a bigarray of [n] int16, at most 120, over memory [k]
    bytes past a 16-byte boundary, [k] at most 16: misaligned for its elements
    if [k] is odd. Every call shares the same memory. *)

val io_device : unit -> Rig.t
(** [io_device ()] is an io device over bigarrays: memory the host does not
    address. *)

val read_only : int -> Rig.Buffer.t
(** [read_only n] is a buffer of [n] bytes on {!io_device} whose memory is
    [Read]. *)

val io_allocations : unit -> int
(** [io_allocations ()] is the number of allocations {!io_device}'s memory has
    made so far. *)

(** A device over host memory whose work runs only when a wait sleeps on it, as
    a device still running work does until the host waits for it. A submit
    queues its fills; its driver's sleep, or its stop, runs every queued fill in
    order and then shows the last value in its word. The host addresses its
    memory. *)
module Late : sig
  type t
  (** The type for a Late device's driver state. *)

  val open_ : string -> Rig.t * t
  (** [open_ name] opens a fresh Late device named [name]. *)

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s sleeps raise its driver's [Fault why] from now
      on, which loses the device in the wait that sleeps. *)
end

val write : Rig.Buffer.t -> string -> unit
(** [write b s] submits, on [b]'s device, work that writes [b] and stores the
    bytes of [s] at [b]'s first byte, a Late device's fill. On a Late device the
    bytes are there once a wait for [b] slept. *)
