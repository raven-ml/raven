(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx2's suites and benches share. *)

val row : int -> (string * int * int) option
(** [row code] is the row of the dtype [code] in [nx_dtype.h], as its name, bits
    and kind (the index of the kind in [enum nx_kind]), or [None] if no dtype
    has the code. *)

val of_int64 : int -> int64 -> int
(** [of_int64 code v] is the bits a store of [v] writes into an element of the
    narrow float dtype [code], through [nx_dtype.h]'s conversions from 64-bit
    integers. *)

val of_uint64 : int -> int64 -> int
(** [of_uint64 code v] is {!of_int64} for the uint64 whose bits [v] holds. *)

val layout : Nx_array.Layout.t -> int array
(** [layout l] is [l]'s fields as C reads them through [nx_layout.h]'s field
    order: rank, flags, offset, lo, hi, then the extents and the strides. *)

val add : 'z -> 'x -> 'y -> int
(** [add z x y] is a float32 kernel through [nx_read] and [nx_coalesce]: it
    stores [x + y] into [z] and answers [nx_array.h]'s code. Its operands are
    untyped, as an array built from parts can be. It is a [[@@noalloc]]
    external, as a kernel that keeps the domain lock is. *)

val copy_into : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
(** [copy_into dst src] is the gather {!Nx_array.copy} runs, into [dst], any
    written operand of [src]'s dtype and shape: it copies [src]'s elements into
    [dst] through [nx_read] and answers [nx_array.h]'s code. *)

val collect : ('v, 's) Nx_array.t -> int
(** [collect a] reads [a] through [nx_read], empties the minor heap and compacts
    the major one while it holds the read, then calls [nx_done]. *)

val io_device : unit -> Rig.t
(** [io_device ()] is an io device over bigarrays: memory the host does not
    address. *)

val io_allocations : unit -> int
(** [io_allocations ()] is the number of allocations {!io_device}'s memory has
    made so far. *)
