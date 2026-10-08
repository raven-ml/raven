(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx2's suites and benches share. *)

val row : int -> (string * int * int) option
(** [row code] is the row of the dtype [code] in [nx_dtype.h], as its name, bits
    and kind (the index of the kind in [enum nx_kind]), or [None] if no dtype
    has the code. *)

val layout : Nx_array.Layout.t -> int array
(** [layout l] is [l]'s fields as C reads them through [nx_layout.h]'s field
    order: rank, flags, offset, lo, hi, then the extents and the strides. *)

val add : 'z -> 'x -> 'y -> int
(** [add z x y] is a float32 kernel through [nx_read] and [nx_coalesce]: it
    stores [x + y] into [z] and answers [nx_array.h]'s code. Its operands are
    untyped, as an array built from parts can be. *)

val add_noalloc : 'z -> 'x -> 'y -> int
(** [add_noalloc] is {!add} as a [[@@noalloc]] external. *)

val collect : ('v, 's) Nx_array.t -> int
(** [collect a] reads [a] through [nx_read], empties the minor heap and compacts
    the major one while it holds the read, then calls [nx_done]. *)

val io_device : unit -> Rig.t
(** [io_device ()] is an io device over bigarrays: memory the host does not
    address. *)

val io_allocations : unit -> int
(** [io_allocations ()] is the number of allocations {!io_device}'s memory has
    made so far. *)
