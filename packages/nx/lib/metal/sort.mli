(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sorts and numberings: plans of the sort kernels (nx_metal.mli's Sorts and
    numbering).

    A sort takes one of three paths by its slices' length and the elements it
    keeps: a block sort in a group's threadgroup memory, a radix select of the
    kept elements then their block sort, or an LSD radix sort through workspace.
    A numbering sorts its rows a word at a time by the radix passes, then
    numbers them. Every launch's estimate is within {!Plan.allowed}: work that
    grows with the extents runs as launches of ranges of its units. *)

val sort :
  holds:(Kernels.t -> int) ->
  axis:int ->
  descending:bool ->
  values:('v, 's) Nx_array.t ->
  positions:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [sort ~holds ~axis ~descending ~values ~positions x] plans the sort of [x]'s
    slices along [axis] into [values] and [positions], [holds] as {!Loop.map}
    takes it. It is [Ok None] where no slice keeps an element, and
    [Error Declined] where the slices sort in a block or by radix and the values
    or positions share a byte with [x], or where one unit of every path's work
    passes {!Plan.allowed}[ ()].

    nx.metal has checked the call: its shapes fit. *)

val unique :
  holds:(Kernels.t -> int) ->
  ids:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  first:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  counts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  count:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [unique ~holds ~ids ~first ~counts ~count keys] plans the numbering of
    [keys]' rows, [holds] as {!Loop.map} takes it. It is [Error Declined] where
    one unit of its work passes {!Plan.allowed}[ ()].

    nx.metal has checked the call: its shapes fit. *)

(**/**)

val of_facts :
  holds:(Kernels.t -> int) ->
  axis:int ->
  descending:bool ->
  shares:bool ->
  values:Operands.operand ->
  positions:Operands.operand ->
  Operands.operand ->
  (Plan.t option, Nx_array.answer) result
(** [of_facts ~holds ~axis ~descending ~shares ~values ~positions x] is
    {!sort}'s plan of a call whose values, positions and [x] have those facts,
    [shares] whether [x] shares a byte with the values or positions. It reads no
    buffer, so tests plan calls of any length. *)

val bound_sorts : parts:int -> wide:int -> unit
(** [bound_sorts ~parts ~wide] makes a radix pass cut a slice into at most
    [parts] parts in place of {!Kernels.sort_parts}, and a slice of at least
    [2{^wide}] elements take 64-bit positions and counts whose low words carry
    at bit [wide], in place of [2{^32}] and 32. It applies to calls planned
    after it returns. For tests.

    Raises [Invalid_argument] if [parts] is not in \[1; {!Kernels.sort_parts}\]
    or [wide] not in \[1; 32\]. *)

(** The type for the plan of a sort whose slices step by more than one element:
    by the lesser estimate, read where they lie, or copied contiguous first. *)
type outer = Estimate | Strided | Transposed

val bound_outer : outer -> unit
(** [bound_outer o] makes a sort of elements of a byte or more, whose slices
    step by more than one element, plan [o]: [Strided] reads them where they
    lie, [Transposed] copies them contiguous into workspace first, and
    [Estimate], the initial value, takes the plan of the lesser estimate where
    no two of x's elements share a line along any axis. It applies to calls
    planned after it returns. For tests. *)

val bound_positions : int -> unit
(** [bound_positions p] makes a select's rounds and block sort take at least [p]
    bytes of position, in place of 1, so that tests reach 8-byte positions with
    small arrays. It applies to calls planned after it returns. For tests.

    Raises [Invalid_argument] if [p] is not in \[1; 8\]. *)

(**/**)
