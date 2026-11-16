(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A contraction's launches of contract.hip.

    A plan takes a call nx.amd found nothing to refuse in and lowers it to
    one call ({!Plan}): packs of the operands a kernel does not read where
    they lie; per window of one launch's bounds, a segmented call's
    prepass and a launch of the contraction; and a pack of y out of the
    workspace. Its association follows the dtypes, the scales and segments
    given and the extents alone, never a layout or an alignment. *)

val plan :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:
    (int64, Nx_array.Dtype.int64_elt) Nx_array.t Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  dst:Nx_array.any ->
  init:Nx_array.any option ->
  Nx_kernel.operand ->
  Nx_kernel.operand ->
  (Plan.t option, Nx_array.answer) result
(** [plan ~batch ~contracting ~segments ~acc ~dst ~init a b] plans
    {!Nx_kernel.S.contract} of a call whose shapes fit, [init] of [dst]'s
    dtype: [Ok None] for a [dst] of no element. It is [Error Declined] for
    a scale with an integer or complex [acc]. *)

(** {1:routines The matrix routines' products}

    A routine plans its products into its own call: their launches join
    the builder's, in order, and take pieces of its workspace. *)

type matrix
(** The type for a product's matrices: an array's, or one in the call's
    workspace. *)

val array : Nx_array.any -> matrix
(** [array x] is [x]'s elements through its layout. *)

val work : at:int -> Nx_array.Dtype.any -> Nx_array.Layout.t -> matrix
(** [work ~at dt l] is the matrix of [dt] elements through [l] in the
    workspace of the builder it is planned into, [l]'s offset counted in
    elements from byte [at], a piece {!Plan.take} gave. *)

type options = {
  init_first : bool;
      (** The first range of k sums from [y], the others from [+0]. *)
  neg : bool;  (** [a]'s values negated. *)
  conj_a : bool;  (** [a] conjugated, for a complex [acc]. *)
  conj_b : bool;  (** [b] conjugated, for a complex [acc]. *)
  tri : [ `Lower | `Upper ] option;
      (** Where [Some], only the tiles of a call of one window that hold
          an element of that triangle, the diagonal included, are
          computed; the others' outputs are left as they are. *)
  ranges : int option;
      (** [Some s]: each output's [k'] terms, [k' = 2k] for a complex
          [acc] (its real terms), in [S = min s ⌈k' / 32⌉] ranges, range
          [i] taking the spans of 32 terms from [⌊i r / S⌋] to below
          [⌊(i + 1) r / S⌋], [r = ⌈k' / 32⌉], whatever the tile. [None]
          takes the model's count, which depends on the batch, so a
          routine whose bits must follow each matrix's extents alone
          passes [Some]. *)
}
(** The type for what a routine's product asks beyond
    {!Nx_kernel.S.contract}. *)

val plain : options
(** [plain] asks nothing more: a contraction's association. *)

val ranges : rows:int -> cols:int -> k:int -> int
(** [ranges ~rows ~cols ~k] is a count of ranges for products of [rows] x
    [cols] outputs of [k] terms, a function of these extents alone. *)

val product :
  Plan.builder ->
  options ->
  acc:Nx_array.Dtype.any ->
  init:bool ->
  y:matrix ->
  matrix ->
  matrix ->
  unit
(** [product b o ~acc ~init ~y a m] plans [y <- op(a) op(m)], or [y <- y +
    op(a) op(m)] where [init], into [b]: [a] of shape [batch; rows; k], [m]
    of [batch; k; cols], [y] of [batch; rows; cols], of [acc]'s dtype and
    any layouts, [acc] [float32], [float64], [complex64] or [complex128],
    [op] [o]'s negation and conjugations. Each output's sum is a fused
    multiply-add sum ({!Nx_amd.section-fused}) in the ranges [o.ranges]
    gives, else the model's; with [o.init_first] its first range starts
    from [y] and the ranges' sums are added in increasing k. Its error
    factor is that section's, [γ(K + 1)] or [γ(2K + 1)] for a complex
    [acc], with or without [o.init_first]. It plans nothing for a [y] of
    no element. [y] shares no byte with [a] or [m].

    Raises [Invalid_argument] if an output lies past one launch's
    bounds. *)

val copy : Plan.builder -> src:matrix -> dst:matrix -> unit
(** [copy b ~src ~dst] plans the copy of [src]'s elements into [dst], of
    [src]'s dtype and shape, both of any layout, into [b]. *)

(**/**)

val contract_wide_k : int -> unit
(** [contract_wide_k n] makes a Contracting call hold its contracted
    indices in 8 bytes past a contracted extent of [n], at least 0 and at
    most [2{^31} - 1]. For tests.

    Raises [Invalid_argument] if [n] is out of range. *)

val contract_unit_tile : int option -> unit
(** [contract_unit_tile rows] makes calls on the matrix units take the tile
    of [rows] rows, 16 or 128, and [None] the tile nx_amd.mli states. For
    tests.

    Raises [Invalid_argument] if [rows] is another number. *)
