(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Descriptors.

    A descriptor holds the attributes of one operation of a family, checked when
    it is made: what a kernel needs beyond its operands' arrays. It is the C
    struct of [nx_spec.h] for its family, held in a string, so that C kernels
    read it in place and equal descriptors are equal strings. *)

type 'f t
(** The type for descriptors of the family ['f]. *)

val shapes : 'f t -> int array array -> (int array array, string) result
(** [shapes s ins] is the shapes of [s]'s results for operands of the shapes
    [ins], in the order the family's kernel takes them, or why they do not fit
    [s]. *)

(** {1:contract Contractions} *)

type contract
(** The family of contractions. *)

val contract :
  batch:(int * int) array ->
  contracting:(int * int) array ->
  acc:Nx_array.Dtype.any ->
  out:Nx_array.Dtype.any ->
  init:bool ->
  contract t
(** [contract ~batch ~contracting ~acc ~out ~init] contracts an operand [a] with
    an operand [b]. Each pair of [batch] and [contracting] is an axis of [a] and
    an axis of [b] of one extent; the other axes are free. The result's axes are
    the batch axes in pair order, then [a]'s free axes, then [b]'s, in axis
    order:
    {[
    y[B, I, J] = out (init[B, I, J] + Σ_K a[B, I, K] · b[B, K, J])
    ]}
    computed in [acc] and rounded once to [out]. Without [init] the sum starts
    from [+0]. Integers wrap in [acc]. A float [acc] errs by at most twice its
    unit roundoff per addition, and the order of the sum is a function of the
    shapes, never of layouts, threads or the device's size.

    Raises [Invalid_argument] if an axis is negative or not below
    {!Nx_array.Layout.max_rank}, if an axis of [a] or of [b] is in two pairs, or
    if [acc] is a boolean or a float narrower than [float32]. *)

val batch : contract t -> (int * int) array
(** [batch s] is [s]'s batch pairs. *)

val contracting : contract t -> (int * int) array
(** [contracting s] is [s]'s contracting pairs. *)

val acc : contract t -> Nx_array.Dtype.any
(** [acc s] is the dtype [s] computes in. *)

val out : contract t -> Nx_array.Dtype.any
(** [out s] is [s]'s result's dtype. *)

val init : contract t -> bool
(** [init s] is [true] iff [s] adds an operand [init] to its sum. *)

(** Contractions with their axes grouped.

    A view groups a contraction's axes into one of each kind, where every
    operand lays out each group's axes as one run of strides, as a matrix
    product reads them. *)
module Contract_view : sig
  type 'f spec := 'f t

  type t
  (** The type for views, rewritten by {!fill}. A C kernel reads a view as
      [nx_spec.h]'s [nx_contract_view]: its value is bytes that begin with
      that struct. *)

  (** The type for a contraction's operands and its result. *)
  type operand = A | B | Init | Dst

  (** The type for a view's axes. [A] has [Batch], [Row] and [Contracted]; [B]
      has [Batch], [Column] and [Contracted]; [Init] and [Dst] have [Batch],
      [Row] and [Column]. *)
  type axis = Batch | Row | Column | Contracted

  val make : unit -> t
  (** [make ()] is a view to {!fill}. *)

  val fill :
    t -> contract spec -> dst:Nx_array.any -> Nx_array.any array -> bool
  (** [fill v s ~dst ops] groups the axes of [ops], ordered as
      {!Nx_kernel.S.contract} takes them, and of [dst] into [v]. It is [false]
      if a group does not merge into one axis, and [v] is then unspecified. It
      allocates nothing.

      Raises [Invalid_argument] if [ops] and [dst] do not fit [s]: another
      number of operands, ranks the pairs do not fit, or extents that differ
      within a group. *)

  val extent : t -> axis -> int
  (** [extent v x] is [v]'s extent along [x]: [1] for a group of no axis. *)

  val offset : t -> operand -> int
  (** [offset v o] is [o]'s first element, in elements from its buffer's first
      byte.

      Raises [Invalid_argument] for [Init] in a view without it. *)

  val stride : t -> operand -> axis -> int
  (** [stride v o x] is [o]'s stride along [x], in elements.

      Raises [Invalid_argument] for an axis [o] does not have, and for [Init] in
      a view without it. *)
end
