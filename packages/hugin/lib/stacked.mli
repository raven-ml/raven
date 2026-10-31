(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stacked lengths: the intervals of a mark's rows, which resolving fits and
    drawing reads alike. *)

type t = {
  stacked : int;  (** The index of the stacked binding. *)
  offset : Hugin_kit.Stack.offset;
  keys : int list;
      (** The indices of the bindings that tell columns apart: the other
          position and the facets, those that hold data. *)
}
(** A mark's stacked length. *)

val of_mark : Figure.mark -> t option
(** [of_mark m] is the stacked length of [m], if any. *)

(** The values of a binding that tell columns apart: quantities, or the codes,
    positions or counts of categories and instants. *)
type key = Floats of Nx.host Nx.float64_t | Ints of Nx.host Nx.int64_t

val intervals :
  Hugin_kit.Stack.offset ->
  int array ->
  lengths:Nx.host Nx.float64_t ->
  keys:key list ->
  float array * float array
(** [intervals offset shape ~lengths ~keys] is the starts and ends, one per row
    of a mark of shape [shape] in row-major order, of [lengths] stacked by
    {!Hugin_kit.Stack.intervals} in row order: two rows are in one column iff
    every one of [keys] has equal values at both, [-0.] equal to [0.].
    [lengths] is [nan] at dropped rows, which cover no interval. [lengths] and
    [keys] broadcast to [shape]. *)
