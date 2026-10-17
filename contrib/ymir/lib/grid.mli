(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Grids: the implementation of {!Ymir.Grid}, with the corner lattices
    {!Region} weighs and the agreement {!Observation} requires.

    Two grids agree where every static datum is equal and every leaf holds
    the same numbers, NaN equal to NaN. Agreement is one boolean per batch
    element, each leaf's comparison reduced over its own axes. *)

type ('w, 'e) t = {
  dtype : (float, 'e) Nx.dtype;
  base : int array;
  shape : int array;
  start : Nx.int64_t;
  transform : (Transform.plane, 'w) Transform.t;
  measured : int;
}
(** {!Ymir.Grid.t}. The window of [shape] cells starts at [start],
    [[...; 2]], among the cells of [base]. [measured] counts [transform]'s
    first stages, those of the world the grid was built with, in which its
    measures are taken. {!map_world} appends stages and keeps [measured]. *)

val cell : Ymir_units.Unit.t

val pixels :
  shape:int array ->
  (float, 'e) Nx.dtype ->
  (Transform.plane, 'w) Transform.t ->
  ('w, 'e) t

(** {!Ymir.Grid.kind}. *)
type ('w, 'e) kind =
  | Pixels : {
      base : int array;
      shape : int array;
      start : Nx.int64_t;
      transform : (Transform.plane, 'w) Transform.t;
    }
      -> ('w, 'e) kind

val kind : ('w, 'e) t -> ('w, 'e) kind
val shape : ('w, 'e) t -> int array
val centres : ('w, 'e) t -> 'w
val corners : ('w, 'e) t -> 'w
val measure : ('w, 'e) t -> (float, 'e) Nx.t Ymir_units.Quantity.t
val map_world : ('w, 'v) Transform.t -> ('w, 'e) t -> ('v, 'e) t
val agree : ('w, 'e) t -> ('w, 'e) t -> Nx.bool_t
val window : start:Nx.int64_t -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
val around : 'w -> shape:int array -> ('w, 'e) t -> ('w, 'e) t
val ptree : unit -> ('w, 'e) t Nx.Ptree.t

val world : ('w, 'e) t -> 'w Transform.endpoint
(** [world g] is the endpoint of [g]'s world. *)

val mapped : ('w, 'e) t -> Transform.value
(** [mapped g] is [g]'s corner lattice, [[...; h + 1; w + 1]] points for
    [shape g] [[|h; w|]], mapped to its world by {!Transform.run_cells} with
    two cell axes, where no point outside a domain raises. *)

val cells : ('a, 'b) Nx.t -> int -> int -> ('a, 'b) Nx.t
(** [cells v k c] is component [c] of each cell's corner [k], [[...; h; w]],
    for [v] a corner lattice [[...; h + 1; w + 1; n]]. The corners of cell
    [(i, j)] count counter-clockwise in the pixel plane: [(i, j)],
    [(i + 1, j)], [(i + 1, j + 1)] and [(i, j + 1)]. *)

val index_text : int array -> string
(** [index_text i] is the element index [i] for a message, as ["(0, 1)"]. *)

val require : string -> ('w, 'e) t -> ('v, 'd) t -> unit
(** [require fn a b] raises [Invalid_argument], naming [fn], where [a] and
    [b] do not {!agree}: immediately for a difference in their static data,
    and through {!Nx.check} where a leaf differs, naming its path and
    element. *)
