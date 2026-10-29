(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Grids: the implementation of {!Ymir.Grid}, with the corner lattices
    {!Region} weighs and the agreement {!Observation} requires.

    Two grids agree where every static datum is equal and every leaf holds
    the same numbers, NaN equal to NaN. Agreement is one boolean per batch
    element, each leaf's comparison reduced over its own axes. *)

type ('w, 'e, 'd) t = {
  dtype : (float, 'e) Nx.dtype;
  base : int array;
  shape : int array;
  start : 'd Nx.int64_t;
  transform : ('d Transform.plane, 'w, 'd) Transform.t;
  measured : int;
}
(** {!Ymir.Grid.t}. The window of [shape] cells starts at [start],
    [[...; 2]], among the cells of [base]. [measured] counts [transform]'s
    first stages, those of the world the grid was built with, in which its
    measures are taken. {!map_world} appends stages and keeps [measured]. *)

val cell : Ymir_units.Unit.t

val pixels :
  at:'d Nx.Placement.t ->
  shape:int array ->
  (float, 'e) Nx.dtype ->
  ('d Transform.plane, 'w, 'd) Transform.t ->
  ('w, 'e, 'd) t

(** {!Ymir.Grid.kind}. *)
type ('w, 'e, 'd) kind =
  | Pixels : {
      base : int array;
      shape : int array;
      start : 'd Nx.int64_t;
      transform : ('d Transform.plane, 'w, 'd) Transform.t;
    }
      -> ('w, 'e, 'd) kind

val kind : ('w, 'e, 'd) t -> ('w, 'e, 'd) kind
val shape : ('w, 'e, 'd) t -> int array
val centres : ('w, 'e, 'd) t -> 'w
val corners : ('w, 'e, 'd) t -> 'w
val measure : ('w, 'e, 'd) t -> (float, 'e, 'd) Nx.t Ymir_units.Quantity.t
val map_world : ('w, 'v, 'd) Transform.t -> ('w, 'e, 'd) t -> ('v, 'e, 'd) t
val agree : ('w, 'e, 'd) t -> ('w, 'e, 'd) t -> 'd Nx.bool_t
val window : start:'d Nx.int64_t -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val around : 'w -> shape:int array -> ('w, 'e, 'd) t -> ('w, 'e, 'd) t
val ptree : unit -> (('w, 'e, 'd) t, 'd) Nx.Ptree.t

val world : ('w, 'e, 'd) t -> ('w, 'd) Transform.endpoint
(** [world g] is the endpoint of [g]'s world. *)

val mapped : ('w, 'e, 'd) t -> 'd Transform.value
(** [mapped g] is [g]'s corner lattice, [[...; h + 1; w + 1]] points for
    [shape g] [[|h; w|]], mapped to its world by {!Transform.run_cells} with
    two cell axes, where no point outside a domain raises. *)

val cells : ('a, 'b, 'd) Nx.t -> int -> int -> ('a, 'b, 'd) Nx.t
(** [cells v k c] is component [c] of each cell's corner [k], [[...; h; w]],
    for [v] a corner lattice [[...; h + 1; w + 1; n]]. The corners of cell
    [(i, j)] count counter-clockwise in the pixel plane: [(i, j)],
    [(i + 1, j)], [(i + 1, j + 1)] and [(i, j + 1)]. *)

val index_text : int array -> string
(** [index_text i] is the element index [i] for a message, as ["(0, 1)"]. *)

val require : string -> ('w, 'e, 'd) t -> ('v, 'f, 'd) t -> unit
(** [require fn a b] raises [Invalid_argument], naming [fn], where [a] and
    [b] do not {!agree}: immediately for a difference in their static data,
    and through {!Nx.check} where a leaf differs, naming its path and
    element. *)
