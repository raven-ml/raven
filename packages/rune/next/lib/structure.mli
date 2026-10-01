(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The messages for two values that should share a structure.

    Every message starts with the entry point that raises it, [fn], such as
    ["Rune.vjp"], and names a tensor by its path ({!Nx.Ptree.Path.to_string}),
    the root as ["the root"]. [this] and [that] name the two values compared,
    such as ["the result"] and ["the cotangents"]. *)

val check :
  string -> 's Nx.Ptree.t -> this:string -> 's -> that:string -> 's -> unit
(** [check fn s ~this x ~that y] is [()] iff [x] and [y] have equal visits
    ({!Nx.Ptree.visits}) and their tensors at each path equal dtypes.

    Raises [Invalid_argument] otherwise, naming the first visit where they
    differ and what each holds there, as in
    ["Rune.scan: 1: length 3 in the carry the body returned, length 2 in the
     carry it received"], or the path and both dtypes, as in
    ["Rune.scan: 0: float32 in the carry the body returned, float64 in the carry
     it received"]. *)

val map2 :
  string ->
  's Nx.Ptree.t ->
  this:string ->
  that:string ->
  ('a 'b. Nx.Ptree.Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) ->
  's ->
  's ->
  's
(** [map2 fn s ~this ~that f x y] is [x] with each tensor [t] at path [p]
    replaced by [f p t u], where [u] is [y]'s tensor at [p], applied in walk
    order.

    Raises [Invalid_argument] as {!check} does, before applying [f], when [x]
    and [y] differ in their visits; and at [p], before applying [f] there, when
    [t] and [u] differ in dtype or shape, as in
    ["Rune.vjp: 0.w: shape [3] in the result, [2] in the cotangents"]. *)
