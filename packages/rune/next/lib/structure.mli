(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The messages for two values that should share a structure, and signatures
    uncurried into one argument.

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

(** {1:signatures Signatures}

    A transformation of a curried function of any number of arguments sees it as
    a function of one value: the arguments nested in pairs, the last one alone.
    That value walks as a sequence with argument [k] at [Index k], so a leaf's
    path starts with its argument's position from 0: the window of the second
    argument is at [1.window], and a first argument that is one tensor is at
    [0]. *)

(** The type for signatures as functions of one value. *)
type 'f signature =
  | Signature : {
      args : 'a Nx.Ptree.t;  (** The arguments, as one value. *)
      result : 'r Nx.Ptree.t;  (** The result. *)
      apply : 'f -> 'a -> 'r;
          (** [apply f a] is [f] applied to the arguments [a]. *)
      curry : ('a -> 'r) -> 'f;
          (** [curry g] is the curried function whose arguments [g] takes as one
              value. *)
    }
      -> 'f signature

val uncurry : string -> ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) signature
(** [uncurry fn s] is [s] as a function of one value, for a transformation that
    only reads its arguments.

    Raises [Invalid_argument], naming [fn], if [s] has no argument, or if [s]
    consumes an argument ({!Nx.Ptree.consumes}), as in
    ["Rune.vmap: the argument at 1 is consumed; only a compiled call consumes
     its arguments"]. *)
