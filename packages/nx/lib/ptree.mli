(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Structures of tensors.

    A {e structure} is a type ['a t] with one function, [walk], that walks the
    parts of a value with a {!Walk.cursor}: the positions of its type parameter,
    the tensors of a fixed type, and the data a compiled program depends on (a
    window, a list's length, a variant's case). Every other operation is derived
    from that function and walks in its order: maps and folds, casts, checkpoint
    names, the lists of tensors that transformations trace, and the keys under
    which compiled programs are cached.

    The intended path:
    + Write a structure: a module of type {!module-type:S} whose [walk] uses the
      {!Walk} functions, one per kind of part.
    + Make a structure at one type, a {!type-t}, with {!instantiate}, {!nest}
      and the combinators {!tensor}, {!unit}, {!pair}, {!option}, {!list} and
      {!iso}. Transformations, optimisers and checkpoints take a {!type-t}.
    + Map, zip and fold its tensors with {!map}, {!map2} and {!fold}, and test
      its walk with {!visits}.
    + Change the payload's type with {!cast} and {!Payload}, which take the
      structure's module.
    + Describe a compiled function's arguments and result with a signature,
      {!type-fn}.

    {!flatten}, {!rebuild} and {!Skeleton} are the low-level operations that
    transformations and optimisers are built on.

    {[
    type 'a linear = { w : 'a; b : 'a option }

    module Linear = struct
      type 'a t = 'a linear

      let walk c { w; b } =
        let open Nx.Ptree.Walk in
        let w = field c "w" leaf w in
        let b = field c "b" (option leaf) b in
        { w; b }
    end

    let linear = Nx.Ptree.instantiate (module Linear)
    let zeros = Nx.Ptree.map linear (fun _ x -> Nx.zeros_like x) params
    ]}

    See doc/06-structures.md for writing a structure, masks by path, payload
    maps and testing a structure with {!visits}. *)

(** {1:paths Paths} *)

(** Paths from the root of a value to one of its parts. *)
module Path : sig
  (** The type for path segments. *)
  type seg =
    | Field of string  (** A named part, such as a record field. *)
    | Index of int
        (** A numbered part: a list or array element, a tuple's component or a
            pair's side. *)

  type t
  (** The type for paths. The root path has no segments. [Some] and a variant's
      case add no segment. *)

  val segments : t -> seg list
  (** [segments p] is [p]'s segments, from the root. *)

  val equal : t -> t -> bool
  (** [equal p q] is [true] iff [p] and [q] have equal segments. The path of the
      one field ["a.b"] and the path of [a] then [b] are not equal, although
      they print alike. *)

  val to_string : t -> string
  (** [to_string p] is [p]'s segments joined with ["."], such as
      ["blocks.3.fc.b"]. It is the name a checkpoint gives the part at [p]. The
      root is [""]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf p] formats [p] as {!to_string} does. *)
end

(** {1:walk Writing a structure} *)

type 's t
(** The type for structures of values of type ['s]. Transformations, optimisers
    and checkpoints take one; {!section-structures} builds one, and
    {!Walk.structure} walks one inside a [walk]. *)

(** The functions a structure's [walk] is written with.

    Each function walks one kind of part at the cursor's path. The contract a
    [walk] keeps with them is {!module-type:S}'s.

    A sub-structure is walked by its own [walk], and a variant names its case
    before its other parts:

    {[
    type 'a weight =
      | Float of 'a
      | Mxfp4 of { blocks : Nx.uint8_t; scales : Nx.uint8_t }

    let walk_weight c =
      let open Nx.Ptree.Walk in
      function
      | Float w ->
          case c "float";
          Float (leaf c w)
      | Mxfp4 { blocks; scales } ->
          case c "mxfp4";
          let blocks = field c "blocks" tensor blocks in
          let scales = field c "scales" tensor scales in
          Mxfp4 { blocks; scales }
    ]} *)
module Walk : sig
  type ('a, 'b) cursor
  (** The type for cursors of a walk that turns the parameter's positions from
      ['a] into ['b]. A cursor stands at a path of the value being walked. *)

  val field :
    ('a, 'b) cursor -> string -> (('a, 'b) cursor -> 's -> 't) -> 's -> 't
  (** [field c name walk x] is [walk c' x], where [c'] stands at [c]'s path
      extended with [Field name]. *)

  val index :
    ('a, 'b) cursor -> int -> (('a, 'b) cursor -> 's -> 't) -> 's -> 't
  (** [index c i walk x] is [walk c' x], where [c'] stands at [c]'s path
      extended with [Index i]. Tuples and arrays number their parts with it; an
      array also reports its length with {!int}. *)

  val leaf : ('a, 'b) cursor -> 'a -> 'b
  (** [leaf c x] walks [x], a position of the type's parameter, at [c]'s path.
  *)

  val tensor : ('a, 'b) cursor -> ('x, 'y) Nx_effect.t -> ('x, 'y) Nx_effect.t
  (** [tensor c x] walks [x], a tensor of a fixed type, at [c]'s path.
      Operations at the structure's one type, such as {!Nx.Ptree.map} and the
      transformations, treat it as any other tensor; {!Nx.Ptree.Payload}
      operations and {!Nx.Ptree.cast} return it unchanged. *)

  val int : ('a, 'b) cursor -> int -> int
  (** [int c n] reports [n] at [c]'s path and is [n]. Compiled programs are
      cached per reported integer, so an integer that changes on every call
      compiles a program per value. A bool is reported as [0] or [1]. *)

  val case : ('a, 'b) cursor -> string -> unit
  (** [case c tag] reports the variant case [tag] at [c]'s path. *)

  val option :
    (('a, 'b) cursor -> 's -> 't) -> ('a, 'b) cursor -> 's option -> 't option
  (** [option walk c x] reports whether [x] is present at [c]'s path and, if it
      is, walks its content with [walk] at the same path. *)

  val list :
    (('a, 'b) cursor -> 's -> 't) -> ('a, 'b) cursor -> 's list -> 't list
  (** [list walk c l] reports [l]'s length at [c]'s path and walks its [i]-th
      element with [walk] at [c]'s path extended with [Index i], from the first.
  *)

  val structure : 's t -> ('a, 'b) cursor -> 's -> 's
  (** [structure s c x] walks [x], a value of the structure [s], at [c]'s path:
      each of [s]'s tensors as {!tensor} walks one and each of its reports as
      [s] makes it, at [s]'s paths extended from [c]'s. [tensor c x] is
      [structure Nx.Ptree.tensor c x].

      It walks a part that has a structure at one type and no module, such as an
      optimiser state over a model or a cache index. A record of such parts is a
      structure without a parameter:

      {[
      type train = {
        params : Nx.float32_t Mlp.t;
        opt : Nx.float32_t Mlp.t Vega.adam_state;
        scale : Vega.Loss_scale.t;
      }

      module Train = struct
        type _ t = train

        let walk c t =
          let open Nx.Ptree.Walk in
          let params = field c "params" (structure mlp) t.params in
          let opt = field c "opt" (structure adam) t.opt in
          let scale =
            field c "scale" (structure Vega.Loss_scale.ptree) t.scale
          in
          { params; opt; scale }
      end
      ]}

      where [mlp] is [Nx.Ptree.instantiate (module Mlp)] and [adam] is
      [Vega.adam_ptree mlp]. Its tensors are at [params.l1.w], ...,
      [opt.mu.l1.w], ..., [opt.step] and [scale.scale]. {!Nx.Ptree.cast} and
      {!Nx.Ptree.Payload} operations keep them, as they keep fixed tensors. *)
end

(** Structures.

    [walk c x] walks every part of [x] once and rebuilds [x] from what each walk
    returns. The contract:
    - each position of the parameter is walked with {!Walk.leaf}, each tensor of
      a fixed type with {!Walk.tensor}, and each sub-structure with its own
      [walk], or with {!Walk.structure} when it has a {!type-t} and no module;
    - each integer that changes what a program computes (a window, a block
      shape, a layer kind) is reported with {!Walk.int}; a bool is an integer,
      and other scalar data a program depends on belongs in a tensor;
    - each variant reports its case with {!Walk.case} before any other part of
      the case, with a tag no other case of the type uses;
    - options and lists are walked with {!Walk.option} and {!Walk.list}, which
      report presence and length, so a part with no tensors still counts;
    - a dictionary reports its size with {!Walk.int}, and each key with
      {!Walk.case} before walking that key's value with {!Walk.field};
    - [walk] rebuilds what it walks: a walk that returns each part unchanged
      rebuilds an equal value.

    The type checker ensures that no position of the parameter is forgotten,
    since {!Walk.leaf} changes its type. A forgotten fixed tensor, integer or
    case compiles; a compiled function then freezes the tensor into its first
    program, or replays a program traced for other data. A test of the
    structure's {!visits} catches each of them.

    Every operation runs the same [walk], so all of them see one order. A record
    literal evaluates its fields in an order OCaml leaves unspecified; write the
    fields as a sequence of [let]s where the order must show.

    A structure without a parameter is [type _ t = r], whose [walk] walks every
    tensor with {!Walk.tensor}. *)
module type S = sig
  type 'a t
  (** The type for values of the structure with parameter ['a]. *)

  val walk : ('a, 'b) Walk.cursor -> 'a t -> 'b t
  (** [walk c x] walks every part of [x] with [c] and is [x] rebuilt from what
      the walks return. *)
end

(** {1:structures Structures at one type} *)

val instantiate : (module U : S) -> ('a, 'b) Nx_effect.t U.t t
(** [instantiate (module U)] is [U] with tensors at its parameter's positions:
    [nest (module U) tensor]. A binding is at one dtype and needs an annotation
    only when nothing in its compilation unit fixes the dtype. *)

val nest : (module U : S) -> 's t -> 's U.t t
(** [nest (module U) s] is [U] whose parameter's positions are values of
    structure [s]. The value at path [p] is walked by [s] with its paths
    extended from [p]. Optimiser states and records of models are nested. *)

val tensor : ('a, 'b) Nx_effect.t t
(** [tensor] is the structure of one tensor, at the root path. *)

val unit : unit t
(** [unit] is the structure of [()], which has no parts. *)

val pair : 'a t -> 'b t -> ('a * 'b) t
(** [pair a b] walks a pair's first component with [a] at [Index 0] and its
    second with [b] at [Index 1]. *)

val option : 'a t -> 'a option t
(** [option a] reports presence as {!Walk.option} does and walks the content
    with [a] at the same path. *)

val list : 'a t -> 'a list t
(** [list a] reports the length as {!Walk.list} does and walks the [i]-th
    element with [a] at [Index i]. *)

val iso : ('a -> 'b) -> ('b -> 'a) -> 'a t -> 'b t
(** [iso f g a] walks a value [y] as [a] walks [g y] and is [f] of the result.
    [f] and [g] must be inverse. It keeps [a]'s paths, so a record adapted from
    {!pair} walks at [0] and [1]; a record whose names matter is a module, whose
    [walk] walks a part that has only a {!type-t} with {!Walk.structure}.

    {[
    type out = { loss : Nx.float32_t; params : Nx.float32_t Mlp.t }

    let out =
      Nx.Ptree.(
        iso
          (fun (loss, params) -> { loss; params })
          (fun o -> (o.loss, o.params))
          (pair tensor mlp))
    ]} *)

(** {1:derived Maps and folds}

    These pass every tensor its path. Each tensor has its own dtype, so their
    functions are polymorphic. Fixed tensors are tensors like the others.

    A mismatch between two values raises [Invalid_argument] with the message
    [fn: p: a in the first value, b in the second], where [p] is the path of the
    first visit at which the values differ and [a] and [b] are what each holds
    there, as {!pp_visit} words it: [a leaf], [int 4], [case "mxfp4"], [Some],
    [None] or [length 3]. A value with no more visits holds [nothing]. When the
    two visits are at different paths, [b] reads [b at q] with [q] the second
    value's path; paths that print alike are shown as lists of segments. *)

val map :
  's t ->
  ('a 'b. Path.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) ->
  's ->
  's
(** [map s f x] is [x] with each tensor [t] at path [p] replaced by [f p t]. [f]
    is applied in walk order. *)

val map2 :
  's t ->
  ('a 'b.
   Path.t ->
   ('a, 'b) Nx_effect.t ->
   ('a, 'b) Nx_effect.t ->
   ('a, 'b) Nx_effect.t) ->
  's ->
  's ->
  's
(** [map2 s f x y] is [x] with each tensor [t] at path [p] replaced by
    [f p t u], where [u] is [y]'s tensor at [p]. [f] is applied in walk order.

    Raises [Invalid_argument] if [x] and [y] have different {!visits}, with the
    mismatch message of {!section-derived} prefixed by ["Nx.Ptree.map2"], or if
    two tensors at one path differ in dtype, with the message
    ["Nx.Ptree.map2: p: float32 in the first value, int32 in the second"]. [f]
    may have been applied to the tensors before [p]. It also raises if [s]'s
    [walk] visits one value two ways, which breaks {!module-type:S}'s contract.
*)

val fold :
  's t ->
  ('a 'b. Path.t -> ('a, 'b) Nx_effect.t -> 'acc -> 'acc) ->
  's ->
  'acc ->
  'acc
(** [fold s f x acc] is [f pn tn (... (f p1 t1 acc))], where [t1], ..., [tn] are
    [x]'s tensors in walk order and [p1], ..., [pn] their paths. *)

(** {1:visits Visits} *)

(** The type for the data a walk reports. *)
type report =
  | Int of int  (** An integer, from {!Walk.int}. *)
  | Case of string  (** A variant's case, from {!Walk.case}. *)
  | Present of bool
      (** Whether an option is present, from {!Walk.option} and {!option}. *)
  | Length of int  (** A list's length, from {!Walk.list} and {!list}. *)

(** The type for the steps of a walk. *)
type visit =
  | Leaf of Path.t  (** A tensor, at its path. *)
  | Report of Path.t * report  (** A report, at the path it is made at. *)

val visits : 's t -> 's -> visit list
(** [visits s x] is the steps of [s]'s walk of [x], in walk order: each tensor
    and each report, with its path. Two values share a compiled program only if
    their visits are equal. A structure's test compares them with the tensors,
    integers and cases it expects [walk] to visit. *)

val pp_visit : Format.formatter -> visit -> unit
(** [pp_visit ppf v] formats [v] as its path and what is visited there, such as
    ["blocks.3.moe.gate_up: case \"mxfp4\""] or ["0.keys: a leaf"]. The root
    path is formatted as ["the root"]. *)

(** {1:payload Maps that change the payload}

    A structure's [walk] also changes the type of its parameter's positions.
    That is how a model is cast and how metadata shaped like a model (a mask, a
    per-leaf learning rate, a sharding plan, per-leaf dimensions) is built and
    used. These operations take the structure's module, since only a module
    names the type constructor [U.t]. *)

val cast :
  (module U : S) ->
  ('c, 'd) Nx_core.Dtype.t ->
  ('a, 'b) Nx_effect.t U.t ->
  ('c, 'd) Nx_effect.t U.t
(** [cast (module U) dtype x] is [x] with each tensor at a position of [U]'s
    parameter cast to [dtype], as {!Nx.cast} does. Fixed tensors are kept. *)

(** Maps and folds at any payload.

    Parameter positions hold payloads of any type. Fixed tensors, and the
    tensors of a part walked with {!Walk.structure}, are kept from the first
    value; they are neither passed to the function nor compared. *)
module Payload : sig
  val map : (module U : S) -> (Path.t -> 'a -> 'b) -> 'a U.t -> 'b U.t
  (** [map (module U) f x] is [x] with each payload [v] at path [p] replaced by
      [f p v], applied in walk order.

      {[
      let sizes =
        Nx.Ptree.Payload.map
          (module Linear)
          (fun _ dims -> List.fold_left ( * ) 1 dims)
          dims
      ]} *)

  val map2 :
    (module U : S) -> (Path.t -> 'a -> 'b -> 'c) -> 'a U.t -> 'b U.t -> 'c U.t
  (** [map2 (module U) f x y] is [x] with each payload [v] at path [p] replaced
      by [f p v w], where [w] is [y]'s payload at [p].

      Raises [Invalid_argument] before applying [f] if [x] and [y] differ in
      their payloads' paths or in their reports, with the mismatch message of
      {!section-derived} prefixed by ["Nx.Ptree.Payload.map2"]. It also raises
      if [U.walk] visits one value two ways. *)

  val fold :
    (module U : S) -> (Path.t -> 'a -> 'acc -> 'acc) -> 'a U.t -> 'acc -> 'acc
  (** [fold (module U) f x acc] is [f pn vn (... (f p1 v1 acc))], where [v1],
      ..., [vn] are [x]'s payloads in walk order and [p1], ..., [pn] their
      paths. *)
end

(** {1:signatures Signatures of compiled functions}

    A signature has the shape of the function a transformation compiles:

    {[
    Nx.Ptree.(
      tensor @-> index @-> consumes caches @@ returns (pair tensor caches))
    ]}

    An argument built with {!( @-> )} is read; one built with {!consumes} is
    given up by each call. Each argument has its own type, so
    [tensor @-> tensor @-> ...] takes two tensors of unrelated dtypes. A
    signature has at least one argument wherever a transformation takes
    [('a -> 'b) fn]. *)

(** The type for how a compiled call treats an argument. *)
type role =
  | Read  (** The call reads the argument. *)
  | Consumed  (** The call consumes the argument. *)

(** The type for signatures of functions of type ['f]. Only {!( @-> )},
    {!consumes} and {!returns} build one; transformations match on it. *)
type _ fn = private
  | Returns : 'a t -> 'a fn  (** A result of structure ['a t]. *)
  | Arg : role * 'a t * 'b fn -> ('a -> 'b) fn
      (** An argument's role and structure, then the rest of the signature. *)

val ( @-> ) : 'a t -> 'b fn -> ('a -> 'b) fn
(** [a @-> f] is the signature of a function that reads an argument of structure
    [a] and continues as [f]. *)

val consumes : 'a t -> 'b fn -> ('a -> 'b) fn
(** [consumes a f] is the signature of a function that consumes an argument of
    structure [a] and continues as [f]. It is written [consumes a @@ f]. *)

val returns : 'a t -> 'a fn
(** [returns a] is the signature of a result of structure [a]. *)

(** {1:flattening Flattening}

    {b Low-level.} Transformations and optimisers work on a value as its tensors
    and its skeleton: compiled functions cache their programs under the skeleton
    and the tensors' dtypes and shapes, and put a program's tensors back into a
    value with {!rebuild}. Most code uses {!map}, {!map2} and {!fold} instead.
*)

(** Skeletons of values. *)
module Skeleton : sig
  type t
  (** The type for a value's skeleton: its {!visits} without the tensors. *)

  val equal : t -> t -> bool
  (** [equal k k'] is [true] iff [k] and [k'] have equal visits, their paths
      compared by segments ({!Path.equal}). *)

  val hash : t -> int
  (** [hash k] hashes [k]'s kinds of visits and its reports. Equal skeletons
      have equal hashes. *)

  val diff : this:string -> t -> that:string -> t -> string option
  (** [diff ~this k ~that k'] is [None] iff [equal k k']. Otherwise it is
      [Some m], where [m] names the first visit at which [k] and [k'] differ and
      what each holds there, in the form of {!section-derived}'s mismatch
      message: ["p: x this, y that"]. [this] and [that] are phrases that place a
      side, such as ["in the result"] and ["in the cotangents"]. For example,
      with [~this:"here"] and [~that:"in the previous key"]:
      ["window: int 8 here, int 4 in the previous key"]. *)
end

val flatten : 's t -> 's -> Nx_effect.packed list * Skeleton.t
(** [flatten s x] is [x]'s tensors in walk order and [x]'s skeleton. *)

val rebuild : 's t -> like:'s -> Nx_effect.packed list -> 's
(** [rebuild s ~like ts] is [like] with its tensors replaced by [ts], in walk
    order. [rebuild s ~like (fst (flatten s like))] is [like].

    Raises [Invalid_argument] if [ts] has fewer tensors than [like], naming the
    path of the first tensor left without one; if it has more, naming both
    counts; or if a tensor of [ts] has another dtype than [like]'s tensor at its
    position, naming the path and both dtypes. *)
