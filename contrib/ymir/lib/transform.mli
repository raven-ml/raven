(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transforms: the implementation of {!Ymir.Transform}, with the stage list
    that {!Wcs} reads and writes and the runs over cells that {!Grid} and
    {!Region} make.

    A transform is a list of stages ({!t}), each a family's parameters and a
    sense. A celestial stage's rotation takes native vectors to the frame's,
    [v = M · u], with [M = Rz(α_p) · Ry(90° - δ_p) · Rz(180° - φ_p)] for the
    celestial pole (α_p, δ_p) of the native sphere and LONPOLE φ_p. The pole
    solves FITS WCS Paper II's equations from CRVAL, the native reference
    point (φ₀, θ₀), LONPOLE and LATPOLE, as WCSLIB's [celset] does. Every
    branch is a [where], so the rotation is differentiable in the header's
    numbers.

    Messages name batch indices as {!index} and {!pp_ints} write them. *)

type plane = (float, Nx.float64_elt) Nx.t Ymir_units.Quantity.t

(** {!Ymir.Transform.code}. *)
type code = Projection.code =
  | Azp
  | Szp
  | Tan
  | Stg
  | Sin
  | Arc
  | Zpn
  | Zea
  | Air
  | Cyp
  | Cea
  | Car
  | Mer

(** {1:stages Stages} *)

(** The type for a stage's sense. [Forward] maps as the family's constructor
    states, and [Inverse] the other way. *)
type sense = Forward | Inverse

(** The type for planar families, each holding the arguments of its
    constructor: [Axes] those of {!axes}, [Shift] of {!shift}, [Linear] of
    {!linear}, [Scale] of {!scale}, [Sip] of {!sip} and [Tpv] of {!tpv}, with
    [stated] filled. *)
type planar =
  | Axes of { perm : int array; origin : int }
  | Shift of plane
  | Linear of plane
  | Scale of plane
  | Sip of {
      a : (float, Nx.float64_elt) Nx.t;
      b : (float, Nx.float64_elt) Nx.t;
      seed :
        ((float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t) option;
    }
  | Tpv of { pv : (float, Nx.float64_elt) Nx.t; stated : int array }

type 'f celestial = {
  code : code;
  frame : 'f Frame.t;
  stated : int array;
  pv : (float, Nx.float64_elt) Nx.t;
  native : plane;
  crval : plane;
  lonpole : plane;
  latpole : plane;
}
(** The type for a projection and the rotation to its frame. It holds the
    arguments of {!celestial}, with [stated] filled. *)

(** The type for stages. [Deproject] is a celestial stage's forward map, from
    the plane to directions, and [Project] its inverse. A rotation's inverse
    is the rotation between its frames swapped. *)
type (_, _) stage =
  | Plane : planar * sense -> (plane, plane) stage
  | Deproject : 'f celestial -> (plane, 'f Direction.t) stage
  | Project : 'f celestial -> ('f Direction.t, plane) stage
  | Rotate :
      'a Frame.fixed Frame.t * 'b Frame.fixed Frame.t
      -> ('a Frame.fixed Direction.t, 'b Frame.fixed Direction.t) stage

(** {!Ymir.Transform.t}: the stages in the order they apply. *)
type (_, _) t =
  | Id : ('a, 'a) t
  | Stage : ('a, 'b) stage * ('b, 'c) t -> ('a, 'c) t

val length : ('a, 'b) t -> int
(** [length t] is the number of [t]'s stages. *)

val code_name : code -> string
(** [code_name c] is {!Projection.name}[ c]. *)

val stage_name : ('a, 'b) stage -> string
(** [stage_name s] is [s]'s family and sense for a message, as
    ["inverse celestial TAN"] or ["rotation icrs galactic"]. *)

(** {1:public Constructors and operations} *)

val id : ('a, 'a) t
val axes : int array -> origin:int -> (plane, plane) t
val shift : plane -> (plane, plane) t
val linear : plane -> (plane, plane) t
val scale : plane -> (plane, plane) t

val sip :
  ?seed:(float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t ->
  (float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t ->
  (plane, plane) t

val tpv : ?stated:int array -> (float, Nx.float64_elt) Nx.t -> (plane, plane) t

val celestial :
  ?stated:int array ->
  code ->
  'f Frame.t ->
  pv:(float, Nx.float64_elt) Nx.t ->
  native:plane ->
  crval:plane ->
  lonpole:plane ->
  latpole:plane ->
  (plane, 'f Direction.t) t

val rotation :
  ('a Frame.fixed as 'f) Frame.t ->
  ('b Frame.fixed as 'g) Frame.t ->
  ('f Direction.t, 'g Direction.t) t

val about : 'f Direction.t -> ('f Direction.t, plane) t
val gnomonic : 'f Direction.t -> ('f Direction.t, plane) t
val ( >> ) : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
val inverse : ('a, 'b) t -> ('b, 'a) t
val apply : ('a, 'b) t -> 'a -> 'b
val covers : ('a, 'b) t -> 'a -> Nx.bool_t
val pp : Format.formatter -> ('a, 'b) t -> unit
val ptree : unit -> ('a, 'b) t Nx.Ptree.t

(** {1:cells Runs over cells} *)

(** The type for points of either endpoint type: [P] on a plane, [D]
    directions. *)
type value = P of plane | D : 'f Direction.t -> value

(** The type for a transform's endpoints, which tell its two point types
    apart at run time. *)
type _ endpoint =
  | Planar : plane endpoint
  | Sky : 'f Frame.t -> 'f Direction.t endpoint

val target : 'a endpoint -> ('a, 'b) t -> 'b endpoint
(** [target e t] is the endpoint [t] maps to, for [e] the one it maps
    from. *)

val source : ('a, 'b) t -> 'b endpoint -> 'a endpoint
(** [source t e] is the endpoint [t] maps from, for [e] the one it maps
    to. *)

val value : 'a endpoint -> 'a -> value
(** [value e x] is [x] as a {!value}. *)

type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
(** The type for functions on tensors of every type. *)

val map_points : 'a endpoint -> fn -> 'a -> 'a
(** [map_points e fn x] is [x] with [fn.f] applied to its tensor, a plane's
    payload or a direction's vectors. *)

val run_cells :
  ?stages:int -> cells:int -> ('a, 'b) t -> value -> value * Nx.bool_t option
(** [run_cells ~stages ~cells t x] is [t]'s first [stages] stages, all by
    default, applied to [x], and where each is defined, as {!covers} has it.
    No point outside a domain raises. Each parameter gets [cells] axes of 1
    after its batch axes, so a parameter of batch shape [b] meets points of
    shape [b @ cells @ [n]]. The mask is [None] when no stage that ran gives
    one.

    Raises [Invalid_argument] if a stage meets points of the other endpoint
    type, or as {!apply} does where a stage reads points of the wrong unit or
    size. *)

val cores : ('a, 'b) t -> int list
(** [cores t] is, for each tensor that [t]'s {!ptree} visits, in walk order,
    the number of its trailing axes that are not batch axes. *)

val expand_tensor : int -> int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [expand_tensor k core v] is [v] with [k] axes of 1 inserted before its
    last [core] axes. *)

(** {1:helpers Helpers} *)

val component : ('a, 'b) Nx.t -> int -> ('a, 'b) Nx.t
(** [component v k] is [v]'s component [k] on its last axis. *)

val broadcast : int array -> int array -> int array
(** [broadcast a b] is the shape [a] and [b] broadcast to.

    Raises [Invalid_argument] if they do not broadcast. *)

val stack : ('a, 'b) Nx.t list -> ('a, 'b) Nx.t
(** [stack vs] is [vs] broadcast together and stacked on a new last axis.

    Raises [Invalid_argument] if their shapes do not broadcast. *)

val numbers : ('a, 'b) Nx.t -> Nx.bool_t
(** [numbers v] is where [v]'s points, on its last axis, have no NaN
    component. *)

val pp_ints : Format.formatter -> int array -> unit
(** [pp_ints ppf a] formats [a] as ["[3; 1]"]. *)

val index : int array -> string
(** [index i] is the batch index [i] for a message: ["12"] for one axis, and
    as {!pp_ints} writes it otherwise. *)

val ints : ('a, 'b) Nx.Ptree.Walk.cursor -> int array -> int array
(** [ints c a] reports [a]'s length, then each of its elements, as static
    data at [c], and is [a]. *)

val quantity :
  ('a, 'b) Nx.Ptree.Walk.cursor ->
  ('c, 'd) Nx.t Ymir_units.Quantity.t ->
  ('c, 'd) Nx.t Ymir_units.Quantity.t
(** [quantity c q] walks [q] at [c] with the structure of quantities. *)
