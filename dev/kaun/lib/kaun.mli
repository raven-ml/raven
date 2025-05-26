(** Kaun – a Flax‑inspired deep‑learning library for OCaml, powered by Rune *)

type ('layout, 'dev) tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

type ('layout, 'dev) ptree =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) ptree list
  | Record of (string * ('layout, 'dev) ptree) list

type ('model, 'layout, 'dev) lens = {
  to_ptree : 'model -> ('layout, 'dev) ptree;
  of_ptree : ('layout, 'dev) ptree -> 'model;
}

module Rng : sig
  type t

  val create : ?seed:int -> unit -> t

  val normal :
    t -> dtype:'layout dtype -> shape:int array -> ('layout, [ `cpu ]) tensor

  val uniform :
    t -> dtype:'layout dtype -> shape:int array -> ('layout, [ `cpu ]) tensor
end

module Activation : sig
  type ('layout, 'dev) t = ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val identity : ('layout, 'dev) t
  val relu : ('layout, 'dev) t
  val tanh : ('layout, 'dev) t
  val sigmoid : ('layout, 'dev) t
  val elu : float -> ('layout, 'dev) t
  val leaky_relu : float -> ('layout, 'dev) t
  val softplus : float -> ('layout, 'dev) t
end

module Initializer : sig
  type ('layout, 'dev) t =
    Rng.t -> int array -> 'layout dtype -> ('layout, 'dev) tensor

  val constant : float -> ('layout, 'dev) t
  val glorot_uniform : in_axis:int -> out_axis:int -> ('layout, 'dev) t
end

module Linear : sig
  type ('layout, 'dev) t

  val init :
    rng:Rng.t ->
    ?use_bias:bool ->
    dtype:(float, 'layout) Rune.dtype ->
    device:[ `cpu ] Rune.device ->
    int ->
    int ->
    ('layout, 'dev) t
  (** [init ~rng ?use_bias ~dtype ~device in_features out_features] *)

  val forward :
    ('layout, 'dev) t -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val update :
    lr:float -> ('layout, 'dev) t -> ('layout, 'dev) t -> ('layout, 'dev) t

  val params : ('a, 'b) t -> ('a, 'c) ptree
  val of_ptree : ('a, 'b) ptree -> ('a, 'c) t
  val lens : (('a, 'b) t, 'a, 'c) lens
end

module Loss : sig
  type ('layout, 'dev) t = ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val sigmoid_binary_cross_entropy :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor
end

module Optimizer : sig
  type 'op spec
  type (_, _, _, _) t

  val sgd : lr:float -> [ `sgd ] spec

  val adam :
    lr:float ->
    ?beta1:float ->
    ?beta2:float ->
    ?eps:float ->
    ?weight_decay:float ->
    unit ->
    [ `adam ] spec

  val init : lens:('m, 'l, 'd) lens -> 'm -> 'op spec -> ('op, 'm, 'l, 'd) t
  val update : ('op, 'a, 'b, 'c) t -> ('b, 'c) ptree -> unit
end

val value_and_grad :
  lens:('model, 'l, 'd) lens ->
  ('model -> ('layout, 'dev) tensor) ->
  'model ->
  ('layout, 'dev) tensor * ('l, 'd) ptree
