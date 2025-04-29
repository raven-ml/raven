(** Kaun – a Flax‑inspired deep‑learning library for OCaml, powered by Rune *)

type ('layout, 'dev) tensor = (float, 'layout, [ `cpu ]) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

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

module Linear : sig
  type ('layout, 'dev) params = {
    w : ('layout, 'dev) tensor;
    b : ('layout, 'dev) tensor option;
  }

  val init :
    rng:Rng.t ->
    ?use_bias:bool ->
    dtype:(float, 'layout) Rune.dtype ->
    device:[ `cpu ] Rune.device ->
    int ->
    int ->
    ('layout, 'dev) params
  (** [init ~rng ?use_bias ~dtype ~device in_features out_features] *)

  val forward :
    ('layout, 'dev) params -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val update :
    lr:float ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params

  val params : ('layout, 'dev) params -> ('layout, 'dev) tensor list
  (** [params p] returns the parameters of the layer as a list of tensors. *)
end

module Optimizer : sig
  type 'op t
  type 'op state

  val sgd : float -> [ `sgd ] t
  val init : 'op t -> ('layout, 'dev) tensor list -> 'op state

  val update :
    'op t ->
    'op state ->
    ('layout, 'dev) tensor list ->
    'op state * ('layout, 'dev) tensor list

  val apply_updates :
    ('layout, 'dev) tensor list ->
    ('layout, 'dev) tensor list ->
    ('layout, 'dev) tensor list
end

module Loss : sig
  type ('layout, 'dev) t = ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val sigmoid_binary_cross_entropy :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor
end
