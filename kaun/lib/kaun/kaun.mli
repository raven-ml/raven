type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

type ('layout, 'dev) params = ('layout, 'dev) Ptree.t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of ('layout, 'dev) params Ptree.Record.t

type module_ = Layer.module_ = {
  init :
    'layout 'dev.
    rngs:Rune.Rng.key ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) Ptree.t;
  apply :
    'layout 'dev.
    ('layout, 'dev) Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor;
}

val init :
  module_ ->
  rngs:Rune.Rng.key ->
  device:'dev device ->
  dtype:'layout dtype ->
  ('layout, 'dev) params

val apply :
  module_ ->
  ('layout, 'dev) params ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  ('layout, 'dev) tensor ->
  ('layout, 'dev) tensor

val value_and_grad :
  (('layout, 'dev) params -> ('layout, 'dev) tensor) ->
  ('layout, 'dev) params ->
  ('layout, 'dev) tensor * ('layout, 'dev) params

val grad :
  (('layout, 'dev) params -> ('layout, 'dev) tensor) ->
  ('layout, 'dev) params ->
  ('layout, 'dev) params

module Ops = Ops
(** @inline *)

module Metrics = Metrics
(** @inline *)

module Dataset = Dataset
(** @inline *)

module Loss = Loss
(** @inline *)

module Initializer = Initializers
(** @inline *)

module Layer = Layer
(** @inline *)

module Checkpoint = Checkpoint
(** @inline *)

module Ptree = Ptree
(** @inline *)

module Optimizer = Optimizer
(** @inline *)

module Activations = Activations
(** @inline *)

module Training = Training
(** @inline *)
