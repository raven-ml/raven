type 'layout tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

type 'layout params = 'layout Ptree.t =
  | Tensor of 'layout tensor
  | List of 'layout params list
  | Record of 'layout params Ptree.Record.t

type module_ = Layer.module_ = {
  init :
    'layout 'dev.
    rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> 'layout Ptree.t;
  apply :
    'layout 'dev.
    'layout Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    'layout tensor ->
    'layout tensor;
}

val init : module_ -> rngs:Rune.Rng.key -> dtype:'layout dtype -> 'layout params

val apply :
  module_ ->
  'layout params ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  'layout tensor ->
  'layout tensor

val value_and_grad :
  ('layout params -> 'layout tensor) ->
  'layout params ->
  'layout tensor * 'layout params

val grad :
  ('layout params -> 'layout tensor) -> 'layout params -> 'layout params

module Metrics = Metrics
(** @inline *)

module Dataset = Dataset
(** @inline *)

module Loss = Loss
(** @inline *)

module Initializers = Initializers
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
