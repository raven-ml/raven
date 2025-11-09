type 'layout tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

type params = Ptree.t =
  | Tensor of Ptree.tensor
  | List of params list
  | Dict of (string * params) list

type module_ = Layer.module_ = {
  init :
    'layout. rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> Ptree.t;
  apply :
    'layout.
    Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t;
}

val init : module_ -> rngs:Rune.Rng.key -> dtype:'layout dtype -> params

val apply :
  module_ ->
  params ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  'layout tensor ->
  'layout tensor

val value_and_grad :
  (params -> 'layout tensor) -> params -> 'layout tensor * params

val grad : (params -> 'layout tensor) -> params -> params

module Metrics = Metrics
(** @inline *)

module Dataset = Dataset
(** @inline *)

module Loss = Loss
(** @inline *)

module Initializers = Initializers
(** @inline *)

module Attention = Attention
(** @inline *)

module Layer = Layer
(** @inline *)

module Checkpoint = Checkpoint
(** @inline *)

module Train_state = Train_state
(** @inline *)

module Ptree = Ptree
(** @inline *)

module Optimizer = Optimizer
(** @inline *)

module Activations = Activations
(** @inline *)

module Training = Training
(** @inline *)
