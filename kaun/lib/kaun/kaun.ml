type 'layout tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

(* Parameter tree - alias for Ptree.t *)
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

let init m ~rngs ~dtype = m.init ~rngs ~dtype
let apply m params ~training ?rngs x = m.apply params ~training ?rngs x
let value_and_grad = Transformations.value_and_grad
let grad = Transformations.grad

module Metrics = Metrics
module Loss = Loss
module Initializers = Initializers
module Layer = Layer
module Checkpoint = Checkpoint
module Train_state = Train_state
module Ptree = Ptree
module Optimizer = Optimizer
module Activations = Activations
module Dataset = Dataset
module Training = Training
