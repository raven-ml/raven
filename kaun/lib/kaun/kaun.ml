type 'layout tensor = (float, 'layout) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype

(* Parameter tree - alias for Ptree.t *)
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

let init m ~rngs ~dtype = m.init ~rngs ~dtype
let apply m params ~training ?rngs x = m.apply params ~training ?rngs x
let value_and_grad = Transformations.value_and_grad
let grad = Transformations.grad

module Ops = Ops
module Metrics = Metrics
module Loss = Loss
module Initializers = Initializers
module Layer = Layer
module Checkpoint = Checkpoint
module Ptree = Ptree
module Optimizer = Optimizer
module Activations = Activations
module Dataset = Dataset
module Training = Training
