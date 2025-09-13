type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

(* Parameter tree - alias for Ptree.t *)
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

let init m ~rngs ~device ~dtype = m.init ~rngs ~device ~dtype
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
