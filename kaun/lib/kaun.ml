type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

(* Parameter tree - alias for Ptree.t *)
type ('layout, 'dev) params = ('layout, 'dev) Ptree.t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of ('layout, 'dev) params Ptree.Record.t

type module_ = Module.t = {
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
let value_and_grad = Import.value_and_grad
let grad = Import.grad

module Module = Module
module Metrics = Metrics
module Loss = Loss
module Initializer = Initializers
module Layer = Layer
module Checkpoint = Kaun_checkpoint
module Ptree = Ptree
module Optimizer = Kaun_optim
module Activations = Activations
module Transformers = Kaun_transformers
module Dataset = Kaun_dataset
module Training = Training
