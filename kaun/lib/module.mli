(** Neural network module abstraction for composable architectures.

    This module defines the core abstraction for neural network components in
    Kaun. A module encapsulates both parameter initialization and forward
    computation logic, following a functional API pattern similar to Flax NNX
    but adapted for OCaml. *)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

type t = {
  init :
    'layout 'dev.
    rngs:Rune.Rng.key ->
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) Ptree.t;
      (** [init ~rngs ~device ~dtype] initializes module parameters.

          Creates a parameter tree containing all trainable parameters for this
          module. The function is polymorphic over layout and device to support
          different tensor backends and memory layouts.

          @param rngs
            Random number generator key for deterministic initialization
          @param device Target device (CPU, CUDA, etc.) for parameter allocation
          @param dtype Data type specification, typically [Rune.float32]

          The RNG key should be split appropriately for modules with multiple
          parameters to ensure independent initialization. *)
  apply :
    'layout 'dev.
    ('layout, 'dev) Ptree.t ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor;
      (** [apply params ~training ?rngs input] performs forward computation.

          Executes the module's forward pass using the provided parameters and
          input tensor.

          @param params Parameter tree from [init] function
          @param training
            Whether module is in training mode (affects dropout, batch norm,
            etc.)
          @param rngs Optional RNG key for stochastic operations (dropout, etc.)
          @param input Input tensor to transform

          The training flag enables different behaviors:
          - Dropout: Applied only when [training=true]
          - Batch normalization: Uses batch statistics when [training=true]
          - Other regularization: Activated based on training mode

          RNG is required for stochastic operations during training. Operations
          needing randomness will fail if [rngs] is [None] when [training=true].
      *)
}
