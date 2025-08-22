type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

type ('layout, 'dev) params = ('layout, 'dev) Ptree.t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of ('layout, 'dev) params Ptree.Record.t

type model =
  | Model : {
      init :
        'layout 'dev.
        rngs:Rune.Rng.key -> ('layout, 'dev) tensor -> ('layout, 'dev) params;
      apply :
        'layout 'dev.
        ('layout, 'dev) params ->
        training:bool ->
        ?rngs:Rune.Rng.key ->
        ('layout, 'dev) tensor ->
        ('layout, 'dev) tensor;
    }
      -> model

val init :
  model -> rngs:Rune.Rng.key -> ('layout, 'dev) tensor -> ('layout, 'dev) params

val apply :
  model ->
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

module Metrics : sig
  type t
  type metric

  (* Metric constructors *)
  val avg : string -> metric
  val sum : string -> metric
  val accuracy : string -> metric

  (* Create metrics collection *)
  val create : metric list -> t

  (* Update with values *)
  val update :
    t ->
    ?loss:('layout, 'dev) tensor ->
    ?logits:('layout, 'dev) tensor ->
    ?labels:('layout, 'dev) tensor ->
    (* Class indices, not one-hot *)
    unit ->
    unit

  (* Get computed values *)
  val compute : t -> (string * float) list
  val get : t -> string -> float

  (* Reset all metrics *)
  val reset : t -> unit
end

module Dataset : sig
  type 'a t

  (* Creation *)
  val of_xy :
    ('l1, 'dev) tensor * ('l2, 'dev) tensor ->
    (('l1, 'dev) tensor * ('l2, 'dev) tensor) t

  (* Transformations *)
  val map : ('a -> 'b) -> 'a t -> 'b t
  val batch : int -> 'a t -> 'a t

  val batch_xy :
    int ->
    (('l1, 'dev) tensor * ('l2, 'dev) tensor) t ->
    (('l1, 'dev) tensor * ('l2, 'dev) tensor) t

  val shuffle : ?seed:int -> 'a t -> 'a t

  (* Iteration *)
  val iter : ('a -> unit) -> 'a t -> unit
  val length : 'a t -> int

  (* Take first n elements *)
  val take : int -> 'a t -> 'a list
end

module Loss : sig
  val softmax_cross_entropy :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val softmax_cross_entropy_with_indices :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val binary_cross_entropy :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val sigmoid_binary_cross_entropy :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val mse :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor

  val mae :
    ('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor
end

module Initializer : sig
  type t

  (** {1 Basic Initializers} *)

  val constant : float -> t
  val zeros : unit -> t
  val ones : unit -> t

  (** {1 Random Initializers} *)

  val uniform : ?scale:float -> unit -> t
  val normal : mean:float -> std:float -> t

  val truncated_normal :
    ?stddev:float -> ?lower:float -> ?upper:float -> unit -> t

  (** {1 Variance Scaling} *)

  val variance_scaling :
    scale:float ->
    mode:[ `Fan_in | `Fan_out | `Fan_avg ] ->
    distribution:[ `Normal | `Truncated_normal | `Uniform ] ->
    in_axis:int ->
    out_axis:int ->
    unit ->
    t

  (** {1 Glorot/Xavier Initializers} *)

  val glorot_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
  val glorot_normal : ?in_axis:int -> ?out_axis:int -> unit -> t
  val xavier_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
  val xavier_normal : ?in_axis:int -> ?out_axis:int -> unit -> t

  (** {1 He/Kaiming Initializers} *)

  val he_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
  val he_normal : ?in_axis:int -> ?out_axis:int -> unit -> t
  val kaiming_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
  val kaiming_normal : ?in_axis:int -> ?out_axis:int -> unit -> t

  (** {1 LeCun Initializers} *)

  val lecun_uniform : ?in_axis:int -> ?out_axis:int -> unit -> t
  val lecun_normal : ?in_axis:int -> ?out_axis:int -> unit -> t

  (** {1 Orthogonal Initializers} *)

  val orthogonal : ?scale:float -> ?column_axis:int -> unit -> t
  val delta_orthogonal : ?scale:float -> ?column_axis:int -> unit -> t

  (** {1 Utility Initializers} *)

  val uniform_range : low:float -> high:float -> unit -> t
  val normal_range : mean:float -> stddev:float -> unit -> t

  val apply :
    t ->
    int ->
    int array ->
    'dev Rune.device ->
    (float, 'layout) Rune.dtype ->
    (float, 'layout, 'dev) Rune.t
end

module Layer : sig
  val conv2d :
    in_channels:int ->
    out_channels:int ->
    ?kernel_size:int * int ->
    unit ->
    model

  val linear :
    in_features:int ->
    out_features:int ->
    ?weight_init:Initializer.t ->
    (* default: glorot_uniform *)
    ?bias_init:Initializer.t ->
    (* default: zeros *)
    unit ->
    model

  val dropout : rate:float -> unit -> model
  val batch_norm : num_features:int -> unit -> model
  val max_pool2d : kernel_size:int * int -> ?stride:int * int -> unit -> model
  val avg_pool2d : kernel_size:int * int -> ?stride:int * int -> unit -> model
  val flatten : unit -> model
  val relu : unit -> model
  val sigmoid : unit -> model
  val tanh : unit -> model
  val sequential : model list -> model

  (** New layers for transformer models *)

  val einsum :
    einsum_str:string ->
    shape:int array ->
    ?kernel_init:Initializer.t ->
    unit ->
    model

  val rms_norm :
    dim:int -> ?eps:float -> ?scale_init:Initializer.t -> unit -> model

  val layer_norm :
    dim:int -> ?eps:float -> ?elementwise_affine:bool -> unit -> model

  val embedding :
    vocab_size:int ->
    embed_dim:int ->
    ?scale:bool ->
    ?embedding_init:Initializer.t ->
    unit ->
    model

  val gelu : unit -> model
  val swish : unit -> model

  (** Attention layers *)

  val multi_head_attention :
    embed_dim:int ->
    num_heads:int ->
    ?num_kv_heads:int ->
    ?head_dim:int ->
    ?dropout:float ->
    ?use_qk_norm:bool ->
    ?attn_logits_soft_cap:float ->
    ?query_pre_attn_scalar:float ->
    unit ->
    model

  (** Positional embeddings *)

  val rope_embedding :
    dim:int ->
    ?max_seq_len:int ->
    ?base_frequency:float ->
    ?scale_factor:float ->
    unit ->
    model

  val sinusoidal_pos_embedding : max_len:int -> embed_dim:int -> unit -> model
end

(** {1 Additional modules for transformer support} *)

(** Cache support for KV caching in autoregressive generation *)
module Cache : sig
  type ('layout, 'dev) t

  val create :
    batch_size:int ->
    max_seq_len:int ->
    num_layers:int ->
    num_kv_heads:int ->
    head_dim:int ->
    device:'dev device ->
    dtype:'layout dtype ->
    ('layout, 'dev) t

  val update :
    ('layout, 'dev) t ->
    layer_idx:int ->
    k:('layout, 'dev) tensor ->
    v:('layout, 'dev) tensor ->
    pos:int ->
    unit

  val get :
    ('layout, 'dev) t ->
    layer_idx:int ->
    ('layout, 'dev) tensor * ('layout, 'dev) tensor
end

(** Extended operations needed for transformers *)
module Ops : sig
  val dynamic_update_slice :
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor ->
    indices:int list ->
    ('layout, 'dev) tensor

  val gather :
    ('layout, 'dev) tensor ->
    indices:('layout, 'dev) tensor ->
    axis:int ->
    ('layout, 'dev) tensor

  val one_hot :
    indices:('layout, 'dev) tensor -> num_classes:int -> ('layout, 'dev) tensor

  val tril : ('layout, 'dev) tensor -> ?k:int -> unit -> ('layout, 'dev) tensor
  val triu : ('layout, 'dev) tensor -> ?k:int -> unit -> ('layout, 'dev) tensor
  val to_float : ('layout, 'dev) tensor -> float
end

(** Learning rate schedules *)
module Schedule : sig
  type t = int -> float

  val constant : lr:float -> t

  val linear_warmup_cosine_decay :
    init_lr:float ->
    peak_lr:float ->
    warmup_steps:int ->
    decay_steps:int ->
    end_lr:float ->
    t

  val exponential_decay :
    init_lr:float ->
    decay_rate:float ->
    decay_steps:int ->
    ?staircase:bool ->
    unit ->
    t
end

(** Tokenization support *)
module Tokenizer : sig
  type t

  val from_sentencepiece : path:string -> t
  val from_tiktoken : encoding:string -> t

  val encode :
    t -> string -> ?add_bos:bool -> ?add_eos:bool -> unit -> int array

  val decode : t -> int array -> ?skip_special_tokens:bool -> unit -> string
  val vocab_size : t -> int
  val bos_id : t -> int option
  val eos_id : t -> int option
  val pad_id : t -> int option
end

module Checkpoint = Kaun_checkpoint
(** Checkpointing *)

module Ptree = Ptree
(** Parameter tree module - operations on parameter trees *)

module Optimizer = Kaun_optim
(** Optimizer module - gradient processing and optimization *)

module Activations : sig
  (** Activation functions for neural networks *)

  (** {1 Standard Activations} *)

  val relu : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val relu6 : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val sigmoid : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val tanh : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  val softmax :
    ?axes:int array -> (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  (** {1 Modern Activations} *)

  val gelu : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val silu : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val swish : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val mish : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  (** {1 Parametric Activations} *)

  val leaky_relu :
    ?negative_slope:float ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  val elu : ?alpha:float -> (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val selu : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  val prelu :
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  (** {1 Gated Linear Units} *)

  val glu :
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  val swiglu : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  val geglu :
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  val reglu :
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  (** {1 Other Activations} *)

  val softplus : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val softsign : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t

  val hard_sigmoid :
    ?alpha:float ->
    ?beta:float ->
    (float, 'a, 'dev) Rune.t ->
    (float, 'a, 'dev) Rune.t

  val hard_tanh : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  val hard_swish : (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
end

module Transformers = Kaun_transformers
(** Transformer building blocks - attention, RoPE *)
