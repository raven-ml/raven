type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type 'layout dtype = (float, 'layout) Rune.dtype
type 'dev device = 'dev Rune.device

type ('layout, 'dev) params =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list

module Rngs : sig
  type t = int

  val create : seed:int -> unit -> t
  val split : t -> t * t
end

type model =
  | Model : {
      init :
        'layout 'dev.
        rngs:Rngs.t -> ('layout, 'dev) tensor -> ('layout, 'dev) params;
      apply :
        'layout 'dev.
        ('layout, 'dev) params ->
        training:bool ->
        ?rngs:Rngs.t ->
        ('layout, 'dev) tensor ->
        ('layout, 'dev) tensor;
    }
      -> model

val init :
  model -> rngs:Rngs.t -> ('layout, 'dev) tensor -> ('layout, 'dev) params

val apply :
  model ->
  ('layout, 'dev) params ->
  training:bool ->
  ?rngs:Rngs.t ->
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
end

module Optimizer : sig
  type ('layout, 'dev) t
  type transform

  (* Optimizers *)
  val sgd : lr:float -> ?momentum:float -> unit -> transform

  val adam :
    lr:float -> ?beta1:float -> ?beta2:float -> ?eps:float -> unit -> transform

  val adamw :
    lr:float ->
    ?beta1:float ->
    ?beta2:float ->
    ?eps:float ->
    ?weight_decay:float ->
    unit ->
    transform

  (* Create optimizer with transform *)
  val create : transform -> ('layout, 'dev) t

  (* Takes parameters and gradients, updates the parameters in-place *)
  val update :
    ('layout, 'dev) t ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params ->
    unit
end
