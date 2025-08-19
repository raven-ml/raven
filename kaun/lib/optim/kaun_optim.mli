(** Optax-inspired gradient processing and optimization library *)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

(** Parameter tree type - recursive structure for model parameters *)
type ('layout, 'dev) params =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list

(** Optimizer state - opaque type containing internal state *)
type ('layout, 'dev) opt_state

(** Type for labeling parameters for multi_transform *)
type label_tree = 
  | LabelTensor of int
  | LabelList of label_tree list
  | LabelRecord of (string * label_tree) list

(** Type for masking parameters *)
type mask_tree = 
  | MaskTensor of bool
  | MaskList of mask_tree list
  | MaskRecord of (string * mask_tree) list

(** Core gradient transformation type *)
type ('layout, 'dev) gradient_transformation = {
  init : ('layout, 'dev) params -> ('layout, 'dev) opt_state;
  update :
    ('layout, 'dev) opt_state ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params * ('layout, 'dev) opt_state;
}

(** {1 Core Transformations} *)

(** Identity transformation - returns gradients unchanged *)
val identity : unit -> ('layout, 'dev) gradient_transformation

(** Scale gradients by a constant factor *)
val scale : float -> ('layout, 'dev) gradient_transformation

(** Scale gradients by -1 (for gradient descent) *)
val scale_by_neg_one : unit -> ('layout, 'dev) gradient_transformation

(** Add decayed value of parameters to updates (weight decay) *)
val add_decayed_weights : float -> ('layout, 'dev) gradient_transformation

(** Clip gradients by global norm *)
val clip_by_global_norm : float -> ('layout, 'dev) gradient_transformation

(** Clip gradients element-wise to [-max_delta, max_delta] *)
val clip : float -> ('layout, 'dev) gradient_transformation

(** {1 Momentum and Adaptive Methods} *)

(** Trace momentum - maintains exponential moving average of gradients *)
val trace :
  decay:float -> ?nesterov:bool -> unit -> ('layout, 'dev) gradient_transformation

(** Scale by RMS of gradients (used in RMSProp, Adam) *)
val scale_by_rms :
  ?decay:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation

(** Scale by Adam-style second moment estimate *)
val scale_by_adam :
  ?b1:float -> ?b2:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation

(** Scale by belief (used in AdaBelief) *)
val scale_by_belief :
  ?b1:float -> ?b2:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation

(** {1 Learning Rate Schedules} *)

module Schedule : sig
  type t = int -> float

  (** Constant learning rate *)
  val constant : float -> t

  (** Exponential decay: lr * decay_rate^(step/decay_steps) *)
  val exponential_decay :
    init_value:float -> decay_rate:float -> decay_steps:int -> t

  (** Polynomial decay *)
  val polynomial_decay :
    init_value:float ->
    end_value:float ->
    power:float ->
    decay_steps:int ->
    t

  (** Cosine decay *)
  val cosine_decay : init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t

  (** Piecewise constant schedule *)
  val piecewise_constant : boundaries:(int * float) list -> t

  (** Linear warmup *)
  val warmup_linear :
    init_value:float -> peak_value:float -> warmup_steps:int -> t

  (** Cosine warmup *)
  val warmup_cosine :
    init_value:float -> peak_value:float -> warmup_steps:int -> t

  (** Join two schedules *)
  val join : t list -> boundaries:int list -> t
end

(** Scale updates by a learning rate schedule *)
val scale_by_schedule : Schedule.t -> ('layout, 'dev) gradient_transformation

(** {1 Composition} *)

(** Chain multiple transformations together *)
val chain :
  ('layout, 'dev) gradient_transformation list ->
  ('layout, 'dev) gradient_transformation

(** Apply different transformations to different parameters
    The labels function maps parameters to integer labels.
    The transforms array maps labels to transformations. *)
val multi_transform :
  transforms:('layout, 'dev) gradient_transformation array ->
  labels:(('layout, 'dev) params -> label_tree) ->
  ('layout, 'dev) gradient_transformation

(** Apply transformation only to masked parameters *)
val masked :
  mask:(('layout, 'dev) params -> mask_tree) ->
  inner:('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation

(** {1 Pre-configured Optimizers} *)

(** Stochastic Gradient Descent *)
val sgd : lr:float -> ?momentum:float -> ?nesterov:bool -> unit -> ('layout, 'dev) gradient_transformation

(** Adam optimizer *)
val adam :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** AdamW optimizer (Adam with weight decay) *)
val adamw :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** RMSProp optimizer *)
val rmsprop :
  lr:float ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** AdaGrad optimizer *)
val adagrad :
  lr:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation

(** AdaBelief optimizer *)
val adabelief :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** LAMB optimizer (Layer-wise Adaptive Moments) *)
val lamb :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** RAdam (Rectified Adam) *)
val radam :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** Yogi optimizer *)
val yogi :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation

(** {1 Utilities} *)

(** Apply updates to parameters: params = params - updates *)
val apply_updates :
  ('layout, 'dev) params ->
  ('layout, 'dev) params ->
  ('layout, 'dev) params

(** Apply updates to parameters in place (mutates first argument) *)
val apply_updates_inplace :
  ('layout, 'dev) params ->
  ('layout, 'dev) params ->
  unit

(** Compute global norm of gradients *)
val global_norm : ('layout, 'dev) params -> float

(** Set step count (for schedules and bias correction) *)
val set_to_zero : ('layout, 'dev) params -> ('layout, 'dev) params

(** {1 Wrapper for Multi-step Updates} *)

(** Accumulate gradients over multiple steps before applying *)
val multi_steps :
  every:int ->
  ('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation

(** {1 Debugging} *)

(** Add gradient statistics logging *)
val with_gradient_stats :
  ?prefix:string ->
  ('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation