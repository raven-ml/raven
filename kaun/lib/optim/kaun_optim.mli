(** Optax-inspired gradient processing and optimization library *)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t

(** Parameter tree type - recursive structure for model parameters *)
type ('layout, 'dev) params =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of (string * ('layout, 'dev) params) list

type ('layout, 'dev) opt_state
(** Optimizer state - opaque type containing internal state *)

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

type ('layout, 'dev) gradient_transformation = {
  init : ('layout, 'dev) params -> ('layout, 'dev) opt_state;
  update :
    ('layout, 'dev) opt_state ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params ->
    ('layout, 'dev) params * ('layout, 'dev) opt_state;
}
(** Core gradient transformation type *)

(** {1 Core Transformations} *)

val identity : unit -> ('layout, 'dev) gradient_transformation
(** Identity transformation - returns gradients unchanged *)

val scale : float -> ('layout, 'dev) gradient_transformation
(** Scale gradients by a constant factor *)

val scale_by_neg_one : unit -> ('layout, 'dev) gradient_transformation
(** Scale gradients by -1 (for gradient descent) *)

val add_decayed_weights : float -> ('layout, 'dev) gradient_transformation
(** Add decayed value of parameters to updates (weight decay) *)

val clip_by_global_norm : float -> ('layout, 'dev) gradient_transformation
(** Clip gradients by global norm *)

val clip : float -> ('layout, 'dev) gradient_transformation
(** Clip gradients element-wise to [-max_delta, max_delta] *)

(** {1 Momentum and Adaptive Methods} *)

val trace :
  decay:float ->
  ?nesterov:bool ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Trace momentum - maintains exponential moving average of gradients *)

val scale_by_rms :
  ?decay:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation
(** Scale by RMS of gradients (used in RMSProp, Adam) *)

val scale_by_adam :
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Scale by Adam-style second moment estimate *)

val scale_by_belief :
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Scale by belief (used in AdaBelief) *)

(** {1 Learning Rate Schedules} *)

module Schedule : sig
  type t = int -> float

  val constant : float -> t
  (** Constant learning rate *)

  val exponential_decay :
    init_value:float -> decay_rate:float -> decay_steps:int -> t
  (** Exponential decay: lr * decay_rate^(step/decay_steps) *)

  val polynomial_decay :
    init_value:float -> end_value:float -> power:float -> decay_steps:int -> t
  (** Polynomial decay *)

  val cosine_decay :
    init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t
  (** Cosine decay *)

  val piecewise_constant : boundaries:(int * float) list -> t
  (** Piecewise constant schedule *)

  val warmup_linear :
    init_value:float -> peak_value:float -> warmup_steps:int -> t
  (** Linear warmup *)

  val warmup_cosine :
    init_value:float -> peak_value:float -> warmup_steps:int -> t
  (** Cosine warmup *)

  val join : t list -> boundaries:int list -> t
  (** Join two schedules *)
end

val scale_by_schedule : Schedule.t -> ('layout, 'dev) gradient_transformation
(** Scale updates by a learning rate schedule *)

(** {1 Composition} *)

val chain :
  ('layout, 'dev) gradient_transformation list ->
  ('layout, 'dev) gradient_transformation
(** Chain multiple transformations together *)

val multi_transform :
  transforms:('layout, 'dev) gradient_transformation array ->
  labels:(('layout, 'dev) params -> label_tree) ->
  ('layout, 'dev) gradient_transformation
(** Apply different transformations to different parameters The labels function
    maps parameters to integer labels. The transforms array maps labels to
    transformations. *)

val masked :
  mask:(('layout, 'dev) params -> mask_tree) ->
  inner:('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation
(** Apply transformation only to masked parameters *)

(** {1 Pre-configured Optimizers} *)

val sgd :
  lr:float ->
  ?momentum:float ->
  ?nesterov:bool ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Stochastic Gradient Descent *)

val adam :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Adam optimizer *)

val adamw :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** AdamW optimizer (Adam with weight decay) *)

val rmsprop :
  lr:float ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** RMSProp optimizer *)

val adagrad :
  lr:float -> ?eps:float -> unit -> ('layout, 'dev) gradient_transformation
(** AdaGrad optimizer *)

val adabelief :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** AdaBelief optimizer *)

val lamb :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** LAMB optimizer (Layer-wise Adaptive Moments) *)

val radam :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** RAdam (Rectified Adam) *)

val yogi :
  lr:float ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  unit ->
  ('layout, 'dev) gradient_transformation
(** Yogi optimizer *)

(** {1 Utilities} *)

val apply_updates :
  ('layout, 'dev) params -> ('layout, 'dev) params -> ('layout, 'dev) params
(** Apply updates to parameters: params = params - updates *)

val apply_updates_inplace :
  ('layout, 'dev) params -> ('layout, 'dev) params -> unit
(** Apply updates to parameters in place (mutates first argument) *)

val global_norm : ('layout, 'dev) params -> float
(** Compute global norm of gradients *)

val set_to_zero : ('layout, 'dev) params -> ('layout, 'dev) params
(** Set step count (for schedules and bias correction) *)

(** {1 Wrapper for Multi-step Updates} *)

val multi_steps :
  every:int ->
  ('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation
(** Accumulate gradients over multiple steps before applying *)

(** {1 Debugging} *)

val with_gradient_stats :
  ?prefix:string ->
  ('layout, 'dev) gradient_transformation ->
  ('layout, 'dev) gradient_transformation
(** Add gradient statistics logging *)
