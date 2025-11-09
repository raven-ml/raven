(** Optax-inspired optimisation algorithms with explicit checkpoint support. *)

type state
(** Optimiser state - opaque container storing algorithm-specific payloads. *)

type algorithm
(** Optimisation algorithm acting on parameter trees. *)

val name : algorithm -> string
(** Human-readable name for the algorithm. *)

val init : algorithm -> Ptree.t -> state
(** Initialise optimiser state for the given parameters. *)

val step : algorithm -> state -> Ptree.t -> Ptree.t -> Ptree.t * state
(** Apply the algorithm to gradients, returning parameter updates and the next
    optimiser state. *)

val serialize : state -> Checkpoint.Snapshot.t
(** Serialise an optimiser state into a checkpoint snapshot. *)

val restore : algorithm -> Checkpoint.Snapshot.t -> (state, string) result
(** Restore optimiser state for the given algorithm from a checkpoint. *)

val step_count : state -> int option
(** Extract a step counter from an optimiser state when available (e.g. Adam).
*)

(** {1 Algorithm building blocks} *)

val identity : unit -> algorithm
val scale : float -> algorithm
val scale_by_neg_one : unit -> algorithm
val add_decayed_weights : float -> algorithm
val clip_by_global_norm : float -> algorithm
val clip : float -> algorithm
val trace : decay:float -> ?nesterov:bool -> unit -> algorithm
val scale_by_rms : ?decay:float -> ?eps:float -> unit -> algorithm
val scale_by_adam : ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm
val scale_by_belief : ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm

(** {1 Learning rate schedules} *)

module Schedule : sig
  type t = int -> float

  val constant : float -> t

  val exponential_decay :
    init_value:float -> decay_rate:float -> decay_steps:int -> t

  val polynomial_decay :
    init_value:float -> end_value:float -> power:float -> decay_steps:int -> t

  val cosine_decay :
    init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t

  val piecewise_constant : boundaries:(int * float) list -> t

  val warmup_linear :
    init_value:float -> peak_value:float -> warmup_steps:int -> t

  val warmup_cosine :
    init_value:float -> peak_value:float -> warmup_steps:int -> t

  val join : t list -> boundaries:int list -> t
end

val scale_by_schedule : Schedule.t -> algorithm

(** {1 Composition helpers} *)

val chain : algorithm list -> algorithm

type label_tree =
  | Label_tensor of int
  | Label_list of label_tree list
  | Label_record of (string * label_tree) list

type mask_tree =
  | Mask_tensor of bool
  | Mask_list of mask_tree list
  | Mask_record of (string * mask_tree) list

val multi_transform :
  transforms:algorithm list -> labels:(Ptree.t -> label_tree) -> algorithm
(** Applies different transforms based on labels computed from params. *)

val masked : mask:(Ptree.t -> mask_tree) -> inner:algorithm -> algorithm
(** Masks gradients/updates based on a function over params. *)

(** {1 Utility functions} *)

val apply_updates : Ptree.t -> Ptree.t -> Ptree.t
val apply_updates_inplace : Ptree.t -> Ptree.t -> unit
val global_norm : Ptree.t -> float
val set_to_zero : Ptree.t -> Ptree.t
val multi_steps : every:int -> algorithm -> algorithm
val with_gradient_stats : ?prefix:string -> algorithm -> algorithm

(** {1 Pre-configured optimisers} *)

val sgd :
  lr:Schedule.t -> ?momentum:float -> ?nesterov:bool -> unit -> algorithm

val adam :
  lr:Schedule.t -> ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm

val adamw :
  lr:Schedule.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  algorithm

val rmsprop :
  lr:Schedule.t ->
  ?decay:float ->
  ?eps:float ->
  ?momentum:float ->
  unit ->
  algorithm

val adagrad : lr:Schedule.t -> ?eps:float -> unit -> algorithm

val adabelief :
  lr:Schedule.t -> ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm

val lamb :
  lr:Schedule.t ->
  ?b1:float ->
  ?b2:float ->
  ?eps:float ->
  ?weight_decay:float ->
  unit ->
  algorithm

val radam :
  lr:Schedule.t -> ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm

val yogi :
  lr:Schedule.t -> ?b1:float -> ?b2:float -> ?eps:float -> unit -> algorithm
