(** High-level training utilities for neural networks. *)

(** Training state management for neural networks.

    This module provides a unified way to manage training state, combining model
    parameters, optimizer state, metrics, and other training metadata. Inspired
    by Flax's TrainState pattern. *)
module State : sig
  type 'layout t = {
    step : int;  (** Current training step *)
    params : 'layout Ptree.t;  (** Model parameters *)
    opt_state : 'layout Optimizer.opt_state;  (** Optimizer state *)
    metrics : 'layout Metrics.Collection.t option;
        (** Optional metrics collection *)
    rngs : Rune.Rng.key;  (** Random number generator keys *)
    model : Layer.module_;  (** The model definition *)
    optimizer : 'layout Optimizer.gradient_transformation;  (** The optimizer *)
  }

  val create :
    model:Layer.module_ ->
    optimizer:'layout Optimizer.gradient_transformation ->
    ?metrics:'layout Metrics.Collection.t ->
    rngs:Rune.Rng.key ->
    dtype:(float, 'layout) Rune.dtype ->
    unit ->
    'layout t
  (** [create ~model ~optimizer ?metrics ~rngs ~dtype] creates a new training
      state. *)

  val apply_gradients : 'layout t -> grads:'layout Ptree.t -> 'layout t
  (** [apply_gradients state ~grads] applies gradients to update parameters and
      optimizer state, incrementing the step counter. *)

  val reset_metrics : 'layout t -> 'layout t
  (** [reset_metrics state] resets all metrics in the state. *)

  val update_metrics :
    'layout t ->
    predictions:(float, 'layout) Rune.t ->
    targets:(float, 'layout) Rune.t ->
    ?loss:(float, 'layout) Rune.t ->
    unit ->
    unit
  (** [update_metrics state ~predictions ~targets ?loss ()] updates metrics with
      new predictions and targets. *)

  val compute_metrics : 'layout t -> (string * (float, 'layout) Rune.t) list
  (** [compute_metrics state] computes current metric values. *)

  val next_rng : 'layout t -> Rune.Rng.key * 'layout t
  (** [next_rng state] splits the RNG key and returns a new key and updated
      state. *)
end

(** Helper functions for accessing training history *)
module History : sig
  type 'layout t = {
    train_loss : float list;
    train_metrics : (string * float list) list;
    val_loss : float list option;
    val_metrics : (string * float list) list option;
  }

  val final_train_loss : 'layout t -> float option
  (** Get the final training loss value *)

  val final_val_loss : 'layout t -> float option
  (** Get the final validation loss value *)

  val final_train_metrics : 'layout t -> (string * float) list
  (** Get the final training metric values *)

  val final_val_metrics : 'layout t -> (string * float) list
  (** Get the final validation metric values *)

  val best_train_loss : 'layout t -> float option
  (** Get the best (minimum) training loss value *)

  val best_val_loss : 'layout t -> float option
  (** Get the best (minimum) validation loss value *)

  val best_epoch : ?monitor:string -> 'layout t -> int option
  (** Get the epoch with the best value for the monitored metric (default:
      "val_loss") *)
end

(** Callback system for training hooks *)
module Callbacks : sig
  type 'layout t
  (** Abstract callback type *)

  type 'layout context = {
    epoch : int;
    state : 'layout State.t;
    history : 'layout History.t;
    train_loss : float option;
    val_loss : float option;
    train_metrics : (string * float) list;
    val_metrics : (string * float) list;
  }
  (** Context passed to callbacks *)

  val early_stopping :
    ?monitor:string ->
    ?patience:int ->
    ?mode:[ `Min | `Max ] ->
    ?min_delta:float ->
    ?baseline:float option ->
    unit ->
    'layout t
  (** [early_stopping ?monitor ?patience ?mode ?min_delta ?baseline ()] creates
      an early stopping callback.
      - [monitor]: Metric to monitor (default: "val_loss")
      - [patience]: Number of epochs with no improvement to wait (default: 5)
      - [mode]: Whether to minimize or maximize the metric (default: `Min)
      - [min_delta]: Minimum change to qualify as improvement (default: 0.0)
      - [baseline]: Baseline value; training stops if metric doesn't exceed it
  *)

  val model_checkpoint :
    filepath:string ->
    ?monitor:string ->
    ?mode:[ `Min | `Max ] ->
    ?save_best_only:bool ->
    ?save_freq:[ `Epoch of int | `Best ] ->
    unit ->
    'layout t
  (** [model_checkpoint ~filepath ?monitor ?mode ?save_best_only ?save_freq ()] creates a checkpoint callback.
      - [filepath]: Path pattern for saving checkpoints (can include {epoch} placeholder)
      - [monitor]: Metric to monitor for best model (default: "val_loss")
      - [mode]: Whether to minimize or maximize the metric (default: `Min)
      - [save_best_only]: Only save when monitored metric improves (default: true)
      - [save_freq]: Save frequency - every N epochs or only best (default: `Best) *)

  val reduce_lr_on_plateau :
    ?monitor:string ->
    ?factor:float ->
    ?patience:int ->
    ?mode:[ `Min | `Max ] ->
    ?min_delta:float ->
    ?cooldown:int ->
    ?min_lr:float ->
    unit ->
    'layout t
  (** [reduce_lr_on_plateau ?monitor ?factor ?patience ?mode ?min_delta
       ?cooldown ?min_lr ()] creates a learning rate reduction callback.
      - [monitor]: Metric to monitor (default: "val_loss")
      - [factor]: Factor by which to reduce learning rate (default: 0.1)
      - [patience]: Number of epochs with no improvement to wait (default: 10)
      - [mode]: Whether to minimize or maximize the metric (default: `Min)
      - [min_delta]: Minimum change to qualify as improvement (default: 0.0001)
      - [cooldown]: Number of epochs to wait before resuming normal operation
        (default: 0)
      - [min_lr]: Lower bound on learning rate (default: 0.0) *)

  val tensorboard :
    log_dir:string ->
    ?update_freq:[ `Epoch | `Batch of int ] ->
    unit ->
    'layout t
  (** [tensorboard ~log_dir ?update_freq ()] creates a TensorBoard logging
      callback.
      - [log_dir]: Directory where to save TensorBoard logs
      - [update_freq]: How often to write logs (default: `Epoch) *)

  val custom :
    ?on_epoch_begin:('layout context -> bool) ->
    ?on_epoch_end:('layout context -> bool) ->
    ?on_train_begin:('layout context -> unit) ->
    ?on_train_end:('layout context -> unit) ->
    unit ->
    'layout t
  (** [custom ?on_epoch_begin ?on_epoch_end ?on_train_begin ?on_train_end ()]
      creates a custom callback with user-defined hooks. Returning false from
      epoch callbacks stops training. *)

  val combine : 'layout t list -> 'layout t
  (** Combine multiple callbacks into one *)
end

val train_step :
  state:'layout State.t ->
  x:(float, 'layout) Rune.t ->
  y:(float, 'layout) Rune.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  'layout State.t * float
(** [train_step ~state ~x ~y ~loss_fn] performs a single training step,
    computing loss, gradients, and updating parameters. Returns updated state
    and loss value. *)

val eval_step :
  state:'layout State.t ->
  x:(float, 'layout) Rune.t ->
  y:(float, 'layout) Rune.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  float
(** [eval_step ~state ~x ~y ~loss_fn] performs evaluation without updating
    parameters. Returns loss value. *)

val train_epoch :
  state:'layout State.t ->
  dataset:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?progress:bool ->
  unit ->
  'layout State.t * float * (string * float) list
(** [train_epoch ~state ~dataset ~loss_fn ?progress ()] trains for one epoch.
    Returns updated state, average loss, and metric values. *)

val evaluate :
  state:'layout State.t ->
  dataset:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?progress:bool ->
  unit ->
  float * (string * float) list
(** [evaluate ~state ~dataset ~loss_fn ?progress ()] evaluates the model.
    Returns average loss and metric values. *)

val fit :
  model:Layer.module_ ->
  optimizer:'layout Optimizer.gradient_transformation ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?metrics:'layout Metrics.Collection.t ->
  train_data:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  ?val_data:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  epochs:int ->
  ?callbacks:'layout Callbacks.t list ->
  ?progress:bool ->
  rngs:Rune.Rng.key ->
  dtype:(float, 'layout) Rune.dtype ->
  unit ->
  'layout State.t * 'layout History.t
(** [fit ~model ~optimizer ~loss_fn ?metrics ~train_data ?val_data ~epochs
     ?callbacks ?progress ~rngs ~dtype ()] trains a model for multiple epochs.
    Returns final training state and training history. *)
