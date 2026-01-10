(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** High-level training utilities operating on Train_state. *)

(** Helper functions for accessing training history. *)
module History : sig
  type t = {
    train_loss : float list;
    train_metrics : (string * float list) list;
    val_loss : float list option;
    val_metrics : (string * float list) list option;
  }

  val final_train_loss : t -> float option
  (** Get the final training loss value *)

  val final_val_loss : t -> float option
  (** Get the final validation loss value *)

  val final_train_metrics : t -> (string * float) list
  (** Get the final training metric values *)

  val final_val_metrics : t -> (string * float) list
  (** Get the final validation metric values *)

  val best_train_loss : t -> float option
  (** Get the best (minimum) training loss value *)

  val best_val_loss : t -> float option
  (** Get the best (minimum) validation loss value *)

  val best_epoch : ?monitor:string -> t -> int option
  (** Get the epoch with the best value for the monitored metric (default:
      "val_loss"). *)
end

(** Callback system for training hooks *)
module Callbacks : sig
  type t
  (** Abstract callback type *)

  type context = {
    epoch : int;
    state : Train_state.t;
    model : Layer.module_;
    optimizer : Optimizer.algorithm;
    history : History.t;
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
    t
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
    t
  (** [model_checkpoint ~filepath ?monitor ?mode ?save_best_only ?save_freq ()]
      creates a checkpoint callback.
      - [filepath]: Path pattern for saving checkpoints (can include an [epoch]
        placeholder)
      - [monitor]: Metric to monitor for best model (default: "val_loss")
      - [mode]: Whether to minimize or maximize the metric (default: `Min)
      - [save_best_only]: Only save when monitored metric improves (default:
        true)
      - [save_freq]: Save frequency - every N epochs or only best (default:
        `Best) *)

  val reduce_lr_on_plateau :
    ?monitor:string ->
    ?factor:float ->
    ?patience:int ->
    ?mode:[ `Min | `Max ] ->
    ?min_delta:float ->
    ?cooldown:int ->
    ?min_lr:float ->
    unit ->
    t
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
    log_dir:string -> ?update_freq:[ `Epoch | `Batch of int ] -> unit -> t
  (** [tensorboard ~log_dir ?update_freq ()] creates a TensorBoard logging
      callback.
      - [log_dir]: Directory where to save TensorBoard logs
      - [update_freq]: How often to write logs (default: `Epoch) *)

  val custom :
    ?on_epoch_begin:(context -> bool) ->
    ?on_epoch_end:(context -> bool) ->
    ?on_train_begin:(context -> unit) ->
    ?on_train_end:(context -> unit) ->
    unit ->
    t
  (** [custom ?on_epoch_begin ?on_epoch_end ?on_train_begin ?on_train_end ()]
      creates a custom callback with user-defined hooks. Returning false from
      epoch callbacks stops training. *)

  val combine : t list -> t
  (** Combine multiple callbacks into one *)
end

val train_step :
  model:Layer.module_ ->
  optimizer:Optimizer.algorithm ->
  state:Train_state.t ->
  x:(float, 'layout) Rune.t ->
  y:(float, 'layout) Rune.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  Train_state.t * float
(** Perform a single training step, returning the updated state and scalar loss.
*)

val eval_step :
  model:Layer.module_ ->
  state:Train_state.t ->
  x:(float, 'layout) Rune.t ->
  y:(float, 'layout) Rune.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  float
(** Evaluate loss without mutating state. *)

val train_epoch :
  model:Layer.module_ ->
  optimizer:Optimizer.algorithm ->
  state:Train_state.t ->
  dataset:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?progress:bool ->
  unit ->
  Train_state.t * float * (string * float) list
(** Run one training epoch and report average loss and metrics. *)

val evaluate :
  model:Layer.module_ ->
  state:Train_state.t ->
  dataset:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?progress:bool ->
  unit ->
  float * (string * float) list
(** Evaluate over a dataset, returning average loss and metrics. *)

val fit :
  model:Layer.module_ ->
  optimizer:Optimizer.algorithm ->
  loss_fn:
    ((float, 'layout) Rune.t ->
    (float, 'layout) Rune.t ->
    (float, 'layout) Rune.t) ->
  ?metrics:Metrics.Collection.t ->
  train_data:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  ?val_data:((float, 'layout) Rune.t * (float, 'layout) Rune.t) Dataset.t ->
  epochs:int ->
  ?callbacks:Callbacks.t list ->
  ?progress:bool ->
  rngs:Rune.Rng.key ->
  dtype:(float, 'layout) Rune.dtype ->
  unit ->
  Train_state.t * History.t
(** Train for multiple epochs, returning the final state and accumulated
    history. *)
