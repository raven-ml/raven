(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** High-level training loop.

    {!Train} composes {!Layer}, {!Grad}, and {!Optim} into a single training
    driver. Users never touch parameter trees, optimizer state, or gradient
    computation directly.

    For advanced use, {!step} exposes a single training step and {!vars} gives
    access to the underlying model variables. *)

(** {1:types Types} *)

type ('i, 'o) t
(** The type for trainers. A trainer pairs a model with an optimizer. *)

type 'l state
(** The type for training state. Bundles model variables and optimizer state. *)

(** {1:core Core} *)

val make : model:('i, 'o) Layer.t -> optimizer:Optim.algorithm -> ('i, 'o) t
(** [make ~model ~optimizer] creates a trainer. *)

val init :
  ('i, 'o) t -> rngs:Rune.Rng.key -> dtype:(float, 'l) Rune.dtype -> 'l state
(** [init trainer ~rngs ~dtype] initializes model variables and optimizer state.
*)

val vars : 'l state -> 'l Layer.vars
(** [vars st] is the current model variables (params + state + dtype). *)

val make_state : ('i, 'o) t -> 'l Layer.vars -> 'l state
(** [make_state trainer vars] is a training state with [vars] and freshly
    initialized optimizer state.

    Use this to start training from pretrained or externally loaded weights
    instead of {!init}. *)

(** {1:training Training} *)

exception Early_stop
(** Raise inside [report] to end training early. {!fit} catches this exception
    and returns the current state. *)

val step :
  ('i, 'o) t ->
  'l state ->
  training:bool ->
  ?rngs:Rune.Rng.key ->
  ?ctx:Context.t ->
  loss:(('o, 'l) Rune.t -> (float, 'l) Rune.t) ->
  ('i, 'in_elt) Rune.t ->
  (float, 'l) Rune.t * 'l state
(** [step trainer st ~training ?rngs ?ctx ~loss x] performs one training step.

    Computes the forward pass, differentiates the loss with respect to trainable
    parameters, applies the optimizer, and threads updated layer state.

    [ctx] is forwarded to the model's forward pass. See {!Context}.

    When [training = false], gradients are still computed and optimizer is still
    applied. Use {!predict} for pure inference. *)

val fit :
  ('i, 'o) t ->
  'l state ->
  ?rngs:Rune.Rng.key ->
  ?ctx:Context.t ->
  ?report:(step:int -> loss:float -> 'l state -> unit) ->
  (('i, 'in_elt) Rune.t * (('o, 'l) Rune.t -> (float, 'l) Rune.t)) Data.t ->
  'l state
(** [fit trainer st ?rngs ?ctx ?report data] trains the model over [data] and
    returns the final state.

    Each element of [data] is a pair [(x, loss_fn)] where [x] is the input
    tensor and [loss_fn] computes the scalar loss from the model output. This
    allows the loss to depend on per-batch labels.

    [ctx] is forwarded to the model's forward pass on each step. See {!Context}.

    When provided, [report] is called after every step with the step number
    (1-based), scalar loss, and training state. Raise {!Early_stop} inside
    [report] to end training early.

    For fixed-data training (same input every step), use {!Data.repeat}:
    {[
      Train.fit trainer st (Data.repeat 1000 (x, loss_fn))
    ]} *)

(** {1:inference Inference} *)

val predict :
  ('i, 'o) t ->
  'l state ->
  ?ctx:Context.t ->
  ('i, 'in_elt) Rune.t ->
  ('o, 'l) Rune.t
(** [predict trainer st ?ctx x] runs the model in evaluation mode (no state
    updates, no dropout).

    [ctx] is forwarded to the model's forward pass. See {!Context}. *)
