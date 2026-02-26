(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Replay buffer for off-policy experience storage.

    A fixed-capacity circular buffer that stores transitions and supports
    uniform random sampling. Observation and action arrays are lazily
    initialized on the first {!add}. *)

(** {1:types Types} *)

type ('obs, 'act) transition = {
  observation : 'obs;  (** State before the action. *)
  action : 'act;  (** Action taken. *)
  reward : float;  (** Scalar reward received. *)
  next_observation : 'obs;  (** State after the action. *)
  terminated : bool;  (** Natural episode ending. *)
  truncated : bool;  (** Forced episode ending. *)
}
(** The type for transitions. *)

type ('obs, 'act) t
(** A replay buffer of transitions. *)

(** {1:constructors Constructors} *)

val create : capacity:int -> ('obs, 'act) t
(** [create ~capacity] is an empty buffer that holds at most [capacity]
    transitions.

    Raises [Invalid_argument] if [capacity <= 0]. *)

(** {1:mutating Mutating} *)

val add : ('obs, 'act) t -> ('obs, 'act) transition -> unit
(** [add buf tr] appends [tr], overwriting the oldest transition when at
    capacity. *)

val clear : ('obs, 'act) t -> unit
(** [clear buf] removes all transitions, keeping storage allocated. *)

(** {1:sampling Sampling} *)

val sample : ('obs, 'act) t -> batch_size:int -> ('obs, 'act) transition array
(** [sample buf ~batch_size] draws [batch_size] transitions uniformly at random
    (with replacement).

    Random keys are drawn from the implicit RNG scope.

    If [batch_size] exceeds {!size}, samples [min batch_size size] transitions.

    Raises [Invalid_argument] if [buf] is empty or [batch_size <= 0]. *)

val sample_arrays :
  ('obs, 'act) t ->
  batch_size:int ->
  'obs array * 'act array * float array * 'obs array * bool array * bool array
(** [sample_arrays buf ~batch_size] is like {!sample} but returns
    structure-of-arrays
    [(observations, actions, rewards, next_observations, terminated, truncated)]
    for direct use in training loops. *)

(** {1:queries Queries} *)

val size : ('obs, 'act) t -> int
(** [size buf] is the number of stored transitions. *)

val is_full : ('obs, 'act) t -> bool
(** [is_full buf] is [true] iff [size buf = capacity]. *)

val capacity : ('obs, 'act) t -> int
(** [capacity buf] is the maximum number of transitions [buf] can hold. *)
