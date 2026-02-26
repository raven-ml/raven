(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Vectorized environments.

    Runs multiple environment instances and batches their outputs. All
    environments must have compatible observation and action spaces. Terminated
    or truncated episodes are automatically reset. *)

(** {1:types Types} *)

type ('obs, 'act, 'render) t
(** The type for vectorized environments. *)

type 'obs step = {
  observations : 'obs array;  (** One observation per environment. *)
  rewards : float array;  (** One reward per environment. *)
  terminated : bool array;  (** Per-environment termination flags. *)
  truncated : bool array;  (** Per-environment truncation flags. *)
  infos : Info.t array;  (** Per-environment info dictionaries. *)
}
(** The type for batched step results. All arrays have length {!num_envs}. *)

(** {1:constructors Constructors} *)

val create : ('obs, 'act, 'render) Env.t list -> ('obs, 'act, 'render) t
(** [create envs] creates a vectorized environment.

    All environments must have structurally identical spaces (checked via
    {!Space.spec} and {!Space.equal_spec}). Raises [Invalid_argument] if [envs]
    is empty or spaces differ. *)

(** {1:accessors Accessors} *)

val num_envs : ('obs, 'act, 'render) t -> int
(** [num_envs t] is the number of environments. *)

val observation_space : ('obs, 'act, 'render) t -> 'obs Space.t
(** [observation_space t] is the shared observation space. *)

val action_space : ('obs, 'act, 'render) t -> 'act Space.t
(** [action_space t] is the shared action space. *)

(** {1:lifecycle Lifecycle} *)

val reset : ('obs, 'act, 'render) t -> unit -> 'obs array * Info.t array
(** [reset t ()] resets all environments. *)

val step : ('obs, 'act, 'render) t -> 'act array -> 'obs step
(** [step t actions] steps all environments with the given actions.

    [actions] must have length [num_envs t]. Terminated or truncated
    environments are automatically reset. The terminal observation is stored in
    the step's info under the key ["final_observation"] as a packed {!Value.t}.
    The terminal info is stored under ["final_info"].

    Raises [Invalid_argument] if [Array.length actions <> num_envs t]. *)

val close : ('obs, 'act, 'render) t -> unit
(** [close t] closes all environments. *)
