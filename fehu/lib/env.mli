(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reinforcement learning environments.

    An environment defines an interactive loop: the agent observes, acts, and
    receives a reward. The environment enforces a lifecycle: {!reset} must be
    called before {!step}, and a terminated or truncated episode requires
    another {!reset}. *)

(** {1:step Step results} *)

type 'obs step = {
  observation : 'obs;  (** The observation after the action. *)
  reward : float;  (** Scalar reward for the transition. *)
  terminated : bool;  (** [true] when the episode ends naturally. *)
  truncated : bool;  (** [true] when the episode is cut short. *)
  info : Info.t;  (** Auxiliary metadata. *)
}
(** The type for step results. *)

val step_result :
  observation:'obs ->
  ?reward:float ->
  ?terminated:bool ->
  ?truncated:bool ->
  ?info:Info.t ->
  unit ->
  'obs step
(** [step_result ~observation ()] constructs a step result. [reward] defaults to
    [0.], [terminated] and [truncated] default to [false], [info] defaults to
    {!Info.empty}. *)

(** {1:render Render modes} *)

type render_mode = [ `Human | `Rgb_array | `Ansi | `Svg | `Custom of string ]
(** Rendering modes supported by environments. *)

val render_mode_to_string : render_mode -> string
(** [render_mode_to_string m] is the string representation of [m]. *)

(** {1:env Environments} *)

type ('obs, 'act, 'render) t
(** Environment handle. Use {!create} or {!wrap} to construct. *)

val create :
  ?id:string ->
  rng:Rune.Rng.key ->
  observation_space:'obs Space.t ->
  action_space:'act Space.t ->
  ?render_mode:render_mode ->
  ?render_modes:string list ->
  reset:(('obs, 'act, 'render) t -> ?options:Info.t -> unit -> 'obs * Info.t) ->
  step:(('obs, 'act, 'render) t -> 'act -> 'obs step) ->
  ?render:(unit -> 'render option) ->
  ?close:(unit -> unit) ->
  unit ->
  ('obs, 'act, 'render) t
(** [create ~rng ~observation_space ~action_space ~reset ~step ()] makes a new
    environment.

    [reset] and [step] receive the environment handle as first argument so
    implementations can use {!take_rng} and {!split_rng} for randomness.

    [render_modes] lists the supported render mode strings. When [render_mode]
    is provided, it must appear in [render_modes].

    Raises [Invalid_argument] if [render_mode] is not in [render_modes]. *)

val wrap :
  ?id:string ->
  observation_space:'obs2 Space.t ->
  action_space:'act2 Space.t ->
  ?render_mode:render_mode ->
  reset:(('obs1, 'act1, 'render) t -> ?options:Info.t -> unit -> 'obs2 * Info.t) ->
  step:(('obs1, 'act1, 'render) t -> 'act2 -> 'obs2 step) ->
  ?render:(('obs1, 'act1, 'render) t -> 'render option) ->
  ?close:(('obs1, 'act1, 'render) t -> unit) ->
  ('obs1, 'act1, 'render) t ->
  ('obs2, 'act2, 'render) t
(** [wrap ~observation_space ~action_space ~reset ~step inner] builds a new
    environment that wraps [inner]. The wrapper shares [inner]'s lifecycle state
    (RNG, closed flag, reset flag). All guards (closed, needs-reset, space
    validation) are enforced by {!reset}/{!step}, so wrappers get them
    automatically.

    The render type is preserved from [inner]. [render_mode] defaults to
    [inner]'s. *)

(** {1:accessors Accessors} *)

val id : ('obs, 'act, 'render) t -> string option
(** [id env] is the environment's identifier, if any. *)

val observation_space : ('obs, 'act, 'render) t -> 'obs Space.t
(** [observation_space env] is the space of valid observations. *)

val action_space : ('obs, 'act, 'render) t -> 'act Space.t
(** [action_space env] is the space of valid actions. *)

val render_mode : ('obs, 'act, 'render) t -> render_mode option
(** [render_mode env] is the render mode chosen at construction, if any. *)

(** {1:rng Random state} *)

val rng : ('obs, 'act, 'render) t -> Rune.Rng.key
(** [rng env] is the current RNG key without consuming it. *)

val set_rng : ('obs, 'act, 'render) t -> Rune.Rng.key -> unit
(** [set_rng env key] replaces the environment's RNG key and marks the
    environment as needing a reset. *)

val take_rng : ('obs, 'act, 'render) t -> Rune.Rng.key
(** [take_rng env] splits the RNG internally, returns one half and keeps the
    other for future use. *)

val split_rng : ('obs, 'act, 'render) t -> n:int -> Rune.Rng.key array
(** [split_rng env ~n] generates [n] independent RNG keys. Splits the
    environment's RNG into [n+1] keys: [n] returned and one kept for the
    environment.

    Raises [Invalid_argument] if [n <= 0]. *)

(** {1:lifecycle Lifecycle} *)

val closed : ('obs, 'act, 'render) t -> bool
(** [closed env] is [true] iff the environment has been closed. *)

val reset : ('obs, 'act, 'render) t -> ?options:Info.t -> unit -> 'obs * Info.t
(** [reset env ()] resets the environment to an initial state.

    Raises [Invalid_argument] if [env] is closed, or if the reset function
    produces an observation outside {!observation_space}. *)

val step : ('obs, 'act, 'render) t -> 'act -> 'obs step
(** [step env action] advances the environment by one timestep.

    Raises [Invalid_argument] if [env] is closed, if no {!reset} has been called
    since the last terminal step, if [action] is outside {!action_space}, or if
    the step function produces an observation outside {!observation_space}. *)

val render : ('obs, 'act, 'render) t -> 'render option
(** [render env] produces a visualization of the current state.

    Raises [Invalid_argument] if [env] is closed. *)

val close : ('obs, 'act, 'render) t -> unit
(** [close env] releases resources held by the environment. Subsequent calls are
    no-ops. *)

(** {1:wrappers Wrappers} *)

val map_observation :
  observation_space:'obs2 Space.t ->
  f:('obs1 -> Info.t -> 'obs2 * Info.t) ->
  ('obs1, 'act, 'render) t ->
  ('obs2, 'act, 'render) t
(** [map_observation ~observation_space ~f env] transforms observations. Every
    observation from {!reset} and {!step} is passed through [f] together with
    the info dictionary. *)

val map_action :
  action_space:'act2 Space.t ->
  f:('act2 -> 'act1) ->
  ('obs, 'act1, 'render) t ->
  ('obs, 'act2, 'render) t
(** [map_action ~action_space ~f env] transforms actions before passing them to
    the inner environment. *)

val map_reward :
  f:(reward:float -> info:Info.t -> float * Info.t) ->
  ('obs, 'act, 'render) t ->
  ('obs, 'act, 'render) t
(** [map_reward ~f env] transforms rewards after each step. *)

(** {1:clip Clipping} *)

val clip_action :
  ('obs, Space.Box.element, 'render) t -> ('obs, Space.Box.element, 'render) t
(** [clip_action env] clamps continuous actions to the bounds of the inner
    environment's {!Space.Box} action space. The wrapper exposes a relaxed space
    that accepts any float values, then clips before forwarding. *)

val clip_observation :
  low:float array ->
  high:float array ->
  (Space.Box.element, 'act, 'render) t ->
  (Space.Box.element, 'act, 'render) t
(** [clip_observation ~low ~high env] clamps observations to \[[low]; [high]\].
    The wrapper's observation space is the intersection of the provided bounds
    and the inner space's bounds.

    Raises [Invalid_argument] if [low] and [high] differ in length or do not
    match the inner space's dimensionality. *)

(** {1:limits Limits} *)

val time_limit :
  max_episode_steps:int -> ('obs, 'act, 'render) t -> ('obs, 'act, 'render) t
(** [time_limit ~max_episode_steps env] enforces a maximum episode length. When
    the limit is reached the step's [truncated] flag is set to [true]. The
    counter resets on {!reset}.

    Raises [Invalid_argument] if [max_episode_steps <= 0]. *)
