(** Core environment interface for reinforcement learning.

    This module defines the standard RL environment interface inspired by OpenAI
    Gymnasium. Environments represent interactive tasks where agents observe
    states, take actions, and receive rewards.

    {1 Environment Lifecycle}

    Create an environment, reset it to get an initial observation, interact by
    stepping with actions, and optionally render or close resources:
    {[
      let env =
        Env.create ~rng ~observation_space ~action_space ~reset ~step ()
      in
      let obs, info = Env.reset env () in
      let transition = Env.step env action in
      Env.close env
    ]}

    {1 Episode Termination}

    Episodes end in two ways:
    - {b Terminated}: Natural completion (e.g., goal reached, game over)
    - {b Truncated}: Artificial cutoff (e.g., time limit, resource exhaustion)

    This distinction matters for bootstrapping: terminated episodes have zero
    future value, while truncated episodes may continue beyond the limit.

    {1 Custom Environments}

    Implement custom environments by providing [reset] and [step] functions:
    {[
      let env =
        Env.create ~rng ~observation_space ~action_space
          ~reset:(fun env ?options () ->
            (* reset logic *)
            (initial_obs, info))
          ~step:(fun env action ->
            (* transition logic *)
            { observation; reward; terminated; truncated; info })
          ()
    ]} *)

type ('obs, 'act, 'render) transition = {
  observation : 'obs;  (** Observation after taking the action *)
  reward : float;  (** Immediate reward from the action *)
  terminated : bool;  (** Whether episode ended naturally *)
  truncated : bool;  (** Whether episode was artificially cut off *)
  info : Info.t;  (** Auxiliary diagnostic information *)
}
(** Transition resulting from taking an action in an environment.

    Returned by {!step}. Contains the next observation, reward, episode status
    flags, and optional metadata. *)

val transition :
  ?reward:float ->
  ?terminated:bool ->
  ?truncated:bool ->
  ?info:Info.t ->
  observation:'obs ->
  unit ->
  ('obs, 'act, 'render) transition
(** [transition ~reward ~terminated ~truncated ~info ~observation ()] constructs
    a transition.

    Defaults: [reward = 0.0], [terminated = false], [truncated = false],
    [info = Info.empty]. *)

type ('obs, 'act, 'render) t
(** Environment handle.

    Encapsulates environment state, observation/action spaces, and RNG. The type
    parameters represent:
    - ['obs]: Observation type
    - ['act]: Action type
    - ['render]: Rendering output type *)

val create :
  ?id:string ->
  ?metadata:Metadata.t ->
  rng:Rune.Rng.key ->
  observation_space:'obs Space.t ->
  action_space:'act Space.t ->
  reset:(('obs, 'act, 'render) t -> ?options:Info.t -> unit -> 'obs * Info.t) ->
  step:(('obs, 'act, 'render) t -> 'act -> ('obs, 'act, 'render) transition) ->
  ?render:(('obs, 'act, 'render) t -> 'render option) ->
  ?close:(('obs, 'act, 'render) t -> unit) ->
  unit ->
  ('obs, 'act, 'render) t
(** [create ~id ~metadata ~rng ~observation_space ~action_space ~reset ~step
     ~render ~close ()] constructs a new environment.

    Parameters:
    - [id]: Optional identifier for the environment
    - [metadata]: Environment metadata (default: {!Metadata.default})
    - [rng]: Random number generator key for reproducibility
    - [observation_space]: Space defining valid observations
    - [action_space]: Space defining valid actions
    - [reset]: Function to reset environment to initial state. Receives optional
      reset [options] and returns initial observation and info
    - [step]: Function to advance environment by one timestep. Receives an
      action and returns a transition
    - [render]: Optional function to render environment state
    - [close]: Optional cleanup function to release resources *)

val id : ('obs, 'act, 'render) t -> string option
(** [id env] returns the environment's identifier, if any. *)

val metadata : ('obs, 'act, 'render) t -> Metadata.t
(** [metadata env] returns the environment's metadata. *)

val set_metadata : ('obs, 'act, 'render) t -> Metadata.t -> unit
(** [set_metadata env metadata] updates the environment's metadata. *)

val rng : ('obs, 'act, 'render) t -> Rune.Rng.key
(** [rng env] returns the current RNG key without consuming it. *)

val set_rng : ('obs, 'act, 'render) t -> Rune.Rng.key -> unit
(** [set_rng env rng] replaces the environment's RNG key. *)

val take_rng : ('obs, 'act, 'render) t -> Rune.Rng.key
(** [take_rng env] returns the current RNG key and generates a fresh one.

    Splits the RNG internally, returning one half and keeping the other for
    future use. Use this to obtain independent random streams. *)

val split_rng : ('obs, 'act, 'render) t -> n:int -> Rune.Rng.key array
(** [split_rng env ~n] generates [n] independent RNG keys.

    Splits the environment's RNG into [n+1] keys: [n] returned in the array and
    one kept for the environment. Use this for parallel operations requiring
    independent randomness. *)

val observation_space : ('obs, 'act, 'render) t -> 'obs Space.t
(** [observation_space env] returns the space of valid observations. *)

val action_space : ('obs, 'act, 'render) t -> 'act Space.t
(** [action_space env] returns the space of valid actions. *)

val reset : ('obs, 'act, 'render) t -> ?options:Info.t -> unit -> 'obs * Info.t
(** [reset env ~options ()] resets the environment to an initial state.

    Returns [(initial_observation, info)] where [info] contains optional
    diagnostic data. Call this at the start of training and after episodes
    complete.

    The [options] parameter allows passing environment-specific reset
    configuration. *)

val step : ('obs, 'act, 'render) t -> 'act -> ('obs, 'act, 'render) transition
(** [step env action] executes [action] in the environment.

    Returns a transition containing the next observation, reward, termination
    flags, and info. The action must be valid according to {!action_space}.

    After an episode terminates or truncates, call {!reset} before stepping
    again. *)

val render : ('obs, 'act, 'render) t -> 'render option
(** [render env] produces a visualization of the current environment state.

    Returns [None] if rendering is not supported or unavailable. *)

val close : ('obs, 'act, 'render) t -> unit
(** [close env] releases resources held by the environment.

    Call this when done using the environment. Subsequent operations on a closed
    environment may fail. *)

val closed : ('obs, 'act, 'render) t -> bool
(** [closed env] checks whether the environment has been closed. *)
