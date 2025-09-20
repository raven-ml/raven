(** Vectorized environments for parallel interaction.

    Vectorized environments run multiple environment instances in parallel,
    enabling efficient data collection. This module follows Gymnasium's
    vectorization API, batching observations, actions, and rewards across
    environments.

    {1 Benefits}

    - Collect trajectories faster by stepping multiple environments
      simultaneously
    - Amortize policy inference costs across batched observations
    - Essential for on-policy algorithms that need large amounts of data per
      update

    {1 Usage}

    Create a vectorized environment from multiple instances:
    {[
      let envs = List.init 8 (fun _ -> make_env ()) in
      let vec_env = Vector_env.create_sync ~envs () in
      let observations, infos = Vector_env.reset vec_env () in
      let actions = (* compute batched actions *) in
      let step = Vector_env.step vec_env actions
    ]}

    With autoreset enabled (default), terminated environments automatically
    reset on the next step, returning their initial observation. This ensures
    continuous data collection without manual intervention. *)

(** Autoreset behavior for terminated episodes.

    With [Next_step], when an environment terminates or truncates, the next call
    to {!step} returns its initial observation instead of requiring an explicit
    {!reset}. This maintains a constant number of active environments. *)
type autoreset_mode =
  | Next_step  (** Reset terminated environments on the next {!step} call *)
  | Disabled  (** Do not automatically reset; requires manual intervention *)

type ('obs, 'act, 'render) step = {
  observations : 'obs array;  (** Observations from each environment *)
  rewards : float array;  (** Rewards from each environment *)
  terminations : bool array;  (** Natural termination flags *)
  truncations : bool array;  (** Artificial truncation flags *)
  infos : Info.t array;  (** Info dictionaries from each environment *)
}
(** Batched step result from all environments. *)

type ('obs, 'act, 'render) t
(** Vectorized environment handle managing multiple environment instances. *)

val create_sync :
  ?autoreset_mode:autoreset_mode ->
  envs:('obs, 'act, 'render) Env.t list ->
  unit ->
  ('obs, 'act, 'render) t
(** [create_sync ~autoreset_mode ~envs ()] creates a synchronous vectorized
    environment.

    Wraps [envs] to provide batched operations. All environments are stepped
    sequentially in the current process. For true parallelism, consider
    asynchronous implementations (not yet provided).

    Parameters:
    - [autoreset_mode]: Controls automatic resetting of terminated environments
      (default: [Next_step])
    - [envs]: List of environment instances to vectorize. Must be non-empty and
      share compatible observation/action spaces

    @raise Invalid_argument if [envs] is empty. *)

val num_envs : ('obs, 'act, 'render) t -> int
(** [num_envs vec_env] returns the number of parallel environments. *)

val observation_space : ('obs, 'act, 'render) t -> Space.packed
(** [observation_space vec_env] returns the observation space of the vectorized
    environment.

    All constituent environments share the same observation space. *)

val action_space : ('obs, 'act, 'render) t -> Space.packed
(** [action_space vec_env] returns the action space of the vectorized
    environment.

    All constituent environments share the same action space. *)

val metadata : ('obs, 'act, 'render) t -> Metadata.t
(** [metadata vec_env] returns the metadata of the vectorized environment. *)

val reset : ('obs, 'act, 'render) t -> unit -> 'obs array * Info.t array
(** [reset vec_env ()] resets all environments.

    Returns [(observations, infos)] where each array has length {!num_envs},
    containing the initial observation and info from each environment. *)

val step : ('obs, 'act, 'render) t -> 'act array -> ('obs, 'act, 'render) step
(** [step vec_env actions] executes actions in all environments.

    Takes an array of actions with length {!num_envs}, steps each environment,
    and returns batched results.

    If autoreset is enabled ([Next_step]), terminated environments automatically
    reset and return their initial observation. The [terminations] and
    [truncations] arrays indicate which environments ended before resetting.

    @raise Invalid_argument if [actions] length doesn't match {!num_envs}. *)

val close : ('obs, 'act, 'render) t -> unit
(** [close vec_env] closes all constituent environments.

    Releases resources held by all environments. Subsequent operations will
    fail. *)
