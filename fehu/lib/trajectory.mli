(** Trajectory collection and episode management.

    Trajectories record sequential agent-environment interactions over multiple
    timesteps. They bundle observations, actions, rewards, and episode
    termination flags along with optional policy information like log
    probabilities and value estimates.

    {1 Usage}

    Collect a trajectory by running a policy:
    {[
      let policy obs =
        let action = (* compute action *) in
        let log_prob = (* compute log probability *) in
        let value = (* compute value estimate *) in
        (action, Some log_prob, Some value)
      in
      let trajectory = Trajectory.collect env ~policy ~n_steps:2048
    ]}

    Collect complete episodes:
    {[
      let episodes = Trajectory.collect_episodes env ~policy ~n_episodes:5 ()
    ]}

    Merge multiple trajectories:
    {[
      let combined = Trajectory.concat [ traj1; traj2; traj3 ]
    ]} *)

type ('obs, 'act) t = {
  observations : 'obs array;  (** Observations at each timestep *)
  actions : 'act array;  (** Actions taken at each timestep *)
  rewards : float array;  (** Rewards received at each timestep *)
  terminateds : bool array;  (** Natural episode terminations *)
  truncateds : bool array;  (** Artificial episode truncations *)
  infos : Info.t array;  (** Auxiliary information per timestep *)
  log_probs : float array option;
      (** Log probabilities log π(a|s), if available *)
  values : float array option;  (** Value estimates V(s), if available *)
}
(** Trajectory recording sequential interactions.

    A trajectory represents a sequence of timesteps, potentially spanning
    multiple episodes. Optional fields ([log_probs], [values]) are [None] if the
    policy doesn't provide them. All arrays have the same length. *)

val create :
  observations:'obs array ->
  actions:'act array ->
  rewards:float array ->
  terminateds:bool array ->
  truncateds:bool array ->
  ?infos:Info.t array ->
  ?log_probs:float array ->
  ?values:float array ->
  unit ->
  ('obs, 'act) t
(** [create ~observations ~actions ~rewards ~terminateds ~truncateds ~infos
     ~log_probs ~values ()] constructs a trajectory from collected data.

    All required arrays must have the same length. If [infos] is empty, creates
    an array of empty info dictionaries. If [infos] is provided, it must match
    the length of other arrays. Optional arrays ([log_probs], [values]) must
    also match the length when provided.

    @raise Invalid_argument if arrays have mismatched lengths. *)

val length : ('obs, 'act) t -> int
(** [length trajectory] returns the number of timesteps in the trajectory. *)

val concat : ('obs, 'act) t list -> ('obs, 'act) t
(** [concat trajectories] concatenates multiple trajectories into one.

    Merges all timesteps in order. Optional fields are preserved only if present
    in all input trajectories; otherwise they become [None] in the result.

    Time complexity: O(total_length) where total_length is the sum of all
    trajectory lengths.

    @raise Invalid_argument if the input list is empty. *)

(** {1 Collection}

    Functions for collecting trajectories by executing policies in environments.
*)

val collect :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act * float option * float option) ->
  n_steps:int ->
  ('obs, 'act) t
(** [collect env ~policy ~n_steps] collects a fixed-length trajectory.

    Executes the policy in the environment for exactly [n_steps] timesteps,
    automatically resetting the environment when episodes terminate or truncate.
    This produces trajectories that may span multiple episodes.

    The policy function returns [(action, log_prob_opt, value_opt)] where:
    - [action]: Action to take in the current state
    - [log_prob_opt]: Optional log probability log π(a|s) from the policy
    - [value_opt]: Optional value estimate V(s) from a critic

    If log probabilities or values are provided for all steps, they are included
    in the trajectory. If any step omits them, the trajectory's corresponding
    field is [None].

    Time complexity: O(n_steps). *)

val collect_episodes :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act * float option * float option) ->
  n_episodes:int ->
  ?max_steps:int ->
  unit ->
  ('obs, 'act) t list
(** [collect_episodes env ~policy ~n_episodes ~max_steps ()] collects complete
    episodes.

    Runs the policy to completion over multiple episodes, returning one
    trajectory per episode. Each episode runs until natural termination,
    truncation, or [max_steps] is reached. The environment is reset at the start
    of each episode.

    Use this when you need episode boundaries preserved, such as for Monte Carlo
    methods or when analyzing per-episode performance.

    Parameters:
    - [n_episodes]: Number of complete episodes to collect
    - [max_steps]: Maximum steps per episode before truncation (default: 1000)

    Time complexity: O(n_episodes × max_steps). *)
