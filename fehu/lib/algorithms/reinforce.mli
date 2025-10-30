(** REINFORCE algorithm (Monte Carlo Policy Gradient).

    REINFORCE is a classic policy gradient algorithm that optimizes policies by
    following the gradient of expected return. It collects complete episodes,
    computes returns using Monte Carlo estimation, and updates the policy to
    increase the probability of actions that led to high returns.

    {1 Algorithm}

    REINFORCE follows these steps:
    - Collect a complete episode following the current policy
    - Compute discounted returns G_t for each timestep
    - Update policy parameters by gradient ascent on log π(a|s) × G_t
    - Optionally use a baseline (value function) to reduce variance

    {1 Usage}

    Basic usage:
    {[
      open Fehu

      (* Create policy network *)
      let policy_net = Kaun.Layer.sequential [
        Kaun.Layer.linear ~in_features:4 ~out_features:32 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:32 ~out_features:2 ();
      ] in

      (* Initialize agent *)
      let agent = Reinforce.create
        ~policy_network:policy_net
        ~n_actions:2
        ~rng:(Rune.Rng.key 42)
        Reinforce.{ default_config with learning_rate = 0.001 }
      in

      (* Train *)
      let agent = Reinforce.learn agent ~env ~total_timesteps:100_000 () in

      (* Use trained policy *)
      let action = Reinforce.predict agent obs
    ]}

    With baseline (actor-critic):
    {[
      let value_net = Kaun.Layer.sequential [...] in
      let agent = Reinforce.create
        ~policy_network:policy_net
        ~baseline_network:value_net
        ~n_actions:2
        ~rng:(Rune.Rng.key 42)
        { default_config with use_baseline = true }
      in
      let agent = Reinforce.learn agent ~env ~total_timesteps:100_000 ()
    ]} *)

type config = {
  learning_rate : float;  (** Learning rate for both policy and baseline *)
  gamma : float;  (** Discount factor for returns *)
  use_baseline : bool;  (** Whether to use a baseline (value function) *)
  reward_scale : float;  (** Scale factor applied to rewards *)
  entropy_coef : float;  (** Coefficient for entropy regularization *)
  max_episode_steps : int;  (** Maximum steps per episode *)
}
(** Configuration for REINFORCE algorithm. *)

val default_config : config
(** Default configuration:
    - [learning_rate = 0.001]
    - [gamma = 0.99]
    - [use_baseline = false]
    - [reward_scale = 1.0]
    - [entropy_coef = 0.01]
    - [max_episode_steps = 1000] *)

type t
(** REINFORCE agent state. Encapsulates policy network, optional baseline,
    optimizers, and training state. *)

type update_metrics = {
  episode_return : float;  (** Undiscounted episode return *)
  episode_length : int;  (** Number of steps in episode *)
  avg_entropy : float;  (** Average policy entropy *)
  avg_log_prob : float;  (** Average log probability of actions *)
  adv_mean : float;  (** Mean of advantages (or returns if no baseline) *)
  adv_std : float;  (** Std of advantages (or returns if no baseline) *)
  value_loss : float option;  (** Value loss if using baseline *)
}
(** Metrics from a single update step. *)

val create :
  policy_network:Kaun.module_ ->
  ?baseline_network:Kaun.module_ ->
  n_actions:int ->
  rng:Rune.Rng.key ->
  config ->
  t
(** [create ~policy_network ~baseline_network ~n_actions ~rng config] creates a
    REINFORCE agent.

    Parameters:
    - [policy_network]: Kaun network that outputs action logits
    - [baseline_network]: Optional value network (required if
      [config.use_baseline = true])
    - [n_actions]: Number of discrete actions
    - [rng]: Random number generator for initialization
    - [config]: Algorithm configuration

    The policy network should take observations and output logits of shape
    [batch_size, n_actions]. The baseline network (if provided) should output
    values of shape [batch_size, 1].

    @raise Invalid_argument
      if [use_baseline = true] but [baseline_network] is [None]. *)

val predict :
  t ->
  (float, Rune.float32_elt) Rune.t ->
  training:bool ->
  (int32, Rune.int32_elt) Rune.t * float
(** [predict agent obs ~training] computes an action from an observation.

    When [training = true], samples from the policy distribution. When
    [training = false], selects the action with highest probability (greedy).

    Returns [(action, log_prob)] where [log_prob] is log π(a|s) for the selected
    action.

    The observation should be a Rune tensor of float32 values. The action
    returned is a scalar int32 tensor. *)

val update :
  t ->
  ( (float, Rune.float32_elt) Rune.t,
    (int32, Rune.int32_elt) Rune.t )
  Fehu.Trajectory.t ->
  t * update_metrics
(** [update agent trajectory] performs one REINFORCE update.

    Computes returns from trajectory rewards, calculates policy gradients, and
    updates both policy and baseline (if present). Returns updated agent and
    training metrics.

    The trajectory should contain one complete episode. *)

val learn :
  t ->
  env:
    ( (float, Rune.float32_elt) Rune.t,
      (int32, Rune.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  total_timesteps:int ->
  ?callback:(iteration:int -> metrics:update_metrics -> bool) ->
  unit ->
  t
(** [learn agent ~env ~total_timesteps ~callback ()] trains the agent.

    Repeatedly collects episodes and performs updates until [total_timesteps]
    environment steps have been executed. The callback is called after each
    episode update with the iteration number and metrics. If the callback
    returns [false], training stops early.

    Time complexity: O(total_timesteps × network_forward_time). *)

val save_to_file : t -> path:string -> unit
(** [save_to_file agent ~path] writes the agent state to a single snapshot file.

    The snapshot stores configuration, RNG, policy and optional baseline
    parameters, and optimizer state. Trajectories and replay buffers are not
    included. *)

val load_from_file :
  path:string ->
  policy_network:Kaun.module_ ->
  policy_optimizer:Kaun.Optimizer.algorithm ->
  ?baseline_network:Kaun.module_ ->
  ?baseline_optimizer:Kaun.Optimizer.algorithm ->
  unit ->
  (t, string) result
(** [load_from_file ~policy_network ~policy_optimizer ?baseline_network
     ?baseline_optimizer ~path] reconstructs an agent from a snapshot file
    produced by {!save_to_file}.

    When the saved configuration uses a baseline, both [baseline_network] and
    [baseline_optimizer] must be provided. Returns either the restored agent or
    an error message. *)
