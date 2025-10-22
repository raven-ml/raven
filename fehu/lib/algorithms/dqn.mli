(** DQN (Deep Q-Network) algorithm.

open Kaun

    DQN is an off-policy value-based algorithm that learns Q-values (expected
    returns) for state-action pairs using neural network function approximation.
    It combines Q-learning with experience replay and target networks for stable
    training.

    {1 Algorithm}

    DQN follows these steps:
    - Collect transitions (s, a, r, s') using epsilon-greedy exploration
    - Store transitions in an experience replay buffer
    - Sample random minibatches from the buffer
    - Update Q-network using TD targets from a frozen target network
    - Periodically update target network by copying Q-network parameters

    The algorithm minimizes the temporal difference error:
    {v L = E[(Q(s,a) - (r + γ max_a' Q_target(s', a')))²] v}

    {1 Usage}

    Basic usage:
    {[
      open Fehu

      (* Create Q-network *)
      let q_net = Kaun.Layer.sequential [
        Kaun.Layer.linear ~in_features:4 ~out_features:64 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:64 ~out_features:2 ();
      ] in

      (* Initialize agent *)
      let agent = Dqn.create
        ~q_network:q_net
        ~n_actions:2
        ~rng:(Rune.Rng.key 42)
        Dqn.{ default_config with batch_size = 64 }
      in

      (* Train *)
      let agent = Dqn.learn agent ~env ~total_timesteps:100_000 () in

      (* Use trained policy (greedy) *)
      let action = Dqn.predict agent obs ~epsilon:0.0
    ]}

    Manual training loop:
    {[
      (* Collect transition *)
      let obs, _info = Env.reset env () in
      let action = Dqn.predict agent obs ~epsilon:0.1 in
      let transition = Env.step env action in

      (* Store in buffer *)
      Dqn.add_transition agent ~observation:obs ~action
        ~reward:transition.reward ~next_observation:transition.observation
        ~terminated:transition.terminated ~truncated:transition.truncated;

      (* Update Q-network *)
      let loss, avg_q = Dqn.update agent in

      (* Periodically update target network *)
      if episode mod 10 = 0 then Dqn.update_target_network agent
    ]}

    {1 Key Features}

    - {b Experience Replay}: Breaks correlation between consecutive samples by
      randomly sampling from a replay buffer of past transitions.
    - {b Target Network}: Uses a separate, periodically-updated network for
      computing TD targets, improving stability.
    - {b Epsilon-Greedy Exploration}: Balances exploration and exploitation with
      decaying epsilon parameter.
    - {b Off-Policy}: Can learn from any transitions, enabling experience reuse
      and data-efficient learning.

    {1 When to Use DQN}

    - Discrete action spaces (e.g., game controls, navigation)
    - Environments where off-policy learning is beneficial
    - Tasks requiring sample efficiency through experience replay
    - Problems with deterministic or near-deterministic dynamics

    For continuous action spaces, consider SAC or DDPG (coming soon). *)

type config = {
  learning_rate : float;
      (** Learning rate for Q-network optimizer (default: 0.001) *)
  gamma : float;  (** Discount factor for future rewards (default: 0.99) *)
  epsilon_start : float;
      (** Initial exploration rate (default: 1.0 = fully random) *)
  epsilon_end : float;
      (** Final exploration rate (default: 0.01 = mostly greedy) *)
  epsilon_decay : float;
      (** Decay rate for epsilon. Epsilon decays as: epsilon_end +
          (epsilon_start
          - epsilon_end) * exp(-timesteps / epsilon_decay) (default: 1000.0) *)
  batch_size : int;
      (** Number of transitions to sample per update (default: 32) *)
  buffer_capacity : int;  (** Maximum size of replay buffer (default: 10_000) *)
  target_update_freq : int;
      (** Update target network every N episodes (default: 10) *)
}
(** Configuration for DQN training.

    {b Typical values}:
    - [learning_rate]: 0.0001-0.001 for stable convergence
    - [gamma]: 0.95-0.99 depending on episode length
    - [epsilon_start]: 1.0 to ensure initial exploration
    - [epsilon_end]: 0.01-0.1 for some exploration after convergence
    - [epsilon_decay]: 500-2000 for gradual exploration-exploitation transition
    - [batch_size]: 32-256 depending on memory and compute
    - [buffer_capacity]: 10_000-1_000_000 depending on problem complexity
    - [target_update_freq]: 5-100 episodes for stability *)

val default_config : config
(** Default configuration:
    - [learning_rate = 0.001]
    - [gamma = 0.99]
    - [epsilon_start = 1.0]
    - [epsilon_end = 0.01]
    - [epsilon_decay = 1000.0]
    - [batch_size = 32]
    - [buffer_capacity = 10_000]
    - [target_update_freq = 10]

    These defaults work reasonably for simple environments like CartPole or
    GridWorld. Adjust for your specific problem. *)

type t = {
  q_network : Kaun.Layer.module_;
  mutable q_params : Bigarray.float32_elt Kaun.Ptree.t;
  target_network : Kaun.Layer.module_;
  mutable target_params : Bigarray.float32_elt Kaun.Ptree.t;
  optimizer : Bigarray.float32_elt Kaun.Optimizer.gradient_transformation;
  mutable opt_state : Bigarray.float32_elt Kaun.Optimizer.opt_state;
  replay_buffer :
    ((float, Bigarray.float32_elt) Rune.t,
     (int32, Bigarray.int32_elt) Rune.t)
    Fehu.Buffer.Replay.t;
  mutable rng : Rune.Rng.key;
  n_actions : int;
  config : config;
}

type update_metrics = {
  episode_return : float;  (** Total reward for the episode *)
  episode_length : int;  (** Number of steps in the episode *)
  epsilon : float;  (** Current exploration rate *)
  avg_q_value : float;  (** Average Q-value from the batch *)
  loss : float;  (** TD loss from the batch update *)
}
(** Metrics from a training episode.

    These metrics help monitor training progress:
    - [episode_return] should increase over time as the policy improves
    - [episode_length] may increase or decrease depending on the task
    - [epsilon] decays from [epsilon_start] to [epsilon_end]
    - [avg_q_value] should stabilize as Q-values converge
    - [loss] should decrease and stabilize (though may fluctuate) *)

val create :
  q_network:Kaun.module_ -> n_actions:int -> rng:Rune.Rng.key -> config -> t
(** [create ~q_network ~n_actions ~rng config] creates a DQN agent.

    Parameters:
    - [q_network]: Kaun network that takes observations and outputs Q-values for
      each action. Should output shape [batch_size, n_actions].
    - [n_actions]: Number of discrete actions in the environment.
    - [rng]: Random number generator key for initialization and exploration.
    - [config]: DQN configuration (learning rate, gamma, epsilon, etc).

    The Q-network should be a standard feedforward network. For image
    observations, use convolutional layers. For vector observations, use
    fully-connected layers.

    Example Q-network architectures:
    {[
      (* For vector observations *)
      let q_net =
        Kaun.Layer.sequential
          [
            Kaun.Layer.linear ~in_features:4 ~out_features:128 ();
            Kaun.Layer.relu ();
            Kaun.Layer.linear ~in_features:128 ~out_features:128 ();
            Kaun.Layer.relu ();
            Kaun.Layer.linear ~in_features:128 ~out_features:n_actions ();
          ]

      (* For image observations *)
      let q_net =
        Kaun.Layer.sequential
          [
            Kaun.Layer.conv2d ~in_channels:4 ~out_channels:32 ~kernel_size:8
              ~stride:4 ();
            Kaun.Layer.relu ();
            Kaun.Layer.conv2d ~in_channels:32 ~out_channels:64 ~kernel_size:4
              ~stride:2 ();
            Kaun.Layer.relu ();
            Kaun.Layer.flatten ();
            Kaun.Layer.linear ~in_features:3136 ~out_features:512 ();
            Kaun.Layer.relu ();
            Kaun.Layer.linear ~in_features:512 ~out_features:n_actions ();
          ]
    ]} *)

val predict :
  t ->
  (float, Bigarray.float32_elt) Rune.t ->
  epsilon:float ->
  (int32, Bigarray.int32_elt) Rune.t
(** [predict agent obs ~epsilon] selects an action using epsilon-greedy policy.

    With probability [epsilon], selects a random action (exploration). With
    probability [1 - epsilon], selects the action with highest Q-value
    (exploitation).

    Parameters:
    - [agent]: DQN agent.
    - [obs]: Observation tensor of shape [obs_dim] or [batch_size, obs_dim].
      Automatically handles batching if needed.
    - [epsilon]: Exploration rate in \[0, 1\]. Use 0.0 for fully greedy policy
      (no exploration), 1.0 for fully random policy.

    Returns action as int32 scalar tensor.

    Example:
    {[
      (* During training with decaying exploration *)
      let epsilon = compute_epsilon ~timesteps in
      let action = Dqn.predict agent obs ~epsilon in

      (* During evaluation (no exploration) *)
      let action = Dqn.predict agent obs ~epsilon:0.0
    ]} *)

val add_transition :
  t ->
  observation:(float, Bigarray.float32_elt) Rune.t ->
  action:(int32, Bigarray.int32_elt) Rune.t ->
  reward:float ->
  next_observation:(float, Bigarray.float32_elt) Rune.t ->
  terminated:bool ->
  truncated:bool ->
  unit
(** [add_transition agent ~observation ~action ~reward ~next_observation
     ~terminated ~truncated] stores a transition in the replay buffer.

    Parameters:
    - [agent]: DQN agent.
    - [observation]: Current state observation (without batch dimension).
    - [action]: Action taken (int32 scalar).
    - [reward]: Immediate reward received.
    - [next_observation]: Resulting next state (without batch dimension).
    - [terminated]: Whether episode ended naturally (reached goal/failure
      state).
    - [truncated]: Whether episode was artificially truncated (timeout).

    Transitions are stored in a circular buffer. When the buffer is full, oldest
    transitions are overwritten.

    The distinction between [terminated] and [truncated] matters for
    bootstrapping: terminal states have value 0, while truncated states may have
    non-zero value. *)

val update : t -> float * float
(** [update agent] performs a single gradient update on the Q-network.

    Samples a batch from the replay buffer, computes TD targets using the target
    network, and updates the Q-network parameters using gradient descent on the
    TD error.

    Returns [(loss, avg_q_value)] where:
    - [loss]: Mean squared TD error for the batch
    - [avg_q_value]: Average Q-value predicted for the batch

    If the replay buffer has fewer samples than [batch_size], returns
    [(0.0, 0.0)] without performing an update.

    The TD target is computed as:
    {v y = r + γ max_a' Q_target(s', a') v}
    where Q_target is the frozen target network.

    Call this function after each environment step during training. *)

val update_target_network : t -> unit
(** [update_target_network agent] updates the target network by copying
    Q-network parameters.

    Should be called periodically (every [config.target_update_freq] episodes)
    to keep the target network stable. Frequent updates can cause instability,
    while infrequent updates can slow learning.

    Example:
    {[
      if episode mod agent.config.target_update_freq = 0 then
        Dqn.update_target_network agent
    ]} *)

val learn :
  t ->
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  total_timesteps:int ->
  ?callback:(episode:int -> metrics:update_metrics -> bool) ->
  ?warmup_steps:int ->
  unit ->
  t
(** [learn agent ~env ~total_timesteps ~callback ~warmup_steps ()] trains the
    DQN agent on an environment.

    Runs episodes until [total_timesteps] is reached. Each episode: 1. Resets
    environment 2. Collects transitions using epsilon-greedy policy 3. Stores
    transitions in replay buffer 4. Samples batches and updates Q-network 5.
    Periodically updates target network

    Parameters:
    - [agent]: DQN agent to train.
    - [env]: Environment to train on.
    - [total_timesteps]: Total number of environment steps to train for.
    - [callback]: Optional callback called after each episode with episode
      number and metrics. Return [false] to stop training early. Default always
      returns [true].
    - [warmup_steps]: Number of initial steps to collect before starting
      training (filling replay buffer). Default: [batch_size].

    Returns the trained agent.

    Example with callback:
    {[
      let agent =
        Dqn.learn agent ~env ~total_timesteps:100_000
          ~callback:(fun ~episode ~metrics ->
            if episode mod 10 = 0 then
              Printf.printf "Episode %d: Return = %.2f, Loss = %.4f\n" episode
                metrics.episode_return metrics.loss;
            true (* continue training *))
          ()
    ]}

    The warmup phase collects random experiences before training begins. This
    ensures the replay buffer has diverse samples before Q-network updates
    start. *)

(** [save agent path] saves the agent state to the specified directory.
    
    Creates the following files:
    - q_params.safetensors: Q-network weights
    - target_params.safetensors: Target network weights  
    - opt_state.safetensors: Optimizer state
    - metadata.json: Config, n_actions, and RNG seed
    Note: The replay buffer is not saved. *)
val save : t -> string -> unit


(** [load path ~q_network ~n_actions] loads an agent from the specified directory.
    
    @param path Directory containing the saved checkpoint
    @param q_network Network architecture (must match the saved agent)
    @param n_actions Number of actions (must match the saved agent)
    @raise Failure if n_actions doesn't match or files are missing
    Note: The replay buffer starts empty and optimizer is reinitialized. *)
val load : string -> q_network:Kaun.module_ -> n_actions:int -> t
