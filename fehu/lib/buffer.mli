(** Experience collection buffers for reinforcement learning algorithms.

    This module provides two buffer types for storing agent-environment
    interactions: replay buffers for off-policy algorithms and rollout buffers
    for on-policy algorithms. Both support efficient batch sampling and storage
    management.

    {1 Buffer Types}

    Replay buffers store transitions with complete state information, supporting
    off-policy algorithms like DQN, SAC, and TD3. They maintain a fixed-capacity
    circular buffer that overwrites oldest experiences when full.

    Rollout buffers store sequential steps with optional value estimates and log
    probabilities, supporting on-policy algorithms like PPO and A2C. They
    compute advantages using Generalized Advantage Estimation (GAE) before
    returning batches.

    {1 Usage}

    Create a replay buffer and add transitions:
    {[
      let buffer = Buffer.Replay.create ~capacity:10000 in
      let transition =
        { observation; action; reward; next_observation; terminated; truncated }
      in
      Buffer.Replay.add buffer transition
    ]}

    Sample a batch for training:
    {[
      let batch = Buffer.Replay.sample buffer ~rng ~batch_size:32 in
      Array.iter (fun t -> (* train on transition *)) batch
    ]}

    Use rollout buffers for on-policy data:
    {[
      let buffer = Buffer.Rollout.create ~capacity:2048 in
      Buffer.Rollout.add buffer
        { observation; action; reward; terminated; truncated; value; log_prob };
      Buffer.Rollout.compute_advantages buffer ~last_value ~last_done ~gamma:0.99 ~gae_lambda:0.95;
      let steps, advantages, returns = Buffer.Rollout.get buffer
    ]} *)

type ('obs, 'act) transition = {
  observation : 'obs;  (** Current state observation *)
  action : 'act;  (** Action taken in current state *)
  reward : float;  (** Immediate reward received *)
  next_observation : 'obs;  (** Resulting next state observation *)
  terminated : bool;  (** Whether episode ended naturally *)
  truncated : bool;  (** Whether episode was artificially truncated *)
}
(** Basic transition for off-policy algorithms.

    Represents a complete state transition containing both the current and next
    observations. Used by replay buffers for algorithms that learn from
    arbitrary past experiences. *)

type ('obs, 'act) step = {
  observation : 'obs;  (** State observation at this step *)
  action : 'act;  (** Action taken at this step *)
  reward : float;  (** Immediate reward received *)
  terminated : bool;  (** Whether episode ended at this step *)
  truncated : bool;  (** Whether the episode was truncated at this step *)
  value : float option;  (** Value estimate V(s) from critic, if available *)
  log_prob : float option;
      (** Log probability log π(a|s) from policy, if available *)
}
(** Rollout step for on-policy algorithms.

    Represents a single timestep with optional policy information. Unlike
    transitions, steps do not store next observations since on-policy data is
    processed sequentially. Value estimates and log probabilities support policy
    gradient methods. *)

(** {1 Replay Buffer (Off-Policy: DQN, SAC, TD3)} *)

(** Replay buffer for off-policy algorithms.

    Implements a fixed-capacity circular buffer storing complete transitions.
    When capacity is reached, oldest transitions are overwritten. Supports
    uniform random sampling for breaking temporal correlations in training data.

    All transitions are stored contiguously in memory. Observation and action
    arrays are lazily initialized on the first call to {!add}. *)
module Replay : sig
  type ('obs, 'act) t
  (** Replay buffer storing transitions of observations and actions. *)

  val create : capacity:int -> ('obs, 'act) t
  (** [create ~capacity] creates an empty replay buffer.

      The buffer stores up to [capacity] transitions. When full, adding new
      transitions overwrites the oldest ones in circular fashion.

      @raise Invalid_argument if [capacity <= 0]. *)

  val add : ('obs, 'act) t -> ('obs, 'act) transition -> unit
  (** [add buffer transition] stores a transition in the buffer.

      Appends [transition] to the buffer, overwriting the oldest transition if
      at capacity. The first call initializes internal storage arrays based on
      the observation and action types.

      Time complexity: O(1). *)

  val add_many : ('obs, 'act) t -> ('obs, 'act) transition array -> unit
  (** [add_many buffer transitions] appends a batch of transitions.

      Equivalent to repeated calls to {!add} but initializes internal storage at
      most once and avoids repeated bounds checks. *)

  val sample :
    ('obs, 'act) t ->
    rng:Rune.Rng.key ->
    batch_size:int ->
    ('obs, 'act) transition array
  (** [sample buffer ~rng ~batch_size] returns uniformly sampled transitions.

      Samples [batch_size] transitions uniformly at random from the buffer. If
      [batch_size] exceeds the current buffer size, samples min(batch_size,
      size) transitions instead. Sampling is with replacement.

      Time complexity: O(batch_size).

      @raise Invalid_argument if [batch_size <= 0] or buffer is empty. *)

  val sample_arrays :
    ('obs, 'act) t ->
    rng:Rune.Rng.key ->
    batch_size:int ->
    'obs array * 'act array * float array * 'obs array * bool array * bool array
  (** [sample_arrays buffer ~rng ~batch_size] returns a struct-of-arrays batch.

      The arrays share references with the underlying transitions (no copying of
      observations/actions is performed). Useful for vectorized algorithms that
      operate on homogeneous arrays. *)

  val sample_tensors :
    (('obs, 'obs_layout) Rune.t, ('act, 'act_layout) Rune.t) t ->
    rng:Rune.Rng.key ->
    batch_size:int ->
    ('obs, 'obs_layout) Rune.t
    * ('act, 'act_layout) Rune.t
    * (float, Rune.float32_elt) Rune.t
    * ('obs, 'obs_layout) Rune.t
    * Rune.bool_t
    * Rune.bool_t
  (** [sample_tensors buffer ~rng ~batch_size] returns a struct-of-arrays batch
      stacked into tensors.

      This is a convenience wrapper over {!sample_arrays} that stacks the
      sampled observations and actions along a leading batch dimension and
      converts rewards/flags into tensors so downstream code can remain
      vectorized. *)

  val size : ('obs, 'act) t -> int
  (** [size buffer] returns the current number of transitions stored.

      Returns values between 0 and capacity. *)

  val is_full : ('obs, 'act) t -> bool
  (** [is_full buffer] checks whether the buffer has reached capacity. *)

  val clear : ('obs, 'act) t -> unit
  (** [clear buffer] removes all transitions from the buffer.

      Resets size to 0 and write position to 0 while keeping internal storage
      arrays allocated for reuse. *)
end

(** {1 Rollout Buffer (On-Policy: PPO, A2C)} *)

(** Rollout buffer for on-policy algorithms.

    Stores sequential steps from policy rollouts along with value estimates and
    log probabilities. Computes advantages using Generalized Advantage
    Estimation (GAE) before returning data for training.

    Unlike replay buffers, rollout buffers process data sequentially and must be
    filled completely before calling {!compute_advantages}. After retrieving
    data with {!get}, the buffer is automatically cleared for the next rollout.
*)
module Rollout : sig
  type ('obs, 'act) t
  (** Rollout buffer storing steps and computed advantages. *)

  val create : capacity:int -> ('obs, 'act) t
  (** [create ~capacity] creates an empty rollout buffer.

      The buffer stores exactly [capacity] steps before requiring a call to
      {!get} or {!clear}. Unlike replay buffers, rollout buffers do not
      overwrite old data.

      @raise Invalid_argument if [capacity <= 0]. *)

  val add : ('obs, 'act) t -> ('obs, 'act) step -> unit
  (** [add buffer step] appends a step to the buffer.

      Stores [step] at the current position. The first call initializes internal
      storage based on the step types.

      Time complexity: O(1).

      @raise Invalid_argument if buffer is full. Call {!get} or {!clear} first.
  *)

  val compute_advantages :
    ('obs, 'act) t ->
    last_value:float ->
    last_done:bool ->
    gamma:float ->
    gae_lambda:float ->
    unit
  (** [compute_advantages buffer ~last_value ~last_done ~gamma ~gae_lambda]
      computes advantages using GAE.

      Updates the buffer in-place, computing advantages and returns for all
      stored steps using the GAE formula:
      {v
        δ_t = r_t + γ V(s_{t+1}) (1 - done_{t+1}) - V(s_t)
        A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        R_t = A_t + V(s_t)
      v}

      Bootstrap values handle incomplete episodes at the end of the buffer. If
      the last step was terminal ([last_done = true]), [last_value] is ignored
      and the terminal value is treated as 0. Otherwise, [last_value] provides
      V(s_final).

      Call this function after filling the buffer and before {!get}.

      Parameters:
      - [last_value]: Value estimate for the state following the last step in
        the buffer. Used for bootstrapping incomplete episodes.
      - [last_done]: Whether an episode terminated at the last step. If [true],
        terminal value is 0 regardless of [last_value].
      - [gamma]: Discount factor, typically 0.99. Controls the weight of future
        rewards.
      - [gae_lambda]: GAE lambda parameter, typically 0.95. Controls the
        bias-variance tradeoff in advantage estimation. Lambda = 0 gives
        one-step TD, lambda = 1 gives Monte Carlo returns. *)

  val get :
    ('obs, 'act) t -> ('obs, 'act) step array * float array * float array
  (** [get buffer] retrieves all data and clears the buffer.

      Returns [(steps, advantages, returns)] where:
      - [steps]: All stored steps in chronological order
      - [advantages]: Advantages computed by {!compute_advantages}
      - [returns]: Returns computed by {!compute_advantages}

      The buffer is cleared after this call, allowing a new rollout to begin. If
      {!compute_advantages} was not called, advantages and returns are all
      zeros.

      Time complexity: O(capacity). *)

  val size : ('obs, 'act) t -> int
  (** [size buffer] returns the current number of steps stored.

      Returns values between 0 and capacity. *)

  val is_full : ('obs, 'act) t -> bool
  (** [is_full buffer] checks whether the buffer has reached capacity. *)

  val clear : ('obs, 'act) t -> unit
  (** [clear buffer] removes all steps from the buffer.

      Resets size to 0. Does not deallocate internal storage. Use this to
      discard partial rollouts. *)
end
