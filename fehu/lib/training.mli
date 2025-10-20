(** Training utilities for reinforcement learning algorithms.

    This module provides core computational functions for RL training, including
    advantage estimation, policy gradient losses, value function losses, and
    evaluation utilities. All functions operate on arrays for efficient batch
    processing.

    {1 Advantage Estimation}

    Advantage estimation computes the expected improvement of taking an action
    over the average. GAE balances bias and variance by combining n-step
    returns. Monte Carlo returns provide unbiased but high-variance estimates by
    summing rewards to episode termination. *)

val compute_gae :
  rewards:float array ->
  values:float array ->
  dones:bool array ->
  gamma:float ->
  gae_lambda:float ->
  float array * float array
(** [compute_gae ~rewards ~values ~dones ~gamma ~gae_lambda] computes advantages
    and returns using Generalized Advantage Estimation.

    Returns [(advantages, returns)] where advantages measure how much better an
    action was than expected, and returns are the advantage plus value baseline.

    GAE uses an exponentially-weighted average of n-step TD errors:
    {v
      δ_t = r_t + γ V(s_{t+1}) (1 - done_{t+1}) - V(s_t)
      A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
      R_t = A_t + V(s_t)
    v}

    Terminal states ([dones] = true) have zero value for the next state,
    truncating the advantage sum. This prevents bootstrapping across episode
    boundaries.

    Parameters:
    - [rewards]: Immediate rewards at each timestep
    - [values]: Value estimates V(s) from the critic at each timestep
    - [dones]: Terminal flags; true if episode ended (terminated OR truncated)
    - [gamma]: Discount factor, typically 0.99. Higher values increase weight of
      future rewards
    - [gae_lambda]: GAE lambda parameter, typically 0.95. Lambda = 0 gives
      one-step TD (low variance, high bias), lambda = 1 gives Monte Carlo (high
      variance, low bias)

    Time complexity: O(n) where n = length of arrays.

    @raise Invalid_argument if arrays have different lengths. *)

val compute_returns :
  rewards:float array -> dones:bool array -> gamma:float -> float array
(** [compute_returns ~rewards ~dones ~gamma] computes Monte Carlo returns.

    Returns discounted cumulative rewards without bootstrapping from value
    estimates. The return at timestep t is:
    {v
      R_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
    v}

    Terminal states reset the accumulation to zero, preventing reward bleeding
    across episode boundaries.

    Use this for algorithms that don't maintain a value function or when
    unbiased return estimates are required despite higher variance.

    Time complexity: O(n) where n = length of arrays.

    @raise Invalid_argument if arrays have different lengths. *)

(** {1 Loss Computations}

    Loss functions for policy optimization. Policy gradient losses measure how
    well actions align with advantages. Value losses train the critic to predict
    returns accurately. *)

val policy_gradient_loss :
  log_probs:float array ->
  advantages:float array ->
  ?normalize:bool ->
  unit ->
  float
(** [policy_gradient_loss ~log_probs ~advantages ~normalize ()] computes the
    policy gradient loss.

    Returns the negative mean of log probabilities weighted by advantages:
    {v -mean(log_prob * advantage) v}

    This gradient estimator increases the probability of actions with positive
    advantages and decreases probability of actions with negative advantages.
    Used by REINFORCE and A2C.

    Advantage normalization (enabled by default) standardizes advantages to zero
    mean and unit variance, improving training stability by keeping gradients in
    a consistent range.

    Parameters:
    - [log_probs]: Log probabilities log π(a|s) of actions taken
    - [advantages]: Advantage estimates for each action
    - [normalize]: Whether to normalize advantages (default: true)

    Time complexity: O(n) where n = length of arrays.

    @raise Invalid_argument if arrays have different lengths. *)

val ppo_clip_loss :
  log_probs:float array ->
  old_log_probs:float array ->
  advantages:float array ->
  clip_range:float ->
  float
(** [ppo_clip_loss ~log_probs ~old_log_probs ~advantages ~clip_range] computes
    the PPO clipped surrogate loss.

    Returns the negative mean of the clipped objective:
    {v -mean(min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)) v}
    where ratio = exp(log_prob - old_log_prob) and ε is [clip_range].

    The probability ratio measures how much the policy changed since data was
    collected. Clipping prevents excessively large updates that destabilize
    training. When ratio exceeds [1 + clip_range] or falls below
    [1 - clip_range], the gradient is zeroed, limiting the policy change per
    update.

    Advantages are automatically normalized to zero mean and unit variance.

    Parameters:
    - [log_probs]: Log probabilities log π_new(a|s) from current policy
    - [old_log_probs]: Log probabilities log π_old(a|s) from policy that
      collected data
    - [advantages]: Advantage estimates for each action
    - [clip_range]: Clipping threshold ε, typically 0.2

    Time complexity: O(n) where n = length of arrays.

    @raise Invalid_argument if arrays have different lengths. *)

val value_loss :
  values:float array ->
  returns:float array ->
  ?clip:float * float array ->
  unit ->
  float
(** [value_loss ~values ~returns ~clip_range ()] computes the value function
    loss.

    Returns the mean squared error between predicted values and target returns:
    {v mean((V - R)^2) v}

    - If [clip] is [None], computes the mean-squared error (MSE) between
      [values] and [returns].
    - If [clip = Some ((clip_range, old_values))], applies PPO-style clipping:

    In that case, PPO-style value clipping is applied:
    [value_clipped = old_values + clamp(values - old_values, ±clip_range)], and
    the loss for each element is
    [max((values - returns)^2, (value_clipped - returns)^2)]. This prevents
    large critic updates that destabilize training.

    Parameters:
    - [values]: Predicted values V(s) from the critic
    - [returns]: Target returns (from GAE or Monte Carlo)
    - [(clip_range, old_values)]: Optional clip range and previous value

    estimates before the update Time complexity: O(n) where n = length of
    arrays.

    @raise Invalid_argument if arrays have different lengths.
    @raise Invalid_argument if clip_range < 0 *)

(** {1 Evaluation}

    Policy evaluation utilities for measuring agent performance. *)

type stats = {
  mean_reward : float;  (** Average cumulative reward per episode *)
  std_reward : float;  (** Standard deviation of episode rewards *)
  mean_length : float;  (** Average episode length in steps *)
  n_episodes : int;  (** Number of episodes evaluated *)
}
(** Evaluation statistics summarizing policy performance. *)

val evaluate :
  ('obs, 'act, 'render) Env.t ->
  policy:('obs -> 'act) ->
  ?n_episodes:int ->
  ?max_steps:int ->
  unit ->
  stats
(** [evaluate env ~policy ~n_episodes ~max_steps ()] evaluates a policy's
    performance.

    Runs the policy for multiple episodes, collecting cumulative rewards and
    episode lengths. Returns statistics summarizing the policy's performance.
    Use this to assess training progress or compare different policies.

    Episodes run until natural termination or truncation, or until [max_steps]
    is reached. The environment is automatically reset between episodes.

    Parameters:
    - [env]: Environment to evaluate in
    - [policy]: Function mapping observations to actions. Should be
      deterministic for reproducible evaluation or stochastic for exploration
      assessment
    - [n_episodes]: Number of episodes to run (default: 10)
    - [max_steps]: Maximum steps per episode before truncation (default: 1000)

    Time complexity: O(n_episodes × max_steps). *)

(** {1 Utilities}

    Helper functions for data processing and diagnostics. *)

val normalize : float array -> ?eps:float -> unit -> float array
(** [normalize arr ~eps ()] normalizes an array to zero mean and unit variance.

    Returns a new array where elements are transformed to:
    {v (x - mean) / (std + eps) v}

    Numerical stability constant [eps] prevents division by zero when the array
    has no variance (default: 1e-8).

    Empty arrays are returned unchanged.

    Time complexity: O(n) where n = length of array. *)

val explained_variance : y_pred:float array -> y_true:float array -> float
(** [explained_variance ~y_pred ~y_true] measures prediction quality.

    Returns the explained variance coefficient:
    {v 1 - Var[y_true - y_pred] / Var[y_true] v}

    This metric helps diagnose value function training. Values range from -∞ to
    1:
    - 1.0: Perfect predictions matching true values exactly
    - 0.0: Predictions are as good as predicting the mean
    - < 0: Predictions are worse than predicting the mean

    Use this to monitor critic training. Increasing explained variance indicates
    the value function is learning to predict returns more accurately.

    Time complexity: O(n) where n = length of arrays.

    @raise Invalid_argument if arrays have different lengths. *)
