# Collection, Buffers, and Evaluation

This guide covers trajectory collection, replay buffers, generalized advantage
estimation, and policy evaluation.

## Trajectory Collection

`Collect` gathers agent-environment interactions into structure-of-arrays form
for batch processing.

### Rollout

`Collect.rollout` collects a fixed number of transitions. It resets the
environment at the start and automatically on episode boundaries:

<!-- $MDX skip -->
```ocaml
open Fehu

let rng = Rune.Rng.key 42
let env = Fehu_envs.Cartpole.make ~rng ()

(* The policy receives an observation and returns
   (action, log_prob option, value_estimate option) *)
let policy obs =
  let act, _ = Space.sample (Env.action_space env)
    ~rng:(Env.take_rng env) in
  (act, None, None)

let trajectory = Collect.rollout env ~policy ~n_steps:1024
```

The returned trajectory contains parallel arrays:

<!-- $MDX skip -->
```ocaml
let n = Collect.length trajectory  (* 1024 *)
let obs = trajectory.observations       (* 'obs array *)
let acts = trajectory.actions           (* 'act array *)
let rews = trajectory.rewards           (* float array *)
let next_obs = trajectory.next_observations  (* 'obs array *)
let terms = trajectory.terminated       (* bool array *)
let truncs = trajectory.truncated       (* bool array *)
let infos = trajectory.infos            (* Info.t array *)
let log_ps = trajectory.log_probs       (* float array option *)
let vals = trajectory.values            (* float array option *)
```

When the policy returns `Some log_prob` or `Some value`, those are collected
into `log_probs` and `values`. When any return is `None`, the corresponding
field is `None` for the entire trajectory.

### Policy Signature

The policy function has the signature:

```
'obs -> 'act * float option * float option
```

The three components are:

1. **action**: the action to take
2. **log_prob** (optional): the log-probability of the action under the current policy, used for importance sampling in PPO
3. **value** (optional): the estimated value of the current state, used for GAE computation

For a simple random policy, return `None` for both:

<!-- $MDX skip -->
```ocaml
let random_policy obs =
  let act, _ = Space.sample (Env.action_space env)
    ~rng:(Env.take_rng env) in
  (act, None, None)
```

For a neural network policy with value head:

<!-- $MDX skip -->
```ocaml
let nn_policy obs =
  let logits, value = forward_pass model obs in
  let act = sample_from_logits logits in
  let log_prob = log_prob_of logits act in
  (act, Some log_prob, Some value)
```

### Episodes

`Collect.episodes` collects complete episodes, one trajectory per episode:

<!-- $MDX skip -->
```ocaml
let episodes = Collect.episodes env
  ~policy ~n_episodes:10
  ~max_steps:500 ()

(* episodes is a ('obs, 'act) Collect.t list *)
let total_rewards = List.map (fun traj ->
  Array.fold_left (+.) 0.0 traj.rewards) episodes
```

Each episode runs until termination, truncation, or `max_steps` (default
1000).

### Concatenating Trajectories

`Collect.concat` merges multiple trajectories into one:

<!-- $MDX skip -->
```ocaml
let combined = Collect.concat [traj1; traj2; traj3]
```

Optional fields (`log_probs`, `values`) are kept only if present in all inputs.

## Replay Buffers

`Buffer` provides a fixed-capacity circular buffer for off-policy experience
storage. It stores individual transitions and supports uniform random sampling.

### Creating and Filling

<!-- $MDX skip -->
```ocaml
open Fehu

let buf = Buffer.create ~capacity:10_000

(* Add transitions one at a time *)
Buffer.add buf {
  observation = obs;
  action = act;
  reward = 1.0;
  next_observation = next_obs;
  terminated = false;
  truncated = false;
}

let n = Buffer.size buf           (* number of stored transitions *)
let full = Buffer.is_full buf     (* true when at capacity *)
let cap = Buffer.capacity buf     (* 10000 *)
```

When the buffer is full, new transitions overwrite the oldest ones.

### Sampling

Draw a batch of transitions uniformly at random (with replacement):

<!-- $MDX skip -->
```ocaml
let rng = Rune.Rng.key 0
let batch, rng' = Buffer.sample buf ~rng ~batch_size:64

(* batch is a transition array *)
let obs_0 = batch.(0).observation
let rew_0 = batch.(0).reward
```

For structure-of-arrays form (more convenient for training):

<!-- $MDX skip -->
```ocaml
let (observations, actions, rewards,
     next_observations, terminated, truncated), rng' =
  Buffer.sample_arrays buf ~rng ~batch_size:64
```

### Clearing

<!-- $MDX skip -->
```ocaml
Buffer.clear buf  (* removes all transitions, keeps storage allocated *)
```

## Generalized Advantage Estimation

`Gae` computes advantages and returns for policy gradient methods. It correctly
handles the distinction between terminated and truncated episodes:

- **Terminated**: the episode ended naturally (e.g., pole fell). Bootstrap
  value is zero.
- **Truncated**: the episode was cut short (e.g., time limit). Bootstrap value
  comes from `next_values`.

### Computing Advantages

<!-- $MDX skip -->
```ocaml
open Fehu

(* From a trajectory with value estimates *)
let advantages, returns = Gae.compute
  ~rewards:trajectory.rewards
  ~values:(Option.get trajectory.values)
  ~terminated:trajectory.terminated
  ~truncated:trajectory.truncated
  ~next_values    (* V(s_{t+1}) for each step *)
  ~gamma:0.99     (* discount factor *)
  ~lambda:0.95    (* GAE smoothing parameter *)
```

When you have values from a value network and the last value estimate,
`compute_from_values` builds `next_values` for you:

<!-- $MDX skip -->
```ocaml
let advantages, returns = Gae.compute_from_values
  ~rewards:trajectory.rewards
  ~values:(Option.get trajectory.values)
  ~terminated:trajectory.terminated
  ~truncated:trajectory.truncated
  ~last_value:0.0   (* V(s_T) for the final state *)
  ~gamma:0.99
  ~lambda:0.95
```

### Monte Carlo Returns

For simpler algorithms that do not need advantages:

<!-- $MDX skip -->
```ocaml
let rets = Gae.returns
  ~rewards:trajectory.rewards
  ~terminated:trajectory.terminated
  ~truncated:trajectory.truncated
  ~gamma:0.99
```

### Normalizing Advantages

Normalize to zero mean and unit variance for training stability:

<!-- $MDX skip -->
```ocaml
let normalized = Gae.normalize advantages
(* or with custom epsilon *)
let normalized = Gae.normalize ~eps:1e-6 advantages
```

## Policy Evaluation

`Eval.run` runs a deterministic or stochastic policy over multiple episodes
and reports summary statistics:

<!-- $MDX skip -->
```ocaml
open Fehu

let rng = Rune.Rng.key 42
let env = Fehu_envs.Cartpole.make ~rng ()

(* Evaluate a random policy *)
let stats = Eval.run env
  ~policy:(fun _obs ->
    let act, _ = Space.sample (Env.action_space env)
      ~rng:(Env.take_rng env) in act)
  ~n_episodes:100
  ~max_steps:500
  ()

let () = Printf.printf
  "Episodes: %d, Mean reward: %.1f +/- %.1f, Mean length: %.0f\n"
  stats.n_episodes
  stats.mean_reward
  stats.std_reward
  stats.mean_length
```

The evaluation policy has a simpler signature than the collection policy: it
only returns an action, not log-probs or value estimates:

```
'obs -> 'act
```

`Eval.run` resets the environment between episodes. Default `n_episodes` is 10
and default `max_steps` is 1000.

## Putting It Together

A typical PPO-style training iteration using these utilities:

<!-- $MDX skip -->
```ocaml
open Fehu

(* 1. Collect rollout *)
let trajectory = Collect.rollout env
  ~policy:(fun obs ->
    let act, log_prob, value = nn_policy obs in
    (act, Some log_prob, Some value))
  ~n_steps:2048

(* 2. Compute advantages *)
let last_value = estimate_value model last_obs in
let advantages, returns = Gae.compute_from_values
  ~rewards:trajectory.rewards
  ~values:(Option.get trajectory.values)
  ~terminated:trajectory.terminated
  ~truncated:trajectory.truncated
  ~last_value
  ~gamma:0.99 ~lambda:0.95

let advantages = Gae.normalize advantages

(* 3. Update policy using trajectory data + advantages *)
(* ... your PPO update here ... *)

(* 4. Evaluate *)
let stats = Eval.run env
  ~policy:(fun obs -> greedy_action model obs)
  ~n_episodes:10 ()
```

## Next Steps

- [Getting Started](../01-getting-started/) -- installation, environments, spaces, step loop
- [Environments and Wrappers](../02-environments/) -- custom environments, wrappers, rendering, vectorized environments
