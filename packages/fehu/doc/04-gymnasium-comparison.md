# Fehu vs. Gymnasium -- A Practical Comparison

This guide explains how Fehu's reinforcement learning API relates to Python's [Gymnasium](https://gymnasium.farama.org/) (and [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for collection/buffer/GAE), focusing on:

* How core concepts map (Env, Space, step loop, wrappers)
* Where the APIs feel similar vs. deliberately different
* How to translate common Gymnasium patterns into Fehu

If you already use Gymnasium, this should be enough to become productive in Fehu quickly.

---

## 1. Big-Picture Differences

| Aspect                | Gymnasium (Python)                                 | Fehu (OCaml)                                                         |
| --------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| Language              | Dynamic, interpreted                               | Statically typed, compiled                                           |
| Environment type      | `gymnasium.Env`                                    | `('obs, 'act, 'render) Env.t`                                        |
| Observation/action    | Untyped (`np.ndarray`, `int`, etc.)                | Parametric: `'obs` and `'act` tracked in the type                    |
| Spaces                | `gymnasium.spaces.*`                               | `'a Space.t` with typed modules (`Space.Discrete`, `Space.Box`, ...) |
| Step result           | Tuple `(obs, reward, terminated, truncated, info)` | Record `Env.step` with named fields                                  |
| Wrappers              | Subclassing `gymnasium.Wrapper`                    | `Env.wrap` or composable combinators (`map_observation`, etc.)       |
| Vectorized envs       | `gymnasium.vector.SyncVectorEnv`                   | `Vec_env.create`                                                     |
| Trajectory collection | External (Stable Baselines3, TorchRL)              | Built-in: `Collect.rollout`, `Collect.episodes`                      |
| Replay buffers        | External (Stable Baselines3, TorchRL)              | Built-in: `Buffer.create`, `Buffer.add`, `Buffer.sample`             |
| GAE                   | External (Stable Baselines3)                       | Built-in: `Gae.compute`, `Gae.returns`, `Gae.normalize`              |
| Policy evaluation     | Manual loop or SB3 `evaluate_policy`               | Built-in: `Eval.run`                                                 |
| RNG                   | `np.random` / seed passed to `env.reset(seed=...)` | Implicit scope via `Nx.Rng.run ~seed`                                |
| Rendering             | String mode `"human"`, `"rgb_array"`               | Polymorphic variants `` `Human ``, `` `Rgb_array ``, etc.            |
| Mutability            | Environments are mutable objects                   | Environments are immutable handles; state is internal                |

**Fehu semantics to know (read once):**
- `Env.reset` must be called before `Env.step`. After a terminal step, another `reset` is required.
- Spaces validate observations and actions automatically -- `Env.step` raises if an action is outside the action space.
- RNG is scoped: wrap your code in `Nx.Rng.run ~seed:42 (fun () -> ...)` instead of passing seeds to individual calls.
- Trajectory collection, replay buffers, GAE, and evaluation are built into Fehu, not external libraries.

---

## 2. Spaces

### 2.1 Discrete

**Gymnasium**

```python
import gymnasium as gym

space = gym.spaces.Discrete(5)           # {0, 1, 2, 3, 4}
space = gym.spaces.Discrete(5, start=1)  # {1, 2, 3, 4, 5}

sample = space.sample()
assert space.contains(sample)
```

**Fehu**

<!-- $MDX skip -->
```ocaml
open Fehu

let space = Space.Discrete.create 5            (* {0, 1, 2, 3, 4} *)
let space = Space.Discrete.create ~start:1 5   (* {1, 2, 3, 4, 5} *)

let sample = Space.sample space
let valid  = Space.contains space sample

let n     = Space.Discrete.n space      (* 5 *)
let start = Space.Discrete.start space  (* 1 *)

(* Convert between discrete elements and ints *)
let action = Space.Discrete.of_int 3
let value  = Space.Discrete.to_int action
```

Discrete elements are `(int32, Nx.int32_elt) Nx.t` scalars, not bare OCaml ints.

### 2.2 Box (continuous)

**Gymnasium**

```python
import numpy as np

space = gym.spaces.Box(
    low=np.array([-1.0, -2.0]),
    high=np.array([1.0, 2.0]),
    dtype=np.float32,
)
sample = space.sample()
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let space =
  Space.Box.create
    ~low:[| -1.0; -2.0 |]
    ~high:[| 1.0; 2.0 |]

let sample = Space.sample space
let (low, high) = Space.Box.bounds space
```

Box elements are `(float, Nx.float32_elt) Nx.t` tensors. Infinite bounds are allowed; sampling falls back to uniform draws in `[-1e6, 1e6]` clamped to bounds.

### 2.3 Multi_binary

**Gymnasium**

```python
space = gym.spaces.MultiBinary(4)  # {0,1}^4
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let space = Space.Multi_binary.create 4
```

Elements are `(int32, Nx.int32_elt) Nx.t` vectors with values 0 or 1.

### 2.4 Multi_discrete

**Gymnasium**

```python
space = gym.spaces.MultiDiscrete([3, 5, 2])  # 3 axes: {0..2}, {0..4}, {0..1}
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let space = Space.Multi_discrete.create [| 3; 5; 2 |]
```

### 2.5 Composite spaces

**Gymnasium**

```python
space = gym.spaces.Tuple((
    gym.spaces.Discrete(3),
    gym.spaces.Box(low=0.0, high=1.0, shape=(2,)),
))

space = gym.spaces.Dict({
    "position": gym.spaces.Box(low=-10.0, high=10.0, shape=(3,)),
    "velocity": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
})
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let space =
  Space.Tuple.create [
    Space.Pack (Space.Discrete.create 3);
    Space.Pack (Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 1.0; 1.0 |]);
  ]

let space =
  Space.Dict.create [
    ("position", Space.Pack (Space.Box.create ~low:[| -10.; -10.; -10. |] ~high:[| 10.; 10.; 10. |]));
    ("velocity", Space.Pack (Space.Box.create ~low:[| -1.; -1.; -1. |] ~high:[| 1.; 1.; 1. |]));
  ]
```

Composite space elements use `Value.t` for heterogeneous data: `Tuple.element = Value.t list`, `Dict.element = (string * Value.t) list`.

### 2.6 Sequence and Text

**Gymnasium**

```python
space = gym.spaces.Sequence(gym.spaces.Discrete(5), seed=42)
space = gym.spaces.Text(max_length=32, charset="abcdef")
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let space = Space.Sequence.create ~max_length:10 (Space.Discrete.create 5)
let space = Space.Text.create ~charset:"abcdef" ~max_length:32 ()
```

### 2.7 Common operations

All space types share the same interface:

<!-- $MDX skip -->
```ocaml
let sample = Space.sample space           (* random element *)
let valid  = Space.contains space sample  (* membership test *)
let spec   = Space.spec space             (* structural description *)
let shape  = Space.shape space            (* dimensionality, if defined *)

(* Serialization via Value.t *)
let packed   = Space.pack space sample
let unpacked = Space.unpack space packed  (* (element, string) result *)

(* Edge cases for testing *)
let edges = Space.boundary_values space
```

---

## 3. Creating Environments

### 3.1 From a registry

**Gymnasium**

```python
env = gym.make("CartPole-v1", render_mode="human")
```

**Fehu** does not have a global registry. Environments are constructed directly:

<!-- $MDX skip -->
```ocaml
let env =
  Env.create
    ~id:"CartPole-v1"
    ~observation_space:(Space.Box.create
      ~low:[| -4.8; Float.neg_infinity; -0.418; Float.neg_infinity |]
      ~high:[| 4.8; Float.infinity; 0.418; Float.infinity |])
    ~action_space:(Space.Discrete.create 2)
    ~render_mode:`Human
    ~render_modes:["human"; "rgb_array"]
    ~reset:(fun _env ?options:_ () ->
      let obs = (* initial state *) in
      (obs, Info.empty))
    ~step:(fun _env action ->
      let obs = (* next state *) in
      Env.step_result ~observation:obs ~reward:1.0 ())
    ()
```

`Env.create` takes the observation space, action space, and two callbacks: `reset` and `step`. Optional `render` and `close` callbacks handle visualization and cleanup.

### 3.2 Step result construction

**Gymnasium** returns a flat tuple from `env.step()`:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**Fehu** uses a record with named fields, and provides a convenience constructor with defaults:

<!-- $MDX skip -->
```ocaml
(* Inside a step callback *)
Env.step_result
  ~observation:obs
  ~reward:1.0
  ~terminated:false
  ~truncated:false
  ~info:Info.empty
  ()

(* Defaults: reward=0., terminated=false, truncated=false, info=Info.empty *)
Env.step_result ~observation:obs ()
```

---

## 4. Step Loop

### 4.1 Basic episode

**Gymnasium**

```python
env = gym.make("CartPole-v1")
obs, info = env.reset(seed=42)

total_reward = 0.0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

env.close()
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let () =
  Nx.Rng.run ~seed:42 (fun () ->
    let env = (* create environment *) in
    let (obs, _info) = Env.reset env () in
    let obs = ref obs in
    let total_reward = ref 0.0 in
    let done_ = ref false in
    while not !done_ do
      let action = Space.sample (Env.action_space env) in
      let step = Env.step env action in
      obs := step.observation;
      total_reward := !total_reward +. step.reward;
      done_ := step.terminated || step.truncated
    done;
    Env.close env)
```

Key differences:
- RNG is scoped with `Nx.Rng.run ~seed:42` rather than passed to `reset`.
- Step results are accessed by field name (`step.observation`, `step.reward`).
- `Env.step` raises `Invalid_argument` if called without a prior `reset` or after a terminal step without resetting.

### 4.2 Multiple episodes

**Gymnasium**

```python
for episode in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

**Fehu** -- manual loop or use `Collect.episodes`:

<!-- $MDX skip -->
```ocaml
(* Manual *)
let () =
  Nx.Rng.run ~seed:0 (fun () ->
    let env = (* create environment *) in
    for _ep = 0 to 9 do
      let (obs, _info) = Env.reset env () in
      let obs = ref obs in
      let done_ = ref false in
      while not !done_ do
        let action = policy !obs in
        let step = Env.step env action in
        obs := step.observation;
        done_ := step.terminated || step.truncated
      done
    done;
    Env.close env)

(* Or use Collect.episodes directly *)
let trajs =
  Nx.Rng.run ~seed:0 (fun () ->
    let env = (* create environment *) in
    Collect.episodes env
      ~policy:(fun obs -> (policy obs, None, None))
      ~n_episodes:10 ())
```

---

## 5. Wrappers

### 5.1 Gymnasium approach: subclassing

**Gymnasium**

```python
class NormalizeObservation(gym.Wrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = mean
        self.std = std

    def observation(self, obs):
        return (obs - self.mean) / self.std

env = NormalizeObservation(env, mean=0.0, std=1.0)
```

### 5.2 Fehu approach: composable functions

**Fehu** provides `Env.wrap` for full control and specialized combinators for common patterns.

**`map_observation`** -- transform observations:

<!-- $MDX skip -->
```ocaml
let normalized_env =
  Env.map_observation
    ~observation_space:obs_space
    ~f:(fun obs _info ->
      let normalized = (* normalize obs *) in
      (normalized, Info.empty))
    env
```

**`map_action`** -- transform actions before passing to the inner env:

<!-- $MDX skip -->
```ocaml
let remapped_env =
  Env.map_action
    ~action_space:new_action_space
    ~f:(fun new_action -> (* convert to inner action *))
    env
```

**`map_reward`** -- transform rewards:

<!-- $MDX skip -->
```ocaml
let scaled_env =
  Env.map_reward
    ~f:(fun ~reward ~info -> (reward *. 0.1, info))
    env
```

**`clip_action`** -- clamp continuous actions to bounds:

<!-- $MDX skip -->
```ocaml
(* Gymnasium *)
(* from gymnasium.wrappers import ClipAction *)
(* env = ClipAction(env) *)

(* Fehu *)
let clipped_env = Env.clip_action env
```

**`clip_observation`** -- clamp observations:

<!-- $MDX skip -->
```ocaml
let clipped_env =
  Env.clip_observation
    ~low:[| -5.0; -5.0 |]
    ~high:[| 5.0; 5.0 |]
    env
```

**`time_limit`** -- enforce maximum episode length:

<!-- $MDX skip -->
```ocaml
(* Gymnasium *)
(* from gymnasium.wrappers import TimeLimit *)
(* env = TimeLimit(env, max_episode_steps=200) *)

(* Fehu *)
let limited_env = Env.time_limit ~max_episode_steps:200 env
```

### 5.3 Full custom wrapper with `Env.wrap`

When the combinators are not enough, use `Env.wrap` directly:

<!-- $MDX skip -->
```ocaml
let custom_env =
  Env.wrap
    ~observation_space:new_obs_space
    ~action_space:new_act_space
    ~reset:(fun inner ?options () ->
      let (obs, info) = Env.reset inner ?options () in
      (transform_obs obs, info))
    ~step:(fun inner action ->
      let step = Env.step inner (transform_action action) in
      { step with observation = transform_obs step.observation })
    env
```

`Env.wrap` receives the inner environment as the first argument to `reset`, `step`, `render`, and `close`. Guards (closed check, needs-reset check, space validation) are enforced automatically.

### 5.4 Composing wrappers

Wrappers compose by chaining:

<!-- $MDX skip -->
```ocaml
let env =
  base_env
  |> Env.time_limit ~max_episode_steps:500
  |> Env.clip_action
  |> Env.map_reward ~f:(fun ~reward ~info -> (reward *. 0.01, info))
```

---

## 6. Vectorized Environments

### 6.1 Synchronous vectorization

**Gymnasium**

```python
envs = gym.vector.SyncVectorEnv([
    lambda: gym.make("CartPole-v1") for _ in range(4)
])

obs, infos = envs.reset()
actions = envs.action_space.sample()  # batch of 4 actions
obs, rewards, terminated, truncated, infos = envs.step(actions)
envs.close()
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let venv =
  Vec_env.create [env1; env2; env3; env4]

let n = Vec_env.num_envs venv  (* 4 *)

let (observations, infos) = Vec_env.reset venv ()
let actions = Array.init n (fun _ -> Space.sample (Vec_env.action_space venv)) in
let step = Vec_env.step venv actions

(* step.observations : 'obs array      -- one per env *)
(* step.rewards      : float array     -- one per env *)
(* step.terminated   : bool array      -- one per env *)
(* step.truncated    : bool array      -- one per env *)
(* step.infos        : Info.t array    -- one per env *)

Vec_env.close venv
```

Key differences:
- `Vec_env.create` takes a list of already-constructed environments. All must have structurally identical spaces.
- Terminated or truncated environments are automatically reset. The terminal observation is stored in the step's info under `"final_observation"` (as a packed `Value.t`), and the terminal info under `"final_info"`.
- The step result is a record with named arrays, not a tuple.

---

## 7. Trajectory Collection

### 7.1 Fixed-step rollout

**Gymnasium + Stable Baselines3**

```python
from stable_baselines3.common.buffers import RolloutBuffer

# Manual loop or SB3 internals
obs, _ = env.reset()
for step in range(2048):
    action, log_prob, value = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(obs, action, reward, ...)
    if terminated or truncated:
        obs, _ = env.reset()
```

**Fehu** -- built-in:

<!-- $MDX skip -->
```ocaml
let trajectory =
  Collect.rollout env
    ~policy:(fun obs ->
      let action = (* select action *) in
      let log_prob = (* optional log probability *) in
      let value = (* optional value estimate *) in
      (action, Some log_prob, Some value))
    ~n_steps:2048
```

`Collect.rollout` handles resets on episode boundaries automatically and returns a `Collect.t` record:

<!-- $MDX skip -->
```ocaml
(* Collect.t fields: *)
trajectory.observations       (* 'obs array *)
trajectory.actions            (* 'act array *)
trajectory.rewards            (* float array *)
trajectory.next_observations  (* 'obs array *)
trajectory.terminated         (* bool array *)
trajectory.truncated          (* bool array *)
trajectory.infos              (* Info.t array *)
trajectory.log_probs          (* float array option *)
trajectory.values             (* float array option *)

let n = Collect.length trajectory
```

### 7.2 Complete episodes

**Gymnasium + manual**

```python
episodes = []
for _ in range(10):
    obs, _ = env.reset()
    episode = []
    done = False
    while not done:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.append((obs, action, reward, next_obs, terminated, truncated))
        obs = next_obs
        done = terminated or truncated
    episodes.append(episode)
```

**Fehu** -- built-in:

<!-- $MDX skip -->
```ocaml
let episode_list =
  Collect.episodes env
    ~policy:(fun obs -> (policy obs, None, None))
    ~n_episodes:10
    ~max_steps:1000
    ()
(* episode_list : ('obs, 'act) Collect.t list *)
```

Each element is one episode as a `Collect.t`. Concatenate them with `Collect.concat`:

<!-- $MDX skip -->
```ocaml
let all_transitions = Collect.concat episode_list
```

---

## 8. Replay Buffers

### 8.1 Standard replay buffer

**Stable Baselines3**

```python
from stable_baselines3.common.buffers import ReplayBuffer

buffer = ReplayBuffer(buffer_size=100_000, observation_space=..., action_space=...)
buffer.add(obs, next_obs, action, reward, done, infos)
batch = buffer.sample(batch_size=256)
```

**Fehu** -- built-in:

<!-- $MDX skip -->
```ocaml
let buf = Buffer.create ~capacity:100_000

let () =
  Buffer.add buf {
    Buffer.observation = obs;
    action;
    reward = 1.0;
    next_observation = next_obs;
    terminated = false;
    truncated = false;
  }

(* Uniform random sampling *)
let batch = Buffer.sample buf ~batch_size:256
(* batch : ('obs, 'act) Buffer.transition array *)

(* Structure-of-arrays form for training loops *)
let (observations, actions, rewards, next_observations, terminated, truncated) =
  Buffer.sample_arrays buf ~batch_size:256
```

### 8.2 Buffer queries

<!-- $MDX skip -->
```ocaml
let n       = Buffer.size buf       (* current number of stored transitions *)
let cap     = Buffer.capacity buf   (* maximum capacity *)
let full    = Buffer.is_full buf    (* true when size = capacity *)
let ()      = Buffer.clear buf      (* remove all transitions, keep storage *)
```

---

## 9. GAE and Returns

### 9.1 Generalized Advantage Estimation

**Stable Baselines3** (internal)

```python
# SB3 computes GAE internally in on-policy algorithms
# or manually:
import numpy as np

def compute_gae(rewards, values, dones, next_values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns
```

**Fehu** -- built-in, with correct terminated/truncated handling:

<!-- $MDX skip -->
```ocaml
let (advantages, returns) =
  Gae.compute
    ~rewards:trajectory.rewards
    ~values:(Option.get trajectory.values)
    ~terminated:trajectory.terminated
    ~truncated:trajectory.truncated
    ~next_values  (* float array: V(s_{t+1}) for each t *)
    ~gamma:0.99
    ~lambda:0.95
```

When you have values from a rollout and a final bootstrap value:

<!-- $MDX skip -->
```ocaml
let (advantages, returns) =
  Gae.compute_from_values
    ~rewards:trajectory.rewards
    ~values:(Option.get trajectory.values)
    ~terminated:trajectory.terminated
    ~truncated:trajectory.truncated
    ~last_value:0.0
    ~gamma:0.99
    ~lambda:0.95
```

`compute_from_values` builds `next_values` from `values` and `last_value` automatically: `next_values.(t) = values.(t+1)` for `t < n-1`, and `next_values.(n-1) = last_value`.

### 9.2 Monte Carlo returns

**Manual Python**

```python
def discounted_returns(rewards, dones, gamma=0.99):
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running * (1 - dones[t])
        returns[t] = running
    return returns
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let mc_returns =
  Gae.returns
    ~rewards:trajectory.rewards
    ~terminated:trajectory.terminated
    ~truncated:trajectory.truncated
    ~gamma:0.99
```

### 9.3 Normalization

<!-- $MDX skip -->
```ocaml
let normalized_advantages = Gae.normalize advantages
let normalized_custom     = Gae.normalize ~eps:1e-5 advantages
```

---

## 10. Policy Evaluation

**Gymnasium + Stable Baselines3**

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
```

**Fehu** -- built-in:

<!-- $MDX skip -->
```ocaml
let stats =
  Eval.run env
    ~policy:(fun obs -> (* deterministic action *))
    ~n_episodes:10
    ~max_steps:1000
    ()

(* stats.mean_reward  : float *)
(* stats.std_reward   : float *)
(* stats.mean_length  : float *)
(* stats.n_episodes   : int   *)
```

`Eval.run` resets the environment between episodes and collects total reward and episode length across all episodes.

---

## 11. Rendering

### 11.1 Render modes

**Gymnasium**

```python
env = gym.make("CartPole-v1", render_mode="human")
env.reset()
env.step(action)
frame = env.render()  # None for "human", np.ndarray for "rgb_array"
```

**Fehu**

<!-- $MDX skip -->
```ocaml
let env =
  Env.create
    ~render_mode:`Human
    ~render_modes:["human"; "rgb_array"]
    ~render:(fun () -> (* return 'render option *))
    (* ... *)
    ()

let frame = Env.render env  (* 'render option *)
```

Render modes are polymorphic variants: `` `Human ``, `` `Rgb_array ``, `` `Ansi ``, `` `Svg ``, `` `Custom of string ``.

### 11.2 Frame type

For `Rgb_array` environments, Fehu uses `Render.image`:

<!-- $MDX skip -->
```ocaml
(* Render.image fields: *)
(* width        : int                              *)
(* height       : int                              *)
(* pixel_format : Render.Pixel.format (Rgb|Rgba|Gray) *)
(* data         : uint8 bigarray                   *)
```

### 11.3 Recording rendered rollouts

**Gymnasium**

```python
from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, video_folder="./videos")
```

**Fehu** -- use `Render.rollout` or `Render.on_render`:

<!-- $MDX skip -->
```ocaml
(* Run a policy and feed frames to a sink *)
Render.rollout env
  ~policy:(fun obs -> (* action *))
  ~steps:500
  ~sink:(fun frame -> (* save or display frame *))
  ()

(* Or wrap the env to capture every rendered frame *)
let recording_env =
  Render.on_render
    ~sink:(fun frame -> (* process frame *))
    env
```

---

## 12. Info Dictionaries

**Gymnasium** uses plain Python dicts for info:

```python
obs, info = env.reset()
print(info.get("elapsed_steps", 0))
```

**Fehu** uses typed `Info.t` dictionaries with `Value.t` values:

<!-- $MDX skip -->
```ocaml
let info = Info.of_list [
  ("elapsed_steps", Info.int 42);
  ("success", Info.bool true);
]

let steps = Info.find "elapsed_steps" info  (* Value.t option *)
let steps = Info.find_exn "elapsed_steps" info  (* Value.t, raises on missing *)

let info' = Info.set "custom_key" (Info.float 3.14) info
let info' = Info.merge info1 info2  (* info2 wins on conflicts *)
let is_empty = Info.is_empty info
```

---

## 13. Quick Cheat Sheet

| Task                 | Gymnasium / SB3                                   | Fehu                                                                              |
| -------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------- |
| Create env           | `gym.make("CartPole-v1")`                         | `Env.create ~observation_space ~action_space ~reset ~step ()`                     |
| Reset                | `obs, info = env.reset(seed=42)`                  | `let (obs, info) = Env.reset env ()`                                              |
| Step                 | `obs, r, term, trunc, info = env.step(a)`         | `let s = Env.step env a` (record fields)                                          |
| Close                | `env.close()`                                     | `Env.close env`                                                                   |
| Discrete space       | `gym.spaces.Discrete(5)`                          | `Space.Discrete.create 5`                                                         |
| Box space            | `gym.spaces.Box(low, high)`                       | `Space.Box.create ~low ~high`                                                     |
| Sample from space    | `space.sample()`                                  | `Space.sample space`                                                              |
| Contains check       | `space.contains(x)`                               | `Space.contains space x`                                                          |
| Observation wrapper  | `class W(gym.ObservationWrapper)`                 | `Env.map_observation ~observation_space ~f env`                                   |
| Action wrapper       | `class W(gym.ActionWrapper)`                      | `Env.map_action ~action_space ~f env`                                             |
| Reward wrapper       | `class W(gym.RewardWrapper)`                      | `Env.map_reward ~f env`                                                           |
| Clip actions         | `ClipAction(env)`                                 | `Env.clip_action env`                                                             |
| Time limit           | `TimeLimit(env, max_episode_steps=N)`             | `Env.time_limit ~max_episode_steps:N env`                                         |
| Vectorize            | `gym.vector.SyncVectorEnv([...])`                 | `Vec_env.create [env1; env2; ...]`                                                |
| Rollout N steps      | Manual loop / SB3 internal                        | `Collect.rollout env ~policy ~n_steps`                                            |
| Collect N episodes   | Manual loop                                       | `Collect.episodes env ~policy ~n_episodes ()`                                     |
| Replay buffer        | `ReplayBuffer(buffer_size=N, ...)`                | `Buffer.create ~capacity:N`                                                       |
| Add to buffer        | `buffer.add(obs, next_obs, ...)`                  | `Buffer.add buf transition`                                                       |
| Sample from buffer   | `buffer.sample(batch_size=B)`                     | `Buffer.sample buf ~batch_size:B`                                                 |
| GAE                  | SB3 internal / manual                             | `Gae.compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma ~lambda` |
| Discounted returns   | Manual loop                                       | `Gae.returns ~rewards ~terminated ~truncated ~gamma`                              |
| Normalize advantages | `(adv - mean) / std`                              | `Gae.normalize advantages`                                                        |
| Evaluate policy      | `evaluate_policy(model, env, n_eval_episodes=10)` | `Eval.run env ~policy ~n_episodes:10 ()`                                          |
| Render               | `env.render()`                                    | `Env.render env`                                                                  |
| Record frames        | `RecordVideo(env, ...)`                           | `Render.on_render ~sink env`                                                      |
| Seed RNG             | `env.reset(seed=42)`                              | `Nx.Rng.run ~seed:42 (fun () -> ...)`                                             |
