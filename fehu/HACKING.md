# Fehu Developer Guide

## Architecture

Fehu is a reinforcement learning library inspired by OpenAI Gymnasium and Stable-Baselines3, providing environment interfaces, vectorization, training utilities, and trajectory management.

### Core Components

- **[lib/env.ml](lib/env.ml)**: Core environment interface and lifecycle
- **[lib/space.ml](lib/space.ml)**: Observation/action space definitions
- **[lib/wrapper.ml](lib/wrapper.ml)**: Environment wrappers for behavior modification
- **[lib/vector_env.ml](lib/vector_env.ml)**: Vectorized parallel environments
- **[lib/training.ml](lib/training.ml)**: RL algorithms (GAE, policy gradients, etc.)
- **[lib/trajectory.ml](lib/trajectory.ml)**: Experience collection and replay
- **[lib/buffer.ml](lib/buffer.ml)**: Trajectory storage and sampling
- **[lib/environments/](lib/environments/)**: Built-in environments (CartPole, etc.)

### Key Design Principles

1. **Functional core**: Environments as values with explicit state
2. **Type-safe spaces**: GADTs for observation/action validation
3. **Composable wrappers**: Chain wrappers via function composition
4. **Reproducible randomness**: Explicit RNG threading via Rune
5. **Tensor-native**: Observations/actions as Rune tensors

## Environment Model

### The Environment Type

```ocaml
type ('obs, 'act, 'render) t = {
  observation_space : 'obs Space.t;
  action_space : 'act Space.t;
  mutable rng : Rune.Rng.key;
  mutable needs_reset : bool;
  reset_impl : t -> ?options:Info.t -> unit -> 'obs * Info.t;
  step_impl : t -> 'act -> ('obs, 'act, 'render) transition;
  render_impl : t -> 'render option;
  close_impl : t -> unit;
}
```

**Design rationale:**
- Mutable RNG: Deterministic stepping without threading RNG everywhere
- `needs_reset` flag: Enforce reset-before-step invariant
- Implementation functions: Delegate to user-defined logic
- Type parameters: `'obs`, `'act`, `'render` for type safety

### Lifecycle States

```
┌─────────┐
│ Created │
│needs_reset=true│
└─────────┘
     │ reset()
     ▼
┌─────────┐
│ Running │ ◄─┐
│needs_reset=false│  │ step()
└─────────┘  │
     │ terminated/truncated
     ▼       │
┌─────────┐  │
│Terminal │ ─┘
│needs_reset=true│
└─────────┘
     │ close()
     ▼
┌─────────┐
│ Closed  │
└─────────┘
```

**Invariants:**
- Must reset before first step
- Must reset after terminal state
- Cannot operate on closed environment

### Transition Type

```ocaml
type ('obs, 'act, 'render) transition = {
  observation : 'obs;
  reward : float;
  terminated : bool;  (* Episode ended successfully *)
  truncated : bool;   (* Episode cut off (time limit) *)
  info : Info.t;      (* Auxiliary information *)
}
```

**Terminated vs. Truncated:**
- `terminated`: Natural episode end (goal reached, failure)
- `truncated`: Artificial cutoff (max steps, time limit)
- Important for bootstrapping: Bootstrap on truncated, not terminated

## Spaces

### Space Types

```ocaml
type 'a t =
  | Discrete : int -> int t
  | Box : { low : float array; high : float array; shape : int array } -> (float, _) Rune.t t
  | Tuple : 'a t * 'b t -> ('a * 'b) t
  | Dict : (string * 'a t) list -> (string * 'a) list t
```

**GADT for type safety:**
- `Discrete n` constrains values to `0..n-1`
- `Box` defines continuous vector bounds
- `Tuple`/`Dict` compose spaces

### Space Operations

```ocaml
(* Check if value is valid *)
val contains : 'a t -> 'a -> bool

(* Sample random value *)
val sample : 'a t -> rng:Rune.Rng.key -> 'a

(* Get shape *)
val shape : 'a t -> int array
```

**Validation:**

```ocaml
let contains space value =
  match space, value with
  | Discrete n, v -> v >= 0 && v < n
  | Box {low; high; _}, tensor ->
      (* Check bounds element-wise *)
      all_elements tensor (fun i x ->
        x >= low.(i) && x <= high.(i))
  | Tuple (s1, s2), (v1, v2) ->
      contains s1 v1 && contains s2 v2
  | ...
```

## Development Workflow

### Building and Testing

```bash
# Build fehu
dune build fehu/

# Run tests
dune build fehu/test/test_fehu.exe && _build/default/fehu/test/test_fehu.exe

# Run environment tests
_build/default/fehu/test/test_fehu.exe test "CartPole"
```

### Testing Environments

**Basic functionality:**

```ocaml
let test_cartpole () =
  let rng = Rune.Rng.key 42 in
  let env = Fehu_envs.Cartpole.make ~rng () in

  (* Test reset *)
  let obs, info = Env.reset env () in
  Alcotest.(check bool) "obs in space"
    true (Space.contains env.observation_space obs);

  (* Test step *)
  let action = 0 in  (* Left *)
  let transition = Env.step env action in
  Alcotest.(check bool) "reward is float"
    true (Float.is_finite transition.reward)
```

**Episode rollout:**

```ocaml
let test_full_episode () =
  let env = make_env () in
  let rec rollout total_reward =
    let action = sample_action env in
    let t = Env.step env action in
    let total = total_reward +. t.reward in
    if t.terminated || t.truncated then total
    else rollout total
  in
  let obs, _ = Env.reset env () in
  let total = rollout 0. in
  assert (total > neg_infinity)  (* Episode completed *)
```

### Testing Reproducibility

```ocaml
let test_deterministic () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 42 in

  let rollout rng =
    let env = make_env ~rng () in
    let _, _ = Env.reset env () in
    let t = Env.step env (sample_action env) in
    t.reward
  in

  let r1 = rollout rng1 in
  let r2 = rollout rng2 in
  Alcotest.(check (float 1e-6)) "same reward" r1 r2
```

## Creating Environments

### Minimal Environment

```ocaml
let simple_env ~rng () =
  Env.create
    ~rng
    ~observation_space:(Space.Box.create ~low:[|-1.|] ~high:[|1.|])
    ~action_space:(Space.Discrete.create 2)
    ~reset:(fun env ?options () ->
      (* Initialize state *)
      let obs = Rune.scalar Float32 0. in
      (obs, Info.empty))
    ~step:(fun env action ->
      (* Update state, compute reward *)
      let obs = Rune.rand Float32 [|1|] in
      let reward = if action = 1 then 1. else -1. in
      Env.transition ~observation:obs ~reward ())
    ()
```

### Stateful Environment

Track state in closure:

```ocaml
let stateful_env ~rng () =
  let state = ref 0. in

  Env.create
    ~rng
    ~observation_space:(Space.Box.create ~low:[|-10.|] ~high:[|10.|])
    ~action_space:(Space.Discrete.create 3)
    ~reset:(fun env ?options () ->
      state := 0.;
      let obs = Rune.scalar Float32 !state in
      (obs, Info.empty))
    ~step:(fun env action ->
      (* Update internal state *)
      state := !state +. (float action -. 1.);
      let obs = Rune.scalar Float32 !state in
      let reward = -. abs_float !state in  (* Reward for staying near 0 *)
      let terminated = abs_float !state > 5. in
      Env.transition ~observation:obs ~reward ~terminated ())
    ()
```

### With Rendering

```ocaml
let visual_env ~rng () =
  Env.create
    ~rng
    ~observation_space:...
    ~action_space:...
    ~reset:...
    ~step:...
    ~render:(fun env ->
      (* Return visualization data *)
      Some (render_state_to_image env))
    ()
```

## Wrappers

### Wrapper Pattern

Wrappers modify environment behavior:

```ocaml
let time_limit ~max_episode_steps base_env =
  let step_count = ref 0 in

  let reset_wrapper env ?options () =
    step_count := 0;
    base_env.reset_impl base_env ?options ()
  in

  let step_wrapper env action =
    incr step_count;
    let t = base_env.step_impl base_env action in
    let truncated = !step_count >= max_episode_steps in
    {t with truncated = t.truncated || truncated}
  in

  {base_env with
   reset_impl = reset_wrapper;
   step_impl = step_wrapper}
```

### Common Wrappers

**Observation normalization:**

```ocaml
let normalize_observation ~mean ~std env =
  let step_wrapper env action =
    let t = env.step_impl env action in
    let normalized_obs = Rune.div (Rune.sub t.observation mean) std in
    {t with observation = normalized_obs}
  in
  {env with step_impl = step_wrapper}
```

**Reward clipping:**

```ocaml
let clip_reward ~min ~max env =
  let step_wrapper env action =
    let t = env.step_impl env action in
    let clipped = Float.max min (Float.min max t.reward) in
    {t with reward = clipped}
  in
  {env with step_impl = step_wrapper}
```

**Frame stacking (for temporal context):**

```ocaml
let frame_stack ~num_frames env =
  let frames = Queue.create () in

  let reset_wrapper env ?options () =
    Queue.clear frames;
    let obs, info = env.reset_impl env ?options () in
    (* Fill queue with initial observation *)
    for _i = 1 to num_frames do Queue.add obs frames done;
    (stack_frames frames, info)
  in

  let step_wrapper env action =
    let t = env.step_impl env action in
    Queue.add t.observation frames;
    if Queue.length frames > num_frames then ignore (Queue.take frames);
    {t with observation = stack_frames frames}
  in

  {env with
   reset_impl = reset_wrapper;
   step_impl = step_wrapper}
```

## Vectorization

### Vector Environment

Run multiple environments in parallel:

```ocaml
type ('obs, 'act, 'render) t = {
  envs : ('obs, 'act, 'render) Env.t array;
  num_envs : int;
}

let step vec_env actions =
  (* Step all environments *)
  let transitions = Array.map2 Env.step vec_env.envs actions in

  (* Stack observations into batch *)
  let obs_batch = stack_observations transitions in
  let rewards = Array.map (fun t -> t.reward) transitions in
  let dones = Array.map (fun t -> t.terminated || t.truncated) transitions in

  (* Auto-reset terminated environments *)
  Array.iteri (fun i done_ ->
    if done_ then
      let obs, _ = Env.reset vec_env.envs.(i) () in
      obs_batch.(i) <- obs
  ) dones;

  (obs_batch, rewards, dones)
```

**Benefits:**
- Batch policy inference
- Parallel data collection
- Amortized environment overhead

## Training Utilities

### Generalized Advantage Estimation (GAE)

Compute advantages for policy gradients:

```ocaml
let compute_gae ~gamma ~lambda rewards values dones =
  let advantages = Array.make (Array.length rewards) 0. in
  let rec loop t gae_t =
    if t < 0 then ()
    else
      let delta = rewards.(t) +. gamma *. values.(t+1) *. (1. -. float dones.(t)) -. values.(t) in
      let gae_t = delta +. gamma *. lambda *. (1. -. float dones.(t)) *. gae_t in
      advantages.(t) <- gae_t;
      loop (t-1) gae_t
  in
  loop (Array.length rewards - 1) 0.;
  advantages
```

**Why GAE?**
- Bias-variance tradeoff via `lambda`
- Reduces gradient variance
- Essential for PPO, A2C, etc.

### Policy Gradient Loss

```ocaml
let policy_gradient_loss ~log_probs ~advantages =
  (* -E[log π(a|s) * A(s,a)] *)
  Array.map2 (fun lp adv -> -. lp *. adv) log_probs advantages
  |> Array.fold_left (+.) 0.
```

### Value Loss

```ocaml
let value_loss ~values ~returns =
  (* MSE between predicted values and actual returns *)
  Array.map2 (fun v r -> (v -. r) ** 2.) values returns
  |> Array.fold_left (+.) 0.
  |> fun total -> total /. float (Array.length values)
```

## Common Pitfalls

### Reset After Terminal

Must reset after episode ends:

```ocaml
(* Wrong: step after terminal *)
let obs, _ = Env.reset env () in
let t = Env.step env 0 in
if t.terminated then
  let t2 = Env.step env 1 in  (* Error: needs_reset = true *)
  ...

(* Correct: reset first *)
if t.terminated then
  let obs, _ = Env.reset env () in
  let t2 = Env.step env 1 in  (* OK *)
  ...
```

### RNG Threading

Don't share RNG across environments:

```ocaml
(* Wrong: shared RNG *)
let rng = Rune.Rng.key 42 in
let env1 = make_env ~rng () in
let env2 = make_env ~rng () in  (* Same RNG! *)

(* Correct: split RNG *)
let rng = Rune.Rng.key 42 in
let rngs = Rune.Rng.split rng ~num:2 in
let env1 = make_env ~rng:rngs.(0) () in
let env2 = make_env ~rng:rngs.(1) ()
```

### Bootstrap on Truncation

Don't bootstrap on termination:

```ocaml
(* Wrong: bootstrap on terminal *)
let target = if done then 0. else reward +. gamma *. next_value

(* Correct: check truncated vs terminated *)
let target =
  if terminated then reward
  else reward +. gamma *. next_value  (* Bootstrap on truncation *)
```

### Space Validation

Always validate actions:

```ocaml
(* Wrong: skip validation *)
let step env action =
  (* Direct step without check *)
  env.step_impl env action

(* Correct: validate in step_impl *)
if not (Space.contains env.action_space action) then
  raise (Invalid_action "Action outside space")
```

## Performance

- **Vectorize**: Use vector environments for parallel collection
- **Batch inference**: Stack observations for policy evaluation
- **Trajectory buffers**: Reuse buffers instead of allocating
- **JIT policies**: Use Rune JIT for policy networks

## Code Style

- **Labeled arguments**: `~rng`, `~gamma`, `~max_episode_steps`
- **Options for info**: Use `Info.t` for auxiliary data
- **Explicit RNG**: Thread RNG explicitly, don't use global state
- **Errors**: Raise `Errors.RL_error` variants

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [rune/HACKING.md](../rune/HACKING.md): Rune tensors and AD
- Gymnasium documentation for environment conventions
- Stable-Baselines3 for algorithm implementations
