# Environments and Wrappers

This guide covers creating custom environments, composing wrappers, rendering,
and running vectorized environments.

## The Env.t Type

An environment `('obs, 'act, 'render) Env.t` is parameterized by its
observation type, action type, and render type. The type system ensures that
policies, wrappers, and collection utilities all agree on these types.

The lifecycle is strict:

1. Call `Env.reset` to get the initial observation
2. Call `Env.step` with an action to advance one timestep
3. When `terminated` or `truncated` is true, call `Env.reset` again
4. Call `Env.close` when done (optional, releases resources)

Calling `step` before `reset`, or after a terminal step without resetting,
raises `Invalid_argument`.

## Creating Custom Environments

Use `Env.create` to build an environment from `reset` and `step` functions.
Both receive the environment handle as their first argument, which provides
access to RNG utilities.

<!-- $MDX skip -->
```ocaml
open Fehu

(* A simple counting environment: agent must choose action 1 *)
let make_counter ~rng () =
  let count = ref 0 in
  Env.create
    ~id:"Counter-v0"
    ~rng
    ~observation_space:(Space.Discrete.create 100)
    ~action_space:(Space.Discrete.create 2)
    ~reset:(fun _env ?options:_ () ->
      count := 0;
      Space.Discrete.of_int 0, Info.empty)
    ~step:(fun _env action ->
      let a = Space.Discrete.to_int action in
      if a = 1 then incr count else count := 0;
      let obs = Space.Discrete.of_int !count in
      let terminated = !count >= 10 in
      Env.step_result ~observation:obs
        ~reward:(if a = 1 then 1.0 else -1.0)
        ~terminated ())
    ()
```

### RNG Management

Environments should use `Env.take_rng` to get a fresh RNG key for each
random operation. This splits the environment's internal key, keeping one half
and returning the other:

<!-- $MDX skip -->
```ocaml
let make_noisy_env ~rng () =
  Env.create ~rng
    ~observation_space:(Space.Box.create
      ~low:[| 0.0 |] ~high:[| 1.0 |])
    ~action_space:(Space.Discrete.create 2)
    ~reset:(fun env ?options:_ () ->
      let rng = Env.take_rng env in
      let obs, _rng =
        Space.sample (Env.observation_space env) ~rng in
      obs, Info.empty)
    ~step:(fun env _action ->
      let rng = Env.take_rng env in
      let obs, _rng =
        Space.sample (Env.observation_space env) ~rng in
      Env.step_result ~observation:obs
        ~reward:1.0 ())
    ()
```

For multiple random operations in a single step, use `Env.split_rng`:

<!-- $MDX skip -->
```ocaml
let keys = Env.split_rng env ~n:3
(* keys.(0), keys.(1), keys.(2) are independent *)
```

## Wrappers

Wrappers transform an environment's observations, actions, or rewards without
modifying the inner environment. They compose: wrap a wrapper to stack
transformations.

### map_observation

Transform observations from reset and step:

<!-- $MDX skip -->
```ocaml
open Fehu

(* Normalize observations to [0, 1] *)
let env = Env.map_observation
  ~observation_space:(Space.Box.create
    ~low:[| 0.0; 0.0; 0.0; 0.0 |]
    ~high:[| 1.0; 1.0; 1.0; 1.0 |])
  ~f:(fun obs info ->
    (* obs is a float32 tensor, transform it *)
    let normalized = normalize_fn obs in
    normalized, info)
  env
```

The function `f` receives both the observation and the info dictionary,
returning both. This allows wrappers to pass metadata through info.

### map_action

Transform actions before they reach the inner environment:

<!-- $MDX skip -->
```ocaml
(* Remap discrete actions *)
let env = Env.map_action
  ~action_space:(Space.Discrete.create 3)
  ~f:(fun act ->
    (* Map from 3-action to 2-action space *)
    let i = Space.Discrete.to_int act in
    Space.Discrete.of_int (if i >= 2 then 1 else i))
  env
```

### map_reward

Transform rewards after each step:

<!-- $MDX skip -->
```ocaml
(* Scale rewards *)
let env = Env.map_reward
  ~f:(fun ~reward ~info -> reward *. 0.01, info)
  env
```

### clip_action

Clamp continuous actions to the action space bounds. The wrapper relaxes the
action space to accept any float values, then clips before forwarding:

<!-- $MDX skip -->
```ocaml
(* Works with Box action spaces *)
let env = Env.clip_action env
```

### clip_observation

Clamp observations to specified bounds:

<!-- $MDX skip -->
```ocaml
let env = Env.clip_observation
  ~low:[| -1.0; -1.0 |]
  ~high:[| 1.0; 1.0 |]
  env
```

### time_limit

Enforce a maximum episode length. When the limit is reached, the step's
`truncated` flag is set to true:

<!-- $MDX skip -->
```ocaml
let env = Env.time_limit ~max_episode_steps:200 env
```

### Custom Wrappers with Env.wrap

For transformations that need full control over reset and step, use `Env.wrap`.
The wrapper shares the inner environment's lifecycle (RNG, closed flag, reset
flag):

<!-- $MDX skip -->
```ocaml
open Fehu

(* A wrapper that tracks episode reward *)
let with_episode_reward env =
  let episode_reward = ref 0.0 in
  Env.wrap
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun inner ?options () ->
      episode_reward := 0.0;
      Env.reset inner ?options ())
    ~step:(fun inner action ->
      let s = Env.step inner action in
      episode_reward := !episode_reward +. s.reward;
      let info =
        if s.terminated || s.truncated then
          Info.set "episode_reward"
            (Info.float !episode_reward) s.info
        else s.info
      in
      { s with info })
    env
```

## Rendering

Environments support optional rendering via render modes. Pass
`~render_mode` at creation time:

<!-- $MDX skip -->
```ocaml
let env = Fehu_envs.Grid_world.make
  ~render_mode:`Ansi ~rng ()

let _obs, _info = Env.reset env ()
match Env.render env with
| Some (Text s) -> print_endline s
| _ -> ()
```

### Render Rollout

`Render.rollout` runs a policy and feeds rendered frames to a sink function:

<!-- $MDX skip -->
```ocaml
open Fehu

(* Collect rendered frames from a policy rollout *)
let frames = ref [] in
Render.rollout env
  ~policy:(fun _obs ->
    let act, _ = Space.sample (Env.action_space env)
      ~rng:(Env.take_rng env) in act)
  ~steps:100
  ~sink:(fun img -> frames := img :: !frames)
  ()
```

### Recording with on_render

`Render.on_render` wraps an environment so that every frame after reset and
step is passed to a sink:

<!-- $MDX skip -->
```ocaml
let env = Render.on_render
  ~sink:(fun img -> save_frame img)
  env
```

## Vectorized Environments

`Vec_env` runs multiple environment instances with batched inputs and outputs.
Terminated or truncated episodes are automatically reset.

<!-- $MDX skip -->
```ocaml
open Fehu

(* Create 4 parallel environments *)
let rng = Rune.Rng.key 42
let keys = Rune.Rng.split rng ~n:4
let envs = Array.to_list (Array.map
  (fun rng -> Fehu_envs.Cartpole.make ~rng ())
  keys)
let vec = Vec_env.create envs

let n = Vec_env.num_envs vec  (* 4 *)

(* Reset all environments *)
let observations, _infos = Vec_env.reset vec ()

(* Step all environments with an array of actions *)
let actions = Array.init n (fun _ -> Space.Discrete.of_int 0) in
let s = Vec_env.step vec actions

(* s.observations, s.rewards, s.terminated, s.truncated *)

(* Clean up *)
Vec_env.close vec
```

All environments must have structurally identical observation and action spaces
(checked via `Space.equal_spec`). On terminal steps, the original terminal
observation is stored in the step info under `"final_observation"` as a packed
`Value.t`, and the terminal info under `"final_info"`.

## Next Steps

- [Getting Started](../01-getting-started/) -- installation, environments, spaces, step loop
- [Collection and Evaluation](../03-collection-and-evaluation/) -- trajectory collection, replay buffers, GAE, evaluation
