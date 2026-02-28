# Getting Started

This guide covers the basics: creating environments, running the step loop,
understanding spaces, and using the built-in environments.

## Installation

<!-- $MDX skip -->
```bash
opam install fehu
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build fehu
```

## Creating an Environment

Environments are created via factory functions in `Fehu_envs`. Randomness is
provided by the implicit RNG scope from `Nx.Rng.run`:

```ocaml
open Fehu

let () = Nx.Rng.run ~seed:42 @@ fun () ->
  let env = Fehu_envs.Cartpole.make () in
  ignore env
```

The seed controls all randomness in the scope. Use the same seed to get
the same episode sequence.

## The Step Loop

An environment follows a strict lifecycle: `reset` must be called before the
first `step`, and again after any terminal step (terminated or truncated).

```ocaml
open Fehu

let () = Nx.Rng.run ~seed:42 @@ fun () ->
  let env = Fehu_envs.Cartpole.make () in

  (* Reset returns the initial observation and info *)
  let _obs, _info = Env.reset env () in

  (* Step returns observation, reward, terminated, truncated, info *)
  let s = Env.step env (Space.Discrete.of_int 0) in
  Printf.printf "reward: %.1f, terminated: %b, truncated: %b\n"
    s.reward s.terminated s.truncated
```

A complete episode loop:

```ocaml
open Fehu

let run_episode env =
  let _obs, _info = Env.reset env () in
  let done_ = ref false in
  let total_reward = ref 0.0 in
  while not !done_ do
    let act = Space.sample (Env.action_space env) in
    let s = Env.step env act in
    total_reward := !total_reward +. s.reward;
    done_ := s.terminated || s.truncated
  done;
  !total_reward

let () = Nx.Rng.run ~seed:42 @@ fun () ->
  let env = Fehu_envs.Cartpole.make () in
  let _reward = run_episode env in ()
```

## Spaces

Spaces define the valid observations and actions for an environment. They
provide sampling, validation, and serialization.

### Discrete

Integer choices. Used for environments with a finite number of actions (e.g.,
left/right).

```ocaml
open Fehu

let space = Space.Discrete.create 4  (* actions 0, 1, 2, 3 *)
let _n = Space.Discrete.n space      (* 4 *)

(* Sample a random action (requires an Nx.Rng scope) *)
let _act = Nx.Rng.run ~seed:0 @@ fun () ->
  Space.sample space

(* Convert between int and discrete element *)
let act = Space.Discrete.of_int 2
let _i = Space.Discrete.to_int act   (* 2 *)

(* Check membership *)
let _valid = Space.contains space act (* true *)
```

### Box

Continuous vectors with per-dimension bounds. Used for continuous observations
(e.g., position, velocity) and continuous actions.

```ocaml
open Fehu

let space = Space.Box.create
  ~low:[| -1.0; -2.0 |]
  ~high:[| 1.0; 2.0 |]

let _low, _high = Space.Box.bounds space
let _obs = Nx.Rng.run ~seed:0 @@ fun () -> Space.sample space
```

### Other Space Types

- **Multi_binary**: binary vectors of fixed length (multi-label scenarios)
- **Multi_discrete**: multiple discrete axes with independent cardinalities
- **Tuple**: fixed-length heterogeneous sequences
- **Dict**: named fields with different space types
- **Sequence**: variable-length homogeneous sequences
- **Text**: character strings from a fixed alphabet

All spaces support `contains`, `sample`, `pack`/`unpack` (to/from the
universal `Value.t` type), and `boundary_values`.

## Available Environments

### CartPole

Classic cart-pole balancing. Push a cart left or right to keep a pole upright.
Reward is +1.0 per step. Terminates when the pole exceeds +/-12 degrees or the
cart leaves +/-2.4. Truncates at 500 steps.

- **Observation**: Box [4] -- x, x_dot, theta, theta_dot
- **Actions**: Discrete 2 -- 0 = push left, 1 = push right

```ocaml
let _env = Nx.Rng.run ~seed:42 @@ fun () -> Fehu_envs.Cartpole.make ()
```

### MountainCar

A car in a valley must build momentum to climb a hill. Reward is -1.0 per
step. Terminates when position >= 0.5 with non-negative velocity. Truncates at
200 steps.

- **Observation**: Box [2] -- position, velocity
- **Actions**: Discrete 3 -- 0 = push left, 1 = coast, 2 = push right

```ocaml
let _env = Nx.Rng.run ~seed:42 @@ fun () -> Fehu_envs.Mountain_car.make ()
```

### GridWorld

5x5 grid navigation with an obstacle. Agent starts at (0,0), goal at (4,4),
obstacle at (2,2). Reward is +10.0 at goal, -1.0 otherwise. Truncates at 200
steps.

- **Observation**: Multi_discrete [5; 5] -- (row, col)
- **Actions**: Discrete 4 -- 0 = up, 1 = down, 2 = left, 3 = right

```ocaml
let _env = Nx.Rng.run ~seed:42 @@ fun () -> Fehu_envs.Grid_world.make ()
```

### RandomWalk

One-dimensional random walk on [-10, 10]. Reward is -|position|. Terminates at
boundaries or after 200 steps.

- **Observation**: Box [1] in [-10.0, 10.0]
- **Actions**: Discrete 2 -- 0 = left, 1 = right

```ocaml
let _env = Nx.Rng.run ~seed:42 @@ fun () -> Fehu_envs.Random_walk.make ()
```

## Render Modes

Environments can optionally render their state. Pass `~render_mode` when
creating the environment:

```ocaml
open Fehu

let () = Nx.Rng.run ~seed:42 @@ fun () ->
  let env = Fehu_envs.Cartpole.make
    ~render_mode:`Ansi () in

  let _obs, _info = Env.reset env () in
  let _s = Env.step env (Space.Discrete.of_int 0) in

  (* Render after reset or step *)
  match Env.render env with
  | Some text -> print_endline text
  | None -> ()
```

Supported render modes vary by environment: `Ansi` for text output,
`Rgb_array` for pixel frames, `Human` for interactive display.

## Next Steps

- [Environments and Wrappers](../02-environments/) -- custom environments, wrappers, rendering, vectorized environments
- [Collection and Evaluation](../03-collection-and-evaluation/) -- trajectory collection, replay buffers, GAE, evaluation
