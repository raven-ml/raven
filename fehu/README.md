# Fehu

Reinforcement learning environment toolkit for OCaml, built on [Rune](../rune/)

Fehu provides type-safe environments, composable wrappers, trajectory
collection, replay buffers, GAE computation, policy evaluation, and
vectorized environments. It follows the Gymnasium interface pattern:
environments expose `reset` and `step` with typed observation and action
spaces.

## Quick Start

Create an environment, run a random policy, and evaluate:

```ocaml
open Fehu

let () =
  let rng = Rune.Rng.key 42 in
  let env = Fehu_envs.Cartpole.make ~rng () in

  (* Run one episode *)
  let _obs, _info = Env.reset env () in
  let done_ = ref false in
  let total_reward = ref 0.0 in
  while not !done_ do
    let act, _ = Space.sample (Env.action_space env)
      ~rng:(Env.take_rng env) in
    let s = Env.step env act in
    total_reward := !total_reward +. s.reward;
    done_ := s.terminated || s.truncated
  done;
  Printf.printf "Episode reward: %.0f\n" !total_reward;

  (* Evaluate over 10 episodes *)
  let stats = Eval.run env
    ~policy:(fun _obs ->
      let act, _ = Space.sample (Env.action_space env)
        ~rng:(Env.take_rng env) in act)
    ~n_episodes:10 ()
  in
  Printf.printf "Mean reward: %.1f (std: %.1f)\n"
    stats.mean_reward stats.std_reward
```

## Features

- **Environments**: typed `('obs, 'act, 'render) Env.t` with lifecycle enforcement (reset before step, auto-guard on terminal states)
- **Spaces**: Discrete, Box, Multi_binary, Multi_discrete, Tuple, Dict, Sequence, Text with sampling, validation, and serialization
- **Wrappers**: `map_observation`, `map_action`, `map_reward`, `clip_action`, `clip_observation`, `time_limit`, and custom wrappers via `Env.wrap`
- **Trajectory collection**: `Collect.rollout` and `Collect.episodes` in structure-of-arrays form with automatic episode resets
- **Replay buffers**: fixed-capacity circular buffer with uniform random sampling (`Buffer.sample`, `Buffer.sample_arrays`)
- **GAE**: generalized advantage estimation with proper terminated/truncated handling (`Gae.compute`, `Gae.returns`)
- **Evaluation**: `Eval.run` computes mean/std reward and episode length over multiple episodes
- **Vectorized environments**: `Vec_env.create` runs multiple environments with batched step and auto-reset
- **Rendering**: `Render.image` and `Render.rollout` for frame capture, `Env.on_render` for recording
- **Built-in environments**: CartPole-v1, MountainCar-v0, GridWorld, RandomWalk

## Libraries

| Library | opam package | Description |
|---------|-------------|-------------|
| `fehu` | `fehu` | Core: environments, spaces, wrappers, collection, buffers, GAE, evaluation |
| `fehu-envs` | `fehu.envs` | Built-in environments (CartPole, MountainCar, GridWorld, RandomWalk) |

## Built-in Environments

| Environment | Observation | Actions | Reward | Termination |
|-------------|------------|---------|--------|-------------|
| CartPole | Box [4] (x, v, θ, ω) | Discrete 2 | +1.0 per step | Pole > ±12° or cart > ±2.4, truncated at 500 |
| MountainCar | Box [2] (position, velocity) | Discrete 3 | −1.0 per step | Position ≥ 0.5 with v ≥ 0, truncated at 200 |
| GridWorld | Multi_discrete [5; 5] | Discrete 4 | +10 at goal, −1 otherwise | Reach (4,4), truncated at 200 |
| RandomWalk | Box [1] | Discrete 2 | −|position| | None, truncated at 200 |

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
