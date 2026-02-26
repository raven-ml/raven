# Fehu

Fehu is a reinforcement learning environment toolkit for OCaml. It provides
type-safe environments, composable wrappers, trajectory collection, replay
buffers, GAE computation, policy evaluation, and vectorized environments.

Fehu follows the Gymnasium interface pattern: environments expose `reset` and
`step` with typed observation and action spaces. Wrappers compose freely.
Collection and evaluation utilities handle the plumbing between environments
and training loops.

## Features

- **Type-safe environments**: observation and action spaces are encoded in the type system
- **Rich space types**: Discrete, Box, Multi_binary, Multi_discrete, Tuple, Dict, Sequence, Text
- **Composable wrappers**: map_observation, map_action, map_reward, clip_action, clip_observation, time_limit
- **Trajectory collection**: rollout and episode collection in structure-of-arrays form
- **Replay buffers**: fixed-capacity circular buffer with uniform random sampling
- **GAE**: generalized advantage estimation with proper terminated/truncated handling
- **Policy evaluation**: run a policy over episodes and get mean/std reward statistics
- **Vectorized environments**: run multiple environments with batched step and auto-reset
- **Built-in environments**: CartPole, MountainCar, GridWorld, RandomWalk

## Quick Start

Create an environment, run a random agent, and evaluate:

<!-- $MDX skip -->
```ocaml
open Fehu

let rng = Rune.Rng.key 42
let env = Fehu_envs.Cartpole.make ~rng ()

(* Run one episode *)
let obs, _info = Env.reset env ()
let done_ = ref false
let total_reward = ref 0.0
while not !done_ do
  let act, _ = Space.sample (Env.action_space env)
    ~rng:(Env.take_rng env) in
  let s = Env.step env act in
  total_reward := !total_reward +. s.reward;
  done_ := s.terminated || s.truncated
done

(* Evaluate over 10 episodes *)
let stats = Eval.run env
  ~policy:(fun _obs ->
    let act, _ = Space.sample (Env.action_space env)
      ~rng:(Env.take_rng env) in act)
  ~n_episodes:10 ()
```

## Next Steps

- [Getting Started](01-getting-started/) -- installation, environments, spaces, step loop
- [Environments and Wrappers](02-environments/) -- custom environments, wrappers, rendering, vectorized environments
- [Collection and Evaluation](03-collection-and-evaluation/) -- trajectory collection, replay buffers, GAE, evaluation
