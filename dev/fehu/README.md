# Fehu ᚠ - Reinforcement Learning Framework for OCaml

Fehu is a reinforcement learning framework built on the Raven ecosystem, providing composable RL-specific components that work seamlessly with Kaun for neural networks.

## Features

- **Gymnasium-compatible API**: Familiar interface for environments and spaces
- **Classic Environments**: CartPole, MountainCar, and Pendulum implementations
- **Replay Buffers**: Efficient experience replay for off-policy algorithms
- **Training Utilities**: GAE computation, return calculation, and normalization
- **Composable Design**: Use Kaun for neural networks, Fehu for RL components

## Installation

Fehu is part of the Raven ecosystem. Build it with dune:

```bash
dune build dev/fehu
```

## Quick Start

### Random Agent Example

```ocaml
open Fehu
module Rng = Rune.Rng

let () =
  (* Create environment *)
  let env = Envs.cartpole () in
  let rng = Rng.key 42 in
  
  (* Run episode *)
  let obs, _ = env.reset () in
  let finished = ref false in
  
  while not !finished do
    (* Sample random action *)
    let action = Space.sample ~rng env.action_space in
    
    (* Step environment *)
    let next_obs, reward, terminated, truncated, _ = env.step action in
    finished := terminated || truncated
  done
```

### DQN with Kaun Integration

```ocaml
open Fehu

(* Create Q-network using Kaun *)
let create_q_network obs_dim n_actions =
  Kaun.Layer.sequential [
    Kaun.Layer.linear ~in_features:obs_dim ~out_features:128 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:128 ~out_features:n_actions ();
  ]

(* Training loop *)
let train_dqn env =
  let q_network = create_q_network 4 2 in
  let buffer = Buffer.create ~capacity:10000 in
  let optimizer = Kaun.Optimizer.adam ~lr:0.001 () in
  
  (* Initialize parameters *)
  let rng = Rune.Rng.key 42 in
  let dummy_input = Rune.zeros Rune.float32 [|4|] in
  let params = Kaun.init q_network ~rngs:rng dummy_input in
  let opt_state = optimizer.init params in
  
  (* Collect experience and train *)
  let obs, _ = env.reset () in
  (* ... training logic ... *)
```

## API Overview

### Spaces

Define observation and action spaces:

```ocaml
(* Discrete space with n actions *)
let action_space = Space.Discrete 2

(* Continuous box space *)
let observation_space = Space.Box {
  low = Rune.create Rune.float32 [|2|] [| -1.0; -1.0 |];
  high = Rune.create Rune.float32 [|2|] [| 1.0; 1.0 |];
  shape = [|2|];
}

(* Sample from space *)
let action = Space.sample ~rng action_space
```

### Environments

Environments follow the Gymnasium API:

```ocaml
(* Create environment *)
let env = Envs.cartpole ()

(* Reset environment *)
let obs, info = env.reset ~seed:42 ()

(* Step environment *)
let next_obs, reward, terminated, truncated, info = env.step action

(* Access environment properties *)
let obs_space = env.observation_space
let act_space = env.action_space
```

Available environments:
- `Envs.cartpole ()` - Classic cart-pole balancing
- `Envs.mountain_car ()` - Car climbing a hill  
- `Envs.pendulum ()` - Continuous control pendulum

### Replay Buffers

Store and sample experience for off-policy learning:

```ocaml
(* Create buffer *)
let buffer = Buffer.create ~capacity:10000

(* Add experience *)
Buffer.add buffer {
  obs = current_obs;
  action = action_taken;
  reward = reward_received;
  next_obs = next_observation;
  terminated = episode_ended;
}

(* Sample batch *)
let batch = Buffer.sample buffer ~rng ~batch_size:32

(* Check buffer status *)
let size = Buffer.size buffer
let is_full = Buffer.is_full buffer
```

### Training Utilities

Helper functions for RL algorithms:

```ocaml
(* Compute Generalized Advantage Estimation *)
let advantages, returns = Training.compute_gae
  ~rewards
  ~values
  ~dones
  ~gamma:0.99
  ~gae_lambda:0.95

(* Compute discounted returns *)
let returns = Training.compute_returns
  ~rewards
  ~dones
  ~gamma:0.99

(* Normalize tensors *)
let normalized = Training.normalize tensor ~eps:1e-8 ()

(* Evaluate policy *)
let stats = Training.evaluate env
  ~policy:(fun obs -> (* your policy *))
  ~n_eval_episodes:10
```

## Examples

The `example/` directory contains:
- `cartpole_random.ml` - Simple random agent on CartPole
- `dqn_cartpole.ml` - Deep Q-Network implementation using Kaun

Run examples with:
```bash
dune exec dev/fehu/example/cartpole_random.exe
dune exec dev/fehu/example/dqn_cartpole.exe
```

## Architecture

Fehu is designed to be composable with other Raven packages:
- **Rune**: Provides tensor operations and automatic differentiation
- **Kaun**: Provides neural network layers and optimizers
- **Fehu**: Provides RL-specific components (environments, buffers, utilities)

This separation allows you to:
- Use any Kaun model as a policy or value function
- Apply Rune's autodiff for policy gradient methods
- Leverage Kaun's optimizers for training
- Combine with other Raven packages for visualization (Hugin) or vision (Sowilo)

## Status

Fehu is in active development. Current status:
- ✅ Core environment API
- ✅ Classic control environments
- ✅ Replay buffers
- ✅ Training utilities
- ✅ Integration with Kaun
- ⚠️  DQN example (needs stack operation fix)
- 🚧 Additional algorithms (PPO, SAC, etc.)
- 🚧 Vectorized environments
- 🚧 More environments
