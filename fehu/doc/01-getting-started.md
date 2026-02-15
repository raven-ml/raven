# Getting Started with fehu

This guide shows you how to create environments and train reinforcement learning agents with fehu.

## Installation

Fehu isn't released yet. When it is, you'll install it with:

<!-- $MDX skip -->
```bash
opam install fehu
```

For now, build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build fehu
```

## Your First Environment

Here's how to create and interact with an environment:

```ocaml
open Fehu

let () =
  let rng = Rune.Rng.key 42 in
  let env = Fehu_envs.Cartpole.make ~rng () in

  (* Reset environment *)
  let obs, _ = Env.reset env () in
  let shape = Rune.shape obs in
  Printf.printf "Observation: %d dimensions\n" shape.(0);

  (* Run a few random steps *)
  let rec loop rng step reward =
    if step >= 10 then
      Printf.printf "Total reward after %d steps: %.0f\n" step reward
    else
      let action, rng = Space.sample ~rng (Env.action_space env) in
      let t = Env.step env action in
      if t.terminated || t.truncated then
        Printf.printf "Episode ended at step %d, reward: %.0f\n"
          (step + 1) (reward +. t.reward)
      else
        loop rng (step + 1) (reward +. t.reward)
  in
  loop rng 0 0.0
```

## Training with REINFORCE

Here's how to set up and train CartPole with the REINFORCE algorithm:

```ocaml
open Kaun

(* Define policy network: 4 obs dims -> 128 hidden -> 2 actions *)
let policy_net = Layer.sequential [
  Layer.linear ~in_features:4 ~out_features:128 ();
  Layer.relu ();
  Layer.linear ~in_features:128 ~out_features:2 ();
]

let () =
  let rng = Rune.Rng.key 42 in
  let env = Fehu_envs.Cartpole.make ~rng () in

  let config = {
    Fehu_algorithms.Reinforce.default_config with
    learning_rate = 0.001;
    gamma = 0.99;
    max_episode_steps = 500;
  } in

  (* Train for a few episodes *)
  let params, state = Fehu_algorithms.Reinforce.train
    ~env ~policy_network:policy_net ~rng ~config
    ~total_timesteps:500
    ~callback:(fun m ->
      Printf.printf "Episode %d: return=%.0f, length=%d\n"
        m.total_episodes m.episode_return m.episode_length;
      `Continue)
    ()
  in

  let m = Fehu_algorithms.Reinforce.metrics state in
  Printf.printf "Trained for %d episodes (%d steps)\n"
    m.total_episodes m.total_steps;
  Printf.printf "Network: %d parameters\n" (Ptree.count_parameters params)
```

## Training with DQN

Here's how to set up and train with Deep Q-Networks:

```ocaml
(* Define Q-network *)
let q_network = Layer.sequential [
  Layer.linear ~in_features:4 ~out_features:64 ();
  Layer.relu ();
  Layer.linear ~in_features:64 ~out_features:64 ();
  Layer.relu ();
  Layer.linear ~in_features:64 ~out_features:2 ();
]

let () =
  let rng = Rune.Rng.key 42 in
  let env = Fehu_envs.Cartpole.make ~rng () in

  let config = {
    Fehu_algorithms.Dqn.default_config with
    learning_rate = 0.001;
    gamma = 0.99;
    buffer_capacity = 1_000;
    batch_size = 32;
  } in

  (* Train for a few steps *)
  let params, state = Fehu_algorithms.Dqn.train
    ~env ~q_network ~rng ~config
    ~total_timesteps:100
    ~callback:(fun m ->
      (match m.episode_return with
       | Some r -> Printf.printf "Episode %d: return=%.0f\n" m.total_episodes r
       | None -> ());
      `Continue)
    ()
  in

  let m = Fehu_algorithms.Dqn.metrics state in
  Printf.printf "Q-network: %d parameters\n" (Ptree.count_parameters params);
  Printf.printf "After %d steps, epsilon=%.2f\n" m.total_steps m.epsilon
```

## Next Steps

- **[Examples](https://github.com/raven-ml/raven/tree/main/fehu/examples)** - Complete, runnable examples for all algorithms

## Available Environments

Fehu currently provides:

- **CartPole-v1** - Classic cart-pole balancing (matches Gymnasium)
- **MountainCar-v0** - Drive up a hill using momentum (matches Gymnasium)
- **GridWorld** - Simple 5x5 grid navigation with obstacles
- **RandomWalk** - One-dimensional random walk

More environments coming soon!

## Available Algorithms

- **REINFORCE** - Monte Carlo policy gradient with optional baseline
- **DQN** - Deep Q-Networks with experience replay and target networks

PPO and A2C coming in future releases.
