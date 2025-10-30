# Getting Started with fehu

This guide shows you how to create environments and train reinforcement learning agents with fehu.

## Installation

Fehu isn't released yet. When it is, you'll install it with:

```bash
opam install fehu
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build fehu
```

## Your First Environment

Here's how to create and interact with an environment:

```ocaml
open Fehu

(* Create environment with RNG *)
let rng = Rune.Rng.key 42 in
let env = Fehu_envs.Cartpole.make ~rng ()

(* Reset environment *)
let obs, _info = Env.reset env ()

(* Run one episode *)
let rec run_episode rng total_reward =
  (* Sample random action *)
  let action, rng = Space.sample ~rng (Env.action_space env) in

  (* Take step *)
  let transition = Env.step env action in
  let new_total = total_reward +. transition.reward in

  if transition.terminated || transition.truncated then
    Printf.printf "Episode finished. Total reward: %.0f\n" new_total
  else
    run_episode rng new_total
in

run_episode rng 0.0
```

## Training with REINFORCE

Here's a complete example training CartPole with the REINFORCE algorithm:

```ocaml
open Fehu
open Kaun

(* Create policy network *)
let policy_net = Layer.sequential [
  Layer.linear ~in_features:4 ~out_features:128 ();
  Layer.relu ();
  Layer.linear ~in_features:128 ~out_features:2 ();
]

(* Create environment and agent *)
let rng = Rune.Rng.key 42 in
let env = Fehu_envs.Cartpole.make ~rng () in

let config = {
  Fehu_algorithms.Reinforce.default_config with
  learning_rate = 0.001;
  gamma = 0.99;
  reward_scale = 0.01;
  entropy_coef = 0.01;
  max_episode_steps = 500;
}

(* Train for 100,000 timesteps *)
let params, state = Fehu_algorithms.Reinforce.train
  ~env ~policy_network:policy_net ~rng ~config
  ~total_timesteps:100_000
  ~callback:(fun metrics ->
    if metrics.total_episodes > 0 && metrics.total_episodes mod 10 = 0 then
      Printf.printf "Episode %d: Return = %.2f, Length = %d\n"
        metrics.total_episodes metrics.episode_return metrics.episode_length;
    `Continue)
  ()

let greedy_action params obs =
  let obs_batched =
    match Rune.shape obs with
    | [| features |] -> Rune.reshape [| 1; features |] obs
    | [| 1; _ |] -> obs
    | _ -> obs
  in
  let logits = Kaun.apply policy_net params ~training:false obs_batched in
  let action_idx =
    Rune.argmax logits ~axis:(-1) ~keepdims:false |> Rune.cast Rune.int32
  in
  let scalar =
    Rune.reshape [||] action_idx |> Rune.to_array |> fun arr -> arr.(0)
  in
  Rune.scalar Rune.int32 scalar

(* Evaluate trained agent *)
let eval_stats = Training.evaluate env
  ~policy:(fun obs -> greedy_action params obs)
  ~n_episodes:10 ()

Printf.printf "Average reward: %.2f\n" eval_stats.mean_reward
```

## Training with DQN

Here's how to train with Deep Q-Networks:

```ocaml
open Fehu
open Kaun

(* Create Q-network *)
let q_network = Layer.sequential [
  Layer.linear ~in_features:4 ~out_features:64 ();
  Layer.relu ();
  Layer.linear ~in_features:64 ~out_features:64 ();
  Layer.relu ();
  Layer.linear ~in_features:64 ~out_features:2 ();
]

(* Create environment *)
let rng = Rune.Rng.key 42 in
let env = Fehu_envs.Cartpole.make ~rng ()

(* Initialize Q-network and experience replay *)
let params = Kaun.init q_network ~rngs:rng ~dtype:Rune.float32 in
let replay_buffer = Buffer.Replay.create ~capacity:10_000 in

(* Training loop (simplified - see examples for full implementation) *)
for episode = 1 to 500 do
  let obs, _info = Env.reset env () in
  (* ... epsilon-greedy action selection ... *)
  (* ... store transitions in replay_buffer ... *)
  (* ... sample batch and update Q-network ... *)
done
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
