# fehu ᚠ Documentation

Fehu is our Gymnasium + Stable Baselines. It's the reinforcement learning framework built on top of Rune and Kaun.

## What fehu Does

Fehu gives you environments and algorithms for reinforcement learning. Create environments like CartPole or GridWorld, train agents with DQN or REINFORCE, and evaluate policies. If you've used Gymnasium or Stable Baselines3, you'll feel at home.

The name comes from the rune ᚠ meaning "wealth" or "reward." Fitting for a reinforcement learning library focused on maximizing returns.

## Current Status

Fehu is ready for alpha release. It provides complete implementations of classic RL environments and algorithms.

What's available:
- **Environments**: CartPole-v1, MountainCar-v0, GridWorld, RandomWalk
- **Algorithms**: REINFORCE (policy gradient), DQN (deep Q-network)
- **Infrastructure**: Experience replay buffers, training utilities, evaluation metrics
- **Examples**: Working examples for all algorithms and environments

This is enough to train real agents and benchmark against standard implementations. More algorithms and environments will come in future releases.

## Design Philosophy

Fehu aims for Gymnasium's simplicity and Stable Baselines3's completeness, but with OCaml's type safety. Environments are strongly typed - the compiler catches observation/action space mismatches at compile time. Agents are immutable values - policy updates return new agents rather than mutating state.

Everything is functional and composable. Experience replay is just an array of transitions. Training loops are pure functions. This makes testing, debugging, and distributed training straightforward.

## Learn More

- [Getting Started](/docs/fehu/getting-started/) - Train your first RL agent
- [DQN GridWorld Demo](./dqn-demo.md)
