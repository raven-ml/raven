# Fehu RL Examples

This directory contains reinforcement learning examples using the Fehu environment library and Kaun deep learning framework.

## Examples

### 01-random-agent
A simple random agent that interacts with the RandomWalk environment.

**Run:**
```bash
dune exec fehu/examples/01-random-agent/random_agent.exe
```

### 02-q-learning
Tabular Q-learning implementation on the RandomWalk environment.

**Run:**
```bash
dune exec fehu/examples/02-q-learning/q_learning.exe
```

### 03-policy-gradient
Policy gradient (REINFORCE) algorithm using Kaun neural networks on RandomWalk.

**Run:**
```bash
dune exec fehu/examples/03-policy-gradient/policy_gradient.exe
```

### 04-dqn
Deep Q-Network implementation using Kaun on a custom GridWorld environment.

**Run:**
```bash
dune exec fehu/examples/04-dqn/dqn.exe
```

### 05-sokoban
A Reinforce agent solving Sokoban levels using Kaun neural networks.

**Run:**
```bash
dune exec fehu/examples/05-sokoban/sokoban.exe
```
