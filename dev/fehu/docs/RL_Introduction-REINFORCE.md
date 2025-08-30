# Reinforcement Learning: An Introduction to REINFORCE

Welcome to reinforcement learning! If you're familiar with supervised learning and neural network training, you're about to discover a fundamentally different approach to machine learning.

{pause}

## What is Reinforcement Learning? {#rl-definition}

{.definition title="Reinforcement Learning"}
Instead of learning from labeled examples, an **agent** learns by **acting** in an **environment** and receiving **rewards**.

{pause up=rl-definition}

### The RL Framework

> **Agent**: The learner (your neural network)
> 
> **Environment**: The world the agent interacts with
> 
> **Actions**: What the agent can do
> 
> **States**: What the agent observes
> 
> **Rewards**: Feedback signal (positive or negative)

{pause}

Think of it like learning to play a game:
- You don't know the rules initially
- You try actions and see what happens
- Good moves get rewarded, bad moves get punished
- You gradually learn a strategy

{pause center=rl-definition}

---

## Key Differences from Supervised Learning {#differences}

{.block title="Supervised Learning"}
- Fixed dataset with input-output pairs
- Learn to minimize prediction error
- Single training phase

{pause}

{.block title="Reinforcement Learning"}
- Dynamic interaction with environment  
- Learn to maximize cumulative reward
- Continuous learning from experience

{pause}

**No labeled data** - the agent must discover what actions are good through trial and error.

{pause down=differences}

---

## The Policy: Your Agent's Strategy {#policy-intro}

{.definition title="Policy π(a|s)"}
The probability of taking action **a** in state **s**.

This is what your neural network learns to represent!

{pause up=policy-intro}

### Why Probabilistic Policies?

From Sutton & Barto:

> "action probabilities change smoothly as a function of the learned parameter, whereas in ε-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values"

{pause}

**Smooth changes** = **stable learning**

{pause}

{.example title="Policy Examples"}
- **Discrete actions**: Softmax over action preferences
- **Continuous actions**: Mean and variance of Gaussian distribution

{pause center=policy-intro}

---

## Episodes and Returns {#episodes}

{.definition title="Episode"}
A complete sequence of interactions from start to terminal state.

{.definition title="Return G_t"}
The total reward from time step t until the end of the episode:
$$G_t = R_{t+1} + R_{t+2} + ... + R_T$$

{pause up=episodes}

### The Goal

**Maximize expected return** by learning a better policy.

{pause}

But how do we improve a policy that's represented by a neural network?

{pause down=episodes}

---

## Enter REINFORCE {#reinforce-intro}

{.theorem title="The REINFORCE Algorithm"}
A **policy gradient** method that directly optimizes the policy parameters to maximize expected return.

{pause up=reinforce-intro}

### Core Insight

We want to:
1. **Increase** the probability of actions that led to high returns
2. **Decrease** the probability of actions that led to low returns

{pause}

From Sutton & Barto:

> "it causes the parameter to move most in the directions that favor actions that yield the highest return"

{pause center=reinforce-intro}

---

## The Policy Gradient Theorem {#gradient-theorem}

The gradient of expected return with respect to policy parameters θ:

$$\nabla_\theta J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla_\theta \pi(a|s,\theta)$$

{pause}

This looks complicated, but REINFORCE gives us a simple way to estimate it!

{pause}

{.theorem title="REINFORCE Gradient Estimate"}
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause up=gradient-theorem}

### What This Means

From Sutton & Barto:

> "Each increment is proportional to the product of a return G_t and a vector, the gradient of the probability of taking the action actually taken divided by the probability of taking that action"

{pause down=gradient-theorem}

---

## REINFORCE Algorithm Steps {#algorithm}

{.block title="REINFORCE Algorithm"}

1. **Initialize** policy parameters θ randomly
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t in episode:
     - Calculate return: $G_t = \sum_{k=t+1}^T R_k$
     - Update: $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)$

{pause up=algorithm}

### Key Properties

From Sutton & Barto:

> "REINFORCE uses the complete return from time t, which includes all future rewards up until the end of the episode"

{pause}

This makes it an **unbiased** but **high variance** estimator.

{pause center=algorithm}

---

## Implementation in Neural Networks {#implementation}

If your policy network outputs action probabilities, the gradient update becomes:

```ocaml
(* Compute log probability gradient *)
let log_prob_grad = compute_gradient_log_prob action_taken state in
(* Scale by return *)
let policy_grad = G_t *. log_prob_grad in
(* Update parameters *)
update_parameters policy_grad learning_rate
```

{pause up=implementation}

### In Practice

You'll typically:
1. Use **automatic differentiation** to compute ∇ ln π
2. **Collect episodes** in batches for stability
3. Apply **baseline subtraction** to reduce variance

{pause down=implementation}

---

## Reducing Variance with Baselines {#baselines}

REINFORCE can be **very noisy**. We can subtract a baseline b(s) from returns:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[(G_t - b(S_t)) \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause up=baselines}

From Sutton & Barto:

> "The baseline can be any function, even a random variable, as long as it does not vary with a; the equation remains valid because the subtracted quantity is zero"

{pause}

> "In some states all actions have high values and we need a high baseline to differentiate the higher valued actions from the less highly valued ones"

{pause}

{.example title="Common Baselines"}
- **Constant**: Average return over recent episodes
- **State-dependent**: Value function V(s) learned separately

{pause center=baselines}

---

## REINFORCE with Baseline {#reinforce-baseline}

{.block title="REINFORCE with Baseline Algorithm"}

1. **Initialize** policy parameters θ and baseline parameters w
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t:
     - $G_t = \sum_{k=t+1}^T R_k$
     - $\delta = G_t - b(S_t,w)$
     - $\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \ln \pi(A_t|S_t,\theta)$
     - $w \leftarrow w + \alpha_w \delta \nabla_w b(S_t,w)$

{pause up=reinforce-baseline}

The baseline is learned to predict expected returns, reducing variance without introducing bias.

{pause down=reinforce-baseline}

---

## Practical Considerations {#practical}

### Learning Rates

From Sutton & Barto:

> "Choosing the step size for values (here α_w) is relatively easy... much less clear how to set the step size for the policy parameters"

{pause up=practical}

**Policy updates are more sensitive** - start with smaller learning rates for θ.

{pause}

### Actor-Critic Methods

From Sutton & Barto:

> "Methods that learn approximations to both policy and value functions are often called actor–critic methods"

{pause}

REINFORCE with baseline is a simple actor-critic method:
- **Actor**: The policy π(a|s,θ)  
- **Critic**: The baseline b(s,w)

{pause center=practical}

---

## Summary {#summary}

{.block title="Key Takeaways"}

✓ **RL learns from interaction**, not labeled data

✓ **REINFORCE optimizes policies directly** using policy gradients  

✓ **Returns weight gradient updates** - high returns → strengthen action probabilities

✓ **Baselines reduce variance** without introducing bias

✓ **Actor-critic architectures** combine policy and value learning

{pause up=summary}

### Next Steps

- Implement REINFORCE on a simple environment
- Experiment with different baseline functions  
- Explore more advanced policy gradient methods (PPO, A3C)
- Consider trust region methods for more stable updates

{pause}

**You now have the foundation to start learning policies through interaction!**

{pause center=summary}

---

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.