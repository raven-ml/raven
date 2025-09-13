# Reinforcement Learning: An Introduction to REINFORCE

Welcome to reinforcement learning! If you're familiar with supervised learning and neural network training, you're about to discover a fundamentally different approach to machine learning.

## What is Reinforcement Learning? {#rl-definition}

{.definition title="Reinforcement Learning"}
Instead of learning from labeled examples, an **agent** learns by **acting** in an **environment** and receiving **rewards**.

{pause center #rl-framework}
### The RL Framework

> **Agent**: The learner (your neural network with weights θ)
> 
> **Environment**: The world the agent interacts with  
> 
> **Actions**: What the agent can do
> 
> **States**: What the agent observes about the environment
> 
> **Rewards**: Feedback signal (positive or negative)

{pause up=rl-framework}

{.example title="Concrete Example: Sokoban Puzzle"}
- **Environment**: Grid world with boxes, walls, targets
- **Agent**: Neural network controlling a character
- **States**: Current positions of character, boxes, walls (as pixel grid or feature vector)
- **Actions**: Move up, down, left, right (4 discrete actions)
- **Rewards**: +10 for solving puzzle, -1 per step, -5 for invalid moves

{pause}

### Workshop Setup: Your First Environment

Let's start by creating a simple grid world environment using Fehu:

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide1.ml}

{pause}

Think of it like learning to play a game:
- You (the neural network) don't know the rules initially
- You try actions and see what happens to the environment
- Good moves get rewarded, bad moves get punished
- You gradually learn a strategy by updating your weights

***

{pause center #rl-supervised-differences}
## Key Differences from Supervised Learning {#differences}

| **Aspect** | **Supervised Learning** | **Reinforcement Learning** |
|------------|-------------------------|---------------------------|
| **Data** | Fixed dataset with input-output pairs | Generated through interaction with environment |
| **Objective** | Minimize loss function (e.g., MSE, cross-entropy) | Maximize expected cumulative reward |
| **Learning** | Single training phase on static data | Continuous learning from experience |
| **Feedback** | Direct labels for each input | Delayed, sparse rewards for sequences of actions |
| **Data Generation** | Pre-collected and labeled | **You generate your own dataset** by acting |

{pause up=rl-supervised-differences}

### What "Dynamic Interaction" Means for Your Neural Network

In supervised learning, your network processes: `input → prediction`

In RL, your network (the agent) does: `state → action → new_state + reward`
- **Each forward pass** produces an action that changes the world
- **The world responds** with a new state and reward
- **You experience the consequences** of your network's outputs
- **Your training data** comes from your own actions

{pause}

**No pre-labeled data** - the agent must discover what actions are good through trial and error.

***

{pause center #policy-intro}
## The Policy: Your Agent's Strategy

{.definition title="Policy π(a|s,θ)"}
The probability of taking action **a** in state **s**, parameterized by neural network weights **θ**.

{.definition title="State vs Observation (Precise Definition)"}
- **Environment state**: Complete description of environment (all Sokoban box/wall positions)
- **Observation**: What the agent sees (may be partial, e.g., local 5×5 grid view)
- **Information state**: What your neural network system represents about the environment (e.g., current observation + recurrent hidden state from past observations)

{pause down}
Both environment and information states are **Markovian** - they capture all relevant history for decision-making.

{pause up=policy-intro #prob-policies}
### Why Probabilistic Policies?
**Key insight**: We need to **synthesize our own training dataset** through exploration!

{pause down}
> From Sutton & Barto:
> 
> > "action probabilities change smoothly as a function of the learned parameter, whereas in ε-greedy selection the action probabilities may change dramatically for an arbitrarily small change in the estimated action values"

{pause up=prob-policies}
**Benefits**:
1. **Smooth changes** = **stable learning**
2. **Natural exploration** - probability spreads across actions
3. **Dataset synthesis** - stochastic policy generates diverse experiences

{pause .example title="Policy Examples in Sokoban"}
- **Neural network output**: 4 action preferences [up, down, left, right]
- **Softmax conversion**: [0.1, 0.6, 0.2, 0.1] probabilities  
- **Action sampling**: Choose "down" with 60% probability
- **Learned parameters**: θ represents all network weights and biases

{pause}

### Workshop Part 2: Create Your First Policy Network

{pause down="~duration:15"}

{.note title="Numerical Stability Tip"}
> When computing log probabilities from a neural network:
> - **Avoid**: `log(softmax(logits))` - can underflow for small probabilities
> - **Prefer**: Direct log-softmax computation: `logits - log(sum(exp(logits)))`
> - **Better**: Use the numerically stable version with max subtraction:
>   ```
>   log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
>   ```
> This prevents overflow in exp() and maintains precision across the full range.

{pause down="~duration:15"}

{slip include src=../example/sokoban/workshop/slide2.ml}

***

{pause up #episodes}
## Episodes and Returns

{.definition title="Episode"}
A complete sequence of interactions from start to terminal state.

{.definition title="Return $G_t$"}
The total reward from time step t until the end of the episode:
$$G_t = R_{t+1} + R_{t+2} + ... + R_T$$

{.definition title="Value of a State $V^\pi(s)$"}
The expected return when starting from state s and following policy π:
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

{pause down}
> ### Supervised Learning Analogy
> 
> | **Supervised Learning** | **Reinforcement Learning** |
> |--------------------------|----------------------------|
> | **Input**: x | **State**: s |
> | **Target**: y | **Return**: G_t (discovered through interaction) |
> | **Loss**: (prediction - target)² . . | **"Loss"**: -(expected return) |
> | **Gradient**: ∇ loss | **Policy Gradient**: ∇ (expected return) |

{pause center}
### The Goal

**Maximize expected return** by updating network weights θ to improve the policy.

{pause}

But how do we compute gradients when the "target" (return) depends on our own actions?

{pause}

### Workshop Part 3: Collect an Episode

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide3.ml}

***

{pause center #reinforce-intro}
## Enter REINFORCE

{.definition title="The REINFORCE Algorithm"}
A **policy gradient** method that directly optimizes the policy parameters (network weights θ) to maximize expected return.

{pause up=reinforce-intro}
### Core Insight

We want to:
1. **Increase** the probability of actions that led to high returns
2. **Decrease** the probability of actions that led to low returns

{pause}

{.example title="Sokoban Example"}
> If pushing a box toward a target (action "up") led to solving the puzzle (G_t = +9):
> - **Increase** probability of choosing "up" in that state
> - **Strengthen** neural network weights that favor "up"

{pause}

From Sutton & Barto:

> "it causes the parameter to move most in the directions that favor actions that yield the highest return"

***

{pause up #gradient-theorem}
## The Policy Gradient Theorem

The gradient of expected return with respect to policy parameters θ:

$$\nabla_\theta J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla_\theta \pi(a|s,\theta)$$

{pause}

This looks complicated, but REINFORCE gives us a simple way to estimate it!

{pause .theorem title="REINFORCE Gradient Estimate"}
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause up=gradient-theorem}
### What This Means

From Sutton & Barto:

> "Each increment is proportional to the product of a return $G_t$ and a vector, the gradient of the probability of taking the action actually taken divided by the probability of taking that action"

***

{pause center #algorithm-reinforce}
## REINFORCE Algorithm Steps

{.block title="REINFORCE Algorithm"}
1. **Initialize** policy parameters θ (neural network weights) randomly
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t in episode:
     - Calculate return: $G_t = \sum_{k=t+1}^T R_k$
     - Update: $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \ln \pi(A_t|S_t,\theta)$

{pause up=algorithm-reinforce}

### Workshop Part 4: Implement Basic REINFORCE

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide4.ml}

{pause center #high-variance-problem}
### Key Properties: High Variance Problem

From Sutton & Barto:

> "REINFORCE uses the complete return from time t, which includes all future rewards up until the end of the episode"

This makes it an **unbiased** but **high variance** estimator.

{#impact-high-variance}
**Practical Impact of High Variance**:
- Learning is **slow** and **unstable**
- Need many episodes to see improvement
- Updates can be huge (good episode) or tiny (bad episode)
- **Episode-by-episode learning** amplifies noise

{pause up=high-variance-problem}
### On-Policy vs Off-Policy

{.definition title="Key Terms"}
- **On-policy**: Using data from the **current** policy π(·|·,θ) 
- **Off-policy**: Using data from a **different** policy (e.g., old θ values)

{pause up=impact-high-variance}

### Batching vs. On-Policy Learning Trade-off

**Batching reduces variance** by averaging over multiple episodes, but:
- **Risk**: Collected episodes become **off-policy** as θ changes during batch collection
- **Why off-policy is bad for REINFORCE**: The gradient estimate ∇ln π(A_t|S_t,θ) assumes actions came from current policy θ, but they came from old θ
- **Solution**: Use smaller batches or update more frequently  
- **Balance**: Variance reduction vs. policy staleness

***

{pause up #implementation-reinforce}
## Implementation in Neural Networks

If your policy network outputs action probabilities, the gradient update becomes:

```ocaml
(* Compute log probability gradient *)
let log_prob_grad =
  compute_gradient_log_prob action_taken state in
(* Scale by return *)
let policy_grad = G_t *. log_prob_grad in
(* Update parameters *)
update_parameters policy_grad learning_rate
```

{pause up=implementation-reinforce}
### In Practice

You'll typically:
1. Use **automatic differentiation** to compute ∇ ln π
2. **Collect episodes** in batches for stability
3. Apply **baseline subtraction** to reduce variance

***

{pause up #baselines}
## Reducing Variance with Baselines

REINFORCE can be **very noisy**. We can subtract a baseline b(s) from returns:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\left[(G_t - b(S_t)) \nabla_\theta \ln \pi(A_t|S_t,\theta)\right]$$

{pause}

From Sutton & Barto:

> "The baseline can be any function, even a random variable, as long as it does not vary with a; the equation remains valid because the subtracted quantity is zero"

{pause}

> "In some states all actions have high values and we need a high baseline to differentiate the higher valued actions from the less highly valued ones"

{pause}

{.example title="Baseline Options"}
> 
> **Simple Average Baseline**:
> - $b = \frac{1}{N} \sum_{i=1}^N G_{t,i}$ (average return over past N episodes)
> - **Not learned** - just computed from episode history
> - Easy to implement, somewhat effective
> 
> **Learned State-Dependent Baseline**:
> - $b(s,w)$ - separate neural network with weights w
> - **Learned** to predict V(s) using gradient descent
> - More complex but much more effective

{pause down}
### Workshop Part 5: Add a Simple Baseline

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide5.ml}

***

{pause center #reinforce-baseline}
## REINFORCE with Learned Baseline

{.block title="REINFORCE with Learned Baseline Algorithm"}
1. **Initialize** policy parameters θ and **baseline parameters w**
2. **For each episode**:
   - Generate episode following π(·|·,θ)
   - For each step t:
     - $G_t = \sum_{k=t+1}^T R_k$
     - $\delta = G_t - b(S_t,w)$ ← **prediction error**
     - $\theta \leftarrow \theta + \alpha_\theta \delta \nabla_\theta \ln \pi(A_t|S_t,\theta)$ ← **policy update**
     - $w \leftarrow w + \alpha_w \delta \nabla_w b(S_t,w)$ ← **baseline update**

{pause up=reinforce-baseline}
The baseline **neural network** is learned to predict expected returns, reducing variance without introducing bias.

**Two networks training simultaneously**:
- **Policy network**: θ parameters, outputs action probabilities
- **Baseline network**: w parameters, outputs state value estimates

{pause}

### Workshop Part 6: Learned Baseline (Actor-Critic)

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide6.ml}

***

{pause center}
### Actor-Critic Methods

From Sutton & Barto:

> "Methods that learn approximations to both policy and value functions are often called actor–critic methods"

REINFORCE with baseline is a simple actor-critic method:
- **Actor**: The policy π(a|s,θ)  
- **Critic**: The baseline b(s,w)

***

{pause center}
## Summary {#summary}

{.block title="Key Takeaways"}
> 
> ✓ **RL learns from interaction**, not labeled data
> 
> ✓ **REINFORCE optimizes policies directly** using policy gradients  
> 
> ✓ **Returns weight gradient updates** - high returns → strengthen action probabilities
> 
> ✓ **Baselines reduce variance** without introducing bias
> 
> ✓ **Actor-critic architectures** combine policy and value learning

{pause up=summary}
### Next Steps

- Implement the Sokoban environment
- Implement a policy network model
- Implement REINFORCE
- Enhance with the constant baseline

{pause}

### Workshop Summary: What We Built

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide_pip.ml}

***

{pause center #policy-ratios}
## Policy Ratios and Importance Sampling

REINFORCE has a fundamental limitation: it can only use data from the **current** policy.

{pause up=policy-ratios}
### The Problem with Policy Updates

After each gradient step, our policy π(a|s,θ) changes. But what about all that expensive experience we just collected?

{.example title="Sokoban Training Reality"}
- Collect 1000 episodes with current policy → expensive!
- Update policy weights θ → policy changes
- Old episodes are now **off-policy** → can't use them directly

{pause center #importance-sampling}
> ### Solution: Importance Sampling
> 
> **Key insight**: We can reuse off-policy data by weighting it appropriately.

{pause}

{.definition title="Policy Ratio"}
$$\text{ratio}_{t} = \frac{\pi_{\theta_{new}}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

This tells us how much more (or less) likely the action was under the new policy vs. old policy.

{pause down}
**Importance-weighted REINFORCE update**:
$$\theta \leftarrow \theta + \alpha \cdot \text{ratio}_t \cdot G_t \cdot \nabla_\theta \ln \pi(a_t|s_t,\theta)$$

{pause up=importance-sampling}
{.example title="Policy Ratio Interpretation"}
- `ratio = 2.0`: New policy twice as likely to take this action
- `ratio = 0.5`: New policy half as likely to take this action  
- `ratio = 1.0`: No change in action probability

***

{pause center #clipping}
## The Problem: Unbounded Policy Updates

Importance sampling allows off-policy learning, but creates a new problem: **unbounded ratios**.

{pause up=clipping}

{.example title="When Ratios Explode"}
If old policy had π_old(action) = 0.01 and new policy has π_new(action) = 0.9:

**ratio = 0.9 / 0.01 = 90**

With a high return G_t = +10: **update = 90 × 10 = 900**

This massive update can destabilize training!

{pause}

### Solution: Clipped Policy Updates

**PPO-style clipping** limits how much the policy can change in one update:

$$L^{CLIP}(\theta) = \min\left(\text{ratio}_t \cdot A_t, \; \text{clip}(\text{ratio}_t, 1-\epsilon, 1+\epsilon) \cdot A_t\right)$$

{pause}

{.definition title="Clipping Parameters"}
- **ε = 0.2** (typical): Allow 20% change in action probabilities
- **clip(x, 1-ε, 1+ε)**: Forces ratio to stay in [0.8, 1.2] range
- **min(...)**: Takes the more conservative update


{pause down .example title="Clipping in Action (ε = 0.2)"}
- `ratio = 90` → clipped to `1.2` → much smaller update
- `ratio = 0.01` → clipped to `0.8` → prevents tiny updates too
- `ratio = 1.1` → no clipping needed, within [0.8, 1.2]

{pause center}
### Workshop Part 7: Add Clipping for Stability

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide7.ml}

***

{pause center #kl-penalty}
## KL Divergence: Keeping Policies Close

Even with clipping, we want an additional safety mechanism to prevent the policy from changing too drastically.

{pause up=kl-penalty}

{.definition title="KL Divergence Penalty"}
$$D_{KL}[\pi_{old} \| \pi_{new}] = \sum_a \pi_{old}(a|s) \log \frac{\pi_{old}(a|s)}{\pi_{new}(a|s)}$$

**Measures how different two probability distributions are.**

{pause #kl-objective}
### KL-Regularized Objective

$$L_{total}(\theta) = L_{policy}(\theta) - \beta \cdot D_{KL}[\pi_{old} \| \pi_{new}]$$

{pause}

{.definition title="KL Penalty Parameters"}
- **β**: Controls penalty strength (e.g., 0.01, 0.04)
- **Higher β**: Keeps policy very close to old policy (stable but slow learning)
- **Lower β**: Allows more exploration (faster learning but less stable)


{pause down .example title="KL Penalty in Practice"}
> If policy changes dramatically → high KL divergence → large penalty  
> → discourages big changes
> 
> The penalty acts like a "trust region" - we trust small changes more than large ones.

{pause center}
### Workshop Part 8: Add KL Penalty

{pause down="~duration:15"}
{slip include src=../example/sokoban/workshop/slide8.ml}

{pause down}
> ### Why Both Clipping AND KL Penalty?
> 
> **Clipping**: Hard constraint on individual action probabilities  
> **KL Penalty**: Soft constraint on overall policy distribution
> 
> Together they provide robust stability for policy optimization.

***

{pause center #grpo-algorithm}
## Group Relative Policy Optimization (GRPO)

Now we can understand GRPO: **REINFORCE + GRPO Innovation + Clipping + KL Penalty**

{pause up=grpo-algorithm}

{.definition title="GRPO: The Complete Picture"}
> GRPO combines all the techniques we've learned:
> 1. **Group baselines** - Compare responses to the **same query** (GRPO's key innovation)
> 2. **Policy ratios** for off-policy learning
> 3. **Clipping** for stable updates  
> 4. **KL penalties** for additional safety

{pause}

{.definition title="Group Baselines: The Key Innovation"}
> Instead of comparing to historical episodes from different queries:
> - Generate **G responses** to the **same query**
> - Compute advantages relative to **this group**: $A_i = (r_i - \text{mean}_\text{group}) / (\text{std}_\text{group} + ε)$
> - Much better signal: "How good was this response compared to other attempts at the same problem?"

{pause center #grpo-steps}
### GRPO Algorithm Steps

{.block title="GRPO for LLM Fine-tuning"}
1. **Sample G responses** per query from current policy π_θ_old
2. **Evaluate rewards** r_i for each response using reward model
3. **Compute group advantages**: A_i = (r_i - mean_group) / (std_group + ε)
4. **Calculate clipped loss**: 
   - ratio_i = π_θ(response_i) / π_θ_old(response_i)
   - L_i = min(ratio_i × A_i, clip(ratio_i, 1-ε, 1+ε) × A_i)
5. **Add KL penalty**: L_total = L_policy - β × KL[π_θ_old || π_θ]
6. **Update policy**: θ ← θ - α ∇L_total

{pause up=grpo-steps}
### Why GRPO Works for LLMs

{.example title="GRPO vs REINFORCE Comparison" #grpo-for-llms}
> 
> **REINFORCE with constant baseline**:
> - Baseline: Average of past episodes (different queries)
> - No clipping → unstable updates
> - No policy ratios → must stay on-policy
> 
> **GRPO**:
> - **Better baseline**: Compare within responses to **same query**
> - **Clipped updates**: Stable learning with large batches
> - **Policy ratios**: Can reuse data across updates

{pause down=grpo-for-llms}

{#grpo-implementation}
### GRPO Implementation in Fehu

{pause down .warning title="Implementation Challenge: State Persistence"}
> **The Gym API limitation**: Standard Gym environments don't support state save/restore
> 
> GRPO requires collecting multiple trajectories from the **same initial state** to compute group-relative advantages. However:
> - Standard Gym API only has `reset()` and `step()` 
> - No `get_state()` or `set_state()` methods
> - After first trajectory, environment state has changed!
> 
> **Workarounds**:
> 1. **Deterministic reset with seed** - Reset to same seed, replay actions to reach target state
> 2. **Extended environment API** - Add state save/restore methods (non-standard)
> 3. **Environment copies** - Fork multiple environment instances (memory intensive)
> 4. **Batched environments** - Run parallel environments from same initial conditions (special case of 3)

***

{pause center #grpo-summary}
## GRPO: Why It Works

{.block title="Key Insights"}
> 
> > ✓ **Group baselines are better** - Compare responses to the same query, not different queries
> > 
> > ✓ **Clipping prevents instability** - Large policy updates are dangerous
> > 
> > ✓ **KL penalties add safety** - Trust regions keep learning stable
> > 
> > ✓ **Perfect for LLM fine-tuning** - Generate multiple responses easily

{pause up=grpo-summary}

### The Evolution: REINFORCE → GRPO

1. **REINFORCE**: Basic policy gradients with historical baselines
2. **+ Clipping**: Stable updates with policy ratios  
3. **+ KL Penalty**: Additional safety through regularization
4. **+ Group Baselines**: Better comparisons for LLM setting
5. **= GRPO**: Industrial-strength policy optimization for LLMs

{pause}

**GRPO doesn't replace REINFORCE - it's REINFORCE evolved for modern LLM training.**

**You now understand the complete journey from REINFORCE to GRPO!**

***

{pause down=ppo-multi-updates}

## PPO's Key Innovation: Multiple Updates Per Batch

While GRPO achieves sample efficiency through multiple trajectories from the same state, PPO takes a different approach: **reusing collected data through multiple gradient updates**.

### The Sample Efficiency Problem

Both REINFORCE and our REINFORCE++ (Slide 9) follow this pattern:
1. Collect one episode
2. Compute one gradient update
3. Discard the episode
4. Repeat

This throws away valuable data after a single use!

### PPO's Solution: Batch Collection and Reuse

```python
# Pseudocode for PPO's core innovation
def train_ppo(env, n_iterations, batch_size, n_epochs):
    for iteration in range(n_iterations):
        # Step 1: Collect BATCH of trajectories
        batch = collect_episodes(env, policy, batch_size)
        old_policy = copy(policy)

        # Step 2: Multiple optimization epochs on SAME data
        for epoch in range(n_epochs):  # Typically 3-10 epochs
            for trajectory in batch:
                # Compute ratio using FIXED old_policy
                ratio = pi(a|s) / old_pi(a|s)
                # Update with clipped objective
                loss = clip(ratio, 1-ε, 1+ε) * advantage
                update_policy(loss)
```

### Why This Works

1. **Fixed Reference Point**: `old_policy` stays constant during all epochs, providing a stable trust region
2. **Clipping Becomes Essential**: As the policy diverges from `old_policy` over multiple updates, clipping prevents overfitting to the batch
3. **Sample Efficiency**: Extract 3-10x more learning from each environment interaction

### The Tradeoff Space

| Algorithm | Trajectories | Updates | Variance Reduction |
|-----------|-------------|---------|-------------------|
| REINFORCE | 1 per step | 1 per trajectory | None |
| REINFORCE++ | 1 per step | 1 per trajectory | Clipping/KL |
| GRPO | Multiple from same state | 1 per trajectory | Relative advantages |
| PPO | Batch collection | Multiple per trajectory | Clipping + reuse |

### The Insight

- **GRPO**: "Let's collect better data" (multiple trajectories from same state)
- **PPO**: "Let's use our data better" (multiple updates per batch)
- **Optimal**: Combine both approaches!

{.note title="Exercise 3: Implementing Full PPO"}
> Upgrade REINFORCE++ to full PPO by adding:
> 1. Batch trajectory collection
> 2. Multiple optimization epochs per batch
> 3. Proper importance sampling with fixed old_policy
>
> See `exercise3.md` for detailed instructions.

### Key Implementation Considerations

1. **Batch Size**: Typically 32-2048 trajectories depending on environment
2. **Number of Epochs**: Usually 3-10 (too many causes overfitting)
3. **Minibatch Updates**: Large batches can be split into minibatches
4. **Early Stopping**: Stop epochs if KL divergence exceeds threshold

This completes our journey through modern policy optimization:
- Started with basic REINFORCE
- Added variance reduction techniques
- Introduced stability through clipping and KL penalties
- Explored GRPO's multiple trajectory approach
- Culminated in PPO's efficient data reuse

Each innovation addresses specific challenges while building on previous insights!

***

{pause down=curriculum}

## Part 2: Curriculum Learning with Sokoban

After mastering policy optimization algorithms, let's explore how to train agents on complex tasks using **curriculum learning** - the art of teaching through progressively harder challenges.

### The Challenge: Learning Complex Tasks

Imagine teaching an agent to solve this Sokoban puzzle:

```
#########
#   @   #  @ = player
# o o o #  o = box
#       #  x = target
# x x x #
#       #
#########
```

Starting with such complexity leads to:
- **Sparse rewards**: Agent rarely solves the puzzle randomly
- **No learning signal**: Without successes, gradients are uninformative
- **Wasted computation**: Millions of failed attempts

### The Solution: Start Simple, Build Up

Just like teaching a child mathematics (counting → addition → algebra), we teach RL agents progressively:

#### Stage 1: Straight Corridor (Difficulty 1/10)
```
#####
#@o #  Push one box straight
# x #  to the target
#####
```
- **Skills learned**: Basic push mechanics, goal understanding
- **Success rate**: 90% after 50 episodes

#### Stage 2: Simple Room (Difficulty 3/10)
```
#######
#@    #  Navigate around
# o   #  to push box
#  x  #
#######
```
- **Skills learned**: Path planning, spatial reasoning
- **Success rate**: 80% after 100 episodes

#### Stage 3: Multiple Boxes (Difficulty 5/10)
```
#######
#@    #  Coordinate multiple
# o o #  boxes to targets
# x x #
#######
```
- **Skills learned**: Sequential planning, avoiding deadlocks
- **Success rate**: 70% after 200 episodes

### Implementation: Automatic Curriculum Progression

{.code title="Slide 10: Curriculum Management"}
```ocaml
type curriculum_state = {
  current_stage: int;
  recent_wins: bool Queue.t;  (* Track last N episodes *)
  episodes_in_stage: int;
}

let should_advance state window_size =
  let wins = Queue.fold (+) 0 state.recent_wins in
  let win_rate = wins / window_size in
  win_rate >= stage.success_threshold

let update_curriculum state won =
  Queue.add won state.recent_wins;
  if should_advance state then
    advance_to_next_stage state
  else
    state
```

### Key Design Decisions

1. **Advancement Criteria**
   - Fixed win rate threshold (e.g., 80% over 100 episodes)
   - Minimum episodes in stage (prevent lucky streaks)
   - Optional: Regression detection (move back if struggling)

2. **Stage Design**
   - Each stage introduces ONE new concept
   - Skills transfer to next stage
   - Difficulty gap not too large

3. **Performance Window**
   - Rolling window (last 100 episodes)
   - Exponential moving average
   - Separate windows per stage

### Results: Curriculum vs Fixed Difficulty

| Metric | With Curriculum | Without (Always Hard) |
|--------|----------------|--------------------|
| Episodes to first win | 50 | 500+ |
| Final success rate | 70% | 30% |
| Training stability | High | Low |
| Skill transfer | Progressive | Random |

### Visualizing Progress

{.code title="Slide 11: Training with Curriculum"}
```
Episode   50 | Stage: Corridor     | Win rate: 92% | [ADVANCING]
Episode  120 | Stage: Simple Room  | Win rate: 85% | [ADVANCING]
Episode  250 | Stage: Multi-Box    | Win rate: 76% | [ADVANCING]
Episode  400 | Stage: Complex      | Win rate: 61% | [STABLE]
```

### Advanced Curriculum Techniques

1. **Adaptive Pacing**
   - Faster advancement for quick learners
   - Slower for struggling agents
   - Individual vs population-based

2. **Skill Decomposition**
   - Separate curricula for different skills
   - Combine learned behaviors
   - Transfer learning between tasks

3. **Procedural Generation**
   - Infinite variations within difficulty level
   - Prevents overfitting to specific layouts
   - Smooth difficulty interpolation

### Connection to Human Learning

Curriculum learning mirrors human education:
- **Scaffolding**: Temporary support removed gradually
- **Zone of Proximal Development**: Tasks just beyond current ability
- **Mastery Learning**: Solid foundation before advancement
- **Spiral Curriculum**: Revisit concepts with increasing depth

### Practical Tips

1. **Start Too Easy Rather Than Too Hard**
   - Early success builds good exploration
   - Prevents random policy collapse

2. **Monitor for Curriculum Hacking**
   - Agent might exploit easy stages
   - Add variety within each stage

3. **Consider Forgetting**
   - Skills from early stages may degrade
   - Occasional review episodes
   - Or lifelong learning techniques

### Exercises

{.note title="Exercise 4: Custom Curriculum Design"}
> Design a curriculum for teaching an agent to play chess:
> 1. Start with endgames (K+R vs K)
> 2. Add pieces gradually
> 3. Increase board complexity
>
> What would your stages be? How would you measure progress?

{.note title="Exercise 5: Reverse Curriculum"}
> Implement "reverse curriculum" where you start hard and make it easier when stuck.
> Compare with forward curriculum. When might each be better?

### Summary

Curriculum learning transforms impossible tasks into learnable sequences:
- **Gradual complexity** ensures constant learning signal
- **Automatic progression** adapts to agent capability
- **Skill building** creates robust, generalizable policies

Combined with modern RL algorithms (PPO, GRPO), curriculum learning enables training on tasks that would otherwise be intractable.

***

{pause down=egocentric-allocentric}

## Egocentric vs Allocentric: How Should Agents See the World?

When designing observations for RL agents, a fundamental choice is between **egocentric** (agent-centered) and **allocentric** (world-centered) representations. This decision profoundly impacts learning efficiency, generalization, and policy complexity.

### Allocentric: The World as It Is

In our Sokoban implementation (Slides 10-11), we use an allocentric representation:

```
#####     Grid position [2,1] = '@' (player)
#@o #     Grid position [2,2] = 'o' (box)
# x #     Grid position [3,2] = 'x' (target)
#####
```

The agent sees the entire grid with their position marked as a special value:
- **Fixed frame of reference**: North is always up
- **Absolute positions**: Player at (2,1) sees themselves at (2,1)
- **Complete information**: Full map visible at once

### Egocentric: The World from the Agent's View

An egocentric representation centers the world on the agent:

```
Allocentric:        Egocentric (agent-centered):
#####               #?#??
#@o #               #o #?
# x #               # x #
#####               #####
                    @@@@@ (agent always at center)
```

Key differences:
- **Agent-relative coordinates**: Agent is always at (0,0) or center
- **Local view**: May only see nearby cells
- **Relative directions**: "Box is 1 step ahead" not "Box is at (2,2)"

### Comparison for Navigation Tasks

| Aspect | Allocentric | Egocentric |
|--------|-------------|------------|
| **Generalization** | Poor - must relearn for new positions | Good - same policy from any position |
| **Sample Efficiency** | Good for small, fixed maps | Good for local patterns |
| **Policy Complexity** | High - must memorize position→action | Low - learn relative rules |
| **Exploration** | Knows where it's been | May revisit same areas |
| **Transfer Learning** | Limited to similar layouts | Transfers across different maps |

### Hybrid Approaches

Many successful systems combine both:

1. **Local + Global**: Egocentric view for navigation, allocentric map for planning
2. **Attention Mechanisms**: Focus on agent-local region within global context
3. **Memory Systems**: Build allocentric map from egocentric observations

### Case Study: Why Our Sokoban Uses Allocentric

For our curriculum learning demonstration, allocentric makes sense:
- **Small grids** (5×5): Full observability is reasonable
- **Puzzle solving**: Requires understanding global configuration
- **Fixed layouts**: Each curriculum stage has specific patterns to learn

However, this limits generalization. The agent learns "at position (1,1) push right" rather than "when box is adjacent, push toward target."

### When to Use Each

**Choose Allocentric when:**
- Map size is small and fixed
- Global planning is essential
- Training on specific layouts
- Full observability is realistic

**Choose Egocentric when:**
- Generalizing across environments
- Local reactive behaviors suffice
- Maps are large or procedurally generated
- Partial observability is natural

### Implementation Considerations

#### Egocentric Transform
```python
def make_egocentric(grid, agent_pos, view_radius=2):
    """Center the observation on the agent."""
    x, y = agent_pos
    # Extract local window around agent
    local_view = grid[
        max(0, x-view_radius):x+view_radius+1,
        max(0, y-view_radius):y+view_radius+1
    ]
    # Pad if near edges
    # Rotate if considering orientation
    return local_view
```

#### Network Architecture Impact
- **Allocentric**: Can use standard CNNs, position embeddings help
- **Egocentric**: Benefits from rotation-invariant architectures, recurrent memory

### The Orientation Question

We keep orientation allocentric (north is always up) for simplicity:
- Sokoban has no turning actions (only move up/down/left/right)
- Rotation invariance adds complexity without clear benefit
- But for agents that can turn, egocentric orientation is often crucial

{.note title="Exercise 4: Egocentric Sokoban"}
> Implement an egocentric version of the Sokoban environment where:
> 1. The agent is always centered in the observation
> 2. The view shows a 5×5 window around the agent
> 3. Cells outside the map are marked as walls
> 4. Compare learning curves with allocentric version
>
> Does it generalize better to new puzzle layouts?
> See `exercise4.md` for implementation guide.

### Future Directions

Modern approaches increasingly use:
- **Transformers**: Attend to relevant parts regardless of position
- **Graph Networks**: Represent spatial relations explicitly
- **World Models**: Learn to transform between egocentric and allocentric

The choice between egocentric and allocentric isn't binary - it's about finding the right representation for your task's structure and generalization requirements.

***

{pause down=fin}

## References

{#refs-sutton-barto}
Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Skiredj, A. (2025). *The Illustrated GRPO: A Detailed and Pedagogical Explanation of GRPO Algorithm*. OCP Solutions & Mohammed VI Polytechnic University, Morocco.

{#fin}