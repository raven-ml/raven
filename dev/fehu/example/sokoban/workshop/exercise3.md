# Exercise 3: Implementing Full PPO with Multiple Updates Per Batch

## Background

You've implemented REINFORCE++ (Slide 9) with clipping and KL penalties, but it still follows the inefficient pattern:
1. Collect one episode
2. Update once
3. Throw away the data
4. Repeat

PPO's key innovation is **reusing collected data through multiple gradient updates**, dramatically improving sample efficiency.

## Your Task

Transform REINFORCE++ into full PPO by implementing batch collection and multiple update epochs.

## Starting Point

You have REINFORCE++ from `slide9.ml` which already includes:
- ✅ Policy ratios (old vs new)
- ✅ Clipped objective
- ✅ KL penalty
- ❌ Batch collection
- ❌ Multiple epochs per batch

## Implementation Steps

### Step 1: Batch Collection

Replace single episode collection with batch collection:

```ocaml
(* Collect multiple episodes before updating *)
let collect_batch env policy_net params batch_size max_steps =
  let episodes = ref [] in
  for i = 1 to batch_size do
    let episode = collect_episode env policy_net params max_steps in
    episodes := episode :: !episodes
  done;
  List.rev !episodes
```

### Step 2: Fixed Old Policy for Multiple Epochs

```ocaml
let train_ppo env n_iterations batch_size n_epochs learning_rate gamma epsilon beta =
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  for iteration = 1 to n_iterations do
    (* Step 1: Collect batch with current policy *)
    let batch = collect_batch env policy_net params batch_size 100 in

    (* Step 2: Fix old policy for all epochs *)
    let old_params = copy_params params in

    (* Step 3: Multiple optimization epochs *)
    for epoch = 1 to n_epochs do
      (* Process each episode in batch *)
      List.iter (fun episode_data ->
        (* Compute gradients using FIXED old_params *)
        let loss, grads = compute_ppo_loss params old_params episode_data gamma epsilon beta in
        (* Update parameters *)
        let updates, new_state = optimizer.update !opt_state params grads in
        opt_state := new_state;
        Kaun.Optimizer.apply_updates_inplace params updates
      ) batch
    done;

    (* Log progress *)
    if iteration mod 5 = 0 then
      log_iteration_stats iteration batch
  done
```

### Step 3: Importance Sampling with Fixed Reference

The key is that `old_params` remains **fixed** during all epochs:

```ocaml
let compute_ppo_loss params old_params episode_data gamma epsilon beta =
  Kaun.value_and_grad (fun p ->
    let total_loss = ref (Rune.scalar device Rune.float32 0.0) in

    (* Process episode states *)
    let returns = compute_returns episode_data.rewards gamma in

    for t = 0 to min 10 (Array.length episode_data.states - 1) do
      (* Get action probabilities from CURRENT params *)
      let new_log_prob = compute_log_prob p episode_data.states.(t) episode_data.actions.(t) in

      (* Get action probabilities from FIXED old_params *)
      let old_log_prob = compute_log_prob old_params episode_data.states.(t) episode_data.actions.(t) in

      (* Importance ratio using FIXED reference *)
      let ratio = Rune.exp (Rune.sub new_log_prob old_log_prob) in

      (* Clipped objective *)
      let clipped_ratio = clip_ratio ratio epsilon in
      let advantage = returns.(t) in
      (* ... rest of loss computation ... *)
    done;
    !total_loss
  ) params
```

## Key Differences from REINFORCE++

| Aspect | REINFORCE++ (Slide 9) | Full PPO (This Exercise) |
|--------|------------------------|--------------------------|
| Data Collection | 1 episode at a time | Batch of episodes |
| Updates per Collection | 1 | n_epochs (typically 3-10) |
| Old Policy Update | Every 3 episodes | Once per batch |
| Sample Efficiency | Low | High |
| Convergence | Slower | Faster |

## Implementation Challenges

### Challenge 1: Memory Management
With batch collection, you're storing multiple episodes in memory. Consider:
- Limiting episode length
- Using efficient data structures
- Clearing old batches

### Challenge 2: Hyperparameter Tuning
New hyperparameters to tune:
- `batch_size`: Start with 8-32 for gridworld
- `n_epochs`: Start with 3-5
- May need to reduce `learning_rate` due to multiple updates

### Challenge 3: Early Stopping
Monitor KL divergence between old and new policies:
```ocaml
(* Stop epochs early if policy changes too much *)
if kl_divergence > max_kl then
  break_epoch_loop ()
```

## Testing Your Implementation

Compare against REINFORCE++:
1. **Sample Efficiency**: PPO should reach same performance with fewer environment steps
2. **Stability**: Multiple epochs shouldn't cause instability (if clipping works)
3. **Final Performance**: Should match or exceed REINFORCE++

## Expected Results

For the 5x5 gridworld:
- REINFORCE++: ~200 episodes to converge
- PPO (batch_size=10, n_epochs=4): ~50-100 episodes (5-10 iterations)
- 2-4x improvement in sample efficiency

## Bonus Challenges

1. **Minibatches**: Split large batches into minibatches for more stable gradients
2. **Advantage Normalization**: Normalize advantages within each batch
3. **Adaptive KL Penalty**: Adjust beta based on KL divergence
4. **GAE (Generalized Advantage Estimation)**: Better advantage estimates

## Hints

- Start with small batches and few epochs, then increase
- Monitor the ratio values - if they're always ≈1, your epochs aren't doing much
- If training becomes unstable, reduce n_epochs or tighten epsilon
- The sweet spot is when clipping activates occasionally but not constantly

## Connection to Previous Work

This exercise combines everything you've learned:
- Episode collection (Slides 2-3)
- Gradient computation (Slide 4)
- Clipping and KL penalties (Slides 7-8)
- REINFORCE++ foundation (Slide 9)

Full PPO is the culmination of these techniques plus the key insight: **don't throw away valuable data after one gradient step!**