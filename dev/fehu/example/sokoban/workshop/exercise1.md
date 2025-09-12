# Exercise 1: Fix REINFORCE Loss Computation

## Problem Statement

The current REINFORCE implementation in `slide4.ml` has a critical limitation: it only processes the first 10 states of each episode (line 34). This severely impacts learning efficiency because:

1. **Lost gradients**: States after position 10 receive no policy updates
2. **Missed crucial states**: States near the goal (the most informative ones!) are often ignored
3. **Suboptimal sample efficiency**: We collect 100 states but only use 10

## The Root Cause

The limitation exists because of an "index out of bounds" error that occurs during automatic differentiation when we try to select specific action log probabilities. The following approaches all fail during backward pass:

```ocaml
(* Approach 1: Direct indexing - FAILS *)
let action_log_prob = log_probs.(action_idx)

(* Approach 2: Slice operation - FAILS *)
let action_log_prob = Rune.slice [Rune.I 0; Rune.I action_idx] log_probs

(* Approach 3: take_along_axis - FAILS *)
let action_expanded = Rune.reshape [|1; 1|] action_idx in
let action_log_prob = Rune.take_along_axis ~axis:(-1) action_expanded log_probs
```

The error manifests as:
```
Fatal error: exception Invalid_argument("index out of bounds")
Raised at Rune__Autodiff.make_reverse_handler in file "rune/lib/autodiff.ml"
```

## Current Workaround

We use a mask-based approach that works but is inefficient:

```ocaml
let mask = Rune.init device Rune.float32 [|1; 4|] (fun idxs ->
  if idxs.(1) = action_int then 1.0 else 0.0
) in
let action_log_prob = Rune.sum (Rune.mul mask log_probs)
```

This works for 10 states but becomes problematic for full episodes.

## Your Challenge

### Option A: Debug the Root Cause (Advanced)
1. Investigate why indexing operations fail in `rune/lib/autodiff.ml`
2. Determine if this is a bug in Rune's autodiff implementation
3. Propose or implement a fix in the Rune library itself

### Option B: Implement a Workaround (Intermediate)
Create a working solution that processes ALL states in an episode. Consider:

1. **Batching approach**: Process all states in a single forward pass
   ```ocaml
   (* Hint: Stack all states into shape [batch_size; 5; 5] *)
   let all_states = Rune.stack (Array.to_list episode_data.states) ~axis:0 in
   (* Compute all log_probs at once *)
   let all_logits = Kaun.apply policy_net p ~training:true all_states in
   ```

2. **Alternative loss formulation**: Use cross-entropy with one-hot encoded actions
   ```ocaml
   (* Create one-hot matrix for all actions at once *)
   let actions_one_hot = (* your implementation *) in
   let loss = cross_entropy_loss all_logits actions_one_hot returns
   ```

3. **Gradient accumulation**: Compute gradients outside the main loop
   ```ocaml
   (* Process in smaller batches if memory is an issue *)
   let batch_size = 32 in
   for batch_start = 0 to Array.length states - 1 step batch_size do
     (* Process batch and accumulate gradients *)
   done
   ```

## Success Criteria

Your solution should:
1. Process ALL states in an episode (not just the first 10)
2. Maintain correct REINFORCE gradient computation: ∇log π(a|s) * G_t
3. Work without "index out of bounds" errors during backpropagation
4. Show improved learning efficiency (faster convergence, better final performance)

## Testing Your Solution

1. Replace the loss computation in `slide4.ml` (lines 29-55)
2. Run the training and verify:
   ```bash
   dune exec dev/fehu/example/sokoban/workshop/run4.exe
   ```
3. Compare learning curves with the original (episodes to reach consistent goal-finding)
4. Verify that later states in trajectories are actually being updated

## Bonus Points

- Benchmark the performance difference between processing 10 vs all states
- Implement multiple approaches and compare their efficiency
- Add unit tests for your gradient computation
- Create a visualization showing which states receive gradient updates

## Resources

- Rune autodiff implementation: `rune/lib/autodiff.ml`
- PyTorch gather operation (for comparison): https://pytorch.org/docs/stable/generated/torch.gather.html
- REINFORCE algorithm: Sutton & Barto Chapter 13
- Original paper: Williams (1992) "Simple Statistical Gradient-Following Algorithms"

## Hints

1. The issue might be related to how Rune handles dynamic indexing in computation graphs
2. Consider that the mask approach works because it uses static operations
3. Look at how other RL libraries (CleanRL, Stable-Baselines3) handle action selection in policy gradients
4. The `Kaun.Loss.cross_entropy` function might provide inspiration

Good luck! This is a real issue affecting the learning efficiency of our RL agent.