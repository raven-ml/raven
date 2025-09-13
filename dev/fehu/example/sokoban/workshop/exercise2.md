# Exercise 2: Extending Environment API for GRPO

## Background

Group Relative Policy Optimization (GRPO) requires generating multiple trajectories from the **same initial state** with different policies. The current Gym API doesn't support state persistence - once you call `step()`, the environment state changes irreversibly.

## The Challenge

### Current Limitation
```ocaml
(* Current API - no way to save/restore state *)
let env = create_simple_gridworld 5 in
let state1 = reset env in
let state2, reward, done = step env action in
(* Can't go back to state1! *)
```

### What GRPO Needs
```ocaml
(* GRPO needs to:
   1. Save current state
   2. Generate trajectory with policy A
   3. Restore saved state
   4. Generate trajectory with policy B (reference)
   5. Compare and learn from both *)
```

## Your Task

Extend the environment API to support state persistence. You have two main approaches:

### Option 1: Add save/restore methods
```ocaml
type 'a env = {
  (* existing methods *)
  reset : unit -> 'a;
  step : action -> 'a * float * bool;
  (* new methods for GRPO *)
  save_state : unit -> env_state;
  restore_state : env_state -> unit;
}
```

### Option 2: Make environments immutable
```ocaml
(* Return new environment state instead of mutating *)
type ('obs, 'state) env = {
  reset : unit -> 'obs * 'state;
  step : 'state -> action -> 'obs * float * bool * 'state;
}
```

## Implementation Steps

1. **Modify the environment type** in `slide1.ml`
2. **Update environment creation** to support state management
3. **Implement GRPO algorithm** using the new API:
   ```ocaml
   let train_grpo env n_episodes learning_rate gamma epsilon beta =
     (* Initialize reference and learning policies *)
     let ref_policy_net, ref_params = initialize_policy () in
     let policy_net, params = initialize_policy () in

     for episode = 1 to n_episodes do
       (* Reset and save state *)
       let initial_state = reset env in
       let saved_state = save_state env in

       (* Generate trajectory with learning policy *)
       let learning_trajectory =
         collect_episode_from_state env policy_net params saved_state 100 in

       (* Generate reference trajectories from same state *)
       restore_state env saved_state;
       let ref_trajectory =
         collect_episode_from_state env ref_policy_net ref_params saved_state 100 in

       (* Compute relative advantages and update *)
       (* ... GRPO update logic ... *)
     done
   ```

## Bonus Challenges

1. **Multiple reference policies**: Generate K reference trajectories and use the best/worst for bounds
2. **Adaptive KL penalty**: Adjust beta based on KL divergence magnitude
3. **State distribution**: Sample initial states from a distribution rather than always starting from reset

## Testing Your Implementation

Compare GRPO against REINFORCE++ from slide9:
- Does GRPO converge faster?
- Is it more stable?
- How does the choice of reference policy affect performance?

## Hints

- Look at how other RL libraries (e.g., Stable Baselines3) handle environment copies
- Consider using OCaml's functional programming features - immutable environments might be cleaner
- The gridworld's state is just the agent position - relatively simple to save/restore

## Expected Outcome

A working GRPO implementation that:
1. Generates multiple trajectories from the same state
2. Uses relative advantages for policy updates
3. Demonstrates improved sample efficiency over REINFORCE++