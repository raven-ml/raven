# Raven Framework: Deep Overview

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Nx: Tensor Operations Foundation](#nx-tensor-operations-foundation)
4. [Rune: Autodiff and JIT Compilation](#rune-autodiff-and-jit-compilation)
5. [Kaun: Neural Network Framework](#kaun-neural-network-framework)
6. [Fehu: Reinforcement Learning](#fehu-reinforcement-learning)
7. [Package Relationships](#package-relationships)
8. [Complete Examples](#complete-examples)

## Introduction

Raven is a comprehensive machine learning ecosystem for OCaml, currently in pre-alpha development. It provides a NumPy-style tensor library, automatic differentiation, neural network abstractions, and reinforcement learning tools—all with type safety and functional programming principles at its core.

The framework consists of several interconnected packages:
- **Nx**: Low-level tensor operations (NumPy-like API)
- **Rune**: Automatic differentiation and JIT compilation layer
- **Kaun**: High-level neural network framework (JAX/Flax-inspired)
- **Fehu**: Reinforcement learning environments and algorithms

## Architecture Overview

```
┌─────────────────────────────────────────┐
│            Fehu (RL)                    │
│  Environments, Training, Trajectories   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           Kaun (Neural Networks)        │
│  Layers, Optimizers, Modules, Loss      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          Rune (Autodiff + JIT)          │
│  Gradients, Device Abstraction          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           Nx (Tensor Operations)        │
│  Core tensor ops, Broadcasting, Slicing │
└─────────────────────────────────────────┘
```

## Nx: Tensor Operations Foundation

Nx provides the fundamental tensor operations layer, implementing a NumPy-like API with OCaml's type safety.

### Core Types

```ocaml
type ('a, 'b) t  (* Tensor type: 'a is OCaml type, 'b is element type *)
type ('a, 'b) dtype  (* Data type specification *)

(* Common tensor types *)
type float32_t = (float, float32_elt) t
type int32_t = (int32, int32_elt) t
```

### Key Features

- **NumPy Compatibility**: Follows NumPy conventions for slicing, broadcasting, and dimension handling
- **Efficient Memory Layout**: Supports both contiguous and strided tensors
- **Zero-copy Operations**: View operations (reshape, slice, transpose) don't copy data when possible
- **Broadcasting**: Automatic shape compatibility for element-wise operations

### Basic Example

```ocaml
open Nx

let example_tensor_ops () =
  (* Create tensors *)
  let a = create float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  let b = ones float32 [|2; 3|] in
  
  (* Element-wise operations with broadcasting *)
  let c = add a b in
  
  (* Slicing (Python-style, exclusive end) *)
  let row = slice [R [0; 1]; All] a in  (* First row *)
  
  (* Reduction operations *)
  let sum = sum ~axis:[|1|] a in  (* Sum along axis 1 *)
  
  (* Reshaping (returns view when possible) *)
  let flat = reshape [|6|] a in
  
  print c;
  print row;
  print sum;
  print flat
```

### Important Conventions

- **Slicing**: `R [1; 4]` selects indices 1, 2, 3 (exclusive end like Python)
- **Single indexing**: Squeezes dimensions (e.g., `get [0]` on shape `[2; 3]` returns shape `[3]`)
- **Comparison ops**: Return `uint8` tensors (0 or 1) due to Bigarray limitations

## Rune: Autodiff and JIT Compilation

Rune builds on Nx to provide automatic differentiation and device abstraction, enabling gradient-based optimization.

### Core Types

```ocaml
type ('a, 'b, 'dev) t  (* Tensor with device parameter *)
type 'dev device = [ `c | `cuda | `metal | ... ]  (* Device types *)

(* Device-aware tensor types *)
type 'dev float32_t = (float, float32_elt, 'dev) t
```

### Key Features

- **Effect-based Autodiff**: Tracks operations for gradient computation
- **Device Abstraction**: Unified interface for CPU, CUDA, Metal backends
- **JIT Compilation**: Optimizes computation graphs (via backend)
- **Random Number Generation**: Device-aware RNG with key-based splitting

### Gradient Computation Example

```ocaml
open Rune

let gradient_example () =
  let device = c in  (* CPU device *)
  
  (* Define a simple loss function *)
  let loss_fn x =
    let y = mul x x in  (* x² *)
    sum y
  in
  
  (* Compute gradient *)
  let x = create device float32 [|3|] [|1.; 2.; 3.|] in
  let grad_fn = grad loss_fn in
  let gradient = grad_fn x in
  
  (* gradient will be [2.; 4.; 6.] (derivative of x² is 2x) *)
  print gradient
```

### Random Number Generation

```ocaml
let rng_example () =
  let rng = Rng.key 42 in
  let keys = Rng.split ~n:2 rng in
  
  (* Generate random tensors *)
  let normal = randn keys.(0) c float32 [|100|] in
  let uniform = rand keys.(1) c float32 [|10; 10|] in
  
  print normal;
  print uniform
```

## Kaun: Neural Network Framework

Kaun provides high-level neural network abstractions inspired by JAX/Flax, with functional parameter management.

### Core Types

```ocaml
type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
type ('layout, 'dev) params =  (* Parameter tree structure *)
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) params list
  | Record of ('layout, 'dev) params Record.t

type module_ = {
  init : 'layout 'dev. 
    rngs:Rune.Rng.key -> 
    device:'dev Rune.device ->
    dtype:(float, 'layout) Rune.dtype ->
    ('layout, 'dev) params;
  apply : 'layout 'dev.
    ('layout, 'dev) params ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ('layout, 'dev) tensor ->
    ('layout, 'dev) tensor;
}
```

### Key Components

- **Modules**: Composable neural network components
- **Layers**: Pre-built layers (linear, conv2d, LSTM, etc.)
- **Optimizers**: Adam, SGD, RMSprop with Optax-style API
- **Loss Functions**: Cross-entropy, MSE, etc.
- **Initializers**: Xavier, He, normal, uniform

### Neural Network Example

```ocaml
open Kaun

let mlp_example () =
  let device = Rune.c in
  let rng = Rune.Rng.key 42 in
  
  (* Define model architecture *)
  let model = Layer.sequential [
    Layer.linear ~in_features:784 ~out_features:128 ();
    Layer.relu ();
    Layer.dropout ~rate:0.2 ();
    Layer.linear ~in_features:128 ~out_features:10 ();
    Layer.softmax ~axis:(-1) ()
  ] in
  
  (* Initialize parameters *)
  let dummy_input = Rune.zeros device Rune.float32 [|1; 784|] in
  let params = init model ~rngs:rng dummy_input in
  
  (* Create optimizer *)
  let optimizer = Optimizer.adam ~lr:0.001 () in
  let opt_state = ref (optimizer.init params) in
  
  (* Training step *)
  let train_step params x y =
    let loss, grads = value_and_grad (fun p ->
      let logits = apply model p ~training:true x in
      Loss.cross_entropy logits y
    ) params in
    
    (* Update parameters *)
    let updates, new_state = optimizer.update !opt_state params grads in
    opt_state := new_state;
    Optimizer.apply_updates_inplace params updates;
    loss
  in
  
  (* Inference *)
  let predict params x =
    apply model params ~training:false x
  in
  
  (train_step, predict)
```

### Custom Module Example

```ocaml
let custom_block ~features () = {
  Module.init = (fun ~rngs x ->
    let keys = Rune.Rng.split ~n:2 rngs in
    let in_features = (Rune.shape x).(Array.length (Rune.shape x) - 1) in
    
    (* Initialize two linear layers *)
    let linear1 = Layer.linear ~in_features ~out_features:features () in
    let linear2 = Layer.linear ~in_features:features ~out_features:features () in
    
    Record [
      ("linear1", init linear1 ~rngs:keys.(0) x);
      ("linear2", init linear2 ~rngs:keys.(1) (Rune.zeros_like x))
    ]
  );
  
  Module.apply = (fun params ~training ?rngs:_ x ->
    let Record fields = params in
    let linear1_params = List.assoc "linear1" fields in
    let linear2_params = List.assoc "linear2" fields in
    
    (* Residual connection *)
    let h = apply (Layer.linear ~in_features:0 ~out_features:features ()) 
              linear1_params ~training x in
    let h = Rune.relu h in
    let h = apply (Layer.linear ~in_features:features ~out_features:features ()) 
              linear2_params ~training h in
    Rune.add x h  (* Skip connection *)
  )
}
```

## Fehu: Reinforcement Learning

Fehu provides RL-specific components that compose with Kaun for neural network policies and value functions.

### Core Types

```ocaml
(* Action/observation spaces *)
type 'dev Space.t =
  | Discrete of int
  | Box of { low: tensor; high: tensor; shape: int array }
  | Multi_discrete of int array

(* Environment interface *)
type 'dev Env.t = {
  observation_space : 'dev Space.t;
  action_space : 'dev Space.t;
  reset : ?seed:int -> unit -> tensor * info;
  step : tensor -> tensor * float * bool * bool * info;
  render : unit -> unit;
  close : unit -> unit;
}

(* Trajectory storage *)
type 'dev Trajectory.t = {
  states : tensor array;
  actions : tensor array;
  rewards : float array;
  log_probs : float array option;
  values : float array option;
  dones : bool array;
}
```

### Key Components

- **Environments**: Standard RL environment interface (OpenAI Gym-like)
- **Spaces**: Action and observation space definitions
- **Buffers**: Experience replay for off-policy algorithms
- **Training Utilities**: GAE, returns computation, normalization
- **Trajectories**: Episode data collection and management
- **Curriculum Learning**: Progressive difficulty scheduling
- **Visualization**: Trajectory logging and visualization

### Complete RL Example: REINFORCE with Baseline

```ocaml
open Fehu
open Kaun

let reinforce_cartpole () =
  let device = Rune.c in
  let rng = Rune.Rng.key 42 in
  
  (* Create environment *)
  let env = Envs.cartpole () in
  
  (* Define policy network *)
  let policy_net = Layer.sequential [
    Layer.linear ~in_features:4 ~out_features:64 ();
    Layer.relu ();
    Layer.linear ~in_features:64 ~out_features:2 ();  (* 2 actions *)
  ] in
  
  (* Define value network (baseline) *)
  let value_net = Layer.sequential [
    Layer.linear ~in_features:4 ~out_features:64 ();
    Layer.relu ();
    Layer.linear ~in_features:64 ~out_features:1 ();
  ] in
  
  (* Initialize parameters *)
  let dummy_obs = Rune.zeros device Rune.float32 [|4|] in
  let policy_params = init policy_net ~rngs:(Rune.Rng.split ~n:1 rng).(0) dummy_obs in
  let value_params = init value_net ~rngs:(Rune.Rng.split ~n:1 rng).(0) dummy_obs in
  
  (* Optimizers *)
  let policy_opt = Optimizer.adam ~lr:0.001 () in
  let value_opt = Optimizer.adam ~lr:0.001 () in
  let policy_opt_state = ref (policy_opt.init policy_params) in
  let value_opt_state = ref (value_opt.init value_params) in
  
  (* Collect trajectory *)
  let collect_trajectory max_steps =
    let obs, _ = env.reset () in
    let states = ref [obs] in
    let actions = ref [] in
    let rewards = ref [] in
    let log_probs = ref [] in
    let values = ref [] in
    
    let rec loop obs step =
      if step >= max_steps then ()
      else
        (* Get action from policy *)
        let logits = apply policy_net policy_params ~training:false obs in
        let probs = Rune.softmax ~axis:(-1) logits in
        let action = (* Sample from categorical distribution *) 
          Rune.categorical ~rng probs in
        
        (* Get value estimate *)
        let value = apply value_net value_params ~training:false obs in
        
        (* Take environment step *)
        let next_obs, reward, terminated, truncated, _ = env.step action in
        
        (* Store transition *)
        states := next_obs :: !states;
        actions := action :: !actions;
        rewards := reward :: !rewards;
        log_probs := Rune.log (Rune.gather probs action) :: !log_probs;
        values := Rune.unsafe_get [] value :: !values;
        
        if terminated || truncated then ()
        else loop next_obs (step + 1)
    in
    
    loop obs 0;
    
    Trajectory.create
      ~states:(Array.of_list (List.rev !states))
      ~actions:(Array.of_list (List.rev !actions))
      ~rewards:(Array.of_list (List.rev !rewards))
      ~log_probs:(Some (Array.of_list (List.rev !log_probs)))
      ~values:(Some (Array.of_list (List.rev !values)))
      ()
  in
  
  (* Training loop *)
  let train_episode trajectory =
    (* Compute returns and advantages *)
    let returns = Training.compute_returns 
      ~rewards:trajectory.rewards 
      ~dones:trajectory.dones 
      ~gamma:0.99 in
    
    let advantages, _ = Training.compute_advantages
      ~rewards:trajectory.rewards
      ~values:(Option.get trajectory.values)
      ~gamma:0.99 in
    
    (* Update policy *)
    let policy_loss, policy_grads = value_and_grad (fun params ->
      (* Recompute log probs with current params *)
      let log_probs = Array.map2 (fun state action ->
        let logits = apply policy_net params ~training:true state in
        let probs = Rune.softmax ~axis:(-1) logits in
        Rune.log (Rune.gather probs action)
      ) trajectory.states trajectory.actions in
      
      (* Policy gradient loss *)
      Training.compute_policy_loss
        ~log_probs:(Array.map (Rune.unsafe_get []) log_probs)
        ~advantages
        ~normalize_advantages:true
    ) policy_params in
    
    (* Update value function *)
    let value_loss, value_grads = value_and_grad (fun params ->
      let predicted_values = Array.map (fun state ->
        apply value_net params ~training:true state
      ) trajectory.states in
      
      (* MSE loss *)
      let returns_tensor = Rune.create device Rune.float32 
        [|Array.length returns|] returns in
      let predicted = Rune.stack predicted_values in
      Loss.mse predicted returns_tensor
    ) value_params in
    
    (* Apply gradients *)
    let policy_updates, new_policy_state = 
      policy_opt.update !policy_opt_state policy_params policy_grads in
    policy_opt_state := new_policy_state;
    Optimizer.apply_updates_inplace policy_params policy_updates;
    
    let value_updates, new_value_state = 
      value_opt.update !value_opt_state value_params value_grads in
    value_opt_state := new_value_state;
    Optimizer.apply_updates_inplace value_params value_updates;
    
    (policy_loss, value_loss)
  in
  
  (* Main training loop *)
  for episode = 1 to 1000 do
    let trajectory = collect_trajectory 500 in
    let policy_loss, value_loss = train_episode trajectory in
    
    if episode mod 10 = 0 then
      Printf.printf "Episode %d: Policy Loss = %.4f, Value Loss = %.4f, Reward = %.1f\n"
        episode 
        (Rune.unsafe_get [] policy_loss)
        (Rune.unsafe_get [] value_loss)
        (Array.fold_left (+.) 0. trajectory.rewards)
  done;
  
  (* Evaluation *)
  let stats = Training.evaluate env 
    ~policy:(fun obs -> 
      let logits = apply policy_net policy_params ~training:false obs in
      Rune.argmax ~axis:(-1) logits)
    ~n_eval_episodes:10 in
  
  Printf.printf "Final evaluation: Mean reward = %.2f ± %.2f\n" 
    stats.mean_reward stats.std_reward
```

## Package Relationships

### Data Flow

1. **Nx → Rune**: Rune extends Nx tensors with device abstraction and autodiff
2. **Rune → Kaun**: Kaun uses Rune tensors for all neural network operations
3. **Kaun → Fehu**: Fehu uses Kaun modules for policies and value functions

### Design Philosophy

- **Nx**: Pure tensor operations, no ML-specific concepts
- **Rune**: Adds differentiation and device management
- **Kaun**: High-level neural network abstractions
- **Fehu**: Domain-specific RL components

### Composition Example

```ocaml
(* Using all layers together *)
let complete_example () =
  (* Nx: Basic tensor operations *)
  let data = Nx.create Nx.float32 [|100; 10|] (Array.init 1000 float_of_int) in
  
  (* Rune: Move to device and enable gradients *)
  let device = Rune.c in
  let x = Rune.of_nx device data in
  
  (* Kaun: Define and train neural network *)
  let model = Kaun.Layer.sequential [
    Kaun.Layer.linear ~in_features:10 ~out_features:20 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:20 ~out_features:1 ()
  ] in
  
  (* Fehu: Use in RL context *)
  let env = Fehu.Envs.cartpole () in
  let trajectory = (* collect experience using trained model *) in
  ()
```

## Complete Examples

### Example 1: Autoencoder with Kaun

```ocaml
open Kaun
open Rune

let autoencoder_example () =
  let device = c in
  let rng = Rng.key 42 in
  
  (* Define encoder *)
  let encoder = Layer.sequential [
    Layer.linear ~in_features:784 ~out_features:256 ();
    Layer.relu ();
    Layer.linear ~in_features:256 ~out_features:64 ();
    Layer.relu ();
    Layer.linear ~in_features:64 ~out_features:32 ();  (* Latent space *)
  ] in
  
  (* Define decoder *)
  let decoder = Layer.sequential [
    Layer.linear ~in_features:32 ~out_features:64 ();
    Layer.relu ();
    Layer.linear ~in_features:64 ~out_features:256 ();
    Layer.relu ();
    Layer.linear ~in_features:256 ~out_features:784 ();
    Layer.sigmoid ();  (* Output in [0, 1] *)
  ] in
  
  (* Combine into autoencoder *)
  let autoencoder = {
    Module.init = (fun ~rngs x ->
      let keys = Rng.split ~n:2 rngs in
      Record [
        ("encoder", init encoder ~rngs:keys.(0) x);
        ("decoder", init decoder ~rngs:keys.(1) 
          (zeros device float32 [|Rune.shape x |> Array.get 0; 32|]))
      ]
    );
    
    Module.apply = (fun params ~training ?rngs:_ x ->
      let Record fields = params in
      let encoded = apply encoder (List.assoc "encoder" fields) ~training x in
      apply decoder (List.assoc "decoder" fields) ~training encoded
    )
  } in
  
  (* Training setup *)
  let dummy_input = randn rng device float32 [|32; 784|] in
  let params = init autoencoder ~rngs:rng dummy_input in
  let optimizer = Optimizer.adam ~lr:0.001 () in
  let opt_state = ref (optimizer.init params) in
  
  (* Training step *)
  let train_step x =
    let loss, grads = value_and_grad (fun p ->
      let reconstructed = apply autoencoder p ~training:true x in
      Loss.mse reconstructed x  (* Reconstruction loss *)
    ) params in
    
    let updates, new_state = optimizer.update !opt_state params grads in
    opt_state := new_state;
    Optimizer.apply_updates_inplace params updates;
    loss
  in
  
  train_step
```

### Example 2: Custom RL Environment with Fehu

```ocaml
open Fehu

let custom_grid_world () =
  (* Define a simple grid world environment *)
  let grid_size = 5 in
  let device = Rune.c in
  
  (* Mutable state *)
  let agent_pos = ref (0, 0) in
  let goal_pos = (4, 4) in
  
  let observation_space = Space.Box {
    low = Rune.zeros device Rune.float32 [|grid_size; grid_size|];
    high = Rune.ones device Rune.float32 [|grid_size; grid_size|];
    shape = [|grid_size; grid_size|];
  } in
  
  let action_space = Space.Discrete 4 in  (* Up, Down, Left, Right *)
  
  let reset ?seed () =
    (* Initialize RNG if seed provided *)
    let _ = Option.map (fun s -> Random.init s) seed in
    agent_pos := (0, 0);
    
    (* Create observation (grid with agent position marked) *)
    let obs = Rune.zeros device Rune.float32 [|grid_size; grid_size|] in
    let x, y = !agent_pos in
    Rune.unsafe_set [|x; y|] 1.0 obs;
    
    (obs, [("agent_start", `List [`Int x; `Int y])])
  in
  
  let step action =
    let action_idx = Rune.unsafe_get [] action |> int_of_float in
    let x, y = !agent_pos in
    
    (* Update position based on action *)
    let new_pos = match action_idx with
      | 0 -> (x, max 0 (y - 1))          (* Up *)
      | 1 -> (x, min (grid_size-1) (y + 1))  (* Down *)
      | 2 -> (max 0 (x - 1), y)          (* Left *)
      | 3 -> (min (grid_size-1) (x + 1), y)  (* Right *)
      | _ -> (x, y)
    in
    
    agent_pos := new_pos;
    
    (* Create new observation *)
    let obs = Rune.zeros device Rune.float32 [|grid_size; grid_size|] in
    let x, y = !agent_pos in
    Rune.unsafe_set [|x; y|] 1.0 obs;
    
    (* Calculate reward *)
    let reward = if new_pos = goal_pos then 1.0 else -0.01 in
    let terminated = new_pos = goal_pos in
    let truncated = false in
    
    let info = [
      ("agent_pos", `List [`Int x; `Int y]);
      ("distance_to_goal", `Float (
        float_of_int (abs (fst goal_pos - x) + abs (snd goal_pos - y))
      ))
    ] in
    
    (obs, reward, terminated, truncated, info)
  in
  
  Env.make
    ~observation_space
    ~action_space
    ~reset
    ~step
    ~render:(fun () -> 
      let x, y = !agent_pos in
      Printf.printf "Agent at (%d, %d)\n" x y)
    ~close:(fun () -> ())
    ()
```

### Example 3: Curriculum Learning with Fehu

```ocaml
open Fehu

let curriculum_learning_example () =
  (* Create environments of increasing difficulty *)
  let easy_env = custom_grid_world ~size:3 () in
  let medium_env = custom_grid_world ~size:5 () in
  let hard_env = custom_grid_world ~size:7 () in
  
  (* Define advancement criterion *)
  let advance_criterion stats =
    (* Advance when mean reward > 0.8 over last 100 episodes *)
    stats.mean_reward > 0.8 && stats.n_episodes >= 100
  in
  
  (* Create curriculum *)
  let curriculum = Curriculum.create
    ~stages:[|easy_env; medium_env; hard_env|]
    ~advance_criterion
    ~window_size:100
    () in
  
  (* Training loop with curriculum *)
  let train_with_curriculum agent =
    for episode = 1 to 1000 do
      (* Get current environment from curriculum *)
      let env = Curriculum.current_env curriculum in
      
      (* Run episode *)
      let trajectory = collect_episode env agent in
      let episode_reward = Array.fold_left (+.) 0. trajectory.rewards in
      
      (* Update curriculum statistics *)
      let stats = {
        Training.episode_reward;
        episode_length = Array.length trajectory.rewards;
        total_timesteps = episode * 200;  (* Approximate *)
        n_episodes = episode;
        mean_reward = episode_reward;  (* Would track running average *)
        std_reward = 0.0;  (* Would track running std *)
      } in
      
      Curriculum.update_stats curriculum stats;
      
      (* Try to advance to next stage *)
      if Curriculum.try_advance curriculum then
        Printf.printf "Advanced to next curriculum stage at episode %d!\n" episode;
      
      (* Train agent on trajectory *)
      train_agent agent trajectory
    done
  in
  
  train_with_curriculum
```

## Best Practices

### Memory Management
- Use `contiguous` when you need guaranteed contiguous memory
- Check `is_contiguous` before operations requiring contiguous layout
- Prefer in-place operations (`apply_updates_inplace`) for large models

### Type Safety
- Use phantom types for device tracking: `'dev` parameter ensures tensors stay on same device
- Pattern match on `dtype` with locally abstract types for GADT operations
- Leverage OCaml's type system to catch shape mismatches at compile time

### Performance
- Batch operations to minimize backend dispatch overhead
- Use `Bigarray.unsafe_get/set` in tight loops (validate indices outside)
- Profile with `dune build @bench` to identify bottlenecks

### Debugging
- Create minimal reproductions in `test/failing/` when bugs are found
- Use `Rune.print` for tensor inspection during development
- Check intermediate shapes and values when debugging shape errors

## Future Directions

The Raven framework is actively evolving. Current development focuses on:

1. **Backend Expansion**: Adding Metal, CUDA, and WebGPU support
2. **Model Zoo**: Pre-trained models and standard architectures
3. **Distributed Training**: Multi-device and multi-node support
4. **Compiler Optimizations**: Enhanced JIT compilation and kernel fusion
5. **RL Algorithms**: Expanding the suite of implemented algorithms (PPO, SAC, etc.)

## Conclusion

The Raven framework brings the power of modern deep learning to OCaml with a clean, functional API. By layering abstractions from low-level tensor operations (Nx) through automatic differentiation (Rune) and neural networks (Kaun) to reinforcement learning (Fehu), it provides a complete ecosystem for machine learning research and applications.

The framework's design emphasizes:
- **Type Safety**: Leveraging OCaml's type system to catch errors at compile time
- **Functional Purity**: Tensors with explicit parameter management
- **Composability**: Small, focused modules that combine into complex systems
- **Performance**: Zero-copy operations and backend optimization where possible

Whether you're implementing a simple neural network or a complex RL agent, Raven provides the tools needed while maintaining the safety and elegance of functional programming.