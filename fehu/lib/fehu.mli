(** Reinforcement learning library for OCaml.

    Fehu provides a complete toolkit for building and training reinforcement
    learning agents in OCaml. Inspired by OpenAI Gymnasium and
    Stable-Baselines3, it offers environment interfaces, vectorization, training
    utilities, and trajectory management with type-safe APIs and efficient
    tensor operations.

    {1 Fehu in a Nutshell}

    Fehu centers around environments that implement the standard RL loop: reset
    to get initial observations, step with actions to receive transitions, and
    optionally render or collect trajectories. Build custom environments, wrap
    them to modify behavior, vectorize for parallel data collection, and use
    training utilities for advantage estimation and policy optimization.

    Quick start with a built-in environment:
    {[
      open Fehu

      let rng = Rune.Rng.key 42 in
      let env = Fehu_envs.Random_walk.make ~rng () in
      let obs, info = Env.reset env () in
      let action, _ = Space.sample (Env.action_space env) in
      let transition = Env.step env action in
      Printf.printf "Reward: %.2f\n" transition.reward
    ]}

    {1 Core Concepts}

    {2 Environments}

    {!Env} defines the standard environment interface. Environments expose
    observation and action spaces, reset to initial states, and step forward
    with actions. Create custom environments by providing reset and step
    functions, or use pre-built environments from {!Fehu_envs}.

    Environments support:
    - Type-safe observation and action spaces via {!Space}
    - Reproducible randomness with Rune RNG keys
    - Rendering for visualization
    - Metadata describing environment properties

    {2 Spaces}

    {!Space} defines valid observations and actions. Spaces include discrete
    choices ({!Space.Discrete}), continuous vectors ({!Space.Box}), and
    composite structures ({!Space.Tuple}, {!Space.Dict}). Each space provides
    validation, sampling, and serialization.

    {2 Wrappers}

    {!Wrapper} modifies environment behavior without changing core logic. Apply
    wrappers to normalize observations, transform actions, clip rewards, or
    enforce time limits. Wrappers compose through chaining:
    {[
      let env =
        make_env ()
        |> Wrapper.time_limit ~max_episode_steps:1000
        |> Wrapper.map_reward ~f:clip_reward
    ]}

    {2 Vectorization}

    {!Vector_env} enables parallel interaction with multiple environment
    instances, essential for on-policy algorithms requiring large amounts of
    data per update. Vectorized environments batch observations, actions, and
    rewards, amortizing policy inference costs.

    {2 Training Utilities}

    {!Training} provides computational functions for RL algorithms: Generalized
    Advantage Estimation (GAE), policy gradient losses, value losses, and
    evaluation utilities. All functions operate on arrays for batch processing.

    {2 Trajectories}

    {!Trajectory} and {!Buffer} manage experience replay. Trajectories store
    sequences of transitions, while buffers accumulate trajectories for
    off-policy learning or batch updates.

    {1 Usage Patterns}

    {2 Custom Environment}

    Implement environments by defining reset and step logic:
    {[
      let env = Env.create
        ~rng
        ~observation_space:(Space.Box.create ~low:[|-1.0|] ~high:[|1.0|])
        ~action_space:(Space.Discrete.create 3)
        ~reset:(fun env ?options () ->
          let obs = (* initialize state *) in
          (obs, Info.empty))
        ~step:(fun env action ->
          let obs = (* compute next state *) in
          let reward = (* compute reward *) in
          Env.transition ~observation:obs ~reward ())
        ()
    ]}

    {2 Training Loop}

    Collect trajectories, compute advantages, and update policies:
    {[
      let collect_data env policy n_steps =
        let buffer = Buffer.create ~capacity:n_steps in
        let obs = ref (fst (Env.reset env ())) in
        for _ = 1 to n_steps do
          let action = policy !obs in
          let t = Env.step env action in
          Buffer.add buffer { observation = !obs; action; reward = t.reward; (* ... *) };
          obs := t.observation
        done;
        buffer
      in

      let train policy env =
        let data = collect_data env policy 2048 in
        let rewards = Buffer.get_rewards data in
        let values = Buffer.get_values data in
        let dones = Buffer.get_dones data in
        let last_value = 0.0 in
        let last_done = true in
        let advantages, returns = Training.compute_gae
          ~rewards ~values ~dones ~last_value ~last_done ~gamma:0.99
          ~gae_lambda:0.95
        in
        (* update policy with advantages and returns *)
    ]}

    {2 Vectorized Data Collection}

    Use vectorized environments for parallel rollouts:
    {[
      let envs = List.init 8 (fun _ -> make_env ()) in
      let vec_env = Vector_env.create_sync ~envs () in
      let observations, _ = Vector_env.reset vec_env () in
      let actions = Array.map policy observations in
      let step = Vector_env.step vec_env actions
    ]}

    {1 Module Organization}

    Core modules:
    - {!Env}: Environment interface and lifecycle
    - {!Space}: Observation and action space definitions
    - {!Wrapper}: Environment modification and composition
    - {!Vector_env}: Parallel environment vectorization

    Training and data:
    - {!Training}: Advantage estimation, losses, evaluation
    - {!Trajectory}: Experience sequence management
    - {!Buffer}: Trajectory accumulation and sampling

    Utilities:
    - {!Info}: Auxiliary diagnostic data
    - {!Metadata}: Environment metadata and properties
    - {!Errors}: Error types for environment operations

    Environment collection: {!Fehu_envs} *)

module Errors = Errors
(** Error types and exception handling for environments. *)

module Info = Info
(** Auxiliary diagnostic information from environment interactions.

    Info dictionaries carry metadata like episode steps, debug values, or
    environment state. Environments return info from reset and step operations.
*)

module Metadata = Metadata
(** Environment metadata describing properties and capabilities.

    Metadata includes supported render modes, environment version, author
    information, and tags. *)

module Space = Space
(** Observation and action space definitions.

    Spaces specify valid observations and actions, providing validation,
    sampling, and serialization. See {!Space} for discrete, continuous, and
    composite space types. *)

module Env = Env
(** Core environment interface for reinforcement learning.

    Defines the standard RL environment API: reset, step, render, and close. All
    environments implement this interface. See {!Env} for lifecycle management
    and custom environment creation. *)

module Wrapper = Wrapper
(** Environment wrappers for behavior modification.

    Wrappers transform observations, actions, or rewards without changing
    environment implementation. See {!Wrapper} for normalization,
    discretization, and time limits. *)

module Vector_env = Vector_env
(** Vectorized environments for parallel interaction.

    Vectorization enables simultaneous stepping of multiple environments,
    essential for efficient on-policy data collection. See {!Vector_env} for
    batched operations. *)

module Buffer = Buffer
(** Experience replay buffer for trajectory storage.

    Buffers accumulate trajectories for batch training or off-policy learning.
*)

module Training = Training
(** Training utilities for reinforcement learning algorithms.

    Provides advantage estimation (GAE, Monte Carlo returns), policy gradient
    losses, value losses, and evaluation functions. See {!Training} for
    computational tools. *)

module Trajectory = Trajectory
(** Trajectory management for experience sequences.

    Trajectories represent sequences of transitions collected during episodes.
*)
