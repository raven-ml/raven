(** Built-in reinforcement learning environments.

    This module provides a collection of ready-to-use environments for testing
    algorithms, learning the Fehu API, and benchmarking. All environments follow
    the standard {!Fehu.Env} interface and are fully compatible with wrappers,
    vectorization, and training utilities.

    {1 Available Environments}

    - {!Random_walk}: One-dimensional random walk with continuous state space
    - {!Grid_world}: Two-dimensional grid navigation with discrete states and
      obstacles
    - {!Cartpole}: Classic cart-pole balancing problem
    - {!Mountain_car}: Drive up a steep hill using momentum

    {1 Usage}

    Create an environment with a Rune RNG key:
    {[
      let rng = Rune.Rng.create () in
      let env = Fehu_envs.Random_walk.make ~rng () in
      let obs, info = Fehu.Env.reset env ()
    ]}

    Environments support rendering for visualization:
    {[
      let env = Fehu_envs.Grid_world.make ~rng () in
      let obs, _ = Fehu.Env.reset env () in
      match Fehu.Env.render env with
      | Some output -> print_endline output
      | None -> ()
    ]}

    {1 Environment Selection Guide}

    Use {!Random_walk} for:
    - Testing continuous observation spaces
    - Debugging value-based algorithms
    - Quick prototyping with minimal complexity

    Use {!Grid_world} for:
    - Learning discrete state/action navigation
    - Testing path planning or exploration strategies
    - Demonstrating obstacle avoidance *)

module Random_walk = Random_walk
(** One-dimensional random walk environment.

    {b ID}: [RandomWalk-v0]

    {b Observation Space}: {!Fehu.Space.Box} with shape [[1]] in range \[-10.0,
    10.0\]. Represents the agent's continuous position on a line.

    {b Action Space}: {!Fehu.Space.Discrete} with 2 choices:
    - [0]: Move left (position -= 1.0)
    - [1]: Move right (position += 1.0)

    {b Rewards}: Negative absolute position ([-|position|]), encouraging the
    agent to stay near the origin. Terminal states at boundaries yield reward
    -10.0.

    {b Episode Termination}:
    - Terminated: Agent reaches position -10.0 or +10.0 (boundaries)
    - Truncated: Episode exceeds 200 steps

    {b Rendering}: ASCII visualization showing agent position ('o') on a line.

    {4 Example}

    Train a simple policy to stay near the origin:
    {[
      let rng = Rune.Rng.create () in
      let env = Fehu_envs.Random_walk.make ~rng () in
      let obs, _ = Fehu.Env.reset env () in
      for _ = 1 to 100 do
        let action = (* policy chooses 0 or 1 *) in
        let t = Fehu.Env.step env action in
        Printf.printf "Position: %.2f, Reward: %.2f\n"
          (Rune.to_array t.observation).(0) t.reward
      done
    ]}

    {4 Tips}

    - The environment is deterministic given the action sequence
    - Optimal policy alternates actions to minimize distance from origin
    - Good for testing value function approximation with continuous states *)

module Grid_world = Grid_world
(** Two-dimensional grid world with goal and obstacles.

    {b ID}: [GridWorld-v0]

    {b Observation Space}: {!Fehu.Space.Multi_discrete} with shape [[5; 5]].
    Represents the agent's (row, column) position as two discrete coordinates,
    each in range \[0, 5).

    {b Action Space}: {!Fehu.Space.Discrete} with 4 choices:
    - [0]: Move up (row -= 1)
    - [1]: Move down (row += 1)
    - [2]: Move left (col -= 1)
    - [3]: Move right (col += 1)

    {b Rewards}:
    - [+10.0]: Reaching the goal at position (4, 4)
    - [-1.0]: Every other step (encourages shortest path)

    {b Episode Termination}:
    - Terminated: Agent reaches the goal position (4, 4)
    - Truncated: Episode exceeds 200 steps

    {b Obstacles}: Position (2, 2) is blocked. Actions attempting to move into
    obstacles or outside the grid leave the agent's position unchanged.

    {b Rendering}: ASCII grid visualization:
    - 'A': Agent position
    - 'G': Goal position (4, 4)
    - '#': Obstacle at (2, 2)
    - '.': Empty cells

    {4 Example}

    Navigate to the goal while avoiding obstacles:
    {[
      let rng = Rune.Rng.create () in
      let env = Fehu_envs.Grid_world.make ~rng () in
      let obs, _ = Fehu.Env.reset env () in
      let rec run_episode steps =
        if steps >= 200 then ()
        else begin
          let action = (* policy maps (row, col) to action 0-3 *) in
          let t = Fehu.Env.step env action in
          match Fehu.Env.render env with
          | Some grid -> print_endline grid
          | None -> ();
          if t.terminated then
            Printf.printf "Goal reached in %d steps!\n" steps
          else
            run_episode (steps + 1)
        end
      in
      run_episode 0
    ]}

    {4 Tips}

    - Optimal policy requires approximately 8 steps (Manhattan distance from
      (0,0) to (4,4))
    - Obstacle at (2, 2) forces agents to plan around it
    - Good for testing Q-learning, DQN, or policy gradient methods on discrete
      spaces *)

module Cartpole = Cartpole
(** Classic cart-pole balancing environment.

    {b ID}: [CartPole-v1]

    {b Observation Space}: {!Fehu.Space.Box} with shape [[4]] in range:
    - Position: \[-4.8, 4.8\]
    - Velocity: \[-∞, ∞\]
    - Angle: \[~-24°, ~24°\]
    - Angular velocity: \[-∞, ∞\]

    {b Action Space}: {!Fehu.Space.Discrete} with 2 choices:
    - [0]: Push cart to the left
    - [1]: Push cart to the right

    {b Rewards}: +1.0 for each step the pole remains upright

    {b Episode Termination}:
    - Terminated: Pole angle exceeds ±12° or cart position exceeds ±2.4
    - Truncated: Episode reaches 500 steps (considered solved if average reward
      ≥ 475 over 100 consecutive episodes)

    {b Rendering}: Text output showing cart position, velocity, pole angle, and
    angular velocity

    {4 Example}

    Train an agent to balance the pole:
    {[
      let rng = Rune.Rng.create () in
      let env = Fehu_envs.Cartpole.make ~rng () in
      let obs, _ = Fehu.Env.reset env () in
      let rec run_episode total_reward =
        let action = (* DQN or policy gradient decision *) in
        let t = Fehu.Env.step env action in
        let new_total = total_reward +. t.reward in
        if t.terminated || t.truncated then
          Printf.printf "Episode reward: %.0f\n" new_total
        else
          run_episode new_total
      in
      run_episode 0.0
    ]}

    {4 Tips}

    - One of the most popular RL benchmarks, considered solved at 475/500
      average reward
    - Good for testing DQN, REINFORCE, A2C, and PPO algorithms
    - Requires learning to balance competing objectives (position and angle)
    - Observation space is continuous, making it ideal for neural network
      policies *)

module Mountain_car = Mountain_car
(** Mountain car environment - drive up a steep hill using momentum.

    {b ID}: [MountainCar-v0]

    {b Observation Space}: {!Fehu.Space.Box} with shape [[2]]:
    - Position: \[-1.2, 0.6\] (goal at 0.5)
    - Velocity: \[-0.07, 0.07\]

    {b Action Space}: {!Fehu.Space.Discrete} with 3 choices:
    - [0]: Push left (accelerate to the left)
    - [1]: No push (coast)
    - [2]: Push right (accelerate to the right)

    {b Rewards}: -1.0 for each step until the goal is reached

    {b Episode Termination}:
    - Terminated: Car reaches position ≥ 0.5 (goal at top of right hill)
    - Truncated: Episode exceeds 200 steps

    {b Initial State}: Random position in \[-0.6, -0.4\] with velocity 0.0

    {b Rendering}: ASCII visualization showing car position ('C') and goal ('G')
    on a track

    {4 Example}

    Train an agent to reach the goal by building momentum:
    {[
      let rng = Rune.Rng.create () in
      let env = Fehu_envs.Mountain_car.make ~rng () in
      let obs, _ = Fehu.Env.reset env () in
      let rec run_episode steps =
        let action = (* policy decision based on position and velocity *) in
        let t = Fehu.Env.step env action in
        if t.terminated then
          Printf.printf "Goal reached in %d steps!\n" steps
        else if t.truncated then
          Printf.printf "Failed to reach goal in 200 steps\n"
        else
          run_episode (steps + 1)
      in
      run_episode 0
    ]}

    {4 Tips}

    - Classic exploration challenge: car engine is too weak to drive directly up
      the hill
    - Agent must learn to build momentum by driving back and forth
    - Sparse reward makes this difficult for naive value-based methods
    - Consider reward shaping (e.g., bonus for reaching higher positions) or
      policy gradient methods
    - Good for testing exploration strategies and delayed reward learning *)
