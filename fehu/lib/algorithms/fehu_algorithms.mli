(** Reinforcement learning algorithms for Fehu.

    This library provides production-ready implementations of standard RL
    algorithms. Each algorithm follows a consistent interface: create an agent
    with a policy network and configuration, train with {!learn}, and use the
    trained policy with {!predict}.

    {1 Available Algorithms}

    {2 Policy Gradient Methods}

    - {!Reinforce}: Monte Carlo Policy Gradient (REINFORCE)

    {2 Value-Based Methods}

    - {!Dqn}: Deep Q-Network (DQN)

    {1 Usage Pattern}

    All algorithms follow this pattern:
    {[
      open Fehu

      (* 1. Create policy network *)
      let policy_net = Kaun.Layer.sequential [...] in

      (* 2. Initialize algorithm *)
      let agent = Algorithm.create
        ~policy_network:policy_net
        ~n_actions:n
        ~rng:(Rune.Rng.key 42)
        Algorithm.default_config
      in

      (* 3. Train *)
      let agent = Algorithm.learn agent ~env ~total_timesteps:100_000 () in

      (* 4. Use trained policy *)
      let action = Algorithm.predict agent obs ~training:false |> fst
    ]}

    {1 Choosing an Algorithm}

    - {b REINFORCE}: Simple policy gradient, works for small discrete action
      spaces, requires complete episodes. Good for learning but sample
      inefficient.
    - {b DQN}: Off-policy value-based method with experience replay, good for
      discrete actions, more sample efficient than REINFORCE.

    Future algorithms:
    - {b PPO}: More sample efficient, supports continuous actions, industry
      standard
    - {b SAC}: Off-policy actor-critic, excellent for continuous control *)

module Reinforce = Reinforce
(** {!Reinforce} algorithm implementation.

    REINFORCE (Monte Carlo Policy Gradient) is a classic policy gradient method
    that collects complete episodes and updates the policy using Monte Carlo
    return estimates. See {!Reinforce} for detailed documentation. *)

module Dqn = Dqn
(** {!Dqn} algorithm implementation.

    DQN (Deep Q-Network) is an off-policy value-based method that uses
    experience replay and target networks for stable training. It learns
    Q-values for discrete actions and selects actions greedily. See {!Dqn} for
    detailed documentation. *)
