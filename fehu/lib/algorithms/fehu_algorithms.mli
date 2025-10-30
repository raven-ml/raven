(** Reinforcement learning algorithms for Fehu.

    Each algorithm follows a functional interface:
    - {!val:Algorithm.init} prepares parameters and algorithm state for a given
      environment;
    - {!val:Algorithm.step} performs a single environment interaction and
      optimisation update;
    - {!val:Algorithm.train} runs a default training loop that repeatedly calls
      {!val:Algorithm.step}.

    {1 Available Algorithms}

    {2 Policy Gradient Methods}

    - {!Reinforce}: Monte Carlo Policy Gradient (REINFORCE)

    {2 Value-Based Methods}

    - {!Dqn}: Deep Q-Network (DQN)

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
