(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Built-in environments for testing and learning.

    Four environments covering the standard RL benchmarks: a simple 1D walk, the
    classic cart-pole, a grid navigation problem, and the sparse-reward mountain
    car. All follow the {!Fehu.Env} interface. *)

(** {1:envs Environments} *)

module Random_walk : sig
  (** One-dimensional random walk.

      The agent moves left or right on a line bounded by \[[-10]; [10]\]. Reward
      is [- |position|]. Episodes terminate when the agent reaches a boundary or
      after 200 steps.

      {b Observation}: {!Fehu.Space.Box} of shape [[1]] in \[[-10.0]; [10.0]\].

      {b Actions}: {!Fehu.Space.Discrete} 2 -- 0 = left, 1 = right.

      {b Render modes}: [ansi]. *)

  type obs = (float, Rune.float32_elt) Rune.t
  type act = (int32, Rune.int32_elt) Rune.t
  type render = string

  val make :
    ?render_mode:Fehu.Env.render_mode ->
    rng:Rune.Rng.key ->
    unit ->
    (obs, act, render) Fehu.Env.t
  (** [make ~rng ()] is a random walk environment seeded with [rng]. *)
end

module Cartpole : sig
  (** Classic cart-pole balancing (CartPole-v1).

      A pole is attached to a cart on a frictionless track. The agent pushes the
      cart left or right to keep the pole upright. Reward is [+1.0] per step
      while the pole stays up. The episode terminates when the pole exceeds
      +/-12 degrees or the cart leaves +/-2.4, and truncates at 500 steps.

      {b Observation}: {!Fehu.Space.Box} of shape [[4]] -- [x], [x_dot],
      [theta], [theta_dot].

      {b Actions}: {!Fehu.Space.Discrete} 2 -- 0 = push left, 1 = push right.

      {b Render modes}: [ansi]. *)

  type obs = (float, Rune.float32_elt) Rune.t
  type act = (int32, Rune.int32_elt) Rune.t
  type render = string

  val make :
    ?render_mode:Fehu.Env.render_mode ->
    rng:Rune.Rng.key ->
    unit ->
    (obs, act, render) Fehu.Env.t
  (** [make ~rng ()] is a cart-pole environment seeded with [rng]. *)
end

module Grid_world : sig
  (** 5x5 grid navigation with obstacle.

      The agent starts at [(0, 0)] and must reach the goal at [(4, 4)]. An
      obstacle at [(2, 2)] blocks movement. Reward is [+10.0] on reaching the
      goal, [-1.0] otherwise. Truncates at 200 steps.

      {b Observation}: {!Fehu.Space.Multi_discrete} [[5; 5]] -- [(row, col)].

      {b Actions}: {!Fehu.Space.Discrete} 4 -- 0 = up, 1 = down, 2 = left, 3 =
      right.

      {b Render modes}: [ansi], [rgb_array]. *)

  type obs = (int32, Rune.int32_elt) Rune.t
  type act = (int32, Rune.int32_elt) Rune.t
  type render = Text of string | Image of Fehu.Render.image

  val make :
    ?render_mode:Fehu.Env.render_mode ->
    rng:Rune.Rng.key ->
    unit ->
    (obs, act, render) Fehu.Env.t
  (** [make ~rng ()] is a grid world environment seeded with [rng]. *)
end

module Mountain_car : sig
  (** Mountain car with sparse reward (MountainCar-v0).

      A car sits in a valley between two hills. The engine is too weak to climb
      the right hill directly; the agent must build momentum by rocking back and
      forth. Reward is [-1.0] per step. The episode terminates when the car
      reaches position >= 0.5 with non-negative velocity, and truncates at 200
      steps.

      {b Observation}: {!Fehu.Space.Box} of shape [[2]] -- [position],
      [velocity].

      {b Actions}: {!Fehu.Space.Discrete} 3 -- 0 = push left, 1 = coast, 2 =
      push right.

      {b Render modes}: [ansi]. *)

  type obs = (float, Rune.float32_elt) Rune.t
  type act = (int32, Rune.int32_elt) Rune.t
  type render = string

  val make :
    ?render_mode:Fehu.Env.render_mode ->
    rng:Rune.Rng.key ->
    unit ->
    (obs, act, render) Fehu.Env.t
  (** [make ~rng ()] is a mountain car environment seeded with [rng]. *)
end
