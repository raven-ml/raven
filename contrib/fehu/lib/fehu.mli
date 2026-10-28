(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reinforcement learning environments and utilities.

    Environments are host data: observations, actions, rewards and stored
    transitions are host values ({!Nx.host}), and samples draw from
    [Nx.Rng]'s scope ({!Nx.Rng.with_key}). A policy on another device set
    places what it reads ({!Nx.place}).

    {1 Core}
    {!modules: Value Info Space Env}

    {1 Collection and training}
    {!modules: Collect Buffer Gae Eval}

    {1 Composition}
    {!modules: Vec_env Render} *)

module Value = Value
module Info = Info
module Space = Space
module Env = Env
module Vec_env = Vec_env
module Collect = Collect
module Buffer = Buffer
module Gae = Gae
module Eval = Eval
module Render = Render
