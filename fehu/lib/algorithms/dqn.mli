(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Deep Q-Network (DQN) training API. *)

type config = {
  learning_rate : float;
  gamma : float;
  epsilon_start : float;
  epsilon_end : float;
  epsilon_decay : float;
  batch_size : int;
  buffer_capacity : int;
  target_update_freq : int;
  warmup_steps : int;
}

val default_config : config

type params = Kaun.Ptree.t

type metrics = {
  loss : float;
  avg_q_value : float;
  epsilon : float;
  episode_return : float option;
  episode_length : int option;
  total_steps : int;
  total_episodes : int;
}

type state

val init :
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  q_network:Kaun.module_ ->
  rng:Rune.Rng.key ->
  config:config ->
  params * state

val step :
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  params:params ->
  state:state ->
  params * state

val metrics : state -> metrics
(** Latest metrics gathered after {!step}. *)

val train :
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  q_network:Kaun.module_ ->
  rng:Rune.Rng.key ->
  config:config ->
  total_timesteps:int ->
  ?callback:(metrics -> [ `Continue | `Stop ]) ->
  unit ->
  params * state

val save : path:string -> params:params -> state:state -> unit

val load :
  path:string ->
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  q_network:Kaun.module_ ->
  config:config ->
  (params * state, string) result
