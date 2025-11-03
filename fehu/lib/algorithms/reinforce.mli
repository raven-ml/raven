(** Monte Carlo policy gradient (REINFORCE) training API. *)

type config = {
  learning_rate : float;
  gamma : float;
  use_baseline : bool;
  reward_scale : float;
  entropy_coef : float;
  max_episode_steps : int;
}

val default_config : config

type params = Kaun.Ptree.t

type metrics = {
  episode_return : float;
  episode_length : int;
  episode_won : bool;
  stage_desc : string;
  avg_entropy : float;
  avg_log_prob : float;
  adv_mean : float;
  adv_std : float;
  value_loss : float option;
  total_steps : int;
  total_episodes : int;
}

type state

val init :
  ?baseline_network:Kaun.module_ ->
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  policy_network:Kaun.module_ ->
  rng:Rune.Rng.key ->
  config:config ->
  unit ->
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

val train :
  ?baseline_network:Kaun.module_ ->
  env:
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t,
      'render )
    Fehu.Env.t ->
  policy_network:Kaun.module_ ->
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
  policy_network:Kaun.module_ ->
  ?baseline_network:Kaun.module_ ->
  config:config ->
  unit ->
  (params * state, string) result
