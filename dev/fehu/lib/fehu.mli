(** Fehu - Reinforcement Learning Framework for OCaml

    Fehu provides RL-specific components that compose with Kaun for neural
    networks. Users should use Kaun models for policies and value functions. *)

(** {1 Visualization} *)

module Visualization = Visualization

(** {1 Spaces} *)

module Space : sig
  type 'dev t =
    | Discrete of int  (** Discrete space with n possible actions *)
    | Box of {
        low : (float, Rune.float32_elt, 'dev) Rune.t;
        high : (float, Rune.float32_elt, 'dev) Rune.t;
        shape : int array;
      }  (** Continuous space with bounds *)
    | Multi_discrete of int array  (** Multiple discrete spaces *)

  val sample :
    rng:Rune.Rng.key ->
    'dev Rune.device ->
    'dev t ->
    (float, Rune.float32_elt, 'dev) Rune.t

  val contains : 'dev t -> (float, Rune.float32_elt, 'dev) Rune.t -> bool
  val shape : 'dev t -> int array
end

(** {1 Environments} *)

module Env : sig
  type info = (string * Yojson.Basic.t) list

  type 'dev t = {
    observation_space : 'dev Space.t;
    action_space : 'dev Space.t;
    reset : ?seed:int -> unit -> (float, Rune.float32_elt, 'dev) Rune.t * info;
    step :
      (float, Rune.float32_elt, 'dev) Rune.t ->
      (float, Rune.float32_elt, 'dev) Rune.t * float * bool * bool * info;
    (* Returns: observation, reward, terminated, truncated, info *)
    render : unit -> unit;
    close : unit -> unit;
  }

  val make :
    observation_space:'dev Space.t ->
    action_space:'dev Space.t ->
    reset:(?seed:int -> unit -> (float, Rune.float32_elt, 'dev) Rune.t * info) ->
    step:
      ((float, Rune.float32_elt, 'dev) Rune.t ->
      (float, Rune.float32_elt, 'dev) Rune.t * float * bool * bool * info) ->
    ?render:(unit -> unit) ->
    ?close:(unit -> unit) ->
    unit ->
    'dev t
end

(** {1 Replay Buffers} *)

module Buffer : sig
  type 'dev transition = {
    obs : (float, Rune.float32_elt, 'dev) Rune.t;
    action : (float, Rune.float32_elt, 'dev) Rune.t;
    reward : float;
    next_obs : (float, Rune.float32_elt, 'dev) Rune.t;
    terminated : bool;
  }

  type 'dev t

  val create : capacity:int -> 'dev t
  val add : 'dev t -> 'dev transition -> unit

  val sample :
    'dev t -> rng:Rune.Rng.key -> batch_size:int -> 'dev transition array

  val size : 'dev t -> int
  val is_full : 'dev t -> bool
end

(** {1 Training Utilities} *)

module Training : sig
  type stats = {
    episode_reward : float;
    episode_length : int;
    total_timesteps : int;
    n_episodes : int;
    mean_reward : float;
    std_reward : float;
  }

  val evaluate :
    'dev Env.t ->
    policy:
      ((float, Rune.float32_elt, 'dev) Rune.t ->
      (float, Rune.float32_elt, 'dev) Rune.t) ->
    n_eval_episodes:int ->
    stats

  val compute_gae :
    rewards:float array ->
    values:float array ->
    dones:bool array ->
    gamma:float ->
    gae_lambda:float ->
    float array * float array
  (** Returns advantages and returns *)

  val compute_returns :
    rewards:float array -> dones:bool array -> gamma:float -> float array
  (** Compute discounted returns *)

  val normalize :
    (float, Rune.float32_elt, 'dev) Rune.t ->
    ?eps:float ->
    unit ->
    (float, Rune.float32_elt, 'dev) Rune.t
  (** Normalize tensor to zero mean and unit variance *)

  val compute_advantages :
    rewards:float array ->
    values:float array ->
    gamma:float ->
    float array * float array
  (** Compute advantages for policy gradient methods. Returns (advantages, returns) *)

  val compute_policy_loss :
    log_probs:float array ->
    advantages:float array ->
    normalize_advantages:bool ->
    (float, Rune.float32_elt, [ `c ]) Rune.t
  (** Compute REINFORCE policy gradient loss *)

  val compute_grpo_loss :
    log_probs:float array ->
    ref_log_probs:float array ->
    advantages:float array ->
    beta:float ->
    (float, Rune.float32_elt, [ `c ]) Rune.t
  (** Compute GRPO (Group Relative Policy Optimization) loss *)
end

(** {1 Trajectories} *)

module Trajectory : sig
  type 'dev t = {
    states : (float, Rune.float32_elt, 'dev) Rune.t array;
    actions : (float, Rune.float32_elt, 'dev) Rune.t array;
    rewards : float array;
    log_probs : float array option;
    values : float array option;
    dones : bool array;
  }

  val create :
    states:(float, Rune.float32_elt, 'dev) Rune.t array ->
    actions:(float, Rune.float32_elt, 'dev) Rune.t array ->
    rewards:float array ->
    ?log_probs:float array option ->
    ?values:float array option ->
    ?dones:bool array ->
    unit ->
    'dev t

  val length : 'dev t -> int
  
  val concat : 'dev t list -> 'dev t
  (** Concatenate multiple trajectories into one *)
end

(** {1 Curriculum Learning} *)

module Curriculum : sig
  type 'dev t

  val create :
    stages:'dev Env.t array ->
    advance_criterion:(Training.stats -> bool) ->
    ?window_size:int ->
    unit ->
    'dev t
  (** Create curriculum with multiple environment stages *)

  val current_env : 'dev t -> 'dev Env.t
  (** Get current environment in curriculum *)

  val update_stats : 'dev t -> Training.stats -> unit
  (** Update curriculum with latest training statistics *)

  val try_advance : 'dev t -> bool
  (** Try to advance to next stage. Returns true if advanced *)

  val reset : 'dev t -> unit
  (** Reset curriculum to first stage *)
end

(** {1 Common Environments} *)

module Envs : sig
  val cartpole : unit -> [ `c ] Env.t
  val mountain_car : unit -> [ `c ] Env.t
  val pendulum : unit -> [ `c ] Env.t
end
