(** Fehu - Reinforcement Learning Framework for OCaml

    Fehu provides RL-specific components that compose with Kaun for neural
    networks. Users should use Kaun models for policies and value functions. *)

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
end

(** {1 Common Environments} *)

module Envs : sig
  val cartpole : unit -> [ `c ] Env.t
  val mountain_car : unit -> [ `c ] Env.t
  val pendulum : unit -> [ `c ] Env.t
end
