(** Environment wrappers for modifying behavior without changing core logic.

    Wrappers intercept environment interactions to transform observations,
    actions, or rewards, add constraints like time limits, or modify metadata.
    They follow the Gymnasium wrapper pattern, composing through function
    chaining.

    {1 Usage}

    Apply wrappers by wrapping an environment:
    {[
      let env = make_base_env () in
      let env = Wrapper.time_limit ~max_episode_steps:1000 env in
      let env = Wrapper.map_reward ~f:(fun ~reward ~info -> (reward *. 0.1, info)) env
    ]}

    Chain multiple wrappers:
    {[
      let env =
        make_base_env ()
        |> Wrapper.map_observation ~observation_space ~f:normalize_obs
        |> Wrapper.time_limit ~max_episode_steps:500
        |> Wrapper.map_reward ~f:clip_reward
    ]}

    {1 Common Patterns}

    - Normalize observations to a standard range
    - Discretize continuous actions or vice versa
    - Clip or scale rewards for training stability
    - Enforce episode time limits
    - Add custom metadata or tracking *)

val map_observation :
  observation_space:'obs Space.t ->
  f:('inner_obs -> Info.t -> 'obs * Info.t) ->
  ('inner_obs, 'act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [map_observation ~observation_space ~f env] transforms observations.

    Wraps [env] so that every observation from {!Env.reset} and {!Env.step} is
    transformed by [f]. The function [f] receives the inner observation and
    info, returning a transformed observation and potentially modified info.

    Common use: Normalizing observations, converting image formats, or adding
    derived features.

    Example normalizing observations:
    {[
      let normalize obs info =
        let normalized = (* scale obs to [-1, 1] *) in
        (normalized, info)
      in
      Wrapper.map_observation ~observation_space:normalized_space ~f:normalize env
    ]} *)

val map_action :
  action_space:'act Space.t ->
  f:('act -> 'inner_act) ->
  ('obs, 'inner_act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [map_action ~action_space ~f env] transforms actions before passing to the
    environment.

    Wraps [env] so that actions provided to {!Env.step} are transformed by [f]
    before reaching the inner environment. This allows changing the action
    interface without modifying environment logic.

    Common use: Discretizing continuous actions, converting action
    representations, or adding action constraints.

    Example discretizing continuous actions:
    {[
      let discretize discrete_action =
        (* map discrete index to continuous value *)
        match discrete_action with
        | 0 -> -1.0
        | 1 -> 0.0
        | 2 -> 1.0
      in
      Wrapper.map_action ~action_space:discrete_space ~f:discretize env
    ]} *)

val map_reward :
  f:(reward:float -> info:Info.t -> float * Info.t) ->
  ('obs, 'act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [map_reward ~f env] transforms rewards after each step.

    Wraps [env] so that rewards from {!Env.step} are transformed by [f]. The
    function [f] receives the reward and info, returning a modified reward and
    potentially updated info.

    Common use: Reward clipping, scaling, or shaping to improve learning.

    Example clipping rewards:
    {[
      let clip ~reward ~info =
        (Float.max (-1.0) (Float.min 1.0 reward), info)
      in
      Wrapper.map_reward ~f:clip env
    ]} *)

val map_info :
  f:(Info.t -> Info.t) ->
  ('obs, 'act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [map_info ~f env] post-processes the info dictionary returned by
    {!Env.reset} and {!Env.step}.

    Use this to inject diagnostics, strip keys, or normalize info payloads
    without mutating the underlying environment. *)

val clip_action :
  ('obs, Space.Box.element, 'render) Env.t ->
  ('obs, Space.Box.element, 'render) Env.t
(** [clip_action env] clamps continuous actions to the bounds of the wrapped
    environment's {!Space.Box} action space.

    The wrapper exposes a relaxed external action space that accepts any float
    values, then clips them before forwarding to the inner environment. This is
    equivalent to Gymnasium's [ActionClipWrapper] and is useful when a policy
    may produce out-of-range actions. *)

val clip_observation :
  (Space.Box.element, 'act, 'render) Env.t ->
  (Space.Box.element, 'act, 'render) Env.t
(** [clip_observation env] clamps continuous observations to the bounds of the
    wrapped environment's {!Space.Box} observation space. *)

val time_limit :
  max_episode_steps:int ->
  ('obs, 'act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [time_limit ~max_episode_steps env] enforces a maximum episode length.

    Wraps [env] to truncate episodes after [max_episode_steps] steps. When the
    limit is reached, the step's [truncated] flag is set to [true] and the
    episode ends. Natural termination ([terminated] = true) occurs
    independently.

    Use this to prevent infinite episodes or ensure fixed-length rollouts.

    The step counter resets on {!Env.reset}. *)

val with_metadata :
  f:(Metadata.t -> Metadata.t) ->
  ('obs, 'act, 'render) Env.t ->
  ('obs, 'act, 'render) Env.t
(** [with_metadata ~f env] modifies environment metadata.

    Wraps [env] so its metadata is transformed by [f]. This allows adding render
    modes, tags, or other metadata without altering environment implementation.

    Example adding a render mode:
    {[
      let add_mode meta = Metadata.add_render_mode "rgb_array" meta in
      Wrapper.with_metadata ~f:add_mode env
    ]} *)
