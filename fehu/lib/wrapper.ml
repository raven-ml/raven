let derive_id env suffix =
  match Env.id env with None -> None | Some id -> Some (id ^ suffix)

let inherit_metadata env = Env.metadata env

let map_observation ~(observation_space : 'obs Space.t)
    ~(f : 'inner_obs -> Info.t -> 'obs * Info.t)
    (env : ('inner_obs, 'act, 'render) Env.t) : ('obs, 'act, 'render) Env.t =
  Env.create
    ?id:(derive_id env "/ObservationWrapper")
    ~metadata:(inherit_metadata env) ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space ~action_space:(Env.action_space env)
    ~reset:(fun _wrapper ?options () ->
      let observation, info = Env.reset env ?options () in
      f observation info)
    ~step:(fun _wrapper action ->
      let transition = Env.step env action in
      let observation, info = f transition.observation transition.info in
      Env.transition ~observation ~reward:transition.reward
        ~terminated:transition.terminated ~truncated:transition.truncated ~info
        ())
    ()

let map_action ~(action_space : 'act Space.t) ~(f : 'act -> 'inner_act)
    (env : ('obs, 'inner_act, 'render) Env.t) : ('obs, 'act, 'render) Env.t =
  Env.create
    ?id:(derive_id env "/ActionWrapper")
    ~metadata:(inherit_metadata env) ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space:(Env.observation_space env)
    ~action_space
    ~reset:(fun _wrapper ?options () -> Env.reset env ?options ())
    ~step:(fun _wrapper action ->
      let transition = Env.step env (f action) in
      Env.transition ~observation:transition.observation
        ~reward:transition.reward ~terminated:transition.terminated
        ~truncated:transition.truncated ~info:transition.info ())
    ()

let map_reward ~(f : reward:float -> info:Info.t -> float * Info.t)
    (env : ('obs, 'act, 'render) Env.t) : ('obs, 'act, 'render) Env.t =
  Env.create
    ?id:(derive_id env "/RewardWrapper")
    ~metadata:(inherit_metadata env) ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun _wrapper ?options () -> Env.reset env ?options ())
    ~step:(fun _wrapper action ->
      let transition = Env.step env action in
      let reward, info = f ~reward:transition.reward ~info:transition.info in
      { transition with reward; info })
    ()

let map_info ~(f : Info.t -> Info.t) (env : ('obs, 'act, 'render) Env.t) :
    ('obs, 'act, 'render) Env.t =
  Env.create
    ?id:(derive_id env "/InfoWrapper")
    ~metadata:(inherit_metadata env) ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun _wrapper ?options () ->
      let observation, info = Env.reset env ?options () in
      (observation, f info))
    ~step:(fun _wrapper action ->
      let transition = Env.step env action in
      { transition with info = f transition.info })
    ()

let clamp_tensor ~low ~high tensor =
  let data = Rune.to_array tensor in
  let clipped = Array.copy data in
  let upper = Array.length clipped - 1 in
  for idx = 0 to upper do
    let lo = low.(idx) in
    let hi = high.(idx) in
    let v = clipped.(idx) in
    if v < lo then clipped.(idx) <- lo else if v > hi then clipped.(idx) <- hi
  done;
  Rune.create Rune.float32 (Rune.shape tensor) clipped

let clip_action (env : ('obs, Space.Box.element, 'render) Env.t) :
    ('obs, Space.Box.element, 'render) Env.t =
  let action_space = Env.action_space env in
  let low, high = Space.Box.bounds action_space in
  let element_count = Array.length low in
  if Array.length high <> element_count then
    invalid_arg "Wrapper.clip_action: mismatched low/high bounds";
  let relaxed_low =
    Array.init element_count (fun idx ->
        if Float.equal low.(idx) high.(idx) then low.(idx)
        else Float.neg_infinity)
  in
  let relaxed_high =
    Array.init element_count (fun idx ->
        if Float.equal low.(idx) high.(idx) then high.(idx) else Float.infinity)
  in
  let relaxed_space = Space.Box.create ~low:relaxed_low ~high:relaxed_high in
  map_action ~action_space:relaxed_space
    ~f:(fun action -> clamp_tensor ~low ~high action)
    env

let clip_observation (env : (Space.Box.element, 'act, 'render) Env.t) :
    (Space.Box.element, 'act, 'render) Env.t =
  let observation_space = Env.observation_space env in
  let low, high = Space.Box.bounds observation_space in
  let element_count = Array.length low in
  if Array.length high <> element_count then
    invalid_arg "Wrapper.clip_observation: mismatched low/high bounds";
  map_observation ~observation_space
    ~f:(fun observation info -> (clamp_tensor ~low ~high observation, info))
    env

let time_limit ~max_episode_steps env =
  if max_episode_steps <= 0 then
    invalid_arg "Wrapper.time_limit: max_episode_steps must be positive";
  let steps = ref 0 in
  let add_time_limit_info info elapsed_steps =
    info
    |> Info.set "time_limit.truncated" (Info.bool true)
    |> Info.set "time_limit.elapsed_steps" (Info.int elapsed_steps)
  in
  Env.create
    ?id:(derive_id env "/TimeLimit")
    ~metadata:(inherit_metadata env) ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun _wrapper ?options () ->
      steps := 0;
      Env.reset env ?options ())
    ~step:(fun _wrapper action ->
      incr steps;
      let transition = Env.step env action in
      if transition.terminated || transition.truncated then (
        steps := 0;
        transition)
      else if !steps >= max_episode_steps then (
        let info = add_time_limit_info transition.info !steps in
        steps := 0;
        { transition with truncated = true; info })
      else transition)
    ()

let with_metadata ~f env =
  let metadata = f (Env.metadata env) in
  Env.create ?id:(Env.id env) ~metadata ~rng:(Env.rng env)
    ~render:(fun _ -> Env.render env)
    ~close:(fun _ -> Env.close env)
    ~observation_space:(Env.observation_space env)
    ~action_space:(Env.action_space env)
    ~reset:(fun _wrapper ?options () -> Env.reset env ?options ())
    ~step:(fun _wrapper action -> Env.step env action)
    ()
