type ('obs, 'act) t = {
  observations : 'obs array;
  actions : 'act array;
  rewards : float array;
  terminateds : bool array;
  truncateds : bool array;
  infos : Info.t array;
  log_probs : float array option;
  values : float array option;
}

let create ~observations ~actions ~rewards ~terminateds ~truncateds
    ?(infos = [||]) ?log_probs ?values () =
  let n = Array.length observations in
  if
    n <> Array.length actions
    || n <> Array.length rewards
    || n <> Array.length terminateds
    || n <> Array.length truncateds
  then invalid_arg "Trajectory.create: arrays must have same length";

  let infos =
    if Array.length infos = 0 then Array.make n Info.empty
    else if Array.length infos = n then infos
    else
      invalid_arg "Trajectory.create: infos array must be empty or same length"
  in

  let log_probs =
    match log_probs with
    | Some arr when Array.length arr <> n ->
        invalid_arg "Trajectory.create: log_probs must have same length"
    | Some arr -> Some arr
    | None -> None
  in

  let values =
    match values with
    | Some arr when Array.length arr <> n ->
        invalid_arg "Trajectory.create: values must have same length"
    | Some arr -> Some arr
    | None -> None
  in

  {
    observations;
    actions;
    rewards;
    terminateds;
    truncateds;
    infos;
    log_probs;
    values;
  }

let length t = Array.length t.observations

let concat trajectories =
  match trajectories with
  | [] -> invalid_arg "Trajectory.concat: empty list"
  | [ t ] -> t
  | ts ->
      let observations = Array.concat (List.map (fun t -> t.observations) ts) in
      let actions = Array.concat (List.map (fun t -> t.actions) ts) in
      let rewards = Array.concat (List.map (fun t -> t.rewards) ts) in
      let terminateds = Array.concat (List.map (fun t -> t.terminateds) ts) in
      let truncateds = Array.concat (List.map (fun t -> t.truncateds) ts) in
      let infos = Array.concat (List.map (fun t -> t.infos) ts) in

      let log_probs =
        if List.for_all (fun t -> Option.is_some t.log_probs) ts then
          Some (Array.concat (List.map (fun t -> Option.get t.log_probs) ts))
        else None
      in

      let values =
        if List.for_all (fun t -> Option.is_some t.values) ts then
          Some (Array.concat (List.map (fun t -> Option.get t.values) ts))
        else None
      in

      {
        observations;
        actions;
        rewards;
        terminateds;
        truncateds;
        infos;
        log_probs;
        values;
      }

let collect env ~policy ~n_steps =
  let observations = ref [] in
  let actions = ref [] in
  let rewards = ref [] in
  let terminateds = ref [] in
  let truncateds = ref [] in
  let infos = ref [] in
  let log_probs = ref [] in
  let values = ref [] in

  let obs, _info = Env.reset env () in
  let current_obs = ref obs in
  let steps_collected = ref 0 in

  while !steps_collected < n_steps do
    let action, log_prob_opt, value_opt = policy !current_obs in

    observations := !current_obs :: !observations;
    actions := action :: !actions;
    (match log_prob_opt with
    | Some lp -> log_probs := lp :: !log_probs
    | None -> ());
    (match value_opt with Some v -> values := v :: !values | None -> ());

    let transition = Env.step env action in

    rewards := transition.Env.reward :: !rewards;
    terminateds := transition.Env.terminated :: !terminateds;
    truncateds := transition.Env.truncated :: !truncateds;
    infos := transition.Env.info :: !infos;

    current_obs := transition.Env.observation;
    steps_collected := !steps_collected + 1;

    if transition.Env.terminated || transition.Env.truncated then
      let obs, _info = Env.reset env () in
      current_obs := obs
  done;

  let log_probs_arr =
    if List.length !log_probs = n_steps then
      Some (Array.of_list (List.rev !log_probs))
    else None
  in
  let values_arr =
    if List.length !values = n_steps then
      Some (Array.of_list (List.rev !values))
    else None
  in
  create
    ~observations:(Array.of_list (List.rev !observations))
    ~actions:(Array.of_list (List.rev !actions))
    ~rewards:(Array.of_list (List.rev !rewards))
    ~terminateds:(Array.of_list (List.rev !terminateds))
    ~truncateds:(Array.of_list (List.rev !truncateds))
    ~infos:(Array.of_list (List.rev !infos))
    ?log_probs:log_probs_arr ?values:values_arr ()

let collect_episodes env ~policy ~n_episodes ?(max_steps = 1000) () =
  let episodes = ref [] in

  for _ = 1 to n_episodes do
    let observations = ref [] in
    let actions = ref [] in
    let rewards = ref [] in
    let terminateds = ref [] in
    let truncateds = ref [] in
    let infos = ref [] in
    let log_probs = ref [] in
    let values = ref [] in

    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let steps = ref 0 in
    let done_flag = ref false in

    while !steps < max_steps && not !done_flag do
      let action, log_prob_opt, value_opt = policy !current_obs in

      observations := !current_obs :: !observations;
      actions := action :: !actions;
      (match log_prob_opt with
      | Some lp -> log_probs := lp :: !log_probs
      | None -> ());
      (match value_opt with Some v -> values := v :: !values | None -> ());

      let transition = Env.step env action in

      rewards := transition.Env.reward :: !rewards;
      terminateds := transition.Env.terminated :: !terminateds;
      truncateds := transition.Env.truncated :: !truncateds;
      infos := transition.Env.info :: !infos;

      current_obs := transition.Env.observation;
      steps := !steps + 1;
      done_flag := transition.Env.terminated || transition.Env.truncated
    done;

    let n = !steps in
    let log_probs_arr =
      if List.length !log_probs = n then
        Some (Array.of_list (List.rev !log_probs))
      else None
    in
    let values_arr =
      if List.length !values = n then Some (Array.of_list (List.rev !values))
      else None
    in
    let episode =
      create
        ~observations:(Array.of_list (List.rev !observations))
        ~actions:(Array.of_list (List.rev !actions))
        ~rewards:(Array.of_list (List.rev !rewards))
        ~terminateds:(Array.of_list (List.rev !terminateds))
        ~truncateds:(Array.of_list (List.rev !truncateds))
        ~infos:(Array.of_list (List.rev !infos))
        ?log_probs:log_probs_arr ?values:values_arr ()
    in
    episodes := episode :: !episodes
  done;

  List.rev !episodes
