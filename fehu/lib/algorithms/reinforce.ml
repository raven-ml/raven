open Kaun
open Fehu
module Snapshot = Checkpoint.Snapshot

type config = {
  learning_rate : float;
  gamma : float;
  use_baseline : bool;
  reward_scale : float;
  entropy_coef : float;
  max_episode_steps : int;
}

let default_config =
  {
    learning_rate = 0.001;
    gamma = 0.99;
    use_baseline = false;
    reward_scale = 1.0;
    entropy_coef = 0.01;
    max_episode_steps = 1000;
  }

type params = Ptree.t

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

type episode_step = {
  observation : (float, Bigarray.float32_elt) Rune.t;
  next_observation : (float, Bigarray.float32_elt) Rune.t;
  info : Info.t;
  action_index : int;
  reward : float;
  terminated : bool;
  truncated : bool;
}

type state = {
  policy_network : module_;
  baseline_network : module_ option;
  policy_optimizer : Optimizer.algorithm;
  baseline_optimizer : Optimizer.algorithm option;
  policy_opt_state : Optimizer.state;
  baseline_opt_state : Optimizer.state option;
  baseline_params : params option;
  rng : Rune.Rng.key;
  config : config;
  total_steps : int;
  total_episodes : int;
  current_obs : (float, Bigarray.float32_elt) Rune.t;
  current_info : Info.t;
  episode_steps_rev : episode_step list;
  episode_return_acc : float;
  episode_length_acc : int;
  action_offset : int;
  n_actions : int;
  last_metrics : metrics;
}

let make_optimizer config =
  let schedule = Optimizer.Schedule.constant config.learning_rate in
  Optimizer.adam ~lr:schedule ()

let discrete_action_info action_space =
  match Space.boundary_values action_space with
  | [ Space.Value.Int start; Int finish ] -> (start, finish - start + 1)
  | [ Space.Value.Int start ] -> (start, 1)
  | _ ->
      invalid_arg
        "Reinforce.init: action space must provide discrete boundary values"

let reshape_observation obs =
  match Rune.shape obs with
  | [||] -> Rune.reshape [| 1 |] obs
  | [| features |] -> Rune.reshape [| 1; features |] obs
  | [| 1; _ |] -> obs
  | [| batch; _ |] when batch = 1 -> obs
  | _ -> obs

let invalid_logit_offset = -1e9

let mask_offsets_of_info ~n_actions info =
  match Info.find "action_mask" info with
  | Some (Info.Bool_array arr) ->
      let len = Array.length arr in
      let offsets =
        Array.init n_actions (fun idx ->
            if idx < len && arr.(idx) then 0.0 else invalid_logit_offset)
      in
      Some offsets
  | _ -> None

let batch_mask_tensor_of_infos ~n_actions infos =
  let rows = Array.length infos in
  if rows = 0 then None
  else
    let has_mask = ref false in
    let data = Array.make (rows * n_actions) 0.0 in
    for row = 0 to rows - 1 do
      match mask_offsets_of_info ~n_actions infos.(row) with
      | Some offsets ->
          has_mask := true;
          Array.blit offsets 0 data (row * n_actions) n_actions
      | None -> ()
    done;
    if !has_mask then Some (Rune.create Rune.float32 [| rows; n_actions |] data)
    else None

let add_mask_to_logits logits = function
  | Some mask -> Rune.add logits mask
  | None -> logits

let compute_returns ~gamma ~rewards ~terminateds ~truncateds =
  let n = Array.length rewards in
  let returns = Array.make n 0.0 in
  let running = ref 0.0 in
  for idx = n - 1 downto 0 do
    if terminateds.(idx) || truncateds.(idx) then running := 0.0;
    running := rewards.(idx) +. (gamma *. !running);
    returns.(idx) <- !running
  done;
  returns

let float_array_mean arr =
  let n = Array.length arr in
  if n = 0 then 0.0 else Array.fold_left ( +. ) 0.0 arr /. float_of_int n

let float_array_std arr ~mean =
  let n = Array.length arr in
  if n = 0 then 0.0
  else
    let variance =
      Array.fold_left
        (fun acc value ->
          let diff = value -. mean in
          acc +. (diff *. diff /. float_of_int n))
        0.0 arr
    in
    sqrt variance

let initial_metrics =
  {
    episode_return = 0.0;
    episode_length = 0;
    episode_won = false;
    stage_desc = "unknown-stage";
    avg_entropy = 0.0;
    avg_log_prob = 0.0;
    adv_mean = 0.0;
    adv_std = 0.0;
    value_loss = None;
    total_steps = 0;
    total_episodes = 0;
  }

let metrics state = state.last_metrics

let log_softmax logits =
  let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
  let shifted = Rune.sub logits max_logits in
  let sum_exp = Rune.sum (Rune.exp shifted) ~axes:[ -1 ] ~keepdims:true in
  Rune.sub shifted (Rune.log sum_exp)

let policy_statistics ?mask ~n_actions ~action_indices ~obs_batch ~network
    ~params () =
  let logits = apply network params ~training:false obs_batch in
  let logits = add_mask_to_logits logits mask in
  let log_probs = log_softmax logits in
  let gather_axis = Array.length (Rune.shape log_probs) - 1 in
  let one_hot =
    Rune.cast Rune.float32 (Rune.one_hot ~num_classes:n_actions action_indices)
  in
  let selected =
    Rune.mul log_probs one_hot |> Rune.sum ~axes:[ gather_axis ] ~keepdims:false
  in
  let probs = Rune.exp log_probs in
  let entropy =
    Rune.mul probs log_probs |> Rune.neg
    |> Rune.sum ~axes:[ gather_axis ] ~keepdims:false
  in
  let log_prob_array = Rune.to_array selected in
  let entropy_array = Rune.to_array entropy in
  let avg_log_prob = float_array_mean log_prob_array in
  let avg_entropy = float_array_mean entropy_array in
  (avg_log_prob, avg_entropy)

type episode_update = {
  params : params;
  policy_opt_state : Optimizer.state;
  baseline_params : params option;
  baseline_opt_state : Optimizer.state option;
  episode_return : float;
  episode_length : int;
  episode_won : bool;
  stage_desc : string;
  avg_entropy : float;
  avg_log_prob : float;
  adv_mean : float;
  adv_std : float;
  value_loss : float option;
}

let perform_episode_update ~params ~(algo_state : state) steps =
  let steps = List.rev steps in
  let n_steps = List.length steps in
  if n_steps = 0 then
    {
      params;
      policy_opt_state = algo_state.policy_opt_state;
      baseline_params = algo_state.baseline_params;
      baseline_opt_state = algo_state.baseline_opt_state;
      episode_return = 0.0;
      episode_length = 0;
      episode_won = false;
      stage_desc = algo_state.last_metrics.stage_desc;
      avg_entropy = 0.0;
      avg_log_prob = 0.0;
      adv_mean = 0.0;
      adv_std = 0.0;
      value_loss = None;
    }
  else
    let observations =
      Array.of_list (List.map (fun step -> step.observation) steps)
    in
    let obs_batch =
      Rune.stack ~axis:0 (Array.to_list observations) |> Rune.cast Rune.float32
    in
    let raw_rewards =
      Array.of_list (List.map (fun step -> step.reward) steps)
    in
    let rewards =
      Array.map
        (fun reward -> reward *. algo_state.config.reward_scale)
        raw_rewards
    in
    let terminateds =
      Array.of_list (List.map (fun step -> step.terminated) steps)
    in
    let truncateds =
      Array.of_list (List.map (fun step -> step.truncated) steps)
    in
    let dones =
      Array.init n_steps (fun idx -> terminateds.(idx) || truncateds.(idx))
    in
    let infos = Array.of_list (List.map (fun step -> step.info) steps) in
    let action_indices_array =
      Array.of_list (List.map (fun step -> step.action_index) steps)
    in
    let action_indices =
      Array.map (fun idx -> Int32.of_int idx) action_indices_array
      |> Rune.create Rune.int32 [| n_steps |]
    in
    let mask_tensor =
      batch_mask_tensor_of_infos ~n_actions:algo_state.n_actions infos
    in
    let baseline_values =
      if algo_state.config.use_baseline then
        match (algo_state.baseline_network, algo_state.baseline_params) with
        | Some net, Some params ->
            apply net params ~training:false obs_batch
            |> Rune.reshape [| n_steps |] |> Rune.to_array
        | _ ->
            invalid_arg
              "Reinforce.step: baseline parameters missing despite \
               use_baseline=true"
      else Array.make n_steps 0.0
    in
    let last_step =
      match List.rev steps with last :: _ -> last | [] -> assert false
    in
    let last_done = last_step.terminated || last_step.truncated in
    let last_value =
      if algo_state.config.use_baseline && not last_done then
        match (algo_state.baseline_network, algo_state.baseline_params) with
        | Some net, Some params ->
            let next_obs =
              reshape_observation last_step.next_observation
              |> Rune.cast Rune.float32
            in
            let value =
              apply net params ~training:false next_obs
              |> Rune.reshape [| 1 |] |> Rune.to_array
            in
            value.(0)
        | _ ->
            invalid_arg
              "Reinforce.step: baseline parameters missing despite \
               use_baseline=true"
      else 0.0
    in
    let stage_desc =
      match Info.find "stage" last_step.info with
      | Some (Info.String s) -> s
      | _ -> "unknown-stage"
    in
    let episode_won = last_step.terminated in
    let advantages_raw, returns_raw =
      if algo_state.config.use_baseline then
        Training.compute_gae ~rewards ~values:baseline_values ~dones ~last_value
          ~last_done ~gamma:algo_state.config.gamma ~gae_lambda:0.95
      else
        let returns =
          compute_returns ~gamma:algo_state.config.gamma ~rewards ~terminateds
            ~truncateds
        in
        (Array.copy returns, returns)
    in
    let adv_mean = float_array_mean advantages_raw in
    let adv_std = float_array_std advantages_raw ~mean:adv_mean in
    let denom = if adv_std < 1e-6 then 1.0 else adv_std in
    let advantages_norm =
      Array.map (fun a -> (a -. adv_mean) /. denom) advantages_raw
    in
    let advantages_tensor =
      Rune.create Rune.float32 [| n_steps |] advantages_norm
    in
    let policy_loss_fn current_params =
      let logits =
        apply algo_state.policy_network current_params ~training:true obs_batch
      in
      let logits = add_mask_to_logits logits mask_tensor in
      let log_probs = log_softmax logits in
      let gather_axis = Array.length (Rune.shape log_probs) - 1 in
      let one_hot =
        Rune.cast Rune.float32
          (Rune.one_hot ~num_classes:algo_state.n_actions action_indices)
      in
      let selected =
        Rune.mul log_probs one_hot
        |> Rune.sum ~axes:[ gather_axis ] ~keepdims:false
      in
      let entropy =
        let probs = Rune.exp log_probs in
        Rune.mul probs log_probs |> Rune.neg
        |> Rune.sum ~axes:[ gather_axis ] ~keepdims:false
      in
      let policy_term =
        Rune.mul advantages_tensor selected |> Rune.neg |> Rune.mean
      in
      let entropy_term =
        let entropy_mean = Rune.mean entropy in
        Rune.mul
          (Rune.scalar Rune.float32 algo_state.config.entropy_coef)
          entropy_mean
      in
      Rune.sub policy_term entropy_term
    in
    let _loss, policy_grads = value_and_grad policy_loss_fn params in
    let updates, policy_opt_state =
      Optimizer.step algo_state.policy_optimizer algo_state.policy_opt_state
        params policy_grads
    in
    let params = Optimizer.apply_updates params updates in
    let avg_log_prob, avg_entropy =
      policy_statistics ?mask:mask_tensor ~n_actions:algo_state.n_actions
        ~action_indices ~obs_batch ~network:algo_state.policy_network ~params ()
    in
    let baseline_params, baseline_opt_state, value_loss =
      if algo_state.config.use_baseline then
        match
          ( algo_state.baseline_network,
            algo_state.baseline_params,
            algo_state.baseline_optimizer,
            algo_state.baseline_opt_state )
        with
        | Some net, Some base_params, Some optimizer, Some opt_state ->
            let returns_tensor =
              Rune.create Rune.float32 [| n_steps |] returns_raw
            in
            let baseline_loss params =
              let values =
                apply net params ~training:true obs_batch
                |> Rune.reshape [| n_steps |]
              in
              let diff = Rune.sub values returns_tensor in
              Rune.mean (Rune.square diff)
            in
            let loss_tensor, grads = value_and_grad baseline_loss base_params in
            let updates, opt_state =
              Optimizer.step optimizer opt_state base_params grads
            in
            let base_params = Optimizer.apply_updates base_params updates in
            let loss = (Rune.to_array loss_tensor).(0) in
            (Some base_params, Some opt_state, Some loss)
        | _ ->
            invalid_arg
              "Reinforce.step: baseline optimizer state missing despite \
               use_baseline=true"
      else (algo_state.baseline_params, algo_state.baseline_opt_state, None)
    in
    let episode_return = Array.fold_left ( +. ) 0.0 raw_rewards in
    {
      params;
      policy_opt_state;
      baseline_params;
      baseline_opt_state;
      episode_return;
      episode_length = n_steps;
      episode_won;
      stage_desc;
      avg_entropy;
      avg_log_prob;
      adv_mean;
      adv_std;
      value_loss;
    }

let init ?baseline_network ~env ~policy_network ~rng ~config () =
  if config.use_baseline && Option.is_none baseline_network then
    invalid_arg
      "Reinforce.init: baseline_network required when use_baseline = true";
  let action_space = Env.action_space env in
  let action_offset, n_actions = discrete_action_info action_space in
  let key_count = if config.use_baseline then 3 else 2 in
  let keys = Rune.Rng.split ~n:key_count rng in
  let policy_key = keys.(0) in
  let rng_key = keys.(key_count - 1) in
  let baseline_key = if config.use_baseline then Some keys.(1) else None in
  let params = init policy_network ~rngs:policy_key ~dtype:Rune.float32 in
  let policy_optimizer = make_optimizer config in
  let policy_opt_state = Optimizer.init policy_optimizer params in
  let baseline_network, baseline_params, baseline_optimizer, baseline_opt_state
      =
    match (config.use_baseline, baseline_network, baseline_key) with
    | true, Some net, Some key ->
        let params = init net ~rngs:key ~dtype:Rune.float32 in
        let optimizer = make_optimizer config in
        let opt_state = Optimizer.init optimizer params in
        (Some net, Some params, Some optimizer, Some opt_state)
    | _ -> (None, None, None, None)
  in
  let current_obs, current_info = Env.reset env () in
  let state =
    {
      policy_network;
      baseline_network;
      policy_optimizer;
      baseline_optimizer;
      policy_opt_state;
      baseline_opt_state;
      baseline_params;
      rng = rng_key;
      config;
      total_steps = 0;
      total_episodes = 0;
      current_obs;
      current_info;
      episode_steps_rev = [];
      episode_return_acc = 0.0;
      episode_length_acc = 0;
      action_offset;
      n_actions;
      last_metrics = initial_metrics;
    }
  in
  (params, state)

let step ~env ~params ~state =
  let keys = Rune.Rng.split state.rng ~n:2 in
  let sample_key = keys.(0) in
  let rng_after = keys.(1) in
  let obs = state.current_obs in
  let obs_batched = reshape_observation obs |> Rune.cast Rune.float32 in
  let mask_tensor =
    match
      mask_offsets_of_info ~n_actions:state.n_actions state.current_info
    with
    | Some offsets ->
        Some (Rune.create Rune.float32 [| 1; state.n_actions |] offsets)
    | None -> None
  in
  let logits = apply state.policy_network params ~training:true obs_batched in
  let logits = add_mask_to_logits logits mask_tensor in
  let indices =
    Rune.Rng.categorical ~key:sample_key logits |> Rune.reshape [| 1 |]
  in
  let idx_array = Rune.to_array indices in
  let action_index = Int32.to_int idx_array.(0) in
  if action_index < 0 || action_index >= state.n_actions then
    invalid_arg "Reinforce.step produced invalid action index";
  let action_value = state.action_offset + action_index in
  let action = Rune.scalar Rune.int32 (Int32.of_int action_value) in
  let transition = Env.step env action in
  let length_after = state.episode_length_acc + 1 in
  let limit_reached = length_after >= state.config.max_episode_steps in
  let truncated =
    transition.truncated || (limit_reached && not transition.terminated)
  in
  let episode_done = transition.terminated || truncated in
  let episode_step =
    {
      observation = state.current_obs;
      next_observation = transition.observation;
      info = state.current_info;
      action_index;
      reward = transition.reward;
      terminated = transition.terminated;
      truncated;
    }
  in
  let episode_steps_rev = episode_step :: state.episode_steps_rev in
  let episode_return_acc = state.episode_return_acc +. transition.reward in
  let episode_length_acc = length_after in
  let total_steps = state.total_steps + 1 in
  let total_episodes =
    if episode_done then state.total_episodes + 1 else state.total_episodes
  in
  let params, policy_opt_state, baseline_params, baseline_opt_state, metrics =
    if episode_done then
      let update =
        perform_episode_update ~params ~algo_state:state episode_steps_rev
      in
      let metrics =
        {
          episode_return = update.episode_return;
          episode_length = update.episode_length;
          episode_won = update.episode_won;
          stage_desc = update.stage_desc;
          avg_entropy = update.avg_entropy;
          avg_log_prob = update.avg_log_prob;
          adv_mean = update.adv_mean;
          adv_std = update.adv_std;
          value_loss = update.value_loss;
          total_steps;
          total_episodes;
        }
      in
      ( update.params,
        update.policy_opt_state,
        update.baseline_params,
        update.baseline_opt_state,
        metrics )
    else
      ( params,
        state.policy_opt_state,
        state.baseline_params,
        state.baseline_opt_state,
        state.last_metrics )
  in
  let next_obs, next_info =
    if episode_done then Env.reset env ()
    else (transition.observation, transition.info)
  in
  let episode_steps_rev = if episode_done then [] else episode_steps_rev in
  let episode_return_acc = if episode_done then 0.0 else episode_return_acc in
  let episode_length_acc = if episode_done then 0 else episode_length_acc in
  ( params,
    {
      state with
      rng = rng_after;
      current_obs = next_obs;
      current_info = next_info;
      episode_steps_rev;
      episode_return_acc;
      episode_length_acc;
      total_steps;
      total_episodes;
      policy_opt_state;
      baseline_opt_state;
      baseline_params;
      last_metrics = metrics;
    } )

let train ?baseline_network ~env ~policy_network ~rng ~config ~total_timesteps
    ?callback () =
  let params, state =
    init ?baseline_network ~env ~policy_network ~rng ~config ()
  in
  let rec loop params state =
    if state.total_steps >= total_timesteps then (params, state)
    else
      let params, state = step ~env ~params ~state in
      let continue =
        match callback with None -> `Continue | Some f -> f (metrics state)
      in
      match continue with
      | `Stop -> (params, state)
      | `Continue -> loop params state
  in
  loop params state

let reinforce_schema_key = "schema"
let reinforce_schema_value = "fehu.reinforce/2"

let config_to_snapshot (c : config) =
  Snapshot.record
    [
      ("learning_rate", Snapshot.float c.learning_rate);
      ("gamma", Snapshot.float c.gamma);
      ("use_baseline", Snapshot.bool c.use_baseline);
      ("reward_scale", Snapshot.float c.reward_scale);
      ("entropy_coef", Snapshot.float c.entropy_coef);
      ("max_episode_steps", Snapshot.int c.max_episode_steps);
    ]

let config_of_snapshot snapshot =
  let open Result in
  let ( let* ) = bind in
  let open Snapshot in
  let error field message =
    Error (Printf.sprintf "Reinforce config field %s %s" field message)
  in
  match snapshot with
  | Record record ->
      let find_float field =
        match Snapshot.Record.find_opt field record with
        | Some (Scalar (Float value)) -> Ok value
        | Some (Scalar (Int value)) -> Ok (float_of_int value)
        | Some _ -> error field "must be float"
        | None -> error field "missing"
      in
      let find_int field =
        match Snapshot.Record.find_opt field record with
        | Some (Scalar (Int value)) -> Ok value
        | Some (Scalar (Float value)) -> Ok (int_of_float value)
        | Some _ -> error field "must be int"
        | None -> error field "missing"
      in
      let find_bool field =
        match Snapshot.Record.find_opt field record with
        | Some (Scalar (Bool value)) -> Ok value
        | Some _ -> error field "must be bool"
        | None -> error field "missing"
      in
      let* learning_rate = find_float "learning_rate" in
      let* gamma = find_float "gamma" in
      let* use_baseline = find_bool "use_baseline" in
      let* reward_scale = find_float "reward_scale" in
      let* entropy_coef = find_float "entropy_coef" in
      let* max_episode_steps = find_int "max_episode_steps" in
      Ok
        {
          learning_rate;
          gamma;
          use_baseline;
          reward_scale;
          entropy_coef;
          max_episode_steps;
        }
  | _ -> Error "Reinforce config must be a record"

let metrics_to_snapshot (m : metrics) =
  Snapshot.record
    [
      ("episode_return", Snapshot.float m.episode_return);
      ("episode_length", Snapshot.int m.episode_length);
      ("episode_won", Snapshot.bool m.episode_won);
      ("stage_desc", Snapshot.string m.stage_desc);
      ("avg_entropy", Snapshot.float m.avg_entropy);
      ("avg_log_prob", Snapshot.float m.avg_log_prob);
      ("adv_mean", Snapshot.float m.adv_mean);
      ("adv_std", Snapshot.float m.adv_std);
      ( "value_loss",
        match m.value_loss with
        | Some loss -> Snapshot.list [ Snapshot.float loss ]
        | None -> Snapshot.list [] );
      ("total_steps", Snapshot.int m.total_steps);
      ("total_episodes", Snapshot.int m.total_episodes);
    ]

let metrics_of_snapshot snapshot =
  let open Result in
  let ( let* ) = bind in
  let open Snapshot in
  match snapshot with
  | Record record ->
      let find_float field =
        match Snapshot.Record.find_opt field record with
        | Some (Scalar (Float value)) -> Ok value
        | Some (Scalar (Int value)) -> Ok (float_of_int value)
        | Some _ ->
            Error (Printf.sprintf "Reinforce metrics %s must be float" field)
        | None -> Error (Printf.sprintf "Reinforce metrics missing %s" field)
      in
      let find_int field =
        match Snapshot.Record.find_opt field record with
        | Some (Scalar (Int value)) -> Ok value
        | Some (Scalar (Float value)) -> Ok (int_of_float value)
        | Some _ ->
            Error (Printf.sprintf "Reinforce metrics %s must be int" field)
        | None -> Error (Printf.sprintf "Reinforce metrics missing %s" field)
      in
      let* episode_return = find_float "episode_return" in
      let* episode_length = find_int "episode_length" in
      let episode_won =
        match Snapshot.Record.find_opt "episode_won" record with
        | Some (Scalar (Bool value)) -> value
        | Some _ -> false
        | None -> false
      in
      let stage_desc =
        match Snapshot.Record.find_opt "stage_desc" record with
        | Some (Scalar (String value)) -> value
        | Some _ -> "unknown-stage"
        | None -> "unknown-stage"
      in
      let* avg_entropy = find_float "avg_entropy" in
      let* avg_log_prob = find_float "avg_log_prob" in
      let* adv_mean = find_float "adv_mean" in
      let* adv_std = find_float "adv_std" in
      let value_loss =
        match Snapshot.Record.find_opt "value_loss" record with
        | Some (List [ Scalar (Float value) ]) -> Some value
        | Some (List [ Scalar (Int value) ]) -> Some (float_of_int value)
        | _ -> None
      in
      let* total_steps = find_int "total_steps" in
      let* total_episodes = find_int "total_episodes" in
      Ok
        {
          episode_return;
          episode_length;
          episode_won;
          stage_desc;
          avg_entropy;
          avg_log_prob;
          adv_mean;
          adv_std;
          value_loss;
          total_steps;
          total_episodes;
        }
  | _ -> Error "Reinforce metrics must be a record"

let save ~path ~params ~(state : state) =
  let baseline_entries =
    match (state.baseline_params, state.baseline_opt_state) with
    | Some params, Some opt_state ->
        [
          ("baseline_params", Snapshot.ptree params);
          ("baseline_optimizer_state", Optimizer.serialize opt_state);
        ]
    | _ -> []
  in
  let snapshot =
    Snapshot.record
      ([
         (reinforce_schema_key, Snapshot.string reinforce_schema_value);
         ("config", config_to_snapshot state.config);
         ("rng", Snapshot.rng state.rng);
         ("policy_params", Snapshot.ptree params);
         ("policy_optimizer_state", Optimizer.serialize state.policy_opt_state);
         ("n_actions", Snapshot.int state.n_actions);
         ("action_offset", Snapshot.int state.action_offset);
         ("total_steps", Snapshot.int state.total_steps);
         ("total_episodes", Snapshot.int state.total_episodes);
         ("last_metrics", metrics_to_snapshot state.last_metrics);
       ]
      @ baseline_entries)
  in
  match
    Checkpoint.write_snapshot_file_with ~path ~encode:(fun () -> snapshot)
  with
  | Ok () -> ()
  | Error err ->
      failwith
        (Printf.sprintf "Reinforce.save: %s" (Checkpoint.error_to_string err))

let load ~path ~env ~policy_network ?baseline_network ~config () =
  let open Result in
  let action_space = Env.action_space env in
  let action_offset_env, n_actions_env = discrete_action_info action_space in
  let decode snapshot =
    let open Snapshot in
    let ( let* ) = bind in
    match snapshot with
    | Record record ->
        let validate_schema () =
          match Snapshot.Record.find_opt reinforce_schema_key record with
          | Some (Scalar (String value))
            when String.equal value reinforce_schema_value ->
              Ok ()
          | Some (Scalar (String value)) ->
              Error ("REINFORCE: unsupported snapshot schema " ^ value)
          | Some _ -> Error "REINFORCE: invalid schema field"
          | None -> Ok ()
        in
        let* () = validate_schema () in
        let* stored_config =
          match Snapshot.Record.find_opt "config" record with
          | Some cfg -> config_of_snapshot cfg
          | None -> Error "REINFORCE: missing config field"
        in
        if stored_config <> config then
          Error "REINFORCE: config mismatch between snapshot and request"
        else
          let require field =
            match Snapshot.Record.find_opt field record with
            | Some value -> Ok value
            | None -> Error ("REINFORCE: missing field " ^ field)
          in
          let decode_int = function
            | Scalar (Int value) -> Ok value
            | Scalar (Float value) -> Ok (int_of_float value)
            | _ -> Error "REINFORCE: expected integer field"
          in
          let decode_rng = function
            | Scalar (Int seed) -> Ok (Rune.Rng.key seed)
            | Scalar (Float seed) -> Ok (Rune.Rng.key (int_of_float seed))
            | _ -> Error "REINFORCE: rng field must be scalar"
          in
          let* rng_node = require "rng" in
          let* rng = decode_rng rng_node in
          let* policy_params_node = require "policy_params" in
          let* params =
            match Snapshot.to_ptree policy_params_node with
            | Ok tree -> Ok tree
            | Error msg -> Error ("REINFORCE: " ^ msg)
          in
          let policy_optimizer = make_optimizer config in
          let* policy_opt_state_node = require "policy_optimizer_state" in
          let* policy_opt_state =
            match Optimizer.restore policy_optimizer policy_opt_state_node with
            | Ok state -> Ok state
            | Error msg -> Error ("REINFORCE: " ^ msg)
          in
          let baseline_entries =
            match
              ( Snapshot.Record.find_opt "baseline_params" record,
                Snapshot.Record.find_opt "baseline_optimizer_state" record )
            with
            | Some params_node, Some opt_state_node ->
                let* baseline_module =
                  match baseline_network with
                  | Some net -> Ok (Some net)
                  | None ->
                      Error
                        "REINFORCE: baseline snapshot found but no baseline \
                         network provided"
                in
                let* baseline_params =
                  match Snapshot.to_ptree params_node with
                  | Ok tree -> Ok tree
                  | Error msg -> Error ("REINFORCE: " ^ msg)
                in
                let optimizer = make_optimizer config in
                let* baseline_opt_state =
                  match Optimizer.restore optimizer opt_state_node with
                  | Ok state -> Ok state
                  | Error msg -> Error ("REINFORCE: " ^ msg)
                in
                Ok
                  ( baseline_module,
                    Some baseline_params,
                    Some optimizer,
                    Some baseline_opt_state )
            | _ ->
                if config.use_baseline then
                  Error "REINFORCE: snapshot missing baseline parameters/state"
                else Ok (baseline_network, None, None, None)
          in
          let* ( baseline_network,
                 baseline_params,
                 baseline_optimizer,
                 baseline_opt_state ) =
            baseline_entries
          in
          if config.use_baseline && Option.is_none baseline_network then
            Error
              "REINFORCE: snapshot requires baseline network but none provided"
          else
            let* n_actions_node = require "n_actions" in
            let* n_actions = decode_int n_actions_node in
            let* action_offset_node = require "action_offset" in
            let* action_offset = decode_int action_offset_node in
            if n_actions <> n_actions_env || action_offset <> action_offset_env
            then
              Error "REINFORCE: environment action space mismatch for snapshot"
            else
              let* total_steps_node = require "total_steps" in
              let* total_steps = decode_int total_steps_node in
              let* total_episodes_node = require "total_episodes" in
              let* total_episodes = decode_int total_episodes_node in
              let last_metrics =
                match Snapshot.Record.find_opt "last_metrics" record with
                | Some snapshot -> (
                    match metrics_of_snapshot snapshot with
                    | Ok metrics -> metrics
                    | Error _ -> initial_metrics)
                | None -> initial_metrics
              in
              let current_obs, current_info = Env.reset env () in
              let state =
                {
                  policy_network;
                  baseline_network;
                  policy_optimizer;
                  baseline_optimizer;
                  policy_opt_state;
                  baseline_opt_state;
                  baseline_params;
                  rng;
                  config;
                  total_steps;
                  total_episodes;
                  current_obs;
                  current_info;
                  episode_steps_rev = [];
                  episode_return_acc = 0.0;
                  episode_length_acc = 0;
                  action_offset;
                  n_actions;
                  last_metrics;
                }
              in
              Ok (params, state)
    | _ -> Error "REINFORCE: invalid snapshot format"
  in
  match Checkpoint.load_snapshot_file_with ~path ~decode with
  | Ok result -> Ok result
  | Error err -> Error (Checkpoint.error_to_string err)
