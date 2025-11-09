open Kaun
open Fehu
module Snapshot = Checkpoint.Snapshot

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

let default_config =
  {
    learning_rate = 0.001;
    gamma = 0.99;
    epsilon_start = 1.0;
    epsilon_end = 0.01;
    epsilon_decay = 1000.0;
    batch_size = 32;
    buffer_capacity = 10_000;
    target_update_freq = 10;
    warmup_steps = 1000;
  }

type params = Ptree.t

type metrics = {
  loss : float;
  avg_q_value : float;
  epsilon : float;
  episode_return : float option;
  episode_length : int option;
  total_steps : int;
  total_episodes : int;
}

type state = {
  q_network : module_;
  target_network : module_;
  optimizer : Optimizer.algorithm;
  target_params : params;
  opt_state : Optimizer.state;
  replay :
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t )
    Buffer.Replay.t;
  rng : Rune.Rng.key;
  epsilon_rng : Rune.Rng.key;
  epsilon_step : int;
  epsilon_schedule : int -> float;
  n_actions : int;
  action_space : (int32, Bigarray.int32_elt) Rune.t Space.t;
  config : config;
  total_steps : int;
  total_episodes : int;
  current_obs : (float, Bigarray.float32_elt) Rune.t;
  episode_return_acc : float;
  episode_length_acc : int;
  last_metrics : metrics;
}

let metrics state = state.last_metrics

let epsilon_value config step =
  config.epsilon_end
  +. (config.epsilon_start -. config.epsilon_end)
     *. Float.exp (-.float_of_int step /. config.epsilon_decay)

let make_optimizer config =
  let schedule = Optimizer.Schedule.constant config.learning_rate in
  Optimizer.adam ~lr:schedule ()

let discrete_cardinality action_space =
  match Space.boundary_values action_space with
  | [ Space.Value.Int start; Int finish ] -> finish - start + 1
  | [ Space.Value.Int _ ] -> 1
  | _ ->
      invalid_arg "Dqn.init: action space must provide discrete boundary values"

let reshape_observation obs =
  let obs = Rune.cast Rune.float32 obs in
  match Rune.shape obs with
  | [| features |] -> Rune.reshape [| 1; features |] obs
  | [| batch; _ |] when batch = 1 -> obs
  | _ -> obs

let not_done_mask terminated truncated =
  let done_mask = Rune.logical_or terminated truncated in
  let done_float = Rune.cast Rune.float32 done_mask in
  let ones = Rune.full_like done_float 1.0 in
  Rune.sub ones done_float

let compute_td_loss config ~n_actions ~params ~target_params ~q_network
    ~target_network ~optimizer ~opt_state ~batch =
  let observations, actions, rewards, next_observations, terminated, truncated =
    batch
  in
  let observations = Rune.cast Rune.float32 observations in
  let next_observations = Rune.cast Rune.float32 next_observations in
  let obs_batch = reshape_observation observations in
  let next_obs_batch = reshape_observation next_observations in
  let gather_axis =
    match Rune.shape obs_batch with
    | [| _; _ |] -> 1
    | shape -> Array.length shape - 1
  in
  let actions_vec = Rune.reshape [| (Rune.shape actions).(0) |] actions in
  let one_hot =
    Rune.cast Rune.float32 (Rune.one_hot ~num_classes:n_actions actions_vec)
  in
  let loss_tensor, grads =
    value_and_grad
      (fun current_params ->
        let q_values =
          apply q_network current_params ~training:true obs_batch
        in
        let masked = Rune.mul q_values one_hot in
        let chosen_q = Rune.sum ~axes:[ gather_axis ] masked ~keepdims:false in
        let target_q_values =
          apply target_network target_params ~training:false next_obs_batch
        in
        let max_next_q =
          Rune.max ~axes:[ gather_axis ] target_q_values ~keepdims:false
        in
        let not_done = not_done_mask terminated truncated in
        let discounted =
          Rune.mul
            (Rune.scalar Rune.float32 config.gamma)
            (Rune.mul not_done max_next_q)
        in
        let target = Rune.add rewards discounted in
        let td_error = Rune.sub chosen_q target in
        Rune.mean (Rune.square td_error))
      params
  in
  let loss = (Rune.to_array loss_tensor).(0) in
  let updates, opt_state' = Optimizer.step optimizer opt_state params grads in
  let params' = Optimizer.apply_updates params updates in
  (loss, params', opt_state')

let compute_avg_q params q_network observation =
  let obs = reshape_observation observation in
  let q_values = apply q_network params ~training:false obs in
  let axis = Array.length (Rune.shape q_values) - 1 in
  let mean =
    Rune.mean ~axes:[ axis ] q_values ~keepdims:false |> Rune.to_array
  in
  if Array.length mean = 0 then 0.0 else mean.(0)

let sample_batch buffer ~rng ~batch_size =
  Buffer.Replay.sample_tensors buffer ~rng ~batch_size

let initial_metrics config =
  {
    loss = 0.0;
    avg_q_value = 0.0;
    epsilon = config.epsilon_start;
    episode_return = None;
    episode_length = None;
    total_steps = 0;
    total_episodes = 0;
  }

let init ~env ~q_network ~rng ~config =
  let keys = Rune.Rng.split ~n:3 rng in
  let params = init q_network ~rngs:keys.(0) ~dtype:Rune.float32 in
  let target_params = Ptree.copy params in
  let optimizer = make_optimizer config in
  let opt_state = Optimizer.init optimizer params in
  let replay = Buffer.Replay.create ~capacity:config.buffer_capacity in
  let obs, _info = Env.reset env () in
  let epsilon_schedule step = epsilon_value config step in
  let action_space = Env.action_space env in
  let n_actions = discrete_cardinality action_space in
  let state =
    {
      q_network;
      target_network = q_network;
      optimizer;
      target_params;
      opt_state;
      replay;
      rng = keys.(2);
      epsilon_rng = keys.(1);
      epsilon_step = 0;
      epsilon_schedule;
      n_actions;
      action_space;
      config;
      total_steps = 0;
      total_episodes = 0;
      current_obs = obs;
      episode_return_acc = 0.;
      episode_length_acc = 0;
      last_metrics = initial_metrics config;
    }
  in
  (params, state)

let update_target_if_needed params state episode_finished =
  if
    episode_finished
    && state.config.target_update_freq > 0
    && (state.total_episodes + 1) mod state.config.target_update_freq = 0
  then Ptree.copy params
  else state.target_params

let perform_step ~env ~params ~state ~epsilon ~allow_update =
  let keys = Rune.Rng.split state.rng ~n:3 in
  let action_rng = keys.(0) in
  let replay_rng = keys.(1) in
  let rng_after = keys.(2) in
  let epsilon_keys = Rune.Rng.split state.epsilon_rng ~n:2 in
  let epsilon_rng_after = epsilon_keys.(0) in
  let coin_rng = epsilon_keys.(1) in
  let coin =
    Rune.Rng.uniform coin_rng Rune.float32 [| 1 |] |> Rune.to_array
    |> fun arr -> arr.(0)
  in
  let to_action idx =
    let idx_clamped = Int.max 0 (Int.min (state.n_actions - 1) idx) in
    match Space.unpack state.action_space (Space.Value.Int idx_clamped) with
    | Ok action -> action
    | Error _ ->
        let action, _ = Space.sample ~rng:action_rng state.action_space in
        action
  in
  let action =
    if coin < epsilon then
      let random_idx =
        let tensor =
          Rune.Rng.randint action_rng ~min:0 ~max:state.n_actions [| 1 |]
        in
        Rune.to_array tensor |> fun arr -> Int32.to_int arr.(0)
      in
      to_action random_idx
    else
      let obs = reshape_observation state.current_obs in
      let q_values = apply state.q_network params ~training:false obs in
      let axis = Array.length (Rune.shape q_values) - 1 in
      let argmax =
        Rune.argmax ~axis q_values ~keepdims:false |> Rune.cast Rune.int32
      in
      let best = Rune.to_array argmax |> fun arr -> Int32.to_int arr.(0) in
      to_action best
  in
  if not (Space.contains state.action_space action) then
    invalid_arg "Dqn.step produced action outside action space";
  let transition = Env.step env action in
  Buffer.Replay.add state.replay
    {
      Buffer.observation = state.current_obs;
      action;
      reward = transition.reward;
      next_observation = transition.observation;
      terminated = transition.terminated;
      truncated = transition.truncated;
    };
  let total_steps = state.total_steps + 1 in
  let episode_return_acc = state.episode_return_acc +. transition.reward in
  let episode_length_acc = state.episode_length_acc + 1 in
  let buffer_ready =
    Buffer.Replay.size state.replay >= state.config.batch_size
  in
  let loss, params', opt_state', avg_q =
    if allow_update && buffer_ready then
      let batch =
        sample_batch state.replay ~rng:replay_rng
          ~batch_size:state.config.batch_size
      in
      let loss, params', opt_state' =
        compute_td_loss state.config ~n_actions:state.n_actions ~params
          ~target_params:state.target_params ~q_network:state.q_network
          ~target_network:state.target_network ~optimizer:state.optimizer
          ~opt_state:state.opt_state ~batch
      in
      let avg_q = compute_avg_q params state.q_network state.current_obs in
      (loss, params', opt_state', avg_q)
    else (0.0, params, state.opt_state, 0.0)
  in
  let episode_finished = transition.terminated || transition.truncated in
  let total_episodes =
    if episode_finished then state.total_episodes + 1 else state.total_episodes
  in
  let next_obs =
    if episode_finished then fst (Env.reset env ()) else transition.observation
  in
  let episode_return, episode_length =
    if episode_finished then (Some episode_return_acc, Some episode_length_acc)
    else (None, None)
  in
  let episode_return_acc =
    if episode_finished then 0.0 else episode_return_acc
  in
  let episode_length_acc = if episode_finished then 0 else episode_length_acc in
  let target_params =
    if allow_update then update_target_if_needed params' state episode_finished
    else state.target_params
  in
  let metrics =
    {
      loss;
      avg_q_value = avg_q;
      epsilon;
      episode_return;
      episode_length;
      total_steps;
      total_episodes;
    }
  in
  let new_state =
    {
      state with
      target_params;
      opt_state = opt_state';
      rng = rng_after;
      epsilon_rng = epsilon_rng_after;
      epsilon_step = state.epsilon_step + 1;
      current_obs = next_obs;
      episode_return_acc;
      episode_length_acc;
      total_steps;
      total_episodes;
      last_metrics = metrics;
    }
  in
  (params', new_state)

let step ~env ~params ~state =
  let epsilon = state.epsilon_schedule state.epsilon_step in
  perform_step ~env ~params ~state ~epsilon ~allow_update:true

let train ~env ~q_network ~rng ~config ~total_timesteps ?callback () =
  let params, state = init ~env ~q_network ~rng ~config in
  let warmup_steps = Int.max 0 config.warmup_steps in
  let rec warmup params state remaining =
    if remaining <= 0 then (params, state)
    else
      let params', state' =
        perform_step ~env ~params ~state ~epsilon:1.0 ~allow_update:false
      in
      warmup params' state' (remaining - 1)
  in
  let params, state = warmup params state warmup_steps in
  let rec loop params state =
    if state.total_steps >= total_timesteps then (params, state)
    else
      let params', state' = step ~env ~params ~state in
      let continue =
        match callback with None -> `Continue | Some f -> f (metrics state')
      in
      match continue with
      | `Stop -> (params', state')
      | `Continue -> loop params' state'
  in
  loop params state

let dqn_schema_key = "schema"
let dqn_schema_value = "fehu.dqn/2"

let config_to_snapshot (c : config) : Snapshot.t =
  Snapshot.record
    [
      ("learning_rate", Snapshot.float c.learning_rate);
      ("gamma", Snapshot.float c.gamma);
      ("epsilon_start", Snapshot.float c.epsilon_start);
      ("epsilon_end", Snapshot.float c.epsilon_end);
      ("epsilon_decay", Snapshot.float c.epsilon_decay);
      ("batch_size", Snapshot.int c.batch_size);
      ("buffer_capacity", Snapshot.int c.buffer_capacity);
      ("target_update_freq", Snapshot.int c.target_update_freq);
      ("warmup_steps", Snapshot.int c.warmup_steps);
    ]

let config_of_snapshot (snapshot : Snapshot.t) : (config, string) result =
  let open Snapshot in
  let open Result in
  let ( let* ) = bind in
  let find_float field record =
    match Snapshot.Record.find_opt field record with
    | Some (Scalar (Float v)) -> Ok v
    | Some (Scalar (Int v)) -> Ok (float_of_int v)
    | Some _ -> Error (Printf.sprintf "DQN: field %s must be float" field)
    | None -> Error (Printf.sprintf "DQN: missing field %s" field)
  in
  let find_int field record =
    match Snapshot.Record.find_opt field record with
    | Some (Scalar (Int v)) -> Ok v
    | Some (Scalar (Float v)) -> Ok (int_of_float v)
    | Some _ -> Error (Printf.sprintf "DQN: field %s must be int" field)
    | None -> Error (Printf.sprintf "DQN: missing field %s" field)
  in
  match snapshot with
  | Record record ->
      let* learning_rate = find_float "learning_rate" record in
      let* gamma = find_float "gamma" record in
      let* epsilon_start = find_float "epsilon_start" record in
      let* epsilon_end = find_float "epsilon_end" record in
      let* epsilon_decay = find_float "epsilon_decay" record in
      let* batch_size = find_int "batch_size" record in
      let* buffer_capacity = find_int "buffer_capacity" record in
      let* target_update_freq = find_int "target_update_freq" record in
      let warmup_steps =
        match Snapshot.Record.find_opt "warmup_steps" record with
        | Some _value -> find_int "warmup_steps" record
        | None -> Ok default_config.warmup_steps
      in
      let* warmup_steps = warmup_steps in
      Ok
        {
          learning_rate;
          gamma;
          epsilon_start;
          epsilon_end;
          epsilon_decay;
          batch_size;
          buffer_capacity;
          target_update_freq;
          warmup_steps;
        }
  | _ -> Error "DQN: config snapshot must be a record"

let save ~path ~params ~state =
  let snapshot =
    Snapshot.record
      [
        (dqn_schema_key, Snapshot.string dqn_schema_value);
        ("config", config_to_snapshot state.config);
        ("n_actions", Snapshot.int state.n_actions);
        ("rng", Snapshot.rng state.rng);
        ("epsilon_rng", Snapshot.rng state.epsilon_rng);
        ("epsilon_step", Snapshot.int state.epsilon_step);
        ("total_steps", Snapshot.int state.total_steps);
        ("total_episodes", Snapshot.int state.total_episodes);
        ("q_params", Snapshot.ptree params);
        ("target_params", Snapshot.ptree state.target_params);
        ("optimizer_state", Optimizer.serialize state.opt_state);
      ]
  in
  match
    Checkpoint.write_snapshot_file_with ~path ~encode:(fun () -> snapshot)
  with
  | Ok () -> ()
  | Error err ->
      failwith (Printf.sprintf "Dqn.save: %s" (Checkpoint.error_to_string err))

let load ~path ~env ~q_network ~config =
  let open Result in
  let decode snapshot =
    let open Snapshot in
    let ( let* ) = bind in
    match snapshot with
    | Record record ->
        let validate_schema () =
          match Snapshot.Record.find_opt dqn_schema_key record with
          | Some (Scalar (String value))
            when String.equal value dqn_schema_value ->
              Ok ()
          | Some (Scalar (String value)) ->
              Error ("DQN: unsupported snapshot schema " ^ value)
          | Some _ -> Error "DQN: invalid schema field"
          | None -> Ok ()
        in
        let* () = validate_schema () in
        let* stored_config =
          match Snapshot.Record.find_opt "config" record with
          | Some c -> config_of_snapshot c
          | None -> Error "DQN: missing config field"
        in
        if stored_config <> config then
          Error "DQN: config mismatch between snapshot and requested config"
        else
          let find field =
            match Snapshot.Record.find_opt field record with
            | Some value -> Ok value
            | None -> Error ("DQN: missing field " ^ field)
          in
          let decode_rng = function
            | Scalar (Int seed) -> Ok (Rune.Rng.key seed)
            | Scalar (Float seed) -> Ok (Rune.Rng.key (int_of_float seed))
            | _ -> Error "DQN: rng field must be scalar"
          in
          let decode_int field =
            match field with
            | Scalar (Int v) -> Ok v
            | Scalar (Float v) -> Ok (int_of_float v)
            | _ -> Error "DQN: expected integer"
          in
          let* rng_node = find "rng" in
          let* rng_value = decode_rng rng_node in
          let* epsilon_rng_node = find "epsilon_rng" in
          let* epsilon_rng = decode_rng epsilon_rng_node in
          let* epsilon_step_node = find "epsilon_step" in
          let* epsilon_step = decode_int epsilon_step_node in
          let* total_steps_node = find "total_steps" in
          let* total_steps = decode_int total_steps_node in
          let* total_episodes_node = find "total_episodes" in
          let* total_episodes = decode_int total_episodes_node in
          let* q_params_node = find "q_params" in
          let* params =
            match Snapshot.to_ptree q_params_node with
            | Ok ptree -> Ok ptree
            | Error msg -> Error ("DQN: " ^ msg)
          in
          let* target_params_node = find "target_params" in
          let* target_params =
            match Snapshot.to_ptree target_params_node with
            | Ok ptree -> Ok ptree
            | Error msg -> Error ("DQN: " ^ msg)
          in
          let optimizer = make_optimizer config in
          let* opt_state_node = find "optimizer_state" in
          let* opt_state =
            match Optimizer.restore optimizer opt_state_node with
            | Ok state -> Ok state
            | Error msg -> Error ("DQN: " ^ msg)
          in
          let* n_actions_node = find "n_actions" in
          let* n_actions = decode_int n_actions_node in
          let replay = Buffer.Replay.create ~capacity:config.buffer_capacity in
          let obs, _ = Env.reset env () in
          let epsilon_schedule step = epsilon_value config step in
          let state =
            {
              q_network;
              target_network = q_network;
              optimizer;
              target_params;
              opt_state;
              replay;
              rng = rng_value;
              epsilon_rng;
              epsilon_step;
              epsilon_schedule;
              n_actions;
              action_space = Env.action_space env;
              config;
              total_steps;
              total_episodes;
              current_obs = obs;
              episode_return_acc = 0.;
              episode_length_acc = 0;
              last_metrics = initial_metrics config;
            }
          in
          Ok (params, state)
    | _ -> Error "DQN: invalid snapshot format"
  in
  match Checkpoint.load_snapshot_file_with ~path ~decode with
  | Ok result -> Ok result
  | Error err -> Error (Checkpoint.error_to_string err)
