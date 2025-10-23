open Kaun

type config = {
  learning_rate : float;
  gamma : float;
  epsilon_start : float;
  epsilon_end : float;
  epsilon_decay : float;
  batch_size : int;
  buffer_capacity : int;
  target_update_freq : int;
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
  }

type t = {
  q_network : module_;
  mutable q_params : params;
  target_network : module_;
  mutable target_params : params;
  optimizer : Optimizer.algorithm;
  mutable opt_state : Optimizer.state;
  replay_buffer :
    ( (float, Bigarray.float32_elt) Rune.t,
      (int32, Bigarray.int32_elt) Rune.t )
    Fehu.Buffer.Replay.t;
  mutable rng : Rune.Rng.key;
  n_actions : int;
  config : config;
}

type update_metrics = {
  episode_return : float;
  episode_length : int;
  epsilon : float;
  avg_q_value : float;
  loss : float;
}

let create ~q_network ~n_actions ~rng config =
  let keys = Rune.Rng.split ~n:2 rng in

  (* Initialize Q-network *)
  let q_params = init q_network ~rngs:keys.(0) ~dtype:Rune.float32 in

  (* Initialize target network with same architecture *)
  let target_params = Ptree.copy q_params in

  (* Create optimizer *)
  let lr = Optimizer.Schedule.constant config.learning_rate in
  let optimizer = Optimizer.adam ~lr () in
  let opt_state = Optimizer.init optimizer q_params in

  (* Create replay buffer *)
  let replay_buffer =
    Fehu.Buffer.Replay.create ~capacity:config.buffer_capacity
  in

  {
    q_network;
    q_params;
    target_network = q_network;
    target_params;
    optimizer;
    opt_state;
    replay_buffer;
    rng = keys.(1);
    n_actions;
    config;
  }

let predict t obs ~epsilon =
  (* Add batch dimension if needed: [features] -> [1, features] *)
  let obs_shape = Rune.shape obs in
  let obs_batched =
    if Array.length obs_shape = 1 then
      let features = obs_shape.(0) in
      Rune.reshape [| 1; features |] obs
    else obs
  in

  (* Epsilon-greedy exploration *)
  let keys = Rune.Rng.split t.rng ~n:2 in
  t.rng <- keys.(0);
  let sample_rng = keys.(1) in

  let uniform_sample = Rune.Rng.uniform sample_rng Rune.float32 [| 1 |] in
  let r = (Rune.to_array uniform_sample).(0) in

  if r < epsilon then (
    (* Random action *)
    let keys = Rune.Rng.split t.rng ~n:2 in
    t.rng <- keys.(0);
    let action_rng = keys.(1) in
    let action_tensor =
      Rune.Rng.randint action_rng ~min:0 ~max:t.n_actions [| 1 |]
    in
    Rune.reshape [||] (Rune.cast Rune.int32 action_tensor))
  else
    (* Greedy action - select action with highest Q-value *)
    let q_values = apply t.q_network t.q_params ~training:false obs_batched in
    (* q_values shape: [1, n_actions] *)
    let q_flat = Rune.reshape [| t.n_actions |] q_values in
    let q_array = Rune.to_array q_flat in

    (* Find argmax *)
    let best_action = ref 0 in
    let best_q = ref q_array.(0) in
    for i = 1 to Array.length q_array - 1 do
      if q_array.(i) > !best_q then (
        best_action := i;
        best_q := q_array.(i))
    done;

    Rune.scalar Rune.int32 (Int32.of_int !best_action)

let add_transition t ~observation ~action ~reward ~next_observation ~terminated
    ~truncated =
  Fehu.Buffer.Replay.add t.replay_buffer
    Fehu.Buffer.
      { observation; action; reward; next_observation; terminated; truncated }

let update t =
  (* Check if we have enough samples *)
  if Fehu.Buffer.Replay.size t.replay_buffer < t.config.batch_size then
    (0.0, 0.0)
  else
    let keys = Rune.Rng.split t.rng ~n:2 in
    t.rng <- keys.(0);
    let sample_rng = keys.(1) in

    (* Sample batch *)
    let batch =
      Fehu.Buffer.Replay.sample t.replay_buffer ~rng:sample_rng
        ~batch_size:t.config.batch_size
    in

    (* Compute avg Q-value for metrics (before gradient computation) *)
    let avg_q =
      let total_q = ref 0.0 in
      Array.iter
        (fun (trans : _ Fehu.Buffer.transition) ->
          let obs_shape = Rune.shape trans.observation in
          let obs_batched =
            if Array.length obs_shape = 1 then
              let features = obs_shape.(0) in
              Rune.reshape [| 1; features |] trans.observation
            else trans.observation
          in
          let q_values =
            apply t.q_network t.q_params ~training:false obs_batched
          in
          let action_idx = Int32.to_int (Rune.to_array trans.action).(0) in
          let current_q = Rune.item [ 0; action_idx ] q_values in
          total_q := !total_q +. current_q)
        batch;
      !total_q /. float_of_int (Array.length batch)
    in

    (* Compute loss and gradients *)
    let loss_tensor, grads =
      value_and_grad
        (fun params ->
          let total_loss = ref 0.0 in

          Array.iter
            (fun (trans : _ Fehu.Buffer.transition) ->
              (* Get observation shape and batch if needed *)
              let obs_shape = Rune.shape trans.observation in
              let obs_batched =
                if Array.length obs_shape = 1 then
                  let features = obs_shape.(0) in
                  Rune.reshape [| 1; features |] trans.observation
                else trans.observation
              in

              (* Current Q-value: Q(s, a) *)
              let q_values =
                apply t.q_network params ~training:true obs_batched
              in
              let action_idx = Int32.to_int (Rune.to_array trans.action).(0) in
              let current_q = Rune.item [ 0; action_idx ] q_values in

              (* Target Q-value *)
              let target_q =
                if trans.terminated then trans.reward
                else
                  (* Get next observation shape and batch if needed *)
                  let next_obs_shape = Rune.shape trans.next_observation in
                  let next_obs_batched =
                    if Array.length next_obs_shape = 1 then
                      let features = next_obs_shape.(0) in
                      Rune.reshape [| 1; features |] trans.next_observation
                    else trans.next_observation
                  in

                  (* Use target network for next Q-values: max_a' Q_target(s',
                     a') *)
                  let next_q_values =
                    apply t.target_network t.target_params ~training:false
                      next_obs_batched
                  in
                  let next_q_flat =
                    Rune.reshape [| t.n_actions |] next_q_values
                  in
                  let next_q_array = Rune.to_array next_q_flat in

                  (* Find max Q-value *)
                  let max_next_q = ref next_q_array.(0) in
                  for i = 1 to Array.length next_q_array - 1 do
                    if next_q_array.(i) > !max_next_q then
                      max_next_q := next_q_array.(i)
                  done;

                  trans.reward +. (t.config.gamma *. !max_next_q)
              in

              (* TD error: (Q(s,a) - target)^2 *)
              let diff = current_q -. target_q in
              total_loss := !total_loss +. (diff *. diff))
            batch;

          let avg_loss = !total_loss /. float_of_int (Array.length batch) in
          Rune.create Rune.float32 [||] [| avg_loss |])
        t.q_params
    in

    let loss_float = (Rune.to_array loss_tensor).(0) in

    (* Apply gradients *)
    let updates, new_opt_state =
      Optimizer.step t.optimizer t.opt_state t.q_params grads
    in
    t.q_params <- Optimizer.apply_updates t.q_params updates;
    t.opt_state <- new_opt_state;

    (loss_float, avg_q)

let update_target_network t = t.target_params <- Ptree.copy t.q_params

let learn t ~env ~total_timesteps
    ?(callback = fun ~episode:_ ~metrics:_ -> true)
    ?(warmup_steps = t.config.batch_size) () =
  let open Fehu in
  let timesteps = ref 0 in
  let episode = ref 0 in

  (* Warmup phase: collect initial experiences *)
  (if warmup_steps > 0 then
     let obs, _info = Env.reset env () in
     let current_obs = ref obs in
     let warmup_done = ref false in

     while !timesteps < warmup_steps && not !warmup_done do
       (* Random action during warmup *)
       let action = predict t !current_obs ~epsilon:1.0 in
       let transition = Env.step env action in

       add_transition t ~observation:!current_obs ~action
         ~reward:transition.Env.reward
         ~next_observation:transition.Env.observation
         ~terminated:transition.Env.terminated
         ~truncated:transition.Env.truncated;

       current_obs := transition.Env.observation;
       timesteps := !timesteps + 1;

       if transition.Env.terminated || transition.Env.truncated then
         let obs, _info = Env.reset env () in
         current_obs := obs
     done);

  (* Training loop *)
  while !timesteps < total_timesteps do
    episode := !episode + 1;

    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let done_flag = ref false in
    let episode_reward = ref 0.0 in
    let episode_length = ref 0 in

    (* Compute epsilon for this episode *)
    let epsilon =
      t.config.epsilon_end
      +. (t.config.epsilon_start -. t.config.epsilon_end)
         *. exp (-.float_of_int !timesteps /. t.config.epsilon_decay)
    in

    let total_loss = ref 0.0 in
    let total_q = ref 0.0 in
    let update_count = ref 0 in

    (* Collect episode *)
    while not !done_flag do
      let action = predict t !current_obs ~epsilon in
      let transition = Env.step env action in

      add_transition t ~observation:!current_obs ~action
        ~reward:transition.Env.reward
        ~next_observation:transition.Env.observation
        ~terminated:transition.Env.terminated
        ~truncated:transition.Env.truncated;

      episode_reward := !episode_reward +. transition.Env.reward;
      episode_length := !episode_length + 1;
      timesteps := !timesteps + 1;

      (* Update Q-network *)
      let loss, avg_q = update t in
      if loss > 0.0 then (
        total_loss := !total_loss +. loss;
        total_q := !total_q +. avg_q;
        update_count := !update_count + 1);

      current_obs := transition.Env.observation;
      done_flag := transition.Env.terminated || transition.Env.truncated
    done;

    (* Update target network periodically *)
    if !episode mod t.config.target_update_freq = 0 then update_target_network t;

    (* Compute metrics *)
    let avg_loss =
      if !update_count > 0 then !total_loss /. float_of_int !update_count
      else 0.0
    in
    let avg_q_value =
      if !update_count > 0 then !total_q /. float_of_int !update_count else 0.0
    in

    let metrics =
      {
        episode_return = !episode_reward;
        episode_length = !episode_length;
        epsilon;
        avg_q_value;
        loss = avg_loss;
      }
    in

    (* Call callback *)
    let continue = callback ~episode:!episode ~metrics in
    if not continue then timesteps := total_timesteps
  done;

  t

let save t dir =
  if not (Sys.file_exists dir) then Unix.mkdir dir 0o755;
  let checkpointer = Kaun.Checkpoint.Checkpointer.create () in
  Kaun.Checkpoint.Checkpointer.save_file checkpointer
    ~path:(Filename.concat dir "q_params.safetensors")
    ~params:t.q_params ();
  Kaun.Checkpoint.Checkpointer.save_file checkpointer
    ~path:(Filename.concat dir "target_params.safetensors")
    ~params:t.target_params ();
  let rng_seed = Rune.Rng.to_int t.rng in
  let metadata = `Assoc [
    ("n_actions", `Int t.n_actions);
    ("rng_seed", `Int rng_seed);
    ("learning_rate", `Float t.config.learning_rate);
    ("gamma", `Float t.config.gamma);
    ("epsilon_start", `Float t.config.epsilon_start);
    ("epsilon_end", `Float t.config.epsilon_end);
    ("epsilon_decay", `Float t.config.epsilon_decay);
    ("batch_size", `Int t.config.batch_size);
    ("buffer_capacity", `Int t.config.buffer_capacity);
    ("target_update_freq", `Int t.config.target_update_freq);
  ] in
  let metadata_path = Filename.concat dir "metadata.json" in
  let oc = open_out metadata_path in
  Yojson.Basic.to_channel oc metadata;
  close_out oc

let load dir = 
  let metadata_path = Filename.concat dir "metadata.json" in
  let metadata = Yojson.Basic.from_file metadata_path in
  let open Yojson.Basic.Util in
  let config = {
    learning_rate = metadata |> member "learning_rate" |> to_float;
    gamma = metadata |> member "gamma" |> to_float;
    epsilon_start = metadata |> member "epsilon_start" |> to_float;
    epsilon_end = metadata |> member "epsilon_end" |> to_float;
    epsilon_decay = metadata |> member "epsilon_decay" |> to_float;
    batch_size = metadata |> member "batch_size" |> to_int;
    buffer_capacity = metadata |> member "buffer_capacity" |> to_int;
    target_update_freq = metadata |> member "target_update_freq" |> to_int;
  } in
  let n_actions = metadata |> member "n_actions" |> to_int in
  let rng_seed = metadata |> member "rng_seed" |> to_int in
  let rng = Rune.Rng.key rng_seed in
  let checkpointer = Kaun.Checkpoint.Checkpointer.create () in

  (* Example: fixed architecture for 2 input features and n_actions *)
  let q_network =
    Kaun.Layer.sequential [
      Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:n_actions ();
    ]
  in

  let q_params = Kaun.Checkpoint.Checkpointer.restore_file checkpointer
    ~path:(Filename.concat dir "q_params.safetensors")
    ~dtype:Rune.float32
  in
  let target_params = Kaun.Checkpoint.Checkpointer.restore_file checkpointer
    ~path:(Filename.concat dir "target_params.safetensors")
    ~dtype:Rune.float32
  in
  let optimizer = Optimizer.adam ~lr:config.learning_rate () in
  let opt_state = optimizer.init q_params in
  let replay_buffer = Fehu.Buffer.Replay.create ~capacity:config.buffer_capacity in
  {
    q_network;
    q_params;
    target_network = q_network;
    target_params;
    optimizer;
    opt_state;
    replay_buffer;
    rng;
    n_actions;
    config;
  }