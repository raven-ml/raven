type ('obs, 'act) transition = {
  observation : 'obs;
  action : 'act;
  reward : float;
  next_observation : 'obs;
  terminated : bool;
  truncated : bool;
}

type ('obs, 'act) step = {
  observation : 'obs;
  action : 'act;
  reward : float;
  terminated : bool;
  truncated : bool;
  value : float option;
  log_prob : float option;
}

module Replay = struct
  type ('obs, 'act) t = {
    capacity : int;
    mutable size : int;
    mutable pos : int;
    mutable observations : 'obs array;
    mutable actions : 'act array;
    rewards : float array;
    mutable next_observations : 'obs array;
    terminateds : bool array;
    truncateds : bool array;
  }

  let create ~capacity =
    if capacity <= 0 then
      invalid_arg "Buffer.Replay.create: capacity must be positive";
    {
      capacity;
      size = 0;
      pos = 0;
      observations = [||];
      actions = [||];
      rewards = Array.make capacity 0.0;
      next_observations = [||];
      terminateds = Array.make capacity false;
      truncateds = Array.make capacity false;
    }

  let ensure_initialized buffer (transition : ('obs, 'act) transition) =
    if Array.length buffer.observations = 0 then (
      buffer.observations <- Array.make buffer.capacity transition.observation;
      buffer.actions <- Array.make buffer.capacity transition.action;
      buffer.next_observations <-
        Array.make buffer.capacity transition.next_observation)

  let write buffer (transition : ('obs, 'act) transition) =
    buffer.observations.(buffer.pos) <- transition.observation;
    buffer.actions.(buffer.pos) <- transition.action;
    buffer.rewards.(buffer.pos) <- transition.reward;
    buffer.next_observations.(buffer.pos) <- transition.next_observation;
    buffer.terminateds.(buffer.pos) <- transition.terminated;
    buffer.truncateds.(buffer.pos) <- transition.truncated;
    buffer.pos <- (buffer.pos + 1) mod buffer.capacity;
    if buffer.size < buffer.capacity then buffer.size <- buffer.size + 1

  let add buffer (transition : ('obs, 'act) transition) =
    ensure_initialized buffer transition;
    write buffer transition

  let add_many buffer (transitions : ('obs, 'act) transition array) =
    if Array.length transitions = 0 then ()
    else (
      ensure_initialized buffer transitions.(0);
      Array.iter (write buffer) transitions)

  let sample_indices buffer ~rng ~batch_size =
    if buffer.size = 0 then invalid_arg "Buffer.Replay.sample: buffer is empty";
    if batch_size <= 0 then
      invalid_arg "Buffer.Replay.sample: batch_size must be positive";
    let actual_batch_size = min batch_size buffer.size in
    let raw_indices =
      Rune.Rng.randint rng ~min:0 ~max:buffer.size [| actual_batch_size |]
    in
    let indices_arr : Int32.t array = Rune.to_array raw_indices in
    ( actual_batch_size,
      Array.init actual_batch_size (fun i -> Int32.to_int indices_arr.(i)) )

  let sample_arrays buffer ~rng ~batch_size =
    let actual_batch_size, indices = sample_indices buffer ~rng ~batch_size in
    let gather arr =
      Array.init actual_batch_size (fun i -> arr.(indices.(i)))
    in
    ( gather buffer.observations,
      gather buffer.actions,
      Array.init actual_batch_size (fun i -> buffer.rewards.(indices.(i))),
      gather buffer.next_observations,
      Array.init actual_batch_size (fun i -> buffer.terminateds.(indices.(i))),
      Array.init actual_batch_size (fun i -> buffer.truncateds.(indices.(i))) )

  let sample buffer ~rng ~batch_size =
    let ( observations,
          actions,
          rewards,
          next_observations,
          terminateds,
          truncateds ) =
      sample_arrays buffer ~rng ~batch_size
    in
    let batch_size = Array.length rewards in
    Array.init batch_size (fun i ->
        {
          observation = observations.(i);
          action = actions.(i);
          reward = rewards.(i);
          next_observation = next_observations.(i);
          terminated = terminateds.(i);
          truncated = truncateds.(i);
        })

  let size buffer = buffer.size
  let is_full buffer = buffer.size = buffer.capacity

  let clear buffer =
    buffer.size <- 0;
    buffer.pos <- 0
end

module Rollout = struct
  type ('obs, 'act) step_with_advantage = {
    step : ('obs, 'act) step;
    mutable advantage : float option;
    mutable return_ : float option;
  }

  type ('obs, 'act) t = {
    capacity : int;
    mutable size : int;
    steps : ('obs, 'act) step_with_advantage array option ref;
  }

  let create ~capacity =
    if capacity <= 0 then
      invalid_arg "Buffer.Rollout.create: capacity must be positive";
    { capacity; size = 0; steps = ref None }

  let add buffer step =
    if buffer.size >= buffer.capacity then
      invalid_arg "Buffer.Rollout.add: buffer is full";
    match !(buffer.steps) with
    | None ->
        let arr =
          Array.make buffer.capacity { step; advantage = None; return_ = None }
        in
        arr.(0) <- { step; advantage = None; return_ = None };
        buffer.steps := Some arr;
        buffer.size <- 1
    | Some arr ->
        arr.(buffer.size) <- { step; advantage = None; return_ = None };
        buffer.size <- buffer.size + 1

  let compute_advantages buffer ~last_value ~last_done ~gamma ~gae_lambda =
    match !(buffer.steps) with
    | None -> ()
    | Some steps ->
        if buffer.size = 0 then ()
        else
          let last_gae_lam = ref 0.0 in
          for t = buffer.size - 1 downto 0 do
            let step = steps.(t).step in
            let value = Option.value step.value ~default:0.0 in
            let terminal_step = step.terminated || step.truncated in
            let next_value =
              if t = buffer.size - 1 then
                if last_done || terminal_step then 0.0 else last_value
              else Option.value steps.(t + 1).step.value ~default:0.0
            in
            let next_non_terminal =
              if t = buffer.size - 1 then
                if last_done || terminal_step then 0.0 else 1.0
              else if terminal_step then 0.0
              else 1.0
            in
            let delta =
              step.reward +. (gamma *. next_value *. next_non_terminal) -. value
            in
            last_gae_lam :=
              delta
              +. (gamma *. gae_lambda *. next_non_terminal *. !last_gae_lam);
            steps.(t).advantage <- Some !last_gae_lam;
            steps.(t).return_ <- Some (!last_gae_lam +. value)
          done

  let get buffer =
    match !(buffer.steps) with
    | None -> ([||], [||], [||])
    | Some steps ->
        let result_steps = Array.init buffer.size (fun i -> steps.(i).step) in
        let advantages =
          Array.init buffer.size (fun i ->
              Option.value steps.(i).advantage ~default:0.0)
        in
        let returns =
          Array.init buffer.size (fun i ->
              Option.value steps.(i).return_ ~default:0.0)
        in
        buffer.steps := None;
        buffer.size <- 0;
        (result_steps, advantages, returns)

  let size buffer = buffer.size
  let is_full buffer = buffer.size = buffer.capacity

  let clear buffer =
    buffer.steps := None;
    buffer.size <- 0
end
