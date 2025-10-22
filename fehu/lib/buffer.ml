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

  let add buffer (transition : ('obs, 'act) transition) =
    let needs_init = buffer.size = 0 in
    if needs_init then (
      buffer.observations <- Array.make buffer.capacity transition.observation;
      buffer.actions <- Array.make buffer.capacity transition.action;
      buffer.next_observations <-
        Array.make buffer.capacity transition.next_observation);
    buffer.observations.(buffer.pos) <- transition.observation;
    buffer.actions.(buffer.pos) <- transition.action;
    buffer.rewards.(buffer.pos) <- transition.reward;
    buffer.next_observations.(buffer.pos) <- transition.next_observation;
    buffer.terminateds.(buffer.pos) <- transition.terminated;
    buffer.truncateds.(buffer.pos) <- transition.truncated;
    buffer.pos <- (buffer.pos + 1) mod buffer.capacity;
    buffer.size <- min (buffer.size + 1) buffer.capacity

  let sample buffer ~rng ~batch_size =
    if buffer.size = 0 then invalid_arg "Buffer.Replay.sample: buffer is empty";
    if batch_size <= 0 then
      invalid_arg "Buffer.Replay.sample: batch_size must be positive";
    let actual_batch_size = min batch_size buffer.size in
    let indices =
      Rune.Rng.randint rng ~min:0 ~max:buffer.size [| actual_batch_size |]
    in
    let indices_arr : Int32.t array = Rune.to_array indices in
    Array.init actual_batch_size (fun i ->
        let idx = Int32.to_int indices_arr.(i) in
        {
          observation = buffer.observations.(idx);
          action = buffer.actions.(idx);
          reward = buffer.rewards.(idx);
          next_observation = buffer.next_observations.(idx);
          terminated = buffer.terminateds.(idx);
          truncated = buffer.truncateds.(idx);
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
