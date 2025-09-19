module Space = struct
  type 'dev t =
    | Discrete of int
    | Box of {
        low : (float, Rune.float32_elt) Rune.t;
        high : (float, Rune.float32_elt) Rune.t;
        shape : int array;
      }
    | Multi_discrete of int array

  let sample ~rng space =
    match space with
    | Discrete n ->
        let action_tensor = Rune.Rng.randint rng ~min:0 ~max:n [| 1 |] in
        Rune.cast Rune.float32 action_tensor
    | Box { low; high; shape } ->
        let uniform = Rune.Rng.uniform rng Rune.float32 shape in
        let range = Rune.sub high low in
        Rune.add low (Rune.mul uniform range)
    | Multi_discrete dims ->
        let actions =
          Array.mapi
            (fun i d ->
              let key = Rune.Rng.fold_in rng i in
              let action_tensor =
                Rune.Rng.randint key ~min:0 ~max:d [| 1 |]
              in
              let action_array = Rune.to_array action_tensor in
              Int32.to_float action_array.(0))
            dims
        in
        Rune.create Rune.float32 [| Array.length dims |] actions

  let contains space x =
    match space with
    | Discrete n ->
        let v = Rune.to_array x in
        let v_int = int_of_float v.(0) in
        v_int >= 0 && v_int < n
    | Box { low; high; _ } ->
        let ge_low = Rune.all (Rune.greater_equal x low) in
        let le_high = Rune.all (Rune.less_equal x high) in
        let ge_low_val = Rune.to_array ge_low in
        let le_high_val = Rune.to_array le_high in
        ge_low_val.(0) > 0 && le_high_val.(0) > 0
    | Multi_discrete dims ->
        let values = Rune.to_array x |> Array.map int_of_float in
        Array.for_all2 (fun v d -> v >= 0 && v < d) values dims

  let shape = function
    | Discrete _ -> [| 1 |]
    | Box { shape; _ } -> shape
    | Multi_discrete dims -> [| Array.length dims |]
end

module Env = struct
  type info = (string * Yojson.Basic.t) list

  type 'dev t = {
    observation_space : 'dev Space.t;
    action_space : 'dev Space.t;
    reset : ?seed:int -> unit -> (float, Rune.float32_elt) Rune.t * info;
    step :
      (float, Rune.float32_elt) Rune.t ->
      (float, Rune.float32_elt) Rune.t * float * bool * bool * info;
    render : unit -> unit;
    close : unit -> unit;
  }

  let make ~observation_space ~action_space ~reset ~step
      ?(render = fun () -> ()) ?(close = fun () -> ()) () =
    { observation_space; action_space; reset; step; render; close }
end

module Buffer = struct
  type 'dev transition = {
    obs : (float, Rune.float32_elt) Rune.t;
    action : (float, Rune.float32_elt) Rune.t;
    reward : float;
    next_obs : (float, Rune.float32_elt) Rune.t;
    terminated : bool;
  }

  type 'dev t = {
    capacity : int;
    mutable buffer : 'dev transition array;
    mutable position : int;
    mutable size : int;
  }

  let create ~capacity = { capacity; buffer = [||]; position = 0; size = 0 }

  let add t transition =
    if Array.length t.buffer = 0 then
      t.buffer <- Array.make t.capacity transition
    else t.buffer.(t.position) <- transition;
    t.position <- (t.position + 1) mod t.capacity;
    t.size <- min (t.size + 1) t.capacity

  let sample t ~rng ~batch_size =
    if t.size < batch_size then failwith "Buffer: Not enough samples";
    let indices =
      Rune.Rng.randint rng ~min:0 ~max:t.size [| batch_size |]
    in
    let indices_array = Rune.to_array indices |> Array.map Int32.to_int in
    Array.map (fun idx -> t.buffer.(idx)) indices_array

  let size t = t.size
  let is_full t = t.size = t.capacity
end

module Training = struct
  type stats = {
    episode_reward : float;
    episode_length : int;
    total_timesteps : int;
    n_episodes : int;
    mean_reward : float;
    std_reward : float;
  }

  let evaluate env ~policy ~n_eval_episodes =
    let open Env in
    let rewards = ref [] in
    let lengths = ref [] in

    for _ = 1 to n_eval_episodes do
      let obs, _ = env.reset () in
      let episode_reward = ref 0.0 in
      let episode_length = ref 0 in
      let finished = ref false in

      while not !finished do
        let action = policy obs in
        let _next_obs, reward, terminated, truncated, _ = env.step action in
        episode_reward := !episode_reward +. reward;
        episode_length := !episode_length + 1;
        finished := terminated || truncated
      done;

      rewards := !episode_reward :: !rewards;
      lengths := !episode_length :: !lengths
    done;

    let mean_reward =
      List.fold_left ( +. ) 0.0 !rewards /. float_of_int n_eval_episodes
    in
    let variance =
      List.fold_left
        (fun acc r -> acc +. ((r -. mean_reward) ** 2.0))
        0.0 !rewards
      /. float_of_int n_eval_episodes
    in

    {
      episode_reward = List.hd !rewards;
      episode_length = List.hd !lengths;
      total_timesteps = List.fold_left ( + ) 0 !lengths;
      n_episodes = n_eval_episodes;
      mean_reward;
      std_reward = sqrt variance;
    }

  let compute_gae ~rewards ~values ~dones ~gamma ~gae_lambda =
    let n = Array.length rewards in
    let advantages = Array.make n 0.0 in
    let returns = Array.make n 0.0 in

    let gae = ref 0.0 in
    for i = n - 1 downto 0 do
      let next_value = if i = n - 1 then 0.0 else values.(i + 1) in
      let next_non_terminal = if i = n - 1 || dones.(i) then 0.0 else 1.0 in

      let delta =
        rewards.(i) +. (gamma *. next_value *. next_non_terminal) -. values.(i)
      in
      gae := delta +. (gamma *. gae_lambda *. next_non_terminal *. !gae);
      advantages.(i) <- !gae;
      returns.(i) <- advantages.(i) +. values.(i)
    done;

    (advantages, returns)

  let compute_returns ~rewards ~dones ~gamma =
    let n = Array.length rewards in
    let returns = Array.make n 0.0 in
    let running_return = ref 0.0 in

    for i = n - 1 downto 0 do
      if dones.(i) then running_return := 0.0;
      running_return := rewards.(i) +. (gamma *. !running_return);
      returns.(i) <- !running_return
    done;

    returns

  let normalize x ?(eps = 1e-8) () =
    let mean = Rune.mean x ~axes:[| 0 |] ~keepdims:true in
    let std = Rune.std x ~axes:[| 0 |] ~keepdims:true in
    let std_eps = Rune.add std (Rune.scalar Rune.float32 eps) in
    Rune.div (Rune.sub x mean) std_eps
end

module Envs = struct
  let cartpole () =
    let gravity = 9.8 in
    let masscart = 1.0 in
    let masspole = 0.1 in
    let total_mass = masspole +. masscart in
    let length = 0.5 in
    let polemass_length = masspole *. length in
    let force_mag = 10.0 in
    let tau = 0.02 in

    let state = ref (Rune.zeros Rune.float32 [| 4 |]) in

    let observation_space =
      Space.Box
        {
          low =
            Rune.create Rune.float32 [| 4 |]
              [| -4.8; -.Float.max_float; -0.42; -.Float.max_float |];
          high =
            Rune.create Rune.float32 [| 4 |]
              [| 4.8; Float.max_float; 0.42; Float.max_float |];
          shape = [| 4 |];
        }
    in

    let action_space = Space.Discrete 2 in

    let reset ?seed () =
      let () = match seed with Some s -> Random.init s | None -> () in
      (* Generate uniform values between -0.05 and 0.05 *)
      let uniform_vals = Array.init 4 (fun _ -> Random.float 0.1 -. 0.05) in
      state := Rune.create Rune.float32 [| 4 |] uniform_vals;
      (!state, [])
    in

    let step action =
      let action_array = Rune.to_array action in
      let action_val = int_of_float action_array.(0) in
      let force = if action_val = 1 then force_mag else -.force_mag in

      let s = Rune.to_array !state in
      let x = s.(0) in
      let x_dot = s.(1) in
      let theta = s.(2) in
      let theta_dot = s.(3) in

      let costheta = cos theta in
      let sintheta = sin theta in

      let temp =
        (force +. (polemass_length *. theta_dot *. theta_dot *. sintheta))
        /. total_mass
      in
      let thetaacc =
        ((gravity *. sintheta) -. (costheta *. temp))
        /. (length
           *. ((4.0 /. 3.0) -. (masspole *. costheta *. costheta /. total_mass))
           )
      in
      let xacc =
        temp -. (polemass_length *. thetaacc *. costheta /. total_mass)
      in

      let x' = x +. (tau *. x_dot) in
      let x_dot' = x_dot +. (tau *. xacc) in
      let theta' = theta +. (tau *. theta_dot) in
      let theta_dot' = theta_dot +. (tau *. thetaacc) in

      state :=
        Rune.create Rune.float32 [| 4 |]
          [| x'; x_dot'; theta'; theta_dot' |];

      let terminated =
        x' < -2.4 || x' > 2.4 || theta' < -0.21 || theta' > 0.21
      in

      (!state, 1.0, terminated, false, [])
    in

    Env.make ~observation_space ~action_space ~reset ~step ()

  let mountain_car () =
    let min_position = -1.2 in
    let max_position = 0.6 in
    let max_speed = 0.07 in
    let goal_position = 0.5 in

    let state = ref (Rune.zeros Rune.float32 [| 2 |]) in

    let observation_space =
      Space.Box
        {
          low =
            Rune.create Rune.float32 [| 2 |]
              [| min_position; -.max_speed |];
          high =
            Rune.create Rune.float32 [| 2 |]
              [| max_position; max_speed |];
          shape = [| 2 |];
        }
    in

    let action_space = Space.Discrete 3 in

    let reset ?seed () =
      let () = match seed with Some s -> Random.init s | None -> () in
      let position = Random.float 0.2 -. 0.6 in
      let velocity = 0.0 in
      state := Rune.create Rune.float32 [| 2 |] [| position; velocity |];
      (!state, [])
    in

    let step action =
      let action_array = Rune.to_array action in
      let action_val = int_of_float action_array.(0) in
      let s = Rune.to_array !state in
      let position = s.(0) in
      let velocity = s.(1) in

      let force = 0.001 in
      let gravity = 0.0025 in

      let force_applied = float_of_int (action_val - 1) *. force in
      let velocity' =
        velocity +. force_applied -. (gravity *. cos (3.0 *. position))
      in
      let velocity' = max (-.max_speed) (min max_speed velocity') in
      let position' = position +. velocity' in
      let position' = max min_position (min max_position position') in

      let velocity' =
        if position' = min_position && velocity' < 0.0 then 0.0 else velocity'
      in

      state :=
        Rune.create Rune.float32 [| 2 |] [| position'; velocity' |];

      let terminated = position' >= goal_position in
      let reward = if terminated then 0.0 else -1.0 in

      (!state, reward, terminated, false, [])
    in

    Env.make ~observation_space ~action_space ~reset ~step ()

  let pendulum () =
    let max_speed = 8.0 in
    let max_torque = 2.0 in
    let dt = 0.05 in
    let g = 10.0 in
    let m = 1.0 in
    let l = 1.0 in

    let state = ref (Rune.zeros Rune.float32 [| 2 |]) in

    let observation_space =
      Space.Box
        {
          low =
            Rune.create Rune.float32 [| 3 |]
              [| -1.0; -1.0; -.max_speed |];
          high =
            Rune.create Rune.float32 [| 3 |] [| 1.0; 1.0; max_speed |];
          shape = [| 3 |];
        }
    in

    let action_space =
      Space.Box
        {
          low = Rune.scalar Rune.float32 (-.max_torque);
          high = Rune.scalar Rune.float32 max_torque;
          shape = [| 1 |];
        }
    in

    let reset ?seed () =
      let () = match seed with Some s -> Random.init s | None -> () in
      let theta = Random.float (2.0 *. Float.pi) -. Float.pi in
      let theta_dot = Random.float 2.0 -. 1.0 in
      state := Rune.create Rune.float32 [| 2 |] [| theta; theta_dot |];
      let obs =
        Rune.create Rune.float32 [| 3 |]
          [| cos theta; sin theta; theta_dot |]
      in
      (obs, [])
    in

    let step action =
      let action_array = Rune.to_array action in
      let u = action_array.(0) in
      let u = max (-.max_torque) (min max_torque u) in

      let s = Rune.to_array !state in
      let theta = s.(0) in
      let theta_dot = s.(1) in

      let costs =
        (theta ** 2.0) +. (0.1 *. (theta_dot ** 2.0)) +. (0.001 *. (u ** 2.0))
      in

      let theta_dot' =
        theta_dot
        +. ((3.0 *. g /. (2.0 *. l) *. sin theta)
           +. (3.0 /. (m *. (l ** 2.0)) *. u))
           *. dt
      in
      let theta_dot' = max (-.max_speed) (min max_speed theta_dot') in
      let theta' = theta +. (theta_dot' *. dt) in

      state := Rune.create Rune.float32 [| 2 |] [| theta'; theta_dot' |];

      let obs =
        Rune.create Rune.float32 [| 3 |]
          [| cos theta'; sin theta'; theta_dot' |]
      in

      (obs, -.costs, false, false, [])
    in

    Env.make ~observation_space ~action_space ~reset ~step ()
end
