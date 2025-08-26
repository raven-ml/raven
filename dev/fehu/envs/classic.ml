(** Classic control environments for RL *)

open Fehu

let cartpole () =
  let gravity = 9.8 in
  let masscart = 1.0 in
  let masspole = 0.1 in
  let total_mass = masspole +. masscart in
  let length = 0.5 in
  let polemass_length = masspole *. length in
  let force_mag = 10.0 in
  let tau = 0.02 in

  let state = ref (Rune.zeros Rune.c Rune.float32 [| 4 |]) in

  let observation_space =
    Space.Box
      {
        low =
          Rune.create Rune.c Rune.float32 [| 4 |]
            [| -4.8; -.Float.max_float; -0.42; -.Float.max_float |];
        high =
          Rune.create Rune.c Rune.float32 [| 4 |]
            [| 4.8; Float.max_float; 0.42; Float.max_float |];
        shape = [| 4 |];
      }
  in

  let action_space = Space.Discrete 2 in

  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    let uniform_vals = Array.init 4 (fun _ -> Random.float 0.1 -. 0.05) in
    state := Rune.create Rune.c Rune.float32 [| 4 |] uniform_vals;
    (!state, [])
  in

  let step action =
    let action_array = Rune.unsafe_to_array action in
    let action_val = int_of_float action_array.(0) in
    let force = if action_val = 1 then force_mag else -.force_mag in

    let s = Rune.unsafe_to_array !state in
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
      Rune.create Rune.c Rune.float32 [| 4 |]
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

  let state = ref (Rune.zeros Rune.c Rune.float32 [| 2 |]) in

  let observation_space =
    Space.Box
      {
        low =
          Rune.create Rune.c Rune.float32 [| 2 |]
            [| min_position; -.max_speed |];
        high =
          Rune.create Rune.c Rune.float32 [| 2 |]
            [| max_position; max_speed |];
        shape = [| 2 |];
      }
  in

  let action_space = Space.Discrete 3 in

  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    let position = Random.float 0.2 -. 0.6 in
    let velocity = 0.0 in
    state := Rune.create Rune.c Rune.float32 [| 2 |] [| position; velocity |];
    (!state, [])
  in

  let step action =
    let action_array = Rune.unsafe_to_array action in
    let action_val = int_of_float action_array.(0) in
    let s = Rune.unsafe_to_array !state in
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
      Rune.create Rune.c Rune.float32 [| 2 |] [| position'; velocity' |];

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

  let state = ref (Rune.zeros Rune.c Rune.float32 [| 2 |]) in

  let observation_space =
    Space.Box
      {
        low =
          Rune.create Rune.c Rune.float32 [| 3 |]
            [| -1.0; -1.0; -.max_speed |];
        high =
          Rune.create Rune.c Rune.float32 [| 3 |] [| 1.0; 1.0; max_speed |];
        shape = [| 3 |];
      }
  in

  let action_space =
    Space.Box
      {
        low = Rune.scalar Rune.c Rune.float32 (-.max_torque);
        high = Rune.scalar Rune.c Rune.float32 max_torque;
        shape = [| 1 |];
      }
  in

  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    let theta = Random.float (2.0 *. Float.pi) -. Float.pi in
    let theta_dot = Random.float 2.0 -. 1.0 in
    state := Rune.create Rune.c Rune.float32 [| 2 |] [| theta; theta_dot |];
    let obs =
      Rune.create Rune.c Rune.float32 [| 3 |]
        [| cos theta; sin theta; theta_dot |]
    in
    (obs, [])
  in

  let step action =
    let action_array = Rune.unsafe_to_array action in
    let u = action_array.(0) in
    let u = max (-.max_torque) (min max_torque u) in

    let s = Rune.unsafe_to_array !state in
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

    state := Rune.create Rune.c Rune.float32 [| 2 |] [| theta'; theta_dot' |];

    let obs =
      Rune.create Rune.c Rune.float32 [| 3 |]
        [| cos theta'; sin theta'; theta_dot' |]
    in

    (obs, -.costs, false, false, [])
  in

  Env.make ~observation_space ~action_space ~reset ~step ()