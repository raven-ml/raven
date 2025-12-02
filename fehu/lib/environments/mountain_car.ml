open Fehu

type observation = (float, Rune.float32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = string

type state = {
  mutable position : float;
  mutable velocity : float;
  mutable steps : int;
  rng : Rune.Rng.key ref;
}

(* Environment parameters matching Gymnasium MountainCar-v0 *)
let min_position = -1.2
let max_position = 0.6
let max_speed = 0.07
let goal_position = 0.5
let goal_velocity = 0.0
let force = 0.001
let gravity = 0.0025

let observation_space =
  Space.Box.create
    ~low:[| min_position; -.max_speed |]
    ~high:[| max_position; max_speed |]

let action_space = Space.Discrete.create 3

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "ansi"
  |> Metadata.with_description
       (Some "Classic mountain car problem - drive up a steep hill")
  |> Metadata.add_author "Fehu"
  |> Metadata.with_version (Some "0.1.0")

let reset _env ?options:_ () state =
  (* Reset to random position in [-0.6, -0.4] with 0 velocity *)
  let keys = Rune.Rng.split !(state.rng) ~n:2 in
  state.rng := keys.(0);

  let r = Rune.rand Rune.float32 ~key:keys.(1) [| 1 |] in
  let v = (Rune.to_array r).(0) in
  state.position <- -0.6 +. (v *. 0.2);
  (* Random in [-0.6, -0.4] *)
  state.velocity <- 0.0;
  state.steps <- 0;

  let obs =
    Rune.create Rune.float32 [| 2 |] [| state.position; state.velocity |]
  in
  (obs, Info.empty)

let step _env action state =
  let action_value =
    let arr : Int32.t array = Rune.to_array action in
    Int32.to_int arr.(0)
  in

  (* Action: 0 = push left, 1 = no push, 2 = push right *)
  let force_direction = float_of_int (action_value - 1) in

  (* Update velocity based on action and gravity (cosine of 3*position) *)
  let velocity =
    state.velocity +. (force_direction *. force)
    -. (gravity *. cos (3.0 *. state.position))
  in
  let velocity = Float.max (-.max_speed) (Float.min velocity max_speed) in

  (* Update position *)
  let position = state.position +. velocity in
  let position = Float.max min_position (Float.min position max_position) in

  (* If hit left bound, reset velocity to 0 *)
  let velocity =
    if position = min_position && velocity < 0.0 then 0.0 else velocity
  in

  state.position <- position;
  state.velocity <- velocity;
  state.steps <- state.steps + 1;

  let terminated = position >= goal_position && velocity >= goal_velocity in
  let truncated = state.steps >= 200 in
  let reward = -1.0 in

  let obs =
    Rune.create Rune.float32 [| 2 |] [| state.position; state.velocity |]
  in

  let info = Info.set "steps" (Info.int state.steps) Info.empty in
  Env.transition ~observation:obs ~reward ~terminated ~truncated ~info ()

let render state =
  (* Simple ASCII visualization *)
  let normalized_pos =
    (state.position -. min_position) /. (max_position -. min_position)
  in
  let car_pos = int_of_float (normalized_pos *. 40.0) in
  let goal_pos =
    int_of_float
      ((goal_position -. min_position) /. (max_position -. min_position) *. 40.0)
  in

  let track = Bytes.make 41 '-' in
  Bytes.set track goal_pos 'G';
  Bytes.set track (max 0 (min 40 car_pos)) 'C';

  Printf.sprintf "MountainCar: [%s] pos=%.3f, vel=%.3f, steps=%d"
    (Bytes.to_string track) state.position state.velocity state.steps

let make ~rng () =
  let state = { position = 0.0; velocity = 0.0; steps = 0; rng = ref rng } in
  Env.create ~id:"MountainCar-v0" ~metadata ~rng ~observation_space
    ~action_space
    ~reset:(fun env ?options () -> reset env ?options () state)
    ~step:(fun env action -> step env action state)
    ~render:(fun _ -> Some (render state))
    ~close:(fun _ -> ())
    ()
