(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

type obs = (float, Nx.float32_elt) Nx.t
type act = (int32, Nx.int32_elt) Nx.t
type render = string

(* Physics constants matching Gymnasium MountainCar-v0 *)

let min_position = -1.2
let max_position = 0.6
let max_speed = 0.07
let goal_position = 0.5
let goal_velocity = 0.0
let force = 0.001
let gravity = 0.0025
let max_steps = 200

let observation_space =
  Space.Box.create
    ~low:[| min_position; -.max_speed |]
    ~high:[| max_position; max_speed |]

let action_space = Space.Discrete.create 3

let make_obs position velocity =
  Nx.create Nx.float32 [| 2 |] [| position; velocity |]

let make ?render_mode () =
  let position = ref 0.0 in
  let velocity = ref 0.0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    let r = Nx.rand Nx.float32 [| 1 |] in
    let v = (Nx.to_array r).(0) in
    position := -0.6 +. (v *. 0.2);
    velocity := 0.0;
    steps := 0;
    (make_obs !position !velocity, Info.empty)
  in
  let step _env action =
    let force_direction = float_of_int (Space.Discrete.to_int action - 1) in
    let vel =
      !velocity +. (force_direction *. force)
      -. (gravity *. cos (3.0 *. !position))
    in
    let vel = Float.max (-.max_speed) (Float.min vel max_speed) in
    let pos = !position +. vel in
    let pos = Float.max min_position (Float.min pos max_position) in
    let vel = if pos = min_position && vel < 0.0 then 0.0 else vel in
    position := pos;
    velocity := vel;
    incr steps;
    let terminated = pos >= goal_position && vel >= goal_velocity in
    let truncated = (not terminated) && !steps >= max_steps in
    let reward = -1.0 in
    let info = Info.set "steps" (Info.int !steps) Info.empty in
    Env.step_result ~observation:(make_obs pos vel) ~reward ~terminated
      ~truncated ~info ()
  in
  let render () =
    let normalized_pos =
      (!position -. min_position) /. (max_position -. min_position)
    in
    let car_pos = int_of_float (normalized_pos *. 40.0) in
    let goal_pos =
      int_of_float
        ((goal_position -. min_position)
        /. (max_position -. min_position)
        *. 40.0)
    in
    let track = Bytes.make 41 '-' in
    Bytes.set track goal_pos 'G';
    Bytes.set track (max 0 (min 40 car_pos)) 'C';
    Some
      (Printf.sprintf "MountainCar: [%s] pos=%.3f, vel=%.3f, steps=%d"
         (Bytes.to_string track) !position !velocity !steps)
  in
  Env.create ?render_mode ~render_modes:[ "ansi" ] ~id:"MountainCar-v0"
    ~observation_space ~action_space ~reset ~step ~render ()
