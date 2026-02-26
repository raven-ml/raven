(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

type obs = (float, Rune.float32_elt) Rune.t
type act = (int32, Rune.int32_elt) Rune.t
type render = string

(* Physics constants matching Gymnasium CartPole-v1 *)

let gravity = 9.8
let masscart = 1.0
let masspole = 0.1
let total_mass = masscart +. masspole
let half_pole_length = 0.5
let polemass_length = masspole *. half_pole_length
let force_mag = 10.0
let tau = 0.02

(* Termination thresholds *)

let theta_threshold = 12. *. Float.pi /. 180.
let x_threshold = 2.4
let max_steps = 500

(* Float32-representable large bound for "unbounded" dimensions *)
let f32_max = 3.4028235e38

let observation_space =
  Space.Box.create
    ~low:[| -4.8; -.f32_max; -.theta_threshold *. 2.; -.f32_max |]
    ~high:[| 4.8; f32_max; theta_threshold *. 2.; f32_max |]

let action_space = Space.Discrete.create 2

let make_obs x x_dot theta theta_dot =
  Rune.create Rune.float32 [| 4 |] [| x; x_dot; theta; theta_dot |]

let make ?render_mode () =
  let x = ref 0.0 in
  let x_dot = ref 0.0 in
  let theta = ref 0.0 in
  let theta_dot = ref 0.0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    let random_state () =
      let r = Rune.rand Rune.float32 [| 1 |] in
      let v = (Rune.to_array r).(0) in
      (v -. 0.5) *. 0.1
    in
    x := random_state ();
    x_dot := random_state ();
    theta := random_state ();
    theta_dot := random_state ();
    steps := 0;
    (make_obs !x !x_dot !theta !theta_dot, Info.empty)
  in
  let step _env action =
    let force =
      if Space.Discrete.to_int action = 1 then force_mag else -.force_mag
    in
    let costheta = cos !theta in
    let sintheta = sin !theta in
    let temp =
      (force +. (polemass_length *. !theta_dot *. !theta_dot *. sintheta))
      /. total_mass
    in
    let thetaacc =
      ((gravity *. sintheta) -. (costheta *. temp))
      /. (half_pole_length
         *. ((4.0 /. 3.0) -. (masspole *. costheta *. costheta /. total_mass)))
    in
    let xacc =
      temp -. (polemass_length *. thetaacc *. costheta /. total_mass)
    in
    x := !x +. (tau *. !x_dot);
    x_dot := !x_dot +. (tau *. xacc);
    theta := !theta +. (tau *. !theta_dot);
    theta_dot := !theta_dot +. (tau *. thetaacc);
    incr steps;
    let terminated =
      !x < -.x_threshold || !x > x_threshold || !theta < -.theta_threshold
      || !theta > theta_threshold
    in
    let truncated = (not terminated) && !steps >= max_steps in
    let reward = if terminated then 0.0 else 1.0 in
    let info = Info.set "steps" (Info.int !steps) Info.empty in
    Env.step_result
      ~observation:(make_obs !x !x_dot !theta !theta_dot)
      ~reward ~terminated ~truncated ~info ()
  in
  let render () =
    Some
      (Printf.sprintf
         "CartPole: x=%.3f, x_dot=%.3f, theta=%.3f\xc2\xb0, theta_dot=%.3f, \
          steps=%d"
         !x !x_dot
         (!theta *. 180. /. Float.pi)
         !theta_dot !steps)
  in
  Env.create ?render_mode ~render_modes:[ "ansi" ] ~id:"CartPole-v1"
    ~observation_space ~action_space ~reset ~step ~render ()
