(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

type obs = (float, Rune.float32_elt) Rune.t
type act = (int32, Rune.int32_elt) Rune.t
type render = string

let step_size = 1.0
let max_position = 10.0
let max_steps = 200

let observation_space =
  Space.Box.create ~low:[| -.max_position |] ~high:[| max_position |]

let action_space = Space.Discrete.create 2
let make_obs position = Rune.create Rune.float32 [| 1 |] [| position |]

let render_ansi position =
  let offset = int_of_float (position +. max_position) in
  let offset = max 0 (min 20 offset) in
  let buffer = Bytes.make 21 '.' in
  Bytes.set buffer offset 'o';
  Printf.sprintf "Position: %+.2f\n|%s|" position (Bytes.to_string buffer)

let make ?render_mode ~rng () =
  let position = ref 0.0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    position := 0.0;
    steps := 0;
    (make_obs 0.0, Info.empty)
  in
  let step _env action =
    let direction =
      if Space.Discrete.to_int action = 0 then -.step_size else step_size
    in
    let updated = !position +. direction in
    let clamped = Float.min max_position (Float.max (-.max_position) updated) in
    position := clamped;
    incr steps;
    let terminated = Float.abs clamped >= max_position in
    let truncated = (not terminated) && !steps >= max_steps in
    let reward = -.Float.abs clamped in
    let info = Info.set "steps" (Info.int !steps) Info.empty in
    Env.step_result ~observation:(make_obs clamped) ~reward ~terminated
      ~truncated ~info ()
  in
  let render () = Some (render_ansi !position) in
  Env.create ?render_mode ~render_modes:[ "ansi" ] ~id:"RandomWalk-v0" ~rng
    ~observation_space ~action_space ~reset ~step ~render ()
