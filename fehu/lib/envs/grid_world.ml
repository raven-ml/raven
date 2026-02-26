(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

type obs = (int32, Rune.int32_elt) Rune.t
type act = (int32, Rune.int32_elt) Rune.t
type render = Text of string | Image of Render.image

let grid_size = 5
let max_steps = 200
let observation_space = Space.Multi_discrete.create [| grid_size; grid_size |]
let action_space = Space.Discrete.create 4
let is_goal row col = row = grid_size - 1 && col = grid_size - 1
let is_obstacle row col = row = 2 && col = 2

let is_valid row col =
  row >= 0 && row < grid_size && col >= 0 && col < grid_size
  && not (is_obstacle row col)

let make_obs row col =
  Rune.create Rune.int32 [| 2 |] [| Int32.of_int row; Int32.of_int col |]

(* ANSI rendering *)

let render_text row col =
  let buffer = Bytes.make (grid_size * grid_size) '.' in
  Bytes.set buffer ((row * grid_size) + col) 'A';
  Bytes.set buffer (((grid_size - 1) * grid_size) + (grid_size - 1)) 'G';
  Bytes.set buffer ((2 * grid_size) + 2) '#';
  let rows =
    List.init grid_size (fun r ->
        Bytes.sub_string buffer (r * grid_size) grid_size)
  in
  Format.asprintf "Position: (%d, %d)@.%a" row col
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@.")
       Format.pp_print_string)
    rows

(* RGB rendering *)

let cell_size = 32
let frame_width = grid_size * cell_size
let frame_height = grid_size * cell_size

let fill_rect data ~x0 ~y0 ~w ~h ~r ~g ~b =
  for dy = 0 to h - 1 do
    let row_offset = (y0 + dy) * frame_width * 3 in
    for dx = 0 to w - 1 do
      let base = row_offset + ((x0 + dx) * 3) in
      Bigarray.Array1.unsafe_set data base r;
      Bigarray.Array1.unsafe_set data (base + 1) g;
      Bigarray.Array1.unsafe_set data (base + 2) b
    done
  done

let render_image row col =
  let len = frame_width * frame_height * 3 in
  let data =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout len
  in
  fill_rect data ~x0:0 ~y0:0 ~w:frame_width ~h:frame_height ~r:30 ~g:33 ~b:36;
  for gr = 0 to grid_size - 1 do
    for gc = 0 to grid_size - 1 do
      let x0 = gc * cell_size in
      let y0 = gr * cell_size in
      fill_rect data ~x0 ~y0 ~w:cell_size ~h:cell_size ~r:44 ~g:48 ~b:52;
      fill_rect data ~x0:(x0 + 1) ~y0:(y0 + 1) ~w:(cell_size - 2)
        ~h:(cell_size - 2) ~r:54 ~g:60 ~b:65
    done
  done;
  let draw_cell cr cc ~r ~g ~b =
    fill_rect data
      ~x0:((cc * cell_size) + 2)
      ~y0:((cr * cell_size) + 2)
      ~w:(cell_size - 4) ~h:(cell_size - 4) ~r ~g ~b
  in
  draw_cell row col ~r:78 ~g:162 ~b:196;
  draw_cell (grid_size - 1) (grid_size - 1) ~r:76 ~g:175 ~b:80;
  draw_cell 2 2 ~r:200 ~g:80 ~b:80;
  Render.image ~width:frame_width ~height:frame_height data

let make ?render_mode () =
  let row = ref 0 in
  let col = ref 0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    row := 0;
    col := 0;
    steps := 0;
    (make_obs 0 0, Info.empty)
  in
  let step _env action =
    let r, c = (!row, !col) in
    let nr, nc =
      match Space.Discrete.to_int action with
      | 0 -> (r - 1, c)
      | 1 -> (r + 1, c)
      | 2 -> (r, c - 1)
      | 3 -> (r, c + 1)
      | _ -> (r, c)
    in
    let nr, nc = if is_valid nr nc then (nr, nc) else (r, c) in
    row := nr;
    col := nc;
    incr steps;
    let terminated = is_goal nr nc in
    let truncated = (not terminated) && !steps >= max_steps in
    let reward = if terminated then 10.0 else -1.0 in
    let info = Info.set "steps" (Info.int !steps) Info.empty in
    Env.step_result ~observation:(make_obs nr nc) ~reward ~terminated ~truncated
      ~info ()
  in
  let render_mode_val = render_mode in
  let render () =
    match render_mode_val with
    | Some `Rgb_array -> Some (Image (render_image !row !col))
    | _ -> Some (Text (render_text !row !col))
  in
  Env.create ?render_mode ~render_modes:[ "ansi"; "rgb_array" ]
    ~id:"GridWorld-v0" ~observation_space ~action_space ~reset ~step ~render ()
