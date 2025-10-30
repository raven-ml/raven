open Fehu

type observation = (int32, Rune.int32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = Render.t
type state = { mutable position : int * int; mutable steps : int }

let grid_size = 5
let observation_space = Space.Multi_discrete.create [| grid_size; grid_size |]
let action_space = Space.Discrete.create 4

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "ansi"
  |> Metadata.add_render_mode "rgb_array"
  |> Metadata.with_render_fps (Some 4)
  |> Metadata.with_description
       (Some "Simple 5x5 grid world with goal and obstacles")
  |> Metadata.add_author "Raven Developers"
  |> Metadata.with_version (Some "0.1.0")

let is_goal (r, c) = r = 4 && c = 4
let is_obstacle (r, c) = r = 2 && c = 2

let is_valid_pos (r, c) =
  r >= 0 && r < grid_size && c >= 0 && c < grid_size && not (is_obstacle (r, c))

let reset _env ?options:_ () state =
  state.position <- (0, 0);
  state.steps <- 0;
  (Rune.create Rune.int32 [| 2 |] [| 0l; 0l |], Info.empty)

let step _env action state =
  let action_value =
    let tensor = Rune.reshape [| 1 |] action in
    let arr : Int32.t array = Rune.to_array tensor in
    Int32.to_int arr.(0)
  in
  let row, col = state.position in
  let candidate =
    match action_value with
    | 0 -> (row - 1, col)
    | 1 -> (row + 1, col)
    | 2 -> (row, col - 1)
    | 3 -> (row, col + 1)
    | _ -> (row, col)
  in
  let next_pos = if is_valid_pos candidate then candidate else state.position in
  state.position <- next_pos;
  state.steps <- state.steps + 1;
  let terminated = is_goal next_pos in
  let truncated = state.steps >= 200 in
  let reward = if terminated then 10.0 else -1.0 in
  let info = Info.set "steps" (Info.int state.steps) Info.empty in
  let r, c = next_pos in
  let observation =
    Rune.create Rune.int32 [| 2 |] [| Int32.of_int r; Int32.of_int c |]
  in
  Env.transition ~observation ~reward ~terminated ~truncated ~info ()

let render_text state =
  let buffer = Bytes.make (grid_size * grid_size) '.' in
  let row, col = state.position in
  let index = (row * grid_size) + col in
  let () = Bytes.set buffer index 'A' in
  let goal_index = ((grid_size - 1) * grid_size) + (grid_size - 1) in
  Bytes.set buffer goal_index 'G';
  let obstacle_index = (2 * grid_size) + 2 in
  Bytes.set buffer obstacle_index '#';
  let rows =
    List.init grid_size (fun r ->
        let start = r * grid_size in
        Bytes.sub_string buffer start grid_size)
  in
  Format.asprintf "Position: (%d, %d)@.%a" row col
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@.")
       Format.pp_print_string)
    rows

let cell_size = 32
let frame_width = grid_size * cell_size
let frame_height = grid_size * cell_size
let clamp_color value = max 0 (min 255 value)

let fill_rect data ~width ~x0 ~y0 ~w ~h (r, g, b) =
  let r = Char.unsafe_chr (clamp_color r) in
  let g = Char.unsafe_chr (clamp_color g) in
  let b = Char.unsafe_chr (clamp_color b) in
  for dy = 0 to h - 1 do
    let y = y0 + dy in
    let row_offset = y * width * 3 in
    for dx = 0 to w - 1 do
      let x = x0 + dx in
      let base = row_offset + (x * 3) in
      Bigarray.Array1.unsafe_set data (base + 0) r;
      Bigarray.Array1.unsafe_set data (base + 1) g;
      Bigarray.Array1.unsafe_set data (base + 2) b
    done
  done

let render_image state =
  let len = frame_width * frame_height * 3 in
  let data = Bigarray.Array1.create Bigarray.char Bigarray.c_layout len in
  fill_rect data ~width:frame_width ~x0:0 ~y0:0 ~w:frame_width ~h:frame_height
    (30, 33, 36);
  (* Draw grid cells with subtle borders *)
  for row = 0 to grid_size - 1 do
    for col = 0 to grid_size - 1 do
      let x0 = col * cell_size in
      let y0 = row * cell_size in
      fill_rect data ~width:frame_width ~x0 ~y0 ~w:cell_size ~h:cell_size
        (44, 48, 52);
      fill_rect data ~width:frame_width ~x0:(x0 + 1) ~y0:(y0 + 1)
        ~w:(cell_size - 2) ~h:(cell_size - 2) (54, 60, 65)
    done
  done;
  (* Overlay entities *)
  let draw_cell row col color =
    let x0 = (col * cell_size) + 2 in
    let y0 = (row * cell_size) + 2 in
    fill_rect data ~width:frame_width ~x0 ~y0 ~w:(cell_size - 4)
      ~h:(cell_size - 4) color
  in
  let row, col = state.position in
  draw_cell row col (78, 162, 196);
  draw_cell (grid_size - 1) (grid_size - 1) (76, 175, 80);
  draw_cell 2 2 (200, 80, 80);
  Render.image_u8 ~width:frame_width ~height:frame_height ~pixel_format:`RGB8
    ~data ()

let render env state =
  let mode = Env.render_mode env in
  match mode with
  | Some `Rgb_array -> Some (Render.Image (render_image state))
  | Some `Ansi -> Some (Render.Text (render_text state))
  | Some (`Custom mode) when String.equal mode "rgb_array" ->
      Some (Render.Image (render_image state))
  | Some (`Custom mode) when String.equal mode "ansi" ->
      Some (Render.Text (render_text state))
  | _ -> Some (Render.Text (render_text state))

let make ~rng ?render_mode () =
  let state = { position = (0, 0); steps = 0 } in
  Env.create ~id:"GridWorld-v0" ~metadata ?render_mode ~rng ~observation_space
    ~action_space
    ~reset:(fun env ?options () -> reset env ?options () state)
    ~step:(fun env action -> step env action state)
    ~render:(fun env -> render env state)
    ~close:(fun _ -> ())
    ()
