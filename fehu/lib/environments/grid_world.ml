open Fehu

type observation = (int32, Rune.int32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = string
type state = { mutable position : int * int; mutable steps : int }

let grid_size = 5
let observation_space = Space.Multi_discrete.create [| grid_size; grid_size |]
let action_space = Space.Discrete.create 4

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "ansi"
  |> Metadata.with_description
       (Some "Simple 5x5 grid world with goal and obstacles")
  |> Metadata.add_author "Fehu New"
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

let render state =
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

let make ~rng () =
  let state = { position = (0, 0); steps = 0 } in
  Env.create ~id:"GridWorld-v0" ~metadata ~rng ~observation_space ~action_space
    ~reset:(fun env ?options () -> reset env ?options () state)
    ~step:(fun env action -> step env action state)
    ~render:(fun _ -> Some (render state))
    ~close:(fun _ -> ())
    ()
