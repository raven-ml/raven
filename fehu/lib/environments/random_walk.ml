open Fehu

type observation = (float, Rune.float32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = string
type state = { mutable position : float; mutable steps : int }

let observation_space = Space.Box.create ~low:[| -10.0 |] ~high:[| 10.0 |]
let action_space = Space.Discrete.create 2

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "ansi"
  |> Metadata.with_description
       (Some "Simple one-dimensional random walk environment")
  |> Metadata.add_author "Fehu New"
  |> Metadata.with_version (Some "0.1.0")

let reset _env ?options:_ () state =
  state.position <- 0.0;
  state.steps <- 0;
  (Rune.create Rune.float32 [| 1 |] [| 0.0 |], Info.empty)

let step _env action state =
  let action_value =
    let tensor = Rune.reshape [| 1 |] action in
    let arr : Int32.t array = Rune.to_array tensor in
    Int32.to_int arr.(0)
  in
  let direction = if action_value = 0 then -1.0 else 1.0 in
  let updated = state.position +. direction in
  let clamped = Float.min 10.0 (Float.max (-10.0) updated) in
  state.position <- clamped;
  state.steps <- state.steps + 1;
  let terminated = Float.abs state.position >= 10.0 in
  let truncated = state.steps >= 200 in
  let reward = -.Float.abs state.position in
  let info = Info.set "steps" (Info.int state.steps) Info.empty in
  let observation = Rune.create Rune.float32 [| 1 |] [| state.position |] in
  Env.transition ~observation ~reward ~terminated ~truncated ~info ()

let render state =
  let offset = int_of_float (state.position +. 10.) in
  let offset = max 0 (min 20 offset) in
  let buffer = Bytes.make 21 '.' in
  Bytes.set buffer offset 'o';
  Format.asprintf "Position: %+.2f@.|%s|" state.position
    (Bytes.to_string buffer)

let make ~rng () =
  let state = { position = 0.0; steps = 0 } in
  Env.create ~id:"RandomWalk-v0" ~metadata ~rng ~observation_space ~action_space
    ~reset:(fun env ?options () -> reset env ?options () state)
    ~step:(fun env action -> step env action state)
    ~render:(fun _ -> Some (render state))
    ~close:(fun _ -> ())
    ()
