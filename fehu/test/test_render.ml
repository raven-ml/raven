open Fehu
open Windtrap

let rng = Rune.Rng.key 42

let make_data n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

(* Image *)

let test_valid_rgb_image () =
  let data = make_data 12 in
  let img = Render.image ~width:2 ~height:2 data in
  equal ~msg:"width" int 2 img.width;
  equal ~msg:"height" int 2 img.height

let test_wrong_data_length_raises () =
  let data = make_data 10 in
  raises_invalid_arg
    "Render.image: data length 10 does not match width * height * channels = 12"
    (fun () -> ignore (Render.image ~width:2 ~height:2 data))

let test_rgba_channels () =
  let data = make_data 16 in
  let img =
    Render.image ~width:2 ~height:2 ~pixel_format:Render.Pixel.Rgba data
  in
  equal ~msg:"width" int 2 img.width;
  equal ~msg:"height" int 2 img.height

let test_gray_channels () =
  let data = make_data 4 in
  let img =
    Render.image ~width:2 ~height:2 ~pixel_format:Render.Pixel.Gray data
  in
  equal ~msg:"width" int 2 img.width;
  equal ~msg:"height" int 2 img.height

let test_pixel_format_default_rgb () =
  let data = make_data 3 in
  let img = Render.image ~width:1 ~height:1 data in
  equal ~msg:"default is Rgb" int 3 (Render.Pixel.channels img.pixel_format)

(* Rollout *)

let make_renderable_env ~rng () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 5.0 in
  let reset _env ?options:_ () =
    state := 5.0;
    (Rune.create Rune.float32 [| 1 |] [| !state |], Info.empty)
  in
  let step _env action =
    let a : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
    state := !state +. if Int32.to_int a.(0) = 0 then -1.0 else 1.0;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    Env.step_result
      ~observation:(Rune.create Rune.float32 [| 1 |] [| !state |])
      ~reward:1.0 ~terminated ()
  in
  let render () =
    let data = make_data 3 in
    Some (Render.image ~width:1 ~height:1 data)
  in
  Env.create ~id:"Renderable-v0" ~rng ~observation_space:obs_space
    ~action_space:act_space ~reset ~step ~render ()

let test_rollout_sink_called () =
  let env = make_renderable_env ~rng () in
  let count = ref 0 in
  let policy _obs = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let sink _frame = incr count in
  Render.rollout env ~policy ~steps:3 ~sink ();
  equal ~msg:"sink called 3 times" int 3 !count

(* on_render *)

let action_right = Rune.create Rune.int32 [| 1 |] [| 1l |]

let test_on_render_frame_count () =
  let env = make_renderable_env ~rng () in
  let count = ref 0 in
  let wrapped = Render.on_render ~sink:(fun _ -> incr count) env in
  let _obs, _info = Env.reset wrapped () in
  let _s1 = Env.step wrapped action_right in
  let _s2 = Env.step wrapped action_right in
  let _s3 = Env.step wrapped action_right in
  (* 1 frame from reset + 3 frames from steps = 4 *)
  equal ~msg:"frame count" int 4 !count

let test_on_render_passthrough () =
  let env = make_renderable_env ~rng () in
  let wrapped = Render.on_render ~sink:(fun _ -> ()) env in
  let _obs, _info = Env.reset wrapped () in
  let step = Env.step wrapped action_right in
  equal ~msg:"reward unchanged" (float 0.0) 1.0 step.reward;
  is_false ~msg:"not terminated" step.terminated;
  is_false ~msg:"not truncated" step.truncated

let test_on_render_id () =
  let env = make_renderable_env ~rng () in
  let wrapped = Render.on_render ~sink:(fun _ -> ()) env in
  equal ~msg:"id suffix" (option string) (Some "Renderable-v0/OnRender")
    (Env.id wrapped)

let () =
  run "Fehu.Render"
    [
      group "image"
        [
          test "valid RGB 2x2" test_valid_rgb_image;
          test "wrong data length raises" test_wrong_data_length_raises;
          test "RGBA 4 channels" test_rgba_channels;
          test "Gray 1 channel" test_gray_channels;
          test "default pixel_format is Rgb" test_pixel_format_default_rgb;
        ];
      group "rollout"
        [ test "sink called for each step" test_rollout_sink_called ];
      group "on_render"
        [
          test "frame count" test_on_render_frame_count;
          test "passthrough" test_on_render_passthrough;
          test "id suffix" test_on_render_id;
        ];
    ]
