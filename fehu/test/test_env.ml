(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

let make_simple_env ~rng () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 0.0 in
  let reset _env ?options:_ () =
    state := 5.0;
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    (obs, Info.empty)
  in
  let step _env action =
    let action_val =
      let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
      Int32.to_int arr.(0)
    in
    state := !state +. if action_val = 0 then -1.0 else 1.0;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    Env.transition ~observation:obs ~reward:1.0 ~terminated ~truncated:false ()
  in
  Env.create ~id:"Simple-v0" ~rng ~observation_space:obs_space
    ~action_space:act_space ~reset ~step ()

let test_env_creation () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  match Env.id env with
  | Some id -> equal ~msg:"env id" string "Simple-v0" id
  | None -> fail "expected env id"

let test_env_reset () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let obs, info = Env.reset env () in
  let shape = Rune.shape obs in
  equal ~msg:"reset obs shape" (array int) [| 1 |] shape;
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  equal ~msg:"reset obs value" (float 0.01) 5.0 arr.(0);
  equal ~msg:"reset info empty" bool true (Info.is_empty info)

let test_env_step () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let shape = Rune.shape transition.Env.observation in
  equal ~msg:"step obs shape" (array int) [| 1 |] shape;
  equal ~msg:"step reward" (float 0.01) 1.0 transition.Env.reward;
  equal ~msg:"step not terminated initially" bool false transition.Env.terminated

let test_env_episode_termination () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let _, _ = Env.reset env () in
  let action_up = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let rec run_to_termination count =
    if count > 20 then fail "episode did not terminate"
    else
      let transition = Env.step env action_up in
      if transition.Env.terminated then count else run_to_termination (count + 1)
  in
  let steps = run_to_termination 0 in
  equal ~msg:"episode terminates" bool true (steps <= 10)

let test_env_metadata () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let metadata = Env.metadata env in
  equal ~msg:"metadata exists" pass () ();
  let updated =
    metadata
    |> Metadata.with_description (Some "Test")
    |> Metadata.add_author "Alice"
  in
  Env.set_metadata env updated;
  let new_metadata = Env.metadata env in
  equal ~msg:"metadata updated" (option string)
    (Some "Test") new_metadata.description

let test_env_rng () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let rng1 = Env.take_rng env in
  let rng2 = Env.rng env in
  equal ~msg:"rng updated after take" bool true (rng1 <> rng2)

let test_env_split_rng () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let rngs = Env.split_rng env ~n:5 in
  equal ~msg:"split produces n rngs" int 5 (Array.length rngs)

let test_env_spaces () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  let obs_space = Env.observation_space env in
  let act_space = Env.action_space env in
  let obs_shape = Space.shape obs_space in
  let act_shape = Space.shape act_space in
  equal ~msg:"obs space shape" (option (array int))
    (Some [| 1 |]) obs_shape;
  equal ~msg:"act space shape" (option (array int)) None act_shape

let test_env_close () =
  let rng = Rune.Rng.key 42 in
  let env = make_simple_env ~rng () in
  equal ~msg:"env initially open" bool false (Env.closed env);
  Env.close env;
  equal ~msg:"env closed after close" bool true (Env.closed env)

let test_env_render () =
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let act_space = Space.Discrete.create 2 in
  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _ ?options:_ () ->
        (Rune.create Rune.float32 [| 1 |] [| 0.5 |], Info.empty))
      ~step:(fun _ _action ->
        Env.transition
          ~observation:(Rune.create Rune.float32 [| 1 |] [| 0.5 |])
          ~reward:0.0 ())
      ~render:(fun _ -> Some "test render")
      ()
  in
  match Env.render env with
  | Some str -> equal ~msg:"render output" string "test render" str
  | None -> fail "expected render output"

let test_transition_builder () =
  let obs = Rune.create Rune.float32 [| 1 |] [| 1.0 |] in
  let t1 = Env.transition ~observation:obs () in
  equal ~msg:"default reward" (float 0.01) 0.0 t1.reward;
  equal ~msg:"default terminated" bool false t1.terminated;
  equal ~msg:"default truncated" bool false t1.truncated;
  let t2 = Env.transition ~observation:obs ~reward:5.0 ~terminated:true () in
  equal ~msg:"custom reward" (float 0.01) 5.0 t2.reward;
  equal ~msg:"custom terminated" bool true t2.terminated

let () =
  run "Env"
    [
      group "Creation"
        [
          test "create env" test_env_creation;
          test "env spaces" test_env_spaces;
        ];
      group "Lifecycle"
        [
          test "reset env" test_env_reset;
          test "step env" test_env_step;
          test "episode termination" test_env_episode_termination;
          test "close env" test_env_close;
        ];
      group "Metadata"
        [
          test "metadata operations" test_env_metadata;
          test "rng operations" test_env_rng;
          test "split rng" test_env_split_rng;
        ];
      group "Utilities"
        [
          test "render" test_env_render;
          test "transition builder" test_transition_builder;
        ];
    ]
