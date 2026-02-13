(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

let string_contains ~needle haystack =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  let rec loop idx =
    if idx + n_len > h_len then false
    else if String.sub haystack idx n_len = needle then true
    else loop (idx + 1)
  in
  if n_len = 0 then true else loop 0

let make_simple_env ?(action_space = Space.Discrete.create 2) ~rng ~id () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let state = ref 5.0 in
  let reset _env ?options:_ () =
    state := 5.0;
    (Rune.create Rune.float32 [| 1 |] [| !state |], Info.empty)
  in
  let step _env action =
    let action_val =
      let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
      Int32.to_int arr.(0)
    in
    let direction = if action_val = 0 then -1.0 else 1.0 in
    state := !state +. direction;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    Env.transition ~observation:obs ~reward:1.0 ~terminated ()
  in
  Env.create ~id ~rng ~observation_space:obs_space ~action_space ~reset ~step ()

let test_vec_env_creation () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let rng3 = Rune.Rng.key 44 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let env3 = make_simple_env ~rng:rng3 ~id:"Env3" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2; env3 ] () in
  equal ~msg:"num envs" int 3 (Vector_env.num_envs vec_env)

let test_vec_env_incompatible_action_space () =
  let rng1 = Rune.Rng.key 101 in
  let rng2 = Rune.Rng.key 202 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 =
    make_simple_env ~action_space:(Space.Discrete.create 3) ~rng:rng2 ~id:"Env2"
      ()
  in
  match Vector_env.create_sync ~envs:[ env1; env2 ] () with
  | exception Errors.Error (Errors.Invalid_metadata msg) ->
      equal ~msg:"error mentions action space" bool true
        (string_contains ~needle:"action" msg)
  | _ -> fail "expected Invalid_metadata due to mismatched action space"

let test_vec_env_reset () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let obs_arr, info_arr = Vector_env.reset vec_env () in
  equal ~msg:"observations array length" int 2 (Array.length obs_arr);
  equal ~msg:"infos array length" int 2 (Array.length info_arr);
  Array.iter
    (fun obs ->
      let shape = Rune.shape obs in
      equal ~msg:"obs shape" (array int) [| 1 |] shape)
    obs_arr

let test_vec_env_step () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let _, _ = Vector_env.reset vec_env () in
  let actions =
    [|
      Rune.create Rune.int32 [| 1 |] [| 0l |];
      Rune.create Rune.int32 [| 1 |] [| 1l |];
    |]
  in
  let result = Vector_env.step vec_env actions in
  equal ~msg:"observations length" int 2 (Array.length result.observations);
  equal ~msg:"rewards length" int 2 (Array.length result.rewards);
  equal ~msg:"terminations length" int 2 (Array.length result.terminations);
  equal ~msg:"truncations length" int 2 (Array.length result.truncations);
  equal ~msg:"infos length" int 2 (Array.length result.infos)

let test_vec_env_autoreset_next_step () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env =
    Vector_env.create_sync ~autoreset_mode:Vector_env.Next_step
      ~envs:[ env1; env2 ] ()
  in
  let _, _ = Vector_env.reset vec_env () in
  let actions =
    [|
      Rune.create Rune.int32 [| 1 |] [| 1l |];
      Rune.create Rune.int32 [| 1 |] [| 1l |];
    |]
  in
  let rec step_until_termination count =
    if count > 20 then fail "no termination"
    else
      let result = Vector_env.step vec_env actions in
      if result.terminations.(0) || result.terminations.(1) then result
      else step_until_termination (count + 1)
  in
  let _result = step_until_termination 0 in
  let next_result = Vector_env.step vec_env actions in
  equal ~msg:"env reset after termination" bool false
    next_result.terminations.(0)

let test_vec_env_autoreset_disabled () =
  let rng1 = Rune.Rng.key 42 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let vec_env =
    Vector_env.create_sync ~autoreset_mode:Vector_env.Disabled ~envs:[ env1 ] ()
  in
  let _, _ = Vector_env.reset vec_env () in
  let actions = [| Rune.create Rune.int32 [| 1 |] [| 1l |] |] in
  let rec step_until_termination count =
    if count > 20 then fail "no termination"
    else
      let result = Vector_env.step vec_env actions in
      if result.terminations.(0) then result
      else step_until_termination (count + 1)
  in
  let result = step_until_termination 0 in
  equal ~msg:"env terminated" bool true result.terminations.(0);
  equal ~msg:"disabled autoreset mode works" pass () ()

let test_vec_env_spaces () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let _obs_space = Vector_env.observation_space vec_env in
  let _act_space = Vector_env.action_space vec_env in
  equal ~msg:"spaces accessible" pass () ()

let test_vec_env_metadata () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let _metadata = Vector_env.metadata vec_env in
  equal ~msg:"metadata accessible" pass () ()

let test_vec_env_close () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  Vector_env.close vec_env;
  equal ~msg:"vec env closed" pass () ()

let () =
  run "Vector_env"
    [
      group "Creation"
        [
          test "create vectorized env" test_vec_env_creation;
          test "reject incompatible action spaces"
            test_vec_env_incompatible_action_space;
        ];
      group "Lifecycle"
        [
          test "reset vec env" test_vec_env_reset;
          test "step vec env" test_vec_env_step;
        ];
      group "Autoreset"
        [
          test "autoreset next_step" test_vec_env_autoreset_next_step;
          test "autoreset disabled" test_vec_env_autoreset_disabled;
        ];
      group "Properties"
        [
          test "spaces" test_vec_env_spaces;
          test "metadata" test_vec_env_metadata;
          test "close" test_vec_env_close;
        ];
    ]
