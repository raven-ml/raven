open Fehu

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
  Env.create ~id ~rng ~observation_space:obs_space ~action_space:action_space
    ~reset ~step ()

let test_vec_env_creation () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let rng3 = Rune.Rng.key 44 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let env3 = make_simple_env ~rng:rng3 ~id:"Env3" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2; env3 ] () in
  Alcotest.(check int) "num envs" 3 (Vector_env.num_envs vec_env)

let test_vec_env_incompatible_action_space () =
  let rng1 = Rune.Rng.key 101 in
  let rng2 = Rune.Rng.key 202 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 =
    make_simple_env
      ~action_space:(Space.Discrete.create 3)
      ~rng:rng2 ~id:"Env2" ()
  in
  match Vector_env.create_sync ~envs:[ env1; env2 ] () with
  | exception Errors.Error (Errors.Invalid_metadata msg) ->
      Alcotest.(check bool)
        "error mentions action space"
        true
        (string_contains ~needle:"action" msg)
  | _ -> Alcotest.fail "expected Invalid_metadata due to mismatched action space"

let test_vec_env_reset () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let obs_arr, info_arr = Vector_env.reset vec_env () in
  Alcotest.(check int) "observations array length" 2 (Array.length obs_arr);
  Alcotest.(check int) "infos array length" 2 (Array.length info_arr);
  Array.iter
    (fun obs ->
      let shape = Rune.shape obs in
      Alcotest.(check (array int)) "obs shape" [| 1 |] shape)
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
  Alcotest.(check int)
    "observations length" 2
    (Array.length result.observations);
  Alcotest.(check int) "rewards length" 2 (Array.length result.rewards);
  Alcotest.(check int)
    "terminations length" 2
    (Array.length result.terminations);
  Alcotest.(check int) "truncations length" 2 (Array.length result.truncations);
  Alcotest.(check int) "infos length" 2 (Array.length result.infos)

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
    if count > 20 then Alcotest.fail "no termination"
    else
      let result = Vector_env.step vec_env actions in
      if result.terminations.(0) || result.terminations.(1) then result
      else step_until_termination (count + 1)
  in
  let _result = step_until_termination 0 in
  let next_result = Vector_env.step vec_env actions in
  Alcotest.(check bool)
    "env reset after termination" false
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
    if count > 20 then Alcotest.fail "no termination"
    else
      let result = Vector_env.step vec_env actions in
      if result.terminations.(0) then result
      else step_until_termination (count + 1)
  in
  let result = step_until_termination 0 in
  Alcotest.(check bool) "env terminated" true result.terminations.(0);
  Alcotest.(check pass) "disabled autoreset mode works" () ()

let test_vec_env_spaces () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let _obs_space = Vector_env.observation_space vec_env in
  let _act_space = Vector_env.action_space vec_env in
  Alcotest.(check pass) "spaces accessible" () ()

let test_vec_env_metadata () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  let _metadata = Vector_env.metadata vec_env in
  Alcotest.(check pass) "metadata accessible" () ()

let test_vec_env_close () =
  let rng1 = Rune.Rng.key 42 in
  let rng2 = Rune.Rng.key 43 in
  let env1 = make_simple_env ~rng:rng1 ~id:"Env1" () in
  let env2 = make_simple_env ~rng:rng2 ~id:"Env2" () in
  let vec_env = Vector_env.create_sync ~envs:[ env1; env2 ] () in
  Vector_env.close vec_env;
  Alcotest.(check pass) "vec env closed" () ()

let () =
  let open Alcotest in
  run "Vector_env"
    [
      ( "Creation",
        [
          test_case "create vectorized env" `Quick test_vec_env_creation;
          test_case "reject incompatible action spaces" `Quick
            test_vec_env_incompatible_action_space;
        ] );
      ( "Lifecycle",
        [
          test_case "reset vec env" `Quick test_vec_env_reset;
          test_case "step vec env" `Quick test_vec_env_step;
        ] );
      ( "Autoreset",
        [
          test_case "autoreset next_step" `Quick
            test_vec_env_autoreset_next_step;
          test_case "autoreset disabled" `Quick test_vec_env_autoreset_disabled;
        ] );
      ( "Properties",
        [
          test_case "spaces" `Quick test_vec_env_spaces;
          test_case "metadata" `Quick test_vec_env_metadata;
          test_case "close" `Quick test_vec_env_close;
        ] );
    ]
