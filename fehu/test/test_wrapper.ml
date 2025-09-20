open Fehu

let make_base_env ~rng () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 0 in
  let reset _env ?options:_ () =
    state := 0;
    (Rune.create Rune.float32 [| 1 |] [| 5.0 |], Info.empty)
  in
  let step _env action =
    let action_val =
      let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
      Int32.to_int arr.(0)
    in
    state := !state + 1;
    let value =
      5.0 +. (float_of_int !state *. if action_val = 0 then -1.0 else 1.0)
    in
    let obs = Rune.create Rune.float32 [| 1 |] [| value |] in
    Env.transition ~observation:obs ~reward:1.0 ()
  in
  Env.create ~rng ~observation_space:obs_space ~action_space:act_space ~reset
    ~step ()

let test_map_observation () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let new_obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 20.0 |] in
  let wrapped =
    Wrapper.map_observation ~observation_space:new_obs_space base_env
      ~f:(fun obs info ->
        let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
        let scaled = Rune.create Rune.float32 [| 1 |] [| arr.(0) *. 2.0 |] in
        (scaled, info))
  in
  let obs, _ = Env.reset wrapped () in
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  Alcotest.(check (float 0.01)) "observation scaled" 10.0 arr.(0)

let test_map_action () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let new_act_space = Space.Discrete.create 4 in
  let wrapped =
    Wrapper.map_action ~action_space:new_act_space base_env ~f:(fun action ->
        let arr : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
        let mapped = if Int32.to_int arr.(0) < 2 then 0l else 1l in
        Rune.create Rune.int32 [| 1 |] [| mapped |])
  in
  let _, _ = Env.reset wrapped () in
  let action = Rune.create Rune.int32 [| 1 |] [| 3l |] in
  let transition = Env.step wrapped action in
  Alcotest.(check pass) "action mapping works" () ();
  Alcotest.(check (float 0.01)) "reward received" 1.0 transition.Env.reward

let test_map_reward () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped =
    Wrapper.map_reward base_env ~f:(fun ~reward ~info ->
        (reward *. 10.0, Info.set "scaled" (Info.bool true) info))
  in
  let _, _ = Env.reset wrapped () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let transition = Env.step wrapped action in
  Alcotest.(check (float 0.01)) "reward scaled" 10.0 transition.Env.reward;
  match Info.find "scaled" transition.Env.info with
  | Some (Info.Bool true) -> Alcotest.(check pass) "info updated" () ()
  | _ -> Alcotest.fail "expected scaled info"

let test_time_limit () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped = Wrapper.time_limit ~max_episode_steps:5 base_env in
  let _, _ = Env.reset wrapped () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let rec step_until_truncated n =
    if n > 10 then Alcotest.fail "did not truncate within 10 steps"
    else
      let transition = Env.step wrapped action in
      if transition.Env.truncated then n else step_until_truncated (n + 1)
  in
  let steps = step_until_truncated 1 in
  Alcotest.(check int) "truncated at step 5" 5 steps

let test_with_metadata () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped =
    Wrapper.with_metadata base_env ~f:(fun metadata ->
        metadata |> Metadata.with_description (Some "Modified"))
  in
  let metadata = Env.metadata wrapped in
  Alcotest.(check (option string))
    "metadata modified" (Some "Modified") metadata.description

let test_chained_wrappers () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped =
    base_env
    |> Wrapper.map_reward ~f:(fun ~reward ~info -> (reward *. 2.0, info))
    |> Wrapper.time_limit ~max_episode_steps:3
  in
  let _, _ = Env.reset wrapped () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let t1 = Env.step wrapped action in
  Alcotest.(check (float 0.01)) "reward scaled in chain" 2.0 t1.Env.reward;
  let _t2 = Env.step wrapped action in
  let t3 = Env.step wrapped action in
  Alcotest.(check bool) "truncated in chain" true t3.Env.truncated

let () =
  let open Alcotest in
  run "Wrapper"
    [
      ( "Map wrappers",
        [
          test_case "map observation" `Quick test_map_observation;
          test_case "map action" `Quick test_map_action;
          test_case "map reward" `Quick test_map_reward;
        ] );
      ( "Utility wrappers",
        [
          test_case "time limit" `Quick test_time_limit;
          test_case "with metadata" `Quick test_with_metadata;
        ] );
      ( "Composition",
        [ test_case "chained wrappers" `Quick test_chained_wrappers ] );
    ]
