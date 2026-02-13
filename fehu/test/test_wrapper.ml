(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

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

let make_continuous_env ~rng () =
  let obs_space = Space.Box.create ~low:[| -5.0 |] ~high:[| 5.0 |] in
  let act_space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let state = ref 0.0 in
  let reset _env ?options:_ () =
    state := 0.0;
    (Rune.create Rune.float32 [| 1 |] [| !state |], Info.empty)
  in
  let step _env action =
    let arr : float array = Rune.to_array action in
    let delta = arr.(0) in
    state := max (-5.0) (min 5.0 (!state +. delta));
    let obs = Rune.create Rune.float32 [| 1 |] [| !state |] in
    Env.transition ~observation:obs ~reward:delta ()
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
  equal ~msg:"observation scaled" (float 0.01) 10.0 arr.(0)

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
  equal ~msg:"action mapping works" pass () ();
  equal ~msg:"reward received" (float 0.01) 1.0 transition.Env.reward

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
  equal ~msg:"reward scaled" (float 0.01) 10.0 transition.Env.reward;
  match Info.find "scaled" transition.Env.info with
  | Some (Info.Bool true) -> equal ~msg:"info updated" pass () ()
  | _ -> fail "expected scaled info"

let test_map_info () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped =
    Wrapper.map_info base_env ~f:(fun info ->
        Info.set "patched" (Info.bool true) info)
  in
  let _, info = Env.reset wrapped () in
  (match Info.find "patched" info with
  | Some (Info.Bool true) -> equal ~msg:"reset info patched" pass () ()
  | _ -> fail "reset info missing patch");
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let transition = Env.step wrapped action in
  match Info.find "patched" transition.Env.info with
  | Some (Info.Bool true) -> equal ~msg:"step info patched" pass () ()
  | _ -> fail "step info missing patch"

let test_clip_action () =
  let rng = Rune.Rng.key 99 in
  let env = make_continuous_env ~rng () |> Wrapper.clip_action in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.float32 [| 1 |] [| 5.0 |] in
  let transition = Env.step env action in
  let obs_arr : float array = Rune.to_array transition.Env.observation in
  equal ~msg:"action clipped" (float 0.001) 1.0 obs_arr.(0);
  equal ~msg:"reward clipped" (float 0.001) 1.0 transition.Env.reward

let test_clip_observation () =
  let rng = Rune.Rng.key 77 in
  let env = make_continuous_env ~rng () |> Wrapper.clip_observation in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.float32 [| 1 |] [| 0.5 |] in
  let transition = Env.step env action in
  let obs_arr : float array = Rune.to_array transition.Env.observation in
  equal ~msg:"observation preserved" (float 0.001) 0.5 obs_arr.(0)

let test_time_limit () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped = Wrapper.time_limit ~max_episode_steps:5 base_env in
  let _, _ = Env.reset wrapped () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let rec step_until_truncated n =
    if n > 10 then fail "did not truncate within 10 steps"
    else
      let transition = Env.step wrapped action in
      if transition.Env.truncated then (n, transition)
      else step_until_truncated (n + 1)
  in
  let steps, transition = step_until_truncated 1 in
  equal ~msg:"truncated at step 5" int 5 steps;
  (match Info.find "time_limit.truncated" transition.Env.info with
  | Some (Info.Bool true) -> equal ~msg:"truncated flag" pass () ()
  | _ -> fail "missing truncated flag");
  match Info.find "time_limit.elapsed_steps" transition.Env.info with
  | Some (Info.Int 5) -> equal ~msg:"elapsed steps recorded" pass () ()
  | Some (Info.Int other) -> failf "unexpected elapsed steps %d" other
  | _ -> fail "missing elapsed steps info"

let test_with_metadata () =
  let rng = Rune.Rng.key 42 in
  let base_env = make_base_env ~rng () in
  let wrapped =
    Wrapper.with_metadata base_env ~f:(fun metadata ->
        metadata |> Metadata.with_description (Some "Modified"))
  in
  let metadata = Env.metadata wrapped in
  equal ~msg:"metadata modified" (option string) (Some "Modified")
    metadata.description

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
  equal ~msg:"reward scaled in chain" (float 0.01) 2.0 t1.Env.reward;
  let _t2 = Env.step wrapped action in
  let t3 = Env.step wrapped action in
  equal ~msg:"truncated in chain" bool true t3.Env.truncated

let () =
  run "Wrapper"
    [
      group "Map wrappers"
        [
          test "map observation" test_map_observation;
          test "map action" test_map_action;
          test "map reward" test_map_reward;
          test "map info" test_map_info;
        ];
      group "Utility wrappers"
        [
          test "time limit" test_time_limit;
          test "with metadata" test_with_metadata;
          test "clip action" test_clip_action;
          test "clip observation" test_clip_observation;
        ];
      group "Composition" [ test "chained wrappers" test_chained_wrappers ];
    ]
