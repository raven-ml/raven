open Fehu
open Windtrap

let make_test_env ?(max_steps = 100) () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let state = ref 5.0 in
  let steps = ref 0 in
  let reset _env ?options:_ () =
    state := 5.0;
    steps := 0;
    (Rune.create Rune.float32 [| 1 |] [| !state |], Info.empty)
  in
  let step _env action =
    let a : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
    state := !state +. if Int32.to_int a.(0) = 0 then -1.0 else 1.0;
    incr steps;
    let terminated = !state <= 0.0 || !state >= 10.0 in
    let truncated = (not terminated) && !steps >= max_steps in
    Env.step_result
      ~observation:(Rune.create Rune.float32 [| 1 |] [| !state |])
      ~reward:1.0 ~terminated ~truncated ()
  in
  Env.create ~id:"Test-v0" ~observation_space:obs_space ~action_space:act_space
    ~reset ~step ()

let action_left = Rune.create Rune.int32 [| 1 |] [| 0l |]
let action_right = Rune.create Rune.int32 [| 1 |] [| 1l |]

let read_obs obs =
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  arr.(0)

let value = testable ~pp:Value.pp ~equal:Value.equal ()

(* State sharing *)

let test_close_wrapper_closes_inner () =
  let env = make_test_env () in
  let wrapped =
    Env.map_observation
      ~observation_space:(Env.observation_space env)
      ~f:(fun obs info -> (obs, info))
      env
  in
  Env.close wrapped;
  is_true ~msg:"inner closed" (Env.closed env)

let test_close_inner_closes_wrapper () =
  let env = make_test_env () in
  let wrapped =
    Env.map_observation
      ~observation_space:(Env.observation_space env)
      ~f:(fun obs info -> (obs, info))
      env
  in
  Env.close env;
  is_true ~msg:"wrapper closed" (Env.closed wrapped)

let test_reset_wrapper_clears_inner () =
  let env = make_test_env () in
  let wrapped =
    Env.map_observation
      ~observation_space:(Env.observation_space env)
      ~f:(fun obs info -> (obs, info))
      env
  in
  let _obs, _info = Env.reset wrapped () in
  let step = Env.step env action_left in
  equal ~msg:"inner step works" (float 0.0) 1.0 step.reward

(* map_observation *)

let test_map_observation_reset () =
  let env = make_test_env () in
  let double_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 20.0 |] in
  let wrapped =
    Env.map_observation ~observation_space:double_space
      ~f:(fun obs info ->
        let v = read_obs obs in
        (Rune.create Rune.float32 [| 1 |] [| v *. 2.0 |], info))
      env
  in
  let obs, _info = Env.reset wrapped () in
  equal ~msg:"doubled reset obs" (float 0.0) 10.0 (read_obs obs)

let test_map_observation_step () =
  let env = make_test_env () in
  let double_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 20.0 |] in
  let wrapped =
    Env.map_observation ~observation_space:double_space
      ~f:(fun obs info ->
        let v = read_obs obs in
        (Rune.create Rune.float32 [| 1 |] [| v *. 2.0 |], info))
      env
  in
  let _obs, _info = Env.reset wrapped () in
  let step = Env.step wrapped action_right in
  (* Inner: 5 + 1 = 6, doubled: 12 *)
  equal ~msg:"doubled step obs" (float 0.0) 12.0 (read_obs step.observation)

let test_map_observation_id () =
  let env = make_test_env () in
  let wrapped =
    Env.map_observation
      ~observation_space:(Env.observation_space env)
      ~f:(fun obs info -> (obs, info))
      env
  in
  equal ~msg:"id suffix" (option string) (Some "Test-v0/ObservationWrapper")
    (Env.id wrapped)

(* map_action *)

let test_map_action_flip () =
  let env = make_test_env () in
  let wrapped =
    Env.map_action ~action_space:(Env.action_space env)
      ~f:(fun action ->
        let a : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
        let flipped = if Int32.to_int a.(0) = 0 then 1l else 0l in
        Rune.create Rune.int32 [| 1 |] [| flipped |])
      env
  in
  let _obs, _info = Env.reset wrapped () in
  (* Send left (0) to wrapper; inner sees right (1): 5 -> 6 *)
  let step = Env.step wrapped action_left in
  equal ~msg:"flipped: left becomes right" (float 0.0) 6.0
    (read_obs step.observation)

let test_map_action_id () =
  let env = make_test_env () in
  let wrapped =
    Env.map_action ~action_space:(Env.action_space env) ~f:Fun.id env
  in
  equal ~msg:"id suffix" (option string) (Some "Test-v0/ActionWrapper")
    (Env.id wrapped)

(* map_reward *)

let test_map_reward () =
  let env = make_test_env () in
  let wrapped =
    Env.map_reward ~f:(fun ~reward ~info -> (reward *. 2.0, info)) env
  in
  let _obs, _info = Env.reset wrapped () in
  let step = Env.step wrapped action_right in
  equal ~msg:"doubled reward" (float 0.0) 2.0 step.reward

let test_map_reward_id () =
  let env = make_test_env () in
  let wrapped = Env.map_reward ~f:(fun ~reward ~info -> (reward, info)) env in
  equal ~msg:"id suffix" (option string) (Some "Test-v0/RewardWrapper")
    (Env.id wrapped)

(* clip_action *)

let make_box_action_env () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
  let last_action = ref 0.0 in
  let reset _env ?options:_ () =
    last_action := 0.0;
    (Rune.create Rune.float32 [| 1 |] [| 5.0 |], Info.empty)
  in
  let step _env action =
    let a : float array = Rune.to_array (Rune.reshape [| 1 |] action) in
    last_action := a.(0);
    Env.step_result
      ~observation:(Rune.create Rune.float32 [| 1 |] [| a.(0) |])
      ~reward:1.0 ()
  in
  let env =
    Env.create ~id:"BoxAct-v0" ~observation_space:obs_space
      ~action_space:act_space ~reset ~step ()
  in
  (env, last_action)

let test_clip_action () =
  let env, last_action = make_box_action_env () in
  let wrapped = Env.clip_action env in
  let _obs, _info = Env.reset wrapped () in
  let _step = Env.step wrapped (Rune.create Rune.float32 [| 1 |] [| 2.0 |]) in
  equal ~msg:"clamped to upper" (float 0.0) 1.0 !last_action;
  let _step = Env.step wrapped (Rune.create Rune.float32 [| 1 |] [| -0.5 |]) in
  equal ~msg:"clamped to lower" (float 0.0) 0.0 !last_action

(* clip_observation *)

let make_box_obs_env () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let obs_val = ref 5.0 in
  let reset _env ?options:_ () =
    obs_val := 5.0;
    (Rune.create Rune.float32 [| 1 |] [| !obs_val |], Info.empty)
  in
  let step _env action =
    let a : Int32.t array = Rune.to_array (Rune.reshape [| 1 |] action) in
    obs_val := !obs_val +. if Int32.to_int a.(0) = 0 then -3.0 else 3.0;
    Env.step_result
      ~observation:(Rune.create Rune.float32 [| 1 |] [| !obs_val |])
      ~reward:1.0 ()
  in
  Env.create ~id:"BoxObs-v0" ~observation_space:obs_space
    ~action_space:act_space ~reset ~step ()

let test_clip_observation () =
  let env = make_box_obs_env () in
  let wrapped = Env.clip_observation ~low:[| 2.0 |] ~high:[| 8.0 |] env in
  let _obs, _info = Env.reset wrapped () in
  (* Step right: inner obs = 8.0, clipped to 8.0 *)
  let s1 = Env.step wrapped action_right in
  let arr1 : float array =
    Rune.to_array (Rune.reshape [| 1 |] s1.observation)
  in
  equal ~msg:"clipped to upper" (float 0.0) 8.0 arr1.(0);
  let _obs, _info = Env.reset wrapped () in
  (* Step left: inner obs = 2.0, within bounds *)
  let s2 = Env.step wrapped action_left in
  let arr2 : float array =
    Rune.to_array (Rune.reshape [| 1 |] s2.observation)
  in
  equal ~msg:"within bounds" (float 0.0) 2.0 arr2.(0)

let test_clip_observation_space () =
  let env = make_box_obs_env () in
  let wrapped = Env.clip_observation ~low:[| 2.0 |] ~high:[| 8.0 |] env in
  let low, high = Space.Box.bounds (Env.observation_space wrapped) in
  equal ~msg:"clipped low" (array (float 0.0)) [| 2.0 |] low;
  equal ~msg:"clipped high" (array (float 0.0)) [| 8.0 |] high

(* time_limit *)

let test_time_limit_truncation () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:3 env in
  let _obs, _info = Env.reset wrapped () in
  let s1 = Env.step wrapped action_right in
  is_false ~msg:"step 1 not truncated" s1.truncated;
  let s2 = Env.step wrapped action_right in
  is_false ~msg:"step 2 not truncated" s2.truncated;
  let s3 = Env.step wrapped action_right in
  is_true ~msg:"step 3 truncated" s3.truncated

let test_time_limit_info () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:2 env in
  let _obs, _info = Env.reset wrapped () in
  let _s1 = Env.step wrapped action_right in
  let s2 = Env.step wrapped action_right in
  is_some ~msg:"time_limit.truncated present"
    (Info.find "time_limit.truncated" s2.info);
  is_some ~msg:"time_limit.elapsed_steps present"
    (Info.find "time_limit.elapsed_steps" s2.info)

let test_time_limit_info_values () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:2 env in
  let _obs, _info = Env.reset wrapped () in
  let _s1 = Env.step wrapped action_right in
  let s2 = Env.step wrapped action_right in
  let tl_truncated = Info.find_exn "time_limit.truncated" s2.info in
  equal ~msg:"truncated is Bool true" value (Value.Bool true) tl_truncated;
  let tl_steps = Info.find_exn "time_limit.elapsed_steps" s2.info in
  equal ~msg:"elapsed_steps is Int 2" value (Value.Int 2) tl_steps

let test_time_limit_counter_resets () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:3 env in
  let _obs, _info = Env.reset wrapped () in
  for _ = 1 to 3 do
    ignore (Env.step wrapped action_right)
  done;
  let _obs, _info = Env.reset wrapped () in
  let s1 = Env.step wrapped action_right in
  is_false ~msg:"counter reset after new episode" s1.truncated

let test_time_limit_nonpositive () =
  let env = make_test_env () in
  raises_invalid_arg ~msg:"max_episode_steps=0"
    "Env.time_limit: max_episode_steps must be positive" (fun () ->
      Env.time_limit ~max_episode_steps:0 env);
  raises_invalid_arg ~msg:"max_episode_steps=-1"
    "Env.time_limit: max_episode_steps must be positive" (fun () ->
      Env.time_limit ~max_episode_steps:(-1) env)

let test_time_limit_needs_reset () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:2 env in
  let _obs, _info = Env.reset wrapped () in
  let _s1 = Env.step wrapped action_right in
  let s2 = Env.step wrapped action_right in
  is_true ~msg:"truncated at limit" s2.truncated;
  raises_invalid_arg ~msg:"step after time_limit truncation"
    "Env: operation 'step' requires calling reset first" (fun () ->
      Env.step wrapped action_right)

let () =
  Rune.Rng.run ~seed:42 @@ fun () ->
  run "Fehu.Env (wrappers)"
    [
      group "state sharing"
        [
          test "close wrapper closes inner" test_close_wrapper_closes_inner;
          test "close inner closes wrapper" test_close_inner_closes_wrapper;
          test "reset wrapper clears inner" test_reset_wrapper_clears_inner;
        ];
      group "map_observation"
        [
          test "doubles reset obs" test_map_observation_reset;
          test "doubles step obs" test_map_observation_step;
          test "id suffix" test_map_observation_id;
        ];
      group "map_action"
        [
          test "flip reverses direction" test_map_action_flip;
          test "id suffix" test_map_action_id;
        ];
      group "map_reward"
        [
          test "doubles reward" test_map_reward;
          test "id suffix" test_map_reward_id;
        ];
      group "clip_action" [ test "clamps out-of-bounds" test_clip_action ];
      group "clip_observation"
        [
          test "clamps to explicit bounds" test_clip_observation;
          test "observation space reflects bounds" test_clip_observation_space;
        ];
      group "time_limit"
        [
          test "truncation at limit" test_time_limit_truncation;
          test "info keys present" test_time_limit_info;
          test "info values correct" test_time_limit_info_values;
          test "counter resets on new episode" test_time_limit_counter_resets;
          test "nonpositive raises" test_time_limit_nonpositive;
          test "needs reset after truncation" test_time_limit_needs_reset;
        ];
    ]
