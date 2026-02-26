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

(* Creation *)

let test_id () =
  let env = make_test_env () in
  equal ~msg:"id is Some Test-v0" (option string) (Some "Test-v0") (Env.id env)

let test_observation_space () =
  let env = make_test_env () in
  let low, high = Space.Box.bounds (Env.observation_space env) in
  equal ~msg:"obs low" (array (float 0.0)) [| 0.0 |] low;
  equal ~msg:"obs high" (array (float 0.0)) [| 10.0 |] high

let test_action_space () =
  let env = make_test_env () in
  equal ~msg:"act n" int 2 (Space.Discrete.n (Env.action_space env))

let test_render_mode_default () =
  let env = make_test_env () in
  is_none ~msg:"render_mode default is None" (Env.render_mode env)

let test_render_mode_invalid () =
  raises_invalid_arg ~msg:"render_mode not in render_modes"
    "Env.create: render mode 'human' not in render_modes []" (fun () ->
      let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 1.0 |] in
      let act_space = Space.Discrete.create 2 in
      Env.create ~observation_space:obs_space ~action_space:act_space
        ~render_mode:`Human ~render_modes:[]
        ~reset:(fun _env ?options:_ () -> assert false)
        ~step:(fun _env _ -> assert false)
        ())

(* Lifecycle *)

let test_reset_obs () =
  let env = make_test_env () in
  let obs, _info = Env.reset env () in
  equal ~msg:"reset obs shape" (array int) [| 1 |] (Rune.shape obs);
  equal ~msg:"reset obs value" (float 0.0) 5.0 (read_obs obs)

let test_step_after_reset () =
  let env = make_test_env () in
  let _obs, _info = Env.reset env () in
  let step = Env.step env action_right in
  equal ~msg:"reward" (float 0.0) 1.0 step.reward;
  is_false ~msg:"not terminated" step.terminated;
  is_false ~msg:"not truncated" step.truncated

let test_step_before_reset () =
  let env = make_test_env () in
  raises_invalid_arg ~msg:"step before reset"
    "Env: operation 'step' requires calling reset first" (fun () ->
      Env.step env action_left)

let test_step_after_terminal () =
  let env = make_test_env () in
  let _obs, _info = Env.reset env () in
  (* Move left 5 times: 5 -> 4 -> 3 -> 2 -> 1 -> 0, terminated *)
  for _ = 1 to 4 do
    ignore (Env.step env action_left)
  done;
  let step = Env.step env action_left in
  is_true ~msg:"terminated" step.terminated;
  raises_invalid_arg ~msg:"step after terminal"
    "Env: operation 'step' requires calling reset first" (fun () ->
      Env.step env action_left)

let test_reset_after_terminal () =
  let env = make_test_env () in
  let _obs, _info = Env.reset env () in
  for _ = 1 to 5 do
    ignore (Env.step env action_left)
  done;
  let obs, _info = Env.reset env () in
  equal ~msg:"reset clears terminal" (float 0.0) 5.0 (read_obs obs)

let test_close () =
  let env = make_test_env () in
  Env.close env;
  is_true ~msg:"closed" (Env.closed env)

let test_step_on_closed () =
  let env = make_test_env () in
  let _obs, _info = Env.reset env () in
  Env.close env;
  raises_invalid_arg ~msg:"step on closed"
    "Env: operation 'step' on a closed environment" (fun () ->
      Env.step env action_left)

let test_reset_on_closed () =
  let env = make_test_env () in
  Env.close env;
  raises_invalid_arg ~msg:"reset on closed"
    "Env: operation 'reset' on a closed environment" (fun () ->
      Env.reset env ())

let test_render_on_closed () =
  let env = make_test_env () in
  Env.close env;
  raises_invalid_arg ~msg:"render on closed"
    "Env: operation 'render' on a closed environment" (fun () -> Env.render env)

let test_close_idempotent () =
  let env = make_test_env () in
  Env.close env;
  Env.close env;
  is_true ~msg:"still closed" (Env.closed env)

(* step_result *)

let test_step_result_defaults () =
  let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
  let s = Env.step_result ~observation:obs () in
  equal ~msg:"default reward" (float 0.0) 0.0 s.reward;
  is_false ~msg:"default terminated" s.terminated;
  is_false ~msg:"default truncated" s.truncated;
  is_true ~msg:"default info empty" (Info.is_empty s.info)

let test_step_result_custom () =
  let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
  let info = Info.set "k" (Info.int 1) Info.empty in
  let s =
    Env.step_result ~observation:obs ~reward:5.0 ~terminated:true
      ~truncated:false ~info ()
  in
  equal ~msg:"custom reward" (float 0.0) 5.0 s.reward;
  is_true ~msg:"custom terminated" s.terminated;
  is_false ~msg:"custom truncated" s.truncated;
  is_some ~msg:"custom info has key" (Info.find "k" s.info)

(* time_limit lifecycle enforcement *)

let test_time_limit_needs_reset () =
  let env = make_test_env () in
  let wrapped = Env.time_limit ~max_episode_steps:3 env in
  let _obs, _info = Env.reset wrapped () in
  for _ = 1 to 2 do
    ignore (Env.step wrapped action_right)
  done;
  let s3 = Env.step wrapped action_right in
  is_true ~msg:"step 3 truncated" s3.truncated;
  raises_invalid_arg ~msg:"step after time_limit truncation requires reset"
    "Env: operation 'step' requires calling reset first" (fun () ->
      Env.step wrapped action_right)

let () =
  Rune.Rng.run ~seed:42 @@ fun () ->
  run "Fehu.Env"
    [
      group "creation"
        [
          test "id" test_id;
          test "observation_space" test_observation_space;
          test "action_space" test_action_space;
          test "render_mode default" test_render_mode_default;
          test "render_mode invalid" test_render_mode_invalid;
        ];
      group "lifecycle"
        [
          test "reset returns valid obs" test_reset_obs;
          test "step after reset" test_step_after_reset;
          test "step before reset" test_step_before_reset;
          test "step after terminal" test_step_after_terminal;
          test "reset after terminal" test_reset_after_terminal;
          test "close" test_close;
          test "step on closed" test_step_on_closed;
          test "reset on closed" test_reset_on_closed;
          test "render on closed" test_render_on_closed;
          test "close idempotent" test_close_idempotent;
          test "time_limit needs reset after truncation"
            test_time_limit_needs_reset;
        ];
      group "step_result"
        [
          test "defaults" test_step_result_defaults;
          test "custom values" test_step_result_custom;
        ];
    ]
