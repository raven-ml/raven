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

(* Run *)

let test_constant_reward_stats () =
  let env = make_test_env ~max_steps:5 () in
  let policy _obs = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let stats = Eval.run env ~policy ~n_episodes:3 ~max_steps:5 () in
  equal ~msg:"mean_reward" (float 1e-6) 5.0 stats.mean_reward;
  equal ~msg:"std_reward" (float 1e-6) 0.0 stats.std_reward;
  equal ~msg:"mean_length" (float 1e-6) 5.0 stats.mean_length;
  equal ~msg:"n_episodes" int 3 stats.n_episodes

let test_n_episodes_matches () =
  let env = make_test_env ~max_steps:5 () in
  let policy _obs = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let stats = Eval.run env ~policy ~n_episodes:7 ~max_steps:5 () in
  equal ~msg:"n_episodes matches" int 7 stats.n_episodes

let () =
  Rune.Rng.run ~seed:42 @@ fun () ->
  run "Fehu.Eval"
    [
      group "run"
        [
          test "constant reward statistics" test_constant_reward_stats;
          test "n_episodes matches stats" test_n_episodes_matches;
        ];
    ]
