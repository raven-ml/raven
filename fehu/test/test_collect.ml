open Fehu
open Windtrap

let rng = Rune.Rng.key 42

let make_test_env ?(max_steps = 100) ~rng () =
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
  Env.create ~id:"Test-v0" ~rng ~observation_space:obs_space
    ~action_space:act_space ~reset ~step ()

(* Rollout *)

let test_rollout_length () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let traj = Collect.rollout env ~policy ~n_steps:5 in
  equal ~msg:"length = 5" int 5 (Collect.length traj)

let test_rollout_arrays_length () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let traj = Collect.rollout env ~policy ~n_steps:5 in
  equal ~msg:"observations" int 5 (Array.length traj.observations);
  equal ~msg:"actions" int 5 (Array.length traj.actions);
  equal ~msg:"rewards" int 5 (Array.length traj.rewards);
  equal ~msg:"next_observations" int 5 (Array.length traj.next_observations);
  equal ~msg:"terminated" int 5 (Array.length traj.terminated);
  equal ~msg:"truncated" int 5 (Array.length traj.truncated);
  equal ~msg:"infos" int 5 (Array.length traj.infos)

let test_rollout_next_obs_populated () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let traj = Collect.rollout env ~policy ~n_steps:3 in
  for i = 0 to 2 do
    let arr : float array =
      Rune.to_array (Rune.reshape [| 1 |] traj.next_observations.(i))
    in
    is_true ~msg:"next_obs is finite" (Float.is_finite arr.(0))
  done

let test_rollout_no_log_probs () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let traj = Collect.rollout env ~policy ~n_steps:3 in
  is_none ~msg:"log_probs" traj.log_probs;
  is_none ~msg:"values" traj.values

let test_rollout_with_log_probs () =
  let env = make_test_env ~rng () in
  let policy _obs =
    (Rune.create Rune.int32 [| 1 |] [| 1l |], Some (-0.5), Some 1.0)
  in
  let traj = Collect.rollout env ~policy ~n_steps:4 in
  is_some ~msg:"log_probs present" traj.log_probs;
  is_some ~msg:"values present" traj.values;
  equal ~msg:"log_probs length" int 4 (Array.length (Option.get traj.log_probs));
  equal ~msg:"values length" int 4 (Array.length (Option.get traj.values))

(* Episodes *)

let test_episodes_count () =
  let env = make_test_env ~max_steps:10 ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let eps = Collect.episodes env ~policy ~n_episodes:2 ~max_steps:10 () in
  equal ~msg:"2 episodes" int 2 (List.length eps)

let test_episodes_positive_length () =
  let env = make_test_env ~max_steps:10 ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let eps = Collect.episodes env ~policy ~n_episodes:2 ~max_steps:10 () in
  List.iter
    (fun ep ->
      is_true ~msg:"episode has positive length" (Collect.length ep > 0))
    eps

(* Concat *)

let test_concat_two () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let t1 = Collect.rollout env ~policy ~n_steps:3 in
  let t2 = Collect.rollout env ~policy ~n_steps:4 in
  let t = Collect.concat [ t1; t2 ] in
  equal ~msg:"total length" int 7 (Collect.length t)

let test_concat_empty_raises () =
  raises_invalid_arg "Collect.concat: empty list" (fun () -> Collect.concat [])

let test_concat_singleton () =
  let env = make_test_env ~rng () in
  let policy _obs = (Rune.create Rune.int32 [| 1 |] [| 1l |], None, None) in
  let t1 = Collect.rollout env ~policy ~n_steps:5 in
  let t = Collect.concat [ t1 ] in
  equal ~msg:"same length" int 5 (Collect.length t)

let () =
  run "Fehu.Collect"
    [
      group "rollout"
        [
          test "length" test_rollout_length;
          test "arrays length" test_rollout_arrays_length;
          test "next_observations populated" test_rollout_next_obs_populated;
          test "no log_probs/values" test_rollout_no_log_probs;
          test "with log_probs/values" test_rollout_with_log_probs;
        ];
      group "episodes"
        [
          test "count" test_episodes_count;
          test "positive length" test_episodes_positive_length;
        ];
      group "concat"
        [
          test "two trajectories" test_concat_two;
          test "empty raises" test_concat_empty_raises;
          test "singleton" test_concat_singleton;
        ];
    ]
