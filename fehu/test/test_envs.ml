(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Fehu_envs

let test_random_walk_creation () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  match Env.id env with
  | Some id ->
      Alcotest.(check bool)
        "random walk id" true
        (String.starts_with ~prefix:"RandomWalk" id)
  | None -> Alcotest.fail "expected env id"

let test_random_walk_reset () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let obs, _ = Env.reset env () in
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  Alcotest.(check (float 0.01)) "random walk starts at 0" 0.0 arr.(0)

let test_random_walk_step_left () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let transition = Env.step env action in
  let arr : float array =
    Rune.to_array (Rune.reshape [| 1 |] transition.Env.observation)
  in
  Alcotest.(check (float 0.01)) "moved left" (-1.0) arr.(0)

let test_random_walk_step_right () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let arr : float array =
    Rune.to_array (Rune.reshape [| 1 |] transition.Env.observation)
  in
  Alcotest.(check (float 0.01)) "moved right" 1.0 arr.(0)

let test_random_walk_termination () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let rec step_to_boundary n =
    if n > 20 then Alcotest.fail "did not terminate"
    else
      let transition = Env.step env action in
      if transition.Env.terminated then transition else step_to_boundary (n + 1)
  in
  let final_transition = step_to_boundary 0 in
  Alcotest.(check bool) "terminated" true final_transition.Env.terminated

let test_random_walk_truncation () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let rec step_until_done n =
    if n > 250 then Alcotest.fail "did not terminate or truncate in 250 steps"
    else
      let transition = Env.step env action in
      if transition.Env.truncated || transition.Env.terminated then
        (n, transition)
      else step_until_done (n + 1)
  in
  let steps, final_transition = step_until_done 1 in
  if final_transition.Env.terminated then
    Alcotest.(check pass) "terminated before truncation" () ()
  else Alcotest.(check int) "truncated at step 200" 200 steps

let test_random_walk_render () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  match Env.render env with
  | Some str ->
      Alcotest.(check bool) "render produces string" true (String.length str > 0)
  | None -> Alcotest.fail "expected render output"

let test_grid_world_creation () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  match Env.id env with
  | Some id ->
      Alcotest.(check bool)
        "grid world id" true
        (String.starts_with ~prefix:"GridWorld" id)
  | None -> Alcotest.fail "expected env id"

let test_grid_world_reset () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let obs, _ = Env.reset env () in
  let arr : Int32.t array = Rune.to_array (Rune.reshape [| 2 |] obs) in
  Alcotest.(check int32) "starts at row 0" 0l arr.(0);
  Alcotest.(check int32) "starts at col 0" 0l arr.(1)

let test_grid_world_move_down () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let arr : Int32.t array =
    Rune.to_array (Rune.reshape [| 2 |] transition.Env.observation)
  in
  Alcotest.(check int32) "moved to row 1" 1l arr.(0);
  Alcotest.(check int32) "stayed at col 0" 0l arr.(1)

let test_grid_world_move_right () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 3l |] in
  let transition = Env.step env action in
  let arr : Int32.t array =
    Rune.to_array (Rune.reshape [| 2 |] transition.Env.observation)
  in
  Alcotest.(check int32) "stayed at row 0" 0l arr.(0);
  Alcotest.(check int32) "moved to col 1" 1l arr.(1)

let test_grid_world_obstacle () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let move_down = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let move_right = Rune.create Rune.int32 [| 1 |] [| 3l |] in
  let _ = Env.step env move_down in
  let _ = Env.step env move_down in
  let _ = Env.step env move_right in
  let transition = Env.step env move_right in
  let arr : Int32.t array =
    Rune.to_array (Rune.reshape [| 2 |] transition.Env.observation)
  in
  Alcotest.(check int32) "cannot enter obstacle" 2l arr.(0);
  Alcotest.(check int32) "stays before obstacle" 1l arr.(1)

let test_grid_world_goal () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let move_down = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let move_right = Rune.create Rune.int32 [| 1 |] [| 3l |] in
  let reach_goal () =
    let _ = Env.step env move_down in
    let _ = Env.step env move_down in
    let _ = Env.step env move_down in
    let _ = Env.step env move_down in
    let _ = Env.step env move_right in
    let _ = Env.step env move_right in
    let _ = Env.step env move_right in
    Env.step env move_right
  in
  let transition = reach_goal () in
  Alcotest.(check bool) "reached goal" true transition.Env.terminated;
  Alcotest.(check (float 0.01)) "goal reward" 10.0 transition.Env.reward

let test_grid_world_render () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  match Env.render env with
  | Some (Render.Text text) ->
      Alcotest.(check bool) "render produces text" true (String.length text > 0)
  | Some (Render.Image image) ->
      Alcotest.(check bool) "render produces image" true (image.width > 0)
  | Some Render.None -> Alcotest.fail "unexpected empty render frame"
  | Some (Render.Svg svg) ->
      Alcotest.(check bool) "render produces svg" true (String.length svg > 0)
  | None -> Alcotest.fail "expected render output"

let () =
  let open Alcotest in
  run "Environments"
    [
      ( "RandomWalk",
        [
          test_case "creation" `Quick test_random_walk_creation;
          test_case "reset" `Quick test_random_walk_reset;
          test_case "step left" `Quick test_random_walk_step_left;
          test_case "step right" `Quick test_random_walk_step_right;
          test_case "termination" `Quick test_random_walk_termination;
          test_case "truncation" `Quick test_random_walk_truncation;
          test_case "render" `Quick test_random_walk_render;
        ] );
      ( "GridWorld",
        [
          test_case "creation" `Quick test_grid_world_creation;
          test_case "reset" `Quick test_grid_world_reset;
          test_case "move down" `Quick test_grid_world_move_down;
          test_case "move right" `Quick test_grid_world_move_right;
          test_case "obstacle collision" `Quick test_grid_world_obstacle;
          test_case "reach goal" `Quick test_grid_world_goal;
          test_case "render" `Quick test_grid_world_render;
        ] );
    ]
