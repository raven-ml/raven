(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Fehu_envs
open Windtrap

let test_random_walk_creation () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  match Env.id env with
  | Some id ->
      equal ~msg:"random walk id" bool true
        (String.starts_with ~prefix:"RandomWalk" id)
  | None -> fail "expected env id"

let test_random_walk_reset () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let obs, _ = Env.reset env () in
  let arr : float array = Rune.to_array (Rune.reshape [| 1 |] obs) in
  equal ~msg:"random walk starts at 0" (float 0.01) 0.0 arr.(0)

let test_random_walk_step_left () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let transition = Env.step env action in
  let arr : float array =
    Rune.to_array (Rune.reshape [| 1 |] transition.Env.observation)
  in
  equal ~msg:"moved left" (float 0.01) (-1.0) arr.(0)

let test_random_walk_step_right () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let arr : float array =
    Rune.to_array (Rune.reshape [| 1 |] transition.Env.observation)
  in
  equal ~msg:"moved right" (float 0.01) 1.0 arr.(0)

let test_random_walk_termination () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let rec step_to_boundary n =
    if n > 20 then fail "did not terminate"
    else
      let transition = Env.step env action in
      if transition.Env.terminated then transition else step_to_boundary (n + 1)
  in
  let final_transition = step_to_boundary 0 in
  equal ~msg:"terminated" bool true final_transition.Env.terminated

let test_random_walk_truncation () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let rec step_until_done n =
    if n > 250 then fail "did not terminate or truncate in 250 steps"
    else
      let transition = Env.step env action in
      if transition.Env.truncated || transition.Env.terminated then
        (n, transition)
      else step_until_done (n + 1)
  in
  let steps, final_transition = step_until_done 1 in
  if final_transition.Env.terminated then
    equal ~msg:"terminated before truncation" pass () ()
  else equal ~msg:"truncated at step 200" int 200 steps

let test_random_walk_render () =
  let rng = Rune.Rng.key 42 in
  let env = Random_walk.make ~rng () in
  let _, _ = Env.reset env () in
  match Env.render env with
  | Some str ->
      equal ~msg:"render produces string" bool true (String.length str > 0)
  | None -> fail "expected render output"

let test_grid_world_creation () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  match Env.id env with
  | Some id ->
      equal ~msg:"grid world id" bool true
        (String.starts_with ~prefix:"GridWorld" id)
  | None -> fail "expected env id"

let test_grid_world_reset () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let obs, _ = Env.reset env () in
  let arr : Int32.t array = Rune.to_array (Rune.reshape [| 2 |] obs) in
  equal ~msg:"starts at row 0" int32 0l arr.(0);
  equal ~msg:"starts at col 0" int32 0l arr.(1)

let test_grid_world_move_down () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 1l |] in
  let transition = Env.step env action in
  let arr : Int32.t array =
    Rune.to_array (Rune.reshape [| 2 |] transition.Env.observation)
  in
  equal ~msg:"moved to row 1" int32 1l arr.(0);
  equal ~msg:"stayed at col 0" int32 0l arr.(1)

let test_grid_world_move_right () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  let action = Rune.create Rune.int32 [| 1 |] [| 3l |] in
  let transition = Env.step env action in
  let arr : Int32.t array =
    Rune.to_array (Rune.reshape [| 2 |] transition.Env.observation)
  in
  equal ~msg:"stayed at row 0" int32 0l arr.(0);
  equal ~msg:"moved to col 1" int32 1l arr.(1)

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
  equal ~msg:"cannot enter obstacle" int32 2l arr.(0);
  equal ~msg:"stays before obstacle" int32 1l arr.(1)

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
  equal ~msg:"reached goal" bool true transition.Env.terminated;
  equal ~msg:"goal reward" (float 0.01) 10.0 transition.Env.reward

let test_grid_world_render () =
  let rng = Rune.Rng.key 42 in
  let env = Grid_world.make ~rng () in
  let _, _ = Env.reset env () in
  match Env.render env with
  | Some (Render.Text text) ->
      equal ~msg:"render produces text" bool true (String.length text > 0)
  | Some (Render.Image image) ->
      equal ~msg:"render produces image" bool true (image.width > 0)
  | Some Render.None -> fail "unexpected empty render frame"
  | Some (Render.Svg svg) ->
      equal ~msg:"render produces svg" bool true (String.length svg > 0)
  | None -> fail "expected render output"

let () =
  run "Environments"
    [
      group "RandomWalk"
        [
          test "creation" test_random_walk_creation;
          test "reset" test_random_walk_reset;
          test "step left" test_random_walk_step_left;
          test "step right" test_random_walk_step_right;
          test "termination" test_random_walk_termination;
          test "truncation" test_random_walk_truncation;
          test "render" test_random_walk_render;
        ];
      group "GridWorld"
        [
          test "creation" test_grid_world_creation;
          test "reset" test_grid_world_reset;
          test "move down" test_grid_world_move_down;
          test "move right" test_grid_world_move_right;
          test "obstacle collision" test_grid_world_obstacle;
          test "reach goal" test_grid_world_goal;
          test "render" test_grid_world_render;
        ];
    ]
