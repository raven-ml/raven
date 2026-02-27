open Fehu
open Fehu_envs
open Windtrap

let read_float obs =
  let arr : float array = Nx.to_array (Nx.reshape [| 1 |] obs) in
  arr.(0)

let read_int32_array obs n =
  let arr : Int32.t array = Nx.to_array (Nx.reshape [| n |] obs) in
  Array.map Int32.to_int arr

let discrete action = Nx.create Nx.int32 [| 1 |] [| Int32.of_int action |]

(* Random_walk *)

let test_rw_creation () =
  let env = Random_walk.make () in
  match Env.id env with
  | Some id ->
      is_true ~msg:"id starts with RandomWalk"
        (String.length id >= 10 && String.sub id 0 10 = "RandomWalk")
  | None -> fail "expected an id"

let test_rw_reset_obs () =
  let env = Random_walk.make () in
  let obs, _info = Env.reset env () in
  equal ~msg:"reset obs is 0.0" (float 1e-6) 0.0 (read_float obs)

let test_rw_step_left () =
  let env = Random_walk.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 0) in
  equal ~msg:"step left to -1.0" (float 1e-6) (-1.0) (read_float s.observation)

let test_rw_step_right () =
  let env = Random_walk.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 1) in
  equal ~msg:"step right to 1.0" (float 1e-6) 1.0 (read_float s.observation)

let test_rw_termination () =
  let env = Random_walk.make () in
  let _obs, _info = Env.reset env () in
  let terminated = ref false in
  for _ = 1 to 20 do
    if not !terminated then begin
      let s = Env.step env (discrete 1) in
      if s.terminated then terminated := true
      else if s.truncated then begin
        let _obs, _info = Env.reset env () in
        ()
      end
    end
  done;
  is_true ~msg:"terminated at boundary" !terminated

let test_rw_ansi_render () =
  let env = Random_walk.make ~render_mode:`Ansi () in
  let _obs, _info = Env.reset env () in
  match Env.render env with
  | Some s -> is_true ~msg:"non-empty render" (String.length s > 0)
  | None -> fail "expected Some render"

(* Cartpole *)

let test_cp_creation () =
  let env = Cartpole.make () in
  match Env.id env with
  | Some id ->
      is_true ~msg:"id starts with CartPole"
        (String.length id >= 8 && String.sub id 0 8 = "CartPole")
  | None -> fail "expected an id"

let test_cp_reset_shape () =
  let env = Cartpole.make () in
  let obs, _info = Env.reset env () in
  let shape = Nx.shape obs in
  equal ~msg:"obs shape [4]" (array int) [| 4 |] shape

let test_cp_step_reward () =
  let env = Cartpole.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 1) in
  is_false ~msg:"not terminated on first step" s.terminated;
  equal ~msg:"reward 1.0" (float 1e-6) 1.0 s.reward

let test_cp_termination () =
  let env = Cartpole.make () in
  let _obs, _info = Env.reset env () in
  let done_flag = ref false in
  for _ = 1 to 600 do
    if not !done_flag then begin
      let s = Env.step env (discrete 0) in
      if s.terminated || s.truncated then done_flag := true
    end
  done;
  is_true ~msg:"episode ends" !done_flag

(* Grid_world *)

let test_gw_creation () =
  let env = Grid_world.make () in
  match Env.id env with
  | Some id ->
      is_true ~msg:"id starts with GridWorld"
        (String.length id >= 9 && String.sub id 0 9 = "GridWorld")
  | None -> fail "expected an id"

let test_gw_reset_obs () =
  let env = Grid_world.make () in
  let obs, _info = Env.reset env () in
  let pos = read_int32_array obs 2 in
  equal ~msg:"row = 0" int 0 pos.(0);
  equal ~msg:"col = 0" int 0 pos.(1)

let test_gw_move_down () =
  let env = Grid_world.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 1) in
  let pos = read_int32_array s.observation 2 in
  equal ~msg:"row = 1 after down" int 1 pos.(0)

let test_gw_move_right () =
  let env = Grid_world.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 3) in
  let pos = read_int32_array s.observation 2 in
  equal ~msg:"col = 1 after right" int 1 pos.(1)

let test_gw_obstacle () =
  let env = Grid_world.make () in
  let _obs, _info = Env.reset env () in
  (* Navigate to (1, 2): down, right, right *)
  let _s = Env.step env (discrete 1) in
  let _s = Env.step env (discrete 3) in
  let s = Env.step env (discrete 3) in
  let pos = read_int32_array s.observation 2 in
  equal ~msg:"at (1,2)" int 1 pos.(0);
  equal ~msg:"at (1,2)" int 2 pos.(1);
  (* Try to move down into obstacle at (2,2) *)
  let s = Env.step env (discrete 1) in
  let pos = read_int32_array s.observation 2 in
  equal ~msg:"blocked row still 1" int 1 pos.(0);
  equal ~msg:"blocked col still 2" int 2 pos.(1)

let test_gw_reach_goal () =
  let env = Grid_world.make () in
  let _obs, _info = Env.reset env () in
  (* Path to (4,4) avoiding obstacle at (2,2): down 4 times to row 4, then right
     4 times to col 4 *)
  for _ = 1 to 4 do
    ignore (Env.step env (discrete 1))
  done;
  let s_right1 = Env.step env (discrete 3) in
  is_false ~msg:"not done yet" s_right1.terminated;
  let _s = Env.step env (discrete 3) in
  let _s = Env.step env (discrete 3) in
  let s = Env.step env (discrete 3) in
  is_true ~msg:"terminated at goal" s.terminated;
  equal ~msg:"reward 10.0" (float 1e-6) 10.0 s.reward

let test_gw_ansi_render () =
  let env = Grid_world.make ~render_mode:`Ansi () in
  let _obs, _info = Env.reset env () in
  match Env.render env with
  | Some (Grid_world.Text s) ->
      is_true ~msg:"non-empty render" (String.length s > 0)
  | Some (Grid_world.Image _) -> fail "expected Text render"
  | None -> fail "expected Some render"

(* Mountain_car *)

let test_mc_creation () =
  let env = Mountain_car.make () in
  match Env.id env with
  | Some id ->
      is_true ~msg:"id starts with MountainCar"
        (String.length id >= 11 && String.sub id 0 11 = "MountainCar")
  | None -> fail "expected an id"

let test_mc_reset_shape () =
  let env = Mountain_car.make () in
  let obs, _info = Env.reset env () in
  let shape = Nx.shape obs in
  equal ~msg:"obs shape [2]" (array int) [| 2 |] shape

let test_mc_step_coast () =
  let env = Mountain_car.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 1) in
  let shape = Nx.shape s.observation in
  equal ~msg:"obs shape after step" (array int) [| 2 |] shape;
  is_false ~msg:"not terminated" s.terminated

let test_mc_reward () =
  let env = Mountain_car.make () in
  let _obs, _info = Env.reset env () in
  let s = Env.step env (discrete 1) in
  equal ~msg:"reward -1.0" (float 1e-6) (-1.0) s.reward

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  run "Fehu_envs"
    [
      group "RandomWalk"
        [
          test "creation" test_rw_creation;
          test "reset observation" test_rw_reset_obs;
          test "step left" test_rw_step_left;
          test "step right" test_rw_step_right;
          test "termination at boundary" test_rw_termination;
          test "ansi render" test_rw_ansi_render;
        ];
      group "CartPole"
        [
          test "creation" test_cp_creation;
          test "reset shape" test_cp_reset_shape;
          test "step reward" test_cp_step_reward;
          test "termination" test_cp_termination;
        ];
      group "GridWorld"
        [
          test "creation" test_gw_creation;
          test "reset observation" test_gw_reset_obs;
          test "move down" test_gw_move_down;
          test "move right" test_gw_move_right;
          test "obstacle blocks movement" test_gw_obstacle;
          test "reach goal" test_gw_reach_goal;
          test "ansi render" test_gw_ansi_render;
        ];
      group "MountainCar"
        [
          test "creation" test_mc_creation;
          test "reset shape" test_mc_reset_shape;
          test "step coast" test_mc_step_coast;
          test "reward" test_mc_reward;
        ];
    ]
