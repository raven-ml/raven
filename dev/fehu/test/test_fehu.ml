open Fehu
module Rng = Rune.Rng

let test_discrete_space () =
  let rng = Rng.key 42 in
  let space = Space.Discrete 5 in
  let action = Space.sample ~rng Rune.c space in
  let valid = Space.contains space action in
  Alcotest.(check bool) "discrete action is valid" true valid;
  let shape = Space.shape space in
  Alcotest.(check (array int)) "discrete space shape" [| 1 |] shape

let test_box_space () =
  let rng = Rng.key 42 in
  let low = Rune.create Rune.c Rune.float32 [| 2 |] [| -1.0; -2.0 |] in
  let high = Rune.create Rune.c Rune.float32 [| 2 |] [| 1.0; 2.0 |] in
  let space = Space.Box { low; high; shape = [| 2 |] } in
  let sample = Space.sample ~rng Rune.c space in
  let valid = Space.contains space sample in
  Alcotest.(check bool) "box sample is valid" true valid;
  let shape = Space.shape space in
  Alcotest.(check (array int)) "box space shape" [| 2 |] shape

let test_buffer () =
  let rng = Rng.key 42 in
  let buffer = Buffer.create ~capacity:10 in
  Alcotest.(check int) "initial size" 0 (Buffer.size buffer);
  Alcotest.(check bool) "not full initially" false (Buffer.is_full buffer);

  (* Add some transitions *)
  for i = 1 to 5 do
    let transition =
      Buffer.
        {
          obs = Rune.scalar Rune.c Rune.float32 (float_of_int i);
          action = Rune.scalar Rune.c Rune.float32 0.0;
          reward = 1.0;
          next_obs = Rune.scalar Rune.c Rune.float32 (float_of_int (i + 1));
          terminated = false;
        }
    in
    Buffer.add buffer transition
  done;

  Alcotest.(check int) "size after adding" 5 (Buffer.size buffer);
  Alcotest.(check bool) "still not full" false (Buffer.is_full buffer);

  (* Fill the buffer *)
  for i = 6 to 15 do
    let transition =
      Buffer.
        {
          obs = Rune.scalar Rune.c Rune.float32 (float_of_int i);
          action = Rune.scalar Rune.c Rune.float32 0.0;
          reward = 1.0;
          next_obs = Rune.scalar Rune.c Rune.float32 (float_of_int (i + 1));
          terminated = false;
        }
    in
    Buffer.add buffer transition
  done;

  Alcotest.(check int) "max size" 10 (Buffer.size buffer);
  Alcotest.(check bool) "is full" true (Buffer.is_full buffer);

  (* Test sampling *)
  let batch = Buffer.sample buffer ~rng ~batch_size:3 in
  Alcotest.(check int) "sampled batch size" 3 (Array.length batch)

let test_cartpole_env () =
  let env = Envs.cartpole () in

  (* Test reset *)
  let obs, info = env.reset () in
  let obs_shape = Rune.shape obs in
  Alcotest.(check (array int)) "observation shape" [| 4 |] obs_shape;
  Alcotest.(check (list pass)) "info is empty" [] info;

  (* Test step *)
  let action = Rune.scalar Rune.c Rune.float32 0.0 in
  let next_obs, reward, _terminated, _truncated, _info = env.step action in
  let next_obs_shape = Rune.shape next_obs in
  Alcotest.(check (array int)) "next observation shape" [| 4 |] next_obs_shape;
  Alcotest.(check (float 0.01)) "reward is positive" 1.0 reward

let test_mountain_car_env () =
  let env = Envs.mountain_car () in

  (* Test reset *)
  let obs, _info = env.reset ~seed:42 () in
  let obs_shape = Rune.shape obs in
  Alcotest.(check (array int)) "observation shape" [| 2 |] obs_shape;

  (* Test step *)
  let action = Rune.scalar Rune.c Rune.float32 1.0 in
  let next_obs, reward, _terminated, _truncated, _info = env.step action in
  let next_obs_shape = Rune.shape next_obs in
  Alcotest.(check (array int)) "next observation shape" [| 2 |] next_obs_shape;
  Alcotest.(check (float 0.01)) "reward is negative" (-1.0) reward

let test_pendulum_env () =
  let env = Envs.pendulum () in

  (* Test reset *)
  let obs, _info = env.reset () in
  let obs_shape = Rune.shape obs in
  Alcotest.(check (array int)) "observation shape" [| 3 |] obs_shape;

  (* Test step with continuous action *)
  let action = Rune.scalar Rune.c Rune.float32 0.5 in
  let next_obs, _reward, terminated, truncated, _info = env.step action in
  let next_obs_shape = Rune.shape next_obs in
  Alcotest.(check (array int)) "next observation shape" [| 3 |] next_obs_shape;
  Alcotest.(check bool) "pendulum never terminates" false terminated;
  Alcotest.(check bool) "pendulum never truncates" false truncated

let test_compute_gae () =
  let rewards = [| 1.0; 2.0; 3.0; 4.0 |] in
  let values = [| 0.5; 1.5; 2.5; 3.5 |] in
  let dones = [| false; false; false; true |] in

  let advantages, returns =
    Training.compute_gae ~rewards ~values ~dones ~gamma:0.99 ~gae_lambda:0.95
  in

  Alcotest.(check int) "advantages length" 4 (Array.length advantages);
  Alcotest.(check int) "returns length" 4 (Array.length returns);

  (* Check that returns = advantages + values *)
  for i = 0 to 3 do
    let expected = advantages.(i) +. values.(i) in
    Alcotest.(check (float 0.001))
      (Printf.sprintf "returns[%d] = advantages[%d] + values[%d]" i i i)
      expected returns.(i)
  done

let () =
  let open Alcotest in
  run "Fehu"
    [
      ( "Space",
        [
          test_case "Discrete space" `Quick test_discrete_space;
          test_case "Box space" `Quick test_box_space;
        ] );
      ("Buffer", [ test_case "Buffer operations" `Quick test_buffer ]);
      ( "Environments",
        [
          test_case "CartPole" `Quick test_cartpole_env;
          test_case "MountainCar" `Quick test_mountain_car_env;
          test_case "Pendulum" `Quick test_pendulum_env;
        ] );
      ("Training", [ test_case "Compute GAE" `Quick test_compute_gae ]);
    ]
