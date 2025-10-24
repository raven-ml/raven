open Fehu

let test_replay_create () =
  let buffer = Buffer.Replay.create ~capacity:10 in
  Alcotest.(check int) "initial size is 0" 0 (Buffer.Replay.size buffer);
  Alcotest.(check bool) "buffer not full" false (Buffer.Replay.is_full buffer)

let test_replay_add_and_sample () =
  let buffer = Buffer.Replay.create ~capacity:5 in
  let rng = Rune.Rng.key 42 in

  (* Add some transitions *)
  for i = 0 to 2 do
    let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int i |] in
    let next_obs =
      Rune.create Rune.float32 [| 1 |] [| float_of_int (i + 1) |]
    in
    let action = Rune.create Rune.int32 [| 1 |] [| Int32.of_int i |] in
    let transition =
      Buffer.
        {
          observation = obs;
          action;
          reward = float_of_int i;
          next_observation = next_obs;
          terminated = false;
          truncated = false;
        }
    in
    Buffer.Replay.add buffer transition
  done;

  Alcotest.(check int) "size after adding 3" 3 (Buffer.Replay.size buffer);

  (* Sample *)
  let rng = (Rune.Rng.split rng ~n:2).(0) in
  let samples = Buffer.Replay.sample buffer ~rng ~batch_size:2 in
  Alcotest.(check int) "sampled batch size" 2 (Array.length samples)

let test_replay_circular () =
  let buffer = Buffer.Replay.create ~capacity:3 in

  (* Add 5 transitions to a buffer of capacity 3 *)
  for i = 0 to 4 do
    let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int i |] in
    let next_obs =
      Rune.create Rune.float32 [| 1 |] [| float_of_int (i + 1) |]
    in
    let action = Rune.create Rune.int32 [| 1 |] [| Int32.of_int i |] in
    let transition =
      Buffer.
        {
          observation = obs;
          action;
          reward = float_of_int i;
          next_observation = next_obs;
          terminated = false;
          truncated = false;
        }
    in
    Buffer.Replay.add buffer transition
  done;

  Alcotest.(check int) "size capped at capacity" 3 (Buffer.Replay.size buffer);
  Alcotest.(check bool) "buffer is full" true (Buffer.Replay.is_full buffer)

let test_replay_clear () =
  let buffer = Buffer.Replay.create ~capacity:5 in

  let obs = Rune.create Rune.float32 [| 1 |] [| 1.0 |] in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let transition =
    Buffer.
      {
        observation = obs;
        action;
        reward = 1.0;
        next_observation = obs;
        terminated = false;
        truncated = false;
      }
  in
  Buffer.Replay.add buffer transition;

  Alcotest.(check int) "size before clear" 1 (Buffer.Replay.size buffer);
  Buffer.Replay.clear buffer;
  Alcotest.(check int) "size after clear" 0 (Buffer.Replay.size buffer)

let test_rollout_create () =
  let buffer = Buffer.Rollout.create ~capacity:10 in
  Alcotest.(check int) "initial size is 0" 0 (Buffer.Rollout.size buffer);
  Alcotest.(check bool) "buffer not full" false (Buffer.Rollout.is_full buffer)

let test_rollout_add_and_get () =
  let buffer = Buffer.Rollout.create ~capacity:5 in

  (* Add some steps *)
  for i = 0 to 2 do
    let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int i |] in
    let action = Rune.create Rune.int32 [| 1 |] [| Int32.of_int i |] in
    let step =
      Buffer.
        {
          observation = obs;
          action;
          reward = float_of_int i;
          terminated = false;
          truncated = false;
          value = Some (float_of_int i);
          log_prob = Some 0.0;
        }
    in
    Buffer.Rollout.add buffer step
  done;

  Alcotest.(check int) "size after adding 3" 3 (Buffer.Rollout.size buffer);

  let steps, advantages, returns = Buffer.Rollout.get buffer in
  Alcotest.(check int) "got 3 steps" 3 (Array.length steps);
  Alcotest.(check int) "got 3 advantages" 3 (Array.length advantages);
  Alcotest.(check int) "got 3 returns" 3 (Array.length returns);
  Alcotest.(check int) "size after get" 0 (Buffer.Rollout.size buffer)

let test_rollout_compute_advantages () =
  let buffer = Buffer.Rollout.create ~capacity:5 in

  (* Add steps with known values *)
  for i = 0 to 2 do
    let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int i |] in
    let action = Rune.create Rune.int32 [| 1 |] [| Int32.of_int i |] in
    let step =
      Buffer.
        {
          observation = obs;
          action;
          reward = 1.0;
          terminated = false;
          truncated = false;
          value = Some 0.0;
          log_prob = Some 0.0;
        }
    in
    Buffer.Rollout.add buffer step
  done;

  (* Compute advantages *)
  Buffer.Rollout.compute_advantages buffer ~last_value:0.0 ~last_done:false
    ~gamma:0.99 ~gae_lambda:0.95;

  let _, advantages, returns = Buffer.Rollout.get buffer in
  Alcotest.(check int) "advantages computed" 3 (Array.length advantages);
  Alcotest.(check int) "returns computed" 3 (Array.length returns);

  (* All advantages should be positive since rewards are 1.0 and values are
     0.0 *)
  Array.iter
    (fun adv -> Alcotest.(check bool) "advantage > 0" true (adv > 0.0))
    advantages

let test_rollout_compute_advantages_truncated () =
  let buffer = Buffer.Rollout.create ~capacity:1 in
  let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let step =
    Buffer.
      {
        observation = obs;
        action;
        reward = 1.0;
        terminated = false;
        truncated = true;
        value = Some 0.5;
        log_prob = None;
      }
  in
  Buffer.Rollout.add buffer step;
  Buffer.Rollout.compute_advantages buffer ~last_value:10.0 ~last_done:false
    ~gamma:0.99 ~gae_lambda:1.0;
  let _, advantages, returns = Buffer.Rollout.get buffer in
  Alcotest.(check (float 1e-6))
    "advantage treats truncation as terminal" 0.5 advantages.(0);
  Alcotest.(check (float 1e-6)) "return respects truncation" 1.0 returns.(0)

let test_rollout_clear () =
  let buffer = Buffer.Rollout.create ~capacity:5 in

  let obs = Rune.create Rune.float32 [| 1 |] [| 1.0 |] in
  let action = Rune.create Rune.int32 [| 1 |] [| 0l |] in
  let step =
    Buffer.
      {
        observation = obs;
        action;
        reward = 1.0;
        terminated = false;
        truncated = false;
        value = None;
        log_prob = None;
      }
  in
  Buffer.Rollout.add buffer step;

  Alcotest.(check int) "size before clear" 1 (Buffer.Rollout.size buffer);
  Buffer.Rollout.clear buffer;
  Alcotest.(check int) "size after clear" 0 (Buffer.Rollout.size buffer)

let () =
  let open Alcotest in
  run "Buffer"
    [
      ( "Replay",
        [
          test_case "create replay buffer" `Quick test_replay_create;
          test_case "add and sample" `Quick test_replay_add_and_sample;
          test_case "circular buffer behavior" `Quick test_replay_circular;
          test_case "clear buffer" `Quick test_replay_clear;
        ] );
      ( "Rollout",
        [
          test_case "create rollout buffer" `Quick test_rollout_create;
          test_case "add and get" `Quick test_rollout_add_and_get;
          test_case "compute advantages" `Quick test_rollout_compute_advantages;
          test_case "compute advantages with truncation" `Quick
            test_rollout_compute_advantages_truncated;
          test_case "clear buffer" `Quick test_rollout_clear;
        ] );
    ]
