(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

let test_replay_create () =
  let buffer = Buffer.Replay.create ~capacity:10 in
  equal ~msg:"initial size is 0" int 0 (Buffer.Replay.size buffer);
  equal ~msg:"buffer not full" bool false (Buffer.Replay.is_full buffer)

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

  equal ~msg:"size after adding 3" int 3 (Buffer.Replay.size buffer);

  (* Sample *)
  let keys = Rune.Rng.split rng ~n:2 in
  let samples = Buffer.Replay.sample buffer ~rng:keys.(0) ~batch_size:2 in
  equal ~msg:"sampled batch size" int 2 (Array.length samples);
  let observations, actions, rewards, next_obs, terminateds, truncateds =
    Buffer.Replay.sample_arrays buffer ~rng:keys.(1) ~batch_size:2
  in
  equal ~msg:"soa observations" int 2 (Array.length observations);
  equal ~msg:"soa actions" int 2 (Array.length actions);
  equal ~msg:"soa rewards" int 2 (Array.length rewards);
  equal ~msg:"soa next observations" int 2 (Array.length next_obs);
  equal ~msg:"soa terminateds" int 2 (Array.length terminateds);
  equal ~msg:"soa truncateds" int 2 (Array.length truncateds)

let test_replay_add_many () =
  let buffer = Buffer.Replay.create ~capacity:5 in
  let mk_transition idx =
    let float_value = float_of_int idx in
    let obs = Rune.create Rune.float32 [| 1 |] [| float_value |] in
    let action = Rune.create Rune.int32 [| 1 |] [| Int32.of_int idx |] in
    Buffer.
      {
        observation = obs;
        action;
        reward = float_value;
        next_observation = obs;
        terminated = false;
        truncated = false;
      }
  in
  let batch = Array.init 3 mk_transition in
  Buffer.Replay.add_many buffer batch;
  equal ~msg:"size after add_many" int 3 (Buffer.Replay.size buffer)

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

  equal ~msg:"size capped at capacity" int 3 (Buffer.Replay.size buffer);
  equal ~msg:"buffer is full" bool true (Buffer.Replay.is_full buffer)

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

  equal ~msg:"size before clear" int 1 (Buffer.Replay.size buffer);
  Buffer.Replay.clear buffer;
  equal ~msg:"size after clear" int 0 (Buffer.Replay.size buffer)

let test_rollout_create () =
  let buffer = Buffer.Rollout.create ~capacity:10 in
  equal ~msg:"initial size is 0" int 0 (Buffer.Rollout.size buffer);
  equal ~msg:"buffer not full" bool false (Buffer.Rollout.is_full buffer)

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

  equal ~msg:"size after adding 3" int 3 (Buffer.Rollout.size buffer);

  let steps, advantages, returns = Buffer.Rollout.get buffer in
  equal ~msg:"got 3 steps" int 3 (Array.length steps);
  equal ~msg:"got 3 advantages" int 3 (Array.length advantages);
  equal ~msg:"got 3 returns" int 3 (Array.length returns);
  equal ~msg:"size after get" int 0 (Buffer.Rollout.size buffer)

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
  equal ~msg:"advantages computed" int 3 (Array.length advantages);
  equal ~msg:"returns computed" int 3 (Array.length returns);

  (* All advantages should be positive since rewards are 1.0 and values are
     0.0 *)
  Array.iter
    (fun adv -> equal ~msg:"advantage > 0" bool true (adv > 0.0))
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
  equal ~msg:"advantage treats truncation as terminal" (float 1e-6) 0.5 advantages.(0);
  equal ~msg:"return respects truncation" (float 1e-6) 1.0 returns.(0)

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

  equal ~msg:"size before clear" int 1 (Buffer.Rollout.size buffer);
  Buffer.Rollout.clear buffer;
  equal ~msg:"size after clear" int 0 (Buffer.Rollout.size buffer)

let () =
  run "Buffer"
    [
      group "Replay"
        [
          test "create replay buffer" test_replay_create;
          test "add and sample" test_replay_add_and_sample;
          test "add_many" test_replay_add_many;
          test "circular buffer behavior" test_replay_circular;
          test "clear buffer" test_replay_clear;
        ];
      group "Rollout"
        [
          test "create rollout buffer" test_rollout_create;
          test "add and get" test_rollout_add_and_get;
          test "compute advantages" test_rollout_compute_advantages;
          test "compute advantages with truncation"
            test_rollout_compute_advantages_truncated;
          test "clear buffer" test_rollout_clear;
        ];
    ]
