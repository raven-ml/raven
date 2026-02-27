open Fehu
open Windtrap

let make_transition obs act rew next_obs term trunc =
  Buffer.
    {
      observation = obs;
      action = act;
      reward = rew;
      next_observation = next_obs;
      terminated = term;
      truncated = trunc;
    }

(* Creation *)

let test_create_empty () =
  let buf = Buffer.create ~capacity:10 in
  equal ~msg:"size = 0" int 0 (Buffer.size buf);
  is_false ~msg:"not full" (Buffer.is_full buf)

let test_capacity () =
  let buf = Buffer.create ~capacity:10 in
  equal ~msg:"capacity = 10" int 10 (Buffer.capacity buf)

let test_create_zero_capacity () =
  raises_invalid_arg "Buffer.create: capacity must be positive" (fun () ->
      Buffer.create ~capacity:0)

let test_create_negative_capacity () =
  raises_invalid_arg "Buffer.create: capacity must be positive" (fun () ->
      Buffer.create ~capacity:(-1))

(* Add/Size *)

let test_add_increments_size () =
  let buf = Buffer.create ~capacity:10 in
  Buffer.add buf (make_transition 1 0 1.0 2 false false);
  equal ~msg:"size = 1" int 1 (Buffer.size buf);
  Buffer.add buf (make_transition 2 1 2.0 3 false false);
  equal ~msg:"size = 2" int 2 (Buffer.size buf)

let test_size_capped_at_capacity () =
  let buf = Buffer.create ~capacity:3 in
  for i = 1 to 5 do
    Buffer.add buf (make_transition i 0 1.0 (i + 1) false false)
  done;
  equal ~msg:"size capped at 3" int 3 (Buffer.size buf)

let test_is_full () =
  let buf = Buffer.create ~capacity:2 in
  Buffer.add buf (make_transition 1 0 1.0 2 false false);
  is_false ~msg:"not yet full" (Buffer.is_full buf);
  Buffer.add buf (make_transition 2 1 2.0 3 false false);
  is_true ~msg:"full" (Buffer.is_full buf)

(* Sample *)

let test_sample_batch_size () =
  let buf = Buffer.create ~capacity:10 in
  for i = 1 to 5 do
    Buffer.add buf (make_transition i 0 1.0 (i + 1) false false)
  done;
  let batch = Buffer.sample buf ~batch_size:3 in
  equal ~msg:"batch length" int 3 (Array.length batch)

let test_sample_empty_raises () =
  let buf = Buffer.create ~capacity:10 in
  raises_invalid_arg "Buffer.sample: buffer is empty" (fun () ->
      Buffer.sample buf ~batch_size:1)

let test_sample_zero_batch_raises () =
  let buf = Buffer.create ~capacity:10 in
  Buffer.add buf (make_transition 1 0 1.0 2 false false);
  raises_invalid_arg "Buffer.sample: batch_size must be positive" (fun () ->
      Buffer.sample buf ~batch_size:0)

let test_sample_arrays_lengths () =
  let buf = Buffer.create ~capacity:10 in
  for i = 1 to 5 do
    Buffer.add buf (make_transition i 0 1.0 (i + 1) false false)
  done;
  let obs, acts, rews, next_obs, terms, truncs =
    Buffer.sample_arrays buf ~batch_size:3
  in
  equal ~msg:"obs length" int 3 (Array.length obs);
  equal ~msg:"acts length" int 3 (Array.length acts);
  equal ~msg:"rews length" int 3 (Array.length rews);
  equal ~msg:"next_obs length" int 3 (Array.length next_obs);
  equal ~msg:"terms length" int 3 (Array.length terms);
  equal ~msg:"truncs length" int 3 (Array.length truncs)

let test_sample_arrays_empty_raises () =
  let buf = Buffer.create ~capacity:10 in
  raises_invalid_arg "Buffer.sample: buffer is empty" (fun () ->
      Buffer.sample_arrays buf ~batch_size:1)

(* Clear *)

let test_clear_resets () =
  let buf = Buffer.create ~capacity:10 in
  Buffer.add buf (make_transition 1 0 1.0 2 false false);
  Buffer.add buf (make_transition 2 1 2.0 3 false false);
  Buffer.clear buf;
  equal ~msg:"size = 0 after clear" int 0 (Buffer.size buf)

let test_add_after_clear () =
  let buf = Buffer.create ~capacity:10 in
  Buffer.add buf (make_transition 1 0 1.0 2 false false);
  Buffer.clear buf;
  Buffer.add buf (make_transition 3 1 3.0 4 false false);
  equal ~msg:"size = 1 after re-add" int 1 (Buffer.size buf)

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  run "Fehu.Buffer"
    [
      group "creation"
        [
          test "empty" test_create_empty;
          test "capacity" test_capacity;
          test "zero capacity raises" test_create_zero_capacity;
          test "negative capacity raises" test_create_negative_capacity;
        ];
      group "add/size"
        [
          test "add increments size" test_add_increments_size;
          test "size capped at capacity" test_size_capped_at_capacity;
          test "is_full" test_is_full;
        ];
      group "sample"
        [
          test "batch size" test_sample_batch_size;
          test "empty raises" test_sample_empty_raises;
          test "zero batch raises" test_sample_zero_batch_raises;
          test "sample_arrays lengths" test_sample_arrays_lengths;
          test "sample_arrays empty raises" test_sample_arrays_empty_raises;
        ];
      group "clear"
        [
          test "resets size" test_clear_resets;
          test "add after clear" test_add_after_clear;
        ];
    ]
