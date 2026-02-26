(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_capacity = "Buffer.create: capacity must be positive"
let err_empty = "Buffer.sample: buffer is empty"
let err_batch_size = "Buffer.sample: batch_size must be positive"

type ('obs, 'act) transition = {
  observation : 'obs;
  action : 'act;
  reward : float;
  next_observation : 'obs;
  terminated : bool;
  truncated : bool;
}

type ('obs, 'act) t = {
  capacity : int;
  mutable size : int;
  mutable pos : int;
  mutable observations : 'obs array;
  mutable actions : 'act array;
  rewards : float array;
  mutable next_observations : 'obs array;
  terminateds : bool array;
  truncateds : bool array;
}

(* Constructor *)

let create ~capacity =
  if capacity <= 0 then invalid_arg err_capacity;
  {
    capacity;
    size = 0;
    pos = 0;
    observations = [||];
    actions = [||];
    rewards = Array.make capacity 0.0;
    next_observations = [||];
    terminateds = Array.make capacity false;
    truncateds = Array.make capacity false;
  }

(* Mutating *)

let ensure_init buf (tr : _ transition) =
  if Array.length buf.observations = 0 then begin
    buf.observations <- Array.make buf.capacity tr.observation;
    buf.actions <- Array.make buf.capacity tr.action;
    buf.next_observations <- Array.make buf.capacity tr.next_observation
  end

let add buf tr =
  ensure_init buf tr;
  buf.observations.(buf.pos) <- tr.observation;
  buf.actions.(buf.pos) <- tr.action;
  buf.rewards.(buf.pos) <- tr.reward;
  buf.next_observations.(buf.pos) <- tr.next_observation;
  buf.terminateds.(buf.pos) <- tr.terminated;
  buf.truncateds.(buf.pos) <- tr.truncated;
  buf.pos <- (buf.pos + 1) mod buf.capacity;
  if buf.size < buf.capacity then buf.size <- buf.size + 1

let clear buf =
  buf.size <- 0;
  buf.pos <- 0

(* Sampling *)

let sample_indices buf ~batch_size =
  if buf.size = 0 then invalid_arg err_empty;
  if batch_size <= 0 then invalid_arg err_batch_size;
  let n = min batch_size buf.size in
  let raw = Rune.randint Rune.int32 ~high:buf.size [| n |] 0 in
  let idx : Int32.t array = Rune.to_array raw in
  (idx, n)

let sample buf ~batch_size =
  let idx, n = sample_indices buf ~batch_size in
  Array.init n (fun i ->
      let j = Int32.to_int idx.(i) in
      {
        observation = buf.observations.(j);
        action = buf.actions.(j);
        reward = buf.rewards.(j);
        next_observation = buf.next_observations.(j);
        terminated = buf.terminateds.(j);
        truncated = buf.truncateds.(j);
      })

let sample_arrays buf ~batch_size =
  let idx, n = sample_indices buf ~batch_size in
  let get arr i = arr.(Int32.to_int idx.(i)) in
  let observations = Array.init n (get buf.observations) in
  let actions = Array.init n (get buf.actions) in
  let rewards = Array.init n (get buf.rewards) in
  let next_observations = Array.init n (get buf.next_observations) in
  let terminated = Array.init n (get buf.terminateds) in
  let truncated = Array.init n (get buf.truncateds) in
  (observations, actions, rewards, next_observations, terminated, truncated)

(* Queries *)

let size buf = buf.size
let is_full buf = buf.size = buf.capacity
let capacity buf = buf.capacity
