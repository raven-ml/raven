(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_lengths = "Gae: all arrays must have the same length"

let err_returns_lengths =
  "Gae.returns: rewards, terminated, and truncated must have the same length"

let err_cfv_lengths =
  "Gae.compute_from_values: all arrays must have the same length"

let compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma ~lambda
    =
  let n = Array.length rewards in
  if
    n <> Array.length values
    || n <> Array.length terminated
    || n <> Array.length truncated
    || n <> Array.length next_values
  then invalid_arg err_lengths;
  let advantages = Array.make n 0.0 in
  let returns = Array.make n 0.0 in
  let last_gae = ref 0.0 in
  for t = n - 1 downto 0 do
    let next_val, continuation =
      if terminated.(t) then (0.0, 0.0)
      else if truncated.(t) then (next_values.(t), 0.0)
      else begin
        let v = if t = n - 1 then next_values.(t) else values.(t + 1) in
        (v, 1.0)
      end
    in
    let delta = rewards.(t) +. (gamma *. next_val) -. values.(t) in
    last_gae := delta +. (gamma *. lambda *. continuation *. !last_gae);
    advantages.(t) <- !last_gae;
    returns.(t) <- !last_gae +. values.(t)
  done;
  (advantages, returns)

let compute_from_values ~rewards ~values ~terminated ~truncated ~last_value
    ~gamma ~lambda =
  let n = Array.length rewards in
  if
    n <> Array.length values
    || n <> Array.length terminated
    || n <> Array.length truncated
  then invalid_arg err_cfv_lengths;
  let next_values =
    Array.init n (fun t -> if t = n - 1 then last_value else values.(t + 1))
  in
  compute ~rewards ~values ~terminated ~truncated ~next_values ~gamma ~lambda

let returns ~rewards ~terminated ~truncated ~gamma =
  let n = Array.length rewards in
  if n <> Array.length terminated || n <> Array.length truncated then
    invalid_arg err_returns_lengths;
  let ret = Array.make n 0.0 in
  let acc = ref 0.0 in
  for t = n - 1 downto 0 do
    let cont = if terminated.(t) || truncated.(t) then 0.0 else 1.0 in
    acc := rewards.(t) +. (gamma *. cont *. !acc);
    ret.(t) <- !acc
  done;
  ret

let normalize ?(eps = 1e-8) arr =
  let n = Array.length arr in
  if n = 0 then arr
  else begin
    let mean = ref 0.0 in
    let m2 = ref 0.0 in
    for i = 0 to n - 1 do
      let k = Float.of_int (i + 1) in
      let x = arr.(i) in
      let delta = x -. !mean in
      mean := !mean +. (delta /. k);
      let delta2 = x -. !mean in
      m2 := !m2 +. (delta *. delta2)
    done;
    let std = sqrt (!m2 /. Float.of_int n) +. eps in
    let mu = !mean in
    Array.init n (fun i -> (arr.(i) -. mu) /. std)
  end
