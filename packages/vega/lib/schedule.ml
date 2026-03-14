(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = int -> float

let constant value _ = value

let linear ~init_value ~end_value ~steps step =
  if step >= steps then end_value
  else
    let ratio = float_of_int step /. float_of_int steps in
    init_value +. ((end_value -. init_value) *. ratio)

let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () step =
  if step >= decay_steps then alpha *. init_value
  else
    let ratio = float_of_int step /. float_of_int decay_steps in
    let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
    (((1. -. alpha) *. cosine_val) +. alpha) *. init_value

let exponential_decay ~init_value ~decay_rate ~decay_steps step =
  let ratio = float_of_int step /. float_of_int decay_steps in
  init_value *. (decay_rate ** ratio)

let polynomial_decay ~init_value ~end_value ~decay_steps ?(power = 1.0) () step
    =
  if step >= decay_steps then end_value
  else
    let ratio = float_of_int step /. float_of_int decay_steps in
    end_value +. ((init_value -. end_value) *. ((1. -. ratio) ** power))

let warmup_cosine ~init_value ~peak_value ~warmup_steps step =
  if step >= warmup_steps then peak_value
  else
    let ratio = float_of_int step /. float_of_int warmup_steps in
    let cosine_val = 0.5 *. (1. -. Stdlib.cos (Float.pi *. ratio)) in
    init_value +. ((peak_value -. init_value) *. cosine_val)

let warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
    ?(end_value = 0.) () step =
  if step <= warmup_steps then
    let ratio = float_of_int step /. float_of_int warmup_steps in
    init_value +. ((peak_value -. init_value) *. ratio)
  else
    let decay_step = step - warmup_steps in
    if decay_step >= decay_steps then end_value
    else
      let ratio = float_of_int decay_step /. float_of_int decay_steps in
      let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
      end_value +. ((peak_value -. end_value) *. cosine_val)

let cosine_decay_restarts ~init_value ~decay_steps ?(t_mul = 1.0) ?(m_mul = 1.0)
    ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay_restarts: decay_steps must be positive";
  fun step ->
    if t_mul = 1.0 then begin
      let cycle = step / decay_steps in
      let pos = step - (cycle * decay_steps) in
      let amp = init_value *. (m_mul ** float_of_int cycle) in
      let ratio = float_of_int pos /. float_of_int decay_steps in
      let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
      (((1. -. alpha) *. cosine_val) +. alpha) *. amp
    end
    else begin
      (* Geometric series: find which cycle step falls in *)
      let remaining = ref step in
      let cycle = ref 0 in
      let period = ref (float_of_int decay_steps) in
      while float_of_int !remaining >= !period do
        remaining := !remaining - int_of_float !period;
        period := !period *. t_mul;
        incr cycle
      done;
      let amp = init_value *. (m_mul ** float_of_int !cycle) in
      let ratio = float_of_int !remaining /. !period in
      let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
      (((1. -. alpha) *. cosine_val) +. alpha) *. amp
    end

let one_cycle ~max_value ~total_steps ?(div_factor = 25.0)
    ?(final_div_factor = 10000.0) ?(pct_start = 0.3) () =
  if total_steps <= 0 then
    invalid_arg "Schedule.one_cycle: total_steps must be positive";
  fun step ->
    let warmup_steps = int_of_float (pct_start *. float_of_int total_steps) in
    let init_value = max_value /. div_factor in
    let end_value = max_value /. final_div_factor in
    if step <= warmup_steps then
      let ratio = float_of_int step /. float_of_int warmup_steps in
      init_value +. ((max_value -. init_value) *. ratio)
    else
      let decay_steps = total_steps - warmup_steps in
      let decay_step = step - warmup_steps in
      if decay_step >= decay_steps then end_value
      else
        let ratio = float_of_int decay_step /. float_of_int decay_steps in
        let cosine_val = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio)) in
        end_value +. ((max_value -. end_value) *. cosine_val)

let piecewise_constant ~boundaries ~values =
  let n_boundaries = List.length boundaries in
  let n_values = List.length values in
  if n_values <> n_boundaries + 1 then
    invalid_arg
      (Printf.sprintf
         "Schedule.piecewise_constant: expected %d values for %d boundaries, \
          got %d"
         (n_boundaries + 1) n_boundaries n_values);
  let boundaries = Array.of_list boundaries in
  let values = Array.of_list values in
  for i = 1 to Array.length boundaries - 1 do
    if boundaries.(i) <= boundaries.(i - 1) then
      invalid_arg
        "Schedule.piecewise_constant: boundaries must be strictly increasing"
  done;
  fun step ->
    let rec find i =
      if i >= Array.length boundaries then values.(Array.length values - 1)
      else if step <= boundaries.(i) then values.(i)
      else find (i + 1)
    in
    find 0

let join segments =
  if segments = [] then invalid_arg "Schedule.join: segments must not be empty";
  List.iter
    (fun (n, _) ->
      if n <= 0 then
        invalid_arg "Schedule.join: segment lengths must be positive")
    segments;
  let segments = Array.of_list segments in
  fun step ->
    let remaining = ref step in
    let i = ref 0 in
    while !i < Array.length segments - 1 && !remaining > fst segments.(!i) do
      remaining := !remaining - fst segments.(!i);
      incr i
    done;
    let _, sched = segments.(!i) in
    sched !remaining
