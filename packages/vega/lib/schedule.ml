(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = int -> float

(* Cosine annealing factor: 1 -> 0 as ratio goes 0 -> 1. *)
let cosine_decay_factor ratio = 0.5 *. (1. +. Stdlib.cos (Float.pi *. ratio))
let constant value _ = value

let linear ~init_value ~end_value ~steps step =
  if steps <= 0 then invalid_arg "Schedule.linear: steps must be positive";
  if step >= steps then end_value
  else
    let ratio = float_of_int step /. float_of_int steps in
    init_value +. ((end_value -. init_value) *. ratio)

let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () step =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay: decay_steps must be positive";
  if step >= decay_steps then alpha *. init_value
  else
    let ratio = float_of_int step /. float_of_int decay_steps in
    let cosine_val = cosine_decay_factor ratio in
    (((1. -. alpha) *. cosine_val) +. alpha) *. init_value

let exponential_decay ~init_value ~decay_rate ~decay_steps step =
  if decay_steps <= 0 then
    invalid_arg "Schedule.exponential_decay: decay_steps must be positive";
  let ratio = float_of_int step /. float_of_int decay_steps in
  init_value *. (decay_rate ** ratio)

let polynomial_decay ~init_value ~end_value ~decay_steps ?(power = 1.0) () step
    =
  if decay_steps <= 0 then
    invalid_arg "Schedule.polynomial_decay: decay_steps must be positive";
  if step >= decay_steps then end_value
  else
    let ratio = float_of_int step /. float_of_int decay_steps in
    end_value +. ((init_value -. end_value) *. ((1. -. ratio) ** power))

let warmup_cosine ~init_value ~peak_value ~warmup_steps step =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine: warmup_steps must be positive";
  if step >= warmup_steps then peak_value
  else
    let ratio = float_of_int step /. float_of_int warmup_steps in
    let cosine_val = 1. -. cosine_decay_factor ratio in
    init_value +. ((peak_value -. init_value) *. cosine_val)

let warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
    ?(end_value = 0.) () step =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: warmup_steps must be positive";
  if decay_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: decay_steps must be positive";
  if step <= warmup_steps then
    let ratio = float_of_int step /. float_of_int warmup_steps in
    init_value +. ((peak_value -. init_value) *. ratio)
  else
    let decay_step = step - warmup_steps in
    if decay_step >= decay_steps then end_value
    else
      let ratio = float_of_int decay_step /. float_of_int decay_steps in
      let cosine_val = cosine_decay_factor ratio in
      end_value +. ((peak_value -. end_value) *. cosine_val)

let cosine_decay_restarts ~init_value ~decay_steps ?(t_mul = 1.0) ?(m_mul = 1.0)
    ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay_restarts: decay_steps must be positive";
  fun step ->
    (* Fast path for uniform period (exact float comparison is intentional: 1.0
       is the unmodified default). *)
    if t_mul = 1.0 then
      let cycle = step / decay_steps in
      let pos = step - (cycle * decay_steps) in
      let amp = init_value *. (m_mul ** float_of_int cycle) in
      let ratio = float_of_int pos /. float_of_int decay_steps in
      let cosine_val = cosine_decay_factor ratio in
      (((1. -. alpha) *. cosine_val) +. alpha) *. amp
    else begin
      (* Geometric period: find which cycle [step] falls in. *)
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
      let cosine_val = cosine_decay_factor ratio in
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
        let cosine_val = cosine_decay_factor ratio in
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

(* Tensor schedules: the scalar formulas in tensor arithmetic over a scalar
   int32 step counter, so a learning rate can be derived inside a jitted step
   from the optimizer state's step leaf. Everything is [where]/[minimum]/[cos]
   arithmetic on the counter — no host reads — so the schedules trace. *)

let f32 = Nx.float32
let step_f32 step = Nx.cast f32 step

let clamp_ratio_t ctx param ~steps step =
  if steps <= 0 then
    invalid_arg (Printf.sprintf "Schedule.%s: %s must be positive" ctx param);
  Nx.div_s
    (Nx.minimum step (Nx.scalar f32 (float_of_int steps)))
    (float_of_int steps)

let constant_t value _step = Nx.scalar f32 value

let linear_t ~init_value ~end_value ~steps step =
  let ratio = clamp_ratio_t "linear_t" "steps" ~steps (step_f32 step) in
  Nx.add_s (Nx.mul_s ratio (end_value -. init_value)) init_value

let cosine_decay_t ~init_value ~decay_steps ?(alpha = 0.) step =
  let ratio =
    clamp_ratio_t "cosine_decay_t" "decay_steps" ~steps:decay_steps
      (step_f32 step)
  in
  let cosine_val =
    Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s ratio Float.pi)) 1.0) 0.5
  in
  Nx.mul_s (Nx.add_s (Nx.mul_s cosine_val (1.0 -. alpha)) alpha) init_value

let warmup_cosine_t ~init_value ~peak_value ~warmup_steps step =
  let ratio =
    clamp_ratio_t "warmup_cosine_t" "warmup_steps" ~steps:warmup_steps
      (step_f32 step)
  in
  let cosine_val =
    Nx.rsub_s 1.0
      (Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s ratio Float.pi)) 1.0) 0.5)
  in
  Nx.add_s (Nx.mul_s cosine_val (peak_value -. init_value)) init_value

let warmup_cosine_decay_t ~init_value ~peak_value ~warmup_steps ~decay_steps
    ?(end_value = 0.) step =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay_t: warmup_steps must be positive";
  let s = step_f32 step in
  let warm_ratio =
    Nx.div_s
      (Nx.minimum s (Nx.scalar f32 (float_of_int warmup_steps)))
      (float_of_int warmup_steps)
  in
  let linear_part =
    Nx.add_s (Nx.mul_s warm_ratio (peak_value -. init_value)) init_value
  in
  let decay_step = Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0 in
  let decay_ratio =
    clamp_ratio_t "warmup_cosine_decay_t" "decay_steps" ~steps:decay_steps
      decay_step
  in
  let cosine_val =
    Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s decay_ratio Float.pi)) 1.0) 0.5
  in
  let decay_part =
    Nx.add_s (Nx.mul_s cosine_val (peak_value -. end_value)) end_value
  in
  Nx.where (Nx.greater_s s (float_of_int warmup_steps)) decay_part linear_part

let one_cycle_t ~max_value ~total_steps ?(div_factor = 25.0)
    ?(final_div_factor = 10000.0) ?(pct_start = 0.3) step =
  if total_steps <= 0 then
    invalid_arg "Schedule.one_cycle_t: total_steps must be positive";
  let warmup_steps = int_of_float (pct_start *. float_of_int total_steps) in
  let init_value = max_value /. div_factor in
  let end_value = max_value /. final_div_factor in
  let s = step_f32 step in
  let decay_steps = total_steps - warmup_steps in
  if warmup_steps <= 0 then
    (* All decay: warmup is degenerate, the counter ramps straight down. *)
    let decay_ratio =
      clamp_ratio_t "one_cycle_t" "total_steps" ~steps:decay_steps s
    in
    let cosine_val =
      Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s decay_ratio Float.pi)) 1.0) 0.5
    in
    Nx.add_s (Nx.mul_s cosine_val (max_value -. end_value)) end_value
  else if decay_steps <= 0 then
    (* All warmup: the schedule never leaves its ramp. *)
    let warm_ratio =
      clamp_ratio_t "one_cycle_t" "total_steps" ~steps:warmup_steps s
    in
    Nx.add_s (Nx.mul_s warm_ratio (max_value -. init_value)) init_value
  else
    let decay_step =
      Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0
    in
    let decay_ratio =
      clamp_ratio_t "one_cycle_t" "total_steps" ~steps:decay_steps decay_step
    in
    let cosine_val =
      Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s decay_ratio Float.pi)) 1.0) 0.5
    in
    let decay_part =
      Nx.add_s (Nx.mul_s cosine_val (max_value -. end_value)) end_value
    in
    let warm_ratio =
      Nx.div_s
        (Nx.minimum s (Nx.scalar f32 (float_of_int warmup_steps)))
        (float_of_int warmup_steps)
    in
    let linear_part =
      Nx.add_s (Nx.mul_s warm_ratio (max_value -. init_value)) init_value
    in
    Nx.where (Nx.greater_s s (float_of_int warmup_steps)) decay_part linear_part

let piecewise_constant_t ~boundaries ~values step =
  let n_boundaries = List.length boundaries in
  let n_values = List.length values in
  if n_values <> n_boundaries + 1 then
    invalid_arg
      (Printf.sprintf
         "Schedule.piecewise_constant_t: expected %d values for %d boundaries, \
          got %d"
         (n_boundaries + 1) n_boundaries n_values);
  let rec check_increasing = function
    | [] | [ _ ] -> ()
    | a :: (b :: _ as rest) ->
        if b <= a then
          invalid_arg
            "Schedule.piecewise_constant_t: boundaries must be strictly \
             increasing";
        check_increasing rest
  in
  check_increasing boundaries;
  let boundaries = Array.of_list boundaries in
  let values = Array.of_list values in
  let s = step_f32 step in
  let acc = ref (Nx.scalar f32 values.(Array.length values - 1)) in
  for i = Array.length boundaries - 1 downto 0 do
    acc :=
      Nx.where
        (Nx.greater_equal_s s (float_of_int boundaries.(i) +. 1.0))
        !acc
        (Nx.scalar f32 values.(i))
  done;
  !acc
