(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Schedules are tensor arithmetic over a scalar int32 step counter —
   [where]/[minimum]/[cos]/[pow] on the counter, no host reads — so the same
   schedule computes eagerly and traces inside a compiled step, where it
   derives the learning rate from the optimizer state's own step leaf. *)

type t = Nx.int32_t -> Nx.float32_t

let f32 = Nx.float32
let to_f32 step = Nx.cast f32 step
let eval sched step = Nx.item [] (sched (Nx.scalar Nx.int32 (Int32.of_int step)))

(* Cosine annealing factor: 1 -> 0 as ratio goes 0 -> 1. *)
let cosine_factor ratio =
  Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s ratio Float.pi)) 1.0) 0.5

(* step / steps, clamped to 1. *)
let clamp_ratio ~steps s =
  Nx.div_s
    (Nx.minimum s (Nx.scalar f32 (float_of_int steps)))
    (float_of_int steps)

(* [base ** p] for [base >= 0] and a constant exponent. Integer and
   half-integer exponents stay [pow], which the compiler decomposes into
   multiplications and square roots; other exponents go through exp/log, whose
   limit at a zero base is the right value for positive [p]. *)
let pow_const base p =
  if Float.is_integer (2.0 *. p) then Nx.pow_s base p
  else Nx.exp (Nx.mul_s (Nx.log base) p)

let constant value _step = Nx.scalar f32 value

let linear ~init_value ~end_value ~steps =
  if steps <= 0 then invalid_arg "Schedule.linear: steps must be positive";
  fun step ->
    let ratio = clamp_ratio ~steps (to_f32 step) in
    Nx.add_s (Nx.mul_s ratio (end_value -. init_value)) init_value

let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay: decay_steps must be positive";
  fun step ->
    let ratio = clamp_ratio ~steps:decay_steps (to_f32 step) in
    let cosine_val = cosine_factor ratio in
    Nx.mul_s (Nx.add_s (Nx.mul_s cosine_val (1.0 -. alpha)) alpha) init_value

let exponential_decay ~init_value ~decay_rate ~decay_steps =
  if decay_steps <= 0 then
    invalid_arg "Schedule.exponential_decay: decay_steps must be positive";
  if decay_rate <= 0.0 then
    invalid_arg "Schedule.exponential_decay: decay_rate must be positive";
  fun step ->
    let ratio = Nx.div_s (to_f32 step) (float_of_int decay_steps) in
    Nx.mul_s (Nx.rpow_s decay_rate ratio) init_value

let polynomial_decay ~init_value ~end_value ~decay_steps ?(power = 1.0) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.polynomial_decay: decay_steps must be positive";
  fun step ->
    let ratio = clamp_ratio ~steps:decay_steps (to_f32 step) in
    let poly = pow_const (Nx.rsub_s 1.0 ratio) power in
    Nx.add_s (Nx.mul_s poly (init_value -. end_value)) end_value

let warmup_cosine ~init_value ~peak_value ~warmup_steps =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine: warmup_steps must be positive";
  fun step ->
    let ratio = clamp_ratio ~steps:warmup_steps (to_f32 step) in
    let cosine_val = Nx.rsub_s 1.0 (cosine_factor ratio) in
    Nx.add_s (Nx.mul_s cosine_val (peak_value -. init_value)) init_value

let warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
    ?(end_value = 0.) () =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: warmup_steps must be positive";
  if decay_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: decay_steps must be positive";
  fun step ->
    let s = to_f32 step in
    let warm_ratio = clamp_ratio ~steps:warmup_steps s in
    let linear_part =
      Nx.add_s (Nx.mul_s warm_ratio (peak_value -. init_value)) init_value
    in
    let decay_step =
      Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0
    in
    let decay_ratio = clamp_ratio ~steps:decay_steps decay_step in
    let decay_part =
      Nx.add_s
        (Nx.mul_s (cosine_factor decay_ratio) (peak_value -. end_value))
        end_value
    in
    Nx.where (Nx.greater_s s (float_of_int warmup_steps)) decay_part linear_part

let cosine_decay_restarts ~init_value ~decay_steps ?(t_mul = 1.0) ?(m_mul = 1.0)
    ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay_restarts: decay_steps must be positive";
  if t_mul < 1.0 then
    invalid_arg "Schedule.cosine_decay_restarts: t_mul must be at least 1";
  if m_mul <= 0.0 then
    invalid_arg "Schedule.cosine_decay_restarts: m_mul must be positive";
  let first = float_of_int decay_steps in
  let anneal cycle ratio =
    let amp = Nx.mul_s (Nx.rpow_s m_mul cycle) init_value in
    Nx.mul (Nx.add_s (Nx.mul_s (cosine_factor ratio) (1.0 -. alpha)) alpha) amp
  in
  (* Exact float comparison is intentional: 1.0 is the unmodified default. *)
  if t_mul = 1.0 then fun step ->
    let s = to_f32 step in
    let cycle = Nx.floor (Nx.div_s s first) in
    let pos = Nx.sub s (Nx.mul_s cycle first) in
    anneal cycle (Nx.div_s pos first)
  else fun step ->
    (* Geometric periods: cycle k spans [start k, start (k+1)) where
       [start k = first * (t_mul^k - 1) / (t_mul - 1)]. Inverting gives
       [k = floor (log q / log t_mul)] with [q = s*(t_mul-1)/first + 1]; the
       two [where]s absorb any float slop at the cycle boundaries. *)
    let s = to_f32 step in
    let start k =
      Nx.mul_s
        (Nx.div_s (Nx.sub_s (Nx.rpow_s t_mul k) 1.0) (t_mul -. 1.0))
        first
    in
    let q = Nx.add_s (Nx.mul_s s ((t_mul -. 1.0) /. first)) 1.0 in
    let k = Nx.floor (Nx.div_s (Nx.log q) (Stdlib.log t_mul)) in
    let k = Nx.where (Nx.less s (start k)) (Nx.sub_s k 1.0) k in
    let k =
      Nx.where (Nx.greater_equal s (start (Nx.add_s k 1.0))) (Nx.add_s k 1.0) k
    in
    let k = Nx.maximum_s k 0.0 in
    let period = Nx.mul_s (Nx.rpow_s t_mul k) first in
    let pos = Nx.sub s (start k) in
    anneal k (Nx.div pos period)

let one_cycle ~max_value ~total_steps ?(div_factor = 25.0)
    ?(final_div_factor = 10000.0) ?(pct_start = 0.3) () =
  if total_steps <= 0 then
    invalid_arg "Schedule.one_cycle: total_steps must be positive";
  let warmup_steps = int_of_float (pct_start *. float_of_int total_steps) in
  let decay_steps = total_steps - warmup_steps in
  let init_value = max_value /. div_factor in
  let end_value = max_value /. final_div_factor in
  if warmup_steps <= 0 then fun step ->
    (* All decay: warmup is degenerate, the counter ramps straight down. *)
    let ratio = clamp_ratio ~steps:decay_steps (to_f32 step) in
    Nx.add_s (Nx.mul_s (cosine_factor ratio) (max_value -. end_value)) end_value
  else if decay_steps <= 0 then fun step ->
    (* All warmup: the schedule never leaves its ramp. *)
    let ratio = clamp_ratio ~steps:warmup_steps (to_f32 step) in
    Nx.add_s (Nx.mul_s ratio (max_value -. init_value)) init_value
  else fun step ->
    let s = to_f32 step in
    let warm_ratio = clamp_ratio ~steps:warmup_steps s in
    let linear_part =
      Nx.add_s (Nx.mul_s warm_ratio (max_value -. init_value)) init_value
    in
    let decay_step =
      Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0
    in
    let decay_ratio = clamp_ratio ~steps:decay_steps decay_step in
    let decay_part =
      Nx.add_s
        (Nx.mul_s (cosine_factor decay_ratio) (max_value -. end_value))
        end_value
    in
    Nx.where (Nx.greater_s s (float_of_int warmup_steps)) decay_part linear_part

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
    let s = to_f32 step in
    let acc = ref (Nx.scalar f32 values.(Array.length values - 1)) in
    for i = Array.length boundaries - 1 downto 0 do
      acc :=
        Nx.where
          (Nx.greater_s s (float_of_int boundaries.(i)))
          !acc
          (Nx.scalar f32 values.(i))
    done;
    !acc

let join segments =
  if segments = [] then invalid_arg "Schedule.join: segments must not be empty";
  List.iter
    (fun (n, _) ->
      if n <= 0 then
        invalid_arg "Schedule.join: segment lengths must be positive")
    segments;
  let segments = Array.of_list segments in
  let n_seg = Array.length segments in
  let starts = Array.make n_seg 0 in
  for i = 1 to n_seg - 1 do
    starts.(i) <- starts.(i - 1) + fst segments.(i - 1)
  done;
  fun step ->
    (* Selected last to first: the earliest segment whose span covers the
       relative counter wins. Inactive segments are still computed (at
       out-of-range counters) and discarded by [where]. *)
    let at i = snd segments.(i) (Nx.sub_s step (Int32.of_int starts.(i))) in
    let acc = ref (at (n_seg - 1)) in
    for i = n_seg - 2 downto 0 do
      let rel = Nx.sub_s step (Int32.of_int starts.(i)) in
      acc :=
        Nx.where
          (Nx.less_equal_s rel (Int32.of_int (fst segments.(i))))
          (at i) !acc
    done;
    !acc
