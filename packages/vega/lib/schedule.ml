(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Schedules are tensor arithmetic over a scalar int32 step counter —
   [where]/[minimum]/[cos]/[pow] on the counter, no host reads — so the same
   schedule computes eagerly and traces inside a compiled step, where it
   derives the learning rate from the optimizer state's own step leaf.

   A constructor's numbers are data. Each is computed on the host in float64
   at construction and rounded once into a 0-d float32 host array ([num]),
   which the schedule captures and places beside the counter when applied
   ([beside]). [f x n] over such an [n] is [f_s x v] over [v]'s float32 bits
   (nx.mli's scalar forms), so the values are those of the numbers written as
   literals, while a compiled step keeps one program for every set of numbers.
   The formulas' own constants (pi, 1, 0.5, 0) stay literals. *)

type 'd t = 'd Nx.int32_t -> 'd Nx.float32_t

let f32 = Nx.float32
let to_f32 step = Nx.cast f32 step
let eval sched step =
  Nx.item [] (sched (Nx.scalar Nx.int32 (Int32.of_int step)))

let num v = Nx.scalar f32 v
let count n = num (float_of_int n)
let beside step n = Nx.place (Nx.beside step) n

(* Cosine annealing factor: 1 -> 0 as ratio goes 0 -> 1. *)
let cosine_factor ratio =
  Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s ratio Float.pi)) 1.0) 0.5

(* s / n, clamped to 1. *)
let clamp_ratio n s = Nx.div (Nx.minimum s n) n

let linear ~init_value ~end_value ~steps =
  if steps <= 0 then invalid_arg "Schedule.linear: steps must be positive";
  let n = count steps and span = num (end_value -. init_value) in
  let init = num init_value in
  fun step ->
    let ratio = clamp_ratio (beside step n) (to_f32 step) in
    Nx.add (Nx.mul ratio (beside step span)) (beside step init)

let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay: decay_steps must be positive";
  let n = count decay_steps and keep = num (1.0 -. alpha) in
  let alpha = num alpha and init = num init_value in
  fun step ->
    let ratio = clamp_ratio (beside step n) (to_f32 step) in
    let cosine_val = cosine_factor ratio in
    Nx.mul
      (Nx.add (Nx.mul cosine_val (beside step keep)) (beside step alpha))
      (beside step init)

let exponential_decay ~init_value ~decay_rate ~decay_steps =
  if decay_steps <= 0 then
    invalid_arg "Schedule.exponential_decay: decay_steps must be positive";
  if decay_rate <= 0.0 then
    invalid_arg "Schedule.exponential_decay: decay_rate must be positive";
  let n = count decay_steps and rate = num decay_rate in
  let init = num init_value in
  fun step ->
    let ratio = Nx.div (to_f32 step) (beside step n) in
    Nx.mul (Nx.pow (beside step rate) ratio) (beside step init)

let polynomial_decay ~init_value ~end_value ~decay_steps ?(power = 1.0) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.polynomial_decay: decay_steps must be positive";
  let n = count decay_steps and power = num power in
  let span = num (init_value -. end_value) and fin = num end_value in
  fun step ->
    (* The fraction left, [(steps - s) / steps], is exactly zero from the end
       of the decay on; [1 - s / steps] can round below it. *)
    let n = beside step n in
    let s = Nx.minimum (to_f32 step) n in
    let remaining = Nx.div (Nx.sub n s) n in
    let poly = Nx.pow remaining (beside step power) in
    Nx.add (Nx.mul poly (beside step span)) (beside step fin)

let warmup_cosine ~init_value ~peak_value ~warmup_steps =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine: warmup_steps must be positive";
  let n = count warmup_steps and span = num (peak_value -. init_value) in
  let init = num init_value in
  fun step ->
    let ratio = clamp_ratio (beside step n) (to_f32 step) in
    let cosine_val = Nx.rsub_s 1.0 (cosine_factor ratio) in
    Nx.add (Nx.mul cosine_val (beside step span)) (beside step init)

let warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
    ?(end_value = 0.) () =
  if warmup_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: warmup_steps must be positive";
  if decay_steps <= 0 then
    invalid_arg "Schedule.warmup_cosine_decay: decay_steps must be positive";
  let w = count warmup_steps and d = count decay_steps in
  let rise = num (peak_value -. init_value) and init = num init_value in
  let fall = num (peak_value -. end_value) and fin = num end_value in
  fun step ->
    let s = to_f32 step and w = beside step w in
    let warm_ratio = clamp_ratio w s in
    let linear_part =
      Nx.add (Nx.mul warm_ratio (beside step rise)) (beside step init)
    in
    let decay_step = Nx.maximum_s (Nx.sub s w) 0.0 in
    let decay_ratio = clamp_ratio (beside step d) decay_step in
    let decay_part =
      Nx.add
        (Nx.mul (cosine_factor decay_ratio) (beside step fall))
        (beside step fin)
    in
    Nx.where (Nx.greater s w) decay_part linear_part

let cosine_decay_restarts ~init_value ~decay_steps ?(t_mul = 1.0) ?(m_mul = 1.0)
    ?(alpha = 0.) () =
  if decay_steps <= 0 then
    invalid_arg "Schedule.cosine_decay_restarts: decay_steps must be positive";
  if t_mul < 1.0 then
    invalid_arg "Schedule.cosine_decay_restarts: t_mul must be at least 1";
  if m_mul <= 0.0 then
    invalid_arg "Schedule.cosine_decay_restarts: m_mul must be positive";
  let first = float_of_int decay_steps in
  let n = num first and m = num m_mul and init = num init_value in
  let keep = num (1.0 -. alpha) and alpha = num alpha in
  let anneal step cycle ratio =
    let amp = Nx.mul (Nx.pow (beside step m) cycle) (beside step init) in
    let keep = beside step keep and alpha = beside step alpha in
    Nx.mul (Nx.add (Nx.mul (cosine_factor ratio) keep) alpha) amp
  in
  (* Exact float comparison is intentional: 1.0 is the unmodified default. *)
  if t_mul = 1.0 then fun step ->
    let s = to_f32 step and n = beside step n in
    let cycle = Nx.floor (Nx.div s n) in
    let pos = Nx.sub s (Nx.mul cycle n) in
    anneal step cycle (Nx.div pos n)
  else
    let t = num t_mul and less = num (t_mul -. 1.0) in
    let slope = num ((t_mul -. 1.0) /. first) in
    let log_t = num (Stdlib.log t_mul) in
    fun step ->
      (* Geometric periods: cycle k spans [start k, start (k+1)) where
         [start k = first * (t_mul^k - 1) / (t_mul - 1)]. Inverting gives
         [k = floor (log q / log t_mul)] with [q = s*(t_mul-1)/first + 1]; the
         two [where]s absorb any float slop at the cycle boundaries. *)
      let s = to_f32 step and n = beside step n and t = beside step t in
      let less = beside step less in
      let start k = Nx.mul (Nx.div (Nx.sub_s (Nx.pow t k) 1.0) less) n in
      let q = Nx.add_s (Nx.mul s (beside step slope)) 1.0 in
      let k = Nx.floor (Nx.div (Nx.log q) (beside step log_t)) in
      let k = Nx.where (Nx.less s (start k)) (Nx.sub_s k 1.0) k in
      let k =
        Nx.where
          (Nx.greater_equal s (start (Nx.add_s k 1.0)))
          (Nx.add_s k 1.0) k
      in
      let k = Nx.maximum_s k 0.0 in
      let period = Nx.mul (Nx.pow t k) n in
      let pos = Nx.sub s (start k) in
      anneal step k (Nx.div pos period)

let one_cycle ~max_value ~total_steps ?(div_factor = 25.0)
    ?(final_div_factor = 10000.0) ?(pct_start = 0.3) () =
  if total_steps <= 0 then
    invalid_arg "Schedule.one_cycle: total_steps must be positive";
  let warmup_steps = int_of_float (pct_start *. float_of_int total_steps) in
  let decay_steps = total_steps - warmup_steps in
  let init_value = max_value /. div_factor in
  let end_value = max_value /. final_div_factor in
  if warmup_steps <= 0 then
    (* All decay: warmup is degenerate, the counter ramps straight down. *)
    let n = count decay_steps and fall = num (max_value -. end_value) in
    let fin = num end_value in
    fun step ->
      let ratio = clamp_ratio (beside step n) (to_f32 step) in
      Nx.add (Nx.mul (cosine_factor ratio) (beside step fall)) (beside step fin)
  else if decay_steps <= 0 then
    (* All warmup: the schedule never leaves its ramp. *)
    linear ~init_value ~end_value:max_value ~steps:warmup_steps
  else
    warmup_cosine_decay ~init_value ~peak_value:max_value ~warmup_steps
      ~decay_steps ~end_value ()

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
  for i = 1 to Array.length boundaries - 1 do
    if boundaries.(i) <= boundaries.(i - 1) then
      invalid_arg
        "Schedule.piecewise_constant: boundaries must be strictly increasing"
  done;
  let boundaries = Array.map count boundaries in
  let values = Array.of_list (List.map num values) in
  fun step ->
    let s = to_f32 step in
    let acc = ref (beside step values.(Array.length values - 1)) in
    for i = Array.length boundaries - 1 downto 0 do
      acc :=
        Nx.where
          (Nx.greater s (beside step boundaries.(i)))
          !acc
          (beside step values.(i))
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
  let int n = Nx.scalar Nx.int32 (Int32.of_int n) in
  let starts = Array.make n_seg 0 in
  for i = 1 to n_seg - 1 do
    starts.(i) <- starts.(i - 1) + fst segments.(i - 1)
  done;
  let starts = Array.map int starts in
  let lengths = Array.map (fun (n, _) -> int n) segments in
  fun step ->
    (* Selected last to first: the earliest segment whose span covers the
       relative counter wins. Inactive segments are still computed (at
       out-of-range counters) and discarded by [where]. *)
    let rel i = Nx.sub step (beside step starts.(i)) in
    let acc = ref (snd segments.(n_seg - 1) (rel (n_seg - 1))) in
    for i = n_seg - 2 downto 0 do
      let rel = rel i in
      acc :=
        Nx.where
          (Nx.less_equal rel (beside step lengths.(i)))
          (snd segments.(i) rel) !acc
    done;
    !acc
