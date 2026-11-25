(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each constructor is checked against the formula its .mli states, computed at
   float64 and read through [S.eval] at a drawn counter. *)

open Windtrap
module S = Vega.Schedule

(* Float32's unit roundoff. A schedule computes at float32: an operation's
   result is within [u] of its magnitude, an ulp of a value is at most [2u] of
   it, Nx.cos is within 2 ulps and Nx.pow within 4. *)
let u = 0x1p-24

(* [within k m] admits [k] unit roundoffs of the magnitude [m]. *)
let within k m = float (k *. u *. m)

(* Numbers *)

(* Multiples of 1/64 up to 4, their sums and differences are exact in float32,
   so a constructor rounds none of the numbers drawn here. *)
let sixty_fourths lo hi =
  Gen.of_list ~pp:Format.pp_print_float
    (List.init (hi - lo + 1) (fun k -> Float.of_int (lo + k) /. 64.))

let value = sixty_fourths 1 256
let value_or_zero = sixty_fourths 0 256
let fraction = sixty_fourths 0 64
let step_count = Gen.int_range 1 1000

(* A counter: one of [specials] (0, the last step, the boundaries) half the
   time, else one from 0 to [hi]. *)
let counter specials hi =
  Gen.frequency
    [
      (1, Gen.of_list ~pp:Format.pp_print_int specials); (1, Gen.int_range 0 hi);
    ]

(* The formulas *)

(* [ratio c n] is [min step n / n] at the counter [c]. *)
let ratio c n = Float.of_int (min c n) /. Float.of_int n

(* [cosine_factor x] is [(1 + cos (pi * x)) / 2]. At float32 it is within [6u]:
   [pi * x] is off by under [8u] (float32's pi by [1.5u], the rounding of [x]
   and of the product by [pi u] each), cos adds [2.5u] and the sum [u], and the
   halving halves them. *)
let cosine_factor x = (1. +. Float.cos (Float.pi *. x)) /. 2.

(* {!S.linear}: "init_value + (end_value - init_value) * min step steps /
   steps". At float32 it is within [3u] of the larger number for numbers of one
   sign: the ratio's rounding, the product's and the sum's. *)
let linear init_value end_value steps c =
  init_value +. ((end_value -. init_value) *. ratio c steps)

(* Basic *)

let linear_formula =
  prop ~count:300 "linear is its formula"
    Gen.(
      let* n = step_count in
      let+ v =
        triple value value_or_zero (counter [ 0; 1; n; n + 1 ] (3 * n))
      in
      (n, v))
    (fun (steps, (init_value, end_value, c)) ->
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "the last step" (c = steps);
      cover "past the end" (c > steps);
      (* [3u], and [u] for the second-order terms. *)
      equal
        (within 4. (Float.max init_value end_value))
        (linear init_value end_value steps c)
        (S.eval (S.linear ~init_value ~end_value ~steps) c))

let linear_refusals =
  cases "linear raises Invalid_argument if steps is not positive"
    ~id:string_of_int [ 0; -1 ] (fun steps ->
      raises_match Exn.invalid_arg (fun () ->
          S.linear ~init_value:1. ~end_value:0. ~steps))

(* Decay *)

let cosine_decay_formula =
  prop ~count:300 "cosine_decay is its formula"
    Gen.(
      let* n = step_count in
      let+ v =
        triple value (option fraction) (counter [ 0; 1; n; n + 1 ] (3 * n))
      in
      (n, v))
    (fun (decay_steps, (init_value, alpha, c)) ->
      let a = Option.value alpha ~default:0. in
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "the last step" (c = decay_steps);
      cover "past the end" (c > decay_steps);
      cover "alpha > 0" (a > 0.);
      cover "the default alpha" (alpha = None);
      (* "init_value * ((1 - alpha) * (1 + cos (pi * min step decay_steps /
         decay_steps)) / 2 + alpha)" *)
      let expected =
        init_value *. (((1. -. a) *. cosine_factor (ratio c decay_steps)) +. a)
      in
      (* The cosine factor's [6u], a product, a sum and a product: [9u] of
         init_value, and [u] for the second-order terms. *)
      equal (within 10. init_value) expected
        (S.eval (S.cosine_decay ~init_value ~decay_steps ?alpha ()) c))

let cosine_decay_refusals =
  cases "cosine_decay raises Invalid_argument if decay_steps is not positive"
    ~id:string_of_int [ 0; -1 ] (fun decay_steps ->
      raises_match Exn.invalid_arg (fun () ->
          S.cosine_decay ~init_value:1. ~decay_steps ()))

(* decay_rate in [7/64, 2]: no value underflows at three decay periods. *)
let exponential_decay_formula =
  prop ~count:300 "exponential_decay is its formula"
    Gen.(
      let* n = step_count in
      let+ v =
        triple value (sixty_fourths 7 128) (counter [ 0; 1; n; n + 1 ] (3 * n))
      in
      (n, v))
    (fun (decay_steps, (init_value, decay_rate, c)) ->
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "decay_steps" (c = decay_steps);
      cover "past decay_steps" (c > decay_steps);
      (* "init_value * decay_rate^(step / decay_steps), the exponent a real
         quotient" *)
      let r = Float.of_int c /. Float.of_int decay_steps in
      let expected = init_value *. (decay_rate ** r) in
      (* The quotient's rounding [u] becomes [|r ln decay_rate| u] through pow,
         which adds 4.5 ulps ([9u]); the product adds [u], and [u] more for the
         second-order terms. *)
      let rel = (Float.abs (r *. Float.log decay_rate) +. 11.) *. u in
      equal (float_rel ~rel ~abs:0.) expected
        (S.eval (S.exponential_decay ~init_value ~decay_rate ~decay_steps) c))

let exponential_decay_refusals =
  cases
    "exponential_decay raises Invalid_argument if decay_steps or decay_rate is \
     not positive"
    ~id:fst
    [
      ("decay_steps 0", (0, 0.5));
      ("decay_steps -1", (-1, 0.5));
      ("decay_rate 0", (10, 0.));
      ("decay_rate -0.5", (10, -0.5));
      ("decay_rate nan", (10, Float.nan));
    ]
    (fun (_, (decay_steps, decay_rate)) ->
      raises_match Exn.invalid_arg (fun () ->
          S.exponential_decay ~init_value:1. ~decay_rate ~decay_steps))

(* power in [1/4, 3]. *)
let polynomial_decay_formula =
  prop ~count:300 "polynomial_decay is its formula"
    Gen.(
      let* n = step_count in
      let+ v =
        quad value value_or_zero
          (option (sixty_fourths 16 192))
          (counter [ 0; 1; n; n + 1 ] (3 * n))
      in
      (n, v))
    (fun (decay_steps, (init_value, end_value, power, c)) ->
      let p = Option.value power ~default:1. in
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "the last step" (c = decay_steps);
      cover "past the end" (c > decay_steps);
      cover "a power below 1" (p < 1.);
      cover "the default power" (power = None);
      (* "end_value + (init_value - end_value) * (1 - step/decay_steps)^power",
         clamped to end_value after decay_steps. *)
      let expected =
        end_value
        +. ((init_value -. end_value) *. ((1. -. ratio c decay_steps) ** p))
      in
      (* The fraction left rounds once, which pow scales by [power] and adds
         [9u] to; the product and the sum add [u] each: [(power + 11) u] of the
         larger number, and [u] for the second-order terms. *)
      equal
        (within (p +. 12.) (Float.max init_value end_value))
        expected
        (S.eval
           (S.polynomial_decay ~init_value ~end_value ~decay_steps ?power ())
           c))

let polynomial_decay_refusals =
  cases
    "polynomial_decay raises Invalid_argument if decay_steps is not positive"
    ~id:string_of_int [ 0; -1 ] (fun decay_steps ->
      raises_match Exn.invalid_arg (fun () ->
          S.polynomial_decay ~init_value:1. ~end_value:0. ~decay_steps ()))

(* Warmup *)

let warmup_cosine_formula =
  prop ~count:300 "warmup_cosine is its formula"
    Gen.(
      let* n = step_count in
      let+ v = triple value value (counter [ 0; 1; n; n + 1 ] (3 * n)) in
      (n, v))
    (fun (warmup_steps, (init_value, peak_value, c)) ->
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "the last step" (c = warmup_steps);
      cover "past the end" (c > warmup_steps);
      (* "init_value + (peak_value - init_value) * (1 - cos (pi * min step
         warmup_steps / warmup_steps)) / 2" *)
      let expected =
        init_value
        +. (peak_value -. init_value)
           *. (1. -. cosine_factor (ratio c warmup_steps))
      in
      (* The cosine factor's [6u], a difference, a product and a sum: [9u] of
         the larger number, and [u] for the second-order terms. *)
      equal
        (within 10. (Float.max init_value peak_value))
        expected
        (S.eval (S.warmup_cosine ~init_value ~peak_value ~warmup_steps) c))

let warmup_cosine_refusals =
  cases "warmup_cosine raises Invalid_argument if warmup_steps is not positive"
    ~id:string_of_int [ 0; -1 ] (fun warmup_steps ->
      raises_match Exn.invalid_arg (fun () ->
          S.warmup_cosine ~init_value:0. ~peak_value:1. ~warmup_steps))

let warmup_cosine_decay_formula =
  prop ~count:300 "warmup_cosine_decay is its formula"
    Gen.(
      let* w, d = pair step_count step_count in
      let specials = [ 0; 1; w; w + 1; w + d; w + d + 1 ] in
      let+ v =
        quad value value (option value_or_zero) (counter specials (3 * (w + d)))
      in
      ((w, d), v))
    (fun ((w, d), (init_value, peak_value, end_value, c)) ->
      let e = Option.value end_value ~default:0. in
      cover "step 0" (c = 0);
      cover "step 1" (c = 1);
      cover "the end of warmup" (c = w);
      cover "the first step of decay" (c = w + 1);
      cover "the last step" (c = w + d);
      cover "past the end" (c > w + d);
      cover "the default end_value" (end_value = None);
      (* "up to warmup_steps it is linear from init_value to peak_value, and at
         s = step - warmup_steps after it end_value + (peak_value - end_value) *
         (1 + cos (pi * min s decay_steps / decay_steps)) / 2" *)
      let expected =
        if c <= w then linear init_value peak_value w c
        else e +. ((peak_value -. e) *. cosine_factor (ratio (c - w) d))
      in
      (* The cosine factor's [6u], a product and a sum: [8u] of the largest
         number, and [u] for the second-order terms. *)
      equal
        (within 9. (Float.max e (Float.max init_value peak_value)))
        expected
        (S.eval
           (S.warmup_cosine_decay ~init_value ~peak_value ~warmup_steps:w
              ~decay_steps:d ?end_value ())
           c))

let warmup_cosine_decay_refusals =
  cases
    "warmup_cosine_decay raises Invalid_argument if warmup_steps or \
     decay_steps is not positive"
    ~id:fst
    [
      ("warmup_steps 0", (0, 10));
      ("warmup_steps -1", (-1, 10));
      ("decay_steps 0", (10, 0));
      ("decay_steps -1", (10, -1));
    ]
    (fun (_, (warmup_steps, decay_steps)) ->
      raises_match Exn.invalid_arg (fun () ->
          S.warmup_cosine_decay ~init_value:0. ~peak_value:1. ~warmup_steps
            ~decay_steps ()))

(* Warm restarts *)

(* [cycle_start n t_mul k] is "c_k = n * (t_mul^k - 1) / (t_mul - 1) (n * k for
   t_mul = 1.)", an integer for a [t_mul] of 1 or 2. *)
let cycle_start n t_mul k = if t_mul = 1. then n * k else n * ((1 lsl k) - 1)

(* A t_mul of 1 or 2 makes every cycle start an integer, computed exactly at
   float32. The counters reach the start of the fourth cycle. *)
let cosine_decay_restarts_formula =
  prop ~count:300 "cosine_decay_restarts is its formula in each cycle"
    Gen.(
      let* n, t_mul =
        pair step_count (option (of_list ~pp:Format.pp_print_float [ 1.; 2. ]))
      in
      let start = cycle_start n (Option.value t_mul ~default:1.) in
      let specials =
        [ 0; start 1 - 1; start 1; start 2 - 1; start 2; start 3 ]
      in
      let+ v =
        quad value
          (option (sixty_fourths 1 128))
          (option fraction)
          (counter specials (start 3))
      in
      ((n, t_mul), v))
    (fun ((n, t_mul), (init_value, m_mul, alpha, c)) ->
      let t = Option.value t_mul ~default:1. in
      let m = Option.value m_mul ~default:1. in
      let a = Option.value alpha ~default:0. in
      let start = cycle_start n t in
      let rec cycle k = if start (k + 1) <= c then cycle (k + 1) else k in
      let k = cycle 0 in
      cover "step 0" (c = 0);
      cover "a restart" (k > 0 && c = start k);
      cover "the last step of a cycle" (c = start (k + 1) - 1);
      cover "the second cycle" (k = 1);
      cover "the third cycle" (k = 2);
      cover "t_mul 2" (t = 2.);
      cover "m_mul other than 1" (m <> 1.);
      cover "alpha > 0" (a > 0.);
      (* In cycle k, "init_value * m_mul^k * ((1 - alpha) * (1 + cos (pi * (step
         - c_k) / (n * t_mul^k))) / 2 + alpha)". *)
      let peak = init_value *. (m ** Float.of_int k) in
      let x =
        Float.of_int (c - start k) /. (Float.of_int n *. (t ** Float.of_int k))
      in
      let expected = peak *. (((1. -. a) *. cosine_factor x) +. a) in
      (* The factor of the peak is within [8u] (the cosine factor's [6u], a
         product and a sum), the peak within [10u] (pow's [9u] and a product),
         and their product adds [u]: [19u] of the peak, and [u] for the
         second-order terms. *)
      equal (within 20. peak) expected
        (S.eval
           (S.cosine_decay_restarts ~init_value ~decay_steps:n ?t_mul ?m_mul
              ?alpha ())
           c))

let cosine_decay_restarts_refusals =
  cases
    "cosine_decay_restarts raises Invalid_argument if decay_steps is not \
     positive, t_mul < 1.0, or m_mul is not positive"
    ~id:fst
    [
      ("decay_steps 0", (0, 1., 1.));
      ("decay_steps -1", (-1, 1., 1.));
      ("t_mul below 1", (10, Float.pred 1., 1.));
      ("t_mul nan", (10, Float.nan, 1.));
      ("m_mul 0", (10, 1., 0.));
      ("m_mul -1", (10, 1., -1.));
      ("m_mul nan", (10, 1., Float.nan));
    ]
    (fun (_, (decay_steps, t_mul, m_mul)) ->
      raises_match Exn.invalid_arg (fun () ->
          S.cosine_decay_restarts ~init_value:1. ~decay_steps ~t_mul ~m_mul ()))

(* pct_start is drawn at 0, at 1, below [1 / total_steps], where the warmup
   truncates to 0, and anywhere in [0, 1]. *)
let one_cycle_formula =
  prop ~count:800 "one_cycle is its formula in each phase"
    Gen.(
      let* total, pct_start =
        let* total = step_count in
        let+ pct_start =
          option
            (frequency
               [
                 (1, of_list ~pp:Format.pp_print_float [ 0.; 1. ]);
                 (1, float_range 0. (1. /. Float.of_int total));
                 (2, float_range 0. 1.);
               ])
        in
        (total, pct_start)
      in
      let w =
        truncate (Option.value pct_start ~default:0.3 *. Float.of_int total)
      in
      let+ v =
        quad value
          (option (float_range 0.5 100.))
          (option (float_range 0.5 1e4))
          (counter [ 0; w; w + 1; total; total + 1 ] (3 * total))
      in
      ((total, pct_start), v))
    (fun ((total_steps, pct_start), (max_value, div_factor, final_div_factor, c))
       ->
      let pct = Option.value pct_start ~default:0.3 in
      let w = truncate (pct *. Float.of_int total_steps) in
      cover "no warmup" (w = 0);
      cover "a pct_start whose warmup truncates to 0" (pct > 0. && w = 0);
      cover "no decay" (w = total_steps);
      cover "both phases" (0 < w && w < total_steps);
      cover "the end of warmup" (0 < w && w < total_steps && c = w);
      cover "the last step" (c = total_steps);
      cover "past the end" (c > total_steps);
      let init_value = max_value /. Option.value div_factor ~default:25. in
      let end_value = max_value /. Option.value final_div_factor ~default:1e4 in
      let decay s n =
        end_value +. ((max_value -. end_value) *. cosine_factor (ratio s n))
      in
      (* "Phase 1 (warmup): linear from max_value / div_factor to max_value over
         w = truncate (pct_start * total_steps) steps. Phase 2 (decay): cosine
         from max_value to max_value / final_div_factor over the remaining
         total_steps - w steps. Where w is 0, the schedule is phase 2 alone over
         total_steps; where w is total_steps, phase 1 alone, which then stays at
         max_value." *)
      let expected =
        if w = 0 then decay c total_steps
        else if w = total_steps || c <= w then linear init_value max_value w c
        else decay (c - w) (total_steps - w)
      in
      (* warmup_cosine_decay's [8u], and [u] each for rounding the two numbers
         one_cycle derives: [10u] of the largest, and [u] for the second-order
         terms. *)
      let largest = Float.max max_value (Float.max init_value end_value) in
      equal (within 11. largest) expected
        (S.eval
           (S.one_cycle ~max_value ~total_steps ?div_factor ?final_div_factor
              ?pct_start ())
           c))

let one_cycle_refusals =
  cases
    "one_cycle raises Invalid_argument if total_steps is not positive, \
     pct_start is outside [0, 1], or a factor is not positive"
    ~id:fst
    [
      ("total_steps 0", (0, 0.3, 25., 1e4));
      ("pct_start -0.1", (10, -0.1, 25., 1e4));
      ("pct_start 1.5", (10, 1.5, 25., 1e4));
      ("pct_start above 1", (10, Float.succ 1., 25., 1e4));
      ("pct_start nan", (10, Float.nan, 25., 1e4));
      ("div_factor 0", (10, 0.3, 0., 1e4));
      ("div_factor nan", (10, 0.3, Float.nan, 1e4));
      ("final_div_factor -1", (10, 0.3, 25., -1.));
      ("final_div_factor nan", (10, 0.3, 25., Float.nan));
    ]
    (fun (_, (total_steps, pct_start, div_factor, final_div_factor)) ->
      raises_match Exn.invalid_arg (fun () ->
          S.one_cycle ~max_value:1. ~total_steps ~pct_start ~div_factor
            ~final_div_factor ()))

(* Composition *)

let piecewise_constant_formula =
  prop ~count:300 "piecewise_constant returns values.(i) in the i-th segment"
    Gen.(
      let* boundaries =
        map (List.sort_uniq compare)
          (list ~size:(int_range 0 3) (int_range 0 200))
      in
      let k = List.length boundaries in
      let last = List.fold_left max 0 boundaries in
      let specials = 0 :: List.concat_map (fun b -> [ b; b + 1 ]) boundaries in
      let+ v =
        pair
          (list ~size:(int_range (k + 1) (k + 1)) value)
          (counter specials (last + 50))
      in
      (boundaries, v))
    (fun (boundaries, (values, c)) ->
      cover "step 0" (c = 0);
      cover "a boundary" (List.mem c boundaries);
      cover "one past a boundary" (List.mem (c - 1) boundaries);
      cover "past the last boundary"
        (boundaries <> [] && List.for_all (fun b -> c > b) boundaries);
      cover "no boundary" (boundaries = []);
      (* "a boundary counter belongs to the segment it ends": the counter's
         segment is the number of boundaries below it. The value is exact in
         float32. *)
      let i = List.length (List.filter (fun b -> b < c) boundaries) in
      equal float_exact (List.nth values i)
        (S.eval (S.piecewise_constant ~boundaries ~values) c))

let piecewise_constant_refusals =
  cases
    "piecewise_constant raises Invalid_argument if values is not one longer \
     than boundaries or boundaries is not strictly increasing"
    ~id:fst
    [
      ("1 value for 1 boundary", ([ 10 ], [ 1. ]));
      ("3 values for 1 boundary", ([ 10 ], [ 1.; 2.; 3. ]));
      ("boundaries [20; 10]", ([ 20; 10 ], [ 1.; 2.; 3. ]));
      ("boundaries [10; 10]", ([ 10; 10 ], [ 1.; 2.; 3. ]));
    ]
    (fun (_, (boundaries, values)) ->
      raises_match Exn.invalid_arg (fun () ->
          S.piecewise_constant ~boundaries ~values))

(* [placed segments] pairs each segment of a join with the counter it starts
   at. *)
let placed segments =
  snd
    (List.fold_left_map
       (fun c0 ((n, _, _, _) as segment) -> (c0 + n, (c0, segment)))
       0 segments)

(* Segments [(n, init_value, end_value, steps)] of {!S.linear} schedules. The
   counters reach 50 past the total. *)
let join_formula =
  prop ~count:500
    "join evaluates each segment at the counter relative to its start"
    Gen.(
      let* segments =
        list ~size:(int_range 1 3)
          (quad (int_range 1 50) value value_or_zero (int_range 1 50))
      in
      let starts = List.map fst (placed segments) in
      let total = List.fold_left (fun t (n, _, _, _) -> t + n) 0 segments in
      let specials =
        total :: List.concat_map (fun c0 -> [ c0 - 1; c0 ]) starts
      in
      let+ c = counter (List.filter (fun c -> c >= 0) specials) (total + 50) in
      (segments, c))
    (fun (segments, c) ->
      let placed = placed segments in
      let starts = List.map fst placed in
      let total = List.fold_left (fun t (n, _, _, _) -> t + n) 0 segments in
      cover "step 0" (c = 0);
      cover "a segment's first counter" (c > 0 && List.mem c starts);
      cover "a segment's last counter" (List.mem (c + 1) (total :: starts));
      cover "past the end" (c >= total);
      (* "a segment that starts at the counter c0 holds the counters c0 to c0 +
         n - 1, where it is s at 0 to n - 1. The last segment's schedule is used
         for all steps beyond the total." *)
      let c0, (_, init_value, end_value, steps) =
        List.fold_left
          (fun found (c0, segment) -> if c0 <= c then (c0, segment) else found)
          (List.hd placed) placed
      in
      let join =
        S.join
          (List.map
             (fun (n, init_value, end_value, steps) ->
               (n, S.linear ~init_value ~end_value ~steps))
             segments)
      in
      equal
        (within 4. (Float.max init_value end_value))
        (linear init_value end_value steps (c - c0))
        (S.eval join c))

(* The second of two ten-step segments starts at the counter 10, where it is its
   schedule at 0. *)
let join_segment_starts =
  test "each segment of a join starts at its own counter 0" (fun () ->
      let s =
        S.join
          [
            (10, S.linear ~init_value:0.0 ~end_value:10.0 ~steps:10);
            (10, S.linear ~init_value:100.0 ~end_value:110.0 ~steps:10);
          ]
      in
      equal
        (list (within 4. 110.))
        [ 0.0; 9.0; 100.0; 109.0; 110.0; 110.0 ]
        (List.map (S.eval s) [ 0; 9; 10; 19; 20; 21 ]))

let join_refusals =
  cases "join raises Invalid_argument if segments is empty or any n <= 0"
    ~id:fst
    [
      ("no segment", []);
      ("n 0", [ (0, Vega.rate 1.) ]);
      ("n -1", [ (-1, Vega.rate 1.) ]);
      ("a second segment of n 0", [ (5, Vega.rate 1.); (0, Vega.rate 1.) ]);
    ]
    (fun (_, segments) ->
      raises_match Exn.invalid_arg (fun () -> S.join segments))

let () =
  exit
    (run "vega schedules"
       [
         group "basic" [ linear_formula; linear_refusals ];
         group "decay"
           [
             cosine_decay_formula;
             cosine_decay_refusals;
             exponential_decay_formula;
             exponential_decay_refusals;
             polynomial_decay_formula;
             polynomial_decay_refusals;
           ];
         group "warmup"
           [
             warmup_cosine_formula;
             warmup_cosine_refusals;
             warmup_cosine_decay_formula;
             warmup_cosine_decay_refusals;
           ];
         group "warm restarts"
           [
             cosine_decay_restarts_formula;
             cosine_decay_restarts_refusals;
             one_cycle_formula;
             one_cycle_refusals;
           ];
         group "composition"
           [
             piecewise_constant_formula;
             piecewise_constant_refusals;
             join_formula;
             join_segment_starts;
             join_refusals;
           ];
       ])
