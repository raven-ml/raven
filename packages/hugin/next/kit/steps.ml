(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Decimal steps and their exact multiples, powers, and the steps of time: the
   guide values Scale gives and the candidates Ticks scores. *)

(* Powers of ten *)

(* [pow10s.(k + 323)] is the float nearest [10^k], for [k] in
   \[[-323];[308]\]. *)
let pow10s =
  Array.init 632 (fun i -> float_of_string ("1e" ^ string_of_int (i - 323)))

let pow10 k =
  if k > 308 then Float.infinity else if k < -323 then 0. else pow10s.(k + 323)

(* [scale x k] is [x] multiplied by [10^k], or divided by [10^-k] if [k < 0], by
   the float nearest the power, or by [10^300] then the rest when that power is
   not a finite float. *)
let scale x k =
  if k >= 0 then
    if k <= 308 then x *. pow10 k else x *. pow10 300 *. pow10 (k - 300)
  else
    let n = -k in
    if n <= 308 then x /. pow10 n else x /. pow10 300 /. pow10 (n - 300)

(* Decimal steps *)

(* The step [m × 10^k], [m >= 1]. *)
type step = { m : int; k : int }

(* [nearest n k] is the float nearest [n × 10^k]. Where [n] and [10^|k|] are
   floats, one multiplication or division rounds once; elsewhere that
   approximation is moved to the float the exact value rounds to. *)
let nearest n k =
  let approx = scale (Float.of_int n) k in
  if Int.abs n <= 1 lsl 53 && Int.abs k <= 22 then approx
  else Decimal.nearest n k approx

(* Adding [0.] turns a negative multiple that underflows to [-0.] into [0.]. *)
let value s i = nearest (i * s.m) s.k +. 0.

(* [decompose t] is [(f, k)] with [t = f × 10^k] and [1 <= f < 10] for a
   positive finite [t]. *)
let decompose t =
  let k = Float.to_int (Float.floor (Float.log10 t)) in
  let f = scale t (-k) in
  if f >= 10. then (f /. 10., k + 1)
  else if f < 1. then (f *. 10., k - 1)
  else (f, k)

(* [target a b count] is the length of \[[a];[b]\], [a < b], over [count],
   decomposed. *)
let target a b count =
  let c = Float.of_int count in
  let l = b -. a in
  if Float.is_finite l then
    let t = l /. c in
    if t >= Float.min_float then decompose t
    else
      let f, k = decompose (l *. pow10 300 /. c) in
      (f, k - 300)
  else
    let f, k = decompose (((b /. 2.) -. (a /. 2.)) /. c) in
    let f = 2. *. f in
    if f >= 10. then (f /. 10., k + 1) else (f, k)

let sqrt50 = Float.sqrt 50.
let sqrt10 = Float.sqrt 10.
let sqrt2 = Float.sqrt 2.

(* [step a b count] is the step nearest by ratio to the length of \[[a];[b]\],
   [a < b], over [count]. *)
let step a b count =
  let f, k = target a b count in
  let m =
    if f >= sqrt50 then 10
    else if f >= sqrt10 then 5
    else if f >= sqrt2 then 2
    else 1
  in
  { m; k }

(* [index s x] is [x / (m × 10^k)], or infinite. *)
let index s x = scale x (-s.k) /. Float.of_int s.m

(* [gap x] is the distance from [|x|] to the float below it, the least subnormal
   at [0.]. *)
let gap x =
  let x = Float.abs x in
  if x = 0. then Float.succ 0. else x -. Float.pred x

(* [is_fine s a b] is [true] iff [s] is below a 32nd of the gap at both [a] and
   [b]: the multiples are then finer than the floats around the ends, and their
   indices there beyond [2^57]. The comparison is of logarithms, so that a step
   below the floats does not underflow. A gap is a power of two, and [32 × m ×
   10^k] either equals one or differs from it by far more than the logarithms'
   error, since [k log2 10] stays away from integers. *)
let is_fine s a b =
  let w = Float.log10 (32. *. Float.of_int s.m) +. Float.of_int s.k in
  w < Float.log10 (gap a) && w < Float.log10 (gap b)

(* [collect f] is the floats [f push] pushes, consecutive duplicates once. *)
let collect f =
  let acc = ref [] in
  let push v =
    match !acc with w :: _ when Float.equal v w -> () | _ -> acc := v :: !acc
  in
  f push;
  Array.of_list (List.rev !acc)

(* [every_float a b] is the floats of \[[a];[b]\], increasing. *)
let every_float a b =
  let rec loop acc x =
    if x > b then List.rev acc else loop (x :: acc) (Float.succ x)
  in
  Array.of_list (loop [] a)

(* [multiples s a b] is the distinct floats of the multiples of [s] in
   \[[a];[b]\], increasing, or every float of the domain if [s] is fine
   there. *)
let multiples s a b =
  if is_fine s a b then every_float a b
  else
    let i0 = Float.to_int (Float.floor (index s a)) - 1
    and i1 = Float.to_int (Float.ceil (index s b)) + 1 in
    let acc = ref [] and last = ref Float.nan in
    for i = i0 to i1 do
      let v = value s i in
      (* Values grow with their index, so equal ones are consecutive. *)
      if a <= v && v <= b && not (Float.equal v !last) then begin
        acc := v :: !acc;
        last := v
      end
    done;
    Array.of_list (List.rev !acc)

(* [floor_multiple s x] and [ceil_multiple s x] are the greatest multiple of [s]
   not above [x] and the least not below it, for a step that is not fine at
   [x]. *)
let floor_multiple s x =
  let i = ref (Float.to_int (Float.floor (index s x))) in
  while value s !i > x do
    decr i
  done;
  while value s (!i + 1) <= x do
    incr i
  done;
  value s !i

let ceil_multiple s x =
  let i = ref (Float.to_int (Float.ceil (index s x))) in
  while value s !i < x do
    incr i
  done;
  while value s (!i - 1) >= x do
    decr i
  done;
  value s !i

(* The linear rule: the multiples of the step for [count], or for [2] when that
   gives none and [count = 1]. *)
let linear a b count =
  let v = multiples (step a b count) a b in
  if Array.length v = 0 && count = 1 then multiples (step a b 2) a b else v

(* Powers *)

let is_integer_base b = Float.is_integer b && b >= 2. && b <= 16.

(* [power b i] is [b^i], the float nearest it in base 10. *)
let power b i = if b = 10. then pow10 i else Float.pow b (Float.of_int i)

(* [multiple b j i] is [j × b^i], computed as decimal multiples in base 10. *)
let multiple b j i =
  if j = 1 then power b i
  else if b = 10. then value { m = j; k = i } 1
  else Float.of_int j *. power b i

(* [stride r] is the [r]th of 1, 2, 5, 10, 20, 50, …, from 0. *)
let stride r = [| 1; 2; 5 |].(r mod 3) * int_of_float (pow10 (r / 3))

(* Steps of time *)

(* A step of time: an interval of the table and its duration in nanoseconds,
   with months of 30 days and years of 365 days. *)
type time_step = { interval : Time.interval; duration : float }

(* The table, finest first. *)
let time_steps =
  let ns_per_s = 1_000_000_000 in
  let day = 86_400e9 in
  let unit strides make =
    List.map
      (fun k ->
        let interval, duration = make k in
        { interval; duration })
      strides
  in
  let fixed ns k =
    let p = ns * k in
    (Time.nanoseconds p, Float.of_int p)
  in
  let decades make first last =
    List.concat_map
      (fun e ->
        let p = int_of_float (pow10 e) in
        unit [ 1; 2; 5 ] (fun k -> make (k * p)))
      (List.init (last - first + 1) (fun e -> first + e))
  in
  let by days make k = (make k, days *. day *. Float.of_int k) in
  Array.of_list
    (List.concat
       [
         decades (fixed 1) 0 8;
         unit [ 1; 5; 15; 30 ] (fixed ns_per_s);
         unit [ 1; 5; 15; 30 ] (fixed (60 * ns_per_s));
         unit [ 1; 3; 6; 12 ] (fixed (3600 * ns_per_s));
         unit [ 1; 2 ] (fixed (86_400 * ns_per_s));
         unit [ 1; 2 ] (by 7. Time.weeks);
         unit [ 1; 3; 6 ] (by 30. Time.months);
         decades (by 365. Time.years) 0 11;
       ])

(* [round_ns h l] is [h × 2^32 + l], [l] in \[[0];[2^32 - 1]\], rounded once to
   a float. *)
let round_ns h l =
  let neg = h < 0 in
  let mh, ml =
    if not neg then (h, l)
    else if l = 0 then (-h, 0)
    else (-h - 1, (1 lsl 32) - l)
  in
  let v =
    if mh < 1 lsl 21 then Float.of_int ((mh lsl 32) lor ml)
    else
      (* Keep 62 bits of the magnitude, the dropped ones as a sticky bit below
         the rounding position, and let the conversion round. *)
      let _, nb = Float.frexp (Float.of_int mh) in
      let s = Int.min 32 (62 - nb) in
      let m = (mh lsl s) lor (ml lsr (32 - s)) in
      let m = if ml land ((1 lsl (32 - s)) - 1) <> 0 then m lor 1 else m in
      Float.ldexp (Float.of_int m) (32 - s)
  in
  if neg then -.v else v

(* [ns_diff t a] is [t - a] in nanoseconds, rounded once to a float. Seconds are
   split into 32-bit halves so that every product fits an [int]. *)
let ns_diff (t : Time.t) (a : Time.t) =
  let hi s = Int64.to_int (Int64.shift_right s 32) in
  let lo s = Int64.to_int (Int64.logand s 0xFFFF_FFFFL) in
  let p = (hi t.sec - hi a.sec) * 1_000_000_000 in
  let q = ((lo t.sec - lo a.sec) * 1_000_000_000) + (t.nsec - a.nsec) in
  round_ns (p + (q asr 32)) (q land 0xFFFF_FFFF)

(* [nearest_time_step t] is the index of the step of time whose duration is
   nearest by ratio to [t > 0], the coarser of two equally near. *)
let nearest_time_step t =
  let best = ref 0 in
  let dist i = Float.abs (Float.log (time_steps.(i).duration /. t)) in
  for i = 1 to Array.length time_steps - 1 do
    (* Neighbouring durations multiply to no square, so no length is exactly as
       near to both. *)
    if dist i <= dist !best then best := i
  done;
  !best
