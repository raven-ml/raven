(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* {!Nx.var} and {!Nx.std}: each lane of golden/spread.golden (written by
   gen/spread.py) against its exact variance and standard deviation within the
   bound nx.mli states, lanes side by side as each alone; integer lanes drawn
   about large offsets against their exact variance; the values, NaNs and zeros
   the definition and its steps give; the laws of a power-of-two scale; on
   ordinary data the kernels' moments' bits; the narrow floats at float32,
   rounded once. *)

open Windtrap
module D = Nx.Dtype

let golden = "golden/spread.golden"

(* [got] within [tol] of [want], a NaN within nothing. *)
let near ~msg ~tol want got =
  equal ~msg (float (Float.max tol Float.min_float)) want got

(* The golden *)

type case = {
  name : string;
  dtype : string;
  ddof : int;
  x : float array;
  var : float * float;
  std : float * float;
}

let read_case ic header =
  let words () =
    match String.split_on_char ' ' (input_line ic) with
    | _ :: vs -> vs
    | [] -> []
  in
  let two () =
    match words () with
    | [ a; b ] -> (float_of_string a, float_of_string b)
    | _ -> failf "%s: a result line" golden
  in
  match String.split_on_char ' ' header with
  | [ "case"; name; dtype; _; ddof ] ->
      let x = Array.of_list (List.map float_of_string (words ())) in
      let var = two () in
      let std = two () in
      { name; dtype; ddof = int_of_string ddof; x; var; std }
  | _ -> failf "%s: a case header %S" golden header

let goldens =
  let path = Filename.concat (Filename.dirname Sys.executable_name) golden in
  In_channel.with_open_text path (fun ic ->
      let rec go acc =
        match In_channel.input_line ic with
        | None -> List.rev acc
        | Some h -> go (read_case ic h :: acc)
      in
      go [])

(* A float format's precision and least normal exponent. *)
type format = F : (float, 's) D.t * int * int -> format

let format = function
  | "float64" -> F (D.Float64, 53, -1022)
  | "float32" -> F (D.Float32, 24, -126)
  | "float16" -> F (D.Float16, 11, -14)
  | "bfloat16" -> F (D.Bfloat16, 8, -126)
  | d -> failf "%s: a dtype %S" golden d

(* Whether [y], a result stored in [dt], is within [tol] of the exact [z], read
   as float64: the bound of its carrier, the reference's rounding, then one
   rounding to [dt]. Where [z] and its bound reach past [dt]'s range, [+∞]. *)
let within (type s) (dt : (float, s) D.t) ~p ~emin z tol y =
  if z = Float.infinity then y = Float.infinity
  else
    let u = Float.ldexp 1. (-p) in
    (* A float64 reference rounds by half a subnormal; a narrow dtype's own
       rounding reaches its least subnormal. *)
    let tiny =
      match dt with
      | D.Float64 -> 0x1p-1075
      | D.Float32 -> 0.
      | _ -> Float.ldexp 1. (emin - p + 1)
    in
    let tol = tol +. (u *. (Float.abs z +. tol)) +. tiny in
    Float.abs (y -. z) <= tol
    || (Float.abs z +. tol >= D.max_value dt && y = Float.infinity)

let of_case (type s) (dt : (float, s) D.t) c =
  Nx.cast dt (Nx.create Nx.float64 [| Array.length c.x |] c.x)

let number v = Nx.item [] (Nx.cast Nx.float64 v)

let check_case c =
  let (F (dt, p, emin)) = format c.dtype in
  let x = of_case dt c in
  let v = number (Nx.var ~ddof:c.ddof x) in
  let s = number (Nx.std ~ddof:c.ddof x) in
  let z_v, tol_v = c.var and z_s, tol_s = c.std in
  if not (within dt ~p ~emin z_v tol_v v) then
    failf "var %s %s: %h, exact %h, allowance %h" c.dtype c.name v z_v tol_v;
  if not (within dt ~p ~emin z_s tol_s s) then
    failf "std %s %s: %h, exact %h, allowance %h" c.dtype c.name s z_s tol_s

(* Lanes of one size and ddof stacked, each lane's result within its own bound:
   each lane takes its own scale, and an extreme lane beside ordinary ones
   changes none of their results' correctness. *)
let groups =
  let tbl = Hashtbl.create 32 in
  List.iter
    (fun c ->
      let k = (c.dtype, Array.length c.x, c.ddof) in
      let old = Option.value ~default:[] (Hashtbl.find_opt tbl k) in
      Hashtbl.replace tbl k (c :: old))
    goldens;
  let all = Hashtbl.fold (fun k v acc -> (k, List.rev v) :: acc) tbl [] in
  List.filter (fun (_, v) -> List.length v > 1) all
  |> List.sort (fun (a, _) (b, _) -> compare a b)

let id_group ((d, n, k), _) = Printf.sprintf "%s n=%d ddof=%d" d n k

let check_group ((dtype, n, ddof), cs) =
  let (F (dt, p, emin)) = format dtype in
  let x = Nx.stack (List.map (fun c -> Nx.reshape [| n |] (of_case dt c)) cs) in
  let values f = Nx.to_array (Nx.cast Nx.float64 (f x)) in
  let vs = values (Nx.var ~axes:[ 1 ] ~ddof) in
  let ss = values (Nx.std ~axes:[ 1 ] ~ddof) in
  List.iteri
    (fun i c ->
      let z_v, tol_v = c.var and z_s, tol_s = c.std in
      if not (within dt ~p ~emin z_v tol_v vs.(i)) then
        failf "var lane %d %s: %h, exact %h" i c.name vs.(i) z_v;
      if not (within dt ~p ~emin z_s tol_s ss.(i)) then
        failf "std lane %d %s: %h, exact %h" i c.name ss.(i) z_s)
    cs

(* Values the definition gives *)

let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs
let var_of ?ddof xs = number (Nx.var ?ddof (f64 xs))
let std_of ?ddof xs = number (Nx.std ?ddof (f64 xs))

(* Each step of nx.mli's computation is exact on these lanes: the mean, the
   deviations, their sums and the quotients. *)
let test_values () =
  let one_to_five = [| 1.; 2.; 3.; 4.; 5. |] in
  equal ~msg:"var [1..5]: 10 / 5" float_exact 2. (var_of one_to_five);
  equal ~msg:"var [1..5], ddof 1: 10 / 4" float_exact 2.5
    (var_of ~ddof:1 one_to_five);
  equal ~msg:"std [2; 4; 4; 4; 5; 5; 7; 9]: √(32 / 8)" float_exact 2.
    (std_of [| 2.; 4.; 4.; 4.; 5.; 5.; 7.; 9. |]);
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 3.; 2.; 2. |] in
  equal ~msg:"var of each row of [[1; 3]; [2; 2]]" (array float_exact)
    [| 1.; 0. |]
    (Nx.to_array (Nx.var ~axes:[ 1 ] m));
  (* σ of ±the least subnormal is that subnormal: no other float is within the
     root of var's bound of it, and it rounds to itself. *)
  equal ~msg:"std of ±the least subnormal" float_exact 0x1p-1074
    (std_of [| 0x1p-1074; -0x1p-1074 |])

(* An element NaN or infinite makes the lane NaN, at every ddof. *)
let nan_lanes =
  [
    ("a NaN", [| Float.nan; 1. |]);
    ("+∞", [| Float.infinity; 1. |]);
    ("-∞", [| Float.neg_infinity; 1. |]);
    ("both infinities", [| Float.infinity; Float.neg_infinity; 0. |]);
    ("a NaN beside an extreme", [| Float.nan; 1e308 |]);
    ("+∞ beside an extreme", [| Float.infinity; 1e308 |]);
  ]

let check_nan_lane (_, xs) =
  equal ~msg:"var" float_exact Float.nan (var_of xs);
  equal ~msg:"std" float_exact Float.nan (std_of xs);
  equal ~msg:"var, ddof 1" float_exact Float.nan (var_of ~ddof:1 xs)

(* A lane of equal elements: the mean is the element, every deviation [+0.] or
   [-0.], their sums from [+0.] are [+0.], and so is the variance. *)
let zero_lanes =
  [
    ("-0. twice", [| -0.; -0. |]);
    ("one element", [| 1e300 |]);
    ("one negative element", [| -1e-300 |]);
    ("zeros", [| 0.; 0.; 0. |]);
    ("subnormals", [| 1e-320; 1e-320 |]);
  ]

let check_zero_lane (_, xs) =
  equal ~msg:"var" float_exact 0. (var_of xs);
  equal ~msg:"std" float_exact 0. (std_of xs)

let test_empty () =
  let nans = array float_exact in
  equal ~msg:"var of no element" float_exact Float.nan
    (number (Nx.var (Nx.zeros Nx.float32 [| 0 |])));
  equal ~msg:"std of no element, ddof 2" float_exact Float.nan
    (number (Nx.std ~ddof:2 (Nx.zeros Nx.float64 [| 0 |])));
  equal ~msg:"std over an empty axis" nans (Array.make 3 Float.nan)
    (Nx.to_array (Nx.std ~axes:[ 1 ] (Nx.zeros Nx.float32 [| 3; 0 |])));
  equal ~msg:"var along an empty leading axis" nans (Array.make 3 Float.nan)
    (Nx.to_array (Nx.var ~axes:[ 0 ] (Nx.zeros Nx.float32 [| 0; 3 |])));
  equal ~msg:"var over an axis of no lane" (array int) [| 0 |]
    (Nx.shape (Nx.var ~axes:[ 1 ] (Nx.zeros Nx.float32 [| 0; 3 |])))

let ddof_rows =
  let two = f64 [| 1.; 2. |] and five = f64 [| 1.; 2.; 3.; 4.; 5. |] in
  [
    ("var, ddof -1", "Nx.var", fun () -> ignore (Nx.var ~ddof:(-1) five));
    ("var, ddof 5 of 5", "Nx.var", fun () -> ignore (Nx.var ~ddof:5 five));
    ("var, ddof 2 of 2", "Nx.var", fun () -> ignore (Nx.var ~ddof:2 two));
    ("std, ddof -1", "Nx.std", fun () -> ignore (Nx.std ~ddof:(-1) two));
  ]

let check_ddof_row (_, by, f) =
  raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

(* Integer lanes against their exact variance *)

(* nx.cpu's bound (nx_cpu.mli, Logsumexp and Moments) on the population variance
   [v] of [n] terms of mean [mu] and mean magnitude [m], its blocks of [L]
   roundings and its tree of depth [D]; then a ddof's relative [2u], and for σ
   the root of that interval, rounded, as nx.mli's var and std state. *)
let moments_bound u n ~ddof ~mu ~m ~vd v =
  let fi = Float.of_int in
  let l = ((Stdlib.min n 1024 + 15) / 16) + Stdlib.min 3 (n - 1) in
  let rec depth b d = if b <= 1 then d else depth ((b + 1) / 2) (d + 1) in
  let ld = fi (l + depth ((n + 1023) / 1024) 0) in
  let e = (ld *. u *. m) +. (u *. Float.abs mu) in
  let f =
    ((ld +. 5.) *. u *. v)
    +. (2. *. (ld +. 1.) *. u *. e *. Float.sqrt v)
    +. (((3. *. ld) +. 7.) *. u *. e *. e)
  in
  let ratio = fi n /. fi (n - ddof) in
  let vb = (f *. ratio *. (1. +. (2. *. u))) +. (2. *. u *. vd) in
  let sigma = Float.sqrt vd in
  let hi = Float.sqrt (vd +. vb) *. (1. +. (2. *. u)) in
  let lo = Float.sqrt (Float.max 0. (vd -. vb)) *. (1. -. (2. *. u)) in
  (vb, sigma, Float.max (hi -. sigma) (sigma -. lo))

(* Integers about the [offsets], some large with a small spread, whose mean's
   rounding a deviation from it carries, in steps of [1]. *)
let integer_lanes offsets =
  let open Gen in
  let* n = int_range 1 64 in
  let* offset, w = of_list offsets in
  let* xs =
    array ~size:(constant n)
      (map (fun i -> Float.of_int (offset + i)) (int_range (-w) w))
  in
  let+ ddof = int_range 0 (Stdlib.min 2 (n - 1)) in
  (1., xs, ddof)

(* Integers within 1 of [offset], above the float's [2^p] so that a step is its
   last bit, in steps of a power of two [step]: with [step²] the largest float
   over [2^10], Σ d² stays finite where (Σ d)², n² times the mean's rounding
   squared, may overflow. *)
let overflow_lanes offset step =
  let open Gen in
  let* n = int_range 100 200 in
  let* xs =
    array ~size:(constant n)
      (map (fun i -> Float.of_int (offset + i)) (int_range (-1) 1))
  in
  let+ ddof = int_range 0 2 in
  (step, xs, ddof)

let offsets = [ (0, 1000); (1 lsl 20, 3); ((1 lsl 20) + 1, 1000) ]

(* Where the mean's rounding, squared, passes float64's bound: float32 holds no
   integer this large. *)
let near_2_40 = ((1 lsl 40) + 1, 3)

(* The exact variance of integers at [ddof], rounded once to float64: about the
   first, so that the squares stay exact, a quotient of integers. *)
let exact ?(ddof = 0) xs =
  let n = Array.length xs and c = int_of_float xs.(0) in
  let s = Array.fold_left (fun s x -> s + (int_of_float x - c)) 0 xs in
  let s2 =
    Array.fold_left
      (fun s x -> s + ((int_of_float x - c) * (int_of_float x - c)))
      0 xs
  in
  Float.of_int ((n * s2) - (s * s)) /. Float.of_int (n * (n - ddof))

(* In the terms' steps, which scale every term of the bound alike; the
   reference's own rounding adds a float64 unit. *)
let law_integers (type s) (dt : (float, s) D.t) u (step, xs, ddof) =
  let n = Array.length xs in
  let x = Nx.create dt [| n |] (Array.map (fun x -> x *. step) xs) in
  if step > 1. then begin
    let c = Nx.item [] (Nx.mean x) /. step in
    let s1 = Array.fold_left (fun s x -> s +. (x -. c)) 0. xs in
    let s2 = Array.fold_left (fun s x -> s +. ((x -. c) *. (x -. c))) 0. xs in
    cover "the square of the deviations' sum overflows"
      (s1 *. s1 >= 0x1p10 && s2 < 0x1p10)
  end;
  let fn = Float.of_int n in
  let mu = Array.fold_left ( +. ) 0. xs /. fn in
  let m = Array.fold_left (fun s x -> s +. Float.abs x) 0. xs /. fn in
  let vd = exact ~ddof xs in
  let vb, sigma, sb = moments_bound u n ~ddof ~mu ~m ~vd (exact xs) in
  near ~msg:"var"
    ~tol:(vb +. (epsilon_float /. 2. *. vd))
    vd
    (Nx.item [] (Nx.var ~ddof x) /. (step *. step));
  near ~msg:"std"
    ~tol:(sb +. (epsilon_float *. sigma))
    sigma
    (Nx.item [] (Nx.std ~ddof x) /. step)

(* Laws of the steps and the rescue *)

(* Elements in [1, 2) and a power: scaling by [2^k] scales the variance by
   [2^(2k)] and the root by [2^k], bit for bit, while nothing leaves the normal
   range: the steps commute with a power of two, and a lane the rescue scales
   takes another. *)
let covariance =
  Gen.(
    let* top = bool in
    let* signed = bool in
    let* n = int_range 2 40 in
    let* xs =
      array ~size:(constant n)
        (map
           (fun r -> if signed && r < 0.3 then -1. -. r else 1. +. r)
           (float_range 0. 0.99999))
    in
    let* k = if top then int_range 950 1022 else int_range (-1022) (-950) in
    let+ ddof = int_range 0 (n - 1) in
    (xs, k, ddof, top))

let law_covariance (xs, k, ddof, top) =
  let n = Array.length xs in
  cover "a scale past the variance's range" top;
  cover "a scale below its range" (not top);
  let at k =
    Nx.create Nx.float64 [| n |] (Array.map (fun x -> Float.ldexp x k) xs)
  in
  let v0 = number (Nx.var ~ddof (at 0)) in
  let want = Float.ldexp v0 (2 * k) in
  if (not (Float.is_finite want)) || Float.abs want >= 0x1p-1000 then
    equal ~msg:"var at 2^k" float_exact want (number (Nx.var ~ddof (at k)));
  let s0 = number (Nx.std ~ddof (at 0)) in
  equal ~msg:"std at 2^k" float_exact (Float.ldexp s0 k)
    (number (Nx.std ~ddof (at k)))

(* Equal elements [t] near the top, whose mean rounds: the deviations are one
   value [δ = t - c], a few units of [t]'s last place, so [n δ] and [n δ²] are
   exact, [Σ d²] equals [Σ d (Σ d / n)] and the variance is [+0.], a lane whose
   sum overflows taking it scaled by a power of two. *)
let constant_top =
  Gen.(
    let* mantissa = float_range 1. 1.99999 in
    let* e = one_of [ int_range 433 1019; int_range 1020 1023 ] in
    let+ n = int_range 2 33 in
    (mantissa, e, n))

(* The greatest float three times, and 1e308 twice. *)
let constant_top_examples =
  [
    (Float.ldexp max_float (-1023), 1023, 3);
    (Float.ldexp 1e308 (-1023), 1023, 2);
  ]

let law_constant_top (mantissa, e, n) =
  let t = Float.ldexp mantissa e in
  cover "a sum that overflows" (Float.of_int n *. t = Float.infinity);
  cover "not a power of two" (Float.round mantissa <> mantissa);
  let x = Nx.full Nx.float64 [| n |] t in
  equal ~msg:"var" float_exact 0. (number (Nx.var x));
  equal ~msg:"std" float_exact 0. (number (Nx.std x))

(* Strided, reversed and broadcast operands read as their copies. *)
let lanes_for_views =
  List.filter
    (fun c -> c.dtype = "float64" && Array.length c.x > 3 && c.ddof <= 1)
    goldens

let contiguous (type s) (dt : (float, s) D.t) a =
  Nx.cast dt (Nx.create Nx.float64 [| Array.length a |] a)

let check_views c =
  let (F (dt, _, _)) = format c.dtype in
  let n = Array.length c.x in
  let plain = contiguous dt c.x in
  let the_same ~msg f y =
    equal ~msg float_exact (number (f plain)) (number (f y))
  in
  let padded =
    Array.init (2 * n) (fun i -> if i mod 2 = 0 then c.x.(i / 2) else Float.nan)
  in
  let strided = Nx.slice [ Nx.Rs (0, -1, 2) ] (contiguous dt padded) in
  the_same ~msg:"strided var" (Nx.var ~ddof:c.ddof) strided;
  the_same ~msg:"strided std" (Nx.std ~ddof:c.ddof) strided;
  let backward = contiguous dt (Array.init n (fun i -> c.x.(n - 1 - i))) in
  the_same ~msg:"reversed var" (Nx.var ~ddof:c.ddof) (Nx.flip backward);
  the_same ~msg:"reversed std" (Nx.std ~ddof:c.ddof) (Nx.flip backward);
  (* Broadcast along a leading axis: each row is the lane. *)
  let wide = Nx.broadcast_to [| 3; n |] plain in
  equal ~msg:"broadcast rows" (array float_exact)
    (Array.make 3 (number (Nx.var ~ddof:c.ddof plain)))
    (Nx.to_array (Nx.cast Nx.float64 (Nx.var ~axes:[ 1 ] ~ddof:c.ddof wide)));
  (* Broadcast along the reduced axis: a constant lane. *)
  let same = Nx.broadcast_to [| n |] (contiguous dt [| c.x.(0) |]) in
  equal ~msg:"broadcast constant std" float_exact 0.
    (number (Nx.std ~ddof:c.ddof same));
  if Float.is_finite c.x.(0) && Float.abs c.x.(0) < 1e300 then
    equal ~msg:"broadcast constant var" float_exact 0.
      (number (Nx.var ~ddof:c.ddof same))

let identity dt =
  Nx_kernel.Prog.v ~ins:[| D.Any dt |] [| Nx_kernel.Prog.In 0 |] ~outs:[| 0 |]

let moments (type s) (x : (float, s, Nx.host) Nx.t) =
  let (_, v), () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
           axes = [| 0 |];
           prog = identity (Nx.dtype x);
           reductions = [ Moments (0, Nx.dtype x) ];
           loads = [| Plain x |];
           at = None;
         })
  in
  number v

let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* [got] within [tol] of the pair [hi + lo]: the error taken before any rounding
   of the pair, [got - hi] exact where they are close. *)
let off_pair ~msg ~tol (hi, lo) got =
  near
    ~msg:(Printf.sprintf "%s: %h, exactly %h" msg got hi)
    ~tol 0.
    (got -. hi -. lo)

(* The root of the pair [hi + lo], as a pair. *)
let root (hi, lo) =
  if hi = 0. then (0., 0.)
  else
    let r = Float.sqrt hi in
    (r, (Float.fma (-.r) r hi +. lo) /. (2. *. r))

(* [v n / k] as a pair [hi + lo], within about [2^-104] of it. *)
let scaled v n k =
  let p = v *. n in
  let pe = Float.fma v n (-.p) in
  let q = p /. k in
  (q, (Float.fma (-.q) k p +. pe) /. k)

(* At ddof 0, the kernels' moments' bits, and their root taken once in the
   carrier. A lane of magnitudes below 1 takes its root from the moments of the
   lane scaled by [2^k] into [[1, 2)], then scaled back: a power of two changes
   no bit of moments that fit, and it keeps the squares of [[least subnormal;
   0]] from underflowing, where {!Nx.std} still states the root of the exact
   variance. At a ddof, the moments scaled by [n / (n - ddof)] within a relative
   [2u], and σ their root within [2u]: half the variance's, and the root's
   rounding. *)
let ordinary =
  Gen.(
    let* is64 = bool in
    let* n = int_range 2 2000 in
    let* ddof = int_range 0 (n - 1) in
    let+ xs = array ~size:(constant n) (float_range (-16.) 16.) in
    (is64, n, ddof, xs))

let law_ordinary (is64, n, ddof, xs) =
  cover "float32" (not is64);
  cover "float64" is64;
  cover "a ddof above one" (ddof > 1);
  let run (type s) (dt : (float, s) D.t) u round =
    let x = Nx.cast dt (Nx.create Nx.float64 [| n |] xs) in
    let var = number (Nx.var ~ddof x) and std = number (Nx.std ~ddof x) in
    if ddof = 0 then begin
      equal ~msg:"var, the kernels' moments" float_exact (moments x) var;
      let es = Nx.to_array x in
      let m = Array.fold_left (fun m e -> Float.max m (Float.abs e)) 0. es in
      let k = if m = 0. || m >= 1. then 0 else 1 - snd (Float.frexp m) in
      let up = Nx.create dt [| n |] (Array.map (fun e -> Float.ldexp e k) es) in
      let root = round (sqrt (moments up)) in
      equal ~msg:"std, their root" float_exact
        (round (Float.ldexp root (-k)))
        std
    end
    else begin
      let hi, lo =
        scaled (moments x) (Float.of_int n) (Float.of_int (n - ddof))
      in
      off_pair ~msg:"var, the moments scaled" ~tol:(2. *. u *. hi) (hi, lo) var;
      let sigma = root (hi, lo) in
      off_pair ~msg:"std, their root" ~tol:(2. *. u *. fst sigma) sigma std
    end
  in
  if is64 then run Nx.float64 0x1p-53 Fun.id else run Nx.float32 0x1p-24 fl32

(* At a float narrower than float32, [f] computes at float32 and rounds once:
   over every code, in rows of 16. *)
let narrow name (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    =
  cases
    (name ^ " at a narrow float is float32's, rounded once")
    ~id:(fun (D.Any dt) -> D.name dt)
    (List.filter (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32) D.all)
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          let n = 1 lsl D.bits dt in
          let codes =
            match D.bits dt with
            | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 n 1)
            | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 n 1)
            | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 n 1)
          in
          let x = Nx.place Nx.Host.on (Nx.reshape [| n / 16; 16 |] codes) in
          equal (array float_exact)
            (Nx.to_array
               (Nx.cast Nx.float64 (Nx.cast dt (f (Nx.cast Nx.float32 x)))))
            (Nx.to_array (Nx.cast Nx.float64 (f x)))
      | _ -> ())

let () =
  exit
    (run "nx var and std"
       [
         group "var and std"
           [
             cases "each lane is within the bound var states of its exact value"
               ~id:(fun (i, c) -> Printf.sprintf "%d %s %s" i c.dtype c.name)
               (List.mapi (fun i c -> (i, c)) goldens)
               (fun (_, c) -> check_case c);
             cases "lanes side by side are each within their own bound"
               ~id:id_group groups check_group;
             test "have the values the definition gives" test_values;
             cases "a lane holding a NaN or an infinity is NaN" ~id:fst
               nan_lanes check_nan_lane;
             cases "a lane of equal elements is +0." ~id:fst zero_lanes
               check_zero_lane;
             test "of no element it is NaN, at every ddof" test_empty;
             cases "raise naming themselves for a negative ddof or one of n"
               ~id:(fun (id, _, _) -> id)
               ddof_rows check_ddof_row;
             prop
               "float64 integer lanes are within the bound of their exact value"
               (integer_lanes (near_2_40 :: offsets))
               (law_integers D.Float64 0x1p-53);
             prop
               "float32 integer lanes are within the bound of their exact value"
               (integer_lanes offsets)
               (law_integers D.Float32 0x1p-24);
             prop "float64 lanes whose (Σ d)² overflows are within the bound"
               (overflow_lanes ((1 lsl 52) + 2) 0x1p507)
               (law_integers D.Float64 0x1p-53);
             prop "float32 lanes whose (Σ d)² overflows are within the bound"
               (overflow_lanes ((1 lsl 23) + 2) 0x1p59)
               (law_integers D.Float32 0x1p-24);
             prop "a power of two scales var by its square and std by itself"
               covariance law_covariance;
             prop "equal elements near the top have variance +0."
               ~examples:constant_top_examples constant_top law_constant_top;
             cases "views and broadcasts read as copies"
               ~id:(fun (i, c) -> Printf.sprintf "%d %s" i c.name)
               (List.mapi (fun i c -> (i, c)) lanes_for_views)
               (fun (_, c) -> check_views c);
             prop "ordinary lanes have the kernels' moments, a ddof within 2u"
               ordinary law_ordinary
               ~examples:
                 [
                   (true, 2, 0, [| 0x1.000000ap-511; 0. |]);
                   (false, 2, 0, [| 0x1p-140; 0. |]);
                 ];
             narrow "var" (fun x -> Nx.var ~axes:[ 1 ] x);
             narrow "std" (fun x -> Nx.std ~axes:[ 1 ] ~ddof:1 x);
           ];
       ])
