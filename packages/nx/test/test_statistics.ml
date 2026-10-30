(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's statistics: all, any and count against folds of their elements at
   every dtype; argmax and argmin against the first extreme in their order;
   var and std within their stated bound of the exact variance of integers;
   logsumexp and logmeanexp against their definitions. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let float_pp ppf v = Format.fprintf ppf "%h" v

let same_bits =
  array
    (Testable.make ~pp:float_pp
       ~equal:(fun a b ->
         (Float.is_nan a && Float.is_nan b)
         || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)))

(* Shapes and axes *)

let shape =
  Gen.array ~size:(Gen.int_range 0 3)
    (Gen.frequency [ (1, Gen.constant 0); (6, Gen.int_range 1 4) ])

let axes_of r =
  let open Gen in
  if r = 0 then constant None
  else
    let+ picks = array ~size:(constant r) bool in
    let axes =
      List.concat
        (List.mapi (fun a p -> if p then [ a ] else []) (Array.to_list picks))
    in
    if axes = [] then None else Some axes

let normal r = function None -> List.init r Fun.id | Some l -> l

(* Indices of [s] in C order. *)
let indices s =
  let n = Array.fold_left ( * ) 1 s in
  List.init n (fun i ->
      let idx = Array.make (Array.length s) 0 and k = ref i in
      for a = Array.length s - 1 downto 0 do
        idx.(a) <- !k mod s.(a);
        k := !k / s.(a)
      done;
      idx)

(* The fold of [f] from [init] over [xs] of shape [s] along [axes], in C order
   of the result. *)
let fold f init s axes xs =
  let keep idx =
    Array.of_list
      (List.filteri (fun a _ -> not (List.mem a axes)) (Array.to_list idx))
  in
  let acc = Hashtbl.create 16 in
  List.iteri
    (fun i idx ->
      let k = keep idx in
      let prev = Option.value ~default:init (Hashtbl.find_opt acc k) in
      Hashtbl.replace acc k (f prev xs.(i)))
    (indices s);
  let out = keep s in
  Array.of_list
    (List.map
       (fun idx -> Option.value ~default:init (Hashtbl.find_opt acc idx))
       (indices out))

(* all, any, count *)

let dtype = Gen.of_list (List.filter (fun (D.Any dt) -> D.bits dt >= 8) D.all)

type case = Case : ('v, 's) D.t * int array * int list option * string -> case

let truth_case =
  let open Gen in
  with_pp
    (fun ppf (Case (dt, s, _, _)) -> Format.fprintf ppf "%s %a" (D.name dt) pp_ints s)
    (let* (D.Any dt) = dtype in
     let* s = shape in
     let* axes = axes_of (Array.length s) in
     let n = Array.fold_left ( * ) 1 s in
     let+ data =
       string_of ~size:(constant (D.bytes dt n))
         (frequency [ (3, constant '\000'); (2, char) ])
     in
     Case (dt, s, axes, data))

let law_truth (Case (dt, s, axes, data)) =
  let x =
    Nx.Repr.of_array Nx.Host.v
      (A.v dt (L.contiguous (Iarray.of_array s)) (Rig.Buffer.of_string data))
  in
  let ax = normal (Array.length s) axes in
  cover "an empty axis" (List.exists (fun a -> s.(a) = 0) ax);
  let truths = Nx.to_array (Nx.cast Nx.bool x) in
  equal ~msg:"all" (array bool) (fold ( && ) true s ax truths)
    (Nx.to_array (Nx.all ?axes x));
  equal ~msg:"any" (array bool) (fold ( || ) false s ax truths)
    (Nx.to_array (Nx.any ?axes x));
  equal ~msg:"count" (array int64)
    (fold (fun n t -> if t then Int64.succ n else n) 0L s ax truths)
    (Nx.to_array (Nx.count ?axes (Nx.cast Nx.bool x)))

let test_truth_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  equal bool true (Nx.item [] (Nx.all (f64 [| nan; -1. |])));
  equal bool false (Nx.item [] (Nx.any (f64 [| -0.; 0. |])));
  equal bool true (Nx.item [] (Nx.all (f64 [||])));
  equal bool false (Nx.item [] (Nx.any (f64 [||])));
  let z = Nx.create Nx.complex128 [| 2 |] Complex.[| { re = 0.; im = 1. }; zero |] in
  equal (array bool) [| true |] (Nx.to_array (Nx.any ~keepdims:true z));
  equal (array bool) [| false |] (Nx.to_array (Nx.all ~keepdims:true z));
  let m = Nx.create Nx.bool [| 2; 3 |] [| true; false; true; true; true; true |] in
  equal (array int64) [| 2L; 3L |] (Nx.to_array (Nx.count ~axes:[ 1 ] m));
  equal (array int64) [| 5L |]
    (Nx.to_array (Nx.count ~keepdims:true (Nx.flatten (Nx.cast Nx.bit m))));
  equal (array int) [| 2; 1 |] (Nx.shape (Nx.all ~axes:[ 1 ] ~keepdims:true m));
  invalid ~by:"Nx.all" (fun () -> Nx.all ~axes:[ 2 ] m);
  invalid ~by:"Nx.count" (fun () -> Nx.count ~axes:[ 0; 0 ] m)

(* argmax, argmin *)

let extremes = [ -1.; 0.; -0.; 1.; 2.; nan; infinity; neg_infinity ]

let arg_case =
  let open Gen in
  let* s = array ~size:(int_range 0 3) (int_range 1 4) in
  let* axis = if Array.length s = 0 then constant None else option (int_range 0 (Array.length s - 1)) in
  let+ xs = array ~size:(constant (Array.fold_left ( * ) 1 s)) (of_list extremes) in
  (s, axis, xs)

(* [a] before [b] in [maximum]'s order, NaN above all. *)
let above a b =
  if Float.is_nan b then false
  else if Float.is_nan a then true
  else a > b || (a = b && Float.sign_bit b && not (Float.sign_bit a))

let first better ts =
  let best = ref 0 in
  Array.iteri (fun i t -> if better t ts.(!best) then best := i) ts;
  Int64.of_int !best

let law_arg (s, axis, xs) =
  let x = Nx.create Nx.float64 s xs in
  let lanes, along =
    match axis with
    | None -> ([ xs ], [||])
    | Some a ->
        let groups = Hashtbl.create 8 and order = ref [] in
        List.iteri
          (fun i idx ->
            let k = Array.of_list (List.filteri (fun b _ -> b <> a) (Array.to_list idx)) in
            if not (Hashtbl.mem groups k) then order := k :: !order;
            Hashtbl.replace groups k (xs.(i) :: Option.value ~default:[] (Hashtbl.find_opt groups k)))
          (indices s);
        ( List.rev_map (fun k -> Array.of_list (List.rev (Hashtbl.find groups k))) !order,
          [| a |] )
  in
  ignore along;
  let check name better (f : ?axis:int -> _ -> _) =
    equal ~msg:name (array int64)
      (Array.of_list (List.map (first better) lanes))
      (Nx.to_array (f ?axis x))
  in
  check "argmax" above (fun ?axis x -> Nx.argmax ?axis x);
  check "argmin" (fun a b -> if Float.is_nan a || Float.is_nan b then above a b else above b a)
    (fun ?axis x -> Nx.argmin ?axis x)

let test_arg_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let at f x = Nx.item [] (f x) in
  equal int64 1L (at Nx.argmax (f64 [| -0.; 0. |]));
  equal int64 1L (at Nx.argmin (f64 [| 0.; -0. |]));
  equal int64 1L (at Nx.argmin (f64 [| 1.; nan; nan; -5. |]));
  equal int64 0L (at Nx.argmax (Nx.create Nx.uint8 [| 2 |] [| 200; 1 |]));
  equal int64 1L (at Nx.argmax (Nx.create Nx.bool [| 3 |] [| false; true; true |]));
  equal int64 1L
    (at Nx.argmax
       (Nx.create Nx.complex128 [| 3 |]
          [| Complex.one; { re = 1.; im = 2. }; { re = 1.; im = 2. } |]));
  let m = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 9l; 3l; 7l; 2l; 9l |] in
  equal int64 1L (at Nx.argmax m);
  equal (array int) [| 1; 1 |] (Nx.shape (Nx.argmax ~keepdims:true m));
  equal (array int) [||] (Nx.shape (Nx.argmax m));
  equal (array int64) [| 1L; 0L; 1L |] (Nx.to_array (Nx.argmax ~axis:0 m));
  equal (array int) [| 2; 1 |] (Nx.shape (Nx.argmin ~axis:(-1) ~keepdims:true m));
  equal (array int64) [| 0L |] (Nx.to_array (Nx.argmax (Nx.scalar Nx.int32 4l) |> Nx.reshape [| 1 |]));
  equal (array int) [| 2; 0 |]
    (Nx.shape (Nx.argmax ~axis:0 (Nx.zeros Nx.int8 [| 0; 2; 0 |])));
  invalid ~by:"Nx.argmax" (fun () -> Nx.argmax (f64 [||]));
  invalid ~by:"Nx.argmin" (fun () -> Nx.argmin ~axis:1 (Nx.zeros Nx.int8 [| 2; 0 |]));
  invalid ~by:"Nx.argmax" (fun () -> Nx.argmax ~axis:2 m)

(* var, std *)

(* Integers about the [offsets], some large with a small spread, whose mean's
   rounding a deviation from it carries, in steps of [1]. *)
let moments_case offsets =
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
let overflow_case offset step =
  let open Gen in
  let* n = int_range 100 200 in
  let* xs =
    array ~size:(constant n)
      (map (fun i -> Float.of_int (offset + i)) (int_range (-1) 1))
  in
  let+ ddof = int_range 0 2 in
  (step, xs, ddof)

let offsets = [ (0, 1000); (1 lsl 20, 3); (1 lsl 20 + 1, 1000) ]

(* Where the mean's rounding, squared, passes float64's bound: float32 holds
   no integer this large. *)
let near_2_40 = (1 lsl 40 + 1, 3)

let gamma u k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u))

(* The exact population variance of integers, rounded once to float64: about
   the first, so that the squares stay exact. *)
let exact xs =
  let n = Array.length xs and c = int_of_float xs.(0) in
  let s = Array.fold_left (fun s x -> s + (int_of_float x - c)) 0 xs in
  let s2 =
    Array.fold_left (fun s x -> s + ((int_of_float x - c) * (int_of_float x - c))) 0 xs
  in
  Float.of_int ((n * s2) - (s * s)) /. Float.of_int (n * n)

(* The bound, in the terms' steps. *)
let law_moments (type s) (dt : (float, s) D.t) u (step, xs, ddof) =
  let n = Array.length xs in
  let x = Nx.create dt [| n |] (Array.map (fun x -> x *. step) xs) in
  if step > 1. then begin
    (* The deviations from the mean, in steps, and whether (Σ d)² overflows
       where Σ d² does not: the float's largest over [step²] is [2^10]. *)
    let c = Nx.item [] (Nx.mean x) /. step in
    let s1 = Array.fold_left (fun s x -> s +. (x -. c)) 0. xs in
    let s2 = Array.fold_left (fun s x -> s +. ((x -. c) *. (x -. c))) 0. xs in
    cover "the square of the deviations' sum overflows"
      (s1 *. s1 >= 0x1p10 && s2 < 0x1p10)
  end;
  let scale = Float.of_int n /. Float.of_int (n - ddof) in
  let v = exact xs *. scale in
  let mean_abs = Array.fold_left (fun s x -> s +. Float.abs x) 0. xs /. Float.of_int n in
  let g = gamma u n and m = mean_abs and e = exact xs in
  let bound =
    (((gamma u (n + 4) +. (2. *. g *. g)) *. e)
     +. (2. *. gamma u (n + 6) *. g *. m *. Float.sqrt e)
     +. (4. *. gamma u (n + 4) *. g *. g *. m *. m))
    *. scale
    +. (3. *. u *. v)
  in
  let got = Nx.item [] (Nx.var ~ddof x) /. (step *. step) in
  if Float.abs (got -. v) > bound then
    failf "var: %h, exactly %h, beyond %h" got v bound;
  let sd = Nx.item [] (Nx.std ~ddof x) /. step in
  if Float.abs (sd -. Float.sqrt v) > Float.sqrt bound +. (2. *. u *. Float.sqrt v) then
    failf "std: %h, exactly %h" sd (Float.sqrt v)

let test_moments_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let one = [| 1.; 2.; 3.; 4.; 5. |] in
  equal same_bits [| 2. |] (Nx.to_array (Nx.var ~keepdims:true (f64 one)));
  equal same_bits [| 2.5 |] (Nx.to_array (Nx.var ~ddof:1 ~keepdims:true (f64 one)));
  equal same_bits [| 2. |]
    (Nx.to_array (Nx.std ~keepdims:true (f64 [| 2.; 4.; 4.; 4.; 5.; 5.; 7.; 9. |])));
  equal same_bits [| nan |] (Nx.to_array (Nx.reshape [| 1 |] (Nx.var (f64 [||]))));
  equal same_bits [| nan |] (Nx.to_array (Nx.reshape [| 1 |] (Nx.std (f64 [| 1.; nan |]))));
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 3.; 2.; 2. |] in
  equal same_bits [| 1.; 0. |] (Nx.to_array (Nx.var ~axes:[ 1 ] m));
  (* Squared deviations that overflow, with the square of their sum. *)
  let var dt xs = Nx.to_array (Nx.var ~keepdims:true (Nx.create dt [| Array.length xs |] xs)) in
  equal same_bits [| infinity |] (var Nx.float32 [| 8.25e37; -8.25e37; 8.25e37 |]);
  equal same_bits [| infinity |] (var Nx.float32 [| 3.3e38; 3.3e38 |]);
  equal same_bits [| infinity |] (var Nx.float64 [| 4.25e307; -4.25e307; 4.25e307 |]);
  equal same_bits [| infinity |] (var Nx.bfloat16 [| 8.25e37; -8.25e37; 8.25e37 |]);
  equal (array int) [| 1; 2 |] (Nx.shape (Nx.std ~axes:[ 0 ] ~keepdims:true m));
  invalid ~by:"Nx.var" (fun () -> Nx.var ~ddof:(-1) (f64 one));
  invalid ~by:"Nx.var" (fun () -> Nx.var ~ddof:5 (f64 one));
  invalid ~by:"Nx.std" (fun () -> Nx.std ~axes:[ 1 ] (f64 one))

(* logsumexp, logmeanexp *)

let lse_case =
  let open Gen in
  let* n = int_range 1 40 in
  array ~size:(constant n) (float_range (-800.) 800.)

let law_lse xs =
  let n = Array.length xs in
  let m = Array.fold_left Float.max neg_infinity xs in
  let r = m +. Float.log (Array.fold_left (fun s x -> s +. Float.exp (x -. m)) 0. xs) in
  let x = Nx.create Nx.float64 [| n |] xs in
  let tol = Float.of_int (2 * n + 4) *. 0x1p-52 *. (Float.abs r +. 1.) in
  let got = Nx.item [] (Nx.logsumexp x) in
  if Float.abs (got -. r) > tol then failf "logsumexp: %h, by definition %h" got r;
  let r = r -. Float.log (Float.of_int n) in
  let got = Nx.item [] (Nx.logmeanexp x) in
  if Float.abs (got -. r) > tol then failf "logmeanexp: %h, by definition %h" got r

let test_lse_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let lse xs = Nx.to_array (Nx.logsumexp ~keepdims:true (f64 xs)) in
  let lme xs = Nx.to_array (Nx.logmeanexp ~keepdims:true (f64 xs)) in
  equal same_bits [| neg_infinity |] (lse [||]);
  equal same_bits [| neg_infinity |] (lse [| neg_infinity; neg_infinity |]);
  equal same_bits [| infinity |] (lse [| infinity; 1. |]);
  equal same_bits [| nan |] (lse [| nan; infinity |]);
  equal (array (float 1e-12)) [| 1000. +. Float.log 2. |] (lse [| 1000.; 1000. |]);
  equal same_bits [| nan |] (lme [||]);
  equal (array (float 1e-12)) [| 1000. |] (lme [| 1000.; 1000. |]);
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 0.; 0.; 1.; 1. |] in
  equal (array (float 1e-6)) [| Float.log 2.; 1. +. Float.log 2. |]
    (Nx.to_array (Nx.logsumexp ~axes:[ -1 ] m));
  invalid ~by:"Nx.logsumexp" (fun () -> Nx.logsumexp ~axes:[ 2 ] m)

(* At a float narrower than float32, [f] computes at float32 and rounds once:
   over every code, in rows of 16. *)
let narrow name
    (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t) =
  cases (name ^ " at a narrow float is float32's, rounded once")
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
          equal same_bits
            (Nx.to_array (Nx.cast dt (f (Nx.cast Nx.float32 x))))
            (Nx.to_array (f x))
      | _ -> ())

let () =
  exit
    (run "nx statistics"
       [
         group "all any count"
           [
             prop "folds of the elements as booleans" truth_case law_truth;
             test "cases" test_truth_cases;
           ];
         group "argmax argmin"
           [
             prop "the first extreme" arg_case law_arg;
             test "cases" test_arg_cases;
           ];
         group "var std"
           [
             prop "float64 within the bound" (moments_case (near_2_40 :: offsets))
               (law_moments D.Float64 0x1p-53);
             prop "float32 within the bound" (moments_case offsets)
               (law_moments D.Float32 0x1p-24);
             prop "float64 within the bound where (Σ d)² overflows"
               (overflow_case ((1 lsl 52) + 2) 0x1p507)
               (law_moments D.Float64 0x1p-53);
             prop "float32 within the bound where (Σ d)² overflows"
               (overflow_case ((1 lsl 23) + 2) 0x1p59)
               (law_moments D.Float32 0x1p-24);
             test "cases" test_moments_cases;
             narrow "var" (fun x -> Nx.var ~axes:[ 1 ] x);
             narrow "std" (fun x -> Nx.std ~axes:[ 1 ] ~ddof:1 x);
           ];
         group "logsumexp logmeanexp"
           [
             prop "by their definitions" lse_case law_lse;
             test "cases" test_lse_cases;
             narrow "logsumexp" (fun x -> Nx.logsumexp ~axes:[ 1 ] x);
             narrow "logmeanexp" (fun x -> Nx.logmeanexp ~axes:[ 1 ] x);
           ];
       ])
