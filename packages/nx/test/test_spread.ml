(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* {!Nx.var} and {!Nx.std} over the whole range of a float dtype: each lane of
   golden/spread.golden (written by gen/spread.py) against its exact variance
   and standard deviation within the bound nx.mli states, lanes side by side
   as each alone, the laws of a power-of-two scale, and on ordinary data the
   kernels' moments' bits. *)

open Windtrap
module D = Nx.Dtype

let golden = "golden/spread.golden"

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

(* Whether [y], a result stored in [dt], is within [tol] of the exact [z],
   read as float64: the bound of its carrier, the reference's rounding, then
   one rounding to [dt]. Where [z] and its bound reach past [dt]'s range,
   [+∞]. *)
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

(* Lanes of one size and ddof stacked, each lane's result within its own
   bound: each lane takes its own scale, and an extreme lane beside ordinary
   ones changes none of their results' correctness. *)
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
  let x =
    Nx.stack (List.map (fun c -> Nx.reshape [| n |] (of_case dt c)) cs)
  in
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

(* Spec values *)

let f64 = Nx.create Nx.float64
let f32 = Nx.create Nx.float32
let bits y = Int64.bits_of_float y
let u64 = epsilon_float /. 2.

(* [y] equals the variance or root stated by the definition, bit for bit. *)
let exactly ~msg z y = equal ~msg int64 (bits z) (bits y)

(* A relative tolerance for the results of the two-term cases, where the
   bound's constants are small. *)
let near ~msg ~rel z y =
  if Float.abs (y -. z) > rel *. Float.abs z then
    failf "%s: %h, expected %h" msg y z

let test_top () =
  let large = 1e308 in
  exactly ~msg:"var of two terms at the top" 0.
    (number (Nx.var (f64 [| 2 |] [| large; large |])));
  exactly ~msg:"std of two terms at the top" 0.
    (number (Nx.std (f64 [| 2 |] [| large; large |])));
  let m = max_float in
  exactly ~msg:"var of the greatest, constant" 0.
    (number (Nx.var (f64 [| 3 |] [| m; m; m |])));
  (* ±m: the variance m² passes the greatest value, the root is m. *)
  equal ~msg:"var of ±greatest" float_exact Float.infinity
    (number (Nx.var (f64 [| 2 |] [| m; -.m |])));
  near ~msg:"std of ±greatest" ~rel:(3. *. u64) m
    (number (Nx.std (f64 [| 2 |] [| m; -.m |])));
  (* An alternating ±1e308 of 64 elements: past the range, never NaN. *)
  let alt = Array.init 64 (fun i -> if i mod 2 = 0 then large else -.large) in
  equal ~msg:"var alternating" float_exact Float.infinity
    (number (Nx.var (f64 [| 64 |] alt)));
  near ~msg:"std alternating" ~rel:(3. *. u64) large
    (number (Nx.std (f64 [| 64 |] alt)));
  (* 1e154, the variance 2/3 of 1e308, fits. *)
  let apart = f64 [| 3 |] [| 1e154; -1e154; 0. |] in
  near ~msg:"var [1e154; -1e154; 0]" ~rel:(9. *. u64) (2. /. 3. *. 1e308)
    (number (Nx.var apart));
  near ~msg:"std [1e200; -1e200]" ~rel:(3. *. u64) 1e200
    (number (Nx.std (f64 [| 2 |] [| 1e200; -1e200 |])));
  (* float32, whose squares overflow at 1.8e19. *)
  let u32 = 0x1p-24 in
  near ~msg:"var f32 [1.5e19; -1.5e19; 0]" ~rel:(12. *. u32)
    (2. /. 3. *. 1.5e19 *. 1.5e19)
    (number (Nx.var (f32 [| 3 |] [| 1.5e19; -1.5e19; 0. |])));
  let s25 = number (Nx.std (f32 [| 2 |] [| 1e25; -1e25 |])) in
  near ~msg:"std f32 [1e25; -1e25]" ~rel:(3. *. u32) 1e25 s25

let test_bottom () =
  (* ±1e-170 and its neighbours: the squares underflow, the root is 1e-170. *)
  near ~msg:"std [1e-170; -1e-170]" ~rel:(3. *. u64) 1e-170
    (number (Nx.std (f64 [| 2 |] [| 1e-170; -1e-170 |])));
  (* The variance 1e-340 is below float64's range. *)
  exactly ~msg:"var [1e-170; -1e-170]" 0.
    (number (Nx.var (f64 [| 2 |] [| 1e-170; -1e-170 |])));
  (* The least subnormal: σ is itself. *)
  exactly ~msg:"std of ±the least subnormal" 0x1p-1074
    (number (Nx.std (f64 [| 2 |] [| 0x1p-1074; -0x1p-1074 |])));
  let u32 = 0x1p-24 in
  near ~msg:"std f32 [1e-25; -1e-25]" ~rel:(3. *. u32)
    (Int32.float_of_bits (Int32.bits_of_float 1e-25))
    (number (Nx.std (f32 [| 2 |] [| 1e-25; -1e-25 |])))

let test_specials () =
  let nan = Float.nan and inf = Float.infinity in
  let is_nan ~msg v = equal ~msg bool true (Float.is_nan v) in
  let lane l = f64 [| Array.length l |] l in
  List.iter
    (fun (name, x) ->
      is_nan ~msg:("var " ^ name) (number (Nx.var x));
      is_nan ~msg:("std " ^ name) (number (Nx.std x));
      is_nan ~msg:("var ddof " ^ name) (number (Nx.var ~ddof:1 x)))
    [
      ("a NaN term", lane [| nan; 1. |]);
      ("+∞", lane [| inf; 1. |]);
      ("-∞", lane [| neg_infinity; 1. |]);
      ("both infinities", lane [| inf; neg_infinity; 0. |]);
      ("a NaN beside an extreme", lane [| nan; 1e308 |]);
      ("+∞ beside an extreme", lane [| inf; 1e308 |]);
    ];
  (* -0 and zero variances are +0. *)
  exactly ~msg:"var of -0" 0. (number (Nx.var (lane [| -0.; -0. |])));
  exactly ~msg:"std of -0" 0. (number (Nx.std (lane [| -0.; -0. |])));
  exactly ~msg:"var of one element" 0. (number (Nx.var (lane [| 1e300 |])));
  exactly ~msg:"std of one element" 0. (number (Nx.std (lane [| -1e-300 |])));
  exactly ~msg:"std of zeros" 0. (number (Nx.std (lane [| 0.; 0.; 0. |])));
  exactly ~msg:"std of constant subnormals" 0.
    (number (Nx.std (lane [| 1e-320; 1e-320 |])))

let test_sizes () =
  let x03 = Nx.zeros Nx.float32 [| 0; 3 |] in
  let x30 = Nx.zeros Nx.float32 [| 3; 0 |] in
  let v = Nx.var (Nx.zeros Nx.float32 [| 0 |]) in
  equal ~msg:"var of no element" bool true (Float.is_nan (number v));
  let nans a = Array.for_all Float.is_nan a in
  equal ~msg:"std over an empty axis" bool true
    (nans (Nx.to_array (Nx.std ~axes:[ 1 ] x30)));
  equal ~msg:"its shape" (array int) [| 3 |]
    (Nx.shape (Nx.std ~axes:[ 1 ] x30));
  equal ~msg:"var along a zero leading axis" bool true
    (nans (Nx.to_array (Nx.var ~axes:[ 0 ] x03)));
  equal ~msg:"var over a nonzero axis of no lane" (array int) [| 0 |]
    (Nx.shape (Nx.var ~axes:[ 1 ] x03));
  equal ~msg:"std of none, ddof 2" bool true
    (Float.is_nan (number (Nx.std ~ddof:2 (Nx.zeros Nx.float64 [| 0 |]))));
  raises_match (Exn.invalid_arg ~sub:"Nx.var: ")
    (fun () -> ignore (Nx.var ~ddof:2 (f64 [| 2 |] [| 1.; 2. |])));
  raises_match (Exn.invalid_arg ~sub:"Nx.std: ")
    (fun () -> ignore (Nx.std ~ddof:(-1) (f64 [| 2 |] [| 1.; 2. |])))

(* Laws *)

let float64 = Nx.float64

(* Elements in [1, 2) and a power: scaling by [2^k] scales the variance by
   [2^(2k)] and the root by [2^k], bit for bit, while nothing leaves the
   normal range. *)
let covariance =
  Gen.(
    let* top = bool in
    let* signed = bool in
    let n = int_range 2 40 in
    let* n = n in
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
  let at k = Nx.create float64 [| n |] (Array.map (fun x -> Float.ldexp x k) xs) in
  let v0 = number (Nx.var ~ddof (at 0)) in
  let v = number (Nx.var ~ddof (at k)) in
  let want = Float.ldexp v0 (2 * k) in
  if not (Float.is_finite want) then equal ~msg:"var past" float_exact want v
  else if Float.abs want >= 0x1p-1000 then
    exactly ~msg:"var at 2^k" want v;
  let s0 = number (Nx.std ~ddof (at 0)) in
  let s = number (Nx.std ~ddof (at k)) in
  exactly ~msg:"std at 2^k" (Float.ldexp s0 k) s

(* Equal elements near the top, whose mean rounds: the variance is exactly
   0, as the definition states. *)
let constant_top =
  Gen.(
    let* mantissa = float_range 1. 1.99999 in
    let* e = one_of [ int_range 433 1019; int_range 1020 1023 ] in
    let+ n = int_range 2 33 in
    (mantissa, e, n))

let law_constant_top (mantissa, e, n) =
  let t = Float.ldexp mantissa e in
  cover "a sum that overflows" (e >= 1020);
  cover "not a power of two" (Float.round mantissa <> mantissa);
  let x = Nx.full float64 [| n |] t in
  exactly ~msg:"var" 0. (number (Nx.var x));
  exactly ~msg:"std" 0. (number (Nx.std x))

(* Strided, reversed and broadcast operands read as their copies. *)
let lanes_for_views =
  List.filter
    (fun c -> c.dtype = "float64" && Array.length c.x > 3 && c.ddof <= 1)
    goldens

let contiguous (type s) (dt : (float, s) D.t) a =
  Nx.cast dt (Nx.create float64 [| Array.length a |] a)

let check_views c =
  let (F (dt, _, _)) = format c.dtype in
  let n = Array.length c.x in
  let plain = contiguous dt c.x in
  let reference f = bits (number (f plain)) in
  let the_same ~msg f y = equal ~msg int64 (reference f) (bits (number (f y))) in
  let padded =
    Array.init (2 * n) (fun i ->
        if i mod 2 = 0 then c.x.(i / 2) else Float.nan)
  in
  let strided = Nx.slice [ Nx.Rs (0, -1, 2) ] (contiguous dt padded) in
  the_same ~msg:"strided var" (Nx.var ~ddof:c.ddof) strided;
  the_same ~msg:"strided std" (Nx.std ~ddof:c.ddof) strided;
  let backward = contiguous dt (Array.init n (fun i -> c.x.(n - 1 - i))) in
  the_same ~msg:"reversed var" (Nx.var ~ddof:c.ddof) (Nx.flip backward);
  the_same ~msg:"reversed std" (Nx.std ~ddof:c.ddof) (Nx.flip backward);
  (* Broadcast along a leading axis: each row is the lane. *)
  let wide = Nx.broadcast_to [| 3; n |] plain in
  let rows = Nx.to_array (Nx.cast float64 (Nx.var ~axes:[ 1 ] ~ddof:c.ddof wide)) in
  Array.iteri
    (fun i y ->
      equal ~msg:(Printf.sprintf "broadcast row %d" i) int64
        (bits (number (Nx.var ~ddof:c.ddof plain)))
        (bits y))
    rows;
  (* Broadcast along the reduced axis: a constant lane. *)
  let one = contiguous dt [| c.x.(0) |] in
  let same = Nx.broadcast_to [| n |] one in
  exactly ~msg:"broadcast constant" 0. (number (Nx.std ~ddof:c.ddof same));
  if Float.is_finite c.x.(0) && Float.abs c.x.(0) < 1e300 then
    exactly ~msg:"broadcast constant var" 0.
      (number (Nx.var ~ddof:c.ddof same))

let identity dt =
  Nx_kernel.Prog.v ~ins:[| D.Any dt |] [| Nx_kernel.Prog.In 0 |] ~outs:[| 0 |]

let moments (type s) (x : (float, s, Nx.host) Nx.t) =
  let (_, v), () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout =
             Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
           axes = [| 0 |];
           prog = identity (Nx.dtype x);
           reductions = [ Moments (0, Nx.dtype x) ];
           loads = [| Plain x |];
         })
  in
  number v

let fl32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Ordinary lanes keep the kernels' moments' bits, then the scale and the
   root taken once in the carrier. *)
let ordinary =
  Gen.(
    let* is64 = bool in
    let* n = int_range 2 2000 in
    let* ddof = int_range 0 (n - 1) in
    let+ xs =
      array ~size:(constant n) (float_range (-16.) 16.)
    in
    (is64, n, ddof, xs))

let law_ordinary (is64, n, ddof, xs) =
  cover "float32" (not is64);
  cover "float64" is64;
  cover "a ddof above one" (ddof > 1);
  let run (type s) (dt : (float, s) D.t) round =
    let x = Nx.cast dt (Nx.create float64 [| n |] xs) in
    let v = moments x in
    let scale = Float.of_int n /. Float.of_int (n - ddof) in
    let want_v = if ddof = 0 then v else round (v *. round scale) in
    let want_s = round (sqrt want_v) in
    exactly ~msg:"var, the kernels' bits" want_v (number (Nx.var ~ddof x));
    exactly ~msg:"std, the root" want_s (number (Nx.std ~ddof x))
  in
  if is64 then run Nx.float64 Fun.id else run Nx.float32 fl32

let () =
  exit
    (run "nx var and std"
       [
         cases "each lane within its bound of the exact value"
           ~id:(fun (i, c) -> Printf.sprintf "%d %s %s" i c.dtype c.name)
           (List.mapi (fun i c -> (i, c)) goldens) (fun (_, c) -> check_case c);
         cases "lanes side by side, each as alone" ~id:id_group groups
           check_group;
         test "at the top of the range" test_top;
         test "at the bottom of the range" test_bottom;
         test "NaN, infinities, zeros" test_specials;
         test "empty lanes and misuse" test_sizes;
         prop "a power of two scales the result" covariance law_covariance;
         prop "equal elements near the top have variance 0" constant_top
           law_constant_top;
         cases "views and broadcasts read as copies"
           ~id:(fun (i, c) -> Printf.sprintf "%d %s" i c.name)
           (List.mapi (fun i c -> (i, c)) lanes_for_views)
           (fun (_, c) -> check_views c);
         prop "ordinary lanes keep the kernels' bits" ordinary law_ordinary;
       ])
