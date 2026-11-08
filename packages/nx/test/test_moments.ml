(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Moments over ranges: each range's mean and variance against the exact
   ones (golden/moments_ranges.golden, written by gen/moments_ranges.py),
   within the bound nx.mli states, and the laws of its bits: a function of
   the range's bounds and rows alone, of no thread count, of each column
   alone, and for a narrow float the carrier's rounded once. *)

open Windtrap
module D = Nx.Dtype

let golden = "golden/moments_ranges.golden"
let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let bounds a = Nx.create Nx.int64 [| Array.length a |] (Array.map Int64.of_int a)
let clip n v = max 0 (min n v)

(* The golden *)

type case = {
  name : string;
  dtype : string;
  x : float array;
  lo : int array;
  hi : int array;
  mean : float array;
  var : float array;
}

let read_case ic header =
  let words () =
    match String.split_on_char ' ' (input_line ic) with
    | _ :: vs -> Array.of_list vs
    | [] -> [||]
  in
  let floats () = Array.map float_of_string (words ()) in
  let ints () = Array.map int_of_string (words ()) in
  match String.split_on_char ' ' header with
  | [ "case"; name; dtype; _; _ ] ->
      let x = floats () in
      let lo = ints () in
      let hi = ints () in
      let mean = floats () in
      let var = floats () in
      { name; dtype; x; lo; hi; mean; var }
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

(* A float format's precision, least normal and greatest exponents, and its
   carrier's unit roundoff. *)
type format = F : (float, 's) D.t * int * int * int * float -> format

let format = function
  | "float64" -> F (D.Float64, 53, -1022, 1023, epsilon_float /. 2.)
  | "float32" -> F (D.Float32, 24, -126, 127, 0x1p-24)
  | "float16" -> F (D.Float16, 11, -14, 15, 0x1p-24)
  | "bfloat16" -> F (D.Bfloat16, 8, -126, 127, 0x1p-24)
  | "float8_e4m3fn" -> F (D.Float8_e4m3fn, 4, -6, 8, 0x1p-24)
  | "float8_e5m2" -> F (D.Float8_e5m2, 3, -14, 15, 0x1p-24)
  | "float4_e2m1fn" -> F (D.Float4_e2m1fn, 2, 0, 2, 0x1p-24)
  | d -> failf "%s: a dtype %S" golden d

(* [v] stored in [dt] and read back: its rounding, saturation and NaN as
   {!Nx.cast} gives them. *)
let stored (type s) (dt : (float, s) D.t) v =
  Nx.item [] (Nx.cast Nx.float64 (Nx.cast dt (Nx.scalar Nx.float64 v)))

let same a b = (Float.is_nan a && Float.is_nan b) || a = b

(* The depth of a range of [k] rows that nx.mli bounds. *)
let depth k =
  if k <= 1 then 0
  else
    let rec bits v = if v = 0 then 0 else 1 + bits (v lsr 1) in
    1 + bits (k - 2)

(* Whether [y], a result stored in [dt], is within [tol] of the exact [z],
   given rounded to float64: the carrier's bound, then one rounding, the
   result's to a narrower [dt] or the reference's to float64, a subnormal's
   included. Where [z] and its bound reach past [dt]'s range, the stored
   extreme. *)
let within (type s) (dt : (float, s) D.t) ~p ~emin ~emax z tol y =
  let u = Float.ldexp 1. (-p) and tiny = Float.ldexp 1. (emin - p + 1) in
  let top = D.max_value dt in
  let top =
    if Float.is_finite top then top
    else Float.ldexp (2. -. Float.ldexp 1. (1 - p)) emax
  in
  let tol = tol +. (u *. (Float.abs z +. tol)) +. tiny in
  Float.abs (y -. z) <= tol
  || (Float.abs z +. tol >= top
      && same y (stored dt (Float.copy_sign Float.infinity z)))

let check_case c =
  let (F (dt, p, emin, emax, u)) = format c.dtype in
  let n = Array.length c.x in
  let x = Nx.cast dt (Nx.create Nx.float64 [| n |] c.x) in
  let mean, var = Nx.moments_ranges ~lo:(bounds c.lo) ~hi:(bounds c.hi) x in
  let mean = Nx.to_array (Nx.cast Nx.float64 mean) in
  let var = Nx.to_array (Nx.cast Nx.float64 var) in
  let wrong = ref [] in
  Array.iteri
    (fun i (lo, hi) ->
      let l = clip n lo and h = clip n hi in
      let k = max 0 (h - l) in
      let em = c.mean.(i) and ev = c.var.(i) in
      let ok =
        if not (Float.is_finite em) then
          same mean.(i) (stored dt em) && same var.(i) (stored dt Float.nan)
        else
          let rows = Array.sub c.x l k in
          let top = Array.fold_left Float.max Float.neg_infinity rows in
          let bottom = Array.fold_left Float.min Float.infinity rows in
          (* Half the rows' range, which no finite rows overflow. *)
          let half = (top *. 0.5) -. (bottom *. 0.5) in
          let d = Float.of_int (depth k) in
          let ur = u *. half *. 2. in
          let tm = (7. *. d *. ur) +. (u *. Float.abs em) in
          let tv =
            (((2. *. d) +. 5.) *. u *. ev)
            +. (((7. *. d *. d) -. (3. *. d)) *. ur *. Float.sqrt ev)
          in
          let ev_ok =
            if Float.is_finite ev then within dt ~p ~emin ~emax ev tv var.(i)
            else var.(i) = Float.infinity || same var.(i) (stored dt ev)
          in
          within dt ~p ~emin ~emax em tm mean.(i) && ev_ok
      in
      if not ok then
        wrong :=
          Printf.sprintf "[%d, %d): mean %h var %h, exact %h %h" lo hi mean.(i)
            var.(i) em ev
          :: !wrong)
    (Array.combine c.lo c.hi);
  equal ~msg:c.name (list string) [] (List.rev !wrong)

(* Laws *)

let level n seed =
  let u = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float64 [| n |] in
  Nx.add_s (Nx.mul_s u 1e-3) 6e4

let bits t = Nx.to_array (Nx.bitcast Nx.int64 t)

(* Rows, the ranges over them, and a range among them whose rows stay. *)
let local =
  Gen.(
    let* n = int_range 1 80 in
    let* k = int_range 1 6 in
    let bound = int_range (-2) (n + 2) in
    let* lo = array ~size:(constant k) bound in
    let* hi = array ~size:(constant k) bound in
    let* fill =
      of_list ~pp:Format.pp_print_float
        [ Float.nan; Float.infinity; Float.neg_infinity; -0.; 1e300 ]
    in
    let* extra = int_range 0 40 in
    let+ seed = int_range 0 1000 in
    (n, lo, hi, fill, extra, seed))

(* A range's bits are its clipped bounds' and its rows': the same with the
   rows outside it filled with [fill], rows appended past [x], and the other
   ranges dropped. *)
let law_local (n, lo, hi, fill, extra, seed) =
  let x = level n seed in
  let one i =
    let l = clip n lo.(i) and h = clip n hi.(i) in
    cover "a range of several rows" (h - l >= 3);
    cover "an empty range" (l >= h);
    let inside =
      Nx.logical_and
        (Nx.greater_equal_s (Nx.arange Nx.int64 0 n 1) (Int64.of_int l))
        (Nx.less_s (Nx.arange Nx.int64 0 n 1) (Int64.of_int h))
    in
    let y = Nx.where inside x (Nx.full Nx.float64 [| n |] fill) in
    let y = Nx.concatenate ~axis:0 [ y; level extra (seed + 1) ] in
    let m, v = Nx.moments_ranges ~lo:(bounds [| l |]) ~hi:(bounds [| h |]) y in
    (bits m, bits v)
  in
  let m, v = Nx.moments_ranges ~lo:(bounds lo) ~hi:(bounds hi) x in
  let m = bits m and v = bits v in
  Array.iteri
    (fun i _ ->
      equal (pair (array int64) (array int64)) ([| m.(i) |], [| v.(i) |]) (one i))
    lo

(* One thread or many, the bits are one. *)
let test_threads () =
  let n = 1 lsl 17 in
  let x = level n 3 in
  let hi = Nx.add_s (Nx.arange Nx.int64 0 n 1) 1L in
  let lo = Nx.sub_s hi 365L in
  let run () =
    let m, v = Nx.moments_ranges ~lo ~hi x in
    (bits m, bits v, bits (Nx.reduce_ranges Add ~lo ~hi x))
  in
  let many = run () in
  let one = Nx_cpu.one_thread run in
  equal (triple (array int64) (array int64) (array int64)) many one

(* Columns are independent: a row of three columns is three ranges' worth
   of each column's. *)
let test_columns () =
  let n = 37 in
  let x =
    Nx.reshape [| n; 3 |]
      (Nx.add_s (Nx.mul_s (Nx.Rng.uniform ~key:(Nx.Rng.key 5) Nx.float64 [| 3 * n |]) 10.) 1e5)
  in
  let lo = bounds [| -3; 0; 5; 30; 36; 12 |] and hi = bounds [| 4; 37; 5; 40; 37; 30 |] in
  let m, v = Nx.moments_ranges ~lo ~hi x in
  for j = 0 to 2 do
    let col = Nx.slice [ Nx.A; Nx.I j ] x in
    let mj, vj = Nx.moments_ranges ~lo ~hi col in
    equal ~msg:"mean" (array int64) (bits mj) (bits (Nx.slice [ Nx.A; Nx.I j ] m));
    equal ~msg:"variance" (array int64) (bits vj) (bits (Nx.slice [ Nx.A; Nx.I j ] v))
  done

(* A narrow float computes in float32 and rounds once. *)
let test_carrier () =
  let n = 70 in
  let lo = bounds [| 0; -2; 10; 60; 69; 30; 5 |]
  and hi = bounds [| 70; 3; 40; 80; 70; 30; 64 |] in
  let check (type s) (dt : (float, s) D.t) =
    let x =
      Nx.cast dt
        (Nx.add_s (Nx.mul_s (Nx.Rng.uniform ~key:(Nx.Rng.key 9) Nx.float64 [| n |]) 6.) 1.)
    in
    let m, v = Nx.moments_ranges ~lo ~hi x in
    let m32, v32 = Nx.moments_ranges ~lo ~hi (Nx.cast Nx.float32 x) in
    let b t = Nx.to_array (Nx.cast Nx.float64 t) in
    equal ~msg:(D.name dt) (pair (array float_exact) (array float_exact))
      (b (Nx.cast dt m32), b (Nx.cast dt v32))
      (b m, b v)
  in
  check D.Float16;
  check D.Bfloat16;
  check D.Float8_e4m3fn;
  check D.Float8_e5m2;
  check D.Float4_e2m1fn

(* Bounds past both ends, no row and one. *)
let test_small () =
  let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let lo = bounds [| -2; 0; 3; 4 |] and hi = bounds [| 1; 3; 3; 9 |] in
  let m, v = Nx.moments_ranges ~lo ~hi x in
  equal (array float_exact) [| 1.; 2.; Float.nan; 5. |] (Nx.to_array m);
  equal (array float_exact) [| 0.; 2. /. 3.; Float.nan; 0. |] (Nx.to_array v);
  let empty = Nx.zeros Nx.float32 [| 0; 2 |] in
  let m, v = Nx.moments_ranges ~lo:(bounds [| 0; -1 |]) ~hi:(bounds [| 1; 3 |]) empty in
  equal (array int) [| 2; 2 |] (Nx.shape m);
  equal (array float_exact) (Array.make 4 Float.nan) (Nx.to_array m);
  equal (array float_exact) (Array.make 4 Float.nan) (Nx.to_array v);
  let one = Nx.create Nx.float64 [| 1 |] [| -0. |] in
  let m, v = Nx.moments_ranges ~lo:(bounds [| 0 |]) ~hi:(bounds [| 1 |]) one in
  equal (array float_exact) [| 0. |] (Nx.to_array m);
  equal (array float_exact) [| 0. |] (Nx.to_array v)

let test_errors () =
  let by = "Nx.moments_ranges" in
  let b = Nx.zeros Nx.int64 [| 2 |] in
  invalid ~by (fun () -> Nx.moments_ranges ~lo:b ~hi:b (Nx.zeros Nx.float32 [||]));
  invalid ~by (fun () ->
      Nx.moments_ranges ~lo:b ~hi:(Nx.zeros Nx.int64 [| 3 |]) (Nx.zeros Nx.float32 [| 2 |]));
  invalid ~by (fun () ->
      let b = Nx.zeros Nx.int64 [| 2; 1 |] in
      Nx.moments_ranges ~lo:b ~hi:b (Nx.zeros Nx.float32 [| 2 |]))

let () =
  exit
    (run "nx moments"
       [
         cases "each range within its bound of the exact moments"
           ~id:(fun c -> c.name ^ " " ^ c.dtype)
           goldens check_case;
         prop "a range's bits read its rows and bounds alone" local law_local;
         test "one thread or many give one result" test_threads;
         test "each column is its own" test_columns;
         test "narrow floats compute in float32 and round once" test_carrier;
         test "bounds past both ends, no row and one" test_small;
         test "misuse raises" test_errors;
       ])
