(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The complex kinds that are not exact, at complex64 and complex128, as maps of
   one kind compiled by Rune.jit, which must compile them whole: no call is
   evaluated. Each holds its normwise bound (rune2.mli, Numerics) of the exact
   value at the points of golden/lower_complex, which gen/lower_complex.py
   writes from mpmath, each part correctly rounded with its remainder. At an
   operand with an infinite or a NaN part, each gives what lower_complex.mli's
   special values state, the sorts of nx's eager complex128 kind. *)

open Windtrap
module D = Nx_array.Dtype
module K = Nx_kernel.Prog
module P = Nx.Prim
module Rune = Rune2
module T = Nx.Ptree

let n = 256

(* Compiling *)

(* [counted f] is [f ()], whose compiled calls must compile whole: a declined
   lowering, evaluated by nx, fails. *)
let counted f =
  let before = Rune.evaluations () in
  let y = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  y

(* [staged dt node] maps [node] over operands of [dt]: nx's eager map, and the
   map staged in Rune.jit, which compiles once per shape. *)
let staged (type s) (dt : (Complex.t, s) D.t) arity (node : K.node) =
  let ins = Iarray.of_array (Array.make arity (D.Any dt)) in
  let node : K.node =
    match node with Op1 (u, _, j) -> Op1 (u, D.Any dt, j) | other -> other
  in
  let prog = K.of_node ~ins node in
  let map xs =
    let layout =
      Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape (List.hd xs)))
    in
    let loads = Iarray.of_list (List.map (fun x -> P.Plain x) xs) in
    fst
      (P.eval ~by:"t" (P.Map { layout; prog; outs = [ dt ]; loads; at = None }))
  in
  let compiled = Rune.jit T.(list tensor @-> returns tensor) map in
  (map, fun xs -> counted (fun () -> compiled xs))

(* Operand [j] of each element of [es], as an array of [dt]. *)
let operands (type s) (dt : (Complex.t, s) D.t) es =
  List.init
    (Array.length es.(0))
    (fun j ->
      Nx.create dt [| Array.length es |] (Array.map (fun e -> e.(j)) es))

(* Parts *)

let f32 v = Int32.float_of_bits (Int32.bits_of_float v)
let round wide v = if wide then v else f32 v
let u wide = if wide then 0x1p-53 else 0x1p-24
let least wide = if wide then Float.min_float else 0x1p-126
let greatest wide = if wide then Float.max_float else 0x1.fffffep127

(* Parts either side of each switch the kinds take: e^z's clamps, its overflow,
   where the hyperbolic kinds take e^|x| / 2, the trigonometric reductions'
   switch, the unit circle's neighbourhood, and the extremes. *)
let points wide =
  let common =
    [ 0.; 1.; 0.5; 2.; 0.6; 0.8; 3.; 1.5707963267948966; 3.141592653589793 ]
    @ [ 19.9; 20.; 20.1; 30.; 1e-30; 1e5; Float.infinity; Float.nan ]
  in
  let edges =
    if wide then
      [ 709.78; 710.4; 711.; 1456.; 2101.; 1e300; 1e-300; 5e-324 ]
      @ [ Float.max_float; 0x1.fffffffffffffp-1; 1. +. epsilon_float ]
    else
      [ 88.7; 89.4; 90.; 194.; 280.; 1e30; 1e-38; 1e-45; 0x1.fffffep127 ]
      @ [ 0x1.fffffep-1; 0x1.000002p0 ]
  in
  List.map (round wide) (List.concat_map (fun v -> [ v; -.v ]) (common @ edges))

let any_bits wide =
  if wide then Gen.map Int64.float_of_bits Gen.int64
  else Gen.map Int32.float_of_bits Gen.int32

let part wide =
  Gen.frequency
    [
      (4, Gen.of_list (points wide));
      (3, Gen.map (round wide) (Gen.float_range (-4.) 4.));
      (1, Gen.map (round wide) (Gen.float_range (-1e3) 1e3));
      (1, any_bits wide);
    ]

let number wide =
  Gen.map
    (fun (re, im) -> { Complex.re; im })
    (Gen.pair (part wide) (part wide))

(* Numbers each kind meets: every pair of parts among [±0], [±1], [±∞] and NaN,
   where Annex G's special values lie, the poles and branch points, both sides
   of the cuts, the unit circle and the axes. *)
let tuples =
  let c re im = { Complex.re; im } and inf = Float.infinity in
  let specials = [ 0.; -0.; 1.; -1.; inf; -.inf; Float.nan ] in
  let others =
    [
      c (-2.) 0.;
      c (-2.) (-0.);
      c 2. 0.;
      c 2. (-0.);
      c 0. 2.;
      c (-0.) 2.;
      c 0. (-2.);
      c (-0.) (-2.);
      c 0.6 0.8;
      c 0.8 (-0.6);
      c (-0.6) 0.8;
      c 1. 1e-30;
      c 1. (-1e-30);
    ]
  in
  List.concat_map (fun re -> List.map (c re) specials) specials @ others

(* |z' - z| / |z| in units of [u wide], [z] each part's [hi + lo]. *)
let normwise wide (z' : Complex.t) ((rh, rl), (ih, il)) =
  let d x h l = x -. h -. l in
  Float.hypot (d z'.re rh rl) (d z'.im ih il) /. Float.hypot rh ih /. u wide

(* The error, in [u wide], of a golden's [hi + lo] at an exact value of modulus
   [m]: each part's remainder is rounded to binary64, within half its least
   subnormal of the remainder's exact value. *)
let golden_error wide m = 0x1p-1074 /. m /. u wide

(* Kinds *)

(* A kind: its node, its bound in u at operands of the dtype [wide], and where
   its bound holds beyond the parts' finiteness and the result's range. *)
type kind = {
  name : string;
  node : K.node;
  arity : int;
  bound : bool -> Complex.t array -> float;
  domain : bool -> Complex.t array -> bool;
}

let fixed b _ _ = b
let finite (z : Complex.t) = Float.is_finite z.re && Float.is_finite z.im
let has_nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im
let zero v = v = 0.

let has_inf (z : Complex.t) =
  Float.abs z.re = Float.infinity || Float.abs z.im = Float.infinity

(* Each part's sort: a NaN, an infinity, of its sign but where [open_re] or
   [open_im] leaves the real or the imaginary part's open, a zero or another
   number. *)
let sorts (open_re, open_im) (z : Complex.t) =
  let c unsigned v =
    if Float.is_nan v then "nan"
    else if Float.abs v < Float.infinity then if v = 0. then "0" else "number"
    else if (not unsigned) && v < 0. then "-inf"
    else "inf"
  in
  (c open_re z.re, c open_im z.im)

(* How close, in [u] relative to the phase, [y ln2] must lie to a non-zero
   multiple of [π/2] for a rounded phase to change a sign: [2u] for the lowered
   phase, as much for the reference's, and under [4u] for this test's own. *)
let phase_margin = 8.

(* Whether [exp2 (x + iy)], at [x = +∞] and a finite [y], may take its parts'
   signs from a phase rounded across a multiple of [π/2]. *)
let crossed wide (z : Complex.t) =
  let t = z.im *. Float.log 2. and q = Float.pi /. 2. in
  let k = Float.round (t /. q) in
  z.re = Float.infinity && Float.is_finite z.im && k <> 0.
  && Float.abs (t -. (k *. q)) <= phase_margin *. u wide *. Float.abs t

(* The parts of [k]'s value at [z] whose infinity's sign Annex G leaves open, as
   Prog's Results section lists them, sin and cos through sinh and cosh at [iz],
   and those exp2's rounded phase gives. *)
let open_signs wide k (z : Complex.t) =
  let inf v = Float.abs v = Float.infinity in
  let wild v = not (Float.is_finite v) in
  match k.name with
  | "exp" -> (z.re = Float.infinity && wild z.im, false)
  | "exp2" ->
      let c = crossed wide z in
      ((z.re = Float.infinity && wild z.im) || c, c)
  | "sinh" -> (inf z.re && wild z.im, false)
  | "cosh" | "cos" -> (inf z.re && inf z.im, false)
  | "sin" -> (false, inf z.im && wild z.re)
  | "sqrt" -> (false, z.re = Float.neg_infinity && Float.is_nan z.im)
  | "asin" | "acos" -> (false, inf z.re && Float.is_nan z.im)
  | "asinh" -> (Float.is_nan z.re && inf z.im, false)
  | _ -> (false, false)

let anywhere _ _ = true

let unary ?(domain = anywhere) name k bound =
  { name; node = Op1 (Unary k, D.Any D.Complex64, 0); arity = 1; bound; domain }

let binary ?(domain = anywhere) name k bound =
  { name; node = Op2 (Binary k, 0, 1); arity = 2; bound; domain }

let tiny w v = v <> 0. && Float.abs v < least w

(* Mul computes each part as a fused multiply-add over the other product, so its
   bound holds where no product of parts is subnormal. *)
let products_normal w (e : Complex.t array) =
  let x = e.(0) and y = e.(1) in
  not
    (List.exists (tiny w)
       [ x.re *. y.re; x.im *. y.im; x.re *. y.im; x.im *. y.re ])

(* Smith's division holds its bound where its operands' parts are below half the
   greatest float and the divisor's larger part is normal. *)
let smith w (e : Complex.t array) =
  let h = greatest w /. 2. and d = e.(Array.length e - 1) in
  Array.for_all
    (fun (z : Complex.t) -> Float.abs z.re < h && Float.abs z.im < h)
    e
  && Float.max (Float.abs d.re) (Float.abs d.im) >= least w

(* pow holds its bound where |w| |log z| is below the greatest float. *)
let products w (e : Complex.t array) =
  Complex.norm e.(1) *. Complex.norm (Complex.log e.(0)) < greatest w

(* |w log z|, which pow's bound grows with. *)
let wlog (e : Complex.t array) =
  let p = Complex.norm (Complex.mul e.(1) (Complex.log e.(0))) in
  if Float.is_nan p then 0. else p

(* Each kind's bound (rune2.mli, Numerics). *)
let unaries =
  K.
    [
      unary "recip" Recip (fixed 4.) ~domain:smith;
      unary "sign" Sign (fixed 3.);
      unary "sqrt" Sqrt (fixed 3.);
      unary "exp" Exp (fixed 31.);
      unary "exp2" Exp2 (fun _ e -> 30. +. (1.4 *. Float.abs e.(0).im));
      unary "log" Log (fixed 18.);
      unary "log2" Log2 (fixed 20.);
      unary "sin" Sin (fixed 67.);
      unary "cos" Cos (fixed 57.);
      unary "tan" Tan (fixed 73.);
      unary "sinh" Sinh (fixed 67.);
      unary "cosh" Cosh (fixed 57.);
      unary "tanh" Tanh (fixed 73.);
      unary "asin" Asin (fixed 23.);
      unary "acos" Acos (fixed 21.);
      unary "atan" Atan (fixed 21.);
      unary "asinh" Asinh (fixed 23.);
      unary "acosh" Acosh (fixed 21.);
      unary "atanh" Atanh (fixed 21.);
    ]

let binaries =
  K.
    [
      binary "mul" Mul (fixed 2.) ~domain:products_normal;
      binary "fdiv" Fdiv (fixed 4.) ~domain:smith;
      binary "pow" Pow (fun _ e -> 31. +. (20. *. wlog e)) ~domain:products;
    ]

(* Meeting the bounds *)

let hex wide v =
  if wide then Printf.sprintf "%Lx" (Int64.bits_of_float v)
  else Printf.sprintf "%lx" (Int32.bits_of_float v)

let show wide (z : Complex.t) =
  Printf.sprintf "(%s %s)" (hex wide z.re) (hex wide z.im)

(* Whether [k]'s bound holds at operands [e] of exact value [r]: every part
   finite, in [k]'s domain, [r]'s modulus a normal float and no part of [r]
   subnormal. *)
let bounded wide k e ((rh, _), (ih, _)) =
  let m = Float.hypot rh ih in
  Array.for_all finite e && k.domain wide e
  && m >= least wide
  && m <= greatest wide
  && (not (tiny wide rh))
  && not (tiny wide ih)

(* Whether [got] meets the exact value [r] within [tol] u normwise, or is
   infinite where [tol] u reaches past the greatest float. *)
let within wide tol (((rh, _), (ih, _)) as r) (got : Complex.t) =
  normwise wide got r <= tol +. golden_error wide (Float.hypot rh ih)
  || (not (finite got))
     && (not (has_nan got))
     && Float.hypot rh ih *. (1. +. (tol *. u wide)) >= greatest wide

(* The direction [Sign] gives at [z], whose infinite parts name it. *)
let direction (z : Complex.t) =
  let d v = if Float.abs v = Float.infinity then Float.copy_sign 1. v else 0. in
  let x = d z.re and y = d z.im in
  let m = Float.hypot x y in
  ((x /. m, 0.), (y /. m, 0.))

(* Whether [got] meets nx's value [r] at operands [e] with a part not finite:
   where a unary kind's operand has an infinite part, each part's sort as [r]'s,
   the signs {!open_signs} leaves open excepted, and [Sign]'s direction where no
   part is a NaN; where an operand has a NaN part and none an infinite one, a
   NaN part, but [1] for [z^w] at a [w] of zero parts. *)
let special wide k e (r : Complex.t) (got : Complex.t) =
  if k.arity = 1 && has_inf e.(0) then
    let signs = open_signs wide k e.(0) in
    sorts signs got = sorts signs r
    && (k.name <> "sign"
       || has_nan e.(0)
       || within wide (k.bound wide e) (direction e.(0)) got)
  else if Array.exists has_nan e && not (Array.exists has_inf e) then
    if k.name = "pow" && zero e.(1).re && zero e.(1).im then
      got.re = 1. && got.im = 0.
    else has_nan got
  else true

let failures name rows =
  equal ~msg:name (list string) []
    (List.filteri (fun i _ -> i < 5) (List.filter_map Fun.id rows))

let describe wide name show_ref e got err tol =
  Printf.sprintf "%s%s = %s at %s, reference %s, %g u over %g" name
    (String.concat "" (Array.to_list (Array.map (show wide) e)))
    (show wide got)
    (if wide then "complex128" else "complex64")
    show_ref err tol

(* Drawn operands *)

let rounded wide (z : Complex.t) =
  { Complex.re = round wide z.re; im = round wide z.im }

(* A case: the dtype, and per element its operands. *)
let case k =
  let open Gen in
  let* wide = bool in
  let z = frequency [ (4, number wide); (1, of_list tuples) ] in
  let element = map Array.of_list (list ~size:(constant k.arity) z) in
  map
    (fun es -> (wide, Array.map (Array.map (rounded wide)) es))
    (array ~size:(constant n) element)

(* Every tuple at each dtype, operand [j] of element [i] the tuple [i + j]. *)
let tupled k =
  let ts = Array.of_list tuples in
  let z wide i = rounded wide ts.(i mod Array.length ts) in
  let element wide i = Array.init k.arity (fun j -> z wide (i + j)) in
  List.map (fun wide -> (wide, Array.init n (element wide))) [ false; true ]

(* [k] at drawn operands with a part not finite, as {!special} states, against
   nx's complex128 kind at the operands widened. *)
let law k =
  let c64 = snd (staged D.Complex64 k.arity k.node)
  and nx, c128 = staged D.Complex128 k.arity k.node in
  prop ~count:20 ~examples:(tupled k) k.name (case k) (fun (wide, es) ->
      let r = Nx.to_array (nx (operands D.Complex128 es)) in
      let got =
        if wide then Nx.to_array (c128 (operands D.Complex128 es))
        else Nx.to_array (c64 (operands D.Complex64 es))
      in
      let check i e =
        if Array.for_all finite e || special wide k e r.(i) got.(i) then None
        else Some (describe wide k.name (show true r.(i)) e got.(i) nan 0.)
      in
      let cover_some label f = cover label (Array.exists f es) in
      cover "complex64" (not wide);
      cover "complex128" wide;
      cover_some "a NaN part" (Array.exists has_nan);
      cover_some "an infinite part" (Array.exists has_inf);
      cover_some "two non-finite parts"
        (Array.exists (fun (z : Complex.t) ->
             (not (Float.is_finite z.re)) && not (Float.is_finite z.im)));
      cover_some "a -0 part beside a part not finite"
        (Array.exists (fun (z : Complex.t) ->
             (not (finite z))
             && ((Float.sign_bit z.re && zero z.re)
                || (Float.sign_bit z.im && zero z.im))));
      failures k.name (Array.to_list (Array.mapi check es)))

(* Goldens *)

(* The rows of golden/lower_complex/<name>.golden at complex128 if [wide], else
   at complex64: operands, then each part of the value correctly rounded and its
   remainder. *)
let golden name wide =
  let ic = open_in ("golden/lower_complex/" ^ name ^ ".golden") in
  let rows = ref [] in
  (try
     while true do
       match String.split_on_char '\t' (input_line ic) with
       | d :: cols when d = if wide then "c128" else "c64" ->
           let r = Array.of_list (List.map Float.of_string cols) in
           let m = Array.length r - 4 in
           let e =
             Array.init (m / 2) (fun j ->
                 { Complex.re = r.(2 * j); im = r.((2 * j) + 1) })
           in
           rows := (e, ((r.(m), r.(m + 1)), (r.(m + 2), r.(m + 3)))) :: !rows
       | _ -> ()
     done
   with End_of_file -> close_in ic);
  greater ~msg:(name ^ "'s rows") int ~than:0 (List.length !rows);
  List.rev !rows

(* [k] within its bound of the goldens' values at both dtypes, where the bound
   holds. *)
let accurate k =
  let at (type s) (dt : (Complex.t, s) D.t) wide =
    let rows =
      List.filter (fun (e, r) -> bounded wide k e r) (golden k.name wide)
    in
    greater ~msg:(k.name ^ "'s bounded rows") int ~than:0 (List.length rows);
    let es = Array.of_list (List.map fst rows) in
    let got = Nx.to_array (snd (staged dt k.arity k.node) (operands dt es)) in
    List.mapi
      (fun i (e, r) ->
        let tol = k.bound wide e in
        if within wide tol r got.(i) then None
        else
          let (rh, _), (ih, _) = r in
          Some
            (describe wide k.name
               (show true { Complex.re = rh; im = ih })
               e got.(i)
               (normwise wide got.(i) r)
               tol))
      rows
  in
  test (k.name ^ " within its bound") (fun () ->
      failures k.name (at D.Complex64 false @ at D.Complex128 true))

let () =
  exit
    (run "Lower_complex"
       [
         group "special values" (List.map law (unaries @ binaries));
         group "bounds" (List.map accurate (unaries @ binaries));
       ])
