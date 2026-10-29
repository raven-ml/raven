(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The complex kinds that are not exact, lowered and compiled on the host,
   within their normwise bounds of a reference: nx's eager complex128 kinds, at
   the operands widened for complex64, and within the sum of both bounds for
   complex128. Each case is one compiled map over a few hundred numbers drawn to
   reach the axes, the branch cuts, the poles, the unit circle and the overflow
   edges. *)

open Windtrap
module D = Nx_array.Dtype
module K = Nx_kernel.Prog
module P = Nx.Prim

let n = 256

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

(* Numbers each kind meets: every pair of parts among [±0], [±1], [±∞] and
   NaN, where Annex G's special values lie, the poles and branch points, both
   sides of the cuts, the unit circle and the axes. *)
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

(* Kinds *)

(* A kind: its node, its bound in u at an element of the dtype [wide], nx's
   bound at complex128, and the cases it must reach. *)
type kind = {
  name : string;
  node : K.node;
  arity : int;
  bound : bool -> Complex.t array -> float;
  nx_bound : Complex.t array -> float;
  domain : bool -> Complex.t array -> bool;
  covers : (string * (bool -> Complex.t array -> bool)) list;
}

let fixed b _ _ = b
let nx b _ = b
let finite (z : Complex.t) = Float.is_finite z.re && Float.is_finite z.im
let has_nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im

let has_inf (z : Complex.t) =
  Float.abs z.re = Float.infinity || Float.abs z.im = Float.infinity

(* Each part's class: a NaN, an infinity, of its sign but where [open_re] or
   [open_im] leaves the real or the imaginary part's open, a zero or another
   number. *)
let classes (open_re, open_im) (z : Complex.t) =
  let c unsigned v =
    if Float.is_nan v then "nan"
    else if Float.abs v < Float.infinity then if v = 0. then "0" else "number"
    else if (not unsigned) && v < 0. then "-inf"
    else "inf"
  in
  (c open_re z.re, c open_im z.im)

(* How close, in [u] relative to the phase, [y ln2] must lie to a non-zero
   multiple of [π/2] for a rounded phase to change a sign: [2u] for the
   lowered phase, as much for the reference's, and under [4u] for this test's
   own. *)
let phase_margin = 8.

(* Whether [exp2 (x + iy)], at [x = +∞] and a finite [y], may take its parts'
   signs from a phase rounded across a multiple of [π/2]. *)
let crossed wide (z : Complex.t) =
  let t = z.im *. Float.log 2. and q = Float.pi /. 2. in
  let k = Float.round (t /. q) in
  z.re = Float.infinity && Float.is_finite z.im && k <> 0.
  && Float.abs (t -. (k *. q)) <= phase_margin *. u wide *. Float.abs t

(* The parts of [k]'s value at [z] whose infinity's sign Annex G leaves open,
   as Prog's Results section lists them, sin and cos through sinh and cosh
   at [iz], and those exp2's rounded phase gives. *)
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

let zero v = v = 0.
let on f = fun w (e : Complex.t array) -> f w e.(0)
let anywhere _ _ = true

let unary ?(covers = []) ?(domain = anywhere) name k b nb =
  {
    name;
    node = Op1 (Unary k, D.Any D.Complex64, 0);
    arity = 1;
    bound = b;
    nx_bound = nb;
    domain;
    covers;
  }

let binary ?(covers = []) ?(domain = anywhere) name k b nb =
  {
    name;
    node = Op2 (Binary k, 0, 1);
    arity = 2;
    bound = b;
    nx_bound = nb;
    domain;
    covers;
  }

(* Smith's division holds its bound where its operands' parts are below half the
   greatest float and the divisor's larger part is normal, so that no
   intermediate overflows or underflows. *)
let smith w (e : Complex.t array) =
  let h = greatest w /. 2. and d = e.(Array.length e - 1) in
  Array.for_all
    (fun (z : Complex.t) -> Float.abs z.re < h && Float.abs z.im < h)
    e
  && Float.max (Float.abs d.re) (Float.abs d.im) >= least w

(* pow holds its bound where |w| |log z| is below the greatest float, so that
   neither product of w log z overflows. *)
let products w (e : Complex.t array) =
  Complex.norm e.(1) *. Complex.norm (Complex.log e.(0)) < greatest w

(* |w log z|, which pow's bound grows with. *)
let wlog (e : Complex.t array) =
  let p = Complex.norm (Complex.mul e.(1) (Complex.log e.(0))) in
  if Float.is_nan p then 0. else p

let cut_real f = on (fun _ z -> zero z.im && f (Float.abs z.re))
let cut_imag f = on (fun _ z -> zero z.re && f (Float.abs z.im))
let real_axis = ("the real axis", on (fun _ z -> zero z.im && z.re <> 0.))
let imag_axis = ("the imaginary axis", on (fun _ z -> zero z.re && z.im <> 0.))

let hyp_far =
  ("far", on (fun _ z -> Float.abs z.re >= 20. && Float.is_finite z.re))

let trig_far =
  ("far", on (fun _ z -> Float.abs z.im >= 20. && Float.is_finite z.im))

let unaries =
  K.
    [
      unary "recip" Recip (fixed 4.) (nx 4.) ~domain:smith
        ~covers:[ ("zero", on (fun _ z -> zero z.re && zero z.im)) ];
      unary "sign" Sign (fixed 3.) (nx 3.)
        ~covers:
          [
            ( "tiny",
              on (fun _ z -> Complex.norm z < 1e-30 && Complex.norm z > 0.) );
          ];
      unary "sqrt" Sqrt (fixed 3.) (nx 3.)
        ~covers:[ ("the cut", cut_real (fun _ -> true)); real_axis ];
      unary "exp" Exp (fixed 31.) (nx 4.)
        ~covers:
          [
            ( "past the overflow of e^x",
              on (fun w z -> (z.re > if w then 709.79 else 88.73) && z.re < 1e4)
            );
            real_axis;
          ];
      unary "exp2" Exp2
        (fun _ e -> 30. +. (1.4 *. Float.abs e.(0).im))
        (nx 4.)
        ~covers:
          [
            ( "past the overflow of 2^x",
              on (fun w z -> (z.re > if w then 1024. else 128.) && z.re < 1e4)
            );
          ];
      unary "log" Log (fixed 18.) (nx 4.)
        ~covers:
          [
            ("the cut", cut_real (fun _ -> true));
            ( "near the unit circle",
              on (fun _ z -> Float.abs (Complex.norm z -. 1.) < 1e-3) );
            ("the pole", on (fun _ z -> zero z.re && zero z.im));
          ];
      unary "log2" Log2 (fixed 20.) (nx 4.)
        ~covers:[ ("the cut", cut_real (fun _ -> true)) ];
      unary "sin" Sin (fixed 67.) (nx 6.) ~covers:[ trig_far; real_axis ];
      unary "cos" Cos (fixed 57.) (nx 6.) ~covers:[ trig_far; imag_axis ];
      unary "tan" Tan (fixed 73.) (nx 6.) ~covers:[ trig_far; real_axis ];
      unary "sinh" Sinh (fixed 67.) (nx 6.) ~covers:[ hyp_far; imag_axis ];
      unary "cosh" Cosh (fixed 57.) (nx 6.) ~covers:[ hyp_far; real_axis ];
      unary "tanh" Tanh (fixed 73.) (nx 6.) ~covers:[ hyp_far; imag_axis ];
      unary "asin" Asin (fixed 23.) (nx 5.)
        ~covers:
          [
            ("a cut", cut_real (fun a -> a > 1.));
            ("a branch point", cut_real (( = ) 1.));
          ];
      unary "acos" Acos (fixed 21.) (nx 5.)
        ~covers:
          [
            ("a cut", cut_real (fun a -> a > 1.));
            ("a branch point", cut_real (( = ) 1.));
          ];
      unary "atan" Atan (fixed 21.) (nx 5.)
        ~covers:
          [
            ("a cut", cut_imag (fun a -> a > 1.));
            ("a pole", cut_imag (( = ) 1.));
          ];
      unary "asinh" Asinh (fixed 23.) (nx 5.)
        ~covers:
          [
            ("a cut", cut_imag (fun a -> a > 1.));
            ("a branch point", cut_imag (( = ) 1.));
          ];
      unary "acosh" Acosh (fixed 21.) (nx 5.)
        ~covers:[ ("the cut", on (fun _ z -> zero z.im && z.re < 1.)) ];
      unary "atanh" Atanh (fixed 21.) (nx 5.)
        ~covers:
          [
            ("a cut", cut_real (fun a -> a > 1.));
            ("a pole", cut_real (( = ) 1.));
          ];
    ]

let binaries =
  K.
    [
      binary "mul" Mul (fixed 2.) (nx 2.)
        ~covers:
          [
            ("an infinite part", fun _ e -> not (finite e.(0) && finite e.(1)));
          ];
      binary "fdiv" Fdiv (fixed 4.) (nx 4.) ~domain:smith
        ~covers:
          [ ("a zero divisor", fun _ e -> zero e.(1).re && zero e.(1).im) ];
      binary "pow" Pow ~domain:products
        (fun _ e -> 31. +. (20. *. wlog e))
        (fun e -> 3. +. (3. *. wlog e))
        ~covers:
          [
            ("a real exponent", fun _ e -> zero e.(1).im && e.(1).re <> 0.);
            ("a zero exponent", fun _ e -> zero e.(1).re && zero e.(1).im);
            ("a zero base", fun _ e -> zero e.(0).re && zero e.(0).im);
          ];
    ]

(* Cases *)

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
let examples k =
  let ts = Array.of_list tuples in
  let z wide i = rounded wide ts.(i mod Array.length ts) in
  let element wide i = Array.init k.arity (fun j -> z wide (i + j)) in
  List.map (fun wide -> (wide, Array.init n (element wide))) [ false; true ]

let hex wide v =
  if wide then Printf.sprintf "%Lx" (Int64.bits_of_float v)
  else Printf.sprintf "%lx" (Int32.bits_of_float v)

let show wide (z : Complex.t) =
  Printf.sprintf "(%s %s)" (hex wide z.re) (hex wide z.im)

let eval (type s) (dt : (Complex.t, s) D.t) k es =
  let xs =
    Array.init k.arity (fun i ->
        Nx.create dt [| n |] (Array.map (fun e -> e.(i)) es))
  in
  let ins = Iarray.of_array (Array.make k.arity (D.Any dt)) in
  let node : K.node =
    match k.node with Op1 (u, _, j) -> Op1 (u, D.Any dt, j) | other -> other
  in
  let prog = K.of_node ~ins node in
  let layout = Nx_array.Layout.contiguous [| n |] in
  let loads = Iarray.of_array (Array.map (fun x -> P.Plain x) xs) in
  fun () ->
    fst
      (P.eval ~by:"t"
         (P.Map { layout; prog; outs = [ dt ]; loads; at = None }))

(* Whether [got] meets the reference [r] at operands [e]: where every operand
   part is finite and in the kind's domain, a NaN part where [r] has one, and
   where [r] is finite no NaN part and, where [r]'s modulus is a normal of the
   dtype, within [tol] u normwise, an infinite part only where [tol] u reaches
   the greatest float; where a unary kind's operand has an infinite part, the
   classes of C99 Annex G's values, nx's, the signs of infinities included
   but those {!open_signs} leaves open; where an operand has a NaN part and
   none an infinite one, a NaN part, but for z^0, which is 1. *)
let meets wide k tol e (r : Complex.t) (got : Complex.t) =
  if Array.for_all finite e then
    let m = Complex.norm r in
    if not (k.domain wide e) then true
    else if has_nan r then has_nan got
    else if not (finite r) then true
    else
      (not (has_nan got))
      && (m < least wide
         || m > greatest wide
         || ((not (finite got)) && m *. (1. +. (tol *. u wide)) >= greatest wide)
         || Complex.norm (Complex.sub got r) /. m <= tol *. u wide)
  else if k.arity = 1 && has_inf e.(0) then
    let signs = open_signs wide k e.(0) in
    classes signs got = classes signs r
  else if Array.exists has_nan e && not (Array.exists has_inf e) then
    has_nan got || (k.name = "pow" && zero e.(1).re && zero e.(1).im)
  else true

let check k (wide, es) =
  let reference = Nx.to_array (eval D.Complex128 k es ()) in
  let got =
    if wide then Lowered.run (eval D.Complex128 k es)
    else Lowered.run (eval D.Complex64 k es)
  in
  let bad = ref [] in
  Array.iteri
    (fun i r ->
      let e = es.(i) in
      let tol = k.bound wide e +. if wide then k.nx_bound e else 0. in
      if (not (meets wide k tol e r got.(i))) && List.length !bad < 5 then
        let err =
          Complex.norm (Complex.sub got.(i) r) /. Complex.norm r /. u wide
        in
        bad :=
          Printf.sprintf "%s%s = %s, reference %s, %g u over %g" k.name
            (String.concat "" (Array.to_list (Array.map (show wide) e)))
            (show wide got.(i))
            (show true r) err tol
          :: !bad)
    reference;
  let cover_some label f = cover label (Array.exists (f wide) es) in
  cover "complex64" (not wide);
  cover "complex128" wide;
  cover_some "a NaN part" (fun _ e -> Array.exists has_nan e);
  cover_some "an infinite part" (fun _ e ->
      Array.exists
        (fun (z : Complex.t) ->
          Float.abs z.re = Float.infinity || Float.abs z.im = Float.infinity)
        e);
  cover_some "two non-finite parts" (fun _ e ->
      Array.exists
        (fun (z : Complex.t) ->
          (not (Float.is_finite z.re)) && not (Float.is_finite z.im))
        e);
  cover_some "a -0 part" (fun _ e ->
      Array.exists
        (fun (z : Complex.t) ->
          (Float.sign_bit z.re && zero z.re)
          || (Float.sign_bit z.im && zero z.im))
        e);
  cover_some "a huge number" (fun w e ->
      Array.exists
        (fun z -> finite z && Complex.norm z > if w then 1e300 else 1e30)
        e);
  List.iter (fun (label, f) -> cover_some label f) k.covers;
  equal (list string) [] (List.rev !bad)

(* Dividends whose parts are subnormal, over divisors whose larger part is
   between the least normal and 8 times it: the quotients are normal, and a
   bit a subnormal intermediate drops is magnified into them. *)
let subnormal_dividends wide =
  let st = Random.State.make [| 3 |] in
  let lo = least wide in
  let draw scale =
    let v = round wide (Random.State.float st scale) in
    if Random.State.bool st then v else -.v
  in
  let element _ =
    let a = { Complex.re = draw lo; im = draw lo } in
    let re = round wide (lo *. (1. +. Random.State.float st 7.)) in
    [| a; { Complex.re; im = draw (8. *. lo) } |]
  in
  (wide, Array.init n element)

let examples k =
  if k.name <> "fdiv" then examples k
  else examples k @ List.map subnormal_dividends [ false; true ]

let prop_of k = prop ~count:20 ~examples:(examples k) k.name (case k) (check k)

let () =
  exit
    (run "Lower_complex"
       [
         group "unary" (List.map prop_of unaries);
         group "binary" (List.map prop_of binaries);
       ])
