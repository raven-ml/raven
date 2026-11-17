(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The complex kinds at complex64 and complex128, as maps of one kind compiled
   by Rune.jit, which must compile them whole: no call is evaluated. A kind
   compiled to nx's own sequence gives eager's bits (Kinds), over drawn operands
   and of a constant inside the compiled function; tolk's folding gives a NaN it
   generates a clear sign bit, where the host's arithmetic sets it, so a folded
   pow may give a zero part the other sign where it copies one (Kinds.law's
   nan_signs). Every function holds rune.mli's rule for a NaN part, from the
   spec. Each other kind holds its normwise bound (rune.mli, Numerics) of the
   exact value at the points of golden/lower_complex, which gen/lower_complex.py
   writes from mpmath, each part correctly rounded with its remainder. At an
   operand with an infinite or a NaN part, each gives what lower_complex.mli's
   special values state, the sorts of nx's eager complex128 kind. *)

open Windtrap
module Kinds = Rune_support.Kinds
module D = Nx_array.Dtype
module K = Nx_kernel.Prog
module P = Nx.Prim
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

(* The parts of [k]'s value at [z] whose infinity's sign Annex G leaves open, as
   Prog's Results section lists them. *)
let open_signs k (z : Complex.t) =
  let inf v = Float.abs v = Float.infinity in
  match k.name with
  | "asin" | "acos" -> (false, inf z.re && Float.is_nan z.im)
  | "asinh" -> (Float.is_nan z.re && inf z.im, false)
  | _ -> (false, false)

let anywhere _ _ = true

let unary ?(domain = anywhere) name k bound =
  { name; node = Op1 (Unary k, D.Any D.Complex64, 0); arity = 1; bound; domain }

let tiny w v = v <> 0. && Float.abs v < least w

(* Each kind's bound (rune.mli, Numerics). *)
let unaries =
  K.
    [
      unary "asin" Asin (fixed 23.);
      unary "acos" Acos (fixed 21.);
      unary "atan" Atan (fixed 21.);
      unary "asinh" Asinh (fixed 23.);
      unary "acosh" Acosh (fixed 21.);
      unary "atanh" Atanh (fixed 21.);
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

(* Whether [got] meets nx's value [r] at operands [e] with a part not finite:
   where a unary kind's operand has an infinite part, each part's sort as [r]'s,
   the signs {!open_signs} leaves open excepted; where an operand has a NaN part
   and none an infinite one, a NaN part. *)
let special k e (r : Complex.t) (got : Complex.t) =
  if k.arity = 1 && has_inf e.(0) then
    let signs = open_signs k e.(0) in
    sorts signs got = sorts signs r
  else if Array.exists has_nan e && not (Array.exists has_inf e) then
    has_nan got
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
        if Array.for_all finite e || special k e r.(i) got.(i) then None
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

(* Kinds of nx's sequence

   Each gives eager's bits over drawn operands (Kinds) and of a constant inside
   the compiled function. *)
let ported k =
  let name = Kinds.name k in
  [
    prop ~count:12
      ~examples:[ Kinds.example k ]
      (name ^ " gives eager's bits")
      (Kinds.gen k)
      (Kinds.law (module Nx.Host) ~f64:true k);
    test (name ^ " of a constant gives eager's bits") (fun () ->
        Kinds.folded ~nan_signs:true (module Nx.Host) ~f64:true k);
  ]

let kinds =
  Kinds.
    [
      cmul;
      cdiv;
      crecip;
      csqrt;
      csign;
      clog;
      clog2;
      cexp;
      cexp2;
      cpow;
      csinh;
      ccosh;
      ctanh;
      csin;
      ccos;
      ctan;
    ]

(* NaN parts

   rune.mli's rule, from the spec: an operand with a NaN part and none infinite
   gives a result with a NaN part, but z^w is 1 at w = 0, and 1 ± 0i at z = 1 ±
   0i with w's imaginary part zero, where the float pow is 1 at x = 1. Each
   function compiled on the host at complex64 and complex128, over numbers of
   the parts NaN, -NaN, ±0, ±1 and 2 with a NaN part among the operands. *)

type fn =
  | F1 of { f1 : 's 'd. (Complex.t, 's, 'd) Nx.t -> (Complex.t, 's, 'd) Nx.t }
  | F2 of {
      f2 :
        's 'd.
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t;
    }

let fns =
  [
    ("mul", F2 { f2 = Nx.mul });
    ("div", F2 { f2 = Nx.div });
    ("pow", F2 { f2 = Nx.pow });
    ("recip", F1 { f1 = Nx.recip });
    ("sqrt", F1 { f1 = Nx.sqrt });
    ("sign", F1 { f1 = Nx.sign });
    ("log", F1 { f1 = Nx.log });
    ("log2", F1 { f1 = Nx.log2 });
    ("exp", F1 { f1 = Nx.exp });
    ("exp2", F1 { f1 = Nx.exp2 });
    ("sinh", F1 { f1 = Nx.sinh });
    ("cosh", F1 { f1 = Nx.cosh });
    ("tanh", F1 { f1 = Nx.tanh });
    ("sin", F1 { f1 = Nx.sin });
    ("cos", F1 { f1 = Nx.cos });
    ("tan", F1 { f1 = Nx.tan });
    ("asin", F1 { f1 = Nx.asin });
    ("acos", F1 { f1 = Nx.acos });
    ("atan", F1 { f1 = Nx.atan });
    ("asinh", F1 { f1 = Nx.asinh });
    ("acosh", F1 { f1 = Nx.acosh });
    ("atanh", F1 { f1 = Nx.atanh });
  ]

let numbers =
  let ps = [ Float.nan; -.Float.nan; 0.; -0.; 1.; -1.; 2. ] in
  List.concat_map (fun re -> List.map (fun im -> { Complex.re; im }) ps) ps

let has_nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im

(* The rule's value at operands [zs]: [`One] where it is 1 + ±0i, else a NaN
   part. *)
let expected name zs =
  match (name, zs) with
  | "pow", [ _; (w : Complex.t) ] when w.re = 0. && w.im = 0. -> `One
  | "pow", [ (z : Complex.t); (w : Complex.t) ]
    when z.re = 1. && z.im = 0. && w.im = 0. ->
      `One
  | _ -> `Nan

let nan_parts (name, f) =
  let tuples =
    match f with
    | F1 _ -> List.map (fun z -> [ z ]) numbers
    | F2 _ ->
        List.concat_map (fun z -> List.map (fun w -> [ z; w ]) numbers) numbers
  in
  let tuples = List.filter (List.exists has_nan) tuples in
  let at (type s) (dt : (Complex.t, s) Nx.dtype) =
    let col j =
      Nx.create dt
        [| List.length tuples |]
        (Array.of_list (List.map (fun t -> List.nth t j) tuples))
    in
    let before = Rune.evaluations () in
    let y =
      match f with
      | F1 { f1 } -> Rune.jit Nx.Ptree.(tensor @-> returns tensor) f1 (col 0)
      | F2 { f2 } ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            f2 (col 0) (col 1)
    in
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
    let show (z : Complex.t) = Printf.sprintf "(%h, %h)" z.re z.im in
    List.concat
      (List.mapi
         (fun i (r : Complex.t) ->
           let zs = List.nth tuples i in
           let ok =
             match expected name zs with
             | `One -> r.re = 1. && r.im = 0.
             | `Nan -> has_nan r
           in
           if ok then []
           else
             [
               Printf.sprintf "%s(%s) at %s = %s" name
                 (String.concat ", " (List.map show zs))
                 (Nx.Dtype.name dt) (show r);
             ])
         (Array.to_list (Nx.to_array y)))
  in
  equal (list string) [] (at Nx.complex64 @ at Nx.complex128)

let () =
  exit
    (run "Lower_complex"
       [
         group "kinds" (List.concat_map ported kinds);
         Windtrap.cases ~id:fst
           "a NaN part and no infinite one give a NaN part, but pow's \
            exceptions"
           fns nan_parts;
         group "special values" (List.map law unaries);
         group "bounds" (List.map accurate unaries);
       ])
