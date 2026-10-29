(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.grad, Rune.vjp and their forms: the adjoint law against forward mode for
   every operation, real and complex in the unconjugated pairing, and the
   gradient as the pullback through the pairing; reverse mode at kinks against
   forward mode; the conventions' values; the interface and its errors;
   differentiation on two domains; grad of jit and jit of grad; the values a
   tape keeps. *)

open Windtrap
open Rune2_support
open Cases
open Witness
module Rune = Rune2

let numel = Operand.numel

(* A cotangent in [-1, 1] for [n] elements, spread from [seed]. *)
let spread seed n =
  Array.init n (fun k -> Float.sin ((Float.of_int k *. 2.3) +. seed))

let sum_abs a = Array.fold_left (fun s x -> s +. Float.abs x) 0. a
let finite a = Array.for_all Float.is_finite a

(* [<w, J v> = <Jᵀ w, v>], from the terms of each side, within [2^-40] of their
   magnitudes and of [size], the magnitude of the products of operands that [J
   v] may cancel: float64's rounding of a few hundred terms. *)
let adjoint ~msg ~size wjv jtwv =
  if finite wjv && finite jtwv then
    let lhs = Array.fold_left ( +. ) 0. wjv
    and rhs = Array.fold_left ( +. ) 0. jtwv in
    let scale = sum_abs wjv +. sum_abs jtwv +. size in
    at_most ~msg float_exact
      ~than:((0x1p-40 *. scale) +. 0x1p-1000)
      (Float.abs (lhs -. rhs))

(* Whether a derivative is a value of its own: C order from the start of its
   memory. *)
let fresh ~msg x =
  let layout = (Nx.Prim.form x).layout in
  equal ~msg (pair bool int) (true, 0)
    (L.is_contiguous layout, fst (L.span layout))

(* The adjoint law *)

(* The pullback of [w] pairs with a tangent as [w] pairs with the tangent's
   image, and so does the gradient of [<w, f x>], the pullback of one through
   the pairing. *)
let adjoint_law c (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let y, jv = Rune.jvp' c.f x t in
  let y', pullback = Rune.vjp' c.f x in
  equal ~msg:"primal" bits_t (bits y) (bits y');
  let seed = Array.fold_left ( +. ) 0.5 ts in
  let w = Nx.create f64 (Nx.shape y) (spread seed (Nx.numel y)) in
  let jtw = pullback w in
  equal ~msg:"cotangent shape" (array int) s (Nx.shape jtw);
  fresh ~msg:"the pullback's result is fresh" jtw;
  let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. xs in
  let size = sum_abs (Nx.to_array w) *. sum_abs ts *. ((1. +. big) ** 2.) in
  let wjv = Nx.to_array (Nx.mul w jv) in
  adjoint ~msg:"<w, J v> = <Jᵀ w, v>" ~size wjv (Nx.to_array (Nx.mul jtw t));
  let g = Rune.grad' (fun x -> Nx.sum (Nx.mul w (c.f x))) x in
  fresh ~msg:"the gradient is fresh" g;
  adjoint ~msg:"<w, J v> = <grad <w, f>, v>" ~size wjv
    (Nx.to_array (Nx.mul g t))

(* On complex values, [Re Σ w (J v) = Re Σ (Jᵀ w) v], unconjugated, and the
   gradient of [Re <w, f z>], [∂/∂x - i ∂/∂y], pairs with [v] alike. *)
let complex_law c (input, f) (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let z = input (Nx.create f64 s xs) and t = input (Nx.create f64 s ts) in
  let y, jv = Rune.jvp' f z t in
  let _, pullback = Rune.vjp' f z in
  let seed = Array.fold_left ( +. ) 0.5 ts in
  let n = Nx.numel y in
  let w =
    Nx.complex c128
      ~re:(Nx.create f64 (Nx.shape y) (spread seed n))
      ~im:(Nx.create f64 (Nx.shape y) (spread (seed +. 1.) n))
  in
  let jtw = pullback w in
  let re x = Nx.to_array (Nx.real f64 x) in
  let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. xs in
  let size =
    (sum_abs (re w) +. sum_abs (Nx.to_array (Nx.imag f64 w)))
    *. sum_abs ts
    *. ((1. +. big) ** 2.)
  in
  let wjv = re (Nx.mul w jv) in
  adjoint ~msg:"Re <w, J v> = Re <Jᵀ w, v>" ~size wjv (re (Nx.mul jtw t));
  let g = Rune.grad' (fun z -> Nx.sum (Nx.real f64 (Nx.mul w (f z)))) z in
  adjoint ~msg:"Re <w, J v> = Re <grad Re <w, f>, v>" ~size wjv
    (re (Nx.mul g t))

let adjoint_props =
  List.map
    (fun c -> prop c.name (gen c) (adjoint_law c))
    (real_cases @ contract_cases @ factor_cases @ complex_cases
   @ complex_factor_cases)

let complex_props =
  List.filter_map
    (fun c -> Option.map (fun z -> prop c.name (gen c) (complex_law c z)) c.z)
    (complex_cases @ complex_factor_cases)

(* Kinks

   At ties and zeros, where an operation has no derivative, reverse mode takes
   the transposes of the tangents the Kinks rules give, so [jacrev'] is
   [jacfwd'] element by element, within the rounding of the two modes'
   arithmetic. The cases are those whose other laws draw values spaced away from
   their kinks. *)

let ties xs =
  let n = Array.length xs in
  let rec go i j =
    if i >= n then false
    else if j >= n then go (i + 1) (i + 2)
    else xs.(i) = xs.(j) || go i (j + 1)
  in
  go 0 1

let kink_law c (s, xs, _) =
  cover "operands that tie" (ties xs);
  cover "a zero" (Array.exists (fun x -> x = 0.) xs);
  let x = Nx.create f64 s xs in
  let fwd = Rune.jacfwd' c.f x and rev = Rune.jacrev' c.f x in
  equal ~msg:"shape" (array int) (Nx.shape fwd) (Nx.shape rev);
  equal ~msg:"jacrev' is jacfwd'"
    (array (float 1e-12))
    (Nx.to_array fwd) (Nx.to_array rev)

let kink_props =
  List.filter_map
    (fun c ->
      match c.values with
      | Spaced _ ->
          Some (prop c.name (gen { c with values = Kinks }) (kink_law c))
      | Uniform _ | Zeros _ | Kinks -> None)
    real_cases

(* Finite gradients

   A composed function selects among arms by [where], and an arm may be
   singular where it is not selected; its zero cotangent must not turn the
   gradient NaN there. Over every tuple of a grid of each function's
   boundaries, wherever forward mode's tangent along an operand is finite at
   operands other than NaN, so is the gradient along it; the adjoint law
   checks its value on each domain. lbeta and the incomplete gamma functions
   and their inverses are left out: their gradients are still NaN at some of
   these points. *)

type 's ft = (float, 's, Nx.host) Nx.t

type composed =
  | One of { fname : string; f1 : 's. 's ft -> 's ft }
  | Two of { fname : string; f2 : 's. 's ft -> 's ft -> 's ft }
  | Three of { fname : string; f3 : 's. 's ft -> 's ft -> 's ft -> 's ft }

let fname = function One c -> c.fname | Two c -> c.fname | Three c -> c.fname
let arity = function One _ -> 1 | Two _ -> 2 | Three _ -> 3

let apply (type s) c (rows : s ft list) : s ft =
  match (c, rows) with
  | One c, [ x ] -> c.f1 x
  | Two c, [ x; y ] -> c.f2 x y
  | Three c, [ x; y; z ] -> c.f3 x y z
  | _ -> invalid_arg "apply"

let composed =
  [
    One { fname = "rsqrt"; f1 = Nx.rsqrt };
    One { fname = "asinh"; f1 = Nx.asinh };
    One { fname = "acosh"; f1 = Nx.acosh };
    One { fname = "atanh"; f1 = Nx.atanh };
    One { fname = "erfc"; f1 = Nx.erfc };
    One { fname = "erfinv"; f1 = Nx.erfinv };
    One { fname = "ndtr"; f1 = Nx.ndtr };
    One { fname = "log_ndtr"; f1 = Nx.log_ndtr };
    One { fname = "ndtri"; f1 = Nx.ndtri };
    One { fname = "i0e"; f1 = Nx.i0e };
    One { fname = "i1e"; f1 = Nx.i1e };
    One { fname = "sigmoid"; f1 = Nx.sigmoid };
    One { fname = "softplus"; f1 = Nx.softplus };
    One { fname = "lgamma"; f1 = Nx.lgamma };
    One { fname = "digamma"; f1 = Nx.digamma };
    Two { fname = "hypot"; f2 = Nx.hypot };
    Two { fname = "atan2"; f2 = Nx.atan2 };
    Three { fname = "lerp"; f3 = Nx.lerp };
    Three { fname = "betainc"; f3 = Nx.betainc };
    Three { fname = "betaincc"; f3 = Nx.betaincc };
    Three { fname = "log_betainc"; f3 = Nx.log_betainc };
    Three { fname = "log_betaincc"; f3 = Nx.log_betaincc };
  ]

(* A float format's precision and greatest exponent. *)
type format = { p : int; emax : int }

let f64_format = { p = 53; emax = 1023 }
let f32_format = { p = 24; emax = 127 }
let f16_format = { p = 11; emax = 15 }
let bf16_format = { p = 8; emax = 127 }

(* Zeros, the branch boundaries of the composed forms and their neighbours in
   the format, its smallest subnormal and normal, its largest finite value,
   the infinities and NaN; a coarser grid for three operands. *)
let boundaries f =
  let eps = Float.ldexp 1. (1 - f.p) in
  let near b = [ b; b *. (1. -. (eps /. 2.)); b *. (1. +. eps) ] in
  let pos =
    [
      0.; Float.ldexp 1. (2 - f.emax - f.p); Float.ldexp 1. (1 - f.emax);
      1e-8; 0.25; 1.5; 3.; 10.; 30.; Float.ldexp (2. -. eps) f.emax;
      Float.ldexp 1. (f.emax / 2); Float.infinity;
    ]
    @ List.concat_map near [ 0x1p-28; 0.5; 1.; 2.; 0x1p28 ]
  in
  (Float.nan :: pos) @ List.map Float.neg pos

let coarse =
  [
    0.; -0.; 0x1p-28; 0.5; 1.; 2.; 30.; 0x1p28; Float.infinity; -1.;
    Float.neg_infinity; Float.nan;
  ]

(* Every [k]-tuple of [vs], as [k] rows. *)
let grid k vs =
  let rec tuples k =
    if k = 0 then [ [] ]
    else
      List.concat_map (fun t -> List.map (fun v -> v :: t) vs) (tuples (k - 1))
  in
  let ts = Array.of_list (tuples k) in
  let n = Array.length ts in
  Nx.create f64 [| k; n |]
    (Array.init (k * n) (fun i -> List.nth ts.(i mod n) (i / n)))

let finite_law (type s) ((w : (float, s) D.t), fmt) c =
  let k = arity c in
  let x = grid k (if k = 3 then coarse else boundaries fmt) in
  let n = (Nx.shape x).(1) in
  let xs = Nx.to_array x in
  let operand q i = xs.((q * n) + i) in
  let f x =
    let x = Nx.cast w x in
    Nx.cast f64 (apply c (List.init k (fun r -> Nx.get [ r ] x)))
  in
  let g = Nx.to_array (Rune.grad' (fun x -> Nx.sum (f x)) x) in
  let bad = ref [] in
  for r = 0 to k - 1 do
    let along = Array.init (k * n) (fun i -> if i / n = r then 1. else 0.) in
    let t = Nx.to_array (snd (Rune.jvp' f x (Nx.create f64 [| k; n |] along))) in
    for i = 0 to n - 1 do
      let ops = List.init k (fun q -> operand q i) in
      let gi = g.((r * n) + i) in
      let nan = List.exists Float.is_nan ops in
      if (not nan) && Float.is_finite t.(i) && not (Float.is_finite gi) then
        let ops = String.concat ", " (List.map (Printf.sprintf "%h") ops) in
        bad :=
          Printf.sprintf "operand %d at (%s): tangent %h, gradient %h" r ops
            t.(i) gi
          :: !bad
    done
  done;
  equal ~msg:"gradients not finite where the tangent is" (list string) []
    (List.rev !bad)

(* [pinv] drops a singular value or eigenvalue below its cutoff; the gradient
   at a matrix with one must stay finite wherever forward mode's tangent
   along an element is. *)
let pinv_law hermitian () =
  let a = Nx.create f64 [| 2; 2 |] [| 1.; 0.; 0.; 0. |] in
  let f a = Nx.pinv ~hermitian a in
  let g = Nx.to_array (Rune.grad' (fun a -> Nx.sum (f a)) a) in
  let bad = ref [] in
  for i = 0 to 3 do
    let e = Array.init 4 (fun j -> if i = j then 1. else 0.) in
    let t = Nx.to_array (snd (Rune.jvp' f a (Nx.create f64 [| 2; 2 |] e))) in
    let tangent = Array.fold_left ( +. ) 0. t in
    if Float.is_finite tangent && not (Float.is_finite g.(i)) then
      bad :=
        Printf.sprintf "element %d: tangent %h, gradient %h" i tangent g.(i)
        :: !bad
  done;
  equal ~msg:"gradients not finite where the tangent is" (list string) []
    (List.rev !bad);
  (* Of pinv at fixed rank, [d sum (pinv a) = -da00 + da01 + da10] here; under
     [~hermitian] [a] is its lower triangle, so [da01] is [da10]. *)
  let want = if hermitian then [| -1.; 0.; 2.; 0. |] else [| -1.; 1.; 1.; 0. |] in
  equal ~msg:"the fixed-rank gradient" (array (float 1e-12)) want g

let finite_cases =
  [
    test "pinv of a singular matrix" (pinv_law false);
    test "pinv of a singular Hermitian matrix" (pinv_law true);
    cases "float64" ~name:fname composed (finite_law (f64, f64_format));
    cases "float32" ~name:fname composed (finite_law (Nx.float32, f32_format));
    cases "float16" ~name:fname composed (finite_law (Nx.float16, f16_format));
    cases "bfloat16" ~name:fname composed
      (finite_law (Nx.bfloat16, bf16_format));
  ]

(* Values the conventions state *)

let floats = array (float 1e-12)
let signed = array (pair bool float_exact)
let with_signs a = Array.map (fun x -> (Float.sign_bit x, x)) (Nx.to_array a)
let z re im = Nx.create c128 [| 1 |] [| { Complex.re; im } |]
let pairs a = Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array a)
let close_pairs = array (pair (float 1e-12) (float 1e-12))
let both (a, b) = Array.append (Nx.to_array a) (Nx.to_array b)
let two () = T.(pair tensor tensor)

(* The gradient of a real [f] of [z = x + iy] is [∂f/∂x - i ∂f/∂y], and a
   pullback applies [f'(z)] to its cotangent. *)
let test_complex () =
  let z0 = z 1. 2. in
  let g f = pairs (Rune.grad' f z0) in
  let re x = Nx.sum (Nx.real f64 x) in
  equal ~msg:"|z|²" close_pairs
    [| (2., -4.) |]
    (g (fun z -> re (Nx.mul z (Nx.conjugate z))));
  let c = Nx.complex c128 ~re:(v [| 3. |]) ~im:(v [| 4. |]) in
  equal ~msg:"Re (c z)" close_pairs
    [| (3., 4.) |]
    (g (fun z -> re (Nx.mul z c)));
  equal ~msg:"Im z" close_pairs
    [| (0., -1.) |]
    (g (fun z -> Nx.sum (Nx.imag f64 z)));
  let y, pullback = Rune.vjp' Nx.exp z0 in
  let w = z 0.5 (-1.) in
  equal ~msg:"exp's pullback" close_pairs
    (pairs (Nx.mul y w))
    (pairs (pullback w))

let test_kinks () =
  equal ~msg:"abs at both zeros" floats [| 1.; 1.; -1. |]
    (Nx.to_array
       (Rune.grad' (fun x -> Nx.sum (Nx.abs x)) (v [| 0.; -0.; -2. |])));
  let maximum (a, b) = Nx.sum (Nx.maximum a b) in
  equal ~msg:"maximum's ties" floats [| 0.5; 1.; 0.5; 0. |]
    (both (Rune.grad (two ()) maximum (v [| 1.; 2. |], v [| 1.; 0. |])));
  equal ~msg:"maximum beside a NaN" floats [| 0.; 0.; 0.; 0. |]
    (both
       (Rune.grad (two ()) maximum
          (v [| Float.nan; 1. |], v [| 1.; Float.nan |])));
  equal ~msg:"an infinite logsumexp" floats [| 0.5; 0.5; 0. |]
    (Nx.to_array
       (Rune.grad'
          (fun x -> Nx.sum (Nx.logsumexp x))
          (v [| Float.infinity; Float.infinity; 1. |])));
  let r = Float.sqrt 0.5 in
  equal ~msg:"hypot where its operands' magnitudes tie" floats
    [| r; -.r; r; r |]
    (both
       (Rune.grad (two ())
          (fun (a, b) -> Nx.sum (Nx.hypot a b))
          (v [| 1.; -2. |], v [| 1.; 2. |])));
  equal ~msg:"pow along a zero exponent and a zero base" floats
    [| 0.; 0.; 0.; 0.; 0.; Float.log 2. |]
    (both
       (Rune.grad (two ())
          (fun (a, b) -> Nx.sum (Nx.pow a b))
          (v [| 0.; 0.; 2. |], v [| 2.; 0.; 0. |])));
  (* More ties than a narrow float counts: 65536 float16 ones each take 2^-16,
     and 257 bfloat16 ones 1/257 rounded, 255 2^-16. *)
  let ends dt n =
    let g = Rune.grad' Nx.max (Nx.ones dt [| n |]) in
    Nx.to_array (Nx.cast Nx.float64 (Nx.stack [ Nx.min g; Nx.max g ]))
  in
  equal ~msg:"max's 65536 ties, float16" floats [| 0x1p-16; 0x1p-16 |]
    (ends Nx.float16 65536);
  equal ~msg:"max's 257 ties, bfloat16" floats
    [| 255. *. 0x1p-16; 255. *. 0x1p-16 |]
    (ends Nx.bfloat16 257);
  (* A mean of moments divides its cotangent by more terms than a narrow float
     counts, as the running mean at the last of as many, under reverse mode and
     as a linear function's transpose: 65536 float16 terms each take 2^-16, and
     257 bfloat16 ones 1/257 rounded, 255 2^-16. *)
  let moments ?(all = false) ~scan ~transpose dt n =
    let any = D.Any dt in
    let prog = P.of_node ~ins:[| any |] (Op1 (Copy, any, 0)) in
    let layout = L.contiguous (Iarray.of_list [ 1; n ]) in
    let mean x =
      let x = Nx.reshape [| 1; n |] x in
      if scan then
        fst
          (Nx.Prim.eval ~by:"test"
             (Scan
                {
                  layout;
                  axis = 1;
                  prog;
                  reduction = Moments (0, dt);
                  loads = [| Plain x |];
                  carry = None;
                }))
      else
        let (m, _), () =
          Nx.Prim.eval ~by:"test"
            (Reduce
               {
                 layout;
                 axes = [| 1 |];
                 prog;
                 reductions = [ Moments (0, dt) ];
                 loads = [| Plain x |];
               })
        in
        m
    in
    (* The cotangent of the means: each 1 / n where [all], else 1 at the
       last. *)
    let ct m =
      if all then Nx.full_like m (D.of_float dt (1. /. Float.of_int n))
      else if scan then
        let zero = D.of_float dt 0. in
        Nx.pad [| (0, 0); (n - 1, 0) |] zero (Nx.ones dt [| 1; 1 |])
      else Nx.ones_like m
    in
    let x = Nx.ones dt [| n |] in
    let g =
      if transpose then
        Rune.linear_transpose T.tensor T.tensor mean x (ct (mean x))
      else Rune.grad' (fun x -> let m = mean x in Nx.sum (Nx.mul m (ct m))) x
    in
    Nx.to_array (Nx.cast Nx.float64 (Nx.stack [ Nx.min g; Nx.max g ]))
  in
  List.iter
    (fun (scan, transpose) ->
      let what =
        (if scan then "a running mean" else "a mean")
        ^ if transpose then ", transposed" else ""
      in
      equal ~msg:(what ^ " of 65536, float16") floats [| 0x1p-16; 0x1p-16 |]
        (moments ~scan ~transpose Nx.float16 65536);
      equal ~msg:(what ^ " of 257, bfloat16") floats
        [| 255. *. 0x1p-16; 255. *. 0x1p-16 |]
        (moments ~scan ~transpose Nx.bfloat16 257))
    [ (false, false); (true, false); (false, true); (true, true) ];
  (* The mean of 65536 float16 running means: element 0's gradient is 2^-16
     H(65536) rounded once, 0x1.754p-13, the last's 2^-32, which rounds to 0.
     That lies 0.05 float16 ulp below a tie, and the sum may take its terms in
     any order: within one float16 ulp, 2^-23. *)
  List.iter
    (fun transpose ->
      equal ~msg:"the mean of running means, float16"
        (array (float 0x1p-23))
        [| 0.; 0x1.754p-13 |]
        (moments ~all:true ~scan:true ~transpose Nx.float16 65536))
    [ false; true ]

(* The operand [Nx.where] does not select takes [+0.], so an infinite derivative
   there makes the gradient NaN; an operand computed on its domain, or from a
   detached input, does not. *)
let test_selections () =
  let at0 = v [| 0. |] in
  let picked x = Nx.greater_s x (-1.) and positive x = Nx.greater_s x 0. in
  equal ~msg:"the unselected operand's cotangent is +0." signed
    [| (true, -0.) |]
    (with_signs
       (snd
          (Rune.grad (two ())
             (fun (x, y) -> Nx.sum (Nx.where (picked x) x (Nx.neg y)))
             (at0, v [| 3. |]))));
  equal ~msg:"an unselected log at 0." bits_t
    (bits (v [| Float.nan |]))
    (bits (Rune.grad' (fun x -> Nx.sum (Nx.where (picked x) x (Nx.log x))) at0));
  let on_domain other x =
    Nx.sum
      (Nx.where (positive x) (Nx.log (Nx.where (positive x) x (other x))) x)
  in
  equal ~msg:"a log computed on its domain" floats [| 1. |]
    (Nx.to_array (Rune.grad' (on_domain Nx.ones_like) at0));
  equal ~msg:"a log of a detached input" floats [| 1. |]
    (Nx.to_array (Rune.grad' (on_domain Rune.detach) at0))

(* A single contribution keeps a zero's sign through maps, reductions, pads,
   windows and slices; through a gather or a broadcast it is added to [+0.]. An
   element that receives none is [+0.]. Each function's cotangent is [-0.] at
   every element of [op x]. *)
let test_zeros () =
  let g op xs =
    let f x =
      let y = op x in
      Nx.sum (Nx.mul y (Nx.full_like y (-0.)))
    in
    with_signs (Rune.grad' f (v xs))
  in
  equal ~msg:"a square at -0." signed
    [| (true, -0.) |]
    (with_signs (Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) (v [| -0. |])));
  equal ~msg:"a map and a sum" signed [| (true, -0.) |] (g Nx.sin [| 0. |]);
  equal ~msg:"abs of a negative" signed [| (false, 0.) |] (g Nx.abs [| -1. |]);
  equal ~msg:"abs of a negative, from +0." signed
    [| (true, -0.) |]
    (with_signs
       (Rune.grad'
          (fun x -> Nx.sum (Nx.mul (Nx.abs x) (Nx.zeros_like x)))
          (v [| -1. |])));
  equal ~msg:"a pad" signed
    [| (true, -0.) |]
    (g (Nx.pad [| (1, 2) |] 3.) [| 1. |]);
  equal ~msg:"windows" signed
    [| (true, -0.); (true, -0.) |]
    (g
       (fun x ->
         Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 2 |]
           ~dilation:[| 1 |]
           ~padding:[| (0, 0) |]
           (Nx.reshape [| 1; 2 |] x))
       [| 1.; 2. |]);
  equal ~msg:"a slice" signed
    [| (true, -0.); (false, 0.) |]
    (g (Nx.slice [ R (0, 1) ]) [| 1.; 2. |]);
  (* A running sum's carry receives one contribution from each element; of one
     element, that one. *)
  equal ~msg:"a running sum's carry" signed
    [| (true, -0.) |]
    (g
       (fun c ->
         scan1
           ~carry:(Nx.reshape [| 1; 1 |] c)
           (Monoid (Sum, 0, f64))
           (Nx.reshape [| 1; 1 |] (v [| 2. |])))
       [| 1. |]);
  equal ~msg:"a gather" signed
    [| (false, 0.); (false, 0.) |]
    (g (Nx.take (Nx.create Nx.int64 [| 1 |] [| 0L |])) [| 1.; 2. |]);
  equal ~msg:"a broadcast's copies" signed
    [| (false, 0.) |]
    (g (Nx.broadcast_to [| 2 |]) [| 1. |])

(* Interface *)

(* An integer parameter's gradient, and an unused one's, are zero. *)
let test_zero_gradients () =
  let n = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
  let gx, gn =
    Rune.grad
      T.(pair tensor tensor)
      (fun (x, n) -> Nx.sum (Nx.mul x (Nx.cast f64 n)))
      (v [| 1.; 2. |], n)
  in
  equal ~msg:"through an integer" floats [| 3.; 4. |] (Nx.to_array gx);
  equal ~msg:"an integer parameter" (array int32) [| 0l; 0l |] (Nx.to_array gn);
  equal ~msg:"an unused parameter" signed
    [| (false, 0.) |]
    (with_signs (Rune.grad' (fun _ -> Nx.sum (v [| 7. |])) (v [| 1. |])))

(* An objective of one element is a scalar whatever its shape. *)
let test_one_element () =
  let x = v [| 1.; -2. |] in
  let g s = Nx.to_array (Rune.grad' (fun x -> Nx.reshape s (Nx.sum x)) x) in
  equal ~msg:"of shape [1]" floats [| 1.; 1. |] (g [| 1 |]);
  equal ~msg:"of shape [1; 1]" floats [| 1.; 1. |] (g [| 1; 1 |])

(* One forward and one backward pass give the value and grad's gradient. *)
let test_value_and_grad () =
  let runs = ref 0 in
  let f (x, y) =
    incr runs;
    Nx.sum (Nx.mul x (Nx.sin y))
  in
  let args = (v [| 0.5; -1. |], v [| 2.; 0.25 |]) in
  let y, (gx, gy) = Rune.value_and_grad (two ()) f args in
  equal ~msg:"f's runs" int 1 !runs;
  let gx', gy' = Rune.grad (two ()) f args in
  equal ~msg:"value" bits_t (bits (f args)) (bits y);
  equal ~msg:"gradient" (pair bits_t bits_t)
    (bits gx', bits gy')
    (bits gx, bits gy)

let test_aux () =
  let y, g, aux =
    Rune.value_and_grad_aux T.tensor T.tensor
      (fun x -> (Nx.sum (Nx.sin x), Nx.cos x))
      (v [| 0.5 |])
  in
  equal ~msg:"value" floats
    [| Float.sin 0.5 |]
    (Nx.to_array (Nx.reshape [| 1 |] y));
  equal ~msg:"gradient" floats [| Float.cos 0.5 |] (Nx.to_array g);
  equal ~msg:"aux" floats [| Float.cos 0.5 |] (Nx.to_array aux)

(* A pullback runs no part of [f] again, gives its own memory, and may be
   applied any number of times. *)
let test_pullback () =
  let runs = ref 0 in
  let f x =
    incr runs;
    Nx.mul x (Nx.sin x)
  in
  let w = v [| 3.; 4. |] in
  let _, pullback = Rune.vjp' f (v [| 1.; 2. |]) in
  let first = bits (pullback w) in
  ignore (Nx.add (Nx.donate (pullback w)) w);
  equal ~msg:"applied again" bits_t first (bits (pullback w));
  equal ~msg:"f's runs" int 1 !runs

let test_int_cotangent () =
  let w = v [| 3.; 4. |] in
  let _, pullback =
    Rune.vjp T.tensor
      T.(pair tensor tensor)
      (fun x -> (Nx.sin x, Nx.cast Nx.int32 x))
      (v [| 1.; 2. |])
  in
  equal ~msg:"an integer cotangent is ignored" bits_t
    (bits (pullback (w, Nx.zeros Nx.int32 [| 2 |])))
    (bits (pullback (w, Nx.create Nx.int32 [| 2 |] [| 5l; -3l |])))

(* One array at two leaves gets one result per visit. *)
let test_two_leaves () =
  let xs = [| 0.5; -2. |] in
  let x = v xs in
  let f (a, b) = Nx.mul a (Nx.sin b) in
  let sin = Array.map Float.sin xs in
  let x_cos = Array.map (fun x -> x *. Float.cos x) xs in
  equal ~msg:"grad" floats (Array.append sin x_cos)
    (both (Rune.grad (two ()) (fun ab -> Nx.sum (f ab)) (x, x)));
  let _, pullback = Rune.vjp (two ()) T.tensor f (x, x) in
  equal ~msg:"vjp" floats (Array.append sin x_cos)
    (both (pullback (Nx.ones_like x)))

(* A gradient whose computation ends with a contraction's axes moved back is a
   value of its own. *)
let test_contraction_fresh () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let pattern = Nx.Pattern.v "k i, k j -> i j | k" in
  let b = Nx.create f64 [| 2; 2 |] [| 1.; -1.; 2.; 0.5 |] in
  fresh ~msg:"in C order from offset 0"
    (Rune.grad' (fun a -> Nx.sum (Nx.contract f64 pattern a b)) a)

let test_nesting () =
  let x = v [| 0.5; -1. |] in
  let cube y = Nx.sum (Nx.mul y (Nx.mul y y)) in
  equal ~msg:"second derivative" floats [| 3.; -6. |]
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (Rune.grad' cube x)) x));
  let minus_sin = Array.map (fun x -> -.Float.sin x) [| 0.5; -1. |] in
  let minus_cos = Array.map (fun x -> -.Float.cos x) [| 0.5; -1. |] in
  equal ~msg:"jvp of grad" floats minus_sin
    (Nx.to_array
       (snd
          (Rune.jvp'
             (Rune.grad' (fun x -> Nx.sum (Nx.sin x)))
             x
             (v [| 1.; 1. |]))));
  equal ~msg:"grad of jvp" floats minus_cos
    (Nx.to_array
       (Rune.grad'
          (fun x -> Nx.sum (snd (Rune.jvp' Nx.cos x (Nx.ones_like x))))
          x));
  let one = v [| 1. |] in
  equal ~msg:"perturbations apart" floats [| 1. |]
    (Nx.to_array
       (Rune.grad'
          (fun x ->
            Nx.sum (Nx.mul x (Rune.grad' (fun y -> Nx.sum (Nx.add x y)) one)))
          one))

let test_read () =
  let f x = if Nx.item [ 0 ] x > 0. then Nx.sum (Nx.sin x) else Nx.sum x in
  equal floats [| Float.cos 1. |] (Nx.to_array (Rune.grad' f (v [| 1. |])))

let test_detach () =
  equal floats [| 2.; -1. |]
    (Nx.to_array
       (Rune.grad'
          (fun x -> Nx.sum (Nx.mul x (Rune.detach x)))
          (v [| 2.; -1. |])))

(* An addition under reverse mode adds its primal, once. *)
let test_total () =
  let total = Rune.Total.make () in
  let _, sum =
    Rune.Total.collect total ~init:(v [| 0. |]) (fun () ->
        Rune.grad'
          (fun x ->
            Rune.Total.add total x;
            Nx.sum (Nx.sin x))
          (v [| 2. |]))
  in
  equal floats [| 2. |] (Nx.to_array sum)

(* A gradient lies where its parameter does when every value it meets lies
   there, a placed creation's included; otherwise at the union of their cuts. A
   place's cotangent moves back to its operand. *)
let test_placement () =
  let where x = Format.asprintf "%a" Nx.Placement.pp (Nx.placement x) in
  let x = Nx.place Count.on (v [| 1.; 2. |]) in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.sin x)) x in
  equal ~msg:"placed" string (where x) (where g);
  let w = Nx.place Count.on (v [| 3.; 4. |]) in
  let z = Nx.zeros_at Count.on f64 [| 2 |] in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) z in
  equal ~msg:"a creation's" string (where z) (where g);
  equal ~msg:"its values" floats [| 3.; 4. |]
    (Nx.to_array (Nx.place Nx.Host.on g));
  let module Two = Operand.Two in
  let x = Nx.place Two.on (v [| 1.; 2. |]) in
  let w = Nx.place (Two.split ~axis:0) (v [| 3.; 4. |]) in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) x in
  equal ~msg:"the union of cuts" string (where w) (where g);
  let h = v [| 1.; 2. |] in
  let there x = Nx.sin (Nx.place Count.on x) in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.place Nx.Host.on (there x))) h in
  equal ~msg:"a place's cotangent" string (where h) (where g)

let test_jacrev () =
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let x = v [| 1.; 2.; -3. |] in
  let j = Rune.jacrev' (fun x -> Nx.matmul a x) x in
  equal ~msg:"a matrix's shape" (array int) [| 2; 3 |] (Nx.shape j);
  equal ~msg:"a matrix" floats (Nx.to_array a) (Nx.to_array j);
  let f x = Nx.mul x (Nx.sin x) in
  equal ~msg:"against jacfwd'" floats
    (Nx.to_array (Rune.jacfwd' f x))
    (Nx.to_array (Rune.jacrev' f x));
  let z =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -1.; im = 0.5 } |]
  in
  equal ~msg:"complex-differentiable" close_pairs
    (pairs (Rune.jacfwd' (fun z -> Nx.mul z z) z))
    (pairs (Rune.jacrev' (fun z -> Nx.mul z z) z));
  equal ~msg:"no element" (array int) [| 0; 3 |]
    (Nx.shape (Rune.jacrev' (fun x -> Nx.slice [ R (0, 0) ] x) x))

let test_check_grads () =
  equal ~msg:"agrees" (result unit string) (Ok ())
    (Rune.check_grads T.tensor (fun x -> Nx.sum (Nx.sin x)) (v [| 0.3; 1. |]));
  (* A jump: the differences of step 1e-4 see a slope of -4999, the gradient one
     of 1. *)
  let jump x = Nx.sum (Nx.where (Nx.greater_s x 0.) x (Nx.add_s x 1.)) in
  equal ~msg:"a jump" (result unit string)
    (Error
       "Rune.check_grads: the directional derivative along ones is -4999, and \
        the gradient predicts 1")
    (Rune.check_grads T.tensor jump (v [| 0. |]))

(* Errors *)

type message = Is of string | Starts of string

(* A float array of any width, so that a cotangent's dtype may differ from its
   result's. *)
type real = Real : (float, 's, Nx.host) Nx.t -> real

let real = T.v (fun c (Real x) -> Real (T.Walk.tensor c x))
let tall = Nx.create f64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 0.5; 0.5 |]

let errors =
  let vector = v [| 1.; 2. |] and ints = Nx.zeros Nx.int32 [| 2 |] in
  let pull f =
    ignore (snd (Rune.vjp T.tensor (two ()) (fun x -> (x, x)) vector) f)
  in
  [
    ( "grad of a vector",
      Is "Rune.grad: the objective must return a real scalar, got float64 [2]",
      fun () -> ignore (Rune.grad T.tensor Fun.id vector) );
    ( "grad' of a complex scalar",
      Is
        "Rune.grad': the objective must return a real scalar, got complex128 \
         [1]",
      fun () -> ignore (Rune.grad' Fun.id (z 1. 0.)) );
    ( "grad' of integer parameters",
      Is "Rune.grad': the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.grad' (fun x -> Nx.sum (Nx.cast f64 x)) ints) );
    ( "value_and_grad of a vector",
      Is
        "Rune.value_and_grad: the objective must return a real scalar, got \
         float64 [2]",
      fun () -> ignore (Rune.value_and_grad T.tensor Fun.id vector) );
    ( "value_and_grad' of a vector",
      Is
        "Rune.value_and_grad': the objective must return a real scalar, got \
         float64 [2]",
      fun () -> ignore (Rune.value_and_grad' Fun.id vector) );
    ( "value_and_grad_aux of a vector",
      Is
        "Rune.value_and_grad_aux: the objective must return a real scalar, got \
         float64 [2]",
      fun () ->
        ignore
          (Rune.value_and_grad_aux T.tensor T.tensor (fun x -> (x, x)) vector)
    );
    ( "vjp of integer parameters",
      Is "Rune.vjp: the parameters hold no real or complex tensor",
      fun () -> ignore (Rune.vjp T.tensor T.tensor (Nx.cast f64) ints) );
    ( "a cotangent of another shape",
      Is "Rune.vjp: 1: shape [2] in the result, [3] in the cotangents",
      fun () -> pull (vector, v [| 1.; 1.; 1. |]) );
    ( "a cotangent of another dtype",
      Is "Rune.vjp: the root: float64 in the result, float32 in the cotangents",
      fun () ->
        let _, pullback = Rune.vjp T.tensor real (fun x -> Real x) vector in
        ignore (pullback (Real (Nx.ones Nx.float32 [| 2 |]))) );
    ( "cotangents of other visits",
      Starts "Rune.vjp: ",
      fun () ->
        let _, pullback =
          Rune.vjp T.tensor T.(list tensor) (fun x -> [ x ]) vector
        in
        ignore (pullback [ vector; vector ]) );
    ( "jacrev' of a complex result",
      Is
        "Rune.jacrev': the result is complex and the argument real, so the \
         Jacobian has no place in the argument's dtype; use Rune.jacfwd'",
      fun () -> ignore (Rune.jacrev' (Nx.cast c128) vector) );
    ( "check_grads of a vector",
      Is
        "Rune.check_grads: the objective must return a real scalar, got \
         float64 [2]",
      fun () -> ignore (Rune.check_grads T.tensor Fun.id vector) );
    ( "a complete QR of a tall matrix",
      Is
        "Rune.grad': the tangent of a complete QR factorisation of a tall \
         matrix has no definition",
      fun () ->
        ignore
          (Rune.grad' (fun x -> Nx.sum (fst (Nx.qr ~factors:Complete x))) tall)
    );
    ( "a complete SVD of a non-square matrix",
      Is
        "Rune.grad': the tangent of a complete SVD of a non-square matrix has \
         no definition",
      fun () ->
        ignore
          (Rune.grad'
             (fun x ->
               let u, _, _ = Nx.svd ~factors:Complete f64 x in
               Nx.sum u)
             tall) );
  ]

let test_error (_, message, f) =
  match f () with
  | () -> failf "returned"
  | exception Invalid_argument m -> (
      match message with
      | Is e -> equal string e m
      | Starts p -> starts_with ~affix:p m)

(* Two domains

   grad and jvp of one function, and one pullback that both domains apply, give
   on two domains what they give on one. *)

let product x = Nx.mul x (Nx.sin x)
let at = v [| 0.5; -1.; 2. |]
let direction k = v [| k; 1.; -2. |]

let domain_commands =
  let start () = snd (Rune.vjp' product at) in
  let pullback = abstract "pullback" in
  let pull p k = bits (p (direction k)) in
  let grad k =
    bits (Rune.grad' (fun x -> Nx.sum (Nx.mul (direction k) (product x))) at)
  in
  let jvp k = bits (snd (Rune.jvp' product at (direction k))) in
  let k = Gen.float_range (-2.) 2. in
  [
    command "start" (Gen.unit @-> makes pullback) start start;
    command "pull" (pullback ^-> k @-> returns bits_t) pull pull;
    command "grad" (k @-> returns bits_t) grad grad;
    command "jvp" (k @-> returns bits_t) jvp jvp;
  ]

(* grad of jit and jit of grad of exact functions give grad's bits *)

type jit_case =
  | Jit_case :
      (float, 's) Nx.dtype * int array * Operand.t * Operand.t * Expr.t
      -> jit_case

(* Expressions of functions a compiled call computes exactly, whose derivatives
   sum no terms. *)
let jit_cases =
  let open Gen in
  (let* s = Operand.shape in
   let* e = Expr.gen ~rank:(Array.length s) in
   let* x = Operand.draw s and+ y = Operand.draw s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   let e = Expr.unsummed (Expr.exact e) in
   match D.kind dt with Float -> Jit_case (dt, s, x, y, e) | _ -> assert false)
  |> with_pp (fun ppf (Jit_case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt Operand.pp_shape
        s Operand.pp_view x.view Operand.pp_view y.view Expr.pp e)

let offset (o : Operand.t) = match o.view with Offset _ -> true | _ -> false

let law_jit (Jit_case (dt, s, xo, yo, e)) =
  let x = Operand.make dt s xo and y = Operand.make dt s yo in
  cover "a zero-size shape" (numel s = 0);
  cover "one element" (numel s = 1);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "an offset operand" (offset xo || offset yo);
  cover "a composition of over 6 functions" (Expr.size e > 6);
  let f x y = Nx.sum (Expr.eval x y e) in
  let pair (a, b) = (bits a, bits b) in
  let grad f =
    outcome (fun () -> pair (Rune.grad (two ()) (fun (x, y) -> f x y) (x, y)))
  in
  let expected = grad f in
  let t = outcome_t (Testable.pair bits_t bits_t) in
  let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
  equal ~msg:"grad of jit" t expected (grad g);
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun x y -> Rune.grad (two ()) (fun (x, y) -> f x y) (x, y))
  in
  equal ~msg:"jit of grad" t expected (outcome (fun () -> pair (staged x y)))

(* Residuals *)

(* [x sin (x) exp (sin x)], one map of several nodes. *)
let fused x =
  let f = D.Any D.Float32 in
  let prog =
    P.v ~ins:[| f |]
      [|
        In 0;
        Op1 (Unary Sin, f, 0);
        Op1 (Unary Exp, f, 1);
        Op2 (Binary Mul, 0, 1);
        Op2 (Binary Mul, 3, 2);
      |]
      ~outs:[| 4 |]
  in
  fst
    (Nx.Prim.eval ~by:"test"
       (Map
          {
            layout = L.contiguous (Iarray.of_list [ Nx.numel x ]);
            prog;
            outs = [ Nx.float32 ];
            loads = [| Plain x |];
            at = None;
          }))

let test_residuals () =
  let n = 4096 in
  let x = Nx.place Memory.On.on (Nx.ones Nx.float32 [| n |]) in
  let kept = ref [] in
  let keep f () = kept := Rune.vjp' f x :: !kept in
  (* The result's bytes alone: the map keeps its operand, [x]. *)
  equal ~msg:"one map" int (4 * n) (Memory.held (keep fused));
  (* Three maps keep their operands, the two intermediates among them. *)
  equal ~msg:"three maps" int
    (3 * 4 * n)
    (Memory.held (keep (fun x -> Nx.mul (Nx.sin x) (Nx.exp x))));
  ignore (Sys.opaque_identity !kept)

let () =
  exit
    (run "rune2 grad"
       [
         group "adjoint" adjoint_props;
         group "complex adjoint" complex_props;
         group "jacrev' is jacfwd' at ties and zeros" kink_props;
         group "finite gradients of composed functions" finite_cases;
         group "conventions"
           [
             test "complex gradients" test_complex;
             test "kinks" test_kinks;
             test "selections" test_selections;
             test "signs of zero" test_zeros;
           ];
         group "interface"
           [
             test "integer and unused parameters" test_zero_gradients;
             test "an objective of one element" test_one_element;
             test "value_and_grad" test_value_and_grad;
             test "value_and_grad_aux" test_aux;
             test "a pullback" test_pullback;
             test "an integer cotangent" test_int_cotangent;
             test "one array at two leaves" test_two_leaves;
             test "a gradient after a contraction" test_contraction_fresh;
             test "nesting" test_nesting;
             test "a read reads the primal" test_read;
             test "detach" test_detach;
             test "totals" test_total;
             test "placement" test_placement;
             test "jacrev'" test_jacrev;
             test "check_grads" test_check_grads;
           ];
         cases ~name:(fun (n, _, _) -> n) "errors" errors test_error;
         stateful ~domains:2 ~count:10 ~steps:8
           "results on two domains are one domain's" domain_commands;
         group "jit"
           [
             prop "grad of jit and jit of grad of exact functions" jit_cases
               law_jit;
           ];
         group "memory"
           [ test "a map's residuals are its loads" test_residuals ];
       ])
