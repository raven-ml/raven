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
open Rune_support
open Cases
open Witness

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
      | Uniform _ | Zeros _ | Kinks | Ties _ -> None)
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
    One { fname = "log10"; f1 = Nx.log10 };
    One { fname = "sinc"; f1 = Nx.sinc };
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
    cases "float64" ~id:fname composed (finite_law (f64, f64_format));
    cases "float32" ~id:fname composed (finite_law (Nx.float32, f32_format));
    cases "float16" ~id:fname composed (finite_law (Nx.float16, f16_format));
    cases "bfloat16" ~id:fname composed (finite_law (Nx.bfloat16, bf16_format));
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

(* The logarithms' gradient is [1 / (x ln b)] at every non-zero [x], below
   zero too, where the value is NaN: Nx.log's convention, which Nx.log10
   keeps by scaling log2's on both arms of its guard. Eager and compiled. *)
let test_logarithms () =
  let x = v [| -1.; -10.; -0.5; 2.; 1e300 |] in
  let at name f ln_b =
    let want = Array.map (fun x -> 1. /. (x *. ln_b)) (Nx.to_array x) in
    let grad x = Rune.grad' (fun x -> Nx.sum (f x)) x in
    equal ~msg:name floats want (Nx.to_array (grad x));
    equal ~msg:(name ^ " compiled") floats want (Nx.to_array (Rune.jit' grad x))
  in
  at "log" Nx.log 1.;
  at "log2" Nx.log2 (Float.log 2.);
  at "log10" Nx.log10 (Float.log 10.)

(* sinc's derivatives

   Below [|x| = ¼] sinc is its Taylor series in [x²], whose coefficients
   [sinc_series] gives, and its derivatives at and near [0] are the series': at
   [0] the odd ones are [0], the second is [-π²/3] and the fourth [π⁴/5].
   Elsewhere the first is [(cos (πx) - sinc x) / x], which is [(-1)ⁿ / n] at an
   integer [n]. *)

(* The coefficient of [x²ᵏ], [(-1)ᵏ π²ᵏ / (2k + 1)!]. *)
let sinc_series k =
  let c = ref 1. in
  for j = 1 to k do
    c := -. !c *. Float.pi *. Float.pi /. Float.of_int (2 * j * ((2 * j) + 1))
  done;
  !c

(* The first derivative: the series' to [x²⁴] below [¼], where the next term is
   under 2⁻⁶⁰, and the closed form above. *)
let sinc_d1 x =
  if Float.abs x < 0.25 then begin
    let d = ref 0. in
    for k = 12 downto 1 do
      d := (!d *. x *. x) +. (Float.of_int (2 * k) *. sinc_series k)
    done;
    !d *. x
  end
  else
    let p = Float.pi *. x in
    (Float.cos p -. (Float.sin p /. p)) /. x

let sinc_gen =
  let open Gen in
  frequency
    [
      (3, float_range (-8.) 8.);
      (1, constant 0.);
      (1, map (fun e -> Float.ldexp 1. (-e)) (int_range 1 60));
      (1, map (fun e -> -.Float.ldexp 1. (-e)) (int_range 1 60));
      (1, map Float.of_int (int_range (-6) 6));
      ( 1,
        let+ n = int_range (-6) 6 and+ e = int_range 4 30 in
        Float.of_int n +. Float.ldexp 1. (-e) );
    ]

(* At float32 the derivative is within [2⁻²¹] of the exact one, a few ulps of
   its scale, where the quotient's would be off by [6.5e-4] near [0]; at float64
   within [2⁻⁴⁸]. *)
let sinc_law x =
  cover "at 0" (x = 0.);
  cover "below 1/4, not 0" (x <> 0. && Float.abs x < 0.25);
  cover "at an integer but 0" (x <> 0. && Float.is_integer x);
  let at (type s) (dt : (float, s) Nx.dtype) tol =
    let x = Nx.Dtype.of_float dt x in
    let g =
      Rune.grad' (fun x -> Nx.sum (Nx.sinc x)) (Nx.create dt [| 1 |] [| x |])
    in
    let want = sinc_d1 x in
    equal ~msg:(Nx.Dtype.name dt) (float tol) want
      (Nx.item [ 0 ] (Nx.cast f64 g))
  in
  at f64 0x1p-48;
  at Nx.float32 0x1p-21

let test_sinc_at_zero () =
  let derivative f x = Rune.grad' (fun x -> Nx.sum (f x)) x in
  let check (type s) (dt : (float, s) Nx.dtype) rel =
    let d1 = derivative Nx.sinc in
    let d2 = derivative d1 in
    let d3 = derivative d2 in
    let d4 = derivative d3 in
    let at d =
      Nx.item [ 0 ] (Nx.cast f64 (d (Nx.create dt [| 1 |] [| 0. |])))
    in
    let name = Nx.Dtype.name dt in
    equal ~msg:(name ^ " first") float_exact 0. (Float.abs (at d1));
    equal ~msg:(name ^ " second") (float_rel ~rel ~abs:0.)
      (2. *. sinc_series 1)
      (at d2);
    equal ~msg:(name ^ " third") float_exact 0. (Float.abs (at d3));
    equal ~msg:(name ^ " fourth") (float_rel ~rel ~abs:0.)
      (24. *. sinc_series 2)
      (at d4)
  in
  check f64 1e-15;
  check Nx.float32 1e-6

(* hypot's gradient

   [(x / r, y / r)] in both modes wherever [r] is positive and finite: at
   subnormal, tiny, unit and huge magnitudes, at ties and beside them, and
   beside a zero or a far smaller operand; for [Nx.magnitude], [x / r - i y /
   r]. The reference divides operands scaled by a power of two into the normal
   range, where [Float.hypot] and the quotients keep float64's precision. *)

let hypot_gen =
  let open Gen in
  let exponents =
    [ -1074; -1060; -1040; -1023; -1000; -600; 0; 600; 1000; 1022 ]
  and ratios = [ 0.; 0x1p-60; 1. /. 3.; 1. -. epsilon_float; 1. ] in
  (let+ e = of_list exponents
   and+ m = float_range 1. 1.5
   and+ k = of_list ratios
   and+ swap = bool
   and+ neg_a = bool
   and+ neg_b = bool in
   let a = Float.ldexp m e in
   let b = a *. k in
   let a = if neg_a then -.a else a and b = if neg_b then -.b else b in
   if swap then (b, a) else (a, b))
  |> with_pp (fun ppf (a, b) -> Format.fprintf ppf "x %h, y %h" a b)

let unit_of a b =
  let s = Float.max (Float.abs a) (Float.abs b) in
  let c =
    if s < 0x1p-900 then 0x1p600 else if s > 0x1p900 then 0x1p-600 else 1.
  in
  let a = a *. c and b = b *. c in
  let r = Float.hypot a b in
  (a /. r, b /. r)

let hypot_law (a, b) =
  let sub x = x <> 0. && Float.abs x < 0x1p-1022 in
  cover "a subnormal operand" (sub a || sub b);
  cover "a tie" (Float.abs a = Float.abs b);
  cover "a zero" (a = 0. || b = 0.);
  cover "a magnitude past 2^512"
    (Float.max (Float.abs a) (Float.abs b) > 0x1p512);
  let ux, uy = unit_of a b in
  let near = pair (float 1e-14) (float 1e-14) in
  let x = v [| a |] and y = v [| b |] in
  let item t = Nx.item [ 0 ] t in
  let gx, gy =
    Rune.grad (two ()) (fun (x, y) -> Nx.sum (Nx.hypot x y)) (x, y)
  in
  equal ~msg:"grad" near (ux, uy) (item gx, item gy);
  let along t =
    item
      (snd (Rune.jvp (two ()) T.tensor (fun (x, y) -> Nx.hypot x y) (x, y) t))
  in
  let zero = v [| 0. |] and one = v [| 1. |] in
  equal ~msg:"jvp" near (ux, uy) (along (one, zero), along (zero, one));
  let g = Rune.grad' (fun z -> Nx.sum (Nx.magnitude f64 z)) (z a b) in
  let c : Complex.t = Nx.item [ 0 ] g in
  equal ~msg:"magnitude's grad" near (ux, -.uy) (c.re, c.im)

(* Softmax and log-softmax shift each lane by its greatest element, on which
   their values do not depend: their gradients agree with finite differences,
   where that element ties with another too. *)
(* Lane [l] of a matrix of shape [s] along [axis]: its elements' positions. *)
let lane s axis l =
  List.init s.(axis) (fun k ->
      if axis = 0 then (k * s.(1)) + l else (l * s.(1)) + k)

(* The position in lane [l] of its greatest [scale x], the first one. *)
let greatest s axis scale xs l =
  List.fold_left
    (fun g i -> if scale *. xs.(i) > scale *. xs.(g) then i else g)
    (List.hd (lane s axis l))
    (lane s axis l)

(* In half the cases, each lane's greatest element is copied to another of
   its positions, so that it ties there. *)
let normalised_gen =
  let open Gen in
  let pp ppf (log, axis, scale, s, _) =
    Format.fprintf ppf "%s along %d, scale %g, [%d; %d]"
      (if log then "log_softmax" else "softmax")
      axis scale s.(0) s.(1)
  in
  with_pp pp
    (let* tie = bool in
     let* axis = int_range 0 1 in
     let low a = if tie && a = axis then 2 else 1 in
     let* m = int_range (low 0) 3 in
     let* n = int_range (low 1) 4 in
     let s = [| m; n |] in
     let+ log = bool
     and+ scale = of_list [ 1.; 0.25; -2.; 3. ]
     and+ xs =
       array ~size:(constant (m * n))
         (frequency [ (1, of_list [ -1.; 0.5; 2. ]); (2, float_range (-3.) 3.) ])
     in
     let xs = Array.copy xs in
     if tie then
       for l = 0 to s.(1 - axis) - 1 do
         let g = greatest s axis scale xs l in
         let other = List.find (fun i -> i <> g) (lane s axis l) in
         xs.(other) <- xs.(g)
       done;
     (log, axis, scale, s, xs))

let normalised_law (log, axis, scale, s, xs) =
  let f = if log then Nx.log_softmax else Nx.softmax in
  let x = Nx.create f64 s xs in
  let tied l =
    let g = greatest s axis scale xs l in
    List.exists
      (fun i -> i <> g && scale *. xs.(i) = scale *. xs.(g))
      (lane s axis l)
  in
  cover "a lane whose greatest element ties"
    (List.exists tied (List.init s.(1 - axis) Fun.id));
  cover "log_softmax" log;
  let w = Nx.create f64 s (spread 0.7 (numel s)) in
  let loss x = Nx.sum (Nx.mul w (f ~axes:[ axis ] ~scale x)) in
  equal ~msg:"finite differences" (result unit string) (Ok ())
    (Rune.check_grads T.tensor loss x)

(* Under [~where], their gradients agree with finite differences, and are
   exactly zero at each element outside it, a lane holding none
   included. *)
let where_gen =
  let open Gen in
  let pp ppf (log, axis, _, s, _, _) =
    Format.fprintf ppf "%s along %d, [%d; %d]"
      (if log then "log_softmax" else "softmax")
      axis s.(0) s.(1)
  in
  with_pp pp
    (let* m = int_range 1 3 in
     let* n = int_range 1 4 in
     let+ log = bool
     and+ axis = int_range 0 1
     and+ scale = of_list [ 1.; -2. ]
     and+ held = array ~size:(constant (m * n)) (frequency [ (2, constant true); (1, constant false) ])
     and+ xs = array ~size:(constant (m * n)) (float_range (-3.) 3.) in
     (log, axis, scale, [| m; n |], held, xs))

type normaliser = {
  f :
    'r 'd.
    ?axes:int list ->
    ?scale:float ->
    ?where:'d Nx.bool_t ->
    (float, 'r, 'd) Nx.t ->
    (float, 'r, 'd) Nx.t;
}

let normaliser log =
  if log then { f = Nx.log_softmax } else { f = Nx.softmax }

let where_loss n ~axis ~scale where w x =
  Nx.sum
    (Nx.where where
       (Nx.mul w (n.f ~axes:[ axis ] ~scale ~where x))
       (Nx.zeros_like x))

let where_law (log, axis, scale, s, held, xs) =
  let f = normaliser log in
  let x = Nx.create f64 s xs and where = Nx.create Nx.bool s held in
  let none l = List.for_all (fun i -> not held.(i)) (lane s axis l) in
  cover "a lane holding none" (List.exists none (List.init s.(1 - axis) Fun.id));
  cover "an element outside where" (Array.exists not held);
  (* log_softmax is -inf outside where: the loss reads only the held. *)
  let w =
    Nx.create f64 s
      (Array.mapi (fun i v -> if held.(i) then v else 0.) (spread 0.7 (numel s)))
  in
  let loss = where_loss f ~axis ~scale where w in
  equal ~msg:"finite differences" (result unit string) (Ok ())
    (Rune.check_grads T.tensor loss x);
  let g = Nx.to_array (Rune.grad' loss x) in
  Array.iteri
    (fun i h ->
      if not h then
        equal ~msg:"outside where" float_exact 0. (Float.abs g.(i)))
    held

(* At float32, whose masked operand is stored as at float64, and at the
   narrow floats, whose path reads the mask in each pass, the gradient is
   float64's of the rounded operands within [16 u] of the largest of its
   elements plus one, [u] the dtype's unit roundoff: the forward pass rounds
   once, the backward pass a few times more, and [p (w - Σ p w)] cancels
   terms of [w]'s size. It is exactly zero outside the mask. *)
let where_narrow_law (type r) (dt : (float, r) D.t) u
    (log, axis, scale, s, held, xs) =
  let f = normaliser log in
  let where = Nx.create Nx.bool s held in
  let x = Nx.cast dt (Nx.create f64 s xs) in
  let w =
    Nx.cast dt
      (Nx.create f64 s
         (Array.mapi (fun i v -> if held.(i) then v else 0.) (spread 0.7 (numel s))))
  in
  let g = Nx.to_array (Nx.cast f64 (Rune.grad' (where_loss f ~axis ~scale where w) x)) in
  let wide =
    Nx.to_array
      (Rune.grad' (where_loss f ~axis ~scale where (Nx.cast f64 w)) (Nx.cast f64 x))
  in
  let top = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. wide in
  let tol = 16. *. u *. (top +. 1.) in
  Array.iteri
    (fun i h ->
      if not h then equal ~msg:"outside where" float_exact 0. (Float.abs g.(i))
      else if Float.abs (g.(i) -. wide.(i)) > tol then
        failf "%s at %d: %h, float64's %h" (D.name dt) i g.(i) wide.(i))
    held

(* Compiled, their gradients are the eager ones. *)
let test_normalised_jit () =
  let x = Nx.create f64 [| 3; 4 |] (spread 0.2 12) in
  let w = Nx.create f64 [| 3; 4 |] (spread 0.7 12) in
  let where =
    Nx.create Nx.bool [| 3; 4 |] (Array.init 12 (fun i -> i mod 3 <> 0 && i < 8))
  in
  List.iter
    (fun (name, loss) ->
      equal ~msg:name (array (float 1e-12))
        (Nx.to_array (Rune.grad' loss x))
        (Nx.to_array (Rune.grad' (Rune.jit' loss) x)))
    [
      ("softmax", fun x -> Nx.sum (Nx.mul w (Nx.softmax ~axes:[ 1 ] ~scale:0.7 x)));
      ( "log_softmax",
        fun x -> Nx.sum (Nx.mul w (Nx.log_softmax ~axes:[ 0 ] ~scale:(-1.5) x)) );
      ( "softmax under where",
        where_loss (normaliser false) ~axis:1 ~scale:1. where w );
      ( "log_softmax under where",
        where_loss (normaliser true) ~axis:0 ~scale:(-1.5) where w );
    ]

(* Mapped over a batch with a mask the map does not batch, their gradients
   are each example's. *)
let test_normalised_vmap () =
  let xs = Nx.create f64 [| 3; 3; 4 |] (spread 0.4 36) in
  let w = Nx.create f64 [| 3; 4 |] (spread 0.7 12) in
  let where =
    Nx.create Nx.bool [| 3; 4 |] (Array.init 12 (fun i -> i mod 3 <> 0 && i < 8))
  in
  List.iter
    (fun (name, f, axis) ->
      let grad x = Rune.grad' (where_loss f ~axis ~scale:0.5 where w) x in
      let each =
        Nx.stack (List.init 3 (fun b -> grad (Nx.slice [ I b ] xs)))
      in
      equal ~msg:name (array (float 1e-12)) (Nx.to_array each)
        (Nx.to_array (Rune.vmap' grad xs)))
    [ ("softmax", normaliser false, 1); ("log_softmax", normaliser true, 0) ]

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

(* A factorisation's values alone have finite gradients where values repeat,
   where its vectors' tangents are not finite: the sum of [eigvalsh]'s is the
   trace, whose gradient is the identity, and the sum of the squares of
   [svdvals]'s is [‖a‖_F²], whose gradient is [2 a]. *)
let test_repeated_values () =
  let i = Nx.eye f64 3 in
  equal ~msg:"eigvalsh at the identity" floats (Nx.to_array i)
    (Nx.to_array (Rune.grad' (fun a -> Nx.sum (Nx.eigvalsh f64 a)) i));
  let a = Nx.create f64 [| 3; 2 |] [| 2.; 0.; 0.; 2.; 0.; 0. |] in
  equal ~msg:"svdvals of two equal values" floats
    (Nx.to_array (Nx.mul_s a 2.))
    (Nx.to_array
       (Rune.grad' (fun a -> Nx.sum (Nx.square (Nx.svdvals f64 a))) a))

(* Softplus is smooth, of derivative sigmoid everywhere: at both zeros,
   subnormals, the infinities and magnitudes where either arm of its form
   saturates, each an example. *)
let softplus_specials =
  [
    0.; -0.; 0x1p-1074; -0x1p-1074; 0x1p-1022; -0x1p-1022; Float.infinity;
    Float.neg_infinity; 1e300; -1e300; 750.; -750.; 37.; -37.;
  ]

let softplus_points =
  Gen.(
    with_pp
      (fun ppf x -> Format.fprintf ppf "%h" x)
      (frequency
         [ (4, float_range (-50.) 50.); (1, of_list softplus_specials) ]))

let softplus_law x =
  let at = v [| x |] in
  equal
    (array (float_rel ~rel:(8. *. epsilon_float) ~abs:0x1p-1074))
    (Nx.to_array (Nx.sigmoid at))
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (Nx.softplus x)) at))

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
                  at = None;
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
                 at = None;
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

(* Two reads of regions of one array, eagerly and compiled: each element
   receives the contributions of the reads that reach it alone, so that one
   [-0.] keeps its sign, and an element no read reaches is [+0.]. Each
   read's cotangent is [-0.] at every element. *)
let test_region_zeros () =
  let zeroed y = Nx.sum (Nx.mul y (Nx.full_like y (-0.))) in
  let two r s x = Nx.add (zeroed (r x)) (zeroed (s x)) in
  let check msg want f xs =
    let x = v xs in
    equal ~msg signed want (with_signs (Rune.grad' f x));
    let g = Rune.jit T.(tensor @-> returns tensor) (Rune.grad' f) in
    equal ~msg:(msg ^ ", compiled") signed want (with_signs (g x))
  in
  let neg = (true, -0.) and pos = (false, 0.) in
  let windows window step = Nx.sliding_window ~window ~step in
  check "two rows" [| neg; neg |]
    (two (Nx.get [ 0 ]) (Nx.get [ 1 ]))
    [| 1.; 2. |];
  check "a slice by steps and a row" [| neg; neg; neg; pos |]
    (two (Nx.slice [ Rs (0, 3, 2) ]) (Nx.get [ 1 ]))
    [| 1.; 2.; 3.; 4. |];
  check "windows and a row" [| neg; neg; neg; pos |]
    (two (windows 1 2) (Nx.get [ 1 ]))
    [| 1.; 2.; 3.; 4. |];
  check "two reads of windows" [| neg; pos; neg; neg |]
    (two (windows 1 2) (windows 1 3))
    [| 1.; 2.; 3.; 4. |];
  check "windows apart and a row" [| neg; neg; neg; neg; neg |]
    (two (windows 2 3) (Nx.get [ 2 ]))
    [| 1.; 2.; 3.; 4.; 5. |];
  (* Along the second axis a dilated window spans 3 of 2: no window. *)
  let patches x =
    Nx.extract_patches ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |]
      ~dilation:[| 1; 2 |]
      ~padding:[| (0, 0); (0, 0) |]
      (Nx.reshape [| 1; 3; 2 |] x)
  in
  check "no window and the whole" (Array.make 6 neg) (two patches Fun.id)
    [| 1.; 2.; 3.; 4.; 5.; 6. |]

(* Reads of regions

   Reads of an array by slices and by windows, each summed and scaled: an
   element's gradient is the sum of the scales of the reads that reach it, a
   term each time a read reaches it, the reads taken from the last back, as
   the transpose meets them, with no zero before the first, and [+0.] where
   none does. A read of windows that share elements contributes [+0.] to each
   element it does not reach. Scales of far magnitudes and zeros of both
   signs show the order of the sums and their signs, in bits, eagerly and
   compiled. *)

type read = Slice of int * int * int | Windows of int * int
type reads = { n : int; reads : (read * float) list }

let pp_reads ppf { n; reads } =
  let pp_read ppf (r, c) =
    match r with
    | Slice (start, count, step) ->
        Format.fprintf ppf "%d from %d by %d, times %h" count start step c
    | Windows (size, step) ->
        Format.fprintf ppf "windows of %d by %d, times %h" size step c
  in
  let sep ppf () = Format.fprintf ppf "; " in
  Format.fprintf ppf "%d elements: %a" n
    (Format.pp_print_list ~pp_sep:sep pp_read)
    reads

let reads_gen =
  let open Gen in
  let slice n =
    if n = 0 then constant (Slice (0, 0, 1))
    else
      let* start = int_range 0 (n - 1)
      and+ step = of_list [ -2; -1; 1; 2; 3 ] in
      let most =
        if step > 0 then ((n - 1 - start) / step) + 1 else (start / -step) + 1
      in
      let+ count = int_range 0 most in
      Slice (start, count, step)
  in
  let windows =
    let+ size = int_range 1 3 and+ step = int_range 1 3 in
    Windows (size, step)
  in
  let scale = of_list [ 0x1p53; 1.; -0x1p53; -0.; 0.; 3.; -2. ] in
  let read n = pair (one_of [ slice n; windows ]) scale in
  let any =
    let* n = int_range 0 6 in
    let+ reads = list ~size:(int_range 1 4) (read n) in
    { n; reads }
  in
  (* Among others, one of the reads [any] draws least often: of two elements or
     more by a negative step, in one case in 15, or of windows that share
     elements, in one in 9. *)
  let rare =
    let* n = int_range 3 6 in
    let reversed =
      let* step = of_list [ -1; -2 ] in
      let* count = int_range 2 (((n - 1) / -step) + 1) in
      let+ start = int_range ((count - 1) * -step) (n - 1) in
      Slice (start, count, step)
    in
    let shared =
      let* size = int_range 2 (Int.min 3 (n - 1)) in
      let+ step = int_range 1 (Int.min (size - 1) (n - size)) in
      Windows (size, step)
    in
    let* one = pair (one_of [ reversed; shared ]) scale
    and+ others = list ~size:(int_range 0 3) (read n) in
    let+ reads = permutation (one :: others) in
    { n; reads }
  in
  (* A [-0.] read of the last element, which the transpose meets after two
     reads that overlap before it. *)
  let after_overlap =
    let* n = int_range 3 6 in
    let* c1 = int_range 2 (n - 1) and+ c2 = int_range 1 (n - 2) in
    let+ k1 = scale and+ k2 = scale in
    let one = (Slice (n - 1, 1, 1), -0.) in
    { n; reads = [ one; (Slice (0, c1, 1), k1); (Slice (1, c2, 1), k2) ] }
  in
  with_pp pp_reads (frequency [ (3, any); (1, after_overlap); (2, rare) ])

(* The elements a read reaches, an element once each time, and whether it
   reads windows that share elements. *)
let reached n = function
  | Slice (start, count, step) ->
      (List.init count (fun j -> start + (j * step)), false)
  | Windows (size, step) ->
      let windows = if size > n then 0 else ((n - size) / step) + 1 in
      let window w = List.init size (fun j -> (w * step) + j) in
      (List.concat (List.init windows window), windows > 1 && step < size)

let read_of n r x =
  match r with
  | Slice (start, 0, _) -> Nx.slice [ R (start, start) ] x
  | Slice (start, count, step) ->
      let last = start + ((count - 1) * step) in
      (* A stop past the axis's start, [-n - 1], counts from the end to
         [-1]. *)
      let stop =
        if step > 0 then last + 1 else if last = 0 then -n - 1 else last - 1
      in
      Nx.slice [ Rs (start, stop, step) ] x
  | Windows (size, step) -> Nx.sliding_window ~window:size ~step x

let regions_law { n; reads } =
  let f x =
    List.fold_left
      (fun s (r, k) -> Nx.add s (Nx.mul_s (Nx.sum (read_of n r x)) k))
      (Nx.scalar f64 0.) reads
  in
  let element i =
    let term (r, k) =
      let at, shares = reached n r in
      match List.filter (( = ) i) at with
      | [] -> if shares then Some 0. else None
      | _ :: rest -> Some (List.fold_left (fun s _ -> s +. k) k rest)
    in
    let add s t =
      match (s, t) with
      | None, t -> t
      | s, None -> s
      | Some a, Some b -> Some (a +. b)
    in
    Option.value ~default:0.
      (List.fold_left (fun s r -> add s (term r)) None (List.rev reads))
  in
  let elements = List.init n Fun.id in
  let reaching i =
    List.filter (fun (r, _) -> List.mem i (fst (reached n r))) reads
  in
  let alone i =
    let e = element i in
    List.length (reaching i) = 1
    && List.length reads > 1 && e = 0. && Float.sign_bit e
  in
  let reversed = function Slice (_, c, s), _ -> c > 1 && s < 0 | _ -> false in
  cover "an element two reads reach"
    (List.exists (fun i -> List.length (reaching i) > 1) elements);
  cover "windows that share elements"
    (List.exists (fun (r, _) -> snd (reached n r)) reads);
  cover "a reversed slice" (List.exists reversed reads);
  cover "a negative zero of one read among others" (List.exists alone elements);
  (* An element only one [-0.] read reaches, which the transpose meets
     after two reads in different regions that share an element: each read
     after it in the program. *)
  let rec overlapped = function
    | [] -> false
    | a :: rest ->
        let shared b = a <> b && List.exists (fun e -> List.mem e b) a in
        List.exists shared rest || overlapped rest
  in
  let after_overlap i =
    let reaches (r, _) = List.mem i (fst (reached n r)) in
    match List.find_index reaches reads with
    | None -> false
    | Some k ->
        let later = List.filteri (fun j _ -> j > k) reads in
        alone i
        && (not (List.exists (fun (r, _) -> snd (reached n r)) reads))
        && overlapped (List.map (fun (r, _) -> fst (reached n r)) later)
  in
  cover "a negative zero of one read after two that overlap"
    (List.exists after_overlap elements);
  let x = Nx.create f64 [| n |] (Array.make n 1.) in
  let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
  let want = Array.map Int64.bits_of_float (Array.init n element) in
  equal ~msg:"eager" (array int64) want (bits (Rune.grad' f x));
  let g = Rune.jit T.(tensor @-> returns tensor) (Rune.grad' f) in
  equal ~msg:"compiled" (array int64) want (bits (g x))

(* A truncated normal over [[0, inf]] does not move with its upper bound,
   whose masses' unselected branches meet a zero cotangent with no infinite
   factor. A draw's tangent through a broadcast bound, which a tangent of
   distinct elements moves, is its tangent through the bound's copy. *)
let test_draws () =
  let lower = Nx.zeros f64 [| 8 |] in
  let over upper = Nx.Rng.truncated_normal ~key:(Nx.Rng.key 3) lower upper in
  equal ~msg:"at an infinite upper bound" floats (Array.make 8 0.)
    (Nx.to_array
       (Rune.grad' (fun u -> Nx.sum (over u)) (Nx.full f64 [| 8 |] infinity)));
  let upper = Nx.full f64 [| 8 |] 2. in
  let from lower = Nx.Rng.truncated_normal ~key:(Nx.Rng.key 5) lower upper in
  let bound = Nx.broadcast_to [| 8 |] (Nx.scalar f64 (-1.)) in
  let t = v (Array.init 8 (fun i -> Float.of_int (i - 3))) in
  equal ~msg:"through a broadcast lower bound" floats
    (Nx.to_array (snd (Rune.jvp' from (Nx.copy bound) t)))
    (Nx.to_array (snd (Rune.jvp' from bound t)))

(* Dirichlet draws

   A dirichlet draw is its gamma draws' row ratios, [x_j = G_j / Σ G], from
   the same key, whose rounds it shares, so in its concentrations its tangent
   is [x_j (δ_j - Σ_k x_k δ_k)], [δ_j = G'_j / G_j] the logarithmic tangent of
   gamma draw [j], and a row's tangents sum to zero. The reference is formed
   in float64 from the gamma draws and their tangents at the draw's dtype.

   The tolerances, [u] the dtype's unit roundoff and [γ(k) = k u / (1 - k
   u)] as rune.mli's Reductions section defines it. The draw computes [x'_j
   = e'_j / d - x_j fs' / d], [e'_j = x_j d (δ_j - δ_t)] the tangent of term
   [j], [t] the row's greatest, and [fs' = Σ e'_k] a float sum, which errs by
   up to [γ(n - 1) Σ |e'_k|] in any order of addition, so by
   [γ(n - 1) x_j S], [S = Σ_k x_k |δ_k - δ_t|], in [x'_j]. Each [e'_j] and
   [δ_j] also rounds its own terms, of the scale [Q = max_k |δ_k| + |s'/s|
   max_k |log G_k|], the logarithms' scale [s] the least of 1 and the row's
   concentrations and [s'] its tangent. [32u Q x_j] is a chosen allowance
   for those roundings, which [Q] need not bound term by term: measured over
   300 cases, an element erred by at most [1.8u max_k |δ_k|]. So element
   [j] is held within [x_j (γ(n - 1) S + 32u Q)] and a row's sum of
   tangents within the sum of these.

   The gradient pairs with a tangent as the cotangent [w] pairs with the
   tangent's image. The forward side errs by [Σ_j |w_j| x_j (γ(n - 1) S +
   32u Q)]. The reverse pass adds float sums over the row: [W = Σ_k w_k
   x_k], whose error moves the pairing by [γ(n - 1) S Σ_k |w_k| x_k]; and
   the cotangents of the row's greatest [z_t], of [s] through the quotient
   [(z_j - z_t) / s] and of [s] through [z_j = s log G_j], sums of terms
   [x_j (w_j - W)] times [1], [log G_j - log G_t] and [log G_j], over [s],
   which cancel exactly. They err by [γ(n - 1)] of their terms' magnitude
   and meet the tangents [z'_t = s' log G_t + s δ_t] and [s'], so by
   [γ(n - 1) Σ_j x_j |w_j - W| (|δ_t| + |s'/s| (|log G_t| + |log G_j -
   log G_t| + |log G_j|))] at most, below [8 γ(n - 1) Q Σ_j |w_j| x_j]
   since [Σ_j x_j |w_j - W| <= 2 Σ_j x_j |w_j|]. With the reverse pass's
   own roundings as the forward one's, the pairings agree within
   [2 Σ_j |w_j| x_j (γ(n - 1) (S + 4Q) + 32u Q)]. *)
type dirichlet_case = { dkey : int; wide : bool; len : int; seed : int }

let dirichlet_gen =
  let open Gen in
  let pp ppf c =
    Format.fprintf ppf "key %d, %s, rows of %d, seed %d" c.dkey
      (if c.wide then "float64" else "float32")
      c.len c.seed
  in
  with_pp pp
    (let+ dkey = int
     and+ wide = bool
     and+ len = of_list [ 1; 2; 8; 1025 ]
     and+ seed = int in
     { dkey; wide; len; seed })

(* A draw's tangent and its weighted sum's gradient, compiled once for each
   dtype: a draw's key, concentrations, tangent and weights are arguments, so
   the cases of one shape share a variant. *)
let compiled_jvp () =
  Rune.jit
    T.(Nx.Rng.ptree @-> tensor @-> tensor @-> returns (pair tensor tensor))
    (fun key a t -> Rune.jvp' (fun a -> Nx.Rng.dirichlet ~key a) a t)

let compiled_grad () =
  Rune.jit
    T.(Nx.Rng.ptree @-> tensor @-> tensor @-> returns tensor)
    (fun key w a ->
      Rune.grad' (fun a -> Nx.sum (Nx.mul w (Nx.Rng.dirichlet ~key a))) a)

let dirichlet_law_at (type s) (dt : (float, s) Nx.dtype) u jvp grad c =
  let key = Nx.Rng.key c.dkey in
  let st = Random.State.make [| c.seed |] in
  let rows = 3 in
  let size = rows * c.len in
  let s = [| rows; c.len |] in
  let a =
    Nx.create dt s
      (Array.init size (fun _ -> 0.5 +. Random.State.float st 19.5))
  in
  let t = Nx.create dt s (spread (Random.State.float st 6.) size) in
  let w = Nx.create dt s (spread (Random.State.float st 6.) size) in
  let draw a = Nx.Rng.dirichlet ~key a in
  let wide x = Nx.to_array (Nx.cast f64 x) in
  let bits x = Array.map Int64.bits_of_float x in
  let x, dx = Rune.jvp' draw a t in
  equal ~msg:"primal" (array int64) (bits (wide (draw a))) (bits (wide x));
  let g, dg = Rune.jvp' (fun a -> Nx.Rng.gamma ~key a) a t in
  let x = wide x and dx = wide dx and g = wide g and dg = wide dg in
  let delta = Array.mapi (fun i gi -> dg.(i) /. gi) g in
  let row r = Array.init c.len (fun j -> (r * c.len) + j) in
  let fold r f = Array.fold_left (fun m i -> f m i) 0. (row r) in
  let mean r = fold r (fun m i -> m +. (x.(i) *. delta.(i))) in
  let top r =
    Array.fold_left (fun t i -> if x.(i) > x.(t) then i else t) (r * c.len)
      (row r)
  in
  let as_ = wide a and ts = wide t in
  (* [s'/s] for row [r]: [s] is its least concentration where that is below
     1, which every row here has unless all are, else 1. *)
  let scale_tangent r =
    let m =
      Array.fold_left
        (fun m i -> if as_.(i) < as_.(m) then i else m)
        (r * c.len) (row r)
    in
    if as_.(m) < 1. then ts.(m) /. as_.(m) else 0.
  in
  let gamma_ k = Float.of_int k *. u /. (1. -. (Float.of_int k *. u)) in
  (* [(S, Q)] for row [r]. *)
  let per r =
    let dt = delta.(top r) in
    let s = fold r (fun m i -> m +. (x.(i) *. Float.abs (delta.(i) -. dt))) in
    let q =
      fold r (fun m i -> Float.max m (Float.abs delta.(i)))
      +. Float.abs (scale_tangent r)
         *. fold r (fun m i -> Float.max m (Float.abs (log g.(i))))
    in
    (s, q)
  in
  let sq = Array.init rows per in
  let g = gamma_ (c.len - 1) in
  (* [γ(n - 1) S + 32u Q] for row [r], an element's tangent's allowance. *)
  let per = Array.map (fun (s, q) -> (g *. s) +. (32. *. u *. q)) sq in
  (* [γ(n - 1) (S + 4Q) + 32u Q] for row [r], the pairing's. *)
  let paired =
    Array.map (fun (s, q) -> (g *. (s +. (4. *. q))) +. (32. *. u *. q)) sq
  in
  let within ~msg y =
    for r = 0 to rows - 1 do
      let p = per.(r) in
      Array.iter
        (fun i ->
          let want = x.(i) *. (delta.(i) -. mean r) in
          at_most ~msg float_exact ~than:(x.(i) *. p)
            (Float.abs (y.(i) -. want)))
        (row r);
      at_most
        ~msg:(msg ^ ": a row's tangents' sum")
        float_exact
        ~than:(fold r (fun m i -> m +. (x.(i) *. p)))
        (Float.abs (fold r (fun m i -> m +. y.(i))))
    done
  in
  within ~msg:"eager" dx;
  let cx, cdx = jvp key a t in
  equal ~msg:"compiled primal" (array int64) (bits x) (bits (wide cx));
  within ~msg:"compiled" (wide cdx);
  let loss a = Nx.sum (Nx.mul w (draw a)) in
  let ws = wide w in
  let allowed =
    2.
    *. Array.fold_left ( +. ) 0.
         (Array.init size (fun i ->
              Float.abs ws.(i) *. x.(i) *. paired.(i / c.len)))
  in
  let pairs ~msg gr =
    let lhs = ref 0. and rhs = ref 0. in
    Array.iteri
      (fun i gi ->
        lhs := !lhs +. (gi *. ts.(i));
        rhs := !rhs +. (ws.(i) *. dx.(i)))
      gr;
    at_most ~msg float_exact ~than:allowed (Float.abs (!lhs -. !rhs))
  in
  pairs ~msg:"gradient" (wide (Rune.grad' loss a));
  pairs ~msg:"compiled gradient" (wide (grad key w a))

let dirichlet_law =
  let jvp32 = compiled_jvp () and grad32 = compiled_grad () in
  let jvp64 = compiled_jvp () and grad64 = compiled_grad () in
  fun c ->
    cover "float32" (not c.wide);
    cover "float64" c.wide;
    cover "a row of one" (c.len = 1);
    cover "a row of two" (c.len = 2);
    cover "a row past a tile of 1024 terms" (c.len > 1024);
    if c.wide then dirichlet_law_at f64 0x1p-53 jvp64 grad64 c
    else dirichlet_law_at Nx.float32 0x1p-24 jvp32 grad32 c

(* At fixed keys and concentrations, away from the edges where a gamma draw
   takes another round, the gradient of a weighted sum of a draw agrees with
   central differences, eager and compiled. *)
let test_dirichlet_differences () =
  List.iter
    (fun (k, conc) ->
      let key = Nx.Rng.key k and n = Array.length conc in
      let a = Nx.create f64 [| 1; n |] conc in
      let w = Nx.create f64 [| 1; n |] (spread 0.4 n) in
      let loss a = Nx.sum (Nx.mul w (Nx.Rng.dirichlet ~key a)) in
      let msg = Printf.sprintf "key %d" k in
      equal ~msg (result unit string) (Ok ())
        (Rune.check_grads T.tensor loss a);
      equal ~msg:(msg ^ ", compiled") (result unit string) (Ok ())
        (Rune.check_grads T.tensor
           (Rune.jit T.(tensor @-> returns tensor) loss)
           a))
    [
      (3, [| 0.7; 2.; 5.; 1.3 |]);
      (11, [| 4.; 0.9; 12.; 3.5; 1.1; 7.; 2.2; 6. |]);
      (29, [| 1.5; 9. |]);
    ]

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

(* The gradient through [Nx_quant.apply] of a Q8_0 or Q6_K weight, whose
   scales are float16 values read from its bytes: in the input, the
   cotangent by the decoded weight, its product, bit for bit; in the
   weight, zero, as an integer array's is, its tangent stopped by the
   bitcasts from its bytes. *)
let test_quantised format () =
  let n = 6 and m = 3 in
  let w = Quants.weight format ~rows:n ~blocks:2 in
  let k = 2 * Quants.block_values format in
  let f32 = Nx.float32 in
  let x = Nx.create f32 [| m; k |] (spread 0.3 (m * k)) in
  let c = Nx.create f32 [| m; n |] (spread 1.1 (m * n)) in
  let loss (w, x) = Nx.sum (Nx.mul c (Nx_quant.apply w x)) in
  let params = Nx.Ptree.(pair (v Nx_quant.walk) tensor) in
  let gw, gx = Rune.grad params loss (w, x) in
  equal ~msg:"in the input" bits_t
    (bits (Nx.matmul c (Nx_quant.dequant f32 w)))
    (bits gx);
  let zero t = Array.for_all (( = ) 0) (Nx.to_array t) in
  match gw with
  | Nx_quant.Q8_0 { blocks } | Nx_quant.Q6_K { blocks } ->
      is_true ~msg:"in the weight" (zero blocks)
  | _ -> fail "the gradient is of another format"

(* An init through [Nx_quant.apply]: its gradient is the cotangent summed
   along the rows it broadcasts over; mapped over its lanes, each lane is
   its product alone, the lanes' rows few enough that nx.cpu sums the
   mapped product in lane order; compiled at float32, where the products'
   sum is already in the dtype, it is the compiled product plus the
   init. *)
let test_quantised_init () =
  let n = 6 and m = 2 and lanes = 3 in
  let w = Quants.weight Quants.Q8_0 ~rows:n ~blocks:2 in
  let k = 2 * Quants.block_values Quants.Q8_0 in
  let f32 = Nx.float32 in
  let x = Nx.create f32 [| m; k |] (spread 0.3 (m * k)) in
  let c = Nx.create f32 [| m; n |] (spread 1.1 (m * n)) in
  let b = Nx.create f32 [| n |] (spread 2.3 n) in
  let product b = Nx_quant.apply ~init:b w x in
  equal ~msg:"grad" bits_t
    (bits (Nx.sum ~axes:[ 0 ] c))
    (bits (Rune.grad' (fun b -> Nx.sum (Nx.mul c (product b))) b));
  let bs = Nx.create f32 [| lanes; n |] (spread 0.7 (lanes * n)) in
  equal ~msg:"vmap" bits_t
    (bits
       (Nx.stack ~axis:0
          (List.init lanes (fun l -> product (Nx.get [ l ] bs)))))
    (bits (Rune.vmap' product bs));
  let jit f = Rune.jit' f x in
  equal ~msg:"jit" bits_t
    (bits (Nx.add (jit (Nx_quant.apply w)) b))
    (bits (jit (Nx_quant.apply ~init:b w)))

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

(* A gradient lies where its parameter does, a placed creation's included,
   whatever the values it meets. A place's cotangent moves back to its
   operand. *)
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
  equal ~msg:"beside a cut" string (where x) (where g);
  let h = v [| 1.; 2. |] in
  let there x = Nx.sin (Nx.place Count.on x) in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.place Nx.Host.on (there x))) h in
  equal ~msg:"a place's cotangent" string (where h) (where g)

(* A take of positions split over two devices, from a value whole on both or
   split along the taken axis, has the gradient it has on one device, where
   the value lies. Beside a value whole on both, each device scatters its
   half of the cotangent, and a position both hold sums the halves; beside
   a value split so, the positions move to each half of the gradient. *)
let test_cut_take () =
  let module Two = Operand.Two in
  let where x = Format.asprintf "%a" Nx.Placement.pp (Nx.placement x) in
  let h = Nx.create f64 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  let ph = Nx.create Nx.int64 [| 4 |] [| 3L; 0L; 3L; 1L |] in
  let f p x = Nx.sum (Nx.square (Nx.take ~axis:0 p x)) in
  let expected = Nx.to_array (Rune.grad' (f ph) h) in
  let split = Two.split ~axis:0 in
  let check msg q =
    let x = Nx.place q h in
    let g = Rune.grad' (f (Nx.place split ph)) x in
    equal ~msg string (where x) (where g);
    equal ~msg floats expected (Nx.to_array (Nx.place Nx.Host.on g))
  in
  check "whole" Two.on;
  check "split" split

(* Placement

   Every tangent and cotangent lies at its primal's placement, so a forward pass
   nx routes has a backward pass it routes: a gradient lies where its parameter
   does and holds the values it holds on one device. *)

module Two = Operand.Two

let where_of x = Format.asprintf "%a" Nx.Placement.pp (Nx.placement x)
let near = array (float_rel ~rel:1e-12 ~abs:1e-12)

(* [numel s] elements from [seed]. *)
let fill seed s =
  Array.init (numel s) (fun k -> Float.sin ((Float.of_int k *. 1.7) +. seed))

let at p ?(seed = 0.) s = Nx.place p (Nx.create f64 s (fill seed s))

type 'd f64_at = (float, D.float64_elt, 'd) Nx.t

(* A function of a parameter and of constants, at any set. *)
type fn = { f : 'd. 'd f64_at array -> 'd f64_at -> 'd f64_at }

(* [f]'s gradient at [x] beside [consts] lies at [x]'s placement and equals the
   gradient on the host. *)
let placed_grad ~msg { f } consts x =
  let g = Rune.grad' (f consts) x in
  equal ~msg string (where_of x) (where_of g);
  let host y = Nx.place Nx.Host.on y in
  let expected = Rune.grad' (f (Array.map host consts)) (host x) in
  equal ~msg near (Nx.to_array expected) (Nx.to_array (host g))

(* A reduction's cotangent, cut along axis 0 by [w], broadcast back over the
   axis it summed meets [c], cut along axis 1. *)
let rows =
  {
    f = (fun k y -> Nx.sum (Nx.mul (Nx.sum ~axes:[ 1 ] (Nx.mul y k.(0))) k.(1)));
  }

let test_cut_residual () =
  placed_grad ~msg:"rows" rows
    [| at (Two.split ~axis:1) [| 4; 6 |]; at (Two.split ~axis:0) [| 4 |] |]
    (at Two.on ~seed:1. [| 4; 6 |])

(* Two contributions, cut along different axes. *)
let cuts =
  {
    f = (fun k x -> Nx.add (Nx.sum (Nx.mul x k.(0))) (Nx.sum (Nx.mul x k.(1))));
  }

let test_two_cuts () =
  placed_grad ~msg:"a sum" cuts
    [|
      at (Two.split ~axis:0) [| 4; 4 |];
      at (Two.split ~axis:1) ~seed:2. [| 4; 4 |];
    |]
    (at Two.on ~seed:1. [| 4; 4 |])

(* A chain of movements whose transposes cannot regroup a cut. *)
let chain r c =
  {
    f =
      (fun k x ->
        Nx.sum (Nx.mul (Nx.transpose (Nx.reshape [| r; c |] x)) k.(0)));
  }

let test_movements () =
  let check msg r c =
    placed_grad ~msg (chain r c)
      [| at (Two.split ~axis:0) [| c; r |] |]
      (at Two.on ~seed:1. [| r * c |])
  in
  check "a reshape and a transpose, uneven" 2 3;
  check "a reshape and a transpose, even" 2 4;
  placed_grad ~msg:"a slice"
    { f = (fun k x -> Nx.sum (Nx.mul (Nx.slice [ R (1, 4) ] x) k.(0))) }
    [| at (Two.split ~axis:0) [| 3; 2 |] |]
    (at Two.on ~seed:1. [| 5; 2 |])

(* The lines of [f]'s gradient program at [x] that place a cut value elsewhere:
   a gather. A place that reads a whole value, which each device views, is
   none. *)
let gathers f x =
  let on = Format.asprintf "%a" Nx.Placement.pp Two.on in
  let program = Rune.program T.tensor T.tensor (Rune.grad' f) x in
  let lines =
    String.split_on_char '\n' (Format.asprintf "%a" Rune.pp_program program)
  in
  (* The text of [l] after [sub], if [l] holds it. *)
  let after sub l =
    let k = String.length sub and n = String.length l in
    let rec go i =
      if i + k > n then None
      else if String.sub l i k = sub then
        Some (String.sub l (i + k) (n - i - k))
      else go (i + 1)
    in
    go 0
  in
  let gather l =
    match after "= Place (" l with
    | None -> false
    | Some operand -> Option.is_none (after (" at " ^ on ^ ")") operand)
  in
  List.filter gather lines

(* Data-parallel gradients: the parameters whole, the batch cut. A bias's or a
   mean's cotangent sums over the cut batch by partials, and the gradient lies
   whole without gathering an activation. *)
let test_data_parallel () =
  let x = at (Two.split ~axis:0) [| 8; 6 |] in
  let w1 = at Two.on ~seed:1. [| 6; 5 |] and b1 = at Two.on ~seed:2. [| 5 |] in
  let w2 = at Two.on ~seed:3. [| 5; 3 |] in
  let mlp (w1, (b1, w2)) =
    let y = Nx.matmul (Nx.tanh (Nx.add (Nx.matmul x w1) b1)) w2 in
    Nx.mean (Nx.mul y y)
  in
  let p = T.(pair tensor (pair tensor tensor)) in
  let g1, (gb, g2) = Rune.grad p mlp (w1, (b1, w2)) in
  let on = Format.asprintf "%a" Nx.Placement.pp Two.on in
  equal ~msg:"placements" (list string) [ on; on; on ]
    (List.map where_of [ g1; gb; g2 ]);
  let flat (w1, (b1, w2)) =
    Nx.concatenate ~axis:0 [ Nx.flatten w1; b1; Nx.flatten w2 ]
  in
  let split v =
    let w1 = Nx.reshape [| 6; 5 |] (Nx.slice [ R (0, 30) ] v) in
    let b1 = Nx.slice [ R (30, 35) ] v in
    (w1, (b1, Nx.reshape [| 5; 3 |] (Nx.slice [ R (35, 50) ] v)))
  in
  let check msg f b = equal ~msg (list string) [] (gathers f b) in
  check "an MLP" (fun v -> mlp (split v)) (flat (w1, (b1, w2)));
  let mu = at Two.on ~seed:4. [| 6 |] in
  check "a mean subtracted" (fun mu -> Nx.sum (Nx.square (Nx.sub x mu))) mu;
  check "a bias added" (fun b -> Nx.sum (Nx.mul (Nx.add x b) x)) mu;
  check "a broadcast read twice"
    (fun b ->
      let bb = Nx.broadcast_to [| 8; 6 |] b in
      Nx.add (Nx.sum (Nx.mul bb x)) (Nx.sum (Nx.mul bb bb)))
    mu

(* A tangent or a cotangent given at another placement than its primal's raises,
   naming both. *)
let test_given_elsewhere () =
  let x = at Two.on [| 4 |] and cut = at (Two.split ~axis:0) ~seed:1. [| 4 |] in
  let on = where_of x and split = where_of cut in
  let given name ~primal ~given =
    Invalid_argument
      (Printf.sprintf "%s: the root: at %s in the %s, at %s in the %s" name on
         primal split given)
  in
  raises (given "Rune.jvp'" ~primal:"parameters" ~given:"tangents") (fun () ->
      Rune.jvp' Nx.sin x cut);
  let _, back = Rune.vjp' Nx.sin x in
  raises (given "Rune.vjp'" ~primal:"result" ~given:"cotangents") (fun () ->
      back cut);
  let _, apply = Rune.linearize T.tensor T.tensor Nx.sin x in
  raises (given "Rune.linearize" ~primal:"parameters" ~given:"tangents")
    (fun () -> apply cut);
  let transpose =
    Rune.linear_transpose T.tensor T.tensor (fun x -> Nx.mul_s x 2.) x
  in
  raises (given "Rune.linear_transpose" ~primal:"result" ~given:"cotangents")
    (fun () -> transpose cut);
  let elsewhere x = Nx.place (Two.split ~axis:0) x in
  let f = Rune.custom_jvp T.tensor T.tensor (fun x -> (Nx.sin x, elsewhere)) in
  raises
    (Invalid_argument
       (Printf.sprintf
          "Rune.custom_jvp: the root: at %s in the tangent map's result, at %s \
           in the result"
          split on))
    (fun () -> Rune.jvp' f x x);
  let g = Rune.custom_vjp T.tensor T.tensor (fun x -> (Nx.sin x, elsewhere)) in
  raises
    (Invalid_argument
       (Printf.sprintf
          "Rune.custom_vjp: the root: at %s in the pullback's result, at %s in \
           the arguments"
          split on))
    (fun () -> Rune.grad' (fun x -> Nx.sum (g x)) x)

(* A Jacobian of a cut argument, whose basis lies at the argument's or the
   result's placement, holds the values it holds on the host. *)
let test_jacobians () =
  let x = at (Two.split ~axis:0) [| 4 |] in
  let host = Nx.place Nx.Host.on x in
  let check msg expected j =
    equal ~msg near (Nx.to_array expected) (Nx.to_array (Nx.place Nx.Host.on j))
  in
  check "jacfwd'" (Rune.jacfwd' Nx.sin host) (Rune.jacfwd' Nx.sin x);
  check "jacrev'" (Rune.jacrev' Nx.sin host) (Rune.jacrev' Nx.sin x)

(* The law: wherever nx routes [f]'s forward pass, over parameters and constants
   each whole or cut along an axis, its gradient lies at the parameter's
   placement, equals the gradient on the host, and agrees with finite
   differences. *)

type case = {
  name : string;
  fn : fn;
  shapes : int -> int -> int array * int array list;
      (** The parameter's shape and the constants', from [m] and [n]. *)
}

let law_cases =
  let open Nx in
  [
    {
      name = "x * a";
      fn = { f = (fun k x -> sum (mul x k.(0))) };
      shapes = (fun m n -> ([| m; n |], [ [| m; n |] ]));
    };
    {
      name = "a reduction's cotangent meets a cut";
      fn = rows;
      shapes = (fun m n -> ([| m; n |], [ [| m; n |]; [| m |] ]));
    };
    {
      name = "two contributions";
      fn = cuts;
      shapes = (fun m n -> ([| m; n |], [ [| m; n |]; [| m; n |] ]));
    };
    {
      name = "a reshape and a transpose";
      fn =
        {
          f =
            (fun k x ->
              let s = shape x in
              sum (mul (transpose (reshape [| s.(1); s.(0) |] x)) k.(0)));
        };
      shapes = (fun m n -> ([| m; n |], [ [| m; n |] ]));
    };
    {
      name = "a slice";
      fn =
        { f = (fun k x -> sum (mul (slice [ R (1, (shape x).(0)) ] x) k.(0))) };
      shapes = (fun m n -> ([| m + 1; n |], [ [| m; n |] ]));
    };
    {
      name = "a bias";
      fn = { f = (fun k x -> sum (mul (add k.(0) x) k.(0))) };
      shapes = (fun m n -> ([| n |], [ [| m; n |] ]));
    };
    {
      name = "exp of a broadcast";
      fn =
        {
          f = (fun k x -> sum (mul (exp (broadcast_to (shape k.(0)) x)) k.(0)));
        };
      shapes = (fun m n -> ([| n |], [ [| m; n |] ]));
    };
    {
      name = "a contraction";
      fn = { f = (fun k x -> sum (square (matmul x k.(0)))) };
      shapes = (fun _ n -> ([| 3; n |], [ [| n; 2 |] ]));
    };
  ]

(* Where an array lies: whole on both devices, or cut along an axis it has. *)
let placement_of s k =
  if k = 0 || Array.length s = 0 then Two.on
  else Two.split ~axis:(min (k - 1) (Array.length s - 1))

(* A case, [m] and [n], and where the parameter then each constant lies: [0]
   whole, [k] cut along axis [k - 1]. *)
let law_gen =
  let open Gen in
  let pp ppf (c, m, n, ks, _) =
    Format.fprintf ppf "%s, m %d, n %d, at %a" c.name m n
      Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
      ks
  in
  with_pp pp
    (let+ c = of_list law_cases
     and+ m = of_list [ 0; 1; 2; 3; 5 ]
     and+ n = of_list [ 1; 2; 3 ]
     and+ ks = list ~size:(constant 3) (int_range 0 2)
     and+ seed = int_range 0 99 in
     (c, m, n, ks, float_of_int seed))

let law_placed (c, m, n, ks, seed) =
  let xs, cs = c.shapes m n in
  let ks = Array.of_list ks in
  let x = at (placement_of xs ks.(0)) ~seed xs in
  let consts =
    Array.of_list
      (List.mapi
         (fun j s ->
           at (placement_of s ks.(j + 1)) ~seed:(seed +. 1. +. float_of_int j) s)
         cs)
  in
  (* A forward pass nx refuses has no backward pass to check. *)
  (match c.fn.f consts x with
  | exception Invalid_argument _ -> reject ()
  | _ -> ());
  let cut y = not (Nx.Placement.equal (Nx.placement y) Two.on) in
  cover c.name true;
  cover "a whole parameter beside a cut"
    ((not (cut x)) && Array.exists cut consts);
  cover "a cut parameter" (cut x);
  cover "uneven windows" (m = 5 && (cut x || Array.exists cut consts));
  cover "an empty window" (m = 1 && (cut x || Array.exists cut consts));
  cover "no element" (Nx.numel x = 0);
  placed_grad ~msg:"gradient" c.fn consts x;
  if Nx.numel x > 0 then
    equal ~msg:"finite differences" (result unit string) (Ok ())
      (Rune.check_grads T.tensor (c.fn.f consts) x)

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
    (Rune.check_grads T.tensor jump (v [| 0. |]));
  (* A slope of -1e6 prints as the shortest decimal that reads back, the same
     on every system: the C library's [%g] wrote 1e+06, 1e+006 under
     mingw-w64. *)
  let cliff x =
    Nx.sum (Nx.where (Nx.greater_s x 0.) x (Nx.add_s x (1e6 +. 1.)))
  in
  equal ~msg:"a cliff" (result unit string)
    (Error
       "Rune.check_grads: the directional derivative along ones is -1000000, \
        and the gradient predicts 1")
    (Rune.check_grads ~eps:0.5 T.tensor cliff (v [| 0. |]))

(* Convolutions *)

(* A convolution, or its transpose, over the spatial axes of [d], the extents of
   the convolution's input: [taps] the kernel's extents, [groups] groups of [cg]
   input and [og] output channels. *)
type conv_case = {
  transpose : bool;
  n : int;
  d : int array;
  taps : int array;
  stride : int array;
  padding : (int * int) array;
  dilation : int array;
  groups : int;
  cg : int;
  og : int;
  seed : float;
}

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat ";" (Array.to_list (Array.map string_of_int a)))

let pp_conv ppf c =
  let pad (lo, hi) = Printf.sprintf "(%d,%d)" lo hi in
  Format.fprintf ppf
    "%s n %d, d %a, taps %a, stride %a, padding [%s], dilation %a, %d groups \
     of %d in and %d out, seed %g"
    (if c.transpose then "conv_transpose" else "conv")
    c.n pp_ints c.d pp_ints c.taps pp_ints c.stride
    (String.concat ";" (Array.to_list (Array.map pad c.padding)))
    pp_ints c.dilation c.groups c.cg c.og c.seed

let conv_gen =
  let open Gen in
  let axes k g = array ~size:(constant k) g in
  (let* k = int_range 1 2 in
   let+ transpose = bool
   and+ n = of_list [ 0; 1; 1; 2 ]
   and+ d = axes k (int_range 0 4)
   and+ taps = axes k (of_list [ 0; 1; 2; 2; 3 ])
   and+ stride = axes k (int_range 1 3)
   and+ padding = axes k (pair (int_range (-1) 2) (int_range (-1) 2))
   and+ dilation = axes k (int_range 1 2)
   and+ groups = int_range 1 2
   and+ cg = int_range 1 2
   and+ og = int_range 1 2
   and+ seed = float_range 0. 6. in
   { transpose; n; d; taps; stride; padding; dilation; groups; cg; og; seed })
  |> with_pp pp_conv

(* Inputs padded to none of their elements: one window of [x] at stride 2,
   which the padding cuts out, and a batch of no image. *)
let conv_examples =
  [
    {
      transpose = false;
      n = 0;
      d = [| 0 |];
      taps = [| 1 |];
      stride = [| 1 |];
      padding = [| (1, 0) |];
      dilation = [| 1 |];
      groups = 1;
      cg = 2;
      og = 1;
      seed = 0.;
    };
    {
      transpose = true;
      n = 1;
      d = [| 0 |];
      taps = [| 1 |];
      stride = [| 2 |];
      padding = [| (1, 1) |];
      dilation = [| 1 |];
      groups = 1;
      cg = 2;
      og = 2;
      seed = 0.;
    };
  ]

let conv_of c x w =
  let op = if c.transpose then Nx.conv_transpose else Nx.conv in
  op ~stride:c.stride ~padding:c.padding ~dilation:c.dilation ~groups:c.groups x
    w

let filled seed s = Nx.create f64 s (spread seed (Array.fold_left ( * ) 1 s))

(* The operands of [c]: a transpose's [x] has the shape of the convolution of an
   input of extents [d]. *)
let conv_operands c =
  let w =
    filled (c.seed +. 1.)
      (Array.concat [ [| c.groups * c.og |]; c.taps; [| c.cg |] ])
  in
  let z =
    Nx.zeros f64 (Array.concat [ [| c.n |]; c.d; [| c.groups * c.cg |] ])
  in
  let x = if c.transpose then conv_of { c with transpose = false } z w else z in
  (filled c.seed (Nx.shape x), w)

(* The pullbacks of [<c̄, f (x, w)>] in [x] and in [w] together, each paired with
   a direction, against the central difference along it, which is exact but for
   rounding: a convolution is linear in each operand. *)
let conv_law pullback c =
  let x, w = conv_operands c in
  cover "a transpose" c.transpose;
  cover "groups" (c.groups > 1);
  cover "one channel in and out a group" (c.cg = 1 && c.og = 1);
  cover "a stride past 1" (Array.exists (fun s -> s > 1) c.stride);
  cover "a cutting padding"
    (Array.exists (fun (lo, hi) -> lo < 0 || hi < 0) c.padding);
  cover "two spatial axes" (Array.length c.d = 2);
  cover "a kernel of no tap" (Array.mem 0 c.taps);
  cover "a dilation past 1" (Array.exists (fun d -> d > 1) c.dilation);
  (* A contraction of windows over a padding of [x] that keeps none of its
     elements. *)
  let pads = Array.exists (fun p -> p <> (0, 0)) c.padding in
  let none =
    c.n = 0
    || Array.exists2
         (fun d (lo, hi) -> d + Int.min lo 0 + Int.min hi 0 <= 0)
         c.d c.padding
  in
  cover "a conv's padding that keeps no element"
    ((not c.transpose) && pads && none
    && (not (Array.mem 0 c.taps))
    && not (c.cg = 1 && c.og = 1));
  let y = conv_of c x w in
  cover "a zero-size result" (Nx.numel y = 0);
  let ct = filled (c.seed +. 2.) (Nx.shape y) in
  let gx, gw = pullback c x w ct in
  equal ~msg:"x's cotangent shape" (array int) (Nx.shape x) (Nx.shape gx);
  equal ~msg:"w's cotangent shape" (array int) (Nx.shape w) (Nx.shape gw);
  let loss x w = Nx.item [] (Nx.sum (Nx.mul ct (conv_of c x w))) in
  let mass a = sum_abs (Nx.to_array a) in
  let size a b = mass ct *. a *. b in
  let u = filled (c.seed +. 3.) (Nx.shape x) in
  let fd = (loss (Nx.add x u) w -. loss (Nx.sub x u) w) /. 2. in
  adjoint ~msg:"x's pullback"
    ~size:(size (mass x +. mass u) (mass w))
    [| fd |]
    (Nx.to_array (Nx.mul gx u));
  let u = filled (c.seed +. 4.) (Nx.shape w) in
  let fd = (loss x (Nx.add w u) -. loss x (Nx.sub w u)) /. 2. in
  adjoint ~msg:"w's pullback"
    ~size:(size (mass x) (mass w +. mass u))
    [| fd |]
    (Nx.to_array (Nx.mul gw u))

let eager_pullback c x w ct =
  let _, pb = Rune.vjp (two ()) T.tensor (fun (x, w) -> conv_of c x w) (x, w) in
  pb ct

let compiled_pullback c x w ct =
  Rune.jit
    T.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
    (fun x w ct -> eager_pullback c x w ct)
    x w ct

(* Scans of a constant

   The running sum, product, maximum or minimum along [x] of the constant 2,
   continued from the carry [c]: the program reads none of its load, so [y]
   varies with [c] alone. [dc] is [∂y_i / ∂c]: for [x] of 3 elements, 1 for a
   sum, [2^(i + 1)] for a product, and for an extreme 1 where [c] beats 2 and 0
   where 2 beats [c]. Where not [loaded], the scan has no load, and where
   [half], it scans float16 from a float32 carry, its carrier. *)

type constant_scan = {
  name : string;
  monoid : Nx_kernel.monoid;
  loaded : bool;
  half : bool;
  c : float;
  dc : float array;
}

let constant_scans =
  let ones = [| 1.; 1.; 1. |] and zeros = [| 0.; 0.; 0. |] in
  let case ?(loaded = true) ?(half = false) name monoid c dc =
    { name; monoid; loaded; half; c; dc }
  in
  [
    case "a sum" Sum 3. ones;
    case "a product" Prod 3. [| 2.; 4.; 8. |];
    case "a maximum at the carry" Max 3. ones;
    case "a maximum at the elements" Max 1. zeros;
    case "a minimum at the carry" Min 1. ones;
    case "a minimum at the elements" Min 3. zeros;
    case "a float16 product" ~half:true Prod 3. [| 2.; 4.; 8. |];
    case "a sum of no load" ~loaded:false Sum 3. ones;
    case "a product of no load" ~loaded:false Prod 3. [| 2.; 4.; 8. |];
    case "a maximum of no load" ~loaded:false Max 3. ones;
  ]

let scan_in (type s r) (dt : (float, s) D.t) (wide : (float, r) D.t) s (x, c) =
  let x = Nx.cast dt x and c = Nx.cast wide c in
  let ins = Iarray.of_list (if s.loaded then [ D.Any dt ] else []) in
  let prog = P.of_exprs ~ins [| P.E.any (P.E.const dt 2.) |] in
  Nx.cast f64
    (Nx.Prim.eval ~by:"test"
       (Scan
          {
            layout = contiguous (Nx.shape x);
            axis = 0;
            prog;
            reduction = Monoid (s.monoid, 0, dt);
            loads = (if s.loaded then [| Plain x |] else [||]);
            carry = Some (Any c);
            at = (if s.loaded then None else Some (Nx.placement c));
          }))

let constant_scan s =
  if s.half then scan_in Nx.float16 Nx.float32 s else scan_in f64 f64 s

(* The tangent along [(1, 1)] is [dc], and the pullback of [w] is [(0, Σ w_i
   dc_i)], eager and compiled. *)
let test_constant_scan s =
  let f = constant_scan s in
  let x = v [| 5.; 6.; 7. |] and c = v [| s.c |] in
  let w = v [| 1.; 10.; 100. |] in
  let jvp x c =
    snd (Rune.jvp (two ()) T.tensor f (x, c) (Nx.ones_like x, Nx.ones_like c))
  in
  let vjp x c w = snd (Rune.vjp (two ()) T.tensor f (x, c)) w in
  let jit_jvp = Rune.jit T.(tensor @-> tensor @-> returns tensor) jvp
  and jit_vjp =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
      vjp
  in
  let gc = [| Nx.item [] (Nx.sum (Nx.mul w (v s.dc))) |] in
  let pullback what (gx, gc') =
    equal ~msg:(what ^ " in x") (array float_exact) [| 0.; 0.; 0. |]
      (Nx.to_array gx);
    equal ~msg:(what ^ " in c") (array float_exact) gc (Nx.to_array gc')
  in
  equal ~msg:"tangent" (array float_exact) s.dc (Nx.to_array (jvp x c));
  equal ~msg:"compiled tangent" (array float_exact) s.dc
    (Nx.to_array (jit_jvp x c));
  pullback "pullback" (vjp x c w);
  pullback "compiled pullback" (jit_vjp x c w)

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
      | Starts p -> starts_with ~prefix:p m)

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

(* Depthwise convolutions

   A convolution of one input and one output channel a group, and its
   transpose, whose loop spreads its input: their pullbacks in the input and
   the kernel, eagerly and compiled, pair with any tangent as the exact
   change of the values, the convolutions being linear in each, and equal the
   pullbacks of the same convolution by a kernel zero off its diagonal, which
   nx computes as a contraction; at strides, paddings that pad or cut,
   dilations, axes of one element and kernels that reach no element. *)

type depthwise = {
  transpose : bool;
  batch : int;
  channels : int;
  spatial : int array;
  taps : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
  seed : float;
}

let pp_semi ppf () = Format.pp_print_string ppf ";"
let pp_ints = Format.pp_print_array ~pp_sep:pp_semi Format.pp_print_int

let pp_depthwise ppf d =
  let pair ppf (lo, hi) = Format.fprintf ppf "(%d, %d)" lo hi in
  Format.fprintf ppf
    "%s n %d c %d spatial [%a] taps [%a] stride [%a] dilation [%a] padding \
     [%a]"
    (if d.transpose then "conv_transpose" else "conv")
    d.batch d.channels pp_ints d.spatial pp_ints d.taps pp_ints d.stride
    pp_ints d.dilation
    (Format.pp_print_array ~pp_sep:pp_semi pair)
    d.padding

(* The windows a padded axis of [n] elements holds. *)
let windows_of n (lo, hi) k s d =
  let span = (d * (k - 1)) + 1 and e = n + lo + hi in
  if e < span then 0 else ((e - span) / s) + 1

let depthwise_gen =
  let open Gen in
  let axis =
    let+ n = int_range 1 5
    and+ k = int_range 1 3
    and+ s = int_range 1 3
    and+ d = int_range 1 2
    and+ lo = int_range (-1) 2
    and+ hi = int_range (-1) 2 in
    (n, k, s, d, (lo, hi))
  in
  with_pp pp_depthwise
  @@ let+ transpose = bool
     and+ batch = int_range 1 2
     and+ channels = int_range 2 3
     and+ axes = list ~size:(int_range 1 2) axis
     and+ seed = float_range 0. 6. in
     let axes = Array.of_list axes in
     let get f = Array.map f axes in
     {
       transpose;
       batch;
       channels;
       spatial = get (fun (n, _, _, _, _) -> n);
       taps = get (fun (_, k, _, _, _) -> k);
       stride = get (fun (_, _, s, _, _) -> s);
       dilation = get (fun (_, _, _, d, _) -> d);
       padding = get (fun (_, _, _, _, p) -> p);
       seed;
     }

let depthwise_law d =
  let counts =
    Array.init (Array.length d.spatial) (fun i ->
        windows_of d.spatial.(i) d.padding.(i) d.taps.(i) d.stride.(i)
          d.dilation.(i))
  in
  cover "a window" (Array.for_all (fun t -> t > 0) counts);
  cover "an axis of one element" (Array.mem 1 d.spatial);
  cover "a stride" (Array.exists (fun s -> s > 1) d.stride);
  cover "a dilation" (Array.exists (fun s -> s > 1) d.dilation);
  cover "a padding that cuts"
    (Array.exists (fun (lo, hi) -> lo < 0 || hi < 0) d.padding);
  cover "conv_transpose" d.transpose;
  (* conv_transpose's input is a convolution's result, of [counts]. *)
  if d.transpose && Array.mem 0 counts then ()
  else begin
    cover "no window" (Array.mem 0 counts);
    let c = d.channels in
    let x_spatial = if d.transpose then counts else d.spatial in
    let xs = Array.concat [ [| d.batch |]; x_spatial; [| c |] ] in
    let ws = Array.concat [ [| c |]; d.taps; [| 1 |] ] in
    let filled seed s =
      Nx.create f64 s (spread seed (Array.fold_left ( * ) 1 s))
    in
    let x = filled d.seed xs and w = filled (d.seed +. 1.) ws in
    let conv ~groups x w =
      let stride = d.stride and dilation = d.dilation and padding = d.padding in
      if d.transpose then
        Nx.conv_transpose ~groups ~stride ~dilation ~padding x w
      else Nx.conv ~groups ~stride ~dilation ~padding x w
    in
    let depthwise (x, w) = conv ~groups:c x w in
    (* [w] on the diagonal of a kernel of [c] input channels. *)
    let diagonal (x, w) =
      let eye =
        Nx.reshape
          (Array.concat [ [| c |]; Array.map (fun _ -> 1) d.taps; [| c |] ])
          (Nx.eye f64 c)
      in
      let square = Array.concat [ [| c |]; d.taps; [| c |] ] in
      conv ~groups:1 x (Nx.mul (Nx.broadcast_to square w) eye)
    in
    let y = depthwise (x, w) in
    let ybar = filled (d.seed +. 2.) (Nx.shape y) in
    let pair = T.(pair tensor tensor) in
    let pullback f (x, w) ybar =
      let _, pb = Rune.vjp pair T.tensor f (x, w) in
      pb ybar
    in
    let compiled f =
      Rune.jit
        T.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
        (fun x w ybar -> pullback f (x, w) ybar)
    in
    let sum_abs t = sum_abs (Nx.to_array t) in
    let size =
      sum_abs ybar *. (sum_abs x +. sum_abs w +. 1.) *. (sum_abs w +. 1.)
    in
    let close msg a b =
      let a = Nx.to_array a and b = Nx.to_array b in
      equal ~msg:(msg ^ " length") int (Array.length a) (Array.length b);
      Array.iteri
        (fun i ai ->
          at_most ~msg float_exact
            ~than:((0x1p-40 *. size) +. 0x1p-1000)
            (Float.abs (ai -. b.(i))))
        a
    in
    let pair_with a b = Nx.item [] (Nx.sum (Nx.mul a b)) in
    let check how (xbar, wbar) =
      (* The exact change along a tangent, that of a linear map. *)
      let u = filled (d.seed +. 3.) xs and v = filled (d.seed +. 4.) ws in
      let change f = pair_with ybar (Nx.sub f y) in
      at_most ~msg:(how ^ ": <x̄, u> is the change along u") float_exact
        ~than:((0x1p-40 *. size *. 4.) +. 0x1p-1000)
        (Float.abs (pair_with xbar u -. change (depthwise (Nx.add x u, w))));
      at_most ~msg:(how ^ ": <w̄, v> is the change along v") float_exact
        ~than:((0x1p-40 *. size *. 4.) +. 0x1p-1000)
        (Float.abs (pair_with wbar v -. change (depthwise (x, Nx.add w v))));
      (* The contraction's pullback in both of conv_transpose's arguments
         raises: one at a time. *)
      let in_x x = diagonal (x, w) and in_w w = diagonal (x, w) in
      let pull f a = snd (Rune.vjp' f a) ybar in
      close (how ^ ": x̄ is the contraction's") xbar (pull in_x x);
      close (how ^ ": w̄ is the contraction's") wbar (pull in_w w)
    in
    check "eager" (pullback depthwise (x, w) ybar);
    check "compiled" ((compiled depthwise) x w ybar)
  end

(* An element of the input that one window reads receives its term, [-0.]
   included, and one no window reads [+0.]; a tap that reads padding gives
   no term, even of an infinite weight. *)
let test_depthwise_zeros () =
  let x = Nx.create f64 [| 1; 4; 1 |] [| 1.; 2.; 3.; 4. |] in
  let check msg want ~stride ~padding w =
    let w = Nx.create f64 [| 1; Array.length w; 1 |] w in
    let f x =
      let y = Nx.conv ~groups:1 ~stride ~padding x w in
      Nx.sum (Nx.mul y (Nx.full_like y (-0.)))
    in
    equal ~msg signed want (with_signs (Rune.grad' f x));
    let g = Rune.jit T.(tensor @-> returns tensor) (Rune.grad' f) in
    equal ~msg:(msg ^ ", compiled") signed want (with_signs (g x))
  in
  let neg = (true, -0.) and pos = (false, 0.) in
  check "windows apart" [| neg; neg; neg; neg |] ~stride:[| 2 |]
    ~padding:[| (0, 0) |] [| 1.; 1. |];
  check "taps apart" [| neg; pos; neg; pos |] ~stride:[| 2 |]
    ~padding:[| (0, 0) |] [| 1. |];
  (* Element 3 is tap 1 of window 3 alone; tap 0 of a fourth window would
     read past the padding. *)
  let w = Nx.create f64 [| 1; 2; 1 |] [| Float.infinity; 2. |] in
  let ybar = Nx.create f64 [| 1; 4; 1 |] [| 1.; 1.; 1.; 3. |] in
  let pull x =
    let _, pb =
      Rune.vjp' (fun x -> Nx.conv ~padding:[| (1, 0) |] x w) x
    in
    pb ybar
  in
  let last g = Nx.item [ 0; 3; 0 ] g in
  equal ~msg:"the last element's term alone" float_exact 6. (last (pull x));
  let g = Rune.jit T.(tensor @-> returns tensor) pull in
  equal ~msg:"compiled" float_exact 6. (last (g x))

let () =
  exit
    (run "rune grad"
       [
         group "adjoint" adjoint_props;
         group "complex adjoint" complex_props;
         group "jacrev' is jacfwd' at ties and zeros" kink_props;
         group "finite gradients of composed functions" finite_cases;
         prop "hypot's gradient is (x / r, y / r)" hypot_gen hypot_law;
         prop "softmax's and log_softmax's gradients" normalised_gen
           normalised_law;
         prop "softmax's and log_softmax's gradients under where" where_gen
           where_law;
         prop "under where at float32, as float64's" where_gen
           (where_narrow_law Nx.float32 0x1p-24);
         prop "under where at float16, as float64's" where_gen
           (where_narrow_law Nx.float16 0x1p-11);
         prop "under where at bfloat16, as float64's" where_gen
           (where_narrow_law Nx.bfloat16 0x1p-8);
         group "depthwise convolutions"
           [
             prop "pullbacks of a depthwise conv" depthwise_gen depthwise_law;
             test "signs of zero and padded taps" test_depthwise_zeros;
           ];
         group "conventions"
           [
             test "complex gradients" test_complex;
             test "kinks" test_kinks;
             test "softmax's and log_softmax's compiled gradients"
               test_normalised_jit;
             test "softmax's and log_softmax's mapped gradients"
               test_normalised_vmap;
             test "values alone at repeated values" test_repeated_values;
             prop "softplus's gradient is sigmoid" ~examples:softplus_specials
               softplus_points softplus_law;
             test "selections" test_selections;
             test "signs of zero" test_zeros;
             test "signs of zero through reads of regions" test_region_zeros;
             prop "reads of regions sum each element's contributions"
               reads_gen regions_law;
             test "a truncated normal's bounds" test_draws;
             prop "a dirichlet draw's tangents and gradient in its \
                   concentrations" dirichlet_gen dirichlet_law;
             test "a dirichlet draw's gradient against differences"
               test_dirichlet_differences;
             test "the logarithms' gradients below zero too" test_logarithms;
             prop "sinc's derivative, near 0 and at integers too" sinc_gen
               sinc_law;
             test "sinc's derivatives at 0 are its series'" test_sinc_at_zero;
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
             test "through a Q8_0 product" (test_quantised Quants.Q8_0);
             test "through a Q6_K product" (test_quantised Quants.Q6_K);
             test "through a product from an init" test_quantised_init;
             test "nesting" test_nesting;
             test "a read reads the primal" test_read;
             test "detach" test_detach;
             test "totals" test_total;
             test "placement" test_placement;
             test "a take of split positions" test_cut_take;
             test "jacrev'" test_jacrev;
             test "check_grads" test_check_grads;
           ];
         group "convolutions"
           [
             prop "pullbacks in both operands" ~examples:conv_examples conv_gen
               (conv_law eager_pullback);
             prop "compiled pullbacks in both operands" ~examples:conv_examples
               conv_gen
               (conv_law compiled_pullback);
           ];
         cases "scans of a constant from a carry"
           ~id:(fun s -> s.name)
           constant_scans test_constant_scan;
         group "placement"
           [
             test "a reduction's cotangent meets a cut" test_cut_residual;
             test "two contributions cut differently" test_two_cuts;
             test "movements" test_movements;
             test "data parallel" test_data_parallel;
             test "given elsewhere" test_given_elsewhere;
             test "Jacobians" test_jacobians;
             prop ~count:200 "where nx routes the forward pass" law_gen
               law_placed;
           ];
         cases "errors" ~id:(fun (n, _, _) -> n) errors test_error;
         stateful ~domains:2 ~count:30 ~steps:8
           "results on two domains are one domain's" domain_commands;
         group "jit"
           [
             prop "grad of jit and jit of grad of exact functions" jit_cases
               law_jit;
           ];
         group "memory"
           [ test "a map's residuals are its loads" test_residuals ];
       ])
