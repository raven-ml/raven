(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jvp: tangents against central differences at float64 for every kind of
   scalar program and every operation that has a derivative, complex functions
   as maps of their parts, the kinks' tangents, jvp of a compiled function
   against jvp, and a map's tangent computed with its primal in one kernel. *)

open Windtrap
open Compose
open Cases
module Rune = Rune2

let h = 1e-6

let law c (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let y, dy = Rune.jvp' c.f x t in
  equal ~msg:"primal" bits_t (bits (c.f x)) (bits y);
  let at e =
    Nx.to_array
      (c.f (Nx.create f64 s (Array.mapi (fun k x -> x +. (e *. ts.(k))) xs)))
  in
  let plus = at h and minus = at (-.h) in
  let dy = Nx.to_array dy in
  equal ~msg:"tangent count" int (Array.length plus) (Array.length dy);
  Array.iteri
    (fun k d ->
      let fd = (plus.(k) -. minus.(k)) /. (2. *. h) in
      (* Differences of non-finite values say nothing. *)
      if
        Float.is_finite plus.(k)
        && Float.is_finite minus.(k)
        && Float.is_finite fd
      then
        less
          ~msg:(Printf.sprintf "tangent %d: %h, differences %h" k d fd)
          float_exact
          ~than:(1e-5 *. (1. +. Float.abs fd))
          (Float.abs (d -. fd)))
    dy

(* Helpers over cases' operands *)

let tangent_props =
  List.map
    (fun c -> prop c.name (gen c) (law c))
    (real_cases @ contract_cases @ factor_cases @ complex_cases
   @ complex_factor_cases)

(* Values the conventions state *)

let floats = array (float 1e-12)
let tangent f x t = Nx.to_array (snd (Rune.jvp' f x t))

(* JAX's: [abs] passes the tangent at either zero; tied operands of an extreme
   share it by halves, and a NaN takes none; tied elements along a reduction's
   axes share it equally; a running extreme takes the first element's. *)
let test_kinks () =
  equal ~msg:"abs" floats [| 1.; 1.; 1.; -1. |]
    (tangent Nx.abs (v [| 0.; -0.; 2.; -2. |]) (v [| 1.; 1.; 1.; 1. |]));
  let pair = T.(pair tensor tensor) in
  let extreme f x y tx ty =
    Nx.to_array
      (snd
         (Rune.jvp pair T.tensor (fun (a, b) -> f a b) (v x, v y) (v tx, v ty)))
  in
  equal ~msg:"maximum" floats [| 2.; 1.; 3.; 0. |]
    (extreme Nx.maximum
       [| 1.; 2.; 0.; Float.nan |]
       [| 1.; 0.; 2.; 1. |] [| 1.; 1.; 1.; 1. |] [| 3.; 3.; 3.; 3. |]);
  equal ~msg:"minimum" floats [| 2.; 3.; 1.; 0. |]
    (extreme Nx.minimum [| 1.; 2.; 0.; 1. |]
       [| 1.; 0.; 2.; Float.nan |]
       [| 1.; 1.; 1.; 1. |] [| 3.; 3.; 3.; 3. |]);
  equal ~msg:"max over an axis" floats [| 3. |]
    (tangent (Nx.max ~keepdims:true) (v [| 1.; 3.; 3. |]) (v [| 1.; 2.; 4. |]));
  equal ~msg:"min over an axis" floats [| 3. |]
    (tangent (Nx.min ~keepdims:true) (v [| 2.; 1.; 1. |]) (v [| 5.; 2.; 4. |]));
  equal ~msg:"cummax" floats [| 1.; 2.; 2.; 2. |]
    (tangent (Nx.cummax ~axis:0)
       (v [| 1.; 3.; 3.; 2. |])
       (v [| 1.; 2.; 4.; 8. |]));
  equal ~msg:"pow at a zero base" floats [| 0.; 0.; 0. |]
    (tangent
       (fun y -> Nx.pow (v [| 0.; 0.; 0. |]) y)
       (v [| -1.; 2.; 0.5 |])
       (v [| 1.; 1.; 1. |]));
  equal ~msg:"pow at a zero exponent" floats [| 0.; 0.; 0. |]
    (Nx.to_array
       (snd
          (Rune.jvp pair T.tensor
             (fun (a, b) -> Nx.pow a b)
             (v [| 0.; 1e-310; 2. |], v [| 0.; 0.; 0. |])
             (v [| 1.; 1.; 1. |], v [| 0.; 0.; 0. |]))));
  (* The tangent along the base is [b a^(b - 1)] at [b = 0] too: its derivative
     along the exponent there is [1 / a], and along the base [0]. *)
  let a = v [| 0.1; 0.5; 2.; 1e-310; Float.infinity |] in
  let ones = v [| 1.; 1.; 1.; 1.; 1. |]
  and zeros = v [| 0.; 0.; 0.; 0.; 0. |] in
  let along_base a b = snd (Rune.jvp' (fun a -> Nx.pow a b) a ones) in
  equal ~msg:"pow's mixed second derivative at a zero exponent" floats
    [| 10.; 2.; 0.5; Float.infinity; 0. |]
    (tangent (along_base a) zeros ones);
  equal ~msg:"pow's second derivative along the base at a zero exponent" floats
    [| 0.; 0.; 0.; 0.; 0. |]
    (tangent
       (fun a -> along_base a zeros)
       (v [| 1e-200; 0.5; 2.; 1e-310; Float.infinity |])
       ones);
  equal ~msg:"a real sign" floats [| 0.; 0. |]
    (tangent Nx.sign (v [| 0.; 2. |]) (v [| 1.; 1. |]))

(* Values the derivatives' definitions give at the edges: zeros, infinities,
   narrow dtypes whose rounding of a primal must not reach its tangent, a
   reduction into another dtype, ties stated by the kinks' rules. *)
let test_edges () =
  let tangent' f x t = Nx.to_array (Nx.cast f64 (snd (Rune.jvp' f x t))) in
  let close tol = array (float tol) in
  equal ~msg:"prod at a zero" floats [| 6. |]
    (tangent (Nx.prod ~keepdims:true) (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
  equal ~msg:"cumprod at a zero" floats [| 1.; 2.; 6. |]
    (tangent Nx.cumprod (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
  equal ~msg:"cumprod's first at an infinity" floats [| 1. |]
    [| (tangent Nx.cumprod (v [| Float.infinity; 1. |]) (v [| 1.; 0. |])).(0) |];
  equal ~msg:"logcumsumexp after -inf" floats [| 1.; 2.; 5. |]
    (tangent Nx.logcumsumexp
       (v [| Float.neg_infinity; Float.neg_infinity; 0. |])
       (v [| 1.; 3.; 5. |]));
  equal ~msg:"logcumsumexp from -inf" floats [| 0.; 1. |]
    (tangent Nx.logcumsumexp (v [| Float.neg_infinity; 0. |]) (v [| 0.; 1. |]));
  let narrow dt xs = Nx.cast dt (v xs) in
  let bf = Nx.bfloat16 and h = Nx.float16 in
  equal ~msg:"logsumexp shifted, bfloat16" (array float_exact) [| 1. |]
    (tangent'
       (Nx.logsumexp ~keepdims:false)
       (narrow bf [| 100.; 100. |])
       (narrow bf [| 1.; 1. |]));
  equal ~msg:"var shifted, float16" (array float_exact) [| 0. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.var x))
       (narrow h [| 2000.; 2001. |])
       (narrow h [| 1.; 1. |]));
  equal ~msg:"var along one, float16" (array float_exact) [| 0.5 |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.var x))
       (narrow h [| 2000.; 2001. |])
       (narrow h [| 0.; 1. |]));
  (* More ties than float16 counts: their tangents' mean is taken in float32. *)
  let ones = Nx.ones h [| 65536 |] in
  equal ~msg:"max's 65536 ties, float16" (array float_exact) [| 1. |]
    (tangent' (fun x -> Nx.reshape [| 1 |] (Nx.max x)) ones ones);
  equal ~msg:"logsumexp's 65536 infinite terms, float16" (array float_exact)
    [| 1. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.logsumexp x))
       (Nx.full h [| 65536 |] Float.infinity)
       ones);
  equal ~msg:"a scatter's 65536 tied updates, float16" (array float_exact)
    [| 1. |]
    (tangent'
       (fun u ->
         Nx.scatter ~combine:Max ~axis:0
           (Nx.zeros Nx.int64 [| 65536 |])
           u (Nx.zeros h [| 1 |]))
       ones ones);
  equal ~msg:"prod in its carrier, float16" (close 1.) [| 300. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.prod x))
       (narrow h [| 300.; 300.; 1. /. 300.; 1. /. 300. |])
       (narrow h [| 0.; 0.; 1.; 0. |]));
  (* A float64 output reduced into float32, and into int32. *)
  let into dt x =
    fst
      (Nx.Prim.eval ~by:"test"
         (Reduce
            {
              layout = contiguous (Nx.shape x);
              axes = [| 0 |];
              prog = identity;
              reductions = [ Monoid (Max, 0, dt) ];
              loads = [| Plain x |];
            }))
  in
  equal ~msg:"max into float32" (array float_exact) [| 2. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (into Nx.float32 x))
       (v [| 0.1; 0.2 |])
       (v [| 1.; 2. |]));
  let prod_into dt x =
    fst
      (Nx.Prim.eval ~by:"test"
         (Reduce
            {
              layout = contiguous (Nx.shape x);
              axes = [| 0 |];
              prog = identity;
              reductions = [ Monoid (Prod, 0, dt) ];
              loads = [| Plain x |];
            }))
  in
  equal ~msg:"prod into float32, in the output's carrier" (float 1e24) 1e30
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (prod_into Nx.float32 x))
       (v [| 1e30; 1e30; 1e-30; 1e-30 |])
       (v [| 0.; 0.; 1.; 0. |])).(0);
  equal ~msg:"max into int32" (array int32) [| 0l |]
    (Nx.to_array
       (snd
          (Rune.jvp'
             (fun x -> Nx.reshape [| 1 |] (into Nx.int32 x))
             (v [| 1.5; 2.5 |])
             (v [| 1.; 2. |]))));
  (* Ties: a scatter's target with the update that ties it, an arg's extreme, a
     running maximum's carry against the element tying it. *)
  let s = T.(pair tensor tensor) in
  let _, t =
    Rune.jvp s T.tensor
      (fun (into, u) -> Nx.scatter ~combine:Max ~axis:0 (idx [| 1; 0 |]) u into)
      (v [| 1.; 5. |], v [| 5.; 0.5 |])
      (v [| 1.; 2. |], v [| 4.; 8. |])
  in
  equal ~msg:"scatter max" floats [| 1.; 3. |] (Nx.to_array t);
  let arg x =
    Nx.reshape [| 1 |] (fst (fst (reduce1 [ Arg (Max, 0, f64) ] x)))
  in
  equal ~msg:"arg's extreme" floats [| 3. |]
    (tangent arg
       (Nx.reshape [| 1; 3 |] (v [| 1.; 3.; 3. |]))
       (Nx.reshape [| 1; 3 |] (v [| 1.; 2.; 4. |])));
  let running x =
    let c, r = carried x in
    Nx.reshape [| 2 |] (scan1 ~carry:c (Monoid (Max, 0, f64)) r)
  in
  equal ~msg:"running max from a tying carry" floats [| 9.; 9. |]
    (tangent running
       (Nx.reshape [| 1; 3 |] (v [| 3.; 3.; 1. |]))
       (Nx.reshape [| 1; 3 |] (v [| 9.; 5.; 7. |])));
  (* What has no derivative raises. *)
  let tall = Nx.create f64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |] in
  invalid
    ~sub:
      "Rune.jvp': the tangent of a complete QR factorisation of a tall matrix \
       has no definition" (fun () ->
      Rune.jvp' (fun a -> fst (Nx.qr ~factors:Complete a)) tall tall);
  invalid
    ~sub:
      "Rune.jvp': the tangent of a complete SVD of a non-square matrix has no \
       definition" (fun () ->
      Rune.jvp'
        (fun a ->
          let _, s, _ = Nx.svd ~factors:Complete f64 a in
          s)
        tall tall);
  invalid ~sub:"Rune.jvp': a contraction's scale has a tangent" (fun () ->
      Rune.jvp'
        (fun s ->
          Nx.Prim.eval ~by:"test"
            (Contract
               {
                 batch = [||];
                 contracting = [| (1, 0) |];
                 acc = any;
                 out = f64;
                 a = v [| 1.; 2. |] |> Nx.reshape [| 1; 2 |];
                 b = Nx.reshape [| 2; 1 |] (v [| 3.; 4. |]);
                 init = None;
                 a_scale = Some (Value s);
                 b_scale = None;
                 segments = None;
               }))
        (Nx.reshape [| 1; 2 |] (v [| 1.; 1. |]))
        (Nx.reshape [| 1; 2 |] (v [| 1.; 1. |])))

(* A tangent is the directional derivative: [f'(z) v] for a holomorphic [f],
   [conj v] for the conjugate, [Re (conj z v) / |z|] for the modulus. *)
let test_complex () =
  let z =
    Nx.create c128 [| 2 |] [| { re = 0.5; im = -1. }; { re = -2.; im = 0.25 } |]
  in
  let t =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -0.5; im = 1. } |]
  in
  let close = array (pair (float 1e-12) (float 1e-12)) in
  let parts z =
    Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array z)
  in
  equal ~msg:"exp" close
    (parts (Nx.mul (Nx.exp z) t))
    (parts (snd (Rune.jvp' Nx.exp z t)));
  equal ~msg:"mul by a constant" close
    (parts (Nx.mul z t))
    (parts (snd (Rune.jvp' (fun w -> Nx.mul w z) z t)));
  equal ~msg:"conjugate" close
    (parts (Nx.conjugate t))
    (parts (snd (Rune.jvp' Nx.conjugate z t)));
  let expected =
    Array.map2
      (fun (z : Complex.t) (t : Complex.t) ->
        ((z.re *. t.re) +. (z.im *. t.im)) /. Complex.norm z)
      (Nx.to_array z) (Nx.to_array t)
  in
  equal ~msg:"magnitude" floats expected
    (Nx.to_array (snd (Rune.jvp' (Nx.magnitude f64) z t)))

(* The gauges factorisations' tangents keep, and their first-order
   reconstructions. *)
let test_factorisations () =
  let x =
    Nx.create f64 [| 3; 3 |]
      [| 0.3; -0.2; 0.5; 0.1; 0.4; -0.6; 0.2; 0.7; -0.1 |]
  in
  let t =
    Nx.create f64 [| 3; 3 |] [| 1.; 0.5; -0.3; 0.2; -1.; 0.4; 0.6; 0.1; 0.8 |]
  in
  let diagonal_zero ~msg m =
    equal ~msg
      (array (float 1e-12))
      [| 0.; 0.; 0. |]
      (Nx.to_array (Nx.diagonal m))
  in
  let pair = T.(pair tensor tensor) in
  let a = spaced (symmetric x) in
  let (_, v), (_, dv) =
    Rune.jvp T.tensor pair (fun a -> Nx.eigh f64 a) a (symmetric t)
  in
  diagonal_zero ~msg:"eigh: vᵀ dv" (Nx.matmul (Nx.matrix_transpose v) dv);
  let a = spaced x in
  let triple = T.(pair tensor (pair tensor tensor)) in
  let (u, (s, vh)), (du, (ds, dvh)) =
    Rune.jvp T.tensor triple
      (fun a ->
        let u, s, vh = Nx.svd f64 a in
        (u, (s, vh)))
      a t
  in
  diagonal_zero ~msg:"svd: vh dvhᵀ" (Nx.matmul vh (Nx.matrix_transpose dvh));
  let scaled u s = Nx.mul u (Nx.unsqueeze ~axes:[ 0 ] s) in
  let first_order =
    Nx.add
      (Nx.add (Nx.matmul (scaled du s) vh) (Nx.matmul (scaled u ds) vh))
      (Nx.matmul (scaled u s) dvh)
  in
  equal ~msg:"svd: da = d(u s vh)"
    (array (float 1e-12))
    (Nx.to_array t) (Nx.to_array first_order);
  let (_, v), (_, dv) =
    Rune.jvp T.tensor
      T.(pair tensor tensor)
      (fun a -> Nx.eig c128 a)
      (spaced (Nx.mul_s x 0.3))
      t
  in
  let re m = Nx.to_array (Nx.real f64 m)
  and im m = Nx.to_array (Nx.imag f64 m) in
  let g = Nx.diagonal (Nx.matmul (adjoint v) dv) in
  equal ~msg:"eig: vᴴ dv, real" (array (float 1e-10)) [| 0.; 0.; 0. |] (re g);
  equal ~msg:"eig: vᴴ dv, imaginary"
    (array (float 1e-10))
    [| 0.; 0.; 0. |] (im g)

(* jvp of a compiled function *)

type jit_case =
  | Test_case :
      (float, 's) Nx.dtype
      * int array
      * operand
      * operand
      * operand
      * operand
      * expr
      -> jit_case

let jit_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s
   and+ y = operand s
   and+ tx = operand s
   and+ ty = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Test_case (dt, s, x, y, tx, ty, e)
   | _ -> assert false)
  |> with_pp (fun ppf (Test_case (dt, s, x, y, _, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_jit (Test_case (dt, s, xo, yo, txo, tyo, e)) =
  let x = make dt s xo and y = make dt s yo in
  let tx = make dt s txo and ty = make dt s tyo in
  cover "a zero-size shape" (numel s = 0);
  cover "a one-element shape" (numel s = 1 && Array.length s > 0);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a composition of over 6 functions" (size e > 6);
  let f x y = eval x y e in
  let pair = T.(pair tensor tensor) in
  let both (p, t) = (bits p, bits t) in
  let eager =
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty)))
  in
  (* Compiled, the derivative gives jvp's bits where every function of it is
     exact. *)
  let e' = elementwise (exact e) in
  let f' x y = eval x y e' in
  let exact_eager =
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> f' x y) (x, y) (tx, ty)))
  in
  let jitted =
    let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f' in
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> g x y) (x, y) (tx, ty)))
  in
  let staged =
    let g =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor @-> tensor
          @-> returns (pair tensor tensor))
        (fun x y tx ty ->
          Rune.jvp pair T.tensor (fun (x, y) -> f' x y) (x, y) (tx, ty))
    in
    outcome (fun () -> both (g x y tx ty))
  in
  cover "a refused operation" (match eager with Raised _ -> true | _ -> false);
  equal ~msg:"jvp of jit"
    (outcome_t (Testable.pair bits_t bits_t))
    exact_eager jitted;
  equal ~msg:"jit of jvp"
    (outcome_t (Testable.pair bits_t bits_t))
    exact_eager staged

(* Every dtype, through the functions every dtype has: complex rules staged and
   evaluated in a call, integers refused alike. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype
      * int array
      * operand
      * operand
      * operand
      * operand
      * expr
      -> any_case

let any_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s
   and+ y = operand s
   and+ tx = operand s
   and+ ty = operand s in
   let+ (D.Any dt) = of_list D.all in
   Any_case (dt, s, x, y, tx, ty, generic e))
  |> with_pp (fun ppf (Any_case (dt, s, x, y, _, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_jit_any (Any_case (dt, s, xo, yo, txo, tyo, e)) =
  let x = make_any dt s xo and y = make_any dt s yo in
  let tx = make_any dt s txo and ty = make_any dt s tyo in
  cover "a complex dtype" (D.is Complex dt);
  cover "a narrow float dtype"
    (D.is Float dt && not (D.equal dt f64 || D.equal dt Nx.float32));
  cover "an integer dtype" (D.is Signed dt || D.is Unsigned dt);
  let pair = T.(pair tensor tensor) in
  let both (p, t) = (codes p, codes t) in
  let e' = elementwise (exact_any dt e) in
  let f' x y = eval_any x y e' in
  let exact_eager =
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> f' x y) (x, y) (tx, ty)))
  in
  let jitted =
    let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f' in
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> g x y) (x, y) (tx, ty)))
  in
  let staged =
    let g =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor @-> tensor
          @-> returns (pair tensor tensor))
        (fun x y tx ty ->
          Rune.jvp pair T.tensor (fun (x, y) -> f' x y) (x, y) (tx, ty))
    in
    outcome (fun () -> both (g x y tx ty))
  in
  let t = outcome_t (Testable.pair codes_t codes_t) in
  equal ~msg:"jvp of jit" t exact_eager jitted;
  equal ~msg:"jit of jvp" t exact_eager staged

(* One kernel per map *)

let test_one_kernel () =
  let x = Nx.place Count.on (v [| 0.5; -1.; 2. |]) in
  let t = Nx.place Count.on (v [| 1.; 1.; 1. |]) in
  let calls f =
    C.reset ();
    ignore (Rune.jvp' f x t);
    (C.calls (), C.positions ())
  in
  equal ~msg:"sin" (pair int int) (1, 3) (calls Nx.sin);
  equal ~msg:"a chain of three" (pair int int) (3, 9)
    (calls (fun x -> Nx.exp (Nx.mul (Nx.sin x) x)));
  equal ~msg:"a sum and its tangent" (pair int int) (1, 3)
    (calls (fun x -> Nx.sum x));
  (* A derivative wider than a block of nx.cpu's holds is one loop too. *)
  let w = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 0.5) in
  let tw = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 1.) in
  let wide_calls f =
    C.reset ();
    ignore (Rune.jvp' f w tw);
    (C.calls (), C.positions ())
  in
  equal ~msg:"a map of twenty operands" (pair int int) (1, 3)
    (wide_calls wide_map);
  equal ~msg:"a sum over twenty operands" (pair int int) (1, 3)
    (wide_calls (wide_reduce (Monoid (Sum, 0, f64))))

(* Structures and constructs *)

let test_structures () =
  let p = T.(pair tensor tensor) in
  invalid ~sub:"Rune.jvp: 1: shape [2] in the parameters, [3] in the tangents"
    (fun () ->
      Rune.jvp p T.tensor
        (fun (a, _) -> a)
        (v [| 1.; 2. |], v [| 1.; 2. |])
        (v [| 1.; 2. |], v [| 1.; 2.; 3. |]));
  invalid ~sub:"Rune.jvp': the parameters hold no real or complex tensor"
    (fun () ->
      Rune.jvp' Fun.id (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 2 |]));
  (* An integer result has a zero tangent, a result that depends on no parameter
     too, and an integer parameter's tangent is ignored. *)
  let i = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
  let s = T.(pair tensor tensor) in
  let (a, b), (ta, tb) =
    Rune.jvp
      T.(pair tensor tensor)
      s
      (fun (x, n) -> (Nx.cast Nx.int32 x, Nx.add n n))
      (v [| 1.5; 2. |], i)
      (v [| 1.; 1. |], i)
  in
  ignore (a, b);
  equal ~msg:"integer result" (array int32) [| 0l; 0l |] (Nx.to_array ta);
  equal ~msg:"integer parameter" (array int32) [| 0l; 0l |] (Nx.to_array tb);
  let _, t = Rune.jvp' (fun _ -> v [| 7. |]) (v [| 1. |]) (v [| 1. |]) in
  equal ~msg:"a constant" floats [| 0. |] (Nx.to_array t);
  (* A tangent is fresh memory: the identity's can be donated and its seed lives
     on. *)
  let x = v [| 1.; 2. |] and dx = v [| 3.; 4. |] in
  let _, t = Rune.jvp' Fun.id x dx in
  ignore (Nx.add (Nx.donate t) x);
  equal floats [| 3.; 4. |] (Nx.to_array dx)

(* A constant parameter, zeros, with a tangent: the dual takes the parameter's
   form, in a loop too. *)
let test_constant_primal () =
  let zero = Nx.zeros f64 [| 2 |] and t = v [| 1.; 2. |] in
  let y, dy = Rune.jvp' (fun x -> Nx.add (Nx.mul x x) x) zero t in
  equal ~msg:"primal" floats [| 0.; 0. |] (Nx.to_array y);
  equal ~msg:"tangent" floats [| 1.; 2. |] (Nx.to_array dy);
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f c =
    fst (Rune.scan' ~f:(fun c x -> (Nx.add c (Nx.mul c x), c)) ~init:c xs)
  in
  let _, dy = Rune.jvp' f zero t in
  (* c_3 = c_0 (1 + x_0) (1 + x_1) (1 + x_2), linear in c_0. *)
  equal ~msg:"through a scan" floats
    [| 1. *. 2. *. 4. *. 6.; 2. *. 3. *. 5. *. 7. |]
    (Nx.to_array dy)

let test_detach () =
  let x = v [| 2.; -1. |] and t = v [| 1.; 3. |] in
  equal ~msg:"detach" floats [| 2.; -3. |]
    (tangent (fun x -> Nx.mul x (Rune.detach x)) x t);
  (* Every differentiation around the call detaches it. *)
  let one = v [| 1. |] in
  let inner x = fst (Rune.jvp' (fun y -> Rune.detach (Nx.mul x y)) one one) in
  equal ~msg:"nested" floats [| 0. |] (tangent inner (v [| 2. |]) one)

(* An OCaml branch on a value reads its primal. *)
let test_read () =
  let f x = if Nx.item [ 0 ] x > 0. then Nx.sin x else Nx.cos x in
  equal floats [| Float.cos 1. |] (tangent f (v [| 1. |]) (v [| 1. |]))

(* Nested forward modes: each tangent belongs to its own. *)
let test_nesting () =
  let one = v [| 1. |] in
  let inner x = snd (Rune.jvp' Nx.sin x one) in
  equal ~msg:"second derivative" floats
    [| -.Float.sin 0.5 |]
    (tangent inner (v [| 0.5 |]) one);
  let f x = Nx.mul x (snd (Rune.jvp' (fun y -> Nx.add x y) one one)) in
  equal ~msg:"perturbations apart" floats [| 1. |] (tangent f one one)

let test_jacfwd () =
  let x = v [| 1.; 2.; -3. |] in
  let j = Rune.jacfwd' (fun x -> Nx.mul x x) x in
  equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape j);
  equal floats [| 2.; 0.; 0.; 0.; 4.; 0.; 0.; 0.; -6. |] (Nx.to_array j);
  (* A linear map's Jacobian is its matrix, rows the result's elements. *)
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let j = Rune.jacfwd' (fun x -> Nx.matmul a x) x in
  equal ~msg:"a matrix's shape" (array int) [| 2; 3 |] (Nx.shape j);
  equal ~msg:"a matrix" floats (Nx.to_array a) (Nx.to_array j);
  (* A transpose's: [J i j k l] is 1 where [(k, l) = (j, i)], in C order. *)
  let m = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let j = Rune.jacfwd' Nx.transpose m in
  equal ~msg:"a transpose's shape" (array int) [| 2; 2; 2; 2 |] (Nx.shape j);
  let expected =
    Array.init 16 (fun n ->
        let i = n / 8
        and jj = n / 4 mod 2
        and k = n / 2 mod 2
        and l = n mod 2 in
        if k = jj && l = i then 1. else 0.)
  in
  equal ~msg:"a transpose" floats expected (Nx.to_array j);
  (* A complex argument's columns are the tangents along its real units. *)
  let z =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -1.; im = 0.5 } |]
  in
  let j = Rune.jacfwd' (fun z -> Nx.mul z z) z in
  let two_z =
    Array.map (fun c -> Complex.mul { re = 2.; im = 0. } c) (Nx.to_array z)
  in
  equal ~msg:"complex"
    (array (pair (float 1e-12) (float 1e-12)))
    (Array.map
       (fun (c : Complex.t) -> (c.re, c.im))
       [| two_z.(0); Complex.zero; Complex.zero; two_z.(1) |])
    (Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array j));
  let j = Rune.jacfwd' (fun x -> Nx.sum x) (Nx.zeros f64 [| 0 |]) in
  equal ~msg:"no element" (array int) [| 0 |] (Nx.shape j);
  invalid ~sub:"Rune.jacfwd': the parameters hold no real or complex tensor"
    (fun () -> Rune.jacfwd' Fun.id (Nx.zeros Nx.int32 [| 2 |]));
  invalid ~sub:"Rune.jacfwd': the tangent of a complete QR factorisation"
    (fun () ->
      Rune.jacfwd'
        (fun a -> fst (Nx.qr ~factors:Complete a))
        (Nx.zeros f64 [| 2; 1 |]))

(* Column [k] of the Jacobian, its elements at index [k] of the argument's axes
   in C order, is the tangent along element [k], bit for bit. *)
let jac_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* k = int_range 0 (Int.max 0 (numel s - 1)) in
   let+ x = operand s in
   (s, x, k, e))
  |> with_pp (fun ppf (s, x, k, e) ->
      Format.fprintf ppf "%a, x %a, column %d: %a" pp_shape s pp_view x.view k
        pp e)

let law_jacobian (s, xo, k, e) =
  let x = make f64 s xo in
  let n = numel s in
  cover "a zero-size shape" (n = 0);
  cover "several elements" (n > 1);
  let f x = eval x x e in
  let jacobian = outcome (fun () -> Rune.jacfwd' f x) in
  cover "a refused function"
    (match jacobian with Raised _ -> true | _ -> false);
  match jacobian with
  | Raised m ->
      (* It raises as jvp' does along any tangent, under its own name. *)
      let renamed m =
        let jvp = "Rune.jvp'" and n = String.length m in
        let k = String.length jvp in
        if n >= k && String.sub m 0 k = jvp then
          "Rune.jacfwd'" ^ String.sub m k (n - k)
        else m
      in
      let along =
        match Rune.jvp' f x (Nx.zeros_like x) with
        | _ -> "no raise"
        | exception Invalid_argument m -> renamed m
      in
      equal ~msg:"raises as jvp'" string along m
  | Value j ->
      let sy = Array.sub (Nx.shape j) 0 (Nx.ndim j - Array.length s) in
      equal ~msg:"shape" (array int) (Array.append sy s) (Nx.shape j);
      if n > 0 then begin
        let column =
          Nx.reshape sy (Nx.slice [ A; I k ] (Nx.reshape [| numel sy; n |] j))
        in
        let unit =
          Nx.reshape s
            (Nx.create f64 [| n |]
               (Array.init n (fun i -> if i = k then 1. else 0.)))
        in
        equal ~msg:"column" bits_t
          (bits (snd (Rune.jvp' f x unit)))
          (bits column)
      end

(* Loops under forward mode: a scan is its fold's, an iterate its while loop's,
   the step traced then evaluated with duals, bit for bit, eagerly and under
   jit. Rows and carry both have tangents. *)
let fold step ~init xs =
  let n = (Nx.shape xs).(0) in
  let rec go i c ys =
    if i = n then (c, List.rev ys)
    else
      let c, y = step c (Nx.get [ i ] xs) in
      go (i + 1) c (y :: ys)
  in
  go 0 init []

type loop_case =
  | Loop_case :
      (float, 's) Nx.dtype
      * int array
      * int
      * operand
      * operand
      * operand
      * operand
      * expr
      * expr
      -> loop_case

let loop_cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let rows = Array.append [| n |] s in
   let* c = operand s and+ xs = operand rows in
   let* tc = operand s and+ txs = operand rows in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Loop_case (dt, s, n, c, xs, tc, txs, g, h)
   | _ -> assert false)
  |> with_pp (fun ppf (Loop_case (dt, s, n, c, xs, _, _, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows, carry %a, xs %a: carry %a, output %a"
        Nx.pp_dtype dt pp_shape s n pp_view c.view pp_view xs.view pp g pp h)

let law_scan (Loop_case (dt, s, n, co, xso, tco, txso, g, h)) =
  let rows = Array.append [| n |] s in
  let init = make dt s co and xs = make dt rows xso in
  let tinit = make dt s tco and txs = make dt rows txso in
  cover "no row" (n = 0);
  cover "several rows" (n > 1);
  cover "a zero-size row" (numel s = 0);
  cover "a view operand" (co.view <> Plain || xso.view <> Plain);
  let step c x =
    (Nx.broadcast_to s (eval c x g), Nx.broadcast_to s (eval c x h))
  in
  let p = T.(pair tensor tensor) in
  let both ((c, ys), (tc, tys)) = ((bits c, bits ys), (bits tc, bits tys)) in
  let folded (c, xs) =
    (* The step is traced at the call, so with no row its errors raise. *)
    if n = 0 then ignore (step c (Nx.zeros dt s));
    let c, ys = fold step ~init:c xs in
    (c, if n = 0 then Nx.zeros dt rows else Nx.stack ~axis:0 ys)
  in
  let scanned (c, xs) = Rune.scan' ~f:step ~init:c xs in
  let jvp f = Rune.jvp p p f (init, xs) (tinit, txs) in
  let expected = outcome (fun () -> both (jvp folded)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* A carry whose step reads nothing keeps a zero tangent where the fold's has
     none. *)
  let l = if constant g then lenient ~nan:true else bits_t in
  let t = Testable.pair (Testable.pair bits_t bits_t) (Testable.pair l l) in
  equal ~msg:"eager" (outcome_t t) expected
    (outcome (fun () -> both (jvp scanned)));
  (* Compiled, the derivative gives the fold's bits where every function of the
     step is exact. *)
  let g' = elementwise (exact g) and h' = elementwise (exact h) in
  let step' c x =
    (Nx.broadcast_to s (eval c x g'), Nx.broadcast_to s (eval c x h'))
  in
  let folded' (c, xs) =
    if n = 0 then ignore (step' c (Nx.zeros dt s));
    let c, ys = fold step' ~init:c xs in
    (c, if n = 0 then Nx.zeros dt rows else Nx.stack ~axis:0 ys)
  in
  let scanned' (c, xs) = Rune.scan' ~f:step' ~init:c xs in
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> tensor @-> returns (pair p p))
      (fun c xs tc txs -> Rune.jvp p p scanned' (c, xs) (tc, txs))
  in
  equal ~msg:"under jit" (outcome_t t)
    (outcome (fun () -> both (jvp folded')))
    (outcome (fun () -> both (staged init xs tinit txs)))

type iterate_case =
  | Iterate_case :
      (float, 's) Nx.dtype * int array * int * operand * operand * expr
      -> iterate_case

let iterate_cases =
  let open Gen in
  (let* s = shape in
   let* trips = int_range 0 3 in
   let* g = expr ~rank:(Array.length s) in
   let* c = operand s and+ tc = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Iterate_case (dt, s, trips, c, tc, g)
   | _ -> assert false)
  |> with_pp (fun ppf (Iterate_case (dt, s, trips, _, _, g)) ->
      Format.fprintf ppf "%a %a, %d trips: %a" Nx.pp_dtype dt pp_shape s trips
        pp g)

let law_iterate (Iterate_case (dt, s, trips, co, tco, g)) =
  cover "no trip" (trips = 0);
  cover "several trips" (trips > 1);
  let init = make dt s co and tinit = make dt s tco in
  let step (x, k) =
    (Nx.broadcast_to s (eval x x g), Nx.add k (Nx.scalar Nx.int32 1l))
  in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int trips))
  in
  let k0 = Nx.scalar Nx.int32 0l in
  let p = T.(pair tensor tensor) in
  let both ((x, _), (t, _)) = (bits x, bits t) in
  let looped c =
    ignore (step c);
    let rec go j c = if j = trips then c else go (j + 1) (step c) in
    go 0 c
  in
  let iterated c = Rune.iterate p ~max:trips ~until ~f:step c in
  let jvp f = Rune.jvp p p f (init, k0) (tinit, k0) in
  let expected = outcome (fun () -> both (jvp looped)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  equal
    (outcome_t (Testable.pair bits_t bits_t))
    expected
    (outcome (fun () -> both (jvp iterated)))

(* An addition under forward mode adds its primal. *)
let test_total () =
  let total = Rune.Total.make () in
  let _, sum =
    Rune.Total.collect total ~zero:(v [| 0. |]) (fun () ->
        Rune.jvp'
          (fun x ->
            Rune.Total.add total x;
            Nx.sin x)
          (v [| 2. |]) (v [| 1. |]))
  in
  equal floats [| 2. |] (Nx.to_array sum)

let () =
  exit
    (run "rune2 jvp"
       [
         group "tangents" tangent_props;
         group "conventions"
           [
             test "kinks take JAX's tangents" test_kinks;
             test "complex tangents are directional derivatives" test_complex;
             test "edges take their definitions' values" test_edges;
             test "factorisations keep their gauges" test_factorisations;
           ];
         group "jit"
           [
             prop "jvp of jit and jit of jvp give jvp's bits" jit_cases law_jit;
             prop "jvp of jit gives jvp's bits at every dtype" any_cases
               law_jit_any;
           ];
         group "loops"
           [
             prop "jvp of scan is jvp of its fold" loop_cases law_scan;
             prop "jvp of iterate is jvp of its loop" iterate_cases law_iterate;
           ];
         group "kernels"
           [ test "a map's tangent is one kernel" test_one_kernel ];
         group "interface"
           [
             test "structures and zero tangents" test_structures;
             test "a constant parameter" test_constant_primal;
             test "detach" test_detach;
             test "a read reads the primal" test_read;
             test "nesting" test_nesting;
             test "jacfwd'" test_jacfwd;
             prop "jacfwd' applied to a tangent is jvp's" jac_cases law_jacobian;
             test "totals" test_total;
           ];
       ])
