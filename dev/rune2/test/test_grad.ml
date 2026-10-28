(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.grad, Rune.vjp and their forms: the adjoint law against forward mode for
   every operation, real and complex in the unconjugated pairing; gradients
   against central differences; the conventions' values; the signs of zero
   contributions; grad of jit, jit of grad and grad of loops against grad; the
   values a tape keeps. *)

open Windtrap
open Compose
open Cases
module Rune = Rune2

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

(* The adjoint law *)

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
  equal ~msg:"gradient shape" (array int) s (Nx.shape jtw);
  let big = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. xs in
  let size = sum_abs (Nx.to_array w) *. sum_abs ts *. ((1. +. big) ** 2.) in
  adjoint ~msg:"<w, J v> = <Jᵀ w, v>" ~size
    (Nx.to_array (Nx.mul w jv))
    (Nx.to_array (Nx.mul jtw t))

(* On complex values, [Re Σ w (J v) = Re Σ (Jᵀ w) v], unconjugated. *)
let complex_law (input, f) (s, xs, ts) =
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
  adjoint ~msg:"Re <w, J v> = Re <Jᵀ w, v>" ~size
    (re (Nx.mul w jv))
    (re (Nx.mul jtw t))

let adjoint_props =
  List.map
    (fun c -> prop c.name (gen c) (adjoint_law c))
    (real_cases @ contract_cases @ factor_cases @ complex_cases
   @ complex_factor_cases)

let complex_props =
  List.filter_map
    (fun c -> Option.map (fun z -> prop c.name (gen c) (complex_law z)) c.z)
    (complex_cases @ complex_factor_cases)

(* Gradients against central differences *)

let h = 1e-6

let fd_law c (s, xs, ts) =
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let y = c.f x in
  let w = Nx.create f64 (Nx.shape y) (spread 0.25 (Nx.numel y)) in
  let objective x = Nx.sum (Nx.mul w (c.f x)) in
  let g = Rune.grad' objective x in
  let at e =
    Nx.to_array
      (Nx.mul w
         (c.f (Nx.create f64 s (Array.mapi (fun k x -> x +. (e *. ts.(k))) xs))))
  in
  let plus = at h and minus = at (-.h) in
  let d = Nx.to_array (Nx.mul g t) in
  if finite plus && finite minus && finite d then begin
    let fd = Array.map2 (fun p m -> (p -. m) /. (2. *. h)) plus minus in
    let fd_sum = Array.fold_left ( +. ) 0. fd
    and d_sum = Array.fold_left ( +. ) 0. d in
    less
      ~msg:(Printf.sprintf "<g, t> %h, differences %h" d_sum fd_sum)
      float_exact
      ~than:(1e-5 *. (1. +. sum_abs fd))
      (Float.abs (d_sum -. fd_sum))
  end

let fd_props =
  List.map
    (fun c -> prop c.name (gen c) (fd_law c))
    (real_cases @ contract_cases @ factor_cases)

(* Values the conventions state *)

let floats = array (float 1e-12)
let signed = array (pair bool float_exact)
let with_signs a = Array.map (fun x -> (Float.sign_bit x, x)) (Nx.to_array a)
let z re im = Nx.create c128 [| 1 |] [| { Complex.re; im } |]
let pairs a = Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array a)
let close_pairs = array (pair (float 1e-12) (float 1e-12))

(* JAX's: the gradient of a real [f] of [z = x + iy] is [∂f/∂x - i ∂f/∂y], and a
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
  let a, b =
    Rune.grad
      T.(pair tensor tensor)
      (fun (a, b) -> Nx.sum (Nx.maximum a b))
      (v [| 1.; 2. |], v [| 1.; 0. |])
  in
  equal ~msg:"maximum's ties" floats [| 0.5; 1.; 0.5; 0. |]
    (Array.append (Nx.to_array a) (Nx.to_array b))

(* A single contribution keeps a zero's sign through maps, reductions, pads,
   windows and slices; through a gather it is added to [+0.]. An element that
   receives none is [+0.]. Each function's cotangent is [-0.] at every element
   of [op x]. *)
let test_zeros () =
  let g op xs =
    let f x =
      let y = op x in
      Nx.sum (Nx.mul y (Nx.full_like y (-0.)))
    in
    with_signs (Rune.grad' f (v xs))
  in
  equal ~msg:"a map and a sum" signed [| (true, -0.) |] (g Nx.sin [| 0. |]);
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
    (g (Nx.take (Nx.create Nx.int64 [| 1 |] [| 0L |])) [| 1.; 2. |])

(* Interface *)

let test_structures () =
  let p = T.(pair tensor tensor) in
  invalid
    ~sub:"Rune.grad: the objective must return a real scalar, got float64 [2]"
    (fun () -> Rune.grad T.tensor (fun x -> x) (v [| 1.; 2. |]));
  invalid
    ~sub:
      "Rune.grad': the objective must return a real scalar, got complex128 [1]"
    (fun () -> Rune.grad' Fun.id (z 1. 0.));
  invalid ~sub:"Rune.grad': the parameters hold no real or complex tensor"
    (fun () ->
      Rune.grad' (fun x -> Nx.sum (Nx.cast f64 x)) (Nx.zeros Nx.int32 [| 2 |]));
  let _, pullback = Rune.vjp T.tensor p (fun x -> (x, x)) (v [| 1.; 2. |]) in
  invalid ~sub:"Rune.vjp: 1: shape [2] in the result, [3] in the cotangents"
    (fun () -> pullback (v [| 1.; 1. |], v [| 1.; 1.; 1. |]));
  (* An integer parameter's gradient and an unused one's are zero. *)
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
    (with_signs (Rune.grad' (fun _ -> Nx.sum (v [| 7. |])) (v [| 1. |])));
  let y, g, aux =
    Rune.value_and_grad_aux T.tensor T.tensor
      (fun x -> (Nx.sum (Nx.sin x), Nx.cos x))
      (v [| 0.5 |])
  in
  equal ~msg:"value" floats
    [| Float.sin 0.5 |]
    (Nx.to_array (Nx.reshape [| 1 |] y));
  equal ~msg:"gradient" floats [| Float.cos 0.5 |] (Nx.to_array g);
  equal ~msg:"aux" floats [| Float.cos 0.5 |] (Nx.to_array aux);
  (* A pullback's result is its own memory, and it runs as often as asked. *)
  let w = v [| 3.; 4. |] in
  let _, pullback = Rune.vjp' Fun.id (v [| 1.; 2. |]) in
  ignore (Nx.add (Nx.donate (pullback w)) w);
  equal ~msg:"fresh" floats [| 3.; 4. |] (Nx.to_array (pullback w));
  (* A gradient whose computation ends with a contraction's axes moved back is
     in C order from the start of its memory. *)
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let pattern = Nx.Pattern.v "k i, k j -> i j | k" in
  let b = Nx.create f64 [| 2; 2 |] [| 1.; -1.; 2.; 0.5 |] in
  let g = Rune.grad' (fun a -> Nx.sum (Nx.contract f64 pattern a b)) a in
  let layout = (Nx.Prim.form g).layout in
  equal ~msg:"in C order" bool true
    (L.is_contiguous layout && fst (L.span layout) = 0)

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
    Rune.Total.collect total ~zero:(v [| 0. |]) (fun () ->
        Rune.grad'
          (fun x ->
            Rune.Total.add total x;
            Nx.sum (Nx.sin x))
          (v [| 2. |]))
  in
  equal floats [| 2. |] (Nx.to_array sum)

(* A gradient lies where its parameter does when every value it meets lies
   there, a placed creation's included. *)
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
    (Nx.to_array (Nx.place Nx.Host.on g))

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
    (Nx.shape (Rune.jacrev' (fun x -> Nx.slice [ R (0, 0) ] x) x));
  invalid ~sub:"Rune.jacrev': the result is complex and the argument real"
    (fun () -> Rune.jacrev' (fun x -> Nx.cast c128 x) x)

let test_check_grads () =
  let ok =
    Rune.check_grads T.tensor (fun x -> Nx.sum (Nx.sin x)) (v [| 0.3; 1. |])
  in
  equal ~msg:"agrees" bool true (Result.is_ok ok);
  (* A jump: the differences see it, the gradient does not. *)
  let jump x = Nx.sum (Nx.where (Nx.greater_s x 0.) x (Nx.add_s x 1.)) in
  match Rune.check_grads T.tensor jump (v [| 0. |]) with
  | Ok () -> failf "a jump passes"
  | Error m ->
      equal ~msg:m bool true
        (String.starts_with
           ~prefix:"Rune.check_grads: the directional derivative along ones is"
           m)

(* grad of jit and jit of grad of exact functions give grad's bits *)

type jit_case =
  | Jit_case :
      (float, 's) Nx.dtype * int array * operand * operand * expr
      -> jit_case

let jit_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with Float -> Jit_case (dt, s, x, y, e) | _ -> assert false)
  |> with_pp (fun ppf (Jit_case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_jit (Jit_case (dt, s, xo, yo, e)) =
  let x = make dt s xo and y = make dt s yo in
  cover "a zero-size shape" (numel s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a composition of over 6 functions" (size e > 6);
  let f x y = Nx.sum (eval x y e) in
  let pair = T.(pair tensor tensor) in
  let both (a, b) = (bits a, bits b) in
  let grad f =
    outcome (fun () -> both (Rune.grad pair (fun (x, y) -> f x y) (x, y)))
  in
  let eager = grad f in
  cover "a refused operation" (match eager with Raised _ -> true | _ -> false);
  let t = outcome_t (Testable.pair bits_t bits_t) in
  (* Compiled, the gradient gives grad's bits where every function of it is
     exact. *)
  let e' = elementwise (exact e) in
  let f' x y = Nx.sum (eval x y e') in
  let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f' in
  equal ~msg:"grad of jit" t (grad f') (grad g);
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun x y -> Rune.grad pair (fun (x, y) -> f' x y) (x, y))
  in
  equal ~msg:"jit of grad" t (grad f') (outcome (fun () -> both (staged x y)))

(* Loops under reverse mode: a scan's gradient is its fold's, eagerly and under
   jit, bit for bit but for the zeros a loop holds where the fold has none
   (Compose.lenient). *)
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
      (float, 's) Nx.dtype * int array * int * operand * operand * expr * expr
      -> loop_case

let loop_cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let* c = operand s and+ xs = operand (Array.append [| n |] s) in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Loop_case (dt, s, n, c, xs, g, h)
   | _ -> assert false)
  |> with_pp (fun ppf (Loop_case (dt, s, n, _, _, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows: carry %a, output %a" Nx.pp_dtype dt
        pp_shape s n pp g pp h)

let law_scan (Loop_case (dt, s, n, co, xso, g, h)) =
  let rows = Array.append [| n |] s in
  let init = make dt s co and xs = make dt rows xso in
  cover "no row" (n = 0);
  cover "several rows" (n > 1);
  let step c x =
    (Nx.broadcast_to s (eval c x g), Nx.broadcast_to s (eval c x h))
  in
  let p = T.(pair tensor tensor) in
  let objective (c, ys) = Nx.add (Nx.sum c) (Nx.sum ys) in
  let folded (c, xs) =
    if n = 0 then ignore (step c (Nx.zeros dt s));
    let c, ys = fold step ~init:c xs in
    objective (c, if n = 0 then Nx.zeros dt rows else Nx.stack ~axis:0 ys)
  in
  let scanned (c, xs) = objective (Rune.scan' ~f:step ~init:c xs) in
  let both (a, b) = (bits a, bits b) in
  let grad f = outcome (fun () -> both (Rune.grad p f (init, xs))) in
  let expected = grad folded in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* The carry's cotangent is a zero where the step reads no carry. *)
  let l = lenient ~nan:(not (reads_x g || reads_x h)) in
  let t = outcome_t (Testable.pair l l) in
  equal ~msg:"eager" t expected (grad scanned);
  (* Compiled, the gradient gives the fold's bits where every function of the
     step is exact. *)
  let g' = elementwise (exact g) and h' = elementwise (exact h) in
  let step' c x =
    (Nx.broadcast_to s (eval c x g'), Nx.broadcast_to s (eval c x h'))
  in
  let folded' (c, xs) =
    if n = 0 then ignore (step' c (Nx.zeros dt s));
    let c, ys = fold step' ~init:c xs in
    objective (c, if n = 0 then Nx.zeros dt rows else Nx.stack ~axis:0 ys)
  in
  let scanned' (c, xs) = objective (Rune.scan' ~f:step' ~init:c xs) in
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun c xs -> Rune.grad p scanned' (c, xs))
  in
  equal ~msg:"under jit" t (grad folded')
    (outcome (fun () -> both (staged init xs)))

(* Residuals *)

let device = Nx_support.memory 1

module Mem = (val Nx.devices [ device ])

(* The bytes [device] holds after four rounds of collection, each of which
   returns to it what it collected. *)
let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* The bytes [device] holds once [f ()] returns beyond those it held before,
   after one uncounted run. *)
let held_after f =
  f ();
  let before = settled () in
  f ();
  settled () - before

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
  let x = Nx.place Mem.on (Nx.ones Nx.float32 [| n |]) in
  let kept = ref [] in
  let keep f () = kept := Rune.vjp' f x :: !kept in
  (* The result's bytes alone: the map keeps its operand, [x]. *)
  equal ~msg:"one map" int (4 * n) (held_after (keep fused));
  (* Three maps keep their operands, the two intermediates among them. *)
  equal ~msg:"three maps" int
    (3 * 4 * n)
    (held_after (keep (fun x -> Nx.mul (Nx.sin x) (Nx.exp x))));
  ignore (Sys.opaque_identity !kept)

let () =
  exit
    (run "rune2 grad"
       [
         group "adjoint" adjoint_props;
         group "complex adjoint" complex_props;
         group "differences" fd_props;
         group "conventions"
           [
             test "complex gradients" test_complex;
             test "kinks" test_kinks;
             test "signs of zero" test_zeros;
           ];
         group "interface"
           [
             test "structures" test_structures;
             test "nesting" test_nesting;
             test "a read reads the primal" test_read;
             test "detach" test_detach;
             test "totals" test_total;
             test "placement" test_placement;
             test "jacrev'" test_jacrev;
             test "check_grads" test_check_grads;
           ];
         group "jit" [ prop "grad of jit and jit of grad" jit_cases law_jit ];
         group "loops"
           [ prop "grad of scan is grad of its fold" loop_cases law_scan ];
         group "memory"
           [ test "a map's residuals are its loads" test_residuals ];
       ])
