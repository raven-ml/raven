(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Sde: Brownian paths and stochastic marches. The trusted side is the
   Gaussian law of increments and areas, Chen's relation, the quadratic a path
   follows inside its finest intervals, and linear SDEs whose solutions are
   known in their calculus: geometric Brownian motion in closed form, and the
   Ornstein–Uhlenbeck process by its exact transition over the path's finest
   intervals. Strong errors are means over many independent components of one
   path. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let tensor = Nx.Ptree.tensor

(* jera's Conventions promise [Invalid_argument] naming the function. *)
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

(* A path over [0, 1] of [n] independent components. *)
let path ?(depth = 10) ?(n = 4000) seed =
  Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape:[| n |] ~t0:0. ~t1:1. ~depth

let mean x = Nx.item [] (Nx.mean x)
let variance x = Nx.item [] (Nx.var x)

(* Brownian *)

(* Two depths of one key, and a node of the coarser: its level and index, over
   [t0, t0 + 1] with ends whose node times are exact. *)
let nodes =
  Gen.(
    let* coarse = int_range 0 30 in
    let* fine = int_range coarse 30 and+ level = int_range 0 (min coarse 20) in
    let+ seed = int_range 0 1000
    and+ t0 = of_list [ 0.; -3.; 1e9 ]
    and+ k = int_range 0 ((1 lsl level) - 1) in
    (seed, t0, coarse, fine, level, k))
  |> Gen.with_pp (fun ppf (seed, t0, coarse, fine, level, k) ->
      Format.fprintf ppf "key %d, t0 %g, depths %d and %d, node %d of level %d"
        seed t0 coarse fine k level)

(* A finest interval of a path over [0, 1], a point in it, and the path's
   shape. *)
let leaves =
  Gen.(
    let* depth = int_range 0 12 in
    let+ seed = int_range 0 1000
    and+ k = int_range 0 ((1 lsl depth) - 1)
    and+ x = float_range 0. 1.
    and+ shape = of_list [ [||]; [| 0 |]; [| 3 |]; [| 2; 2 |] ] in
    (seed, depth, k, x, shape))
  |> Gen.with_pp (fun ppf (seed, depth, k, x, shape) ->
      Format.fprintf ppf "key %d, depth %d, leaf %d, x %h, shape [%s]" seed
        depth k x
        (String.concat "; " (Array.to_list (Array.map string_of_int shape))))

let nodes_agree (seed, t0, coarse, fine, level, k) =
  (* Node [k] of [level] spans [t0 + k 2^-level, t0 + (k + 1) 2^-level], exact
     in float64 for these ends. Each side sums one node's draws, at most 31
     terms of at most 4: 1e-12 bounds their rounding. *)
  cover "a deeper path" (coarse < fine);
  cover "the root" (level = 0);
  cover "a node finer than the floats near 1e9" (t0 = 1e9 && fine >= 24);
  let path depth =
    Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape:[| 4 |] ~t0 ~t1:(t0 +. 1.)
      ~depth
  in
  let at i = t0 +. Float.ldexp (Float.of_int i) (-level) in
  let node depth =
    let dw, h =
      Sde.Brownian.increment (path depth) (scalar (at k)) (scalar (at (k + 1)))
    in
    (dw, Nx.mul_s h (at (k + 1) -. at k))
  in
  equal
    (Oracle.structure ~abs:1e-12 Nx.Ptree.(pair tensor tensor))
    (node coarse) (node fine)

let quadratic (seed, depth, k, x, shape) =
  (* Over a leaf [a, a + δ] of increment ΔW and area H, the path is W(a + x δ) −
     W(a) = x ΔW + 6 x (1 − x) H, the quadratic through ΔW that matches the
     leaf's integral (Foster, Lyons and Oberhauser, 2020, Thm 2.4), and the area
     over [a, a + x δ] is that quadratic's, x² H. [x] is read back from the
     rounded time. Values of at most 4 summed over 13 levels round within
     1e-12. *)
  cover "a path of depth 0" (depth = 0);
  cover "a scalar path" (shape = [||]);
  cover "a path of no components" (shape = [| 0 |]);
  cover "a path of two axes" (Array.length shape = 2);
  let w = Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape ~t0:0. ~t1:1. ~depth in
  let leaf = Float.ldexp 1. (-depth) in
  let a = Float.of_int k *. leaf in
  let t = a +. (x *. leaf) in
  let x = (t -. a) /. leaf in
  let dw, h = Sde.Brownian.increment w (scalar a) (scalar (a +. leaf)) in
  let dw', h' = Sde.Brownian.increment w (scalar a) (scalar t) in
  let tol = Oracle.tensor ~abs:1e-12 () in
  equal ~msg:"W" tol
    (Nx.add (Nx.mul_s dw x) (Nx.mul_s h (6. *. x *. (1. -. x))))
    dw';
  equal ~msg:"area times length" tol
    (Nx.mul_s h ((t -. a) *. x *. x))
    (Nx.mul_s h' (t -. a))

let brownian_tests =
  [
    prop "paths of one key agree at the nodes of the coarser path" nodes
      ~examples:[ (3, 1e9, 0, 24, 0, 0); (5, 0., 3, 30, 2, 3) ]
      nodes_agree;
    cases "a float32 path of any depth has its root's increment"
      ~id:string_of_int [ 25; 30 ] (fun depth ->
        (* The root's draws give the increment over [t0, t1] at every depth. Its
           ends each sum depth + 1 terms of at most 4 in float32, which round by
           at most 2 (31) 4 eps32 = 1.5e-5. *)
        let f32 = Nx.float32 in
        let root depth =
          let w =
            Sde.Brownian.v (Nx.Rng.key 7) f32 ~shape:[| 64 |] ~t0:0. ~t1:1.
              ~depth
          in
          fst (Sde.Brownian.increment w (Nx.scalar f32 0.) (Nx.scalar f32 1.))
        in
        equal (Oracle.tensor ~abs:2e-5 ()) (root 0) (root depth));
    prop
      "inside a finest interval the path is the quadratic of its increment and \
       area"
      leaves ~count:300 quadratic;
    test "a query is a pure function of the key and the times" (fun () ->
        let inc w = Sde.Brownian.increment w (scalar 0.2) (scalar 0.7) in
        equal
          (Oracle.structure Nx.Ptree.(pair tensor tensor))
          (inc (path ~n:8 3))
          (inc (path ~n:8 3)));
  ]

(* Brownian.v *)

let v_tests =
  let fn = "Jera.Sde.Brownian.v" in
  let v ?(depth = 3) ?(shape = [||]) t0 t1 () =
    Sde.Brownian.v (Nx.Rng.key 0) f64 ~shape ~t0 ~t1 ~depth
  in
  [
    cases "v rejects ends that do not bound a finite interval"
      ~id:(fun (t0, t1) -> Printf.sprintf "[%g, %g]" t0 t1)
      [
        (0., Float.infinity);
        (Float.neg_infinity, 0.);
        (Float.nan, 1.);
        (1., 1.);
        (1., 0.);
        (-1e308, 1e308);
      ]
      (fun (t0, t1) -> raises_in fn (v t0 t1));
    cases "v rejects ends that round together or overflow in float32"
      ~id:(fun (t0, t1) -> Printf.sprintf "[%g, %.10g]" t0 t1)
      [ (1., 1. +. 1e-9); (0., 1e39) ]
      (fun (t0, t1) ->
        raises_in fn (fun () ->
            Sde.Brownian.v (Nx.Rng.key 0) Nx.float32 ~shape:[||] ~t0 ~t1
              ~depth:3));
    cases "v rejects a depth outside [0, 30]" ~id:string_of_int [ -1; 31 ]
      (fun depth -> raises_in fn (v ~depth 0. 1.));
    test "v rejects a negative dimension" (fun () ->
        raises_in fn (v ~shape:[| 2; -1 |] 0. 1.));
  ]

(* Brownian.increment *)

let times =
  Gen.(triple (float_range 0. 1.) (float_range 0. 1.) (float_range 0. 1.))
  |> Gen.with_pp (fun ppf (s, t, u) -> Format.fprintf ppf "(%h, %h, %h)" s t u)

let chen w (s, t, u) =
  (* Chen's relation for W and for the area times its interval's length, h H =
     I(t) − I(s) − h (W s + W t) / 2, which holds for times in any order. Values
     of at most 4 round within 1e-12. *)
  cover "a reversed interval" (s > t || t > u);
  let inc a b = Sde.Brownian.increment w (scalar a) (scalar b) in
  let w1, h1 = inc s t and w2, h2 = inc t u and w12, h12 = inc s u in
  let tol = Oracle.tensor ~abs:1e-12 () in
  equal ~msg:"W" tol w12 (Nx.add w1 w2);
  let l1 = t -. s and l2 = u -. t in
  let chen =
    Nx.add
      (Nx.add (Nx.mul_s h1 l1) (Nx.mul_s h2 l2))
      (Nx.div_s (Nx.sub (Nx.mul_s w1 l2) (Nx.mul_s w2 l1)) 2.)
  in
  equal ~msg:"area times length" tol (Nx.mul_s h12 (u -. s)) chen

let increment_tests =
  let w = path ~n:8 3 in
  [
    prop "increments compose by Chen's relation" times
      ~examples:[ (0.3, 0.3, 0.7); (0., 1., 1.); (0.9, 0.2, 0.5) ]
      (chen w);
    prop "an interval read backward has the negated increment and the same area"
      Gen.(pair (float_range 0. 1.) (float_range 0. 1.))
      (fun (s, t) ->
        (* Both sides divide the same difference of integrals and subtract
           values of at most 4: 1e-12 bounds their rounding. *)
        let dw, h = Sde.Brownian.increment w (scalar s) (scalar t) in
        let back = Sde.Brownian.increment w (scalar t) (scalar s) in
        equal
          (Oracle.structure ~abs:1e-12 Nx.Ptree.(pair tensor tensor))
          (Nx.neg dw, h)
          back);
    cases "an empty interval has no increment and no area" ~id:string_of_float
      [ 0.; 0.3; 1. ] (fun t ->
        let dw, h = Sde.Brownian.increment w (scalar t) (scalar t) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 8 |]) dw;
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 8 |]) h);
    cases "increments and areas have their Gaussian law"
      ~id:(fun (s, t) -> Printf.sprintf "[%g, %g]" s t)
      [ (0., 1.); (0.25, 0.75); (0.1, 0.43); (0.6001, 0.6123) ]
      (fun (s, t) ->
        (* 4000 components: a mean's standard error is σ / 63, a variance's
           about σ² / 45; five of either is the bound. The key is fixed, so the
           run is deterministic. *)
        let dw, h = Sde.Brownian.increment (path 11) (scalar s) (scalar t) in
        let l = t -. s in
        less ~msg:"W mean" (float 1.)
          ~than:(5. *. Float.sqrt l /. 63.)
          (Float.abs (mean dw));
        less ~msg:"W variance" (float 1.)
          ~than:(5. *. l /. 45.)
          (Float.abs (variance dw -. l));
        less ~msg:"H mean" (float 1.)
          ~than:(5. *. Float.sqrt (l /. 12.) /. 63.)
          (Float.abs (mean h));
        less ~msg:"H variance" (float 1.)
          ~than:(5. *. l /. 12. /. 45.)
          (Float.abs (variance h -. (l /. 12.)));
        less ~msg:"W H covariance" (float 1.)
          ~than:(5. *. l /. Float.sqrt 12. /. 63.)
          (Float.abs (mean (Nx.mul dw h))));
    test "increments over disjoint intervals are uncorrelated" (fun () ->
        let w = path 12 in
        let a, _ = Sde.Brownian.increment w (scalar 0.1) (scalar 0.4) in
        let b, _ = Sde.Brownian.increment w (scalar 0.4) (scalar 0.9) in
        less (float 1.)
          ~than:(5. *. Float.sqrt (0.3 *. 0.5) /. 63.)
          (Float.abs (mean (Nx.mul a b))));
    cases "increment rejects a time outside the path"
      ~id:(fun (s, t) -> Printf.sprintf "[%h, %h]" s t)
      [ (-0.1, 0.5); (0., 1.5); (0., Float.succ 1.); (Float.nan, 0.5) ]
      (fun (s, t) ->
        raises_in "Jera.Sde.Brownian.increment" (fun () ->
            Sde.Brownian.increment w (scalar s) (scalar t)));
  ]

(* Geometric Brownian motion and the Ornstein–Uhlenbeck process *)

let mu = 0.5
and sigma = 0.8

let x0 n = Nx.full f64 [| n |] 1.
let gbm_drift _ x = Nx.mul_s x mu
let gbm_diffusion _ x dw = Nx.mul (Nx.mul_s x sigma) dw

(* X(1) of the Itô and Stratonovich geometric Brownian motions from 1. *)
let gbm_exact ~ito w =
  let dw, _ = Sde.Brownian.increment w (scalar 0.) (scalar 1.) in
  let drift = if ito then mu -. (sigma *. sigma /. 2.) else mu in
  Nx.exp (Nx.add_s (Nx.mul_s dw sigma) drift)

let ou_drift _ x = Nx.neg x
let ou_diffusion _ _ dw = Nx.mul_s dw sigma

(* X(1) of dX = −X dt + σ dW from 1, by the exact transition over each of the
   path's finest intervals δ: e^(−δ) X + σ (ΔW − δ (ΔW / 2 + H)), whose error is
   O(δ^(5/2)) per interval. *)
let ou_exact w ~depth =
  let k = 1 lsl depth in
  let d = 1. /. Float.of_int k in
  let x = ref (x0 4000) in
  for i = 0 to k - 1 do
    let dw, h =
      Sde.Brownian.increment w
        (scalar (Float.of_int i *. d))
        (scalar (Float.of_int (i + 1) *. d))
    in
    let noise =
      Nx.mul_s (Nx.sub dw (Nx.mul_s (Nx.add (Nx.div_s dw 2.) h) d)) sigma
    in
    x := Nx.add (Nx.mul_s !x (Float.exp (-.d))) noise
  done;
  !x

(* Brownian.walk *)

let walk_tests =
  [
    test
      "a compiled march that captures its path draws eager's numbers on every \
       call" (fun () ->
        Compiled.here ();
        let w = path ~depth:1 ~n:3 36 in
        let f x =
          Sde.march tensor Sde.euler_maruyama ~steps:4 ~drift:gbm_drift
            ~diffusion:gbm_diffusion w
            ~at:(vec [| 0.; 0.5; 1. |])
            x
        in
        let x = vec [| 1.; 0.5; 2. |] in
        let eager = f x in
        equal (Oracle.tensor ~rel:1e-12 ()) eager (Rune.jit' f x);
        equal (Oracle.tensor ~rel:1e-12 ()) eager (Rune.jit' f x));
  ]

(* Methods *)

(* One path for the geometric motions, and one with its exact OU solution for
   the additive noise: each costs its queries once per run. *)
let gbm_path = lazy (path 21)

let ou =
  lazy
    (let w = path ~depth:9 23 in
     (w, ou_exact w ~depth:9))

(* The mean, over the components, of the error at t = 1 of [m] from [x0] in 4,
   8, 16 and 32 steps against [exact]. *)
let errors m ~drift ~diffusion w x0 exact =
  List.map
    (fun steps ->
      let y =
        Sde.march tensor m ~steps ~drift ~diffusion w ~at:(vec [| 0.; 1. |]) x0
      in
      mean (Nx.abs (Nx.sub (Nx.get [ 1 ] y) exact)))
    [ 4; 8; 16; 32 ]

(* The least-squares slope of log₂ error against log₂ steps, negated. Each error
   is a mean over 4000 components, so its log₂ has a standard error near 0.017,
   and the slope over four points one apart about 0.011: a margin of 0.15 below
   the order is over ten of those, with the pre-asymptotic curve. *)
let order errors =
  let k = Float.of_int (List.length errors) in
  let xs = List.mapi (fun i _ -> Float.of_int i) errors
  and ys = List.map Float.log2 errors in
  let mx = List.fold_left ( +. ) 0. xs /. k
  and my = List.fold_left ( +. ) 0. ys /. k in
  let sxy =
    List.fold_left2 (fun acc x y -> acc +. ((x -. mx) *. (y -. my))) 0. xs ys
  in
  let sxx =
    List.fold_left (fun acc x -> acc +. ((x -. mx) *. (x -. mx))) 0. xs
  in
  -.sxy /. sxx

let gbm_order m ~ito =
  let w = Lazy.force gbm_path in
  order
    (errors m ~drift:gbm_drift ~diffusion:gbm_diffusion w (x0 4000)
       (gbm_exact ~ito w))

let ou_order m =
  let w, exact = Lazy.force ou in
  order (errors m ~drift:ou_drift ~diffusion:ou_diffusion w (x0 4000) exact)

(* dX = a X dt + B X dW₁ + c X dW₂ on pairs, B = b [[0, 1], [1, 0]]: the
   diffusion's first component reads the second, so the noise is not diagonal.
   A, B and c I commute, so X(1) = e^(a − (b² + c²) / 2 + c W₂) (cosh (b W₁) X₀
   + sinh (b W₁) B X₀ / b). *)
let pair_order () =
  let a = 0.5 and b = 0.6 and c = 0.4 in
  let w =
    Sde.Brownian.v (Nx.Rng.key 26) f64 ~shape:[| 4000; 2 |] ~t0:0. ~t1:1.
      ~depth:10
  in
  let column j x = Nx.slice [ Nx.A; Nx.R (j, j + 1) ] x in
  let swap x = Nx.concatenate ~axis:1 [ column 1 x; column 0 x ] in
  let diffusion _ x dw =
    Nx.add
      (Nx.mul_s (Nx.mul (swap x) (column 0 dw)) b)
      (Nx.mul_s (Nx.mul x (column 1 dw)) c)
  in
  let x0 = Nx.tile [| 4000; 1 |] (Nx.create f64 [| 1; 2 |] [| 1.; 0.5 |]) in
  let dw, _ = Sde.Brownian.increment w (scalar 0.) (scalar 1.) in
  let w1 = Nx.mul_s (column 0 dw) b and w2 = column 1 dw in
  let scale =
    Nx.exp (Nx.add_s (Nx.mul_s w2 c) (a -. (((b *. b) +. (c *. c)) /. 2.)))
  in
  let exact =
    Nx.mul scale
      (Nx.add (Nx.mul (Nx.cosh w1) x0) (Nx.mul (Nx.sinh w1) (swap x0)))
  in
  order
    (errors Sde.milstein ~drift:(fun _ x -> Nx.mul_s x a) ~diffusion w x0 exact)

let method_tests =
  [
    test "euler_maruyama has strong order 1/2 on Itô's geometric motion"
      (fun () ->
        let p = gbm_order Sde.euler_maruyama ~ito:true in
        at_least (float 1e-9) ~than:0.35 p;
        (* Above 3/4 it would be another method's order. *)
        at_most (float 1e-9) ~than:0.75 p);
    test "milstein has strong order 1 on Itô's geometric motion" (fun () ->
        at_least (float 1e-9) ~than:0.85 (gbm_order Sde.milstein ~ito:true));
    test "milstein has strong order 1/2 on noise that is not diagonal"
      (fun () -> at_least (float 1e-9) ~than:0.35 (pair_order ()));
    test "sra1 has strong order 3/2 on additive noise" (fun () ->
        at_least (float 1e-9) ~than:1.3 (ou_order Sde.sra1));
    test "sra1 has strong order 1/2 on Itô's geometric motion" (fun () ->
        at_least (float 1e-9) ~than:0.35 (gbm_order Sde.sra1 ~ito:true));
    test "reversible_heun has strong order 1 on additive noise" (fun () ->
        at_least (float 1e-9) ~than:0.85 (ou_order Sde.reversible_heun));
    test
      "reversible_heun has strong order 1/2 on Stratonovich's geometric motion"
      (fun () ->
        at_least (float 1e-9) ~than:0.4
          (gbm_order Sde.reversible_heun ~ito:false));
  ]

(* March *)

(* A march of sra1 on dX = −X dt + σ dW drawn over a path of any depth and
   lanes, along up to four times of [0, 1], the path's ends among them. *)
let marches =
  Gen.(
    let+ seed = int_range 0 1000
    and+ depth = int_range 0 12
    and+ lanes = int_range 0 5
    and+ steps = int_range 1 6
    and+ ts =
      list ~size:(int_range 1 4)
        (frequency
           [ (1, constant 0.); (1, constant 1.); (3, float_range 0. 1.) ])
    in
    let ts = List.sort_uniq compare ts in
    (seed, depth, lanes, steps, Array.of_list ts))
  |> Gen.with_pp (fun ppf (seed, depth, lanes, steps, ts) ->
      Format.fprintf ppf "key %d, depth %d, %d lanes, %d steps, at [%s]" seed
        depth lanes steps
        (String.concat "; " (Array.to_list (Array.map string_of_float ts))))

(* sra1's step on dX = −X dt + σ dW from its statement (Rößler, 2010), with the
   increment and area of [[s, t]] queried from the path. *)
let sra1_ou w s t x =
  let h = t -. s in
  let dw, area = Sde.Brownian.increment w (scalar s) (scalar t) in
  let i10 = Nx.add (Nx.div_s dw 2.) area in
  let k1 = Nx.neg x in
  let stage =
    Nx.add (Nx.add x (Nx.mul_s k1 (0.75 *. h))) (Nx.mul_s i10 (1.5 *. sigma))
  in
  let k2 = Nx.neg stage in
  Nx.add
    (Nx.add x (Nx.mul_s (Nx.add (Nx.div_s k1 3.) (Nx.mul_s k2 (2. /. 3.))) h))
    (Nx.mul_s dw sigma)

let reference w ~steps at x =
  let state = ref x in
  let states = ref [ x ] in
  for i = 0 to Array.length at - 2 do
    let a = at.(i) and b = at.(i + 1) in
    let h = (b -. a) /. Float.of_int steps in
    for j = 0 to steps - 1 do
      let s = a +. (Float.of_int j *. h) in
      let t = if j = steps - 1 then b else a +. (Float.of_int (j + 1) *. h) in
      state := sra1_ou w s t !state
    done;
    states := !state :: !states
  done;
  Nx.stack ~axis:0 (List.rev !states)

let composes (seed, depth, lanes, steps, ts) =
  cover "a path of depth 0" (depth = 0);
  cover "a path of depth 1" (depth = 1);
  cover "one time" (Array.length ts = 1);
  cover "several intervals" (Array.length ts > 2);
  cover "the path's whole interval"
    (ts.(0) = 0. && ts.(Array.length ts - 1) = 1.);
  cover "one step per interval" (steps = 1);
  cover "no lanes" (lanes = 0);
  let w =
    Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape:[| lanes |] ~t0:0. ~t1:1. ~depth
  in
  let x = Nx.linspace f64 0.5 2. lanes in
  let y =
    Sde.march tensor Sde.sra1 ~steps ~drift:ou_drift ~diffusion:ou_diffusion w
      ~at:(vec ts) x
  in
  equal (Oracle.tensor ~abs:1e-12 ()) (reference w ~steps ts x) y

(* The methods whose state is the march's alone, so that a march restarted from
   a state continues it. *)
let exact_methods =
  [
    ("euler_maruyama", Sde.euler_maruyama);
    ("milstein", Sde.milstein);
    ("sra1", Sde.sra1);
  ]

let methods = exact_methods @ [ ("reversible_heun", Sde.reversible_heun) ]

let restarts =
  Gen.(
    let+ m = of_list exact_methods
    and+ seed = int_range 0 1000
    and+ depth = int_range 0 12
    and+ steps = int_range 1 6
    and+ ts = times in
    let a, b, c = ts in
    let ts =
      match List.sort compare [ a; b; c ] with
      | [ a; b; c ] -> (a, b, c)
      | _ -> assert false
    in
    (m, seed, depth, steps, ts))
  |> Gen.with_pp (fun ppf ((name, _), seed, depth, steps, (a, b, c)) ->
      Format.fprintf ppf "%s, key %d, depth %d, %d steps, at [%g; %g; %g]" name
        seed depth steps a b c)

let continues ((_, m), seed, depth, steps, (a, b, c)) =
  (* Bit for bit: the interval from b continues from the path's W and I carried
     to b, the restarted march from a query of the path at b. *)
  assume (a < b && b < c);
  cover "a path of depth 0" (depth = 0);
  cover "a path of depth 1" (depth = 1);
  let w =
    Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape:[| 3 |] ~t0:0. ~t1:1. ~depth
  in
  let run at x =
    Sde.march tensor m ~steps ~drift:gbm_drift ~diffusion:gbm_diffusion w
      ~at:(vec at) x
  in
  let whole = run [| a; b; c |] (x0 3) in
  let rest = run [| b; c |] (Nx.get [ 1 ] whole) in
  equal (Oracle.tensor ()) (Nx.get [ 2 ] whole) (Nx.get [ 1 ] rest)

(* The state at b after one step of [m] from 0 over [a, b], where b − a rounds
   so that a + (b − a) is not b, under no drift and a diffusion that is zero
   except at b; and the increment and area over [a, b]. *)
let live_at_end m =
  let a = -7.074863082399303 and b = 4.437342817489506 in
  let w =
    Sde.Brownian.v (Nx.Rng.key 35) f64 ~shape:[| 3 |] ~t0:(-8.) ~t1:5. ~depth:4
  in
  let y =
    Sde.march tensor m ~steps:1
      ~drift:(fun _ x -> Nx.zeros_like x)
      ~diffusion:(fun t _ dw -> Nx.mul dw (Nx.cast f64 (Nx.equal_s t b)))
      w
      ~at:(vec [| a; b |])
      (Nx.zeros f64 [| 3 |])
  in
  let dw, h = Sde.Brownian.increment w (scalar a) (scalar b) in
  (Nx.get [ 1 ] y, dw, h)

let march_tests =
  let w = path ~n:3 31 in
  let gbm_march m ~steps at x =
    Sde.march tensor m ~steps ~drift:gbm_drift ~diffusion:gbm_diffusion w ~at x
  in
  let three = vec [| 0.; 0.5; 1. |] in
  [
    test "a march stacks the state at each time, the start first" (fun () ->
        let y = gbm_march Sde.euler_maruyama ~steps:4 three (x0 3) in
        equal (array int) [| 3; 3 |] (Nx.shape y);
        equal (Oracle.tensor ()) (x0 3) (Nx.get [ 0 ] y));
    test "one time is the start alone" (fun () ->
        let y = gbm_march Sde.reversible_heun ~steps:3 (vec [| 0.4 |]) (x0 3) in
        equal (Oracle.tensor ()) (Nx.reshape [| 1; 3 |] (x0 3)) y);
    prop "a march composes its steps over the path's increments" marches
      ~examples:
        [
          (7, 0, 2, 3, [| 0.; 0.5; 1. |]);
          (8, 1, 1, 2, [| 0.25; 0.75 |]);
          (9, 3, 0, 1, [| 0.4 |]);
        ]
      composes;
    test "marches with different steps see one path" (fun () ->
        (* With zero drift and a constant diffusion the state is x0 + σ W,
           whatever the steps. *)
        let at = vec [| 0.; 0.3; 1. |] in
        let walk steps =
          Sde.march tensor Sde.euler_maruyama ~steps
            ~drift:(fun _ x -> Nx.zeros_like x)
            ~diffusion:ou_diffusion w ~at (x0 3)
        in
        equal (Oracle.tensor ~abs:1e-14 ()) (walk 1) (walk 7));
    prop "a march restarted at one of its times continues it" restarts
      ~examples:
        [
          (("sra1", Sde.sra1), 5, 0, 2, (0.1, 0.4, 0.9));
          (("milstein", Sde.milstein), 6, 1, 3, (0., 0.5, 1.));
        ]
      continues;
    cases "a march to the path's end takes any number of steps"
      ~id:(fun (t1, steps) -> Printf.sprintf "[0, %g] in %d" t1 steps)
      [ (0.3, 10); (0.9, 7); (5., 12) ]
      (fun (t1, steps) ->
        (* The last step ends at at's last time, the path's end, even where
           steps additions of t1 / steps round past it, as on these. *)
        let w =
          Sde.Brownian.v (Nx.Rng.key 33) f64 ~shape:[| 3 |] ~t0:0. ~t1 ~depth:4
        in
        let y =
          Sde.march tensor Sde.euler_maruyama ~steps ~drift:gbm_drift
            ~diffusion:gbm_diffusion w
            ~at:(vec [| 0.; t1 |])
            (x0 3)
        in
        equal (array int) [| 2; 3 |] (Nx.shape y));
    cases "steps whose span rounds to zero leave the state as one step does"
      ~id:fst methods (fun (_, m) ->
        (* Of 4 steps over [1, 1 + 2^-52], three draw their increment over no
           time; their drift's step of 2^-54 moves a state near 1 by below
           1e-12. *)
        let w =
          Sde.Brownian.v (Nx.Rng.key 34) f64 ~shape:[| 3 |] ~t0:0. ~t1:2.
            ~depth:6
        in
        let at = vec [| 1.; 1. +. Float.epsilon |] in
        let run steps =
          Sde.march tensor m ~steps ~drift:gbm_drift ~diffusion:gbm_diffusion w
            ~at (x0 3)
        in
        equal (Oracle.tensor ~abs:1e-12 ()) (run 1) (run 4));
    cases "no step starts past its interval's end" ~id:fst
      [ ("euler_maruyama", Sde.euler_maruyama); ("milstein", Sde.milstein) ]
      (fun (_, m) ->
        (* 1.5e-323 is three subnormal units: 5 steps of a unit each start at 0,
           1, 2, 3 and 3. The drift, read at each step's start, is NaN past the
           end. *)
        let b = 1.5e-323 in
        let drift t x =
          Nx.where (Nx.greater_s t b) (Nx.full_like x Float.nan) (gbm_drift t x)
        in
        let y =
          Sde.march tensor m ~steps:5 ~drift ~diffusion:gbm_diffusion w
            ~at:(vec [| 0.; b |])
            (x0 3)
        in
        equal (Oracle.tensor ()) (Nx.ones f64 [| 2; 3 |]) y);
    test "sra1 reads the diffusion at its interval's end itself" (fun () ->
        (* SRA1 applies g (t + h) to ΔW − I₁₀ / h = ΔW / 2 − H (Rößler,
           2010). *)
        let y, dw, h = live_at_end Sde.sra1 in
        equal (Oracle.tensor ()) (Nx.sub (Nx.div_s dw 2.) h) y);
    test "reversible_heun reads the diffusion at its interval's end itself"
      (fun () ->
        (* The reversible Heun method applies g at the step's end to ΔW / 2
           (Kidger, Foster, Li and Lyons, 2021). *)
        let y, dw, _ = live_at_end Sde.reversible_heun in
        equal (Oracle.tensor ()) (Nx.div_s dw 2.) y);
    cases "a step evaluates the drift and the diffusion as its method states"
      ~id:(fun (name, _, _) -> name)
      [
        ("euler_maruyama", Sde.euler_maruyama, (1, 1));
        ("milstein", Sde.milstein, (1, 3));
        ("sra1", Sde.sra1, (2, 3));
        (* One drift per step, and one at the march's start. *)
        ("reversible_heun", Sde.reversible_heun, (2, 2));
      ]
      (fun (_, m, expected) ->
        (* One step of one interval runs no loop, so each call is one
           evaluation. *)
        let drifts = ref 0 and diffusions = ref 0 in
        let drift t x =
          incr drifts;
          gbm_drift t x
        and diffusion t x dw =
          incr diffusions;
          gbm_diffusion t x dw
        in
        ignore
          (Sde.march tensor m ~steps:1 ~drift ~diffusion w
             ~at:(vec [| 0.; 1. |])
             (x0 3));
        equal (pair int int) expected (!drifts, !diffusions));
    cases "a pair state steps as each leaf alone along one path" ~id:fst methods
      (fun (_, m) ->
        let both = Nx.Ptree.(pair tensor tensor) in
        let x = vec [| 1.; 0.5; 2. |] and z = vec [| -1.; 3.; 0.25 |] in
        let pair =
          Sde.march both m ~steps:3
            ~drift:(fun t (x, z) -> (gbm_drift t x, gbm_drift t z))
            ~diffusion:(fun t (x, z) dw ->
              (gbm_diffusion t x dw, gbm_diffusion t z dw))
            w ~at:three (x, z)
        in
        let alone x = gbm_march m ~steps:3 three x in
        equal (Oracle.structure both) (alone x, alone z) pair);
    cases "an integer leaf is carried unchanged" ~id:fst methods (fun (_, m) ->
        let both = Nx.Ptree.(pair tensor tensor) in
        let count = Nx.create Nx.int32 [| 2 |] [| 7l; 9l |] in
        let _, counts =
          Sde.march both m ~steps:3
            ~drift:(fun t (x, i) -> (gbm_drift t x, i))
            ~diffusion:(fun t (x, i) dw -> (gbm_diffusion t x dw, i))
            (path ~n:2 31) ~at:three
            (vec [| 1.; 2. |], count)
        in
        equal (Oracle.tensor ()) (Nx.stack [ count; count; count ]) counts);
    cases "a float32 leaf steps as float64 does, to float32's rounding" ~id:fst
      methods (fun (_, m) ->
        (* Six steps of about fifteen operations each on values near 1, each
           rounding by eps32 = 6e-8: within 1e-5. *)
        let f32 = Nx.float32 in
        let y =
          Sde.march tensor m ~steps:3 ~drift:gbm_drift
            ~diffusion:(fun t x dw -> gbm_diffusion t x (Nx.cast f32 dw))
            w ~at:three
            (Nx.cast f32 (vec [| 1.; 0.5; 2. |]))
        in
        equal (array string) [| "float32" |] [| Nx.Dtype.name (Nx.dtype y) |];
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (gbm_march m ~steps:3 three (vec [| 1.; 0.5; 2. |]))
          (Nx.cast f64 y));
    cases "grad in the initial state is the finite difference" ~id:fst methods
      (fun (_, m) ->
        (* Central differences at 1e-6: truncation near 1e-12, rounding near eps
           / 1e-6 = 2e-10, so 1e-7 holds. *)
        let f x = Nx.sum (Nx.get [ 2 ] (gbm_march m ~steps:4 three x)) in
        let x = vec [| 1.; 0.5; 2. |] in
        let v = vec [| 1.; -1.; 0.5 |] in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 f x v))
          (scalar (Oracle.dot (Rune.grad' f x) v)));
    cases "grad in a value the diffusion reads is the finite difference" ~id:fst
      methods (fun (_, m) ->
        (* Three times, so each interval is computed again in reverse; the
           tolerance as for the initial state. *)
        let f s =
          Nx.sum
            (Nx.get [ 2 ]
               (Sde.march tensor m ~steps:4 ~drift:gbm_drift
                  ~diffusion:(fun _ x dw -> Nx.mul (Nx.mul x s) dw)
                  w ~at:three
                  (vec [| 1.; 0.5; 2. |])))
        in
        let s = scalar sigma in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Oracle.central ~eps:1e-6 f s (scalar 1.))
          (Rune.grad' f s));
    (* Each depth compiles its own program, whose descent grows with the depth:
       every method at depth 1, and depths 0 and 3 on one. *)
    cases "a compiled march with the path an argument equals eager"
      ~id:(fun (name, _, depth) -> Printf.sprintf "%s, depth %d" name depth)
      [
        ("euler_maruyama", Sde.euler_maruyama, 0);
        ("euler_maruyama", Sde.euler_maruyama, 1);
        ("euler_maruyama", Sde.euler_maruyama, 3);
        ("milstein", Sde.milstein, 1);
        ("sra1", Sde.sra1, 1);
        ("reversible_heun", Sde.reversible_heun, 1);
      ]
      (fun (_, m, depth) ->
        (* Compiled code may fuse and reorder some fifty operations, each
           rounding once: within 1e-12. *)
        Compiled.here ();
        let f w x =
          Sde.march tensor m ~steps:4 ~drift:gbm_drift ~diffusion:gbm_diffusion
            w ~at:three x
        in
        let compiled =
          Rune.jit
            Nx.Ptree.(
              Nx.Ptree.v Sde.Brownian.walk @-> tensor @-> returns tensor)
            f
        in
        let x = vec [| 1.; 0.5; 2. |] in
        let w = path ~depth ~n:3 31 in
        equal (Oracle.tensor ~rel:1e-12 ()) (f w x) (compiled w x);
        (* Another key replays the same program on another path. *)
        let w' = path ~depth ~n:3 32 in
        equal (Oracle.tensor ~rel:1e-12 ()) (f w' x) (compiled w' x));
    test "vmap of a march over the initial state is each lane's march"
      (fun () ->
        let w =
          Sde.Brownian.v (Nx.Rng.key 37) f64 ~shape:[||] ~t0:0. ~t1:1. ~depth:1
        in
        let one x =
          Sde.march tensor Sde.euler_maruyama ~steps:4 ~drift:gbm_drift
            ~diffusion:gbm_diffusion w ~at:three x
        in
        let x = vec [| 1.; 0.5; 2. |] in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> one (Nx.get [ i ] x))))
          (Rune.vmap' one x));
    test "march rejects steps below 1" (fun () ->
        raises_in "Jera.Sde.march" (fun () ->
            gbm_march Sde.euler_maruyama ~steps:0 (vec [| 0.; 1. |]) (x0 3)));
    test "march rejects at of two axes" (fun () ->
        raises_in "Jera.Sde.march" (fun () ->
            gbm_march Sde.euler_maruyama ~steps:1
              (Nx.create f64 [| 1; 2 |] [| 0.; 1. |])
              (x0 3)));
    cases "march rejects times that are not strictly increasing inside the path"
      ~id:(fun at ->
        Printf.sprintf "[%s]"
          (String.concat "; " (Array.to_list (Array.map string_of_float at))))
      [
        [||];
        [| 1.; 0.5 |];
        [| 0.5; 0.5 |];
        [| 0.; Float.nan |];
        [| 0.; 2. |];
        [| -0.5; 0.5 |];
      ]
      (fun at ->
        raises_in "Jera.Sde.march" (fun () ->
            gbm_march Sde.euler_maruyama ~steps:1 (vec at) (x0 3)));
    test "march rejects a drift of another shape" (fun () ->
        raises_in "Jera.Sde.march" (fun () ->
            Sde.march tensor Sde.sra1 ~steps:1
              ~drift:(fun _ x -> Nx.sum x)
              ~diffusion:ou_diffusion w
              ~at:(vec [| 0.; 1. |])
              (x0 3)));
    test "march rejects a diffusion of another shape" (fun () ->
        raises_in "Jera.Sde.march" (fun () ->
            Sde.march tensor Sde.euler_maruyama ~steps:1 ~drift:gbm_drift
              ~diffusion:(fun _ _ dw -> Nx.sum dw)
              w
              ~at:(vec [| 0.; 1. |])
              (x0 3)));
  ]

let () =
  exit
    (run "Jera.Sde"
       [
         group "Brownian" brownian_tests;
         group "Brownian.v" v_tests;
         group "Brownian.increment" increment_tests;
         group "Brownian.walk" walk_tests;
         group "methods" method_tests;
         group "march" march_tests;
       ])
