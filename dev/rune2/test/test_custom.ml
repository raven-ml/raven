(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Custom differentiation rules: a rule replaces the derivative of its function
   in both modes and at every order, a custom_vjp has no forward derivative, the
   rule runs on its arguments' values where no transformation but the
   differentiation applying it tracks them and is traced elsewhere, and compiled
   functions apply it as eager code does. Deliberately wrong rules show which
   derivative is used. *)

open Windtrap
module Rune = Rune2
module Total = Rune.Total

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let tensor = Nx.Ptree.tensor
let pair = Nx.Ptree.(pair tensor tensor)
let v3 () = vec [| 0.7; -1.3; 2.1 |]
let along () = vec [| 0.5; 1.; -2. |]
let values x = Nx.to_array (Nx.reshape [| -1 |] x)

let near ?(tol = 1e-12) ?msg expected actual =
  equal ?msg (array (float tol)) (values expected) (values actual)

let exact ?msg expected actual =
  equal ?msg (array float_exact) (values expected) (values actual)

let invalid ?msg ~sub f = raises_match ?msg (Exn.invalid_arg ~substring:sub) f

(* sin with its true pullback, and with a pullback of 100 to show it is used. *)
let my_sin =
  Rune.custom_vjp tensor tensor (fun x ->
      (Nx.sin x, fun ct -> Nx.mul ct (Nx.cos x)))

let fake_grad_sin =
  Rune.custom_vjp tensor tensor (fun x ->
      (Nx.sin x, fun ct -> Nx.mul_s ct 100.))

(* sin with its true tangent map, and with a map of 100. *)
let my_sin_fwd =
  Rune.custom_jvp tensor tensor (fun x ->
      (Nx.sin x, fun dx -> Nx.mul dx (Nx.cos x)))

let fake_jvp_sin =
  Rune.custom_jvp tensor tensor (fun x ->
      (Nx.sin x, fun dx -> Nx.mul_s dx 100.))

let no_forward =
  "Rune.jvp': a custom_vjp rule has no forward derivative; give the function a \
   custom_jvp rule"

(* custom_vjp *)

let vjp_tests =
  [
    test "the rule replaces the derivative" (fun () ->
        exact (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (fake_grad_sin x)) (v3 ())));
    test "a true rule matches the function's derivative" (fun () ->
        near
          (Rune.grad' (fun x -> Nx.sum (Nx.sin x)) (v3 ()))
          (Rune.grad' (fun x -> Nx.sum (my_sin x)) (v3 ())));
    test "the rule composes inside a function" (fun () ->
        let x = v3 () in
        near
          (Nx.mul_s (Nx.mul (Nx.sin x) (Nx.cos x)) 2.)
          (Rune.grad'
             (fun x ->
               let s = my_sin x in
               Nx.sum (Nx.mul s s))
             x));
    test "with no differentiation the call is its rule's value" (fun () ->
        exact (Nx.sin (v3 ())) (my_sin (v3 ())));
    test "a call on a constant inside grad contributes nothing" (fun () ->
        let c = v3 () in
        let s = Nx.item [] (Nx.sum (Nx.sin c)) in
        near
          (vec [| 4. *. s |])
          (Rune.grad'
             (fun x -> Nx.mul (Nx.sum (Nx.mul x x)) (Nx.sum (my_sin c)))
             (vec [| 2. |])));
    test "a rule over two arguments gives each its gradient" (fun () ->
        let f =
          Rune.custom_vjp pair tensor (fun (a, b) ->
              (Nx.mul a b, fun ct -> (Nx.mul ct b, Nx.mul ct a)))
        in
        let a = v3 () and b = vec [| 1.9; 0.8; -0.6 |] in
        let ga, gb = Rune.grad pair (fun p -> Nx.sum (f p)) (a, b) in
        exact ~msg:"a" b ga;
        exact ~msg:"b" a gb);
    test "a pullback of another structure than the arguments is refused"
      (fun () ->
        let f =
          Rune.custom_vjp
            Nx.Ptree.(list tensor)
            tensor
            (fun p -> (Nx.sum (List.hd p), fun ct -> [ ct ]))
        in
        let args = [ v3 (); vec [| 1. |] ] in
        exact ~msg:"no differentiation applies it" (Nx.sum (v3 ())) (f args);
        invalid ~sub:"Rune.custom_vjp: the root: length" (fun () ->
            Rune.grad Nx.Ptree.(list tensor) (fun p -> f p) args));
    test "a structured result" (fun () ->
        let f =
          Rune.custom_vjp tensor pair (fun x ->
              ( (Nx.sin x, Nx.mul x x),
                fun (ds, dsq) ->
                  Nx.add (Nx.mul ds (Nx.cos x)) (Nx.mul dsq (Nx.mul_s x 2.)) ))
        in
        let x = v3 () in
        near ~msg:"both results"
          (Nx.add (Nx.cos x) (Nx.mul_s x 2.))
          (Rune.grad'
             (fun x ->
               let s, sq = f x in
               Nx.add (Nx.sum s) (Nx.sum sq))
             x);
        near ~msg:"an unused result has a zero cotangent" (Nx.cos x)
          (Rune.grad' (fun x -> Nx.sum (fst (f x))) x));
    test "a vjp pullback applies the rule each time" (fun () ->
        let y, pb = Rune.vjp' fake_grad_sin (v3 ()) in
        exact ~msg:"value" (Nx.sin (v3 ())) y;
        exact ~msg:"first" (Nx.mul_s (along ()) 100.) (pb (along ()));
        exact ~msg:"second" (Nx.full f64 [| 3 |] 100.)
          (pb (Nx.ones f64 [| 3 |])));
    test "jvp of a custom_vjp is refused" (fun () ->
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' my_sin (v3 ()) (along ())));
    test "a custom_vjp with no float result runs under forward mode" (fun () ->
        let runs = ref 0 in
        let tap =
          Rune.custom_vjp tensor Nx.Ptree.unit (fun y ->
              incr runs;
              ((), fun () -> Nx.zeros_like y))
        in
        let x = v3 () and v = along () in
        let _, dy =
          Rune.jvp'
            (fun x ->
              let y = Nx.sin x in
              tap y;
              y)
            x v
        in
        near ~msg:"tangent" (Nx.mul v (Nx.cos x)) dy;
        equal ~msg:"rule runs" int 1 !runs);
  ]

(* custom_jvp *)

let jvp_tests =
  [
    test "the tangent map replaces the derivative" (fun () ->
        exact (Nx.full f64 [| 3 |] 100.)
          (snd (Rune.jvp' fake_jvp_sin (v3 ()) (Nx.ones f64 [| 3 |]))));
    test "a true tangent map matches the function's derivative" (fun () ->
        near
          (snd (Rune.jvp' Nx.sin (v3 ()) (along ())))
          (snd (Rune.jvp' my_sin_fwd (v3 ()) (along ()))));
    test "a structured result" (fun () ->
        let f =
          Rune.custom_jvp tensor pair (fun x ->
              ( (Nx.sin x, Nx.mul x x),
                fun dx -> (Nx.mul dx (Nx.cos x), Nx.mul dx (Nx.mul_s x 2.)) ))
        in
        let x = v3 () and v = along () in
        near ~msg:"first"
          (Nx.mul v (Nx.cos x))
          (snd (Rune.jvp' (fun x -> fst (f x)) x v));
        near ~msg:"second"
          (Nx.mul v (Nx.mul_s x 2.))
          (snd (Rune.jvp' (fun x -> snd (f x)) x v)));
    test "an argument the differentiation does not track has a zero tangent"
      (fun () ->
        let seen = Total.make () in
        let f =
          Rune.custom_jvp pair tensor (fun (a, b) ->
              ( Nx.mul a b,
                fun (da, db) ->
                  Total.add seen db;
                  Nx.add (Nx.mul da b) (Nx.mul a db) ))
        in
        let b = vec [| 1.9; 0.8; -0.6 |] in
        let (_, dy), t =
          Total.collect seen ~init:(Nx.zeros f64 [| 3 |]) (fun () ->
              Rune.jvp' (fun a -> f (a, b)) (v3 ()) (along ()))
        in
        exact ~msg:"tangent" (Nx.mul (along ()) b) dy;
        exact ~msg:"the untracked tangent" (Nx.zeros f64 [| 3 |]) t);
    test "a tangent of another shape than the result is refused" (fun () ->
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              (Nx.sin x, fun dx -> Nx.sum dx))
        in
        let message =
          "Rune.custom_jvp: the root: shape [] in the tangent map's result, \
           [3] in the result"
        in
        exact ~msg:"no differentiation applies it" (Nx.sin (v3 ())) (f (v3 ()));
        invalid ~sub:message (fun () -> Rune.jvp' f (v3 ()) (along ()));
        invalid ~sub:message (fun () ->
            Rune.grad' (fun x -> Nx.sum (f x)) (v3 ())));
    test "under reverse mode the tangent map is transposed" (fun () ->
        near ~msg:"a true map"
          (Nx.cos (v3 ()))
          (Rune.grad' (fun x -> Nx.sum (my_sin_fwd x)) (v3 ()));
        exact ~msg:"the map's own derivative" (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (fake_jvp_sin x)) (v3 ())));
    test "with no differentiation the call is its rule's value" (fun () ->
        exact (Nx.sin (v3 ())) (my_sin_fwd (v3 ())));
    test "the rule runs once per call, its derivative once where applied"
      (fun () ->
        let rules = ref 0 and derivatives = ref 0 in
        let counted y d =
          incr rules;
          ( y,
            fun t ->
              incr derivatives;
              d t )
        in
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              counted (Nx.sin x) (fun dx -> Nx.mul dx (Nx.cos x)))
        and g =
          Rune.custom_vjp tensor tensor (fun x ->
              counted (Nx.sin x) (fun ct -> Nx.mul ct (Nx.cos x)))
        in
        let runs ~derivatives:n msg h =
          rules := 0;
          derivatives := 0;
          ignore (h (v3 ()));
          equal ~msg (Windtrap.pair int int) (1, n) (!rules, !derivatives)
        in
        let grad f = Rune.grad' (fun x -> Nx.sum (f x)) in
        runs ~derivatives:0 "eagerly" f;
        runs ~derivatives:1 "under jvp" (fun x -> Rune.jvp' f x (along ()));
        runs ~derivatives:1 "under grad" (grad f);
        runs ~derivatives:1 "under a vjp" (fun x ->
            snd (Rune.vjp' f x) (along ()));
        runs ~derivatives:1 "under jit"
          (Rune.jit' (fun x -> snd (Rune.jvp' f x (along ()))));
        runs ~derivatives:0 "a custom_vjp eagerly" g;
        runs ~derivatives:1 "a custom_vjp under grad" (grad g);
        (* A tangent map applies once, at the call; a pullback once per backward
           pass. *)
        let twice f x =
          let _, pb = Rune.vjp' f x in
          ignore (pb (along ()));
          pb (along ())
        in
        runs ~derivatives:1 "a vjp pullback of a custom_jvp, twice" (twice f);
        runs ~derivatives:2 "a vjp pullback of a custom_vjp, twice" (twice g));
    test "a custom_jvp with no float result runs once under reverse mode"
      (fun () ->
        let runs = ref 0 in
        let tap =
          Rune.custom_jvp tensor Nx.Ptree.unit (fun _ ->
              incr runs;
              ((), fun _ -> ()))
        in
        let loss tap x =
          let y = Nx.sin x in
          tap y;
          Nx.sum (Nx.mul y y)
        in
        let l, g = Rune.value_and_grad' (loss tap) (v3 ()) in
        let l', g' = Rune.value_and_grad' (loss ignore) (v3 ()) in
        exact ~msg:"loss" l' l;
        exact ~msg:"gradient" g' g;
        equal ~msg:"rule runs" int 1 !runs);
  ]

(* Integer result leaves *)

let int_leaf () = Nx.create Nx.int32 [| 2 |] [| 4l; 5l |]
let with_int = Nx.Ptree.(pair tensor tensor)

let int_tests =
  [
    test "a custom_jvp's integer result leaf has a zero tangent" (fun () ->
        let f =
          Rune.custom_jvp tensor with_int (fun x ->
              ( (Nx.sin x, int_leaf ()),
                fun dx -> (Nx.mul dx (Nx.cos x), Nx.zeros Nx.int32 [| 2 |]) ))
        in
        let x = v3 () in
        let _, (ds, dk) = Rune.jvp tensor with_int f x (along ()) in
        near ~msg:"float" (Nx.mul (along ()) (Nx.cos x)) ds;
        equal ~msg:"integer" (array int32) [| 0l; 0l |] (Nx.to_array dk);
        near ~msg:"under grad" (Nx.cos x)
          (Rune.grad' (fun x -> Nx.sum (fst (f x))) x));
    test "a custom_vjp's integer result leaf gets a zero cotangent" (fun () ->
        let f =
          Rune.custom_vjp tensor with_int (fun x ->
              ( (Nx.sin x, int_leaf ()),
                fun (g, gk) ->
                  Nx.add (Nx.mul g (Nx.cos x)) (Nx.cast f64 (Nx.sum gk)) ))
        in
        let x = v3 () in
        near (Nx.cos x) (Rune.grad' (fun x -> Nx.sum (fst (f x))) x));
  ]

(* Second order *)

(* The softplus Hessian at 0: d²/dx² (softplus x)² = 2σ² + 2 softplus · σ'. *)
let softplus_hessian_at_0 = (2. *. 0.5 *. 0.5) +. (2. *. Float.log 2. *. 0.25)

let stable x =
  Nx.add (Nx.maximum_s x 0.) (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) 1.))

let softplus =
  Rune.custom_jvp tensor tensor (fun x ->
      (stable x, fun dx -> Nx.mul (Nx.sigmoid x) dx))

let squared x = Nx.sum (Nx.square (softplus x))
let hvp f x = snd (Rune.jvp' (Rune.grad' f) x (Nx.ones_like x))

(* Twice the function's derivative, so that an order that applies the rule
   differs from one that differentiates the function's code. The derivative
   reads [cos x], a residual the rule computes. *)
let doubled =
  Rune.custom_jvp tensor tensor (fun x ->
      let c = Nx.mul_s (Nx.cos x) 2. in
      (Nx.sin x, fun dx -> Nx.mul c dx))

let doubled_vjp =
  Rune.custom_vjp tensor tensor (fun x ->
      let c = Nx.mul_s (Nx.cos x) 2. in
      (Nx.sin x, fun g -> Nx.mul c g))

let total : (float, Nx.float64_elt, _) Total.t = Total.make ()

(* A rule with no tensor in its result, whose tangent map adds the sum of the
   tangents to [total]. *)
let mark =
  Rune.custom_jvp tensor Nx.Ptree.unit (fun _ ->
      ((), fun dy -> Total.add total (Nx.sum dy)))

(* f x = x³ and its derivative 3x², with an addition of x in its rule. *)
let cube =
  Rune.custom_jvp tensor tensor (fun x ->
      Total.add total (Nx.sum x);
      (Nx.mul x (Nx.square x), fun dx -> Nx.mul (Nx.mul_s (Nx.square x) 3.) dx))

let order_tests =
  let x0 = scalar 0. and x = 0.4 in
  let d2 f = Rune.grad' (fun x -> Nx.sum (Rune.grad' f x)) in
  [
    test "the softplus Hessian at 0 in every pair of modes" (fun () ->
        let h = scalar softplus_hessian_at_0 in
        near ~msg:"forward over reverse" h
          (Rune.jacfwd' (Rune.grad' squared) x0);
        near ~msg:"reverse over reverse" h (d2 squared x0);
        near ~msg:"jvp over reverse" h (hvp squared x0);
        near ~msg:"forward over forward" h
          (snd
             (Rune.jvp'
                (fun x -> snd (Rune.jvp' squared x (scalar 1.)))
                x0 (scalar 1.)));
        near ~msg:"reverse over forward" h
          (Rune.grad' (fun x -> snd (Rune.jvp' squared x (scalar 1.))) x0));
    test "the first derivative is the rule's" (fun () ->
        let d = scalar (2. *. Float.cos x) in
        near ~msg:"grad" d (Rune.grad' (fun x -> Nx.sum (doubled x)) (scalar x));
        near ~msg:"jvp" d (snd (Rune.jvp' doubled (scalar x) (scalar 1.))));
    test "an enclosing differentiation applies the rule to the inner primal"
      (fun () ->
        let d = scalar (2. *. Float.cos x *. 3.) in
        near ~msg:"forward" d
          (snd
             (Rune.jvp'
                (fun x -> fst (Rune.jvp' doubled x (scalar 1.)))
                (scalar x) (scalar 3.)));
        near ~msg:"reverse"
          (scalar (2. *. Float.cos x))
          (Rune.grad'
             (fun x -> Nx.sum (fst (Rune.jvp' doubled x (scalar 1.))))
             (scalar x)));
    test "the second derivative differentiates the tangent map's code"
      (fun () ->
        let d = scalar (-2. *. Float.sin x) and f x = Nx.sum (doubled x) in
        near ~msg:"forward over reverse" d
          (Rune.jacfwd' (Rune.grad' f) (scalar x));
        near ~msg:"reverse over reverse" d (d2 f (scalar x));
        near ~msg:"reverse over forward" d
          (Rune.grad' (fun x -> snd (Rune.jvp' f x (scalar 1.))) (scalar x));
        near ~msg:"forward over forward" d
          (snd
             (Rune.jvp'
                (fun x -> snd (Rune.jvp' f x (scalar 1.)))
                (scalar x) (scalar 1.))));
    test "the third derivative differentiates the tangent map's code" (fun () ->
        let f x = Nx.sum (doubled x) in
        near
          (scalar (-2. *. Float.cos x))
          (d2 (fun x -> Nx.sum (Rune.grad' f x)) (scalar x)));
    test "a custom_vjp's second derivative differentiates its pullback's code"
      (fun () ->
        let d = scalar (-2. *. Float.sin x) and f x = Nx.sum (doubled_vjp x) in
        near ~msg:"reverse over reverse" d (d2 f (scalar x));
        near ~msg:"forward over reverse" d (hvp f (scalar x));
        near ~msg:"third order"
          (scalar (-2. *. Float.cos x))
          (d2 (fun x -> Nx.sum (Rune.grad' f x)) (scalar x));
        (* The rule's addition counts once at every order. *)
        let t = Total.make () in
        let adds =
          Rune.custom_vjp tensor tensor (fun x ->
              Total.add t (Nx.sum x);
              (Nx.sin x, fun g -> Nx.mul (Nx.mul_s (Nx.cos x) 2.) g))
        in
        let _, total =
          Total.collect t ~init:(scalar 0.) (fun () ->
              d2 (fun x -> Nx.sum (adds x)) (scalar x))
        in
        exact ~msg:"one addition" (scalar x) total);
    test "a forward level between reverse levels keeps a custom_vjp's rule"
      (fun () ->
        let f x = Nx.sum (doubled_vjp x) in
        let d3 = scalar (-2. *. Float.cos x) in
        near ~msg:"forward over reverse over reverse" d3
          (snd
             (Rune.jvp'
                (Rune.grad' (fun x -> Nx.sum (Rune.grad' f x)))
                (scalar x) (scalar 1.)));
        near ~msg:"under a map"
          (vec [| -2. *. Float.cos x; -2. *. Float.cos 0.1 |])
          (Rune.vmap'
             (fun x ->
               snd
                 (Rune.jvp'
                    (Rune.grad' (fun x -> Nx.sum (Rune.grad' f x)))
                    x (Nx.ones_like x)))
             (vec [| x; 0.1 |]));
        (* f = Σ (sin x)², with a pullback of twice the derivative: the outer
           reverse mode applies it to sin x, the forward mode in the middle
           differentiates the rule's code. *)
        let g x = Nx.sum (Nx.square (doubled_vjp x)) in
        let s = Float.sin x and c = Float.cos x in
        near ~msg:"reverse over forward over reverse"
          (scalar (-20. *. s *. c))
          (Rune.grad' (fun x -> Nx.sum (hvp g x)) (scalar x)));
    test "an outer reverse mode applies a custom_vjp's pullback to the primal"
      (fun () ->
        let primal x = fst (Rune.vjp' doubled_vjp x) in
        near ~msg:"reverse over reverse"
          (scalar (2. *. Float.cos x))
          (Rune.grad' (fun x -> Nx.sum (primal x)) (scalar x));
        near ~msg:"through value_and_grad"
          (scalar (2. *. Float.cos x))
          (Rune.grad'
             (fun x ->
               fst (Rune.value_and_grad' (fun x -> Nx.sum (doubled_vjp x)) x))
             (scalar x));
        near ~msg:"forward over reverse differentiates the rule's code"
          (scalar (Float.cos x))
          (snd (Rune.jvp' primal (scalar x) (scalar 1.))));
    test "a differentiation of a capture leaves the rule to the argument's"
      (fun () ->
        (* s sin x with a tangent map of 100 s along x: the differentiation over
           s differentiates the code, the one over x applies the rule. *)
        let rule s =
          Rune.custom_jvp tensor tensor (fun x ->
              let c = Nx.mul_s s 100. in
              (Nx.mul s (Nx.sin x), fun dx -> Nx.mul c dx))
        in
        let s = scalar 1.5 and dx = scalar 3. in
        let reverse x = Rune.value_and_grad' (fun s -> Nx.sum (rule s x)) s in
        let forward x = Rune.jvp' (fun s -> rule s x) s (scalar 1.) in
        let jvp f = snd (Rune.jvp' f (scalar x) dx) in
        near ~msg:"jvp over reverse" (scalar 450.)
          (jvp (fun x -> fst (reverse x)));
        near ~msg:"jvp over forward" (scalar 450.)
          (jvp (fun x -> fst (forward x)));
        near ~msg:"under jit" (scalar 450.)
          (Rune.jit'
             (fun x -> snd (Rune.jvp' (fun x -> fst (reverse x)) x dx))
             (scalar x));
        near ~msg:"under a map"
          (vec [| 450.; 450. |])
          (Rune.vmap'
             (fun x -> snd (Rune.jvp' (fun x -> fst (reverse x)) x dx))
             (vec [| x; 0.1 |]));
        near ~msg:"grad over reverse" (scalar 150.)
          (Rune.grad' (fun x -> fst (reverse x)) (scalar x));
        near ~msg:"grad over forward" (scalar 150.)
          (Rune.grad' (fun x -> fst (forward x)) (scalar x));
        near ~msg:"the derivative along s"
          (scalar (Float.sin x))
          (snd (reverse (scalar x)));
        near ~msg:"its derivative along x"
          (scalar (3. *. Float.cos x))
          (jvp (fun x -> snd (reverse x)));
        (* The rule applied inside, its tangent map's code along s outside. *)
        let applied s = Rune.grad' (fun x -> Nx.sum (rule s x)) (scalar x) in
        near ~msg:"jvp of the rule's derivative" (scalar 100.)
          (snd (Rune.jvp' applied s (scalar 1.)));
        near ~msg:"grad of the rule's derivative" (scalar 100.)
          (Rune.grad' (fun s -> Nx.sum (applied s)) s);
        (* A capture whose primal the differentiation of x tracks. *)
        invalid ~sub:"its own differentiation tracks; pass it as an argument"
          (fun () ->
            jvp (fun x ->
                fst
                  (Rune.value_and_grad'
                     (fun s -> Nx.sum (rule s x))
                     (Nx.mul_s x 2.)))));
    test "a differentiation of a capture differentiates the rule's additions"
      (fun () ->
        (* The rule adds s² to [total]: from a zero of s, the total's derivative
           along s is 1 + 2 s = 4. *)
        let rule s =
          Rune.custom_jvp tensor tensor (fun x ->
              Total.add total (Nx.mul s s);
              (Nx.mul s x, fun dx -> dx))
        in
        let collected ~zero x s =
          snd (Total.collect total ~init:(zero s) (fun () -> ignore (rule s x)))
        in
        let s = scalar 1.5 and x = scalar 2. in
        let at_zero _ = scalar 0. in
        near ~msg:"an untraced argument" (scalar 3.)
          (Rune.grad' (collected ~zero:at_zero x) s);
        near ~msg:"from a zero of s" (scalar 4.)
          (Rune.grad' (collected ~zero:Fun.id x) s);
        near ~msg:"under jvp of the argument" (scalar 3.)
          (fst
             (Rune.jvp'
                (fun x -> Rune.grad' (collected ~zero:at_zero x) s)
                x (scalar 1.)));
        near ~msg:"under jvp of the argument, from a zero of s" (scalar 4.)
          (fst
             (Rune.jvp'
                (fun x -> Rune.grad' (collected ~zero:Fun.id x) s)
                x (scalar 1.)));
        near ~msg:"under grad of the argument" (scalar 4.)
          (fst
             (Rune.value_and_grad'
                (fun x ->
                  snd (Rune.jvp' (collected ~zero:Fun.id x) s (scalar 1.)))
                x)));
    test "a custom_vjp under a differentiation of a capture" (fun () ->
        let rule s =
          Rune.custom_vjp tensor tensor (fun x ->
              let c = Nx.mul_s s 100. in
              (Nx.mul s (Nx.sin x), fun g -> Nx.mul c g))
        in
        let s = scalar 1.5 in
        let reverse x = Rune.value_and_grad' (fun s -> Nx.sum (rule s x)) s in
        let forward x = Rune.jvp' (fun s -> rule s x) s (scalar 1.) in
        near ~msg:"grad over reverse" (scalar 150.)
          (Rune.grad' (fun x -> fst (reverse x)) (scalar x));
        near ~msg:"grad over forward" (scalar 150.)
          (Rune.grad' (fun x -> fst (forward x)) (scalar x));
        near ~msg:"the derivative along s"
          (scalar (Float.sin x))
          (snd (reverse (scalar x)));
        (* The pullback applied inside, its code along s outside. *)
        near ~msg:"jvp of the pullback" (scalar 100.)
          (snd
             (Rune.jvp'
                (fun s -> Rune.grad' (fun x -> Nx.sum (rule s x)) (scalar x))
                s (scalar 1.)));
        (* No reverse mode over x applies the pullback. *)
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' (fun x -> fst (reverse x)) (scalar x) (scalar 1.)));
    test "a rule's addition counts once in every pair of modes" (fun () ->
        let x = scalar 0.5 in
        let check msg f =
          let _, t = Total.collect total ~init:(scalar 0.) f in
          exact ~msg (scalar 0.5) t
        in
        let f x = Nx.sum (cube x) in
        check "jvp ∘ jvp" (fun () ->
            Rune.jvp' (fun x -> snd (Rune.jvp' f x (scalar 1.))) x (scalar 1.));
        check "grad ∘ jvp" (fun () ->
            Rune.grad' (fun x -> snd (Rune.jvp' f x (scalar 1.))) x);
        check "jvp ∘ grad" (fun () -> hvp f x);
        check "grad ∘ grad" (fun () -> d2 f x);
        (* A rule that captures s, under a differentiation of s alone. *)
        let scaled s =
          Rune.custom_jvp tensor tensor (fun x ->
              Total.add total (Nx.sum x);
              (Nx.mul s x, fun dx -> Nx.mul s dx))
        in
        let s = scalar 1.5 in
        let g s x = Nx.sum (scaled s x) in
        check "grad of a capture" (fun () -> Rune.grad' (fun s -> g s x) s);
        check "jvp ∘ grad of a capture" (fun () ->
            Rune.jvp' (fun x -> Rune.grad' (fun s -> g s x) s) x (scalar 1.));
        check "grad ∘ jvp of a capture" (fun () ->
            Rune.grad'
              (fun x -> snd (Rune.jvp' (fun s -> g s x) s (scalar 1.)))
              x);
        near ~msg:"the third derivative" (scalar 6.)
          (d2 (fun x -> Nx.sum (Rune.grad' f x)) x));
    test "a rule with no tensor result is not applied under grad" (fun () ->
        let g, t =
          Total.collect total ~init:(scalar 0.) (fun () ->
              Rune.grad'
                (fun x ->
                  mark x;
                  Nx.sum (Nx.mul x x))
                (vec [| 1.; 2. |]))
        in
        exact ~msg:"gradient" (vec [| 2.; 4. |]) g;
        exact ~msg:"total" (scalar 0.) t);
    test "a rule with no tensor result observes the tangents under jvp"
      (fun () ->
        let _, t =
          Total.collect total ~init:(scalar 0.) (fun () ->
              Rune.jvp'
                (fun x ->
                  mark x;
                  Nx.sum (Nx.mul x x))
                (vec [| 1.; 2. |])
                (vec [| 3.; -1. |]))
        in
        exact (scalar 2.) t);
  ]

(* Reads *)

(* √x by Newton's iteration from x, which stops once the residual it reads with
   [Nx.item] is small: the number of steps depends on the argument's values.
   [steps] counts them. *)
let steps = ref 0

let newton x =
  let rec go y n =
    let r = Nx.item [] (Nx.max (Nx.abs (Nx.sub (Nx.mul y y) x))) in
    if r <= 1e-12 || n = 100 then y
    else begin
      incr steps;
      go (Nx.mul_s (Nx.add y (Nx.div x y)) 0.5) (n + 1)
    end
  in
  go x 0

let newton_jvp =
  Rune.custom_jvp tensor tensor (fun x ->
      let y = newton x in
      (y, fun dx -> Nx.div dx (Nx.mul_s y 2.)))

let newton_vjp =
  Rune.custom_vjp tensor tensor (fun x ->
      let y = newton x in
      (y, fun g -> Nx.div g (Nx.mul_s y 2.)))

(* [z] with [a z = b], for positive [a], by Richardson's iteration, which stops
   once the residual it reads is small. *)
let solve a b =
  let rate = 1. /. Nx.item [] (Nx.max a) in
  let rec go z n =
    let r = Nx.sub b (Nx.mul a z) in
    let size = Nx.item [] (Nx.max (Nx.abs r)) in
    if size <= 1e-14 || n = 1000 then z
    else go (Nx.add z (Nx.mul_s r rate)) (n + 1)
  in
  go (Nx.zeros_like b) 0

let solving_jvp =
  Rune.custom_jvp tensor tensor (fun x ->
      let y = newton x in
      (y, fun dx -> solve (Nx.mul_s y 2.) dx))

let solving_vjp =
  Rune.custom_vjp tensor tensor (fun x ->
      let y = newton x in
      (y, fun g -> solve (Nx.mul_s y 2.) g))

let read_tests =
  let x = vec [| 4.; 9.; 2. |] in
  let root = Nx.sqrt x in
  let half = Nx.div (Nx.ones f64 [| 3 |]) (Nx.mul_s root 2.) in
  let sum f x = Nx.sum (f x) in
  [
    test "a rule reads its arguments eagerly and under one differentiation"
      (fun () ->
        (* The steps the iteration takes outside every rule. *)
        steps := 0;
        ignore (newton x);
        let expected = !steps in
        not_equal ~msg:"steps" int 0 expected;
        let counted msg f =
          steps := 0;
          let r = f () in
          equal ~msg:(msg ^ ", steps") int expected !steps;
          r
        in
        near ~msg:"eagerly" root (counted "eagerly" (fun () -> newton_jvp x));
        near ~msg:"a custom_vjp eagerly" root
          (counted "a custom_vjp eagerly" (fun () -> newton_vjp x));
        near ~msg:"jvp"
          (Nx.mul (along ()) half)
          (counted "jvp" (fun () -> snd (Rune.jvp' newton_jvp x (along ()))));
        near ~msg:"grad" half
          (counted "grad" (fun () -> Rune.grad' (sum newton_jvp) x));
        near ~msg:"grad of a custom_vjp" half
          (counted "grad of a custom_vjp" (fun () ->
               Rune.grad' (sum newton_vjp) x));
        steps := 0;
        near ~msg:"at a root" (Nx.ones f64 [| 3 |])
          (newton_jvp (Nx.ones f64 [| 3 |]));
        equal ~msg:"at a root, steps" int 0 !steps;
        (* A map and a loop step that capture the argument alone run the rule as
           written: the step once per loop call. *)
        let cs = Nx.create f64 [| 2; 1 |] [| 0.5; -2. |] in
        near ~msg:"a map over captures"
          (Nx.mul cs (Nx.reshape [| 1; 3 |] root))
          (counted "a map over captures" (fun () ->
               Rune.vmap' (fun c -> Nx.mul c (newton_jvp x)) cs));
        near ~msg:"a loop step over captures"
          (Nx.mul cs (Nx.reshape [| 1; 3 |] root))
          (counted "a loop step over captures" (fun () ->
               snd
                 (Rune.scan Nx.Ptree.unit tensor tensor
                    ~f:(fun () c -> ((), Nx.mul c (newton_jvp x)))
                    ~init:() cs))));
    test "a read in the rule raises where the rule is traced" (fun () ->
        let traced = "Nx.item: the value is traced by Rune.custom_jvp" in
        let rows = Nx.reshape [| 3; 1 |] x in
        invalid ~sub:traced (fun () -> Rune.jit' newton_jvp x);
        invalid ~sub:traced (fun () -> Rune.vmap' newton_jvp rows);
        invalid ~sub:traced (fun () ->
            Rune.scan Nx.Ptree.unit tensor tensor
              ~f:(fun () r -> ((), newton_jvp r))
              ~init:() rows);
        invalid ~sub:traced (fun () ->
            Rune.grad' (fun x -> Nx.sum (Rune.grad' (sum newton_jvp) x)) x);
        invalid ~sub:traced (fun () ->
            Rune.remat Nx.Ptree.(tensor @-> returns tensor) newton_jvp x);
        invalid ~sub:traced (fun () ->
            Rune.switch tensor (Nx.scalar Nx.int32 0l)
              [ (fun () -> newton_jvp x) ]);
        invalid ~sub:traced (fun () ->
            Rune.root tensor
              ~residual:(fun y -> Nx.sub (Nx.mul y y) x)
              (fun () -> newton_jvp x));
        (* A map over captures alone, under a differentiation of the
           argument. *)
        invalid ~sub:traced (fun () ->
            Rune.grad'
              (fun x ->
                let each c = Nx.mul c (newton_jvp x) in
                Nx.sum (Rune.vmap' each (vec [| 1. |])))
              x);
        (* A loop step that captures an argument a differentiation tracks. *)
        invalid ~sub:traced (fun () ->
            Rune.grad'
              (fun x ->
                Nx.sum
                  (snd
                     (Rune.scan Nx.Ptree.unit tensor tensor
                        ~f:(fun () c -> ((), Nx.mul c (newton_jvp x)))
                        ~init:() (vec [| 1. |]))))
              x);
        (* A differentiation between the call and the one tracking the
           argument. *)
        invalid ~sub:traced (fun () ->
            Rune.jvp'
              (fun x ->
                Rune.grad'
                  (fun s -> Nx.sum (Nx.mul s (newton_jvp x)))
                  (scalar 1.))
              x (along ()));
        invalid ~sub:"Nx.item: the value is traced by Rune.custom_vjp"
          (fun () ->
            Rune.grad' (fun x -> Nx.sum (Rune.grad' (sum newton_vjp) x)) x));
    test "a derivative reads its tangents and cotangents where it applies"
      (fun () ->
        (* The tangent map and the pullback solve their linear equation by an
           iteration that reads its residual, so their steps depend on the
           tangents' values. *)
        near ~msg:"jvp"
          (Nx.mul (along ()) half)
          (snd (Rune.jvp' solving_jvp x (along ())));
        near ~msg:"grad" half (Rune.grad' (sum solving_vjp) x);
        let _, pb = Rune.vjp' solving_vjp x in
        near ~msg:"a vjp pullback" (Nx.mul (along ()) half) (pb (along ()));
        near ~msg:"the pullback again" half (pb (Nx.ones f64 [| 3 |]));
        near ~msg:"a zero cotangent" (Nx.zeros f64 [| 3 |])
          (pb (Nx.zeros f64 [| 3 |])));
    test "a read in a derivative raises where it is traced" (fun () ->
        let traced = "Nx.item: the value is traced" in
        (* Under reverse mode the tangent map runs on the tape's tangents. *)
        invalid ~sub:traced (fun () -> Rune.grad' (sum solving_jvp) x);
        invalid ~sub:traced (fun () ->
            Rune.jit' (Rune.grad' (sum solving_vjp)) x);
        invalid ~sub:traced (fun () ->
            Rune.jit' (fun x -> snd (Rune.jvp' solving_jvp x (along ()))) x);
        invalid ~sub:traced (fun () ->
            Rune.grad' (fun x -> Nx.sum (Rune.grad' (sum solving_vjp) x)) x);
        let _, pb = Rune.vjp' solving_vjp x in
        invalid ~sub:traced (fun () -> Rune.vmap' pb (Nx.reshape [| 1; 3 |] x)));
    (* murder's dilqr: a rule whose primal and derivative read a sum. *)
    test "a rule that reads its arguments' sum" (fun () ->
        let x = vec [| 1.; 2. |] in
        let scaled y = Nx.mul_s y (Nx.item [] (Nx.sum x)) in
        let f =
          Rune.custom_vjp tensor tensor (fun x ->
              let s = Nx.item [] (Nx.sum x) in
              (Nx.mul_s x s, fun g -> Nx.mul_s g s))
        and g =
          Rune.custom_jvp tensor tensor (fun x ->
              let s = Nx.item [] (Nx.sum x) in
              (Nx.mul_s x s, fun dx -> Nx.mul_s dx s))
        in
        exact ~msg:"custom_vjp eagerly" (scaled x) (f x);
        exact ~msg:"custom_vjp under grad"
          (scaled (Nx.ones f64 [| 2 |]))
          (Rune.grad' (sum f) x);
        exact ~msg:"custom_jvp eagerly" (scaled x) (g x);
        exact ~msg:"custom_jvp under jvp" (scaled x) (snd (Rune.jvp' g x x)));
  ]

(* Errors *)

let tracked =
  "the rule uses a value its own differentiation tracks; pass it as an argument"

let nonlinear f = Rune.custom_jvp tensor tensor (fun x -> (Nx.sin x, f))

let error_tests =
  [
    test "a pullback's result is checked in the backward pass" (fun () ->
        let f =
          Rune.custom_vjp tensor tensor (fun x -> (x, fun g -> Nx.sum g))
        in
        let _, pb = Rune.vjp' f (v3 ()) in
        invalid
          ~sub:
            "Rune.custom_vjp: the root: shape [] in the pullback's result, [3] \
             in the arguments" (fun () -> pb (along ())));
    test "a draw without a key in the rule raises" (fun () ->
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              (Nx.add x (Nx.Rng.uniform f64 [| 3 |]), Fun.id))
        in
        let message =
          "Nx.Rng.uniform: no key inside Rune.custom_jvp, which refuses \
           keyless draws"
        in
        raises ~msg:"eagerly" (Invalid_argument message) (fun () -> f (v3 ()));
        raises ~msg:"under grad" (Invalid_argument message) (fun () ->
            Rune.grad' (fun x -> Nx.sum (f x)) (v3 ()));
        raises ~msg:"under jit" (Invalid_argument message) (fun () ->
            Rune.jit' f (v3 ()));
        let drawing t = Nx.add t (Nx.Rng.uniform f64 [| 3 |]) in
        let g = Rune.custom_jvp tensor tensor (fun x -> (x, drawing))
        and h = Rune.custom_vjp tensor tensor (fun x -> (x, drawing)) in
        raises ~msg:"in a tangent map under jvp" (Invalid_argument message)
          (fun () -> Rune.jvp' g (v3 ()) (along ()));
        let scaling =
          Rune.custom_jvp tensor tensor (fun x ->
              (x, fun t -> Nx.mul t (Nx.Rng.uniform f64 [| 3 |])))
        in
        raises ~msg:"in a tangent map under grad" (Invalid_argument message)
          (fun () -> Rune.grad' (fun x -> Nx.sum (scaling x)) (v3 ()));
        raises ~msg:"in a pullback under grad"
          (Invalid_argument
             "Nx.Rng.uniform: no key inside Rune.custom_vjp, which refuses \
              keyless draws") (fun () ->
            Rune.grad' (fun x -> Nx.sum (h x)) (v3 ())));
    test "a pullback on values may not take a value its differentiation tracks"
      (fun () ->
        (* It runs in the backward pass, after the differentiation returns. *)
        let through pb x =
          let w = Nx.mul x x in
          Nx.sum (Rune.custom_vjp tensor tensor (fun y -> (y, pb w)) x)
        in
        let returned = "used after Rune.grad' returned" in
        invalid ~sub:returned (fun () ->
            Rune.grad' (through (fun w g -> Nx.mul g w)) (v3 ()));
        invalid ~sub:returned (fun () ->
            Rune.grad'
              (through (fun w g -> Nx.mul_s g (Nx.item [ 0 ] w)))
              (v3 ())));
    test "an exception of the tangent map reaches the call that applies it"
      (fun () ->
        let f =
          Rune.custom_jvp tensor tensor (fun x -> (x, fun _ -> failwith "map"))
        in
        exact ~msg:"no differentiation applies it" (v3 ()) (f (v3 ()));
        raises (Failure "map") (fun () -> Rune.jvp' f (v3 ()) (along ())));
    test "an exception of the rule keeps the backtrace of its raise" (fun () ->
        let raise_exit () = raise (Sys.opaque_identity Exit) in
        let f =
          Rune.custom_jvp tensor tensor (fun _ -> (raise_exit (), Fun.id))
        in
        let check msg g =
          equal ~msg string "test_custom.ml"
            (Nx_support.raised_in (fun () -> g (v3 ())))
        in
        check "eagerly" f;
        check "under jvp" (fun x -> Rune.jvp' f x (along ()));
        check "under grad" (Rune.grad' (fun x -> Nx.sum (f x)));
        check "under a map" (Rune.vmap' f);
        check "under jit" (Rune.jit' f);
        check "in a collect" (fun x ->
            Total.collect total ~init:(scalar 0.) (fun () -> f x)));
    test "a rule that uses a value its differentiation tracks raises" (fun () ->
        let uses rule x =
          let w = Nx.mul x x in
          Nx.sum (rule w x)
        in
        invalid ~sub:("Rune.custom_jvp: " ^ tracked) (fun () ->
            Rune.grad'
              (uses (fun w ->
                   Rune.custom_jvp tensor tensor (fun y ->
                       (Nx.mul y w, fun dy -> Nx.mul dy w))))
              (v3 ()));
        invalid ~sub:("Rune.custom_vjp: " ^ tracked) (fun () ->
            Rune.grad'
              (uses (fun w ->
                   Rune.custom_vjp tensor tensor (fun y ->
                       (Nx.mul y w, fun g -> Nx.mul g w))))
              (v3 ()));
        invalid ~sub:("Rune.custom_jvp: " ^ tracked) (fun () ->
            Rune.jvp'
              (uses (fun w ->
                   Rune.custom_jvp tensor tensor (fun y ->
                       (Nx.mul y w, fun dy -> Nx.mul dy w))))
              (v3 ()) (along ())));
    test "a rule may use an enclosing differentiation's value" (fun () ->
        (* The inner gradient of Σ x·w is w, so the outer one of its sum is 1
           per element: the outer grad differentiates the rule's code. *)
        let outer w =
          let g =
            Rune.custom_jvp tensor tensor (fun x ->
                (Nx.mul x w, fun dx -> Nx.mul dx w))
          in
          Nx.sum (Rune.grad' (fun x -> Nx.sum (g x)) (vec [| 1.; 1. |]))
        in
        exact (vec [| 1.; 1. |]) (Rune.grad' outer (vec [| 3.; 4. |]));
        let tracked_only w =
          let g =
            Rune.custom_jvp tensor tensor (fun x ->
                (Nx.mul x w, fun dx -> Nx.mul_s dx 100.))
          in
          Nx.sum (g (vec [| 1.; 2. |]))
        in
        exact ~msg:"a rule over constants is its code"
          (vec [| 1.; 2. |])
          (Rune.grad' tracked_only (vec [| 3.; 4. |])));
    test "a value another differentiation tracks is a constant of the rule"
      (fun () ->
        (* [x] is the outer function's; the inner jvp applies the rule. *)
        let inner x =
          let f =
            Rune.custom_jvp tensor tensor (fun y ->
                (Nx.mul x y, fun dy -> Nx.mul x dy))
          in
          Nx.sum (snd (Rune.jvp' f (vec [| 1.; 2.; 3. |]) (along ())))
        in
        near (along ()) (Rune.grad' inner (v3 ())));
    test "a tangent map that is not linear raises under reverse mode" (fun () ->
        let f = nonlinear Nx.exp in
        near ~msg:"forward mode applies it"
          (Nx.exp (along ()))
          (snd (Rune.jvp' f (v3 ()) (along ())));
        invalid
          ~sub:
            "Rune.grad': a custom_jvp tangent map applies Exp to a tangent; a \
             tangent map must be linear in its tangents" (fun () ->
            Rune.grad' (fun x -> Nx.sum (f x)) (v3 ()));
        exact ~msg:"a value added to a tangent counts as zero"
          (Nx.ones f64 [| 3 |])
          (Rune.grad'
             (fun x -> Nx.sum (nonlinear (fun d -> Nx.add_s d 1.) x))
             (v3 ()));
        let _, pb = Rune.vjp' f (v3 ()) in
        invalid ~sub:"Rune.vjp': a custom_jvp tangent map applies Exp"
          (fun () -> pb (along ()));
        (* An iterate whose stop reads a tangent takes a number of steps that
           depends on it. *)
        let halved =
          nonlinear
            (Rune.iterate' ~max:4
               ~until:(fun d -> Nx.less (Nx.max d) (Nx.scalar f64 1.))
               ~f:(fun d -> Nx.mul_s d 0.5))
        in
        invalid
          ~sub:
            "Rune.grad': a custom_jvp tangent map applies an iterate's stop to \
             a tangent; a tangent map must be linear in its tangents" (fun () ->
            Rune.grad' (fun x -> Nx.sum (halved x)) (v3 ())));
    test "a tangent map that adds a tangent to a total raises under grad"
      (fun () ->
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              ( Nx.sin x,
                fun dx ->
                  Total.add total (Nx.sum dx);
                  Nx.mul dx (Nx.cos x) ))
        in
        invalid
          ~sub:
            "Rune.Total.add: a custom_jvp tangent map adds a tangent under \
             reverse mode; a total takes values" (fun () ->
            Total.collect total ~init:(scalar 0.) (fun () ->
                Rune.grad' (fun x -> Nx.sum (f x)) (v3 ()))));
  ]

(* Inside constructs *)

let compiled_tests =
  [
    test "a custom_jvp under jit" (fun () ->
        let x = v3 () in
        exact ~msg:"value" (Nx.sin x) (Rune.jit' fake_jvp_sin x);
        let f =
          Rune.jit' (fun x -> snd (Rune.jvp' fake_jvp_sin x (Nx.mul_s x 2.)))
        in
        List.iter
          (fun x -> exact ~msg:"jit of jvp" (Nx.mul_s x 200.) (f x))
          [ v3 (); vec [| -0.3; 0.4; 1.2 |] ];
        exact ~msg:"jvp of jit"
          (Nx.mul_s (along ()) 100.)
          (snd (Rune.jvp' (Rune.jit' fake_jvp_sin) x (along ())));
        exact ~msg:"grad of jit" (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (Rune.jit' fake_jvp_sin x)) x);
        exact ~msg:"jit of grad" (Nx.full f64 [| 3 |] 100.)
          (Rune.jit' (Rune.grad' (fun x -> Nx.sum (fake_jvp_sin x))) x));
    test "a custom_vjp under jit" (fun () ->
        let f =
          Rune.jit' (Rune.grad' (fun x -> Nx.sum (Nx.mul (fake_grad_sin x) x)))
        in
        List.iter
          (fun x -> near (Nx.add (Nx.mul_s x 100.) (Nx.sin x)) (f x))
          [ v3 (); vec [| -0.3; 0.4; 1.2 |] ];
        exact ~msg:"grad of jit" (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (Rune.jit' fake_grad_sin x)) (v3 ())));
    test "a custom_vjp whose pullback scatters, under jit" (fun () ->
        let indices = Nx.create Nx.int64 [| 3 |] [| 2L; 0L; 2L |] in
        let take =
          Rune.custom_vjp tensor tensor (fun x ->
              ( Nx.take ~axis:0 indices x,
                fun ct ->
                  Nx.scatter ~combine:Add ~axis:0 indices (Nx.mul_s ct 7.)
                    (Nx.zeros_like x) ))
        in
        let f =
          Rune.jit'
            (Rune.grad' (fun x ->
                 let y = take x in
                 Nx.sum (Nx.mul y y)))
        in
        near ~msg:"first"
          (vec [| 14.; 0.; 84.; 0. |])
          (f (vec [| 1.; 2.; 3.; 4. |]));
        near ~msg:"again"
          (vec [| 28.; 0.; 112.; 0. |])
          (f (vec [| 2.; 3.; 4.; 5. |])));
    test "a custom rule in a scan's step" (fun () ->
        let step c x = (Nx.add c (fake_jvp_sin x), ()) in
        let scan xs =
          fst
            (Rune.scan tensor tensor Nx.Ptree.unit ~f:step
               ~init:(Nx.zeros f64 [| 3 |]) xs)
        in
        let run xs = Nx.sum (scan xs) in
        let xs =
          Nx.reshape [| 2; 3 |] (Nx.concatenate ~axis:0 [ v3 (); along () ])
        in
        near ~msg:"value"
          (Nx.add (Nx.sin (v3 ())) (Nx.sin (along ())))
          (scan xs);
        exact ~msg:"grad" (Nx.full f64 [| 2; 3 |] 100.) (Rune.grad' run xs));
    test "a custom_vjp in a scan's step and a remat" (fun () ->
        let xs =
          Nx.reshape [| 2; 3 |] (Nx.concatenate ~axis:0 [ v3 (); along () ])
        in
        let hundreds = Nx.full f64 [| 2; 3 |] 100. in
        let total ~f xs =
          Nx.sum
            (fst
               (Rune.scan tensor tensor Nx.Ptree.unit ~f
                  ~init:(Nx.zeros f64 [| 3 |]) xs))
        in
        let rows c x = (Nx.add c (fake_grad_sin x), ()) in
        exact ~msg:"grad of the rows" hundreds (Rune.grad' (total ~f:rows) xs);
        exact ~msg:"jit of grad" hundreds
          (Rune.jit' (Rune.grad' (total ~f:rows)) xs);
        (* The carry's tangent starts as a zero the scan instantiates, which the
           rule's tangents read alone on the first step. *)
        let carried c x = (Nx.add (fake_grad_sin c) x, ()) in
        let c1 = Nx.add (Nx.sin (Nx.zeros f64 [| 3 |])) (v3 ()) in
        let c2 = Nx.add (Nx.sin c1) (along ()) in
        near ~msg:"value" (Nx.sum c2) (total ~f:carried xs);
        exact ~msg:"grad through the carry"
          (Nx.reshape [| 2; 3 |]
             (Nx.concatenate ~axis:0
                [ Nx.full f64 [| 3 |] 100.; Nx.full f64 [| 3 |] 1. ]))
          (Rune.grad' (total ~f:carried) xs);
        let r = Rune.remat Nx.Ptree.(tensor @-> returns tensor) fake_grad_sin in
        exact ~msg:"grad of a remat" (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (r x)) (v3 ())));
  ]

(* A rule whose tangent map calls custom rules on its tangents: [v c] with its
   own rule, and a custom_vjp, so that reverse mode records the inner rules'
   operations on the tape. f x = sin x, f' = cos x, f'' = -sin x. *)
let product =
  Rune.custom_jvp pair tensor (fun (v, c) ->
      (Nx.mul v c, fun (dv, dc) -> Nx.add (Nx.mul dv c) (Nx.mul v dc)))

let product_vjp =
  Rune.custom_vjp pair tensor (fun (v, c) ->
      (Nx.mul v c, fun g -> (Nx.mul g c, Nx.mul g v)))

let nested inner =
  Rune.custom_jvp tensor tensor (fun x ->
      let c = Nx.cos x in
      (Nx.sin x, fun dx -> inner (dx, c)))

(* An inner rule whose tangent map is wrong along [c]: forward mode around a
   tangent map applies it, reverse mode records the inner function. *)
let wrong =
  Rune.custom_jvp pair tensor (fun (v, c) ->
      ( Nx.mul v c,
        fun (dv, dc) -> Nx.add (Nx.mul dv c) (Nx.mul_s (Nx.mul v dc) 100.) ))

let split_test =
  test
    "a tangent map's rule: applied around forward mode, its code around \
     reverse mode" (fun () ->
      let x = 0.4 in
      let g x = Nx.sum (nested wrong x) in
      let applied = scalar (-100. *. Float.sin x)
      and code = scalar (-.Float.sin x) in
      near ~msg:"jvp ∘ jvp" applied
        (snd
           (Rune.jvp'
              (fun x -> snd (Rune.jvp' g x (scalar 1.)))
              (scalar x) (scalar 1.)));
      near ~msg:"grad ∘ jvp" applied
        (Rune.grad' (fun x -> snd (Rune.jvp' g x (scalar 1.))) (scalar x));
      near ~msg:"jvp ∘ grad" code (hvp g (scalar x));
      near ~msg:"grad ∘ grad" code
        (Rune.grad' (fun x -> Nx.sum (Rune.grad' g x)) (scalar x)))

let nested_tests =
  let x = 0.4 in
  let d1 = scalar (Float.cos x) and d2 = scalar (-.Float.sin x) in
  let cases =
    [ ("custom_jvp", nested product); ("custom_vjp", nested product_vjp) ]
  in
  List.map
    (fun (name, f) ->
      test ("a tangent map that calls a " ^ name) (fun () ->
          let g x = Nx.sum (f x) in
          near ~msg:"grad" d1 (Rune.grad' g (scalar x));
          near ~msg:"jit of grad" d1 (Rune.jit' (Rune.grad' g) (scalar x));
          let step c x = (Nx.add c (Rune.grad' g x), ()) in
          near ~msg:"grad in a scan step" d1
            (fst
               (Rune.scan tensor tensor Nx.Ptree.unit ~f:step ~init:(scalar 0.)
                  (vec [| x |])));
          near ~msg:"grad ∘ grad" d2
            (Rune.grad' (fun x -> Nx.sum (Rune.grad' g x)) (scalar x));
          if name = "custom_jvp" then begin
            near ~msg:"jvp ∘ grad" d2 (hvp g (scalar x));
            near ~msg:"grad ∘ jvp" d2
              (Rune.grad' (fun x -> snd (Rune.jvp' g x (scalar 1.))) (scalar x))
          end))
    cases
  @ [ split_test ]

(* Under a map *)

let xs () = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let dxs () = Nx.create f64 [| 2; 3 |] [| 1.; 0.5; -2.; 0.3; 1.5; -0.7 |]
let row i x = Nx.get [ i ] x
let stack n f = Nx.stack ~axis:0 (List.init n f)

let map_tests =
  [
    test "per-lane gradients through custom rules" (fun () ->
        near ~msg:"custom_vjp"
          (Nx.cos (xs ()))
          (Rune.vmap' (Rune.grad' (fun x -> Nx.sum (my_sin x))) (xs ()));
        exact ~msg:"custom_jvp"
          (Nx.full f64 [| 2; 3 |] 100.)
          (Rune.vmap' (Rune.grad' (fun x -> Nx.sum (fake_jvp_sin x))) (xs ())));
    test "a map of a custom rule is the map of its value" (fun () ->
        exact ~msg:"custom_vjp" (Nx.sin (xs ())) (Rune.vmap' my_sin (xs ()));
        exact ~msg:"custom_jvp"
          (Nx.sin (xs ()))
          (Rune.vmap' fake_jvp_sin (xs ())));
    test "grad of a map applies the rule to the lanes" (fun () ->
        let g f = Rune.grad' (fun x -> Nx.sum (Rune.vmap' f x)) (xs ()) in
        near ~msg:"a true pullback" (Nx.cos (xs ())) (g my_sin);
        exact ~msg:"a pullback of 100"
          (Nx.full f64 [| 2; 3 |] 100.)
          (g fake_grad_sin);
        near ~msg:"a true tangent map" (Nx.cos (xs ())) (g my_sin_fwd);
        exact ~msg:"a tangent map of 100"
          (Nx.full f64 [| 2; 3 |] 100.)
          (g fake_jvp_sin));
    test "jvp of a map applies a custom_jvp's tangent map to the lanes"
      (fun () ->
        near ~msg:"true"
          (Nx.mul (dxs ()) (Nx.cos (xs ())))
          (snd (Rune.jvp' (Rune.vmap' my_sin_fwd) (xs ()) (dxs ())));
        near ~msg:"fake"
          (Nx.mul_s (dxs ()) 100.)
          (snd (Rune.jvp' (Rune.vmap' fake_jvp_sin) (xs ()) (dxs ()))));
    test "jvp of a map of a custom_vjp is refused" (fun () ->
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' (Rune.vmap' fake_grad_sin) (xs ()) (dxs ())));
    test "a tangent map inside a map sees each lane's tangent" (fun () ->
        let seen = Total.make () in
        let observe =
          Rune.custom_jvp tensor Nx.Ptree.unit (fun _ ->
              ((), fun dy -> Total.add seen dy))
        in
        let _, total =
          Total.collect seen ~init:(Nx.zeros f64 [| 3 |]) (fun () ->
              Rune.jvp'
                (Rune.vmap' (fun r ->
                     observe (Nx.sin r);
                     r))
                (xs ()) (dxs ()))
        in
        near (Nx.sum ~axes:[ 0 ] (Nx.mul (dxs ()) (Nx.cos (xs ())))) total);
    test "a map passes on a rule that reads its lanes" (fun () ->
        let ys =
          Nx.create f64 [| 4; 3 |]
            (Array.init 12 (fun i -> Float.sin (Float.of_int i)))
        in
        let cs = xs () and x0 = v3 () and w = scalar 1.5 in
        let scaled c =
          Rune.custom_jvp tensor tensor (fun x ->
              (Nx.mul x c, fun dx -> Nx.mul dx c))
        in
        let check ~msg f loop =
          let v, g = Rune.value_and_grad' (fun w -> Nx.sum (f w)) w in
          near ~msg:(msg ^ ", value") (Nx.mul_s (Nx.sum loop) 1.5) v;
          near ~msg:(msg ^ ", gradient") (Nx.sum loop) g;
          let y, dy = Rune.jvp' f w (scalar 1.) in
          near ~msg:(msg ^ ", primal") (Nx.mul_s loop 1.5) y;
          near ~msg:(msg ^ ", tangent") loop dy
        in
        check ~msg:"nested maps"
          (fun w ->
            Rune.vmap'
              (fun c -> Rune.vmap' (fun x -> Nx.mul w (scaled c x)) ys)
              cs)
          (stack 2 (fun j -> stack 4 (fun i -> Nx.mul (row i ys) (row j cs))));
        check ~msg:"unbatched arguments"
          (fun w -> Rune.vmap' (fun c -> Nx.mul w (scaled c x0)) cs)
          (stack 2 (fun j -> Nx.mul x0 (row j cs))));
    test "a custom_vjp's cotangent of an argument the lanes share is their sum"
      (fun () ->
        (* Pullbacks of 1 for [w], whatever the cotangent, so that the gradient
           counts the pullbacks applied. *)
        let shared =
          Rune.custom_vjp pair tensor (fun (w, c) ->
              (Nx.mul w c, fun _ -> (Nx.ones_like w, Nx.zeros_like c)))
        in
        let each w = Nx.sum (Rune.vmap' (fun c -> shared (w, c)) (xs ())) in
        exact ~msg:"one pullback per lane" (vec [| 2. |])
          (Rune.grad' each (vec [| 1.5 |]));
        (* A call that reads no lane computes once, outside the lanes, and its
           pullback receives their summed cotangent. *)
        let once =
          Rune.custom_vjp tensor tensor (fun w -> (w, fun _ -> Nx.ones_like w))
        in
        let outside w =
          Nx.sum (Rune.vmap' (fun c -> Nx.mul (once w) c) (xs ()))
        in
        exact ~msg:"one pullback" (vec [| 1. |])
          (Rune.grad' outside (vec [| 1.5 |]));
        exact ~msg:"one pullback of a scalar" (scalar 1.)
          (Rune.grad' outside (scalar 1.5)));
    test "a pullback's additions inside a map count each lane" (fun () ->
        let t = Total.make () in
        let f =
          Rune.custom_vjp tensor tensor (fun x ->
              ( Nx.sin x,
                fun g ->
                  Total.add t (Nx.sum g);
                  Nx.mul g (Nx.cos x) ))
        in
        let _, total =
          Total.collect t ~init:(scalar 0.) (fun () ->
              Rune.grad' (fun x -> Nx.sum (Rune.vmap' f x)) (xs ()))
        in
        exact (scalar 6.) total);
    test "a differentiation inside the map reads the residuals in order"
      (fun () ->
        let t = Total.make () in
        (* The tangent map reads [cos x] and adds [Σ sin x] first. *)
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              let a = Nx.cos x and b = Nx.sin x in
              ( Nx.sin x,
                fun dx ->
                  Total.add t (Nx.sum b);
                  Nx.mul dx a ))
        in
        let sines = Nx.sum (Nx.sin (xs ())) in
        let jv, total =
          Total.collect t ~init:(scalar 0.) (fun () ->
              Rune.vmap' (fun x -> snd (Rune.jvp' f x (Nx.ones_like x))) (xs ()))
        in
        near ~msg:"jvp, tangent" (Nx.cos (xs ())) jv;
        near ~msg:"jvp, total" sines total;
        let g, total =
          Total.collect t ~init:(scalar 0.) (fun () ->
              Rune.vmap' (Rune.grad' (fun x -> Nx.sum (f x))) (xs ()))
        in
        near ~msg:"grad, gradient" (Nx.cos (xs ())) g;
        near ~msg:"grad, total" sines total);
    test "a map of no lane" (fun () ->
        let none = Nx.zeros f64 [| 0; 3 |] in
        exact ~msg:"value" none (Rune.vmap' fake_jvp_sin none);
        exact ~msg:"grad" none
          (Rune.grad' (fun x -> Nx.sum (Rune.vmap' fake_jvp_sin x)) none);
        exact ~msg:"jvp" none
          (snd (Rune.jvp' (Rune.vmap' fake_jvp_sin) none none)));
    test "jit around the map" (fun () ->
        exact
          (Nx.full f64 [| 2; 3 |] 100.)
          (Rune.jit'
             (Rune.grad' (fun x -> Nx.sum (Rune.vmap' fake_jvp_sin x)))
             (xs ())));
    test "a complex rule under a map" (fun () ->
        let z =
          Nx.create Nx.complex128 [| 2; 2 |]
            (Array.init 4 (fun k ->
                 {
                   Complex.re = 0.3 *. Float.of_int k;
                   im = 1. -. Float.of_int k;
                 }))
        in
        let two = { Complex.re = 2.; im = 0. } in
        let square =
          Rune.custom_jvp tensor tensor (fun z ->
              (Nx.mul z z, fun dz -> Nx.mul (Nx.mul_s z two) dz))
        in
        let pairs a =
          Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array a)
        in
        let t = Nx.ones_like z in
        equal
          (array (Windtrap.pair (float 1e-12) (float 1e-12)))
          (pairs (snd (Rune.jvp' (fun z -> Nx.mul z z) z t)))
          (pairs (snd (Rune.jvp' (Rune.vmap' square) z t))));
    test
      "a tangent map that reads a map's lanes runs where a differentiation \
       applies it" (fun () ->
        let axis = Rune.axis () in
        let mark =
          Rune.custom_jvp tensor Nx.Ptree.unit (fun _ ->
              ((), fun dy -> ignore (Rune.lanes axis dy)))
        in
        let loss x =
          mark x;
          Nx.sum (Nx.mul x x)
        in
        let x = v3 () in
        exact ~msg:"eagerly" (Nx.sum (Nx.mul x x)) (loss x);
        exact ~msg:"under grad" (Nx.mul_s x 2.) (Rune.grad' loss x);
        invalid ~sub:"Rune.lanes" (fun () -> Rune.jvp' loss x (along ())));
    test "grad of a map of a custom_transpose applies its transpose" (fun () ->
        let declared =
          Rune.custom_transpose tensor tensor
            (fun v -> Nx.mul_s (fst (Nx.sort v)) 2.)
            ~transpose:(fun w -> Nx.mul_s w 3.)
        in
        let f =
          Rune.custom_jvp tensor tensor (fun x -> (Nx.mul_s x 2., declared))
        in
        exact
          (Nx.full f64 [| 4; 1 |] 3.)
          (Rune.grad'
             (fun x -> Nx.sum (Rune.vmap' f x))
             (Nx.create f64 [| 4; 1 |] [| 0.1; 0.2; 0.3; 0.4 |])));
  ]

(* custom_transpose *)

let mat () =
  Nx.create f64 [| 3; 3 |] [| 4.; 1.; 0.5; -1.; 3.; 0.2; 0.3; -0.4; 2.5 |]

let rhs () = vec [| 1.5; -0.5; 2. |]
let transposes = ref 0

(* [v ↦ a⁻¹ v], whose transpose is the transposed solve, as a Riccati
   recursion's adjoint is a backward one. [transposes] counts its traces. *)
let inverse a =
  Rune.custom_transpose tensor tensor
    (fun v -> Nx.solve a v)
    ~transpose:(fun w ->
      incr transposes;
      Nx.solve (Nx.matrix_transpose a) w)

(* x = a⁻¹ b, whose tangent is a⁻¹ (db - da x). *)
let solve =
  Rune.custom_jvp pair tensor (fun (a, b) ->
      let x = Nx.solve a b in
      (x, fun (da, db) -> inverse a (Nx.sub db (Nx.matmul da x))))

(* v ↦ 2 v through a sort, whose operations have no transpose under reverse
   mode, declared with the transpose w ↦ 3 w, so that a result of 3 shows the
   declared transpose. *)
let twice v = Nx.mul_s (fst (Nx.sort v)) 2.

let declared =
  Rune.custom_transpose tensor tensor twice ~transpose:(fun w -> Nx.mul_s w 3.)

let through_declared =
  Rune.custom_jvp tensor tensor (fun x -> (Nx.mul_s x 2., declared))

let half_sum y = Nx.mul_s (Nx.sum (Nx.square y)) 0.5

let transpose_tests =
  let a = mat () and b = rhs () in
  let plain (a, b) = Nx.solve a b in
  let loss f (a, b) = half_sum (f (a, b)) in
  [
    test "outside every differentiation the call is its function" (fun () ->
        exact (Nx.solve a b) (inverse a b);
        exact (Nx.solve a b) (solve (a, b)));
    test "forward mode applies the function" (fun () ->
        let da = Nx.mul_s (Nx.ones_like a) 0.1 and db = vec [| 1.; 0.; -1. |] in
        let _, t = Rune.jvp pair tensor solve (a, b) (da, db) in
        let _, t' = Rune.jvp pair tensor plain (a, b) (da, db) in
        near ~tol:1e-12 t' t;
        exact ~msg:"the function's own tangent" (Nx.solve a db)
          (snd (Rune.jvp' (inverse a) b db)));
    test "reverse mode applies the declared transpose" (fun () ->
        let ga, gb = Rune.grad pair (loss solve) (a, b) in
        let ga', gb' = Rune.grad pair (loss plain) (a, b) in
        near ~tol:1e-12 ~msg:"a" ga' ga;
        near ~tol:1e-12 ~msg:"b" gb' gb;
        let one = vec [| 0.7 |] in
        exact ~msg:"the declared transpose's value" (vec [| 3. |])
          (Rune.grad' (fun x -> Nx.sum (through_declared x)) one);
        invalid
          ~sub:
            "Rune.grad': a custom_jvp tangent map applies a sort to a tangent"
          (fun () ->
            Rune.grad'
              (fun x ->
                Nx.sum
                  (Rune.custom_jvp tensor tensor
                     (fun x -> (Nx.mul_s x 2., twice))
                     x))
              one));
    test "the transpose does not run where the function runs on values"
      (fun () ->
        transposes := 0;
        ignore (Rune.grad pair (loss solve) (a, b));
        equal ~msg:"grad" int 1 !transposes;
        ignore (inverse a b);
        equal ~msg:"no differentiation" int 1 !transposes;
        ignore (Rune.jvp pair tensor solve (a, b) (a, b));
        equal ~msg:"forward mode" int 1 !transposes);
    test "the function runs on values eagerly and on forward mode's tangents"
      (fun () ->
        (* v ↦ 2 v, which reads whether v is zero. *)
        let reading v =
          if Nx.item [] (Nx.max (Nx.abs v)) = 0. then Nx.zeros_like v
          else Nx.mul_s v 2.
        in
        let linear =
          Rune.custom_transpose tensor tensor reading ~transpose:(fun w ->
              Nx.mul_s w 2.)
        in
        let rule =
          Rune.custom_jvp tensor tensor (fun x -> (Nx.mul_s x 2., linear))
        in
        let x = vec [| 0.5; -1.; 2. |] in
        exact ~msg:"eagerly" (Nx.mul_s x 2.) (linear x);
        exact ~msg:"under jvp" (Nx.mul_s x 2.) (snd (Rune.jvp' rule x x));
        invalid ~sub:"Nx.item: the value is traced by Rune.custom_transpose"
          ~msg:"traced under grad" (fun () ->
            Rune.grad' (fun x -> Nx.sum (rule x)) x);
        raises ~msg:"a draw without a key"
          (Invalid_argument
             "Nx.Rng.uniform: no key inside Rune.custom_transpose, which \
              refuses keyless draws") (fun () ->
            Rune.custom_transpose tensor tensor
              (fun v -> Nx.add v (Nx.Rng.uniform f64 [| 3 |]))
              ~transpose:Fun.id x));
    test "the function runs on values whose captures are tracked" (fun () ->
        (* [b] is untracked, so [inverse a] runs on it, and grad and vmap follow
           [Nx.solve] through the captured [a]. *)
        let solved a = half_sum (inverse a b) in
        near ~tol:1e-12 ~msg:"grad over the capture"
          (Rune.grad' (fun a -> half_sum (Nx.solve a b)) a)
          (Rune.grad' solved a);
        let stacked = Nx.stack [ a; Nx.mul_s a 2. ] in
        near ~tol:1e-12 ~msg:"vmap over the capture"
          (Nx.stack [ Nx.solve a b; Nx.solve (Nx.mul_s a 2.) b ])
          (Rune.vmap' (fun a -> inverse a b) stacked);
        invalid ~sub:"Nx.item: the value is traced by Rune.vmap"
          ~msg:"a read of the mapped capture" (fun () ->
            Rune.vmap'
              (fun a ->
                Rune.custom_transpose tensor tensor
                  (fun v -> Nx.mul_s v (Nx.item [ 0; 0 ] a))
                  ~transpose:Fun.id b)
              stacked));
    test "a total collected inside the function keeps its additions" (fun () ->
        (* v ↦ v + sum v, symmetric, its sum collected in a total. *)
        let t = Total.make () in
        let summed v =
          let _, s =
            Total.collect t ~init:(scalar 0.) (fun () -> Total.add t (Nx.sum v))
          in
          Nx.add v (Nx.broadcast_to [| 3 |] s)
        in
        let direct v = Nx.add v (Nx.broadcast_to [| 3 |] (Nx.sum v)) in
        exact (direct b)
          (Rune.custom_transpose tensor tensor summed ~transpose:direct b));
    test "the transpose's transpose is the function" (fun () ->
        let u = vec [| 0.4 |] and v0 = vec [| 1. |] in
        let pullback w = snd (Rune.vjp' declared v0) w in
        exact ~msg:"the transpose" (Nx.mul_s u 3.) (pullback u);
        exact ~msg:"its transpose" (Nx.mul_s u 2.)
          (Rune.grad' (fun w -> Nx.sum (Nx.mul u (pullback w))) (vec [| 5. |])));
    test "second order in every pair of modes" (fun () ->
        let dir_a = Nx.mul_s (Nx.ones_like a) 0.1
        and dir_b = vec [| 1.; 0.; -1. |] in
        let g f p = Rune.grad pair (loss f) p in
        let along f = Rune.jvp pair pair (g f) (a, b) (dir_a, dir_b) |> snd in
        let ha, hb = along solve and ha', hb' = along plain in
        near ~tol:1e-10 ~msg:"forward over reverse, a" ha' ha;
        near ~tol:1e-10 ~msg:"forward over reverse, b" hb' hb;
        let rev f =
          Rune.grad pair
            (fun p ->
              let ga, gb = g f p in
              Nx.add (Nx.sum (Nx.mul ga dir_a)) (Nx.sum (Nx.mul gb dir_b)))
            (a, b)
        in
        let ra, rb = rev solve and ra', rb' = rev plain in
        near ~tol:1e-10 ~msg:"reverse over reverse, a" ra' ra;
        near ~tol:1e-10 ~msg:"reverse over reverse, b" rb' rb;
        let fwd f p = snd (Rune.jvp pair tensor f p (dir_a, dir_b)) in
        let fa, fb = Rune.grad pair (fun p -> half_sum (fwd solve p)) (a, b) in
        let fa', fb' =
          Rune.grad pair (fun p -> half_sum (fwd plain p)) (a, b)
        in
        near ~tol:1e-10 ~msg:"reverse over forward, a" fa' fa;
        near ~tol:1e-10 ~msg:"reverse over forward, b" fb' fb);
    test "under jit" (fun () ->
        exact ~msg:"value" (Nx.solve a b) (Rune.jit' (inverse a) b);
        exact ~msg:"jit of grad" (vec [| 3. |])
          (Rune.jit'
             (Rune.grad' (fun x -> Nx.sum (through_declared x)))
             (vec [| 0.7 |]));
        exact ~msg:"grad of jit" (vec [| 3. |])
          (Rune.grad'
             (fun x -> Nx.sum (Rune.jit' through_declared x))
             (vec [| 0.7 |])));
    test "a linear call that adds to a total is refused" (fun () ->
        let t = Total.make () in
        let message =
          "Rune.custom_transpose: the function or its transpose adds to a \
           total; a linear call makes no addition"
        in
        let adding v =
          Total.add t (Nx.sum v);
          v
        in
        invalid ~sub:message ~msg:"the function, eagerly" (fun () ->
            Rune.custom_transpose tensor tensor adding ~transpose:Fun.id b);
        invalid ~sub:message ~msg:"the function, under jit" (fun () ->
            Rune.jit'
              (Rune.custom_transpose tensor tensor adding ~transpose:Fun.id)
              b);
        invalid ~sub:message ~msg:"the transpose, under grad" (fun () ->
            Rune.grad'
              (fun v ->
                Nx.sum
                  (Rune.custom_transpose tensor tensor Fun.id ~transpose:adding
                     v))
              b);
        exact ~msg:"eagerly the transpose does not run" b
          (Rune.custom_transpose tensor tensor Fun.id ~transpose:adding b));
    test "a transpose of another shape than the arguments is refused" (fun () ->
        let short =
          Rune.custom_transpose tensor tensor Fun.id ~transpose:(fun w ->
              Nx.slice [ Nx.R (0, 2) ] w)
        in
        invalid
          ~sub:
            "Rune.custom_transpose: the root: shape [2] in the transpose's \
             result, [3] in the arguments" (fun () ->
            Rune.grad' (fun v -> Nx.sum (short v)) b);
        exact ~msg:"eagerly the transpose does not run" b (short b));
    test "check_transpose holds the adjoint law" (fun () ->
        equal ~msg:"a true transpose" (result unit string) (Ok ())
          (Rune.check_transpose tensor tensor
             (fun v -> Nx.solve a v)
             ~transpose:(fun w -> Nx.solve (Nx.matrix_transpose a) w)
             b);
        match
          Rune.check_transpose tensor tensor
            (fun v -> Nx.solve a v)
            ~transpose:(fun w -> Nx.solve a w)
            b
        with
        | Ok () -> fail "a wrong transpose passed"
        | Error msg -> (
            let prefix = "Rune.check_transpose: pair 1:" in
            equal ~msg string prefix (String.sub msg 0 (String.length prefix));
            (* On complex arrays the pairing is unconjugated: multiplying by [c]
               is its own transpose, and by [conj c] is not. *)
            let z re im =
              Nx.create Nx.complex128 [| 2 |]
                (Array.map2 (fun re im -> { Complex.re; im }) re im)
            in
            let c = z [| 0.5; -1. |] [| 2.; 0.3 |]
            and v0 = z [| 1.; 0. |] [| 0.; 1. |] in
            let check t =
              Rune.check_transpose tensor tensor
                (fun v -> Nx.mul c v)
                ~transpose:t v0
            in
            equal ~msg:"complex, unconjugated" (result unit string) (Ok ())
              (check (fun w -> Nx.mul c w));
            (match
               Rune.check_transpose tensor tensor
                 (fun v -> Nx.mul_s v 3000.)
                 ~transpose:(fun w -> Nx.mul_s w 0.)
                 (Nx.ones Nx.float16 [| 3 |])
             with
            | Ok () -> fail "a wrong float16 transpose passed"
            | Error msg ->
                equal ~msg string prefix
                  (String.sub msg 0 (String.length prefix)));
            match check (fun w -> Nx.mul (Nx.conjugate c) w) with
            | Ok () -> fail "a conjugated transpose passed"
            | Error msg ->
                equal ~msg string prefix
                  (String.sub msg 0 (String.length prefix))));
  ]

(* Laws *)

(* A rule whose map is f's own derivative, f x = sin x · x: forward mode applies
   it, reverse mode transposes it, so the pullback is the tangent map's
   adjoint. *)
let product =
  Rune.custom_jvp tensor tensor (fun x ->
      ( Nx.mul (Nx.sin x) x,
        fun dx -> Nx.mul (Nx.add (Nx.mul (Nx.cos x) x) (Nx.sin x)) dx ))

let gen_vecs =
  let open Gen in
  let* n = int_range 0 6 in
  let v = array ~size:(constant n) (float_range (-3.) 3.) in
  triple v v v

let adjoint_law (xs, vs, ws) =
  cover "no element" (Array.length xs = 0);
  cover "several elements" (Array.length xs > 1);
  let x = vec xs and v = vec vs and w = vec ws in
  let dy = snd (Rune.jvp' product x v) in
  near ~tol:1e-12 ~msg:"against the function's"
    (snd (Rune.jvp' (fun x -> Nx.mul (Nx.sin x) x) x v))
    dy;
  let _, pb = Rune.vjp' product x in
  let g = pb w in
  let dot a b = Nx.item [] (Nx.sum (Nx.mul a b)) in
  equal ~msg:"<w, J v> = <Jᵀ w, v>" (float 1e-12) (dot w dy) (dot g v)

(* s sin x with a tangent map of 100 s along x: a differentiation of x applies
   the rule, one of s alone differentiates the code. *)
let capture_law (xs, ss, vs) =
  cover "no element" (Array.length xs = 0);
  cover "several elements" (Array.length xs > 1);
  let x = vec xs and s = vec ss and v = vec vs in
  let rule s =
    Rune.custom_jvp tensor tensor (fun x ->
        let c = Nx.mul_s s 100. in
        (Nx.mul s (Nx.sin x), fun dx -> Nx.mul c dx))
  in
  let reverse x = Rune.value_and_grad' (fun s -> Nx.sum (rule s x)) s in
  near ~tol:1e-9 ~msg:"the rule along x"
    (Nx.sum (Nx.mul (Nx.mul_s s 100.) v))
    (snd (Rune.jvp' (fun x -> fst (reverse x)) x v));
  near ~tol:1e-9 ~msg:"the code along s" (Nx.sin x) (snd (reverse x));
  near ~tol:1e-9 ~msg:"the code's derivative along x"
    (Nx.mul (Nx.cos x) v)
    (snd (Rune.jvp' (fun x -> snd (reverse x)) x v))

(* [f x] computed in a scan's step over one row, where [f]'s rule is traced. *)
let in_step f x =
  let n = Nx.numel x in
  let _, ys =
    Rune.scan Nx.Ptree.unit tensor tensor
      ~f:(fun () r -> ((), f r))
      ~init:()
      (Nx.reshape [| 1; n |] x)
  in
  Nx.reshape [| n |] ys

(* f x = sin x · x with its derivative, as a tangent map and as a pullback, at
   any float or complex dtype. *)
let rules () =
  let d x = Nx.add (Nx.mul (Nx.cos x) x) (Nx.sin x) in
  ( Rune.custom_jvp tensor tensor (fun x ->
        (Nx.mul (Nx.sin x) x, fun dx -> Nx.mul (d x) dx)),
    Rune.custom_vjp tensor tensor (fun x ->
        (Nx.mul (Nx.sin x) x, fun g -> Nx.mul (d x) g)) )

let specials = [ Float.nan; Float.infinity; Float.neg_infinity; -0.; 0. ]

let gen_breaking =
  let open Gen in
  let* n = int_range 0 6 in
  let element =
    frequency [ (3, float_range (-3.) 3.); (1, of_list specials) ]
  in
  let v = array ~size:(constant n) element in
  triple (pair v v) (pair v v) (pair v v)

let special x = (not (Float.is_finite x)) || x = 0.

(* A rule computed on its arguments' values gives the bits of the rule traced,
   [bits] reading a value's floats. *)
let agree bits (x, v, w) =
  let same msg a b = equal ~msg (array float_exact) (bits a) (bits b) in
  let jvp_rule, vjp_rule = rules () in
  let pulled f = snd (Rune.vjp' f x) w in
  List.iter
    (fun (msg, f) ->
      same (msg ^ ", value") (in_step f x) (f x);
      same (msg ^ ", pullback") (pulled (in_step f)) (pulled f);
      same (msg ^ ", jacrev") (Rune.jacrev' (in_step f) x) (Rune.jacrev' f x))
    [ ("custom_jvp", jvp_rule); ("custom_vjp", vjp_rule) ];
  same "custom_jvp, tangent"
    (snd (Rune.jvp' (in_step jvp_rule) x v))
    (snd (Rune.jvp' jvp_rule x v));
  same "custom_jvp, jacfwd"
    (Rune.jacfwd' (in_step jvp_rule) x)
    (Rune.jacfwd' jvp_rule x)

let real_law ((xs, _), (vs, _), (ws, _)) =
  cover "no element" (Array.length xs = 0);
  cover "several elements" (Array.length xs > 1);
  cover "a special value" (Array.exists special xs);
  agree values (vec xs, vec vs, vec ws)

let complex_law ((xr, xi), (vr, vi), (wr, wi)) =
  cover "no element" (Array.length xr = 0);
  cover "several elements" (Array.length xr > 1);
  cover "a special part" (Array.exists special xr || Array.exists special xi);
  let z re im = Nx.complex Nx.complex128 ~re:(vec re) ~im:(vec im) in
  let bits a = Array.append (values (Nx.real f64 a)) (values (Nx.imag f64 a)) in
  agree bits (z xr xi, z vr vi, z wr wi)

(* f x = sin x · x, whose derivative is scaled by s = Σ |f x|, plus Σ |u| of the
   derivative's input u with [~input]. With [read] the derivative reads s with
   {!Nx.item}, as an adaptive derivative reads a tolerance, which works where it
   runs on values; without, it multiplies by s as an array, which works where it
   is traced. *)
let f_sin x = Nx.mul (Nx.sin x) x
let d_sin x = Nx.add (Nx.mul (Nx.cos x) x) (Nx.sin x)

let by ~read ~input y u t =
  let s = Nx.sum (Nx.abs y) in
  let s = if input then Nx.add s (Nx.sum (Nx.abs u)) else s in
  if read then Nx.mul_s t (Nx.item [] s) else Nx.mul t s

let scaled_jvp ~read ~input =
  Rune.custom_jvp tensor tensor (fun x ->
      let y = f_sin x in
      (y, fun dx -> by ~read ~input y dx (Nx.mul (d_sin x) dx)))

let scaled_vjp ~read =
  Rune.custom_vjp tensor tensor (fun x ->
      let y = f_sin x in
      (y, fun g -> by ~read ~input:true y g (Nx.mul (d_sin x) g)))

(* The tangent map v ↦ s · f' x · v as a custom_transpose, its own transpose.
   With [read] it also reads v, without using the read, to stay linear in v. *)
let scaled_linear ~read =
  Rune.custom_jvp tensor tensor (fun x ->
      let y = f_sin x in
      let scale v =
        if read then ignore (Nx.item [] (Nx.sum v));
        by ~read ~input:false y v (Nx.mul (d_sin x) v)
      in
      (y, Rune.custom_transpose tensor tensor scale ~transpose:scale))

(* A derivative that reads values, run on values, gives the bits of its twin
   that does not read, traced. Reverse mode transposes a tangent map, so the
   tangent map reads its tangent only under jvp. On finite arguments, grad is
   the closed form s · f' x, with s + n for the pullback, which reads n ones. *)
let read_law ((xs, _), (vs, _), (ws, _)) =
  let n = Array.length xs in
  let finite = n > 0 && Array.for_all Float.is_finite xs in
  cover "no element" (n = 0);
  cover "one element" (n = 1);
  cover "several elements" (n > 1);
  cover "a special value" (Array.exists special xs);
  cover "every value finite" finite;
  let x = vec xs and v = vec vs and w = vec ws in
  let sum f x = Nx.sum (f x) in
  List.iter
    (fun (msg, read, held) ->
      exact ~msg:(msg ^ ", jvp")
        (snd (Rune.jvp' (in_step held) x v))
        (snd (Rune.jvp' read x v)))
    [
      ( "tangent map reading its tangent",
        scaled_jvp ~read:true ~input:true,
        scaled_jvp ~read:false ~input:true );
      ("custom_transpose", scaled_linear ~read:true, scaled_linear ~read:false);
    ];
  List.iter
    (fun (msg, read, held, extra) ->
      exact ~msg:(msg ^ ", grad")
        (Rune.grad' (sum (in_step held)) x)
        (Rune.grad' (sum read) x);
      exact ~msg:(msg ^ ", vjp")
        (snd (Rune.vjp' (in_step held) x) w)
        (snd (Rune.vjp' read x) w);
      if finite then
        let s = Nx.item [] (Nx.sum (Nx.abs (f_sin x))) +. extra in
        near ~tol:1e-10 ~msg:(msg ^ ", closed form")
          (Nx.mul_s (d_sin x) s)
          (Rune.grad' (sum read) x))
    [
      ( "custom_jvp",
        scaled_jvp ~read:true ~input:false,
        scaled_jvp ~read:false ~input:false,
        0. );
      ( "custom_vjp",
        scaled_vjp ~read:true,
        scaled_vjp ~read:false,
        float_of_int n );
    ]

let law_tests =
  [
    prop "a rule on values gives the traced rule's bits" gen_breaking real_law;
    prop "a complex rule on values gives the traced rule's bits" gen_breaking
      complex_law;
    prop "a derivative that reads values gives its traced twin's bits"
      gen_breaking read_law;
    prop "the pullback is the tangent map's adjoint" gen_vecs adjoint_law;
    prop "a differentiation of a capture leaves the rule to the argument's"
      gen_vecs capture_law;
  ]

let () =
  exit
  @@ run "rune2 custom rules"
       [
         group "custom_vjp" vjp_tests;
         group "custom_jvp" jvp_tests;
         group "integer result leaves" int_tests;
         group "second order" order_tests;
         group "reads" read_tests;
         group "errors" error_tests;
         group "inside constructs" compiled_tests;
         group "rules in a tangent map" nested_tests;
         group "under a map" map_tests;
         group "custom_transpose" transpose_tests;
         group "laws" law_tests;
       ]
