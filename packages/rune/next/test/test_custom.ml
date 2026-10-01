(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Custom differentiation rules: a rule replaces the derivative of its function,
   a custom_jvp serves both modes, a custom_vjp has no forward derivative, a
   call with no tensor result runs under the mode its rule lacks, and maps pass
   every rule on. Deliberately wrong rules show that the rule is used. *)

open Windtrap
module Rune = Rune_next.Rune

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-12 ~abs:1e-12 ()
let tensor = Nx.Ptree.tensor
let pair = Nx.Ptree.(pair tensor tensor)
let v3 () = vec [| 0.7; -1.3; 2.1 |]
let along () = vec [| 0.5; 1.; -2. |]
let xs () = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let dxs () = Nx.create f64 [| 2; 3 |] [| 1.; 0.5; -2.; 0.3; 1.5; -0.7 |]

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
        equal (exact ()) (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (fake_grad_sin x)) (v3 ())));
    test "a true rule matches the function's derivative" (fun () ->
        equal (close ())
          (Rune.grad' (fun x -> Nx.sum (Nx.sin x)) (v3 ()))
          (Rune.grad' (fun x -> Nx.sum (my_sin x)) (v3 ())));
    test "the rule composes inside a function" (fun () ->
        let x = v3 () in
        equal (close ())
          (Nx.mul_s (Nx.mul (Nx.sin x) (Nx.cos x)) 2.)
          (Rune.grad'
             (fun x ->
               let s = my_sin x in
               Nx.sum (Nx.mul s s))
             x));
    test "with no differentiation the call is its rule's value" (fun () ->
        equal (exact ()) (Nx.sin (v3 ())) (my_sin (v3 ())));
    test "a call on a constant inside grad contributes nothing" (fun () ->
        let c = v3 () in
        let s = Nx.item [] (Nx.sum (Nx.sin c)) in
        equal (close ())
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
        equal ~msg:"a" (exact ()) b ga;
        equal ~msg:"b" (exact ()) a gb);
    test "a pullback of another structure than the arguments is refused"
      (fun () ->
        let f =
          Rune.custom_vjp
            Nx.Ptree.(list tensor)
            tensor
            (fun p -> (Nx.sum (List.hd p), fun ct -> [ ct ]))
        in
        raises_match (Exn.invalid_arg ~substring:"the root: length 2")
          (fun () -> Rune.grad Nx.Ptree.(list tensor) f [ v3 (); vec [| 1. |] ]));
    test "a structured result" (fun () ->
        let f =
          Rune.custom_vjp tensor pair (fun x ->
              ( (Nx.sin x, Nx.mul x x),
                fun (ds, dsq) ->
                  Nx.add (Nx.mul ds (Nx.cos x)) (Nx.mul dsq (Nx.mul_s x 2.)) ))
        in
        let x = v3 () in
        equal ~msg:"both results" (close ())
          (Nx.add (Nx.cos x) (Nx.mul_s x 2.))
          (Rune.grad'
             (fun x ->
               let s, sq = f x in
               Nx.add (Nx.sum s) (Nx.sum sq))
             x);
        equal ~msg:"an unused result has a zero cotangent" (close ()) (Nx.cos x)
          (Rune.grad' (fun x -> Nx.sum (fst (f x))) x));
    test "jvp of a custom_vjp is refused" (fun () ->
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' my_sin (v3 ()) (along ())));
    test "a custom_vjp with no tensor result runs under forward mode" (fun () ->
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
        equal ~msg:"tangent" (close ()) (Nx.mul v (Nx.cos x)) dy;
        equal ~msg:"rule runs" int 1 !runs);
  ]

(* custom_jvp *)

let jvp_tests =
  [
    test "the tangent map replaces the derivative" (fun () ->
        equal (exact ()) (Nx.full f64 [| 3 |] 100.)
          (snd (Rune.jvp' fake_jvp_sin (v3 ()) (Nx.ones f64 [| 3 |]))));
    test "a true tangent map matches the function's derivative" (fun () ->
        equal (close ())
          (snd (Rune.jvp' Nx.sin (v3 ()) (along ())))
          (snd (Rune.jvp' my_sin_fwd (v3 ()) (along ()))));
    test "a structured result" (fun () ->
        let f =
          Rune.custom_jvp tensor pair (fun x ->
              ( (Nx.sin x, Nx.mul x x),
                fun dx -> (Nx.mul dx (Nx.cos x), Nx.mul dx (Nx.mul_s x 2.)) ))
        in
        let x = v3 () and v = along () in
        equal ~msg:"first" (close ())
          (Nx.mul v (Nx.cos x))
          (snd (Rune.jvp' (fun x -> fst (f x)) x v));
        equal ~msg:"second" (close ())
          (Nx.mul v (Nx.mul_s x 2.))
          (snd (Rune.jvp' (fun x -> snd (f x)) x v)));
    test "a tangent of another shape than the result is refused" (fun () ->
        let f =
          Rune.custom_jvp tensor tensor (fun x ->
              (Nx.sin x, fun dx -> Nx.sum dx))
        in
        raises_match (Exn.invalid_arg ~substring:"shape") (fun () ->
            Rune.jvp' f (v3 ()) (v3 ())));
    test "under reverse mode the tangent map is transposed" (fun () ->
        equal ~msg:"a true map" (close ())
          (Nx.cos (v3 ()))
          (Rune.grad' (fun x -> Nx.sum (my_sin_fwd x)) (v3 ()));
        equal ~msg:"the map's own derivative" (exact ())
          (Nx.full f64 [| 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (fake_jvp_sin x)) (v3 ())));
    test "with no differentiation the call is its rule's value" (fun () ->
        equal (exact ()) (Nx.sin (v3 ())) (my_sin_fwd (v3 ())));
    test "a custom_jvp with no tensor result runs once under reverse mode"
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
        equal ~msg:"loss" (exact ()) l' l;
        equal ~msg:"gradient" (exact ()) g' g;
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
        equal ~msg:"float" (close ()) (Nx.mul (along ()) (Nx.cos x)) ds;
        equal ~msg:"integer" (exact ()) (Nx.zeros Nx.int32 [| 2 |]) dk;
        equal ~msg:"under grad" (close ()) (Nx.cos x)
          (Rune.grad' (fun x -> Nx.sum (fst (f x))) x));
    test "a custom_vjp's integer result leaf gets a zero cotangent" (fun () ->
        let seen = ref None in
        let f =
          Rune.custom_vjp tensor with_int (fun x ->
              ( (Nx.sin x, int_leaf ()),
                fun (g, gk) ->
                  seen := Some gk;
                  Nx.mul g (Nx.cos x) ))
        in
        let x = v3 () in
        equal ~msg:"gradient" (close ()) (Nx.cos x)
          (Rune.grad' (fun x -> Nx.sum (fst (f x))) x);
        equal ~msg:"the integer cotangent" (exact ())
          (Nx.zeros Nx.int32 [| 2 |])
          (Option.get !seen));
  ]

(* Under a map *)

(* [scaled c] multiplies by [c] with a true rule; [c] is a map's lane. *)
let lane i x = Nx.get [ i ] x
let stack n f = Nx.stack (List.init n f)

let map_tests =
  [
    test "per-example gradients through a custom_vjp" (fun () ->
        equal (close ())
          (Nx.cos (xs ()))
          (Rune.vmap' (Rune.grad' (fun x -> Nx.sum (my_sin x))) (xs ())));
    test "a map of a custom_vjp is the map of its value" (fun () ->
        equal (exact ()) (Nx.sin (xs ())) (Rune.vmap' my_sin (xs ())));
    test "grad of a map applies a custom_vjp's pullback to the lanes" (fun () ->
        equal ~msg:"true" (close ())
          (Nx.cos (xs ()))
          (Rune.grad' (fun x -> Nx.sum (Rune.vmap' my_sin x)) (xs ()));
        equal ~msg:"fake" (exact ())
          (Nx.full f64 [| 2; 3 |] 100.)
          (Rune.grad' (fun x -> Nx.sum (Rune.vmap' fake_grad_sin x)) (xs ())));
    test "jvp of a map of a custom_vjp is refused" (fun () ->
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' (Rune.vmap' fake_grad_sin) (xs ()) (dxs ())));
    test "jvp of a map applies a custom_jvp's tangent map to the lanes"
      (fun () ->
        equal ~msg:"true" (close ())
          (Nx.mul (dxs ()) (Nx.cos (xs ())))
          (snd (Rune.jvp' (Rune.vmap' my_sin_fwd) (xs ()) (dxs ())));
        equal ~msg:"fake" (close ())
          (Nx.mul_s (dxs ()) 100.)
          (snd (Rune.jvp' (Rune.vmap' fake_jvp_sin) (xs ()) (dxs ()))));
    test "a tangent map inside a map sees each lane's tangent" (fun () ->
        let seen : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make () in
        let observe =
          Rune.custom_jvp tensor Nx.Ptree.unit (fun _ ->
              ((), fun dy -> Rune.Total.add seen dy))
        in
        let _, total =
          Rune.Total.collect seen ~zero:(Nx.zeros f64 [| 3 |]) (fun () ->
              Rune.jvp'
                (Rune.vmap' (fun r ->
                     observe (Nx.sin r);
                     r))
                (xs ()) (dxs ()))
        in
        equal (close ())
          (Nx.sum ~axes:[ 0 ] (Nx.mul (dxs ()) (Nx.cos (xs ()))))
          total);
    test "grad of a map applies a custom_jvp's transposed tangent map"
      (fun () ->
        equal (close ())
          (Nx.cos (xs ()))
          (Rune.grad' (fun x -> Nx.sum (Rune.vmap' my_sin_fwd x)) (xs ())));
    test "a map passes on a custom_jvp that reads its lanes" (fun () ->
        let ys =
          Nx.create f64 [| 4; 3 |]
            (Array.init 12 (fun i -> Float.sin (Float.of_int i)))
        in
        let cs = xs () and x0 = v3 () and w = Nx.scalar f64 1.5 in
        let scaled c =
          Rune.custom_jvp tensor tensor (fun x ->
              (Nx.mul x c, fun dx -> Nx.mul dx c))
        in
        let check ~msg f loop =
          let v, g = Rune.value_and_grad' (fun w -> Nx.sum (f w)) w in
          equal ~msg:(msg ^ ", value") (close ()) (Nx.mul_s (Nx.sum loop) 1.5) v;
          equal ~msg:(msg ^ ", gradient") (close ()) (Nx.sum loop) g;
          let y, dy = Rune.jvp' f w (Nx.scalar f64 1.) in
          equal ~msg:(msg ^ ", primal") (close ()) (Nx.mul_s loop 1.5) y;
          equal ~msg:(msg ^ ", tangent") (close ()) loop dy
        in
        check ~msg:"nested maps"
          (fun w ->
            Rune.vmap'
              (fun c -> Rune.vmap' (fun x -> Nx.mul w (scaled c x)) ys)
              cs)
          (stack 2 (fun j -> stack 4 (fun i -> Nx.mul (lane i ys) (lane j cs))));
        check ~msg:"unbatched arguments"
          (fun w -> Rune.vmap' (fun c -> Nx.mul w (scaled c x0)) cs)
          (stack 2 (fun j -> Nx.mul x0 (lane j cs))));
    test "a map passes on a custom_vjp that reads its lanes" (fun () ->
        let ys =
          Nx.create f64 [| 4; 3 |]
            (Array.init 12 (fun i -> Float.sin (Float.of_int i)))
        in
        let cs = xs () and x0 = v3 () and w = Nx.scalar f64 1.5 in
        (* The pullback is twice the true one. *)
        let scaled c =
          Rune.custom_vjp tensor tensor (fun p ->
              ( Nx.mul p c,
                fun g ->
                  let g = Nx.mul_s (Nx.mul g c) 2. in
                  if Nx.ndim p = 0 then Nx.sum g else g ))
        in
        let check ~msg f loop =
          let v, g = Rune.value_and_grad' (fun w -> Nx.sum (f w)) w in
          equal ~msg:(msg ^ ", value") (close ()) (Nx.mul_s (Nx.sum loop) 1.5) v;
          equal ~msg:(msg ^ ", gradient") (close ())
            (Nx.mul_s (Nx.sum loop) 2.)
            g
        in
        check ~msg:"nested maps"
          (fun w ->
            Rune.vmap'
              (fun c -> Rune.vmap' (fun x -> scaled c (Nx.mul w x)) ys)
              cs)
          (stack 2 (fun j -> stack 4 (fun i -> Nx.mul (lane i ys) (lane j cs))));
        check ~msg:"unbatched arguments"
          (fun w -> Rune.vmap' (fun c -> Nx.mul (scaled c w) x0) cs)
          (stack 2 (fun j -> Nx.mul x0 (lane j cs)));
        let shared =
          Rune.custom_vjp tensor tensor (fun w ->
              (w, fun _ -> Nx.scalar f64 1.))
        in
        equal ~msg:"a cotangent every lane shares, once per lane" (exact ())
          (Nx.scalar f64 2.)
          (Rune.grad'
             (fun w -> Nx.sum (Rune.vmap' (fun c -> Nx.mul (shared w) c) cs))
             w));
  ]

(* Under a compiled function *)

let compiled_tests =
  [
    test "a custom_vjp's pullback under jit, replayed" (fun () ->
        let f =
          Rune.jit' (Rune.grad' (fun x -> Nx.sum (Nx.mul (fake_grad_sin x) x)))
        in
        List.iter
          (fun x ->
            equal (close ()) (Nx.add (Nx.mul_s x 100.) (Nx.sin x)) (f x))
          [ v3 (); vec [| -0.3; 0.4; 1.2 |] ]);
    test "a custom_jvp's tangent map under jit, replayed" (fun () ->
        let f =
          Rune.jit' (fun x -> snd (Rune.jvp' fake_jvp_sin x (Nx.mul_s x 2.)))
        in
        List.iter
          (fun x -> equal (close ()) (Nx.mul_s x 200.) (f x))
          [ v3 (); vec [| -0.3; 0.4; 1.2 |] ]);
    test "a custom_vjp whose pullback scatters, under jit, replayed" (fun () ->
        let indices = Nx.create Nx.int64 [| 3 |] [| 2L; 0L; 2L |] in
        let take =
          Rune.custom_vjp tensor tensor (fun x ->
              ( Nx.take ~axis:0 ~indices x,
                fun ct ->
                  Nx.scatter ~mode:`Add ~axis:0 ~indices
                    ~values:(Nx.mul_s ct 7.) (Nx.zeros_like x) ))
        in
        let f =
          Rune.jit'
            (Rune.grad' (fun x ->
                 let y = take x in
                 Nx.sum (Nx.mul y y)))
        in
        equal ~msg:"first" (close ())
          (vec [| 14.; 0.; 84.; 0. |])
          (f (vec [| 1.; 2.; 3.; 4. |]));
        equal ~msg:"replayed" (close ())
          (vec [| 28.; 0.; 112.; 0. |])
          (f (vec [| 2.; 3.; 4.; 5. |])));
  ]

let () =
  exit
    (run "Rune custom rules"
       [
         group "custom_vjp" vjp_tests;
         group "custom_jvp" jvp_tests;
         group "integer result leaves" int_tests;
         group "under a map" map_tests;
         group "under a compiled function" compiled_tests;
       ])
