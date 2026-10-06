(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune's constructs (remat, the custom rules, lanes, lane indices, totals) with
   no transformation around them, and the exceptions and effects of the code
   they run: each reaches the call or the application that ran it, under every
   transformation. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let tensor = Nx.Ptree.tensor

(* With no transformation *)

let default_tests =
  [
    test "remat with no transformation runs its function once" (fun () ->
        let runs = ref 0 in
        let f x =
          incr runs;
          Nx.sin x
        in
        let x = vec [| 0.5; -1. |] in
        equal (exact ()) (Nx.sin x)
          (Rune.remat Nx.Ptree.(tensor @-> returns tensor) f x);
        equal ~msg:"runs" int 1 !runs);
    test "custom_jvp with no transformation runs its rule once and not its map"
      (fun () ->
        let rules = ref 0 and maps = ref 0 in
        let g =
          Rune.custom_jvp tensor tensor (fun x ->
              incr rules;
              ( Nx.sin x,
                fun dx ->
                  incr maps;
                  dx ))
        in
        let x = vec [| 0.5 |] in
        equal (exact ()) (Nx.sin x) (g x);
        equal ~msg:"rules" int 1 !rules;
        equal ~msg:"maps" int 0 !maps);
    test
      "custom_vjp with no transformation runs its rule once and not its \
       pullback" (fun () ->
        let rules = ref 0 and pullbacks = ref 0 in
        let g =
          Rune.custom_vjp tensor tensor (fun x ->
              incr rules;
              ( Nx.sin x,
                fun ct ->
                  incr pullbacks;
                  ct ))
        in
        let x = vec [| 0.5 |] in
        equal (exact ()) (Nx.sin x) (g x);
        equal ~msg:"rules" int 1 !rules;
        equal ~msg:"pullbacks" int 0 !pullbacks);
    test "lanes with no map around the call is one lane" (fun () ->
        let x = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
        equal (exact ())
          (Nx.unsqueeze ~axes:[ 0 ] x)
          (Rune.lanes (Rune.axis ()) x));
    test "the lane index with no map around the call is 0" (fun () ->
        equal (exact ()) (Nx.scalar Nx.int32 0l) (Rune.lane_index ()));
    test "an addition with no scope open does nothing" (fun () ->
        let t : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make () in
        Rune.Total.add t (scalar 5.);
        let (), sum = Rune.Total.collect t ~zero:(scalar 1.) (fun () -> ()) in
        equal (exact ()) (scalar 1.) sum);
  ]

(* Exceptions *)

exception Boom

let boom () = raise Boom

(* The calls whose code raises: [remat]'s function, each rule, a scan's body. *)
let calls =
  [
    ( "remat",
      fun x ->
        Rune.remat Nx.Ptree.(tensor @-> returns tensor) (fun _ -> boom ()) x );
    ("custom_jvp", fun x -> Rune.custom_jvp tensor tensor (fun _ -> boom ()) x);
    ("custom_vjp", fun x -> Rune.custom_vjp tensor tensor (fun _ -> boom ()) x);
    ( "scan",
      fun x ->
        fst
          (Rune.scan'
             ~f:(fun _ _ -> boom ())
             ~init:x
             (Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])) );
  ]

(* The transformations' own errors, each raised at the operation that makes it:
   an operation with no derivative, a lane read inside a map, a tangent map's
   tangent of another shape, an addition of another shape. *)
let no_derivative =
  ( "an operation with no derivative",
    fun x ->
      ignore (Nx.svd ~full_matrices:true (Nx.reshape [| 1; 3 |] x));
      x )

let read_in_a_map =
  ( "a value read in a map",
    fun x ->
      ignore (Nx.to_array x);
      x )

let tangent_shape =
  ( "a tangent map's tangent of another shape",
    fun x ->
      Rune.custom_jvp tensor tensor
        (fun x -> (Nx.mul_s x 2., fun dx -> Nx.sum dx))
        x )

(* [guarded call finalised x] is [call x], or [3 x] when it raises [Boom] or
   [Invalid_argument]; it counts its finaliser's runs. *)
let guarded call finalised x =
  match Fun.protect ~finally:(fun () -> incr finalised) (fun () -> call x) with
  | y -> y
  | exception (Boom | Invalid_argument _) -> Nx.mul_s x 3.

let x0 () = vec [| 0.5; -1.2; 2.1 |]
let v0 () = vec [| 1.; 0.5; -2. |]
let xs () = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let vs () = Nx.create f64 [| 2; 3 |] [| 1.; 0.5; -2.; 0.3; 1.5; -0.7 |]
let seen : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let addition_shape =
  ( "an addition of another shape",
    fun x ->
      Rune.Total.add seen (Nx.sum x);
      x )

(* Each transformation of a function [g], its argument, the transformation of [3
   x], and the errors of its own it raises. *)
let transformations =
  [
    ( "no transformation",
      (fun g x -> g x),
      x0,
      (fun () -> Nx.mul_s (x0 ()) 3.),
      [] );
    ( "grad",
      (fun g x -> Rune.grad' (fun x -> Nx.sum (g x)) x),
      x0,
      (fun () -> Nx.full f64 [| 3 |] 3.),
      [ no_derivative ] );
    ( "jvp",
      (fun g x -> snd (Rune.jvp' g x (v0 ()))),
      x0,
      (fun () -> Nx.mul_s (v0 ()) 3.),
      [ no_derivative; tangent_shape ] );
    ( "vmap",
      (fun g x -> Rune.vmap' g x),
      xs,
      (fun () -> Nx.mul_s (xs ()) 3.),
      [ read_in_a_map ] );
    ( "jvp of a map",
      (fun g x -> snd (Rune.jvp' (Rune.vmap' g) x (vs ()))),
      xs,
      (fun () -> Nx.mul_s (vs ()) 3.),
      [ no_derivative; read_in_a_map; tangent_shape ] );
    ( "a total's scope",
      (fun g x ->
        fst
          (Rune.Total.collect seen ~zero:(Nx.zeros f64 [| 3 |]) (fun () -> g x))),
      x0,
      (fun () -> Nx.mul_s (x0 ()) 3.),
      [ addition_shape ] );
    ( "a remat's function",
      (fun g x ->
        Rune.grad'
          (fun x ->
            Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) g x))
          x),
      x0,
      (fun () -> Nx.full f64 [| 3 |] 3.),
      [] );
  ]

let exception_tests =
  List.map
    (fun (name, transform, x, expected, errors) ->
      let check ~compile call () =
        let finalised = ref 0 in
        let run = transform (guarded call finalised) in
        let y = if compile then Rune.jit' run (x ()) else run (x ()) in
        equal (Oracle.tensor ~rel:1e-12 ()) (expected ()) y;
        is_true ~msg:"finalised" (!finalised > 0)
      in
      group ("under " ^ name)
        (List.concat_map
           (fun (call_name, call) ->
             [
               test call_name (check ~compile:false call);
               test (call_name ^ ", compiled") (check ~compile:true call);
             ])
           (calls @ errors)))
    transformations

let later_tests =
  [
    test "an operation a compiled function refuses raises at its call"
      (fun () ->
        let guarded x =
          match Nx.rfft Nx.complex128 x with
          | _ -> x
          | exception Rune.Jit_error _ -> Nx.mul_s x 3.
        in
        equal (exact ()) (Nx.mul_s (x0 ()) 3.) (Rune.jit' guarded (x0 ())));
    test "a pullback's exception reaches the backward pass's caller" (fun () ->
        let g =
          Rune.custom_vjp tensor tensor (fun x -> (Nx.sin x, fun _ -> boom ()))
        in
        raises Boom (fun () -> Rune.grad' (fun x -> Nx.sum (g x)) (x0 ())));
    test "a pullback's exception reaches the pullback's caller" (fun () ->
        let g =
          Rune.custom_vjp tensor tensor (fun x -> (Nx.sin x, fun _ -> boom ()))
        in
        let _, pb = Rune.vjp' g (x0 ()) in
        raises Boom (fun () -> pb (x0 ())));
    test "a tangent map's exception under grad reaches the call" (fun () ->
        let g =
          Rune.custom_jvp tensor tensor (fun x -> (Nx.sin x, fun _ -> boom ()))
        in
        let caught = ref false in
        ignore
          (Rune.grad'
             (fun x ->
               match g x with
               | y -> Nx.sum y
               | exception Boom ->
                   caught := true;
                   Nx.sum x)
             (x0 ()));
        is_true !caught);
    test "a tangent map's exception under jvp reaches the call" (fun () ->
        let g =
          Rune.custom_jvp tensor tensor (fun x -> (Nx.sin x, fun _ -> boom ()))
        in
        equal (exact ()) (v0 ())
          (snd
             (Rune.jvp'
                (fun x -> match g x with y -> y | exception Boom -> x)
                (x0 ()) (v0 ()))));
    test
      "a remat function that raises when it runs again raises from the \
       backward pass" (fun () ->
        let runs = ref 0 in
        let f x =
          incr runs;
          if !runs > 1 then boom () else Nx.sin x
        in
        raises Boom (fun () ->
            Rune.grad'
              (fun x ->
                Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) f x))
              (x0 ())));
  ]

(* Effects *)

type _ Effect.t += Ask : float Effect.t

(* [answer f] is [f ()] with [Ask] answered by 1. *)
let answer f =
  Effect.Deep.match_with f ()
    {
      retc = Fun.id;
      exnc = raise;
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Ask ->
              Some
                (fun (k : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue k 1.)
          | _ -> None);
    }

(* The custom rules whose rule performs [Ask]. *)
let rules runs =
  let asking x =
    incr runs;
    Nx.add_s x (Effect.perform Ask)
  in
  [
    ( "custom_jvp",
      fun x -> Rune.custom_jvp tensor tensor (fun x -> (asking x, Fun.id)) x );
    ( "custom_vjp",
      fun x -> Rune.custom_vjp tensor tensor (fun x -> (asking x, Fun.id)) x );
  ]

let effect_tests =
  let unanswered runs x =
    incr runs;
    Nx.add_s x (Effect.perform Ask)
  in
  let effect_calls runs =
    [
      ( "remat",
        fun x ->
          Rune.remat Nx.Ptree.(tensor @-> returns tensor) (unanswered runs) x );
      ( "custom_jvp",
        fun x ->
          Rune.custom_jvp tensor tensor (fun x -> (unanswered runs x, Fun.id)) x
      );
      ( "custom_vjp",
        fun x ->
          Rune.custom_vjp tensor tensor (fun x -> (unanswered runs x, Fun.id)) x
      );
    ]
  in
  [
    test "an effect a construct's code leaves unhandled is that code's"
      (fun () ->
        let runs = ref 0 in
        List.iter
          (fun (name, call) ->
            runs := 0;
            let g x =
              match call x with
              | y -> y
              | exception Effect.Unhandled Ask -> Nx.mul_s x 3.
            in
            equal ~msg:name (exact ()) (Nx.full f64 [| 3 |] 3.)
              (Rune.grad' (fun x -> Nx.sum (g x)) (x0 ()));
            equal ~msg:(name ^ ": runs") int 1 !runs)
          (effect_calls runs));
    test
      "a custom rule's effects reach the handlers around its call when nothing \
       differentiates it" (fun () ->
        let runs = ref 0 in
        List.iter
          (fun (name, call) ->
            equal ~msg:name (exact ())
              (Nx.add_s (x0 ()) 1.)
              (answer (fun () -> call (x0 ()))))
          (rules runs));
    test
      "a custom rule's effects reach the handlers around its call under the \
       differentiation that applies it" (fun () ->
        let runs = ref 0 in
        List.iter
          (fun (name, call) ->
            equal ~msg:name (exact ()) (Nx.ones f64 [| 3 |])
              (Rune.grad' (fun x -> answer (fun () -> Nx.sum (call x))) (x0 ())))
          (rules runs));
  ]

(* Values with no derivative *)

(* [a + √c], whose derivative in [c] is infinite at [c = 0]: a construct that
   gave [c] a zero tangent, or a zero cotangent, would meet that infinity and
   give NaN. [c] is a constant, passed as an argument. *)
let pair = Nx.Ptree.(pair tensor tensor)
let root a c = Nx.add a (Nx.sqrt c)

let through_constructs =
  [
    ( "remat",
      fun a c ->
        Rune.remat Nx.Ptree.(tensor @-> tensor @-> returns tensor) root a c );
    ( "custom_jvp",
      fun a c ->
        Rune.custom_jvp pair tensor
          (fun (a, c) -> (root a c, fun (da, _) -> da))
          (a, c) );
    ( "custom_vjp",
      fun a c ->
        Rune.custom_vjp pair tensor
          (fun (a, c) ->
            (root a c, fun g -> (g, Nx.div g (Nx.mul_s (Nx.sqrt c) 2.))))
          (a, c) );
  ]

let no_forward =
  "Rune.jvp': a custom_vjp rule has no forward derivative; give the function a \
   custom_jvp rule"

let untracked_tests =
  let zero () = vec [| 0. |]
  and two () = vec [| 2. |]
  and one () = vec [| 1. |] in
  List.concat_map
    (fun (name, g) ->
      [
        test (name ^ " under jvp: a constant argument adds no term") (fun () ->
            let run () =
              snd (Rune.jvp' (fun x -> g x (zero ())) (two ()) (one ()))
            in
            if name = "custom_vjp" then raises (Invalid_argument no_forward) run
            else equal (exact ()) (one ()) (run ()));
        test (name ^ " under grad: a constant argument adds no term") (fun () ->
            equal (exact ()) (one ())
              (Rune.grad' (fun x -> Nx.sum (g x (zero ()))) (two ())));
      ])
    through_constructs
  @
  let passed () =
    Rune.remat
      Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun a c -> (a, c))
  in
  let f x =
    let a, c = passed () x (zero ()) in
    root a c
  in
  [
    test "a remat's result with no tangent adds no term under jvp" (fun () ->
        equal (exact ()) (one ()) (snd (Rune.jvp' f (two ()) (one ()))));
    test "a remat's result with no tangent adds no term under grad" (fun () ->
        equal (exact ()) (one ()) (Rune.grad' (fun x -> Nx.sum (f x)) (two ())));
  ]

(* Rules whose result is an argument, rules under a map *)

let rule_tests =
  let w () = vec [| 2.; -3. |] and ones () = vec [| 1.; 1. |] in
  [
    test "a custom_vjp whose result is its argument adds its cotangent once"
      (fun () ->
        let id = Rune.custom_vjp tensor tensor (fun x -> (x, fun ct -> ct)) in
        equal (exact ())
          (vec [| 5.; -5. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.add (id x) (Nx.mul x x))) (w ())));
    test "a remat of the identity adds its cotangent once" (fun () ->
        let r = Rune.remat Nx.Ptree.(tensor @-> returns tensor) Fun.id in
        equal (exact ())
          (vec [| 4.; -6. |])
          (Rune.grad' (fun x -> Nx.sum (Nx.mul (r x) x)) (w ())));
    test
      "a custom_jvp whose result is its argument keeps the argument's tangent"
      (fun () ->
        let double =
          Rune.custom_jvp tensor tensor (fun x -> (x, fun dx -> Nx.mul_s dx 2.))
        in
        equal ~msg:"rule first" (exact ())
          (vec [| 3.; 3. |])
          (snd (Rune.jvp' (fun x -> Nx.add (double x) x) (w ()) (ones ())));
        equal ~msg:"rule second" (exact ())
          (vec [| 3.; 3. |])
          (snd (Rune.jvp' (fun x -> Nx.add x (double x)) (w ()) (ones ()))));
    test "under vmap a custom_vjp's pullback runs for each lane" (fun () ->
        (* The pullback is twice the true one, so applying it shows. *)
        let doubled =
          Rune.custom_vjp tensor tensor (fun x ->
              (Nx.sin x, fun g -> Nx.mul g (Nx.mul_s (Nx.cos x) 2.)))
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.mul_s (Nx.cos (xs ())) 2.)
          (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' doubled xs)) (xs ())));
    test "jvp of a mapped custom_vjp is refused" (fun () ->
        let rule =
          Rune.custom_vjp tensor tensor (fun x -> (Nx.sin x, fun g -> g))
        in
        raises (Invalid_argument no_forward) (fun () ->
            Rune.jvp' (Rune.vmap' rule) (xs ()) (xs ())));
  ]

(* A remat and what it captures *)

(* A layer that closes over its weight: every transformation reaches the weight
   through the remat as it does without one. *)
let at () = vec [| 0.7; -1.3; 2.1 |]
let along () = vec [| 0.5; 1.; -2. |]
let layer w x = Nx.mul (Nx.exp x) w
let rematted = Rune.remat Nx.Ptree.(tensor @-> returns tensor)
let plainly f = f
let close () = Oracle.tensor ~rel:1e-12 ()

let hvp loss w =
  Rune.grad' (fun w -> Nx.sum (Nx.mul (Rune.grad' loss w) (along ()))) w

let capture_tests =
  [
    test
      "a remat whose rerun captures a value its first run did not raises at \
       the backward pass" (fun () ->
        let again = ref false in
        let f w x =
          if !again then Nx.mul x w
          else begin
            again := true;
            Nx.sin x
          end
        in
        raises
          (Invalid_argument
             "Rune.grad': a function run again for its transpose reads a value \
              the differentiation tracks that its first run did not") (fun () ->
            Rune.grad' (fun w -> Nx.sum (rematted (f w) (Nx.cos w))) (at ())));
    test "a remat capturing two weights gives each its gradient" (fun () ->
        let loss r (u, v) =
          Nx.sum
            (r (fun x -> Nx.mul (Nx.mul (Nx.sin x) u) (Nx.mul v u)) (at ()))
        in
        equal
          (Oracle.structure ~rel:1e-12 Nx.Ptree.(pair tensor tensor))
          (Rune.grad
             Nx.Ptree.(pair tensor tensor)
             (loss plainly)
             (along (), at ()))
          (Rune.grad
             Nx.Ptree.(pair tensor tensor)
             (loss rematted)
             (along (), at ())));
    test "a remat whose result depends on no argument passes no cotangent"
      (fun () ->
        let c = vec [| 1.; 2.; 3. |] in
        let r = Rune.remat Nx.Ptree.(tensor @-> returns tensor) (fun _ -> c) in
        equal (exact ()) (Nx.ones f64 [| 3 |])
          (Rune.grad' (fun x -> Nx.add (Nx.sum x) (Nx.sum (r x))) (at ())));
    test "a captured weight's gradient" (fun () ->
        let loss r w = Nx.sum (Nx.sin (r (layer w) (at ()))) in
        equal (close ())
          (Rune.grad' (loss plainly) (along ()))
          (Rune.grad' (loss rematted) (along ())));
    test "a captured weight's tangent" (fun () ->
        let tangent r w =
          snd (Rune.jvp' (fun w -> r (layer w) (at ())) w (along ()))
        in
        equal (close ()) (tangent plainly (at ())) (tangent rematted (at ())));
    test "a weight both captured and passed gets both shares" (fun () ->
        let loss r w = Nx.sum (r (layer w) (Nx.mul w w)) in
        equal (close ())
          (Rune.grad' (loss plainly) (along ()))
          (Rune.grad' (loss rematted) (along ())));
    test "an argument the function also captures gets both shares" (fun () ->
        let loss r w = Nx.sum (r (fun x -> Nx.mul (Nx.sin x) w) w) in
        equal (close ())
          (Rune.grad' (loss plainly) (along ()))
          (Rune.grad' (loss rematted) (along ())));
    test "second derivatives in a weight passed to the remat" (fun () ->
        let passed r w =
          Nx.sum
            (Nx.sin (r (fun w x -> Nx.tanh (Nx.mul x w)) w (Nx.cos (at ()))))
        in
        equal (close ())
          (hvp (passed plainly) (along ()))
          (hvp
             (passed
                (Rune.remat Nx.Ptree.(tensor @-> tensor @-> returns tensor)))
             (along ())));
    test "second derivatives in a weight the remat captures" (fun () ->
        let captured r w =
          Nx.sum (Nx.sin (r (fun x -> Nx.mul (Nx.exp x) (Nx.mul w w)) (at ())))
        in
        equal (close ())
          (hvp (captured plainly) (along ()))
          (hvp (captured rematted) (along ())));
    test "a remat inside a map captures the lane" (fun () ->
        let lane r x = r (fun c -> Nx.mul (Nx.sin c) x) (along ()) in
        equal ~msg:"values" (close ())
          (Rune.vmap' (lane plainly) (xs ()))
          (Rune.vmap' (lane rematted) (xs ()));
        let lanes r xs = Nx.sum (Rune.vmap' (lane r) xs) in
        equal ~msg:"grad of vmap" (close ())
          (Rune.grad' (lanes plainly) (xs ()))
          (Rune.grad' (lanes rematted) (xs ())));
  ]

let () =
  exit
    (run "Rune constructs"
       [
         group "with no transformation" default_tests;
         group "exceptions" (exception_tests @ later_tests);
         group "effects" effect_tests;
         group "values with no derivative" untracked_tests;
         group "rules whose result is an argument, rules under a map" rule_tests;
         group "a remat and what it captures" capture_tests;
       ])
