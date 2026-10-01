(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune's constructs (remat, the custom rules, lanes, lane indices, totals) with
   no transformation around them, and the exceptions and effects of the code
   they run: each reaches the call or the application that ran it, under every
   transformation. *)

open Windtrap
module Rune = Rune_next.Rune

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

(* [guarded call finalised x] is [call x], or [3 x] when it raises [Boom]; it
   counts its finaliser's runs. *)
let guarded call finalised x =
  match Fun.protect ~finally:(fun () -> incr finalised) (fun () -> call x) with
  | y -> y
  | exception Boom -> Nx.mul_s x 3.

let x0 () = vec [| 0.5; -1.2; 2.1 |]
let v0 () = vec [| 1.; 0.5; -2. |]
let xs () = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |]
let seen : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

(* Each transformation of a function [g], its argument, and the transformation
   of [3 x]. *)
let transformations =
  [
    ("no transformation", (fun g x -> g x), x0, fun () -> Nx.mul_s (x0 ()) 3.);
    ( "grad",
      (fun g x -> Rune.grad' (fun x -> Nx.sum (g x)) x),
      x0,
      fun () -> Nx.full f64 [| 3 |] 3. );
    ( "jvp",
      (fun g x -> snd (Rune.jvp' g x (v0 ()))),
      x0,
      fun () -> Nx.mul_s (v0 ()) 3. );
    ("vmap", (fun g x -> Rune.vmap' g x), xs, fun () -> Nx.mul_s (xs ()) 3.);
    ( "a total's scope",
      (fun g x ->
        fst
          (Rune.Total.collect seen ~zero:(Nx.zeros f64 [| 3 |]) (fun () -> g x))),
      x0,
      fun () -> Nx.mul_s (x0 ()) 3. );
    ( "a remat's function",
      (fun g x ->
        Rune.grad'
          (fun x ->
            Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) g x))
          x),
      x0,
      fun () -> Nx.full f64 [| 3 |] 3. );
  ]

let exception_tests =
  List.map
    (fun (name, transform, x, expected) ->
      group ("under " ^ name)
        (List.map
           (fun (call_name, call) ->
             test call_name (fun () ->
                 let finalised = ref 0 in
                 equal
                   (Oracle.tensor ~rel:1e-12 ())
                   (expected ())
                   (transform (guarded call finalised) (x ()));
                 is_true ~msg:"finalised" (!finalised > 0)))
           calls))
    transformations

let later_tests =
  [
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
      "a custom rule's effects reach the handlers around the differentiation \
       that applies it" (fun () ->
        let runs = ref 0 in
        List.iter
          (fun (name, call) ->
            equal ~msg:name (exact ()) (Nx.ones f64 [| 3 |])
              (answer (fun () -> Rune.grad' (fun x -> Nx.sum (call x)) (x0 ()))))
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

let () =
  exit
    (run "Rune constructs"
       [
         group "with no transformation" default_tests;
         group "exceptions" (exception_tests @ later_tests);
         group "effects" effect_tests;
         group "values with no derivative" untracked_tests;
       ])
