(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Loops under a stager on the host: each transformation installed between a
   compiled call's stager and a loop passes the loop on transformed, and the
   stager folds it. The stager's protocol is private to rune, so this suite
   links rune_internals. The trusted side is the same loss run eagerly. *)

open Windtrap
module Rune = Rune_internals.Rune

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let close () = Oracle.tensor ~rel:1e-12 ()

module Construct = Rune_internals.Construct

(* [staged f] is [f ()] under a stager that answers each loop by folding it at
   its call, as a compiled call stages one: every installation between passes
   the loop on transformed, and its step runs inside them, under the stager
   again. [steps] counts the step's runs, [outputs] the output tensors of each
   loop it folded, last first, and [late] the barriers that reached it with a
   traced value, one another installation should have passed on. *)
let steps = ref 0
let outputs = ref []
let late = ref 0

let rec staged : 'a. (unit -> 'a) -> 'a =
 fun f ->
  let traced =
    List.exists (fun (Nx.P x) ->
        match Nx.Repr.v x with Traced _ -> true | Host _ | Placed _ -> false)
  in
  let call : type r. r Construct.t -> r Construct.answer option =
   fun c ->
    match[@warning "@4@8"] c with
    | Loop r ->
        let req_step i c x =
          incr steps;
          staged (fun () -> r.req_step i c x)
        in
        Some
          (Construct.here (fun () ->
               let r = Rune_internals.Trips.fold { r with req_step } in
               outputs := List.length r.r_ys :: !outputs;
               r))
    | Barrier { values; after } ->
        if traced values || traced after then incr late;
        Some (Construct.value (fun () -> values))
    | Compiled _ | Remat _ | Custom _ | Root _ | At_map _ | Lanes _
    | Lane_index _ | Lane_count _ | Add _ | Detach _ ->
        None
  in
  Construct.install { op = None; call } f

let sw = vec [| 0.3; -0.7; 1.1 |]
let sv = vec [| 1.; 0.5; -2. |]

let sws =
  Nx.create f64 [| 4; 3 |]
    (Array.init 12 (fun i -> (0.1 *. Float.of_int i) -. 0.4))

let sxs =
  Nx.create f64 [| 5; 3 |] (Array.init 15 (fun i -> sin (Float.of_int i)))

let one_tensor = Nx.Ptree.(tensor @-> returns tensor)
let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

(* Losses of [w] over a scan of [sxs], each with a shape of carry, rows or step
   that a transformed scan treats apart. *)
let staged_losses =
  [
    ( "a tracked carry",
      fun w ->
        let c, ys =
          Rune.scan'
            ~f:(fun c x ->
              let c = Nx.sin (Nx.add (Nx.mul c w) x) in
              (c, Nx.sum (Nx.mul c x)))
            ~init:(Nx.mul_s w 0.5) sxs
        in
        Nx.add (Nx.sum c) (Nx.sum (Nx.mul_s ys 2.)) );
    ( "a carry that becomes tracked",
      fun w ->
        let c, ys =
          Rune.scan'
            ~f:(fun c x -> (Nx.add (Nx.mul c x) (Nx.mul w x), Nx.mul c c))
            ~init:(Nx.zeros f64 [| 3 |]) sxs
        in
        Nx.add (Nx.sum (Nx.mul c c)) (Nx.sum ys) );
    ( "a carry passed through and outputs of a capture",
      fun w ->
        let c, ys =
          Rune.scan'
            ~f:(fun c x -> (c, Nx.mul w (Nx.add x c)))
            ~init:(Nx.ones f64 [| 3 |]) sxs
        in
        Nx.add (Nx.sum c) (Nx.sum (Nx.mul ys ys)) );
    ( "tracked rows",
      fun w ->
        let c, ys =
          Rune.scan'
            ~f:(fun c x -> (Nx.add (Nx.mul c x) x, Nx.exp x))
            ~init:(Nx.ones f64 [| 3 |])
            (Nx.mul sxs (Nx.reshape [| 1; 3 |] w))
        in
        Nx.add (Nx.sum c) (Nx.sum ys) );
    ( "a scan in the step",
      fun w ->
        let inner c x =
          let rows =
            Nx.reshape [| 3; 3 |] (Nx.concatenate ~axis:0 [ x; x; x ])
          in
          let d, _ =
            Rune.scan'
              ~f:(fun d y ->
                let d = Nx.mul (Nx.add d y) w in
                (d, d))
              ~init:c rows
          in
          (Nx.sin d, Nx.sum d)
        in
        Nx.sum (fst (Rune.scan' ~f:inner ~init:w sxs)) );
    ( "a total in the step",
      fun w ->
        let (c, ()), t =
          Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
              Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
                ~f:(fun c x ->
                  Rune.Total.add total (Nx.sum (Nx.mul (Nx.mul c w) x));
                  (Nx.sin (Nx.add c x), ()))
                ~init:w sxs)
        in
        Nx.add (Nx.sum c) t );
    ( "a remat in the step",
      fun w ->
        let step c x =
          let c =
            Rune.remat one_tensor (fun c -> Nx.sin (Nx.mul c w)) (Nx.add c x)
          in
          (c, c)
        in
        Nx.sum (fst (Rune.scan' ~f:step ~init:w sxs)) );
    ( "a scan in a remat that captures",
      fun w ->
        let u = Nx.mul_s w 1.5 in
        let f a =
          let step c x =
            let c = Nx.sin (Nx.add (Nx.mul c u) (Nx.mul x a)) in
            (c, Nx.mul c w)
          in
          let c, ys = Rune.scan' ~f:step ~init:a sxs in
          Nx.add (Nx.sum c) (Nx.sum ys)
        in
        let r = Rune.remat one_tensor f (Nx.cos w) in
        Nx.mul r r );
  ]

(* Losses of [w] over an iterate, which stops once every element of its carry is
   small: each lane of a map takes its own number of trips. *)
let small x = Nx.less_s (Nx.max (Nx.abs x)) 0.05
let contract w x = Nx.mul_s (Nx.mul x (Nx.tanh (Nx.add_s (Nx.mul w x) 0.3))) 0.9

let iterate_losses =
  [
    ( "an iterate with a tracked carry",
      fun w ->
        Nx.sum (Nx.sin (Rune.iterate' ~max:80 ~until:small ~f:(contract w) w))
    );
    ( "an iterate whose carry becomes tracked",
      fun w ->
        let x, k =
          Rune.iterate
            Nx.Ptree.(pair tensor tensor)
            ~max:80
            ~until:(fun (x, _) -> small x)
            ~f:(fun (x, k) -> (contract w x, Nx.add_s k 1.))
            (Nx.ones f64 [| 3 |], scalar 0.)
        in
        Nx.add (Nx.sum (Nx.mul x x)) k );
    ( "an iterate in a scan's step",
      fun w ->
        let step c x =
          let c =
            Rune.iterate' ~max:80 ~until:small ~f:(contract w)
              (Nx.add c (Nx.mul_s x 0.1))
          in
          (c, Nx.sum c)
        in
        let c, ys = Rune.scan' ~f:step ~init:w sxs in
        Nx.add (Nx.sum c) (Nx.sum ys) );
    ( "an iterate in an iterate's step",
      fun w ->
        let step x =
          let y =
            Rune.iterate' ~max:80 ~until:small ~f:(contract w) (Nx.mul_s x 1.5)
          in
          Nx.mul_s (Nx.add x y) 0.5
        in
        let coarse x = Nx.less_s (Nx.max (Nx.abs x)) 0.1 in
        Nx.sum (Nx.sin (Rune.iterate' ~max:80 ~until:coarse ~f:step w)) );
    ( "an untracked iterate beside a tracked value",
      fun w ->
        let x =
          Rune.iterate' ~max:80 ~until:small
            ~f:(contract (scalar 0.4))
            (Nx.ones f64 [| 3 |])
        in
        Nx.sum (Nx.mul (Nx.sin w) (Nx.add_s x 1.)) );
  ]

let applications =
  [
    ("the value", fun l -> l sw);
    ("grad", fun l -> Rune.grad' l sw);
    ("jvp", fun l -> snd (Rune.jvp' l sw sv));
    ("vmap", fun l -> Rune.vmap' l sws);
    ("vmap of grad", fun l -> Rune.vmap' (Rune.grad' l) sws);
    ("jvp of grad", fun l -> snd (Rune.jvp' (Rune.grad' l) sw sv));
    ( "grad of grad",
      fun l -> Rune.grad' (fun w -> Nx.sum (Nx.mul (Rune.grad' l w) sv)) sw );
    ( "grad of vmap",
      fun l ->
        Rune.grad'
          (fun w ->
            Nx.sum (Rune.vmap' l (Nx.mul sws (Nx.reshape [| 1; 3 |] w))))
          sw );
    ("vmap of jvp", fun l -> Rune.vmap' (fun w -> snd (Rune.jvp' l w sv)) sws);
  ]

(* The applications compiled, their inputs arguments of the compiled call. *)
let compiled_applications =
  let two = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
  [
    ("the value", fun l -> Rune.jit' l sw);
    ("grad", fun l -> Rune.jit' (Rune.grad' l) sw);
    ("jvp", fun l -> Rune.jit two (fun w v -> snd (Rune.jvp' l w v)) sw sv);
    ("vmap", fun l -> Rune.jit' (Rune.vmap' l) sws);
    ("vmap of grad", fun l -> Rune.jit' (Rune.vmap' (Rune.grad' l)) sws);
    ( "jvp of grad",
      fun l ->
        Rune.jit two (fun w v -> snd (Rune.jvp' (Rune.grad' l) w v)) sw sv );
    ( "grad of grad",
      fun l ->
        Rune.jit two
          (fun w v ->
            Rune.grad' (fun w -> Nx.sum (Nx.mul (Rune.grad' l w) v)) w)
          sw sv );
    ( "grad of vmap",
      fun l ->
        Rune.jit two
          (fun w ws ->
            Rune.grad'
              (fun w ->
                Nx.sum (Rune.vmap' l (Nx.mul ws (Nx.reshape [| 1; 3 |] w))))
              w)
          sw sws );
    ( "vmap of jvp",
      fun l ->
        Rune.jit two
          (fun ws v -> Rune.vmap' (fun w -> snd (Rune.jvp' l w v)) ws)
          sws sv );
  ]

let counter w =
  let (c, k), () =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun (c, k) x -> ((Nx.mul (Nx.add c x) w, Nx.add_s k 1l), ()))
      ~init:(w, Nx.scalar Nx.int32 0l)
      sxs
  in
  Nx.mul_s (Nx.sum c) (Int32.to_float (Nx.item [] k))

(* Untracked loops: one whose carry and outputs depend on no tracked value is
   not differentiated, whatever its step reads. Under grad the stager folds the
   forward loop alone, with its own outputs; a tracked one adds the carry it
   keeps per trip and the transposed loop. *)
let untracked_law =
  prop "grad differentiates a loop only when its carry or outputs are tracked"
    Gen.(triple bool bool bool)
    (fun (scan, tracked, reads) ->
      cover "an untracked loop whose step reads a tracked value"
        ((not tracked) && reads);
      let loss w =
        let start = if tracked then w else Nx.ones f64 [| 3 |] in
        let step x =
          let y = contract (scalar 0.4) x in
          if reads then ignore (Nx.mul y w);
          y
        in
        let x =
          if scan then
            fst
              (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
                 ~f:(fun x _ -> (step x, ()))
                 ~init:start sxs)
          else Rune.iterate' ~max:80 ~until:small ~f:step start
        in
        Nx.sum (Nx.mul (Nx.sin w) (Nx.add_s x 1.))
      in
      outputs := [];
      equal (close ()) (Rune.grad' loss sw)
        (staged (fun () -> Rune.grad' loss sw));
      (* The forward loop keeps the carry, and an iterate its trip count; a
         transposed loop follows. *)
      equal (list int) (if tracked then [ 0; 1 ] else [ 0 ]) !outputs)

let staged_tests =
  List.concat_map
    (fun (loss, l) ->
      List.map
        (fun (app, a) ->
          test
            (app ^ " of a loss with " ^ loss ^ " is the folded loop's")
            (fun () -> equal (close ()) (a l) (staged (fun () -> a l))))
        applications)
    (staged_losses @ iterate_losses)
  @ [ untracked_law ]
  @ List.map
      (fun (app, a) ->
        test (app ^ " of a loss with an integer counter in the carry")
          (fun () ->
            equal (close ()) (a counter) (staged (fun () -> a counter))))
      (List.filter
         (fun (app, _) -> List.mem app [ "grad"; "jvp"; "vmap" ])
         applications)
  @ [
      test
        "grad of a carry that becomes tracked restarts the step, and \
         transposes it once per row" (fun () ->
          let l = List.assoc "a carry that becomes tracked" staged_losses in
          steps := 0;
          ignore (staged (fun () -> Rune.grad' l sw));
          equal int (1 + 5 + 5) !steps);
      test "grad inside a total's scope adds what the forward pass adds"
        (fun () ->
          let f () =
            let g, t =
              Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
                  Rune.grad'
                    (fun w ->
                      let step c x =
                        Rune.Total.add total (Nx.sum (Nx.mul c x));
                        (Nx.sin (Nx.add (Nx.mul c w) x), ())
                      in
                      Nx.sum
                        (fst
                           (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor
                              Nx.Ptree.unit ~f:step ~init:w sxs)))
                    sw)
            in
            Nx.add (Nx.sum g) (Nx.mul_s t 1000.)
          in
          equal (close ()) (f ()) (staged f));
      test
        "a step's transpose replays its runs: what a later run would read does \
         not reach the gradient" (fun () ->
          let runs = ref 0 in
          let l w =
            let step c x =
              incr runs;
              if !runs > 5 then (Nx.mul c w, c) else (Nx.sin (Nx.add c x), c)
            in
            Nx.sum (fst (Rune.scan' ~f:step ~init:(Nx.mul_s w 0.5) sxs))
          in
          let plain w =
            Nx.sum
              (fst
                 (Rune.scan'
                    ~f:(fun c x -> (Nx.sin (Nx.add c x), c))
                    ~init:(Nx.mul_s w 0.5) sxs))
          in
          equal (close ()) (Rune.grad' plain sw)
            (staged (fun () -> Rune.grad' l sw)));
      test
        "a remat's barrier under jvp of grad reaches the stager with no traced \
         value" (fun () ->
          let u0 = vec [| 0.2; 0.4; -0.3 |] in
          let g u =
            Rune.grad'
              (fun w -> Nx.sum (Nx.mul (Rune.remat one_tensor Nx.sin w) u))
              sw
          in
          late := 0;
          let d = staged (fun () -> snd (Rune.jvp' g u0 sv)) in
          equal (close ()) (snd (Rune.jvp' g u0 sv)) d;
          equal int 0 !late);
    ]

(* Iterates compiled: each application of a loss compiles its loops, the loss's
   eager value the trusted side. *)
let compiled_tests =
  List.concat_map
    (fun (loss, l) ->
      List.map
        (fun (app, a) ->
          test
            (app ^ " of a loss with " ^ loss ^ ", compiled, is eager's")
            (fun () ->
              equal
                (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ())
                ((List.assoc app applications) l)
                (a l)))
        compiled_applications)
    iterate_losses

let () =
  exit
    (run "Loops staged"
       [ group "on the host" staged_tests; group "compiled" compiled_tests ])
