(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scans under a stager on the host: each transformation installed between a
   compiled call's stager and a scan passes the scan on transformed, and the
   stager folds it. The stager's protocol is private to rune, so this suite
   links rune_internals. The trusted side is the same loss run eagerly. *)

open Windtrap
module Rune = Rune_internals.Rune

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let close () = Oracle.tensor ~rel:1e-12 ()

module Construct = Rune_internals.Construct

(* [staged f] is [f ()] under a stager that answers each scan by folding it
   where it answers it, as a compiled call stages one: every installation
   between passes the scan on transformed, and its step runs outside them.
   [steps] counts the step's runs and [late] the barriers that reached it with a
   traced value, one another installation should have passed on. *)
let steps = ref 0
let late = ref 0

let staged f =
  let traced =
    List.exists (fun (Nx.P x) ->
        match Nx.Repr.v x with Traced _ -> true | Host _ | Placed _ -> false)
  in
  let call : type r. r Construct.t -> (unit -> r) option =
   fun c ->
    match[@warning "@4@8"] c with
    | Scan r ->
        let req_step c x =
          incr steps;
          r.req_step c x
        in
        Some (fun () -> Rune_internals.Scan.fold { r with req_step })
    | Barrier { values; after } ->
        if traced values || traced after then incr late;
        Some (fun () -> values)
    | Compiled _ | Remat _ | Custom _ | Lanes _ | Lane_index _ | Lane_count _
    | Add _ | Detach _ ->
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

let staged_tests =
  List.concat_map
    (fun (loss, l) ->
      List.map
        (fun (app, a) ->
          test
            (app ^ " of a loss with " ^ loss ^ " is the folded scan's")
            (fun () -> equal (close ()) (a l) (staged (fun () -> a l))))
        applications)
    staged_losses
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
        "a step whose rerun reads a tracked value its first run did not raises \
         at the transpose" (fun () ->
          let runs = ref 0 in
          let l w =
            let step c x =
              incr runs;
              if !runs > 5 then (Nx.mul c w, c) else (Nx.sin (Nx.add c x), c)
            in
            Nx.sum (fst (Rune.scan' ~f:step ~init:(Nx.mul_s w 0.5) sxs))
          in
          raises
            (Invalid_argument
               "Rune.grad': a function run again for its transpose reads a \
                value the differentiation tracks that its first run did not")
            (fun () -> staged (fun () -> Rune.grad' l sw)));
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

let () = exit (run "Rune.scan staged" [ group "on the host" staged_tests ])
