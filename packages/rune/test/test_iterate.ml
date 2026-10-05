(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.iterate: a loop that tests a condition before each step, at most [max]
   steps. Under vmap each lane stops on its own; under the derivatives the
   derivative covers the steps each lane took. The trusted side is the plain
   OCaml loop, and the steps each lane took written out. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-10 ~abs:1e-12 ()
let holds b = Nx.item [] b
let lane i x = Nx.get [ i ] x
let lanes x = (Nx.shape x).(0)
let stack n f = Nx.stack (List.init n f)

let failure max =
  Printf.sprintf "Rune.iterate: until is still false after max = %d steps" max

(* [loop ~max ~until ~f x] is the loop iterate states: [until] tested before
   each step, at most [max] steps; [Ok (x, k)] after [k] steps, or [Error ()]
   when [until] still fails after [max]. *)
let loop ~max ~until ~f x =
  let rec go k x =
    if holds (until x) then Ok (x, k)
    else if k = max then Error ()
    else go (k + 1) (f x)
  in
  go 0 x

(* [written k f x] is [f] applied [k] times. *)
let rec written k f x = if k = 0 then x else written (k - 1) f (f x)

let trips ~max ~until ~f x =
  match loop ~max ~until ~f x with
  | Ok (_, k) -> k
  | Error () -> invalid_arg "trips: the loop does not end"

(* A contracting step: each element shrinks by at least a tenth, so a loop whose
   stop is a small maximum ends, after a number of steps that depends on the
   start. *)
let contract w x = Nx.mul_s (Nx.mul x (Nx.tanh (Nx.add_s (Nx.mul w x) 0.3))) 0.9
let small tol x = Nx.less_s (Nx.max (Nx.abs x)) tol

(* The loop *)

let program_step p x = Nx.mul_s (Expr.eval p x) 0.5

let loop_tests =
  [
    prop "iterate is the loop that tests until before each step"
      Gen.(
        triple Expr.gen (pair Expr.point (float_range 0.01 2.)) (int_range 0 8))
      (fun (p, (x0, tol), max) ->
        let until = small tol and f = program_step p in
        let got () = Rune.iterate' ~max ~until ~f x0 in
        match loop ~max ~until ~f x0 with
        | Ok (x, k) ->
            cover "no step" (k = 0);
            cover "some steps" (k > 0);
            cover "at the bound" (k = max && max > 0);
            equal (exact ()) x (got ())
        | Error () ->
            cover "raises" true;
            raises (Invalid_argument (failure max)) got);
    test "a start that satisfies until is returned unchanged" (fun () ->
        let x = vec [| 0.25; -0.5 |] in
        equal (exact ()) x
          (Rune.iterate' ~max:3 ~until:(small 1.) ~f:(fun _ -> assert false) x));
    test "max = 0 tests until once and steps never" (fun () ->
        raises
          (Invalid_argument (failure 0))
          (fun () ->
            Rune.iterate' ~max:0 ~until:(small 0.1) ~f:Fun.id (scalar 1.)));
    test "a structured carry with a budget in until" (fun () ->
        (* The method reports: it stops at its budget and says whether it
           converged. *)
        let budget = 4 in
        let x, k =
          Rune.iterate
            Nx.Ptree.(pair tensor tensor)
            ~max:budget
            ~until:(fun (x, k) ->
              Nx.logical_or (small 1e-3 x)
                (Nx.greater_equal_s k (Int32.of_int budget)))
            ~f:(fun (x, k) -> (Nx.mul_s x 0.5, Nx.add_s k 1l))
            (scalar 1., Nx.scalar Nx.int32 0l)
        in
        equal (exact ()) (scalar 0.0625) x;
        equal (exact ()) (Nx.scalar Nx.int32 4l) k);
    test "a step runs once per trip" (fun () ->
        let runs = ref 0 in
        let f x =
          incr runs;
          Nx.mul_s x 0.5
        in
        ignore (Rune.iterate' ~max:10 ~until:(small 0.1) ~f (scalar 1.));
        equal int 4 !runs);
  ]

(* Refusals *)

let refusal_tests =
  [
    test "a negative max is refused" (fun () ->
        raises (Invalid_argument "Rune.iterate: max = -1 is negative")
          (fun () ->
            Rune.iterate' ~max:(-1) ~until:(small 1.) ~f:Fun.id (scalar 0.)));
    test "until must return one boolean" (fun () ->
        raises
          (Invalid_argument
             "Rune.iterate: until must return one boolean, got shape [2]")
          (fun () ->
            Rune.iterate' ~max:3
              ~until:(fun x -> Nx.less_s x 1.)
              ~f:Fun.id
              (vec [| 0.; 2. |])));
    test "a step that changes the carry's shape is refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.iterate: the root: shape [3] in the carry the step \
              returned, [2] in the carry it received") (fun () ->
            Rune.iterate' ~max:3 ~until:(small 0.1)
              ~f:(fun x ->
                Nx.concatenate ~axis:0 [ x; Nx.slice [ R (0, 1) ] x ])
              (vec [| 1.; 2. |])));
    test "a step that changes the carry's visits is refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.iterate: the root: length 2 in the carry the step returned, \
              length 1 in the carry it received") (fun () ->
            Rune.iterate
              Nx.Ptree.(list tensor)
              ~max:3
              ~until:(fun l -> small 0.1 (List.hd l))
              ~f:(fun l -> l @ l)
              [ scalar 1. ]));
    test "under vmap until must return one boolean per lane" (fun () ->
        raises
          (Invalid_argument
             "Rune.iterate: until must return one boolean, got shape [2]")
          (fun () ->
            Rune.vmap'
              (Rune.iterate' ~max:3 ~until:(fun x -> Nx.less_s x 1.) ~f:Fun.id)
              (Nx.zeros f64 [| 3; 2 |])));
    test "under grad a step that changes the carry's shape is refused"
      (fun () ->
        raises
          (Invalid_argument
             "Rune.iterate: the root: shape [3] in the carry the step \
              returned, [2] in the carry it received") (fun () ->
            Rune.grad'
              (fun x ->
                Nx.sum
                  (Rune.iterate' ~max:3 ~until:(small 0.1)
                     ~f:(fun x ->
                       Nx.concatenate ~axis:0 [ x; Nx.slice [ R (0, 1) ] x ])
                     x))
              (vec [| 1.; 2. |])));
  ]

(* Lanes *)

let xs () = vec [| 0.3; 1.9; -1.4; 0.01; 1. |]

(* [per_lane f xs] is [f] on each lane of [xs], stacked. *)
let per_lane f xs = stack (lanes xs) (fun i -> f (lane i xs))
let tol = 0.05
let iterated w x = Rune.iterate' ~max:80 ~until:(small tol) ~f:(contract w) x
let w0 = scalar 0.7

let lane_tests =
  [
    test "each lane keeps the carry its own loop ends with" (fun () ->
        let xs = xs () in
        equal (exact ())
          (per_lane (iterated w0) xs)
          (Rune.vmap' (iterated w0) xs));
    test "the lanes take different numbers of trips, none for one" (fun () ->
        (* The other lane tests depend on it. *)
        let ks =
          List.init 5 (fun i ->
              trips ~max:80 ~until:(small tol) ~f:(contract w0) (lane i (xs ())))
        in
        equal int 0 (List.fold_left min max_int ks);
        at_least int ~than:3 (List.length (List.sort_uniq compare ks)));
    test "nested maps keep each lane's carry" (fun () ->
        let xs =
          Nx.reshape [| 2; 3 |]
            (Nx.slice
               [ R (0, 6) ]
               (Nx.concatenate ~axis:0 [ xs (); vec [| 0.7 |] ]))
        in
        equal (exact ())
          (per_lane (per_lane (iterated w0)) xs)
          (Rune.vmap' (Rune.vmap' (iterated w0)) xs));
    test "an iterate in an iterate's step keeps each lane's carry" (fun () ->
        let outer x =
          Rune.iterate' ~max:80 ~until:(small tol)
            ~f:(fun x -> Nx.mul_s (Nx.add x (iterated w0 x)) 0.4)
            x
        in
        equal (exact ()) (per_lane outer (xs ())) (Rune.vmap' outer (xs ())));
    test "under nested maps the error names the lanes outermost first"
      (fun () ->
        raises
          (Invalid_argument (failure 3 ^ ", in lane 1, in lane 2"))
          (fun () ->
            Rune.vmap'
              (Rune.vmap' (fun x ->
                   Rune.iterate' ~max:3 ~until:(small 0.5) ~f:Fun.id x))
              (Nx.create f64 [| 2; 3 |] [| 0.1; 0.1; 0.1; 0.1; 0.1; 2. |])));
    test "the error names the first lane still running" (fun () ->
        raises
          (Invalid_argument (failure 5 ^ ", in lane 1"))
          (fun () ->
            Rune.vmap'
              (fun x -> Rune.iterate' ~max:5 ~until:(small 0.5) ~f:Fun.id x)
              (vec [| 0.1; 2.; 3. |])));
    test "a stopped lane evaluates the step only where a lane reached"
      (fun () ->
        (* A step checks its own precondition: a held lane evaluated at its own
           carry would fail it. *)
        let f x =
          Nx.check (Nx.greater_s x 0.) (fun _ -> "the step left its domain");
          Nx.sub_s (Nx.sqrt x) 1.
        in
        let until x = Nx.less_equal_s x 0. in
        let x = vec [| 1.; 16. |] in
        equal (exact ())
          (per_lane (Rune.iterate' ~max:10 ~until ~f) x)
          (Rune.vmap' (Rune.iterate' ~max:10 ~until ~f) x));
    test "inside the step a stopped lane's lane_index is its donor's" (fun () ->
        (* Each carry holds its lane's index; a stopped lane runs its donor's
           carry, so the two indices agree on every lane at every trip. *)
        let step (x, id) =
          Nx.check
            (Nx.equal id (Rune.lane_index ()))
            (fun _ -> "the carry's lane is not the step's lane");
          (contract w0 x, id)
        in
        let run x id =
          Rune.iterate
            Nx.Ptree.(pair tensor tensor)
            ~max:80
            ~until:(fun (x, _) -> small tol x)
            ~f:step (x, id)
        in
        let xs = xs () in
        let ys, ids =
          Rune.vmap
            Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
            run xs (Nx.arange Nx.int32 0 5 1)
        in
        equal (exact ()) (per_lane (iterated w0) xs) ys;
        equal (exact ()) (Nx.arange Nx.int32 0 5 1) ids);
  ]

(* Totals *)

let count : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let counted x =
  Rune.iterate' ~max:80 ~until:(small tol)
    ~f:(fun x ->
      Rune.Total.add count (scalar 1.);
      contract w0 x)
    x

let trips_of xs =
  List.fold_left ( +. ) 0.
    (List.init (lanes xs) (fun i ->
         Float.of_int
           (trips ~max:80 ~until:(small tol) ~f:(contract w0) (lane i xs))))

let total_tests =
  [
    test "a restarted attempt adds nothing to a scope around the map" (fun () ->
        (* The carry starts untracked and gains the tracked [w] at the first
           step, which restarts grad's loop under the map's fold. *)
        let xs = xs () in
        let loss w x =
          Nx.sum
            (Rune.iterate' ~max:80 ~until:(small tol)
               ~f:(fun y ->
                 Rune.Total.add count (scalar 1.);
                 contract w y)
               (Nx.add_s (Nx.mul_s x 0.5) 1.))
        in
        let expected =
          List.fold_left ( +. ) 0.
            (List.init (lanes xs) (fun i ->
                 Float.of_int
                   (trips ~max:80 ~until:(small tol) ~f:(contract w0)
                      (Nx.add_s (Nx.mul_s (lane i xs) 0.5) 1.))))
        in
        let _, n =
          Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
              Rune.vmap' (fun x -> Rune.grad' (fun w -> loss w x) w0) xs)
        in
        equal (exact ()) (scalar expected) n);
    test "a scope around the map counts each lane's trips" (fun () ->
        let xs = xs () in
        let _, n =
          Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
              Rune.vmap' counted xs)
        in
        equal (exact ()) (scalar (trips_of xs)) n);
    test "a scope inside the map counts its lane's trips" (fun () ->
        let xs = xs () in
        let n =
          Rune.vmap'
            (fun x ->
              snd
                (Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
                     counted x)))
            xs
        in
        equal (exact ())
          (per_lane
             (fun x ->
               scalar
                 (Float.of_int
                    (trips ~max:80 ~until:(small tol) ~f:(contract w0) x)))
             xs)
          n);
    test "a scope around grad of the map counts each lane's trips once"
      (fun () ->
        let xs = xs () in
        let _, n =
          Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
              Rune.grad' (fun xs -> Nx.sum (Rune.vmap' counted (Nx.sin xs))) xs)
        in
        equal (exact ()) (scalar (trips_of (Nx.sin xs))) n);
  ]

(* Collectives *)

let collective_tests =
  let a = Rune.axis () in
  [
    test "lanes of the map raises in a step whose lanes stop apart" (fun () ->
        raises
          (Invalid_argument
             "Rune.lanes: a lane of Rune.iterate that stopped takes no more \
              trips, so a step cannot gather every lane; give every lane the \
              same stop") (fun () ->
            Rune.vmap' ~axis:a
              (fun x ->
                Rune.iterate' ~max:80 ~until:(small tol)
                  ~f:(fun x ->
                    Nx.add (contract w0 x)
                      (Nx.mul_s (Nx.sum (Rune.lanes a x)) 0.))
                  x)
              (xs ())));
    test "a stop every lane shares admits lanes" (fun () ->
        (* The stop reads a counter no lane changes, so no lane is held. *)
        let run x =
          fst
            (Rune.iterate
               Nx.Ptree.(pair tensor tensor)
               ~max:5
               ~until:(fun (_, k) -> Nx.greater_equal_s k 3l)
               ~f:(fun (x, k) ->
                 (Nx.add x (Nx.sum (Rune.lanes a x)), Nx.add_s k 1l))
               (x, Nx.scalar Nx.int32 0l))
        in
        let x = vec [| 1.; 2. |] in
        (* Each trip adds the sum of both lanes: 3, then 9, then 27. *)
        equal (exact ()) (vec [| 40.; 41. |]) (Rune.vmap' ~axis:a run x));
  ]

(* Derivatives over the trips taken *)

let loss w x = Nx.sum (Nx.sin (iterated w x))

(* [written_loss w x] is [loss w x] with its loop written out at the number of
   steps it takes from [x]. *)
let written_loss w x =
  let k = trips ~max:80 ~until:(small tol) ~f:(contract w0) x in
  Nx.sum (Nx.sin (written k (contract w) x))

let v () = vec [| 0.4; -1.1; 0.6; 2.; -0.3 |]

(* Each application of a transformation to a loss of one lane, batched, and the
   same application to the lane's loss written out, lane by lane. *)
let batched =
  let one = Nx.Ptree.tensor in
  [
    ( "vmap of grad",
      (fun l xs _ -> Rune.vmap' (Rune.grad' (l w0)) xs),
      fun l xs _ -> per_lane (Rune.grad' (l w0)) xs );
    ( "grad of vmap",
      (fun l xs _ -> Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (l w0) xs)) xs),
      fun l xs _ -> per_lane (Rune.grad' (l w0)) xs );
    ( "vmap of jvp",
      (fun l xs vs ->
        Rune.vmap
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          (fun x v -> snd (Rune.jvp' (l w0) x v))
          xs vs),
      fun l xs vs ->
        stack (lanes xs) (fun i ->
            snd (Rune.jvp' (l w0) (lane i xs) (lane i vs))) );
    ( "jvp of vmap",
      (fun l xs vs -> snd (Rune.jvp' (Rune.vmap' (l w0)) xs vs)),
      fun l xs vs ->
        stack (lanes xs) (fun i ->
            snd (Rune.jvp' (l w0) (lane i xs) (lane i vs))) );
    ( "vmap of grad of grad",
      (fun l xs _ ->
        Rune.vmap'
          (Rune.grad' (fun x -> Nx.sum (Nx.mul_s (Rune.grad' (l w0) x) 1.5)))
          xs),
      fun l xs _ ->
        per_lane
          (Rune.grad' (fun x -> Nx.sum (Nx.mul_s (Rune.grad' (l w0) x) 1.5)))
          xs );
    ( "vmap of jvp of grad",
      (fun l xs vs ->
        Rune.vmap
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          (fun x v -> snd (Rune.jvp' (Rune.grad' (l w0)) x v))
          xs vs),
      fun l xs vs ->
        stack (lanes xs) (fun i ->
            snd (Rune.jvp' (Rune.grad' (l w0)) (lane i xs) (lane i vs))) );
    ( "grad of a captured parameter through vmap",
      (fun l xs _ -> Rune.grad' (fun w -> Nx.sum (Rune.vmap' (l w) xs)) w0),
      fun l xs _ ->
        Rune.grad'
          (fun w -> Nx.sum (stack (lanes xs) (fun i -> l w (lane i xs))))
          w0 );
    ( "vmap of vjp",
      (fun l xs vs ->
        Rune.vmap
          Nx.Ptree.(tensor @-> tensor @-> returns tensor)
          (fun x v ->
            let _, pb = Rune.vjp one one (l w0) x in
            pb (Nx.sum v))
          xs vs),
      fun l xs vs ->
        stack (lanes xs) (fun i ->
            let _, pb = Rune.vjp one one (l w0) (lane i xs) in
            pb (Nx.sum (lane i vs))) );
  ]

let derivative_tests =
  List.map
    (fun (name, batched, written) ->
      test (name ^ " covers each lane's own trips") (fun () ->
          let xs = xs () and vs = v () in
          equal (close ()) (written written_loss xs vs) (batched loss xs vs)))
    batched
  @ [
      test "grad covers the trips taken" (fun () ->
          let x = scalar 1.9 in
          equal (close ())
            (Rune.grad' (written_loss w0) x)
            (Rune.grad' (loss w0) x));
      test "jvp covers the trips taken" (fun () ->
          let x = scalar 1.9 in
          equal (close ())
            (snd (Rune.jvp' (written_loss w0) x (scalar 0.3)))
            (snd (Rune.jvp' (loss w0) x (scalar 0.3))));
      prop "vmap of grad over generated steps covers each lane's trips"
        Gen.(pair Expr.smooth (Expr.points 3))
        (fun (p, xs) ->
          let f x = Nx.mul_s (Nx.mul x (Nx.tanh (Expr.eval p x))) 0.9 in
          let until = small 0.1 in
          let l x = Nx.sum (Nx.sin (Rune.iterate' ~max:200 ~until ~f x)) in
          let written x =
            let k = trips ~max:200 ~until ~f x in
            Nx.sum (Nx.sin (written k f x))
          in
          let ks =
            List.init 3 (fun i -> trips ~max:200 ~until ~f (lane i xs))
          in
          cover "lanes stop apart" (List.length (List.sort_uniq compare ks) > 1);
          equal (close ())
            (per_lane (Rune.grad' written) xs)
            (Rune.vmap' (Rune.grad' l) xs));
    ]

(* Nested loops *)

(* An iterate in an iterate's step: the outer step averages its carry with an
   inner loop's limit from it, so both loops take a number of trips that depends
   on the start. *)
let inner_tol = 0.02
let outer_tol = 0.1
let inner w x = Rune.iterate' ~max:80 ~until:(small inner_tol) ~f:(contract w) x
let outer_step w x = Nx.mul_s (Nx.add x (inner w (Nx.mul_s x 1.5))) 0.5

let nested w x =
  Rune.iterate' ~max:80 ~until:(small outer_tol) ~f:(outer_step w) x

let nested_loss w x = Nx.sum (Nx.sin (nested w x))

(* [schedule w x] is the number of trips the inner loop takes at each trip of
   the outer one, from [x]. *)
let schedule w x =
  let inner_trips x =
    trips ~max:80 ~until:(small inner_tol) ~f:(contract w) (Nx.mul_s x 1.5)
  in
  let rec go x acc =
    if holds (small outer_tol x) then List.rev acc
    else go (outer_step w x) (inner_trips x :: acc)
  in
  go x []

(* [written_nested ks w x] is the nested loop written out at the schedule
   [ks]. *)
let written_nested ks w x =
  let step x k =
    Nx.mul_s (Nx.add x (written k (contract w) (Nx.mul_s x 1.5))) 0.5
  in
  Nx.sum (Nx.sin (List.fold_left step x ks))

(* The nested loops with each lane's index in both carries, and an addition of
   one per inner trip: a lane the outer loop stopped runs the outer step on its
   donor's carry, and the inner loop must hold it too. *)
let counted_nested w xs =
  let inner (y, id) =
    Rune.iterate
      Nx.Ptree.(pair tensor tensor)
      ~max:80
      ~until:(fun (y, _) -> small inner_tol y)
      ~f:(fun (y, id) ->
        Nx.check
          (Nx.equal id (Rune.lane_index ()))
          (fun _ -> "the inner carry's lane is not the step's lane");
        Rune.Total.add count (scalar 1.);
        (contract w y, id))
      (y, id)
  in
  let outer (x, id) =
    let y, _ = inner (Nx.mul_s x 1.5, id) in
    (Nx.mul_s (Nx.add x y) 0.5, id)
  in
  let run x id =
    fst
      (Rune.iterate
         Nx.Ptree.(pair tensor tensor)
         ~max:80
         ~until:(fun (x, _) -> small outer_tol x)
         ~f:outer (x, id))
  in
  Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
      Rune.vmap
        Nx.Ptree.(tensor @-> tensor @-> returns tensor)
        run xs
        (Nx.arange Nx.int32 0 (Nx.dim 0 xs) 1))

let nested_hold_law =
  prop ~count:30 "a lane the outer loop stopped is held in the inner loop"
    Gen.(pair (float_range 0.3 1.2) (Expr.points 3))
    (fun (w, xs) ->
      let w = scalar w in
      let xs = Nx.reshape [| 3; 6 |] xs in
      let ks = List.init 3 (fun i -> schedule w (lane i xs)) in
      cover "outer lanes stop apart"
        (List.length (List.sort_uniq compare (List.map List.length ks)) > 1);
      let ys, n = counted_nested w xs in
      equal ~msg:"carries" (exact ())
        (stack 3 (fun i -> nested w (lane i xs)))
        ys;
      equal ~msg:"inner trips" (exact ())
        (scalar (Float.of_int (List.fold_left ( + ) 0 (List.concat ks))))
        n)

let nested_law =
  prop ~count:40 "an iterate in an iterate's step is the loops written out"
    Gen.(pair (float_range 0.3 1.2) (Expr.points 3))
    (fun (w, xs) ->
      let w = scalar w in
      let xs = Nx.reshape [| 3; 6 |] xs in
      let ks = List.init 3 (fun i -> schedule w (lane i xs)) in
      cover "outer lanes stop apart"
        (List.length (List.sort_uniq compare (List.map List.length ks)) > 1);
      cover "inner loops stop apart"
        (List.exists (fun k -> List.length (List.sort_uniq compare k) > 1) ks);
      let written f = stack 3 (fun i -> f (List.nth ks i) (lane i xs)) in
      let vs = Nx.cos (Nx.mul_s xs 3.) in
      equal ~msg:"value" (exact ())
        (stack 3 (fun i -> nested w (lane i xs)))
        (Rune.vmap' (nested w) xs);
      let grads = written (fun k -> Rune.grad' (written_nested k w)) in
      equal ~msg:"vmap of grad" (close ()) grads
        (Rune.vmap' (Rune.grad' (nested_loss w)) xs);
      equal ~msg:"grad of vmap" (close ()) grads
        (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (nested_loss w) xs)) xs);
      equal ~msg:"jvp of vmap" (close ())
        (Nx.sum ~axes:[ 1 ] (Nx.mul grads vs))
        (snd (Rune.jvp' (Rune.vmap' (nested_loss w)) xs vs));
      equal ~msg:"grad of a captured parameter" (close ())
        (Rune.grad'
           (fun w ->
             Nx.sum
               (stack 3 (fun i -> written_nested (List.nth ks i) w (lane i xs))))
           w)
        (Rune.grad' (fun w -> Nx.sum (Rune.vmap' (nested_loss w) xs)) w);
      (* The schedule holds near each start, so the written-out loss is smooth
         there. *)
      List.iteri
        (fun i k ->
          let x = lane i xs and v = lane i vs in
          equal ~msg:"finite difference"
            (Oracle.tensor ~rel:1e-6 ~abs:1e-8 ())
            (Oracle.central ~eps:1e-6 (written_nested k w) x v)
            (scalar (Oracle.dot (lane i grads) v)))
        ks)

(* Stopped lanes hold *)

(* [x ↦ √x − 1] has an infinite derivative at 0, where the first lane stops
   after one step while the second takes three more. *)
let root_step x = Nx.sub_s (Nx.sqrt x) 1.
let nonpositive x = Nx.less_equal_s x 0.
let rooted x = Rune.iterate' ~max:10 ~until:nonpositive ~f:root_step x
let starts () = vec [| 1.; 16. |]

let written_rooted x =
  written (trips ~max:10 ~until:nonpositive ~f:root_step x) root_step x

let hold_tests =
  [
    test "a stopped lane keeps its carry bit for bit" (fun () ->
        equal (exact ())
          (per_lane rooted (starts ()))
          (Rune.vmap' rooted (starts ())));
    test "a stopped lane's gradient is its own trips' and finite" (fun () ->
        let g = Rune.vmap' (Rune.grad' rooted) (starts ()) in
        equal (close ()) (per_lane (Rune.grad' written_rooted) (starts ())) g;
        equal (exact ()) (vec [| 0.5 |]) (Nx.slice [ R (0, 1) ] g));
    test "grad of the map is finite where every step taken is" (fun () ->
        let g =
          Rune.grad' (fun x -> Nx.sum (Rune.vmap' rooted x)) (starts ())
        in
        equal (close ()) (per_lane (Rune.grad' written_rooted) (starts ())) g);
    test "a stopped lane's tangent is its own trips'" (fun () ->
        let t =
          snd (Rune.jvp' (Rune.vmap' rooted) (starts ()) (vec [| 1.; 1. |]))
        in
        equal (close ()) (per_lane (Rune.grad' written_rooted) (starts ())) t);
  ]

(* The loop written in OCaml *)

(* [plain ~max ~until ~f x] is the loop iterate states, written in OCaml: grad
   and jvp differentiate the steps it takes. *)
let plain ~max ~until ~f x =
  match loop ~max ~until ~f x with
  | Ok (x, _) -> x
  | Error () -> invalid_arg (failure max)

(* [plain_iterate] is [plain] at {!Rune.iterate}'s arguments. *)
let plain_iterate _ ~max ~until ~f x = plain ~max ~until ~f x
let lane_failure max i = failure max ^ Printf.sprintf ", in lane %d" i

let starts =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 4) (float_range (-2.) 2.)))

(* Bounds *)

(* [outcomes ~max xs] is each lane's own loop from [xs]. *)
let outcomes ~max xs =
  List.init (lanes xs) (fun i ->
      loop ~max ~until:(small tol) ~f:(contract w0) (lane i xs))

let at_bound ~max ks =
  List.exists
    (function Ok (_, k) -> max > 0 && k = max | Error () -> false)
    ks

let bound_tests =
  [
    prop "under vmap each lane ends within max, or the first running is named"
      Gen.(pair starts (int_range 0 5))
      (fun (xs, max) ->
        let ks = outcomes ~max xs in
        let got () =
          Rune.vmap' (Rune.iterate' ~max ~until:(small tol) ~f:(contract w0)) xs
        in
        cover "max = 0" (max = 0);
        cover "max = 1" (max = 1);
        match List.find_index Result.is_error ks with
        | Some i ->
            cover "a lane still runs at max" true;
            raises (Invalid_argument (lane_failure max i)) got
        | None ->
            cover "a lane stops at max" (at_bound ~max ks);
            equal (exact ())
              (stack (lanes xs) (fun i -> fst (Result.get_ok (List.nth ks i))))
              (got ()));
    prop
      "under the derivatives each lane ends within max, or the first running \
       is named"
      Gen.(pair starts (int_range 0 5))
      (fun (xs, max) ->
        let until = small tol and f = contract w0 in
        let loss x = Nx.sum (Nx.sin (Rune.iterate' ~max ~until ~f x)) in
        let ks = outcomes ~max xs in
        let applications =
          [
            ("vmap of grad", fun () -> Rune.vmap' (Rune.grad' loss) xs);
            ( "grad of vmap",
              fun () -> Rune.grad' (fun xs -> Nx.sum (Rune.vmap' loss xs)) xs );
            ( "jvp of vmap",
              fun () -> snd (Rune.jvp' (Rune.vmap' loss) xs (Nx.ones_like xs))
            );
          ]
        in
        match List.find_index Result.is_error ks with
        | Some i ->
            cover "a lane still runs at max" true;
            List.iter
              (fun (msg, a) ->
                raises ~msg (Invalid_argument (lane_failure max i)) a)
              applications
        | None ->
            cover "a lane stops at max" (at_bound ~max ks);
            let expected =
              per_lane
                (Rune.grad' (fun x -> Nx.sum (Nx.sin (plain ~max ~until ~f x))))
                xs
            in
            List.iter
              (fun (msg, a) -> equal ~msg (close ()) expected (a ()))
              applications);
    test "max = 1 takes the one step until needs" (fun () ->
        equal (exact ()) (scalar 0.5)
          (Rune.iterate' ~max:1
             ~until:(fun x -> Nx.less_s x 0.75)
             ~f:(fun x -> Nx.mul_s x 0.5)
             (scalar 1.)));
    test "max = 1 raises when one step is not enough" (fun () ->
        raises
          (Invalid_argument (failure 1))
          (fun () ->
            Rune.iterate' ~max:1
              ~until:(fun x -> Nx.less_s x 0.3)
              ~f:(fun x -> Nx.mul_s x 0.5)
              (scalar 1.)));
  ]

(* A start that satisfies until *)

let unstepped =
  let f _ = failwith "the step ran" in
  let x = vec [| 0.01; -0.02; -0. |] in
  let id max x = Rune.iterate' ~max ~until:(small tol) ~f x in
  let loss max x = Nx.sum (Nx.sin (id max x)) in
  let xs = Nx.reshape [| 3; 1 |] x in
  let applications max =
    [
      ("the value", fun () -> (x, id max x));
      ("grad", fun () -> (Nx.cos x, Rune.grad' (loss max) x));
      ("jvp", fun () -> (Nx.cos x, snd (Rune.jvp' (id max) x (Nx.cos x))));
      ("vmap", fun () -> (xs, Rune.vmap' (id max) xs));
      ( "vmap of grad",
        fun () -> (Nx.cos xs, Rune.vmap' (Rune.grad' (loss max)) xs) );
      ( "grad of vmap",
        fun () ->
          ( Nx.cos xs,
            Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (loss max) xs)) xs ) );
    ]
  in
  List.concat_map
    (fun max ->
      List.map
        (fun (name, a) ->
          test
            (Printf.sprintf
               "%s of a start that satisfies until, max = %d, takes no step"
               name max) (fun () ->
              let expected, got = a () in
              equal (exact ()) expected got))
        (applications max))
    [ 0; 3 ]

(* Non-finite carries *)

(* Lanes that stop at their start on a carry that is not finite, beside lanes
   that run. *)
let settled x = Nx.logical_or (Nx.logical_not (Nx.isfinite x)) (small tol x)

let nonfinite () =
  vec [| Float.nan; 1.9; Float.infinity; -0.; -1.4; Float.neg_infinity |]

let settle w x = Rune.iterate' ~max:80 ~until:settled ~f:(contract w) x
let plain_settle w x = plain ~max:80 ~until:settled ~f:(contract w) x

(* A loss whose derivative is finite at every carry: the select comes before the
   sine. *)
let finite_loss y =
  Nx.sum (Nx.sin (Nx.where (Nx.isfinite y) y (Nx.zeros_like y)))

let nonfinite_tests =
  [
    test "a stopped lane keeps a NaN or infinite carry bit for bit" (fun () ->
        let xs = nonfinite () in
        equal (exact ())
          (per_lane (plain_settle w0) xs)
          (Rune.vmap' (settle w0) xs));
    test "vmap of grad gives a lane stopped on NaN a zero" (fun () ->
        let xs = nonfinite () in
        equal (close ())
          (per_lane (Rune.grad' (fun x -> finite_loss (plain_settle w0 x))) xs)
          (Rune.vmap' (Rune.grad' (fun x -> finite_loss (settle w0 x))) xs));
    test "a captured parameter's gradient stays finite beside NaN lanes"
      (fun () ->
        let xs = nonfinite () in
        equal (close ())
          (Rune.grad'
             (fun w ->
               Nx.sum
                 (stack (lanes xs) (fun i ->
                      finite_loss (plain_settle w (lane i xs)))))
             w0)
          (Rune.grad'
             (fun w ->
               Nx.sum (Rune.vmap' (fun x -> finite_loss (settle w x)) xs))
             w0));
    test "a stopped lane's NaN outside the carry reaches no gradient" (fun () ->
        (* Each lane's step reads its own [w]; the lanes whose [w] is not finite
           stop at their start. *)
        let ws = vec [| Float.nan; 0.7; Float.infinity; 0.5; Float.nan |] in
        let xs = vec [| 0.01; 1.9; -0.02; -1.4; 0. |] in
        let pair = Nx.Ptree.(pair tensor tensor) in
        let loss iterate x w =
          Nx.sin (iterate ~max:80 ~until:(small tol) ~f:(contract w) x)
        in
        let gx, gw =
          Rune.grad pair
            (fun (xs, ws) ->
              Nx.sum
                (Rune.vmap
                   Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                   (loss Rune.iterate') xs ws))
            (xs, ws)
        in
        let per_lane_grad i =
          Rune.grad pair (fun (x, w) -> loss plain x w) (lane i xs, lane i ws)
        in
        equal ~msg:"x" (close ()) (stack 5 (fun i -> fst (per_lane_grad i))) gx;
        equal ~msg:"w" (close ()) (stack 5 (fun i -> snd (per_lane_grad i))) gw);
    test "jvp of vmap gives a lane stopped on NaN its tangent" (fun () ->
        let xs = nonfinite () in
        let vs = vec [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        equal (close ())
          (stack (lanes xs) (fun i ->
               snd (Rune.jvp' (plain_settle w0) (lane i xs) (lane i vs))))
          (snd (Rune.jvp' (Rune.vmap' (settle w0)) xs vs)));
  ]

(* Stopped lanes hold, over generated starts *)

let carried : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

(* Each trip checks that the step runs as the lane its carry names, and adds its
   carry to [carried]. *)
let checked_step (x, id) =
  Nx.check
    (Nx.equal id (Rune.lane_index ()))
    (fun _ -> "the carry's lane is not the step's lane");
  Rune.Total.add carried x;
  (contract w0 x, id)

let checked x id =
  fst
    (Rune.iterate
       Nx.Ptree.(pair tensor tensor)
       ~max:80
       ~until:(fun (x, _) -> small tol x)
       ~f:checked_step (x, id))

(* [carries x] is the sum of the carries each trip of [x]'s own loop starts
   from. *)
let carries x =
  let rec go x acc =
    if holds (small tol x) then acc else go (contract w0 x) (Nx.add acc x)
  in
  go x (scalar 0.)

let hold_law =
  prop "a stopped lane's lane_index is its donor's and its additions drop"
    Gen.(
      map
        (fun l -> vec (Array.of_list l))
        (list ~size:(int_range 2 5) (float_range (-2.) 2.)))
    (fun xs ->
      let n = lanes xs in
      let ks =
        List.init n (fun i ->
            trips ~max:80 ~until:(small tol) ~f:(contract w0) (lane i xs))
      in
      cover "lanes stop apart" (List.length (List.sort_uniq compare ks) > 1);
      let ids = Nx.arange Nx.int32 0 n 1 in
      let mapped =
        Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns tensor) checked
      in
      let expected = Nx.sum (per_lane carries xs) in
      let ys, total =
        Rune.Total.collect carried ~zero:(scalar 0.) (fun () -> mapped xs ids)
      in
      equal ~msg:"carry" (exact ()) (per_lane (iterated w0) xs) ys;
      equal ~msg:"total" (close ()) expected total;
      let g, total =
        Rune.Total.collect carried ~zero:(scalar 0.) (fun () ->
            Rune.grad' (fun xs -> Nx.sum (Nx.sin (mapped xs ids))) xs)
      in
      equal ~msg:"grad" (close ())
        (per_lane (Rune.grad' (fun x -> Nx.sum (Nx.sin (iterated w0 x)))) xs)
        g;
      equal ~msg:"total under grad" (close ()) expected total)

(* Dtypes *)

(* [flips x t] is [x] flipped [t] times: every trip moves each element of a row
   that is not a palindrome, so a lane that takes one trip too many or too few
   shows. *)
let flips x t =
  fst
    (Rune.iterate
       Nx.Ptree.(pair tensor tensor)
       ~max:3
       ~until:(fun (_, k) -> Nx.greater_equal k t)
       ~f:(fun (x, k) -> (Nx.flip x, Nx.add_s k 1l))
       (x, Nx.scalar Nx.int32 0l))

let held_dtype name x =
  test (name ^ " carries hold each lane's bits") (fun () ->
      let t = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 2l; 3l |] in
      let expected =
        stack 4 (fun i ->
            let r = lane i x in
            if i mod 2 = 1 then Nx.flip r else r)
      in
      equal (exact ()) expected
        (Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns tensor) flips x t))

let rows dt a = Nx.create dt [| 4; 3 |] a

let floats =
  Float.
    [|
      nan;
      -0.;
      infinity;
      1.5;
      2.;
      -3.;
      neg_infinity;
      0.;
      4.;
      5e-324;
      max_float;
      -1.;
    |]

let ints lo hi = [| lo; -1; 0; 1; 2; hi; 3; 0; lo; hi; 1; 2 |]
let uints hi = [| 0; 1; hi; hi; 0; 2; 3; 4; 1; 5; 0; hi |]

let bools =
  [|
    true; false; false; false; true; true; true; true; false; false; false; true;
  |]

let complexes =
  Array.mapi (fun i re -> { Complex.re; im = Float.of_int (i - 5) }) floats

let dtype_tests =
  [
    held_dtype "float64" (rows Nx.float64 floats);
    held_dtype "float32" (rows Nx.float32 floats);
    held_dtype "float16" (rows Nx.float16 floats);
    held_dtype "bfloat16" (rows Nx.bfloat16 floats);
    held_dtype "float8_e4m3" (rows Nx.float8_e4m3 floats);
    held_dtype "float8_e5m2" (rows Nx.float8_e5m2 floats);
    held_dtype "int4" (rows Nx.int4 (ints (-8) 7));
    held_dtype "uint4" (rows Nx.uint4 (uints 15));
    held_dtype "int8" (rows Nx.int8 (ints (-128) 127));
    held_dtype "uint8" (rows Nx.uint8 (uints 255));
    held_dtype "int16" (rows Nx.int16 (ints (-32768) 32767));
    held_dtype "uint16" (rows Nx.uint16 (uints 65535));
    held_dtype "int32"
      (rows Nx.int32
         Int32.
           [|
             min_int; -1l; 0l; 1l; 2l; max_int; 3l; 0l; min_int; max_int; 1l; 2l;
           |]);
    held_dtype "uint32" (rows Nx.uint32 (Array.map Int32.of_int (uints (-1))));
    held_dtype "int64"
      (rows Nx.int64
         Int64.
           [|
             min_int; -1L; 0L; 1L; 2L; max_int; 3L; 0L; min_int; max_int; 1L; 2L;
           |]);
    held_dtype "uint64" (rows Nx.uint64 (Array.map Int64.of_int (uints (-1))));
    held_dtype "complex64" (rows Nx.complex64 complexes);
    held_dtype "complex128" (rows Nx.complex128 complexes);
    held_dtype "bool" (rows Nx.bool bools);
    held_dtype "bit" (rows Nx.bit bools);
    test "an integer carry has a zero gradient and a zero tangent" (fun () ->
        let pair = Nx.Ptree.(pair tensor tensor) in
        let run iterate (x, k) =
          iterate pair ~max:80
            ~until:(fun (x, _) -> small tol x)
            ~f:(fun (x, k) -> (contract w0 x, Nx.add_s k 1l))
            (x, k)
        in
        let loss iterate xk =
          let x, k = run iterate xk in
          Nx.mul (Nx.sum (Nx.sin x)) (Nx.cast f64 k)
        in
        let xk = (scalar 1.9, Nx.scalar Nx.int32 5l) in
        let gx, gk = Rune.grad pair (loss Rune.iterate) xk in
        let gx', _ = Rune.grad pair (loss plain_iterate) xk in
        equal ~msg:"float gradient" (close ()) gx' gx;
        equal ~msg:"integer gradient" (exact ()) (Nx.scalar Nx.int32 0l) gk;
        let dxk = (scalar 1., Nx.scalar Nx.int32 7l) in
        let (_, k), (dx, dk) = Rune.jvp pair pair (run Rune.iterate) xk dxk in
        let (_, k'), (dx', _) = Rune.jvp pair pair (run plain_iterate) xk dxk in
        equal ~msg:"integer value" (exact ()) k' k;
        equal ~msg:"float tangent" (close ()) dx' dx;
        equal ~msg:"integer tangent" (exact ()) (Nx.scalar Nx.int32 0l) dk);
    test "vmap of grad over an integer carry gives each lane its own trips"
      (fun () ->
        let loss iterate x =
          let x, k =
            iterate
              Nx.Ptree.(pair tensor tensor)
              ~max:80
              ~until:(fun (x, _) -> small tol x)
              ~f:(fun (x, k) -> (contract w0 x, Nx.add_s k 1l))
              (x, Nx.scalar Nx.int32 0l)
          in
          Nx.mul (Nx.sum (Nx.sin x)) (Nx.cast f64 k)
        in
        equal (close ())
          (per_lane (Rune.grad' (loss plain_iterate)) (xs ()))
          (Rune.vmap' (Rune.grad' (loss Rune.iterate)) (xs ())));
  ]

(* Complex carries *)

let cw = Nx.create Nx.complex128 [||] [| { Complex.re = 0.3; im = -0.8 } |]
let c0 = Nx.create Nx.complex128 [||] [| { Complex.re = 0.6; im = 0.5 } |]
let csmall z = Nx.less_s (Nx.max (Nx.real f64 (Nx.mul z (Nx.conjugate z)))) 1e-3
let crun iterate c z = iterate ~max:80 ~until:csmall ~f:(fun z -> Nx.mul z c) z
let closs iterate c z = Nx.sum (Nx.real f64 (Nx.mul (crun iterate c z) cw))

let zs () =
  Nx.create Nx.complex128 [| 4 |]
    Complex.
      [|
        { re = 0.01; im = 0. };
        { re = 1.; im = -1. };
        { re = -0.2; im = 0.3 };
        { re = 2.; im = 0.5 };
      |]

let complex_tests =
  [
    test "vmap of grad over a complex carry covers each lane's trips" (fun () ->
        equal (close ())
          (per_lane (Rune.grad' (closs plain c0)) (zs ()))
          (Rune.vmap' (Rune.grad' (closs Rune.iterate' c0)) (zs ())));
    test "grad of a captured complex parameter through vmap" (fun () ->
        let zs = zs () in
        equal (close ())
          (Rune.grad'
             (fun c ->
               Nx.sum (stack (lanes zs) (fun i -> closs plain c (lane i zs))))
             c0)
          (Rune.grad'
             (fun c -> Nx.sum (Rune.vmap' (closs Rune.iterate' c) zs))
             c0));
    test "jvp of vmap over a complex carry" (fun () ->
        let zs = zs () in
        let v = Nx.conjugate zs in
        equal (close ())
          (stack (lanes zs) (fun i ->
               snd (Rune.jvp' (crun plain c0) (lane i zs) (lane i v))))
          (snd (Rune.jvp' (Rune.vmap' (crun Rune.iterate' c0)) zs v)));
  ]

(* Structures and sizes *)

let structure_tests =
  let s = Nx.Ptree.(pair (list tensor) (option tensor)) in
  let run iterate o x =
    let l, o =
      iterate s ~max:80
        ~until:(fun (l, _) -> small tol (List.hd l))
        ~f:(fun (l, o) ->
          (List.map (contract w0) l, Option.map (fun y -> Nx.mul_s y 0.5) o))
        ([ x; Nx.mul_s x 0.5 ], Option.map (fun f -> f x) o)
    in
    Nx.stack (l @ Option.to_list o)
  in
  let some = Some (fun x -> Nx.add_s x 1.) in
  [
    test "a carry of a list and an option holds each lane's" (fun () ->
        List.iter
          (fun (msg, o) ->
            equal ~msg (exact ())
              (per_lane (run plain_iterate o) (xs ()))
              (Rune.vmap' (run Rune.iterate o) (xs ())))
          [ ("Some", some); ("None", None) ]);
    test "grad of vmap over a structured carry" (fun () ->
        let loss iterate x = Nx.sum (Nx.sin (run iterate some x)) in
        equal (close ())
          (per_lane (Rune.grad' (loss plain_iterate)) (xs ()))
          (Rune.grad'
             (fun xs -> Nx.sum (Rune.vmap' (loss Rune.iterate) xs))
             (xs ())));
    test "a counter every lane starts from counts each lane's own trips"
      (fun () ->
        let run x =
          Rune.iterate
            Nx.Ptree.(pair tensor tensor)
            ~max:80
            ~until:(fun (x, _) -> small tol x)
            ~f:(fun (x, k) -> (contract w0 x, Nx.add_s k 1l))
            (x, Nx.scalar Nx.int32 0l)
        in
        let xs = xs () in
        let _, ks =
          Rune.vmap Nx.Ptree.(tensor @-> returns (pair tensor tensor)) run xs
        in
        equal (exact ())
          (Nx.create Nx.int32 [| 5 |]
             (Array.init 5 (fun i ->
                  Int32.of_int
                    (trips ~max:80 ~until:(small tol) ~f:(contract w0)
                       (lane i xs)))))
          ks);
    test "a map of no lanes" (fun () ->
        let none = Nx.zeros f64 [| 0 |] in
        equal ~msg:"value" (exact ()) none (Rune.vmap' (iterated w0) none);
        equal ~msg:"grad" (exact ()) none
          (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (iterated w0) xs)) none));
    test "a map of one lane is the loop" (fun () ->
        let x = vec [| 1.9 |] in
        let loss x = Nx.sum (Nx.sin (iterated w0 x)) in
        equal ~msg:"value" (exact ())
          (stack 1 (fun _ -> iterated w0 (scalar 1.9)))
          (Rune.vmap' (iterated w0) x);
        equal ~msg:"grad" (close ())
          (stack 1 (fun _ -> Rune.grad' loss (scalar 1.9)))
          (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' loss xs)) x));
    test "vmap of grad of vmap covers each lane's trips" (fun () ->
        let xs =
          Nx.create f64 [| 2; 3 |] [| 1.9; 0.01; -1.4; 0.3; 1.; -0.02 |]
        in
        let loss x = Nx.sum (Nx.sin (iterated w0 x)) in
        equal (close ())
          (per_lane (per_lane (Rune.grad' loss)) xs)
          (Rune.vmap' (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' loss xs))) xs));
    test "a zero-size carry tensor" (fun () ->
        let pair = Nx.Ptree.(pair tensor tensor) in
        let run (x, e) =
          Rune.iterate pair ~max:80
            ~until:(fun (x, _) -> small tol x)
            ~f:(fun (x, e) -> (contract w0 x, Nx.mul_s e 2.))
            (x, e)
        in
        let loss xe =
          let x, e = run xe in
          Nx.add (Nx.sum (Nx.sin x)) (Nx.sum e)
        in
        let e = Nx.zeros f64 [| 0 |] in
        let gx, ge = Rune.grad pair loss (scalar 1.9, e) in
        equal ~msg:"x" (close ())
          (Rune.grad'
             (fun x ->
               Nx.sum
                 (Nx.sin (plain ~max:80 ~until:(small tol) ~f:(contract w0) x)))
             (scalar 1.9))
          gx;
        equal ~msg:"e" (exact ()) e ge;
        let es = Nx.zeros f64 [| 5; 0 |] in
        let _, ys =
          Rune.vmap
            Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
            (fun x e -> run (x, e))
            (xs ()) es
        in
        equal ~msg:"lanes" (exact ()) es ys);
  ]

(* iterate' *)

let primed_law =
  prop "iterate' is iterate at one tensor, eagerly and under grad"
    Gen.(pair Expr.gen (pair Expr.point (float_range 0.01 2.)))
    (fun (p, (x0, tol)) ->
      let until = small tol and f = program_step p in
      let one () = Rune.iterate Nx.Ptree.tensor ~max:6 ~until ~f x0 in
      let primed () = Rune.iterate' ~max:6 ~until ~f x0 in
      match one () with
      | exception Invalid_argument m ->
          cover "raises" true;
          raises (Invalid_argument m) primed
      | x ->
          cover "returns" true;
          equal (exact ()) x (primed ());
          let loss iterate x = Nx.sum (Nx.sin (iterate x)) in
          equal (exact ())
            (Rune.grad'
               (loss (fun x -> Rune.iterate Nx.Ptree.tensor ~max:6 ~until ~f x))
               x0)
            (Rune.grad' (loss (fun x -> Rune.iterate' ~max:6 ~until ~f x)) x0))

(* Loops with scans *)

let scan_rows = vec [| 0.4; -0.9; 1.3; 0.2 |]

(* An iterate in a scan's step, and that scan written out over its rows. *)
let iterate_in_scan iterate w x =
  let step c r =
    let c = iterate ~max:80 ~until:(small tol) ~f:(contract w) (Nx.add c r) in
    (c, Nx.sin c)
  in
  let c, ys = Rune.scan' ~f:step ~init:x scan_rows in
  Nx.add (Nx.sum c) (Nx.sum ys)

let iterate_in_written_scan w x =
  let rec go i c acc =
    if i = Nx.dim 0 scan_rows then Nx.add (Nx.sum c) acc
    else
      let c =
        plain ~max:80 ~until:(small tol) ~f:(contract w)
          (Nx.add c (lane i scan_rows))
      in
      go (i + 1) c (Nx.add acc (Nx.sin c))
  in
  go 0 x (scalar 0.)

(* A scan in an iterate's step, and both written out. *)
let small_rows = vec [| 0.02; -0.01; 0.03 |]

let scan_step w x =
  fst
    (Rune.scan'
       ~f:(fun c r -> (Nx.add (contract w c) (Nx.mul_s r 0.1), c))
       ~init:x small_rows)

let written_scan_step w x =
  let rec go i c =
    if i = Nx.dim 0 small_rows then c
    else go (i + 1) (Nx.add (contract w c) (Nx.mul_s (lane i small_rows) 0.1))
  in
  go 0 x

let scan_in_iterate iterate step w x =
  Nx.sum (Nx.sin (iterate ~max:80 ~until:(small 0.1) ~f:(step w) x))

let scan_in = scan_in_iterate Rune.iterate' scan_step
let scan_in_written = scan_in_iterate plain written_scan_step

let scan_law =
  prop ~count:20
    "iterates and scans nested either way are the loops written out"
    Gen.(
      map
        (fun l -> vec (Array.of_list l))
        (list ~size:(int_range 3 3) (float_range (-2.) 2.)))
    (fun xs ->
      List.iter
        (fun (name, loss, written) ->
          let msg what = name ^ ": " ^ what in
          let grads = per_lane (Rune.grad' (written w0)) xs in
          equal ~msg:(msg "value") (close ())
            (per_lane (written w0) xs)
            (Rune.vmap' (loss w0) xs);
          equal ~msg:(msg "grad") (close ())
            (Rune.grad' (written w0) (lane 0 xs))
            (Rune.grad' (loss w0) (lane 0 xs));
          equal ~msg:(msg "vmap of grad") (close ()) grads
            (Rune.vmap' (Rune.grad' (loss w0)) xs);
          equal ~msg:(msg "grad of vmap") (close ()) grads
            (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (loss w0) xs)) xs);
          equal ~msg:(msg "jvp of vmap") (close ()) grads
            (snd (Rune.jvp' (Rune.vmap' (loss w0)) xs (Nx.ones_like xs)));
          equal
            ~msg:(msg "grad of a captured parameter")
            (close ())
            (Rune.grad'
               (fun w ->
                 Nx.sum (stack (lanes xs) (fun i -> written w (lane i xs))))
               w0)
            (Rune.grad' (fun w -> Nx.sum (Rune.vmap' (loss w) xs)) w0))
        [
          ( "an iterate in a scan's step",
            iterate_in_scan Rune.iterate',
            iterate_in_written_scan );
          ("a scan in an iterate's step", scan_in, scan_in_written);
        ])

(* What until reads *)

let until_tests =
  let from_start t =
    Rune.iterate' ~max:80
      ~until:(fun x -> Nx.less (Nx.abs x) t)
      ~f:(contract w0) (scalar 1.9)
  in
  [
    test "an until that reads a tracked value adds no gradient" (fun () ->
        equal (exact ()) (scalar 0.)
          (Rune.grad' (fun t -> Nx.sum (from_start t)) (scalar 0.05)));
    test "an until that reads a tracked value adds no tangent" (fun () ->
        equal (exact ()) (scalar 0.)
          (snd (Rune.jvp' from_start (scalar 0.05) (scalar 1.))));
    test "under vmap each lane stops on its own tracked tolerance" (fun () ->
        let ts = vec [| 0.5; 0.05; 0.005 |] in
        equal ~msg:"value" (exact ())
          (per_lane
             (fun t ->
               plain ~max:80
                 ~until:(fun x -> Nx.less (Nx.abs x) t)
                 ~f:(contract w0) (scalar 1.9))
             ts)
          (Rune.vmap' from_start ts);
        equal ~msg:"grad" (exact ()) (Nx.zeros_like ts)
          (Rune.grad' (fun ts -> Nx.sum (Rune.vmap' from_start ts)) ts));
  ]

(* Compilation: a compiled call runs the loop with its stop read before each
   trip, its step traced once. A compiled program may contract or reorder
   arithmetic, so its values are compared within [close]; a carry that a lane
   holds is compared bit for bit. *)

let two = Nx.Ptree.(tensor @-> tensor @-> returns tensor)

let compiled_loop_law =
  prop ~count:25
    ~examples:[ (Expr.X, (Nx.ones f64 Expr.shape, 0.6), 1) ]
    "compiled, iterate is the loop that tests until before each step"
    Gen.(
      triple Expr.gen (pair Expr.point (float_range 0.01 2.)) (int_range 0 8))
    (fun (p, (x0, tol), max) ->
      let until = small tol and f = program_step p in
      let got () = Rune.jit' (Rune.iterate' ~max ~until ~f) x0 in
      match loop ~max ~until ~f x0 with
      | Ok (x, k) ->
          cover "no step" (k = 0);
          cover "some steps" (k > 0);
          cover "at the bound" (k = max && max > 0);
          equal (close ()) x (got ())
      | Error () ->
          cover "raises" true;
          raises (Invalid_argument (failure max)) got)

let compiled_bound_law =
  prop ~count:25
    ~examples:[ (vec [| 0.06; 0.01 |], 1) ]
    "compiled, under vmap each lane ends within max, or the first running is \
     named"
    Gen.(pair starts (int_range 0 5))
    (fun (xs, max) ->
      let until = small tol and f = contract w0 in
      let loss x = Nx.sum (Nx.sin (Rune.iterate' ~max ~until ~f x)) in
      let ks = outcomes ~max xs in
      let applications =
        [
          ("vmap", Rune.jit' (Rune.vmap' (Rune.iterate' ~max ~until ~f)));
          ("vmap of grad", Rune.jit' (Rune.vmap' (Rune.grad' loss)));
          ( "grad of vmap",
            Rune.jit' (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' loss xs))) );
        ]
      in
      cover "max = 0" (max = 0);
      cover "max = 1" (max = 1);
      match List.find_index Result.is_error ks with
      | Some i ->
          cover "a lane still runs at max" true;
          List.iter
            (fun (msg, a) ->
              raises ~msg
                (Invalid_argument (lane_failure max i))
                (fun () -> a xs))
            applications
      | None ->
          cover "a lane stops at max" (at_bound ~max ks);
          let carries =
            stack (lanes xs) (fun i -> fst (Result.get_ok (List.nth ks i)))
          and grads =
            per_lane
              (Rune.grad' (fun x -> Nx.sum (Nx.sin (plain ~max ~until ~f x))))
              xs
          in
          List.iter2
            (fun (msg, a) expected -> equal ~msg (close ()) expected (a xs))
            applications [ carries; grads; grads ])

let compiled_hold_law =
  prop ~count:15
    "compiled, a stopped lane's lane_index is its donor's and its additions \
     drop"
    Gen.(
      map
        (fun l -> vec (Array.of_list l))
        (list ~size:(int_range 2 5) (float_range (-2.) 2.)))
    (fun xs ->
      let n = lanes xs in
      let ids = Nx.arange Nx.int32 0 n 1 in
      let mapped =
        Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns tensor) checked
      in
      let collected f =
        Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (fun xs ->
            Rune.Total.collect carried ~zero:(scalar 0.) (fun () -> f xs))
          xs
      in
      let expected = Nx.sum (per_lane carries xs) in
      let ys, total = collected (fun xs -> mapped xs ids) in
      equal ~msg:"carry" (close ()) (per_lane (iterated w0) xs) ys;
      equal ~msg:"total" (close ()) expected total;
      let g, total =
        collected (Rune.grad' (fun xs -> Nx.sum (Nx.sin (mapped xs ids))))
      in
      equal ~msg:"grad" (close ())
        (per_lane (Rune.grad' (fun x -> Nx.sum (Nx.sin (iterated w0 x)))) xs)
        g;
      equal ~msg:"total under grad" (close ()) expected total)

let compiled_nested_law =
  prop ~count:10
    "compiled, an iterate in an iterate's step is the loops written out"
    Gen.(pair (float_range 0.3 1.2) (Expr.points 3))
    (fun (w, xs) ->
      let w = scalar w in
      let xs = Nx.reshape [| 3; 6 |] xs in
      let ks = List.init 3 (fun i -> schedule w (lane i xs)) in
      cover "outer lanes stop apart"
        (List.length (List.sort_uniq compare (List.map List.length ks)) > 1);
      cover "inner loops stop apart"
        (List.exists (fun k -> List.length (List.sort_uniq compare k) > 1) ks);
      let grads =
        stack 3 (fun i ->
            Rune.grad' (written_nested (List.nth ks i) w) (lane i xs))
      in
      equal ~msg:"value" (close ())
        (stack 3 (fun i -> nested w (lane i xs)))
        (Rune.jit' (Rune.vmap' (nested w)) xs);
      equal ~msg:"vmap of grad" (close ()) grads
        (Rune.jit' (Rune.vmap' (Rune.grad' (nested_loss w))) xs);
      equal ~msg:"grad of vmap" (close ()) grads
        (Rune.jit'
           (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' (nested_loss w) xs)))
           xs))

let compiled_tests =
  List.map
    (fun (name, batched, written) ->
      test (name ^ ", compiled, covers each lane's own trips") (fun () ->
          let xs = xs () and vs = v () in
          equal (close ())
            (written written_loss xs vs)
            (Rune.jit two (batched loss) xs vs)))
    batched
  @ [
      test "grad, compiled, covers the trips taken" (fun () ->
          let x = scalar 1.9 in
          equal (close ())
            (Rune.grad' (written_loss w0) x)
            (Rune.jit' (Rune.grad' (loss w0)) x));
      test "jvp, compiled, covers the trips taken" (fun () ->
          equal (close ())
            (snd (Rune.jvp' (written_loss w0) (scalar 1.9) (scalar 0.3)))
            (Rune.jit two
               (fun x v -> snd (Rune.jvp' (loss w0) x v))
               (scalar 1.9) (scalar 0.3)));
      test "compiled, the error names the first lane still running" (fun () ->
          raises
            (Invalid_argument (failure 5 ^ ", in lane 1"))
            (fun () ->
              Rune.jit'
                (Rune.vmap' (Rune.iterate' ~max:5 ~until:(small 0.5) ~f:Fun.id))
                (vec [| 0.1; 2.; 3. |])));
      test
        "compiled, under nested maps the error names the lanes outermost first"
        (fun () ->
          raises
            (Invalid_argument (failure 3 ^ ", in lane 1, in lane 2"))
            (fun () ->
              Rune.jit'
                (Rune.vmap'
                   (Rune.vmap'
                      (Rune.iterate' ~max:3 ~until:(small 0.5) ~f:Fun.id)))
                (Nx.create f64 [| 2; 3 |] [| 0.1; 0.1; 0.1; 0.1; 0.1; 2. |])));
      test "compiled, a stopped lane keeps its carry bit for bit" (fun () ->
          equal (exact ())
            (per_lane rooted (vec [| 1.; 16. |]))
            (Rune.jit' (Rune.vmap' rooted) (vec [| 1.; 16. |])));
      test "compiled, a stopped lane's gradient is its own trips' and finite"
        (fun () ->
          let g =
            Rune.jit' (Rune.vmap' (Rune.grad' rooted)) (vec [| 1.; 16. |])
          in
          equal (close ())
            (per_lane (Rune.grad' written_rooted) (vec [| 1.; 16. |]))
            g;
          equal (exact ()) (vec [| 0.5 |]) (Nx.slice [ R (0, 1) ] g));
      test "compiled, a stopped lane keeps a NaN or infinite carry bit for bit"
        (fun () ->
          let xs = nonfinite () in
          equal (exact ())
            (Nx.where (Nx.isfinite xs) (Nx.zeros_like xs) xs)
            (Nx.where (Nx.isfinite xs) (Nx.zeros_like xs)
               (Rune.jit' (Rune.vmap' (settle w0)) xs)));
      test "compiled, a scope around the map counts each lane's trips"
        (fun () ->
          let xs = xs () in
          let _, n =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun xs ->
                Rune.Total.collect count ~zero:(scalar 0.) (fun () ->
                    Rune.vmap' counted xs))
              xs
          in
          equal (exact ()) (scalar (trips_of xs)) n);
      test
        "compiled, iterates and scans nested either way are the loops written \
         out" (fun () ->
          let x = scalar 0.7 in
          List.iter
            (fun (name, loss, written) ->
              let msg what = name ^ ": " ^ what in
              equal ~msg:(msg "value") (close ()) (written w0 x)
                (Rune.jit' (loss w0) x);
              equal ~msg:(msg "grad") (close ())
                (Rune.grad' (written w0) x)
                (Rune.jit' (Rune.grad' (loss w0)) x))
            [
              ( "an iterate in a scan's step",
                iterate_in_scan Rune.iterate',
                iterate_in_written_scan );
              ("a scan in an iterate's step", scan_in, scan_in_written);
            ]);
      test
        "compiled, an iterate in a custom_jvp tangent map under reverse mode \
         raises Jit_error" (fun () ->
          let f =
            Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
                ( Nx.sin x,
                  fun dx ->
                    fst
                      (Rune.iterate
                         Nx.Ptree.(pair tensor tensor)
                         ~max:5
                         ~until:(fun (_, k) -> Nx.greater_equal_s k 3l)
                         ~f:(fun (d, k) -> (Nx.mul_s d 0.5, Nx.add_s k 1l))
                         (Nx.mul (Nx.cos x) dx, Nx.scalar Nx.int32 0l)) ))
          in
          raises
            (Rune.Jit_error
               "Rune.jit: Rune.iterate cannot be compiled inside a custom_jvp \
                tangent map under reverse mode") (fun () ->
              Rune.jit' (Rune.grad' (fun x -> Nx.sum (f x))) (scalar 0.3)));
      test "compiled, max = 0 tests until once and steps never" (fun () ->
          raises
            (Invalid_argument (failure 0))
            (fun () ->
              Rune.jit'
                (Rune.iterate' ~max:0 ~until:(small 0.1) ~f:Fun.id)
                (scalar 1.)));
      test "compiled, the step is traced once" (fun () ->
          let steps = ref 0 in
          let f x =
            incr steps;
            contract w0 x
          in
          let g = Rune.jit' (Rune.iterate' ~max:80 ~until:(small tol) ~f) in
          equal (close ()) (iterated w0 (scalar 1.9)) (g (scalar 1.9));
          equal ~msg:"steps traced" int 1 !steps);
      compiled_loop_law;
      compiled_bound_law;
      compiled_hold_law;
      compiled_nested_law;
    ]

let () =
  exit
    (run "Rune.iterate"
       [
         group "loop" loop_tests;
         group "refusals" refusal_tests;
         group "lanes" lane_tests;
         group "totals" total_tests;
         group "collectives" collective_tests;
         group "derivatives" derivative_tests;
         group "holds" hold_tests;
         group "nested" [ nested_law; nested_hold_law ];
         group "bounds" bound_tests;
         group "a start that satisfies until" unstepped;
         group "non-finite carries" nonfinite_tests;
         group "stopped lanes" [ hold_law ];
         group "dtypes" dtype_tests;
         group "complex carries" complex_tests;
         group "structures and sizes" structure_tests;
         group "iterate'" [ primed_law ];
         group "scans" [ scan_law ];
         group "until" until_tests;
         group "compiled" compiled_tests;
       ])
