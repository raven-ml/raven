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
             "Rune.iterate: until must return one boolean, got bool [2]")
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

(* Compilation *)

let compiled_tests =
  [
    test "a loop that stops on a condition raises Jit_error under jit"
      (fun () ->
        raises
          (Rune.Jit_error
             "Rune.jit: a loop that stops on a condition (Rune.iterate) cannot \
              be compiled") (fun () -> Rune.jit' (iterated w0) (scalar 1.9)));
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
         group "compiled" compiled_tests;
       ])
