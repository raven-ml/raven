(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.switch and Rune.cond: the taken branch alone computes, under forward
   mode, reverse mode, maps and jit, against the branch run as OCaml code. *)

open Windtrap
module T = Nx.Ptree
module Total = Rune.Total

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let scalar x = Nx.scalar f64 x
let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f

(* Floats equal within a relative [1e-12], NaN equal to NaN. *)
let close =
  let same a b =
    Float.equal a b
    || (Float.is_nan a && Float.is_nan b)
    || Float.abs (a -. b) <= 1e-12 *. Float.max 1. (Float.abs a)
  in
  Testable.make ~pp:Nx.pp ~equal:(fun x y ->
      Nx.shape x = Nx.shape y
      && Array.for_all2 same
           (Nx.to_array (Nx.cast f64 x))
           (Nx.to_array (Nx.cast f64 y)))

(* A piecewise function: [log x] above zero, [x] elsewhere, written with a
   branch, and the same written as OCaml code on the value. *)
let piecewise x =
  Rune.cond' (Nx.greater_s x 0.) (fun () -> Nx.log x) (fun () -> Nx.mul_s x 1.)

let written x = if Nx.item [] x > 0. then Nx.log x else Nx.mul_s x 1.
let written' x = if Nx.item [] x > 0. then Nx.recip x else Nx.ones_like x
let points = [ -2.; -0.; 0.; 1e-300; 0.5; 3. ]

(* Each mode of applying a function of a 0-d value. *)
let modes : (string * ((_ -> _) -> _ -> _)) list =
  [
    ("eager", fun f x -> f x);
    ("jit", fun f x -> Rune.jit' f x);
    ( "in a scan's step",
      fun f x ->
        fst
          (Rune.scan' ~init:(Nx.zeros_like x)
             ~f:(fun _ r -> (f r, r))
             (Nx.reshape [| 1 |] x)) );
  ]

let test_values () =
  List.iter
    (fun (mode, apply) ->
      List.iter
        (fun p ->
          let x = scalar p in
          equal
            ~msg:(Printf.sprintf "%s at %g" mode p)
            close (written x) (apply piecewise x))
        points)
    modes

let test_switch () =
  let branches x =
    [
      (fun () -> Nx.mul_s x 2.); (fun () -> Nx.add_s x 1.); (fun () -> Nx.neg x);
    ]
  in
  let x = vec [| 1.; 2. |] in
  List.iteri
    (fun k expected ->
      equal ~msg:(string_of_int k) close expected
        (Rune.switch T.tensor
           (Nx.scalar Nx.int32 (Int32.of_int k))
           (branches x)))
    [ vec [| 2.; 4. |]; vec [| 2.; 3. |]; vec [| -1.; -2. |] ]

let test_derivatives () =
  List.iter
    (fun (mode, apply) ->
      List.iter
        (fun p ->
          let x = scalar p in
          let msg = Printf.sprintf "%s at %g" mode p in
          equal ~msg:("grad " ^ msg) close (written' x)
            (apply (Rune.grad' piecewise) x);
          equal ~msg:("jvp " ^ msg) close (written' x)
            (apply (fun x -> snd (Rune.jvp' piecewise x (Nx.ones_like x))) x);
          equal ~msg:("grad of " ^ msg) close (written' x)
            (Rune.grad' (fun x -> apply piecewise x) x))
        points)
    modes

(* The untaken branch computes nothing: no NaN reaches the derivative, no check
   fails, no addition counts. *)
let test_untaken () =
  let zero = scalar 0. in
  equal ~msg:"where's NaN" close (scalar Float.nan)
    (Rune.grad'
       (fun x -> Nx.where (Nx.greater_s x 0.) (Nx.log x) (Nx.mul_s x 1.))
       zero);
  equal ~msg:"cond's" close (scalar 1.) (Rune.grad' piecewise zero);
  let x = scalar (-1.) in
  let t = Total.make () in
  let counted x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Total.add t (scalar 1.);
        Nx.check T.tensor (Nx.greater_s x 0.) x (fun _ _ -> Exit);
        x)
      (fun () -> Nx.neg x)
  in
  List.iter
    (fun (mode, apply) ->
      let y, n =
        Total.collect t ~init:(scalar 0.) (fun () -> apply counted x)
      in
      equal ~msg:(mode ^ " value") close (scalar 1.) y;
      equal ~msg:(mode ^ " additions") close (scalar 0.) n;
      let _, n =
        Total.collect t ~init:(scalar 0.) (fun () -> apply counted (scalar 2.))
      in
      equal ~msg:(mode ^ " taken additions") close (scalar 1.) n)
    modes

let test_traced_once () =
  let runs = ref 0 in
  let f x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        incr runs;
        x)
      (fun () ->
        incr runs;
        Nx.neg x)
  in
  ignore (f (scalar 1.));
  equal int 2 !runs

let test_reads () =
  invalid ~sub:"Nx.item: the value is traced by Rune.cond'" (fun () ->
      Rune.cond' (Nx.scalar Nx.bool true)
        (fun () ->
          if Nx.item [] (Nx.ones f64 [||]) > 0. then scalar 1. else scalar 2.)
        (fun () -> scalar 0.))

let test_refusals () =
  invalid ~sub:"Rune.switch: no branches" (fun () ->
      Rune.switch T.tensor (Nx.scalar Nx.int32 0l) []);
  invalid ~sub:"Rune.switch: index 3 outside 0 .. 1" (fun () ->
      Rune.switch T.tensor (Nx.scalar Nx.int32 3l)
        [ (fun () -> scalar 0.); (fun () -> scalar 1.) ]);
  raises
    (Invalid_argument
       "Rune.switch: the root: shape [3] in branch 1, [2] in branch 0")
    (fun () ->
      Rune.switch T.tensor (Nx.scalar Nx.int32 0l)
        [ (fun () -> vec [| 1.; 2. |]); (fun () -> vec [| 1.; 2.; 3. |]) ]);
  invalid ~sub:"Rune.switch" (fun () ->
      Rune.switch (T.list T.tensor) (Nx.scalar Nx.int32 0l)
        [ (fun () -> [ scalar 1. ]); (fun () -> [ scalar 1.; scalar 2. ]) ]);
  (* Inside an iterate whose lanes stop apart, a held lane computes its donor's
     index: the error names the lane that runs. *)
  invalid ~sub:"Rune.switch: index 2 outside 0 .. 1, in lane 1" (fun () ->
      Rune.vmap'
        (Rune.iterate' ~max:3
           ~until:(fun x -> Nx.greater_equal_s x 1.)
           ~f:(fun x ->
             Rune.switch T.tensor
               (Nx.cast Nx.int32 (Nx.add_s x 2.))
               [ (fun () -> Nx.add_s x 1.); (fun () -> Nx.add_s x 1.) ]))
        (vec [| 1.; 0. |]))

(* Maps: each lane takes its own branch, and its derivatives are that
   branch's. *)
let lanes = vec [| -2.; 0.5; -0.; 3.; 1e-300 |]

let per_lane f xs =
  Nx.stack (List.init (Nx.dim 0 xs) (fun i -> f (Nx.get [ i ] xs)))

let test_maps () =
  equal ~msg:"vmap" close (per_lane written lanes) (Rune.vmap' piecewise lanes);
  equal ~msg:"vmap of grad" close (per_lane written' lanes)
    (Rune.vmap' (Rune.grad' piecewise) lanes);
  equal ~msg:"grad of vmap" close (per_lane written' lanes)
    (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' piecewise xs)) lanes);
  equal ~msg:"jit of vmap" close (per_lane written lanes)
    (Rune.jit' (Rune.vmap' piecewise) lanes);
  equal ~msg:"jit of grad of vmap" close (per_lane written' lanes)
    (Rune.jit' (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' piecewise xs))) lanes);
  equal ~msg:"no lane" close (Nx.zeros f64 [| 0 |])
    (Rune.vmap' piecewise (Nx.zeros f64 [| 0 |]));
  (* A shared index: every lane takes one branch. *)
  let shared p xs =
    Rune.vmap' (fun x -> Rune.cond' p (fun () -> Nx.log x) (fun () -> x)) xs
  in
  equal ~msg:"shared" close
    (Nx.log (vec [| 1.; 2. |]))
    (shared (Nx.scalar Nx.bool true) (vec [| 1.; 2. |]))

let test_lane_masks () =
  let t = Total.make () in
  let f x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Total.add t (scalar 1.);
        Nx.check T.tensor (Nx.greater_s x 0.) x (fun _ _ -> Exit);
        Nx.log x)
      (fun () -> x)
  in
  (* The lanes' additions lie where the lanes do: so does the running value. *)
  let zero = Nx.mul_s (Nx.sum lanes) 0. in
  let y, n = Total.collect t ~init:zero (fun () -> Rune.vmap' f lanes) in
  equal ~msg:"value" close (per_lane written lanes) y;
  equal ~msg:"additions" close (scalar 3.) n;
  let _, n =
    Total.collect t ~init:zero (fun () -> Rune.vmap' f (vec [| -1.; -2. |]))
  in
  equal ~msg:"no lane takes it" close (scalar 0.) n;
  invalid ~sub:"Rune.switch: index 2 outside 0 .. 1, in lane 1" (fun () ->
      Rune.vmap'
        (fun k ->
          Rune.switch T.tensor (Nx.cast Nx.int32 k)
            [ (fun () -> scalar 0.); (fun () -> scalar 1.) ])
        (vec [| 0.; 2.; 1. |]))

(* Lanes, each with a value and a branch among three: a map of the switch and
   its derivatives are each lane's own branch's, by every route. *)
let lane_cases =
  Gen.(
    map
      (fun l ->
        (vec (Array.of_list (List.map fst l)), Array.of_list (List.map snd l)))
      (list ~size:(int_range 0 5) (pair (float_range (-2.) 2.) (int_range 0 2))))

let three x k =
  Rune.switch T.tensor k
    [
      (fun () -> Nx.sin x);
      (fun () -> Nx.mul x x);
      (fun () -> Nx.exp (Nx.neg x));
    ]

let derivative x k =
  match k with
  | 0 -> Nx.cos x
  | 1 -> Nx.mul_s x 2.
  | _ -> Nx.neg (Nx.exp (Nx.neg x))

let law_lanes (xs, ks) =
  let n = Array.length ks in
  cover "no lane" (n = 0);
  cover "lanes taking different branches"
    (Array.exists (fun k -> k <> ks.(0)) ks);
  let idx = Nx.create Nx.int32 [| n |] (Array.map Int32.of_int ks) in
  let each f =
    if n = 0 then Nx.zeros f64 [| 0 |]
    else Nx.stack (List.init n (fun j -> f (Nx.get [ j ] xs) ks.(j)))
  in
  let value x k = three x (Nx.scalar Nx.int32 (Int32.of_int k)) in
  let mapped f = Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns tensor) f in
  begin
    equal ~msg:"vmap" close (each value) (mapped three xs idx);
    equal ~msg:"jit of vmap" close (each value)
      (Rune.jit
         Nx.Ptree.(tensor @-> tensor @-> returns tensor)
         (mapped three) xs idx);
    equal ~msg:"grad of vmap" close (each derivative)
      (Rune.grad' (fun xs -> Nx.sum (mapped three xs idx)) xs);
    equal ~msg:"vmap of grad" close (each derivative)
      (mapped (fun x k -> Rune.grad' (fun x -> three x k) x) xs idx);
    equal ~msg:"jvp of vmap" close (each derivative)
      (snd (Rune.jvp' (fun xs -> mapped three xs idx) xs (Nx.ones_like xs)))
  end

(* A lane's gradient is its own branch's, sign of zero included: lane 0's branch
   multiplies by [-0.], giving [-0.]; lane 1's reads nothing of [x], giving
   [+0.], whatever lane 0's branch gives there. *)
let test_lane_zeros () =
  let f x k =
    Rune.switch T.tensor k
      [ (fun () -> Nx.mul_s x (-0.)); (fun () -> Nx.ones_like x) ]
  in
  let xs = vec [| 2.; 3. |] and idx = Nx.create Nx.int32 [| 2 |] [| 0l; 1l |] in
  let s = T.(tensor @-> tensor @-> returns tensor) in
  let grad_vmap xs = Rune.grad' (fun xs -> Nx.sum (Rune.vmap s f xs idx)) xs in
  let codes x = Array.map Int64.bits_of_float (Nx.to_array x) in
  let expected = codes (vec [| -0.; 0. |]) in
  equal ~msg:"grad of vmap" (array int64) expected (codes (grad_vmap xs));
  equal ~msg:"vmap of grad" (array int64) expected
    (codes (Rune.vmap s (fun x k -> Rune.grad' (fun x -> f x k) x) xs idx));
  equal ~msg:"jit of grad of vmap" (array int64) expected
    (codes (Rune.jit' grad_vmap xs))

(* The traced per-lane path: under jit and in a scan's step, each lane's branch,
   its additions and its checks. *)
let test_traced_lanes () =
  let t = Total.make () in
  let f x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Total.add t (scalar 1.);
        Nx.check T.tensor (Nx.greater_s x 0.) x (fun _ _ -> Exit);
        Nx.log x)
      (fun () -> x)
  in
  let zero = Nx.mul_s (Nx.sum lanes) 0. in
  let y, n =
    Total.collect t ~init:zero (fun () -> Rune.jit' (Rune.vmap' f) lanes)
  in
  equal ~msg:"jit value" close (per_lane written lanes) y;
  equal ~msg:"jit additions" close (scalar 3.) n;
  let stepped =
    fst
      (Rune.scan' ~init:(Nx.zeros_like lanes)
         ~f:(fun _ r -> (Rune.vmap' f (Nx.add lanes r), r))
         (Nx.zeros f64 [| 1 |]))
  in
  equal ~msg:"in a step" close (per_lane written lanes) stepped;
  invalid ~sub:"Rune.switch: index 2 outside 0 .. 1, in lane 1" (fun () ->
      Rune.jit'
        (Rune.vmap' (fun k ->
             Rune.switch T.tensor (Nx.cast Nx.int32 k)
               [ (fun () -> scalar 0.); (fun () -> scalar 1.) ]))
        (vec [| 0.; 2.; 1. |]))

let test_out_of_range () =
  let pick k =
    Rune.switch T.tensor k [ (fun () -> scalar 0.); (fun () -> scalar 1.) ]
  in
  invalid ~sub:"Rune.switch: index -1 outside 0 .. 1" (fun () ->
      pick (Nx.scalar Nx.int32 (-1l)));
  invalid ~sub:"Rune.switch: index 2 outside 0 .. 1" (fun () ->
      Rune.jit' (fun x -> Nx.add x (pick (Nx.cast Nx.int32 x))) (scalar 2.));
  invalid ~sub:"Rune.switch: index 2 outside 0 .. 1" (fun () ->
      Rune.scan' ~init:(scalar 0.)
        ~f:(fun c r -> (Nx.add c (pick (Nx.cast Nx.int32 r)), c))
        (vec [| 1.; 2. |]))

(* A switch per lane inside an iterate whose lanes stop apart, and an iterate
   inside a branch: each lane computes its own code. *)
let test_nested_masks () =
  let collatz x =
    Rune.iterate' ~max:200
      ~until:(fun x -> Nx.less_equal_s x 1.)
      ~f:(fun x ->
        Rune.cond'
          (Nx.equal_s (Nx.mod_s x 2.) 0.)
          (fun () -> Nx.div_s x 2.)
          (fun () -> Nx.add_s (Nx.mul_s x 3.) 1.))
      x
  in
  let xs = vec [| 1.; 6.; 7.; 27. |] in
  let written x =
    let rec go x =
      if x <= 1. then x
      else go (if Float.rem x 2. = 0. then x /. 2. else (3. *. x) +. 1.)
    in
    scalar (go (Nx.item [] x))
  in
  equal ~msg:"switch in iterate" close (per_lane written xs)
    (Rune.vmap' collatz xs);
  let halve x =
    Rune.cond' (Nx.greater_s x 1.)
      (fun () ->
        Rune.iterate' ~max:100
          ~until:(fun y -> Nx.less_equal_s y 1.)
          ~f:(fun y -> Nx.div_s y 2.)
          x)
      (fun () -> x)
  in
  let written x =
    let rec go y = if y <= 1. then y else go (y /. 2.) in
    scalar (go (Nx.item [] x))
  in
  equal ~msg:"iterate in switch" close (per_lane written xs)
    (Rune.vmap' halve xs)

(* An addition in a branch: its tangent is the taken branch's. *)
let test_total_derivatives () =
  let t = Total.make () in
  let f x =
    snd
      (Total.collect t ~init:(Nx.mul_s x 0.) (fun () ->
           Rune.cond' (Nx.greater_s x 0.)
             (fun () ->
               Total.add t (Nx.mul x x);
               x)
             (fun () ->
               Total.add t (Nx.mul_s x 3.);
               x)))
  in
  List.iter
    (fun (mode, apply) ->
      List.iter
        (fun (p, d) ->
          equal
            ~msg:(Printf.sprintf "%s at %g" mode p)
            close (scalar d)
            (apply (fun x -> snd (Rune.jvp' f x (Nx.ones_like x))) (scalar p)))
        [ (2., 4.); (-1., 3.) ])
    modes

let test_lanes_in_branch () =
  let a = Rune.axis () in
  let xs = vec [| -1.; 2.; -3.; 4. |] in
  equal ~msg:"lane index" close
    (vec [| 0.; 1.; 0.; 3. |])
    (Rune.vmap' ~axis:a
       (fun x ->
         Rune.cond' (Nx.greater_s x 0.)
           (fun () -> Nx.cast f64 (Rune.lane_index a x))
           (fun () -> Nx.mul_s x 0.))
       xs);
  invalid ~sub:"Rune.lanes: the lanes of the map are held apart" (fun () ->
      Rune.vmap' ~axis:a
        (fun x ->
          Rune.cond' (Nx.greater_s x 0.)
            (fun () -> Nx.sum (Rune.lanes a x))
            (fun () -> x))
        xs)

let test_stated () =
  equal ~msg:"NaN takes the else branch" close (scalar 0.)
    (Rune.cond'
       (Nx.greater_s (scalar Float.nan) 0.)
       (fun () -> scalar 1.)
       (fun () -> scalar 0.));
  invalid ~sub:"Rune.switch: the index has shape [2], not []" (fun () ->
      Rune.switch T.tensor (Nx.zeros Nx.int32 [| 2 |]) [ (fun () -> scalar 0.) ]);
  ()

(* rune2.mli's examples: the safe logarithm, and where's idiom on its input. *)
let safe_log x =
  Rune.cond' (Nx.greater_s x 0.) (fun () -> Nx.log x) (fun () -> x)

let test_examples () =
  equal ~msg:"safe_log at -1" close (scalar 1.)
    (Rune.grad' safe_log (scalar (-1.)));
  equal ~msg:"safe_log at 2" close (scalar 0.5)
    (Rune.grad' safe_log (scalar 2.));
  let idiom x =
    let c = Nx.greater_s x 0. in
    Nx.where c (Nx.log (Nx.where c x (Rune.detach x))) x
  in
  equal ~msg:"where's idiom at 0" close (scalar 1.)
    (Rune.grad' idiom (scalar 0.))

(* Under jit a switch compiles with its program, a loop per branch, and
   computes eager's values: a branch's additions, checks and loops included. *)
let test_compiled () =
  let t = Total.make () in
  let counted x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Total.add t (Nx.mul x x);
        Nx.check T.tensor (Nx.greater_s x 0.) x (fun _ _ -> Exit);
        Nx.log x)
      (fun () ->
        Total.add t (Nx.mul_s x 3.);
        Nx.neg x)
  in
  (* A total only one branch adds to. *)
  let one_sided x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Total.add t (Nx.mul_s x 5.);
        x)
      (fun () -> Nx.neg x)
  in
  let collected f x = Total.collect t ~init:(scalar 0.) (fun () -> f x) in
  let collatz x =
    Rune.iterate' ~max:200
      ~until:(fun x -> Nx.less_equal_s x 1.)
      ~f:(fun x ->
        Rune.cond'
          (Nx.equal_s (Nx.mod_s x 2.) 0.)
          (fun () -> Nx.div_s x 2.)
          (fun () -> Nx.add_s (Nx.mul_s x 3.) 1.))
      x
  in
  let halve x =
    Rune.cond' (Nx.greater_s x 1.)
      (fun () ->
        Rune.iterate' ~max:100
          ~until:(fun y -> Nx.less_equal_s y 1.)
          ~f:(fun y -> Nx.div_s y 2.)
          x)
      (fun () -> x)
  in
  let index x = Nx.cast Nx.int32 (Nx.add_s x 1.) in
  let cases =
    [
      ("cond", piecewise, [ -1.; 0.; 2. ]);
      ("switch", (fun x -> three x (index x)), [ -1.; 0.; 1. ]);
      ("grad", Rune.grad' piecewise, [ -1.; 2. ]);
      ("jvp", (fun x -> snd (Rune.jvp' counted x (Nx.ones_like x))), [ -1.; 2. ]);
      ("in an iterate", collatz, [ 1.; 6.; 27. ]);
      ("an iterate in a branch", halve, [ 0.5; 40. ]);
      ( "a scan over constants in a branch",
        (fun x ->
          Rune.cond' (Nx.greater_s x 0.)
            (fun () ->
              Nx.add x
                (fst
                   (Rune.scan' ~init:(Nx.scalar f64 0.5)
                      ~f:(fun c r -> (Nx.add (Nx.mul_s c 2.) r, c))
                      (vec [| 1.; 2.; 3. |]))))
            (fun () -> Nx.neg x)),
        [ -1.; 2. ] );
      ( "one branch",
        (fun x -> Rune.switch T.tensor (index x) [ (fun () -> Nx.sin x) ]),
        [ -1. ] );
      ( "an empty result",
        (fun x ->
          let y, e =
            Rune.cond
              T.(pair tensor tensor)
              (Nx.greater_s x 0.)
              (fun () -> (Nx.mul_s x 2., Nx.zeros f64 [| 0 |]))
              (fun () -> (Nx.neg x, Nx.ones f64 [| 0 |]))
          in
          Nx.add y (Nx.sum e)),
        [ -1.; 2. ] );
    ]
  in
  List.iter
    (fun (msg, f, points) ->
      List.iter
        (fun p ->
          let x = scalar p in
          let msg = Printf.sprintf "%s at %g" msg p in
          let before = Rune.evaluations () in
          let y = Rune.jit' f x in
          equal ~msg:(msg ^ ": compiled") int 0 (Rune.evaluations () - before);
          equal ~msg close (f x) y)
        points)
    cases;
  List.iter
    (fun p ->
      let x = scalar p in
      let msg = Printf.sprintf "additions at %g" p in
      let before = Rune.evaluations () in
      List.iter
        (fun f ->
          let y, n = collected (Rune.jit' f) x in
          equal ~msg:(msg ^ ": compiled") int 0 (Rune.evaluations () - before);
          let y', n' = collected f x in
          equal ~msg close y' y;
          equal ~msg close n' n)
        [ counted; one_sided ])
    [ -1.; 2. ];
  (* A check failing in the taken branch raises, and one made before the
     switch raises before its index's range. *)
  let checked x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Nx.check T.tensor (Nx.less_s x 5.) x (fun _ _ -> Exit);
        x)
      (fun () -> Nx.neg x)
  in
  let before = Rune.evaluations () in
  raises Exit (fun () -> Rune.jit' checked (scalar 10.));
  equal ~msg:"checked" close (scalar 3.) (Rune.jit' checked (scalar 3.));
  raises Exit (fun () ->
      Rune.jit'
        (fun x ->
          Nx.check T.tensor (Nx.less_s x 5.) x (fun _ _ -> Exit);
          three x (index x))
        (scalar 10.));
  (* A loop whose results go unread still fails as eagerly. *)
  let unread x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        ignore
          (Rune.iterate' ~max:3
             ~until:(fun y -> Nx.less_equal_s y 0.)
             ~f:(fun y -> Nx.add_s y 1.)
             x);
        x)
      (fun () -> Nx.neg x)
  in
  let max3 = "until is still false after max = 3 steps" in
  invalid ~sub:max3 (fun () -> unread (scalar 5.));
  invalid ~sub:max3 (fun () -> Rune.jit' unread (scalar 5.));
  equal ~msg:"checks: compiled" int 0 (Rune.evaluations () - before);
  let before = Rune.evaluations () in
  invalid ~sub:"Rune.switch: index 3 outside 0 .. 2" (fun () ->
      Rune.jit' (fun x -> three x (index x)) (scalar 2.));
  invalid ~sub:"Rune.switch: index -1 outside 0 .. 2" (fun () ->
      Rune.jit' (fun x -> three x (index x)) (scalar (-2.)));
  equal ~msg:"out of range: compiled" int 0 (Rune.evaluations () - before);
  (* An untaken branch runs none of its work under jit: here a loop of [trips]
     steps, each over the [4096] elements of a capture, which takes hundreds
     of milliseconds where it runs. *)
  let trips = 1_000_000 and w = Nx.linspace f64 0. 1. 4096 in
  let slow x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        fst
          (Rune.scan' ~init:x
             ~f:(fun c _ -> (Nx.sum (Nx.mul w c), c))
             (Nx.zeros f64 [| trips; 0 |])))
      (fun () -> Nx.neg x)
  in
  (* The same loop over constants alone, the function's own arrays. *)
  let constant x =
    Rune.cond' (Nx.greater_s x 0.)
      (fun () ->
        Nx.add x
          (fst
             (Rune.scan' ~init:(Nx.scalar f64 0.)
                ~f:(fun c _ -> (Nx.sum (Nx.mul w (Nx.add_s c 1.)), c))
                (Nx.zeros f64 [| trips; 0 |]))))
      (fun () -> Nx.neg x)
  in
  List.iter
    (fun (msg, f) ->
      let jitted = Rune.jit' f in
      equal ~msg close (scalar 1.) (jitted (scalar (-1.)));
      let start = Sys.time () in
      equal ~msg close (scalar 2.) (jitted (scalar (-2.)));
      less
        ~msg:(msg ^ ": processor seconds")
        float_exact ~than:0.05
        (Sys.time () -. start))
    [ ("a loop over a capture", slow); ("a loop over constants", constant) ]

let () =
  exit
    (run "rune2 switch"
       [
         test "the taken branch's values" test_values;
         test "switch picks its branch" test_switch;
         test "the taken branch's derivatives" test_derivatives;
         test "the untaken branch computes nothing" test_untaken;
         test "each branch is traced once" test_traced_once;
         test "a read in a branch raises" test_reads;
         test "refusals" test_refusals;
         test "maps" test_maps;
         test "a lane computes only its branch" test_lane_masks;
         prop "each lane takes its own branch" lane_cases law_lanes;
         test "a lane's zero gradient keeps its sign" test_lane_zeros;
         test "lanes under a trace" test_traced_lanes;
         test "an index out of range raises" test_out_of_range;
         test "nested lane masks" test_nested_masks;
         test "a total's tangent is the taken branch's" test_total_derivatives;
         test "lanes in a branch" test_lanes_in_branch;
         test "stated cases" test_stated;
         test "the documented examples" test_examples;
         test "a switch compiles under jit" test_compiled;
       ])
