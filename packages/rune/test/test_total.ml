(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Totals. The oracle is the plain program: an addition counts once per
   execution of the code that makes it, eagerly, staged, replayed, under a map
   (as the loop over its lanes) and under reverse mode, which reruns code. *)

open Windtrap
open Rune_test_support.Support

let series seed shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f64 shape
    (Array.init n (fun i -> Float.sin (Float.of_int ((7 * i) + seed)) /. 2.0))

let w0 = series 1 [| 3; 3 |]
let h0 = vec64 [| 0.1; -0.2; 0.3 |]
let lane i x = Nx.slice [ Nx.I i ] x
let stack n f = Nx.stack ~axis:0 (List.init n f)
let zero () = Nx.zeros f64 [||]
let cell w h x = Nx.tanh (Nx.add (Nx.matmul w h) x)

(* A rollout whose body adds the sum of each state to [t]. *)
let rollout ?(runs = ref 0) ?(add = Rune.Total.add) t w h xs =
  snd
    (Rune.scan'
       ~f:(fun h x ->
         incr runs;
         let h = cell w h x in
         add t (Nx.sum h);
         (h, h))
       ~init:h xs)

(* [staged ~runs f] runs the checks [f count n] at lengths 4 and 8, and checks
   that the body counted [runs] runs in [count] at both. *)
let staged ~runs f =
  List.iter
    (fun n ->
      let count = ref 0 in
      f count n;
      equal ~msg:(Printf.sprintf "body runs, length %d" n) int runs !count)
    [ 4; 8 ]

(* Scopes *)

let test_no_scope_is_inert () =
  let t = Rune.Total.make () in
  let xs = series 2 [| 4; 3 |] in
  let ys = rollout t w0 h0 xs in
  let _, total = Rune.Total.collect t ~zero:(zero ()) (fun () -> ()) in
  check_arr ~msg:"no addition" [| 0.0 |] total;
  check_arr ~msg:"outside a scope" (to_arr (Rune.jit' (rollout t w0 h0) xs)) ys

let test_innermost_scope () =
  let t = Rune.Total.make () and u = Rune.Total.make () in
  let (inner, u_total), outer =
    Rune.Total.collect t ~zero:(Nx.scalar f64 10.0) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        let inner =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (Nx.scalar f64 2.0))
        in
        let u_total =
          snd
            (Rune.Total.collect u ~zero:(zero ()) (fun () ->
                 Rune.Total.add t (Nx.scalar f64 4.0);
                 Rune.Total.add u (Nx.scalar f64 8.0)))
        in
        (snd inner, u_total))
  in
  check_arr ~msg:"inner" [| 2.0 |] inner;
  check_arr ~msg:"another total" [| 8.0 |] u_total;
  check_arr ~msg:"outer, from its zero" [| 15.0 |] outer

let test_shape_mismatch_raises () =
  let t = Rune.Total.make () in
  let _, total =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        raises
          (Invalid_argument
             "Rune.Total.add: shape [3] does not match the total's []")
          (fun () -> Rune.Total.add t h0))
  in
  check_arr ~msg:"an addition before the caught exception counts" [| 1.0 |]
    total

let test_an_exception_leaves_the_scope () =
  let t = Rune.Total.make () in
  raises Exit (fun () ->
      ignore
        (Rune.Total.collect t ~zero:(zero ()) (fun () ->
             Rune.Total.add t (Nx.scalar f64 1.0);
             raise Exit)))

let test_a_caught_exception_keeps_its_additions () =
  let t = Rune.Total.make () in
  let _, total =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        (try
           Rune.Total.add t (Nx.scalar f64 2.0);
           raise Exit
         with Exit -> ());
        Rune.Total.add t (Nx.scalar f64 4.0))
  in
  check_arr ~msg:"all three count" [| 7.0 |] total

(* Scans and remats *)

let expected_total xs = Nx.sum (rollout (Rune.Total.make ()) w0 h0 xs)

let test_eager_scan () =
  let t = Rune.Total.make () and xs = series 2 [| 5; 3 |] in
  let runs = ref 0 in
  let _, total =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> rollout ~runs t w0 h0 xs)
  in
  check_arr ~msg:"once per step" (to_arr (expected_total xs)) total;
  equal ~msg:"body runs" int 5 !runs

let collected_rollout ?runs t w xs =
  let ys, total =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> rollout ?runs t w h0 xs)
  in
  (ys, total)

let test_staged_scan () =
  let t = Rune.Total.make () in
  staged ~runs:1 (fun runs n ->
      let f =
        Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (collected_rollout ~runs t w0)
      in
      List.iter
        (fun seed ->
          let xs = series seed [| n; 3 |] in
          let ys, total = f xs in
          check_arr ~eps:1e-9 ~msg:"outputs" (to_arr (rollout t w0 h0 xs)) ys;
          check_arr ~eps:1e-9 ~msg:"once per step, replayed"
            (to_arr (expected_total xs))
            total)
        [ 2; 3 ])

let remat_adding t =
  Rune.remat
    Nx.Ptree.(tensor @-> returns tensor)
    (fun x ->
      Rune.Total.add t (Nx.sum (Nx.mul x x));
      Nx.sin x)

let test_remat () =
  let t = Rune.Total.make () and x = series 3 [| 4 |] in
  let f x =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> (remat_adding t) x)
  in
  let expected = to_arr (Nx.sum (Nx.mul x x)) in
  check_arr ~msg:"eager" expected (snd (f x));
  let y, total =
    Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) f x
  in
  check_arr ~msg:"compiled result" (to_arr (Nx.sin x)) y;
  check_arr ~msg:"compiled" expected total

(* An exception raised while the scope runs a scan or a remat reaches the code
   that called it, which may catch it inside the scope. *)
let test_exceptions_reach_the_performer () =
  let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
  let caught f = match f () with _ -> 1.0 | exception Exit -> 2.0 in
  let within f =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        let r = caught f in
        Rune.Total.add t (Nx.scalar f64 r);
        Nx.scalar f64 r)
  in
  let scan xs =
    within (fun () -> Rune.scan' ~f:(fun _ _ -> raise Exit) ~init:h0 xs)
  in
  let check ~msg (r, total) =
    check_arr ~msg:(msg ^ ": caught") [| 2.0 |] r;
    check_arr ~msg:(msg ^ ": total") [| 3.0 |] total
  in
  check ~msg:"scan" (scan xs);
  check ~msg:"compiled scan"
    (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) scan xs);
  check ~msg:"remat"
    (within (fun () ->
         Rune.remat
           Nx.Ptree.(tensor @-> returns tensor)
           (fun _ -> raise Exit)
           (lane 0 xs)))

(* A jit inside a scope runs its function eagerly and its additions count. *)
let test_jit_inside_a_scope () =
  let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
  let f = Rune.jit' (rollout t w0 h0) in
  let ys, total = Rune.Total.collect t ~zero:(zero ()) (fun () -> f xs) in
  check_arr ~msg:"result" (to_arr (rollout t w0 h0 xs)) ys;
  check_arr ~msg:"counted" (to_arr (expected_total xs)) total

(* Maps *)

(* An addition crossing a map is the sum of its lanes' additions. *)
let test_addition_crossing_a_map () =
  let t = Rune.Total.make () and xs = series 4 [| 4; 3 |] in
  let _, total =
    Rune.Total.collect t ~zero:(Nx.zeros f64 [| 3 |]) (fun () ->
        Rune.vmap'
          (fun x ->
            Rune.Total.add t (Nx.mul x x);
            Rune.Total.add t h0;
            x)
          xs)
  in
  let loop =
    List.fold_left
      (fun acc i -> Nx.add acc (Nx.add (Nx.mul (lane i xs) (lane i xs)) h0))
      (Nx.zeros f64 [| 3 |]) [ 0; 1; 2; 3 ]
  in
  check_arr ~msg:"the loop" (to_arr loop) total;
  let xss = series 5 [| 4; 6; 3 |] in
  let _, total =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.vmap' (fun xs -> rollout t w0 h0 xs) xss)
  in
  let loop =
    List.fold_left
      (fun acc i -> Nx.add acc (expected_total (lane i xss)))
      (zero ()) [ 0; 1; 2; 3 ]
  in
  check_arr ~msg:"a map over a scan" (to_arr loop) total;
  let compiled =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (fun xss ->
        Rune.Total.collect t ~zero:(zero ()) (fun () ->
            Rune.vmap' (fun xs -> rollout t w0 h0 xs) xss))
  in
  check_arr ~eps:1e-9 ~msg:"compiled" (to_arr loop) (snd (compiled xss))

(* A scope inside a map collects per lane. *)
let test_scope_inside_a_map () =
  let t = Rune.Total.make () and xss = series 5 [| 3; 6; 3 |] in
  let totals =
    Rune.vmap' (fun xs ->
        snd
          (Rune.Total.collect t ~zero:(zero ()) (fun () -> rollout t w0 h0 xs)))
  in
  let expected = to_arr (stack 3 (fun i -> expected_total (lane i xss))) in
  check_arr ~msg:"per lane" expected (totals xss);
  check_arr ~eps:1e-9 ~msg:"compiled" expected (Rune.jit' totals xss)

(* Differentiation *)

(* A collected total is a value like any other: jvp differentiates it and grad
   tapes it, eagerly and staged. *)
let test_a_total_is_differentiated () =
  let t = Rune.Total.make () in
  let xs = series 2 [| 6; 3 |] and dw = series 4 [| 3; 3 |] in
  let collected w =
    snd (Rune.Total.collect t ~zero:(zero ()) (fun () -> rollout t w h0 xs))
  in
  let explicit w = Nx.sum (rollout (Rune.Total.make ()) w h0 xs) in
  let jvp f w dw = Rune.jvp' f w dw in
  let compiled_jvp f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
      (jvp f)
  in
  let v, d = jvp explicit w0 dw in
  let check ~msg (v', d') =
    check_arr ~eps:1e-9 ~msg:(msg ^ ": value") (to_arr v) v';
    check_arr ~eps:1e-9 ~msg:(msg ^ ": tangent") (to_arr d) d'
  in
  check ~msg:"jvp" (jvp collected w0 dw);
  check ~msg:"compiled jvp" (compiled_jvp collected w0 dw);
  let g = to_arr (Rune.grad' explicit w0) in
  check_arr ~eps:1e-9 ~msg:"grad" g (Rune.grad' collected w0);
  check_arr ~eps:1e-9 ~msg:"compiled grad" g
    (Rune.jit' (Rune.grad' collected) w0)

(* Reverse mode reruns code; its additions count once. *)

let test_grad_outside_scope () =
  let t = Rune.Total.make () and xs = series 2 [| 5; 3 |] in
  let loss ?add w = Nx.sum (rollout ?add t w h0 xs) in
  let expected = to_arr (expected_total xs) in
  let run f w =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> Rune.grad' f w)
  in
  let g = Rune.grad' (loss ~add:(fun _ _ -> ())) w0 in
  let g', total = run loss w0 in
  check_arr ~msg:"eager gradient" (to_arr g) g';
  check_arr ~msg:"eager" expected total;
  let compiled f =
    Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) (run f)
  in
  let g', total = compiled loss w0 in
  check_arr ~eps:1e-9 ~msg:"staged gradient" (to_arr g) g';
  check_arr ~eps:1e-9 ~msg:"staged" expected total;
  let no_grad t v = Rune.no_grad (fun () -> Rune.Total.add t v) in
  let _, total = compiled (loss ~add:no_grad) w0 in
  check_arr ~eps:1e-9 ~msg:"staged, under no_grad" expected total;
  let x = series 3 [| 4 |] in
  let expected = to_arr (Nx.sum (Nx.mul x x)) in
  let remat_loss x = Nx.sum (remat_adding t x) in
  check_arr ~msg:"remat" expected (snd (run remat_loss x));
  check_arr ~msg:"compiled remat" expected (snd (compiled remat_loss x))

(* [counts_once ~msg t expected loss x] checks that a scope of [t] outside
   [grad] of [loss] at [x] collects [expected], eagerly and compiled. *)
let counts_once ~msg t expected loss x =
  let run x =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> Rune.grad' loss x)
  in
  check_arr ~eps:1e-9 ~msg:(msg ^ ", eager") expected (snd (run x));
  check_arr ~eps:1e-9 ~msg:(msg ^ ", compiled") expected
    (snd (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) run x))

let remat f = Rune.remat Nx.Ptree.(tensor @-> returns tensor) f
let rows x = Nx.reshape [| Nx.numel x; 1 |] x

(* A scan adding the square of each element of [x]. *)
let adding_scan t x =
  snd
    (Rune.scan'
       ~f:(fun c r ->
         Rune.Total.add t (Nx.sum (Nx.mul r r));
         (c, r))
       ~init:(zero ()) (rows x))

(* Code that reverse mode reruns inside rerun code, under a tape of its own, is
   rerun too. *)
let test_rerun_inside_rerun_code () =
  let t = Rune.Total.make () and x = series 3 [| 4 |] in
  let expected = to_arr (Nx.sum (Nx.mul x x)) in
  counts_once ~msg:"a remat in a remat" t expected
    (fun x -> Nx.sum (remat (fun x -> Nx.sin (remat_adding t x)) x))
    x;
  counts_once ~msg:"a scan in a remat" t expected
    (fun x -> Nx.sum (remat (fun x -> Nx.sin (adding_scan t x)) x))
    x

(* Under no_grad, rerun code still runs a scan, a remat or a custom call where
   the rerun drops its additions. *)
let test_no_grad_in_rerun_code () =
  let t = Rune.Total.make () and x = series 3 [| 4 |] in
  let expected = to_arr (Nx.sum (Nx.mul x x)) in
  let untaped f x =
    ignore (Rune.no_grad (fun () -> f x));
    Nx.sin x
  in
  counts_once ~msg:"a scan in a remat" t expected
    (fun x -> Nx.sum (remat (untaped (adding_scan t)) x))
    x;
  counts_once ~msg:"a remat in a remat" t expected
    (fun x -> Nx.sum (remat (untaped (remat_adding t)) x))
    x;
  counts_once ~msg:"a remat in a scan" t expected
    (fun x ->
      Nx.sum
        (snd
           (Rune.scan'
              ~f:(fun c r -> (c, untaped (remat_adding t) r))
              ~init:(zero ()) (rows x))))
    x;
  let xs = series 5 [| 3; 4 |] in
  let tap_vjp x =
    Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
      ~fwd:(fun x ->
        Rune.Total.add t (Nx.sum (Nx.mul x x));
        (x, ()))
      ~bwd:(fun () g -> g)
      x
  in
  let tap_jvp x =
    Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
      ~f:(fun x ->
        Rune.Total.add t (Nx.sum (Nx.mul x x));
        x)
      ~jvp:(fun x dx -> (x, dx))
      x
  in
  List.iter
    (fun (msg, tap) ->
      let loss x = Nx.sum (remat (untaped tap) x) in
      let _, total =
        Rune.Total.collect t ~zero:(zero ()) (fun () ->
            Rune.vmap' (Rune.grad' loss) xs)
      in
      check_arr ~eps:1e-9 ~msg (to_arr (Nx.sum (Nx.mul xs xs))) total)
    [
      ("a custom vjp in a remat under a map", tap_vjp);
      ("a custom jvp in a remat under a map", tap_jvp);
    ]

(* A function reverse mode runs in its own context, a custom call's, runs under
   the rerun's drop. *)
let test_custom_call_in_rerun_code () =
  let t = Rune.Total.make () and x = series 3 [| 4 |] in
  let tap x =
    Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
      ~fwd:(fun x ->
        Rune.Total.add t (Nx.sum (Nx.mul x x));
        (Nx.sin x, x))
      ~bwd:(fun x g -> Nx.mul g (Nx.cos x))
      x
  in
  counts_once ~msg:"fwd in a remat" t
    (to_arr (Nx.sum (Nx.mul x x)))
    (fun x -> Nx.sum (remat tap x))
    x

(* Forward over reverse, reverse over reverse and a pullback run twice rerun
   code; each addition still counts once. *)
let test_higher_order () =
  let t = Rune.Total.make () and x = series 3 [| 4 |] in
  let expected = to_arr (Nx.sum (Nx.mul x x)) in
  let inner x = Nx.sum (remat_adding t x) in
  let collect f = snd (Rune.Total.collect t ~zero:(zero ()) f) in
  check_arr ~msg:"jvp of grad" expected
    (collect (fun () -> Rune.jvp' (Rune.grad' inner) x (Nx.ones_like x)));
  check_arr ~msg:"grad of grad" expected
    (collect (fun () ->
         Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.grad' inner x) x)) x));
  check_arr ~msg:"a pullback run twice" expected
    (collect (fun () ->
         let _, pullback = Rune.vjp_fun' (remat_adding t) x in
         ignore (pullback (Nx.ones_like x));
         ignore (pullback (Nx.ones_like x))));
  check_arr ~msg:"jacrev" expected
    (collect (fun () -> Rune.jacrev' (remat_adding t) x));
  check_arr ~msg:"jacfwd, a map over the columns"
    (to_arr (Nx.mul_s (Nx.sum (Nx.mul x x)) 4.0))
    (collect (fun () -> Rune.jacfwd' (remat_adding t) x))

(* The scope declines a scan no stager lies beyond: the scan folds where it was
   performed, under the key scope between them. *)
let test_key_scope_inside_a_scope () =
  let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
  let draws () =
    Nx.Rng.with_key (Nx.Rng.key 3) (fun () ->
        snd
          (Rune.scan'
             ~f:(fun c x ->
               let r = Nx.add x (Nx.rand f64 [| 3 |]) in
               Rune.Total.add t (Nx.sum r);
               (c, r))
             ~init:h0 xs))
  in
  let expected = draws () in
  let ys, total = Rune.Total.collect t ~zero:(zero ()) draws in
  check_arr ~msg:"seeded draws" (to_arr expected) ys;
  check_arr ~msg:"total" (to_arr (Nx.sum expected)) total

(* vmap over tangents around a scope around jvp of a scan, compiled: forward and
   vmap restart the step's trace to carry a tangent and a lane, and the
   restarted traces' additions are discarded. *)
let test_restarts_through_a_scope () =
  let t = Rune.Total.make () and k = 3 in
  let f runs xs dirs =
    Rune.vmap
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (fun d ->
        let (_, dy), total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.jvp' (fun w -> rollout ~runs t w h0 xs) w0 d)
        in
        (dy, total))
      dirs
  in
  staged ~runs:3 (fun runs n ->
      let xs = series 2 [| n; 3 |] and dirs = series 6 [| k; 3; 3 |] in
      let dy, total =
        Rune.jit
          Nx.Ptree.(tensor @-> returns (pair tensor tensor))
          (f runs xs) dirs
      in
      let dy', total' = f (ref 0) xs dirs in
      check_arr ~eps:1e-9 ~msg:"tangents" (to_arr dy') dy;
      check_arr ~eps:1e-9 ~msg:"eager totals"
        (to_arr (stack k (fun _ -> expected_total xs)))
        total';
      check_arr ~eps:1e-9 ~msg:"compiled totals" (to_arr total') total)

(* jit restarts a trace to place a scan's carry across the devices; the
   restarted traces' additions are discarded. *)
let test_placement_restarts () =
  let t = Rune.Total.make () in
  let devices = List.map Rune.device [ "CPU:1"; "CPU:2"; "CPU:3"; "CPU:4" ] in
  let traces = ref 0 in
  let step (a, b) x =
    incr traces;
    let a' = Nx.add (Nx.mul_s a 0.5) x in
    let b' = Nx.add (Nx.mul_s b 0.5) a in
    Rune.Total.add t (Nx.cast f64 (Nx.sum (Nx.mul a' b')));
    ((a', b'), Nx.mul a' b')
  in
  let f xs =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        let (a, b), ys =
          Rune.scan
            Nx.Ptree.(pair tensor tensor)
            Nx.Ptree.tensor Nx.Ptree.tensor ~f:step
            ~init:(Nx.zeros f32 [| 16 |], Nx.zeros f32 [| 16 |])
            xs
        in
        Nx.add (Nx.sum a) (Nx.add (Nx.sum b) (Nx.sum ys)))
  in
  let xs = Nx.cast f32 (series 2 [| 6; 16 |]) in
  let y, total = f xs in
  traces := 0;
  let y', total' =
    Rune.jit ~devices
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      f
      (Nx.place (Nx.Placement.sharded ~axis:1 devices) xs)
  in
  equal ~msg:"the trace restarted" bool true (!traces > 1);
  check_arr ~eps:1e-5 ~msg:"value" (to_arr y) y';
  check_arr ~eps:1e-5 ~msg:"total" (to_arr total) total'

(* Sketches. A second-order forward-mode optimizer measures a model along k
   directions: the loss, its tangents [C] and the Gauss-Newton matrix [Σ Yᵀ H
   Y], where [Y] holds the k tangents of a prediction and [H] the curvature of
   the little loss it feeds. The directions are lanes of a named map around jvp;
   a little loss marks its prediction with a unit-result custom_jvp whose rule
   gathers the lanes of the tangent and adds the block to a total the sketch
   collects inside the map. *)

let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let mark scale y =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit ~f:ignore
    ~jvp:(fun _ dy ->
      let ys = Rune.lanes directions dy in
      let rows t = Nx.reshape [| (Nx.shape t).(0); -1 |] t in
      let b = Nx.matmul (rows ys) (Nx.transpose (rows (Nx.mul_s ys scale))) in
      Rune.Total.add curvature (Nx.mul_s (Nx.add b (Nx.transpose b)) 0.5);
      ((), ()))
    y

let mse ~target y =
  mark (2.0 /. Float.of_int (Nx.numel y)) y;
  Nx.mean (Nx.square (Nx.sub y target))

let readout = series 9 [| 2; 3 |]

let model_loss ?(runs = ref 0) ?(loss = mse) xs targets w =
  let _, ls =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun h (x, target) ->
        incr runs;
        let h = cell w h x in
        (h, loss ~target (Nx.matmul readout h)))
      ~init:h0 (xs, targets)
  in
  Nx.sum ls

let sketch ?runs xs targets w dirs =
  let k = (Nx.shape dirs).(0) in
  let (l, c), ggn =
    Rune.vmap ~axis:directions
      Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
      (fun d ->
        Rune.Total.collect curvature
          ~zero:(Nx.zeros f64 [| k; k |])
          (fun () -> Rune.jvp' (model_loss ?runs xs targets) w d))
      dirs
  in
  (lane 0 l, c, lane 0 ggn)

(* The explicit reference: one jvp per direction of the stacked predictions,
   [Σ_t (2/m) Y_tᵢ · Y_tⱼ]. *)
let reference xs targets w dirs =
  let k = (Nx.shape dirs).(0) in
  let predictions w =
    snd
      (Rune.scan'
         ~f:(fun h x ->
           let h = cell w h x in
           (h, Nx.matmul readout h))
         ~init:h0 xs)
  in
  let plain ~target y = Nx.mean (Nx.square (Nx.sub y target)) in
  let ys = List.init k (fun i -> snd (Rune.jvp' predictions w (lane i dirs))) in
  let ggn =
    Nx.init f64 [| k; k |] (fun ij ->
        let yi = List.nth ys ij.(0) and yj = List.nth ys ij.(1) in
        Nx.item [] (Nx.mul_s (Nx.sum (Nx.mul yi yj)) (2.0 /. 2.0)))
  in
  let c =
    stack k (fun i ->
        snd (Rune.jvp' (model_loss ~loss:plain xs targets) w (lane i dirs)))
  in
  (model_loss ~loss:plain xs targets w, c, ggn)

let test_sketch () =
  let k = 4 in
  let check ~msg (l, c, ggn) (l', c', ggn') =
    check_arr ~eps:1e-9 ~msg:(msg ^ ": loss") (to_arr l) l';
    check_arr ~eps:1e-9 ~msg:(msg ^ ": C") (to_arr c) c';
    check_arr ~eps:1e-9 ~msg:(msg ^ ": GGN") (to_arr ggn) ggn'
  in
  staged ~runs:3 (fun runs n ->
      let compiled =
        Rune.jit
          Nx.Ptree.(
            tensor @-> tensor @-> tensor @-> tensor
            @-> returns (pair (pair tensor tensor) tensor))
          (fun xs targets w dirs ->
            let l, c, ggn = sketch ~runs xs targets w dirs in
            ((l, c), ggn))
      in
      List.iter
        (fun seed ->
          let xs = series seed [| n; 3 |]
          and targets = series (seed + 1) [| n; 2 |]
          and w = series (seed + 2) [| 3; 3 |]
          and dirs = series (seed + 3) [| k; 3; 3 |] in
          let expected = reference xs targets w dirs in
          check ~msg:"eager" expected (sketch xs targets w dirs);
          let (l, c), ggn = compiled xs targets w dirs in
          check ~msg:"compiled" expected (l, c, ggn))
        [ 2; 5 ])

let tests =
  [
    group "scopes"
      [
        test "no scope is inert" test_no_scope_is_inert;
        test "the innermost scope of a total collects" test_innermost_scope;
        test "a shape mismatch raises at the addition"
          test_shape_mismatch_raises;
        test "an exception leaves the scope" test_an_exception_leaves_the_scope;
        test "a caught exception keeps its additions"
          test_a_caught_exception_keeps_its_additions;
        test "a jit inside a scope runs eagerly" test_jit_inside_a_scope;
      ];
    group "scans and remats"
      [
        test "an eager scan counts each step" test_eager_scan;
        test "a staged scan counts each step, replayed" test_staged_scan;
        test "a remat" test_remat;
        test "a key scope inside a scope keeps its draws"
          test_key_scope_inside_a_scope;
        test "restarted traces discard their additions"
          test_restarts_through_a_scope;
        test "an exception reaches the performer"
          test_exceptions_reach_the_performer;
        test "placement restarts discard their additions"
          test_placement_restarts;
      ];
    group "maps"
      [
        test "an addition crossing a map is the loop's"
          test_addition_crossing_a_map;
        test "a scope inside a map collects per lane" test_scope_inside_a_map;
      ];
    group "differentiation"
      [ test "a total is differentiated" test_a_total_is_differentiated ];
    group "sketch" [ test "a marked loss's Gauss-Newton sketch" test_sketch ];
    group "reverse mode"
      [
        test "a scope outside grad counts once" test_grad_outside_scope;
        test "rerun code inside rerun code" test_rerun_inside_rerun_code;
        test "no_grad in rerun code" test_no_grad_in_rerun_code;
        test "a custom call in rerun code" test_custom_call_in_rerun_code;
        test "higher order" test_higher_order;
      ];
  ]

let () = exit (run "rune total" tests)
