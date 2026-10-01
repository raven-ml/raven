(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Totals. The oracle is the plain program: an addition counts once per
   execution of the code that makes it, under a scope, a scan, a remat, a map
   (as the loop over its lanes) and reverse mode, which runs code again. The
   compiled cases that compile most are slow. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-12 ~abs:1e-12 ()

let series seed shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f64 shape
    (Array.init n (fun i -> Float.sin (Float.of_int ((7 * i) + seed)) /. 2.))

let w0 () = series 1 [| 3; 3 |]
let h0 () = vec [| 0.1; -0.2; 0.3 |]
let lane i x = Nx.get [ i ] x
let stack n f = Nx.stack (List.init n f)
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

let expected_total xs =
  Nx.sum (rollout ~add:(fun _ _ -> ()) (Rune.Total.make ()) (w0 ()) (h0 ()) xs)

(* Scopes *)

let scope_tests =
  [
    test "with no scope an addition does nothing" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
        let ys = rollout t (w0 ()) (h0 ()) xs in
        let _, total = Rune.Total.collect t ~zero:(zero ()) (fun () -> ()) in
        equal ~msg:"no addition" (exact ()) (zero ()) total;
        equal ~msg:"the result" (exact ())
          (rollout ~add:(fun _ _ -> ()) t (w0 ()) (h0 ()) xs)
          ys);
    test "the innermost scope of a total collects" (fun () ->
        let t = Rune.Total.make () and u = Rune.Total.make () in
        let (inner, u_total), outer =
          Rune.Total.collect t ~zero:(scalar 10.) (fun () ->
              Rune.Total.add t (scalar 1.);
              let _, inner =
                Rune.Total.collect t ~zero:(zero ()) (fun () ->
                    Rune.Total.add t (scalar 2.))
              in
              let _, u_total =
                Rune.Total.collect u ~zero:(zero ()) (fun () ->
                    Rune.Total.add t (scalar 4.);
                    Rune.Total.add u (scalar 8.))
              in
              (inner, u_total))
        in
        equal ~msg:"inner" (exact ()) (scalar 2.) inner;
        equal ~msg:"another total" (exact ()) (scalar 8.) u_total;
        equal ~msg:"outer, from its zero" (exact ()) (scalar 15.) outer);
    test "an addition of another shape is refused where it is made" (fun () ->
        let t = Rune.Total.make () in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (scalar 1.);
              raises_match (Exn.invalid_arg ~substring:"Rune.Total.add")
                (fun () -> Rune.Total.add t (h0 ())))
        in
        equal ~msg:"an addition before the caught refusal counts" (exact ())
          (scalar 1.) total);
    test "an exception leaves the scope" (fun () ->
        let t = Rune.Total.make () in
        raises Exit (fun () ->
            Rune.Total.collect t ~zero:(zero ()) (fun () ->
                Rune.Total.add t (scalar 1.);
                raise Exit)));
    test "a caught exception keeps its additions" (fun () ->
        let t = Rune.Total.make () in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (scalar 1.);
              (try
                 Rune.Total.add t (scalar 2.);
                 raise Exit
               with Exit -> ());
              Rune.Total.add t (scalar 4.))
        in
        equal (exact ()) (scalar 7.) total);
    test "a jit inside a scope runs eagerly and its additions count" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
        let f = Rune.jit' (rollout t (w0 ()) (h0 ())) in
        let _, total = Rune.Total.collect t ~zero:(zero ()) (fun () -> f xs) in
        equal (close ()) (expected_total xs) total);
  ]

(* Scans and remats *)

let remat_adding t =
  Rune.remat
    Nx.Ptree.(tensor @-> returns tensor)
    (fun x ->
      Rune.Total.add t (Nx.sum (Nx.mul x x));
      Nx.sin x)

let scan_tests =
  [
    test "a scan counts each step once" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 5; 3 |] in
        let runs = ref 0 in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              rollout ~runs t (w0 ()) (h0 ()) xs)
        in
        equal ~msg:"total" (close ()) (expected_total xs) total;
        equal ~msg:"body runs" int 5 !runs);
    slow "a staged scan counts each step, replayed" (fun () ->
        let t = Rune.Total.make () in
        let f =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (pair tensor tensor))
            (fun xs ->
              Rune.Total.collect t ~zero:(zero ()) (fun () ->
                  rollout t (w0 ()) (h0 ()) xs))
        in
        List.iter
          (fun seed ->
            let xs = series seed [| 4; 3 |] in
            equal (close ()) (expected_total xs) (snd (f xs)))
          [ 2; 3 ]);
    test "a remat counts its addition once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let y, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () -> remat_adding t x)
        in
        equal ~msg:"result" (exact ()) (Nx.sin x) y;
        equal ~msg:"total" (close ()) (Nx.sum (Nx.mul x x)) total);
    test "a key scope inside a scope keeps a scan's draws" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
        let draws () =
          Nx.Rng.with_key (Nx.Rng.key 3) (fun () ->
              snd
                (Rune.scan'
                   ~f:(fun c x ->
                     let r = Nx.add x (Nx.rand f64 [| 3 |]) in
                     Rune.Total.add t (Nx.sum r);
                     (c, r))
                   ~init:(h0 ()) xs))
        in
        let expected = draws () in
        let ys, total = Rune.Total.collect t ~zero:(zero ()) draws in
        equal ~msg:"draws" (exact ()) expected ys;
        equal ~msg:"total" (close ()) (Nx.sum expected) total);
    test "an exception of a scan or a remat reaches its call inside the scope"
      (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
        let within f =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (scalar 1.);
              let r = match f () with _ -> 1. | exception Exit -> 2. in
              Rune.Total.add t (scalar r);
              r)
        in
        let check ~msg (r, total) =
          equal ~msg:(msg ^ ": caught") float_exact 2. r;
          equal ~msg:(msg ^ ": total") (exact ()) (scalar 3.) total
        in
        check ~msg:"scan"
          (within (fun () ->
               Rune.scan' ~f:(fun _ _ -> raise Exit) ~init:(h0 ()) xs));
        check ~msg:"remat"
          (within (fun () ->
               Rune.remat
                 Nx.Ptree.(tensor @-> returns tensor)
                 (fun _ -> raise Exit)
                 (lane 0 xs))));
    test "restarted traces discard their additions" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 4; 3 |] in
        let dirs = series 6 [| 3; 3; 3 |] in
        let f dirs =
          Rune.vmap
            Nx.Ptree.(tensor @-> returns (pair tensor tensor))
            (fun d ->
              let (_, dy), total =
                Rune.Total.collect t ~zero:(zero ()) (fun () ->
                    Rune.jvp' (fun w -> rollout t w (h0 ()) xs) (w0 ()) d)
              in
              (dy, total))
            dirs
        in
        let dy, total = f dirs in
        let dy', total' =
          Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) f dirs
        in
        equal ~msg:"tangents" (close ()) dy dy';
        equal ~msg:"totals" (close ()) total total');
  ]

(* Maps *)

let map_tests =
  [
    test "an addition crossing a map is the loop's" (fun () ->
        let t = Rune.Total.make () and xs = series 4 [| 4; 3 |] in
        let _, total =
          Rune.Total.collect t ~zero:(Nx.zeros f64 [| 3 |]) (fun () ->
              Rune.vmap'
                (fun x ->
                  Rune.Total.add t (Nx.mul x x);
                  Rune.Total.add t (h0 ());
                  x)
                xs)
        in
        let loop =
          List.fold_left
            (fun acc i ->
              Nx.add acc (Nx.add (Nx.mul (lane i xs) (lane i xs)) (h0 ())))
            (Nx.zeros f64 [| 3 |]) [ 0; 1; 2; 3 ]
        in
        equal (close ()) loop total);
    test "an addition crossing a map over a scan is the loop's" (fun () ->
        let t = Rune.Total.make () and xss = series 5 [| 4; 6; 3 |] in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.vmap' (fun xs -> rollout t (w0 ()) (h0 ()) xs) xss)
        in
        let loop =
          List.fold_left
            (fun acc i -> Nx.add acc (expected_total (lane i xss)))
            (zero ()) [ 0; 1; 2; 3 ]
        in
        equal (close ()) loop total);
    test "a scope inside a map collects per lane" (fun () ->
        let t = Rune.Total.make () and xss = series 5 [| 3; 6; 3 |] in
        let totals =
          Rune.vmap'
            (fun xs ->
              snd
                (Rune.Total.collect t ~zero:(zero ()) (fun () ->
                     rollout t (w0 ()) (h0 ()) xs)))
            xss
        in
        equal (close ()) (stack 3 (fun i -> expected_total (lane i xss))) totals);
  ]

(* Differentiation *)

let collected t xs w =
  snd (Rune.Total.collect t ~zero:(zero ()) (fun () -> rollout t w (h0 ()) xs))

let explicit xs w =
  Nx.sum (rollout ~add:(fun _ _ -> ()) (Rune.Total.make ()) w (h0 ()) xs)

let differentiation_tests =
  [
    test "a collected total is differentiated as a value" (fun () ->
        let t = Rune.Total.make () in
        let xs = series 2 [| 6; 3 |] and dw = series 4 [| 3; 3 |] in
        let v, d = Rune.jvp' (explicit xs) (w0 ()) dw in
        let v', d' = Rune.jvp' (collected t xs) (w0 ()) dw in
        equal ~msg:"value" (close ()) v v';
        equal ~msg:"tangent" (close ()) d d';
        equal ~msg:"gradient" (close ())
          (Rune.grad' (explicit xs) (w0 ()))
          (Rune.grad' (collected t xs) (w0 ())));
    slow "a collected total is differentiated as a value, compiled" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 6; 3 |] in
        equal (close ())
          (Rune.grad' (explicit xs) (w0 ()))
          (Rune.jit' (Rune.grad' (collected t xs)) (w0 ())));
  ]

(* Code that runs again *)

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

(* The total a scope of [t] outside [grad] of [loss] at [x] collects. *)
let around_grad t loss x =
  snd (Rune.Total.collect t ~zero:(zero ()) (fun () -> Rune.grad' loss x))

let squares x = Nx.sum (Nx.mul x x)

let again_tests =
  [
    test "a scope outside grad counts a scan's additions once" (fun () ->
        let t = Rune.Total.make () and xs = series 2 [| 5; 3 |] in
        let loss ?add w = Nx.sum (rollout ?add t w (h0 ()) xs) in
        let g, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.grad' loss (w0 ()))
        in
        equal ~msg:"gradient" (close ())
          (Rune.grad' (loss ~add:(fun _ _ -> ())) (w0 ()))
          g;
        equal ~msg:"total" (close ()) (expected_total xs) total);
    test "a scope outside grad counts a remat's addition once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        equal (close ()) (squares x)
          (around_grad t (fun x -> Nx.sum (remat_adding t x)) x));
    test "a remat in a remat counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        equal (close ()) (squares x)
          (around_grad t
             (fun x -> Nx.sum (remat (fun x -> Nx.sin (remat_adding t x)) x))
             x));
    test "a scan in a remat counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        equal (close ()) (squares x)
          (around_grad t
             (fun x -> Nx.sum (remat (fun x -> Nx.sin (adding_scan t x)) x))
             x));
    test "a custom_vjp rule in a remat counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let tap =
          Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
              Rune.Total.add t (squares x);
              (Nx.sin x, fun g -> Nx.mul g (Nx.cos x)))
        in
        equal (close ()) (squares x)
          (around_grad t (fun x -> Nx.sum (remat tap x)) x));
    test "a custom_jvp rule with no tensor result in a remat counts once"
      (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let tap =
          Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit (fun x ->
              Rune.Total.add t (squares x);
              ((), fun _ -> ()))
        in
        equal (close ()) (squares x)
          (around_grad t
             (fun x ->
               Nx.sum
                 (remat
                    (fun x ->
                      tap x;
                      Nx.sin x)
                    x))
             x));
    test "forward over reverse counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let inner x = Nx.sum (remat_adding t x) in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.jvp' (Rune.grad' inner) x (Nx.ones_like x))
        in
        equal (close ()) (squares x) total);
    test "reverse over reverse counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let inner x = Nx.sum (remat_adding t x) in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.grad' (fun x -> Nx.sum (Nx.mul (Rune.grad' inner x) x)) x)
        in
        equal (close ()) (squares x) total);
    test "a pullback applied twice counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              let _, pullback = Rune.vjp' (remat_adding t) x in
              ignore (pullback (Nx.ones_like x));
              ignore (pullback (Nx.ones_like x)))
        in
        equal (close ()) (squares x) total);
    test "jacrev' counts once" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.jacrev' (remat_adding t) x)
        in
        equal (close ()) (squares x) total);
    test "jacfwd', a map over the columns, counts once per column" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let _, total =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.jacfwd' (remat_adding t) x)
        in
        equal (close ()) (Nx.mul_s (squares x) 4.) total);
    slow "a scope outside grad counts once, compiled" (fun () ->
        let t = Rune.Total.make () and x = series 3 [| 4 |] in
        let run x =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.grad' (fun x -> Nx.sum (remat_adding t x)) x)
        in
        equal (close ()) (squares x)
          (snd
             (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) run x)));
  ]

(* Sketches. A second-order forward-mode optimiser measures a model along k
   directions: the loss, its tangents C and the Gauss-Newton matrix Σ Yᵀ H Y,
   where Y holds the k tangents of a prediction and H the curvature of the
   little loss it feeds. The directions are lanes of a named map around jvp; a
   little loss marks its prediction with a custom_jvp with no tensor result,
   whose tangent map gathers the lanes of the tangent and adds the block to a
   total the sketch collects inside the map. *)

let directions = Rune.axis ()
let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let mark scale y =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit
    (fun _ ->
      ( (),
        fun dy ->
          let ys = Rune.lanes directions dy in
          let rows t = Nx.reshape [| (Nx.shape t).(0); -1 |] t in
          let b =
            Nx.matmul (rows ys) (Nx.transpose (rows (Nx.mul_s ys scale)))
          in
          Rune.Total.add curvature (Nx.mul_s (Nx.add b (Nx.transpose b)) 0.5) ))
    y

let mse ~target y =
  mark (2. /. Float.of_int (Nx.numel y)) y;
  Nx.mean (Nx.square (Nx.sub y target))

let plain_mse ~target y = Nx.mean (Nx.square (Nx.sub y target))
let readout () = series 9 [| 2; 3 |]

let model_loss ?(loss = mse) xs targets w =
  let _, ls =
    Rune.scan Nx.Ptree.tensor
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor
      ~f:(fun h (x, target) ->
        let h = cell w h x in
        (h, loss ~target (Nx.matmul (readout ()) h)))
      ~init:(h0 ()) (xs, targets)
  in
  Nx.sum ls

let sketch loss w dirs =
  let k = (Nx.shape dirs).(0) in
  let (l, c), ggn =
    Rune.vmap ~axis:directions
      Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
      (fun d ->
        Rune.Total.collect curvature
          ~zero:(Nx.zeros f64 [| k; k |])
          (fun () -> Rune.jvp' loss w d))
      dirs
  in
  (lane 0 l, c, lane 0 ggn)

let check_sketch (l, c, ggn) (l', c', ggn') =
  equal ~msg:"loss" (close ()) l l';
  equal ~msg:"C" (close ()) c c';
  equal ~msg:"GGN" (close ()) ggn ggn'

let sketch_tests =
  let k = 4 in
  let xs () = series 2 [| 5; 3 |] and targets () = series 3 [| 5; 2 |] in
  let w () = series 4 [| 3; 3 |] and dirs () = series 5 [| k; 3; 3 |] in
  [
    test "a marked loss's Gauss-Newton sketch" (fun () ->
        let predictions w =
          snd
            (Rune.scan'
               ~f:(fun h x ->
                 let h = cell w h x in
                 (h, Nx.matmul (readout ()) h))
               ~init:(h0 ()) (xs ()))
        in
        let ys =
          List.init k (fun i ->
              snd (Rune.jvp' predictions (w ()) (lane i (dirs ()))))
        in
        (* Σ_t (2/m) Y_tᵢ · Y_tⱼ, with m = 2 outputs per step. *)
        let ggn =
          Nx.init f64 [| k; k |] (fun ij ->
              Nx.item []
                (Nx.sum (Nx.mul (List.nth ys ij.(0)) (List.nth ys ij.(1)))))
        in
        let plain = model_loss ~loss:plain_mse (xs ()) (targets ()) in
        let c =
          stack k (fun i -> snd (Rune.jvp' plain (w ()) (lane i (dirs ()))))
        in
        check_sketch
          (plain (w ()), c, ggn)
          (sketch (model_loss (xs ()) (targets ())) (w ()) (dirs ())));
    test "a marked model trains under grad" (fun () ->
        let l, g =
          Rune.value_and_grad' (model_loss (xs ()) (targets ())) (w ())
        in
        let l', g' =
          Rune.value_and_grad'
            (model_loss ~loss:plain_mse (xs ()) (targets ()))
            (w ())
        in
        equal ~msg:"loss" (exact ()) l' l;
        equal ~msg:"gradient" (exact ()) g' g);
    test "a mark inside the model's own map" (fun () ->
        let k = 3 and b = 5 in
        let xs = series 2 [| b; 3 |] and targets = series 3 [| b; 2 |] in
        let w = series 4 [| 3; 3 |] and dirs = series 5 [| k; 3; 3 |] in
        let batch_loss w =
          Nx.sum
            (Rune.vmap
               Nx.Ptree.(tensor @-> tensor @-> returns tensor)
               (fun x target ->
                 mse ~target (Nx.matmul (readout ()) (cell w (h0 ()) x)))
               xs targets)
        in
        let predictions w =
          Rune.vmap' (fun x -> Nx.matmul (readout ()) (cell w (h0 ()) x)) xs
        in
        let ys =
          List.init k (fun i -> snd (Rune.jvp' predictions w (lane i dirs)))
        in
        let ggn =
          Nx.init f64 [| k; k |] (fun ij ->
              Nx.item []
                (Nx.sum (Nx.mul (List.nth ys ij.(0)) (List.nth ys ij.(1)))))
        in
        let c = stack k (fun i -> snd (Rune.jvp' batch_loss w (lane i dirs))) in
        check_sketch (batch_loss w, c, ggn) (sketch batch_loss w dirs));
    slow "a marked loss's sketch, compiled" (fun () ->
        let f =
          Rune.jit
            Nx.Ptree.(
              tensor @-> tensor @-> returns (pair (pair tensor tensor) tensor))
            (fun w dirs ->
              let l, c, ggn = sketch (model_loss (xs ()) (targets ())) w dirs in
              ((l, c), ggn))
        in
        let (l, c), ggn = f (w ()) (dirs ()) in
        check_sketch
          (sketch (model_loss (xs ()) (targets ())) (w ()) (dirs ()))
          (l, c, ggn));
  ]

let () =
  exit
    (run "Rune totals"
       [
         group "scopes" scope_tests;
         group "scans and remats" scan_tests;
         group "maps" map_tests;
         group "differentiation" differentiation_tests;
         group "code that runs again" again_tests;
         group "sketches" sketch_tests;
       ])
