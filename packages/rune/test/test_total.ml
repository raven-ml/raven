(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Write-only totals: a scope collects what code anywhere inside it adds,
   transformations pass the additions on, sum them over lanes or drop what they
   re-run, and scans and remats discharge through the scope. The last group
   builds the RFC's marked little loss: a custom_jvp with a unit result whose
   rule gathers the direction lanes and adds a curvature block to a total the
   sketch driver collects. *)

open Windtrap
open Rune_test_support.Support

let zero () = Nx.scalar f64 0.0

(* The scope *)

let test_collect_sums_additions () =
  let t = Rune.Total.make () in
  let r, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 2.0);
        Rune.Total.add t (Nx.scalar f64 3.5);
        7.0)
  in
  equal ~msg:"result" float_exact 7.0 r;
  check_arr ~msg:"total" [| 5.5 |] tot

let test_no_additions_is_zero () =
  let t = Rune.Total.make () in
  let r, tot = Rune.Total.collect t ~zero:(zero ()) (fun () -> 1.0) in
  equal ~msg:"result" float_exact 1.0 r;
  check_arr ~msg:"zero" [| 0.0 |] tot

let test_an_unhandled_addition_is_dropped () =
  let t = Rune.Total.make () in
  Rune.Total.add t (Nx.scalar f64 3.0);
  let _, tot = Rune.Total.collect t ~zero:(zero ()) (fun () -> ()) in
  check_arr ~msg:"nothing to collect" [| 0.0 |] tot

let test_the_innermost_collect_owns_the_addition () =
  let t = Rune.Total.make () in
  let inner, outer =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        let _, inner =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (Nx.scalar f64 2.0);
              Rune.Total.add t (Nx.scalar f64 3.0);
              ())
        in
        Rune.Total.add t (Nx.scalar f64 10.0);
        inner)
  in
  check_arr ~msg:"inner" [| 5.0 |] inner;
  check_arr ~msg:"outer" [| 11.0 |] outer

let test_another_total_passes_through () =
  let t1 = Rune.Total.make () and t2 = Rune.Total.make () in
  let t2_sum, t1_sum =
    Rune.Total.collect t1 ~zero:(zero ()) (fun () ->
        Rune.Total.add t1 (Nx.scalar f64 1.0);
        let _, t2_sum =
          Rune.Total.collect t2 ~zero:(zero ()) (fun () ->
              Rune.Total.add t2 (Nx.scalar f64 2.0);
              Rune.Total.add t1 (Nx.scalar f64 4.0))
        in
        t2_sum)
  in
  check_arr ~msg:"t1" [| 5.0 |] t1_sum;
  check_arr ~msg:"t2" [| 2.0 |] t2_sum

let test_shape_mismatch_raises_at_the_addition () =
  let t = Rune.Total.make () in
  raises
    (Invalid_argument
       "Rune.Total: the addition's shape [2] does not match the scope's zero []")
    (fun () ->
      ignore
        (Rune.Total.collect t ~zero:(zero ()) (fun () ->
             Rune.Total.add t (vec64 [| 1.0; 2.0 |]))))

let test_the_body_exception_is_delivered () =
  let t = Rune.Total.make () in
  raises Exit (fun () ->
      ignore
        (Rune.Total.collect t ~zero:(zero ()) (fun () ->
             Rune.Total.add t (Nx.scalar f64 1.0);
             raise Exit)))

let test_a_caught_exception_leaves_its_additions () =
  let t = Rune.Total.make () in
  let _, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.Total.add t (Nx.scalar f64 1.0);
        (try
           Rune.Total.add t (Nx.scalar f64 2.0);
           raise Exit
         with Exit -> ());
        Rune.Total.add t (Nx.scalar f64 4.0))
  in
  check_arr ~msg:"all three count" [| 7.0 |] tot

(* Transformations *)

let test_jvp_passes_the_addition_on () =
  let t = Rune.Total.make () in
  let x = vec64 [| 1.0; 2.0; 3.0 |] and v = vec64 [| 1.0; 1.0; 1.0 |] in
  (* The scope is outside the jvp: the addition crosses forward mode, which
     passes it on without a tangent. *)
  let (y, dy), tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.jvp'
          (fun x ->
            Rune.Total.add t (Nx.sum x);
            Nx.sum (Nx.mul x x))
          x v)
  in
  check_arr ~msg:"value" [| 14.0 |] y;
  check_arr ~msg:"tangent" [| 12.0 |] dy;
  check_arr ~msg:"total" [| 6.0 |] tot

let test_vmap_sums_the_lanes () =
  let t = Rune.Total.make () in
  let _, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.vmap'
          (fun x ->
            Rune.Total.add t x;
            x)
          (vec64 [| 1.0; 2.0; 3.0 |]))
  in
  check_arr ~msg:"sum of the lanes" [| 6.0 |] tot

let test_vmap_counts_a_constant_once_per_lane () =
  let t = Rune.Total.make () in
  let _, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.vmap'
          (fun x ->
            Rune.Total.add t (Nx.scalar f64 1.0);
            x)
          (vec64 [| 1.0; 2.0; 3.0 |]))
  in
  check_arr ~msg:"one per lane" [| 3.0 |] tot

let test_vmap_sums_lane_vectors () =
  let t = Rune.Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let _, tot =
    Rune.Total.collect t ~zero:(Nx.zeros f64 [| 2 |]) (fun () ->
        Rune.vmap'
          (fun row ->
            Rune.Total.add t row;
            row)
          xs)
  in
  check_arr ~msg:"elementwise sum" [| 9.0; 12.0 |] tot

let test_vmap_stacks_the_totals_of_a_scope_inside_it () =
  let t = Rune.Total.make () in
  let ys =
    Rune.vmap'
      (fun x ->
        snd
          (Rune.Total.collect t ~zero:(Nx.scalar f64 0.0) (fun () ->
               Rune.Total.add t x)))
      (vec64 [| 1.0; 2.0; 3.0 |])
  in
  check_arr ~msg:"one total per lane" [| 1.0; 2.0; 3.0 |] ys

let test_grad_passes_the_first_run_on () =
  let t = Rune.Total.make () in
  let x = vec64 [| 1.0; 2.0; 3.0 |] in
  let g, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.grad'
          (fun x ->
            Rune.Total.add t (Nx.sum x);
            Nx.sum (Nx.mul x x))
          x)
  in
  check_arr ~msg:"gradient" [| 2.0; 4.0; 6.0 |] g;
  check_arr ~msg:"total" [| 6.0 |] tot

let test_jit_inside_a_scope_runs_eagerly () =
  let t = Rune.Total.make () in
  let f =
    Rune.jit'
      ~devices:[ Rune.device "CPU" ]
      (fun x ->
        Rune.Total.add t x;
        Nx.mul_s x 2.0)
  in
  let y, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () -> f (Nx.scalar f64 3.0))
  in
  check_arr ~msg:"value" [| 6.0 |] y;
  check_arr ~msg:"total" [| 3.0 |] tot

let test_a_compiled_scope_returns_its_total () =
  let t = Rune.Total.make () in
  let f =
    Rune.jit'
      ~devices:[ Rune.device "CPU" ]
      (fun (x : Nx.float64_t) ->
        let r, tot =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.Total.add t (Nx.sum x);
              Nx.sum (Nx.mul x x))
        in
        Nx.add r tot)
  in
  check_arr ~msg:"compiled total" [| 20.0 |] (f (vec64 [| 1.0; 2.0; 3.0 |]))

(* Scan and remat discharge *)

let scan_sum t xs =
  snd
    (Rune.scan' ~init:(Nx.scalar f64 0.0)
       ~f:(fun c x ->
         let c = Nx.add c x in
         Rune.Total.add t c;
         (c, c))
       xs)

let test_scan_discharges_silently () =
  let t = Rune.Total.make () in
  let ys, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        scan_sum t (vec64 [| 1.0; 2.0; 3.0 |]))
  in
  check_arr ~msg:"carries" [| 1.0; 3.0; 6.0 |] ys;
  check_arr ~msg:"total" [| 10.0 |] tot

let test_scan_discharges_through_a_staged_loop () =
  let t = Rune.Total.make () in
  let f =
    Rune.jit'
      ~devices:[ Rune.device "CPU" ]
      (fun (xs : Nx.float64_t) ->
        let ys, tot =
          Rune.Total.collect t ~zero:(zero ()) (fun () -> scan_sum t xs)
        in
        Nx.add (Nx.sum ys) tot)
  in
  check_arr ~msg:"sums and total" [| 20.0 |] (f (vec64 [| 1.0; 2.0; 3.0 |]))

let test_a_staged_backward_step_adds_once () =
  let t = Rune.Total.make () in
  let xs = vec64 [| 1.0; 2.0; 3.0 |] in
  let loss p =
    Nx.sum
      (snd
         (Rune.scan' ~init:p
            ~f:(fun c x ->
              Rune.Total.add t (Nx.sum c);
              (Nx.add c x, Nx.sum c))
            xs))
  in
  let f =
    Rune.jit'
      ~devices:[ Rune.device "CPU" ]
      (fun (p : Nx.float64_t) ->
        let (l, g), tot =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.value_and_grad' loss p)
        in
        Nx.stack ~axis:0 [ l; tot; Nx.sum g ])
  in
  (* The forward fold adds the incoming carry at each step: p, p + 1, p + 3. At
     [p = 0] the loss is 4, the total 4 and the gradient sums to 3; the backward
     loop re-runs the step, whose additions must be dropped. *)
  check_arr ~msg:"loss, total, gradient" [| 4.0; 4.0; 3.0 |]
    (f (Nx.scalar f64 0.0))

let test_remat_discharges () =
  let t = Rune.Total.make () in
  let f x =
    Rune.Total.add t (Nx.sum x);
    Nx.sum (Nx.mul x x)
  in
  let y, tot =
    Rune.Total.collect t ~zero:(zero ()) (fun () ->
        Rune.remat
          Nx.Ptree.(tensor @-> returns tensor)
          f
          (vec64 [| 1.; 2.; 3. |]))
  in
  check_arr ~msg:"value" [| 14.0 |] y;
  check_arr ~msg:"total" [| 6.0 |] tot

let test_a_recomputed_remat_adds_once () =
  let t = Rune.Total.make () in
  let f p =
    Rune.Total.add t (Nx.sum p);
    Nx.sum (Nx.mul p p)
  in
  let f =
    Rune.jit'
      ~devices:[ Rune.device "CPU" ]
      (fun (p : Nx.float64_t) ->
        let (l, g), tot =
          Rune.Total.collect t ~zero:(zero ()) (fun () ->
              Rune.value_and_grad'
                (fun p -> Rune.remat Nx.Ptree.(tensor @-> returns tensor) f p)
                p)
        in
        Nx.stack ~axis:0 [ l; tot; Nx.sum g ])
  in
  (* The forward run adds sum p once; the recomputation must not add again. *)
  check_arr ~msg:"loss, total, gradient" [| 14.0; 6.0; 12.0 |]
    (f (vec64 [| 1.0; 2.0; 3.0 |]))

(* The marked little loss *)

let test_marked_loss_reports_curvature () =
  let directions = Rune.axis () in
  let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make () in
  let mark curv y =
    Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit ~f:ignore y ~jvp:(fun _ dy ->
        let ys = Rune.lanes directions dy in
        let rows t = Nx.reshape [| Nx.dim 0 t; -1 |] t in
        let b = Nx.matmul (rows ys) (Nx.transpose (rows (Nx.mul_s ys curv))) in
        Rune.Total.add curvature (Nx.mul_s (Nx.add b (Nx.transpose b)) 0.5);
        ((), ()))
  in
  let p = vec64 [| 0.3; -0.7; 1.1 |] in
  let x = mat64 4 3 [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1.; 1.; 1.; 0. |] in
  let t = vec64 [| 0.1; -0.2; 0.4; 1.0 |] in
  let k = 2 in
  let dirs = mat64 2 3 [| 1.; 0.; 0.; 0.; 1.; 0. |] in
  let loss p =
    let y = Nx.matmul x p in
    mark 0.5 y;
    Nx.mean (Nx.square (Nx.sub y t))
  in
  let sketch () =
    Rune.vmap ~axis:directions
      Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
      (fun dir ->
        Rune.Total.collect curvature
          ~zero:(Nx.zeros f64 [| k; k |])
          (fun () -> Rune.jvp Nx.Ptree.tensor Nx.Ptree.tensor loss p dir))
      dirs
  in
  let (l, c), ggn = sketch () in
  (* The loss and the block are constants of the direction map; C is the jvp's
     tangent of the loss, one lane per direction. *)
  let l = Nx.slice [ Nx.I 0 ] l and ggn = Nx.slice [ Nx.I 0 ] ggn in
  (* The oracle: loss = mean (x p - t)², the gradient sketch C = Θ∇L, and the
     Gauss-Newton block Y H Yᵀ with Y = Θ xᵀ and H = 0.5. *)
  let res = Nx.sub (Nx.matmul x p) t in
  let grad = Nx.mul_s (Nx.matmul (Nx.transpose x) res) (2.0 /. 4.0) in
  let yk = Nx.matmul dirs (Nx.transpose x) in
  check_arr ~msg:"loss" [| 0.685 |] l;
  check_arr ~msg:"C" (to_arr (Nx.matmul dirs grad)) c;
  check_arr ~msg:"ggn"
    (to_arr (Nx.mul_s (Nx.matmul yk (Nx.transpose yk)) 0.5))
    ggn;
  (* The mark is inert under grad: the same model trains with any optimizer. *)
  check_arr ~msg:"gradient" (to_arr grad) (Rune.grad' loss p);
  (* The same sketch compiles, scans and all. *)
  let compiled =
    Rune.jit
      ~devices:[ Rune.device "CPU" ]
      Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
      (fun p ->
        Rune.vmap ~axis:directions
          Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
          (fun dir ->
            Rune.Total.collect curvature
              ~zero:(Nx.zeros f64 [| k; k |])
              (fun () -> Rune.jvp Nx.Ptree.tensor Nx.Ptree.tensor loss p dir))
          dirs)
  in
  let (lc, cc), ggnc = compiled p in
  check_arr ~msg:"compiled loss" (to_arr l) (Nx.slice [ Nx.I 0 ] lc);
  check_arr ~msg:"compiled C" (to_arr c) cc;
  check_arr ~msg:"compiled ggn" (to_arr ggn) (Nx.slice [ Nx.I 0 ] ggnc)

(* The RFC's mark inside the user's own map over examples: vmap passes the
   custom_jvp on and the block sums over the example lanes. *)
let test_marked_loss_inside_the_users_own_map () =
  let directions = Rune.axis () in
  let curvature : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make () in
  let mark curv y =
    Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.unit ~f:ignore y ~jvp:(fun _ dy ->
        let ys = Rune.lanes directions dy in
        let rows t = Nx.reshape [| Nx.dim 0 t; -1 |] t in
        let b = Nx.matmul (rows ys) (Nx.transpose (rows (Nx.mul_s ys curv))) in
        Rune.Total.add curvature (Nx.mul_s (Nx.add b (Nx.transpose b)) 0.5);
        ((), ()))
  in
  let p = vec64 [| 0.3; -0.7; 1.1 |] in
  let dirs = mat64 2 3 [| 1.; 0.; 0.; 0.; 1.; 0. |] in
  let k = 2 in
  let data = mat64 4 2 [| 1.; 0.1; 0.; -0.2; 0.; 0.4; 1.; 1.0 |] in
  let loss p =
    Nx.sum
      (Rune.vmap'
         (fun row ->
           let xi = Nx.get [ 0 ] row and ti = Nx.get [ 1 ] row in
           let y = Nx.mul p xi in
           mark 2.0 y;
           Nx.sum (Nx.square (Nx.sub y ti)))
         data)
  in
  let (l, c), ggn =
    Rune.vmap ~axis:directions
      Nx.Ptree.(tensor @-> returns (pair (pair tensor tensor) tensor))
      (fun dir ->
        Rune.Total.collect curvature
          ~zero:(Nx.zeros f64 [| k; k |])
          (fun () -> Rune.jvp Nx.Ptree.tensor Nx.Ptree.tensor loss p dir))
      dirs
  in
  let l = Nx.slice [ Nx.I 0 ] l and ggn = Nx.slice [ Nx.I 0 ] ggn in
  (* Σᵢ Σⱼ (pⱼ xᵢ - tᵢ)²: ∇L = 2 Σᵢ xᵢ² p - 2 (Σᵢ xᵢ tᵢ) 1, H = 2 I, and Y = Θ
     xᵀ, so the block is 2 (Σᵢ xᵢ²) Θ Θᵀ. *)
  let x = Nx.slice [ Nx.A; Nx.I 0 ] data
  and t = Nx.slice [ Nx.A; Nx.I 1 ] data in
  let sxx = Nx.sum (Nx.mul x x) and sxt = Nx.sum (Nx.mul x t) in
  let grad =
    Nx.sub
      (Nx.mul_s p (2.0 *. scalar sxx))
      (Nx.full f64 [| 3 |] (2.0 *. scalar sxt))
  in
  let yk = Nx.matmul dirs (Nx.transpose dirs) in
  check_arr ~msg:"loss" [| 5.67 |] l;
  check_arr ~msg:"C" (to_arr (Nx.matmul dirs grad)) c;
  check_arr ~msg:"ggn" (to_arr (Nx.mul_s yk (2.0 *. scalar sxx))) ggn

let tests =
  [
    group "scope"
      [
        test "collect sums additions" test_collect_sums_additions;
        test "no additions is zero" test_no_additions_is_zero;
        test "an unhandled addition is dropped"
          test_an_unhandled_addition_is_dropped;
        test "the innermost collect owns the addition"
          test_the_innermost_collect_owns_the_addition;
        test "another total passes through" test_another_total_passes_through;
        test "a shape mismatch raises at the addition"
          test_shape_mismatch_raises_at_the_addition;
        test "the body exception is delivered"
          test_the_body_exception_is_delivered;
        test "a caught exception leaves its additions"
          test_a_caught_exception_leaves_its_additions;
      ];
    group "transformations"
      [
        test "jvp passes the addition on" test_jvp_passes_the_addition_on;
        test "vmap sums the lanes" test_vmap_sums_the_lanes;
        test "vmap counts a constant once per lane"
          test_vmap_counts_a_constant_once_per_lane;
        test "vmap sums lane vectors" test_vmap_sums_lane_vectors;
        test "vmap stacks the totals of a scope inside it"
          test_vmap_stacks_the_totals_of_a_scope_inside_it;
        test "grad passes the first run on" test_grad_passes_the_first_run_on;
        test "jit inside a scope runs eagerly"
          test_jit_inside_a_scope_runs_eagerly;
        test "a compiled scope returns its total"
          test_a_compiled_scope_returns_its_total;
      ];
    group "discharge"
      [
        test "a scan discharges silently" test_scan_discharges_silently;
        test "a scan discharges through a staged loop"
          test_scan_discharges_through_a_staged_loop;
        test "a staged backward step adds once"
          test_a_staged_backward_step_adds_once;
        test "a remat discharges" test_remat_discharges;
        test "a recomputed remat adds once" test_a_recomputed_remat_adds_once;
      ];
    group "marked loss"
      [
        test "reports curvature" test_marked_loss_reports_curvature;
        test "reports curvature under the user's own map"
          test_marked_loss_inside_the_users_own_map;
      ];
  ]

let () = exit (run "rune total" tests)
