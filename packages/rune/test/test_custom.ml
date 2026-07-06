(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* User-defined differentiation rules: the rule replaces autodiff in its own
   mode, the mismatched mode raises, undifferentiated calls run the forward
   function, and rules compose with vmap. *)

open Windtrap
open Rune_test_support.Support

(* A single-tensor parameter structure. *)
module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

(* sin with its correct custom rule; the residual saves the input. *)
let my_sin x =
  Rune.custom_vjp
    (module Single)
    ~fwd:(fun x -> (Nx.sin x, x))
    ~bwd:(fun x ct -> Nx.mul ct (Nx.cos x))
    x

(* A deliberately wrong rule, to prove the rule is used, not autodiff. *)
let fake_grad_sin x =
  Rune.custom_vjp
    (module Single)
    ~fwd:(fun x -> (Nx.sin x, ()))
    ~bwd:(fun () ct -> Nx.mul_s ct 100.0)
    x

let v3 () = vec64 [| 0.7; -1.3; 2.1 |]

(* custom_vjp *)

let test_rule_replaces_autodiff () =
  let g = Rune.grad' (fun x -> Nx.sum (fake_grad_sin x)) (v3 ()) in
  check_arr ~msg:"fake rule" [| 100.0; 100.0; 100.0 |] g

let test_correct_rule_matches_autodiff () =
  let x = v3 () in
  let expected = Rune.grad' (fun x -> Nx.sum (Nx.sin x)) x in
  let actual = Rune.grad' (fun x -> Nx.sum (my_sin x)) x in
  check_arr ~msg:"my_sin grad" (to_arr expected) actual

let test_rule_composes_in_a_graph () =
  (* d/dx sum(my_sin(x)²) = 2 sin x cos x; the rule sits between traced
     operations. *)
  let x = v3 () in
  let f x =
    let s = my_sin x in
    Nx.sum (Nx.mul s s)
  in
  let g = Rune.grad' f x in
  let expected = to_arr (Nx.mul_s (Nx.mul (Nx.sin x) (Nx.cos x)) 2.0) in
  check_arr ~msg:"chain" expected g

let test_undifferentiated_call_runs_fwd () =
  check_arr ~msg:"value" (to_arr (Nx.sin (v3 ()))) (my_sin (v3 ()))

let test_constants_pass_through () =
  (* The custom function applied to a constant inside grad neither raises nor
     contributes. *)
  let x = vec64 [| 2.0 |] in
  let c = v3 () in
  let f x = Nx.mul (Nx.sum (Nx.mul x x)) (Nx.sum (my_sin c)) in
  let g = Rune.grad' f x in
  check_arr ~msg:"dx" [| 2.0 *. 2.0 *. scalar (Nx.sum (Nx.sin c)) |] g

let test_multi_leaf_rule () =
  (* f(a, b) = a * b with a hand-written product rule through a Pair. *)
  let f p =
    Rune.custom_vjp
      (module Pair)
      ~fwd:(fun p -> (Nx.mul p.fst p.snd, p))
      ~bwd:(fun p ct -> { fst = Nx.mul ct p.snd; snd = Nx.mul ct p.fst })
      p
  in
  let a = v3 () and b = vec64 [| 1.9; 0.8; -0.6 |] in
  let g =
    Rune.grad (module Pair) (fun p -> Nx.sum (f p)) { fst = a; snd = b }
  in
  check_arr ~msg:"da" (to_arr b) g.fst;
  check_arr ~msg:"db" (to_arr a) g.snd

let test_custom_vjp_rejects_forward_mode () =
  raises_invalid_arg (fun () ->
      ignore (Rune.jvp' my_sin (v3 ()) (tangent_like (v3 ()))))

(* custom_jvp *)

let my_sin_fwd x =
  Rune.custom_jvp
    (module Single)
    ~f:Nx.sin
    ~jvp:(fun x dx -> (Nx.sin x, Nx.mul dx (Nx.cos x)))
    x

let fake_jvp_sin x =
  Rune.custom_jvp
    (module Single)
    ~f:Nx.sin
    ~jvp:(fun x dx -> (Nx.sin x, Nx.mul_s dx 100.0))
    x

let test_jvp_rule_replaces_autodiff () =
  let _, dy = Rune.jvp' fake_jvp_sin (v3 ()) (vec64 [| 1.0; 1.0; 1.0 |]) in
  check_arr ~msg:"fake jvp rule" [| 100.0; 100.0; 100.0 |] dy

let test_correct_jvp_rule_matches_autodiff () =
  let x = v3 () in
  let v = tangent_like x in
  let _, expected = Rune.jvp' Nx.sin x v in
  let _, actual = Rune.jvp' my_sin_fwd x v in
  check_arr ~msg:"my_sin_fwd jvp" (to_arr expected) actual

let test_custom_jvp_rejects_reverse_mode () =
  raises_invalid_arg (fun () ->
      ignore (Rune.grad' (fun x -> Nx.sum (my_sin_fwd x)) (v3 ())))

let test_custom_jvp_undifferentiated () =
  check_arr ~msg:"value" (to_arr (Nx.sin (v3 ()))) (my_sin_fwd (v3 ()))

(* Composition *)

let test_custom_rule_under_vmap_of_grad () =
  (* Per-sample gradients through a custom rule: the rule's bwd runs batched. *)
  let xs = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |] in
  let g = Rune.vmap' (fun x -> Rune.grad' (fun x -> Nx.sum (my_sin x)) x) xs in
  check_arr ~msg:"per-sample custom grads" (to_arr (Nx.cos xs)) g

let test_custom_fwd_under_plain_vmap () =
  (* Without differentiation, vmap batches the forward function. *)
  let xs = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |] in
  let y = Rune.vmap' my_sin xs in
  check_arr ~msg:"vmapped fwd" (to_arr (Nx.sin xs)) y

let test_grad_of_vmap_of_custom () =
  (* grad of vmap of a custom function: vmap batches the forward computation,
     and the enclosing grad differentiates it — the rule applies only to a
     differentiation inside the vmap. Regression: the un-translated custom call
     used to escape the batching scope and double-count each row. *)
  let xs = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |] in
  let g = Rune.grad' (fun x -> Nx.sum (Rune.vmap' my_sin x)) xs in
  check_arr ~msg:"grad of vmapped custom" (to_arr (Nx.cos xs)) g

let test_jvp_of_vmap_of_custom_jvp () =
  (* Same boundary in forward mode: the tangent keeps the mapped shape. *)
  let xs = Nx.create f64 [| 2; 3 |] [| 0.5; -1.2; 2.1; 1.7; -0.4; 0.9 |] in
  let v = tangent_like xs in
  let _, dy = Rune.jvp' (fun x -> Rune.vmap' my_sin_fwd x) xs v in
  check_arr ~msg:"jvp of vmapped custom" (to_arr (Nx.mul v (Nx.cos xs))) dy

let tests =
  [
    group "custom_vjp"
      [
        test "the rule replaces autodiff" test_rule_replaces_autodiff;
        test "a correct rule matches autodiff"
          test_correct_rule_matches_autodiff;
        test "the rule composes inside a graph" test_rule_composes_in_a_graph;
        test "undifferentiated calls run fwd"
          test_undifferentiated_call_runs_fwd;
        test "constants pass through" test_constants_pass_through;
        test "multi-leaf structures get per-leaf gradients" test_multi_leaf_rule;
        test "rejects forward mode" test_custom_vjp_rejects_forward_mode;
      ];
    group "custom_jvp"
      [
        test "the rule replaces autodiff" test_jvp_rule_replaces_autodiff;
        test "a correct rule matches autodiff"
          test_correct_jvp_rule_matches_autodiff;
        test "rejects reverse mode" test_custom_jvp_rejects_reverse_mode;
        test "undifferentiated calls run f" test_custom_jvp_undifferentiated;
      ];
    group "composition"
      [
        test "per-sample gradients through a custom rule"
          test_custom_rule_under_vmap_of_grad;
        test "plain vmap batches the forward function"
          test_custom_fwd_under_plain_vmap;
        test "grad of vmap differentiates the batched forward computation"
          test_grad_of_vmap_of_custom;
        test "jvp of vmap keeps the mapped tangent shape"
          test_jvp_of_vmap_of_custom_jvp;
      ];
  ]

let () = run "rune custom" tests
