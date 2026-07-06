(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Control-flow combinators: eager semantics, and composition with grad and
   vmap. *)

open Windtrap
open Rune_test_support.Support

module Single = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) t = f t

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    f a b

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) t = f t
end

let v4 () = vec64 [| 0.5; -1.2; 2.1; 0.8 |]

(* scan with a running-sum carry is a cumulative sum. *)
let cumsum_scan xs =
  snd
    (Rune.scan
       (module Single)
       ~f:(fun c x ->
         let c = Nx.add c x in
         (c, c))
       ~init:(Nx.scalar f64 0.0) xs)

let test_scan_is_cumsum () =
  check_arr ~msg:"scan cumsum"
    (to_arr (Nx.cumsum (v4 ())))
    (cumsum_scan (v4 ()))

let test_scan_final_carry () =
  let carry, _ =
    Rune.scan
      (module Single)
      ~f:(fun c x -> (Nx.add c x, c))
      ~init:(Nx.scalar f64 0.0) (v4 ())
  in
  check_arr ~msg:"carry" [| 0.5 -. 1.2 +. 2.1 +. 0.8 |] carry

let test_grad_through_scan () =
  (* Gradients through the scan equal gradients through the primitive. *)
  let f xs = Nx.sum (Nx.mul (cumsum_scan xs) (cumsum_scan xs)) in
  let g xs = Nx.sum (Nx.mul (Nx.cumsum xs) (Nx.cumsum xs)) in
  check_arr ~msg:"d scan" (to_arr (Rune.grad' g (v4 ()))) (Rune.grad' f (v4 ()))

let test_vmap_of_scan () =
  let x =
    Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]
  in
  let y = Rune.vmap' cumsum_scan x in
  check_arr ~msg:"vmap scan" (to_arr (Nx.cumsum ~axis:1 x)) y

let test_scan_rejects_scalar () =
  raises_invalid_arg (fun () -> ignore (cumsum_scan (Nx.scalar f64 1.0)))

let test_cond_branches () =
  let branch x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then" [| 0.25 +. 4.0 |] (branch (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else" [| -2.5 |] (branch (vec64 [| -0.5; -2.0 |]))

let test_grad_through_cond () =
  (* The taken branch is what gets differentiated. *)
  let f x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar f64 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  check_arr ~msg:"then grad" [| 1.0; 4.0 |]
    (Rune.grad' f (vec64 [| 0.5; 2.0 |]));
  check_arr ~msg:"else grad" [| 1.0; 1.0 |]
    (Rune.grad' f (vec64 [| -0.5; -2.0 |]))

let test_while_loop () =
  (* Double until the sum exceeds 10: 1.5 -> 3 -> 6 -> 12. *)
  let y =
    Rune.while_loop
      (module Single)
      ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
      ~body:(fun c -> Nx.mul_s c 2.0)
      (vec64 [| 1.0; 0.5 |])
  in
  check_arr ~msg:"final" [| 8.0; 4.0 |] y

let test_grad_through_while_loop () =
  (* Each x doubles k times before the loop exits; d/dx sum = 2^k. *)
  let f x =
    Nx.sum
      (Rune.while_loop
         (module Single)
         ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar f64 10.0))
         ~body:(fun c -> Nx.mul_s c 2.0)
         x)
  in
  check_arr ~msg:"d while" [| 8.0; 8.0 |] (Rune.grad' f (vec64 [| 1.0; 0.5 |]))

let tests =
  [
    group "scan"
      [
        test "running-sum scan is cumsum" test_scan_is_cumsum;
        test "returns the final carry" test_scan_final_carry;
        test "differentiates like the primitive" test_grad_through_scan;
        test "vectorizes over the batch" test_vmap_of_scan;
        test "rejects a scalar input" test_scan_rejects_scalar;
      ];
    group "cond"
      [
        test "selects the branch by predicate" test_cond_branches;
        test "differentiates the taken branch" test_grad_through_cond;
      ];
    group "while_loop"
      [
        test "iterates until the predicate fails" test_while_loop;
        test "differentiates the taken iterations" test_grad_through_while_loop;
      ];
  ]

let () = run "rune control" tests
