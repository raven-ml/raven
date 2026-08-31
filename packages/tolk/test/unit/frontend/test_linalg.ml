(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Numeric end-to-end tests for {!Tolk_frontend.Linalg}: build the unrolled
   factorizations, realize them on the CPU backend, and assert the properties
   that do not need a reference implementation — reconstruction, orthogonality,
   and solve residuals, which are independent of the algorithm that produced
   the factors. *)

open Windtrap
module T = Tolk_frontend.Tensor
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Mv = Tolk_frontend.Movement
module Run = Tolk_frontend.Run
module Linalg = Tolk_frontend.Linalg

let fa ~shape data = Run.of_float_array ~shape data

let max_abs_run t = (Rd.max (El.abs t) |> Run.to_float_array).(0)

let close ~tol a b = Float.abs (a -. b) < tol

let check_zero ~tol ~msg t =
  let got = max_abs_run t in
  if not (close ~tol 0.0 got) then
    failf "%s: expected residual 0, got %g" msg got

(* Swap the last two axes of a matrix or batch of matrices. *)
let transpose2 t =
  let rank = List.length (T.shape t) in
  Mv.permute t
    (List.init rank (fun i ->
         if i = rank - 2 then rank - 1 else if i = rank - 1 then rank - 2 else i))

let eye_batch ~dtype batch m =
  Mv.expand (Op.eye ~m ~dtype m) (batch @ [ m; m ])

let qr_tests =
  group "qr"
    [
      test "rectangular reconstruction and orthogonality" (fun () ->
          (* Full (not reduced) factors reconstruct for tall and wide inputs
             alike, and Q is orthogonal. *)
          let tall = fa ~shape:[ 4; 2 ] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.5 |] in
          let wide = fa ~shape:[ 2; 4 ] [| 1.5; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
          List.iter
            (fun a ->
              let q, r = Linalg.qr ~reduced:false a in
              check_zero ~tol:1e-5 ~msg:"|A - QR|"
                (El.sub a (Op.matmul q r));
              let m = List.nth (T.shape a) 0 in
              check_zero ~tol:1e-5 ~msg:"|Q'Q - I|"
                (El.sub (Op.matmul (transpose2 q) q)
                   (Op.eye ~m ~dtype:(T.val_dtype a) m)))
            [ tall; wide ]);
      test "batched reconstruction" (fun () ->
          let a =
            fa ~shape:[ 2; 3; 3 ]
              [| 2.; 1.; 1.; 1.; 3.; 2.; 1.; 2.; 4.; 5.; 4.; 1.; 4.; 5.; 2.; 1.;
                 2.; 3. |]
          in
          let q, r = Linalg.qr ~reduced:true a in
          check_zero ~tol:1e-5 ~msg:"batched |A - QR|"
            (El.sub a (Op.matmul q r)));
      test "a zero-tail column takes no reflector" (fun () ->
          (* The second column's tail is zero, so it takes no reflector (its
             R diagonal keeps its sign); the third column is full. Both paths
             must still reconstruct. *)
          let a = fa ~shape:[ 3; 3 ] [| 1.; 0.; 2.; 0.; 2.; 3.; 0.; 0.; 4. |] in
          let q, r = Linalg.qr ~reduced:true a in
          check_zero ~tol:1e-5 ~msg:"|A - QR| (zero tail)"
            (El.sub a (Op.matmul q r)));
    ]

(* The coefficient matrix actually solved for a flag combination, built from
   the data triangle of [d]: the named triangle (diagonal replaced by ones
   under [unit_diag]), transposed when [transpose] is set — exactly what
   [solve_triangular] reads. The other triangle of [a] is filled with garbage
   to check it is never read. *)
let effective_system ~upper ~transpose ~unit_diag d garbage =
  let dtype = T.val_dtype d in
  let n = List.nth (T.shape d) 0 in
  let strict, tri, ignored =
    if upper then
      (Op.triu ~diagonal:1 d, Op.triu d, Op.tril ~diagonal:(-1) garbage)
    else
      (Op.tril ~diagonal:(-1) d, Op.tril d, Op.triu ~diagonal:1 garbage)
  in
  let a = El.add tri ignored in
  let e0 = if unit_diag then El.add strict (Op.eye ~m:n ~dtype n) else tri in
  let e = if transpose then transpose2 e0 else e0 in
  (a, e)

let solve_tests =
  group "solve_triangular"
    [
      test "every flag combination solves the system it reads" (fun () ->
          let d =
            fa ~shape:[ 3; 3 ] [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6. |]
          in
          let b = fa ~shape:[ 3; 2 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          let garbage =
            El.mul
              (fa ~shape:[ 3; 3 ]
                 [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1. |])
              (T.f 7.0)
          in
          List.iter
            (fun (upper, transpose, unit_diag) ->
              let msg =
                Printf.sprintf "upper %b transpose %b unit_diag %b" upper
                  transpose unit_diag
              in
              let a, e =
                effective_system ~upper ~transpose ~unit_diag d garbage
              in
              let x =
                Linalg.solve_triangular ~upper ~transpose ~unit_diag a b
              in
              check_zero ~tol:1e-4 ~msg:("|e·x - b| " ^ msg)
                (El.sub (Op.matmul e x) b))
            [
              (false, false, false);
              (true, false, false);
              (false, true, false);
              (true, true, false);
              (false, false, true);
              (true, false, true);
              (false, true, true);
              (true, true, true);
            ]);
      test "vector right-hand side" (fun () ->
          let a = fa ~shape:[ 3; 3 ] [| 2.; 1.; 0.; 1.; 3.; 1.; 0.; 1.; 4. |] in
          let b = fa ~shape:[ 3 ] [| 1.; 2.; 3. |] in
          (* The solver reads the lower triangle only; the residual is checked
             against that triangle, not the full matrix. *)
          let e = Op.tril a in
          let x =
            Linalg.solve_triangular ~upper:false ~transpose:false
              ~unit_diag:false a b
          in
          check_zero ~tol:1e-5 ~msg:"|e·x - b| (vector rhs)"
            (El.sub (Op.matmul e x) b));
      test "batched" (fun () ->
          let a =
            fa ~shape:[ 2; 3; 3 ]
              [| 4.; 1.; 2.; 0.; 5.; 3.; 0.; 0.; 6.; 2.; 1.; 0.; 0.; 3.; 1.;
                 0.; 0.; 4. |]
          in
          let b = fa ~shape:[ 2; 3; 1 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          let x =
            Linalg.solve_triangular ~upper:true ~transpose:false
              ~unit_diag:false a b
          in
          check_zero ~tol:1e-5 ~msg:"|a·x - b| (batched upper)"
            (El.sub (Op.matmul a x) b));
      test "wide right-hand sides take the blocked path" (fun () ->
          (* 80 rows span several 32-row blocks plus a partial trailing
             block; the residual is independent of the solver. *)
          let n = 80 in
          let a_data =
            Array.init (n * n) (fun k ->
                let i, j = k / n, k mod n in
                if i > j then
                  Float.of_int ((((i * 37) + (j * 11)) mod 13) - 6) /. 8.0
                else if i = j then 2.0
                else 0.0)
          in
          let b_data =
            Array.init (n * n) (fun k ->
                Float.of_int ((((k / n) * 5) + (k mod n)) mod 7) -. 3.)
          in
          let a = fa ~shape:[ n; n ] a_data in
          let b = fa ~shape:[ n; n ] b_data in
          let x =
            Linalg.solve_triangular ~upper:false ~transpose:false
              ~unit_diag:false a b
          in
          check_zero ~tol:1e-3 ~msg:"blocked |a·x - b|"
            (El.sub (Op.matmul a x) b);
          (* The flags compose with blocking: ~upper reads the strict upper
             triangle of a matrix whose stored diagonal is garbage (never read
             under ~unit_diag), and ~transpose solves the transposed system —
             so the effective system is [I + strict_upper(m)ᵀ]. *)
          let au =
            El.add
              (El.mul (Op.triu ~diagonal:1 (transpose2 a)) (T.f 0.125))
              (El.mul (Op.eye ~m:n ~dtype:(T.val_dtype a) n) (T.f 7.0))
          in
          let e =
            (* The solve reads [I + strict_upper(au)] and transposes it, and
               [strict_upper(au) = 0.125·(strict_lower a)ᵀ], so the effective
               system is [I + 0.125·strict_lower(a)]. *)
            El.add
              (Op.eye ~m:n ~dtype:(T.val_dtype a) n)
              (El.mul (Op.tril ~diagonal:(-1) a) (T.f 0.125))
          in
          let x =
            Linalg.solve_triangular ~upper:true ~transpose:true ~unit_diag:true
              au b
          in
          check_zero ~tol:1e-3 ~msg:"blocked flags |e·x - b|"
            (El.sub (Op.matmul e x) b));
    ]

let cholesky_tests =
  group "cholesky"
    [
      test "batched reconstruction in both triangles" (fun () ->
          let a =
            fa ~shape:[ 2; 3; 3 ]
              [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6.; 6.; 2.; 1.; 2.; 7.; 2.; 1.;
                 2.; 8. |]
          in
          let l = Linalg.cholesky ~upper:false a in
          check_zero ~tol:1e-4 ~msg:"batched |A - LL'|"
            (El.sub a (Op.matmul l (transpose2 l)));
          let u = Linalg.cholesky ~upper:true a in
          check_zero ~tol:1e-4 ~msg:"batched |A - U'U|"
            (El.sub a (Op.matmul (transpose2 u) u)));
      test "a non-positive-definite input yields nans" (fun () ->
          (* The graph has no host control flow to raise Linalg_error with, so
             the failed square root of a negative pivot propagates as nan. *)
          let a = fa ~shape:[ 2; 2 ] [| 1.; 0.; 0.; -1. |] in
          let l = Linalg.cholesky ~upper:false a in
          let got = Run.to_float_array l in
          if not (Array.exists Float.is_nan got) then
            failf "expected nans in the factor of a non-PD input");
    ]

let () = run "Tolk_frontend_linalg" [ qr_tests; solve_tests; cholesky_tests ]
