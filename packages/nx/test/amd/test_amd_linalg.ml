(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.amd's spectral routines, as nx_amd.mli states them, on the machine's
   AMD GPU 0, skipping without one: each result against its defining
   relation within nx.amd's bound, on the suite's cases, at orders past
   Jacobi's, a panel and a leaf, on spread and scaled magnitudes and at
   every small shape whose matrices share a group; the spectra against
   known ones; a small svd's bits beside a neighbour of its group; each
   iteration's limit; the workspace each call holds. test_amd_factor.ml
   checks their bits under every layout and in every batch, with the
   factorisations'. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support
open Linalg_cases
open Linalg_laws

(* nx_amd.mli's bounds: eigh's and svd's values alone resolved to [k]
   times the least normal number; eig's vectors positive at their largest
   element on real matrices too. The factorisations' are nx.amd's to
   state with their family; their laws do not run here. *)
let least_normal (D.Any dt) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-126 | _ -> 0x1p-1022

let bounds =
  {
    cholesky = (fun n -> n + 1);
    lu = (fun k -> k + 2);
    solve = (fun n -> n + 2);
    alone =
      (fun ~e dt k ->
        let k = if is_complex dt then 2 * k else k in
        float_of_int k *. Float.ldexp (least_normal dt) e);
    positive = true;
  }

(* A case of [dt] and order [n], one matrix. *)
let fixed dt n =
  {
    dt;
    batch = [||];
    n;
    r = 0;
    triangle = Nx_kernel.Lower;
    adjoint = false;
    unit_diagonal = false;
    seed = 7;
    how = Contiguous;
    scale = Plain;
  }

(* svd of every small shape whose matrices share a group of work-items, at
   every dtype: each group's LDS within the device's, every result within
   its bound. *)
let shared_sizes = [ 1; 2; 3; 5; 8; 13; 26; 31; 32 ]

(* A batch of two m x n matrices of [dt]: their values and the array. *)
let shared_operand dt m n =
  let r = Random.State.make [| m; n |] in
  let v = Array.init (2 * m * n) (fun _ -> draw r dt) in
  (v, of_values dt [| 2; m; n |] v)

let test_svd_shapes (b : Support.backend) () =
  List.iter
    (fun dt ->
      List.iter
        (fun m ->
          List.iter
            (fun n ->
              let c = { (fixed dt m) with batch = [| 2 |] } in
              let v, x = shared_operand dt m n in
              match
                (run b (Svd (Some Reduced)) [| x |], run b (Svd None) [| x |])
              with
              | Some y, Some y' ->
                  let k = min m n in
                  let s = values y.(0) in
                  check_svd dt ~e:0 ~count:(count c) ~m ~n ~uc:k ~vr:k v s
                    (values y.(1))
                    (values y.(2));
                  check_values bounds dt ~e:0 ~count:2 ~m ~n v s
                    (values y'.(0))
              | _ -> ())
            shared_sizes)
        shared_sizes)
    dtypes

(* A small svd's matrix has the bits it has alone beside a matrix of its
   group that may sweep longer: [x] of columns graded by 2^6 beside [y], a
   plain draw. Each shape's group holds both: at most 128 elements a
   matrix. *)
let neighbour_shapes =
  [ (2, 2); (3, 2); (4, 3); (4, 4); (6, 5); (8, 4); (8, 8); (11, 7); (16, 8) ]

let test_svd_neighbours (b : Support.backend) () =
  let check dt (m, n) seed r =
    let st = Random.State.make [| m; n; seed |] in
    let x =
      Array.init (m * n) (fun k -> ldexp (6 * (k mod n)) (draw st dt))
    in
    let y = Array.init (m * n) (fun _ -> draw st dt) in
    match
      ( run b r [| of_values dt [| m; n |] x |],
        run b r [| of_values dt [| 2; m; n |] (Array.append x y) |] )
    with
    | Some one, Some two ->
        Array.iteri
          (fun i z ->
            equal
              ~msg:
                (Format.asprintf "%a %d x %d, seed %d, result %d" pp_dtype dt
                   m n seed i)
              same_bits (bits z)
              (bits (alone 0 two.(i))))
          one
    | _ -> ()
  in
  List.iter
    (fun dt ->
      List.iter
        (fun shape ->
          for seed = 0 to 7 do
            check dt shape seed (Svd (Some Reduced));
            check dt shape seed (Svd None)
          done)
        neighbour_shapes)
    dtypes

(* Limits

   nx_amd.mli's limits of iteration, lowered by bound_iterations: a drawn
   matrix needs more and is NaN in every result, while a diagonal one
   beside it, which converges at once, has the bits it has alone. *)

let limited ?(sweeps = 60) ?(steps = 30) ?(qr = 30) ?(rotations = 6) f =
  Nx_amd.bound_iterations ~sweeps ~steps ~qr ~rotations;
  Fun.protect f ~finally:(fun () ->
      Nx_amd.bound_iterations ~sweeps:60 ~steps:30 ~qr:30 ~rotations:6)

(* The parts of [x], the imaginary ones of a complex dtype, that are not
   NaN, and the count of its parts. *)
let parts (A.Any x) =
  let dt = D.Any (A.dtype x) and v = values (A.Any x) in
  (numbers dt v, Array.length v * if is_complex dt then 2 else 1)

let fails (b : Support.backend) r dt ~rows ~cols x =
  let diagonal =
    Array.init (rows * cols) (fun k ->
        if k / cols = k mod cols then
          { Complex.re = float_of_int ((k / cols) + 1); im = 0. }
        else Complex.zero)
  in
  let pair = of_values dt [| 2; rows; cols |] (Array.append x diagonal) in
  let lone = of_values dt [| rows; cols |] diagonal in
  match (run b r [| pair |], run b r [| lone |]) with
  | Some y, Some one ->
      Array.iteri
        (fun i y ->
          let msg what =
            Format.asprintf "%a %d x %d, result %d: %s" pp_dtype dt rows cols
              i what
          in
          equal ~msg:(msg "numbers in the drawn one") int 0
            (fst (parts (alone 0 y)));
          equal ~msg:(msg "the diagonal one is alone") same_bits
            (bits one.(i)) (bits (alone 1 y));
          let numbers, all = parts one.(i) in
          equal ~msg:(msg "numbers in the diagonal one") int all numbers)
        y
  | _ -> ()

(* [each_dtype check b ()] is [check b dt drawn st] at each dtype, [st]
   a fixed seed's state and [drawn rows cols] a plain draw of it. *)
let each_dtype check (b : Support.backend) () =
  List.iter
    (fun dt ->
      let st = Random.State.make [| 5 |] in
      let drawn rows cols = Array.init (rows * cols) (fun _ -> draw st dt) in
      check b dt drawn st)
    dtypes

let test_sweeps =
  each_dtype (fun b dt drawn st ->
      limited ~sweeps:1 (fun () ->
          let x = hermitian st dt 8 in
          fails b (Eigh true) dt ~rows:8 ~cols:8 x;
          fails b (Eigh false) dt ~rows:8 ~cols:8 x;
          let x = drawn 8 6 in
          fails b (Svd (Some Reduced)) dt ~rows:8 ~cols:6 x;
          fails b (Svd None) dt ~rows:8 ~cols:6 x))

let test_leaf_sweeps =
  each_dtype (fun b dt _ st ->
      limited ~sweeps:1 (fun () ->
          fails b (Eigh true) dt ~rows:40 ~cols:40 (hermitian st dt 40)))

let test_steps =
  each_dtype (fun b dt _ st ->
      limited ~steps:1 (fun () ->
          fails b (Eigh true) dt ~rows:40 ~cols:40 (hermitian st dt 40)))

let test_qr =
  each_dtype (fun b dt drawn _ ->
      limited ~qr:0 (fun () ->
          let x = drawn 6 6 in
          fails b (Eig true) dt ~rows:6 ~cols:6 x;
          fails b (Eig false) dt ~rows:6 ~cols:6 x))

let test_rotations =
  each_dtype (fun b dt drawn _ ->
      limited ~rotations:0 (fun () ->
          fails b (Svd (Some Reduced)) dt ~rows:40 ~cols:36 (drawn 40 36)))

(* Memory

   nx_amd.mli's workspace of each routine, with every call taking a
   workspace of its own, which returns once its work is done: calls in a
   loop hold one call's, its own pieces and its largest product's. *)

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024

let all_fresh f =
  let restore () =
    Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* [e] float32 elements on 256 bytes. *)
let piece e = ((4 * e) + 255) land lnot 255

(* The most bytes a float32 product of [rows] x [cols] outputs of [k]
   terms takes, as nx_amd.mli states a contraction's: a copy of each
   operand, its contracted extent padded to 16 bytes, of the result, and
   with [init] of [init]. *)
let product ~rows ~cols ~k ~init =
  let k = (k + 3) / 4 * 4 in
  let y = piece (rows * cols) in
  piece (rows * k) + piece (k * cols) + if init then 2 * y else y

(* [own] float32 elements and bytes, alignment and the largest of
   [products] as {!product} bounds it. *)
let held ~own products =
  own + 512 + 2048
  + List.fold_left
      (fun most (rows, cols, k, init) ->
        Int.max most (product ~rows ~cols ~k ~init))
      0 products

(* [x], a float32 matrix of [m] x [n] in [b]'s buffer, read through its
   transpose where [transposed]. *)
let operand (b : Support.backend) ~transposed m n =
  let base = A.create b.device D.Float32 [| n; m |] in
  if transposed then
    A.v D.Float32 (A.Layout.v ~strides:[| 1; m |] [| m; n |]) (A.buffer base)
  else A.v D.Float32 (A.Layout.contiguous [| m; n |]) (A.buffer base)

(* An eigh with [v] of order [n] takes [2n² + 97 n + 2048] elements,
   [3 n² + 78 n - 2 l (32 - l)] reals and [24 n + 32] bytes, [l] its last
   leaf's order; and [n²] elements more where [a] is transposed, which it
   copies. Its products are of at most [n] x [n] outputs of 32 terms, which
   update a matrix, and [32] x [n] of [n]. *)
let test_loop_eigh ~n ~transposed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = operand b ~transposed n n in
  let w = A.create b.device D.Float32 [| n |] in
  let v = A.create b.device D.Float32 [| n; n |] in
  let l = n - (32 * (((n + 31) / 32) - 1)) in
  let copy = if transposed then n * n else 0 in
  let elements = (2 * n * n) + (97 * n) + 2048 + copy in
  let reals = (3 * n * n) + (78 * n) - (2 * l * (32 - l)) in
  let own = (4 * (elements + reals)) + (24 * n) + 32 in
  let bound = held ~own [ (n, n, 32, true); (32, n, n, false) ] in
  all_fresh (fun () ->
      Kernels_cases.loop_held b ~bound (fun () ->
          equal answer A.Done (K.eigh ~w ~v a);
          Rig.Buffer.wait (A.buffer v) Read_write))

(* A reduced svd of [m] x [n], [M = max (m, n)] and [k = min (m, n)],
   takes [M (2k + 33) + k² + 66 k + 2048] elements, [2 k² + 263 k + 260]
   reals and [4 (192 k + 203)] bytes. Its products are of at most [k] x [M]
   outputs of 32 terms, which update a matrix, and [32] x [k] of [M]. *)
let test_loop_svd ~m ~n ~transposed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = operand b ~transposed m n in
  let mm = Int.max m n and k = Int.min m n in
  let s = A.create b.device D.Float32 [| k |] in
  let u = A.create b.device D.Float32 [| m; k |] in
  let vh = A.create b.device D.Float32 [| k; n |] in
  let copy = if transposed then m * n else 0 in
  let elements = (mm * ((2 * k) + 33)) + (k * k) + (66 * k) + 2048 + copy in
  let reals = (2 * k * k) + (263 * k) + 260 in
  let own = (4 * (elements + reals)) + (4 * ((192 * k) + 203)) in
  let bound = held ~own [ (k, mm, 32, true); (32, k, mm, false) ] in
  all_fresh (fun () ->
      Kernels_cases.loop_held b ~bound (fun () ->
          equal answer A.Done (K.svd ~s ~vectors:(~u, ~vh) a);
          Rig.Buffer.wait (A.buffer vh) Read_write))

(* An eig with [v] of order [n] takes [4 n² + n] elements, [2 n² + 4 n]
   reals and 16 bytes; its product is of [n] x [n] outputs of [n] terms. *)
let test_loop_eig ~n ~transposed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = operand b ~transposed n n in
  let w = A.create b.device D.Complex64 [| n |] in
  let v = A.create b.device D.Complex64 [| n; n |] in
  let copy = if transposed then n * n else 0 in
  let elements = (4 * n * n) + n + copy in
  let reals = (2 * n * n) + (4 * n) in
  let own = (4 * (elements + reals)) + 16 in
  let bound = held ~own [ (n, n, n, false) ] in
  all_fresh (fun () ->
      Kernels_cases.loop_held b ~bound (fun () ->
          equal answer A.Done (K.eig ~w ~v a);
          Rig.Buffer.wait (A.buffer v) Read_write))

(* The machine's AMD GPU 0, opened at a test's first call; a test skips
   without it. *)
let amd () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

let laws () =
  let on f x = f (amd ()) x in
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt ->
        f (amd ()) dt)
  in
  group "amd"
    [
      group "each result meets its relation within its bound"
        [
          prop "eigh's results within γ(16n)" case (on (law_eigh bounds));
          prop "svd's results within γ(16 max (m, n) + 100)" case
            (on (law_svd bounds));
          prop
            "svd's results on parts at the least subnormal, below the least \
             normal and near 1"
            spread (on (law_svd_spread bounds));
          test
            "svd's results on a row whose rest is below the least normal \
             number over u"
            (on test_svd_tiny_rows);
          test "svd's results at every small shape that shares a group"
            (on test_svd_shapes);
          test "svd's results on a row whose rest is the least subnormal"
            (on (test_svd_spread_row bounds));
          cases
            "svd's results on matrices of rank 10 at 2^400 over rows of the \
             least subnormal"
            (test_svd_deficient bounds);
          cases "svd's results on 16384 matrices of order 2"
            (test_svd_pairs bounds);
          prop
            "eig's results within γ(16n), its vectors positive at their largest"
            case (on (law_eig bounds));
          test "eig's results on a graded cycle" (on test_eig_graded_cycle);
          test "eig's results on a subnormal matrix with a complex pair"
            (on test_eig_subnormal_pair);
        ];
      group "each spectrum is within its bound of the exact one"
        [
          prop "eigh's eigenvalues within γ(16n) ‖a‖₂" case
            (on law_eigh_spectrum);
          test "eigh's eigenvalues at large orders" (on test_eigh_large);
          test
            "eigh's eigenvalues of a secular equation with a root beside its \
             pole"
            (on test_eigh_secular);
          prop "svd's values within γ(16 max (m, n) + 100) ‖a‖₂" case
            (on law_svd_spectrum);
          test "svd's values at large orders" (on test_svd_large);
          prop "eig's eigenvalues of a normal matrix within γ(16n) ‖a‖_F" case
            (on law_eig_spectrum);
          test "eig's eigenvalues of a graded matrix, balanced"
            (on test_eig_balanced);
          test
            "eig's eigenvalues of a matrix spanning the exponent range, \
             balanced before it is scaled"
            (on test_eig_extremes);
          test "eig's eigenvalues of Wilkinson's matrices are eigh's"
            (on test_wilkinson);
        ];
      group "a matrix past an iteration's limit is NaN in every result"
        [
          test "Jacobi's sweeps" (on test_sweeps);
          test "Jacobi's sweeps on a leaf of divide and conquer"
            (on test_leaf_sweeps);
          test "a secular equation's steps" (on test_steps);
          test "eig's QR iterations" (on test_qr);
          test "the bidiagonal QR's rotations" (on test_rotations);
        ];
      group "memory"
        [
          test "eighs in a loop hold one eigh's workspace"
            (on (test_loop_eigh ~n:64 ~transposed:false));
          test "eighs of a copied operand in a loop hold one eigh's workspace"
            (on (test_loop_eigh ~n:80 ~transposed:true));
          test "svds in a loop hold one svd's workspace"
            (on (test_loop_svd ~m:72 ~n:40 ~transposed:false));
          test "svds of a copied operand in a loop hold one svd's workspace"
            (on (test_loop_svd ~m:36 ~n:65 ~transposed:true));
          test "eigs in a loop hold one eig's workspace"
            (on (test_loop_eig ~n:40 ~transposed:false));
          test "eigs of a copied operand in a loop hold one eig's workspace"
            (on (test_loop_eig ~n:33 ~transposed:true));
        ];
      group "bits are a function of the values alone"
        [
          test "whatever a small svd's group holds beside it"
            (on test_svd_neighbours);
        ];
    ]

let () = exit (Windtrap.run "nx.amd linalg" [ laws () ])
