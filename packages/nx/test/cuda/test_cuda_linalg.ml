(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cuda's matrix routines, as nx_cuda.mli states them, on the machine's
   NVIDIA GPU 0 through CUDA: each result
   against its defining relation within nx.cuda's bound, on the suite's
   cases, at orders past every blocking, a panel's meeting and a split
   product, on spread and scaled magnitudes and at every small shape whose
   matrices share a group; the spectra against known ones; the Upper
   factor's bits; LU, Cholesky and the solves bit for bit the unblocked
   algorithm's; the same bits under every layout, in every batch, beside
   whatever a small svd's group holds and over memory that admits only
   reads; past a lowered limit of iteration, NaN in every result of a
   matrix and its bits alone for a neighbour that converges. Through
   rig.nv, CUDA's bits. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support
open Linalg_cases
open Linalg_laws

(* nx_cuda.mli's bounds: the unblocked algorithm's; eigh's and svd's
   values alone resolved to [k] times the least normal number; eig's
   vectors positive at their largest element on real matrices too. *)
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

(* Orders past every blocking: LU's panels of 64 columns and its rows held
   by 9 blocks over a launch a step (1100 rows), Cholesky's blocks of 64,
   a solve's leaves of 64, QR's blocks of 128 columns, a QR whose products
   split k (2500 rows, seed 10: reduced), the unblocked algorithm's bits
   past them all, LU's where its rows are held by 5 blocks (600 rows), and
   QR's bits whatever blocks its rows split over. *)
let test_large (b : Support.backend) dt =
  law_lu bounds b (fixed dt 1100 100);
  law_cholesky bounds b (fixed dt 300 0);
  law_solve bounds b (fixed dt 300 3);
  law_solve bounds b (fixed ~triangle:Upper ~adjoint:true dt 200 5);
  law_qr b (fixed ~seed:10 dt 300 140);
  law_qr b (fixed ~seed:10 dt 2500 40);
  law_unblocked b (fixed dt 200 70);
  lu_unblocked b (fixed dt 600 70);
  List.iter (qr_blocks b dt) [ (300, 40); (4500, 40); (32768, 8) ]

(* Read-only operands *)

(* [x] and [y] alike: floats as [same_bits] compares them, LU's int64
   positions equal. *)
let alike ~msg (A.Any x) y =
  match A.dtype x with
  | D.Int64 ->
      let y = A.expect D.Int64 y in
      equal ~msg (array int64)
        (A.to_array (Support.host x))
        (A.to_array (Support.host y))
  | _ -> equal ~msg same_bits (bits (A.Any x)) (bits y)

(* [x] on [b]'s device over memory that admits only [Read]: a borrow of
   host pages holding a copy of its bytes. *)
let read_only (b : Support.backend) (A.Any x) =
  let y = Nx_array_support.mapped_read_only (Support.host x) in
  match Rig.Buffer.borrow b.device (A.buffer y) with
  | Some buf -> A.Any (A.v (A.dtype y) (A.layout y) buf)
  | None -> fail "the GPU maps no host pages"

(* Operands whose memory admits only [Read] give the bits writable ones
   give: read where they lie by QR's one launch, copied by a QR past it,
   and read by LU, Cholesky and a solve. *)
let test_read_only (b : Support.backend) dt =
  let check name r ops =
    match (run b r ops, run b r (Array.map (read_only b) ops)) with
    | Some y, Some z ->
        Array.iteri
          (fun i y ->
            alike ~msg:(Printf.sprintf "%s, result %d" name i) y z.(i))
          y
    | _ -> ()
  in
  let few = fixed dt 12 3 and wide = fixed dt 100 70 in
  let square = fixed dt 100 3 in
  check "qr, one launch" (Qr Reduced) [| snd (lu_operand few) |];
  check "qr, blocked" (Qr Reduced) [| snd (lu_operand wide) |];
  check "lu" Lu [| snd (lu_operand few) |];
  check "lu, blocked" Lu [| snd (lu_operand wide) |];
  check "cholesky" (Cholesky Lower) [| cholesky_operand square |];
  check "solve" (solve square) (solve_operands square)

(* svd of every small shape whose matrices share a group of threads, at
   every dtype: each group's shared memory within the device's, every
   result within its bound. *)
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
              let c = { (fixed dt m 0) with batch = [| 2 |] } in
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
                  check_values bounds dt ~e:0 ~count:2 ~m ~n v s (values y'.(0))
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

   nx_cuda.mli's limits of iteration, lowered by bound_iterations: a drawn
   matrix needs more and is NaN in every result, while a diagonal one
   beside it, which converges at once, has the bits it has alone. *)

let limited ?(sweeps = 60) ?(steps = 30) ?(qr = 30) ?(rotations = 6) f =
  Nx_cuda.bound_iterations ~sweeps ~steps ~qr ~rotations;
  Fun.protect f ~finally:(fun () ->
      Nx_cuda.bound_iterations ~sweeps:60 ~steps:30 ~qr:30 ~rotations:6)

(* The parts of [x] that are not NaN, the imaginary ones of a complex
   dtype, and the count of its parts. *)
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

(* A secular equation whose first guesses are its roots, as the one
   merge of a tridiagonal matrix of order 34: a diagonal leaf of 32, its
   last value 0 once torn, and a leaf of 2 of values 1 and 3, coupled by
   3 / 2, so that the merge is 1 + 3 Σ z_j² / (d_j − λ) over d = (0, 1, 3)
   and z² = (1/2, 7/24, 5/24), the leaf's other values set aside. Its
   roots are 1/2, 2 and 9/2: each interior one the middle of its poles,
   the last its pole plus ρ / 2, where each first guess evaluates the far
   poles, so that it is the root. At a limit of one iterate, the first
   guess, eigh's results meet check_eigh's relations. *)
let test_first_guess =
  each_dtype (fun b dt _ _ ->
      let n = 34 and beta = 1.5 and r = sqrt 35. /. 6. in
      let diag =
        Array.init n (fun i ->
            if i < 31 then -.float_of_int (i + 1) /. 8.
            else if i = 31 then beta
            else if i = 32 then (11. /. 6.) +. beta
            else 13. /. 6.)
      in
      let a =
        Array.init (n * n) (fun p ->
            let i = p / n and j = p mod n in
            let v =
              if i = j then diag.(i)
              else if min i j = 31 && abs (i - j) = 1 then beta
              else if min i j = 32 && abs (i - j) = 1 then r
              else 0.
            in
            { Complex.re = v; im = 0. })
      in
      let x = of_values dt [| n; n |] a in
      limited ~steps:1 (fun () ->
          match run b (Eigh true) [| x |] with
          | Some y ->
              check_eigh dt ~e:0 ~count:1 ~n (values x) (values y.(0))
                (values y.(1))
          | None -> ()))

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

(* GPU 0 through CUDA and through rig.nv, opened at a test's first call;
   a test skips without its GPU. *)
let cuda_name = "cuda"
let nv_name = "cuda/nv"

let cuda () =
  match Support.gpus () with
  | b :: _ -> b
  | [] -> skip ~reason:"CUDA opens no GPU" ()

let nv () =
  match Support.nv () with
  | Some b -> b
  | None -> skip ~reason:"rig.nv opens no GPU" ()

let laws ~name b =
  let on f x = f (b ()) x in
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt ->
        f (b ()) dt)
  in
  group name
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case
            (on (law_cholesky bounds));
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (on law_upper);
          prop "a triangular solve within γ(n + 2)" case
            (on (law_solve bounds));
          prop "LU's factors within γ(k + 2)" case (on (law_lu bounds));
          test "LU of a pivot whose reciprocal is subnormal"
            (on (test_huge_pivot bounds));
          test "QR of matrices scaled to the ends of the exponent range"
            (on test_qr_scaled);
          prop "QR's factors within γ(k(2m + 10))" case (on law_qr);
          cases "past every blocking" test_large;
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
          test
            "eigh's, qr's and svd's results on matrices scaled below the least \
             normal number and near the largest"
            (on (test_scaled bounds));
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
          test "a secular equation's iterates" (on test_steps);
          test "a secular equation whose first guesses are its roots"
            (on test_first_guess);
          test "eig's QR iterations" (on test_qr);
          test "the bidiagonal QR's rotations" (on test_rotations);
        ];
      group "bits are a function of the values alone"
        [
          prop "LU's, Cholesky's and a solve's are the unblocked algorithm's"
            case (on law_unblocked);
          prop "whatever the layout" case (on law_layouts);
          cases "whatever memory the operands admit" test_read_only;
          (* Each case makes 40 calls. *)
          prop ~count:25 "whatever the batch" case (on law_batch);
          test "whatever a small svd's group holds beside it"
            (on test_svd_neighbours);
        ];
    ]

(* rig.nv *)

(* A matrix's results are a function of its values alone, so rig.nv
   gives CUDA's bits: CUDA's laws are its own. What it does beyond CUDA,
   loading the cubin, launching with a group's shared memory, taking
   workspaces and copying, shows in its bits on the suite's cases, at
   orders past every blocking and at every small shape that shares a
   group. *)

(* Each [r] of its [ops] has the same bits on [b] as on [first], any NaN
   as one, since a NaN's sign and payload are unspecified; LU's positions
   the same integers. The calls on [first] come first, then
   those on [b]: two drivers' calls interleaved switch the GPU between
   their contexts at each. *)
let agree ~(first : Support.backend) (b : Support.backend) calls =
  let results d =
    List.map
      (fun (r, ops) ->
        match run d r ops with
        | None -> [||]
        | Some y ->
            Array.mapi
              (fun i y ->
                match (r, i) with
                | Lu, (1 | 2) -> `Ints (int64s y)
                | _ -> `Bits (nan_bits y))
              y)
      calls
  in
  let want = results first in
  let got = results b in
  List.iteri
    (fun k (want, got) ->
      Array.iteri
        (fun i w ->
          let msg = Printf.sprintf "call %d, result %d" k i in
          match (w, got.(i)) with
          | `Ints w, `Ints g -> equal ~msg (array int) w g
          | `Bits w, `Bits g -> equal ~msg same_bits w g
          | _ -> fail msg)
        want)
    (List.combine want got)

(* Each routine a case calls, by name, with its operands as the case lays
   them out. *)
let routines =
  let lu c = snd (lu_operand c)
  and svd c = match svd_operand c with _, _, x -> x in
  [
    ("cholesky", fun c -> (Cholesky c.triangle, [| cholesky_operand c |]));
    ("lu", fun c -> (Lu, [| lu c |]));
    ("qr", fun c -> (Qr (factors c), [| lu c |]));
    ("solve", fun c -> (solve c, solve_operands c));
    ("eigh", fun c -> (Eigh true, [| eigh_operand c |]));
    ("eigh's values", fun c -> (Eigh false, [| eigh_operand c |]));
    ("svd", fun c -> (Svd (Some (factors c)), [| svd c |]));
    ("svd's values", fun c -> (Svd None, [| svd c |]));
    ("eig", fun c -> (Eig true, [| eig_operand c |]));
    ("eig's values", fun c -> (Eig false, [| eig_operand c |]));
  ]

let routine =
  Gen.of_list
    ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
    routines

let law_agrees ~first b (c, (name, f)) =
  List.iter (fun (n, _) -> cover n (n = name)) routines;
  cover "an order past 64" (c.n > 64);
  cover "a batch" (count c > 1);
  let r, ops = f c in
  let (A.Any x) = laid_out c.how ops.(0) in
  let x = Support.on first x in
  cover "laid out with gaps"
    (c.how = Padded && c.n > 1 && Support.gapped (A.layout x));
  ops.(0) <- A.Any x;
  agree ~first b [ (r, ops) ]

(* test_large's matrices, each with the routine it takes there. *)
let large dt =
  let lu c = (Lu, [| snd (lu_operand c) |])
  and qr c = (Qr (factors c), [| snd (lu_operand c) |])
  and solves c = (solve c, solve_operands c)
  and cholesky c = (Cholesky Lower, [| cholesky_operand c |]) in
  let batch = { (fixed ~seed:10 dt 300 40) with batch = [| 50 |] } in
  [
    lu (fixed dt 1100 100);
    cholesky (fixed dt 300 0);
    solves (fixed dt 300 3);
    solves (fixed ~triangle:Upper ~adjoint:true dt 200 5);
    qr (fixed ~seed:10 dt 300 140);
    qr (fixed ~seed:10 dt 2500 40);
    lu (fixed dt 200 70);
    cholesky (fixed dt 200 70);
    solves (fixed dt 200 70);
    lu (fixed dt 600 70);
    (Qr Reduced, [| snd (lu_operand batch) |]);
  ]

let test_shared_agrees ~first b () =
  List.iter
    (fun dt ->
      List.concat_map
        (fun m ->
          List.concat_map
            (fun n ->
              let _, x = shared_operand dt m n in
              [ (Svd (Some Reduced), [| x |]); (Svd None, [| x |]) ])
            shared_sizes)
        shared_sizes
      |> agree ~first b)
    dtypes

let agreement ~first b =
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt -> f dt)
  in
  group nv_name
    [
      group
        (Printf.sprintf "bits are %s's" cuda_name)
        [
          (* One routine a case, so that each is drawn often. *)
          prop ~count:300 "on the suite's cases" (Gen.pair case routine)
            (fun c -> law_agrees ~first:(first ()) (b ()) c);
          cases "past every blocking" (fun dt ->
              agree ~first:(first ()) (b ()) (large dt));
          test "at every small shape that shares a group" (fun () ->
              test_shared_agrees ~first:(first ()) (b ()) ());
        ];
    ]

let () =
  exit
    (Windtrap.run "nx.cuda linalg"
       [ laws ~name:cuda_name cuda; agreement ~first:cuda nv ])
