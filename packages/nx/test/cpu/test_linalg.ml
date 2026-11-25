(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's factorisations and triangular solves, as nx_cpu.mli states
   them, by the promise each group holds: each result against its defining
   relation within its bound, by residuals computed in double-double, on
   operands well and badly conditioned, below the least normal number and
   near the largest, at orders past its reductions' crossovers and on
   matrices its algorithms find hard; the Upper factor's bits; the same bits
   under every layout of the same values, in a batch and on one thread as on
   the job's; a solve into its operand; its refusals; and no byte read or
   written past an operand or a result. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support
open Linalg_cases
open Linalg_laws

(* nx_cpu.mli's bounds. *)
let bounds =
  {
    cholesky = (fun n -> n + 1);
    lu = (fun k -> 2 * k);
    solve = (fun n -> 2 * n);
    alone = (fun ~e dt k -> epsilon ~underflow:Support.Gradual ~e dt k);
    positive = false;
  }

(* Failures and reads *)

(* A solve of as many right-hand sides as rows, into a destination
   identical to its triangle or to its right-hand sides, gives the bits a
   fresh destination receives. *)
let law_solve_aliased (b : Support.backend) c =
  let module K = (val b.kernels) in
  let c = { c with r = c.n } in
  let ops = solve_operands c in
  let into k =
    let (A.Any a) = ops.(0) in
    let dt = A.dtype a in
    let a = A.copy a and rhs = A.copy (A.expect dt ops.(1)) in
    let dst = if k = 0 then a else rhs in
    match K.solve_triangular c.triangle ~adjoint:c.adjoint
            ~unit_diagonal:c.unit_diagonal ~dst a rhs with
    | A.Done -> Some (bits (A.Any dst))
    | A.Declined -> None
    | e -> failf "the kernels answered %a" pp_answer e
  in
  match run b (solve c) ops with
  | Some y ->
      List.iter
        (fun (msg, k) ->
          Option.iter (equal ~msg same_bits (bits y.(0))) (into k))
        [ ("into the triangle", 0); ("into the right-hand sides", 1) ]
  | None -> ()

(* nx.cpu *)

(* The bits of a result, a position's as its value's. *)
let result_bits (A.Any x as r) =
  match A.dtype x with
  | D.Int64 -> Array.map float_of_int (int64s r)
  | _ -> bits r

(* [o] repeated along [c]'s batch. *)
let repeated c (A.Any x) =
  let s = Iarray.append (Iarray.of_array c.batch) (L.shape (A.layout x)) in
  A.Any (A.copy (Option.get (A.move (M.Broadcast s) x)))

(* [c] with a batch of at least as many matrices as the host has cores,
   along one axis or two by its seed: nx.cpu then computes each matrix
   on one thread. *)
let wide c =
  let t = Domain.recommended_domain_count () in
  let batch = if c.seed mod 2 = 0 then [| t |] else [| 2; (t + 1) / 2 |] in
  { c with batch }

(* The routines with work of their own on a matrix's threads, besides their
   products, each with its name and operands of [c]'s matrix: solves' units,
   LU's steps, eigh's and svd's matrix-vector products and their merges'
   roots. *)
let with_units c =
  let _, lu = lu_operand c in
  let _, _, s = svd_operand c in
  [
    ("cholesky", Cholesky Lower, [| cholesky_operand c |]);
    ("solve", solve c, solve_operands c);
    ("lu", Lu, [| lu |]);
    ("eigh", Eigh true, [| eigh_operand c |]);
    ("svd", Svd (Some (factors c)), [| s |]);
    ("svd's values", Svd None, [| s |]);
  ]

(* Every routine, each with its name and operands of [c]'s matrix. *)
let every_routine c =
  let _, lu = lu_operand c in
  with_units c
  @ [ ("qr", Qr (factors c), [| lu |]); ("eig", Eig true, [| eig_operand c |]) ]

(* nx.cpu computes the [routines] of [c]'s matrix alone, on the call's threads,
   and of copies of it in a wide batch, one a thread: every result of every
   matrix of the batch has the bits of the matrix's alone. *)
let threads_agree (b : Support.backend) routines c =
  let c = wide c in
  let k = count c in
  let one = { c with batch = [||] } in
  let agree msg y z =
    Array.iteri
      (fun j z ->
        let v = result_bits y.(j) and want = result_bits z in
        let n = Array.length want in
        for i = 0 to k - 1 do
          let msg = Printf.sprintf "%s's result %d, matrix %d" msg j i in
          equal ~msg same_bits want (Array.sub v (i * n) n)
        done)
      z
  in
  List.iter
    (fun (msg, r, ops) ->
      match (run b r (Array.map (repeated c) ops), run b r ops) with
      | Some y, Some z -> agree msg y z
      | _ -> failf "nx.cpu declined %s at %a" msg pp_dtype c.dt)
    (routines one)

(* nx.cpu computes the [routines] of [c]'s matrix on the call's threads and on
   one thread: every result has the one thread's bits. *)
let one_thread_agrees (b : Support.backend) routines c =
  List.iter
    (fun (msg, r, ops) ->
      match (run b r ops, Nx_cpu.one_thread (fun () -> run b r ops)) with
      | Some y, Some z ->
          Array.iteri
            (fun j z ->
              let msg = Printf.sprintf "%s's result %d" msg j in
              equal ~msg same_bits (result_bits z) (result_bits y.(j)))
            z
      | _ -> failf "nx.cpu declined %s at %a" msg pp_dtype c.dt)
    (routines c)

(* A plain contiguous case of [dt], of order [n] and [r] right-hand sides. *)
let fixed dt n r =
  {
    dt;
    batch = [||];
    n;
    r;
    triangle = Lower;
    adjoint = false;
    unit_diagonal = false;
    seed = 7;
    how = Contiguous;
    scale = Plain;
  }

(* nx.cpu computes every routine at [dt], and a matrix of order 65, past a
   block of 64, alone gives the bits it has in a batch. *)
let test_cpu_threads b dt = threads_agree b every_routine (fixed dt 65 3)

(* A matrix of order 256 and 160 right-hand sides, of the routines whose work on
   a matrix's threads goes beyond their products, gives the bits of one thread,
   which one_thread computes: a batch of a matrix a thread would grow the test's
   work with the host's cores. Where jobs run on one thread, nothing runs on
   several. *)
let test_cpu_units b dt =
  if Nx_cpu.most_threads () = 1 then skip ~reason:"one thread" ();
  one_thread_agrees b with_units (fixed dt 256 160)

(* A routine's operand [a], right-hand sides [b] for a solve, and
   destinations [dsts], each near a fit. *)
type misfit = {
  routine : routine;
  a : int array;
  b : int array;
  dsts : int array array;
}

let routine_name = function
  | Cholesky _ -> "cholesky"
  | Lu -> "lu"
  | Qr Reduced -> "qr reduced"
  | Qr Complete -> "qr complete"
  | Eigh true -> "eigh"
  | Eigh false -> "eigh's values"
  | Svd None -> "svd's values"
  | Svd (Some Reduced) -> "svd reduced"
  | Svd (Some Complete) -> "svd complete"
  | Eig true -> "eig"
  | Eig false -> "eig's values"
  | Solve _ -> "solve_triangular"

let pp_misfit ppf c =
  Format.fprintf ppf "%s of %a%s into %s" (routine_name c.routine) pp_ints c.a
    (if c.b <> [||] then Format.asprintf " and %a" pp_ints c.b else "")
    (String.concat ", "
       (Array.to_list (Array.map (Format.asprintf "%a" pp_ints) c.dsts)))

(* The destinations' shapes Nx_kernel.Shape gives. *)
let shapes c =
  let module S = Nx_kernel.Shape in
  match c.routine with
  | Cholesky _ -> Result.map (fun y -> [| y |]) (S.cholesky c.a)
  | Lu -> Result.map (fun (l, p, q) -> [| l; p; q |]) (S.lu c.a)
  | Qr f -> Result.map (fun (q, r) -> [| q; r |]) (S.qr f c.a)
  | Eigh true | Eig true -> Result.map (fun (w, x) -> [| w; x |]) (S.eigh c.a)
  | Eigh false | Eig false -> Result.map (fun (w, _) -> [| w |]) (S.eigh c.a)
  | Svd None ->
      Result.map (fun (s, _, _) -> [| s |]) (S.svd Reduced c.a)
  | Svd (Some f) -> Result.map (fun (s, u, vh) -> [| s; u; vh |]) (S.svd f c.a)
  | Solve _ -> Result.map (fun y -> [| y |]) (S.solve_triangular c.a c.b)

let arity = function
  | Cholesky _ | Solve _ | Eigh false | Svd None | Eig false -> 1
  | Lu | Svd (Some _) -> 3
  | Qr _ | Eigh true | Eig true -> 2

(* Square matrices half the time, and a vector now and then. *)
let misfits =
  Gen.with_pp pp_misfit
    (let open Gen in
     let solve =
       Solve { triangle = Lower; adjoint = false; unit_diagonal = false }
     in
     let* routine =
       of_list
         [
           Cholesky Lower;
           Lu;
           Qr Reduced;
           Qr Complete;
           Eigh true;
           Eigh false;
           Svd None;
           Svd (Some Reduced);
           Svd (Some Complete);
           Eig true;
           Eig false;
           solve;
         ]
     in
     let* batch = array ~size:(int_range 0 2) (int_range 0 3) in
     let* m = int_range 0 4 in
     let* n = int_range 0 4 in
     let* square = bool in
     let n = if square then m else n in
     let* vector = int_range 0 9 in
     let a = if vector = 0 then [| m |] else Array.append batch [| m; n |] in
     let* cols = int_range 0 3 in
     let* b = Elements.near (Array.append batch [| n; cols |]) in
     let b = if routine = solve then b else [||] in
     let c = { routine; a; b; dsts = [||] } in
     let+ dsts =
       match shapes c with
       | Ok ys ->
           let rec go i acc =
             if i < 0 then constant (Array.of_list acc)
             else
               let* y = Elements.near ys.(i) in
               go (i - 1) (y :: acc)
           in
           go (Array.length ys - 1) []
       | Error _ -> constant (Array.make (arity routine) a)
     in
     { c with dsts })

let law_cpu_refuses (b : Support.backend) c =
  let module K = (val b.kernels) in
  let zeros s =
    A.of_array D.Float32 (Iarray.of_array s)
      (Array.make (Elements.total s) 0.)
  in
  let a = zeros c.a in
  let dtype i =
    match c.routine with
    | Lu when i > 0 -> D.Any D.Int64
    | Eig _ -> D.Any D.Complex64
    | _ -> D.Any D.Float32
  in
  let dsts = Array.mapi (fun i s -> (dtype i, s)) c.dsts in
  let call (g : Misfit.guard) ds =
    let a = g.g a in
    let dst i = A.expect D.Float32 ds.(i) in
    let ints i = A.expect D.Int64 ds.(i) in
    let cdst i = A.expect D.Complex64 ds.(i) in
    match c.routine with
    | Cholesky t -> K.cholesky t ~dst:(dst 0) a
    | Lu -> K.lu ~lu:(dst 0) ~pivots:(ints 1) ~perm:(ints 2) a
    | Qr _ -> K.qr ~q:(dst 0) ~r:(dst 1) a
    | Eigh true -> K.eigh ~w:(dst 0) ~v:(dst 1) a
    | Eigh false -> K.eigh ~w:(dst 0) a
    | Svd None -> K.svd ~s:(dst 0) a
    | Svd (Some _) -> K.svd ~s:(dst 0) ~vectors:(~u:(dst 1), ~vh:(dst 2)) a
    | Eig true -> K.eig ~w:(cdst 0) ~v:(cdst 1) a
    | Eig false -> K.eig ~w:(cdst 0) a
    | Solve { triangle; adjoint; unit_diagonal } ->
        K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst:(dst 0) a
          (g.g (zeros c.b))
  in
  (* A QR's or an SVD's destinations fit as either factors' shapes. *)
  let either r =
    List.exists
      (fun f -> shapes { c with routine = r f } = Ok c.dsts)
      [ Nx_kernel.Reduced; Complete ]
  in
  let fit =
    match c.routine with
    | Qr _ -> either (fun f -> Qr f)
    | Svd (Some _) -> either (fun f -> Svd (Some f))
    | _ -> shapes c = Ok c.dsts
  in
  cover "fits" fit;
  cover "misfits" (not fit);
  Misfit.law ~msg:"a routine" ~what:"a shape" b ~fit dsts call;
  (* nx_cpu.mli: every case that fits is computed. *)
  if fit then
    let fresh (D.Any d, s) = A.Any (A.create Rig.host d (Iarray.of_array s)) in
    equal ~msg:"a call that fits" answer A.Done
      (call { Misfit.g = Fun.id } (Array.map fresh dsts))

(* Routines that do not fit, by a shape, or values of another precision than
   [a]'s: not Done, and nothing written outside their destinations. *)
let test_cpu_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let f32 = A.of_array D.Float32 [| 2; 2 |] [| 4.; 1.; 1.; 3. |] in
  let wide = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let tall = A.of_array D.Float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let c128 = A.of_array D.Complex128 [| 2; 2 |] (Array.make 4 Complex.one) in
  let f = D.Float32 and c = D.Complex128 in
  let misfit ~msg ~what dsts call = Misfit.check ~msg ~what b dsts call in
  misfit ~msg:"not square" ~what:"a shape" [| (D.Any f, [| 2; 3 |]) |]
    (fun g ds -> K.cholesky Lower ~dst:(A.expect f ds.(0)) (g.Misfit.g wide));
  misfit ~msg:"a result of another shape" ~what:"a shape"
    [| (D.Any f, [| 3; 3 |]) |]
    (fun g ds -> K.cholesky Lower ~dst:(A.expect f ds.(0)) (g.Misfit.g f32));
  misfit ~msg:"right-hand sides of other rows" ~what:"a shape"
    [| (D.Any f, [| 3; 2 |]) |]
    (fun g ds ->
      K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
        ~dst:(A.expect f ds.(0)) (g.Misfit.g f32)
        (g.Misfit.g (A.create Rig.host f [| 3; 2 |])));
  misfit ~msg:"a reduced q with a complete r" ~what:"a shape"
    [| (D.Any f, [| 3; 2 |]); (D.Any f, [| 3; 2 |]) |]
    (fun g ds ->
      K.qr ~q:(A.expect f ds.(0)) ~r:(A.expect f ds.(1)) (g.Misfit.g tall));
  (* Values of another precision than [a]'s, and of [a]'s own complex
     dtype's. *)
  misfit ~msg:"float64 singular values of float32" ~what:"a dtype"
    [| (D.Any D.Float64, [| 2 |]) |]
    (fun g ds -> K.svd ~s:(A.expect D.Float64 ds.(0)) (g.Misfit.g f32));
  misfit ~msg:"float32 singular values of complex128" ~what:"a dtype"
    [| (D.Any f, [| 2 |]); (D.Any c, [| 2; 2 |]); (D.Any c, [| 2; 2 |]) |]
    (fun g ds ->
      K.svd ~s:(A.expect f ds.(0))
        ~vectors:(~u:(A.expect c ds.(1)), ~vh:(A.expect c ds.(2)))
        (g.Misfit.g c128));
  misfit ~msg:"float16 eigenvalues of float32" ~what:"a dtype"
    [| (D.Any D.Float16, [| 2 |]) |]
    (fun g ds -> K.eigh ~w:(A.expect D.Float16 ds.(0)) (g.Misfit.g f32));
  misfit ~msg:"float32 eigenvalues of complex128" ~what:"a dtype"
    [| (D.Any f, [| 2 |]); (D.Any c, [| 2; 2 |]) |]
    (fun g ds ->
      K.eigh ~w:(A.expect f ds.(0)) ~v:(A.expect c ds.(1)) (g.Misfit.g c128))

(* Guard pages *)

(* Cases of orders about a vector's tail and a block's, a few matrices at
   most, their operands through any layout. *)
let guard_cases =
  Gen.pair
    (Gen.with_pp pp_case
       (let open Gen in
        let* c = case in
        let* n =
          of_list ~pp:Format.pp_print_int
            [ 0; 1; 2; 3; 5; 8; 9; 17; 26; 33; 63; 64; 65 ]
        in
        let* batch = array ~size:(int_range 0 1) (int_range 1 2) in
        let+ how =
          of_list ~pp:pp_how
            [ Contiguous; Transposed; Stepped; Padded; Reversed ]
        in
        { c with n; batch; how }))
    Elements.sides

(* Each routine gives the same bits with its operands, through the case's
   layout, and its results against an inaccessible page. *)
let law_guards (b : Support.backend) (c, side) =
  cover "a column-reversed operand at the page after"
    (c.how = Reversed && side = Support.After);
  cover "a stepped operand at the page after"
    (c.how = Stepped && side = Support.After);
  cover "a transposed operand at the page before"
    (c.how = Transposed && side = Support.Before);
  cover "a block's order or past" (c.n >= 64);
  let lay = laid_out c.how in
  let _, lu_x = lu_operand c in
  let _, _, svd_x = svd_operand c in
  let routines =
    [
      (Cholesky c.triangle, [| cholesky_operand c |]);
      (Lu, [| lu_x |]);
      (Qr (factors c), [| lu_x |]);
      (Eigh (c.seed mod 2 = 0), [| eigh_operand c |]);
      (Svd (if c.seed mod 3 = 0 then None else Some (factors c)), [| svd_x |]);
      (Eig (c.seed mod 2 = 1), [| eig_operand c |]);
      (solve c, solve_operands c);
    ]
  in
  List.iter
    (fun (r, ops) ->
      let ops = Array.map lay ops in
      let want = Option.get (run b r ops) in
      Elements.with_guards side (fun guard place ->
          match run ~alloc:place b r (Array.map guard ops) with
          | None -> failf "%s declined on guarded memory" (routine_name r)
          | Some got ->
              Array.iteri
                (fun i w ->
                  equal ~msg:(routine_name r) same_bits (result_bits w)
                    (result_bits got.(i)))
                want))
    routines

(* [law_layouts], and qr's bits through the case's layout, which it leaves
   out. *)
let law_layouts_qr (b : Support.backend) c =
  law_layouts b c;
  let _, x = lu_operand c in
  let qr ops = run b (Qr (factors c)) ops in
  match (qr [| x |], qr [| laid_out c.how x |]) with
  | Some y, Some z ->
      equal ~msg:"qr's q" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"qr's r" same_bits (bits y.(1)) (bits z.(1))
  | _ -> fail "nx.cpu declined a qr"

(* A layout whose rows lie further apart than their elements: [law_layouts]
   draws it in about one case in six. *)
let gapped_example = { (fixed D.(Any Float64) 8 3) with how = Padded }

(* One case of each layout and side the guard law covers, and one past a block
   of 64. *)
let guard_examples =
  let c how n = { (fixed D.(Any Float32) n 3) with how } in
  Support.
    [
      (c Reversed 17, After);
      (c Stepped 17, After);
      (c Transposed 17, Before);
      (c Contiguous 65, Before);
    ]

(* The suite *)

let dtype_name = Format.asprintf "%a" pp_dtype

(* What nx_cpu.mli's Matrices section states, under one target table, in its
   order. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let test name f = test name (fun () -> b.around (f b)) in
  let cases name f =
    cases name ~id:dtype_name dtypes (fun dt -> b.around (fun () -> f b dt))
  in
  group b.name
    [
      group "matrices"
        [
          prop
            "computes every case that fits; others are not Done and write \
             nothing outside their results"
            misfits
            (run (law_cpu_refuses b));
          test "routines of other dtypes and shapes are not Done"
            test_cpu_refusals;
          prop ~count:100 ~examples:guard_examples
            "reads and writes no byte past its operands and results, whatever \
             pages lie beside them"
            guard_cases
            (run (law_guards b));
          prop ~examples:[ gapped_example ]
            "bits are the same whatever the layout" case
            (run (law_layouts_qr b));
          cases
            "bits are the same whatever the threads, every routine at every \
             dtype"
            test_cpu_threads;
          cases "bits are the same whatever the threads a matrix's units run on"
            test_cpu_units;
        ];
      group "cholesky"
        [
          prop
            ~examples:(List.map (fun dt -> fixed dt 272 0) dtypes)
            "the factor is within γ(n + 1)" case
            (run (law_cholesky bounds b));
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (run (law_upper b));
        ];
      group "lu"
        [
          prop "the factors are within γ(2k)" case (run (law_lu bounds b));
          test
            "the factors of a pivot whose reciprocal is subnormal are within \
             their bound"
            (test_huge_pivot bounds);
        ];
      group "qr"
        [ prop "the factors are within γ(k(2m + 10))" case (run (law_qr b)) ];
      group "solve_triangular"
        [
          prop "a solve is within γ(2n)" case (run (law_solve bounds b));
          prop "a solve into its operand gives a fresh destination's bits" case
            (run (law_solve_aliased b));
        ];
      group "eigh"
        [
          prop "the results are within γ(16n)" case (run (law_eigh bounds b));
          prop "the eigenvalues are within γ(16n) ‖a‖₂" case
            (run (law_eigh_spectrum b));
          test "the eigenvalues at large orders are within their bound"
            test_eigh_large;
          test
            "the eigenvalues through the two-stage reduction are within their \
             bound"
            test_eigh_two_stage;
          test
            "the eigenvalues of a secular equation with a root beside its pole \
             are within their bound"
            test_eigh_secular;
        ];
      group "svd"
        [
          prop "the results are within γ(16 max (m, n) + 100)" case
            (run (law_svd bounds b));
          prop
            "the results on parts at the least subnormal, below the least \
             normal and near 1 are within their bound"
            spread
            (run (law_svd_spread bounds b));
          test
            "the results on a row whose rest is below the least normal number \
             over u are within their bound"
            test_svd_tiny_rows;
          test
            "the results on a row whose rest is the least subnormal are within \
             their bound"
            (test_svd_spread_row bounds);
          cases
            "the results on matrices of rank 10 at 2^400 over rows of the \
             least subnormal are within their bound"
            (test_svd_deficient bounds);
          cases
            "the results on 16384 matrices of order 2 are within their bound"
            (test_svd_pairs bounds);
          test
            "the vectors on a bidiagonal whose merges deflate are within their \
             bound"
            test_svd_deflation;
          prop "the values are within γ(16 max (m, n) + 100) ‖a‖₂" case
            (run (law_svd_spectrum b));
          test "the values at large orders are within their bound"
            test_svd_large;
          test
            "the values through the two-stage reduction are within their bound"
            test_svd_two_stage;
        ];
      group "eig"
        [
          prop "the results are within γ(16n)" case (run (law_eig bounds b));
          test "the results on a graded cycle are within their bound"
            test_eig_graded_cycle;
          test
            "the results on a subnormal matrix with a complex pair are within \
             their bound"
            test_eig_subnormal_pair;
          prop ~examples:[ twice_i; stalls_x86 ]
            "the eigenvalues of a normal matrix are within γ(16n) ‖a‖_F" case
            (run (law_eig_spectrum b));
          test
            "the eigenvalues of two rotation blocks coupled by 2^-43 are \
             within their bound"
            test_eig_stalled;
          test "the eigenvalues of a graded matrix are balanced"
            test_eig_balanced;
          test
            "the eigenvalues of a matrix spanning the exponent range are \
             balanced before it is scaled"
            test_eig_extremes;
          test "the eigenvalues of Wilkinson's matrices are eigh's"
            test_wilkinson;
        ];
      group "scaled matrices"
        [
          test
            "eigh's, qr's and svd's results on matrices scaled below the least \
             normal number and near the largest are within their bounds"
            (test_scaled bounds);
        ];
    ]

(* The test hooks, run once *)

(* [f ()] on another domain. *)
let on_domain f = Domain.join (Domain.spawn f)

(* An eigh of a complex128 matrix of order 257 with its vectors, through the
   two-stage reduction under [with_two_stage 128]: x86-64's crossovers never
   take it there, and arm64's only from 1024. *)
let eigh_257 () =
  let c = fixed D.(Any Complex128) 257 0 in
  ignore (Option.get (run Support.cpu (Eigh true) [| eigh_operand c |]))

let band () = Nx_cpu.two_stage_splits Nx_cpu.Band

let test_two_stage_nested () =
  if Nx_cpu.most_threads () = 1 then skip ~reason:"one thread" ();
  Nx_cpu.with_two_stage 128 (fun () ->
      let was = band () in
      eigh_257 ();
      greater int ~msg:"from order 128: a band step on several threads"
        ~than:was (band ());
      let was = band () in
      Nx_cpu.with_two_stage 4096 eigh_257;
      equal int ~msg:"inside a nested order of 4096: none" was (band ());
      eigh_257 ();
      greater int ~msg:"once the nested order returns: one again" ~than:was
        (band ()))

let test_two_stage_domains () =
  if Nx_cpu.most_threads () = 1 then skip ~reason:"one thread" ();
  Nx_cpu.with_two_stage 128 (fun () ->
      let splits =
        on_domain (fun () ->
            let was = band () in
            eigh_257 ();
            band () - was)
      in
      equal int ~msg:"band steps on several threads there" 0 splits)

(* An svd of a float64 matrix of order 200 with its vectors, through the
   two-stage reduction: one Back step for u and one for vh. *)
let test_two_back_steps () =
  if Nx_cpu.most_threads () = 1 then skip ~reason:"one thread" ();
  Nx_cpu.with_two_stage 128 (fun () ->
      let c = fixed D.(Any Float64) 200 200 in
      let _, _, s = svd_operand c in
      let was = Nx_cpu.two_stage_splits Nx_cpu.Back in
      ignore
        (Option.get (run Support.cpu (Svd (Some Nx_kernel.Reduced)) [| s |]));
      equal int (was + 2) (Nx_cpu.two_stage_splits Nx_cpu.Back))

let test_one_thread () =
  equal int ~msg:"under one_thread" 1 (Nx_cpu.one_thread Nx_cpu.most_threads);
  equal int ~msg:"after a nested one_thread returns" 1
    (Nx_cpu.one_thread (fun () ->
         Nx_cpu.one_thread ignore;
         Nx_cpu.most_threads ()))

let test_one_thread_domains () =
  let host = Nx_cpu.most_threads () in
  if host = 1 then skip ~reason:"one thread" ();
  equal int host (Nx_cpu.one_thread (fun () -> on_domain Nx_cpu.most_threads))

let hooks =
  group "test hooks"
    [
      cases
        "with_two_stage raises Invalid_argument for an order below 128 but 0"
        ~id:string_of_int [ -1; 1; 127 ] (fun n ->
          raises_match Exn.invalid_arg (fun () ->
              Nx_cpu.with_two_stage n ignore));
      cases "with_two_stage takes 0 and 128" ~id:string_of_int [ 0; 128 ]
        (fun n -> Nx_cpu.with_two_stage n ignore);
      test "with_two_stage's nested order holds until it returns"
        test_two_stage_nested;
      test "with_two_stage leaves another domain's order" test_two_stage_domains;
      test "two_stage_splits counts an svd with vectors' two Back steps"
        test_two_back_steps;
      test "most_threads is 1 under one_thread, nested ones included"
        test_one_thread;
      test "one_thread leaves another domain's threads" test_one_thread_domains;
    ]

(* The references' division *)

(* A float32 number: a sign, an exponent from below the least subnormal's
   to past the largest finite's, a mantissa; under [tiny], below 2^-102,
   where nx_kinds.h's Fdiv scales a dividend both of whose parts lie. *)
let float32 ~tiny =
  let open Gen in
  let* e = if tiny then int_range (-150) (-103) else int_range (-150) 128 in
  let* m = int_range 0 0x7FFFFF in
  let+ negative = bool in
  let x = Float.ldexp (1. +. Float.ldexp (float_of_int m) (-23)) e in
  Int32.float_of_bits (Int32.bits_of_float (if negative then -.x else x))

let complex ~tiny =
  let open Gen in
  let* re = float32 ~tiny in
  let+ im = float32 ~tiny in
  { Complex.re; im }

let quotients =
  let pp ppf ((x : Complex.t), (d : Complex.t)) =
    Format.fprintf ppf "(%h, %h) / (%h, %h)" x.re x.im d.re d.im
  in
  Gen.with_pp pp
    (let open Gen in
     let* tiny = bool in
     let* x = complex ~tiny in
     let+ d = complex ~tiny:false in
     (x, d))

(* The bit references' complex division, at gradual underflow, gives
   nx.cpu's Fdiv's bits, any NaN as one. A divisor of no imaginary part
   is left out: there the references divide each part by its real part,
   as a routine divides by Cholesky's real diagonal. *)
let law_divd ((x : Complex.t), (d : Complex.t)) =
  assume (d.im <> 0.);
  cover "a dividend below 2^-102 in both parts"
    (Float.abs x.re < 0x1p-102 && Float.abs x.im < 0x1p-102);
  cover "a divisor whose imaginary part is the larger"
    (Float.abs d.im > Float.abs d.re);
  let bits32 f =
    if Float.is_nan f then 0x7FC00000
    else Int32.to_int (Int32.bits_of_float f) land 0xFFFFFFFF
  in
  let ar = arith ~underflow:Support.Gradual D.(Any Complex64) in
  let got = (divd ar [| x |] d).(0) in
  let want =
    Nx_kinds_support.c64 "fdiv" (Array.map bits32 [| x.re; x.im; d.re; d.im |])
  in
  let canonical b = bits32 (Int32.float_of_bits (Int32.of_int b)) in
  equal (pair int int)
    (canonical want.(0), canonical want.(1))
    (bits32 got.re, bits32 got.im)

let () =
  exit
    (Windtrap.run "nx_cpu.linalg"
       (List.map cpu Support.cpus
       @ [
           hooks;
           prop "the bit references' complex division is Fdiv's" quotients
             law_divd;
         ]))
