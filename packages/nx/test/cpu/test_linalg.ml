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
    alone = (fun ~e dt k -> epsilon ~e dt k);
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

(* The routines with work of their own on a matrix's threads, besides
   their products: solves' units, LU's steps, eigh's and svd's
   matrix-vector products and their merges' roots. *)
let with_units =
  [ "cholesky"; "solve"; "lu"; "eigh"; "svd"; "svd's values" ]

(* nx.cpu computes [routines], by name, of [c]'s matrix alone, on the
   call's threads, and of copies of it in a wide batch, one a thread:
   every result of every matrix of the batch has the bits of the
   matrix's alone. *)
let threads_agree (b : Support.backend) routines c =
  let c = wide c in
  let k = count c in
  let one = { c with batch = [||] } in
  let x = cholesky_operand one and ops = solve_operands one in
  let _, lu = lu_operand one in
  let _, _, s = svd_operand one in
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
      if List.mem msg routines then
        match (run b r (Array.map (repeated c) ops), run b r ops) with
        | Some y, Some z -> agree msg y z
        | _ -> failf "nx.cpu declined %s at %a" msg pp_dtype c.dt)
    [
      ("cholesky", Cholesky Lower, [| x |]);
      ("solve", solve one, ops);
      ("lu", Lu, [| lu |]);
      ("qr", Qr (factors one), [| lu |]);
      ("eigh", Eigh true, [| eigh_operand one |]);
      ("svd", Svd (Some (factors one)), [| s |]);
      ("svd's values", Svd None, [| s |]);
      ("eig", Eig true, [| eig_operand one |]);
    ]

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
let test_cpu_threads b dt =
  threads_agree b ([ "qr"; "eig" ] @ with_units) (fixed dt 65 3)

(* Matrices of order 256 and 160 right-hand sides, of the routines whose
   work on a matrix's threads goes beyond their products, which give the
   bits of one thread. On one core nothing runs on several threads. *)
let test_cpu_units b dt =
  if Domain.recommended_domain_count () = 1 then skip ~reason:"one core" ();
  threads_agree b with_units (fixed dt 256 160)

(* One matrix of order 272, past four blocks of 64 and a partial one. *)
let test_cpu_large b dt = law_cholesky bounds b (fixed dt 272 0)

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
  Misfit.law ~msg:"a routine" ~what:"a shape" b ~fit dsts call

(* Routines that do not fit, by a dtype, a shape, or values of another
   precision than [a]'s: not Done, and nothing written outside their
   destinations. *)
let test_cpu_refusals (b : Support.backend) () =
  let module K = (val b.kernels) in
  let f32 = A.of_array D.Float32 [| 2; 2 |] [| 4.; 1.; 1.; 3. |] in
  let i32 = A.of_array D.Int32 [| 2; 2 |] [| 4l; 1l; 1l; 3l |] in
  let wide = A.of_array D.Float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let tall = A.of_array D.Float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let c128 = A.of_array D.Complex128 [| 2; 2 |] (Array.make 4 Complex.one) in
  let f = D.Float32 and c = D.Complex128 in
  let misfit ~msg ~what dsts call = Misfit.check ~msg ~what b dsts call in
  misfit ~msg:"int32" ~what:"a dtype" [| (D.Any D.Int32, [| 2; 2 |]) |]
    (fun g ds ->
      K.cholesky Lower ~dst:(A.expect D.Int32 ds.(0)) (g.Misfit.g i32));
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

(* The suite *)

let dtype_name = Format.asprintf "%a" pp_dtype

(* What nx_cpu.mli states beyond the contract: its bounds, the Upper factor's
   bits, a solve into its operand, layout and thread invariance, and the
   matrices its algorithms find hard. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  let test name f = test name (fun () -> b.around (f b)) in
  let cases name f =
    cases name ~id:dtype_name dtypes (fun dt -> b.around (fun () -> f b dt))
  in
  group b.name
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case (run (law_cholesky bounds b));
          cases "Cholesky's factor of order 272" test_cpu_large;
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (run (law_upper b));
          prop "a triangular solve within γ(2n)" case (run (law_solve bounds b));
          prop "LU's factors within γ(2k)" case (run (law_lu bounds b));
          test "LU of a pivot whose reciprocal is subnormal" (test_huge_pivot bounds);
          prop "QR's factors within γ(k(2m + 10))" case (run (law_qr b));
          prop "eigh's results within γ(16n)" case (run (law_eigh bounds b));
          prop "svd's results within γ(16 max (m, n) + 100)" case
            (run (law_svd bounds b));
          prop
            "svd's results on parts at the least subnormal, below the least \
             normal and near 1"
            spread (run (law_svd_spread bounds b));
          test
            "svd's results within their bound on a row whose rest is below \
             the least normal number over u"
            test_svd_tiny_rows;
          test "svd's results on a row whose rest is the least subnormal"
            (test_svd_spread_row bounds);
          cases
            "svd's results on matrices of rank 10 at 2^400 over rows of the \
             least subnormal"
            (test_svd_deficient bounds);
          cases "svd's results on 16384 matrices of order 2"
            (test_svd_pairs bounds);
          test "svd's vectors on a bidiagonal whose merges deflate"
            test_svd_deflation;
          prop "eig's results within γ(16n)" case (run (law_eig bounds b));
          test "eig's results on a graded cycle" test_eig_graded_cycle;
          test "eig's results on a subnormal matrix with a complex pair"
            test_eig_subnormal_pair;
          test "eigh's, qr's and svd's results on matrices scaled below the \
                least normal number and near the largest"
            (test_scaled bounds);
        ];
      group "each spectrum is within its bound of the exact one"
        [
          prop "eigh's eigenvalues within γ(16n) ‖a‖₂" case
            (run (law_eigh_spectrum b));
          test "eigh's eigenvalues at large orders" test_eigh_large;
          test "eigh's eigenvalues of a secular equation with a root beside \
                its pole"
            test_eigh_secular;
          prop "svd's values within γ(16 max (m, n) + 100) ‖a‖₂" case
            (run (law_svd_spectrum b));
          test "svd's values at large orders" test_svd_large;
          prop "eig's eigenvalues of a normal matrix within γ(16n) ‖a‖_F" case
            (run (law_eig_spectrum b));
          test "eig's eigenvalues of a graded matrix, balanced"
            test_eig_balanced;
          test "eig's eigenvalues of a matrix spanning the exponent range, \
                balanced before it is scaled"
            test_eig_extremes;
          test "eig's eigenvalues of Wilkinson's matrices are eigh's"
            test_wilkinson;
        ];
      group "bits are a function of the values alone"
        [
          prop "whatever the layout" case (run (law_layouts b));
          cases "whatever the threads, every routine at every dtype"
            test_cpu_threads;
          cases "whatever the threads a matrix's units run on" test_cpu_units;
          prop "a solve into its operand gives a fresh destination's bits" case
            (run (law_solve_aliased b));
        ];
      group "misfits are not Done and write nothing outside their results"
        [
          prop "computes what Nx_kernel.Shape fits, of any other shapes"
            misfits (run (law_cpu_refuses b));
          test "of dtypes and shapes" test_cpu_refusals;
        ];
      prop ~count:100
        "reads and writes no byte past its operands and results, whatever \
         pages lie beside them"
        guard_cases (run (law_guards b));
    ]

let () = exit (Windtrap.run "nx_cpu.linalg" (List.map cpu Support.cpus))
