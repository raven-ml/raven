(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Factorisations and triangular solves through every kernel library the host
   runs: LU's pivots as nx_kernel.mli's lu chooses them; a failing matrix
   NaN alone in its batch; a finite matrix NaN in no result, though an
   element's modulus overflows; only the triangle a routine names read; the
   dtypes routines take. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module Support = Nx_kernels_support
open Linalg_cases

(* The complex dtype of [dt]'s precision: eig's results'. *)
let complex_of (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> D.Any D.Complex64
  | _ -> D.Any D.Complex128

(* The float of [dt]'s precision: svd's and eigh's values'. *)
let real_of (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> D.Any D.Float32
  | _ -> D.Any D.Float64

(* Spread magnitudes *)

(* A complex matrix whose largest element is finite but of a modulus past
   the largest float, diag (0.9 max (1 + i), 1): a matrix of numbers, on
   which svd's and eig's results, with vectors and without, hold no NaN.
   Scaled by the modulus, the routines would scale by 0 and give NaN. *)
let test_overflowing_modulus (b : Support.backend) () =
  let big = 0.9 *. Float.max_float in
  let a =
    Complex.[| { re = big; im = big }; zero; zero; one |]
  in
  let dt = D.Any D.Complex128 in
  let x = of_values dt [| 2; 2 |] a in
  List.iter
    (fun (msg, r) ->
      match run b r [| x |] with
      | Some y ->
          Array.iteri
            (fun k y ->
              let v = values y in
              equal ~msg:(Printf.sprintf "%s: numbers in result %d" msg k) int
                (2 * Array.length v) (numbers dt v))
            y
      | None -> ())
    [
      ("svd", Svd (Some Reduced)); ("eig", Eig true); ("svd's values", Svd None);
    ]

(* lu's pivot: a scan of the column from the diagonal down that
   moves to each later row of greater magnitude, |re| + |im| on complex
   numbers, a NaN never greater and nothing greater than a NaN. Each case is
   a column and the pivot the rule picks. *)
let test_pivots (b : Support.backend) () =
  let nan = Float.nan in
  let c re im = { Complex.re; im } in
  let pivot ~msg dt column want =
    let m = Array.length column in
    match run b Lu [| of_values dt [| m; 1 |] column |] with
    | None -> ()
    | Some y ->
        equal ~msg (array int) [| want |] (int64s y.(1));
        let perm = Array.init m Fun.id in
        perm.(0) <- want;
        perm.(want) <- 0;
        equal ~msg:(msg ^ ": perm") (array int) perm (int64s y.(2))
  in
  List.iter
    (fun dt ->
      pivot ~msg:"a zero diagonal" dt [| c 0. 0.; c 1. 0. |] 1;
      pivot ~msg:"a tie keeps the first" dt [| c 1. 0.; c (-1.) 0. |] 0;
      pivot ~msg:"the greatest past a smaller one" dt
        [| c 1. 0.; c 3. 0.; c (-2.) 0.; c (-4.) 0. |] 3;
      pivot ~msg:"nothing is greater than a NaN" dt
        [| c nan 0.; c 5. 0.; c (-7.) 0. |] 0;
      pivot ~msg:"a NaN is never greater" dt [| c 1. 0.; c nan 0.; c 2. 0. |] 2)
    dtypes;
  List.iter
    (fun dt ->
      pivot ~msg:"|re| + |im|, not the modulus" dt [| c 3. 0.; c 2. 2. |] 1)
    D.[ Any Complex64; Any Complex128 ]

(* Failures and reads *)

(* Matrix [m] of [v], [rows] × [cols] each. *)
let matrix v ~rows ~cols m = Array.sub v (m * rows * cols) (rows * cols)

(* A matrix that fails, between two that do not, is NaN, and leaves theirs
   the bits they have alone. [spoil] makes a matrix's values fail. *)
let check_alone (b : Support.backend) r ~dt ?(result = dt) ~rows ~cols ~spoil
    ops =
  let n = Array.length ops in
  let vs = Array.map values ops in
  let three k = Array.concat [ vs.(k); (if k = 0 then spoil vs.(k) else vs.(k)); vs.(k) ] in
  let shape k =
    let (A.Any x) = ops.(k) in
    Array.append [| 3 |] (Iarray.to_array (L.shape (A.layout x)))
  in
  let batch = Array.init n (fun k -> of_values dt (shape k) (three k)) in
  match (run b r batch, run b r ops) with
  | Some y, Some alone ->
      let yv = values y.(0) in
      let want = bits alone.(0) in
      let mat m =
        bits (of_values result [| rows; cols |] (matrix yv ~rows ~cols m))
      in
      equal ~msg:"the first is alone" same_bits want (mat 0);
      equal ~msg:"the last is alone" same_bits want (mat 2);
      equal ~msg:"numbers in the failing one" int 0
        (numbers result (matrix yv ~rows ~cols 1))
  | _ -> ()

(* [x] with element (k, k) at -1, which makes the pivot of column [k]
   negative. *)
let cholesky_fails (b : Support.backend) c x k =
  let spoil v =
    let v = Array.copy v in
    v.((k * c.n) + k) <- { Complex.re = -1.; im = 0. };
    v
  in
  check_alone b (Cholesky Lower) ~dt:c.dt ~rows:c.n ~cols:c.n ~spoil [| x |]

let law_cholesky_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then cholesky_fails b c (cholesky_operand c) (c.seed mod c.n)

(* Pivots failing past the first block of 64 columns: column 64, the
   second block's first, 100 within it, and 128, a third block's only
   column. *)
let test_cholesky_fails_late (b : Support.backend) () =
  let c =
    {
      dt = D.(Any Float64);
      batch = [||];
      n = 129;
      r = 0;
      triangle = Lower;
      adjoint = false;
      unit_diagonal = false;
      seed = 7;
      how = Contiguous;
      scale = Plain;
    }
  in
  List.iter
    (fun dt ->
      let c = { c with dt } in
      List.iter (cholesky_fails b c (cholesky_operand c)) [ 64; 100; 128 ])
    dtypes

(* NaN or an infinity in the lower triangle. *)
let law_eigh_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then
    let i = c.seed mod c.n in
    let j = c.seed / c.n mod (i + 1) in
    let bad = [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.((i * c.n) + j) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(real_of c.dt) ~rows:c.n ~cols:1
          ~spoil [| eigh_operand c |])
      [ Eigh true; Eigh false ]

(* NaN or an infinity anywhere. *)
let law_eig_fails (b : Support.backend) c =
  let c = one c in
  if c.n > 0 then
    let p = c.seed mod (c.n * c.n) in
    let bad = Float.[| nan; infinity; neg_infinity |].(c.seed mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.(p) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(complex_of c.dt) ~rows:c.n ~cols:1
          ~spoil [| eig_operand c |])
      [ Eig true; Eig false ]

(* NaN or an infinity anywhere. *)
let law_svd_fails (b : Support.backend) c =
  let c = one c in
  let m, n, x = svd_operand c in
  if m > 0 && n > 0 then
    let p = c.seed mod (m * n) in
    let bad = [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed / 7 mod 3) in
    let spoil v =
      let v = Array.copy v in
      v.(p) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(real_of c.dt) ~rows:(min m n)
          ~cols:1 ~spoil [| x |])
      [ Svd (Some (factors c)); Svd None ]

let law_solve_fails (b : Support.backend) c =
  let c = { (one c) with unit_diagonal = false; r = max 1 c.r } in
  if c.n > 0 then
    let k = c.seed mod c.n in
    let spoil v =
      let v = Array.copy v in
      v.((k * c.n) + k) <- Complex.zero;
      v
    in
    check_alone b (solve c) ~dt:c.dt ~rows:c.n ~cols:c.r ~spoil
      (solve_operands c)

(* [v] with NaN wherever [unread i j] holds, and imaginary parts drawn on the
   diagonal where [diagonal_im]. *)
let garbage c v ~unread ~diagonal_im =
  let r = Random.State.make [| c.seed + 1 |] in
  let n = c.n in
  Array.mapi
    (fun k (x : Complex.t) ->
      let i = k / n mod n and j = k mod n in
      if unread i j then { Complex.re = Float.nan; im = Float.nan }
      else if i = j && diagonal_im && is_complex c.dt then { x with im = uniform r }
      else x)
    v

(* What a routine does not read changes no bit of its results. *)
let law_cholesky_reads (b : Support.backend) c =
  let x = cholesky_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  match (run b (Cholesky Lower) [| x |], run b (Cholesky Lower) [| of_values c.dt (square c) v |]) with
  | Some y, Some z -> equal same_bits (bits y.(0)) (bits z.(0))
  | _ -> ()

let law_eigh_reads (b : Support.backend) c =
  let x = eigh_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  let y = of_values c.dt (square c) v in
  match (run b (Eigh true) [| x |], run b (Eigh true) [| y |]) with
  | Some y, Some z ->
      equal ~msg:"w" same_bits (bits y.(0)) (bits z.(0));
      equal ~msg:"v" same_bits (bits y.(1)) (bits z.(1))
  | _ -> ()

let law_solve_reads (b : Support.backend) c =
  let ops = solve_operands c in
  let unread i j =
    (if c.triangle = Lower then j > i else j < i) || (c.unit_diagonal && i = j)
  in
  let v = garbage c (values ops.(0)) ~unread ~diagonal_im:false in
  let spoilt = [| of_values c.dt (square c) v; ops.(1) |] in
  match (run b (solve c) ops, run b (solve c) spoilt) with
  | Some y, Some z -> equal same_bits (bits y.(0)) (bits z.(0))
  | _ -> ()

(* Dtypes *)

(* Every routine computes or declines float32, float64, complex64 and
   complex128, and answers no Wrong_dtype there. A call at another dtype,
   or of [eig]'s, [svd]'s or [eigh]'s values of another precision than
   [a]'s, does not fit: not Done, and nothing written outside its
   destinations. *)
let test_dtypes (b : Support.backend) () =
  let module K = (val b.kernels) in
  let i64 = D.Any D.Int64 in
  List.iter
    (fun (D.Any dt) ->
      let takes =
        match dt with
        | D.Float32 | D.Float64 | D.Complex64 | D.Complex128 -> true
        | _ -> false
      in
      let sq = [| 2; 2 |] and n = [| 2 |] in
      let a = A.create b.device dt (Iarray.of_array sq) in
      let wide = match dt with D.Float64 | D.Complex128 -> true | _ -> false in
      (* [call] into destinations of the dtypes and shapes [dsts]. *)
      let case ~msg ~fit dsts call =
        if not fit then Misfit.check ~msg ~what:"a dtype" b dsts call
        else
          let fresh (D.Any d, s) =
            A.Any (A.create b.device d (Iarray.of_array s))
          in
          let got = call { Misfit.g = Fun.id } (Array.map fresh dsts) in
          if got = A.Wrong_dtype then failf "%s answered Wrong_dtype" msg
      in
      let x ds k = A.expect dt ds.(k) in
      let eig (type c) (c : (Complex.t, c) D.t) =
        let fit = takes && wide = (D.Any c = D.Any D.Complex128) in
        let msg = Format.asprintf "eig of %a into %a" D.pp dt D.pp c in
        let z ds k = A.expect c ds.(k) in
        case ~msg ~fit [| (D.Any c, n); (D.Any c, sq) |] (fun g ds ->
            K.eig ~w:(z ds 0) ~v:(z ds 1) (g.Misfit.g a));
        case ~msg:(msg ^ ", values") ~fit [| (D.Any c, n) |] (fun g ds ->
            K.eig ~w:(z ds 0) (g.Misfit.g a))
      in
      eig D.Complex64;
      eig D.Complex128;
      let real (type r) (f : (float, r) D.t) =
        let fit =
          takes && wide = (D.Any f = D.Any D.Float64) && D.bits f >= 32
        in
        let msg name = Format.asprintf "%s of %a into %a" name D.pp dt D.pp f in
        let y ds = A.expect f ds.(0) and m = (D.Any dt, sq) in
        case ~msg:(msg "svd") ~fit [| (D.Any f, n); m; m |] (fun g ds ->
            K.svd ~s:(y ds)
              ~vectors:(~u:(x ds 1), ~vh:(x ds 2))
              (g.Misfit.g a));
        case ~msg:(msg "svd's values") ~fit [| (D.Any f, n) |] (fun g ds ->
            K.svd ~s:(y ds) (g.Misfit.g a));
        case ~msg:(msg "eigh") ~fit [| (D.Any f, n); m |] (fun g ds ->
            K.eigh ~w:(y ds) ~v:(x ds 1) (g.Misfit.g a));
        case ~msg:(msg "eigh's values") ~fit [| (D.Any f, n) |] (fun g ds ->
            K.eigh ~w:(y ds) (g.Misfit.g a))
      in
      real D.Float16;
      real D.Float32;
      real D.Float64;
      let m = (D.Any dt, sq) in
      List.iter
        (fun (name, dsts, call) ->
          let msg = Format.asprintf "%s of %a" name D.pp dt in
          case ~msg ~fit:takes dsts call)
        [
          ( "cholesky",
            [| m |],
            fun g ds -> K.cholesky Lower ~dst:(x ds 0) (g.Misfit.g a) );
          ( "lu",
            [| m; (i64, n); (i64, n) |],
            fun g ds ->
              K.lu ~lu:(x ds 0)
                ~pivots:(A.expect D.Int64 ds.(1))
                ~perm:(A.expect D.Int64 ds.(2))
                (g.Misfit.g a) );
          ( "qr",
            [| m; m |],
            fun g ds -> K.qr ~q:(x ds 0) ~r:(x ds 1) (g.Misfit.g a) );
          ( "solve_triangular",
            [| m |],
            fun g ds ->
              K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
                ~dst:(x ds 0) (g.Misfit.g a)
                (g.Misfit.g (A.create b.device dt (Iarray.of_array sq))) );
        ])
    D.all

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "svd of a matrix with NaN or an infinity is NaN alone" case
        (run (law_svd_fails b));
      prop "eig of a matrix with NaN or an infinity is NaN alone" case
        (run (law_eig_fails b));
      prop "a matrix that is not definite is NaN alone" case
        (run (law_cholesky_fails b));
      prop "a singular triangle is NaN alone" case (run (law_solve_fails b));
      prop "eigh of a matrix with NaN or an infinity is NaN alone" case
        (run (law_eigh_fails b));
      prop "eigh reads the lower triangle and the diagonal's real parts" case
        (run (law_eigh_reads b));
      prop "Cholesky reads the lower triangle and the diagonal's real parts"
        case (run (law_cholesky_reads b));
      prop "a solve reads its triangle, and its diagonal unless unit" case
        (run (law_solve_reads b));
      test "a pivot failing past the first block is NaN alone" (fun () ->
          b.around (test_cholesky_fails_late b));
      test "LU's pivot is the scan's" (fun () -> b.around (test_pivots b));
      test
        "routines take floats and complex numbers alone; a call at another \
         dtype is not Done and writes nothing outside its destinations"
        (fun () -> b.around (test_dtypes b));
      test "a finite matrix gives numbers in every result" (fun () ->
          b.around (test_overflowing_modulus b));
    ]

(* Memory of calls in a loop

   nx_cuda.mli's memory of the matrix routines, held by calls in a loop:
   float32 elements of 4 bytes. *)

let wait (A.Any x) = Rig.Buffer.wait (A.buffer x) Read_write

(* A solve whose [a]'s batch axes do not form one run of strides copies
   [a]: [batch n²] elements. *)
let test_loop_solve (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 8 and d = b.device in
  let base = A.create d D.Float32 [| 3; 2; n; n |] in
  let a =
    A.v D.Float32
      (A.Layout.v ~strides:[| n * n; 2 * n * n; n; 1 |] [| 2; 3; n; n |])
      (A.buffer base)
  in
  let rhs = A.create d D.Float32 [| 2; 3; n; 4 |] in
  let dst = A.create d D.Float32 [| 2; 3; n; 4 |] in
  Kernels_cases.loop_held b ~bound:(6 * n * n * 4) (fun () ->
      equal answer A.Done
        (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst a rhs);
      wait (A.Any dst))

(* A qr of 64 columns copies [a], [m n] elements, and takes
   [m b + 2 b max(n, c) + b² ⌈k / b⌉] more, [b = min k 128]; each piece
   from aligned bytes. *)
let test_loop_qr (b : Support.backend) () =
  let module K = (val b.kernels) in
  let m = 256 and n = 64 and d = b.device in
  let a = A.create d D.Float32 [| m; n |] in
  let q = A.create d D.Float32 [| m; n |] and r = A.create d D.Float32 [| n; n |] in
  let k = n and c = n in
  let bk = Int.min k 128 in
  let elements = (m * n) + (m * bk) + (2 * bk * Int.max n c) + (bk * bk * ((k + bk - 1) / bk)) in
  Kernels_cases.loop_held b ~bound:((4 * elements) + 1024) (fun () ->
      equal answer A.Done (K.qr ~q ~r a);
      wait (A.Any r))

(* An eigh with [v] of order [n] takes [2n² + 97 n + 2048] elements,
   [3 n² + 78 n - 2 l (32 - l)] reals and [24 n + 32] bytes, [l] its last
   leaf's order, and at most 512 bytes of alignment a matrix and 2 KiB a
   call; and [n²] elements more where [a] is transposed, which it copies. *)
let test_loop_eigh ~n ~transposed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let d = b.device in
  let base = A.create d D.Float32 [| n; n |] in
  let a =
    if transposed then
      A.v D.Float32 (A.Layout.v ~strides:[| 1; n |] [| n; n |]) (A.buffer base)
    else base
  in
  let w = A.create d D.Float32 [| n |] and v = A.create d D.Float32 [| n; n |] in
  let l = n - (32 * (((n + 31) / 32) - 1)) in
  let copy = if transposed then n * n else 0 in
  let elements = (2 * n * n) + (97 * n) + 2048 + copy in
  let reals = (3 * n * n) + (78 * n) - (2 * l * (32 - l)) in
  let bytes = (4 * (elements + reals)) + (24 * n) + 32 + 512 + 2048 in
  Kernels_cases.loop_held b ~bound:bytes (fun () ->
      equal answer A.Done (K.eigh ~w ~v a);
      wait (A.Any v))

let cuda (b : Support.backend) =
  let test name f = test name (fun () -> b.around (f b)) in
  group ("nx.cuda " ^ b.name)
    [
      test "solves in a loop hold one solve's copy" test_loop_solve;
      test "qrs in a loop hold one qr's memory" test_loop_qr;
      test "eighs in a loop hold one eigh's memory"
        (test_loop_eigh ~n:64 ~transposed:false);
      test "eighs of a copied operand in a loop hold one eigh's memory"
        (test_loop_eigh ~n:80 ~transposed:true);
    ]

let backends = Support.cpus @ Support.gpus ()

let () =
  exit
    (Windtrap.run "nx_kernel.linalg"
       (List.map laws backends @ List.map cuda (Support.gpus ())))
