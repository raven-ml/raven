(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The matrix routines through every kernel library the host runs, as
   nx_kernel.mli states them, routine by routine: a failing matrix NaN alone in
   its batch; only the triangle a routine names read; lu's pivots and its zero
   pivot; a finite matrix NaN in no result, though an element's modulus
   overflows; a destination over its operand's bytes refused; the dtypes
   routines take. The defining relations within each library's bounds are
   linalg_laws.ml's, run by each library's own suite. *)

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

(* A complex matrix whose largest element is finite but of a modulus past the
   largest float, diag (0.9 max (1 + i), 1): a matrix of numbers, on which svd's
   and eig's results, with vectors and without, hold no NaN. Scaled by the
   modulus, the routines would scale by 0 and give NaN. *)
let test_overflowing_modulus (b : Support.backend) () =
  let big = 0.9 *. Float.max_float in
  let a = Complex.[| { re = big; im = big }; zero; zero; one |] in
  let dt = D.Any D.Complex128 in
  let x = of_values dt [| 2; 2 |] a in
  List.iter
    (fun (msg, r) ->
      match run b r [| x |] with
      | Some y ->
          Array.iteri
            (fun k y ->
              let v = values y in
              equal
                ~msg:(Printf.sprintf "%s: numbers in result %d" msg k)
                int
                (2 * Array.length v)
                (numbers dt v))
            y
      | None -> skip ~reason:"declines complex128, as its library states" ())
    [
      ("svd", Svd (Some Reduced)); ("eig", Eig true); ("svd's values", Svd None);
    ]

(* lu's pivot (nx_kernel.mli): a scan of the column from the diagonal down that
   moves to each later row of greater magnitude, |re| + |im| on complex numbers,
   a NaN never greater and nothing greater than a NaN. Each row is a rule, a
   column and the pivot the rule picks. *)
let pivot_rules =
  let nan = Float.nan in
  let c re im = { Complex.re; im } in
  let real =
    [
      ("a zero diagonal", [| c 0. 0.; c 1. 0. |], 1);
      ("a tie keeps the first", [| c 1. 0.; c (-1.) 0. |], 0);
      ( "the greatest past a smaller one",
        [| c 1. 0.; c 3. 0.; c (-2.) 0.; c (-4.) 0. |],
        3 );
      ("nothing is greater than a NaN", [| c nan 0.; c 5. 0.; c (-7.) 0. |], 0);
      ("a NaN is never greater", [| c 1. 0.; c nan 0.; c 2. 0. |], 2);
    ]
  in
  List.concat_map (fun dt -> List.map (fun r -> (dt, r)) real) dtypes
  @ List.map
      (fun dt ->
        (dt, ("|re| + |im|, not the modulus", [| c 3. 0.; c 2. 2. |], 1)))
      D.[ Any Complex64; Any Complex128 ]

let pivot_id (dt, (rule, _, _)) = Format.asprintf "%s, %a" rule pp_dtype dt

(* A library that declines a dtype, as it states, skips its rows. *)
let declined () = skip ~reason:"a dtype its library declines, as it states" ()

let test_pivot (b : Support.backend) (dt, (_, column, want)) =
  let m = Array.length column in
  match run b Lu [| of_values dt [| m; 1 |] column |] with
  | None -> declined ()
  | Some y ->
      equal ~msg:"pivot" (array int) [| want |] (int64s y.(1));
      let perm = Array.init m Fun.id in
      perm.(0) <- want;
      perm.(want) <- 0;
      equal ~msg:"perm" (array int) perm (int64s y.(2))

(* nx_kernel.mli: a zero pivot stays in U and its column of L is not divided.
   [[0; 1]; [0; 2]]'s first column has no row greater than its zero, so row 0
   stays, L's element below it stays 0, not 0 / 0, and U's last element is 2 - 0
   · 1. *)
let test_zero_pivot (b : Support.backend) dt =
  let c re = { Complex.re; im = 0. } in
  let a = of_values dt [| 2; 2 |] [| c 0.; c 1.; c 0.; c 2. |] in
  match run b Lu [| a |] with
  | None -> declined ()
  | Some y ->
      equal ~msg:"lu" same_bits (bits a) (bits y.(0));
      equal ~msg:"pivots" (array int) [| 0; 1 |] (int64s y.(1));
      equal ~msg:"perm" (array int) [| 0; 1 |] (int64s y.(2))

(* Failures and reads *)

(* Matrix [m] of [v], [rows] × [cols] each. *)
let matrix v ~rows ~cols m = Array.sub v (m * rows * cols) (rows * cols)

(* A matrix that fails, between two that do not, is NaN, and leaves theirs the
   bits they have alone. [spoil] makes a matrix's values fail. A library
   declines only a dtype it states it declines ({!Linalg_cases.run}). *)
let check_alone (b : Support.backend) r ~dt ?(result = dt) ~rows ~cols ~spoil
    ops =
  let n = Array.length ops in
  let vs = Array.map values ops in
  let three k =
    Array.concat [ vs.(k); (if k = 0 then spoil vs.(k) else vs.(k)); vs.(k) ]
  in
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
  | None, None -> collect "a dtype its library declines"
  | _ -> fail "declined the batch or the matrix alone, not both"

(* The classes a law's cases fall in: each dtype, an order past 0, past a block
   of 64, and a Subnormal scale. *)
let classes c =
  List.iter (fun d -> cover (Format.asprintf "%a" pp_dtype d) (c.dt = d)) dtypes;
  cover "an order past 0" (c.n > 0);
  cover "an order past a block of 64" (c.n > 64);
  cover "a Subnormal scale" (c.scale = Subnormal)

(* The cases the laws draw: orders below a block of 64, of 1 one time in
   eight. *)
let small = case_over [ 0; 1; 2; 3; 5; 8; 17; 26 ]

(* Orders past a block of 64: 65, the second block's first column, at each
   dtype, and 129, a third block's, at one (on nx.cuda an eig of 129 takes 0.15
   s); and an order of one and a batch of three matrices, which draws reach a
   few times a run. *)
let examples =
  let c dt n seed =
    {
      dt;
      batch = [||];
      n;
      r = 3;
      triangle = Lower;
      adjoint = false;
      unit_diagonal = false;
      seed;
      how = Contiguous;
      scale = Plain;
    }
  in
  List.mapi (fun i dt -> c dt 65 (11 + i)) dtypes
  @ [
      c D.(Any Complex64) 129 6;
      c D.(Any Float32) 1 3;
      { (c D.(Any Complex128) 5 4) with batch = [| 3 |] };
    ]

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
  classes c;
  if c.n > 0 then cholesky_fails b c (cholesky_operand c) (c.seed mod c.n)

(* Pivots failing past the first block of 64 columns: column 64, the second
   block's first, 100 within it, and 128, a third block's only column. *)
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
  classes c;
  if c.n > 0 then
    let i = c.seed mod c.n in
    let j = c.seed / c.n mod (i + 1) in
    let bad =
      [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed mod 3)
    in
    let spoil v =
      let v = Array.copy v in
      v.((i * c.n) + j) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(real_of c.dt) ~rows:c.n ~cols:1 ~spoil
          [| eigh_operand c |])
      [ Eigh true; Eigh false ]

(* NaN or an infinity anywhere; unless [vectors], of the values alone. *)
let law_eig_fails ?(vectors = true) (b : Support.backend) c =
  let c = one c in
  classes c;
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
          ~spoil
          [| eig_operand c |])
      (if vectors then [ Eig true; Eig false ] else [ Eig false ])

(* NaN or an infinity anywhere. *)
let law_svd_fails (b : Support.backend) c =
  let c = one c in
  classes c;
  let m, n, x = svd_operand c in
  cover "wide" (m < n);
  cover "tall" (m > n);
  if m > 0 && n > 0 then
    let p = c.seed mod (m * n) in
    let bad =
      [| Float.nan; Float.infinity; Float.neg_infinity |].(c.seed / 7 mod 3)
    in
    let spoil v =
      let v = Array.copy v in
      v.(p) <- { Complex.re = bad; im = 0. };
      v
    in
    List.iter
      (fun r ->
        check_alone b r ~dt:c.dt ~result:(real_of c.dt) ~rows:(min m n) ~cols:1
          ~spoil [| x |])
      [ Svd (Some (factors c)); Svd None ]

let law_solve_fails (b : Support.backend) c =
  let c = { (one c) with unit_diagonal = false; r = max 1 c.r } in
  classes c;
  cover "Upper" (c.triangle = Upper);
  cover "adjoint" c.adjoint;
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
      else if i = j && diagonal_im && is_complex c.dt then
        { x with im = uniform r }
      else x)
    v

(* What a routine does not read changes no bit of its results: [r]'s results on
   [x] and on [y] have one set of bits. *)
let same_results (b : Support.backend) r x y =
  match (run b r x, run b r y) with
  | Some y, Some z ->
      Array.iteri
        (fun k y ->
          equal
            ~msg:(Printf.sprintf "result %d" k)
            same_bits (bits y)
            (bits z.(k)))
        y
  | None, None -> collect "a dtype its library declines"
  | _ -> fail "declined one call of two"

(* Cholesky reads the lower triangle and the diagonal's real parts alone,
   whatever its triangle (nx_kernel.mli). *)
let law_cholesky_reads (b : Support.backend) c =
  classes c;
  cover "Upper" (c.triangle = Upper);
  cover "a batch of several" (count c > 1);
  let x = cholesky_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  same_results b (Cholesky c.triangle) [| x |] [| of_values c.dt (square c) v |]

let law_eigh_reads (b : Support.backend) c =
  classes c;
  cover "a batch of several" (count c > 1);
  let x = eigh_operand c in
  let v = garbage c (values x) ~unread:(fun i j -> j > i) ~diagonal_im:true in
  same_results b (Eigh true) [| x |] [| of_values c.dt (square c) v |]

let law_solve_reads (b : Support.backend) c =
  classes c;
  cover "Upper" (c.triangle = Upper);
  cover "adjoint" c.adjoint;
  cover "a unit diagonal" c.unit_diagonal;
  cover "a batch of several" (count c > 1);
  let ops = solve_operands c in
  let unread i j =
    (if c.triangle = Lower then j > i else j < i) || (c.unit_diagonal && i = j)
  in
  let v = garbage c (values ops.(0)) ~unread ~diagonal_im:false in
  same_results b (solve c) ops [| of_values c.dt (square c) v; ops.(1) |]

(* A dst over a's bytes, one element on: the door refuses the call as
   Overlapping (nx_kernel.mli), before any write. *)
let test_overlapping (b : Support.backend) () =
  let module K = (val b.kernels) in
  let x = A.create b.device D.Float32 [| 17 |] in
  let from start =
    let s =
      Option.get (A.move (M.Slice [| { M.start; count = 16; step = 1 } |]) x)
    in
    Option.get (A.move (M.Reshape [| 4; 4 |]) s)
  in
  equal answer A.Overlapping (K.cholesky Lower ~dst:(from 1) (from 0))

(* Dtypes *)

(* A row's call into destinations of the dtypes and shapes [dsts]: a call that
   does not [fit] is not Done and writes nothing outside them; one that fits
   answers no Wrong_dtype. *)
let dtype_case (b : Support.backend) ~fit dsts call =
  if not fit then Misfit.check ~msg:"" ~what:"a dtype" b dsts call
  else
    let fresh (D.Any d, s) = A.Any (A.create b.device d (Iarray.of_array s)) in
    let got = call { Misfit.g = Fun.id } (Array.map fresh dsts) in
    if got = A.Wrong_dtype then fail "answered Wrong_dtype"

(* Each routine at each dtype of [a]: cholesky, lu, qr and solve_triangular take
   float32, float64, complex64 and complex128; eig's results are complex, svd's
   and eigh's values floats, each of [a]'s precision. Each row is its routine,
   [a]'s dtype, the results' and whether it fits. *)
let dtype_rows =
  let takes (D.Any dt) =
    match dt with
    | D.Float32 | D.Float64 | D.Complex64 | D.Complex128 -> true
    | _ -> false
  in
  let wide (D.Any dt) =
    match dt with D.Float64 | D.Complex128 -> true | _ -> false
  in
  List.concat_map
    (fun (D.Any dt as d) ->
      let sq = [| 2; 2 |] and n = [| 2 |] in
      let row name ?into fit call =
        let into =
          match into with None -> "" | Some (D.Any r) -> " into " ^ D.name r
        in
        (Printf.sprintf "%s of %s%s" name (D.name dt) into, fit, call)
      in
      let a b = A.create b.Support.device dt (Iarray.of_array sq) in
      let x ds k = A.expect dt ds.(k) in
      let m = (d, sq) in
      let eig (type c) (c : (Complex.t, c) D.t) =
        let fit = takes d && wide d = (D.Any c = D.Any D.Complex128) in
        let z ds k = A.expect c ds.(k) in
        [
          row "eig" ~into:(D.Any c) fit (fun b ->
              let (module K : Nx_kernel.S) = b.Support.kernels in
              dtype_case b ~fit
                [| (D.Any c, n); (D.Any c, sq) |]
                (fun g ds -> K.eig ~w:(z ds 0) ~v:(z ds 1) (g.Misfit.g (a b))));
          row "eig's values" ~into:(D.Any c) fit (fun b ->
              let (module K : Nx_kernel.S) = b.Support.kernels in
              dtype_case b ~fit
                [| (D.Any c, n) |]
                (fun g ds -> K.eig ~w:(z ds 0) (g.Misfit.g (a b))));
        ]
      in
      let real (type r) (f : (float, r) D.t) =
        let fit =
          takes d && wide d = (D.Any f = D.Any D.Float64) && D.bits f >= 32
        in
        let y ds = A.expect f ds.(0) in
        let into = D.Any f in
        let kernels b = b.Support.kernels in
        [
          row "svd" ~into fit (fun b ->
              let (module K : Nx_kernel.S) = kernels b in
              dtype_case b ~fit
                [| (into, n); m; m |]
                (fun g ds ->
                  K.svd ~s:(y ds)
                    ~vectors:(~u:(x ds 1), ~vh:(x ds 2))
                    (g.Misfit.g (a b))));
          row "svd's values" ~into fit (fun b ->
              let (module K : Nx_kernel.S) = kernels b in
              dtype_case b ~fit
                [| (into, n) |]
                (fun g ds -> K.svd ~s:(y ds) (g.Misfit.g (a b))));
          row "eigh" ~into fit (fun b ->
              let (module K : Nx_kernel.S) = kernels b in
              dtype_case b ~fit
                [| (into, n); m |]
                (fun g ds -> K.eigh ~w:(y ds) ~v:(x ds 1) (g.Misfit.g (a b))));
          row "eigh's values" ~into fit (fun b ->
              let (module K : Nx_kernel.S) = kernels b in
              dtype_case b ~fit
                [| (into, n) |]
                (fun g ds -> K.eigh ~w:(y ds) (g.Misfit.g (a b))));
        ]
      in
      let i64 = D.Any D.Int64 in
      let fit = takes d in
      let routine name dsts call =
        row name fit (fun b ->
            let (module K : Nx_kernel.S) = b.Support.kernels in
            dtype_case b ~fit dsts (fun g ds ->
                call (module K : Nx_kernel.S) b g ds))
      in
      eig D.Complex64 @ eig D.Complex128 @ real D.Float16 @ real D.Float32
      @ real D.Float64
      @ [
          routine "cholesky" [| m |] (fun (module K) b g ds ->
              K.cholesky Lower ~dst:(x ds 0) (g.Misfit.g (a b)));
          routine "lu"
            [| m; (i64, n); (i64, n) |]
            (fun (module K) b g ds ->
              K.lu ~lu:(x ds 0)
                ~pivots:(A.expect D.Int64 ds.(1))
                ~perm:(A.expect D.Int64 ds.(2))
                (g.Misfit.g (a b)));
          routine "qr" [| m; m |] (fun (module K) b g ds ->
              K.qr ~q:(x ds 0) ~r:(x ds 1) (g.Misfit.g (a b)));
          routine "solve_triangular" [| m |] (fun (module K) b g ds ->
              K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
                ~dst:(x ds 0)
                (g.Misfit.g (a b))
                (g.Misfit.g (a b)));
        ])
    D.all

(* The suite *)

let laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group d.label
    [
      group "cholesky"
        [
          prop ~examples "a matrix that is not definite is NaN alone" small
            (run law_cholesky_fails);
          test "a pivot failing past the first block is NaN alone"
            (run (fun b () -> test_cholesky_fails_late b ()));
          prop ~examples
            "reads the lower triangle and the diagonal's real parts, whatever \
             its triangle"
            small (run law_cholesky_reads);
          test "a dst over a's bytes is refused as Overlapping"
            (run (fun b () -> test_overlapping b ()));
        ];
      group "lu"
        [
          cases ~id:pivot_id
            "the pivot is the first row of greatest magnitude, a NaN never \
             greater and nothing greater than a NaN"
            pivot_rules (run test_pivot);
          cases
            ~id:(Format.asprintf "%a" pp_dtype)
            "a zero pivot stays in U and its column of L is not divided" dtypes
            (run test_zero_pivot);
        ];
      group "svd"
        [
          prop ~examples "a matrix with NaN or an infinity is NaN alone" small
            (run law_svd_fails);
        ];
      group "eigh"
        [
          prop ~examples "a matrix with NaN or an infinity is NaN alone" small
            (run law_eigh_fails);
          prop ~examples
            "reads the lower triangle and the diagonal's real parts" small
            (run law_eigh_reads);
        ];
      group "eig"
        [
          prop ~examples "a matrix with NaN or an infinity is NaN alone" small
            (run (fun b c -> law_eig_fails b c));
        ];
      test
        "svd and eig of a matrix of finite elements whose modulus overflows \
         give numbers"
        (run (fun b () -> test_overflowing_modulus b ()));
      group "solve_triangular"
        [
          prop ~examples "a zero on a read diagonal is NaN alone" small
            (run law_solve_fails);
          prop ~examples "reads its triangle, and its diagonal unless unit"
            small (run law_solve_reads);
        ];
      cases
        ~id:(fun (id, _, _) -> id)
        "routines take float32, float64, complex64 and complex128; a call at \
         another dtype is not Done and writes nothing outside its destinations"
        dtype_rows
        (run (fun b (_, _, call) -> call b));
    ]

(* Memory of calls in a loop

   nx_cuda.mli's memory of the matrix routines, held by calls in a loop: float32
   elements of 4 bytes. *)

let wait (A.Any x) = Rig.Buffer.wait (A.buffer x) Read_write

(* A solve whose [a]'s batch axes do not form one run of strides copies [a]:
   [batch n²] elements. *)
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
  Kernels_cases.loop_held b
    ~bound:(6 * n * n * 4)
    (fun () ->
      equal answer A.Done
        (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false ~dst a rhs);
      wait (A.Any dst))

(* A qr of 64 columns copies [a], [m n] elements, and takes [m b + 2 b max(n, c)
   + b² ⌈k / b⌉] more, [b = min k 128]; each piece from aligned bytes. *)
let test_loop_qr (b : Support.backend) () =
  let module K = (val b.kernels) in
  let m = 256 and n = 64 and d = b.device in
  let a = A.create d D.Float32 [| m; n |] in
  let q = A.create d D.Float32 [| m; n |]
  and r = A.create d D.Float32 [| n; n |] in
  let k = n and c = n in
  let bk = Int.min k 128 in
  let elements =
    (m * n) + (m * bk) + (2 * bk * Int.max n c) + (bk * bk * ((k + bk - 1) / bk))
  in
  Kernels_cases.loop_held b
    ~bound:((4 * elements) + 1024)
    (fun () ->
      equal answer A.Done (K.qr ~q ~r a);
      wait (A.Any r))

(* An eigh with [v] of order [n] takes [2n² + 97 n + 2048] elements, [3 n² + 78
   n - 2 l (32 - l)] reals and [24 n + 32] bytes, [l] its last leaf's order, and
   at most 512 bytes of alignment a matrix and 2 KiB a call; and [n²] elements
   more where [a] is transposed, which it copies. *)
let test_loop_eigh ~n ~transposed (b : Support.backend) () =
  let module K = (val b.kernels) in
  let d = b.device in
  let base = A.create d D.Float32 [| n; n |] in
  let a =
    if transposed then
      A.v D.Float32 (A.Layout.v ~strides:[| 1; n |] [| n; n |]) (A.buffer base)
    else base
  in
  let w = A.create d D.Float32 [| n |]
  and v = A.create d D.Float32 [| n; n |] in
  let l = n - (32 * (((n + 31) / 32) - 1)) in
  let copy = if transposed then n * n else 0 in
  let elements = (2 * n * n) + (97 * n) + 2048 + copy in
  let reals = (3 * n * n) + (78 * n) - (2 * l * (32 - l)) in
  let bytes = (4 * (elements + reals)) + (24 * n) + 32 + 512 + 2048 in
  Kernels_cases.loop_held b ~bound:bytes (fun () ->
      equal answer A.Done (K.eigh ~w ~v a);
      wait (A.Any v))

let cuda (d : Support.declared) =
  let test name f =
    test name (fun () ->
        let b = Support.opened d in
        b.around (f b))
  in
  group ("nx.cuda " ^ d.label)
    [
      test "solves in a loop hold one solve's copy" test_loop_solve;
      test "qrs in a loop hold one qr's memory" test_loop_qr;
      test "eighs in a loop hold one eigh's memory"
        (test_loop_eigh ~n:64 ~transposed:false);
      test "eighs of a copied operand in a loop hold one eigh's memory"
        (test_loop_eigh ~n:80 ~transposed:true);
    ]

(* nx.metal declines every routine at float64 and complex128, and eig's vectors
   at every dtype (nx_metal.mli). *)
let test_metal_declines (b : Support.backend) () =
  let module K = (val b.kernels) in
  let sq = Iarray.of_array [| 2; 2 |] and n = Iarray.of_array [| 2 |] in
  let each (type a s rs cs) (dt : (a, s) D.t) (re : (float, rs) D.t)
      (cx : (Complex.t, cs) D.t) =
    let declines name got =
      equal ~msg:(Format.asprintf "%s of %a" name D.pp dt) answer A.Declined got
    in
    let m () = A.create b.device dt sq in
    let pos () = A.create b.device D.Int64 n in
    let a = m () and w = A.create b.device re n in
    if not (b.arith (D.Any dt)) then begin
      declines "cholesky" (K.cholesky Lower ~dst:(m ()) a);
      declines "lu" (K.lu ~lu:(m ()) ~pivots:(pos ()) ~perm:(pos ()) a);
      declines "solve_triangular"
        (K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
           ~dst:(m ()) a (m ()));
      declines "qr" (K.qr ~q:(m ()) ~r:(m ()) a);
      declines "eigh" (K.eigh ~w ~v:(m ()) a);
      declines "svd" (K.svd ~s:w ~vectors:(~u:(m ()), ~vh:(m ())) a)
    end;
    declines "eig with vectors"
      (K.eig ~w:(A.create b.device cx n) ~v:(A.create b.device cx sq) a)
  in
  each D.Float32 D.Float32 D.Complex64;
  each D.Complex64 D.Float32 D.Complex64;
  each D.Float64 D.Float64 D.Complex128;
  each D.Complex128 D.Float64 D.Complex128

(* nx.metal computes the calls nx_metal.mli states: cholesky, lu and
   solve_triangular of any order; eigh of any order, svd by Jacobi up to 32 rows
   and columns, svd's values and eig's past them. *)
let metal (d : Support.declared) =
  let on held f c =
    assume (held c);
    let b = Support.opened d in
    b.around (fun () -> f b c)
  in
  let any = on (fun _ -> true) in
  let svd_jacobi =
    on (fun c ->
        let m, n, _ = svd_operand (one c) in
        m <= 32 && n <= 32)
  in
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  group ("nx.metal " ^ d.label)
    [
      test
        "every routine declines at float64 and complex128, eig's vectors at \
         every dtype"
        (run (fun b () -> test_metal_declines b ()));
      prop "a matrix eigh fails on is NaN alone" case (any law_eigh_fails);
      prop "a matrix svd fails on is NaN alone" case (svd_jacobi law_svd_fails);
      prop "a matrix eig's values fail on is NaN alone" case
        (any (fun b c -> law_eig_fails ~vectors:false b c));
      prop "eigh reads the lower triangle and the diagonal's real parts" case
        (any law_eigh_reads);
      prop "a matrix that is not definite is NaN alone" case
        (any law_cholesky_fails);
      prop "a singular triangle is NaN alone" case (any law_solve_fails);
      prop "Cholesky reads the lower triangle and the diagonal's real parts"
        case (any law_cholesky_reads);
      prop "a solve reads its triangle, and its diagonal unless unit" case
        (any law_solve_reads);
      cases ~id:pivot_id "LU's pivot is the scan's" pivot_rules (run test_pivot);
      cases
        ~id:(fun (id, _, _) -> id)
        "routines take floats and complex numbers alone; a call at another \
         dtype is not Done and writes nothing outside its destinations"
        dtype_rows
        (run (fun b (_, _, call) -> call b));
    ]

let () =
  let declared = List.filter_map Fun.id in
  exit
    (Windtrap.run "nx_kernel.linalg"
       (List.map laws
          (List.map Support.declare Support.cpus
          @ declared [ Support.declared_cuda; Support.declared_amd ])
       @ List.map cuda (declared [ Support.declared_cuda ])
       @ List.map metal (declared [ Support.declared_metal ])))
