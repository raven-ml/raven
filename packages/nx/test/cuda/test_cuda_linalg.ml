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
   algorithm's; the same bits under every layout and in every batch.
   Through rig.nv, CUDA's bits. *)

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

(* A case of [dt], order [n] and [r] columns, one matrix. *)
let fixed ?(seed = 7) ?(triangle = Nx_kernel.Lower) ?(adjoint = false) dt n r =
  {
    dt;
    batch = [||];
    n;
    r;
    triangle;
    adjoint;
    unit_diagonal = false;
    seed;
    how = Contiguous;
    scale = Plain;
  }

(* QR of matrices scaled below the least normal number, past it, and near
   the largest, through its one launch, its blocks and its meetings: every
   result a number and within its bound, residuals taken on the values
   scaled back by 2^e. *)
let test_qr_scaled (b : Support.backend) () =
  let f32 = [ (1e20, -67); (1e-20, 66); (1e-40, 133) ]
  and f64 = [ (1e300, -997); (1e-300, 997); (1e-310, 1030) ] in
  List.iter
    (fun (dt, scales) ->
      List.iter
        (fun (batch, m, n) ->
          let _, x = lu_operand { (fixed ~seed:10 dt m n) with batch } in
          let count = Array.fold_left ( * ) 1 batch in
          List.iter
            (fun (scale, e) ->
              let v =
                Array.map
                  (fun z -> Complex.mul z { re = scale; im = 0. })
                  (values x)
              in
              let shape = Array.append batch [| m; n |] in
              match run b (Qr Reduced) [| of_values dt shape v |] with
              | None -> ()
              | Some y ->
                  let msg =
                    Format.asprintf "%a %dx%d scaled by %g" pp_dtype dt m n
                      scale
                  in
                  Array.iteri
                    (fun i y ->
                      let w = values y in
                      equal
                        ~msg:(Printf.sprintf "%s: numbers in result %d" msg i)
                        int
                        ((if is_complex dt then 2 else 1) * Array.length w)
                        (numbers dt w))
                    y;
                  let k = min m n in
                  check_qr dt ~e ~count ~m ~n ~qc:k (up e v)
                    (values y.(0))
                    (up e (values y.(1))))
            scales)
        [ ([| 4 |], 8, 8); ([||], 40, 40); ([||], 300, 20); ([||], 300, 40) ])
    D.
      [
        (Any Float32, f32);
        (Any Complex64, f32);
        (Any Float64, f64);
        (Any Complex128, f64);
      ]

(* The unblocked algorithm on the host *)

(* Arithmetic of a routine's format, on doubles holding its values: a
   float32 sum, product, quotient and square root are the double's rounded
   once more, which gives the float32 result (the double holds twice its
   bits and two more); a fused multiply-add rounds its exact value once. *)
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* fma in float32: the exact a b + c is s + e (a b exact in a double,
   TwoSum); s rounded to float32 is the result unless s is the midpoint of
   two float32s, where e breaks the tie. *)
let fmaf a b c =
  let p = a *. b in
  let s = p +. c in
  let z = s -. p in
  let e = p -. (s -. z) +. (c -. z) in
  let r = round32 s in
  if e = 0. || r = s then r
  else
    let step x up =
      let bits = Int32.bits_of_float x in
      let bits =
        if (x > 0. || (x = 0. && up)) = up then Int32.add bits 1l
        else Int32.sub bits 1l
      in
      Int32.float_of_bits bits
    in
    let other = step r (s > r) in
    if s -. r = other -. s then if e > 0. = (other > r) then other else r else r

type arith = {
  fma : float -> float -> float -> float;
  rnd : float -> float;
  cx : bool;
}

let arith (D.Any dt as d) =
  let cx = is_complex d in
  match dt with
  | D.Float32 | D.Complex64 -> { fma = fmaf; rnd = round32; cx }
  | _ -> { fma = Float.fma; rnd = Fun.id; cx }

(* c + a b: one fused multiply-add on reals, two a part on complex
   numbers, the terms in nx_kernel.mli's order. *)
let mac ar (c : Complex.t) (a : Complex.t) (b : Complex.t) =
  if not ar.cx then { c with re = ar.fma a.re b.re c.re }
  else
    let re = ar.fma (-.a.im) b.im (ar.fma a.re b.re c.re) in
    let im = ar.fma a.im b.re (ar.fma a.re b.im c.im) in
    { Complex.re; im }

let nmac ar c (a : Complex.t) b = mac ar c (Complex.neg a) b

(* Each x / d: by d's real part where its imaginary part is 0, else
   nx.cpu's Fdiv, which nx.cuda's shares (nx_kinds.h), in one call. *)
let divd dt ar (xs : Complex.t array) (d : Complex.t) =
  if d.im = 0. then
    Array.map
      (fun (x : Complex.t) ->
        { Complex.re = ar.rnd (x.re /. d.re); im = ar.rnd (x.im /. d.re) })
      xs
  else if Array.length xs = 0 then [||]
  else
    let n = Array.length xs in
    let (A.Any xa) = of_values dt [| n |] xs in
    let da = A.expect (A.dtype xa) (of_values dt [| n |] (Array.make n d)) in
    let dst = A.create Rig.host (A.dtype xa) (Iarray.of_array [| n |]) in
    match
      Support.map2 (module Nx_cpu) (Nx_kernel.Prog.Binary Fdiv) ~dst xa da
    with
    | A.Done -> values (A.Any dst)
    | _ -> failwith "Fdiv"

let is_zero (z : Complex.t) = z.re = 0. && z.im = 0.

(* LU's factors, pivots and perm of the m x n matrix [a], the unblocked
   right-looking algorithm with the contract's search. *)
let lu_ref dt ~m ~n (a : Complex.t array) =
  let ar = arith dt in
  let a = Array.copy a and k = min m n in
  let mag (z : Complex.t) = ar.rnd (Float.abs z.re +. Float.abs z.im) in
  let piv = Array.make k 0 and perm = Array.init m Fun.id in
  for j = 0 to k - 1 do
    let p = ref j and best = ref (mag a.((j * n) + j)) in
    for i = j + 1 to m - 1 do
      let g = mag a.((i * n) + j) in
      if g > !best then begin
        best := g;
        p := i
      end
    done;
    piv.(j) <- !p;
    if !p <> j then begin
      for c = 0 to n - 1 do
        let t = a.((j * n) + c) in
        a.((j * n) + c) <- a.((!p * n) + c);
        a.((!p * n) + c) <- t
      done;
      let t = perm.(j) in
      perm.(j) <- perm.(!p);
      perm.(!p) <- t
    end;
    let d = a.((j * n) + j) in
    let below = Array.init (m - j - 1) (fun t -> a.(((j + 1 + t) * n) + j)) in
    let ls = if is_zero d then below else divd dt ar below d in
    for i = j + 1 to m - 1 do
      let l = ls.(i - j - 1) in
      a.((i * n) + j) <- l;
      for c = j + 1 to n - 1 do
        a.((i * n) + c) <- nmac ar a.((i * n) + c) l a.((j * n) + c)
      done
    done
  done;
  (a, piv, perm)

(* Cholesky's lower factor of the n x n [a], or None where a pivot is not
   positive. *)
let cholesky_ref dt ~n (a : Complex.t array) =
  let ar = arith dt in
  let a = Array.copy a in
  let ok = ref true in
  for j = 0 to n - 1 do
    let d = a.((j * n) + j).re in
    if not (d > 0.) then ok := false;
    let s = ar.rnd (Float.sqrt d) in
    let sd = { Complex.re = s; im = 0. } in
    let col =
      divd dt ar
        (Array.init (n - j - 1) (fun t -> a.(((j + 1 + t) * n) + j)))
        sd
    in
    Array.iteri (fun t l -> a.(((j + 1 + t) * n) + j) <- l) col;
    a.((j * n) + j) <- sd;
    for i = j + 1 to n - 1 do
      for c = j + 1 to i do
        a.((i * n) + c) <-
          nmac ar a.((i * n) + c) (Complex.conj a.((c * n) + j)) a.((i * n) + j)
      done
    done
  done;
  if not !ok then None
  else
    Some (Array.mapi (fun k z -> if k mod n > k / n then Complex.zero else z) a)

(* x with op(a) x = b, n x n [a] and n x r [b], substituting from the first
   row of a lower op(a), from the last of an upper one. *)
let solve_ref dt ~n ~r ~upper ~adjoint ~unit (a : Complex.t array)
    (b : Complex.t array) =
  let ar = arith dt in
  let op i q =
    if adjoint then Complex.conj a.((q * n) + i) else a.((i * n) + q)
  in
  let x = Array.copy b in
  let zero = ref false in
  for t = 0 to n - 1 do
    let i = if upper then n - 1 - t else t in
    let row =
      Array.init r (fun j ->
          let v = ref x.((i * r) + j) in
          for u = 0 to t - 1 do
            let q = if upper then n - 1 - u else u in
            v := nmac ar !v (op i q) x.((q * r) + j)
          done;
          !v)
    in
    let row =
      if unit then row
      else begin
        if is_zero (op i i) then zero := true;
        divd dt ar row (op i i)
      end
    in
    Array.blit row 0 x (i * r) r
  done;
  if !zero then
    Array.map (fun _ -> { Complex.re = Float.nan; im = Float.nan }) x
  else x

(* Bits as same_bits compares them, any NaN as one. *)
let nan_bits a =
  Array.map (fun f -> if Float.is_nan f then Float.nan else f) (bits a)

let same msg want got = equal ~msg same_bits (nan_bits want) (nan_bits got)

let lu_unblocked (b : Support.backend) c =
  let dt = c.dt in
  let cols, x = lu_operand c in
  match run b Lu [| x |] with
  | None -> ()
  | Some y ->
      let m = c.n in
      let lu, piv, perm = lu_ref dt ~m ~n:cols (values x) in
      same "lu" (of_values dt [| m; cols |] lu) y.(0);
      equal ~msg:"lu's pivots" (array int) piv (int64s y.(1));
      equal ~msg:"lu's perm" (array int) perm (int64s y.(2))

let law_unblocked (b : Support.backend) c =
  let c = one c in
  let dt = c.dt in
  lu_unblocked b c;
  (let x = cholesky_operand c in
   match run b (Cholesky Lower) [| x |] with
   | None -> ()
   | Some y -> (
       match cholesky_ref dt ~n:c.n (values x) with
       | Some l -> same "cholesky" (of_values dt (square c) l) y.(0)
       | None -> equal ~msg:"cholesky fails" int 0 (numbers dt (values y.(0)))));
  let ops = solve_operands c in
  match run b (solve c) ops with
  | None -> ()
  | Some y ->
      let upper = c.triangle = Upper <> c.adjoint in
      let x =
        solve_ref dt ~n:c.n ~r:c.r ~upper ~adjoint:c.adjoint
          ~unit:c.unit_diagonal
          (values ops.(0))
          (values ops.(1))
      in
      same "solve" (of_values dt [| c.n; c.r |] x) y.(0)

(* Matrix [k] of a batch of one batch axis. *)
let alone k (A.Any x) =
  let l = A.layout x in
  let shape = Iarray.to_array (Nx_array.Layout.shape l) in
  let one = Array.sub shape 1 (Array.length shape - 1) in
  let v = values (A.Any x) in
  let per = Array.fold_left ( * ) 1 one in
  of_values (D.Any (A.dtype x)) one (Array.sub v (k * per) per)

(* The first matrix of [x], of one batch axis, as a view. *)
let first (A.Any x) =
  let s = Iarray.to_array (Nx_array.Layout.shape (A.layout x)) in
  let whole e = { Nx_array.Move.start = 0; count = e; step = 1 } in
  let one =
    Array.mapi
      (fun i e -> if i = 0 then { (whole e) with count = 1 } else whole e)
      s
  in
  A.Any (Option.get (A.move (Nx_array.Move.Slice (Iarray.of_array one)) x))

(* QR of a matrix of [m] rows and [n] columns alone, its rows split over
   several blocks a step, has the bits it has in a batch of 50, whose
   matrices take a block each: 300 rows over 2 blocks in two launches a
   step, 4500 over 18, and 32768 over 128, the most a panel's rows split
   into. The batch is the matrix broadcast. *)
let qr_blocks (b : Support.backend) dt (m, n) =
  let _, x = lu_operand { (fixed ~seed:10 dt m n) with batch = [| 1 |] } in
  let (A.Any x) = x in
  let x = Support.on b x in
  let batch =
    Option.get
      (A.move (Nx_array.Move.Broadcast (Iarray.of_array [| 50; m; n |])) x)
  in
  match
    (run b (Qr Reduced) [| A.Any batch |], run b (Qr Reduced) [| A.Any x |])
  with
  | Some y, Some z ->
      Array.iteri
        (fun i z ->
          equal
            ~msg:(Printf.sprintf "%dx%d: result %d" m n i)
            same_bits
            (bits (first y.(i)))
            (bits z))
        z
  | _ -> ()

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

(* A matrix in a batch of others has the bits it has alone. A case makes
   all its calls before it reads any result, so that the device runs them
   without a round trip between. *)
let law_batch (b : Support.backend) c =
  let c = { c with batch = [| 3 |] } in
  let check name r ops =
    match run b r ops with
    | None -> fun () -> ()
    | Some y ->
        let alone_runs =
          List.init 3 (fun k -> run b r (Array.map (alone k) ops))
        in
        fun () ->
          List.iteri
            (fun k z ->
              match z with
              | None -> ()
              | Some z ->
                  Array.iteri
                    (fun i z ->
                      (* LU's positions are int64s, laws of their own. *)
                      if i = 0 || name <> "lu" then
                        equal
                          ~msg:
                            (Printf.sprintf "%s, matrix %d, result %d" name k i)
                          same_bits
                          (bits (alone k y.(i)))
                          (bits z))
                    z)
            alone_runs
  in
  let _, _, x = svd_operand c in
  let checks =
    [
      check "cholesky" (Cholesky Lower) [| cholesky_operand c |];
      check "lu" Lu [| snd (lu_operand c) |];
      check "qr" (Qr (factors c)) [| snd (lu_operand c) |];
      check "solve" (solve c) (solve_operands c);
      check "eigh" (Eigh true) [| eigh_operand c |];
      check "eigh's values" (Eigh false) [| eigh_operand c |];
      check "svd" (Svd (Some (factors c))) [| x |];
      check "svd's values" (Svd None) [| x |];
      check "eig" (Eig true) [| eig_operand c |];
      check "eig's values" (Eig false) [| eig_operand c |];
    ]
  in
  List.iter (fun check -> check ()) checks

let laws (b : Support.backend) =
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt -> f b dt)
  in
  group b.name
    [
      group "each result meets its relation within its bound"
        [
          prop "Cholesky's factor within γ(n + 1)" case (law_cholesky bounds b);
          prop "the Upper factor is the Lower one's conjugate transpose" case
            (law_upper b);
          prop "a triangular solve within γ(n + 2)" case (law_solve bounds b);
          prop "LU's factors within γ(k + 2)" case (law_lu bounds b);
          test "LU of a pivot whose reciprocal is subnormal"
            (test_huge_pivot bounds b);
          test "QR of matrices scaled to the ends of the exponent range"
            (test_qr_scaled b);
          prop "QR's factors within γ(k(2m + 10))" case (law_qr b);
          cases "past every blocking" test_large;
          prop "eigh's results within γ(16n)" case (law_eigh bounds b);
          prop "svd's results within γ(16 max (m, n) + 100)" case
            (law_svd bounds b);
          prop
            "svd's results on parts at the least subnormal, below the least \
             normal and near 1"
            spread (law_svd_spread bounds b);
          test
            "svd's results on a row whose rest is below the least normal \
             number over u"
            (test_svd_tiny_rows b);
          test "svd's results at every small shape that shares a group"
            (test_svd_shapes b);
          test "svd's results on a row whose rest is the least subnormal"
            (test_svd_spread_row bounds b);
          cases
            "svd's results on matrices of rank 10 at 2^400 over rows of the \
             least subnormal"
            (test_svd_deficient bounds);
          cases "svd's results on 16384 matrices of order 2"
            (test_svd_pairs bounds);
          prop
            "eig's results within γ(16n), its vectors positive at their largest"
            case (law_eig bounds b);
          test "eig's results on a graded cycle" (test_eig_graded_cycle b);
          test "eig's results on a subnormal matrix with a complex pair"
            (test_eig_subnormal_pair b);
          test
            "eigh's, qr's and svd's results on matrices scaled below the least \
             normal number and near the largest"
            (test_scaled bounds b);
        ];
      group "each spectrum is within its bound of the exact one"
        [
          prop "eigh's eigenvalues within γ(16n) ‖a‖₂" case
            (law_eigh_spectrum b);
          test "eigh's eigenvalues at large orders" (test_eigh_large b);
          test
            "eigh's eigenvalues of a secular equation with a root beside its \
             pole"
            (test_eigh_secular b);
          prop "svd's values within γ(16 max (m, n) + 100) ‖a‖₂" case
            (law_svd_spectrum b);
          test "svd's values at large orders" (test_svd_large b);
          prop "eig's eigenvalues of a normal matrix within γ(16n) ‖a‖_F" case
            (law_eig_spectrum b);
          test "eig's eigenvalues of a graded matrix, balanced"
            (test_eig_balanced b);
          test
            "eig's eigenvalues of a matrix spanning the exponent range, \
             balanced before it is scaled"
            (test_eig_extremes b);
          test "eig's eigenvalues of Wilkinson's matrices are eigh's"
            (test_wilkinson b);
        ];
      group "bits are a function of the values alone"
        [
          prop "LU's, Cholesky's and a solve's are the unblocked algorithm's"
            case (law_unblocked b);
          prop "whatever the layout" case (law_layouts b);
          (* Each case makes 40 calls. *)
          prop ~count:25 "whatever the batch" case (law_batch b);
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
  cover "laid out with gaps" (c.how = Padded && c.n > 1);
  let r, ops = f c in
  ops.(0) <- laid_out c.how ops.(0);
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

let agreement ~first (b : Support.backend) =
  let cases name f =
    cases name ~id:(Format.asprintf "%a" pp_dtype) dtypes (fun dt -> f dt)
  in
  group b.name
    [
      group
        (Printf.sprintf "bits are %s's" first.Support.name)
        [
          (* One routine a case, so that each is drawn often. *)
          prop ~count:300 "on the suite's cases" (Gen.pair case routine)
            (law_agrees ~first b);
          cases "past every blocking" (fun dt -> agree ~first b (large dt));
          test "at every small shape that shares a group"
            (test_shared_agrees ~first b);
        ];
    ]

let () =
  let tests =
    match (Support.gpus (), Support.nv ()) with
    | [], _ -> []
    | first :: _, None -> [ laws first ]
    | first :: _, Some nv -> [ laws first; agreement ~first nv ]
  in
  exit (Windtrap.run "nx.cuda linalg" tests)
