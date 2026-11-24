(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Matrix routines lowered, held to the relations Lower_linalg states: each
   factorisation reconstructs its operand and its unitary factors are
   orthonormal within the stated bounds, its values agree with nx's eager ones
   within both bounds, and its edges are exact: triangles of +0, pivots by the
   rule, NaN for a matrix that fails. A check's own float64 rounding is added
   to the bound it checks. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim

let linalg l = P.eval ~by:"t" (P.Linalg l)

(* Dtypes *)

type real = R : string * (float, 's) D.t * float -> real

let reals =
  [ R ("float32", D.Float32, 0x1p-24); R ("float64", D.Float64, 0x1p-53) ]
let gamma u k = k *. u /. (1. -. (k *. u))
let u64 = 0x1p-53

(* Matrices

   A batch of [b] matrices of [m] rows and [n] columns, in C order. *)

type mat = { b : int; m : int; n : int; x : float array }

let get t k i j = t.x.((((k * t.m) + i) * t.n) + j)
let mat b m n x = { b; m; n; x }

let of_nx (type s) (a : (float, s, _) Nx.t) =
  let s = Nx.shape a in
  let r = Array.length s in
  let b = Array.fold_left ( * ) 1 (Array.sub s 0 (r - 2)) in
  mat b s.(r - 2) s.(r - 1) (Nx.to_array a)

let init b m n f =
  mat b m n
    (Array.init (b * m * n) (fun e -> f (e / (m * n)) (e / n mod m) (e mod n)))

let product ?(abs = false) p q =
  let f = if abs then Float.abs else Fun.id in
  init p.b p.m q.n (fun k i j ->
      let s = ref 0. in
      for l = 0 to p.n - 1 do
        s := !s +. (f (get p k i l) *. f (get q k l j))
      done;
      !s)

let transpose t = init t.b t.n t.m (fun k i j -> get t k j i)

let bits xs = Array.map Int64.bits_of_float xs

(* [bounded ~msg ~than v] asserts that [v] is a number at most [than]. *)
let bounded ~msg ~than v =
  equal ~msg:(msg ^ " is a number") bool false (Float.is_nan v);
  at_most ~msg float_exact ~than v

(* [elementwise ~msg d bound] asserts each element of [d] at most [bound]'s. *)
let elementwise ~msg d bound =
  Array.iteri
    (fun e v ->
      bounded ~msg:(Printf.sprintf "%s, element %d" msg e) ~than:bound.x.(e)
        (Float.abs v))
    d.x

let diff p q = { p with x = Array.mapi (fun e v -> v -. q.x.(e)) p.x }

(* Operands

   Elements uniform in [-1, 1], laid out by rows, by columns, or one matrix
   broadcast along a batch of two. *)

type layout = Rows | Columns | Broadcast

let layouts = [ Rows; Columns; Broadcast ]

let layout_name = function
  | Rows -> "rows"
  | Columns -> "columns"
  | Broadcast -> "broadcast"

let uniform st = Random.State.float st 2. -. 1.

let operand (type s) (dt : (float, s) D.t) layout batch m n f =
  let count = Array.fold_left ( * ) 1 batch * m * n in
  match layout with
  | Rows ->
      Nx.create dt (Array.append batch [| m; n |])
        (Array.init count (fun e -> f (e / (m * n)) (e / n mod m) (e mod n)))
  | Columns ->
      Nx.matrix_transpose
        (Nx.create dt (Array.append batch [| n; m |])
           (Array.init count (fun e ->
                f (e / (m * n)) (e mod m) (e / m mod n))))
  | Broadcast ->
      Nx.broadcast_to [| 2; m; n |]
        (Nx.create dt [| m; n |]
           (Array.init (m * n) (fun e -> f 0 (e / n) (e mod n))))

type case = { dt : real; layout : layout; batch : int array; seed : int }

let pp_case ppf c =
  let (R (name, _, _)) = c.dt in
  Format.fprintf ppf "(%s, %s, [%s], %d)" name (layout_name c.layout)
    (String.concat ";" (Array.to_list (Array.map string_of_int c.batch)))
    c.seed

(* A seed's step that keeps its variant and flags ({!variant}, {!flag}) and its
   factors' choice. *)
let seed_step = 24

(* [cases_of examples] draws a case of the dtype, layout, batch and shape of one
   of [examples], each as likely, of a seed of the example's variant and flags:
   a run lowers and compiles each example's routine once and draws its cases'
   elements anew. *)
let cases_of examples =
  Gen.with_pp
    (fun ppf (c, s) ->
      Format.fprintf ppf "%a %s" pp_case c
        (String.concat "x" (List.map string_of_int s)))
    (Gen.map
       (fun ((c, shape), k) ->
         ({ c with seed = c.seed + (seed_step * k) }, shape))
       (Gen.pair (Gen.of_list examples) (Gen.int_range 1 40_000)))

(* Every dtype, layout and shape, in turn. *)
let examples shapes =
  List.concat
    (List.mapi
       (fun i shape ->
         List.mapi
           (fun j dt ->
             ( {
                 dt;
                 layout = List.nth layouts ((i + j) mod 3);
                 batch = (if (i + j) mod 2 = 0 then [||] else [| 2 |]);
                 seed = (2 * i) + j;
               },
               shape ))
           reals)
       shapes)

let cover_case c shape =
  let (R (name, _, _)) = c.dt in
  cover name true;
  cover (layout_name c.layout) true;
  cover "an empty matrix" (List.mem 0 shape);
  cover "one element" (List.for_all (( = ) 1) shape)

let state c = Random.State.make [| c.seed |]

(* An operand of the form of [c]'s, of [m] rows and [n] columns: its dtype,
   layout and batch, its elements zeros. *)
let shaped (type s) (dt : (float, s) D.t) c m n =
  operand dt c.layout c.batch m n (fun _ _ _ -> 0.)

(* Factors complete or reduced, chosen by [c]'s seed. *)
let factors_of c =
  if c.seed / 3 mod 2 = 1 then Nx_kernel.Complete else Nx_kernel.Reduced

(* A case's variant, of three, and its flags: chosen by its seed, so that the
   examples reach each. *)
let variant c = c.seed mod 3 = 0
let flag c bit = (c.seed lsr bit) land 1 = 1

(* A float of the other triangle, which no routine reads: NaN or large. *)
let garbage st = if Random.State.bool st then Float.nan else 1e30 *. uniform st

(* Cholesky *)

(* [spd st n] is [x xᵀ + (n + 1) I] for [x] uniform: symmetric positive
   definite and well conditioned, a function of its row and column. *)
let spd st n =
  let x = Array.init (n * n) (fun _ -> uniform st) in
  Array.init (n * n) (fun e ->
      let i = e / n and j = e mod n in
      let s = ref (if i = j then float_of_int (n + 1) else 0.) in
      for l = 0 to n - 1 do
        s := !s +. (x.((i * n) + l) *. x.((j * n) + l))
      done;
      !s)

let cholesky_op a triangle () = linalg (Cholesky { triangle; a })

let cholesky_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let n = List.hd shape in
  let a = shaped dt c n n in
  [ Lowered.One (cholesky_op a Lower); Lowered.One (cholesky_op a Upper) ]

let cholesky_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let n = List.hd shape in
  cover_case c shape;
  let st = state c in
  let mats = Array.init 2 (fun _ -> spd st n) in
  let junk = Array.init (2 * n * n) (fun _ -> garbage st) in
  let a =
    operand dt c.layout c.batch n n (fun k i j ->
        if i >= j then mats.(k).((i * n) + j)
        else junk.((((k * n) + i) * n) + j))
  in
  cover "an upper triangle of NaN" (Array.exists Float.is_nan (Nx.to_array a));
  let am = of_nx a in
  let run triangle = Lowered.run (cholesky_op a triangle) in
  let l = { am with x = run Lower } and uf = run Upper in
  equal ~msg:"Upper is Lower's transpose" (array int64)
    (bits (transpose l).x) (bits uf);
  for k = 0 to am.b - 1 do
    for i = 0 to n - 1 do
      greater ~msg:"a positive diagonal" float_exact ~than:0. (get l k i i);
      for j = i + 1 to n - 1 do
        equal ~msg:"+0 above the diagonal" int64 0L
          (Int64.bits_of_float (get l k i j))
      done
    done
  done;
  let sym = init am.b n n (fun k i j -> get am k (max i j) (min i j)) in
  let bound = product ~abs:true l (transpose l) in
  let tol =
    gamma u (float_of_int (n + 1)) +. gamma u64 (float_of_int (n + 1))
  in
  elementwise ~msg:"L Lᵀ = a" (diff (product l (transpose l)) sym)
    { bound with x = Array.map (( *. ) tol) bound.x }

(* A batch of a positive definite matrix and one that is not: the second alone
   is NaN. *)
let cholesky_fails () =
  let a =
    Nx.create D.Float64 [| 2; 2; 2 |] [| 4.; 0.; 2.; 5.; 1.; 0.; 2.; 1. |]
  in
  let l = Lowered.run (fun () -> linalg (Cholesky { triangle = Lower; a })) in
  equal (array float_exact) [| 2.; 0.; 1.; 2. |] (Array.sub l 0 4);
  equal (array bool) [| true; true; true; true |]
    (Array.map Float.is_nan (Array.sub l 4 4))

(* A pivot of exactly +0 or -0 is not positive: a first pivot of each, and a
   second of [1 - 1 = +0] and of [-0 - 0 = -0]. Each such matrix alone is
   NaN. *)
let cholesky_zero_pivot () =
  let a =
    Nx.create D.Float64 [| 5; 2; 2 |]
      [|
        0.; 0.; 0.; 1.; -0.; 0.; 0.; 1.; 1.; 0.; 1.; 1.; 1.; 0.; 0.; -0.; 4.;
        0.; 2.; 5.;
      |]
  in
  let l = Lowered.run (fun () -> linalg (Cholesky { triangle = Lower; a })) in
  equal (array bool) (Array.make 16 true)
    (Array.map Float.is_nan (Array.sub l 0 16));
  equal (array float_exact) [| 2.; 0.; 1.; 2. |] (Array.sub l 16 4)

(* LU *)

(* The graphs of the factors, the pivots and the row order of the LU of [a],
   each alone, as a user who reads only one computes it. *)
let lu_factors a () =
  let x, _, _ = linalg (Lu a) in
  x

let lu_pivots a () =
  let _, p, _ = linalg (Lu a) in
  p

let lu_order a () =
  let _, _, p = linalg (Lu a) in
  p

(* [factors a] is the elements of the lowered LU of [a], its pivots and its row
   order. *)
let factors a =
  ( Lowered.run (lu_factors a),
    Lowered.run (lu_pivots a),
    Lowered.run (lu_order a) )

let lu_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let a = shaped dt c m n in
  Lowered.[ One (lu_factors a); One (lu_pivots a); One (lu_order a) ]

let lu_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let k = Int.min m n in
  cover_case c shape;
  let st = state c in
  let zero_column = variant c in
  cover "a zero pivot" (zero_column && m > 0 && n > 0);
  let a =
    operand dt c.layout c.batch m n (fun _ _ j ->
        if zero_column && j = 0 then 0. else uniform st)
  in
  let am = of_nx a in
  let lu, pivots, perm = factors a in
  let lu = { am with x = lu } in
  for b = 0 to am.b - 1 do
    let rows = Array.init m Fun.id in
    for j = 0 to k - 1 do
      let p = Int64.to_int pivots.((b * k) + j) in
      if p < j || p >= m then failf "pivot %d of step %d" p j;
      let t = rows.(j) in
      rows.(j) <- rows.(p);
      rows.(p) <- t
    done;
    equal ~msg:"perm follows the pivots" (array int)
      rows
      (Array.init m (fun i -> Int64.to_int perm.((b * m) + i)))
  done;
  let l =
    init am.b m k (fun b i j ->
        if i = j then 1. else if i > j then get lu b i j else 0.)
  in
  let r = init am.b k n (fun b i j -> if i <= j then get lu b i j else 0.) in
  Array.iter (fun x -> bounded ~msg:"|L| <= 1" ~than:1. (Float.abs x)) l.x;
  let pa =
    init am.b m n (fun b i j ->
        get am b (Int64.to_int perm.((b * m) + i)) j)
  in
  let bound = product ~abs:true l r in
  let tol = gamma u (float_of_int k) +. gamma u64 (float_of_int (k + 1)) in
  elementwise ~msg:"L U = P a" (diff (product l r) pa)
    { bound with x = Array.map (( *. ) tol) bound.x }

(* [pivots a] is the pivots and the row order of the lowered LU of [a]. *)
let pivots rows a =
  let a = Nx.create D.Float64 [| rows; Array.length a / rows |] a in
  (Lowered.run (lu_pivots a), Lowered.run (lu_order a))

let pivot_rule () =
  (* A NaN below the diagonal is never the pivot; a tie keeps the first. *)
  equal (pair (array int64) (array int64)) ([| 2L; 1L |], [| 2L; 1L; 0L |])
    (pivots 3 [| 1.; 0.; Float.nan; 0.; 2.; 0. |]);
  (* Nothing is greater than a NaN on the diagonal. *)
  equal (pair (array int64) (array int64)) ([| 0L |], [| 0L; 1L |])
    (pivots 2 [| Float.nan; 5. |]);
  equal (pair (array int64) (array int64)) ([| 0L |], [| 0L; 1L |])
    (pivots 2 [| 1.; -1. |])

(* Triangular solves *)

(* The triangle, whether adjoint and whether of a unit diagonal, chosen by [c]'s
   seed. *)
let solve_flags c =
  ((if flag c 0 then Nx_kernel.Upper else Lower), flag c 1, flag c 2)

let solve_op (triangle, adjoint, unit_diagonal) a b () =
  linalg (Solve_triangular { triangle; adjoint; unit_diagonal; a; b })

(* The right-hand side of [a]'s batch, of [n] rows and [r] columns. *)
let rhs (type s) (dt : (float, s) D.t) a n r f =
  let batch = Array.sub (Nx.shape a) 0 (Array.length (Nx.shape a) - 2) in
  operand dt Rows batch n r f

let solve_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let n, r = match shape with [ n; r ] -> (n, r) | _ -> assert false in
  let a = shaped dt c n n in
  [ Lowered.One (solve_op (solve_flags c) a (rhs dt a n r (fun _ _ _ -> 0.))) ]

let solve_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let n, r = match shape with [ n; r ] -> (n, r) | _ -> assert false in
  cover_case c shape;
  let st = state c in
  let ((triangle, adjoint, unit_diagonal) as flags) = solve_flags c in
  cover "upper" (triangle = Upper);
  cover "adjoint" adjoint;
  cover "unit diagonal" unit_diagonal;
  let read i j = if triangle = Lower then i >= j else i <= j in
  let a =
    operand dt c.layout c.batch n n (fun _ i j ->
        if i = j && not unit_diagonal then
          let sign = if Random.State.bool st then 1. else -1. in
          sign *. (1. +. Random.State.float st 1.)
        else if read i j && i <> j then uniform st /. float_of_int n
        else garbage st)
  in
  let am = of_nx a in
  let b = rhs dt a n r (fun _ _ _ -> uniform st) in
  let x = Lowered.run (solve_op flags a b) in
  let x = mat am.b n r x in
  let t =
    init am.b n n (fun k i j ->
        let i, j = if adjoint then (j, i) else (i, j) in
        if i = j && unit_diagonal then 1.
        else if read i j then get am k i j
        else 0.)
  in
  let bound = product ~abs:true t x in
  let tol = gamma u (float_of_int n) +. gamma u64 (float_of_int (n + 1)) in
  elementwise ~msg:"op(a) x = b" (diff (product t x) (of_nx b))
    { bound with x = Array.map (( *. ) tol) bound.x }

let solve_singular () =
  let a = Nx.create D.Float64 [| 2; 2 |] [| 1.; 0.; 3.; 0. |] in
  let b = Nx.create D.Float64 [| 2; 1 |] [| 1.; 2. |] in
  let solve unit_diagonal =
    Lowered.run (fun () ->
        linalg
          (Solve_triangular
             { triangle = Lower; adjoint = false; unit_diagonal; a; b }))
  in
  equal (array bool) [| true; true |] (Array.map Float.is_nan (solve false));
  equal (array float_exact) [| 1.; -1. |] (solve true)

(* QR *)

(* [orthonormal ~msg ~tol q] asserts that the columns of each [q] have inner
   products within [tol] of the identity's, elementwise. *)
let orthonormal ~msg ~tol q =
  let g = product (transpose q) q in
  let d =
    init g.b g.m g.n (fun k i j -> get g k i j -. if i = j then 1. else 0.)
  in
  elementwise ~msg d { d with x = Array.make (Array.length d.x) tol }

(* [columns_near ~msg ~g a q r] asserts that each column [j] of [a] is within
   [g ‖a_j‖₂], plus the check's rounding, of [q r]'s. *)
let columns_near ~msg ~g a q r =
  let p = product q r and pa = product ~abs:true q r in
  for b = 0 to a.b - 1 do
    for j = 0 to a.n - 1 do
      let col t =
        Float.sqrt
          (List.fold_left ( +. ) 0.
             (List.init a.m (fun i -> get t b i j ** 2.)))
      in
      let check = gamma u64 (float_of_int (q.n + 2)) *. col pa in
      bounded ~msg:(Printf.sprintf "%s, column %d" msg j)
        ~than:((g *. col a) +. check)
        (col (diff a p))
    done
  done

let qr_op factors a () = linalg (Qr { factors; a })

let qr_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let f = qr_op (factors_of c) (shaped dt c m n) in
  Lowered.[ One (fun () -> fst (f ())); One (fun () -> snd (f ())) ]

let qr_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let k = Int.min m n in
  cover_case c shape;
  let st = state c in
  let factors = factors_of c in
  cover "complete" (factors = Complete && m > n);
  let zero_column = variant c in
  cover "a zero column" (zero_column && m > 1 && n > 1);
  let a =
    operand dt c.layout c.batch m n (fun _ _ j ->
        if zero_column && j = 1 then 0. else uniform st)
  in
  let am = of_nx a in
  let qc = if factors = Complete then m else k in
  let qr = qr_op factors a in
  let q = Lowered.run (fun () -> fst (qr ()))
  and r = Lowered.run (fun () -> snd (qr ())) in
  let q = mat am.b m qc q and r = mat am.b qc n r in
  for b = 0 to am.b - 1 do
    for i = 0 to qc - 1 do
      for j = 0 to Int.min i n - 1 do
        equal ~msg:"+0 below the diagonal" int64 0L
          (Int64.bits_of_float (get r b i j))
      done
    done
  done;
  let g = gamma u (float_of_int (k * ((2 * m) + 10))) in
  orthonormal ~msg:"qᵀ q = I" ~tol:(g +. gamma u64 (float_of_int (m + 1))) q;
  columns_near ~msg:"a = q r" ~g am q r

(* Singular values and Hermitian eigenvalues *)

(* [first t k] is the first [k] columns of [t]. *)
let first t k = init t.b t.m k (fun b i j -> get t b i j)

(* [rows_of t k] is the first [k] rows of [t]. *)
let rows_of t k = init t.b k t.n (fun b i j -> get t b i j)

(* The rounds of Jacobi rotations Lower_linalg takes at most for [n] positions
   at a roundoff of [u], [extra] [1] for the SVD's. *)
let rounds ?(extra = 0) n u =
  let p = -int_of_float (Float.log2 u) in
  let rec log2 k = if 1 lsl k >= n then k else log2 (k + 1) in
  if n < 2 then 0
  else (((p * (log2 0 + extra)) + 9) / 10) * (n - 1 + (n mod 2))

let frobenius t k =
  let s = ref 0. in
  for i = 0 to t.m - 1 do
    for j = 0 to t.n - 1 do
      s := !s +. (get t k i j ** 2.)
    done
  done;
  Float.sqrt !s

let eta u = if u = 0x1p-24 then 0x1p-149 else 0x1p-1074

(* [frobenius_near ~msg ~tol t] asserts that each matrix of [t] has a norm of at
   most [tol k], [k] its index in the batch. *)
let frobenius_near ~msg ~tol t =
  for b = 0 to t.b - 1 do
    bounded ~msg:(Printf.sprintf "%s, matrix %d" msg b) ~than:(tol b)
      (frobenius t b)
  done

let gram q =
  let g = product (transpose q) q in
  init g.b g.m g.n (fun k i j -> get g k i j -. if i = j then 1. else 0.)

let scaled_columns t s =
  init t.b t.m t.n (fun b i j -> get t b i j *. s.((b * t.n) + j))

(* [values_near ~msg ~tol ours eager] asserts that each value is within
   [tol b], [b] its matrix's index, of eager's. *)
let values_near ~msg ~tol k ours eager =
  Array.iteri
    (fun e v ->
      let b = if k = 0 then 0 else e / k in
      bounded ~msg:(Printf.sprintf "%s, value %d" msg e) ~than:(tol b)
        (Float.abs (v -. eager.(e))))
    ours

(* The SVD's bound [g] for [m] rows and [n] columns at roundoff [u], and the
   distance [near norm] its values may lie from eager's for a matrix of
   Frobenius norm [norm]: three [g] and eager's own error. *)
let svd_bound u m n =
  let k = Int.min m n and p = Int.max m n in
  let rf = float_of_int (rounds ~extra:1 k u) and kf = float_of_int k in
  let pf = float_of_int p in
  let g = gamma u (8. *. kf *. ((pf *. (pf +. 6.)) +. (8. *. rf))) in
  let near norm =
    (3. *. g *. norm) +. (gamma u ((16. *. pf) +. 100.) *. norm)
    +. (eta u *. ((16. *. pf) +. 100.))
  in
  (g, near)

let svd_values_op dt a () = linalg (Svd_values { dtype = dt; a })

(* Matrices whose squares overflow or underflow their dtype: their values are
   eager's within the bound. *)
let svd_extremes () =
  let check (type s) name (dt : (float, s) D.t) u m n x =
    let a = Nx.create dt [| m; n |] x in
    let s = Lowered.run (svd_values_op dt a) in
    let eager = Nx.to_array (linalg (Svd_values { dtype = dt; a })) in
    let _, near = svd_bound u m n in
    let norm = frobenius (of_nx a) 0 in
    values_near ~msg:name ~tol:(fun _ -> near norm) (Int.min m n) s eager
  in
  let st = Random.State.make [| 7 |] in
  let big scale = Array.init 12 (fun _ -> scale *. uniform st) in
  let values dt m n =
    Lowered.One
      (svd_values_op dt (Nx.create dt [| m; n |] (Array.make (m * n) 0.)))
  in
  Lowered.prime
    [
      values D.Float32 1 1;
      values D.Float32 4 3;
      values D.Float64 1 1;
      values D.Float64 3 4;
    ];
  check "float32 1e-25" D.Float32 0x1p-24 1 1 [| 1e-25 |];
  check "float32 1e20" D.Float32 0x1p-24 1 1 [| 1e20 |];
  check "float32 entries near 3e19" D.Float32 0x1p-24 4 3 (big 3e19);
  check "float64 1e-170" D.Float64 0x1p-53 1 1 [| 1e-170 |];
  check "float64 1e160" D.Float64 0x1p-53 1 1 [| 1e160 |];
  check "float64 entries near 1e155" D.Float64 0x1p-53 3 4 (big 1e155)

(* One program computes all three results, as a caller of [Svd] would. *)
let svd_op factors dt a () =
  let s, u, v = linalg (Svd { factors; dtype = dt; a }) in
  Nx.concatenate ~axis:0 [ Nx.flatten s; Nx.flatten u; Nx.flatten v ]

let svd_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let a = shaped dt c m n in
  [ Lowered.One (svd_op (factors_of c) dt a); Lowered.One (svd_values_op dt a) ]

let svd_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let k = Int.min m n and p = Int.max m n in
  cover_case c shape;
  let st = state c in
  let factors = factors_of c in
  cover "complete" (factors = Complete && m <> n);
  cover "wide" (m < n);
  let rank_one = variant c in
  cover "a rank-deficient matrix" (rank_one && k > 1);
  let ls = Array.init (2 * m) (fun _ -> uniform st)
  and rs = Array.init (2 * n) (fun _ -> uniform st) in
  let a =
    operand dt c.layout c.batch m n (fun b i j ->
        if rank_one then ls.((b * m) + i) *. rs.((b * n) + j) else uniform st)
  in
  let am = of_nx a in
  let uc = if factors = Complete then m else k
  and vr = if factors = Complete then n else k in
  let all = Lowered.run (svd_op factors dt a) in
  let ns = am.b * k and nu = am.b * m * uc in
  let s = Array.sub all 0 ns in
  let uu = mat am.b m uc (Array.sub all ns nu) in
  let vh = mat am.b vr n (Array.sub all (ns + nu) (am.b * vr * n)) in
  let values = Lowered.run (svd_values_op dt a) in
  equal ~msg:"values alone are Svd's" (array int64) (bits s) (bits values);
  let eager = Nx.to_array (linalg (Svd_values { dtype = dt; a })) in
  for b = 0 to am.b - 1 do
    for i = 0 to k - 1 do
      let si = s.((b * k) + i) in
      bounded ~msg:"not negative" ~than:si 0.;
      if i > 0 then bounded ~msg:"descending" ~than:s.((b * k) + i - 1) si
    done
  done;
  let pf = float_of_int p in
  let g, near = svd_bound u m n in
  let check = gamma u64 (float_of_int (p + 2)) in
  let norm b = frobenius am b in
  let near b = near (norm b) in
  values_near ~msg:"s against eager" ~tol:near k s eager;
  frobenius_near ~msg:"uᵀ u = I" ~tol:(fun _ -> g +. check) (gram uu);
  frobenius_near ~msg:"vh vhᵀ = I"
    ~tol:(fun _ -> g +. check)
    (gram (transpose vh));
  let usv = product (scaled_columns (first uu k) s) (rows_of vh k) in
  frobenius_near ~msg:"a = u diag s vh"
    ~tol:(fun b -> ((g +. (check *. pf)) *. norm b))
    (diff am usv)

(* A batch of a matrix with an infinity and one without: the first alone is NaN
   in every result. *)
let svd_fails () =
  let a =
    Nx.create D.Float64 [| 2; 2; 2 |]
      [| 1.; Float.infinity; 0.; 1.; 3.; 0.; 0.; 2. |]
  in
  let s =
    Lowered.run (fun () -> linalg (Svd_values { dtype = D.Float64; a }))
  in
  equal (array bool) [| true; true; false; false |] (Array.map Float.is_nan s);
  equal (array float_exact) [| 3.; 2. |] (Array.sub s 2 2)

(* The spectra the rotations converge on slowest, at the size where their count
   is tightest: 16 columns of values graded from 1 down to twice the norm the
   convergence test exempts, or of three values each repeated, in a random
   basis. The test passes and the values are eager's within the bound. *)
let spectrum_law (R (name, dt, u), (kind, seed)) =
  let k = 16 in
  cover name true;
  cover kind true;
  let st = Random.State.make [| seed |] in
  let basis () =
    let g =
      Nx.create D.Float64 [| k; k |] (Array.init (k * k) (fun _ -> uniform st))
    in
    Nx.to_array (fst (linalg (Qr { factors = Reduced; a = g })))
  in
  let q = basis () and v = basis () in
  let f = float_of_int in
  let noise =
    8. *. f (rounds ~extra:1 k u) *. (1. +. Float.sqrt (f k)) *. u
  in
  let s =
    Array.init k (fun i ->
        if kind = "graded" then (2. *. noise) ** (f i /. f (k - 1))
        else [| 1.; 0.5; 0.25 |].(i mod 3))
  in
  let x =
    Array.init (k * k) (fun e ->
        let i = e / k and j = e mod k in
        let acc = ref 0. in
        for l = 0 to k - 1 do
          acc := !acc +. (q.((i * k) + l) *. s.(l) *. v.((j * k) + l))
        done;
        !acc)
  in
  let a = Nx.create dt [| k; k |] x in
  let ours = Lowered.run (svd_values_op dt a) in
  let eager = Nx.to_array (linalg (Svd_values { dtype = dt; a })) in
  let _, near = svd_bound u k k in
  let norm = frobenius (of_nx a) 0 in
  values_near ~msg:"s against eager" ~tol:(fun _ -> near norm) k ours eager

let spectra =
  Gen.with_pp
    (fun ppf (R (name, _, _), (kind, seed)) ->
      Format.fprintf ppf "(%s, %s, %d)" name kind seed)
    (Gen.pair (Gen.of_list reals)
       (Gen.pair
          (Gen.of_list [ "graded"; "repeated" ])
          (Gen.int_range 0 1_000_000)))

let spectrum_examples =
  List.concat_map
    (fun dt -> [ (dt, ("graded", 1)); (dt, ("repeated", 2)) ])
    reals

(* A zero matrix of [dt] of [n] rows and columns, [b] of them where [b]. *)
let zeros ?b dt n =
  let batch = Option.to_list b in
  let count = List.fold_left ( * ) (n * n) batch in
  Nx.create dt (Array.of_list (batch @ [ n; n ])) (Array.make count 0.)

let spectrum_graphs (R (_, dt, _), _) =
  [ Lowered.One (svd_values_op dt (zeros dt 16)) ]

(* A zero singular value is +0, of a zero matrix and of a zero column. *)
let svd_zero () =
  let a =
    Nx.create D.Float64 [| 2; 2; 2 |] [| 0.; 0.; 0.; 0.; 1.; 0.; 2.; 0. |]
  in
  let s =
    Lowered.run (fun () -> linalg (Svd_values { dtype = D.Float64; a }))
  in
  equal (array int64) [| 0L; 0L; 0L |] (bits [| s.(0); s.(1); s.(3) |])

(* The distance an eigenvalue of [n] rows at roundoff [u] may lie from
   eager's for a matrix of Frobenius norm [norm]: the rotations' bound and
   eager's own error. *)
let eigh_near u n norm =
  let rf = float_of_int (rounds n u) and nf = float_of_int n in
  (gamma u (16. *. (rf +. nf)) *. norm)
  +. (gamma u (16. *. nf) *. norm)
  +. (eta u *. 16. *. nf)

(* [eigh_holds dt u n a] asserts the lowered [Eigh] and [Eigh_values] of the
   matrices of [a], of [n] rows, read from their lower triangle, against the
   bounds: ascending values within [eigh_near] of eager's, [vᵀ v = I] and [a =
   v diag w vᵀ] within [g]. *)
let eigh_op dt a () = linalg (Eigh { dtype = dt; a })
let eigh_values_op dt a () = linalg (Eigh_values { dtype = dt; a })

(* The eigenvalues and eigenvectors of the lowered [Eigh] of [a], each its own
   graph. *)
let eigh_results dt a =
  let f = eigh_op dt a in
  (Lowered.run (fun () -> fst (f ())), Lowered.run (fun () -> snd (f ())))

(* The graphs {!eigh_holds} computes of [a]. *)
let eigh_graphs dt a =
  let f = eigh_op dt a in
  Lowered.
    [
      One (fun () -> fst (f ()));
      One (fun () -> snd (f ()));
      One (eigh_values_op dt a);
    ]

let eigh_holds (type s) (dt : (float, s) D.t) u n a =
  let am = of_nx a in
  let sym = init am.b n n (fun k i j -> get am k (max i j) (min i j)) in
  let w, v = eigh_results dt a in
  let v = mat am.b n n v in
  let values = Lowered.run (eigh_values_op dt a) in
  let eager = Nx.to_array (linalg (Eigh_values { dtype = dt; a })) in
  for b = 0 to am.b - 1 do
    for i = 1 to n - 1 do
      bounded ~msg:"ascending" ~than:w.((b * n) + i) w.((b * n) + i - 1)
    done
  done;
  let rf = float_of_int (rounds n u) and nf = float_of_int n in
  let g = gamma u (16. *. (rf +. nf) *. (1. +. Float.sqrt nf)) in
  let check = gamma u64 (nf +. 2.) in
  let norm b = frobenius sym b in
  let near b = eigh_near u n (norm b) in
  values_near ~msg:"w against eager" ~tol:near n w eager;
  values_near ~msg:"values alone against eager" ~tol:near n values eager;
  frobenius_near ~msg:"vᵀ v = I" ~tol:(fun _ -> g +. check) (gram v);
  let vwv = product (scaled_columns v w) (transpose v) in
  frobenius_near ~msg:"a = v diag w vᵀ"
    ~tol:(fun b -> (g +. (check *. nf)) *. norm b)
    (diff sym vwv)

let eigh_law_graphs (c, shape) =
  let (R (_, dt, _)) = c.dt in
  let n = List.hd shape in
  eigh_graphs dt (shaped dt c n n)

let eigh_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let n = List.hd shape in
  cover_case c shape;
  let st = state c in
  let repeated = variant c in
  cover "a repeated eigenvalue" (repeated && n > 1);
  let junk = Array.init (2 * n * n) (fun _ -> garbage st) in
  let low = Array.init (2 * n * n) (fun _ -> uniform st) in
  (* A repeated eigenvalue: [I + q qᵀ] for a unit [q], [Q diag(1, …, 1, 2) Qᵀ],
     which the rotations must turn to diagonal. *)
  let q =
    Array.init 2 (fun _ ->
        let v = Array.init n (fun _ -> uniform st) in
        let norm = Float.sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0. v) in
        Array.map (fun x -> x /. norm) v)
  in
  let a =
    operand dt c.layout c.batch n n (fun b i j ->
        if i < j then junk.((((b * n) + i) * n) + j)
        else if repeated then
          (if i = j then 1. else 0.) +. (q.(b).(i) *. q.(b).(j))
        else low.((((b * n) + i) * n) + j))
  in
  eigh_holds dt u n a

(* Spectra

   The matrices the rotations converge on slowest, and those they settle at
   once: eigenvalues spread log-uniformly over 16 decades, half [-1] and half
   within [1e-10] of [1], one of [1e6] beside values in [[-1, 1]], in a random
   basis; a Gram matrix [g gᵀ] of [2n] columns whose scales fall over 6
   decades; and a diagonal matrix of signed zeros and values in [[-1, 1]], whose
   elements off the diagonal are zeros of either sign. *)

type spectrum = Graded | Clustered | Dominant | Gram | Diagonal | Nan | Infinite

let spectrum_name = function
  | Graded -> "graded"
  | Clustered -> "clustered"
  | Dominant -> "dominant"
  | Gram -> "gram"
  | Diagonal -> "diagonal"
  | Nan -> "NaN"
  | Infinite -> "an infinity"

(* The lower triangle of a symmetric matrix of [n] rows of [kind], in C order,
   upper elements NaN, or of [Nan] and [Infinite] a uniform matrix with one
   such element in its lower triangle. *)
let symmetric st kind n =
  let gauss () = uniform st +. uniform st +. uniform st in
  let basis () =
    let g =
      Nx.create D.Float64 [| n; n |] (Array.init (n * n) (fun _ -> gauss ()))
    in
    Nx.to_array (fst (linalg (Qr { factors = Reduced; a = g })))
  in
  let in_basis ev =
    let q = basis () in
    Array.init (n * n) (fun e ->
        let i = e / n and j = e mod n in
        let s = ref 0. in
        for l = 0 to n - 1 do
          s := !s +. (q.((i * n) + l) *. ev.(l) *. q.((j * n) + l))
        done;
        !s)
  in
  let full =
    match kind with
    | Graded ->
        in_basis
          (Array.init n (fun _ -> 10. ** (-16. *. Random.State.float st 1.)))
    | Clustered ->
        in_basis
          (Array.init n (fun i ->
               if i < n / 2 then -1. else 1. +. (1e-10 *. uniform st)))
    | Dominant ->
        in_basis (Array.init n (fun i -> if i = 0 then 1e6 else uniform st))
    | Gram ->
        let m = 2 * n in
        let g =
          Array.init (n * m) (fun e ->
              let j = float_of_int (e mod m) in
              gauss () *. (10. ** (-6. *. j /. float_of_int m)))
        in
        Array.init (n * n) (fun e ->
            let i = e / n and j = e mod n in
            let s = ref 0. in
            for l = 0 to m - 1 do
              s := !s +. (g.((i * m) + l) *. g.((j * m) + l))
            done;
            !s)
    | Diagonal ->
        let zero () = if Random.State.bool st then 0. else -0. in
        let d =
          Array.init n (fun _ ->
              if Random.State.bool st then zero () else uniform st)
        in
        Array.init (n * n) (fun e ->
            let i = e / n and j = e mod n in
            if i = j then d.(i) else zero ())
    | Nan | Infinite ->
        let x = Array.init (n * n) (fun _ -> uniform st) in
        let i = Random.State.int st n in
        let j = Random.State.int st (i + 1) in
        x.((i * n) + j) <-
          (if kind = Nan then Float.nan else Float.neg_infinity);
        x
  in
  Array.mapi (fun e x -> if e mod n > e / n then Float.nan else x) full

(* The slow spectra, at sizes even and odd, one and two: the values are
   eager's, and the vectors orthonormal, within the bounds. *)
let eigh_spectrum (R (name, dt, u), (kind, (n, seed))) =
  cover name true;
  cover (spectrum_name kind) true;
  cover "an odd size" (n mod 2 = 1);
  let st = Random.State.make [| seed |] in
  let a = Nx.create dt [| n; n |] (symmetric st kind n) in
  eigh_holds dt u n a

let eigh_spectrum_graphs (R (_, dt, _), (_, (n, _))) =
  eigh_graphs dt (zeros dt n)

let slow = [ Graded; Clustered; Dominant; Gram ]
let sizes = [ 1; 2; 3; 7; 16 ]

let eigh_spectra =
  Gen.with_pp
    (fun ppf (R (name, _, _), (kind, (n, seed))) ->
      Format.fprintf ppf "(%s, %s, %d, %d)" name (spectrum_name kind) n seed)
    (Gen.pair (Gen.of_list reals)
       (Gen.pair (Gen.of_list slow)
          (Gen.pair (Gen.of_list sizes) (Gen.int_range 0 1_000_000))))

let eigh_spectrum_examples =
  List.concat_map
    (fun dt ->
      List.mapi (fun i kind -> (dt, (kind, (16 - (i mod 2 * 9), i)))) slow)
    reals

(* A matrix's results are its own, bit for bit: beside a matrix that needs more
   sweeps or fails, they are those it has beside itself, in a batch of the same
   shape. *)
let eigh_alone_graphs (R (_, dt, _), (_, (n, _))) =
  eigh_graphs dt (zeros ~b:2 dt n)

let eigh_alone (R (name, dt, _), ((ka, kb), (n, seed))) =
  cover name true;
  cover "a settled matrix beside one that rotates"
    (ka = Diagonal && kb <> Diagonal);
  cover "beside a matrix that fails" (kb = Nan || kb = Infinite);
  cover "a matrix that fails" (ka = Nan || ka = Infinite);
  cover "an odd size" (n mod 2 = 1);
  let st = Random.State.make [| seed |] in
  let x = symmetric st ka n and y = symmetric st kb n in
  let pair x y = Nx.create dt [| 2; n; n |] (Array.append x y) in
  let results a =
    let w, v = eigh_results dt a in
    let values = Lowered.run (eigh_values_op dt a) in
    (Array.sub w 0 n, Array.sub v 0 (n * n), Array.sub values 0 n)
  in
  let w, v, values = results (pair x x) in
  let w', v', values' = results (pair x y) in
  equal ~msg:"w" (array int64) (bits w) (bits w');
  equal ~msg:"v" (array int64) (bits v) (bits v');
  equal ~msg:"values alone" (array int64) (bits values) (bits values')

let kinds = [ Graded; Clustered; Dominant; Gram; Diagonal; Nan; Infinite ]

(* Pairs of kinds of the dtype and size of one of the examples below, each as
   likely: a run lowers and compiles each example's routines once. *)
let eigh_pair_examples =
  let f32 = List.nth reals 0 and f64 = List.nth reals 1 in
  [
    (f64, ((Diagonal, Graded), (4, 1)));
    (f32, ((Diagonal, Clustered), (5, 2)));
    (f64, ((Graded, Nan), (3, 3)));
    (f32, ((Infinite, Gram), (8, 4)));
    (f64, ((Diagonal, Infinite), (2, 5)));
  ]

let eigh_pairs =
  Gen.with_pp
    (fun ppf (R (name, _, _), ((ka, kb), (n, seed))) ->
      Format.fprintf ppf "(%s, %s beside %s, %d, %d)" name (spectrum_name ka)
        (spectrum_name kb) n seed)
    (Gen.map
       (fun ((dt, (_, (n, _))), (pair, seed)) -> (dt, (pair, (n, seed))))
       (Gen.pair
          (Gen.of_list eigh_pair_examples)
          (Gen.pair
             (Gen.pair (Gen.of_list kinds) (Gen.of_list kinds))
             (Gen.int_range 0 1_000_000))))

(* NaN in the lower triangle makes its matrix NaN in both results. *)
let eigh_fails () =
  let a =
    Nx.create D.Float32 [| 2; 2; 2 |]
      [| 1.; 0.; Float.nan; 1.; 2.; 0.; 0.; 3. |]
  in
  let w =
    Lowered.run (fun () -> linalg (Eigh_values { dtype = D.Float32; a }))
  in
  equal (array bool) [| true; true; false; false |] (Array.map Float.is_nan w);
  equal (array float_exact) [| 2.; 3. |] (Array.sub w 2 2)

(* Matrices of a norm past the dtype's range or of subnormal elements, alone
   and in one batch, whose matrices each take their own scale: [[a, b],
   [b, -a]] has eigenvalues [±hypot a b] and norm [√2 hypot a b]. Each computed
   eigenvalue is within the stated [γ(16 (R + n))] of the norm, and a
   subnormal's spacing, of the exact one, and [‖vᵀ v - I‖_F] is within the
   stated [g]. *)
let eigh_extremes () =
  let check (type s) name (dt : (float, s) D.t) u pairs =
    let k = List.length pairs in
    let m =
      Nx.create dt [| k; 2; 2 |]
        (Array.concat (List.map (fun (a, b) -> [| a; b; b; -.a |]) pairs))
    in
    let x = Nx.to_array m in
    let l = Array.init k (fun i -> Float.hypot x.(4 * i) x.((4 * i) + 1)) in
    let exact =
      Array.init (2 * k) (fun e ->
          if e mod 2 = 0 then -.l.(e / 2) else l.(e / 2))
    in
    let rf = float_of_int (rounds 2 u) in
    let tol b =
      (gamma u (16. *. (rf +. 2.)) *. l.(b) *. Float.sqrt 2.) +. eta u
    in
    let g = gamma u (16. *. (rf +. 2.) *. (1. +. Float.sqrt 2.)) in
    values_near ~msg:name ~tol 2 (Lowered.run (eigh_values_op dt m)) exact;
    let w, v = eigh_results dt m in
    values_near ~msg:(name ^ ", with vectors") ~tol 2 w exact;
    let v = mat k 2 2 v in
    frobenius_near ~msg:(name ^ ": vᵀ v = I")
      ~tol:(fun _ -> g +. gamma u64 4.)
      (gram v)
  in
  let big32 = (2.5e38, 1e38) and small32 = (1e-39, 4e-40) in
  let big64 = (1.2e308, 5e307) and small64 = (1e-309, 4e-310) in
  Lowered.prime
    (List.concat_map
       (fun b -> eigh_graphs D.Float32 (zeros ~b D.Float32 2))
       [ 1; 2 ]
    @ List.concat_map
        (fun b -> eigh_graphs D.Float64 (zeros ~b D.Float64 2))
        [ 1; 2 ]);
  check "float32 norm past 3.4e38" D.Float32 0x1p-24 [ big32 ];
  check "float32 subnormal" D.Float32 0x1p-24 [ small32 ];
  check "float32 both in one batch" D.Float32 0x1p-24 [ big32; small32 ];
  check "float64 norm past 1.8e308" D.Float64 0x1p-53 [ big64 ];
  check "float64 subnormal" D.Float64 0x1p-53 [ small64 ];
  check "float64 both in one batch" D.Float64 0x1p-53 [ big64; small64 ]

(* A compiled [Nx.eigh] of 32 rows of float64 gives eager's eigenvalues within
   the bound above, and compiles within a minute: the largest magnitude of the
   symmetric matrix folds 32 lanes into a chain of maxima in a loop of 32
   trips, which LLVM 22's and 23's loop idiom recognition did not finish. *)
let eigh_32 () =
  let n = 32 in
  let st = Random.State.make [| n |] in
  let x = Array.init (n * n) (fun _ -> uniform st) in
  let sym = init 1 n n (fun _ i j -> x.((max i j * n) + min i j)) in
  let a = Nx.create D.Float64 [| n; n |] sym.x in
  let eigh =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (fun a -> Nx.eigh D.Float64 a)
  in
  let w, _ = eigh a in
  let eager = fst (Nx.eigh D.Float64 a) in
  values_near ~msg:"w against eager"
    ~tol:(fun _ -> eigh_near u64 n (frobenius sym 0))
    n (Nx.to_array w) (Nx.to_array eager)

(* An SVD's results of a matrix are its own, bit for bit: beside a matrix that
   needs more sweeps or fails, they are those it has beside itself, in a batch
   of the same shape. A matrix is the symmetric one of {!symmetric}'s kind,
   both triangles stored, whose singular values are its eigenvalues'
   magnitudes. *)
let svd_s dt a () =
  let s, _, _ = linalg (Svd { dtype = dt; factors = Reduced; a }) in
  s

let svd_u dt a () =
  let _, u, _ = linalg (Svd { dtype = dt; factors = Reduced; a }) in
  u

let svd_vh dt a () =
  let _, _, vh = linalg (Svd { dtype = dt; factors = Reduced; a }) in
  vh

let svd_alone_graphs (R (_, dt, _), (_, (n, _))) =
  let a = zeros ~b:2 dt n in
  Lowered.
    [
      One (svd_u dt a);
      One (svd_s dt a);
      One (svd_vh dt a);
      One (svd_values_op dt a);
    ]

let svd_alone (R (name, dt, _), ((ka, kb), (n, seed))) =
  cover name true;
  cover "a settled matrix beside one that rotates"
    (ka = Diagonal && kb <> Diagonal);
  cover "beside a matrix that fails" (kb = Nan || kb = Infinite);
  cover "a matrix that fails" (ka = Nan || ka = Infinite);
  cover "an odd size" (n mod 2 = 1);
  let st = Random.State.make [| seed |] in
  let full kind =
    let l = symmetric st kind n in
    Array.init (n * n) (fun e ->
        let i = e / n and j = e mod n in
        l.((max i j * n) + min i j))
  in
  let x = full ka and y = full kb in
  let pair x y = Nx.create dt [| 2; n; n |] (Array.append x y) in
  (* The first matrix's [u], [s] and [vh], each read alone as a user who reads
     one computes it, and its values alone. *)
  let results a =
    let first k f = Array.sub (Lowered.run (f dt a)) 0 k in
    ( first (n * n) svd_u,
      first n svd_s,
      first (n * n) svd_vh,
      first n svd_values_op )
  in
  let u, s, vh, values = results (pair x x) in
  let u', s', vh', values' = results (pair x y) in
  equal ~msg:"u" (array int64) (bits u) (bits u');
  equal ~msg:"s" (array int64) (bits s) (bits s');
  equal ~msg:"vh" (array int64) (bits vh) (bits vh');
  equal ~msg:"values alone" (array int64) (bits values) (bits values')

(* Stopping changes no value: on the host each matrix stops once its sweeps
   settle, and its values, [-0] equal to [+0], are those of the full count of
   sweeps a device whose work is not this process's code runs. Of a batch of
   two matrices of {!symmetric}'s kinds, each read whole for the SVD. *)
(* The results in two graphs, flattened in turn: those of the factors, and the
   values alone. Each graph's results read [a] and [b], which share no
   intermediate value, and none reads another's result. *)
let stop_ops dt a b =
  let factors () =
    let w, v = linalg (Eigh { dtype = dt; a }) in
    let s, u, vh = linalg (Svd { dtype = dt; factors = Reduced; a = b }) in
    Nx.concatenate ~axis:0 (List.map Nx.flatten [ w; v; s; u; vh ])
  in
  let values () =
    Nx.concatenate ~axis:0
      (List.map Nx.flatten
         [
           linalg (Eigh_values { dtype = dt; a });
           linalg (Svd_values { dtype = dt; a = b });
         ])
  in
  (factors, values)

let stop_graphs (R (_, dt, _), (_, (n, _))) =
  let factors, values = stop_ops dt (zeros ~b:2 dt n) (zeros ~b:2 dt n) in
  [ Lowered.One factors; Lowered.One values ]

let stop_law (R (name, dt, _), ((ka, kb), (n, seed))) =
  cover name true;
  cover "a settled matrix beside one that rotates"
    (ka = Diagonal && kb <> Diagonal);
  cover "a matrix that fails" (ka = Nan || ka = Infinite);
  cover "an odd size" (n mod 2 = 1);
  let st = Random.State.make [| seed |] in
  let x = symmetric st ka n and y = symmetric st kb n in
  let full l =
    Array.init (n * n) (fun e ->
        let i = e / n and j = e mod n in
        l.((max i j * n) + min i j))
  in
  let a = Nx.create dt [| 2; n; n |] (Array.append x y) in
  let b = Nx.create dt [| 2; n; n |] (Array.append (full x) (full y)) in
  let value v =
    Int64.bits_of_float (if Float.is_nan v then Float.nan else v +. 0.)
  in
  let factors, values = stop_ops dt a b in
  let parts lens x =
    let at = ref 0 in
    List.map
      (fun len ->
        let p = Array.sub x !at len in
        at := !at + len;
        Array.map value p)
      lens
  in
  let same msgs lens f =
    let fixed = Lowered.run ~local:false f and stopped = Lowered.run f in
    List.iter2
      (fun msg (f, s) -> equal ~msg (array int64) f s)
      msgs
      (List.combine (parts lens fixed) (parts lens stopped))
  in
  same
    [ "w"; "v"; "s"; "u"; "vh" ]
    [ 2 * n; 2 * n * n; 2 * n; 2 * n * n; 2 * n * n ]
    factors;
  same [ "eigenvalues alone"; "singular values alone" ] [ 2 * n; 2 * n ] values

(* A compiled [Nx.eigvalsh] in a scan's step, its loop of sweeps inside the
   scan's loop of steps: each step's values are eager's within the bound, for
   matrices that need different numbers of sweeps. *)
let eigh_in_scan () =
  let n = 6 in
  let st = Random.State.make [| n |] in
  let kinds = [ Diagonal; Graded; Clustered; Gram ] in
  let mats = List.map (fun kind -> symmetric st kind n) kinds in
  let xs = Nx.create D.Float64 [| 4; n; n |] (Array.concat mats) in
  let step c a =
    let w = Nx.eigvalsh D.Float64 a in
    (Nx.add c (Nx.sum w), w)
  in
  let scan xs = snd (Rune.scan' step ~init:(Nx.scalar D.Float64 0.) xs) in
  let ours = Nx.to_array (Rune.jit' scan xs) in
  let eager = Nx.to_array (Nx.eigvalsh D.Float64 xs) in
  let am = of_nx xs in
  let sym = init am.b n n (fun k i j -> get am k (max i j) (min i j)) in
  values_near ~msg:"each step's values against eager"
    ~tol:(fun b -> eigh_near u64 n (frobenius sym b))
    n ours eager

(* Through Nx: a compiled [Nx.svdvals] and [Nx.eigvalsh] are eager's within
   the bounds above, [eigvalsh] of either triangle. *)
let values_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  cover_case c shape;
  let st = state c in
  match shape with
  | [ m; n ] ->
      let a = operand dt c.layout c.batch m n (fun _ _ _ -> uniform st) in
      let ours = Nx.to_array (Rune.jit' (Nx.svdvals dt) a) in
      let eager = Nx.to_array (Nx.svdvals dt a) in
      let _, near = svd_bound u m n in
      let am = of_nx a in
      values_near ~msg:"svdvals"
        ~tol:(fun b -> near (frobenius am b))
        (Int.min m n) ours eager
  | n :: _ ->
      let triangle : Nx.triangle = if flag c 1 then Upper else Lower in
      let a = operand dt c.layout c.batch n n (fun _ _ _ -> uniform st) in
      let ours = Nx.to_array (Rune.jit' (Nx.eigvalsh ~triangle dt) a) in
      let eager = Nx.to_array (Nx.eigvalsh ~triangle dt a) in
      let am = of_nx a in
      let sym =
        init am.b n n (fun k i j ->
            match triangle with
            | Lower -> get am k (max i j) (min i j)
            | Upper -> get am k (min i j) (max i j))
      in
      values_near ~msg:"eigvalsh"
        ~tol:(fun b -> eigh_near u n (frobenius sym b))
        n ours eager
  | [] -> fail "a shape"

(* Refusals *)

let refuses f =
  raises_match
    (function Rune.Lower_arith.Unlowered _ -> true | _ -> false)
    (fun () -> Lowered.run f)

let refused () =
  let z = Nx.create D.Complex64 [| 1; 1 |] [| Complex.one |] in
  let a = Nx.create D.Float32 [| 1; 1 |] [| 1. |] in
  refuses (fun () -> linalg (Cholesky { triangle = Lower; a = z }));
  refuses (fun () -> fst (linalg (Eig { dtype = D.Complex64; a })))

(* Suites *)

let law name ~count ?(extra = []) ~graphs shapes f =
  let examples = examples shapes @ extra in
  prop name ~count ~examples (cases_of examples)
    (Lowered.primed graphs examples f)

(* Complete factors of a tall and a wide matrix: q's and u's columns past [n],
   and vh's rows past [m], which no reduced factor holds. Seeds [3] and [4]
   choose complete factors, [3] the variant. *)
let complete_examples =
  let f32 = List.nth reals 0 and f64 = List.nth reals 1 in
  [
    ({ dt = f32; layout = Rows; batch = [||]; seed = 3 }, [ 3; 2 ]);
    ({ dt = f64; layout = Columns; batch = [| 2 |]; seed = 4 }, [ 4; 2 ]);
    ({ dt = f32; layout = Broadcast; batch = [||]; seed = 4 }, [ 2; 4 ]);
    ({ dt = f64; layout = Rows; batch = [||]; seed = 3 }, [ 2; 3 ]);
  ]

let square = [ [ 0 ]; [ 1 ]; [ 2 ]; [ 3 ]; [ 5 ] ]
let general = [ [ 0; 2 ]; [ 2; 0 ]; [ 1; 1 ]; [ 3; 2 ]; [ 2; 3 ]; [ 4; 4 ] ]

let () =
  exit
    (run "Lower_linalg"
       [
         group "cholesky"
           [
             law "L Lᵀ reconstructs a" ~count:4 ~graphs:cholesky_graphs square
               cholesky_law;
             test "a matrix that is not positive definite is NaN"
               cholesky_fails;
             test "a zero pivot of either sign is NaN" cholesky_zero_pivot;
           ];
         group "lu"
           [
             law "L U reconstructs P a" ~count:4 ~graphs:lu_graphs general
               lu_law;
             test "pivots follow the rule" pivot_rule;
           ];
         group "solve_triangular"
           [
             law "x solves op(a) x = b" ~count:6 ~graphs:solve_graphs
               [ [ 0; 1 ]; [ 1; 1 ]; [ 3; 2 ]; [ 4; 0 ]; [ 5; 1 ] ]
               solve_law;
             test "a zero diagonal is NaN" solve_singular;
           ];
         group "qr"
           [
             law "q r reconstructs a" ~count:4 ~extra:complete_examples
               ~graphs:qr_graphs general qr_law;
           ];
         group "svd"
           [
             law "u diag s vh reconstructs a" ~count:4 ~extra:complete_examples
               ~graphs:svd_graphs general svd_law;
             test "a matrix with an infinity is NaN" svd_fails;
             prop "graded and repeated spectra converge" ~count:2
               ~examples:spectrum_examples spectra
               (Lowered.primed spectrum_graphs spectrum_examples spectrum_law);
             test "squares past the dtype's range" svd_extremes;
             test "a zero singular value is +0" svd_zero;
             prop "each matrix's results are its own, bit for bit" ~count:6
               ~examples:eigh_pair_examples eigh_pairs
               (Lowered.primed svd_alone_graphs eigh_pair_examples svd_alone);
           ];
         group "eigh"
           [
             law "v diag w vᵀ reconstructs a" ~count:4 ~graphs:eigh_law_graphs
               square eigh_law;
             test "NaN in the lower triangle is NaN" eigh_fails;
             test "norms past the dtype's range, and subnormals" eigh_extremes;
             prop "graded, clustered, dominant and Gram spectra converge"
               ~count:4 ~examples:eigh_spectrum_examples eigh_spectra
               (Lowered.primed eigh_spectrum_graphs eigh_spectrum_examples
                  eigh_spectrum);
             prop "each matrix's results are its own, bit for bit" ~count:6
               ~examples:eigh_pair_examples eigh_pairs
               (Lowered.primed eigh_alone_graphs eigh_pair_examples eigh_alone);
             test "compiled in a scan's step" eigh_in_scan;
             test "compiled for 32 rows of float64 gives eager's eigenvalues"
               eigh_32;
           ];
         prop "stopping early gives the full count's values" ~count:6
           ~examples:eigh_pair_examples eigh_pairs
           (Lowered.primed ~local:false stop_graphs eigh_pair_examples
              (Lowered.primed stop_graphs eigh_pair_examples stop_law));
         law "Nx's values alone compiled are eager's" ~count:6
           ~graphs:(fun _ -> [])
           (square @ general) values_law;
         test "complex operands and eig are refused" refused;
       ])
