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

let cases_of shapes =
  Gen.with_pp
    (fun ppf (c, s) ->
      Format.fprintf ppf "%a %s" pp_case c
        (String.concat "x" (List.map string_of_int s)))
    (Gen.map
       (fun (((dt, layout), batch), (shape, seed)) ->
         ({ dt; layout; batch; seed }, shape))
       (Gen.pair
          (Gen.pair (Gen.pair (Gen.of_list reals) (Gen.of_list layouts))
             (Gen.of_list [ [||]; [| 2 |] ]))
          (Gen.pair (Gen.of_list shapes) (Gen.int_range 0 1_000_000))))

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
  let run triangle =
    Lowered.run (fun () -> linalg (Cholesky { triangle; a }))
  in
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
  let lu, pivots, perm =
    ( Lowered.run (fun () ->
          let x, _, _ = linalg (Lu a) in
          x),
      Lowered.run (fun () ->
          let _, p, _ = linalg (Lu a) in
          p),
      Lowered.run (fun () ->
          let _, _, p = linalg (Lu a) in
          p) )
  in
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
  ( Lowered.run (fun () ->
        let _, p, _ = linalg (Lu a) in
        p),
    Lowered.run (fun () ->
        let _, _, p = linalg (Lu a) in
        p) )

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

let solve_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let n, r = match shape with [ n; r ] -> (n, r) | _ -> assert false in
  cover_case c shape;
  let st = state c in
  let triangle = if flag c 0 then Nx_kernel.Upper else Lower in
  let adjoint = flag c 1 and unit_diagonal = flag c 2 in
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
  let batch = Array.sub (Nx.shape a) 0 (Array.length (Nx.shape a) - 2) in
  let b = operand dt Rows batch n r (fun _ _ _ -> uniform st) in
  let x =
    Lowered.run (fun () ->
        linalg (Solve_triangular { triangle; adjoint; unit_diagonal; a; b }))
  in
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

let qr_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let k = Int.min m n in
  cover_case c shape;
  let st = state c in
  let factors = if c.seed / 3 mod 2 = 1 then Nx_kernel.Complete else Reduced in
  cover "complete" (factors = Complete && m > n);
  let zero_column = variant c in
  cover "a zero column" (zero_column && m > 1 && n > 1);
  let a =
    operand dt c.layout c.batch m n (fun _ _ j ->
        if zero_column && j = 1 then 0. else uniform st)
  in
  let am = of_nx a in
  let qc = if factors = Complete then m else k in
  let q =
    mat am.b m qc
      (Lowered.run (fun () -> fst (linalg (Qr { factors; a }))))
  in
  let r =
    mat am.b qc n
      (Lowered.run (fun () -> snd (linalg (Qr { factors; a }))))
  in
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

(* The rounds of Jacobi rotations Lower_linalg takes for [n] positions at a
   precision of [p] bits. *)
let rounds n u =
  let p = -int_of_float (Float.log2 u) in
  let rec log2 k = if 1 lsl k >= n then k else log2 (k + 1) in
  if n < 2 then 0 else (((p * log2 0) + 9) / 10) * (n - 1 + (n mod 2))

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
  let rf = float_of_int (rounds k u) and kf = float_of_int k in
  let pf = float_of_int p in
  let g = gamma u (16. *. (rf +. (pf *. pf)) *. (kf +. 2.) *. Float.sqrt kf) in
  let near norm =
    (3. *. g *. norm) +. (gamma u ((16. *. pf) +. 100.) *. norm)
    +. (eta u *. ((16. *. pf) +. 100.))
  in
  (g, near)

(* Matrices whose squares overflow or underflow their dtype: their values are
   eager's within the bound. *)
let svd_extremes () =
  let check (type s) name (dt : (float, s) D.t) u m n x =
    let a = Nx.create dt [| m; n |] x in
    let s = Lowered.run (fun () -> linalg (Svd_values { dtype = dt; a })) in
    let eager = Nx.to_array (linalg (Svd_values { dtype = dt; a })) in
    let _, near = svd_bound u m n in
    let norm = frobenius (of_nx a) 0 in
    values_near ~msg:name ~tol:(fun _ -> near norm) (Int.min m n) s eager
  in
  let st = Random.State.make [| 7 |] in
  let big scale = Array.init 12 (fun _ -> scale *. uniform st) in
  check "float32 1e-25" D.Float32 0x1p-24 1 1 [| 1e-25 |];
  check "float32 1e20" D.Float32 0x1p-24 1 1 [| 1e20 |];
  check "float32 entries near 3e19" D.Float32 0x1p-24 4 3 (big 3e19);
  check "float64 1e-170" D.Float64 0x1p-53 1 1 [| 1e-170 |];
  check "float64 1e160" D.Float64 0x1p-53 1 1 [| 1e160 |];
  check "float64 entries near 1e155" D.Float64 0x1p-53 3 4 (big 1e155)

let svd_law (c, shape) =
  let (R (_, dt, u)) = c.dt in
  let m, n = match shape with [ m; n ] -> (m, n) | _ -> assert false in
  let k = Int.min m n and p = Int.max m n in
  cover_case c shape;
  let st = state c in
  let factors = if c.seed / 3 mod 2 = 1 then Nx_kernel.Complete else Reduced in
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
  (* One program computes all three results, as a caller of [Svd] would. *)
  let all =
    Lowered.run (fun () ->
        let s, u, v = linalg (Svd { factors; dtype = dt; a }) in
        Nx.concatenate ~axis:0 [ Nx.flatten s; Nx.flatten u; Nx.flatten v ])
  in
  let ns = am.b * k and nu = am.b * m * uc in
  let s = Array.sub all 0 ns in
  let uu = mat am.b m uc (Array.sub all ns nu) in
  let vh = mat am.b vr n (Array.sub all (ns + nu) (am.b * vr * n)) in
  let values = Lowered.run (fun () -> linalg (Svd_values { dtype = dt; a })) in
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

(* A zero singular value is +0, of a zero matrix and of a zero column. *)
let svd_zero () =
  let a =
    Nx.create D.Float64 [| 2; 2; 2 |] [| 0.; 0.; 0.; 0.; 1.; 0.; 2.; 0. |]
  in
  let s =
    Lowered.run (fun () -> linalg (Svd_values { dtype = D.Float64; a }))
  in
  equal (array int64) [| 0L; 0L; 0L |] (bits [| s.(0); s.(1); s.(3) |])

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
  let am = of_nx a in
  let sym = init am.b n n (fun k i j -> get am k (max i j) (min i j)) in
  let w = Lowered.run (fun () -> fst (linalg (Eigh { dtype = dt; a }))) in
  let v =
    mat am.b n n
      (Lowered.run (fun () -> snd (linalg (Eigh { dtype = dt; a }))))
  in
  let values = Lowered.run (fun () -> linalg (Eigh_values { dtype = dt; a })) in
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
  let near b =
    (gamma u (16. *. (rf +. nf)) *. norm b) +. (gamma u (16. *. nf) *. norm b)
    +. (eta u *. 16. *. nf)
  in
  values_near ~msg:"w against eager" ~tol:near n w eager;
  values_near ~msg:"values alone against eager" ~tol:near n values eager;
  frobenius_near ~msg:"vᵀ v = I" ~tol:(fun _ -> g +. check) (gram v);
  let vwv = product (scaled_columns v w) (transpose v) in
  frobenius_near ~msg:"a = v diag w vᵀ"
    ~tol:(fun b -> (g +. (check *. nf)) *. norm b)
    (diff sym vwv)

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

(* Refusals *)

let refuses f =
  raises_match (function Invalid_argument _ -> true | _ -> false) (fun () ->
      Lowered.run f)

let refused () =
  let z = Nx.create D.Complex64 [| 1; 1 |] [| Complex.one |] in
  let a = Nx.create D.Float32 [| 1; 1 |] [| 1. |] in
  refuses (fun () -> linalg (Cholesky { triangle = Lower; a = z }));
  refuses (fun () -> fst (linalg (Eig { dtype = D.Complex64; a })))

(* Suites *)

let law name ~count ?(extra = []) shapes f =
  prop name ~count ~examples:(examples shapes @ extra) (cases_of shapes) f

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
             law "L Lᵀ reconstructs a" ~count:4 square cholesky_law;
             test "a matrix that is not positive definite is NaN"
               cholesky_fails;
             test "a zero pivot of either sign is NaN" cholesky_zero_pivot;
           ];
         group "lu"
           [
             law "L U reconstructs P a" ~count:4 general lu_law;
             test "pivots follow the rule" pivot_rule;
           ];
         group "solve_triangular"
           [
             law "x solves op(a) x = b" ~count:6
               [ [ 0; 1 ]; [ 1; 1 ]; [ 3; 2 ]; [ 4; 0 ]; [ 5; 1 ] ]
               solve_law;
             test "a zero diagonal is NaN" solve_singular;
           ];
         group "qr"
           [
             law "q r reconstructs a" ~count:4 ~extra:complete_examples general
               qr_law;
           ];
         group "svd"
           [
             law "u diag s vh reconstructs a" ~count:4 ~extra:complete_examples
               general svd_law;
             test "a matrix with an infinity is NaN" svd_fails;
             test "squares past the dtype's range" svd_extremes;
             test "a zero singular value is +0" svd_zero;
           ];
         group "eigh"
           [
             law "v diag w vᵀ reconstructs a" ~count:4 square eigh_law;
             test "NaN in the lower triangle is NaN" eigh_fails;
           ];
         test "complex operands and eig are refused" refused;
       ])
