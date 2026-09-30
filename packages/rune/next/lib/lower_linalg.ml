(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

let dtype = Ops.dtype
let ints l = List.map (fun n -> Ops.Int n) l
let zero u = Ops.const_like u (`Float 0.)
let float u x = Ops.const_like u (`Float x)

let dims u =
  List.map
    (function Ops.Int n -> n | Ops.Sym _ -> invalid_arg "a symbolic size")
    (Ops.shape u)

(* The batch axes of the matrices of [u], their rows and their columns. *)
let matrix u =
  match List.rev (dims u) with
  | n :: m :: batch -> (List.rev batch, m, n)
  | _ -> invalid_arg "a matrix of fewer than two axes"

let unsqueeze u axis =
  let s = Ops.shape u in
  Ops.reshape u
    (List.filteri (fun i _ -> i < axis) s
    @ (Ops.Int 1 :: List.filteri (fun i _ -> i >= axis) s))

(* [transpose u] swaps the last two axes of [u]. *)
let transpose u =
  let r = Ops.ndim u in
  Ops.permute u
    (List.init r (fun i ->
         if i = r - 2 then r - 1 else if i = r - 1 then r - 2 else i))

(* [block u rows cols] is the rows and columns of the matrices of [u] in the
   bounds [rows] and [cols], each all of them if [None]. *)
let block u rows cols =
  let r = Ops.ndim u in
  let bound = Option.map (fun (lo, hi) -> (Ops.Int lo, Ops.Int hi)) in
  Ops.shrink u
    (List.init r (fun i ->
         if i = r - 2 then bound rows
         else if i = r - 1 then bound cols
         else None))

let row u i = block u (Some (i, i + 1)) None
let column u j = block u None (Some (j, j + 1))
let entry u i j = block u (Some (i, i + 1)) (Some (j, j + 1))

(* The row and the column of each element of a matrix of [m] rows and [n]
   columns. *)
let row_index m = Ops.reshape (Ops.arange ~dtype:Int32 m) (ints [ m; 1 ])
let column_index n = Ops.reshape (Ops.arange ~dtype:Int32 n) (ints [ 1; n ])
let is u i = Ops.eq u (Ops.int i)
let eye dt n m = Ops.cast (Ops.eq (row_index n) (Ops.arange ~dtype:Int32 m)) dt

(* The constant [c] of type [dt] broadcast to [shape]. *)
let filled dt c shape =
  Ops.expand
    (Ops.reshape (Ops.const ~dtype:dt c) (List.map (fun _ -> Ops.Int 1) shape))
    (ints shape)

(* Narrow floats compute at [float32] and round each result once. *)
let widen u =
  if Dtype.is_float (dtype u) && Dtype.itemsize (dtype u) < 4 then
    Ops.cast u Float32
  else u

let fdiv x y = Lower_arith.binary Fdiv x (Ops.expand y (Ops.shape x))

(* Products

   [dot a b] multiplies the matrices of [a] and [b] as the reference's [dot]
   does: each row of [a] against each column of [b] along a new last axis,
   broadcast over the batch axes, multiplied and summed by [sum]. *)

let dot ?(sum = fun u axis -> Ops.rop u Op.Add [ axis ]) a b =
  let a = unsqueeze a (Ops.ndim a - 1) in
  let b = unsqueeze (transpose b) (Ops.ndim b - 2) in
  let p = Ops.mul a b in
  sum p (Ops.ndim p - 1)

let matmul a b =
  let dt = dtype a in
  let acc = Lower_reduce.accumulator dt in
  let sum u axis = Lower_reduce.reduce Sum ~axes:[ axis ] u in
  Ops.cast (dot ~sum (Ops.cast a acc) (Ops.cast b acc)) dt

(* Householder QR

   Step [i] reflects the part of column [i] of [r] on and below the diagonal,
   [x], onto the diagonal: by [I - w vᵀ], [v] being 1 at row [i] and [x] scaled
   below it, applied to [r] on the left and accumulated in [q] on the right. The
   diagonal takes the sign opposite to [x]'s first element, so that no
   cancellation occurs, and a zero column takes no reflection. *)

let sum_last u =
  let axis = Ops.ndim u - 1 in
  unsqueeze (Ops.rop u Op.Add [ axis ]) axis

let sign x =
  Ops.where
    (Ops.ne x (zero x))
    (Ops.where (Ops.lt x (zero x)) (float x (-1.)) (float x 1.))
    (zero x)

let householder a =
  let batch, m, n = matrix a in
  let idx = Ops.arange ~dtype:Int32 m in
  let reflect (q, r) i =
    let at_i = is idx i in
    let c = Ops.squeeze ~axis:(-1) (column r i) in
    let x = Ops.where (Ops.ge idx (Ops.int i)) c (zero c) in
    let norm = Ops.sqrt (sum_last (Ops.mul x x)) in
    let x0 = sum_last (Ops.where at_i x (zero x)) in
    let sgn = Ops.where (Ops.ne x0 (zero x0)) (sign x0) (float x0 1.) in
    let active = Ops.ne norm (zero norm) in
    let u0 = Ops.O.(x0 + (sgn * norm)) in
    let v = Ops.div (Ops.where at_i u0 x) (Ops.where active u0 (float u0 1.)) in
    let v = unsqueeze v (Ops.ndim v) in
    let tau =
      Ops.where active
        (Ops.div (Ops.mul sgn u0) (Ops.where active norm (float norm 1.)))
        (zero norm)
    in
    let w = Ops.mul (unsqueeze tau (Ops.ndim tau)) v in
    ( Ops.sub q (dot (dot q v) (transpose w)),
      Ops.sub r (dot w (dot (transpose v) r)) )
  in
  let q = Ops.expand (eye (dtype a) m m) (ints (batch @ [ m; m ])) in
  List.fold_left reflect (q, a) (List.init (Int.min m n) Fun.id)

(* The upper triangle of the matrices of [u], zeros below it. *)
let triu u =
  let _, m, n = matrix u in
  Ops.where (Ops.le (row_index m) (column_index n)) u (zero u)

let qr ~reduced a =
  let dt = dtype a in
  let _, m, n = matrix a in
  let k = Int.min m n in
  let q, r = householder (widen a) in
  let r = triu r in
  let q, r =
    if reduced then (block q None (Some (0, k)), block r (Some (0, k)) None)
    else (q, r)
  in
  (Ops.cast q dt, Ops.cast r dt)

(* Singular values by one-sided Jacobi rotations

   The columns of [u], [r]'s leading square, are made orthogonal by rotating
   pairs of them, each by the angle that zeroes their inner product; [v]
   accumulates the rotations. A round rotates [num / 2] disjoint pairs, drawn as
   in a round-robin tournament, so that each pair meets once every [num - 1]
   rounds: the shape fixes them all. *)

(* The pairs of each of [rounds] rounds of a tournament of [num] columns. *)
let tournament num rounds =
  let h = num / 2 in
  let next p =
    if num mod 2 = 1 then Array.map (fun i -> (i + num - 1) mod num) p
    else
      Array.mapi
        (fun k i -> if k = 0 then i else ((i + num - 3) mod (num - 1)) + 1)
        p
  in
  let first = Array.init num (fun i -> if i < h then i else num - 1 - i + h) in
  let rec go p k =
    if k = 0 then []
    else List.init h (fun i -> (p.(i), p.(h + i))) :: go (next p) (k - 1)
  in
  go first rounds

(* The rounds that bring [num] columns to orthogonality. A sweep meets every
   pair once, in [num - 1] rounds, or [num] for an odd [num]. The rotations
   converge quadratically once the columns are nearly orthogonal, and the sweeps
   it takes to get there grow as the logarithm of [num]. *)
let rounds num =
  let rec log2 k = if 1 lsl k >= num then k else log2 (k + 1) in
  if num < 2 then 0 else (log2 0 + 3) * (num - 1 + (num mod 2))

let columns u ks =
  match List.map (column u) ks with
  | c :: cs -> Ops.cat ~axis:(Ops.ndim u - 1) c cs
  | [] -> invalid_arg "no column"

let rotate (u, v) pairs =
  let first, second = List.split pairs in
  let sums x =
    let axis = Ops.ndim x - 2 in
    unsqueeze (Ops.rop x Op.Add [ axis ]) axis
  in
  let ui = columns u first and uj = columns u second in
  let gamma = sums (Ops.mul ui uj) in
  let alpha = sums (Ops.mul ui ui) and beta = sums (Ops.mul uj uj) in
  let rot = Ops.ne gamma (zero gamma) in
  let tau =
    Ops.div (Ops.sub beta alpha)
      (Ops.mul (float gamma 2.) (Ops.where rot gamma (float gamma 1.)))
  in
  let t =
    Ops.div
      (Ops.where (Ops.ne tau (zero tau)) (sign tau) (float tau 1.))
      (Ops.add
         (Lower_arith.unary Abs tau)
         (Ops.sqrt (Ops.add (float tau 1.) (Ops.mul tau tau))))
  in
  let t = Ops.where rot t (zero t) in
  let c = Ops.reciprocal (Ops.sqrt (Ops.add (float t 1.) (Ops.mul t t))) in
  let s = Ops.mul c t in
  (* Column [k] of the rotated [x] is among [x]'s columns, then the rotated
     firsts, then the rotated seconds. *)
  let _, _, num = matrix u and h = List.length pairs in
  let order k =
    match
      (List.find_index (( = ) k) first, List.find_index (( = ) k) second)
    with
    | Some i, _ -> num + i
    | None, Some i -> num + h + i
    | None, None -> k
  in
  let turn x =
    let xi = columns x first and xj = columns x second in
    let xi = Ops.O.((c * xi) - (s * xj)) and xj = Ops.O.((s * xi) + (c * xj)) in
    columns (Ops.cat ~axis:(Ops.ndim x - 1) x [ xi; xj ]) (List.init num order)
  in
  (turn u, turn v)

let svd ~full_matrices a =
  let dt = dtype a in
  let batch, m, n = matrix a in
  let num = Int.min m n and q_num = Int.max m n in
  let x = widen a in
  let wdt = dtype x in
  let q, r = householder (if m >= n then x else transpose x) in
  let square k = Ops.expand (eye wdt k k) (ints (batch @ [ k; k ])) in
  let u = Ops.contiguous (block (triu r) (Some (0, num)) (Some (0, num))) in
  let v = Ops.contiguous (square num) in
  let u, v = List.fold_left rotate (u, v) (tournament num (rounds num)) in
  let r = Ops.ndim u in
  let norms = Ops.sqrt (Ops.rop (Ops.mul u u) Op.Add [ r - 2 ]) in
  let order = Lower_reduce.argsort ~descending:true ~axis:(r - 2) norms in
  let s = Lower_index.gather (r - 2) order norms in
  let by_order x =
    Lower_index.gather (r - 1)
      (Ops.expand (unsqueeze order (r - 2)) (ints (batch @ [ num; num ])))
      x
  in
  (* The sorted columns are orthogonal: the reflections that triangularize them
     are an orthogonal basis of which each column, with the sign it takes on the
     diagonal, is the direction, and which completes the columns of zero
     norm. *)
  let q_u, r_u = householder (by_order u) in
  let diagonal =
    Ops.rop
      (Ops.where (Ops.eq (row_index num) (column_index num)) r_u (zero r_u))
      Op.Add
      [ r - 2 ]
  in
  let signs =
    Ops.where
      (Ops.lt diagonal (zero diagonal))
      (float diagonal (-1.)) (float diagonal 1.)
  in
  let u = Ops.mul q_u (unsqueeze signs (r - 2)) and v = by_order v in
  let inside =
    Ops.bitwise_and
      (Ops.lt (row_index q_num) (Ops.int num))
      (Ops.lt (column_index q_num) (Ops.int num))
  in
  let pad =
    List.map (fun _ -> None) batch
    @ [
        Some (Ops.Int 0, Ops.Int (q_num - num));
        Some (Ops.Int 0, Ops.Int (q_num - num));
      ]
  in
  let u = dot q (Ops.where inside (Ops.pad u pad) (square q_num)) in
  let u = if full_matrices then u else block u None (Some (0, num)) in
  let u, vt = if m >= n then (u, transpose v) else (v, transpose u) in
  (Ops.cast u dt, Ops.cast s Float64, Ops.cast vt dt)

(* LU with partial pivoting

   Step [j] takes as pivot the first row of largest magnitude in column [j] on
   or below the diagonal, a NaN on the diagonal being the largest and one below
   it the least, and swaps it with row [j], moving each element with its bits.
   The column below the pivot is divided by it, unless it is zero, and the
   trailing rows take the rank-one update. *)

let lu a =
  let dt = dtype a in
  let x = widen a in
  let batch, m, n = matrix x in
  let rank = Ops.ndim x in
  let rows = row_index m and cols = column_index n in
  (* [swap u j p] exchanges row [j] and row [p], one per matrix, of [u]. *)
  let swap u j p =
    let _, _, c = matrix u in
    let row_p =
      Lower_index.gather (rank - 2) (Ops.expand p (ints (batch @ [ 1; c ]))) u
    in
    Ops.where (is rows j) row_p (Ops.where (Ops.eq rows p) (row u j) u)
  in
  let step (x, perm, pivots) j =
    let col = column x j in
    let never = float col Float.neg_infinity in
    let magnitude =
      Ops.where
        (Ops.lt rows (Ops.int j))
        never
        (Ops.where (Ops.ne col col)
           (Ops.where (is rows j) (float col Float.infinity) never)
           (Lower_arith.unary Abs col))
    in
    let p = Lower_reduce.arg_reduce Argmax ~axis:(rank - 2) magnitude in
    let pp = unsqueeze p (rank - 2) in
    let x = swap x j pp and perm = swap perm j pp in
    let pivot = entry x j j and col = column x j in
    let l = Ops.where (Ops.ne pivot (zero pivot)) (fdiv col pivot) col in
    let below = Ops.gt rows (Ops.int j) in
    let x =
      Ops.where
        (Ops.bitwise_and below (is cols j))
        l
        (Ops.where
           (Ops.bitwise_and below (Ops.gt cols (Ops.int j)))
           (Ops.sub x (Ops.mul l (row x j)))
           x)
    in
    (x, perm, p :: pivots)
  in
  let perm = Ops.expand (Ops.arange ~dtype:Int32 m) (ints (batch @ [ m ])) in
  let x, perm, pivots =
    List.fold_left step
      (x, unsqueeze perm (rank - 1), [])
      (List.init (Int.min m n) Fun.id)
  in
  let pivots =
    match List.rev pivots with
    | [] -> filled Int32 (`Int Z.zero) (batch @ [ 0 ])
    | ps -> Lower_index.cat (rank - 2) ps
  in
  (Ops.cast x dt, pivots, Ops.squeeze ~axis:(rank - 1) perm)

(* Cholesky

   Step [j] takes column [j] of the working matrix, on and below the diagonal,
   as column [j] of [L]: its diagonal element's square root heads it, and the
   rest is divided by that root. The working matrix then loses the product of
   the column with itself. Only the lower triangle is ever read. *)

let cholesky ~upper a =
  let dt = dtype a in
  let x = widen a in
  let _, n, _ = matrix x in
  let rank = Ops.ndim x in
  let rows = row_index n in
  let step (s, columns) j =
    let d = entry s j j in
    let root = Ops.sqrt (Ops.where (Ops.gt d (zero d)) d (float d Float.nan)) in
    let l =
      Ops.where (is rows j)
        (Ops.expand root (Ops.shape (column s j)))
        (Ops.where
           (Ops.gt rows (Ops.int j))
           (fdiv (column s j) root)
           (zero root))
    in
    (Ops.sub s (Ops.mul l (transpose l)), l :: columns)
  in
  if n = 0 then a
  else
    let _, columns = List.fold_left step (x, []) (List.init n Fun.id) in
    let l = Lower_index.cat (rank - 1) (List.rev columns) in
    Ops.cast (if upper then transpose l else l) dt

(* Triangular solve

   The system is made lower triangular, [aᵀ] under [transpose], and reversed
   along both axes, with [b]'s rows, when the triangle it reads is the upper
   one. Row [i] of the solution is then row [i] of [b] less the strictly lower
   part of row [i] of the matrix times the rows solved before it, divided by the
   diagonal element. *)

let solve_triangular ~upper ~transpose:t ~unit_diag a b =
  let dt = dtype b in
  let vector = Ops.ndim b = Ops.ndim a - 1 in
  let b = widen (if vector then unsqueeze b (Ops.ndim b) else b) in
  let m = widen (if t then transpose a else a) in
  let rank = Ops.ndim m in
  let reversed = upper <> t in
  let m = if reversed then Ops.flip m [ rank - 2; rank - 1 ] else m in
  let b = if reversed then Ops.flip b [ rank - 2 ] else b in
  let _, n, _ = matrix m in
  let rows = row_index n in
  let strict = Ops.where (Ops.lt (column_index n) rows) m (zero m) in
  let solve x i =
    let rest = Ops.sub (row b i) (dot (row strict i) x) in
    let xi = if unit_diag then rest else fdiv rest (entry m i i) in
    Ops.where (is rows i) (Ops.expand xi (Ops.shape x)) x
  in
  let x = List.fold_left solve (zero b) (List.init n Fun.id) in
  let x = if reversed then Ops.flip x [ rank - 2 ] else x in
  Ops.cast (if vector then Ops.squeeze ~axis:(rank - 1) x else x) dt
