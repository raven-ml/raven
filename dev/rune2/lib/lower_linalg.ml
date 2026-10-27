(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module P = Nx.Prim

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

let dtype = Ops.dtype
let ints l = List.map (fun n -> Ops.Int n) l
let zero u = Shape.const_like u (`Float 0.)
let float u x = Shape.const_like u (`Float x)
let transpose u = Shape.transpose u (-2) (-1)
let abs x = Lower_real.unary Abs x
let fdiv x y = Lower_real.binary Fdiv x (Shape.expand y (Shape.shape x))

(* The batch axes of the matrices of [u], their rows and their columns. *)
let matrix u =
  let dims = Shape.max_shape u in
  let r = List.length dims in
  ( List.filteri (fun i _ -> i < r - 2) dims,
    List.nth dims (r - 2),
    List.nth dims (r - 1) )

(* [block u rows cols] is the rows and columns of the matrices of [u] in the
   bounds [rows] and [cols], each all of them if [None]. *)
let block u rows cols =
  let r = Shape.ndim u in
  let bound = Option.map (fun (lo, hi) -> (Ops.Int lo, Ops.Int hi)) in
  Shape.shrink u
    (List.init r (fun i ->
         if i = r - 2 then bound rows
         else if i = r - 1 then bound cols
         else None))

(* The diagonal of the matrices of [u]: the first column of their elements laid
   out in rows of one more element. *)
let diagonal u =
  let batch, n, _ = matrix u in
  let flat = Shape.reshape u (ints (batch @ [ n * n ])) in
  let padded =
    Shape.pad flat
      (List.map (fun _ -> None) batch @ [ Some (Ops.Int 0, Ops.Int n) ])
  in
  Shape.reshape
    (block
       (Shape.reshape padded (ints (batch @ [ n; n + 1 ])))
       None
       (Some (0, 1)))
    (ints (batch @ [ n ]))

(* The row and the column of each element of a matrix of [m] rows and [n]
   columns. *)
let row_index m = Shape.reshape (Shape.arange ~dtype:Int32 m) (ints [ m; 1 ])
let column_index n = Shape.reshape (Shape.arange ~dtype:Int32 n) (ints [ 1; n ])

(* Indexed reads

   [take x axis p] is the element of [x] at position [p] along [axis], at each
   index of [p], whose other extents are [x]'s or smaller: a read of [x]'s
   elements in C order at [Σ index · stride], which moves each element with its
   bits. A position outside the axis reads its nearest end; one arises only on
   a matrix whose results are NaN. *)
let take x axis p =
  let s = Shape.max_shape x and ps = Shape.max_shape p in
  let n = List.nth s axis in
  if n = 0 || List.mem 0 ps then
    Shape.expand (Ops.const ~dtype:(dtype x) (`Int Bigint.zero)) (ints ps)
  else
    let r = List.length s in
    let stride d = List.fold_left ( * ) 1 (List.filteri (fun e _ -> e > d) s) in
    let coord d =
      if d = axis then
        let bound k = Shape.const_like p (`Int (Bigint.of_int k)) in
        let p = Ops.maximum (Ops.minimum p (bound (n - 1))) (bound 0) in
        Ops.cast p Weak_int
      else
        let k = List.nth ps d in
        Shape.expand
          (Shape.reshape
             (Shape.arange ~dtype:Weak_int k)
             (ints (List.init r (fun e -> if e = d then k else 1))))
          (ints ps)
    in
    let term d =
      match stride d with 1 -> coord d | k -> Ops.mul (coord d) (Ops.int k)
    in
    let at =
      List.fold_left (fun at d -> Ops.add at (term d)) (term 0)
        (List.init (r - 1) succ)
    in
    Ops.index (Shape.reshape x [ Ops.Int (List.fold_left ( * ) 1 s) ]) [ at ]

(* [at axis u i] is element [i] of [u] along [axis], one per matrix, the axis
   kept of extent one: [i] is a scalar or of [u]'s rank. *)
let at axis u i =
  let s = Shape.max_shape u in
  let one = List.mapi (fun d n -> if d = axis then 1 else n) s in
  let i =
    if Shape.ndim i = List.length s then i
    else Shape.reshape i (ints (List.map (fun _ -> 1) s))
  in
  take u axis (Shape.expand i (ints one))

(* [first_max axis u] is the position of the first greatest element of [u]
   along [axis], which holds no NaN, the axis kept of extent one: of the
   elements equal to the maximum, the one that counts down furthest from the
   axis's extent. *)
let first_max axis u =
  let r = Shape.ndim u and n = List.nth (Shape.max_shape u) axis in
  let keep op x = Shape.unsqueeze (Shape.rop x op [ axis ]) axis in
  let down =
    Shape.reshape
      (Shape.arange ~start:n ~step:(-1) ~dtype:Int32 0)
      (ints (List.init r (fun d -> if d = axis then n else 1)))
  in
  let hit = Ops.where (Ops.eq u (keep Op.Max u)) down (Ops.int 0) in
  Ops.sub (Ops.int n) (keep Op.Max (Ops.cast hit Int32))

(* Failing matrices

   A matrix that breaks a routine's precondition on its values has results
   whose every element is NaN, as nx states. Whether each matrix keeps its
   results is a flag of its batch axes and two axes of one. *)

(* [defined ok u] is [u] where the flag [ok] holds, and NaN in every element of
   the matrices of [u] where it does not: of its vectors, when [u] has an axis
   fewer than [ok]. *)
let defined ok u =
  let ok =
    if Shape.ndim u < Shape.ndim ok then Shape.squeeze ~axis:(-1) ok else ok
  in
  Ops.where (Shape.expand ok (Shape.shape u)) u (float u Float.nan)

(* Loops

   A routine repeats its step in a loop on the device of its operand's
   storage ({!Loop.repeat}). A loop of no trip, or of values with no element,
   runs nothing and needs none. *)

let repeat device n init step =
  let empty u = Shape.max_numel u = 0 in
  if n = 0 || List.for_all empty init then init
  else
    match device with
    | Some d -> Loop.repeat d n init step
    | None -> unlowered "a Linalg of an operand that reads no storage"

(* LU with partial pivoting

   Step [j] takes as pivot the first row of largest magnitude in column [j] on
   or below the diagonal, a NaN on the diagonal being the largest and one below
   it the least, and swaps it with row [j], moving each element with its bits.
   The column below the pivot is divided by it, unless it is zero, and the
   trailing rows take the rank-one update. Step [j] writes its pivot's row at
   position [j] of the pivots. *)

let lu device a =
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let rank = Shape.ndim a in
  let index = row_index m and cols = column_index n in
  let positions = Shape.arange ~dtype:Int32 k in
  (* [swap u j p] exchanges row [j] and row [p], one per matrix, of [u]. *)
  let swap u j p =
    let _, _, c = matrix u in
    let row_p = take u (rank - 2) (Shape.expand p (ints (batch @ [ 1; c ]))) in
    Ops.where (Ops.eq index j) row_p
      (Ops.where (Ops.eq index p) (at (rank - 2) u j) u)
  in
  let step j = function
    | [ x; perm; pivots ] ->
        let col = at (rank - 1) x j in
        let never = float col Float.neg_infinity in
        let magnitude =
          Ops.where (Ops.lt index j) never
            (Ops.where (Ops.ne col col)
               (Ops.where (Ops.eq index j) (float col Float.infinity) never)
               (abs col))
        in
        let p = first_max (rank - 2) magnitude in
        let x = swap x j p and perm = swap perm j p in
        let col = at (rank - 1) x j in
        let pivot = at (rank - 2) col j in
        let l = Ops.where (Ops.ne pivot (zero pivot)) (fdiv col pivot) col in
        let below = Ops.gt index j in
        let x =
          Ops.where
            (Ops.bitwise_and below (Ops.eq cols j))
            l
            (Ops.where
               (Ops.bitwise_and below (Ops.gt cols j))
               (Ops.sub x (Ops.mul l (at (rank - 2) x j)))
               x)
        in
        let p = Ops.cast (Shape.squeeze ~axis:(-1) p) Int64 in
        let pivots =
          Ops.where (Ops.eq positions j)
            (Shape.expand p (Shape.shape pivots))
            pivots
        in
        [ x; perm; pivots ]
    | _ -> invalid_arg "Lower_linalg.lu"
  in
  let perm = Shape.expand (Ops.cast index Int64) (ints (batch @ [ m; 1 ])) in
  let pivots =
    Shape.expand
      (Ops.const ~dtype:Int64 (`Int Bigint.zero))
      (ints (batch @ [ k ]))
  in
  match repeat device k [ a; perm; pivots ] step with
  | [ x; perm; pivots ] -> [| x; pivots; Shape.squeeze ~axis:(-1) perm |]
  | _ -> invalid_arg "Lower_linalg.lu"

(* Cholesky

   Step [j] takes column [j] of the working matrix, on and below the diagonal,
   as column [j] of [L]: its diagonal element's square root heads it, and the
   rest is divided by that root. The trailing rows and columns then lose the
   product of the column with itself. Only the lower triangle is ever read. A
   pivot [d] is positive exactly where its root is: [sqrt] of a negative or NaN
   [d] is NaN, of [+0] or [-0] a zero. *)

let cholesky device ~upper a =
  let _, n, _ = matrix a in
  let rank = Shape.ndim a in
  let rows = row_index n and cols = column_index n in
  let lower = Ops.ge rows cols in
  let step j = function
    | [ s ] ->
        let c = at (rank - 1) s j in
        let root = Ops.sqrt (at (rank - 2) c j) in
        let l =
          Ops.where (Ops.eq rows j)
            (Shape.expand root (Shape.shape c))
            (Ops.where (Ops.gt rows j) (fdiv c root) (zero root))
        in
        let trailing = Ops.bitwise_and (Ops.gt rows j) (Ops.gt cols j) in
        [
          Ops.where (Ops.eq cols j) l
            (Ops.where trailing (Ops.sub s (Ops.mul l (transpose l))) s);
        ]
    | _ -> invalid_arg "Lower_linalg.cholesky"
  in
  let s = List.hd (repeat device n [ Ops.where lower a (zero a) ] step) in
  let l = Ops.where lower s (zero s) in
  let d = diagonal l in
  let failed = Ops.cast (Ops.logical_not (Ops.gt d (zero d))) Int32 in
  let none = Ops.eq (Shape.rop failed Op.Add [ rank - 2 ]) (Ops.int 0) in
  let ok = Shape.unsqueeze (Shape.unsqueeze none (-1)) (-1) in
  defined ok (if upper then transpose l else l)

(* Triangular solve

   The system is made lower triangular, [aᵀ] under [adjoint], and reversed
   along both axes, with [b]'s rows, when the triangle it reads is the upper
   one. Step [i] divides row [i] of the right-hand sides by the diagonal
   element, which makes it row [i] of the solution, and takes its product with
   column [i] of the matrix from the rows below. A zero diagonal element makes
   the matrix singular. *)

let solve_triangular device ~upper ~adjoint ~unit_diagonal a b =
  let m = if adjoint then transpose a else a in
  let rank = Shape.ndim m in
  let reversed = upper <> adjoint in
  let m = if reversed then Shape.flip m [ rank - 2; rank - 1 ] else m in
  let b = if reversed then Shape.flip b [ rank - 2 ] else b in
  let batch, n, _ = matrix m in
  let rows = row_index n in
  let step i = function
    | [ x ] ->
        let c = at (rank - 1) m i in
        let xi = at (rank - 2) x i in
        let xi = if unit_diagonal then xi else fdiv xi (at (rank - 2) c i) in
        [
          Ops.where (Ops.eq rows i)
            (Shape.expand xi (Shape.shape x))
            (Ops.where (Ops.gt rows i) (Ops.sub x (Ops.mul c xi)) x);
        ]
    | _ -> invalid_arg "Lower_linalg.solve_triangular"
  in
  let x = List.hd (repeat device n [ b ] step) in
  let x = if reversed then Shape.flip x [ rank - 2 ] else x in
  if unit_diagonal || n = 0 then x
  else
    let d = diagonal m in
    let zeros = Ops.cast (Ops.eq d (zero d)) Int32 in
    let none = Ops.eq (Shape.rop zeros Op.Add [ rank - 2 ]) (Ops.int 0) in
    defined (Shape.reshape none (ints (batch @ [ 1; 1 ]))) x

(* Operations *)

(* The device of [u]'s storage: its own, or that of the first node it reads
   that has one. *)
let device u =
  match Ops.device u with
  | Some d -> Some d
  | None -> List.find_map Ops.device (Ops.toposort ~calls:Skip u)

let real (type v s) name (dt : (v, s) D.t) =
  if D.is D.Complex dt then unlowered ("a complex " ^ name)

let op (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type d r)
    (l : (d, r) P.linalg) : Ops.t array =
  let operand name a =
    real name (Nx.dtype a);
    let u = node a in
    (device u, u)
  in
  match l with
  | Cholesky { triangle; a } ->
      let d, a = operand "Cholesky" a in
      [| cholesky d ~upper:(triangle = Upper) a |]
  | Lu a ->
      let d, a = operand "Lu" a in
      lu d a
  | Qr _ -> unlowered "Qr"
  | Svd_values _ -> unlowered "Svd_values"
  | Svd _ -> unlowered "Svd"
  | Eigh_values _ -> unlowered "Eigh_values"
  | Eigh _ -> unlowered "Eigh"
  | Solve_triangular { triangle; adjoint; unit_diagonal; a; b } ->
      let d, a = operand "Solve_triangular" a in
      let e, b = operand "Solve_triangular" b in
      let d = if Option.is_some d then d else e in
      [|
        solve_triangular d ~upper:(triangle = Upper) ~adjoint ~unit_diagonal a
          b;
      |]
  | Eig_values _ -> unlowered "Eig_values"
  | Eig _ -> unlowered "Eig"
