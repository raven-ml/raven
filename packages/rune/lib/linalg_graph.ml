(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Trace-time linear algebra for the jit tracer.

   The eager backend runs QR and triangular solves as data-dependent C kernels
   (nx_c_qr.c, nx_c_tri.c). Tolk cannot express those: its Uop vocabulary has no
   host control flow, and reading a traced value to steer one is exactly what a
   compiled trace excludes. Neither operation needs data-dependent iteration,
   though — both take a number of steps fixed by the shapes alone. They are
   therefore unrolled here, at trace time, into ordinary Tolk compositions
   (matmuls, element-wise arithmetic, movement ops): min(m, n) Householder
   reflectors for QR, n forward-substitution steps for a triangular solve. The
   lowering sees a plain static graph and compiles it for every Tolk device.

   Conventions follow the eager kernels: the LAPACK reflector sign (beta =
   -sign(alpha)·‖x‖, so R's diagonal takes the reflected sign) and a column
   whose tail is already zero taking no reflector (tau = 0, R[j][j] = alpha).
   Compiled results match eager execution up to floating-point association. The
   subdiagonal of the factored working matrix stores the reflector tails — the
   LAPACK layout, sound because column j is never touched after step j — so
   forming Q re-reads them from the same graph with no extra storage.

   Graph size grows linearly in the matrix dimension: one small kernel cluster
   per step. Blocked (panel) factorizations would amortize the constant, not
   change the model. A linear solve is then a two-liner at trace time, [let q, r
   = qr ~reduced:true a in triangular_solve r (matmul qᵀ b)]; the eager
   [Nx.solve] itself still refuses to trace because its singularity check reads
   a traced value, so inside jit the composition is written out by hand and a
   singular system yields infinities rather than an error. *)

module F = Tolk_frontend

(* {1 Helpers} *)

(* Slice the last two axes of [t]; [None] leaves an axis whole. *)
let slice2 t rows cols =
  let shape = F.Tensor.shape t in
  let rank = List.length shape in
  F.Movement.shrink t
    (List.mapi
       (fun i s ->
         if i = rank - 2 then Option.value rows ~default:(0, s)
         else if i = rank - 1 then Option.value cols ~default:(0, s)
         else (0, s))
       shape)

(* Swap the last two axes. *)
let swap2 t =
  let rank = List.length (F.Tensor.shape t) in
  F.Movement.permute t
    (List.init rank (fun i ->
         if i = rank - 2 then rank - 1 else if i = rank - 1 then rank - 2 else i))

(* A constant of shape [batch @ [d1; d2]] in [t]'s dtype. *)
let scalar2 t batch d1 d2 v =
  F.Creation.full ~buffer:false ~dtype:(F.Tensor.val_dtype t)
    (batch @ [ d1; d2 ])
    (F.Tensor.Sfloat v)

let zero1 t batch = scalar2 t batch 1 1 0.0
let one1 t batch = scalar2 t batch 1 1 1.0

(* Concatenate matrix pieces along the second-to-last axis, dropping empty
   pieces ([cat] of a zero-height piece pads by zero, but there is no need to
   lean on that). At least one piece is never empty at every call site. *)
let cat2 ts =
  let nonempty t =
    let sh = F.Tensor.shape t in
    List.nth sh (List.length sh - 2) > 0
  in
  match List.filter nonempty ts with
  | [] -> invalid_arg "linalg_graph.cat2: every piece is empty"
  | [ x ] -> x
  | x :: xs -> F.Op.cat ~dim:(-2) x xs

let require_float ~what dt =
  if not (Tolk_uop.Dtype.is_float dt) then
    invalid_arg (Printf.sprintf "linalg_graph.%s: dtype must be float" what)

(* {1 QR}

   Householder QR, one reflector per column of the working matrix. Step j
   reflects the column tail below (and including) the diagonal to [beta·e_j],
   overwrites column j with the reflector (beta on the diagonal, the tail
   divided by [alpha - beta] below — the stored v), and applies H_j to the
   columns to the right as a rank-1 update built from two small matmuls. The
   reflector sign and the no-reflector case for zero tails follow the eager
   kernel. *)

let qr ~reduced a =
  let module E = F.Elementwise in
  require_float ~what:"qr" (F.Tensor.val_dtype a);
  let shape = F.Tensor.shape a in
  let rank = List.length shape in
  if rank < 2 then
    invalid_arg "linalg_graph.qr: input requires at least 2 dimensions";
  let m = List.nth shape (rank - 2) and n = List.nth shape (rank - 1) in
  let batch = List.filteri (fun i _ -> i < rank - 2) shape in
  let k = Stdlib.min m n in
  let work = ref a in
  let taus = Array.make k (zero1 a batch) in
  for j = 0 to k - 1 do
    let mtail = m - j - 1 in
    let alpha = slice2 !work (Some (j, j + 1)) (Some (j, j + 1)) in
    let tail = slice2 !work (Some (j + 1, m)) (Some (j, j + 1)) in
    let xnorm2 =
      if mtail = 0 then zero1 a batch
      else
        F.Reduce.sum ~axis:[ -2 ] ~keepdim:true ~dtype:(F.Tensor.val_dtype a)
          (E.mul tail tail)
    in
    (* beta = -sign(alpha)·‖x‖; a zero tail takes no reflector. *)
    let has = E.ne xnorm2 (zero1 a batch) in
    let anorm = E.sqrt (E.add (E.mul alpha alpha) xnorm2) in
    let beta = E.where (E.ge alpha (zero1 a batch)) (E.neg anorm) anorm in
    let tau = E.where has (E.div (E.sub beta alpha) beta) (zero1 a batch) in
    (* The reflector vector: 1 at the pivot, the scaled tail below, exactly 0
       throughout when no reflector is taken (so the trailing update is a no-op
       rather than a nan source when [alpha - beta] vanishes). *)
    let vtail =
      if mtail = 0 then tail
      else E.where has (E.div tail (E.sub alpha beta)) (zero1 a batch)
    in
    taus.(j) <- tau;
    (* Overwrite column j: the rows above the pivot are R's and stay, beta lands
       on the diagonal, the reflector tail below. *)
    let new_col =
      cat2
        ((if j > 0 then [ slice2 !work (Some (0, j)) (Some (j, j + 1)) ] else [])
        @ [ E.where has beta alpha ]
        @ if mtail > 0 then [ vtail ] else [])
    in
    (* Apply H_j to the columns right of j (v is zero above the pivot, so the
       rows above are untouched) and splice the result back. *)
    let left = if j = 0 then [] else [ slice2 !work None (Some (0, j)) ] in
    work :=
      if n - j - 1 = 0 then
        match left with
        | [] -> new_col
        | l :: tl -> F.Op.cat ~dim:(-1) l (tl @ [ new_col ])
      else
        let trailing = slice2 !work None (Some (j + 1, n)) in
        let v =
          cat2
            ((if j > 0 then [ scalar2 a batch j 1 0.0 ] else [])
            @ [ one1 a batch ]
            @ if mtail > 0 then [ vtail ] else [])
        in
        let proj = F.Op.matmul (swap2 v) trailing in
        let trailing' = E.sub trailing (E.mul tau (F.Op.matmul v proj)) in
        match left @ [ new_col; trailing' ] with
        | hd :: tl -> F.Op.cat ~dim:(-1) hd tl
        | [] -> assert false
  done;
  let nq = if reduced then k else m in
  let r =
    if reduced then slice2 (F.Op.triu !work) (Some (0, k)) None
    else F.Op.triu !work
  in
  (* Q = H_0 · H_1 ··· H_{k-1}, applied to the identity in reverse order,
     reading each v back out of the stored subdiagonal. A tau of 0 makes the
     steps for skipped columns (and the empty-tail last column of a square
     matrix) no-ops. *)
  let q =
    ref
      (F.Movement.expand
         (F.Op.eye ~m:nq ~dtype:(F.Tensor.val_dtype a) m)
         (batch @ [ m; nq ]))
  in
  for j = k - 1 downto 0 do
    let v =
      cat2
        ((if j > 0 then [ scalar2 a batch j 1 0.0 ] else [])
        @ [ one1 a batch ]
        @
        if m - j - 1 > 0 then
          [ slice2 !work (Some (j + 1, m)) (Some (j, j + 1)) ]
        else [])
    in
    let proj = E.mul taus.(j) (F.Op.matmul (swap2 v) !q) in
    q := E.sub !q (F.Op.matmul v proj)
  done;
  (!q, r)

(* {1 Triangular solve}

   Forward substitution over a lower triangular matrix, unrolled over the rows:
   row i subtracts the accumulated contributions [L[i, :i]·x[:i]] (a [1×i] by
   [i×nrhs] matmul) from the right-hand side and divides by the diagonal. The
   four flag combinations normalize to this one loop — the operand is transposed
   when [transpose] is set, and the whole system (coefficient matrix, right-hand
   side, and result, each on their row axis) is flipped when the triangle points
   up. Vector right-hand sides ride through the loop with a trailing unit axis
   and are squeezed at the end. *)

let triangular_solve ~upper ~transpose ~unit_diag a b =
  let module E = F.Elementwise in
  require_float ~what:"triangular_solve" (F.Tensor.val_dtype a);
  let shape = F.Tensor.shape a in
  let rank = List.length shape in
  let n = List.nth shape (rank - 1) in
  let batch = List.filteri (fun i _ -> i < rank - 2) shape in
  let vector_rhs = List.length (F.Tensor.shape b) = rank - 1 in
  let bm = if vector_rhs then F.Movement.unsqueeze b (-1) else b in
  if n = 0 then bm
  else
    (* Transposing swaps which triangle is which, so the whole system flips
       exactly when the effective triangle points up: [upper <> transpose]. *)
    let low =
      let m0 = if transpose then swap2 a else a in
      if upper <> transpose then F.Movement.flip m0 [ -2; -1 ] else m0
    in
    let flipped = upper <> transpose in
    let rhs = if flipped then F.Movement.flip bm [ -2 ] else bm in
    let diag_or_one i =
      if unit_diag then one1 a batch
      else slice2 low (Some (i, i + 1)) (Some (i, i + 1))
    in
    let x = ref (E.div (slice2 rhs (Some (0, 1)) None) (diag_or_one 0)) in
    for i = 1 to n - 1 do
      let row = slice2 low (Some (i, i + 1)) (Some (0, i)) in
      let partial = F.Op.matmul row !x in
      x :=
        F.Op.cat ~dim:(-2) !x
          [
            E.div
              (E.sub (slice2 rhs (Some (i, i + 1)) None) partial)
              (diag_or_one i);
          ]
    done;
    let x = if flipped then F.Movement.flip !x [ -2 ] else !x in
    if vector_rhs then F.Movement.squeeze ~dim:(-1) x else x

(* {1 Cholesky}

   Left-looking factorization, unrolled over the columns: with the first [j]
   columns of [L] finished, column [j] is the current column of [A] minus its
   projection onto those finished columns, and the pivot [L[j][j]] is the
   positive square root of the column's diagonal entry. This is the same
   per-element arithmetic as the eager kernel's unblocked path (same sums, in
   the same order), so results agree up to the blocked kernel's association
   differences. The eager kernel raises [Linalg_error] on a non-positive-
   definite matrix; the compiled graph has no host control flow to raise with,
   so a non-positive-definite input yields nans instead. [A = Uᵀ·U] is the
   transpose problem — factor [Aᵀ] and swap the result back. *)

let cholesky ~upper a =
  let module E = F.Elementwise in
  require_float ~what:"cholesky" (F.Tensor.val_dtype a);
  let shape = F.Tensor.shape a in
  let rank = List.length shape in
  if rank < 2 then
    invalid_arg "linalg_graph.cholesky: input requires at least 2 dimensions";
  let n = List.nth shape (rank - 1) in
  let batch = List.filteri (fun i _ -> i < rank - 2) shape in
  let low = if upper then swap2 a else a in
  if n = 0 then if upper then swap2 low else low
  else
    let dt = F.Tensor.val_dtype a in
    let l = ref (F.Creation.full ~buffer:false ~dtype:dt (batch @ [ n; 0 ]) (F.Tensor.Sfloat 0.0)) in
    for j = 0 to n - 1 do
      (* Project the finished columns off the current column of [A]: the
         sums [Σ_k L[i,k]·L[j,k] for every row i] in one [n×j by j×1]
         product. Empty at [j = 0]. *)
      let proj =
        if j = 0 then zero1 a batch
        else F.Op.matmul !l (swap2 (slice2 !l (Some (j, j + 1)) None))
      in
      let col = E.sub (slice2 low None (Some (j, j + 1))) proj in
      let ljj = E.sqrt (slice2 col (Some (j, j + 1)) (Some (0, 1))) in
      let tail =
        if n - j - 1 = 0 then col
        else E.div (slice2 col (Some (j + 1, n)) (Some (0, 1))) ljj
      in
      (* The finished column: zeros above the diagonal, the pivot, the scaled
         tail below. *)
      let new_col =
        cat2
          ((if j > 0 then [ scalar2 a batch j 1 0.0 ] else [])
          @ [ ljj ]
          @ if n - j - 1 > 0 then [ tail ] else [])
      in
      l := (if j = 0 then new_col else F.Op.cat ~dim:(-1) !l [ new_col ])
    done;
    if upper then swap2 !l else !l
