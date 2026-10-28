(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module Prim = Nx.Prim

(* Matrices *)

let lit x v = Nx.scalar_like x (D.of_float (Nx.dtype x) v)
let adjoint x = Nx.conjugate (Nx.matrix_transpose x)

(* The identity of [n] rows in [dt], whole on each of [x]'s devices. *)
let eye_beside x dt n = Nx.eye_at (Nx.Placement.uncut (Nx.placement x)) dt n

(* [x]'s real part in its own dtype. *)
let real_part x = Nx.mul (Nx.add x (Nx.conjugate x)) (lit x 0.5)

(* [d] as the diagonal of a stack of matrices. *)
let diag_matrix d =
  let n = Nx.dim (-1) d and r = Nx.ndim d in
  Nx.mul (eye_beside d (Nx.dtype d) n) (Nx.unsqueeze ~axes:[ r - 1 ] d)

(* The Hermitian matrix that the strict lower triangle of [x] and the real parts
   of its diagonal name. *)
let hermitian x =
  let low = Nx.tril ~k:(-1) x in
  Nx.add (Nx.add low (adjoint low)) (diag_matrix (real_part (Nx.diagonal x)))

let zero_where c x = Nx.where c (Nx.zeros_like x) x

(* The rows [rows] and columns [cols] of [x]'s matrices. *)
let block x (r0, r1) (c0, c1) =
  Nx.slice
    (List.init (Nx.ndim x - 2) (fun _ -> Nx.A)
    @ [ Nx.R (r0, r1); Nx.R (c0, c1) ])
    x

(* [x] with zeros around its matrices, [rows] and [cols] before and after. *)
let padded x rows cols =
  let r = Nx.ndim x in
  Nx.pad
    (Array.init r (fun a ->
         if a = r - 2 then rows else if a = r - 1 then cols else (0, 0)))
    (D.zero (Nx.dtype x))
    x

(* The matrix of [1 / (d_j - d_i)] off the diagonal and [0] on it, for a stack
   of vectors [d]: a spectral derivative's coefficients, infinite where two of
   [d] are equal. *)
let gaps d =
  let r = Nx.ndim d in
  let eye = eye_beside d (Nx.dtype d) (Nx.dim (-1) d) in
  let diffs =
    Nx.sub (Nx.unsqueeze ~axes:[ r - 1 ] d) (Nx.unsqueeze ~axes:[ r ] d)
  in
  Nx.sub (Nx.recip (Nx.add diffs eye)) eye

let solve ?transpose triangle a b = Nx.solve_triangular ~triangle ?transpose a b

(* Routines *)

(* [a = L Lᴴ], read from [a]'s lower triangle as [H], gives [dL = L Φ(L⁻¹ dH
   L⁻ᴴ)], [Φ] the lower triangle less half the diagonal. Under [Upper] the
   factor is [U = Lᴴ]. *)
let cholesky (triangle : Nx_kernel.triangle) y da =
  let l = match triangle with Lower -> y | Upper -> adjoint y in
  let left b = solve Lower l b in
  let m = adjoint (left (adjoint (left (hermitian da)))) in
  let d = Nx.diagonal m in
  let phi = Nx.sub (Nx.tril m) (diag_matrix (Nx.mul d (lit d 0.5))) in
  let dl = Nx.matmul l phi in
  match triangle with Lower -> dl | Upper -> adjoint dl

(* [P a = L U], packed, [a] of [m] rows and [n] columns, [k = min m n]: with [L]
   completed to an [m × m] unit lower matrix and [U] to an [n × n] upper one by
   an identity block, and [X = L⁻¹ P da U⁻¹], [dL = L tril₋₁(X)] and [dU =
   triu(X) U], packed as the factors are. *)
let lu packed perm da =
  let m = Nx.dim (-2) packed and n = Nx.dim (-1) packed in
  let k = Int.min m n and dt = Nx.dtype packed in
  let rows =
    Nx.broadcast_to (Nx.shape da) (Nx.unsqueeze ~axes:[ Nx.ndim perm ] perm)
  in
  let pda = Nx.take_along_axis ~axis:(Nx.ndim da - 2) rows da in
  let l =
    Nx.add
      (padded (Nx.tril ~k:(-1) (block packed (0, m) (0, k))) (0, 0) (0, m - k))
      (eye_beside packed dt m)
  in
  let u =
    Nx.add
      (padded (Nx.triu (block packed (0, k) (0, n))) (0, n - k) (0, 0))
      (padded (eye_beside packed dt (n - k)) (k, 0) (k, 0))
  in
  let y = Nx.solve_triangular ~triangle:Lower ~unit_diag:true l pda in
  let x =
    Nx.matrix_transpose
      (solve Lower (Nx.matrix_transpose u) (Nx.matrix_transpose y))
  in
  Nx.add (Nx.matmul l (Nx.tril ~k:(-1) x)) (Nx.matmul (Nx.triu x) u)

(* [a = Q R], [R] square and upper triangular with a real diagonal: with [X = Qᴴ
   da R⁻¹] and [Ω] the skew-Hermitian matrix of [X]'s strict lower triangle and
   the imaginary part of its diagonal, [dR = (X - Ω) R] and [dQ = da R⁻¹ - Q (X
   - Ω)]. *)
let qr_square q r da =
  let da_rinv = adjoint (solve ~transpose:Adjoint Upper r (adjoint da)) in
  let x = Nx.matmul (adjoint q) da_rinv in
  let low = Nx.tril ~k:(-1) x and d = Nx.diagonal x in
  let omega =
    Nx.add (Nx.sub low (adjoint low)) (diag_matrix (Nx.sub d (real_part d)))
  in
  let upper = Nx.sub x omega in
  (Nx.sub da_rinv (Nx.matmul q upper), Nx.matmul upper r)

(* A wide [a = [a₁ a₂]] has [R = [R₁ R₂]], [a₁ = Q R₁] square, so [dR₂ = Qᴴ (da₂
   - dQ R₂)]. *)
let qr name (factors : Nx_kernel.factors) q r da =
  let m = Nx.dim (-2) da and n = Nx.dim (-1) da in
  if m >= n then begin
    if factors = Complete && m > n then
      invalid_arg
        (name
       ^ ": the tangent of a complete QR factorisation of a tall matrix has no \
          definition");
    qr_square q r da
  end
  else
    let cols lo hi x = block x (0, m) (lo, hi) in
    let r1 = cols 0 m r and r2 = cols m n r in
    let dq, dr1 = qr_square q r1 (cols 0 m da) in
    let dr2 = Nx.matmul (adjoint q) (Nx.sub (cols m n da) (Nx.matmul dq r2)) in
    (dq, Nx.concatenate ~axis:(Nx.ndim r - 1) [ dr1; dr2 ])

(* [a = U S Vᴴ], thin. With [P = Uᴴ da V], [dS = Re diag P]. *)
let svd_values dtype u vh da =
  Nx.cast dtype
    (Nx.diagonal (Nx.matmul (adjoint u) (Nx.matmul da (adjoint vh))))

(* With [P] as above, the tangent of each right singular vector orthogonal to it
   ([Vᴴ dV] has a zero diagonal), and the left ones carrying the phase that
   keeps [a = U S Vᴴ]: [dU = U (F ∘ (P S + S Pᴴ) + i Im(diag P) S⁻¹)] and [dV =
   V (F ∘ (S P + Pᴴ S))], [F_ij = 1 / (s_j² - s_i²)], plus the parts outside the
   span of [U] or [V] of a tall or wide [a]. *)
let svd name (factors : Nx_kernel.factors) dtype s u vh da =
  let m = Nx.dim (-2) da and n = Nx.dim (-1) da and dt = Nx.dtype da in
  if factors = Complete && m <> n then
    invalid_arg
      (name
     ^ ": the tangent of a complete SVD of a non-square matrix has no \
        definition");
  let v = adjoint vh in
  let s' = Nx.cast dt s in
  let r = Nx.ndim s' in
  let row = Nx.unsqueeze ~axes:[ r - 1 ] s'
  and col = Nx.unsqueeze ~axes:[ r ] s' in
  let p = Nx.matmul (adjoint u) (Nx.matmul da v) in
  let ds = Nx.cast dtype (Nx.diagonal p) in
  let f = gaps (Nx.square s') in
  let inv = zero_where (Nx.equal s' (Nx.zeros_like s')) (Nx.recip s') in
  let phase =
    Nx.mul (Nx.mul (Nx.sub p (adjoint p)) (lit p 0.5)) (diag_matrix inv)
  in
  let ps = Nx.mul p row and sp = Nx.mul col p in
  let du = Nx.matmul u (Nx.add (Nx.mul f (Nx.add ps (adjoint ps))) phase) in
  let dv = Nx.matmul v (Nx.mul f (Nx.add sp (adjoint sp))) in
  let outside basis y =
    Nx.div (Nx.sub y (Nx.matmul basis (Nx.matmul (adjoint basis) y))) row
  in
  let du = if m > n then Nx.add du (outside u (Nx.matmul da v)) else du in
  let dv =
    if n > m then Nx.add dv (outside v (Nx.matmul (adjoint da) u)) else dv
  in
  (ds, du, adjoint dv)

(* [H = V Λ Vᴴ] read from [a]'s lower triangle. With [P = Vᴴ dH V], [dΛ = Re
   diag P] and, the tangent of each eigenvector orthogonal to it, [dV = V (F ∘
   P)], [F_ij = 1 / (λ_j - λ_i)]. *)
let eigh_values dtype v da =
  Nx.cast dtype
    (Nx.diagonal (Nx.matmul (adjoint v) (Nx.matmul (hermitian da) v)))

let eigh dtype w v da =
  let p = Nx.matmul (adjoint v) (Nx.matmul (hermitian da) v) in
  ( Nx.cast dtype (Nx.diagonal p),
    Nx.matmul v (Nx.mul (gaps (Nx.cast (Nx.dtype v) w)) p) )

(* [a V = V Λ]. With [C = V⁻¹ da V], [dΛ = diag C] and [dV = V (F ∘ C)], [F_ij =
   1 / (λ_j - λ_i)], less each column's component along itself, so that each
   eigenvector keeps its unit norm and its tangent is orthogonal to it. *)
let eig_values v da =
  let da = Nx.cast (Nx.dtype v) da in
  Nx.diagonal (Nx.matmul (Nx.inv v) (Nx.matmul da v))

let eig w v da =
  let da = Nx.cast (Nx.dtype v) da in
  let c = Nx.matmul (Nx.inv v) (Nx.matmul da v) in
  let dv = Nx.matmul v (Nx.mul (gaps w) c) in
  let along =
    Nx.sum ~axes:[ Nx.ndim v - 2 ] ~keepdims:true (Nx.mul (Nx.conjugate v) dv)
  in
  (Nx.diagonal c, Nx.sub dv (Nx.mul v along))

(* [op(a) x = b], so [op(a) dx = db - op(da) x], [da] restricted to the triangle
   the solve reads, less its diagonal under [unit_diagonal]; [op] is the
   conjugate transpose under [adjoint]. *)
let solve_triangular ~by (triangle : Nx_kernel.triangle) ~adjoint:adj
    ~unit_diagonal a x da db =
  let rhs =
    match da with
    | None -> Option.get db
    | Some da -> (
        let read =
          match triangle with Lower -> Nx.tril da | Upper -> Nx.triu da
        in
        let read =
          if unit_diagonal then Nx.sub read (diag_matrix (Nx.diagonal read))
          else read
        in
        let op = if adj then adjoint read else read in
        let p = Nx.matmul op x in
        match db with None -> Nx.neg p | Some db -> Nx.sub db p)
  in
  Prim.eval ~by
    (Linalg
       (Solve_triangular { triangle; adjoint = adj; unit_diagonal; a; b = rhs }))
