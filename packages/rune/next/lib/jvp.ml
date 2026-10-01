(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

type t = {
  entry : string;
  slots : Linear.tape option;
  rerun : rerun option;
  id : unit ref;  (** The installation's identity, which its duals name. *)
}

(* A rerun's installation adopts the duals of its parent, and of the parent's
   ancestors, that its function uses: each becomes a dual of its own, whose
   tangent is a slot nothing feeds, on first use. *)
and rerun = { parent : t; mutable captures : capture list }

(* A dual of an ancestor and the slot that stands for its tangent. *)
and capture = Capture : ('a, 'b) Nx.t * ('a, 'b) Nx.t -> capture

let create ?slots entry = { entry; slots; rerun = None; id = ref () }

type (_, _) Repr.node +=
  | Dual : {
      owner : t;
      primal : ('a, 'b) Nx.t;
      tangent : ('a, 'b) Nx.t;
    }
      -> ('a, 'b) Repr.node

let dual owner primal tangent =
  Repr.Traced.v ~context:(Repr.context primal) (Nx.placement primal)
    (Nx.dtype primal) (Nx.shape primal)
    (Dual { owner; primal; tangent })

let rec adopts i owner =
  match i.rerun with
  | Some r -> r.parent.id == owner.id || adopts r.parent owner
  | None -> false

let rec captured : type a b. capture list -> (a, b) Nx.t -> (a, b) Nx.t option =
 fun captures x ->
  match captures with
  | Capture (d, s) :: rest -> (
      match Nx_dtype.equal_witness (Nx.dtype d) (Nx.dtype x) with
      | Some Type.Equal when d == x -> Some s
      | _ -> captured rest x)
  | [] -> None

(* [capture i x primal] is the slot of [i] that stands for the tangent of [x], a
   dual of an ancestor of [i]. *)
let capture i x primal =
  match (i.rerun, i.slots) with
  | Some r, Some tape -> (
      match captured r.captures x with
      | Some s -> s
      | None ->
          let s = Linear.input tape primal in
          r.captures <- r.captures @ [ Capture (x, s) ];
          s)
  | _ -> assert false (* Only a rerun's installation adopts. *)

let own (type a b) i (x : (a, b) Nx.t) : ((a, b) Nx.t * (a, b) Nx.t) option =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Dual { owner; primal; tangent } ->
          if owner.id == i.id then Some (primal, tangent)
          else if adopts i owner then Some (primal, capture i x primal)
          else None
      | _ -> None)
  | Host _ | Placed _ -> None

let owns i x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Dual { owner; _ } -> owner.id == i.id || adopts i owner
      | _ -> false)
  | Host _ | Placed _ -> false

let split i x =
  match own i x with Some (p, dx) -> (p, Some dx) | None -> (x, None)

let primal i x = match own i x with Some (p, _) -> p | None -> x

(* [unwrap i x] is the primal and tangent of [x], an operand through which [i]
   claims the operation. *)
let unwrap i x =
  match own i x with
  | Some d -> d
  | None -> assert false (* [i] claims the operation through [x]. *)

(* Coefficients *)

let unary k x = eval (Unary (k, x))
let binary k a b = eval (Binary (k, a, b))
let mul a b = binary Mul a b
let scalar x v = Nx.full (Nx.dtype x) [||] (Nx_dtype.of_float (Nx.dtype x) v)
let one x = Nx_dtype.one (Nx.dtype x)

(* The real part of [x] in [x]'s dtype. *)
let real_part x =
  if Nx_dtype.is_complex (Nx.dtype x) then
    Nx.cast (Nx.dtype x) (Nx.cast Nx.float64 x)
  else x

let adjoint x = Nx.conjugate (Nx.matrix_transpose x)

(* [d] as the diagonal of a stack of matrices. *)
let diag_matrix d =
  Nx.mul (Nx.eye (Nx.dtype d) (Nx.dim (-1) d)) (Nx.unsqueeze ~axes:[ -2 ] d)

(* The Hermitian matrix that the strict lower triangle of [x] and the real part
   of its diagonal name. *)
let hermitian x =
  let low = Nx.tril ~k:(-1) x in
  Nx.add (Nx.add low (adjoint low)) (diag_matrix (real_part (Nx.diagonal x)))

(* [x] with the reduced axes kept as ones, of a reduction of [x] to [y]. *)
let kept ~axes x y =
  let shape = Array.copy (Nx.shape x) in
  Array.iter (fun a -> shape.(a) <- 1) axes;
  Nx.reshape shape y

(* Ties share the derivative of a maximum or minimum equally. *)
let extrema' (type a b) ~axes (x : (a, b) Nx.t) y : (a, b) Nx.t =
  let mask = Nx.equal x (kept ~axes x y) in
  let share (type c) (dt : (float, c) Nx.dtype) =
    let mask = Nx.cast dt mask in
    Nx.cast (Nx.dtype x)
      (Nx.div mask (Nx.sum ~axes:(Array.to_list axes) ~keepdims:true mask))
  in
  match Nx.dtype x with Float64 -> share Nx.float64 | _ -> share Nx.float32

(* [shifted ~axis d fill x] is [x] moved [d] places along [axis], [d] elements
   of [fill] first and its last [d] elements dropped. *)
let shifted ~axis d fill x =
  let shape = Nx.shape x in
  let pads =
    Array.mapi (fun a _ -> if a = axis then (d, 0) else (0, 0)) shape
  in
  Nx.shrink (Array.map (fun n -> (0, n)) shape) (Nx.pad pads fill x)

(* [others ~axes x] is, at each element of [x], the product of the other
   elements over [axes]: the product before it in their order times the product
   after it. No division, so it is exact at zeros and where the product
   underflows, at every order. *)
let others ~axes x =
  let shape = Nx.shape x in
  let kept =
    List.filter (fun a -> not (Array.mem a axes)) (List.init (Nx.ndim x) Fun.id)
  in
  let order = Array.of_list (kept @ Array.to_list axes) in
  let moved = Nx.contiguous (Nx.transpose ~axes:(Array.to_list order) x) in
  let k = List.length kept in
  let n = Array.fold_left (fun n a -> n * shape.(a)) 1 axes in
  let flat =
    Nx.reshape (Array.of_list (List.map (Array.get shape) kept @ [ n ])) moved
  in
  let exclusive x = shifted ~axis:k 1 (one x) (Nx.cumprod ~axis:k x) in
  let rev x = Nx.flip ~axes:[ k ] x in
  let o = Nx.mul (exclusive flat) (rev (exclusive (rev flat))) in
  let inverse = Array.make (Array.length order) 0 in
  Array.iteri (fun i a -> inverse.(a) <- i) order;
  Nx.transpose ~axes:(Array.to_list inverse) (Nx.reshape (Nx.shape moved) o)

(* The position along [axis] of the element each running extremum [y] takes: the
   last position where the extremum changed or became NaN, so ties keep the
   first and a NaN extremum the NaN element. *)
let running_arg ~axis y =
  let shape = Nx.shape y in
  let n = shape.(axis) in
  let along = Array.mapi (fun a _ -> if a = axis then n else 1) shape in
  let iota =
    Nx.broadcast_to shape (Nx.reshape along (Nx.arange Nx.int64 0 n 1))
  in
  let before = shifted ~axis 1 (Nx_dtype.zero (Nx.dtype y)) y in
  let changed =
    Nx.bitwise_and (Nx.not_equal y before)
      (Nx.logical_not (Nx.bitwise_and (Nx.isnan y) (Nx.isnan before)))
  in
  Nx.cummax ~axis (Nx.where changed iota (Nx.zeros_like iota))

(* [linear_scan ~axis a b] is [r] with [r_k = a_k r_(k-1) + b_k] along [axis]
   and [r_(-1) = 0], composed by doubling in about [log2 n] rounds of products
   and sums: no division, so it is exact where [a] has zeros, and it is linear
   in [b]. *)
let linear_scan ~axis a b =
  let n = (Nx.shape a).(axis) in
  let zero = Nx_dtype.zero (Nx.dtype b) in
  let rec go d a b =
    if
      (d >= n)
      [@mutate
        off "a round at d = n shifts everything out and leaves b unchanged"]
    then b
    else
      go (2 * d)
        (Nx.mul a (shifted ~axis d (one a) a))
        (Nx.add (Nx.mul a (shifted ~axis d zero b)) b)
  in
  go 1 a b

(* On complex dtypes [sign z = z / |z|] turns with [z]: its derivative along [v]
   is [i s Im (conj s v) / |z|], zero at the origin. *)
let sign_push z s v =
  let m = Nx.abs z in
  let origin = Nx.equal m (Nx.zeros_like m) in
  let inv =
    Nx.where origin (Nx.zeros_like m)
      (Nx.recip (Nx.where origin (Nx.ones_like m) m))
  in
  let w = Nx.mul (Nx.conjugate s) v in
  Nx.mul (Nx.mul s (Nx.sub w (real_part w))) inv

(* Rules *)

(* [terms da db] is the sum of the terms that exist: an operand with no tangent
   adds none, so its coefficient, which may be infinite or NaN where the operand
   is constant, never meets a zero. *)
let terms da db =
  match (da, db) with
  | Some da, Some db -> binary Add da db
  | Some d, None | None, Some d -> d
  | None, None -> assert false (* A rule runs for an operand with a tangent. *)

let unary_tangent k x y dx =
  let coef c = mul dx c in
  match[@warning "@4@8"] (k : Nx_backend.unary) with
  | Neg -> unary Neg dx
  | Recip -> coef (Nx.neg (Nx.recip (Nx.mul x x)))
  | Sqrt -> coef (Nx.recip (Nx.mul (scalar y 2.) y))
  | Exp -> coef y
  | Log -> binary Fdiv dx x
  | Sin -> coef (Nx.cos x)
  | Cos -> coef (Nx.neg (Nx.sin x))
  | Tan -> coef (Nx.recip (Nx.square (Nx.cos x)))
  | Asin -> coef (Nx.recip (Nx.sqrt (Nx.rsub_s (one x) (Nx.mul x x))))
  | Acos -> coef (Nx.neg (Nx.recip (Nx.sqrt (Nx.rsub_s (one x) (Nx.mul x x)))))
  | Atan -> coef (Nx.recip (Nx.add_s (Nx.mul x x) (one x)))
  | Sinh -> coef (Nx.cosh x)
  | Cosh -> coef (Nx.sinh x)
  | Tanh -> coef (Nx.rsub_s (one y) (Nx.mul y y))
  | Erf ->
      coef
        (Nx.mul
           (scalar x 1.12837916709551257390)
           (Nx.exp (Nx.neg (Nx.mul x x))))
  | Abs -> real_part (mul dx (Nx.conjugate (Nx.sign x)))
  | Sign -> sign_push x y dx
  | Trunc | Ceil | Floor | Round -> assert false (* A plain result. *)

(* The tangent of a selection of [a] where [first] holds and [b] elsewhere. *)
let selected y first da db =
  let mask = Nx.cast (Nx.dtype y) first in
  terms
    (Option.map (fun da -> mul da mask) da)
    (Option.map (fun db -> mul db (Nx.rsub_s (one y) mask)) db)

(* [zero_where c x] is [x] with zeros where [c] holds. *)
let zero_where c x = Nx.where c (Nx.zeros_like x) x

let binary_tangent k a b y da db =
  let term f = Option.map f in
  match[@warning "@4@8"] (k : Nx_backend.binary) with
  | Add -> terms da db
  | Sub -> terms da (term (unary Neg) db)
  | Mul -> terms (term (fun da -> mul da b) da) (term (mul a) db)
  | Fdiv ->
      terms
        (term (fun da -> binary Fdiv da b) da)
        (term (fun db -> mul db (unary Neg (binary Fdiv y b))) db)
  | Pow ->
      (* 0 ** b is constant along b wherever it is defined, and a ** 0 along a,
         whose coefficient b a ** (b - 1) is 0 times infinity only at a = 0.
         Masking only there keeps the coefficient's own derivative, 1 / a along
         b at b = 0, everywhere else. *)
      let zero x = Nx.equal x (Nx.zeros_like x) in
      let both = Nx.bitwise_and (zero a) (zero b) in
      terms
        (term
           (fun da ->
             mul da (zero_where both (Nx.mul b (Nx.pow a (Nx.sub_s b (one b))))))
           da)
        (term (fun db -> mul db (zero_where (zero a) (Nx.mul y (Nx.log a)))) db)
  | Maximum -> selected y (Nx.bitwise_or (Nx.less b a) (Nx.isnan a)) da db
  | Minimum -> selected y (Nx.bitwise_or (Nx.less a b) (Nx.isnan a)) da db
  | Atan2 ->
      let denom = Nx.add (Nx.mul a a) (Nx.mul b b) in
      terms
        (term (fun da -> mul da (Nx.div b denom)) da)
        (term (fun db -> mul db (Nx.neg (Nx.div a denom))) db)
  | Mod ->
      terms da (term (fun db -> mul db (Nx.neg (Nx.trunc (Nx.div a b)))) db)
  | Idiv | And | Or | Xor -> assert false (* A plain result. *)

(* [zeros_or dx x] is [x]'s tangent, or zeros like it if it has none. *)
let zeros_or dx x = match dx with Some dx -> dx | None -> Nx.zeros_like x

(* [viewable m dx] is [dx], copied to C order if [m] is a reshape that its view
   cannot take: a tangent need not share its primal's strides. *)
let viewable m dx =
  match[@warning "@4@8"] m with
  | Reshape _ -> if Nx.is_c_contiguous dx then dx else Nx.contiguous dx
  | Expand _ | Permute _ | Shrink _ | Flip _ | Window _ -> dx

(* Linear algebra. Each rule's tangent is linear in the operand's tangent:
   coefficients come from the primals, the tangent meets only products, triangle
   masks and solves in which it is the right-hand side. *)

(* The factor reads H = [hermitian x] = L Lᴴ, so dL = L Φ(L^-1 dH L^-H), with Φ
   the lower triangle less half the diagonal. Under [upper] the factor is U =
   Lᴴ. *)
let cholesky' ~upper y dx =
  let l = if upper then adjoint y else y in
  let dh = hermitian dx in
  let left b =
    Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:false l b
  in
  let m = adjoint (left (adjoint (left dh))) in
  let phi =
    let d = Nx.diagonal m in
    Nx.sub (Nx.tril m)
      (diag_matrix (Nx.div_s d (Nx_dtype.of_float (Nx.dtype d) 2.)))
  in
  let dl = Nx.matmul l phi in
  if upper then adjoint dl else dl

(* op(A) X = B, so op(A) dX = dB - op(dA) X, with dA restricted to the triangle
   the solve reads, less its diagonal under [unit_diag]; op is the conjugate
   transpose under [transpose]. *)
let solve' ~upper ~transpose ~unit_diag a b x da db =
  let rhs =
    match da with
    | None -> zeros_or db b
    | Some da -> (
        let used =
          let tri = if upper then Nx.triu da else Nx.tril da in
          if unit_diag then Nx.sub tri (diag_matrix (Nx.diagonal tri)) else tri
        in
        let op_da = if transpose then adjoint used else used in
        let vector = Nx.ndim x = Nx.ndim a - 1 in
        let x2 = if vector then Nx.unsqueeze ~axes:[ -1 ] x else x in
        let p = Nx.matmul op_da x2 in
        let p = if vector then Nx.reshape (Nx.shape b) p else p in
        match db with None -> Nx.neg p | Some db -> Nx.sub db p)
  in
  eval (Solve_triangular { upper; transpose; unit_diag; a; b = rhs })

(* [block x rows cols] is the block of the last two axes of [x] that the ranges
   [rows] and [cols] select; [padded x rows cols] is [x] with zeros around its
   last two axes, [rows] and [cols] before and after. *)
let block x rows cols =
  let r = Nx.ndim x in
  Nx.shrink
    (Array.mapi
       (fun a d ->
         if a = r - 2 then rows else if a = r - 1 then cols else (0, d))
       (Nx.shape x))
    x

let padded x rows cols =
  let r = Nx.ndim x in
  Nx.pad
    (Array.init r (fun a ->
         if a = r - 2 then rows else if a = r - 1 then cols else (0, 0)))
    (Nx_dtype.zero (Nx.dtype x))
    x

(* P A = L U, packed, for A of [m] rows, [n] columns and [k = min m n]: with L
   completed to an [m × m] unit lower matrix, U to an [n × n] upper one by an
   identity block, and X = L^-1 P dA U^-1, dL = L tril_-1(X) and dU = triu(X) U,
   packed as the factors are. *)
let lu' packed perm da =
  let m = Nx.dim (-2) packed and n = Nx.dim (-1) packed in
  let k = Int.min m n and dt = Nx.dtype packed in
  let pda =
    Nx.take_along_axis ~axis:(-2)
      ~indices:(Nx.broadcast_to (Nx.shape da) (Nx.unsqueeze ~axes:[ -1 ] perm))
      da
  in
  let l =
    Nx.add
      (padded (Nx.tril ~k:(-1) (block packed (0, m) (0, k))) (0, 0) (0, m - k))
      (Nx.eye dt m)
  in
  let u =
    Nx.add
      (padded (Nx.triu (block packed (0, k) (0, n))) (0, n - k) (0, 0))
      (padded (Nx.eye dt (n - k)) (k, 0) (k, 0))
  in
  let y = Nx.solve_triangular ~upper:false ~unit_diag:true l pda in
  let x =
    Nx.matrix_transpose
      (Nx.solve_triangular ~upper:false (Nx.matrix_transpose u)
         (Nx.matrix_transpose y))
  in
  Nx.add (Nx.matmul l (Nx.tril ~k:(-1) x)) (Nx.matmul (Nx.triu x) u)

(* The matrix of [1 / (d_j - d_i)] off the diagonal and [0] on it, for a stack
   of vectors [d]: the coefficients of a spectral derivative, infinite where two
   of [d] are equal. *)
let gaps d =
  let eye = Nx.eye (Nx.dtype d) (Nx.dim (-1) d) in
  let diffs =
    Nx.sub (Nx.unsqueeze ~axes:[ -2 ] d) (Nx.unsqueeze ~axes:[ -1 ] d)
  in
  Nx.sub (Nx.recip (Nx.add diffs eye)) eye

(* A = U S Vᴴ, thin. With P = Uᴴ dA V, dS = Re diag P, and the tangent of each
   right singular vector orthogonal to it (Vᴴ dV has a zero diagonal), the left
   vectors carrying the phase that keeps A = U S Vᴴ: dU = U (F ∘ (P S + S Pᴴ) +
   i Im(diag P) S^-1) and dV = V (F ∘ (S P + Pᴴ S)), F_ij = 1 / (s_j² - s_i²),
   plus the parts outside the span of U or V of a tall or wide A. *)
let svd' i ~full_matrices x u s vt dx =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x and dt = Nx.dtype x in
  if full_matrices && m <> n then
    invalid_arg
      (i.entry
     ^ ": the tangent of a complete SVD of a non-square matrix has no \
        definition");
  let v = adjoint vt in
  let s = Nx.cast dt s in
  let row = Nx.unsqueeze ~axes:[ -2 ] s and col = Nx.unsqueeze ~axes:[ -1 ] s in
  let p = Nx.matmul (adjoint u) (Nx.matmul dx v) in
  let ds = Nx.cast Nx.float64 (Nx.diagonal p) in
  let f = gaps (Nx.square s) in
  let inv = zero_where (Nx.equal s (Nx.zeros_like s)) (Nx.recip s) in
  let phase =
    Nx.mul
      (Nx.mul_s (Nx.sub p (adjoint p)) (Nx_dtype.of_float dt 0.5))
      (diag_matrix inv)
  in
  let ps = Nx.mul p row and sp = Nx.mul col p in
  let du = Nx.matmul u (Nx.add (Nx.mul f (Nx.add ps (adjoint ps))) phase) in
  let dv = Nx.matmul v (Nx.mul f (Nx.add sp (adjoint sp))) in
  let outside basis y =
    Nx.div (Nx.sub y (Nx.matmul basis (Nx.matmul (adjoint basis) y))) row
  in
  let du =
    if (m > n) [@mutate off "on a square matrix U Uᴴ = I and the term vanishes"]
    then Nx.add du (outside u (Nx.matmul dx v))
    else du
  in
  let dv =
    if (n > m) [@mutate off "on a square matrix V Vᴴ = I and the term vanishes"]
    then Nx.add dv (outside v (Nx.matmul (adjoint dx) u))
    else dv
  in
  (du, ds, adjoint dv)

(* [hermitian x] = Q Λ Qᴴ. With P = Qᴴ dA Q, dΛ = Re diag P and, the tangent of
   each eigenvector orthogonal to it (Qᴴ dQ has a zero diagonal), dQ = Q (F ∘
   P), F_ij = 1 / (λ_j - λ_i). *)
let eigh' w q dx =
  let p = Nx.matmul (adjoint q) (Nx.matmul (hermitian dx) q) in
  ( Nx.cast Nx.float64 (Nx.diagonal p),
    Nx.matmul q (Nx.mul (gaps (Nx.cast (Nx.dtype q) w)) p) )

(* A V = V Λ. With C = V^-1 dA V, dΛ = diag C and dV = V (F ∘ C), F_ij = 1 /
   (λ_j - λ_i), less each column's component along itself, so that each
   eigenvector keeps its unit norm and its tangent is orthogonal to it (Vᴴ dV
   has a zero diagonal). *)
let eig' values v dx =
  let dx = Nx.cast (Nx.dtype v) dx in
  let c = Nx.matmul (Nx.inv v) (Nx.matmul dx v) in
  let dv = Nx.matmul v (Nx.mul (gaps values) c) in
  let along = Nx.sum ~axes:[ -2 ] ~keepdims:true (Nx.mul (Nx.conjugate v) dv) in
  (Nx.diagonal c, Nx.sub dv (Nx.mul v along))

(* A = Q R with R square, upper triangular with a real diagonal. With X = Qᴴ dA
   R^-1 and Ω the skew-Hermitian matrix of X's strict lower triangle and the
   imaginary part of its diagonal, dR = (X - Ω) R and dQ = dA R^-1 - Q (X -
   Ω). *)
let qr_square q r da =
  (* dA R^-1 = (R^-H dAᴴ)ᴴ. *)
  let da_rinv =
    adjoint
      (Nx.solve_triangular ~upper:true ~transpose:true ~unit_diag:false r
         (adjoint da))
  in
  let x = Nx.matmul (adjoint q) da_rinv in
  let low = Nx.tril ~k:(-1) x and d = Nx.diagonal x in
  let omega =
    Nx.add (Nx.sub low (adjoint low)) (diag_matrix (Nx.sub d (real_part d)))
  in
  let upper = Nx.sub x omega in
  (Nx.sub da_rinv (Nx.matmul q upper), Nx.matmul upper r)

(* A tall or square A has R square. A wide A = [A₁ A₂] has R = [R₁ R₂], with A₁
   = Q R₁ square, so dR₂ = Qᴴ (dA₂ - dQ R₂). *)
let qr' i ~reduced x q r dx =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x in
  if
    (m >= n)
    [@mutate off "a square matrix through the wide branch has an empty R₂"]
  then begin
    if (not reduced) && m > n then
      invalid_arg
        (i.entry
       ^ ": the tangent of a complete QR factorisation of a tall matrix has no \
          definition");
    qr_square q r dx
  end
  else
    let cols lo hi a = block a (0, m) (lo, hi) in
    let r1 = cols 0 m r and r2 = cols m n r in
    let dq, dr1 = qr_square q r1 (cols 0 m dx) in
    let dr2 = Nx.matmul (adjoint q) (Nx.sub (cols m n dx) (Nx.matmul dq r2)) in
    (dq, Nx.concatenate ~axis:(-1) [ dr1; dr2 ])

(* The interpreter *)

(* [run i op] is [op], one of whose operands is a dual of [i]. A rule forwards
   [op] on the primals and pairs its result with the tangent. *)
let run : type r. t -> r Nx.Op.t -> r =
 fun i op ->
  let linear x f =
    let x, dx = unwrap i x in
    dual i (f x) (f dx)
  in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> (
      let x, dx = unwrap i x in
      let y = unary k x in
      match[@warning "@4@8"] (k : Nx_backend.unary) with
      | Sign when not (Nx_dtype.is_complex (Nx.dtype x)) -> y
      | Trunc | Ceil | Floor | Round -> y
      | Neg | Recip | Abs | Sqrt | Sign | Exp | Log | Sin | Cos | Tan | Asin
      | Acos | Atan | Sinh | Cosh | Tanh | Erf ->
          dual i y (unary_tangent k x y dx))
  | Binary (k, a, b) -> (
      let a, da = split i a and b, db = split i b in
      let y = binary k a b in
      match[@warning "@4@8"] (k : Nx_backend.binary) with
      | Idiv | And | Or | Xor -> y
      | Add | Sub | Mul | Fdiv | Mod | Pow | Atan2 | Maximum | Minimum ->
          dual i y (binary_tangent k a b y da db))
  | Compare (k, a, b) -> eval (Compare (k, primal i a, primal i b))
  | Where (c, a, b) ->
      let a, da = split i a and b, db = split i b in
      dual i
        (eval (Where (c, a, b)))
        (eval (Where (c, zeros_or da a, zeros_or db b)))
  | Reduce (k, axes, x) -> (
      let x, dx = unwrap i x in
      let y = eval (Reduce (k, axes, x)) in
      match[@warning "@4@8"] (k : Nx_backend.reduce) with
      | Sum -> dual i y (eval (Reduce (Sum, axes, dx)))
      | Max | Min ->
          dual i y (eval (Reduce (Sum, axes, mul dx (extrema' ~axes x y))))
      | Prod -> dual i y (eval (Reduce (Sum, axes, mul dx (others ~axes x)))))
  | Scan (k, axis, x) -> (
      let x, dx = unwrap i x in
      let y = eval (Scan (k, axis, x)) in
      match[@warning "@4@8"] (k : Nx_backend.reduce) with
      | Sum -> dual i y (eval (Scan (Sum, axis, dx)))
      | Prod ->
          (* dy_k = x_k dy_(k-1) + y_(k-1) dx_k *)
          let before = shifted ~axis 1 (one x) y in
          dual i y (linear_scan ~axis x (mul dx before))
      | Max | Min -> dual i y (eval (Gather (axis, running_arg ~axis y, dx))))
  | Arg_reduce (k, axis, x) -> eval (Arg_reduce (k, axis, primal i x))
  | Sort { descending; axis; x } ->
      let x, dx = unwrap i x in
      let indices = eval (Argsort { descending; axis; x }) in
      dual i
        (eval (Sort { descending; axis; x }))
        (eval (Gather (axis, indices, dx)))
  | Argsort s -> eval (Argsort { s with x = primal i s.x })
  | Pad (padding, v, x) ->
      let x, dx = unwrap i x in
      dual i
        (eval (Pad (padding, v, x)))
        (eval (Pad (padding, Nx_dtype.zero (Nx.dtype x), dx)))
  | Cat (axis, xs) ->
      let xs = List.map (split i) xs in
      dual i
        (eval (Cat (axis, List.map fst xs)))
        (eval (Cat (axis, List.map (fun (x, dx) -> zeros_or dx x) xs)))
  | Convert (Cast, dtype, x) ->
      let x, dx = unwrap i x in
      let y = eval (Convert (Cast, dtype, x)) in
      if Nx_dtype.is_float dtype || Nx_dtype.is_complex dtype then
        dual i y (eval (Convert (Cast, dtype, dx)))
      else y
  | Convert (Bitcast, dtype, x) -> eval (Convert (Bitcast, dtype, primal i x))
  | Threefry _ -> assert false (* Its int32 operands are never duals. *)
  | Gather (axis, indices, x) ->
      linear x (fun x -> eval (Gather (axis, indices, x)))
  | Scatter s ->
      let updates, du = split i s.updates and into, di = split i s.into in
      dual i
        (eval (Scatter { s with updates; into }))
        (eval
           (Scatter
              { s with updates = zeros_or du updates; into = zeros_or di into }))
  | Update (x, starts, v) ->
      let x, dx = split i x and v, dv = split i v in
      dual i
        (eval (Update (x, starts, v)))
        (eval (Update (zeros_or dx x, starts, zeros_or dv v)))
  | Unfold u -> linear u.x (fun x -> eval (Unfold { u with x }))
  | Fold f -> linear f.x (fun x -> eval (Fold { f with x }))
  | Matmul (a, b) ->
      let a, da = split i a and b, db = split i b in
      dual i
        (eval (Matmul (a, b)))
        (terms
           (Option.map (fun da -> eval (Matmul (da, b))) da)
           (Option.map (fun db -> eval (Matmul (a, db))) db))
  | Fft f -> linear f.x (fun x -> eval (Fft { f with x }))
  | Rfft f -> linear f.x (fun x -> eval (Rfft { f with x }))
  | Irfft f -> linear f.x (fun x -> eval (Irfft { f with x }))
  | Contiguous x -> linear x (fun x -> eval (Contiguous x))
  | Cholesky { upper; x } ->
      let x, dx = unwrap i x in
      let y = eval (Cholesky { upper; x }) in
      dual i y (cholesky' ~upper y dx)
  | Qr { reduced; x } ->
      let x, dx = unwrap i x in
      let q, r = eval (Qr { reduced; x }) in
      let dq, dr = qr' i ~reduced x q r dx in
      (dual i q dq, dual i r dr)
  | Lu x ->
      let x, dx = unwrap i x in
      let packed, pivots, perm = eval (Lu x) in
      (dual i packed (lu' packed perm dx), pivots, perm)
  | Svd { full_matrices; x } ->
      let x, dx = unwrap i x in
      let u, s, vt = eval (Svd { full_matrices; x }) in
      let du, ds, dvt = svd' i ~full_matrices x u s vt dx in
      (dual i u du, dual i s ds, dual i vt dvt)
  | Eigh { vectors; x } -> (
      (* The primal is the operation as written: without vectors, the
         factorisation the tangent needs is a second one, whose values may
         differ in rounding. *)
      let x, dx = unwrap i x in
      let w, q = eval (Eigh { vectors = true; x }) in
      let q = Option.get q in
      let dw, dq = eigh' w q dx in
      match vectors with
      | true -> (dual i w dw, Some (dual i q dq))
      | false -> (dual i (fst (eval (Eigh { vectors; x }))) dw, None))
  | Eig { vectors; x } -> (
      (* The tangent pairs the values with vectors by position with those
         without: nx's eig gives them in the same order either way. *)
      let x, dx = unwrap i x in
      let values, v = eval (Eig { vectors = true; x }) in
      let v = Option.get v in
      let dvalues, dv = eig' values v dx in
      match vectors with
      | true -> (dual i values dvalues, Some (dual i v dv))
      | false -> (dual i (fst (eval (Eig { vectors; x }))) dvalues, None))
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      let a, da = split i a and b, db = split i b in
      let x = eval (Solve_triangular { upper; transpose; unit_diag; a; b }) in
      dual i x (solve' ~upper ~transpose ~unit_diag a b x da db)
  | Move (x, m) ->
      let x, dx = unwrap i x in
      dual i (eval (Move (x, m))) (eval (Move (viewable m dx, m)))
  | Place (p, x) -> linear x (fun x -> eval (Place (p, x)))
  | Read { by; x } -> eval (Read { by; x = primal i x })

(* Leaves *)

(* [pick flags l] is the elements of [l] that [flags] marks. *)
let pick flags l =
  List.filter_map
    (fun (f, x) -> if f then Some x else None)
    (List.combine flags l)

(* [duals i flags leaves ts] is [leaves] with each one [flags] marks a dual of
   [i] whose tangent is the next of [ts]. *)
let rec duals i flags leaves ts =
  match (flags, leaves, ts) with
  | true :: flags, Nx.P x :: leaves, t :: ts ->
      Nx.P (dual i x (Nx.unpack (Nx.dtype x) t)) :: duals i flags leaves ts
  | false :: flags, x :: leaves, ts -> x :: duals i flags leaves ts
  | [], [], [] -> []
  | _ -> assert false (* A flag per leaf, a tangent per flag set. *)

(* [seed i tape flags leaves] is [leaves] with each one [flags] marks a dual of
   [i] whose tangent is a fresh input slot of [tape], and those slots. *)
let seed i tape flags leaves =
  let slots =
    List.map (fun (Nx.P x) -> Nx.P (Linear.input tape x)) (pick flags leaves)
  in
  (duals i flags leaves slots, slots)

let owned i = List.map (fun (Nx.P x) -> owns i x)
let primal_leaf i (Nx.P x) = Nx.P (primal i x)

(* [zero i x] is the tangent of [x] when [i] tracks none: zeros, or under
   reverse mode a slot nothing feeds. *)
let zero i x =
  match i.slots with
  | Some tape -> Linear.input tape x
  | None -> Nx.zeros_like x

(* [tangent i x] is [x]'s tangent, or [zero i x] if [i] does not track it. *)
let tangent i x = match split i x with _, Some dx -> dx | x, None -> zero i x

(* The tangents of the leaves [flags] marks. *)
let tangents i flags leaves =
  List.map (fun (Nx.P x) -> Nx.P (tangent i x)) (pick flags leaves)

(* Custom rules *)

let holds_tensor q y = Nx.Ptree.fold q (fun _ _ _ -> true) y false

let holds_own i p args =
  Nx.Ptree.fold p (fun _ x any -> any || owns i x) args false

(* [guarded i ~entry ~loops f] is [f ()], raising at an operation on one of
   [i]'s duals: a rule receives its arguments' primals, and a value [i] tracks
   that it captures would lose its derivative. Unless [loops], a scan [f]
   performs is declined, so that it unrolls into operations the recorder
   sees. *)
let guarded i ~entry ~loops f =
  let run _ =
    invalid_arg
      (entry
     ^ ": the rule uses a value its own differentiation tracks; pass it as an \
        argument")
  in
  let claims op = List.exists (fun (Nx.P x) -> owns i x) (operands op) in
  let call : type r. r Construct.t -> (unit -> r) option = function
    | Scan _ when not loops -> Some (fun () -> raise Scan.Not_staged)
    | Scan _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
    | Lane_count _ | Add _ | Detach _ ->
        None
  in
  Construct.install { op = Some { run; claims }; call } f

(* A custom_jvp call: its result is the answer of the differentiations around
   [i] to the rule at the primals, and its tangent the rule's tangent map at
   [i]'s tangents. *)
let custom_jvp i p q rule args ~again =
  let entry = "Rune.custom_jvp" in
  let a = Nx.Ptree.map p (fun _ x -> primal i x) args in
  let run () = guarded i ~entry ~loops:true (fun () -> rule a) in
  let value, map = if again then Total.discarding run else run () in
  let y =
    Construct.perform
      (Custom (Jvp_rule { p; q; rule; args = a; value = Some value }))
  in
  if Option.is_some i.slots && not (holds_tensor q y) then y
  else
    let da = Nx.Ptree.map p (fun _ x -> tangent i x) args in
    let dy =
      guarded i ~entry ~loops:(Option.is_none i.slots) (fun () -> map da)
    in
    Structure.map2 entry q ~this:"the result" ~that:"the tangent map's result"
      (fun _ y dy -> if Linear.differentiable y then dual i y dy else y)
      y dy

(* A custom_vjp call: its result is the rule's at the primals, and under reverse
   mode a linear call whose transpose is the rule's pullback. *)
let custom_vjp i p q rule args =
  let entry = "Rune.custom_vjp" in
  let a = Nx.Ptree.map p (fun _ x -> primal i x) args in
  let y, pullback = guarded i ~entry ~loops:true (fun () -> rule a) in
  match i.slots with
  | _ when not (holds_tensor q y) -> y
  | None ->
      invalid_arg
        (i.entry
       ^ ": a custom_vjp rule has no forward derivative; give the function a \
          custom_jvp rule")
  | Some tape ->
      let leaves, _ = Nx.Ptree.flatten p args in
      let tracked = owned i leaves in
      let ys, _ = Nx.Ptree.flatten q y in
      let outputs = List.map (fun (Nx.P y) -> Linear.differentiable y) ys in
      let conj (Nx.P c) = Nx.P (Nx.conjugate c) in
      let transpose cts =
        let rec fill outputs ys cts =
          match (outputs, ys, cts) with
          | true :: outputs, _ :: ys, ct :: cts ->
              conj ct :: fill outputs ys cts
          | false :: outputs, Nx.P y :: ys, cts ->
              Nx.P (Nx.zeros_like y) :: fill outputs ys cts
          | _ -> []
        in
        let g = pullback (Nx.Ptree.rebuild q ~like:y (fill outputs ys cts)) in
        ignore
          (Structure.map2 entry p ~this:"the arguments"
             ~that:"the pullback's result"
             (fun _ x _ -> x)
             a g);
        List.map conj (pick tracked (fst (Nx.Ptree.flatten p g)))
      in
      let slots =
        Linear.call tape (tangents i tracked leaves) transpose (pick outputs ys)
      in
      Nx.Ptree.rebuild q ~like:y (duals i outputs ys slots)

let custom : type q. t -> q Construct.rule -> (unit -> q) option =
 fun i r ->
  match r with
  | Jvp_rule { p; q; rule; args; value } ->
      if holds_own i p args then
        Some
          (fun () -> custom_jvp i p q rule args ~again:(Option.is_some value))
      else None
  | Vjp_rule { p; q; rule; args } ->
      if holds_own i p args then Some (fun () -> custom_vjp i p q rule args)
      else None

(* Remat *)

(* The primal of a dual, read from its node. *)
let node_primal (type a b) (x : (a, b) Nx.t) : (a, b) Nx.t =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Dual { primal; _ } -> primal
      | _ -> assert false (* A capture is a dual. *))
  | Host _ | Placed _ -> assert false (* A capture is a dual. *)

let child i tape captures =
  {
    entry = i.entry;
    slots = Some tape;
    rerun = Some { parent = i; captures };
    id = ref ();
  }

let captures_of c = match c.rerun with Some r -> r.captures | None -> []

(* [with_tangents i flags leaves] is the primals of [leaves] followed by the
   tangents of those [flags] marks; [of_tangents i flags n l] is the [n] leaves
   of [l] with the tangents after them. *)
let with_tangents i flags leaves =
  List.map (primal_leaf i) leaves @ tangents i flags leaves

let of_tangents i flags n l =
  let leaves, ts = Scan.split n l in
  duals i flags leaves ts

(* A barrier with duals of [i] among its values or [after] reads the values'
   primals, and with value tangents their tangents too, once the primals of
   [after] exist; a slot needs no barrier. *)
let barrier i values after =
  let flags = owned i values and after = List.map (primal_leaf i) after in
  match i.slots with
  | None ->
      let values = with_tangents i flags values in
      of_tangents i flags (List.length flags)
        (Construct.perform (Barrier { values; after }))
  | Some _ ->
      let primals = List.map (primal_leaf i) values in
      let read = Construct.perform (Barrier { values = primals; after }) in
      duals i flags read (tangents i flags values)

let rec answer : type r. t -> r Construct.t -> (unit -> r) option =
 fun i c ->
  match[@warning "@4@8"] c with
  | Detach x ->
      Option.map (fun (x, _) () -> Construct.perform (Detach x)) (own i x)
  | Custom r -> custom i r
  | Add (t, v) ->
      Option.map (fun (v, _) () -> Construct.perform (Add (t, v))) (own i v)
  | Remat { p; q; f; args; recomputed } -> (
      match i.slots with
      | None -> Some (fun () -> remat_values i p q f args recomputed)
      | Some tape -> Some (fun () -> remat i tape p q f args))
  | Lanes (axis, x) ->
      let lanes x = Construct.perform (Lanes (axis, x)) in
      Option.map (fun (x, dx) () -> dual i (lanes x) (lanes dx)) (own i x)
  | Scan r -> (
      match i.slots with
      | None -> Some (fun () -> scan_values i r)
      | Some tape -> Some (fun () -> scan_slots i tape r))
  | Barrier { values; after } ->
      if List.exists (fun (Nx.P x) -> owns i x) (values @ after) then
        Some (fun () -> barrier i values after)
      else None
  | Lane_index _ | Lane_count _ -> None

(* With value tangents a scan passes on as the scan of its jvp: the carry and
   the rows gain the tangents of those that have one, the outputs those of
   theirs, and the step runs the body under [i] reinstalled. *)
and scan_values : t -> Scan.request -> Scan.result =
 fun i r ->
  let nc = List.length r.req_carry and nx = List.length r.req_xs in
  let rows = owned i r.req_xs in
  Scan.fixpoint (owned i r.req_carry) (fun ~grow carried ->
      let outputs = ref [] in
      let req_step c x =
        let c', y =
          install i (fun () ->
              r.req_step (of_tangents i carried nc c) (of_tangents i rows nx x))
        in
        grow (owned i c');
        outputs := owned i y;
        (with_tangents i carried c', with_tangents i !outputs y)
      in
      let req_carry = with_tangents i carried r.req_carry
      and req_xs = with_tangents i rows r.req_xs in
      let result =
        Construct.perform (Scan { r with req_carry; req_xs; req_step })
      in
      {
        Scan.r_carry = of_tangents i carried nc result.r_carry;
        r_ys = of_tangents i !outputs (List.length !outputs) result.r_ys;
      })

(* Under reverse mode a scan passes on as its primal scan, whose step runs the
   body under a child of [i] on a scratch tape it drops and also outputs the
   carry it received. Once it returns, the tape gains one linear call from the
   tangents of the tracked initial carry, rows and captures to those of the
   dependent final carry and outputs; its transpose is the reversed scan that
   runs each step again at its carry, threading the carry's cotangent and
   summing the captures'. *)
and scan_slots : t -> Linear.tape -> Scan.request -> Scan.result =
 fun i tape r ->
  let nc = List.length r.req_carry and nx = List.length r.req_xs in
  let rows = owned i r.req_xs in
  let step l =
    let c, x = Scan.split nc l in
    let c', y = r.req_step c x in
    c' @ y
  in
  Scan.fixpoint (owned i r.req_carry) (fun ~grow carried ->
      let flags = carried @ rows in
      let outputs = ref [] and captures = ref [] in
      let req_step c x =
        let ch, _, out =
          region i (Linear.create i.entry) [] flags (c @ x) step
        in
        let c', y = Scan.split nc out in
        grow (owned ch c');
        outputs := owned ch y;
        captures := captures_of ch;
        (List.map (primal_leaf ch) c', List.map (primal_leaf ch) y @ c)
      in
      let xs = List.map (primal_leaf i) r.req_xs in
      let result =
        Construct.perform
          (Scan
             {
               r with
               req_carry = List.map (primal_leaf i) r.req_carry;
               req_xs = xs;
               req_step;
             })
      in
      let outputs = !outputs and captures = !captures in
      let ys, carries = Scan.split (List.length outputs) result.r_ys in
      let final = result.r_carry in
      if not (List.mem true carried || List.mem true outputs) then
        { Scan.r_carry = final; r_ys = ys }
      else
        let initial = owned i r.req_carry in
        let inputs =
          tangents i initial r.req_carry
          @ tangents i rows r.req_xs
          @ List.map (fun (Capture (d, _)) -> Nx.P (tangent i d)) captures
        in
        let nk = List.length (List.filter Fun.id carried) in
        let transpose cts =
          let ct_carry, ct_ys = Scan.split nk cts in
          let sum (Nx.P a) (Nx.P b) =
            Nx.P (Nx.add a (Nx.unpack (Nx.dtype a) (Nx.P b)))
          in
          let req_step carry row =
            Total.discarding @@ fun () ->
            let ct_c, ct_caps = Scan.split nk carry in
            let cx, ct_y = Scan.split (nc + nx) row in
            let ct_in, ct_captured =
              pullback i captures flags cx step (carried @ outputs) (ct_c @ ct_y)
            in
            let ct_c, ct_x = Scan.split nk ct_in in
            (ct_c @ List.map2 sum ct_caps ct_captured, ct_x)
          in
          let zeros (Capture (d, _)) = Nx.P (Nx.zeros_like (node_primal d)) in
          let request =
            {
              Scan.req_carry = ct_carry @ List.map zeros captures;
              req_xs = carries @ xs @ ct_ys;
              req_step;
              req_reverse = not r.req_reverse;
            }
          in
          let result = Construct.scan request in
          let ct_c0, ct_caps = Scan.split nk result.r_carry in
          pick (pick carried initial) ct_c0 @ result.r_ys @ ct_caps
        in
        let slots =
          Linear.call tape inputs transpose
            (pick carried final @ pick outputs ys)
        in
        let r_carry, r_ys =
          Scan.split nc (duals i (carried @ outputs) (final @ ys) slots)
        in
        { Scan.r_carry; r_ys })

(* With value tangents a remat passes on as the remat of its function's jvp,
   over the arguments' primals and tangents, so that no dual of [i] crosses into
   the transformation that runs it: the function runs under [i] reinstalled, at
   duals of the arguments [i] tracks. Zeros fill the tangents of the others and
   of the results that come out with none, and stay plain: a value with no
   tangent remains no dual. *)
and remat_values : type p q.
    t -> p Nx.Ptree.t -> q Nx.Ptree.t -> (p -> q) -> p -> bool -> q =
 fun i p q f args recomputed ->
  let leaves, _ = Nx.Ptree.flatten p args in
  let tracked = owned i leaves in
  let dependent = ref [] in
  let f (a, da) =
    install i (fun () ->
        let a_leaves, _ = Nx.Ptree.flatten p a
        and da_leaves, _ = Nx.Ptree.flatten p da in
        let leaves = duals i tracked a_leaves (pick tracked da_leaves) in
        let y = f (Nx.Ptree.rebuild p ~like:a leaves) in
        let ys, _ = Nx.Ptree.flatten q y in
        dependent := owned i ys;
        ( Nx.Ptree.map q (fun _ y -> primal i y) y,
          Nx.Ptree.map q (fun _ y -> tangent i y) y ))
  in
  let args =
    ( Nx.Ptree.map p (fun _ x -> primal i x) args,
      Nx.Ptree.map p (fun _ x -> tangent i x) args )
  in
  let y, dy =
    Construct.perform
      (Remat
         { p = Nx.Ptree.pair p p; q = Nx.Ptree.pair q q; f; args; recomputed })
  in
  let ys, _ = Nx.Ptree.flatten q y and dys, _ = Nx.Ptree.flatten q dy in
  Nx.Ptree.rebuild q ~like:y (duals i !dependent ys (pick !dependent dys))

(* [region i tape captures flags leaves f] is [f] run at [leaves] under a child
   of [i] recording on [tape], each leaf [flags] marks a dual of the child with
   an input slot: the child, those slots and the result. *)
and region : type r.
    t ->
    Linear.tape ->
    capture list ->
    bool list ->
    Nx.packed list ->
    (Nx.packed list -> r) ->
    t * Nx.packed list * r =
 fun i tape captures flags leaves f ->
  let c = child i tape captures in
  let leaves, inputs = seed c tape flags leaves in
  (c, inputs, Linear.install tape (fun () -> install c (fun () -> f leaves)))

(* [pullback i captures flags leaves f dependent cts] runs [f] again at
   [leaves], under a child of [i] whose captures are fresh slots for [captures]:
   the cotangents of the slots of the leaves [flags] marks and of the captures,
   once [cts] reach the results [dependent] marks. *)
and pullback i captures flags leaves f dependent cts =
  let tape = Linear.create i.entry in
  let fresh (Capture (d, _)) = Capture (d, Linear.input tape d) in
  let c, inputs, ys = region i tape (List.map fresh captures) flags leaves f in
  if List.length (captures_of c) > List.length captures then
    invalid_arg
      (i.entry
     ^ ": a function run again for its transpose reads a value the \
        differentiation tracks that its first run did not");
  let received = Linear.cotangents tape in
  let seed (Nx.P y) ct =
    Option.iter
      (fun (_, dy) -> Linear.add received dy (Nx.unpack (Nx.dtype y) ct))
      (own c y)
  in
  List.iter2 seed (pick dependent ys) cts;
  Linear.transpose received;
  let cotangent (Nx.P s) =
    match Linear.cotangent received s with
    | Some g -> Nx.P g
    | None -> Nx.P (Nx.zeros_like s)
  in
  ( List.map cotangent inputs,
    List.map (fun (Capture (_, s)) -> cotangent (Nx.P s)) (captures_of c) )

(* Under reverse mode a remat is a linear call from its arguments' and captures'
   tangents to its dependent results', whose transpose runs [f] again. The
   forward run records onto a scratch tape and drops it, so [f]'s intermediates
   are not kept; each transpose reruns [f] at the arguments, read once the
   cotangents exist, inside a scope that drops the additions the first run
   counted. *)
and remat : type p q.
    t -> Linear.tape -> p Nx.Ptree.t -> q Nx.Ptree.t -> (p -> q) -> p -> q =
 fun i tape p q f args ->
  let leaves, _ = Nx.Ptree.flatten p args in
  let tracked = owned i leaves in
  let a = Nx.Ptree.map p (fun _ x -> primal i x) args in
  let run l = f (Nx.Ptree.rebuild p ~like:a l) in
  let captures = ref [] and dependent = ref [] in
  let forward a =
    let c, _, y =
      region i (Linear.create i.entry) [] tracked
        (fst (Nx.Ptree.flatten p a))
        run
    in
    dependent := owned c (fst (Nx.Ptree.flatten q y));
    captures := captures_of c;
    Nx.Ptree.map q (fun _ y -> primal c y) y
  in
  let y =
    Construct.perform (Remat { p; q; f = forward; args = a; recomputed = true })
  in
  if not (List.mem true !dependent) then y
  else
    let captures = !captures and dependent = !dependent in
    let inputs =
      tangents i tracked leaves
      @ List.map (fun (Capture (d, _)) -> Nx.P (tangent i d)) captures
    in
    let ys, _ = Nx.Ptree.flatten q y in
    let transpose cts =
      Total.discarding @@ fun () ->
      let kept = fst (Nx.Ptree.flatten p a) in
      let kept = Construct.perform (Barrier { values = kept; after = cts }) in
      let rerun l = fst (Nx.Ptree.flatten q (run l)) in
      let ct_args, ct_captured =
        pullback i captures tracked kept rerun dependent cts
      in
      ct_args @ ct_captured
    in
    let slots = Linear.call tape inputs transpose (pick dependent ys) in
    Nx.Ptree.rebuild q ~like:y (duals i dependent ys slots)

and install : type a. t -> (unit -> a) -> a =
 fun i f ->
  let owner = { Construct.owns = (fun x -> owns i x) } in
  let claims op = Construct.claims owner op in
  Construct.install
    {
      op = Some { run = (fun op -> run i op); claims };
      call = (fun c -> answer i c);
    }
    f
