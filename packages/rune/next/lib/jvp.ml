(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

type t = unit ref

let create () = ref ()

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

let own (type a b) i (x : (a, b) Nx.t) : ((a, b) Nx.t * (a, b) Nx.t) option =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Dual { owner; primal; tangent } when owner == i -> Some (primal, tangent)
      | _ -> None)
  | Host _ | Placed _ -> None

let owns i x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Dual { owner; _ } -> owner == i
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

let no_rule op =
  invalid_arg
    (Printf.sprintf "Rune: the tangent of %s is not implemented" (name op))

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

(* The derivative of a product at one zero is the product of the other factors,
   and at two zeros or more it is zero; no branch divides by zero. *)
let prod' ~axes x y =
  let y = kept ~axes x y and axes = Array.to_list axes in
  let zero = Nx.equal x (Nx.zeros_like x) in
  let safe = Nx.where zero (Nx.ones_like x) x in
  let count = Nx.sum ~axes ~keepdims:true (Nx.cast Nx.int32 zero) in
  let at_zero =
    Nx.where
      (Nx.equal count (Nx.ones_like count))
      (Nx.prod ~axes ~keepdims:true safe)
      (Nx.zeros_like y)
  in
  Nx.where zero at_zero (Nx.div y safe)

(* [shifted ~axis d fill x] is [x] moved [d] places along [axis], [d] elements
   of [fill] first and its last [d] elements dropped. *)
let shifted ~axis d fill x =
  let shape = Nx.shape x in
  let pads =
    Array.mapi (fun a _ -> if a = axis then (d, 0) else (0, 0)) shape
  in
  Nx.shrink (Array.map (fun n -> (0, n)) shape) (Nx.pad pads fill x)

(* The position along [axis] of the element each running extremum [y] takes: the
   last position where the extremum changed, so ties keep the first. *)
let running_arg ~axis y =
  let shape = Nx.shape y in
  let n = shape.(axis) in
  let along = Array.mapi (fun a _ -> if a = axis then n else 1) shape in
  let iota =
    Nx.broadcast_to shape (Nx.reshape along (Nx.arange Nx.int32 0 n 1))
  in
  let before = shifted ~axis 1 (Nx_dtype.zero (Nx.dtype y)) y in
  Nx.cummax ~axis (Nx.where (Nx.not_equal y before) iota (Nx.zeros_like iota))

(* [linear_scan ~axis a b] is [r] with [r_k = a_k r_(k-1) + b_k] along [axis]
   and [r_(-1) = 0], composed by doubling in about [log2 n] rounds of products
   and sums: no division, so it is exact where [a] has zeros, and it is linear
   in [b]. *)
let linear_scan ~axis a b =
  let n = (Nx.shape a).(axis) in
  let zero = Nx_dtype.zero (Nx.dtype b) in
  let rec go d a b =
    if d >= n then b
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

let binary_tangent op k a b y da db =
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
      terms
        (term (fun da -> mul da (Nx.mul b (Nx.pow a (Nx.sub_s b (one b))))) da)
        (term (fun db -> mul db (Nx.mul y (Nx.log a))) db)
  | Maximum -> selected y (Nx.less b a) da db
  | Minimum -> selected y (Nx.less a b) da db
  | Atan2 ->
      let denom = Nx.add (Nx.mul a a) (Nx.mul b b) in
      terms
        (term (fun da -> mul da (Nx.div b denom)) da)
        (term (fun db -> mul db (Nx.neg (Nx.div a denom))) db)
  | Mod -> no_rule op
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

(* The factor reads the Hermitian matrix that the strict lower triangle of [x]
   and the real part of its diagonal name, H = L Lᴴ, so dL = L Φ(L^-1 dH L^-H),
   with Φ the lower triangle less half the diagonal. Under [upper] the factor is
   U = Lᴴ. *)
let cholesky' ~upper y dx =
  let l = if upper then adjoint y else y in
  let dh =
    let low = Nx.tril ~k:(-1) dx in
    Nx.add (Nx.add low (adjoint low)) (diag_matrix (real_part (Nx.diagonal dx)))
  in
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

(* P A = L U for a square A: with X = L^-1 P dA U^-1, dL = L tril_-1(X) and dU =
   triu(X) U, packed as the factors are. *)
let lu' packed perm da =
  let pda =
    Nx.take_along_axis ~axis:(-2)
      ~indices:(Nx.broadcast_to (Nx.shape da) (Nx.unsqueeze ~axes:[ -1 ] perm))
      da
  in
  let y =
    Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:true packed pda
  in
  let x =
    Nx.matrix_transpose
      (Nx.solve_triangular ~upper:false ~transpose:false ~unit_diag:false
         (Nx.matrix_transpose packed)
         (Nx.matrix_transpose y))
  in
  let l =
    Nx.add (Nx.tril ~k:(-1) packed)
      (Nx.eye (Nx.dtype packed) (Nx.dim (-1) packed))
  in
  Nx.add
    (Nx.matmul l (Nx.tril ~k:(-1) x))
    (Nx.matmul (Nx.triu x) (Nx.triu packed))

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
let qr' ~reduced x q r dx =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x in
  if m >= n then begin
    if (not reduced) && m > n then
      invalid_arg
        "Rune: the tangent of a complete QR factorisation of a tall matrix has \
         no definition";
    qr_square q r dx
  end
  else
    let cols lo hi a =
      let shape = Nx.shape a in
      let k = Array.length shape - 1 in
      Nx.shrink
        (Array.mapi (fun i d -> if i = k then (lo, hi) else (0, d)) shape)
        a
    in
    let r1 = cols 0 m r and r2 = cols m n r in
    let dq, dr1 = qr_square q r1 (cols 0 m dx) in
    let dr2 = Nx.matmul (adjoint q) (Nx.sub (cols m n dx) (Nx.matmul dq r2)) in
    (dq, Nx.concatenate ~axis:(-1) [ dr1; dr2 ])

(* The interpreter *)

let claims : type r. t -> r Nx.Op.t -> bool =
 fun i op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> owns i x
  | Binary (_, a, b) -> owns i a || owns i b
  | Reduce (_, _, x) -> owns i x
  | Move (x, _) -> owns i x
  | Matmul (a, b) -> owns i a || owns i b
  | Compare _ | Where _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _
  | Cat _ | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _
  | Fold _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ | Read _ ->
      List.exists (fun (Nx.P x) -> owns i x) (operands op)

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
          dual i y (binary_tangent op k a b y da db))
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
      | Prod -> dual i y (eval (Reduce (Sum, axes, mul dx (prod' ~axes x y)))))
  | Scan (k, axis, x) -> (
      let x, dx = unwrap i x in
      let y = eval (Scan (k, axis, x)) in
      let axis = if axis < 0 then axis + Nx.ndim x else axis in
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
  | Argsort { descending; axis; x } ->
      eval (Argsort { descending; axis; x = primal i x })
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
  | Threefry (key, ctr) -> eval (Threefry (primal i key, primal i ctr))
  | Gather (axis, indices, x) ->
      linear x (fun x -> eval (Gather (axis, indices, x)))
  | Scatter { mode; unique; axis; indices; updates; into } ->
      let updates, du = split i updates and into, di = split i into in
      dual i
        (eval (Scatter { mode; unique; axis; indices; updates; into }))
        (eval
           (Scatter
              {
                mode;
                unique;
                axis;
                indices;
                updates = zeros_or du updates;
                into = zeros_or di into;
              }))
  | Update (x, starts, v) ->
      let x, dx = split i x and v, dv = split i v in
      dual i
        (eval (Update (x, starts, v)))
        (eval (Update (zeros_or dx x, starts, zeros_or dv v)))
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      linear x (fun x ->
          eval (Unfold { kernel_size; stride; dilation; padding; x }))
  | Fold { output_size; kernel_size; stride; dilation; padding; x } ->
      linear x (fun x ->
          eval (Fold { output_size; kernel_size; stride; dilation; padding; x }))
  | Matmul (a, b) ->
      let a, da = split i a and b, db = split i b in
      dual i
        (eval (Matmul (a, b)))
        (terms
           (Option.map (fun da -> eval (Matmul (da, b))) da)
           (Option.map (fun db -> eval (Matmul (a, db))) db))
  | Fft { inverse; axes; x } ->
      linear x (fun x -> eval (Fft { inverse; axes; x }))
  | Rfft { dtype; axes; x } ->
      let x, dx = unwrap i x in
      dual i
        (eval (Rfft { dtype; axes; x }))
        (eval (Rfft { dtype; axes; x = dx }))
  | Irfft { dtype; axes; s; x } ->
      let x, dx = unwrap i x in
      dual i
        (eval (Irfft { dtype; axes; s; x }))
        (eval (Irfft { dtype; axes; s; x = dx }))
  | Contiguous x -> linear x (fun x -> eval (Contiguous x))
  | Cholesky { upper; x } ->
      let x, dx = unwrap i x in
      let y = eval (Cholesky { upper; x }) in
      dual i y (cholesky' ~upper y dx)
  | Qr { reduced; x } ->
      let x, dx = unwrap i x in
      let q, r = eval (Qr { reduced; x }) in
      let dq, dr = qr' ~reduced x q r dx in
      (dual i q dq, dual i r dr)
  | Lu x ->
      let x, dx = unwrap i x in
      if Nx.dim (-1) x <> Nx.dim (-2) x then no_rule op;
      let packed, pivots, perm = eval (Lu x) in
      (dual i packed (lu' packed perm dx), pivots, perm)
  | Svd _ | Eig _ | Eigh _ -> no_rule op
  | Solve_triangular { upper; transpose; unit_diag; a; b } ->
      let a, da = split i a and b, db = split i b in
      let x = eval (Solve_triangular { upper; transpose; unit_diag; a; b }) in
      dual i x (solve' ~upper ~transpose ~unit_diag a b x da db)
  | Move (x, m) ->
      let x, dx = unwrap i x in
      dual i (eval (Move (x, m))) (eval (Move (viewable m dx, m)))
  | Place (p, x) -> linear x (fun x -> eval (Place (p, x)))
  | Read x -> eval (Read (primal i x))

let call : type r. t -> r Construct.t -> (unit -> r) option =
 fun i c ->
  match[@warning "@4@8"] c with
  | Detach x ->
      Option.map (fun (x, _) () -> Construct.perform (Detach x)) (own i x)
  | Scan _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
  | Lane_count _ | Add _ ->
      None

let install i f =
  Construct.install
    {
      op = Some { run = (fun op -> run i op); claims = (fun op -> claims i op) };
      call = (fun c -> call i c);
    }
    f
