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

(* [unwrap i x] is the primal and tangent of [x], an operand of an operation [i]
   claims through it. *)
let unwrap i x =
  match own i x with
  | Some d -> d
  | None -> assert false (* [i] claims the operation through [x]. *)

(* Rules *)

let no_rule op =
  invalid_arg
    (Printf.sprintf "Rune: the tangent of %s is not implemented" (name op))

let unary k x = eval (Unary (k, x))
let binary k a b = eval (Binary (k, a, b))
let mul a b = binary Mul a b

(* [terms da db] is the sum of the terms that exist: an operand with no tangent
   adds none, so its coefficient, which may be infinite or NaN where the operand
   is constant, never meets a zero. *)
let terms da db =
  match (da, db) with
  | Some da, Some db -> binary Add da db
  | Some d, None | None, Some d -> d
  | None, None -> assert false (* A rule runs for an operand with a tangent. *)

let unary_tangent op k x y dx =
  match[@warning "@4@8"] (k : Nx_backend.unary) with
  | Neg -> unary Neg dx
  | Sin -> mul dx (unary Cos x)
  | Exp -> mul dx y
  | Log -> binary Fdiv dx x
  | Tanh -> mul dx (Nx.rsub_s (Nx_dtype.one (Nx.dtype y)) (mul y y))
  | Recip | Abs | Sqrt | Sign | Cos | Tan | Asin | Acos | Atan | Sinh | Cosh
  | Trunc | Ceil | Floor | Round | Erf ->
      no_rule op

let binary_tangent op k a b y da db =
  match[@warning "@4@8"] (k : Nx_backend.binary) with
  | Add -> terms da db
  | Sub -> terms da (Option.map (unary Neg) db)
  | Mul -> terms (Option.map (fun da -> mul da b) da) (Option.map (mul a) db)
  | Fdiv ->
      terms
        (Option.map (fun da -> binary Fdiv da b) da)
        (Option.map (fun db -> mul db (unary Neg (binary Fdiv y b))) db)
  | Maximum ->
      let mask = Nx.cast (Nx.dtype y) (eval (Compare (Less, b, a))) in
      terms
        (Option.map (fun da -> mul da mask) da)
        (Option.map
           (fun db -> mul db (Nx.rsub_s (Nx_dtype.one (Nx.dtype y)) mask))
           db)
  | Idiv | Mod | Pow | Atan2 | Minimum | And | Or | Xor -> no_rule op

let claims : type r. t -> r Nx.Op.t -> bool =
 fun i op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> owns i x
  | Reduce (_, _, x) -> owns i x
  | Move (x, _) -> owns i x
  | Read x -> owns i x
  | Binary (_, a, b) -> owns i a || owns i b
  | Matmul (a, b) -> owns i a || owns i b
  | Compare _ | Where _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _
  | Cat _ | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _
  | Fold _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ ->
      List.exists (fun (Nx.P x) -> owns i x) (operands op)

(* [run i op] is [op], one of whose operands is a dual of [i]. *)
let run : type r. t -> r Nx.Op.t -> r =
 fun i op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) ->
      let x, dx = unwrap i x in
      let y = unary k x in
      dual i y (unary_tangent op k x y dx)
  | Binary (k, a, b) ->
      let a, da = split i a and b, db = split i b in
      let y = binary k a b in
      dual i y (binary_tangent op k a b y da db)
  | Reduce (k, axes, x) -> (
      let x, dx = unwrap i x in
      match[@warning "@4@8"] (k : Nx_backend.reduce) with
      | Sum -> dual i (eval (Reduce (k, axes, x))) (eval (Reduce (k, axes, dx)))
      | Prod | Max | Min -> no_rule op)
  | Move (x, m) ->
      let x, dx = unwrap i x in
      dual i (eval (Move (x, m))) (eval (Move (dx, m)))
  | Matmul (a, b) ->
      let a, da = split i a and b, db = split i b in
      dual i
        (eval (Matmul (a, b)))
        (terms
           (Option.map (fun da -> eval (Matmul (da, b))) da)
           (Option.map (fun db -> eval (Matmul (a, db))) db))
  | Read x -> eval (Read (fst (unwrap i x)))
  | Compare _ | Where _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _
  | Cat _ | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _
  | Fold _ | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _
  | Svd _ | Eig _ | Eigh _ | Solve_triangular _ | Place _ ->
      no_rule op

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
