(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx.Op
module Repr = Nx.Repr

type entry =
  | Input
  | Recorded : ('a, 'b) Nx.t Nx.Op.t * ('a, 'b) Nx_dtype.t -> entry

type tape = {
  entry : string;
  mutable entries : entry array;
  mutable length : int;
}

type (_, _) Repr.node +=
  | Slot : { tape : tape; index : int } -> ('a, 'b) Repr.node

let create entry = { entry; entries = Array.make 64 Input; length = 0 }

let owns t x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { tape; _ } -> tape == t
      | _ -> false)
  | Host _ | Placed _ -> false

let index x =
  match Repr.v x with
  | Traced tr -> (
      match Repr.Traced.node tr with
      | Slot { index; _ } -> index
      | _ -> assert false)
  | Host _ | Placed _ -> assert false

let slot t ~context placement dtype shape e =
  if t.length = Array.length t.entries then begin
    let entries = Array.make (2 * t.length) Input in
    Array.blit t.entries 0 entries 0 t.length;
    t.entries <- entries
  end;
  let index = t.length in
  t.entries.(index) <- e;
  t.length <- index + 1;
  Repr.Traced.v ~context placement dtype shape (Slot { tape = t; index })

let input t x =
  slot t ~context:(Repr.context x) (Nx.placement x) (Nx.dtype x) (Nx.shape x)
    Input

(* Recording *)

let nonlinear t op =
  invalid_arg
    (Printf.sprintf
       "%s: a custom_jvp tangent map applies %s to a tangent; a tangent map \
        must be linear in its tangents"
       t.entry (name op))

let untransposed op =
  invalid_arg
    (Printf.sprintf "Rune: the transpose of %s is not implemented" (name op))

(* [record t op x] is the slot of [op]'s result, recorded on [t]; [x] is a slot
   operand, whose context the result takes. *)
let record t op x =
  let dtype = Nx.Op.dtype op in
  slot t ~context:(Repr.context x) (placement op) dtype (Nx.Op.shape op)
    (Recorded (op, dtype))

let claims : type r. tape -> r Nx.Op.t -> bool =
 fun t op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> owns t x
  | Reduce (_, _, x) -> owns t x
  | Move (x, _) -> owns t x
  | Read x -> owns t x
  | Binary (_, a, b) -> owns t a || owns t b
  | Where (_, a, b) -> owns t a || owns t b
  | Matmul (a, b) -> owns t a || owns t b
  | Compare _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _ | Cat _
  | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _
  | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _ | Svd _
  | Eig _ | Eigh _ | Solve_triangular _ | Place _ ->
      List.exists (fun (Nx.P x) -> owns t x) (operands op)

(* [run t op] is [op], one of whose operands is a slot of [t]. *)
let run : type r. tape -> r Nx.Op.t -> r =
 fun t op ->
  match[@warning "@4@8"] op with
  | Unary (k, x) -> (
      match[@warning "@4@8"] (k : Nx_backend.unary) with
      | Neg -> record t op x
      | Recip | Abs | Sqrt | Sign | Exp | Log | Sin | Cos | Tan | Asin | Acos
      | Atan | Sinh | Cosh | Tanh | Trunc | Ceil | Floor | Round | Erf ->
          nonlinear t op)
  | Binary (k, a, b) ->
      let sa = owns t a and sb = owns t b in
      let linear =
        match[@warning "@4@8"] (k : Nx_backend.binary) with
        | Add | Sub -> sa && sb
        | Mul -> sa <> sb
        | Fdiv -> not sb
        | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor -> false
      in
      if linear then record t op (if sa then a else b) else nonlinear t op
  | Where (_, a, b) -> record t op (if owns t a then a else b)
  | Reduce (k, _, x) -> (
      match[@warning "@4@8"] (k : Nx_backend.reduce) with
      | Sum -> record t op x
      | Prod | Max | Min -> nonlinear t op)
  | Move (x, m) -> (
      match[@warning "@4@8"] m with
      | Reshape _ | Expand _ | Permute _ -> record t op x
      | Shrink _ | Flip _ | Window _ -> untransposed op)
  | Matmul (a, b) ->
      let sa = owns t a in
      if sa && owns t b then nonlinear t op
      else record t op (if sa then a else b)
  | Read _ ->
      invalid_arg
        (Printf.sprintf
           "%s: a custom_jvp tangent map reads a tangent's value; under \
            reverse mode a tangent has none"
           t.entry)
  | Compare _ | Scan _ | Arg_reduce _ | Sort _ | Argsort _ | Pad _ | Cat _
  | Convert _ | Threefry _ | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _
  | Fft _ | Rfft _ | Irfft _ | Contiguous _ | Cholesky _ | Qr _ | Lu _ | Svd _
  | Eig _ | Eigh _ | Solve_triangular _ | Place _ ->
      untransposed op

let call : type r. tape -> r Construct.t -> (unit -> r) option =
 fun t c ->
  match[@warning "@4@8"] c with
  | Detach x -> if owns t x then Some (fun () -> x) else None
  | Add (_, v) ->
      if owns t v then
        Some
          (fun () ->
            invalid_arg
              "Rune.Total.add: a custom_jvp tangent map adds a tangent under \
               reverse mode; a total takes values")
      else None
  | Scan _ | Remat _ | Barrier _ | Custom _ | Lanes _ | Lane_index _
  | Lane_count _ ->
      None

let install t f =
  Construct.install
    {
      op = Some { run = (fun op -> run t op); claims = (fun op -> claims t op) };
      call = (fun c -> call t c);
    }
    f

(* Transposing *)

type cotangents = { tape : tape; cts : Nx.packed option array }

let cotangents t = { tape = t; cts = Array.make t.length None }

let add cts x ct =
  if owns cts.tape x then
    let i = index x in
    cts.cts.(i) <-
      Some
        (match cts.cts.(i) with
        | None -> Nx.P ct
        | Some prev -> Nx.P (Nx.add (Nx.unpack (Nx.dtype ct) prev) ct))

let cotangent cts x =
  if owns cts.tape x then Option.map (Nx.unpack (Nx.dtype x)) cts.cts.(index x)
  else None

(* [unbroadcast ct shape] is [ct] summed over the axes along which [shape] was
   broadcast to [ct]'s shape. *)
let unbroadcast ct shape =
  let ct_shape = Nx.shape ct in
  if ct_shape = shape then ct
  else
    let lead = Array.length ct_shape - Array.length shape in
    let axes =
      List.filter
        (fun i -> i < lead || (shape.(i - lead) = 1 && ct_shape.(i) <> 1))
        (List.init (Array.length ct_shape) Fun.id)
    in
    Nx.reshape shape (Nx.sum ~axes ct)

let transpose_op : type a b.
    cotangents -> (a, b) Nx.t Nx.Op.t -> (a, b) Nx.t -> unit =
 fun cts op ct ->
  let owns x = owns cts.tape x and add x v = add cts x v in
  match[@warning "@4@8"] op with
  | Unary (_, x) -> add x (Nx.neg ct)
  | Binary (k, a, b) -> (
      match[@warning "@4@8"] (k : Nx_backend.binary) with
      | Add ->
          add a ct;
          add b ct
      | Sub ->
          add a ct;
          add b (Nx.neg ct)
      | Mul -> if owns a then add a (Nx.mul ct b) else add b (Nx.mul a ct)
      | Fdiv -> add a (Nx.div ct b)
      | Idiv | Mod | Pow | Atan2 | Maximum | Minimum | And | Or | Xor ->
          assert false (* Never recorded. *))
  | Where (c, a, b) ->
      let zeros = Nx.zeros_like ct in
      add a (Nx.where c ct zeros);
      add b (Nx.where c zeros ct)
  | Reduce (_, axes, x) ->
      let shape = Nx.shape x in
      let kept =
        Array.mapi (fun i d -> if Array.mem i axes then 1 else d) shape
      in
      add x (Nx.broadcast_to shape (Nx.reshape kept ct))
  | Move (x, m) -> (
      match[@warning "@4@8"] m with
      | Reshape _ -> add x (Nx.reshape (Nx.shape x) ct)
      | Expand _ -> add x (unbroadcast ct (Nx.shape x))
      | Permute p ->
          let inverse = Array.make (Array.length p) 0 in
          Array.iteri (fun i j -> inverse.(j) <- i) p;
          add x (Nx.transpose ~axes:(Array.to_list inverse) ct)
      | Shrink _ | Flip _ | Window _ -> assert false (* Never recorded. *))
  | Matmul (a, b) ->
      if owns a then
        add a (unbroadcast (Nx.matmul ct (Nx.matrix_transpose b)) (Nx.shape a))
      else
        add b (unbroadcast (Nx.matmul (Nx.matrix_transpose a) ct) (Nx.shape b))
  | Compare _ | Scan _ | Sort _ | Pad _ | Cat _ | Convert _ | Threefry _
  | Gather _ | Scatter _ | Update _ | Unfold _ | Fold _ | Fft _ | Rfft _
  | Irfft _ | Contiguous _ | Cholesky _ | Solve_triangular _ | Place _
  | Arg_reduce _ | Argsort _ | Read _ ->
      assert false (* Never recorded. *)

let transpose cts =
  let t = cts.tape in
  for i = t.length - 1 downto 0 do
    match (t.entries.(i), cts.cts.(i)) with
    | Recorded (op, dtype), Some ct -> transpose_op cts op (Nx.unpack dtype ct)
    | (Input | Recorded _), None | Input, Some _ -> ()
  done
