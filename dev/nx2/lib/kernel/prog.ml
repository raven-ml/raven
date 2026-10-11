(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type unary =
  | Neg
  | Recip
  | Abs
  | Sign
  | Sqrt
  | Exp
  | Exp2
  | Log
  | Log2
  | Log1p
  | Expm1
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Erf
  | Floor
  | Ceil
  | Round
  | Trunc

type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
  | Idiv
  | Mod
  | Pow
  | Atan2
  | Maximum
  | Minimum
  | And
  | Or
  | Xor
  | Threefry

type compare = Equal | Not_equal | Less | Less_equal
type op0 = Fill of string | Iota of int
type op1 = Copy | Unary of unary | Cast | Bitcast
type op2 = Binary of binary | Compare of compare
type op3 = Where | Fma

module D = Nx_array.Dtype

(* Domains. A kind takes a set of the dtypes' kinds of number, as a mask. *)

let float = 1
let signed = 2
let unsigned = 4
let boolean = 8
let complex = 16
let integers = signed lor unsigned
let real = float lor integers
let numbers = real lor complex
let every = numbers lor boolean

let class_of : type v s. (v, s) D.t -> int =
 fun dt ->
  match D.kind dt with
  | Float -> float
  | Signed -> signed
  | Unsigned -> unsigned
  | Boolean -> boolean
  | Complex -> complex

let unary_takes = function
  | Neg | Recip -> numbers
  | Abs | Sign | Floor | Ceil | Round | Trunc -> real
  | Sqrt | Exp | Exp2 | Log | Log2 | Log1p | Expm1 | Sin | Cos | Tan | Asin
  | Acos | Atan | Sinh | Cosh | Tanh | Erf ->
      float

let binary_takes = function
  | Add | Sub | Mul -> numbers
  | Fdiv -> float lor complex
  | Idiv -> integers
  | Mod | Pow -> real
  | Atan2 -> float
  | Maximum | Minimum -> every
  | And | Or | Xor -> integers lor boolean
  | Threefry -> 0

let takes mask dt = mask land class_of dt <> 0

let accepts0 k dt =
  match k with Fill _ -> true | Iota _ -> takes real dt

let accepts1 k x y =
  match k with
  | Copy -> D.equal x y
  | Unary u -> D.equal x y && takes (unary_takes u) x
  | Cast -> true
  | Bitcast -> D.bits x = D.bits y

let accepts2 k x =
  match k with
  | Binary Threefry -> D.equal x D.Uint64
  | Binary b -> takes (binary_takes b) x
  | Compare _ -> true

let accepts3 k c x =
  match k with
  | Where -> D.is D.Boolean c
  | Fma -> D.equal c x && takes real x

(* Bits *)

(* nx_dtype.h's store of a float into a float format of at most 16 bits. *)
external narrow_bits : (int[@untagged]) -> (float[@unboxed]) -> (int[@untagged])
  = "nx_kernel_narrow_bits_byte" "nx_kernel_narrow_bits"
[@@noalloc]

let check_range dt x =
  if x < D.min_value dt || x > D.max_value dt then
    invalid_arg
      (Printf.sprintf "Nx_kernel.Prog.bits: %d is outside %s's range" x
         (D.name dt))

let bits : type v s. (v, s) D.t -> v -> string =
 fun dt x ->
  let b = Bytes.create (D.bytes dt 1) in
  (match dt with
  | Float64 -> Bytes.set_int64_ne b 0 (Int64.bits_of_float x)
  | Float32 -> Bytes.set_int32_ne b 0 (Int32.bits_of_float x)
  | Float16 -> Bytes.set_uint16_ne b 0 (narrow_bits (D.code dt) x)
  | Bfloat16 -> Bytes.set_uint16_ne b 0 (narrow_bits (D.code dt) x)
  | Float8_e4m3fn -> Bytes.set_uint8 b 0 (narrow_bits (D.code dt) x)
  | Float8_e5m2 -> Bytes.set_uint8 b 0 (narrow_bits (D.code dt) x)
  | Float4_e2m1fn -> Bytes.set_uint8 b 0 (narrow_bits (D.code dt) x)
  | Int64 -> Bytes.set_int64_ne b 0 x
  | Uint64 -> Bytes.set_int64_ne b 0 x
  | Int32 -> Bytes.set_int32_ne b 0 x
  | Uint32 -> Bytes.set_int32_ne b 0 x
  | Int16 ->
      check_range dt x;
      Bytes.set_int16_ne b 0 x
  | Uint16 ->
      check_range dt x;
      Bytes.set_uint16_ne b 0 x
  | Int8 ->
      check_range dt x;
      Bytes.set_int8 b 0 x
  | Uint8 ->
      check_range dt x;
      Bytes.set_uint8 b 0 x
  | Int4 ->
      check_range dt x;
      Bytes.set_uint8 b 0 (x land 0xF)
  | Uint4 ->
      check_range dt x;
      Bytes.set_uint8 b 0 x
  | Complex128 ->
      Bytes.set_int64_ne b 0 (Int64.bits_of_float x.re);
      Bytes.set_int64_ne b 8 (Int64.bits_of_float x.im)
  | Complex64 ->
      Bytes.set_int32_ne b 0 (Int32.bits_of_float x.re);
      Bytes.set_int32_ne b 4 (Int32.bits_of_float x.im)
  | Bool -> Bytes.set_uint8 b 0 (Bool.to_int x)
  | Bit -> Bytes.set_uint8 b 0 (Bool.to_int x));
  Bytes.unsafe_to_string b

(* Programs *)

type node =
  | In of int
  | Coord of int
  | Const of D.any * string
  | Op1 of op1 * D.any * int
  | Op2 of op2 * int * int
  | Op3 of op3 * int * int * int

type t = { ins : D.any iarray; nodes : node iarray; outs : int iarray }

let max_operands = 16

(* Whether [n]'s kind takes node operands of the dtypes [d0], [d1] and [d2],
   in order, past its arity unread. *)
let takes n d0 d1 d2 =
  match n with
  | In _ | Coord _ | Const _ -> true
  | Op1 (k, D.Any y, _) ->
      let (D.Any x) = d0 in
      accepts1 k x y
  | Op2 (k, _, _) ->
      let (D.Any x) = d0 in
      let (D.Any x') = d1 in
      D.equal x x' && accepts2 k x
  | Op3 (k, _, _, _) ->
      let (D.Any c) = d0 in
      let (D.Any x) = d1 in
      let (D.Any x') = d2 in
      D.equal x x' && accepts3 k c x

let strf = Printf.sprintf

(* Whether [b] is the bits of an element of [dt]. *)
let is_element (D.Any dt) b =
  String.length b = D.bytes dt 1
  &&
  match D.bits dt with
  | 1 -> Char.code b.[0] <= 1
  | 4 -> Char.code b.[0] < 16
  | 8 when D.equal dt D.Bool -> Char.code b.[0] <= 1
  | _ -> true

let refers i k = Some (strf "node %d refers to node %d" i k)
let refused i =
  Some (strf "node %d's kind does not take its operands' dtypes" i)

(* Whether node [k] comes before node [i]. *)
let before i k = k >= 0 && k < i

(* Why node [i], [nd], is no node of a program over [ins] whose earlier
   node [k] has the dtype [ty k], if it is not: an operand, axis or node out
   of reach, bits of no element, or a kind that does not take its operands'
   dtypes. *)
let node_problem ins ty i nd =
  match nd with
  | In k when k < 0 || k >= Iarray.length ins ->
      Some (strf "node %d reads operand %d" i k)
  | Coord a when a < 0 || a >= Nx_array.Layout.max_rank ->
      Some (strf "node %d reads axis %d" i a)
  | Const (dt, b) when not (is_element dt b) ->
      Some (strf "node %d's bits are no element of its dtype" i)
  | In _ | Coord _ | Const _ -> None
  | Op1 (_, _, a) ->
      if not (before i a) then refers i a
      else if takes nd (ty a) (ty a) (ty a) then None
      else refused i
  | Op2 (_, a, b) ->
      if not (before i a) then refers i a
      else if not (before i b) then refers i b
      else if takes nd (ty a) (ty b) (ty b) then None
      else refused i
  | Op3 (_, a, b, c) ->
      if not (before i a) then refers i a
      else if not (before i b) then refers i b
      else if not (before i c) then refers i c
      else if takes nd (ty a) (ty b) (ty c) then None
      else refused i

(* [nd]'s dtype over operands of dtypes [ins], [ty env k] being an earlier
   node [k]'s: a Binary kind and an Op3 give an operand's. The lookup takes
   its environment apart from itself so that [dtype] follows nodes back
   through a closed function, allocating nothing. *)
let node_dtype ins ty env = function
  | In k -> Iarray.get ins k
  | Coord _ -> D.Any D.Int64
  | Const (dt, _) | Op1 (_, dt, _) -> dt
  | Op2 (Binary _, a, _) -> ty env a
  | Op2 (Compare _, _, _) -> D.Any D.Bool
  | Op3 (_, _, b, _) -> ty env b

let too_many n =
  Some (strf "%d operands and outputs, more than %d" n max_operands)

(* Why [ins], [nodes] and [outs] make no program, if they do not. *)
let problem ~ins nodes ~outs =
  let nins = Iarray.length ins and n = Iarray.length nodes in
  let nouts = Iarray.length outs in
  let types = Array.make n (D.Any D.Bool) in
  let ty = Array.get types in
  let rec from i =
    if i = n then None
    else
      let nd = Iarray.get nodes i in
      match node_problem ins ty i nd with
      | Some _ as why -> why
      | None ->
          types.(i) <- node_dtype ins Array.get types nd;
          from (i + 1)
  in
  let rec out k =
    if k = nouts then None
    else
      let o = Iarray.get outs k in
      if o < 0 || o >= n then Some (strf "output %d is no node" o)
      else out (k + 1)
  in
  if nouts = 0 then Some "no output"
  else if nins + nouts > max_operands then too_many (nins + nouts)
  else match from 0 with Some _ as why -> why | None -> out 0

let invalid why = invalid_arg ("Nx_kernel.Prog.v: " ^ why)

let v ~ins nodes ~outs =
  match problem ~ins nodes ~outs with
  | Some why -> invalid why
  | None -> { ins; nodes; outs }

(* of_node's operand nodes and its outputs, by operand count, made once: a
   one-node program allocates its nodes array and itself. *)
let operand_nodes = Array.init max_operands (fun k -> In k)
let output = Array.init max_operands (fun k -> Iarray.init 1 (fun _ -> k))

(* Its operands' nodes being valid by construction, a one-node program is
   checked at its one node, as [v] checks it. *)
let of_node ~ins n =
  let k = Iarray.length ins in
  let why =
    if k + 1 > max_operands then too_many (k + 1)
    else node_problem ins (Iarray.get ins) k n
  in
  match why with
  | Some why -> invalid why
  | None ->
      let node i = if i < k then operand_nodes.(i) else n in
      let nodes = Iarray.init (k + 1) node in
      { ins; nodes; outs = output.(k) }

let rec dtype_of p i = node_dtype p.ins dtype_of p (Iarray.get p.nodes i)

let dtype p i =
  if i < 0 || i >= Iarray.length p.nodes then
    invalid_arg (strf "Nx_kernel.Prog.dtype: no node %d" i);
  dtype_of p i

let is_identity p =
  Iarray.length p.ins = 1
  && Iarray.length p.outs = 1
  && match Iarray.get p.nodes (Iarray.get p.outs 0) with
     | In _ -> true
     | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> false
