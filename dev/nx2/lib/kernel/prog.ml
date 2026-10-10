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

(* A program is nx_spec.h's nx_prog: int32 counts of operands, nodes and
   outputs and a zero word, then a 40-byte record per node (tag, kind,
   dtype, three operands, then 16 bytes of constant bits, zero-padded), then
   the operands' dtype codes and the outputs' nodes, int32 in the host's byte
   order. *)
type t = string

let max_operands = 16
let header = 16
let record = 40
let at_bits = 24
let get s at = Int32.to_int (String.get_int32_ne s at)
let set b at x = Bytes.set_int32_ne b at (Int32.of_int x)

(* The dtype whose code is at [at] of a program, which holds only dtypes'
   codes. *)
let dtype_at p at = Option.get (D.of_code (get p at))

(* Kinds as nx_spec.h's codes: one enum per arity, in the order of the types'
   constructors. *)

let unaries =
  [|
    Neg; Recip; Abs; Sign; Sqrt; Exp; Exp2; Log; Log2; Log1p; Expm1; Sin; Cos;
    Tan; Asin; Acos; Atan; Sinh; Cosh; Tanh; Erf; Floor; Ceil; Round; Trunc;
  |]

let binaries =
  [|
    Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or; Xor;
    Threefry;
  |]

let compares = [| Equal; Not_equal; Less; Less_equal |]

let index_of xs x =
  let rec go i = if xs.(i) = x then i else go (i + 1) in
  go 0

let code1 = function
  | Copy -> 0
  | Cast -> 1
  | Bitcast -> 2
  | Unary u -> 3 + index_of unaries u

let code2 = function
  | Binary b -> index_of binaries b
  | Compare c -> Array.length binaries + index_of compares c

let code3 = function Where -> 0 | Fma -> 1

let op1_of = function
  | 0 -> Some Copy
  | 1 -> Some Cast
  | 2 -> Some Bitcast
  | c when c >= 3 && c < 3 + Array.length unaries -> Some (Unary unaries.(c - 3))
  | _ -> None

let op2_of c =
  let nb = Array.length binaries in
  if c >= 0 && c < nb then Some (Binary binaries.(c))
  else if c >= nb && c < nb + Array.length compares then
    Some (Compare compares.(c - nb))
  else None

let op3_of = function 0 -> Some Where | 1 -> Some Fma | _ -> None

let accepts n dts =
  let arity = Array.length dts in
  match n with
  | In _ | Coord _ | Const _ -> arity = 0
  | Op1 (k, D.Any y, _) ->
      arity = 1
      &&
      let (D.Any x) = dts.(0) in
      accepts1 k x y
  | Op2 (k, _, _) ->
      arity = 2
      &&
      let (D.Any x) = dts.(0) in
      let (D.Any x') = dts.(1) in
      D.equal x x' && accepts2 k x
  | Op3 (k, _, _, _) ->
      arity = 3
      &&
      let (D.Any c) = dts.(0) in
      let (D.Any x) = dts.(1) in
      let (D.Any x') = dts.(2) in
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

(* Why node [i], [nd], is no node of a program of [nins] operands, if it is
   not: an operand, axis or node out of reach, or bits of no element. *)
let node_problem nins i nd =
  let refers ks =
    Option.map
      (fun k -> strf "node %d refers to node %d" i k)
      (List.find_opt (fun k -> k < 0 || k >= i) ks)
  in
  match nd with
  | In k when k < 0 || k >= nins -> Some (strf "node %d reads operand %d" i k)
  | Coord a when a < 0 || a >= Nx_array.Layout.max_rank ->
      Some (strf "node %d reads axis %d" i a)
  | Const (dt, b) when not (is_element dt b) ->
      Some (strf "node %d's bits are no element of its dtype" i)
  | In _ | Coord _ | Const _ -> None
  | Op1 (_, _, a) -> refers [ a ]
  | Op2 (_, a, b) -> refers [ a; b ]
  | Op3 (_, a, b, c) -> refers [ a; b; c ]

(* Why [ins], [nodes] and [outs] make no program, if they do not, with each
   node's dtype written to [types] until the first problem. *)
let problem ~ins nodes ~outs types =
  let nins = Array.length ins and n = Array.length nodes in
  let nouts = Array.length outs in
  let operands = function
    | In _ | Coord _ | Const _ -> [||]
    | Op1 (_, _, a) -> [| types.(a) |]
    | Op2 (_, a, b) -> [| types.(a); types.(b) |]
    | Op3 (_, a, b, c) -> [| types.(a); types.(b); types.(c) |]
  in
  let rec from i =
    if i = n then None
    else
      let nd = nodes.(i) in
      match node_problem nins i nd with
      | Some _ as why -> why
      | None ->
          let dts = operands nd in
          if not (accepts nd dts) then
            Some (strf "node %d's kind does not take its operands' dtypes" i)
          else begin
            types.(i) <-
              (match nd with
              | In k -> ins.(k)
              | Coord _ -> D.Any D.Int64
              | Const (dt, _) | Op1 (_, dt, _) -> dt
              | Op2 (Binary _, _, _) -> dts.(0)
              | Op2 (Compare _, _, _) -> D.Any D.Bool
              | Op3 (_, _, _, _) -> dts.(1));
            from (i + 1)
          end
  in
  if nouts = 0 then Some "no output"
  else if nins + nouts > max_operands then
    Some
      (strf "%d operands and outputs, more than %d" (nins + nouts) max_operands)
  else
    match from 0 with
    | Some _ as why -> why
    | None ->
        Array.find_map
          (fun o ->
            if o < 0 || o >= n then Some (strf "output %d is no node" o)
            else None)
          outs

(* The bytes of the program [ins], [nodes] and [outs], whose nodes have the
   dtypes [types]. *)
let encode ~ins nodes ~outs types =
  let nins = Array.length ins and n = Array.length nodes in
  let nouts = Array.length outs in
  let at_ins = header + (record * n) in
  let at_outs = at_ins + (4 * nins) in
  let b = Bytes.make (at_outs + (4 * nouts)) '\000' in
  set b 0 nins;
  set b 4 n;
  set b 8 nouts;
  let code (D.Any dt) = D.code dt in
  Array.iteri
    (fun i nd ->
      let at = header + (record * i) in
      let fields tag kind a b' c =
        set b at tag;
        set b (at + 4) kind;
        set b (at + 8) (code types.(i));
        set b (at + 12) a;
        set b (at + 16) b';
        set b (at + 20) c
      in
      match nd with
      | In k -> fields 0 0 k 0 0
      | Coord a -> fields 1 0 a 0 0
      | Const (_, bits) ->
          fields 2 0 0 0 0;
          Bytes.blit_string bits 0 b (at + at_bits) (String.length bits)
      | Op1 (k, _, a) -> fields 3 (code1 k) a 0 0
      | Op2 (k, a, b') -> fields 4 (code2 k) a b' 0
      | Op3 (k, a, b', c) -> fields 5 (code3 k) a b' c)
    nodes;
  Array.iteri (fun k dt -> set b (at_ins + (4 * k)) (code dt)) ins;
  Array.iteri (fun k o -> set b (at_outs + (4 * k)) o) outs;
  Bytes.unsafe_to_string b

let v ~ins nodes ~outs =
  let types = Array.make (Array.length nodes) (D.Any D.Bool) in
  match problem ~ins nodes ~outs types with
  | Some why -> invalid_arg ("Nx_kernel.Prog.v: " ^ why)
  | None -> encode ~ins nodes ~outs types

let length p = get p 4
let at_ins p = header + (record * length p)

let ins p =
  Array.init (get p 0) (fun k -> dtype_at p (at_ins p + (4 * k)))

let outs p =
  let at = at_ins p + (4 * get p 0) in
  Array.init (get p 8) (fun k -> get p (at + (4 * k)))

let check_node fn p i =
  if i < 0 || i >= length p then
    invalid_arg (Printf.sprintf "Nx_kernel.Prog.%s: no node %d" fn i)

let dtype p i =
  check_node "dtype" p i;
  dtype_at p (header + (record * i) + 8)

(* The node whose record is the [i]th of the bytes [s], or [None] if the
   record names none. *)
let node_at s i =
  let at = header + (record * i) in
  let f k = get s (at + k) in
  let a = f 12 and b = f 16 and c = f 20 in
  match (f 0, D.of_code (f 8)) with
  | 0, _ -> Some (In a)
  | 1, _ -> Some (Coord a)
  | 2, Some (D.Any d as dt) ->
      Some (Const (dt, String.sub s (at + at_bits) (D.bytes d 1)))
  | 3, Some dt -> Option.map (fun k -> Op1 (k, dt, a)) (op1_of (f 4))
  | 4, _ -> Option.map (fun k -> Op2 (k, a, b)) (op2_of (f 4))
  | 5, _ -> Option.map (fun k -> Op3 (k, a, b, c)) (op3_of (f 4))
  | _ -> None

let node p i =
  check_node "node" p i;
  Option.get (node_at p i)

let of_node ~ins n =
  let k = Array.length ins in
  v ~ins (Array.append (Array.init k (fun i -> In i)) [| n |]) ~outs:[| k |]

(* The [n] values [f] gives, or [None] if one is [None]. *)
let all n f =
  let xs = Array.init n f in
  if Array.for_all Option.is_some xs then Some (Array.map Option.get xs)
  else None

let of_string s =
  let n = String.length s in
  if n < header then None
  else
    let nins = get s 0 and nnodes = get s 4 and nouts = get s 8 in
    if
      nins < 0 || nnodes < 0 || nouts < 0
      || n <> header + (record * nnodes) + (4 * (nins + nouts))
    then None
    else
      let at_ins = header + (record * nnodes) in
      let dtype k = D.of_code (get s (at_ins + (4 * k))) in
      match (all nins dtype, all nnodes (node_at s)) with
      | Some ins, Some nodes -> (
          let at_outs = at_ins + (4 * nins) in
          let outs = Array.init nouts (fun k -> get s (at_outs + (4 * k))) in
          let types = Array.make nnodes (D.Any D.Bool) in
          match problem ~ins nodes ~outs types with
          | Some _ -> None
          | None ->
              let p = encode ~ins nodes ~outs types in
              if String.equal p s then Some p else None)
      | _ -> None
