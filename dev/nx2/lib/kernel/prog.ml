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

external loop_operands : unit -> (int[@untagged])
  = "nx_kernel_max_operands_byte" "nx_kernel_max_operands"
[@@noalloc]

let max_operands = loop_operands ()
let header = 16
let record = 40
let at_bits = 24
let get s at = Int32.to_int (String.get_int32_ne s at)
let set b at x = Bytes.set_int32_ne b at (Int32.of_int x)
let dtypes = Array.of_list D.all

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

let op1_of c =
  match c with
  | 0 -> Copy
  | 1 -> Cast
  | 2 -> Bitcast
  | c -> Unary unaries.(c - 3)

let op2_of c =
  let nb = Array.length binaries in
  if c < nb then Binary binaries.(c) else Compare compares.(c - nb)

let op3_of = function 0 -> Where | _ -> Fma

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

let invalid fmt = Format.kasprintf invalid_arg ("Nx_kernel.Prog.v: " ^^ fmt)

(* Whether [b] is the bits of an element of [dt]. *)
let is_element (D.Any dt) b =
  String.length b = D.bytes dt 1
  &&
  match D.bits dt with
  | 1 -> Char.code b.[0] <= 1
  | 4 -> Char.code b.[0] < 16
  | 8 when D.equal dt D.Bool -> Char.code b.[0] <= 1
  | _ -> true

let v ~ins nodes ~outs =
  let nins = Array.length ins and n = Array.length nodes in
  let nouts = Array.length outs in
  if nouts = 0 then invalid "no output";
  if nins + nouts > max_operands then
    invalid "%d operands and outputs, more than %d" (nins + nouts) max_operands;
  let types = Array.make n (D.Any D.Bool) in
  let earlier i k =
    if k < 0 || k >= i then invalid "node %d refers to node %d" i k;
    types.(k)
  in
  Array.iteri
    (fun i nd ->
      let dts =
        match nd with
        | In k ->
            if k < 0 || k >= nins then invalid "node %d reads operand %d" i k;
            [||]
        | Coord a ->
            if a < 0 || a >= Nx_array.Layout.max_rank then
              invalid "node %d reads axis %d" i a;
            [||]
        | Const (dt, b) ->
            if not (is_element dt b) then
              invalid "node %d's bits are no element of its dtype" i;
            [||]
        | Op1 (_, _, a) -> [| earlier i a |]
        | Op2 (_, a, b) -> [| earlier i a; earlier i b |]
        | Op3 (_, a, b, c) -> [| earlier i a; earlier i b; earlier i c |]
      in
      if not (accepts nd dts) then
        invalid "node %d's kind does not take its operands' dtypes" i;
      types.(i) <-
        (match nd with
        | In k -> ins.(k)
        | Coord _ -> D.Any D.Int64
        | Const (dt, _) | Op1 (_, dt, _) -> dt
        | Op2 (Binary _, _, _) -> dts.(0)
        | Op2 (Compare _, _, _) -> D.Any D.Bool
        | Op3 (_, _, _, _) -> dts.(1)))
    nodes;
  Array.iter
    (fun o -> if o < 0 || o >= n then invalid "output %d is no node" o)
    outs;
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

let length p = get p 4
let at_ins p = header + (record * length p)

let ins p =
  Array.init (get p 0) (fun k -> dtypes.(get p (at_ins p + (4 * k))))

let outs p =
  let at = at_ins p + (4 * get p 0) in
  Array.init (get p 8) (fun k -> get p (at + (4 * k)))

let check_node fn p i =
  if i < 0 || i >= length p then
    invalid_arg (Printf.sprintf "Nx_kernel.Prog.%s: no node %d" fn i)

let dtype p i =
  check_node "dtype" p i;
  dtypes.(get p (header + (record * i) + 8))

let node p i =
  check_node "node" p i;
  let at = header + (record * i) in
  let f k = get p (at + k) in
  let a = f 12 and b = f 16 and c = f 20 in
  match f 0 with
  | 0 -> In a
  | 1 -> Coord a
  | 2 ->
      let (D.Any dt as d) = dtypes.(f 8) in
      Const (d, String.sub p (at + at_bits) (D.bytes dt 1))
  | 3 -> Op1 (op1_of (f 4), dtypes.(f 8), a)
  | 4 -> Op2 (op2_of (f 4), a, b)
  | _ -> Op3 (op3_of (f 4), a, b, c)

let of_node ~ins n =
  let k = Array.length ins in
  v ~ins (Array.append (Array.init k (fun i -> In i)) [| n |]) ~outs:[| k |]

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
      match
        v ~ins:(ins s) (Array.init nnodes (node s)) ~outs:(outs s)
      with
      | p -> if String.equal p s then Some p else None
      | exception Invalid_argument _ -> None
