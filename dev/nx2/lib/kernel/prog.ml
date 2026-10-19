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
  | Asinh
  | Acosh
  | Atanh
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
  | Neg | Recip | Sign -> numbers
  | Abs | Floor | Ceil | Round | Trunc -> real
  | Sqrt | Exp | Exp2 | Log | Log2 | Sin | Cos | Tan | Asin | Acos | Atan
  | Sinh | Cosh | Tanh ->
      float lor complex
  | Asinh | Acosh | Atanh -> complex
  | Log1p | Expm1 | Erf -> float

let binary_takes = function
  | Add | Sub | Mul -> numbers
  | Fdiv -> float lor complex
  | Idiv -> integers
  | Mod -> real
  | Pow -> numbers
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
  if not (D.holds dt x) then
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

let max_operands = Nx_array.max_operands - 2

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

(* Expressions *)

module E = struct
  (* nx_kernel_stubs.c's walk reads this layout: the tags in this order, a
     dtype as its immediate constructor, and an operation's operands from
     field 2. *)
  type ('v, 's) t =
    | In : int * ('v, 's) D.t -> ('v, 's) t
    | Coord : int -> (int64, D.int64_elt) t
    | Const : ('v, 's) D.t * string -> ('v, 's) t
    | Op1 : op1 * ('v, 's) D.t * ('a, 'b) t -> ('v, 's) t
    | Op2 : op2 * ('v, 's) D.t * ('a, 'b) t * ('a, 'b) t -> ('v, 's) t
    | Op3 :
        op3 * ('v, 's) D.t * ('c, 'e) t * ('a, 'b) t * ('a, 'b) t
        -> ('v, 's) t

  type any = Any : ('v, 's) t -> any [@@unboxed]

  let any e = Any e

  let dtype : type v s. (v, s) t -> (v, s) D.t = function
    | Coord _ -> D.Int64
    | In (_, dt) -> dt
    | Const (dt, _) -> dt
    | Op1 (_, dt, _) -> dt
    | Op2 (_, dt, _, _) -> dt
    | Op3 (_, dt, _, _, _) -> dt

  let refuse by dts =
    let name (D.Any dt) = D.name dt in
    invalid_arg
      (strf "Nx_kernel.Prog.E.%s: the kind does not take %s" by
         (String.concat ", " (List.map name dts)))

  let input k dt =
    if k < 0 || k >= max_operands then
      invalid_arg (strf "Nx_kernel.Prog.E.input: no operand %d" k);
    In (k, dt)

  let coord a =
    if a < 0 || a >= Nx_array.Layout.max_rank then
      invalid_arg (strf "Nx_kernel.Prog.E.coord: no axis %d" a);
    Coord a

  let const dt x = Const (dt, bits dt x)

  let op1 k dt x =
    let a = dtype x in
    if not (accepts1 k a dt) then refuse "op1" [ D.Any a; D.Any dt ];
    Op1 (k, dt, x)

  (* [k], a kind of two operands, over [x] and [y]. *)
  let op2 by k x y =
    let a = dtype x in
    if not (accepts2 k a) then refuse by [ D.Any a ];
    Op2 (k, a, x, y)

  let binary k x y = op2 "binary" (Binary k) x y
  let compare k x y = Op2 (Compare k, D.Bool, x, y)

  let op3 k c x y =
    let a = dtype c and b = dtype x in
    if not (accepts3 k a b) then refuse "op3" [ D.Any a; D.Any b ];
    Op3 (k, b, c, x, y)

  (* Kinds. The types of those whose domain a type states, the floats', the
     booleans' and every dtype's, make their nodes valid; the others check.
     Each names its kind as a literal, which allocates nothing. *)

  let unary k x = op1 k (dtype x) x
  let neg x = unary (Unary Neg) x
  let recip x = unary (Unary Recip) x
  let abs x = unary (Unary Abs) x
  let sign x = unary (Unary Sign) x
  let sqrt x = unary (Unary Sqrt) x
  let exp x = unary (Unary Exp) x
  let exp2 x = unary (Unary Exp2) x
  let log x = unary (Unary Log) x
  let log2 x = unary (Unary Log2) x
  let log1p x = Op1 (Unary Log1p, dtype x, x)
  let expm1 x = Op1 (Unary Expm1, dtype x, x)
  let sin x = unary (Unary Sin) x
  let cos x = unary (Unary Cos) x
  let tan x = unary (Unary Tan) x
  let asin x = unary (Unary Asin) x
  let acos x = unary (Unary Acos) x
  let atan x = unary (Unary Atan) x
  let sinh x = unary (Unary Sinh) x
  let cosh x = unary (Unary Cosh) x
  let tanh x = unary (Unary Tanh) x
  let asinh x = unary (Unary Asinh) x
  let acosh x = unary (Unary Acosh) x
  let atanh x = unary (Unary Atanh) x
  let erf x = Op1 (Unary Erf, dtype x, x)
  let floor x = unary (Unary Floor) x
  let ceil x = unary (Unary Ceil) x
  let round x = unary (Unary Round) x
  let trunc x = unary (Unary Trunc) x
  let add x y = op2 "add" (Binary Add) x y
  let sub x y = op2 "sub" (Binary Sub) x y
  let mul x y = op2 "mul" (Binary Mul) x y
  let fdiv x y = op2 "fdiv" (Binary Fdiv) x y
  let idiv x y = op2 "idiv" (Binary Idiv) x y
  let rem x y = op2 "rem" (Binary Mod) x y
  let pow x y = op2 "pow" (Binary Pow) x y
  let atan2 x y = Op2 (Binary Atan2, dtype x, x, y)
  let maximum x y = Op2 (Binary Maximum, dtype x, x, y)
  let minimum x y = Op2 (Binary Minimum, dtype x, x, y)
  let logand x y = op2 "logand" (Binary And) x y
  let logor x y = op2 "logor" (Binary Or) x y
  let logxor x y = op2 "logxor" (Binary Xor) x y
  let threefry x y = Op2 (Binary Threefry, D.Uint64, x, y)
  let equal x y = Op2 (Compare Equal, D.Bool, x, y)
  let not_equal x y = Op2 (Compare Not_equal, D.Bool, x, y)
  let less x y = Op2 (Compare Less, D.Bool, x, y)
  let less_equal x y = Op2 (Compare Less_equal, D.Bool, x, y)
  let copy x = Op1 (Copy, dtype x, x)
  let cast dt x = Op1 (Cast, dt, x)
  let bitcast dt x = op1 Bitcast dt x
  let where c x y = Op3 (Where, dtype x, c, x, y)
  let fma x y z = op3 Fma x y z

  (* Operators *)

  let ( ~- ) = neg
  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul

  let ( / ) (type v s) (x : (v, s) t) y =
    match D.kind (dtype x) with
    | Signed | Unsigned -> idiv x y
    | Float | Complex | Boolean -> fdiv x y

  let ( mod ) = rem
  let ( ** ) = pow
  let ( land ) = logand
  let ( lor ) = logor
  let ( lxor ) = logxor
  let ( && ) x y = Op2 (Binary And, dtype x, x, y)
  let ( || ) x y = Op2 (Binary Or, dtype x, x, y)
  let ( = ) = equal
  let ( <> ) = not_equal
  let ( < ) = less
  let ( <= ) = less_equal
  let ( > ) x y = less y x
  let ( >= ) x y = less_equal y x
end

(* [walk outs] walks the expressions [outs] depth first, operands left to
   right, and gives each block the node of its structure, made as the walk
   finishes the first block of that structure: a block whose kind, fields
   and operands' nodes are another's is that node. It keys blocks by address
   and allocates nothing on the OCaml heap until it ends, so no collection
   can move a block under it. Its result holds native int32 words: the
   number of nodes, the number of outputs and each output's node, then six
   words per node, its tag, two fields and three operands ([-1] past its
   arity), then the constants' bits. A field is an operand, an axis, a
   dtype's code or a kind's code ([nx_prog.h]), as the tag reads it: In's
   operand and dtype, Coord's axis, Const's dtype, Op1's kind and dtype,
   Op2's and Op3's kind. A constant's second field is the word where its
   bits start, and its first operand their length in bytes. *)
external walk : E.any iarray -> string = "nx_kernel_prog_walk"

(* Prog's kinds by their codes in nx_prog.h, whose macros number each type's
   constructors in the order prog.ml declares them. *)
let op1s =
  [|
    Copy; Cast; Bitcast; Unary Neg; Unary Recip; Unary Abs; Unary Sign;
    Unary Sqrt; Unary Exp; Unary Exp2; Unary Log; Unary Log2; Unary Log1p;
    Unary Expm1; Unary Sin; Unary Cos; Unary Tan; Unary Asin; Unary Acos;
    Unary Atan; Unary Sinh; Unary Cosh; Unary Tanh; Unary Asinh; Unary Acosh;
    Unary Atanh; Unary Erf; Unary Floor;
    Unary Ceil; Unary Round; Unary Trunc;
  |]

let op2s =
  [|
    Binary Add; Binary Sub; Binary Mul; Binary Fdiv; Binary Idiv; Binary Mod;
    Binary Pow; Binary Atan2; Binary Maximum; Binary Minimum; Binary And;
    Binary Or; Binary Xor; Binary Threefry; Compare Equal; Compare Not_equal;
    Compare Less; Compare Less_equal;
  |]

let op3s = [| Where; Fma |]

(* The dtype of code [c], which the walk read from a dtype. *)
let dtype_of_code c = Option.get (D.of_code c)

(* The walk's tags: E.t's constructors in order, Op3 last. *)
let tag_in = 0
let tag_coord = 1
let tag_const = 2
let tag_op1 = 3
let tag_op2 = 4

(* Expressions checked their nodes as v would, and the walk made each node
   after its operands: what remains is what an expression cannot know, its
   program's operands and outputs. *)

(* Whether [ins] has an operand [k] of dtype [dt]. *)
let reads ins k (D.Any dt) =
  k < Iarray.length ins
  &&
  let (D.Any d) = Iarray.get ins k in
  D.equal d dt

let of_exprs ~ins outs =
  let invalid why = invalid_arg ("Nx_kernel.Prog.of_exprs: " ^ why) in
  let w = walk outs in
  let word i = Int32.to_int (String.get_int32_ne w (4 * i)) in
  let node n =
    let i = 2 + word 1 + (6 * n) in
    let x0 = word (i + 1) and x1 = word (i + 2) and a = word (i + 3) in
    let tag = word i in
    if tag = tag_in then begin
      if not (reads ins x0 (dtype_of_code x1)) then
        invalid (strf "In %d is no operand of ins" x0);
      In x0
    end
    else if tag = tag_coord then Coord x0
    else if tag = tag_const then
      Const (dtype_of_code x0, String.sub w (4 * x1) a)
    else if tag = tag_op1 then Op1 (op1s.(x0), dtype_of_code x1, a)
    else if tag = tag_op2 then Op2 (op2s.(x0), a, word (i + 4))
    else Op3 (op3s.(x0), a, word (i + 4), word (i + 5))
  in
  let nodes = Iarray.init (word 0) node in
  let outs = Iarray.init (word 1) (fun o -> word (2 + o)) in
  let n = Iarray.length ins + Iarray.length outs in
  if Iarray.length outs = 0 then invalid "no output"
  else if n > max_operands then invalid (Option.get (too_many n))
  else { ins; nodes; outs }

(* Binding *)

let same_dtype (D.Any x) (D.Any y) = D.equal x y

(* x's nodes come first and keep their indices; [p]'s follow, each reference
   renumbered, a read of operand [k] taking x's value. Both are programs over
   the result's operands, so their nodes keep their dtypes and kinds, and
   the result is one. *)
let bind p k x =
  let fn = "Nx_kernel.Prog.bind" in
  let n = Iarray.length p.ins in
  if k < 0 || k >= n then invalid_arg (strf "%s: no operand %d" fn k);
  let ins =
    Iarray.init (n - 1) (fun i -> Iarray.get p.ins (if i < k then i else i + 1))
  in
  let q = of_exprs ~ins [| x |] in
  let out = Iarray.get q.outs 0 in
  if not (same_dtype (dtype q out) (Iarray.get p.ins k)) then
    invalid_arg (strf "%s: the expression is not of operand %d's dtype" fn k);
  let m = Iarray.length q.nodes and np = Iarray.length p.nodes in
  let nodes = Array.make (m + np) (In 0) and count = ref m in
  Iarray.iteri (fun i nd -> nodes.(i) <- nd) q.nodes;
  let index = Array.make np 0 in
  let at i = index.(i) in
  for i = 0 to np - 1 do
    let nd = Iarray.get p.nodes i in
    match nd with
    | In j when j = k -> index.(i) <- out
    | _ ->
        nodes.(!count) <-
          (match nd with
          | In j -> if j > k then In (j - 1) else nd
          | Coord _ | Const _ -> nd
          | Op1 (o, dt, x) -> Op1 (o, dt, at x)
          | Op2 (o, x, y) -> Op2 (o, at x, at y)
          | Op3 (o, x, y, z) -> Op3 (o, at x, at y, at z));
        index.(i) <- !count;
        incr count
  done;
  { ins; nodes = Iarray.init !count (Array.get nodes); outs = Iarray.map at p.outs }
