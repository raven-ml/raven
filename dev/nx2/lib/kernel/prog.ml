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

(* The bits a store of [x] writes into the float dtype [dt] of at most 16
   bits. *)
let narrow_bits dt x = Int64.to_int (D.store dt x)

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

(* Whether [b] is the bits of an element of [dt]: any bits of its width, a
   sub-byte element's in the low bits of its byte. A boolean byte is
   [true] iff it is not zero. *)
let is_element (D.Any dt) b =
  String.length b = D.bytes dt 1
  &&
  match D.bits dt with
  | 1 -> Char.code b.[0] <= 1
  | 4 -> Char.code b.[0] < 16
  | _ -> true

let refers i k = Some (strf "node %d refers to node %d" i k)
let refused i =
  Some (strf "node %d's kind does not take its operands' dtypes" i)

(* Whether node [k] comes before node [i]. *)
let before i k = k >= 0 && k < i

(* Why node [i], [nd], is no node of a program over [ins] whose earlier
   node [k] has the dtype [ty env k], if it is not: an operand, axis or node
   out of reach, bits of no element, or a kind that does not take its
   operands' dtypes. The lookup takes its environment apart, as
   [node_dtype]'s does, so that no closure is made per call. *)
let node_problem ins ty env i nd =
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
      else if takes nd (ty env a) (ty env a) (ty env a) then None
      else refused i
  | Op2 (_, a, b) ->
      if not (before i a) then refers i a
      else if not (before i b) then refers i b
      else if takes nd (ty env a) (ty env b) (ty env b) then None
      else refused i
  | Op3 (_, a, b, c) ->
      if not (before i a) then refers i a
      else if not (before i b) then refers i b
      else if not (before i c) then refers i c
      else if takes nd (ty env a) (ty env b) (ty env c) then None
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

(* Why [ins], [nodes] and [outs] make no program, if they do not. *)
let problem ~ins nodes ~outs =
  let n = Iarray.length nodes in
  let nouts = Iarray.length outs in
  let types = Array.make n (D.Any D.Bool) in
  let rec from i =
    if i = n then None
    else
      let nd = Iarray.get nodes i in
      match node_problem ins Array.get types i nd with
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
  else match from 0 with Some _ as why -> why | None -> out 0

let invalid why = invalid_arg ("Nx_kernel.Prog.v: " ^ why)

let v ~ins nodes ~outs =
  match problem ~ins nodes ~outs with
  | Some why -> invalid why
  | None -> { ins; nodes; outs }

(* of_node's operand nodes, its runs of them and its outputs, by operand
   count up to a kind's three, made once: such a one-node program allocates
   its node alone in an array, its nodes array and itself, and no
   closure. *)
let kept = 4
let operand_nodes = Array.init kept (fun k -> In k)
let operand_runs =
  Array.init kept (fun k -> Iarray.init k (Array.get operand_nodes))
let output = Array.init kept (fun k -> Iarray.init 1 (fun _ -> k))
let runs k =
  if k < kept then operand_runs.(k) else Iarray.init k (fun j -> In j)

let outputs k = if k < kept then output.(k) else Iarray.init 1 (fun _ -> k)

(* Its operands' nodes being valid by construction, a one-node program is
   checked at its one node, as [v] checks it. *)
let of_node ~ins n =
  let k = Iarray.length ins in
  match node_problem ins Iarray.get ins k n with
  | Some why -> invalid why
  | None ->
      let nodes = Iarray.append (runs k) [| n |] in
      { ins; nodes; outs = outputs k }

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

(* Whether nodes [i] to [last - 1] of [p] are each [In] of its own index. *)
let rec operands_first p i last =
  i = last
  || (match Iarray.get p.nodes i with In j -> j = i | _ -> false)
     && operands_first p (i + 1) last

(* The last node of [p] where it is one node over its operands, read in
   order ([single]), and [In 0] otherwise, which ends no such program. *)
let last_single p =
  let m = Iarray.length p.nodes - 1 in
  let no = operand_nodes.(0) in
  if
    Iarray.length p.ins <> m
    || Iarray.length p.outs <> 1
    || Iarray.get p.outs 0 <> m
    || not (operands_first p 0 m)
  then no
  else
    match Iarray.get p.nodes m with
    | (Const _ | Coord _) as n when m = 0 -> n
    | Op1 (_, _, 0) as n when m = 1 -> n
    | Op2 (_, 0, 1) as n when m = 2 -> n
    | Op3 (_, 0, 1, 2) as n when m = 3 -> n
    | In _ | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> no

let is_single p =
  match last_single p with
  | In _ -> false
  | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _ -> true

let single p =
  match last_single p with
  | In _ -> None
  | (Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _) as n -> Some n

(* Expressions *)

module E = struct
  (* Each expression has an id no other expression made has, by which
     of_exprs walks a shared subexpression once. *)
  type ('v, 's) t =
    | In : { id : int; k : int; dt : ('v, 's) D.t } -> ('v, 's) t
    | Coord : { id : int; axis : int } -> (int64, D.int64_elt) t
    | Const : { id : int; dt : ('v, 's) D.t; bits : string } -> ('v, 's) t
    | Op1 : {
        id : int;
        k : op1;
        dt : ('v, 's) D.t;
        x : ('a, 'b) t;
      }
        -> ('v, 's) t
    | Op2 : {
        id : int;
        k : op2;
        dt : ('v, 's) D.t;
        x : ('a, 'b) t;
        y : ('a, 'b) t;
      }
        -> ('v, 's) t
    | Op3 : {
        id : int;
        k : op3;
        dt : ('v, 's) D.t;
        c : ('c, 'e) t;
        x : ('a, 'b) t;
        y : ('a, 'b) t;
      }
        -> ('v, 's) t

  type any = Any : ('v, 's) t -> any [@@unboxed]

  let any e = Any e
  let ids = Atomic.make 0
  let fresh () = Atomic.fetch_and_add ids 1

  let id : type v s. (v, s) t -> int = function
    | In { id; _ } -> id
    | Coord { id; _ } -> id
    | Const { id; _ } -> id
    | Op1 { id; _ } -> id
    | Op2 { id; _ } -> id
    | Op3 { id; _ } -> id

  let dtype : type v s. (v, s) t -> (v, s) D.t = function
    | Coord _ -> D.Int64
    | In { dt; _ } -> dt
    | Const { dt; _ } -> dt
    | Op1 { dt; _ } -> dt
    | Op2 { dt; _ } -> dt
    | Op3 { dt; _ } -> dt

  let make1 k dt x = Op1 { id = fresh (); k; dt; x }
  let make2 k dt x y = Op2 { id = fresh (); k; dt; x; y }
  let make3 k dt c x y = Op3 { id = fresh (); k; dt; c; x; y }

  let refuse by dts =
    let name (D.Any dt) = D.name dt in
    invalid_arg
      (strf "Nx_kernel.Prog.E.%s: the kind does not take %s" by
         (String.concat ", " (List.map name dts)))

  let input k dt =
    if k < 0 then
      invalid_arg (strf "Nx_kernel.Prog.E.input: no operand %d" k);
    In { id = fresh (); k; dt }

  let coord a =
    if a < 0 || a >= Nx_array.Layout.max_rank then
      invalid_arg (strf "Nx_kernel.Prog.E.coord: no axis %d" a);
    Coord { id = fresh (); axis = a }

  let const dt x = Const { id = fresh (); dt; bits = bits dt x }

  let op1 k dt x =
    let a = dtype x in
    if not (accepts1 k a dt) then refuse "op1" [ D.Any a; D.Any dt ];
    make1 k dt x

  (* [k], a kind of two operands, over [x] and [y]. *)
  let op2 by k x y =
    let a = dtype x in
    if not (accepts2 k a) then refuse by [ D.Any a ];
    make2 k a x y

  let binary k x y = op2 "binary" (Binary k) x y
  let compare k x y = make2 (Compare k) D.Bool x y

  let op3 k c x y =
    let a = dtype c and b = dtype x in
    if not (accepts3 k a b) then refuse "op3" [ D.Any a; D.Any b ];
    make3 k b c x y

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
  let log1p x = make1 (Unary Log1p) (dtype x) x
  let expm1 x = make1 (Unary Expm1) (dtype x) x
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
  let erf x = make1 (Unary Erf) (dtype x) x
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
  let atan2 x y = make2 (Binary Atan2) (dtype x) x y
  let maximum x y = make2 (Binary Maximum) (dtype x) x y
  let minimum x y = make2 (Binary Minimum) (dtype x) x y
  let logand x y = op2 "logand" (Binary And) x y
  let logor x y = op2 "logor" (Binary Or) x y
  let logxor x y = op2 "logxor" (Binary Xor) x y
  let threefry x y = make2 (Binary Threefry) D.Uint64 x y
  let equal x y = compare Equal x y
  let not_equal x y = compare Not_equal x y
  let less x y = compare Less x y
  let less_equal x y = compare Less_equal x y
  let copy x = make1 Copy (dtype x) x
  let cast dt x = make1 Cast dt x
  let bitcast dt x = op1 Bitcast dt x
  let where c x y = make3 Where (dtype x) c x y
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
  let ( && ) x y = make2 (Binary And) (dtype x) x y
  let ( || ) x y = make2 (Binary Or) (dtype x) x y
  let ( = ) = equal
  let ( <> ) = not_equal
  let ( < ) = less
  let ( <= ) = less_equal
  let ( > ) x y = less y x
  let ( >= ) x y = less_equal y x
end

(* Each kind's code in nx_prog.h, whose macros number each type's
   constructors in the order prog.ml declares them. *)
let unary_code : unary -> int = function
  | Neg -> 0
  | Recip -> 1
  | Abs -> 2
  | Sign -> 3
  | Sqrt -> 4
  | Exp -> 5
  | Exp2 -> 6
  | Log -> 7
  | Log2 -> 8
  | Log1p -> 9
  | Expm1 -> 10
  | Sin -> 11
  | Cos -> 12
  | Tan -> 13
  | Asin -> 14
  | Acos -> 15
  | Atan -> 16
  | Sinh -> 17
  | Cosh -> 18
  | Tanh -> 19
  | Asinh -> 20
  | Acosh -> 21
  | Atanh -> 22
  | Erf -> 23
  | Floor -> 24
  | Ceil -> 25
  | Round -> 26
  | Trunc -> 27

let code1 = function
  | Copy -> 0
  | Cast -> 1
  | Bitcast -> 2
  | Unary u -> 3 + unary_code u

let binary_code : binary -> int = function
  | Add -> 0
  | Sub -> 1
  | Mul -> 2
  | Fdiv -> 3
  | Idiv -> 4
  | Mod -> 5
  | Pow -> 6
  | Atan2 -> 7
  | Maximum -> 8
  | Minimum -> 9
  | And -> 10
  | Or -> 11
  | Xor -> 12
  | Threefry -> 13

let code2 = function
  | Binary b -> binary_code b
  | Compare Equal -> 14
  | Compare Not_equal -> 15
  | Compare Less -> 16
  | Compare Less_equal -> 17

let code3 = function Where -> 0 | Fma -> 1

(* The codes of each arity's kinds: one past its last kind's. *)
let op1_codes = 1 + code1 (Unary Trunc)
let op2_codes = 1 + code2 (Compare Less_equal)
let op3_codes = 1 + code3 Fma

(* One-node programs *)

let dtypes = List.length D.all

(* The one-node programs made so far, by kind and dtypes, each made once: a
   map that finds its program builds none. A slot that two domains fill at
   once keeps either's program; they are equal. *)
let nodes1 : t option array =
  Array.make (op1_codes * dtypes * dtypes) None

let nodes2 : t option array = Array.make (op2_codes * dtypes) None

let nodes3 : t option array =
  Array.make (op3_codes * dtypes * dtypes) None

(* [p] kept at [table.(i)]. *)
let kept table i p =
  let p = Some p in
  table.(i) <- p;
  p

let node1 k x y =
  let i = (((code1 k * dtypes) + D.code x) * dtypes) + D.code y in
  match nodes1.(i) with
  | Some _ as p -> p
  | None when not (accepts1 k x y) -> None
  | None -> kept nodes1 i (of_node ~ins:[| D.Any x |] (Op1 (k, D.Any y, 0)))

let node2 k x =
  let i = (code2 k * dtypes) + D.code x in
  match nodes2.(i) with
  | Some _ as p -> p
  | None when not (accepts2 k x) -> None
  | None -> kept nodes2 i (of_node ~ins:[| D.Any x; D.Any x |] (Op2 (k, 0, 1)))

let node3 k c x =
  let i = (((code3 k * dtypes) + D.code c) * dtypes) + D.code x in
  match nodes3.(i) with
  | Some _ as p -> p
  | None when not (accepts3 k c x) -> None
  | None ->
      kept nodes3 i
        (of_node ~ins:[| D.Any c; D.Any x; D.Any x |] (Op3 (k, 0, 1, 2)))

(* Expressions checked their nodes as v would, and the walk makes each node
   after its operands: what remains is what an expression cannot know, its
   program's operands and outputs. *)

(* Whether [ins] has an operand [k] of dtype [dt]. *)
let reads ins k dt =
  k < Iarray.length ins
  &&
  let (D.Any d) = Iarray.get ins k in
  D.equal d dt

(* The walk keeps two tables: [seen] gives an expression walked before, by
   its id, its node, so that a shared subexpression is walked once;
   [index] gives a node and its dtype the index of the first node of that
   structure, so that two expressions of one structure are one node. An id
   is unique within one process alone: an expression read back through
   [Marshal] can carry the id of another here, so a hit in [seen] counts
   only for the expression itself. *)
let of_exprs ~ins outs =
  let invalid why = invalid_arg ("Nx_kernel.Prog.of_exprs: " ^ why) in
  if Iarray.length outs = 0 then invalid "no output";
  let seen = Hashtbl.create 16 and index = Hashtbl.create 16 in
  let nodes = ref [] and count = ref 0 in
  let node nd dt =
    match Hashtbl.find_opt index (nd, dt) with
    | Some i -> i
    | None ->
        Hashtbl.add index (nd, dt) !count;
        nodes := nd :: !nodes;
        incr count;
        !count - 1
  in
  let rec walk : type v s. (v, s) E.t -> int =
   fun e ->
    let id = E.id e in
    match Hashtbl.find_opt seen id with
    | Some (E.Any walked, i) when E.Any walked == E.Any e -> i
    | Some _ | None ->
        let nd =
          match e with
          | In { k; dt; _ } ->
              if not (reads ins k dt) then
                invalid (strf "In %d is no operand of ins" k);
              In k
          | Coord { axis; _ } -> Coord axis
          | Const { dt; bits; _ } -> Const (D.Any dt, bits)
          | Op1 { k; dt; x; _ } -> Op1 (k, D.Any dt, walk x)
          | Op2 { k; x; y; _ } ->
              let a = walk x in
              Op2 (k, a, walk y)
          | Op3 { k; c; x; y; _ } ->
              let a = walk c in
              let b = walk x in
              Op3 (k, a, b, walk y)
        in
        let i = node nd (D.Any (E.dtype e)) in
        Hashtbl.replace seen id (E.Any e, i);
        i
  in
  let out o =
    let (E.Any e) = Iarray.get outs o in
    walk e
  in
  let outs = Iarray.init (Iarray.length outs) out in
  { ins; nodes = Iarray.of_list (List.rev !nodes); outs }

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
