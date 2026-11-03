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
   sub-byte element's in the low bits of its byte. A bool byte is [true]
   iff it is not zero; a bit byte is 0 or 1. *)
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

  (* The constant of [dt] whose bits are [b], an element of [dt]. *)
  let of_bits dt b = Const { id = fresh (); dt; bits = b }
  let const dt x = of_bits dt (bits dt x)

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
  (* A cast into its operand's own dtype is the operand: Cast computes
     nothing there, and a node fewer lets a library run a kind alone
     (single). *)
  let cast (type v s w r) (dt : (w, r) D.t) (x : (v, s) t) : (w, r) t =
    match D.equal_witness (dtype x) dt with
    | Some Type.Equal -> x
    | None -> make1 Cast dt x
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

(* Expressions checked their nodes as v would, and the walk makes each node
   after its operands: what remains is what an expression cannot know, its
   program's operands and outputs. *)

(* Whether [ins] has an operand [k] of dtype [dt]. *)
let reads ins k dt =
  k < Iarray.length ins
  &&
  let (D.Any d) = Iarray.get ins k in
  D.equal d dt

(* [e]'s node where it reads no other node: an [In] of an operand of
   [ins], a [Coord] or a [Const]. An [In] of no operand of [ins] is
   [None], for the walk to refuse. *)
let leaf : type v s. D.any iarray -> (v, s) E.t -> node option =
 fun ins e ->
  match e with
  | In { k; dt; _ } when reads ins k dt -> Some (In k)
  | Coord { axis; _ } -> Some (Coord axis)
  | Const { dt; bits; _ } -> Some (Const (D.Any dt, bits))
  | In _ | Op1 _ | Op2 _ | Op3 _ -> None

(* Two nodes of one structure, compared field by field, the kind last. *)
let node_equal (a : node) (b : node) =
  match (a, b) with
  | In i, In j | Coord i, Coord j -> i = j
  | Const (D.Any x, s), Const (D.Any y, t) -> D.equal x y && String.equal s t
  | Op1 (k, D.Any x, i), Op1 (l, D.Any y, j) -> i = j && D.equal x y && k = l
  | Op2 (k, i, j), Op2 (l, i', j') -> i = i' && j = j' && k = l
  | Op3 (k, i, j, m), Op3 (l, i', j', m') -> i = i' && j = j' && m = m' && k = l
  | (In _ | Coord _ | Const _ | Op1 _ | Op2 _ | Op3 _), _ -> false

(* Nodes as a table's keys, hashed by their operands and compared field by
   field: the runtime's polymorphic hash and compare of a whole node,
   called for each node a lowering adds, cost a fifth of a draw from beta.
   [hash] mixes a node's operands and its leaf's facts; nodes over the same
   operands differ by their kind, which [equal] compares last, once the
   operands agree. *)
module Nodes = Hashtbl.Make (struct
  type t = node

  let equal = node_equal

  (* An odd multiplier keeps distinct operands apart in the low bits a
     table indexes by. *)
  let mix h x = (h * 0x9e3779b1) lxor x

  let hash = function
    | In k -> k
    | Coord a -> mix 1 a
    | Const (D.Any dt, b) -> mix (Hashtbl.hash b) (D.code dt)
    | Op1 (_, D.Any dt, a) -> mix a (D.code dt)
    | Op2 (_, a, b) -> mix a b
    | Op3 (_, a, b, c) -> mix (mix a b) c
end)

(* A lowering finds an expression walked before, so that a shared
   subexpression is walked once, and the first node of a node's structure,
   so that two expressions of one structure are one node. A node's dtype
   follows from its structure and the nodes before it, so the node alone is
   the key.

   Up to [small] nodes and [walked_bound] walked expressions it finds both
   by scanning [walked] and [nodes]: a table's buckets cost more than the
   scans of a small program, which most composites lower. Past either it
   builds two tables from them and keeps them: [seen] gives a walked
   expression, by its id, its node, and [index] a node its index. An id is
   unique within one process alone: an expression read back through
   [Marshal] can carry the id of another here, so a hit in [seen] counts
   only for the expression itself, as a scan compares the expression
   itself. *)
type tables = { seen : (int, E.any * int) Hashtbl.t; index : int Nodes.t }

type lowering = {
  fn : string;
  operands : D.any iarray;
  mutable nodes : node array;  (* The program's nodes: the first [count]. *)
  mutable count : int;
  mutable walked : E.any array;  (* Scanned: the first [nwalked]. *)
  mutable at : int array;  (* Each walked expression's node. *)
  mutable nwalked : int;
  mutable tables : tables option;
}

(* Up to 16 nodes the scans take no longer than the tables; at 40 they
   take about 0.8 us longer, and save the tables' 400 to 450 words. *)
let small = 40

(* The walked expressions a scan holds at most. *)
let walked_bound = 2 * small

(* [a] of [n] elements in an array twice as long, filled with [fill]. *)
let grown a n fill =
  let b = Array.make (max 8 (2 * n)) fill in
  Array.blit a 0 b 0 n;
  b

(* [l]'s tables, of about [n] nodes, holding what it found so far. *)
let tables_of l n =
  let seen = Hashtbl.create n and index = Nodes.create n in
  for i = 0 to l.nwalked - 1 do
    let (E.Any e) = l.walked.(i) in
    Hashtbl.replace seen (E.id e) (l.walked.(i), l.at.(i))
  done;
  for i = 0 to l.count - 1 do
    Nodes.add index l.nodes.(i) i
  done;
  l.walked <- [||];
  l.at <- [||];
  l.nwalked <- 0;
  { seen; index }

(* A lowering over [ins] of about [n] nodes, refusing in the name of [fn]:
   with its tables from the start where [n] is past [small]. *)
let lowering fn ~ins n =
  let l =
    {
      fn;
      operands = ins;
      nodes = Array.make (max 8 n) (In 0);
      count = 0;
      walked = [||];
      at = [||];
      nwalked = 0;
      tables = None;
    }
  in
  if n > small then l.tables <- Some (tables_of l n);
  l

let refuse_in l k =
  invalid_arg (strf "%s: In %d is no operand of ins" l.fn k)

let rec find_node l nd i =
  if i = l.count then -1
  else if node_equal (Array.unsafe_get l.nodes i) nd then i
  else find_node l nd (i + 1)

(* The index of the node [nd], added unless one of its structure is. *)
let add l nd =
  let found =
    match l.tables with
    | None -> find_node l nd 0
    | Some t -> ( match Nodes.find_opt t.index nd with Some i -> i | None -> -1)
  in
  if found >= 0 then found
  else begin
    let i = l.count in
    if i = Array.length l.nodes then l.nodes <- grown l.nodes i nd;
    l.nodes.(i) <- nd;
    l.count <- i + 1;
    (match l.tables with
    | Some t -> Nodes.add t.index nd i
    | None -> if l.count > small then l.tables <- Some (tables_of l (2 * i)));
    i
  end

let rec find_walked l e i =
  if i = l.nwalked then -1
  else if Array.unsafe_get l.walked i == e then Array.unsafe_get l.at i
  else find_walked l e (i + 1)

(* The node of [e], walked before, or [-1]. *)
let walked_before (type v s) l (e : (v, s) E.t) =
  match l.tables with
  | None -> find_walked l (E.Any e) 0
  | Some t -> (
      match Hashtbl.find_opt t.seen (E.id e) with
      | Some (walked, i) when walked == E.Any e -> i
      | Some _ | None -> -1)

(* Records that [e] is node [i]. *)
let walk_done (type v s) l (e : (v, s) E.t) i =
  match l.tables with
  | Some t -> Hashtbl.replace t.seen (E.id e) (E.Any e, i)
  | None ->
      let k = l.nwalked in
      if k = Array.length l.walked then begin
        l.walked <- grown l.walked k (E.Any e);
        l.at <- grown l.at k 0
      end;
      l.walked.(k) <- E.Any e;
      l.at.(k) <- i;
      l.nwalked <- k + 1;
      (* Fresh copies of one structure walk many expressions into few
         nodes: the scan is bounded by them too. *)
      if l.nwalked > walked_bound then
        l.tables <- Some (tables_of l (2 * l.nwalked))

let rec walk : type v s. lowering -> (v, s) E.t -> int =
 fun l e ->
  let i = walked_before l e in
  if i >= 0 then i
  else
    let nd =
      match (leaf l.operands e, e) with
      | Some nd, _ -> nd
      | None, In { k; _ } -> refuse_in l k
      | None, Op1 { k; dt; x; _ } -> Op1 (k, D.Any dt, walk l x)
      | None, Op2 { k; x; y; _ } ->
          let a = walk l x in
          Op2 (k, a, walk l y)
      | None, Op3 { k; c; x; y; _ } ->
          let a = walk l c in
          let b = walk l x in
          Op3 (k, a, b, walk l y)
      | None, (Coord _ | Const _) -> assert false
    in
    let i = add l nd in
    walk_done l e i;
    i

let lowered l outs =
  let nodes = l.nodes in
  { ins = l.operands; nodes = Iarray.init l.count (Array.get nodes); outs }

(* One output that reads no other node, as a fill's constant is, is its one
   node: no table is needed to share it. *)
let lower fn ~ins outs =
  if Iarray.length outs = 0 then invalid_arg (fn ^ ": no output");
  let (E.Any first) = Iarray.get outs 0 in
  match if Iarray.length outs = 1 then leaf ins first else None with
  | Some n -> { ins; nodes = [| n |]; outs = [| 0 |] }
  | None ->
      let l = lowering fn ~ins 16 in
      let out o =
        let (E.Any e) = Iarray.get outs o in
        walk l e
      in
      lowered l (Iarray.init (Iarray.length outs) out)

let of_exprs ~ins outs = lower "Nx_kernel.Prog.of_exprs" ~ins outs

(* Substitution *)

(* The nodes the outputs reach are lowered as of_exprs lowers the
   expressions they stand for, without making those: each is added once by
   its index in [p], after its operands in depth-first order, and a leaf's
   substitute is walked into the same tables. A draw binds its position
   into its sampler's program, of thousands of nodes for beta, this way. *)
let subst ?outs ~ins f p =
  let fn = "Nx_kernel.Prog.subst" in
  let picked =
    match outs with
    | None -> p.outs
    | Some outs ->
        let n = Iarray.length p.outs in
        if Iarray.length outs = 0 then invalid_arg (fn ^ ": no output");
        let pick o =
          if o < 0 || o >= n then invalid_arg (strf "%s: no output %d" fn o);
          Iarray.get p.outs o
        in
        Iarray.map pick outs
  in
  let n = Iarray.length p.nodes in
  let l = lowering fn ~ins n in
  (* Node [i]'s index in the result, [-1] until it is added. *)
  let index = Array.make n (-1) in
  (* Leaf [i], [nd], of dtype [dt]. *)
  let leaf i nd (D.Any dt) =
    match f nd with
    | Some (E.Any e) ->
        if not (D.equal (E.dtype e) dt) then
          invalid_arg
            (strf "%s: node %d's expression is not of its dtype" fn i);
        walk l e
    | None -> (
        match nd with
        | In k when not (reads ins k dt) -> refuse_in l k
        | _ -> add l nd)
  in
  let rec go i =
    if index.(i) >= 0 then index.(i)
    else
      let r =
        match Iarray.get p.nodes i with
        | In k as nd -> leaf i nd (Iarray.get p.ins k)
        | Coord _ as nd -> leaf i nd (D.Any D.Int64)
        | Const (dt, _) as nd -> leaf i nd dt
        | Op1 (k, dt, a) ->
            let a = go a in
            add l (Op1 (k, dt, a))
        | Op2 (k, a, b) ->
            let a = go a in
            let b = go b in
            add l (Op2 (k, a, b))
        | Op3 (k, a, b, c) ->
            let a = go a in
            let b = go b in
            let c = go c in
            add l (Op3 (k, a, b, c))
      in
      index.(i) <- r;
      r
  in
  lowered l (Iarray.map go picked)

(* Ranges *)

(* A node's range in the program [ranges] builds: the nodes of its least and
   greatest values, or [Unranged] for a float or complex node, whose
   ranges no rule follows. *)
type range = Unranged | Range of int * int

let ranged (D.Any dt) =
  match D.kind dt with
  | Signed | Unsigned | Boolean -> true
  | Float | Complex -> false

let is_signed (D.Any dt) = D.is D.Signed dt
let width (D.Any dt) = D.bits dt

(* The node of the integer or boolean constant [x] of [dt], within its
   range; an unsigned 32-bit one's bits from [x]'s low half. *)
let int_bits (D.Any dt) (x : int64) =
  let b : type v s. (v, s) D.t -> string =
   fun dt ->
    match dt with
    | Int64 -> bits dt x
    | Uint64 -> bits dt x
    | Int32 -> bits dt (Int64.to_int32 x)
    | Uint32 -> bits dt (Int64.to_int32 x)
    | Int16 -> bits dt (Int64.to_int x)
    | Uint16 -> bits dt (Int64.to_int x)
    | Int8 -> bits dt (Int64.to_int x)
    | Uint8 -> bits dt (Int64.to_int x)
    | Int4 -> bits dt (Int64.to_int x)
    | Uint4 -> bits dt (Int64.to_int x)
    | Bool -> bits dt (x <> 0L)
    | Bit -> bits dt (x <> 0L)
    | Float64 | Float32 | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2
    | Float4_e2m1fn | Complex128 | Complex64 ->
        invalid_arg "Nx_kernel.Prog.ranges: a float constant"
  in
  Const (D.Any dt, b dt)

let extreme (D.Any dt) ~high =
  Const (D.Any dt, bits dt (if high then D.max_value dt else D.min_value dt))

(* The bounds a cast from the integer dtype [s] to the integer dtype [t]
   must check, as values of [s]: [t]'s least where it is above [s]'s, and
   its greatest where it is below [s]'s. *)
let cast_checks s t =
  let ws = width s and wt = width t in
  let smax w = Int64.pred (Int64.shift_left 1L (w - 1)) in
  let umax w = Int64.pred (Int64.shift_left 1L w) in
  match (is_signed s, is_signed t) with
  | true, true when wt < ws ->
      (Some (Int64.neg (Int64.shift_left 1L (wt - 1))), Some (smax wt))
  | true, true -> (None, None)
  | true, false -> (Some 0L, if wt < ws then Some (umax wt) else None)
  | false, false -> (None, if wt < ws then Some (umax wt) else None)
  | false, true -> (None, if wt <= ws then Some (smax wt) else None)

(* Each node's range follows from its operands' by its kind's rule, over
   the values of its dtype as its kind computes them:

     a + b, a - b, a · b   the ends' sums, differences, the corners'
                           products; the whole range where one may wrap
     a / b, a mod b        where b's range holds no 0: the corners'
                           quotients, a remainder within b's magnitude
     -a, |a|, sign a       the ends', the whole range at a signed least
     max, min, and, or     the ends' (on booleans and is min, or max)
     comparisons           true at every pair where the ranges are apart,
                           false where they cannot meet
     cast                  the ends' where the range fits the dtype, and
                           to bool, true where 0 is outside it
     where c a b           a's, b's, or their union where c is unknown
     any other kind        its value where each operand's range is one
                           value, the whole range otherwise

   A float or complex node has no range, and an integer or boolean node
   computed from one, as a comparison of floats, has its dtype's whole
   range. *)
let ranges ~axes ?outs p =
  let fn = "Nx_kernel.Prog.ranges" in
  if axes < 0 || axes > Nx_array.Layout.max_rank then
    invalid_arg (strf "%s: %d axes" fn axes);
  let picked =
    match outs with
    | None -> p.outs
    | Some outs ->
        let n = Iarray.length p.outs in
        if Iarray.length outs = 0 then invalid_arg (fn ^ ": no output");
        let pick o =
          if o < 0 || o >= n then invalid_arg (strf "%s: no output %d" fn o);
          Iarray.get p.outs o
        in
        Iarray.map pick outs
  in
  let nin = Iarray.length p.ins in
  let ins =
    Iarray.init
      ((2 * nin) + (2 * axes))
      (fun k -> if k < 2 * nin then Iarray.get p.ins (k / 2) else D.Any D.Int64)
  in
  let n = Iarray.length p.nodes in
  let l = lowering fn ~ins (4 * n) in
  let node nd = add l nd in
  let op1 k dt a = node (Op1 (k, dt, a)) in
  let bin k a b = node (Op2 (Binary k, a, b)) in
  let cmp k a b = node (Op2 (Compare k, a, b)) in
  let where c a b = node (Op3 (Where, c, a, b)) in
  let int dt x = node (int_bits dt x) in
  let truth x = int (D.Any D.Bool) (if x then 1L else 0L) in
  let conj a b = bin And a b and disj a b = bin Or a b in
  let negation a = cmp Equal a (truth false) in
  let low dt = node (extreme dt ~high:false) in
  let high dt = node (extreme dt ~high:true) in
  let top dt = if ranged dt then Range (low dt, high dt) else Unranged in
  let guard dt ok (a, b) = Range (where ok a (low dt), where ok b (high dt)) in
  let exact (a, b) = cmp Equal a b in
  (* Whether [s], [x] + [y] or [x] - [y] wrapped, did not wrap. *)
  let added dt x y s =
    if is_signed dt then cmp Equal (cmp Less y (int dt 0L)) (cmp Less s x)
    else cmp Less_equal x s
  in
  let subtracted dt x y s =
    if is_signed dt then cmp Equal (cmp Less y (int dt 0L)) (cmp Less x s)
    else cmp Less_equal y x
  in
  (* Whether [s], [x] · [y] wrapped, did not wrap: then [s / x] is [y],
     since a wrap moves the product by more than [|x|]; but a signed least
     value by [-1] gives itself. *)
  let multiplied dt x y s =
    let back = cmp Equal (bin Idiv s x) y in
    let fits =
      if not (is_signed dt) then back
      else
        let least = conj (cmp Equal x (int dt (-1L))) (cmp Equal y (low dt)) in
        conj back (negation least)
    in
    disj (cmp Equal x (int dt 0L)) fits
  in
  let least4 a b c d = bin Minimum (bin Minimum a b) (bin Minimum c d) in
  let most4 a b c d = bin Maximum (bin Maximum a b) (bin Maximum c d) in
  (* The corners of [k] over [a, b] and [c, d]: their least and greatest. *)
  let corners k (a, b) (c, d) =
    let ac = bin k a c and ad = bin k a d in
    let bc = bin k b c and bd = bin k b d in
    ((ac, ad, bc, bd), (least4 ac ad bc bd, most4 ac ad bc bd))
  in
  let product dt x y =
    let (ac, ad, bc, bd), ends = corners Mul x y in
    let (a, b), (c, d) = (x, y) in
    let ok =
      conj
        (conj (multiplied dt a c ac) (multiplied dt a d ad))
        (conj (multiplied dt b c bc) (multiplied dt b d bd))
    in
    (ok, ends)
  in
  let sum dt (a, b) (c, d) =
    let lo = bin Add a c and hi = bin Add b d in
    (conj (added dt a c lo) (added dt b d hi), (lo, hi))
  in
  (* [k] at the least values, where every operand's range is one value. *)
  let fallback dt nd rs =
    if not (ranged dt) then Unranged
    else if List.exists (fun r -> r = Unranged) rs then top dt
    else
      let ends =
        List.map
          (function Range (a, b) -> (a, b) | Unranged -> assert false)
          rs
      in
      let ok =
        List.fold_left (fun ok e -> conj ok (exact e)) (truth true) ends
      in
      let lo = List.map fst ends in
      let v =
        match (nd, lo) with
        | Op1 (k, dt, _), [ a ] -> op1 k dt a
        | Op2 (k, _, _), [ a; b ] -> node (Op2 (k, a, b))
        | Op3 (k, _, _, _), [ a; b; c ] -> node (Op3 (k, a, b, c))
        | _ -> assert false
      in
      guard dt ok (v, v)
  in
  let r = Array.make n None in
  let rec go i =
    match r.(i) with
    | Some x -> x
    | None ->
        let x = rule i (Iarray.get p.nodes i) in
        r.(i) <- Some x;
        x
  and rule i nd =
    let dt = dtype_of p i in
    match nd with
    | In k ->
        if ranged dt then Range (node (In (2 * k)), node (In ((2 * k) + 1)))
        else Unranged
    | Coord a ->
        if a >= axes then invalid_arg (strf "%s: node %d reads axis %d" fn i a);
        let k = (2 * nin) + (2 * a) in
        Range (node (In k), node (In (k + 1)))
    | Const _ ->
        if ranged dt then
          let c = node nd in
          Range (c, c)
        else Unranged
    | Op1 (k, _, x) -> unary dt nd k x
    | Op2 (Compare k, x, y) -> compare dt nd k x y
    | Op2 (Binary k, x, y) -> binary dt nd k x y
    | Op3 (Where, c, x, y) -> (
        match (go c, go x, go y) with
        | Range (cl, ch), Range (xl, xh), Range (yl, yh) ->
            let lo = where cl xl (where ch (bin Minimum xl yl) yl) in
            let hi = where cl xh (where ch (bin Maximum xh yh) yh) in
            Range (lo, hi)
        | _ -> top dt)
    | Op3 (Fma, x, y, z) -> (
        match (go x, go y, go z) with
        | Range (a, b), Range (c, d), Range (e, f) ->
            let mok, ends = product dt (a, b) (c, d) in
            let sok, ends = sum dt ends (e, f) in
            guard dt (conj mok sok) ends
        | _ -> top dt)
  and unary dt nd k x =
    let sx = dtype_of p x in
    match (k, go x) with
    | _, Unranged -> top dt
    | Copy, r -> r
    | Cast, Range (a, b) ->
        if not (ranged dt) then Unranged
        else if D.(let (Any t) = dt in is Boolean t) then
          if D.(let (Any s) = sx in is Boolean s) then
            Range (op1 Cast dt a, op1 Cast dt b)
          else
            let z = int sx 0L in
            let apart = disj (cmp Less z a) (cmp Less b z) in
            let zero = conj (cmp Equal a z) (cmp Equal b z) in
            (* Both ends are bool's: a cast gives them bit's. *)
            Range (op1 Cast dt apart, op1 Cast dt (negation zero))
        else if D.(let (Any s) = sx in is Boolean s) then
          Range (op1 Cast dt a, op1 Cast dt b)
        else
          let ends = (op1 Cast dt a, op1 Cast dt b) in
          let ok =
            match cast_checks sx dt with
            | None, None -> None
            | Some lo, None -> Some (cmp Less_equal (int sx lo) a)
            | None, Some hi -> Some (cmp Less_equal b (int sx hi))
            | Some lo, Some hi ->
                Some
                  (conj (cmp Less_equal (int sx lo) a)
                     (cmp Less_equal b (int sx hi)))
          in
          (match ok with
          | None -> Range (fst ends, snd ends)
          | Some ok -> guard dt ok ends)
    | Unary Neg, Range (a, b) when is_signed dt ->
        guard dt (negation (cmp Equal a (low dt))) (op1 k dt b, op1 k dt a)
    | Unary Abs, Range (a, b) when is_signed dt ->
        let z = int dt 0L in
        let neg x = op1 (Unary Neg) dt x in
        let lo = bin Maximum (bin Maximum a (neg b)) z in
        guard dt (negation (cmp Equal a (low dt))) (lo, bin Maximum (neg a) b)
    | Unary (Abs | Floor | Ceil | Round | Trunc), r -> r
    | Unary Sign, Range (a, b) -> Range (op1 k dt a, op1 k dt b)
    | Bitcast, _ when D.(let (Any s) = sx in is Boolean s) ->
        (* A boolean's byte is any byte, true where it is not 0. *)
        top dt
    | (Unary _ | Bitcast), r -> fallback dt nd [ r ]
  and compare dt nd k x y =
    match (go x, go y) with
    | Range (a, b), Range (c, d) -> (
        let lt = cmp Less and le = cmp Less_equal in
        let equal () =
          conj (conj (exact (a, b)) (exact (c, d))) (cmp Equal a c)
        in
        match k with
        | Less -> Range (lt b c, lt a d)
        | Less_equal -> Range (le b c, le a d)
        | Equal -> Range (equal (), conj (le a d) (le c b))
        | Not_equal -> Range (disj (lt b c) (lt d a), negation (equal ())))
    | rx, ry -> fallback dt nd [ rx; ry ]
  and binary dt nd k x y =
    match (k, go x, go y) with
    | _, Unranged, _ | _, _, Unranged -> top dt
    | Add, Range (a, b), Range (c, d) ->
        let ok, ends = sum dt (a, b) (c, d) in
        guard dt ok ends
    | Sub, Range (a, b), Range (c, d) ->
        let lo = bin Sub a d and hi = bin Sub b c in
        guard dt (conj (subtracted dt a d lo) (subtracted dt b c hi)) (lo, hi)
    | Mul, Range (a, b), Range (c, d) ->
        let ok, ends = product dt (a, b) (c, d) in
        guard dt ok ends
    | Idiv, Range (a, b), Range (c, d) ->
        let z = int dt 0L in
        if is_signed dt then
          let apart = disj (cmp Less z c) (cmp Less d z) in
          let m1 = int dt (-1L) in
          let least =
            conj (cmp Equal a (low dt))
              (conj (cmp Less_equal c m1) (cmp Less_equal m1 d))
          in
          let _, ends = corners Idiv (a, b) (c, d) in
          guard dt (conj apart (negation least)) ends
        else guard dt (cmp Less z c) (bin Idiv a d, bin Idiv b c)
    | Mod, Range (a, b), Range (c, d) ->
        let z = int dt 0L and one = int dt 1L in
        if is_signed dt then
          let apart = disj (cmp Less z c) (cmp Less d z) in
          let neg x = op1 (Unary Neg) dt x in
          let m = bin Maximum (bin Sub d one) (neg (bin Add c one)) in
          let lo = where (cmp Less_equal z a) z (bin Maximum a (neg m)) in
          let hi = where (cmp Less_equal b z) z (bin Minimum b m) in
          guard dt apart (lo, hi)
        else guard dt (cmp Less z c) (z, bin Minimum b (bin Sub d one))
    | (Maximum | Minimum), Range (a, b), Range (c, d) ->
        Range (bin k a c, bin k b d)
    | (And | Or), Range (a, b), Range (c, d)
      when D.(let (Any t) = dt in is Boolean t) ->
        Range (bin k a c, bin k b d)
    | _, rx, ry -> fallback dt nd [ rx; ry ]
  in
  let out o =
    match go o with
    | Range (a, b) -> (a, b)
    | Unranged ->
        invalid_arg (strf "%s: node %d is a float or complex output" fn o)
  in
  let pairs = Iarray.map out picked in
  let outs =
    Iarray.init
      (2 * Iarray.length pairs)
      (fun k ->
        let a, b = Iarray.get pairs (k / 2) in
        if k mod 2 = 0 then a else b)
  in
  lowered l outs
