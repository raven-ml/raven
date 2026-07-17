open Tolk_uop

type node = Uop.t
type program = Tolk.Program_spec.program

type op = [
  | `Add | `Sub | `Mul | `Fdiv | `Cdiv | `Cmod | `Floordiv | `Floormod | `Max
  | `Cmplt | `Cmpeq | `Cmpne
  | `And | `Or | `Xor | `Shl | `Shr
  | `Neg | `Exp2 | `Log2 | `Sin | `Sqrt | `Recip | `Trunc
  | `Where | `Mulacc
]

type instr =
  | Param of { slot : int; dtype : Dtype.t }
  | Const of { value : Const.t; dtype : Dtype.t }
  | Index of {
      ptr : node;
      idxs : node list;
      dtype : Dtype.t;
    }
  | Load of {
      src : node;
      alt : node option;
      gate : node option;
      dtype : Dtype.t;
    }
  | Store of { dst : node; value : node; gate : node option }
  | Binary of { op : op; lhs : node; rhs : node; dtype : Dtype.t }
  | Unary of { op : op; src : node; dtype : Dtype.t }
  | Ternary of {
      op : op;
      a : node;
      b : node;
      c : node;
      dtype : Dtype.t;
    }
  | Cast of { src : node; dtype : Dtype.t }
  | Bitcast of { src : node; dtype : Dtype.t }
  | Range of {
      size : node;
      dtype : Dtype.t;
      axis : int;
      sub : int list;
      kind : Axis_type.t;
    }
  | End_range of { dep : node; range : node }
  | Barrier
  | Buffer of {
      slot : int option;
      size : int;
      dtype : Dtype.t;
      addrspace : Dtype.addr_space;
    }
  | Special of { dim : Tolk.Gpu_dim.t; size : node; dtype : Dtype.t }
  | If of { cond : node; idx_for_dedup : node }
  | Endif of { if_ : node }
  | Stack of { srcs : node list; dtype : Dtype.t }
  | Value_index of { src : node; idxs : node list; dtype : Dtype.t }
  | Variable of {
      name : string;
      min_val : int;
      max_val : int;
      dtype : Dtype.t;
    }
  | Custom_inline of { fmt : string; args : node list; dtype : Dtype.t }

type t = program ref

let create () = ref []
let append b node = b := node :: !b; node

let invalid_dtype instr expected actual =
  invalid_arg
    (Printf.sprintf "%s expected %s, got %s" instr expected actual)

let expect_dtype instr expected node =
  let actual = Uop.dtype node in
  if Dtype.equal actual expected then node
  else
    invalid_dtype instr (Dtype.to_string expected) (Dtype.to_string actual)

let ops = function
  | `Add -> Ops.Add
  | `Sub -> Ops.Sub
  | `Mul -> Ops.Mul
  | `Fdiv -> Ops.Fdiv
  | `Cdiv -> Ops.Cdiv
  | `Cmod -> Ops.Cmod
  | `Floordiv -> Ops.Floordiv
  | `Floormod -> Ops.Floormod
  | `Max -> Ops.Max
  | `Cmplt -> Ops.Cmplt
  | `Cmpeq -> Ops.Cmpeq
  | `Cmpne -> Ops.Cmpne
  | `And -> Ops.And
  | `Or -> Ops.Or
  | `Xor -> Ops.Xor
  | `Shl -> Ops.Shl
  | `Shr -> Ops.Shr
  | `Neg -> Ops.Neg
  | `Exp2 -> Ops.Exp2
  | `Log2 -> Ops.Log2
  | `Sin -> Ops.Sin
  | `Sqrt -> Ops.Sqrt
  | `Recip -> Ops.Reciprocal
  | `Trunc -> Ops.Trunc
  | `Where -> Ops.Where
  | `Mulacc -> Ops.Mulacc

let scalar_index = function
  | [ idx ] -> idx
  | idxs -> Uop.stack idxs

let emit b instr =
  let node =
    match instr with
    | Param { slot; dtype } ->
        Uop.param ~slot ~dtype ~shape:(Uop.const_int (-1)) ()
    | Const { value; dtype } ->
        if Dtype.equal (Const.dtype value) dtype then Uop.const value
        else
          invalid_dtype "Const" (Dtype.to_string dtype)
            (Dtype.to_string (Const.dtype value))
    | Index { ptr; idxs; dtype } ->
        Uop.index ~ptr ~idxs:[(scalar_index idxs)] ()
        |> expect_dtype "Index" dtype
    | Load { src; alt; gate; dtype } ->
        Uop.load ~src ?alt ?gate () |> expect_dtype "Load" dtype
    | Store { dst; value; gate } -> Uop.store ~dst ~value ?gate ()
    | Binary { op; lhs; rhs; dtype } ->
        Uop.alu_binary ~op:(ops op) ~lhs ~rhs
        |> expect_dtype "Binary" dtype
    | Unary { op; src; dtype } ->
        Uop.alu_unary ~op:(ops op) ~src |> expect_dtype "Unary" dtype
    | Ternary { op; a; b; c; dtype } ->
        Uop.alu_ternary ~op:(ops op) ~a ~b ~c
        |> expect_dtype "Ternary" dtype
    | Cast { src; dtype } -> Uop.cast ~src ~dtype
    | Bitcast { src; dtype } -> Uop.bitcast ~src ~dtype
    | Range { size; dtype; axis; sub; kind } ->
        Uop.range ~size ~axis ~kind ~sub ~dtype ()
    | End_range { dep; range } -> Uop.end_ ~value:dep ~ranges:[ range ]
    | Barrier -> Uop.barrier ()
    | Buffer { slot; size; dtype; addrspace } ->
        let slot = Option.value slot ~default:(List.length !b) in
        Uop.buffer ~slot ~dtype ~shape:(Uop.const_int size) ~addrspace ()
    | Special { dim; size; dtype } ->
        Uop.special ~name:(Tolk.Gpu_dim.to_special_name dim) ~size ~dtype ()
    | If { cond; idx_for_dedup } -> Uop.if_ ~cond ~idx_for_dedup
    | Endif { if_ } -> Uop.endif ~if_
    | Stack { srcs; dtype } -> Uop.stack ~dtype srcs
    | Value_index { src; idxs; dtype } ->
        Uop.index ~ptr:src ~idxs:[(scalar_index idxs)] ()
        |> expect_dtype "Value_index" dtype
    | Variable { name; min_val; max_val; dtype } ->
        Uop.variable ~name ~min_val ~max_val ~dtype ()
    | Custom_inline { fmt; args; dtype } -> Uop.custom_inline ~fmt ~args ~dtype
  in
  append b node

let finish b = List.rev !b
