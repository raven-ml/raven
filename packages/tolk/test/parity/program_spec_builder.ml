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
  | Param of { slot : int; dtype : Dtype.Ptr.t }
  | Const of { value : Const.t; dtype : Dtype.Val.t }
  | Index of {
      ptr : node;
      idxs : node list;
      dtype : Dtype.Ptr.t;
    }
  | Load of {
      src : node;
      alt : node option;
      gate : node option;
      dtype : Dtype.Val.t;
    }
  | Store of { dst : node; value : node; gate : node option }
  | Binary of { op : op; lhs : node; rhs : node; dtype : Dtype.Val.t }
  | Unary of { op : op; src : node; dtype : Dtype.Val.t }
  | Ternary of {
      op : op;
      a : node;
      b : node;
      c : node;
      dtype : Dtype.Val.t;
    }
  | Cast of { src : node; dtype : Dtype.Val.t }
  | Bitcast of { src : node; dtype : Dtype.Val.t }
  | Range of {
      size : node;
      dtype : Dtype.Val.t;
      axis : int;
      sub : int list;
      kind : Axis_type.t;
    }
  | End_range of { dep : node; range : node }
  | Barrier
  | Buffer of { slot : int option; size : int; dtype : Dtype.Ptr.t }
  | Special of { dim : Tolk.Gpu_dim.t; size : node; dtype : Dtype.Val.t }
  | If of { cond : node; idx_for_dedup : node }
  | Endif of { if_ : node }
  | Stack of { srcs : node list; dtype : Dtype.Val.t }
  | Value_index of { src : node; idxs : node list; dtype : Dtype.Val.t }
  | Variable of {
      name : string;
      min_val : int;
      max_val : int;
      dtype : Dtype.Val.t;
    }
  | Custom_inline of { fmt : string; args : node list; dtype : Dtype.Val.t }

type t = program ref

let create () = ref []
let append b node = b := node :: !b; node

let invalid_dtype instr expected actual =
  invalid_arg
    (Printf.sprintf "%s expected %s, got %s" instr expected actual)

let expect_val_dtype instr expected node =
  match Uop.dtype node with
  | Dtype.Val actual when Dtype.Val.equal actual expected -> node
  | Dtype.Val actual ->
      invalid_dtype instr (Dtype.Val.to_string expected)
        (Dtype.Val.to_string actual)
  | Dtype.Ptr actual ->
      invalid_dtype instr (Dtype.Val.to_string expected)
        (Dtype.Ptr.to_string actual)

let expect_ptr_dtype instr expected node =
  match Uop.dtype node with
  | Dtype.Ptr actual when Dtype.Ptr.equal actual expected -> node
  | Dtype.Ptr actual ->
      invalid_dtype instr (Dtype.Ptr.to_string expected)
        (Dtype.Ptr.to_string actual)
  | Dtype.Val actual ->
      invalid_dtype instr (Dtype.Ptr.to_string expected)
        (Dtype.Val.to_string actual)

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
    | Param { slot; dtype } -> Uop.param ~slot ~dtype:(Dtype.Ptr dtype) ()
    | Const { value; dtype } ->
        if Dtype.Val.equal (Const.dtype value) dtype then Uop.const value
        else
          invalid_dtype "Const" (Dtype.Val.to_string dtype)
            (Dtype.Val.to_string (Const.dtype value))
    | Index { ptr; idxs; dtype } ->
        Uop.index ~ptr ~idxs:[(scalar_index idxs)] ~as_ptr:true ()
        |> expect_ptr_dtype "Index" dtype
    | Load { src; alt; gate; dtype } ->
        Uop.load ~src ?alt ?gate () |> expect_val_dtype "Load" dtype
    | Store { dst; value; gate } -> Uop.store ~dst ~value ?gate ()
    | Binary { op; lhs; rhs; dtype } ->
        Uop.alu_binary ~op:(ops op) ~lhs ~rhs
        |> expect_val_dtype "Binary" dtype
    | Unary { op; src; dtype } ->
        Uop.alu_unary ~op:(ops op) ~src |> expect_val_dtype "Unary" dtype
    | Ternary { op; a; b; c; dtype } ->
        Uop.alu_ternary ~op:(ops op) ~a ~b ~c
        |> expect_val_dtype "Ternary" dtype
    | Cast { src; dtype } -> Uop.cast ~src ~dtype:(Dtype.Val dtype)
    | Bitcast { src; dtype } -> Uop.bitcast ~src ~dtype:(Dtype.Val dtype)
    | Range { size; dtype; axis; sub; kind } ->
        Uop.range ~size ~axis ~kind ~sub ~dtype ()
    | End_range { dep; range } -> Uop.end_ ~value:dep ~ranges:[ range ]
    | Barrier -> Uop.barrier ()
    | Buffer { slot; size; dtype } ->
        let dtype =
          if Dtype.Ptr.size dtype = size then dtype
          else Dtype.Ptr.create (Dtype.Ptr.base dtype)
              ~addrspace:(Dtype.Ptr.addrspace dtype) ~size
        in
        let slot = Option.value slot ~default:(List.length !b) in
        Uop.buffer ~slot ~dtype:(Dtype.Ptr dtype) ~shape:(Uop.const_int size)
          ~addrspace:(Dtype.Ptr.addrspace dtype) ()
    | Special { dim; size; dtype } ->
        Uop.special ~name:(Tolk.Gpu_dim.to_special_name dim) ~size ~dtype ()
    | If { cond; idx_for_dedup } -> Uop.if_ ~cond ~idx_for_dedup
    | Endif { if_ } -> Uop.endif ~if_
    | Stack { srcs; dtype } -> Uop.stack ~dtype srcs
    | Value_index { src; idxs; dtype } ->
        Uop.index ~ptr:src ~idxs:[(scalar_index idxs)] ()
        |> expect_val_dtype "Value_index" dtype
    | Variable { name; min_val; max_val; dtype } ->
        Uop.variable ~name ~min_val ~max_val ~dtype ()
    | Custom_inline { fmt; args; dtype } -> Uop.custom_inline ~fmt ~args ~dtype
  in
  append b node

let finish b = List.rev !b
