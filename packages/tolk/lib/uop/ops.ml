(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Declaration order is load-bearing: toposort, compare, and commutative
   operand canonicalisation read it as the ordinal. Keep this order identical
   to tinygrad.uop.Ops. *)
type t =
  (* 1 -- defines/special *)
  | Bind
  | Special
  | Buffer
  (* 2 -- non-op uops *)
  | Noop
  | Rewrite_error
  | Param
  | Function
  | Call
  | Program
  | Linear
  | Source
  | Binary
  | Sink
  | After
  | Group
  | Stack
  | Tuple
  | Gettuple
  | Getaddr
  (* 3 -- load/store *)
  | Index
  | Shrink
  | Load
  | Store
  (* 4 -- math *)
  | Wmma
  | Cast
  | Bitcast
  | Exp2
  | Log2
  | Sin
  | Sqrt
  | Reciprocal
  | Neg
  | Trunc
  | Add
  | Mul
  | Shl
  | Shr
  | Cdiv
  | Max
  | Cmod
  | Cmplt
  | Cmpne
  | Cmpeq
  | Xor
  | Or
  | And
  | Threefry
  | Sub
  | Fdiv
  | Pow
  | Floordiv
  | Floormod
  | Where
  | Mulacc
  (* 5 -- control flow / consts / custom *)
  | Barrier
  | Range
  | If
  | End
  | Endif
  | Wait
  | Const
  | Custom
  | Customi
  | Ins
  (* 6 -- ops that don't exist in programs *)
  | Contiguous
  | Contiguous_backward
  | Detach
  | Stage
  | Copy
  | Slice
  | Mselect
  | Mstack
  | Custom_function
  | Reshape
  | Permute
  | Expand
  | Pad
  | Flip
  | Multi
  | Reduce
  | Allreduce
  (* 7 -- pattern compiler IR *)
  | Pyliteral

let equal : t -> t -> bool = ( = )
let compare : t -> t -> int = Stdlib.compare

let name = function
  | Bind -> "BIND"
  | Special -> "SPECIAL"
  | Buffer -> "BUFFER"
  | Noop -> "NOOP"
  | Rewrite_error -> "REWRITE_ERROR"
  | Param -> "PARAM"
  | Function -> "FUNCTION"
  | Call -> "CALL"
  | Program -> "PROGRAM"
  | Linear -> "LINEAR"
  | Source -> "SOURCE"
  | Binary -> "BINARY"
  | Sink -> "SINK"
  | After -> "AFTER"
  | Group -> "GROUP"
  | Stack -> "STACK"
  | Tuple -> "TUPLE"
  | Gettuple -> "GETTUPLE"
  | Getaddr -> "GETADDR"
  | Index -> "INDEX"
  | Shrink -> "SHRINK"
  | Load -> "LOAD"
  | Store -> "STORE"
  | Wmma -> "WMMA"
  | Cast -> "CAST"
  | Bitcast -> "BITCAST"
  | Exp2 -> "EXP2"
  | Log2 -> "LOG2"
  | Sin -> "SIN"
  | Sqrt -> "SQRT"
  | Reciprocal -> "RECIPROCAL"
  | Neg -> "NEG"
  | Trunc -> "TRUNC"
  | Add -> "ADD"
  | Mul -> "MUL"
  | Shl -> "SHL"
  | Shr -> "SHR"
  | Cdiv -> "CDIV"
  | Max -> "MAX"
  | Cmod -> "CMOD"
  | Cmplt -> "CMPLT"
  | Cmpne -> "CMPNE"
  | Cmpeq -> "CMPEQ"
  | Xor -> "XOR"
  | Or -> "OR"
  | And -> "AND"
  | Threefry -> "THREEFRY"
  | Sub -> "SUB"
  | Fdiv -> "FDIV"
  | Pow -> "POW"
  | Floordiv -> "FLOORDIV"
  | Floormod -> "FLOORMOD"
  | Where -> "WHERE"
  | Mulacc -> "MULACC"
  | Barrier -> "BARRIER"
  | Range -> "RANGE"
  | If -> "IF"
  | End -> "END"
  | Endif -> "ENDIF"
  | Wait -> "WAIT"
  | Const -> "CONST"
  | Custom -> "CUSTOM"
  | Customi -> "CUSTOMI"
  | Ins -> "INS"
  | Contiguous -> "CONTIGUOUS"
  | Contiguous_backward -> "CONTIGUOUS_BACKWARD"
  | Detach -> "DETACH"
  | Stage -> "STAGE"
  | Copy -> "COPY"
  | Slice -> "SLICE"
  | Mselect -> "MSELECT"
  | Mstack -> "MSTACK"
  | Custom_function -> "CUSTOM_FUNCTION"
  | Reshape -> "RESHAPE"
  | Permute -> "PERMUTE"
  | Expand -> "EXPAND"
  | Pad -> "PAD"
  | Flip -> "FLIP"
  | Multi -> "MULTI"
  | Reduce -> "REDUCE"
  | Allreduce -> "ALLREDUCE"
  | Pyliteral -> "PYLITERAL"

let pp fmt op = Format.pp_print_string fmt (name op)

module Group = struct
  let mem op group = List.exists (equal op) group

  let union a b =
    List.fold_left
      (fun acc op -> if mem op acc then acc else op :: acc)
      [] (a @ b)
    |> List.rev

  let without group excluded =
    List.filter (fun op -> not (mem op excluded)) group

  let unary = [ Exp2; Log2; Sin; Sqrt; Reciprocal; Neg; Trunc ]

  let binary =
    [
      Add;
      Mul;
      Shl;
      Shr;
      Cdiv;
      Max;
      Cmod;
      Cmplt;
      Cmpne;
      Cmpeq;
      Xor;
      Or;
      And;
      Threefry;
      Sub;
      Fdiv;
      Pow;
      Floordiv;
      Floormod;
    ]

  let ternary = [ Where; Mulacc ]
  let alu = unary @ binary @ ternary
  let broadcastable = binary @ ternary
  let elementwise = [ Cast; Bitcast ] @ alu
  let defines = [ Buffer; Param ]
  let irreducible = [ Special; Param; Getaddr; Range; Const ]
  let movement = [ Shrink; Reshape; Permute; Expand; Pad; Flip ]
  let commutative = [ Add; Mul; Max; Cmpne; Cmpeq; Xor; Or; And ]
  let associative = [ Add; Mul; Max; Or; And ]
  let idempotent = [ Max; Or; And ]
  let reduce = [ Add; Mul; Max ]
  let comparison = [ Cmplt; Cmpne; Cmpeq ]

  let all =
    [
      Bind;
      Special;
      Buffer;
      Noop;
      Rewrite_error;
      Param;
      Function;
      Call;
      Program;
      Linear;
      Source;
      Binary;
      Sink;
      After;
      Group;
      Stack;
      Tuple;
      Gettuple;
      Getaddr;
      Index;
      Shrink;
      Load;
      Store;
      Wmma;
      Cast;
      Bitcast;
      Exp2;
      Log2;
      Sin;
      Sqrt;
      Reciprocal;
      Neg;
      Trunc;
      Add;
      Mul;
      Shl;
      Shr;
      Cdiv;
      Max;
      Cmod;
      Cmplt;
      Cmpne;
      Cmpeq;
      Xor;
      Or;
      And;
      Threefry;
      Sub;
      Fdiv;
      Pow;
      Floordiv;
      Floormod;
      Where;
      Mulacc;
      Barrier;
      Range;
      If;
      End;
      Endif;
      Wait;
      Const;
      Custom;
      Customi;
      Ins;
      Contiguous;
      Contiguous_backward;
      Detach;
      Stage;
      Copy;
      Slice;
      Mselect;
      Mstack;
      Custom_function;
      Reshape;
      Permute;
      Expand;
      Pad;
      Flip;
      Multi;
      Reduce;
      Allreduce;
      Pyliteral;
    ]

  let is_unary = function
    | Exp2 | Log2 | Sin | Sqrt | Reciprocal | Neg | Trunc -> true
    | _ -> false

  let is_binary = function
    | Add | Mul | Shl | Shr | Cdiv | Max | Cmod | Cmplt | Cmpne | Cmpeq | Xor
    | Or | And | Threefry | Sub | Fdiv | Pow | Floordiv | Floormod ->
        true
    | _ -> false

  let is_ternary = function Where | Mulacc -> true | _ -> false
  let is_alu op = is_unary op || is_binary op || is_ternary op
  let is_broadcastable op = is_binary op || is_ternary op
  let is_elementwise op = is_alu op || equal op Cast || equal op Bitcast

  let is_define = function Param | Buffer -> true | _ -> false

  let is_irreducible = function
    | Const | Special | Range | Param | Getaddr -> true
    | _ -> false

  let is_movement = function
    | Reshape | Expand | Permute | Pad | Shrink | Flip -> true
    | _ -> false

  let is_commutative = function
    | Add | Mul | Max | Cmpne | Cmpeq | Xor | And | Or -> true
    | _ -> false

  let is_associative = function
    | Add | Mul | And | Or | Max -> true
    | _ -> false

  let is_idempotent = function Or | And | Max -> true | _ -> false
  let is_reduce = function Add | Mul | Max -> true | _ -> false
  let is_comparison = function Cmplt | Cmpne | Cmpeq -> true | _ -> false
end
