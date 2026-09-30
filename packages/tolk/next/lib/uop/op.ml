(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* The declaration order is the order of [compare]: graph orderings and
   canonical operand orders are derived from it. *)
type t =
  (* Definitions *)
  | Special
  | Buffer
  | Alloc
  (* Structure *)
  | Noop
  | Param
  | Call
  | Program
  | Linear
  | Source
  | Binary
  | Sink
  | After
  | Group
  | Stack
  | Getaddr
  (* Memory *)
  | Index
  | Shrink
  | Load
  | Store
  (* Arithmetic *)
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
  (* Control flow, constants and target code *)
  | Barrier
  | Range
  | If
  | End
  | Endif
  | Backedge
  | Const
  | Custom
  | Customi
  | Ins
  (* Tensor graph *)
  | Contiguous_backward
  | Detach
  | Stage
  | Copy
  | Mselect
  | Mstack
  | Custom_function
  | Reshape
  | Permute
  | Expand
  | Pad
  | Flip
  | Unshard
  | Reduce
  | Allreduce

let ops =
  [
    Special;
    Buffer;
    Alloc;
    Noop;
    Param;
    Call;
    Program;
    Linear;
    Source;
    Binary;
    Sink;
    After;
    Group;
    Stack;
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
    Backedge;
    Const;
    Custom;
    Customi;
    Ins;
    Contiguous_backward;
    Detach;
    Stage;
    Copy;
    Mselect;
    Mstack;
    Custom_function;
    Reshape;
    Permute;
    Expand;
    Pad;
    Flip;
    Unshard;
    Reduce;
    Allreduce;
  ]

(* [info o] is the position of [o] in the declaration order and its name. *)
let info = function
  | Special -> (0, "SPECIAL")
  | Buffer -> (1, "BUFFER")
  | Alloc -> (2, "ALLOC")
  | Noop -> (3, "NOOP")
  | Param -> (4, "PARAM")
  | Call -> (5, "CALL")
  | Program -> (6, "PROGRAM")
  | Linear -> (7, "LINEAR")
  | Source -> (8, "SOURCE")
  | Binary -> (9, "BINARY")
  | Sink -> (10, "SINK")
  | After -> (11, "AFTER")
  | Group -> (12, "GROUP")
  | Stack -> (13, "STACK")
  | Getaddr -> (14, "GETADDR")
  | Index -> (15, "INDEX")
  | Shrink -> (16, "SHRINK")
  | Load -> (17, "LOAD")
  | Store -> (18, "STORE")
  | Wmma -> (19, "WMMA")
  | Cast -> (20, "CAST")
  | Bitcast -> (21, "BITCAST")
  | Exp2 -> (22, "EXP2")
  | Log2 -> (23, "LOG2")
  | Sin -> (24, "SIN")
  | Sqrt -> (25, "SQRT")
  | Reciprocal -> (26, "RECIPROCAL")
  | Neg -> (27, "NEG")
  | Trunc -> (28, "TRUNC")
  | Add -> (29, "ADD")
  | Mul -> (30, "MUL")
  | Shl -> (31, "SHL")
  | Shr -> (32, "SHR")
  | Cdiv -> (33, "CDIV")
  | Max -> (34, "MAX")
  | Cmod -> (35, "CMOD")
  | Cmplt -> (36, "CMPLT")
  | Cmpne -> (37, "CMPNE")
  | Cmpeq -> (38, "CMPEQ")
  | Xor -> (39, "XOR")
  | Or -> (40, "OR")
  | And -> (41, "AND")
  | Threefry -> (42, "THREEFRY")
  | Sub -> (43, "SUB")
  | Fdiv -> (44, "FDIV")
  | Pow -> (45, "POW")
  | Floordiv -> (46, "FLOORDIV")
  | Floormod -> (47, "FLOORMOD")
  | Where -> (48, "WHERE")
  | Mulacc -> (49, "MULACC")
  | Barrier -> (50, "BARRIER")
  | Range -> (51, "RANGE")
  | If -> (52, "IF")
  | End -> (53, "END")
  | Endif -> (54, "ENDIF")
  | Backedge -> (55, "BACKEDGE")
  | Const -> (56, "CONST")
  | Custom -> (57, "CUSTOM")
  | Customi -> (58, "CUSTOMI")
  | Ins -> (59, "INS")
  | Contiguous_backward -> (60, "CONTIGUOUS_BACKWARD")
  | Detach -> (61, "DETACH")
  | Stage -> (62, "STAGE")
  | Copy -> (63, "COPY")
  | Mselect -> (64, "MSELECT")
  | Mstack -> (65, "MSTACK")
  | Custom_function -> (66, "CUSTOM_FUNCTION")
  | Reshape -> (67, "RESHAPE")
  | Permute -> (68, "PERMUTE")
  | Expand -> (69, "EXPAND")
  | Pad -> (70, "PAD")
  | Flip -> (71, "FLIP")
  | Unshard -> (72, "UNSHARD")
  | Reduce -> (73, "REDUCE")
  | Allreduce -> (74, "ALLREDUCE")

let to_int o = fst (info o)
let name o = snd (info o)
let equal (o : t) o' = o = o'
let compare o o' = Int.compare (to_int o) (to_int o')

let of_string s =
  match List.find_opt (fun o -> String.equal (name o) s) ops with
  | Some o -> Ok o
  | None -> Error (Printf.sprintf "%S is not an operation" s)

let pp ppf o = Format.fprintf ppf "Ops.%s" (name o)

(* Sets *)

module Set = struct
  let pp_op = pp

  (* Indexed by [to_int]; never mutated once built. *)
  type t = bool array

  let of_list os =
    let s = Array.make (List.length ops) false in
    List.iter (fun o -> s.(to_int o) <- true) os;
    s

  let mem o s = Array.unsafe_get s (to_int o)
  let union s s' = Array.map2 ( || ) s s'
  let diff s s' = Array.map2 (fun m m' -> m && not m') s s'
  let to_list s = List.filter (fun o -> mem o s) ops
  let equal s s' = Array.for_all2 Bool.equal s s'

  let pp ppf s =
    let pp_sep ppf () = Format.fprintf ppf ",@ " in
    Format.fprintf ppf "@[<1>{%a}@]"
      (Format.pp_print_list ~pp_sep pp_op)
      (to_list s)

  let unary = of_list [ Exp2; Log2; Sin; Sqrt; Reciprocal; Neg; Trunc ]

  let binary =
    of_list
      [
        Add;
        Mul;
        Cdiv;
        Max;
        Cmod;
        Cmplt;
        Cmpne;
        Cmpeq;
        Xor;
        Shl;
        Shr;
        Or;
        And;
        Threefry;
        Sub;
        Fdiv;
        Pow;
        Floordiv;
        Floormod;
      ]

  let ternary = of_list [ Where; Mulacc ]
  let alu = union unary (union binary ternary)
  let broadcastable = union binary ternary
  let elementwise = union alu (of_list [ Cast; Bitcast ])
  let defines = of_list [ Param; Buffer; Alloc ]
  let irreducible = of_list [ Const; Special; Range; Param; Getaddr ]
  let movement = of_list [ Reshape; Expand; Permute; Pad; Shrink; Flip ]
  let commutative = of_list [ Add; Mul; Max; Cmpne; Cmpeq; Xor; And; Or ]
  let associative = of_list [ Add; Mul; And; Or; Max ]
  let idempotent = of_list [ Or; And; Max ]
  let reduce = of_list [ Add; Mul; Max ]
  let comparison = of_list [ Cmplt; Cmpne; Cmpeq ]
  let all = of_list ops
end
