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
  | Custom_function
  | Reshape
  | Permute
  | Expand
  | Pad
  | Reduce

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
    Custom_function;
    Reshape;
    Permute;
    Expand;
    Pad;
    Reduce;
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
  | Stack -> (12, "STACK")
  | Getaddr -> (13, "GETADDR")
  | Index -> (14, "INDEX")
  | Shrink -> (15, "SHRINK")
  | Load -> (16, "LOAD")
  | Store -> (17, "STORE")
  | Wmma -> (18, "WMMA")
  | Cast -> (19, "CAST")
  | Bitcast -> (20, "BITCAST")
  | Exp2 -> (21, "EXP2")
  | Log2 -> (22, "LOG2")
  | Sin -> (23, "SIN")
  | Sqrt -> (24, "SQRT")
  | Reciprocal -> (25, "RECIPROCAL")
  | Neg -> (26, "NEG")
  | Trunc -> (27, "TRUNC")
  | Add -> (28, "ADD")
  | Mul -> (29, "MUL")
  | Shl -> (30, "SHL")
  | Shr -> (31, "SHR")
  | Cdiv -> (32, "CDIV")
  | Max -> (33, "MAX")
  | Cmod -> (34, "CMOD")
  | Cmplt -> (35, "CMPLT")
  | Cmpne -> (36, "CMPNE")
  | Cmpeq -> (37, "CMPEQ")
  | Xor -> (38, "XOR")
  | Or -> (39, "OR")
  | And -> (40, "AND")
  | Threefry -> (41, "THREEFRY")
  | Sub -> (42, "SUB")
  | Fdiv -> (43, "FDIV")
  | Pow -> (44, "POW")
  | Floordiv -> (45, "FLOORDIV")
  | Floormod -> (46, "FLOORMOD")
  | Where -> (47, "WHERE")
  | Mulacc -> (48, "MULACC")
  | Barrier -> (49, "BARRIER")
  | Range -> (50, "RANGE")
  | If -> (51, "IF")
  | End -> (52, "END")
  | Endif -> (53, "ENDIF")
  | Backedge -> (54, "BACKEDGE")
  | Const -> (55, "CONST")
  | Custom -> (56, "CUSTOM")
  | Customi -> (57, "CUSTOMI")
  | Ins -> (58, "INS")
  | Contiguous_backward -> (59, "CONTIGUOUS_BACKWARD")
  | Detach -> (60, "DETACH")
  | Stage -> (61, "STAGE")
  | Copy -> (62, "COPY")
  | Custom_function -> (63, "CUSTOM_FUNCTION")
  | Reshape -> (64, "RESHAPE")
  | Permute -> (65, "PERMUTE")
  | Expand -> (66, "EXPAND")
  | Pad -> (67, "PAD")
  | Reduce -> (68, "REDUCE")

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
  let movement = of_list [ Reshape; Expand; Permute; Pad; Shrink ]
  let commutative = of_list [ Add; Mul; Max; Cmpne; Cmpeq; Xor; And; Or ]
  let associative = of_list [ Add; Mul; And; Or; Max ]
  let idempotent = of_list [ Or; And; Max ]
  let reduce = of_list [ Add; Mul; Max ]
  let comparison = of_list [ Cmplt; Cmpne; Cmpeq ]
  let all = of_list ops
end
