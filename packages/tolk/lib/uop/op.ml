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
  | Refparam
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
  | Recip
  | Neg
  | Trunc
  | Add
  | Mul
  | Shl
  | Shr
  | Truncdiv
  | Max
  | Truncmod
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
  | Div
  | Mod
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
    Refparam;
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
    Recip;
    Neg;
    Trunc;
    Add;
    Mul;
    Shl;
    Shr;
    Truncdiv;
    Max;
    Truncmod;
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
    Div;
    Mod;
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
  | Refparam -> (5, "REFPARAM")
  | Call -> (6, "CALL")
  | Program -> (7, "PROGRAM")
  | Linear -> (8, "LINEAR")
  | Source -> (9, "SOURCE")
  | Binary -> (10, "BINARY")
  | Sink -> (11, "SINK")
  | After -> (12, "AFTER")
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
  | Recip -> (26, "RECIP")
  | Neg -> (27, "NEG")
  | Trunc -> (28, "TRUNC")
  | Add -> (29, "ADD")
  | Mul -> (30, "MUL")
  | Shl -> (31, "SHL")
  | Shr -> (32, "SHR")
  | Truncdiv -> (33, "TRUNCDIV")
  | Max -> (34, "MAX")
  | Truncmod -> (35, "TRUNCMOD")
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
  | Div -> (46, "DIV")
  | Mod -> (47, "MOD")
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
  | Custom_function -> (64, "CUSTOM_FUNCTION")
  | Reshape -> (65, "RESHAPE")
  | Permute -> (66, "PERMUTE")
  | Expand -> (67, "EXPAND")
  | Pad -> (68, "PAD")
  | Reduce -> (69, "REDUCE")

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

  (* Bits indexed by [to_int], the first [width] in [low] and the others in
     [high]: a set is three words and testing it reads one. *)
  type t = { low : int; high : int }

  let width = 62
  let () = assert (List.length ops <= 2 * width)
  let empty = { low = 0; high = 0 }

  let add s o =
    let i = to_int o in
    if i < width then { s with low = s.low lor (1 lsl i) }
    else { s with high = s.high lor (1 lsl (i - width)) }

  let of_list os = List.fold_left add empty os

  let mem o s =
    let i = to_int o in
    if i < width then (s.low lsr i) land 1 = 1
    else (s.high lsr (i - width)) land 1 = 1

  let union s s' = { low = s.low lor s'.low; high = s.high lor s'.high }

  let diff s s' =
    { low = s.low land lnot s'.low; high = s.high land lnot s'.high }

  let to_list s = List.filter (fun o -> mem o s) ops
  let equal s s' = s.low = s'.low && s.high = s'.high

  let pp ppf s =
    let pp_sep ppf () = Format.fprintf ppf ",@ " in
    Format.fprintf ppf "@[<1>{%a}@]"
      (Format.pp_print_list ~pp_sep pp_op)
      (to_list s)

  let unary = of_list [ Exp2; Log2; Sin; Sqrt; Recip; Neg; Trunc ]

  let binary =
    of_list
      [
        Add;
        Mul;
        Truncdiv;
        Max;
        Truncmod;
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
        Div;
        Mod;
      ]

  let ternary = of_list [ Where; Mulacc ]
  let alu = union unary (union binary ternary)
  let broadcastable = union binary ternary
  let elementwise = union alu (of_list [ Cast; Bitcast ])
  let defines = of_list [ Param; Refparam; Buffer; Alloc; Binary ]
  let irreducible = of_list [ Const; Special; Range; Param; Getaddr ]
  let movement = of_list [ Reshape; Expand; Permute; Pad; Shrink ]
  let effectful = of_list [ Store; Custom; Call; Barrier ]
  let commutative = of_list [ Add; Mul; Max; Cmpne; Cmpeq; Xor; And; Or ]
  let associative = of_list [ Add; Mul; And; Or; Max; Xor ]
  let idempotent = of_list [ Or; And; Max ]
  let reduce = of_list [ Add; Mul; Max ]
  let comparison = of_list [ Cmplt; Cmpne; Cmpeq ]
  let all = of_list ops
end
