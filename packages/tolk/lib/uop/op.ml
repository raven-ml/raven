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

let name = function
  | Special -> "SPECIAL"
  | Buffer -> "BUFFER"
  | Alloc -> "ALLOC"
  | Noop -> "NOOP"
  | Param -> "PARAM"
  | Refparam -> "REFPARAM"
  | Call -> "CALL"
  | Program -> "PROGRAM"
  | Linear -> "LINEAR"
  | Source -> "SOURCE"
  | Binary -> "BINARY"
  | Sink -> "SINK"
  | After -> "AFTER"
  | Stack -> "STACK"
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
  | Recip -> "RECIP"
  | Neg -> "NEG"
  | Trunc -> "TRUNC"
  | Add -> "ADD"
  | Mul -> "MUL"
  | Shl -> "SHL"
  | Shr -> "SHR"
  | Truncdiv -> "TRUNCDIV"
  | Max -> "MAX"
  | Truncmod -> "TRUNCMOD"
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
  | Div -> "DIV"
  | Mod -> "MOD"
  | Where -> "WHERE"
  | Mulacc -> "MULACC"
  | Barrier -> "BARRIER"
  | Range -> "RANGE"
  | If -> "IF"
  | End -> "END"
  | Endif -> "ENDIF"
  | Backedge -> "BACKEDGE"
  | Const -> "CONST"
  | Custom -> "CUSTOM"
  | Customi -> "CUSTOMI"
  | Ins -> "INS"
  | Contiguous_backward -> "CONTIGUOUS_BACKWARD"
  | Detach -> "DETACH"
  | Stage -> "STAGE"
  | Copy -> "COPY"
  | Custom_function -> "CUSTOM_FUNCTION"
  | Reshape -> "RESHAPE"
  | Permute -> "PERMUTE"
  | Expand -> "EXPAND"
  | Pad -> "PAD"
  | Reduce -> "REDUCE"

external to_int : t -> int = "%identity"

let equal (o : t) o' = o = o'
let compare o o' = Int.compare (to_int o) (to_int o')

(* Each operation by its name, looked up by hashing: a graph read back names
   an operation at each of its nodes. *)
let by_name =
  let t = Hashtbl.create 128 in
  List.iter (fun o -> Hashtbl.replace t (name o) o) ops;
  t

let of_string s =
  match Hashtbl.find_opt by_name s with
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
