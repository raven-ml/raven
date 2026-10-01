(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type move = Reshape | Expand | Permute | Shrink | Flip | Window

type t =
  | Unary of Nx_backend.unary
  | Binary of Nx_backend.binary
  | Compare of Nx_backend.compare
  | Where
  | Fma
  | Reduce of Nx_backend.reduce
  | Scan of Nx_backend.reduce
  | Arg_reduce of Nx_backend.arg_reduce
  | Sort
  | Argsort
  | Pad
  | Cat
  | Cast
  | Bitcast
  | Threefry
  | Gather
  | Scatter of Nx_backend.scatter
  | Update
  | Unfold
  | Fold
  | Matmul
  | Fft
  | Rfft
  | Irfft
  | Contiguous
  | Cholesky
  | Qr
  | Lu
  | Svd
  | Eig
  | Eigh
  | Solve_triangular
  | Move of move
  | Place
  | Read
  | Check

let of_op : type r. r Nx.Op.t -> t =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (k, _) -> Unary k
  | Binary (k, _, _) -> Binary k
  | Compare (k, _, _) -> Compare k
  | Where _ -> Where
  | Fma _ -> Fma
  | Reduce (k, _, _) -> Reduce k
  | Scan (k, _, _) -> Scan k
  | Arg_reduce (k, _, _) -> Arg_reduce k
  | Sort _ -> Sort
  | Argsort _ -> Argsort
  | Pad _ -> Pad
  | Cat _ -> Cat
  | Convert (Cast, _, _) -> Cast
  | Convert (Bitcast, _, _) -> Bitcast
  | Threefry _ -> Threefry
  | Gather _ -> Gather
  | Scatter { mode; _ } -> Scatter mode
  | Update _ -> Update
  | Unfold _ -> Unfold
  | Fold _ -> Fold
  | Matmul _ -> Matmul
  | Fft _ -> Fft
  | Rfft _ -> Rfft
  | Irfft _ -> Irfft
  | Contiguous _ -> Contiguous
  | Cholesky _ -> Cholesky
  | Qr _ -> Qr
  | Lu _ -> Lu
  | Svd _ -> Svd
  | Eig _ -> Eig
  | Eigh _ -> Eigh
  | Solve_triangular _ -> Solve_triangular
  | Move (_, Reshape _) -> Move Reshape
  | Move (_, Expand _) -> Move Expand
  | Move (_, Permute _) -> Move Permute
  | Move (_, Shrink _) -> Move Shrink
  | Move (_, Flip _) -> Move Flip
  | Move (_, Window _) -> Move Window
  | Place _ -> Place
  | Read _ -> Read
  | Check _ -> Check

let issued f =
  let ops = ref [] in
  let run : type r. r Nx.Op.t -> r =
   fun op ->
    ops := (of_op op, Nx.Op.name op) :: !ops;
    Nx.Op.eval op
  in
  ignore (Nx.Op.intercept { run; claims = (fun _ -> true) } f);
  List.rev !ops

let unaries : Nx_backend.unary list =
  [
    Neg;
    Recip;
    Abs;
    Sqrt;
    Sign;
    Exp;
    Log;
    Log1p;
    Expm1;
    Sin;
    Cos;
    Tan;
    Asin;
    Acos;
    Atan;
    Sinh;
    Cosh;
    Tanh;
    Trunc;
    Ceil;
    Floor;
    Round;
    Erf;
  ]

let binaries : Nx_backend.binary list =
  [ Add; Sub; Mul; Fdiv; Idiv; Mod; Pow; Atan2; Maximum; Minimum; And; Or; Xor ]

let reduces : Nx_backend.reduce list = [ Sum; Prod; Max; Min ]

let all =
  List.concat
    [
      List.map (fun k -> Unary k) unaries;
      List.map (fun k -> Binary k) binaries;
      List.map
        (fun k -> Compare k)
        Nx_backend.[ Equal; Not_equal; Less; Less_equal ];
      [ Where; Fma ];
      List.map (fun k -> Reduce k) reduces;
      List.map (fun k -> Scan k) reduces;
      [ Arg_reduce Argmax; Arg_reduce Argmin ];
      [
        Sort;
        Argsort;
        Pad;
        Cat;
        Cast;
        Bitcast;
        Threefry;
        Gather;
        Scatter `Set;
        Scatter `Add;
        Scatter `Max;
        Scatter `Min;
        Update;
        Unfold;
        Fold;
        Matmul;
        Fft;
        Rfft;
        Irfft;
        Contiguous;
        Cholesky;
        Qr;
        Lu;
        Svd;
        Eig;
        Eigh;
        Solve_triangular;
      ];
      List.map
        (fun m -> Move m)
        [ Reshape; Expand; Permute; Shrink; Flip; Window ];
      [ Place; Read; Check ];
    ]

let unary_name : Nx_backend.unary -> string = function
  | Neg -> "neg"
  | Recip -> "recip"
  | Abs -> "abs"
  | Sqrt -> "sqrt"
  | Sign -> "sign"
  | Exp -> "exp"
  | Log -> "log"
  | Log1p -> "log1p"
  | Expm1 -> "expm1"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Trunc -> "trunc"
  | Ceil -> "ceil"
  | Floor -> "floor"
  | Round -> "round"
  | Erf -> "erf"

let binary_name : Nx_backend.binary -> string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Fdiv -> "fdiv"
  | Idiv -> "idiv"
  | Mod -> "mod"
  | Pow -> "pow"
  | Atan2 -> "atan2"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> "and"
  | Or -> "or"
  | Xor -> "xor"

let reduce_name : Nx_backend.reduce -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

let name = function
  | Unary k -> "unary " ^ unary_name k
  | Binary k -> "binary " ^ binary_name k
  | Compare Equal -> "compare equal"
  | Compare Not_equal -> "compare not_equal"
  | Compare Less -> "compare less"
  | Compare Less_equal -> "compare less_equal"
  | Where -> "where"
  | Fma -> "fma"
  | Reduce k -> "reduce " ^ reduce_name k
  | Scan k -> "scan " ^ reduce_name k
  | Arg_reduce Argmax -> "arg_reduce argmax"
  | Arg_reduce Argmin -> "arg_reduce argmin"
  | Sort -> "sort"
  | Argsort -> "argsort"
  | Pad -> "pad"
  | Cat -> "cat"
  | Cast -> "cast"
  | Bitcast -> "bitcast"
  | Threefry -> "threefry"
  | Gather -> "gather"
  | Scatter `Set -> "scatter set"
  | Scatter `Add -> "scatter add"
  | Scatter `Max -> "scatter max"
  | Scatter `Min -> "scatter min"
  | Update -> "update"
  | Unfold -> "unfold"
  | Fold -> "fold"
  | Matmul -> "matmul"
  | Fft -> "fft"
  | Rfft -> "rfft"
  | Irfft -> "irfft"
  | Contiguous -> "contiguous"
  | Cholesky -> "cholesky"
  | Qr -> "qr"
  | Lu -> "lu"
  | Svd -> "svd"
  | Eig -> "eig"
  | Eigh -> "eigh"
  | Solve_triangular -> "solve_triangular"
  | Move Reshape -> "move reshape"
  | Move Expand -> "move expand"
  | Move Permute -> "move permute"
  | Move Shrink -> "move shrink"
  | Move Flip -> "move flip"
  | Move Window -> "move window"
  | Place -> "place"
  | Read -> "read"
  | Check -> "check"

let equal a b = String.equal (name a) (name b)
let compare a b = String.compare (name a) (name b)
let pp ppf r = Format.pp_print_string ppf (name r)
