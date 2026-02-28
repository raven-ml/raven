(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* ───── Types ───── *)

type tensor_core = {
  dims : int * int * int;
  threads : int;
  elements_per_thread : int * int * int;
  dtype_in : Dtype.scalar;
  dtype_out : Dtype.scalar;
  opts : string list;
  swizzle :
    (string list * string list * string list)
    * (string list * string list * string list);
}

(* ALU operations a backend can provide custom rendering for. *)
type code_op =
  | Sqrt
  | Recip
  | Neg
  | Exp2
  | Log2
  | Sin
  | Trunc
  | And
  | Xor
  | Or
  | Add
  | Sub
  | Mul
  | Mod
  | Idiv
  | Cmpne
  | Shr
  | Shl
  | Cmplt
  | Where
  | Cmpeq
  | Fdiv
  | Max
  | Mulacc
  | Threefry

(* Backend capability flags consumed by decomposition passes. *)
type supported_ops = {
  has_exp2 : bool;
  has_log2 : bool;
  has_sin : bool;
  has_sqrt : bool;
  has_recip : bool;
  has_neg : bool;
  has_sub : bool;
  has_max : bool;
  has_shl : bool;
  has_shr : bool;
  has_and : bool;
  has_or : bool;
  has_cmplt : bool;
  has_cmpeq : bool;
  has_fdiv : bool;
  has_threefry : bool;
  has_mulacc : bool;
}

let all_supported_ops =
  {
    has_exp2 = true;
    has_log2 = true;
    has_sin = true;
    has_sqrt = true;
    has_recip = true;
    has_neg = true;
    has_sub = true;
    has_max = true;
    has_shl = true;
    has_shr = true;
    has_and = true;
    has_or = true;
    has_cmplt = true;
    has_cmpeq = true;
    has_fdiv = true;
    has_threefry = true;
    has_mulacc = true;
  }

let supported_ops_of_code_for_op (ops : code_op list) : supported_ops =
  let has op = List.mem op ops in
  {
    has_exp2 = has Exp2;
    has_log2 = has Log2;
    has_sin = has Sin;
    has_sqrt = has Sqrt;
    has_recip = has Recip;
    has_neg = has Neg;
    has_sub = has Sub;
    has_max = has Max;
    has_shl = has Shl;
    has_shr = has Shr;
    has_and = has And;
    has_or = has Or;
    has_cmplt = has Cmplt;
    has_cmpeq = has Cmpeq;
    has_fdiv = has Fdiv;
    has_threefry = has Threefry;
    has_mulacc = has Mulacc;
  }

type t = {
  name : string;
  device : string;
  has_local : bool;
  has_threads : bool;
  has_shared : bool;
  global_max : int list option;
  local_max : int list option;
  shared_max : int;
  tensor_cores : tensor_core list;
  load_store_widths : Dtype.t -> int list;
  render : ?name:string -> Ir.Program.t -> string;
  code_for_op : code_op list;
  supported_ops : supported_ops;
}

(* ───── Accessors ───── *)

let name t = t.name
let device t = t.device
let has_local t = t.has_local
let has_threads t = t.has_threads
let has_shared t = t.has_shared
let global_max t = t.global_max
let local_max t = t.local_max
let shared_max t = t.shared_max
let tensor_cores t = t.tensor_cores
let load_store_widths t = t.load_store_widths
let render t = t.render
let code_for_op t = t.code_for_op
let supported_ops t = t.supported_ops

(* ───── Construction ───── *)

(* 0x3FFFFFFF: conservative upper bound for grid dimensions that fits in a
   31-bit OCaml int. Backends override with actual hardware limits (e.g., CUDA
   gridDim.x = 2^31-1). *)
let make ?(tensor_cores = []) ?(load_store_widths = fun _ -> [ 1 ])
    ?(has_threads = false)
    ?(global_max = Some [ 0x3FFFFFFF; 0x3FFFFFFF; 0x3FFFFFFF ])
    ?(local_max = Some [ 0x3FFFFFFF; 0x3FFFFFFF; 0x3FFFFFFF ])
    ?(code_for_op = []) ?supported_ops ~name ~device ~has_local ~has_shared
    ~shared_max ~render () =
  let supported_ops =
    match supported_ops with
    | Some ops -> ops
    | None ->
        if code_for_op = [] then all_supported_ops
        else supported_ops_of_code_for_op code_for_op
  in
  {
    name;
    device;
    has_local;
    has_threads;
    has_shared;
    global_max;
    local_max;
    shared_max;
    tensor_cores;
    load_store_widths;
    render;
    code_for_op;
    supported_ops;
  }
