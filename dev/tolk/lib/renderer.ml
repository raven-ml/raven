(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type tensor_core = {
  dims : int * int * int;
  threads : int;
  elements_per_thread : int * int * int;
  dtype_in : Tolk_ir.Dtype.scalar;
  dtype_out : Tolk_ir.Dtype.scalar;
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

let all_supported_ops : Tolk_ir.Decompositions.supported_ops =
  {
    has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt = true;
    has_recip = true; has_neg = true; has_sub = true; has_max = true;
    has_shl = true; has_shr = true; has_and = true; has_or = true;
    has_cmplt = true; has_cmpeq = true; has_fdiv = true;
    has_threefry = true; has_mulacc = true;
    disable_fast_idiv = false; force_transcendental = false;
  }

let supported_ops_of_code_for_op (ops : code_op list) : Tolk_ir.Decompositions.supported_ops =
  let has op = List.mem op ops in
  {
    has_exp2 = has Exp2; has_log2 = has Log2; has_sin = has Sin;
    has_sqrt = has Sqrt; has_recip = has Recip; has_neg = has Neg;
    has_sub = has Sub; has_max = has Max; has_shl = has Shl;
    has_shr = has Shr; has_and = has And; has_or = has Or;
    has_cmplt = has Cmplt; has_cmpeq = has Cmpeq; has_fdiv = has Fdiv;
    has_threefry = has Threefry;
    has_mulacc = has Mulacc;
    disable_fast_idiv = false;
    force_transcendental = false;
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
  load_store_widths : Tolk_ir.Dtype.t -> int list;
  render : ?name:string -> Tolk_ir.Program.t -> string;
  code_for_op : code_op list;
  supported_ops : Tolk_ir.Decompositions.supported_ops;
  pre_matcher : (Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) option;
  extra_matcher : (Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option) option;
}

(* Accessors *)

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
let pre_matcher t = t.pre_matcher
let extra_matcher t = t.extra_matcher

(* dtype support — checks whether the backend natively supports a given dtype
   and lists float types that need emulation (promoted to a wider float). *)
let supports_dtype _t _dt = true
let emulated_float_dtypes _t : (Tolk_ir.Dtype.scalar * Tolk_ir.Dtype.scalar) list = []

(* Construction *)

(* 0x8FFFFFFF: conservative upper bound for grid/block dimensions.
   Backends override with actual hardware limits (e.g., CUDA
   gridDim.x = 2^31-1). *)
let make ?(tensor_cores = []) ?(load_store_widths = fun _ -> [ 1 ])
    ?(has_threads = false)
    ?(global_max = Some [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ])
    ?(local_max = Some [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ])
    ?(code_for_op = []) ?supported_ops ?pre_matcher ?extra_matcher
    ~name ~device ~has_local ~has_shared ~shared_max ~render () =
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
    pre_matcher;
    extra_matcher;
  }
