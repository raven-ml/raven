(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)


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
  | Cmod
  | Cdiv
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

let all_supported_ops : Decomp_op.supported_ops =
  {
    has_exp2 = true; has_log2 = true; has_sin = true; has_sqrt = true;
    has_neg = true; has_sub = true; has_max = true; has_shl = true;
    has_shr = true; has_and = true; has_or = true; has_cmplt = true;
    has_cmpeq = true; has_fdiv = true;
    has_threefry = true; has_mulacc = true;
    is_metal = false; supports_dtype = (fun _ -> true);
    disable_fast_idiv = false; force_transcendental = false;
  }

let supported_ops_of_code_for_op (ops : code_op list) : Decomp_op.supported_ops =
  let has op = List.mem op ops in
  {
    has_exp2 = has Exp2; has_log2 = has Log2; has_sin = has Sin;
    has_sqrt = has Sqrt; has_neg = has Neg;
    has_sub = has Sub; has_max = has Max; has_shl = has Shl;
    has_shr = has Shr; has_and = has And; has_or = has Or;
    has_cmplt = has Cmplt; has_cmpeq = has Cmpeq; has_fdiv = has Fdiv;
    has_threefry = has Threefry;
    has_mulacc = has Mulacc;
    is_metal = false;
    supports_dtype = (fun _ -> true);
    disable_fast_idiv = false;
    force_transcendental = false;
  }

type t = {
  name : string;
  device : string;
  compiler : Compiler.t option;
  has_local : bool;
  has_threads : bool;
  has_shared : bool;
  global_max : int list option;
  global_prod_max : int list option;
  local_max : int list option;
  shared_max : int;
  tensor_cores : Tc.t list;
  supports_float4 : bool;
  image_pitch_alignment : int option;
  render : ?name:string -> Tolk_uop.Uop.t list -> string;
  aux : Tolk_uop.Uop.t list -> string list;
  code_for_op : code_op list;
  supported_ops : Decomp_op.supported_ops;
  extra_matcher : (Tolk_uop.Uop.t -> Tolk_uop.Uop.t option) option;
  supports_dtype : Tolk_uop.Dtype.t -> bool;
  emulated_floats : (Tolk_uop.Dtype.t * Tolk_uop.Dtype.t) list;
}

(* Accessors *)

let name t = t.name
let device t = t.device
let compiler t = t.compiler
let has_local t = t.has_local
let has_threads t = t.has_threads
let has_shared t = t.has_shared
let global_max t = t.global_max
let global_prod_max t = t.global_prod_max
let local_max t = t.local_max
let shared_max t = t.shared_max
let tensor_cores t = t.tensor_cores
let supports_float4 t = t.supports_float4
let image_pitch_alignment t = t.image_pitch_alignment
let render t = t.render
let aux t = t.aux
let code_for_op t = t.code_for_op
let supported_ops t = t.supported_ops
let extra_matcher t = t.extra_matcher

(* dtype support — checks whether the backend natively supports a given dtype
   and lists float types that need emulation (promoted to a wider float). *)
let supports_dtype t dt = t.supports_dtype dt
let emulated_float_dtypes t : (Tolk_uop.Dtype.t * Tolk_uop.Dtype.t) list =
  t.emulated_floats

(* Construction *)

(* 0x8FFFFFFF: conservative upper bound for grid/block dimensions.
   Backends override with actual hardware limits (e.g., CUDA
   gridDim.x = 2^31-1). *)
let with_compiler compiler t = { t with compiler = Some compiler }

let make ?(tensor_cores = []) ?(supports_float4 = true)
    ?image_pitch_alignment
    ?(has_threads = false)
    ?(global_max = [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ])
    ?global_prod_max
    ?(local_max = [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ])
    ?(code_for_op = []) ?supported_ops ?compiler ?extra_matcher
    ?(supports_dtype = fun _ -> true) ?(aux = fun _ -> [])
    ?(emulated_floats = [])
    ~name ~device ~has_local ~has_shared ~shared_max ~render () =
  let supported_ops =
    match supported_ops with
    | Some ops -> ops
    | None ->
        if code_for_op = [] then all_supported_ops
        else supported_ops_of_code_for_op code_for_op
  in
  let supported_ops = { supported_ops with is_metal = name = "metal"; supports_dtype } in
  {
    name;
    device;
    compiler;
    has_local;
    has_threads;
    has_shared;
    global_max = Some global_max;
    global_prod_max;
    local_max = Some local_max;
    shared_max;
    tensor_cores;
    supports_float4;
    image_pitch_alignment;
    render;
    aux;
    code_for_op;
    supported_ops;
    extra_matcher;
    supports_dtype;
    emulated_floats;
  }
