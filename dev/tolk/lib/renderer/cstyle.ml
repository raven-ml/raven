(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Program = Ir.Program

(* ───── Platform Detection ───── *)

(* Metal tensor cores (simdgroup matrix operations) require arm64 Apple Silicon.
   Intel Macs don't support these instructions. *)

let is_arm64 =
  lazy
    (try
       let ic = Unix.open_process_in "uname -m" in
       let result = try input_line ic with End_of_file -> "" in
       let _ = Unix.close_process_in ic in
       String.trim result = "arm64"
     with Unix.Unix_error _ | Sys_error _ -> false)

(* ───── Error Messages ───── *)

let err_base_no_workitem = "base_lang: no workitem support"
let err_base_type_map = "base_lang: type_map"
let err_base_render_const = "base_lang: render_const"
let err_base_render_cast = "base_lang: render_cast"
let err_base_render_bitcast = "base_lang: render_bitcast"
let err_base_render_vector = "base_lang: render_vector"
let err_clang_no_gpu = "clang: no GPU thread support"
let err_custom_unmatched_open = "render_custom: unmatched '{'"
let err_custom_unmatched_close = "render_custom: unmatched '}'"
let err_code_for_op_void = "render_expr: code_for_op returned value for void op"
let err_bitcast_no_dtype = "render_expr: bitcast source dtype unavailable"
let err_gep_no_dtype = "render_expr: gep source dtype unavailable"
let err_load_alt_gate = "render_stmt: load alt requires gated Index"
let err_unhandled_alu = "render_kernel: ALU op not handled by code_for_op"

(* ───── Tensor Core Definitions ───── *)

(* Swizzle patterns control how tensor core operands (A, B, C in D = A*B+C) are
   laid out across threads and registers. Each swizzle is a pair of permutations
   (source, dest), where each permutation is a triple of string lists:
   (local_dims, upcast_dims, reduce_dims).

   The string prefixes encode dimension types: - "l<n>": local (workgroup)
   dimension n — distributes work across threads - "u<n>": upcast dimension n —
   maps to per-thread register elements - "r<n>": reduce dimension n — the
   contraction (K) axis

   The swizzle defines how to remap the kernel's shape dimensions to match the
   hardware's expected operand layout. *)

let tensor_core ~dims ~threads ~elements_per_thread ~dtype_in ~dtype_out ~opts
    ~swizzle : Renderer.tensor_core =
  { dims; threads; elements_per_thread; dtype_in; dtype_out; opts; swizzle }

(* threads=32 for NVIDIA/Metal (warp size), threads=64 for AMD (wavefront size).
   These match the number of threads that cooperate to execute one WMMA/MFMA
   instruction, equal to 2^(number of "l" entries in opts). *)

let cuda_tc_opts = [ "u0"; "l0"; "l0"; "l1"; "l1"; "l1"; "u1" ]

let tc_for_dtypes ~dims ~threads ~elements_per_thread ~opts ~swizzle dtypes =
  List.map
    (fun (dtype_in, dtype_out) ->
      tensor_core ~dims ~threads ~elements_per_thread ~dtype_in ~dtype_out ~opts
        ~swizzle)
    dtypes

let cuda_81616 =
  let swizzle =
    ( ( [ "r1"; "r2"; "l2"; "l3"; "l4" ],
        [ "u1"; "r3" ],
        [ "l0"; "l1"; "u0"; "r0" ] ),
      ( [ "r1"; "r2"; "u0"; "l0"; "l1" ],
        [ "r0"; "r3" ],
        [ "l2"; "l3"; "l4"; "u1" ] ) )
  in
  tc_for_dtypes ~dims:(8, 16, 16) ~threads:32 ~elements_per_thread:(8, 4, 4)
    ~opts:cuda_tc_opts ~swizzle
    [
      (Dtype.Float16, Dtype.Float32);
      (Dtype.Bfloat16, Dtype.Float32);
      (Dtype.Float16, Dtype.Float16);
    ]

let cuda_81632_f8 =
  let swizzle =
    ( ( [ "r2"; "r3"; "l2"; "l3"; "l4" ],
        [ "u1"; "r4" ],
        [ "l0"; "l1"; "u0"; "r0"; "r1" ] ),
      ( [ "r2"; "r3"; "u0"; "l0"; "l1" ],
        [ "r1"; "r4" ],
        [ "l2"; "l3"; "l4"; "u1"; "r0" ] ) )
  in
  tc_for_dtypes ~dims:(8, 16, 32) ~threads:32 ~elements_per_thread:(16, 8, 4)
    ~opts:cuda_tc_opts ~swizzle
    [ (Dtype.Fp8e4m3, Dtype.Float32); (Dtype.Fp8e5m2, Dtype.Float32) ]

let cuda_8168_f16 =
  let swizzle =
    ( ([ "r1"; "r2"; "l2"; "l3"; "l4" ], [ "r0"; "u1" ], [ "l0"; "l1"; "u0" ]),
      ([ "r1"; "r2"; "u0"; "l0"; "l1" ], [ "u1"; "r0" ], [ "l2"; "l3"; "l4" ])
    )
  in
  tc_for_dtypes ~dims:(8, 16, 8) ~threads:32 ~elements_per_thread:(4, 2, 4)
    ~opts:cuda_tc_opts ~swizzle
    [ (Dtype.Float16, Dtype.Float32); (Dtype.Float16, Dtype.Float16) ]

let cuda_8168_tf32 =
  let swizzle =
    ( ([ "r0"; "r1"; "l2"; "l3"; "l4" ], [ "u1"; "r2" ], [ "l0"; "l1"; "u0" ]),
      ([ "r0"; "r1"; "u0"; "l0"; "l1" ], [ "u1"; "r2" ], [ "l2"; "l3"; "l4" ])
    )
  in
  [
    tensor_core ~dims:(8, 16, 8) ~threads:32 ~elements_per_thread:(4, 2, 4)
      ~dtype_in:Dtype.Float32 ~dtype_out:Dtype.Float32 ~opts:cuda_tc_opts
      ~swizzle;
  ]

let cuda_sm75 = cuda_8168_f16
let cuda_sm80 = cuda_81616 @ cuda_8168_f16 @ cuda_8168_tf32
let cuda_sm89 = cuda_sm80 @ cuda_81632_f8

let amd_rdna3 =
  let swizzle =
    ( ( [ "l4"; "u0"; "u1"; "u2"; "l0" ],
        [ "r1"; "r2"; "r3" ],
        [ "l1"; "l2"; "l3"; "r0" ] ),
      ( [ "l0"; "l1"; "l2"; "l3"; "l4" ],
        [ "r1"; "r2"; "r3" ],
        [ "u0"; "u1"; "u2"; "r0" ] ) )
  in
  tc_for_dtypes ~dims:(16, 16, 16) ~threads:32 ~elements_per_thread:(16, 16, 8)
    ~opts:[ "l0"; "l0"; "l0"; "l0"; "l1"; "u1"; "u1"; "u1" ]
    ~swizzle
    [
      (Dtype.Float16, Dtype.Float32);
      (Dtype.Float16, Dtype.Float16);
      (Dtype.Bfloat16, Dtype.Float32);
    ]

let amd_rdna4 =
  let swizzle =
    ( ( [ "u0"; "u1"; "u2"; "l4"; "r2" ],
        [ "r0"; "r1"; "r3" ],
        [ "l0"; "l1"; "l2"; "l3" ] ),
      ( [ "l0"; "l1"; "l2"; "l3"; "r2" ],
        [ "r0"; "r1"; "r3" ],
        [ "l4"; "u0"; "u1"; "u2" ] ) )
  in
  tc_for_dtypes ~dims:(16, 16, 16) ~threads:32 ~elements_per_thread:(8, 8, 8)
    ~opts:[ "l0"; "l0"; "l0"; "l0"; "u1"; "u1"; "u1"; "l1" ]
    ~swizzle
    [
      (Dtype.Float16, Dtype.Float32);
      (Dtype.Float16, Dtype.Float16);
      (Dtype.Bfloat16, Dtype.Float32);
      (Dtype.Bfloat16, Dtype.Bfloat16);
    ]

let amd_cdna_161616 =
  let swizzle =
    ( ( [ "u0"; "u1"; "l4"; "l5"; "r2"; "r3" ],
        [ "r0"; "r1" ],
        [ "l0"; "l1"; "l2"; "l3" ] ),
      ( [ "l0"; "l1"; "l2"; "l3"; "r2"; "r3" ],
        [ "r0"; "r1" ],
        [ "l4"; "l5"; "u0"; "u1" ] ) )
  in
  tc_for_dtypes ~dims:(16, 16, 16) ~threads:64 ~elements_per_thread:(4, 4, 4)
    ~opts:[ "l0"; "l0"; "l0"; "l0"; "u1"; "u1"; "l1"; "l1" ]
    ~swizzle
    [ (Dtype.Float16, Dtype.Float32); (Dtype.Bfloat16, Dtype.Float32) ]

let amd_cdna_161632 =
  let swizzle =
    ( ( [ "u0"; "u1"; "l4"; "l5"; "r3"; "r4" ],
        [ "r0"; "r1" ],
        [ "l0"; "l1"; "l2"; "l3"; "r2" ] ),
      ( [ "l0"; "l1"; "l2"; "l3"; "r3"; "r4" ],
        [ "r0"; "r1" ],
        [ "l4"; "l5"; "u0"; "u1"; "r2" ] ) )
  in
  tc_for_dtypes ~dims:(16, 16, 32) ~threads:64 ~elements_per_thread:(8, 8, 4)
    ~opts:[ "l0"; "l0"; "l0"; "l0"; "u1"; "u1"; "l1"; "l1" ]
    ~swizzle
    [
      (Dtype.Fp8e5m2, Dtype.Float32);
      (Dtype.Fp8e4m3, Dtype.Float32);
      (Dtype.Float16, Dtype.Float32);
      (Dtype.Bfloat16, Dtype.Float32);
    ]

let amd_cdna_1616128 =
  let swizzle =
    ( ( [ "u0"; "u1"; "l4"; "l5"; "r5"; "r6" ],
        [ "r0"; "r1" ],
        [ "l0"; "l1"; "l2"; "l3"; "r2"; "r3"; "r4" ] ),
      ( [ "l0"; "l1"; "l2"; "l3"; "r5"; "r6" ],
        [ "r0"; "r1" ],
        [ "l4"; "l5"; "u0"; "u1"; "r2"; "r3"; "r4" ] ) )
  in
  tc_for_dtypes ~dims:(16, 16, 128) ~threads:64 ~elements_per_thread:(32, 32, 4)
    ~opts:[ "l0"; "l0"; "l0"; "l0"; "u1"; "u1"; "l1"; "l1" ]
    ~swizzle
    [ (Dtype.Fp8e5m2, Dtype.Float32); (Dtype.Fp8e4m3, Dtype.Float32) ]

(* CDNA3 supports fp8 variants (first two) from 16x16x32 plus all 16x16x16. *)
let amd_cdna3 =
  List.filteri (fun i _ -> i < 2) amd_cdna_161632 @ amd_cdna_161616

let amd_cdna4 = amd_cdna_1616128 @ amd_cdna_161632 @ amd_cdna_161616

let metal_tc =
  let swizzle =
    ( ([ "r1"; "l1"; "l2"; "r2"; "l4" ], [ "r0" ], [ "u0"; "l0"; "l3" ]),
      ([ "l0"; "r0"; "r1"; "l3"; "r2" ], [ "u0" ], [ "l1"; "l2"; "l4" ]) )
  in
  tc_for_dtypes ~dims:(8, 8, 8) ~threads:32 ~elements_per_thread:(2, 2, 2)
    ~opts:[ "u0"; "l0"; "l1"; "l1"; "l0"; "l1" ]
    ~swizzle
    [
      (Dtype.Float32, Dtype.Float32);
      (Dtype.Float16, Dtype.Float32);
      (Dtype.Float16, Dtype.Float16);
      (Dtype.Bfloat16, Dtype.Float32);
      (Dtype.Bfloat16, Dtype.Bfloat16);
    ]

let amx =
  let swizzle =
    ( ([], [ "u0"; "u1"; "u2"; "u3"; "u4"; "u5"; "u6"; "u7" ], []),
      ([], [ "u4"; "u5"; "u6"; "u7"; "u0"; "u1"; "u2"; "u3" ], []) )
  in
  let dtype = Dtype.Float32 in
  let size = 64 / Dtype.itemsize { Dtype.scalar = dtype; count = 1 } in
  [
    tensor_core ~dims:(size, size, 1) ~threads:1
      ~elements_per_thread:(size, size, size * size)
      ~dtype_in:dtype ~dtype_out:dtype
      ~opts:[ "u0"; "u0"; "u0"; "u0"; "u1"; "u1"; "u1"; "u1" ]
      ~swizzle;
  ]

let intel_tc =
  let swizzle =
    ( ([ "r1"; "r2"; "r3" ], [ "u0"; "u1"; "u2" ], [ "l0"; "l1"; "l2"; "r0" ]),
      ([ "l0"; "l1"; "l2" ], [ "r1"; "r2"; "r3" ], [ "u0"; "u1"; "u2"; "r0" ])
    )
  in
  [
    tensor_core ~dims:(8, 8, 16) ~threads:8 ~elements_per_thread:(16, 16, 8)
      ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
      ~opts:[ "l0"; "l0"; "l0"; "u1"; "u1"; "u1" ]
      ~swizzle;
  ]

type cuda_arch = SM75 | SM80 | SM89
type amd_arch = RDNA3 | RDNA4 | CDNA3 | CDNA4

let cuda_tensor_cores = function
  | SM89 -> cuda_sm89
  | SM80 -> cuda_sm80
  | SM75 -> cuda_sm75

let amd_tensor_cores = function
  | CDNA4 -> amd_cdna4
  | CDNA3 -> amd_cdna3
  | RDNA4 -> amd_rdna4
  | RDNA3 -> amd_rdna3

let parse_cuda_arch arch =
  let buf = Buffer.create (String.length arch) in
  String.iter
    (fun c -> if c >= '0' && c <= '9' then Buffer.add_char buf c else ())
    arch;
  if Buffer.length buf = 0 then None
  else int_of_string_opt (Buffer.contents buf)

let default_cuda_arch () =
  let raw =
    match Sys.getenv_opt "CUDA_ARCH" with
    | Some arch when String.trim arch <> "" -> String.trim arch
    | _ -> (
        match Sys.getenv_opt "CUDA_SM" with
        | Some arch when String.trim arch <> "" -> String.trim arch
        | _ -> "")
  in
  match parse_cuda_arch raw with
  | Some ver when ver >= 89 -> Some SM89
  | Some ver when ver >= 80 -> Some SM80
  | Some ver when ver >= 75 -> Some SM75
  | _ -> None

let getenv_bool ~default name =
  match Sys.getenv_opt name with
  | None -> default
  | Some raw -> (
      match int_of_string_opt (String.trim raw) with
      | Some v -> v <> 0
      | None -> default)

let clang_has_amx = getenv_bool ~default:false "AMX"
let clang_tensor_cores = if clang_has_amx then amx else []
let threads_enabled () = getenv_bool ~default:true "THREADS"

let cpu_count =
  lazy
    (match Sys.getenv_opt "CPU_COUNT" with
    | Some raw -> (
        match int_of_string_opt (String.trim raw) with
        | Some n when n > 0 -> n
        | _ -> 1)
    | None -> (
        let read_count cmd =
          try
            let ic = Unix.open_process_in cmd in
            let line =
              try Some (String.trim (input_line ic)) with End_of_file -> None
            in
            let _ = Unix.close_process_in ic in
            match line with Some s -> int_of_string_opt s | None -> None
          with Unix.Unix_error _ | Sys_error _ -> None
        in
        match read_count "getconf _NPROCESSORS_ONLN 2>/dev/null" with
        | Some n when n > 0 -> n
        | _ -> 1))

(* ───── Language Configuration ───── *)

type wmma_config = {
  wmma_name : string;
  wmma_n : int;
  wmma_m : int;
  wmma_k : int;
  wmma_dtype_in : Dtype.scalar;
  wmma_dtype_out : Dtype.scalar;
  wmma_upcast_a : int;
  wmma_upcast_b : int;
  wmma_upcast_c : int;
}

type render_meta = {
  uses_gid : bool;
  uses_lid : bool;
  uses_tid : bool;
  uses_half : bool;
  uses_bf16 : bool;
  uses_fp8 : bool;
  uses_infinity_nan : bool;
  vector_types : Dtype.t list;
  wmma_configs : wmma_config list;
  ocml_funcs : (string * Dtype.t) list;
  launch_bounds : int;
}

type non_native_config = {
  scalars : Dtype.scalar list;
  emulate_alu : Program.instr -> bool;
  emulate_cmp : bool;
  emulate_where : bool;
  rewrite_casts : bool;
}

type rewrite_config = {
  non_native : non_native_config list;
  cast_f64_to_f16 : bool;
  fp8_cast_via_f32 : bool;
  manual_bf16_cast : bool;
}

type lang = {
  kernel_prefix : render_meta -> string;
  buffer_prefix : string;
  buffer_suffix : string;
  arg_int_prefix : string;
  smem_align : string;
  smem_prefix : string;
  barrier : string;
  gep_arr_threshold : int;
  extra_params : render_meta -> string list;
  param_annot : int -> string option;
  code_for_workitem : Ir.special_dim -> string;
  type_map : Dtype.t -> string;
  render_const : Program.const -> Dtype.t -> string;
  render_cast : src:Dtype.t -> dst:Dtype.t -> string -> string;
  render_bitcast : src:Dtype.t -> dst:Dtype.t -> string -> string;
  render_vector : Dtype.t -> string list -> string;
  code_for_op : Program.instr -> (int -> string) -> string option;
  rewrite : rewrite_config;
  fixed_abi : bool;
}

let base_lang =
  {
    kernel_prefix = (fun _ -> "void");
    buffer_prefix = "";
    buffer_suffix = "";
    arg_int_prefix = "const int";
    smem_align = "";
    smem_prefix = "";
    barrier = "";
    gep_arr_threshold = 4;
    extra_params = (fun _ -> []);
    param_annot = (fun _ -> None);
    code_for_workitem = (fun _ -> failwith err_base_no_workitem);
    type_map = (fun _ -> failwith err_base_type_map);
    render_const = (fun _ _ -> failwith err_base_render_const);
    render_cast = (fun ~src:_ ~dst:_ _ -> failwith err_base_render_cast);
    render_bitcast = (fun ~src:_ ~dst:_ _ -> failwith err_base_render_bitcast);
    render_vector = (fun _ _ -> failwith err_base_render_vector);
    code_for_op = (fun _ _ -> None);
    rewrite =
      {
        non_native = [];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = false;
      };
    fixed_abi = false;
  }

(* ───── Type Mappings ───── *)

let base_scalar_name = function
  | Dtype.Void -> "void"
  | Dtype.Bool -> "bool"
  | Dtype.Int8 -> "signed char"
  | Dtype.Int16 -> "short"
  | Dtype.Int32 -> "int"
  | Dtype.Int64 -> "long long"
  | Dtype.Uint8 -> "unsigned char"
  | Dtype.Uint16 -> "unsigned short"
  | Dtype.Uint32 -> "unsigned int"
  | Dtype.Uint64 -> "unsigned long long"
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "__bf16"
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Fp8e4m3 -> "unsigned char"
  | Dtype.Fp8e5m2 -> "unsigned char"
  | Dtype.Index -> "long long"

let make_type_map scalar_name (dt : Dtype.t) =
  let base = scalar_name dt.scalar in
  if dt.count > 1 then
    (* Replace spaces with underscores only for vector types (e.g.,
       "unsigned_char4"). Scalar types like "signed char" must keep spaces to be
       valid C. *)
    let base_id = String.map (fun c -> if c = ' ' then '_' else c) base in
    Printf.sprintf "%s%d" base_id dt.count
  else base

(* Clang uses C99/_Bool for bool and __fp16 for half precision. These are the
   types that Clang's LLVM backend optimizes for, unlike 'bool' or 'half'. *)
let clang_scalar_name = function
  | Dtype.Bool -> "_Bool"
  | Dtype.Float16 -> "__fp16"
  | s -> base_scalar_name s

let clang_type_map = make_type_map clang_scalar_name

(* OpenCL vector types require short names (char4, uchar8, not signed_char4). *)
let opencl_scalar_name = function
  | Dtype.Void -> "void"
  | Dtype.Bool -> "bool"
  | Dtype.Int8 -> "char"
  | Dtype.Int16 -> "short"
  | Dtype.Int32 -> "int"
  | Dtype.Int64 -> "long"
  | Dtype.Uint8 -> "uchar"
  | Dtype.Uint16 -> "ushort"
  | Dtype.Uint32 -> "uint"
  | Dtype.Uint64 -> "ulong"
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "ushort" (* bfloat16 stored as ushort in OpenCL *)
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Fp8e4m3 -> "uchar"
  | Dtype.Fp8e5m2 -> "uchar"
  | Dtype.Index -> "long"

let opencl_type_map = make_type_map opencl_scalar_name

(* MSL uses short type names (long, not long long). Falls back to base for types
   that already match. *)
let metal_scalar_name = function
  | Dtype.Int8 -> "char"
  | Dtype.Uint8 -> "uchar"
  | Dtype.Uint16 -> "ushort"
  | Dtype.Uint32 -> "uint"
  | Dtype.Uint64 -> "ulong"
  | Dtype.Int64 -> "long"
  | Dtype.Bfloat16 -> "bfloat"
  | Dtype.Index -> "long"
  | s -> base_scalar_name s

let metal_type_map = make_type_map metal_scalar_name

let cuda_scalar_name = function
  | Dtype.Bfloat16 -> "nv_bfloat16"
  | Dtype.Fp8e4m3 -> "__nv_fp8_e4m3"
  | Dtype.Fp8e5m2 -> "__nv_fp8_e5m2"
  | s -> base_scalar_name s

let cuda_type_map = make_type_map cuda_scalar_name

(* _Float16: AMD compilers support this C23 type natively, unlike __fp16. *)
let amd_scalar_name = function
  | Dtype.Bfloat16 -> "hip_bfloat16"
  | Dtype.Fp8e4m3 -> "hip_fp8"
  | Dtype.Fp8e5m2 -> "hip_bf8"
  | Dtype.Float16 -> "_Float16"
  | s -> base_scalar_name s

let amd_type_map = make_type_map amd_scalar_name

(* ───── AMD WMMA Helpers ───── *)

let amd_wmma_type_map = function
  | Dtype.Bfloat16 -> "bf16"
  | Dtype.Float32 -> "f32"
  | Dtype.Float16 -> "f16"
  | Dtype.Fp8e4m3 -> "_fp8_fp8"
  | Dtype.Fp8e5m2 -> "_bf8_bf8"
  | Dtype.Void | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32
  | Uint64 | Float64 | Index ->
      "f32"

let amd_wmma_out_type_map = function
  | Dtype.Float32 -> "f32"
  | Dtype.Float16 -> "f16"
  | Dtype.Void | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32
  | Uint64 | Float64 | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Index ->
      "f32"

(* ───── Constant Rendering ───── *)

let truncate_uint32 i = Int64.to_int (Int64.logand (Int64.of_int i) 0xFFFFFFFFL)
let truncate_uint64 i = Int64.logand (Int64.of_int i) 0xFFFFFFFFFFFFFFFFL

let render_float_literal f =
  let s = Printf.sprintf "%g" f in
  if String.contains s '.' || String.contains s 'e' || String.contains s 'E'
  then s
  else s ^ ".0"

(* bfloat16 is the upper 16 bits of float32, with round-to-nearest-even. *)
let float_to_bf16_bits (f : float) : int =
  let bits = Int32.bits_of_float f in
  if not (Float.is_finite f) then
    Int32.to_int (Int32.shift_right_logical bits 16)
  else
    let rounded =
      Int32.add bits
        (Int32.add 0x7fffl
           (Int32.logand (Int32.shift_right_logical bits 16) 1l))
    in
    Int32.to_int (Int32.shift_right_logical rounded 16)

(* Small types (int8/16, uint8/16, fp16/bf16/fp8) are rendered as larger type
   literals and cast, since C lacks literals for these types. *)
let render_const_with_cast ~infinity ~nan ~render_cast (c : Program.const)
    (dt : Dtype.t) : string =
  let src_like scalar = { dt with scalar } in
  let cast_literal ~src expr =
    Printf.sprintf "(%s)" (render_cast ~src ~dst:dt expr)
  in
  match c with
  | Bool b -> if b then "1" else "0"
  | Int i -> (
      match dt.scalar with
      | Dtype.Int64 -> Printf.sprintf "%Ldll" (Int64.of_int i)
      | Dtype.Uint64 -> Printf.sprintf "%Luull" (truncate_uint64 i)
      | Dtype.Uint32 -> Printf.sprintf "%uu" (truncate_uint32 i)
      (* Small integer types: render as larger type and cast *)
      | Dtype.Uint8 | Dtype.Uint16 ->
          cast_literal ~src:(src_like Dtype.Uint32) (Printf.sprintf "%du" i)
      | Dtype.Int8 | Dtype.Int16 ->
          cast_literal ~src:(src_like Dtype.Int32) (string_of_int i)
      | _ -> string_of_int i)
  | Float f -> (
      if Float.is_nan f then cast_literal ~src:(src_like Dtype.Float32) nan
      else if f = Float.infinity then
        cast_literal ~src:(src_like Dtype.Float32) infinity
      else if f = Float.neg_infinity then
        cast_literal ~src:(src_like Dtype.Float32) ("-" ^ infinity)
      else
        let lit = render_float_literal f in
        match dt.scalar with
        | Dtype.Float64 -> lit
        (* fp16/bf16/fp8: render as float and cast *)
        | Dtype.Float16 | Dtype.Bfloat16 | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 ->
            cast_literal ~src:(src_like Dtype.Float32) (lit ^ "f")
        | _ -> lit ^ "f")

(* OpenCL: bf16 constants render as their ushort bit pattern. *)
let render_const_opencl ~render_cast (c : Program.const) (dt : Dtype.t) : string
    =
  match (c, dt.scalar) with
  | Float f, Dtype.Bfloat16 -> Printf.sprintf "%uu" (float_to_bf16_bits f)
  | _ ->
      render_const_with_cast ~infinity:"INFINITY" ~nan:"NAN" ~render_cast c dt

(* ───── Code Generation for Ops ───── *)

let is_half_or_bfloat (dt : Dtype.t) =
  match dt.scalar with Dtype.Float16 | Dtype.Bfloat16 -> true | _ -> false

let is_f64 (dt : Dtype.t) = dt.scalar = Dtype.Float64

let base_code_for_op instr r =
  let unary fn src = Some (Printf.sprintf "%s(%s)" fn (r src)) in
  let binary_fn fn lhs rhs =
    Some (Printf.sprintf "%s(%s,%s)" fn (r lhs) (r rhs))
  in
  let binary_op op lhs rhs =
    Some (Printf.sprintf "(%s%s%s)" (r lhs) op (r rhs))
  in
  let ternary cond a b =
    Some (Printf.sprintf "(%s?%s:%s)" (r cond) (r a) (r b))
  in
  match instr with
  | Program.Neg { src; _ } -> Some (Printf.sprintf "(-%s)" (r src))
  | Exp2 { src; _ } -> unary "exp2" src
  | Log2 { src; _ } -> unary "log2" src
  | Sin { src; _ } -> unary "sin" src
  | Sqrt { src; _ } -> unary "sqrt" src
  | Recip { src; _ } -> Some (Printf.sprintf "(1/%s)" (r src))
  | Trunc { src; _ } -> unary "trunc" src
  | Add { lhs; rhs; _ } -> binary_op "+" lhs rhs
  | Sub { lhs; rhs; _ } -> binary_op "-" lhs rhs
  | Mul { lhs; rhs; _ } -> binary_op "*" lhs rhs
  | Fdiv { lhs; rhs; _ } -> binary_op "/" lhs rhs
  | Idiv { lhs; rhs; _ } -> binary_op "/" lhs rhs
  | Mod { lhs; rhs; _ } -> binary_op "%" lhs rhs
  | Max { lhs; rhs; _ } -> binary_fn "max" lhs rhs
  | Pow { lhs; rhs; _ } -> binary_fn "pow" lhs rhs
  | Shl { lhs; rhs; _ } -> binary_op "<<" lhs rhs
  | Shr { lhs; rhs; _ } -> binary_op ">>" lhs rhs
  | And { lhs; rhs; _ } -> binary_op "&" lhs rhs
  | Or { lhs; rhs; _ } -> binary_op "|" lhs rhs
  | Xor { lhs; rhs; _ } -> binary_op "^" lhs rhs
  | Threefry { lhs; rhs; _ } -> binary_fn "threefry" lhs rhs
  | Cmplt { lhs; rhs; _ } -> binary_op "<" lhs rhs
  | Cmpeq { lhs; rhs; _ } -> binary_op "==" lhs rhs
  | Cmpne { lhs; rhs; _ } -> binary_op "!=" lhs rhs
  | Where { cond; a; b; _ } -> ternary cond a b
  | Mulacc { a; b; c; _ } ->
      Some (Printf.sprintf "((%s*%s)+%s)" (r a) (r b) (r c))
  | _ -> None

let clang_code_for_op instr r =
  match instr with
  | Program.Sqrt { src; dtype } ->
      let f = if is_f64 dtype then "__builtin_sqrt" else "__builtin_sqrtf" in
      Some (Printf.sprintf "%s(%s)" f (r src))
  | Trunc { src; dtype } ->
      let f = if is_f64 dtype then "__builtin_trunc" else "__builtin_truncf" in
      Some (Printf.sprintf "%s(%s)" f (r src))
  | Fdiv { lhs; rhs; _ } -> Some (Printf.sprintf "(%s/%s)" (r lhs) (r rhs))
  | _ -> base_code_for_op instr r

let metal_code_for_op instr r =
  match instr with
  | Program.Sin { src; _ } -> Some (Printf.sprintf "precise::sin(%s)" (r src))
  | _ -> base_code_for_op instr r

(* CUDA half-precision intrinsics: hexp2, hlog2, hsin, hsqrt, hrcp, htrunc. *)
let cuda_code_for_op instr r =
  let cuda_half_op op dtype =
    if is_half_or_bfloat dtype then "h" ^ op else op
  in
  let cuda_unary op src dtype =
    Some (Printf.sprintf "%s(%s)" (cuda_half_op op dtype) (r src))
  in
  match instr with
  | Program.Exp2 { src; dtype } -> cuda_unary "exp2" src dtype
  | Log2 { src; dtype } -> cuda_unary "log2" src dtype
  | Sin { src; dtype } -> cuda_unary "sin" src dtype
  | Sqrt { src; dtype } -> cuda_unary "sqrt" src dtype
  | Recip { src; dtype } ->
      if is_half_or_bfloat dtype then Some (Printf.sprintf "hrcp(%s)" (r src))
      else Some (Printf.sprintf "(1/%s)" (r src))
  | Trunc { src; dtype } -> cuda_unary "trunc" src dtype
  | _ -> base_code_for_op instr r

(* AMD OCML: __ocml_{op}_f{16,32,64} *)
let ocml_func op (dt : Dtype.t) =
  let bits =
    match dt.scalar with Dtype.Float16 -> 16 | Dtype.Float64 -> 64 | _ -> 32
  in
  Printf.sprintf "__ocml_%s_f%d" op bits

let amd_code_for_op instr r =
  let amd_unary op src dtype =
    Some (Printf.sprintf "%s(%s)" (ocml_func op dtype) (r src))
  in
  match instr with
  | Program.Exp2 { src; dtype } -> amd_unary "exp2" src dtype
  | Log2 { src; dtype } -> amd_unary "log2" src dtype
  | Sin { src; dtype } -> amd_unary "sin" src dtype
  | Sqrt { src; dtype } -> amd_unary "sqrt" src dtype
  | Trunc { src; dtype } -> amd_unary "trunc" src dtype
  | _ -> base_code_for_op instr r

let non_native_bf16_all : non_native_config =
  {
    scalars = [ Dtype.Bfloat16 ];
    emulate_alu = (fun _ -> true);
    emulate_cmp = true;
    emulate_where = true;
    rewrite_casts = true;
  }

(* Metal has hardware bf16 for basic arithmetic but not for transcendentals, so
   only those are promoted to f32. *)
let non_native_bf16_metal : non_native_config =
  {
    scalars = [ Dtype.Bfloat16 ];
    emulate_alu =
      (function Program.Exp2 _ | Log2 _ | Sin _ | Sqrt _ -> true | _ -> false);
    emulate_cmp = false;
    emulate_where = false;
    rewrite_casts = false;
  }

let non_native_fp8_all : non_native_config =
  {
    scalars = [ Dtype.Fp8e4m3; Dtype.Fp8e5m2 ];
    emulate_alu = (fun _ -> true);
    emulate_cmp = true;
    emulate_where = true;
    rewrite_casts = true;
  }

(* CUDA handles fp8 casts natively via __nv_fp8 types, so only ALU/cmp/where
   need promotion to f32. *)
let non_native_fp8_nocast : non_native_config =
  {
    scalars = [ Dtype.Fp8e4m3; Dtype.Fp8e5m2 ];
    emulate_alu = (fun _ -> true);
    emulate_cmp = true;
    emulate_where = true;
    rewrite_casts = false;
  }

(* ───── Render Utilities ───── *)

let vector_ctor_cast ~type_map ~delims (dt : Dtype.t) elems =
  let left, right = delims in
  Printf.sprintf "(%s)%s%s%s" (type_map dt) left (String.concat "," elems) right

let vector_ctor_make ~type_map (dt : Dtype.t) elems =
  Printf.sprintf "make_%s(%s)" (type_map dt) (String.concat "," elems)

let vector_ctor_fn ~type_map (dt : Dtype.t) elems =
  Printf.sprintf "%s(%s)" (type_map dt) (String.concat "," elems)

(* ───── Language Configurations ───── *)

let clang_lang =
  let render_cast ~src:_ ~dst expr =
    if dst.Dtype.count > 1 then
      Printf.sprintf "__builtin_convertvector(%s, %s)" expr (clang_type_map dst)
    else Printf.sprintf "(%s)(%s)" (clang_type_map dst) expr
  in
  {
    base_lang with
    buffer_suffix = " restrict";
    gep_arr_threshold = 0;
    code_for_workitem = (fun _ -> failwith err_clang_no_gpu);
    type_map = clang_type_map;
    render_const =
      render_const_with_cast ~infinity:"__builtin_inff()"
        ~nan:{|__builtin_nanf("")|} ~render_cast;
    render_cast;
    render_bitcast =
      (fun ~src ~dst expr ->
        Printf.sprintf "__builtin_bit_cast(%s, (%s)(%s))" (clang_type_map dst)
          (clang_type_map src) expr);
    render_vector = vector_ctor_cast ~type_map:clang_type_map ~delims:("{", "}");
    code_for_op = clang_code_for_op;
    rewrite =
      {
        non_native = [ non_native_bf16_all ];
        cast_f64_to_f16 = true;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = true;
      };
    fixed_abi = true;
  }

let opencl_lang =
  let render_cast ~src:_ ~dst expr =
    Printf.sprintf "(%s)(%s)" (opencl_type_map dst) expr
  in
  {
    base_lang with
    kernel_prefix = (fun _ -> "__kernel void");
    buffer_prefix = "__global ";
    smem_align = "__attribute__ ((aligned (16))) ";
    smem_prefix = "__local ";
    barrier = "barrier(CLK_LOCAL_MEM_FENCE);";
    code_for_workitem =
      (fun dim ->
        let axis = Ir.special_axis dim in
        match dim with
        | Ir.Group_id _ -> Printf.sprintf "get_group_id(%d)" axis
        | Ir.Local_id _ -> Printf.sprintf "get_local_id(%d)" axis
        | Ir.Global_idx _ -> Printf.sprintf "get_global_id(%d)" axis);
    type_map = opencl_type_map;
    render_const = render_const_opencl ~render_cast;
    render_cast;
    render_bitcast =
      (fun ~src ~dst expr ->
        Printf.sprintf "as_%s((%s)(%s))" (opencl_type_map dst)
          (opencl_type_map src) expr);
    render_vector = vector_ctor_cast ~type_map:opencl_type_map ~delims:("(", ")");
    code_for_op = base_code_for_op;
    rewrite =
      {
        non_native = [ non_native_bf16_all ];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = true;
      };
  }

let intel_lang =
  let render_cast ~src ~dst expr =
    match (src.Dtype.scalar, dst.Dtype.scalar) with
    | Dtype.Float32, Dtype.Bfloat16 ->
        Printf.sprintf "intel_convert_bfloat16_as_ushort(%s)" expr
    | Dtype.Bfloat16, Dtype.Float32 ->
        Printf.sprintf "intel_convert_as_bfloat16_float(%s)" expr
    | _ -> Printf.sprintf "(%s)(%s)" (opencl_type_map dst) expr
  in
  {
    opencl_lang with
    kernel_prefix =
      (fun _ -> "__attribute__((intel_reqd_sub_group_size(8)))\n__kernel void");
    render_cast;
    render_const = render_const_opencl ~render_cast;
    rewrite =
      {
        non_native = [ non_native_bf16_all ];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = false;
      };
  }

let metal_lang =
  let render_cast ~src:_ ~dst expr =
    Printf.sprintf "(%s)(%s)" (metal_type_map dst) expr
  in
  {
    base_lang with
    kernel_prefix = (fun _ -> "kernel void");
    buffer_prefix = "device ";
    arg_int_prefix = "constant int&";
    smem_prefix = "threadgroup __attribute__((aligned(16))) ";
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    extra_params =
      (fun _meta ->
        [
          "uint3 gid [[threadgroup_position_in_grid]]";
          "uint3 lid [[thread_position_in_threadgroup]]";
        ]);
    code_for_workitem =
      (fun dim ->
        let prefix =
          match dim with
          | Ir.Group_id _ -> "gid"
          | Ir.Local_id _ -> "lid"
          | Ir.Global_idx _ -> "tid"
        in
        Printf.sprintf "%s.%c" prefix
          (Char.chr (Char.code 'x' + Ir.special_axis dim)));
    type_map = metal_type_map;
    render_const =
      render_const_with_cast ~infinity:"INFINITY" ~nan:"NAN" ~render_cast;
    render_cast;
    render_bitcast =
      (fun ~src ~dst expr ->
        Printf.sprintf "as_type<%s>((%s)(%s))" (metal_type_map dst)
          (metal_type_map src) expr);
    render_vector = vector_ctor_fn ~type_map:metal_type_map;
    code_for_op = metal_code_for_op;
    rewrite =
      {
        non_native = [ non_native_bf16_metal ];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = false;
      };
  }

let cuda_lang =
  let render_cast ~src:_ ~dst expr =
    Printf.sprintf "(%s)(%s)" (cuda_type_map dst) expr
  in
  {
    base_lang with
    kernel_prefix =
      (fun meta ->
        if meta.launch_bounds > 1 then
          Printf.sprintf "extern \"C\" __global__ void __launch_bounds__(%d)"
            meta.launch_bounds
        else "extern \"C\" __global__ void");
    smem_prefix = "__shared__ __align__(16) ";
    barrier = "__syncthreads();";
    gep_arr_threshold = 8;
    code_for_workitem =
      (fun dim ->
        let c = Char.chr (Char.code 'x' + Ir.special_axis dim) in
        match dim with
        | Ir.Group_id _ -> Printf.sprintf "blockIdx.%c" c
        | Ir.Local_id _ -> Printf.sprintf "threadIdx.%c" c
        | Ir.Global_idx _ ->
            Printf.sprintf "(blockIdx.%c*blockDim.%c+threadIdx.%c)" c c c);
    type_map = cuda_type_map;
    render_const =
      render_const_with_cast ~infinity:"INFINITY" ~nan:"NAN" ~render_cast;
    render_cast;
    render_bitcast =
      (fun ~src ~dst expr ->
        Printf.sprintf "tg_bitcast<%s>((%s)(%s))" (cuda_type_map dst)
          (cuda_type_map src) expr);
    render_vector = vector_ctor_make ~type_map:cuda_type_map;
    code_for_op = cuda_code_for_op;
    rewrite =
      {
        non_native = [ non_native_fp8_nocast ];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = true;
        manual_bf16_cast = false;
      };
  }

let amd_lang =
  let render_cast ~src ~dst expr =
    let fp8_index dt =
      match dt.Dtype.scalar with Dtype.Fp8e5m2 -> 1 | _ -> 0
    in
    match (src.Dtype.scalar, dst.Dtype.scalar) with
    | (Dtype.Fp8e4m3 | Dtype.Fp8e5m2), Dtype.Float32 ->
        Printf.sprintf "fp8_to_f32(%s, %d)" expr (fp8_index src)
    | Dtype.Float32, (Dtype.Fp8e4m3 | Dtype.Fp8e5m2) ->
        Printf.sprintf "f32_to_fp8(%s, %d)" expr (fp8_index dst)
    | _ -> Printf.sprintf "(%s)(%s)" (amd_type_map dst) expr
  in
  {
    base_lang with
    kernel_prefix =
      (fun meta ->
        Printf.sprintf
          {|extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, %d)))|}
          (max 1 meta.launch_bounds));
    smem_prefix = "__attribute__((shared, aligned(16))) ";
    (* Explicit release-acquire fences required around s_barrier for correctness
       on AMD; unlike CUDA's __syncthreads which implies a full fence. *)
    barrier =
      {|__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");|};
    code_for_workitem =
      (fun dim ->
        let axis = Ir.special_axis dim in
        match dim with
        | Ir.Group_id _ -> Printf.sprintf "__ockl_get_group_id(%d)" axis
        | Ir.Local_id _ -> Printf.sprintf "__ockl_get_local_id(%d)" axis
        | Ir.Global_idx _ ->
            Printf.sprintf
              "(__ockl_get_group_id(%d)*__ockl_get_local_size(%d)+__ockl_get_local_id(%d))"
              axis axis axis);
    type_map = amd_type_map;
    render_const =
      render_const_with_cast ~infinity:"__builtin_inff()"
        ~nan:{|__builtin_nanf("")|} ~render_cast;
    render_cast;
    render_bitcast =
      (fun ~src ~dst expr ->
        Printf.sprintf "__builtin_bit_cast(%s, (%s)(%s))" (amd_type_map dst)
          (amd_type_map src) expr);
    render_vector = vector_ctor_make ~type_map:amd_type_map;
    code_for_op = amd_code_for_op;
    rewrite =
      {
        non_native = [ non_native_bf16_all; non_native_fp8_all ];
        cast_f64_to_f16 = false;
        fp8_cast_via_f32 = false;
        manual_bf16_cast = true;
      };
  }

let gep_swizzle = "xyzwabcd"

(* Vector element names: xyzw for the first 4 (OpenCL/GLSL convention),
   abcdefghijkl for 5–16, v16..v31 for wider vectors. *)
let vec_elem_names =
  Array.of_list
    (List.init 16 (fun i -> String.make 1 "xyzwabcdefghijkl".[i])
    @ List.init 16 (fun i -> Printf.sprintf "v%d" (i + 16)))

let vec_elem_name i =
  if i < Array.length vec_elem_names then vec_elem_names.(i)
  else Printf.sprintf "v%d" i

let compute_use_counts (program : Program.t) : int array =
  let counts = Array.make (Array.length program) 0 in
  Array.iter
    (fun instr ->
      List.iter (fun r -> counts.(r) <- counts.(r) + 1) (Program.refs_of instr))
    program;
  counts

let compute_launch_bounds (program : Program.t) : int =
  let rec upper_bound_of_ref ref_idx =
    let map2 f a b =
      match (a, b) with Some x, Some y -> Some (f x y) | _ -> None
    in
    match program.(ref_idx) with
    | Program.Const { value = Int n; _ } when n > 0 -> Some n
    | Define_var { hi; _ } when hi > 0 -> Some hi
    | Cast { src; _ } | Bitcast { src; _ } -> upper_bound_of_ref src
    | Add { lhs; rhs; _ } ->
        map2 ( + ) (upper_bound_of_ref lhs) (upper_bound_of_ref rhs)
    | Mul { lhs; rhs; _ } ->
        map2 ( * ) (upper_bound_of_ref lhs) (upper_bound_of_ref rhs)
    | Max { lhs; rhs; _ } ->
        map2 max (upper_bound_of_ref lhs) (upper_bound_of_ref rhs)
    | _ -> None
  in
  let add_local_dim acc instr =
    match instr with
    | Program.Special { dim = Ir.Local_id _; size; _ } -> (
        match upper_bound_of_ref size with Some n -> acc * n | None -> acc)
    | _ -> acc
  in
  Array.fold_left add_local_dim 1 program

let wmma_config_of_instr ~(index : int) (wmma : Program.instr) =
  match wmma with
  | Program.Wmma
      {
        name;
        dims = n, m, k;
        dtype_in;
        dtype_out;
        upcast_axes = upcast_a, upcast_b, upcast_c;
        _;
      } ->
      let upcast_size axes =
        List.fold_left (fun acc (_, size) -> acc * size) 1 axes
      in
      {
        wmma_name = name;
        wmma_n = n;
        wmma_m = m;
        wmma_k = k;
        wmma_dtype_in = dtype_in;
        wmma_dtype_out = dtype_out;
        wmma_upcast_a = upcast_size upcast_a;
        wmma_upcast_b = upcast_size upcast_b;
        wmma_upcast_c = upcast_size upcast_c;
      }
  | _ ->
      let rendered = Format.asprintf "%a" Program.pp_instr wmma in
      failwith
        (Printf.sprintf
           "wmma_config_of_instr: expected WMMA instruction at %d, got %s" index
           rendered)

type meta_flags = {
  gid : bool;
  lid : bool;
  tid : bool;
  half : bool;
  bf16 : bool;
  fp8 : bool;
  infinity_nan : bool;
}

let no_flags =
  {
    gid = false;
    lid = false;
    tid = false;
    half = false;
    bf16 = false;
    fp8 = false;
    infinity_nan = false;
  }

let add_unique seen key value acc =
  if Hashtbl.mem seen key then acc
  else (
    Hashtbl.add seen key ();
    value :: acc)

let collect_metadata (program : Program.t) : render_meta =
  let vector_types_seen = Hashtbl.create 4 in
  let wmma_seen = Hashtbl.create 4 in
  let ocml_seen = Hashtbl.create 8 in
  let vectors = ref [] in
  let wmmas = ref [] in
  let ocmls = ref [] in

  let scan_instr flags index instr =
    let flags =
      match instr with
      | Program.Special { dim; _ } -> (
          match dim with
          | Ir.Group_id _ -> { flags with gid = true }
          | Ir.Local_id _ -> { flags with lid = true }
          | Ir.Global_idx _ -> { flags with tid = true })
      | Program.Wmma _ as wmma ->
          let cfg = wmma_config_of_instr ~index wmma in
          let key =
            ( cfg.wmma_name,
              cfg.wmma_n,
              cfg.wmma_m,
              cfg.wmma_k,
              cfg.wmma_dtype_in,
              cfg.wmma_dtype_out,
              cfg.wmma_upcast_a,
              cfg.wmma_upcast_b,
              cfg.wmma_upcast_c )
          in
          wmmas := add_unique wmma_seen key cfg !wmmas;
          flags
      | Program.Exp2 { dtype; _ } ->
          ocmls := add_unique ocml_seen ("exp2", dtype) ("exp2", dtype) !ocmls;
          flags
      | Program.Log2 { dtype; _ } ->
          ocmls := add_unique ocml_seen ("log2", dtype) ("log2", dtype) !ocmls;
          flags
      | Program.Sin { dtype; _ } ->
          ocmls := add_unique ocml_seen ("sin", dtype) ("sin", dtype) !ocmls;
          flags
      | Program.Sqrt { dtype; _ } ->
          ocmls := add_unique ocml_seen ("sqrt", dtype) ("sqrt", dtype) !ocmls;
          flags
      | Program.Trunc { dtype; _ } ->
          ocmls := add_unique ocml_seen ("trunc", dtype) ("trunc", dtype) !ocmls;
          flags
      | Program.Const { value = Float f; _ } when not (Float.is_finite f) ->
          { flags with infinity_nan = true }
      | _ -> flags
    in
    match Program.dtype_of instr with
    | Some dt ->
        let flags =
          match dt.scalar with
          | Dtype.Float16 -> { flags with half = true }
          | Dtype.Bfloat16 -> { flags with bf16 = true }
          | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> { flags with fp8 = true }
          | _ -> flags
        in
        if dt.count > 1 then
          vectors := add_unique vector_types_seen dt dt !vectors;
        flags
    | None -> flags
  in
  let flags =
    Array.fold_left
      (fun (flags, index) instr -> (scan_instr flags index instr, index + 1))
      (no_flags, 0) program
    |> fst
  in
  {
    uses_gid = flags.gid;
    uses_lid = flags.lid;
    uses_tid = flags.tid;
    uses_half = flags.half;
    uses_bf16 = flags.bf16;
    uses_fp8 = flags.fp8;
    uses_infinity_nan = flags.infinity_nan;
    vector_types = List.rev !vectors;
    wmma_configs = List.rev !wmmas;
    ocml_funcs = List.rev !ocmls;
    launch_bounds = compute_launch_bounds program;
  }

(* ───── IR Rewrites ───── *)

let float_like (dt : Dtype.t) = { dt with scalar = Dtype.Float32 }
let bool_like (dt : Dtype.t) = { dt with scalar = Dtype.Bool }

let rewrite_non_native_float (config : non_native_config) (program : Program.t)
    : Program.t =
  let is_target dt = List.mem dt.Dtype.scalar config.scalars in
  Program.rebuild
    (fun ~emit ~map_ref instr ->
      let dtype_of_ref ref_idx = Program.dtype_of program.(ref_idx) in
      let cast_to_f32 ref_idx =
        match dtype_of_ref ref_idx with
        | Some dt when dt.scalar = Dtype.Float32 -> map_ref ref_idx
        | Some dt ->
            emit (Program.Cast { src = map_ref ref_idx; dtype = float_like dt })
        | None -> map_ref ref_idx
      in
      let rewrite_cast src_dt src dst =
        let src' = map_ref src in
        let f32 =
          emit (Program.Cast { src = src'; dtype = float_like src_dt })
        in
        emit (Program.Cast { src = f32; dtype = dst })
      in
      let emit_alu_via_f32 instr dt =
        let alu =
          Program.map_alu ~map_ref:cast_to_f32 ~dtype:(float_like dt) instr
        in
        emit (Program.Cast { src = emit alu; dtype = dt })
      in
      match instr with
      | Program.Cast { src; dtype } when config.rewrite_casts -> (
          match dtype_of_ref src with
          | Some src_dt when is_target dtype && src_dt.scalar <> Dtype.Float32
            ->
              Some (rewrite_cast src_dt src dtype)
          | Some src_dt when is_target src_dt && dtype.scalar <> Dtype.Float32
            ->
              Some (rewrite_cast src_dt src dtype)
          | _ -> None)
      | Program.Where { cond; a; b; dtype }
        when config.emulate_where && is_target dtype ->
          let a' = cast_to_f32 a in
          let b' = cast_to_f32 b in
          let float_dt = float_like dtype in
          let where_ref =
            emit
              (Program.Where
                 { cond = map_ref cond; a = a'; b = b'; dtype = float_dt })
          in
          Some (emit (Program.Cast { src = where_ref; dtype }))
      | Program.Cmplt { lhs; rhs; dtype }
      | Program.Cmpeq { lhs; rhs; dtype }
      | Program.Cmpne { lhs; rhs; dtype }
        when config.emulate_cmp -> (
          match (dtype_of_ref lhs, dtype_of_ref rhs) with
          | Some lhs_dt, Some rhs_dt when is_target lhs_dt && is_target rhs_dt
            ->
              let cmp = Program.map_alu ~map_ref:cast_to_f32 ~dtype instr in
              Some (emit cmp)
          | _ -> None)
      | _ when Program.is_alu instr && config.emulate_alu instr -> (
          match Program.dtype_of instr with
          | Some dt when is_target dt -> Some (emit_alu_via_f32 instr dt)
          | _ -> None)
      | _ -> None)
    program

let rewrite_cast_via_f32 (program : Program.t)
    ~(pred : src:Dtype.t -> dst:Dtype.t -> bool) : Program.t =
  Program.rebuild
    (fun ~emit ~map_ref instr ->
      match instr with
      | Program.Cast { src; dtype } -> (
          match Program.dtype_of program.(src) with
          | Some src_dt when pred ~src:src_dt ~dst:dtype ->
              let src' = map_ref src in
              let f32 =
                emit (Program.Cast { src = src'; dtype = float_like dtype })
              in
              Some (emit (Program.Cast { src = f32; dtype }))
          | _ -> None)
      | _ -> None)
    program

let rewrite_cast_f64_to_f16 (program : Program.t) : Program.t =
  rewrite_cast_via_f32 program ~pred:(fun ~src ~dst ->
      src.scalar = Dtype.Float64 && dst.scalar = Dtype.Float16)

let rewrite_fp8_cast_variants (program : Program.t) : Program.t =
  let is_fp8_scalar = function
    | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> true
    | _ -> false
  in
  rewrite_cast_via_f32 program ~pred:(fun ~src ~dst ->
      is_fp8_scalar src.scalar && is_fp8_scalar dst.scalar
      && src.scalar <> dst.scalar)

(* Software bf16<->f32 conversion via bit manipulation with
   round-to-nearest-even. *)
let rewrite_manual_bf16_cast (program : Program.t) : Program.t =
  Program.rebuild
    (fun ~emit ~map_ref instr ->
      let dtype_of_ref ref_idx = Program.dtype_of program.(ref_idx) in
      match instr with
      | Program.Cast { src; dtype } -> (
          match dtype_of_ref src with
          | Some src_dt
            when src_dt.scalar = Dtype.Bfloat16 && dtype.scalar = Dtype.Float32
            ->
              (* bf16 -> f32: bitcast to u16, widen to u32, shift left 16 *)
              let src' = map_ref src in
              let u16_dt = { src_dt with scalar = Dtype.Uint16 } in
              let u32_dt = { src_dt with scalar = Dtype.Uint32 } in
              let u16 = emit (Program.Bitcast { src = src'; dtype = u16_dt }) in
              let u32 = emit (Program.Cast { src = u16; dtype = u32_dt }) in
              let c16 =
                emit (Program.Const { value = Int 16; dtype = u32_dt })
              in
              let shl =
                emit (Program.Shl { lhs = u32; rhs = c16; dtype = u32_dt })
              in
              Some (emit (Program.Bitcast { src = shl; dtype }))
          | Some src_dt
            when src_dt.scalar = Dtype.Float32 && dtype.scalar = Dtype.Bfloat16
            ->
              (* f32 -> bf16: round-to-nearest-even via bit manipulation. x =
                 bitcast_to_u32(x) if normal: x = x + ((x >> 16) & 1) + 0x7fff
                 if subnormal with nonzero low bits: x = x | 0x10000 result =
                 bitcast_to_bf16(x >> 16) *)
              let u32_dt = { src_dt with scalar = Dtype.Uint32 } in
              let u16_dt = { src_dt with scalar = Dtype.Uint16 } in
              let bool_dt = bool_like src_dt in
              let cu v =
                emit (Program.Const { value = Int v; dtype = u32_dt })
              in
              let ( &: ) lhs rhs =
                emit (Program.And { lhs; rhs; dtype = u32_dt })
              in
              let ( |: ) lhs rhs =
                emit (Program.Or { lhs; rhs; dtype = u32_dt })
              in
              let ( ^: ) lhs rhs =
                emit (Program.Xor { lhs; rhs; dtype = u32_dt })
              in
              let ( +: ) lhs rhs =
                emit (Program.Add { lhs; rhs; dtype = u32_dt })
              in
              let shr lhs rhs =
                emit (Program.Shr { lhs; rhs; dtype = u32_dt })
              in
              let ne lhs rhs =
                emit (Program.Cmpne { lhs; rhs; dtype = bool_dt })
              in
              (* Bitwise select: cond ? a : b via (mask & a) | (~mask & b) *)
              let select cond a b =
                let cond_u32 =
                  emit (Program.Cast { src = cond; dtype = u32_dt })
                in
                let mask =
                  emit (Program.Neg { src = cond_u32; dtype = u32_dt })
                in
                let inv = mask ^: cu (lnot 0) in
                a &: mask |: (b &: inv)
              in
              let src' = map_ref src in
              let x = emit (Program.Bitcast { src = src'; dtype = u32_dt }) in
              let lsb = shr x (cu 16) &: cu 1 in
              let rounded = x +: cu 0x7fff +: lsb in
              let neg = emit (Program.Neg { src = x; dtype = u32_dt }) in
              let nonzero = ne (neg &: cu 0x7f800000) (cu 0) in
              let has_nan = ne (x &: cu 0xffff) (cu 0) in
              let nanfix = select has_nan (x |: cu 0x10000) x in
              let adjusted = select nonzero rounded nanfix in
              let shifted = shr adjusted (cu 16) in
              let u16 = emit (Program.Cast { src = shifted; dtype = u16_dt }) in
              Some (emit (Program.Bitcast { src = u16; dtype }))
          | _ -> None)
      | _ -> None)
    program

let apply_rewrites ~(lang : lang) (program : Program.t) : Program.t =
  let program =
    List.fold_left
      (fun acc cfg -> rewrite_non_native_float cfg acc)
      program lang.rewrite.non_native
  in
  let program =
    if lang.rewrite.cast_f64_to_f16 then rewrite_cast_f64_to_f16 program
    else program
  in
  let program =
    if lang.rewrite.fp8_cast_via_f32 then rewrite_fp8_cast_variants program
    else program
  in
  if lang.rewrite.manual_bf16_cast then rewrite_manual_bf16_cast program
  else program

(* ───── Preamble Generators ───── *)

let preamble_clang (meta : render_meta) : string =
  let render_vector_def (dt : Dtype.t) =
    let scalar = clang_scalar_name dt.scalar in
    let vec = clang_type_map dt in
    let size = Dtype.itemsize dt in
    let alignment = 1 lsl int_of_float (log (float_of_int size) /. log 2.0) in
    Printf.sprintf
      "typedef %s %s __attribute__((aligned(%d),ext_vector_type(%d)));" scalar
      vec alignment dt.count
  in
  let vec_defs = List.map render_vector_def meta.vector_types in
  let amx_defs =
    if meta.wmma_configs = [] then []
    else
      let macros =
        [
          "#define AMX_SET(imm5) __asm(\"nop\\nnop\\nnop\\n.word \
           (0x201000+(%0<<5)+%1)\" : : \"i\"(17), \"i\"(imm5) : \"memory\")";
          "#define AMX(op, gpr, btf) __asm(\".word (0x201000+(%0 << \
           5)+0%1-((0%1>>4)*6))\" : : \"i\"(op), \"r\"((unsigned long \
           long)(gpr)+(btf)) : \"memory\")";
        ]
      in
      let funcs =
        List.map
          (fun cfg ->
            let { wmma_name; wmma_n; wmma_m; wmma_dtype_in; _ } = cfg in
            let out =
              clang_type_map
                { Dtype.scalar = wmma_dtype_in; count = wmma_n * wmma_n }
            in
            let dt1 =
              clang_type_map { Dtype.scalar = wmma_dtype_in; count = wmma_n }
            in
            let dt2 =
              clang_type_map { Dtype.scalar = wmma_dtype_in; count = wmma_m }
            in
            Printf.sprintf
              {|static %s __%s(%s data1, %s data2, %s data0){
  AMX_SET(0);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }
  AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull);
  for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }
  AMX_SET(1);
  return data0;
}|}
              out wmma_name dt1 dt2 out)
          meta.wmma_configs
      in
      macros @ funcs
  in
  let defs = vec_defs @ amx_defs in
  if defs = [] then "" else String.concat "\n" defs ^ "\n"

let preamble_opencl (meta : render_meta) : string =
  if meta.uses_half then "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
  else ""

let preamble_intel (meta : render_meta) : string =
  let lines = ref [] in
  let add s = lines := s :: !lines in

  if meta.uses_half then add "#pragma OPENCL EXTENSION cl_khr_fp16 : enable";

  List.iter
    (fun cfg ->
      let { wmma_name; wmma_dtype_in; wmma_dtype_out; _ } = cfg in
      let dt_in_name, dt_in_suffix =
        match wmma_dtype_in with
        | Dtype.Bfloat16 -> ("ushort", "bf16")
        | _ -> ("half", "f16")
      in
      let dt_out_name =
        match wmma_dtype_out with Dtype.Float16 -> "half" | _ -> "float"
      in
      add
        (Printf.sprintf
           {|%s8 __%s(%s16 a, %s16 b, %s8 c) {
    return intel_sub_group_%s_%s_matrix_mad_k16(as_int8(a), as_int8(b), c);
}|}
           dt_out_name wmma_name dt_in_name dt_in_name dt_out_name dt_in_suffix
           dt_in_suffix))
    meta.wmma_configs;

  if !lines = [] then "" else String.concat "\n" (List.rev !lines) ^ "\n"

let preamble_metal (meta : render_meta) : string =
  let header = "#include <metal_stdlib>\nusing namespace metal;\n" in
  (* Only generate WMMA functions on arm64 *)
  let wmma_funcs =
    if Lazy.force is_arm64 then
      List.map
        (fun cfg ->
          let { wmma_name; wmma_dtype_in; wmma_dtype_out; _ } = cfg in
          (* Metal uses vec(2) for WMMA - each thread holds 2 elements *)
          let in_type = metal_type_map { scalar = wmma_dtype_in; count = 2 } in
          let out_type =
            metal_type_map { scalar = wmma_dtype_out; count = 2 }
          in
          let simd_in = metal_type_map { scalar = wmma_dtype_in; count = 1 } in
          let simd_out =
            metal_type_map { scalar = wmma_dtype_out; count = 1 }
          in
          Printf.sprintf
            {|%s __%s(%s a, %s b, %s c) {
  simdgroup_%s8x8 mat_a, mat_b; simdgroup_%s8x8 mat_c;
  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);
  return %s(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);
}|}
            out_type wmma_name in_type in_type out_type simd_in simd_out
            out_type)
        meta.wmma_configs
    else []
  in
  if wmma_funcs = [] then header
  else header ^ String.concat "\n" wmma_funcs ^ "\n"

let preamble_cuda (meta : render_meta) : string =
  let base =
    [
      "#define INFINITY (__int_as_float(0x7f800000))";
      "#define NAN (__int_as_float(0x7fffffff))";
      "template <class T, class F> __device__ __forceinline__ T tg_bitcast(F \
       v) { union U { F f; T t; }; U u; u.f = v; return u.t; }";
    ]
  in
  let includes =
    (if meta.uses_fp8 then [ "#include <cuda_fp8.h>" ] else [])
    @ (if meta.uses_half then [ "#include <cuda_fp16.h>" ] else [])
    @ if meta.uses_bf16 then [ "#include <cuda_bf16.h>" ] else []
  in
  let vec_defs =
    List.filter_map
      (fun (dt : Dtype.t) ->
        let need_def =
          match dt.scalar with
          | Dtype.Float16 | Dtype.Bfloat16 -> List.mem dt.count [ 4; 8 ]
          | Dtype.Fp8e4m3 | Dtype.Fp8e5m2 -> List.mem dt.count [ 2; 4; 8; 16 ]
          | _ -> false
        in
        if need_def then
          let vec = cuda_type_map dt in
          let scal = cuda_type_map { dt with count = 1 } in
          let elems =
            String.concat ", " (List.init dt.count (fun j -> vec_elem_name j))
          in
          let header =
            String.concat ", "
              (List.init dt.count (fun j ->
                   Printf.sprintf "%s %s" scal (vec_elem_name j)))
          in
          Some
            (Printf.sprintf
               "struct __align__(%d) %s { %s %s; }; __device__ %s make_%s(%s) \
                { %s r={%s}; return r; }"
               (Dtype.itemsize dt) vec scal elems vec vec header vec elems)
        else None)
      meta.vector_types
  in
  let wmma_defs =
    let dt_map_in = function
      | Dtype.Float32 -> "tf32"
      | Dtype.Float16 -> "f16"
      | Dtype.Bfloat16 -> "bf16"
      | Dtype.Fp8e4m3 -> "e4m3"
      | Dtype.Fp8e5m2 -> "e5m2"
      | _ -> "f16"
    in
    let dt_map_out = function
      | Dtype.Float32 -> "f32"
      | Dtype.Float16 -> "f16"
      | _ -> "f32"
    in
    let render_wmma cfg =
      let {
        wmma_name;
        wmma_n;
        wmma_m;
        wmma_k;
        wmma_dtype_in;
        wmma_dtype_out;
        wmma_upcast_a;
        wmma_upcast_b;
        wmma_upcast_c;
        _;
      } =
        cfg
      in
      let scalar_itemsize scalar = Dtype.itemsize { Dtype.scalar; count = 1 } in
      let n_c = max 1 (wmma_upcast_c * scalar_itemsize wmma_dtype_out / 4) in
      let n_a = max 1 (wmma_upcast_a * scalar_itemsize wmma_dtype_in / 4) in
      let n_b = max 1 (wmma_upcast_b * scalar_itemsize wmma_dtype_in / 4) in
      let total = n_c + n_a + n_b in
      let operands = Array.init total (fun i -> Printf.sprintf "%%%d" i) in
      let slice start len =
        let segment = Array.sub operands start len in
        String.concat ", " (Array.to_list segment)
      in
      let c_ops = slice 0 n_c in
      let a_ops = slice n_c n_a in
      let b_ops = slice (n_c + n_a) n_b in
      let c_constraints =
        String.concat ", "
          (List.init n_c (fun i -> Printf.sprintf "\"+r\"(c_pk[%d])" i))
      in
      let a_constraints =
        String.concat ", "
          (List.init n_a (fun i -> Printf.sprintf "\"r\"(a_pk[%d])" i))
      in
      let b_constraints =
        String.concat ", "
          (List.init n_b (fun i -> Printf.sprintf "\"r\"(b_pk[%d])" i))
      in
      let upcast_type scalar count =
        cuda_type_map { Dtype.scalar; count = max 1 count }
      in
      let type_a = upcast_type wmma_dtype_in wmma_upcast_a in
      let type_b = upcast_type wmma_dtype_in wmma_upcast_b in
      let type_c = upcast_type wmma_dtype_out wmma_upcast_c in
      Printf.sprintf
        {|__device__ %s __%s(%s a, %s b, %s c) {
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m%dn%dk%d.row.col.%s.%s.%s.%s {%s}, {%s}, {%s}, {%s};"
      : %s
      : %s, %s);
  return c;
}|}
        type_c wmma_name type_a type_b type_c wmma_m wmma_n wmma_k
        (dt_map_out wmma_dtype_out)
        (dt_map_in wmma_dtype_in) (dt_map_in wmma_dtype_in)
        (dt_map_out wmma_dtype_out)
        c_ops a_ops b_ops c_ops c_constraints a_constraints b_constraints
    in
    List.map render_wmma meta.wmma_configs
  in
  String.concat "\n" (base @ includes @ vec_defs @ wmma_defs) ^ "\n"

let preamble_amd ~arch (meta : render_meta) : string =
  let lines = ref [] in
  let add s = lines := s :: !lines in
  let cdna = match arch with CDNA3 | CDNA4 -> true | RDNA3 | RDNA4 -> false in

  if meta.uses_infinity_nan then begin
    add "#define INFINITY (__builtin_inff())";
    add {|#define NAN (__builtin_nanf(""))|}
  end;

  if meta.uses_gid || meta.uses_lid || meta.uses_tid then begin
    add "typedef long unsigned int size_t;";
    List.iter
      (fun fname ->
        add
          (Printf.sprintf
             {|extern "C" __attribute__((device, const)) unsigned int %s(size_t);|}
             fname))
      [ "__ockl_get_local_id"; "__ockl_get_group_id"; "__ockl_get_local_size" ]
  end;

  List.iter
    (fun (op, (dt : Dtype.t)) ->
      let bits =
        match dt.scalar with
        | Dtype.Float16 -> 16
        | Dtype.Float64 -> 64
        | _ -> 32
      in
      let type_str =
        match dt.scalar with
        | Dtype.Float16 -> "_Float16"
        | Dtype.Float64 -> "double"
        | _ -> "float"
      in
      let attr =
        match op with "sqrt" -> "const" | "exp2" | "log2" -> "pure" | _ -> ""
      in
      let attr_str = if attr = "" then "" else Printf.sprintf ", %s" attr in
      add
        (Printf.sprintf
           {|extern "C" __attribute__((device%s)) %s __ocml_%s_f%d(%s);|}
           attr_str type_str op bits type_str))
    meta.ocml_funcs;

  if meta.uses_bf16 then add "typedef unsigned short hip_bfloat16;";
  if meta.uses_half then add "#define half _Float16";
  if meta.uses_fp8 then begin
    add "typedef unsigned char hip_bf8;";
    add "typedef unsigned char hip_fp8;";
    add
      {|static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {
  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(v,is_bf8?57344.0f:448.0f,is_bf8?-57344.0f:-448.0f) : v;
  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false):__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));
}|};
    add
      {|static inline __attribute__((device)) float fp8_to_f32(unsigned char v, int is_bf8) {
  return is_bf8 ? __builtin_amdgcn_cvt_f32_bf8((unsigned int)v, 0) : __builtin_amdgcn_cvt_f32_fp8((unsigned int)v, 0);
}|}
  end;

  List.iter
    (fun (dt : Dtype.t) ->
      let vec = amd_type_map dt in
      let scal = amd_type_map { dt with count = 1 } in
      let elems =
        String.concat ", " (List.init dt.count (fun j -> vec_elem_name j))
      in
      let header =
        String.concat ", "
          (List.init dt.count (fun j ->
               Printf.sprintf "%s %s" scal (vec_elem_name j)))
      in
      add
        (Printf.sprintf
           "typedef %s %s __attribute__((ext_vector_type(%d)));\n\
            static inline __attribute__((device)) %s make_%s(%s) { return { %s \
            }; }"
           scal vec dt.count vec vec header elems))
    meta.vector_types;

  List.iter
    (fun cfg ->
      let {
        wmma_name;
        wmma_n;
        wmma_m;
        wmma_k;
        wmma_dtype_in;
        wmma_dtype_out;
        _;
      } =
        cfg
      in
      let type_in = amd_wmma_type_map wmma_dtype_in in
      let type_out = amd_wmma_out_type_map wmma_dtype_out in
      if cdna then begin
        let type_suffix =
          match (wmma_n, wmma_m, wmma_k) with
          | 16, 16, 16 when wmma_dtype_in = Dtype.Bfloat16 -> "bf16_1k"
          | 16, 16, 32 -> Printf.sprintf "_%s" type_in
          | 16, 16, 128 -> "_f8f6f4"
          | _ -> type_in
        in
        let scale_prefix = if wmma_k = 128 then "scale_" else "" in
        let builtin =
          Printf.sprintf "__builtin_amdgcn_mfma_%s%s_%dx%dx%d%s" scale_prefix
            type_out wmma_n wmma_m wmma_k type_suffix
        in
        if wmma_k = 128 then
          let fp8_idx =
            match wmma_dtype_in with Dtype.Fp8e5m2 -> 1 | _ -> 0
          in
          add
            (Printf.sprintf
               {|#define __%s(a, b, c) %s(a, b, c, %d, %d, 0, 0, 0, 0)|}
               wmma_name builtin fp8_idx fp8_idx)
        else
          add
            (Printf.sprintf {|#define __%s(a, b, c) %s(a, b, c, 0, 0, 0)|}
               wmma_name builtin)
      end
      else begin
        let rdna4 = arch = RDNA4 in
        let suffix = if rdna4 then "_w32_gfx12" else "_w32" in
        if rdna4 || wmma_dtype_out = Dtype.Float32 then
          add
            (Printf.sprintf
               "#define __%s __builtin_amdgcn_wmma_%s_16x16x16_%s%s" wmma_name
               type_out type_in suffix)
        else
          (* RDNA3 half-precision output requires wrapper function *)
          add
            (Printf.sprintf
               {|static inline __attribute__((device)) half8 __%s(half16 a, half16 b, half8 c) {
  half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;
}|}
               wmma_name)
      end)
    meta.wmma_configs;

  if !lines = [] then "" else String.concat "\n" (List.rev !lines) ^ "\n"

(* ───── Custom Format String Parser ───── *)

let render_custom_braces ~(r : int -> string) fmt args =
  let arg_strs = Array.of_list (List.map r args) in
  let nargs = Array.length arg_strs in
  let next_auto = ref 0 in
  let get_arg idx =
    if idx < 0 || idx >= nargs then
      failwith
        (Printf.sprintf "render_custom: placeholder index %d out of bounds" idx)
    else arg_strs.(idx)
  in
  let len = String.length fmt in
  let buf = Buffer.create len in
  let i = ref 0 in
  while !i < len do
    match fmt.[!i] with
    | '{' ->
        if !i + 1 < len && fmt.[!i + 1] = '{' then begin
          Buffer.add_char buf '{';
          i := !i + 2
        end
        else begin
          let j = ref (!i + 1) in
          while !j < len && fmt.[!j] <> '}' do
            incr j
          done;
          if !j >= len then failwith err_custom_unmatched_open;
          let field = String.sub fmt (!i + 1) (!j - !i - 1) |> String.trim in
          let idx =
            if String.equal field "" then begin
              let k = !next_auto in
              incr next_auto;
              k
            end
            else
              let field_name =
                match String.split_on_char ':' field with
                | hd :: _ -> String.trim hd
                | [] -> field
              in
              match int_of_string_opt field_name with
              | Some k -> k
              | None ->
                  failwith
                    (Printf.sprintf
                       "render_custom: unsupported placeholder {%s}" field)
          in
          Buffer.add_string buf (get_arg idx);
          i := !j + 1
        end
    | '}' ->
        if !i + 1 < len && fmt.[!i + 1] = '}' then begin
          Buffer.add_char buf '}';
          i := !i + 2
        end
        else failwith err_custom_unmatched_close
    | ch ->
        Buffer.add_char buf ch;
        incr i
  done;
  Buffer.contents buf

(* ───── Core Render Function ───── *)

type render_ctx = {
  program : Program.t;
  names : string array;
  use_counts : int array;
  counters : (string, int) Hashtbl.t;
  lines : string list ref;
  depth : int ref;
  params : (int * string * Dtype.ptr) list ref;
  scalar_params : (string * Dtype.t) list ref;
  lang : lang;
}

let ctx_r ctx idx = ctx.names.(idx)

let ctx_next_name ctx prefix =
  let c = Hashtbl.find_opt ctx.counters prefix |> Option.value ~default:0 in
  Hashtbl.replace ctx.counters prefix (c + 1);
  Printf.sprintf "%s%d" prefix c

let ctx_emit ctx s =
  ctx.lines := (String.make (!(ctx.depth) * 2) ' ' ^ s) :: !(ctx.lines)

let strip_parens s =
  let len = String.length s in
  if len >= 2 && s.[0] = '(' && s.[len - 1] = ')' then String.sub s 1 (len - 2)
  else s

let is_associative_binop a b =
  match (a, b) with
  | Program.Add _, Program.Add _
  | Mul _, Mul _
  | And _, And _
  | Or _, Or _
  | Xor _, Xor _ ->
      true
  | _ -> false

(* Strip redundant parens for associative ops (a + (b + c) -> a + b + c). *)
let ctx_r_for ctx instr idx =
  let expr = ctx_r ctx idx in
  match instr with
  | Program.Add _ | Mul _ | And _ | Or _ | Xor _ -> (
      match ctx.program.(idx) with
      | op when is_associative_binop instr op -> strip_parens expr
      | _ -> expr)
  | _ -> expr

let prefix_of = function
  | Program.Wmma _ -> "wmma"
  | Define_local _ -> "temp"
  | Define_reg _ -> "acc"
  | Define_var _ -> "var"
  | Const _ -> "const"
  | Cast _ | Bitcast _ | Vectorize _ -> "cast"
  | Custom_inline _ | Custom _ -> "custom"
  | Gep _ -> "gep"
  | Index _ -> "bidx"
  | Load _ -> "val"
  | _ -> "alu"

let axis_prefix = function
  | Ir.Global -> "g"
  | Ir.Thread -> "t"
  | Ir.Local -> "l"
  | Ir.Warp -> "w"
  | Ir.Loop -> "L"
  | Ir.Group_reduce -> "G"
  | Ir.Reduce -> "R"
  | Ir.Upcast -> "u"
  | Ir.Unroll -> "r"
  | Ir.Outer -> "O"
  | Ir.Placeholder -> "P"

let render_expr ctx instr =
  let r = ctx_r ctx in
  let r_for = ctx_r_for ctx in
  let dtype_of_ref ref_idx = Program.dtype_of ctx.program.(ref_idx) in
  let render_cast_expr src dtype =
    let src_dt = Option.value ~default:dtype (dtype_of_ref src) in
    ctx.lang.render_cast ~src:src_dt ~dst:dtype (r src)
  in
  match ctx.lang.code_for_op instr (r_for instr) with
  | Some expr ->
      let dtype =
        match Program.dtype_of instr with
        | Some dt -> dt
        | None -> invalid_arg err_code_for_op_void
      in
      Some (expr, dtype)
  | None -> (
      match instr with
      | Program.Const { value; dtype } ->
          Some (ctx.lang.render_const value dtype, dtype)
      | Cast { src; dtype } ->
          Some (Printf.sprintf "(%s)" (render_cast_expr src dtype), dtype)
      | Bitcast { src; dtype } ->
          let src_dt =
            match dtype_of_ref src with
            | Some dt -> dt
            | None -> invalid_arg err_bitcast_no_dtype
          in
          Some (ctx.lang.render_bitcast ~src:src_dt ~dst:dtype (r src), dtype)
      | Vectorize { srcs; dtype } ->
          Some (ctx.lang.render_vector dtype (List.map r srcs), dtype)
      | Gep { src; idx; dtype } ->
          let src_dt =
            match Program.dtype_of ctx.program.(src) with
            | Some dt -> dt
            | None -> invalid_arg err_gep_no_dtype
          in
          let expr =
            if src_dt.count > ctx.lang.gep_arr_threshold then
              Printf.sprintf "%s[%d]" (r src) idx
            else Printf.sprintf "%s.%c" (r src) gep_swizzle.[idx]
          in
          Some (expr, dtype)
      | Index { ptr; idxs; dtype; _ } ->
          let idx_expr =
            match idxs with
            | [] -> "0"
            | [ idx ] -> r idx
            | _ -> String.concat "+" (List.map r idxs)
          in
          Some (Printf.sprintf "(%s+%s)" (r ptr) idx_expr, dtype.base)
      | Custom_inline { fmt; args; dtype } ->
          Some (render_custom_braces ~r fmt args, dtype)
      | Load { src; alt = None; dtype } -> (
          match ctx.program.(src) with
          | Index { dtype = idx_dt; _ } when idx_dt.addrspace = Dtype.Reg ->
              Some (Printf.sprintf "(*%s)" (r src), dtype)
          | _ -> None)
      | _ -> None)

let rec index_ref program idx =
  match program.(idx) with
  | Program.Index _ -> Some idx
  | Program.Cast { src; _ } | Program.Bitcast { src; _ } ->
      index_ref program src
  | _ -> None

(* Inlining policy: constants, indices, GEPs, and custom_inline are always
   inlined. Single-use ALU ops (except Where) and casts are inlined. Register
   loads are inlined unconditionally. *)
let should_inline ctx i instr =
  let count = ctx.use_counts.(i) in
  match instr with
  | Program.Const _ | Index _ | Gep _ | Custom_inline _ -> true
  | Load { src; alt = None; _ } -> (
      match ctx.program.(src) with
      | Index { dtype; _ } when dtype.addrspace = Dtype.Reg -> true
      | _ -> false)
  | _ when Program.is_alu instr -> (
      match instr with Program.Where _ -> false | _ -> count <= 1)
  | Cast { dtype; _ } -> dtype.count = 1 && count <= 1
  | Bitcast _ | Vectorize _ -> count <= 1
  | _ -> false

let bind_expr ctx i instr ~prefix ~dtype expr =
  if should_inline ctx i instr then ctx.names.(i) <- expr
  else begin
    let var = ctx_next_name ctx prefix in
    ctx.names.(i) <- var;
    ctx_emit ctx
      (Printf.sprintf "%s %s = %s;" (ctx.lang.type_map dtype) var expr)
  end

let render_stmt ctx i instr =
  let r = ctx_r ctx in
  let emit = ctx_emit ctx in
  let next_name = ctx_next_name ctx in
  match instr with
  | Program.Param { idx; dtype } ->
      let var_name =
        if dtype.size > 0 then Printf.sprintf "data%d_%d" idx dtype.size
        else Printf.sprintf "data%d" idx
      in
      ctx.names.(i) <- var_name;
      ctx.params := (idx, var_name, dtype) :: !(ctx.params);
      true
  | Define_var { name; dtype; _ } ->
      ctx.names.(i) <- name;
      ctx.scalar_params := (name, dtype) :: !(ctx.scalar_params);
      true
  | Define_local { size; dtype } ->
      let var = next_name "temp" in
      ctx.names.(i) <- var;
      emit
        (Printf.sprintf "%s%s%s %s[%d];" ctx.lang.smem_align
           ctx.lang.smem_prefix
           (ctx.lang.type_map dtype.base)
           var size);
      true
  | Define_reg { size; dtype } ->
      let var = next_name "acc" in
      ctx.names.(i) <- var;
      emit (Printf.sprintf "%s %s[%d];" (ctx.lang.type_map dtype.base) var size);
      true
  | Load { src; alt; dtype } ->
      let var = next_name "val" in
      ctx.names.(i) <- var;
      (match alt with
      | None ->
          emit
            (Printf.sprintf "%s %s = (*%s);" (ctx.lang.type_map dtype) var
               (r src))
      | Some alt_ref -> (
          match index_ref ctx.program src with
          | Some idx_ref -> (
              match ctx.program.(idx_ref) with
              | Program.Index { gate = Some gate; _ } ->
                  emit
                    (Printf.sprintf "%s %s = (%s?*%s:%s);"
                       (ctx.lang.type_map dtype) var (r gate) (r src)
                       (r alt_ref))
              | _ -> invalid_arg err_load_alt_gate)
          | None -> invalid_arg err_load_alt_gate));
      true
  (* Non-binding instructions (Store, End_range, If, Endif, Barrier) consume an
     "alu" name slot to keep variable indices dense. *)
  | Store { dst; value } ->
      ignore (next_name "alu");
      emit (Printf.sprintf "*%s = %s;" (r dst) (r value));
      true
  | Range { size; dtype; axis; kind } ->
      let var = Printf.sprintf "%sidx%d" (axis_prefix kind) axis in
      ctx.names.(i) <- var;
      emit
        (Printf.sprintf "for (%s %s = 0; %s < %s; %s++) {"
           (ctx.lang.type_map dtype) var var (r size) var);
      incr ctx.depth;
      true
  | End_range _ | Endif _ ->
      ignore (next_name "alu");
      decr ctx.depth;
      emit "}";
      true
  | If { cond; _ } ->
      ignore (next_name "alu");
      emit (Printf.sprintf "if (%s) {" (r cond));
      incr ctx.depth;
      true
  | Barrier ->
      ignore (next_name "alu");
      emit ctx.lang.barrier;
      true
  | Special { dim; size; dtype } ->
      let var =
        match dim with
        | Ir.Group_id axis -> Printf.sprintf "gidx%d" axis
        | Ir.Local_id axis -> Printf.sprintf "lidx%d" axis
        | Ir.Global_idx axis -> Printf.sprintf "idx%d" axis
      in
      let expr = ctx.lang.code_for_workitem dim in
      ctx.names.(i) <- var;
      emit
        (Printf.sprintf "%s %s = %s; /* %s */" (ctx.lang.type_map dtype) var
           expr (r size));
      true
  | Wmma { name = wmma_name; a; b; c; dtype; _ } ->
      let var = next_name "wmma" in
      ctx.names.(i) <- var;
      emit
        (Printf.sprintf "%s %s = __%s(%s, %s, %s);" (ctx.lang.type_map dtype)
           var wmma_name (r a) (r b) (r c));
      true
  | Custom { fmt; args } ->
      ctx.names.(i) <- next_name "custom";
      emit (render_custom_braces ~r fmt args);
      true
  | _ -> false

let assemble_kernel ~(lang : lang) ~(preamble_gen : render_meta -> string) ~name
    ~meta ctx =
  let params =
    List.sort
      (fun (i, _, _) (j, _, _) -> Int.compare i j)
      (List.rev !(ctx.params))
  in
  let param_strs =
    List.map
      (fun (param_idx, var, (dtype : Dtype.ptr)) ->
        let base =
          Printf.sprintf "%s%s*%s %s" lang.buffer_prefix
            (lang.type_map dtype.base) lang.buffer_suffix var
        in
        match lang.param_annot param_idx with
        | None -> base
        | Some annot -> Printf.sprintf "%s %s" base annot)
      params
  in
  let scalar_param_strs =
    List.rev_map
      (fun (name, dtype) ->
        match dtype.Dtype.scalar with
        | Dtype.Int32 -> Printf.sprintf "%s %s" lang.arg_int_prefix name
        | _ -> Printf.sprintf "%s %s" (lang.type_map dtype) name)
      !(ctx.scalar_params)
  in
  let extra_params = lang.extra_params meta in
  (* CPU JIT uses a fixed ABI (bufs + vals) to avoid a libffi dependency. The
     kernel is generated with individual params, and a wrapper with the fixed
     ABI signature is appended. *)
  let fixed_abi = lang.fixed_abi in
  let all_params = param_strs @ scalar_param_strs @ extra_params in
  let kernel_prefix = lang.kernel_prefix meta in
  let preamble = preamble_gen meta in
  let inner_name =
    if fixed_abi && String.equal name "kern" then "kern_" else name
  in
  let signature =
    if fixed_abi then
      Printf.sprintf "static %s %s(%s)" kernel_prefix inner_name
        (String.concat ", " all_params)
    else
      Printf.sprintf "%s %s(%s)" kernel_prefix name
        (String.concat ", " all_params)
  in
  let body = String.concat "\n" (List.rev !(ctx.lines)) in
  let kernel_src = Printf.sprintf "%s%s {\n%s\n}" preamble signature body in
  if fixed_abi then
    let cast_args =
      List.map
        (fun (param_idx, _var, (dtype : Dtype.ptr)) ->
          Printf.sprintf "(%s*)bufs[%d]" (lang.type_map dtype.base) param_idx)
        params
    in
    let scalar_args =
      List.mapi
        (fun i (_name, dtype) ->
          Printf.sprintf "(%s)vals[%d]" (lang.type_map dtype) i)
        (List.rev !(ctx.scalar_params))
    in
    let all_args = cast_args @ scalar_args in
    Printf.sprintf
      "%s\n\
       void kern(const unsigned long long *bufs, const long long *vals) {\n\
      \  %s(%s);\n\
       }"
      kernel_src inner_name
      (String.concat ", " all_args)
  else kernel_src

let render_kernel ~(lang : lang) ~(preamble_gen : render_meta -> string)
    ?(name = "kernel") (program : Program.t) : string =
  let program = apply_rewrites ~lang program in
  let n = Array.length program in
  let ctx =
    {
      program;
      names = Array.make n "";
      use_counts = compute_use_counts program;
      counters = Hashtbl.create 16;
      lines = ref [];
      depth = ref 1;
      params = ref [];
      scalar_params = ref [];
      lang;
    }
  in
  let meta = collect_metadata program in
  Array.iteri
    (fun i instr ->
      match render_expr ctx instr with
      | Some (expr, dtype) ->
          bind_expr ctx i instr ~prefix:(prefix_of instr) ~dtype expr
      | None ->
          if not (render_stmt ctx i instr) then invalid_arg err_unhandled_alu)
    program;
  assemble_kernel ~lang ~preamble_gen ~name ~meta ctx

(* ───── Exported Renderers ───── *)

let clang_load_store_widths (dtype : Dtype.t) =
  if Dtype.is_float dtype then
    if clang_has_amx then [ 16; 8; 4; 2; 1 ] else [ 4; 2; 1 ]
  else [ 1 ]

let gpu_load_store_widths (dtype : Dtype.t) =
  if Dtype.is_float dtype then [ 4; 2; 1 ] else [ 1 ]

(* ALU operations rendered natively by GPU C-style backends. *)
let cstyle_code_for_op =
  [
    Renderer.Sqrt;
    Recip;
    Neg;
    Exp2;
    Log2;
    Sin;
    Trunc;
    And;
    Xor;
    Or;
    Add;
    Sub;
    Mul;
    Mod;
    Idiv;
    Cmpne;
    Shr;
    Shl;
    Cmplt;
    Where;
    Cmpeq;
  ]

(* Clang natively handles fdiv but not reciprocal/exp2/log2/sin (these are
   lowered by the decomposition pass). *)
let clang_code_for_op_keys =
  [
    Renderer.Sqrt;
    Neg;
    Trunc;
    And;
    Xor;
    Or;
    Add;
    Sub;
    Mul;
    Mod;
    Idiv;
    Cmpne;
    Shr;
    Shl;
    Cmplt;
    Where;
    Cmpeq;
    Fdiv;
  ]

let clang =
  Renderer.make ~name:"clang" ~device:"CPU" ~has_local:false ~has_shared:false
    ~has_threads:(threads_enabled ())
    ~global_max:(Some [ Lazy.force cpu_count; 0; 0 ])
    ~shared_max:0 ~tensor_cores:clang_tensor_cores
    ~load_store_widths:clang_load_store_widths
    ~code_for_op:clang_code_for_op_keys
    ~render:(render_kernel ~lang:clang_lang ~preamble_gen:preamble_clang)
    ()

let clang_no_abi =
  let lang = { clang_lang with fixed_abi = false } in
  Renderer.make ~name:"clang" ~device:"CPU" ~has_local:false ~has_shared:false
    ~has_threads:(threads_enabled ())
    ~global_max:(Some [ Lazy.force cpu_count; 0; 0 ])
    ~shared_max:0 ~tensor_cores:clang_tensor_cores
    ~load_store_widths:clang_load_store_widths
    ~code_for_op:clang_code_for_op_keys
    ~render:(render_kernel ~lang ~preamble_gen:preamble_clang)
    ()

(* shared_max values are compile-time scheduling hints, not actual device
   limits. OpenCL: 32KB (CL_DEVICE_LOCAL_MEM_SIZE minimum), CUDA: 48KB
   (per-block default), AMD: 64KB (LDS size on GCN/CDNA/RDNA). *)
let opencl =
  Renderer.make ~name:"opencl" ~device:"CL" ~has_local:true ~has_shared:true
    ~tensor_cores:[] ~shared_max:32768 ~load_store_widths:gpu_load_store_widths
    ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:opencl_lang ~preamble_gen:preamble_opencl)
    ()

let intel =
  Renderer.make ~name:"intel" ~device:"CL" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~tensor_cores:intel_tc
    ~load_store_widths:gpu_load_store_widths ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:intel_lang ~preamble_gen:preamble_intel)
    ()

let qcom =
  Renderer.make ~name:"qcom" ~device:"QCOM" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~tensor_cores:[] ~load_store_widths:gpu_load_store_widths
    ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:opencl_lang ~preamble_gen:preamble_opencl)
    ()

let metal =
  Renderer.make ~name:"metal" ~device:"METAL" ~has_local:true ~has_shared:true
    ~shared_max:32768
    ~tensor_cores:(if Lazy.force is_arm64 then metal_tc else [])
    ~load_store_widths:gpu_load_store_widths ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:metal_lang ~preamble_gen:preamble_metal)
    ()

(* Grid limits from the CUDA programming guide: gridDim max (2^31-1, 65535,
   65535), blockDim max (1024, 1024, 64). *)
let cuda ?arch () =
  let tensor_cores =
    match arch with
    | Some a -> cuda_tensor_cores a
    | None -> (
        match default_cuda_arch () with
        | Some a -> cuda_tensor_cores a
        | None -> [])
  in
  Renderer.make ~name:"cuda" ~device:"CUDA" ~has_local:true ~has_shared:true
    ~shared_max:49152 ~tensor_cores
    ~global_max:(Some [ 2147483647; 65535; 65535 ])
    ~local_max:(Some [ 1024; 1024; 64 ])
    ~load_store_widths:gpu_load_store_widths ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:cuda_lang ~preamble_gen:preamble_cuda)
    ()

let amd ?(arch = RDNA3) () =
  Renderer.make ~name:"amd" ~device:"AMD" ~has_local:true ~has_shared:true
    ~shared_max:65536 ~tensor_cores:(amd_tensor_cores arch)
    ~global_max:(Some [ 2147483647; 65535; 65535 ])
    ~load_store_widths:gpu_load_store_widths ~code_for_op:cstyle_code_for_op
    ~render:(render_kernel ~lang:amd_lang ~preamble_gen:(preamble_amd ~arch))
    ()
