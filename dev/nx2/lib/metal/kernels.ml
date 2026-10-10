(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* kernels.h as the host plans with it: the metallib's kernels, the dense
   tiles' geometry, the other kernels' threads, the bits of [order] and each
   parameter struct's fields by byte offset. test/metal/test_metal_kernels.ml
   checks every fact against kernels.h. *)

(* Kernels *)

(* An instance's arguments, as its name in NX_METAL_KERNELS states them: the
   operands' dtype, and whether a is stored [k][m] and b [n][k]. *)
type dtype = F32 | F16 | Bf16
type order = { a_t : bool; b_t : bool }

type instance =
  | Large of dtype * order
  | Small of dtype
  | Wide of dtype * order
  | Checked of dtype
  | Int8 of order
  | Skinny of dtype * bool (* b stored [n][k] *)
  | Combine
  | Int

let nn = { a_t = false; b_t = false }
let nt = { a_t = false; b_t = true }
let tn = { a_t = true; b_t = false }
let tt = { a_t = true; b_t = true }

(* NX_METAL_KERNELS's rows in order: each kernel's name and instance. *)
let kernels =
  [|
    ("contract_f32_nn", Large (F32, nn));
    ("contract_f32_nt", Large (F32, nt));
    ("contract_f32_tn", Large (F32, tn));
    ("contract_f32_tt", Large (F32, tt));
    ("contract_f16_nn", Large (F16, nn));
    ("contract_f16_nt", Large (F16, nt));
    ("contract_f16_tn", Large (F16, tn));
    ("contract_f16_tt", Large (F16, tt));
    ("contract_bf16_nn", Large (Bf16, nn));
    ("contract_bf16_nt", Large (Bf16, nt));
    ("contract_bf16_tn", Large (Bf16, tn));
    ("contract_bf16_tt", Large (Bf16, tt));
    ("contract_f32_s", Small F32);
    ("contract_f16_s", Small F16);
    ("contract_bf16_s", Small Bf16);
    ("contract_f32_wnn", Wide (F32, nn));
    ("contract_f32_wnt", Wide (F32, nt));
    ("contract_f32_wtn", Wide (F32, tn));
    ("contract_f32_wtt", Wide (F32, tt));
    ("contract_f16_wnn", Wide (F16, nn));
    ("contract_f16_wnt", Wide (F16, nt));
    ("contract_f16_wtn", Wide (F16, tn));
    ("contract_f16_wtt", Wide (F16, tt));
    ("contract_bf16_wnn", Wide (Bf16, nn));
    ("contract_bf16_wnt", Wide (Bf16, nt));
    ("contract_bf16_wtn", Wide (Bf16, tn));
    ("contract_bf16_wtt", Wide (Bf16, tt));
    ("contract_f32_l", Checked F32);
    ("contract_bf16_l", Checked Bf16);
    ("contract_i8_nn", Int8 nn);
    ("contract_i8_nt", Int8 nt);
    ("contract_i8_tn", Int8 tn);
    ("contract_i8_tt", Int8 tt);
    ("skinny_f32_n", Skinny (F32, false));
    ("skinny_f32_t", Skinny (F32, true));
    ("skinny_f16_n", Skinny (F16, false));
    ("skinny_f16_t", Skinny (F16, true));
    ("skinny_bf16_n", Skinny (Bf16, false));
    ("skinny_bf16_t", Skinny (Bf16, true));
    ("contract_combine", Combine);
    ("contract_int", Int);
  |]

(* Constants *)

(* NX_METAL_THREADS: a dense or skinny threadgroup's threads. *)
let threads = 128

(* The dense tiles, rows by columns: NX_METAL_LARGE, NX_METAL_SMALL, and
   NX_METAL_WIDE_M by NX_METAL_WIDE_N. *)
let large = 64
let small = 32
let wide_m = 16
let wide_n = 64

(* The steps of k a dense tile stages: NX_METAL_BK_HALF, NX_METAL_BK and
   NX_METAL_BK_WIDE. *)
let bk_half = 32
let bk = 16
let bk_wide = 32

(* The columns of out a skinny threadgroup computes, b stored [n][k]
   (NX_METAL_SKINNY_T) or [k][n] (NX_METAL_SKINNY_N). *)
let skinny_t = 4
let skinny_n = 32

(* The SIMD integer kernel's tile side and threads: NX_METAL_INT_TILE and
   NX_METAL_INT_THREADS. *)
let int_tile = 64
let int_threads = 256

(* nx_metal_contract's bits of [order]: NX_METAL_A_T and NX_METAL_B_T. *)
let a_t = 1
let b_t = 2

(* NX_DTYPE_COUNT: the [init_dtype] of a call with no init. *)
let no_init = 21

(* Parameters *)

(* Each struct's size and its fields' byte offsets. *)

module Contract_params = struct
  let size = 120
  let a = 0
  let b = 8
  let init = 16
  let out = 24
  let a_batch = 32
  let b_batch = 40
  let init_batch = 48
  let a_m = 56
  let a_k = 60
  let b_k = 64
  let b_n = 68
  let init_m = 72
  let init_n = 76
  let batch = 80
  let m = 84
  let n = 88
  let k = 92
  let init_dtype = 96
  let out_dtype = 100
  let swizzle = 104
  let dtype = 108
  let acc = 112
  let order = 116
end

module Combine_params = struct
  let size = 64
  let out = 0
  let parts = 8
  let init = 16
  let init_batch = 24
  let init_m = 32
  let init_n = 36
  let batch = 40
  let m = 44
  let n = 48
  let split = 52
  let init_dtype = 56
  let out_dtype = 60
end
