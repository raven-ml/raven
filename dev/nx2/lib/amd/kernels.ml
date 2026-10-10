(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* kernels.h as the host plans with it: the code object's kernels, the WMMA
   tiles, the bits of [aligned], the work-items of the kernels that are not
   WMMA's and each parameter struct's fields by byte offset.
   test/amd/test_amd_kernels.ml checks every fact against kernels.h. *)

(* Kernels *)

(* An instance's arguments, as NX_AMD_KERNELS names them. *)
type kind = Bf16 | F16 | S8
type axis = K | M | N
type acc = F32 | F64 | I64
type form = Column | Across
type tile = T128x128 | T64x64 | T16x64

type instance =
  | Zero
  | Pack
  | Wmma of kind * axis * axis * tile
  | Simt of acc * int
  | Skinny of acc * form

(* NX_AMD_KERNELS's rows in order: each kernel's name and instance. *)
let kernels =
  [|
    ("zero_u32", Zero);
    ("pack", Pack);
    ("contract_bf16_kk_t128x128", Wmma (Bf16, K, K, T128x128));
    ("contract_bf16_kn_t128x128", Wmma (Bf16, K, N, T128x128));
    ("contract_bf16_mk_t128x128", Wmma (Bf16, M, K, T128x128));
    ("contract_bf16_mn_t128x128", Wmma (Bf16, M, N, T128x128));
    ("contract_bf16_kk_t64x64", Wmma (Bf16, K, K, T64x64));
    ("contract_bf16_kk_t16x64", Wmma (Bf16, K, K, T16x64));
    ("contract_f16_kk_t128x128", Wmma (F16, K, K, T128x128));
    ("contract_f16_kn_t128x128", Wmma (F16, K, N, T128x128));
    ("contract_f16_mk_t128x128", Wmma (F16, M, K, T128x128));
    ("contract_f16_mn_t128x128", Wmma (F16, M, N, T128x128));
    ("contract_f16_kk_t64x64", Wmma (F16, K, K, T64x64));
    ("contract_s8_kk_t128x128", Wmma (S8, K, K, T128x128));
    ("contract_simt_f32_128", Simt (F32, 128));
    ("contract_simt_f32_64", Simt (F32, 64));
    ("contract_skinny_f32", Skinny (F32, Column));
    ("contract_skinny_across_f32", Skinny (F32, Across));
    ("contract_simt_f64_64", Simt (F64, 64));
    ("contract_simt_i64_64", Simt (I64, 64));
    ("contract_skinny_f64", Skinny (F64, Column));
    ("contract_skinny_i64", Skinny (I64, Column));
    ("contract_skinny_across_f64", Skinny (F64, Across));
    ("contract_skinny_across_i64", Skinny (I64, Across));
  |]

(* Tiles *)

(* NX_AMD_TILES's columns. *)
type shape = { bm : int; bn : int; bkb : int; wm : int; wn : int }

(* NX_AMD_TILES's rows in order. *)
let tiles = [| T128x128; T64x64; T16x64 |]

let shape = function
  | T128x128 -> { bm = 128; bn = 128; bkb = 64; wm = 64; wn = 32 }
  | T64x64 -> { bm = 64; bn = 64; bkb = 64; wm = 32; wn = 32 }
  | T16x64 -> { bm = 16; bn = 64; bkb = 128; wm = 16; wn = 16 }

(* Constants *)

(* contract_params' bits of [aligned]. *)
let a_vectors = 1
let b_vectors = 2
let y_whole = 4

(* NX_CONTRACT_THREADS: the work-items of a workgroup of pack, zero_u32 and the
   SIMT and skinny kernels. *)
let threads = 256

(* Parameters *)

(* Each struct's size and its fields' byte offsets. *)

module Contract_params = struct
  let size = 192
  let a = 0
  let b = 8
  let init = 16
  let y = 24
  let partials = 32
  let tickets = 40
  let sa = 48
  let sb = 72
  let si = 96
  let sy = 120
  let batch = 144
  let m = 148
  let n = 152
  let k = 156
  let splits = 160
  let a_dtype = 164
  let b_dtype = 168
  let init_dtype = 172
  let y_dtype = 176
  let acc_dtype = 180
  let aligned = 184
  let zero = 188
end

module Pack_params = struct
  let size = 72
  let src = 0
  let dst = 8
  let s = 16
  let lead = 40
  let batch = 48
  let rows = 52
  let k = 56
  let dtype = 60
  let out = 64
  let bytes = 68
end

module Zero_params = struct
  let size = 16
  let p = 0
  let n = 8
end
