(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* kernels.h as the host plans with it: the cubin's kernels, the mma tiles, the
   bits of [aligned], the skinny block's rows and each parameter struct's
   fields by byte offset. test/cuda/test_cuda_kernels.ml checks every fact
   against kernels.h. *)

(* Kernels *)

(* An instance's arguments, as NX_CUDA_KERNELS names them. *)
type kind = Bf16 | F16 | S8 | Any
type axis = K | M | N
type acc = F32 | F64 | I64
type tile = T128x128 | T128x256 | T64x64 | T16x64

type instance =
  | Pack
  | Mma of kind * axis * axis * tile
  | Simt of acc * int
  | Skinny of acc

(* NX_CUDA_KERNELS's rows in order: each kernel's name and instance. *)
let kernels =
  [|
    ("pack", Pack);
    ("contract_bf16_kk_t128x256", Mma (Bf16, K, K, T128x256));
    ("contract_bf16_kn_t128x256", Mma (Bf16, K, N, T128x256));
    ("contract_bf16_mk_t128x256", Mma (Bf16, M, K, T128x256));
    ("contract_bf16_mn_t128x256", Mma (Bf16, M, N, T128x256));
    ("contract_bf16_kk_t128x128", Mma (Bf16, K, K, T128x128));
    ("contract_bf16_kk_t64x64", Mma (Bf16, K, K, T64x64));
    ("contract_f16_kk_t128x256", Mma (F16, K, K, T128x256));
    ("contract_f16_kn_t128x256", Mma (F16, K, N, T128x256));
    ("contract_f16_mk_t128x256", Mma (F16, M, K, T128x256));
    ("contract_f16_mn_t128x256", Mma (F16, M, N, T128x256));
    ("contract_f16_kk_t128x128", Mma (F16, K, K, T128x128));
    ("contract_f16_kk_t64x64", Mma (F16, K, K, T64x64));
    ("contract_any_kk_t16x64", Mma (Any, K, K, T16x64));
    ("contract_s8_kk_t128x256", Mma (S8, K, K, T128x256));
    ("contract_simt_f32_128", Simt (F32, 128));
    ("contract_simt_f32_64", Simt (F32, 64));
    ("contract_skinny_f32", Skinny F32);
    ("contract_simt_f64_64", Simt (F64, 64));
    ("contract_simt_i64_64", Simt (I64, 64));
    ("contract_skinny_f64", Skinny F64);
    ("contract_skinny_i64", Skinny I64);
  |]

(* Tiles *)

(* NX_CUDA_TILES's columns. *)
type shape = { bm : int; bn : int; bkb : int; wm : int; wn : int; stages : int }

(* NX_CUDA_TILES's rows in order. *)
let tiles = [| T128x128; T128x256; T64x64; T16x64 |]

let shape = function
  | T128x128 -> { bm = 128; bn = 128; bkb = 64; wm = 64; wn = 32; stages = 4 }
  | T128x256 -> { bm = 128; bn = 256; bkb = 64; wm = 64; wn = 64; stages = 4 }
  | T64x64 -> { bm = 64; bn = 64; bkb = 128; wm = 32; wn = 32; stages = 4 }
  | T16x64 -> { bm = 16; bn = 64; bkb = 128; wm = 16; wn = 16; stages = 4 }

(* Constants *)

(* contract_params' bits of [aligned]. *)
let a_vectors = 1
let b_vectors = 2
let b_across = 4
let y_whole = 8

(* NX_SKINNY_ROWS. *)
let skinny_rows = 4

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
  let unused = 188
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
