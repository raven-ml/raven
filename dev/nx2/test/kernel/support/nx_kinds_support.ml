(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external f32 : string -> int array -> int = "nx_kinds_support_f32"
external f64 : string -> float array -> float = "nx_kinds_support_f64"

external int : string -> string -> int64 array -> int64 = "nx_kinds_support_int"
external threefry : int64 -> int64 -> int64 = "nx_kinds_support_threefry"
external c64 : string -> int array -> int array = "nx_kinds_support_c64"
external c128 : string -> float array -> float array = "nx_kinds_support_c128"

external run :
  string ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  unit = "nx_kinds_support_run"
[@@noalloc]
