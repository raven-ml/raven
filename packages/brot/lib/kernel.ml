(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let available = Sys.backend_type = Sys.Native

(* ABI pins, mirrored in brot_kernels.c — change neither without the other: the
   [byte_level] field order is the C [BROT_BL_*] enum, the [reason] constructor
   order the [BROT_*] enum, and the cursor offsets the [CUR_*] enum. The cache
   entry layout is pinned in bpe.ml, the span word layout in spans.ml. *)

type byte_level = {
  lead : Bytes.t; (* 0: BROT_BL_LEAD *)
  cache : Bytes.t; (* 1: BROT_BL_CACHE *)
  cache_mask : int; (* 2: BROT_BL_CACHE_MASK *)
  byte_ids : int array; (* 3: BROT_BL_BYTE_IDS *)
  merge_keys : int array; (* 4: BROT_BL_MERGE_KEYS *)
  merge_values : int array; (* 5: BROT_BL_MERGE_VALUES *)
  merge_mask : int; (* 6: BROT_BL_MERGE_MASK *)
  len_table : int array; (* 7: BROT_BL_LEN_TABLE *)
  merge : bool; (* 8: BROT_BL_MERGE *)
}

type reason = Done | Spans_full | Ids_full | Class | Encode

external get64u : Bytes.t -> int -> int64 = "%caml_bytes_get64u"
external set64u : Bytes.t -> int -> int64 -> unit = "%caml_bytes_set64u"

let cursor () = Bytes.make 40 '\000'

let set cur ~spans ~ids ~marks =
  set64u cur 0 (Int64.of_int spans);
  set64u cur 8 (Int64.of_int ids);
  set64u cur 16 (Int64.of_int marks)

let[@inline] spans cur = Int64.to_int (get64u cur 0)
let[@inline] ids cur = Int64.to_int (get64u cur 8)
let[@inline] marks cur = Int64.to_int (get64u cur 16)
let[@inline] resume cur = Int64.to_int (get64u cur 24)
let[@inline] code_point cur = Int64.to_int (get64u cur 32)

(* [@@noalloc] holds by construction: the C allocates nothing, raises nothing
   and never releases the runtime lock, so there is no GC point in a call. *)
external byte_level_encode :
  string ->
  int ->
  int ->
  Bytes.t ->
  int array ->
  int array ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  reason = "brot_byte_level_encode_byte" "brot_byte_level_encode"
[@@noalloc]
