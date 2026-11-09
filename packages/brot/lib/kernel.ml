(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The walkers read native words as little-endian, so a big-endian machine
   takes the OCaml path. *)
let available = Sys.backend_type = Sys.Native && not Sys.big_endian

(* ABI pins, mirrored in brot_kernels.c — change neither without the other: the
   [byte_level] field order is the C [BROT_BL_*] enum, the [reason] constructor
   order the [BROT_*] enum, and the cursor offsets the [CUR_*] enum. The cache
   entry layout is pinned in pretoken_cache.mli, the span word layout in
   spans.ml. *)

type byte_level = {
  lead : Bytes.t; (* 0: BROT_BL_LEAD *)
  cache : Bytes.t; (* 1: BROT_BL_CACHE *)
  cache_base : int; (* 2: BROT_BL_CACHE_BASE *)
  cache_mask : int; (* 3: BROT_BL_CACHE_MASK *)
  byte_ids : int array; (* 4: BROT_BL_BYTE_IDS *)
  merge_keys : int array; (* 5: BROT_BL_MERGE_KEYS *)
  merge_values : int array; (* 6: BROT_BL_MERGE_VALUES *)
  merge_mask : int; (* 7: BROT_BL_MERGE_MASK *)
  len_table : int array; (* 8: BROT_BL_LEN_TABLE *)
  words : Bytes.t; (* 9: BROT_BL_WORDS *)
  words_mask : int; (* 10: BROT_BL_WORDS_MASK *)
}

type sp = {
  cache : Bytes.t; (* 0: BROT_SP_CACHE *)
  cache_base : int; (* 1: BROT_SP_CACHE_BASE *)
  cache_mask : int; (* 2: BROT_SP_CACHE_MASK *)
  merge_keys : int array; (* 3: BROT_SP_MERGE_KEYS *)
  merge_values : int array; (* 4: BROT_SP_MERGE_VALUES *)
  merge_mask : int; (* 5: BROT_SP_MERGE_MASK *)
  len_table : int array; (* 6: BROT_SP_LEN_TABLE *)
  ascii_ids : int array; (* 7: BROT_SP_ASCII_IDS *)
  char_keys : int array; (* 8: BROT_SP_CHAR_KEYS *)
  char_values : int array; (* 9: BROT_SP_CHAR_VALUES *)
  char_mask : int; (* 10: BROT_SP_CHAR_MASK *)
  scan : Bytes.t; (* 11: BROT_SP_SCAN *)
  punct_safe : bool array; (* 12: BROT_SP_PUNCT_SAFE *)
  fallback_ids : int array; (* 13: BROT_SP_FALLBACK_IDS *)
  marker : string; (* 14: BROT_SP_MARKER *)
}

type reason = Done | Spans_full | Ids_full | Class | Encode
type ids32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

external get64u : Bytes.t -> int -> int64 = "%caml_bytes_get64u"
external set64u : Bytes.t -> int -> int64 -> unit = "%caml_bytes_set64u"

(* The counts, then the stage: 328 spans of 40 bytes ([STAGE_CAP] and [staged]
   in the C). *)
let cursor () = Bytes.make (72 + (328 * 40)) '\000'
let restart cur = set64u cur 40 0L

let set cur ~spans ~ids ~marks =
  set64u cur 0 (Int64.of_int spans);
  set64u cur 8 (Int64.of_int ids);
  set64u cur 16 (Int64.of_int marks)

let[@inline] spans cur = Int64.to_int (get64u cur 0)
let[@inline] ids cur = Int64.to_int (get64u cur 8)
let[@inline] marks cur = Int64.to_int (get64u cur 16)
let[@inline] resume cur = Int64.to_int (get64u cur 24)
let[@inline] code_point cur = Int64.to_int (get64u cur 32)
let[@inline] unit_stop cur = Int64.to_int (get64u cur 32)
let[@inline] added cur = Int64.to_int (get64u cur 64)

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

external sp_encode :
  string -> int -> int -> int array -> Bytes.t -> sp -> reason
  = "brot_sp_encode_byte" "brot_sp_encode"
[@@noalloc]

external byte_level_encode_ids32 :
  string -> int -> int -> ids32 -> Bytes.t -> Bytes.t -> byte_level -> reason
  = "brot_byte_level_encode_ids32_byte" "brot_byte_level_encode_ids32"
[@@noalloc]

external split_encode :
  string ->
  int ->
  int ->
  Bytes.t ->
  int array ->
  int array ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  int ->
  reason = "brot_split_encode_byte" "brot_split_encode"
[@@noalloc]

external split_encode_ids32 :
  string ->
  int ->
  int ->
  ids32 ->
  Bytes.t ->
  Bytes.t ->
  byte_level ->
  int ->
  reason = "brot_split_encode_ids32_byte" "brot_split_encode_ids32"
[@@noalloc]

external sp_encode_ids32 :
  string -> int -> int -> ids32 -> Bytes.t -> sp -> reason
  = "brot_sp_encode_ids32_byte" "brot_sp_encode_ids32"
[@@noalloc]

external find_byte : string -> int -> int -> int -> int = "brot_find_byte"
[@@noalloc]

external cache_hash : int64 -> int64 -> int = "brot_cache_hash" [@@noalloc]
external cache_align : Bytes.t -> int = "brot_cache_align" [@@noalloc]
