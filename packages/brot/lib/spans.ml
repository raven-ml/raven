(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external set64u : bytes -> int -> int64 -> unit = "%caml_bytes_set64u"
external get64u : bytes -> int -> int64 = "%caml_bytes_get64u"

type t = { buf : bytes; capacity : int; mutable count : int }

let create ~capacity =
  { buf = Bytes.create (8 * capacity); capacity; count = 0 }

let[@inline] capacity t = t.capacity
let[@inline] count t = t.count
let[@inline] clear t = t.count <- 0
let[@inline] set_count t n = t.count <- n

let[@inline] start t k =
  Int64.to_int (Int64.logand (get64u t.buf (k * 8)) 0xFFFFFFFFL)

let[@inline] stop t k =
  Int64.to_int (Int64.shift_right_logical (get64u t.buf (k * 8)) 32)

let[@inline] write t k s e =
  set64u t.buf (k * 8)
    (Int64.logor (Int64.of_int s) (Int64.shift_left (Int64.of_int e) 32))
