(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Bigarray.Array1

type buf = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t

let starts_scalar (v : buf) j = A.unsafe_get v j land 0xc0 <> 0x80

(* A string column holds valid UTF-8, so a sequence that runs past [stop] only
   guards every read to stay within the row. *)
let width (v : buf) j stop =
  let b0 = A.unsafe_get v j in
  let w =
    if b0 < 0xc0 then 1
    else if b0 < 0xe0 then 2
    else if b0 < 0xf0 then 3
    else if b0 < 0xf8 then 4
    else 1
  in
  if j + w > stop then 1 else w

let scalar (v : buf) j w =
  let b0 = A.unsafe_get v j and b1 = A.unsafe_get v (j + 1) land 0x3f in
  if w = 2 then ((b0 land 0x1f) lsl 6) lor b1
  else
    let b2 = A.unsafe_get v (j + 2) land 0x3f in
    if w = 3 then ((b0 land 0x0f) lsl 12) lor (b1 lsl 6) lor b2
    else
      let b3 = A.unsafe_get v (j + 3) land 0x3f in
      ((b0 land 0x07) lsl 18) lor (b1 lsl 12) lor (b2 lsl 6) lor b3

let decode (v : buf) j stop =
  let w = width v j stop in
  if w = 1 then A.unsafe_get v j else scalar v j w

let decode_before (v : buf) first j =
  let k = ref (j - 1) in
  while !k > first && not (starts_scalar v !k) do
    decr k
  done;
  decode v !k j

let rec within bounds u lo hi =
  if lo = hi then lo land 1 = 1
  else
    let mid = (lo + hi) / 2 in
    if bounds.(mid) <= u then within bounds u (mid + 1) hi
    else within bounds u lo mid
