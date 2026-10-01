(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Nx_device.Buffer
module A = Bigarray.Array1

type bytes = (int, Nx.uint8_elt) Nx_ragged.t

(* [reading ~by x f] is [f b] for [b] a host buffer of [x]'s elements in C
   order, under a read claim so that no compiled call lends its memory while [f]
   reads it. *)
let reading ~by x f =
  let b = Nx.Op.eval (Read { by; x }) in
  B.Claim.read b;
  Fun.protect ~finally:(fun () -> B.Claim.release b) (fun () -> f b)

(* UTF-8 *)

(* [within a j stop lo hi] is [true] iff byte [j] lies before [stop] and in [lo,
   hi]. *)
let within a j stop lo hi =
  j < stop
  &&
  let c = A.unsafe_get a j in
  lo <= c && c <= hi

(* [sequence a i stop] is the length of the valid UTF-8 sequence that starts at
   byte [i], or [0]. The ranges of the second byte exclude overlong forms,
   surrogates and values past U+10FFFF (RFC 3629, section 4). *)
let sequence a i stop =
  let b = A.unsafe_get a i in
  let tail j = within a j stop 0x80 0xbf in
  if b < 0x80 then 1
  else if b < 0xc2 then 0
  else if b < 0xe0 then if tail (i + 1) then 2 else 0
  else if b < 0xf0 then
    let lo, hi =
      match b with
      | 0xe0 -> (0xa0, 0xbf)
      | 0xed -> (0x80, 0x9f)
      | _ -> (0x80, 0xbf)
    in
    if within a (i + 1) stop lo hi && tail (i + 2) then 3 else 0
  else if b < 0xf5 then
    let lo, hi =
      match b with
      | 0xf0 -> (0x90, 0xbf)
      | 0xf4 -> (0x80, 0x8f)
      | _ -> (0x80, 0xbf)
    in
    if within a (i + 1) stop lo hi && tail (i + 2) && tail (i + 3) then 4 else 0
  else 0

(* [invalid a i stop] is the first byte of [i, stop) that starts no valid
   sequence, or [-1]. *)
let rec invalid a i stop =
  if i >= stop then -1
  else match sequence a i stop with 0 -> i | n -> invalid a (i + n) stop

let utf_8 ~by ?mask r =
  let n = Nx_ragged.length r and read x f = reading ~by x f in
  let first o v skip =
    let rec row i =
      if i = n then None
      else if skip i then row (i + 1)
      else
        let start = Int64.to_int (A.unsafe_get o i) in
        match invalid v start (Int64.to_int (A.unsafe_get o (i + 1))) with
        | -1 -> row (i + 1)
        | j -> Some (i, Printf.sprintf "invalid UTF-8 at byte %d" (j - start))
    in
    row 0
  in
  read (Nx_ragged.offsets r) @@ fun o ->
  read (Nx_ragged.values r) @@ fun v ->
  let o = B.bigarray Bigarray.int64 o
  and v = B.bigarray Bigarray.int8_unsigned v in
  match mask with
  | None -> first o v (Fun.const false)
  | Some mask ->
      read mask @@ fun m ->
      let m = B.bigarray Bigarray.int8_unsigned m in
      first o v (fun i -> A.unsafe_get m i = 0)
