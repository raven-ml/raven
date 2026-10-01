(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type bigbytes = (int, int8_unsigned_elt, c_layout) Array1.t

external overlap : bigbytes -> bigbytes -> bool = "caml_compress_lz4_overlap"
[@@noalloc]

(* [block src dst io] decodes a block; the positions travel in [io], [| src_pos;
   src_end; dst_hist; dst_pos; dst_end |]. *)
external block : bigbytes -> bigbytes -> int array -> int
  = "caml_compress_lz4_block"

external xxh32 : bigbytes -> int -> int -> int = "caml_compress_lz4_xxh32"

exception Malformed of int * string

let malformed at msg = raise (Malformed (at, msg))

let block_message = function
  | 1 -> "truncated data"
  | 2 -> "data longer than the destination"
  | 3 -> "match at offset 0"
  | 4 -> "match before the data"
  | _ -> assert false

(* Decodes the block [src[first, last)] into [dst] from [io.(3)], with matches
   back to [hist]. *)
let decode_block src first last dst ~hist io =
  io.(0) <- first;
  io.(1) <- last;
  io.(2) <- hist;
  let status = block src dst io in
  if status <> 0 then malformed io.(0) (block_message status)

let run ~fn decode src dst =
  if overlap src dst then invalid_arg (fn ^ ": src and dst overlap");
  let io = [| 0; 0; 0; 0; Array1.dim dst |] in
  match decode src dst io with
  | () when io.(3) <> Array1.dim dst ->
      Error
        (Printf.sprintf "data is %d bytes, not %d at byte %d" io.(3)
           (Array1.dim dst) (Array1.dim src))
  | () -> Ok ()
  | exception Malformed (at, msg) ->
      Error (Printf.sprintf "%s at byte %d" msg at)

module Block = struct
  let decompress_into src dst =
    let decode src dst io =
      if Array1.dim src = 0 then malformed 0 "truncated data";
      decode_block src 0 (Array1.dim src) dst ~hist:0 io
    in
    run ~fn:"Compress_lz4.Block.decompress_into" decode src dst
end

module Frame = struct
  let magic = 0x184D2204
  let legacy_magic = 0x184C2102

  let le32 src at =
    if at + 4 > Array1.dim src then malformed at "truncated data";
    Array1.get src at
    lor (Array1.get src (at + 1) lsl 8)
    lor (Array1.get src (at + 2) lsl 16)
    lor (Array1.get src (at + 3) lsl 24)

  let byte src at =
    if at >= Array1.dim src then malformed at "truncated data";
    Array1.get src at

  (* Decodes the frame at [at] and is the offset after it. *)
  let frame src dst io at =
    let flg = byte src (at + 4) in
    let bd = byte src (at + 5) in
    if flg lsr 6 <> 1 then malformed (at + 4) "unknown frame version";
    if flg land 0x02 <> 0 || bd land 0x8F <> 0 then
      malformed (at + 4) "reserved bits set";
    if flg land 0x01 <> 0 then malformed (at + 4) "dictionary needed";
    let block_max = 1 lsl (8 + (2 * ((bd lsr 4) land 7))) in
    if (bd lsr 4) land 7 < 4 then
      malformed (at + 5) "invalid block maximum size";
    let independent = flg land 0x20 <> 0 in
    let block_checksum = flg land 0x10 <> 0 in
    let has_size = flg land 0x08 <> 0 in
    let content_checksum = flg land 0x04 <> 0 in
    let descriptor = if has_size then 10 else 2 in
    let hc = byte src (at + 4 + descriptor) in
    if (xxh32 src (at + 4) descriptor lsr 8) land 0xFF <> hc then
      malformed (at + 4 + descriptor) "header checksum mismatch";
    let size =
      if has_size then (
        let low = le32 src (at + 6) and high = le32 src (at + 10) in
        if high >= 0x80000000 then malformed (at + 6) "content size too large";
        Some ((high lsl 32) lor low))
      else None
    in
    let start = io.(3) in
    let rec blocks pos =
      let header = le32 src pos in
      if header = 0 then pos + 4
      else begin
        let length = header land 0x7FFFFFFF in
        let data = pos + 4 in
        if length > block_max then
          malformed pos "block larger than the frame's maximum";
        if data + length > Array1.dim src then malformed pos "truncated data";
        if block_checksum && le32 src (data + length) <> xxh32 src data length
        then malformed (data + length) "block checksum mismatch";
        if header land 0x80000000 <> 0 then begin
          if length > io.(4) - io.(3) then
            malformed data "data longer than the destination";
          Array1.blit
            (Array1.sub src data length)
            (Array1.sub dst io.(3) length);
          io.(3) <- io.(3) + length
        end
        else
          decode_block src data (data + length) dst
            ~hist:(if independent then io.(3) else start)
            io;
        blocks (data + length + if block_checksum then 4 else 0)
      end
    in
    let after = blocks (at + 5 + descriptor) in
    let produced = io.(3) - start in
    (match size with
    | Some size when size <> produced -> malformed after "content size mismatch"
    | _ -> ());
    if content_checksum then begin
      if le32 src after <> xxh32 dst start produced then
        malformed after "content checksum mismatch";
      after + 4
    end
    else after

  let decompress_into src dst =
    let decode src dst io =
      let n = Array1.dim src in
      if n = 0 then malformed 0 "no frame";
      let rec frames at =
        if at < n then begin
          let m = le32 src at in
          if m land 0xFFFFFFF0 = 0x184D2A50 then
            frames (at + 8 + le32 src (at + 4))
          else if m = legacy_magic then malformed at "legacy frame format"
          else if m <> magic then malformed at "not an LZ4 frame"
          else frames (frame src dst io at)
        end
        else if at > n then malformed n "truncated data"
      in
      frames 0
    in
    run ~fn:"Compress_lz4.Frame.decompress_into" decode src dst
end
