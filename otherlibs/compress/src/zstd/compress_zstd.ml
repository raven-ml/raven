(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

type bigbytes = (int, int8_unsigned_elt, c_layout) Array1.t
type context

external create : unit -> context = "caml_compress_zstd_create"
external reset : context -> unit = "caml_compress_zstd_reset" [@@noalloc]

external overlap : bigbytes -> bigbytes -> bool = "caml_compress_zstd_overlap"
[@@noalloc]

(* [block z src dst io] decodes a compressed block; the positions travel in
   [io], [| src_pos; src_end; dst_hist; dst_pos; dst_end |]. *)
external block : context -> bigbytes -> bigbytes -> int array -> int
  = "caml_compress_zstd_block"

external xxh64 : bigbytes -> int -> int -> int = "caml_compress_zstd_xxh64"

exception Malformed of int * string

let malformed at msg = raise (Malformed (at, msg))

let block_message = function
  | 1 -> "truncated data"
  | 2 -> "invalid literals section"
  | 3 -> "invalid Huffman table"
  | 4 -> "invalid sequences section"
  | 5 -> "match before the data"
  | 6 -> "data longer than the destination"
  | 7 -> "match at offset 0"
  | _ -> assert false

let magic = 0xFD2FB528
let block_max = 128 * 1024

(* The [n]-byte little-endian integer at [at]. *)
let le src at n =
  if at + n > Array1.dim src then malformed at "truncated data";
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Array1.get src (at + i)
  done;
  !v

(* Decodes the frame at [at] and is the offset after it. *)
let frame z src dst io at =
  let fhd = le src (at + 4) 1 in
  let single = fhd land 0x20 <> 0 in
  if fhd land 0x08 <> 0 then malformed (at + 4) "reserved bit set";
  let pos = at + 5 + if single then 0 else 1 in
  let dict_bytes = [| 0; 1; 2; 4 |].(fhd land 3) in
  if dict_bytes > 0 && le src pos dict_bytes <> 0 then
    malformed pos "dictionary needed";
  let pos = pos + dict_bytes in
  let size_bytes = [| (if single then 1 else 0); 2; 4; 8 |].(fhd lsr 6) in
  let size =
    if size_bytes = 0 then None
    else if size_bytes = 8 && le src (pos + 7) 1 >= 0x40 then
      malformed pos "content size too large"
    else Some (le src pos size_bytes + if size_bytes = 2 then 256 else 0)
  in
  reset z;
  let start = io.(3) in
  let rec blocks pos =
    let header = le src pos 3 in
    let last = header land 1 <> 0 in
    let size = header lsr 3 in
    let data = pos + 3 in
    if size > block_max then malformed pos "block larger than 128 KiB";
    let next =
      match (header lsr 1) land 3 with
      | 0 ->
          if data + size > Array1.dim src then malformed data "truncated data";
          if size > io.(4) - io.(3) then
            malformed pos "data longer than the destination";
          Array1.blit (Array1.sub src data size) (Array1.sub dst io.(3) size);
          io.(3) <- io.(3) + size;
          data + size
      | 1 ->
          let byte = le src data 1 in
          if size > io.(4) - io.(3) then
            malformed pos "data longer than the destination";
          Array1.fill (Array1.sub dst io.(3) size) byte;
          io.(3) <- io.(3) + size;
          data + 1
      | 2 ->
          if data + size > Array1.dim src then malformed data "truncated data";
          io.(0) <- data;
          io.(1) <- data + size;
          io.(2) <- start;
          let status = block z src dst io in
          if status <> 0 then malformed pos (block_message status);
          data + size
      | _ -> malformed pos "reserved block type"
    in
    if last then next else blocks next
  in
  let after = blocks (pos + size_bytes) in
  let produced = io.(3) - start in
  (match size with
  | Some size when size <> produced -> malformed after "content size mismatch"
  | _ -> ());
  if fhd land 0x04 = 0 then after
  else begin
    if le src after 4 <> xxh64 dst start produced then
      malformed after "content checksum mismatch";
    after + 4
  end

let decompress src dst =
  if overlap src dst then
    invalid_arg "Compress_zstd.decompress: src and dst overlap";
  let n = Array1.dim src in
  let io = [| 0; 0; 0; 0; Array1.dim dst |] in
  let z = create () in
  let rec frames at =
    if at < n then begin
      let m = le src at 4 in
      if m land 0xFFFFFFF0 = 0x184D2A50 then frames (at + 8 + le src (at + 4) 4)
      else if m <> magic then malformed at "not a Zstandard frame"
      else frames (frame z src dst io at)
    end
    else if at > n then malformed n "truncated data"
  in
  match
    if n = 0 then malformed 0 "no frame";
    frames 0;
    if io.(3) <> Array1.dim dst then
      malformed n
        (Printf.sprintf "data is %d bytes, not %d" io.(3) (Array1.dim dst))
  with
  | () -> Ok ()
  | exception Malformed (at, msg) ->
      Error (Printf.sprintf "%s at byte %d" msg at)
