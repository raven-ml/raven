(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  FLAC in its native container (RFC 9639): the fLaC marker, metadata blocks,
  then frames. A load reads the metadata: STREAMINFO, the channel mask of a
  Vorbis comment, and skips the other blocks. nx_io_flac.c decodes frames.
  ---------------------------------------------------------------------------*)

open Bigarray
module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* A FLAC stream: STREAMINFO's facts, the channel mask, the byte [first] of the
   first frame, and [stream], the facts nx_io_flac.c reads, in the order of its
   [Fields]. *)
type t = {
  file : B.t;
  rate : int;
  channels : int;
  bits : int;
  channel_mask : int;
  max_block : int;
  length : int;
  first : int;
  dt : D.any;
  stream : int array;
}

let rate t = t.rate
let length t = t.length
let channels t = t.channels
let bits t = t.bits
let channel_mask t = t.channel_mask

(* Metadata (RFC 9639, section 8) *)

let streaminfo = 0
let seektable = 3
let vorbis_comment = 4
let forbidden = 127

(* The speakers of each count of channels, in the order of section 9.1.3, whose
   "back/surround" channels of 5 and 6 are the side ones, as the reference
   decoder and FFmpeg read them. *)
let default_masks = [| 0x4; 0x3; 0x7; 0x33; 0x607; 0x60F; 0x70F; 0x63F |]
let u32_le s i = Int32.to_int (String.get_int32_le s i) land 0xffff_ffff

(* The channel mask a Vorbis comment [s] states, if any (8.6, 8.6.2). Field
   names compare without case. *)
let channel_mask_field s =
  let len = String.length s in
  let malformed () = failwith "the Vorbis comment is malformed" in
  let field at =
    if at > len - 4 then malformed ();
    let n = u32_le s at in
    if n > len - at - 4 then malformed ();
    (String.sub s (at + 4) n, at + 4 + n)
  in
  let _, at = field 0 in
  if at > len - 4 then malformed ();
  let count = u32_le s at in
  let name = "WAVEFORMATEXTENSIBLE_CHANNEL_MASK=" in
  let rec fields k at mask =
    if k = count then mask
    else
      let f, at = field at in
      let n = String.length name in
      if
        String.length f >= n
        && String.lowercase_ascii (String.sub f 0 n)
           = String.lowercase_ascii name
      then begin
        if mask <> None then
          failwith "the Vorbis comment states two channel masks";
        let v = String.sub f n (String.length f - n) in
        let hex =
          String.length v > 2
          && String.lowercase_ascii (String.sub v 0 2) = "0x"
          && String.for_all
               (function
                 | '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true | _ -> false)
               (String.sub v 2 (String.length v - 2))
        in
        let mask = if hex then int_of_string_opt v else None in
        match mask with
        | Some m when m <= 0xffff_ffff -> fields (k + 1) at (Some m)
        | _ ->
            Fail.failf
              "the Vorbis comment's channel mask %S is no 32-bit hex number" v
      end
      else fields (k + 1) at mask
  in
  fields 0 (at + 4) None

(* The 64-bit value that marks a placeholder point. *)
let placeholder = -1L

(* [check_points s] checks the seek table [s] (8.5.1): whole points of 18 bytes,
   each a row, a byte and a count of rows, sorted by row, distinct, placeholders
   last. *)
let check_points s =
  let len = String.length s in
  if len mod 18 <> 0 then
    Fail.failf "the seek table's %d bytes are not whole points of 18" len;
  let malformed () = failwith "the seek table is malformed" in
  let u64 i =
    let v = String.get_int64_be s i in
    if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
      malformed ();
    Int64.to_int v
  in
  let rec go i last =
    if i < len then
      if String.get_int64_be s i = placeholder then begin
        for j = 1 to ((len - i) / 18) - 1 do
          if String.get_int64_be s (i + (18 * j)) <> placeholder then
            malformed ()
        done
      end
      else begin
        let row = u64 i in
        ignore (u64 (i + 8) : int);
        if row <= last then malformed ();
        go (i + 18) row
      end
  in
  go 0 (-1)

(* STREAMINFO's fields (8.2). *)
type info = {
  min_block : int;
  max_block : int;
  min_frame : int;
  max_frame : int;
  rate : int;
  channels : int;
  bits : int;
  total : int;
}

let info s =
  let u16 i = String.get_uint16_be s i in
  let u24 i =
    (String.get_uint8 s i lsl 16) lor String.get_uint16_be s (i + 1)
  in
  let v = String.get_int64_be s 10 in
  let field shift width =
    Int64.to_int (Int64.shift_right_logical v shift) land ((1 lsl width) - 1)
  in
  let i =
    {
      min_block = u16 0;
      max_block = u16 2;
      min_frame = u24 4;
      max_frame = u24 7;
      rate = field 44 20;
      channels = field 41 3 + 1;
      bits = field 36 5 + 1;
      total = field 0 36;
    }
  in
  if i.min_block < 16 || i.max_block < i.min_block then
    Fail.failf
      "STREAMINFO states blocks of %d to %d rows, not within 16 to 65535"
      i.min_block i.max_block;
  if i.bits < 4 then Fail.failf "STREAMINFO states %d bits, fewer than 4" i.bits;
  i

(* Reading *)

external decode :
  Storage.bytes ->
  int array ->
  int array ->
  int ->
  Storage.bytes ->
  int ->
  int ->
  Storage.bytes ->
  int = "caml_nx_io_flac_decode_byte" "caml_nx_io_flac_decode"

(* The largest bytes a frame's byte decodes to, 59,918 for 17-bit samples of 8
   channels in constant subframes of 65535 rows: so a length that would take
   more is refused before anything is allocated (RFC 9639, section 11). *)
let expansion = 1 lsl 16

(* The index of the length in [stream]. *)
let length_field = 7
let empty = Array1.create int8_unsigned c_layout 0
let bytes n = Array1.create int8_unsigned c_layout n

(* A cursor at the first frame. *)
let start t = [| t.first; 0 |]

(* [run t cursor ~skip dst ~at ~rows rest] decodes as nx_io_flac.c states,
   naming the frame that fails. *)
let run t cursor ~skip dst ~at ~rows rest =
  Storage.read (Storage.whole t.file) @@ fun src ->
  try decode src t.stream cursor skip dst at rows rest
  with Failure why ->
    Fail.failf "frame at row %d, byte %d: %s" cursor.(1) cursor.(0) why

let read file =
  let length = B.length file in
  let rec blocks pos acc =
    let h = Storage.string file ~pos 4 in
    let last = Char.code h.[0] land 0x80 <> 0
    and kind = Char.code h.[0] land 0x7f in
    let n = (Char.code h.[1] lsl 16) lor String.get_uint16_be h 2 in
    if kind = forbidden then
      Fail.failf "the metadata block at byte %d is of type 127" pos;
    if n > length - pos - 4 then
      Fail.failf "the metadata block at byte %d ends past the file" pos;
    let acc = (kind, pos + 4, n) :: acc in
    if last then (List.rev acc, pos + 4 + n) else blocks (pos + 4 + n) acc
  in
  let all, first = blocks 4 [] in
  let body (_, at, n) = Storage.string file ~pos:at n in
  let info =
    match all with
    | ((k, _, 34) as b) :: _ when k = streaminfo -> info (body b)
    | (k, _, _) :: _ when k = streaminfo ->
        failwith "STREAMINFO is not 34 bytes"
    | _ -> failwith "STREAMINFO is not the first metadata block"
  in
  let count k = List.length (List.filter (fun (k', _, _) -> k' = k) all) in
  if count streaminfo > 1 then failwith "the file has two STREAMINFO blocks";
  if count seektable > 1 then failwith "the file has two seek tables";
  if count vorbis_comment > 1 then failwith "the file has two Vorbis comments";
  let channel_mask =
    match List.find_opt (fun (k, _, _) -> k = vorbis_comment) all with
    | Some b -> channel_mask_field (body b)
    | None -> None
  in
  let channel_mask =
    Option.value channel_mask ~default:default_masks.(info.channels - 1)
  in
  Option.iter
    (fun b -> check_points (body b))
    (List.find_opt (fun (k, _, _) -> k = seektable) all);
  let dt, width =
    if info.bits <= 8 then (D.Any D.Int8, 1)
    else if info.bits <= 16 then (D.Any D.Int16, 2)
    else (D.Any D.Int32, 4)
  in
  (* The first frame's blocking strategy bit; frames before 2007 marked a
     variable block size only by STREAMINFO's sizes (B.1). *)
  let strategy =
    if first + 1 < length then
      Char.code (Storage.string file ~pos:(first + 1) 1).[0] land 1
    else 0
  in
  let numbered_by_row = strategy = 1 || info.min_block <> info.max_block in
  let stated = if info.total = 0 then -1 else info.total in
  if stated * info.channels * width > expansion * (length - first) then
    Fail.failf "STREAMINFO states %d rows, more than the frames' %d bytes code"
      stated (length - first);
  let stream =
    [|
      info.channels;
      info.bits;
      info.rate;
      info.min_block;
      info.max_block;
      info.min_frame;
      info.max_frame;
      stated;
      strategy;
      Bool.to_int numbered_by_row;
      length;
      width;
    |]
  in
  let t =
    {
      file;
      rate = info.rate;
      channels = info.channels;
      bits = info.bits;
      channel_mask;
      max_block = info.max_block;
      length = stated;
      first;
      dt;
      stream;
    }
  in
  if stated >= 0 then t
  else begin
    (* Without a length, every frame is decoded to count the rows. *)
    let cursor = start t in
    ignore (run t cursor ~skip:0 empty ~at:0 ~rows:(-1) empty : int);
    let n = cursor.(1) in
    let stream = Array.copy stream in
    stream.(length_field) <- n;
    { t with length = n; stream }
  end

let samples t =
  let (D.Any dt) = t.dt in
  let stride = t.channels * D.bytes dt 1 in
  let out = B.create Rig.host (t.length * stride) in
  let rest = bytes (t.max_block * stride) in
  B.write out int8_unsigned (fun dst ->
      ignore (run t (start t) ~skip:0 dst ~at:0 ~rows:t.length rest : int));
  let layout = L.contiguous (Iarray.of_array [| t.length; t.channels |]) in
  Nx.P (Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt layout out))
