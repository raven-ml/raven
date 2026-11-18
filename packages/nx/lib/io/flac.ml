(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  FLAC in its native container (RFC 9639): the fLaC marker, metadata blocks,
  then frames. A load reads the metadata: STREAMINFO, the channel mask of a
  Vorbis comment, and skips the other blocks. nx_io_flac.c decodes frames and
  adds their samples to a digest held in an OCaml bytes value, a fresh copy for
  each decode; nx_io_md5.c holds the MD5.
  ---------------------------------------------------------------------------*)

open Bigarray
module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* A seek point (8.5.1): the first row of the frame it names, the frame's byte
   from the first frame's, and its rows. *)
type point = { row : int; offset : int; rows : int }

(* A FLAC stream: STREAMINFO's facts and MD5, the channel mask, the byte [first]
   of the first frame, and [stream], the facts nx_io_flac.c reads, in the order
   of its [Fields]. *)
type t = {
  file : B.t;
  rate : int;
  channels : int;
  bits : int;
  channel_mask : int;
  max_block : int;
  length : int;
  md5 : string;
  first : int;
  points : point array;
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

(* The points of a seek table [s] (8.5.1), whole points of 18 bytes sorted by
   row and distinct, but for its placeholders, which come last and are left
   out. *)
let points s =
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
  let rec go i last acc =
    if i = len then List.rev acc
    else if String.get_int64_be s i = placeholder then begin
      for j = 1 to ((len - i) / 18) - 1 do
        if String.get_int64_be s (i + (18 * j)) <> placeholder then malformed ()
      done;
      List.rev acc
    end
    else
      let p =
        {
          row = u64 i;
          offset = u64 (i + 8);
          rows = String.get_uint16_be s (i + 16);
        }
      in
      if p.row <= last then malformed ();
      go (i + 18) p.row (p :: acc)
  in
  Array.of_list (go 0 (-1) [])

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
  md5 : string;
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
      md5 = String.sub s 18 16;
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
  Bytes.t ->
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

(* The indexes in [stream] of the length, and of the window's length and
   whether the file goes on past it. *)
let length_field = 7
let end_field = 10
let more_field = 12
let empty = Array1.create int8_unsigned c_layout 0
let bytes n = Array1.create int8_unsigned c_layout n

external md5_init : unit -> Bytes.t = "caml_nx_io_md5_init"
external md5_final : Bytes.t -> string = "caml_nx_io_md5_final"

(* A cursor at the first frame. *)
let start t = [| t.first; 0 |]

(* [stream_of t n ~more] is [t]'s facts for a window of [n] bytes, [more]
   whether the file goes on past it. *)
let stream_of t n ~more =
  let s = Array.copy t.stream in
  s.(end_field) <- n;
  s.(more_field) <- Bool.to_int more;
  s

(* [run t cursor ~md5 ~skip dst ~at ~rows rest] decodes as nx_io_flac.c
   states, naming the frame that fails. It reads the file in windows from
   the cursor's frame: the bytes [rows] rows take at the stream's mean
   bytes a row, with a frame and 1 KiB to spare, and twice as many each
   time a frame runs past a window. So a read of a file the system cannot
   map copies about the bytes of the frames it decodes. *)
let run t cursor ~md5 ~skip dst ~at ~rows rest =
  let len = B.length t.file in
  let per_row = ((len - t.first) / Int.max 1 t.length) + 1 in
  let rec go ~skip ~at ~rows window =
    let base = cursor.(0) in
    let n = Int.min window (len - base) in
    let c = [| 0; cursor.(1); 0; 0 |] in
    let kept =
      Storage.read (Storage.pages t.file ~first:base ~length:n) @@ fun src ->
      try
        decode src (stream_of t n ~more:(base + n < len)) c md5 skip dst at
          rows rest
      with Failure why ->
        Fail.failf "frame at row %d, byte %d: %s" c.(1) (base + c.(0)) why
    in
    cursor.(0) <- base + c.(0);
    cursor.(1) <- c.(1);
    if kept >= 0 then kept
    else
      let rows = if rows < 0 then rows else rows - c.(2) in
      go ~skip:c.(3) ~at:(at + c.(2)) ~rows (2 * window)
  in
  if rows < 0 then go ~skip ~at ~rows (len - cursor.(0))
  else go ~skip ~at ~rows (((rows + skip + t.max_block) * per_row) + 1024)

(* STREAMINFO states no MD5 with zeros (8.2). *)
let unstated = String.make 16 '\000'

(* A digest to feed from the first frame, or none, empty, if STREAMINFO states
   none. *)
let digest md5 = if md5 = unstated then Bytes.empty else md5_init ()

(* [check md5 ctx] raises if the digest [ctx] of every frame differs from
   STREAMINFO's [md5]. *)
let check md5 ctx =
  if Bytes.length ctx > 0 then
    let got = md5_final ctx in
    if got <> md5 then
      Fail.failf "the samples' MD5 is %s, STREAMINFO states %s"
        (Digest.to_hex got) (Digest.to_hex md5)

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
  let points =
    match List.find_opt (fun (k, _, _) -> k = seektable) all with
    | Some b -> points (body b)
    | None -> [||]
  in
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
      0;
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
      md5 = info.md5;
      first;
      points;
      dt;
      stream;
    }
  in
  if stated >= 0 then t
  else begin
    (* Without a length, every frame is decoded to count the rows, and checked
       by the MD5. *)
    let cursor = start t in
    let md5 = digest t.md5 in
    ignore (run t cursor ~md5 ~skip:0 empty ~at:0 ~rows:(-1) empty : int);
    check t.md5 md5;
    let n = cursor.(1) in
    let stream = Array.copy stream in
    stream.(length_field) <- n;
    { t with length = n; stream }
  end

(* Seeking *)

external header : Storage.bytes -> int array -> int -> int array -> bool
  = "caml_nx_io_flac_header"

(* The most bytes a frame's header takes (RFC 9639, 9.1): 4, a coded
   number of 7, an uncommon block size and rate of 2 each, and the
   CRC-8. *)
let header_bytes = 16

(* The cursor of the frame a read from row [row] starts at: the last seek point
   at or before [row] whose frame names its row and rows, or the first frame. A
   point is a claim the frame must confirm (RFC 9639, section 11), so an
   unconfirmed one is passed over. *)
let seek t row =
  let len = B.length t.file in
  let found = [| 0; 0 |] in
  let confirmed p =
    p.row <= row
    && p.offset < len - t.first
    &&
    let pos = t.first + p.offset in
    let n = Int.min header_bytes (len - pos) in
    Storage.read (Storage.pages t.file ~first:pos ~length:n) (fun src ->
        header src (stream_of t n ~more:false) 0 found)
    && found.(0) = p.row
    && found.(1) = p.rows
  in
  let rec last i =
    if i < 0 then start t
    else if confirmed t.points.(i) then
      [| t.first + t.points.(i).offset; t.points.(i).row |]
    else last (i - 1)
  in
  last (Array.length t.points - 1)

(* Blocks *)

(* Where a pass is: the cursor at the next frame to decode, the [kept] rows at
   the start of [rest] that the last frame decoded holds and no block has given,
   and the digest of every frame decoded, or none, empty. A node owns its fields
   and never writes them, so a sequence forced twice reads the same. *)
type pass = {
  cursor : int array;
  rest : Storage.bytes;
  kept : int;
  md5 : Bytes.t;
}

let blocks t ~row ~rows =
  let (D.Any dt) = t.dt in
  let stride = t.channels * D.bytes dt 1 in
  let n = t.length in
  let layout m = L.contiguous (Iarray.of_array [| m; t.channels |]) in
  (* The block of [m] rows from row [row], [skip] rows after the cursor. *)
  let block ~row ~skip m p =
    let out = B.create Rig.host (m * stride) in
    let given = Int.min p.kept m in
    let p =
      B.write out int8_unsigned @@ fun dst ->
      Array1.blit
        (Array1.sub p.rest 0 (given * stride))
        (Array1.sub dst 0 (given * stride));
      if p.kept >= m then
        {
          p with
          rest = Array1.sub p.rest (m * stride) ((p.kept - m) * stride);
          kept = p.kept - m;
        }
      else
        let cursor = Array.copy p.cursor and md5 = Bytes.copy p.md5 in
        let rest = bytes (t.max_block * stride) in
        let kept =
          run t cursor ~md5 ~skip dst ~at:given ~rows:(m - given) rest
        in
        { cursor; rest; kept; md5 }
    in
    (* The node that gives the last block checks the digest. *)
    if row + m = n then check t.md5 p.md5;
    (Nx.P (Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt (layout m) out)), p)
  in
  let rec from ~row ~skip p () =
    if row >= n then Seq.Nil
    else
      let m = Int.min rows (n - row) in
      let x, p = block ~row ~skip m p in
      Seq.Cons (x, from ~row:(row + m) ~skip:0 p)
  in
  let first () =
    if row = 0 then
      from ~row ~skip:0
        { cursor = start t; rest = empty; kept = 0; md5 = digest t.md5 }
        ()
    else
      let cursor = seek t row in
      from ~row
        ~skip:(row - cursor.(1))
        { cursor; rest = empty; kept = 0; md5 = Bytes.empty }
        ()
  in
  first

(* A window of no row reads nothing. *)
let window t ~row ~rows =
  let (D.Any dt) = t.dt in
  let none = Nx.P (Nx.zeros dt [| 0; t.channels |]) in
  if rows = 0 then none
  else
    match blocks t ~row ~rows () with Seq.Cons (x, _) -> x | Seq.Nil -> none

let samples t = window t ~row:0 ~rows:t.length
