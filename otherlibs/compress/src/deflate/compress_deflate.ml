(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bytesrw

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type level = int
type Bytes.Stream.error += Error of string

let format_error format =
  let case msg = Error msg in
  let message = function Error msg -> msg | _ -> assert false in
  Bytes.Stream.make_format_error ~format ~case ~message

external init : unit -> unit = "caml_compress_deflate_init"

let () = init ()

external overlap : bigbytes -> bigbytes -> bool
  = "caml_compress_deflate_overlap"
[@@noalloc]

(* Checksums *)

external crc32_bytes : int -> bytes -> int -> int -> int
  = "caml_compress_deflate_crc32_bytes"
[@@noalloc]

external crc32_bigbytes : int -> bigbytes -> int -> int -> int
  = "caml_compress_deflate_crc32_bigbytes"

external adler32_bytes : int -> bytes -> int -> int -> int
  = "caml_compress_deflate_adler32_bytes"
[@@noalloc]

external adler32_bigbytes : int -> bigbytes -> int -> int -> int
  = "caml_compress_deflate_adler32_bigbytes"

let crc32_byte crc b =
  let c = ref (lnot crc land 0xFFFFFFFF lxor b) in
  for _ = 1 to 8 do
    c := (!c lsr 1) lxor (0xEDB88320 land -(!c land 1))
  done;
  lnot !c land 0xFFFFFFFF

module Crc32 = struct
  type t = int

  let bigbytes ?(crc = 0) b = crc32_bigbytes crc b 0 (Bigarray.Array1.dim b)

  let slice ?(crc = 0) s =
    crc32_bytes crc (Bytes.Slice.bytes s) (Bytes.Slice.first s)
      (Bytes.Slice.length s)
end

(* Inflaters. Positions travel in an int array, [| src_pos; src_end; dst_hist;
   dst_pos; dst_end |], that the C code advances. *)

type inflate

external inflate_create : unit -> inflate = "caml_compress_inflate_create"

external inflate_reset : inflate -> unit = "caml_compress_inflate_reset"
[@@noalloc]

external inflate_bytes : inflate -> bytes -> bytes -> int array -> bool -> int
  = "caml_compress_inflate_bytes"
[@@noalloc]

external inflate_bigbytes :
  inflate -> bigbytes -> bigbytes -> int array -> bool -> int
  = "caml_compress_inflate_bigbytes"

let inflate_end = 0
let inflate_input = 1
let inflate_output = 2

let inflate_message = function
  | 3 -> "truncated data"
  | 4 -> "invalid block type"
  | 5 -> "stored block length does not match its complement"
  | 6 -> "too many length or distance codes"
  | 7 -> "invalid Huffman code lengths"
  | 8 -> "code length repeat out of range"
  | 9 -> "no end-of-block code"
  | 10 -> "invalid symbol"
  | 11 -> "distance beyond the data"
  | _ -> assert false

(* Framings. Headers and trailers are read through [next], which returns the
   next byte, and [pos], the position of that byte. Malformed data raises
   [Malformed] at the position where decoding failed. *)

exception Malformed of int * string

type framing = Raw | Zlib | Gzip
type cursor = { next : unit -> int; pos : unit -> int }

let malformed pos msg = raise (Malformed (pos, msg))

let zlib_header c =
  let at = c.pos () in
  let cmf = c.next () in
  let flg = c.next () in
  if cmf land 0x0F <> 8 || cmf lsr 4 > 7 || (cmf lsl 8) lor flg mod 31 <> 0 then
    malformed at "invalid zlib header";
  if flg land 0x20 <> 0 then malformed at "preset dictionary needed"

let gzip_header c =
  let at = c.pos () in
  let crc = ref 0 in
  let byte () =
    let b = c.next () in
    crc := crc32_byte !crc b;
    b
  in
  if byte () <> 0x1F || byte () <> 0x8B then malformed at "not a gzip member";
  if byte () <> 8 then malformed at "unknown compression method";
  let flags = byte () in
  if flags land 0xE0 <> 0 then malformed at "reserved flags set";
  for _ = 1 to 6 do
    ignore (byte ())
  done;
  if flags land 0x04 <> 0 then begin
    let n = byte () in
    for _ = 1 to n lor (byte () lsl 8) do
      ignore (byte ())
    done
  end;
  if flags land 0x08 <> 0 then
    while byte () <> 0 do
      ()
    done;
  if flags land 0x10 <> 0 then
    while byte () <> 0 do
      ()
    done;
  if flags land 0x02 <> 0 then begin
    let expected = !crc land 0xFFFF in
    let at = c.pos () in
    let b0 = c.next () in
    if b0 lor (c.next () lsl 8) <> expected then
      malformed at "header checksum mismatch"
  end

let le32 c =
  let b0 = c.next () in
  let b1 = c.next () in
  let b2 = c.next () in
  b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (c.next () lsl 24)

let header framing c =
  match framing with Raw -> () | Zlib -> zlib_header c | Gzip -> gzip_header c

(* [trailer framing c ~check ~size] reads the trailer of a stream whose data has
   checksum [check] and [size] bytes. *)
let trailer framing c ~check ~size =
  match framing with
  | Raw -> ()
  | Zlib ->
      let at = c.pos () in
      let b0 = c.next () in
      let b1 = c.next () in
      let b2 = c.next () in
      let adler = (b0 lsl 24) lor (b1 lsl 16) lor (b2 lsl 8) lor c.next () in
      if adler <> check then malformed at "Adler-32 mismatch"
  | Gzip ->
      let at = c.pos () in
      if le32 c <> check then malformed at "CRC-32 mismatch";
      let at = c.pos () in
      if le32 c <> size land 0xFFFFFFFF then malformed at "length mismatch"

let initial_check = function Zlib -> 1 | Raw | Gzip -> 0

(* In memory *)

let decompress framing ~fn src dst =
  if overlap src dst then invalid_arg (fn ^ ": src and dst overlap");
  let src_len = Bigarray.Array1.dim src in
  let dst_len = Bigarray.Array1.dim dst in
  let io = [| 0; src_len; 0; 0; dst_len |] in
  let c =
    let next () =
      let p = io.(0) in
      if p >= src_len then malformed p "truncated data";
      io.(0) <- p + 1;
      Bigarray.Array1.unsafe_get src p
    in
    { next; pos = (fun () -> io.(0)) }
  in
  let state = inflate_create () in
  let rec stream () =
    header framing c;
    let start = io.(3) in
    io.(2) <- start;
    let status = inflate_bigbytes state src dst io true in
    if status = inflate_output then
      malformed io.(0) (Printf.sprintf "data longer than %d bytes" dst_len);
    if status <> inflate_end then malformed io.(0) (inflate_message status);
    let size = io.(3) - start in
    let check =
      match framing with
      | Raw -> 0
      | Zlib -> adler32_bigbytes 1 dst start size
      | Gzip -> crc32_bigbytes 0 dst start size
    in
    trailer framing c ~check ~size;
    if framing = Gzip && io.(0) < src_len then begin
      inflate_reset state;
      stream ()
    end
  in
  match
    stream ();
    if io.(0) <> src_len then malformed io.(0) "data after the stream";
    if io.(3) <> dst_len then
      malformed src_len
        (Printf.sprintf "data is %d bytes, not %d" io.(3) dst_len)
  with
  | () -> Ok ()
  | exception Malformed (pos, msg) ->
      Result.Error (Printf.sprintf "%s at byte %d" msg pos)

(* Readers. Compressed input is copied into [input]; [base] is the position in
   [r] of its first byte. The output buffer holds the 32 KiB history that
   matches refer to, followed by the room for one slice. *)

let window = 32768
let input_length = 65536

type phase = Header | Data | Trailer | Done

let reads framing format ?pos ?(slice_length = Bytes.Slice.io_buffer_size) r =
  let input = Bytes.create input_length in
  let output = Bytes.create (window + slice_length) in
  let io = [| 0; 0; 0; 0; 0 |] in
  let base = ref (Bytes.Reader.pos r) in
  let pending = ref Bytes.Slice.eod in
  let ended = ref false in
  let final () = !ended && Bytes.Slice.is_eod !pending in
  (* Moves the unread input to the front and appends to it. Returns [false] if
     no byte could be added. *)
  let refill () =
    let unread = io.(1) - io.(0) in
    Bytes.blit input io.(0) input 0 unread;
    base := !base + io.(0);
    io.(0) <- 0;
    io.(1) <- unread;
    let rec fill () =
      if io.(1) < input_length && not (final ()) then begin
        if Bytes.Slice.is_eod !pending then begin
          pending := Bytes.Reader.read r;
          if Bytes.Slice.is_eod !pending then ended := true
        end;
        let s = !pending in
        let n = Int.min (Bytes.Slice.length s) (input_length - io.(1)) in
        Bytes.blit (Bytes.Slice.bytes s) (Bytes.Slice.first s) input io.(1) n;
        io.(1) <- io.(1) + n;
        pending := Bytes.Slice.drop_first_or_eod n s;
        fill ()
      end
    in
    fill ();
    io.(1) > unread
  in
  let c =
    let next () =
      if io.(0) = io.(1) && not (refill ()) then
        malformed (!base + io.(0)) "truncated data";
      let b = Bytes.get_uint8 input io.(0) in
      io.(0) <- io.(0) + 1;
      b
    in
    { next; pos = (fun () -> !base + io.(0)) }
  in
  let state = inflate_create () in
  let phase = ref Header in
  let check = ref (initial_check framing) in
  let size = ref 0 in
  let update first length =
    size := !size + length;
    match framing with
    | Raw -> ()
    | Zlib -> check := adler32_bytes !check output first length
    | Gzip -> check := crc32_bytes !check output first length
  in
  let rec read () =
    match !phase with
    | Done -> Bytes.Slice.eod
    | Header ->
        header framing c;
        io.(2) <- io.(3);
        check := initial_check framing;
        size := 0;
        phase := Data;
        read ()
    | Data ->
        if io.(3) > window then begin
          let shift = io.(3) - window in
          Bytes.blit output shift output 0 window;
          io.(2) <- Int.max 0 (io.(2) - shift);
          io.(3) <- window
        end;
        let first = io.(3) in
        io.(4) <- first + slice_length;
        let status = inflate_bytes state input output io (final ()) in
        if status = inflate_end then phase := Trailer
        else if status = inflate_input then ignore (refill ())
        else if status <> inflate_output then
          malformed (!base + io.(0)) (inflate_message status);
        let length = io.(3) - first in
        if length = 0 then read ()
        else begin
          update first length;
          Bytes.Slice.make output ~first ~length
        end
    | Trailer ->
        trailer framing c ~check:!check ~size:!size;
        let more = io.(0) < io.(1) || refill () in
        if not more then phase := Done
        else if framing = Gzip then begin
          inflate_reset state;
          phase := Header
        end
        else malformed (!base + io.(0)) "data after the stream";
        read ()
  in
  let read () =
    try read ()
    with Malformed (pos, msg) ->
      phase := Done;
      Bytes.Reader.error format r ~pos msg
  in
  Bytes.Reader.make ?pos ~slice_length read

(* Writers *)

type deflate

external deflate_out_max : unit -> int = "caml_compress_deflate_out_max"
external deflate_create : int -> deflate = "caml_compress_deflate_create"

external deflate_free : deflate -> unit = "caml_compress_deflate_free"
[@@noalloc]

external deflate_input : deflate -> bytes -> int -> int -> int
  = "caml_compress_deflate_input"
[@@noalloc]

external deflate_encode : deflate -> bytes -> bool -> int
  = "caml_compress_deflate_encode"
[@@noalloc]

let out_max = deflate_out_max ()

let zlib_header_of_level level =
  let flevel =
    if level < 2 then 0x01
    else if level < 6 then 0x5E
    else if level = 6 then 0x9C
    else 0xDA
  in
  Printf.sprintf "\x78%c" (Char.chr flevel)

let gzip_header_bytes = "\x1F\x8B\x08\x00\x00\x00\x00\x00\x00\xFF"

let writes framing ~fn ?(level = 6) () =
  if level < 0 || level > 9 then
    invalid_arg (Printf.sprintf "%s: level %d is not in [0;9]" fn level);
  fun ?pos ?slice_length ~eod w ->
    let slice_length =
      match slice_length with
      | Some length -> length
      | None -> Bytes.Writer.slice_length w
    in
    let encoder = deflate_create level in
    let out = Bytes.create out_max in
    let check = ref (initial_check framing) in
    let size = ref 0 in
    let started = ref false in
    let emit n =
      let max = Bytes.Writer.slice_length w in
      let rec loop first =
        if first < n then begin
          let length = Int.min max (n - first) in
          Bytes.Writer.write w (Bytes.Slice.make out ~first ~length);
          loop (first + length)
        end
      in
      loop 0
    in
    let rec drain ~eod =
      let n = deflate_encode encoder out eod in
      if n > 0 then begin
        emit n;
        drain ~eod
      end
    in
    let rec push bytes first length =
      let n = deflate_input encoder bytes first length in
      drain ~eod:false;
      if n < length then push bytes (first + n) (length - n)
    in
    let write s =
      if not !started then begin
        started := true;
        match framing with
        | Raw -> ()
        | Zlib -> Bytes.Writer.write_string w (zlib_header_of_level level)
        | Gzip -> Bytes.Writer.write_string w gzip_header_bytes
      end;
      if Bytes.Slice.is_eod s then begin
        drain ~eod:true;
        deflate_free encoder;
        let trailer = Bytes.create 8 in
        (match framing with
        | Raw -> ()
        | Zlib ->
            Bytes.set_int32_be trailer 0 (Int32.of_int !check);
            Bytes.Writer.write w (Bytes.Slice.make trailer ~first:0 ~length:4)
        | Gzip ->
            Bytes.set_int32_le trailer 0 (Int32.of_int !check);
            Bytes.set_int32_le trailer 4 (Int32.of_int !size);
            Bytes.Writer.write w (Bytes.Slice.make trailer ~first:0 ~length:8));
        if eod then Bytes.Writer.write_eod w
      end
      else begin
        let bytes = Bytes.Slice.bytes s in
        let first = Bytes.Slice.first s in
        let length = Bytes.Slice.length s in
        (match framing with
        | Raw -> ()
        | Zlib -> check := adler32_bytes !check bytes first length
        | Gzip -> check := crc32_bytes !check bytes first length);
        size := !size + length;
        push bytes first length
      end
    in
    Bytes.Writer.make ?pos ~slice_length write

(* Formats *)

module Deflate = struct
  let format = format_error "deflate"
  let decompress_reads () = reads Raw format

  let compress_writes ?level () =
    writes Raw ~fn:"Compress_deflate.Deflate.compress_writes" ?level ()

  let decompress = decompress Raw ~fn:"Compress_deflate.Deflate.decompress"
end

module Zlib = struct
  let format = format_error "zlib"
  let decompress_reads () = reads Zlib format

  let compress_writes ?level () =
    writes Zlib ~fn:"Compress_deflate.Zlib.compress_writes" ?level ()

  let decompress = decompress Zlib ~fn:"Compress_deflate.Zlib.decompress"
end

module Gzip = struct
  let format = format_error "gzip"
  let decompress_reads () = reads Gzip format

  let compress_writes ?level () =
    writes Gzip ~fn:"Compress_deflate.Gzip.compress_writes" ?level ()

  let decompress = decompress Gzip ~fn:"Compress_deflate.Gzip.decompress"
end
