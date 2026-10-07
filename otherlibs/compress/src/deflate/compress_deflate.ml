(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type level = int

external init : unit -> unit = "caml_compress_deflate_init"

let () = init ()

external overlap : bigbytes -> bigbytes -> bool
  = "caml_compress_deflate_overlap"
[@@noalloc]

let check_range fn b first length =
  if first < 0 || length < 0 || first > Bytes.length b - length then
    invalid_arg
      (Printf.sprintf "%s: range %d+%d is not in a buffer of %d bytes" fn first
         length (Bytes.length b))

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

  let string ?(crc = 0) ?(first = 0) ?length s =
    let b = Bytes.unsafe_of_string s in
    let length = Option.value length ~default:(String.length s - first) in
    check_range "Compress_deflate.Crc32.string" b first length;
    crc32_bytes crc b first length
end

(* Framings. Headers and trailers are parsers that take their bytes one at a
   time, with each byte's position, so that a decoder suspends them wherever its
   input runs out. Malformed data raises [Malformed] at the position where
   decoding failed. *)

exception Malformed of int * string

type framing = Raw | Zlib | Gzip
type parser = Done | Byte of (int -> int -> parser)

let module_name = function
  | Raw -> "Compress_deflate.Deflate"
  | Zlib -> "Compress_deflate.Zlib"
  | Gzip -> "Compress_deflate.Gzip"

let malformed pos msg = raise (Malformed (pos, msg))
let initial_check = function Zlib -> 1 | Raw | Gzip -> 0

let zlib_header =
  Byte
    (fun at cmf ->
      Byte
        (fun _ flg ->
          if
            cmf land 0x0F <> 8
            || cmf lsr 4 > 7
            || (cmf lsl 8) lor flg mod 31 <> 0
          then malformed at "invalid zlib header";
          if flg land 0x20 <> 0 then malformed at "preset dictionary needed";
          Done))

let gzip_header () =
  let crc = ref 0 in
  let byte k =
    Byte
      (fun pos b ->
        crc := crc32_byte !crc b;
        k pos b)
  in
  let rec skip n k = if n = 0 then k () else byte (fun _ _ -> skip (n - 1) k) in
  let rec zero_terminated k =
    byte (fun _ b -> if b = 0 then k () else zero_terminated k)
  in
  byte @@ fun at id1 ->
  byte @@ fun _ id2 ->
  if id1 <> 0x1F || id2 <> 0x8B then malformed at "not a gzip member";
  byte @@ fun _ cm ->
  if cm <> 8 then malformed at "unknown compression method";
  byte @@ fun _ flags ->
  if flags land 0xE0 <> 0 then malformed at "reserved flags set";
  let extra k =
    if flags land 0x04 = 0 then k ()
    else byte (fun _ lo -> byte (fun _ hi -> skip (lo lor (hi lsl 8)) k))
  in
  let text flag k = if flags land flag = 0 then k () else zero_terminated k in
  skip 6 @@ fun () ->
  extra @@ fun () ->
  text 0x08 @@ fun () ->
  text 0x10 @@ fun () ->
  if flags land 0x02 = 0 then Done
  else
    let expected = !crc land 0xFFFF in
    Byte
      (fun at lo ->
        Byte
          (fun _ hi ->
            if lo lor (hi lsl 8) <> expected then
              malformed at "header checksum mismatch";
            Done))

let header = function
  | Raw -> Done
  | Zlib -> zlib_header
  | Gzip -> gzip_header ()

(* [word ~big k] reads a 32-bit word, big-endian iff [big], and continues with
   [k] applied to the position of its first byte and its value. *)
let word ~big k =
  Byte
    (fun at b0 ->
      Byte
        (fun _ b1 ->
          Byte
            (fun _ b2 ->
              Byte
                (fun _ b3 ->
                  k at
                    (if big then
                       (b0 lsl 24) lor (b1 lsl 16) lor (b2 lsl 8) lor b3
                     else b0 lor (b1 lsl 8) lor (b2 lsl 16) lor (b3 lsl 24))))))

(* [trailer framing ~check ~size] reads the trailer of a stream whose data has
   checksum [check] and [size] bytes. *)
let trailer framing ~check ~size =
  match framing with
  | Raw -> Done
  | Zlib ->
      word ~big:true @@ fun at adler ->
      if adler <> check then malformed at "Adler-32 mismatch";
      Done
  | Gzip ->
      word ~big:false @@ fun at crc ->
      if crc <> check then malformed at "CRC-32 mismatch";
      word ~big:false @@ fun at length ->
      if length <> size land 0xFFFFFFFF then malformed at "length mismatch";
      Done

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

let error_message pos msg = Printf.sprintf "%s at byte %d" msg pos

(* Between byte arrays *)

let decompress_into framing src dst =
  if overlap src dst then
    invalid_arg (module_name framing ^ ".decompress_into: src and dst overlap");
  let src_len = Bigarray.Array1.dim src in
  let dst_len = Bigarray.Array1.dim dst in
  let io = [| 0; src_len; 0; 0; dst_len |] in
  let rec parse = function
    | Done -> ()
    | Byte k ->
        let p = io.(0) in
        if p >= src_len then malformed p "truncated data";
        io.(0) <- p + 1;
        parse (k p (Bigarray.Array1.unsafe_get src p))
  in
  let state = inflate_create () in
  let rec stream () =
    parse (header framing);
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
    parse (trailer framing ~check ~size);
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
  | exception Malformed (pos, msg) -> Error (error_message pos msg)

(* Encoders and decoders take their input from a [pending], which holds what
   they were given and have not taken yet. *)

type pending = {
  mutable bytes : bytes;
  mutable first : int;
  mutable length : int;
  mutable ended : bool;
  mutable awaiting : bool;
}

let pending () =
  { bytes = Bytes.empty; first = 0; length = 0; ended = false; awaiting = true }

let give fn p s first length =
  check_range fn s first length;
  if not p.awaiting then invalid_arg (fn ^ ": input is not awaited");
  p.awaiting <- false;
  if length = 0 then p.ended <- true
  else begin
    p.bytes <- s;
    p.first <- first;
    p.length <- length
  end

let take p n =
  p.first <- p.first + n;
  p.length <- p.length - n

let await p =
  p.awaiting <- true;
  `Await

(* Decoders. The pending input is copied into [input] as room frees up; [base]
   is the stream position of [input]'s first byte. [output] holds the 32 KiB
   history that matches refer to, followed by the room for one slice. *)

let window = 32768
let input_length = 65536
let slice_length = 65536

module Decoder = struct
  type phase =
    | Header of parser
    | Data
    | Trailer of parser
    | Next
    | Ended
    | Failed of string

  type t = {
    framing : framing;
    inflate : inflate;
    pending : pending;
    input : bytes;
    output : bytes;
    io : int array;
    mutable base : int;
    mutable phase : phase;
    mutable check : int;
    mutable size : int;
  }

  let make framing =
    {
      framing;
      inflate = inflate_create ();
      pending = pending ();
      input = Bytes.create input_length;
      output = Bytes.create (window + slice_length);
      io = [| 0; 0; 0; 0; 0 |];
      base = 0;
      phase = Header (header framing);
      check = 0;
      size = 0;
    }

  let src d = give "Compress_deflate.Decoder.src" d.pending

  (* Moves the unread input to the front of [input] and appends pending input to
     it. Returns [false] if no byte could be added. *)
  let refill d =
    let io = d.io and p = d.pending in
    let unread = io.(1) - io.(0) in
    Bytes.blit d.input io.(0) d.input 0 unread;
    d.base <- d.base + io.(0);
    io.(0) <- 0;
    let n = Int.min p.length (input_length - unread) in
    Bytes.blit p.bytes p.first d.input unread n;
    take p n;
    io.(1) <- unread + n;
    n > 0

  let pos d = d.base + d.io.(0)
  let available d = d.io.(0) < d.io.(1) || refill d

  (* [feed d p] runs [p] on the input. It is [None] once [p] is done and [Some
     p'] if [p'] needs input not given yet. *)
  let rec feed d = function
    | Done -> None
    | Byte k as p ->
        if available d then begin
          let at = pos d in
          let b = Bytes.get_uint8 d.input d.io.(0) in
          d.io.(0) <- d.io.(0) + 1;
          feed d (k at b)
        end
        else if d.pending.ended then malformed (pos d) "truncated data"
        else Some p

  let rec step d =
    let io = d.io in
    match d.phase with
    | Ended -> `End
    | Failed msg -> `Error msg
    | Header p -> (
        match feed d p with
        | Some p ->
            d.phase <- Header p;
            await d.pending
        | None ->
            io.(2) <- io.(3);
            d.check <- initial_check d.framing;
            d.size <- 0;
            d.phase <- Data;
            step d)
    | Data ->
        if io.(3) > window then begin
          let shift = io.(3) - window in
          Bytes.blit d.output shift d.output 0 window;
          io.(2) <- Int.max 0 (io.(2) - shift);
          io.(3) <- window
        end;
        let first = io.(3) in
        io.(4) <- first + slice_length;
        let status =
          inflate_bytes d.inflate d.input d.output io d.pending.ended
        in
        if status > inflate_output then
          malformed (pos d) (inflate_message status);
        let length = io.(3) - first in
        d.size <- d.size + length;
        (match d.framing with
        | Raw -> ()
        | Zlib -> d.check <- adler32_bytes d.check d.output first length
        | Gzip -> d.check <- crc32_bytes d.check d.output first length);
        if status = inflate_end then
          d.phase <- Trailer (trailer d.framing ~check:d.check ~size:d.size);
        if length > 0 then `Data (d.output, first, length)
        else if status <> inflate_input || refill d then step d
        else await d.pending
    | Trailer p -> (
        match feed d p with
        | Some p ->
            d.phase <- Trailer p;
            await d.pending
        | None ->
            d.phase <- Next;
            step d)
    | Next ->
        if available d then begin
          if d.framing <> Gzip then malformed (pos d) "data after the stream";
          inflate_reset d.inflate;
          d.phase <- Header (header Gzip);
          step d
        end
        else if d.pending.ended then begin
          d.phase <- Ended;
          `End
        end
        else await d.pending

  let decode d =
    try step d
    with Malformed (pos, msg) ->
      let msg = error_message pos msg in
      d.phase <- Failed msg;
      `Error msg
end

(* Encoders. The C encoder copies the pending input as it has room for it and
   writes at most one block to [output] per call. *)

type deflate

external deflate_out_max : unit -> int = "caml_compress_deflate_out_max"
external deflate_create : int -> deflate = "caml_compress_deflate_create"

external deflate_free : deflate -> unit = "caml_compress_deflate_free"
[@@noalloc]

external deflate_reset : deflate -> unit = "caml_compress_deflate_reset"
[@@noalloc]

external deflate_input : deflate -> bytes -> int -> int -> int
  = "caml_compress_deflate_input"
[@@noalloc]

external deflate_encode : deflate -> bytes -> bool -> int
  = "caml_compress_deflate_encode"
[@@noalloc]

let out_max = deflate_out_max ()

module Encoder = struct
  type phase = Header | Data | Trailer | Ended

  type t = {
    framing : framing;
    level : level;
    deflate : deflate;
    pending : pending;
    output : bytes;
    keep : bool;  (** whether [deflate] outlives the stream *)
    mutable phase : phase;
    mutable check : int;
    mutable size : int;
  }

  let check_level framing level =
    if level < 0 || level > 9 then
      invalid_arg
        (Printf.sprintf "%s: level %d is not in [0;9]" (module_name framing)
           level)

  let start framing level ~keep deflate output =
    {
      framing;
      level;
      deflate;
      pending = pending ();
      output;
      keep;
      phase = Header;
      check = initial_check framing;
      size = 0;
    }

  let make framing ?(level = 6) () =
    check_level framing level;
    start framing level ~keep:false (deflate_create level)
      (Bytes.create out_max)

  let src e s first length =
    give "Compress_deflate.Encoder.src" e.pending s first length;
    (match e.framing with
    | Raw -> ()
    | Zlib -> e.check <- adler32_bytes e.check s first length
    | Gzip -> e.check <- crc32_bytes e.check s first length);
    e.size <- e.size + length

  let zlib_flevel level =
    if level < 2 then 0x01
    else if level < 6 then 0x5E
    else if level = 6 then 0x9C
    else 0xDA

  let rec encode e =
    let p = e.pending in
    match e.phase with
    | Ended -> `End
    | Header -> (
        e.phase <- Data;
        match e.framing with
        | Raw -> encode e
        | Zlib ->
            Bytes.set_uint8 e.output 0 0x78;
            Bytes.set_uint8 e.output 1 (zlib_flevel e.level);
            `Data (e.output, 0, 2)
        | Gzip ->
            Bytes.blit_string "\x1F\x8B\x08\x00\x00\x00\x00\x00\x00\xFF" 0
              e.output 0 10;
            `Data (e.output, 0, 10))
    | Data ->
        if p.length > 0 then
          take p (deflate_input e.deflate p.bytes p.first p.length);
        let n = deflate_encode e.deflate e.output p.ended in
        if n > 0 then `Data (e.output, 0, n)
        else if p.length > 0 then encode e
        else if p.ended then begin
          if not e.keep then deflate_free e.deflate;
          e.phase <- Trailer;
          encode e
        end
        else await p
    | Trailer -> (
        e.phase <- Ended;
        match e.framing with
        | Raw -> `End
        | Zlib ->
            Bytes.set_int32_be e.output 0 (Int32.of_int e.check);
            `Data (e.output, 0, 4)
        | Gzip ->
            Bytes.set_int32_le e.output 0 (Int32.of_int e.check);
            Bytes.set_int32_le e.output 4 (Int32.of_int e.size);
            `Data (e.output, 0, 8))
end

(* Whole buffers *)

(* [compress] keeps one encoder state and output buffer per domain and level,
   reset for each call: a state holds about 1 MiB, which a call on a small
   string would otherwise allocate and clear. A reset state writes the bytes of
   a fresh one. *)
let states = Domain.DLS.new_key (fun () -> Array.make 10 None)

let compress framing ?(level = 6) s =
  Encoder.check_level framing level;
  let cache = Domain.DLS.get states in
  let deflate, output =
    match cache.(level) with
    | Some (d, o) ->
        cache.(level) <- None;
        deflate_reset d;
        (d, o)
    | None -> (deflate_create level, Bytes.create out_max)
  in
  let e = Encoder.start framing level ~keep:true deflate output in
  let b = Buffer.create (64 + (String.length s / 2)) in
  let rec loop () =
    match Encoder.encode e with
    | `Data (bytes, first, length) ->
        Buffer.add_subbytes b bytes first length;
        loop ()
    | `Await ->
        Encoder.src e Bytes.empty 0 0;
        loop ()
    | `End ->
        cache.(level) <- Some (deflate, output);
        Buffer.contents b
  in
  Encoder.src e (Bytes.unsafe_of_string s) 0 (String.length s);
  loop ()

let decompress framing s =
  let d = Decoder.make framing in
  let b = Buffer.create (64 + (2 * String.length s)) in
  let rec loop () =
    match Decoder.decode d with
    | `Data (bytes, first, length) ->
        Buffer.add_subbytes b bytes first length;
        loop ()
    | `Await ->
        Decoder.src d Bytes.empty 0 0;
        loop ()
    | `End -> Ok (Buffer.contents b)
    | `Error msg -> Error msg
  in
  Decoder.src d (Bytes.unsafe_of_string s) 0 (String.length s);
  loop ()

(* Formats *)

module Deflate = struct
  let compress ?level s = compress Raw ?level s
  let decompress s = decompress Raw s
  let encoder ?level () = Encoder.make Raw ?level ()
  let decoder () = Decoder.make Raw
  let decompress_into src dst = decompress_into Raw src dst
end

module Zlib = struct
  let compress ?level s = compress Zlib ?level s
  let decompress s = decompress Zlib s
  let encoder ?level () = Encoder.make Zlib ?level ()
  let decoder () = Decoder.make Zlib
  let decompress_into src dst = decompress_into Zlib src dst
end

module Gzip = struct
  let compress ?level s = compress Gzip ?level s
  let decompress s = decompress Gzip s
  let encoder ?level () = Encoder.make Gzip ?level ()
  let decoder () = Decoder.make Gzip
  let decompress_into src dst = decompress_into Gzip src dst
end
