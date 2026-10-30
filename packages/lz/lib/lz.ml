open Bigarray

type bigbytes = (int, int8_unsigned_elt, c_layout) Array1.t

type why =
  | Truncated
  | Malformed of string
  | Too_long
  | Unsupported of string

type error = { at : int; why : why }

let pp_error ppf { at; why } =
  match why with
  | Truncated -> Format.fprintf ppf "truncated at byte %d" at
  | Too_long -> Format.fprintf ppf "output too long at byte %d" at
  | Malformed m -> Format.fprintf ppf "malformed at byte %d: %s" at m
  | Unsupported m -> Format.fprintf ppf "unsupported at byte %d: %s" at m

type progress = { consumed : int; produced : int }
type step = Done of int | More of int

let check_level name ~lo ~hi level =
  if level < lo || level > hi then
    invalid_arg
      (Printf.sprintf "Lz.%s: level %d is not in [%d, %d]" name level lo hi)

(* [bound_of name f n] is [f n], a stub's bound, which is -1 past
   max_int. *)
let bound_of name f n =
  if n < 0 then invalid_arg (Printf.sprintf "Lz.%s.bound: %d is negative" name n);
  let b = f n in
  if b < 0 then
    invalid_arg
      (Printf.sprintf "Lz.%s.bound: the bound of %d bytes passes max_int" name n);
  b

(* Strings staged through bigarrays, copied by memcpy. *)

external blit_in : string -> bigbytes -> unit = "lz_blit_in" [@@noalloc]
external blit_out : bigbytes -> bytes -> int -> unit = "lz_blit_out" [@@noalloc]

let of_string s =
  let b = Array1.create int8_unsigned c_layout (String.length s) in
  blit_in s b;
  b

let to_string b n =
  let s = Bytes.create n in
  blit_out b s n;
  Bytes.unsafe_to_string s

(* [stage ~bound ~encode s] is [encode] of [s] into a destination of
   [bound] bytes, as a string. *)
let stage ~bound ~encode s =
  let src = of_string s in
  let dst = Array1.create int8_unsigned c_layout (bound (String.length s)) in
  match encode src dst with
  | Some n -> to_string dst n
  | None -> invalid_arg "Lz: bound does not hold the stream"

(* Streams *)

(* A kernel's state, a custom block of stubs.c, and one step's outcome: the
   input bytes taken, the run at [start] of [len] bytes in the codec's
   buffer, and whether the stream ended. A decoder step also returns its
   buffer, which it replaces when a header fixes a larger window. *)
type state
type outcome = { taken : int; start : int; len : int; fin : bool }
type lent = { run : outcome; buf : bigbytes }

external dec_step :
  state -> bool -> bigbytes -> bigbytes -> (lent, error) result
  = "lz_dec_step"

external enc_step : state -> bool -> bigbytes -> bigbytes -> outcome
  = "lz_enc_step"

external run_size : state -> int = "lz_run_size"

(* A decoder or encoder: its state, the buffer its runs are lent from, and
   who holds it. A decoder's buffer is empty until its first step asks
   for one; held here, it is counted by the collector, and a run kept
   past [f] keeps it alive. A step's entry takes [busy] and its return
   gives it back, so that a step from another domain or thread, or from
   [f], finds it taken. Only the holder reads or writes [buf] and
   [dead]. *)
type stream = {
  st : state;
  mutable buf : bigbytes;
  busy : bool Atomic.t;
  mutable dead : bool;
}

let stream st =
  let buf = Array1.create int8_unsigned c_layout (run_size st) in
  { st; buf; busy = Atomic.make false; dead = false }

let enter name s =
  if not (Atomic.compare_and_set s.busy false true) then
    invalid_arg (name ^ ": in use by another step");
  if s.dead then (
    Atomic.set s.busy false;
    invalid_arg (name ^ ": dead after an error or a raise"))

let leave s = Atomic.set s.busy false

let kill s =
  s.dead <- true;
  leave s

(* [lend s o f] applies [f] to the run of [o]; a raise kills [s]. *)
let lend s o f =
  match f (Array1.sub s.buf o.start o.len) with
  | () ->
      leave s;
      if o.fin then Done o.taken else More o.taken
  | exception e ->
      kill s;
      raise e

module Decoder = struct
  type t = stream

  let step d ~last src f =
    enter "Lz.Decoder.step" d;
    match dec_step d.st last src d.buf with
    | Ok { run; buf } ->
        d.buf <- buf;
        Ok (lend d run f)
    | Error _ as e ->
        kill d;
        e
    | exception e ->
        kill d;
        raise e
end

module Encoder = struct
  type t = stream

  let step e ~last src f =
    enter "Lz.Encoder.step" e;
    match enc_step e.st last src e.buf with
    | o -> lend e o f
    | exception x ->
        kill e;
        raise x
end

(* Deflate, zlib and gzip *)

(* The wrapper, as deflate.h's enum lz_wrap numbers it. *)
let raw = 0
let zlib = 1
let gzip = 2

external deflate_decode : int -> bigbytes -> bigbytes -> (progress, error) result
  = "lz_deflate_decode_whole"

external deflate_encode : int -> int -> bigbytes -> bigbytes -> int option
  = "lz_deflate_encode_whole"

external deflate_bound : int -> int -> int = "lz_deflate_bound_stub"
external deflate_decoder : int -> state = "lz_deflate_decoder"
external deflate_encoder : int -> int -> state = "lz_deflate_encoder"

module Deflate_family (W : sig
  val name : string
  val wrap : int
end) =
struct
  let level_of ?(level = 6) () =
    check_level W.name ~lo:1 ~hi:9 level;
    level

  let decode src dst = deflate_decode W.wrap src dst

  let bound n = bound_of W.name (deflate_bound W.wrap) n

  let encode ?level src dst = deflate_encode W.wrap (level_of ?level ()) src dst

  let encode_string ?level s =
    stage ~bound ~encode:(deflate_encode W.wrap (level_of ?level ())) s

  let decoder () = stream (deflate_decoder W.wrap)
  let encoder ?level () = stream (deflate_encoder W.wrap (level_of ?level ()))
end

module Deflate = Deflate_family (struct
  let name = "Deflate"
  let wrap = raw
end)

module Zlib = Deflate_family (struct
  let name = "Zlib"
  let wrap = zlib
end)

module Gzip = Deflate_family (struct
  let name = "Gzip"
  let wrap = gzip
end)

(* Zstandard *)

external zstd_decode : bigbytes -> bigbytes -> (progress, error) result
  = "lz_zstd_decode_whole"

external zstd_encode : int -> bigbytes -> bigbytes -> int option
  = "lz_zstd_encode_whole"

external zstd_bound : int -> int = "lz_zstd_bound_stub"
external zstd_decoder : int -> state = "lz_zstd_decoder"
external zstd_encoder : int -> state = "lz_zstd_encoder"

module Zstd = struct
  let level_of ?(level = 3) () =
    check_level "Zstd" ~lo:1 ~hi:12 level;
    level

  let decode = zstd_decode

  let bound n = bound_of "Zstd" zstd_bound n

  let encode ?level src dst = zstd_encode (level_of ?level ()) src dst

  let encode_string ?level s =
    stage ~bound ~encode:(zstd_encode (level_of ?level ())) s

  (* The largest window the reference decoder accepts by default, and the
     smallest a frame declares (RFC 8878, 3.1.1.1.2). *)
  let default_window = 128 lsl 20
  let min_window = 1 lsl 10

  let decoder ?(window = default_window) () =
    if window < min_window then
      invalid_arg
        (Printf.sprintf "Lz.Zstd.decoder: window %d is below %d" window
           min_window);
    stream (zstd_decoder window)

  let encoder ?level () = stream (zstd_encoder (level_of ?level ()))
end

(* LZ4 *)

module Lz4 = struct
  module Block = struct
    external decode : bigbytes -> bigbytes -> (progress, error) result
      = "lz_lz4_block_decode_stub"

    external encode : bigbytes -> bigbytes -> int option
      = "lz_lz4_block_encode_stub"

    external bound : int -> int = "lz_lz4_block_bound_stub"

    let bound n = bound_of "Lz4.Block" bound n

    let encode_string s = stage ~bound ~encode s
  end

  module Frame = struct
    external decode : bigbytes -> bigbytes -> (progress, error) result
      = "lz_lz4_frame_decode_stub"

    external encode : bigbytes -> bigbytes -> int option
      = "lz_lz4_frame_encode_stub"

    external bound : int -> int = "lz_lz4_frame_bound_stub"

    let bound n = bound_of "Lz4.Frame" bound n

    let encode_string s = stage ~bound ~encode s
  end
end

(* Snappy *)

module Snappy = struct
  external decode : bigbytes -> bigbytes -> (progress, error) result
    = "lz_snappy_decode_stub"

  external encode : bigbytes -> bigbytes -> int option = "lz_snappy_encode_stub"
  external bound : int -> int = "lz_snappy_bound_stub"

  (* The largest length a preamble states (format_description.txt, 1). *)
  let max_length = 0xffff_ffff

  let too_long name n =
    if n > max_length then
      invalid_arg
        (Printf.sprintf "Lz.Snappy.%s: %d bytes exceed a block's %d" name n
           max_length)

  let bound n =
    bound_of "Snappy"
      (fun n ->
        too_long "bound" n;
        bound n)
      n

  let encode src dst =
    too_long "encode" (Array1.dim src);
    encode src dst

  let encode_string s =
    too_long "encode_string" (String.length s);
    stage ~bound ~encode s
end

(* Checksums *)

(* From this many bytes a checksum of a bigarray releases the runtime lock:
   the smallest power of two whose call, at the fastest checksum's vector
   speed, lasts at least 100 times a release and reacquisition with the
   systhreads library linked and a second domain running. Below it the
   [noalloc] entry runs: it neither allocates nor releases. *)
let release_bytes = 1 lsl 20

let range name s first length =
  let n = String.length s in
  let length = match length with Some l -> l | None -> n - first in
  if first < 0 || length < 0 || first > n - length then
    invalid_arg (Printf.sprintf "Lz.%s.string: range out of bounds" name);
  length

module Crc32 = struct
  external small :
    (int[@untagged]) ->
    bigbytes ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) = "lz_crc32_ba_byte" "lz_crc32_ba"
  [@@noalloc]

  external long : int -> bigbytes -> int -> int -> int = "lz_crc32_ba_long"

  external str :
    (int[@untagged]) ->
    string ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) = "lz_crc32_str_byte" "lz_crc32_str"
  [@@noalloc]

  let check name crc =
    if crc < 0 || crc > 0xffff_ffff then
      invalid_arg
        (Printf.sprintf "Lz.Crc32.%s: crc %d is not in [0, 2^32)" name crc)

  let bigbytes ?(crc = 0) b =
    check "bigbytes" crc;
    let n = Array1.dim b in
    if n >= release_bytes then long crc b 0 n else small crc b 0 n

  let string ?(crc = 0) ?(first = 0) ?length s =
    check "string" crc;
    str crc s first (range "Crc32" s first length)
end

module Adler32 = struct
  external small :
    (int32[@unboxed]) ->
    bigbytes ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int32[@unboxed]) = "lz_adler32_ba_byte" "lz_adler32_ba"
  [@@noalloc]

  external long : int32 -> bigbytes -> int -> int -> int32
    = "lz_adler32_ba_long"

  external str :
    (int32[@unboxed]) ->
    string ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int32[@unboxed]) = "lz_adler32_str_byte" "lz_adler32_str"
  [@@noalloc]

  let bigbytes ?(adler = 1l) b =
    let n = Array1.dim b in
    if n >= release_bytes then long adler b 0 n else small adler b 0 n

  let string ?(adler = 1l) ?(first = 0) ?length s =
    str adler s first (range "Adler32" s first length)
end

module Xxh32 = struct
  external small :
    (int32[@unboxed]) ->
    bigbytes ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int32[@unboxed]) = "lz_xxh32_ba_byte" "lz_xxh32_ba"
  [@@noalloc]

  external long : int32 -> bigbytes -> int -> int -> int32
    = "lz_xxh32_ba_long"

  external str :
    (int32[@unboxed]) ->
    string ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int32[@unboxed]) = "lz_xxh32_str_byte" "lz_xxh32_str"
  [@@noalloc]

  let bigbytes ?(seed = 0l) b =
    let n = Array1.dim b in
    if n >= release_bytes then long seed b 0 n else small seed b 0 n

  let string ?(seed = 0l) ?(first = 0) ?length s =
    str seed s first (range "Xxh32" s first length)
end

module Xxh64 = struct
  external small :
    (int64[@unboxed]) ->
    bigbytes ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int64[@unboxed]) = "lz_xxh64_ba_byte" "lz_xxh64_ba"
  [@@noalloc]

  external long : int64 -> bigbytes -> int -> int -> int64
    = "lz_xxh64_ba_long"

  external str :
    (int64[@unboxed]) ->
    string ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int64[@unboxed]) = "lz_xxh64_str_byte" "lz_xxh64_str"
  [@@noalloc]

  let bigbytes ?(seed = 0L) b =
    let n = Array1.dim b in
    if n >= release_bytes then long seed b 0 n else small seed b 0 n

  let string ?(seed = 0L) ?(first = 0) ?length s =
    str seed s first (range "Xxh64" s first length)
end

(* Paths *)

external checksum_path_names : int -> string array = "lz_checksum_path_names"

external checksum_path : int -> int -> int32 -> bigbytes -> int32
  = "lz_checksum_path"

external zstd_dec_path_names : unit -> string array = "lz_zstd_dec_path_names"

external zstd_decode_path :
  int -> bigbytes -> bigbytes -> (progress, error) result
  = "lz_zstd_decode_path"

external zstd_enc_path_names : unit -> string array = "lz_zstd_enc_path_names"

external zstd_encode_path : int -> int -> bigbytes -> bigbytes -> int option
  = "lz_zstd_encode_path"

external deflate_enc_path_names : unit -> string array
  = "lz_deflate_enc_path_names"

external deflate_encode_path : int -> int -> bigbytes -> bigbytes -> int option
  = "lz_deflate_encode_path"

module Paths = struct
  (* [named names f] pairs path [i]'s name with [f i]. *)
  let named names f = Array.to_list (Array.mapi (fun i n -> (n, f i)) names)

  (* The kind of checksum, as stubs.c numbers it. *)
  let crc32 = named (checksum_path_names 0) (checksum_path 0)
  let adler32 = named (checksum_path_names 1) (checksum_path 1)
  let zstd_decode = named (zstd_dec_path_names ()) zstd_decode_path

  let zstd_encode =
    named (zstd_enc_path_names ()) (fun i ~level ->
        zstd_encode_path i (Zstd.level_of ~level ()))

  let deflate_encode =
    named (deflate_enc_path_names ()) (fun i ~level ->
        deflate_encode_path i (Deflate.level_of ~level ()))
end

external zstd_encode_limit : int -> int -> bigbytes -> bigbytes -> int option
  = "lz_zstd_encode_limit"

external zstd_encoder_limit : int -> int -> state = "lz_zstd_encoder_limit"

let check_limit name limit =
  if limit < 0 || limit > 0xffff_ffff then
    invalid_arg
      (Printf.sprintf "Lz.%s: limit %d is not in [0, 2^32)" name limit)

let zstd_encode_limit ~limit ~level src dst =
  check_limit "zstd_encode_limit" limit;
  zstd_encode_limit limit (Zstd.level_of ~level ()) src dst

let zstd_encoder_limit ~limit ~level =
  check_limit "zstd_encoder_limit" limit;
  stream (zstd_encoder_limit limit (Zstd.level_of ~level ()))

external releases : unit -> int = "lz_releases" [@@noalloc]
