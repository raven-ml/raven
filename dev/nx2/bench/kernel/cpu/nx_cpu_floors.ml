(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

type work = Copy of int | Cast of D.any * D.any * int

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external floor_move : int -> int -> int -> bytes -> bytes -> int -> unit
  = "nx_cpu_bench_floor_move_byte" "nx_cpu_bench_floor_move"

(* Conversions whose own operations can cost more than their bytes' move: those
   no instruction does, and float32 to float16 and int8, whose instructions cost
   more than an integer narrowing on the M1. *)
type codec =
  | F32_bf16
  | F32_e4m3
  | E4m3_f32
  | I64_f32
  | U64_f32
  | F64_f16
  | F32_f16
  | F32_i8

external floor_codec : int -> codec -> bytes -> bytes -> int -> unit
  = "nx_cpu_bench_floor_codec"

external codecs_run : unit -> bool = "nx_cpu_bench_codecs_run" [@@noalloc]

external block_transposed : bytes -> bytes -> int -> unit
  = "nx_cpu_bench_block_transposed"
[@@noalloc]

external cores : unit -> int = "nx_cpu_bench_cores" [@@noalloc]

external performance_cores : unit -> int = "nx_cpu_bench_performance_cores"
[@@noalloc]

let strf = Printf.sprintf
let row name setup f = Thumper.bench_with_setup ~setup name f
let kib = 1024
let mib = 1024 * kib

let count n =
  if n mod mib = 0 then strf "%dM" (n / mib)
  else if n mod kib = 0 then strf "%dK" (n / kib)
  else string_of_int n

let short (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float64 -> "f64"
  | D.Float32 -> "f32"
  | D.Float16 -> "f16"
  | D.Bfloat16 -> "bf16"
  | D.Float8_e4m3fn -> "e4m3"
  | D.Float8_e5m2 -> "e5m2"
  | D.Float4_e2m1fn -> "e2m1"
  | D.Int64 -> "i64"
  | D.Uint64 -> "u64"
  | D.Int32 -> "i32"
  | D.Uint32 -> "u32"
  | D.Int16 -> "i16"
  | D.Uint16 -> "u16"
  | D.Int8 -> "i8"
  | D.Uint8 -> "u8"
  | D.Int4 -> "i4"
  | D.Uint4 -> "u4"
  | D.Complex128 -> "c128"
  | D.Complex64 -> "c64"
  | D.Bool -> "bool"
  | D.Bit -> "bit"

(* Floors: memcpy of [n] bytes, [n] elements of [inb] bytes moved into [outb],
   or [n] elements through a codec. *)
type floor =
  | Memcpy of int
  | Move of { inb : int; outb : int; n : int }
  | Codec of codec * int

(* A codec's source and destination, and their bytes per element. *)
let codec_pair = function
  | F32_bf16 -> ("f32", 4, "bf16", 2)
  | F32_e4m3 -> ("f32", 4, "e4m3", 1)
  | E4m3_f32 -> ("e4m3", 1, "f32", 4)
  | I64_f32 -> ("i64", 8, "f32", 4)
  | U64_f32 -> ("u64", 8, "f32", 4)
  | F64_f16 -> ("f64", 8, "f16", 2)
  | F32_f16 -> ("f32", 4, "f16", 2)
  | F32_i8 -> ("f32", 4, "i8", 1)

let codecs =
  [ F32_bf16; F32_e4m3; E4m3_f32; I64_f32; U64_f32; F64_f16; F32_f16; F32_i8 ]

let floor_name = function
  | Memcpy n -> strf "copy-%s" (count n)
  | Move f -> strf "move-%d-%d-%s" f.inb f.outb (count f.n)
  | Codec (c, n) ->
      let s, _, d, _ = codec_pair c in
      strf "codec-%s-%s-%s" s d (count n)

(* A copy's floor is its memcpy. A cast's moves its elements' bytes, a sub-byte
   side's pairs of elements as bytes, and a codec's loop where one converts its
   pair. *)
let floors = function
  | Copy n -> [ Memcpy n ]
  | Cast (D.Any s, D.Any d, n) ->
      let w dt = D.bits dt / 8 in
      let move =
        if D.bits s < 8 then Move { inb = 1; outb = 2 * w d; n = n / 2 }
        else if D.bits d < 8 then Move { inb = 2 * w s; outb = 1; n = n / 2 }
        else Move { inb = w s; outb = w d; n }
      in
      let codec c =
        let cs, _, cd, _ = codec_pair c in
        if (cs, cd) = (short s, short d) then Some (Codec (c, n)) else None
      in
      move :: List.filter_map codec codecs

(* The floors of [ws], each once, in the order the work asks for them. *)
let needed ws =
  List.fold_left
    (fun acc f -> if List.mem f acc then acc else acc @ [ f ])
    []
    (List.concat_map floors ws)

(* A buffer of [n] bytes the setup wrote. *)
let buffer n =
  let b =
    Rig.Buffer.bigarray Bigarray.int8_unsigned
      (Rig.Buffer.create Rig.host (max 1 n))
  in
  Bigarray.Array1.fill b 1;
  b

let floor_rows ws =
  let threads =
    [ ("1t", 1); ("performance", performance_cores ()); ("all", cores ()) ]
  in
  let run f threads =
    match f with
    | Memcpy n ->
        ( (fun () -> (buffer n, buffer n)),
          fun (d, s) -> floor_move threads 0 0 d s n )
    | Move { inb; outb; n } ->
        ( (fun () -> (buffer (outb * n), buffer (inb * n))),
          fun (d, s) -> floor_move threads inb outb d s n )
    | Codec (c, n) ->
        let _, inb, _, outb = codec_pair c in
        ( (fun () -> (buffer (outb * n), buffer (inb * n))),
          fun (d, s) -> floor_codec threads c d s n )
  in
  let floors =
    List.filter
      (function Codec _ -> codecs_run () | Memcpy _ | Move _ -> true)
      (needed ws)
  in
  List.concat_map
    (fun f ->
      List.map
        (fun (t, threads) ->
          let setup, body = run f threads in
          row (strf "floor-%s-%s" (floor_name f) t) setup body)
        threads)
    floors

let block_transposed_row =
  row "block-transposed-512x512-1t"
    (fun () ->
      let b () =
        let b = buffer (4 * 512 * 512) in
        for i = 0 to Bigarray.Array1.dim b - 1 do
          Bigarray.Array1.unsafe_set b i (i land 255)
        done;
        b
      in
      (b (), b ()))
    (fun (d, s) -> block_transposed d s 512)

let rows ws = floor_rows ws @ [ block_transposed_row ]
