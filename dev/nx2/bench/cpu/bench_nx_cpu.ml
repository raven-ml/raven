(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's copies and casts, beside the floors that bound them: a copy of one
   element, the cost of a call, and loops that move each row's bytes on one
   thread, on the performance cores and on every core. A copy's floor is memcpy
   of its bytes; a cast's moves its elements' bytes in and out with an integer
   truncation or extension per element. A cast whose conversion can cost more
   than that has a second floor, the fastest loop of its codec.
   floor/move-4-2-1M-all moves 1 Mi elements of 4 bytes into 2 on every core,
   floor/copy-4M-1t copies 4 MiB on one thread, floor/codec-f32-bf16-1M-1t
   encodes 1 Mi bfloat16 on one thread. *)

module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external floor_move : int -> int -> int -> bytes -> bytes -> int -> unit
  = "nx_cpu_bench_floor_move_byte" "nx_cpu_bench_floor_move"

(* Conversions whose own operations can cost more than their bytes' move:
   those no instruction does, and float32 to float16 and int8, whose
   instructions cost more than an integer narrowing on the M1. *)
type codec =
  | F32_bf16
  | F32_e4m3
  | E4m3_f32
  | I64_f32
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
let ok = function
  | A.Done -> ()
  | _ -> failwith "nx.cpu refused a bench operand"
let kib = 1024
let mib = 1024 * kib

let count n =
  if n mod mib = 0 then strf "%dM" (n / mib)
  else if n mod kib = 0 then strf "%dK" (n / kib)
  else string_of_int n

(* The element counts of the size classes: 4 KiB, 256 KiB, 4 MiB and 64 MiB of
   float32, from L1 to memory. *)
let sizes = [ kib; 64 * kib; mib; 16 * mib ]

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

(* An array of [dt] and shape [s] holding stores of 251 values in [-46.25,
   46.25], as data a kernel meets. *)
let filled (type v s) (dt : (v, s) D.t) s : (v, s) A.t =
  let n = Array.fold_left ( * ) 1 s in
  let x i = float_of_int ((i * 7919 mod 251) - 125) *. 0.37 in
  let src = A.of_array D.Float64 s (Array.init n x) in
  let a = A.create Rig.host dt s in
  ok (Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst:a src);
  a

(* Floors: memcpy of [n] bytes, [n] elements of [inb] bytes moved into [outb],
   or [n] elements through a codec. *)
type floor =
  | Copy of int
  | Move of { inb : int; outb : int; n : int }
  | Codec of codec * int

(* A codec's source and destination, and their bytes per element. *)
let codec_pair = function
  | F32_bf16 -> ("f32", 4, "bf16", 2)
  | F32_e4m3 -> ("f32", 4, "e4m3", 1)
  | E4m3_f32 -> ("e4m3", 1, "f32", 4)
  | I64_f32 -> ("i64", 8, "f32", 4)
  | F64_f16 -> ("f64", 8, "f16", 2)
  | F32_f16 -> ("f32", 4, "f16", 2)
  | F32_i8 -> ("f32", 4, "i8", 1)

let floor_name = function
  | Copy n -> strf "copy-%s" (count n)
  | Move f -> strf "move-%d-%d-%s" f.inb f.outb (count f.n)
  | Codec (c, n) ->
      let s, _, d, _ = codec_pair c in
      strf "codec-%s-%s-%s" s d (count n)

let floors = ref []
let need f = if not (List.mem f !floors) then floors := !floors @ [ f ]

let copy ?floor name a =
  Option.iter need floor;
  row name
    (fun () ->
      let a = a () in
      (a, A.create Rig.host (A.dtype a) (A.Layout.shape (A.layout a))))
    (fun (a, dst) -> ok (Nx_cpu.apply1 Nx_kernel.Prog.Copy ~dst a))

(* A cast of [n] elements of [s] into [d]. Its floor moves a sub-byte side's
   pairs of elements as bytes. *)
let cast (type v s w r) (s : (v, s) D.t) (d : (w, r) D.t) n =
  let w dt = D.bits dt / 8 in
  need
    (if D.bits s < 8 then Move { inb = 1; outb = 2 * w d; n = n / 2 }
     else if D.bits d < 8 then Move { inb = 2 * w s; outb = 1; n = n / 2 }
     else Move { inb = w s; outb = w d; n });
  List.iter
    (fun c ->
      let cs, _, cd, _ = codec_pair c in
      if (cs, cd) = (short s, short d) then need (Codec (c, n)))
    [ F32_bf16; F32_e4m3; E4m3_f32; I64_f32; F64_f16; F32_f16; F32_i8 ];
  row
    (strf "cast-%s-%s-%s" (short s) (short d) (count n))
    (fun () -> (filled s [| n |], A.create Rig.host d [| n |]))
    (fun (a, dst) -> ok (Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst a))

let f32 = D.Float32

let copy_rows () =
  Thumper.group "copy"
    ([ copy "copy-f32-1" (fun () -> filled f32 [| 1 |]) ]
    @ List.map
        (fun n ->
          copy
            ~floor:(Copy (4 * n))
            ("copy-f32-" ^ count n)
            (fun () -> filled f32 [| n |]))
        sizes
    @ [
        copy
          ~floor:(Copy (4 * 512 * 512))
          "copy-transposed-512x512"
          (fun () ->
            Option.get
              (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 512; 512 |])));
        copy
          ~floor:(Copy (4 * 4096 * 4096))
          "copy-transposed-4096x4096"
          (fun () ->
            Option.get
              (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 4096; 4096 |])));
        (* Past the M1 Max's 48 MiB system cache and TLB reach, as 4096x4096
           is, with rows that are no power of two apart. *)
        copy
          ~floor:(Copy (4 * 4000 * 4000))
          "copy-transposed-4000x4000"
          (fun () ->
            Option.get
              (A.move (M.Permute [| 1; 0 |]) (filled f32 [| 4000; 4000 |])));
        (* The copy kernel alone on one thread, the transpose of 512x512
           float32 in cache: the vendors' one-thread transposes bound it. *)
        row "block-transposed-512x512-1t"
          (fun () ->
            let b () =
              let b =
                Rig.Buffer.bigarray Bigarray.int8_unsigned
                  (Rig.Buffer.create Rig.host (4 * 512 * 512))
              in
              for i = 0 to Bigarray.Array1.dim b - 1 do
                Bigarray.Array1.unsafe_set b i (i land 255)
              done;
              b
            in
            (b (), b ()))
          (fun (d, s) -> block_transposed d s 512);
        (* A row broadcast over rows. *)
        copy ~floor:(Copy (4 * mib)) "copy-broadcast-1024x1024-1024" (fun () ->
            Option.get
              (A.move (M.Broadcast [| 1024; 1024 |]) (filled f32 [| 1024 |])));
        (* The 2x2 windows of a pooling layer, the column window's axis before
           the row window's. *)
        copy
          ~floor:(Copy (4 * 32 * 16 * 26 * 26))
          "copy-2x2-windows-32x16x26x26"
          (fun () ->
            let a = filled f32 [| 32; 16; 26; 26 |] in
            let w axis = { M.axis; size = 2; step = 2; dilation = 1 } in
            let v = Option.get (A.move (M.Window [| w 2; w 3 |]) a) in
            Option.get (A.move (M.Permute [| 0; 1; 2; 3; 5; 4 |]) v));
        copy
          ~floor:(Copy (mib / 2))
          "copy-i4-1M"
          (fun () -> filled D.Int4 [| mib |]);
      ])

(* float32 to float16 and to int32 at every size, then one cast per class of
   pairs at 1 Mi elements: narrow floats encoded and decoded, floats to integers
   and back, widening and narrowing, a double to a narrow float, 64-bit integers
   to floats, sub-byte unpacking and packing, complex and boolean. *)
let cast_rows () =
  let m = mib in
  Thumper.group "cast"
    (List.concat_map
       (fun n -> [ cast f32 D.Float16 n; cast f32 D.Int32 n ])
       sizes
    @ [
        cast D.Float16 f32 m;
        cast f32 D.Bfloat16 m;
        cast D.Bfloat16 f32 m;
        cast f32 D.Float8_e4m3fn m;
        cast D.Float8_e4m3fn f32 m;
        cast D.Int32 f32 m;
        cast f32 D.Int8 m;
        cast f32 D.Uint8 m;
        cast D.Int8 f32 m;
        cast D.Int64 f32 m;
        cast D.Uint64 f32 m;
        cast f32 D.Float64 m;
        cast D.Float64 f32 m;
        cast D.Float64 D.Float16 m;
        cast D.Int32 D.Int64 m;
        cast D.Int64 D.Int32 m;
        cast D.Int4 D.Int8 m;
        cast D.Int8 D.Int4 m;
        cast D.Complex64 D.Complex128 m;
        cast f32 D.Bool m;
        cast D.Bool f32 m;
      ])

(* The floors the rows need, over bytes the setup wrote. *)
let floor_rows () =
  let buffer n =
    let b =
      Rig.Buffer.bigarray Bigarray.int8_unsigned
        (Rig.Buffer.create Rig.host (max 1 n))
    in
    Bigarray.Array1.fill b 1;
    b
  in
  let threads =
    [ ("1t", 1); ("performance", performance_cores ()); ("all", cores ()) ]
  in
  let run f threads =
    match f with
    | Copy n ->
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
      (function Codec _ -> codecs_run () | Copy _ | Move _ -> true)
      !floors
  in
  Thumper.group "floor"
    (List.concat_map
       (fun f ->
         List.map
           (fun (t, threads) ->
             let setup, body = run f threads in
             row (strf "%s-%s" (floor_name f) t) setup body)
           threads)
       floors)

let () =
  let copy_rows = copy_rows () in
  let cast_rows = cast_rows () in
  exit @@ Thumper.run "nx_cpu" [ copy_rows; cast_rows; floor_rows () ]
