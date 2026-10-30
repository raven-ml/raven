(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype

type work =
  | Copy of int
  | Cast of D.any * D.any * int
  | Fma of D.any * int
  | Read of int
  | Stream of { ins : int; inb : int; outb : int; n : int }
  | Select of { inb : int; n : int }

type bytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external floor_move : int -> int -> int -> bytes -> bytes -> int -> unit
  = "nx_cpu_bench_floor_move_byte" "nx_cpu_bench_floor_move"

external floor_stream :
  int -> int -> int -> int -> bytes -> bytes -> bytes -> bytes -> int -> unit
  = "nx_cpu_bench_floor_stream_byte" "nx_cpu_bench_floor_stream"

external floor_fma : int -> bool -> int -> unit = "nx_cpu_bench_floor_fma"
external floor_read : int -> bytes -> unit = "nx_cpu_bench_floor_read"
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

(* Floors: memcpy of [n] bytes, [n] elements of [inb] bytes moved into [outb],
   [n] flops of fused multiply-adds, float64 ones if [f64], or a read of [n]
   bytes. *)
type floor =
  | Memcpy of int
  | Move of { inb : int; outb : int; n : int }
  | Peak of { f64 : bool; n : int }
  | Bytes_read of int
  | Streamed of { ins : int; inb : int; outb : int; n : int }
  | Selected of { inb : int; n : int }

let floor_name = function
  | Memcpy n -> strf "copy-%s" (count n)
  | Move f -> strf "move-%d-%d-%s" f.inb f.outb (count f.n)
  | Peak { f64; n } -> strf "fma-%s-%s" (if f64 then "f64" else "f32") (count n)
  | Bytes_read n -> strf "read-%s" (count n)
  | Streamed s -> strf "stream-%dx%d-%d-%s" s.ins s.inb s.outb (count s.n)
  | Selected s -> strf "select-%d-%s" s.inb (count s.n)

(* The flops of a peak's row: a rate, the same for every contraction, about
   10 ms on one core. A row's floor is its flops at that rate. *)
let peak_flops = 1024 * mib

(* A copy's floor is its memcpy. A cast's moves its elements' bytes, a sub-byte
   side's pairs of elements as bytes. *)
let floors = function
  | Copy n -> [ Memcpy n ]
  | Cast (D.Any s, D.Any d, n) ->
      let w dt = D.bits dt / 8 in
      if D.bits s < 8 then [ Move { inb = 1; outb = 2 * w d; n = n / 2 } ]
      else if D.bits d < 8 then [ Move { inb = 2 * w s; outb = 1; n = n / 2 } ]
      else [ Move { inb = w s; outb = w d; n } ]
  | Fma (dt, _) -> [ Peak { f64 = dt = D.Any D.Float64; n = peak_flops } ]
  | Read n -> [ Bytes_read n ]
  | Stream { ins; inb; outb; n } -> [ Streamed { ins; inb; outb; n } ]
  | Select { inb; n } -> [ Selected { inb; n } ]

(* The floors of [ws], each once, in the order the work asks for them. *)
let needed ws =
  List.fold_left
    (fun acc f -> if List.mem f acc then acc else acc @ [ f ])
    []
    (List.concat_map floors ws)

(* A buffer of [n] bytes the setup wrote. *)
let buffer n =
  let b =
    Rig.Buffer.export
      (Rig.Buffer.create Rig.host (max 1 n))
      Bigarray.int8_unsigned
  in
  Bigarray.Array1.fill b 1;
  b

let rows ws =
  let threads =
    [ ("1t", 1); ("performance", performance_cores ()); ("all", cores ()) ]
  in
  let run name f threads =
    let pair setup body = row name setup body in
    match f with
    | Memcpy n ->
        pair
          (fun () -> (buffer n, buffer n))
          (fun (d, s) -> floor_move threads 0 0 d s n)
    | Move { inb; outb; n } ->
        pair
          (fun () -> (buffer (outb * n), buffer (inb * n)))
          (fun (d, s) -> floor_move threads inb outb d s n)
    | Peak { f64; n } -> pair (fun () -> ()) (fun () -> floor_fma threads f64 n)
    | Bytes_read n -> pair (fun () -> buffer n) (fun s -> floor_read threads s)
    | Streamed { ins; inb; outb; n } ->
        let src k = buffer (if k < ins then inb * n else 1) in
        pair
          (fun () -> (buffer (outb * n), src 0, src 1, src 2))
          (fun (d, a, b, c) -> floor_stream threads ins inb outb d a b c n)
    | Selected { inb; n } ->
        pair
          (fun () -> (buffer (inb * n), buffer n, buffer (inb * n), buffer (inb * n)))
          (fun (d, c, a, b) -> floor_stream threads (-1) inb inb d c a b n)
  in
  List.concat_map
    (fun f ->
      List.map
        (fun (t, threads) ->
          run (strf "floor-%s-%s" (floor_name f) t) f threads)
        threads)
    (needed ws)
