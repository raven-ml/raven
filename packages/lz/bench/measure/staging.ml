(* §7.4 and §7.5: moving bytes between a bigarray and [bytes], in ns per byte,
   at 4 KiB, 64 KiB and 1 MiB.

   §7.4 is a C memcpy each way, the staging a bytesrw filter over lz would pay
   per slice. §7.5 is what OCaml code pays today without such a blit:
   [Bytesrw.Bytes.Slice.of_bigbytes], and the per-byte loops of two raven
   readers, reproduced here with their types, against the C blit doing the same
   job. Every bigarray here is [(int, int8_unsigned_elt, c_layout)], the type
   both loops read. *)

open Bigarray

type ints = (int, int8_unsigned_elt, c_layout) Array1.t

external clock : unit -> int = "lzm_clock" [@@noalloc]

external ba_to_bytes : ints -> int -> bytes -> int -> int -> unit
  = "lzm_ba_to_bytes"
[@@noalloc]

external bytes_to_ba : bytes -> int -> ints -> int -> int -> unit
  = "lzm_bytes_to_ba"
[@@noalloc]

(* packages/nx/lib/io/zip_archive.ml:596-599, inside [give], which copies at
   most a 64 KiB chunk of an entry's data into a [chunk] made once per entry.
   The loop is that one, with [encoded.data] named [data]. *)
let zip_loop (data : ints) chunk first n =
  for i = 0 to n - 1 do
    Bytes.unsafe_set chunk i
      (Char.unsafe_chr (Array1.unsafe_get data (first + i)))
  done

(* contrib/ymir/lib/fits/tiles.ml:255-258, in [gunzip], which copies a tile's
   whole compressed bytes into a [bytes] made per tile. The code is those lines,
   [A] being [Bigarray.Array1]. *)
let tiles_loop (src : ints) =
  let input = Bytes.create (Array1.dim src) in
  for i = 0 to Array1.dim src - 1 do
    Bytes.unsafe_set input i (Char.unsafe_chr (Array1.unsafe_get src i))
  done;
  input

(* The same jobs with the C blit. *)
let tiles_blit (src : ints) =
  let input = Bytes.create (Array1.dim src) in
  ba_to_bytes src 0 input 0 (Array1.dim src);
  input

let rounds = 11
let round_ns = 20_000_000

(* ns per call of [f]: the median and the best of [rounds] rounds, each of as
   many calls as fill about [round_ns]. *)
let time f =
  let t0 = clock () in
  f ();
  let once = max 1 (clock () - t0) in
  let calls = max 1 (round_ns / once) in
  let r =
    Array.init rounds (fun _ ->
        let t0 = clock () in
        for _ = 1 to calls do
          f ()
        done;
        float (clock () - t0) /. float calls)
  in
  Array.sort compare r;
  (r.(rounds / 2), r.(0))

let row size label f =
  let med, best = time f in
  let n = float size in
  Printf.printf "| %d | %s | %.4f | %.4f | %.1f |\n%!" size label (med /. n)
    (best /. n) (n /. med)

let check label a b = if a <> b then failwith (label ^ ": wrong bytes")
let sizes = [ 4096; 65536; 1048576 ]

let () =
  Printf.printf
    "| bytes | operation | ns/byte (median) | ns/byte (best) | GB/s (median) |\n";
  Printf.printf "|---|---|---|---|---|\n";
  List.iter
    (fun size ->
      let ba =
        Array1.init int8_unsigned c_layout size (fun i -> i * 7 land 255)
      in
      let back = Array1.create int8_unsigned c_layout size in
      let b = Bytes.create size in
      let expect = Bytes.init size (fun i -> Char.chr (i * 7 land 255)) in
      (* Check each operation once before timing it. *)
      ba_to_bytes ba 0 b 0 size;
      check "ba_to_bytes" b expect;
      bytes_to_ba b 0 back 0 size;
      check "bytes_to_ba" (Bytes.init size (fun i -> Char.chr back.{i})) expect;
      let z = Bytes.make size '\000' in
      zip_loop ba z 0 size;
      check "zip_loop" z expect;
      check "tiles_loop" (tiles_loop ba) expect;
      check "of_bigbytes"
        (Bytes.of_string
           (Bytesrw.Bytes.Slice.to_string (Bytesrw.Bytes.Slice.of_bigbytes ba)))
        expect;
      row size "§7.4 C memcpy bigarray to bytes" (fun () ->
          ba_to_bytes ba 0 b 0 size);
      row size "§7.4 C memcpy bytes to bigarray" (fun () ->
          bytes_to_ba b 0 back 0 size);
      row size "§7.4 C memcpy there and back" (fun () ->
          ba_to_bytes ba 0 b 0 size;
          bytes_to_ba b 0 back 0 size);
      row size "§7.5 zip_archive loop into a reused chunk" (fun () ->
          zip_loop ba b 0 size);
      row size "§7.5 tiles loop into a new bytes" (fun () ->
          ignore (Sys.opaque_identity (tiles_loop ba)));
      row size "§7.5 C blit into a new bytes (tiles' job)" (fun () ->
          ignore (Sys.opaque_identity (tiles_blit ba)));
      row size "§7.5 Bytesrw.Bytes.Slice.of_bigbytes" (fun () ->
          ignore (Sys.opaque_identity (Bytesrw.Bytes.Slice.of_bigbytes ba))))
    sizes
