(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffers on the host.

   A buffer is bytes of one device's memory. What the bytes mean is the
   caller's: rig moves, orders and returns bytes. This example makes buffers on
   the host, reads them as elements of some kind, views part of one, borrows an
   OCaml bigarray, and copies between them. *)

open Rig

let int32s b = Buffer.bigarray Bigarray.int32 b

let show name b =
  let a = int32s b in
  let xs = List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}) in
  Printf.printf "%-6s %2d bytes [%s]\n" name (Buffer.length b)
    (String.concat "; " xs)

let () =
  (* The host is a device: its memory is the process's heap, its work the
     process's own code. *)
  Printf.printf "%s computes: %b, runs on the host: %b\n\n" (name host)
    (computes host) (runs_on_host host);

  (* An owned buffer of 16 bytes, written as four 32-bit integers. Its contents
     start unspecified. *)
  let a = Buffer.create host 16 in
  let ints = int32s a in
  for i = 0 to 3 do
    ints.{i} <- Int32.of_int ((i + 1) * 10)
  done;
  show "a" a;

  (* A view is a range of the same memory: writing through it writes [a]. *)
  let mid = Buffer.view a ~first:4 ~length:8 in
  (int32s mid).{0} <- 99l;
  show "mid" mid;
  show "a" a;
  Printf.printf "a spans its memory: %b, mid: %b, they overlap: %b\n\n"
    (Buffer.spans a) (Buffer.spans mid) (Buffer.overlaps a mid);

  (* A bigarray the program already holds is borrowed without a copy. The same
     eight bytes read as a float and as an integer. *)
  let floats = Bigarray.(Array1.create float64 c_layout 1) in
  floats.{0} <- 1.0;
  let f = Buffer.of_bigarray floats in
  Printf.printf "f is borrowed: %b; 1.0 as int64 bits: 0x%Lx\n\n"
    (Buffer.is_borrowed f)
    (Buffer.bigarray Bigarray.int64 f).{0};

  (* A copy moves bytes between buffers of one length. *)
  let b = Buffer.create host 8 in
  Buffer.copy ~src:mid ~dst:b;
  show "b" b;

  (* Misuse raises Invalid_argument: lengths that differ, bytes that are not a
     whole number of elements. *)
  let fails f =
    match f () with
    | () -> ()
    | exception Invalid_argument msg ->
        Printf.printf "Invalid_argument: %s\n" msg
  in
  fails (fun () -> Buffer.copy ~src:a ~dst:b);
  fails (fun () ->
      ignore
        (Buffer.bigarray Bigarray.int64 (Buffer.view a ~first:0 ~length:12)))
