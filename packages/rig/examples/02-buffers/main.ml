(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Buffers on the host.

   A buffer is bytes of one device's memory. rig does not interpret the
   bytes, and the caller gives them their meaning. This example makes
   buffers on the host, reads them as elements of some kind, views part of
   one, wraps an OCaml bigarray, and copies between them. *)

open Rig

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

let show name b =
  Buffer.read b Bigarray.int32 (fun a ->
      Printf.printf "%-6s %2d bytes [%s]\n" name (Buffer.length b) (ints a))

let () =
  (* The host is a device. Its memory is the process's memory, and its work
     the process's own code. *)
  Printf.printf "%s computes: %b, runs on the host: %b\n\n" (name host)
    (computes host) (runs_on_host host);

  (* An owned buffer of 16 bytes, written as four 32-bit integers. Its contents
     start unspecified. *)
  let a = Buffer.create host 16 in
  Buffer.write a Bigarray.int32 (fun w ->
      for i = 0 to 3 do
        w.{i} <- Int32.of_int ((i + 1) * 10)
      done);
  show "a" a;

  (* A view is a range of the same memory, so writing through it writes
     [a]. *)
  let mid = Buffer.view a ~first:4 ~length:8 in
  Buffer.write mid Bigarray.int32 (fun w -> w.{0} <- 99l);
  show "mid" mid;
  show "a" a;
  Printf.printf "a spans its memory: %b, mid: %b, they overlap: %b\n\n"
    (Buffer.spans a) (Buffer.spans mid) (Buffer.overlaps a mid);

  (* A buffer wraps a bigarray the program already holds, without a copy.
     The same eight bytes read as a float and as an integer. *)
  let floats = Bigarray.(Array1.create float64 c_layout 1) in
  floats.{0} <- 1.0;
  let f = Buffer.of_bigarray floats in
  Printf.printf "f is borrowed: %b; 1.0 as int64 bits: 0x%Lx\n\n"
    (Buffer.is_borrowed f)
    (Buffer.read f Bigarray.int64 (fun a -> a.{0}));

  (* A copy moves bytes between buffers of one length. *)
  let b = Buffer.create host 8 in
  Buffer.copy ~src:mid ~dst:b;
  show "b" b;

  (* Misuse raises Invalid_argument, such as a copy between lengths that
     differ, or a read of bytes that are not a whole number of elements. *)
  let fails f =
    match f () with
    | () -> ()
    | exception Invalid_argument msg ->
        Printf.printf "Invalid_argument: %s\n" msg
  in
  fails (fun () -> Buffer.copy ~src:a ~dst:b);
  fails (fun () ->
      Buffer.read (Buffer.view a ~first:0 ~length:12) Bigarray.int64 ignore)
