(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Host programs.

   A host program is a C function compiled to a relocatable ELF object, linked
   into executable memory and called as [void f(void **buffers, const int64_t
   *values)]. Its buffers here are rig's host buffers, passed by address. A
   split call cuts its iterations into blocks that the host's cores run at once,
   each with the bounds of its range in two of the values. *)

open Rig

let int64s b = Buffer.bigarray Bigarray.int64 b

(* The object for this host's machine: [affine.c] compiled for each. *)
let object_file () =
  match arch host with
  | "arm64" -> "affine_aarch64.o"
  | "x86_64" -> "affine_x86_64.o"
  | a -> failwith ("no object for " ^ a)

let () =
  let obj = In_channel.with_open_bin (object_file ()) In_channel.input_all in
  let p = Result.get_ok (Rig_host.link ~entry:"affine" obj) in

  (* One call over 8 iterations: out[i] = 3 * in[i] + 1. *)
  let n = 8 in
  let inp = Buffer.create host (8 * n) and out = Buffer.create host (8 * n) in
  for i = 0 to n - 1 do
    (int64s inp).{i} <- Int64.of_int i
  done;
  let buffers = [| Buffer.address out; Buffer.address inp |] in
  (* CR: Keep [inp] alive through each call with
     [ignore (Sys.opaque_identity inp)] after it. The address array holds
     only integers, so collection can free or recycle the input before the
     host program finishes. The split call's [xs] is also dead by then. *)
  Rig_host.call p buffers [| 0; n; 3; 1 |];
  Printf.printf "out = [%s]\n"
    (String.concat "; "
       (List.init n (fun i -> Int64.to_string (int64s out).{i})));

  (* A split call: a million iterations in 64 blocks, each call given its range
     in values 0 and 1. The blocks run at once on up to [Rig_host.workers ()]
     threads. *)
  let n = 1_000_000 in
  let inp = Buffer.create host (8 * n) and out = Buffer.create host (8 * n) in
  let xs = int64s inp and ys = int64s out in
  for i = 0 to n - 1 do
    xs.{i} <- Int64.of_int i
  done;
  let buffers = [| Buffer.address out; Buffer.address inp |] in
  let split = { Rig_host.extent = n; blocks = 64; lo = 0; hi = 1 } in
  Rig_host.call ~split p buffers [| 0; 0; 2; -1 |];
  let wrong = ref 0 in
  for i = 0 to n - 1 do
    if ys.{i} <> Int64.of_int ((2 * i) - 1) then incr wrong
  done;
  Printf.printf "split over %d iterations: %d wrong\n" n !wrong;

  (* An object without the entry is an [Error] saying so. *)
  match Rig_host.link ~entry:"missing" obj with
  | Ok _ -> ()
  | Error why -> print_endline why
