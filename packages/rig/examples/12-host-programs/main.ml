(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Host code.

   The host loads a C function compiled to a relocatable ELF object as an
   image, and runs it as a launch, as a GPU runs a kernel. The function is
   called as [void f(const void *params, int64_t first, int64_t end)] with
   the launch's parameters, which hold its buffers' host addresses, and a
   range of its threads. The host's cores run the groups of the grid in
   parallel. *)

open Rig

(* The object for this host's machine: [affine.c] compiled for each. *)
let object_file () =
  match arch host with
  | "arm64" -> "affine_aarch64.o"
  | "x86_64" -> "affine_x86_64.o"
  | a -> failwith ("no object for " ^ a)

(* out[i] = a * inp[i] + c for i below n, over [groups] groups of [threads]
   threads. *)
let affine fn ~groups:g ~threads:t ~n ~a ~c out inp =
  let refs = Submission.[| write 0 ~at:0; read 1 ~at:8 |] in
  let s = Submission.(make host [| launch ~refs fn ~params:40 |]) in
  let run = Submission.Run.make () and k = Submission.block s 0 in
  Submission.Run.(
    groups run k g 1 1;
    threads run k t 1 1;
    int64 run k 16 n;
    int64 run k 24 a;
    int64 run k 32 c);
  ignore (submit s ~run ~buffers:[| out; inp |])

let iota n =
  let b = Buffer.create host (8 * n) in
  Buffer.write b Bigarray.int64 (fun xs ->
      for i = 0 to n - 1 do
        xs.{i} <- Int64.of_int i
      done);
  b

let () =
  let obj = In_channel.with_open_bin (object_file ()) In_channel.input_all in
  let image = Result.get_ok (Image.load host obj) in
  let fn = Option.get (Image.resolve image "affine") in

  (* One group of 8 threads: out[i] = 3 * in[i] + 1. *)
  let n = 8 in
  let inp = iota n and out = Buffer.create host (8 * n) in
  affine fn ~groups:1 ~threads:n ~n ~a:3 ~c:1 out inp;
  Buffer.read out Bigarray.int64 (fun ys ->
      Printf.printf "out = [%s]\n"
        (String.concat "; " (List.init n (fun i -> Int64.to_string ys.{i}))));

  (* A million threads in 64 groups, which the host's cores run in
     parallel. *)
  let n = 1_000_000 in
  let inp = iota n and out = Buffer.create host (8 * n) in
  affine fn ~groups:64 ~threads:((n + 63) / 64) ~n ~a:2 ~c:(-1) out inp;
  let wrong =
    Buffer.read out Bigarray.int64 (fun ys ->
        let wrong = ref 0 in
        for i = 0 to n - 1 do
          if ys.{i} <> Int64.of_int ((2 * i) - 1) then incr wrong
        done;
        !wrong)
  in
  Printf.printf "%d threads in 64 groups: %d wrong\n" n wrong;

  (* An image's functions are the named symbols of its object's executable
     sections. *)
  if Option.is_none (Image.resolve image "missing") then
    print_endline "no function \"missing\""
