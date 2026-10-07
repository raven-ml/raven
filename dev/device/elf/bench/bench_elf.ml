(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading the objects a program load reads: an AMD code object of 128 kernels
   (139 KB, its symbol table stripped, as a library ships its kernels), a cubin
   as the NV loader reads it, and a host object of six kernels. Then finding a
   kernel's descriptor in that code object, as the AMD loader does for each
   kernel it loads: the last one, behind every other symbol. *)

module Elf = Device_elf

(* The suite's fixtures. ../test/fixtures/README.md says how each is made. *)

let read path =
  In_channel.with_open_bin
    (Filename.concat "../test/fixtures" path)
    In_channel.input_all

let code_object = read "amd_many_gfx1100.hsaco"
let cubin = read "simple_add_sm89.cubin"
let host = read "host_aarch64.o"

let of_string ?align obj () =
  match Elf.of_string ?align obj with Ok o -> o | Error e -> failwith e

let last_kernel (o : Elf.t) =
  let last = ref "" in
  Iarray.iter
    (fun (s : Elf.symbol) ->
      match s.place with
      | Image _ when String.ends_with ~suffix:".kd" s.name -> last := s.name
      | Image _ | Undefined | Absolute _ | Outside _ -> ())
    o.symbols;
  !last

let () =
  let o = of_string code_object () in
  let kernel = last_kernel o in
  exit
    (Thumper.run "device_elf"
       [
         Thumper.group "of-string"
           [
             Thumper.bench "amd-128-kernels" (of_string code_object);
             Thumper.bench "cubin-sm89" (of_string ~align:128 cubin);
             Thumper.bench "host-aarch64" (of_string host);
           ];
         Thumper.group "symbol"
           [
             Thumper.bench "amd-128-kernels-last-kernel" (fun () ->
                 Elf.symbol o kernel);
           ];
       ])
