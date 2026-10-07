(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reading the objects a program load reads: the largest AMD code object the
   library carries (126 KB, every unary kernel at float16), a cubin as the NV
   loader reads it, and a host object of six kernels. Then finding a kernel's
   descriptor in that code object, as the AMD loader does for each kernel it
   loads: the last one, behind every other symbol. *)

module Elf = Device_elf

(* Copies of the repository's objects: unary.float16.co of
   packages/nx/lib/amd/kernels/gfx12-generic/, which `uv run
   packages/nx/lib/amd/kernels/gen.py` compiles from src/ there, and
   simple_add_sm89.cubin of packages/tolk/test/runtime/ops_nv/, NVRTC 12.8's
   sm_89 code of simple_add.cu there. host.c says how host_arm64.o was
   compiled. *)

let read path = In_channel.with_open_bin path In_channel.input_all
let code_object = read "unary.float16.co"
let cubin = read "simple_add_sm89.cubin"
let host = read "host_arm64.o"

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
             Thumper.bench "amd-unary-float16" (of_string code_object);
             Thumper.bench "cubin-sm89" (of_string ~align:128 cubin);
             Thumper.bench "host-arm64" (of_string host);
           ];
         Thumper.group "symbol"
           [
             Thumper.bench "amd-unary-float16-last-kernel" (fun () ->
                 Elf.symbol o kernel);
           ];
       ])
