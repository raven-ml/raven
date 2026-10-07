(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The reader on a real cubin and AMD code object, and a refusal. The tester
   owns the suite. *)

open Windtrap
module Elf = Device_elf

let root = "../../../.."

let read path =
  In_channel.with_open_bin (Filename.concat root path) In_channel.input_all

let load ?align obj =
  match Elf.of_string ?align obj with Ok o -> o | Error e -> failf "%s" e

let test_cubin () =
  let obj = read "packages/tolk/test/runtime/ops_nv/simple_add_sm89.cubin" in
  let o = load ~align:128 obj in
  equal ~msg:"constant bank, then code" int 896 (String.length o.image);
  equal ~msg:"the kernel's code" (option int) (Some 384)
    (Elf.symbol o "simple_add");
  equal ~msg:"debugging relocations are left out" int 0
    (List.length o.relocations);
  match Elf.of_string (String.sub obj 0 100) with
  | Ok _ -> fail "a truncated cubin is read"
  | Error _ -> ()

let test_code_object () =
  let o =
    load
      (read
         "packages/tolk/test/gen/runtime/ops_amd_fixtures/simple_add_gfx1100.hsaco")
  in
  equal ~msg:"sections at their addresses" int 0x1880 (String.length o.image);
  equal ~msg:"the kernel descriptor" (option int) (Some 0x5c0)
    (Elf.symbol o "simple_add.kd");
  equal ~msg:"its OS ABI, AMD HSA" int 64 o.os_abi;
  equal ~msg:"the symbol table" int 10 (Iarray.length o.symbols)

let test_refusal () =
  equal ~msg:"not an ELF object" (result unit string)
    (Error "not an ELF object")
    (Result.map ignore (Elf.of_string "not an object"))

let () =
  exit
  @@ run "device_elf"
       [
         test "a cubin" test_cubin;
         test "an AMD code object" test_code_object;
         test "a refusal" test_refusal;
       ]
