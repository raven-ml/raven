(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A cubin NVRTC made (dev/device/elf/test/fixtures): its kernel and image. *)

open Windtrap
open Device_nv_abi

let cubin () =
  In_channel.with_open_bin "../../../elf/test/fixtures/simple_add_sm89.cubin"
    In_channel.input_all
  |> Cubin.of_string
  |> require_ok ~pp:Format.pp_print_string

let tests =
  group "simple_add_sm89"
    [
      test "its one kernel is simple_add" (fun () ->
          equal (list string) [ "simple_add" ] (Cubin.kernels (cubin ())));
      test "simple_add's registers, parameters and bank 0" (fun () ->
          let k = require_some (Cubin.kernel (cubin ()) "simple_add") in
          equal ~msg:"code bytes" int 512 k.code_bytes;
          equal ~msg:"registers" int 12 k.registers;
          equal ~msg:"params offset" int 0x160 k.params_offset;
          equal ~msg:"bank 0's bytes"
            (list (pair int int))
            [ (0, 380) ]
            (List.map (fun (b : Cubin.bank) -> (b.index, b.bytes)) k.banks));
      test "its image is its code's end rounded to 4 KiB, and 4 KiB more"
        (fun () ->
          let c = cubin () in
          let k = require_some (Cubin.kernel c "simple_add") in
          equal int
            (((k.code + k.code_bytes + 0xfff) land lnot 0xfff) + 0x1000)
            (Cubin.size c));
    ]

let () = exit (run "device_nv_abi.cubin" [ tests ])
