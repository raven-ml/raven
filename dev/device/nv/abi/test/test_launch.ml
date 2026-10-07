(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch rules: the classes a launch takes, its shared memory and threads. *)

open Windtrap
open Device_nv_abi

let gpu compute_class =
  {
    Gpu.compute_class;
    sass_version = 0x89;
    gpcs = 12;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window = 0x7294_0000_0000;
    local_window = 0x7293_0000_0000;
  }

let kernel ?(registers = 32) ?(shared_bytes = 0) () =
  {
    Cubin.code = 0x80;
    code_bytes = 0x100;
    registers;
    shared_bytes;
    stack_bytes = 0x20;
    params_offset = 0;
    banks = [];
  }

let launch ?(cls = 0xc9c0) k =
  require_ok ~pp:Format.pp_print_string (Launch.make (gpu cls) k)

let tests =
  group "make"
    [
      test "a class the library does not know is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Launch.make") (fun () ->
              Launch.make (gpu 0xcbc0) (kernel ())));
      test "shared memory up to 100 KiB, the driver's 1 KiB included" (fun () ->
          let limit = (100 * 1024) - 1024 in
          ignore (launch (kernel ~shared_bytes:limit ()));
          is_error
            (Launch.make (gpu 0xc9c0) (kernel ~shared_bytes:(limit + 1) ())));
      test "a block takes 1024 threads, fewer for many registers" (fun () ->
          equal ~msg:"16 registers" int 1024
            (Launch.max_threads (launch (kernel ~registers:16 ())));
          equal ~msg:"128 registers" int 512
            (Launch.max_threads (launch (kernel ~registers:128 ()))));
    ]

let () = exit (run "device_nv_abi.launch" [ tests ])
