(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The words a program holds live with the AMD library linked, after a call to
   each of its modules' tables. *)

open Device_amd_abi

let () =
  ignore (Sys.opaque_identity Live_stdlib.linked);
  ignore (Sys.opaque_identity (Obj.repr Device_elf.of_string));
  ignore (Sys.opaque_identity Capability.key);
  ignore (Sys.opaque_identity (Packet.encode Int64.of_int [ Dword 0 ]));
  ignore (Sys.opaque_identity (Code_object.of_string ""));
  List.iter
    (fun gc ->
      let g =
        {
          Gpu.target = gc;
          gc;
          sdma = (6, 0, 0);
          xccs = 1;
          shader_engines = 1;
          compute_units = 1;
          scratch_slots = 1;
        }
      in
      ignore (Sys.opaque_identity (Gpu.processor g));
      ignore (Sys.opaque_identity (Register.registers g));
      ignore (Sys.opaque_identity (Scratch.tmpring g 0));
      ignore (Sys.opaque_identity (Scratch.descriptor g ~base:0 0));
      ignore (Sys.opaque_identity (Pm4.run g []));
      ignore (Sys.opaque_identity (Sdma.copy g ~dst:0 ~src:0 1));
      ignore (Sys.opaque_identity (Aql.indirect_buffer 0 ~dwords:0));
      ignore (Sys.opaque_identity (Thread_trace.start g ~size:4096 Fun.id));
      ignore
        (Sys.opaque_identity (Thread_trace.waves g (String.make 64 '\001'))))
    [ (9, 4, 3); (11, 0, 0); (12, 0, 0) ];
  Gc.full_major ();
  print_int (Gc.stat ()).live_words;
  print_newline ()
