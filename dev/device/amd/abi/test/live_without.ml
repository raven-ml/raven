(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The words a program holds live with the modules the AMD library links, but
   not the library: its baseline. *)

let () =
  ignore (Sys.opaque_identity Live_stdlib.linked);
  ignore (Sys.opaque_identity (Obj.repr Device_elf.of_string));
  Gc.full_major ();
  print_int (Gc.stat ()).live_words;
  print_newline ()
