(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [high.exe PROG ARG...] runs PROG with ARGs holding [past] descriptors that
   its programs inherit, its limit of descriptors raised to [limit] if lower:
   every descriptor PROG and the processes it starts open has a number past
   [past], beyond select's FD_SETSIZE of 1024. It prints why and exits 1 if
   the system's limit is below [limit]. *)

let past = 1100
let limit = 4096

let () =
  if Rig_support.descriptor_limit () < limit then begin
    let e = Rig_support.set_descriptor_limit limit in
    if e <> 0 then begin
      Printf.printf "the system limits descriptors below %d: %d\n" limit e;
      exit 1
    end
  end;
  for _ = 1 to past do
    ignore (Unix.dup Unix.stdin)
  done;
  let args = Array.sub Sys.argv 1 (Array.length Sys.argv - 1) in
  Unix.execvp args.(0) args
