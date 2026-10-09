(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An agent for the bench: [bench_agent.exe] listens on the loopback at a port
   the system chooses, prints it, and serves one job of the key "k" x 32, whose
   kind "POLLED" is a Polled device that runs its own queue as work arrives, as
   a GPU runs beside its host. *)

let () =
  let key = Result.get_ok (Rig_remote.key (String.make 32 'k')) in
  match Rig_remote.listen ~key "127.0.0.1" 0 with
  | Error why ->
      prerr_endline why;
      exit 1
  | Ok a ->
      Printf.printf "%d\n%!" (Rig_remote.port a);
      let polled () =
        Ok [ fst (Rig_support.Polled.open_ ~runs:`Itself "POLLED:0") ]
      in
      exit
        (match Rig_remote.serve a [ ("POLLED", polled) ] with
        | Ok () -> 0
        | Error _ -> 2)
