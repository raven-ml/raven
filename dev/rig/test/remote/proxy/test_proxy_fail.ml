(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A proxy lost while its job goes on: Rig.fail loses every device of the
   process for good, so this suite has a process of its own. *)

open Windtrap
open Proxy_machine

(* With a copy into this process held at the agent, the process fails. The
   proxy's stop may write its last value only once no work of it runs
   (Rig.Driver.stop): the word stays below the copy's value until the agent
   reports it, the copy's bytes land before that, and the report fails
   nothing. *)
let stopped_in_flight () =
  with_machine @@ fun m ->
  let n = 4096 in
  let far = far_of_string m.host (String.make n 'f') in
  let here = host_buffer (String.make n '.') in
  pause m.ag;
  let v =
    Rig.Point.value (submit (copy_submission m.host ~src:far ~dst:here))
  in
  Rig.fail "the process fails";
  Thread.delay 0.2;
  let early = Rig.signaled m.host in
  resume m.ag;
  until ~what:"the agent's report" (fun () ->
      List.exists
        (function Handover { value; _ } -> value = v | _ -> false)
        (events m.ag));
  Thread.delay 0.2;
  less ~msg:"the word while the agent held the copy" int ~than:v early;
  equal ~msg:"the job, after the agent's report" (option string) None
    (Link.failure m.ag.job);
  equal ~msg:"the copy's bytes" string (String.make n 'f') (read_host here)

let () =
  Watchdog.start ();
  exit
    (run "rig_remote_proxy.proxy_fail"
       [
         group ~timeout:60. "proxy"
           [
             xfail ~reason:"stop writes the last value handed over at once"
               (test "a proxy lost with work in flight stops after the work"
                  stopped_in_flight);
           ];
       ])
