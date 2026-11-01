(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A check on a device of another machine copies whether its flags hold to the
   host at issue, and its settling reads that copy in place. The agent is a
   real process on the loopback (Remote_job). *)

open Windtrap

exception Failed

(* The copies [events] record, as their source's and destination's names. *)
let copies events =
  List.filter_map
    (function
      | Rig.Profile.Copy { src; dst; _ } -> Some (Rig.name src, Rig.name dst)
      | _ -> None)
    events

let copy_w = list (pair string string)

let test_copied_at_issue () =
  Remote_job.with_job @@ fun j _ ->
  let d = List.hd (Remote_job.mem (List.hd (Rig_remote.hosts j))) in
  let module R = (val Nx.devices [ d ]) in
  let flag b = Nx.place R.on (Nx.create Nx.bool [||] [| b |]) in
  let first = flag true and second = flag true and failing = flag false in
  let issue ok = Nx.check Nx.Ptree.unit ok () (fun _ () -> Failed) in
  let one = [ (Rig.name d, Rig.name Rig.host) ] in
  let (), at_issue = Rig.Profile.take (fun () -> issue first) in
  equal copy_w ~msg:"at issue" one (copies at_issue);
  (* Issuing the second settles the first once its copy is reported done. *)
  let (), next = Rig.Profile.take (fun () -> issue second) in
  equal copy_w ~msg:"at the next issue" one (copies next);
  let (), at_wait = Rig.Profile.take (fun () -> Nx.sync second) in
  equal copy_w ~msg:"at the wait" [] (copies at_wait);
  issue failing;
  raises Failed (fun () -> Nx.sync failing)

let () =
  Watchdog.start ();
  exit
    (run "nx check on another machine"
       [
         test "a check copies its flags' result once, at issue"
           test_copied_at_issue;
       ])
