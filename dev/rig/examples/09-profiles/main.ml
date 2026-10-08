(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Profiles.

   A profile holds what every device did while a function ran: spans of work on
   each device's lanes, copies, allocations, and the loads and counters of GPU
   work. Every time is on the host clock. This example prints the events without
   their times, which differ from run to run, and writes them in Chrome's trace
   format, which Perfetto opens. *)

open Rig

let mib = 1024 * 1024

let show = function
  | Profile.Span { device; lane; name; _ } ->
      Printf.printf "span        %s %s: %s\n" (Rig.name device) lane name
  | Copy { src; dst; bytes; _ } ->
      Printf.printf "copy        %s -> %s: %d MiB\n" (Rig.name src)
        (Rig.name dst) (bytes / mib)
  | Allocation { device; allocated; _ } ->
      Printf.printf "allocation  %s: %d MiB\n" (Rig.name device)
        (allocated / mib)
  | Load _ | Counters _ | Trace _ | Overwritten _ -> ()

let () =
  let d = Result.get_ok (memory_device "M") in
  let work () =
    let src = Buffer.create host (4 * mib) in
    let dst = Profile.span "allocate" (fun () -> Buffer.create d (4 * mib)) in
    Profile.span "upload" (fun () -> Buffer.copy ~src ~dst);
    Printf.printf "profiling: %b\n" (Profile.enabled ())
  in
  let (), events = Profile.take work in
  Printf.printf "profiling: %b\n\n" (Profile.enabled ());
  List.iter show events;

  let oc = open_out "profile.json" in
  Profile.output_chrome_trace oc events;
  close_out oc;
  Printf.printf "\n%d events written to profile.json\n" (List.length events)
