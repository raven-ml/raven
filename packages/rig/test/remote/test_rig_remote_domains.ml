(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls of rig.remote's controller from two domains, in a suite of its own:
   a process that spawned a domain can no longer fork. *)

open Windtrap
open Remote_job

(* Two domains behind a barrier make the first devices of each kind: both
   give the same devices. *)
let devices_racing () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let race kind =
    let waiting = Atomic.make 2 in
    let first () =
      Atomic.decr waiting;
      while Atomic.get waiting > 0 do
        Domain.cpu_relax ()
      done;
      Rig_remote.devices h kind
    in
    let other = Domain.spawn first in
    let mine = first () in
    match (mine, Domain.join other) with
    | Ok ds, Ok ds' ->
        equal ~msg:(kind ^ "'s count") int (List.length ds) (List.length ds');
        equal ~msg:kind (list bool)
          (List.map (fun _ -> true) ds)
          (List.map2 Rig.equal ds ds')
    | Error why, _ | _, Error why -> failf "%s: %s" kind why
  in
  List.iter race [ "MEM"; "POLLED"; "TIGHT"; "HIDDEN"; "MAPLESS" ]

let () =
  Watchdog.start ();
  exit
    (run "rig_remote domains"
       [
         group ~timeout:60. "two domains"
           [
             test "two first devices of a kind are the same devices"
               devices_racing;
           ];
       ])
