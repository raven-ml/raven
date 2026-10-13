(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A process opens and closes devices without a bound on how many: a device
   that closed gives its slot to the next one, which continues the slot's
   timeline. A device lost with work queued gives its slot too, and keeps
   only what answers its lost points. *)

open Windtrap
module Sub = Rig.Submission
module Support = Rig_support
module P = Rig_support.Polled

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

(* More opens than the process has slots, 65,535: [warm] cycles, then
   [clean] cycles of memory devices that close, then [losses] cycles of
   Polled devices lost with work queued. *)
let warm = 10_000
let clean = 60_000
let losses = 1_000

(* A device that closed keeps nothing once its slot passes on: its line is
   static, its C record goes with its OCaml record, and the slot's next
   device takes its word. The heaps may grow by [slack] over the clean
   cycles, as macOS's count of the C heap steps by 16 KiB or more: a leak of
   two bytes per open would exceed it.

   A device lost with values unreached keeps what answers its points for
   good: its C record and the range of its slot that holds its lost values,
   and its OCaml record, which the range holds. [c_loss] and [ml_loss] bound
   each per loss, baselines measured on the M1 Max: about 1 KiB of C heap
   and 1.7 KiB of OCaml heap. *)
let slack = 64 * 1024
let c_loss = 2 * 1024
let ml_loss = 4 * 1024

let heaps () =
  Gc.full_major ();
  let c =
    match Support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  (c, (Gc.stat ()).live_words * (Sys.word_size / 8))

let lose run name =
  let d, h = P.open_ name in
  let p = Rig.submit (Sub.make d [||]) ~run ~buffers:[||] in
  P.fault h "lost";
  (match Rig.Buffer.create d 8 with
  | _ -> fail "an allocation of a faulted device returned"
  | exception Rig.Lost _ -> ());
  Rig.close d;
  (d, p)

let close_one run =
  let d = Rig.memory_device "limit:cycle" in
  let p = Rig.submit (Sub.make d [||]) ~run ~buffers:[||] in
  Rig.close d;
  (d, p)

let test_open_close () =
  let run = Sub.Run.make () in
  (* The first loss is the process's failure, which keeps its slot. *)
  ignore (lose run "limit:first");
  let first = close_one run in
  for _ = 2 to warm do
    ignore (close_one run)
  done;
  let c0, ml0 = heaps () in
  for _ = 1 to clean do
    ignore (close_one run)
  done;
  let c1, ml1 = heaps () in
  at_most int ~msg:"the C heap's growth over the clean cycles" ~than:slack
    (c1 - c0);
  at_most int ~msg:"the OCaml heap's growth over the clean cycles"
    ~than:slack (ml1 - ml0);
  let lost = lose run "limit:lost" in
  for _ = 2 to losses do
    ignore (lose run "limit:lost")
  done;
  let c2, ml2 = heaps () in
  at_most int ~msg:"the C heap's growth over the losses"
    ~than:(slack + (losses * c_loss)) (c2 - c1);
  at_most int ~msg:"the OCaml heap's growth over the losses"
    ~than:(slack + (losses * ml_loss)) (ml2 - ml1);
  let d, p = first in
  equal ~msg:"the first device's last value" int (Rig.Point.value p)
    (Rig.submitted d);
  equal ~msg:"its point, reached" bool true (Rig.Point.reached p);
  Rig.Point.wait p;
  equal ~msg:"its point's device" (option device) None (Rig.Point.device p);
  let d, p = lost in
  raises_match ~msg:"a lost point's wait"
    (function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false)
    (fun () -> Rig.Point.wait p);
  equal ~msg:"a lost point's device" (option device) None (Rig.Point.device p)

let tests =
  [
    test ~timeout:(Rig_support.time 60.)
      "71,000 opens and closes, some lost with work queued, succeed with \
       the heaps bounded"
      test_open_close;
  ]

let () = exit (run "rig.limit" tests)
