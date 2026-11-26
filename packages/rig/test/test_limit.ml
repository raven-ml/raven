(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The process's 65,535 device slots (rig.mli, "Opening and closing"): opens
   past every slot, an open while every slot is held, what a device keeps once
   its slot passed on, and the first failure, which holds its slot for good.
   Each needs the process's slots as a whole, so the suite runs in a process of
   its own. *)

open Windtrap
module Sub = Rig.Submission
module Support = Rig_support
module P = Rig_support.Polled

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let opened r = require_ok ~pp:Format.pp_print_string r
let submit s = Rig.submit s ~buffers:[||] ~args:[||]
let empty d = Sub.make d [||]

(* rig.mli: "A device holds a slot of the process's 65,535". *)
let slots = 65_535

(* A collection, then a drain of every device's memory, which {!Rig.free_cache}
   of the host runs: memory collected returns, and the slots of closed devices
   whose memory it was pass on. *)
let released () =
  Gc.full_major ();
  Rig.free_cache Rig.host

(* A Polled device lost with work queued, then closed, and the point of that
   work. *)
let lose name =
  let d, h = P.open_ name in
  let p = submit (empty d) in
  P.fault h "lost";
  (match Rig.Buffer.create d 8 with
  | _ -> fail "an allocation of a faulted device returned"
  | exception Rig.Lost _ -> ());
  Rig.close d;
  (d, p)

(* A software device that ran one submission, then closed, and the point of that
   submission. *)
let close_one () =
  let d = opened (Rig.software "limit:cycle") in
  let p = submit (empty d) in
  Rig.close d;
  (d, p)

(* [in_child child] runs [child] in a forked child, which holds what it opened
   until it ends, and is the lines it returned, with how it ended. The child
   writes them to a pipe and leaves without running this process's exit. *)
let in_child child =
  Support.needs_fork ();
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try child () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
      Support.reaping pid @@ fun () ->
      let lines = In_channel.input_lines ic in
      (lines, snd (Unix.waitpid [] pid))

let answer = function Ok d -> "Ok " ^ Rig.name d | Error _ -> "Error"

(* The first failure, made in a child, is a software device lost with every
   value reached, so that its slot is held for that reason alone. The child then
   opens software devices until every slot is held: none takes the first
   failure's slot, which its point shows by naming it. *)
let test_first_failure () =
  if Rig.failure () <> None then
    skip ~reason:"a device of this process failed before this test" ();
  let lines, ended =
    in_child (fun () ->
        let d = opened (Rig.software "limit:first-failure") in
        let p = submit (empty d) in
        Rig.Point.wait p;
        Rig.lose d "the process's first failure";
        Rig.close d;
        released ();
        let rec fill n =
          match Rig.software (Printf.sprintf "limit:after-failure-%d" n) with
          | Ok _ when n <= slots -> fill (n + 1)
          | r -> answer r
        in
        let refused = fill 1 in
        let named =
          match Rig.Point.device p with
          | Some d' when Rig.equal d d' -> "the first failure"
          | Some d' -> Rig.name d'
          | None -> "none"
        in
        [ refused; named ])
  in
  equal
    ~msg:
      "the child: an open once every slot is held, and the device the first \
       failure's point names"
    (pair (list string) bool)
    ([ "Error"; "the first failure" ], true)
    (lines, ended = Unix.WEXITED 0)

(* More opens than the process has slots, each of a device that closes: the
   first device's point answers as reached and names no device once its slot
   passed on. A device lost with work queued leaves its slot too, and its lost
   point raises its loss with it. *)
let opens = 71_000

let test_past_every_slot () =
  (* The process's first failure holds its slot: one comes first. *)
  if Rig.failure () = None then ignore (lose "limit:first-loss");
  let first = close_one () in
  let lost = lose "limit:lost-queued" in
  for _ = 3 to opens do
    ignore (close_one ())
  done;
  let d, p = first in
  equal ~msg:"the first device's last value, and its point reached"
    (pair int bool)
    (Rig.Point.value p, true)
    (Rig.submitted d, Rig.Point.reached p);
  Rig.Point.wait p;
  equal ~msg:"its point's device" (option device) None (Rig.Point.device p);
  let d, p = lost in
  raises_match ~msg:"a lost point's wait"
    (function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false)
    (fun () -> Rig.Point.wait p);
  equal ~msg:"a lost point's device" (option device) None (Rig.Point.device p)

(* A child opens software devices and keeps them until an open is [Error], which
   comes within [slots] + 1 opens, as other devices of the process may hold
   slots too. Once one of them closed and its slot passed on, an open succeeds.
   The child holds every slot, which no other test sees. *)
let test_every_slot_held () =
  let lines, ended =
    in_child (fun () ->
        let rec fill held n =
          match Rig.software (Printf.sprintf "limit:held-%d" n) with
          | Ok d when n <= slots -> fill (d :: held) (n + 1)
          | r -> (held, r)
        in
        let held, refused = fill [] 1 in
        Rig.close (List.hd held);
        released ();
        [ answer refused; answer (Rig.software "limit:held-after") ])
  in
  equal ~msg:"the child"
    (pair (list string) bool)
    ([ "Error"; "Ok limit:held-after" ], true)
    (lines, ended = Unix.WEXITED 0)

(* A device that closed keeps nothing once its slot passes on: its line is
   static, its C record goes with its OCaml record, and the slot's next device
   takes its word. Each heap may grow by [slack] over [clean] cycles, as macOS's
   count of the C heap steps by 16 KiB or more: a leak of two bytes per open
   would exceed it. The cycles start after [warm] of them. *)
let warm = 10_000
let clean = 60_000
let slack = 64 * 1024

let heaps () =
  Gc.full_major ();
  let c =
    match Support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  (c, (Gc.stat ()).live_words * (Sys.word_size / 8))

let test_closed_keeps_nothing () =
  for _ = 1 to warm do
    ignore (close_one ())
  done;
  let c0, ml0 = heaps () in
  for _ = 1 to clean do
    ignore (close_one ())
  done;
  let c1, ml1 = heaps () in
  at_most int ~msg:"the C heap's growth over the clean cycles" ~than:slack
    (c1 - c0);
  at_most int ~msg:"the OCaml heap's growth over the clean cycles" ~than:slack
    (ml1 - ml0)

(* A device lost with values unreached keeps what answers its points for good:
   its C record and the range of its slot that holds its lost values, and its
   OCaml record, which the range holds. [c_loss] and [ml_loss] bound each per
   loss, baselines measured on the M1 Max: about 1 KiB of C heap and 1.7 KiB of
   OCaml heap. *)
let losses = 1_000
let c_loss = 2 * 1024
let ml_loss = 4 * 1024

let test_lost_keeps_record () =
  for _ = 1 to warm do
    ignore (close_one ())
  done;
  ignore (lose "limit:lost-warm");
  let c0, ml0 = heaps () in
  for _ = 1 to losses do
    ignore (lose "limit:lost")
  done;
  let c1, ml1 = heaps () in
  at_most int ~msg:"the C heap's growth over the losses"
    ~than:(slack + (losses * c_loss))
    (c1 - c0);
  at_most int ~msg:"the OCaml heap's growth over the losses"
    ~than:(slack + (losses * ml_loss))
    (ml1 - ml0)

let () =
  exit
    (run "rig.limit"
       [
         group "slots"
           [
             test "the process's first failure holds its slot for good"
               test_first_failure;
             test "a process opens devices past every slot" test_past_every_slot;
             test "an open while every slot is held is Error"
               test_every_slot_held;
             test "a closed device keeps nothing once its slot passes on"
               test_closed_keeps_nothing;
             test "a device lost with work queued keeps its record alone"
               test_lost_keeps_record;
           ];
       ])
