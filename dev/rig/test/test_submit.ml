(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (Rig.memory_device name)

(* [b] borrowed on [d], as a run on [d] takes it. *)
let on d b = require_some (B.borrow d b)
let empty ?(reads = 0) ?(writes = 0) d = Sub.make ~reads ~writes d [||]

let submit ?(run = Sub.Run.make ()) ?(reads = [||]) ?(writes = [||])
    ?(waits = [||]) s =
  Rig.submit s ~run ~reads ~writes ~waits

let page_bytes = 1 lsl 16

(* A part that adds 1 to the 64-bit word at the start of [arg]. *)
let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let word b = Support.load (B.address b)

let test_values () =
  let d = memory "submit:values" in
  let s = empty d and run = Sub.Run.make () in
  let values = List.init 5 (fun _ -> Rig.Point.value (submit ~run s)) in
  equal (list int) [ 1; 2; 3; 4; 5 ] values;
  equal int 5 (Rig.submitted d);
  equal int 5 (Rig.signaled d);
  equal bool true (Rig.equal d (Rig.Point.device (submit ~run s)))

let test_fill () =
  let d = memory "submit:fill" in
  let arg = B.create Rig.host 8 in
  Support.store (B.address arg) 0;
  let s = Sub.make ~reads:0 ~writes:0 d [| bump arg; bump arg |] in
  let run = Sub.Run.make () in
  ignore (submit ~run s);
  ignore (submit ~run s);
  equal int 4 (word arg)

(* Polled runs nothing until a wait reaches its sleep. *)
let test_polled () =
  let d, p = P.open_ "submit:polled" in
  let s = empty d and run = Sub.Run.make () in
  let a = submit ~run s in
  let b = submit ~run s in
  equal int 2 (P.queued p);
  equal int 0 (Rig.signaled d);
  Rig.wait d (Rig.Point.value b);
  equal int 0 (P.queued p);
  equal bool true (Rig.Point.value a < Rig.Point.value b)

(* Work that runs long is waited for where the driver states no hang bound. *)
let test_still () =
  let d, p = P.open_ "submit:still" in
  let v = Rig.Point.value (submit (empty d)) in
  P.stall p 3;
  Rig.wait d v;
  equal (option string) None (Rig.lost d);
  equal int 4 (List.length (List.filter (( = ) "sleep") (P.log p)))

(* Hang bounds *)

let hang_ms = 100
let lost = function Rig.Lost _ -> true | _ -> false

(* Work whose word stands still past the bound loses the device, not before
   the bound, and the driver's stop is given the reason. *)
let test_hang () =
  let d, p = P.open_ ~hang_ms "submit:hang" in
  let v = Rig.Point.value (submit (empty d)) in
  P.stall p max_int;
  let t0 = Rig.Profile.now () in
  raises_match lost (fun () -> Rig.wait d v);
  let waited = (Rig.Profile.now () - t0) / 1_000_000 in
  let why = Printf.sprintf "no progress for %d ms" hang_ms in
  equal (option string) ~msg:"the loss" (Some why) (Rig.lost d);
  equal (option string) ~msg:"the stop's fault" (Some why) (P.stop_fault p);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* A word that moves within the bound loses nothing, however long the work
   runs in all: here 15 values reached 20 ms apart. *)
let test_hang_moving () =
  let d, p = P.open_ ~hang_ms "submit:hang-moving" in
  let s = empty d and run = Sub.Run.make () in
  for _ = 1 to 15 do
    ignore (submit ~run s)
  done;
  let last = Rig.submitted d in
  P.stall p max_int;
  let move () =
    for v = 1 to last do
      Thread.delay 0.02;
      P.set_word p v
    done
  in
  let mover = Thread.create move () in
  Rig.wait d last;
  Thread.join mover;
  equal (option string) None (Rig.lost d)

(* A device idle for longer than its bound loses nothing by it: the clock counts
   only while a committed value stays above the word. *)
let test_hang_idle () =
  let d, _ = P.open_ ~hang_ms "submit:hang-idle" in
  let s = empty d and run = Sub.Run.make () in
  Rig.wait d (Rig.Point.value (submit ~run s));
  Thread.delay (3. *. Float.of_int hang_ms /. 1000.);
  Rig.wait d (Rig.Point.value (submit ~run s));
  equal (option string) None (Rig.lost d)

(* A device whose queue waits on a value of a device lost to its bound is lost
   with it. *)
let test_hang_spread () =
  let producer, pp = P.open_ ~hang_ms "submit:hang-producer" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "submit:hang-consumer" in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.stall pp max_int;
  raises_match lost (fun () -> Rig.wait producer (Rig.Point.value a));
  equal (option string) (Some "submit:hang-producer lost") (Rig.lost consumer)

(* A device whose queue waits for another device's work loses nothing to its
   bound while that work runs, however long: its clock starts once the work is
   reached. The producer's kernel runs three bounds; the consumer's first two
   sleeps return with its work unrun. *)
let test_hang_in_queue () =
  let producer, _ = P.open_ ~runs:`Itself "submit:in-queue-producer" in
  let consumer, pc =
    P.open_ ~hang_ms ~waits_on:[ `Host ] "submit:in-queue-consumer"
  in
  let ms = B.create Rig.host 8 in
  Support.store (B.address ms) (3 * hang_ms);
  let fill =
    Sub.Fill { fill = Support.slow; arg = ms; ring_units = 0; segment_bytes = 0 }
  in
  let kernel = { Sub.queue = "COMPUTE:0"; after = [||]; work = fill } in
  let a = submit (Sub.make ~reads:0 ~writes:0 producer [| kernel |]) in
  let v = Rig.Point.value (submit (empty consumer) ~waits:[| a |]) in
  P.stall pc 2;
  Rig.wait consumer v;
  equal (option string) None (Rig.lost consumer)

let test_refusals () =
  let d = memory "submit:refusals" in
  let arg = B.create Rig.host 8 in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 d [| { (bump arg) with after = [| 0 |] } |]);
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 d [| { (bump arg) with queue = "COPY:9" } |]);
  raises_match Exn.invalid_arg (fun () -> Sub.make ~reads:(-1) ~writes:0 d [||]);
  raises_match Exn.invalid_arg (fun () -> submit (empty ~reads:1 d))

(* Each misuse [Submission.make] and [submit] state raises
   [Invalid_argument]. *)
let test_make_refusals () =
  let d, _ = P.open_ ~host_visible:false "submit:make-refusals" in
  let arg = B.create Rig.host 8 in
  let unseen = B.create d 8 and other = B.create d 16 in
  let dead = B.create d 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ [ dead ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" dead));
  let copy src dst =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let make d parts = ignore (Sub.make ~reads:0 ~writes:0 d parts) in
  let s = Sub.make ~reads:1 ~writes:1 d [||] in
  let elsewhere = B.create (fst (P.open_ "submit:elsewhere")) 8 in
  List.iter
    (fun (msg, f) -> raises_match ~msg Exn.invalid_arg f)
    [
      ("on the host", fun () -> make Rig.host [||]);
      ( "after a negative index",
        fun () -> make d [| { (bump arg) with after = [| -1 |] } |] );
      ( "a fill's argument the host does not address",
        fun () -> make d [| bump unseen |] );
      ( "a copy of buffers of two sizes",
        fun () -> make d [| copy unseen other |] );
      ( "a copy of another device's memory",
        fun () -> make d [| copy arg (B.create Rig.host 8) |] );
      ("a part's dead buffer", fun () -> make d [| copy dead unseen |]);
      ( "a run's read of the host's memory",
        fun () -> ignore (submit s ~reads:[| arg |] ~writes:[| unseen |]) );
      ( "a run's write of another device's memory",
        fun () -> ignore (submit s ~reads:[| unseen |] ~writes:[| elsewhere |])
      );
      ( "a run of more reads than made",
        fun () ->
          ignore (submit s ~reads:[| unseen; unseen |] ~writes:[| unseen |]) );
      ( "a run of fewer writes than made",
        fun () -> ignore (submit s ~reads:[| unseen |] ~writes:[||]) );
      ( "a run's dead buffer",
        fun () -> ignore (submit s ~reads:[| dead |] ~writes:[| unseen |]) );
    ]

(* A submit that raises keeps nothing of its run: the next submit of the
   submission waits for none of its points. *)
let test_cleared_on_raise () =
  let d = memory "submit:cleared" in
  let producer, pp = P.open_ "submit:cleared-producer" in
  let point = submit (empty producer) in
  P.fail pp;
  (try ignore (submit (empty producer)) with Rig.Lost _ -> ());
  let s = empty ~reads:1 d and b = on d (B.create Rig.host 8) in
  let run = Sub.Run.make () in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit ~run s ~reads:[| b |] ~waits:[| point |]);
  equal int 1 (Rig.Point.value (submit ~run s ~reads:[| b |]))

(* A run's buffer whose memory was consumed through another buffer refuses the
   submit. *)
let test_dead_slot () =
  let d = memory "submit:dead-slot" in
  let b = B.create Rig.host 8 in
  let s = empty ~reads:1 d and borrowed = on d b in
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" b));
  raises_match Exn.invalid_arg (fun () -> submit s ~reads:[| borrowed |]);
  equal int 0 (Rig.submitted d)

let test_wait_beyond () =
  let d = memory "submit:beyond" and e = memory "submit:beyond-2" in
  let p = submit (empty e) in
  raises_match Exn.invalid_arg (fun () -> Rig.wait e (Rig.Point.value p + 1));
  ignore (submit (empty d) ~waits:[| p |])

(* A read of memory another device wrote waits for that write: on the host,
   since a memory device waits on no other device in its queue. *)
let test_read_waits () =
  let producer, pp = P.open_ "submit:producer" in
  let consumer = memory "submit:consumer" in
  let b = B.create producer page_bytes in
  ignore (submit (empty ~writes:1 producer) ~writes:[| b |]);
  equal int 1 (P.queued pp);
  ignore (submit (empty ~reads:1 consumer) ~reads:[| on consumer b |]);
  equal int 0 (P.queued pp)

(* A part's buffers are ordered as a run's are: a copy waits for another
   device's write of its source and for another device's read of its
   destination. *)
let test_part_points () =
  let d, _ = P.open_ "submit:parts" in
  let e, pe = P.open_ "submit:parts-other" in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let s = Sub.make ~reads:0 ~writes:0 d [| copy |] and run = Sub.Run.make () in
  ignore (submit (empty ~writes:1 e) ~writes:[| on e src |]);
  ignore (submit ~run s);
  equal ~msg:"after a write of the source" int 0 (P.queued pe);
  ignore (submit (empty ~reads:1 e) ~reads:[| on e dst |]);
  ignore (submit ~run s);
  equal ~msg:"after a read of the destination" int 0 (P.queued pe)

(* A run's buffers stay reachable until their stamps are raised: a buffer only
   the run's array reaches, while the submit waits on the host for another
   device, is not collected, and a buffer made meanwhile by another domain,
   after a full collection and a drain, gets other memory. *)
let test_run_keeps () =
  let producer, pp = P.open_ "submit:keeps-producer" in
  let d, _ = P.open_ "submit:keeps" in
  let n = 4096 in
  let a = submit (empty producer) in
  P.gate pp;
  let other =
    Domain.spawn (fun () ->
        Support.await "the submit waiting on the producer" (fun () ->
            P.sleepers pp = 1);
        Gc.full_major ();
        let b = B.create d n in
        P.open_gate pp;
        b)
  in
  let at = ref 0 in
  let run =
    submit (empty ~writes:1 d)
      ~writes:
        (let w = B.create d n in
         at := B.address w;
         [| w |])
      ~waits:[| a |]
  in
  let b = Domain.join other in
  equal ~msg:"another domain's buffer" bool false (B.address b = !at);
  Rig.wait d (Rig.Point.value run)

(* A Polled device that waits on host-written words waits for a producer in its
   queue: the submit hands it over without waiting. *)
let in_queue ~completion =
  let name = match completion with `Host -> "host" | `Object -> "object" in
  let producer, pp = P.open_ ~completion ("submit:iq-producer-" ^ name) in
  let consumer, cp =
    P.open_
      ~waits_on:[ (completion :> [ `Store | `Host | `Object ]) ]
      ("submit:iq-consumer-" ^ name)
  in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  (* The wait names the producer's word, or its object, by rig_edge.h's kinds:
     Polled's word and object are both at its word's address. *)
  let kind =
    match completion with
    | `Host -> Support.rig_word
    | `Object -> Support.rig_object
  in
  equal
    (list (triple int int int))
    [ (kind, P.word_at pp, 1) ]
    (P.last_waits cp);
  equal int 1 (P.queued pp);
  equal int 1 (P.queued cp);
  equal int 0 (P.run cp);
  Rig.wait producer (Rig.Point.value a);
  equal int 1 (P.run cp);
  equal int (Rig.Point.value b) (Rig.signaled consumer)

(* A submission with more unreached producers than its device's queue waits on
   in one submission waits on the host for the others: its queue gets
   [max_waits] waits, and it completes once those are reached. *)
let test_max_waits () =
  let producers =
    List.init 3 (fun i -> P.open_ (Printf.sprintf "submit:bound-producer-%d" i))
  in
  let consumer, cp = P.open_ ~waits_on:[ `Host ] ~max_waits:2 "submit:bound" in
  let points = List.map (fun (d, _) -> submit (empty d)) producers in
  let b = submit (empty consumer) ~waits:(Array.of_list points) in
  let reached = List.filter (fun (_, p) -> P.queued p = 0) producers in
  equal ~msg:"waits in the queue" int 2 (List.length (P.last_waits cp));
  equal ~msg:"producers reached on the host" int 1 (List.length reached);
  equal ~msg:"before its waits hold" int 0 (P.run cp);
  List.iter (fun (_, p) -> ignore (P.run p)) producers;
  equal ~msg:"once they hold" int 1 (P.run cp);
  equal int (Rig.Point.value b) (Rig.signaled consumer)

(* A driver answering fewer than zero waits has a queue with no room for any:
   every producer is waited for on the host. *)
let test_negative_max_waits () =
  let producer, pp = P.open_ "submit:negative-producer" in
  let consumer, cp =
    P.open_ ~waits_on:[ `Host ] ~max_waits:(-1) "submit:negative-consumer"
  in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp)

(* A producer whose word the device cannot map is waited for on the host: the
   submit returns with the producer's work done and no wait in the queue. *)
let test_unmapped_wait () =
  let producer, pp = P.open_ "submit:unmapped-producer" in
  let consumer, cp =
    P.open_ ~peers:false ~waits_on:[ `Host ] "submit:unmapped-consumer"
  in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp);
  equal int (Rig.Point.value a) (Rig.signaled producer)

let test_in_queue () = in_queue ~completion:`Host
let test_in_queue_object () = in_queue ~completion:`Object

(* A Polled device that runs itself completes its work though no wait sleeps
   on it. *)
let test_runs_itself () =
  let d, p = P.open_ ~runs:`Itself "submit:itself" in
  let a = submit (empty d) in
  Support.await "the device's own run" (fun () -> P.queued p = 0);
  equal int (Rig.Point.value a) (Rig.signaled d)

(* A wait on a device whose queue waits for a producer's work returns though
   nothing else runs the producer: a device runs its own work while the host
   sleeps on another. *)
let test_wait_in_queue () =
  let producer, _ = P.open_ "submit:sleep-producer" in
  let consumer, cp = P.open_ ~waits_on:[ `Host ] "submit:sleep-consumer" in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  equal ~msg:"waits in the queue" int 1 (List.length (P.last_waits cp));
  Rig.wait consumer (Rig.Point.value b);
  equal ~msg:"the producer's word" int (Rig.Point.value a)
    (Rig.signaled producer)

(* A submit hands its driver the handle of each region its run and parts use,
   once. A case reads the buffers [reads] picks among [n], and writes those
   [writes] picks. *)
type handles = { n : int; reads : int list; writes : int list }

let pp_handles ppf c =
  let ints = Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int) in
  Format.fprintf ppf "@[{ n = %d;@ reads = [%a];@ writes = [%a] }@]" c.n ints
    c.reads ints c.writes

let handles_case =
  let open Gen in
  with_pp pp_handles
    (let* n = int_range 1 40 in
     let picks k = list ~size:(int_range 0 k) (int_range 0 (n - 1)) in
     map
       (fun (reads, writes) -> { n; reads; writes })
       (pair (picks 24) (picks 8)))

let handles_device = lazy (P.open_ "submit:handles")

(* One run for every case's submits: each follows a submit that named other
   regions, whose names the run kept. *)
let handles_run = Sub.Run.make ()

let handles_law c =
  let d, p = Lazy.force handles_device in
  let bs = Array.init c.n (fun _ -> B.create d 8) in
  let reads = List.length c.reads and writes = List.length c.writes in
  let s = Sub.make ~reads ~writes d [| bump (B.create Rig.host 8) |] in
  let pick ks = Array.of_list (List.map (fun k -> bs.(k)) ks) in
  ignore
    (submit ~run:handles_run s ~reads:(pick c.reads) ~writes:(pick c.writes));
  let named = List.sort_uniq Int.compare (c.reads @ c.writes) in
  cover "a buffer named twice"
    (List.length named < List.length c.reads + List.length c.writes);
  cover "more than 16 buffers named" (List.length named > 16);
  equal (list int)
    (List.sort Int.compare (List.map (fun k -> B.address bs.(k)) named))
    (List.sort Int.compare (P.last_handles p))

(* Each submit of one submission hands its driver the handles that submit's run
   and parts name, whatever the submits before it named. A case submits once per
   element of [runs], each picking the run's buffers among [n]. *)
type rerun = { n : int; reads : int; writes : int; runs : int list list }

let pp_rerun ppf c =
  let ints = Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int) in
  let runs = Format.(pp_print_list ~pp_sep:(fun ppf () -> fprintf ppf ";@ ")) in
  Format.fprintf ppf "@[{ n = %d;@ reads = %d;@ writes = %d;@ runs = [%a] }@]"
    c.n c.reads c.writes
    (runs (fun ppf r -> Format.fprintf ppf "[%a]" ints r))
    c.runs

let rerun_case =
  let open Gen in
  with_pp pp_rerun
    (let* n = int_range 1 6 in
     let* reads = int_range 0 4 in
     let* writes = int_range 0 2 in
     let run = list ~size:(constant (reads + writes)) (int_range 0 (n - 1)) in
     map
       (fun runs -> { n; reads; writes; runs })
       (list ~size:(int_range 2 4) run))

let rerun_law c =
  let d, p = Lazy.force handles_device in
  let bs = Array.init c.n (fun _ -> B.create d 8) in
  let s =
    Sub.make ~reads:c.reads ~writes:c.writes d [| bump (B.create Rig.host 8) |]
  in
  let named run =
    let run = Array.of_list (List.map (fun k -> bs.(k)) run) in
    ignore
      (submit ~run:handles_run s ~reads:(Array.sub run 0 c.reads)
         ~writes:(Array.sub run c.reads c.writes));
    List.sort Int.compare (P.last_handles p)
  in
  List.iteri
    (fun i run ->
      if i > 0 then begin
        let last = List.nth c.runs (i - 1) in
        cover "a submit names the last one's buffers" (last = run);
        cover "a submit names other buffers" (last <> run)
      end;
      equal (list int)
        (List.sort_uniq Int.compare (List.map (fun k -> B.address bs.(k)) run))
        (named run))
    c.runs

(* A full queue answers Later: the submit waits for one more value. *)
let test_room () =
  let d, p = P.open_ ~capacity:1 "submit:room" in
  let arg = B.create Rig.host 8 in
  let s = Sub.make ~reads:0 ~writes:0 d [| bump arg |] in
  let run = Sub.Run.make () in
  ignore (submit ~run s);
  ignore (submit ~run s);
  equal int 1 (P.queued p);
  equal int 1 (Rig.signaled d)

(* Runs *)

(* While a submit of a submission waits on the host for a producer, a submit of
   the same submission on another domain, with its own run, completes. *)
let test_run_beside_wait () =
  let producer, pp = P.open_ "submit:beside-producer" in
  let d, _ = P.open_ "submit:beside" in
  let s = empty d and point = submit (empty producer) in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> Rig.Point.value (submit s ~waits:[| point |]))
  in
  Support.await "a sleep at the gate" (fun () -> P.sleepers pp = 1);
  equal ~msg:"the submit beside the wait" int 1 (Rig.Point.value (submit s));
  P.open_gate pp;
  equal ~msg:"the waiting submit" int 2 (Domain.join waiting)

(* A submit made inside another's wait, as a signal handler runs, completes
   with its own run, and raises with the run the waiting submit holds. *)
let test_run_reentrant () =
  let producer, pp = P.open_ "submit:reentrant-producer" in
  let d, _ = P.open_ "submit:reentrant" in
  let s = empty d and point = submit (empty producer) in
  let held = Sub.Run.make () in
  let inner = ref None and refused = ref None in
  let handle _ =
    inner := Some (Rig.Point.value (submit s));
    refused :=
      Some
        (match submit ~run:held s with
        | _ -> "submitted"
        | exception Invalid_argument why -> why)
  in
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let outer =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit ~run:held s ~waits:[| point |])
  in
  equal ~msg:"the submit inside the wait" (option int) (Some 1) !inner;
  equal ~msg:"the waiting submit" int 2 (Rig.Point.value outer);
  is_some ~msg:"a submit with the held run raised" !refused;
  contains ~msg:"its reason" ~sub:"another submit is using the run"
    (Option.get !refused)

(* A copy on a device that runs no copies is refused where the caller can act:
   when the submission is made. *)
let test_copy_refused () =
  let d, _ = P.open_ ~copies:false "submit:no-copies" in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy =
    { Sub.queue = "COMPUTE:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 d [| copy |])

(* A part of a kind its queue does not run is refused when the submission is
   made: Polled's queues run fills and copies, and with no copies, fills alone,
   and its compute queue also launches. *)
let test_runs () =
  let d, _ = P.open_ "submit:runs" in
  let e, _ = P.open_ ~copies:false "submit:runs-fills" in
  let runs d =
    List.map (fun (q : Rig.queue) -> (q.name, q.runs)) (Rig.queues d)
  in
  let kind ppf k =
    Format.pp_print_string ppf
      (match k with
      | Rig.Words -> "Words"
      | Fill -> "Fill"
      | Copy -> "Copy"
      | Launch -> "Launch")
  in
  let queue ppf (name, runs) =
    Format.fprintf ppf "%s: %a" name
      (Format.pp_print_list ~pp_sep:Format.pp_print_space kind)
      runs
  in
  let kinds =
    Testable.make ~pp:(Format.pp_print_list queue) ~equal:( = )
  in
  equal ~msg:"Polled's queues" kinds
    [ ("COMPUTE:0", [ Rig.Launch; Fill; Copy ]); ("COPY:0", [ Fill; Copy ]) ]
    (runs d);
  equal ~msg:"Polled's queues with no copies" kinds
    [ ("COMPUTE:0", [ Rig.Launch; Fill ]) ]
    (runs e);
  let words = B.create Rig.host 8 in
  let part queue work = { Sub.queue; after = [||]; work } in
  List.iter
    (fun queue ->
      raises_match ~msg:("words on " ^ queue) Exn.invalid_arg (fun () ->
          Sub.make ~reads:0 ~writes:0 d [| part queue (Sub.Words words) |]))
    [ "COMPUTE:0"; "COPY:0" ];
  let src = B.create e 8 and dst = B.create e 8 in
  raises_match ~msg:"a copy where no queue copies" Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 e
        [| part "COMPUTE:0" (Sub.Copy { src; dst }) |]);
  let image = Result.get_ok (Rig.Image.load d "code:64") in
  let launch = Sub.Launch { image; kernel = "main"; params = 0; refs = [||] } in
  raises_match ~msg:"a launch on COPY:0" Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 d [| part "COPY:0" launch |]);
  equal ~msg:"no queues on the host" int 0 (List.length (Rig.queues Rig.host))

(* A value's work starts once the previous value's completed, on every queue:
   a copy on COPY:0, then a fill on COMPUTE:0 that reads what the copy wrote
   without naming it, sees the copy, and the other way round. Neither names a
   buffer of the other, so only the device's order orders them. *)
let test_device_order () =
  let d, p = P.open_ "submit:device-order" in
  (* Host buffers of 64 KiB start on a page, so the device borrows them. *)
  let n = 64 * 1024 in
  let filled c =
    let b = B.create Rig.host n in
    Bigarray.Array1.fill (B.bigarray Bigarray.char b) c;
    b
  in
  let a = filled 'a' and b = B.create d n in
  let c = filled '-' and out = filled '-' and z = filled 'z' in
  let bytes x =
    String.init n (Bigarray.Array1.get (B.bigarray Bigarray.char x))
  in
  let fill ~dst ~src =
    let arg = B.create Rig.host 24 in
    let at = B.address arg in
    Support.store at dst;
    Support.store (at + 8) src;
    Support.store (at + 16) n;
    {
      Sub.queue = "COMPUTE:0";
      after = [||];
      work =
        Sub.Fill
          { fill = Support.carry; arg; ring_units = 0; segment_bytes = 0 };
    }
  in
  let copy src dst =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let run parts = ignore (submit (Sub.make ~reads:0 ~writes:0 d parts)) in
  let a' = require_some (B.borrow d a) and c' = require_some (B.borrow d c) in
  run [| copy a' b |];
  run [| fill ~dst:(B.address out) ~src:(B.address b) |];
  ignore (P.run p);
  equal ~msg:"a fill after a copy" string (String.make n 'a') (bytes out);
  run [| fill ~dst:(B.address b) ~src:(B.address z) |];
  run [| copy b c' |];
  ignore (P.run p);
  equal ~msg:"a copy after a fill" string (String.make n 'z') (bytes c);
  ignore (Sys.opaque_identity (a, out, z))

(* Lifetime against the driver's frees *)

(* Each submission copies between two fresh buffers that only it holds, and each
   submit writes a fresh slot buffer the caller drops at once. Polled logs every
   free with its word: no free may name a part's memory while its submission is
   reachable, nor memory a submit used before the device's word reached that
   submit's value. *)
type watched = {
  addresses : int list;
  since : int;  (** Frees logged before the memory was made. *)
  mutable dropped : int option;  (** Frees logged before it was dropped. *)
  mutable last : int;  (** The last value that used it. *)
}

type device = { d : Rig.t; p : P.t; mutable watched : watched list }

type submission = {
  dev : device;
  mutable s : Sub.t option;
  run : Sub.Run.t;
  parts : watched;
}

type device_model = { mutable value : int }
type submission_model = { model : device_model; mutable live : bool }

let devices = Atomic.make 0

let open_device () =
  let n = Atomic.fetch_and_add devices 1 in
  let d, p = P.open_ (Printf.sprintf "submit:lifetime-%d" n) in
  { d; p; watched = [] }

let frees t = List.length (P.frees t.p)

let watch t ?dropped ~last addresses =
  let w = { addresses; since = frees t; dropped; last } in
  t.watched <- w :: t.watched;
  w

let check_frees _ t =
  cover "memory a submission named is freed"
    (List.exists
       (fun (at, _) -> List.exists (fun w -> List.mem at w.addresses) t.watched)
       (P.frees t.p));
  List.iteri
    (fun i (at, word) ->
      List.iter
        (fun w ->
          if i >= w.since && List.mem at w.addresses then
            match w.dropped with
            | Some j when j <= i ->
                at_least
                  ~msg:(Printf.sprintf "the word when %#x was freed" at)
                  int ~than:w.last word
            | _ -> failf "%#x was freed while a submission held it" at)
        t.watched)
    (P.frees t.p)

let make t =
  let src = B.create t.d 256 and dst = B.create t.d 256 in
  let parts = watch t ~last:0 [ B.address src; B.address dst ] in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let s = Sub.make ~reads:0 ~writes:1 t.d [| copy |] in
  { dev = t; s = Some s; run = Sub.Run.make (); parts }

let submit_sub sub =
  let t = sub.dev in
  let s = Option.get sub.s in
  let out = B.create t.d 256 in
  let at = B.address out in
  let v = Rig.Point.value (submit ~run:sub.run s ~writes:[| out |]) in
  sub.parts.last <- v;
  ignore (watch t ~dropped:(frees t) ~last:v [ at ]);
  v

let drop sub =
  sub.s <- None;
  sub.parts.dropped <- Some (frees sub.dev);
  Gc.full_major ()

let drain t =
  Gc.full_major ();
  ignore (B.create t.d 64);
  Rig.free_cache t.d

let device = abstract ~invariant:check_frees "d"
let submission = abstract "s"
let nothing _ = ()

let lifetime =
  [
    command "open"
      (Gen.unit @-> makes device)
      (fun () -> { value = 0 })
      open_device;
    command "make"
      (device ^-> makes submission)
      (fun model -> { model; live = true })
      make;
    command "submit"
      ~pre:(fun r -> r.live)
      (submission ^-> returns int)
      (fun r ->
        r.model.value <- r.model.value + 1;
        r.model.value)
      submit_sub;
    command "drop"
      ~pre:(fun r -> r.live)
      (submission ^-> returns unit)
      (fun r -> r.live <- false)
      drop;
    command "drain" (device ^-> returns unit) nothing drain;
    command "device runs"
      (device ^-> returns unit)
      nothing
      (fun t -> ignore (P.run t.p));
    command "wait"
      (device ^-> returns unit)
      nothing
      (fun t -> Rig.wait t.d (Rig.submitted t.d));
  ]

let test_allocation () =
  let d = memory "submit:words" in
  let s = empty ~reads:1 d and reads = [| on d (B.create Rig.host 8) |] in
  let run = Sub.Run.make () in
  ignore (Rig.submit s ~run ~reads ~writes:[||] ~waits:[||]);
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    ignore
      (Sys.opaque_identity (Rig.submit s ~run ~reads ~writes:[||] ~waits:[||]))
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  equal int 0 words

(* Two domains submitting one submission: the submits take turns, each takes the
   device's next value, and every fill runs. A program counts values from
   [base], its device's value when it began. *)
type shared = { dev : Rig.t; base : int; arg : B.t; sub : Sub.t }
type shared_model = { mutable values : int }

(* Devices go back to a pool when their program ends. *)
let shared_pool = Mutex.create ()
let shared_free = ref []
let shared_opened = Atomic.make 0

let make_shared () =
  let d =
    match
      Mutex.protect shared_pool (fun () ->
          match !shared_free with
          | d :: rest ->
              shared_free := rest;
              Some d
          | [] -> None)
    with
    | Some d -> d
    | None ->
        memory
          (Printf.sprintf "submit:shared-%d"
             (Atomic.fetch_and_add shared_opened 1))
  in
  let arg = B.create Rig.host 8 in
  Support.store (B.address arg) 0;
  {
    dev = d;
    base = Rig.submitted d;
    arg;
    sub = Sub.make ~reads:0 ~writes:0 d [| bump arg; bump arg |];
  }

(* Every submit ran both its fills. *)
let release_shared t =
  equal ~msg:"fills" int (2 * (Rig.submitted t.dev - t.base)) (word t.arg);
  Mutex.protect shared_pool (fun () -> shared_free := t.dev :: !shared_free)

let shared =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "values %d" r.values)
    ~release:release_shared "s"

(* Each domain submits with its own run. *)
let shared_run = Domain.DLS.new_key Sub.Run.make

let judge_submit r = function
  | Ok v ->
      equal ~msg:"value" int (r.values + 1) v;
      r.values <- v
  | Error e -> raise e

let shared_commands =
  [
    command "make"
      (Gen.unit @-> makes shared)
      (fun () -> { values = 0 })
      make_shared;
    command "submit"
      (shared ^-> judges int)
      judge_submit
      (fun t ->
        let run = Domain.DLS.get shared_run in
        Rig.Point.value (submit ~run t.sub) - t.base);
  ]

let tests =
  [
    group ~timeout "values"
      [
        test "values follow one another from 1" test_values;
        test "a submission's fills run with its value" test_fill;
        stateful ~domains:2 "two domains submit one submission at once"
          shared_commands;
        test "a device's work completes once a wait reaches it" test_polled;
        test "a wait names a submitted value" test_wait_beyond;
        test "a word still for three intervals loses nothing" test_still;
      ];
    group ~timeout "hangs"
      [
        test "work still past the hang bound loses the device" test_hang;
        test "a word moving within the bound loses nothing" test_hang_moving;
        test "an idle device loses nothing to the bound" test_hang_idle;
        test "a queue waiting on a hung device is lost with it"
          test_hang_spread;
        test "a queue waiting on longer work elsewhere loses nothing"
          test_hang_in_queue;
      ];
    group ~timeout "runs"
      [
        test "a submit beside another's wait completes with its own run"
          test_run_beside_wait;
        test "a submit inside another's wait completes with its own run"
          test_run_reentrant;
      ];
    group ~timeout "refusals"
      [
        test "a submission refuses what it cannot run" test_refusals;
        test "a submission refuses each misuse its interface states"
          test_make_refusals;
        test "a submit that raises keeps nothing of its run"
          test_cleared_on_raise;
        test "a run's buffer that died refuses the submit" test_dead_slot;
        test "a copy on a device that runs no copies is refused"
          test_copy_refused;
        test "a part of a kind its queue does not run is refused at make"
          test_runs;
      ];
    group ~timeout "order"
      [
        test "a value's work follows the previous value's, on every queue"
          test_device_order;
        test "a read waits for another device's write" test_read_waits;
        test "a run's buffers stay reachable until its stamps are raised"
          test_run_keeps;
        test "a part's buffers wait as a run's do" test_part_points;
        test "a queue that waits on host words waits in the queue" test_in_queue;
        test "a wait on a queue waiting for a producer runs the producer"
          test_wait_in_queue;
        test "a Polled device that runs itself runs its work unasked"
          test_runs_itself;
        test "a producer whose word the device cannot map is waited on the host"
          test_unmapped_wait;
        test "waits beyond the queue's bound are waited for on the host"
          test_max_waits;
        test "a driver answering negative waits has the host wait for all"
          test_negative_max_waits;
        test "a queue that waits on objects waits on the producer's object"
          test_in_queue_object;
        test "a full queue's submit waits for room" test_room;
        prop "a submit names each region it uses once" handles_case handles_law;
        prop "each submit names the regions it uses, whatever the last named"
          rerun_case rerun_law;
      ];
    group ~timeout "lifetime"
      [
        stateful "memory a submission names returns after its last submit"
          lifetime;
      ];
    group ~timeout "cost"
      [ test "a submit that does not wait allocates nothing" test_allocation ];
  ]

let () = exit (run "rig.submit" tests)
