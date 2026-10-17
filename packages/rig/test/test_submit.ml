(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = Rig_support.time 60.
let memory name = Result.get_ok (Rig.memory_device name)

(* [b] borrowed on [d], as a run on [d] takes it. *)
let on d b = require_some (B.borrow d b)
(* A submission of no part, or of one fill that does nothing and uses a slot per
   element of [uses] ({!Support.touching}). *)
let empty ?(uses = [||]) d =
  if uses = [||] then Sub.make d [||] else Sub.make d [| Support.touching uses |]

(* Submits [s] with [run], which must be a run of [s], or with a fresh run. *)
let submit ?run ?(buffers = [||]) ?waits s =
  let run = match run with Some r -> r | None -> Sub.Run.make s in
  if Sub.Run.submission run != s then fail "a run of another submission";
  Rig.submit ?waits run ~buffers

let page_bytes = 1 lsl 16

(* A part that adds 1 to the 64-bit word in run slot 0. *)
let bump =
  Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.bump ~params:8

let idle =
  (Sub.fill Support.noop ~params:0)

let word b = Support.load (Support.Reader.host b)

(* The handle of [b]'s memory: the address of a Polled region. *)
let handle b = Nativeint.to_int (B.handle b)

(* The driver's copy facts describe the original allocation through views and
   borrows. Polled exposes the payload that actually reached its queue. *)
let copy_origins =
  [
    ("device", `Own B.Device, true);
    ("device returned", `Returned, true);
    ("mapped", `Own B.Mapped, false);
    ("pinned", `Own B.Pinned, false);
    ("host", `Host, false);
    ("peer device", `Peer B.Device, false);
    ("peer mapped", `Peer B.Mapped, false);
    ("peer pinned", `Peer B.Pinned, false);
  ]

let copy_serial = Atomic.make 0

let test_copy_memory (length, src_first, dst_first) =
  let serial = Atomic.fetch_and_add copy_serial 1 in
  let name = Printf.sprintf "submit:copy-memory-%d" serial in
  let d, p = P.open_ ~host_visible:false name in
  let peer, _ = P.open_ ~host_visible:false (name ^ "-peer") in
  let bytes = length + max src_first dst_first in
  let endpoint first (label, origin, device_memory) =
    let root =
      match origin with
      | `Own memory -> B.create ~memory d bytes
      | `Returned -> on d (on peer (B.create d bytes))
      | `Peer memory -> B.create ~memory peer bytes
      | `Host -> B.create Rig.host (if bytes = 0 then 0 else max page_bytes bytes)
    in
    let buffer = on d (B.view (on d root) ~first ~length) in
    (label, buffer, device_memory && bytes <> 0)
  in
  let sources = List.map (endpoint src_first) copy_origins in
  let destinations = List.map (endpoint dst_first) copy_origins in
  let rows =
    List.concat_map
      (fun (src_name, src, src_device) ->
        List.map
          (fun (dst_name, dst, dst_device) ->
            ( src_name ^ " to " ^ dst_name,
              (Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))),
              src_device && dst_device ))
          destinations)
      sources
  in
  let parts = Array.of_list (List.map (fun (_, part, _) -> part) rows) in
  Rig.Point.wait (submit (Sub.make d parts));
  equal (list (pair string bool))
    (List.map (fun (label, _, expected) -> (label, expected)) rows)
    (List.map2
       (fun (label, _, _) actual -> (label, actual))
       rows (P.copy_device_memory p));
  Rig.close d;
  Rig.close peer

let copy_memory_law (length, src_first, dst_first) =
  cover "empty view of an allocation"
    (length = 0 && max src_first dst_first > 0);
  cover "one byte" (length = 1);
  cover "different offsets" (src_first <> dst_first);
  test_copy_memory (length, src_first, dst_first)

(* Buffer.copy and a submission's copies carry the facts when a device of
   another machine receives local host addresses directly, interleaved with its
   own memory, through a buffer or a slot. Two local ends are refused, and so
   is local memory in a slot a ref names. *)
let test_copy_memory_sequence () =
  let machine = Support.machine "submit:copy-memory-machine" in
  let p = P.make ~host_visible:false () in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         (P.opener "copy-memory" p))
  in
  let a = B.create d 8 and b = B.create d 8 in
  let mapped = B.create ~memory:B.Mapped d 8 in
  let pinned = B.create ~memory:B.Pinned d 8 in
  let host = B.create Rig.host 8 in
  List.iter
    (fun (src, dst) -> B.copy ~src ~dst)
    [ (a, b); (host, a); (a, b); (b, host); (mapped, a); (a, b); (a, pinned) ];
  let copy src dst buffers =
    Rig.Point.wait (submit (Sub.make d [| Sub.copy ~src ~dst |]) ~buffers)
  in
  copy (Slot 0) (Buffer a) [| host |];
  copy (Buffer a) (Slot 0) [| host |];
  copy (Buffer host) (Slot 0) [| b |];
  equal (list bool)
    [ true; false; true; false; false; true; false; false; false; false ]
    (P.copy_device_memory p);
  equal (list string)
    [
      "none"; "src"; "none"; "dst"; "none"; "none"; "none"; "src"; "dst"; "src";
    ]
    (List.map
       (function `None -> "none" | `Src -> "src" | `Dst -> "dst")
       (P.copy_sides p));
  let other = B.create Rig.host 8 in
  raises_match ~msg:"two local buffers" Exn.invalid_arg (fun () ->
      Sub.make d Sub.[| copy ~src:(Buffer host) ~dst:(Buffer other) |]);
  raises_match ~msg:"a local buffer and a local slot" Exn.invalid_arg (fun () ->
      copy (Buffer host) (Slot 0) [| other |]);
  let fill = Sub.make d [| Support.touching [| Sub.read |] |] in
  raises_match ~msg:"a ref's local buffer" Exn.invalid_arg (fun () ->
      submit fill ~buffers:[| host |]);
  Rig.close d

(* Copy ends *)

(* A copy from a view of [src_len] bytes at [src_first] of one allocation into
   a view of [dst_len] bytes at [dst_first] of the same allocation, if
   [shared], or of another; each end a run's slot, if [src_slot] or
   [dst_slot], or a buffer the submission names. *)
type ends = {
  shared : bool;
  src_first : int;
  dst_first : int;
  src_len : int;
  dst_len : int;
  src_slot : bool;
  dst_slot : bool;
}

let ends_alloc = 40

let pp_ends ppf c =
  let side slot = if slot then "slot" else "buffer" in
  Format.fprintf ppf "%s %d+%d -> %s %s%d+%d" (side c.src_slot) c.src_first
    c.src_len (side c.dst_slot)
    (if c.shared then "same " else "")
    c.dst_first c.dst_len

let gen_ends =
  let open Gen in
  let+ shared = bool
  and+ src_first = int_range 0 24
  and+ dst_first = int_range 0 24
  and+ len = int_range 0 16
  and+ other =
    frequency [ (4, constant None); (1, map Option.some (int_range 0 16)) ]
  and+ src_slot = bool
  and+ dst_slot = bool in
  let dst_len = Option.value other ~default:len in
  { shared; src_first; dst_first; src_len = len; dst_len; src_slot; dst_slot }

let ends_overlap c =
  c.shared && c.src_len > 0 && c.dst_len > 0
  && c.src_first < c.dst_first + c.dst_len
  && c.dst_first < c.src_first + c.src_len

let contents b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

(* A copy whose ends are slots or buffers, in each of the four mixes, copies
   the bytes as a host copy does. Buffers of two lengths or that share a byte
   are refused, by make where both ends are buffers and by the submit
   otherwise, which then takes no value. Copies between two slots run through
   one submission, each case with its own buffers. *)
let ends_law cases =
  let d = memory (Printf.sprintf "submit:ends-%d" (Random.bits ())) in
  let slots = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |] in
  let run = Sub.Run.make slots in
  (* The run of [slots] for its submits, and a fresh one for the others. *)
  let run_of s = if s == slots then Some run else None in
  cover "several runs of one submission"
    (List.length (List.filter (fun c -> c.src_slot && c.dst_slot) cases) > 1);
  List.iteri
    (fun i c ->
      cover "slot to slot" (c.src_slot && c.dst_slot);
      cover "slot to buffer" (c.src_slot && not c.dst_slot);
      cover "buffer to slot" ((not c.src_slot) && c.dst_slot);
      cover "buffer to buffer" (not (c.src_slot || c.dst_slot));
      cover "buffers of two lengths" (c.src_len <> c.dst_len);
      cover "overlapping views of one allocation" (ends_overlap c);
      cover "disjoint views of one allocation"
        (c.shared && c.src_len = c.dst_len && c.src_len > 0
       && not (ends_overlap c));
      let pattern k =
        String.init ends_alloc (fun j -> Char.chr (((k * 61) + j) land 255))
      in
      let a = B.create d ends_alloc and b = B.create d ends_alloc in
      B.blit_from_string (pattern (2 * i)) 0 a 0 ends_alloc;
      B.blit_from_string (pattern ((2 * i) + 1)) 0 b 0 ends_alloc;
      let src = B.view a ~first:c.src_first ~length:c.src_len in
      let dst =
        B.view (if c.shared then a else b) ~first:c.dst_first ~length:c.dst_len
      in
      let refused = c.src_len <> c.dst_len || ends_overlap c in
      let buffers =
        Array.of_list
          ((if c.src_slot then [ src ] else [])
          @ if c.dst_slot then [ dst ] else [])
      in
      let side slot k b = if slot then Sub.Slot k else Sub.Buffer b in
      let make () =
        if c.src_slot && c.dst_slot then slots
        else
          let dst_side = side c.dst_slot (Bool.to_int c.src_slot) dst in
          Sub.make d [| Sub.copy ~src:(side c.src_slot 0 src) ~dst:dst_side |]
      in
      let before = Rig.submitted d in
      if refused && not (c.src_slot || c.dst_slot) then
        raises_match ~msg:"make" Exn.invalid_arg (fun () -> make ())
      else if refused then begin
        let s = make () in
        raises_match ~msg:"the submit" Exn.invalid_arg (fun () ->
            submit ?run:(run_of s) s ~buffers);
        equal ~msg:"a refused run's value" int before (Rig.submitted d)
      end
      else begin
        let into = if c.shared then 2 * i else (2 * i) + 1 in
        let expected = Bytes.of_string (pattern into) in
        Bytes.blit_string (pattern (2 * i)) c.src_first expected c.dst_first
          c.src_len;
        let s = make () in
        Rig.Point.wait (submit ?run:(run_of s) s ~buffers);
        if not c.shared then
          equal ~msg:"the source's allocation" string (pattern (2 * i))
            (contents a);
        equal ~msg:"the destination's allocation" string
          (Bytes.to_string expected)
          (contents (if c.shared then a else b))
      end)
    cases

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let test_values () =
  let d = memory "submit:values" in
  let base = Rig.submitted d in
  let s = empty d in
  let run = Sub.Run.make s in
  let values = List.init 5 (fun _ -> Rig.Point.value (submit ~run s ~buffers:[||])) in
  equal (list int) (List.init 5 (fun i -> base + i + 1)) values;
  equal int (base + 5) (Rig.submitted d);
  equal int (base + 5) (Rig.signaled d);
  equal (option device) (Some d)
    (Rig.Point.device (submit ~run s ~buffers:[||]))

let test_fill () =
  let d = memory "submit:fill" in
  let arg = B.create Rig.host 8 in
  Support.store (Support.Reader.host arg) 0;
  let s = Sub.make d [| bump; bump |] in
  let run = Sub.Run.make s and buffers = [| on d arg |] in
  Sub.Run.int64 run (Sub.block s 0) 0 0L;
  Sub.Run.int64 run (Sub.block s 1) 0 0L;
  ignore (submit ~run ~buffers s);
  ignore (submit ~run ~buffers s);
  equal int 4 (word arg)

(* Polled runs nothing until a wait reaches its sleep. *)
let test_polled () =
  let d, p = P.open_ "submit:polled" in
  let base = Rig.submitted d in
  let s = empty d in
  let run = Sub.Run.make s in
  let a = submit ~run s ~buffers:[||] in
  let b = submit ~run s ~buffers:[||] in
  equal int 2 (P.queued p);
  equal int base (Rig.signaled d);
  Rig.Point.wait b;
  equal int 0 (P.queued p);
  equal bool true (Rig.Point.value a < Rig.Point.value b)

(* Work that runs long is waited for where the driver states no hang bound. *)
let test_still () =
  let d, p = P.open_ "submit:still" in
  let pt = submit (empty d) in
  P.stall p 3;
  Rig.Point.wait pt;
  equal (option string) None (Rig.lost d);
  equal int 4 (P.sleeps p)

(* Hang bounds *)

let hang_ms = 100
let lost = function Rig.Lost _ -> true | _ -> false

(* Work whose word stands still past the bound loses the device, not before the
   bound, and the driver's stop, which its close runs, is given the reason. *)
let test_hang () =
  let d, p = P.open_ ~hang_ms "submit:hang" in
  let pt = submit (empty d) in
  P.stall p max_int;
  let t0 = Rig.Profile.now () in
  raises_match lost (fun () -> Rig.Point.wait pt);
  let waited = (Rig.Profile.now () - t0) / 1_000_000 in
  let why = Printf.sprintf "no progress for %d ms" hang_ms in
  equal (option string) ~msg:"the loss" (Some why) (Rig.lost d);
  Rig.close d;
  equal (option string) ~msg:"the stop's fault" (Some why) (P.stop_fault p);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* Only a wait checks the bound: work stalled three bounds with nothing waiting
   loses nothing, and the clock starts at the wait that finds it. *)
let test_hang_unwatched () =
  let d, p = P.open_ ~hang_ms "submit:hang-unwatched" in
  let pt = submit (empty d) in
  P.stall p max_int;
  Thread.delay (3. *. Float.of_int hang_ms /. 1000.);
  equal (option string) ~msg:"unwatched" None (Rig.lost d);
  let t0 = Rig.Profile.now () in
  raises_match lost (fun () -> Rig.Point.wait pt);
  let waited = (Rig.Profile.now () - t0) / 1_000_000 in
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* A word that moves within the bound loses nothing, however long the work runs
   in all: here 45 values reached 20 ms apart, three times the bound of 300 ms,
   which leaves a mover the scheduler keeps waiting on a busy machine its
   slack. *)
let test_hang_moving () =
  let d, p = P.open_ ~hang_ms:300 "submit:hang-moving" in
  let s = empty d in
  let run = Sub.Run.make s in
  let points = List.init 45 (fun _ -> submit ~run s ~buffers:[||]) in
  let last_point = List.nth points 44 in
  let last = Rig.Point.value last_point in
  P.stall p max_int;
  let move () =
    for v = 1 to last do
      Thread.delay 0.02;
      P.set_word p v
    done
  in
  let mover = Thread.create move () in
  Rig.Point.wait last_point;
  Thread.join mover;
  equal (option string) None (Rig.lost d)

(* A device idle for longer than its bound loses nothing by it: the clock counts
   only while a committed value stays above the word. *)
let test_hang_idle () =
  let d, _ = P.open_ ~hang_ms "submit:hang-idle" in
  let s = empty d in
  let run = Sub.Run.make s in
  Rig.Point.wait (submit ~run s ~buffers:[||]);
  Thread.delay (3. *. Float.of_int hang_ms /. 1000.);
  Rig.Point.wait (submit ~run s ~buffers:[||]);
  equal (option string) None (Rig.lost d)

(* A device whose queue waits on a value of a device lost to its bound is lost
   with it. *)
let test_hang_spread () =
  let producer, pp = P.open_ ~hang_ms "submit:hang-producer" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "submit:hang-consumer" in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.stall pp max_int;
  raises_match lost (fun () -> Rig.Point.wait a);
  equal (option string) (Some "submit:hang-producer lost") (Rig.lost consumer)

(* A device whose queue waits for another device's work loses nothing to its
   bound while that work runs, however long: its clock starts once the work is
   reached. The producer's work runs only when a sleep runs its queue, and the
   consumer's first two sleeps run nothing: the consumer waits two still
   intervals, past its bound, on a value its producer has not reached. *)
let test_hang_in_queue () =
  let producer, _ = P.open_ "submit:in-queue-producer" in
  let consumer, pc =
    P.open_ ~hang_ms ~waits_on:[ `Host ] "submit:in-queue-consumer"
  in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  P.stall pc 2;
  let t0 = Rig.Profile.now () in
  Rig.Point.wait b;
  let waited = (Rig.Profile.now () - t0) / 1_000_000 in
  equal (option string) None (Rig.lost consumer);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* A word that moved while a wait was held between its read and the hang check
   loses nothing: the check judges the word as it stands at the check. The wait
   is held at its producer's gate, since a wait whose word stood still for a
   still interval looks at the producers its queue waits on before it sleeps.
   The clock started at the wait's previous sleep, more than the bound before
   the check. *)
let test_hang_held_read () =
  let producer, pp = P.open_ "submit:held-read-producer" in
  let d, p =
    P.open_ ~hang_ms ~completion:`Store ~waits_on:[ `Host ] "submit:held-read"
  in
  let a = submit (empty producer) in
  ignore (submit (empty d) ~waits:[| a |]);
  let b = submit (empty d) in
  Rig.Point.wait a;
  P.stall pp max_int;
  P.gate p;
  P.stall p 1;
  let waiting =
    Domain.spawn (fun () ->
        match Rig.Point.wait b with
        | () -> None
        | exception Rig.Lost (_, why) -> Some why)
  in
  Support.await "the clock's first sleep" (fun () -> P.sleepers p = 1);
  P.gate pp;
  P.open_gate p;
  Support.await "the wait held after its read" (fun () -> P.sleepers pp = 1);
  P.set_word p 1;
  Thread.delay (2. *. Float.of_int hang_ms /. 1000.);
  P.open_gate pp;
  equal (option string) ~msg:"the wait's loss" None (Domain.join waiting);
  equal (option string) ~msg:"the loss" None (Rig.lost d)

let test_refusals () =
  let d = memory "submit:refusals" in
  raises_match Exn.invalid_arg (fun () ->
      submit (empty ~uses:[| Sub.read |] d))

(* Each misuse [Submission.make] and [submit] state raises
   [Invalid_argument]. *)
let test_make_refusals () =
  let d, _ = P.open_ ~host_visible:false "submit:make-refusals" in
  let arg = B.create Rig.host 8 in
  let unseen = B.create d 8 and other = B.create d 16 in
  let dead = B.create d 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ dead ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" dead));
  let copy src dst =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let make d parts = ignore (Sub.make d parts) in
  let s = Sub.make d [| Support.touching [| Sub.read; Sub.write |] |] in
  let slot_copy = Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |] in
  let into_unseen =
    Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Buffer unseen) |]
  in
  let elsewhere = B.create (fst (P.open_ "submit:elsewhere")) 8 in
  List.iter
    (fun (msg, f) -> raises_match ~msg Exn.invalid_arg f)
    [
      ("on an io device", fun () -> make (Rig_support.io "submit:io") [||]);
      ( "a copy of buffers of two sizes",
        fun () -> make d [| copy unseen other |] );
      ( "a copy of another device's memory",
        fun () -> make d [| copy arg (B.create Rig.host 8) |] );
      ("a part's dead buffer", fun () -> make d [| copy dead unseen |]);
      ( "a copy of a slot into itself",
        fun () -> make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 0) |] );
      ( "a copy's negative slot",
        fun () -> make d Sub.[| copy ~src:(Slot (-1)) ~dst:(Buffer unseen) |] );
      ( "a slot no ref or copy names",
        fun () -> make d Sub.[| copy ~src:(Slot 1) ~dst:(Buffer unseen) |] );
      ( "a slot a copy and an unordered ref name",
        fun () ->
          make d
            Sub.
              [|
                copy ~src:(Slot 0) ~dst:(Buffer unseen);
                Support.touching [| unordered |];
              |] );
      ( "a run's copy of buffers of two sizes",
        fun () -> ignore (submit slot_copy ~buffers:[| unseen; other |]) );
      ( "a run's copy into a buffer of another size",
        fun () -> ignore (submit into_unseen ~buffers:[| other |]) );
      ( "a run's copy of the host's memory",
        fun () -> ignore (submit slot_copy ~buffers:[| arg; unseen |]) );
      ( "a run's copy of a dead buffer",
        fun () -> ignore (submit slot_copy ~buffers:[| dead; unseen |]) );
      ( "a run's read of the host's memory",
        fun () -> ignore (submit s ~buffers:[| arg; unseen |]) );
      ( "a run's write of another device's memory",
        fun () -> ignore (submit s ~buffers:[| unseen; elsewhere |]) );
      ( "a run of more buffers than made",
        fun () -> ignore (submit s ~buffers:[| unseen; unseen; unseen |]) );
      ( "a run of fewer buffers than made",
        fun () -> ignore (submit s ~buffers:[| unseen |]) );
      ( "a run's dead buffer",
        fun () -> ignore (submit s ~buffers:[| dead; unseen |]) );
    ]

(* A copy's buffers share no byte, so a driver copies between disjoint
   ranges: a copy between halves of one buffer is made, and one between
   views that share bytes is refused. *)
let test_overlapping_copy () =
  let d = memory "submit:overlapping-copy" in
  let b = B.create d 16 in
  let copy src dst =
    let view first = B.view b ~first ~length:8 in
    Sub.(copy ~src:(Buffer (view src)) ~dst:(Buffer (view dst)))
  in
  ignore (Sub.make d [| copy 0 8 |]);
  List.iter
    (fun (msg, src, dst) ->
      raises_match ~msg (Exn.invalid_arg ~substring:"overlap") (fun () ->
          Sub.make d [| copy src dst |]))
    [ ("into itself", 0, 0); ("forwards", 0, 4); ("backwards", 4, 0) ]

(* A submit that raises keeps nothing of its run: the next submit of the
   submission waits for none of its points. *)
let test_cleared_on_raise () =
  let d = memory "submit:cleared" in
  let base = Rig.submitted d in
  let producer, pp = P.open_ "submit:cleared-producer" in
  let point = submit (empty producer) in
  P.fail pp;
  (try ignore (submit (empty producer)) with Rig.Lost _ -> ());
  let s = empty ~uses:[| Sub.read |] d and b = on d (B.create Rig.host 8) in
  let run = Sub.Run.make s in
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> submit ~run s ~buffers:[| b |] ~waits:[| point |]);
  equal int (base + 1) (Rig.Point.value (submit ~run s ~buffers:[| b |]))

(* A run's buffer whose memory was consumed through another buffer refuses the
   submit. *)
let test_dead_slot () =
  let d = memory "submit:dead-slot" in
  let base = Rig.submitted d in
  let b = B.create Rig.host 8 in
  let s = empty ~uses:[| Sub.read |] d and borrowed = on d b in
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" b));
  raises_match Exn.invalid_arg (fun () -> submit s ~buffers:[| borrowed |]);
  equal int base (Rig.submitted d)

(* A submission checks the buffers it was made with: a part's buffer that died
   refuses the submit after the caller's parts array changed. *)
let test_parts_changed () =
  let d = memory "submit:parts-changed" in
  let base = Rig.submitted d in
  let a = B.create d 8 and b = B.create d 8 and dst = B.create d 8 in
  let copy src =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let parts = [| copy a |] in
  let s = Sub.make d parts in
  parts.(0) <- copy b;
  Rig.Claim.with_ ~read:[] ~donate:[ a ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" a));
  raises_match Exn.invalid_arg (fun () -> submit s);
  equal int base (Rig.submitted d)

(* A submission keeps its parts' buffers after the caller's array changes. *)
let test_parts_kept () =
  let d = memory "submit:parts-kept" in
  let collected = ref false in
  let fresh () =
    let a = B.create d 8 in
    Support.store (Support.Reader.host a) 41;
    Gc.finalise_last (fun () -> collected := true) a;
    a
  in
  let b = B.create d 8 and dst = B.create d 8 in
  Support.store (Support.Reader.host b) 42;
  let copy src =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let parts = [| copy (fresh ()) |] in
  let s = Sub.make d parts in
  parts.(0) <- copy b;
  Gc.full_major ();
  equal ~msg:"the part's buffer collected" bool false !collected;
  ignore (submit s);
  equal ~msg:"the original part's bytes" int 41 (word dst);
  equal ~msg:"the array's new buffer" int 42 (word b);
  ignore (Sys.opaque_identity s)

(* A submit whose hand-over fails has assigned its value: the stamps of the
   memory it writes name it, so a read of that memory raises the loss, and
   one of memory it only read does not. *)
let test_failed_names () =
  let d, p = P.open_ "submit:failed-names" in
  let w = B.create Rig.host page_bytes and r = B.create Rig.host page_bytes in
  let s = empty ~uses:[| Sub.read; Sub.write |] d in
  P.fail p;
  let lost = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false in
  raises_match ~msg:"the submit" lost (fun () ->
      submit s ~buffers:[| on d r; on d w |]);
  equal int ~msg:"its value" 1 (Rig.submitted d);
  raises_match ~msg:"a read of its write" lost (fun () -> B.wait w Rig.Read);
  B.wait r Rig.Read;
  raises_match ~msg:"a write of its read" lost (fun () ->
      B.wait r Rig.Read_write)

(* A read of memory another device wrote waits for that write: on the host,
   since a memory device waits on no other device in its queue. *)
let test_read_waits () =
  let producer, pp = P.open_ "submit:producer" in
  let consumer = memory "submit:consumer" in
  let b = B.create producer page_bytes in
  ignore (submit (empty ~uses:[| Sub.write |] producer) ~buffers:[| b |]);
  equal int 1 (P.queued pp);
  ignore
    (submit
       (empty ~uses:[| Sub.read |] consumer)
       ~buffers:[| on consumer b |]);
  equal int 0 (P.queued pp)

(* An [Unordered] buffer orders nothing: a submit that uses memory another
   device's queued work writes waits for none of it, that device's later write
   does not wait for the use, and neither does a wait of the host. *)
let test_unordered () =
  let a, pa = P.open_ "submit:unordered-writer" in
  let b, pb = P.open_ "submit:unordered-user" in
  let x = B.create a page_bytes in
  let write () =
    ignore (submit (empty ~uses:[| Sub.write |] a) ~buffers:[| x |])
  in
  write ();
  ignore (submit (empty ~uses:[| Sub.unordered |] b) ~buffers:[| on b x |]);
  equal ~msg:"the writer's work, after the use" int 1 (P.queued pa);
  write ();
  equal ~msg:"the use, after the next write" int 1 (P.queued pb);
  B.wait x Rig.Read_write;
  equal ~msg:"the use, after the host's wait" int 1 (P.queued pb);
  ignore (P.run pb)

(* An [Unordered] buffer stays reachable until the work is done: memory of one
   device that only such a submit on another named returns once that device ran
   the work, not before. *)
let test_unordered_keeps () =
  let a, pa = P.open_ "submit:unordered-keeps-owner" in
  let b, pb = P.open_ "submit:unordered-keeps-user" in
  (fun () ->
    let x = B.create a page_bytes in
    ignore (submit (empty ~uses:[| Sub.unordered |] b) ~buffers:[| on b x |]))
    ();
  let drain () =
    Gc.full_major ();
    Gc.full_major ();
    ignore (B.create ~memory:Pinned a 8);
    Rig.free_cache a
  in
  let frees () = List.length (List.filter (( = ) "free") (P.log pa)) in
  drain ();
  equal ~msg:"while the work is queued" int 0 (frees ());
  ignore (P.run pb);
  drain ();
  equal ~msg:"once it ran" int 1 (frees ())

(* The old buffer's queued write must not touch a later buffer, even when either
   lifetime uses Unordered access. *)
let[@inline never] dropped_bump d use =
  let b = B.create d page_bytes in
  Support.store (Support.Reader.host b) 0;
  let bump = Sub.fill ~refs:[| use 0 ~at:0 |] Support.bump ~params:8 in
  let s = Sub.make d [| bump |] in
  let run = Sub.Run.make s in
  Sub.Run.int64 run (Sub.block s 0) 0 0L;
  submit ~run s ~buffers:[| b |]

let test_cached_unordered_host () =
  let d, p = P.open_ "submit:cached-unordered-host" in
  let old = dropped_bump d Sub.unordered in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"the earlier work is pending" int 1 (P.queued p);
  let b = B.create d page_bytes in
  B.wait b Rig.Read_write;
  Support.store (Support.Reader.host b) 41;
  Rig.Point.wait old;
  equal ~msg:"the new buffer after the old work finishes" int 41 (word b)

let test_cached_unordered_foreign (name, access) =
  let a, pa = P.open_ ("submit:cached-unordered-owner:" ^ name) in
  let d, _ = P.open_ ("submit:cached-unordered-user:" ^ name) in
  let old = dropped_bump a access in
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"the earlier work is pending" int 1 (P.queued pa);
  let b = B.create a page_bytes in
  let poke =
    Sub.fill ~refs:[| Sub.unordered 0 ~at:0 |] Support.poke ~params:16
  in
  let s = Sub.make d [| poke |] in
  let run = Sub.Run.make s in
  Sub.Run.int64 run (Sub.block s 0) 8 41L;
  Rig.Point.wait (submit ~run ~buffers:[| on d b |] s);
  Rig.Point.wait old;
  equal ~msg:"the new buffer after the old work finishes" int 41 (word b)

(* A part's buffers are ordered as a run's are: a copy waits for another
   device's write of its source and for another device's read of its
   destination. *)
let test_part_points () =
  let d, _ = P.open_ "submit:parts" in
  let e, pe = P.open_ "submit:parts-other" in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let s = Sub.make d [| copy |] in
  let run = Sub.Run.make s in
  ignore (submit (empty ~uses:[| Sub.write |] e) ~buffers:[| on e src |]);
  ignore (submit ~run s ~buffers:[||]);
  equal ~msg:"after a write of the source" int 0 (P.queued pe);
  ignore (submit (empty ~uses:[| Sub.read |] e) ~buffers:[| on e dst |]);
  ignore (submit ~run s ~buffers:[||]);
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
    submit
      (empty ~uses:[| Sub.write |] d)
      ~buffers:
        (let w = B.create d n in
         at := Support.Reader.host w;
         [| w |])
      ~waits:[| a |]
  in
  let b = Domain.join other in
  equal ~msg:"another domain's buffer" bool false (Support.Reader.host b = !at);
  Rig.Point.wait run

(* A Polled device that waits on the words of producers of [completion] waits
   for a producer in its queue: the submit hands it over without waiting, and
   the wait names the producer's word. *)
let in_queue ~completion =
  let name = match completion with `Host -> "host" | `Store -> "store" in
  let producer, pp = P.open_ ~completion ("submit:iq-producer-" ^ name) in
  let consumer, cp =
    P.open_
      ~waits_on:[ (completion :> [ `Store | `Host ]) ]
      ("submit:iq-consumer-" ^ name)
  in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  equal (list (pair int int)) [ (P.word_at pp, 1) ] (P.last_waits cp);
  equal int 1 (P.queued pp);
  equal int 1 (P.queued cp);
  equal int 0 (P.run cp);
  Rig.Point.wait a;
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

(* A producer whose word the device cannot map, as a device that maps no host
   memory, is waited for on the host: the submit returns with the producer's
   work done and no wait in the queue. *)
let test_unmapped_wait () =
  let producer, pp = P.open_ "submit:unmapped-producer" in
  let consumer, cp =
    P.open_ ~peers:false ~maps_host:false ~waits_on:[ `Host ]
      "submit:unmapped-consumer"
  in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp);
  equal int (Rig.Point.value a) (Rig.signaled producer)

let test_in_queue () = in_queue ~completion:`Host
let test_in_queue_store () = in_queue ~completion:`Store

(* A Polled device that runs itself completes its work though no wait sleeps on
   it. *)
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
  Rig.Point.wait b;
  equal ~msg:"the producer's word" int (Rig.Point.value a)
    (Rig.signaled producer)

(* A device's most parts is what it always runs, not a cap: a submission of
   more parts runs where the device's queues have room for it, and only room
   refuses one they never hold. *)
let test_most_parts () =
  let d, _ = P.open_ ~capacity:4 ~most_parts:2 "submit:most-parts" in
  equal ~msg:"the device's count" int 2 (Rig.most_parts d);
  equal ~msg:"the host's" int max_int (Rig.most_parts Rig.host);
  let four = Sub.make d (Array.make 4 idle) in
  Rig.Point.wait (submit four);
  let five = Sub.make d (Array.make 5 idle) in
  raises_match ~msg:"more than the queue holds"
    (Exn.invalid_arg ~substring:"never fit") (fun () -> submit five)

(* A full queue answers Later: the submit waits for one more value. *)
let test_room () =
  let d, p = P.open_ ~capacity:1 "submit:room" in
  let base = Rig.submitted d in
  let s = Sub.make d [| idle |] in
  let run = Sub.Run.make s in
  Sub.Run.groups run (Sub.block s 0) 0 0 0;
  ignore (submit ~run s ~buffers:[||]);
  ignore (submit ~run s ~buffers:[||]);
  equal int 1 (P.queued p);
  equal int (base + 1) (Rig.signaled d)

(* Growth *)

let count call p = List.length (List.filter (( = ) call) (P.log p))

(* A device whose queue holds 4 parts and grows. *)
let growing name = P.open_ ~capacity:4 ~grows:max_int name

(* [n] parts that each add 1 to the host word [arg] borrows on [d], and their
   run. *)
let bumps d arg n =
  let s = Sub.make d (Array.make n bump) in
  let run = Sub.Run.make s in
  for i = 0 to n - 1 do
    Sub.Run.int64 run (Sub.block s i) 0 0L
  done;
  (* Some tests submit it on another domain. *)
  Sub.Run.release run;
  (s, run, [| on d arg |])

(* A submission of more parts than the queue holds runs, as one value, once the
   queue grew to hold it; one the queue holds needs no growth. *)
let test_grows n =
  let d, p = growing (Printf.sprintf "submit:grows-%d" n) in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, run, buffers = bumps d arg n in
  let pt = submit ~run ~buffers s in
  Rig.Point.wait pt;
  equal ~msg:"its value" int (base + 1) (Rig.Point.value pt);
  equal ~msg:"the parts that ran" int n (word arg);
  equal ~msg:"growths" int (if n > 4 then 1 else 0) (count "grow" p)

(* A growth the device has no memory for is asked again after each round of
   reclamation, and raises Out_of_memory after the last, having assigned no
   value; once memory returns, the submission runs. *)
let test_grow_no_memory () =
  let d, p = growing "submit:grow-no-memory" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, run, buffers = bumps d arg 5 in
  P.fail_at p 1 (`Refuse max_int);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~run ~buffers s);
  P.fail_at p 1 (`Refuse 0);
  at_least ~msg:"times asked" int ~than:2 (count "grow" p);
  equal ~msg:"values assigned" int base (Rig.submitted d);
  Rig.Point.wait (submit ~run ~buffers s);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth the device had no memory for at first is made once a round of
   reclamation is done. *)
let test_grow_reclaims () =
  let d, p = growing "submit:grow-reclaims" in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let s, run, buffers = bumps d arg 5 in
  P.before p "grow" (fun () -> P.fail_at p 1 (`Refuse 1));
  Rig.Point.wait (submit ~run ~buffers s);
  equal ~msg:"times asked" int 2 (count "grow" p);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth past the memory the device has raises Out_of_memory and leaves no
   want behind: a smaller growth after it is made at its first ask. *)
let test_grow_past_memory () =
  let d, p = P.open_ ~capacity:4 ~grows:8 "submit:grow-past-memory" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let large, run, buffers = bumps d arg 9 in
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~run ~buffers large);
  let asked = count "grow" p in
  at_least ~msg:"times asked" int ~than:2 asked;
  equal ~msg:"values assigned" int base (Rig.submitted d);
  let small, run, buffers = bumps d arg 5 in
  Rig.Point.wait (submit ~run ~buffers small);
  equal ~msg:"times asked after" int (asked + 1) (count "grow" p);
  equal ~msg:"the parts that ran" int 5 (word arg)

(* A growth short of memory for one submission fails no other: while domain
   A's growth past the memory runs, domain B's smaller growth is made at its
   first ask and B's submission runs; A raises Out_of_memory. *)
let test_grow_short_other () =
  let d, p = P.open_ ~capacity:4 ~grows:8 "submit:grow-short-other" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let large, run, buffers = bumps d arg 9 in
  let small, small_run, small_buffers = bumps d arg 5 in
  let other = ref None and asked = ref 0 in
  P.before p "grow" (fun () ->
      let submits () =
        Rig.Point.value (submit ~run:small_run ~buffers:small_buffers small)
      in
      other := Some (Domain.join (Domain.spawn submits));
      asked := count "grow" p);
  raises_match
    (function
      | Rig.Out_of_memory (d', n) -> Rig.equal d d' && n = 4096 | _ -> false)
    (fun () -> submit ~run ~buffers large);
  equal ~msg:"the other domain's value" (option int) (Some (base + 1)) !other;
  equal ~msg:"growths asked until B's ran" int 2 !asked;
  B.wait arg Rig.Read;
  equal ~msg:"the parts that ran" int 5 (word arg)

(* While a growth runs, holding no turn, a submission of another domain that
   needs a growth too takes the device's next value; the first submission
   takes the one after, and every part runs once. *)
let test_grow_interleaved () =
  let d, p = growing "submit:grow-interleaved" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  Support.store (Support.Reader.host arg) 0;
  let first, run, buffers = bumps d arg 9 in
  let second, other_run, other_buffers = bumps d arg 5 in
  let other = ref None in
  P.before p "grow" (fun () ->
      let submits () =
        Rig.Point.value (submit ~run:other_run ~buffers:other_buffers second)
      in
      other := Some (Domain.join (Domain.spawn submits)));
  let v = Rig.Point.value (submit ~run ~buffers first) in
  equal ~msg:"the other domain's value" (option int) (Some (base + 1)) !other;
  equal ~msg:"its value" int (base + 2) v;
  B.wait arg Rig.Read;
  equal ~msg:"the parts that ran" int 14 (word arg);
  equal ~msg:"growths" int 2 (count "grow" p)

(* A growth that faults loses the device, and the submission assigns no
   value. *)
let test_grow_fault () =
  let d, p = growing "submit:grow-fault" in
  let base = Rig.submitted d in
  let arg = B.create Rig.host page_bytes in
  let s, run, buffers = bumps d arg 5 in
  P.before p "grow" (fun () -> P.fail_at p 1 (`Fault "grow fault"));
  raises_match lost (fun () -> submit ~run ~buffers s);
  equal ~msg:"values assigned" int base (Rig.submitted d);
  equal (option string) ~msg:"the loss" (Some "grow fault") (Rig.lost d)

(* Runs *)

(* While a submit of a submission waits on the host for a producer, a submit of
   the same submission on another domain, with its own run, completes. *)
let test_run_beside_wait () =
  let producer, pp = P.open_ "submit:beside-producer" in
  let d, p = P.open_ "submit:beside" in
  let base = Rig.submitted d in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let part =
    Sub.launch (require_some (Rig.Image.resolve image "main")) ~params:8
  in
  let s = Sub.make d [| part |] and point = submit (empty producer) in
  let block = Sub.block s 0 in
  let initialized value =
    let run = Sub.Run.make s in
    Sub.Run.groups run block 1 1 1;
    Sub.Run.threads run block 1 1 1;
    Sub.Run.int64 run block 0 (Int64.of_int value);
    Sub.Run.release run;
    run
  in
  let first = initialized 41 and second = initialized 42 in
  P.gate pp;
  let waiting =
    Domain.spawn (fun () -> submit ~run:first s ~waits:[| point |] ~buffers:[||])
  in
  let finished = ref None in
  Fun.protect
    ~finally:(fun () ->
      P.open_gate pp;
      finished := Some (Domain.join waiting))
    (fun () ->
      Support.await "a sleep at the gate" (fun () -> P.sleepers pp = 1);
      equal ~msg:"the submit beside the wait" int (base + 1)
        (Rig.Point.value (submit ~run:second s ~buffers:[||])));
  let finished = Option.get !finished in
  equal ~msg:"the waiting submit" int (base + 2) (Rig.Point.value finished);
  Rig.Point.wait finished;
  equal ~msg:"each launch reads its run's parameter" (list int) [ 42; 41 ]
    (List.map
       (fun (launch : P.launch) ->
         Int64.to_int (String.get_int64_le launch.params 0))
       (P.launches p))

(* A submit made inside another's wait, as a signal handler runs, completes with
   its own run, and raises with the run the waiting submit holds. *)
let test_run_reentrant () =
  let producer, pp = P.open_ "submit:reentrant-producer" in
  let d, _ = P.open_ "submit:reentrant" in
  let base = Rig.submitted d in
  let s = Sub.make d [| idle |] and point = submit (empty producer) in
  let held = Sub.Run.make s in
  let inner = ref None and refused = ref None in
  let handle _ =
    inner := Some (Rig.Point.value (submit s));
    refused :=
      Some
        (match submit ~run:held s ~buffers:[||] with
        | _ -> "submitted"
        | exception Invalid_argument why -> why)
  in
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let outer =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> submit ~run:held s ~waits:[| point |] ~buffers:[||])
  in
  equal ~msg:"the submit inside the wait" (option int) (Some (base + 1)) !inner;
  equal ~msg:"the waiting submit" int (base + 2) (Rig.Point.value outer);
  is_some ~msg:"a submit with the held run raised" !refused;
  contains ~msg:"its reason" ~sub:"another submit is using the run"
    (Option.get !refused)

(* A run of a submission with no launch or fill holds no bytes: a submit of it
   inside another submit's wait of it, as a signal handler runs, completes. *)
let test_bare_reentrant () =
  let producer, pp = P.open_ "submit:bare-producer" in
  let d, _ = P.open_ "submit:bare" in
  let base = Rig.submitted d in
  let run = Sub.Run.make (empty d) and point = submit (empty producer) in
  let inner = ref None in
  let handle _ =
    inner := Some (Rig.Point.value (Rig.submit run ~buffers:[||]))
  in
  P.interrupt pp;
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let outer =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> Rig.submit run ~waits:[| point |] ~buffers:[||])
  in
  equal ~msg:"the submit inside the wait" (option int) (Some (base + 1)) !inner;
  equal ~msg:"the waiting submit" int (base + 2) (Rig.Point.value outer);
  equal ~msg:"take" bool true (Sub.Run.take run)

(* A submit claims its buffers while it runs: a donation inside its wait is no
   write in place, one before or after it is, and a buffer held exclusive is
   refused. *)
let test_claims () =
  let producer, pp = P.open_ "submit:claims-producer" in
  let d, _ = P.open_ "submit:claims" in
  let buffers = Array.init 5 (fun _ -> B.create d 64) in
  let s = empty ~uses:(Array.make 5 Sub.read) d in
  let point = submit (empty producer) in
  let donated () =
    let donate = Array.to_list buffers in
    Rig.Claim.with_ ~read:[] ~donate (fun c ->
        Array.to_list (Array.map (Rig.Claim.exclusive c) buffers))
  in
  let exclusive = List.init 5 (fun _ -> true) in
  equal ~msg:"before the submit" (list bool) exclusive (donated ());
  let during = ref None in
  P.interrupt pp;
  let before =
    Sys.signal Sys.sigint
      (Sys.Signal_handle (fun _ -> during := Some (donated ())))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () -> ignore (submit s ~buffers ~waits:[| point |]));
  equal ~msg:"inside the submit's wait"
    (option (list bool))
    (Some (List.init 5 (fun _ -> false)))
    !during;
  equal ~msg:"after the submit" (list bool) exclusive (donated ());
  Array.iteri
    (fun i m ->
      Rig.Claim.with_ ~read:[] ~donate:[ m ] (fun c ->
          equal ~msg:"held exclusive" bool true (Rig.Claim.exclusive c m);
          raises_match ~msg:"a submit of it"
            (Exn.invalid_arg ~substring:"held exclusive") (fun () ->
              submit s ~buffers));
      equal
        ~msg:(Printf.sprintf "after refusal at buffer %d" i)
        (list bool) exclusive (donated ()))
    buffers

(* A copy on a device that runs no copies is refused where the caller can act:
   when the submission is made. *)
let test_copy_refused () =
  let d, _ = P.open_ ~copies:false "submit:no-copies" in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  raises_match Exn.invalid_arg (fun () -> Sub.make d [| copy |])

(* A part no queue of its device runs is refused when the submission is made:
   the host runs no words, and Polled with no copy queue no copy. *)
let test_runs () =
  raises_match ~msg:"words where no queue takes them" Exn.invalid_arg
    (fun () -> Sub.make Rig.host [| Sub.words (B.create Rig.host 8) |]);
  let e, _ = P.open_ ~copies:false "submit:runs-fills" in
  let src = B.create e 8 and dst = B.create e 8 in
  raises_match ~msg:"a copy where no queue copies" Exn.invalid_arg (fun () ->
      Sub.make e [| Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) |]);
  let image = Result.get_ok (Rig.Image.load e "code:64") in
  let main = require_some (Rig.Image.resolve image "main") in
  ignore (Sub.make e [| Sub.launch main ~params:0 |])

(* Words, whose effect rig cannot see, follow every earlier part; a later part
   follows them by running on their queue or by sharing memory with them.
   Polled's COMPUTE:0 (queue 0) runs launches, fills and words, COPY:0 (queue
   1) is its copy engine:
   - 0, a copy a -> b: on the copy engine;
   - 1, words reading ws: on COMPUTE:0, the queue of launches, after 0, though
     they share no memory;
   - 2, a fill: on COMPUTE:0, after the words by that queue's order;
   - 3, a copy c -> e: on the copy engine, concurrent with the words;
   - 4, a copy e -> ws: it follows 3 by e, and the words by ws, so it runs on
     3's queue after the words. *)
let test_words_order () =
  let d, p = P.open_ "submit:words-order" in
  let ws = B.create Rig.host page_bytes in
  let a = B.create d 8 and b = B.create d 8 in
  let c = B.create d page_bytes and e = B.create d page_bytes in
  let copy src dst = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  let parts =
    [|
      copy a b;
      Sub.words ws;
      Sub.fill Support.noop ~params:0;
      copy c e;
      copy e (on d ws);
    |]
  in
  Rig.Point.wait (submit (Sub.make d parts));
  equal
    (list (pair int (list int)))
    [ (1, []); (0, [ 0 ]); (0, []); (1, []); (1, [ 1 ]) ]
    (P.layout p)

(* Words whose buffer holds a part of a word are refused by length, when the
   part is made. *)
let test_part_word () =
  List.iter
    (fun n ->
      raises_match
        ~msg:(Printf.sprintf "%d bytes" n)
        (Exn.invalid_arg ~substring:(Printf.sprintf "of %d bytes" n))
        (fun () -> Sub.words (B.create Rig.host n)))
    [ 1; 6; 7 ]

(* Fills of 4,096 bytes of parameters: as many as a run holds, each counting
   16 bytes plus its parameters against 2^31 - 1, are made, and one more is
   refused. *)
let test_blocks_bound () =
  let d, _ = P.open_ "submit:blocks-bound" in
  let params = 4096 in
  let most = ((1 lsl 31) - 1) / (16 + params) in
  let fills n = Array.make n (Sub.fill Support.noop ~params) in
  ignore (Sub.make d (fills most));
  raises_match
    (Exn.invalid_arg ~substring:"take more than a run holds")
    (fun () -> Sub.make d (fills (most + 1)))

(* [make]'s stack does not grow with its parts: 2^20 fills make within a stack
   of 2^16 words, whatever the parts' count. *)
let test_parts_stack () =
  let d, _ = P.open_ "submit:parts-stack" in
  let parts = Array.make (1 lsl 20) (Sub.fill Support.noop ~params:0) in
  let gc = Gc.get () in
  Gc.set { gc with stack_limit = 1 lsl 16 };
  let s =
    Fun.protect ~finally:(fun () -> Gc.set gc) (fun () -> Sub.make d parts)
  in
  ignore (Sys.opaque_identity s)

(* Order *)

(* A case of the order law: parts on a Polled device, copies between views of
   three allocations, fills whose refs name slots, words, and fixed memory.
   The allocations are two of the device's, which the host addresses only with
   [host_visible], and one of the host's, borrowed. A slot's refs are all of
   its kind. Each words part reads a host buffer of its own. *)
type view = { alloc : int; first : int; length : int }
type use = Read | Write | Unordered
type order_part = Copy of view * view | Fill of int list | Words

type order_case = {
  host_visible : bool;
  kinds : use array;  (** Each slot's refs' kind. *)
  parts : order_part list;
  fixed : (view * Rig.access) list;
}

let pp_view ppf v = Format.fprintf ppf "%d[%d+%d]" v.alloc v.first v.length

let pp_order_case ppf c =
  let kind = function Read -> "r" | Write -> "w" | Unordered -> "u" in
  let part ppf = function
    | Copy (s, d) -> Format.fprintf ppf "copy %a->%a" pp_view s pp_view d
    | Words -> Format.fprintf ppf "words"
    | Fill slots ->
        Format.fprintf ppf "fill [%s]"
          (String.concat ";"
             (List.map (fun k -> string_of_int k ^ kind c.kinds.(k)) slots))
  in
  Format.fprintf ppf "%sparts [%a]; fixed [%a]"
    (if c.host_visible then "" else "unaddressed; ")
    (Format.pp_print_list ~pp_sep:Format.pp_print_space part)
    c.parts
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (v, a) ->
         Format.fprintf ppf "%a%s" pp_view v (if a = Rig.Read then "r" else "w")))
    c.fixed

let overlap v v' =
  v.alloc = v'.alloc && v.length > 0 && v'.length > 0
  && v.first < v'.first + v'.length
  && v'.first < v.first + v.length

let nallocs = 3
let alloc_bytes = 64

let gen_order_case =
  let open Gen in
  (* Views start and end on 8-byte steps, so that views often share or abut
     their bounds; empty views overlap nothing. *)
  let step = 8 in
  let view length =
    let+ alloc = int_range 0 (nallocs - 1)
    and+ first =
      map (( * ) step) (int_range 0 ((alloc_bytes - length) / step))
    in
    { alloc; first; length }
  in
  let length = map (( * ) step) (int_range 0 (alloc_bytes / step)) in
  (* Fills of unordered slots and read-only fixed memory leave copies free of
     them, so that a copy on the copy engine can follow one on a fill's queue. *)
  let* kinds =
    array ~size:(constant 3)
      (frequency
         [ (1, constant Read); (1, constant Write); (3, constant Unordered) ])
  in
  let part =
    one_of
      [
        (* A copy's buffers share no byte (Submission.copy): a [dst] that
           would share bytes with [src] moves to the next allocation. *)
        (let* length = length in
         let+ src = view length and+ dst = view length in
         let dst =
           if overlap src dst then
             { dst with alloc = (dst.alloc + 1) mod nallocs }
           else dst
         in
         Copy (src, dst));
        (let+ slots = list ~size:(int_range 0 2) (int_range 0 2) in
         Fill (List.sort_uniq Int.compare slots));
        constant Words;
      ]
  in
  let fixed =
    let* length = length in
    let+ v = view length
    and+ a = frequency [ (3, constant Rig.Read); (1, constant Rig.Read_write) ] in
    (v, a)
  in
  let+ host_visible = bool
  and+ parts = list ~size:(int_range 1 40) part
  and+ fixed =
    list
      ~size:(frequency [ (1, constant 0); (2, constant 1); (1, constant 3) ])
      fixed
  in
  { host_visible; kinds; parts; fixed }

(* The case's slots, renumbered from 0 in their order of use, so that every
   slot is a ref's. *)
let slot_order c =
  List.fold_left
    (fun acc -> function
      | Fill slots ->
          List.fold_left
            (fun acc k -> if List.mem k acc then acc else acc @ [ k ])
            acc slots
      | Copy _ | Words -> acc)
    [] c.parts

(* What a part uses, as Submission.make states it: [`View] a buffer it names,
   [`Slot] a run's buffer, with whether it writes. *)
let uses c p =
  let fixed = List.map (fun (v, a) -> (`View v, a = Rig.Read_write)) c.fixed in
  match p with
  | Copy (src, dst) -> (`View src, false) :: (`View dst, true) :: fixed
  | Words -> fixed
  | Fill slots ->
      List.filter_map
        (fun k ->
          match c.kinds.(k) with
          | Unordered -> None
          | Read -> Some (`Slot, false)
          | Write -> Some (`Slot, true))
        slots
      @ fixed

let conflict c p p' =
  List.exists
    (fun (u, w) ->
      List.exists
        (fun (u', w') ->
          (w || w')
          &&
          match (u, u') with
          | `View v, `View v' -> overlap v v'
          | _ -> true)
        (uses c p'))
    (uses c p)

(* The layout the order law states: a fill on queue 0, the first that runs it;
   words on queue 0, the first that runs launches and words; a copy on the
   queue of the last earlier part it follows, else on the copy engine, queue
   1; and per other queue, the last earlier part it follows. A part follows
   the earlier parts that may share memory with it, and words every earlier
   part. *)
let expected_layout c =
  let parts = Array.of_list c.parts in
  let queues = Array.make (Array.length parts) 0 in
  Array.mapi
    (fun j p ->
      let follows i = p = Words || conflict c parts.(i) p in
      let earlier = List.init j Fun.id in
      let q =
        match p with
        | Fill _ | Words -> 0
        | Copy _ -> (
            match List.filter follows (List.rev earlier) with
            | i :: _ -> queues.(i)
            | [] -> 1)
      in
      queues.(j) <- q;
      let after =
        List.filter_map
          (fun q' ->
            if q' = q then None
            else
              List.fold_left
                (fun last i -> if queues.(i) = q' && follows i then Some i else last)
                None earlier)
          [ 0; 1 ]
      in
      (q, List.sort Int.compare after))
    parts
  |> Array.to_list

let order_law c =
  let d, p =
    P.open_ ~host_visible:c.host_visible
      (Printf.sprintf "submit:order-%d" (Random.bits ()))
  in
  let host =
    B.view (on d (B.create Rig.host page_bytes)) ~first:0 ~length:alloc_bytes
  in
  let allocs = [| B.create d alloc_bytes; B.create d alloc_bytes; host |] in
  let buffer v = B.view allocs.(v.alloc) ~first:v.first ~length:v.length in
  let slots = slot_order c in
  let index k =
    let rec go i = function
      | [] -> assert false
      | k' :: rest -> if k = k' then i else go (i + 1) rest
    in
    go 0 slots
  in
  let ref_of i k =
    let slot = index k and at = 8 * i in
    match c.kinds.(k) with
    | Read -> Sub.read slot ~at
    | Write -> Sub.write slot ~at
    | Unordered -> Sub.unordered slot ~at
  in
  let part = function
    | Copy (src, dst) ->
        Sub.(copy ~src:(Buffer (buffer src)) ~dst:(Buffer (buffer dst)))
    | Fill ks ->
        Sub.fill
          ~refs:(Array.of_list (List.mapi ref_of ks))
          Support.noop
          ~params:(8 * List.length ks)
    | Words -> Sub.words (B.create Rig.host 8)
  in
  let fixed = List.map (fun (v, a) -> (buffer v, a)) c.fixed in
  let s = Sub.make ~fixed d (Array.of_list (List.map part c.parts)) in
  let buffers = Array.of_list (List.map (fun _ -> B.create d 8) slots) in
  Rig.Point.wait (submit s ~buffers);
  let expected = expected_layout c in
  let copy_after_fill =
    List.exists
      (fun (q, after) -> q = 1 && after <> [])
      expected
  in
  cover "a copy on the copy engine after a fill" copy_after_fill;
  cover "a copy on a fill's queue"
    (List.exists2
       (fun p (q, _) -> match p with Copy _ -> q = 0 | Fill _ | Words -> false)
       c.parts expected);
  let parts = Array.of_list c.parts in
  cover "words after a part of another queue"
    (List.exists2
       (fun p (_, after) -> p = Words && after <> [])
       c.parts expected);
  cover "a copy concurrent with earlier words"
    (List.exists
       (fun (j, (q, after)) ->
         q = 1
         && List.exists
              (fun i -> parts.(i) = Words && not (List.mem i after))
              (List.init j Fun.id))
       (List.mapi (fun j x -> (j, x)) expected));
  cover "an unordered slot"
    (List.exists
       (function
         | Fill ks -> List.exists (fun k -> c.kinds.(k) = Unordered) ks
         | Copy _ | Words -> false)
       c.parts);
  let views =
    List.concat_map
      (function Copy (s, d) -> [ s; d ] | Fill _ | Words -> [])
      c.parts
    @ List.map fst c.fixed
  in
  cover "two views that share some bytes and differ"
    (List.exists
       (fun v -> List.exists (fun v' -> v <> v' && overlap v v') views)
       views);
  equal
    (list (pair int (list int)))
    ~msg:"each part's queue and after" expected (P.layout p);
  ignore (Sys.opaque_identity allocs)

(* A value's work starts once the previous value's completed, on every queue: a
   copy on COPY:0, then a fill on COMPUTE:0 that reads what the copy wrote with
   unordered slots, sees the copy, and the other way round. Only the device's
   order orders them. *)
let test_device_order () =
  let d, p = P.open_ "submit:device-order" in
  (* Host buffers of 64 KiB start on a page, so the device borrows them. *)
  let n = 64 * 1024 in
  let filled c =
    let b = B.create Rig.host n in
    B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a c);
    b
  in
  let a = filled 'a' and b = B.create d n in
  let c = filled '-' and out = filled '-' and z = filled 'z' in
  let bytes x = B.read x Bigarray.char (fun a -> String.init n (Bigarray.Array1.get a)) in
  let fill dst src =
    let part =
      Sub.fill
        ~refs:[| Sub.unordered 0 ~at:0; Sub.unordered 1 ~at:8 |]
        Support.carry ~params:24
    in
    let s = Sub.make d [| part |] in
    let run = Sub.Run.make s and block = Sub.block s 0 in
    Sub.Run.int64 run block 0 0L;
    Sub.Run.int64 run block 8 0L;
    Sub.Run.int64 run block 16 (Int64.of_int n);
    ignore (submit ~run ~buffers:[| on d dst; on d src |] s)
  in
  let copy src dst =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let run parts = ignore (submit (Sub.make d parts)) in
  let a' = require_some (B.borrow d a) and c' = require_some (B.borrow d c) in
  run [| copy a' b |];
  fill out b;
  ignore (P.run p);
  equal ~msg:"a fill after a copy" string (String.make n 'a') (bytes out);
  fill b z;
  run [| copy b c' |];
  ignore (P.run p);
  equal ~msg:"a copy after a fill" string (String.make n 'z') (bytes c)

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

type device = {
  d : Rig.t;
  p : P.t;
  base : int;  (** The value its word held when it opened. *)
  mutable watched : watched list;
  mutable last_point : Rig.Point.t option;
}

type submission = {
  dev : device;
  mutable run : Sub.Run.t option;  (* Which keeps its submission. *)
  parts : watched;
}

type device_model = { mutable value : int }
type submission_model = { model : device_model; mutable live : bool }

let devices = Atomic.make 0

let open_device () =
  let n = Atomic.fetch_and_add devices 1 in
  let d, p = P.open_ (Printf.sprintf "submit:lifetime-%d" n) in
  { d; p; base = Rig.submitted d; watched = []; last_point = None }

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
  let parts = watch t ~last:0 [ handle src; handle dst ] in
  let copy =
    Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))
  in
  let s = Sub.make t.d [| copy; Support.touching [| Sub.write |] |] in
  { dev = t; run = Some (Sub.Run.make s); parts }

let submit_sub sub =
  let t = sub.dev in
  let run = Option.get sub.run in
  let out = B.create t.d 256 in
  let at = handle out in
  let pt = Rig.submit run ~buffers:[| out |] in
  t.last_point <- Some pt;
  let v = Rig.Point.value pt in
  sub.parts.last <- v;
  ignore (watch t ~dropped:(frees t) ~last:v [ at ]);
  v - t.base

let drop sub =
  sub.run <- None;
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
      (fun t -> Option.iter Rig.Point.wait t.last_point);
  ]

let test_allocation () =
  let d = memory "submit:words" in
  let bytes_of_submits s buffers =
    let run = Sub.Run.make s in
    ignore (Rig.submit run ~buffers);
    Support.allocated_bytes (fun () ->
        for _ = 1 to 100 do
          ignore (Sys.opaque_identity (Rig.submit run ~buffers))
        done)
  in
  let fixed = B.create d 8 in
  List.iter
    (fun (msg, s, buffers) -> equal ~msg int 0 (bytes_of_submits s buffers))
    [
      ( "a fill's slot",
        empty ~uses:[| Sub.read |] d,
        [| on d (B.create Rig.host 8) |] );
      ( "a copy's slots",
        Sub.make d Sub.[| copy ~src:(Slot 0) ~dst:(Slot 1) |],
        [| B.create d 8; B.create d 8 |] );
      ( "a copy from a buffer into a slot",
        Sub.make d Sub.[| copy ~src:(Buffer fixed) ~dst:(Slot 0) |],
        [| B.create d 8 |] );
    ]

(* Two domains submitting one submission: the submits take turns, each takes the
   device's next value, and every fill runs. A program counts values from
   [base], its device's value when it began. *)
type shared = {
  dev : Rig.t;
  base : int;
  arg : B.t;
  buffers : B.t array;
  sub : Sub.t;
  runs : Sub.Run.t list Atomic.t;
      (* The runs of [sub], as a library keeps them. *)
}

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
  Support.store (Support.Reader.host arg) 0;
  {
    dev = d;
    base = Rig.submitted d;
    arg;
    buffers = [| on d arg |];
    sub = Sub.make d [| bump; bump |];
    runs = Atomic.make [];
  }

(* Every submit ran both its fills. *)
let release_shared t =
  equal ~msg:"fills" int (2 * (Rig.submitted t.dev - t.base)) (word t.arg);
  Mutex.protect shared_pool (fun () -> shared_free := t.dev :: !shared_free)

let shared =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "values %d" r.values)
    ~release:release_shared "s"

(* A run of [t.sub] that the calling thread took: the first of [t.runs] whose
   take is true, or a fresh one added. *)
let rec add_run t r =
  let l = Atomic.get t.runs in
  if not (Atomic.compare_and_set t.runs l (r :: l)) then add_run t r

let take_run t =
  match List.find_opt Sub.Run.take (Atomic.get t.runs) with
  | Some r -> r
  | None ->
      let r = Sub.Run.make t.sub in
      add_run t r;
      r

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
        let run = take_run t in
        match
          Sub.Run.int64 run (Sub.block t.sub 0) 0 0L;
          Sub.Run.int64 run (Sub.block t.sub 1) 0 0L
        with
        | () -> Rig.Point.value (submit ~run ~buffers:t.buffers t.sub) - t.base
        | exception e ->
            Sub.Run.release run;
            raise e);
  ]

(* Two domains copying on one device at once, each between buffers of its
   own: through one run of a submission whose copy's ends are its two slots,
   which holds no bytes, and through Buffer.copy, which shares what it
   prepares for the device between domains.
   Each copy's destination ends with its source's word. The host addresses no
   memory of the device, so its copy engine runs each copy between two of its
   buffers. *)
let slot_copier =
  let copier =
    lazy
      (let d, _ = P.open_ ~host_visible:false "submit:slot-copier" in
       (d, Sub.(Run.make (make d [| copy ~src:(Slot 0) ~dst:(Slot 1) |]))))
  in
  (* Two domains may make the first call at once, and a lazy value forced
     from two domains at once raises [CamlinternalLazy.Undefined]. *)
  let lock = Mutex.create () in
  fun () -> Mutex.protect lock (fun () -> Lazy.force copier)

(* A buffer of [d] holding [v], and one to copy it into. *)
let copy_ends d v =
  let src = B.create d 8 and dst = B.create d 8 in
  let h = B.create Rig.host 8 in
  Support.store (Support.Reader.host h) v;
  B.copy ~src:h ~dst:src;
  (src, dst)

(* The word of [b], a buffer of a device. *)
let device_word b =
  let h = B.create Rig.host 8 in
  B.copy ~src:b ~dst:h;
  word h

let copy_commands =
  [
    command "submit"
      (Gen.int @-> returns int)
      Fun.id
      (fun v ->
        let d, run = slot_copier () in
        let src, dst = copy_ends d v in
        Rig.Point.wait (Rig.submit run ~buffers:[| src; dst |]);
        device_word dst);
    command "Buffer.copy"
      (Gen.int @-> returns int)
      Fun.id
      (fun v ->
        let d, _ = slot_copier () in
        let src, dst = copy_ends d v in
        B.copy ~src ~dst;
        device_word dst);
  ]

let tests =
  [
    group ~timeout "copy ends"
      [
        prop ~count:200
          "a copy between slots and buffers copies as the host does, and \
           refuses buffers of two lengths or that share a byte"
          (Gen.list ~size:(Gen.int_range 1 4) (Gen.with_pp pp_ends gen_ends))
          ends_law;
      ];
    group ~timeout "copy memory"
      [
        test "empty allocations name no device memory" (fun () ->
            test_copy_memory (0, 0, 0));
        prop ~count:300
          "a part follows the earlier parts that may share memory with it, \
           on its queue or by its after"
          (Gen.with_pp pp_order_case gen_order_case)
          order_law;
        prop "views and borrows preserve copy allocation facts"
          ~examples:[ (0, 1, 2); (1, 0, 0); (17, 3, 7) ]
          (Gen.triple (Gen.int_range 0 64) (Gen.int_range 0 7)
             (Gen.int_range 0 7))
          copy_memory_law;
        test "copies carry both facts across memory kinds"
          test_copy_memory_sequence;
      ];
    group ~timeout "values"
      [
        test "values follow one another from 1" test_values;
        test "a submission's fills run with its value" test_fill;
        stateful ~domains:2 "two domains submit one submission at once"
          shared_commands;
        stateful ~domains:2
          "two domains copy at once through one run, each between buffers of \
           its own"
          copy_commands;
        test "a device's work completes once a wait reaches it" test_polled;
        test "a word still for three intervals loses nothing" test_still;
      ];
    group ~timeout "hangs"
      [
        test "work still past the hang bound loses the device" test_hang;
        test "a word moving within the bound loses nothing" test_hang_moving;
        test "an idle device loses nothing to the bound" test_hang_idle;
        test "work nothing waits for loses nothing to the bound"
          test_hang_unwatched;
        test "a queue waiting on a hung device is lost with it" test_hang_spread;
        test "a queue waiting on longer work elsewhere loses nothing"
          test_hang_in_queue;
        test "a word that moved while a wait was held loses nothing"
          test_hang_held_read;
      ];
    group ~timeout "runs"
      [
        test "a submit beside another's wait completes with its own run"
          test_run_beside_wait;
        test "a submit inside another's wait completes with its own run"
          test_run_reentrant;
        test "a run that holds no bytes serves a submit inside its own wait"
          test_bare_reentrant;
      ];
    group ~timeout "refusals"
      [
        test "a submission refuses what it cannot run" test_refusals;
        test "a submission refuses each misuse its interface states"
          test_make_refusals;
        test "a submission refuses a copy whose buffers overlap"
          test_overlapping_copy;
        test "a submit that raises keeps nothing of its run"
          test_cleared_on_raise;
        test "a run's buffer that died refuses the submit" test_dead_slot;
        test "a submit claims its buffers while it runs" test_claims;
        test
          "a part's buffer that died refuses the submit after its array changed"
          test_parts_changed;
        test "a submission keeps its parts' buffers after their array changed"
          test_parts_kept;
        test "a copy on a device that runs no copies is refused"
          test_copy_refused;
        test "a part of a kind its queue does not run is refused at make"
          test_runs;
        test "words follow every earlier part, and later parts of their queue"
          test_words_order;
        test "words of a length no multiple of 4 are refused"
          test_part_word;
        test "launches and fills past what a run holds are refused at make"
          test_blocks_bound;
      ];
    group ~timeout "order"
      [
        test "a value's work follows the previous value's, on every queue"
          test_device_order;
        test "a read waits for another device's write" test_read_waits;
        test "a submit its driver fails to take names its memory"
          test_failed_names;
        test "an Unordered buffer orders no work" test_unordered;
        test "an Unordered buffer stays reachable until its work is done"
          test_unordered_keeps;
        test "cached Unordered work cannot change a new buffer's host write"
          test_cached_unordered_host;
        cases "cached work cannot change a new buffer's Unordered write"
          ~name:fst
          [
            ("after ordered work", Sub.write);
            ("after Unordered work", Sub.unordered);
          ]
          test_cached_unordered_foreign;
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
        test "a queue that waits on stores waits on the producer's word"
          test_in_queue_store;
        test "a submission of more than the most parts runs where room has it"
          test_most_parts;
        test "a full queue's submit waits for room" test_room;
      ];
    group ~timeout "growth"
      [
        cases ~name:(Printf.sprintf "%d parts on a queue of 4 that grows")
          "a submission runs once its queue grew to hold it" [ 1; 4; 5; 9; 300 ]
          test_grows;
        test "a growth without memory raises Out_of_memory after reclamation"
          test_grow_no_memory;
        test "a growth short at first is made after a round of reclamation"
          test_grow_reclaims;
        test "a growth past the memory leaves no want behind"
          test_grow_past_memory;
        test "a submission of another domain runs while a growth runs"
          test_grow_interleaved;
        test "a growth short for one submission fails no other"
          test_grow_short_other;
        test "a growth that faults loses the device" test_grow_fault;
      ];
    group ~timeout "lifetime"
      [
        stateful "memory a submission names returns after its last submit"
          lifetime;
      ];
    group ~timeout "cost"
      [
        test "a submit that does not wait allocates nothing" test_allocation;
        test "make takes no stack per part" test_parts_stack;
      ];
  ]

let () = exit (run "rig.submit" tests)
