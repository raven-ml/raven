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
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make d [||]
let submit s = Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[||]

(* A drain on [d]: what {!Buffer.create} does first, also on a lost [d],
   where it then raises. *)
let drain d =
  match B.create d 8 with
  | b -> ignore (Sys.opaque_identity b)
  | exception Rig.Lost (d', _) when Rig.equal d d' -> ()

(* A value whose collection counts in [runs]. *)
let counted runs =
  let v = ref () in
  Gc.finalise (fun _ -> Atomic.incr runs) v;
  v

(* A drain on [d], then a collection: a value the drain lets go of is
   collected. *)
let released d =
  drain d;
  Gc.full_major ()

(* Submits once on [d] a submission that keeps a value whose collection counts
   in [runs], and drops both: the submission is unreachable once this
   returns. *)
let[@inline never] submit_kept d runs =
  Rig.Point.value (submit (Sub.make ~keep:(counted runs) d [||]))

(* Releases *)

(* A value two submissions keep, on two devices, is let go once both are
   done. *)
let test_two_devices () =
  let d, pd = P.open_ "keep:first" and e, pe = P.open_ "keep:second" in
  let runs = Atomic.make 0 in
  (fun () ->
    let v = counted runs in
    let on x = Sub.make ~keep:v x [||] in
    ignore (submit (on d));
    ignore (submit (on e)))
    ();
  Gc.full_major ();
  ignore (P.run pd);
  drain d;
  released e;
  equal ~msg:"with one device done" int 0 (Atomic.get runs);
  ignore (P.run pe);
  released e;
  equal ~msg:"with both done" int 1 (Atomic.get runs)

let test_release () =
  let d, p = P.open_ "keep:release" in
  let runs = Atomic.make 0 in
  ignore (submit_kept d runs);
  Gc.full_major ();
  equal ~msg:"after a collection" int 0 (Atomic.get runs);
  released d;
  equal ~msg:"after a drain before its work is done" int 0 (Atomic.get runs);
  ignore (P.run p);
  released d;
  equal ~msg:"after a drain once its work is done" int 1 (Atomic.get runs);
  released d;
  equal ~msg:"after the next drain" int 1 (Atomic.get runs)

(* A submission that never ran lets go of its value once unreachable. *)
let test_release_unsubmitted () =
  let runs = Atomic.make 0 in
  (fun () ->
    let s = Sub.make ~keep:(counted runs) Rig.host [||] in
    ignore (Sys.opaque_identity s))
    ();
  Gc.full_major ();
  released Rig.host;
  equal int 1 (Atomic.get runs)

(* A submission keeps its value reachable until its work is done, and no
   longer. *)
let[@inline never] keep_bytes d w =
  let v = Bytes.of_string "kept" in
  Weak.set w 0 (Some v);
  ignore (submit (Sub.make ~keep:v d [||]))

let test_keeps_value () =
  let d, p = P.open_ "keep:keeps" in
  let w = Weak.create 1 in
  keep_bytes d w;
  Gc.full_major ();
  drain d;
  equal ~msg:"while its work is undone" bool true (Weak.check w 0);
  ignore (P.run p);
  drain d;
  Gc.full_major ();
  equal ~msg:"once its work is done" bool false (Weak.check w 0)

(* On a lost device a submission keeps its value until the stop's answer and
   the word reaching the last value: an Unknown answer with a word short of it
   keeps the value. *)
let test_release_lost () =
  let d, p = P.open_ ~answer:`Unknown "keep:lost" in
  let runs = Atomic.make 0 in
  ignore (submit_kept d runs);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  equal int 1 (count "stop" p);
  Gc.full_major ();
  released d;
  equal ~msg:"with the word short" int 0 (Atomic.get runs);
  P.set_word p (Rig.submitted d);
  released d;
  equal ~msg:"once the word drained" int 1 (Atomic.get runs)

(* A drain that finds a kept value's device behind a transport faulted when it
   reads its word loses the device and returns: the stop the loss runs drains
   too, and no lock of the first drain is held across the read. *)
let test_release_transport_fault () =
  let d, p = P.open_ ~transport:true "keep:transport" in
  let runs = Atomic.make 0 in
  ignore (submit_kept d runs);
  Gc.full_major ();
  Gc.full_major ();
  P.fault_word p "the link went down";
  drain d;
  equal (option string) (Some "the link went down") (Rig.lost d)

(* A submission submitted twice keeps its value until the work of both submits
   is done: the first's done keeps it. *)
let[@inline never] submit_twice d runs =
  let s = Sub.make ~keep:(counted runs) d [||] in
  let first = Rig.Point.value (submit s) in
  ignore (submit s);
  first

let test_release_last_submit () =
  let d, p = P.open_ "keep:last-submit" in
  let runs = Atomic.make 0 in
  let first = submit_twice d runs in
  P.set_word p first;
  Gc.full_major ();
  released d;
  equal ~msg:"with its first submit's work done" int 0 (Atomic.get runs);
  P.set_word p (Rig.submitted d);
  released d;
  equal ~msg:"with both done" int 1 (Atomic.get runs)

(* A device ended while a submission keeps a value holds its slot until a call
   on it lets the value go: the device's points name it while the submission is
   reachable, and once it is not and a drain let the value go, a later device
   takes the slot. The later device's open takes the slot freed last, which
   rig.mli does not promise: it promises only a free slot, and no call names
   a slot. *)
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let[@inline never] kept_submission d runs held =
  let s = Sub.make ~keep:(counted runs) d [||] in
  held := Some s;
  submit s

let slot_passes name ~end_ =
  let d, p = P.open_ name in
  let runs = Atomic.make 0 and held = ref None in
  let pt = kept_submission d runs held in
  end_ d p;
  Gc.full_major ();
  released d;
  ignore (P.open_ (name ^ "-later"));
  equal ~msg:"while the submission is reachable" (option device) (Some d)
    (Rig.Point.device pt);
  equal ~msg:"its value, while the submission is reachable" int 0
    (Atomic.get runs);
  held := None;
  Gc.full_major ();
  released d;
  equal ~msg:"its value, once let go" int 1 (Atomic.get runs);
  ignore (P.open_ (name ^ "-last"));
  equal ~msg:"once its value went" (option device) None (Rig.Point.device pt)

let test_slot_closed () =
  slot_passes "keep:slot-closed" ~end_:(fun d _ -> Rig.close d)

let test_slot_lost () =
  (* The process's first failure keeps its slot: one comes first. *)
  let first, pf = P.open_ "keep:slot-first" in
  P.fail pf;
  raises_match (lost first) (fun () -> submit (empty first));
  slot_passes "keep:slot-lost" ~end_:(fun d p ->
      P.fail p;
      raises_match (lost d) (fun () -> submit (empty d)))

(* A kept value orders no work: a submission that keeps it follows nothing
   another device's submission that keeps it did. *)
let test_keep_orders_nothing () =
  let d, _ = P.open_ "keep:orders" and e, pe = P.open_ "keep:orders-other" in
  let v = ref () in
  ignore (submit (Sub.make ~keep:v e [||]));
  Rig.Point.wait (submit (Sub.make ~keep:v d [||]));
  equal ~msg:"the other device's work, unrun" int 1 (P.queued pe)

(* Fixed memory *)

let answer = Testable.make ~pp:Support.Reader.pp_answer ~equal:( = )

(* A host read of [m] that claims, as a reader in C does: whether it must
   wait. *)
let host_read m =
  let a = Support.Reader.claim m Rig.Read in
  if a = Support.Reader.Claimed then Support.Reader.release m;
  a

(* Submits on [d] once a submission with [m], [d]'s memory, fixed for
   [access]: its point. *)
let fixing d m access = submit (Sub.make ~fixed:[ (m, access) ] d [||])
let fixed_value d m access = Rig.Point.value (fixing d m access)

(* Fixed memory read is ordered as a read: a read of it, on the host or on
   another device, waits for no other reader, and a write waits for every
   reader. *)
let test_fixed_read () =
  let d, _ = P.open_ "keep:fixed-read" and e, pe = P.open_ "keep:fixed-read-e" in
  let m = B.create Rig.host (1 lsl 16) in
  let on_d = require_some (B.borrow d m) and on_e = require_some (B.borrow e m) in
  let v = fixed_value e on_e Rig.Read in
  equal ~msg:"a host read" answer Support.Reader.Claimed (host_read m);
  Rig.Point.wait (fixing d on_d Rig.Read);
  equal ~msg:"another device's read, unrun" int 1 (P.queued pe);
  ignore (fixing d on_d Rig.Read_write);
  at_least ~msg:"a write follows the read" int ~than:v (Rig.signaled e)

(* Fixed memory written is ordered as a written run buffer: a read follows
   it. *)
let test_fixed_write () =
  let d, _ = P.open_ "keep:fixed-write" and e, pe = P.open_ "keep:fixed-write-e" in
  let m = B.create Rig.host (1 lsl 16) in
  let on_d = require_some (B.borrow d m) and on_e = require_some (B.borrow e m) in
  let v = fixed_value e on_e Rig.Read_write in
  equal ~msg:"a host read" answer Support.Reader.Wait (host_read m);
  Support.Reader.release m;
  equal ~msg:"the write, unrun" int 1 (P.queued pe);
  ignore (fixing d on_d Rig.Read);
  at_least ~msg:"a read follows the write" int ~than:v (Rig.signaled e)

(* One memory fixed in two submissions, of two programs, is ordered by its own
   stamps: each submission's fixed use is the memory's. *)
let test_fixed_shared () =
  let d, p = P.open_ "keep:fixed-shared" in
  let m = B.create d 64 in
  let reads = Sub.make ~fixed:[ (m, Rig.Read) ] d [||] in
  let writes = Sub.make ~fixed:[ (m, Rig.Read_write) ] d [||] in
  ignore (submit reads);
  equal ~msg:"after a read" answer Support.Reader.Claimed (host_read m);
  let v = Rig.Point.value (submit writes) in
  equal ~msg:"after a write" answer Support.Reader.Wait (host_read m);
  Support.Reader.release m;
  B.wait m Rig.Read;
  equal ~msg:"queued after the wait" int 0 (P.queued p);
  at_least int ~than:v (Rig.signaled d)

(* A copy into fixed memory a submission then reads is what it reads: the
   copy's write orders the submission. *)
let test_fixed_after_copy () =
  let d, p = P.open_ ~host_visible:false "keep:fixed-copy" in
  let m = B.create d 64 in
  let into = B.create Rig.host 64 in
  B.write into Bigarray.char (fun a -> Bigarray.Array1.fill a 'f');
  B.copy ~src:into ~dst:m;
  let v = fixed_value d m Rig.Read in
  ignore (P.run p);
  at_least int ~than:v (Rig.signaled d);
  let back = B.create Rig.host 64 in
  B.copy ~src:m ~dst:back;
  equal string (String.make 64 'f')
    (B.read back Bigarray.char (fun ba -> String.init 64 (Bigarray.Array1.get ba)))

(* A submission keeps its fixed memory: it returns once the submission is
   unreachable and its work done. *)
let test_fixed_returns () =
  let d, p = P.open_ "keep:fixed-returns" in
  let at =
    (fun () ->
      let m = B.create d 64 in
      ignore (fixing d m Rig.Read);
      Support.at m)
      ()
  in
  let freed () =
    Gc.full_major ();
    drain d;
    Rig.free_cache d;
    drain d;
    List.exists (fun (a, _) -> a = at) (P.frees p)
  in
  equal ~msg:"while its work is undone" bool false (freed ());
  ignore (P.run p);
  equal ~msg:"once it is done" bool true (freed ())

let refused ~msg f = raises_match ~msg Exn.invalid_arg f

let test_fixed_refusals () =
  let d, _ = P.open_ "keep:fixed-refusals" and e, _ = P.open_ "keep:other" in
  let make fixed = Sub.make ~fixed d [||] in
  let dead = B.create d 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ dead ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" dead));
  refused ~msg:"a dead buffer" (fun () -> make [ (dead, Rig.Read) ]);
  refused ~msg:"another device's" (fun () -> make [ (B.create e 8, Rig.Read) ]);
  let m = B.create d 8 in
  let s = make [ (m, Rig.Read) ] in
  ignore (Rig.Claim.with_ ~read:[] ~donate:[ m ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"consumed" m)));
  refused ~msg:"fixed memory that died" (fun () -> submit s)

(* Two domains submitting submissions that fix one memory for reading, each on
   a device of its own: no submit waits for the other domain's, whose work
   stays queued, and the memory is read by both. *)
type readers = { mutable submits : int }

let readers_t =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "%d submits" r.submits) "r"

(* Each domain's device, and the memory both borrow, made once for every
   case before any domain runs. A case then maps and frees nothing, so the
   devices' logs stay short, and allocates no buffer a device can borrow,
   each of which can bring a major cycle. *)
let reader_memory = B.create Rig.host (1 lsl 16)

let reader_devices =
  let device name =
    let d, p = P.open_ name in
    (d, p, require_some (B.borrow d reader_memory))
  in
  (device "keep:readers-0", device "keep:readers-1")

(* Each domain's submission over the memory, on its device. The earlier
   cases' work runs first, so a drain walks one case's releases, never every
   case's. *)
let make_readers () =
  let (d0, p0, on0), (d1, p1, on1) = reader_devices in
  ignore (P.run p0);
  ignore (P.run p1);
  let device d on = Sub.make ~fixed:[ (on, Rig.Read) ] d [||] in
  (reader_memory, device d0 on0, device d1 on1)

let submit_read (_, s, s') domain =
  ignore (submit (if domain = 0 then s else s'))

let reader_commands =
  [
    command "make" (Gen.unit @-> makes readers_t)
      (fun () -> { submits = 0 })
      make_readers;
    command "submit"
      (readers_t ^-> Gen.int_range 0 1 @-> judges unit)
      (fun r _ outcome ->
        (match outcome with Ok () -> () | Error e -> raise e);
        r.submits <- r.submits + 1)
      submit_read;
    command "host read"
      (readers_t ^-> judges answer)
      (fun _ outcome ->
        match outcome with
        | Ok a -> equal ~msg:"no reader waits for a read" answer Support.Reader.Claimed a
        | Error e -> raise e)
      (fun (m, _, _) -> host_read m);
  ]

let tests =
  [
    group ~timeout "releases"
      [
        test "a value kept on two devices is let go once both are done"
          test_two_devices;
        test "a kept value is let go in a drain after its work is done"
          test_release;
        test "a submission never submitted lets go once unreachable"
          test_release_unsubmitted;
        test "a kept value waits for a lost device's word and answer"
          test_release_lost;
        test ~timeout:10. "a drain that finds a transport's fault returns"
          test_release_transport_fault;
        test "a kept value orders no work" test_keep_orders_nothing;
        test "a submission keeps its value until its work is done"
          test_keeps_value;
        test "a value submitted twice waits for both submits' work"
          test_release_last_submit;
        test "a closed device keeps its slot while it keeps a value"
          test_slot_closed;
        test "a lost device keeps its slot while it keeps a value"
          test_slot_lost;
      ];
    group ~timeout "fixed memory"
      [
        test "fixed memory read waits for no other reader" test_fixed_read;
        test "fixed memory written orders as a written run buffer"
          test_fixed_write;
        test "one memory fixed in two submissions is ordered by its stamps"
          test_fixed_shared;
        test "a copy into fixed memory orders the submission that reads it"
          test_fixed_after_copy;
        test "fixed memory returns once its submission is unreachable and done"
          test_fixed_returns;
        test "fixed memory refuses what make and submit state"
          test_fixed_refusals;
        stateful ~domains:2
          "two domains' reads of one fixed memory wait for no other read"
          reader_commands;
      ];
  ]

let () = exit (run "rig.keep" tests)
