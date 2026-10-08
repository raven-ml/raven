(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)

let empty ?(reads = 0) ?(writes = 0) ?(waits = 0) d =
  Sub.make ~reads ~writes ~waits d [||]

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
  let s = empty d in
  let values = List.init 5 (fun _ -> C.Point.value (C.submit s)) in
  equal (list int) [ 1; 2; 3; 4; 5 ] values;
  equal int 5 (C.submitted d);
  equal int 5 (C.signaled d);
  equal bool true (C.equal d (C.Point.device (C.submit s)))

let test_fill () =
  let d = memory "submit:fill" in
  let arg = B.create C.host 8 in
  Support.store (B.address arg) 0;
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg; bump arg |] in
  ignore (C.submit s);
  ignore (C.submit s);
  equal int 4 (word arg)

(* Polled runs nothing until a wait reaches its sleep. *)
let test_polled () =
  let d, p = P.open_ "submit:polled" in
  let s = empty d in
  let a = C.submit s in
  let b = C.submit s in
  equal int 2 (P.queued p);
  equal int 0 (C.signaled d);
  C.wait d (C.Point.value b);
  equal int 0 (P.queued p);
  equal bool true (C.Point.value a < C.Point.value b)

(* Work that runs long is waited for: only a driver declares a hang. *)
let test_still () =
  let d, p = P.open_ "submit:still" in
  let v = C.Point.value (C.submit (empty d)) in
  P.stall p 3;
  C.wait d v;
  equal (option string) None (C.lost d);
  equal int 4 (List.length (List.filter (( = ) "sleep") (P.log p)))

let test_refusals () =
  let d = memory "submit:refusals" in
  let arg = B.create C.host 8 in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d
        [| { (bump arg) with after = [| 0 |] } |]);
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d
        [| { (bump arg) with queue = "COPY:9" } |]);
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:(-1) ~writes:0 ~waits:0 d [||]);
  raises_match Exn.invalid_arg (fun () -> C.submit (empty ~reads:1 d))

(* Each misuse [Submission.make] and the slot setters state raises
   [Invalid_argument]. *)
let test_make_refusals () =
  let d, _ = P.open_ ~host_visible:false "submit:make-refusals" in
  let arg = B.create C.host 8 in
  let unseen = B.create d 8 and other = B.create d 16 in
  let dead = B.create d 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ dead ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" dead));
  let copy src dst =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let make d parts = ignore (Sub.make ~reads:0 ~writes:0 ~waits:0 d parts) in
  let s = Sub.make ~reads:1 ~writes:1 ~waits:1 d [||] in
  List.iter
    (fun (msg, f) -> raises_match ~msg Exn.invalid_arg f)
    [
      ("on the host", fun () -> make C.host [||]);
      ( "after a negative index",
        fun () -> make d [| { (bump arg) with after = [| -1 |] } |] );
      ( "a fill's argument the host does not address",
        fun () -> make d [| bump unseen |] );
      ( "a copy of buffers of two sizes",
        fun () -> make d [| copy unseen other |] );
      ( "a copy of another device's memory",
        fun () -> make d [| copy arg (B.create C.host 8) |] );
      ("a part's dead buffer", fun () -> make d [| copy dead unseen |]);
      ("a read slot past the count", fun () -> Sub.read s 1 arg);
      ("a write slot past the count", fun () -> Sub.write s 1 arg);
      ( "a negative wait slot",
        fun () -> Sub.wait_for s (-1) (C.submit (empty d)) );
    ]

(* A submit that raises clears the slots set for it: the next submit finds its
   read slot unset. *)
let test_cleared_on_raise () =
  let d = memory "submit:cleared" in
  let producer, pp = P.open_ "submit:cleared-producer" in
  let point = C.submit (empty producer) in
  P.fail pp;
  (try ignore (C.submit (empty producer)) with C.Lost _ -> ());
  let s = empty ~reads:1 ~waits:1 d in
  Sub.read s 0 (B.create C.host 8);
  Sub.wait_for s 0 point;
  raises_match (function C.Lost _ -> true | _ -> false) (fun () -> C.submit s);
  raises_match Exn.invalid_arg (fun () -> C.submit s)

(* A slot whose buffer died after it was set refuses the submit. *)
let test_dead_slot () =
  let d = memory "submit:dead-slot" in
  let b = B.create C.host 8 in
  let s = empty ~reads:1 d in
  Sub.read s 0 b;
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  raises_match Exn.invalid_arg (fun () -> C.submit s);
  equal int 0 (C.submitted d)

let test_unset_wait () =
  let d = memory "submit:unset-wait" in
  equal int 1 (C.Point.value (C.submit (empty ~waits:2 d)))

let test_wait_beyond () =
  let d = memory "submit:beyond" and e = memory "submit:beyond-2" in
  let p = C.submit (empty e) in
  raises_match Exn.invalid_arg (fun () -> C.wait e (C.Point.value p + 1));
  let s = empty ~waits:1 d in
  Sub.wait_for s 0 p;
  ignore (C.submit s)

(* A read of memory another device wrote waits for that write: on the host,
   since a memory device waits on no other device in its queue. *)
let test_read_waits () =
  let producer, pp = P.open_ "submit:producer" in
  let consumer = memory "submit:consumer" in
  let on = B.create producer page_bytes in
  let w = Sub.make ~reads:0 ~writes:1 ~waits:0 producer [||] in
  Sub.write w 0 on;
  ignore (C.submit w);
  equal int 1 (P.queued pp);
  let r = Sub.make ~reads:1 ~writes:0 ~waits:0 consumer [||] in
  Sub.read r 0 on;
  ignore (C.submit r);
  equal int 0 (P.queued pp)

(* A part's buffers are ordered as slots are: a copy waits for another device's
   write of its source and for another device's read of its destination. *)
let test_part_points () =
  let d, _ = P.open_ "submit:parts" in
  let e, pe = P.open_ "submit:parts-other" in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| copy |] in
  let on_e ~reads ~writes b =
    let w = Sub.make ~reads ~writes ~waits:0 e [||] in
    if reads = 1 then Sub.read w 0 b else Sub.write w 0 b;
    ignore (C.submit w)
  in
  on_e ~reads:0 ~writes:1 src;
  ignore (C.submit s);
  equal ~msg:"after a write of the source" int 0 (P.queued pe);
  on_e ~reads:1 ~writes:0 dst;
  ignore (C.submit s);
  equal ~msg:"after a read of the destination" int 0 (P.queued pe)

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
  let a = C.submit (empty producer) in
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  let b = C.submit s in
  (* The wait names the producer's word, or its object, by rig_edge.h's kinds:
     Polled's word and object are both at its [self]. *)
  let kind =
    match completion with
    | `Host -> Support.rig_word
    | `Object -> Support.rig_object
  in
  equal
    (list (triple int int int))
    [ (kind, Nativeint.to_int (P.self pp), 1) ]
    (P.last_waits cp);
  equal int 1 (P.queued pp);
  equal int 1 (P.queued cp);
  equal int 0 (P.run cp);
  C.wait producer (C.Point.value a);
  equal int 1 (P.run cp);
  equal int (C.Point.value b) (C.signaled consumer)

(* A submission with more unreached producers than its device's queue waits on
   in one submission waits on the host for the others: its queue gets
   [max_waits] waits, and it completes once those are reached. *)
let test_max_waits () =
  let producers =
    List.init 3 (fun i -> P.open_ (Printf.sprintf "submit:bound-producer-%d" i))
  in
  let consumer, cp = P.open_ ~waits_on:[ `Host ] ~max_waits:2 "submit:bound" in
  let points = List.map (fun (d, _) -> C.submit (empty d)) producers in
  let s = empty ~waits:3 consumer in
  List.iteri (Sub.wait_for s) points;
  let b = C.submit s in
  let reached = List.filter (fun (_, p) -> P.queued p = 0) producers in
  equal ~msg:"waits in the queue" int 2 (List.length (P.last_waits cp));
  equal ~msg:"producers reached on the host" int 1 (List.length reached);
  equal ~msg:"before its waits hold" int 0 (P.run cp);
  List.iter (fun (_, p) -> ignore (P.run p)) producers;
  equal ~msg:"once they hold" int 1 (P.run cp);
  equal int (C.Point.value b) (C.signaled consumer)

(* A driver answering fewer than zero waits has a queue with no room for any:
   every producer is waited for on the host. *)
let test_negative_max_waits () =
  let producer, pp = P.open_ "submit:negative-producer" in
  let consumer, cp =
    P.open_ ~waits_on:[ `Host ] ~max_waits:(-1) "submit:negative-consumer"
  in
  let a = C.submit (empty producer) in
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  ignore (C.submit s);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp)

(* A producer whose word the device cannot map is waited for on the host: the
   submit returns with the producer's work done and no wait in the queue. *)
let test_unmapped_wait () =
  let producer, pp = P.open_ "submit:unmapped-producer" in
  let consumer, cp =
    P.open_ ~peers:false ~waits_on:[ `Host ] "submit:unmapped-consumer"
  in
  let a = C.submit (empty producer) in
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  ignore (C.submit s);
  equal ~msg:"waits in the queue" int 0 (List.length (P.last_waits cp));
  equal ~msg:"the producer's work" int 0 (P.queued pp);
  equal int (C.Point.value a) (C.signaled producer)

let test_in_queue () = in_queue ~completion:`Host
let test_in_queue_object () = in_queue ~completion:`Object

(* A submit hands its driver the handle of each region its slots and parts use,
   once. A case sets the read slots to the buffers [reads] picks among [n], and
   the write slots to those [writes] picks. *)
type handles = { n : int; reads : int list; writes : int list }

let pp_handles ppf c =
  let ints = Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int) in
  Format.fprintf ppf "@[{ n = %d;@ reads = [%a];@ writes = [%a] }@]" c.n ints
    c.reads ints c.writes

let handles_case =
  let open Gen in
  with_pp pp_handles
    (let* n = int_range 1 40 in
     let slots k = list ~size:(int_range 0 k) (int_range 0 (n - 1)) in
     map
       (fun (reads, writes) -> { n; reads; writes })
       (pair (slots 24) (slots 8)))

let handles_device = lazy (P.open_ "submit:handles")

let handles_law c =
  let d, p = Lazy.force handles_device in
  let bs = Array.init c.n (fun _ -> B.create d 8) in
  let reads = List.length c.reads and writes = List.length c.writes in
  let s = Sub.make ~reads ~writes ~waits:0 d [| bump (B.create C.host 8) |] in
  List.iteri (fun i k -> Sub.read s i bs.(k)) c.reads;
  List.iteri (fun i k -> Sub.write s i bs.(k)) c.writes;
  ignore (C.submit s);
  let named = List.sort_uniq Int.compare (c.reads @ c.writes) in
  cover "a buffer named twice"
    (List.length named < List.length c.reads + List.length c.writes);
  cover "more than 16 buffers named" (List.length named > 16);
  equal (list int)
    (List.sort Int.compare (List.map (fun k -> B.address bs.(k)) named))
    (List.sort Int.compare (P.last_handles p))

(* Each submit of one submission hands its driver the handles that submit's
   slots and parts name, whatever the submits before it named. A case submits
   once per element of [runs], each picking the slots' buffers among [n]. *)
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
    Sub.make ~reads:c.reads ~writes:c.writes ~waits:0 d
      [| bump (B.create C.host 8) |]
  in
  let submit run =
    List.iteri
      (fun i k ->
        if i < c.reads then Sub.read s i bs.(k)
        else Sub.write s (i - c.reads) bs.(k))
      run;
    ignore (C.submit s);
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
        (submit run))
    c.runs

(* A full queue answers Later: the submit waits for one more value. *)
let test_room () =
  let d, p = P.open_ ~capacity:1 "submit:room" in
  let arg = B.create C.host 8 in
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg |] in
  ignore (C.submit s);
  ignore (C.submit s);
  equal int 1 (P.queued p);
  equal int 1 (C.signaled d)

(* A copy on a device that runs no copies is refused where the caller can act:
   when the submission is made. *)
let test_copy_refused () =
  let d, _ = P.open_ ~copies:false "submit:no-copies" in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy =
    { Sub.queue = "COMPUTE:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d [| copy |])

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

type device = { d : C.t; p : P.t; mutable watched : watched list }
type submission = { dev : device; mutable s : Sub.t option; parts : watched }
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
  let s = Sub.make ~reads:0 ~writes:1 ~waits:0 t.d [| copy |] in
  { dev = t; s = Some s; parts }

let submit sub =
  let t = sub.dev in
  let s = Option.get sub.s in
  let slot = B.create t.d 256 in
  let at = B.address slot in
  Sub.write s 0 slot;
  let v = C.Point.value (C.submit s) in
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
  C.free_cache t.d

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
      submit;
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
      (fun t -> C.wait t.d (C.submitted t.d));
  ]

let test_allocation () =
  let d = memory "submit:words" in
  let s = empty ~reads:1 d and b = B.create C.host 8 in
  Sub.read s 0 b;
  ignore (C.submit s);
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    Sub.read s 0 b;
    ignore (Sys.opaque_identity (C.submit s))
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  equal int 0 words

(* Two domains submitting one submission: the submits take turns, each takes the
   device's next value, and every fill runs. A program counts values from
   [base], its device's value when it began. *)
type shared = { dev : C.t; base : int; arg : B.t; sub : Sub.t }
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
  let arg = B.create C.host 8 in
  Support.store (B.address arg) 0;
  {
    dev = d;
    base = C.submitted d;
    arg;
    sub = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg; bump arg |];
  }

(* Every submit ran both its fills. *)
let release_shared t =
  equal ~msg:"fills" int (2 * (C.submitted t.dev - t.base)) (word t.arg);
  Mutex.protect shared_pool (fun () -> shared_free := t.dev :: !shared_free)

let shared =
  abstract
    ~pp:(fun ppf r -> Format.fprintf ppf "values %d" r.values)
    ~release:release_shared "s"

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
      (fun t -> C.Point.value (C.submit t.sub) - t.base);
  ]

let tests =
  [
    group ~timeout "values"
      [
        test "values follow one another from 1" test_values;
        test "a submission's fills run with its value" test_fill;
        stateful ~domains:2 "two domains submit one submission in turn"
          shared_commands;
        test "a device's work completes once a wait reaches it" test_polled;
        test "a wait slot left unset waits for nothing" test_unset_wait;
        test "a wait names a submitted value" test_wait_beyond;
        test "a word still for three intervals loses nothing" test_still;
      ];
    group ~timeout "refusals"
      [
        test "a submission refuses what it cannot run" test_refusals;
        test "a submission refuses each misuse its interface states"
          test_make_refusals;
        test "a submit that raises clears its slots" test_cleared_on_raise;
        test "a slot whose buffer died refuses the submit" test_dead_slot;
        test "a copy on a device that runs no copies is refused"
          test_copy_refused;
      ];
    group ~timeout "order"
      [
        test "a read waits for another device's write" test_read_waits;
        test "a part's buffers wait as slots do" test_part_points;
        test "a queue that waits on host words waits in the queue" test_in_queue;
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
