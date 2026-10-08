(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support

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
  (* The wait names the producer's word, or its object, by nx_edge.h's kinds:
     Polled's word and object are both at its [self]. *)
  let kind =
    match completion with
    | `Host -> Support.nx_word
    | `Object -> Support.nx_object
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

let test_in_queue () = in_queue ~completion:`Host
let test_in_queue_object () = in_queue ~completion:`Object

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

let tests =
  [
    group ~timeout "values"
      [
        test "values follow one another from 1" test_values;
        test "a submission's fills run with its value" test_fill;
        test "a device's work completes once a wait reaches it" test_polled;
        test "a wait slot left unset waits for nothing" test_unset_wait;
        test "a wait names a submitted value" test_wait_beyond;
        test "a word still for three intervals loses nothing" test_still;
      ];
    group ~timeout "refusals"
      [
        test "a submission refuses what it cannot run" test_refusals;
        test "a slot whose buffer died refuses the submit" test_dead_slot;
        test "a copy on a device that runs no copies is refused"
          test_copy_refused;
      ];
    group ~timeout "order"
      [
        test "a read waits for another device's write" test_read_waits;
        test "a part's buffers wait as slots do" test_part_points;
        test "a queue that waits on host words waits in the queue" test_in_queue;
        test "a queue that waits on objects waits on the producer's object"
          test_in_queue_object;
        test "a full queue's submit waits for room" test_room;
      ];
    group ~timeout "lifetime"
      [
        stateful "memory a submission names returns after its last submit"
          lifetime;
      ];
    group ~timeout "cost"
      [ test "a submit that does not wait allocates nothing" test_allocation ];
  ]

let () = exit (run "device_core.submit" tests)
