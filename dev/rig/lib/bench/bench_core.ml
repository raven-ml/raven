(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits on Polled, a driver over host memory whose queue runs when the bench
   runs it, each row beside the floor that bounds it: the driver's own room and
   submit entries called directly, with the same parts and the same runs of the
   queue. A row's distance to its floor is the core's share of a submit.

   A row that does not wait runs the queue every [drain] submits, as its floor
   does, so the queue stays short. A replay row runs the queue once per run,
   before its submit: the device completes run N-1 while the host prepares run
   N, and the wait for run N-2 finds it reached. *)

module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support
module Claim = Rig.Claim

let strf = Printf.sprintf

external floor_new :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint
  = "rig_bench_floor_new"

external floor_submit : nativeint -> int -> unit = "rig_bench_floor_submit"
[@@noalloc]

external floor_turn_submit : nativeint -> int -> unit
  = "rig_bench_floor_turn_submit"

external floor_share : nativeint -> nativeint -> unit = "rig_bench_floor_share"

external claim_3 : B.t -> B.t -> B.t -> unit = "rig_bench_claim_3"
[@@noalloc]

let drain = 64
let slots = 24
let runs = 100

type dev = { d : C.t; p : P.t; mutable n : int }

let opened = ref 0

let dev () =
  incr opened;
  let d, p = P.open_ (strf "bench:%d" !opened) in
  { d; p; n = 0 }

let drained t =
  t.n <- t.n + 1;
  if t.n mod drain = 0 then ignore (P.run t.p)

let words d n = Array.init n (fun _ -> B.create d 8)

(* A part that adds 1 to the word [arg] holds. *)
let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let bumping ?(reads = 0) ?(writes = 0) d =
  Sub.make ~reads ~writes d [| bump (B.create C.host 8) |]

(* A submit of a run that reads nothing, writes nothing and waits for
   nothing. *)
let submit s = C.submit s ~reads:[||] ~writes:[||] ~waits:[||]
let submit_read s bs = ignore (C.submit s ~reads:bs ~writes:[||] ~waits:[||])

(* Submits *)

let empty () =
  let t = dev () in
  (t, Sub.make ~reads:0 ~writes:0 t.d [||])

let costing () =
  let t = dev () in
  (t, bumping t.d)

let reading () =
  let t = dev () in
  (t, bumping ~reads:slots t.d, words t.d slots)

(* Buffers of another device, its last write reached, borrowed on [t]'s. *)
let foreign () =
  let t = dev () and o = dev () in
  let bs = words o.d slots in
  let w = Sub.make ~reads:0 ~writes:slots o.d [||] in
  let v = C.Point.value (C.submit w ~reads:[||] ~writes:bs ~waits:[||]) in
  ignore (P.run o.p);
  C.wait o.d v;
  ( t,
    bumping ~reads:slots t.d,
    Array.map (fun b -> Option.get (B.borrow t.d b)) bs )

(* Another domain submitting to the same device until the row ends. *)
let contended () =
  let t, s = costing () in
  let stop = Atomic.make false in
  let other = { t with n = 0 } and s' = bumping t.d in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          ignore (submit s');
          drained other
        done)
  in
  (t, s, stop, rival)

let submit_rows =
  let row name setup f = Thumper.bench_with_setup ~setup name f in
  Thumper.group "submit/polled"
    [
      row "empty" empty (fun (t, s) ->
          let v = C.Point.value (submit s) in
          ignore (P.run t.p);
          C.wait t.d v);
      row "cost" costing (fun (t, s) ->
          ignore (submit s);
          drained t);
      row "slots-24" reading (fun (t, s, bs) ->
          submit_read s bs;
          drained t);
      row "foreign-24" foreign (fun (t, s, bs) ->
          submit_read s bs;
          drained t);
      Thumper.bench_with_setup ~setup:contended
        ~teardown:(fun (_, _, stop, rival) ->
          Atomic.set stop true;
          Domain.join rival)
        "two-domains"
        (fun (t, s, _, _) ->
          ignore (submit s);
          drained t);
    ]

(* Replays *)

type copy = { s : Sub.t; args : B.t; at : int; outs : B.t array }
type replay = { t : dev; params : B.t array; copies : copy array }

(* Two copies of a step over [slots] parameters: each reads them, writes its
   output, and fills its own arguments, which the host rewrites before each
   run. *)
let replaying () =
  let t = dev () in
  let copy () =
    let args = B.create C.host 8 in
    let s = Sub.make ~reads:slots ~writes:1 t.d [| bump args |] in
    { s; args; at = B.address args; outs = [| B.create t.d 8 |] }
  in
  { t; params = words t.d slots; copies = [| copy (); copy () |] }

let run r =
  let c = r.copies.(r.t.n land 1) in
  r.t.n <- r.t.n + 1;
  B.wait c.args B.Read_write;
  ignore (P.run r.t.p);
  Support.store c.at r.t.n;
  ignore (C.submit c.s ~reads:r.params ~writes:c.outs ~waits:[||])

let replay_rows =
  Thumper.group "replay/polled"
    [
      Thumper.bench_with_setup ~setup:replaying "params-24" run;
      Thumper.bench_with_setup ~setup:replaying "pipelined-100" (fun r ->
          for _ = 1 to runs do
            run r
          done;
          ignore (P.run r.t.p);
          C.wait r.t.d (C.submitted r.t.d));
    ]

(* Memory *)

let kib = 1024
let mib = 1024 * kib

let memory () =
  incr opened;
  match C.memory_device (strf "memory:%d" !opened) with
  | Ok d -> d
  | Error why -> failwith why

let host n = B.create C.host n
let chars n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* Buffers of [d] whose last write, a submission of [d], is reached. *)
let written d n =
  let bs = Array.init n (fun _ -> B.create d 8) in
  let w = Sub.make ~reads:0 ~writes:n d [||] in
  C.wait d (C.Point.value (C.submit w ~reads:[||] ~writes:bs ~waits:[||]));
  bs

let row name setup f = Thumper.bench_with_setup ~setup name f
let create d n () = ignore (B.create d n)

let buffer_rows =
  Thumper.group "buffer"
    [
      Thumper.bench "host-create-16" (create C.host 16);
      Thumper.bench "host-create-1M" (create C.host mib);
      row "memory-create-cached-4K" memory (fun d -> create d (4 * kib) ());
      row "wait-reached-24"
        (fun () -> written (memory ()) slots)
        (fun bs ->
          for i = 0 to slots - 1 do
            B.wait (Array.unsafe_get bs i) B.Read_write
          done);
      (* A view of 64 MiB a memory device holds, borrowed on the host: the view
         paces no collection. *)
      Thumper.bench_with_setup
        ~metrics:Thumper.Metric.[ wall_time; alloc_words; major_collections ]
        ~setup:(fun () ->
          Option.get (B.borrow C.host (B.create (memory ()) (64 * mib))))
        "bigarray-64M"
        (fun b -> B.bigarray Bigarray.char b);
    ]

let claim_rows =
  Thumper.group "claim"
    [
      row "read-release"
        (fun () -> host 16)
        (fun b ->
          Claim.read b;
          Claim.release b);
      row "with-24"
        (fun () -> (List.init slots (fun _ -> host 16), [ [ host 16 ] ]))
        (fun (read, donate) -> Claim.with_ ~read ~donate ignore);
      row "c-read-release-3"
        (fun () -> (host 16, host 16, host 16))
        (fun (a, b, c) -> claim_3 a b c);
    ]

let copy_rows =
  let pair d n () = (B.create d n, B.create d n) in
  let copy (src, dst) = B.copy ~src ~dst in
  Thumper.group "copy"
    [
      row "host-4K" (pair C.host (4 * kib)) copy;
      row "host-64M" (pair C.host (64 * mib)) copy;
      row "memory-4K" (fun () -> pair (memory ()) (4 * kib) ()) copy;
    ]

(* A collection hands the memory of 1,000 dropped buffers back, and the next
   create drains it into the device's cache. *)
let dropped = 1000

let drain_rows =
  Thumper.group "drain"
    [
      row "dropped-1000" memory (fun d ->
          for _ = 1 to dropped do
            create d (4 * kib) ()
          done;
          Gc.full_major ();
          create d (4 * kib) ());
    ]

let wait_rows =
  Thumper.group "wait/memory"
    [
      row "reached"
        (fun () ->
          let d = memory () in
          (d, C.Point.value (submit (Sub.make ~reads:0 ~writes:0 d [||]))))
        (fun (d, v) -> C.wait d v);
    ]

(* A host buffer of 64 MiB collected: the end of the cycle returns it, being
   more than the cache keeps. *)
(* A host buffer of 64 KiB taken and dropped; the minor collection hands its
   memory back to the cache, from which the next take comes. *)
let take () =
  create C.host (64 * kib) ();
  Gc.minor ()

(* The host cache holding [others] buffers of other sizes, 68 KiB to 464 KiB, 26
   MiB in all, under the cache's floor: a take walks only its own size's. *)
let others = 100

let caching () =
  let keep = List.init others (fun i -> host ((17 + i) * 4 * kib)) in
  ignore (Sys.opaque_identity keep);
  Gc.full_major ()

let heap_rows =
  Thumper.group "heap"
    [
      Thumper.bench "trim-64M" (fun () ->
          create C.host (64 * mib) ();
          Gc.full_major ());
      Thumper.bench "take-64K" take;
      row "take-64K-cached-100" caching take;
    ]

let memory_floor_rows =
  let word () = B.address (host 8) in
  let blit (src, dst) = Bigarray.Array1.blit src dst in
  let chars2 n () = (chars n, chars n) in
  [
    Thumper.bench "bigarray-create-16" (fun () -> chars 16);
    Thumper.bench "bigarray-create-1M" (fun () -> chars mib);
    row "word-load" word Support.load;
    row "word-load-24" word (fun a ->
        for _ = 1 to slots do
          ignore (Support.load a)
        done);
    row "atomic-cas-2"
      (fun () -> Atomic.make 0)
      (fun a ->
        ignore (Atomic.compare_and_set a 0 1);
        ignore (Atomic.compare_and_set a 1 0));
    row "memcpy-4K" (chars2 (4 * kib)) blit;
    row "memcpy-64M" (chars2 (64 * mib)) blit;
    Thumper.bench "bigarray-dropped-1000" (fun () ->
        for _ = 1 to dropped do
          ignore (chars (4 * kib))
        done;
        Gc.full_major ();
        chars (4 * kib));
    Thumper.bench "bigarray-collected-64M" (fun () ->
        ignore (chars (64 * mib));
        Gc.full_major ());
  ]

(* Floors *)

(* [timeline] is the address of the driver's timeline word, which a floor loads
   as the core reads it. *)
type floor = {
  f : nativeint;
  fp : P.t;
  word : int;
  timeline : int;
  mutable k : int;
}

let floor () =
  let fp = P.make () in
  let f = floor_new (P.self fp) P.room_entry P.submit_entry Support.bump in
  let timeline = Option.get (P.address (P.word fp)) in
  { f; fp; word = B.address (B.create C.host 8); timeline; k = 0 }

let floor_drained t =
  t.k <- t.k + 1;
  if t.k mod drain = 0 then ignore (P.run t.fp)

let floor_run t =
  t.k <- t.k + 1;
  ignore (P.run t.fp);
  Support.store t.word t.k;
  floor_submit t.f 1

(* Another domain calling the same device's entries until the row ends, with a
   floor of its own: the driver alone serializes the two. With [share], each
   submit also takes one turn the two share, as the core takes a device's: by
   try-lock, and otherwise with the runtime released. *)
let floor_contended ~share () =
  let t = floor () in
  let stop = Atomic.make false in
  let other =
    {
      t with
      f = floor_new (P.self t.fp) P.room_entry P.submit_entry Support.bump;
      k = 0;
    }
  in
  if share then floor_share t.f other.f;
  let submit = if share then floor_turn_submit else floor_submit in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          submit other.f 1;
          floor_drained other
        done)
  in
  (t, stop, rival)

let floor_rows =
  let row name f = Thumper.bench_with_setup ~setup:floor name f in
  Thumper.group "floor"
    ([
       row "polled/release" (fun t ->
           floor_submit t.f 0;
           ignore (P.run t.fp);
           ignore (Support.load t.timeline));
       row "polled/cost" (fun t ->
           floor_submit t.f 1;
           floor_drained t);
       Thumper.bench_with_setup
         ~setup:(floor_contended ~share:false)
         ~teardown:(fun (_, stop, rival) ->
           Atomic.set stop true;
           Domain.join rival)
         "polled/two-domains"
         (fun (t, _, _) ->
           floor_submit t.f 1;
           floor_drained t);
       Thumper.bench_with_setup
         ~setup:(floor_contended ~share:true)
         ~teardown:(fun (_, stop, rival) ->
           Atomic.set stop true;
           Domain.join rival)
         "polled/two-domains-turn"
         (fun (t, _, _) ->
           floor_turn_submit t.f 1;
           floor_drained t);
       row "polled/run" floor_run;
       row "polled/pipelined-100" (fun t ->
           for _ = 1 to runs do
             floor_run t
           done;
           ignore (P.run t.fp);
           ignore (Support.load t.timeline));
       Thumper.bench_with_setup ~setup:Mutex.create "mutex-section" (fun m ->
           Mutex.lock m;
           Mutex.unlock m);
     ]
    @ memory_floor_rows)

let () =
  exit
  @@ Thumper.run "rig"
       [
         submit_rows;
         replay_rows;
         buffer_rows;
         claim_rows;
         copy_rows;
         drain_rows;
         wait_rows;
         heap_rows;
         floor_rows;
       ]
