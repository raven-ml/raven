(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits on Polled, a driver over host memory whose queue runs when the bench
   runs it, each row beside the floor that bounds it: the driver's own room and
   submit entries called directly, with the same parts and the same runs of the
   queue. A row's distance to its floor is rig's share of a submit.

   A row that does not wait runs the queue every [drain] submits, as its floor
   does, so the queue stays short. A replay row runs the queue once per run,
   before its submit: the device completes run N-1 while the host prepares run
   N, with each submit copying its parameters before the run is reused. *)

module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support
module Claim = Rig.Claim

let strf = Printf.sprintf

external floor_new : nativeint -> nativeint -> nativeint = "rig_bench_floor_new"

external floor_fill : nativeint -> nativeint -> string -> unit
  = "rig_bench_floor_fill"

external floor_int64 : nativeint -> int -> int -> unit = "rig_bench_floor_int64"
[@@noalloc]

external floor_submit : nativeint -> int -> unit = "rig_bench_floor_submit"
[@@noalloc]

external floor_turn_submit : nativeint -> int -> unit
  = "rig_bench_floor_turn_submit"

external floor_share : nativeint -> nativeint -> unit = "rig_bench_floor_share"
external claim_3 : B.t -> B.t -> B.t -> unit = "rig_bench_claim_3" [@@noalloc]

let drain = 64
let slots = 24
let runs = 100

type dev = { d : Rig.t; p : P.t; mutable n : int }

let opened = ref 0

let dev () =
  incr opened;
  let d, p = P.open_ (strf "bench:%d" !opened) in
  { d; p; n = 0 }

let drained t =
  t.n <- t.n + 1;
  if t.n mod drain = 0 then ignore (P.run t.p)

let words d n = Array.init n (fun _ -> B.create d 8)

(* One queue part, with no work beyond the fill call: with [uses], a ref to
   each slot by that kind, which the fill does not read. *)
let costing_sub ?(uses = [||]) d =
  Sub.make d [| Support.touching uses |]

let costing_run _ = Sub.Run.make ()

(* A submit of a run that reads nothing, writes nothing and waits for
   nothing. *)
let submit ~run s = Rig.submit s ~run ~buffers:[||]
let submit_read ~run s bs = ignore (Rig.submit s ~run ~buffers:bs)

(* Submits *)

(* A row's setup makes its submission and the one run it submits with. *)

let empty () =
  let t = dev () in
  (t, Sub.make t.d [||], Sub.Run.make ())

let costing () =
  let t = dev () in
  let s = costing_sub t.d in
  (t, s, costing_run s)

let reading () =
  let t = dev () in
  let s = costing_sub ~uses:(Array.make slots Sub.read) t.d in
  (t, s, costing_run s, words t.d slots)

(* Buffers of another device, its last write reached, borrowed on [t]'s. *)
let foreign () =
  let t = dev () and o = dev () in
  let bs = words o.d slots in
  let w = costing_sub ~uses:(Array.make slots Sub.write) o.d in
  let p = Rig.submit w ~run:(Sub.Run.make ()) ~buffers:bs in
  ignore (P.run o.p);
  Rig.Point.wait p;
  let s = costing_sub ~uses:(Array.make slots Sub.read) t.d in
  (t, s, costing_run s, Array.map (fun b -> Option.get (B.borrow t.d b)) bs)

(* Another domain submitting to the same device until the row ends, with its own
   run. *)
let contended () =
  let t, s, run = costing () in
  let stop = Atomic.make false in
  let other = { t with n = 0 } and s' = costing_sub t.d in
  let rival =
    Domain.spawn (fun () ->
        let run' = costing_run s' in
        while not (Atomic.get stop) do
          ignore (submit ~run:run' s');
          drained other
        done)
  in
  (t, s, run, stop, rival)

(* A submission of no part whose 200 fixed buffers it reads: each submit stamps
   each, as a compiled step does its constants. *)
let fixed = 200

let fixing () =
  let t = dev () in
  let fixed =
    List.map (fun b -> (b, Rig.Read)) (Array.to_list (words t.d fixed))
  in
  (t, Sub.make ~fixed t.d [||], Sub.Run.make ())

(* A launch whose 8 refs name a run of 4 reads and 4 writes: each submit stores
   the refs' offsets, as a caller does for each call. *)
let refs = 8

let launching () =
  let t = dev () in
  let image = Result.get_ok (Rig.Image.load t.d "functions") in
  let rs =
    Array.init refs (fun k ->
        if k < refs / 2 then Sub.read k ~at:(8 * k) else Sub.write k ~at:(8 * k))
  in
  let s = Sub.make t.d [| Sub.launch ~refs:rs image "main" ~params:(8 * refs) |] in
  let run = Sub.Run.make () and b = Sub.block s 0 in
  Sub.Run.groups run b 1 1 1;
  Sub.Run.threads run b 1 1 1;
  (t, s, run, b, words t.d refs)

(* A submission of 4096 parts on two queues: copies between views of 16
   allocations and fills that read one slot and write another, every part
   reading 4 fixed buffers. Making it places and orders each part. *)
let many = 4096

let making () =
  let t = dev () in
  let allocs = Array.init 16 (fun _ -> B.create t.d 256) in
  let view k =
    B.view allocs.(k mod 16) ~first:(64 * (k / 16 mod 4)) ~length:64
  in
  let part k =
    if k mod 2 = 0 then
      Sub.(copy ~src:(Buffer (view k)) ~dst:(Buffer (view (k + 1))))
    else
      Sub.fill
        ~refs:[| Sub.read (k mod 8) ~at:0; Sub.write ((k + 3) mod 8) ~at:8 |]
        Support.noop ~params:16
  in
  let fixed = List.map (fun b -> (b, Rig.Read)) (Array.to_list (words t.d 4)) in
  (t, Array.init many part, fixed)

let submit_rows =
  let row name setup f = Thumper.bench_with_setup ~setup name f in
  Thumper.group "submit/polled"
    [
      row "empty" empty (fun (t, s, run) ->
          let p = submit ~run s in
          ignore (P.run t.p);
          Rig.Point.wait p);
      row "cost" costing (fun (t, s, run) ->
          ignore (submit ~run s);
          drained t);
      row "slots-24" reading (fun (t, s, run, bs) ->
          submit_read ~run s bs;
          drained t);
      row "foreign-24" foreign (fun (t, s, run, bs) ->
          submit_read ~run s bs;
          drained t);
      row "fixed-200" fixing (fun (t, s, run) ->
          ignore (submit ~run s);
          drained t);
      row "make-4096" making (fun (t, parts, fixed) ->
          ignore (Sub.make ~fixed t.d parts));
      row "launch-refs-8" launching (fun (t, s, run, b, buffers) ->
          for k = 0 to refs - 1 do
            Sub.Run.int64 run b (8 * k) 0
          done;
          ignore (Rig.submit s ~run ~buffers);
          drained t);
      Thumper.bench_with_setup ~setup:contended
        ~teardown:(fun (_, _, _, stop, rival) ->
          Atomic.set stop true;
          Domain.join rival)
        "two-domains"
        (fun (t, s, run, _, _) ->
          ignore (submit ~run s);
          drained t);
    ]

(* Replays *)

type copy = {
  s : Sub.t;
  run : Sub.Run.t;
  block : Sub.block;
  buffers : B.t array;
}

type replay = { t : dev; copies : copy array }

(* Two copies of a step over [slots] parameters: each reads them, writes its
   output through its run parameters, which the host rewrites before each
   run. *)
let replaying () =
  let t = dev () in
  let params = words t.d slots in
  let copy () =
    (* poke stores at its first ref; the others name the parameters it reads. *)
    let refs =
      Array.append
        [| Sub.write slots ~at:0 |]
        (Array.init slots (fun k -> Sub.read k ~at:(16 + (8 * k))))
    in
    let part = Sub.fill ~refs Support.poke ~params:(16 + (8 * slots)) in
    let s = Sub.make t.d [| part |] in
    let run = Sub.Run.make () in
    let buffers = Array.append params [| B.create t.d 8 |] in
    { s; run; block = Sub.block s 0; buffers }
  in
  { t; copies = [| copy (); copy () |] }

let run r =
  let c = r.copies.(r.t.n land 1) in
  r.t.n <- r.t.n + 1;
  ignore (P.run r.t.p);
  Sub.Run.int64 c.run c.block 0 0;
  Sub.Run.int64 c.run c.block 8 r.t.n;
  Rig.submit c.s ~run:c.run ~buffers:c.buffers

let replay_rows =
  Thumper.group "replay/polled"
    [
      Thumper.bench_with_setup ~setup:replaying "params-24" (fun r ->
          ignore (run r));
      Thumper.bench_with_setup ~setup:replaying "pipelined-100" (fun r ->
          for _ = 1 to runs - 1 do
            ignore (run r)
          done;
          let last = run r in
          ignore (P.run r.t.p);
          Rig.Point.wait last);
    ]

(* Memory *)

let kib = 1024
let mib = 1024 * kib

(* A copy of 64 MiB times at one of three levels 5% apart on an i9-9900K, set by
   where its two buffers lie physically, apart for each row of a process: the
   memcpy floor, which runs no rig code, shows them too. The budget admits all
   three; the 4 KiB rows time the same paths at the default 5%. *)
let copy_64M = [ Thumper.Budget.no_slower_than 0.15 ]

let memory () =
  incr opened;
  Rig.memory_device (strf "memory:%d" !opened)

let host n = B.create Rig.host n
let chars n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* Buffers of [d] whose last write, a submission of [d], is reached. *)
let written d n =
  let bs = Array.init n (fun _ -> B.create d 8) in
  let w = costing_sub ~uses:(Array.make n Sub.write) d in
  let p = Rig.submit w ~run:(Sub.Run.make ()) ~buffers:bs in
  Rig.Point.wait p;
  bs

(* Host memory that two memory devices wrote through borrows, each write
   reached. *)
let written_twice () =
  let b = host 8 in
  let write d =
    let w = costing_sub ~uses:[| Sub.write |] d in
    let bd = Option.get (B.borrow d b) in
    Rig.Point.wait (Rig.submit w ~run:(Sub.Run.make ()) ~buffers:[| bd |])
  in
  write (memory ());
  write (memory ());
  b

let row name setup f = Thumper.bench_with_setup ~setup name f
let create d n () = ignore (B.create d n)

let buffer_rows =
  Thumper.group "buffer"
    [
      Thumper.bench "host-create-16" (create Rig.host 16);
      Thumper.bench "host-create-1M" (create Rig.host mib);
      row "memory-create-cached-4K" memory (fun d -> create d (4 * kib) ());
      row "wait-reached-24"
        (fun () -> written (memory ()) slots)
        (fun bs ->
          for i = 0 to slots - 1 do
            B.wait (Array.unsafe_get bs i) Rig.Read_write
          done);
      row "wait-written-2-devices" written_twice (fun b -> B.wait b Rig.Read);
      (* A view of 64 MiB a memory device holds, read by the host directly: the
         view paces no collection. *)
      Thumper.bench_with_setup
        ~metrics:Thumper.Metric.[ wall_time; alloc_words; major_collections ]
        ~setup:(fun () -> B.create (memory ()) (64 * mib))
        "read-64M"
        (fun b -> B.read b Bigarray.char ignore);
      row "blit-from-string-4K"
        (fun () -> (String.make (4 * kib) 's', host (4 * kib)))
        (fun (s, b) -> B.blit_from_string s 0 b 0 (4 * kib));
      Thumper.bench_with_setup ~budgets:copy_64M
        ~setup:(fun () -> (host (64 * mib), Bytes.create (64 * mib)))
        "blit-to-bytes-64M"
        (fun (b, s) -> B.blit_to_bytes b 0 s 0 (64 * mib));
    ]

let claim_rows =
  Thumper.group "claim"
    [
      row "with-24"
        (fun () -> (List.init slots (fun _ -> host 16), [ host 16 ]))
        (fun (read, donate) -> Claim.with_ ~read ~donate ignore);
      (* A donation of one buffer, the call an operation makes per donated
         operand. *)
      row "with-donate-1"
        (fun () ->
          let b = host 16 in
          ([ b ], fun c -> Claim.exclusive c b))
        (fun (donate, f) -> Claim.with_ ~read:[] ~donate f);
      (* A donation's answer on memory no one shared, a thousand times under one
         with_. *)
      row "exclusive-1000"
        (fun () -> host 16)
        (fun b ->
          Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
              for _ = 1 to 1000 do
                ignore (Sys.opaque_identity (Claim.exclusive c b))
              done));
      (* A share of memory already shared, as a library makes on each value it
         hands out over the same memory. *)
      row "share-shared"
        (fun () ->
          let b = host 16 in
          Claim.share b;
          b)
        Claim.share;
      row "c-read-release-3"
        (fun () -> (host 16, host 16, host 16))
        (fun (a, b, c) -> claim_3 a b c);
    ]

(* A device whose memory the host does not address: a copy between it and the
   host goes through the host's staging memory. *)
let unaddressed () =
  incr opened;
  fst (P.open_ ~host_visible:false (strf "bench:%d" !opened))

(* A staged copy of [n] bytes after [closed] other devices copied through the
   host's staging memory and were closed: the staging memory's records follow
   the devices still open. *)
let staged ~closed n () =
  for _ = 1 to closed do
    let d = unaddressed () in
    B.copy ~src:(B.create Rig.host n) ~dst:(B.create d n);
    Rig.close d
  done;
  (B.create Rig.host n, B.create (unaddressed ()) n)

let copy_rows =
  let pair d n () = (B.create d n, B.create d n) in
  let copy (src, dst) = B.copy ~src ~dst in
  Thumper.group "copy"
    [
      row "host-4K" (pair Rig.host (4 * kib)) copy;
      Thumper.bench_with_setup ~budgets:copy_64M
        ~setup:(pair Rig.host (64 * mib))
        "host-64M" copy;
      row "memory-4K" (fun () -> pair (memory ()) (4 * kib) ()) copy;
      row "queue-64K"
        (fun () ->
          (B.create Rig.host (64 * kib), B.create (unaddressed ()) (64 * kib)))
        copy;
      row "staged-4K" (staged ~closed:0 (4 * kib)) copy;
      row "staged-4K/after-1024-closed" (staged ~closed:1024 (4 * kib)) copy;
    ]

(* A collection hands the memory of 1,000 dropped buffers back, and the next
   create drains it into the device's cache. *)
let dropped = 1000

let collect_releases d =
  for _ = 1 to 3 do
    Gc.full_major ();
    ignore (Sys.opaque_identity (B.create d 0))
  done

let[@inline never] warm_allocations d =
  ignore (Sys.opaque_identity (Array.init 128 (fun _ -> B.create d (4 * kib))))

let[@inline never] pending_buffers d n =
  let buffers = Array.init n (fun _ -> B.create d 64) in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  let run = Sub.Run.make () in
  Array.iter (fun b -> submit_read ~run s [| b |]) buffers

let[@inline never] pending_holds d n =
  let run = Sub.Run.make () in
  for _ = 1 to n do
    ignore (submit ~run (Sub.make ~keep:(Bytes.create 64) d [||]))
  done

let[@inline never] pending_images d n =
  let images =
    Array.init n (fun _ -> Result.get_ok (Rig.Image.load d "functions"))
  in
  ignore (submit ~run:(Sub.Run.make ()) (Sub.make d [||]));
  ignore (Sys.opaque_identity images)

(* Every measured create faces the same pending population. Setup warms reusable
   4 KiB memory before dropping the other objects; only teardown runs their
   work. The collector may recycle measured allocations in between. *)
let pending populate n () =
  incr opened;
  let d, p = P.open_ ~capacity:(n + drain) (strf "drain:%d" !opened) in
  warm_allocations d;
  collect_releases d;
  populate d n;
  collect_releases d;
  { d; p; n = P.queued p }

let completed populate n () =
  let t = pending populate n () in
  ignore (P.run t.p);
  collect_releases t.d;
  { t with n = 0 }

let closed_devices n () =
  for _ = 1 to n do
    let t = dev () in
    Rig.close t.d
  done;
  pending pending_buffers 0 ()

let finish_pending t =
  if P.queued t.p <> t.n then failwith "drain: the pending population changed";
  ignore (P.run t.p);
  collect_releases t.d;
  Rig.close t.d

let drain_rows =
  let allocation name setup =
    Thumper.bench_with_setup ~setup ~teardown:finish_pending name (fun t ->
        B.create t.d (4 * kib))
  in
  Thumper.group "drain"
    [
      row "dropped-1000" memory (fun d ->
          for _ = 1 to dropped do
            create d (4 * kib) ()
          done;
          Gc.full_major ();
          create d (4 * kib) ());
      allocation "allocate-4K/pending-0" (pending pending_buffers 0);
      allocation "allocate-4K/pending-buffers-64" (pending pending_buffers 64);
      allocation "allocate-4K/pending-buffers-1024"
        (pending pending_buffers 1024);
      allocation "allocate-4K/completed-buffers-1024"
        (completed pending_buffers 1024);
      allocation "allocate-4K/pending-holds-64" (pending pending_holds 64);
      allocation "allocate-4K/pending-holds-1024" (pending pending_holds 1024);
      allocation "allocate-4K/completed-holds-1024"
        (completed pending_holds 1024);
      allocation "allocate-4K/pending-images-64" (pending pending_images 64);
      allocation "allocate-4K/pending-images-1024" (pending pending_images 1024);
      allocation "allocate-4K/completed-images-1024"
        (completed pending_images 1024);
      allocation "allocate-4K/closed-devices-64" (closed_devices 64);
      allocation "allocate-4K/closed-devices-1024" (closed_devices 1024);
    ]

let wait_rows =
  Thumper.group "wait/memory"
    [
      row "reached"
        (fun () ->
          let d = memory () and run = Sub.Run.make () in
          let s = Sub.make d [||] in
          submit ~run s)
        Rig.Point.wait;
    ]

(* A host buffer of 64 MiB collected: the end of the cycle returns it, being
   more than the cache keeps. *)
(* A host buffer of 64 KiB taken and dropped; the minor collection hands its
   memory back to the cache, from which the next take comes. *)
let take () =
  create Rig.host (64 * kib) ();
  Gc.minor ()

(* The host cache holding [others] buffers of other sizes, 68 KiB to 464 KiB, 26
   MiB in all, under the cache's floor: a take walks only its own size's. *)
let others = 100

let caching () =
  let keep = List.init others (fun i -> host ((17 + i) * 4 * kib)) in
  ignore (Sys.opaque_identity keep);
  Gc.full_major ()

(* Eight host buffers of 8 MiB held at once, a word of each 4 KiB written, then
   dropped: 64 MiB a round, past the cache's floor. The collector frees a
   round's buffers while the next rounds run, which take them from the cache
   rather than fault fresh pages in. *)
let round () =
  let bs = Array.init 8 (fun _ -> host (8 * mib)) in
  Array.iter
    (fun b ->
      let a = Support.Reader.host b in
      for i = 0 to (8 * mib / (4 * kib)) - 1 do
        Support.store (a + (i * 4 * kib)) i
      done)
    bs

let heap_rows =
  Thumper.group "heap"
    [
      Thumper.bench "trim-64M" (fun () ->
          create Rig.host (64 * mib) ();
          Gc.full_major ());
      Thumper.bench "take-64K" take;
      row "take-64K-cached-100" caching take;
      Thumper.bench "reuse-8M-x8" round;
    ]

let memory_floor_rows =
  (* A word of host memory, beside the buffer that holds it. *)
  let word () =
    let b = host 8 in
    (b, Support.Reader.host b)
  in
  let blit (src, dst) = Bigarray.Array1.blit src dst in
  let chars2 n () = (chars n, chars n) in
  [
    Thumper.bench "bigarray-create-16" (fun () -> chars 16);
    Thumper.bench "bigarray-create-1M" (fun () -> chars mib);
    row "word-load" word (fun (_, a) -> Support.load a);
    row "word-load-24" word (fun (_, a) ->
        for _ = 1 to slots do
          ignore (Support.load a)
        done);
    row "atomic-cas-2"
      (fun () -> Atomic.make 0)
      (fun a ->
        ignore (Atomic.compare_and_set a 0 1);
        ignore (Atomic.compare_and_set a 1 0));
    row "memcpy-4K" (chars2 (4 * kib)) blit;
    Thumper.bench_with_setup ~budgets:copy_64M
      ~setup:(chars2 (64 * mib))
      "memcpy-64M" blit;
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
   as rig reads it. *)
type floor = {
  f : nativeint;
  fp : P.t;
  word : B.t; (* Host memory the floor rewrites before each submit. *)
  timeline : int;
  mutable k : int;
}

let floor () =
  let fp = P.make () in
  let f = floor_new (P.facts fp).edge Support.noop in
  let timeline = Option.get (P.locate (P.facts fp).word).address in
  { f; fp; word = B.create Rig.host 8; timeline; k = 0 }

let floor_drained t =
  t.k <- t.k + 1;
  if t.k mod drain = 0 then ignore (P.run t.fp)

let floor_replay () =
  let t = floor () in
  let params = Bytes.make 16 '\000' in
  Bytes.set_int64_ne params 0 (Int64.of_int (Support.Reader.host t.word));
  floor_fill t.f Support.poke (Bytes.unsafe_to_string params);
  t

let floor_run t =
  t.k <- t.k + 1;
  ignore (P.run t.fp);
  floor_int64 t.f 8 t.k;
  floor_submit t.f 1

(* Another domain calling the same device's entries until the row ends, with a
   floor of its own: the driver alone serializes the two. With [share], each
   submit also takes one turn the two share, as rig takes a device's: by
   try-lock, and otherwise with the runtime released. *)
let floor_contended ~share () =
  let t = floor () in
  let stop = Atomic.make false in
  let other =
    { t with f = floor_new (P.facts t.fp).edge Support.noop; k = 0 }
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
  let replay_row name f = Thumper.bench_with_setup ~setup:floor_replay name f in
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
       replay_row "polled/run" floor_run;
       replay_row "polled/pipelined-100" (fun t ->
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
