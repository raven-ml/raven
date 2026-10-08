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

module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let strf = Printf.sprintf

external floor_new :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint
  = "device_core_bench_floor_new"

external floor_submit : nativeint -> int -> unit
  = "device_core_bench_floor_submit"
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

let words d n = Array.init n (fun _ -> B.create d S.UInt64 1)

(* A part that adds 1 to the word [arg] holds. *)
let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let bumping ?(reads = 0) ?(writes = 0) d =
  Sub.make ~reads ~writes ~waits:0 d [| bump (B.create C.host S.UInt64 1) |]

let submit_read s bs =
  for i = 0 to Array.length bs - 1 do
    Sub.read s i (Array.unsafe_get bs i)
  done;
  ignore (C.submit s)

(* Submits *)

let empty () =
  let t = dev () in
  (t, Sub.make ~reads:0 ~writes:0 ~waits:0 t.d [||])

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
  let w = Sub.make ~reads:0 ~writes:slots ~waits:0 o.d [||] in
  Array.iteri (fun i b -> Sub.write w i b) bs;
  let v = C.Point.value (C.submit w) in
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
          ignore (C.submit s');
          drained other
        done)
  in
  (t, s, stop, rival)

let submit_rows =
  let row name setup f = Thumper.bench_with_setup ~setup name f in
  Thumper.group "submit/polled"
    [
      row "empty" empty (fun (t, s) ->
          let v = C.Point.value (C.submit s) in
          ignore (P.run t.p);
          C.wait t.d v);
      row "cost" costing (fun (t, s) ->
          ignore (C.submit s);
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
          ignore (C.submit s);
          drained t);
    ]

(* Replays *)

type copy = { s : Sub.t; args : B.t; at : int; out : B.t }
type replay = { t : dev; params : B.t array; copies : copy array }

(* Two copies of a step over [slots] parameters: each reads them, writes its
   output, and fills its own arguments, which the host rewrites before each
   run. *)
let replaying () =
  let t = dev () in
  let copy () =
    let args = B.create C.host S.UInt64 1 in
    let s = Sub.make ~reads:slots ~writes:1 ~waits:0 t.d [| bump args |] in
    { s; args; at = B.address args; out = B.create t.d S.UInt64 1 }
  in
  { t; params = words t.d slots; copies = [| copy (); copy () |] }

let run r =
  let c = r.copies.(r.t.n land 1) in
  r.t.n <- r.t.n + 1;
  B.wait c.args B.Read_write;
  ignore (P.run r.t.p);
  Support.store c.at r.t.n;
  for i = 0 to slots - 1 do
    Sub.read c.s i (Array.unsafe_get r.params i)
  done;
  Sub.write c.s 0 c.out;
  ignore (C.submit c.s)

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

(* Floors *)

type floor = { f : nativeint; fp : P.t; word : int; mutable k : int }

let floor () =
  let fp = P.make () in
  let f = floor_new (P.self fp) P.room_entry P.submit_entry Support.bump in
  { f; fp; word = B.address (B.create C.host S.UInt64 1); k = 0 }

let floor_drained t =
  t.k <- t.k + 1;
  if t.k mod drain = 0 then ignore (P.run t.fp)

let floor_run t =
  t.k <- t.k + 1;
  ignore (P.run t.fp);
  Support.store t.word t.k;
  floor_submit t.f 1

(* Another domain calling the same device's entries until the row ends, with a
   floor of its own: the driver alone serializes the two. *)
let floor_contended () =
  let t = floor () in
  let stop = Atomic.make false in
  let other =
    {
      t with
      f = floor_new (P.self t.fp) P.room_entry P.submit_entry Support.bump;
      k = 0;
    }
  in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          floor_submit other.f 1;
          floor_drained other
        done)
  in
  (t, stop, rival)

let floor_rows =
  let row name f = Thumper.bench_with_setup ~setup:floor name f in
  Thumper.group "floor"
    [
      row "polled/release" (fun t ->
          floor_submit t.f 0;
          ignore (P.run t.fp);
          ignore (P.signaled t.fp));
      row "polled/cost" (fun t ->
          floor_submit t.f 1;
          floor_drained t);
      Thumper.bench_with_setup ~setup:floor_contended
        ~teardown:(fun (_, stop, rival) ->
          Atomic.set stop true;
          Domain.join rival)
        "polled/two-domains"
        (fun (t, _, _) ->
          floor_submit t.f 1;
          floor_drained t);
      row "polled/run" floor_run;
      row "polled/pipelined-100" (fun t ->
          for _ = 1 to runs do
            floor_run t
          done;
          ignore (P.run t.fp);
          ignore (P.signaled t.fp));
      Thumper.bench_with_setup ~setup:Mutex.create "mutex-section" (fun m ->
          Mutex.lock m;
          Mutex.unlock m);
    ]

let () =
  exit @@ Thumper.run "device_core" [ submit_rows; replay_rows; floor_rows ]
