(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call fills its domain's plan, writes the plan into its run's blocks and
   submits the submission of the plan's sequence, made once per device,
   through the door:

   contract --> device (== scan) --> frame (the domain's, or a fresh one) -->
   View.fill, Plan.choose --> workspace, submission --> Plan.write into the run
   --> door --> Rig.submit

   reduce and scan take the same path with Fold's plan. A call that finds
   every cache warm allocates nothing. *)

module A = Nx_array
module V = Nx_kernel.Spec.Contract_view
module Sub = Rig.Submission

let name = "nx.cuda"

external cubin : string -> string option = "nx_cuda_cubin"

(* Devices *)

(* A device's buffer, of [bytes] bytes, named in the writes of each call that
   uses it, so rig orders those calls: the workspace, which holds a call's
   packed operands and split sums, grows to [kept] bytes and is kept, a call
   that needs more taking a buffer of its own; the tickets of split sums are
   zero words, which every call leaves zero. *)
type workspace = { buffer : Rig.Buffer.t; bytes : int }

let kept = 64 * 1024 * 1024

type device = {
  image : Rig.Image.t;
  queue : string;  (** The queue that runs launches. *)
  workspace : workspace Atomic.t;
  tickets : workspace Atomic.t;
  subs : Sub.t option array;  (** By {!Plan.sequence}, made on first use. *)
  folds : Sub.t option array;  (** By {!Fold.sequence}. *)
}

(* Every device asked about and what this library keeps for it, [None] where it
   does not compute. Replaced whole under [lock], read without it. *)
let devices : (Rig.t * device option) array Atomic.t = Atomic.make [||]
let lock = Mutex.create ()

(* The index of [d] in [ds] from [i], or -1: a loop with its state as arguments,
   which allocates no closure. *)
let rec find ds d i =
  if i = Array.length ds then -1
  else if fst (Array.unsafe_get ds i) == d then i
  else find ds d (i + 1)

let launch_queue d =
  List.find_map
    (fun (q : Rig.queue) ->
      if List.mem Rig.Launch q.runs then Some q.name else None)
    (Rig.queues d)

let load d =
  match launch_queue d with
  | None -> None
  | Some queue -> (
      match cubin (Rig.arch d) with
      | None -> None
      | Some bin -> (
          match Rig.Image.load d bin with
          | Error _ -> None
          | Ok image ->
              let none = { buffer = Rig.Buffer.create d 0; bytes = 0 } in
              Some
                {
                  image;
                  queue;
                  workspace = Atomic.make none;
                  tickets = Atomic.make none;
                  subs = Array.make Plan.sequences None;
                  folds = Array.make Fold.sequences None;
                }))

let device d =
  let ds = Atomic.get devices in
  let i = find ds d 0 in
  if i >= 0 then snd ds.(i)
  else
    Mutex.protect lock (fun () ->
        let ds = Atomic.get devices in
        let i = find ds d 0 in
        if i >= 0 then snd ds.(i)
        else
          let dv = load d in
          Atomic.set devices (Array.append ds [| (d, dv) |]);
          dv)

let computes_on d = Option.is_some (device d)

(* [cell]'s buffer of at least [need] bytes, grown to twice its bytes or
   [need] by [make] at most [cap]. A race stores only a larger one, so it
   never shrinks. A call whose buffer is large enough reads it without
   reaching here, which would allocate [make]'s closure. *)
let grow cell need ~cap make =
  let w = Atomic.get cell in
  let bytes = Int.min cap (Int.max need (2 * w.bytes)) in
  let fresh = { buffer = make bytes; bytes } in
  let rec store () =
    let w = Atomic.get cell in
    if w.bytes >= need then w.buffer
    else if Atomic.compare_and_set cell w fresh then fresh.buffer
    else store ()
  in
  store ()

let workspace d dv need =
  let w = Atomic.get dv.workspace in
  if w.bytes >= need then w.buffer
  else if need > kept then Rig.Buffer.create d need
  else grow dv.workspace need ~cap:kept (Rig.Buffer.create d)

let zeros d bytes =
  let b = Rig.Buffer.create d bytes in
  Rig.Buffer.copy ~src:(Rig.Buffer.of_string (String.make bytes '\000')) ~dst:b;
  b

let tickets d dv need =
  let w = Atomic.get dv.tickets in
  if w.bytes >= need then w.buffer
  else grow dv.tickets need ~cap:max_int (zeros d)

(* The submission of [p]'s sequence on [d]. Two domains that make one at once
   each make it, and the last store wins. *)
let submission d dv p =
  let key = Plan.sequence p in
  match dv.subs.(key) with
  | Some _ as s -> s
  | None ->
      let parts = Plan.parts p dv.image ~queue:dv.queue in
      let s =
        Some (Sub.make ~reads:(Plan.reads p) ~writes:(Plan.writes p) d parts)
      in
      dv.subs.(key) <- s;
      s

let fold_submission d dv p =
  let key = Fold.sequence p in
  match dv.folds.(key) with
  | Some _ as s -> s
  | None ->
      let parts = Fold.parts p dv.image ~queue:dv.queue in
      let s = Some (Sub.make ~reads:1 ~writes:(Fold.writes p) d parts) in
      dv.folds.(key) <- s;
      s

(* Frames *)

(* A call's state: one per domain, guarded by [busy]. A call that finds it busy,
   such as one of another systhread of the domain or a signal handler's, uses a
   fresh frame it does not keep. A call's buffers and arrays sit in arrays of
   their count, made once: [reads] and [writes] are the submit's, [read] and
   [written] the door's. *)
type frame = {
  busy : bool Atomic.t;
  view : V.t;
  plan : Plan.t;
  fold : Fold.t;
  run : Sub.Run.t;
  mutable sub : Sub.t option;
  read_buffers : Rig.Buffer.t array array;  (** By count, up to 3. *)
  write_buffers : Rig.Buffer.t array array;
  read_arrays : A.any array array;
  written_arrays : A.any array array;
  mutable reads : Rig.Buffer.t array;
  mutable writes : Rig.Buffer.t array;
  mutable read : A.any array;
  mutable written : A.any array;
}

(* What a frame holds between calls: no user buffer. *)
let no_buffer = Rig.Buffer.of_string ""
let no_array = A.Any (A.create Rig.host A.Dtype.Uint8 [| 0 |])
let by_count x = Array.init 4 (fun n -> Array.make n x)

let frame ~busy =
  let read_buffers = by_count no_buffer and read_arrays = by_count no_array in
  {
    busy = Atomic.make busy;
    view = V.make ();
    plan = Plan.make ();
    fold = Fold.make ();
    run = Sub.Run.make ();
    sub = None;
    read_buffers;
    write_buffers = by_count no_buffer;
    read_arrays;
    written_arrays = by_count no_array;
    reads = read_buffers.(0);
    writes = read_buffers.(0);
    read = read_arrays.(0);
    written = read_arrays.(0);
  }

let frames = Domain.DLS.new_key (fun () -> frame ~busy:false)

let take () =
  let f = Domain.DLS.get frames in
  if Atomic.compare_and_set f.busy false true then f else frame ~busy:true

let release f =
  for n = 1 to 3 do
    Array.fill f.read_buffers.(n) 0 n no_buffer;
    Array.fill f.write_buffers.(n) 0 n no_buffer;
    Array.fill f.read_arrays.(n) 0 n no_array;
    Array.fill f.written_arrays.(n) 0 n no_array
  done;
  f.sub <- None;
  Atomic.set f.busy false

(* Makes the frame's arrays hold the call's buffers: [ops] read, [dst], then
   [w1] and [w2] while [writes] counts them, written. *)
let bind f ~dst ops ~writes w1 w2 =
  let n = Array.length ops in
  f.reads <- f.read_buffers.(n);
  f.read <- f.read_arrays.(n);
  for i = 0 to n - 1 do
    let (A.Any x) = ops.(i) in
    f.reads.(i) <- A.buffer x;
    f.read.(i) <- ops.(i)
  done;
  let (A.Any y) = dst in
  f.writes <- f.write_buffers.(writes);
  f.writes.(0) <- A.buffer y;
  if writes >= 2 then f.writes.(1) <- w1;
  if writes = 3 then f.writes.(2) <- w2;
  f.written <- f.written_arrays.(1);
  f.written.(0) <- dst

let issue f =
  match f.sub with
  | None -> ()
  | Some s ->
      ignore
        (Rig.submit s ~run:f.run ~reads:f.reads ~writes:f.writes ~waits:[||])

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead ops i =
  i < Array.length ops && (dead ops.(i) || any_dead ops (i + 1))

(* Contractions *)

let run d dv f s ~dst ops =
  if not (V.fill f.view s ~dst ops) then A.Declined
  else if dead dst || any_dead ops 0 then
    (* The door refuses a dead buffer, before the plan reads its address. *)
    A.door ~written:[| dst |] ~read:ops nothing ()
  else
    match Plan.choose f.plan f.view s ~dst ops with
    | Declined -> A.Declined
    | Nothing -> A.door ~written:[| dst |] ~read:ops nothing ()
    | Launches ->
        let ws = workspace d dv (Plan.workspace f.plan) in
        let tk = tickets d dv (Plan.tickets f.plan) in
        let sub = submission d dv f.plan in
        f.sub <- sub;
        Plan.write f.run (Option.get sub) f.plan;
        bind f ~dst ops ~writes:(Plan.writes f.plan) ws tk;
        A.door ~written:f.written ~read:f.read issue f

let contract s ~dst ops =
  let (A.Any y) = dst in
  let d = A.device y in
  match device d with
  | None -> A.Declined
  | Some dv -> (
      let f = take () in
      match run d dv f s ~dst ops with
      | answer ->
          release f;
          answer
      | exception e ->
          release f;
          raise e)

(* Reductions and scans *)

let fold d dv f family s ~dst x =
  match Fold.choose f.fold family s ~dst x with
  | Declined -> A.Declined
  | Refused r -> r
  | Nothing -> A.door ~written:[| dst |] ~read:[| x |] nothing ()
  | Launches ->
      let ws = workspace d dv (Fold.workspace f.fold) in
      let sub = fold_submission d dv f.fold in
      f.sub <- sub;
      Fold.write f.run (Option.get sub) f.fold;
      let ops = f.read_arrays.(1) in
      ops.(0) <- x;
      bind f ~dst ops ~writes:(Fold.writes f.fold) ws no_buffer;
      A.door ~written:f.written ~read:f.read issue f

(* The case Fold computes has one destination and one operand. *)
let folds family s ~dsts ops =
  if Array.length dsts <> 1 || Array.length ops <> 1 then A.Declined
  else
    let (A.Any y) = dsts.(0) in
    let d = A.device y in
    match device d with
    | None -> A.Declined
    | Some dv -> (
        let f = take () in
        match fold d dv f family s ~dst:dsts.(0) ops.(0) with
        | answer ->
            release f;
            answer
        | exception e ->
            release f;
            raise e)

let reduce s ~dsts ops = folds `Reduce s ~dsts ops
let scan s ~dsts ops = folds `Scan s ~dsts ops

(* Elementwise *)

let apply0 _ ~dst:_ = A.Declined
let apply1 _ ~dst:_ _ = A.Declined
let apply2 _ ~dst:_ _ _ = A.Declined
let apply3 _ ~dst:_ _ _ _ = A.Declined
let map _ ~dsts:_ _ = A.Declined
