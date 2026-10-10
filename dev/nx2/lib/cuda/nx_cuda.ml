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

(* A device keeps two buffers, each named [Read_write] by every call that uses
   it, so rig orders those calls: the workspace, which holds a call's packed
   operands and split sums, grows to [kept] bytes and is kept, a call that
   needs more taking a buffer of its own; the tickets of split sums are zero
   words, which every call leaves zero. *)
let kept = 64 * 1024 * 1024

(* What this library keeps for a device: [queue] runs launches, and [subs] and
   [folds] hold the submissions by {!Plan.sequence} and {!Fold.sequence}, made
   on first use. *)
type device = {
  image : Rig.Image.t;
  queue : string;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : Sub.t option array;
  folds : Sub.t option array;
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
              Some
                {
                  image;
                  queue;
                  workspace = Atomic.make (Rig.Buffer.create d 0);
                  tickets = Atomic.make (Rig.Buffer.create d 0);
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

(* [cell]'s buffer of at least [need] bytes, grown to twice its length or
   [need] by [make] at most [cap]. A race stores only a larger one, so it
   never shrinks. A call whose buffer is large enough reads it without
   reaching here, which would allocate [make]'s closure. *)
let grow cell need ~cap make =
  let w = Atomic.get cell in
  let fresh = make (Int.min cap (Int.max need (2 * Rig.Buffer.length w))) in
  let rec store () =
    let w = Atomic.get cell in
    if Rig.Buffer.length w >= need then w
    else if Atomic.compare_and_set cell w fresh then fresh
    else store ()
  in
  store ()

let workspace d dv need =
  let w = Atomic.get dv.workspace in
  if Rig.Buffer.length w >= need then w
  else if need > kept then Rig.Buffer.create d need
  else grow dv.workspace need ~cap:kept (Rig.Buffer.create d)

let zeros d bytes =
  let b = Rig.Buffer.create d bytes in
  Rig.Buffer.copy ~src:(Rig.Buffer.of_string (String.make bytes '\000')) ~dst:b;
  b

let tickets d dv need =
  let w = Atomic.get dv.tickets in
  if Rig.Buffer.length w >= need then w
  else grow dv.tickets need ~cap:max_int (zeros d)

(* The submission of [p]'s sequence on [d]. Two domains that make one at once
   each make it, and the last store wins. *)
let submission d dv p =
  let key = Plan.sequence p in
  match dv.subs.(key) with
  | Some _ as s -> s
  | None ->
      let parts = Plan.parts p dv.image ~queue:dv.queue in
      let s = Some (Sub.make ~access:(Plan.access p) d parts) in
      dv.subs.(key) <- s;
      s

let fold_submission d dv p =
  let key = Fold.sequence p in
  match dv.folds.(key) with
  | Some _ as s -> s
  | None ->
      let parts = Fold.parts p dv.image ~queue:dv.queue in
      let s = Some (Sub.make ~access:(Fold.access p) d parts) in
      dv.folds.(key) <- s;
      s

(* Frames *)

(* A call's state: one per domain, guarded by [busy]. A call that finds it busy,
   such as one of another systhread of the domain or a signal handler's, uses a
   fresh frame it does not keep. A call's buffers and arrays sit in arrays of
   their count, made once: [buffers] are the submit's, up to 6, in its slot
   order, and [read] and [written] the door's, up to 3. *)
type frame = {
  busy : bool Atomic.t;
  view : V.t;
  plan : Plan.t;
  fold : Fold.t;
  run : Sub.Run.t;
  mutable sub : Sub.t option;
  buffer_arrays : Rig.Buffer.t array array;
  read_arrays : A.any array array;
  written_arrays : A.any array array;
  mutable buffers : Rig.Buffer.t array;
  mutable read : A.any array;
  mutable written : A.any array;
}

(* What a frame holds between calls: no user buffer. *)
let no_buffer = Rig.Buffer.of_string ""
let no_array = A.Any (A.create Rig.host A.Dtype.Uint8 [| 0 |])
let by_count most x = Array.init (most + 1) (fun n -> Array.make n x)

let frame ~busy =
  let read_arrays = by_count 3 no_array in
  {
    busy = Atomic.make busy;
    view = V.make ();
    plan = Plan.make ();
    fold = Fold.make ();
    run = Sub.Run.make ();
    sub = None;
    buffer_arrays = by_count 6 no_buffer;
    read_arrays;
    written_arrays = by_count 3 no_array;
    buffers = [||];
    read = read_arrays.(0);
    written = read_arrays.(0);
  }

let frames = Domain.DLS.new_key (fun () -> frame ~busy:false)

let take () =
  let f = Domain.DLS.get frames in
  if Atomic.compare_and_set f.busy false true then f else frame ~busy:true

let release f =
  for n = 1 to 6 do
    Array.fill f.buffer_arrays.(n) 0 n no_buffer
  done;
  for n = 1 to 3 do
    Array.fill f.read_arrays.(n) 0 n no_array;
    Array.fill f.written_arrays.(n) 0 n no_array
  done;
  f.sub <- None;
  Atomic.set f.busy false

let buffer (A.Any x) = A.buffer x

(* Makes the door's arrays hold [ops] read and [dst] written, and [buffers]
   the submit's [n] buffers, which the caller fills. *)
let bind f ~dst ops n =
  let k = Array.length ops in
  f.read <- f.read_arrays.(k);
  for i = 0 to k - 1 do
    f.read.(i) <- ops.(i)
  done;
  f.written <- f.written_arrays.(1);
  f.written.(0) <- dst;
  f.buffers <- f.buffer_arrays.(n)

(* The door's work: [f.sub], which the call stored before it entered the
   door. *)
let issue f =
  ignore
    (Rig.submit (Option.get f.sub) ~run:f.run ~buffers:f.buffers ~waits:[||])

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
        (* {!Plan.access}'s slots: [a], [b], [dst], then [init], the
           workspace and the tickets where the plan has them. *)
        let n = Array.length ops in
        let uses_ws = Plan.workspace f.plan > 0 in
        let uses_tk = Plan.tickets f.plan > 0 in
        let k = n + 1 + Bool.to_int uses_ws + Bool.to_int uses_tk in
        bind f ~dst ops k;
        let b = f.buffers in
        b.(0) <- buffer ops.(0);
        b.(1) <- buffer ops.(1);
        b.(2) <- buffer dst;
        if n = 3 then b.(3) <- buffer ops.(2);
        if uses_ws then b.(n + 1) <- ws;
        if uses_tk then b.(k - 1) <- tk;
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
      (* {!Fold.access}'s slots: [x], [dst], then the workspace. *)
      let uses_ws = Fold.workspace f.fold > 0 in
      bind f ~dst ops (2 + Bool.to_int uses_ws);
      f.buffers.(0) <- buffer x;
      f.buffers.(1) <- buffer dst;
      if uses_ws then f.buffers.(2) <- ws;
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

(* Gathers, scatters, sorts, assemblies and folds *)

let gather _ ~dst:_ _ _ = A.Declined
let scatter _ ~dst:_ ~into:_ _ _ = A.Declined
let sort _ ~values:_ ~positions:_ _ = A.Declined
let assemble _ ~dst:_ _ = A.Declined
let fold _ ~dst:_ _ = A.Declined
(* Declined, once the dtypes are the ones [s] takes and gives. *)
let decline s ~dsts ops =
  let dtype (A.Any x) = A.Dtype.Any (A.dtype x) in
  match Nx_kernel.Spec.dtypes s (Array.map dtype ops) with
  | Ok ds when ds = Array.map dtype dsts -> A.Declined
  | _ -> A.Wrong_dtype

let fft s ~dst x = decline s ~dsts:[| dst |] [| x |]
let linalg s ~dsts ops = decline s ~dsts ops
