(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call fills its domain's view and plan, writes the plan into its run's
   blocks and submits the submission of the plan's sequence, made once per
   device, through the door:

   contract --> device (== scan) --> frame (the domain's, or a fresh one) -->
   View.fill, Plan.choose --> workspace, submission --> Plan.write into the run
   --> door --> Rig.submit

   A call that finds every cache warm allocates nothing. *)

module A = Nx_array
module V = Nx_kernel.Spec.Contract_view
module Sub = Rig.Submission

let name = "nx.amd"

external code_object : string -> string option = "nx_amd_code_object"

(* Devices *)

(* The workspace holds a call's packed operands and split sums. It is named in
   each call's writes, so rig orders the calls that share it. It grows to [kept]
   bytes and is kept; a call that needs more takes a buffer of its own. *)
type workspace = { buffer : Rig.Buffer.t; bytes : int }

let kept = 64 * 1024 * 1024

type device = {
  image : Rig.Image.t;
  queue : string;  (** The queue that runs launches. *)
  workspace : workspace Atomic.t;
  subs : Sub.t option array;  (** By {!Plan.sequence}, made on first use. *)
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
      match code_object (Rig.arch d) with
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
                  subs = Array.make Plan.sequences None;
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

(* The workspace of at least [need] bytes. A race stores only a larger one, so
   it never shrinks. *)
let workspace d dv need =
  let w = Atomic.get dv.workspace in
  if w.bytes >= need then w.buffer
  else if need > kept then Rig.Buffer.create d need
  else begin
    let bytes = Int.min kept (Int.max need (2 * w.bytes)) in
    let fresh = { buffer = Rig.Buffer.create d bytes; bytes } in
    let rec store () =
      let w = Atomic.get dv.workspace in
      if w.bytes >= need then w.buffer
      else if Atomic.compare_and_set dv.workspace w fresh then fresh.buffer
      else store ()
    in
    store ()
  end

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

(* Frames *)

(* A call's state: one per domain, guarded by [busy]. A call that finds it busy,
   such as one of another systhread of the domain or a signal handler's, uses a
   fresh frame it does not keep. *)
type frame = {
  busy : bool Atomic.t;
  view : V.t;
  plan : Plan.t;
  run : Sub.Run.t;
  mutable sub : Sub.t option;
  reads2 : Rig.Buffer.t array;
  reads3 : Rig.Buffer.t array;
  writes1 : Rig.Buffer.t array;
  writes2 : Rig.Buffer.t array;
  read2 : A.any array;
  read3 : A.any array;
  written : A.any array;
  mutable reads : Rig.Buffer.t array;
  mutable writes : Rig.Buffer.t array;
  mutable read : A.any array;
}

(* What a frame holds between calls: no user buffer. *)
let no_buffer = Rig.Buffer.of_string ""
let no_array = A.Any (A.create Rig.host A.Dtype.Uint8 [| 0 |])

let frame ~busy =
  let reads2 = Array.make 2 no_buffer and writes1 = Array.make 1 no_buffer in
  let read2 = Array.make 2 no_array in
  {
    busy = Atomic.make busy;
    view = V.make ();
    plan = Plan.make ();
    run = Sub.Run.make ();
    sub = None;
    reads2;
    reads3 = Array.make 3 no_buffer;
    writes1;
    writes2 = Array.make 2 no_buffer;
    read2;
    read3 = Array.make 3 no_array;
    written = Array.make 1 no_array;
    reads = reads2;
    writes = writes1;
    read = read2;
  }

let frames = Domain.DLS.new_key (fun () -> frame ~busy:false)

let take () =
  let f = Domain.DLS.get frames in
  if Atomic.compare_and_set f.busy false true then f else frame ~busy:true

let release f =
  Array.fill f.reads3 0 3 no_buffer;
  Array.fill f.reads2 0 2 no_buffer;
  Array.fill f.writes2 0 2 no_buffer;
  f.writes1.(0) <- no_buffer;
  Array.fill f.read3 0 3 no_array;
  Array.fill f.read2 0 2 no_array;
  f.written.(0) <- no_array;
  f.sub <- None;
  Atomic.set f.busy false

(* Makes the frame's arrays hold the call's buffers: [a], [b] and [init] read,
   [dst] and the workspace written. *)
let bind f ~dst ops ws =
  let n = Array.length ops in
  f.reads <- (if n = 3 then f.reads3 else f.reads2);
  f.read <- (if n = 3 then f.read3 else f.read2);
  for i = 0 to n - 1 do
    let (A.Any x) = ops.(i) in
    f.reads.(i) <- A.buffer x;
    f.read.(i) <- ops.(i)
  done;
  let (A.Any y) = dst in
  f.writes <- (if Plan.writes f.plan = 2 then f.writes2 else f.writes1);
  f.writes.(0) <- A.buffer y;
  if Plan.writes f.plan = 2 then f.writes.(1) <- ws;
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
        let sub = submission d dv f.plan in
        f.sub <- sub;
        Plan.write f.run (Option.get sub) f.plan;
        bind f ~dst ops ws;
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

(* Elementwise *)

let apply0 _ ~dst:_ = A.Declined
let apply1 _ ~dst:_ _ = A.Declined
let apply2 _ ~dst:_ _ _ = A.Declined
let apply3 _ ~dst:_ _ _ _ = A.Declined
let map _ ~dsts:_ _ = A.Declined
let reduce _ ~dsts:_ _ = A.Declined
let scan _ ~dsts:_ _ = A.Declined
let gather _ ~dst:_ _ _ = A.Declined
let scatter _ ~dst:_ ~into:_ _ _ = A.Declined
let sort _ ~values:_ ~positions:_ _ = A.Declined
let assemble _ ~dst:_ _ = A.Declined
let fold _ ~dst:_ _ = A.Declined
