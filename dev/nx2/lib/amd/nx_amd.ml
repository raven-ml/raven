(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call fills its domain's view and plan, writes the plan into its run's
   blocks and submits the submission of the plan's sequence, made once per
   device, through the door:

   contract --> device (Rig.Local) --> frame (the domain's, or a fresh one) -->
   View.fill, Plan.choose --> workspace, submission --> Plan.write into the run
   --> door --> Rig.submit

   A call that finds every cache warm allocates nothing. *)

module A = Nx_array
module V = Nx_kernel.Contract_view
module Sub = Rig.Submission

let name = "nx.amd"

external code_object : string -> string option = "nx_amd_code_object"

(* Devices *)

(* The workspace holds a call's packed operands and split sums. Each call that
   uses it names it [Read_write], so rig orders the calls that share it. It
   grows to [kept] bytes and is kept; a call that needs more takes a buffer of
   its own. *)
let kept = 64 * 1024 * 1024

(* What this library keeps for a device: [subs]
   holds the submissions by {!Plan.sequence}, made on first use. *)
type device = {
  image : Rig.Image.t;
  workspace : Rig.Buffer.t Atomic.t;
  subs : Sub.t option array;
}

let load d =
  match code_object (Rig.arch d) with
  | None -> None
  | Some bin -> (
      match Rig.Image.load d bin with
      | Error _ -> None
      | Ok image ->
          Some
            {
              image;
              workspace = Atomic.make (Rig.Buffer.create d 0);
              subs = Array.make Plan.sequences None;
            })

(* Each device's [load], [None] where it does not compute. *)
let devices = Rig.Local.new_key load
let device d = Rig.Local.get devices d

let computes_on d = Option.is_some (device d)

(* The workspace of at least [need] bytes. A race stores only a larger one, so
   it never shrinks. *)
let workspace d dv need =
  let w = Atomic.get dv.workspace in
  if Rig.Buffer.length w >= need then w
  else if need > kept then Rig.Buffer.create d need
  else begin
    let bytes = Int.min kept (Int.max need (2 * Rig.Buffer.length w)) in
    let fresh = Rig.Buffer.create d bytes in
    let rec store () =
      let w = Atomic.get dv.workspace in
      if Rig.Buffer.length w >= need then w
      else if Atomic.compare_and_set dv.workspace w fresh then fresh
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
      let parts = Plan.parts p dv.image in
      let s = Some (Sub.make d parts) in
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
  buffers3 : Rig.Buffer.t array;
  buffers4 : Rig.Buffer.t array;
  buffers5 : Rig.Buffer.t array;
  read2 : A.any array;
  read3 : A.any array;
  written : A.any array;
  mutable buffers : Rig.Buffer.t array;
  mutable read : A.any array;
}

(* What a frame holds between calls: no user buffer. *)
let no_buffer = Rig.Buffer.of_string ""
let no_array = A.Any (A.create Rig.host A.Dtype.Uint8 [| 0 |])

let frame ~busy =
  let buffers3 = Array.make 3 no_buffer and read2 = Array.make 2 no_array in
  {
    busy = Atomic.make busy;
    view = V.make ();
    plan = Plan.make ();
    run = Sub.Run.make ();
    sub = None;
    buffers3;
    buffers4 = Array.make 4 no_buffer;
    buffers5 = Array.make 5 no_buffer;
    read2;
    read3 = Array.make 3 no_array;
    written = Array.make 1 no_array;
    buffers = buffers3;
    read = read2;
  }

let frames = Domain.DLS.new_key (fun () -> frame ~busy:false)

let take () =
  let f = Domain.DLS.get frames in
  if Atomic.compare_and_set f.busy false true then f else frame ~busy:true

let release f =
  Array.fill f.buffers3 0 3 no_buffer;
  Array.fill f.buffers4 0 4 no_buffer;
  Array.fill f.buffers5 0 5 no_buffer;
  Array.fill f.read3 0 3 no_array;
  Array.fill f.read2 0 2 no_array;
  f.written.(0) <- no_array;
  f.sub <- None;
  Atomic.set f.busy false

let buffer (A.Any x) = A.buffer x

(* Makes the frame's [read] and [written] hold the call's arrays: [a], [b],
   then [init] where given, and [dst]. *)
let hold f ~dst ~init a b =
  f.written.(0) <- A.Any dst;
  match init with
  | None ->
      f.read <- f.read2;
      f.read2.(0) <- A.Any a;
      f.read2.(1) <- A.Any b
  | Some i ->
      f.read <- f.read3;
      f.read3.(0) <- A.Any a;
      f.read3.(1) <- A.Any b;
      f.read3.(2) <- A.Any i

(* Makes the frame's buffers those of the held arrays in {!Plan.access}'s
   slot order: [a], [b], [dst], then [init] and the workspace where the plan
   has them. *)
let bind f ws =
  let n = Array.length f.read in
  let uses_ws = Plan.workspace f.plan > 0 in
  let k = n + 1 + Bool.to_int uses_ws in
  f.buffers <-
    (if k = 5 then f.buffers5 else if k = 4 then f.buffers4 else f.buffers3);
  f.buffers.(0) <- buffer f.read.(0);
  f.buffers.(1) <- buffer f.read.(1);
  f.buffers.(2) <- buffer f.written.(0);
  if n = 3 then f.buffers.(3) <- buffer f.read.(2);
  if uses_ws then f.buffers.(n + 1) <- ws

(* The door's work: [f.sub], which the call stored before it entered the
   door. *)
let issue f =
  ignore (Rig.submit (Option.get f.sub) ~run:f.run ~buffers:f.buffers)

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Array.length xs && (dead xs.(i) || any_dead xs (i + 1))

(* Contractions *)

let run d dv f ~batch ~contracting ~acc ~dst ~init a b =
  if not (V.fill f.view ~batch ~contracting ~dst ~init a b) then A.Declined
  else begin
    hold f ~dst ~init a b;
    if any_dead f.written 0 || any_dead f.read 0 then
      (* The door refuses a dead buffer, before the plan reads its address. *)
      A.door ~written:f.written ~read:f.read nothing ()
    else
      match Plan.choose f.plan f.view ~acc ~dst ~init a b with
      | Declined -> A.Declined
      | Nothing -> A.door ~written:f.written ~read:f.read nothing ()
      | Launches ->
          let ws = workspace d dv (Plan.workspace f.plan) in
          let sub = submission d dv f.plan in
          f.sub <- sub;
          Plan.write f.run (Option.get sub) f.plan;
          bind f ws;
          A.door ~written:f.written ~read:f.read issue f
  end

let contract ~batch ~contracting ~acc ~dst ~init a b =
  let d = A.device dst in
  match device d with
  | None -> A.Declined
  | Some dv -> (
      let f = take () in
      match run d dv f ~batch ~contracting ~acc ~dst ~init a b with
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

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (Nx_array.Dtype.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let reduce p ~axes:_ rs ~dsts:_ _ =
  if Array.for_all (accepted p) rs then A.Declined else A.Wrong_dtype

let scan p ~axis:_ rk ~dsts:_ _ =
  if accepted p rk then A.Declined else A.Wrong_dtype

let gather ~axis:_ ~dst:_ _ _ = A.Declined

(* [Declined] where {!Nx_kernel.Prog.Add} takes [x]'s dtype, as a sum's
   operands. *)
let adds x =
  if Nx_kernel.Prog.accepts2 (Binary Add) (A.dtype x) then A.Declined
  else A.Wrong_dtype

let scatter (c : Nx_kernel.combine) ~unique:_ ~axis:_ ~dst:_ ~into:_ _ u =
  match c with Add -> adds u | Set | Max | Min -> A.Declined

let sort ~axis:_ ~descending:_ ~values:_ ~positions:_ _ = A.Declined
let assemble ~fill:_ ~dst:_ _ = A.Declined
let fold _ ~dst:_ x = adds x

(* Transforms and factorisations: declined, once the dtypes are ones they
   take. *)

let declined takes = if takes then A.Declined else A.Wrong_dtype
let routine a = declined (Nx_kernel.full_precision (A.dtype a))
let c2c _ ~axes:_ ~dst:_ _ = A.Declined

let r2c ~axes:_ ~dst x =
  declined (Nx_kernel.same_precision (A.dtype x) (A.dtype dst))

let c2r ~axes:_ ~dst x =
  declined (Nx_kernel.same_precision (A.dtype dst) (A.dtype x))

let cholesky _ ~dst:_ a = routine a
let lu ~lu:_ ~pivots:_ ~perm:_ a = routine a
let qr _ ~q:_ ~r:_ a = routine a
let svd _ ~u:_ ~s:_ ~vh:_ a = routine a
let svd_values ~s:_ a = routine a
let eigh ~w:_ ~v:_ a = routine a
let eigh_values ~w:_ a = routine a
let eig ~w:_ ~v:_ _ = A.Declined
let eig_values ~w:_ _ = A.Declined
let solve_triangular _ ~adjoint:_ ~unit_diagonal:_ ~dst:_ a _ = routine a
