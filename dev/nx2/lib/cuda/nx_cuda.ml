(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call groups its view and chooses its launch, writes the launch into its
   domain's run and submits the submission of the launch's sequence, made
   once per device, through the door:

   contract --> device (Rig.Local) --> Plan.choose --> workspace, submission
   --> frame (the domain's, or a fresh one) --> Plan.write into its run -->
   door --> Rig.submit

   reduce and scan take the same path with Reduce's launch.
   A call that finds every cache warm allocates its view, its launch, its
   buffers and the door's work, and nothing that outlives it. *)

module A = Nx_array
module Sub = Rig.Submission

let name = "nx.cuda"

external cubin : string -> string option = "nx_cuda_cubin"

(* Devices *)

(* A device keeps two buffers, each named [Read_write] by every call that uses
   it, so rig orders those calls: the workspace, which holds a call's packed
   operands and split sums, grows to [kept] bytes and is kept, a call that
   needs more taking a buffer of its own; the tickets of split sums are zero
   words, which every call leaves zero: combine.cuh's last block stores its
   tile's ticket back to zero, so no launch zeroes them. *)
let kept = 64 * 1024 * 1024

(* What this library keeps for a device: [subs] and [reduce_subs] hold the
   submissions by {!Plan.sequence} and {!Reduce.sequence}, made on first
   use. *)
type device = {
  image : Rig.Image.t;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : Sub.t option array;
  reduce_subs : Sub.t option array;
}

(* A device of an architecture [cubin] serves, whose queues run launches. *)
let load d =
  match cubin (Rig.arch d) with
  | None -> None
  | Some bin -> (
      match Rig.Image.load d bin with
      | Error _ -> None
      | Ok image ->
          Some
            {
              image;
              workspace = Atomic.make (Rig.Buffer.create d 0);
              tickets = Atomic.make (Rig.Buffer.create d 0);
              subs = Array.make Plan.sequences None;
              reduce_subs = Array.make Reduce.sequences None;
            })

(* Each device's [load], [None] where it does not compute. *)
let devices = Rig.Local.new_key load
let device d = Rig.Local.get devices d

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
  Rig.Buffer.fill b '\000';
  b

let tickets d dv need =
  let w = Atomic.get dv.tickets in
  if Rig.Buffer.length w >= need then w
  else grow dv.tickets need ~cap:max_int (zeros d)

(* The submission of [l]'s sequence on [d]. Two domains that make one at once
   each make it, and the last store wins. *)
let submission d dv l =
  let key = Plan.sequence l in
  match dv.subs.(key) with
  | Some s -> s
  | None ->
      let s = Sub.make d (Plan.parts key dv.image) in
      dv.subs.(key) <- Some s;
      s

let reduce_submission d dv l =
  let key = Reduce.sequence l in
  match dv.reduce_subs.(key) with
  | Some s -> s
  | None ->
      let s = Sub.make d (Reduce.parts key dv.image) in
      dv.reduce_subs.(key) <- Some s;
      s

(* Frames *)

(* The run of a call, which needs it alone from its {!Plan.write} to its
   submit's return: one per domain, guarded by [busy]. A call that finds it
   busy, such as one of another systhread of the domain or a signal
   handler's, uses a fresh frame it does not keep. *)
type frame = { busy : bool Atomic.t; run : Sub.Run.t }

let frame ~busy = { busy = Atomic.make busy; run = Sub.Run.make () }
let frames = Domain.DLS.new_key (fun () -> frame ~busy:false)

let take () =
  let f = Domain.DLS.get frames in
  if Atomic.compare_and_set f.busy false true then f else frame ~busy:true

let release f = Atomic.set f.busy false

(* The arrays a call reads: [a], [b], then [init] where given. *)
let reads ~init a b =
  match init with
  | None -> [| A.Any a; A.Any b |]
  | Some i -> [| A.Any a; A.Any b; A.Any i |]

(* The door's work: a launch's submission, with the run its parameters are
   written into and its buffers. *)
let issue (sub, run, buffers) = ignore (Rig.submit sub ~run ~buffers)

(* [write]s a launch of [sub] into a frame's run and submits [sub] with
   [buffers] through the door. *)
let submit ~written ~read sub buffers write =
  let f = take () in
  match
    write f.run sub;
    A.door ~written ~read issue (sub, f.run, buffers)
  with
  | answer ->
      release f;
      answer
  | exception e ->
      release f;
      raise e

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Array.length xs && (dead xs.(i) || any_dead xs (i + 1))

(* Contractions *)

(* Lane order, on the skinny kernels, where rows or columns are 4 or fewer;
   chain order, on the SIMT kernels, past them. On the RTX 5000 Ada, float32
   at 2 to 4 rows by 20480 x 2880 takes 1.8-1.9 ms in lane order and 3.2 ms
   in chain; 20480 rows by 1 or 4 columns 0.60 ms in lane and 0.83 ms in
   chain (float64 by 1: 4.1 and 8.7 ms); 8 rows by 5120 x 2880 0.72 ms in
   chain and 0.77 ms in lane. *)
let lane_most = 4

let contract_order ~rows ~columns : Nx_kernel.order =
  if rows <= lane_most || columns <= lane_most then Lane else Chain

let contract ~order ~batch ~contracting ~acc ~dst ~init a b =
  let d = A.device dst in
  match device d with
  | None -> A.Declined
  | Some dv -> (
      let written = [| A.Any dst |] and read = reads ~init a b in
      if any_dead written 0 || any_dead read 0 then
        (* The door refuses a dead buffer, before the plan reads its
           address. *)
        A.door ~written ~read nothing ()
      else
        match Plan.choose ~order ~batch ~contracting ~acc ~dst ~init a b with
        | Error answer -> answer
        | Ok Nothing -> A.door ~written ~read nothing ()
        | Ok (Launch l) ->
            let workspace = workspace d dv (Plan.workspace l) in
            let tickets = tickets d dv (Plan.tickets l) in
            let sub = submission d dv l in
            let buffers = Plan.buffers l ~workspace ~tickets in
            submit ~written ~read sub buffers (Plan.write l))

(* Reductions and scans *)

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (Nx_array.Dtype.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

(* What a reduction or scan of [x] into [dst] refuses, where [shape] gives
   the destination's shape for [x]'s: an [x] of another dtype than [p]'s
   operand, and a destination of another shape. *)
let refusal (p : Nx_kernel.Prog.t) x ~dst shape =
  let (A.Any y) = dst in
  let shape_of a = A.Layout.shape (A.layout a) in
  let wrong_dtype =
    Iarray.length p.ins = 1
    &&
    let (A.Dtype.Any operand) = Iarray.get p.ins 0 in
    not (A.Dtype.equal (A.dtype x) operand)
  in
  if wrong_dtype then Some A.Wrong_dtype
  else
    match shape (shape_of x) with
    | Ok s when s = shape_of y -> None
    | Ok _ | Error _ -> Some A.Shape_mismatch

(* Plans the call of [x] into [dst] with [plan] on [dst]'s device, once
   [refusal] finds nothing to refuse. *)
let reduce_on p x ~dst shape plan =
  let (A.Any y) = dst in
  let d = A.device y in
  match device d with
  | None -> A.Declined
  | Some dv -> (
      match refusal p x ~dst shape with
      | Some refused -> refused
      | None -> (
          let written = [| dst |] and read = [| A.Any x |] in
          match plan () with
          | Error answer -> answer
          | Ok Reduce.Nothing -> A.door ~written ~read nothing ()
          | Ok (Launch l) ->
              let workspace = workspace d dv (Reduce.workspace l) in
              let sub = reduce_submission d dv l in
              let buffers = Reduce.buffers l ~workspace in
              submit ~written ~read sub buffers (Reduce.write l)))

(* Reduce computes a monoid of output 0 of the program into one destination,
   read plain. *)
let reduce p ~axes rs ~dsts loads =
  if not (Array.for_all (accepted p) rs) then A.Wrong_dtype
  else
    match (rs, dsts, loads) with
    | [| (Nx_kernel.Monoid m, 0) |], [| dst |], [| Nx_kernel.Plain x |] ->
        reduce_on p x ~dst
          (Nx_kernel.Shape.reduce ~axes [| Nx_kernel.Monoid m |])
          (fun () -> Reduce.reduce ~axes m p ~dst x)
    | _ -> A.Declined

let scan p ~axis r ~dsts loads =
  if not (accepted p r) then A.Wrong_dtype
  else
    match (r, dsts, loads) with
    | (Nx_kernel.Monoid m, 0), [| dst |], [| Nx_kernel.Plain x |] ->
        reduce_on p x ~dst (Nx_kernel.Shape.scan ~axis) (fun () ->
            Reduce.scan ~axis m p ~dst x)
    | _ -> A.Declined

(* Elementwise *)

let apply0 _ ~dst:_ = A.Declined
let apply1 _ ~dst:_ _ = A.Declined
let apply2 _ ~dst:_ _ _ = A.Declined
let apply3 _ ~dst:_ _ _ _ = A.Declined
let map _ ~dsts:_ _ = A.Declined

(* Gathers, scatters, sorts, assemblies and folds *)

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
let qr ~q:_ ~r:_ a = routine a
let svd ~s:_ ?vectors:_ a = routine a
let eigh ~w:_ ?v:_ a = routine a
let eig ~w ?v:_ a =
  declined
    (Nx_kernel.full_precision (A.dtype a)
    && Nx_kernel.same_precision (A.dtype a) (A.dtype w))
let solve_triangular _ ~adjoint:_ ~unit_diagonal:_ ~dst:_ a _ = routine a
