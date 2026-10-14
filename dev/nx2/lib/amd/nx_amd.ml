(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call groups its view and chooses its launches, one a window of its
   results, and through the door writes each into its domain's run and submits
   the submission of the launch's sequence, made once per device:

   contract --> device (Rig.Local) --> Plan.choose --> workspace --> frame
   (the domain's, or a fresh one) --> door --> for each launch: submission,
   Plan.write into the run, Rig.submit

   A call that finds every cache warm allocates its view, its launches, their
   buffers and the door's work, and nothing that outlives it. *)

module A = Nx_array
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

(* The door's work: the launches [ls] on [d], each written into [run] and
   submitted in order, the workspace their only shared buffer, which each names
   [Read_write]. *)
let issue (d, dv, run, workspace, ls) =
  for i = 0 to Array.length ls - 1 do
    let l = ls.(i) in
    let sub = submission d dv l in
    Plan.write l run sub;
    ignore (Rig.submit sub ~run ~buffers:(Plan.buffers l ~workspace))
  done

(* Submits [ls] on [d] through the door, in a frame's run. *)
let submit ~written ~read d dv ls =
  let need = ref 0 in
  for i = 0 to Array.length ls - 1 do
    need := Int.max !need (Plan.workspace ls.(i))
  done;
  let workspace = workspace d dv !need in
  let f = take () in
  match A.door ~written ~read issue (d, dv, f.run, workspace, ls) with
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

(* Lane order, on the lane kernels, at 1 row or at 4 columns or fewer, which
   a lane workgroup takes as its short extent; chain order, on WMMA or SIMT,
   at every other shape. On the R9700, by 20480 x 2880, float32 takes 1.54
   ms in lane order at 1 row, but 2.7-4.9 ms at 2 to 4 rows against 1.6 ms
   in chain (bfloat16 at 4 rows: 3.2 and 1.6 ms); 20480 rows by 1 column
   0.40 ms in lane and 1.16 ms in chain (float64: 0.77 and 18 ms). Float64
   at 2 rows is the cost: 3.0 ms in lane, 3.9 ms in chain. *)
let lane_most = 4

let contract_order ~rows ~columns : Nx_kernel.order =
  if rows = 1 || columns <= lane_most then Lane else Chain

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
        | Ok [||] -> A.door ~written ~read nothing ()
        | Ok ls -> submit ~written ~read d dv ls)

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

let reduce p ~shape:_ ~axes:_ rs ~dsts:_ _ =
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
let qr ~q:_ ~r:_ a = routine a
let svd ~s:_ ?vectors:_ a = routine a
let eigh ~w:_ ?v:_ a = routine a
let eig ~w ?v:_ a =
  declined
    (Nx_kernel.full_precision (A.dtype a)
    && Nx_kernel.same_precision (A.dtype a) (A.dtype w))
let solve_triangular _ ~adjoint:_ ~unit_diagonal:_ ~dst:_ a _ = routine a
