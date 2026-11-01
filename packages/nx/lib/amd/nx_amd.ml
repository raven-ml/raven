(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call groups its view and chooses its launches, one a window of its
   results, and through the door writes each into a run of the submission of
   the launch's sequence, made once per device, and submits it:

   contract --> device (Rig.Local) --> Plan.choose --> workspace --> door -->
   for each launch: submission, a run of it, Plan.write into the run,
   Rig.submit

   A call that finds every cache warm allocates its view, its launches, their
   buffers and the door's work, and nothing that outlives it. *)

module A = Nx_array
module D = Nx_array.Dtype
module Sub = Rig.Submission

let name = "nx.amd"

external code_object : string -> string option = "nx_amd_code_object"

(* Devices *)

(* The workspace holds a call's packed operands and split sums. Each call that
   uses it names it [Read_write], so rig orders the calls that share it. It
   grows to [kept] bytes and is kept; a call that needs more takes a buffer of
   its own. *)
let kept = 64 * 1024 * 1024

(* A submission with the runs made for it, which a call takes one at a
   time ({!run}). *)
type prepared = { sub : Sub.t; runs : Sub.Run.t list Atomic.t }

(* What this library keeps for a device: [subs] holds the submissions by
   {!Plan.sequence}, made on first use. *)
type device = {
  image : Rig.Image.t;
  workspace : Rig.Buffer.t Atomic.t;
  subs : prepared option array;
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
  | Some p -> p
  | None ->
      let sub = Sub.make d (Plan.parts key dv.image) in
      let p = { sub; runs = Atomic.make [] } in
      dv.subs.(key) <- Some p;
      p

(* Runs *)

(* A run of [p] the calling thread takes: the first of [p]'s that is free,
   else a new one, which joins them. A submit frees its run; a call that
   raises before its submit releases it. *)
let rec take_from p = function
  | r :: rs -> if Sub.Run.take r then r else take_from p rs
  | [] ->
      let r = Sub.Run.make p.sub in
      let rec add () =
        let rs = Atomic.get p.runs in
        if not (Atomic.compare_and_set p.runs rs (r :: rs)) then add ()
      in
      add ();
      r

let run p = take_from p (Atomic.get p.runs)

(* The arrays a call reads: [a], [b], then [init] where given. *)
let reads ~init a b =
  match init with
  | None -> [| A.Any a; A.Any b |]
  | Some i -> [| A.Any a; A.Any b; A.Any i |]

(* The door's work: the launches [ls] on [d], each written into a run of its
   submission and submitted in order, the workspace their only shared buffer,
   which each names [Read_write]. *)
let issue (d, dv, workspace, ls) =
  for i = 0 to Array.length ls - 1 do
    let l = ls.(i) in
    let p = submission d dv l in
    let r = run p in
    match Plan.write l r p.sub with
    | () -> ignore (Rig.submit r ~buffers:(Plan.buffers l ~workspace))
    | exception e ->
        Sub.Run.release r;
        raise e
  done

(* Submits [ls] on [d] through the door. *)
let submit ~written ~read d dv ls =
  let need = ref 0 in
  for i = 0 to Array.length ls - 1 do
    need := Int.max !need (Plan.workspace ls.(i))
  done;
  let workspace = workspace d dv !need in
  A.door ~written ~read issue (d, dv, workspace, ls)

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Array.length xs && (dead xs.(i) || any_dead xs (i + 1))

(* Contractions *)

let shape_of a = Iarray.to_array (A.Layout.shape (A.layout a))

(* [o]'s shape and its scale's, as {!Nx_kernel.Shape.contract} takes them. *)
let shapes (o : Nx_kernel.operand) =
  let (A.Any x) = o.array in
  let scale =
    match o.scale with
    | None -> None
    | Some (Exponent e) -> Some (shape_of e)
    | Some (Value v) -> Some (shape_of v)
  in
  (shape_of x, scale)

(* A scaled or segmented call: declined once its shapes fit. *)
let unplanned ~batch ~contracting ~segments ~dst ~init a b =
  let segments =
    match (segments : _ Nx_kernel.segments option) with
    | None -> None
    | Some (Rows r) ->
        Some (Nx_kernel.Rows { r with offsets = shape_of r.offsets })
    | Some (Contracting c) ->
        Some (Nx_kernel.Contracting { c with offsets = shape_of c.offsets })
  in
  let fits y =
    y = shape_of dst
    && match init with None -> true | Some i -> y = shape_of i
  in
  match
    Nx_kernel.Shape.contract ~batch ~contracting ~segments (shapes a)
      (shapes b)
  with
  | Ok y when fits y -> A.Declined
  | Ok _ | Error _ -> A.Shape_mismatch

let contract ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let (D.Any acc_dt) = acc in
  let d = A.device dst in
  if not (Nx_kernel.accumulates acc_dt) then A.Wrong_dtype
  else
    match device d with
    | None -> A.Declined
    | Some _
      when Option.is_some segments || Option.is_some oa.scale
           || Option.is_some ob.scale ->
        unplanned ~batch ~contracting ~segments ~dst ~init oa ob
    | Some dv -> (
        let (A.Any a) = oa.array in
        let (A.Any b) = ob.array in
        let written = [| A.Any dst |] and read = reads ~init a b in
        if any_dead written 0 || any_dead read 0 then
          (* The door refuses a dead buffer, before the plan reads its
             address. *)
          A.door ~written ~read nothing ()
        else
          match Plan.choose ~batch ~contracting ~acc ~dst ~init a b with
          | Error answer -> answer
          | Ok [||] -> A.door ~written ~read nothing ()
          | Ok ls -> submit ~written ~read d dv ls)

(* Elementwise *)

(* A map refuses destinations or loads other than its program's outputs and
   operands before it declines. *)
let map (p : Nx_kernel.Prog.t) ~dsts loads =
  if
    Array.length dsts <> Iarray.length p.outs
    || Array.length loads <> Iarray.length p.ins
  then A.Shape_mismatch
  else A.Declined

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

let reduce p ~shape:_ ~axes:_ rs ~dsts:_ _ =
  if Array.for_all (accepted p) rs then A.Declined else A.Wrong_dtype

let scan ?carry:_ p ~axis:_ rk ~dsts:_ _ =
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

(* A routine of float results of [a]'s precision. *)
let real_routine a r =
  declined
    (Nx_kernel.full_precision (A.dtype a)
    && Nx_kernel.same_precision (A.dtype a) (A.dtype r))

let svd ~s ?vectors:_ a = real_routine a s
let eigh ~w ?v:_ a = real_routine a w
let eig ~w ?v:_ a =
  declined
    (Nx_kernel.full_precision (A.dtype a)
    && Nx_kernel.same_precision (A.dtype a) (A.dtype w))
let solve_triangular _ ~adjoint:_ ~unit_diagonal:_ ~dst:_ a _ = routine a
