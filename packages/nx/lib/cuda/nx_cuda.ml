(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call groups its view and chooses its launches, then, through the door,
   writes each into a run of the submission of the launch's sequence, made
   once per device, and submits it:

   contract --> device (Rig.Local) --> Plan.choose --> workspace --> door
   --> submission, a run of it, Plan.write into the run --> Rig.submit

   reduce and scan write Reduce's one launch before the door.
   A call that finds every cache warm allocates its view, its launches, their
   buffers and the door's work, and nothing that outlives it. *)

module A = Nx_array
module D = Nx_array.Dtype
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

(* A submission with the runs made for it, which a call takes one at a
   time ({!run}). *)
type prepared = { sub : Sub.t; runs : Sub.Run.t list Atomic.t }

(* What this library keeps for a device: [subs] and [reduce_subs] hold the
   submissions by {!Plan.sequence} and {!Reduce.sequence}, made on first
   use. *)
type device = {
  image : Rig.Image.t;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : prepared option array;
  reduce_subs : prepared option array;
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

let prepare d parts = { sub = Sub.make d parts; runs = Atomic.make [] }

(* The submission of [l]'s sequence on [d]. Two domains that make one at once
   each make it, and the last store wins. *)
let submission d dv l =
  let key = Plan.sequence l in
  match dv.subs.(key) with
  | Some p -> p
  | None ->
      let p = prepare d (Plan.parts key dv.image) in
      dv.subs.(key) <- Some p;
      p

let reduce_submission d dv l =
  let key = Reduce.sequence l in
  match dv.reduce_subs.(key) with
  | Some p -> p
  | None ->
      let p = prepare d (Reduce.parts key dv.image) in
      dv.reduce_subs.(key) <- Some p;
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

(* The door's work: a launch's run, its parameters written, and its
   buffers. *)
let issue (run, buffers) = ignore (Rig.submit run ~buffers)

(* [write]s a launch of [p] into a run of it and submits the run with
   [buffers] through the door. *)
let submit ~written ~read p buffers write =
  let r = run p in
  match
    write r p.sub;
    A.door ~written ~read issue (r, buffers)
  with
  | answer ->
      Sub.Run.release r;
      answer
  | exception e ->
      Sub.Run.release r;
      raise e

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Array.length xs && (dead xs.(i) || any_dead xs (i + 1))

(* Contractions *)

(* The door's work of a contraction's launches on [d]: each written into a
   run of its submission and submitted in order, which rig keeps by the
   buffers they share: y, the workspace and the tickets. *)
let issue_launches (d, dv, ls, workspace, tickets) =
  for i = 0 to Array.length ls - 1 do
    let l = ls.(i) in
    let p = submission d dv l in
    let r = run p in
    match Plan.write l r p.sub with
    | () -> ignore (Rig.submit r ~buffers:(Plan.buffers l ~workspace ~tickets))
    | exception e ->
        Sub.Run.release r;
        raise e
  done

(* Submits [ls] through the door. *)
let submit_launches ~written ~read d dv ls ~workspace ~tickets =
  A.door ~written ~read issue_launches (d, dv, ls, workspace, tickets)

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
          | Ok Nothing -> A.door ~written ~read nothing ()
          | Ok (Launches ls) ->
              (* The launches run one after another: the most workspace and
                 tickets any of them takes. *)
              let bytes = ref 0 and ticket_bytes = ref 0 in
              for i = 0 to Array.length ls - 1 do
                bytes := Int.max !bytes (Plan.workspace ls.(i));
                ticket_bytes := Int.max !ticket_bytes (Plan.tickets ls.(i))
              done;
              let workspace = workspace d dv !bytes in
              let tickets = tickets d dv !ticket_bytes in
              submit_launches ~written ~read d dv ls ~workspace ~tickets)

(* Reductions and scans *)

(* Whether reduction [r] accepts the dtype of [p]'s output [k]. *)
let accepted (p : Nx_kernel.Prog.t) (r, k) =
  let (D.Any dt) = Nx_kernel.Prog.dtype p (Iarray.get p.outs k) in
  Nx_kernel.accepts r dt

(* What a reduction or scan of [x] into [dst] refuses, where [shape] gives
   the destination's shape for [x]'s: an [x] of another dtype than [p]'s
   operand, and a destination of another shape. *)
let refusal (p : Nx_kernel.Prog.t) x ~dst shape =
  let (A.Any y) = dst in
  let wrong_dtype =
    Iarray.length p.ins = 1
    &&
    let (D.Any operand) = Iarray.get p.ins 0 in
    not (D.equal (A.dtype x) operand)
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

(* Whether load [l] has the loop's [shape] once loaded. *)
let fits shape (l : Nx_kernel.load) =
  match l with
  | Plain x -> shape_of x = shape
  | Padded { x; pad; _ } -> Nx_kernel.Shape.padded pad (shape_of x) = Ok shape

(* Reduce computes a monoid of output 0 of the program into one destination,
   read plain. *)
let reduce p ~shape ~axes rs ~dsts loads =
  if not (Array.for_all (accepted p) rs) then A.Wrong_dtype
  else if not (Array.for_all (fits shape) loads) then A.Shape_mismatch
  else
    match (rs, dsts, loads) with
    | [| (Nx_kernel.Monoid m, 0) |], [| dst |], [| Nx_kernel.Plain x |] ->
        reduce_on p x ~dst
          (Nx_kernel.Shape.reduce ~axes [| Nx_kernel.Monoid m |])
          (fun () -> Reduce.reduce ~axes:(Iarray.to_array axes) m p ~dst x)
    | _ -> A.Declined

(* What a scan refuses of its [carry]: one given with a reduction other than
   a monoid or of another shape than [dsts]'s with extent 1 along [axis],
   and one of another dtype than the output's carrier. *)
let carry_refusal p ~axis (r, k) ~dsts (A.Any c) =
  let monoid = match r with Nx_kernel.Monoid _ -> true | _ -> false in
  let fits =
    match dsts with
    | [| A.Any y |] ->
        let s = shape_of y in
        axis >= 0 && axis < Array.length s
        && (s.(axis) <- 1;
            s = shape_of c)
    | _ -> false
  in
  if not (monoid && fits) then A.Shape_mismatch
  else
    let (D.Any carrier) =
      Nx_kernel.carrier (Nx_kernel.Prog.dtype p (Iarray.get p.outs k))
    in
    if D.equal (A.dtype c) carrier then A.Declined else A.Wrong_dtype

let scan ?carry p ~axis r ~dsts loads =
  if not (accepted p r) then A.Wrong_dtype
  else
    match (carry, r, dsts, loads) with
    | Some c, _, _, _ -> carry_refusal p ~axis r ~dsts c
    | None, (Nx_kernel.Monoid m, 0), [| dst |], [| Nx_kernel.Plain x |] ->
        reduce_on p x ~dst (Nx_kernel.Shape.scan ~axis) (fun () ->
            Reduce.scan ~axis m p ~dst x)
    | _ -> A.Declined

(* Elementwise *)

(* A map refuses destinations or loads other than its program's outputs and
   operands before it declines. *)
let map (p : Nx_kernel.Prog.t) ~dsts loads =
  if
    Array.length dsts <> Iarray.length p.outs
    || Array.length loads <> Iarray.length p.ins
  then A.Shape_mismatch
  else A.Declined

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
