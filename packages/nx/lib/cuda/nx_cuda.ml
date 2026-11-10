(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call refuses what nx.kernel's rules refuse, then its family plans
   it as calls ({!Launch}), each one submission of its launches, and the
   door submits them in order:

   map --> device (Rig.Local) --> Loop.map --> calls --> door
   --> per call: the submission of its key, its stand-ins bound to the
       device's workspace and tickets, a run of it written --> Rig.submit

   A call that finds every cache warm allocates its plan, its launches,
   their buffers and the door's work, and nothing that outlives it. *)

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
   words, which every call leaves zero: combine.cuh's and loop.cu's last
   block stores its ticket back to zero, so no launch zeroes them. They too
   grow to [kept] bytes, a call that needs more taking zeros of its own. *)
let kept = 64 * 1024 * 1024

(* What this library keeps for a device [rig]: [subs] holds the
   submissions of the calls made, by their key ({!Launch.key}), a cache for
   each first kernel; [fft] the transforms' plans of lines. *)
type device = {
  rig : Rig.t;
  image : Rig.Image.t;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : (int array, Sub.t) Cache.t array;
  fft : Fft.cache;
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
              rig = d;
              image;
              workspace = Atomic.make (Rig.Buffer.create d 0);
              tickets = Atomic.make (Rig.Buffer.create d 0);
              subs =
                Array.init (Array.length Kernels.kernels) (fun _ ->
                    Cache.make ());
              fft = Fft.cache d;
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

let workspace dv need =
  let w = Atomic.get dv.workspace in
  if Rig.Buffer.length w >= need then w
  else if need > kept then Rig.Buffer.create dv.rig need
  else grow dv.workspace need ~cap:kept (Rig.Buffer.create dv.rig)

let zeros d bytes =
  let b = Rig.Buffer.create d bytes in
  Rig.Buffer.fill b '\000';
  b

let tickets dv need =
  let w = Atomic.get dv.tickets in
  if Rig.Buffer.length w >= need then w
  else if need > kept then zeros dv.rig need
  else grow dv.tickets need ~cap:kept (zeros dv.rig)

(* Submissions

   A device keeps the submission of each call's key ({!Launch.key}), made at
   its first use, in a cache for each first kernel ({!Cache}) that holds
   the [kept_subs] most recently made, so that none grows with the calls a
   program makes and a lookup reads few keys. *)
let kept_subs = 64

(* The submission of [c] on [d]: the one of its key, made at its first
   use, or one of its own where it is not kept. *)
let submission dv (c : _ Launch.call) =
  if not c.kept then Sub.make dv.rig (Launch.parts dv.image c)
  else
    let cell = dv.subs.(c.launches.(0).kernel) in
    match Cache.find_by Launch.matches cell c with
    | Some s -> s
    | None ->
        Cache.keep ~count:kept_subs cell (Launch.key c)
          (Sub.make dv.rig (Launch.parts dv.image c))

(* Binds [c]'s stand-ins to the device's workspace and tickets of [c]'s
   bytes, each taken once: past [kept] bytes, a call's own, which it
   answers. *)
let bind dv (c : _ Launch.call) =
  let bs = c.buffers in
  let w = ref Launch.workspace and t = ref Launch.tickets in
  for i = 0 to Array.length bs - 1 do
    if bs.(i) == Launch.workspace then begin
      if !w == Launch.workspace then w := workspace dv c.workspace;
      bs.(i) <- !w
    end
    else if bs.(i) == Launch.tickets then begin
      if !t == Launch.tickets then t := tickets dv c.tickets;
      bs.(i) <- !t
    end
  done;
  let own b need stand = if b != stand && need > kept then [ b ] else [] in
  own !w c.workspace Launch.workspace @ own !t c.tickets Launch.tickets

(* Writes [c], its parameters by [w], into a run of its submission and
   submits it, then frees the workspace and tickets of its own, which
   return once that work is done. A write that raises releases the run. *)
let issue_call dv w (c : _ Launch.call) =
  let s = submission dv c in
  let own = bind dv c in
  let r = Sub.take s in
  match Launch.write w c r s with
  | () ->
      ignore (Rig.submit r ~buffers:c.buffers);
      List.iter (fun b -> ignore (Rig.Buffer.free b)) own
  | exception e ->
      Sub.Run.release r;
      raise e

(* A call of no launch, a routine's of no element, has no work. *)
let rec issue_list dv w = function
  | [] -> ()
  | (c : _ Launch.call) :: cs ->
      if Array.length c.launches > 0 then issue_call dv w c;
      issue_list dv w cs

(* The door's work of a plan's calls, in order: each one submission, which
   rig orders by the buffers they share. The buffers the plan made for its
   work then return once that work is done. *)
let issue_plan (dv, w, cs) =
  issue_list dv w cs;
  List.iter
    (fun (c : _ Launch.call) ->
      List.iter (fun b -> ignore (Rig.Buffer.free b)) c.frees)
    cs

(* The arrays a contraction reads: [a], [b], then [init], the segments'
   ids and the scales where given. *)
let reads ~init ~segments (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let scale (o : Nx_kernel.operand) =
    match o.scale with
    | None -> []
    | Some (Exponent e) -> [ A.Any e ]
    | Some (Value v) -> [ A.Any v ]
  in
  let ids =
    match (segments : _ Nx_kernel.segments option) with
    | None -> []
    | Some (Rows r) -> [ A.Any r.ids ]
    | Some (Contracting g) -> [ A.Any g.ids ]
  in
  Iarray.of_list
    (oa.array :: ob.array
     :: Option.to_list (Option.map (fun i -> A.Any i) init)
    @ ids @ scale oa @ scale ob)

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Iarray.length xs && (dead (Iarray.get xs i) || any_dead xs (i + 1))

(* A call that refuses nothing: on [d], [plan] of its device, its calls
   written by [w] through the door, claiming [written] and [read]. The
   door refuses a dead buffer before the plan reads its address. *)
let run d ~written ~read w plan =
  match device d with
  | None -> A.Declined
  | Some dv -> (
      if any_dead written 0 || any_dead read 0 then
        A.door ~written ~read nothing ()
      else
        match plan dv with
        | Error answer -> answer
        | Ok [] -> A.door ~written ~read nothing ()
        | Ok cs -> A.door ~written ~read issue_plan (dv, w, cs))

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

(* Whether the call's shapes fit: {!Nx_kernel.Shape.contract}'s, [dst]'s
   and [init]'s. *)
let contract_fits ~batch ~contracting ~segments ~dst ~init a b =
  let segments =
    match (segments : _ Nx_kernel.segments option) with
    | None -> None
    | Some (Rows r) -> Some (Nx_kernel.Rows { r with ids = shape_of r.ids })
    | Some (Contracting c) ->
        Some (Nx_kernel.Contracting { c with ids = shape_of c.ids })
  in
  match
    Nx_kernel.Shape.contract ~batch ~contracting ~segments (shapes a) (shapes b)
  with
  | Ok y -> (
      y = shape_of dst
      && match init with None -> true | Some i -> y = shape_of i)
  | Error _ -> false

let contract ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let (D.Any acc_dt) = acc in
  if not (Nx_kernel.accumulates acc_dt) then A.Wrong_dtype
  else if not (contract_fits ~batch ~contracting ~segments ~dst ~init oa ob)
  then A.Shape_mismatch
  else
    run (A.device dst) ~written:[| A.Any dst |]
      ~read:(reads ~init ~segments oa ob) Launch.bytes (fun _ ->
        Contract.plan ~batch ~contracting ~segments ~acc ~dst:(A.Any dst)
          ~init:(Option.map (fun i -> A.Any i) init)
          oa ob)

let contract_wide_k = Contract.set_wide_k

(* Loops *)

(* The arrays a loop reads: its loads' arrays, then a carry. *)
let load_arrays ?carry (loads : Nx_kernel.load iarray) =
  let n = Iarray.length loads in
  let array i =
    match Iarray.get loads i with
    | Nx_kernel.Plain x -> A.Any x
    | Padded { x; _ } -> A.Any x
  in
  match carry with
  | None -> Iarray.init n array
  | Some c -> Iarray.init (n + 1) (fun i -> if i < n then array i else c)

(* Refusals: Nx_kernel's rules, which read dtypes and counts, then the
   shapes they leave to a library: each load's once loaded, the
   destinations' and a carry's, as {!Nx_kernel.Shape} gives them. *)

let any_dtype (A.Any a) = D.Any (A.dtype a)

let arrays =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype = (fun a i -> any_dtype (Iarray.get a i));
  }

let reductions =
  {
    Nx_kernel.reductions = Iarray.length;
    kind = (fun rs i -> fst (Iarray.get rs i));
    output = (fun rs i -> snd (Iarray.get rs i));
  }

let loads_of =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype =
      (fun (a : Nx_kernel.load iarray) i ->
        match Iarray.get a i with
        | Plain x -> D.Any (A.dtype x)
        | Padded { x; _ } -> D.Any (A.dtype x));
  }

(* The rank of a loop over the shape of [dsts]' first, [0] for none. *)
let rank_of dsts =
  if Iarray.length dsts = 0 then 0
  else
    let (A.Any d) = Iarray.get dsts 0 in
    A.Layout.rank (A.layout d)

(* Whether each load has the loop's [shape] once loaded and each of [dsts]
   the shape [want]. *)
let fits ~shape loads ~want dsts =
  let loaded : Nx_kernel.load -> bool = function
    | Plain x -> shape_of x = shape
    | Padded { x; pad; _ } -> Nx_kernel.Shape.padded pad (shape_of x) = Ok shape
  in
  Iarray.for_all loaded loads
  && Iarray.for_all (fun (A.Any y) -> shape_of y = want) dsts

(* A loop or a move that refuses nothing: [plan] on [y]'s device through
   the door. *)
let planned y ~written ~read plan =
  run (A.device y) ~written ~read Launch.bytes plan

let map (p : Nx_kernel.Prog.t) ~dsts loads =
  match
    Nx_kernel.map_refusal p ~rank:(rank_of dsts) ~dsts:arrays dsts
      ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None when Iarray.length dsts = 0 ->
      A.door ~written:dsts ~read:(load_arrays loads) nothing ()
  | None ->
      let (A.Any y) = Iarray.get dsts 0 in
      let shape = shape_of y in
      if not (fits ~shape loads ~want:shape dsts) then A.Shape_mismatch
      else
        planned y ~written:dsts ~read:(load_arrays loads) (fun _ ->
            Loop.map p ~dsts loads)

(* The program of a reduction's value of no term into [dsts], [rs]'s
   destinations in order: [+0] for a sum, [1] for a product, [-∞] for a
   Logsumexp and NaN for Moments, each in its output's carrier, cast. *)
let identities (p : Nx_kernel.Prog.t) rs (dsts : A.any iarray) =
  let nodes = ref [] and outs = ref [] and at = ref 0 in
  let push n =
    nodes := n :: !nodes;
    List.length !nodes - 1
  in
  Iarray.iter
    (fun ((r : Nx_kernel.reduction), k) ->
      let (D.Any c) =
        Nx_kernel.carrier (Nx_kernel.Prog.dtype p (Iarray.get p.outs k))
      in
      let value =
        match r with
        | Monoid Prod -> D.one c
        | Logsumexp -> D.of_float c Float.neg_infinity
        | Moments -> D.of_float c Float.nan
        | Monoid _ | Arg _ -> D.zero c
      in
      let const =
        push (Nx_kernel.Prog.Const (D.Any c, Nx_kernel.Prog.bits c value))
      in
      for _ = 1 to Loop.destinations r do
        let (A.Any y) = Iarray.get dsts !at in
        let n = push (Nx_kernel.Prog.Op1 (Cast, D.Any (A.dtype y), const)) in
        outs := n :: !outs;
        incr at
      done)
    rs;
  Nx_kernel.Prog.v ~ins:[||]
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list (List.rev !outs))

let reduce p ~shape ~axes rs ~dsts loads =
  match
    Nx_kernel.reduce_refusal p ~rank:(Iarray.length shape) ~reductions rs
      ~dsts:arrays dsts ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None -> (
      let s = Iarray.to_array shape in
      let kinds =
        Array.init (Iarray.length rs) (fun i -> fst (Iarray.get rs i))
      in
      match Nx_kernel.Shape.reduce ~axes kinds s with
      | Ok want when fits ~shape:s loads ~want dsts ->
          let (A.Any y) = Iarray.get dsts 0 in
          let read = load_arrays loads in
          if Array.mem 0 s && not (Array.mem 0 want) then
            map (identities p rs dsts) ~dsts [||]
          else
            planned y ~written:dsts ~read (fun _ ->
                Loop.reduce p ~shape ~axes rs ~dsts loads)
      | Ok _ | Error _ -> A.Shape_mismatch)

let scan ?carry p ~axis ((r, k) as rk) ~dsts loads =
  match
    Nx_kernel.scan_refusal p ~rank:(rank_of dsts) r ~output:k ~dsts:arrays dsts
      ?carry:(Option.map any_dtype carry)
      ~loads:loads_of loads
  with
  | Some (refused, _) -> refused
  | None -> (
      let (A.Any y) = Iarray.get dsts 0 in
      let shape = shape_of y in
      let carried =
        match carry with
        | None -> true
        | Some (A.Any c) -> Nx_kernel.Shape.carry ~axis shape = Ok (shape_of c)
      in
      match Nx_kernel.Shape.scan ~axis shape with
      | Ok want when carried && fits ~shape loads ~want dsts ->
          planned y ~written:dsts ~read:(load_arrays ?carry loads) (fun _ ->
              Loop.scan ?carry p ~axis rk ~dsts loads)
      | Ok _ | Error _ -> A.Shape_mismatch)

(* Moves: gathers, scatters, assemblies and folds

   Each refuses what {!Nx_kernel.S} states it refuses, by
   {!Nx_kernel.Shape}'s rules, then runs one call of {!Moves} through the
   door. *)

(* Whether [want], the shapes {!Nx_kernel.Shape} gives, are [got]. *)
let fit want got = match want with Ok s -> s = got | Error _ -> false

let gather ~axis ~dst idx x =
  let want = Nx_kernel.Shape.gather ~axis (shape_of idx) (shape_of x) in
  if not (fit want (shape_of dst)) then A.Shape_mismatch
  else
    planned dst ~written:[| A.Any dst |] ~read:[| A.Any idx; A.Any x |]
      (fun _ -> Moves.gather ~axis ~dst idx x)

(* A range gather's [size] is [dst]'s extent along the runs' axis. *)
let gather_ranges ~dst ~starts ~ends x =
  let ys = shape_of dst and a = A.Layout.rank (A.layout starts) - 1 in
  let want =
    if a < 0 || a >= Array.length ys then Error "no runs' axis"
    else
      Nx_kernel.Shape.gather_ranges ~size:ys.(a) (shape_of starts)
        (shape_of ends) (shape_of x)
  in
  if not (fit want ys) then A.Shape_mismatch
  else
    planned dst ~written:[| A.Any dst |]
      ~read:[| A.Any starts; A.Any ends; A.Any x |] (fun _ ->
        Moves.gather_ranges ~dst ~starts ~ends x)

(* Whether {!Nx_kernel.Prog.Add} takes [x]'s dtype, as a sum's operands. *)
let adds x = Nx_kernel.Prog.accepts2 (Binary Add) (A.dtype x)

let scatter (c : Nx_kernel.combine) ~unique ~axis ~dst ~into idx u =
  if c = Add && not (adds u) then A.Wrong_dtype
  else
    let want =
      Nx_kernel.Shape.scatter ~axis (shape_of into) (shape_of idx) (shape_of u)
    in
    if not (fit want (shape_of dst)) then A.Shape_mismatch
    else
      planned dst ~written:[| A.Any dst |]
        ~read:[| A.Any into; A.Any idx; A.Any u |] (fun _ ->
          Moves.scatter c ~unique ~axis ~dst ~into idx u)

(* Sorts and numberings *)

(* A sort keeps [values]' extent along [axis]: {!Nx_kernel.Shape.sort}'s
   [k]. *)
let sort ~axis ~descending ~values ~positions x =
  let vs = shape_of values in
  let k =
    if axis >= 0 && axis < Array.length vs then Some vs.(axis) else None
  in
  if shape_of positions <> vs then A.Shape_mismatch
  else
    match Nx_kernel.Shape.sort ~axis ~k (shape_of x) with
    | Ok s when s = vs ->
        run (A.device values) Launch.bytes
          ~written:[| A.Any values; A.Any positions |] ~read:[| A.Any x |]
          (fun _ -> Sort.sort ~axis ~descending ~values ~positions x)
    | Ok _ | Error _ -> A.Shape_mismatch

(* A numbering's [size] is [first]'s length, and [counts] has [first]'s
   shape. *)
let unique ~ids ~first ~counts ~count keys =
  let want =
    match shape_of first with
    | [| size |] -> Nx_kernel.Shape.unique ~size (shape_of keys)
    | _ -> Error "first is not a vector"
  in
  match want with
  | Ok got
    when shape_of counts = shape_of first
         && got = (shape_of ids, shape_of first, shape_of count) ->
      run (A.device ids) Launch.bytes
        ~written:[| A.Any ids; A.Any first; A.Any counts; A.Any count |]
        ~read:[| A.Any keys |] (fun _ ->
          Sort.unique ~ids ~first ~counts ~count keys)
  | Ok _ | Error _ -> A.Shape_mismatch

let assemble ~fill ~dst pieces =
  let n = Iarray.length pieces in
  let regions = Array.init n (fun i -> fst (Iarray.get pieces i)) in
  let want = Nx_kernel.Shape.assemble (shape_of dst) regions in
  let got = Array.init n (fun i -> shape_of (snd (Iarray.get pieces i))) in
  if not (fit want got) then A.Shape_mismatch
  else
    planned dst ~written:[| A.Any dst |]
      ~read:(Iarray.map (fun (_, x) -> A.Any x) pieces)
      (fun _ -> Moves.assemble ~fill ~dst pieces)

(* A fold's operand has the shape a [Padded] load of [dst] gives. *)
let fold p ~dst x =
  if not (adds x) then A.Wrong_dtype
  else if not (fit (Nx_kernel.Shape.padded p (shape_of dst)) (shape_of x)) then
    A.Shape_mismatch
  else
    planned dst ~written:[| A.Any dst |] ~read:[| A.Any x |] (fun _ ->
        Moves.fold p ~dst x)

(* Transforms *)

(* A transform that refuses nothing: [plan] of [dst]'s device's plans of
   lines through the door, reading [x] and writing [dst]. *)
let transformed ~dst x plan =
  run (A.device dst) ~written:[| A.Any dst |] ~read:[| A.Any x |] Fft.write
    (fun dv -> plan dv.fft)

let c2c direction ~axes ~dst x =
  if not (fit (Nx_kernel.Shape.c2c ~axes (shape_of x)) (shape_of dst)) then
    A.Shape_mismatch
  else transformed ~dst x (fun c -> Fft.c2c c direction ~axes ~dst x)

(* The program that stores +0 into a complex [dst]. *)
let zero (type s) (dst : (Complex.t, s) A.t) =
  let dt = A.dtype dst in
  Nx_kernel.Prog.v ~ins:[||]
    (Iarray.of_list
       [ Nx_kernel.Prog.Const (D.Any dt, Nx_kernel.Prog.bits dt (D.zero dt)) ])
    ~outs:(Iarray.of_list [ 0 ])

let r2c ~axes ~dst x =
  if not (Nx_kernel.same_precision (A.dtype x) (A.dtype dst)) then A.Wrong_dtype
  else if not (fit (Nx_kernel.Shape.r2c ~axes (shape_of x)) (shape_of dst)) then
    A.Shape_mismatch
  else
    let last = Iarray.get axes (Iarray.length axes - 1) in
    (* A line of no point: one bin of +0. *)
    if (shape_of x).(last) = 0 then map (zero dst) ~dsts:[| A.Any dst |] [||]
    else transformed ~dst x (fun c -> Fft.r2c c ~axes ~dst x)

(* A c2r's points along the last of [axes] are [dst]'s extent there. *)
let c2r ~axes ~dst x =
  let ys = shape_of dst and last = Iarray.length axes - 1 in
  let a = if last >= 0 then Iarray.get axes last else -1 in
  let want =
    if a < 0 || a >= Array.length ys then Error "no last axis"
    else Nx_kernel.Shape.c2r ~axes ~n:ys.(a) (shape_of x)
  in
  if not (Nx_kernel.same_precision (A.dtype dst) (A.dtype x)) then A.Wrong_dtype
  else if not (fit want ys) then A.Shape_mismatch
  else transformed ~dst x (fun c -> Fft.c2r c ~axes ~dst x)

(* Matrix routines *)

let full a = Nx_kernel.full_precision (A.dtype a)

(* A routine whose dtypes and shapes fit: [plan] on [written]'s device
   through the door. *)
let routine ~(written : A.any iarray) ~(read : A.any iarray) plan =
  let (A.Any y) = Iarray.get written 0 in
  run (A.device y) ~written ~read Launch.bytes plan

(* [plan] where the dtypes are taken and [want], the shapes {!Nx_kernel.Shape}
   gives, are [got]. *)
let computed takes want got plan =
  if not takes then A.Wrong_dtype
  else
    match want with
    | Ok s when s = got -> plan ()
    | Ok _ | Error _ -> A.Shape_mismatch

let cholesky t ~dst a =
  computed (full a)
    (Nx_kernel.Shape.cholesky (shape_of a))
    (shape_of dst)
    (fun () ->
      routine ~written:[| A.Any dst |] ~read:[| A.Any a |] (fun _ ->
          Factor.cholesky t ~dst:(A.Any dst) (A.Any a)))

let lu ~lu ~pivots ~perm a =
  computed (full a)
    (Nx_kernel.Shape.lu (shape_of a))
    (shape_of lu, shape_of pivots, shape_of perm)
    (fun () ->
      routine ~written:[| A.Any lu; A.Any pivots; A.Any perm |]
        ~read:[| A.Any a |] (fun _ ->
          Factor.lu ~lu:(A.Any lu) ~pivots:(A.Any pivots) ~perm:(A.Any perm)
            (A.Any a)))

(* [q] and [r] state the factors: reduced or complete, whichever fits. *)
let qr ~q ~r a =
  let got = (shape_of q, shape_of r) in
  let fits f = Nx_kernel.Shape.qr f (shape_of a) = Ok got in
  computed (full a)
    (if fits Reduced then Ok got else Nx_kernel.Shape.qr Complete (shape_of a))
    got
    (fun () ->
      routine ~written:[| A.Any q; A.Any r |] ~read:[| A.Any a |] (fun _ ->
          Factor.qr ~q:(A.Any q) ~r:(A.Any r) (A.Any a)))

(* Whether [r], a result of floats or complex numbers, is of [a]'s
   precision. *)
let paired a r = Nx_kernel.same_precision (A.dtype a) (A.dtype r)

(* [svd]'s factors are those its destinations' shapes state: reduced or
   complete, whichever fits. *)
let svd ~s ?vectors a =
  let shape f = Nx_kernel.Shape.svd f (shape_of a) in
  let takes = full a && paired a s in
  match vectors with
  | None ->
      computed takes
        (Result.map (fun (s, _, _) -> s) (shape Reduced))
        (shape_of s)
        (fun () ->
          routine ~written:[| A.Any s |] ~read:[| A.Any a |] (fun _ ->
              Spectral.svd ~s:(A.Any s) (A.Any a)))
  | Some (~u, ~vh) ->
      let got = (shape_of s, shape_of u, shape_of vh) in
      computed takes
        (if shape Reduced = Ok got then Ok got else shape Complete)
        got
        (fun () ->
          routine ~written:[| A.Any s; A.Any u; A.Any vh |] ~read:[| A.Any a |]
            (fun _ ->
              Spectral.svd ~s:(A.Any s) ~vectors:(A.Any u, A.Any vh) (A.Any a)))

let eigh ~w ?v a =
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes = full a && paired a w in
  match v with
  | None ->
      computed takes (Result.map fst want) (shape_of w) (fun () ->
          routine ~written:[| A.Any w |] ~read:[| A.Any a |] (fun _ ->
              Spectral.eigh ~w:(A.Any w) (A.Any a)))
  | Some v ->
      computed takes want
        (shape_of w, shape_of v)
        (fun () ->
          routine ~written:[| A.Any w; A.Any v |] ~read:[| A.Any a |] (fun _ ->
              Spectral.eigh ~w:(A.Any w) ~v:(A.Any v) (A.Any a)))

let eig ~w ?v a =
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes =
    full a && paired a w && Option.fold ~none:true ~some:(paired a) v
  in
  match v with
  | None ->
      computed takes (Result.map fst want) (shape_of w) (fun () ->
          routine ~written:[| A.Any w |] ~read:[| A.Any a |] (fun _ ->
              Spectral.eig ~w:(A.Any w) (A.Any a)))
  | Some v ->
      computed takes want
        (shape_of w, shape_of v)
        (fun () ->
          routine ~written:[| A.Any w; A.Any v |] ~read:[| A.Any a |] (fun _ ->
              Spectral.eig ~w:(A.Any w) ~v:(A.Any v) (A.Any a)))

let solve_triangular t ~adjoint ~unit_diagonal ~dst a b =
  computed (full a)
    (Nx_kernel.Shape.solve_triangular (shape_of a) (shape_of b))
    (shape_of dst)
    (fun () ->
      routine ~written:[| A.Any dst |] ~read:[| A.Any a; A.Any b |] (fun _ ->
          Factor.solve_triangular t ~adjoint ~unit_diagonal ~dst:(A.Any dst)
            (A.Any a) (A.Any b)))

(* Attention *)

(* Attention computes once it refuses nothing: Nx_kernel's rule, then the
   shapes {!Nx_kernel.Shape.attention} gives. *)
let attention (p : Nx_kernel.Prog.t) ~scale ~o ~lse ?lengths q kv loads =
  let rank = A.Layout.rank (A.layout q) in
  match
    Nx_kernel.attention_refusal p ~q:(A.dtype q) ~rank ~o:(A.dtype o)
      ~acc:(A.dtype lse) ~loads:arrays loads
  with
  | Some (refused, _) -> refused
  | None -> (
      let shapes =
        match kv with
        | Nx_kernel.Dense { k; v } ->
            Nx_kernel.Dense { k = shape_of k; v = shape_of v }
        | Nx_kernel.Paged { k; v; table } ->
            Nx_kernel.Paged
              { k = shape_of k; v = shape_of v; table = shape_of table }
      in
      let lens = Option.map shape_of lengths in
      let lshapes =
        Array.init (Iarray.length loads) (fun i ->
            let (A.Any l) = Iarray.get loads i in
            shape_of l)
      in
      match
        Nx_kernel.Shape.attention (shape_of q) shapes ~lengths:lens lshapes
      with
      | Ok (os, ls) when os = shape_of o && ls = shape_of lse ->
          let read =
            Iarray.concat
              [
                (match kv with
                | Nx_kernel.Dense { k; v } -> [| A.Any q; A.Any k; A.Any v |]
                | Paged { k; v; table } ->
                    [| A.Any q; A.Any k; A.Any v; A.Any table |]);
                (match lengths with Some l -> [| A.Any l |] | None -> [||]);
                loads;
              ]
          in
          run (A.device q) ~written:[| A.Any o; A.Any lse |] ~read Launch.bytes
            (fun _ -> Attention.attention p ~scale ~o ~lse ?lengths q kv loads)
      | Ok _ | Error _ -> A.Shape_mismatch)

let numbering_words = Sort.set_run_words
