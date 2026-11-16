(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call refuses what nx.kernel's rules refuse, then its family plans it
   as one call ({!Plan}), one submission of its launches, which the door
   submits:

   map --> device (Rig.Local) --> Loop.map --> Plan.t --> door
   --> the submission of its key, its slots bound to the call's arrays,
       the device's workspace and tickets and fresh buffers, a run of it
       written --> Rig.submit --> fresh buffers freed

   A call that finds every cache warm allocates its plan, its launches, its
   slots' buffers and the door's work, and nothing that outlives it. *)

module A = Nx_array
module D = Nx_array.Dtype
module Sub = Rig.Submission

let name = "nx.amd"

external code_object : string -> string option = "nx_amd_code_object"

(* Devices *)

(* What this library keeps for a device [rig]: its kernels' functions by
   {!Kernels.index}; the workspace, which holds a call's temporaries, grown
   to {!Plan.kept_workspace} bytes; the tickets, zero words that every call
   leaves zero, grown to {!Plan.kept_tickets} bytes, so no launch zeroes
   them; and in [subs] the submissions of the calls made, by their key
   ({!Plan.key}), a cell for each first kernel ({!Kept}); and in
   [constants] the constants' buffers by key. Every call that uses the
   workspace or the tickets names them as written, so rig orders those
   calls. *)
type device = {
  rig : Rig.t;
  funcs : Rig.Image.func array;
  workspace : Rig.Buffer.t Atomic.t;
  tickets : Rig.Buffer.t Atomic.t;
  subs : (int array, Sub.t) Kept.t;
  constants : (string, Rig.Buffer.t) Cache.t;
}

let rec resolve image fs i =
  if i = Array.length Kernels.all then Some (Array.of_list (List.rev fs))
  else
    match Rig.Image.resolve image (Kernels.name Kernels.all.(i)) with
    | None -> None
    | Some f -> resolve image (f :: fs) (i + 1)

(* A device of a processor the code object runs on, which loads it and
   every kernel of it. *)
let load d =
  match code_object (Rig.arch d) with
  | None -> None
  | Some co -> (
      match Rig.Image.load d co with
      | Error _ -> None
      | Ok image -> (
          match resolve image [] 0 with
          | None -> None
          | Some funcs ->
              Some
                {
                  rig = d;
                  funcs;
                  workspace = Atomic.make (Rig.Buffer.create d 0);
                  tickets = Atomic.make (Rig.Buffer.create d 0);
                  subs =
                    Kept.make
                      ~cells:(Array.length Kernels.all)
                      ~count:Plan.kept_count;
                  constants = Cache.make ();
                }))

(* Each device's [load], [None] where it does not compute. *)
let devices = Rig.Local.new_key load
let device d = Rig.Local.get devices d
let computes_on d = Option.is_some (device d)

(* [cell]'s buffer of at least [need] bytes, grown to twice its length or
   [need] by [make], at most [cap]. A race stores only a larger one, so it
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
  else
    grow dv.workspace need ~cap:Plan.kept_workspace (Rig.Buffer.create dv.rig)

let zeros d bytes =
  let b = Rig.Buffer.create d bytes in
  Rig.Buffer.fill b '\000';
  b

let tickets dv need =
  let t = Atomic.get dv.tickets in
  if Rig.Buffer.length t >= need then t
  else grow dv.tickets need ~cap:Plan.kept_tickets (zeros dv.rig)

(* Submissions

   A device keeps the submission of each call's key ({!Plan.key}), made at
   its first use, in a cell for each first kernel ({!Kept}) of at most
   {!Plan.kept_count}, so that a lookup reads few keys, and at most
   {!Plan.kept_most} bytes of them in all, the least recently used dropped
   past them. A submission keeps a run for each call that wrote it while
   another did ({!Sub.take}). A call that is not kept ({!Plan.finish})
   takes a submission of its own. *)
let submission dv (c : Plan.t) =
  if not c.kept then Sub.make dv.rig (Plan.parts dv.funcs c)
  else
    let cell = Kernels.index c.launches.(0).kernel in
    match Kept.find_by Plan.matches dv.subs cell c with
    | Some s -> s
    | None ->
        let s = Sub.make dv.rig (Plan.parts dv.funcs c) in
        Kept.keep dv.subs cell ~most:(Plan.kept_most ()) ~bytes:c.bytes
          (Plan.key c) s;
        s

(* Constants

   The most bytes of constants a device keeps, {!Plan.kept_constants}
   but in tests. *)
let constants_bound = Atomic.make Plan.kept_constants

let bound_constants n =
  if n < 0 || n > Plan.kept_constants then
    invalid_arg "Nx_amd.bound_constants";
  Atomic.set constants_bound n

let wrong_length key ~length n =
  invalid_arg
    (Printf.sprintf "Nx_amd: constant %S of %d bytes, not %d" key n length)

(* The buffer of the constant [key] of [length] bytes [bytes ()] on [dv],
   and whether [dv] keeps it: the one it keeps, else a new one that
   [bytes ()] are uploaded into, kept with the newest others while they
   number at most {!Plan.kept_constant_count} and fit [bound]. A buffer
   the cache drops is left to rig's reclaim, never freed: a call on
   another domain may have just found it. *)
let constant dv ~bound key ~length bytes =
  match Cache.find dv.constants key with
  | Some b ->
      let n = Rig.Buffer.length b in
      if n <> length then wrong_length key ~length n;
      (b, true)
  | None ->
      let bytes = bytes () in
      let n = Bigarray.Array1.dim bytes in
      if n <> length then wrong_length key ~length n;
      let b = Rig.Buffer.create dv.rig n in
      (match Rig.Buffer.copy ~src:(Rig.Buffer.of_bigarray bytes) ~dst:b with
      | () -> ()
      | exception e ->
          ignore (Rig.Buffer.free b);
          raise e);
      let size = Rig.Buffer.length and count = Plan.kept_constant_count in
      (Cache.keep ~count ~bytes:(bound, size) dv.constants key b, n <= bound)

(* Binding

   [owned.(i)] is whether the buffer of slot [i] is the call's own, which
   the door frees once it has submitted the call. *)

let bind dv (c : Plan.t) owned i (s : Plan.slot) =
  match s with
  | Array b -> b
  | Workspace -> workspace dv c.workspace
  | Tickets -> tickets dv c.tickets
  | Fresh n ->
      owned.(i) <- true;
      Rig.Buffer.create dv.rig n
  | Constant { key; length; bytes } ->
      let bound = Atomic.get constants_bound in
      let b, kept = constant dv ~bound key ~length bytes in
      owned.(i) <- not kept;
      b

let free_owned owned buffers =
  Array.iteri
    (fun i own -> if own then ignore (Rig.Buffer.free buffers.(i)))
    owned

(* [c]'s slots' buffers, bound in order. A bind that raises frees the
   call's own buffers bound before it. *)
let bind_all dv (c : Plan.t) owned =
  let made = ref [] in
  let bind_one i s =
    let b = bind dv c owned i s in
    if owned.(i) then made := b :: !made;
    b
  in
  match Array.mapi bind_one c.slots with
  | buffers -> buffers
  | exception e ->
      List.iter (fun b -> ignore (Rig.Buffer.free b)) !made;
      raise e

(* Writes [c] into a run of its submission and submits it with its slots'
   buffers, then frees its own buffers, which return once that work is
   done, also where the write or the submit raises. A write that raises
   releases the run; a submit releases it itself. *)
let issue (dv, (c : Plan.t)) =
  let s = submission dv c in
  let owned = Array.make (Array.length c.slots) false in
  let buffers = bind_all dv c owned in
  Fun.protect ~finally:(fun () -> free_owned owned buffers) @@ fun () ->
  let r = Sub.take s in
  match Plan.write c r s with
  | () -> ignore (Rig.submit r ~buffers)
  | exception e ->
      Sub.Run.release r;
      raise e

let nothing () = ()
let dead (A.Any x) = Option.is_some (Rig.Buffer.dead (A.buffer x))

let rec any_dead xs i =
  i < Iarray.length xs && (dead (Iarray.get xs i) || any_dead xs (i + 1))

(* A call that refuses nothing: on [d], [plan] of its device through the
   door, claiming [written] and [read]. The door refuses a dead buffer
   before the plan reads its address. A plan of no launch, a call of no
   element, has no work. *)
let run d ~written ~read plan =
  match device d with
  | None -> A.Declined
  | Some dv -> (
      if any_dead written 0 || any_dead read 0 then
        A.door ~written ~read nothing ()
      else
        match plan dv with
        | Error answer -> answer
        | Ok None -> A.door ~written ~read nothing ()
        | Ok (Some c) -> A.door ~written ~read issue (dv, c))

let shape_of a = Iarray.to_array (A.Layout.shape (A.layout a))

(* Loops *)

(* The arrays a loop reads: its loads' arrays, and its carry. *)
let load_arrays ?carry (loads : Nx_kernel.load iarray) =
  let xs =
    Iarray.map
      (function Nx_kernel.Plain x -> A.Any x | Padded { x; _ } -> A.Any x)
      loads
  in
  match carry with None -> xs | Some c -> Iarray.append xs (Iarray.of_list [ c ])

(* Refusals: Nx_kernel's rules, which read dtypes and counts, then the
   shapes they leave to a library: each load's once loaded and the
   destinations', as {!Nx_kernel.Shape} gives them. *)

let any_dtype (A.Any a) = D.Any (A.dtype a)

let reductions =
  {
    Nx_kernel.reductions = Iarray.length;
    kind = (fun rs i -> fst (Iarray.get rs i));
    output = (fun rs i -> snd (Iarray.get rs i));
  }

let arrays =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype = (fun a i -> any_dtype (Iarray.get a i));
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
        run (A.device y) ~written:dsts ~read:(load_arrays loads) (fun _ ->
            Loop.map p ~dsts loads)

(* Reductions *)

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
          (* No term: each destination element takes the identity. *)
          if Array.mem 0 s && not (Array.mem 0 want) then
            map (identities p rs dsts) ~dsts [||]
          else
            let (A.Any y) = Iarray.get dsts 0 in
            run (A.device y) ~written:dsts ~read:(load_arrays loads) (fun _ ->
                Loop.reduce p ~shape ~axes rs ~dsts loads)
      | Ok _ | Error _ -> A.Shape_mismatch)

(* Scans *)

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
          run (A.device y) ~written:dsts ~read:(load_arrays ?carry loads)
            (fun _ -> Loop.scan ?carry p ~axis rk ~dsts loads)
      | Ok _ | Error _ -> A.Shape_mismatch)

(* Gathers, scatters, assemblies and folds

   Each refuses what {!Nx_kernel.S} states it refuses, by {!Nx_kernel.Shape}'s
   rules, then plans one call of {!Moves}. *)

(* Whether [want], the shapes {!Nx_kernel.Shape} gives, are [got]. *)
let fit want got = match want with Ok s -> s = got | Error _ -> false

let gather ~axis ~dst idx x =
  let want = Nx_kernel.Shape.gather ~axis (shape_of idx) (shape_of x) in
  if not (fit want (shape_of dst)) then A.Shape_mismatch
  else
    run (A.device dst) ~written:[| A.Any dst |] ~read:[| A.Any idx; A.Any x |]
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
    run (A.device dst) ~written:[| A.Any dst |]
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
      run (A.device dst) ~written:[| A.Any dst |]
        ~read:[| A.Any into; A.Any idx; A.Any u |] (fun _ ->
          Moves.scatter c ~unique ~axis ~dst ~into idx u)

let assemble ~fill ~dst pieces =
  let n = Iarray.length pieces in
  let regions = Array.init n (fun i -> fst (Iarray.get pieces i)) in
  let want = Nx_kernel.Shape.assemble (shape_of dst) regions in
  let got = Array.init n (fun i -> shape_of (snd (Iarray.get pieces i))) in
  if not (fit want got) then A.Shape_mismatch
  else
    run (A.device dst) ~written:[| A.Any dst |]
      ~read:(Iarray.map (fun (_, x) -> A.Any x) pieces)
      (fun _ -> Moves.assemble ~fill ~dst pieces)

(* A fold's operand has the shape a [Padded] load of [dst] gives. *)
let fold p ~dst x =
  if not (adds x) then A.Wrong_dtype
  else if not (fit (Nx_kernel.Shape.padded p (shape_of dst)) (shape_of x)) then
    A.Shape_mismatch
  else
    run (A.device dst) ~written:[| A.Any dst |] ~read:[| A.Any x |] (fun _ ->
        Moves.fold p ~dst x)

(* Transforms *)

(* A transform that refuses nothing: [plan ()] through the door, reading
   [x] and writing [dst]. *)
let transformed ~dst x plan =
  run (A.device dst) ~written:[| A.Any dst |] ~read:[| A.Any x |] (fun _ ->
      Ok (plan ()))

let c2c direction ~axes ~dst x =
  if not (fit (Nx_kernel.Shape.c2c ~axes (shape_of x)) (shape_of dst)) then
    A.Shape_mismatch
  else transformed ~dst x (fun () -> Fft.c2c direction ~axes ~dst x)

let r2c ~axes ~dst x =
  if not (Nx_kernel.same_precision (A.dtype x) (A.dtype dst)) then A.Wrong_dtype
  else if not (fit (Nx_kernel.Shape.r2c ~axes (shape_of x)) (shape_of dst)) then
    A.Shape_mismatch
  else transformed ~dst x (fun () -> Fft.r2c ~axes ~dst x)

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
  else transformed ~dst x (fun () -> Fft.c2r ~axes ~dst x)

(* Contractions *)

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

(* The answer a contraction gives before it plans, if any. *)
let refusal ~batch ~contracting ~segments ~acc ~dst ~init a b =
  let (D.Any acc_dt) = acc in
  if not (Nx_kernel.accumulates acc_dt) then Some A.Wrong_dtype
  else if not (contract_fits ~batch ~contracting ~segments ~dst ~init a b) then
    Some A.Shape_mismatch
  else None

let plan_contract ~batch ~contracting ~segments ~acc ~dst ~init oa ob =
  Contract.plan ~batch ~contracting ~segments ~acc ~dst:(A.Any dst)
    ~init:(Option.map (fun i -> A.Any i) init)
    oa ob

let contract_plan ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  match refusal ~batch ~contracting ~segments ~acc ~dst ~init oa ob with
  | Some e -> Error e
  | None -> plan_contract ~batch ~contracting ~segments ~acc ~dst ~init oa ob

let contract ~batch ~contracting ~segments ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  match refusal ~batch ~contracting ~segments ~acc ~dst ~init oa ob with
  | Some e -> e
  | None ->
      run (A.device dst) ~written:[| A.Any dst |]
        ~read:(reads ~init ~segments oa ob) (fun _ ->
          plan_contract ~batch ~contracting ~segments ~acc ~dst ~init oa ob)

(* Sorts and numberings *)

let sort ~axis ~descending ~values ~positions x =
  let vs = shape_of values in
  let k =
    if axis >= 0 && axis < Array.length vs then Some vs.(axis) else None
  in
  if shape_of positions <> vs then A.Shape_mismatch
  else if not (fit (Nx_kernel.Shape.sort ~axis ~k (shape_of x)) vs) then
    A.Shape_mismatch
  else
    run (A.device values) ~written:[| A.Any values; A.Any positions |]
      ~read:[| A.Any x |] (fun _ ->
        Sort.sort ~axis ~descending ~values ~positions x)

(* A numbering's [size] is [first]'s length, and [counts] has [first]'s
   shape. *)
let unique ~ids ~first ~counts ~count keys =
  let want =
    match shape_of first with
    | [| size |] -> Nx_kernel.Shape.unique ~size (shape_of keys)
    | _ -> Error "first is not a vector"
  in
  let got = (shape_of ids, shape_of first, shape_of count) in
  if shape_of counts <> shape_of first || not (fit want got) then
    A.Shape_mismatch
  else
    run (A.device ids)
      ~written:[| A.Any ids; A.Any first; A.Any counts; A.Any count |]
      ~read:[| A.Any keys |] (fun _ ->
        Sort.unique ~ids ~first ~counts ~count keys)

(* Matrix routines *)

let full a = Nx_kernel.full_precision (A.dtype a)

(* Whether [r], a result of floats or complex numbers, is of [a]'s
   precision. *)
let paired a r = Nx_kernel.same_precision (A.dtype a) (A.dtype r)

(* [plan] where the dtypes are taken and [want], the shapes {!Nx_kernel.Shape}
   gives, are [got]: on [written]'s first device through the door. *)
let routine ~takes want got ~(written : A.any iarray) ~(read : A.any iarray)
    plan =
  if not takes then A.Wrong_dtype
  else
    match want with
    | Ok s when s = got ->
        let (A.Any y) = Iarray.get written 0 in
        run (A.device y) ~written ~read (fun _ -> Ok (plan ()))
    | Ok _ | Error _ -> A.Shape_mismatch

let cholesky t ~dst a =
  routine ~takes:(full a)
    (Nx_kernel.Shape.cholesky (shape_of a))
    (shape_of dst) ~written:[| A.Any dst |] ~read:[| A.Any a |] (fun () ->
      Factor.cholesky t ~dst:(A.Any dst) (A.Any a))

let lu ~lu ~pivots ~perm a =
  routine ~takes:(full a)
    (Nx_kernel.Shape.lu (shape_of a))
    (shape_of lu, shape_of pivots, shape_of perm)
    ~written:[| A.Any lu; A.Any pivots; A.Any perm |] ~read:[| A.Any a |]
    (fun () ->
      Factor.lu ~lu:(A.Any lu) ~pivots:(A.Any pivots) ~perm:(A.Any perm)
        (A.Any a))

(* [q] and [r] state the factors: reduced or complete, whichever fits. *)
let qr ~q ~r a =
  let got = (shape_of q, shape_of r) in
  let fits f = Nx_kernel.Shape.qr f (shape_of a) = Ok got in
  routine ~takes:(full a)
    (if fits Reduced then Ok got else Nx_kernel.Shape.qr Complete (shape_of a))
    got ~written:[| A.Any q; A.Any r |] ~read:[| A.Any a |] (fun () ->
      Factor.qr ~q:(A.Any q) ~r:(A.Any r) (A.Any a))

let solve_triangular t ~adjoint ~unit_diagonal ~dst a b =
  routine ~takes:(full a)
    (Nx_kernel.Shape.solve_triangular (shape_of a) (shape_of b))
    (shape_of dst) ~written:[| A.Any dst |] ~read:[| A.Any a; A.Any b |]
    (fun () ->
      Factor.solve_triangular t ~adjoint ~unit_diagonal ~dst:(A.Any dst)
        (A.Any a) (A.Any b))

(* [svd]'s factors are those its destinations' shapes state: reduced or
   complete, whichever fits. *)
let svd ~s ?vectors a =
  let shape f = Nx_kernel.Shape.svd f (shape_of a) in
  let takes = full a && paired a s in
  match vectors with
  | None ->
      routine ~takes
        (Result.map (fun (s, _, _) -> s) (shape Reduced))
        (shape_of s) ~written:[| A.Any s |] ~read:[| A.Any a |] (fun () ->
          Spectral.svd ~s:(A.Any s) (A.Any a))
  | Some (~u, ~vh) ->
      let got = (shape_of s, shape_of u, shape_of vh) in
      routine ~takes
        (if shape Reduced = Ok got then Ok got else shape Complete)
        got ~written:[| A.Any s; A.Any u; A.Any vh |] ~read:[| A.Any a |]
        (fun () ->
          Spectral.svd ~s:(A.Any s) ~vectors:(A.Any u, A.Any vh) (A.Any a))

let eigh ~w ?v a =
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes = full a && paired a w in
  match v with
  | None ->
      routine ~takes (Result.map fst want) (shape_of w) ~written:[| A.Any w |]
        ~read:[| A.Any a |] (fun () -> Spectral.eigh ~w:(A.Any w) (A.Any a))
  | Some v ->
      routine ~takes want
        (shape_of w, shape_of v)
        ~written:[| A.Any w; A.Any v |] ~read:[| A.Any a |] (fun () ->
          Spectral.eigh ~w:(A.Any w) ~v:(A.Any v) (A.Any a))

let eig ~w ?v a =
  let want = Nx_kernel.Shape.eigh (shape_of a) in
  let takes =
    full a && paired a w && Option.fold ~none:true ~some:(paired a) v
  in
  match v with
  | None ->
      routine ~takes (Result.map fst want) (shape_of w) ~written:[| A.Any w |]
        ~read:[| A.Any a |] (fun () -> Spectral.eig ~w:(A.Any w) (A.Any a))
  | Some v ->
      routine ~takes want
        (shape_of w, shape_of v)
        ~written:[| A.Any w; A.Any v |] ~read:[| A.Any a |] (fun () ->
          Spectral.eig ~w:(A.Any w) ~v:(A.Any v) (A.Any a))

(* Families not yet computed here

   Each declines every call, which the contract admits of any call, before
   it reads an array. *)

let attention _ ~scale:_ ~o:_ ~lse:_ ?lengths:_ _ _ _ = A.Declined

module Kernels = Kernels
module Kept = Kept
module Plan = Plan
module Sort = Sort

let bound_fresh = Plan.bound_fresh
let contract_wide_k = Contract.contract_wide_k
let contract_unit_tile = Contract.contract_unit_tile
let bound_kept = Plan.bound_kept
let bound_scatters = Moves.bound_scatters
let bound_iterations = Spectral.bound_iterations

let kept_calls d =
  match device d with None -> 0 | Some dv -> Kept.count dv.subs

let kept_bytes d =
  match device d with None -> 0 | Some dv -> Kept.bytes dv.subs

let drop_kept d =
  match device d with None -> () | Some dv -> Kept.clear dv.subs

let kept_constants d =
  match device d with None -> 0 | Some dv -> Cache.count dv.constants

let call d ~written ~read c = run d ~written ~read (fun _ -> Ok (Some c))

(* A routine's product, for tests: its matrices where they lie, or with
   [workspace] each copied into a piece of the workspace laid out as it
   is, the product computed there and y copied back. *)
let product ?(init_first = false) ?(neg = false) ?(conj_a = false)
    ?(conj_b = false) ?tri ?ranges ?(workspace = false) ~acc ~init ~y a m =
  let o = { Contract.init_first; neg; conj_a; conj_b; tri; ranges } in
  let (A.Any yy) = y in
  run (A.device yy) ~written:[| y |] ~read:[| a; m |] (fun _ ->
      let b = Plan.make () in
      if not workspace then
        Contract.product b o ~acc ~init ~y:(Contract.array y) (Contract.array a)
          (Contract.array m)
      else begin
        (* A piece of the workspace laid out as [x] is in its buffer. *)
        let piece (A.Any x) =
          let l = A.layout x and dt = A.dtype x in
          let at = Plan.take b (snd (A.Layout.span l) * (D.bits dt / 8)) in
          Contract.work ~at (D.Any dt) l
        in
        let into x =
          let w = piece x in
          Contract.copy b ~src:(Contract.array x) ~dst:w;
          w
        in
        let wa = into a and wm = into m in
        (* [tri] keeps the outputs of the tiles it skips. *)
        let wy = if init || tri <> None then into y else piece y in
        Contract.product b o ~acc ~init ~y:wy wa wm;
        Contract.copy b ~src:wy ~dst:(Contract.array y)
      end;
      Ok (Plan.finish_some b))
