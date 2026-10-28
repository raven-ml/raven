(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

module A = Nx_array

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* The live [Extent] interpretations of the process. *)
let extents = Atomic.make 0
let quiet () = Atomic.get extents = 0

(* The eager meaning *)

(* Raises unless the operand [x] is alive and concrete or a constant: an
   interpretation receives every operation on its traced values before eager
   execution can. *)
let checked (type v s d) ~by (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s reached eager execution" by
        owner.name
  | Array _ | Shards _ | Handle _ | Deferred _ -> Prim.alive ~by x

let is_concrete x = not (Prim.is_constant x)

(* Whether [p] lies on the host's set. *)
let on_host_set p = Devices.number (Devices.set p) = Devices.number Devices.host

(* [x] placed at [p]: a constant computed there, a value onto the host from
   another set waited for first. *)
let placed : type v s d e.
    by:string -> e Devices.placement -> (v, s, d) t -> (v, s, e) t =
 fun ~by p x ->
  match x with
  | Deferred _ -> Place.value ~by p (Constant.at p x)
  | Array _ | Shards _ | Handle _ | Traced _ ->
      if on_host_set p && not (on_host_set (Prim.placement x)) then
        Check.wait ~by x;
      Exec.place ~by p x

(* [x] readable at [p]: a constant computed there. *)
let placer ~by p x = Exec.placer ~by p (Constant.at p x)

(* The memory of a handle [c] claimed that [op]'s first result may take
   ({!Donate.reuse}): the handle's buffer, dead from then on, and the array
   the result takes. A result written there lies where the handle does, so
   only an operand that lies where [r] puts the results may hold it. *)
let taken c (r : Prim.routed) op =
  match (Prim.own_index op, r.route) with
  | Some (own, layout, Nx_array.Dtype.Any dt), Some route ->
      let (Prim.Operands xs) = Prim.operands op in
      let xs = Array.of_list xs in
      let at_result i =
        let (Value.Any x) = xs.(i) in
        match Prim.at x with
        | Some p -> Devices.equal p route.result
        | None -> false
      in
      let own = List.filter at_result own in
      Option.map (fun (b, a) -> (b, A.Any a)) (Donate.reuse c ~own dt layout)
  | Some _, None | None, _ -> None

(* [f ()], the claim [c] ended once it returns or raises. Where it raises, the
   memory [into] took for its result is freed too, since no result holds
   it. *)
let ending ~by c into f =
  match f () with
  | y ->
      Donate.consumed c;
      y
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Donate.consumed c;
      let why = "donated to " ^ by in
      Option.iter
        (fun (A.Any a) -> ignore (Rig.Buffer.free ~why (A.buffer a) : bool))
        into;
      Printexc.raise_with_backtrace e bt

let rec run : type r. by:string -> r prim -> r =
 fun ~by op -> if Exec.is_local op then Exec.local ~by op else general ~by op

and general : type r. by:string -> r prim -> r =
 fun ~by op ->
  Prim.iteri (fun _ x -> checked ~by x) op;
  if Prim.exists Donate.is_handle op then donated ~by op else plain ~by op

(* [op] over donated operands: a movement that maps elements one to one passes
   a new handle on; any other operation checks its rule, claims the handles,
   reads their memory, writes its first result into a handle's memory where it
   may, and ends the claim when it returns or raises. *)
and donated : type r. by:string -> r prim -> r =
 fun ~by op ->
  match op with
  | Move (mv, (Handle d as x)) ->
      (* A view of a handle leaves the memory unshared: the handle it passes on
         has the same donor, so its consumer may still write into it.
         The movement claims nothing: the deaths wait for that consumer, which
         checks every operand first, so an operation that aligns its operands
         by movements kills nothing when its checks refuse it. *)
      let moved () =
        let r = run ~by (Move (mv, Donate.live x)) in
        (Prim.placement r, Prim.shape r, Prim.arrays r)
      in
      let at, shape, arrays =
        if Iarray.length d.arrays <> 1 then moved ()
        else begin
          ignore (Prim.layouts ~by op);
          match A.move mv (Iarray.get d.arrays 0) with
          | Some v ->
              (d.at, Iarray.to_array (A.Layout.shape (A.layout v)), [| v |])
          | None -> moved ()
        end
      in
      Handle { donor = d.donor; at; shape; arrays }
  | Map _ | Reduce _ | Scan _ | Gather _ | Scatter _ | Sort _ | Fft _
  | Linalg _ | Fold _ | Assemble _ | Contract _ | Copy _ | Move _ | Bitcast _
  | Place _ | Check _ -> (
      (* Checked before the claim: an operation its rule refuses neither takes
         a handle's memory nor shows another domain a handle dead. Once
         claimed, the donors stay dead whatever the kernels answer. *)
      let r = Prim.route ~by op in
      let (Prim.Operands xs) = Prim.operands op in
      let c = Donate.claim ~by xs in
      let op = Donate.called c op in
      match taken c r op with
      | Some (b, (A.Any a as into)) ->
          (* The handle's memory is the result's, now over [b']: the kernel
             reads each index of it before it writes the result there. *)
          let b' = A.buffer a in
          let over (type v s d) (y : (v, s, d) t) : (v, s, d) t =
            match y with
            | Array r when A.buffer r.a == b ->
                Array { r with a = A.v (A.dtype r.a) (A.layout r.a) b' }
            | Array _ | Shards _ | Handle _ | Deferred _ | Traced _ -> y
          in
          let op = Prim.map over op in
          ending ~by c (Some into) (fun () -> routed ~by ~into r op)
      | None -> ending ~by c None (fun () -> routed ~by r op))

and plain : type r. by:string -> r prim -> r =
 fun ~by op ->
  match op with
  | Place (p, x) -> placed ~by p x
  | Check { ok; data; fail } ->
      ignore (Prim.layouts ~by op);
      Check.issue ~by ok data fail
  | _ ->
      if not (Prim.exists is_concrete op) then Constant.defer ~by op
      else routed ~by (Prim.route ~by op) op

(* [op], a placement, a check, or an operation of a concrete operand, computed
   as its rule decided [r] ({!Prim.route}): over operands placed where the
   route reads them, by partials or by windows as it computes, its first
   result into [into] where it is given. *)
and routed : type r. by:string -> ?into:A.any -> Prim.routed -> r prim -> r =
 fun ~by ?into r op ->
  match op with
  | Place (p, x) -> placed ~by p x
  | Check { ok; data; fail } -> Check.issue ~by ok data fail
  | _ -> (
      (* A concrete operand has a placement, so the operation has a route. *)
      let route = Option.get r.Prim.route in
      let place i x = placer ~by route.operands.(i) x in
      let op = Prim.mapi place op in
      match route.compute with
      | Partials -> Partial.run ~by route op
      | Windows -> Exec.windows ~by ?into route r.layouts op)

(* The nodes of the formulas [i]'s rule returned. *)
let add_formula i id =
  match i.formulas with
  | Some t -> Hashtbl.replace t id ()
  | None ->
      let t = Hashtbl.create 16 in
      Hashtbl.replace t id ();
      i.formulas <- Some t

let is_formula i id =
  match i.formulas with Some t -> Hashtbl.mem t id | None -> false

(* [i]'s rule on [op]. The formulas it returns are [i]'s own: an operation
   reads them as they are. *)
let applied i ~by op =
  let r = i.rule i ~by op in
  (match i.reach with
  | Values | Adopting _ -> ()
  | Extent ->
      Prim.iter_results
        (fun y ->
          match y with
          | Deferred { node = Node n; _ } -> add_formula i n.id
          | Array _ | Shards _ | Handle _ | Traced _ -> ())
        op r);
  r

(* A formula's operation rebuilt at another brand: a map's dtypes and a loop's
   reductions, which carry it. *)
type 'd outs_at = Outs : ('d, 'r) outs -> 'd outs_at
type 'd reduction_at = Reduction : ('d, 'r) reduction -> 'd reduction_at
type 'd reductions_at = Reductions : ('d, 'r) reductions -> 'd reductions_at

let rec outs_at : type e r d. (e, r) outs -> d outs_at = function
  | [] -> Outs []
  | dt :: rest ->
      let (Outs o) = outs_at rest in
      Outs (dt :: o)

let reduction_at : type e r d. (e, r) reduction -> d reduction_at = function
  | Monoid (m, k, dt) -> Reduction (Monoid (m, k, dt))
  | Logsumexp (k, dt) -> Reduction (Logsumexp (k, dt))
  | Moments (k, dt) -> Reduction (Moments (k, dt))
  | Arg (x, k, dt) -> Reduction (Arg (x, k, dt))

let rec reductions_at : type e r d. (e, r) reductions -> d reductions_at =
  function
  | [] -> Reductions []
  | r :: rest ->
      let (Reduction r) = reduction_at r in
      let (Reductions rs) = reductions_at rest in
      Reductions (r :: rs)

(* [i]'s results on the formula operation [op] rebuilt over [sub] of its
   operands, in position order. Each constructor names its results at [sub]'s
   brand. *)
let replayed : type r d.
    interpretation ->
    by:string ->
    ('v 's 'e. ('v, 's, 'e) t -> ('v, 's, d) t) ->
    r prim ->
    d any list =
 fun i ~by sub op ->
  let load : type e. e load -> d load = function
    | Plain x -> Plain (sub x)
    | Padded { x; fill; pad } -> Padded { x = sub x; fill; pad }
  in
  let one : type v s. (v, s, d) t prim -> d any list =
   fun op -> [ Any (applied i ~by op) ]
  in
  let two : type v s w q. ((v, s, d) t * (w, q, d) t) prim -> d any list =
   fun op ->
    let a, b = applied i ~by op in
    [ Any a; Any b ]
  in
  let three : type v s w q.
      ((v, s, d) t * (w, q, d) t * (w, q, d) t) prim -> d any list =
   fun op ->
    let a, b, c = applied i ~by op in
    [ Any a; Any b; Any c ]
  in
  let all = ref ([] : d any list) in
  let add y = all := List.cons (Any y) !all in
  match op with
  | Map { layout; prog; outs; loads } ->
      let (Outs outs) = outs_at outs in
      let op = Map { layout; prog; outs; loads = Iarray.map load loads } in
      Prim.iter_outs add outs (applied i ~by op);
      List.rev !all
  | Reduce { layout; axes; prog; reductions; loads } ->
      let (Reductions reductions) = reductions_at reductions in
      let loads = Iarray.map load loads in
      let op = Reduce { layout; axes; prog; reductions; loads } in
      Prim.iter_reductions add reductions (applied i ~by op);
      List.rev !all
  | Scan { layout; axis; prog; reduction; loads; carry } ->
      let (Reduction reduction) = reduction_at reduction in
      let loads = Iarray.map load loads in
      let carry = Option.map (fun (Any c) -> Any (sub c)) carry in
      let op = Scan { layout; axis; prog; reduction; loads; carry } in
      Prim.iter_reduction add reduction (applied i ~by op);
      List.rev !all
  | Gather { axis; idx; x } -> one (Gather { axis; idx = sub idx; x = sub x })
  | Scatter { combine; unique; axis; idx; updates; into } ->
      let idx = sub idx and updates = sub updates and into = sub into in
      one (Scatter { combine; unique; axis; idx; updates; into })
  | Sort { axis; descending; k; x } ->
      two (Sort { axis; descending; k; x = sub x })
  | Fold { shape; pad; x } -> one (Fold { shape; pad; x = sub x })
  | Assemble { dtype; shape; fill; pieces } ->
      let pieces = List.map (fun (r, x) -> (r, sub x)) pieces in
      one (Assemble { dtype; shape; fill; pieces })
  | Contract
      { batch; contracting; acc; out; a; b; init; a_scale; b_scale; segments }
      ->
      let a = sub a and b = sub b and init = Option.map sub init in
      let scale : type e. e scale -> d scale = function
        | Exponent x -> Exponent (sub x)
        | Value x -> Value (sub x)
      in
      let a_scale = Option.map scale a_scale in
      let b_scale = Option.map scale b_scale in
      let segments =
        Option.map
          (fun (g : _ Nx_kernel.segments) : _ Nx_kernel.segments ->
            match g with
            | Rows r -> Rows { r with offsets = sub r.offsets }
            | Contracting r -> Contracting { r with offsets = sub r.offsets })
          segments
      in
      one
        (Contract
           { batch; contracting; acc; out; a; b; init; a_scale; b_scale;
             segments })
  | Fft (C2c { direction; axes; x }) ->
      one (Fft (C2c { direction; axes; x = sub x }))
  | Fft (R2c { dtype; axes; x }) -> one (Fft (R2c { dtype; axes; x = sub x }))
  | Fft (C2r { dtype; n; axes; x }) ->
      one (Fft (C2r { dtype; n; axes; x = sub x }))
  | Linalg (Cholesky { triangle; a }) ->
      one (Linalg (Cholesky { triangle; a = sub a }))
  | Linalg (Lu a) -> three (Linalg (Lu (sub a)))
  | Linalg (Qr { factors; a }) -> two (Linalg (Qr { factors; a = sub a }))
  | Linalg (Svd_values { dtype; a }) ->
      one (Linalg (Svd_values { dtype; a = sub a }))
  | Linalg (Svd { factors; dtype; a }) ->
      three (Linalg (Svd { factors; dtype; a = sub a }))
  | Linalg (Eigh_values { dtype; a }) ->
      one (Linalg (Eigh_values { dtype; a = sub a }))
  | Linalg (Eigh { dtype; a }) -> two (Linalg (Eigh { dtype; a = sub a }))
  | Linalg (Eig_values { dtype; a }) ->
      one (Linalg (Eig_values { dtype; a = sub a }))
  | Linalg (Eig { dtype; a }) -> two (Linalg (Eig { dtype; a = sub a }))
  | Linalg (Solve_triangular { triangle; adjoint; unit_diagonal; a; b }) ->
      let a = sub a and b = sub b in
      one
        (Linalg
           (Solve_triangular { triangle; adjoint; unit_diagonal; a; b }))
  | Copy x -> one (Copy (sub x))
  | Move (mv, x) -> one (Move (mv, sub x))
  | Bitcast (dt, x) -> one (Bitcast (dt, sub x))
  | Place _ | Check _ ->
      invalid_arg "Eval: a formula's operation is a placement or a check"

(* [x], a formula [i]'s rule did not return, as [i]'s results on its nodes:
   each node's operation rebuilt at [x]'s brand and given to [i]'s rule in
   dependency order, each once. A node of a formula [i]'s rule returned is read
   as it is. *)
let replay (type v s d) i (x : (v, s, d) t) : (v, s, d) t =
  let sub result (type w q e) (y : (w, q, e) t) : (w, q, d) t =
    match y with
    | Deferred { form; node = Node n as node; k } ->
        if is_formula i n.id then
          Deferred { form = { form with placement = None }; node; k }
        else Prim.expect form.dtype (result node ()).(k)
    | Array _ | Shards _ | Handle _ | Traced _ ->
        invalid_arg "Eval: a formula's operand is a formula"
  in
  let reads (Node n) () =
    let (Prim.Operands xs) = Prim.operands n.op in
    List.filter_map
      (fun (Any y) ->
        match y with
        | Deferred { node = Node m as node; _ } when not (is_formula i m.id) ->
            Some (node, ())
        | Deferred _ | Array _ | Shards _ | Handle _ | Traced _ -> None)
      xs
  in
  let rebuilt result (Node n) () =
    Array.of_list (replayed i ~by:n.by (fun y -> sub result y) n.op)
  in
  match x with
  | Deferred { form; node; k } ->
      let same () () = true in
      Prim.expect form.dtype (Prim.evaluate ~same ~reads rebuilt node ()).(k)
  | Array _ | Shards _ | Handle _ | Traced _ -> x

(* Whether [i]'s rule receives [x] changed: for an [Extent], a formula its
   rule did not return replayed. *)
let changes (type v s d) i (x : (v, s, d) t) =
  match x with
  | Deferred { node = Node n; _ } -> (
      match i.reach with
      | Extent -> not (is_formula i n.id)
      | Values | Adopting _ -> false)
  | Array _ | Shards _ | Handle _ | Traced _ -> false

let received (type v s d) i (x : (v, s, d) t) : (v, s, d) t =
  match x with
  | Deferred _ when changes i x -> replay i x
  | Deferred _ | Array _ | Shards _ | Handle _ | Traced _ -> x

let is_traced (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced _ -> true
  | Array _ | Shards _ | Handle _ | Deferred _ -> false

(* Whether [i] receives [o]'s values as its own: [o] is [i] or what [i]
   adopts. *)
let rec adopts i o =
  o == i
  || match i.reach with Adopting p -> adopts p o | Values | Extent -> false

let owns i (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> adopts i owner
  | Array _ | Shards _ | Handle _ | Deferred _ -> false

let refuse ~by o =
  if Atomic.get o.live then
    Format.kasprintf invalid_arg "%s: a value of %s, used outside %s's extent"
      by o.name o.name
  else
    Format.kasprintf invalid_arg "%s: a value of %s, used after %s returned" by
      o.name o.name

(* Dispatch

   [Receive] carries an operation to the handlers around the caller. The
   innermost whose interpretation reaches it runs the rule in its clause, on
   the handler's parent fiber: outside the interpretation and every handler
   inside it, with the extent suspended until the rule returns. The cell is
   emptied once no handler answered: [Effect.Unhandled] holds the effect, and
   the runtime keeps the last exception raised reachable while backtraces are
   recorded.

   [Ask] puts a question about the caller to [of_]'s handler. [Around] asks
   whether that handler is around the caller, which it answers alone.
   [Readable] asks whether it is around the caller with no [Extent] handler
   between them, and an [Extent] handler in between answers [false]. Under
   no handler of [of_], the answer is [false]. *)
type question = Around | Readable

type _ Effect.t +=
  | Receive : { by : string; mutable op : 'r prim option } -> 'r Effect.t
  | Ask : { of_ : interpretation; what : question } -> bool Effect.t

let ask o what =
  match Effect.perform (Ask { of_ = o; what }) with
  | b -> b
  | exception Effect.Unhandled (Ask _) -> false

(* Whether [o]'s handler is around the caller. *)
let around o = ask o Around

(* [i]'s rule on [op], in [i]'s clause. Another interpretation's traced operand
   is refused unless its interpretation is around [i], which is asked once per
   interpretation; one [i] adopts is around by construction ({!interpret}). The
   rule reads each handle as its arrays, lent, and kills nothing. *)
let deliver i ~by op =
  let checked = ref ([] : interpretation list) in
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ }
        when (not (adopts i owner)) && not (List.memq owner !checked) ->
          if not (around owner) then refuse ~by owner;
          checked := List.cons owner !checked
      | Traced _ | Array _ | Shards _ | Handle _ | Deferred _ -> ())
    op;
  Prim.iteri (fun _ x -> Prim.alive ~by x) op;
  let op =
    if Prim.exists (fun x -> changes i x || Donate.is_handle x) op then
      Prim.map (fun x -> Donate.lent (received i x)) op
    else op
  in
  applied i ~by op

let reaches i op =
  match i.reach with
  | Extent -> true
  | Values | Adopting _ -> Prim.exists (fun x -> owns i x) op

(* Whether [i], whose handler was installed on the domain [here], answers on
   the calling domain, asked by its handler once it has taken an effect. An
   adopting interpretation's rule may share its parent's state, which no lock
   guards: on another domain it answers only where its parent is around too,
   so that the parent's extent moved with it and no rule of theirs runs
   elsewhere. *)
let present i here =
  match i.reach with
  | Values | Extent -> true
  | Adopting p -> (Domain.self () :> int) = here || around p

(* Refuses [op], which [i] reaches where it is not present: names the first
   operand's owner that [i] adopts. *)
let absent i ~by op =
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ } when adopts i owner -> refuse ~by owner
      | Traced _ | Array _ | Shards _ | Handle _ | Deferred _ -> ())
    op;
  invalid_arg "Eval.absent: an operation its interpretation does not reach"

let handler i =
  let here = (Domain.self () :> int) in
  {
    Effect.Deep.effc =
      (fun (type a) (e : a Effect.t) ->
        match e with
        | Receive { by; op = Some op } when reaches i op ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                match
                  if present i here then deliver i ~by op else absent i ~by op
                with
                | r -> Effect.Deep.continue k r
                | exception x ->
                    let bt = Printexc.get_raw_backtrace () in
                    Effect.Deep.discontinue_with_backtrace k x bt)
        | Ask { of_; _ } when of_ == i ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                Effect.Deep.continue k (present i here))
        | Ask { what = Readable; _ } when i.reach = Extent ->
            (* An extent inside the owner stages what it reaches: a read
               there would hide its value from it. *)
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                Effect.Deep.continue k false)
        | _ -> None);
  }

(* Refuses to adopt [p] unless its values may skip the check that [p] is
   around: [p] reaches its values alone, and is live and around the caller,
   so around the adopting extent. *)
let adoptable ~name p =
  match p.reach with
  | Extent ->
      Format.kasprintf invalid_arg "%s: adopts %s, an Extent" name p.name
  | Values | Adopting _ ->
      if not (Atomic.get p.live) then
        Format.kasprintf invalid_arg "%s: adopts %s, which returned" name
          p.name;
      if not (around p) then
        Format.kasprintf invalid_arg "%s: adopts %s outside %s's extent" name
          p.name p.name

let interpret ~name reach
    (rule : 'r. interpretation -> by:string -> 'r prim -> 'r) f =
  (match reach with
  | Adopting p -> adoptable ~name p
  | Values | Extent -> ());
  let live = Atomic.make true in
  let i = { name; reach; rule; live; formulas = None } in
  let run () = Effect.Deep.try_with f i (handler i) in
  match reach with
  | Values | Adopting _ ->
      Fun.protect ~finally:(fun () -> Atomic.set live false) run
  | Extent ->
      Atomic.incr extents;
      Fun.protect
        ~finally:(fun () ->
          Atomic.set live false;
          Atomic.decr extents)
        run

let traced i form payload = Traced { form; owner = i; payload; twin = None }

let twinned i c payload =
  Traced { form = Prim.form c; owner = i; payload; twin = Some c }

(* Whether [o]'s handler is around the caller with no [Extent] handler
   between them. *)
let readable o = Atomic.get o.live && ask o Readable

let read_through (type v s d) (x : (v, s, d) t) : (v, s, d) t =
  let rec go : (v, s, d) t -> (v, s, d) t = function
    | Traced { owner; twin = Some c; _ } when readable owner -> go c
    | x -> x
  in
  go x

let payload (type v s d) i (x : (v, s, d) t) =
  match x with
  | Traced { owner; payload; _ } when owner == i -> Some payload
  | Traced _ | Array _ | Shards _ | Handle _ | Deferred _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> Some owner
  | Array _ | Shards _ | Handle _ | Deferred _ -> None

(* An operation no handler answered computes, unless an operand is traced. *)
let unreached ~by op =
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ } -> refuse ~by owner
      | Array _ | Shards _ | Handle _ | Deferred _ -> ())
    op;
  run ~by op

let eval ~by op =
  if quiet () && (Exec.is_local op || not (Prim.exists is_traced op)) then
    run ~by op
  else
    let e = Receive { by; op = Some op } in
    match Effect.perform e with
    | r -> r
    | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e ->
        (match e with Receive cell -> cell.op <- None | _ -> ());
        unreached ~by op

(* The one-node maps, built and evaluated: the paths an interpretation may
   reach. *)
let eval1 ~by k dt x =
  let v, () = eval ~by (Prim.op1 ~by k dt x) in
  v

let eval2 ~by k dt x y =
  let v, () = eval ~by (Prim.op2 ~by k dt x y) in
  v

let eval3 ~by k c x y =
  let v, () = eval ~by (Prim.op3 ~by k c x y) in
  v

(* No traced operand and no live [Extent] is the condition under which {!eval}
   performs nothing, checked on the operands before the operation is built.
   Live arrays at physically one placement whose set has kernels, where [k]
   takes their dtypes, then compute by {!Exec.map}, and every other operand
   by {!run} of the operation. *)
let fast1 ~by k dt x =
  match (x, Exec.live_kernels x, Prim.node1 k (Prim.dtype x) dt) with
  | Array { at; a; _ }, Some kernels, Some p ->
      Exec.map ~by kernels at p dt a [| Nx_kernel.Plain a |]
  | _ -> fst (run ~by (Prim.op1 ~by k dt x))

let fast2 ~by k dt x y =
  match (x, y, Exec.live_kernels x, Prim.node2 k (Prim.dtype x)) with
  | Array { at; a; _ }, Array { a = b; _ }, Some kernels, Some p
    when Exec.beside x y ->
      Exec.map ~by kernels at p dt a [| Nx_kernel.Plain a; Nx_kernel.Plain b |]
  | _ -> fst (run ~by (Prim.op2 ~by k dt x y))

let fast3 ~by k c x y =
  match
    (c, x, y, Exec.live_kernels c, Prim.node3 k (Prim.dtype c) (Prim.dtype x))
  with
  | ( Array { at; a = ca; _ },
      Array { a; _ },
      Array { a = b; _ },
      Some kernels,
      Some p )
    when Exec.beside c x && Exec.beside c y ->
      let loads =
        [| Nx_kernel.Plain ca; Nx_kernel.Plain a; Nx_kernel.Plain b |]
      in
      Exec.map ~by kernels at p (A.dtype a) a loads
  | _ -> fst (run ~by (Prim.op3 ~by k c x y))

let apply1 ~by k dt x =
  match x with
  | Traced _ -> eval1 ~by k dt x
  | Array _ | Shards _ | Handle _ | Deferred _ ->
      if quiet () then fast1 ~by k dt x else eval1 ~by k dt x

let apply2 ~by k dt x y =
  match (x, y) with
  | Traced _, _ | _, Traced _ -> eval2 ~by k dt x y
  | _ -> if quiet () then fast2 ~by k dt x y else eval2 ~by k dt x y

let apply3 ~by k c x y =
  match (c, x, y) with
  | Traced _, _, _ | _, Traced _, _ | _, _, Traced _ -> eval3 ~by k c x y
  | _ -> if quiet () then fast3 ~by k c x y else eval3 ~by k c x y

let contract ~by ~batch ~contracting ~acc dt a b init =
  let fast =
    if quiet () then Exec.contract ~by ~batch ~contracting ~acc dt a b init
    else None
  in
  match fast with
  | Some y -> y
  | None ->
      eval ~by
        (Contract
           {
             batch;
             contracting;
             acc;
             out = dt;
             a;
             b;
             init;
             a_scale = None;
             b_scale = None;
             segments = None;
           })

let place ~by p x =
  match x with
  | Array r when r.at == p && r.dead == Live && quiet () ->
      (* A second value over the memory shares it. *)
      Rig.Claim.share (Nx_array.buffer r.a);
      Array { at = p; a = r.a; dead = Live }
  | Array _ | Shards _ | Handle _ | Deferred _ | Traced _ ->
      eval ~by (Place (p, x))
