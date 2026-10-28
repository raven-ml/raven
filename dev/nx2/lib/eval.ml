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

(* Raises unless the operand [x] is alive and concrete: an interpretation
   receives every operation on its traced values before eager execution
   can. *)
let checked (type v s d) ~by (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s reached eager execution" by
        owner.name
  | Array _ | Shards _ | Handle _ -> Prim.alive ~by x

(* [x] placed at [p], a value onto the host from another set waited for
   first. *)
let placed ~by p x =
  if Devices.on_host_set p && not (Devices.on_host_set (Prim.placement x)) then
    Check.wait ~by x;
  Exec.place ~by p x

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
        Devices.equal (Prim.placement x) route.result
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
  | Move (mv, (Handle { donor; over } as x)) ->
      (* A view of a handle's one array, made here, leaves the memory
         unshared, where [run]'s view would share it: the handle it passes on
         has the same donor, so its consumer may still write into it. The
         movement claims nothing: the deaths wait for that consumer, which
         checks every operand first, so an operation that aligns its operands
         by movements kills nothing when its checks refuse it. *)
      let moved () = run ~by (Move (mv, Donate.live x)) in
      let over =
        match over with
        | Array r -> (
            ignore (Prim.layouts ~by op);
            match A.move mv r.a with
            | Some a -> Array { r with a; dead = Live }
            | None -> moved ())
        | Shards _ | Handle _ | Traced _ -> moved ()
      in
      Handle { donor; over }
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
          let as_taken (type v s d) (y : (v, s, d) t) : (v, s, d) t =
            match y with
            | Array r when A.buffer r.a == b ->
                Array { r with a = A.v (A.dtype r.a) (A.layout r.a) b' }
            | Array _ | Shards _ | Handle _ | Traced _ -> y
          in
          let op = Prim.map as_taken op in
          ending ~by c (Some into) (fun () -> routed ~by ~into r op)
      | None -> ending ~by c None (fun () -> routed ~by r op))

and plain : type r. by:string -> r prim -> r =
 fun ~by op ->
  match op with
  | Place (p, x) -> placed ~by p x
  | Check { ok; data; fail } ->
      ignore (Prim.layouts ~by op);
      Check.issue ~by ok data fail
  | _ -> routed ~by (Prim.route ~by op) op

(* [op], a placement, a check, or another operation, computed as its rule
   decided [r] ({!Prim.route}): by partials or by windows as the route
   computes, its first result into [into] where it is given. *)
and routed : type r. by:string -> ?into:A.any -> Prim.routed -> r prim -> r =
 fun ~by ?into r op ->
  match op with
  | Place (p, x) -> placed ~by p x
  | Check { ok; data; fail } -> Check.issue ~by ok data fail
  | _ -> (
      (* Every operation but a placement and a check has a route. *)
      let route = Option.get r.Prim.route in
      match route.compute with
      | Partials -> Partial.run ~by route op
      | Windows -> Exec.windows ~by ?into route r.layouts op)

let is_traced (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced _ -> true
  | Array _ | Shards _ | Handle _ -> false

(* Whether [i] receives [o]'s values as its own: [o] is [i] or what [i]
   adopts. *)
let rec adopts i o =
  o == i
  || match i.reach with Adopting p -> adopts p o | Values | Extent -> false

let owns i (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> adopts i owner
  | Array _ | Shards _ | Handle _ -> false

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
      | Traced _ | Array _ | Shards _ | Handle _ -> ())
    op;
  Prim.iteri (fun _ x -> Prim.alive ~by x) op;
  let op =
    if Prim.exists Donate.is_handle op then Prim.map Donate.lent op else op
  in
  i.rule i ~by op

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
      | Traced _ | Array _ | Shards _ | Handle _ -> ())
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
  let i = { name; reach; rule; live } in
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
  | Traced _ | Array _ | Shards _ | Handle _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> Some owner
  | Array _ | Shards _ | Handle _ -> None

(* An operation no handler answered computes, unless an operand is traced. *)
let unreached ~by op =
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ } -> refuse ~by owner
      | Array _ | Shards _ | Handle _ -> ())
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
  | Array _ | Shards _ | Handle _ ->
      if quiet () then fast1 ~by k dt x else eval1 ~by k dt x

let apply2 ~by k dt x y =
  match (x, y) with
  | Traced _, _ | _, Traced _ -> eval2 ~by k dt x y
  | _ -> if quiet () then fast2 ~by k dt x y else eval2 ~by k dt x y

let apply3 ~by k c x y =
  match (c, x, y) with
  | Traced _, _, _ | _, Traced _, _ | _, _, Traced _ -> eval3 ~by k c x y
  | _ -> if quiet () then fast3 ~by k c x y else eval3 ~by k c x y

(* The map of [prog] over [x], into [dt]. *)
let map_of prog dt x =
  let layout = A.Layout.contiguous (Iarray.of_array (Prim.shape x)) in
  Map { layout; prog; outs = [ dt ]; loads = [| Plain x |]; at = None }

let apply ~by prog dt x =
  match (x, Exec.live_kernels x) with
  | Array { at; a; _ }, Some kernels when quiet () ->
      Exec.map ~by kernels at prog dt a [| Nx_kernel.Plain a |]
  | _ ->
      let v, () = eval ~by (map_of prog dt x) in
      v

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
  | Array _ | Shards _ | Handle _ | Traced _ ->
      eval ~by (Place (p, x))
