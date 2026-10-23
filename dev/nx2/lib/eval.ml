(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

(* The live [Extent] interpretations of the process. *)
let extents = Atomic.make 0
let quiet () = Atomic.get extents = 0

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
          | Array _ | Shards _ | Donated _ | Traced _ -> ())
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
      { order; batch; contracting; acc; out; a; b; init; a_scale; b_scale;
        segments } ->
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
           { order; batch; contracting; acc; out; a; b; init; a_scale; b_scale;
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
  | Linalg (Svd { vectors = No_vectors; a }) ->
      let s, () =
        applied i ~by (Linalg (Svd { vectors = No_vectors; a = sub a }))
      in
      [ Any s ]
  | Linalg (Svd { vectors = Vectors f; a }) ->
      let s, (u, vh) =
        applied i ~by (Linalg (Svd { vectors = Vectors f; a = sub a }))
      in
      [ Any s; Any u; Any vh ]
  | Linalg (Eigh { vectors = No_vectors; a }) ->
      let w, () =
        applied i ~by (Linalg (Eigh { vectors = No_vectors; a = sub a }))
      in
      [ Any w ]
  | Linalg (Eigh { vectors = Vectors (); a }) ->
      two (Linalg (Eigh { vectors = Vectors (); a = sub a }))
  | Linalg (Eig { vectors = No_vectors; dtype; a }) ->
      let w, () =
        applied i ~by
          (Linalg (Eig { vectors = No_vectors; dtype; a = sub a }))
      in
      [ Any w ]
  | Linalg (Eig { vectors = Vectors (); dtype; a }) ->
      two (Linalg (Eig { vectors = Vectors (); dtype; a = sub a }))
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
   dependency order, each once, with a stack of its own: a chain of any length
   takes no stack depth. A node of a formula [i]'s rule returned is read as it
   is. *)
let replay (type v s d) i (x : (v, s, d) t) : (v, s, d) t =
  let results : (int, d any array) Hashtbl.t = Hashtbl.create 8 in
  let known id = is_formula i id || Hashtbl.mem results id in
  let sub : type w q e. (w, q, e) t -> (w, q, d) t = function
    | Deferred { form; node = Node n as node; k } ->
        if is_formula i n.id then
          Deferred { form = { form with placement = None }; node; k }
        else Prim.expect form.dtype (Hashtbl.find results n.id).(k)
    | Array _ | Shards _ | Donated _ | Traced _ ->
        invalid_arg "Eval: a formula's operand is a formula"
  in
  let missing (Node n) =
    let (Prim.Operands xs) = Prim.operands n.op in
    List.filter_map
      (fun (Any y) ->
        match y with
        | Deferred { node = Node m as node; _ } when not (known m.id) ->
            Some node
        | Deferred _ | Array _ | Shards _ | Donated _ | Traced _ -> None)
      xs
  in
  let stack = Stack.create () in
  (match x with
  | Deferred { node; _ } -> Stack.push node stack
  | Array _ | Shards _ | Donated _ | Traced _ -> ());
  while not (Stack.is_empty stack) do
    let (Node n as top) = Stack.top stack in
    if known n.id then ignore (Stack.pop stack)
    else
      match missing top with
      | [] ->
          ignore (Stack.pop stack);
          let rs = replayed i ~by:n.by sub n.op in
          Hashtbl.replace results n.id (Array.of_list rs)
      | deps -> List.iter (fun d -> Stack.push d stack) deps
  done;
  sub x

(* Whether [i]'s rule receives [x] changed: for an [Extent], a formula its
   rule did not return replayed. *)
let changes (type v s d) i (x : (v, s, d) t) =
  match x with
  | Deferred { node = Node n; _ } -> (
      match i.reach with
      | Extent -> not (is_formula i n.id)
      | Values | Adopting _ -> false)
  | Array _ | Shards _ | Donated _ | Traced _ -> false

let received (type v s d) i (x : (v, s, d) t) : (v, s, d) t =
  match x with
  | Deferred _ when changes i x -> replay i x
  | Deferred _ | Array _ | Shards _ | Donated _ | Traced _ -> x

let is_traced (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced _ -> true
  | Array _ | Shards _ | Donated _ | Deferred _ -> false

(* Whether [i] receives [o]'s values as its own: [o] is [i] or what [i]
   adopts. *)
let rec adopts i o =
  o == i
  || match i.reach with Adopting p -> adopts p o | Values | Extent -> false

let owns i (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> adopts i owner
  | Array _ | Shards _ | Donated _ | Deferred _ -> false

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
   recorded. [Around i] is answered by [i]'s handler alone. *)
type _ Effect.t +=
  | Receive : { by : string; mutable op : 'r prim option } -> 'r Effect.t
  | Around : interpretation -> unit Effect.t

(* Whether [o]'s handler is around the caller. *)
let around o =
  match Effect.perform (Around o) with
  | () -> true
  | exception Effect.Unhandled (Around _) -> false

(* [i]'s rule on [op], in [i]'s clause. Another interpretation's traced operand
   is refused unless its interpretation is around [i], which is asked once per
   interpretation; one [i] adopts is around by construction ({!interpret}). An
   operation the rule receives consumes its handles, a movement too: the rule's
   results hold none to pass on. The rule reads it as it was called
   ({!Donate.consume}). *)
let deliver i ~by op =
  let checked = ref ([] : interpretation list) in
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ }
        when (not (adopts i owner)) && not (List.memq owner !checked) ->
          if not (around owner) then refuse ~by owner;
          checked := List.cons owner !checked
      | Traced _ | Array _ | Shards _ | Donated _ | Deferred _ -> ())
    op;
  let op = Donate.consume ~by op in
  let op =
    if Prim.exists (fun x -> changes i x) op then
      Prim.map (fun x -> received i x) op
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
      | Traced _ | Array _ | Shards _ | Donated _ | Deferred _ -> ())
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
        | Around j when j == i ->
            Some
              (fun (k : (a, _) Effect.Deep.continuation) ->
                if present i here then Effect.Deep.continue k ()
                else Effect.Deep.discontinue k (Effect.Unhandled e))
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

let traced i form payload = Traced { form; owner = i; payload }

let payload (type v s d) i (x : (v, s, d) t) =
  match x with
  | Traced { owner; payload; _ } when owner == i -> Some payload
  | Traced _ | Array _ | Shards _ | Donated _ | Deferred _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> Some owner
  | Array _ | Shards _ | Donated _ | Deferred _ -> None

(* An operation no handler answered computes, unless an operand is traced. *)
let unreached ~by op =
  Prim.iteri
    (fun _ x ->
      match x with
      | Traced { owner; _ } -> refuse ~by owner
      | Array _ | Shards _ | Donated _ | Deferred _ -> ())
    op;
  Exec.run ~by op

let eval ~by op =
  if quiet () && not (Prim.exists is_traced op) then Exec.run ~by op
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
   performs nothing, checked on the operands before the operation is built. *)
let apply1 ~by k dt x =
  match x with
  | Traced _ -> eval1 ~by k dt x
  | Array _ | Shards _ | Donated _ | Deferred _ ->
      if quiet () then Exec.apply1 ~by k dt x else eval1 ~by k dt x

let apply2 ~by k dt x y =
  match (x, y) with
  | Traced _, _ | _, Traced _ -> eval2 ~by k dt x y
  | _ ->
      if quiet () then Exec.apply2 ~by k dt x y
      else eval2 ~by k dt x y

let apply3 ~by k c x y =
  match (c, x, y) with
  | Traced _, _, _ | _, Traced _, _ | _, _, Traced _ -> eval3 ~by k c x y
  | _ ->
      if quiet () then Exec.apply3 ~by k c x y
      else eval3 ~by k c x y

let contract ~by ~batch ~contracting ~acc dt a b init =
  let order = Prim.contract_order ~batch ~contracting a b in
  let fast =
    if quiet () then
      Exec.contract ~by ~order ~batch ~contracting ~acc dt a b init
    else None
  in
  match fast with
  | Some y -> y
  | None ->
      eval ~by
        (Contract
           {
             order;
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
  | Array r
    when Devices.rebrand r.at == p
         && r.dead == Live
         && quiet () ->
      (* A second value over the memory shares it. *)
      Rig.Claim.share (Nx_array.buffer r.a);
      Array { at = p; a = r.a; dead = Live }
  | Array _ | Shards _ | Donated _ | Deferred _ | Traced _ ->
      eval ~by (Place (p, x))

let expand i ~by
    ~(repeat :
       'd.
       int -> ('d index -> 'd any list -> 'd any list) -> 'd any list ->
       'd any list) op =
  if not (Atomic.get i.live) then
    Format.kasprintf invalid_arg "%s: an expansion for %s after it returned" by
      i.name;
  Effect.Deep.try_with
    (fun () -> Expand.run eval ~by ~repeat op)
    () (handler i)
