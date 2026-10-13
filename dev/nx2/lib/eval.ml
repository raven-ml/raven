(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value

(* [i]'s rule on [op]. The formulas it returns are [i]'s own: an operation
   reads them as they are. *)
let applied i ~by op =
  let r = Interp.apply i ~by op in
  (match i.reach with
  | Values -> ()
  | Extent ->
      Prim.iter_results
        (fun y ->
          match y with
          | Deferred { node = Node n; _ } -> Interp.add_formula i n.id
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
      let op = Map { layout; prog; outs; loads = Array.map load loads } in
      Prim.iter_outs add outs (applied i ~by op);
      List.rev !all
  | Reduce { layout; axes; prog; reductions; loads } ->
      let (Reductions reductions) = reductions_at reductions in
      let loads = Array.map load loads in
      let op = Reduce { layout; axes; prog; reductions; loads } in
      Prim.iter_reductions add reductions (applied i ~by op);
      List.rev !all
  | Scan { layout; axis; prog; reduction; loads } ->
      let (Reduction reduction) = reduction_at reduction in
      let loads = Array.map load loads in
      let op = Scan { layout; axis; prog; reduction; loads } in
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
  | Contract { batch; contracting; acc; out; a; b; init } ->
      let a = sub a and b = sub b and init = Option.map sub init in
      one (Contract { batch; contracting; acc; out; a; b; init })
  | Fft (C2c { direction; axes; x }) ->
      one (Fft (C2c { direction; axes; x = sub x }))
  | Fft (R2c { dtype; axes; x }) -> one (Fft (R2c { dtype; axes; x = sub x }))
  | Fft (C2r { dtype; n; axes; x }) ->
      one (Fft (C2r { dtype; n; axes; x = sub x }))
  | Linalg (Cholesky { triangle; a }) ->
      one (Linalg (Cholesky { triangle; a = sub a }))
  | Linalg (Lu a) -> three (Linalg (Lu (sub a)))
  | Linalg (Qr { factors; a }) -> two (Linalg (Qr { factors; a = sub a }))
  | Linalg (Svd { factors; a }) -> three (Linalg (Svd { factors; a = sub a }))
  | Linalg (Svd_values a) -> one (Linalg (Svd_values (sub a)))
  | Linalg (Eigh a) -> two (Linalg (Eigh (sub a)))
  | Linalg (Eigh_values a) -> one (Linalg (Eigh_values (sub a)))
  | Linalg (Eig a) -> two (Linalg (Eig (sub a)))
  | Linalg (Eig_values a) -> one (Linalg (Eig_values (sub a)))
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
  let known id = Interp.is_formula i id || Hashtbl.mem results id in
  let sub : type w q e. (w, q, e) t -> (w, q, d) t = function
    | Deferred { form; node = Node n as node; k } ->
        if Interp.is_formula i n.id then
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

(* Whether [i]'s rule receives [x] changed: a handle as its arrays, and for an
   [Extent], a formula its rule did not return replayed. *)
let changes (type v s d) i (x : (v, s, d) t) =
  match x with
  | Donated _ -> true
  | Deferred { node = Node n; _ } -> (
      match i.reach with
      | Extent -> not (Interp.is_formula i n.id)
      | Values -> false)
  | Array _ | Shards _ | Traced _ -> false

let received (type v s d) i (x : (v, s, d) t) : (v, s, d) t =
  match x with
  | Donated _ -> Exec.live x
  | Deferred _ when changes i x -> replay i x
  | Deferred _ | Array _ | Shards _ | Traced _ -> x

let eval ~by op =
  match Interp.receiver ~by op with
  | None -> Exec.run ~by op
  | Some i ->
      let (Prim.Operands xs) = Prim.operands op in
      List.iteri (fun k (Any x) -> Prim.alive ~by k x) xs;
      let op =
        if Prim.exists (fun x -> changes i x) op then
          Prim.map (fun x -> received i x) op
        else op
      in
      applied i ~by op

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

(* No traced operand and no live [Extent] on the domain is the condition under
   which {!Interp.receiver} gives [None], checked on the operands before the
   operation is built. *)
let apply1 ~by k dt x =
  match x with
  | Traced _ -> eval1 ~by k dt x
  | Array _ | Shards _ | Donated _ | Deferred _ ->
      if Interp.quiet () then Exec.apply1 ~by k dt x else eval1 ~by k dt x

let apply2 ~by k dt x y =
  match (x, y) with
  | Traced _, _ | _, Traced _ -> eval2 ~by k dt x y
  | _ ->
      if Interp.quiet () then Exec.apply2 ~by k dt x y
      else eval2 ~by k dt x y

let apply3 ~by k c x y =
  match (c, x, y) with
  | Traced _, _, _ | _, Traced _, _ | _, _, Traced _ -> eval3 ~by k c x y
  | _ ->
      if Interp.quiet () then Exec.apply3 ~by k c x y
      else eval3 ~by k c x y

let contract ~by ~batch ~contracting ~acc dt a b init =
  let fast =
    if Interp.quiet () then
      Exec.contract ~by ~batch ~contracting ~acc dt a b init
    else None
  in
  match fast with
  | Some y -> y
  | None ->
      eval ~by (Contract { batch; contracting; acc; out = dt; a; b; init })

let place ~by p x =
  match x with
  | Array r
    when Devices.rebrand r.at == p
         && r.dead == Live
         && Interp.quiet () ->
      (* A second value over the memory shares it. *)
      Rig.Claim.share (Nx_array.buffer r.a);
      Array { at = p; a = r.a; dead = Live }
  | Array _ | Shards _ | Donated _ | Deferred _ | Traced _ ->
      eval ~by (Place (p, x))

let expand i ~by op = Interp.expanding i (fun () -> Expand.run eval ~by op)
