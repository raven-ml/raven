(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout

(* Arrays kept untyped, one per device, at [dt]. *)
let typed dt (per : A.any array) =
  Iarray.init (Array.length per) (fun j -> A.expect dt per.(j))

(* Whether [p] cuts no axis: one device, or the whole on each. *)
let uncut p = not (Grid.is_cut (Devices.grid p))

(* The next node's id. *)
let nodes = Atomic.make 0

let defer ~by op =
  let layouts = Prim.layouts ~by op in
  let id = Atomic.fetch_and_add nodes 1 in
  let node = Value.Node { id; by; op; layouts } in
  (* Its operands are of every set, so its results are too. *)
  Prim.results_at None layouts
    (fun k form -> Value.Deferred { form; node; k })
    op

(* The route of the constant operation [op] whose results are wanted at [p]:
   where it computes them and reads each operand ({!Route.at}). Where [p] cuts
   no axis, or the rule is elementwise, that is [p] for all. *)
let route_at : type r.
    by:string -> r Value.prim -> unit Devices.placement -> unit Route.t =
 fun ~by op p ->
  let shapes = ref [] in
  Prim.iteri (fun _ x -> shapes := Prim.shape x :: !shapes) op;
  let shapes = Array.of_list (List.rev !shapes) in
  let at_p () =
    let operands = Array.make (Array.length shapes) p in
    { Route.operands; result = p; compute = Windows }
  in
  if uncut p then at_p ()
  else
    match Prim.rule op with
    | Route.Elementwise -> at_p ()
    | rule -> Route.at ~by rule shapes p

(* The constant operands of [node] at [p], each with where it reads them. *)
let reads (Value.Node n) p =
  let r = route_at ~by:n.by n.op p in
  let found = ref [] in
  Prim.iteri
    (fun i x ->
      match x with
      | Value.Deferred { node; _ } -> found := (node, r.operands.(i)) :: !found
      | Value.Array _ | Value.Shards _ | Value.Handle _ | Value.Traced _ -> ())
    n.op;
  List.rev !found

(* [x] at [q], a constant read from its computed results, [result]. *)
let resolved : type v s d.
    (Value.node -> unit Devices.placement -> A.any array array) ->
    d Devices.placement ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t =
 fun result q x ->
  match x with
  | Value.Deferred { form; node; k } ->
      let arrays = result node q in
      Prim.of_arrays q
        ~shape:(Iarray.to_array (L.shape form.layout))
        (typed form.dtype arrays.(k))
  | Value.Array _ | Value.Shards _ | Value.Handle _ | Value.Traced _ -> x

(* [node]'s results at [p], its constant operands already computed where it
   reads them, as [result] gives them. Its rule held when it was made. It
   computes at its route's result, [p] with the axes its rule keeps whole made
   whole, over operands where the route reads them; a creation computes there
   too. *)
let compute_node result (Value.Node n) p =
  let by = n.by in
  let route = route_at ~by n.op p in
  let q = route.result in
  let op =
    Prim.mapi
      (fun i y -> resolved result route.operands.(i) y)
      n.op
  in
  let r =
    match route.compute with
    | Partials -> Partial.run ~by route op
    | Windows -> Exec.windows ~by route n.layouts op
  in
  if Devices.equal q p then Prim.result_arrays op r
  else
    (* Each device of [p] keeps its window, cut from its window at [q], in
       memory of its own, so that it holds no more. *)
    let per = ref [] in
    Prim.iter_results
      (fun y ->
        let y = Place.value ~by p y in
        per := Prim.any (Iarray.map A.copy (Prim.arrays y)) :: !per)
      op r;
    Array.of_list (List.rev !per)

let at : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x ->
  match x with
  | Value.Array _ | Value.Shards _ | Value.Handle _ | Value.Traced _ -> x
  | Value.Deferred { form; node; k } ->
      let arrays =
        Prim.evaluate ~same:Devices.equal ~reads compute_node node p
      in
      Prim.of_arrays p
        ~shape:(Iarray.to_array (L.shape form.layout))
        (typed form.dtype arrays.(k))
