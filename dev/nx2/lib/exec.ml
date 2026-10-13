(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Messages *)

(* The move a program makes where a device's kernels decline an operation:
   nx moves no data between devices on its own. *)
let move = "place its operands with Nx.place on a set whose kernels compute it"

let refuses ~by ~kernels kind d dts =
  invalid_argf "%s: %s does not compute %s on %s (%s); %s" by kernels kind
    (String.concat ", " (List.map (fun (A.Any a) -> D.name (A.dtype a)) dts))
    (Rig.name d) move

(* As [refuses], for an optional case whose expansion nx does not have yet. *)
let unexpanded ~by ~kernels kind d dts =
  invalid_argf
    "%s: %s does not compute %s on %s (%s), and its expansion is not \
     available yet; %s"
    by kernels kind
    (String.concat ", " (List.map (fun (A.Any a) -> D.name (A.dtype a)) dts))
    (Rig.name d) move

let declined ~by ~kernels node d dts =
  refuses ~by ~kernels (Prim.kind node) d dts

let kernels_of ~by ~op set : (module Nx_kernel.S) =
  match Devices.kernels set with
  | Some k -> k
  | None ->
      invalid_argf
        "%s: %a has no kernels to compute %s; place the value on a set with \
         kernels, mint the set with kernels, or apply the function where a \
         compiler stages it"
        by Devices.pp set op

(* A kernel's answer: [Done], a refusal raised, or [Declined] passed on. *)
let done_or_declined ~by answer operands =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal -> A.refused by refusal operands

(* As [done_or_declined] over [dsts] then [ops], listing them only to raise. *)
let ran ~by answer dsts ops =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal -> A.refused by refusal (Array.to_list (Array.append dsts ops))

(* The array a kernel's load reads. *)
let load_array : Nx_kernel.load -> A.any = function
  | Nx_kernel.Plain a -> A.Any a
  | Nx_kernel.Padded { x; _ } -> A.Any x

(* As [ran], for a loop over [loads]. *)
let ran_loads ~by answer dsts loads =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal ->
      A.refused by refusal
        (Array.to_list dsts @ List.map load_array (Array.to_list loads))

(* As [ran], [None] where done and the operands [ops] where declined. *)
let outcome ~by answer dsts ops =
  if ran ~by answer dsts ops then None else Some ops

(* Values from arrays *)

let make = Prim.of_arrays

(* Arrays kept untyped, one per device, at [dt]. *)
let typed dt (per : A.any array) =
  Iarray.init (Array.length per) (fun j -> A.expect dt per.(j))

let any arrays =
  Array.init (Iarray.length arrays) (fun j -> A.Any (Iarray.get arrays j))

(* Raises unless operand [i], [x], is alive and concrete or a constant: an
   interpretation receives every operation on its traced values before eager
   execution can. *)
let checked (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s reached eager execution" by
        owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _ ->
      Prim.alive ~by i x

(* Donation *)

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Prim.alive ~by 0 x;
  let handle at arrays donor =
    Value.Donated { at; arrays; donor; spent = Value.Live }
  in
  match x with
  | Value.Array r ->
      handle r.at (Iarray.of_list [ r.a ]) [%atomic.loc r.dead]
  | Value.Shards r -> handle r.at r.arrays [%atomic.loc r.dead]
  | Value.Donated _ | Value.Deferred _ | Value.Traced _ -> x

(* A handle's arrays as a live value, which the operation reads. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated { at; arrays; _ } -> Prim.of_arrays at arrays
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

(* A donated operand and its index among the operation's operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

let is_handle (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated _ -> true
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

let is_concrete x = not (Prim.is_constant x)

(* The donated operands among [xs]. Two handles over one memory raise naming
   [Nx.donate]: the same handle twice, two donations of one value, or one
   handle and a view a movement passed on. *)
let handles ~by xs =
  let hs =
    List.filter (fun (Handle (_, x)) -> is_handle x)
      (List.mapi (fun i (Value.Any x) -> Handle (i, x)) xs)
  in
  let rec check = function
    | [] -> ()
    | Handle (_, x) :: rest ->
        List.iter
          (fun (Handle (_, y)) ->
            let same =
              match (x, y) with
              | Value.Donated a, Value.Donated b ->
                  Iarray.exists
                       (fun p ->
                         Iarray.exists
                           (fun q ->
                             Rig.Buffer.overlaps (A.buffer p) (A.buffer q))
                           b.arrays)
                       a.arrays
              | _ -> false
            in
            if same then
              invalid_argf "Nx.donate: %s reads one donation twice" by)
          rest;
        check rest
  in
  check hs;
  hs

(* The buffers of a handle, one per device. *)
let buffers (Handle (_, x)) = Iarray.to_list (Iarray.map A.buffer (Prim.arrays x))

(* Sets the death word [w] of operand [i] to [died] for [by], where a first claim
   from [Live] failed: a word another operation took, on any domain, raises
   naming it. No death is undone, so a word read [Live] is only retried. *)
let rec contested ~by i (w : Value.death Atomic.Loc.t) died =
  match Atomic.Loc.get w with
  | Value.Died earlier ->
      invalid_argf "%s: operand %d was donated to %s" by (i + 1) earlier
  | Value.Live ->
      if not (Atomic.Loc.compare_and_set w Value.Live died) then
        contested ~by i w died

(* Marks the handle [x], operand [i], read by the operation [died] names: one
   operation reads a handle. *)
let spend (type v s d) ~by i died (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d ->
      if not (Atomic.Loc.compare_and_set [%atomic.loc d.spent] Value.Live died)
      then contested ~by i [%atomic.loc d.spent] died
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

(* Claims the handles [hs] for their consumer [by], before it reads them: each
   handle is spent, then its donor dies naming [by]. A donor that died first,
   on any domain, raises naming its consumer: the donor's one word orders the
   consumers of all its handles. *)
let claim_handle (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d ->
      let died = Value.Died by in
      spend ~by i died x;
      if not (Atomic.Loc.compare_and_set d.donor Value.Live died) then
        contested ~by i d.donor died
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

let claim ~by hs = List.iter (fun (Handle (i, x)) -> claim_handle ~by i x) hs

(* [b]'s memory consumed for [by], where rig holds it exclusive: no other value
   reads it. The live buffer over it, if so. *)
let take ~by b = Rig.Claim.take ~why:("donated to " ^ by) b

(* Ends the claimed handles [hs] once [by] read them: memory no other value
   reads, and that [by] did not write its result into ([reused]), is
   consumed. *)
let consumed ~by ~reused hs =
  List.iter
    (fun (Handle (_, x) as h) ->
      match x with
      | Value.Donated _ ->
          let bufs = buffers h in
          let why = "donated to " ^ by in
          if not (List.exists (fun b -> List.memq b reused) bufs) then
            Rig.Claim.with_ ~read:[] ~donate:bufs (fun c ->
                if List.for_all (Rig.Claim.exclusive c) bufs then
                  List.iter (fun b -> ignore (Rig.Claim.consume c ~why b)) bufs)
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ())
    hs

(* The load of a map whose memory takes the map's first result, if one may: a
   handle on one device, of the result's dtype, C-contiguous with the map's
   layout over all of its memory, beside loads at its placement, and whose
   memory no other value reads (rig holds it exclusive). Its memory is then
   consumed, naming [by], and its index, buffer and new array are the
   answer. *)
let reuse (type d r) ~by layout (outs : (d, r) Value.outs)
    (loads : d Value.load array) =
  match outs with
  | [] -> None
  | dt :: _ -> (
      let fits = function
        | Value.Padded _ -> false
        | Value.Plain (Value.Donated { at; arrays; _ })
          when Iarray.length arrays = 1 ->
            let a = Iarray.get arrays 0 in
            D.equal (A.dtype a) dt
            && L.equal (A.layout a) layout
            && Rig.Buffer.spans (A.buffer a)
            && Array.for_all
                 (fun l ->
                   let (Value.Any z) = Prim.load_any l in
                   match Prim.at z with
                   | Some q -> Devices.rebrand q == at
                   | None -> true)
                 loads
        | Value.Plain
            ( Value.Array _ | Value.Shards _ | Value.Donated _
            | Value.Deferred _ | Value.Traced _ ) ->
            false
      in
      match Array.find_index fits loads with
      | None -> None
      | Some i -> (
          match loads.(i) with
          | Value.Plain (Value.Donated d) ->
              let a = Iarray.get d.arrays 0 in
              let b = A.buffer a in
              Option.map
                (fun b' ->
                  (i, b, A.Any (A.v (A.dtype a) (A.layout a) b')))
                (Rig.Claim.take ~why:("donated to " ^ by) b)
          | Value.Plain _ | Value.Padded _ -> None))

(* A view made over [a]'s memory, outside any handle chain: the memory is shared
   from then on, so that no donation of [a]'s value writes what the view
   reads. *)
let view_of a = Rig.Claim.share (A.buffer a)

(* The whole of an axis of extent [d]. *)
let whole d = { M.start = 0; count = d; step = 1 }
let extents (w : M.range array) = Array.map (fun (r : M.range) -> r.count) w
let starts (w : M.range array) = Array.map (fun (r : M.range) -> r.start) w

(* Fresh C-contiguous arrays of a value of [shape] at [p]: each device's
   window. *)
let fresh ~by p dt shape =
  let set = Devices.set p in
  match Grid.one (Devices.grid p) with
  | Some k -> Iarray.of_list [ A.create (Devices.rig set k) dt shape ]
  | None ->
      let devices = Grid.devices (Devices.grid p) in
      Iarray.init (Array.length devices) (fun j ->
          A.create (Devices.rig set devices.(j)) dt
            (extents (Devices.window ~by p shape j)))

(* A maker of fresh results at [at], or at their form's placement, recording the
   placement in [where]. *)
let alloc (type v s d) ~by ?(at : unit Devices.placement option) ?where ?into k
    (f : (v, s, d) Value.form) : (v, s, d) Value.t =
  let p =
    match (at, f.placement) with
    | Some p, _ -> Devices.rebrand p
    | None, Some p -> p
    | None, None -> invalid_arg "Exec.alloc: a value of every set"
  in
  Option.iter (fun r -> r := Some (Devices.rebrand p)) where;
  match into with
  | Some a when k = 0 ->
      Value.Array { at = p; a = A.expect f.dtype a; dead = Value.Live }
  | Some _ | None -> make p (fresh ~by p f.dtype (L.shape f.layout))

(* Programs on one device *)

(* Whether [nd] is a [Coord] of an axis whose window starts at [first > 0]. *)
let shifted first (nd : P.node) =
  match nd with Coord c -> first.(c) <> 0 | _ -> false

(* [prog] with each [Coord] of an axis whose window starts at [s > 0] replaced
   by [Coord + s], so that a device computes its window. Its operands' nodes
   come first, and each [In] reads its operand's. *)
let with_offsets (prog : P.t) (first : int array) =
  if
    Array.for_all (fun s -> s = 0) first
    || not (Iarray.exists (shifted first) prog.nodes)
  then prog
  else begin
    let open Build in
    let b = make () in
    let operand k = input b (Iarray.get prog.ins k) in
    let operands = Array.init (Iarray.length prog.ins) operand in
    let index = Array.make (Iarray.length prog.nodes) 0 in
    Iarray.iteri
      (fun i nd ->
        let at j = index.(j) in
        index.(i) <-
          (match (nd : P.node) with
          | In k -> operands.(k)
          | Coord c when first.(c) <> 0 ->
              let coord = push b nd in
              let start = const b D.Int64 (Int64.of_int first.(c)) in
              push b (Op2 (Binary Add, coord, start))
          | Op1 (k, dt, j) -> push b (Op1 (k, dt, at j))
          | Op2 (k, j, l) -> push b (Op2 (k, at j, at l))
          | Op3 (k, j, l, m) -> push b (Op3 (k, at j, at l, at m))
          | Coord _ | Const _ -> push b nd))
      prog.nodes;
    program b (Iarray.map (fun o -> index.(o)) prog.outs)
  end

(* [node]'s kernel into [dst], over [ops]. *)
let apply_node (module K : Nx_kernel.S) node (A.Any dst) (ops : A.any array) =
  match (node : P.node) with
  | Const (_, bits) -> K.apply0 (Fill bits) ~dst
  | Coord c -> K.apply0 (Iota c) ~dst
  | Op1 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply1 k ~dst x
  | Op2 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply2 k ~dst x (A.expect (A.dtype x) ops.(1))
  | Op3 (k, _, _, _) ->
      (* Where and Fma give their second and third operands' dtype. *)
      let (A.Any c) = ops.(0) in
      let dt = A.dtype dst in
      K.apply3 k ~dst c (A.expect dt ops.(1)) (A.expect dt ops.(2))
  | In _ -> Declined

(* [prog] over [ops] into [dsts] on device [d], whose window starts at [first]:
   its kernel where it is one node, else the kernels' map. [false] where the
   kernels decline the map; a one-node program they decline raises, since its
   node is core. *)
let map_on ~by (module K : Nx_kernel.S) d prog first ops dsts =
  let prog = with_offsets prog first in
  match Expand.single prog with
  | Some node ->
      let arrays = Array.append dsts ops in
      ran ~by (apply_node (module K) node dsts.(0) ops) dsts ops
      ||
      (* A decline at base dtypes is the kernels' error; at others the node
         expands. *)
      if Array.for_all (fun (A.Any a) -> Expand.base (D.Any (A.dtype a))) arrays
      then declined ~by ~kernels:K.name node d (Array.to_list arrays)
      else false
  | None ->
      let loads = Array.map (fun (A.Any a) -> Nx_kernel.Plain a) ops in
      ran ~by (K.map prog ~dsts loads) dsts ops

(* A load's operand on device [k] for its window [w] of the loop: a padded
   load's whole operand, which its route reads on each device. *)
let load_view (type d) ~by k w (l : d Value.load) =
  match l with
  | Value.Plain x -> A.Any (Place.view ~by x k w)
  | Value.Padded { x; _ } ->
      A.Any (Place.view ~by x k (Array.map whole (Prim.shape x)))

(* As [load_view], the load a kernel reads. *)
let kernel_load (type d) ~by k w (l : d Value.load) : Nx_kernel.load =
  match l with
  | Value.Plain x -> Nx_kernel.Plain (Place.view ~by x k w)
  | Value.Padded { x; fill; pad } ->
      let x = Place.view ~by x k (Array.map whole (Prim.shape x)) in
      Nx_kernel.Padded { x; fill; pad }

(* [K]'s copy of [a] into [dst]; where it declines [a]'s dtype, its copy of
   their bits in the dtype {!Expand.kept} gives, where that has their width. *)
let copy_into (type v s) (module K : Nx_kernel.S) ~(dst : (v, s) A.t)
    (a : (v, s) A.t) =
  match K.apply1 Copy ~dst a with
  | A.Declined ->
      let dt = A.dtype a in
      let (D.Any u) = Expand.kept (D.Any dt) in
      if D.code u = D.code dt || D.bits u <> D.bits dt then A.Declined
      else
        (* One width keeps the layout, so both bitcasts are views. *)
        K.apply1 Copy
          ~dst:(Option.get (A.bitcast u dst))
          (Option.get (A.bitcast u a))
  | answer -> answer

(* [a] stored afresh, C-contiguous, on its device. *)
let copy_array ~by kernels a =
  let (module K : Nx_kernel.S) = kernels in
  let dst = A.create (A.device a) (A.dtype a) (L.shape (A.layout a)) in
  if
    not (done_or_declined ~by (copy_into kernels ~dst a) [ A.Any dst; A.Any a ])
  then
    declined ~by ~kernels:K.name
      (Op1 (Copy, D.Any (A.dtype a), 0))
      (A.device a) [ A.Any a ];
  dst

(* [a] moved by [mv]: a view over [a]'s memory where a layout expresses it, a
   copy moved otherwise. *)
let moved ~by set mv a =
  match A.move mv a with
  | Some v ->
      view_of a;
      v
  | None ->
      let kernels = kernels_of ~by ~op:"Copy" set in
      Option.get (A.move mv (copy_array ~by kernels a))

(* [a] bitcast to [dt], as {!moved}. *)
let cast ~by set dt a =
  match A.bitcast dt a with
  | Some v ->
      view_of a;
      v
  | None ->
      let kernels = kernels_of ~by ~op:"Copy" set in
      Option.get (A.bitcast dt (copy_array ~by kernels a))

(* [mv] on one shard of a value of [shape] cut as [cuts], the shard of shape
   [local]: a cut axis is one the movement keeps in order ({!Route.moved}), so
   its extents are divided where the movement names them. *)
let localize (mv : M.t) shape cuts local : M.t =
  let tiles a =
    Array.fold_left (fun n (c, t) -> if c = a then t else n) 1 cuts
  in
  let target s' =
    let s' = Array.copy s' in
    Array.iter
      (fun (a, t) ->
        Option.iter (fun a' -> s'.(a') <- s'.(a') / t) (Route.moved mv shape a))
      cuts;
    s'
  in
  match mv with
  | Permute _ | Window _ -> mv
  | Reshape s' -> Reshape (target s')
  | Broadcast s' -> Broadcast (target s')
  | Slice rs ->
      Slice
        (Array.mapi
           (fun a (r : M.range) ->
             if tiles a > 1 then { M.start = 0; count = local.(a); step = 1 }
             else r)
           rs)

let unravel i shape =
  let r = Array.length shape in
  let idx = Array.make r 0 and k = ref i in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod shape.(a);
    k := !k / shape.(a)
  done;
  idx

(* Whether [p] cuts no axis: one device, or the whole on each. *)
let uncut p = not (Grid.is_cut (Devices.grid p))

(* The constant operation at [p] reads its operands at this placement: [p] for a
   map, whose operands have its shape, and for a placement that cuts no axis;
   the whole on every device otherwise. *)
(* CR: Keep [p] only for maps without Prim.is_padded loads.
   Padding deferred ones[4] to [6], then placing it split over two
   devices, forces [2] operands although load_view needs the whole [4].
   Use the existing whole [q] and final crop for padded maps too,
   so their kernels receive whole operands and destinations. *)
let operand_at : type r.
    r Value.prim -> unit Devices.placement -> unit Devices.placement =
 fun op p ->
  match op with
  | Value.Map _ -> p
  | _ -> if uncut p then p else Devices.on (Devices.set p)

(* How a constant's results are read: from its memo, where they are kept, or
   computed into memory of their own. *)
type policy = Memo | Fresh

(* Whether [op] is a view of its operand: a constant's keeps no results of its
   own, and an operation reads it through to its operand's. *)
let is_view : type r. r Value.prim -> bool = function
  | Value.Move _ | Value.Bitcast _ -> true
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fft _ | Value.Linalg _
  | Value.Fold _ | Value.Assemble _ | Value.Contract _ | Value.Copy _
  | Value.Place _ | Value.Check _ ->
      false

let find memo p =
  List.find_map
    (fun (q, a) -> if Devices.equal q p then Some a else None)
    (Atomic.get memo)

(* Keeps [a] as [p]'s results unless another domain stored first. *)
let rec remember memo p a =
  let old = Atomic.get memo in
  match
    List.find_map
      (fun (q, _) -> if Devices.equal q p then Some () else None)
      old
  with
  | Some () -> ()
  | None ->
      if not (Atomic.compare_and_set memo old ((p, a) :: old)) then
        remember memo p a

(* The constant operands of [node] at [p] not yet computed where it reads
   them. *)
let pending (Value.Node n) p =
  let (Prim.Operands xs) = Prim.operands n.op in
  let q = operand_at n.op p in
  List.filter_map
    (fun (Value.Any x) ->
      match x with
      | Value.Deferred { node = Value.Node m as node; _ } ->
          if find m.memo q = None then Some (node, q) else None
      | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
          None)
    xs

(* The host's device, at the brand of a value this library reads there: a
   check's flag, a value read out ({!on_host}). No value at it reaches a
   function of nx's own. *)
let host () = Devices.rebrand (Devices.one Devices.host 0)

(* The next node's id. *)
let nodes = Atomic.make 0

(* The results of a map into arrays of the dtypes of [dsts]. *)
type 'd outs = Outs : ('d, 'r) Value.outs -> 'd outs

let rec outs_of : type d. A.any list -> d outs = function
  | [] -> Outs Value.[]
  | A.Any a :: rest ->
      let (Outs o) = outs_of rest in
      Outs Value.(A.dtype a :: o)

(* [f j k w] for each device [k] of [p], [j] its position, [w] its window of a
   value of [shape]. *)
let each_device ~by p shape f =
  match Grid.one (Devices.grid p) with
  | Some k -> f 0 k (Array.map whole shape)
  | None ->
      Array.iteri
        (fun j k -> f j k (Devices.window ~by p shape j))
        (Grid.devices (Devices.grid p))

(* The first device of [p] whose kernels decline, with the operands they were
   given: [f j k w] runs device [k]'s kernels, [j] its position and [w] its
   window of a value of [shape], and is [None] once done, [Some ops] where they
   declined over [ops]. No device runs after a decline. *)
let first_decline ~by p shape f =
  let declined = ref None in
  each_device ~by p shape (fun j k w ->
      if Option.is_none !declined then
        Option.iter
          (fun ops -> declined := Some (Devices.rig (Devices.set p) k, ops))
          (f j k w));
  !declined

(* The window [w] of an operand of [shape] read whole along [axis]. *)
let along axis shape w =
  let w = Array.copy w in
  w.(axis) <- whole shape.(axis);
  w

(* The window of an operand of [shape] whose leading axes are [w]'s first, read
   whole along the rest: a routine's matrices, below a batch window. *)
let batch w shape =
  let b = min (Array.length w) (Array.length shape - 2) in
  Array.mapi (fun a d -> if a < b then w.(a) else whole d) shape

(* The transform [f] on device [d] into the result [y]'s array at position
   [j], for its window [w], as {!first_decline}'s [f] answers. *)
let fft_on (type d r) ~by (f : (d, r) Value.fft) (module K : Nx_kernel.S)
    (y : r) j d w =
  let view x =
    let s = Prim.shape x in
    Place.view ~by x d
      (Array.fold_left (fun w a -> along a s w) w (Prim.fft_axes f))
  in
  match f with
  | Value.C2c { direction; axes; x } ->
      let dst = Iarray.get (Prim.arrays y) j and x = view x in
      outcome ~by (K.c2c direction ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.R2c { axes; x; _ } ->
      let dst = Iarray.get (Prim.arrays y) j and x = view x in
      outcome ~by (K.r2c ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.C2r { axes; x; _ } ->
      let dst = Iarray.get (Prim.arrays y) j and x = view x in
      outcome ~by (K.c2r ~axes ~dst x) [| A.Any dst |] [| A.Any x |]

(* The routine [l] on device [d] into the results [y]'s arrays at position
   [j], for its window [w] of the first, as {!first_decline}'s [f] answers. *)
let linalg_on (type d r) ~by (l : (d, r) Value.linalg)
    (module K : Nx_kernel.S) (y : r) j d w =
  let view x = Place.view ~by x d (batch w (Prim.shape x)) in
  let at x = Iarray.get (Prim.arrays x) j in
  match (l, y) with
  | Value.Cholesky { triangle; a }, y ->
      let dst = at y and a = view a in
      outcome ~by (K.cholesky triangle ~dst a) [| A.Any dst |] [| A.Any a |]
  | Value.Lu a, (lu, pivots, perm) ->
      let lu = at lu and pivots = at pivots and perm = at perm and a = view a in
      outcome ~by
        (K.lu ~lu ~pivots ~perm a)
        [| A.Any lu; A.Any pivots; A.Any perm |]
        [| A.Any a |]
  | Value.Qr { factors; a }, (q, r) ->
      let q = at q and r = at r and a = view a in
      outcome ~by (K.qr factors ~q ~r a) [| A.Any q; A.Any r |] [| A.Any a |]
  | Value.Svd { factors; a }, (u, s, vh) ->
      let u = at u and s = at s and vh = at vh and a = view a in
      outcome ~by
        (K.svd factors ~u ~s ~vh a)
        [| A.Any u; A.Any s; A.Any vh |]
        [| A.Any a |]
  | Value.Svd_values a, s ->
      let s = at s and a = view a in
      outcome ~by (K.svd_values ~s a) [| A.Any s |] [| A.Any a |]
  | Value.Eigh a, (w, v) ->
      let w = at w and v = at v and a = view a in
      outcome ~by (K.eigh ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
  | Value.Eigh_values a, w ->
      let w = at w and a = view a in
      outcome ~by (K.eigh_values ~w a) [| A.Any w |] [| A.Any a |]
  | Value.Eig a, (w, v) ->
      let w = at w and v = at v and a = view a in
      outcome ~by (K.eig ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
  | Value.Eig_values a, w ->
      let w = at w and a = view a in
      outcome ~by (K.eig_values ~w a) [| A.Any w |] [| A.Any a |]
  | Value.Solve_triangular { triangle; adjoint; unit_diagonal; a; b }, y ->
      let dst = at y and a = view a and b = view b in
      outcome ~by
        (K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst a b)
        [| A.Any dst |]
        [| A.Any a; A.Any b |]

(* Two handles of one donor in [op] raise before any operand's death does: the
   operation is malformed whatever has died. *)
let rec run : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  if not (Prim.exists is_handle op) then begin
    Prim.iteri (fun i x -> checked ~by i x) op;
    plain ~by op
  end
  else
    let (Prim.Operands xs) = Prim.operands op in
    let hs = handles ~by xs in
    Prim.iteri (fun i x -> checked ~by i x) op;
    donated ~by op hs

(* [op] over donated operands [hs]: a movement that maps elements one to one
   passes a new handle on; any other operation reads the handles' memory, writes
   its first result into a handle's memory where it may, and ends the
   handles. *)
and donated : type r. by:string -> r Value.prim -> handle list -> r =
 fun ~by op hs ->
  match op with
  | Value.Move (mv, (Value.Donated d as x)) ->
      (* A view of a handle leaves the memory unshared: the handle it passes on
         has the same donor, so its consumer may still write into it. The
         deaths wait for that consumer, which checks every operand first, so
         an operation that aligns its operands by movements kills nothing when
         it is refused. *)
      let moved () =
        let r = run ~by (Value.Move (mv, live x)) in
        (Prim.placement r, Prim.arrays r)
      in
      let at, arrays =
        if Iarray.length d.arrays <> 1 then moved ()
        else begin
          ignore (Prim.prepare ~by (fun _ y -> y) op);
          match A.move mv (Iarray.get d.arrays 0) with
          | Some v -> (d.at, Iarray.of_list [ v ])
          | None -> moved ()
        end
      in
      spend ~by 0 (Value.Died by) x;
      Value.Donated { at; arrays; donor = d.donor; spent = Value.Live }
  | Value.Map ({ loads; _ } as m) -> (
      claim ~by hs;
      match reuse ~by m.layout m.outs loads with
      | Some (i, b, a) ->
          (* Load [i] is the result's memory: the kernel reads each index of it
             before it writes the result there. *)
          let load j = function
            | Value.Padded p -> Value.Padded { p with x = live p.x }
            | Value.Plain y when j <> i -> Value.Plain (live y)
            | Value.Plain y ->
                let at = Prim.placement y in
                Value.Plain
                  (Value.Array
                     { at; a = A.expect (Prim.dtype y) a; dead = Value.Live })
          in
          let r =
            plain ~by ~into:a
              (Value.Map { m with loads = Array.mapi load loads })
          in
          consumed ~by ~reused:[ b ] hs;
          r
      | None ->
          let r = run ~by (Prim.map live op) in
          consumed ~by ~reused:[] hs;
          r)
  | Value.Reduce _ | Value.Scan _ | Value.Gather _ | Value.Scatter _
  | Value.Sort _ | Value.Fft _ | Value.Linalg _ | Value.Fold _
  | Value.Assemble _ | Value.Contract _ | Value.Copy _ | Value.Move _
  | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      claim ~by hs;
      let r = run ~by (Prim.map live op) in
      consumed ~by ~reused:[] hs;
      r

and plain : type r. by:string -> ?into:A.any -> r Value.prim -> r =
 fun ~by ?into op ->
  match op with
  | Value.Place (p, x) -> (
      match x with
      | Value.Deferred _ -> Place.value ~by p (own (Devices.rebrand p) x)
      | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
          let y = Place.value ~by p x in
          (* A value over [x]'s memory, kept or borrowed, shares it. *)
          let xs = Prim.arrays x in
          Iarray.iter
            (fun b ->
              if
                Iarray.exists
                  (fun a -> Rig.Buffer.overlaps (A.buffer a) (A.buffer b))
                  xs
              then Rig.Claim.share (A.buffer b))
            (Prim.arrays y);
          y)
  | Value.Check { ok; data; fail } ->
      Prim.results ~by (fun k f -> alloc ~by k f) op;
      check ~by ok data fail
  | _ ->
      if not (Prim.exists is_concrete op) then defer ~by op
      else compute ~by ?into (Prim.prepare ~by (fun p x -> placer ~by p x) op)

(* [x] readable at [p]. Exec defers an operation over values of every set alone,
   so [p] is [Some _] here. *)
and placer : type v s d.
    by:string ->
    d Devices.placement option ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t =
 fun ~by p x ->
  let p = Option.get p in
  match x with
  | Value.Deferred _ -> at p x
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
      if Devices.equal (Prim.placement x) p then x else Place.value ~by p x

and defer : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  let id = Atomic.fetch_and_add nodes 1 in
  let node = Value.Node { id; by; op; memo = Atomic.make [] } in
  Prim.results ~by (fun k form -> Value.Deferred { form; node; k }) op

(* [op] computed over operands that lie where its route reads them, or at [at]
   for a constant's evaluation: its results are allocated there, and its
   expansion's operations compute there. A loop of no load is its expansion:
   no operand gives the kernels its extents. *)
and compute : type r.
    by:string -> ?into:A.any -> ?at:unit Devices.placement -> r Value.prim -> r
    =
 fun ~by ?into ?at op ->
  match op with
  | Value.Reduce { loads = [||]; _ } | Value.Scan { loads = [||]; _ } -> (
      match at with
      | Some q -> expanded ~by ~at:q q op
      | None -> invalid_arg "Exec.compute: a loop of no load at no placement")
  | Value.Map { prog; loads; _ } when Array.exists Prim.is_padded loads ->
      each_result ~by ?at op ~kind:"Map" (fun (module K) r j k w ->
          let dsts = Array.map (fun per -> per.(j)) (Prim.result_arrays op r) in
          let loads = Array.map (kernel_load ~by k w) loads in
          if ran_loads ~by (K.map prog ~dsts loads) dsts loads then None
          else Some (Array.map load_array loads))
  | Value.Map { layout; prog; loads; _ } ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ?at ~where ?into k f) op in
      map_devices ~by (Option.get !where) (L.shape layout) prog
        (fun k w -> Array.map (load_view ~by k w) loads)
        (Prim.result_arrays op r);
      r
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ?at ~where k f) op in
      let rs = Prim.reductions reductions in
      let reduce (module K : Nx_kernel.S) prog dsts loads =
        K.reduce prog ~axes rs ~dsts loads
      in
      if
        loop_devices ~by (Option.get !where) (L.shape layout) ~axes reduce prog
          (fun k w -> Array.map (kernel_load ~by k w) loads)
          (Prim.result_arrays op r)
      then r
      else expanded ~by ?at (Option.get !where) op
  | Value.Scan { layout; axis; prog; reduction; loads } ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ?at ~where k f) op in
      let reduction = Prim.reduction reduction in
      let scan (module K : Nx_kernel.S) prog dsts loads =
        K.scan prog ~axis reduction ~dsts loads
      in
      if
        loop_devices ~by (Option.get !where) (L.shape layout) ~axes:[||] scan
          prog
          (fun k w -> Array.map (kernel_load ~by k w) loads)
          (Prim.result_arrays op r)
      then r
      else expanded ~by ?at (Option.get !where) op
  | Value.Contract { batch; contracting; acc; a; b; init; _ } ->
      let where = ref None in
      let r =
        Prim.results ~by (fun k f -> alloc ~by ?at ~where ?into k f) op
      in
      let p = Option.get !where in
      let (module K) = kernels_of ~by ~op:"Contract" (Devices.set p) in
      let view x k = Place.view ~by x k (Array.map whole (Prim.shape x)) in
      let dsts = Prim.arrays r in
      let declined =
        first_decline ~by p (Prim.shape r) (fun j k _ ->
            let dst = Iarray.get dsts j in
            let a = view a k and b = view b k in
            let init = Option.map (fun i -> view i k) init in
            let ops =
              match init with
              | None -> [| A.Any a; A.Any b |]
              | Some i -> [| A.Any a; A.Any b; A.Any i |]
            in
            outcome ~by
              (K.contract ~batch ~contracting ~acc ~dst ~init a b)
              [| A.Any dst |] ops)
      in
      if Option.is_none declined then r else expanded ~by ?at p op
  | Value.Copy x ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ?at ~where k f) op in
      let dt = D.Any (Prim.dtype x) in
      let prog = P.of_node ~ins:[| dt |] (Op1 (Copy, dt, 0)) in
      map_devices ~by (Option.get !where) (Prim.shape x) prog
        (fun k w -> [| A.Any (Place.view ~by x k w) |])
        (Prim.result_arrays op r);
      r
  | Value.Move (mv, x) ->
      let xp = Prim.placement x and shape = Prim.shape x in
      let xs = Prim.arrays x in
      let cuts = Grid.cuts (Devices.grid xp) and set = Devices.set xp in
      let move a =
        moved ~by set (localize mv shape cuts (L.shape (A.layout a))) a
      in
      shard_views ~by op xp (fun () -> Iarray.map move xs)
  | Value.Bitcast (dt, x) ->
      let xp = Prim.placement x and xs = Prim.arrays x in
      let set = Devices.set xp in
      shard_views ~by op xp (fun () -> Iarray.map (cast ~by set dt) xs)
  | Value.Gather { axis; idx; x } ->
      let r, p = alloc_one ~by ?at op in
      let (module K) = kernels_of ~by ~op:"Gather" (Devices.set p) in
      let dsts = Prim.arrays r in
      first_decline ~by p (Prim.shape r) (fun j k w ->
          let dst = Iarray.get dsts j and i = Place.view ~by idx k w in
          let v = Place.view ~by x k (along axis (Prim.shape x) w) in
          outcome ~by (K.gather ~axis ~dst i v) [| A.Any dst |]
            [| A.Any i; A.Any v |])
      |> or_expanded ~by ~kernels:K.name ?at op r
  | Value.Scatter { combine; unique; axis; idx; updates; into = t } ->
      let r, p = alloc_one ~by ?at op in
      let (module K) = kernels_of ~by ~op:"Scatter" (Devices.set p) in
      let dsts = Prim.arrays r and s = Prim.shape updates in
      first_decline ~by p (Prim.shape r) (fun j k w ->
          let dst = Iarray.get dsts j and into = Place.view ~by t k w in
          let w' = along axis s w in
          let i = Place.view ~by idx k w' and u = Place.view ~by updates k w' in
          outcome ~by
            (K.scatter combine ~unique ~axis ~dst ~into i u)
            [| A.Any dst |]
            [| A.Any into; A.Any i; A.Any u |])
      |> or_expanded ~by ~kernels:K.name ?at op r
  | Value.Sort { axis; descending; x; _ } ->
      let where = ref None in
      let ((values, positions) as r) =
        Prim.results ~by (fun j f -> alloc ~by ?at ~where j f) op
      in
      let p = Option.get !where in
      let (module K) = kernels_of ~by ~op:"Sort" (Devices.set p) in
      let vs = Prim.arrays values and ps = Prim.arrays positions in
      first_decline ~by p (Prim.shape values) (fun j d w ->
          let v = Iarray.get vs j and q = Iarray.get ps j in
          let a = Place.view ~by x d (along axis (Prim.shape x) w) in
          outcome ~by
            (K.sort ~axis ~descending ~values:v ~positions:q a)
            [| A.Any v; A.Any q |] [| A.Any a |])
      |> or_expanded ~by ~kernels:K.name ?at op r
  | Value.Fold { pad; x; _ } ->
      each_result ~by ?at op ~kind:"Fold" (fun (module K) r j d _ ->
          let dst = Iarray.get (Prim.arrays r) j in
          let v = Place.view ~by x d (Array.map whole (Prim.shape x)) in
          outcome ~by (K.fold pad ~dst v) [| A.Any dst |] [| A.Any v |])
  | Value.Fft f -> each_result ~by ?at op ~kind:"Fft" (fft_on ~by f)
  | Value.Linalg l ->
      each_result ~by ?at op ~kind:(Prim.routine_name l) (linalg_on ~by l)
  | Value.Assemble { dtype; shape; fill; pieces } ->
      assemble ~by ?at dtype shape fill pieces
  | Value.Place _ | Value.Check _ -> run ~by op

(* [op]'s expansion, its operations applied where their operands lie, or at
   [at] for a constant's evaluation; [None] for a core case. *)
and expand : type r.
    by:string -> ?at:unit Devices.placement -> r Value.prim -> r option =
 fun ~by ?at op ->
  match at with
  | None -> Expand.run run ~by op
  | Some q -> Expand.run (fun ~by op -> apply_at q ~by op) ~by op

(* An operation of a constant's evaluation at [q]: one over a concrete operand
   computes where its route reads them; a creation, its other operands
   constants, computes at [q], where the evaluation reads it. *)
and apply_at : type r. unit Devices.placement -> by:string -> r Value.prim -> r
    =
 fun q ~by op ->
  if Prim.exists is_concrete op then run ~by op
  else compute ~by ~at:q (Prim.map (fun y -> at (Devices.rebrand q) y) op)

(* [op]'s results [r], fresh where its rule places them, computed on each
   device by [kernel kernels r j d w] for device [d], at position [j], and its
   window [w] of the first result, which answers as {!first_decline}'s [f].
   Where a device's kernels decline it, its expansion; an optional case nx has
   no expansion for raises, naming [kind]. *)
and each_result : type r.
    by:string ->
    ?at:unit Devices.placement ->
    r Value.prim ->
    kind:string ->
    ((module Nx_kernel.S) ->
    r ->
    int ->
    int ->
    M.range array ->
    A.any array option) ->
    r =
 fun ~by ?at op ~kind kernel ->
  let where = ref None and shape = ref [||] in
  let r =
    Prim.results ~by
      (fun k f ->
        if k = 0 then shape := L.shape f.layout;
        alloc ~by ?at ~where k f)
      op
  in
  let p = Option.get !where in
  let ((module K) as kernels) = kernels_of ~by ~op:kind (Devices.set p) in
  match first_decline ~by p !shape (kernel kernels r) with
  | None -> r
  | Some (d, ops) -> (
      match expand ~by ?at op with
      | Some r -> r
      | None -> unexpanded ~by ~kernels:K.name kind d (Array.to_list ops))

(* [op], a loop the kernels of [p]'s set declined, as its expansion. A plain
   loop has none: the decline raises naming the kernels. *)
and expanded : type r.
    by:string -> ?at:unit Devices.placement -> unit Devices.placement ->
    r Value.prim -> r =
 fun ~by ?at p op ->
  match expand ~by ?at op with
  | Some r -> r
  | None ->
      let (module K) = kernels_of ~by ~op:(Prim.name op) (Devices.set p) in
      invalid_argf "%s: %s does not compute %a; %s" by K.name Prim.pp op move

(* [op]'s result [r] where every device computed it. Where a device's kernels
   declined it, with operands [ops] on device [d], its expansion, on the
   devices that computed [r]; else the decline's error. *)
and or_expanded : type r.
    by:string ->
    kernels:string ->
    ?at:unit Devices.placement ->
    r Value.prim ->
    r ->
    (Rig.t * A.any array) option ->
    r =
 fun ~by ~kernels ?at op r declined ->
  match declined with
  | None -> r
  | Some (d, ops) -> (
      match expand ~by ?at op with
      | Some r -> r
      | None -> refuses ~by ~kernels (Prim.name op) d (Array.to_list ops))

(* A loop's results [dsts] at [p], each device's window of them computed there
   by [kernel] over its loads [loads k w] for its window [w] of the loop's
   iteration of [shape]: the results' window, whole along the reduced [axes].
   [false] where the kernels decline it. *)
and loop_devices ~by (p : unit Devices.placement) shape ~axes kernel prog loads
    dsts =
  let kernels = kernels_of ~by ~op:"Reduce" (Devices.set p) in
  let widen (w : M.range array) =
    let j = ref 0 in
    Array.mapi
      (fun a extent ->
        if Array.mem a axes then whole extent
        else begin
          let r = w.(!j) in
          incr j;
          r
        end)
      shape
  in
  let computed = ref true in
  each_device ~by p (Prim.reduced shape axes) (fun j k w ->
      if !computed then begin
        let w = widen w in
        let loads = loads k w and dsts = Array.map (fun per -> per.(j)) dsts in
        computed :=
          ran_loads ~by
            (kernel kernels (with_offsets prog (starts w)) dsts loads)
            dsts loads
      end);
  !computed

(* An assembly's result at [at], or where its rule places it: its kernel on each
   device, or its expansion where the kernels decline it. *)
and assemble : type v s d.
    by:string ->
    ?at:unit Devices.placement ->
    (v, s) D.t ->
    int array ->
    v ->
    (M.range array * (v, s, d) Value.t) list ->
    (v, s, d) Value.t =
 fun ~by ?at dtype shape fill pieces ->
  let op = Value.Assemble { dtype; shape; fill; pieces } in
  let r, p = alloc_one ~by ?at op in
  let (module K) = kernels_of ~by ~op:"Assemble" (Devices.set p) in
  let dsts = Prim.arrays r in
  let declined =
    first_decline ~by p shape (fun j k _ ->
        let view (rs, x) =
          (rs, Place.view ~by x k (Array.map whole (Prim.shape x)))
        in
        let pieces = Array.of_list (List.map view pieces) in
        let dst = Iarray.get dsts j in
        outcome ~by
          (K.assemble ~fill ~dst pieces)
          [| A.Any dst |]
          (Array.map (fun (_, v) -> A.Any v) pieces))
  in
  if Option.is_none declined then r
  else expanded ~by ?at (Devices.rebrand p) op

(* [op]'s one result, fresh at [at] or at the placement its rule gives, and that
   placement. *)
and alloc_one : type v s d.
    by:string ->
    ?at:unit Devices.placement ->
    (v, s, d) Value.t Value.prim ->
    (v, s, d) Value.t * d Devices.placement =
 fun ~by ?at op ->
  let where = ref None in
  let r = Prim.results ~by (fun k f -> alloc ~by ?at ~where k f) op in
  (r, Devices.rebrand (Option.get !where))

(* The one result of [op] over the arrays [views ()], one per device of [xp] in
   order: at [op]'s placement, whose devices hold them. [views] runs once [op]'s
   rule holds, so that no kernel copies for an operation it refuses. *)
and shard_views : type v s d.
    by:string ->
    (v, s, d) Value.t Value.prim ->
    d Devices.placement ->
    (unit -> (v, s) A.t iarray) ->
    (v, s, d) Value.t =
 fun ~by op xp views ->
  let xdev = Grid.devices (Devices.grid xp) in
  Prim.results ~by
    (fun _ f ->
      let vs = views () in
      (* Its operand lies at [xp], so it has a placement. *)
      let p = Option.get f.placement in
      let devices = Grid.devices (Devices.grid p) in
      make p
        (Iarray.init (Array.length devices) (fun i ->
             let j = Option.get (Array.find_index (( = ) devices.(i)) xdev) in
             A.expect f.dtype (A.Any (Iarray.get vs j)))))
    op

and check : type d.
    by:string ->
    (bool, D.bool_elt, d) Value.t ->
    d Value.any list ->
    (int array -> d Value.any list -> exn) ->
    unit =
 fun ~by ok data fail ->
  let host : d Devices.placement = host () in
  let shape = Prim.shape ok in
  match Array.find_index not (A.to_array (on_host ~by ok)) with
  | None -> ()
  | Some i ->
      let idx = unravel i shape in
      let at_idx (Value.Any x) =
        let a = on_host ~by x in
        let one =
          Array.map (fun i -> { M.start = i; count = 1; step = 1 }) idx
        in
        let v =
          Option.get (A.move (Reshape [||]) (Option.get (A.move (Slice one) a)))
        in
        Value.Any (Value.Array { at = host; a = v; dead = Value.Live })
      in
      raise (fail idx (List.map at_idx data))

(* A constant read at [p] by an operation: a view of a constant is its operand's
   results read at [p] and viewed; any other its results there, kept in its
   memo, which every later read at [p] finds. *)
and at : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x -> through_views Memo p x

(* [x]'s elements in an array on the host: a constant computed there, any other
   value placed there. *)
and on_host : type v s d. by:string -> (v, s, d) Value.t -> (v, s) A.t =
 fun ~by x ->
  let host : d Devices.placement = host () in
  match x with
  | Value.Deferred _ -> Iarray.get (Prim.arrays (at host x)) 0
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: the value is traced by %s" by owner.name
  | Value.Array _ | Value.Shards _ | Value.Donated _ ->
      Iarray.get (Prim.arrays (Place.value ~by host x)) 0

(* The constant [c] computed at [p] into memory of its own, its operands taken
   from their memos or computed for this alone: a value given to a caller shares
   no memory with a memo. A view of a constant views its operand computed into
   memory of its own. *)
and own : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x -> through_views Fresh p x

(* The constant [x] at [p]: the chain of views it reads descended to the first
   operation that is not one, whose results [policy] gives at the placement the
   chain reads it at, then each view replayed outward over its operand's. A
   chain of any length takes no stack depth. *)
and through_views : type v s d.
    policy -> d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun policy p x ->
  match x with
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  | Value.Deferred { form; node; k } ->
      (* Each view and where it is read, the innermost first. *)
      let views = ref [] and node = ref node in
      let key = ref (Devices.rebrand p) in
      let continue = ref true in
      while !continue do
        let (Value.Node n) = !node in
        match Prim.operands n.op with
        | Prim.Operands [ Value.Any (Value.Deferred { node = m; _ }) ]
          when is_view n.op ->
            views := (!node, !key) :: !views;
            key := operand_at n.op !key;
            node := m
        | Prim.Operands _ -> continue := false
      done;
      let (Value.Node n) = !node in
      let base =
        match policy with
        | Fresh -> evaluate !node !key
        | Memo -> (
            match find n.memo !key with
            | Some a -> a
            | None ->
                remember n.memo !key (evaluate !node !key);
                Option.get (find n.memo !key))
      in
      (* A view's one operand is the constant read just before it. *)
      let replay arrays (view, key) =
        let operand (type w r e) (q : e Devices.placement)
            (y : (w, r, e) Value.t) : (w, r, e) Value.t =
          match y with
          | Value.Deferred { form; k; _ } ->
              make q (typed form.dtype arrays.(k))
          | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
              y
        in
        compute_node ~resolve:operand view key
      in
      let arrays = List.fold_left replay base !views in
      make p (typed form.dtype arrays.(k))

(* [root]'s results at [p], fresh: the constants it reads are taken from their
   memos where an operation read them at the placement it reads them, and
   computed otherwise in a table of this evaluation's alone, dropped after. Each
   node computes once in an evaluation, in dependency order, with a stack of its
   own: a chain of any length takes no stack depth. *)
and evaluate root p =
  match pending root p with
  | [] -> compute_node ~resolve:memoised root p
  | _ -> evaluate_all root p

(* A constant operand at [q], from its memo: every one an operation reads is
   there. *)
and memoised : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun q x ->
  match x with
  | Value.Deferred { form; node = Value.Node n; k } ->
      let arrays = Option.get (find n.memo (Devices.rebrand q)) in
      make q (typed form.dtype arrays.(k))
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x

(* [evaluate] where some constant [root] reads is in no memo. *)
and evaluate_all root p =
  let local = Hashtbl.create 16 in
  let lookup (Value.Node n) q =
    match find n.memo q with
    | Some a -> Some a
    | None ->
        List.find_map
          (fun (q', a) -> if Devices.equal q' q then Some a else None)
          (Option.value ~default:[] (Hashtbl.find_opt local n.id))
  in
  (* How this evaluation reads a constant operand at [q]: from its memo or from
     the evaluation's own table. *)
  let resolve (type v s d) (q : d Devices.placement) (x : (v, s, d) Value.t) :
      (v, s, d) Value.t =
    match x with
    | Value.Deferred { form; node; k } ->
        let arrays = Option.get (lookup node (Devices.rebrand q)) in
        make q (typed form.dtype arrays.(k))
    | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  in
  let missing node q =
    List.filter (fun (dep, q) -> lookup dep q = None) (pending node q)
  in
  let stack = Stack.create () in
  List.iter (fun d -> Stack.push d stack) (missing root p);
  while not (Stack.is_empty stack) do
    let (Value.Node n as top), q = Stack.top stack in
    if lookup top q <> None then ignore (Stack.pop stack)
    else
      match missing top q with
      | [] ->
          ignore (Stack.pop stack);
          let a = compute_node ~resolve top q in
          let kept = Option.value ~default:[] (Hashtbl.find_opt local n.id) in
          Hashtbl.replace local n.id ((q, a) :: kept)
      | deps -> List.iter (fun d -> Stack.push d stack) deps
  done;
  compute_node ~resolve root p

(* A map's results [dsts] at [p], computed per device: [ops k w] are the
   operands on device [k] for its window [w]. A device whose kernels decline the
   map expands it over its own operands. *)
and map_devices ~by (p : unit Devices.placement) shape prog ops dsts =
  let set = Devices.set p in
  let kernels = kernels_of ~by ~op:"Map" set in
  match Grid.one (Devices.grid p) with
  | Some k ->
      let whole =
        Array.map (fun count -> { M.start = 0; count; step = 1 }) shape
      in
      let first = Array.make (Array.length shape) 0 in
      let ops = ops k whole and dsts = Array.map (fun per -> per.(0)) dsts in
      if not (map_on ~by kernels (Devices.rig set k) prog first ops dsts) then
        expand_on ~by set k prog shape first ops dsts
  | None ->
      Array.iteri
        (fun j k ->
          let w = Devices.window ~by p shape j in
          let shape = extents w and first = starts w in
          let ops = ops k w and dsts = Array.map (fun per -> per.(j)) dsts in
          if not (map_on ~by kernels (Devices.rig set k) prog first ops dsts)
          then expand_on ~by set k prog shape first ops dsts)
        (Grid.devices (Devices.grid p))

(* [prog] over [ops] into [dsts] on device [k] of [set], whose window of [shape]
   starts at [first], as the map's expansion ({!Expand.run}): one-node maps over
   the device's arrays, computed there, then copied into [dsts]. *)
and expand_on ~by (set : unit Devices.t) k prog shape first ops dsts =
  let at = Devices.one set k in
  let loads =
    Array.map
      (fun (A.Any a) -> Value.Plain (Value.Array { at; a; dead = Value.Live }))
      ops
  in
  let (Outs outs) = outs_of (Array.to_list dsts) in
  let op =
    Value.Map
      {
        layout = L.contiguous shape;
        prog = with_offsets prog first;
        outs;
        loads;
      }
  in
  match expand ~by ~at op with
  | None ->
      let (module K) = kernels_of ~by ~op:"Map" set in
      invalid_argf "%s: %s does not compute %a; %s" by K.name Prim.pp op move
  | Some r ->
      let ((module K) as kernels) = kernels_of ~by ~op:"Copy" set in
      Array.iteri
        (fun i per ->
          let (A.Any dst) = dsts.(i) in
          let (A.Any v) = per.(0) in
          let v = A.expect (A.dtype dst) (A.Any v) in
          if
            not
              (ran ~by (copy_into kernels ~dst v) [| A.Any dst |] [| A.Any v |])
          then
            declined ~by ~kernels:K.name
              (Op1 (Copy, D.Any (A.dtype v), 0))
              (Devices.rig set k) [ A.Any v ])
        (Prim.result_arrays op r)

(* [node]'s results at [p], its constant operands already computed where it
   reads them. *)
and compute_node :
    resolve:
      ('v 's 'd.
       'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t) ->
    Value.node ->
    unit Devices.placement ->
    A.any array array =
 fun ~resolve (Value.Node n) p ->
  let by = n.by in
  match n.op with
  | Value.Move (mv, x) when uncut p ->
      (* A view at a placement that cuts no axis: each device's whole operand
         there, moved. Its rule held when it was made. *)
      let xs = Prim.arrays (resolve (Devices.rebrand p) x) in
      [| any (Iarray.map (moved ~by (Devices.set p) mv) xs) |]
  | Value.Bitcast (dt, x) when uncut p ->
      let xs = Prim.arrays (resolve (Devices.rebrand p) x) in
      [| any (Iarray.map (cast ~by (Devices.set p) dt) xs) |]
  | op ->
      (* Its rule held when it was made. Its operands lie at [q], and its
         results are allocated there, where its rule may place them nowhere (a
         creation) or elsewhere (a replicated rule). *)
      let q = operand_at op p in
      let op' = Prim.map (fun y -> resolve (Devices.rebrand q) y) op in
      let arrays = Prim.result_arrays op' (compute ~by ~at:q op') in
      if Devices.equal q p then arrays
      else
        (* Whole on every device: each device of [p] keeps its window, in memory
           of its own, so that no memo holds the whole on each. *)
        Array.map
          (fun per ->
            let (A.Any a0) = per.(0) in
            let shape = L.shape (A.layout a0) in
            Array.mapi
              (fun j k ->
                let (A.Any a) = per.(k) in
                let w = Devices.window ~by p shape j in
                A.Any (A.copy (Option.get (A.move (Slice w) a))))
              (Grid.devices (Devices.grid p)))
          arrays

(* The fast path *)

(* A destination for a result of [dt] beside [x]: over [x]'s layout where it is
   C-contiguous from offset 0 and of [dt], fresh otherwise. A dtype narrower
   than a byte is always fresh, whose spare bits in its last byte are zero. *)
let destination (type v s w r) (dt : (w, r) D.t) (x : (v, s) A.t) : (w, r) A.t =
  let l = A.layout x in
  if
    L.is_contiguous l
    && L.offset l = 0
    && D.equal dt (A.dtype x)
    && D.bits dt >= 8
  then A.v dt l (Rig.Buffer.create (A.device x) (D.bytes dt (L.numel l)))
  else A.create (A.device x) dt (L.shape l)

(* The one-node map over the handle [h], operand [i], whose array is [a], its
   result of [dt]: [h] is claimed, then its memory takes the
   result where rig holds it exclusive and it has the result's dtype and
   C-contiguous layout over all of its memory; otherwise the result is fresh
   and the memory, where no other value reads it, is consumed after the kernel.
   [run dst a'] is the kernel into [dst] with [a'] for [h]. *)
let donated (type v s w r d) ~by i (dt : (w, r) D.t) (h : (v, s, d) Value.t)
    (a : (v, s) A.t) (run : (w, r) A.t -> (v, s) A.t -> (w, r, d) Value.t) :
    (w, r, d) Value.t =
  let l = A.layout a in
  let reusable =
    L.is_contiguous l && L.offset l = 0 && Rig.Buffer.spans (A.buffer a)
  in
  claim_handle ~by i h;
  match D.equal_witness (A.dtype a) dt with
  | Some Type.Equal when reusable -> (
      match take ~by (A.buffer a) with
      | Some b' ->
          let a' = A.v (A.dtype a) l b' in
          run a' a'
      | None -> run (destination dt a) a)
  | Some _ | None ->
      let r = run (destination dt a) a in
      ignore (take ~by (A.buffer a));
      r

(* Whether [x] is a live array at [at]. *)
let live_at (type v s d) (at : d Devices.placement) (x : (v, s, d) Value.t) =
  match x with
  | Value.Array r -> r.at == at && r.dead == Value.Live
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      false

(* Whether [h] is a handle over one array at [at], whose set has kernels: the
   donated fast paths'. *)
let handle_at (type v s d) (at : d Devices.placement) (h : (v, s, d) Value.t) =
  match h with
  | Value.Donated d ->
      d.at == at
      && Iarray.length d.arrays = 1
      && Devices.kernels (Devices.set at) <> None
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* [a] as a live value at [at]. *)
let value_of at a = Value.Array { at; a; dead = Value.Live }

(* [at]'s kernels, where [handle_at] holds. *)
let kernels_at at = Option.get (Devices.kernels (Devices.set at))

let apply1 (type v s w r d) ~slow ~by k (dt : (w, r) D.t)
    (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match x with
  | Value.Array rx when rx.dead == Value.Live -> (
      let at = rx.at and a = rx.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k dt x
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply1 k ~dst a with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype -> slow ~by k dt x
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ]))
  | Value.Donated dx when handle_at dx.at x ->
      let (module K) = kernels_at dx.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          match K.apply1 k ~dst a with
          | Done -> value_of dx.at dst
          | Declined | Wrong_dtype -> slow ~by k dt (value_of dx.at a)
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ])
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _
  | Value.Traced _ ->
      slow ~by k dt x

(* [k] into [dst] over the arrays [a] and [b] at [at]: [Done] is the result, a
   decline or a dtype or shape refusal the slow path over them as live
   values. *)
let kernel2 (type v s w r d) ~slow ~by (module K : Nx_kernel.S) k
    (dt : (w, r) D.t) (at : d Devices.placement) (dst : (w, r) A.t)
    (a : (v, s) A.t) (b : (v, s) A.t) : (w, r, d) Value.t =
  match K.apply2 k ~dst a b with
  | Done -> value_of at dst
  | Declined | Wrong_dtype | Shape_mismatch ->
      slow ~by k dt (value_of at a) (value_of at b)
  | refusal -> A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]

(* As [kernel2], for [apply3]. *)
let kernel3 (type a b v s d) ~slow ~by (module K : Nx_kernel.S) k
    (at : d Devices.placement) (dst : (v, s) A.t) (c : (a, b) A.t)
    (x : (v, s) A.t) (y : (v, s) A.t) : (v, s, d) Value.t =
  match K.apply3 k ~dst c x y with
  | Done -> value_of at dst
  | Declined | Wrong_dtype | Shape_mismatch ->
      slow ~by k (value_of at c) (value_of at x) (value_of at y)
  | refusal -> A.refused by refusal [ A.Any dst; A.Any c; A.Any x; A.Any y ]

(* Whether the one-node map [k] over [x] and [y] is well formed. A constant is
   read for an operation only once its rule holds, so that a refused operation
   computes nothing. *)
let well_formed2 k x y = P.accepts2 k (Prim.dtype x) && Prim.same_shape x y

let well_formed3 k c x y =
  P.accepts3 k (Prim.dtype c) (Prim.dtype x)
  && Prim.same_shape c x && Prim.same_shape x y

let rec apply2 : type v s w r d.
    slow:
      (by:string ->
      P.op2 ->
      (w, r) D.t ->
      (v, s, d) Value.t ->
      (v, s, d) Value.t ->
      (w, r, d) Value.t) ->
    by:string ->
    P.op2 ->
    (w, r) D.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (w, r, d) Value.t =
 fun ~slow ~by k dt x y ->
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rx.at and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k dt x y
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply2 k ~dst a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype | Shape_mismatch -> slow ~by k dt x y
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]))
  | Value.Donated dx, Value.Array ry when handle_at ry.at x && live_at ry.at y
    ->
      let (module K) = kernels_at ry.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          kernel2 ~slow ~by (module K) k dt ry.at dst a ry.a)
  | Value.Array rx, Value.Donated dy when handle_at rx.at y && live_at rx.at x
    ->
      let (module K) = kernels_at rx.at in
      donated ~by 1 dt y (Iarray.get dy.arrays 0) (fun dst b ->
          kernel2 ~slow ~by (module K) k dt rx.at dst rx.a b)
  | Value.Array rx, Value.Deferred _
    when rx.dead == Value.Live && well_formed2 k x y ->
      (* A constant beside a value is read where the value lies. *)
      apply2 ~slow ~by k dt x (at rx.at y)
  | Value.Deferred _, Value.Array ry
    when ry.dead == Value.Live && well_formed2 k x y ->
      apply2 ~slow ~by k dt (at ry.at x) y
  | _ -> slow ~by k dt x y

(* Where the constants among [c], [x] and [y] are read: the one placement of the
   live arrays among them, where every other is a constant. *)
let beside3 (type a b v s d) (c : (a, b, d) Value.t) (x : (v, s, d) Value.t)
    (y : (v, s, d) Value.t) : d Devices.placement option =
  let place : type v s. (v, s, d) Value.t -> d Devices.placement option option =
    function
    | Value.Deferred _ -> Some None
    | Value.Array r when r.dead == Value.Live -> Some (Some r.at)
    | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> None
  in
  match (place c, place x, place y) with
  | Some pc, Some px, Some py -> (
      match List.filter_map Fun.id [ pc; px; py ] with
      | p :: rest when List.for_all (( == ) p) rest -> Some p
      | _ -> None)
  | _ -> None

let rec apply3 : type a b v s d.
    slow:
      (by:string ->
      P.op3 ->
      (a, b, d) Value.t ->
      (v, s, d) Value.t ->
      (v, s, d) Value.t ->
      (v, s, d) Value.t) ->
    by:string ->
    P.op3 ->
    (a, b, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t =
 fun ~slow ~by k c x y ->
  match (c, x, y) with
  | Value.Array rc, Value.Array rx, Value.Array ry
    when rc.at == rx.at && rc.at == ry.at && rc.dead == Value.Live
         && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rc.at and ca = rc.a and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k c x y
      | Some (module K) -> (
          let dst = destination (A.dtype a) a in
          match K.apply3 k ~dst ca a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype | Shape_mismatch -> slow ~by k c x y
          | refusal ->
              A.refused by refusal [ A.Any dst; A.Any ca; A.Any a; A.Any b ]))
  | Value.Donated dc, Value.Array rx, Value.Array ry
    when handle_at rx.at c && live_at rx.at x && live_at rx.at y ->
      let (module K) = kernels_at rx.at in
      donated ~by 0 (A.dtype rx.a) c (Iarray.get dc.arrays 0)
        (fun dst c' -> kernel3 ~slow ~by (module K) k rx.at dst c' rx.a ry.a)
  | Value.Array rc, Value.Donated dx, Value.Array ry
    when handle_at rc.at x && live_at rc.at c && live_at rc.at y ->
      let (module K) = kernels_at rc.at in
      donated ~by 1 (A.dtype ry.a) x (Iarray.get dx.arrays 0)
        (fun dst x' -> kernel3 ~slow ~by (module K) k rc.at dst rc.a x' ry.a)
  | Value.Array rc, Value.Array rx, Value.Donated dy
    when handle_at rc.at y && live_at rc.at c && live_at rc.at x ->
      let (module K) = kernels_at rc.at in
      donated ~by 2 (A.dtype rx.a) y (Iarray.get dy.arrays 0)
        (fun dst y' -> kernel3 ~slow ~by (module K) k rc.at dst rc.a rx.a y')
  | (Value.Deferred _, _, _ | _, Value.Deferred _, _ | _, _, Value.Deferred _)
    when well_formed3 k c x y -> (
      match beside3 c x y with
      | Some p -> apply3 ~slow ~by k (at p c) (at p x) (at p y)
      | None -> slow ~by k c x y)
  | _ -> slow ~by k c x y

(* The contraction of live arrays on one device, at physically one placement
   whose set has kernels, computed by them without building the operation.
   [None] elsewhere, and where the kernels decline it: the operation decides
   those. *)
let contract (type v s a b c e d) ~by ~batch ~contracting ~acc (dt : (v, s) D.t)
    (x : (a, b, d) Value.t) (y : (c, e, d) Value.t)
    (init : (v, s, d) Value.t option) : (v, s, d) Value.t option =
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rx.at in
      let set = Devices.set at in
      let array =
        match init with
        | None -> Some None
        | Some (Value.Array ri) when ri.at == at && ri.dead == Value.Live ->
            Some (Some ri.a)
        | Some _ -> None
      in
      match (array, Grid.one (Devices.grid at), Devices.kernels set) with
      | Some array, Some k, Some (module K) -> (
          let s = Prim.contract_shape ~by ~batch ~contracting ~acc x y init in
          let dst = A.create (Devices.rig set k) dt s in
          match
            K.contract ~batch ~contracting ~acc ~dst ~init:array rx.a ry.a
          with
          | Done -> Some (Value.Array { at; a = dst; dead = Value.Live })
          | Declined -> None
          | refusal ->
              let init = Option.to_list (Option.map (fun i -> A.Any i) array) in
              A.refused by refusal
                (A.Any dst :: A.Any rx.a :: A.Any ry.a :: init))
      | _ -> None)
  | _ -> None
