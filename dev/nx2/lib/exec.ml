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
let ran_loads ~by ?(reads = []) answer dsts loads =
  match (answer : A.answer) with
  | Done -> true
  | Declined -> false
  | refusal ->
      A.refused by refusal
        (Array.to_list dsts @ List.map load_array (Array.to_list loads) @ reads)

(* Values from arrays *)

let make = Prim.of_arrays

(* Arrays kept untyped, one per device, at [dt]. *)
let typed dt (per : A.any array) =
  Iarray.init (Array.length per) (fun j -> A.expect dt per.(j))

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

(* The key the next donor is offered with. *)
let offers = Atomic.make 0

(* Gives the live donor word [w] a key of its own, which every handle of every
   donation of it reads. *)
let offer (w : Value.death Atomic.Loc.t) =
  match Atomic.Loc.get w with
  | Value.Live ->
      (* Fails only where another donation offered it first: a donor dies only
         through a handle, which it has after it is offered. *)
      let k = Atomic.fetch_and_add offers 1 in
      ignore (Atomic.Loc.compare_and_set w Value.Live (Value.Offered k))
  | Value.Offered _ | Value.Died _ -> ()

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Prim.alive ~by 0 x;
  let handle at shape arrays donor =
    offer donor;
    Value.Donated { at; shape; arrays; donor; spent = Value.Live }
  in
  match x with
  | Value.Array r ->
      handle r.at
        (L.shape (A.layout r.a))
        ([| r.a |] : _ iarray)
        [%atomic.loc r.dead]
  | Value.Shards r -> handle r.at r.shape r.arrays [%atomic.loc r.dead]
  | Value.Donated _ | Value.Deferred _ | Value.Traced _ -> x

(* A handle's arrays as a live value, which the operation reads. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated { at; shape; arrays; _ } -> Prim.of_arrays at ~shape arrays
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

(* A donated operand and its index among the operation's operands. *)
type handle = Handle : int * ('v, 's, 'd) Value.t -> handle

let is_handle (type v s d) (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated _ -> true
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

let is_concrete x = not (Prim.is_constant x)

let is_cut x =
  match Prim.at x with Some p -> Grid.is_cut (Devices.grid p) | None -> false

(* Whether the donor words [w] and [w'] are one donor's: an offered donor has
   one key. *)
let one_donor (w : Value.death Atomic.Loc.t) w' =
  match (Atomic.Loc.get w, Atomic.Loc.get w') with
  | Value.Offered k, Value.Offered k' -> k = k'
  | (Value.Live | Value.Offered _ | Value.Died _), _ -> false

(* The donated operands among [xs]. Two handles of one donor or over one memory
   raise naming [Nx.donate]: the same handle twice, two donations of one value,
   or one handle and a value a movement passed on. *)
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
                  one_donor a.donor b.donor
                  || Iarray.exists
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

(* Claims

   A consumer claims its handles in their donors' key order, each handle's own
   word before its donor's, setting each word to its death; it gives none back
   when it loses one. Each word has one place in the order, since every handle
   of a donor reads the donor's one key, and a consumer's words rise strictly,
   since [handles] refuses two handles of one donor. So of consumers racing for
   the same handles, at least one succeeds: a consumer that loses at a word
   lost it to one that claimed that word, which, if it lost too, lost at a word
   above it. Following who each loser lost to climbs the order, so it ends at a
   consumer that claimed all its words.

   Claimed in operand order, [add h1 h2] and [add h2 h1] on two domains could
   each take its first handle and lose its second to the other. In key order a
   loser may still name a loser. With h1 < h2 < h3, [add h1 h2] takes h1,
   [add h2 h3] takes h2, [add h1 h3] loses h1 naming the first, which loses h2
   naming the second, which takes h3 and succeeds. *)

let died_to ~by i earlier =
  invalid_argf "%s: operand %d was donated to %s" by (i + 1) earlier

(* Sets the death word [w] of operand [i] to [died] unless it died first, on
   any domain, which raises naming that consumer. No death is undone, so a word
   read live is only retried. *)
let rec claim_word ~by i (w : Value.death Atomic.Loc.t) died =
  match Atomic.Loc.get w with
  | Value.Died earlier -> died_to ~by i earlier
  | (Value.Live | Value.Offered _) as seen ->
      if not (Atomic.Loc.compare_and_set w seen died) then
        claim_word ~by i w died

(* Sets the handle [x]'s own word, operand [i], to [died] unless an operation
   read the handle first, which raises naming it. The word is set in place:
   passed to [claim_word], its location would be allocated. *)
let rec spend_word : type v s d.
    by:string -> int -> Value.death -> (v, s, d) Value.t -> unit =
 fun ~by i died x ->
  match x with
  | Value.Donated d -> (
      if not (Atomic.Loc.compare_and_set [%atomic.loc d.spent] Value.Live died)
      then
        match d.spent with
        | Value.Died earlier -> died_to ~by i earlier
        | Value.Live | Value.Offered _ -> spend_word ~by i died x)
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

(* Marks the handle [x], operand [i], read by the movement [by], which passes a
   new handle on: one operation reads a handle. *)
let spend ~by i x = spend_word ~by i (Value.Died by) x

(* Claims the handle [x], operand [i], for its consumer [by], before it reads
   it: its own word, then its donor's, which orders the consumers of all the
   donor's handles. *)
let claim_handle (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d ->
      let died = Value.Died by in
      spend_word ~by i died x;
      claim_word ~by i d.donor died
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

(* The key of the handle [x]'s donor, operand [i]. A donor that died raises
   naming its consumer. *)
let key (type v s d) ~by i (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d -> (
      match Atomic.Loc.get d.donor with
      | Value.Offered k -> k
      | Value.Died earlier -> died_to ~by i earlier
      | Value.Live -> invalid_arg "Exec.key: a handle's donor was not offered")
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
      invalid_arg "Exec.key: not a handle"

(* Claims the handles [hs] for their consumer [by], in key order. Their keys
   differ: [handles] refused two handles of one donor. *)
let claim ~by hs =
  let keyed = List.map (fun (Handle (i, x) as h) -> (key ~by i x, h)) hs in
  List.iter
    (fun (_, Handle (i, x)) -> claim_handle ~by i x)
    (List.sort (fun (k, _) (k', _) -> Int.compare k k') keyed)

let consume ~by op =
  let (Prim.Operands xs) = Prim.operands op in
  let hs = if Prim.exists is_handle op then handles ~by xs else [] in
  Prim.iteri (fun i x -> Prim.alive ~by i x) op;
  claim ~by hs

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

let extents (w : M.range array) = Array.map (fun (r : M.range) -> r.count) w

(* Fresh C-contiguous arrays of a value of [shape] at [p]: each device's
   window. *)
let fresh ~by p dt shape =
  let set = Devices.set p in
  match Grid.one (Devices.grid p) with
  | Some k -> ([| A.create (Devices.rig set k) dt shape |] : _ iarray)
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
  | Some _ | None ->
      let shape = L.shape f.layout in
      make p ~shape (fresh ~by p f.dtype shape)

(* Programs on one device *)

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

(* [prog] over [loads] into [dsts] on device [d]: its kernel where it is one
   node over plain loads, else the kernels' map. [false] where the kernels
   decline the map; a one-node program they decline raises, since its node is
   core. *)
let map_on ~by (module K : Nx_kernel.S) d (prog : P.t) loads dsts =
  let plain = function
    | Nx_kernel.Plain _ -> true
    | Nx_kernel.Padded _ -> false
  in
  match Expand.single prog with
  | Some node when Array.for_all plain loads ->
      let ops = Array.map load_array loads in
      let arrays = Array.append dsts ops in
      ran ~by (apply_node (module K) node dsts.(0) ops) dsts ops
      ||
      (* A decline at base dtypes is the kernels' error; at others the node
         expands. *)
      if Array.for_all (fun (A.Any a) -> Expand.base (D.Any (A.dtype a))) arrays
      then declined ~by ~kernels:K.name node d (Array.to_list arrays)
      else false
  | Some _ | None -> ran_loads ~by (K.map prog ~dsts loads) dsts loads

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
   where the movement names its extent, the shard's own extent along it
   replaces it. *)
let localize (mv : M.t) shape cuts local : M.t =
  let cut a = Array.exists (fun (c, _) -> c = a) cuts in
  let target s' =
    let s' = Array.copy s' in
    Array.iter
      (fun (a, _) ->
        Option.iter (fun a' -> s'.(a') <- local.(a)) (Route.moved mv shape a))
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
             if cut a then { M.start = 0; count = local.(a); step = 1 } else r)
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

(* The route of the constant operation [op] whose results are wanted at [p]:
   where it computes them and reads each operand ({!Route.at}); [None] where
   that is [p] for all: [p] cuts no axis, or the rule is elementwise. *)
let route_at : type r.
    by:string -> r Value.prim -> unit Devices.placement -> unit Route.t option =
 fun ~by op p ->
  if uncut p then None
  else
    let rule = Prim.rule op in
    if rule = Route.Elementwise then None
    else begin
      let shapes = ref [] in
      Prim.iteri (fun _ x -> shapes := Prim.shape x :: !shapes) op;
      Some (Route.at ~by rule (Array.of_list (List.rev !shapes)) p)
    end

(* Where a route [r] from [route_at] for [p] reads operand [i]. *)
let read_at (r : unit Route.t option) p i =
  match r with None -> p | Some r -> r.operands.(i)

(* How a constant's results are read: from its memo, where they are kept, or
   computed into memory of their own. *)
type policy = Memo | Fresh

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
  let r = route_at ~by:n.by n.op p in
  let found = ref [] in
  Prim.iteri
    (fun i x ->
      match x with
      | Value.Deferred { node = Value.Node m as node; _ } ->
          let q = read_at r p i in
          if find m.memo q = None then found := (node, q) :: !found
      | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> ())
    n.op;
  List.rev !found

(* The host's device, at the brand of a value this library reads there: a
   check's flag, a value read out ({!on_host}). No value at it reaches a
   function of nx's own. *)
let host () = Devices.rebrand (Devices.one Devices.host 0)

(* The next node's id. *)
let nodes = Atomic.make 0

(* A layout no result has, before its first is known. *)
let no_layout = L.contiguous [||]

(* The one array of [x], a value on one device. *)
let one (type v s d) (x : (v, s, d) Value.t) = Iarray.get (Prim.arrays x) 0

(* The load a kernel reads of [l], over a value on one device. *)
let kernel_load (type d) (l : d Value.load) : Nx_kernel.load =
  match l with
  | Value.Plain x -> Nx_kernel.Plain (one x)
  | Value.Padded { x; fill; pad } -> Nx_kernel.Padded { x = one x; fill; pad }

(* The arrays at position [j] of [x] and of the results [r] of [op]. *)
let nth (type v s d) j (x : (v, s, d) Value.t) = Iarray.get (Prim.arrays x) j
let dsts_at op r j = Array.map (fun per -> per.(j)) (Prim.result_arrays op r)

(* [op]'s kernel on device [d], over its operands' arrays there, into the
   results [r]'s arrays at position [j]: [false] where it declines. A refusal
   raises naming [by]. *)
let call : type r.
    by:string ->
    (module Nx_kernel.S) ->
    Rig.t ->
    r Value.prim ->
    r ->
    int ->
    bool =
 fun ~by kernels d op r j ->
  let (module K : Nx_kernel.S) = kernels in
  match op with
  | Value.Map { prog; loads; _ } ->
      map_on ~by kernels d prog (Array.map kernel_load loads) (dsts_at op r j)
  | Value.Reduce { layout; axes; prog; reductions; loads } ->
      let dsts = dsts_at op r j and loads = Array.map kernel_load loads in
      let rs = Prim.reductions reductions and shape = L.shape layout in
      ran_loads ~by (K.reduce prog ~shape ~axes rs ~dsts loads) dsts loads
  | Value.Scan { axis; prog; reduction; loads; carry; _ } ->
      let dsts = dsts_at op r j and loads = Array.map kernel_load loads in
      let reduction = Prim.reduction reduction in
      let carry = Option.map (fun (Value.Any c) -> A.Any (one c)) carry in
      let reads = Option.to_list carry in
      ran_loads ~by ~reads (K.scan ?carry prog ~axis reduction ~dsts loads) dsts
        loads
  | Value.Contract { order; batch; contracting; acc; a; b; init; _ } ->
      let dst = nth j r and a = one a and b = one b in
      let init = Option.map one init in
      let ops =
        match init with
        | None -> [| A.Any a; A.Any b |]
        | Some i -> [| A.Any a; A.Any b; A.Any i |]
      in
      ran ~by
        (K.contract ~order ~batch ~contracting ~acc ~dst ~init a b)
        [| A.Any dst |] ops
  | Value.Gather { axis; idx; x } ->
      let dst = nth j r and i = one idx and x = one x in
      ran ~by (K.gather ~axis ~dst i x) [| A.Any dst |] [| A.Any i; A.Any x |]
  | Value.Scatter { combine; unique; axis; idx; updates; into } ->
      let dst = nth j r and into = one into in
      let i = one idx and u = one updates in
      ran ~by
        (K.scatter combine ~unique ~axis ~dst ~into i u)
        [| A.Any dst |]
        [| A.Any into; A.Any i; A.Any u |]
  | Value.Sort { axis; descending; x; _ } ->
      let values, positions = r in
      let v = nth j values and q = nth j positions and a = one x in
      ran ~by
        (K.sort ~axis ~descending ~values:v ~positions:q a)
        [| A.Any v; A.Any q |] [| A.Any a |]
  | Value.Fold { pad; x; _ } ->
      let dst = nth j r and v = one x in
      ran ~by (K.fold pad ~dst v) [| A.Any dst |] [| A.Any v |]
  | Value.Fft (Value.C2c { direction; axes; x }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.c2c direction ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Fft (Value.R2c { axes; x; _ }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.r2c ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Fft (Value.C2r { axes; x; _ }) ->
      let dst = nth j r and x = one x in
      ran ~by (K.c2r ~axes ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Linalg l -> (
      match (l, r) with
      | Value.Cholesky { triangle; a }, y ->
          let dst = nth j y and a = one a in
          ran ~by (K.cholesky triangle ~dst a) [| A.Any dst |] [| A.Any a |]
      | Value.Lu a, (lu, pivots, perm) ->
          let lu = nth j lu and pivots = nth j pivots and perm = nth j perm in
          let a = one a in
          ran ~by (K.lu ~lu ~pivots ~perm a)
            [| A.Any lu; A.Any pivots; A.Any perm |]
            [| A.Any a |]
      | Value.Qr { a; _ }, (q, r) ->
          let q = nth j q and r = nth j r and a = one a in
          ran ~by (K.qr ~q ~r a) [| A.Any q; A.Any r |] [| A.Any a |]
      | Value.Svd { vectors = No_vectors; a }, (s, ()) ->
          let s = nth j s and a = one a in
          ran ~by (K.svd ~s a) [| A.Any s |] [| A.Any a |]
      | Value.Svd { vectors = Vectors _; a }, (s, (u, vh)) ->
          let s = nth j s and u = nth j u and vh = nth j vh and a = one a in
          ran ~by
            (K.svd ~s ~vectors:(~u, ~vh) a)
            [| A.Any s; A.Any u; A.Any vh |]
            [| A.Any a |]
      | Value.Eigh { vectors = No_vectors; a }, (w, ()) ->
          let w = nth j w and a = one a in
          ran ~by (K.eigh ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eigh { vectors = Vectors (); a }, (w, v) ->
          let w = nth j w and v = nth j v and a = one a in
          ran ~by (K.eigh ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
      | Value.Eig { vectors = No_vectors; a; _ }, (w, ()) ->
          let w = nth j w and a = one a in
          ran ~by (K.eig ~w a) [| A.Any w |] [| A.Any a |]
      | Value.Eig { vectors = Vectors (); a; _ }, (w, v) ->
          let w = nth j w and v = nth j v and a = one a in
          ran ~by (K.eig ~w ~v a) [| A.Any w; A.Any v |] [| A.Any a |]
      | Value.Solve_triangular { triangle; adjoint; unit_diagonal; a; b }, y ->
          let dst = nth j y and a = one a and b = one b in
          ran ~by
            (K.solve_triangular triangle ~adjoint ~unit_diagonal ~dst a b)
            [| A.Any dst |] [| A.Any a; A.Any b |])
  | Value.Assemble { fill; pieces; _ } ->
      let dst = nth j r in
      let pieces =
        Array.of_list (List.map (fun (rs, x) -> (rs, one x)) pieces)
      in
      ran ~by
        (K.assemble ~fill ~dst pieces)
        [| A.Any dst |]
        (Array.map (fun (_, v) -> A.Any v) pieces)
  | Value.Copy x ->
      let dst = nth j r and x = one x in
      ran ~by (copy_into kernels ~dst x) [| A.Any dst |] [| A.Any x |]
  | Value.Move _ | Value.Bitcast _ | Value.Place _ | Value.Check _ ->
      invalid_arg "Exec.call: an operation no kernel computes"

(* Raises for [op], whose kernels on [d] declined it and which nx does not
   expand: a core case, or a routine whose expansion is not available yet. *)
let refused : type r. by:string -> kernels:string -> Rig.t -> r Value.prim -> 'a
    =
 fun ~by ~kernels d op ->
  let dts = ref [] in
  Prim.iteri (fun _ x -> dts := A.Any (one x) :: !dts) op;
  let dts = List.rev !dts in
  match op with
  | Value.Linalg l -> unexpanded ~by ~kernels (Prim.routine_name l) d dts
  | _ -> refuses ~by ~kernels (Prim.name op) d dts

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
        (Prim.placement r, Prim.shape r, Prim.arrays r)
      in
      let at, shape, arrays =
        if Iarray.length d.arrays <> 1 then moved ()
        else begin
          ignore (Prim.prepare ~by (fun _ y -> y) op);
          match A.move mv (Iarray.get d.arrays 0) with
          | Some v -> (d.at, L.shape (A.layout v), [| v |])
          | None -> moved ()
        end
      in
      spend ~by 0 x;
      Value.Donated { at; shape; arrays; donor = d.donor; spent = Value.Live }
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
   expansion's operations compute there. A loop of no load, a constant's, has
   only [at] to place it. *)
and compute : type r.
    by:string -> ?into:A.any -> ?at:unit Devices.placement -> r Value.prim -> r
    =
 fun ~by ?into ?at op ->
  match op with
  | (Value.Reduce { loads = [||]; _ } | Value.Scan { loads = [||]; _ })
    when Option.is_none at ->
      invalid_arg "Exec.compute: a loop of no load at no placement"
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
  | Value.Place _ | Value.Check _ -> run ~by op
  | Value.Reduce { layout; axes; prog; reductions; loads }
    when Prim.exists is_cut op -> (
      match Prim.loads_route ~by op layout loads with
      | Some ({ compute = Partials; _ } as r) ->
          Partial.reduce (fun ~by op -> folded ~by ?at op) ~by r layout axes prog
            reductions loads
      | Some { compute = Windows; _ } | None -> sharded ~by ?into ?at op)
  | Value.Scan { layout; axis; prog; reduction; loads; carry = None }
    when Prim.exists is_cut op -> (
      match Prim.loads_route ~by op layout loads with
      | Some ({ compute = Partials; _ } as r) ->
          Partial.scan (fun ~by op -> folded ~by ?at op) ~by r layout axis prog
            reduction loads
      | Some { compute = Windows; _ } | None -> sharded ~by ?into ?at op)
  | Value.Contract { order; batch; contracting; acc; out; a; b; init }
    when Prim.exists is_cut op -> (
      match Prim.contract_route ~by op a b init with
      | Some ({ compute = Partials; _ } as r) ->
          Partial.contract (fun ~by op -> folded ~by ?at op) ~by r ~order ~batch
            ~contracting ~acc out a b init
      | Some { compute = Windows; _ } | None -> sharded ~by ?into ?at op)
  | Value.Map _ | Value.Reduce _ | Value.Scan _ | Value.Gather _
  | Value.Scatter _ | Value.Sort _ | Value.Fold _ | Value.Fft _ | Value.Linalg _
  | Value.Assemble _ | Value.Contract _ | Value.Copy _ ->
      sharded ~by ?into ?at op

(* [op]'s results, fresh where its route places them or at [at], each device's
   window [w] of the first computed there by its kernels over each operand's
   window ({!Route.window}). A device whose kernels decline computes its window
   by the expansion, over its own operands: no data leaves the device. On one
   device the expansion's results are the results; on several, or into a donated
   operand's memory, they are copied into the device's window. *)
and sharded : type r.
    by:string -> ?into:A.any -> ?at:unit Devices.placement -> r Value.prim -> r
    =
 fun ~by ?into ?at op ->
  let where = ref None and first = ref no_layout in
  let r =
    Prim.results ~by
      (fun k f ->
        if k = 0 then first := f.layout;
        alloc ~by ?at ~where ?into k f)
      op
  in
  let p = Option.get !where in
  let set = Devices.set p in
  let kernels = kernels_of ~by ~op:(Prim.name op) set in
  match Grid.one (Devices.grid p) with
  | Some k when Option.is_none into ->
      (* On one device the expansion's results are the results: no result of an
         expansion is an operand ({!Expand.run}), so nothing is copied. *)
      let d = Devices.rig set k in
      if call ~by kernels d op r 0 then r
      else
        begin match expand ~by ?at op with
        | Some y -> y
        | None ->
            let (module K : Nx_kernel.S) = kernels in
            refused ~by ~kernels:K.name d op
        end
  | Some k ->
      on_device ~by kernels set op r 0 k op;
      r
  | None ->
      let rule = Prim.rule op and first = L.shape !first in
      Array.iteri
        (fun j k ->
          let w = Devices.window ~by p first j in
          let view i x =
            let at = Devices.one (Devices.set (Prim.placement x)) k in
            let w' = Route.window rule ~operand:i (Prim.shape x) w in
            let a = Place.view ~by x k w' in
            Value.Array { at; a; dead = Value.Live }
          in
          on_device ~by kernels set op r j k (Prim.local (Prim.mapi view op) w))
        (Grid.devices (Devices.grid p));
      r

(* [local], [op] on device [k] of [set] at position [j], into [op]'s results
   [r]: by the kernels, or where they decline by the expansion copied there. *)
and on_device : type r.
    by:string ->
    (module Nx_kernel.S) ->
    unit Devices.t ->
    r Value.prim ->
    r ->
    int ->
    int ->
    r Value.prim ->
    unit =
 fun ~by kernels set op r j k local ->
  let d = Devices.rig set k and q = Devices.one set k in
  if not (call ~by kernels d local r j) then
    match expand ~by ~at:q local with
    | Some y -> copied ~by kernels q local y op r j
    | None ->
        let (module K : Nx_kernel.S) = kernels in
        refused ~by ~kernels:K.name d local

(* [y], [local]'s results at [q], copied into the results [r] of [op] at
   position [j]. A result may be a constant, a creation that an expansion of
   one of the expansion's operations defers: it is computed at [q]. *)
and copied : type r.
    by:string ->
    (module Nx_kernel.S) ->
    unit Devices.placement ->
    r Value.prim ->
    r ->
    r Value.prim ->
    r ->
    int ->
    unit =
 fun ~by kernels q local y op r j ->
  let (module K) = kernels in
  let dsts = Prim.result_arrays op r in
  let srcs = ref [] in
  Prim.iter_results
    (fun x ->
      let x = if Prim.is_constant x then at (Devices.rebrand q) x else x in
      srcs := A.Any (one x) :: !srcs)
    local y;
  List.iteri
    (fun i (A.Any src) ->
      let (A.Any dst) = dsts.(i).(j) in
      let v = A.expect (A.dtype dst) (A.Any src) in
      if not (ran ~by (copy_into kernels ~dst v) [| A.Any dst |] [| A.Any v |])
      then
        declined ~by ~kernels:K.name
          (Op1 (Copy, D.Any (A.dtype v), 0))
          (A.device v) [ A.Any v ])
    (List.rev !srcs)

(* [op]'s expansion, its operations applied where their operands lie, or at
   [at] for a constant's evaluation; [None] for a core case. *)
and expand : type r.
    by:string -> ?at:unit Devices.placement -> r Value.prim -> r option =
 fun ~by ?at op ->
  match at with
  | None -> Expand.run run ~by op
  | Some q -> Expand.run (fun ~by op -> apply_at q ~by op) ~by op

(* An operation of a computation by partials ({!Partial}), applied where its
   operands lie, or at [at] for a constant's evaluation. *)
and folded : type r. by:string -> ?at:unit Devices.placement -> r Value.prim -> r
    =
 fun ~by ?at op ->
  match at with None -> run ~by op | Some q -> apply_at q ~by op

(* An operation of a constant's evaluation at [q]: one over a concrete operand
   computes where its route reads them; a creation, its other operands
   constants, computes at [q], where the evaluation reads it. *)
and apply_at : type r. unit Devices.placement -> by:string -> r Value.prim -> r
    =
 fun q ~by op ->
  if Prim.exists is_concrete op then run ~by op
  else compute ~by ~at:q (Prim.map (fun y -> at (Devices.rebrand q) y) op)

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
      make p ~shape:(L.shape f.layout)
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
  if Array.mem 0 shape then ()
  else
    (* [ok] reduces where it lies, so only scalars reach the host. A passing
       check runs one [Min]; [Arg], for the first [false] in C order, costs
       more and runs only on a failure. *)
    let reduce r =
      run ~by
        (Value.Reduce
           {
             layout = L.contiguous shape;
             axes = Array.init (Array.length shape) Fun.id;
             prog = P.of_node ~ins:[| D.Any D.Bool |] (In 0);
             reductions = Value.[ r ];
             loads = [| Plain ok |];
           })
    in
    let scalar x = A.get (on_host ~by x) [||] in
    let all, () = reduce (Monoid (Min, 0, D.Bool)) in
    if not (scalar all) then begin
      let (_, first), () = reduce (Arg (Min, 0, D.Bool)) in
      let idx = unravel (Int64.to_int (scalar first)) shape in
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
    end

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
      (* Each view and where it is read, the innermost first. A view, a Move
         or a Bitcast, keeps no results of its own: it is read through to its
         operand's. *)
      let views = ref [] and node = ref node in
      let key = ref (Devices.rebrand p) in
      let continue = ref true in
      while !continue do
        let (Value.Node n) = !node in
        match n.op with
        | Value.Move (_, Value.Deferred { node = m; _ })
        | Value.Bitcast (_, Value.Deferred { node = m; _ }) ->
            views := (!node, !key) :: !views;
            key := read_at (route_at ~by:n.by n.op !key) !key 0;
            node := m
        | Value.Move _ | Value.Bitcast _ | Value.Map _ | Value.Reduce _
        | Value.Scan _ | Value.Gather _ | Value.Scatter _ | Value.Sort _
        | Value.Fft _ | Value.Linalg _ | Value.Fold _ | Value.Assemble _
        | Value.Contract _ | Value.Copy _ | Value.Place _ | Value.Check _ ->
            continue := false
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
              make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
          | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
              y
        in
        compute_node ~resolve:operand view key
      in
      let arrays = List.fold_left replay base !views in
      make p ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))

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
      make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
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
        make q ~shape:(L.shape form.layout) (typed form.dtype arrays.(k))
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
      [| Prim.any (Iarray.map (moved ~by (Devices.set p) mv) xs) |]
  | Value.Bitcast (dt, x) when uncut p ->
      let xs = Prim.arrays (resolve (Devices.rebrand p) x) in
      [| Prim.any (Iarray.map (cast ~by (Devices.set p) dt) xs) |]
  | op ->
      (* Its rule held when it was made. It computes at its route's result, [p]
         with the axes its rule keeps whole made whole, over operands where the
         route reads them; a creation computes there too. *)
      let route = route_at ~by op p in
      let q = match route with None -> p | Some r -> r.result in
      let op' =
        Prim.mapi
          (fun i y -> resolve (Devices.rebrand (read_at route p i)) y)
          op
      in
      let r = compute ~by ~at:q op' in
      let arrays = Prim.result_arrays op' r in
      if Devices.equal q p then arrays
      else
        (* Each device of [p] keeps its window, cut from its window at [q], in
           memory of its own, so that no memo holds more. *)
        let shapes = ref [] in
        Prim.iter_results (fun y -> shapes := Prim.shape y :: !shapes) op' r;
        let shapes = Array.of_list (List.rev !shapes) in
        let qs = Grid.devices (Devices.grid q)
        and ps = Grid.devices (Devices.grid p) in
        Array.map2
          (fun per shape ->
            Array.mapi
              (fun j k ->
                let jq = Option.get (Array.find_index (( = ) k) qs) in
                let wp = Devices.window ~by p shape j in
                let wq = Devices.window ~by q shape jq in
                let w =
                  Array.map2
                    (fun (rp : M.range) (rq : M.range) ->
                      { rp with M.start = rp.start - rq.start })
                    wp wq
                in
                let (A.Any a) = per.(jq) in
                A.Any (A.copy (Option.get (A.move (Slice w) a))))
              ps)
          arrays shapes

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

(* The one-node maps, built and run: the paths off the fast one. *)
let run1 ~by k dt x =
  let v, () = run ~by (Prim.op1 ~by k dt x) in
  v

let run2 ~by k dt x y =
  let v, () = run ~by (Prim.op2 ~by k dt x y) in
  v

let run3 ~by k c x y =
  let v, () = run ~by (Prim.op3 ~by k c x y) in
  v

let apply1 (type v s w r d) ~by k (dt : (w, r) D.t)
    (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match x with
  | Value.Array rx when rx.dead == Value.Live -> (
      let at = rx.at and a = rx.a in
      match Devices.kernels (Devices.set at) with
      | None -> run1 ~by k dt x
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply1 k ~dst a with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype -> run1 ~by k dt x
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ]))
  | Value.Donated dx when handle_at dx.at x ->
      let (module K) = kernels_at dx.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          match K.apply1 k ~dst a with
          | Done -> value_of dx.at dst
          | Declined | Wrong_dtype -> run1 ~by k dt (value_of dx.at a)
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ])
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _
  | Value.Traced _ ->
      run1 ~by k dt x

(* [k] into [dst] over the arrays [a] and [b] at [at]: [Done] is the result, a
   decline or a dtype or shape refusal the operation run over them as live
   values. *)
let kernel2 (type v s w r d) ~by (module K : Nx_kernel.S) k
    (dt : (w, r) D.t) (at : d Devices.placement) (dst : (w, r) A.t)
    (a : (v, s) A.t) (b : (v, s) A.t) : (w, r, d) Value.t =
  match K.apply2 k ~dst a b with
  | Done -> value_of at dst
  | Declined | Wrong_dtype | Shape_mismatch ->
      run2 ~by k dt (value_of at a) (value_of at b)
  | refusal -> A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]

(* As [kernel2], for [apply3]. *)
let kernel3 (type a b v s d) ~by (module K : Nx_kernel.S) k
    (at : d Devices.placement) (dst : (v, s) A.t) (c : (a, b) A.t)
    (x : (v, s) A.t) (y : (v, s) A.t) : (v, s, d) Value.t =
  match K.apply3 k ~dst c x y with
  | Done -> value_of at dst
  | Declined | Wrong_dtype | Shape_mismatch ->
      run3 ~by k (value_of at c) (value_of at x) (value_of at y)
  | refusal -> A.refused by refusal [ A.Any dst; A.Any c; A.Any x; A.Any y ]

(* Whether the one-node map [k] over [x] and [y] is well formed. A constant is
   read for an operation only once its rule holds, so that a refused operation
   computes nothing. *)
let well_formed2 k x y = P.accepts2 k (Prim.dtype x) && Prim.same_shape x y

let well_formed3 k c x y =
  P.accepts3 k (Prim.dtype c) (Prim.dtype x)
  && Prim.same_shape c x && Prim.same_shape x y

let rec apply2 : type v s w r d.
    by:string ->
    P.op2 ->
    (w, r) D.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (w, r, d) Value.t =
 fun ~by k dt x y ->
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rx.at and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> run2 ~by k dt x y
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply2 k ~dst a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype | Shape_mismatch -> run2 ~by k dt x y
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]))
  | Value.Donated dx, Value.Array ry when handle_at ry.at x && live_at ry.at y
    ->
      let (module K) = kernels_at ry.at in
      donated ~by 0 dt x (Iarray.get dx.arrays 0) (fun dst a ->
          kernel2 ~by (module K) k dt ry.at dst a ry.a)
  | Value.Array rx, Value.Donated dy when handle_at rx.at y && live_at rx.at x
    ->
      let (module K) = kernels_at rx.at in
      donated ~by 1 dt y (Iarray.get dy.arrays 0) (fun dst b ->
          kernel2 ~by (module K) k dt rx.at dst rx.a b)
  | Value.Array rx, Value.Deferred _
    when rx.dead == Value.Live && well_formed2 k x y ->
      (* A constant beside a value is read where the value lies. *)
      apply2 ~by k dt x (at rx.at y)
  | Value.Deferred _, Value.Array ry
    when ry.dead == Value.Live && well_formed2 k x y ->
      apply2 ~by k dt (at ry.at x) y
  | _ -> run2 ~by k dt x y

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
    by:string ->
    P.op3 ->
    (a, b, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t ->
    (v, s, d) Value.t =
 fun ~by k c x y ->
  match (c, x, y) with
  | Value.Array rc, Value.Array rx, Value.Array ry
    when rc.at == rx.at && rc.at == ry.at && rc.dead == Value.Live
         && rx.dead == Value.Live && ry.dead == Value.Live -> (
      let at = rc.at and ca = rc.a and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> run3 ~by k c x y
      | Some (module K) -> (
          let dst = destination (A.dtype a) a in
          match K.apply3 k ~dst ca a b with
          | Done -> Value.Array { at; a = dst; dead = Value.Live }
          | Declined | Wrong_dtype | Shape_mismatch -> run3 ~by k c x y
          | refusal ->
              A.refused by refusal [ A.Any dst; A.Any ca; A.Any a; A.Any b ]))
  | Value.Donated dc, Value.Array rx, Value.Array ry
    when handle_at rx.at c && live_at rx.at x && live_at rx.at y ->
      let (module K) = kernels_at rx.at in
      donated ~by 0 (A.dtype rx.a) c (Iarray.get dc.arrays 0)
        (fun dst c' -> kernel3 ~by (module K) k rx.at dst c' rx.a ry.a)
  | Value.Array rc, Value.Donated dx, Value.Array ry
    when handle_at rc.at x && live_at rc.at c && live_at rc.at y ->
      let (module K) = kernels_at rc.at in
      donated ~by 1 (A.dtype ry.a) x (Iarray.get dx.arrays 0)
        (fun dst x' -> kernel3 ~by (module K) k rc.at dst rc.a x' ry.a)
  | Value.Array rc, Value.Array rx, Value.Donated dy
    when handle_at rc.at y && live_at rc.at c && live_at rc.at x ->
      let (module K) = kernels_at rc.at in
      donated ~by 2 (A.dtype rx.a) y (Iarray.get dy.arrays 0)
        (fun dst y' -> kernel3 ~by (module K) k rc.at dst rc.a rx.a y')
  | (Value.Deferred _, _, _ | _, Value.Deferred _, _ | _, _, Value.Deferred _)
    when well_formed3 k c x y -> (
      match beside3 c x y with
      | Some p -> apply3 ~by k (at p c) (at p x) (at p y)
      | None -> run3 ~by k c x y)
  | _ -> run3 ~by k c x y

(* The contraction of live arrays on one device, at physically one placement
   whose set has kernels, computed by them without building the operation.
   [None] elsewhere, and where the kernels decline it: the operation decides
   those. *)
let contract (type v s a b c e d) ~by ~order ~shape ~batch ~contracting ~acc
    (dt : (v, s) D.t)
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
          let dst = A.create (Devices.rig set k) dt shape in
          match
            K.contract ~order ~batch ~contracting ~acc ~dst ~init:array rx.a
              ry.a
          with
          | Done -> Some (Value.Array { at; a = dst; dead = Value.Live })
          | Declined -> None
          | refusal ->
              let init = Option.to_list (Option.map (fun i -> A.Any i) array) in
              A.refused by refusal
                (A.Any dst :: A.Any rx.a :: A.Any ry.a :: init))
      | _ -> None)
  | _ -> None
