(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module S = Nx_kernel.Spec

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Messages *)

let declined ~by ~kernels node d dts =
  invalid_argf "%s: %s does not compute %s on %s (%s)" by kernels
    (Prim.kind node)
    (String.concat ", " (List.map (fun (A.Any a) -> D.name (A.dtype a)) dts))
    (Rig.name d)

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

(* Values from arrays *)

let make = Prim.of_arrays

let arrays_of (type v s d) (x : (v, s, d) Value.t) : (v, s) A.t array =
  match x with
  | Value.Array { a; _ } -> [| a |]
  | Value.Shards { arrays; _ } | Value.Donated { arrays; _ } -> arrays
  | Value.Deferred _ -> invalid_arg "Exec: a constant has no arrays"
  | Value.Traced _ -> invalid_arg "Exec: a traced value has no arrays"

(* A live value: its death word is [""]. *)
let live_word w = String.length w = 0

(* Raises unless operand [i], [x], is alive and concrete or a constant: an
   interpretation receives every operation on its traced values before eager
   execution can. *)
let checked ~by i (Value.Any x) =
  match x with
  | Value.Traced { owner; _ } ->
      invalid_argf "%s: a value traced by %s reached eager execution" by
        owner.name
  | Value.Donated d when live_word d.chain.consumer && not (live_word d.spent)
    ->
      (* A handle passed on and read again: the chain read twice. *)
      invalid_argf "Nx.donate: %s reads one donation twice" by
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _ ->
      Prim.alive ~by i x

(* Donation *)

let donations = Atomic.make 0

let donate (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  Prim.alive ~by 0 x;
  let chain claim =
    {
      Value.origin = Atomic.fetch_and_add donations 1;
      claim;
      consumer = Prim.live;
    }
  in
  match x with
  | Value.Array r ->
      let claim why =
        if Atomic.Loc.compare_and_set [%atomic.loc r.dead] Prim.live why then ""
        else r.dead
      in
      Value.Donated
        {
          at = r.at;
          arrays = [| r.a |];
          chain = chain claim;
          spent = Prim.live;
        }
  | Value.Shards r ->
      let claim why =
        if Atomic.Loc.compare_and_set [%atomic.loc r.dead] Prim.live why then ""
        else r.dead
      in
      Value.Donated
        { at = r.at; arrays = r.arrays; chain = chain claim; spent = Prim.live }
  | Value.Donated _ | Value.Deferred _ | Value.Traced _ -> x

(* A handle's arrays as a live value, which the operation reads. *)
let live (type v s d) (x : (v, s, d) Value.t) : (v, s, d) Value.t =
  match x with
  | Value.Donated { at; arrays; _ } -> Prim.of_arrays at arrays
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> x

type handle = Handle : ('v, 's, 'd) Value.t -> handle

(* The donated operands among [xs]. Two handles of one donation, or over one
   memory, raise naming [Nx.donate]. *)
let handles ~by xs =
  let hs =
    List.filter_map
      (fun (Value.Any x) ->
        match x with
        | Value.Donated _ -> Some (Handle x)
        | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ ->
            None)
      xs
  in
  let rec check = function
    | [] -> ()
    | Handle x :: rest ->
        List.iter
          (fun (Handle y) ->
            let same =
              match (x, y) with
              | Value.Donated a, Value.Donated b ->
                  a.chain.origin = b.chain.origin
                  || Array.exists
                       (fun p ->
                         Array.exists
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
let buffers (Handle x) = Array.to_list (Array.map A.buffer (arrays_of x))

(* Claims the handles [hs] for their consumer [by], before it reads them: each
   chain, then its donor, dies naming [by]. A chain or donor another consumer
   claimed first, on any domain, raises naming it, with nothing computed. *)
let claim_handle (type v s d) ~by (x : (v, s, d) Value.t) =
  match x with
  | Value.Donated d ->
      let lost earlier =
        invalid_argf "%s: a donated operand was donated to %s" by earlier
      in
      if
        not
          (Atomic.Loc.compare_and_set [%atomic.loc d.chain.consumer] Prim.live
             by)
      then lost d.chain.consumer;
      let earlier = d.chain.claim by in
      if not (live_word earlier) then lost earlier;
      d.spent <- by
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ()

let claim ~by hs = List.iter (fun (Handle x) -> claim_handle ~by x) hs

(* [b]'s memory consumed for [by], where rig holds it exclusive: no other value
   reads it. The live buffer over it, if so. *)
let take ~by b =
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      if Rig.Claim.exclusive c b then
        Some (Rig.Claim.consume c ~why:("donated to " ^ by) b)
      else None)

(* Ends the claimed handles [hs] once [by] read them: memory no other value
   reads, and that [by] did not write its result into, is consumed. *)
let consumed ~by ~reused hs =
  List.iter
    (fun (Handle x as h) ->
      match x with
      | Value.Donated _ ->
          if not (reused h) then begin
            let bufs = buffers h in
            Rig.Claim.with_ ~read:[] ~donate:[ bufs ] (fun c ->
                if List.for_all (Rig.Claim.exclusive c) bufs then
                  List.iter
                    (fun b ->
                      ignore (Rig.Claim.consume c ~why:("donated to " ^ by) b))
                    bufs)
          end
      | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> ())
    hs

let origin_is origin (Handle y) =
  match y with
  | Value.Donated d -> d.chain.origin = origin
  | Value.Array _ | Value.Shards _ | Value.Deferred _ | Value.Traced _ -> false

(* The load of a map whose memory takes the map's first result, if one may: a
   handle on one device, of the result's dtype, C-contiguous with the map's
   layout over all of its memory, beside loads at its placement, and whose
   memory no other value reads (rig holds it exclusive). Its memory is then
   consumed, naming [by], and its index, chain and new array are the answer. *)
let reuse (type d r) ~by layout (outs : (d, r) Value.outs)
    (loads : d Value.load array) =
  match outs with
  | [] -> None
  | dt :: _ -> (
      let fits (Value.Plain y) =
        match y with
        | Value.Donated { at; arrays = [| a |]; _ } ->
            D.equal (A.dtype a) dt
            && L.equal (A.layout a) layout
            && Rig.Buffer.spans (A.buffer a)
            && Array.for_all
                 (fun (Value.Plain z) ->
                   match Prim.at z with
                   | Some q -> Devices.rebrand q == at
                   | None -> true)
                 loads
        | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _
        | Value.Traced _ ->
            false
      in
      match Array.find_index fits loads with
      | None -> None
      | Some i -> (
          match loads.(i) with
          | Value.Plain (Value.Donated d) ->
              let a = d.arrays.(0) in
              let b = A.buffer a in
              Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
                  if not (Rig.Claim.exclusive c b) then None
                  else
                    let b' = Rig.Claim.consume c ~why:("donated to " ^ by) b in
                    Some
                      ( i,
                        d.chain.origin,
                        A.Any (A.v (A.dtype a) (A.layout a) b') ))
          | Value.Plain _ -> None))

(* A view made over [a]'s memory, outside any handle chain: the memory is shared
   from then on, so that no donation of [a]'s value writes what the view
   reads. *)
let view_of a = Rig.Claim.share (A.buffer a)

(* Whether [mv] maps the elements of a value of [shape] one to one, which passes
   a handle on. *)
let bijective (mv : M.t) shape =
  match mv with
  | Reshape _ | Permute _ -> true
  | Broadcast s -> s = shape
  | Slice rs ->
      Array.for_all2
        (fun (r : M.range) n -> r.count = n && (r.step = 1 || r.step = -1))
        rs shape
  | Window _ -> false

let extents (w : M.range array) = Array.map (fun (r : M.range) -> r.count) w
let starts (w : M.range array) = Array.map (fun (r : M.range) -> r.start) w

(* Fresh C-contiguous arrays of a value of [shape] at [p]: each device's
   window. *)
let fresh ~by p dt shape =
  let set = Devices.set p in
  match Grid.one (Devices.grid p) with
  | Some k -> [| A.create (Devices.rig set k) dt shape |]
  | None ->
      Array.mapi
        (fun j k ->
          A.create (Devices.rig set k) dt
            (extents (Devices.window ~by p shape j)))
        (Grid.devices (Devices.grid p))

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
  | Some a when k = 0 -> make p [| A.expect f.dtype a |]
  | Some _ | None -> make p (fresh ~by p f.dtype (L.shape f.layout))

(* Programs on one device *)

(* [prog] with each [Coord] of an axis whose window starts at [s > 0] replaced
   by [Coord + s], so that a device computes its window. *)
let with_offsets prog (first : int array) =
  let r = Array.length first in
  let shifted = function P.Coord c -> first.(r - 1 - c) <> 0 | _ -> false in
  if
    Array.for_all (fun s -> s = 0) first
    || not (List.exists shifted (List.init (P.length prog) (P.node prog)))
  then prog
  else begin
    let nodes = ref [] and n = ref 0 in
    let index = Array.make (P.length prog) 0 in
    let push nd =
      nodes := nd :: !nodes;
      incr n;
      !n - 1
    in
    for i = 0 to P.length prog - 1 do
      let at j = index.(j) in
      let nd : P.node =
        match P.node prog i with
        | Op1 (k, dt, j) -> Op1 (k, dt, at j)
        | Op2 (k, j, l) -> Op2 (k, at j, at l)
        | Op3 (k, j, l, m) -> Op3 (k, at j, at l, at m)
        | (In _ | Coord _ | Const _) as nd -> nd
      in
      index.(i) <-
        (match nd with
        | Coord c when first.(r - 1 - c) <> 0 ->
            let coord = push nd in
            let start =
              push
                (Const
                   ( D.Any D.Int64,
                     P.bits D.Int64 (Int64.of_int first.(r - 1 - c)) ))
            in
            push (Op2 (Binary Add, coord, start))
        | _ -> push nd)
    done;
    P.v ~ins:(P.ins prog)
      (Array.of_list (List.rev !nodes))
      ~outs:(Array.map (fun o -> index.(o)) (P.outs prog))
  end

(* The node a one-node program computes from its operands in order, if it is
   one: every node before its output reads operand [i] at index [i]. *)
let single prog =
  let n = P.length prog and outs = P.outs prog in
  let ins_first =
    let ok = ref true in
    for i = 0 to n - 2 do
      match P.node prog i with P.In j when j = i -> () | _ -> ok := false
    done;
    !ok
  in
  if Array.length outs <> 1 || outs.(0) <> n - 1 || not ins_first then None
  else
    let m = n - 1 in
    match P.node prog m with
    | (Const _ | Coord _) as nd -> Some nd
    | Op1 (_, _, 0) as nd when m = 1 -> Some nd
    | Op2 (_, 0, 1) as nd when m = 2 -> Some nd
    | Op3 (_, 0, 1, 2) as nd when m = 3 -> Some nd
    | _ -> None

(* [node]'s kernel into [dst], over [ops]. *)
let apply_node (module K : Nx_kernel.S) node (A.Any dst) (ops : A.any array) =
  let rank = L.rank (A.layout dst) in
  match (node : P.node) with
  | Const (_, bits) -> K.apply0 (Fill bits) ~dst
  | Coord c -> K.apply0 (Iota (rank - 1 - c)) ~dst
  | Op1 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply1 k ~dst x
  | Op2 (k, _, _) ->
      let (A.Any x) = ops.(0) in
      K.apply2 k ~dst x (A.expect (A.dtype x) ops.(1))
  | Op3 (k, _, _, _) ->
      let (A.Any c) = ops.(0) in
      let (A.Any x) = ops.(1) in
      K.apply3 k ~dst c x (A.expect (A.dtype x) ops.(2))
  | In _ -> Declined

(* [prog] over [ops] into [dsts] on device [d], whose window starts at [first]:
   its kernel where it is one node, else the kernels' map. [false] where the
   kernels decline the map; a one-node program they decline raises, since its
   node is core. *)
let map_on ~by (module K : Nx_kernel.S) d prog first ops dsts =
  let prog = with_offsets prog first in
  match single prog with
  | Some node ->
      ran ~by (apply_node (module K) node dsts.(0) ops) dsts ops
      || declined ~by ~kernels:K.name node d
           (Array.to_list (Array.append dsts ops))
  | None ->
      ran ~by
        (K.map
           (S.map prog ~loads:(Array.make (Array.length ops) S.Plain))
           ~dsts ops)
        dsts ops

let load_view (type d) ~by k w (Value.Plain x : d Value.load) =
  A.Any (Place.view ~by x k w)

(* [a] stored afresh, C-contiguous, on its device. *)
let copy_array ~by (module K : Nx_kernel.S) a =
  let dst = A.create (A.device a) (A.dtype a) (L.shape (A.layout a)) in
  if not (done_or_declined ~by (K.apply1 Copy ~dst a) [ A.Any dst; A.Any a ])
  then
    declined ~by ~kernels:K.name
      (Op1 (Copy, D.Any (A.dtype a), 0))
      (A.device a) [ A.Any a ];
  dst

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

(* The constant operation at [p] reads its operands at this placement: [p] for a
   map, whose operands have its shape, and for a placement that cuts no axis;
   the whole on every device otherwise. *)
let operand_at : type r.
    r Value.prim -> unit Devices.placement -> unit Devices.placement =
 fun op p ->
  match op with
  | Value.Map _ -> p
  | _ ->
      if Grid.cuts (Devices.grid p) = [||] then p
      else Devices.on (Devices.set p)

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
   check's flag, a value of every set an interpretation receives. No value at it
   reaches a function of nx's own. *)
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

let rec run : type r. by:string -> r Value.prim -> r =
 fun ~by op ->
  let (Prim.Operands xs) = Prim.operands op in
  List.iteri (checked ~by) xs;
  match handles ~by xs with [] -> plain ~by op | hs -> donated ~by op hs

(* [op] over donated operands [hs]: a movement that maps elements one to one
   passes a new handle on; any other operation reads the handles' memory, writes
   its first result into a handle's memory where it may, and ends the
   handles. *)
and donated : type r. by:string -> r Value.prim -> handle list -> r =
 fun ~by op hs ->
  match op with
  | Value.Move (mv, (Value.Donated d as x)) when bijective mv (Prim.shape x) ->
      let r = run ~by (Value.Move (mv, live x)) in
      d.spent <- by;
      Value.Donated
        {
          at = Prim.placement r;
          arrays = arrays_of r;
          chain = d.chain;
          spent = Prim.live;
        }
  | Value.Map ({ loads; _ } as m) -> (
      claim ~by hs;
      match reuse ~by m.layout m.outs loads with
      | Some (i, origin, a) ->
          (* Load [i] is the result's memory: the kernel reads each index of it
             before it writes the result there. *)
          let load j (Value.Plain y) =
            if j <> i then Value.Plain (live y)
            else
              let at = Prim.placement y in
              Value.Plain
                (Value.Array
                   { at; a = A.expect (Prim.dtype y) a; dead = Prim.live })
          in
          let r =
            plain ~by ~into:a
              (Value.Map { m with loads = Array.mapi load loads })
          in
          consumed ~by ~reused:(origin_is origin) hs;
          r
      | None ->
          let r = run ~by (Prim.map live op) in
          consumed ~by ~reused:(fun _ -> false) hs;
          r)
  | Value.Copy _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      claim ~by hs;
      let r = run ~by (Prim.map live op) in
      consumed ~by ~reused:(fun _ -> false) hs;
      r

and plain : type r. by:string -> ?into:A.any -> r Value.prim -> r =
 fun ~by ?into op ->
  let (Prim.Operands xs) = Prim.operands op in
  match op with
  | Value.Place (p, x) -> (
      match x with
      | Value.Deferred _ -> Place.value ~by p (own (Devices.rebrand p) x)
      | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
          let y = Place.value ~by p x in
          (* A value over [x]'s memory, kept or borrowed, shares it. *)
          let xs = arrays_of x in
          Array.iter
            (fun b ->
              if
                Array.exists
                  (fun a -> Rig.Buffer.overlaps (A.buffer a) (A.buffer b))
                  xs
              then Rig.Claim.share (A.buffer b))
            (arrays_of y);
          y)
  | Value.Check { ok; data; fail } ->
      Prim.results ~by (fun k f -> alloc ~by k f) op;
      check ~by ok data fail
  | _ ->
      if List.for_all (fun (Value.Any x) -> Prim.is_constant x) xs then
        defer ~by op
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

and compute : type r. by:string -> ?into:A.any -> r Value.prim -> r =
 fun ~by ?into op ->
  match op with
  | Value.Map { layout; prog; loads; _ } ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ~where ?into k f) op in
      map_devices ~by (Option.get !where) (L.shape layout) prog
        (fun k w -> Array.map (load_view ~by k w) loads)
        (Prim.arrays op r);
      r
  | Value.Copy x ->
      let where = ref None in
      let r = Prim.results ~by (fun k f -> alloc ~by ~where k f) op in
      let prog =
        Prim.program
          (Op1 (Copy, D.Any (Prim.dtype x), 0))
          [| D.Any (Prim.dtype x) |]
      in
      map_devices ~by (Option.get !where) (Prim.shape x) prog
        (fun k w -> [| A.Any (Place.view ~by x k w) |])
        (Prim.arrays op r);
      r
  | Value.Move (mv, x) ->
      let xp = Prim.placement x and shape = Prim.shape x and xs = arrays_of x in
      let cuts = Grid.cuts (Devices.grid xp) in
      let moved a =
        let local = localize mv shape cuts (L.shape (A.layout a)) in
        match A.move local a with
        | Some v ->
            view_of a;
            v
        | None ->
            let kernels = kernels_of ~by ~op:"Copy" (Devices.set xp) in
            Option.get (A.move local (copy_array ~by kernels a))
      in
      shard_views ~by op xp (fun () -> Array.map moved xs)
  | Value.Bitcast (dt, x) ->
      let xp = Prim.placement x and xs = arrays_of x in
      let cast a =
        match A.bitcast dt a with
        | Some v ->
            view_of a;
            v
        | None ->
            let kernels = kernels_of ~by ~op:"Copy" (Devices.set xp) in
            Option.get (A.bitcast dt (copy_array ~by kernels a))
      in
      shard_views ~by op xp (fun () -> Array.map cast xs)
  | Value.Place _ | Value.Check _ -> run ~by op

(* The one result of [op] over the arrays [views ()], one per device of [xp] in
   order: at [op]'s placement, whose devices hold them. [views] runs once [op]'s
   rule holds, so that no kernel copies for an operation it refuses. *)
and shard_views : type v s d.
    by:string ->
    (v, s, d) Value.t Value.prim ->
    d Devices.placement ->
    (unit -> (v, s) A.t array) ->
    (v, s, d) Value.t =
 fun ~by op xp views ->
  let xdev = Grid.devices (Devices.grid xp) in
  Prim.results ~by
    (fun _ f ->
      let vs = views () in
      (* Its operand lies at [xp], so it has a placement. *)
      let p = Option.get f.placement in
      make p
        (Array.map
           (fun k ->
             let j = Option.get (Array.find_index (( = ) k) xdev) in
             A.expect f.dtype (A.Any vs.(j)))
           (Grid.devices (Devices.grid p))))
    op

and check : type d.
    by:string ->
    (bool, D.bool_elt, d) Value.t ->
    d Value.any list ->
    (int array -> d Value.any list -> exn) ->
    unit =
 fun ~by ok data fail ->
  let host : d Devices.placement = host () in
  let on_host (type v s) (x : (v, s, d) Value.t) : (v, s) A.t =
    match x with
    | Value.Deferred _ -> (arrays_of (at host x)).(0)
    | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ ->
        (arrays_of (Place.value ~by host x)).(0)
  in
  let shape = Prim.shape ok in
  match Array.find_index not (A.to_array (on_host ok)) with
  | None -> ()
  | Some i ->
      let idx = unravel i shape in
      let at_idx (Value.Any x) =
        let a = on_host x in
        let one =
          Array.map (fun i -> { M.start = i; count = 1; step = 1 }) idx
        in
        let v =
          Option.get (A.move (Reshape [||]) (Option.get (A.move (Slice one) a)))
        in
        Value.Any (Value.Array { at = host; a = v; dead = Prim.live })
      in
      raise (fail idx (List.map at_idx data))

(* A constant read at [p] by an operation: its results there, kept in its memo,
   which every later read at [p] finds. *)
and at : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x ->
  match x with
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  | Value.Deferred { form; node = Value.Node n as node; k } ->
      let key = Devices.rebrand p in
      let arrays =
        match find n.memo key with
        | Some a -> a
        | None ->
            let a = evaluate node key in
            remember n.memo key a;
            Option.get (find n.memo key)
      in
      make p (Array.map (A.expect form.dtype) arrays.(k))

(* The constant [c] computed at [p] into memory of its own, its operands taken
   from their memos or computed for this alone: a value given to a caller shares
   no memory with a memo. *)
and own : type v s d.
    d Devices.placement -> (v, s, d) Value.t -> (v, s, d) Value.t =
 fun p x ->
  match x with
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _ -> x
  | Value.Deferred { form; node; k } ->
      let arrays = evaluate node (Devices.rebrand p) in
      make p (Array.map (A.expect form.dtype) arrays.(k))

(* [root]'s results at [p], fresh: the constants it reads are taken from their
   memos where an operation read them at the placement it reads them, and
   computed otherwise in a table of this evaluation's alone, dropped after. Each
   node computes once in an evaluation, in dependency order, with a stack of its
   own: a chain of any length takes no stack depth. *)
and evaluate root p =
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
        make q (Array.map (A.expect form.dtype) arrays.(k))
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
      (fun (A.Any a) -> Value.Plain (Value.Array { at; a; dead = Prim.live }))
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
  let on_device ~by op = compute_at ~by ~resolve:(fun _ x -> x) at op in
  match Expand.run on_device ~by op with
  | None -> invalid_arg "Exec.expand_on: a map of one node"
  | Some r ->
      let (module K) = kernels_of ~by ~op:"Copy" set in
      Array.iteri
        (fun i per ->
          let (A.Any dst) = dsts.(i) in
          let (A.Any v) = per.(0) in
          if not (ran ~by (K.apply1 Copy ~dst v) [| A.Any dst |] [| A.Any v |])
          then
            declined ~by ~kernels:K.name
              (Op1 (Copy, D.Any (A.dtype v), 0))
              (Devices.rig set k) [ A.Any v ])
        (Prim.arrays op r)

(* A map [op] computed at [p], its results there: its operands lie at [p] or are
   constants, and its rule holds. Any other operation is [run]'s. *)
and compute_at : type r.
    by:string ->
    resolve:
      ('v 's 'd.
       'd Devices.placement -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t) ->
    unit Devices.placement ->
    r Value.prim ->
    r =
 fun ~by ~resolve p op ->
  match op with
  | Value.Map { loads; layout; prog; _ } ->
      let r = Prim.results ~by (fun k f -> alloc ~by ~at:p k f) op in
      let view k w (Value.Plain x) =
        A.Any (Place.view ~by (resolve (Devices.rebrand p) x) k w)
      in
      map_devices ~by p (L.shape layout) prog
        (fun k w -> Array.map (view k w) loads)
        (Prim.arrays op r);
      r
  | Value.Copy _ | Value.Move _ | Value.Bitcast _ | Value.Place _
  | Value.Check _ ->
      run ~by op

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
  | Value.Map _ ->
      (* Its loads are computed at [p] already. *)
      Prim.arrays n.op (compute_at ~by ~resolve p n.op)
  | op ->
      let q = operand_at op p in
      (* Its rule held when it was made, and its operands lie at [q]. *)
      let op' = Prim.map (fun y -> resolve (Devices.rebrand q) y) op in
      let arrays = Prim.arrays op' (compute ~by op') in
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

let read x = own (host ()) x

(* The fast path *)

(* A destination for a result of [dt] beside [x]: over [x]'s layout where it is
   C-contiguous from offset 0 and of [dt], fresh otherwise. *)
let destination (type v s w r) (dt : (w, r) D.t) (x : (v, s) A.t) : (w, r) A.t =
  let l = A.layout x in
  if L.is_contiguous l && L.offset l = 0 && D.equal dt (A.dtype x) then
    A.v dt l (Rig.Buffer.create (A.device x) (D.bytes dt (L.numel l)))
  else A.create (A.device x) dt (L.shape l)

(* The one-node map [k] over the handle [h], whose array is [a], and the live
   array [o], [first] if [h] is the first operand, on [at]'s one device: [h] is
   claimed, then its memory takes the result where rig holds it exclusive and it
   has the result's dtype and C-contiguous layout over all of its memory;
   otherwise the result is fresh and the memory, where no other value reads it,
   is consumed after the kernel. *)
let donated2 (type v s w r d) ~slow ~by k (dt : (w, r) D.t)
    (at : d Devices.placement) (h : (v, s, d) Value.t) (a : (v, s) A.t)
    (o : (v, s) A.t) ~first : (w, r, d) Value.t =
  let (module K) = Option.get (Devices.kernels (Devices.set at)) in
  let l = A.layout a in
  let reusable =
    L.is_contiguous l && L.offset l = 0 && Rig.Buffer.spans (A.buffer a)
  in
  claim_handle ~by h;
  let run_on (dst : (w, r) A.t) x y : (w, r, d) Value.t =
    match K.apply2 k ~dst x y with
    | A.Done -> Value.Array { at; a = dst; dead = Prim.live }
    | Declined | Wrong_dtype | Shape_mismatch ->
        let lx = Value.Array { at; a = x; dead = Prim.live }
        and ly = Value.Array { at; a = y; dead = Prim.live } in
        slow ~by k dt lx ly
    | refusal -> A.refused by refusal [ A.Any dst; A.Any x; A.Any y ]
  in
  let pair a' = if first then (a', o) else (o, a') in
  let fresh () =
    let x, y = pair a in
    let r = run_on (destination dt a) x y in
    ignore (take ~by (A.buffer a));
    r
  in
  match D.equal_witness (A.dtype a) dt with
  | Some Type.Equal when reusable -> (
      match take ~by (A.buffer a) with
      | Some b' ->
          let a' = A.v (A.dtype a) l b' in
          let x, y = pair a' in
          run_on a' x y
      | None ->
          let x, y = pair a in
          run_on (destination dt a) x y)
  | Some _ | None -> fresh ()

let apply1 (type v s w r d) ~slow ~by k (dt : (w, r) D.t)
    (x : (v, s, d) Value.t) : (w, r, d) Value.t =
  match x with
  | Value.Array rx when live_word rx.dead -> (
      let at = rx.at and a = rx.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k dt x
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply1 k ~dst a with
          | Done -> Value.Array { at; a = dst; dead = Prim.live }
          | Declined | Wrong_dtype -> slow ~by k dt x
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a ]))
  | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Deferred _
  | Value.Traced _ ->
      slow ~by k dt x

let apply2 (type v s w r d) ~slow ~by k (dt : (w, r) D.t)
    (x : (v, s, d) Value.t) (y : (v, s, d) Value.t) : (w, r, d) Value.t =
  match (x, y) with
  | Value.Array rx, Value.Array ry
    when rx.at == ry.at && live_word rx.dead && live_word ry.dead -> (
      let at = rx.at and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k dt x y
      | Some (module K) -> (
          let dst = destination dt a in
          match K.apply2 k ~dst a b with
          | Done -> Value.Array { at; a = dst; dead = Prim.live }
          | Declined | Wrong_dtype | Shape_mismatch -> slow ~by k dt x y
          | refusal -> A.refused by refusal [ A.Any dst; A.Any a; A.Any b ]))
  | Value.Donated dx, Value.Array ry
    when dx.at == ry.at && live_word ry.dead
         && Array.length dx.arrays = 1
         && Devices.kernels (Devices.set dx.at) <> None ->
      donated2 ~slow ~by k dt dx.at x dx.arrays.(0) ry.a ~first:true
  | Value.Array rx, Value.Donated dy
    when dy.at == rx.at && live_word rx.dead
         && Array.length dy.arrays = 1
         && Devices.kernels (Devices.set dy.at) <> None ->
      donated2 ~slow ~by k dt dy.at y dy.arrays.(0) rx.a ~first:false
  | _ -> slow ~by k dt x y

let apply3 (type a b v s d) ~slow ~by k (c : (a, b, d) Value.t)
    (x : (v, s, d) Value.t) (y : (v, s, d) Value.t) : (v, s, d) Value.t =
  match (c, x, y) with
  | Value.Array rc, Value.Array rx, Value.Array ry
    when rc.at == rx.at && rc.at == ry.at && live_word rc.dead
         && live_word rx.dead && live_word ry.dead -> (
      let at = rc.at and ca = rc.a and a = rx.a and b = ry.a in
      match Devices.kernels (Devices.set at) with
      | None -> slow ~by k c x y
      | Some (module K) -> (
          let dst = destination (A.dtype a) a in
          match K.apply3 k ~dst ca a b with
          | Done -> Value.Array { at; a = dst; dead = Prim.live }
          | Declined | Wrong_dtype | Shape_mismatch -> slow ~by k c x y
          | refusal ->
              A.refused by refusal [ A.Any dst; A.Any ca; A.Any a; A.Any b ]))
  | _ -> slow ~by k c x y
