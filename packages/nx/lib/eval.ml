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

(* [x] placed at [p]. Onto the host from another set it is a wait: on [x]'s
   set first, then for the copies that brought the elements, so that it
   returns with them there. *)
let placed ~by p x =
  if Devices.on_host_set p && not (Devices.on_host_set (Prim.placement x))
  then begin
    Check.wait ~by x;
    let y = Exec.place ~by p x in
    Iarray.iter (fun a -> Rig.Buffer.wait (A.buffer a) Read) (Prim.arrays y);
    y
  end
  else Exec.place ~by p x

(* Memory a donation gives a result: one handle's buffer and the array the
   result takes over it, or over several devices each shard's. *)
type taken = One of Rig.Buffer.t * A.any | Cut of (Rig.Buffer.t * A.any) array

(* The memory of a handle [c] claimed that [op]'s first result may take
   ({!Donate.reuse}, {!Donate.reuse_cut} over several devices): the
   handle's buffers, dead from then on, and the arrays the result takes
   over their memory. A result written there lies where the handle does,
   so only an operand that lies where [r] puts the results may hold it. A
   route that computes by partials writes its result from the closing, so
   none takes the memory. *)
let taken ~by c (r : Prim.routed) op =
  match (Prim.own_index op, r.route) with
  | ( Some (own, layout, Nx_array.Dtype.Any dt),
      Some ({ compute = Windows; _ } as route) ) -> (
      let (Prim.Operands xs) = Prim.operands op in
      let xs = Array.of_list xs in
      let p = route.result in
      let at_result i =
        let (Value.Any x) = xs.(i) in
        Devices.equal (Prim.placement x) p
      in
      let own = List.filter at_result own in
      if Grid.count (Devices.grid p) = 1 then
        Option.map
          (fun (b, a) -> One (b, A.Any a))
          (Donate.reuse c ~own dt layout)
      else
        let shape = Iarray.to_array (Nx_array.Layout.shape layout) in
        let window j _ =
          Nx_array.Layout.contiguous
            (Route.extents (Devices.window ~by p shape j))
        in
        let layouts = Array.mapi window (Grid.devices (Devices.grid p)) in
        let any (b, a) = (b, A.Any a) in
        Option.map
          (fun ts -> Cut (Array.map any ts))
          (Donate.reuse_cut c ~own dt layouts))
  | Some _, Some { compute = Partials; _ } | Some _, None | None, _ -> None

(* Whether [checked] raises for [x]. *)
let unusable (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced _ -> true
  | Array _ | Shards _ | Handle _ -> (
      match Prim.death x with Live -> false | Died _ -> true)

(* Partitions *)

(* Whether each piece of [x] on [e] repeats its elements along axis [a], as a
   broadcast does: a piece's layout says so, an array's or a stand-in's
   form's. *)
let repeats (type v s d) (e : Partition.engine) (x : (v, s, d) t) a =
  let along l = A.Layout.numel l = 0 || A.Layout.stride l a = 0 in
  let at = Prim.placement x in
  let set = Devices.set at in
  let layout (y : (v, s, d) t) =
    match y with
    | Traced { form; _ } -> along form.layout
    | Array _ | Shards _ | Handle _ ->
        Iarray.for_all (fun arr -> along (A.layout arr)) (Prim.arrays y)
  in
  let ks = Grid.devices (Devices.grid at) in
  let rec all j =
    j >= Array.length ks
    || (layout (e.piece x j (Devices.one set ks.(j))) && all (j + 1))
  in
  all 0

let rec partition : type r. Partition.engine -> by:string -> r prim -> r =
 fun e ~by op ->
  match op with
  | Place (p, x) -> Partial.place e ~by p x
  | Check _ -> invalid_arg "Eval.partition: a check"
  | _ -> partitioned e ~by (Prim.route ~by op) op

(* [op] computed on [e] as its route [r] says, its first result into [into]
   where it is given. *)
and partitioned : type r d.
    Partition.engine -> by:string -> ?into:d any -> Prim.routed -> r prim -> r
    =
 fun e ~by ?into r op ->
  (* Every operation but a placement and a check has a route. *)
  let route = Option.get r.Prim.route in
  let op = gather e ~by route op in
  match route.compute with
  | Partials -> Partial.run e ~by route op
  | Windows -> Partition.windows e ~by ?into route r.layouts op

(* [op] with the positions [route] gathers moved where it reads them
   ({!Route.gathers}). Only a gather's and a scatter's positions, operand
   [0], and a range gather's starts and ends, operands [0] and [1], move, so
   other operations check nothing, and positions read where they lie check
   one placement each. *)
and gather : type r.
    Partition.engine -> by:string -> unit Route.t -> r prim -> r prim =
 fun e ~by route op ->
  match op with
  | Gather g when moves route 0 g.idx ->
      Gather { g with idx = moved e ~by route 0 g.idx }
  | Scatter s when moves route 0 s.idx ->
      Scatter { s with idx = moved e ~by route 0 s.idx }
  | Gather_ranges g when moves route 0 g.starts || moves route 1 g.ends ->
      let at i x = if moves route i x then moved e ~by route i x else x in
      let starts = at 0 g.starts in
      Gather_ranges { g with starts; ends = at 1 g.ends }
  | _ -> op

and moves : type v s d. unit Route.t -> int -> (v, s, d) t -> bool =
 fun route i x ->
  let p = Prim.placement x in
  (not (Grid.equal (Devices.grid p) (Devices.grid route.operands.(i))))
  && Route.gathers route i p (Prim.shape x)

(* [x] where [route] reads it. An axis that [x]'s placement does not cut and
   that each of its pieces repeats moves one element long, whole on each
   device, and is stretched back by a view, so that only [x]'s own elements
   move. *)
and moved : type v s d.
    Partition.engine ->
    by:string ->
    unit Route.t ->
    int ->
    (v, s, d) t ->
    (v, s, d) t =
 fun e ~by route i x ->
  let q = Devices.grid route.operands.(i) in
  let s = Prim.shape x in
  let at = Prim.placement x in
  let set = Devices.set at in
  let cuts = Grid.cuts (Devices.grid at) in
  let cut a = Array.exists (fun (c, _) -> c = a) cuts in
  let repeats a = s.(a) > 1 && (not (cut a)) && repeats e x a in
  let axes = Array.init (Array.length s) Fun.id in
  if not (Array.exists repeats axes) then
    Partition.place e ~by (Devices.v ~by set q) x
  else
    let one a =
      { A.Move.start = 0; count = (if repeats a then 1 else s.(a)); step = 1 }
    in
    let whole q a = if repeats a then Grid.uncut q ~axis:a else q in
    let q = Devices.v ~by set (Array.fold_left whole q axes) in
    let slice = A.Move.Slice (Iarray.init (Array.length s) one) in
    let core = partition e ~by (Move (slice, x)) in
    let core = Partition.place e ~by q core in
    partition e ~by (Move (Broadcast (Iarray.of_array s), core))

let rec run : type r. by:string -> r prim -> r =
 fun ~by op -> if Exec.is_local op then Exec.local ~by op else general ~by op

(* The raising walk, a closure over [by], is built only for an operand that
   fails. *)
and general : type r. by:string -> r prim -> r =
 fun ~by op ->
  if Prim.exists unusable op then Prim.iteri (fun _ x -> checked ~by x) op;
  if Prim.exists Donate.is_handle op then donated ~by op else plain ~by op

(* [op] over donated operands: a movement that maps elements one to one passes
   a new handle on; any other operation checks its rule and its route, then
   consumes the handles as {!Donate.with_consumer} does, reading their memory
   and writing its first result into a handle's memory where it may. *)
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
  | Map _ | Reduce _ | Scan _ | Gather _ | Gather_ranges _ | Scatter _
  | Sort _ | Unique _ | Fft _ | Linalg _ | Fold _ | Assemble _ | Contract _
  | Copy _ | Move _ | Bitcast _ | Place _ | Check _ | Attention _ -> (
      (* Routed before the claim: an operation its rule refuses neither takes
         a handle's memory nor shows another domain a handle dead. Once
         claimed, the donors stay dead whatever the kernels answer. *)
      let r = Prim.route ~by op in
      let (Prim.Operands xs) = Prim.operands op in
      Donate.with_consumer ~by xs (fun c ->
          let op = Donate.called c op in
          match taken ~by c r op with
          | Some (One (b, (A.Any a as into))) ->
              (* The handle's memory is the result's, now over [b']: the
                 kernel reads each index of it before it writes the result
                 there. *)
              let b' = A.buffer a in
              let as_taken (type v s d) (y : (v, s, d) t) : (v, s, d) t =
                match y with
                | Array r when A.buffer r.a == b ->
                    Array { r with a = A.v (A.dtype r.a) (A.layout r.a) b' }
                | Array _ | Shards _ | Handle _ | Traced _ -> y
              in
              routed ~by ~into r (Prim.map as_taken op)
          | Some (Cut ts) ->
              (* Each device's memory of the handle is the result's, now
                 over its [b'], read and written as on one device. *)
              let over (type v s) (a : (v, s) A.t) : (v, s) A.t =
                match Array.find_opt (fun (b, _) -> A.buffer a == b) ts with
                | Some (_, A.Any a') ->
                    A.v (A.dtype a) (A.layout a) (A.buffer a')
                | None -> a
              in
              let as_taken (type v s d) (y : (v, s, d) t) : (v, s, d) t =
                match y with
                | Array r -> Array { r with a = over r.a }
                | Shards r ->
                    Shards { r with arrays = Iarray.map over r.arrays }
                | Handle _ | Traced _ -> y
              in
              let cut = Array.map snd ts in
              routed ~by ~cut r (Prim.map as_taken op)
          | None -> routed ~by r op))

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
   computes, its first result into [into] where it is given, or over
   several devices into [cut], each device's array of its window. *)
and routed : type r.
    by:string ->
    ?into:A.any ->
    ?cut:A.any array ->
    Prim.routed ->
    r prim ->
    r =
 fun ~by ?into ?cut r op ->
  match op with
  | Place (p, x) -> placed ~by p x
  | Check { ok; data; fail } -> Check.issue ~by ok data fail
  | _ -> (
      match (into, cut) with
      | None, None -> partitioned Exec.eager ~by r op
      | Some (A.Any a), _ ->
          (* A value on the route's one device, over the donated memory. *)
          let at = (Option.get r.Prim.route).result in
          let into = Value.Any (Array { at; a; dead = Live }) in
          partitioned Exec.eager ~by ~into r op
      | None, Some arrays ->
          (* A value at the route's result, each device's window over its
             donated memory. *)
          let at = (Option.get r.Prim.route).result in
          let (A.Any a0) = arrays.(0) in
          let dt = A.dtype a0 in
          let arrays = Iarray.map (A.expect dt) (Iarray.of_array arrays) in
          let l = r.layouts.(0) in
          let shape = Iarray.to_array (Nx_array.Layout.shape l) in
          let into = Value.Any (Prim.of_arrays at ~shape arrays) in
          partitioned Exec.eager ~by ~into r op)

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
  (* The owners [i] does not adopt, once each, newest first: none in the
     common case, which then allocates nothing. *)
  let others =
    Prim.fold
      (fun i acc x ->
        match x with
        | Traced { owner; _ }
          when (not (adopts i owner)) && not (List.memq owner acc) ->
            List.cons owner acc
        | Traced _ | Array _ | Shards _ | Handle _ -> acc)
      i [] op
  in
  (match others with
  | [] -> ()
  | _ :: _ ->
      List.iter
        (fun o -> if not (around o) then refuse ~by o)
        (List.rev others));
  Prim.fold (fun by () x -> Prim.alive ~by x) by () op;
  let op =
    if Prim.exists Donate.is_handle op then Prim.map Donate.lent op else op
  in
  i.rule i ~by op

let reaches i op =
  match i.reach with
  | Extent -> true
  | Values | Adopting _ ->
      Prim.fold (fun i found x -> found || owns i x) i false op

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

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let handler i =
  let here = (Domain.self () :> int) in
  {
    Effect.Deep.retc = Fun.id;
    exnc = reraise;
    effc =
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
  let rec i = { name; reach; rule; live; self = Some i } in
  let run () = Effect.Deep.match_with f i (handler i) in
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

let traced i form payload =
  Traced { form; owner = i; payload; held = Some payload; twin = None }

let twinned i c payload =
  Traced
    {
      form = Prim.form c;
      owner = i;
      payload;
      held = Some payload;
      twin = Some c;
    }

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
  | Traced { owner; held; _ } when owner == i -> held
  | Traced _ | Array _ | Shards _ | Handle _ -> None

let owner (type v s d) (x : (v, s, d) t) =
  match x with
  | Traced { owner; _ } -> owner.self
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

let dispatch ~by op =
  if quiet () && (Exec.is_local op || not (Prim.exists is_traced op)) then
    run ~by op
  else
    let e = Receive { by; op = Some op } in
    match Effect.perform e with
    | r -> r
    | exception Effect.Unhandled e' when Obj.repr e' == Obj.repr e ->
        (match e with Receive cell -> cell.op <- None | _ -> ());
        unreached ~by op

(* Composites *)

(* Whether [x]'s placement cuts an axis or is a sum. *)
let spread x =
  let g = Devices.grid (Prim.placement x) in
  Grid.is_cut g || Grid.is_partial g

let apart x y =
  spread x || spread y
  || not (Devices.equal (Prim.placement x) (Prim.placement y))

type 'd arg =
  | Arg : string * ('v, 's, 'd) t -> 'd arg
  | Parts : string * string * 'd any list -> 'd arg

let composite ~by args f =
  let part name place (Any x) =
    (name, place, Prim.placement x, Composite.Value x)
  in
  let arg = function
    | Arg (name, x) -> List.[ part name "Nx.place" (Any x) ]
    | Parts (name, place, xs) -> List.map (part name place) xs
  in
  Composite.call ~by (List.concat_map arg args) f

(* Notes, inside a composite, how [op]'s results [r] derive from its
   operands that derive from an argument ({!Composite.note}): those of the
   first's shape, as the rule's windows are the first's. A placement's
   results lie where it puts them, whatever its operand's placement. An
   operation none of whose operands derives allocates nothing here. *)
let note (type r) (op : r prim) (r : r) =
  match op with
  | Place _ | Check _ -> ()
  | _ when not (Prim.exists Composite.derives op) -> ()
  | _ ->
      let n = Prim.fold (fun () n _ -> n + 1) () 0 op in
      let operands = Array.make n None in
      Prim.iteri
        (fun i x ->
          match Composite.source x with
          | Some d -> operands.(i) <- Some (Prim.layout_shape x, d)
          | None -> ())
        op;
      let node = Composite.made (Prim.rule op) operands in
      let first = ref None in
      Prim.iter_results
        (fun y ->
          let s = Prim.layout_shape y in
          match !first with
          | None ->
              first := Some s;
              Composite.note y node
          | Some f -> if s = f then Composite.note y node)
        op r

let eval ~by op =
  let r = dispatch ~by op in
  if not (Composite.idle ()) then note op r;
  r

(* Notes, inside a composite, [y] as an elementwise result of [xs]. *)
let mapped xs y =
  let source (Any x) =
    Option.map (fun d -> (Prim.layout_shape x, d)) (Composite.source x)
  in
  let operands = Array.map source xs in
  if Array.exists Option.is_some operands then
    Composite.note y (Composite.made Route.Elementwise operands)

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
      if quiet () then begin
        let y = fast1 ~by k dt x in
        if not (Composite.idle ()) then mapped [| Any x |] y;
        y
      end
      else eval1 ~by k dt x

let apply2 ~by k dt x y =
  match (x, y) with
  | Traced _, _ | _, Traced _ -> eval2 ~by k dt x y
  | _ ->
      if quiet () then begin
        let z = fast2 ~by k dt x y in
        if not (Composite.idle ()) then mapped [| Any x; Any y |] z;
        z
      end
      else eval2 ~by k dt x y

let apply3 ~by k c x y =
  match (c, x, y) with
  | Traced _, _, _ | _, Traced _, _ | _, _, Traced _ -> eval3 ~by k c x y
  | _ ->
      if quiet () then begin
        let z = fast3 ~by k c x y in
        if not (Composite.idle ()) then mapped [| Any c; Any x; Any y |] z;
        z
      end
      else eval3 ~by k c x y

(* The map of [prog] over [x], into [dt]. *)
let map_of prog dt x =
  let layout = A.Layout.contiguous (Iarray.of_array (Prim.shape x)) in
  Map { layout; prog; outs = [ dt ]; loads = [| Plain x |]; at = None }

let apply ~by prog dt x =
  match (x, Exec.live_kernels x) with
  | Array { at; a; _ }, Some kernels when quiet () ->
      let y = Exec.map ~by kernels at prog dt a [| Nx_kernel.Plain a |] in
      if not (Composite.idle ()) then mapped [| Any x |] y;
      y
  | _ ->
      let v, () = eval ~by (map_of prog dt x) in
      v

let contraction ~sum ~batch ~contracting ~acc dt a b init =
  Contract
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
      sum;
    }

let contract ~by ?(sum = false) ~batch ~contracting ~acc dt a b init =
  let fast =
    if quiet () then Exec.contract ~by ~batch ~contracting ~acc dt a b init
    else None
  in
  match fast with
  | Some y ->
      if not (Composite.idle ()) then
        note (contraction ~sum ~batch ~contracting ~acc dt a b init) y;
      y
  | None -> eval ~by (contraction ~sum ~batch ~contracting ~acc dt a b init)

let place ~by p x =
  match x with
  | Array r when r.at == p && r.dead == Live && quiet () ->
      (* A second value over the memory shares it. *)
      Rig.Claim.share (Nx_array.buffer r.a);
      Array { at = p; a = r.a; dead = Live }
  | Array _ | Shards _ | Handle _ | Traced _ ->
      eval ~by (Place (p, x))

(* Movements *)

(* With no live [Extent], a live array moves where it lies, as
   {!Exec.windows} moves it: a view of its memory, built without the
   operation. A movement no layout expresses, one that does not apply, and
   any other operand go through {!eval}, which copies or raises naming
   [by]. A handle never takes this path, so donation is unchanged. *)
let move ~by mv x =
  match x with
  | Array ({ at; a; _ } as r)
    when r.dead == Live && quiet () && not (Grid.is_cut (Devices.grid at)) -> (
      match A.move mv a with
      | Some v ->
          Exec.view_of a;
          let y = Array { at; a = v; dead = Live } in
          if not (Composite.idle ()) then note (Move (mv, x)) y;
          y
      | None | (exception Invalid_argument _) -> eval ~by (Move (mv, x)))
  | Array _ | Shards _ | Handle _ | Traced _ -> eval ~by (Move (mv, x))

(* A bitcast, as {!move} is. *)
let bitcast ~by dt x =
  match x with
  | Array ({ at; a; _ } as r)
    when r.dead == Live && quiet () && not (Grid.is_cut (Devices.grid at)) -> (
      match A.bitcast dt a with
      | Some v ->
          Exec.view_of a;
          let y = Array { at; a = v; dead = Live } in
          if not (Composite.idle ()) then note (Bitcast (dt, x)) y;
          y
      | None | (exception Invalid_argument _) -> eval ~by (Bitcast (dt, x)))
  | Array _ | Shards _ | Handle _ | Traced _ -> eval ~by (Bitcast (dt, x))

(* A movement to the shape [x] holds makes no view: [x]'s memory stays
   unshared, so a later donation of [x] may still write into it. *)
let reshape ~by s x =
  if Prim.has_shape x s then begin
    Prim.alive ~by x;
    x
  end
  else move ~by (A.Move.Reshape (Iarray.of_array s)) x

(* A loop: [Iarray.for_all] over an [init] would allocate. *)
let identity p r =
  Iarray.length p = r
  &&
  let i = ref 0 in
  while !i < r && Iarray.get p !i = !i do
    incr i
  done;
  !i = r

let mapped ~by p x =
  if identity p (Prim.rank x) then begin
    Prim.alive ~by x;
    x
  end
  else move ~by (A.Move.Permute p) x

let broadcast ~by s x =
  if Prim.has_shape x s then begin
    Prim.alive ~by x;
    x
  end
  else move ~by (A.Move.Broadcast (Iarray.of_array s)) x
