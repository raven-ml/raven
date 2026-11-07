(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Loop-invariant parts

   A movement of an invariant part stays in the body, and the part below it is
   computed once: a view costs nothing where it is read, while computing it once
   would store a transposed or broadcast copy that the loop then reads. *)

let cut ?(hoist = true) ~own next args body =
  let reading = Ops.Tbl.create 64
  and storing = Ops.Tbl.create 64
  and rebuilt = Ops.Tbl.create 64 in
  let sources u =
    if Ops.op u = Op.Call then Ops.src_without_body u else Ops.src u
  in
  let memo table f =
    let rec go u =
      match Ops.Tbl.find_opt table u with
      | Some v -> v
      | None ->
          let v = f u || List.exists go (sources u) in
          Ops.Tbl.add table u v;
          v
    in
    go
  in
  let reads_param =
    memo reading (fun u -> Ops.op u = Op.Param && Ops.Tbl.mem own u)
  in
  let reads_storage =
    memo storing (fun u ->
        match Ops.op u with
        | Op.Buffer | Op.After -> true
        | Op.Param -> Ops.addrspace u = Some Tolk.Dtype.Global
        | _ -> false)
  in
  let rec storage u =
    match (Ops.op u, Ops.src u) with
    | (Op.Param | Op.Buffer | Op.Alloc), _ | _, [] -> u
    | _, x :: _ -> storage x
  in
  (* A call writes the arguments of the parameters its body stores into. A
     buffer's slot is its identity, which may equal any argument's index, and
     the body also reaches stores into storage around it, such as an enclosing
     loop's first carries. *)
  let writes c =
    let slots =
      List.filter_map
        (fun u ->
          if Ops.op u <> Op.Store then None
          else
            let s = storage (Ops.nth u 0) in
            match (Ops.op s, Ops.arg s) with
            | Op.Param, Ops.Param p -> Some p.slot
            | _ -> None)
        (Ops.toposort ~calls:Skip (Ops.body c))
    in
    List.filteri (fun k _ -> List.mem k slots) (Ops.src_without_body c)
    |> List.map storage
  in
  let calls =
    List.filter_map
      (fun u -> if Ops.op u = Op.Call then Some (u, writes u) else None)
      (Ops.toposort ~calls:Skip body)
  in
  let open_ranges u = Ops.Nodes.cardinal (Ops.ranges u) > 0 in
  (* The storage a call of the body that runs each trip writes, such as a loop's
     carry it updates in place: what reads it, its first value included, runs
     each trip too. A call runs each trip when it reads an own parameter, or
     storage such a call writes. The ranges a call runs inside, its own loop's
     among them, repeat it within a trip, the same in every trip. *)
  let written = Ops.Tbl.create 8 in
  let rec settle () =
    let reads_written =
      memo (Ops.Tbl.create 64) (fun u -> Ops.Tbl.mem written u)
    in
    let runs c = reads_param c || reads_written c in
    match
      List.concat_map
        (fun (c, bs) ->
          if runs c then List.filter (fun b -> not (Ops.Tbl.mem written b)) bs
          else [])
        calls
    with
    | [] -> reads_written
    | fresh ->
        List.iter (fun b -> Ops.Tbl.replace written b ()) fresh;
        settle ()
  in
  let reads_written = settle () in
  let varies u = reads_param u || reads_written u || open_ranges u in
  (* A value of no width, such as an index computed from stored integers, has no
     storage to pass: the body computes it from the stored values it reads. *)
  let weak u = List.exists (Dtype.equal (Ops.dtype u)) Dtype.weaks in
  (* Without hoisting, only storage itself is passed, a buffer or another
     call's parameter: what reads it stays, and so does a value after the
     work that writes it, such as a nested loop's carry. A store is no value
     to pass: it stays, with what it writes and where passed where they do
     not vary, as an assembly's store of a piece no trip changes beside one
     each trip computes. *)
  let stored u =
    match Ops.op u with
    | Op.Buffer -> true
    | Op.Param -> Ops.addrspace u = Some Tolk.Dtype.Global
    | _ -> false
  in
  let rec rebuild u =
    match Ops.Tbl.find_opt rebuilt u with
    | Some v -> v
    | None ->
        let v =
          if
            varies u
            || reads_storage u
               && (Op.Set.mem (Ops.op u) Op.Set.movement
                  || weak u
                  || Ops.op u = Op.Store
                  || ((not hoist) && not (stored u)))
          then
            let src = List.map rebuild (sources u) in
            Ops.replace u
              ~src:(if Ops.op u = Op.Call then Ops.body u :: src else src)
          else if not (reads_storage u) then u
          else
            let slot = next () and arg = Ops.contiguous u in
            args := (slot, arg) :: !args;
            Shape.param_like arg slot
        in
        Ops.Tbl.add rebuilt u v;
        v
  in
  rebuild body

(* Repeated steps *)

(* A carry is storage of its value. [view s] is the value over storage [s] of
   [numel] elements. *)
type carry =
  | Empty of Ops.t
  | Carry of { slot : int; init : Ops.t; numel : int; view : Ops.t -> Ops.t }

let carry slot u =
  let view s = Shape.reshape s (Shape.shape u) in
  Carry { slot; init = u; numel = Shape.max_numel u; view }

let empty u = Shape.max_numel u = 0

let placed device u =
  if Option.is_none (Ops.device u) then Ops.copy_to_device u device else u

(* A loop's body as it is built: the call's next slot, from [0], the carries
   and the parameters that stand for them in the step, the call's own
   parameters, what the call binds to its slots, and the body's stores. *)
type body = {
  next : unit -> int;
  carries : carry list;
  params : Ops.t list;
  own : unit Ops.Tbl.t;
  args : (int * Ops.t) list ref;
  stores : Ops.t list ref;
}

let body device init =
  let count = ref 0 in
  let next () =
    let k = !count in
    incr count;
    k
  in
  let carries =
    List.map (fun u -> if empty u then Empty u else carry (next ()) u) init
  in
  let own = Ops.Tbl.create 8 in
  (* A value with no element stands in the step as zeros: reading it reads
     nothing. *)
  let params =
    List.map
      (function
        | Empty u ->
            Shape.expand
              (Ops.const ~dtype:(Ops.dtype u) (`Int Bigint.zero))
              (Shape.shape u)
        | Carry { slot; init; numel; view } ->
            let p =
              Shape.param ~shape:[ Ops.Int numel ] ~device slot (Ops.dtype init)
            in
            Ops.Tbl.replace own p ();
            view p)
      carries
  in
  { next; carries; params; own; args = ref []; stores = ref [] }

(* The trip's index, an [int32] scalar parameter the call binds to [index]. *)
let index b index =
  let slot = b.next () in
  let p = Shape.param ~addrspace:Alu slot Int32 in
  Ops.Tbl.replace b.own p ();
  b.args := (slot, index) :: !(b.args);
  p

(* Each carry's buffer, holding its initial value, which the trip's value
   [v] then replaces, and its value once the loop [e] is done. *)
let carried device b finals =
  List.map2
    (fun (c, param) v ->
      match c with
      | Empty u -> fun _ -> u
      | Carry { slot; init; numel; view } ->
          let s = Ops.new_buffer device numel (Ops.dtype init) in
          let first = Ops.after s [ Ops.store (view s) init ] in
          b.args := (slot, first) :: !(b.args);
          b.stores := Ops.store param v :: !(b.stores);
          fun e -> view (Ops.after s [ e ]))
    (List.combine b.carries b.params)
    finals

let call b =
  let body = cut ~own:b.own b.next b.args (Ops.sink (List.rev !(b.stores))) in
  let by_slot (a, _) (b, _) = Int.compare a b in
  Ops.call ~precompile:true body (List.map snd (List.sort by_slot !(b.args)))

let repeat device n init step =
  if n = 0 || List.for_all empty init then init
  else
    let init = List.map (placed device) init in
    (* An odd count takes its first step before the loop. *)
    let init =
      if n mod 2 <> 1 then init
      else
        List.map Ops.contiguous
          (step (Ops.const ~dtype:Int32 (`Int Bigint.zero)) init)
    in
    if n < 2 then init
    else
      let b = body device init in
      (* The loop runs [n / 2] trips of one call, whose body is the two steps
         traced once. A trip takes two steps: the first into storage of its own,
         the second from it into the carries' storage, which it then no longer
         reads. The call passes the first step's index, and the second's is one
         more. *)
      let range =
        Ops.range ~axis_type:Loop (Ops.Int (n / 2)) [ Ops.unique_num () ]
      in
      let i = index b Ops.O.((range * int 2) + int (n mod 2)) in
      let i' = Ops.add i (Shape.const_like i (`Int Bigint.one)) in
      let finals = step i' (List.map Ops.contiguous (step i b.params)) in
      let results = carried device b finals in
      let e = Ops.end_ (call b) [ range ] in
      List.map (fun result -> result e) results

let until device n init ~stop step =
  if n <= 0 || List.for_all empty init then init
  else
    let init = List.map (placed device) init in
    let b = body device init in
    (* Each trip takes one step into storage of its own, from which it stores
       the carries and the flag: whether the stop does not hold yet, which the
       engine reads before the next trip. *)
    let range = Ops.range ~axis_type:Loop (Ops.Int n) [ Ops.unique_num () ] in
    let i = index b range in
    let finals = List.map Ops.contiguous (step i b.params) in
    let results = carried device b finals in
    let running vs = Ops.logical_not (Shape.reshape (stop vs) [ Ops.Int 1 ]) in
    let first = placed device (running init) in
    let slot = b.next () in
    let flag =
      Shape.param ~shape:[ Ops.Int 1 ] ~device slot (Ops.dtype first)
    in
    Ops.Tbl.replace b.own flag ();
    let s = Ops.new_buffer device 1 (Ops.dtype first) in
    let first = Ops.after s [ Ops.store s first ] in
    b.args := (slot, first) :: !(b.args);
    b.stores := Ops.store flag (running finals) :: !(b.stores);
    let e = Ops.backedge (call b) ~loop:range ~cond:first in
    List.map (fun result -> result e) results
