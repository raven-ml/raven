(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Loop-invariant parts *)

let cut ~own next args body =
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
  let writes c =
    let slots =
      List.filter_map
        (fun u ->
          if Ops.op u <> Op.Store then None
          else
            match Ops.arg (storage (Ops.nth u 0)) with
            | Ops.Param p -> Some p.slot
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
     each trip too. A call runs each trip when it varies, or reads storage such
     a call writes. *)
  let written = Ops.Tbl.create 8 in
  let rec settle () =
    let reads_written =
      memo (Ops.Tbl.create 64) (fun u -> Ops.Tbl.mem written u)
    in
    let runs c = reads_param c || open_ranges c || reads_written c in
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
  let rec rebuild u =
    match Ops.Tbl.find_opt rebuilt u with
    | Some v -> v
    | None ->
        let v =
          if
            varies u
            || (reads_storage u && Op.Set.mem (Ops.op u) Op.Set.movement)
          then
            let src = List.map rebuild (sources u) in
            Ops.replace u
              ~src:(if Ops.op u = Op.Call then Ops.body u :: src else src)
          else if not (reads_storage u) then u
          else
            let slot = next () and arg = Ops.contiguous u in
            args := (slot, arg) :: !args;
            Call.param_like arg slot
        in
        Ops.Tbl.add rebuilt u v;
        v
  in
  rebuild body

(* Repeated steps *)

(* A carry is storage of each device's part of its value: the whole of it, or
   its shard along [axis]. [view s] is the value over storage [s] of [numel]
   elements. *)
type carry =
  | Empty of Ops.t
  | Carry of { slot : int; init : Ops.t; numel : int; view : Ops.t -> Ops.t }

let carry device slot axis u =
  let shape = Shape.max_shape u in
  match (device, axis) with
  | Ops.Multi devices, Some axis ->
      let part =
        List.mapi
          (fun d n -> if d = axis then n / List.length devices else n)
          shape
      in
      let view s =
        Call.unshard
          (Shape.reshape s (List.map (fun n -> Ops.Int n) part))
          [ axis ]
      in
      Carry { slot; init = u; numel = List.fold_left ( * ) 1 part; view }
  | _ ->
      let view s = Shape.reshape s (Shape.shape u) in
      Carry { slot; init = u; numel = Shape.max_numel u; view }

let repeat device n init step =
  let empty u = Shape.max_numel u = 0 in
  if n = 0 || List.for_all empty init then init
  else
    let placed u =
      if Option.is_none (Ops.device u) then Ops.copy_to_device u device else u
    in
    let init = List.map placed init in
    (* Each carry is laid out as the step lays out its next value: sharded where
       it mixes with a sharded value. An odd count takes its first step before
       the loop. *)
    let first = step init in
    let axes = List.map Call.axis first in
    let init = if n mod 2 = 1 then List.map Ops.contiguous first else init in
    if n < 2 then init
    else
      let count = ref 0 in
      let next () =
        let k = !count in
        incr count;
        k
      in
      let carries =
        List.map2
          (fun u axis ->
            if empty u then Empty u else carry device (next ()) axis u)
          init axes
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
                  Call.param ~shape:[ Ops.Int numel ] ~device slot
                    (Ops.dtype init)
                in
                Ops.Tbl.replace own p ();
                view p)
          carries
      in
      (* A trip takes two steps: the first into storage of its own, the second
         from it into the carries' storage, which it then no longer reads. *)
      let finals = step (List.map Ops.contiguous (step params)) in
      let args = ref [] and stores = ref [] in
      let results =
        List.map2
          (fun (c, param) v ->
            match c with
            | Empty u -> fun _ -> u
            | Carry { slot; init; numel; view } ->
                let b = Ops.new_buffer device numel (Ops.dtype init) in
                args := (slot, Ops.after b [ Ops.store (view b) init ]) :: !args;
                stores := Ops.store param v :: !stores;
                fun e -> view (Ops.after b [ e ]))
          (List.combine carries params)
          finals
      in
      let body = cut ~own next args (Ops.sink (List.rev !stores)) in
      let range =
        Ops.range ~axis_type:Loop (Ops.Int (n / 2)) [ Ops.unique_num () ]
      in
      let by_slot (a, _) (b, _) = Int.compare a b in
      let e =
        Ops.end_
          (Ops.call ~precompile:true body
             (List.map snd (List.sort by_slot !args)))
          [ range ]
      in
      List.map (fun result -> result e) results
