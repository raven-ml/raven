(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module L = Nx_array.Layout
module M = Nx_array.Move

let contains (outer : M.range array) (inner : M.range array) =
  Array.for_all2
    (fun (o : M.range) (i : M.range) ->
      i.start >= o.start && i.start + i.count <= o.start + o.count)
    outer inner

(* [inner], a window of the whole, as a window of the array holding [outer]. *)
let relative (outer : M.range array) (inner : M.range array) =
  Array.map2
    (fun (o : M.range) (i : M.range) -> { i with start = i.start - o.start })
    outer inner

(* The window [a] and [b] share, or [None] where they share no element. *)
let meet (a : M.range array) (b : M.range array) =
  let box =
    Array.map2
      (fun (x : M.range) (y : M.range) ->
        let start = max x.start y.start in
        let stop = min (x.start + x.count) (y.start + y.count) in
        { x with start; count = stop - start })
      a b
  in
  if Array.exists (fun (r : M.range) -> r.count <= 0) box then None
  else Some box

(* [a]'s elements in window [w]: [a] itself when [w] is all of it. *)
let slice a (w : M.range array) =
  let l = A.layout a in
  let whole = ref (Array.length w = L.rank l) and i = ref 0 in
  while !whole && !i < Array.length w do
    let r = w.(!i) in
    whole := r.start = 0 && r.count = L.dim l !i;
    incr i
  done;
  if !whole then a
  else
    match A.move (M.Slice (Iarray.of_array w)) a with
    | Some v -> v
    | None -> invalid_arg "Place: a slice of an array is a view"

let hosted a = Rig.reaches Rig.host (A.device a)

(* [a] over its own memory on [d], where the host addresses both: a device that
   only maps host memory takes a copy instead, so its work reads memory of its
   own. *)
let share d a =
  if Rig.reaches Rig.host d && hosted a then A.borrow d a else None

(* [a] on [d]: itself, its memory shared, or a copy of its elements. *)
let bring d a =
  if Rig.equal (A.device a) d then a
  else match share d a with Some b -> b | None -> A.to_device d a

(* [w] on [d] from one array of [sources] that holds it: a view of the array on
   [d], a borrow, or a copy of the window's elements. *)
let from_one d w sources =
  let views =
    Array.fold_right
      (fun (sw, a) vs ->
        if contains sw w then slice a (relative sw w) :: vs else vs)
      sources []
  in
  match List.find_opt (fun v -> Rig.equal (A.device v) d) views with
  | Some v -> Some v
  | None -> (
      match List.find_map (share d) views with
      | Some v -> Some v
      | None -> Option.map (A.to_device d) (List.nth_opt views 0))

(* [w] on [d] gathered into a fresh C-contiguous array: each distinct window of
   [sources] gives the box it shares with [w], from an array on [d] where one
   holds it. A dtype narrower than a byte gathers on the host where the host
   does not address [d]'s memory, since a run may start inside a byte, and is
   then brought to [d]. *)
let gather d dt w sources =
  let on =
    if A.Dtype.bits dt mod 8 = 0 || Rig.reaches Rig.host d then d else Rig.host
  in
  let dst = A.create on dt (Array.map (fun (r : M.range) -> r.count) w) in
  let seen = ref [] in
  Array.iter
    (fun (sw, a) ->
      if not (List.mem sw !seen) then begin
        seen := sw :: !seen;
        match meet sw w with
        | None -> ()
        | Some box ->
            let here (sw', a') = sw' = sw && Rig.equal (A.device a') d in
            let a =
              match Array.find_opt here sources with
              | Some (_, a) -> a
              | None -> a
            in
            A.transfer
              ~src:(slice a (relative sw box))
              ~dst:(slice dst (relative w box))
      end)
    sources;
  bring d dst

let value (type v s d e) ~by (p : e Devices.placement) (x : (v, s, d) Value.t) :
    (v, s, e) Value.t =
  let shape = Prim.shape x in
  let make arrays = Prim.of_arrays p ~shape arrays in
  let at = Prim.placement x in
  let arrays =
    match x with
    | Value.Array r -> ([| r.a |] : _ iarray)
    | Value.Shards r -> r.arrays
    | Value.Deferred _ ->
        invalid_arg "Place.value: a constant is placed computed"
    | Value.Traced _ ->
        invalid_arg "Place.value: a traced value is placed by its owner"
    | Value.Donated _ ->
        invalid_arg "Place.value: a donated value is placed as its consumer"
  in
  if Devices.equal at p then make arrays
  else begin
    let dt = Prim.dtype x in
    let sources =
      Array.init (Iarray.length arrays) (fun i ->
          (Devices.window ~by at shape i, Iarray.get arrays i))
    in
    let set = Devices.set p in
    let devices = Grid.devices (Devices.grid p) in
    (* An empty window is memory of no bytes of its own: a view of a source
       would hold, and share, all of that source's memory. *)
    let empty (w : M.range array) =
      Array.exists (fun (r : M.range) -> r.count = 0) w
    in
    make
      (Iarray.init (Array.length devices) (fun j ->
           let w = Devices.window ~by p shape j
           and d = Devices.rig set devices.(j) in
           if empty w then
             A.create d dt (Array.map (fun (r : M.range) -> r.count) w)
           else
             match from_one d w sources with
             | Some a -> a
             | None -> gather d dt w sources))
  end

(* [x]'s window [w] from the array of device [k], which holds it. *)
let held (type v s d) ~by (x : (v, s, d) Value.t) k w : (v, s) A.t =
  let at = Prim.placement x and shape = Prim.shape x in
  let arrays =
    match x with
    | Value.Array r -> ([| r.a |] : _ iarray)
    | Value.Shards r -> r.arrays
    | Value.Deferred _ -> invalid_arg "Place.view: a constant has no arrays"
    | Value.Traced _ -> invalid_arg "Place.view: a traced value has no arrays"
    | Value.Donated _ ->
        invalid_arg "Place.view: a donated value is viewed live"
  in
  let devices = Grid.devices (Devices.grid at) in
  match Array.find_index (( = ) k) devices with
  | None -> invalid_arg "Place.view: no array on the device"
  | Some j ->
      let held = Devices.window ~by at shape j in
      if not (contains held w) then
        invalid_arg "Place.view: the window is not held";
      slice (Iarray.get arrays j) (relative held w)

let view (type v s d) ~by (x : (v, s, d) Value.t) k w : (v, s) A.t =
  match x with
  | Value.Array r -> (
      (* On one device the array is the whole. *)
      match Grid.one (Devices.grid r.at) with
      | Some j when j = k -> slice r.a w
      | Some _ | None -> held ~by x k w)
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      held ~by x k w
