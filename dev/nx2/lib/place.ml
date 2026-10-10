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
    match A.move (M.Slice w) a with
    | Some v -> v
    | None -> invalid_arg "Place: a slice of an array is a view"

let hosted a = Rig.shares_host_memory (A.device a)

(* [a] over its own memory on [d], where both are the host's: a device that only
   maps host memory takes a copy instead, so its work reads memory of its
   own. *)
let share d a =
  if Rig.shares_host_memory d && hosted a then A.borrow d a else None

(* [a] on [d]: itself, its memory shared, or a copy keeping its layout. *)
let bring d a =
  if Rig.equal (A.device a) d then a
  else match share d a with Some b -> b | None -> A.to_device d a

(* The bytes [A.to_device] copies of [a]: those between its first and last
   position. *)
let reach a =
  let lo, hi = L.span (A.layout a) and bits = A.Dtype.bits (A.dtype a) in
  (((hi * bits) + 7) / 8) - (lo * bits / 8)

let iter_box extents f =
  let r = Array.length extents in
  if not (Array.exists (( = ) 0) extents) then begin
    let idx = Array.make r 0 in
    let go = ref true in
    while !go do
      f idx;
      let a = ref (r - 1) in
      while !a >= 0 && idx.(!a) + 1 = extents.(!a) do
        idx.(!a) <- 0;
        decr a
      done;
      if !a < 0 then go := false else idx.(!a) <- idx.(!a) + 1
    done
  end

(* The runs that copy between layouts [s] and [t] of one shape: [(k, n)] where
   the axes from [k] on lie as [n] consecutive positions in both, so each index
   of the axes before [k] starts a run of [n] elements. An axis of extent 1
   joins any run. *)
let run s t =
  let n = ref 1 and a = ref (L.rank s - 1) in
  while
    !a >= 0 && (L.dim s !a = 1 || (L.stride s !a = !n && L.stride t !a = !n))
  do
    n := !n * L.dim s !a;
    decr a
  done;
  (!a + 1, !n)

(* [a] on memory the host addresses: itself, or a copy of the bytes it
   reaches. *)
let addressed a = if hosted a then a else A.to_device Rig.host a

(* Copies [s] into [t], arrays of one shape. The host copies into memory it
   addresses, from [s] or from the bytes [s] reaches brought to it. Otherwise
   [t]'s elements are whole bytes, copied by runs through [Rig.Buffer.copy],
   which chooses the path between the devices; [s] is first packed in C order on
   the host where that lengthens the runs, as for a transposed or broadcast
   array. *)
let copy_box s t =
  if hosted t then A.blit ~src:(addressed s) ~dst:t
  else begin
    let lt = A.layout t and eb = A.Dtype.bits (A.dtype t) / 8 in
    let _, packed = run (L.contiguous (L.shape lt)) lt in
    let s =
      if snd (run (A.layout s) lt) < packed then A.copy (addressed s) else s
    in
    let ls = A.layout s in
    let k, n = run ls lt in
    let sb = A.buffer s and tb = A.buffer t in
    iter_box
      (Array.sub (L.shape ls) 0 k)
      (fun idx ->
        let ps = ref (L.offset ls) and pt = ref (L.offset lt) in
        Array.iteri
          (fun a i ->
            ps := !ps + (i * L.stride ls a);
            pt := !pt + (i * L.stride lt a))
          idx;
        Rig.Buffer.copy
          ~src:(Rig.Buffer.view sb ~first:(!ps * eb) ~length:(n * eb))
          ~dst:(Rig.Buffer.view tb ~first:(!pt * eb) ~length:(n * eb)))
  end

let bytes dt (w : M.range array) =
  A.Dtype.bytes dt (Array.fold_left (fun n (r : M.range) -> n * r.count) 1 w)

(* [w] on [d] from one array of [sources] that holds it, where that allocates no
   more than [w]'s bytes: a view of the array on [d], a borrow, or a copy of the
   bytes the window reaches where they are no more than [w]'s, as for whole rows
   or a broadcast. *)
let from_one d dt w sources =
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
      | None ->
          List.find_opt (fun v -> reach v <= bytes dt w) views
          |> Option.map (A.to_device d))

(* [w] on [d] gathered into a fresh C-contiguous array: each distinct window of
   [sources] gives the box it shares with [w], from an array on [d] where one
   holds it. A dtype narrower than a byte gathers on the host where the host
   does not address [d]'s memory, since a run may start inside a byte, and is
   then brought to [d]. *)
let gather d dt w sources =
  let on =
    if A.Dtype.bits dt mod 8 = 0 || Rig.shares_host_memory d then d
    else Rig.host
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
            copy_box (slice a (relative sw box)) (slice dst (relative w box))
      end)
    sources;
  bring d dst

let value (type v s d e) ~by (p : e Devices.placement) (x : (v, s, d) Value.t) :
    (v, s, e) Value.t =
  let make arrays = Prim.of_arrays p arrays in
  let at = Prim.placement x in
  let arrays =
    match x with
    | Value.Array { a; _ } -> Iarray.of_list [ a ]
    | Value.Shards { arrays; _ } -> arrays
    | Value.Deferred _ ->
        invalid_arg "Place.value: a constant is placed computed"
    | Value.Traced _ ->
        invalid_arg "Place.value: a traced value is placed by its owner"
    | Value.Donated _ ->
        invalid_arg "Place.value: a donated value is placed as its consumer"
  in
  if Devices.equal at p then make arrays
  else begin
    let shape = Prim.shape x and dt = Prim.dtype x in
    let sources =
      Array.init (Iarray.length arrays) (fun i ->
          (Devices.window ~by at shape i, Iarray.get arrays i))
    in
    let set = Devices.set p in
    let take j k =
      let w = Devices.window ~by p shape j and d = Devices.rig set k in
      match from_one d dt w sources with
      | Some a -> a
      | None -> gather d dt w sources
    in
    let devices = Grid.devices (Devices.grid p) in
    make (Iarray.init (Array.length devices) (fun j -> take j devices.(j)))
  end

(* [x]'s window [w] from the array of device [k], which holds it. *)
let held (type v s d) ~by (x : (v, s, d) Value.t) k w : (v, s) A.t =
  let at = Prim.placement x and shape = Prim.shape x in
  let arrays =
    match x with
    | Value.Array { a; _ } -> Iarray.of_list [ a ]
    | Value.Shards { arrays; _ } -> arrays
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
  | Value.Array { at; a; _ } -> (
      (* On one device the array is the whole. *)
      match Grid.one (Devices.grid at) with
      | Some j when j = k -> slice a w
      | Some _ | None -> held ~by x k w)
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      held ~by x k w
