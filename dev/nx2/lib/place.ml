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

(* [a] on [d]: itself, a borrow, or a copy keeping its layout. *)
let bring d a =
  if Rig.equal (A.device a) d then a
  else match A.borrow d a with Some b -> b | None -> A.to_device d a

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

(* Copies [c], the C-contiguous host array of [box]'s extents, into [dst], the
   C-contiguous host array of [shape], at [box]. Whole bytes copy as runs: the
   axes after [t] that [box] spans whole merge with axis [t] into one run per
   index of the axes before it. A dtype narrower than a byte copies element by
   element. *)
let copy_box dst shape (box : M.range array) c =
  let r = Array.length shape in
  let extents = Array.map (fun (b : M.range) -> b.count) box in
  let bits = A.Dtype.bits (A.dtype dst) in
  if r = 0 || bits mod 8 <> 0 then
    iter_box extents (fun idx ->
        let at = Array.mapi (fun a i -> box.(a).start + i) idx in
        A.set dst at (A.get c idx))
  else if not (Array.exists (( = ) 0) extents) then begin
    let eb = bits / 8 in
    let t = ref (r - 1) in
    while !t > 0 && extents.(!t) = shape.(!t) do
      decr t
    done;
    let t = !t in
    let stride = Array.make r 1 in
    for a = r - 2 downto 0 do
      stride.(a) <- stride.(a + 1) * shape.(a + 1)
    done;
    let run = extents.(t) * stride.(t) in
    let dbuf = A.buffer dst and cbuf = A.buffer c in
    let doff = L.offset (A.layout dst) and coff = L.offset (A.layout c) in
    let n = ref 0 in
    iter_box (Array.sub extents 0 t) (fun idx ->
        let at = ref (box.(t).start * stride.(t)) in
        Array.iteri
          (fun a i -> at := !at + ((box.(a).start + i) * stride.(a)))
          idx;
        Rig.Buffer.copy
          ~src:
            (Rig.Buffer.view cbuf
               ~first:((coff + (!n * run)) * eb)
               ~length:(run * eb))
          ~dst:
            (Rig.Buffer.view dbuf ~first:((doff + !at) * eb) ~length:(run * eb));
        incr n)
  end

(* The whole value on the host, from arrays holding the windows [sources]: one
   array per distinct window. *)
let assemble dt shape sources =
  let dst = A.create Rig.host dt shape in
  let seen = ref [] in
  Array.iter
    (fun (w, a) ->
      if not (List.mem w !seen) then begin
        seen := w :: !seen;
        copy_box dst shape w (A.copy (bring Rig.host a))
      end)
    sources;
  dst

let value (type v s d e) ~by (p : e Devices.placement) (x : (v, s, d) Value.t) :
    (v, s, e) Value.t =
  let make arrays =
    if Array.length arrays = 1 then Value.Array { at = p; a = arrays.(0) }
    else Value.Shards { at = p; arrays }
  in
  let at = Prim.placement x in
  let arrays =
    match x with
    | Value.Array { a; _ } -> [| a |]
    | Value.Shards { arrays; _ } -> arrays
    | Value.Deferred _ ->
        invalid_arg "Place.value: a constant is placed computed"
  in
  if Devices.equal at p then make arrays
  else begin
    let shape = Prim.shape x and dt = Prim.dtype x in
    let sources =
      Array.mapi (fun i a -> (Devices.window ~by at shape i, a)) arrays
    in
    let whole = lazy (assemble dt shape sources) in
    let set = Devices.set p in
    let take j k =
      let w = Devices.window ~by p shape j and d = Devices.rig set k in
      let holds (sw, _) = contains sw w in
      let here (sw, a) = holds (sw, a) && Rig.equal (A.device a) d in
      let view =
        match Array.find_opt here sources with
        | Some (sw, a) -> slice a (relative sw w)
        | None -> (
            match Array.find_opt holds sources with
            | Some (sw, a) -> slice a (relative sw w)
            | None -> slice (Lazy.force whole) w)
      in
      bring d view
    in
    make (Array.mapi take (Grid.devices (Devices.grid p)))
  end

(* [x]'s window [w] from the array of device [k], which holds it. *)
let held (type v s d) ~by (x : (v, s, d) Value.t) k w : (v, s) A.t =
  let at = Prim.placement x and shape = Prim.shape x in
  let arrays =
    match x with
    | Value.Array { a; _ } -> [| a |]
    | Value.Shards { arrays; _ } -> arrays
    | Value.Deferred _ -> invalid_arg "Place.view: a constant has no arrays"
  in
  let devices = Grid.devices (Devices.grid at) in
  match Array.find_index (( = ) k) devices with
  | None -> invalid_arg "Place.view: no array on the device"
  | Some j ->
      let held = Devices.window ~by at shape j in
      if not (contains held w) then
        invalid_arg "Place.view: the window is not held";
      slice arrays.(j) (relative held w)

let view (type v s d) ~by (x : (v, s, d) Value.t) k w : (v, s) A.t =
  match x with
  | Value.Array { at; a } -> (
      (* On one device the array is the whole. *)
      match Grid.one (Devices.grid at) with
      | Some j when j = k -> slice a w
      | Some _ | None -> held ~by x k w)
  | Value.Shards _ | Value.Deferred _ -> held ~by x k w
