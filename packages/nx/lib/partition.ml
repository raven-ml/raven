(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module L = Nx_array.Layout
module M = Nx_array.Move

type engine = {
  run :
    'r 'd.
    by:string ->
    ?into:'d Value.any ->
    unit Devices.placement ->
    Nx_array.Layout.t array ->
    'r Value.prim ->
    'r;
  piece :
    'v 's 'd.
    ('v, 's, 'd) Value.t -> int -> 'd Devices.placement -> ('v, 's, 'd) Value.t;
  view :
    'v 's 'd. Nx_array.Move.t -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t;
  bring :
    'v 's 'd 'e.
    'e Devices.placement ->
    (('v, 's, 'd) Value.t * Nx_array.Move.range array) list ->
    ('v, 's, 'e) Value.t;
  gather :
    'v 's 'd 'e.
    'e Devices.placement ->
    ('v, 's) Value.dtype ->
    int iarray ->
    (Nx_array.Move.range array
    * ('v, 's, 'd) Value.t
    * Nx_array.Move.range array)
    list ->
    ('v, 's, 'e) Value.t;
  assemble :
    'v 's 'd 'e.
    'e Devices.placement ->
    int array ->
    ('v, 's) Value.dtype ->
    ('v, 's, 'd) Value.t option array ->
    ('v, 's, 'e) Value.t;
}

(* Windows *)

(* Whether [w] starts at the origin, from axis [i] on. *)
let rec at_origin (w : M.range array) i =
  i = Array.length w || (w.(i).start = 0 && at_origin w (i + 1))

(* [inner], a window of the whole, as a window of the piece holding [outer]:
   [inner] itself where [outer] starts at the origin, as a whole's does. *)
let relative (outer : M.range array) (inner : M.range array) =
  if at_origin outer 0 then inner
  else
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

let slice w = M.Slice (Iarray.of_array w)
let counts w = Array.map (fun (r : M.range) -> r.count) w

(* Whether [a] and [b] are one window, without the polymorphic comparison's
   call into the runtime. *)
let rec same_from (a : M.range array) (b : M.range array) i =
  i = Array.length a
  ||
  let x = a.(i) and y = b.(i) in
  x.start = y.start && x.count = y.count && x.step = y.step
  && same_from a b (i + 1)

let same a b = Array.length a = Array.length b && same_from a b 0

(* Reading *)

(* [read] of [x] of [shape]. *)
let read_of (type v s d) e ~by (x : (v, s, d) Value.t) shape k w :
    (v, s, d) Value.t =
  let at = Prim.placement x in
  if Route.empty w then
    let here = Devices.one (Devices.set at) k in
    e.assemble here (counts w) (Prim.dtype x) [| None |]
  else
    let j = Grid.position (Devices.grid at) k in
    if j < 0 then invalid_arg "Partition.read: no piece on the device";
    let held = Devices.window ~by at shape j in
    if not (Route.contains held w) then
      invalid_arg "Partition.read: the window is not held";
    let piece = e.piece x j (Devices.one (Devices.set at) k) in
    if same held w then piece else e.view (slice (relative held w)) piece

let read e ~by x k w = read_of e ~by x (Prim.shape x) k w

(* Windows *)

let is_local (route : unit Route.t) op =
  let p = route.result in
  Grid.count (Devices.grid p) = 1 && Prim.lies_at p op

(* [op] where its device [here] computes it: a map of no loads names its
   placement. *)
let at_device : type r. unit Devices.placement -> r Value.prim -> r Value.prim
    =
 fun here op ->
  match op with
  | Value.Map ({ at = Some _; _ } as m) -> Value.Map { m with at = Some here }
  | _ -> op

(* The shape of [op]'s result at position [i], laid out as [layouts]: one
   layout per result of a routine, one shared by the results of any other
   operation. *)
let shape_at layouts i =
  let l = if i < Array.length layouts then layouts.(i) else layouts.(0) in
  Iarray.to_array (L.shape l)

let windows : type r d.
    engine ->
    by:string ->
    ?into:d Value.any ->
    unit Route.t ->
    L.t array ->
    r Value.prim ->
    r =
 fun e ~by ?into route layouts op ->
  (match route.compute with
  | Windows -> ()
  | Partials ->
      invalid_arg "Partition.windows: a route that computes by partials");
  if is_local route op then e.run ~by ?into route.result layouts op
  else
    let p = route.result in
    let set = Devices.set p in
    match Grid.one (Devices.grid p) with
    | Some k ->
        (* On one device each operand is read whole: itself where it lies
           there, and otherwise its piece there. *)
        let read _ x =
          if Prim.placement x == p then x
          else
            let s = Prim.shape x in
            read_of e ~by x s k (Array.map Route.full s)
        in
        e.run ~by ?into p layouts (Prim.mapi read op)
    | None ->
        (* Each device's window [w] of the first result, computed there over
           each operand's window ({!Route.window}). A device computes where
           a result's window there holds elements. Balanced windows put the
           empty ones last, so the first device's windows are empty only
           where every device's are; it then computes, so that the kernels
           answer for the operation as on one device. *)
        let rule = Prim.rule op in
        let devices = Grid.devices (Devices.grid p) in
        (* Each device's window of each layout's results, the first's
           first; its results there are fresh and C-contiguous. *)
        let windows l =
          let s = Iarray.to_array (L.shape l) in
          Array.mapi (fun j _ -> Devices.window ~by p s j) devices
        in
        let ws = Array.map windows layouts in
        let holds j (w : M.range array array) = not (Route.empty w.(j)) in
        let computes j = j = 0 || Array.exists (holds j) ws in
        let local j =
          let k = devices.(j) and w = ws.(0).(j) in
          let view i x =
            let s = Prim.shape x in
            read_of e ~by x s k (Route.window rule ~operand:i s w)
          in
          let here = Devices.one set k in
          let op = Prim.local (Prim.mapi view op) w in
          let contiguous w = L.contiguous (Route.extents w.(j)) in
          let layouts = Array.map contiguous ws in
          e.run ~by here layouts (at_device here op)
        in
        let rs =
          Array.mapi
            (fun j _ -> if computes j then Some (local j) else None)
            devices
        in
        Prim.zip op rs (fun i vs ->
            let dt = Prim.dtype (Option.get (Array.find_map Fun.id vs)) in
            e.assemble p (shape_at layouts i) dt vs)

(* Placing *)

(* Whether a source window before index [i] is [sw]: each distinct source
   window is read at its first index, so that the boxes come in that
   order. *)
let rec seen src i sw = i >= 0 && (same src.(i) sw || seen src (i - 1) sw)

(* The first index from [i] of a source of window [sw] on [d], else [i]. *)
let rec on_rig at ks src d sw i i' =
  if i' = Array.length src then i
  else if same src.(i') sw && Rig.equal (Devices.rig (Devices.set at) ks.(i')) d
  then i'
  else on_rig at ks src d sw i (i' + 1)

(* [x]'s piece on its [i]th device, made once, when a window first reads it:
   [pieces.(i)] is [x] until then. *)
let piece_of eng x pieces at ks i =
  if pieces.(i) != x then pieces.(i)
  else
    let v = eng.piece x i (Devices.one (Devices.set at) ks.(i)) in
    pieces.(i) <- v;
    v

(* The window [w] of a value [x] at [at], its pieces' windows [src], on the
   device [devices.(j)] of [set]: from the pieces that hold all of it, or
   gathered from the box each distinct source window shares with it, from a
   piece on that device where one holds that window. *)
let window_of eng x pieces src at ks set devices j w =
  let n = Array.length src in
  let here = Devices.one set devices.(j) in
  let holders = ref [] in
  for i = n - 1 downto 0 do
    if Route.contains src.(i) w then
      holders := (piece_of eng x pieces at ks i, relative src.(i) w) :: !holders
  done;
  match !holders with
  | _ :: _ -> eng.bring here !holders
  | [] ->
      let d = Devices.rig set devices.(j) in
      let boxes = ref [] in
      for i = n - 1 downto 0 do
        let sw = src.(i) in
        if not (seen src (i - 1) sw) then
          match meet sw w with
          | None -> ()
          | Some box ->
              let v = piece_of eng x pieces at ks (on_rig at ks src d sw i i) in
              boxes := (relative w box, v, relative sw box) :: !boxes
      done;
      eng.gather here (Prim.dtype x) (Route.extents w) !boxes

type ('v, 's, 'd) held = {
  x : ('v, 's, 'd) Value.t;
  src : M.range array array;
  pieces : ('v, 's, 'd) Value.t array;
}

let held (type v s d) ~by (x : (v, s, d) Value.t) : (v, s, d) held =
  let at = Prim.placement x and shape = Prim.shape x in
  let n = Grid.count (Devices.grid at) in
  let src = Array.init n (fun i -> Devices.window ~by at shape i) in
  { x; src; pieces = Array.make n x }

let take eng h here w =
  let at = Prim.placement h.x in
  let ks = Grid.devices (Devices.grid at) in
  let set = Devices.set here and devices = Grid.devices (Devices.grid here) in
  window_of eng h.x h.pieces h.src at ks set devices 0 w

let place (type v s d e) eng ~by (p : e Devices.placement)
    (x : (v, s, d) Value.t) : (v, s, e) Value.t =
  let at = Prim.placement x and shape = Prim.shape x and dt = Prim.dtype x in
  let ks = Grid.devices (Devices.grid at) in
  let n = Array.length ks in
  if Devices.equal at p then begin
    let ps = Array.make n None in
    for j = 0 to n - 1 do
      ps.(j) <- Some (eng.piece x j (Devices.one (Devices.set at) ks.(j)))
    done;
    eng.assemble p shape dt ps
  end
  else begin
    let src = Array.make n [||] in
    for i = 0 to n - 1 do
      src.(i) <- Devices.window ~by at shape i
    done;
    let pieces = Array.make n x in
    let set = Devices.set p and devices = Grid.devices (Devices.grid p) in
    let ps = Array.make (Array.length devices) None in
    for j = 0 to Array.length devices - 1 do
      let w = Devices.window ~by p shape j in
      if not (Route.empty w) then
        ps.(j) <- Some (window_of eng x pieces src at ks set devices j w)
    done;
    eng.assemble p shape dt ps
  end
