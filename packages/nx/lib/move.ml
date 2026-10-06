(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Movements

   A movement of a placed value is view arithmetic over the same storage, the
   same on every device. A split value moves shard by shard, so the movement
   must leave every element on its device: the split axis may move, stay whole,
   or be reshaped with the whole axes before it, but it is never flipped,
   windowed or cut across shards. A cut inside one shard is a view of that
   shard, on its device alone. These are the rules tolk applies to a compiled
   program over split values (schedule/multi.ml), so a value moved eagerly has
   the placement the same movement has in a compiled program, except for a cut
   inside one shard: tolk copies a whole shard to every device and refuses a
   part of one. *)

type t =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Window of { axis : int; size : int; step : int }

(* [view v m] is the view [v] moved by [m]. *)
let view v = function
  | Reshape shape -> View.reshape v shape
  | Expand shape -> View.expand v shape
  | Permute order -> View.permute v order
  | Shrink limits -> View.shrink v limits
  | Flip dims -> View.flip v dims
  | Window { axis; size; step } ->
      View.sliding_window v ~axis ~window:size ~step

(* What a movement does to one cut axis: the axis stays cut, at an index, or the
   movement keeps a single tile along it. *)
type split_axis = Split of int | Shard of int

(* [split_axis ~axis ~n shape m] is what [m] does to the tensor [axis] of a
   value of shape [shape] cut in [n] tiles along it. Every cut is decided
   against the whole shapes, as tolk's rewrite does, so a value cut along
   several axes cannot land two cuts on one axis. Raises [Invalid_argument] if
   [m] would move elements between tiles. *)
let split_axis ~axis ~n shape m =
  let k = shape.(axis) / n in
  let across what =
    invalid_arg
      (Printf.sprintf
         "Nx: a %s of the split axis %d of shape %s would move elements \
          between devices; place the value replicated or on one device first"
         what axis (Shape.to_string shape))
  in
  match m with
  | Permute order ->
      let a = ref 0 in
      Array.iteri (fun i o -> if o = axis then a := i) order;
      Split !a
  | Expand _ ->
      (* The split axis spans at least two shards, so it is never broadcast. *)
      Split axis
  | Reshape target ->
      (* The split axis becomes the last axis whose leading extents multiply to
         those before the split axis, and whose extents from it on to those from
         the split axis on: an empty axis makes every later leading product 0.
         Its extent must divide over the shards. *)
      let product a lo hi =
        let p = ref 1 in
        for d = lo to hi - 1 do
          p := !p * a.(d)
        done;
        !p
      in
      let rank = Array.length shape and r = Array.length target in
      if product target 0 r <> product shape 0 rank then
        invalid_arg
          (Printf.sprintf "Nx.reshape: cannot reshape %s to %s"
             (Shape.to_string shape) (Shape.to_string target));
      let lead = product shape 0 axis and tail = product shape axis rank in
      let a = ref (-1) in
      for b = 0 to r - 1 do
        if product target 0 b = lead && product target b r = tail then a := b
      done;
      if !a < 0 || target.(!a) mod n <> 0 then across "reshape";
      Split !a
  | Shrink limits ->
      let lo, hi = limits.(axis) in
      if lo = 0 && hi = shape.(axis) then Split axis
      else if lo = hi then
        (* An empty cut moves no element: it stays with the shard where it
           starts, the last at the axis's end. *)
        Shard (Int.min (lo / k) (n - 1))
      else if lo / k = (hi - 1) / k then Shard (lo / k)
      else across "cut"
  | Flip dims -> if dims.(axis) then across "flip" else Split axis
  | Window { axis = a; _ } -> if a = axis then across "window" else Split axis

(* [fates p shape m] is what [m] does to each cut axis of a value of shape
   [shape] at [p]: the axis, its number of tiles and its fate. *)
let fates p shape m =
  List.map
    (fun (axis, n) -> (axis, n, split_axis ~axis ~n shape m))
    (Placement.cuts p)

(* [localize shape m fates] is [m] as one tile of a value of shape [shape] sees
   it, [fates] giving each cut axis, its number of tiles and what [m] does to
   it. *)
let localize shape m fates =
  let each f =
    List.iter (fun (axis, n, fate) -> f axis (shape.(axis) / n) n fate) fates
  in
  match m with
  | Reshape target ->
      let local = Array.copy target in
      each (fun _ _ n fate ->
          match fate with
          | Split a -> local.(a) <- target.(a) / n
          | Shard _ -> ());
      Reshape local
  | Expand target ->
      let local = Array.copy target in
      each (fun axis k _ _ ->
          if axis < Array.length local && target.(axis) = shape.(axis) then
            local.(axis) <- k);
      Expand local
  | Shrink limits ->
      let local = Array.copy limits in
      each (fun axis k _ fate ->
          let lo, hi = limits.(axis) in
          local.(axis) <-
            (match fate with
            | Split _ -> (0, k)
            | Shard j -> (lo - (j * k), hi - (j * k))));
      Shrink local
  | Permute _ | Flip _ | Window _ -> m

(* The placement of a value at [p] moved as [fates] say: an axis that stays cut
   moves where the movement puts it, and a cut to one tile keeps the devices
   that hold that tile. *)
let placement_after p fates =
  let g =
    List.fold_left
      (fun g (axis, _, fate) ->
        match fate with Shard j -> Placement.select g ~axis j | Split _ -> g)
      p fates
  in
  Placement.map_axes
    (fun a ->
      match List.find (fun (axis, _, _) -> axis = a) fates with
      | _, _, Split a' -> a'
      | _, _, Shard _ -> a)
    g

(* [placement p shape m] is the placement of a value of shape [shape] at [p]
   moved by [m]. Raises [Invalid_argument] as [split_axis] does. *)
let placement p shape m = placement_after p (fates p shape m)

(* [split_view p v m] is the placement and per-shard view of a value at [p]
   whose per-shard view is [v], moved by [m]. *)
let split_view p v m =
  match Placement.cuts p with
  | [] -> (p, view v m)
  | _ ->
      let shape = Value.global p (View.shape v) in
      let fates = fates p shape m in
      (placement_after p fates, view v (localize shape m fates))

(* [apply x m] is [x] moved by [m]: view arithmetic over the same storage. *)
let apply (type a b) (x : (a, b) Value.t) m : (a, b) Value.t =
  match x with
  | Host a -> Host { a with view = view a.view m }
  | Placed r ->
      let r_placement, r_view = split_view r.r_placement r.r_view m in
      Placed { r with r_id = Value.fresh_id (); r_placement; r_view }
  | Traced _ -> Value.outside_trace ()
