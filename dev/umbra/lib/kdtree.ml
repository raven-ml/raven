(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type node =
  | Leaf
  | Node of {
      idx : int;
      x : float;
      y : float;
      z : float;
      split : int;
      left : node;
      right : node;
    }

type t = { root : node; size : int }

let coord split x y z = match split with 0 -> x | 1 -> y | _ -> z

let build xs ys zs =
  let n = Array.length xs in
  if n <> Array.length ys || n <> Array.length zs then
    invalid_arg "Kdtree.build: arrays must have the same length";
  let indices = Array.init n Fun.id in
  let rec build_rec start len depth =
    if len = 0 then Leaf
    else if len = 1 then
      let i = indices.(start) in
      Node
        {
          idx = i;
          x = xs.(i);
          y = ys.(i);
          z = zs.(i);
          split = depth mod 3;
          left = Leaf;
          right = Leaf;
        }
    else begin
      let split = depth mod 3 in
      let sub = Array.sub indices start len in
      Array.sort
        (fun a b ->
          Float.compare
            (coord split xs.(a) ys.(a) zs.(a))
            (coord split xs.(b) ys.(b) zs.(b)))
        sub;
      Array.blit sub 0 indices start len;
      let mid = len / 2 in
      let mi = indices.(start + mid) in
      let left = build_rec start mid (depth + 1) in
      let right = build_rec (start + mid + 1) (len - mid - 1) (depth + 1) in
      Node
        { idx = mi; x = xs.(mi); y = ys.(mi); z = zs.(mi); split; left; right }
    end
  in
  { root = build_rec 0 n 0; size = n }

let sq_dist px py pz qx qy qz =
  let dx = px -. qx and dy = py -. qy and dz = pz -. qz in
  (dx *. dx) +. (dy *. dy) +. (dz *. dz)

let nearest tree qx qy qz =
  if tree.size = 0 then invalid_arg "Kdtree.nearest: empty tree";
  let best_idx = ref 0 in
  let best_dist = ref Float.infinity in
  let rec search node =
    match node with
    | Leaf -> ()
    | Node { idx; x; y; z; split; left; right } ->
        let d = sq_dist x y z qx qy qz in
        if d < !best_dist then begin
          best_dist := d;
          best_idx := idx
        end;
        let q_split = coord split qx qy qz in
        let p_split = coord split x y z in
        let diff = q_split -. p_split in
        let near, far = if diff < 0.0 then (left, right) else (right, left) in
        search near;
        if diff *. diff < !best_dist then search far
  in
  search tree.root;
  (!best_idx, !best_dist)

let within tree qx qy qz max_dist_sq =
  let results = ref [] in
  let rec search node =
    match node with
    | Leaf -> ()
    | Node { idx; x; y; z; split; left; right } ->
        let d = sq_dist x y z qx qy qz in
        if d <= max_dist_sq then results := (idx, d) :: !results;
        let q_split = coord split qx qy qz in
        let p_split = coord split x y z in
        let diff = q_split -. p_split in
        let near, far = if diff < 0.0 then (left, right) else (right, left) in
        search near;
        if diff *. diff <= max_dist_sq then search far
  in
  search tree.root;
  !results
