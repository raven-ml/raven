(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Image

type connectivity = [ `Four | `Eight ]

(* IoU *)

let iou dt a b =
  let fn = "Mynd.Mask.iou" in
  mask_shape fn a;
  mask_shape fn b;
  if Nx.ndim a < 4 || Nx.ndim b < 4 then
    invalid_argf "%s: masks have axes [...; n; h; w; 1]" fn;
  let flat m =
    let s = Nx.shape m in
    let r = Array.length s in
    Nx.reshape
      (Array.append (Array.sub s 0 (r - 3)) [| s.(r - 3) * s.(r - 2) |])
      (Nx.cast Nx.int64 m)
  in
  let a = flat a and b = flat b in
  let both = Nx.matmul a (Nx.swapaxes (-1) (-2) b) in
  let count m = Nx.sum ~axes:[ -1 ] m in
  let either =
    Nx.sub
      (Nx.add
         (Nx.unsqueeze ~axes:[ -1 ] (count a))
         (Nx.unsqueeze ~axes:[ -2 ] (count b)))
      both
  in
  let empty = Nx.equal_s either 0L in
  let safe = Nx.where empty (Nx.ones_like either) either in
  (* The exact counts' quotient, rounded once into [dt]. float32 rounds it once
     for counts below 2²⁴. Through float64, a quotient of counts below 2ᵏ lies
     at least 2⁻ᵏ⁻ᵖ⁻¹ relatively from a midpoint of a float of [p] bits, beyond
     float64's rounding while [k + p + 1 < 53]: below 2²⁸ pixels for every
     float. *)
  let quotient (type w) (wide : (float, w) Nx.dtype) =
    Nx.cast dt (Nx.div (Nx.cast wide both) (Nx.cast wide safe))
  in
  let pixels = Nx.dim (-1) a in
  let r =
    if Nx.Dtype.equal dt Nx.float32 && pixels < 1 lsl 24 then
      quotient Nx.float32
    else quotient Nx.float64
  in
  Nx.where empty (Nx.zeros_like r) r

(* Components *)

(* Liu and Tarjan's algorithm R with two shortcuts a round over the pixel graph
   ("Simple Concurrent Connected Components Algorithms"). Every pixel starts as
   its own root. A round reads the parents [p] once: for each edge [{v, w}]
   between true pixels whose parent [p v] is a root and [p w < p v], the root
   takes the least such [p w] as its parent (parent-root-connect, a scatter of
   [Min]); then each pixel's parent becomes its grandparent, twice. The rounds end when none changes a parent, every tree
   then flat and rooted at its least pixel. Their §4.3: with two shortcuts Lemma
   23 holds for [b = 2^(1/3)], so no tree is active once [b^(k - 5) > 3n / 2],
   at most [6 + ⌊3 log₂ (3n / 2)⌋] rounds with the last, which changes
   nothing. *)

let neighbours = function
  | `Four -> [ (-1, 0); (0, -1); (0, 1); (1, 0) ]
  | `Eight ->
      [ (-1, -1); (-1, 0); (-1, 1); (0, -1); (0, 1); (1, -1); (1, 0); (1, 1) ]

let component_rounds n =
  6
  + Float.to_int
      (Float.ceil (3. *. Float.log2 (1.5 *. Float.of_int (Int.max 1 n))))

let label conn m =
  let fn = "Mynd.Mask.components" in
  mask_shape fn m;
  let s = Nx.shape m in
  let r = Array.length s in
  let b = Array.sub s 0 (r - 3) and h = s.(r - 3) and w = s.(r - 2) in
  let n = h * w in
  let fg = Nx.reshape (Array.append b [| h; w |]) m in
  let flat x = Nx.reshape (Array.append b [| n |]) x in
  let grid x = Nx.reshape (Array.append b [| h; w |]) x in
  let none = Int64.max_int in
  let pads =
    Array.append (Array.map (fun _ -> (0, 0)) b) [| (1, 1); (1, 1) |]
  in
  let offsets = neighbours conn in
  let at_offset q (dy, dx) =
    Nx.slice
      (List.init (Array.length b) (fun _ -> Nx.A)
      @ [ Nx.R (1 + dy, 1 + dy + h); Nx.R (1 + dx, 1 + dx + w) ])
      q
  in
  let step (p, (_, k)) =
    (* The least parent of a true neighbour, [none] where there is none. *)
    let q = Nx.where fg (grid p) (Nx.full_like (grid p) none) in
    let q = Nx.pad pads none q in
    let least =
      List.fold_left
        (fun acc o -> Nx.minimum acc (at_offset q o))
        (at_offset q (List.hd offsets))
        (List.tl offsets)
    in
    let least = flat (Nx.where fg least (Nx.full_like least none)) in
    let root = Nx.equal (Nx.take_along_axis ~axis:(-1) p p) p in
    let hook = Nx.logical_and root (Nx.less least p) in
    let target = Nx.where hook p (Nx.full_like p (-1L)) in
    let p' = Nx.scatter ~combine:Nx.Min ~axis:(-1) target least p in
    let shortcut p = Nx.take_along_axis ~axis:(-1) p p in
    let p' = shortcut (shortcut p') in
    (p', (Nx.any (Nx.not_equal p' p), Nx.add_s k 1l))
  in
  let at = Nx.placement m in
  let start =
    Nx.copy
      (Nx.broadcast_to (Array.append b [| n |])
         (Nx.arange_at at Nx.int64 0 n 1))
  in
  let carry = Nx.Ptree.(pair tensor (pair tensor tensor)) in
  let p, (_, rounds) =
    Rune.iterate carry ~max:(component_rounds n)
      ~until:(fun (_, (changed, _)) -> Nx.logical_not changed)
      step
      (start, (Nx.scalar_at at Nx.bool true, Nx.scalar_at at Nx.int32 0l))
  in
  let labels = Nx.where (flat fg) p (Nx.full_like p (-1L)) in
  (Nx.reshape s labels, rounds)

let components conn m = fst (label conn m)
let components_rounds = label

(* The components of [m] that hold a marker pixel: each mask's labels are
   indices into its own flattened pixels, so a scatter of [Max] along the last
   axis marks the labels that a marker pixel inside [m] carries, per mask, and a
   gather reads each pixel's mark. A label [-1] writes nothing and reads
   [false]. *)
let reconstruct conn ~marker m =
  let fn = "Mynd.Mask.reconstruct" in
  mask_shape fn m;
  if Nx.shape marker <> Nx.shape m then
    invalid_argf "%s: a marker of shape [%s] for masks [%s]" fn
      (dims (Nx.shape marker))
      (dims (Nx.shape m));
  let s = Nx.shape m in
  let r = Array.length s in
  let flat x =
    Nx.reshape
      (Array.append (Array.sub s 0 (r - 3)) [| s.(r - 3) * s.(r - 2) |])
      x
  in
  let l = flat (components conn m) in
  let marked =
    Nx.scatter ~combine:Nx.Max ~axis:(-1) l
      (flat (Nx.logical_and marker m))
      (Nx.zeros_like (flat m))
  in
  Nx.reshape s (Nx.take_along_axis ~axis:(-1) l marked)
