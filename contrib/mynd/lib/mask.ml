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

(* Distances *)

(* Felzenszwalb and Huttenlocher's separable form (§2.2), with every value an
   exact integer. Rows first: per pixel, the nearest true column in its row
   (the left one on a tie) and the square [f] of its offset. Then down each
   column, D(i) = min over i' of (i - i')² + f(i'), and the least i' attaining
   it. That cost is Monge, c(i, a) + c(j, b) - c(i, b) - c(j, a) =
   2 (j - i)(a - b) <= 0 for i < j and a < b, so the least minimiser i'(i)
   never decreases as i grows. The column pass divides and conquers over it
   with a static recursion tree that depends on [h] alone: each level finds
   the least minimiser of the middle row of each subproblem, and the rows
   above search only up to it, those below only from it. A level's search
   ranges overlap only at their ends, so its candidates fit a buffer of
   [h + n] a column for [n] subproblems, laid out by a cumsum of the ranges'
   lengths; scatters of [Min] give each subproblem's least cost and least
   minimiser. [⌈log₂ (h + 1)⌉] levels, with no loop and no host read, so it
   compiles and maps.

   Index arrays (rows, columns, range ends, lengths, offsets, segment
   numbers) are int32, which holds every value for the sides below 2²⁹ that
   [distance] and [closest] accept. With [S = h + n <= 2h]
   (a level-ℓ subproblem has at most ⌊h / 2^ℓ⌋ rows, so a level runs only
   while [n = 2^ℓ <= h]): rows lie in [-1, h], [h] standing for "no
   minimiser"; the non-empty ranges of a level are ordered and meet only at
   their ends (a parent [clo, chi] with least minimiser [a] in it has
   children [clo, a] and [a, chi]), so their lengths sum below [S]; and a
   candidate row before its clamp lies in (-2h, 3h). Pass one's columns lie
   in [-1, 2w], [2w] standing for "no true pixel on that side". So int32
   holds them all while 3h and 2w are below 2³¹. Costs are int64: a real
   cost is at most [(h - 1)² + (w - 1)² < 2⁵⁹], below the sentinels [none]
   and [beyond]. A position a gather or a scatter reads is cast to int64
   where it is used. *)

(* The sides [distance] and [closest] accept are below [2^side_bits]. *)
let side_bits = 29

let none = Int64.shift_left 1L 60 (* no true pixel: FH's infinity *)
let beyond = Int64.shift_left 1L 61 (* a buffer slot past a level's ranges *)

(* The recursion tree over [h] rows: per level, each subproblem's rows [lo,
   hi] (empty where [lo > hi]) and its middle row. *)
let levels h =
  let rec go ivs acc =
    if not (List.exists (fun (lo, hi) -> lo <= hi) ivs) then List.rev acc
    else
      let mid (lo, hi) = if lo <= hi then (lo + hi) / 2 else -1 in
      let mids = List.map mid ivs in
      let split (lo, hi) m =
        if lo <= hi then [ (lo, m - 1); (m + 1, hi) ]
        else [ (lo, hi); (lo, hi) ]
      in
      go (List.concat (List.map2 split ivs mids)) ((ivs, mids) :: acc)
  in
  go [ (0, h - 1) ] []

(* Pass one over masks [r; w]: [(f, c)], the int64 squared offset to the
   nearest true pixel of each pixel's row ([none] where the row has none) and
   its column. *)
let rows at m =
  let r = Nx.dim 0 m and w = Nx.dim 1 m in
  let k = Int32.of_int in
  let c v = Nx.scalar_at at Nx.int32 (k v) in
  let far = 2 * w in
  let j = Nx.broadcast_to [| r; w |] (Nx.arange_at at Nx.int32 0 w 1) in
  let left = Nx.cummax ~axis:1 (Nx.where m j (c (-1))) in
  let right =
    Nx.flip ~axes:[ 1 ]
      (Nx.cummin ~axis:1 (Nx.flip ~axes:[ 1 ] (Nx.where m j (c far))))
  in
  let dl = Nx.where (Nx.greater_equal_s left (k 0)) (Nx.sub j left) (c far) in
  let dr = Nx.where (Nx.less_s right (k far)) (Nx.sub right j) (c far) in
  let col = Nx.where (Nx.less_equal dl dr) left right in
  let g = Nx.cast Nx.int64 (Nx.minimum dl dr) in
  let f =
    Nx.where
      (Nx.less_s g (Int64.of_int far))
      (Nx.mul g g)
      (Nx.scalar_at at Nx.int64 none)
  in
  (f, col)

(* Pass two down each lane of [f], [[l; h]]: [(d, a)], the least
   [(i - i')² + f(i')] at each row [i] and the least [i'] attaining it. *)
let columns at f =
  let l = Nx.dim 0 f and h = Nx.dim 1 f in
  let k = Int32.of_int in
  let full n v = Nx.full_at at Nx.int32 [| l; n |] (k v) in
  let host dt a = Nx.place at (Nx.create dt [| Array.length a |] a) in
  let long x = Nx.cast Nx.int64 x in
  let level (d, a, lo, hi) (ivs, mids) =
    let n = List.length ivs in
    let open_ = Array.of_list (List.map (fun (lo, hi) -> lo <= hi) ivs) in
    let mids = Array.of_list mids in
    let ne = Nx.reshape [| 1; n |] (host Nx.bool open_) in
    let mid = host Nx.int32 (Array.map (fun m -> k (Int.max m 0)) mids) in
    (* Set positions of the middle rows; [h], outside, for an empty one. *)
    let at_mid =
      Nx.broadcast_to [| l; n |]
        (Nx.reshape [| 1; n |]
           (host Nx.int64
              (Array.mapi
                 (fun i m -> Int64.of_int (if open_.(i) then m else h))
                 mids)))
    in
    (* Lay the ranges [lo, hi] out one after another in a buffer of [size]
       slots: [seg] is the subproblem of each slot, [row] its candidate. *)
    let size = h + n in
    let len = Nx.where ne (Nx.add_s (Nx.sub hi lo) (k 1)) (full 1 0) in
    let ends = Nx.cumsum ~axis:1 len in
    let starts = Nx.sub ends len in
    let total = Nx.slice [ Nx.A; Nx.R (n - 1, n) ] ends in
    let marks =
      Nx.scatter ~combine:Nx.Add ~axis:1 (long starts) (Nx.ones_like starts)
        (full size 0)
    in
    let seg = Nx.sub_s (Nx.cumsum ~axis:1 marks) (k 1) in
    let seg = long (Nx.minimum_s seg (k (n - 1))) in
    let s = Nx.reshape [| 1; size |] (Nx.arange_at at Nx.int32 0 size 1) in
    let valid = Nx.less s total in
    let row =
      Nx.add
        (Nx.take_along_axis ~axis:1 seg lo)
        (Nx.sub s (Nx.take_along_axis ~axis:1 seg starts))
    in
    let row = Nx.clamp ~min:(k 0) ~max:(k (h - 1)) row in
    let dy = Nx.cast Nx.int64 (Nx.sub (Nx.take ~axis:0 seg mid) row) in
    let cost =
      Nx.add (Nx.mul dy dy) (Nx.take_along_axis ~axis:1 (long row) f)
    in
    let cost = Nx.where valid cost (Nx.scalar_at at Nx.int64 beyond) in
    let best =
      Nx.scatter ~combine:Nx.Min ~axis:1 seg cost
        (Nx.full_at at Nx.int64 [| l; n |] beyond)
    in
    let hit =
      Nx.logical_and valid
        (Nx.equal cost (Nx.take_along_axis ~axis:1 seg best))
    in
    let arg =
      Nx.scatter ~combine:Nx.Min ~axis:1 seg
        (Nx.where hit row (Nx.scalar_at at Nx.int32 (k h)))
        (full n h)
    in
    (* The children: [lo, arg] above the middle row, [arg, hi] below. An
       empty subproblem's children are empty, and [ne] drops their ranges. *)
    let pair a b = Nx.reshape [| l; 2 * n |] (Nx.stack ~axis:2 [ a; b ]) in
    ( Nx.scatter ~axis:1 at_mid best d,
      Nx.scatter ~axis:1 at_mid arg a,
      pair lo arg,
      pair arg hi )
  in
  let d, a, _, _ =
    List.fold_left level
      ( Nx.full_at at Nx.int64 [| l; h |] none,
        full h (-1),
        full 1 0,
        full 1 (h - 1) )
      (levels h)
  in
  (d, a)

(* [n] and the index of the closest pixel, over masks [b; h; w], both int64
   [[b; h; w]], [none] and [-1] where a mask has no true pixel. *)
let nearest_in at m b h w =
  let f, col = rows at (Nx.reshape [| b * h; w |] m) in
  let by_column x =
    Nx.reshape [| b * w; h |]
      (Nx.transpose ~axes:[ 0; 2; 1 ] (Nx.reshape [| b; h; w |] x))
  in
  let d, a = columns at (by_column f) in
  let by_row x =
    Nx.copy (Nx.transpose ~axes:[ 0; 2; 1 ] (Nx.reshape [| b; w; h |] x))
  in
  let d = by_row d and a = Nx.cast Nx.int64 (by_row a) in
  let j =
    Nx.take_along_axis ~axis:1
      (Nx.clamp ~min:0L ~max:(Int64.of_int (h - 1)) a)
      (Nx.reshape [| b; h; w |] col)
  in
  let some = Nx.less_s d none in
  let index = Nx.add (Nx.mul_s a (Int64.of_int w)) (Nx.cast Nx.int64 j) in
  (some, d, Nx.where some index (Nx.scalar_at at Nx.int64 (-1L)))

(* [nearest fn m] is [(some, n, index)] over masks [[…; h; w; 1]], each of
   their shape, or [None] for masks with no pixel. It reads the shape alone
   before it raises. *)
let nearest fn m =
  mask_shape fn m;
  let s = Nx.shape m in
  let r = Array.length s in
  let h = s.(r - 3) and w = s.(r - 2) in
  if h >= 1 lsl side_bits || w >= 1 lsl side_bits then
    invalid_argf "%s: masks of %d rows and %d columns, past 2^%d" fn h w
      side_bits;
  let b = Array.fold_left ( * ) 1 (Array.sub s 0 (r - 3)) in
  if b * h * w = 0 then None
  else
    let some, n, index = nearest_in (Nx.placement m) m b h w in
    Some (Nx.reshape s some, Nx.reshape s n, Nx.reshape s index)

(* The float64 root of an exact [n < 2⁵³] is its correctly rounded root, and
   one rounding into a format of at most 25 bits keeps it correctly rounded
   for [n < 2⁵²]: a midpoint [m] of that format with a fraction lies at least
   [2^(k - 2p - 2)] from [√n] in [\[2ᵏ, 2ᵏ⁺¹)], and an integer one other
   than [√n] at least [2^(-k - 2)], both beyond float64's half ulp
   [2^(k - 53)] there. Where [√n] is that integer, float64 holds it exactly
   and the cast alone rounds it. *)
let root dt n = Nx.cast dt (Nx.sqrt (Nx.cast Nx.float64 n))

let distance dt m =
  match nearest "Mynd.Mask.distance" m with
  | None -> Nx.zeros_at (Nx.placement m) dt (Nx.shape m)
  | Some (some, n, _) ->
      let greatest = Nx.scalar_at (Nx.placement m) dt (Nx.Dtype.max_value dt) in
      Nx.where some (root dt n) greatest

let closest m =
  match nearest "Mynd.Mask.closest" m with
  | None -> Nx.zeros_at (Nx.placement m) Nx.int64 (Nx.shape m)
  | Some (_, _, index) -> index
