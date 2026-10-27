(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module Arith = Lower_arith
module Bits = Lower_bits

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

let extents u = List.init (Shape.ndim u) (Bits.extent u)
let dims s = List.map (fun n -> Ops.Int n) s
let numel s = List.fold_left ( * ) 1 s
let int u n = Shape.const_like u (`Int (Bigint.of_int n))

let broadcast c s =
  Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) s)) (dims s)

let zeros t s = broadcast (Ops.const ~dtype:t (`Int Bigint.zero)) s

(* [laid s axis v] is the vector [v] laid along [axis] of the shape [s]. *)
let laid s axis v =
  let r = List.length s in
  let along = List.init r (fun i -> if i = axis then List.nth s i else 1) in
  Shape.expand (Shape.reshape v (dims along)) (dims s)

(* [joined axis a b] is [a] then [b] along [axis]. Shape.cat adds zero-padded
   pieces, which would turn a [-0] into [+0]: floats join as their bits. *)
let joined axis a b =
  match Ops.dtype a with
  | t when Dtype.is_float t ->
      let u = Bits.unsigned t in
      Ops.bitcast (Shape.cat ~axis (Ops.bitcast a u) [ Ops.bitcast b u ]) t
  | _ -> Shape.cat ~axis a [ b ]

(* [negative_zero u] is whether the float [u] is [-0]: its sign bit alone. *)
let negative_zero u =
  let b = Bits.bits u in
  let w = 8 * Dtype.itemsize (Ops.dtype u) in
  Ops.eq b (Shape.const_like b (`Int (Bigint.shift_left Bigint.one (w - 1))))

(* Gathers

   A gather is tolk's index of the operand's elements, flattened, at each
   element's offset: its own index's along the other axes, its position's along
   [axis]. A position outside the axis is read at [0], so that every read is in
   bounds, and selected as zero. *)

let take u axis p =
  let s = extents u and t = extents p in
  let n = List.nth s axis in
  if n = 0 || numel t = 0 then zeros (Ops.dtype u) t
  else
    let inside =
      Ops.lt (Ops.bitcast p Uint64)
        (Shape.const_like ~dtype:Uint64 p (`Int (Bigint.of_int n)))
    in
    let at = Ops.cast (Ops.where inside p (int p 0)) Weak_int in
    let stride d = numel (List.filteri (fun i _ -> i > d) s) in
    let term d =
      let v =
        if d = axis then at
        else laid t d (Shape.arange ~dtype:Weak_int (List.nth t d))
      in
      match stride d with 1 -> v | k -> Ops.mul v (Ops.int k)
    in
    let offset =
      List.fold_left
        (fun o d -> Ops.add o (term d))
        (term 0)
        (List.init (List.length s - 1) succ)
    in
    let read = Ops.index (Shape.reshape u [ Ops.Int (numel s) ]) [ offset ] in
    Ops.where inside read (int read 0)

let gather d ~axis idx x =
  Arith.of_parts d (List.map (fun u -> take u axis idx) (Arith.parts d x))

(* Scatters

   Each element of [into] meets each update along [axis] in a mask along a new
   last axis, true where the update's position is the element's: n x k work for
   n positions and k updates along [axis]. Updates that land on one element
   differ only along [axis], so C order of updates is their order along it. *)

(* [summed ca ~landed xs us] is, in the carrier [ca], the sum of each element of
   [xs] and the updates [us] that land on it, [landed] their mask along a last
   axis. It is [-0] where every term is, which a sum from tolk's [+0]
   accumulator would make [+0]. *)
let summed ca ~landed xs us =
  let r = Shape.ndim landed - 1 in
  let terms x u = joined r x (Ops.where landed u (int u 0)) in
  let sums =
    Lower_reduce.reduce (Monoid Sum) [ r ] ~src:ca ca
      (Arith.of_parts ca (List.map2 terms xs us))
  in
  let signed (x, u) y =
    if not (Dtype.is_float (Ops.dtype y)) then y
    else
      let other =
        Ops.bitwise_and landed (Ops.logical_not (negative_zero u))
      in
      let other = Shape.rop (Ops.cast other Uint8) Op.Max [ r ] in
      let x = Shape.reshape x (Shape.shape other) in
      let none = Ops.eq other (int other 0) in
      let all = Ops.bitwise_and (negative_zero x) none in
      Ops.where all (Shape.const_like y (`Float (-0.))) y
  in
  Arith.of_parts ca
    (List.map2 signed (List.combine xs us) (Arith.parts ca (List.hd sums)))

let scatter (D.Any dt as d) (c : Nx_kernel.combine) ~axis ~idx ~updates into =
  let s = extents (List.hd (Arith.parts d into)) and t = extents idx in
  let r = List.length s in
  let n = List.nth s axis and k = List.nth t axis in
  if n = 0 || numel t = 0 then into
  else
    let wide = s @ [ k ] in
    (* A node of [idx]'s shape moved to [wide], its axis [axis] last. *)
    let spread u =
      let others = List.filter (( <> ) axis) (List.init r Fun.id) in
      let one = List.mapi (fun i e -> if i = axis then 1 else e) s @ [ k ] in
      Shape.expand
        (Shape.reshape (Shape.permute u (others @ [ axis ])) (dims one))
        (dims wide)
    in
    let own u = Shape.reshape u (dims (s @ [ 1 ])) in
    let mask =
      Ops.eq (spread idx) (laid wide axis (Shape.arange ~dtype:Int64 n))
    in
    let landed =
      let m = Shape.rop (Ops.cast mask Uint8) Op.Max [ r ] in
      Ops.ne m (int m 0)
    in
    let kept y =
      Arith.of_parts d
        (List.map2 (Ops.where landed) (Arith.parts d y) (Arith.parts d into))
    in
    let extreme m =
      (* [into]'s element, first, then each update, the element standing in
         for one that does not land. *)
      let terms x u =
        let x = own x in
        joined r x (Ops.where mask (spread u) (Shape.expand x (dims wide)))
      in
      let xs = List.map2 terms (Arith.parts d into) (Arith.parts d updates) in
      let y = Lower_reduce.reduce (Monoid m) [ r ] ~src:d d in
      kept (List.hd (y (Arith.of_parts d xs)))
    in
    match c with
    | Set ->
        (* The last update to land, of highest position along [axis], wins:
           its bits are picked among zeros by an unsigned maximum. *)
        let order = laid wide r (Shape.arange ~dtype:Int64 k) in
        let latest =
          Shape.rop (Ops.where mask order (int order (-1))) Op.Max [ r ]
        in
        let last = Ops.eq order (Shape.expand (own latest) (dims wide)) in
        let pick u =
          let b = Bits.bits (spread u) in
          let b = Shape.rop (Ops.where last b (int b 0)) Op.Max [ r ] in
          Ops.bitcast b (Ops.dtype u)
        in
        kept (Arith.of_parts d (List.map pick (Arith.parts d updates)))
    | Max -> extreme Max
    | Min -> extreme Min
    | Add ->
        if D.is D.Boolean dt then unlowered "a boolean Add Scatter";
        let ca = Nx_kernel.carrier d in
        let lift u = Arith.parts ca (Arith.cast ~src:d ca u) in
        let xs = List.map own (lift into) in
        let us = List.map spread (lift updates) in
        kept (Arith.cast ~src:ca d (summed ca ~landed:mask xs us))

(* Folds

   A fold undoes a padded load's movements in reverse, each by its adjoint: a
   window's sums the windows' elements back into place, the ends' pads what
   they cropped and crops what they padded, and the interior padding's keeps
   every [k + 1]th element. Each part sums in the carrier, an integer on an
   unsigned pattern of 32 or 64 bits as {!Lower_reduce} folds integers, and
   rounds once. The same movements over a mark per term, [2] for a term other
   than [-0] and [1] for [-0], folded by their maximum, find where every term
   is [-0], which the sum, from tolk's [+0] accumulator, makes [+0]. *)

let ceil_div a b = (a + b - 1) / b

(* [only r i b] is [b] at axis [i] of [r] axes, [None] elsewhere. *)
let only r i b = List.init r (fun j -> if j = i then Some b else None)

(* [rows u i k] is [u] with its axis [i] split into rows of [k]. *)
let rows u i k =
  let e = extents u in
  let split = List.mapi (fun j d -> if j = i then [ d / k; k ] else [ d ]) e in
  Shape.reshape u (dims (List.concat split))

let ints a b = (Ops.Int a, Ops.Int b)

(* [unwindow op u w s] is the adjoint of the window [w] over a value of shape
   [s]: [u]'s elements, of the windowed shape, folded by [op] into the place
   each reads. The axis is laid out as rows one element longer than
   [w.dilation] apart from a whole number of copies of it, so that the
   window's element [j] at window [l] lands [j·dilation + l·step] into a copy:
   the copies' sum is the fold. *)
let unwindow op u (w : Nx_array.Move.window) s =
  let a = w.axis and r = List.length s in
  let n = List.nth s a and l = Bits.extent u a in
  if n = 0 || l = 0 then zeros (Ops.dtype u) s
  else
    let others = List.filter (( <> ) a) (List.init r Fun.id) in
    let o = List.map (List.nth s) others in
    let scale = Int.max 1 (ceil_div ((l * w.step) - w.dilation) n) in
    let row = (n * scale) + w.dilation in
    let copies = ceil_div (w.size * row) n in
    let reshape x e = Shape.reshape x (dims (o @ e)) in
    let pad_to x e = Shape.pad_to x (List.map Option.some (dims (o @ e))) in
    let x = Shape.permute u (others @ [ r; a ]) in
    let x = reshape x [ w.size; l; 1 ] in
    let x = pad_to x [ w.size; l; w.step ] in
    let x = reshape x [ w.size; l * w.step ] in
    let x = pad_to x [ w.size; row ] in
    let x = reshape x [ w.size * row ] in
    let x = pad_to x [ copies * n ] in
    let x = Shape.rop (reshape x [ copies; n ]) op [ r - 1 ] in
    let back i = if i = a then r - 1 else if i < a then i else i - 1 in
    Shape.permute x (List.init r back)

(* [unends lo hi s u] is the adjoint of padding a value of shape [s] by [lo]
   and [hi]: [u] padded with zeros where the load cropped, then cropped where
   it padded. *)
let unends lo hi s u =
  let at f = List.init (List.length s) (fun i -> Some (f i)) in
  let cropped i a = Int.max 0 (-Iarray.get a i) in
  let u = Shape.pad u (at (fun i -> ints (cropped i lo) (cropped i hi))) in
  Shape.shrink u
    (at (fun i ->
         let b = Int.max 0 (Iarray.get lo i) in
         ints b (b + List.nth s i)))

(* [uninterior k s u] is the adjoint of spacing a value of shape [s] by [k]:
   every [k.(i) + 1]th element of [u] along each axis [i]. *)
let uninterior k s u =
  let r = List.length s in
  let every u i k =
    if k = 0 || List.nth s i = 0 then u
    else
      let u = rows (Shape.pad u (only r i (ints 0 k))) i (k + 1) in
      let u = Shape.shrink u (only (r + 1) (i + 1) (ints 0 1)) in
      Shape.flatten ~start:i ~stop:(i + 1) u
  in
  let u = ref u in
  Iarray.iteri (fun i k -> u := every !u i k) k;
  !u

let fold (D.Any dt as d) (pad : Nx_kernel.pad) shape x =
  if D.is D.Boolean dt then unlowered "a boolean Fold";
  let s0 = Iarray.to_list shape in
  let spaced i n =
    if n = 0 then 0 else n + ((n - 1) * Iarray.get pad.interior i)
  in
  let s1 = List.mapi spaced s0 in
  let s2 =
    List.mapi (fun i n -> n + Iarray.get pad.lo i + Iarray.get pad.hi i) s1
  in
  (* Each window with the shape it windows, the last first. *)
  let windows, _ =
    Iarray.fold_left
      (fun (ws, s) (w : Nx_array.Move.window) ->
        let m = Nx_array.Move.Window (Iarray.of_list [ w ]) in
        let s' = Iarray.to_list (Nx_array.Move.shape m (Iarray.of_list s)) in
        ((w, s) :: ws, s'))
      ([], s2) pad.windows
  in
  let back op u =
    let u = List.fold_left (fun u (w, s) -> unwindow op u w s) u windows in
    uninterior pad.interior s0 (unends pad.lo pad.hi s1 u)
  in
  let ca = Nx_kernel.carrier d in
  let part u =
    let t = Ops.dtype u in
    if not (Dtype.is_float t) then
      let wide : Dtype.t =
        if Dtype.itemsize t < 4 then Uint32 else Bits.unsigned t
      in
      let y = back Op.Add (Ops.cast u wide) in
      (* A cast, not a bitcast, back to a signed type: tolk renders a
         bitcast of a constant of 64 bits from a 32-bit literal. *)
      Ops.cast (Ops.cast y (Bits.unsigned t)) t
    else
      let y = back Op.Add u in
      let mark =
        Ops.where (negative_zero u) (int (Bits.bits u) 1) (int (Bits.bits u) 2)
      in
      let mark = back Op.Max (Ops.cast mark Uint8) in
      Ops.where (Ops.eq mark (int mark 1)) (Shape.const_like y (`Float (-0.))) y
  in
  let x = Arith.cast ~src:d ca x in
  Arith.cast ~src:ca d (Arith.of_parts ca (List.map part (Arith.parts ca x)))

(* Assemblies

   A piece is placed into the result's shape by the adjoint of its region's
   slice, which spaces its elements [step] apart and pads around them, and so
   is a mask of where it lies. Each piece in turn selects its elements where
   it lies over what came before. *)

(* [placed s ranges u] is [u] at the positions [ranges] keep of a shape [s],
   zeros elsewhere. *)
let placed s (ranges : Nx_array.Move.range iarray) u =
  let r = List.length s in
  let place u i ({ start; count; step } : Nx_array.Move.range) =
    let u, first, step =
      if step < 0 then (Shape.flip u [ i ], start + ((count - 1) * step), -step)
      else (u, start, step)
    in
    let span = ((count - 1) * step) + 1 in
    let u =
      if step = 1 then u
      else
        let gaps = only (r + 1) (i + 1) (ints 0 (step - 1)) in
        let u = Shape.pad (rows u i 1) gaps in
        let u = Shape.flatten ~start:i ~stop:(i + 1) u in
        Shape.shrink u (only r i (ints 0 span))
    in
    Shape.pad u (only r i (ints first (List.nth s i - first - span)))
  in
  let u = ref u in
  Iarray.iteri (fun i range -> u := place !u i range) ranges;
  !u

(* [words dt x] is the bits of each part of the element [x] of [dt], as
   unsigned integers in the host's order. *)
let words (type v s) (dt : (v, s) D.t) (x : v) =
  let b = Nx_kernel.Prog.bits dt x in
  let word b =
    let n = ref Bigint.zero in
    for i = String.length b - 1 downto 0 do
      n := Bigint.add (Bigint.shift_left !n 8) (Bigint.of_int (Char.code b.[i]))
    done;
    `Int !n
  in
  if not (D.is D.Complex dt) then [ word b ]
  else
    let h = String.length b / 2 in
    [ word (String.sub b 0 h); word (String.sub b h h) ]

(* A float assembles as its bits: tolk folds a constant float fill joined to
   another part by an addition, which quiets a signalling NaN. *)
let assemble (type v s) (dt : (v, s) D.t) (fill : v) shape pieces =
  let d = D.Any dt in
  let s = Iarray.to_list shape in
  let tdt = Arith.dtype d in
  let floats = Dtype.is_float tdt in
  let start =
    if floats then
      let tb = Bits.unsigned tdt in
      List.map (fun w -> broadcast (Ops.const ~dtype:tb w) s) (words dt fill)
    else
      let const c = broadcast (Ops.const ~dtype:tdt c) s in
      List.map const (Arith.fill dt fill)
  in
  let empty (r : Nx_array.Move.range) = r.count = 0 in
  let put acc (ranges, u) =
    if Iarray.exists empty ranges then acc
    else
      let parts = Arith.parts d u in
      let parts = if floats then List.map Bits.bits parts else parts in
      let ones = Shape.const_like ~dtype:Bool (List.hd parts) (`Bool true) in
      let mask = placed s ranges ones in
      List.map2 (fun a p -> Ops.where mask (placed s ranges p) a) acc parts
  in
  let y = Arith.of_parts d (List.fold_left put start pieces) in
  if floats then Ops.bitcast y tdt else y
