(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type range = { start : int; count : int; step : int }
type window = { axis : int; size : int; step : int; dilation : int }

type t =
  | Reshape of int iarray
  | Broadcast of int iarray
  | Permute of int iarray
  | Slice of range iarray
  | Window of window iarray

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* [n] is the number of elements of [s]. *)
let reshape s n s' =
  Shape.check_rank "Move.Reshape" (Iarray.length s');
  let n' = Shape.numel "Move.Reshape" s' in
  if n <> n' then
    invalid_argf "Move.Reshape: %a has %d elements, %a has %d" Shape.pp s n
      Shape.pp s' n';
  s'

let broadcast s s' =
  let r = Iarray.length s and r' = Iarray.length s' in
  Shape.check_rank "Move.Broadcast" r';
  ignore (Shape.numel "Move.Broadcast" s');
  if r' < r then
    invalid_argf "Move.Broadcast: %a has fewer axes than %a" Shape.pp s'
      Shape.pp s;
  for i = 0 to r - 1 do
    let d = Iarray.get s i and d' = Iarray.get s' (r' - r + i) in
    if d <> 1 && d <> d' then
      invalid_argf
        "Move.Broadcast: %a does not broadcast to %a: axis %d has %d, neither \
         1 nor %d"
        Shape.pp s Shape.pp s' i d d'
  done;
  s'

(* The axes [p] names are marked in the bits of an int: a rank is at most
   [Shape.max_rank], below an int's width. *)
let permute s p =
  let r = Iarray.length s in
  if Iarray.length p <> r then
    invalid_argf "Move.Permute: %a has %d axes, not %d" Shape.pp p
      (Iarray.length p) r;
  let seen = ref 0 in
  for i = 0 to r - 1 do
    let a = Iarray.get p i in
    if a < 0 || a >= r then
      invalid_argf
        "Move.Permute: %a is not a permutation of %d axes: entry %d, %d, is \
         not an axis of %d"
        Shape.pp p r i a r;
    if !seen land (1 lsl a) <> 0 then
      invalid_argf
        "Move.Permute: %a is not a permutation of %d axes: entry %d repeats \
         axis %d"
        Shape.pp p r i a;
    seen := !seen lor (1 lsl a)
  done;
  let s' = Shape.zeros r in
  for i = 0 to r - 1 do
    s'.(i) <- Iarray.get s (Iarray.get p i)
  done;
  Shape.freeze s'

(* Whether element [count - 1] of [r] lies in an axis of extent [d], with
   [r.start] in it: [(count - 1)·|step|] is compared through a division, so no
   product overflows. *)
let reaches d (r : range) =
  let room =
    if r.step > 0 then (d - 1 - r.start) / r.step
    else if r.step = min_int then 0
    else r.start / -r.step
  in
  r.count - 1 <= room

let slice s rs =
  let r = Iarray.length s in
  if Iarray.length rs <> r then
    invalid_argf "Move.Slice: %d ranges for %d axes" (Iarray.length rs) r;
  Iarray.init r (fun i ->
      let (x : range) = Iarray.get rs i and d = Iarray.get s i in
      if x.step = 0 then invalid_argf "Move.Slice: axis %d has step 0" i;
      if x.count < 0 then
        invalid_argf "Move.Slice: axis %d has count %d" i x.count;
      if x.count > 0 && (x.start < 0 || x.start >= d || not (reaches d x)) then
        invalid_argf
          "Move.Slice: axis %d of extent %d has no elements %d + j·%d for j < \
           %d"
          i d x.start x.step x.count;
      x.count)

(* The number of windows of [w] along an axis of extent [d]: none where the
   dilated window, [dilation·(size - 1) + 1] elements, is longer than the axis.
   The reach is compared through a division, so no product overflows. *)
let windows d (w : window) =
  if d < 1 || w.size - 1 > (d - 1) / w.dilation then 0
  else ((d - 1 - (w.dilation * (w.size - 1))) / w.step) + 1

let window s ws =
  let r = Iarray.length s and k = Iarray.length ws in
  Shape.check_rank "Move.Window" (r + k);
  let s' = Array.init (r + k) (fun i -> if i < r then Iarray.get s i else 0) in
  Iarray.iteri
    (fun j (w : window) ->
      if w.axis < 0 || w.axis >= r then
        invalid_argf "Move.Window: axis %d of %d" w.axis r;
      if j > 0 && w.axis <= (Iarray.get ws (j - 1)).axis then
        invalid_argf
          "Move.Window: axes are not strictly increasing: window %d's axis %d \
           is not after window %d's axis %d"
          j w.axis (j - 1) (Iarray.get ws (j - 1)).axis;
      if w.size < 1 || w.step < 1 || w.dilation < 1 then
        invalid_argf
          "Move.Window: window %d has size %d, step %d and dilation %d; give \
           at least 1"
          j w.size w.step w.dilation;
      s'.(w.axis) <- windows (Iarray.get s w.axis) w;
      s'.(r + j) <- w.size)
    ws;
  let s' = Shape.freeze s' in
  ignore (Shape.numel "Move.Window" s');
  s'

(* An argument of more than [max_rank] axes has no layout, and [permute]'s bit
   set tells only that many axes apart: refuse it before any movement. *)
let shape m s =
  Shape.check_rank "Move.shape" (Iarray.length s);
  let n = Shape.numel "Move.shape" s in
  match m with
  | Reshape s' -> reshape s n s'
  | Broadcast s' -> broadcast s s'
  | Permute p -> permute s p
  | Slice rs -> slice s rs
  | Window ws -> window s ws
