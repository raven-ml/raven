(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type range = { start : int; count : int; step : int }
type window = { axis : int; size : int; step : int; dilation : int }

type t =
  | Reshape of int array
  | Broadcast of int array
  | Permute of int array
  | Slice of range array
  | Window of window array

let invalid_argf fmt = Format.kasprintf invalid_arg fmt
let pp_ints = Shape.pp
let numel = Shape.numel
let check_rank = Shape.check_rank

(* [n] is the number of elements of [s]. A caller's array is read once: [s'] is
   copied before it is checked, since another domain may write it. *)
let reshape s n s' =
  let s' = Shape.copy s' in
  check_rank "Move.Reshape" (Array.length s');
  let n' = numel "Move.Reshape" s' in
  if n <> n' then
    invalid_argf "Move.Reshape: %a has %d elements, %a has %d" pp_ints s n
      pp_ints s' n';
  s'

(* A caller's array is read once, as in [reshape]. *)
let broadcast s s' =
  let s' = Shape.copy s' in
  let r = Array.length s and r' = Array.length s' in
  check_rank "Move.Broadcast" r';
  ignore (numel "Move.Broadcast" s');
  if r' < r then
    invalid_argf "Move.Broadcast: %a has fewer axes than %a" pp_ints s' pp_ints
      s;
  for i = 0 to r - 1 do
    let d = s.(i) and d' = s'.(r' - r + i) in
    if d <> 1 && d <> d' then
      invalid_argf
        "Move.Broadcast: %a does not broadcast to %a: axis %d has %d, neither \
         1 nor %d"
        pp_ints s pp_ints s' i d d'
  done;
  s'

(* The axes [p] names are marked in the bits of an int: a rank is at most
   [Shape.max_rank], below an int's width. *)
let permute s p =
  let r = Array.length s in
  if Array.length p <> r then
    invalid_argf "Move.Permute: %a has %d axes, not %d" pp_ints p
      (Array.length p) r;
  let seen = ref 0 in
  for i = 0 to r - 1 do
    let a = p.(i) in
    if a < 0 || a >= r then
      invalid_argf
        "Move.Permute: %a is not a permutation of %d axes: entry %d, %d, is \
         not an axis of %d"
        pp_ints p r i a r;
    if !seen land (1 lsl a) <> 0 then
      invalid_argf
        "Move.Permute: %a is not a permutation of %d axes: entry %d repeats \
         axis %d"
        pp_ints p r i a;
    seen := !seen lor (1 lsl a)
  done;
  let s' = Shape.zeros r in
  for i = 0 to r - 1 do
    s'.(i) <- s.(p.(i))
  done;
  s'

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
  let r = Array.length s in
  if Array.length rs <> r then
    invalid_argf "Move.Slice: %d ranges for %d axes" (Array.length rs) r;
  Array.mapi
    (fun i (x : range) ->
      let d = s.(i) in
      if x.step = 0 then invalid_argf "Move.Slice: axis %d has step 0" i;
      if x.count < 0 then
        invalid_argf "Move.Slice: axis %d has count %d" i x.count;
      if x.count > 0 && (x.start < 0 || x.start >= d || not (reaches d x)) then
        invalid_argf
          "Move.Slice: axis %d of extent %d has no elements %d + j·%d for j < \
           %d"
          i d x.start x.step x.count;
      x.count)
    rs

(* The number of windows of [w] along an axis of extent [d], or [-1] if [w] does
   not fit in it. *)
let windows d (w : window) =
  if w.size < 1 || w.step < 1 || w.dilation < 1 || d < 1 then -1
  else if w.size - 1 > (d - 1) / w.dilation then -1
  else ((d - 1 - (w.dilation * (w.size - 1))) / w.step) + 1

let window s ws =
  let r = Array.length s and k = Array.length ws in
  check_rank "Move.Window" (r + k);
  let s' = Array.append s (Array.make k 0) in
  Array.iteri
    (fun j (w : window) ->
      if w.axis < 0 || w.axis >= r then
        invalid_argf "Move.Window: axis %d of %d" w.axis r;
      if j > 0 && w.axis <= ws.(j - 1).axis then
        invalid_argf
          "Move.Window: axes are not strictly increasing: window %d's axis %d \
           is not after window %d's axis %d"
          j w.axis (j - 1) ws.(j - 1).axis;
      let n = windows s.(w.axis) w in
      if n < 0 then
        invalid_argf
          "Move.Window: a window of size %d, step %d and dilation %d does not \
           fit in axis %d of extent %d"
          w.size w.step w.dilation w.axis s.(w.axis);
      s'.(w.axis) <- n;
      s'.(r + j) <- w.size)
    ws;
  ignore (numel "Move.Window" s');
  s'

(* An argument of more than [max_rank] axes has no layout, and [permute]'s bit
   set tells only that many axes apart: refuse it before any movement. *)
let shape m s =
  check_rank "Move.shape" (Array.length s);
  let n = numel "Move.shape" s in
  match m with
  | Reshape s' -> reshape s n s'
  | Broadcast s' -> broadcast s s'
  | Permute p -> permute s p
  | Slice rs -> slice s rs
  | Window ws -> window s ws
