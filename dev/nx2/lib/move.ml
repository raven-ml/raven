(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let max_rank = 32

type range = { start : int; count : int; step : int }
type window = { axis : int; size : int; step : int; dilation : int }

type t =
  | Reshape of int array
  | Broadcast of int array
  | Permute of int array
  | Slice of range array
  | Window of window array

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_ints ppf a =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_array
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    a

(* The number of elements of [s], refusing a negative extent and a product that
   overflows: each product is formed only once it is known to fit. *)
let numel fn s =
  let n = ref 1 in
  for i = 0 to Array.length s - 1 do
    let d = s.(i) in
    if d < 0 then invalid_argf "%s: extent %d of %a is negative" fn d pp_ints s;
    if d <> 0 && !n > max_int / d then
      invalid_argf "%s: the number of elements of %a overflows" fn pp_ints s;
    n := !n * d
  done;
  !n

let check_rank fn r =
  if r > max_rank then invalid_argf "%s: rank %d exceeds %d" fn r max_rank

let reshape s s' =
  check_rank "Move.Reshape" (Array.length s');
  let n = numel "Move.Reshape" s and n' = numel "Move.Reshape" s' in
  if n <> n' then
    invalid_argf "Move.Reshape: %a has %d elements, %a has %d" pp_ints s n
      pp_ints s' n';
  Array.copy s'

let broadcast s s' =
  let r = Array.length s and r' = Array.length s' in
  check_rank "Move.Broadcast" r';
  ignore (numel "Move.Broadcast" s');
  if r' < r then
    invalid_argf "Move.Broadcast: %a has fewer axes than %a" pp_ints s' pp_ints
      s;
  for i = 0 to r - 1 do
    let d = s.(i) and d' = s'.(r' - r + i) in
    if d <> 1 && d <> d' then
      invalid_argf "Move.Broadcast: %a does not broadcast to %a" pp_ints s
        pp_ints s'
  done;
  Array.copy s'

(* The axes [p] names are marked in the bits of an int: a rank is at most
   [max_rank], below an int's width. *)
let permute s p =
  let r = Array.length s in
  if Array.length p <> r then
    invalid_argf "Move.Permute: %a has %d axes, not %d" pp_ints p
      (Array.length p) r;
  let seen = ref 0 in
  for i = 0 to r - 1 do
    let a = p.(i) in
    if a < 0 || a >= r || !seen land (1 lsl a) <> 0 then
      invalid_argf "Move.Permute: %a is not a permutation of %d axes" pp_ints p
        r;
    seen := !seen lor (1 lsl a)
  done;
  let s' = Array.make r 0 in
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
        invalid_argf "Move.Window: axes are not strictly increasing";
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

let shape m s =
  ignore (numel "Move.shape" s);
  match m with
  | Reshape s' -> reshape s s'
  | Broadcast s' -> broadcast s s'
  | Permute p -> permute s p
  | Slice rs -> slice s rs
  | Window ws -> window s ws
