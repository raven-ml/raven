(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let pp_ints a =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_int a)) ^ "|]"

(* [x] padded with [fill], cut into windows: [[n; t…; c; window…]], the
   window axes last, from views of one padded copy. *)
let windows ~by ~window ?stride ?padding fill x =
  let k = Array.length window in
  if k = 0 then invalid_argf "Pool.%s: window has no extent" by;
  let entries name default = function
    | None -> default
    | Some a when Array.length a <> k ->
        invalid_argf "Pool.%s: %d %s for %d window extents" by (Array.length a)
          name k
    | Some a -> a
  in
  let stride = entries "strides" window stride in
  let padding = entries "paddings" (Array.make k (0, 0)) padding in
  if Array.exists (fun v -> v < 1) window then
    invalid_argf "Pool.%s: a window extent below 1 in %s" by (pp_ints window);
  if Array.exists (fun v -> v < 1) stride then
    invalid_argf "Pool.%s: a stride below 1 in %s" by (pp_ints stride);
  if Array.exists (fun (lo, hi) -> lo < 0 || hi < 0) padding then
    invalid_argf "Pool.%s: a negative padding" by;
  let s = Nx.shape x in
  if Array.length s <> k + 2 then
    invalid_argf "Pool.%s: %d window extents take an input of %d axes, got %d"
      by k (k + 2) (Array.length s);
  for i = 0 to k - 1 do
    let lo, hi = padding.(i) in
    if s.(1 + i) + lo + hi < window.(i) then
      invalid_argf "Pool.%s: no window of %s fits the input's spatial axes %s"
        by (pp_ints window)
        (pp_ints (Array.sub s 1 k))
  done;
  let x =
    if Array.for_all (( = ) (0, 0)) padding then x
    else Nx.pad (Array.concat [ [| (0, 0) |]; padding; [| (0, 0) |] ]) fill x
  in
  let v = ref x in
  for i = 0 to k - 1 do
    v := Nx.sliding_window ~axis:(1 + i) ~window:window.(i) ~step:stride.(i) !v
  done;
  (!v, List.init k (fun i -> k + 2 + i))

let max ~window ?stride ?padding x =
  let fill = Nx.Dtype.min_value (Nx.dtype x) in
  let v, axes = windows ~by:"max" ~window ?stride ?padding fill x in
  Nx.max ~axes v

let avg ~window ?stride ?padding x =
  let v, axes = windows ~by:"avg" ~window ?stride ?padding 0. x in
  Nx.mean ~axes v
