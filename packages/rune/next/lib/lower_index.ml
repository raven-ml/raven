(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

let dtype = Ops.dtype
let ints l = List.map (fun n -> Ops.Int n) l
let pads padding = List.map (fun (b, a) -> Some (Ops.Int b, Ops.Int a)) padding

(* [along rank axis v] is the vector [v] as the axis [axis] of a node of [rank]
   axes. *)
let along rank axis v =
  Ops.reshape v
    (List.init rank (fun d ->
         if d = axis then List.hd (Ops.shape v) else Ops.Int 1))

(* [one_hot idx n] is whether each index of [idx] is each position of a new last
   axis of [n] elements. *)
let one_hot idx n = Ops.eq (Ops.unsqueeze idx (-1)) (Ops.arange n)

(* Bits

   An element selected among zeros is summed over the elements' bit patterns as
   unsigned integers: a float sum would turn [-0.] into [+0.], and quiet a
   signalling NaN. *)

let unsigned dt =
  List.find (fun u -> Dtype.itemsize u = Dtype.itemsize dt) Dtype.uints

let bits u =
  if Dtype.equal (dtype u) Bool then Ops.cast u Uint8
  else Ops.bitcast u (unsigned (dtype u))

let of_bits dt b =
  if Dtype.equal dt Bool then Ops.cast b Bool else Ops.bitcast b dt

(* [pick mask b] is the bits [b] where [mask] holds, summed over the last axis:
   the element [mask] selects, or zero where it selects none. *)
let pick mask b =
  let selected = Ops.where mask b (Ops.int 0) in
  Ops.rop selected Op.Add [ Ops.ndim selected - 1 ]

(* Assembly *)

let everywhere u = Ops.const_like ~dtype:Bool u (`Bool true)

let pad padding fill x =
  let padding = pads (Array.to_list padding) in
  match fill with
  | `Float f
    when Int64.equal (Int64.bits_of_float f) (Int64.bits_of_float (-0.)) ->
      (* A pad of zeros fills with [+0.]: [-0.] is selected on the padding. *)
      Ops.where
        (Ops.pad (everywhere x) padding)
        (Ops.pad x padding)
        (Ops.const ~dtype:(dtype x) fill)
  | fill -> Ops.pad ~value:fill x padding

(* Pieces of one length are stacked. Pieces of different lengths are each
   selected on the stretch they fill. *)
let cat axis x xs =
  let length u = List.nth (Ops.max_shape u) axis in
  match List.filter (fun u -> length u > 0) (x :: xs) with
  | [] -> x
  | y :: rest when List.for_all (fun u -> length u = length y) rest ->
      Ops.cat ~axis y rest
  | pieces ->
      let total = List.fold_left (fun n u -> n + length u) 0 pieces in
      let place (start, placed) u =
        let spread =
          pads
            (List.init (Ops.ndim u) (fun d ->
                 if d = axis then (start, total - start - length u) else (0, 0)))
        in
        let stretch = (Ops.pad (everywhere u) spread, Ops.pad u spread) in
        (start + length u, stretch :: placed)
      in
      let _, placed = List.fold_left place (0, []) pieces in
      List.fold_left
        (fun rest (stretch, u) -> Ops.where stretch u rest)
        (snd (List.hd placed))
        (List.tl placed)

(* Indexed access

   The indices meet the positions of [x] along [axis] as a one-hot mask along a
   new last axis, as the reference builds them. *)

let gather axis indices x =
  Lower_reduce.take
    (Ops.shrink_to x
       (List.mapi
          (fun d i -> if d = axis then None else Some i)
          (Ops.shape indices)))
    axis indices

let scatter ~mode ~unique ~axis ~indices ~updates x =
  let n = List.nth (Ops.max_shape x) axis and r = Ops.ndim x in
  (* Each position of [x] against each update along [axis], moved last. *)
  let within u = Ops.pad_to u (List.map Option.some (Ops.shape x) @ [ None ]) in
  let mask = within (Ops.transpose (one_hot indices n) axis r) in
  let src =
    within
      (Ops.transpose
         (Ops.expand
            (Ops.unsqueeze updates (-1))
            (Ops.shape updates @ [ Ops.Int n ]))
         axis r)
  in
  let reached = Ops.rop mask Op.Max [ r ] in
  match mode with
  | `Add ->
      let added = Ops.where mask src (Ops.const_like src (`Int Z.zero)) in
      Ops.where reached
        (Lower_reduce.reduce Sum ~axes:[ r ]
           (Ops.cat ~axis:r (Ops.unsqueeze x (-1)) [ added ]))
        x
  | `Set ->
      (* The last update that reaches each position is the one of highest index
         along [axis]; with [unique], it is the only one. *)
      let last =
        if unique then mask
        else
          let order =
            along (r + 1) r (Ops.arange (List.nth (Ops.max_shape mask) r))
          in
          let latest =
            Ops.rop (Ops.where mask order (Ops.int (-1))) Op.Max [ r ]
          in
          Ops.eq order (Ops.unsqueeze latest (-1))
      in
      Ops.where reached (of_bits (dtype x) (pick last (bits src))) x

(* The window is [v] moved along each axis it does not fill to its start there:
   the positions of [x] along that axis against those of [v] as a one-hot mask,
   as a gather builds it. Along an axis [v] fills, the start is 0. *)
let update x ~starts v =
  let n = Ops.max_shape x and k = Ops.max_shape v and r = Ops.ndim x in
  let moved =
    List.filter (fun d -> List.nth k d < List.nth n d) (List.init r Fun.id)
  in
  let start d =
    Ops.reshape (Ops.shrink starts [ Some (Ops.Int d, Ops.Int (d + 1)) ]) []
  in
  let shift b d =
    let at = along (r + 1) d (Ops.arange (List.nth n d)) in
    let offset = along (r + 1) r (Ops.arange (List.nth k d)) in
    pick
      (Ops.eq at (Ops.add offset (start d)))
      (Ops.transpose (Ops.unsqueeze b (-1)) d r)
  in
  let inside d =
    let at = along r d (Ops.arange (List.nth n d)) in
    let first = start d in
    Ops.bitwise_and (Ops.le first at)
      (Ops.lt at (Ops.add first (Ops.int (List.nth k d))))
  in
  match List.map inside moved with
  | [] -> v
  | window :: rest ->
      Ops.where
        (List.fold_left Ops.bitwise_and window rest)
        (of_bits (dtype x) (List.fold_left shift (bits v) moved))
        x

(* Windows *)

let product = List.fold_left ( * ) 1

(* Whether one of [windows] windows, [stride] apart from the start of a padded
   axis, each of [kernel] elements [dilation] apart, reads one of the [size]
   elements of the image that follow [before] elements of padding. *)
let reads_image ~kernel ~stride ~dilation ~windows ~before ~size =
  List.exists
    (fun w ->
      List.exists
        (fun j ->
          let p = (w * stride) + (j * dilation) in
          p >= before && p < before + size)
        (List.init kernel Fun.id))
    (List.init windows Fun.id)

let unfold ~kernel_size ~stride ~dilation ~padding x =
  let k = Array.length kernel_size in
  let lead = Ops.ndim x - k in
  let kept = List.filteri (fun d _ -> d < lead) (Ops.max_shape x) in
  let spatial = List.filteri (fun d _ -> d >= lead) (Ops.max_shape x) in
  let count a size =
    let before, after = padding.(a) in
    let reach = (dilation.(a) * (kernel_size.(a) - 1)) + 1 in
    let padded = size + before + after in
    if padded < reach then 0 else ((padded - reach) / stride.(a)) + 1
  in
  let counts = List.mapi count spatial in
  let reads a size =
    reads_image ~kernel:kernel_size.(a) ~stride:stride.(a)
      ~dilation:dilation.(a) ~windows:(List.nth counts a)
      ~before:(fst padding.(a))
      ~size
  in
  let shape =
    ints (kept @ [ product (Array.to_list kernel_size); product counts ])
  in
  (* Along an axis whose windows read only padding, or that has no window, every
     patch is the pad's zeros. *)
  if not (List.for_all Fun.id (List.mapi reads spatial)) then
    Ops.expand (Ops.const ~dtype:(Ops.dtype x) (`Int Z.zero)) shape
  else
    let padded =
      Ops.pad x (List.init lead (fun _ -> None) @ pads (Array.to_list padding))
    in
    let windows =
      Ops.pool ~stride:(Array.to_list stride) ~dilation:(Array.to_list dilation)
        padded
        (Array.to_list kernel_size)
    in
    Ops.reshape
      (Ops.permute windows
         (List.init lead Fun.id
         @ List.init k (fun a -> lead + k + a)
         @ List.init k (fun a -> lead + a)))
      shape

(* An axis as [Ops.pool] cuts it: [windows] windows of [kernel] elements from
   [size] padded elements, read from [copies] copies of the axis laid end to end
   in rows of [row] elements, one row per element of the kernel. *)
type cut = {
  kernel : int;
  stride : int;
  windows : int;
  size : int;
  row : int;
  copies : int;
}

let cut ~kernel ~stride ~dilation ~size =
  let ceil_div a b = (a + b - 1) / b in
  (* A kernel that spans more than the padded axis has no window. *)
  let windows =
    Int.max 0 (ceil_div (size - (dilation * (kernel - 1))) stride)
  in
  let scale =
    (ceil_div
       ((windows * stride) - dilation)
       size [@mutate off "more copies cut the same windows"])
  in
  let row = (size * Int.max 1 scale) + dilation in
  { kernel; stride; windows; size; row; copies = ceil_div (kernel * row) size }

(* A fold is the transpose of the unfold: each movement of [Ops.pool] is undone
   in reverse order, a shrink by a pad of zeros, and the copies of the input are
   summed, which sums the windows where they overlap. *)
let fold ~output_size ~kernel_size ~stride ~dilation ~padding x =
  let k = Array.length kernel_size in
  let lead = Ops.ndim x - 2 in
  let kept = List.filteri (fun d _ -> d < lead) (Ops.max_shape x) in
  let cuts =
    List.init k (fun a ->
        let before, after = padding.(a) in
        cut ~kernel:kernel_size.(a) ~stride:stride.(a) ~dilation:dilation.(a)
          ~size:(output_size.(a) + before + after))
  in
  (* Along an axis without a window, or whose every window reads only padding,
     no element lands in the output, which is zeros. *)
  let lands a c =
    reads_image ~kernel:c.kernel ~stride:c.stride ~dilation:dilation.(a)
      ~windows:c.windows
      ~before:(fst padding.(a))
      ~size:output_size.(a)
  in
  if not (List.for_all Fun.id (List.mapi lands cuts)) then
    Ops.expand
      (Ops.const ~dtype:(Ops.dtype x) (`Int Z.zero))
      (ints (kept @ Array.to_list output_size))
  else
    let shape f = ints (kept @ List.concat_map f cuts) in
    let reshape u f = Ops.reshape u (shape f) in
    let pad_to u f = Ops.pad_to u (List.map Option.some (shape f)) in
    let x =
      Ops.reshape x
        (ints
           (kept @ Array.to_list kernel_size
           @ List.map (fun c -> c.windows) cuts))
    in
    let x =
      Ops.permute x
        (List.init lead Fun.id
        @ List.concat (List.init k (fun a -> [ lead + a; lead + k + a ])))
    in
    let x = reshape x (fun c -> [ c.kernel; c.windows; 1 ]) in
    let x = pad_to x (fun c -> [ c.kernel; c.windows; c.stride ]) in
    let x = reshape x (fun c -> [ c.kernel; c.windows * c.stride ]) in
    let x = pad_to x (fun c -> [ c.kernel; c.row ]) in
    let x = reshape x (fun c -> [ c.kernel * c.row ]) in
    let x = pad_to x (fun c -> [ c.copies * c.size ]) in
    let x = reshape x (fun c -> [ c.copies; c.size ]) in
    Ops.shrink
      (Lower_reduce.reduce Sum ~axes:(List.init k (fun a -> lead + (2 * a))) x)
      (List.map (fun _ -> None) kept
      @ List.init k (fun a ->
          let before = fst padding.(a) in
          Some (Ops.Int before, Ops.Int (before + output_size.(a)))))
