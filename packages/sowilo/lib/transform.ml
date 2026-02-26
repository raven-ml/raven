(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type interpolation = Nearest | Bilinear

(* Spatial axes for H and W given tensor rank *)

let hw_axes rank =
  match rank with
  | 3 -> (0, 1) (* [H; W; C] *)
  | 4 -> (1, 2) (* [N; H; W; C] *)
  | n -> invalid_arg (Helpers.err_rank n)

let float_range size = Rune.arange_f Rune.float32 0.0 (float size) 1.0

let compute_nearest_indices ~size_in ~size_out =
  if size_out = 1 || size_in = 1 then
    Rune.full Rune.int32 [| size_out |] Int32.zero
  else
    let scale = float size_in /. float size_out in
    let coords = float_range size_out in
    let src = Rune.sub_s (Rune.mul_s (Rune.add_s coords 0.5) scale) 0.5 in
    let src_clipped = Rune.clip ~min:0.0 ~max:(float (size_in - 1)) src in
    Rune.astype Rune.int32 (Rune.round src_clipped)

let compute_linear_axis ~size_in ~size_out =
  if size_out = 1 || size_in = 1 then
    let zeros_i = Rune.full Rune.int32 [| size_out |] Int32.zero in
    let zeros_f = Rune.full Rune.float32 [| size_out |] 0.0 in
    (zeros_i, zeros_i, zeros_f)
  else
    let scale = float (size_in - 1) /. float (size_out - 1) in
    let src = Rune.mul_s (float_range size_out) scale in
    let idx0 = src |> Rune.floor |> Rune.astype Rune.int32 in
    let one = Rune.scalar_like idx0 Int32.(of_int 1) in
    let max_idx = Rune.scalar_like idx0 Int32.(of_int (size_in - 1)) in
    let idx1 = Rune.minimum (Rune.add idx0 one) max_idx in
    let delta = Rune.sub src (Rune.astype Rune.float32 idx0) in
    (idx0, idx1, delta)

let resize : type a b.
    ?interpolation:interpolation ->
    height:int ->
    width:int ->
    (a, b) Rune.t ->
    (a, b) Rune.t =
 fun ?(interpolation = Bilinear) ~height:out_h ~width:out_w img ->
  if out_h <= 0 || out_w <= 0 then
    invalid_arg "resize: height and width must be positive";
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let h_ax, w_ax = hw_axes rank in
  let in_h = shape.(h_ax) and in_w = shape.(w_ax) in
  match interpolation with
  | Nearest ->
      let y_idx = compute_nearest_indices ~size_in:in_h ~size_out:out_h in
      let x_idx = compute_nearest_indices ~size_in:in_w ~size_out:out_w in
      img |> Rune.take ~axis:h_ax y_idx |> Rune.take ~axis:w_ax x_idx
  | Bilinear ->
      let img_f = Rune.astype Rune.float32 img in
      let y0, y1, dy = compute_linear_axis ~size_in:in_h ~size_out:out_h in
      let x0, x1, dx = compute_linear_axis ~size_in:in_w ~size_out:out_w in
      let top = Rune.take ~axis:h_ax y0 img_f in
      let bottom = Rune.take ~axis:h_ax y1 img_f in
      let top_left = Rune.take ~axis:w_ax x0 top in
      let top_right = Rune.take ~axis:w_ax x1 top in
      let bottom_left = Rune.take ~axis:w_ax x0 bottom in
      let bottom_right = Rune.take ~axis:w_ax x1 bottom in
      (* Reshape dx and dy for broadcasting *)
      let make_broadcastable ax size =
        let s = Array.make rank 1 in
        s.(ax) <- size;
        s
      in
      let dx_b = Rune.reshape (make_broadcastable w_ax out_w) dx in
      let dy_b = Rune.reshape (make_broadcastable h_ax out_h) dy in
      let one_dx = Rune.sub (Rune.ones_like dx_b) dx_b in
      let one_dy = Rune.sub (Rune.ones_like dy_b) dy_b in
      let top_interp =
        Rune.add (Rune.mul one_dx top_left) (Rune.mul dx_b top_right)
      in
      let bottom_interp =
        Rune.add (Rune.mul one_dx bottom_left) (Rune.mul dx_b bottom_right)
      in
      let blended =
        Rune.add (Rune.mul one_dy top_interp) (Rune.mul dy_b bottom_interp)
      in
      Rune.astype (Rune.dtype img) blended

let crop ~y ~x ~height ~width img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let h_ax, w_ax = hw_axes rank in
  let in_h = shape.(h_ax) and in_w = shape.(w_ax) in
  if
    y < 0 || x < 0 || height <= 0 || width <= 0
    || y + height > in_h
    || x + width > in_w
  then
    invalid_arg
      (Printf.sprintf "crop: region y=%d x=%d h=%d w=%d exceeds image %dx%d" y x
         height width in_h in_w);
  let slices =
    List.init rank (fun ax ->
        if ax = h_ax then Rune.R (y, y + height)
        else if ax = w_ax then Rune.R (x, x + width)
        else Rune.A)
  in
  Rune.slice slices img

let center_crop ~height ~width img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let h_ax, w_ax = hw_axes rank in
  let in_h = shape.(h_ax) and in_w = shape.(w_ax) in
  if height > in_h || width > in_w then
    invalid_arg
      (Printf.sprintf "center_crop: target %dx%d exceeds image %dx%d" height
         width in_h in_w);
  let y = (in_h - height) / 2 in
  let x = (in_w - width) / 2 in
  crop ~y ~x ~height ~width img

let hflip img =
  let rank = Array.length (Rune.shape img) in
  let _, w_ax = hw_axes rank in
  Rune.flip ~axes:[ w_ax ] img

let vflip img =
  let rank = Array.length (Rune.shape img) in
  let h_ax, _ = hw_axes rank in
  Rune.flip ~axes:[ h_ax ] img

let rotate90 ?(k = 1) img =
  let k = ((k mod 4) + 4) mod 4 in
  if k = 0 then img
  else
    let rank = Array.length (Rune.shape img) in
    let h_ax, w_ax = hw_axes rank in
    let rotate_once t =
      (* CCW 90: transpose H,W then flip W *)
      let axes = Array.init rank Fun.id in
      axes.(h_ax) <- w_ax;
      axes.(w_ax) <- h_ax;
      let transposed = Rune.transpose ~axes:(Array.to_list axes) t in
      Rune.flip ~axes:[ h_ax ] transposed
    in
    let result = ref img in
    for _ = 1 to k do
      result := rotate_once !result
    done;
    !result

let pad : type a b.
    ?value:float -> int * int * int * int -> (a, b) Rune.t -> (a, b) Rune.t =
 fun ?(value = 0.0) (top, bottom, left, right) img ->
  let rank = Array.length (Rune.shape img) in
  let h_ax, w_ax = hw_axes rank in
  let padding = Array.make rank (0, 0) in
  padding.(h_ax) <- (top, bottom);
  padding.(w_ax) <- (left, right);
  let fill : a = Nx_core.Dtype.of_float (Rune.dtype img) value in
  Rune.pad padding fill img
