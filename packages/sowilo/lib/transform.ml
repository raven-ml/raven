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

let float_range size = Nx.arange_f Nx.float32 0.0 (float size) 1.0

let compute_nearest_indices ~size_in ~size_out =
  if size_out = 1 || size_in = 1 then Nx.full Nx.int32 [| size_out |] Int32.zero
  else
    let scale = float size_in /. float size_out in
    let coords = float_range size_out in
    let src = Nx.sub_s (Nx.mul_s (Nx.add_s coords 0.5) scale) 0.5 in
    let src_clipped = Nx.clip ~min:0.0 ~max:(float (size_in - 1)) src in
    Nx.astype Nx.int32 (Nx.round src_clipped)

let compute_linear_axis ~size_in ~size_out =
  if size_out = 1 || size_in = 1 then
    let zeros_i = Nx.full Nx.int32 [| size_out |] Int32.zero in
    let zeros_f = Nx.full Nx.float32 [| size_out |] 0.0 in
    (zeros_i, zeros_i, zeros_f)
  else
    let scale = float (size_in - 1) /. float (size_out - 1) in
    let src = Nx.mul_s (float_range size_out) scale in
    let idx0 = src |> Nx.floor |> Nx.astype Nx.int32 in
    let one = Nx.scalar_like idx0 Int32.(of_int 1) in
    let max_idx = Nx.scalar_like idx0 Int32.(of_int (size_in - 1)) in
    let idx1 = Nx.minimum (Nx.add idx0 one) max_idx in
    let delta = Nx.sub src (Nx.astype Nx.float32 idx0) in
    (idx0, idx1, delta)

let resize : type a b.
    ?interpolation:interpolation ->
    height:int ->
    width:int ->
    (a, b) Nx.t ->
    (a, b) Nx.t =
 fun ?(interpolation = Bilinear) ~height:out_h ~width:out_w img ->
  if out_h <= 0 || out_w <= 0 then
    invalid_arg "resize: height and width must be positive";
  let shape = Nx.shape img in
  let rank = Array.length shape in
  let h_ax, w_ax = hw_axes rank in
  let in_h = shape.(h_ax) and in_w = shape.(w_ax) in
  match interpolation with
  | Nearest ->
      let y_idx = compute_nearest_indices ~size_in:in_h ~size_out:out_h in
      let x_idx = compute_nearest_indices ~size_in:in_w ~size_out:out_w in
      img |> Nx.take ~axis:h_ax y_idx |> Nx.take ~axis:w_ax x_idx
  | Bilinear ->
      let img_f = Nx.astype Nx.float32 img in
      let y0, y1, dy = compute_linear_axis ~size_in:in_h ~size_out:out_h in
      let x0, x1, dx = compute_linear_axis ~size_in:in_w ~size_out:out_w in
      let top = Nx.take ~axis:h_ax y0 img_f in
      let bottom = Nx.take ~axis:h_ax y1 img_f in
      let top_left = Nx.take ~axis:w_ax x0 top in
      let top_right = Nx.take ~axis:w_ax x1 top in
      let bottom_left = Nx.take ~axis:w_ax x0 bottom in
      let bottom_right = Nx.take ~axis:w_ax x1 bottom in
      (* Reshape dx and dy for broadcasting *)
      let make_broadcastable ax size =
        let s = Array.make rank 1 in
        s.(ax) <- size;
        s
      in
      let dx_b = Nx.reshape (make_broadcastable w_ax out_w) dx in
      let dy_b = Nx.reshape (make_broadcastable h_ax out_h) dy in
      let one_dx = Nx.sub (Nx.ones_like dx_b) dx_b in
      let one_dy = Nx.sub (Nx.ones_like dy_b) dy_b in
      let top_interp =
        Nx.add (Nx.mul one_dx top_left) (Nx.mul dx_b top_right)
      in
      let bottom_interp =
        Nx.add (Nx.mul one_dx bottom_left) (Nx.mul dx_b bottom_right)
      in
      let blended =
        Nx.add (Nx.mul one_dy top_interp) (Nx.mul dy_b bottom_interp)
      in
      Nx.astype (Nx.dtype img) blended

let crop ~y ~x ~height ~width img =
  let shape = Nx.shape img in
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
        if ax = h_ax then Nx.R (y, y + height)
        else if ax = w_ax then Nx.R (x, x + width)
        else Nx.A)
  in
  Nx.slice slices img

let center_crop ~height ~width img =
  let shape = Nx.shape img in
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
  let rank = Array.length (Nx.shape img) in
  let _, w_ax = hw_axes rank in
  Nx.flip ~axes:[ w_ax ] img

let vflip img =
  let rank = Array.length (Nx.shape img) in
  let h_ax, _ = hw_axes rank in
  Nx.flip ~axes:[ h_ax ] img

let rotate90 ?(k = 1) img =
  let k = ((k mod 4) + 4) mod 4 in
  if k = 0 then img
  else
    let rank = Array.length (Nx.shape img) in
    let h_ax, w_ax = hw_axes rank in
    let rotate_once t =
      (* CCW 90: transpose H,W then flip W *)
      let axes = Array.init rank Fun.id in
      axes.(h_ax) <- w_ax;
      axes.(w_ax) <- h_ax;
      let transposed = Nx.transpose ~axes:(Array.to_list axes) t in
      Nx.flip ~axes:[ h_ax ] transposed
    in
    let result = ref img in
    for _ = 1 to k do
      result := rotate_once !result
    done;
    !result

let pad : type a b.
    ?value:float -> int * int * int * int -> (a, b) Nx.t -> (a, b) Nx.t =
 fun ?(value = 0.0) (top, bottom, left, right) img ->
  let rank = Array.length (Nx.shape img) in
  let h_ax, w_ax = hw_axes rank in
  let padding = Array.make rank (0, 0) in
  padding.(h_ax) <- (top, bottom);
  padding.(w_ax) <- (left, right);
  let fill : a = Nx_core.Dtype.of_float (Nx.dtype img) value in
  Nx.pad padding fill img
