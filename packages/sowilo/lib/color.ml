(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let to_grayscale img =
  let shape = Nx.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  (* Flatten spatial dims, matmul with [3;1] weights, reshape back *)
  let spatial = Array.sub shape 0 (rank - 1) in
  let n_pixels = Array.fold_left ( * ) 1 spatial in
  let flat = Nx.reshape [| n_pixels; shape.(c_axis) |] img in
  let weights = Nx.create Nx.float32 [| 3; 1 |] [| 0.299; 0.587; 0.114 |] in
  let result = Nx.matmul flat weights in
  let out_shape = Array.copy shape in
  out_shape.(c_axis) <- 1;
  Nx.reshape out_shape result

(* RGB to HSV conversion using piecewise hue computation *)

let rgb_to_hsv img =
  let shape = Nx.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let slice_channel i =
    let slices =
      List.init rank (fun ax -> if ax = c_axis then Nx.R (i, i + 1) else Nx.A)
    in
    Nx.slice slices img
  in
  let r = slice_channel 0 in
  let g = slice_channel 1 in
  let b = slice_channel 2 in
  let cmax = Nx.maximum (Nx.maximum r g) b in
  let cmin = Nx.minimum (Nx.minimum r g) b in
  let delta = Nx.sub cmax cmin in
  let eps = Nx.scalar_like img 1e-7 in
  let delta_safe = Nx.add delta eps in
  (* Hue computation: piecewise by which channel is max *)
  let is_r_max = Nx.equal cmax r in
  let is_g_max = Nx.logical_and (Nx.equal cmax g) (Nx.logical_not is_r_max) in
  let h_r = Nx.div (Nx.sub g b) delta_safe in
  let h_g = Nx.add_s (Nx.div (Nx.sub b r) delta_safe) 2.0 in
  let h_b = Nx.add_s (Nx.div (Nx.sub r g) delta_safe) 4.0 in
  let h = Nx.where is_r_max h_r (Nx.where is_g_max h_g h_b) in
  (* Normalize to [0, 1]: divide by 6, wrap negatives *)
  let h = Nx.div_s h 6.0 in
  let h = Nx.where (Nx.less h (Nx.zeros_like h)) (Nx.add_s h 1.0) h in
  (* Saturation *)
  let s =
    Nx.where (Nx.greater cmax eps)
      (Nx.div delta (Nx.add cmax eps))
      (Nx.zeros_like cmax)
  in
  (* Value *)
  let v = cmax in
  Nx.concatenate ~axis:c_axis [ h; s; v ]

(* HSV to RGB conversion *)

let hsv_to_rgb img =
  let shape = Nx.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let slice_channel i =
    let slices =
      List.init rank (fun ax -> if ax = c_axis then Nx.R (i, i + 1) else Nx.A)
    in
    Nx.slice slices img
  in
  let h = slice_channel 0 in
  let s = slice_channel 1 in
  let v = slice_channel 2 in
  (* h is in [0, 1], scale to [0, 6) *)
  let h6 = Nx.mul_s h 6.0 in
  let hi = Nx.floor h6 in
  let f = Nx.sub h6 hi in
  let one = Nx.ones_like v in
  let p = Nx.mul v (Nx.sub one s) in
  let q = Nx.mul v (Nx.sub one (Nx.mul s f)) in
  let t_ = Nx.mul v (Nx.sub one (Nx.mul s (Nx.sub one f))) in
  (* Select r, g, b based on hi mod 6 *)
  let hi_mod = Nx.mod_ h6 (Nx.scalar_like h6 6.0) in
  let hi_floor = Nx.floor hi_mod in
  let is_sect n =
    let n_t = Nx.scalar_like hi_floor (Float.of_int n) in
    Nx.logical_and
      (Nx.greater_equal hi_floor n_t)
      (Nx.less hi_floor (Nx.scalar_like hi_floor (Float.of_int (n + 1))))
  in
  let s0 = is_sect 0 in
  let s1 = is_sect 1 in
  let s2 = is_sect 2 in
  let s3 = is_sect 3 in
  let s4 = is_sect 4 in
  (* s5 is the remainder *)
  let r =
    Nx.where s0 v
      (Nx.where s1 q (Nx.where s2 p (Nx.where s3 p (Nx.where s4 t_ v))))
  in
  let g =
    Nx.where s0 t_
      (Nx.where s1 v (Nx.where s2 v (Nx.where s3 q (Nx.where s4 p p))))
  in
  let b =
    Nx.where s0 p
      (Nx.where s1 p (Nx.where s2 t_ (Nx.where s3 v (Nx.where s4 v q))))
  in
  Nx.concatenate ~axis:c_axis [ r; g; b ]

let adjust_brightness factor img =
  Nx.clip ~min:0.0 ~max:1.0 (Nx.mul_s img factor)

let adjust_contrast factor img =
  let shape = Nx.shape img in
  let rank = Array.length shape in
  (* Mean per channel, keep spatial dims *)
  let axes = List.init (rank - 1) Fun.id in
  let mean = Nx.mean ~axes ~keepdims:true img in
  let shifted = Nx.sub img mean in
  Nx.clip ~min:0.0 ~max:1.0 (Nx.add mean (Nx.mul_s shifted factor))

let adjust_saturation factor img =
  let hsv = rgb_to_hsv img in
  let shape = Nx.shape hsv in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let h =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (0, 1) else Nx.A))
      hsv
  in
  let s =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (1, 2) else Nx.A))
      hsv
  in
  let v =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (2, 3) else Nx.A))
      hsv
  in
  let s' = Nx.clip ~min:0.0 ~max:1.0 (Nx.mul_s s factor) in
  hsv_to_rgb (Nx.concatenate ~axis:c_axis [ h; s'; v ])

let adjust_hue delta img =
  let hsv = rgb_to_hsv img in
  let shape = Nx.shape hsv in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let h =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (0, 1) else Nx.A))
      hsv
  in
  let s =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (1, 2) else Nx.A))
      hsv
  in
  let v =
    Nx.slice
      (List.init rank (fun ax -> if ax = c_axis then Nx.R (2, 3) else Nx.A))
      hsv
  in
  (* Wrap hue to [0, 1] *)
  let h' = Nx.add_s h delta in
  let h' = Nx.sub h' (Nx.floor h') in
  hsv_to_rgb (Nx.concatenate ~axis:c_axis [ h'; s; v ])

let adjust_gamma gamma img = Nx.pow_s img gamma
let invert img = Nx.sub (Nx.ones_like img) img
