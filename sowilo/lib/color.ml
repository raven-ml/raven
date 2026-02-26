(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let to_grayscale img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  (* Flatten spatial dims, matmul with [3;1] weights, reshape back *)
  let spatial = Array.sub shape 0 (rank - 1) in
  let n_pixels = Array.fold_left ( * ) 1 spatial in
  let flat = Rune.reshape [| n_pixels; shape.(c_axis) |] img in
  let weights = Rune.create Rune.float32 [| 3; 1 |] [| 0.299; 0.587; 0.114 |] in
  let result = Rune.matmul flat weights in
  let out_shape = Array.copy shape in
  out_shape.(c_axis) <- 1;
  Rune.reshape out_shape result

(* RGB to HSV conversion using piecewise hue computation *)

let rgb_to_hsv img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let slice_channel i =
    let slices =
      List.init rank (fun ax ->
          if ax = c_axis then Rune.R (i, i + 1) else Rune.A)
    in
    Rune.slice slices img
  in
  let r = slice_channel 0 in
  let g = slice_channel 1 in
  let b = slice_channel 2 in
  let cmax = Rune.maximum (Rune.maximum r g) b in
  let cmin = Rune.minimum (Rune.minimum r g) b in
  let delta = Rune.sub cmax cmin in
  let eps = Rune.scalar_like img 1e-7 in
  let delta_safe = Rune.add delta eps in
  (* Hue computation: piecewise by which channel is max *)
  let is_r_max = Rune.equal cmax r in
  let is_g_max =
    Rune.logical_and (Rune.equal cmax g) (Rune.logical_not is_r_max)
  in
  let h_r = Rune.div (Rune.sub g b) delta_safe in
  let h_g = Rune.add_s (Rune.div (Rune.sub b r) delta_safe) 2.0 in
  let h_b = Rune.add_s (Rune.div (Rune.sub r g) delta_safe) 4.0 in
  let h = Rune.where is_r_max h_r (Rune.where is_g_max h_g h_b) in
  (* Normalize to [0, 1]: divide by 6, wrap negatives *)
  let h = Rune.div_s h 6.0 in
  let h = Rune.where (Rune.less h (Rune.zeros_like h)) (Rune.add_s h 1.0) h in
  (* Saturation *)
  let s =
    Rune.where (Rune.greater cmax eps)
      (Rune.div delta (Rune.add cmax eps))
      (Rune.zeros_like cmax)
  in
  (* Value *)
  let v = cmax in
  Rune.concatenate ~axis:c_axis [ h; s; v ]

(* HSV to RGB conversion *)

let hsv_to_rgb img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let slice_channel i =
    let slices =
      List.init rank (fun ax ->
          if ax = c_axis then Rune.R (i, i + 1) else Rune.A)
    in
    Rune.slice slices img
  in
  let h = slice_channel 0 in
  let s = slice_channel 1 in
  let v = slice_channel 2 in
  (* h is in [0, 1], scale to [0, 6) *)
  let h6 = Rune.mul_s h 6.0 in
  let hi = Rune.floor h6 in
  let f = Rune.sub h6 hi in
  let one = Rune.ones_like v in
  let p = Rune.mul v (Rune.sub one s) in
  let q = Rune.mul v (Rune.sub one (Rune.mul s f)) in
  let t_ = Rune.mul v (Rune.sub one (Rune.mul s (Rune.sub one f))) in
  (* Select r, g, b based on hi mod 6 *)
  let hi_mod = Rune.mod_ h6 (Rune.scalar_like h6 6.0) in
  let hi_floor = Rune.floor hi_mod in
  let is_sect n =
    let n_t = Rune.scalar_like hi_floor (Float.of_int n) in
    Rune.logical_and
      (Rune.greater_equal hi_floor n_t)
      (Rune.less hi_floor (Rune.scalar_like hi_floor (Float.of_int (n + 1))))
  in
  let s0 = is_sect 0 in
  let s1 = is_sect 1 in
  let s2 = is_sect 2 in
  let s3 = is_sect 3 in
  let s4 = is_sect 4 in
  (* s5 is the remainder *)
  let r =
    Rune.where s0 v
      (Rune.where s1 q (Rune.where s2 p (Rune.where s3 p (Rune.where s4 t_ v))))
  in
  let g =
    Rune.where s0 t_
      (Rune.where s1 v (Rune.where s2 v (Rune.where s3 q (Rune.where s4 p p))))
  in
  let b =
    Rune.where s0 p
      (Rune.where s1 p (Rune.where s2 t_ (Rune.where s3 v (Rune.where s4 v q))))
  in
  Rune.concatenate ~axis:c_axis [ r; g; b ]

let adjust_brightness factor img =
  Rune.clip ~min:0.0 ~max:1.0 (Rune.mul_s img factor)

let adjust_contrast factor img =
  let shape = Rune.shape img in
  let rank = Array.length shape in
  (* Mean per channel, keep spatial dims *)
  let axes = List.init (rank - 1) Fun.id in
  let mean = Rune.mean ~axes ~keepdims:true img in
  let shifted = Rune.sub img mean in
  Rune.clip ~min:0.0 ~max:1.0 (Rune.add mean (Rune.mul_s shifted factor))

let adjust_saturation factor img =
  let hsv = rgb_to_hsv img in
  let shape = Rune.shape hsv in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let h =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (0, 1) else Rune.A))
      hsv
  in
  let s =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (1, 2) else Rune.A))
      hsv
  in
  let v =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (2, 3) else Rune.A))
      hsv
  in
  let s' = Rune.clip ~min:0.0 ~max:1.0 (Rune.mul_s s factor) in
  hsv_to_rgb (Rune.concatenate ~axis:c_axis [ h; s'; v ])

let adjust_hue delta img =
  let hsv = rgb_to_hsv img in
  let shape = Rune.shape hsv in
  let rank = Array.length shape in
  let c_axis = rank - 1 in
  let h =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (0, 1) else Rune.A))
      hsv
  in
  let s =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (1, 2) else Rune.A))
      hsv
  in
  let v =
    Rune.slice
      (List.init rank (fun ax -> if ax = c_axis then Rune.R (2, 3) else Rune.A))
      hsv
  in
  (* Wrap hue to [0, 1] *)
  let h' = Rune.add_s h delta in
  let h' = Rune.sub h' (Rune.floor h') in
  hsv_to_rgb (Rune.concatenate ~axis:c_axis [ h'; s; v ])

let adjust_gamma gamma img = Rune.pow_s img gamma
let invert img = Rune.sub (Rune.ones_like img) img
