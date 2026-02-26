(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compute both gradients in a single pass via correlate2d *)

let gradient_pair ~kernel_x ~kernel_y img =
  Helpers.with_batch_pair
    (fun img ->
      let gx = Helpers.convolve_per_channel kernel_x img in
      let gy = Helpers.convolve_per_channel kernel_y img in
      (gx, gy))
    img

let sobel_kx =
  Rune.create Rune.float32 [| 3; 3 |]
    [| -1.; 0.; 1.; -2.; 0.; 2.; -1.; 0.; 1. |]

let sobel_ky =
  Rune.create Rune.float32 [| 3; 3 |]
    [| -1.; -2.; -1.; 0.; 0.; 0.; 1.; 2.; 1. |]

let scharr_kx =
  Rune.create Rune.float32 [| 3; 3 |]
    [| -3.; 0.; 3.; -10.; 0.; 10.; -3.; 0.; 3. |]

let scharr_ky =
  Rune.create Rune.float32 [| 3; 3 |]
    [| -3.; -10.; -3.; 0.; 0.; 0.; 3.; 10.; 3. |]

let sobel ?(ksize = 3) img =
  ignore ksize;
  gradient_pair ~kernel_x:sobel_kx ~kernel_y:sobel_ky img

let scharr img =
  gradient_pair ~kernel_x:scharr_kx ~kernel_y:scharr_ky img

let laplacian ?(ksize = 3) img =
  ignore ksize;
  (* Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0] *)
  let kernel =
    Rune.create Rune.float32 [| 3; 3 |]
      [| 0.0; 1.0; 0.0; 1.0; -4.0; 1.0; 0.0; 1.0; 0.0 |]
  in
  Helpers.with_batch (fun img -> Helpers.convolve_per_channel kernel img) img

let canny ~low ~high ?(sigma = 1.4) img =
  Helpers.with_batch
    (fun img ->
      (* 1. Gaussian blur *)
      let blurred = Filter.gaussian_blur ~sigma img in
      (* 2. Gradient computation *)
      let gx, gy =
        gradient_pair ~kernel_x:sobel_kx ~kernel_y:sobel_ky blurred
      in
      let mag = Rune.sqrt (Rune.add (Rune.square gx) (Rune.square gy)) in
      let angle = Rune.atan2 gy gx in
      let shape = Rune.shape mag in
      let n = shape.(0) and h = shape.(1) and w = shape.(2) in
      let mag3 = Rune.reshape [| n; h; w |] mag in
      let angle3 = Rune.reshape [| n; h; w |] angle in
      (* 3. Non-maximum suppression *)
      let angle_deg = Rune.mul_s angle3 (180.0 /. Float.pi) in
      let angle_pos =
        Rune.where
          (Rune.less angle_deg (Rune.zeros_like angle_deg))
          (Rune.add_s angle_deg 180.0)
          angle_deg
      in
      let scalar v = Rune.scalar_like angle_pos v in
      let is_horizontal =
        Rune.logical_or
          (Rune.logical_and
             (Rune.greater_equal angle_pos (scalar 0.0))
             (Rune.less angle_pos (scalar 22.5)))
          (Rune.logical_and
             (Rune.greater_equal angle_pos (scalar 157.5))
             (Rune.less_equal angle_pos (scalar 180.0)))
      in
      let is_diag1 =
        Rune.logical_and
          (Rune.greater_equal angle_pos (scalar 22.5))
          (Rune.less angle_pos (scalar 67.5))
      in
      let is_vertical =
        Rune.logical_and
          (Rune.greater_equal angle_pos (scalar 67.5))
          (Rune.less angle_pos (scalar 112.5))
      in
      let is_diag2 =
        Rune.logical_and
          (Rune.greater_equal angle_pos (scalar 112.5))
          (Rune.less angle_pos (scalar 157.5))
      in
      let mag_padded = Rune.pad [| (0, 0); (1, 1); (1, 1) |] 0.0 mag3 in
      let center =
        Rune.slice [ Rune.A; Rune.R (1, h + 1); Rune.R (1, w + 1) ] mag_padded
      in
      let left =
        Rune.slice [ Rune.A; Rune.R (1, h + 1); Rune.R (0, w) ] mag_padded
      in
      let right =
        Rune.slice [ Rune.A; Rune.R (1, h + 1); Rune.R (2, w + 2) ] mag_padded
      in
      let top =
        Rune.slice [ Rune.A; Rune.R (0, h); Rune.R (1, w + 1) ] mag_padded
      in
      let bottom =
        Rune.slice [ Rune.A; Rune.R (2, h + 2); Rune.R (1, w + 1) ] mag_padded
      in
      let tr =
        Rune.slice [ Rune.A; Rune.R (0, h); Rune.R (2, w + 2) ] mag_padded
      in
      let bl =
        Rune.slice [ Rune.A; Rune.R (2, h + 2); Rune.R (0, w) ] mag_padded
      in
      let tl = Rune.slice [ Rune.A; Rune.R (0, h); Rune.R (0, w) ] mag_padded in
      let br =
        Rune.slice [ Rune.A; Rune.R (2, h + 2); Rune.R (2, w + 2) ] mag_padded
      in
      let ge a b = Rune.greater_equal a b in
      let is_max =
        Rune.logical_or
          (Rune.logical_or
             (Rune.logical_and is_horizontal
                (Rune.logical_and (ge center left) (ge center right)))
             (Rune.logical_and is_diag1
                (Rune.logical_and (ge center tr) (ge center bl))))
          (Rune.logical_or
             (Rune.logical_and is_vertical
                (Rune.logical_and (ge center top) (ge center bottom)))
             (Rune.logical_and is_diag2
                (Rune.logical_and (ge center tl) (ge center br))))
      in
      let nms = Rune.where is_max mag3 (Rune.zeros_like mag3) in
      (* 4. Double thresholding *)
      let strong = Rune.greater nms (Rune.scalar_like nms high) in
      let weak =
        Rune.logical_and
          (Rune.greater_equal nms (Rune.scalar_like nms low))
          (Rune.logical_not strong)
      in
      (* 5. Hysteresis via dilation *)
      let one = Rune.ones_like nms in
      let zero = Rune.zeros_like nms in
      let strong_map = Rune.where strong one zero in
      let strong_4d = Rune.reshape [| n; h; w; 1 |] strong_map in
      let k3 = Morphology.structuring_element Rect (3, 3) in
      let dilated =
        Morphology.dilate ~kernel:k3 (Morphology.dilate ~kernel:k3 strong_4d)
      in
      let dilated3 = Rune.reshape [| n; h; w |] dilated in
      let connected = Rune.greater dilated3 zero in
      let final =
        Rune.where
          (Rune.logical_and connected (Rune.logical_or strong weak))
          one zero
      in
      Rune.reshape [| n; h; w; 1 |] final)
    img
