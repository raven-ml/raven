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
  Nx.create Nx.float32 [| 3; 3 |] [| -1.; 0.; 1.; -2.; 0.; 2.; -1.; 0.; 1. |]

let sobel_ky =
  Nx.create Nx.float32 [| 3; 3 |] [| -1.; -2.; -1.; 0.; 0.; 0.; 1.; 2.; 1. |]

let scharr_kx =
  Nx.create Nx.float32 [| 3; 3 |] [| -3.; 0.; 3.; -10.; 0.; 10.; -3.; 0.; 3. |]

let scharr_ky =
  Nx.create Nx.float32 [| 3; 3 |] [| -3.; -10.; -3.; 0.; 0.; 0.; 3.; 10.; 3. |]

let sobel ?(ksize = 3) img =
  ignore ksize;
  gradient_pair ~kernel_x:sobel_kx ~kernel_y:sobel_ky img

let scharr img = gradient_pair ~kernel_x:scharr_kx ~kernel_y:scharr_ky img

let laplacian ?(ksize = 3) img =
  ignore ksize;
  (* Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0] *)
  let kernel =
    Nx.create Nx.float32 [| 3; 3 |]
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
      let mag = Nx.sqrt (Nx.add (Nx.square gx) (Nx.square gy)) in
      let angle = Nx.atan2 gy gx in
      let shape = Nx.shape mag in
      let n = shape.(0) and h = shape.(1) and w = shape.(2) in
      let mag3 = Nx.reshape [| n; h; w |] mag in
      let angle3 = Nx.reshape [| n; h; w |] angle in
      (* 3. Non-maximum suppression *)
      let angle_deg = Nx.mul_s angle3 (180.0 /. Float.pi) in
      let angle_pos =
        Nx.where
          (Nx.less angle_deg (Nx.zeros_like angle_deg))
          (Nx.add_s angle_deg 180.0) angle_deg
      in
      let scalar v = Nx.scalar_like angle_pos v in
      let is_horizontal =
        Nx.logical_or
          (Nx.logical_and
             (Nx.greater_equal angle_pos (scalar 0.0))
             (Nx.less angle_pos (scalar 22.5)))
          (Nx.logical_and
             (Nx.greater_equal angle_pos (scalar 157.5))
             (Nx.less_equal angle_pos (scalar 180.0)))
      in
      let is_diag1 =
        Nx.logical_and
          (Nx.greater_equal angle_pos (scalar 22.5))
          (Nx.less angle_pos (scalar 67.5))
      in
      let is_vertical =
        Nx.logical_and
          (Nx.greater_equal angle_pos (scalar 67.5))
          (Nx.less angle_pos (scalar 112.5))
      in
      let is_diag2 =
        Nx.logical_and
          (Nx.greater_equal angle_pos (scalar 112.5))
          (Nx.less angle_pos (scalar 157.5))
      in
      let mag_padded = Nx.pad [| (0, 0); (1, 1); (1, 1) |] 0.0 mag3 in
      let center =
        Nx.slice [ Nx.A; Nx.R (1, h + 1); Nx.R (1, w + 1) ] mag_padded
      in
      let left = Nx.slice [ Nx.A; Nx.R (1, h + 1); Nx.R (0, w) ] mag_padded in
      let right =
        Nx.slice [ Nx.A; Nx.R (1, h + 1); Nx.R (2, w + 2) ] mag_padded
      in
      let top = Nx.slice [ Nx.A; Nx.R (0, h); Nx.R (1, w + 1) ] mag_padded in
      let bottom =
        Nx.slice [ Nx.A; Nx.R (2, h + 2); Nx.R (1, w + 1) ] mag_padded
      in
      let tr = Nx.slice [ Nx.A; Nx.R (0, h); Nx.R (2, w + 2) ] mag_padded in
      let bl = Nx.slice [ Nx.A; Nx.R (2, h + 2); Nx.R (0, w) ] mag_padded in
      let tl = Nx.slice [ Nx.A; Nx.R (0, h); Nx.R (0, w) ] mag_padded in
      let br = Nx.slice [ Nx.A; Nx.R (2, h + 2); Nx.R (2, w + 2) ] mag_padded in
      let ge a b = Nx.greater_equal a b in
      let is_max =
        Nx.logical_or
          (Nx.logical_or
             (Nx.logical_and is_horizontal
                (Nx.logical_and (ge center left) (ge center right)))
             (Nx.logical_and is_diag1
                (Nx.logical_and (ge center tr) (ge center bl))))
          (Nx.logical_or
             (Nx.logical_and is_vertical
                (Nx.logical_and (ge center top) (ge center bottom)))
             (Nx.logical_and is_diag2
                (Nx.logical_and (ge center tl) (ge center br))))
      in
      let nms = Nx.where is_max mag3 (Nx.zeros_like mag3) in
      (* 4. Double thresholding *)
      let strong = Nx.greater nms (Nx.scalar_like nms high) in
      let weak =
        Nx.logical_and
          (Nx.greater_equal nms (Nx.scalar_like nms low))
          (Nx.logical_not strong)
      in
      (* 5. Hysteresis via dilation *)
      let one = Nx.ones_like nms in
      let zero = Nx.zeros_like nms in
      let strong_map = Nx.where strong one zero in
      let strong_4d = Nx.reshape [| n; h; w; 1 |] strong_map in
      let k3 = Morphology.structuring_element Rect (3, 3) in
      let dilated =
        Morphology.dilate ~kernel:k3 (Morphology.dilate ~kernel:k3 strong_4d)
      in
      let dilated3 = Nx.reshape [| n; h; w |] dilated in
      let connected = Nx.greater dilated3 zero in
      let final =
        Nx.where (Nx.logical_and connected (Nx.logical_or strong weak)) one zero
      in
      Nx.reshape [| n; h; w; 1 |] final)
    img
