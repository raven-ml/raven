(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let gradient_pair ~kernel_x ~kernel_y img =
  Helpers.with_batch_pair
    (fun img ->
      let gx = Helpers.convolve_per_channel kernel_x img in
      let gy = Helpers.convolve_per_channel kernel_y img in
      (gx, gy))
    img

(* Sobel kernels *)

let check_sobel_ksize ksize =
  match ksize with
  | 1 | 3 | 5 | 7 -> ()
  | _ ->
      invalid_arg
        (Printf.sprintf "sobel: ksize must be 1, 3, 5 or 7, got %d" ksize)

let check_laplacian_ksize ksize =
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg
      (Printf.sprintf "laplacian: ksize must be a positive odd integer, got %d"
         ksize)

(* The 1D Sobel kernel of derivative [order] for aperture [ksize]: Pascal's
   rule smooths [ksize - order - 1] times, then [order] differences follow, so
   order 0 of aperture 5 is [1 4 6 4 1] and order 1 is [-1 -2 0 2 1]. Aperture
   1 does not smooth: order 0 is [1], a derivative takes 3 taps. *)
let deriv ~order ksize =
  let size = if ksize = 1 && order > 0 then 3 else ksize in
  let k = Array.make size 0.0 in
  k.(0) <- 1.0;
  for _ = 1 to size - order - 1 do
    for j = size - 1 downto 1 do
      k.(j) <- k.(j) +. k.(j - 1)
    done
  done;
  for _ = 1 to order do
    for j = size - 1 downto 1 do
      k.(j) <- k.(j - 1) -. k.(j)
    done;
    k.(0) <- -.k.(0)
  done;
  k

(* The 2D kernel whose rows follow [ky] and columns follow [kx], beside
   [img]. *)
let outer img ky kx =
  let h = Array.length ky and w = Array.length kx in
  Helpers.beside img
    (Nx.create Nx.float32 [| h; w |]
       (Array.init (h * w) (fun i -> ky.(i / w) *. kx.(i mod w))))

let scharr_x = [| -3.; 0.; 3.; -10.; 0.; 10.; -3.; 0.; 3. |]
let scharr_y = [| -3.; -10.; -3.; 0.; 0.; 0.; 3.; 10.; 3. |]

let sobel ?(ksize = 3) img =
  check_sobel_ksize ksize;
  let d = deriv ~order:1 ksize and s = deriv ~order:0 ksize in
  gradient_pair ~kernel_x:(outer img s d) ~kernel_y:(outer img d s) img

let scharr img =
  let kernel taps = Helpers.beside img (Nx.create Nx.float32 [| 3; 3 |] taps) in
  gradient_pair ~kernel_x:(kernel scharr_x) ~kernel_y:(kernel scharr_y) img

(* The sum of the second x and y Sobel derivatives. For aperture 1 the two
   3-tap kernels [1 -2 1] add up to the aperture [0 1 0; 1 -4 1; 0 1 0]. *)
let laplacian ?(ksize = 1) img =
  check_laplacian_ksize ksize;
  let d = deriv ~order:2 ksize and s = deriv ~order:0 ksize in
  let dxx, dyy =
    gradient_pair ~kernel_x:(outer img s d) ~kernel_y:(outer img d s) img
  in
  Nx.add dxx dyy

let canny ~low ~high ?(sigma = 1.4) img =
  Helpers.with_batch
    (fun img ->
      (* 1. Gaussian blur *)
      let blurred = Filter.gaussian_blur ~sigma img in
      (* 2. Gradient computation *)
      let gx, gy = sobel blurred in
      let mag = Nx.sqrt (Nx.add (Nx.square gx) (Nx.square gy)) in
      let angle = Nx.atan2 gy gx in
      let shape = Nx.shape mag in
      let n = shape.(0) and h = shape.(1) and w = shape.(2) in
      let mag3 = Nx.reshape [| n; h; w |] mag in
      let angle3 = Nx.reshape [| n; h; w |] angle in
      (* 3. Non-maximum suppression *)
      let angle_deg = Nx.mul_s angle3 (180.0 /. Float.pi) in
      let angle_pos =
        Nx.where (Nx.less_s angle_deg 0.0) (Nx.add_s angle_deg 180.0) angle_deg
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
      let nms = Nx.where is_max mag3 (Nx.scalar_like mag3 0.0) in
      (* 4. Double thresholding *)
      let strong = Nx.greater nms (Nx.scalar_like nms high) in
      let weak =
        Nx.logical_and
          (Nx.greater_equal nms (Nx.scalar_like nms low))
          (Nx.logical_not strong)
      in
      (* 5. Hysteresis via dilation *)
      let one = Nx.scalar_like nms 1.0 in
      let zero = Nx.scalar_like nms 0.0 in
      let strong_map = Nx.where strong one zero in
      let strong_4d = Nx.reshape [| n; h; w; 1 |] strong_map in
      let k3 = Morphology.structuring_element Rect (3, 3) in
      let dilated =
        Morphology.dilate ~kernel:k3 (Morphology.dilate ~kernel:k3 strong_4d)
      in
      let dilated3 = Nx.reshape [| n; h; w |] dilated in
      let connected = Nx.greater_s dilated3 0.0 in
      let final =
        Nx.where (Nx.logical_and connected (Nx.logical_or strong weak)) one zero
      in
      Nx.reshape [| n; h; w; 1 |] final)
    img
