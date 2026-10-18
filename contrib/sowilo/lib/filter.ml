(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let generate_gaussian_kernel size sigma =
  let center = float (size / 2) in
  let sigma2_sq = 2.0 *. sigma *. sigma in
  let positions = Nx.arange_f Nx.float32 0.0 (float size) 1.0 in
  let x = Nx.sub_s positions center in
  let kernel = Nx.exp (Nx.div_s (Nx.neg (Nx.square x)) sigma2_sq) in
  let sum = Nx.sum kernel in
  Nx.div kernel (Nx.reshape [||] sum)

(* [2 * half + 1] fits an int while [half] is below [max_int / 2]; an
   infinite [sigma] fails the test too. *)
let default_ksize sigma =
  let half = Float.ceil (3.0 *. sigma) in
  if not (half < Float.of_int (max_int / 2)) then
    invalid_arg
      (Printf.sprintf "gaussian_blur: sigma %g is too large for a default ksize"
         sigma);
  (2 * int_of_float half) + 1

(* The taps OpenCV's getGaussianKernel fixes for a sigma of 0 or less, up to
   an odd size of 9: the kernel is the taps over their sum. *)
let fixed_taps = function
  | 1 -> Some [| 1. |]
  | 3 -> Some [| 1.; 2.; 1. |]
  | 5 -> Some [| 1.; 4.; 6.; 4.; 1. |]
  | 7 -> Some [| 2.; 7.; 14.; 18.; 14.; 7.; 2. |]
  | 9 -> Some [| 4.; 13.; 30.; 51.; 60.; 51.; 30.; 13.; 4. |]
  | _ -> None

(* The kernel of a sigma of 0 or less, as getGaussianKernel builds it: a fixed
   table, or beyond the tables the sigma derived from [ksize]. *)
let derived_kernel ksize =
  match fixed_taps ksize with
  | Some taps ->
      let sum = Array.fold_left ( +. ) 0.0 taps in
      Nx.create Nx.float32 [| ksize |] (Array.map (fun t -> t /. sum) taps)
  | None ->
      let sigma = (0.3 *. ((Float.of_int (ksize - 1) *. 0.5) -. 1.0)) +. 0.8 in
      generate_gaussian_kernel ksize sigma

let gaussian_blur ~sigma ?ksize img =
  if Float.is_nan sigma then invalid_arg "gaussian_blur: sigma is NaN";
  let positive = sigma > 0.0 in
  let ksize =
    match ksize with
    | Some ksize -> ksize
    | None when positive -> default_ksize sigma
    | None ->
        invalid_arg
          (Printf.sprintf
             "gaussian_blur: sigma must be positive without a ksize, got %g"
             sigma)
  in
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg
      (Printf.sprintf
         "gaussian_blur: ksize must be a positive odd integer, got %d" ksize);
  let kernel_1d =
    if positive then generate_gaussian_kernel ksize sigma
    else derived_kernel ksize
  in
  let kernel_h = Nx.reshape [| 1; ksize |] kernel_1d in
  let kernel_v = Nx.reshape [| ksize; 1 |] kernel_1d in
  Helpers.with_batch
    (fun img ->
      let temp = Helpers.convolve_per_channel kernel_h img in
      Helpers.convolve_per_channel kernel_v temp)
    img

let box_blur ~ksize img =
  if ksize <= 0 then invalid_arg "box_blur: ksize must be positive";
  let value = 1.0 /. float (ksize * ksize) in
  let kernel = Nx.full Nx.float32 [| ksize; ksize |] value in
  Helpers.with_batch (fun img -> Helpers.convolve_per_channel kernel img) img

let median_blur ~ksize img =
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg "median_blur: ksize must be a positive odd integer";
  let pad_size = ksize / 2 in
  Helpers.with_batch
    (fun img ->
      let shape = Nx.shape img in
      let h = shape.(1) and w = shape.(2) in
      let padded =
        Nx.pad
          [| (0, 0); (pad_size, pad_size); (pad_size, pad_size); (0, 0) |]
          0.0 img
      in
      let windows = ref [] in
      for dy = 0 to ksize - 1 do
        for dx = 0 to ksize - 1 do
          let slice =
            Nx.slice [ Nx.A; Nx.R (dy, dy + h); Nx.R (dx, dx + w); Nx.A ] padded
          in
          windows := slice :: !windows
        done
      done;
      let stacked = Nx.stack ~axis:0 (List.rev !windows) in
      let sorted, _ = Nx.sort ~axis:0 stacked in
      let median_idx = ksize * ksize / 2 in
      Nx.slice [ Nx.I median_idx; Nx.A; Nx.A; Nx.A; Nx.A ] sorted)
    img

let filter2d kernel img =
  Helpers.with_batch (fun img -> Helpers.convolve_per_channel kernel img) img

let unsharp_mask ~sigma ?(amount = 1.0) img =
  let blurred = gaussian_blur ~sigma img in
  let diff = Nx.sub img blurred in
  Nx.add img (Nx.mul_s diff amount)
