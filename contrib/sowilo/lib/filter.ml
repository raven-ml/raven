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

let gaussian_blur ~sigma ?(ksize = 0) img =
  let ksize =
    if ksize > 0 then ksize
    else (2 * int_of_float (Float.round (3.0 *. sigma))) + 1
  in
  if ksize <= 0 || ksize mod 2 = 0 then
    invalid_arg "gaussian_blur: ksize must be a positive odd integer";
  let kernel_1d = generate_gaussian_kernel ksize sigma in
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
