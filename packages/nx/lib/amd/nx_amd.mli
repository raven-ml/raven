(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's eager kernels on AMD GPUs.

    {!backend} computes eagerly on the memory of an AMD GPU that rig opens,
    through either of its drivers ([Rig_amd_amdgpu], [Rig_amd_pci]), from code
    objects the library carries, compiled for [gfx12-generic]:
    {[
      let d =
        let gpu = Result.get_ok (Rig.open_ (Rig_amd_amdgpu.gpu 0)) in
        Nx.Device.make ~backend:Nx_amd.backend gpu
    ]}
    GPUs of the gfx12 generation (gfx1200, gfx1201) copy values
    ({!Nx.copy}, {!Nx.contiguous}), cast them ({!Nx.cast}), pad, concatenate and
    gather them, write windows of them and scatter into them ({!Nx.pad},
    {!Nx.concatenate}, {!Nx.take}, {!Nx.take_along_axis}, {!Nx.set},
    {!Nx.scatter}), take their sliding windows and sum them back
    ({!Nx.extract_patches}, {!Nx.combine_patches}), and compute the elementwise
    functions of one, two and three operands, the comparisons and {!Nx.where},
    reduce them ({!Nx.sum}, {!Nx.prod}, {!Nx.max}, {!Nx.min}, {!Nx.argmax},
    {!Nx.argmin}), run along them ({!Nx.cumsum}, {!Nx.cumprod}, {!Nx.cummax},
    {!Nx.cummin}), sort them ({!Nx.sort}, {!Nx.argsort}), multiply matrices
    ({!Nx.matmul}) and draw random values ({!Nx.Rng}), of every dtype but the
    complex ones, [int4], [uint4] and [bit]: bit for bit as the host does, but
    the transcendental functions, which keep the accuracy {!Nx} states, the sign
    and payload of a NaN an arithmetic operation makes, the NaN that {!Nx.max}
    or {!Nx.min} over several axes takes, the first in C order, and the rounding
    of float sums, running sums, scattered sums, products and matrix products,
    whose terms associate differently. Any other eager operation, dtype or GPU
    raises [Invalid_argument] naming the remedies: [Rune.jit] compiles for every
    AMD device, and [Nx.place] moves values to the host. Constants, views and
    reads work on every AMD device. *)

val backend : Nx_backend.t
(** [backend] is nx.amd's kernels. They compute on every AMD GPU's memory
    ([Nx_backend.S.runs_on]) and own it ([Nx_backend.S.owns]), so a device
    over that memory computed by them is named after the GPU alone. *)
