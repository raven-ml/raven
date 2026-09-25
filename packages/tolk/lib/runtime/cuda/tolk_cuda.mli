(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** CUDA GPU device backend.

    [Tolk_cuda] provides a {!Tolk.Device.t} that executes compiled kernels on
    NVIDIA GPUs through the CUDA driver API. Construct a device with {!create}
    and interact with it through the {!Tolk.Device} interface.

    The CUDA driver ([libcuda.so.1]) and NVRTC ([libnvrtc.so]) libraries are
    loaded dynamically at device creation, so this library links on machines
    without a CUDA installation; {!create} raises there instead.

    {1:compilation Kernel compilation}

    Kernels are compiled to PTX with NVRTC for the device's compute-capability
    architecture and JIT-compiled by the driver at module load. Compilation results
    are stored in the on-disk compile cache.

    {1:env Environment variables}

    - [DEV=CUDA:CUDA:sm_80] overrides the renderer and compiler architecture.
      The default is the device's exact compute capability.
    - [CUDA_PATH] — CUDA toolkit root used for compile-time include paths.
      Defaults to the standard system locations. *)

(** {1:device Device} *)

val create : string -> Tolk.Device.t
(** [create name] is a CUDA device identified by [name].

    The device ordinal is parsed from [name] (["CUDA:1"] opens device 1;
    ["CUDA"] defaults to device 0). The device uses an LRU-cached allocator
    over CUDA device memory with pinned host staging for host-to-device and
    device-to-host copies, a {!Tolk.Cstyle.cuda} renderer compiled through
    NVRTC, and host-compiled compute and copy queues sharing the allocation,
    timeline and replay protocol with the other backends.

    Raises [Failure] if no CUDA driver or GPU is available, and
    [Invalid_argument] if the suffix after [':'] is not an integer. *)
