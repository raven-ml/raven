(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs, as their formats depend on them.

    A GPU's compute engine class selects its launch descriptor version, and
    its multiprocessors' version selects the machine code its kernels run.
    Its local memory is sized by the number of its multiprocessors and the
    warps each runs. A launch also depends on its {e windows}: the addresses
    at which the GPU's channels show kernels their shared and local memory.
    A driver reads the class and the counts from the GPU when it opens it,
    chooses the windows and sets them on each channel it opens
    ({!Method.shared_memory_window}), and holds the local memory its
    channels give kernels. It declares the whole as a {!t} ({!declare}), and
    compiled code finds it under {!key}. Neither library links the
    other.

    Ring segments, launch descriptors and semaphores must lie below [2{^40}]
    ({!Method}), and the driver keeps the memory that holds them there. The
    windows lie in the 8 GiB below [2{^40}], where no memory the driver gives
    lies; its [Device] memory may lie at any other address the GPU's
    channels reach. *)

type t = {
  compute_class : int;
      (** The class of its compute engine: one of [0xc7c0] (Ampere), [0xc9c0]
          (Ada) and [0xcec0] (Blackwell). *)
  sass_version : int;
      (** The version of the machine code its multiprocessors run, from the SM
          version the GPU reports (NV2080_CTRL_GR_INFO_INDEX_SM_VERSION): that
          version's bits 8 to 11 in bits 4 to 7, and its bits 0 to 3 in bits 0
          to 3. The SM version [0x809] (sm_89) gives [0x89], [0xa04] (sm_120)
          gives [0xa4]. *)
  gpcs : int;  (** Its graphics processing clusters. *)
  tpcs_per_gpc : int;  (** The texture processing clusters of one. *)
  sms_per_tpc : int;  (** The streaming multiprocessors of one of those. *)
  warps_per_sm : int;  (** The most warps a multiprocessor runs concurrently. *)
  shared_window : int;
      (** The address at which its kernels see their shared memory. *)
  local_window : int;
      (** The address at which its kernels see their local memory. *)
  local : int -> (unit, string) result;
      (** The GPU's channels give every kernel they launch one local memory, the
          driver's ({!Method.local_memory}). [local n] makes it serve kernels
          whose threads need up to [n] bytes each ({!Launch.local_bytes}): it
          does nothing if the memory already does, else replaces it with a
          larger one ({!Local_memory.make}) and keeps the old one until the work
          placed before the call completes. The result is [Error msg] if the
          device cannot allocate it. Any domain may call it. *)
}
(** The type for GPUs. Every count is positive. *)
