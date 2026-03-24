(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Resolved GPU target descriptors for renderer construction.

    This module owns CUDA/AMD target selection policy. Renderers consume these
    resolved targets and do not read environment variables themselves. *)

(** CUDA SM architecture tiers used by source generation. *)
type cuda = SM75 | SM80 | SM89

(** AMD GPU architecture families used by source generation. *)
type amd = RDNA3 | RDNA4 | CDNA3 | CDNA4

val cuda_of_env : unit -> cuda option
(** [cuda_of_env ()] resolves [CUDA_ARCH] or [CUDA_SM] to the nearest supported
    CUDA SM tier. Returns [None] when no supported tier is configured. *)

val amd_of_env : unit -> amd option
(** [amd_of_env ()] resolves common AMD arch environment variables such as
    [AMD_ARCH], [HIP_ARCH], [HCC_AMDGPU_TARGET], or [HSA_OVERRIDE_GFX_VERSION].
    Returns [None] when no supported arch family is configured. *)
