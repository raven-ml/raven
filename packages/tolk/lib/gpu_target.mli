(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Resolved GPU target descriptors for renderer construction.

    These descriptors select source-generation capabilities. Exact compiler
    architectures and runtime selection live in {!Tolk_uop.Target}. *)

(** CUDA SM architecture tiers used by source generation. *)
type cuda = SM75 | SM80 | SM89 | SM90

(** AMD GPU architecture families used by source generation. *)
type amd = RDNA3 | RDNA4 | CDNA3 | CDNA4

(** Metal GPU family used by source generation. Mirrors tinygrad's Metal
    target architecture string, e.g. ["Apple7"] or ["Mac2"]. *)
type metal = Apple of int | Mac of int

(** OpenCL target architecture string used by source generation. Mirrors
    tinygrad's comma-separated [Target.arch], normally the device extension
    list such as ["cl_khr_fp16,cl_khr_fp64"]. *)
type opencl = string

(** CPU architecture family used by source generation. Mirrors tinygrad's
    normalized CPU target architecture prefix. *)
type cpu = X86_64 | Arm64 | Riscv64

val cpu_of_machine : string -> cpu option
(** [cpu_of_machine s] normalizes host machine names such as ["amd64"] or
    ["aarch64"] to CPU renderer targets. *)

val host_cpu : unit -> cpu
(** [host_cpu ()] is the CPU renderer target for the current host.

    Raises [Invalid_argument] if the host architecture is unsupported. *)

val cuda_of_sm : int -> cuda option
(** [cuda_of_sm sm] is the source-generation tier for compute capability [sm],
    given as major*10+minor (e.g. [89] for sm_89). Capabilities newer than the
    highest supported tier map to that tier. Returns [None] below [75]. *)

val parse_cuda_arch : string -> cuda option
(** [parse_cuda_arch s] resolves a CUDA architecture such as ["sm_89"] to its
    source-generation tier. Returns [None] for an unsupported architecture. *)

val parse_amd_arch : string -> amd option
(** [parse_amd_arch s] normalizes AMD architecture names such as ["gfx1100"]
    or dotted graphics versions such as ["11.0.0"] to their renderer target
    family. Returns [None] for an unsupported architecture. *)

val parse_metal_arch : string -> metal option
(** [parse_metal_arch s] parses tinygrad-style Metal architecture names such as
    ["Apple7"] and ["Mac2"]. *)
