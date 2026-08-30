(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVRTC kernel compiler.

    Compiles CUDA C kernel source through the NVRTC library ([libnvrtc.so]),
    loaded dynamically at first use so this library links on machines without
    a CUDA installation; compilation raises there instead. *)

val create : ?ptx:bool -> cache_key:string -> string -> Tolk.Compiler.t
(** [create ~cache_key arch] is a compiler producing PTX for GPU
    architecture [arch] (e.g. ["sm_89"]); with [~ptx:false] it produces the
    cubin binary for exactly that architecture instead.

    The compiler is named [String.uppercase_ascii cache_key] and stores
    results in the on-disk compile cache under
    ["compile_" ^ cache_key ^ "_" ^ arch]. PTX and cubin producers must use
    distinct [cache_key]s so their cached outputs never mix.

    Include paths default to the standard CUDA toolkit locations, overridden
    by the [CUDA_PATH] environment variable.

    Raises [Failure] if the NVRTC library is not available. *)
