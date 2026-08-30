(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** HIP kernel compiler.

    Compiles HIP C++ kernel source to an HSA code object through the AMD code
    object manager library. [libamd_comgr] is loaded dynamically at first
    compile — searched in [$ROCM_PATH/lib] (default [/opt/rocm/lib]), then on
    the dynamic linker paths — so this library builds and loads on machines
    without ROCm; compilation raises there instead. *)

val create : arch:string -> Tolk.Compiler.t
(** [create ~arch] is a compiler producing HSA code objects for [arch], a
    full gfx architecture string such as ["gfx1100"].

    {!Tolk.Compiler.compile} raises [Tolk.Compiler.Compile_error] when
    compilation fails (the message carries the compiler log) and [Failure]
    when [libamd_comgr] cannot be loaded. *)

val version : unit -> int * int
(** [version ()] is the major and minor version of the loaded [libamd_comgr].
    Raises [Failure] when the library cannot be loaded. *)
