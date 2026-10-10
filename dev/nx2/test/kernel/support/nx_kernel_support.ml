(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external contract_fields : Nx_kernel.Spec.contract Nx_kernel.Spec.t -> int array
  = "nx_kernel_support_contract"

external view_fields : Nx_kernel.Spec.Contract_view.t -> int array
  = "nx_kernel_support_view"

external prog : Nx_kernel.Prog.t -> string = "nx_kernel_support_prog"

external loop : [< `Map | `Reduce | `Scan ] Nx_kernel.Spec.t -> string
  = "nx_kernel_support_loop"

external axis_fields : [< `Gather | `Scatter | `Sort ] Nx_kernel.Spec.t -> int array
  = "nx_kernel_support_axis"

external shaped : [< `Assemble | `Fold ] Nx_kernel.Spec.t -> string
  = "nx_kernel_support_shaped"

external fft_fields : Nx_kernel.Spec.fft Nx_kernel.Spec.t -> int array
  = "nx_kernel_support_fft"

external linalg_fields : Nx_kernel.Spec.linalg Nx_kernel.Spec.t -> int array
  = "nx_kernel_support_linalg"
