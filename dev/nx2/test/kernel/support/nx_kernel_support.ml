(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external view_fields : Nx_kernel.Contract_view.t -> int array
  = "nx_kernel_support_view"

external prog : Nx_kernel.Prog.t -> string = "nx_kernel_support_prog"

external tag : 'a -> int = "nx_kernel_support_tag"
external code : string -> int = "nx_kernel_support_code"
