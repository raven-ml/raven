(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external prog : Nx_kernel.Prog.t -> string = "nx_kernel_support_prog"

external op1 : Nx_kernel.Prog.op1 -> int = "nx_kernel_support_op1"
external op2 : Nx_kernel.Prog.op2 -> int = "nx_kernel_support_op2"
external tag : 'a -> int = "nx_kernel_support_tag"
external code : string -> int = "nx_kernel_support_code"
