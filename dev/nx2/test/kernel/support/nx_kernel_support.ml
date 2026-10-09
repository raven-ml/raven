(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external contract_fields : Nx_kernel.Spec.contract Nx_kernel.Spec.t -> int array
  = "nx_kernel_support_contract"
