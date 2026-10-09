(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels for host memory.

    [computes_on d] is {!Rig.shares_host_memory}[ d]. Each kernel computes from
    operands of [dst]'s shape and any layouts, on memory the host addresses. It
    claims every operand through the door of [nx_array.h] for the extent of its
    call. It answers [Done] once its work is done, or, before any write, a
    refusal: the door's, or [Shape_mismatch] if the operands' shapes differ.

    [apply1] computes [Copy] and [Cast], and answers [Declined] for the other
    kinds. A [Copy] whose operand's dtype is not [dst]'s is refused with
    [Wrong_dtype]. [contract] answers [Declined]. *)

include Nx_kernel.S
