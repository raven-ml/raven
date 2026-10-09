(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx.kernel's suite needs beyond the library's interface. *)

val contract_fields : Nx_kernel.Spec.contract Nx_kernel.Spec.t -> int array
(** [contract_fields s] is [s] read by C through [nx_spec.h]'s
    [nx_spec_contract]: the family, [acc]'s and [out]'s codes, [init], the
    counts of batch and contracting pairs, then each pair's two axes, batch
    pairs first. *)
