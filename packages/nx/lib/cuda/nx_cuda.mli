(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CUDA GPUs as nx devices.

    A device of this library is a CUDA GPU's memory, which {!Nx_cuda_device}
    opens, named [CUDA], [CUDA:1], ... It computes eagerly with no backend:
    [Rune.jit] compiles for it, and an eager operation on its values raises
    [Invalid_argument] naming the remedies. Constants, views, reads and
    [Nx.place] work on it.

    GPUs are numbered in PCI bus order, so GPU [i] is the one [nvidia-smi]
    numbers [i] when the driver sees every GPU. *)

val get : int -> (Nx.Device.t, string) result
(** [get i] is CUDA GPU [i], opened now, or [Error msg] with the driver's
    reason, such as that the driver library cannot be loaded or that there is no
    GPU [i]. Every [get i] that succeeds gives an equal device. A failed open is
    not remembered, so a later [get] tries again, except where the driver
    library failed to load.

    Raises [Invalid_argument] if [i < 0]. *)

val device : int -> Nx.Device.t
(** [device i] is {!get}[ i].

    Raises [Failure] with {!get}'s reason if it does not open, and as {!get}
    does. *)
