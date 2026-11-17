(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Results that take the memory of donated arguments, on any set of
    devices. *)

val owned : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [owned x] is a copy of [x] where it lies, in memory of its own, which a
    donation of it may lend: a compiled copy, so that it needs no eager
    kernels there. A value placed where it lies already borrows its memory,
    which shares it, so that its donation lends nothing. *)

val carry : (module Nx.Devices) -> unit
(** [carry s] checks that a scan's final carry on [s]'s devices whose result
    takes the memory of a donated argument it does not read gives the
    undonated call's values, for a float carry holding the result's bytes
    less than the undonated call, and for a carry of four bits, which the
    loop holds unpacked, too. Where the host computes on the devices
    ({!Rig.runs_on_host}), each result lies in the argument's memory. *)
