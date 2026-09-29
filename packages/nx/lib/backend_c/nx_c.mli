(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx.c's kernels over host memory.

    A value of [Nx_c] is a view of a buffer on {!Nx_device.host}. Its
    operations are nx's vocabulary, run by the C engine on the calling domain
    and the engine's thread pool. [Nx_c] is the kernel layer of
    [Nx.Backend.host]: programs use [Nx], whose values on the host are these. *)

type context = unit
(** nx.c needs no context: the host is its only device. *)

include Nx_core.Backend_intf.S with type context := context

val view : ('a, 'b) t -> Nx_core.View.t
(** [view t] is [t]'s shape, strides and offset over its buffer, in elements. *)

val dtype : ('a, 'b) t -> ('a, 'b) Nx_dtype.t
(** [dtype t] is [t]'s dtype. *)

val context : ('a, 'b) t -> context
(** [context t] is [()]. *)
