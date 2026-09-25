(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor preparation before range assignment.

    Owns call inlining, multi-device normalization, materialization, store
    hazards, reduction splitting and movement cleanup. *)

val prepare_rangeify : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [prepare_rangeify sink] normalizes the tensor graph for range assignment.
    Anonymous materializations declare storage whose owner is bound when the
    enclosing call is scheduled. *)

val movement_ops : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [movement_ops u] pushes movement through INDEX, AFTER and END. *)

val contiguous_view : Tolk_uop.Uop.t -> (Tolk_uop.Uop.t * int) option
(** [contiguous_view u] is the graph anchor and byte offset of a proven
    contiguous view. The anchor retains pending effects and may be a bitcast.
    It need not own allocated storage. Returns [None] when a constant offset
    cannot be proved, or the device does not support views.

    Raises [Invalid_argument] if the byte offset does not fit a host integer. *)

val detect_expanded : Tolk_uop.Uop.t -> bool list
(** [detect_expanded u] identifies axes broadcast by the movements above [u]'s
    first non-movement node. Empty when [u] has no concrete shape. *)
