(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** rune's constructs in a compiled call's trace.

    A trace lowers every operation of its extent ({!Lower.op}) and answers
    rune's constructs ({!Construct.t}) outside that interception, so that the
    operations an answer issues are lowered in a scope of its choosing. *)

val install : Lower.scope -> (unit -> 'a) -> 'a
(** [install s f] is [f ()] traced in [s]: every operation of its extent lowered
    by [Lower.op s], and the constructs it performs answered as follows.
    - [Detach x] is [x].
    - Every other construct passes outward: a scan folds where it is written,
      its steps traced one after the other.

    Raises as [Lower.op] does. *)
