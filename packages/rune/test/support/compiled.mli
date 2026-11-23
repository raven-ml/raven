(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tests of what a compiled call does on the host. *)

val here : unit -> unit
(** [here ()] skips the running test where rig links no host code on {!Rig.host}
    ({!Rig.Image.links}), saying so. [Rune.jit] declines a program on the host
    there and evaluates it, so what a test asserts of a compiled call, such as
    that no call evaluated, does not hold. *)
