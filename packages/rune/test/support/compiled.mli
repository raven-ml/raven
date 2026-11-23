(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tests of what a compiled call does on the host. *)

val here : ?on:Rig.t -> unit -> unit
(** [here ~on ()] skips the running test where [on] (defaults to {!Rig.host}) is
    a host or a software device and rig links no host code for its host's
    instruction set ({!Rig.Image.links}), saying so. [Rune.jit] declines a
    program that computes there and evaluates it, so what a test asserts of a
    compiled call, such as that no call evaluated, does not hold. *)
