(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tests of what a compiled call does. *)

val here : ?on:Rig.t -> unit -> unit
(** [here ~on ()] skips the running test where [on]'s programs compile with
    Clang, as a host's, a software device's or another machine's host's do, and
    cannot compile on this machine ({!Tolk_engine.compiles}), with the reason,
    as where Clang is missing or rig links no host code for [on]. A GPU's
    compiler that fails fails the test. [on] defaults to {!Rig.host}. [Rune.jit]
    then raises at the call that compiles, or evaluates the program, so what a
    test asserts of a compiled call does not hold. A test calls it before it
    compiles. *)
