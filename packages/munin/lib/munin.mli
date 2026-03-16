(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Local experiment tracking for Raven.

    Munin is a local-first experiment tracker. Start with {!Session} to write
    runs, {!Run} to read them back, {!Run_monitor} for live polling, {!Store}
    for discovery, and {!Artifact} for versioned payloads.

    {1:library Library [munin]}
    {!modules:Value Session Run Run_monitor Store Artifact} *)

module Value = Value
module Artifact = Artifact
module Run = Run
module Run_monitor = Run_monitor
module Session = Session
module Store = Store
