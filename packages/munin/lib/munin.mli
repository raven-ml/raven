(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Local experiment tracking for Raven.

    Munin is a local-first experiment tracker. Start with {!Session} to write
    runs and {!Metric} to log scalars into them, {!Run} to read them back,
    {!Store} for discovery, and {!Artifact} for versioned payloads.

    {1:library Library [munin]}
    {!modules:Value Metric Provenance Session Run Store Artifact} *)

module Value = Value
module Metric = Metric
module Provenance = Provenance
module Artifact = Artifact
module Run = Run
module Session = Session
module Store = Store
