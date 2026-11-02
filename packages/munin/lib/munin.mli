(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Records of runs and artifacts, in plain files.

    munin records what a program does as it runs: its params, its metrics
    step by step, readings of its machine and devices, files it produces,
    and versioned artifacts such as checkpoints. Everything lives in a
    {e store}, a directory of JSON and text files that {!Store_format}
    specifies, so that [jq], [tail -f] or another program reads a store as
    well as this library does. No server and no account is involved.

    A {e run} is one execution of a program. The program opens a
    {!Session}, the run's one writer, declares its metrics and logs samples
    into them, and finishes the session at the end. Readers list runs
    ({!Store.runs}), each a {!Run} as its first and last records state it,
    and fold a run's log into its {!History}: series, statistics and best
    values ({!Metric}). An {e artifact} is a name with immutable versions that
    a run publishes and other runs use ({!Artifact}). {!Value} and
    {!Provenance} are what runs record about themselves.

    A run of a training loop, then the best loss read back:
    {[
      let store = Munin.Store.open_ () in
      let id =
        Munin.Session.within
          (Munin.Session.start ~store ~experiment:"mnist" ())
        @@ fun s ->
        let loss = Munin.Session.metric s ~goal:`Minimize "train/loss" in
        for step = 0 to 999 do
          Munin.Session.log loss ~step (train_step ())
        done;
        Munin.Session.id s
      in
      let run = Option.get (Munin.Store.run store id) in
      let h = Munin.History.read run in
      Munin.History.best h "train/loss"
    ]} *)

module Store_format = Store_format
module Value = Value
module Provenance = Provenance
module Metric = Metric
module Run = Run
module History = History
module Artifact = Artifact
module Store = Store
module Session = Session
