(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Training run logging and monitoring.

    Kaun_board records training metrics to disk and reads them back for
    dashboards and analysis.

    {!Log} is the writer: it creates a {e run directory} and appends scalar
    events as JSONL. {!Run} and {!Store} are readers: they load run directories
    and aggregate metrics incrementally.

    A {e run directory} contains a [run.json] manifest and an append-only
    [events.jsonl] log. Run IDs follow the format [YYYY-MM-DD_HH-MM-SS_XXXX]
    with an optional experiment suffix.

    {1:modules Modules}
    {!modules:Log Event Run Store Env} *)

module Env = Env
module Event = Event
module Log = Log
module Run = Run
module Store = Store
