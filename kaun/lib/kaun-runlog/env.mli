(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Environment configuration for run storage.

    This module resolves the default base directory for storing training runs
    using XDG conventions. *)

val base_dir : unit -> string
(** [base_dir ()] returns the default directory for storing runs.

    Resolution order:
    + [RAVEN_RUNS_DIR] environment variable if set
    + [XDG_CACHE_HOME/raven/runs] if [XDG_CACHE_HOME] is set
    + [~/.cache/raven/runs] otherwise

    @raise Not_found if [HOME] is unset and fallback is needed. *)
