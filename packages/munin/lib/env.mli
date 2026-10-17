(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The default store root, read from the environment. *)

val root : unit -> string
(** [root ()] is [$RAVEN_TRACKING_DIR] if set, otherwise [raven/munin] under
    [$XDG_DATA_HOME], otherwise [$HOME/.local/share/raven/munin]. It reads the
    environment on each call and checks no path.

    Raises [Not_found] if none of the three variables is set. *)
