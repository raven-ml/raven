(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The keys: one table that dispatches them, prints the status bar and
    prints help.

    The screen's table holds a row per key, whose action means what the
    depth makes of it ({!Model.act}). Each layer over the screen has a table
    of its own. *)

type group = Move | Look | Compare | Find | Panes
(** The type for help's groups. *)

type mode = Screen | Filtering | Info | Help
(** The type for the tables: the screen's, and each layer's. *)

type depth = [ `Charts | `Strips | `Zoom ]
(** The type for the depths, as the status bar names them. *)

type place = [ depth | `Bare ]
(** The type for where the status bar shows hints: a depth, or a screen
    without panes, such as that of an empty store. *)

type row = {
  keys : (Mosaic.Shortcut.t * Model.action) list;  (** What dispatches. *)
  label : string;  (** How the keys print, as [↑↓]. *)
  alt : string;  (** How the other keys print in help, as [k j], or [""]. *)
  group : group;
  help : string;  (** What help says the keys do. *)
  hints : (place * string) list;
      (** The status bar's word for the keys at each place that shows them. *)
  on_run : bool;
      (** Whether the keys act only on the run under the cursor, so that the
          status bar leaves them out while the filter keeps no run. *)
}
(** The type for a table's rows. *)

val mode : Model.t -> mode
(** [mode m] is the table of [m]'s layer. *)

val depth : Model.t -> depth
(** [depth m] is [m]'s depth. *)

val table : mode -> row list
(** [table mode] is [mode]'s rows. *)

val bindings : mode -> (Mosaic.Shortcut.t * Model.action) list
(** [bindings mode] is what [mode]'s keys do. *)

val hints : Model.t -> (string * string) list * (string * string) option
(** [hints m] is the status bar's keys at [m], each label with its word,
    without the keys that act on a run when no run is under the cursor: the
    hints in the order the status bar drops them, from the last, and the
    help key's, which it never drops. *)

val group_name : group -> string
(** [group_name g] is [g]'s title in help. *)
