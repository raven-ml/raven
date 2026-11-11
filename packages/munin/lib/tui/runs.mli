(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Which runs the runs pane lists, in which order, under which headers, and
    where its cursor stops. *)

(** {1:filters Filters} *)

type filter
(** The type for filters: the words typed after [/]. *)

val no_filter : filter
(** [no_filter] keeps every run. *)

val filter : string -> filter
(** [filter text] keeps the runs of which each word of [text] matches, ignoring
    case: [key<v] and [key>v], with [v] a number in JSON's syntax, a param
    [key] that is a number below or above [v], any other word a substring of
    the name, the id, the experiment, a tag, the state word or a param
    written [key=value]. *)

val text : filter -> string
(** [text f] is the words [f] was made from. *)

(** {1:order Order} *)

type order = Started | Best | Name
(** The type for orders: newest first, by the best value of the experiment's
    key metric, or by name. *)

val next : order -> order
(** [next o] is the order after [o]: started, best, name, started. *)

val word : order -> string
(** [word o] is [started], [best] or [name]. *)

(** {1:runs Runs and lines} *)

val keep :
  filter -> order -> keys:(string -> string option) -> Facts.run list ->
  Facts.run list
(** [keep f o ~keys runs] is the runs of [runs] that [f] keeps, in order [o],
    each experiment's together in the order of its first run; [keys e] is the
    key metric of the experiment [e]. *)

module Sset : Set.S with type elt = string

type folds = Unset | Folded of Sset.t
(** The type for the experiments folded: [Unset] until the first listing. *)

val first_folds : Facts.run list -> cursor:string option -> Sset.t
(** [first_folds runs ~cursor] is the experiments that start folded: every one
    but the cursor's, once there are more than 30 runs. *)

type line =
  | Header of { experiment : string; runs : Facts.run list; folded : bool }
  | Run of Facts.run
(** The type for the pane's lines: a header per experiment, and a line per
    run of an open experiment. *)

val lines : Facts.run list -> Sset.t -> line list
(** [lines kept folded] is the lines of the runs [kept]. *)

val at : line list -> string -> int option
(** [at lines id] is the index of the line that stands for the run [id]: its
    own, or its experiment's header when that is folded. *)

val step : line list -> string -> int -> string option
(** [step lines id d] is the run [d] stops from the run [id]'s line, clamped
    to the first and last: a cursor stops at each run of an open experiment
    and at each folded header, which stands for its first run. *)

val first : line list -> string option
(** [first lines] is the run of the first stop. *)

val last : line list -> string option
(** [last lines] is the run of the last stop. *)

val star : Facts.run list -> key:string -> string option
(** [star runs ~key] is the run of [runs] with the best value of [key] by its
    goal, the first among equals. *)
