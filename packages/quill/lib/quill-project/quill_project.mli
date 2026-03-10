(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Notebook project model.

    A project is a collection of notebooks, optionally organized by a config
    file. Without a config file, any directory of [.md] files is a valid
    project. *)

(** {1:types Types} *)

type notebook = { title : string; path : string }
(** A notebook reference. [path] is relative to the project root. Empty for
    placeholders. *)

type toc_item =
  | Notebook of notebook * toc_item list
  | Section of string
  | Separator
      (** An entry in the table of contents. [Notebook (nb, children)] is a
          notebook with optional sub-entries. [Section title] introduces a named
          group. [Separator] is a visual break. *)

type config = {
  title : string option;
  authors : string list;
  description : string option;
  output : string option;
  edit_url : string option;
}
(** Project configuration. *)

type t = {
  title : string;  (** Project title. *)
  root : string;  (** Absolute path to the project directory. *)
  toc : toc_item list;  (** Table of contents. *)
  config : config;  (** Configuration. *)
}
(** A project. *)

(** {1:config Configuration} *)

val default_config : config
(** Default configuration with all fields empty/none. *)

val parse_config : string -> (config * toc_item list, string) result
(** [parse_config s] parses a [quill.conf] file. Returns the configuration and
    table of contents entries.

    Format: [key = value] lines for metadata, [[Section Name]] for groups,
    [Title = path] for notebooks (indentation creates nesting), [---] for
    separators, [#] for comments. *)

(** {1:titles Titles} *)

val title_of_filename : string -> string
(** [title_of_filename "01-intro.md"] is ["Intro"]. Strips the extension, strips
    leading digits and separators, replaces dashes and underscores with spaces,
    and capitalizes the first letter. *)

(** {1:queries Queries} *)

val notebooks : t -> notebook list
(** [notebooks t] is the flat, ordered list of all notebooks in [t], excluding
    placeholders. *)

val all_notebooks : toc_item list -> notebook list
(** [all_notebooks toc] flattens [toc] into an ordered list of all notebooks,
    including placeholders. *)

val is_placeholder : notebook -> bool
(** [is_placeholder nb] is [true] iff [nb] has no file (empty path). *)

val prev_notebook : t -> notebook -> notebook option
(** [prev_notebook t nb] is the notebook before [nb], or [None]. *)

val next_notebook : t -> notebook -> notebook option
(** [next_notebook t nb] is the notebook after [nb], or [None]. *)

val number : toc_item list -> notebook -> int list
(** [number toc nb] is the section number of [nb] in [toc], derived from its
    position. E.g. [[1; 2]] for the second entry in the first group. Returns
    [[]] if [nb] is not found. *)

val number_string : int list -> string
(** [number_string [1; 2]] is ["1.2"]. Returns [""] for [[]]. *)
