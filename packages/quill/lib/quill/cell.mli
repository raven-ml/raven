(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Notebook cells.

    A cell is the atomic unit of a notebook: either a block of text or an
    executable code block with outputs. *)

(** {1:ids Cell identifiers} *)

type id = string
(** The type for cell identifiers. Stable across serialization. *)

val fresh_id : unit -> id
(** [fresh_id ()] is a fresh unique identifier. *)

(** {1:outputs Execution outputs} *)

type output =
  | Stdout of string
  | Stderr of string
  | Error of string
  | Display of { mime : string; data : string }
      (** The type for cell execution outputs. A single execution may produce
          multiple outputs (e.g. stdout text followed by a displayed image).

          - [Stdout s] is captured standard output.
          - [Stderr s] is captured standard error.
          - [Error s] is an execution error message.
          - [Display {mime; data}] is rich content identified by MIME type (e.g.
            ["text/html"], ["image/png"]). Binary data is base64-encoded in
            [data]. *)

type Format.stag +=
  | Display_tag of { mime : string; data : string }
        (** Semantic tag for rich display output. When a pretty-printer opens
            this tag on a formatter configured by the notebook kernel, the
            payload is emitted as a {!Display} output. On other formatters the
            tag is silently ignored and only the text content between the
            open/close tags is printed. *)

(** {1:attrs Cell attributes} *)

type attrs = {
  collapsed : bool;
      (** When [true], the cell renders as a compact one-line bar. Source and
          outputs are hidden until the user expands the cell. Purely
          presentational — execution is unaffected. *)
  hide_source : bool;
      (** When [true], the code editor is folded away and only outputs are
          visible. Clicking the placeholder reveals the source. Only meaningful
          for code cells. Purely presentational — execution is unaffected. *)
}
(** The type for cell display attributes. All attributes are purely
    presentational and never affect execution semantics. *)

val default_attrs : attrs
(** [default_attrs] is [{collapsed = false; hide_source = false}]. *)

(** {1:cells Cells} *)

type t = private
  | Code of {
      id : id;
      source : string;
      language : string;
      outputs : output list;
      execution_count : int;
      attrs : attrs;
    }
  | Text of { id : id; source : string; attrs : attrs }
      (** The type for notebook cells.

          - [Code] is an executable code cell. [language] identifies the kernel
            (e.g. ["ocaml"]). [execution_count] tracks how many times this cell
            has been executed (starts at [0]).
          - [Text] is a text cell whose [source] is markdown.

          The type is private: pattern matching is allowed, but cells must be
          constructed via {!code} and {!text}. *)

(** {1:constructors Constructors} *)

val code : ?id:id -> ?language:string -> ?attrs:attrs -> string -> t
(** [code ?id ?language ?attrs source] is a code cell with the given [source].
    [language] defaults to ["ocaml"]. [attrs] defaults to {!default_attrs}. A
    fresh identifier is generated when [id] is not provided. *)

val text : ?id:id -> ?attrs:attrs -> string -> t
(** [text ?id ?attrs source] is a text cell with the given [source]. [attrs]
    defaults to {!default_attrs}. A fresh identifier is generated when [id] is
    not provided. *)

(** {1:accessors Accessors} *)

val id : t -> id
(** [id c] is the unique identifier of cell [c]. *)

val source : t -> string
(** [source c] is the source text of cell [c]. *)

val attrs : t -> attrs
(** [attrs c] is the display attributes of cell [c]. *)

(** {1:transformations Transformations} *)

val set_source : string -> t -> t
(** [set_source s c] is [c] with source replaced by [s]. *)

val set_attrs : attrs -> t -> t
(** [set_attrs a c] is [c] with display attributes replaced by [a]. *)

val set_outputs : output list -> t -> t
(** [set_outputs os c] is [c] with outputs replaced by [os]. Text cells are
    returned unchanged. *)

val append_output : output -> t -> t
(** [append_output o c] appends [o] to the outputs of [c]. Text cells are
    returned unchanged. *)

val clear_outputs : t -> t
(** [clear_outputs c] is [c] with an empty output list. Text cells are returned
    unchanged. *)

val increment_execution_count : t -> t
(** [increment_execution_count c] increments the execution count of a code cell.
    Text cells are returned unchanged. *)
