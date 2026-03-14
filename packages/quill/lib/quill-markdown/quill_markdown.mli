(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Markdown notebook format.

    Parses markdown files into {!Quill.Doc.t} and renders documents back to
    markdown. Code blocks become code cells; everything else becomes text cells.
*)

val of_string : ?base_dir:string -> string -> Quill.Doc.t
(** [of_string ?base_dir s] parses markdown string [s] into a document.

    Fenced code blocks with a language info string become code cells. All other
    content between code blocks is merged into text cells. Adjacent non-code
    content forms a single text cell.

    When [base_dir] is provided, image Display outputs that reference files on
    disk (e.g. [<img src="figures/cell-id.png">]) are resolved by reading the
    file relative to [base_dir] and base64-encoding the contents. *)

val to_string : Quill.Doc.t -> string
(** [to_string doc] renders [doc] as a markdown string.

    Text cells are emitted verbatim. Code cells are rendered as fenced code
    blocks. Cell outputs are not included. *)

val to_string_with_outputs : ?figures_dir:string -> Quill.Doc.t -> string
(** [to_string_with_outputs ?figures_dir doc] renders [doc] as markdown with
    outputs.

    Like {!to_string} but code cell outputs are serialized between
    [<!-- quill:output -->] and [<!-- /quill:output -->] comment markers after
    each code block.

    Display outputs are rendered as inline HTML:
    - Image outputs become [<img>] tags with data URIs (default) or file
      references (when [figures_dir] is set).
    - HTML outputs are emitted as inline HTML.

    When [figures_dir] is provided, images are written to disk as
    [<figures_dir>/<cell-id>.<ext>] and referenced by relative path. Orphaned
    figure files (from deleted or changed cells) are cleaned up automatically. *)

module Edit : module type of Edit
