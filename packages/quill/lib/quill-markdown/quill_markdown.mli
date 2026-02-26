(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Markdown notebook format.

    Parses markdown files into {!Quill.Doc.t} and renders documents back to
    markdown. Code blocks become code cells; everything else becomes text cells.
*)

val of_string : string -> Quill.Doc.t
(** [of_string s] parses markdown string [s] into a document.

    Fenced code blocks with a language info string become code cells. All other
    content between code blocks is merged into text cells. Adjacent non-code
    content forms a single text cell. *)

val to_string : Quill.Doc.t -> string
(** [to_string doc] renders [doc] as a markdown string.

    Text cells are emitted verbatim. Code cells are rendered as fenced code
    blocks. Cell outputs are not included. *)

val to_string_with_outputs : Quill.Doc.t -> string
(** [to_string_with_outputs doc] renders [doc] as markdown with outputs.

    Like {!to_string} but code cell outputs are serialized between
    [<!-- quill:output -->] and [<!-- /quill:output -->] comment markers after
    each code block. *)

module Edit : module type of Edit
