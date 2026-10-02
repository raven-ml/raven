(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** HTML rendering for project notebooks.

    Converts notebook documents to HTML pages suitable for a static site. Text
    cells are rendered via cmarkit. Code cells include syntax-highlighted source
    and execution outputs. *)

val escape_html : string -> string
(** [escape_html s] escapes HTML special characters in [s]. *)

val notebook_output_path : Quill_project.notebook -> string
(** [notebook_output_path nb] is the output HTML path for [nb] relative to the
    project root (e.g. ["chapters/01-intro/index.html"]). *)

val chapter_html : Quill.Doc.t -> string
(** [chapter_html doc] renders [doc] to an HTML content fragment. Code cell
    outputs are rendered after their source blocks: stdout as [<pre>], images as
    [<img>], and errors with appropriate styling. *)

val toc_html :
  Quill_project.t ->
  current:Quill_project.notebook ->
  root_path:string ->
  string
(** [toc_html project ~current ~root_path] renders the sidebar table of contents
    with [current] highlighted as the active notebook. [root_path] is the
    relative path from the notebook to the project root (e.g. ["../../"]). *)

val page_html :
  book_title:string ->
  chapter_title:string ->
  toc_html:string ->
  prev:(string * string) option ->
  next:(string * string) option ->
  root_path:string ->
  content:string ->
  edit_url:string option ->
  live_reload_script:string ->
  string
(** [page_html] wraps a content fragment in the full page template with
    navigation. [prev] and [next] are [(url, title)] pairs. [root_path] is the
    relative path from the notebook to the project root (e.g. ["../../"]).
    [edit_url] is an optional URL for an "Edit this page" link.
    [live_reload_script] is empty for static builds or a [<script>] tag for live
    serve mode. *)

val strip_html_tags : string -> string
(** [strip_html_tags s] removes HTML tags from [s] and collapses whitespace. *)

val replace_all : pattern:string -> with_:string -> string -> string
(** [replace_all ~pattern ~with_ s] replaces all occurrences of [pattern] in [s]
    with [with_]. *)

val print_page_html :
  book_title:string -> chapters:(string * string) list -> string
(** [print_page_html ~book_title ~chapters] renders a print page containing all
    chapters. Each element of [chapters] is a [(title, content_html)] pair. *)

val standalone_page_html :
  title:string -> content:string -> live_reload_script:string -> string
(** [standalone_page_html ~title ~content ~live_reload_script] renders a
    self-contained HTML page for a single notebook. CSS is inlined, no sidebar
    or multi-page navigation. *)
