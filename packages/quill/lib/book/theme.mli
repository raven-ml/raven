(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The default book theme: its stylesheet and HTML page templates.

    A template is an HTML page with placeholders written [{{name}}], which
    {!Render} replaces. Every page loads KaTeX from [cdn.jsdelivr.net] to
    typeset the TeX math it holds, and keeps the reader's choice of light or
    dark theme in the browser's [localStorage]. *)

val style_css : string
(** [style_css] is the stylesheet of every page, which {!Build} writes as
    [style.css] at the root of a site. *)

val template_html : string
(** [template_html] is the page of a chapter, with a sidebar, search and
    page navigation. Its placeholders are [book_title], [chapter_title],
    [root_path], [toc], [edit_link], [content], [page_toc], [prev_nav],
    [next_nav] and [live_reload_script]. It links [style.css], [index.html],
    [print.html] and [searchindex.json] under [root_path]. *)

val print_template_html : string
(** [print_template_html] is the page holding every chapter, for printing.
    Its placeholders are [book_title] and [chapters]. It links [style.css]
    next to it. *)

val standalone_html : string
(** [standalone_html] is the page of a single notebook outside a site, with
    {!style_css} inlined. Its placeholders are [title], [content],
    [page_toc] and [live_reload_script]. *)
