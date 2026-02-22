(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quill -- notebook core library.

    Quill provides the foundational types and protocol for building notebook
    applications. It is frontend-agnostic: web, TUI, and desktop frontends can
    all be built on the core.

    {1 Overview}

    A notebook is a {!Doc.t} containing an ordered sequence of {!Cell.t} values.
    Each cell is either text or executable code with outputs.

    Code execution is handled by a {!Kernel.t}, an abstract interface that
    supports different backends (OCaml toplevel, subprocess, remote).

    A {!Session.t} manages the document and kernel together, processing
    {!Session.request} values from frontends and producing
    {!Session.notification} values.

    For batch evaluation of notebooks without an interactive session, see
    {!Eval}.

    {1 Modules} *)

module Cell = Cell
module Doc = Doc
module Kernel = Kernel
module Eval = Eval
module Session = Session
