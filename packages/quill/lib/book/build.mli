(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Project build pipeline.

    Executes code cells in each notebook, renders to HTML, and writes the output
    as a static site. *)

val build :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  ?skip_eval:bool ->
  ?output:string ->
  ?live_reload_script:string ->
  Quill_project.t ->
  unit
(** [build ~create_kernel project] builds the project. For each notebook: reads
    the markdown, executes code cells via {!Quill.Eval.run} (unless [skip_eval]
    is [true]), renders to HTML, and copies assets to the output directory.
    Source files are never modified.

    [output] defaults to [build/] inside the project root. [live_reload_script]
    defaults to [""] (empty). *)

val build_file :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  ?skip_eval:bool ->
  ?output:string ->
  ?live_reload_script:string ->
  string ->
  unit
(** [build_file ~create_kernel path] builds a single notebook file to a
    self-contained HTML page. [output] is the output directory (defaults to the
    directory containing the source file). [live_reload_script] defaults to [""]
    (empty). *)
