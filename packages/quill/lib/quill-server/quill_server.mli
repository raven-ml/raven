(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Web notebook server.

    [Quill_server] serves a Jupyter-like notebook interface over HTTP and
    WebSocket. *)

val serve :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  ?addr:string ->
  ?port:int ->
  ?on_ready:(unit -> unit) ->
  string ->
  unit
(** [serve ~create_kernel path] starts the web notebook server for the notebook
    at [path]. [create_kernel] is called once to obtain a kernel. [addr]
    defaults to ["127.0.0.1"], [port] to [8888]. [on_ready] is called after the
    server socket is bound and listening, before the accept loop starts. Blocks
    until the server is stopped. Exits the process with status [1] if [path]
    does not exist. *)

(** {1:dir Directory mode} *)

val serve_dir :
  create_kernel:(on_event:(Quill.Kernel.event -> unit) -> Quill.Kernel.t) ->
  ?addr:string ->
  ?port:int ->
  ?on_ready:(unit -> unit) ->
  ?prelude:(string -> string option) ->
  toc:Quill_project.toc_item list ->
  string ->
  unit
(** [serve_dir ~create_kernel ~toc root] starts the web notebook server for a
    directory of notebooks at [root]. [toc] defines the table of contents
    structure shown in the sidebar. Each notebook gets its own kernel, created
    lazily on first access. [prelude] is called with the notebook's relative
    path and may return OCaml code to execute before the notebook's cells.
    [addr] defaults to ["127.0.0.1"], [port] to [8888]. *)
