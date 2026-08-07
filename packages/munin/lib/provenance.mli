(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run provenance.

    Provenance records how and where a run executed: its command line, working
    directory, machine, and source revision. {!detect} captures it from the
    current process; pass a value explicitly to {!Session.start} to record
    something else. *)

type t = {
  command : string list;  (** Command line that started the run. *)
  cwd : string;  (** Working directory at run start. *)
  hostname : string option;  (** Machine hostname. *)
  pid : int;  (** Process identifier. *)
  git_commit : string option;  (** Git HEAD commit hash. *)
  git_dirty : bool option;  (** Whether the working tree was dirty. *)
  env : (string * string) list;  (** Captured environment variables. *)
}
(** The type for run provenance. Free-form run notes are not provenance; see
    {!Run.notes} and {!Session.set_notes}. *)

val detect : ?capture_env:string list -> ?cwd:string -> unit -> t
(** [detect ()] captures provenance from the current process: [command] is
    [Sys.argv], [hostname] is the machine hostname, [pid] is the process
    identifier, and the git fields are read from the repository containing
    [cwd]. The git fields are [None] outside a repository.

    - [capture_env] names the environment variables to record in [env]. Unset
      variables are skipped. Defaults to [[]], so [env] is empty.
    - [cwd] defaults to [Sys.getcwd ()]. Git detection runs from it. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] prints [p] as one [field: value] line per field. *)
