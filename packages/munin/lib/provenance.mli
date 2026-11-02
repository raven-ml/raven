(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** How and where a run started.

    Provenance records what is known to change a run's results and what a
    person needs to run it again: the command, the code, the toolchain,
    the machine and the seeds. A run's start records it, and so does each
    resume, since a resumed run may run other code
    ({!Store_format.section-log}). {!detect} reads it from the current process;
    the seeds are the one part that only the program knows, and it names
    them. *)

type git = {
  commit : string;  (** The commit checked out, in full hexadecimal. *)
  dirty : bool;
      (** [true] iff a tracked file differs from [commit]. Untracked files
          do not count. *)
}
(** The type for the state of a git work tree. *)

type t = {
  command : string list;  (** The command line, [Sys.argv]. *)
  cwd : string;  (** The working directory. *)
  hostname : string option;  (** The machine's host name, if it has one. *)
  pid : int;  (** The process's id, as the system names it. *)
  ocaml : string;  (** The OCaml compiler's version, [Sys.ocaml_version]. *)
  raven : string;  (** The version of raven that built the program. *)
  os : string;
      (** The system's name and release, such as ["Linux 6.12.1"] or
          ["Darwin 25.3.0"]. *)
  arch : string;  (** The processor architecture, such as ["x86_64"]. *)
  cpu : string option;  (** The processor model, if the system names it. *)
  git : git option;  (** The work tree of [cwd], if [cwd] is in one. *)
  seeds : (string * int) list;
      (** The seeds of the program's random number generators, by name. *)
  env : (string * string) list;
      (** The environment variables the program asked to record. *)
}
(** The type for provenance. *)

val detect :
  ?seeds:(string * int) list -> ?capture_env:string list -> ?cwd:string ->
  unit -> t
(** [detect ()] is the current process's provenance. It runs
    [git status --porcelain=v2 --branch --untracked-files=no] once in
    [cwd], without a shell; [git] is [None] when that fails, as outside a
    work tree or without git.

    - [seeds] defaults to [[]].
    - [capture_env] names the environment variables to record. Unset ones
      are skipped. Defaults to [[]].
    - [cwd] defaults to [Sys.getcwd ()].

    Raises [Invalid_argument] if a seed's name is not a text
    ({!Store_format.section-names}) or repeats, or if two names of
    [capture_env] are equal once escaped as UTF-8
    ({!Store_format.section-values}). *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] prints [p] as one [field: value] line per field. *)
