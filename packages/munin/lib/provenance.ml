(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type git = { commit : string; dirty : bool }

type t = {
  command : string list;
  cwd : string;
  hostname : string option;
  pid : int;
  ocaml : string;
  raven : string;
  os : string;
  arch : string;
  cpu : string option;
  git : git option;
  seeds : (string * int) list;
  env : (string * string) list;
}

(* git status --porcelain=v2 --branch: "# branch.oid <commit>" names the commit,
   "(initial)" before the first; every line not starting with '#' is a changed
   tracked file, since untracked files are not listed. *)
let git cwd =
  match
    File.output
      [|
        "git";
        "-C";
        cwd;
        "status";
        "--porcelain=v2";
        "--branch";
        "--untracked-files=no";
      |]
  with
  | None -> None
  | Some out ->
      let lines = List.filter (( <> ) "") (String.split_on_char '\n' out) in
      let commit =
        List.find_map
          (fun l ->
            match String.split_on_char ' ' l with
            | [ "#"; "branch.oid"; c ] when c <> "(initial)" -> Some c
            | _ -> None)
          lines
      in
      let dirty = List.exists (fun l -> l.[0] <> '#') lines in
      Option.map (fun commit -> { commit; dirty }) commit

let raven_version = if Raven_version.v = "" then "dev" else Raven_version.v

let detect ?(seeds = []) ?(capture_env = []) ?cwd () =
  (* Names are written with invalid bytes escaped (Record), so two that escape
     alike would repeat a member of the provenance. *)
  (match File.repeat File.utf8 capture_env with
  | Some name ->
      invalid_arg
        (Printf.sprintf
           "Provenance.detect: capture_env names %S twice, once escaped" name)
  | None -> ());
  List.iter
    (fun (name, _) ->
      if not (File.is_text name) then
        invalid_arg
          (Printf.sprintf "Provenance.detect: seed name %S is not a text" name))
    seeds;
  Option.iter
    (fun name ->
      invalid_arg (Printf.sprintf "Provenance.detect: seed %S repeats" name))
    (File.repeat fst seeds);
  let cwd = match cwd with Some d -> d | None -> Sys.getcwd () in
  let os, arch = File.uname () in
  {
    command = Array.to_list Sys.argv;
    cwd;
    hostname =
      (match Unix.gethostname () with
      | "" -> None
      | h -> Some h
      | exception Unix.Unix_error _ -> None);
    pid = File.pid ();
    ocaml = Sys.ocaml_version;
    raven = raven_version;
    os;
    arch;
    cpu = File.cpu_model ();
    git = git cwd;
    seeds;
    env =
      List.filter_map
        (fun v -> Option.map (fun x -> (v, x)) (Sys.getenv_opt v))
        capture_env;
  }

let pp ppf p =
  let line k v = Format.fprintf ppf "%s: %s@\n" k v in
  let opt = Option.value ~default:"-" in
  line "command" (String.concat " " p.command);
  line "cwd" p.cwd;
  line "hostname" (opt p.hostname);
  line "pid" (string_of_int p.pid);
  line "ocaml" p.ocaml;
  line "raven" p.raven;
  line "os" p.os;
  line "arch" p.arch;
  line "cpu" (opt p.cpu);
  (match p.git with
  | None -> line "git" "-"
  | Some g -> line "git" (g.commit ^ if g.dirty then " (dirty)" else ""));
  List.iter (fun (k, v) -> line ("seed " ^ k) (string_of_int v)) p.seeds;
  List.iter (fun (k, v) -> line ("env " ^ k) v) p.env
