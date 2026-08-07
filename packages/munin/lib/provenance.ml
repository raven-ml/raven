(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  command : string list;
  cwd : string;
  hostname : string option;
  pid : int;
  git_commit : string option;
  git_dirty : bool option;
  env : (string * string) list;
}

let git_output cwd args =
  let command =
    String.concat " " (List.map Filename.quote ("git" :: "-C" :: cwd :: args))
  in
  Fs.command_output command

let detect_git_commit cwd = git_output cwd [ "rev-parse"; "HEAD" ]

let detect_git_dirty cwd =
  match git_output cwd [ "status"; "--porcelain"; "--untracked-files=no" ] with
  | None -> None
  | Some output -> Some (output <> "")

let detect ?(capture_env = []) ?cwd () =
  let cwd = Option.value cwd ~default:(Sys.getcwd ()) in
  {
    command = Array.to_list Sys.argv;
    cwd;
    hostname = Some (Unix.gethostname ());
    pid = Unix.getpid ();
    git_commit = detect_git_commit cwd;
    git_dirty = detect_git_dirty cwd;
    env =
      List.filter_map
        (fun name ->
          Option.map (fun value -> (name, value)) (Sys.getenv_opt name))
        capture_env;
  }

let pp_unknown ppf = function
  | None -> Format.pp_print_string ppf "-"
  | Some text -> Format.pp_print_string ppf text

let pp ppf t =
  Format.fprintf ppf
    "@[<v>command: %s@,\
     cwd: %s@,\
     hostname: %a@,\
     pid: %d@,\
     git_commit: %a@,\
     git_dirty: %a@,\
     env: %s@]"
    (String.concat " " t.command)
    t.cwd pp_unknown t.hostname t.pid pp_unknown t.git_commit pp_unknown
    (Option.map string_of_bool t.git_dirty)
    (String.concat " " (List.map (fun (k, v) -> k ^ "=" ^ v) t.env))
