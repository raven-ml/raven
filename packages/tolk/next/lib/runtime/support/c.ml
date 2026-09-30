(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let is_file p = Sys.file_exists p && not (Sys.is_directory p)

let is_symlink p =
  try (Unix.lstat p).st_kind = Unix.S_LNK with Unix.Unix_error _ -> false

let env_dirs var =
  match Sys.getenv_opt var with
  | None -> []
  | Some s ->
      String.split_on_char (if Sys.win32 then ';' else ':') s
      |> List.filter (( <> ) "")

let macos = Host_config.system = "macosx"

(* Debian's name of the host, as Python's sysconfig reports it. *)
let multiarch =
  let machine =
    match Host_config.architecture with
    | "amd64" -> "x86_64"
    | "arm64" -> "aarch64"
    | "riscv" -> "riscv64"
    | a -> a
  in
  machine ^ "-linux-gnu"

let is_elf path =
  try
    In_channel.with_open_bin path (fun ic ->
        In_channel.really_input_string ic 4 = Some "\x7fELF")
  with Sys_error _ -> false

(* [libp.so] and its versions: [libp.so] followed by digits and dots. *)
let is_versioned_so p f =
  let base = "lib" ^ p ^ ".so" in
  String.starts_with ~prefix:base f
  && String.for_all
       (fun c -> c = '.' || (c >= '0' && c <= '9'))
       (String.sub f (String.length base)
          (String.length f - String.length base))

let library_in p dir =
  if Sys.win32 then
    let l = Filename.concat dir (p ^ ".dll") in
    if is_file l then Some l else None
  else if macos then
    List.find_map
      (fun base ->
        let l = Filename.concat dir base in
        if is_file l || (Filename.check_suffix dir ".framework" && is_symlink l)
        then Some l
        else None)
      [ "lib" ^ p ^ ".dylib"; p ^ ".dylib"; p ]
  else
    let files = Sys.readdir dir in
    Array.sort String.compare files;
    Array.to_list files
    |> List.find_map (fun f ->
        let l = Filename.concat dir f in
        if is_versioned_so p f && is_file l && is_elf l then Some l else None)

let findlib ?(extra_paths = []) name paths =
  let var =
    String.map (function '-' -> '_' | c -> c) (String.uppercase_ascii name)
    ^ "_PATH"
  in
  let path = Option.value (Sys.getenv_opt var) ~default:"" in
  let dirs p =
    let posix =
      env_dirs "LD_LIBRARY_PATH"
      @ [ "/usr/lib64"; "/usr/lib"; "/usr/local/lib" ]
    in
    let system =
      if Sys.win32 then env_dirs "PATH"
      else if macos then
        posix
        @ [
            "/opt/homebrew/lib";
            "/System/Library/Frameworks/" ^ p ^ ".framework";
            "/System/Library/PrivateFrameworks/" ^ p ^ ".framework";
          ]
      else if Host_config.system = "linux" then
        posix @ [ "/usr/lib/wsl/lib/"; "/lib"; "/lib64"; "/lib/" ^ multiarch ]
      else posix
    in
    (if path = "" then [] else [ path ]) @ system @ extra_paths
  in
  if is_file path then Some path
  else
    List.find_map
      (fun p ->
        if not (Filename.is_relative p) then if is_file p then Some p else None
        else
          List.find_map
            (fun dir ->
              if Sys.file_exists dir && Sys.is_directory dir then
                library_in p dir
              else None)
            (dirs p))
      paths
