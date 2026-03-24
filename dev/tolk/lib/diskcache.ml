(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Simple file-based disk cache.
   Uses Marshal for serialization and individual files per key. *)

let cache_version = 1

let cache_dir =
  let base =
    match Sys.getenv_opt "XDG_CACHE_HOME" with
    | Some dir when dir <> "" -> dir
    | _ -> (
        match Sys.getenv_opt "HOME" with
        | Some home -> (
            match Sys.os_type with
            | "Unix" ->
                (* macOS uses ~/Library/Caches, Linux uses ~/.cache *)
                let macos_dir = Filename.concat home "Library/Caches" in
                if Sys.file_exists macos_dir then macos_dir
                else Filename.concat home ".cache"
            | _ -> Filename.concat home ".cache")
        | None -> Filename.current_dir_name)
  in
  Filename.concat base "tolk"

let ensure_dir dir =
  if not (Sys.file_exists dir) then begin
    (* Create parent dirs recursively *)
    let rec mkdir_p d =
      if not (Sys.file_exists d) then begin
        mkdir_p (Filename.dirname d);
        (try Unix.mkdir d 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
      end
    in
    mkdir_p dir
  end

let cache_path ~table ~key =
  let dir = Filename.concat cache_dir table in
  let hash = Digest.to_hex (Digest.string key) in
  Filename.concat dir (hash ^ ".cache")

let get ~table ~key =
  let path = cache_path ~table ~key in
  if not (Sys.file_exists path) then None
  else
    try
      let ic = open_in_bin path in
      Fun.protect
        ~finally:(fun () -> close_in ic)
        (fun () ->
          let version : int = Marshal.from_channel ic in
          if version <> cache_version then None
          else
            let value = Marshal.from_channel ic in
            Some value)
    with _ -> None

let put ~table ~key value =
  let path = cache_path ~table ~key in
  ensure_dir (Filename.dirname path);
  try
    let oc = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out oc)
      (fun () ->
        Marshal.to_channel oc cache_version [];
        Marshal.to_channel oc value [])
  with _ -> ()
