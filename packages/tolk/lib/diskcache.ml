(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Simple file-based disk cache.
   Uses Marshal for serialization and individual files per key. *)

(* Bump whenever a change makes previously cached entries wrong.

   The obvious case is when an entry can no longer be replayed — an
   optimisation a beam search could once select stops being legal, so its
   cached opt list now fails. The case that is easy to miss: a change that
   alters what a serialised key component *means* without altering how it is
   spelled. The key still matches, so the entry is still found, and it is now
   an answer to a different question. Renaming a variant and reusing the old
   name for a new concept does exactly this. Bump for those too. *)
let cache_version = 19

let cache_dir =
  let base =
    match Sys.getenv_opt "XDG_CACHE_HOME" with
    | Some dir when dir <> "" -> dir
    | _ -> (
        match Sys.getenv_opt "HOME" with
        | Some home ->
            Filename.concat home
              (if Host_config.system = "macosx" then "Library/Caches" else ".cache")
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

(* Write-then-rename keeps entries atomic: a concurrent reader or writer of
   the same key never observes a partially written file, only the previous
   complete entry or the new one. The temporary is created exclusively under
   a random name next to the entry, so concurrent writers — in different
   processes or in different domains of the same one — cannot clobber each
   other's in-progress writes. (A process id would not do: on Windows
   [Unix.getpid] is a handle value that sibling processes routinely share.) *)
let put ~table ~key value =
  let path = cache_path ~table ~key in
  let dir = Filename.dirname path in
  ensure_dir dir;
  match
    Filename.open_temp_file ~mode:[ Open_binary ] ~temp_dir:dir
      (Filename.basename path ^ ".")
      ".tmp"
  with
  | exception Sys_error _ -> ()
  | tmp, oc -> (
      let remove_tmp () = try Sys.remove tmp with Sys_error _ -> () in
      try
        Fun.protect
          ~finally:(fun () -> close_out oc)
          (fun () ->
            Marshal.to_channel oc cache_version [];
            Marshal.to_channel oc value []);
        (* A rename that loses (on Windows, to a reader or writer holding the
           entry open) leaves the previous complete entry in place; the
           temporary must not outlive it. *)
        try Unix.rename tmp path with Unix.Unix_error _ -> remove_tmp ()
      with _ -> remove_tmp ())
