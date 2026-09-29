(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external sha256_raw : string -> string = "caml_nx_sha256"
external unzstd : string -> string option = "caml_nx_unzstd"
external unxz : string -> string option = "caml_nx_unxz"
external download : string -> (string, string) result = "caml_nx_download"

let sha256 s =
  String.concat ""
    (List.init 32 (fun i ->
         Printf.sprintf "%02x" (Char.code (sha256_raw s).[i])))

let cache () =
  match Sys.getenv_opt "RAVEN_CACHE_ROOT" with
  | Some root when root <> "" -> Filename.concat root "firmware"
  | _ ->
      let base =
        match Sys.getenv_opt "XDG_CACHE_HOME" with
        | Some d when d <> "" -> d
        | _ ->
            Filename.concat
              (Option.value ~default:"." (Sys.getenv_opt "HOME"))
              ".cache"
      in
      Filename.concat (Filename.concat base "raven") "firmware"

let read file =
  try Some (In_channel.with_open_bin file In_channel.input_all)
  with Sys_error _ -> None

(* [file] plain, or compressed as the system can decompress it. *)
let local file =
  let decompressed ext f =
    Option.bind (read (file ^ ext)) (fun s -> try f s with Failure _ -> None)
  in
  match read file with
  | Some _ as s -> s
  | None -> (
      match decompressed ".zst" unzstd with
      | Some _ as s -> s
      | None -> decompressed ".xz" unxz)

let rec mkdir_p d =
  if not (Sys.file_exists d) then begin
    mkdir_p (Filename.dirname d);
    try Sys.mkdir d 0o755 with Sys_error _ when Sys.file_exists d -> ()
  end

(* Written to a temporary file then renamed, so a reader never sees half. *)
let keep file s =
  mkdir_p (Filename.dirname file);
  let tmp = Printf.sprintf "%s.%d.tmp" file (Unix.getpid ()) in
  Out_channel.with_open_bin tmp (fun oc -> output_string oc s);
  Sys.rename tmp file

let get ?dir ~url name ~sha256:digest =
  let matches s = sha256 s = digest in
  let wrong file s =
    Error
      (Printf.sprintf "%s has SHA-256 %s, not the pinned %s" file (sha256 s)
         digest)
  in
  let from_dir () =
    match dir with
    | None -> None
    | Some d -> (
        let file = Filename.concat d name in
        match local file with
        | Some s when matches s -> Some (Ok s)
        | Some s -> Some (wrong file s)
        | None -> None)
  in
  let from_system () =
    match local (Filename.concat "/lib/firmware" name) with
    | Some s when matches s -> Some (Ok s)
    | _ -> None
  in
  let cached = Filename.concat (cache ()) name in
  let from_cache () =
    match read cached with Some s when matches s -> Some (Ok s) | _ -> None
  in
  let fetch () =
    match download (url ^ name) with
    | Error why -> Error (Printf.sprintf "downloading %s%s: %s" url name why)
    | Ok s when matches s ->
        (* The cache only saves the next download. *)
        (try keep cached s with Sys_error _ -> ());
        Ok s
    | Ok s -> wrong (url ^ name) s
  in
  match
    List.find_map
      (fun source -> source ())
      [ from_dir; from_system; from_cache ]
  with
  | Some r -> r
  | None -> fetch ()
