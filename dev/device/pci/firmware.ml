(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external sha256_raw : string -> string = "caml_device_pci_sha256"
external unzstd : string -> string option = "caml_device_pci_unzstd"
external unxz : string -> string option = "caml_device_pci_unxz"

let system_dir = "/lib/firmware"

let sha256 s =
  let d = sha256_raw s in
  String.concat ""
    (List.init 32 (fun i -> Printf.sprintf "%02x" (Char.code d.[i])))

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

let wrong ~digest file s =
  Error
    (Printf.sprintf "%s has SHA-256 %s, not the pinned %s" file (sha256 s)
       digest)

(* The image from /lib/firmware, where distributions ship other versions, then
   from the cache. *)
let installed name ~digest =
  let matches s = sha256 s = digest in
  match local (Filename.concat system_dir name) with
  | Some s when matches s -> Some s
  | _ -> (
      match read (Filename.concat (cache ()) name) with
      | Some s when matches s -> Some s
      | _ -> None)

let find ?dir name ~sha256:digest =
  let in_dir d =
    let file = Filename.concat d name in
    Option.map
      (fun s -> if sha256 s = digest then Ok s else wrong ~digest file s)
      (local file)
  in
  match Option.bind dir in_dir with
  | Some r -> r
  | None -> (
      match installed name ~digest with
      | Some s -> Ok s
      | None ->
          let places = Option.to_list dir @ [ system_dir; cache () ] in
          Error
            (Printf.sprintf "%s with SHA-256 %s is in none of %s" name digest
               (String.concat ", " places)))
