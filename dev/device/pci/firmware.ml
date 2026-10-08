(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let digest s = Digest.BLAKE256.to_hex (Digest.BLAKE256.string s)

(* The contents of [file], [Ok None] if it does not exist, or [Error why] if it
   cannot be read, [why] the system's "FILE: cause". *)
let read file =
  if not (Sys.file_exists file) then Ok None
  else
    try Ok (Some (In_channel.with_open_bin file In_channel.input_all))
    with Sys_error why -> Error why

let missing dirs name ~digest others unreadable =
  let others =
    match others with
    | [] -> ""
    | l -> "; another digest in " ^ String.concat ", " (List.rev l)
  in
  let unreadable =
    String.concat "" (List.rev_map (fun why -> "; reading " ^ why) unreadable)
  in
  strf "%s with digest %s is in none of [%s]%s%s" name digest
    (String.concat ", " dirs) others unreadable

let find dirs name ~digest:pinned =
  let rec go others unreadable = function
    | [] -> Error (missing dirs name ~digest:pinned others unreadable)
    | dir :: rest -> (
        let file = Filename.concat dir name in
        match read file with
        | Ok (Some s) when digest s = pinned -> Ok s
        | Ok (Some _) -> go (file :: others) unreadable rest
        | Ok None -> go others unreadable rest
        | Error why -> go others (why :: unreadable) rest)
  in
  go [] [] dirs
