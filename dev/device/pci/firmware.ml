(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let digest s = Digest.BLAKE256.to_hex (Digest.BLAKE256.string s)

let read file =
  try Some (In_channel.with_open_bin file In_channel.input_all)
  with Sys_error _ -> None

let missing dirs name ~digest others =
  let others =
    match others with
    | [] -> ""
    | l -> "; another digest in " ^ String.concat ", " (List.rev l)
  in
  Printf.sprintf "%s with digest %s is in none of [%s]%s" name digest
    (String.concat ", " dirs) others

let find dirs name ~digest:pinned =
  let rec go others = function
    | [] -> Error (missing dirs name ~digest:pinned others)
    | dir :: rest -> (
        let file = Filename.concat dir name in
        match read file with
        | Some s when digest s = pinned -> Ok s
        | Some _ -> go (file :: others) rest
        | None -> go others rest)
  in
  go [] dirs
