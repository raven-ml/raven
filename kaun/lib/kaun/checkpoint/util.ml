(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let rec mkdir_p path =
  if path = "" || path = "." then ()
  else if Sys.file_exists path then
    if Sys.is_directory path then ()
    else
      invalid_arg (Printf.sprintf "Path %s exists and is not a directory" path)
  else
    let parent = Filename.dirname path in
    if parent <> path then mkdir_p parent;
    Unix.mkdir path 0o755

let rec remove_tree path =
  if Sys.file_exists path then
    if Sys.is_directory path then (
      Sys.readdir path
      |> Array.iter (fun entry -> remove_tree (Filename.concat path entry));
      Unix.rmdir path)
    else Sys.remove path

let encoded_prefix = "kaun:"

let encode_path path =
  let buf =
    Buffer.create (String.length encoded_prefix + (String.length path * 2))
  in
  Buffer.add_string buf encoded_prefix;
  String.iter
    (fun ch -> Buffer.add_string buf (Printf.sprintf "%02x" (Char.code ch)))
    path;
  Buffer.contents buf

let decode_path encoded =
  let prefix_len = String.length encoded_prefix in
  let is_hex c =
    match c with '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true | _ -> false
  in
  let len = String.length encoded in
  if
    len < prefix_len
    || not (String.equal (String.sub encoded 0 prefix_len) encoded_prefix)
  then encoded
  else
    let hex = String.sub encoded prefix_len (len - prefix_len) in
    if hex = "" || String.length hex mod 2 <> 0 then encoded
    else
      let valid =
        let rec loop idx =
          if idx >= String.length hex then true
          else if is_hex hex.[idx] then loop (idx + 1)
          else false
        in
        loop 0
      in
      if not valid then encoded
      else
        let buf = Buffer.create (String.length hex / 2) in
        let rec loop idx =
          if idx < String.length hex then (
            let byte = int_of_string ("0x" ^ String.sub hex idx 2) in
            Buffer.add_char buf (Char.chr byte);
            loop (idx + 2))
        in
        loop 0;
        Buffer.contents buf

let slugify label =
  let buf = Buffer.create (String.length label) in
  let push_dash = ref false in
  String.iter
    (fun ch ->
      let push_dash_if_needed () =
        if not !push_dash then Buffer.add_char buf '-';
        push_dash := true
      in
      match ch with
      | 'a' .. 'z' | '0' .. '9' ->
          Buffer.add_char buf ch;
          push_dash := true
      | 'A' .. 'Z' ->
          Buffer.add_char buf (Char.lowercase_ascii ch);
          push_dash := true
      | ' ' | '-' | ':' | '.' | '/' | '\\' -> push_dash_if_needed ()
      | _ ->
          (* Percent-encode unsupported characters *)
          Buffer.add_string buf (Printf.sprintf "%02x" (Char.code ch));
          push_dash := true)
    label;
  let result = Buffer.contents buf in
  if result = "" then "artifact" else result

let now_unix () = Unix.time ()
