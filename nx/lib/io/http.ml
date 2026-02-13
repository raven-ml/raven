(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let curl_available = lazy (Unix.system "command -v curl >/dev/null 2>&1" = Unix.WEXITED 0)

let check_curl () =
  if not (Lazy.force curl_available) then
    failwith "curl not found on PATH. Install curl to download files."

let rec mkdir_p path =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else if not (Sys.file_exists path) then (
    mkdir_p (Filename.dirname path);
    try Unix.mkdir path 0o755
    with Unix.Unix_error (Unix.EEXIST, _, _) -> ())

let header_flags headers =
  headers
  |> List.map (fun (k, v) ->
         Printf.sprintf "-H %s" (Filename.quote (k ^ ": " ^ v)))
  |> String.concat " "

let download ?(show_progress = false) ?(headers = []) ~url ~dest () =
  check_curl ();
  mkdir_p (Filename.dirname dest);
  let silent = if show_progress then "" else "-s" in
  let hdr = header_flags headers in
  let cmd =
    Printf.sprintf "curl -L --fail %s %s -o %s %s" silent hdr
      (Filename.quote dest) (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> ()
  | _ ->
      (try Sys.remove dest with Sys_error _ -> ());
      failwith (Printf.sprintf "Failed to download %s" url)

let get ?(headers = []) url =
  check_curl ();
  let hdr = header_flags headers in
  let cmd =
    Printf.sprintf "curl -s -L --fail %s %s" hdr (Filename.quote url)
  in
  let ic = Unix.open_process_in cmd in
  let buf = Buffer.create 4096 in
  (try
     let tmp = Bytes.create 4096 in
     let rec loop () =
       let n = input ic tmp 0 4096 in
       if n > 0 then (
         Buffer.add_subbytes buf tmp 0 n;
         loop ())
     in
     loop ()
   with End_of_file -> ());
  match Unix.close_process_in ic with
  | Unix.WEXITED 0 -> Buffer.contents buf
  | _ -> failwith (Printf.sprintf "Failed to fetch %s" url)
