(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A stand-in for curl that answers from a directory, [$FAKE_CURL_DIR], with
   no network. It takes the options tessera's requests pass, refuses any
   other, and writes what [curl -i] would: the status line, the headers, a
   blank line and the body.

   - A URL whose host and path name a file under [files/] reads that file as
     a server of byte ranges does: [Range: bytes=a-b] is 206 with
     [Content-Range], or 416 past the end, and no range is 200. Its ETag is
     the line of [<file>.etag] for the request's number, the last line once
     they run out, and ["0x1"] without the file; an [If-Match] of another
     ETag is 412, Azure Blob Storage's answer. The first [n] requests, [n]
     the number in [<file>.deny], are 403. With [<file>.lie], each range
     request after the first is answered wrongly, as the file names: [whole]
     (200 of the file), [short] (a byte short), [span] (the next byte's
     span), [length] (another file length) or [past] (416).
   - Any other URL takes the first answer of [script/], in name order, and
     removes it. An answer [curl-exit N] exits with code N instead.

   Each request is a line of [log]: the method and URL, then each header and
   the body, separated by tabs. *)

let dir = Sys.getenv "FAKE_CURL_DIR"
let path n = Filename.concat dir n
let read f = In_channel.with_open_bin f In_channel.input_all

let fail fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("fake curl: " ^ s);
      exit 2)
    fmt

(* Options *)

let flags = [ "-sS"; "-i"; "--suppress-connect-headers" ]
let valued = [ "--connect-timeout"; "--speed-limit"; "--speed-time" ]

(* As curl's [--proto =https]: any other scheme is refused, exit 1. *)
let https url =
  if not (String.starts_with ~prefix:"https://" url) then (
    prerr_endline ("fake curl: protocol refused: " ^ url);
    exit 1)

let headers, body, url =
  let rec go hs body url = function
    | [] -> (List.rev hs, body, url)
    | f :: rest when List.mem f flags -> go hs body url rest
    | f :: _ :: rest when List.mem f valued -> go hs body url rest
    | "--proto" :: "=https" :: rest -> go hs body url rest
    | "-H" :: h :: rest -> go (h :: hs) body url rest
    | "--data-raw" :: b :: rest -> go hs (Some b) url rest
    | "--url" :: u :: rest when url = None -> go hs body (Some u) rest
    | a :: _ -> fail "unknown argument %S" a
  in
  match go [] None None (List.tl (Array.to_list Sys.argv)) with
  | hs, body, Some url -> (hs, body, url)
  | _, _, None -> fail "no URL"

let () = https url

let header name =
  let prefix = String.lowercase_ascii name ^ ":" in
  List.find_map
    (fun h ->
      let l = String.lowercase_ascii h in
      if String.starts_with ~prefix l then
        let n = String.length prefix in
        Some (String.trim (String.sub h n (String.length h - n)))
      else None)
    headers

let () =
  let meth = if body = None then "GET" else "POST" in
  let line =
    String.concat "\t" (((meth ^ " " ^ url) :: headers) @ Option.to_list body)
  in
  Out_channel.with_open_gen
    [ Open_append; Open_creat; Open_binary ]
    0o644 (path "log")
    (fun oc -> output_string oc (line ^ "\n"))

(* Answers *)

let answer status reason hs body =
  set_binary_mode_out stdout true;
  Printf.printf "HTTP/1.1 %d %s\r\n" status reason;
  List.iter (fun (k, v) -> Printf.printf "%s: %s\r\n" k v) hs;
  Printf.printf "Content-Length: %d\r\n\r\n%s" (String.length body) body;
  exit 0

(* The ETag of request [k] of [file], counted from 0. *)
let etag file k =
  let f = file ^ ".etag" in
  if not (Sys.file_exists f) then "\"0x1\""
  else
    let lines = List.filter (( <> ) "") (String.split_on_char '\n' (read f)) in
    List.nth lines (min k (List.length lines - 1))

let count file =
  let f = file ^ ".count" in
  let k = if Sys.file_exists f then int_of_string (read f) else 0 in
  Out_channel.with_open_bin f (fun oc ->
      output_string oc (string_of_int (k + 1)));
  k

let serve file =
  let data = read file and k = count file in
  let len = String.length data in
  let tag = etag file k in
  let tagged = [ ("ETag", tag); ("Accept-Ranges", "bytes") ] in
  let deny = file ^ ".deny" in
  if Sys.file_exists deny && k < int_of_string (String.trim (read deny)) then
    answer 403 "Server failed to authenticate the request." [] "";
  (match header "If-Match" with
  | Some m when m <> tag ->
      answer 412
        "The condition specified using HTTP conditional header(s) is not met."
        [ ("x-ms-error-code", "ConditionNotMet") ]
        ""
  | _ -> ());
  match header "Range" with
  | None -> answer 200 "OK" tagged data
  | Some r -> (
      match Scanf.sscanf_opt r "bytes=%d-%d%!" (fun a b -> (a, b)) with
      | None -> fail "range %S" r
      | Some (a, _) when a >= len ->
          answer 416 "The range specified is invalid for the current size."
            [ ("Content-Range", Printf.sprintf "bytes */%d" len) ]
            ""
      | Some (a, b) -> (
          let b = min b (len - 1) in
          let part a b n =
            let cr = Printf.sprintf "bytes %d-%d/%d" a b n in
            answer 206 "Partial Content"
              (("Content-Range", cr) :: tagged)
              (String.sub data a (b - a + 1))
          in
          let lie = file ^ ".lie" in
          let lie = if k > 0 && Sys.file_exists lie then read lie else "" in
          match String.trim lie with
          | "" -> part a b len
          | "whole" -> answer 200 "OK" tagged data
          | "short" ->
              let cr = Printf.sprintf "bytes %d-%d/%d" a b len in
              answer 206 "Partial Content"
                (("Content-Range", cr) :: tagged)
                (String.sub data a (b - a))
          | "span" -> part (a + 1) (min (b + 1) (len - 1)) len
          | "length" -> part a b (len + 1)
          | "past" ->
              answer 416 "The range specified is invalid for the current size."
                [ ("Content-Range", Printf.sprintf "bytes */%d" len) ]
                ""
          | l -> fail "lie %S" l))

let script () =
  let d = path "script" in
  let names = if Sys.file_exists d then Sys.readdir d else [||] in
  Array.sort compare names;
  if names = [||] then (
    prerr_endline ("fake curl: nothing answers " ^ url);
    exit 6);
  let f = Filename.concat d names.(0) in
  let s = read f in
  Sys.remove f;
  match Scanf.sscanf_opt s "curl-exit %d" Fun.id with
  | Some n -> exit n
  | None ->
      set_binary_mode_out stdout true;
      print_string s;
      exit 0

(* [serve] and [script] exit. *)
let () =
  let no_query = List.hd (String.split_on_char '?' url) in
  let prefix = "https://" in
  let n = String.length prefix in
  let file =
    Filename.concat (path "files")
      (String.sub no_query n (String.length no_query - n))
  in
  if String.starts_with ~prefix no_query && Sys.file_exists file then
    serve file
  else script ()
