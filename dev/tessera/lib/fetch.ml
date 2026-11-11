(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Error messages *)

let err_no_curl = "curl not found on PATH"
let err_curl name n = strf "%s: curl exited %d" name n
let err_signal name = strf "%s: curl ended by a signal" name

let err_https name = strf "%s: not an https:// URL" name

let err_status name status ~retry body =
  let n = min (String.length body) 200 in
  let retry =
    match retry with Some s -> strf " (Retry-After %s)" s | None -> ""
  in
  strf "%s: HTTP status %d%s: %s" name status retry (String.sub body 0 n)

let err_answer name = strf "%s: malformed HTTP answer from curl" name
let err_json name e = strf "%s: %s" name e
let err_token name = strf "%s: no token or msft:expiry in the token answer" name

let err_range name what =
  strf "%s: the answer is not the range asked for: %s" name what


(* curl *)

(* The exit codes of curl's transient failures: no connection (7), a
   transfer cut short (18), a timeout (28), a TLS handshake (35), no answer
   (52), a send or receive failure (55, 56), an HTTP/2 stream error (92). *)
let transient = [ 7; 18; 28; 35; 52; 55; 56; 92 ]
let attempts = 5
let first_delay = 1.

(* The waits of one request, in seconds: at most a minute in all. *)
let max_wait = 60.

let options =
  [
    "-sS";
    "-i";
    "--proto";
    "=https";
    "--suppress-connect-headers";
    "--connect-timeout";
    "30";
    "--speed-limit";
    "1";
    "--speed-time";
    "60";
  ]

type answer = {
  status : int;
  headers : (string * string) list; (* Names in lower case. *)
  body : string;
}

let rec wait pid =
  try snd (Unix.waitpid [] pid)
  with Unix.Unix_error (Unix.EINTR, _, _) -> wait pid

(* [curl] runs without a shell, so no argument needs quoting, and its output
   comes back through a pipe, read whole before the process is reaped. *)
let curl args =
  let r, w = Unix.pipe ~cloexec:true () in
  let pid =
    match
      Unix.create_process "curl"
        (Array.of_list ("curl" :: args))
        Unix.stdin w Unix.stderr
    with
    | pid ->
        Unix.close w;
        pid
    | exception Unix.Unix_error (Unix.ENOENT, _, _) ->
        Unix.close w;
        Unix.close r;
        failwith err_no_curl
  in
  let ic = Unix.in_channel_of_descr r in
  let out =
    Fun.protect
      ~finally:(fun () -> close_in_noerr ic)
      (fun () -> In_channel.input_all ic)
  in
  (wait pid, out)

let crlf2 = "\r\n\r\n"

(* The first blank line of [s], where its headers end. *)
let blank s =
  let rec go i =
    if i + 4 > String.length s then None
    else if s.[i] = '\r' && String.sub s i 4 = crlf2 then Some i
    else go (i + 1)
  in
  go 0

let header_line l =
  match String.index_opt l ':' with
  | None -> None
  | Some i ->
      let name = String.lowercase_ascii (String.trim (String.sub l 0 i)) in
      Some (name, String.trim (String.sub l (i + 1) (String.length l - i - 1)))

(* curl's [-i] output: a status line ([HTTP/<version> <code> ...]) and headers
   up to a blank line, then the body. Interim answers (1xx) come first, each
   with its own blank line. *)
let rec parse name out =
  let fail () = failwith (err_answer name) in
  let stop = match blank out with Some i -> i | None -> fail () in
  let lines = String.split_on_char '\n' (String.sub out 0 stop) in
  let lines = List.map String.trim lines in
  let status =
    match lines with
    | first :: _ when String.starts_with ~prefix:"HTTP/" first -> (
        match String.split_on_char ' ' first with
        | _ :: code :: _ -> (
            match int_of_string_opt code with Some c -> c | None -> fail ())
        | _ -> fail ())
    | _ -> fail ()
  in
  let rest = String.sub out (stop + 4) (String.length out - stop - 4) in
  if status >= 100 && status < 200 then parse name rest
  else
    let headers = List.filter_map header_line (List.tl lines) in
    { status; headers; body = rest }

let header a name = List.assoc_opt name a.headers
let retried a = a.status = 429 || a.status >= 500

let backoff k = first_delay *. Float.pow 2. (Float.of_int k)

let delay a k =
  match Option.bind (header a "retry-after") int_of_string_opt with
  | Some s when s >= 0 -> Float.of_int s
  | _ -> backoff k

let https = "https://"

(* [send ~name url] is the answer to a request of the URL [url ()], made
   again on each attempt, so a signed URL is signed afresh. [name] is the
   unsigned URL, which failures name. A retried status that persists, or
   whose wait would pass [max_wait], is the last answer's. *)
let send ?(headers = []) ?body ~name url =
  if not (String.starts_with ~prefix:https name) then
    failwith (err_https name);
  let headers =
    List.concat_map (fun (k, v) -> [ "-H"; k ^ ": " ^ v ]) headers
  in
  let body = match body with Some b -> [ "--data-raw"; b ] | None -> [] in
  let rec go k waited =
    let last = k + 1 = attempts in
    let again wait =
      Unix.sleepf wait;
      go (k + 1) (waited +. wait)
    in
    match curl (options @ headers @ body @ [ "--url"; url () ]) with
    | Unix.WEXITED 0, out ->
        let a = parse name out in
        let wait = delay a k in
        if retried a && (not last) && waited +. wait <= max_wait then
          again wait
        else a
    | Unix.WEXITED 127, _ -> failwith err_no_curl
    | Unix.WEXITED n, _ when List.mem n transient && not last ->
        again (backoff k)
    | Unix.WEXITED n, _ -> failwith (err_curl name n)
    | (Unix.WSIGNALED _ | Unix.WSTOPPED _), _ -> failwith (err_signal name)
  in
  go 0 0.

let fail_status name a =
  failwith (err_status name a.status ~retry:(header a "retry-after") a.body)

(* JSON *)

let json name a =
  if a.status < 200 || a.status > 299 then fail_status name a;
  match Jsont_bytesrw.decode_string Jsont.json a.body with
  | Ok j -> j
  | Error e -> failwith (err_json name e)

let get_json url = json url (send ~name:url (fun () -> url))

let post_json url body =
  let body =
    match Jsont_bytesrw.encode_string Jsont.json body with
    | Ok s -> s
    | Error e -> invalid_arg e
  in
  let headers = [ ("Content-Type", "application/json") ] in
  json url (send ~headers ~body ~name:url (fun () -> url))

(* Sessions *)

module Smap = Map.Make (String)

let margin = 60.

type session = (string * float) Smap.t Atomic.t

let session () = Atomic.make Smap.empty

(* A token is replaced only if none valid at [now] was stored meanwhile, so
   each call returns what some order of the calls would. *)
let token s ~now key fetch =
  let valid m =
    match Smap.find_opt key m with
    | Some (t, expiry) when now < expiry -. margin -> Some t
    | _ -> None
  in
  match valid (Atomic.get s) with
  | Some t -> t
  | None ->
      let t, expiry = fetch () in
      let rec store () =
        let m = Atomic.get s in
        match valid m with
        | Some t' -> t'
        | None ->
            if Atomic.compare_and_set s m (Smap.add key (t, expiry) m) then t
            else store ()
      in
      store ()

let rec forget s key =
  let m = Atomic.get s in
  if not (Atomic.compare_and_set s m (Smap.remove key m)) then forget s key

(* Signing *)

let mpc_hosts =
  [
    "sentinel2l2a01.blob.core.windows.net";
    "sentinel1euwestrtc.blob.core.windows.net";
  ]

let token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/"

(* The index of the first of [cs] in [s] from [i], or [s]'s length. *)
let upto s i cs =
  let rec go i =
    if i >= String.length s || String.contains cs s.[i] then i else go (i + 1)
  in
  go i

(* The account and container of a blob of MPC's,
   [https://<account>.blob.core.windows.net/<container>/<blob>], as the token
   endpoint's path names them: ["<account>/<container>"]. The host ends at
   the first '/', '?' or '#', and must be one of [mpc_hosts]; an href with a
   fragment is not signed, since a query after it would not be sent. *)
let container href =
  let n = String.length https in
  if (not (String.starts_with ~prefix:https href)) || String.contains href '#'
  then None
  else
    let stop = upto href n "/?#" in
    let host = String.sub href n (stop - n) in
    if not (List.mem host mpc_hosts && stop < String.length href) then None
    else if href.[stop] <> '/' then None
    else
      let path = String.sub href (stop + 1) (upto href stop "?" - stop - 1) in
      match String.split_on_char '/' path with
      | container :: _ :: _ when container <> "" ->
          let account = String.sub host 0 (String.index host '.') in
          Some (account ^ "/" ^ container)
      | _ -> None

(* Days from 1970-01-01 to a day of the proleptic Gregorian calendar
   (Hinnant, "chrono-Compatible Low-Level Date Algorithms", days_from_civil). *)
let days (y, m, d) =
  let y = if m <= 2 then y - 1 else y in
  let era = (if y >= 0 then y else y - 399) / 400 in
  let yoe = y - (era * 400) in
  let doy = ((153 * (if m > 2 then m - 3 else m + 9)) + 2) / 5 + d - 1 in
  let doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy in
  (era * 146097) + doe - 719468

let unix_time (t : Jord_stac.time) =
  (Float.of_int (days t.date) *. 86400.) +. t.seconds

let string name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, Jsont.String (s, _)) -> Some s
      | _ -> None)
  | _ -> None

(* The endpoint answers [{"msft:expiry": <RFC 3339>, "token": <query>}]. *)
let request_token key () =
  let url = token_url ^ key in
  let j = get_json url in
  let time s = Result.to_option (Jord_stac.time s) in
  match (string "token" j, Option.bind (string "msft:expiry" j) time) with
  | Some t, Some e -> (t, unix_time e)
  | _ -> failwith (err_token url)

let sign s href =
  match container href with
  | None -> href
  | Some key ->
      let now = Unix.gettimeofday () in
      let sep = if String.contains href '?' then "&" else "?" in
      href ^ sep ^ token s ~now key (request_token key)

(* A request of [href], signed on each attempt; after a 403 to a signed
   request, once more with a new token. *)
let send_signed s ~headers href =
  let a = send ~headers ~name:href (fun () -> sign s href) in
  match container href with
  | Some key when a.status = 403 ->
      forget s key;
      send ~headers ~name:href (fun () -> sign s href)
  | _ -> a

(* Ranges *)

let first_read = 65536
let merge_gap = 16384

exception Changed of string

type file = {
  session : session;
  href : string;
  length : int;
  etag : string option;
}

let range (at, n) = ("Range", strf "bytes=%d-%d" at (at + n - 1))

(* [Content-Range: bytes <first>-<last>/<length>], or [bytes */<length>] in
   an answer of status 416. *)
let content_range name a =
  let fail what = failwith (err_range name what) in
  let v =
    match header a "content-range" with
    | Some v -> v
    | None -> fail "no Content-Range"
  in
  let int s =
    match int_of_string_opt s with Some i when i >= 0 -> i | _ -> fail v
  in
  match String.split_on_char ' ' v with
  | [ "bytes"; spec ] -> (
      match String.split_on_char '/' spec with
      | [ "*"; length ] -> (None, int length)
      | [ span; length ] -> (
          match String.split_on_char '-' span with
          | [ first; last ] -> (Some (int first, int last), int length)
          | _ -> fail v)
      | _ -> fail v)
  | _ -> fail v

let open_file session href =
  let a = send_signed session ~headers:[ range (0, first_read) ] href in
  let etag = header a "etag" in
  let body = a.body in
  let held length =
    let p = Nx_io.Partial.v ~name:href ~length in
    ({ session; href; length; etag }, Nx_io.Partial.add p ~at:0 body)
  in
  match a.status with
  | 200 -> held (String.length body)
  | 206 -> (
      match content_range href a with
      | Some (0, last), length
        when last + 1 = String.length body && last < length ->
          held length
      | _ -> failwith (err_range href "bytes 0 onward"))
  | 416 -> (
      match content_range href a with
      | None, 0 -> held 0
      | _ -> fail_status href a)
  | _ -> fail_status href a

let get f (at, n) =
  let if_match =
    match f.etag with Some e -> [ ("If-Match", e) ] | None -> []
  in
  let headers = range (at, n) :: if_match in
  let a = send_signed f.session ~headers f.href in
  if a.status = 412 then raise (Changed f.href);
  (match (f.etag, header a "etag") with
  | Some e, Some e' when e <> e' -> raise (Changed f.href)
  | _ -> ());
  match a.status with
  | 206 -> (
      let last = at + n - 1 in
      match content_range f.href a with
      | Some span, length
        when span = (at, last) && length = f.length
             && String.length a.body = n ->
          a.body
      | _ -> failwith (err_range f.href (strf "bytes %d-%d" at last)))
  | _ -> fail_status f.href a

(* Sorted ranges, those less than [merge_gap] apart joined. *)
let merge rs =
  let join acc (at, n) =
    match acc with
    | (a, m) :: rest when at - (a + m) < merge_gap ->
        (a, max (a + m) (at + n) - a) :: rest
    | _ -> (at, n) :: acc
  in
  List.rev (List.fold_left join [] rs)

let fill f p rs =
  List.fold_left
    (fun p r -> Nx_io.Partial.add p ~at:(fst r) (get f r))
    p (merge rs)
