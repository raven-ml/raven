(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Error messages *)

let err_no_curl url = Printf.sprintf "%s: curl not found on PATH" url
let err_download url = Printf.sprintf "Failed to download %s" url

let err_spawn url e =
  Printf.sprintf "%s: curl could not run: %s" url (Unix.error_message e)

let err_nul url =
  Printf.sprintf "%s: holds a NUL, which no command line can carry" url

let err_unix url e fn arg =
  Printf.sprintf "%s: %s %s: %s" url fn arg (Unix.error_message e)

(* Directories *)

(* [trim path] drops [path]'s trailing separators, keeping a root whole. Older
   Windows C runtimes fail to stat a directory named with one. *)
let rec trim path =
  let n = String.length path in
  let sep c = c = '/' || c = Filename.dir_sep.[0] in
  if n > 1 && sep path.[n - 1] && Filename.dirname path <> path then
    trim (String.sub path 0 (n - 1))
  else path

(* Parents first, through [Filename.dirname], which knows each system's roots:
   ["/"] on Unix, ["C:\\"] and ["\\\\server\\share"] on Windows. A root is its
   own [dirname], which ends the recursion. *)
let rec mkdir_p path =
  let path = trim path in
  if path = "" then ()
  else if Sys.file_exists path then (
    if not (Sys.is_directory path) then
      failwith
        (Printf.sprintf "mkdir_p: '%s' exists but is not a directory" path))
  else
    let parent = Filename.dirname path in
    if parent <> path then mkdir_p parent;
    try Unix.mkdir path 0o755
    with Unix.Unix_error (Unix.EEXIST, _, _) ->
      if not (Sys.is_directory path) then
        failwith
          (Printf.sprintf "mkdir_p: '%s' appeared as non-directory after EEXIST"
             path)

(* Files written whole *)

(* [create_beside path] is a file created exclusively beside [path], under a
   random name, and its descriptor: a name a crashed run left behind is skipped
   for another. *)
let create_beside path =
  let flags = Unix.[ O_WRONLY; O_CREAT; O_EXCL; O_CLOEXEC ] in
  let rec go tries =
    let tmp = Printf.sprintf "%s.%08x.tmp" path (Random.State.bits rng) in
    match Unix.openfile tmp flags 0o644 with
    | fd -> (tmp, fd)
    | exception Unix.Unix_error (Unix.EEXIST, _, _) when tries > 1 ->
        go (tries - 1)
  and rng = Random.State.make_self_init () in
  go 16

let write path f =
  let tmp, fd = create_beside path in
  let remove () = try Sys.remove tmp with Sys_error _ -> () in
  let fill () =
    f fd;
    Unix.fsync fd
  in
  match Fun.protect ~finally:(fun () -> Unix.close fd) fill with
  | () -> (
      try Unix.rename tmp path
      with Unix.Unix_error _ as e ->
        remove ();
        if not (Sys.file_exists path) then raise e)
  | exception e ->
      remove ();
      raise e

let rec write_all fd b off n =
  if n > 0 then
    let k = Unix.write fd b off n in
    write_all fd b (off + k) (n - k)

(* curl *)

(* [curl] runs without a shell, so no argument needs quoting and the lookup on
   [PATH] is the same on every system. A missing program is exit code 127 from
   the forked child on Unix and [ENOENT] from process creation on Windows.
   Any other failure to make the process, such as a program on [PATH] that
   cannot be executed, fails the request for [url]. *)
let spawn ~url ~stdin args out =
  try
    Unix.create_process "curl"
      (Array.of_list ("curl" :: args))
      stdin out Unix.stderr
  with
  | Unix.Unix_error (Unix.ENOENT, _, _) -> failwith (err_no_curl url)
  | Unix.Unix_error (e, _, _) -> failwith (err_spawn url e)

let rec wait pid =
  try snd (Unix.waitpid [] pid)
  with Unix.Unix_error (Unix.EINTR, _, _) -> wait pid

let succeeded ~url = function
  | Unix.WEXITED 0 -> true
  | Unix.WEXITED 127 -> failwith (err_no_curl url)
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> false

let curl ~url ~stdin args ~out =
  succeeded ~url (wait (spawn ~url ~stdin args out))

(* [curl args] with its output read through a pipe, each chunk given to [f]
   as it arrives. *)
let curl_read ~url ~stdin args ~f =
  let r, w = Unix.pipe ~cloexec:true () in
  let pid =
    Fun.protect
      ~finally:(fun () -> Unix.close w)
      (fun () ->
        try spawn ~url ~stdin args w
        with e ->
          Unix.close r;
          raise e)
  in
  let buf = Bytes.create 65536 in
  let rec copy () =
    match Unix.read r buf 0 (Bytes.length buf) with
    | 0 -> ()
    | n ->
        f buf n;
        copy ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> copy ()
  in
  match copy () with
  | () ->
      Unix.close r;
      succeeded ~url (wait pid)
  | exception e ->
      (* curl, writing to a closed pipe, ends. *)
      Unix.close r;
      ignore (wait pid);
      raise e

(* The headers curl reads on its standard input ([-H @-]), so that none, a
   token's above all, appears on its command line, which every user of the
   machine can read. They are written to the pipe before curl starts: at most
   1024 bytes, well under the 4096 a pipe is made with on Windows and the page a
   Linux pipe keeps at least, so the write cannot block with no reader, and it
   cannot raise SIGPIPE while this process holds the read end. *)
let max_header_bytes = 1024

let header_text headers =
  String.concat "" (List.map (fun (k, v) -> k ^ ": " ^ v ^ "\n") headers)

let header_input headers =
  let text = header_text headers in
  let r, w = Unix.pipe ~cloexec:true () in
  Fun.protect
    ~finally:(fun () -> Unix.close w)
    (fun () ->
      try ignore (Unix.write_substring w text 0 (String.length text))
      with e ->
        Unix.close r;
        raise e);
  r

let with_headers headers f =
  match headers with
  | [] -> f ~stdin:Unix.stdin []
  | _ ->
      let r = header_input headers in
      Fun.protect
        ~finally:(fun () -> Unix.close r)
        (fun () -> f ~stdin:r [ "-H"; "@-" ])

(* With [sha256], the download is hashed as it is written and refused unless
   its digest is [sha256]. *)
let fetch ~headers ?sha256 ~url fd =
  let ok =
    with_headers headers @@ fun ~stdin header_args ->
    (* [-q] first: curl reads no ~/.curlrc, whose options, such as
       [--location-trusted] or [-o], would act on the download. *)
    let args = [ "-q"; "-L"; "--fail"; "-s" ] @ header_args @ [ url ] in
    match sha256 with
    | None -> curl ~url ~stdin args ~out:fd
    | Some want ->
        let h = Sha256.init () in
        let f b n =
          Sha256.feed h b 0 n;
          write_all fd b 0 n
        in
        let ok = curl_read ~url ~stdin args ~f in
        let got = Sha256.hex h in
        if ok && got <> want then
          failwith (Printf.sprintf "%s: SHA-256 %s, expected %s" url got want);
        ok
  in
  if not ok then failwith (err_download url)

let file ?(headers = []) ?sha256 ~url path =
  if String.length (header_text headers) > max_header_bytes then
    invalid_arg
      (Printf.sprintf "Kaun_download.file: the headers %s exceed %d bytes"
         (String.concat ", " (List.map fst headers))
         max_header_bytes);
  (* A NUL ends a command line, and a path the URL's parts name: refused
     before either is made. *)
  if String.contains url '\000' then failwith (err_nul url);
  try
    (try mkdir_p (Filename.dirname path)
     with Failure m -> failwith (Printf.sprintf "%s: %s" url m));
    write path (fetch ~headers ?sha256 ~url)
  with Unix.Unix_error (e, fn, arg) -> failwith (err_unix url e fn arg)
