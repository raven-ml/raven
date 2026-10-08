(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type image = { path : string; digest : string; url : string }

(* A path of the list names a file under DIR: relative, no segment empty, [.] or
   [..]. *)
let confined p =
  Filename.is_relative p
  && List.for_all
       (fun s -> s <> "" && s <> "." && s <> "..")
       (String.split_on_char '/' p)

let images list =
  let image l =
    if l = "" || l.[0] = '#' then None
    else
      match String.split_on_char '\t' l with
      | [ path; digest; url ] when confined path -> Some { path; digest; url }
      | _ -> invalid_arg ("Firmware.images: " ^ l)
  in
  List.filter_map image (String.split_on_char '\n' list)

let fail why =
  Proc.write Unix.stderr ("rig: " ^ why ^ "\n");
  exit 123

let contents file =
  try Some (In_channel.with_open_bin file In_channel.input_all)
  with Sys_error _ -> None

let rec waitpid pid =
  try snd (Unix.waitpid [] pid)
  with Unix.Unix_error (Unix.EINTR, _, _) -> waitpid pid

(* curl writes the image on a pipe, so that a download with another digest
   reaches no file. A transfer that stalls for 60 s fails, however large the
   image. *)
let download url =
  let r, w = Proc.pipe () in
  let args =
    [| "curl"; "-fsSL"; "--speed-limit"; "1"; "--speed-time"; "60"; url |]
  in
  let pid =
    try
      Proc.spawn "curl" args ~stdin:Unix.stdin ~stdout:w ~stderr:Unix.stderr
    with
    | Unix.Unix_error (Unix.ENOENT, _, _) ->
        fail "curl is not on PATH; rig firmware downloads with it"
    | Unix.Unix_error (e, _, _) -> fail ("curl: " ^ Unix.error_message e)
  in
  Unix.close w;
  let buf = Buffer.create 65536 and chunk = Bytes.create 65536 in
  let rec read () =
    match Unix.read r chunk 0 (Bytes.length chunk) with
    | 0 -> ()
    | n ->
        Buffer.add_subbytes buf chunk 0 n;
        read ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> read ()
  in
  read ();
  Unix.close r;
  match waitpid pid with
  | Unix.WEXITED 0 -> Ok (Buffer.contents buf)
  | st -> Error ("curl " ^ Proc.cause st)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755
  end

(* The image goes to a temporary file beside its place, flushed to the disk and
   renamed into it: the place holds the old file or the whole image, never a
   part, even after a crash. *)
let write file data =
  let part = file ^ ".part" in
  let save () =
    let fd =
      Unix.openfile part [ O_WRONLY; O_CREAT; O_TRUNC; O_CLOEXEC ] 0o644
    in
    Fun.protect
      ~finally:(fun () -> Unix.close fd)
      (fun () ->
        ignore (Unix.write_substring fd data 0 (String.length data));
        Unix.fsync fd)
  in
  let failed why =
    (try Sys.remove part with Sys_error _ -> ());
    Error why
  in
  match
    mkdir_p (Filename.dirname file);
    save ();
    Unix.rename part file
  with
  | () -> Ok ()
  | exception Sys_error why -> failed why
  | exception Unix.Unix_error (e, _, arg) ->
      failed (strf "%s: %s" arg (Unix.error_message e))

let fetched i file =
  match download i.url with
  | Error _ as e -> e
  | Ok data ->
      let got = Rig_pci.Firmware.digest data in
      if got <> i.digest then
        Error (strf "the download's digest is %s; the pin is %s" got i.digest)
      else write file data

(* Each image is fetched in turn, the failures said as they come. *)
let fetch list dir =
  let image ok i =
    let file = Filename.concat dir i.path in
    match contents file with
    | Some s when Rig_pci.Firmware.digest s = i.digest ->
        print_endline ("kept " ^ i.path);
        ok
    | _ -> (
        match fetched i file with
        | Ok () ->
            print_endline ("fetched " ^ i.path);
            ok
        | Error why ->
            Proc.write Unix.stderr (strf "rig: %s: %s\n" i.path why);
            false)
  in
  exit (if List.fold_left image true (images list) then 0 else 123)
