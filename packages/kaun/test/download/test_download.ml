(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nothing here touches the network: a [curl] stand-in on [PATH] serves what a
   test downloads. *)

open Windtrap

let read path = In_channel.with_open_bin path In_channel.input_all
let write_text path s =
  Out_channel.with_open_bin path (fun oc -> output_string oc s)
let listing dir = List.sort compare (Array.to_list (Sys.readdir dir))
let url = "https://example.com/data/f.bin"

(* A directory whose write bit is cleared refuses new names and renames, except
   to root. The mode is restored for the test's cleanup. *)
let unwritable () =
  if Sys.win32 then skip ~reason:"directory modes are POSIX's" ();
  if Unix.geteuid () = 0 then skip ~reason:"root writes any directory" ()

let restore dir = Unix.chmod dir 0o755

(* A [curl] alone on [PATH] whose script is the lines [body]. *)
let stand_in body =
  let bin = temp_dir ~prefix:"bin" () in
  let curl = Filename.concat bin "curl" in
  write_text curl (String.concat "\n" (("#!/bin/sh" :: body) @ [ "" ]));
  Unix.chmod curl 0o755;
  setenv "PATH" (Some bin)

(* write *)

let writes_whole () =
  let dir = temp_dir () in
  let path = Filename.concat dir "f" in
  Kaun_download.write path (fun fd ->
      ignore (Unix.write_substring fd "payload" 0 7));
  equal ~msg:"contents" string "payload" (read path);
  equal ~msg:"files" (list string) [ "f" ] (listing dir)

let failure_leaves_path () =
  let dir = temp_dir () in
  let path = Filename.concat dir "f" in
  raises_match (Exn.failure ~sub:"midway") (fun () ->
      Kaun_download.write path (fun fd ->
          ignore (Unix.write_substring fd "part" 0 4);
          failwith "midway"));
  equal ~msg:"files" (list string) [] (listing dir)

(* [f] clears the directory's write bit, so the rename fails: nothing reaches
   the path. The temporary file cannot be removed there either. *)
let rename_fails () =
  unwritable ();
  let dir = temp_dir () in
  let path = Filename.concat dir "f" in
  Fun.protect
    ~finally:(fun () -> restore dir)
    (fun () ->
      raises_match
        (function Unix.Unix_error (Unix.EACCES, _, _) -> true | _ -> false)
        (fun () ->
          Kaun_download.write path (fun fd ->
              ignore (Unix.write_substring fd "payload" 0 7);
              Unix.chmod dir 0o555)));
  equal ~msg:"the path" bool false (Sys.file_exists path)

(* A path written meanwhile, here before the call, is kept when the rename
   fails. *)
let keeps_a_written_path () =
  unwritable ();
  let dir = temp_dir () in
  let path = Filename.concat dir "f" in
  write_text path "old";
  Fun.protect
    ~finally:(fun () -> restore dir)
    (fun () ->
      Kaun_download.write path (fun fd ->
          ignore (Unix.write_substring fd "new" 0 3);
          Unix.chmod dir 0o555));
  equal ~msg:"contents" string "old" (read path)

(* mkdir_p *)

let makes_parents () =
  let dir = Filename.concat (Filename.concat (temp_dir ()) "a") "b" in
  Kaun_download.mkdir_p dir;
  equal bool true (Sys.is_directory dir)

let refuses_a_file () =
  let file = Filename.concat (temp_dir ()) "a" in
  write_text file "";
  raises_match (Exn.failure ~sub:"not a directory") (fun () ->
      Kaun_download.mkdir_p (Filename.concat file "b"))

(* file *)

(* curl's stand-in writes [payload] and succeeds. *)
let downloads () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  stand_in [ "printf payload" ];
  let dir = Filename.concat (temp_dir ()) "cache" in
  let path = Filename.concat dir "f.bin" in
  Kaun_download.file ~url path;
  equal ~msg:"contents" string "payload" (read path);
  equal ~msg:"files" (list string) [ "f.bin" ] (listing dir)

(* curl's stand-in clears the cache directory's write bit, then writes its
   payload and succeeds: the rename fails. *)
let rename_fails_naming_url () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  unwritable ();
  let dir = temp_dir () in
  let path = Filename.concat dir "f.bin" in
  stand_in
    [
      Printf.sprintf "/bin/chmod a-w %s" (Filename.quote dir); "printf payload";
    ];
  Fun.protect
    ~finally:(fun () -> restore dir)
    (fun () ->
      raises_match (Exn.failure ~sub:url) (fun () ->
          Kaun_download.file ~url path));
  equal ~msg:"the path" bool false (Sys.file_exists path)

(* A cache directory that refuses new names: the temporary file cannot be
   made, and curl never runs. *)
let unwritable_directory () =
  if Sys.win32 then skip ~reason:"the curl stand-in is a shell script" ();
  unwritable ();
  let ran = Filename.concat (temp_dir ()) "ran" in
  stand_in [ Printf.sprintf ": > %s" (Filename.quote ran); "printf payload" ];
  let dir = temp_dir () in
  Unix.chmod dir 0o555;
  Fun.protect
    ~finally:(fun () -> restore dir)
    (fun () ->
      raises_match (Exn.failure ~sub:url) (fun () ->
          Kaun_download.file ~url (Filename.concat dir "f.bin")));
  equal ~msg:"curl ran" bool false (Sys.file_exists ran)

let not_a_directory () =
  let file = Filename.concat (temp_dir ()) "a" in
  write_text file "";
  raises_match (Exn.failure ~sub:url) (fun () ->
      Kaun_download.file ~url (Filename.concat (Filename.concat file "b") "f"))

let () =
  exit
    (run "kaun download"
       [
         group "write"
           [
             test
               "renames what f wrote into the path, leaving nothing beside it"
               writes_whole;
             test "removes the file when f raises" failure_leaves_path;
             test "raises when the rename fails, leaving no path" rename_fails;
             test "keeps a path written meanwhile when the rename fails"
               keeps_a_written_path;
           ];
         group "mkdir_p"
           [
             test "makes the missing parents" makes_parents;
             test "raises Failure on a file in the path" refuses_a_file;
           ];
         group "file"
           [
             test "renames curl's output into the path" downloads;
             test
               "raises Failure naming the URL when the rename fails, leaving \
                no path"
               rename_fails_naming_url;
             test
               "raises Failure naming the URL when the file cannot be made"
               unwritable_directory;
             test
               "raises Failure naming the URL on a file in the path's \
                directories"
               not_a_directory;
           ];
       ])
