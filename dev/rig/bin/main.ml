(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig: argv, the help pages, and the commands. *)

let strf = Printf.sprintf

let misuse cmd why =
  prerr_string (strf "rig: %s\nTry '%s --help'.\n" why cmd);
  exit 124

let page p =
  print_string p;
  exit 0

(* rig run *)

let machines s =
  let misuse = misuse "rig run" in
  let names = String.split_on_char ',' s in
  if List.mem "" names then misuse "--on has an empty name";
  List.iter
    (fun n ->
      match Address.machine n with
      | Ok _ -> ()
      | Error why -> misuse ("--on: " ^ why))
    names;
  let rec twice = function
    | [] -> ()
    | n :: rest ->
        if List.mem n rest then misuse (strf "--on names '%s' twice" n);
        twice rest
  in
  twice names;
  if List.length names < 2 then
    misuse "--on names one machine, expected two or more";
  names

let run args =
  let misuse = misuse "rig run" in
  let rec parse on = function
    | "--help" :: _ -> page Help.run
    | "--" :: [] -> misuse "no program after --"
    | "--" :: prog :: args -> (
        match on with
        | None -> misuse "--on is missing"
        | Some on -> (machines on, prog, args))
    | "--on" :: [] -> misuse "--on needs its machines"
    | "--on" :: s :: rest -> on_ on s rest
    | a :: rest when String.starts_with ~prefix:"--on=" a ->
        on_ on (String.sub a 5 (String.length a - 5)) rest
    | a :: _ when String.starts_with ~prefix:"-" a ->
        misuse (strf "unknown option '%s'" a)
    | _ :: _ | [] -> (
        match on with
        | None -> misuse "--on is missing"
        | Some _ -> misuse "-- is missing before the program")
  and on_ on s rest =
    if on <> None then misuse "--on is given twice";
    if s = "" then misuse "--on needs its machines";
    parse (Some s) rest
  in
  let names, prog, args = parse None args in
  Run.run ~misuse names prog args

(* rig agent *)

let agent args =
  let misuse = misuse "rig agent" in
  match args with
  | "--help" :: _ -> page Help.agent
  | [] -> misuse "the address is missing"
  | a :: _ when String.length a > 1 && a.[0] = '-' ->
      misuse (strf "unknown option '%s'" a)
  | [ a ] -> (
      match Address.host_port a with
      | Error why -> misuse why
      | Ok (host, port) ->
          if Sys.getenv_opt "RIG_REMOTE_REPORT" = None then Agent.half a
          else Agent.agent host port)
  | _ :: extra :: _ -> misuse (strf "unexpected argument '%s'" extra)

let main () =
  let misuse = misuse "rig" in
  match List.tl (Array.to_list Sys.argv) with
  | [] -> misuse "no command; the commands are run and agent"
  | "--help" :: _ -> page Help.rig
  | "--version" :: _ -> page (Line.version ^ "\n")
  | "run" :: args -> run args
  | "agent" :: args -> agent args
  | c :: _ ->
      misuse (strf "unknown command '%s'; the commands are run and agent" c)

let () =
  try main ()
  with e ->
    prerr_endline ("rig: a bug in rig: " ^ Printexc.to_string e);
    exit 125
