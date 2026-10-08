(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rig: argv, the help pages, and the commands. *)

let strf = Printf.sprintf

let misuse cmd why =
  Proc.write Unix.stderr (strf "rig: %s\nTry '%s --help'.\n" why cmd);
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

(* A directory of --firmware: misuse when empty. *)
let directory misuse d =
  if d = "" then misuse "--firmware needs its directory";
  d

let run args =
  let misuse = misuse "rig run" in
  let rec parse on dirs = function
    | "--help" :: _ -> page Help.run
    | "--" :: [] -> misuse "no program after --"
    | "--" :: prog :: args -> (
        match on with
        | None -> misuse "--on is missing"
        | Some on -> (machines on, List.rev dirs, prog, args))
    | "--on" :: [] -> misuse "--on needs its machines"
    | "--on" :: s :: rest -> on_ on dirs s rest
    | a :: rest when String.starts_with ~prefix:"--on=" a ->
        on_ on dirs (String.sub a 5 (String.length a - 5)) rest
    | "--firmware" :: [] -> misuse "--firmware needs its directory"
    | "--firmware" :: d :: rest -> parse on (directory misuse d :: dirs) rest
    | a :: rest when String.starts_with ~prefix:"--firmware=" a ->
        let d = String.sub a 11 (String.length a - 11) in
        parse on (directory misuse d :: dirs) rest
    | a :: _ when String.starts_with ~prefix:"-" a ->
        misuse (strf "unknown option '%s'" a)
    | _ :: _ | [] -> (
        match on with
        | None -> misuse "--on is missing"
        | Some _ -> misuse "-- is missing before the program")
  and on_ on dirs s rest =
    if on <> None then misuse "--on is given twice";
    if s = "" then misuse "--on needs its machines";
    parse (Some s) dirs rest
  in
  let names, firmware, prog, args = parse None [] args in
  Run.run ~misuse ~firmware names prog args

(* rig agent *)

let agent args =
  let misuse = misuse "rig agent" in
  let rec parse dirs = function
    | "--help" :: _ -> page Help.agent
    | "--firmware" :: [] -> misuse "--firmware needs its directory"
    | "--firmware" :: d :: rest -> parse (directory misuse d :: dirs) rest
    | a :: rest when String.starts_with ~prefix:"--firmware=" a ->
        let d = String.sub a 11 (String.length a - 11) in
        parse (directory misuse d :: dirs) rest
    | a :: _ when String.length a > 1 && a.[0] = '-' ->
        misuse (strf "unknown option '%s'" a)
    | [] -> misuse "the address is missing"
    | [ a ] -> (List.rev dirs, a)
    | _ :: extra :: _ -> misuse (strf "unexpected argument '%s'" extra)
  in
  let firmware, a = parse [] args in
  match Address.host_port a with
  | Error why -> misuse why
  | Ok (host, port) ->
      if Sys.getenv_opt "RIG_REMOTE_REPORT" = None then Agent.half ~firmware a
      else Agent.agent ~firmware host port

(* rig firmware *)

let firmware args =
  let misuse = misuse "rig firmware" in
  let option a = String.length a > 1 && a.[0] = '-' in
  if List.mem "--help" args then page Help.firmware;
  match args with
  | a :: _ when option a -> misuse (strf "unknown option '%s'" a)
  | [] -> misuse "the driver is missing; the drivers are amd and nv"
  | driver :: rest -> (
      let list =
        match driver with
        | "amd" -> Images.amd
        | "nv" -> Images.nv
        | d -> misuse (strf "unknown driver '%s'; the drivers are amd and nv" d)
      in
      match rest with
      | [] -> misuse "the directory is missing"
      | a :: _ when option a -> misuse (strf "unknown option '%s'" a)
      | [ "" ] -> misuse "the directory is empty"
      | [ dir ] -> Firmware.fetch list dir
      | _ :: extra :: _ -> misuse (strf "unexpected argument '%s'" extra))

let main () =
  let misuse = misuse "rig" in
  match List.tl (Array.to_list Sys.argv) with
  | [] -> misuse "no command; the commands are run, agent and firmware"
  | "--help" :: _ -> page Help.rig
  | "--version" :: _ -> page (Version.v ^ "\n")
  | "run" :: args -> run args
  | "agent" :: args -> agent args
  | "firmware" :: args -> firmware args
  | c :: _ ->
      misuse
        (strf "unknown command '%s'; the commands are run, agent and firmware" c)

let () =
  try main ()
  with e ->
    Proc.write Unix.stderr ("rig: a bug in rig: " ^ Printexc.to_string e ^ "\n");
    exit 125
