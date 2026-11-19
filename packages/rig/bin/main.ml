(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_bin

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

(* rig run's options so far. *)
type run_options = {
  on : string option;
  rig : string option;
  dirs : string list;  (** --firmware's, last first. *)
  silence : float option;
}

let run args =
  let misuse = misuse "rig run" in
  let rec parse o = function
    | "--help" :: _ -> page Help.run
    | "--" :: [] -> misuse "no program after --"
    | "--" :: prog :: args -> (
        match o.on with
        | None -> misuse "--on is missing"
        | Some on ->
            let rig = Option.value o.rig ~default:"rig" in
            let silence =
              Option.value o.silence ~default:Rig_remote.default_silence
            in
            (machines on, rig, List.rev o.dirs, silence, prog, args))
    | "--on" :: [] -> misuse "--on needs its machines"
    | "--on" :: s :: rest -> on_ o s rest
    | a :: rest when String.starts_with ~prefix:"--on=" a ->
        on_ o (String.sub a 5 (String.length a - 5)) rest
    | "--rig" :: [] -> misuse "--rig needs its program"
    | "--rig" :: r :: rest -> rig_ o r rest
    | a :: rest when String.starts_with ~prefix:"--rig=" a ->
        rig_ o (String.sub a 6 (String.length a - 6)) rest
    | "--firmware" :: [] -> misuse "--firmware needs its directory"
    | "--firmware" :: d :: rest ->
        parse { o with dirs = directory misuse d :: o.dirs } rest
    | a :: rest when String.starts_with ~prefix:"--firmware=" a ->
        let d = String.sub a 11 (String.length a - 11) in
        parse { o with dirs = directory misuse d :: o.dirs } rest
    | "--silence" :: [] -> misuse "--silence needs its seconds"
    | "--silence" :: v :: rest -> silence_ o v rest
    | a :: rest when String.starts_with ~prefix:"--silence=" a ->
        silence_ o (String.sub a 10 (String.length a - 10)) rest
    | a :: _ when String.starts_with ~prefix:"-" a ->
        misuse (strf "unknown option '%s'" a)
    | _ :: _ | [] -> (
        match o.on with
        | None -> misuse "--on is missing"
        | Some _ -> misuse "-- is missing before the program")
  and on_ o s rest =
    if o.on <> None then misuse "--on is given twice";
    if s = "" then misuse "--on needs its machines";
    parse { o with on = Some s } rest
  and rig_ o r rest =
    if o.rig <> None then misuse "--rig is given twice";
    if r = "" then misuse "--rig needs its program";
    parse { o with rig = Some r } rest
  and silence_ o v rest =
    if o.silence <> None then misuse "--silence is given twice";
    match float_of_string_opt v with
    | Some s when Rig_remote.valid_silence s ->
        parse { o with silence = Some s } rest
    | _ -> misuse (strf "--silence: '%s' is not a silence bound" v)
  in
  let names, rig, firmware, silence, prog, args =
    parse { on = None; rig = None; dirs = []; silence = None } args
  in
  Run.run ~misuse ~rig ~firmware ~silence names prog args

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
      | [] -> Firmware.fetch list None
      | a :: _ when option a -> misuse (strf "unknown option '%s'" a)
      | [ "" ] -> misuse "the directory is empty"
      | [ dir ] -> Firmware.fetch list (Some dir)
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
