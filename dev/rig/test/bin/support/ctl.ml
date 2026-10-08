(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A controller for the tests: [ctl.exe MODE] counts its attempts in the file
   attempts of the current directory, prints "attempt N", and does as MODE says.

   - copy: copies 4 KiB to each machine's host and back, prints "copied N" for N
   machines, closes the job. - exit N: exits N once the job started; raise:
   raises. - env: prints the agents' names and the key's length, then whether
   the three variables are gone once the job started. - early N: exits N before
   starting the job. - kill: kills itself once the job started; kill-first: at
   its first attempt, then copies. - wait-first: at its first attempt, waits for
   the job to fail and raises; then copies. *)

let attempt () =
  let n =
    match In_channel.with_open_text "attempts" In_channel.input_all with
    | s -> int_of_string (String.trim s) + 1
    | exception Sys_error _ -> 1
  in
  Out_channel.with_open_text "attempts" (fun oc ->
      output_string oc (string_of_int n));
  Printf.printf "attempt %d\n%!" n;
  n

let job () =
  match Rig_remote.launched () with
  | Some (Ok j) -> j
  | Some (Error why) ->
      prerr_endline ("ctl.exe: " ^ why);
      exit 1
  | None ->
      prerr_endline "ctl.exe: not launched";
      exit 1

let copy j =
  let n = 4096 in
  let hosts = Rig_remote.hosts j in
  let check h =
    let here = Rig.Buffer.create Rig.host n
    and back = Rig.Buffer.create Rig.host n
    and there = Rig.Buffer.create h n in
    Bigarray.Array1.fill (Rig.Buffer.bigarray Bigarray.char here) 'r';
    Rig.Buffer.copy ~src:here ~dst:there;
    Rig.Buffer.copy ~src:there ~dst:back;
    if
      Rig.Buffer.bigarray Bigarray.char back
      <> Rig.Buffer.bigarray Bigarray.char here
    then failwith "the bytes did not come back"
  in
  List.iter check hosts;
  Printf.printf "copied %d\n%!" (List.length hosts);
  Rig_remote.close j

let names () =
  Sys.getenv "RIG_REMOTE_AGENTS"
  |> String.split_on_char ','
  |> List.map (fun a -> List.hd (String.split_on_char '=' a))
  |> String.concat " "

let gone v = Sys.getenv_opt v = None || Sys.getenv_opt v = Some ""
let kill_self () = Unix.kill (Unix.getpid ()) Sys.sigkill

let () =
  let n = attempt () in
  match List.tl (Array.to_list Sys.argv) with
  | [ "copy" ] -> copy (job ())
  | [ "exit"; s ] ->
      ignore (job ());
      exit (int_of_string s)
  | [ "raise" ] ->
      ignore (job ());
      failwith "ctl raised"
  | [ "env" ] ->
      Printf.printf "agents %s, key of %d characters\n%!" (names ())
        (String.length (Sys.getenv "RIG_REMOTE_KEY"));
      let j = job () in
      let vars =
        [ "RIG_REMOTE_AGENTS"; "RIG_REMOTE_KEY"; "RIG_REMOTE_REPORT" ]
      in
      Printf.printf "variables gone: %b\n%!" (List.for_all gone vars);
      Rig_remote.close j
  | [ "early"; s ] -> exit (int_of_string s)
  | [ "kill" ] ->
      ignore (job ());
      kill_self ()
  | [ "kill-first" ] ->
      let j = job () in
      if n = 1 then kill_self () else copy j
  | [ "wait-first" ] ->
      let j = job () in
      if n > 1 then copy j
      else begin
        while Rig_remote.failure j = None do
          Unix.sleepf 0.05
        done;
        failwith "the job failed"
      end
  | _ ->
      prerr_endline "usage: ctl.exe MODE";
      exit 1
