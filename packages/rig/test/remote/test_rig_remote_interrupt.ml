(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of rig.remote's controller. A call is made once for each
   allocation it makes, with a landing at that allocation on the calling
   thread: another call from a Gc.Memprof callback, an exit from one, or an
   exception. The callback's call answers as it would outside the callback,
   no device is lost, and a launched job's report ends as the job does. *)

open Windtrap
open Remote_job

let strf = Printf.sprintf

(* [walk attempt] runs [attempt n] from [n = 1] until an attempt no longer
   reaches its [n]-th allocation. *)
let walk attempt =
  let rec from n = if attempt n then from (n + 1) in
  from 1

let names ds = List.map Rig.name ds

(* Devices *)

(* A devices of MEM from a callback in a devices of NONE, a kind whose opener
   fails, so that each call asks the agent: both answer, the host is not lost,
   and MEM gives the devices it gave first. *)
let devices_in_devices () =
  with_job @@ fun j agents ->
  let h = List.hd (Rig_remote.hosts j) in
  let at = at (List.hd agents) (Rig.name h) in
  let first = ref [] in
  walk (fun n ->
      let outer = ref (Ok []) and inner = ref None in
      let call () = inner := Some (Rig_remote.devices h "MEM") in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "NONE")
      in
      let msg what = strf "from allocation %d, %s" n what in
      equal ~msg:(msg "NONE")
        (result (list string) string)
        (Error (at ^ ": no such hardware"))
        (Result.map names !outer);
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      let ds = mem h in
      if !first = [] then begin
        equal ~msg:(msg "MEM") (list string)
          [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
          (names ds);
        first := ds
      end;
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal !first ds'));
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal !first ds);
      reached);
  List.iter
    (fun d ->
      let s = "on " ^ Rig.name d in
      equal ~msg:"a copy there and back" string s (read (far_of_string d s)))
    !first

(* A first devices of MEM from a callback in the first devices of MEM, in a job
   of its own: both give the same devices, or the callback's raises
   Invalid_argument where it opens a device that the interrupted call is
   opening. The host is not lost, and the devices serve copies. *)
let first_in_first () =
  let refusals = ref 0 in
  walk (fun n ->
      with_job @@ fun j agents ->
      let h = List.hd (Rig_remote.hosts j) in
      let at = at (List.hd agents) (Rig.name h) in
      let outer = ref (Ok []) and inner = ref None in
      let call () =
        match Rig_remote.devices h "MEM" with
        | r -> inner := Some r
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "MEM")
      in
      let msg what = strf "from allocation %d, %s" n what in
      let ds =
        match !outer with
        | Ok ds -> ds
        | Error why -> failf "%s: %s" (msg "MEM") why
      in
      equal ~msg:(msg "MEM") (list string)
        [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
        (names ds);
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal ds ds'));
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal ds (mem h));
      List.iter
        (fun d ->
          let s = "on " ^ Rig.name d in
          equal
            ~msg:(msg "a copy there and back")
            string s
            (read (far_of_string d s)))
        ds;
      reached);
  greater ~msg:"devices refused" int ~than:0 !refusals

(* A first devices of MEM from a callback in the first devices of MEM, which
   closes the last device it gets, in a job of its own: both give the same
   devices, the last closed, or the callback's raises Invalid_argument where it
   opens a device that the interrupted call is opening. The host is not lost,
   and the first device serves copies. *)
let close_in_first () =
  let refusals = ref 0 and closes = ref 0 in
  walk (fun n ->
      with_job @@ fun j agents ->
      let h = List.hd (Rig_remote.hosts j) in
      let at = at (List.hd agents) (Rig.name h) in
      let outer = ref (Ok []) and inner = ref None in
      let call () =
        match Rig_remote.devices h "MEM" with
        | Ok ds as r ->
            Rig.close (List.nth ds (List.length ds - 1));
            incr closes;
            inner := Some r
        | r -> inner := Some r
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "MEM")
      in
      let msg what = strf "from allocation %d, %s" n what in
      let ds =
        match !outer with
        | Ok ds -> ds
        | Error why -> failf "%s: %s" (msg "MEM") why
      in
      equal ~msg:(msg "MEM") (list string)
        [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
        (names ds);
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal ds ds');
          equal ~msg:(msg "the losses")
            (list (option string))
            [ None; Some "closed" ] (List.map Rig.lost ds));
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal ds (mem h));
      let d = List.hd ds in
      let s = "on " ^ Rig.name d in
      equal
        ~msg:(msg "a copy there and back")
        string s
        (read (far_of_string d s));
      reached);
  greater ~msg:"devices refused" int ~than:0 !refusals;
  greater ~msg:"devices closed" int ~than:0 !closes

(* Launched jobs *)

(* Runs support/launched.exe in [mode] with a landing at each allocation of
   the close of its launched job, whose hosts are closed, one run for each
   from the first on, until [check msg code report words agent] is [false]:
   [msg] names the run, [code] is its exit code, [words] what it prints after
   its hosts' names, and [agent] its agent's exit code and lines. *)
let close_walk mode check =
  walk (fun n ->
      with_hex_agents @@ fun agents ->
      let args = [ string_of_int n ] in
      let code, report, said = ended (launch ~args (launch_vars agents) mode) in
      let msg what = strf "from allocation %d, %s" n what in
      let words =
        match said with
        | "connected" :: "CPU@m1" :: words -> words
        | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " said)
      in
      check msg code report words (finish (List.hd agents)))

(* The report and the agent's end of a job that closed. *)
let closed msg code report agent =
  equal
    ~msg:(msg "its exit and report")
    (pair int (list string))
    (0, [ "started"; "closed" ])
    (code, report);
  equal ~msg:(msg "the agent's end") exit_w (0, [ "closed" ]) agent

(* "reached" or "past", the last of a run's words, and whether it reached the
   allocation. *)
let reached msg words =
  match words with
  | [ "reached" ] -> true
  | [ "past" ] -> false
  | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " words)

(* A close from the callback: both return, and the job closed. *)
let close_in_close () =
  close_walk "reenter" (fun msg code report words agent ->
      closed msg code report agent;
      reached msg words)

(* An exception at the allocation and each one after it, then an exit, which
   closes the job again: the job closed. *)
let raise_in_close () =
  close_walk "raise-in-close" (fun msg code report words agent ->
      closed msg code report agent;
      reached msg words)

(* An exit from the callback, whose close at exit runs inside the close: the
   program exits 0, and the job closed, or failed with the reason rig
   refused that close for, at the agent too. Both are reached. *)
let exit_in_close () =
  let failures = ref 0 and closes = ref 0 in
  close_walk "exit-in-close" (fun msg code report words agent ->
      (match report with
      | [ "started"; "closed" ] ->
          incr closes;
          closed msg code report agent
      | [ "started"; f ] when String.starts_with ~prefix:"failed " f ->
          incr failures;
          let why = String.sub f 7 (String.length f - 7) in
          equal
            ~msg:(msg "its exit and the agent's end")
            (pair int exit_w)
            (0, (2, [ "failed: " ^ why ]))
            (code, agent)
      | _ ->
          failf "%s: [%s]" (msg "its report") (String.concat "; " report));
      match words with
      | [] -> true
      | [ "past" ] -> false
      | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " words));
  greater ~msg:"closes" int ~than:0 !closes;
  greater ~msg:"failures" int ~than:0 !failures

let () =
  Watchdog.start ();
  exit
    (run "rig_remote interrupt"
       [
         group ~timeout:60. "a call from a callback at any allocation"
           [
             test "a devices in a devices answers" devices_in_devices;
             test "a first devices in a first devices of its kind answers"
               first_in_first;
             test
               "a first devices in a first devices that closes a device answers"
               close_in_first;
             test "a close in a close returns" close_in_close;
             test "an exit in a close reports the job's end" exit_in_close;
             test "an exception in a close reports the close at the exit"
               raise_in_close;
           ];
       ])
