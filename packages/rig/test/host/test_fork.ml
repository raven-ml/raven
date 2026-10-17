(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A forked child of a process whose host ran launches on the pool's workers:
   the host is never lost, so the child runs host code as its parent did. Its
   own suite: a process that ran a domain cannot fork. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Run = Sub.Run
module S = Rig_host_support

let timeout = 60.

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> "killed by " ^ Sys.signal_to_string n
  | Unix.WSTOPPED n -> "stopped by " ^ Sys.signal_to_string n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try child () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let lines = In_channel.input_lines ic in
      close_in ic;
      let _, status = Unix.waitpid [] pid in
      (lines, status)

let load () =
  Rig.Image.load Rig.host (S.fixture ~dir:"../fixtures" "affine")
  |> Result.get_ok

(* Whether a launch of affine.c over [n] words, in groups of 64 threads, writes
   [a * i + c] at each [i]: enough groups that the pool's workers run some. *)
let affine image ~n ~a ~c =
  let inb = S.words n and out = S.words n in
  for i = 0 to n - 1 do
    inb.{i} <- Int64.of_int i
  done;
  let refs = Sub.[| write 0 ~at:0; read 1 ~at:8 |] in
  let fn = require_some (Rig.Image.resolve image "affine") in
  let s = Sub.make Rig.host [| Sub.launch ~refs fn ~params:40 |] in
  let run = Run.make s and k = Sub.block s 0 in
  Run.groups run k ((n + 63) / 64) 1 1;
  Run.threads run k 64 1 1;
  Run.int64 run k 16 n;
  Run.int64 run k 24 a;
  Run.int64 run k 32 c;
  ignore
    (Rig.submit run ~buffers:[| B.of_bigarray out; B.of_bigarray inb |]);
  let right = ref true in
  for i = 0 to n - 1 do
    if out.{i} <> Int64.of_int ((a * i) + c) then right := false
  done;
  if !right then "right" else "wrong"

let n = 1 lsl 16

let test_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let image = load () in
  equal ~msg:"the parent's launch" string "right" (affine image ~n ~a:3 ~c:1);
  let lines, ended =
    in_child (fun () ->
        [
          Option.value ~default:"not lost" (Rig.lost Rig.host);
          affine image ~n ~a:5 ~c:2;
          affine (load ()) ~n ~a:7 ~c:3;
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "not lost"; "right"; "right" ] lines;
  equal ~msg:"the parent's launch after the fork" string "right"
    (affine image ~n ~a:11 ~c:4)

let () =
  exit
    (run "rig_host.fork"
       [
         group ~timeout "fork"
           [
             test "a forked child runs host code, its parent's and its own"
               test_child;
           ];
       ])
