(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A controller and a real agent process over loopback. *)

open Windtrap
module B = Rig.Buffer

let timeout = 60.

(* A key in a file of this test's own, readable by its user alone. *)
let key_file () =
  let file = "test_rig_remote.key" in
  (try Sys.remove file with Sys_error _ -> ());
  let oc = open_out_gen [ Open_wronly; Open_creat; Open_trunc ] 0o600 file in
  output_string oc (String.init 32 (fun i -> Char.chr (65 + i)));
  close_out oc;
  file

(* Starts an agent and is its process and port. *)
let agent key =
  let out, w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process "support/agent.exe"
      [| "support/agent.exe"; key |]
      Unix.stdin w Unix.stderr
  in
  Unix.close w;
  let ic = Unix.in_channel_of_descr out in
  let port = int_of_string (input_line ic) in
  close_in ic;
  (pid, port)

let exit_code pid =
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED n -> n
  | Unix.WSIGNALED n | Unix.WSTOPPED n -> -n

let round_trip () =
  let file = key_file () in
  let key = Result.get_ok (Rig_remote.read_key file) in
  let pid, port = agent file in
  match Rig_remote.connect ~key [ ("127.0.0.1", port) ] with
  | Error why -> fail why
  | Ok j ->
      let h = List.hd (Rig_remote.hosts j) in
      let mem =
        match Rig_remote.devices h "MEM" with
        | Ok ds -> ds
        | Error why -> fail why
      in
      equal (list string)
        [
          Printf.sprintf "MEM:0@127.0.0.1:%d" port;
          Printf.sprintf "MEM:1@127.0.0.1:%d" port;
        ]
        (List.map Rig.name mem);
      let src = B.of_string "to an agent and back" in
      let far = B.create (List.hd mem) (B.length src) in
      let back = B.create Rig.host (B.length src) in
      B.copy ~src ~dst:far;
      B.copy ~src:far ~dst:back;
      equal string "to an agent and back"
        (let ba = B.bigarray Bigarray.char back in
         String.init (B.length back) (Bigarray.Array1.get ba));
      Rig_remote.close j;
      equal (option string) None (Rig_remote.failure j);
      equal int 0 (exit_code pid)

(* Two agents connect to each other, each serves its machine, and both end with
   the job. *)
let two_agents () =
  let file = key_file () in
  let key = Result.get_ok (Rig_remote.read_key file) in
  let p1, port1 = agent file and p2, port2 = agent file in
  match
    Rig_remote.connect ~key [ ("127.0.0.1", port1); ("127.0.0.1", port2) ]
  with
  | Error why -> fail why
  | Ok j ->
      List.iteri
        (fun i h ->
          let d = List.hd (Result.get_ok (Rig_remote.devices h "MEM")) in
          let s = Printf.sprintf "machine %d" i in
          let far = B.create d (String.length s) in
          let back = B.create Rig.host (String.length s) in
          B.copy ~src:(B.of_string s) ~dst:far;
          B.copy ~src:far ~dst:back;
          equal string s
            (let ba = B.bigarray Bigarray.char back in
             String.init (B.length back) (Bigarray.Array1.get ba)))
        (Rig_remote.hosts j);
      Rig_remote.close j;
      equal (list int) [ 0; 0 ] [ exit_code p1; exit_code p2 ]

let jobs =
  group ~timeout "job"
    [
      test "bytes reach an agent's device and come back, then the job closes"
        round_trip;
      test "two agents serve one job and end with it" two_agents;
    ]

let () = exit (run "rig_remote" [ jobs ])
