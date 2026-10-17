(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Another machine's devices.

   A job is one process, the controller, and an agent on each other machine,
   which opens its machine's devices and runs the work the controller
   submits to them. The controller uses each such device through a proxy,
   a device of its own whose memory is that device's, with the same calls
   as a device of its own machine.

   Here the program starts itself a second time as its agent, which serves a
   memory device "M" on the loopback, so the job needs no second machine. *)

open Rig

(* The job's key. This agent listens only on the loopback. A job across a
   network reads a random key from a file ([Rig_remote.read_key]). *)
let key = Result.get_ok (Rig_remote.key (String.make 32 'k'))

(* [a]'s elements, joined. *)
let ints a =
  String.concat "; "
    (List.init (Bigarray.Array1.dim a) (fun i -> Int32.to_string a.{i}))

(* A bigarray of [l]. *)
let int32s l = Bigarray.(Array1.of_array int32 c_layout (Array.of_list l))

(* [main.exe agent] listens on the loopback at a port the system chooses,
   prints the port, and serves one job. *)
let agent () =
  let a = Result.get_ok (Rig_remote.listen ~key "127.0.0.1" 0) in
  Printf.printf "%d\n%!" (Rig_remote.port a);
  let open_ () = Result.map (fun d -> [ d ]) (memory_device "M") in
  match Rig_remote.serve a [ ("M", open_) ] with
  | Ok () -> ()
  | Error why ->
      prerr_endline why;
      exit 2

let controller () =
  let exe = Sys.executable_name in
  let ic = Unix.open_process_args_in exe [| exe; "agent" |] in
  let port = int_of_string (String.trim (input_line ic)) in
  let j = Result.get_ok (Rig_remote.connect ~key [ ("127.0.0.1", port) ]) in

  (* The agent's machine has a host, and the devices its agent serves. *)
  let h = List.hd (Rig_remote.hosts j) in
  let m = List.hd (Result.get_ok (Rig_remote.devices h "M")) in
  Printf.printf "M's host is the agent's: %b, of this host's arch: %b\n"
    (equal (host_of m) h)
    (arch h = arch host);

  (* The bytes cross to the agent once, and M's copy runs there. *)
  let x = Buffer.create m 16 and y = Buffer.create m 16 in
  Buffer.copy ~src:(Buffer.of_bigarray (int32s [ 1l; 2l; 3l; 4l ])) ~dst:x;
  let s = Submission.(make m [| copy ~src:(Buffer x) ~dst:(Buffer y) |]) in
  let run = Submission.Run.make () in
  ignore (submit s ~run ~buffers:[||]);

  (* The copy back reads [y], so it follows M's copy, which writes it. *)
  let z = int32s [ 0l; 0l; 0l; 0l ] in
  Buffer.copy ~src:y ~dst:(Buffer.of_bigarray z);
  Printf.printf "there and back: [%s]\n" (ints z);

  (* Closing the job ends the agent's [serve]. *)
  Rig_remote.close j;
  match Unix.close_process_in ic with
  | WEXITED 0 -> print_endline "the agent has stopped"
  | _ -> print_endline "the agent failed"

let () =
  match Sys.argv with
  | [| _; "agent" |] -> agent ()
  | _ -> controller ()
