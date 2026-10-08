(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A peer of the Link suite in a process of its own: an agent's end of one link
   to the test's process, which listens on the loopback port it is given. In
   mode serve it answers requests, prints each other command and closes the
   job when the controller does; in mode idle it waits for the job to end.
   Either prints the job's last state, "closed" or "failed: WHY". *)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

(* The suite's reference states the same answers. *)
let answer : type a. a Wire.request -> (a, string) result = function
  | Wire.Open kind ->
      Ok
        [
          { Wire.id = 0; name = kind; arch = "echo"; budget = 0; reaches = [] };
        ]
  | Wire.Entry { image; name } -> Ok (if name = "" then None else Some image)
  | Wire.Alloc { bytes; _ } -> Ok (bytes mod 2 = 0)
  | _ -> Error "the peer answers Open, Entry and Alloc"

let rec serve j l =
  match Link.next l with
  | Ok (Wire.Request r) ->
      Link.answer l r (answer r);
      serve j l
  | Ok (Wire.Drop id) ->
      Printf.printf "drop %d\n%!" id;
      serve j l
  | Ok (Wire.Handover _) ->
      print_endline "handover";
      serve j l
  | Ok Wire.Close ->
      print_endline "close";
      Link.close j
  | Error _ -> ()

let rec idle j = match Link.wait j ~ms:1000 with Link.Open -> idle j | _ -> ()

let () =
  let mode = Sys.argv.(1) and port = int_of_string Sys.argv.(2) in
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
  let j = Link.job () in
  let l = Link.make j fd ~name:"controller" ~peer:Wire.Controller in
  (match mode with
  | "serve" -> serve j l
  | "idle" -> idle j
  | m -> invalid_arg ("link_peer: no mode " ^ m));
  match Link.wait j ~ms:0 with
  | Link.Open -> print_endline "open"
  | Link.Closed -> print_endline "closed"
  | Link.Failed why -> Printf.printf "failed: %s\n%!" why
