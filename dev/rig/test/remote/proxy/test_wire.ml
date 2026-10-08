(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The handshake between two ends of a loopback connection. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire

let timeout = 30.

(* A connected pair of loopback TCP sockets: the dialing end and the accepting
   end. *)
let connected () =
  let l = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  let port =
    match Unix.getsockname l with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  let d = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect d (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
  let a, _ = Unix.accept l in
  Unix.close l;
  (d, a)

(* Runs both ends' handshakes at once, the accepting end on a thread. *)
let handshake ~dialing_key ~accepting_key ~admit =
  let d, a = connected () in
  let accepted = ref None in
  let t =
    Thread.create
      (fun () -> accepted := Some (Wire.accept a ~key:accepting_key ~admit))
      ()
  in
  let dialed =
    Wire.dial d ~key:dialing_key ~self:Wire.Controller ~peer:(Wire.Agent 1)
  in
  Thread.join t;
  Unix.close d;
  Unix.close a;
  (dialed, Option.get !accepted)

let key = String.make 32 'k'

let process =
  Testable.structural ~pp:(fun ppf -> function
    | Wire.Controller -> Format.pp_print_string ppf "controller"
    | Wire.Agent i -> Format.fprintf ppf "agent %d" i)

let handshakes =
  group ~timeout "handshake"
    [
      test "both ends that hold the key admit each other" (fun () ->
          let dialed, accepted =
            handshake ~dialing_key:key ~accepting_key:key ~admit:(fun _ ->
                Ok ())
          in
          equal (result unit string) (Ok ()) dialed;
          equal
            (result (pair process process) string)
            (Ok (Wire.Controller, Wire.Agent 1))
            accepted);
      test "an end with another key is refused" (fun () ->
          let dialed, accepted =
            handshake ~dialing_key:key ~accepting_key:(String.make 32 'x')
              ~admit:(fun _ -> Ok ())
          in
          equal (result unit string)
            (Error "the dialing end does not know the job's key") dialed;
          equal
            (result (pair process process) string)
            (Error "the dialing end does not know the job's key") accepted);
    ]

let () = exit (run "rig_remote_proxy.wire" [ handshakes ])
