(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NICs of this machine: what the library refuses on any machine, and two queue
   pairs of one NIC connected to each other and the transfers between their
   regions. Those run, as root, on the detached BCM57608 the variable RIG_BNXT
   numbers, and skip without it. *)

open Windtrap
module B = Rig_bnxt
module W = Rig_pci.Window

(* [f nic] on the NIC RIG_BNXT numbers, closed afterwards. *)
let with_nic f =
  let i =
    match Option.bind (Sys.getenv_opt "RIG_BNXT") int_of_string_opt with
    | Some i -> i
    | None -> skip ~reason:"needs a detached BCM57608 (RIG_BNXT)" ()
  in
  match B.open_ i with
  | Error e -> failf "NIC %d did not open: %s" i e
  | Ok nic -> Fun.protect ~finally:(fun () -> B.close nic) (fun () -> f nic)

let ok what = function Ok v -> v | Error e -> failf "%s: %s" what e

(* Two queue pairs of [nic] connected to each other. *)
let pair nic =
  let a = ok "a queue pair" (B.Qp.make nic) in
  let b = ok "a queue pair" (B.Qp.make nic) in
  ok "connecting" (B.Qp.connect a (B.Qp.endpoint b));
  ok "connecting" (B.Qp.connect b (B.Qp.endpoint a));
  (a, b)

let wait qp side = ok "the completion" (B.Qp.wait qp side ~us:5_000_000)

let pattern n =
  String.init n (fun i -> Char.chr (((i * 31) + (i lsr 8)) land 0xff))

let refusals =
  group "refusals"
    [
      test "a NIC past the last is an error naming it" (fun () ->
          let i = B.count () in
          let name = if i = 0 then "BNXT" else Printf.sprintf "BNXT:%d" i in
          match B.open_ i with
          | Ok _ -> fail "opened a NIC past the last"
          | Error e ->
              equal string ~msg:"the message's start" name
                (String.sub e 0 (min (String.length e) (String.length name))));
      test "a negative index raises" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
              ignore (B.open_ (-1)));
          raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
              ignore (B.detach (-1)));
          raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
              ignore (B.attach (-1))));
    ]

let transfers =
  group "transfers"
    [
      test "a send lands its bytes in the peer's receive" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let n = (3 * 4096) + 17 in
          let src, ws = ok "a region" (B.Region.host nic n) in
          let dst, wd = ok "a region" (B.Region.host nic n) in
          W.write ws 0 (pattern n);
          B.Qp.receive b dst ~at:0 n;
          B.Qp.send a src ~at:0 n;
          equal int ~msg:"bytes sent" n (wait a Send);
          equal int ~msg:"bytes received" n (wait b Receive);
          equal string (pattern n) (W.read wd 0 n));
      test "a receive reports the length of a shorter message" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          B.Qp.receive b r ~at:0 64;
          B.Qp.send a r ~at:0 5;
          equal int ~msg:"bytes sent" 5 (wait a Send);
          equal int ~msg:"bytes received" 5 (wait b Receive));
      test "sends land in the order of the receives" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let src, ws = ok "a region" (B.Region.host nic 4096) in
          let dst, wd = ok "a region" (B.Region.host nic 4096) in
          W.write ws 0 "abcd";
          for i = 0 to 3 do
            B.Qp.receive b dst ~at:(i * 8) 1
          done;
          for i = 0 to 3 do
            B.Qp.send a src ~at:i 1
          done;
          for _ = 0 to 3 do
            ignore (wait a Send);
            ignore (wait b Receive)
          done;
          equal string "a\000\000\000\000\000\000\000b" (W.read wd 0 9);
          equal string "c" (W.read wd 16 1);
          equal string "d" (W.read wd 24 1));
      test "a send longer than its receive fails the queue pairs" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          B.Qp.receive b r ~at:0 8;
          B.Qp.receive b r ~at:0 8;
          B.Qp.send a r ~at:0 16;
          is_error ~msg:"the receive" (B.Qp.wait b Receive ~us:5_000_000);
          is_error ~msg:"the send" (B.Qp.wait a Send ~us:5_000_000);
          is_error ~msg:"the receive behind it"
            (B.Qp.wait b Receive ~us:5_000_000);
          B.Qp.send a r ~at:0 1;
          is_error ~msg:"a send posted after" (B.Qp.wait a Send ~us:5_000_000));
      test "a full ring refuses a post" (fun () ->
          with_nic @@ fun nic ->
          let _, b = pair nic in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          for _ = 1 to B.Qp.room b Receive do
            B.Qp.receive b r ~at:0 1
          done;
          equal int 0 (B.Qp.room b Receive);
          raises_match (Exn.invalid_arg ~substring:"full") (fun () ->
              B.Qp.receive b r ~at:0 1));
    ]

let misuse =
  group "misuse"
    [
      test "a post outside its region raises" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r, w = ok "a region" (B.Region.host nic 1) in
          let n = W.length w in
          raises_match (Exn.invalid_arg ~substring:"bytes") (fun () ->
              B.Qp.send a r ~at:1 n);
          raises_match (Exn.invalid_arg ~substring:"bytes") (fun () ->
              B.Qp.send a r ~at:(-1) 1));
      test "a post on an unconnected queue pair raises" (fun () ->
          with_nic @@ fun nic ->
          let qp = ok "a queue pair" (B.Qp.make nic) in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          raises_match (Exn.invalid_arg ~substring:"not connected") (fun () ->
              B.Qp.send qp r ~at:0 1));
      test "a deregistered region refuses posts" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          ok "deregistering" (B.Region.deregister r);
          ok "a second deregister" (B.Region.deregister r);
          raises_match (Exn.invalid_arg ~substring:"deregistered") (fun () ->
              B.Qp.send a r ~at:0 1));
      test "a wait with nothing due raises" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          equal (option (result int string)) None (B.Qp.poll a Send);
          raises_match (Exn.invalid_arg ~substring:"due") (fun () ->
              ignore (B.Qp.wait a Send ~us:0)));
      test "a closed NIC refuses every call" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r, _ = ok "a region" (B.Region.host nic 4096) in
          B.close nic;
          B.close nic;
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              ignore (B.Qp.make nic));
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              B.Qp.send a r ~at:0 1);
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              ignore (B.Region.deregister r)));
    ]

let () = exit (run "rig_bnxt" [ refusals; transfers; misuse ])
