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

let variable = "RIG_BNXT"

(* The NIC RIG_BNXT numbers. *)
let index () =
  match Sys.getenv_opt variable with
  | None -> skip ~reason:"needs a detached BCM57608 (RIG_BNXT)" ()
  | Some s -> (
      match int_of_string_opt s with
      | Some i -> i
      | None -> failf "%s=%S numbers no NIC" variable s)

let open_nic () =
  let i = index () in
  match B.open_ i with
  | Ok nic -> nic
  | Error e -> failf "NIC %d did not open: %s" i e

(* [f nic] on the NIC RIG_BNXT numbers, closed afterwards. *)
let with_nic f =
  let nic = open_nic () in
  Fun.protect ~finally:(fun () -> B.close nic) (fun () -> f nic)

let ok what = function Ok v -> v | Error e -> failf "%s: %s" what e

(* Two queue pairs of [nic] connected to each other. *)
let pair nic =
  let a = ok "a queue pair" (B.Qp.make nic) in
  let b = ok "a queue pair" (B.Qp.make nic) in
  ok "connecting" (B.Qp.connect a (B.Qp.endpoint b));
  ok "connecting" (B.Qp.connect b (B.Qp.endpoint a));
  (a, b)

let bound = 5_000_000

let completion qp side =
  match B.Qp.wait qp side ~us:bound with
  | Some r -> r
  | None -> fail "no completion within 5 s"

let wait qp side = ok "the completion" (completion qp side)
let region nic n = ok "a region" (B.Region.host nic n)

let pattern n =
  String.init n (fun i -> Char.chr (((i * 31) + (i lsr 8)) land 0xff))

(* The message of the [Invalid_argument] that [f ()] raises while the calling
   thread holds one more lock, as a callback inside one does. *)
let inside_lock f =
  Rig_support.inside_lock @@ fun () ->
  match f () with
  | () -> fail "the call ran inside a lock"
  | exception Invalid_argument why -> why

let test_inside_lock () =
  let before = Rig_support.holds () in
  List.iter
    (fun (what, f) ->
      starts_with ~msg:what
        ~affix:(Printf.sprintf "Rig_bnxt.%s: called from" what)
        (inside_lock f))
    [
      ("open_", fun () -> ignore (B.open_ (B.count ())));
      ("detach", fun () -> ignore (B.detach 0));
      ("attach", fun () -> ignore (B.attach 0));
    ];
  equal ~msg:"locks held" int before (Rig_support.holds ());
  match B.open_ (B.count ()) with
  | Ok _ -> fail "opened a NIC past the last"
  | Error e -> contains ~msg:"an open outside a lock" ~sub:"no such NIC" e

let refusals =
  group "refusals"
    [
      test
        "an open, a detach and an attach from inside a lock raise \
         Invalid_argument as they start, and an open after runs"
        test_inside_lock;
      test "a NIC past the last is an error naming it" (fun () ->
          let i = B.count () in
          let name = if i = 0 then "BNXT" else Printf.sprintf "BNXT:%d" i in
          match B.open_ i with
          | Ok _ -> fail "opened a NIC past the last"
          | Error e ->
              equal string ~msg:"the message's start" name
                (String.sub e 0 (min (String.length e) (String.length name)));
              contains ~sub:"no such NIC" e);
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
          let src = region nic n and dst = region nic n in
          B.Region.write src 0 (pattern n);
          B.Qp.receive b dst ~at:0 n;
          B.Qp.send a src ~at:0 n;
          equal int ~msg:"bytes sent" n (wait a Send);
          equal int ~msg:"bytes received" n (wait b Receive);
          equal string (pattern n) (B.Region.read dst 0 n));
      test "a receive reports the length of a shorter message" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let r = region nic 4096 in
          B.Qp.receive b r ~at:0 64;
          B.Qp.send a r ~at:0 5;
          equal int ~msg:"bytes sent" 5 (wait a Send);
          equal int ~msg:"bytes received" 5 (wait b Receive));
      test "sends land in the order of the receives" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let src = region nic 4096 and dst = region nic 4096 in
          B.Region.write src 0 "abcd";
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
          equal string "a\000\000\000\000\000\000\000b" (B.Region.read dst 0 9);
          equal string "c" (B.Region.read dst 16 1);
          equal string "d" (B.Region.read dst 24 1));
      test "a send longer than its receive fails the queue pairs" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let r = region nic 4096 in
          B.Qp.receive b r ~at:0 8;
          B.Qp.receive b r ~at:0 8;
          B.Qp.send a r ~at:0 16;
          is_error ~msg:"the receive" (completion b Receive);
          is_error ~msg:"the send" (completion a Send);
          is_error ~msg:"the receive behind it" (completion b Receive);
          B.Qp.send a r ~at:0 1;
          is_error ~msg:"a send posted after" (completion a Send));
      test "a wait at its bound leaves the entry due" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          let r = region nic 4096 in
          B.Qp.receive b r ~at:0 1;
          equal (option (result int string)) None (B.Qp.wait b Receive ~us:1000);
          equal int ~msg:"room" 4094 (B.Qp.room b Receive);
          B.Qp.send a r ~at:0 1;
          equal int ~msg:"bytes received" 1 (wait b Receive));
      test "a ring takes 4095 entries and refuses one more" (fun () ->
          with_nic @@ fun nic ->
          let _, b = pair nic in
          let r = region nic 4096 in
          equal int ~msg:"room" 4095 (B.Qp.room b Receive);
          for _ = 1 to 4095 do
            B.Qp.receive b r ~at:0 1
          done;
          equal int 0 (B.Qp.room b Receive);
          raises_match (Exn.invalid_arg ~substring:"full") (fun () ->
              B.Qp.receive b r ~at:0 1));
      test "a destroyed queue pair refuses every call" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r = region nic 4096 in
          ok "destroying" (B.Qp.destroy a);
          ok "a second destroy" (B.Qp.destroy a);
          raises_match (Exn.invalid_arg ~substring:"destroyed") (fun () ->
              B.Qp.send a r ~at:0 1));
    ]

(* From two domains: a transfer one way runs while one the other way does. Each
   way holds its own lock, so that a side runs one call at a time. *)
let two_domains =
  let name = "transfers both ways from two domains each land their byte" in
  match Sys.getenv_opt variable with
  | None -> test name (fun () -> ignore (index ()))
  | Some _ ->
      let way =
        lazy
          (let nic = open_nic () in
           let a, b = pair nic in
           let way src dst =
             (src, dst, region nic 4096, region nic 4096, Mutex.create ())
           in
           (way a b, way b a))
      in
      let transfer pick c =
        let src, dst, s, d, lock = pick (Lazy.force way) in
        Mutex.protect lock @@ fun () ->
        B.Region.write s 0 (String.make 1 (Char.chr c));
        B.Qp.receive dst d ~at:0 1;
        B.Qp.send src s ~at:0 1;
        ignore (wait src Send);
        ignore (wait dst Receive);
        Char.code (B.Region.read d 0 1).[0]
      in
      let command' label pick =
        command label
          (Gen.int_range 0 255 @-> returns int)
          Fun.id (transfer pick)
      in
      stateful ~domains:2 ~count:20 name
        [ command' "a to b" fst; command' "b to a" snd ]

let misuse =
  group "misuse"
    [
      test "a post outside its region raises" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r = region nic 1 in
          let n = B.Region.length r in
          raises_match (Exn.invalid_arg ~substring:"bytes") (fun () ->
              B.Qp.send a r ~at:1 n);
          raises_match (Exn.invalid_arg ~substring:"bytes") (fun () ->
              B.Qp.send a r ~at:(-1) 1);
          raises_match (Exn.invalid_arg ~substring:"bytes") (fun () ->
              ignore (B.Region.read r 1 n)));
      test "a post on an unconnected queue pair raises" (fun () ->
          with_nic @@ fun nic ->
          let qp = ok "a queue pair" (B.Qp.make nic) in
          let r = region nic 4096 in
          raises_match (Exn.invalid_arg ~substring:"not connected") (fun () ->
              B.Qp.send qp r ~at:0 1));
      test "a second connect raises" (fun () ->
          with_nic @@ fun nic ->
          let a, b = pair nic in
          raises_match (Exn.invalid_arg ~substring:"connected") (fun () ->
              ignore (B.Qp.connect a (B.Qp.endpoint b))));
      test "a deregistered region refuses posts, reads and writes" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r = region nic 4096 in
          ok "deregistering" (B.Region.deregister r);
          ok "a second deregister" (B.Region.deregister r);
          raises_match (Exn.invalid_arg ~substring:"deregistered") (fun () ->
              B.Qp.send a r ~at:0 1);
          raises_match (Exn.invalid_arg ~substring:"deregistered") (fun () ->
              ignore (B.Region.read r 0 1));
          raises_match (Exn.invalid_arg ~substring:"deregistered") (fun () ->
              B.Region.write r 0 "x"));
      test "a wait with nothing due raises" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          equal (option (result int string)) None (B.Qp.poll a Send);
          raises_match (Exn.invalid_arg ~substring:"due") (fun () ->
              ignore (B.Qp.wait a Send ~us:0)));
      test "a closed NIC refuses every call" (fun () ->
          with_nic @@ fun nic ->
          let a, _ = pair nic in
          let r = region nic 4096 in
          B.close nic;
          B.close nic;
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              ignore (B.Qp.make nic));
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              B.Qp.send a r ~at:0 1);
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              ignore (B.Region.read r 0 1));
          raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
              ignore (B.Region.deregister r)));
    ]

let () =
  exit
    (run "rig_bnxt"
       [ refusals; transfers; group "domains" [ two_domains ]; misuse ])
