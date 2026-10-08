(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Links of one job over loopback: the controller's end and an agent's end in
   this process, connected to each other. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

let timeout = 30.

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

(* A job whose controller's end [c] and agent's end [a] talk to each other. *)
let with_job f =
  let j = Link.job () in
  let d, a = connected () in
  let c = Link.make j d ~name:"agent" ~peer:(Wire.Agent 1) in
  let a = Link.make j a ~name:"controller" ~peer:Wire.Controller in
  Fun.protect
    ~finally:(fun () -> if Link.failure j = None then Link.fail j "test ended")
    (fun () -> f j c a)

let account : Wire.account =
  { id = 1; name = "MEM:0"; arch = "arm64"; budget = 4096; reaches = [ 2 ] }

type answer = { answer : 'r. 'r Wire.request -> ('r, string) result }

(* Answers [n] requests on the agent's end, each with [answer]. *)
let serve a n { answer } =
  Thread.create
    (fun () ->
      for _ = 1 to n do
        match Link.next a with
        | Ok (Wire.Request r) -> Link.answer a r (answer r)
        | Ok _ | Error _ -> ()
      done)
    ()

let get_u64 b at =
  let v = ref 0 in
  for i = 7 downto 0 do
    v := (!v lsl 8) lor Char.code b.{at + i}
  done;
  !v

let set_u64 b at v =
  for i = 0 to 7 do
    b.{at + i} <- Char.chr ((v lsr (8 * i)) land 0xff)
  done

let requests =
  group ~timeout "request"
    [
      test "an agent's answer is the request's" (fun () ->
          with_job @@ fun _ c a ->
          let t =
            serve a 1
              {
                answer =
                  (fun (type r) (r : r Wire.request) : (r, string) result ->
                    match r with
                    | Wire.Open _ -> Ok [ account ]
                    | _ -> Error "unexpected");
              }
          in
          let got = Link.request c (Wire.Open "MEM") in
          Thread.join t;
          equal (result int string) (Ok 1) (Result.map List.length got);
          equal (result string string) (Ok "MEM:0")
            (Result.map (fun l -> (List.hd l).Wire.name) got));
      test "an agent's refusal is the request's error" (fun () ->
          with_job @@ fun _ c a ->
          let t = serve a 1 { answer = (fun _ -> Error "no such kind") } in
          let got = Link.request c (Wire.Open "GPU") in
          Thread.join t;
          equal (result int string) (Error "no such kind")
            (Result.map List.length got));
    ]

let rails =
  group ~timeout "rail"
    [
      test "a transfer's bytes are in place once arrived reaches its count"
        (fun () ->
          with_job @@ fun _ c a ->
          let t : Rig_remote_abi.transfer = { src = 8; dst = 16; length = 5 } in
          let s = Link.rail c ~id:1 ~send:[| t |] ~receive:[||] in
          let r = Link.rail a ~id:1 ~send:[||] ~receive:[| t |] in
          String.iteri (fun i ch -> s.outbound.{8 + i} <- ch) "hello";
          set_u64 s.counts 0 1;
          while get_u64 r.counts 256 < 1 do
            Thread.yield ()
          done;
          equal string "hello" (String.init 5 (fun i -> r.inbound.{16 + i}));
          equal int 1 (get_u64 r.counts 256));
    ]

let state =
  Testable.structural ~pp:(fun ppf -> function
    | Link.Open -> Format.pp_print_string ppf "open"
    | Link.Closed -> Format.pp_print_string ppf "closed"
    | Link.Failed why -> Format.fprintf ppf "failed: %s" why)

let failures =
  group ~timeout "failure"
    [
      test "a job closed at both ends of its links is closed" (fun () ->
          with_job @@ fun j _ _ ->
          Link.close j;
          equal state Link.Closed (Link.wait j ~ms:0));
      test "a failed job is failed with its root cause" (fun () ->
          with_job @@ fun j c _ ->
          Link.fail j "the test fails it";
          equal (option string) (Some "the test fails it") (Link.failure j);
          equal (result int string) (Error "the test fails it")
            (Result.map List.length (Link.request c (Wire.Open "MEM"))));
    ]

let () = exit (run "rig_remote_proxy.link" [ requests; rails; failures ])
