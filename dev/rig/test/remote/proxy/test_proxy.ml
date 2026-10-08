(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A proxy of another machine's host, whose agent's end is a loop in this
   process that keeps the machine's memory as bytes. *)

open Windtrap
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module B = Rig.Buffer

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

let host_account : Wire.account =
  { id = 0; name = "CPU"; arch = "arm64"; budget = 1 lsl 30; reaches = [] }

let area_of_bytes b =
  let a =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (Bytes.length b)
  in
  Bytes.iteri (fun i c -> a.{i} <- c) b;
  a

(* The agent's end: memory as bytes by id, each hand-over's copies run in order,
   then its word. *)
let agent a =
  let memory = Hashtbl.create 8 in
  let rec loop () =
    match Link.next a with
    | Error _ | Ok Wire.Close -> ()
    | Ok (Wire.Request (Wire.Alloc { id; bytes; _ } as r)) ->
        Hashtbl.replace memory id (Bytes.make bytes '\000');
        Link.answer a r (Ok true);
        loop ()
    | Ok (Wire.Request r) ->
        Link.answer a r (Error "not served here");
        loop ()
    | Ok (Wire.Drop id) ->
        Hashtbl.remove memory id;
        loop ()
    | Ok (Wire.Handover (h, local)) ->
        let k = ref 0 in
        Array.iter
          (function
            | Wire.Copy
                { src = Wire.Local; dst = Wire.Region { id; offset }; bytes } ->
                let m = Hashtbl.find memory id in
                for i = 0 to bytes - 1 do
                  Bytes.set m (offset + i) local.(!k).{i}
                done;
                incr k
            | Wire.Copy
                { src = Wire.Region { id; offset }; dst = Wire.Local; bytes } ->
                let m = Hashtbl.find memory id in
                Link.bytes a ~device:h.device ~value:h.value
                  (area_of_bytes (Bytes.sub m offset bytes))
            | _ -> ())
          h.parts;
        Link.word a ~device:h.device h.value;
        loop ()
  in
  loop ()

let machines = Atomic.make 0

(* A job whose host proxy talks to [agent] on a thread. *)
let with_host f =
  let j = Link.job () in
  let d, a = connected () in
  let c = Link.make j d ~name:"far" ~peer:(Wire.Agent 1) in
  let a = Link.make j a ~name:"controller" ~peer:Wire.Controller in
  let t = Thread.create agent a in
  let machine = Printf.sprintf "proxy:%d" (Atomic.fetch_and_add machines 1) in
  let host =
    Rig_remote_proxy.make c host_account
      (Rig_remote_abi.Host
         { machine; rail = (fun _ ~send:_ ~receive:_ -> Error "no rails") })
  in
  let h =
    match
      Rig.open_
        (module Rig_remote_proxy)
        ~machine ~host:true ~name:"CPU"
        (fun () -> Ok host)
    with
    | Ok h -> h
    | Error e -> failwith e
  in
  Fun.protect
    ~finally:(fun () ->
      Link.fail j "test ended";
      Thread.join t)
    (fun () -> f h)

let copies =
  group ~timeout "copy"
    [
      test "bytes copied to another machine's host come back" (fun () ->
          with_host @@ fun h ->
          let src = B.of_string "to the agent and back" in
          let far = B.create h (B.length src) in
          let back = B.create Rig.host (B.length src) in
          B.copy ~src ~dst:far;
          B.copy ~src:far ~dst:back;
          equal string "to the agent and back"
            (let ba = B.bigarray Bigarray.char back in
             String.init (B.length back) (Bigarray.Array1.get ba)));
      test "a proxy's name ends with its machine's" (fun () ->
          with_host @@ fun h ->
          equal string "CPU@proxy:" (String.sub (Rig.name h) 0 10));
    ]

let () = exit (run "rig_remote_proxy.proxy" [ copies ])
