(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The uverbs path on a Soft-RoCE device (rdma_rxe), the kernel's RoCE v2 over
   an Ethernet interface: the path's own calls with empty driver data, as a
   device of another driver than mlx5 takes them, through the library's private
   Path, copied here. A context and its port, regions of each access, a
   completion queue, two queue pairs taken through their states to ready to
   send, connected to each other, their destruction, and an owner killed with
   its queue pairs, which the kernel tears down (rxe.sh counts them).

   Every test skips where the machine has no device named rxe*; rxe.sh makes
   one, runs this suite as root, and removes it. *)

open Windtrap
module M = Rig_mlx5

external address : Request.params -> int = "caml_rig_mlx5_uverbs_address"

let device () =
  match List.find_opt (String.starts_with ~prefix:"rxe") (Path.devices "/") with
  | Some name -> name
  | None -> skip ~reason:"no Soft-RoCE device" ()

let open_path () =
  let name = device () in
  match Path.open_ ~driver:Defs.rdma_driver_rxe ~root:"/" name with
  | Ok p -> p
  | Error e -> failf "opening %s: %s" name e

let ok what = function Ok v -> v | Error e -> failf "%s: %s" what e

let opened () =
  let p = open_path () in
  let c = ok "the context" (p.context "" 0) in
  (p, c)

let memory n =
  let b = Request.params n in
  (b, M.Region.Host { address = address b; bytes = n })

let context =
  test "a context reports a RoCE v2 port and its limits" (fun () ->
      let p, c = opened () in
      (match c.port with
      | `Ethernet (_ :: _) -> ()
      | `Ethernet [] -> fail "no RoCE v2 global identifier"
      | `Infiniband _ -> fail "an InfiniBand port");
      at_least ~msg:"MTU" int ~than:256 c.mtu;
      at_least ~msg:"reads started" int ~than:1 c.reads;
      at_least ~msg:"reads served" int ~than:1 c.served;
      p.close ())

let regions =
  cases
    ~name:(function
      | M.Region.Local -> "local"
      | Remote_write -> "remote write"
      | Remote -> "remote")
    "memory registers for each access, with distinct keys"
    [ M.Region.Local; Remote_write; Remote ]
    (fun access ->
      let p, _ = opened () in
      let b, m = memory 8192 in
      let r = ok "registering" (p.register access m) in
      let s = ok "registering again" (p.register access m) in
      not_equal ~msg:"local keys" int r.local s.local;
      p.destroy `Region r.handle;
      p.destroy `Region s.handle;
      ignore (Sys.opaque_identity b);
      p.close ())

(* Two queue pairs of one port connected to each other. *)
let connect (p : M.path) (c : M.context) =
  let gid, g =
    match c.port with
    | `Ethernet (g :: _) -> g
    | _ -> fail "no RoCE v2 global identifier"
  in
  let cq, _ = ok "making a completion queue" (p.cq ~entries:16 ~tag:1 "" 0) in
  let a, qa, _ = ok "making a queue pair" (p.qp ~cq ~entries:8 ~tag:2 "" 0) in
  let b, qb, _ = ok "making a queue pair" (p.qp ~cq ~entries:8 ~tag:3 "" 0) in
  let endpoint qp psn = { M.Qp.qp; psn; address = Gid g; mtu = c.mtu } in
  let steps h peer psn =
    [
      M.Init;
      Ready_to_receive { peer; mtu = c.mtu; gid = Some gid; served = c.served };
      Ready_to_send { psn; timeout = 14; retries = 7; reads = c.reads };
    ]
    |> List.iter (fun t -> ok "changing a queue pair's state" (p.modify h t))
  in
  steps a (endpoint qb 0x200) 0x100;
  steps b (endpoint qa 0x100) 0x200;
  (cq, a, b)

let queue_pairs =
  test "queue pairs go through their states and are destroyed" (fun () ->
      let p, c = opened () in
      let cq, a, b = connect p c in
      p.destroy `Qp a;
      p.destroy `Qp b;
      p.destroy `Cq cq;
      p.close ())

let refusals =
  test "a completion queue with queue pairs is not destroyed" (fun () ->
      let p, c = opened () in
      let cq, a, b = connect p c in
      raises_match
        (fun e -> match e with Failure _ -> true | _ -> false)
        (fun () -> p.destroy `Cq cq);
      p.destroy `Qp a;
      p.destroy `Qp b;
      p.destroy `Cq cq;
      p.close ())

let events =
  test "a wait with no event takes its time" (fun () ->
      let p, _ = opened () in
      let start = Unix.gettimeofday () in
      equal int 0 (List.length (p.wait 50));
      at_least ~msg:"seconds waited" float_exact ~than:0.049
        (Unix.gettimeofday () -. start);
      p.close ())

(* A child makes connected queue pairs and dies by SIGKILL, closing nothing: the
   kernel destroys them with its file. *)
let killed =
  test "a killed owner's queue pairs are its kernel's to destroy" (fun () ->
      ignore (device () : string);
      let r, w = Unix.pipe () in
      match Unix.fork () with
      | 0 ->
          let p, c = opened () in
          ignore (connect p c : int * int * int);
          ignore (Unix.write_substring w "x" 0 1 : int);
          Unix.sleep 60;
          exit 0
      | child ->
          let b = Bytes.create 1 in
          equal ~msg:"the child made its queue pairs" int 1 (Unix.read r b 0 1);
          Unix.kill child Sys.sigkill;
          let _, st = Unix.waitpid [] child in
          equal ~msg:"the child died by SIGKILL" bool true
            (st = Unix.WSIGNALED Sys.sigkill))

let () =
  exit
    (run "rig_mlx5_uverbs.rxe"
       [ context; regions; queue_pairs; refusals; events; killed ])
