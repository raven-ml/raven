(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The uverbs path on a Soft-RoCE device (rdma_rxe), the kernel's RoCE v2 over
   an Ethernet interface: the path's own calls with no driver data in and room
   for rxe's answers, as a device of another driver than mlx5 takes them,
   through Rig_mlx5_uverbs.path. A context and its port, regions of each access,
   a completion queue, two queue pairs taken through their states to ready to
   send, connected to each other, their destruction, an owner killed with its
   queue pairs, which the kernel tears down, and Rig_mlx5.make's refusal of a
   device that is no mlx5.

   Every test skips where the machine has no device named rxe*. Making one is a
   machine's setup, as root, over an Ethernet interface IF:

   modprobe rdma_rxe rdma link add rxe0 type rxe netdev IF *)

open Windtrap
module M = Rig_mlx5

let setup =
  "no Soft-RoCE device; as root: modprobe rdma_rxe; rdma link add rxe0 type \
   rxe netdev IF"

let devices () =
  match Sys.readdir "/sys/class/infiniband" with
  | a -> Array.to_list a
  | exception Sys_error _ -> []

let device () =
  match List.find_opt (String.starts_with ~prefix:"rxe") (devices ()) with
  | Some name -> name
  | None -> skip ~reason:setup ()

let open_path () =
  let name = device () in
  match Rig_mlx5_uverbs.path name with
  | Ok p -> p
  | Error e -> failf "opening %s: %s" name e

let ok what = function Ok v -> v | Error e -> failf "%s: %s" what e

let opened () =
  let p = open_path () in
  let c = ok "the context" (p.context "" 0) in
  (p, c)

let memory n =
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill b '\000';
  (b, M.Region.Host b)

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
  let access_name : M.Region.access -> string = function
    | Local -> "local"
    | Remote_write -> "remote write"
    | Remote -> "remote"
  in
  let order_name : M.Region.order -> string = function
    | In_order -> "in order"
    | Any_order -> "in any order"
  in
  cases
    ~name:(fun (a, o) -> access_name a ^ ", " ^ order_name o)
    "memory registers for each access and order, with distinct keys"
    (List.concat_map
       (fun a -> [ (a, M.Region.In_order); (a, Any_order) ])
       [ M.Region.Local; Remote_write; Remote ])
    (fun (access, order) ->
      let p, _ = opened () in
      let b, m = memory 8192 in
      let r = ok "registering" (p.register access order m) in
      let s = ok "registering again" (p.register access order m) in
      not_equal ~msg:"local keys" int r.local s.local;
      ok "deregistering" (p.destroy `Region r.handle);
      ok "deregistering" (p.destroy `Region s.handle);
      ignore (Sys.opaque_identity b);
      p.close ())

(* The bytes rxe answers on making a completion queue and a queue pair, the
   locations of their rings (rdma_user_rxe.h: struct rxe_create_cq_resp and
   struct rxe_create_qp_resp). With less room it refuses the call. *)
let cq_answer = 16
let qp_answer = 32

(* Two queue pairs of one port connected to each other. *)
let connect (p : M.path) (c : M.context) =
  let gid, g =
    match c.port with
    | `Ethernet (g :: _) -> g
    | _ -> fail "no RoCE v2 global identifier"
  in
  let cq, _ =
    ok "making a completion queue" (p.cq ~entries:16 ~tag:1 "" cq_answer)
  in
  let qp tag =
    ok "making a queue pair" (p.qp ~cq ~entries:8 ~tag "" qp_answer)
  in
  let a, qa, _ = qp 2 in
  let b, qb, _ = qp 3 in
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
      ok "destroying" (p.destroy `Qp a);
      ok "destroying" (p.destroy `Qp b);
      ok "destroying" (p.destroy `Cq cq);
      p.close ())

let refusals =
  test "a completion queue with queue pairs is not destroyed" (fun () ->
      let p, c = opened () in
      let cq, a, b = connect p c in
      is_error (p.destroy `Cq cq);
      ok "destroying" (p.destroy `Qp a);
      ok "destroying" (p.destroy `Qp b);
      ok "destroying" (p.destroy `Cq cq);
      p.close ())

let events =
  test "a wait with no event takes its time" (fun () ->
      let p, _ = opened () in
      let start = Unix.gettimeofday () in
      equal int 0 (List.length (p.wait 50));
      at_least ~msg:"seconds waited" float_exact ~than:0.049
        (Unix.gettimeofday () -. start);
      p.close ())

(* The descriptors this process holds on verbs files. *)
let verbs_files () =
  let fds = Sys.readdir "/proc/self/fd" in
  let on_verbs fd =
    match Unix.readlink (Filename.concat "/proc/self/fd" fd) with
    | target -> String.starts_with ~prefix:"/dev/infiniband/uverbs" target
    | exception Unix.Unix_error _ -> false
  in
  List.length (List.filter on_verbs (Array.to_list fds))

(* rxe is no mlx5 device: it gives none of the doorbell registers make needs, or
   refuses mlx5's driver data. *)
let make =
  test
    "a make over a device of another driver is an Error naming it, its path \
     closed" (fun () ->
      let name = device () in
      let before = verbs_files () in
      let p = open_path () in
      equal ~msg:"the path's verbs file" int (before + 1) (verbs_files ());
      let e =
        require_error
          ~pp:(fun ppf _ -> Format.pp_print_string ppf "a NIC")
          (M.make p)
      in
      starts_with ~msg:"its name" ~affix:(name ^ ": ") e;
      equal ~msg:"verbs files once make failed" int before (verbs_files ()))

let status =
  let pp ppf = function
    | Unix.WEXITED n -> Format.fprintf ppf "exited %d" n
    | WSIGNALED n -> Format.fprintf ppf "killed by %s" (Sys.signal_to_string n)
    | WSTOPPED n -> Format.fprintf ppf "stopped by %s" (Sys.signal_to_string n)
  in
  Testable.make ~pp ~equal:( = )

(* The queue pairs of [name]'s port as the kernel lists them, its own among
   them. *)
let qps name =
  let ic =
    Unix.open_process_args_in "rdma"
      [| "rdma"; "resource"; "show"; "qp"; "link"; name ^ "/1" |]
  in
  let lines = In_channel.input_lines ic in
  equal ~msg:"rdma resource show" status (Unix.WEXITED 0)
    (Unix.close_process_in ic);
  let qp l = List.mem "lqpn" (String.split_on_char ' ' l) in
  List.length (List.filter qp lines)

(* A child makes connected queue pairs and dies by SIGKILL, closing nothing: the
   kernel destroys them with its file. *)
let killed =
  test "a killed owner's queue pairs are its kernel's to destroy" (fun () ->
      let name = device () in
      let before = qps name in
      let r, w = Unix.pipe () in
      match Unix.fork () with
      | 0 ->
          (try
             let p, c = opened () in
             ignore (connect p c : int * int * int);
             ignore (Unix.write_substring w "x" 0 1 : int);
             Unix.sleep 60
           with _ -> ());
          Unix._exit 0
      | child ->
          Unix.close w;
          let made = Unix.read r (Bytes.create 1) 0 1 in
          let during = qps name in
          Unix.kill child Sys.sigkill;
          let _, st = Unix.waitpid [] child in
          equal ~msg:"the child made its queue pairs" int 1 made;
          equal ~msg:"queue pairs while the child lived" int (before + 2) during;
          equal ~msg:"the child died by SIGKILL" status
            (Unix.WSIGNALED Sys.sigkill) st;
          equal ~msg:"queue pairs once the child died" int before (qps name))

let () =
  exit
    (run "rig_mlx5_uverbs.rxe"
       [ context; regions; queue_pairs; refusals; events; killed; make ])
