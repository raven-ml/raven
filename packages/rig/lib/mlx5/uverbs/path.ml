(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The calls follow Linux v6.12's drivers/infiniband/core: uverbs_ioctl.c (how
   the kernel reads the ioctl), uverbs_std_types_device.c (invoking a command),
   uverbs_cmd.c and uverbs_std_types_mr.c; and rdma-core v56.0's
   libibverbs/cmd_*.c, which makes the same calls. The stubs pack them. *)

module M = Rig_mlx5
open Verbs

let strf = Printf.sprintf
let ( let* ) = Result.bind

let failed what e = Error (strf "%s: %s" what (Unix.error_message e))

(* [guard what f] is [f ()], or [Error] saying that [what] failed and why. *)
let guard what f =
  match f () with
  | v -> Ok v
  | exception Unix.Unix_error (e, _, _) -> failed what e

(* The machine's files *)

let read_file path =
  match In_channel.with_open_bin path In_channel.input_all with
  | s -> Some (String.trim s)
  | exception Sys_error _ -> None

let entries dir =
  match Sys.readdir dir with
  | a ->
      Array.sort compare a;
      Array.to_list a
  | exception Sys_error _ -> []

let ( // ) = Filename.concat
let class_dir root = root // "sys/class/infiniband"

(* A device's PCI function, as its uevent file lists it: KEY=value lines. *)
let uevent root name key =
  match read_file (class_dir root // name // "device/uevent") with
  | None -> None
  | Some s ->
      List.find_map
        (fun l ->
          match String.index_opt l '=' with
          | Some i when String.sub l 0 i = key ->
              Some (String.sub l (i + 1) (String.length l - i - 1))
          | _ -> None)
        (String.split_on_char '\n' s)

let devices root = entries (class_dir root)
let exists root name = Sys.file_exists (class_dir root // name)
let driver_name root name = uevent root name "DRIVER"

(* The verbs file of [name]: the uverbsN whose ibdev is [name]. *)
let verbs_file root name =
  let dir = root // "sys/class/infiniband_verbs" in
  List.find_map
    (fun e ->
      if read_file (dir // e // "ibdev") = Some name then
        Some (root // "dev/infiniband" // e)
      else None)
    (entries dir)

(* A global identifier as the kernel prints it, eight groups of 4 hex digits:
   its 16 bytes, or [None] for the zero identifier of an unused entry. *)
let gid_of_text t =
  match String.split_on_char ':' t with
  | groups when List.length groups = 8 -> (
      match List.map (fun g -> int_of_string ("0x" ^ g)) groups with
      | ws ->
          let b = Bytes.create 16 in
          List.iteri (fun i w -> Bytes.set_uint16_be b (2 * i) w) ws;
          let s = Bytes.to_string b in
          if s = String.make 16 '\000' then None else Some s
      | exception Failure _ -> None)
  | _ -> None

let roce_v2 = "RoCE v2"

(* The port's RoCE v2 identifiers, by index. A type file of an unused entry
   cannot be read. *)
let gids root name =
  let port = class_dir root // name // "ports/1" in
  List.filter_map
    (fun e ->
      match
        (int_of_string_opt e, read_file (port // "gid_attrs/types" // e))
      with
      | Some i, Some ty when ty = roce_v2 ->
          Option.bind (read_file (port // "gids" // e)) gid_of_text
          |> Option.map (fun g -> (i, g))
      | _ -> None)
    (entries (port // "gids"))
  |> List.sort compare

(* The verbs file, open *)

type file = {
  fd : int;
  driver : int; (* the kernel's id of the device's driver *)
  lock : Mutex.t;
  mutable pd : int;
  mutable async : int; (* the descriptor of asynchronous events, or -1 *)
  mutable channel : int; (* the descriptor of completion events, or -1 *)
  mutable maps : (int * int) list;
}

let context f root name d n =
  let* async, answer =
    guard "making the context" (fun () -> get_context f.fd f.driver d n)
  in
  f.async <- async;
  let* channel =
    guard "making the completion channel" (fun () ->
        comp_channel f.fd f.driver)
  in
  f.channel <- channel;
  let* pd =
    guard "making the protection domain" (fun () -> alloc_pd f.fd f.driver)
  in
  f.pd <- pd;
  let* reads, served =
    guard "querying the device" (fun () -> query_device f.fd f.driver)
  in
  let* active, link, lid, mtu =
    guard "querying port 1" (fun () -> query_port f.fd f.driver)
  in
  let* () = if active then Ok () else Error "port 1 is not active" in
  let* port =
    match link with
    | Infiniband -> Ok (`Infiniband lid)
    | Ethernet -> Ok (`Ethernet (gids root name))
    | Link l ->
        Error
          (strf "port 1's link layer %d is neither InfiniBand nor Ethernet" l)
  in
  Ok { M.answer; port; mtu; reads; served }

let map f off n =
  let* at =
    guard (strf "mapping %d bytes at offset 0x%x" n off) (fun () ->
        map_raw f.fd off n)
  in
  Mutex.protect f.lock (fun () -> f.maps <- (at, n) :: f.maps);
  Ok at

(* The stub's code of an access and an order: the access in bits 0 and 1, and
   bit 2 set for [Any_order]. *)
let access (a : M.Region.access) (o : M.Region.order) =
  let what = match a with Local -> 0 | Remote_write -> 1 | Remote -> 2 in
  match o with In_order -> what | Any_order -> what lor 4

let registered (handle, local, remote) = { M.handle; local; remote }

let register f a o (m : M.Region.memory) =
  match m with
  | Host b ->
      let address = address b and bytes = Bigarray.Array1.dim b in
      let r =
        guard (strf "registering %d bytes at 0x%x" bytes address) (fun () ->
            reg_mr f.fd f.driver f.pd address bytes (access a o))
      in
      ignore (Sys.opaque_identity b);
      Result.map registered r
  | Dmabuf { fd; offset; bytes } ->
      guard (strf "registering %d bytes of dma-buf %d" bytes fd) (fun () ->
          reg_dmabuf f.fd f.driver f.pd fd offset bytes (access a o))
      |> Result.map registered

let cq f ~entries ~tag d n =
  guard (strf "making a completion queue of %d entries" entries) (fun () ->
      create_cq f.fd f.driver f.channel entries tag d n)

let qp f ~cq ~entries ~tag d n =
  guard (strf "making a queue pair of %d entries" entries) (fun () ->
      create_qp f.fd f.driver f.pd cq entries tag d n)

let modify f h (t : M.transition) =
  let change, what =
    match t with
    | Init -> (Init, "initialising")
    | Ready_to_receive { peer; mtu; gid; served } ->
        let dest =
          match (peer.address, gid) with
          | Lid lid, _ -> Lid lid
          | Gid g, Some index -> Gid (g, index)
          | Gid _, None ->
              invalid_arg "Rig_mlx5_uverbs: a global address with no local one"
        in
        ( Rtr { qp = peer.qp; psn = peer.psn; served; mtu; dest },
          "making it ready to receive" )
    | Ready_to_send { psn; timeout; retries; reads } ->
        (Rts { psn; timeout; retries; reads }, "making it ready to send")
  in
  guard (strf "queue pair %d: %s" h what) (fun () ->
      modify_qp f.fd f.driver h change)

let destroy f kind h =
  let code, what =
    match kind with
    | `Region -> (0, "deregistering a region")
    | `Cq -> (1, "destroying a completion queue")
    | `Qp -> (2, "destroying a queue pair")
  in
  guard what (fun () -> destroy_raw f.fd f.driver code h)

(* Events *)

let async_event (kind, element) : M.kernel_event =
  match kind with
  | Cq_err -> Cq_error element
  | Qp_fatal -> Qp_failed (element, "fatal error")
  | Qp_req_err -> Qp_failed (element, "request error")
  | Qp_access_err -> Qp_failed (element, "access error")
  | Device_fatal -> Nic_failed
  | Port_active -> Port_changed "active"
  | Port_err -> Port_changed "down"

let wait f ms : M.kernel_event list =
  match poll f.async f.channel ms with
  | 0 | (exception Unix.Unix_error _) -> []
  | ready ->
      let async =
        if ready land 1 = 0 then []
        else List.rev_map async_event (async_events f.async)
      in
      let completions =
        if ready land 2 = 0 then []
        else List.rev_map (fun cq -> M.Completed cq) (comp_events f.channel)
      in
      (* An event descriptor hangs up once the device is gone, and stays
         readable: the NIC failed. *)
      let gone = if ready land 4 = 0 then [] else [ M.Nic_failed ] in
      async @ completions @ gone

let close f () =
  List.iter (fun (at, n) -> unmap at n) f.maps;
  if f.channel >= 0 then close_fd f.channel;
  if f.async >= 0 then close_fd f.async;
  close_fd f.fd

let open_ ~root name =
  let bus = Option.value ~default:"" (uevent root name "PCI_SLOT_NAME") in
  let* path =
    Option.to_result
      ~none:(strf "%s: no verbs file" name)
      (verbs_file root name)
  in
  let* fd = guard path (fun () -> open_raw path) in
  let* driver =
    match driver_id name with
    | d -> Ok d
    | exception Unix.Unix_error (e, _, _) ->
        close_fd fd;
        failed (name ^ ": its driver's id") e
  in
  let f =
    {
      fd;
      driver;
      lock = Mutex.create ();
      pd = 0;
      async = -1;
      channel = -1;
      maps = [];
    }
  in
  Ok
    {
      M.name;
      bus;
      page = page_size ();
      context = context f root name;
      map = map f;
      register = register f;
      cq = cq f;
      qp = qp f;
      modify = modify f;
      destroy = destroy f;
      wait = wait f;
      close = close f;
    }
