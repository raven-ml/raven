(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module R = Request
module M = Rig_mlx5

let strf = Printf.sprintf
let ( let* ) = Result.bind

external open_raw : string -> int = "caml_rig_mlx5_uverbs_open"
external close_fd : int -> unit = "caml_rig_mlx5_uverbs_close"
external nonblock : int -> int = "caml_rig_mlx5_uverbs_nonblock"
external ioctl : int -> int -> R.params -> int = "caml_rig_mlx5_uverbs_ioctl"
external map_raw : int -> int -> int -> int = "caml_rig_mlx5_uverbs_map"
external unmap : int -> int -> unit = "caml_rig_mlx5_uverbs_unmap"
external page_size : unit -> int = "caml_rig_mlx5_uverbs_page_size"
external poll : int -> int -> int -> int = "caml_rig_mlx5_uverbs_poll"
external read_raw : int -> R.params -> int = "caml_rig_mlx5_uverbs_read"
external strerror : int -> string = "caml_rig_mlx5_uverbs_strerror"

(* A stub's result: non-negative, or errno negated. *)
let result what r =
  if r < 0 then Error (strf "%s: %s" what (strerror (-r))) else Ok r

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

(* Runs [r], kept alive until the kernel returned. *)
let run f what r =
  let e = ioctl f.fd R.number (R.bytes r) in
  ignore (Sys.opaque_identity r : R.t);
  Result.map ignore (result what e)

(* Invokes the command [cmd] with the request [core], filled by [fields],
   answering [out] bytes. *)
let command f what ~cmd ~core ?(out = 0) ?(uhw = "") ?(uhw_out = 0) fields =
  let req = R.params core in
  List.iter (fun (at, v) -> R.set req at v) fields;
  let out = R.params out and uhw_out = R.params uhw_out in
  let* () = run f what (R.write ~driver:f.driver ~cmd req ~out ~uhw ~uhw_out) in
  Ok (out, uhw_out)

let event_file what fd =
  let* fd = result what fd in
  let* _ = result what (nonblock fd) in
  Ok fd

(* The MTU of an IBV_MTU_* value: 256 bytes for the first, doubling. *)
let mtu_bytes v = 256 lsl (v - D.ibv_mtu_256)
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)
let mtu_value bytes = D.ibv_mtu_256 + log2 (bytes / 256)
let port_number = 1

let context f root name d n =
  let* resp, answer =
    command f "making the context" ~cmd:D.ib_user_verbs_cmd_get_context
      ~core:D.Get_context.sizeof ~out:D.Get_context_resp.sizeof ~uhw:d
      ~uhw_out:n []
  in
  let* async =
    event_file "reading the asynchronous events"
      (R.field resp D.Get_context_resp.async_fd)
  in
  f.async <- async;
  let* resp, _ =
    command f "making the completion channel"
      ~cmd:D.ib_user_verbs_cmd_create_comp_channel
      ~core:D.Create_comp_channel.sizeof ~out:D.Create_comp_channel_resp.sizeof
      []
  in
  let* channel =
    event_file "reading the completion events"
      (R.field resp D.Create_comp_channel_resp.fd)
  in
  f.channel <- channel;
  let* resp, _ =
    command f "making the protection domain" ~cmd:D.ib_user_verbs_cmd_alloc_pd
      ~core:D.Alloc_pd.sizeof ~out:D.Alloc_pd_resp.sizeof []
  in
  f.pd <- R.field resp D.Alloc_pd_resp.pd_handle;
  let* device, _ =
    command f "querying the device" ~cmd:D.ib_user_verbs_cmd_query_device
      ~core:D.Query_device.sizeof ~out:D.Query_device_resp.sizeof []
  in
  let* port, _ =
    command f "querying port 1" ~cmd:D.ib_user_verbs_cmd_query_port
      ~core:D.Query_port.sizeof ~out:D.Query_port_resp.sizeof
      [ (D.Query_port.port_num, port_number) ]
  in
  let get = R.field port in
  let* () =
    if get D.Query_port_resp.state = D.ibv_port_active then Ok ()
    else Error "port 1 is not active"
  in
  let* port =
    let link = get D.Query_port_resp.link_layer in
    if link = D.ibv_link_layer_infiniband then
      Ok (`Infiniband (get D.Query_port_resp.lid))
    else if link = D.ibv_link_layer_ethernet then
      Ok (`Ethernet (gids root name))
    else
      Error
        (strf "port 1's link layer %d is neither InfiniBand nor Ethernet" link)
  in
  Ok
    {
      M.answer = R.to_string answer;
      port;
      mtu = mtu_bytes (get D.Query_port_resp.active_mtu);
      reads = R.field device D.Query_device_resp.max_qp_init_rd_atom;
      served = R.field device D.Query_device_resp.max_qp_rd_atom;
    }

let map f off n =
  let* at =
    result (strf "mapping %d bytes at offset 0x%x" n off) (map_raw f.fd off n)
  in
  Mutex.protect f.lock (fun () -> f.maps <- (at, n) :: f.maps);
  Ok at

let access_flags : M.Region.access -> int = function
  | Local -> D.ib_uverbs_access_local_write
  | Remote_write ->
      D.ib_uverbs_access_local_write lor D.ib_uverbs_access_remote_write
      lor D.ib_uverbs_access_relaxed_ordering
  | Remote ->
      D.ib_uverbs_access_local_write lor D.ib_uverbs_access_remote_write
      lor D.ib_uverbs_access_remote_read

let register f access (m : M.Region.memory) =
  let flags = access_flags access in
  match m with
  | Host { address; bytes } ->
      let* resp, _ =
        command f
          (strf "registering %d bytes at 0x%x" bytes address)
          ~cmd:D.ib_user_verbs_cmd_reg_mr ~core:D.Reg_mr.sizeof
          ~out:D.Reg_mr_resp.sizeof
          [
            (D.Reg_mr.start, address);
            (D.Reg_mr.length, bytes);
            (D.Reg_mr.hca_va, address);
            (D.Reg_mr.pd_handle, f.pd);
            (D.Reg_mr.access_flags, flags);
          ]
      in
      let get = R.field resp in
      Ok
        {
          M.handle = get D.Reg_mr_resp.mr_handle;
          local = get D.Reg_mr_resp.lkey;
          remote = get D.Reg_mr_resp.rkey;
        }
  | Dmabuf { fd; offset; bytes } ->
      let lkey = R.params 4 and rkey = R.params 4 in
      let r =
        R.call ~driver:f.driver ~obj:D.uverbs_object_mr
          ~meth:D.uverbs_method_reg_dmabuf_mr
          [
            Made D.uverbs_attr_reg_dmabuf_mr_handle;
            Handle (D.uverbs_attr_reg_dmabuf_mr_pd_handle, f.pd);
            Value (D.uverbs_attr_reg_dmabuf_mr_offset, offset);
            Value (D.uverbs_attr_reg_dmabuf_mr_length, bytes);
            Value (D.uverbs_attr_reg_dmabuf_mr_iova, offset);
            Word (D.uverbs_attr_reg_dmabuf_mr_fd, fd);
            Word (D.uverbs_attr_reg_dmabuf_mr_access_flags, flags);
            Out (D.uverbs_attr_reg_dmabuf_mr_resp_lkey, lkey);
            Out (D.uverbs_attr_reg_dmabuf_mr_resp_rkey, rkey);
          ]
      in
      let* () = run f (strf "registering %d bytes of dma-buf %d" bytes fd) r in
      Ok
        {
          M.handle = R.made r D.uverbs_attr_reg_dmabuf_mr_handle;
          local = R.field lkey (0, 4);
          remote = R.field rkey (0, 4);
        }

let cq f ~entries ~tag d n =
  let* resp, answer =
    command f
      (strf "making a completion queue of %d entries" entries)
      ~cmd:D.ib_user_verbs_cmd_create_cq ~core:D.Create_cq.sizeof
      ~out:D.Create_cq_resp.sizeof ~uhw:d ~uhw_out:n
      [
        (D.Create_cq.user_handle, tag);
        (D.Create_cq.cqe, entries - 1);
        (D.Create_cq.comp_channel, f.channel);
      ]
  in
  Ok (R.field resp D.Create_cq_resp.cq_handle, R.to_string answer)

let qp f ~cq ~entries ~tag d n =
  let* resp, answer =
    command f
      (strf "making a queue pair of %d entries" entries)
      ~cmd:D.ib_user_verbs_cmd_create_qp ~core:D.Create_qp.sizeof
      ~out:D.Create_qp_resp.sizeof ~uhw:d ~uhw_out:n
      [
        (D.Create_qp.user_handle, tag);
        (D.Create_qp.pd_handle, f.pd);
        (D.Create_qp.send_cq_handle, cq);
        (D.Create_qp.recv_cq_handle, cq);
        (D.Create_qp.max_send_wr, entries);
        (D.Create_qp.max_send_sge, 1);
        (D.Create_qp.qp_type, D.ib_uverbs_qpt_rc);
      ]
  in
  let get = R.field resp in
  Ok
    ( get D.Create_qp_resp.qp_handle,
      get D.Create_qp_resp.qpn,
      R.to_string answer )

(* What a queue pair answers a peer's request that finds no receive, after which
   a peer retries: 12, 0.64 ms, InfiniBand's [min_rnr_timer] encoding. These
   queue pairs post no receives, and peers send them none. *)
let rnr_timer = 12
let rnr_retries = 7

(* The hops a RoCE v2 packet crosses before a router drops it. *)
let hop_limit = 64

let modify f h (t : M.transition) =
  let module Q = D.Modify_qp in
  let fields, mask, what =
    match t with
    | Init ->
        ( [
            (Q.qp_state, D.ibv_qps_init);
            (Q.pkey_index, 0);
            (Q.port_num, port_number);
            ( Q.qp_access_flags,
              D.ib_uverbs_access_remote_write lor D.ib_uverbs_access_remote_read
            );
          ],
          D.ibv_qp_state lor D.ibv_qp_pkey_index lor D.ibv_qp_port
          lor D.ibv_qp_access_flags,
          "initialising" )
    | Ready_to_receive { peer; mtu; gid; served } ->
        let dest =
          match (peer.address, gid) with
          | Lid lid, _ -> [ (Q.dest__dlid, lid) ]
          | Gid g, Some index ->
              [
                (Q.dest__is_global, 1);
                (Q.dest__sgid_index, index);
                (Q.dest__hop_limit, hop_limit);
              ]
              @ List.init 16 (fun i ->
                  ((fst Q.dest__dgid + i, 1), Char.code g.[i]))
          | Gid _, None ->
              invalid_arg "Rig_mlx5_uverbs: a global address with no local one"
        in
        ( [
            (Q.qp_state, D.ibv_qps_rtr);
            (Q.path_mtu, mtu_value mtu);
            (Q.dest_qp_num, peer.qp);
            (Q.rq_psn, peer.psn);
            (Q.max_dest_rd_atomic, served);
            (Q.min_rnr_timer, rnr_timer);
            (Q.dest__port_num, port_number);
          ]
          @ dest,
          D.ibv_qp_state lor D.ibv_qp_av lor D.ibv_qp_path_mtu
          lor D.ibv_qp_dest_qpn lor D.ibv_qp_rq_psn
          lor D.ibv_qp_max_dest_rd_atomic lor D.ibv_qp_min_rnr_timer,
          "making it ready to receive" )
    | Ready_to_send { psn; timeout; retries; reads } ->
        ( [
            (Q.qp_state, D.ibv_qps_rts);
            (Q.timeout, timeout);
            (Q.retry_cnt, retries);
            (Q.rnr_retry, rnr_retries);
            (Q.sq_psn, psn);
            (Q.max_rd_atomic, reads);
          ],
          D.ibv_qp_state lor D.ibv_qp_timeout lor D.ibv_qp_retry_cnt
          lor D.ibv_qp_rnr_retry lor D.ibv_qp_sq_psn
          lor D.ibv_qp_max_qp_rd_atomic,
          "making it ready to send" )
  in
  let* _ =
    command f
      (strf "queue pair %d: %s" h what)
      ~cmd:D.ib_user_verbs_cmd_modify_qp ~core:Q.sizeof
      ((Q.qp_handle, h) :: (Q.attr_mask, mask) :: fields)
  in
  Ok ()

let destroy f kind h =
  let r =
    match kind with
    | `Region ->
        command f "deregistering a region" ~cmd:D.ib_user_verbs_cmd_dereg_mr
          ~core:D.Dereg_mr.sizeof
          [ (D.Dereg_mr.mr_handle, h) ]
    | `Cq ->
        command f "destroying a completion queue"
          ~cmd:D.ib_user_verbs_cmd_destroy_cq ~core:D.Destroy_cq.sizeof
          ~out:D.Destroy_cq_resp.sizeof
          [ (D.Destroy_cq.cq_handle, h) ]
    | `Qp ->
        command f "destroying a queue pair" ~cmd:D.ib_user_verbs_cmd_destroy_qp
          ~core:D.Destroy_qp.sizeof ~out:D.Destroy_qp_resp.sizeof
          [ (D.Destroy_qp.qp_handle, h) ]
  in
  match r with Ok _ -> () | Error e -> failwith ("Rig_mlx5_uverbs: " ^ e)

(* Events *)

let event_bytes = 64 * D.Async_event_desc.sizeof

let drain fd size decode =
  let buf = R.params event_bytes in
  let rec go acc =
    let n = read_raw fd buf in
    if n <= 0 then List.rev acc
    else
      go
        (List.rev_append
           (List.init (n / size) (fun i -> decode buf (i * size)))
           acc)
  in
  go []

let async_event buf at : M.kernel_event option =
  let element = R.field buf (at + fst D.Async_event_desc.element, 8) in
  let kind = R.field buf (at + fst D.Async_event_desc.event_type, 4) in
  if kind = D.ibv_event_cq_err then Some (Cq_error element)
  else if kind = D.ibv_event_qp_fatal then
    Some (Qp_failed (element, "fatal error"))
  else if kind = D.ibv_event_qp_req_err then
    Some (Qp_failed (element, "request error"))
  else if kind = D.ibv_event_qp_access_err then
    Some (Qp_failed (element, "access error"))
  else if kind = D.ibv_event_device_fatal then Some Nic_failed
  else if kind = D.ibv_event_port_active then Some (Port_changed "active")
  else if kind = D.ibv_event_port_err then Some (Port_changed "down")
  else None

let wait f ms : M.kernel_event list =
  match poll f.async f.channel ms with
  | r when r <= 0 -> []
  | ready ->
      let async =
        if ready land 1 = 0 then []
        else
          List.filter_map Fun.id
            (drain f.async D.Async_event_desc.sizeof async_event)
      in
      let completions =
        if ready land 2 = 0 then []
        else
          drain f.channel D.Comp_event_desc.sizeof (fun buf at ->
              M.Completed
                (R.field buf (at + fst D.Comp_event_desc.cq_handle, 8)))
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

let open_ ~driver ~root name =
  let bus = Option.value ~default:"" (uevent root name "PCI_SLOT_NAME") in
  let* path =
    Option.to_result
      ~none:(strf "%s: no verbs file" name)
      (verbs_file root name)
  in
  let* fd = result path (open_raw path) in
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
