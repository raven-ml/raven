(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The calls follow Linux v6.12's drivers/infiniband/core: uverbs_ioctl.c (how
   the kernel reads the ioctl), uverbs_std_types_device.c (invoking a command),
   uverbs_cmd.c and uverbs_std_types_mr.c; and rdma-core v56.0's
   libibverbs/cmd_*.c, which makes the same calls. Each call lays out its
   request from the uapi structures of [Defs], and one stub runs the ioctl. *)

module D = Defs

let strf = Printf.sprintf
let ( let* ) = Result.bind

type buffer =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* The stubs raise [Unix.Unix_error] where a system call fails. *)

external ioctl : Unix.file_descr -> buffer -> unit
  = "caml_rig_mlx5_uverbs_ioctl"

external address : buffer -> int = "caml_rig_mlx5_uverbs_address"
external driver_id : string -> int = "caml_rig_mlx5_uverbs_driver"
external descr : int -> Unix.file_descr = "caml_rig_mlx5_uverbs_descr"

external map_raw : Unix.file_descr -> int -> int -> int
  = "caml_rig_mlx5_uverbs_map"

external unmap : int -> int -> unit = "caml_rig_mlx5_uverbs_unmap"
external page_size : unit -> int = "caml_rig_mlx5_uverbs_page_size"

external poll : Unix.file_descr -> Unix.file_descr -> int -> int
  = "caml_rig_mlx5_uverbs_poll"

external take : int -> Bytes.t -> int -> int -> int
  = "caml_rig_mlx5_uverbs_take"
[@@noalloc]

external fetch_and_add : int Atomic.t -> int -> int = "%atomic_fetch_add"

(* [guard what f] is [f ()], or [Error] saying that [what] failed and why. *)
let guard what f =
  match f () with
  | v -> Ok v
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "%s: %s" what (Unix.error_message e))

(* Structures as bytes *)

let set b at n v =
  for i = 0 to n - 1 do
    Bytes.set b (at + i) (Char.unsafe_chr ((v lsr (8 * i)) land 0xff))
  done

(* [encode size fields] is [size] zero bytes with each [((at, n), v)] of
   [fields] stored little-endian. *)
let encode size fields =
  let b = Bytes.make size '\000' in
  List.iter (fun ((at, n), v) -> set b at n v) fields;
  Bytes.to_string b

(* The little-endian unsigned integer in the [n] bytes at [at] of [s]. *)
let field s (at, n) =
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Char.code s.[at + i]
  done;
  !v

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

let devices root =
  let dir = class_dir root in
  match Sys.readdir dir with
  | a ->
      Array.sort compare a;
      Ok (Array.to_list a)
  | exception Sys_error why -> if Sys.file_exists dir then Error why else Ok []

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
  fd : Unix.file_descr;
  driver : int; (* the kernel's id of the device's driver *)
  root : string;
  name : string;
  mutable pd : int;
  mutable async : int; (* the descriptor of asynchronous events, or -1 *)
  mutable channel : int; (* the descriptor of completion events, or -1 *)
  maps : (int * int) list Atomic.t; (* pushed with no lock *)
  counts : (int * (int * int Atomic.t)) list Atomic.t;
      (* each queue's handle and count of completion events, by tag *)
  completions : pending;
  asyncs : pending;
}

(* Events taken from the kernel and not yet reported: the first [filled]
   bytes of [taken], of which the first [counted] are counted. *)
and pending = { taken : Bytes.t; mutable filled : int; mutable counted : int }

(* The verbs ioctl

   A call is a header, its attributes, then the bytes the attributes point to,
   all in one buffer that does not move while the kernel reads and writes it.
   An attribute carries up to 8 bytes inline, a handle, or the address of
   bytes. Every attribute is mandatory, so a kernel that does not know one
   refuses the call. *)

type attr =
  | In of int * string  (** Bytes the kernel reads. *)
  | Out of int * int  (** Room for bytes the kernel writes. *)
  | Handle of int * int  (** An object's handle. *)
  | Made of int  (** A new object's handle, which the kernel writes. *)

let put (b : buffer) at n v =
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set b (at + i)
      (Char.unsafe_chr ((v lsr (8 * i)) land 0xff))
  done

(* [call f what ~obj ~meth attrs] runs method [meth] of object [obj] over
   [attrs]: for each attribute, the bytes an [Out] received, the 8 bytes of
   the handle a [Made] received, or [""]. *)
let call f what ~obj ~meth attrs =
  let attrs = Array.of_list attrs in
  let head = D.Ioctl_hdr.sizeof + (Array.length attrs * D.Attr.sizeof) in
  let room = function
    | In (_, s) when String.length s > 8 -> String.length s
    | Out (_, n) -> n
    | In _ | Handle _ | Made _ -> 0
  in
  let total = Array.fold_left (fun n a -> n + room a) head attrs in
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout total in
  Bigarray.Array1.fill b '\000';
  let base = address b in
  let hdr (at, n) v = put b at n v in
  hdr D.Ioctl_hdr.length head;
  hdr D.Ioctl_hdr.object_id obj;
  hdr D.Ioctl_hdr.method_id meth;
  hdr D.Ioctl_hdr.num_attrs (Array.length attrs);
  hdr D.Ioctl_hdr.driver_id f.driver;
  let next = ref head in
  let slot i = D.Ioctl_hdr.sizeof + (i * D.Attr.sizeof) in
  let data i = slot i + fst D.Attr.data in
  Array.iteri
    (fun i a ->
      let attr (at, n) v = put b (slot i + at) n v in
      let id, len =
        match a with
        | In (id, s) -> (id, String.length s)
        | Out (id, n) -> (id, n)
        | Handle (id, _) | Made id -> (id, 0)
      in
      attr D.Attr.attr_id id;
      attr D.Attr.len len;
      attr D.Attr.flags D.uverbs_attr_f_mandatory;
      match a with
      | In (_, s) when len <= 8 ->
          String.iteri (fun k c -> Bigarray.Array1.set b (data i + k) c) s
      | In (_, s) ->
          String.iteri (fun k c -> Bigarray.Array1.set b (!next + k) c) s;
          put b (data i) 8 (base + !next);
          next := !next + len
      | Out (_, n) ->
          put b (data i) 8 (base + !next);
          next := !next + n
      | Handle (_, h) -> put b (data i) 8 h
      | Made _ -> ())
    attrs;
  let* () = guard what (fun () -> ioctl f.fd b) in
  let bytes at n = String.init n (fun k -> Bigarray.Array1.get b (at + k)) in
  let at = ref head in
  Ok
    (Array.mapi
       (fun i a ->
         match a with
         | Out (_, n) ->
             at := !at + n;
             bytes (!at - n) n
         | Made _ -> bytes (data i) 8
         | In (_, s) ->
             if String.length s > 8 then at := !at + String.length s;
             ""
         | Handle _ -> "")
       attrs)

let u64 v = encode 8 [ ((0, 8), v) ]
let u32 v = encode 4 [ ((0, 4), v) ]

(* [invoke f what cmd req n] runs the verbs command [cmd] on the request [req]:
   the command's answer, of [n] bytes, and the driver's answer of [uhw_out]
   bytes to its data [uhw]. *)
let invoke f what cmd req n ?(uhw = "") ?(uhw_out = 0) () =
  let opt b a = if b then [ a ] else [] in
  let attrs =
    [ In (D.uverbs_attr_write_cmd, u64 cmd); In (D.uverbs_attr_core_in, req) ]
    @ opt (n > 0) (Out (D.uverbs_attr_core_out, n))
    @ opt (uhw <> "") (In (D.uverbs_attr_uhw_in, uhw))
    @ opt (uhw_out > 0) (Out (D.uverbs_attr_uhw_out, uhw_out))
  in
  let* r =
    call f what ~obj:D.uverbs_object_device ~meth:D.uverbs_method_invoke_write
      attrs
  in
  let last = Array.length r - 1 in
  Ok ((if n > 0 then r.(2) else ""), if uhw_out > 0 then r.(last) else "")

(* The port the calls work on. *)
let port_num = 1

(* Makes reads of the event file [fd] answer at once when it holds
   nothing. *)
let nonblock what fd = guard what (fun () -> Unix.set_nonblock (descr fd))

let context f d n =
  let module C = D.Uverbs_get_context_resp in
  let* resp, answer =
    invoke f "making the context" D.ib_user_verbs_cmd_get_context
      (encode D.Uverbs_get_context.sizeof [])
      C.sizeof ~uhw:d ~uhw_out:n ()
  in
  f.async <- field resp C.async_fd;
  let* () = nonblock "reading the context's events" f.async in
  let module Ch = D.Uverbs_create_comp_channel_resp in
  let* resp, _ =
    invoke f "making the completion channel"
      D.ib_user_verbs_cmd_create_comp_channel
      (encode D.Uverbs_create_comp_channel.sizeof [])
      Ch.sizeof ()
  in
  f.channel <- field resp Ch.fd;
  let* () = nonblock "reading the completion channel" f.channel in
  let* resp, _ =
    invoke f "making the protection domain" D.ib_user_verbs_cmd_alloc_pd
      (encode D.Uverbs_alloc_pd.sizeof [])
      D.Uverbs_alloc_pd_resp.sizeof ()
  in
  f.pd <- field resp D.Uverbs_alloc_pd_resp.pd_handle;
  let module Q = D.Uverbs_query_device_resp in
  let* device, _ =
    invoke f "querying the device" D.ib_user_verbs_cmd_query_device
      (encode D.Uverbs_query_device.sizeof [])
      Q.sizeof ()
  in
  let module P = D.Uverbs_query_port_resp in
  let* p, _ =
    invoke f "querying port 1" D.ib_user_verbs_cmd_query_port
      (encode D.Uverbs_query_port.sizeof
         [ (D.Uverbs_query_port.port_num, port_num) ])
      P.sizeof ()
  in
  let* () =
    if field p P.state = D.ibv_port_active then Ok ()
    else Error "port 1 is not active"
  in
  let link = field p P.link_layer in
  let* port =
    if link = D.ibv_link_layer_infiniband then
      Ok (`Infiniband (field p P.lid))
    else if link = D.ibv_link_layer_ethernet then
      Ok (`Ethernet (gids f.root f.name))
    else
      Error
        (strf "port 1's link layer %d is neither InfiniBand nor Ethernet" link)
  in
  let mtu =
    let m = field p P.active_mtu in
    if m < D.ibv_mtu_256 then 0 else 256 lsl (m - D.ibv_mtu_256)
  in
  Ok
    {
      Path.answer;
      port;
      mtu;
      reads = field device Q.max_qp_init_rd_atom;
      served = field device Q.max_qp_rd_atom;
      cqes = field device Q.max_cqe + 1;
      message = field p P.max_msg_sz;
    }

let rec push a x =
  let l = Atomic.get a in
  if not (Atomic.compare_and_set a l (x :: l)) then push a x

let map f off n =
  let* at =
    guard (strf "mapping %d bytes at offset 0x%x" n off) (fun () ->
        map_raw f.fd off n)
  in
  push f.maps (at, n);
  Ok at

(* Objects *)

let reg_mr f ~flags b =
  let module R = D.Uverbs_reg_mr in
  let module A = D.Uverbs_reg_mr_resp in
  let at = address b and n = Bigarray.Array1.dim b in
  let* resp, _ =
    invoke f
      (strf "registering %d bytes at 0x%x" n at)
      D.ib_user_verbs_cmd_reg_mr
      (encode R.sizeof
         [
           (R.start, at);
           (R.length, n);
           (R.hca_va, at);
           (R.pd_handle, f.pd);
           (R.access_flags, flags);
         ])
      A.sizeof ()
  in
  ignore (Sys.opaque_identity b);
  Ok
    {
      Path.handle = field resp A.mr_handle;
      local = field resp A.lkey;
      remote = field resp A.rkey;
    }

let reg_dmabuf f ~flags ~fd off n =
  let* r =
    call f
      (strf "registering %d bytes of dma-buf %d" n fd)
      ~obj:D.uverbs_object_mr ~meth:D.uverbs_method_reg_dmabuf_mr
      [
        Made D.uverbs_attr_reg_dmabuf_mr_handle;
        Handle (D.uverbs_attr_reg_dmabuf_mr_pd_handle, f.pd);
        In (D.uverbs_attr_reg_dmabuf_mr_offset, u64 off);
        In (D.uverbs_attr_reg_dmabuf_mr_length, u64 n);
        In (D.uverbs_attr_reg_dmabuf_mr_iova, u64 off);
        In (D.uverbs_attr_reg_dmabuf_mr_fd, u32 fd);
        In (D.uverbs_attr_reg_dmabuf_mr_access_flags, u32 flags);
        Out (D.uverbs_attr_reg_dmabuf_mr_resp_lkey, 4);
        Out (D.uverbs_attr_reg_dmabuf_mr_resp_rkey, 4);
      ]
  in
  Ok
    {
      Path.handle = field r.(0) (0, 8);
      local = field r.(7) (0, 4);
      remote = field r.(8) (0, 4);
    }

(* The access flags of what peers may do, to a region or through a queue
   pair. *)
let peers (a : Path.access) =
  match a with
  | Local -> 0
  | Remote_write -> D.ib_uverbs_access_remote_write
  | Remote -> D.ib_uverbs_access_remote_write lor D.ib_uverbs_access_remote_read

(* The access flags of a region's access and order. *)
let flags (a : Path.access) (o : Path.order) =
  let peers = peers a in
  let order =
    match o with
    | In_order -> 0
    | Any_order -> D.ib_uverbs_access_relaxed_ordering
  in
  D.ib_uverbs_access_local_write lor peers lor order

let register f a o (m : Path.memory) =
  match m with
  | Host b -> reg_mr f ~flags:(flags a o) b
  | Dmabuf { fd; offset; bytes } ->
      reg_dmabuf f ~flags:(flags a o) ~fd offset bytes

let cq f ~entries ~tag d n =
  let module C = D.Uverbs_create_cq in
  let* resp, answer =
    invoke f
      (strf "making a completion queue of %d entries" entries)
      D.ib_user_verbs_cmd_create_cq
      (encode C.sizeof
         [
           (C.user_handle, tag);
           (C.cqe, entries - 1);
           (C.comp_channel, f.channel);
         ])
      D.Uverbs_create_cq_resp.sizeof ~uhw:d ~uhw_out:n ()
  in
  let handle = field resp D.Uverbs_create_cq_resp.cq_handle in
  let count = Atomic.make 0 in
  let rec add () =
    let cs = Atomic.get f.counts in
    if not (Atomic.compare_and_set f.counts cs ((tag, (handle, count)) :: cs))
    then add ()
  in
  add ();
  Ok (handle, (fun () -> Atomic.get count), answer)

let qp f ~cq ~sends ~receives ~tag d n =
  let module C = D.Uverbs_create_qp in
  let module R = D.Uverbs_create_qp_resp in
  let* resp, answer =
    invoke f
      (strf "making a queue pair of %d sends and %d receives" sends receives)
      D.ib_user_verbs_cmd_create_qp
      (encode C.sizeof
         [
           (C.user_handle, tag);
           (C.pd_handle, f.pd);
           (C.send_cq_handle, cq);
           (C.recv_cq_handle, cq);
           (C.max_send_wr, sends);
           (C.max_send_sge, 1);
           (C.max_recv_wr, receives);
           (C.max_recv_sge, Bool.to_int (receives > 0));
           (C.qp_type, D.ib_uverbs_qpt_rc);
         ])
      R.sizeof ~uhw:d ~uhw_out:n ()
  in
  Ok (field resp R.qp_handle, field resp R.qpn, answer)

(* What a queue pair answers a peer's send that finds no receive posted, after
   which the peer retries: 12, 0.64 ms, InfiniBand's [min_rnr_timer]
   encoding. A sender given 7 retries such a send without end, as
   [Rig_mlx5.Qp] states. Linux's drivers/infiniband/sw/rxe/rxe_comp.c is a
   datapoint: it never counts a retry down from 7. *)
let rnr_timer = 12
let rnr_retries = 7

(* The hops a RoCE v2 packet crosses before a router drops it. *)
let hop_limit = 64

(* The MTU enumeration of [bytes], a power of two from 256. *)
let rec mtu_code bytes =
  if bytes <= 256 then D.ibv_mtu_256 else 1 + mtu_code (bytes / 2)

let modify f h (t : Path.transition) =
  let module Q = D.Uverbs_modify_qp in
  let mask bits = (Q.attr_mask, List.fold_left ( lor ) D.ibv_qp_state bits) in
  let what, fields =
    match t with
    | Init { access } ->
        ( "initialising",
          [
            (Q.qp_state, D.ibv_qps_init);
            (Q.port_num, port_num);
            (Q.qp_access_flags, peers access);
            mask [ D.ibv_qp_pkey_index; D.ibv_qp_port; D.ibv_qp_access_flags ];
          ] )
    | Ready_to_receive { peer = { qp; psn; address; _ }; mtu; gid; served } ->
        let dest =
          match (address, gid) with
          | Lid lid, _ -> [ (Q.dest__dlid, lid) ]
          | Gid _, None ->
              invalid_arg "Rig_mlx5: a global address with no local one"
          | Gid _, Some index ->
              [
                (Q.dest__is_global, 1);
                (Q.dest__sgid_index, index);
                (Q.dest__hop_limit, hop_limit);
              ]
        in
        ( "making it ready to receive",
          [
            (Q.qp_state, D.ibv_qps_rtr);
            (Q.dest_qp_num, qp);
            (Q.rq_psn, psn);
            (Q.max_dest_rd_atomic, served);
            (Q.path_mtu, mtu_code mtu);
            (Q.min_rnr_timer, rnr_timer);
            (Q.dest__port_num, port_num);
            mask
              [
                D.ibv_qp_av;
                D.ibv_qp_path_mtu;
                D.ibv_qp_dest_qpn;
                D.ibv_qp_rq_psn;
                D.ibv_qp_max_dest_rd_atomic;
                D.ibv_qp_min_rnr_timer;
              ];
          ]
          @ dest )
    | Ready_to_send { psn; timeout; retries; reads } ->
        ( "making it ready to send",
          [
            (Q.qp_state, D.ibv_qps_rts);
            (Q.sq_psn, psn);
            (Q.timeout, timeout);
            (Q.retry_cnt, retries);
            (Q.rnr_retry, rnr_retries);
            (Q.max_rd_atomic, reads);
            mask
              [
                D.ibv_qp_timeout;
                D.ibv_qp_retry_cnt;
                D.ibv_qp_rnr_retry;
                D.ibv_qp_sq_psn;
                D.ibv_qp_max_qp_rd_atomic;
              ];
          ] )
  in
  let req = Bytes.of_string (encode Q.sizeof ((Q.qp_handle, h) :: fields)) in
  (* A global identifier's 16 bytes go as the wire sends them. *)
  (match t with
  | Ready_to_receive { peer = { address = Gid gid; _ }; _ } ->
      Bytes.blit_string gid 0 req (fst Q.dest__dgid) (snd Q.dest__dgid)
  | _ -> ());
  let what = strf "queue pair %d: %s" h what in
  let* _ =
    invoke f what D.ib_user_verbs_cmd_modify_qp (Bytes.to_string req) 0 ()
  in
  Ok ()

let destroy f kind h =
  let module R = D.Uverbs_dereg_mr in
  let module C = D.Uverbs_destroy_cq in
  let module Q = D.Uverbs_destroy_qp in
  let what, cmd, req, n =
    match kind with
    | `Region ->
        ( "deregistering a region",
          D.ib_user_verbs_cmd_dereg_mr,
          encode R.sizeof [ (R.mr_handle, h) ],
          0 )
    | `Cq ->
        ( "destroying a completion queue",
          D.ib_user_verbs_cmd_destroy_cq,
          encode C.sizeof [ (C.cq_handle, h) ],
          D.Uverbs_destroy_cq_resp.sizeof )
    | `Qp ->
        ( "destroying a queue pair",
          D.ib_user_verbs_cmd_destroy_qp,
          encode Q.sizeof [ (Q.qp_handle, h) ],
          D.Uverbs_destroy_qp_resp.sizeof )
  in
  let* _ = invoke f what cmd req n () in
  let rec forget () =
    let cs = Atomic.get f.counts in
    let rest = List.filter (fun (_, (c, _)) -> c <> h) cs in
    if not (Atomic.compare_and_set f.counts cs rest) then forget ()
  in
  if kind = `Cq then forget ();
  Ok ()

(* Events *)

let async_events =
  [
    (D.ibv_event_cq_err, fun e -> Path.Cq_error e);
    (D.ibv_event_qp_fatal, fun e -> Path.Qp_failed (e, "fatal error"));
    (D.ibv_event_qp_req_err, fun e -> Path.Qp_failed (e, "request error"));
    (D.ibv_event_qp_access_err, fun e -> Path.Qp_failed (e, "access error"));
    (D.ibv_event_device_fatal, fun _ -> Path.Nic_failed);
    (D.ibv_event_port_active, fun _ -> Path.Port_changed "active");
    (D.ibv_event_port_err, fun _ -> Path.Port_changed "down");
  ]

(* An asynchronous event of a kind the library reports, from its record at
   [at] of [b]. *)
let async_event b at =
  let module E = D.Uverbs_async_event_desc in
  let r = Bytes.sub_string b at E.sizeof in
  List.assoc_opt (field r E.event_type) async_events
  |> Option.map (fun make -> make (field r E.element))

(* Events

   The kernel's records are taken into a [pending] buffer of each event
   file, a completion counted at its queue's count, and the buffers emptied
   once [reported] says the caller holds the events, each in a step nothing
   can split. A queue's count is its arm's sequence: one the count misses
   makes the NIC ignore each later arm. An exception that a signal handler,
   finaliser or Gc.Memprof callback raises before [reported] leaves every
   record pending, a completion counted once: the next [wait] returns it
   again. *)

let comp_size = D.Uverbs_comp_event_desc.sizeof
let async_size = D.Uverbs_async_event_desc.sizeof

let pending size =
  { taken = Bytes.create (64 * size); filled = 0; counted = 0 }

let[@poll error] fill p fd size = p.filled <- take fd p.taken p.filled size

let[@poll error] count_one p c =
  ignore (fetch_and_add c 1);
  p.counted <- p.counted + comp_size

let[@poll error] reported f =
  f.completions.filled <- 0;
  f.completions.counted <- 0;
  f.asyncs.filled <- 0

let rec count_pending f =
  let p = f.completions in
  if p.counted < p.filled then begin
    let tag = Int64.to_int (Bytes.get_int64_le p.taken p.counted) in
    (match List.assoc_opt tag (Atomic.get f.counts) with
    | Some (_, c) -> count_one p c
    | None -> p.counted <- p.counted + comp_size);
    count_pending f
  end

let wait f ms =
  let c = f.completions and a = f.asyncs in
  let ms = if c.filled > 0 || a.filled > 0 then 0 else ms in
  let ready =
    match poll (descr f.async) (descr f.channel) ms with
    | r -> r
    | exception Unix.Unix_error _ -> 0
  in
  count_pending f;
  if ready land 1 <> 0 then fill a f.async async_size;
  if ready land 2 <> 0 then begin
    fill c f.channel comp_size;
    count_pending f
  end;
  let async =
    List.init (a.filled / async_size) (fun i -> i * async_size)
    |> List.filter_map (async_event a.taken)
  in
  let completions =
    List.init (c.filled / comp_size) (fun i ->
        let tag = Bytes.get_int64_le c.taken (i * comp_size) in
        Path.Completed (Int64.to_int tag))
  in
  (* An event file hangs up once the device is gone, and stays readable:
     the NIC failed. *)
  let gone = if ready land 4 = 0 then [] else [ Path.Nic_failed ] in
  async @ completions @ gone

let close f () =
  List.iter (fun (at, n) -> unmap at n) (Atomic.get f.maps);
  List.iter
    (fun fd -> if fd >= 0 then Unix.close (descr fd))
    [ f.channel; f.async ];
  Unix.close f.fd

let open_ ~root name =
  let bus = Option.value ~default:"" (uevent root name "PCI_SLOT_NAME") in
  let* path =
    Option.to_result
      ~none:(strf "%s: no verbs file" name)
      (verbs_file root name)
  in
  let* fd =
    guard path (fun () -> Unix.openfile path [ O_RDWR; O_CLOEXEC ] 0)
  in
  let* driver =
    match driver_id name with
    | d -> Ok d
    | exception Unix.Unix_error (e, _, _) ->
        Unix.close fd;
        Error (strf "%s: its driver's id: %s" name (Unix.error_message e))
  in
  let f =
    {
      fd;
      driver;
      root;
      name;
      pd = 0;
      async = -1;
      channel = -1;
      maps = Atomic.make [];
      counts = Atomic.make [];
      completions = pending comp_size;
      asyncs = pending async_size;
    }
  in
  Ok
    {
      Path.name;
      bus;
      page = page_size ();
      context = context f;
      map = map f;
      register = register f;
      cq = cq f;
      qp = qp f;
      modify = modify f;
      destroy = destroy f;
      wait = wait f;
      reported = (fun () -> reported f);
      close = close f;
    }
