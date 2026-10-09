(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Rig_mlx5_abi
module D = Defs

let strf = Printf.sprintf
let ( let* ) = Result.bind

external pages : int -> int = "caml_rig_mlx5_pages"
external free : int -> int -> unit = "caml_rig_mlx5_free"
external view : int -> int -> A.buffer = "caml_rig_mlx5_view"

external ring_stub : int -> int -> int -> int -> unit = "caml_rig_mlx5_ring"
[@@noalloc]

external consumed : int -> int -> unit = "caml_rig_mlx5_consumed" [@@noalloc]

external arm_stub : int -> int -> int -> int -> unit = "caml_rig_mlx5_arm"
[@@noalloc]

external acquire : unit -> unit = "caml_rig_mlx5_acquire" [@@noalloc]

(* Types *)

type memory =
  | Host of { address : int; bytes : int }
  | Dmabuf of { fd : int; offset : int; bytes : int }

type access = Local | Remote_write | Remote
type address = Lid of int | Gid of string
type endpoint = { qp : int; psn : int; address : address; mtu : int }

type kernel_event =
  | Completed of int
  | Cq_error of int
  | Qp_failed of int * string
  | Nic_failed
  | Port_changed of string

type transition =
  | Init
  | Ready_to_receive of {
      peer : endpoint;
      mtu : int;
      gid : int option;
      served : int;
    }
  | Ready_to_send of { psn : int; timeout : int; retries : int; reads : int }

type context = {
  answer : string;
  port : [ `Infiniband of int | `Ethernet of (int * string) list ];
  mtu : int;
  reads : int;
  served : int;
}

type registered = { handle : int; local : int; remote : int }

type path = {
  name : string;
  bus : string;
  page : int;
  context : string -> int -> (context, string) result;
  map : int -> int -> (int, string) result;
  register : access -> memory -> (registered, string) result;
  cq : entries:int -> tag:int -> string -> int -> (int * string, string) result;
  qp :
    cq:int ->
    entries:int ->
    tag:int ->
    string ->
    int ->
    (int * int * string, string) result;
  modify : int -> transition -> (unit, string) result;
  destroy : [ `Region | `Cq | `Qp ] -> int -> unit;
  wait : int -> kernel_event list;
  close : unit -> unit;
}

(* Memory a ring and its doorbell record live in: the ring's bytes rounded up to
   pages, then a page whose first 8 bytes are the record. *)
type ring = { mem : int; size : int; buf : A.buffer; record : int }

type t = {
  path : path;
  facts : context; (* the port and limits *)
  uar : int array; (* the host address of each system page of registers *)
  per_page : int; (* access regions per system page *)
  register_size : int;
  gid : int option; (* the index of the RoCE v2 identifier queue pairs use *)
  address : address; (* the port's, as peers name it *)
  random : Random.State.t;
  lock : Mutex.t; (* guards [cqs], [qps], [tags] and [random] *)
  closed : bool Atomic.t;
  mutable tags : int;
  mutable cqs : (int * cq) list; (* by tag *)
  mutable qps : (int * qp) list; (* by tag *)
}

and cq = {
  c_nic : t;
  c_handle : int;
  c_number : int;
  c_tag : int;
  c_entries : int;
  c_ring : ring;
  mutable count : int; (* completions consumed *)
  sequence : int Atomic.t; (* events raised *)
  members : qp list Atomic.t;
  mutable c_destroyed : bool;
}

and qp = {
  q_nic : t;
  q_cq : cq;
  q_handle : int;
  q_number : int;
  q_tag : int;
  q_entries : int;
  q_ring : ring;
  register : int; (* the host address of its doorbell register *)
  psn : int;
  posted : int Atomic.t; (* the producer count *)
  completed : int Atomic.t; (* entries whose slots are free again *)
  mutable rung : int;
  mutable connected : bool;
  mutable q_destroyed : bool;
}

let with_lock nic f =
  Mutex.lock nic.lock;
  Fun.protect ~finally:(fun () -> Mutex.unlock nic.lock) f

let alive fn nic =
  if Atomic.get nic.closed then
    invalid_arg (strf "Rig_mlx5.%s: %s is closed" fn nic.path.name)

let tag nic =
  with_lock nic (fun () ->
      nic.tags <- nic.tags + 1;
      nic.tags)

let round_up n m = (n + m - 1) / m * m
let rec pow2_at_least n k = if k >= n then k else pow2_at_least n (2 * k)

let ring_of nic bytes =
  let page = nic.path.page in
  let size = round_up bytes page + page in
  let mem = pages size in
  if mem < 0 then
    Error (strf "allocating %d bytes of rings: errno %d" size (-mem))
  else
    Ok { mem; size; buf = view mem bytes; record = mem + round_up bytes page }

let free_ring r = free r.mem r.size

(* Driver data *)

let data size fields =
  let b = Bytes.make size '\000' in
  List.iter
    (fun ((at, n), v) ->
      for i = 0 to n - 1 do
        Bytes.set b (at + i) (Char.unsafe_chr ((v lsr (8 * i)) land 0xff))
      done)
    fields;
  Bytes.to_string b

let read s (at, n) =
  if at + n > String.length s then 0
  else
    let v = ref 0 in
    for i = n - 1 downto 0 do
      v := (!v lsl 8) lor Char.code s.[at + i]
    done;
    !v

(* NICs *)

let name nic = nic.path.name
let bus nic = nic.path.bus

let link nic =
  match nic.facts.port with
  | `Infiniband _ -> `Infiniband
  | `Ethernet _ -> `Ethernet

(* A process asks for one access region's shared registers: queue pairs share
   them, and a doorbell, an 8-byte store to an uncached page, needs no register
   of its own. *)
let registers = D.mlx5_non_fp_bfregs_per_uar

(* The identifier RoCE v2 routes by: the first that maps an IPv4 address, else
   the first. *)
let ipv4_mapped (_, g) =
  String.length g = 16
  && String.sub g 0 10 = String.make 10 '\000'
  && String.sub g 10 2 = "\xff\xff"

let port_address = function
  | `Infiniband lid -> Ok (None, Lid lid)
  | `Ethernet [] -> Error "the port has no RoCE v2 global identifier"
  | `Ethernet (first :: _ as gs) ->
      let i, g = Option.value ~default:first (List.find_opt ipv4_mapped gs) in
      Ok (Some i, Gid g)

let make p =
  let attempt () =
    let module R = D.Ucontext_resp in
    let req =
      data D.Ucontext_req.sizeof
        [
          (D.Ucontext_req.total_num_bfregs, registers);
          (D.Ucontext_req.max_cqe_version, D.mlx5_cqe_version_v0);
          (D.Ucontext_req.lib_caps, D.mlx5_lib_cap_4k_uar);
        ]
    in
    let* facts = p.context req R.sizeof in
    let* gid, address = port_address facts.port in
    let resp = facts.answer in
    let per_page =
      if read resp R.log_uar_size = 0 && read resp R.num_uars_per_page = 0 then
        1
      else read resp R.num_uars_per_page
    in
    let total = read resp R.tot_bfregs in
    let per_sys_page = per_page * registers in
    if total < per_sys_page then
      Error
        (strf "the kernel gave %d doorbell registers, fewer than a page's %d"
           total per_sys_page)
    else
      let rec map_pages i acc =
        if i = total / per_sys_page then Ok (Array.of_list (List.rev acc))
        else
          let* at = p.map (A.Uar.mapping ~page:p.page i) p.page in
          map_pages (i + 1) (at :: acc)
      in
      let* uar = map_pages 0 [] in
      Ok
        {
          path = p;
          facts;
          uar;
          per_page;
          register_size = read resp R.bf_reg_size;
          gid;
          address;
          random = Random.State.make_self_init ();
          lock = Mutex.create ();
          closed = Atomic.make false;
          tags = 0;
          cqs = [];
          qps = [];
        }
  in
  match attempt () with
  | Ok nic -> Ok nic
  | Error e ->
      p.close ();
      Error (strf "%s: %s" p.name e)

let close nic =
  if not (Atomic.exchange nic.closed true) then begin
    nic.path.close ();
    let cqs, qps = with_lock nic (fun () -> (nic.cqs, nic.qps)) in
    List.iter (fun (_, q) -> free_ring q.q_ring) qps;
    List.iter (fun (_, c) -> free_ring c.c_ring) cqs
  end

(* Regions *)

type nic = t

module Region = struct
  type nonrec memory = memory =
    | Host of { address : int; bytes : int }
    | Dmabuf of { fd : int; offset : int; bytes : int }

  type nonrec access = access = Local | Remote_write | Remote

  type t = {
    nic : nic;
    handle : int;
    lkey : int;
    rkey : int;
    address : int;
    bytes : int;
    access : access;
    mutable gone : bool;
  }

  let register nic access m =
    alive "Region.register" nic;
    let address, bytes =
      match m with
      | Host { address; bytes } -> (address, bytes)
      | Dmabuf { fd; offset; bytes } ->
          if fd < 0 then
            invalid_arg (strf "Rig_mlx5.Region.register: descriptor %d" fd);
          (offset, bytes)
    in
    if bytes <= 0 || address < 0 then
      invalid_arg
        (strf "Rig_mlx5.Region.register: %d bytes at 0x%x" bytes address);
    let* (g : registered) = nic.path.register access m in
    Ok
      {
        nic;
        handle = g.handle;
        lkey = g.local;
        rkey = g.remote;
        address;
        bytes;
        access;
        gone = false;
      }

  let deregister r =
    alive "Region.deregister" r.nic;
    if not r.gone then begin
      r.gone <- true;
      r.nic.path.destroy `Region r.handle
    end

  let check fn r at n =
    alive ("Region." ^ fn) r.nic;
    if r.gone then invalid_arg (strf "Rig_mlx5.Region.%s: deregistered" fn);
    if at < 0 || n < 0 || at + n > r.bytes then
      invalid_arg
        (strf "Rig_mlx5.Region.%s: %d bytes at %d of %d" fn n at r.bytes)

  let local r at n : A.Entry.local =
    check "local" r at n;
    { address = r.address + at; bytes = n; key = r.lkey }

  let remote r at : A.Entry.remote =
    check "remote" r at 0;
    if at = r.bytes then
      invalid_arg (strf "Rig_mlx5.Region.remote: offset %d" at);
    if r.access = Local then
      invalid_arg "Rig_mlx5.Region.remote: a local region";
    { address = r.address + at; key = r.rkey }
end

(* Completion queues *)

(* A completion queue or queue pair that calls may use: its NIC open, itself not
   destroyed. *)
let live_cq fn cq =
  alive fn cq.c_nic;
  if cq.c_destroyed then invalid_arg (strf "Rig_mlx5.%s: destroyed" fn)

let live_qp fn q =
  alive fn q.q_nic;
  if q.q_destroyed then invalid_arg (strf "Rig_mlx5.%s: destroyed" fn)

module Cq = struct
  type t = cq

  let max_entries = 1 lsl 22

  let make nic n =
    alive "Cq.make" nic;
    if n < 1 || n > max_entries then
      invalid_arg (strf "Rig_mlx5.Cq.make: %d entries" n);
    let entries = pow2_at_least (max n 2) 1 in
    let* ring = ring_of nic (entries * A.Completion.size) in
    for i = 0 to entries - 1 do
      A.Completion.invalidate ring.buf (i * A.Completion.size)
    done;
    let tag = tag nic in
    let d =
      data D.Create_cq.sizeof
        [
          (D.Create_cq.buf_addr, ring.mem);
          (D.Create_cq.db_addr, ring.record);
          (D.Create_cq.cqe_size, A.Completion.size);
        ]
    in
    match nic.path.cq ~entries ~tag d D.Create_cq_resp.sizeof with
    | Error e ->
        free_ring ring;
        Error e
    | Ok (handle, resp) ->
        let cq =
          {
            c_nic = nic;
            c_handle = handle;
            c_number = read resp D.Create_cq_resp.cqn;
            c_tag = tag;
            c_entries = entries;
            c_ring = ring;
            count = 0;
            sequence = Atomic.make 0;
            members = Atomic.make [];
            c_destroyed = false;
          }
        in
        with_lock nic (fun () -> nic.cqs <- (tag, cq) :: nic.cqs);
        Ok cq

  (* A completion of entry [index] of [q] frees the slots up to it: the latest
     posted entry with those low 16 bits, which a ring of at most 2^15 entries
     makes the one. *)
  let complete q index =
    let posted = Atomic.get q.posted in
    let last = posted - 1 - ((posted - 1 - index) land 0xffff) in
    let rec raise_to v =
      let c = Atomic.get q.completed in
      if v > c && not (Atomic.compare_and_set q.completed c v) then raise_to v
    in
    raise_to (last + 1)

  let poll cq =
    live_cq "Cq.poll" cq;
    let at = cq.count land (cq.c_entries - 1) * A.Completion.size in
    if
      not
        (A.Completion.owned cq.c_ring.buf at ~count:cq.count
           ~entries:cq.c_entries)
    then None
    else begin
      acquire ();
      let c = A.Completion.read cq.c_ring.buf at in
      cq.count <- cq.count + 1;
      consumed (cq.c_ring.record + A.Doorbell.consumed) cq.count;
      List.iter
        (fun q -> if q.q_number = c.qp then complete q c.index)
        (Atomic.get cq.members);
      Some c
    end

  let arm cq =
    live_cq "Cq.arm" cq;
    let word, number =
      A.Doorbell.arm ~sequence:(Atomic.get cq.sequence) ~count:cq.count
        ~cq:cq.c_number
    in
    arm_stub
      (cq.c_ring.record + A.Doorbell.armed)
      (cq.c_nic.uar.(0) + A.Uar.cq_doorbell)
      word number

  let destroy cq =
    let nic = cq.c_nic in
    alive "Cq.destroy" nic;
    if (not cq.c_destroyed) && Atomic.get cq.members <> [] then
      invalid_arg "Rig_mlx5.Cq.destroy: a queue pair of it is not destroyed";
    if not cq.c_destroyed then begin
      cq.c_destroyed <- true;
      nic.path.destroy `Cq cq.c_handle;
      with_lock nic (fun () -> nic.cqs <- List.remove_assoc cq.c_tag nic.cqs);
      free_ring cq.c_ring
    end
end

(* Queue pairs *)

module Qp = struct
  type t = qp
  type nonrec address = address = Lid of int | Gid of string

  type nonrec endpoint = endpoint = {
    qp : int;
    psn : int;
    address : address;
    mtu : int;
  }

  let max_entries = 1 lsl 15

  (* A peer's request is retried 7 times, each after 4.096 us * 2^timeout: 1.07
     s on InfiniBand, 67 ms on RoCE, where a lost packet is likelier and a
     switch's buffers shorter. *)
  let retries = 7
  let timeout = function `Infiniband _ -> 18 | `Ethernet _ -> 14

  let make cq n =
    let nic = cq.c_nic in
    live_cq "Qp.make" cq;
    if n < 1 || n > max_entries then
      invalid_arg (strf "Rig_mlx5.Qp.make: %d entries" n);
    let entries = pow2_at_least n 1 in
    let held =
      List.fold_left (fun a q -> a + q.q_entries) 0 (Atomic.get cq.members)
    in
    if held + entries > cq.c_entries then
      invalid_arg
        (strf "Rig_mlx5.Qp.make: %d entries beside %d, past the queue's %d"
           entries held cq.c_entries);
    let* ring = ring_of nic (entries * A.Entry.size) in
    let tag = tag nic in
    let d =
      data D.Create_qp.sizeof
        [
          (D.Create_qp.buf_addr, ring.mem);
          (D.Create_qp.db_addr, ring.record);
          (D.Create_qp.sq_wqe_count, entries);
          (D.Create_qp.uidx, D.mlx5_ib_default_uidx);
        ]
    in
    let made =
      let* handle, number, resp =
        nic.path.qp ~cq:cq.c_handle ~entries ~tag d D.Create_qp_resp.sizeof
      in
      let page, at =
        A.Uar.register ~per_page:nic.per_page ~size:nic.register_size
          (read resp D.Create_qp_resp.bfreg_index)
      in
      if page >= Array.length nic.uar then begin
        nic.path.destroy `Qp handle;
        Error
          (strf "the kernel gave a doorbell register on page %d, unmapped" page)
      end
      else Ok (handle, number, nic.uar.(page) + at)
    in
    match made with
    | Error e ->
        free_ring ring;
        Error e
    | Ok (handle, number, register) ->
        let psn =
          with_lock nic (fun () -> Random.State.bits nic.random land 0xff_ffff)
        in
        let q =
          {
            q_nic = nic;
            q_cq = cq;
            q_handle = handle;
            q_number = number;
            q_tag = tag;
            q_entries = entries;
            q_ring = ring;
            register;
            psn;
            posted = Atomic.make 0;
            completed = Atomic.make 0;
            rung = 0;
            connected = false;
            q_destroyed = false;
          }
        in
        with_lock nic (fun () ->
            nic.qps <- (tag, q) :: nic.qps;
            Atomic.set cq.members (q :: Atomic.get cq.members));
        Ok q

  let number q =
    live_qp "Qp.number" q;
    q.q_number

  let endpoint q =
    live_qp "Qp.endpoint" q;
    let nic = q.q_nic in
    { qp = q.q_number; psn = q.psn; address = nic.address; mtu = nic.facts.mtu }

  let connect q (e : endpoint) =
    let nic = q.q_nic in
    live_qp "Qp.connect" q;
    if q.connected then invalid_arg "Rig_mlx5.Qp.connect: connected";
    (match (nic.facts.port, e.address) with
    | `Ethernet _, Gid g when String.length g <> 16 ->
        invalid_arg
          (strf "Rig_mlx5.Qp.connect: a global identifier of %d bytes"
             (String.length g))
    | `Infiniband _, Lid _ | `Ethernet _, Gid _ -> ()
    | _ -> invalid_arg "Rig_mlx5.Qp.connect: an address of another link layer");
    let p = nic.path and f = nic.facts in
    let steps =
      [
        Init;
        Ready_to_receive
          { peer = e; mtu = min f.mtu e.mtu; gid = nic.gid; served = f.served };
        Ready_to_send
          { psn = q.psn; timeout = timeout f.port; retries; reads = f.reads };
      ]
    in
    let* () =
      List.fold_left
        (fun r t -> Result.bind r (fun () -> p.modify q.q_handle t))
        (Ok ()) steps
    in
    q.connected <- true;
    Ok ()

  let room q =
    live_qp "Qp.room" q;
    q.q_entries - (Atomic.get q.posted - Atomic.get q.completed)

  let post q e =
    live_qp "Qp.post" q;
    if not q.connected then invalid_arg "Rig_mlx5.Qp.post: not connected";
    if room q = 0 then invalid_arg "Rig_mlx5.Qp.post: the ring is full";
    let index = Atomic.get q.posted in
    A.Entry.write q.q_ring.buf
      (index land (q.q_entries - 1) * A.Entry.size)
      ~qp:q.q_number ~index e;
    Atomic.set q.posted (index + 1)

  let ring q =
    live_qp "Qp.ring" q;
    let posted = Atomic.get q.posted in
    if posted <> q.rung then begin
      let last = (posted - 1) land (q.q_entries - 1) in
      ring_stub
        (q.q_ring.record + A.Doorbell.send)
        posted q.register
        (q.q_ring.mem + (last * A.Entry.size));
      q.rung <- posted
    end

  let destroy q =
    let nic = q.q_nic in
    alive "Qp.destroy" nic;
    if not q.q_destroyed then begin
      q.q_destroyed <- true;
      nic.path.destroy `Qp q.q_handle;
      with_lock nic (fun () ->
          nic.qps <- List.remove_assoc q.q_tag nic.qps;
          Atomic.set q.q_cq.members
            (List.filter (fun m -> m != q) (Atomic.get q.q_cq.members)));
      free_ring q.q_ring
    end
end

(* Events *)

type event = Completion of Cq.t | Failure of string | Port of string

let wait nic ~ms =
  alive "wait" nic;
  if ms < 0 then invalid_arg (strf "Rig_mlx5.wait: %d ms" ms);
  let find tags tag = with_lock nic (fun () -> List.assoc_opt tag (tags ())) in
  let qp_name tag =
    match find (fun () -> nic.qps) tag with
    | Some q -> strf "queue pair 0x%x" q.q_number
    | None -> "a destroyed queue pair"
  in
  let event = function
    | Completed tag -> (
        match find (fun () -> nic.cqs) tag with
        | Some c ->
            Atomic.incr c.sequence;
            Some (Completion c)
        | None -> None)
    | Cq_error tag ->
        let what =
          match find (fun () -> nic.cqs) tag with
          | Some c -> strf "completion queue 0x%x" c.c_number
          | None -> "a destroyed completion queue"
        in
        Some (Failure (strf "%s: %s overflowed" nic.path.name what))
    | Qp_failed (tag, kind) ->
        Some (Failure (strf "%s: %s: %s" nic.path.name (qp_name tag) kind))
    | Nic_failed -> Some (Failure (strf "%s: the NIC failed" nic.path.name))
    | Port_changed s -> Some (Port (strf "%s: port 1 %s" nic.path.name s))
  in
  List.filter_map event (nic.path.wait ms)
