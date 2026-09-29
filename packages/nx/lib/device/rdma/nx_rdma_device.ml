(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Nx_device.Buffer
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Remote = Nx_device_support.Remote

let buses ?remote () =
  Pci.scan ?remote ~vendor:0x14e4 ~class_:0x02 [ (0xffff, [ 0x1760 ]) ]

(* The largest WQE: its length is 32 bits. *)
let chunk = 1 lsl 30

type nic = {
  id : int; (* the order adapters are taken in *)
  machine : Nx_device.t; (* the host of the adapter's machine *)
  bus : int; (* its PCI bus number *)
  bnxt : Bnxt.t;
  lock : Mutex.t; (* its mailboxes, queues and regions *)
  memory : (nativeint, Mmio.t * int list) Hashtbl.t;
      (* its buffers and their pages, by address *)
  keys : (string * int, int) Hashtbl.t;
      (* the regions over GPU memory, by the name of the GPU's device and the
         allocation's address *)
  mutable dev : Nx_device.t option;
}

(* The opened adapters, by the host of their machine and index there. *)
let opened : ((Nx_device.t * int) * nic) list Atomic.t = Atomic.make []

(* The queue pairs, one per pair of devices, by their names: the queue pair on
   each device's adapter. *)
let pairs : (string * string, (nic * Bnxt.qp) * (nic * Bnxt.qp)) Hashtbl.t =
  Hashtbl.create 8

let pairs_lock = Mutex.create ()
let nics () = List.map snd (Atomic.get opened)

(* The bus number of a PCI address such as ["0000:41:00.0"]. *)
let bus_number address =
  match String.split_on_char ':' address with
  | [ _; b; _ ] | [ b; _ ] -> int_of_string ("0x" ^ b)
  | _ -> invalid_arg ("Nx_rdma_device: no bus in " ^ address)

(* The opened adapter of [host]'s machine closest to the function at [bus]. *)
let closest host bus =
  List.fold_left
    (fun best n ->
      if n.machine != host || n.dev = None then best
      else
        match best with
        | Some b when abs (b.bus - bus) <= abs (n.bus - bus) -> best
        | _ -> Some n)
    None (nics ())

(* Regions *)

(* The region of [n] over the allocation [b] lies in, registered at its first
   use at the allocation's device address, and deregistered when its device
   frees it. [n] is locked. *)
let key n (b : B.t) (dma : Nx_device.dma) =
  let alloc = Nativeint.to_int (B.address b) - B.offset b in
  let owner = Nx_device.name (B.device b) in
  match Hashtbl.find_opt n.keys (owner, alloc) with
  | Some k -> k
  | None ->
      let log = Bnxt.log_page ~va:alloc dma.pages in
      let size = List.fold_left (fun s (_, k) -> s + k) 0 dma.pages in
      let k =
        Bnxt.register_mem n.bnxt
          (Bnxt.pages_of ~log dma.pages)
          ~size ~log_page:log ~va:alloc
      in
      Hashtbl.replace n.keys (owner, alloc) k;
      B.on_free b (fun () ->
          Mutex.protect n.lock (fun () ->
              Hashtbl.remove n.keys (owner, alloc);
              Bnxt.unregister_mem n.bnxt k));
      k

(* Queue pairs *)

(* The queue pair of the devices [a] and [b], through the adapters [na] and
   [nb], created and connected at its first use. *)
let queue_pair a na b nb =
  let name = Nx_device.name in
  let key, flip =
    if name a <= name b then ((name a, name b), false)
    else ((name b, name a), true)
  in
  let (n1, q1), (_, q2) =
    Mutex.protect pairs_lock (fun () ->
        match Hashtbl.find_opt pairs key with
        | Some p -> p
        | None ->
            let n1, n2 = if flip then (nb, na) else (na, nb) in
            let q1 = Mutex.protect n1.lock (fun () -> Bnxt.create_qp n1.bnxt) in
            let q2 = Mutex.protect n2.lock (fun () -> Bnxt.create_qp n2.bnxt) in
            let connect (n, q) (n', q') =
              Mutex.protect n.lock (fun () ->
                  Bnxt.connect q ~qpn:q'.Bnxt.qpn ~gid:n'.bnxt.Bnxt.gid
                    ~mac:n'.bnxt.mac)
            in
            connect (n1, q1) (n2, q2);
            connect (n2, q2) (n1, q1);
            let p = ((n1, q1), (n2, q2)) in
            Hashtbl.replace pairs key p;
            p)
  in
  if n1 == na then (q1, q2) else (q2, q1)

(* The link: per chunk, the destination's adapter posts the receive and the
   source's the send, and both completions are waited for. *)
let move ns nd ~src ~dst =
  let s = B.device src and d = B.device dst in
  let qs, qd = queue_pair s ns d nd in
  let timeout_ms =
    Int.min
      (Nx_device.timeout (Option.get ns.dev))
      (Nx_device.timeout (Option.get nd.dev))
  in
  let first, second = if ns.id < nd.id then (ns, nd) else (nd, ns) in
  Mutex.protect first.lock @@ fun () ->
  Mutex.protect second.lock @@ fun () ->
  let ks = key ns src (B.dma src) and kd = key nd dst (B.dma dst) in
  let n = B.nbytes src in
  let at b off = Nativeint.to_int (B.address b) + off in
  let rec go off =
    if off < n then begin
      let len = Int.min chunk (n - off) in
      let r = Bnxt.post_recv qd ~va:(at dst off) ~key:kd len in
      let t = Bnxt.post_send qs ~va:(at src off) ~key:ks len in
      Bnxt.poll qs ~send:true ~timeout_ms t;
      Bnxt.poll qd ~send:false ~timeout_ms r;
      go (off + len)
    end
  in
  go 0

(* A copy between two machines' GPUs is this adapter's when it is the one
   closest to the source's GPU, and the destination's machine has one open. *)
let link nic ~src ~dst =
  let s = B.device src and d = B.device dst in
  let hs = Nx_device.host_of s and hd = Nx_device.host_of d in
  if hs == hd then None
  else
    match (B.dma src, B.dma dst) with
    | exception Invalid_argument _ -> None
    | ds, dd -> (
        match
          (closest hs (bus_number ds.bus), closest hd (bus_number dd.bus))
        with
        | Some ns, Some nd when ns == nic ->
            Some
              {
                Nx_device.through = [ Option.get ns.dev; Option.get nd.dev ];
                move = move ns nd;
              }
        | _ -> None)

(* Opening *)

let name ~machine i =
  let local = if i = 0 then "RDMA" else Printf.sprintf "RDMA:%d" i in
  match Nx_remote_device.remote machine with
  | None -> local
  | Some r -> local ^ "@" ^ Remote.name r

(* Locked system memory of the adapter's machine, at the same address there for
   the host and the adapter. *)
let allocator n pci =
  let alloc bytes =
    match Pci.alloc_sysmem pci bytes with
    | exception Failure _ -> None
    | m, pages ->
        let a = Mmio.address m in
        Mutex.protect n.lock (fun () -> Hashtbl.replace n.memory a (m, pages));
        Some { Nx_device.host = Some a; device = a; handle = a }
  in
  let free (m : Nx_device.memory) =
    match
      Mutex.protect n.lock (fun () -> Hashtbl.find_opt n.memory m.device)
    with
    | Some (mmio, _) ->
        Mutex.protect n.lock (fun () -> Hashtbl.remove n.memory m.device);
        Pci.free_sysmem pci mmio
    | None -> ()
  in
  { Nx_device.alloc; free }

(* The pages of a buffer of the adapter, as other functions reach them. *)
let dma n pci (m : Nx_device.memory) =
  match Mutex.protect n.lock (fun () -> Hashtbl.find_opt n.memory m.device) with
  | None -> Error "no buffer of this adapter"
  | Some (_, pages) ->
      let page = Pci.page pci in
      Ok
        {
          Nx_device.bus = Pci.bus pci;
          pages = List.map (fun p -> (p, page)) pages;
        }

let master_off pci =
  let command = 0x04 and master = 0x04 in
  Pci.write_config pci command 2 (Pci.read_config pci command 2 land lnot master)

(* At exit a healthy adapter unregisters from its firmware; every adapter stops
   reaching the memory the process releases. *)
let finalize n pci ~failed =
  Fun.protect
    ~finally:(fun () -> master_off pci)
    (fun () ->
      if not failed then Mutex.protect n.lock (fun () -> Bnxt.fini n.bnxt))

let ids = Atomic.make 0

let open_nic ~machine ~remote index =
  let buses = buses ?remote () in
  let bus =
    match List.nth_opt buses index with
    | Some b -> b
    | None ->
        failwith
          (Printf.sprintf "no adapter %d; there are %d" index
             (List.length buses))
  in
  let pci = Pci.take ?remote ~lock:"bnxt" bus in
  match
    let bnxt = Bnxt.boot pci in
    let n =
      {
        id = Atomic.fetch_and_add ids 1;
        machine;
        bus = bus_number bus;
        bnxt;
        lock = Mutex.create ();
        memory = Hashtbl.create 8;
        keys = Hashtbl.create 8;
        dev = None;
      }
    in
    n.dev <-
      Some
        (Nx_device.make ~name:(name ~machine index) ~arch:"" ~host:machine
           ~budget:max_int ~memory:(allocator n pci) ~link:(link n)
           ~dma:(dma n pci) ~finalize:(finalize n pci) ());
    n
  with
  | n -> n
  | exception e ->
      (try master_off pci with Failure _ -> ());
      Pci.release pci;
      raise e

let lock = Mutex.create ()

let count ?(host = Nx_device.host) () =
  List.length (buses ?remote:(Nx_remote_device.remote host) ())

let get ?(host = Nx_device.host) i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_rdma_device.get: %d < 0" i);
  let remote = Nx_remote_device.remote host in
  Mutex.protect lock (fun () ->
      match List.assoc_opt (host, i) (Atomic.get opened) with
      | Some n -> Ok (Option.get n.dev)
      | None -> (
          match open_nic ~machine:host ~remote i with
          | n ->
              Atomic.set opened (((host, i), n) :: Atomic.get opened);
              Ok (Option.get n.dev)
          | exception (Failure msg | Sys_error msg | Invalid_argument msg) ->
              Error ("RDMA: " ^ msg)
          | exception Unix.Unix_error (e, f, arg) ->
              Error
                (Printf.sprintf "RDMA: %s %s: %s" f arg (Unix.error_message e))))

let v ?host i =
  match get ?host i with Ok d -> d | Error msg -> invalid_arg msg
