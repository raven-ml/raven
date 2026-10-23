(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A GPU's memory over the GPUs of a fixture tree, taken physically, which needs
   Linux. The GPU's page tables live in a table that counts every access, so a
   test sees what reaches the GPU after it is given back. *)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf
let ranges = list (pair hex hex)

(* A size of memory the tests map, a multiple of every machine's page. *)
let page = 16 * kib

(* Addresses: the tables' virtual addresses from [tables_base], and the space
   the GPUs share inside them. *)
let tables_base = 1 lsl 40
let space_base = 1 lsl 41
let space_length = 1 lsl 30

(* Adjacent ranges merged, so that two spellings of the same bytes compare. *)
let merge l =
  List.fold_right
    (fun (a, n) acc ->
      match acc with
      | (b, m) :: rest when a + n = b -> (a, n + m) :: rest
      | _ -> (a, n) :: acc)
    l []

let gpu_memory = 64 * mib

let large_pages =
  [ (2 * mib, 2 * mib); (64 * kib, 64 * kib); (4 * kib, 4 * kib) ]

(* How many 4 KiB blocks the GPU's main pool and its space hand out before
   [None], all given back after. Memory given back in full hands out the same
   blocks again, so two equal counts show no leak. *)
let drain alloc free =
  let rec go acc =
    match alloc () with Some a -> go (a :: acc) | None -> acc
  in
  let taken = go [] in
  List.iter free taken;
  List.length taken

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with Memory.Gpu -> "Gpu" | Bar -> "Bar" | Host -> "Host")

let pp_region ppf r = Format.fprintf ppf "the region at 0x%x" (Memory.address r)

let pp_source ppf s =
  match s with
  | Memory.Allocated -> Format.pp_print_string ppf "Allocated"
  | Borrowed a -> Format.fprintf ppf "Borrowed 0x%x" a
  | Peer r -> Format.fprintf ppf "Peer (%a)" pp_region r

(* A peer's region is the very region its owner gave. *)
let source =
  Testable.make ~pp:pp_source ~equal:(fun a b ->
      match (a, b) with
      | Memory.Peer r, Memory.Peer r' -> r == r'
      | Peer _, _ | _, Peer _ -> false
      | a, b -> a = b)

let pp_target ppf t =
  Format.pp_print_string ppf
    (match t with
    | Page_table.Gpu -> "Gpu"
    | System -> "System"
    | Peer i -> strf "Peer %d" i)

let target = Testable.make ~pp:pp_target ~equal:( = )

(* On a machine's files

   GPUs of a fixture tree, taken physically, which needs Linux, with page tables
   in the fake format. Their system memory lies in the tree's memory files, at
   the frames its page map gives: those of the first 64 MiB of the space, and
   those of 2 MiB past it, where the tests allocate memory of the machine that
   GPUs borrow. *)

let lend_base = space_base + space_length

(* The page map gives the first [space_frames] bytes of the space the frames
   from [first_frame] on, one after the other. *)
let space_frames = 64 * mib
let first_frame = 0x20_0000

type on_tree = {
  t_fn : Function.t;
  t_g : Tables.memory;
  t_tables : Page_table.t;
  t_memory : Memory.t;
}

let needs_linux () =
  if not on_linux then skip ~reason:"flock on a function's file needs Linux" ()

(* The machine of a new tree of [n] GPUs under one root port, as root reads
   it, its root and their buses, with the space and the lent range
   reserved. *)
let tree_machine ?(gpu = fun _ bus -> Tree.gpu bus) ?(others = [])
    ?(space = space_base) n =
  needs_linux ();
  let buses = List.init n (fun i -> strf "0000:%02x:00.0" (3 + i)) in
  let root = Tree.make ~privileged:true (List.mapi gpu buses @ others) in
  let m = Machine.at root in
  List.iter
    (fun (base, n) ->
      match Machine.reserve m ~base n with
      | Ok () -> ()
      | Error why -> skip ~reason:why ())
    [ (space, space_length); (lend_base, 2 * mib) ];
  let page = Machine.page m in
  Tree.pagemap root ~page space
    (List.init (space_frames / page) (fun i -> first_frame + i));
  Tree.pagemap root ~page lend_base
    (List.init (2 * mib / page) (fun i -> 0x30_0000 + i));
  (root, m, buses)

(* The GPU at [bus] of [m], its function taken, released at the end of the
   test. *)
let take_gpu ?link ?(memory = gpu_memory) ?(pa_bits = 52)
    ?(tables = Page_table.Pool) ?(format = Fun.id) ~space m bus =
  let fn = require_ok (Function.take m bus) in
  let g = Tables.memory () in
  let tables =
    Page_table.create ~base:tables_base
      (format { (Tables.format g) with pa_bits })
      space ~memory ~boot:mib ~tables ~pages:large_pages
  in
  Page_table.booted tables;
  {
    t_fn = fn;
    t_g = g;
    t_tables = tables;
    t_memory = Memory.create ?link fn tables ~bar:0;
  }

(* [with_gpus n f] is [f root m gpus], [gpus] the [n] GPUs of a new tree sharing
   a space, released after. *)
let with_gpus ?(memories = []) n f =
  let root, m, buses = tree_machine n in
  let space = Space.create ~base:space_base space_length in
  let gpus =
    List.mapi
      (fun i bus -> take_gpu ?memory:(List.nth_opt memories i) ~space m bus)
      buses
  in
  Fun.protect
    ~finally:(fun () -> List.iter (fun x -> Function.release x.t_fn) gpus)
    (fun () -> f root m gpus)

(* [with_gpu f] is [f root m x], [x] the one GPU of a new tree, released
   after. *)
let with_gpu ?memory ?tables ?format f =
  let root, m, buses = tree_machine 1 in
  let space = Space.create ~base:space_base space_length in
  let x = take_gpu ?memory ?tables ?format ~space m (List.hd buses) in
  Fun.protect
    ~finally:(fun () -> Function.release x.t_fn)
    (fun () -> f root m x)

(* A mapping the machine and the tables have room for. *)
let mapped r = require_some ~msg:"room" (require_ok r)

let given = function
  | Ok (Some r) -> r
  | Ok None -> fail "no room"
  | Error why -> fail why

let tree_capacity x =
  let t = x.t_tables and s = Page_table.space x.t_tables in
  ( drain (fun () -> Page_table.palloc ~zero:false t 4096) (Page_table.pfree t),
    drain (fun () -> Space.alloc s page) (Space.free s) )

(* The memory files under [root], without the markers of their reachers, and the
   bytes they hold. *)
let host_files root =
  Sys.readdir (Filename.concat root "dev/hugepages")
  |> Array.to_list
  |> List.filter (fun f -> not (String.contains f '.'))

let held_bytes root =
  List.fold_left
    (fun n f -> n + Tree.stored (Filename.concat root ("dev/hugepages/" ^ f)))
    0 (host_files root)

(* Memory of the machine that a GPU borrows: [n] bytes [alloc_dma] gave its
   function in the lent range. *)
let lend x n = given (Function.alloc_dma ~va:lend_base x.t_fn n)

let unpinned ~msg x (a, n) =
  raises_match ~msg (Exn.invalid_arg ~substring:"") (fun () ->
      Function.unpin x.t_fn a n)

(* The regions of each source on GPU [x]: [Gpu], [Bar] and [Host] memory, the
   pages of memory [alloc_dma] gave borrowed, and the [Gpu] and [Host] memory of
   the peer [o] mapped. *)
type from = Of_kind of Memory.kind | Borrowing | Peer_of of Memory.kind

let pp_from ppf = function
  | Of_kind k -> pp_kind ppf k
  | Borrowing -> Format.pp_print_string ppf "Borrowed"
  | Peer_of k -> Format.fprintf ppf "Peer of %a" pp_kind k

let froms =
  [
    Of_kind Memory.Gpu;
    Of_kind Bar;
    Of_kind Host;
    Borrowing;
    Peer_of Gpu;
    Peer_of Host;
  ]

(* [region x o lent from] is a new region of [x] from [from], and the range its
   function pinned for it, if any. *)
let region x o lent from =
  match from with
  | Of_kind k -> (given (Memory.alloc x.t_memory k (64 * kib)), None)
  | Borrowing ->
      let a = Window.address lent in
      (mapped (Memory.map_host x.t_memory a page), Some (a, page))
  | Peer_of k ->
      let r = given (Memory.alloc o.t_memory k (64 * kib)) in
      let p = mapped (Memory.map_peer x.t_memory r) in
      (p, if k = Host then Some (Memory.address r, 64 * kib) else None)

(* A free gives back what the region took: the GPU's memory and addresses, its
   system memory, and its pin. *)
let test_round_trip from =
  with_gpus 2 @@ fun root _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let lent, _ = lend x page in
  let before = tree_capacity x and bytes = held_bytes root in
  let r, pinned = region x o lent from in
  let host = Memory.host r in
  Memory.free x.t_memory r;
  (match from with
  | Peer_of _ -> (
      match Memory.source r with
      | Peer origin -> Memory.free o.t_memory origin
      | _ -> fail "a peer's region")
  | _ -> ());
  equal ~msg:"the GPU's memory and addresses" (pair int int) before
    (tree_capacity x);
  equal ~msg:"the system memory held" int bytes (held_bytes root);
  Option.iter (unpinned ~msg:"its pin" x) pinned;
  if from = Of_kind Host then
    raises_match ~msg:"its system memory" (Exn.invalid_arg ~substring:"")
      (fun () -> Function.free_dma x.t_fn (require_some host));
  Function.free_dma x.t_fn lent

(* Bytes that start past a page's first byte and end inside the next page. *)
let test_unaligned () =
  with_gpus 1 @@ fun _ m gpus ->
  let x = List.hd gpus and page = Machine.page m in
  let lent, runs = lend x (2 * page) in
  let a = Window.address lent + 100 and n = page in
  let r = mapped (Memory.map_host x.t_memory a n) in
  equal ~msg:"borrowed at a" source (Borrowed a) (Memory.source r);
  let pa = fst (List.hd runs) + 100 in
  let held = merge (snd (Memory.pages r)) in
  equal ~msg:"from byte a's physical address" (option hex) (Some pa)
    (Option.map fst (List.nth_opt held 0));
  satisfies ~msg:"the pages that hold the bytes"
    ~claim:
      (strf "one range from 0x%x, within the lent pages, of %d bytes or more" pa
         n)
    ranges
    (function
      | [ (b, k) ] ->
          b = pa && k >= n && b + k <= fst (List.hd runs) + (2 * page)
      | _ -> false)
    held;
  equal ~msg:"its address is byte a's, 100 bytes into its page" int 100
    (Memory.address r mod page);
  let w = require_some ~msg:"a window" (Memory.host r) in
  equal ~msg:"the window on the n bytes" (pair hex int) (a, n)
    (Window.address w, Window.length w);
  Memory.free x.t_memory r;
  unpinned ~msg:"its pin" x (Window.address lent, 2 * page);
  Function.free_dma x.t_fn lent

(* Who reaches whose memory: [x] maps [o]'s GPU memory iff [reaches x o]. *)
type pairing = {
  memory : int;  (** [o]'s memory, which a BAR of 256 MiB reaches below it. *)
  links : (int64 * int64) option;  (** The fabrics of [x] and [o]. *)
  apart : [ `No | `Machines | `Spaces ];
  reaches : bool;
}

let pairings =
  [
    ( "two GPUs of one tree, the BAR as large as the memory",
      { memory = gpu_memory; links = None; apart = `No; reaches = true } );
    ( "a BAR smaller than the memory",
      { memory = 512 * mib; links = None; apart = `No; reaches = false } );
    ( "links of one fabric, the BAR smaller than the memory",
      { memory = 512 * mib; links = Some (7L, 7L); apart = `No; reaches = true }
    );
    ( "links of two fabrics, the BAR smaller than the memory",
      {
        memory = 512 * mib;
        links = Some (7L, 8L);
        apart = `No;
        reaches = false;
      } );
    ( "links of two fabrics, the BAR as large as the memory",
      {
        memory = gpu_memory;
        links = Some (7L, 8L);
        apart = `No;
        reaches = true;
      } );
    ( "two machines",
      { memory = gpu_memory; links = None; apart = `Machines; reaches = false }
    );
    ( "two spaces",
      { memory = gpu_memory; links = None; apart = `Spaces; reaches = false } );
  ]

let test_reaches (_, c) =
  let _, m, buses = tree_machine 2 in
  let space = Space.create ~base:space_base space_length in
  let link fabric node = { Memory.fabric; node } in
  let x_link, o_link =
    match c.links with
    | Some (fx, fo) -> (Some (link fx 1), Some (link fo 2))
    | None -> (None, None)
  in
  let x = take_gpu ?link:x_link ~space m (List.nth buses 0) in
  let o_machine, o_bus =
    match c.apart with
    | `Machines ->
        let _, m', buses' = tree_machine 1 in
        (m', List.hd buses')
    | `No | `Spaces -> (m, List.nth buses 1)
  in
  let o_space =
    match c.apart with
    | `Spaces -> Space.create ~base:space_base space_length
    | `No | `Machines -> space
  in
  let o =
    take_gpu ?link:o_link ~memory:c.memory ~space:o_space o_machine o_bus
  in
  Fun.protect ~finally:(fun () ->
      Function.release x.t_fn;
      Function.release o.t_fn)
  @@ fun () ->
  let x_reach = Memory.reach x.t_memory in
  let again = Memory.create ?link:x_link x.t_fn x.t_tables ~bar:0 in
  equal ~msg:"reaches itself" bool false (Memory.reaches x_reach x_reach);
  equal ~msg:"reaches itself, opened again" bool false
    (Memory.reaches x_reach (Memory.reach again));
  equal ~msg:"reaches" bool c.reaches
    (Memory.reaches x_reach (Memory.reach o.t_memory));
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let peer = Memory.map_peer x.t_memory r in
  equal ~msg:"maps" bool c.reaches (Result.is_ok peer);
  let over_link =
    match c.links with Some (fx, fo) -> Int64.equal fx fo | None -> false
  in
  Result.iter
    (fun p ->
      let p = require_some ~msg:"room" p in
      equal ~msg:"its target" target
        (if over_link then Page_table.Peer 2 else System)
        (fst (Memory.pages p));
      Memory.free x.t_memory p)
    peer;
  Memory.free o.t_memory r

(* Whether the bus carries one GPU's requests to the other's BAR, as Linux's
   p2pdma decides: a bridge above both without ACS redirection, or else a CPU
   or host bridges that carry them through the root complex. *)
(* The [i]th GPU under a root port of its own, or a host bridge of its own. *)
let port_of i bus = Tree.gpu ~bridges:[ strf "0000:00:0%d.0" (i + 1) ] bus

let host_of i bus =
  let h = if i = 0 then "00" else "40" in
  Tree.gpu ~host:("pci0000:" ^ h) ~bridges:[ strf "0000:%s:01.0" h ] bus

(* A host bridge's first device, at 00.0 of its root bus, Intel's [device]. *)
let first_device h device root =
  let d = strf "sys/devices/pci0000:%s/0000:%s:00.0/" h h in
  Tree.add root (d ^ "vendor") "0x8086\n";
  Tree.add root (d ^ "device") (strf "0x%04x\n" device);
  Tree.add root (d ^ "class") "0x060000\n"

let root_port = "sys/devices/pci0000:00/0000:00:01.0/config"

(* The first GPU's port, 00:01.0, made Intel's [device], with a PCI Express
   capability of port type [kind]: 4 for a root port, 6 for a downstream
   port. *)
let intel_port device kind root =
  let d = "sys/devices/pci0000:00/0000:00:01.0/" in
  Tree.add root (d ^ "vendor") "0x8086\n";
  Tree.add root (d ^ "device") (strf "0x%04x\n" device);
  let b = Bytes.make 4096 '\000' in
  Bytes.set_uint8 b 0x34 0x40;
  Bytes.set_uint8 b 0x40 0x10;
  Bytes.set_uint16_le b 0x42 (kind lsl 4);
  Tree.add root root_port (Bytes.to_string b)

(* A root port whose ACS, the first extended capability, redirects peer
   requests. *)
let acs_redirect root =
  let b = Bytes.make 4096 '\000' in
  Bytes.set_int32_le b 0x100 (Int32.of_int (0x0d lor (1 lsl 16)));
  Bytes.set_uint16_le b 0x106 0x04;
  Tree.add root root_port (Bytes.to_string b)

let routes =
  [
    ("under one root port", (fun _ bus -> Tree.gpu bus), ignore, true);
    ( "under two root ports of a host bridge Linux does not list",
      port_of,
      ignore,
      false );
    ( "under two root ports of a host bridge Linux lists",
      port_of,
      first_device "00" 0x2020,
      true );
    ( "under two root ports of a host bridge Linux lists for itself",
      port_of,
      first_device "00" 0x3c00,
      true );
    ( "under two root ports, the first a root port Linux lists",
      port_of,
      intel_port 0x2030 4,
      true );
    ( "under two root ports, the first a downstream port of a listed device",
      port_of,
      intel_port 0x2030 6,
      false );
    ( "under two host bridges Linux lists for themselves",
      host_of,
      (fun root ->
        first_device "00" 0x3c00 root;
        first_device "40" 0x3c00 root),
      false );
    ( "under two host bridges Linux lists",
      host_of,
      (fun root ->
        first_device "00" 0x2020 root;
        first_device "40" 0x2020 root),
      true );
    ( "under a root port whose ACS redirects",
      (fun _ bus -> Tree.gpu bus),
      acs_redirect,
      false );
    ( "under a root port whose ACS cannot be read",
      (fun _ bus -> Tree.gpu bus),
      (fun root -> Tree.add root root_port (String.make 64 '\000')),
      false );
    ( "under two root ports, on an AMD CPU from Zen on",
      port_of,
      (fun root ->
        Tree.add root "proc/cpuinfo"
          "vendor_id\t: AuthenticAMD\ncpu family\t: 25\n"),
      true );
    ( "under two root ports, on an AMD CPU before Zen",
      port_of,
      (fun root ->
        Tree.add root "proc/cpuinfo"
          "vendor_id\t: AuthenticAMD\ncpu family\t: 21\n"),
      false );
  ]

let test_route (_, gpu, setup, carried) =
  let root, m, buses = tree_machine ~gpu 2 in
  setup root;
  let space = Space.create ~base:space_base space_length in
  let x = take_gpu ~space m (List.nth buses 0) in
  let o = take_gpu ~space m (List.nth buses 1) in
  Fun.protect ~finally:(fun () ->
      Function.release x.t_fn;
      Function.release o.t_fn)
  @@ fun () ->
  equal ~msg:"reaches" bool carried
    (Memory.reaches (Memory.reach x.t_memory) (Memory.reach o.t_memory));
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  (match Memory.map_peer x.t_memory r with
  | Ok p ->
      equal ~msg:"maps" bool carried true;
      Option.iter (Memory.free x.t_memory) p
  | Error why ->
      equal ~msg:"maps" bool carried false;
      contains ~sub:"carries requests between them" why);
  Memory.free o.t_memory r

(* A peer's region maps the region its owner gave, at its address. *)
let test_peer_of_peer () =
  with_gpus 3 @@ fun _ _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 and y = List.nth gpus 2 in
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let p = mapped (Memory.map_peer x.t_memory r) in
  let q = mapped (Memory.map_peer y.t_memory p) in
  equal ~msg:"its source" source (Peer r) (Memory.source q);
  equal ~msg:"its address" hex (Memory.address r) (Memory.address q);
  equal ~msg:"the owner's memory through its BAR" ranges
    (merge (snd (Memory.pages p)))
    (merge (snd (Memory.pages q)));
  List.iter2 Memory.free [ y.t_memory; x.t_memory; o.t_memory ] [ q; p; r ]

(* A free the GPU does not confirm keeps the region's memory and addresses, of
   every source, until a free once the function is released, which gives back
   system memory and addresses only. *)
let test_unconfirmed () =
  with_gpus 1 @@ fun root _ gpus ->
  let x = List.hd gpus in
  let lent, _ = lend x page in
  let before = tree_capacity x in
  let host = given (Memory.alloc x.t_memory Host (64 * kib)) in
  let borrowed =
    mapped (Memory.map_host x.t_memory (Window.address lent) page)
  in
  let gpu = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  let later = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  Function.free_dma x.t_fn lent;
  let bytes = held_bytes root and held = tree_capacity x in
  x.t_g.confirms <- false;
  List.iter (Memory.free x.t_memory) [ host; borrowed; gpu ];
  equal ~msg:"system memory kept" int bytes (held_bytes root);
  equal ~msg:"the GPU's memory and addresses kept" (pair int int) held
    (tree_capacity x);
  Function.release x.t_fn;
  equal ~msg:"kept through the release" int bytes (held_bytes root);
  Memory.free x.t_memory later;
  equal ~msg:"given back at a free after the release" int 0 (held_bytes root);
  equal ~msg:"the addresses" int (snd before) (snd (tree_capacity x))

(* Once the function is released, a free of any region writes no table. *)
let test_free_released from =
  with_gpus 2 @@ fun _ _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let lent, _ = lend x page in
  let r, _ = region x o lent from in
  Function.release x.t_fn;
  let touches = x.t_g.touches in
  Memory.free x.t_memory r;
  equal ~msg:"entries, zeroes and flushes" int touches x.t_g.touches;
  (match Memory.source r with
  | Peer origin -> Memory.free o.t_memory origin
  | Allocated | Borrowed _ -> ());
  Function.free_dma x.t_fn lent

(* System memory one GPU allocated and another maps outlives the first GPU's
   release while the second still reaches it: the machine keeps the owner's
   file, listing both, until the second is released too. *)
let test_peer_reach () =
  with_gpus 2 @@ fun root _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 in
  let mem = given (Memory.alloc o.t_memory Host page) in
  let p = mapped (Memory.map_peer x.t_memory mem) in
  let name = require_some (List.nth_opt (host_files root) 0) in
  let file = Filename.concat root ("dev/hugepages/" ^ name) in
  let stored = Tree.stored file in
  at_least ~msg:"the memory's pages" int ~than:page stored;
  Memory.free o.t_memory mem;
  equal ~msg:"pages kept through the owner's free" int stored (Tree.stored file);
  Function.release o.t_fn;
  equal ~msg:"kept through the owner's release" (list string) [ name ]
    (host_files root);
  Memory.free x.t_memory p;
  Function.release x.t_fn;
  equal ~msg:"gone once neither reaches it" (list string) [] (host_files root)

(* A peer whose reach cannot be recorded for after the process's death is
   refused before its page table points at the memory. *)
let test_peer_unrecorded () =
  with_gpus 2 @@ fun root _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 in
  let mem = given (Memory.alloc o.t_memory Host page) in
  let file =
    Filename.concat root
      ("dev/hugepages/" ^ require_some (List.nth_opt (host_files root) 0))
  in
  (* A directory where the peer's marker goes refuses the marker. *)
  let marker = file ^ ".reach-" ^ Function.bus x.t_fn in
  Unix.mkdir marker 0o700;
  contains ~sub:"could not be written"
    (require_error (Memory.map_peer x.t_memory mem));
  Unix.rmdir marker;
  let p = mapped (Memory.map_peer x.t_memory mem) in
  Memory.free x.t_memory p;
  Memory.free o.t_memory mem

(* Requests on a released GPU raise and write nothing: [mine] is the released
   GPU's memory and [theirs] the other GPU's, both allocated before the
   release. *)
let test_released_refused =
  cases "a released GPU refuses requests, writing nothing"
    ~name:(fun (what, _) -> what)
    [
      ("alloc Gpu", fun x _ _ _ -> ignore (Memory.alloc x.t_memory Gpu page));
      ("alloc Host", fun x _ _ _ -> ignore (Memory.alloc x.t_memory Host page));
      ( "map_host",
        fun x _ _ _ -> ignore (Memory.map_host x.t_memory lend_base page) );
      ( "map_peer for it",
        fun x _ _ theirs -> ignore (Memory.map_peer x.t_memory theirs) );
    ]
    (fun (_, request) ->
      with_gpus 2 @@ fun _ _ gpus ->
      let x = List.nth gpus 0 and o = List.nth gpus 1 in
      let mine = given (Memory.alloc x.t_memory Gpu page) in
      let theirs = given (Memory.alloc o.t_memory Gpu page) in
      Function.release x.t_fn;
      let before = tree_capacity x and touches = x.t_g.touches in
      raises_match (Exn.invalid_arg ~substring:"released") (fun () ->
          request x o mine theirs);
      equal ~msg:"entries, zeroes and flushes" int touches x.t_g.touches;
      equal ~msg:"nothing taken" (pair int int) before (tree_capacity x);
      Memory.free o.t_memory theirs)

(* A released GPU's memory, allocated before its release, mapped by the other
   GPU: its system memory stays until its region's free, so it maps as before;
   its GPU memory another open may have reset and given out, so a view of it
   is refused, naming the GPU, taking nothing. *)
let test_released_peer () =
  with_gpus 2 @@ fun _ _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let host = given (Memory.alloc x.t_memory Host page) in
  let gpu = given (Memory.alloc x.t_memory Gpu page) in
  let bar = given (Memory.alloc x.t_memory Bar page) in
  Function.release x.t_fn;
  let before = tree_capacity o in
  List.iter
    (fun (msg, r) ->
      contains ~msg ~sub:(Function.bus x.t_fn)
        (require_error (Memory.map_peer o.t_memory r)))
    [ ("Gpu memory", gpu); ("Bar memory", bar) ];
  equal ~msg:"nothing taken" (pair int int) before (tree_capacity o);
  let p = mapped (Memory.map_peer o.t_memory host) in
  equal ~msg:"the view's address" hex (Memory.address host) (Memory.address p);
  equal ~msg:"the view's pages" (pair target ranges) (Memory.pages host)
    (Memory.pages p);
  Memory.free o.t_memory p;
  List.iter (Memory.free x.t_memory) [ host; gpu; bar ]

(* Placement *)

let sum ranges = List.fold_left (fun n (_, k) -> n + k) 0 ranges

(* The pages the tables map for [r], by virtual address. *)
let entries x r =
  let lo = Memory.address r / 4096 * 4096 in
  let hi = lo + sum (snd (Memory.pages r)) in
  List.filter
    (fun (e : Tables.entry) -> e.va >= lo && e.va < hi)
    (fst (Tables.walk x.t_g x.t_tables))

let entry_ranges es =
  merge
    (List.map
       (fun (e : Tables.entry) -> (e.pa, 1 lsl Tables.shifts.(e.level)))
       es)

(* The physical address the tree's page map gives the space's address [va]. *)
let space_pa m va =
  let page = Machine.page m in
  (first_frame + ((va - space_base) / page)) * page

let bar_file root x =
  Filename.concat root
    (strf "sys/bus/pci/devices/%s/resource0" (Function.bus x.t_fn))

let read_file file off n =
  In_channel.with_open_bin file (fun ic ->
      In_channel.seek ic (Int64.of_int off);
      Option.value ~default:"" (In_channel.really_input_string ic n))

(* What each kind promises of a new region. *)
let test_kind kind =
  with_gpu @@ fun root m x ->
  let r = given (Memory.alloc x.t_memory kind (64 * kib)) in
  let tg, rs = Memory.pages r in
  let size = sum rs and es = entries x r in
  let each f = List.map f es in
  equal ~msg:"allocated" source Allocated (Memory.source r);
  equal ~msg:"the tables map its pages" ranges (merge rs) (entry_ranges es);
  equal ~msg:"its entries in its memory" (list target)
    (each (fun _ -> tg))
    (each (fun (e : Tables.entry) -> e.target));
  match kind with
  | Memory.Gpu ->
      equal ~msg:"in the GPU's memory" target Gpu tg;
      equal ~msg:"no page outside the GPU's memory" ranges []
        (List.filter (fun (pa, n) -> pa + n > gpu_memory) rs);
      is_none ~msg:"no window" (Memory.host r);
      Memory.free x.t_memory r
  | Bar ->
      equal ~msg:"in the GPU's memory" target Gpu tg;
      let pa =
        match merge rs with [ (pa, _) ] -> pa | _ -> fail "one block"
      in
      let w = require_some ~msg:"a window" (Memory.host r) in
      equal ~msg:"as long as the memory" int size (Window.length w);
      Window.write w 0 "the block";
      equal ~msg:"the BAR's bytes at the block" string "the block"
        (read_file (bar_file root x) pa 9);
      Memory.free x.t_memory r
  | Host ->
      equal ~msg:"system memory" target System tg;
      equal ~msg:"uncached" (list bool)
        (each (fun _ -> true))
        (each (fun (e : Tables.entry) -> e.uncached));
      equal ~msg:"snooped" (list bool)
        (each (fun _ -> true))
        (each (fun (e : Tables.entry) -> e.snooped));
      let w = require_some ~msg:"a window" (Memory.host r) in
      equal ~msg:"the process's address is the GPU's" hex (Memory.address r)
        (Window.address w);
      equal ~msg:"at the frames of its addresses" ranges
        [ (space_pa m (Memory.address r), size) ]
        (merge rs);
      Memory.free x.t_memory r

(* Sizes around 4 KiB, a machine's page, 8 MiB and 2 MiB. *)
let sizes =
  Gen.one_of
    [
      Gen.of_list ~pp:pp_hex
        [
          1;
          4095;
          4096;
          4097;
          (8 * mib) - 4097;
          (8 * mib) - 1;
          8 * mib;
          (8 * mib) + 1;
          (10 * mib) - 1;
          (10 * mib) + 1;
        ];
      Gen.with_pp pp_hex (Gen.int_range 1 (20 * mib));
    ]

let kinds = Gen.of_list ~pp:pp_kind [ Memory.Gpu; Bar; Host ]

(* A GPU of 512 MiB on a tree, taken for the run. *)
let large =
  fixture
    ~teardown:(fun (_, x) -> Function.release x.t_fn)
    (fun () ->
      let _, m, buses = tree_machine 1 in
      let space = Space.create ~base:space_base space_length in
      (m, take_gpu ~memory:(512 * mib) ~space m (List.hd buses)))

let test_sizes =
  prop
    "sizes round up to the machine's page in system memory, to 4 KiB in the \
     GPU's, and to 2 MiB from 8 MiB on, where they map with large pages"
    (Gen.pair kinds sizes) (fun (kind, n) ->
      let m, x = large () in
      let r = given (Memory.alloc x.t_memory kind n) in
      let rs = snd (Memory.pages r) in
      let in_gpu = kind <> Host in
      cover "a large GPU block" (in_gpu && n >= 8 * mib);
      cover "a small GPU block" (in_gpu && n < 8 * mib);
      cover "system memory" (not in_gpu);
      let expected =
        if not in_gpu then round_up n (Machine.page m)
        else if n >= 8 * mib then round_up n (2 * mib)
        else round_up n 4096
      in
      equal ~msg:"its size" hex expected (sum rs);
      if in_gpu && n >= 8 * mib then begin
        equal ~msg:"its address on 2 MiB" hex 0 (Memory.address r mod (2 * mib));
        equal ~msg:"no block off 2 MiB" ranges []
          (List.filter
             (fun (pa, len) -> pa mod (2 * mib) <> 0 || len mod (2 * mib) <> 0)
             rs)
      end;
      Memory.free x.t_memory r)

let test_uncached (kind, uncached, expected) =
  with_gpu @@ fun _ _ x ->
  let r = given (Memory.alloc ~uncached x.t_memory kind (64 * kib)) in
  let es = entries x r in
  equal (list bool)
    (List.map (fun _ -> expected) es)
    (List.map (fun (e : Tables.entry) -> e.uncached) es);
  Memory.free x.t_memory r

external c_combines : Window.t -> bool = "rig_pci_test_combines"

(* Combining is the driver's choice: Bar memory's window maps the BAR as the
   driver's own windows on it do, uncached where it has none. *)
let test_bar_follows way =
  with_gpu @@ fun _ _ x ->
  Option.iter
    (fun combine ->
      ignore (require_ok (Function.map ~combine ~length:4096 x.t_fn 0)))
    way;
  let r = given (Memory.alloc x.t_memory Bar (64 * kib)) in
  equal bool
    (Option.value way ~default:false)
    (c_combines (require_some (Memory.host r)));
  Memory.free x.t_memory r

(* A format that zeroes through the BAR, as a driver whose BAR does not reach
   all of its memory does: it records each zero past the BAR in [past]. *)
let zero_within bar past (f : Page_table.format) =
  {
    f with
    zero =
      (fun pa n ->
        if pa + n > bar then past := (pa, n) :: !past;
        f.zero pa n);
  }

(* The fixture's BAR 0 is 256 MiB, half the GPU's memory. *)
let test_small_bar_fills () =
  let bar = 256 * mib and past = ref [] in
  with_gpu ~memory:(512 * mib) ~format:(zero_within bar past) @@ fun _ _ x ->
  let rec fill acc =
    match require_ok (Memory.alloc x.t_memory Bar mib) with
    | Some r -> fill (r :: acc)
    | None -> acc
  in
  let blocks = fill [] in
  not_equal ~msg:"blocks" int 0 (List.length blocks);
  equal ~msg:"no block past the BAR" ranges []
    (List.concat_map
       (fun r ->
         List.filter (fun (pa, n) -> pa + n > bar) (snd (Memory.pages r)))
       blocks);
  equal ~msg:"nothing zeroed past the BAR" ranges [] !past;
  is_some ~msg:"the GPU's memory past the BAR remains"
    (require_ok (Memory.alloc x.t_memory Gpu (64 * mib)))

let no_room ~msg r = is_none ~msg ~pp:pp_region (require_ok r)

let test_out_of_memory () =
  with_gpu @@ fun root _ x ->
  let before = tree_capacity x and bytes = held_bytes root in
  no_room ~msg:"more than the GPU's memory"
    (Memory.alloc x.t_memory Gpu (gpu_memory + 4096));
  no_room ~msg:"a BAR block larger than the memory"
    (Memory.alloc x.t_memory Bar (gpu_memory + 4096));
  no_room ~msg:"system memory larger than the space"
    (Memory.alloc x.t_memory Host (2 * space_length));
  equal ~msg:"no system memory held" int bytes (held_bytes root);
  equal ~msg:"nothing held" (pair int int) before (tree_capacity x)

(* The main pool full, with the tables in it: system memory whose frames do not
   start on 2 MiB maps with 4 KiB pages, whose leaf tables have no room. *)
let test_tables_full () =
  with_gpu ~tables:Main ~memory:(16 * mib) @@ fun root m x ->
  let page = Machine.page m in
  Tree.pagemap root ~page space_base
    (List.init (space_frames / page) (fun i -> first_frame + 1 + i));
  let rec go n =
    if n >= 4096 then
      match require_ok (Memory.alloc x.t_memory Gpu n) with
      | Some _ -> go n
      | None -> go (n / 2)
  in
  go (8 * mib);
  let bytes = held_bytes root in
  no_room ~msg:"system memory" (Memory.alloc x.t_memory Host (4 * mib));
  equal ~msg:"no system memory held" int bytes (held_bytes root)

(* System memory without an address takes addresses of the process's own,
   from a range, then from another once no range has room: eight 2 MiB blocks
   pass a range of 16 MiB, which holds fewer, as its allocator needs a free
   block of twice a request and its alignment. Its physical addresses need
   root, read from the system's page map. *)
let test_own_addresses () =
  needs_linux ();
  if Unix.geteuid () <> 0 then
    skip ~reason:"the physical addresses of the process's memory need root" ();
  let bus = "0000:03:00.0" in
  let root = Tree.make ~privileged:true [ Tree.gpu bus ] in
  Tree.link root "proc/self/pagemap" "/proc/self/pagemap";
  let f = require_ok (Function.take (Machine.at root) bus) in
  let was = Function.set_own_length (16 * mib) in
  Fun.protect ~finally:(fun () ->
      ignore (Function.set_own_length was : int);
      Function.release f)
  @@ fun () ->
  let block () =
    fst (given (Function.alloc_dma ~contiguous:true f (2 * mib)))
  in
  let blocks = List.init 8 (fun _ -> block ()) in
  List.iter (Function.free_dma f) blocks

(* The machine refuses system memory when its memory directory cannot be
   written, and a window on a BAR whose file is shorter than the BAR. *)
let test_refused () =
  if Unix.geteuid () = 0 then
    skip ~reason:"root writes a directory whatever its mode" ();
  with_gpu @@ fun root _ x ->
  let before = tree_capacity x in
  let dir = Filename.concat root "dev/hugepages" in
  Unix.chmod dir 0o500;
  let host = Memory.alloc x.t_memory Host (64 * kib) in
  Unix.chmod dir 0o755;
  ignore (require_error ~msg:"system memory" host);
  equal ~msg:"its addresses given back" (pair int int) before (tree_capacity x);
  Out_channel.with_open_bin (bar_file root x) (fun oc ->
      output_string oc (String.make 4096 '\000'));
  contains ~msg:"a BAR window, naming the file" ~sub:"resource0"
    (require_error (Memory.alloc x.t_memory Bar (64 * kib)));
  equal ~msg:"its memory and addresses given back" (pair int int) before
    (tree_capacity x)

(* Memory mapped already, by an allocation or a borrow, maps again at other
   addresses, pinned once per map. *)
let test_map_host_mapped () =
  with_gpu @@ fun _ _ x ->
  let host = given (Memory.alloc x.t_memory Host page) in
  let at = Memory.address host in
  let again = mapped (Memory.map_host x.t_memory at page) in
  let lent, _ = lend x (3 * page) in
  let a = Window.address lent in
  let first = mapped (Memory.map_host x.t_memory a (2 * page)) in
  let second = mapped (Memory.map_host x.t_memory (a + page) (2 * page)) in
  equal ~msg:"four addresses" int 4
    (List.length
       (List.sort_uniq compare
          (List.map Memory.address [ host; again; first; second ])));
  List.iter (Memory.free x.t_memory) [ again; first; second; host ];
  List.iter
    (unpinned ~msg:"unpinned" x)
    [ (at, page); (a, 2 * page); (a + page, 2 * page) ];
  Function.free_dma x.t_fn lent

(* Borrowed addresses belong to the GPU's machine. A window on them keeps that
   machine's transport, including for a borrow starting within a page. *)
let test_map_host_transport () =
  let base = 0x100000 in
  let tr = Window.unsafe_transport (far base page) in
  let w = Window.through tr base page in
  let unused () = fail "unexpected machine operation" in
  let fn : Machine.fn =
    {
      addressing = Physical;
      inherited = false;
      config = (fun _ _ -> unused ());
      set_config = (fun _ _ _ -> unused ());
      bar = (fun i -> if i = 0 then Some (0, gpu_memory) else None);
      map = (fun ~combine:_ _ _ _ -> unused ());
      unmap = (fun _ -> unused ());
      interrupt = (fun _ -> unused ());
      reset = unused;
      forget = unused;
      alloc_dma = (fun ~contiguous:_ ~va:_ ~below:_ _ -> unused ());
      free_dma = (fun _ -> unused ());
      pin = (fun ~below:_ a n -> Ok (Some [ (a, n) ]));
      unpin = (fun _ _ -> ());
      release = (fun () -> true);
    }
  in
  let m =
    Machine.make ~name:"far"
      {
        transport = tr;
        page;
        functions = (fun () -> Ok []);
        take = (fun _ -> Ok fn);
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  let space = Space.create ~base:space_base space_length in
  let x = take_gpu ~space m "0000:03:00.0" in
  Fun.protect ~finally:(fun () -> Function.release x.t_fn) @@ fun () ->
  let r = mapped (Memory.map_host x.t_memory (base + 4) 8) in
  Fun.protect ~finally:(fun () -> Memory.free x.t_memory r) @@ fun () ->
  let h = require_some (Memory.host r) in
  equal ~msg:"the window keeps its transport" bool false (Window.mapped h);
  equal ~msg:"borrowed range" (pair hex int)
    (base + 4, 8)
    (Window.address h, Window.length h);
  Window.set32 w 4 0x1234;
  equal ~msg:"reads reach the machine" int 0x1234 (Window.get32 h 0);
  Window.set32 h 4 0x5678;
  equal ~msg:"writes reach the machine" int 0x5678 (Window.get32 w 8)

(* The GPU of a machine a transport reaches, which has no system memory free and
   pins any page, its space reserved there if [reserved]. *)
let far_gpu ?(reserved = true) () =
  let unused () = fail "unexpected machine operation" in
  let fn : Machine.fn =
    {
      addressing = Physical;
      inherited = false;
      config = (fun _ _ -> unused ());
      set_config = (fun _ _ _ -> unused ());
      bar = (fun i -> if i = 0 then Some (0, gpu_memory) else None);
      map = (fun ~combine:_ _ _ _ -> unused ());
      unmap = (fun _ -> unused ());
      interrupt = (fun _ -> unused ());
      reset = unused;
      forget = unused;
      alloc_dma = (fun ~contiguous:_ ~va:_ ~below:_ _ -> Ok None);
      free_dma = (fun _ -> unused ());
      pin = (fun ~below:_ a n -> Ok (Some [ (a, n) ]));
      unpin = (fun _ _ -> ());
      release = (fun () -> true);
    }
  in
  let m =
    Machine.make ~name:"far"
      {
        transport = Window.unsafe_transport (far 0 page);
        page;
        functions = (fun () -> Ok []);
        take = (fun _ -> Ok fn);
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  if reserved then require_ok (Machine.reserve m ~base:space_base space_length);
  let space = Space.create ~base:space_base space_length in
  take_gpu ~space m "0000:03:00.0"

let with_far_gpu ?reserved f =
  let x = far_gpu ?reserved () in
  Fun.protect ~finally:(fun () -> Function.release x.t_fn) (fun () -> f x)

(* Sizes whose rounding up passes max_int, which no memory holds. *)
let test_too_large kind =
  with_far_gpu @@ fun x ->
  List.iter
    (fun n -> no_room ~msg:(strf "%d bytes" n) (Memory.alloc x.t_memory kind n))
    [ max_int; max_int - page + 2; max_int - (2 * mib) + 2 ]

let test_map_host_past_end () =
  with_far_gpu @@ fun x ->
  List.iter
    (fun (a, n) ->
      ignore
        (require_error
           ~pp:(Format.pp_print_option pp_region)
           ~msg:(strf "%d bytes at 0x%x" n a)
           (Memory.map_host x.t_memory a n)))
    [ (page, max_int - page + 1); (max_int - 4, 8); (page, max_int) ]

let test_host_unreserved () =
  with_far_gpu ~reserved:false @@ fun x ->
  raises_match (Exn.invalid_arg ~substring:"Machine.reserve") (fun () ->
      Memory.alloc x.t_memory Host page)

(* The process's own pages go back to the system when it dies: a GPU taken
   physically is refused them, with the pin's reason. *)
let test_map_host_unpinnable () =
  with_gpu @@ fun _ m x ->
  let page = Machine.page m in
  let a = round_up (memory (2 * page)) page in
  contains ~sub:"alloc_dma" (require_error (Memory.map_host x.t_memory a page))

let test_not_mine () =
  with_gpus 2 @@ fun _ _ gpus ->
  let x = List.nth gpus 0 and y = List.nth gpus 1 in
  let mem = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  let theirs = given (Memory.alloc y.t_memory Gpu (64 * kib)) in
  let refused msg f = raises_match ~msg (Exn.invalid_arg ~substring:"") f in
  refused "free of another GPU's region" (fun () -> Memory.free y.t_memory mem);
  let p = mapped (Memory.map_peer x.t_memory theirs) in
  Memory.free x.t_memory p;
  refused "free of a freed region" (fun () -> Memory.free x.t_memory p);
  Memory.free y.t_memory theirs;
  refused "map_peer of a freed region" (fun () ->
      Memory.map_peer x.t_memory theirs);
  Memory.free x.t_memory mem

(* A peer's memory maps at its owner's address, which means that memory only in
   the space both GPUs share. *)
let test_two_spaces () =
  let _, m, buses = tree_machine 2 in
  let take bus =
    take_gpu ~space:(Space.create ~base:space_base space_length) m bus
  in
  let x = take (List.nth buses 0) and o = take (List.nth buses 1) in
  Fun.protect ~finally:(fun () ->
      Function.release x.t_fn;
      Function.release o.t_fn)
  @@ fun () ->
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let why = require_error (Memory.map_peer x.t_memory r) in
  contains ~msg:"names the mapper" ~sub:(Function.bus x.t_fn) why;
  contains ~msg:"names the owner" ~sub:(Function.bus o.t_fn) why;
  Memory.free o.t_memory r

(* BAR 1 is the upper half of the fixture's 64-bit BAR 0. *)
let test_no_bar () =
  with_gpu @@ fun _ _ x ->
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.create x.t_fn x.t_tables ~bar:1)

let bindings g =
  List.sort compare
    (Hashtbl.fold (fun a e l -> (a, e) :: l) g.Tables.entries [])

(* The GPU reset and opened again: a new instance's tables in the same GPU
   memory. The old instance frees all it holds, writing none of it, and gives
   back its system memory. *)
let test_reopened () =
  with_gpus 2 @@ fun _ _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let shared = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let held =
    List.map
      (fun k -> given (Memory.alloc x.t_memory k (3 * mib)))
      [ Memory.Gpu; Bar; Host ]
  in
  let host = require_some (Memory.host (List.nth held 2)) in
  let p = mapped (Memory.map_peer x.t_memory shared) in
  Function.release x.t_fn;
  let reopened =
    Page_table.create ~base:tables_base (Tables.format x.t_g)
      (Page_table.space x.t_tables)
      ~memory:gpu_memory ~boot:mib ~tables:Pool ~pages:large_pages
  in
  Page_table.booted reopened;
  let fresh = require_some (Page_table.alloc reopened (3 * mib)) in
  let before = bindings x.t_g and touches = x.t_g.touches in
  List.iter (Memory.free x.t_memory) (p :: held);
  equal ~msg:"no entry written" int touches x.t_g.touches;
  equal ~msg:"the new instance's entries unchanged"
    (list (pair hex int64))
    before (bindings x.t_g);
  raises_match ~msg:"its system memory given back"
    (Exn.invalid_arg ~substring:"") (fun () -> Function.free_dma x.t_fn host);
  equal ~msg:"the new instance's free" bool true
    (Page_table.free reopened fresh);
  Memory.free o.t_memory shared

let placement =
  group ~timeout:patience "placement and refusals"
    [
      cases "each kind is placed by its rule"
        ~name:(Format.asprintf "%a" pp_kind)
        [ Memory.Gpu; Bar; Host ] test_kind;
      test_sizes;
      cases
        "the GPU bypasses its caches when asked, and always for system memory"
        ~name:(fun (k, u, _) -> Format.asprintf "%a, uncached %b" pp_kind k u)
        [
          (Memory.Gpu, false, false);
          (Gpu, true, true);
          (Bar, false, false);
          (Bar, true, true);
          (Host, false, true);
          (Host, true, true);
        ]
        test_uncached;
      cases "Bar memory's window maps as the driver's windows on its BAR"
        ~name:(function
          | None -> "no window"
          | Some true -> "combining"
          | Some false -> "uncached")
        [ None; Some true; Some false ]
        test_bar_follows;
      test "a small BAR's blocks are placed inside it before they are zeroed"
        test_small_bar_fills;
      test "no room in the GPU's memory or the space is Ok None"
        test_out_of_memory;
      test "no room for a table is Ok None" test_tables_full;
      test
        "system memory or a BAR window the machine refuses is an Error, having \
         given back what it took"
        test_refused;
      test "system memory without an address passes a range of addresses"
        test_own_addresses;
      test "memory the GPU maps already maps again elsewhere"
        test_map_host_mapped;
      test "map_host preserves the machine's transport" test_map_host_transport;
      cases "a size no memory holds is Ok None"
        ~name:(Format.asprintf "%a" pp_kind)
        [ Memory.Gpu; Bar; Host ] test_too_large;
      test "map_host of bytes past the last address is an Error"
        test_map_host_past_end;
      test "system memory at addresses no reservation holds raises"
        test_host_unreserved;
      test "map_host refuses the process's own pages with the pin's reason"
        test_map_host_unpinnable;
      test "a region of another GPU, or freed, is refused" test_not_mine;
      test "map_peer refuses GPUs of two spaces, naming both" test_two_spaces;
      test "create refuses a BAR the function lacks" test_no_bar;
      test "a reopened GPU is not written by its old instance" test_reopened;
    ]

(* Requests *)

let test_no_bytes (_, n, request) =
  with_gpu @@ fun root _ x ->
  let lent, _ = lend x page in
  let before = tree_capacity x and bytes = held_bytes root in
  raises_match (Exn.invalid_arg ~substring:"Memory.") (fun () ->
      request x (Window.address lent) n);
  equal ~msg:"nothing taken" (pair int int) before (tree_capacity x);
  equal ~msg:"no system memory taken" int bytes (held_bytes root);
  Function.free_dma x.t_fn lent

let no_bytes =
  List.concat_map
    (fun n ->
      [
        ("alloc Gpu", n, fun x _ n -> ignore (Memory.alloc x.t_memory Gpu n));
        ("alloc Host", n, fun x _ n -> ignore (Memory.alloc x.t_memory Host n));
        ("map_host", n, fun x a n -> ignore (Memory.map_host x.t_memory a n));
      ])
    [ 0; -1 ]

(* Entries of 32 bits of address hold none of the tree's frames, which lie from
   8 GiB on, nor the peer's BAR at 0x7c_0000_0000. The GPU's function reaches
   no system memory past them (Memory.create), so a request for system memory
   finds none, and a peer's BAR is refused, naming the bits. *)
let pa_bits_requests =
  [
    ( "alloc Host",
      `None,
      fun x _ _ _ -> Memory.alloc x.t_memory Host (64 * kib) );
    ( "map_host",
      `None,
      fun x lent _ _ -> Memory.map_host x.t_memory (Window.address lent) page );
    ( "map_peer of GPU memory",
      `Refused,
      fun x _ gpu _ -> Memory.map_peer x.t_memory gpu );
    ( "map_peer of system memory",
      `None,
      fun x _ _ host -> Memory.map_peer x.t_memory host );
  ]

let test_pa_bits (_, answer, request) =
  let root, m, buses = tree_machine 2 in
  let space = Space.create ~base:space_base space_length in
  let x = take_gpu ~pa_bits:32 ~space m (List.nth buses 0) in
  let o = take_gpu ~space m (List.nth buses 1) in
  Fun.protect ~finally:(fun () ->
      Function.release x.t_fn;
      Function.release o.t_fn)
  @@ fun () ->
  let lent, _ = lend o page in
  let gpu = given (Memory.alloc o.t_memory Gpu page) in
  let host = given (Memory.alloc o.t_memory Host page) in
  let before = tree_capacity x and bytes = held_bytes root in
  (match answer with
  | `Refused ->
      contains ~msg:"names the bits" ~sub:"32 bits"
        (require_error (request x lent gpu host))
  | `None -> is_none ~msg:"no memory" (require_ok (request x lent gpu host)));
  equal ~msg:"nothing taken" (pair int int) before (tree_capacity x);
  equal ~msg:"no system memory taken" int bytes (held_bytes root);
  unpinned ~msg:"no pin" x (Window.address lent, page);
  Memory.free o.t_memory gpu;
  Memory.free o.t_memory host;
  Function.free_dma o.t_fn lent

(* Peers *)

(* A peer's window on its region is the origin's. *)
let test_peer_host kind =
  with_gpus 2 @@ fun _ _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 in
  let r = given (Memory.alloc o.t_memory kind (64 * kib)) in
  let p = mapped (Memory.map_peer x.t_memory r) in
  let window w = (Window.address w, Window.length w) in
  equal
    (option (pair hex int))
    (Option.map window (Memory.host r))
    (Option.map window (Memory.host p));
  Memory.free x.t_memory p;
  Memory.free o.t_memory r

(* [drain_tables x] makes tables far from any address the space hands out until
   the tables' pool has no room for another. *)
let drain_tables x =
  let rec go k =
    let va = tables_base + (1 lsl 44) + (k lsl 30) in
    match Page_table.tables x.t_tables ~va 4096 with
    | Some _ -> go (k + 1)
    | None -> ()
  in
  go 0

let test_no_room () =
  with_gpus 2 @@ fun root _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let lent, _ = lend o page in
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let bytes = held_bytes root in
  drain_tables x;
  is_none ~msg:"map_host" ~pp:pp_region
    (require_ok (Memory.map_host x.t_memory (Window.address lent) page));
  unpinned ~msg:"map_host's pin given back" x (Window.address lent, page);
  is_none ~msg:"map_peer" ~pp:pp_region
    (require_ok (Memory.map_peer x.t_memory r));
  equal ~msg:"no system memory held" int bytes (held_bytes root);
  Memory.free o.t_memory r;
  Function.free_dma o.t_fn lent

(* Memory a GPU maps already, a peer's it mapped or its own, given or through
   a peer's region, maps again as a region over the same mapping: same
   address and pages, no pin and no table written, and the mapping stays
   until the last region over it is freed. *)
type again = Peers | Own | Own_through_peer

let test_mapped_again (kind, again) =
  with_gpus 2 @@ fun root _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 in
  let r = given (Memory.alloc o.t_memory kind (64 * kib)) in
  let p = mapped (Memory.map_peer x.t_memory r) in
  let own = again <> Peers in
  let y, first = if own then (o, r) else (x, p) in
  let bytes = held_bytes root and touches = y.t_g.touches in
  let via = match again with Peers | Own -> r | Own_through_peer -> p in
  let q = mapped (Memory.map_peer y.t_memory via) in
  equal ~msg:"its source" source (Peer r) (Memory.source q);
  equal ~msg:"its address" hex (Memory.address first) (Memory.address q);
  equal ~msg:"its pages" ranges
    (merge (snd (Memory.pages first)))
    (merge (snd (Memory.pages q)));
  equal ~msg:"system memory held" int bytes (held_bytes root);
  equal ~msg:"entries, zeroes and flushes" int touches y.t_g.touches;
  Memory.free y.t_memory (if own then q else first);
  equal ~msg:"the first free writes no table" int touches y.t_g.touches;
  if own then Memory.free x.t_memory p;
  Memory.free y.t_memory (if own then first else q);
  at_least ~msg:"the last free unmaps" int ~than:(touches + 1) y.t_g.touches;
  if not own then Memory.free o.t_memory r;
  equal ~msg:"system memory held at the end" int 0 (held_bytes root)

let mapped_again =
  List.concat_map
    (fun again -> [ (Memory.Gpu, again); (Memory.Host, again) ])
    [ Peers; Own; Own_through_peer ]

(* A peer's region that was freed is refused, though the region it maps
   lives. *)
let test_freed_peer () =
  with_gpus 3 @@ fun _ _ gpus ->
  let o = List.nth gpus 0 and x = List.nth gpus 1 and y = List.nth gpus 2 in
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let p = mapped (Memory.map_peer x.t_memory r) in
  Memory.free x.t_memory p;
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_peer y.t_memory p);
  Memory.free o.t_memory r

(* After the release

   Two GPUs of a tree sharing a space, under one root port as root reads it,
   so that each reaches the other's memory (Memory.reaches), against a
   model of what each region holds: the GPU's memory and addresses, or system
   memory. A free writes its GPU's tables before the release and none after;
   what the model says nothing holds, the tree holds no more. *)

type m_region = {
  owner : int;
  system : bool;  (** It holds system memory of the space's blocks. *)
  gpu : bool;  (** Its memory is a GPU's own. *)
  owning : bool;  (** It holds addresses or the GPU's memory. *)
  origin : m_region option;  (** For a peer's region, the region it maps. *)
  label : string;
  gpus_released : bool array;  (** Its pair's GPUs released. *)
  mutable peers : int;  (** The live peers' regions over it. *)
  mapping : m_region list ref;
      (** The live regions over its mapping, which the last free unmaps. *)
  mutable freed : bool;
}

type m_pair = { released : bool array; mutable regions : m_region list }

type s_pair = {
  root : string;
  slot : int;
  gpus : on_tree array;
  lent : Window.t;
  initial : (int * int) array;
  bytes : int;  (** System memory held at the start: the lent memory. *)
  mutable live : (int * Memory.region) list;
}

let live_regions m = List.filter (fun r -> not r.freed) m.regions

(* Each pair's addresses are a slot of its own, so that the pairs of one program
   hold no memory in each other's blocks: a space of 14 MiB, then 2 MiB of
   memory the GPUs borrow. A slot goes back when its pair is released. Each
   capacity check drains the space a page at a time. *)
let slot_size = 16 * mib
let slots = 20
let slots_base = lend_base + slot_size
let free_slots = Atomic.make (List.init slots Fun.id)

let rec take_slot () =
  match Atomic.get free_slots with
  | [] -> failf "more than %d pairs" slots
  | k :: rest as all ->
      if Atomic.compare_and_set free_slots all rest then k else take_slot ()

let rec give_slot k =
  let all = Atomic.get free_slots in
  if not (Atomic.compare_and_set free_slots all (k :: all)) then give_slot k

let create_pair () =
  needs_linux ();
  let k = take_slot () in
  let base = slots_base + (k * slot_size) in
  let borrowed = base + slot_size - (2 * mib) in
  let buses = [ "0000:03:00.0"; "0000:04:00.0" ] in
  let root = Tree.make ~privileged:true (List.map Tree.gpu buses) in
  let m = Machine.at root in
  (match Machine.reserve m ~base:slots_base (slots * slot_size) with
  | Ok () -> ()
  | Error why -> skip ~reason:why ());
  let page = Machine.page m in
  Tree.pagemap root ~page base
    (List.init (slot_size / page) (fun i -> first_frame + i));
  let space = Space.create ~base (borrowed - base) in
  (* Each capacity check drains the GPU's pool 4 KiB at a time: 8 MiB hold a
     program's regions. *)
  let gpus =
    Array.of_list (List.map (take_gpu ~memory:(8 * mib) ~space m) buses)
  in
  let lent =
    fst (given (Function.alloc_dma ~va:borrowed gpus.(0).t_fn (2 * page)))
  in
  {
    root;
    slot = k;
    gpus;
    lent;
    initial = Array.map tree_capacity gpus;
    bytes = held_bytes root;
    live = [];
  }

let release_pair s =
  List.iter (fun (i, r) -> Memory.free s.gpus.(i).t_memory r) s.live;
  s.live <- [];
  Array.iter (fun x -> Function.release x.t_fn) s.gpus;
  Function.free_dma s.gpus.(0).t_fn s.lent;
  equal ~msg:"system memory held once all is given back" int 0
    (held_bytes s.root);
  give_slot s.slot

let pairs = abstract "t" ~release:release_pair
let regions = abstract "r"

let made s i r =
  s.live <- (i, r) :: s.live;
  (s, i, r)

let add m r =
  m.regions <- r :: m.regions;
  r

let live_gpu m i what = if m.released.(i) then invalid_arg (what ^ ": released")

let new_region m ?origin ?(mapping = ref []) ~system ~gpu ~owning ~label i =
  let r =
    {
      owner = i;
      system;
      gpu;
      owning;
      origin;
      label;
      gpus_released = m.released;
      peers = 0;
      mapping;
      freed = false;
    }
  in
  mapping := r :: !mapping;
  r

let alloc_ref m i kind =
  live_gpu m i "alloc";
  let label = Format.asprintf "%a" pp_kind kind in
  add m
    (new_region m ~system:(kind = Host) ~gpu:(kind <> Host) ~owning:true ~label
       i)

let alloc_sys s i kind =
  made s i (given (Memory.alloc s.gpus.(i).t_memory kind (64 * kib)))

let map_host_ref m i _ =
  live_gpu m i "map_host";
  add m
    (new_region m ~system:false ~gpu:false ~owning:true ~label:"Borrowed" i)

let map_host_sys s i slot =
  let a = Window.address s.lent + (slot * page) in
  made s i (mapped (Memory.map_host s.gpus.(i).t_memory a page))

let origin r = Option.value r.origin ~default:r

(* Withheld, for a bug with a test of its own: a freed peer's region whose
   origin lives. *)
let peer_pre _ _ r =
  let o = origin r in
  not (r.freed && Option.is_some r.origin && not o.freed)

(* A region of another pair's GPU is of another machine. *)
exception Other_machine

(* A view of a released GPU's own memory is refused. *)
exception Released_memory

let map_peer_ref m i r =
  let o = origin r in
  live_gpu m i "map_peer";
  if r.freed || o.freed then invalid_arg "map_peer: freed";
  let same = o.gpus_released == m.released in
  cover "a region of another machine" (not same);
  if not same then raise Other_machine;
  if o.gpus_released.(o.owner) && o.gpu then raise Released_memory;
  (* The GPU maps the memory already: its own, or a peer's it mapped. *)
  let over p = p.owner = i && origin p == o in
  let mapping =
    Option.map (fun p -> p.mapping) (List.find_opt over (live_regions m))
  in
  cover "memory the GPU maps already" (Option.is_some mapping);
  o.peers <- o.peers + 1;
  add m
    (new_region m ~origin:o ?mapping ~system:o.system ~gpu:o.gpu ~owning:false
       ~label:("Peer of " ^ o.label) i)

let map_peer_sys s i (_, _, r) =
  match Memory.map_peer s.gpus.(i).t_memory r with
  | Ok (Some p) -> made s i p
  | Ok None -> fail "no room"
  | Error why when String.starts_with ~prefix:"the GPUs are on different" why ->
      raise Other_machine
  | Error why when String.ends_with ~suffix:"given its memory out" why ->
      raise Released_memory
  | Error why -> fail why

let free_ref r =
  if r.freed then invalid_arg "free: freed";
  r.freed <- true;
  Option.iter (fun o -> o.peers <- o.peers - 1) r.origin;
  r.mapping := List.filter (fun p -> p != r) !(r.mapping);
  let released = r.gpus_released.(r.owner) in
  cover "a free after the release" released;
  cover "a free before the release" (not released);
  not released && !(r.mapping) = []

let free_sys (s, i, r) =
  let x = s.gpus.(i) in
  let touches = x.t_g.touches in
  Memory.free x.t_memory r;
  s.live <- List.filter (fun (_, r') -> r' != r) s.live;
  x.t_g.touches <> touches

(* A free after the release gives back no GPU memory to the tables. *)
let as_at_start (memory, addresses) x =
  let memory', addresses' = tree_capacity x in
  addresses = addresses' && (Function.released x.t_fn || memory = memory')

let gpu_index = Gen.of_list ~pp:Format.pp_print_int [ 0; 1 ]

let release_commands =
  [
    command "create"
      (Gen.unit @-> makes pairs)
      (fun () -> { released = [| false; false |]; regions = [] })
      create_pair;
    command "alloc"
      (pairs ^-> gpu_index
      @-> Gen.of_list ~pp:pp_kind [ Memory.Gpu; Host ]
      @-> makes regions)
      alloc_ref alloc_sys;
    command "map_host"
      (pairs ^-> gpu_index
      @-> Gen.of_list ~pp:Format.pp_print_int [ 0; 1 ]
      @-> makes regions)
      map_host_ref map_host_sys;
    command "map_peer" ~pre:peer_pre
      (pairs ^-> gpu_index @-> regions ^-> makes regions)
      map_peer_ref map_peer_sys;
    (* The peers' frees come first: a region's free leaves a peer's mapping on
       addresses the space gives out again. *)
    command "free (writes a table)"
      ~pre:(fun r -> r.peers = 0)
      (regions ^-> returns bool)
      free_ref free_sys;
    command "release"
      (pairs ^-> gpu_index @-> returns unit)
      (fun m i -> m.released.(i) <- true)
      (fun s i -> Function.release s.gpus.(i).t_fn);
    command "capacity as at the start"
      ~pre:(fun m -> not (List.exists (fun r -> r.owning) (live_regions m)))
      (pairs ^-> returns bool)
      (fun _ -> true)
      (fun s -> Array.for_all2 as_at_start s.initial s.gpus);
    command "system memory as at the start"
      ~pre:(fun m -> not (List.exists (fun r -> r.system) (live_regions m)))
      (pairs ^-> returns bool)
      (fun _ -> true)
      (fun s -> held_bytes s.root = s.bytes);
  ]

let test_after_release =
  if on_linux then
    stateful "after its release, a GPU's memory is given back without a write"
      ~count:100 ~steps:20 release_commands
  else
    test "after its release, a GPU's memory is given back without a write"
      (fun () -> skip ~reason:"flock on a function's file needs Linux" ())

let on_files =
  group ~timeout:patience "on a machine's files"
    [
      cases "a free gives back what its region took"
        ~name:(Format.asprintf "%a" pp_from)
        froms test_round_trip;
      test
        "map_host of bytes off a page maps the pages that hold them, its \
         address byte a's and its window the bytes'"
        test_unaligned;
      cases "reaches is whether map_peer maps the GPU's memory" ~name:fst
        pairings test_reaches;
      cases "reaches holds where the bus carries requests between the GPUs"
        ~name:(fun (n, _, _, _) -> n)
        routes test_route;
      test "map_peer of a peer's region maps the region its owner gave"
        test_peer_of_peer;
      test
        "a free the GPU does not confirm keeps the region's memory and \
         addresses until a free after the release"
        test_unconfirmed;
      test "system memory a peer maps stays until neither GPU reaches it"
        test_peer_reach;
      test "a peer whose reach cannot be recorded is refused"
        test_peer_unrecorded;
      test_released_refused;
      test
        "a released GPU's system memory maps, and its GPU memory is refused, \
         naming it"
        test_released_peer;
      cases "after the release, a free writes no table"
        ~name:(Format.asprintf "%a" pp_from)
        froms test_free_released;
      cases "no bytes, or fewer, are refused"
        ~name:(fun (what, n, _) -> strf "%s %d" what n)
        no_bytes test_no_bytes;
      cases
        "system memory past the entries' address bits is None, and a peer's \
         BAR past them is refused, naming them"
        ~name:(fun (n, _, _) -> n)
        pa_bits_requests test_pa_bits;
      cases "a peer's window is its origin's"
        ~name:(Format.asprintf "%a" pp_kind)
        [ Memory.Gpu; Bar; Host ] test_peer_host;
      test "map_host and map_peer without room for a table are Ok None"
        test_no_room;
      test "map_peer refuses a peer's region that was freed" test_freed_peer;
      cases
        "a map_peer of memory the GPU maps already shares its mapping until \
         the last free"
        ~name:(fun (k, again) ->
          Format.asprintf "%a%s" pp_kind k
            (match again with
            | Peers -> ""
            | Own -> ", its own"
            | Own_through_peer -> ", its own through a peer's region"))
        mapped_again test_mapped_again;
      test_after_release;
    ]

(* Memory given to another function

   A network adapter at [nic_bus], under the GPUs' root port unless a case
   places it elsewhere. *)

let nic_bus = "0000:10:00.0"

let nic ?group ?bridges () =
  {
    (Tree.gpu ?group ?bridges nic_bus) with
    vendor = 0x14e4;
    device = 0x1760;
    class_ = 0x020000;
  }

(* The GPU's BAR 0, as [Tree.gpu] places it. *)
let bar_base = 0x7c_0000_0000

(* Memory [Memory.bus] gave stays held until the process exits, its system
   memory included, which the process holds by address: each GPU here takes
   its space at addresses of its own, so no later test's system memory lands
   on the addresses of memory an earlier one gave. *)
let given_spaces = ref 0

let with_nic ?(nic = nic ()) ?memory f =
  incr given_spaces;
  let base = (1 lsl 42) + (!given_spaces * space_length) in
  let root, m, buses = tree_machine ~others:[ nic ] ~space:base 1 in
  let space = Space.create ~base space_length in
  let x = take_gpu ?memory ~space m (List.hd buses) in
  Fun.protect
    ~finally:(fun () -> Function.release x.t_fn)
    (fun () -> f root m (List.hd buses) x)

let aligned_runs runs =
  List.for_all (fun (a, n) -> a mod 4096 = 0 && n mod 4096 = 0) runs

(* GPU memory lies at the BAR's bus address plus its physical address, system
   memory at its physical addresses, in the region's page order. *)
let test_bus_runs kind =
  with_nic @@ fun _ _ _ x ->
  let r = given (Memory.alloc x.t_memory kind (64 * kib)) in
  let target, pages = Memory.pages r in
  let expected =
    match target with
    | Page_table.Gpu -> List.map (fun (pa, n) -> (pa + bar_base, n)) pages
    | System | Peer _ -> pages
  in
  let runs = require_ok (Memory.bus x.t_memory r ~client:nic_bus) in
  equal ~msg:"its runs" ranges expected runs;
  satisfies ~msg:"whole pages" ~claim:"multiples of 4 KiB" ranges aligned_runs
    runs;
  equal ~msg:"asked again" (result ranges string) (Ok runs)
    (Memory.bus x.t_memory r ~client:nic_bus)

(* Once given, the memory never returns: a free gives back nothing, no later
   allocation takes its pages, and the GPU's release turns its bus mastering
   off and keeps it taken until the process exits. *)
let command_of root bus =
  let c =
    In_channel.with_open_bin
      (Filename.concat root (strf "sys/bus/pci/devices/%s/config" bus))
      In_channel.input_all
  in
  String.get_uint16_le c 4

let overlaps runs (a, n) =
  List.exists (fun (b, k) -> a < b + k && b < a + n) runs

let test_bus_reserved () =
  with_nic @@ fun root m bus x ->
  let r = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  let given_pages = snd (Memory.pages r) in
  ignore (require_ok (Memory.bus x.t_memory r ~client:nic_bus));
  Memory.free x.t_memory r;
  raises_match ~msg:"freed once" (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.free x.t_memory r);
  let rec fill acc =
    match Memory.alloc x.t_memory Gpu (64 * kib) with
    | Ok (Some r) -> fill (r :: acc)
    | Ok None | Error _ -> acc
  in
  let later = fill [] in
  satisfies ~msg:"later allocations" ~claim:"none over the given pages"
    (list ranges)
    (List.for_all (fun pages -> not (List.exists (overlaps given_pages) pages)))
    (List.map (fun r -> snd (Memory.pages r)) later);
  List.iter (Memory.free x.t_memory) later;
  Function.set_bus_master x.t_fn true;
  Function.release x.t_fn;
  equal ~msg:"bus mastering after the release" int 0
    (command_of root bus land 0x4);
  match Function.take m bus with
  | Ok f ->
      Function.release f;
      fail "a take of a GPU whose memory was given succeeded"
  | Error why -> contains ~sub:"given to another PCI function" why

(* Each refusal names its cause and reserves nothing: the region frees as
   any. *)
let refusals =
  [
    ( "a client that is no bus address",
      (fun f -> with_nic f),
      Memory.Gpu,
      "10:00.0",
      "10:00.0" );
    ( "a client that names no function",
      (fun f -> with_nic f),
      Gpu,
      "0000:11:00.0",
      "0000:11:00.0" );
    ( "a client behind an IOMMU that translates",
      (fun f -> with_nic ~nic:(nic ~group:"7" ()) f),
      Host,
      nic_bus,
      "behind an IOMMU" );
    ( "a BAR smaller than the GPU's memory",
      (fun f -> with_nic ~memory:(512 * mib) f),
      Gpu,
      nic_bus,
      "Resizable BAR" );
    ( "no route the bus carries",
      (fun f -> with_nic ~nic:(nic ~bridges:[ "0000:00:02.0" ] ()) f),
      Gpu,
      nic_bus,
      "does not carry" );
  ]

let test_bus_refused (_, with_, kind, client, sub) =
  with_ @@ fun root _ _ x ->
  let before = tree_capacity x and bytes = held_bytes root in
  let r = given (Memory.alloc x.t_memory kind (64 * kib)) in
  (match Memory.bus x.t_memory r ~client with
  | Ok _ -> fail "given"
  | Error why -> contains ~sub why);
  Memory.free x.t_memory r;
  equal ~msg:"the GPU's memory and addresses" (pair int int) before
    (tree_capacity x);
  equal ~msg:"the system memory held" int bytes (held_bytes root)

(* What a small BAR or a missing route refuses, system memory still gives. *)
let test_bus_host_small () =
  with_nic ~memory:(512 * mib) ~nic:(nic ~bridges:[ "0000:00:02.0" ] ())
  @@ fun _ _ _ x ->
  let r = given (Memory.alloc x.t_memory Host (64 * kib)) in
  equal ~msg:"its physical pages" (result ranges string)
    (Ok (snd (Memory.pages r)))
    (Memory.bus x.t_memory r ~client:nic_bus)

let test_bus_far () =
  with_far_gpu @@ fun x ->
  let r = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  (match Memory.bus x.t_memory r ~client:nic_bus with
  | Ok _ -> fail "given"
  | Error why -> contains ~sub:"transport" why);
  Memory.free x.t_memory r

let test_bus_misuse () =
  with_gpus 2 @@ fun _ _ gpus ->
  let x = List.nth gpus 0 and o = List.nth gpus 1 in
  let lent, _ = lend x page in
  let borrowed =
    mapped (Memory.map_host x.t_memory (Window.address lent) page)
  in
  let r = given (Memory.alloc o.t_memory Gpu (64 * kib)) in
  let peer = mapped (Memory.map_peer x.t_memory r) in
  let freed = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
  Memory.free x.t_memory freed;
  List.iter
    (fun (msg, r) ->
      raises_match ~msg (Exn.invalid_arg ~substring:"") (fun () ->
          Memory.bus x.t_memory r ~client:nic_bus))
    [
      ("borrowed", borrowed);
      ("a peer's", peer);
      ("freed", freed);
      ("another GPU's", r);
    ];
  Memory.free x.t_memory peer;
  Memory.free o.t_memory r;
  Memory.free x.t_memory borrowed;
  Function.free_dma x.t_fn lent

(* A GPU whose bus mastering a release could not turn off

   A kernel locked down while the process holds the GPU refuses every write
   to its configuration space, the one that turns its bus mastering off
   included. The GPU may then go on writing its memory, so no free gives it
   back: neither the free after the release, nor one after a release that
   keeps the GPU taken because another function was given its memory. *)

type stop = Released | Kept

let stops = [ ("released", Released); ("kept for another function", Kept) ]

let test_refused_stop (_, stop) =
  incr given_spaces;
  let base = (1 lsl 42) + (!given_spaces * space_length) in
  let root, m, buses = tree_machine ~others:[ nic () ] ~space:base 1 in
  let bus = List.hd buses in
  let config =
    Tree.in_memory root (strf "sys/bus/pci/devices/%s/config" bus)
  in
  Fun.protect ~finally:(fun () -> Unix.close config) @@ fun () ->
  let x = take_gpu ~space:(Space.create ~base space_length) m bus in
  let host = given (Memory.alloc x.t_memory Host (64 * kib)) in
  (match stop with
  | Released -> ()
  | Kept ->
      let r = given (Memory.alloc x.t_memory Gpu (64 * kib)) in
      ignore (require_ok (Memory.bus x.t_memory r ~client:nic_bus)));
  Function.set_bus_master x.t_fn true;
  Tree.refuse_writes config;
  let bytes = held_bytes root and addresses = snd (tree_capacity x) in
  Memory.release x.t_memory;
  equal ~msg:"bus mastering after the release" int 0x4
    (command_of root bus land 0x4);
  Memory.free x.t_memory host;
  equal ~msg:"system memory kept through the free" int bytes
    (held_bytes root);
  equal ~msg:"its addresses kept" int addresses (snd (tree_capacity x))

let others =
  group ~timeout:patience "memory given to another function"
    [
      cases "bus gives the memory's bus addresses"
        ~name:(Format.asprintf "%a" pp_kind)
        [ Memory.Gpu; Bar; Host ] test_bus_runs;
      test
        "given memory never returns, and the GPU stays taken until the \
         process exits"
        test_bus_reserved;
      cases "a refusal names its cause and reserves nothing"
        ~name:(fun (n, _, _, _, _) -> n)
        refusals test_bus_refused;
      cases
        "a GPU whose bus mastering a release could not turn off keeps its \
         memory through every free"
        ~name:fst stops test_refused_stop;
      test "system memory needs no BAR and no route" test_bus_host_small;
      test "a GPU a transport reaches gives no memory" test_bus_far;
      test "bus of memory alloc did not give raises" test_bus_misuse;
    ]

let () = exit @@ run "rig_pci.memory" [ placement; on_files; others ]
