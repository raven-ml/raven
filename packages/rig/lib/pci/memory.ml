(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external reraise : exn -> 'a = "%reraise"

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

type kind = Gpu | Bar | Host
type link = { fabric : int64; node : int }

type reach = {
  machine : Machine.t;
  bus : string; (* the GPU's, which names it on its machine *)
  space : Space.t;
  addressing : Machine.addressing;
  small_bar : bool;
  link : link option;
  route : Sysfs.route option option Atomic.t;
      (* its way up the bus, without a transport: [None] until its first use
         reads it *)
}

(* The memory a GPU addresses: the facts [reaches] and errors read, and [held],
   what its regions need. [held] goes once the function is released and no
   region is out. Another GPU's [map_peer] and [free] read [held]'s regions
   and write their [peers] and [reached] while this GPU's owner makes its
   calls: those reads and every write of them take [lock], which holds no
   other lock, so no two GPUs' calls wait for each other in a cycle. *)
type t = {
  reach : reach;
  bar : int;
  bar_base : int;
  bar_size : int;
  lender : Function.t; (* the GPU's function, which lends ([enter]) *)
  lock : Lock.t;
  mutable held : held option;
}

(* The function, and every mapping it holds by GPU address, so that a free or a
   map of memory it does not hold is refused. A region of system memory whose
   unmap the GPU did not confirm waits in [unconfirmed] until a free after the
   function's release. A mapping whose last region was freed while other GPUs
   map its memory waits in [parked], with its memory and addresses, until
   none does: [due] is set, under [lock], once one may have none left, and
   the owner's next call gives back those that have none, or, once the
   function is released, the peer's free that left the last. [release] empties
   [tables], which only a live GPU's requests use: a free after it writes no
   table. *)
and held = {
  fn : Function.t;
  regions : shared Tables.Address.t;
  mutable unconfirmed : region list;
  mutable parked : shared list;
  due : bool Atomic.t;
  mutable tables : Page_table.t option; (* [None] once released *)
  mutable given : int list;
      (* the GPU addresses of the mappings [bus] gave another function, which
         nothing gives back *)
}

(* A region maps [mapping] for [owner]'s GPU. Its first byte lies [off] bytes
   into the mapping, as [map_host]'s does for memory that does not start on a
   page. [pinned] is the range [owner]'s function pinned for it. *)
and region = {
  owner : t;
  mapping : Page_table.mapping;
  off : int;
  host : Window.t option;
  source : source;
  pinned : (int * int) option;
}

and source = Allocated | Borrowed of int | Peer of region

(* A mapping and the regions out over it: [holder] made it and holds what it
   took, and each [map_peer] of memory the GPU maps already adds a region to
   [out]. The last free gives back [holder]'s. [peers] counts other GPUs'
   mappings of [holder]'s memory that are out, and [reached] is whether
   another GPU freed one that it may still reach: its GPU did not confirm
   the unmap, or its function was released and may still master the bus.
   Such memory never goes back. *)
and shared = {
  holder : region;
  mutable out : region list;
  mutable peers : int;
  mutable reached : bool;
}

(* The GPU's page is 4 KiB, the page every format's leaf level maps
   (Page_table.format). Its large page is 2 MiB, what the level above maps: a
   leaf table is a 4 KiB page of 512 8-byte entries. Allocations of [large]
   bytes or more round to large pages, so that their tail maps with them; the
   rounding then costs at most a quarter of the allocation. *)
let page = 0x1000
let large_page = 2 lsl 20
let large = 4 * large_page
let round_up n a = (n + a - 1) / a * a

(* Whether [n] rounded up to [a] passes [max_int]: no memory holds it. *)
let wraps n a = n > max_int - a + 1

let create ?link fn tables ~bar =
  let bar_base, bar_size =
    match Function.bar fn bar with
    | Some b -> b
    | None ->
        invalid_argf "Memory.create: %s has no BAR %d" (Function.bus fn) bar
  in
  Function.set_dma_bits fn (Page_table.pa_bits tables);
  {
    reach =
      {
        machine = Function.machine fn;
        bus = Function.bus fn;
        space = Page_table.space tables;
        addressing = Function.addressing fn;
        small_bar = bar_size < Page_table.memory tables;
        link;
        route = Atomic.make None;
      };
    bar;
    bar_base;
    bar_size;
    lender = fn;
    lock = Lock.create ();
    held =
      Some
        {
          fn;
          regions = Tables.Address.create 64;
          unconfirmed = [];
          parked = [];
          due = Atomic.make false;
          tables = Some tables;
          given = [];
        };
  }

(* Drops [h] once nothing needs it: the function is released and no region is
   out. *)
let settle m h =
  if Function.released h.fn then
    Lock.protect m.lock (fun () ->
        if
          Tables.Address.length h.regions = 0
          && h.unconfirmed = [] && h.parked = []
        then m.held <- None)

let source r = r.source
let address r = r.mapping.va + r.off

let pages r =
  let rec from off = function
    | (_, n) :: rest when off >= n -> from (off - n) rest
    | (a, n) :: rest -> (a + off, n - off) :: rest
    | [] -> []
  in
  (r.mapping.target, from r.off r.mapping.pages)

let rec host r =
  match r.source with
  | Peer origin -> host origin
  | Allocated | Borrowed _ -> r.host

let reach m = m.reach

(* A route is read from the machine's files when a peer or another function
   first needs it, which a session that reaches neither never does: 4 ms of an
   open on an R9700. The read takes no lock, as a peer's check must not
   ([Rig_edge.Driver.peer]): checks at once, a callback's inside one
   included, each read the same files, and the first store stays. An
   exception that ends a read stores nothing, so the next check reads
   again. *)
let route (r : reach) =
  match Atomic.get r.route with
  | Some route -> route
  | None ->
      let route =
        Option.bind (Machine.files r.machine) (fun files ->
            Sysfs.route files r.bus)
      in
      ignore (Atomic.compare_and_set r.route None (Some route));
      route

let linked (r : reach) (o : reach) =
  match (r.link, o.link) with
  | Some l, Some l' -> Int64.equal l.fabric l'.fabric
  | _ -> false

(* Whether the bus carries [r]'s requests to [o]'s BAR, which only a machine
   reached without a transport tells. *)
let carried (r : reach) (o : reach) =
  match (route r, route o) with
  | Some client, Some provider -> Sysfs.forwards ~provider ~client
  | _ -> false

(* A GPU maps its own memory already. A link's entries name the peer's own
   memory, which crosses no IOMMU; a BAR's name its bus address, which the
   IOMMU translates. *)
let reaches (r : reach) (o : reach) =
  r.machine == o.machine
  && r.bus <> o.bus
  && r.space == o.space
  && (linked r o
     || r.addressing = Machine.Physical
        && o.addressing = Machine.Physical
        && (not o.small_bar) && carried r o)

(* Errors *)

let beyond m tables runs =
  let bits = Page_table.pa_bits tables in
  match List.find_opt (fun (pa, n) -> pa > (1 lsl bits) - n) runs with
  | None -> None
  | Some (pa, _) ->
      Some
        (strf
           "the peer's memory lies at 0x%x, past the %d bits of address %s's \
            page tables hold"
           pa bits m.reach.bus)

(* Allocating *)

let allocated m mapping =
  {
    owner = m;
    mapping;
    off = 0;
    host = None;
    source = Allocated;
    pinned = None;
  }

(* System memory at the same address for the process and the GPU. *)
let system m h tables n =
  let page = Machine.page m.reach.machine in
  if wraps n page then Ok None
  else
    let n = round_up n page in
    match Space.alloc ~align:page m.reach.space n with
    | None -> Ok None
    | Some va -> (
        let give_back view =
          Option.iter (Function.free_dma h.fn) view;
          Space.free m.reach.space va
        in
        match Function.alloc_dma h.fn ~va n with
        | Error _ as e ->
            give_back None;
            e
        | Ok None ->
            give_back None;
            Ok None
        | exception (Invalid_argument _ as e) ->
            (* Function.alloc_dma refuses its arguments, such as addresses no
               reservation holds, before it takes anything. Another exception
               may come once the memory is mapped at [va], which then stays
               taken, since the function may still reach it there. *)
            Space.free m.reach.space va;
            reraise e
        | Ok (Some (view, runs)) -> (
            match
              Page_table.map ~snooped:true ~uncached:true tables ~va System
                runs
            with
            | Some mapping ->
                Ok (Some { (allocated m mapping) with host = Some view })
            | None ->
                give_back (Some view);
                Ok None))

(* The GPU's memory, one block the process reaches through the BAR when [bar],
   placed under the BAR's end before anything is written, or [None] if no block
   fits there. *)
let gpu m h tables ~uncached ~bar n =
  let unit = if n >= large then large_page else page in
  if wraps n unit then Ok None
  else
    let n = round_up n unit in
    let below = if bar then Some m.bar_size else None in
    match Page_table.alloc ~uncached ~contiguous:bar ?below tables n with
    | None -> Ok None
    | Some mapping when not bar -> Ok (Some (allocated m mapping))
    | Some mapping -> (
        let pa = fst (List.hd mapping.pages) in
        match Function.map ~off:pa ~length:mapping.size h.fn m.bar with
        | Ok host -> Ok (Some { (allocated m mapping) with host = Some host })
        | Error why ->
            (* Memory the GPU may still reach stays: the GPU is lost then. *)
            ignore (Page_table.free tables mapping : bool);
            Error why)

let positive fn n =
  if n <= 0 then invalid_argf "Memory.%s: %d bytes, expected more than 0" fn n

(* What a GPU not released holds, with its tables. A released GPU may be
   another instance's: no request writes its tables. *)
let live fn m =
  match m.held with
  | Some ({ tables = Some tables; _ } as h) when not (Function.released h.fn) ->
      (h, tables)
  | Some _ | None -> invalid_argf "Memory.%s: %s is released" fn m.reach.bus

(* Gives back what [r] holds but its entries, with the GPU's [tables] while it
   is live: once the function is released the GPU may be another instance's, so
   a free gives back system memory, pins and addresses only. The addresses go
   back last: the vendor's GPUs share the space, and another GPU's system memory
   there would be mapped over this one's. A released function that may still
   reach its memory ([Function.reaching]) keeps the memory, and so its
   addresses. *)
let give_back m h tables r =
  let map = r.mapping in
  (match (r.source, map.target, r.host, tables) with
  | Allocated, System, Some view, _ -> Function.free_dma h.fn view
  | Allocated, _, Some view, Some _ -> Function.unmap_used h.fn view
  | _ -> ());
  Option.iter (fun (a, n) -> Function.unpin h.fn a n) r.pinned;
  (match (r.source, map.target, tables) with
  | Allocated, Gpu, Some tables ->
      List.iter (fun (pa, _) -> Page_table.pfree tables pa) map.pages
  | _ -> ());
  match r.source with
  | (Allocated | Borrowed _) when not (Function.reaching h.fn) ->
      Space.free m.reach.space map.va
  | Allocated | Borrowed _ | Peer _ -> ()

(* [r]'s entry, out or parked. *)
let entry_of h r =
  match Tables.Address.find_opt h.regions r.mapping.va with
  | Some e when e.holder == r -> Some e
  | Some _ | None -> List.find_opt (fun e -> e.holder == r) h.parked

(* Under [m]'s lock: the parked mappings no other GPU maps now, taken off
   [parked], those that one may still reach dropped, which never go back. *)
let take_due h =
  Atomic.set h.due false;
  let gone, left = List.partition (fun e -> e.peers = 0) h.parked in
  h.parked <- left;
  List.filter (fun e -> not e.reached) gone

(* Whether a mapping of [r]'s memory lends its owner's function: GPU memory,
   which a reset of the owner would give out. *)
let lent r = r.mapping.target <> System

(* Uncounts a mapping of the origin [r] that another GPU, of function
   [writer], gave back, or never made, and ends its lend. Unless [gone], that
   GPU may still write the memory, which then stays reached, and the lend
   lasts until [writer] is released and quiet. The last one of a parked
   mapping makes it due. Once the owner's function is released, no owner's
   call may come to give it back, so the peer's thread gives it back here:
   that writes no table, and takes only the locks of the function's memory
   and of the space, below this one. *)
let rec leave r ~gone ~writer =
  let o = r.owner in
  let section () =
    match o.held with
    | None -> None
    | Some h -> (
        match entry_of h r with
        | Some e ->
            e.peers <- e.peers - 1;
            if not gone then e.reached <- true;
            if e.peers = 0 && e.out = [] then begin
              Atomic.set h.due true;
              Some h
            end
            else None
        | None -> None)
  in
  (match Lock.protect o.lock section with
  | Some h when Function.released h.fn ->
      give_due o h None;
      settle o h
  | Some _ | None -> ());
  if lent r then
    if gone then Function.unlend o.lender
    else Function.unlend_after o.lender writer

(* Unmaps [e], a mapping no region is out over and no other GPU maps, and
   gives back its holder's, or leaves it to a free after the release if the
   GPU does not confirm the unmap. A mapping of another GPU's memory is
   uncounted there once unmapped. *)
and release_entry m h tables e =
  let r = e.holder in
  let confirmed =
    match tables with
    | None -> true
    | Some tables -> Page_table.unmap tables ~va:r.mapping.va r.mapping.size
  in
  (match r.source with
  | Peer o ->
      leave o ~gone:(confirmed && not (Function.reaching h.fn)) ~writer:h.fn
  | Allocated | Borrowed _ -> ());
  (* Memory the GPU may still reach stays until the GPU is released. *)
  if confirmed then give_back m h tables r
  else h.unconfirmed <- r :: h.unconfirmed

(* Gives back the parked mappings that no other GPU maps any more, on the
   owner's call, [tables] as that call has them, or on a peer's once the
   function is released. Each is taken off [parked] under [m]'s lock, so it
   goes back once. *)
and give_due m h tables =
  if Atomic.get h.due then
    List.iter (release_entry m h tables)
      (Lock.protect m.lock (fun () -> take_due h))

let release m =
  Option.iter
    (fun h ->
      Function.release h.fn;
      h.tables <- None;
      give_due m h None;
      settle m h)
    m.held

(* [using fn m f] is [f h tables] for the GPU [m] holds, run as a use of its
   function: its tables lie behind the function's BAR windows, which a
   release that begins meanwhile leaves mapped until the use ends. The use's
   handler is the next step after its count. *)
let using fn m f =
  let h, tables = live fn m in
  if not (Function.use h.fn) then
    invalid_argf "Memory.%s: %s is released" fn m.reach.bus;
  match
    give_due m h (Some tables);
    f h tables
  with
  | r ->
      Function.unuse h.fn;
      r
  | exception e ->
      Function.unuse h.fn;
      reraise e

let keep m h r =
  let e = { holder = r; out = [ r ]; peers = 0; reached = false } in
  Lock.protect m.lock (fun () ->
      Tables.Address.replace h.regions r.mapping.va e)

let alloc ?(uncached = false) m kind n =
  using "alloc" m @@ fun h tables ->
  positive "alloc" n;
  let r =
    match kind with
    | Host -> system m h tables n
    | Gpu -> gpu m h tables ~uncached ~bar:false n
    | Bar -> gpu m h tables ~uncached ~bar:true n
  in
  (match r with Ok (Some r) -> keep m h r | Ok None | Error _ -> ());
  r

(* Mapping *)

(* Pinned pages map at [va], with what [r] needs to give them back. *)
let map_pinned m h tables ~va ~source ~host a n =
  match Function.pin h.fn a n with
  | (Error _ | Ok None) as r -> r
  | Ok (Some runs) -> (
      match
        Page_table.map ~snooped:true ~uncached:true tables ~va System runs
      with
      | None ->
          Function.unpin h.fn a n;
          Ok None
      | Some mapping ->
          Ok
            (Some
               {
                 owner = m;
                 mapping;
                 off = 0;
                 host;
                 source;
                 pinned = Some (a, n);
               }))

(* Borrowed memory goes at addresses of the GPU's space, wherever it lies for
   the process: a process's heap and stacks lie far above the addresses a GPU's
   rings take. *)
let map_host m a n =
  using "map_host" m @@ fun h tables ->
  positive "map_host" n;
  let page = Machine.page m.reach.machine in
  if n > max_int - a || wraps (a + n) page then
    Error
      (strf "%d bytes at 0x%x pass the last address of %s's machine" n a
         m.reach.bus)
  else
    let first = a / page * page in
    let bytes = round_up (a + n) page - first in
    match Space.alloc ~align:page m.reach.space bytes with
    | None -> Ok None
    | Some va -> (
        let host = Some (Machine.window m.reach.machine a n) in
        match
          map_pinned m h tables ~va ~source:(Borrowed a) ~host first bytes
        with
        | (Error _ | Ok None) as r ->
            Space.free m.reach.space va;
            r
        | Ok (Some r) ->
            let r = { r with off = a - first } in
            keep m h r;
            Ok (Some r))

let rec origin r = match r.source with Peer o -> origin o | _ -> r

let holds h r =
  match Tables.Address.find_opt h.regions r.mapping.va with
  | Some e -> List.memq r e.out
  | None -> false

(* What [m] holds, of which [r] is a region. *)
let mine fn m r =
  match m.held with
  | Some h when holds h r -> h
  | Some _ | None ->
      invalid_argf
        "Memory.%s: the memory at 0x%x is not %s's, or was given back" fn
        r.mapping.va m.reach.bus

(* [mine], called by another GPU's request, as its owner may be in a call. *)
let theirs fn m r = Lock.protect m.lock (fun () -> mine fn m r)

(* Peers

   Whether a region's memory may go back at its last free depends on the
   other GPUs that map it, which their owners' calls change: each change is
   one section under the owner's lock, the only lock it holds. *)

(* The refusal of a released GPU's memory, which another open may have reset
   and given out. *)
let released_error (o : t) =
  Error
    (strf
       "%s was released, and another open may have reset it and given its \
        memory out"
       o.reach.bus)

(* Counts a new mapping of the origin [r] by another GPU, under [r]'s owner's
   lock, if [r] is still out, and for GPU memory as a lend of the owner's
   function, which keeps its release from giving its take back
   ({!Function.lend}). A release begun first refuses it. The counts are its
   last steps. *)
let enter fn r =
  let o = r.owner in
  let counted =
    Lock.protect o.lock @@ fun () ->
    let h = mine fn o r in
    let e = Tables.Address.find h.regions r.mapping.va in
    if lent r && not (Function.lend o.lender) then false
    else begin
      e.peers <- e.peers + 1;
      true
    end
  in
  (* A refused lend settles the owner's function outside the lock, since
     that may give back its take ({!Function.lend_refused}). *)
  if not counted then Function.lend_refused o.lender;
  counted

(* [f ()], a new mapping of the origin [r]'s memory for another GPU, counted
   from before it writes the tables. [f] is made before the count, and the
   count is the last step before the handler. A refusal takes nothing; any
   other exception may leave the tables pointing at the memory, which then
   stays reached. *)
let counted_map h r f =
  if not (enter "map_peer" r) then released_error r.owner
  else
    match f () with
    | exception (Invalid_argument _ as e) ->
        leave r ~gone:true ~writer:h.fn;
        reraise e
    | exception e ->
        leave r ~gone:false ~writer:h.fn;
        reraise e
    | (Ok None | Error _) as e ->
        leave r ~gone:true ~writer:h.fn;
        e
    | Ok (Some _) as ok -> ok

(* A new mapping, for [m]'s GPU, of [r], another GPU's memory, at its owner's
   address. *)
let map_new m h tables r =
  let o = r.owner in
  let map = r.mapping in
  let mapped = function
    | Ok (Some p) ->
        let p = { p with off = r.off } in
        keep m h p;
        Ok (Some p)
    | (Ok None | Error _) as e -> e
  in
  if m.reach.machine != o.reach.machine then
    Error "the GPUs are on different machines"
  else if m.reach.space != o.reach.space then
    Error
      (strf
         "%s and %s address different spaces: a peer's memory maps at its \
          owner's address, which is that memory only in a space both share"
         m.reach.bus o.reach.bus)
  else
    match map.target with
    | System ->
        (* Borrowed memory is pinned where the process holds it, allocated
           memory at its own address. *)
        let a, n = Option.value r.pinned ~default:(map.va, map.size) in
        mapped
          (counted_map h r (fun () ->
               map_pinned m h tables ~va:map.va ~source:(Peer r) ~host:None a
                 n))
    | (Gpu | Peer _) when not (reaches m.reach o.reach) ->
        Error
          (strf
             "%s does not reach %s's memory: it needs a link of one fabric, or \
              both GPUs off an IOMMU, a memory BAR as large as the memory, \
              which Resizable BAR in the firmware settings gives, and a bus \
              that carries requests between them, as Linux's p2pdma decides"
             m.reach.bus o.reach.bus)
    | Gpu | Peer _ -> (
        let target, pages =
          match (m.reach.link, o.reach.link) with
          | Some _, Some l when linked m.reach o.reach ->
              (Page_table.Peer l.node, map.pages)
          | _ ->
              ( Page_table.System,
                List.map (fun (pa, n) -> (pa + o.bar_base, n)) map.pages )
        in
        match beyond m tables pages with
        | Some why -> Error why
        | None -> (
            let uncached = Some map.uncached in
            mapped
              (counted_map h r (fun () ->
                   match
                     Page_table.map ~snooped:true ?uncached tables ~va:map.va
                       target pages
                   with
                   | None -> Ok None
                   | Some mapping ->
                       Ok
                         (Some
                            {
                              owner = m;
                              mapping;
                              off = 0;
                              host = None;
                              source = Peer r;
                              pinned = None;
                            })))))

(* Misuse is refused before anything is pinned or written: a region freed, or
   its origin freed. A released GPU's own memory is refused: another open may
   have reset that GPU and given the memory out. Its system memory stays until
   its region's free, and maps as a live GPU's does. Memory the GPU maps
   already, its own or a peer's it mapped, lies at its owner's address, so a
   new region there shares that mapping. *)
let map_peer m r =
  using "map_peer" m @@ fun h tables ->
  ignore (theirs "map_peer" r.owner r : held);
  let r = origin r in
  let owner = theirs "map_peer" r.owner r in
  let gone =
    m.reach.machine == r.owner.reach.machine
    && r.mapping.target <> System
    && Function.released owner.fn
  in
  match Tables.Address.find_opt h.regions r.mapping.va with
  | _ when gone -> released_error r.owner
  | None -> map_new m h tables r
  | Some e when origin e.holder != r ->
      invalid_argf "Memory.map_peer: %s maps other memory at 0x%x" m.reach.bus
        r.mapping.va
  | Some e ->
      let p =
        {
          e.holder with
          off = r.off;
          host = None;
          source = Peer r;
          pinned = None;
        }
      in
      Lock.protect m.lock (fun () -> e.out <- p :: e.out);
      Ok (Some p)

(* Giving memory to another function

   Another function of the machine, such as a network adapter, reaches GPU
   memory at its bus address through the memory BAR, as a peer GPU does
   ([map_new]), and system memory at its physical address. Neither request
   may cross an IOMMU that translates it, and a BAR request must take a route
   Linux's p2pdma allows (drivers/pci/p2pdma.c). The function's writes go to
   the memory itself, whatever the GPU's tables say, and nothing tells this
   library when they stop: so the memory is never given back, and the GPU's
   function stays the process's ([Function.keep]). *)

let refuse fmt = Printf.ksprintf (fun s -> Error s) fmt

let reach_check m files ~client =
  let gpu = m.reach.bus in
  if m.reach.addressing = Machine.Iommu then
    refuse
      "%s is behind an IOMMU, which translates its requests and not another \
       function's"
      gpu
  else if Sysfs.translates files client then
    refuse "%s is behind an IOMMU, which translates its requests" client
  else Ok ()

let bar_check m files ~client =
  let gpu = m.reach.bus in
  if m.reach.small_bar then
    refuse
      "%s's memory BAR is smaller than its memory; Resizable BAR in the \
       firmware settings makes it as large"
      gpu
  else
    match (route m.reach, Sysfs.route files client) with
    | Some provider, Some client when Sysfs.forwards ~provider ~client -> Ok ()
    | _ ->
        refuse
          "the bus does not carry %s's requests to %s's memory BAR, as Linux's \
           p2pdma decides"
          client gpu

let known m client =
  match Machine.functions m.reach.machine with
  | Error why ->
      refuse "listing the functions of %s's machine: %s" m.reach.bus why
  | Ok ids when List.exists (fun (id : Machine.id) -> id.bus = client) ids ->
      Ok ()
  | Ok _ -> refuse "%s names no PCI function of %s's machine" client m.reach.bus

let bus m r ~client =
  let h, _ = live "bus" m in
  ignore (mine "bus" m r : held);
  if r.source <> Allocated then
    invalid_argf "Memory.bus: the memory at 0x%x is not memory alloc gave"
      r.mapping.va;
  let* () =
    if Option.is_none (Bus_address.numbers client) then
      refuse "%S is no PCI bus address, expected DDDD:BB:DD.F" client
    else Ok ()
  in
  let* files =
    match Machine.files m.reach.machine with
    | Some files -> Ok files
    | None ->
        refuse
          "%s's machine is reached through a transport, which no other \
           function's requests cross"
          m.reach.bus
  in
  let* () = known m client in
  let* () = reach_check m files ~client in
  let map = r.mapping in
  let* runs =
    match map.target with
    | Page_table.System -> Ok map.pages
    | Gpu | Peer _ ->
        let* () = bar_check m files ~client in
        Ok (List.map (fun (pa, n) -> (pa + m.bar_base, n)) map.pages)
  in
  if not (List.mem map.va h.given) then h.given <- map.va :: h.given;
  Function.keep h.fn;
  Ok runs

(* Freeing *)

(* What a free leaves of a mapping: other regions out over it, nothing,
   nothing with memory that another GPU maps, or nothing with memory that
   never goes back: another GPU may still reach it, or another function was
   given it ({!bus}). *)
type left = Others | Last | Parked | Kept

(* Takes [r] off its entry [e] of [h], under [m]'s lock. The last region of a
   mapping whose memory another GPU maps parks it, with its memory and
   addresses, in the same section, so that [leave] finds it. *)
let take_off m h e r =
  let given = List.mem r.mapping.va h.given in
  Lock.protect m.lock @@ fun () ->
  let out = List.filter (fun r' -> r' != r) e.out in
  e.out <- out;
  if out <> [] then Others
  else begin
    Tables.Address.remove h.regions r.mapping.va;
    if e.reached || given then Kept
    else if e.peers > 0 then begin
      h.parked <- e :: h.parked;
      Parked
    end
    else Last
  end

(* The last region out over a mapping gives the mapping back, or parks it
   while another GPU maps its memory. [tables] are the GPU's while the free
   counts as a use of the function. *)
let free_with m h r tables =
  let e = Tables.Address.find h.regions r.mapping.va in
  (match take_off m h e r with
  | Others | Parked | Kept -> ()
  | Last -> release_entry m h tables e);
  give_due m h tables;
  if Option.is_none tables && h.unconfirmed <> [] then begin
    List.iter (give_back m h None) h.unconfirmed;
    h.unconfirmed <- []
  end;
  settle m h

(* A free that writes the GPU's tables, through its BAR windows, counts as a
   use of the function, as [using]'s requests do, so that a release that
   begins meanwhile leaves the function's give-back, which unmaps them, to
   the free's end. The use's handler is the next step after its count. Once
   a release has begun the free writes no table. *)
let free m r =
  let h = mine "free" m r in
  if Function.use h.fn then
    match free_with m h r h.tables with
    | () -> Function.unuse h.fn
    | exception e ->
        Function.unuse h.fn;
        reraise e
  else free_with m h r None
