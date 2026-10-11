(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type kind = Gpu | Bar | Host
type link = { fabric : int64; node : int }

(* The memory a GPU addresses: the facts [reaches] and errors read, and [held],
   what its regions need. A driver keeps [t] past the GPU's stop, for the life
   of the process, so [held] goes once the function is released and no region
   is out, and [t] then keeps a few words. *)
type t = {
  bus : string;
  machine : Machine.t;
  iommu : bool; (* the function reaches system memory through an IOMMU *)
  space : Space.t;
  bar : int;
  bar_base : int;
  bar_size : int;
  small_bar : bool; (* the BAR reaches part of the memory *)
  link : link option;
  mutable held : held option;
}

(* The function, and every region it holds by GPU address, so that a free or a
   map of memory it does not hold is refused. A region of system memory whose
   unmap the GPU did not confirm waits in [unconfirmed] until a free after the
   function's release. [release] empties [tables], which only a live GPU's
   requests use: a free after it writes no table. *)
and held = {
  fn : Function.t;
  regions : region Tables.Address.t;
  mutable unconfirmed : region list;
  mutable tables : Page_table.t option; (* [None] once released *)
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
  {
    bus = Function.bus fn;
    machine = Function.machine fn;
    iommu = Function.addressing fn = Machine.Iommu;
    space = Page_table.space tables;
    bar;
    bar_base;
    bar_size;
    small_bar = bar_size < Page_table.memory tables;
    link;
    held =
      Some
        {
          fn;
          regions = Tables.Address.create 64;
          unconfirmed = [];
          tables = Some tables;
        };
  }

(* Drops [h] once nothing needs it: the function is released and no region is
   out. *)
let settle m h =
  if
    Function.released h.fn
    && Tables.Address.length h.regions = 0
    && h.unconfirmed = []
  then m.held <- None

let release m =
  Option.iter
    (fun h ->
      Function.release h.fn;
      h.tables <- None;
      settle m h)
    m.held

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

let linked m o =
  match (m.link, o.link) with
  | Some l, Some l' -> Int64.equal l.fabric l'.fabric
  | _ -> false

(* A link's entries name the peer's own memory, which crosses no IOMMU; a BAR's
   name its bus address, which the IOMMU translates. *)
let reaches m o =
  m != o
  && m.machine == o.machine
  && m.space == o.space
  && (linked m o || not (m.iommu || o.iommu || o.small_bar))

(* Errors *)

let beyond m tables runs =
  let bits = Page_table.pa_bits tables in
  match List.find_opt (fun (pa, n) -> pa > (1 lsl bits) - n) runs with
  | None -> None
  | Some (pa, _) ->
      Some
        (strf
           "the machine gave memory at 0x%x, past the %d bits of address %s's \
            page tables hold"
           pa bits m.bus)

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
  let page = Machine.page m.machine in
  if wraps n page then Ok None
  else
    let n = round_up n page in
    match Space.alloc ~align:page m.space n with
    | None -> Ok None
    | Some va -> (
        let give_back view =
          Option.iter (Function.free_dma h.fn) view;
          Space.free m.space va
        in
        match Function.alloc_dma h.fn ~va n with
        | Error _ as e ->
            give_back None;
            e
        | Ok None ->
            give_back None;
            Ok None
        | Ok (Some (view, runs)) -> (
            match beyond m tables runs with
            | Some why ->
                give_back (Some view);
                Error why
            | None -> (
                match
                  Page_table.map ~snooped:true ~uncached:true tables ~va
                    System runs
                with
                | Some mapping ->
                    Ok (Some { (allocated m mapping) with host = Some view })
                | None ->
                    give_back (Some view);
                    Ok None)))

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
  | Some _ | None -> invalid_argf "Memory.%s: %s is released" fn m.bus

let keep h r = Tables.Address.replace h.regions r.mapping.va r

let alloc ?(uncached = false) m kind n =
  let h, tables = live "alloc" m in
  positive "alloc" n;
  let r =
    match kind with
    | Host -> system m h tables n
    | Gpu -> gpu m h tables ~uncached ~bar:false n
    | Bar -> gpu m h tables ~uncached ~bar:true n
  in
  (match r with Ok (Some r) -> keep h r | Ok None | Error _ -> ());
  r

(* Mapping *)

(* Pinned pages map at [va], with what [r] needs to give them back. *)
let map_pinned m h tables ~va ~source ~host a n =
  match Function.pin h.fn a n with
  | Error _ as e -> e
  | Ok runs -> (
      let unpin () = Function.unpin h.fn a n in
      match beyond m tables runs with
      | Some why ->
          unpin ();
          Error why
      | None -> (
          match
            Page_table.map ~snooped:true ~uncached:true tables ~va System runs
          with
          | None ->
              unpin ();
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
                   })))

(* Borrowed memory goes at addresses of the GPU's space, wherever it lies for
   the process: a process's heap and stacks lie far above the addresses a GPU's
   rings take. *)
let map_host m a n =
  let h, tables = live "map_host" m in
  positive "map_host" n;
  let page = Machine.page m.machine in
  if n > max_int - a || wraps (a + n) page then
    Error
      (strf "%d bytes at 0x%x pass the last address of %s's machine" n a m.bus)
  else
    let first = a / page * page in
    let bytes = round_up (a + n) page - first in
    match Space.alloc ~align:page m.space bytes with
    | None -> Ok None
    | Some va -> (
        let host = Some (Machine.window m.machine a n) in
        match
          map_pinned m h tables ~va ~source:(Borrowed a) ~host first bytes
        with
        | (Error _ | Ok None) as r ->
            Space.free m.space va;
            r
        | Ok (Some r) ->
            let r = { r with off = a - first } in
            keep h r;
            Ok (Some r))

let rec origin r = match r.source with Peer o -> origin o | _ -> r

let holds h r =
  match Tables.Address.find_opt h.regions r.mapping.va with
  | Some r' -> r' == r
  | None -> false

(* What [m] holds, of which [r] is a region. *)
let mine fn m r =
  match m.held with
  | Some h when holds h r -> h
  | Some _ | None ->
      invalid_argf
        "Memory.%s: the memory at 0x%x is not %s's, or was given back" fn
        r.mapping.va m.bus

(* Misuse is refused before anything is pinned or written: a region freed, its
   origin freed, or addresses the GPU maps already. *)
let map_peer m r =
  let h, tables = live "map_peer" m in
  ignore (mine "map_peer" r.owner r : held);
  let r = origin r in
  let o = r.owner in
  ignore (live "map_peer" o : held * Page_table.t);
  ignore (mine "map_peer" o r : held);
  if o == m then
    invalid_argf "Memory.map_peer: the memory at 0x%x is %s's own" r.mapping.va
      m.bus;
  if Tables.Address.mem h.regions r.mapping.va then
    invalid_argf "Memory.map_peer: %s maps the memory at 0x%x already"
      m.bus r.mapping.va;
  let map = r.mapping in
  let mapped = function
    | Ok (Some p) ->
        let p = { p with off = r.off } in
        keep h p;
        Ok (Some p)
    | (Ok None | Error _) as e -> e
  in
  if m.machine != o.machine then Error "the GPUs are on different machines"
  else if m.space != o.space then
    Error
      (strf
         "%s and %s address different spaces: a peer's memory maps at its \
          owner's address, which is that memory only in a space both share"
         m.bus o.bus)
  else
    match map.target with
    | System ->
        (* Borrowed memory is pinned where the process holds it, allocated
           memory at its own address. *)
        let a, n = Option.value r.pinned ~default:(map.va, map.size) in
        mapped
          (map_pinned m h tables ~va:map.va ~source:(Peer r) ~host:None a n)
    | (Gpu | Peer _) when not (reaches m o) ->
        Error
          (strf
             "%s does not reach %s's memory: it needs a link of one fabric, or \
              both GPUs off an IOMMU and a memory BAR as large as the memory, \
              which Resizable BAR in the firmware settings gives"
             m.bus o.bus)
    | Gpu | Peer _ -> (
        let target, pages =
          match (m.link, o.link) with
          | Some _, Some l when linked m o -> (Page_table.Peer l.node, map.pages)
          | _ ->
              ( Page_table.System,
                List.map (fun (pa, n) -> (pa + o.bar_base, n)) map.pages )
        in
        match beyond m tables pages with
        | Some why -> Error why
        | None -> (
            match
              Page_table.map ~snooped:true ~uncached:map.uncached tables
                ~va:map.va target pages
            with
            | None -> Ok None
            | Some mapping ->
                mapped
                  (Ok
                     (Some
                        {
                          owner = m;
                          mapping;
                          off = 0;
                          host = None;
                          source = Peer r;
                          pinned = None;
                        }))))

(* Freeing *)

(* Gives back what [r] holds but its entries, with the GPU's [tables] while it
   is live: once the function is released the GPU may be another instance's, so
   a free gives back system memory, pins and addresses only. The addresses go
   back last: the vendor's GPUs share the space, and another GPU's system memory
   there would be mapped over this one's. *)
let give_back m h tables r =
  let map = r.mapping in
  (match (r.source, map.target, r.host, tables) with
  | Allocated, System, Some view, _ -> Function.free_dma h.fn view
  | Allocated, _, Some view, Some _ -> Function.unmap h.fn view
  | _ -> ());
  Option.iter (fun (a, n) -> Function.unpin h.fn a n) r.pinned;
  (match (r.source, map.target, tables) with
  | Allocated, Gpu, Some tables ->
      List.iter (fun (pa, _) -> Page_table.pfree tables pa) map.pages
  | _ -> ());
  match r.source with
  | Allocated | Borrowed _ -> Space.free m.space map.va
  | Peer _ -> ()

let free m r =
  let h = mine "free" m r in
  Tables.Address.remove h.regions r.mapping.va;
  let tables = if Function.released h.fn then None else h.tables in
  let confirmed =
    match tables with
    | None -> true
    | Some tables -> Page_table.unmap tables ~va:r.mapping.va r.mapping.size
  in
  (* Memory the GPU may still reach stays until the GPU is released. *)
  if confirmed then give_back m h tables r
  else h.unconfirmed <- r :: h.unconfirmed;
  if Option.is_none tables && h.unconfirmed <> [] then begin
    List.iter (give_back m h None) h.unconfirmed;
    h.unconfirmed <- []
  end;
  settle m h
