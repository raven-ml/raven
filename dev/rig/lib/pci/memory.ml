(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type kind = Gpu | Bar | Host
type link = { fabric : int64; node : int }

(* The memory a GPU addresses, with every region it holds by GPU address, so
   that a free or a map of memory it does not hold is refused. A region of
   system memory whose unmap the GPU did not confirm waits in [unconfirmed]
   until the function's release. *)
type t = {
  fn : Function.t;
  tables : Page_table.t;
  bar : int;
  bar_base : int;
  bar_size : int;
  link : link option;
  regions : region Tables.Address.t;
  mutable unconfirmed : region list;
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

let create ?link fn tables ~bar =
  let bar_base, bar_size =
    match Function.bar fn bar with
    | Some b -> b
    | None ->
        invalid_argf "Memory.create: %s has no BAR %d" (Function.bus fn) bar
  in
  {
    fn;
    tables;
    bar;
    bar_base;
    bar_size;
    link;
    regions = Tables.Address.create 64;
    unconfirmed = [];
  }

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

(* Whether the process reaches all of the GPU's memory through its BAR. *)
let small_bar m = m.bar_size < Page_table.memory m.tables
let iommu m = Function.addressing m.fn = Machine.Iommu
let machine m = Function.machine m.fn
let space m = Page_table.space m.tables

let linked m o =
  match (m.link, o.link) with
  | Some l, Some l' -> Int64.equal l.fabric l'.fabric
  | _ -> false

(* A link's entries name the peer's own memory, which crosses no IOMMU; a BAR's
   name its bus address, which the IOMMU translates. *)
let reaches m o =
  m != o
  && machine m == machine o
  && space m == space o
  && (linked m o || not (iommu m || iommu o || small_bar o))

(* Errors *)

let beyond m runs =
  let bits = Page_table.pa_bits m.tables in
  match List.find_opt (fun (pa, n) -> pa > (1 lsl bits) - n) runs with
  | None -> None
  | Some (pa, _) ->
      Some
        (strf
           "the machine gave memory at 0x%x, past the %d bits of address %s's \
            page tables hold"
           pa bits (Function.bus m.fn))

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
let system m n =
  let page = Machine.page (machine m) in
  let n = round_up n page in
  match Space.alloc ~align:page (space m) n with
  | None -> Ok None
  | Some va -> (
      let give_back view =
        Option.iter (Function.free_dma m.fn) view;
        Space.free (space m) va
      in
      match Function.alloc_dma m.fn ~va n with
      | Error _ as e ->
          give_back None;
          e
      | Ok None ->
          give_back None;
          Ok None
      | Ok (Some (view, runs)) -> (
          match beyond m runs with
          | Some why ->
              give_back (Some view);
              Error why
          | None -> (
              match
                Page_table.map ~snooped:true ~uncached:true m.tables ~va System
                  runs
              with
              | Some mapping ->
                  Ok (Some { (allocated m mapping) with host = Some view })
              | None ->
                  give_back (Some view);
                  Ok None)))

(* The GPU's memory, one block the process reaches through the BAR when [bar],
   placed under the BAR's end before anything is written, or [None] if no block
   fits there. *)
let gpu m ~uncached ~bar n =
  let n = round_up n (if n >= large then large_page else page) in
  let below = if bar then Some m.bar_size else None in
  match Page_table.alloc ~uncached ~contiguous:bar ?below m.tables n with
  | None -> Ok None
  | Some mapping when not bar -> Ok (Some (allocated m mapping))
  | Some mapping -> (
      let pa = fst (List.hd mapping.pages) in
      match Function.map ~off:pa ~length:mapping.size m.fn m.bar with
      | Ok host -> Ok (Some { (allocated m mapping) with host = Some host })
      | Error why ->
          (* Memory the GPU may still reach stays: the GPU is lost then. *)
          ignore (Page_table.free m.tables mapping : bool);
          Error why)

let positive fn n =
  if n <= 0 then invalid_argf "Memory.%s: %d bytes, expected more than 0" fn n

(* A released GPU may be another instance's: no request writes its tables. *)
let live fn m =
  if Function.released m.fn then
    invalid_argf "Memory.%s: %s is released" fn (Function.bus m.fn)

let keep m r = Tables.Address.replace m.regions r.mapping.va r

let alloc ?(uncached = false) m kind n =
  live "alloc" m;
  positive "alloc" n;
  let r =
    match kind with
    | Host -> system m n
    | Gpu -> gpu m ~uncached ~bar:false n
    | Bar -> gpu m ~uncached ~bar:true n
  in
  (match r with Ok (Some r) -> keep m r | Ok None | Error _ -> ());
  r

(* Mapping *)

(* Pinned pages map at [va], with what [r] needs to give them back. *)
let map_pinned m ~va ~source ~host a n =
  match Function.pin m.fn a n with
  | Error _ as e -> e
  | Ok runs -> (
      let unpin () = Function.unpin m.fn a n in
      match beyond m runs with
      | Some why ->
          unpin ();
          Error why
      | None -> (
          match
            Page_table.map ~snooped:true ~uncached:true m.tables ~va System runs
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
  live "map_host" m;
  positive "map_host" n;
  let page = Machine.page (machine m) in
  let first = a / page * page in
  let bytes = round_up (a + n) page - first in
  match Space.alloc ~align:page (space m) bytes with
  | None -> Ok None
  | Some va -> (
      let host = Some (Window.v a n) in
      match map_pinned m ~va ~source:(Borrowed a) ~host first bytes with
      | (Error _ | Ok None) as r ->
          Space.free (space m) va;
          r
      | Ok (Some r) ->
          let r = { r with off = a - first } in
          keep m r;
          Ok (Some r))

let rec origin r = match r.source with Peer o -> origin o | _ -> r

let mine fn m r =
  match Tables.Address.find_opt m.regions r.mapping.va with
  | Some r' when r' == r -> ()
  | _ ->
      invalid_argf
        "Memory.%s: the memory at 0x%x is not %s's, or was given back" fn
        r.mapping.va (Function.bus m.fn)

(* Misuse is refused before anything is pinned or written: a region freed, its
   origin freed, or addresses the GPU maps already. *)
let map_peer m r =
  live "map_peer" m;
  mine "map_peer" r.owner r;
  let r = origin r in
  let o = r.owner in
  live "map_peer" o;
  mine "map_peer" o r;
  if o == m then
    invalid_argf "Memory.map_peer: the memory at 0x%x is %s's own" r.mapping.va
      (Function.bus m.fn);
  if Tables.Address.mem m.regions r.mapping.va then
    invalid_argf "Memory.map_peer: %s maps the memory at 0x%x already"
      (Function.bus m.fn) r.mapping.va;
  let map = r.mapping in
  let mapped = function
    | Ok (Some p) ->
        let p = { p with off = r.off } in
        keep m p;
        Ok (Some p)
    | (Ok None | Error _) as e -> e
  in
  if machine m != machine o then Error "the GPUs are on different machines"
  else if space m != space o then
    Error
      (strf
         "%s and %s address different spaces: a peer's memory maps at its \
          owner's address, which is that memory only in a space both share"
         (Function.bus m.fn) (Function.bus o.fn))
  else
    match map.target with
    | System ->
        (* Borrowed memory is pinned where the process holds it, allocated
           memory at its own address. *)
        let a, n = Option.value r.pinned ~default:(map.va, map.size) in
        mapped (map_pinned m ~va:map.va ~source:(Peer r) ~host:None a n)
    | (Gpu | Peer _) when not (reaches m o) ->
        Error
          (strf
             "%s does not reach %s's memory: it needs a link of one fabric, or \
              both GPUs off an IOMMU and a memory BAR as large as the memory, \
              which Resizable BAR in the firmware settings gives"
             (Function.bus m.fn) (Function.bus o.fn))
    | Gpu | Peer _ -> (
        let target, pages =
          match (m.link, o.link) with
          | Some _, Some l when linked m o -> (Page_table.Peer l.node, map.pages)
          | _ ->
              ( Page_table.System,
                List.map (fun (pa, n) -> (pa + o.bar_base, n)) map.pages )
        in
        match beyond m pages with
        | Some why -> Error why
        | None -> (
            match
              Page_table.map ~snooped:true ~uncached:map.uncached m.tables
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

(* Gives back what [r] holds but its entries: once the function is released the
   GPU may be another instance's, so that is all a free does then. The addresses
   go back last: the vendor's GPUs share the space, and another GPU's system
   memory there would be mapped over this one's. *)
let give_back m r =
  let map = r.mapping and live = not (Function.released m.fn) in
  (match (r.source, map.target, r.host) with
  | Allocated, System, Some view -> Function.free_dma m.fn view
  | Allocated, _, Some view when live -> Function.unmap m.fn view
  | _ -> ());
  Option.iter (fun (a, n) -> Function.unpin m.fn a n) r.pinned;
  (match (r.source, map.target) with
  | Allocated, Gpu ->
      List.iter (fun (pa, _) -> Page_table.pfree m.tables pa) map.pages
  | _ -> ());
  match r.source with
  | Allocated | Borrowed _ -> Space.free (space m) map.va
  | Peer _ -> ()

let free m r =
  mine "free" m r;
  Tables.Address.remove m.regions r.mapping.va;
  let released = Function.released m.fn in
  let confirmed =
    released || Page_table.unmap m.tables ~va:r.mapping.va r.mapping.size
  in
  (* Memory the GPU may still reach stays until the GPU is released. *)
  if confirmed then give_back m r else m.unconfirmed <- r :: m.unconfirmed;
  if released && m.unconfirmed <> [] then begin
    List.iter (give_back m) m.unconfirmed;
    m.unconfirmed <- []
  end
