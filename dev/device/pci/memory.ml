(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type kind = Gpu | Bar | Host | Visible
type source = Allocated | Borrowed | Peer

type region = {
  mapping : Page_table.mapping;
  host : Window.t option;
  source : source;
}

(* The memory a GPU addresses, with what it allocated and mapped, by virtual
   address, so that a free or an unmap of memory it does not hold is refused. *)
type t = {
  fn : Function.t;
  tables : Page_table.t;
  bar : int;
  bar_size : int;
  peer : (int * int) list -> (int * int) list * Page_table.target;
  allocated : (int, region) Hashtbl.t;
  mapped : (int, region) Hashtbl.t;
}

(* Large allocations round to the GPU's large pages, so their tail maps with
   them. Small ones round to the GPU's page. *)
let large = 8 lsl 20
let large_page = 2 lsl 20
let page = 0x1000
let round_up n a = (n + a - 1) / a * a

let create ?peer fn tables ~bar =
  let base, bar_size =
    match Function.bar fn bar with
    | Some b -> b
    | None ->
        invalid_argf "Memory.create: %s has no BAR %d" (Function.bus fn) bar
  in
  let through_bar ranges =
    (List.map (fun (p, n) -> (p + base, n)) ranges, Page_table.System)
  in
  {
    fn;
    tables;
    bar;
    bar_size;
    peer = Option.value peer ~default:through_bar;
    allocated = Hashtbl.create 64;
    mapped = Hashtbl.create 16;
  }

let small_bar m = m.bar_size < Page_table.memory m.tables

(* Allocating *)

(* System memory at the same address for the process and the GPU. *)
let system m n =
  let page = Machine.page (Function.machine m.fn) in
  let n = round_up n page in
  let space = Page_table.space m.tables in
  match Space.alloc ~align:page space n with
  | None -> Ok None
  | Some va -> (
      match Function.alloc_dma m.fn ~va n with
      | exception e ->
          Space.free space va;
          raise e
      | Error why ->
          Space.free space va;
          Error why
      | Ok (view, runs) -> (
          match
            Page_table.map ~snooped:true ~uncached:true m.tables ~va System runs
          with
          | Some mapping ->
              Ok (Some { mapping; host = Some view; source = Allocated })
          | None ->
              Function.free_dma m.fn view;
              Space.free space va;
              Ok None))

(* The GPU's memory, one block the process reaches through the BAR when [bar],
   or [None] if the block lies beyond it. *)
let gpu m ~uncached ~bar n =
  let n = round_up n (if n >= large then large_page else page) in
  match Page_table.alloc ~uncached ~contiguous:bar m.tables n with
  | None -> Ok None
  | Some mapping when not bar ->
      Ok (Some { mapping; host = None; source = Allocated })
  | Some mapping -> (
      let pa = fst (List.hd mapping.pages) in
      if pa + mapping.size > m.bar_size then begin
        Page_table.free m.tables mapping;
        Ok None
      end
      else
        match
          Function.map ~combine:true ~off:pa ~length:mapping.size m.fn m.bar
        with
        | Ok host -> Ok (Some { mapping; host = Some host; source = Allocated })
        | Error why ->
            Page_table.free m.tables mapping;
            Error why)

let positive fn n =
  if n <= 0 then invalid_argf "Memory.%s: %d bytes, expected more than 0" fn n

let alloc ?(uncached = false) m kind n =
  positive "alloc" n;
  let mem =
    match kind with
    | Host -> system m n
    | Visible when small_bar m -> system m n
    | Gpu -> gpu m ~uncached ~bar:false n
    | Bar | Visible -> gpu m ~uncached ~bar:true n
  in
  (match mem with
  | Ok (Some mem) -> Hashtbl.replace m.allocated mem.mapping.va mem
  | Ok None | Error _ -> ());
  mem

(* Removes [mem] from [table], or refuses it. *)
let forget table fn mem =
  match Hashtbl.find_opt table mem.mapping.va with
  | Some mem' when mem' == mem -> Hashtbl.remove table mem.mapping.va
  | _ ->
      invalid_argf
        "Memory.%s: the memory at 0x%x is not this GPU's, or was given back" fn
        mem.mapping.va

(* Once the function is released the GPU may be another instance's: only system
   memory, pins and addresses are given back, and no entry is written. The
   addresses go back last: the vendor's GPUs share the space, and another GPU's
   system memory there would be mapped over this one's. *)
let free m mem =
  forget m.allocated "free" mem;
  let map = mem.mapping and live = not (Function.released m.fn) in
  if live then Page_table.unmap m.tables ~va:map.va map.size;
  (match (map.target, mem.host) with
  | System, Some view -> Function.free_dma m.fn view
  | _, Some view when live -> Function.unmap m.fn view
  | _ -> ());
  if map.target = Gpu then
    List.iter (fun (pa, _) -> Page_table.pfree m.tables pa) map.pages;
  Space.free (Page_table.space m.tables) map.va

(* Mapping *)

let no_room = "no GPU memory left for a page table"

let map_host m a n =
  positive "map_host" n;
  let page = Machine.page (Function.machine m.fn) in
  let n = round_up n page in
  let base = Page_table.base m.tables in
  if a mod page <> 0 then
    Error (strf "the memory at 0x%x does not start on a %d-byte page" a page)
  else if a < base || a + n > base + Page_table.span m.tables then
    Error
      (strf "the memory at 0x%x is outside the GPU's addresses [0x%x, 0x%x)" a
         base
         (base + Page_table.span m.tables))
  else
    match Function.pin m.fn a n with
    | Error _ as e -> e
    | Ok runs -> (
        match
          Page_table.map ~snooped:true ~uncached:true m.tables ~va:a System runs
        with
        | exception (Invalid_argument _ as e) ->
            Function.unpin m.fn a n;
            raise e
        | Some mapping ->
            let mem = { mapping; host = None; source = Borrowed } in
            Hashtbl.replace m.mapped a mem;
            Ok mem
        | None ->
            Function.unpin m.fn a n;
            Error no_room)

let map_peer m ~owner mem =
  (match Hashtbl.find_opt owner.allocated mem.mapping.va with
  | Some mem' when mem' == mem -> ()
  | _ ->
      invalid_argf "Memory.map_peer: the memory at 0x%x is not its owner's"
        mem.mapping.va);
  let map = mem.mapping in
  let iommu f = Function.addressing f = Machine.Iommu in
  if Function.machine m.fn != Function.machine owner.fn then
    Error "the GPUs are on different machines"
  else if iommu m.fn || iommu owner.fn then
    Error
      "a GPU behind an IOMMU reaches only its own memory and the memory the \
       process maps for it"
  else if map.target <> System && small_bar owner then
    Error
      "the other GPU's memory BAR is too small for peer access; enable \
       Resizable BAR in the firmware settings"
  else
    let pages, target =
      match map.target with
      | System -> (map.pages, Page_table.System)
      | Gpu | Peer _ -> owner.peer map.pages
    in
    match
      Page_table.map ~snooped:true ~uncached:map.uncached m.tables ~va:map.va
        target pages
    with
    | Some mapping ->
        let mem = { mapping; host = None; source = Peer } in
        Hashtbl.replace m.mapped map.va mem;
        Ok mem
    | None -> Error no_room

let unmap m mem =
  forget m.mapped "unmap" mem;
  let map = mem.mapping in
  if not (Function.released m.fn) then
    Page_table.unmap m.tables ~va:map.va map.size;
  if mem.source = Borrowed then Function.unpin m.fn map.va map.size
