(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kind = Gpu | Bar | Host | Visible
type source = Allocated | Borrowed | Peer

type memory = {
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
  allocated : (int, memory) Hashtbl.t;
  mapped : (int, memory) Hashtbl.t;
}

(* A memory BAR of this size reaches only part of a GPU's memory. *)
let small = 256 lsl 20

(* Large allocations round to large pages, so their tail maps with them. *)
let large = 8 lsl 20
let large_page = 2 lsl 20
let page = 0x1000
let round_up n a = (n + a - 1) / a * a

let create ?peer fn tables ~bar =
  let base, bar_size =
    match Function.bar fn bar with
    | Some b -> b
    | None ->
        invalid_arg
          (Printf.sprintf "Memory.create: %s has no BAR %d" (Function.bus fn)
             bar)
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

let small_bar m = m.bar_size = small

(* Allocating *)

(* System memory at the same address for the process and the GPU. *)
let system m n =
  let page = Machine.page (Function.machine m.fn) in
  let n = round_up n page in
  let space = Page_table.space m.tables in
  match Space.alloc ~align:page space n with
  | None -> None
  | Some va -> (
      let view, runs =
        try Function.alloc_dma m.fn ~va n
        with e ->
          Space.free space va;
          raise e
      in
      match
        Page_table.map ~snooped:true ~uncached:true m.tables ~va System runs
      with
      | Some mapping -> Some { mapping; host = Some view; source = Allocated }
      | None ->
          Function.free_dma m.fn view;
          Space.free space va;
          None)

(* The GPU's memory, one block the process reaches through the BAR when [bar],
   or [None] if the block lies beyond it. *)
let gpu m ~uncached ~bar n =
  let n = round_up n (if n >= large then large_page else page) in
  match Page_table.alloc ~uncached ~contiguous:bar m.tables n with
  | None -> None
  | Some mapping when not bar ->
      Some { mapping; host = None; source = Allocated }
  | Some mapping ->
      let pa = fst (List.hd mapping.pages) in
      if pa + mapping.size > m.bar_size then begin
        Page_table.free m.tables mapping;
        None
      end
      else
        let host = Function.map ~off:pa ~length:mapping.size m.fn m.bar in
        Some { mapping; host = Some host; source = Allocated }

let alloc ?(uncached = false) m kind n =
  let mem =
    match kind with
    | Host -> system m n
    | Visible when small_bar m -> system m n
    | Gpu -> gpu m ~uncached ~bar:false n
    | Bar | Visible -> gpu m ~uncached ~bar:true n
  in
  Option.iter (fun mem -> Hashtbl.replace m.allocated mem.mapping.va mem) mem;
  mem

(* Removes [mem] from [table], or refuses it. *)
let forget table fn mem =
  match Hashtbl.find_opt table mem.mapping.va with
  | Some mem' when mem' == mem -> Hashtbl.remove table mem.mapping.va
  | _ -> invalid_arg (Printf.sprintf "Memory.%s: memory it does not hold" fn)

(* Once the function is released the GPU may be another instance's: only system
   memory, pins and addresses are given back, and no entry is written. *)
let free m mem =
  forget m.allocated "free" mem;
  let map = mem.mapping in
  if Function.released m.fn then begin
    Space.free (Page_table.space m.tables) map.va;
    if map.target = Gpu then
      List.iter (fun (pa, _) -> Page_table.pfree m.tables pa) map.pages
  end
  else Page_table.free m.tables map;
  match (map.target, mem.host) with
  | System, Some view -> Function.free_dma m.fn view
  | _, Some view when not (Function.released m.fn) -> Function.unmap m.fn view
  | _ -> ()

(* Mapping *)

let no_room = "no GPU memory left for a page table"

let map_host m a n =
  let page = Machine.page (Function.machine m.fn) in
  let n = round_up n page in
  let base = Page_table.base m.tables in
  if a mod page <> 0 then
    Error (Printf.sprintf "the memory at 0x%x does not start on a page" a)
  else if a < base || a + n > base + Page_table.span m.tables then
    Error (Printf.sprintf "the memory at 0x%x is outside the GPU's addresses" a)
  else
    match Function.pin m.fn a n with
    | exception Failure why -> Error why
    | runs -> (
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
  | _ -> invalid_arg "Memory.map_peer: memory its owner did not allocate");
  let map = mem.mapping in
  let iommu f = Function.addressing f = Function.Iommu in
  if Function.machine m.fn != Function.machine owner.fn then
    Error "the GPUs are on different machines"
  else if iommu m.fn || iommu owner.fn then
    Error
      "a GPU behind an IOMMU reaches only its own memory and the memory the \
       process maps for it"
  else if map.target <> System && small_bar owner then
    Error "the other GPU's memory BAR is too small for peer access"
  else
    let pages, target =
      match map.target with
      | System -> (map.pages, Page_table.System)
      | Gpu | Peer -> owner.peer map.pages
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
