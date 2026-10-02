(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  pci : Pci.t;
  tables : Page_table.t;
  bar : int;
  peer : (int * int) list -> (int * int) list * Page_table.space;
}

type source = Allocated | Pinned | Peer

type memory = {
  mapping : Page_table.mapping;
  host : Mmio.t option;
  source : source;
}

let round_up n a = (n + a - 1) / a * a

let create ?peer pci tables ~bar =
  let peer =
    match peer with
    | Some f -> f
    | None ->
        let start = fst (Pci.bar pci bar) in
        fun ranges ->
          (List.map (fun (p, n) -> (p + start, n)) ranges, Page_table.Sys)
  in
  { pci; tables; bar; peer }

let small_bar m = snd (Pci.bar m.pci m.bar) = 256 lsl 20

(* System memory at the same address for the process and the GPU, one entry per
   page of the process. *)
let sysmem m n =
  let page = Pci.page m.pci in
  let n = round_up n page in
  let space = Page_table.space m.tables in
  match Page_table.Space.alloc ~align:page space n with
  | None -> None
  | Some va -> (
      match Pci.alloc_sysmem m.pci ~va n with
      | exception e ->
          Page_table.Space.free space va;
          raise e
      | view, pages -> (
          let pages = List.map (fun p -> (p, page)) pages in
          match
            Page_table.map ~snooped:true ~uncached:true m.tables ~va
              Page_table.Sys pages
          with
          | mapping -> Some { mapping; host = Some view; source = Allocated }
          | exception e ->
              Pci.free_sysmem m.pci view;
              Page_table.Space.free space va;
              raise e))

let alloc ?(host = false) ?(uncached = false) ?(cpu_access = false)
    ?(devmem = false) ?(zero = false) m n =
  if host || (cpu_access && small_bar m && not devmem) then sysmem m n
  else
    let n = round_up n (if n >= 8 lsl 20 then 2 lsl 20 else 0x1000) in
    Option.map
      (fun (mapping : Page_table.mapping) ->
        let host =
          if cpu_access then
            Some
              (Pci.map_bar
                 ~offset:(fst (List.hd mapping.pages))
                 ~length:mapping.size m.pci m.bar)
          else None
        in
        { mapping; host; source = Allocated })
      (Page_table.alloc ~uncached ~contiguous:cpu_access ~zero m.tables n)

let free m mem =
  Page_table.free m.tables mem.mapping;
  match (mem.mapping.space, mem.host) with
  | Page_table.Sys, Some view -> Pci.free_sysmem m.pci view
  | _, Some view -> Pci.unmap_bar view
  | _, None -> ()

let map_host m a n =
  let page = Pci.page m.pci in
  let lo = Nativeint.to_int a and n = round_up n page in
  let base = Page_table.base m.tables in
  if lo mod page <> 0 then
    Error (Printf.sprintf "the host memory at 0x%x does not start on a page" lo)
  else if lo < base || lo + n > base + Page_table.span m.tables then
    Error
      (Printf.sprintf "the host memory at 0x%x is outside the GPU's addresses"
         lo)
  else
    match Pci.pin m.pci a n with
    | exception Failure why -> Error why
    | pages -> (
        let pages = List.map (fun p -> (p, page)) pages in
        match
          Page_table.map ~snooped:true ~uncached:true m.tables ~va:lo
            Page_table.Sys pages
        with
        | mapping -> Ok { mapping; host = None; source = Pinned }
        | exception e ->
            Pci.unpin m.pci a n;
            raise e)

let map_peer m m' mem =
  let map = mem.mapping in
  if Pci.addressing m.pci = Pci.Iommu || Pci.addressing m'.pci = Pci.Iommu then
    Error
      "a GPU behind an IOMMU reaches only its own memory and the memory the \
       process maps for it"
  else if map.space <> Page_table.Sys && small_bar m' then
    Error "the other GPU's memory BAR is too small for peer access"
  else
    let pages, space =
      match map.space with
      | Page_table.Sys -> (map.pages, Page_table.Sys)
      | Phys | Peer -> m'.peer map.pages
    in
    let mapping =
      Page_table.map ~snooped:true ~uncached:map.uncached m.tables ~va:map.va
        space pages
    in
    Ok { mapping; host = None; source = Peer }

let unmap m mem =
  Page_table.unmap m.tables ~va:mem.mapping.va mem.mapping.size;
  if mem.source = Pinned then
    Pci.unpin m.pci (Nativeint.of_int mem.mapping.va) mem.mapping.size
