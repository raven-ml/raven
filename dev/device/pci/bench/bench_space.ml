(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Virtual address spaces and the page tables that map them, as every GPU
   allocation over PCI uses them.

   [space/alloc-free] allocates a range and frees it in a space holding 16 or
   4096 free ranges of twice each size, the class the allocation searches: an
   allocation and a free take a time bounded independently of what the space
   holds, so the two columns match. [space/churn] allocates and frees ranges of
   mixed sizes in a fresh space. [page-table/map-unmap] maps physical memory
   next to a resident page and unmaps it, or 16 MiB of one-page runs, or a page
   where no table is, making three tables and freeing them;
   [page-table/alloc-free] is the whole allocation: addresses, physical memory,
   mapping. The page tables live in a buffer. *)

module Space = Device_pci.Space
module Page_table = Device_pci.Page_table
open Device_pci_support

(* Spaces *)

let space_base = 1 lsl 44
let space_length = 1 lsl 44
let sizes = [ ("4KiB", 4 * kib); ("2MiB", 2 * mib); ("256MiB", 256 * mib) ]
let fragments = [ 16; 4096 ]

(* The churn: [steps] steps over [slots] slots, each freeing its slot's range if
   it holds one and allocating one of the step's size otherwise. *)
let slots = 64
let steps = 1024

let churn_sizes =
  [| 4 * kib; 16 * kib; 64 * kib; 256 * kib; mib; 2 * mib; 8 * mib; 64 * mib |]

let churn_slot, churn_size =
  let x = ref 0x2545_f491 in
  let next () =
    x := (!x * 1103515245) + 12345;
    (!x lsr 16) land 0x7fff
  in
  let slot = Array.make steps 0 and size = Array.make steps 0 in
  for i = 0 to steps - 1 do
    slot.(i) <- next () mod slots;
    size.(i) <- churn_sizes.(next () mod Array.length churn_sizes)
  done;
  (slot, size)

(* A space with [n] free ranges of twice each size, apart: [2 n] ranges
   allocated in a row, then every other one freed. *)
let fragmented n =
  let s = Space.create ~base:space_base space_length in
  let fragment (_, size) =
    let ranges =
      Array.init (2 * n) (fun _ -> Option.get (Space.alloc s (2 * size)))
    in
    Array.iteri (fun i a -> if i land 1 = 0 then Space.free s a) ranges
  in
  List.iter fragment sizes;
  s

let alloc_free s n =
  match Space.alloc s n with
  | Some a -> Space.free s a
  | None -> failwith "alloc-free: the space is full"

let churn s live =
  for i = 0 to steps - 1 do
    let slot = churn_slot.(i) in
    if live.(slot) >= 0 then begin
      Space.free s live.(slot);
      live.(slot) <- -1
    end
    else live.(slot) <- Option.get (Space.alloc s churn_size.(i))
  done;
  for slot = 0 to slots - 1 do
    if live.(slot) >= 0 then begin
      Space.free s live.(slot);
      live.(slot) <- -1
    end
  done

let space_rows =
  let spaces = List.map (fun n -> (n, lazy (fragmented n))) fragments in
  let alloc_free (name, size) =
    let size = Thumper.black_box size in
    let row (n, s) =
      Thumper.bench_with_setup
        ~setup:(fun () -> Lazy.force s)
        (Printf.sprintf "%d-free" n)
        (fun s -> alloc_free s size)
    in
    Thumper.group name (List.map row spaces)
  in
  [
    Thumper.group "alloc-free" (List.map alloc_free sizes);
    Thumper.bench_with_setup
      ~setup:(fun () ->
        (Space.create ~base:space_base space_length, Array.make slots (-1)))
      "churn"
      (fun (s, live) -> churn s live);
  ]

(* Page tables *)

(* Mappings next to a resident page at [va], of physical memory at [pa]. *)
let va = (1 lsl 40) + gib
let pa = 128 * mib

let maps =
  [
    ("4KiB", 4 * kib, 4 * kib);
    ("64KiB", 64 * kib, 64 * kib);
    ("2MiB", 2 * mib, 2 * mib);
  ]

let page_table () =
  let t = Buffer_tables.create (Space.create ~base:(1 lsl 40) (1 lsl 40)) in
  ignore (Option.get (Page_table.map t ~va Gpu [ (pa, 4 * kib) ]));
  ignore (Option.get (Page_table.alloc t (4 * kib)));
  t

let map_unmap t va size pa =
  match Page_table.map t ~va Gpu [ (pa, size) ] with
  | Some _ -> Page_table.unmap t ~va size
  | None -> failwith "map-unmap: no room for a table"

let alloc_free t n =
  match Page_table.alloc t n with
  | Some m -> Page_table.free t m
  | None -> failwith "alloc-free: no memory"

(* System memory taken page by page: 16 MiB of one-page runs, every other
   physical page, so that no two join and each maps with its own entry. *)
let runs_size = 16 * mib

let page_runs =
  List.init (runs_size / (4 * kib)) (fun i -> (pa + (2 * i * 4 * kib), 4 * kib))

let map_unmap_runs t va =
  match Page_table.map t ~va System page_runs with
  | Some _ -> Page_table.unmap t ~va runs_size
  | None -> failwith "map-unmap: no room for a table"

(* Each mapping starts at its own size past the resident page. *)
let page_table_rows =
  let t = lazy (page_table ()) in
  let setup () = Lazy.force t in
  let map_row (name, size, at) =
    let va = Thumper.black_box (va + at) and size = Thumper.black_box size in
    Thumper.bench_with_setup ~setup name (fun t -> map_unmap t va size pa)
  in
  let alloc_row (name, size, _) =
    let size = Thumper.black_box size in
    Thumper.bench_with_setup ~setup name (fun t -> alloc_free t size)
  in
  let runs_row =
    let va = Thumper.black_box (va + runs_size) in
    Thumper.bench_with_setup ~setup "16MiB-pages" (fun t -> map_unmap_runs t va)
  in
  (* 512 GiB on, under another entry of the root. *)
  let tables_row =
    let va = Thumper.black_box (va + (512 * gib)) in
    Thumper.bench_with_setup ~setup "4KiB-tables" (fun t ->
        map_unmap t va (4 * kib) pa)
  in
  [
    Thumper.group "map-unmap" (List.map map_row maps @ [ runs_row; tables_row ]);
    Thumper.group "alloc-free" (List.map alloc_row maps);
  ]

let () =
  exit
    (Thumper.run "device_pci_space"
       [
         Thumper.group "space" space_rows;
         Thumper.group "page-table" page_table_rows;
       ])
