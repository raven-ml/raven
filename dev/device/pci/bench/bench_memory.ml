(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The memory of a GPU, as a driver allocates and frees it.

   [alloc-free/KIND/SIZE] allocates [SIZE] bytes of [KIND] and frees them, on a
   GPU of 1 GiB of a machine whose operations return at once, with page tables
   in a buffer: what is timed is the bookkeeping. [gpu] maps physical blocks of
   the GPU's memory; [bar] maps one block and opens a window on it through the
   memory BAR; [visible], on a GPU whose BAR is 256 MiB, is system memory, which
   the machine gives as one run per 4 KiB page, apart, mapped an entry each. An
   allocation of each kind and size stays resident, so the rows measure the warm
   path, without table creation. *)

module Machine = Device_pci.Machine
module Function = Device_pci.Function
module Window = Device_pci.Window
module Space = Device_pci.Space
module Page_table = Device_pci.Page_table
module Memory = Device_pci.Memory

let kib = 1024
let mib = 1024 * kib
let gib = 1024 * mib
let sizes = [ ("4KiB", 4 * kib); ("16MiB", 16 * mib) ]

(* The machine *)

let page = 4 * kib
let bar_base = 1 lsl 39

(* The pages of system memory: one run each, every other page, so no two
   merge. *)
let system_base = 1 lsl 36

let runs =
  List.map
    (fun (_, n) ->
      (n, List.init (n / page) (fun i -> (system_base + (2 * i * page), page))))
    sizes

let fn ~bar_size =
  {
    Machine.addressing = Physical;
    config8 = (fun _ -> 0);
    config16 = (fun _ -> 0);
    config32 = (fun _ -> 0);
    set_config8 = (fun _ _ -> ());
    set_config16 = (fun _ _ -> ());
    set_config32 = (fun _ _ -> ());
    bar = (fun i -> if i = 0 then Some (bar_base, bar_size) else None);
    map = (fun _ off n -> Ok (Window.v (bar_base + off) n));
    unmap = ignore;
    interrupt = (fun _ -> false);
    reset = (fun () -> Ok ());
    alloc_dma =
      (fun ~contiguous:_ ~va n ->
        Ok (Window.v (Option.get va) n, List.assoc n runs));
    free_dma = ignore;
    pin = (fun a n -> Ok [ (a, n) ]);
    unpin = (fun _ _ -> ());
    release = ignore;
  }

let bus = "0000:01:00.0"

let take ~bar_size =
  let fn = fn ~bar_size in
  let machine =
    Machine.make ~name:"bench"
      {
        transport = Window.unsafe_transport 0;
        page;
        functions = (fun () -> []);
        take = (fun _ -> Ok fn);
        reserve = (fun ~base:_ _ -> Ok ());
      }
  in
  Result.get_ok (Function.take machine bus)

(* Page tables

   A GPU of 1 GiB whose page tables have a pool: 1 MiB of boot memory, then 2
   MiB of tables. The buffer holds both; the format's entries hold the address,
   bit 0 valid and bit 1 a table. *)

let memory = gib
let boot = mib
let tables_end = boot + (2 * mib)
let pages = [ (2 * mib, 2 * mib); (4 * kib, 4 * kib) ]

type entries = (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t

let entries () : entries =
  let b =
    Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout (tables_end / 8)
  in
  Bigarray.Array1.fill b 0L;
  b

let get (b : entries) ~level:_ ~table i =
  Bigarray.Array1.get b ((table lsr 3) + i)

let set (b : entries) ~level:_ ~table i e =
  Bigarray.Array1.set b ((table lsr 3) + i) e

(* Memory past the tables holds no entries and is not kept. *)
let zero (b : entries) pa n =
  if pa < tables_end then
    for i = pa lsr 3 to ((pa + n) lsr 3) - 1 do
      Bigarray.Array1.set b i 0L
    done

let encode ~level:_ ~table _ ~uncached:_ ~snooped:_ ~fragment:_ ~valid pa =
  Int64.of_int (pa lor (if valid then 1 else 0) lor if table then 2 else 0)

let valid e = Int64.logand e 1L = 1L
let leaf ~level e = level = 3 || Int64.logand e 2L = 0L
let address e = Int64.to_int e land lnot 0xfff
let large ~level = level >= 2
let levels = [ 12; 21; 30; 39 ]
let bits = 48
let space_base = 1 lsl 40

let page_table () =
  let b = entries () in
  let format =
    {
      Page_table.levels;
      bits;
      first = 0;
      get = get b;
      set = set b;
      encode;
      valid;
      leaf;
      address;
      large;
      zero = zero b;
      flush = ignore;
    }
  in
  let space = Space.create ~base:space_base space_base in
  let t = Page_table.create format space ~memory ~boot ~tables:Pool ~pages in
  Page_table.booted t;
  t

(* Memories *)

let kinds =
  [
    ("gpu", Memory.Gpu, gib);
    ("bar", Memory.Bar, gib);
    ("visible", Memory.Visible, 256 * mib);
  ]

let resident m kind =
  List.iter (fun (_, n) -> ignore (Memory.alloc m kind n : _ result))

let memory kind ~bar_size =
  let m = Memory.create (take ~bar_size) (page_table ()) ~bar:0 in
  resident m kind sizes;
  m

let alloc_free m kind n =
  match Memory.alloc m kind n with
  | Ok (Some mem) -> Memory.free m mem
  | Ok None -> failwith "alloc-free: no memory"
  | Error why -> failwith ("alloc-free: " ^ why)

let rows =
  let kind (name, kind, bar_size) =
    let m = lazy (memory kind ~bar_size) in
    let row (size_name, n) =
      let n = Thumper.black_box n in
      Thumper.bench_with_setup
        ~setup:(fun () -> Lazy.force m)
        size_name
        (fun m -> alloc_free m kind n)
    in
    Thumper.group name (List.map row sizes)
  in
  List.map kind kinds

let () =
  exit (Thumper.run "device_pci_memory" [ Thumper.group "alloc-free" rows ])
