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
   path, without table creation.

   [alloc-free/fragmented/bar/16MiB] is [bar] on a GPU whose memory holds 4096
   blocks of 4 KiB, every other one free, below the rest: its BAR reaches all of
   the memory, so the bound the BAR sets excludes nothing and costs what no
   bound does.

   [alloc-free/huge/two-gpus/64KiB] allocates 64 KiB of system memory for a
   function taken physically, in a 2 MiB block that holds memory already, and
   frees it, while a second function of the machine does the same from another
   domain: the lock that orders every allocation of the process, the zeroing of
   the reused bytes and the bookkeeping. The functions are a fixture tree's,
   whose memory lies in this machine's hugetlbfs at [/dev/hugepages]: the row
   exists where the process may write there and the system has free huge
   pages. *)

module Machine = Rig_pci.Machine
module Function = Rig_pci.Function
module Window = Rig_pci.Window
module Space = Rig_pci.Space
module Page_table = Rig_pci.Page_table
module Memory = Rig_pci.Memory
open Rig_pci_support

let sizes = [ ("4KiB", 4 * kib); ("16MiB", 16 * mib) ]

(* The machine *)

let page = 4 * kib
let bar_base = 1 lsl 39

(* The GPU's virtual addresses, which the machine reserves. *)
let space_base = 1 lsl 40

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
    map = (fun ~combine:_ _ off n -> Ok (Window.v (bar_base + off) n));
    unmap = ignore;
    interrupt = (fun _ -> false);
    reset = (fun () -> Ok ());
    alloc_dma =
      (fun ~contiguous:_ ~va n ->
        Ok (Some (Window.v (Option.get va) n, List.assoc n runs)));
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
  Result.get_ok (Machine.reserve machine ~base:space_base space_base);
  Result.get_ok (Function.take machine bus)

(* Page tables *)

let page_table () =
  Buffer_tables.create (Space.create ~base:space_base space_base)

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

let fragments = 4096

let fragmented () =
  let m = memory Memory.Bar ~bar_size:gib in
  List.init fragments (fun _ ->
      match Memory.alloc m Gpu (4 * kib) with
      | Ok (Some mem) -> mem
      | Ok None | Error _ -> failwith "fragmented: no memory")
  |> List.iteri (fun i mem -> if i mod 2 = 0 then Memory.free m mem);
  m

let group ?(sizes = sizes) name kind m =
  let row (size_name, n) =
    let n = Thumper.black_box n in
    Thumper.bench_with_setup
      ~setup:(fun () -> Lazy.force m)
      size_name
      (fun m -> alloc_free m kind n)
  in
  Thumper.group name (List.map row sizes)

(* Physically taken functions *)

let hugetlbfs = "/dev/hugepages"
let free_huge = "/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"

let huge_pages () =
  let mounted () =
    In_channel.with_open_text "/proc/mounts" In_channel.input_lines
    |> List.exists (fun l ->
        match String.split_on_char ' ' l with
        | _ :: dir :: "hugetlbfs" :: _ -> dir = hugetlbfs
        | _ -> false)
  in
  let free () =
    In_channel.with_open_text free_huge In_channel.input_all
    |> String.trim |> int_of_string
  in
  try
    mounted ()
    && free () >= 2
    &&
    (Unix.access hugetlbfs [ W_OK ];
     true)
  with Sys_error _ | Failure _ | Unix.Unix_error _ -> false

type pair = {
  fns : Function.t list;
  resident : (Function.t * Window.t) list;
  stop : bool Atomic.t;
  rival : unit Domain.t;
}

let small = 64 * kib

(* Two functions of a fixture tree whose memory lies in this machine's huge
   pages, each holding a page at the start of a 2 MiB block of its own; the
   second allocates and frees beside it from another domain until [stop]. *)
let pair () =
  let buses = [ "0000:03:00.0"; "0000:04:00.0" ] in
  let root = Tree.make (List.map Tree.gpu buses) in
  Unix.rmdir (Filename.concat root "dev/hugepages");
  Tree.link root "dev/hugepages" hugetlbfs;
  let m = Machine.at root in
  let page = Machine.page m and blocks = 2 * 2 * mib in
  Result.get_ok (Machine.reserve m ~base:free_base blocks);
  Tree.pagemap root ~page free_base
    (List.init (blocks / page) (fun i -> 0x10_0000 + i));
  let fns = List.map (fun bus -> Result.get_ok (Function.take m bus)) buses in
  let take f va n =
    match Function.alloc_dma ~va f n with
    | Ok (Some (w, _)) -> w
    | Ok None -> failwith "huge: no free huge page"
    | Error why -> failwith ("huge: " ^ why)
  in
  let resident =
    List.mapi (fun i f -> (f, take f (free_base + (i * 2 * mib)) page)) fns
  in
  let stop = Atomic.make false and rival = List.nth fns 1 in
  let va = free_base + (2 * mib) + small in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          Function.free_dma rival (take rival va small)
        done)
  in
  { fns; resident; stop; rival }

let release p =
  Atomic.set p.stop true;
  Domain.join p.rival;
  List.iter (fun (f, w) -> Function.free_dma f w) p.resident;
  List.iter Function.release p.fns

let alloc_free_huge p =
  let f = List.hd p.fns in
  match Function.alloc_dma ~va:(free_base + small) f small with
  | Ok (Some (w, _)) -> Function.free_dma f w
  | Ok None -> failwith "huge: no free huge page"
  | Error why -> failwith ("huge: " ^ why)

let huge =
  if not (huge_pages ()) then []
  else
    [
      Thumper.group "huge"
        [
          Thumper.group "two-gpus"
            [
              Thumper.bench_with_setup ~setup:pair ~teardown:release "64KiB"
                alloc_free_huge;
            ];
        ];
    ]

let rows =
  List.map
    (fun (name, kind, bar_size) ->
      group name kind (lazy (memory kind ~bar_size)))
    kinds
  @ [
      Thumper.group "fragmented"
        [
          group
            ~sizes:[ ("16MiB", 16 * mib) ]
            "bar" Memory.Bar
            (lazy (fragmented ()));
        ];
    ]
  @ huge

let () = exit (Thumper.run "rig_pci_memory" [ Thumper.group "alloc-free" rows ])
