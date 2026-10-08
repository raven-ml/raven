(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A GPU's memory over a fake machine: its functions record the system memory
   and pins they give, and the GPU's page tables live in a table that counts
   every access, so a test sees what reaches the GPU after it is given back. *)

open Windtrap
open Rig_pci
open Rig_pci_support

let strf = Printf.sprintf
let ranges = list (pair hex hex)

(* The fake machine's page, larger than the GPU's 4 KiB. *)
let page = 16 * kib

(* Addresses: the GPUs' BARs on the bus, the tables' virtual addresses from
   [tables_base], and the space the GPUs share inside them. *)
let bar_base = 1 lsl 36
let bar_slot = 512 * mib
let tables_base = 1 lsl 40
let space_base = 1 lsl 41
let space_length = 1 lsl 30

(* The bytes behind the BARs and the space's system memory, should the library
   access them. Allocated once, mapped lazily by calloc. *)
let bars = lazy (Window.unsafe_transport (far bar_base (4 * bar_slot)))
let system = lazy (Window.unsafe_transport (far space_base space_length))
let link = lazy (Window.unsafe_transport (far 0 0))

(* The fake function *)

type fake = {
  addressing : Machine.addressing;
  bar : int * int;  (** BAR 0's bus address and size. *)
  mutable dma : (int * (int * int) list) list;
      (** Windows [alloc_dma] gave and [free_dma] did not take back: address and
          runs. *)
  mutable pins : (int * int) list;  (** Pins held. *)
  mutable refuse : string option;
      (** [alloc_dma], [pin] and [map] fail with it. *)
  mutable combined : bool list;  (** Each [map]'s [combine], newest first. *)
  mutable freeing : unit -> unit;  (** Called as [free_dma] starts. *)
}

(* Physical pages with gaps between them, so no two runs merge. *)
let next_page = ref 0

let fresh () =
  incr next_page;
  (1 lsl 34) + (!next_page * 2 * page)

let runs k n =
  match k.addressing with
  | Iommu -> [ (fresh (), n) ]
  | Physical ->
      List.init
        ((n + page - 1) / page)
        (fun i -> (fresh (), min page (n - (i * page))))

(* [Error why] if [k] refuses, else [Ok (f ())]. *)
let unless_refused k f =
  match k.refuse with Some why -> Error why | None -> Ok (f ())

let rec remove x = function
  | [] -> []
  | y :: l -> if x = y then l else y :: remove x l

let ops k =
  {
    Machine.addressing = k.addressing;
    config8 = (fun _ -> 0);
    config16 = (fun _ -> 0);
    config32 = (fun _ -> 0);
    set_config8 = (fun _ _ -> ());
    set_config16 = (fun _ _ -> ());
    set_config32 = (fun _ _ -> ());
    bar = (fun i -> if i = 0 then Some k.bar else None);
    map =
      (fun ~combine _ off n ->
        k.combined <- combine :: k.combined;
        unless_refused k @@ fun () ->
        Window.through (Lazy.force bars) (fst k.bar + off) n);
    unmap = ignore;
    interrupt = (fun _ -> false);
    reset = (fun () -> Ok ());
    alloc_dma =
      (fun ~contiguous:_ ~va n ->
        unless_refused k @@ fun () ->
        let va =
          match va with
          | Some va -> va
          | None ->
              fail "system memory for the GPU is allocated without an address"
        in
        let n = round_up n page in
        let r = runs k n in
        k.dma <- (va, r) :: k.dma;
        (Window.through (Lazy.force system) va n, r));
    free_dma =
      (fun w ->
        k.freeing ();
        k.dma <- List.filter (fun (a, _) -> a <> Window.address w) k.dma);
    pin =
      (fun a n ->
        unless_refused k @@ fun () ->
        k.pins <- (a, n) :: k.pins;
        runs k n);
    unpin = (fun a n -> k.pins <- remove (a, n) k.pins);
    release = ignore;
  }

(* A machine whose functions are fakes, made as [config] says when taken. *)
type machine = {
  machine : Machine.t;
  fakes : (string, fake) Hashtbl.t;
  config : (Machine.addressing * int) ref;
      (** How the next function taken reaches system memory, and its BAR's size.
      *)
}

let machine () =
  let fakes = Hashtbl.create 4 and config = ref (Machine.Physical, 0) in
  let take bus =
    let addressing, size = !config in
    let slot = Hashtbl.length fakes mod 4 in
    let k =
      {
        addressing;
        bar = (bar_base + (slot * bar_slot), size);
        dma = [];
        pins = [];
        refuse = None;
        combined = [];
        freeing = ignore;
      }
    in
    Hashtbl.replace fakes bus k;
    Ok (ops k)
  in
  let ops =
    {
      Machine.transport = Lazy.force link;
      page;
      functions = (fun () -> []);
      take;
      reserve = (fun ~base:_ _ -> Ok ());
    }
  in
  let machine = Machine.make ~name:"fake" ops in
  require_ok (Machine.reserve machine ~base:space_base space_length);
  { machine; fakes; config }

(* Bus addresses no other GPU of the run holds. *)
let buses = ref 0

let bus () =
  incr buses;
  Machine.address ~domain:0 ~bus:(!buses mod 256)
    ~device:(!buses / 256 mod 32)
    ~fn:0

(* The GPU's memory *)

type leaf = {
  va : int;
  pa : int;
  size : int;
  target : Page_table.target;
  uncached : bool;
  snooped : bool;
}

let pp_leaf ppf l =
  Format.fprintf ppf "{va 0x%x; pa 0x%x; %d bytes}" l.va l.pa l.size

let leaf = Testable.make ~pp:pp_leaf ~equal:( = )

(* The pages the tables map, by virtual address, read without counting. *)
let leaves g t =
  List.map
    (fun (e : Tables.entry) ->
      {
        va = e.va;
        pa = e.pa;
        size = 1 lsl Tables.shifts.(e.level);
        target = e.target;
        uncached = e.uncached;
        snooped = e.snooped;
      })
    (fst (Tables.walk g t))

(* Adjacent ranges merged, so that two spellings of the same bytes compare. *)
let merge l =
  List.fold_right
    (fun (a, n) acc ->
      match acc with
      | (b, m) :: rest when a + n = b -> (a, n + m) :: rest
      | _ -> (a, n) :: acc)
    l []

(* The ranges the tables map from [va] on for [n] bytes, as (physical address,
   bytes), merged. *)
let mapped g t ~va n =
  leaves g t
  |> List.filter (fun l -> l.va >= va && l.va < va + n)
  |> List.map (fun l -> (l.pa, l.size))
  |> merge

let gpu_memory = 64 * mib

let large_pages =
  [ (2 * mib, 2 * mib); (64 * kib, 64 * kib); (4 * kib, 4 * kib) ]

type gpu = {
  size : int;  (** The GPU's memory in bytes. *)
  fn : Function.t;
  fake : fake;
  g : Tables.memory;
  tables : Page_table.t;
  memory : Memory.t;
}

let gpu ?(addressing = Machine.Physical) ?(memory = gpu_memory) ?bar
    ?(tables = Page_table.Pool) ?(format = Fun.id) ?peer ?space ?machine:m () =
  let m = match m with Some m -> m | None -> machine () in
  let bar = Option.value bar ~default:memory in
  m.config := (addressing, bar);
  let b = bus () in
  let fn =
    match Function.take m.machine b with
    | Ok fn -> fn
    | Error why -> failf "taking a fake function: %s" why
  in
  let space =
    match space with
    | Some s -> s
    | None -> Space.create ~base:space_base space_length
  in
  let g = Tables.memory () in
  let tables =
    Page_table.create ~base:tables_base
      (format (Tables.format g))
      space ~memory ~boot:mib ~tables ~pages:large_pages
  in
  Page_table.booted tables;
  let size = memory in
  let memory = Memory.create ?peer fn tables ~bar:0 in
  { size; fn; fake = Hashtbl.find m.fakes b; g; tables; memory }

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

let capacity x =
  let t = x.tables and s = Page_table.space x.tables in
  ( drain (fun () -> Page_table.palloc ~zero:false t 4096) (Page_table.pfree t),
    drain (fun () -> Space.alloc s page) (Space.free s) )

(* [Memory.alloc], which the fake machine never refuses unless asked. *)
let alloc_opt ?uncached m kind n = require_ok (Memory.alloc ?uncached m kind n)

let alloc ?uncached x kind n =
  match alloc_opt ?uncached x.memory kind n with
  | Some mem -> mem
  | None -> fail "the GPU's memory has room"

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | Memory.Gpu -> "Gpu"
    | Bar -> "Bar"
    | Host -> "Host"
    | Visible -> "Visible")

let pp_source ppf s =
  Format.pp_print_string ppf
    (match s with
    | Memory.Allocated -> "Allocated"
    | Borrowed -> "Borrowed"
    | Peer -> "Peer")

let pp_region ppf (mem : Memory.region) =
  Format.fprintf ppf "%d bytes at 0x%x" mem.mapping.size mem.mapping.va

let source = Testable.make ~pp:pp_source ~equal:( = )

let pp_target ppf t =
  Format.pp_print_string ppf
    (match t with
    | Page_table.Gpu -> "Gpu"
    | System -> "System"
    | Peer i -> strf "Peer %d" i)

let target = Testable.make ~pp:pp_target ~equal:( = )

(* Placement *)

let test_small_bar =
  cases "a BAR that cannot reach all of the GPU's memory is small"
    ~name:(fun (bar, _) -> strf "%d MiB" (bar / mib))
    [
      (64 * mib, true);
      (256 * mib, true);
      ((512 * mib) - 4096, true);
      (512 * mib, false);
      (1024 * mib, false);
    ]
    (fun (bar, small) ->
      let x = gpu ~memory:(512 * mib) ~bar () in
      equal bool small (Memory.small_bar x.memory);
      Function.release x.fn)

let test_no_bytes =
  cases "no bytes, or fewer, are refused"
    ~name:(fun (name, n, _) -> strf "%s %d" name n)
    (List.concat_map
       (fun n ->
         [
           ("alloc Gpu", n, fun x -> ignore (alloc_opt x.memory Gpu n));
           ("alloc Host", n, fun x -> ignore (alloc_opt x.memory Host n));
           ( "map_host",
             n,
             fun x -> ignore (Memory.map_host x.memory tables_base n) );
         ])
       [ 0; -1 ])
    (fun (_, _, f) ->
      let x = gpu () in
      raises_match (Exn.invalid_arg ~substring:"Memory.") (fun () -> f x);
      equal ~msg:"no pin held" ranges [] x.fake.pins;
      Function.release x.fn)

let test_no_bar () =
  let x = gpu () in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.create x.fn x.tables ~bar:1);
  Function.release x.fn

type rule = In_gpu | Through_bar | In_host

(* What each rule promises of a fresh allocation [mem] of [x]. *)
let placed x rule (mem : Memory.region) =
  let m = mem.mapping in
  equal ~msg:"allocated" source Allocated mem.source;
  equal ~msg:"the tables map its pages" ranges (merge m.pages)
    (mapped x.g x.tables ~va:m.va m.size);
  let entries =
    List.filter
      (fun l -> l.va >= m.va && l.va < m.va + m.size)
      (leaves x.g x.tables)
  in
  let each f = List.map f entries in
  equal ~msg:"its entries in its target" (list target)
    (each (fun _ -> m.target))
    (each (fun l -> l.target));
  equal ~msg:"its entries snooped as it is" (list bool)
    (each (fun _ -> m.snooped))
    (each (fun l -> l.snooped));
  match rule with
  | In_gpu ->
      equal ~msg:"in the GPU's memory" target Gpu m.target;
      equal ~msg:"no page outside the GPU's memory" ranges []
        (List.filter (fun (pa, n) -> pa < 0 || pa + n > x.size) m.pages);
      is_none ~msg:"no window" mem.host
  | Through_bar ->
      equal ~msg:"in the GPU's memory" target Gpu m.target;
      let pa =
        match merge m.pages with [ (pa, _) ] -> pa | _ -> fail "one block"
      in
      let w = require_some ~msg:"a window" mem.host in
      equal ~msg:"the window is the BAR's bytes at the block" hex
        (fst x.fake.bar + pa)
        (Window.address w);
      equal ~msg:"as long as the memory" int m.size (Window.length w);
      equal ~msg:"its window uncached, as the driver mapped the BAR no way"
        (option bool) (Some false)
        (List.nth_opt x.fake.combined 0)
  | In_host ->
      equal ~msg:"system memory" target System m.target;
      equal ~msg:"uncached" bool true m.uncached;
      equal ~msg:"snooped" bool true m.snooped;
      equal ~msg:"on the machine's pages" int 0 (m.va mod page);
      let w = require_some ~msg:"a window" mem.host in
      equal ~msg:"the process's address is the GPU's" hex m.va
        (Window.address w);
      let runs =
        match List.assoc_opt m.va x.fake.dma with
        | Some r -> r
        | None -> fail "allocated by the function at its address"
      in
      equal ~msg:"the GPU reaches it at the function's runs" ranges (merge runs)
        (merge m.pages)

let test_kinds =
  cases "each kind is placed by its rule"
    ~name:(fun (k, bar, _) ->
      Format.asprintf "%a, %s BAR" pp_kind k
        (if bar = 256 * mib then "a small" else "a large"))
    [
      (Memory.Gpu, 512 * mib, In_gpu);
      (Gpu, 256 * mib, In_gpu);
      (Bar, 512 * mib, Through_bar);
      (Bar, 256 * mib, Through_bar);
      (Host, 512 * mib, In_host);
      (Host, 256 * mib, In_host);
      (Visible, 512 * mib, Through_bar);
      (Visible, 256 * mib, In_host);
    ]
    (fun (kind, bar, rule) ->
      let x = gpu ~memory:(512 * mib) ~bar () in
      placed x rule (alloc x kind (64 * kib));
      Function.release x.fn)

(* Sizes around the machine's page, 4 KiB, 8 MiB and 2 MiB. *)
let sizes =
  Gen.one_of
    [
      Gen.of_list ~pp:pp_hex
        [
          1;
          4095;
          4096;
          4097;
          page - 1;
          page;
          page + 1;
          (8 * mib) - 4097;
          (8 * mib) - 1;
          8 * mib;
          (8 * mib) + 1;
          (10 * mib) - 1;
          (10 * mib) + 1;
        ];
      Gen.with_pp pp_hex (Gen.int_range 1 (20 * mib));
    ]

let kinds = Gen.of_list ~pp:pp_kind [ Memory.Gpu; Bar; Host; Visible ]

let test_sizes =
  let x = lazy (gpu ~memory:(512 * mib) ()) in
  prop
    "sizes round up to the machine's page in system memory, to 4 KiB in the \
     GPU's, and to 2 MiB from 8 MiB on, where they map with large pages"
    (Gen.pair kinds sizes) (fun (kind, n) ->
      let x = Lazy.force x in
      let mem = alloc x kind n in
      let m = mem.mapping in
      let in_gpu = kind <> Host in
      cover "a large GPU block" (in_gpu && n >= 8 * mib);
      cover "a small GPU block" (in_gpu && n < 8 * mib);
      cover "system memory" (not in_gpu);
      let expected =
        if not in_gpu then round_up n page
        else if n >= 8 * mib then round_up n (2 * mib)
        else round_up n 4096
      in
      equal ~msg:"its size" hex expected m.size;
      if in_gpu && n >= 8 * mib then begin
        equal ~msg:"its address on 2 MiB" hex 0 (m.va mod (2 * mib));
        equal ~msg:"no block off 2 MiB" ranges []
          (List.filter
             (fun (pa, len) -> pa mod (2 * mib) <> 0 || len mod (2 * mib) <> 0)
             m.pages)
      end;
      Memory.free x.memory mem)

let test_uncached =
  cases "the GPU bypasses its caches when asked, and always for system memory"
    ~name:(fun (k, u, _) -> Format.asprintf "%a, uncached %b" pp_kind k u)
    [
      (Memory.Gpu, false, false);
      (Gpu, true, true);
      (Bar, false, false);
      (Bar, true, true);
      (Host, false, true);
      (Host, true, true);
    ]
    (fun (kind, uncached, expected) ->
      let x = gpu () in
      let mem = alloc ~uncached x kind (64 * kib) in
      equal ~msg:"the mapping" bool expected mem.mapping.uncached;
      equal ~msg:"no entry otherwise" (list hex) []
        (List.filter_map
           (fun l -> if l.uncached = expected then None else Some l.va)
           (leaves x.g x.tables));
      Function.release x.fn)

(* Exhaustion *)

let test_out_of_memory () =
  let x = gpu () in
  let before = capacity x in
  is_none ~msg:"more than the GPU's memory" ~pp:pp_region
    (alloc_opt x.memory Gpu (gpu_memory + 4096));
  is_none ~msg:"a BAR block larger than the memory" ~pp:pp_region
    (alloc_opt x.memory Bar (gpu_memory + 4096));
  let y = gpu ~memory:(512 * mib) () in
  is_none ~msg:"more than the space" ~pp:pp_region
    (alloc_opt y.memory Gpu (2 * space_length));
  is_none ~msg:"system memory larger than the space" ~pp:pp_region
    (alloc_opt y.memory Host (2 * space_length));
  equal ~msg:"no system memory held" int 0 (List.length y.fake.dma);
  equal ~msg:"nothing held" (pair int int) before (capacity x);
  Function.release x.fn;
  Function.release y.fn

let test_small_bar_fills () =
  let x = gpu ~memory:(512 * mib) ~bar:(256 * mib) () in
  let rec fill acc =
    match alloc_opt x.memory Bar mib with
    | Some mem -> fill (mem :: acc)
    | None -> acc
  in
  let blocks = fill [] in
  equal ~msg:"no block past the BAR" ranges []
    (List.concat_map
       (fun (mem : Memory.region) ->
         List.filter (fun (pa, n) -> pa + n > 256 * mib) mem.mapping.pages)
       blocks);
  not_equal ~msg:"blocks" int 0 (List.length blocks);
  is_some ~msg:"the GPU's memory past the BAR remains"
    (alloc_opt x.memory Gpu (64 * mib));
  Function.release x.fn

(* The main pool full, with the tables in it: system memory that needs new
   tables has no room for them. *)
let fill_main x =
  let rec go n =
    if n < 4096 then ()
    else
      match alloc_opt x.memory Gpu n with Some _ -> go n | None -> go (n / 2)
  in
  go (8 * mib)

let test_tables_full () =
  let x = gpu ~tables:Main ~memory:(16 * mib) () in
  fill_main x;
  is_none ~msg:"system memory" ~pp:pp_region
    (alloc_opt x.memory Host (64 * mib));
  equal ~msg:"no system memory held" int 0 (List.length x.fake.dma);
  Function.release x.fn

let test_system_refused () =
  let x = gpu ~memory:(512 * mib) ~bar:(256 * mib) () in
  let before = capacity x in
  x.fake.refuse <- Some "fake: the locked-memory limit (ulimit -l) is reached";
  List.iter
    (fun kind ->
      contains ~sub:"ulimit -l"
        (require_error (Memory.alloc x.memory kind (64 * kib))))
    [ Memory.Host; Visible; Bar ];
  equal ~msg:"its addresses returned" (pair int int) before (capacity x);
  x.fake.refuse <- None;
  is_some ~msg:"allocated once the limit is lifted"
    (alloc_opt x.memory Host (64 * kib));
  Function.release x.fn

(* A format whose [set_page] raises [Exit] while [armed]. *)
let raising armed (f : Page_table.format) =
  {
    f with
    set_page =
      (fun ~level ~table i ~pa tg ~uncached ~snooped ~fragment ->
        if !armed then raise Exit;
        f.set_page ~level ~table i ~pa tg ~uncached ~snooped ~fragment);
  }

let test_raising_format () =
  let armed = ref false in
  let x = gpu ~format:(raising armed) () in
  let before = capacity x in
  armed := true;
  raises Exit (fun () ->
      Memory.map_host x.memory (tables_base + (4 * mib)) page);
  equal ~msg:"no pin held" ranges [] x.fake.pins;
  List.iter
    (fun kind -> raises Exit (fun () -> Memory.alloc x.memory kind (64 * kib)))
    [ Memory.Gpu; Bar; Host ];
  equal ~msg:"no system memory held" int 0 (List.length x.fake.dma);
  equal ~msg:"its memory and addresses" (pair int int) before (capacity x);
  Function.release x.fn

(* Combining is the driver's choice: Bar memory's window maps the BAR as the
   driver's own windows on it do, uncached where it has none. *)
let test_bar_follows =
  cases "Bar memory's window maps as the driver's windows on its BAR"
    ~name:(function
      | None -> "no window"
      | Some true -> "combining"
      | Some false -> "uncached")
    [ None; Some true; Some false ]
    (fun way ->
      let x = gpu () in
      Option.iter
        (fun combine ->
          ignore (require_ok (Function.map ~combine ~length:page x.fn 0)))
        way;
      ignore (alloc x Bar (64 * kib));
      equal ~msg:"its window asked" (option bool)
        (Some (Option.value way ~default:false))
        (List.nth_opt x.fake.combined 0);
      Function.release x.fn)

(* Freeing *)

let test_free =
  cases "free returns the memory and its addresses"
    ~name:(Format.asprintf "%a" pp_kind) [ Memory.Gpu; Bar; Host; Visible ]
    (fun kind ->
      let x = gpu () in
      let before = capacity x in
      let mem = alloc x kind (3 * mib) in
      Memory.free x.memory mem;
      equal ~msg:"unmapped" ranges []
        (mapped x.g x.tables ~va:mem.mapping.va mem.mapping.size);
      equal ~msg:"its memory and addresses" (pair int int) before (capacity x);
      equal ~msg:"its system memory" int 0 (List.length x.fake.dma);
      Function.release x.fn)

(* The vendor's GPUs share the space: another GPU's system memory at addresses
   handed out again would be mapped over memory still being freed. *)
let test_free_order =
  cases "system memory is freed before its addresses are handed out again"
    ~name:(fun released -> if released then "released" else "live")
    [ false; true ]
    (fun released ->
      let x = gpu () in
      let s = Page_table.space x.tables in
      let mem = alloc x Host page in
      let meanwhile = ref None in
      x.fake.freeing <-
        (fun () ->
          meanwhile := Space.alloc ~align:page s page;
          Option.iter (Space.free s) !meanwhile);
      if released then Function.release x.fn;
      Memory.free x.memory mem;
      satisfies ~msg:"addresses handed out while it is freed"
        ~claim:"not the memory's" (option hex)
        (fun a -> a <> Some mem.mapping.va)
        !meanwhile;
      if not released then Function.release x.fn)

let test_free_refused () =
  let x = gpu () and y = gpu () in
  let mem = alloc x Gpu (64 * kib) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.free y.memory mem);
  let borrowed = Result.get_ok (Memory.map_host x.memory tables_base page) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.free x.memory borrowed);
  Memory.free x.memory mem;
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.free x.memory mem);
  Function.release x.fn;
  Function.release y.fn

(* Borrowing *)

let test_map_host () =
  let x = gpu () in
  let a = tables_base + (4 * mib) and n = 3 * page in
  let mem =
    match Memory.map_host x.memory a n with
    | Ok mem -> mem
    | Error why -> fail why
  in
  let m = mem.mapping in
  equal ~msg:"borrowed" source Borrowed mem.source;
  equal ~msg:"at its address" hex a m.va;
  equal ~msg:"its bytes" hex n m.size;
  equal ~msg:"system memory" target System m.target;
  equal ~msg:"uncached" bool true m.uncached;
  equal ~msg:"snooped" bool true m.snooped;
  equal ~msg:"pinned" ranges [ (a, n) ] x.fake.pins;
  equal ~msg:"the tables map its pages" ranges (merge m.pages)
    (mapped x.g x.tables ~va:a n);
  Memory.unmap x.memory mem;
  equal ~msg:"unmapped" ranges [] (mapped x.g x.tables ~va:a n);
  equal ~msg:"unpinned" ranges [] x.fake.pins;
  Function.release x.fn

let test_map_host_refused =
  let span_end = tables_base + (1 lsl 48) in
  cases "map_host refuses"
    ~name:(fun (name, _, _) -> name)
    [
      ("an address off a page", tables_base + 1, page);
      ("an address on 4 KiB off the machine's page", tables_base + 4096, page);
      ("an address below the GPU's", tables_base - page, page);
      ("an address past the GPU's", span_end, page);
      ("bytes across the end of the GPU's addresses", span_end - page, 2 * page);
    ]
    (fun (_, a, n) ->
      let x = gpu () in
      is_error ~msg:"refused" (Memory.map_host x.memory a n);
      equal ~msg:"no pin held" ranges [] x.fake.pins;
      Function.release x.fn)

let test_map_host_unpinnable () =
  let x = gpu () in
  x.fake.refuse <- Some "fake: the pages cannot be locked";
  (match Memory.map_host x.memory tables_base page with
  | Ok _ -> fail "memory that cannot be pinned is mapped"
  | Error why -> contains ~msg:"the pin's reason" ~sub:"cannot be locked" why);
  Function.release x.fn

let test_map_host_no_tables () =
  let x = gpu ~tables:Main ~memory:(16 * mib) () in
  fill_main x;
  is_error ~msg:"refused"
    (Memory.map_host x.memory (tables_base + (1 lsl 45)) page);
  equal ~msg:"no pin held" ranges [] x.fake.pins;
  Function.release x.fn

let test_map_host_mapped () =
  let x = gpu () in
  let host = alloc x Host page in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_host x.memory host.mapping.va page);
  let a = tables_base + (4 * mib) in
  let borrowed = Result.get_ok (Memory.map_host x.memory a (2 * page)) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_host x.memory (a + page) (2 * page));
  equal ~msg:"only the first pin held" ranges [ (a, 2 * page) ] x.fake.pins;
  Memory.unmap x.memory borrowed;
  Memory.free x.memory host;
  Function.release x.fn

let test_unmap_refused () =
  let x = gpu () and y = gpu () in
  let mem = alloc x Gpu (64 * kib) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.unmap x.memory mem);
  let borrowed = Result.get_ok (Memory.map_host x.memory tables_base page) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.unmap y.memory borrowed);
  Memory.unmap x.memory borrowed;
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.unmap x.memory borrowed);
  Function.release x.fn;
  Function.release y.fn

(* Peers *)

(* Two GPUs of one machine sharing a space. *)
let pair_of ?owner_bar ?peer ?(owner_addressing = Machine.Physical)
    ?(addressing = Machine.Physical) () =
  let m = machine () in
  let space = Space.create ~base:space_base space_length in
  let owner =
    gpu ~machine:m ~space ?bar:owner_bar ?peer ~addressing:owner_addressing
      ~memory:(512 * mib) ()
  in
  let x = gpu ~machine:m ~space ~addressing () in
  (owner, x)

let release_all l = List.iter (fun x -> Function.release x.fn) l

let peer_ok x owner mem =
  match Memory.map_peer x.memory ~owner:owner.memory mem with
  | Ok p -> p
  | Error why -> fail why

let test_peer =
  cases "a peer maps the owner's memory at its address"
    ~name:(Format.asprintf "%a" pp_kind) [ Memory.Gpu; Bar; Host ] (fun kind ->
      let owner, x = pair_of () in
      let mem = alloc owner kind (3 * mib) in
      let p = peer_ok x owner mem in
      equal ~msg:"a peer's" source Peer p.source;
      equal ~msg:"at its address" hex mem.mapping.va p.mapping.va;
      equal ~msg:"its bytes" hex mem.mapping.size p.mapping.size;
      equal ~msg:"system memory to the peer" target System p.mapping.target;
      let expected =
        match kind with
        | Host -> merge mem.mapping.pages
        | _ ->
            merge
              (List.map
                 (fun (pa, n) -> (fst owner.fake.bar + pa, n))
                 mem.mapping.pages)
      in
      equal ~msg:"through the owner's BAR, or at its pages" ranges expected
        (merge p.mapping.pages);
      equal ~msg:"the peer's tables map them" ranges expected
        (mapped x.g x.tables ~va:p.mapping.va p.mapping.size);
      let owners = leaves owner.g owner.tables in
      Memory.unmap x.memory p;
      equal ~msg:"unmapped from the peer" ranges []
        (mapped x.g x.tables ~va:p.mapping.va p.mapping.size);
      equal ~msg:"the owner's tables unchanged" (list leaf) owners
        (leaves owner.g owner.tables);
      Memory.free owner.memory mem;
      release_all [ owner; x ])

let link_base = 1 lsl 44

let test_peer_link () =
  let asked = ref [] in
  let peer r =
    asked := r;
    (List.map (fun (pa, n) -> (link_base + pa, n)) r, Page_table.Peer 3)
  in
  let owner, x = pair_of ~peer () in
  let mem = alloc owner Gpu (3 * mib) in
  let p = peer_ok x owner mem in
  equal ~msg:"asked for the memory's ranges" ranges (merge mem.mapping.pages)
    (merge !asked);
  equal ~msg:"in the link's target" target (Peer 3) p.mapping.target;
  equal ~msg:"at the link's addresses" ranges
    (merge (List.map (fun (pa, n) -> (link_base + pa, n)) mem.mapping.pages))
    (merge p.mapping.pages);
  release_all [ owner; x ]

let test_peer_refused () =
  let refused msg (owner, x) kind =
    let mem = alloc owner kind (64 * kib) in
    is_error ~msg (Memory.map_peer x.memory ~owner:owner.memory mem);
    release_all [ owner; x ]
  in
  refused "GPUs on two machines" (gpu ~memory:(512 * mib) (), gpu ()) Gpu;
  refused "an owner behind an IOMMU" (pair_of ~owner_addressing:Iommu ()) Host;
  refused "a mapper behind an IOMMU" (pair_of ~addressing:Iommu ()) Host;
  refused "the GPU's memory past a small BAR"
    (pair_of ~owner_bar:(256 * mib) ())
    Gpu;
  let owner, x = pair_of ~owner_bar:(256 * mib) () in
  let mem = alloc owner Host (64 * kib) in
  is_ok ~msg:"system memory of an owner with a small BAR"
    (Memory.map_peer x.memory ~owner:owner.memory mem);
  release_all [ owner; x ]

let test_peer_no_tables () =
  let m = machine () in
  let space = Space.create ~base:space_base space_length in
  let owner = gpu ~machine:m ~space () in
  let x = gpu ~machine:m ~space ~tables:Main ~memory:(16 * mib) () in
  fill_main x;
  (* At addresses no table of [x] covers yet, in pages that need leaf tables. *)
  let mem = alloc owner Host (64 * mib) in
  is_error ~msg:"refused" (Memory.map_peer x.memory ~owner:owner.memory mem);
  release_all [ owner; x ]

let test_peer_not_owned () =
  let owner, x = pair_of () in
  let mine = alloc x Gpu (64 * kib) in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_peer x.memory ~owner:owner.memory mine);
  let borrowed =
    Result.get_ok (Memory.map_host owner.memory tables_base page)
  in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_peer x.memory ~owner:owner.memory borrowed);
  let mem = alloc owner Gpu (64 * kib) in
  let p = peer_ok x owner mem in
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_peer x.memory ~owner:owner.memory mem);
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.map_peer owner.memory ~owner:x.memory p);
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.free x.memory p);
  Memory.unmap x.memory p;
  raises_match (Exn.invalid_arg ~substring:"") (fun () ->
      Memory.unmap x.memory p);
  release_all [ owner; x ]

(* After the GPU is given back *)

(* The GPU reset and opened again: a new instance's tables in the same GPU
   memory. The old instance frees and unmaps all it holds, and the new one's
   entries are as they were. *)
let bindings g =
  List.sort compare
    (Hashtbl.fold (fun a e l -> (a, e) :: l) g.Tables.entries [])

let test_reopened () =
  let m = machine () in
  let space = Space.create ~base:space_base space_length in
  let owner = gpu ~machine:m ~space () in
  let x = gpu ~machine:m ~space () in
  let shared = alloc owner Gpu (64 * kib) in
  let addresses = snd (capacity x) in
  let held =
    List.map (fun k -> alloc x k (3 * mib)) [ Memory.Gpu; Bar; Host; Visible ]
  in
  let borrowed = Result.get_ok (Memory.map_host x.memory tables_base page) in
  let p = peer_ok x owner shared in
  Function.release x.fn;
  let reopened =
    Page_table.create ~base:tables_base (Tables.format x.g) space
      ~memory:gpu_memory ~boot:mib ~tables:Pool ~pages:large_pages
  in
  Page_table.booted reopened;
  let fresh = require_some (Page_table.alloc reopened (3 * mib)) in
  let entries = bindings x.g and touches = x.g.touches in
  List.iter (Memory.free x.memory) held;
  Memory.unmap x.memory borrowed;
  Memory.unmap x.memory p;
  equal ~msg:"no entry read or written" int touches x.g.touches;
  equal ~msg:"the new instance's entries unchanged"
    (list (pair hex int64))
    entries (bindings x.g);
  equal ~msg:"its system memory returned" int 0 (List.length x.fake.dma);
  equal ~msg:"its pins returned" ranges [] x.fake.pins;
  Page_table.free reopened fresh;
  equal ~msg:"its addresses returned" int addresses (snd (capacity x));
  release_all [ owner ]

(* Allocations, borrows, frees, unmaps and the release of a GPU against a model
   that knows which calls may write the GPU's memory: every free and unmap
   before the release, none after. *)
module Model = struct
  type gpu = {
    mutable released : bool;
    mutable dma : int;  (** Host allocations held. *)
    mutable pins : int list;  (** Slots borrowed. *)
    mutable allocated : int;
  }

  type mem = {
    owner : gpu;
    source : Memory.source;
    host : bool;
    slot : int;
    mutable held : bool;
  }

  let create () = { released = false; dma = 0; pins = []; allocated = 0 }

  let alloc kind _ g =
    let host = kind = Memory.Host in
    if host then g.dma <- g.dma + 1;
    g.allocated <- g.allocated + 1;
    { owner = g; source = Allocated; host; slot = -1; held = true }

  let map_host slot g =
    g.pins <- slot :: g.pins;
    { owner = g; source = Borrowed; host = false; slot; held = true }

  let give_back source g m =
    if m.owner != g || m.source <> source || not m.held then
      invalid_arg "not held";
    m.held <- false;
    cover "given back after the release" g.released;
    cover "given back before the release" (not g.released);
    if m.host then g.dma <- g.dma - 1;
    if source = Borrowed then g.pins <- List.filter (( <> ) m.slot) g.pins
    else g.allocated <- g.allocated - 1;
    not g.released

  let free g m = give_back Allocated g m
  let unmap g m = give_back Borrowed g m
  let release g = g.released <- true
  let held g = (g.dma, List.length g.pins)
end

let slot_address s = tables_base + (s * 4 * mib)

let touched x f =
  let before = x.g.touches in
  f ();
  x.g.touches > before

(* [mine] is the released GPU's memory and [theirs] the other GPU's, both
   allocated before the release. *)
let test_released_refused =
  cases "a released GPU refuses requests, writing nothing"
    ~name:(fun (what, _) -> what)
    [
      ("alloc Gpu", fun x _ _ _ -> ignore (Memory.alloc x.memory Gpu page));
      ("alloc Host", fun x _ _ _ -> ignore (Memory.alloc x.memory Host page));
      ( "map_host",
        fun x _ _ _ -> ignore (Memory.map_host x.memory (slot_address 0) page)
      );
      ( "map_peer of its memory",
        fun x owner mine _ ->
          ignore (Memory.map_peer owner.memory ~owner:x.memory mine) );
      ( "map_peer for it",
        fun x owner _ theirs ->
          ignore (Memory.map_peer x.memory ~owner:owner.memory theirs) );
    ]
    (fun (_, request) ->
      let owner, x = pair_of () in
      let mine = alloc x Gpu page and theirs = alloc owner Gpu page in
      Function.release x.fn;
      let before = capacity x in
      equal ~msg:"nothing written" bool false
        (touched x (fun () ->
             raises_match (Exn.invalid_arg ~substring:"released") (fun () ->
                 request x owner mine theirs)));
      equal ~msg:"nothing taken" (pair int int) before (capacity x);
      release_all [ owner ])

let test_released =
  let gpus = abstract "g" ~release:(fun x -> Function.release x.fn) in
  let mems = abstract "mem" in
  let small = Gen.with_pp pp_hex (Gen.int_range 1 mib) in
  let slots = Gen.int_range 0 7 in
  let live (g : Model.gpu) = not g.released in
  stateful "after its release, a GPU's memory is given back without a write"
    ~steps:30
    [
      command "create" (Gen.unit @-> makes gpus) Model.create (fun () -> gpu ());
      command "alloc"
        ~pre:(fun _ _ g -> live g)
        (kinds @-> small @-> gpus ^-> makes mems)
        Model.alloc
        (fun kind n x -> alloc x kind n);
      command "map_host"
        ~pre:(fun s g -> live g && not (List.mem s g.pins))
        (slots @-> gpus ^-> makes mems)
        Model.map_host
        (fun s x ->
          match Memory.map_host x.memory (slot_address s) page with
          | Ok m -> m
          | Error why -> fail why);
      command "free"
        (gpus ^-> mems ^-> returns bool)
        Model.free
        (fun x m -> touched x (fun () -> Memory.free x.memory m));
      command "unmap"
        (gpus ^-> mems ^-> returns bool)
        Model.unmap
        (fun x m -> touched x (fun () -> Memory.unmap x.memory m));
      command "release"
        (gpus ^-> returns unit)
        Model.release
        (fun x -> Function.release x.fn);
      command "held"
        (gpus ^-> returns (pair int int))
        Model.held
        (fun x -> (List.length x.fake.dma, List.length x.fake.pins));
      command "addresses"
        ~pre:(fun (g : Model.gpu) -> g.allocated = 0)
        (gpus ^-> returns bool)
        (fun _ -> true)
        (fun x ->
          let s = Page_table.space x.tables in
          match Space.alloc s (space_length / 4) with
          | Some a ->
              Space.free s a;
              true
          | None -> false);
    ]

let () =
  exit
  @@ run "rig_pci.memory"
       [
         group ~timeout:patience "placement"
           [
             test_small_bar;
             test "a BAR the function lacks is refused" test_no_bar;
             test_no_bytes;
             test_kinds;
             test_sizes;
             test_uncached;
           ];
         group ~timeout:patience "exhaustion"
           [
             test "no room is None" test_out_of_memory;
             test "a small BAR's blocks stay inside it" test_small_bar_fills;
             test "no room for a table is None" test_tables_full;
             test
               "system memory or a BAR window refused is an Error, having \
                given back its addresses"
               test_system_refused;
             test "a format that raises passes through, having given back"
               test_raising_format;
             test_bar_follows;
           ];
         group ~timeout:patience "freeing"
           [
             test_free;
             test_free_order;
             test "memory not allocated by the GPU, or freed, is refused"
               test_free_refused;
           ];
         group ~timeout:patience "borrowing"
           [
             test "map_host maps and pins, unmap unpins" test_map_host;
             test_map_host_refused;
             test "an unpinnable range is refused with the pin's reason"
               test_map_host_unpinnable;
             test "no room for a table is refused" test_map_host_no_tables;
             test "memory the GPU maps already is refused, holding no pin"
               test_map_host_mapped;
             test "memory not borrowed by the GPU, or unmapped, is refused"
               test_unmap_refused;
           ];
         group ~timeout:patience "peers"
           [
             test_peer;
             test "a link maps through the owner's peer function" test_peer_link;
             test "refusals" test_peer_refused;
             test "no room for a table is refused" test_peer_no_tables;
             test
               "memory the owner did not allocate, or mapped already, is \
                refused"
               test_peer_not_owned;
           ];
         group ~timeout:patience "after the release"
           [
             test "a reopened GPU is not written by its old instance"
               test_reopened;
             test_released_refused;
             test_released;
           ];
       ]
