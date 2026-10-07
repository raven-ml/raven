(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Page tables over a fake GPU memory that keeps the tables' entries by address.
   A walk of the tables from the root reads back what they map; a reference
   computes, from the module's rules, what they must map. *)

open Windtrap
open Device_pci
open Device_pci_support
open Tables

let page = 4096

(* Entries *)

let space =
  Testable.make
    ~pp:(fun ppf s ->
      Format.fprintf ppf "the space of %d addresses at 0x%x" (Space.length s)
        (Space.base s))
    ~equal:( == )

(* [zeroed ~msg zs (a, n)] asserts that one of the ranges [zs] zeroed holds the
   [n] bytes at [a]. *)
let zeroed ~msg zs (a, n) =
  satisfies ~msg ~claim:"one zeroed range holds the block"
    (list (pair hex int))
    (List.exists (fun (z, k) -> z <= a && a + n <= z + k))
    zs

(* Entries compared on all but their fragment, which a law of its own states. *)
let placed =
  Testable.make ~pp:pp_entry ~equal:(fun a b ->
      { a with fragment = 0 } = { b with fragment = 0 })

let pages g t = fst (walk g t)

(* The entries that map [ranges] from [va] on. Each page maps at the highest
   level that allows pages, to whose size its virtual and physical addresses are
   aligned and that the rest of its range fills. Its fragment is the largest
   naturally aligned block of its run, in both address spaces, holding it: its
   run joins the ranges that follow each other physically. Virtual addresses are
   aligned from [base], which the tests keep aligned to 1 TiB, so blocks are
   aligned in the space and in the tables alike. *)
let expect ~base ?(target = Page_table.Gpu) ?(uncached = false)
    ?(snooped = false) ~va ranges =
  let rec runs = function
    | (a, n) :: (b, k) :: rest when a + n = b -> runs ((a, n + k) :: rest)
    | r :: rest -> r :: runs rest
    | [] -> []
  in
  let ranges = runs ranges in
  let out = ref [] in
  let start = ref va in
  List.iter
    (fun (pa, len) ->
      let v0 = !start in
      let off = ref 0 in
      while !off < len do
        let v = v0 + !off and p = pa + !off in
        let fits s =
          let size = 1 lsl s in
          (v - base) land (size - 1) = 0
          && p land (size - 1) = 0
          && len - !off >= size
        in
        let rec level l =
          if large l && fits shifts.(l) then l else level (l + 1)
        in
        let level = level 0 in
        let rec fragment k =
          let b = 1 lsl (13 + k) in
          let vb = v land lnot (b - 1) in
          if vb >= v0 && vb + b <= v0 + len && (p - v) land (b - 1) = 0 then
            fragment (k + 1)
          else k
        in
        out :=
          {
            va = v;
            level;
            pa = p;
            target;
            uncached;
            snooped;
            fragment = fragment 0;
          }
          :: !out;
        off := !off + (1 lsl shifts.(level))
      done;
      start := v0 + len)
    ranges;
  List.rev !out

let expect_mapping ~base (m : Page_table.mapping) =
  expect ~base ~target:m.target ~uncached:m.uncached ~snooped:m.snooped ~va:m.va
    m.pages

(* Fixtures *)

let base = 1 lsl 40

let tables ?(tables = Page_table.Main) ?(memory = 66 * mib) ?(boot = mib)
    ?(pages = [ (2 * mib, 2 * mib); (64 * kib, 64 * kib); (page, page) ])
    ?(length = 1 lsl 40) ?(booted = true) () =
  let g = Tables.memory () in
  let s = Space.create ~base length in
  let t = Page_table.create (format g) s ~memory ~boot ~tables ~pages in
  if booted then Page_table.booted t;
  (t, g)

(* The number of 4 KiB blocks [alloc] hands out before it answers [None], all
   given back after. The fit bound leaves it short of the free bytes, so it
   measures a leak only by comparison: memory given back in full hands out the
   same blocks again. *)
let drain alloc free =
  let rec go acc =
    match alloc () with Some a -> go (a :: acc) | None -> acc
  in
  let taken = go [] in
  List.iter free taken;
  List.length taken

let capacity t =
  drain (fun () -> Page_table.palloc ~zero:false t page) (Page_table.pfree t)

let space_capacity t =
  let s = Page_table.space t in
  drain (fun () -> Space.alloc s page) (Space.free s)

(* Creating tables *)

let test_create () =
  let g = Tables.memory () in
  let s = Space.create ~base (1 lsl 40) in
  let t =
    Page_table.create (format g) s ~memory:(66 * mib) ~boot:mib ~tables:Main
      ~pages:[ (page, page) ]
  in
  equal ~msg:"its space" space s (Page_table.space t);
  equal ~msg:"translates from the space's base" hex base (Page_table.base t);
  equal ~msg:"2^bits addresses" hex (1 lsl 48) (Page_table.span t);
  less ~msg:"the root in the boot pool" hex ~than:mib (Page_table.root t);
  let t =
    Page_table.create ~base:0 (format g) s ~memory:(66 * mib) ~boot:mib
      ~tables:Main
      ~pages:[ (page, page) ]
  in
  equal ~msg:"or from the base given" hex 0 (Page_table.base t)

(* The main pool is what the boot pool and the tables' pool leave at the end of
   memory: the tables' pool is [memory / 512] rounded up to 1 MiB. A fresh
   pool's first block is its start. *)
let test_main_pool =
  cases
    ~name:(fun (memory, boot, kind, _) ->
      Printf.sprintf "%d KiB, a boot pool of %d KiB, %s" (memory / kib)
        (boot / kib)
        (match kind with
        | Page_table.Pool -> "a tables' pool"
        | Main -> "no tables' pool"))
    "the main pool"
    [
      (66 * mib, mib, Page_table.Pool, 64 * mib);
      (66 * mib, mib, Main, 65 * mib);
      (gib, mib, Pool, gib - (3 * mib));
      (600 * mib, 4 * mib, Pool, 594 * mib);
      (2 * mib, mib, Pool, 0);
      (mib, mib, Main, 0);
    ]
    (fun (memory, boot, kind, main) ->
      let t, _ = tables ~memory ~boot ~tables:kind () in
      equal ~msg:"the GPU's memory" int memory (Page_table.memory t);
      equal ~msg:"the main pool's first block" (option hex)
        (if main = 0 then None else Some (memory - main))
        (Page_table.palloc ~zero:false t page))

let test_create_refusals =
  cases
    ~name:(fun (why, _, _, _) -> why)
    "refuses"
    [
      ("a boot pool of no bytes", 66 * mib, 0, Page_table.Main);
      ("a boot pool smaller than the root table", 66 * mib, page - 1, Main);
      ("a boot pool larger than memory", mib, mib + page, Main);
      ("a tables' pool past memory", mib + page, mib, Pool);
    ]
    (fun (_, memory, boot, kind) ->
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          tables ~memory ~boot ~tables:kind ()))

(* Levels rise from bit 12, the leaf's, to below [bits]. *)
let test_level_refusals =
  cases
    ~name:(fun (why, _, _) -> why)
    "refuses levels"
    [
      ("a leaf level of 8 KiB pages", [ 13; 21; 30; 39 ], 48);
      ("levels that do not rise", [ 12; 30; 21; 39 ], 48);
      ("a root level at the top bit", [ 12; 21; 30; 39 ], 39);
      ("more bits than an int holds", [ 12; 21; 30; 39 ], Sys.int_size);
    ]
    (fun (_, levels, bits) ->
      let g = Tables.memory () in
      let s = Space.create ~base (1 lsl 40) in
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Page_table.create
            { (format g) with levels; bits }
            s ~memory:(66 * mib) ~boot:mib ~tables:Main
            ~pages:[ (page, page) ]))

(* The largest page the fake format maps is 1 GiB: the tables' base must be a
   multiple of it. *)
let test_base_refusals =
  cases
    ~name:(fun base -> Printf.sprintf "base 0x%x" base)
    "refuses a base off the largest page"
    [ page; 2 * mib; gib + (2 * mib); -gib ]
    (fun b ->
      let g = Tables.memory () in
      let s = Space.create ~base (1 lsl 40) in
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Page_table.create ~base:b (format g) s ~memory:(66 * mib) ~boot:mib
            ~tables:Main
            ~pages:[ (page, page) ]))

(* Fragments are aligned in the space's addresses. From a base at 1 GiB, two 1
   GiB pages at the base are the first 2 GiB the tables translate, but they
   straddle the space's 2 GiB blocks: each is its own fragment. *)
let test_fragments_from_base () =
  let g = Tables.memory () in
  let s = Space.create ~base (1 lsl 40) in
  let t =
    Page_table.create ~base:gib (format g) s ~memory:(66 * mib) ~boot:mib
      ~tables:Main
      ~pages:[ (page, page) ]
  in
  ignore (require_some (Page_table.map t ~va:gib Gpu [ (0, 2 * gib) ]));
  equal ~msg:"(va, level, fragment)"
    (list (triple hex int int))
    [ (gib, 1, 18); (2 * gib, 1, 18) ]
    (List.map (fun e -> (e.va, e.level, e.fragment)) (pages g t))

(* Mapping *)

let pp_ranges =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
    (fun ppf (pa, n) -> Format.fprintf ppf "0x%x+0x%x" pa n)

type case = {
  offset : int;  (** From the space's base. *)
  ranges : (int * int) list;
  tg : Page_table.target;
  unc : bool;
  snoop : bool;
}

let pp_case ppf c =
  Format.fprintf ppf "va base+0x%x [%a] %a%s%s" c.offset pp_ranges c.ranges
    pp_target c.tg
    (if c.unc then " uncached" else "")
    (if c.snoop then " snooped" else "")

(* Addresses aligned to 4 KiB, 64 KiB, 2 MiB and 1 GiB, sizes of a few pages, of
   2 MiB and a few pages past it, and one case in ten a 1 GiB page and a page
   past it. Physically contiguous neighbours are moved apart by a page, so that
   each range is one run; one case in four then splits its first range in two
   that follow each other, one run of two ranges. *)
let case_gen =
  let addr =
    Gen.frequency
      [
        (3, Gen.map (fun k -> k * page) (Gen.int_range 0 0xF_FFFF));
        (1, Gen.map (fun k -> k * 64 * kib) (Gen.int_range 0 0xFFFF));
        (3, Gen.map (fun k -> k * 2 * mib) (Gen.int_range 0 1023));
        (1, Gen.map (fun k -> k * gib) (Gen.int_range 0 15));
      ]
  in
  let size =
    Gen.frequency
      [
        (4, Gen.map (fun k -> k * page) (Gen.int_range 1 40));
        (1, Gen.map (fun k -> k * 64 * kib) (Gen.int_range 1 8));
        (2, Gen.map (fun k -> k * 2 * mib) (Gen.int_range 1 2));
        (2, Gen.map (fun k -> (2 * mib) + (k * page)) (Gen.int_range 1 3));
      ]
  in
  let apart ranges =
    let rec go prev = function
      | [] -> []
      | (pa, n) :: rest ->
          let pa = if pa = prev then pa + page else pa in
          (pa, n) :: go (pa + n) rest
    in
    go (-1) ranges
  in
  let gib_page =
    Gen.map
      (fun (k, j) -> (k * gib, [ (j * gib, gib + page) ]))
      (Gen.pair (Gen.int_range 0 15) (Gen.int_range 0 15))
  in
  let ranges =
    Gen.pair addr (Gen.list ~size:(Gen.int_range 1 3) (Gen.pair addr size))
  in
  (* [split k ranges] cuts the first range [k] pages in, when [k > 0] and it has
     more pages than that. *)
  let split k = function
    | (pa, n) :: rest when k > 0 && k * page < n ->
        (pa, k * page) :: (pa + (k * page), n - (k * page)) :: rest
    | ranges -> ranges
  in
  let cut = Gen.frequency [ (3, Gen.constant 0); (1, Gen.int_range 1 1024) ] in
  Gen.with_pp pp_case
    (Gen.map
       (fun (((offset, ranges), k), tg, (unc, snoop)) ->
         { offset; ranges = split k (apart ranges); tg; unc; snoop })
       (Gen.triple
          (Gen.pair (Gen.frequency [ (9, ranges); (1, gib_page) ]) cut)
          (Gen.frequency
             [
               (1, Gen.constant ~pp:pp_target Page_table.Gpu);
               (1, Gen.constant ~pp:pp_target Page_table.System);
               (1, Gen.map (fun i -> Page_table.Peer i) (Gen.int_range 0 15));
             ])
          (Gen.pair Gen.bool Gen.bool)))

(* A mapping writes exactly the entries its pages, levels and fragments call
   for, and flushes; unmapping it leaves the root alone and its tables back in
   the pool. *)
let test_map =
  prop "map writes the entries of its ranges and unmap removes them" ~count:200
    case_gen (fun c ->
      let t, g = tables ~memory:(4 * mib) () in
      let free = capacity t in
      let va = base + c.offset in
      let size = List.fold_left (fun n (_, k) -> n + k) 0 c.ranges in
      let m =
        require_some ~msg:"mapped"
          (Page_table.map ~uncached:c.unc ~snooped:c.snoop t ~va c.tg c.ranges)
      in
      equal ~msg:"va" hex va m.va;
      equal ~msg:"size" int size m.size;
      equal ~msg:"pages" (list (pair hex int)) c.ranges m.pages;
      equal ~msg:"uncached" bool c.unc m.uncached;
      equal ~msg:"snooped" bool c.snoop m.snooped;
      equal ~msg:"target" target c.tg m.target;
      let want =
        expect ~base ~target:c.tg ~uncached:c.unc ~snooped:c.snoop ~va c.ranges
      in
      cover "a 1 GiB page" (List.exists (fun e -> e.level = 1) want);
      cover "2 MiB and 4 KiB pages in one mapping"
        (List.exists (fun e -> e.level = 2) want
        && List.exists (fun e -> e.level = 3) want);
      cover "a fragment the physical address bounds"
        (List.exists
           (fun e ->
             e.pa land ((page lsl (e.fragment + 1)) - 1) <> 0
             && e.va land ((page lsl (e.fragment + 1)) - 1) = 0)
           want);
      cover "a run of two ranges"
        (match c.ranges with
        | (pa, n) :: (pb, _) :: _ -> pa + n = pb
        | _ -> false);
      equal ~msg:"entries" (list placed) want (pages g t);
      equal ~msg:"flushed" int 0 g.unflushed;
      Page_table.unmap t ~va size;
      equal ~msg:"no entry after unmap" (list entry) [] (pages g t);
      equal ~msg:"flushed after unmap" int 0 g.unflushed;
      equal ~msg:"no table but the root" (list hex)
        [ Page_table.root t ]
        (snd (walk g t));
      equal ~msg:"the tables went back to the pool" int free (capacity t))

(* Each entry's fragment is the largest naturally aligned block of its run, in
   both address spaces, that holds it. *)
let test_fragments =
  prop "an entry's fragment is the largest aligned block of its run" ~count:200
    case_gen (fun c ->
      let t, g = tables ~memory:(4 * mib) () in
      let va = base + c.offset in
      ignore (Page_table.map t ~va c.tg c.ranges);
      let fragments = List.map (fun e -> (e.va, e.fragment)) in
      equal
        (list (pair hex int))
        (fragments (expect ~base ~target:c.tg ~va c.ranges))
        (fragments (pages g t)))

(* Fragments the rule gives, at the space's base: ranges that follow each other
   are one run, and a block must be aligned in both address spaces. *)
let test_fragment_cases =
  cases
    ~name:(fun (what, _, _, _) -> what)
    "fragments"
    [
      ("three pages", 0, [ (0, 0x3000) ], [ 1; 1; 0 ]);
      ("two ranges, one run", 0, [ (0, page); (page, page) ], [ 1; 1 ]);
      ("a run across a block's bound", page, [ (page, 0x2000) ], [ 0; 0 ]);
      ( "64 KiB aligned in both",
        0,
        [ (0x1_0000, 0x1_0000) ],
        List.init 16 (fun _ -> 4) );
      ("aligned virtually only", 0, [ (page, 0x4000) ], [ 0; 0; 0; 0 ]);
    ]
    (fun (_, offset, ranges, want) ->
      let t, g = tables () in
      ignore (require_some (Page_table.map t ~va:(base + offset) Gpu ranges));
      equal (list int) want (List.map (fun e -> e.fragment) (pages g t)))

(* Ranges that follow each other are one run, empty ones aside: a 2 MiB page may
   span them. *)
let test_page_across_ranges () =
  let t, g = tables () in
  let va = base + (2 * mib) in
  let half = mib in
  ignore
    (require_some
       (Page_table.map t ~va Gpu
          [ (4 * mib, half); (0, 0); ((4 * mib) + half, half) ]));
  equal ~msg:"one 2 MiB page"
    (list (pair hex int))
    [ (va, 2) ]
    (List.map (fun e -> (e.va, e.level)) (pages g t))

let test_gib_pages () =
  let t, g = tables () in
  let va = base + (2 * gib) in
  ignore (require_some (Page_table.map t ~va Gpu [ (4 * gib, gib) ]));
  equal ~msg:"one 1 GiB page" (list entry)
    [
      {
        va;
        level = 1;
        pa = 4 * gib;
        target = Gpu;
        uncached = false;
        snooped = false;
        fragment = 18;
      };
    ]
    (pages g t);
  Page_table.unmap t ~va gib;
  let pa = (4 * gib) + (2 * mib) in
  ignore (require_some (Page_table.map t ~va Gpu [ (pa, gib) ]));
  let ps = pages g t in
  equal ~msg:"memory aligned to 2 MiB maps with 2 MiB pages" int 512
    (List.length ps);
  List.iter
    (fun e ->
      equal ~msg:"level" int 2 e.level;
      equal ~msg:"fragment: 2 MiB, aligned in both" int 9 e.fragment;
      equal ~msg:"pa" hex (pa + (e.va - va)) e.pa)
    ps

(* A mapping of three pages at the start of a 2 MiB region and one of a 2 MiB
   page after it, and calls that touch them or break the rules on ranges. A
   refused call leaves the tables as they were. *)
let test_map_refusals =
  let va = base + (2 * mib) and large = base + (4 * mib) in
  cases
    ~name:(fun (what, _) -> what)
    "refuses"
    [
      ( "a map whose first address is mapped",
        fun t ->
          ignore (Page_table.map t ~va:(va + 0x2000) System [ (0, 0x2000) ]) );
      ( "a map whose last address is mapped",
        fun t ->
          ignore (Page_table.map t ~va:(va - page) System [ (0, 0x2000) ]) );
      ( "a map over the whole mapping",
        fun t ->
          ignore (Page_table.map t ~va:(va - page) System [ (0, 0x5000) ]) );
      ( "an unmap whose first address is not mapped",
        fun t -> Page_table.unmap t ~va:(va - page) 0x2000 );
      ( "an unmap whose last address is not mapped",
        fun t -> Page_table.unmap t ~va 0x4000 );
      ( "an unmap of nothing mapped",
        fun t -> Page_table.unmap t ~va:(va + 0x3000) page );
      ( "an unmap of half a 2 MiB page",
        fun t -> Page_table.unmap t ~va:large mib );
      ( "a map at an address off a page",
        fun t ->
          ignore (Page_table.map t ~va:(va + 0x3001) System [ (0, page) ]) );
      ( "a map of memory off a page",
        fun t ->
          ignore (Page_table.map t ~va:(va + 0x3000) System [ (0x800, page) ])
      );
      ( "a map of part of a page",
        fun t ->
          ignore (Page_table.map t ~va:(va + 0x3000) System [ (0, 0x800) ]) );
      ( "a map past the tables' reach",
        fun t ->
          let last = base + Page_table.span t - page in
          ignore (Page_table.map t ~va:last System [ (0, 2 * page) ]) );
      ( "an unmap below the base",
        fun t -> Page_table.unmap t ~va:(base - page) page );
      ( "the tables of an address off a page",
        fun t -> ignore (Page_table.tables t ~va:(va + 0x3001) page) );
      ( "the tables of a range past the tables' reach",
        fun t ->
          let last = base + Page_table.span t - page in
          ignore (Page_table.tables t ~va:last (2 * page)) );
      ( "the tables of a page inside a 2 MiB page",
        fun t -> ignore (Page_table.tables t ~va:(large + page) page) );
    ]
    (fun (_, f) ->
      let t, g = tables () in
      ignore
        (require_some (Page_table.map t ~va System [ (0x40_0000, 0x3000) ]));
      ignore
        (require_some
           (Page_table.map t ~va:large System [ (0x80_0000, 2 * mib) ]));
      let before = pages g t in
      raises_match (Exn.invalid_arg ~substring:"") (fun () -> f t);
      equal ~msg:"the tables as they were" (list entry) before (pages g t))

(* Out of memory for tables *)

(* A 4 KiB page mapped in each 512 GiB region from the third on takes three
   tables of its own. *)
let far k = base + ((k + 2) lsl 39)

let fill t =
  let rec go k acc =
    match Page_table.map t ~va:(far k) System [ (0x10_0000, page) ] with
    | Some m -> go (k + 1) (m :: acc)
    | None -> (k, List.rev acc)
  in
  go 0 []

let test_out_of_tables () =
  let t, g = tables ~tables:Pool ~length:(64 * mib) () in
  let free = capacity t and addresses = space_capacity t in
  let k, mapped = fill t in
  let want =
    List.concat_map
      (fun (m : Page_table.mapping) -> expect_mapping ~base m)
      mapped
  in
  greater ~msg:"mappings before the pool ran out" int ~than:10 k;
  equal ~msg:"a failed map leaves no entry" (list placed) want (pages g t);
  equal ~msg:"and no table" int
    (1 + (3 * List.length mapped))
    (List.length (snd (walk g t)));
  is_none ~msg:"tables refuses too" (Page_table.tables t ~va:(far k) page);
  for _ = 1 to 20 do
    is_none ~msg:"alloc" (Page_table.alloc t page)
  done;
  equal ~msg:"a failed alloc gives back its memory" int free (capacity t);
  equal ~msg:"and its addresses" int addresses (space_capacity t);
  (* Two mappings' tables, so that the freed range passes the fit bound. *)
  List.iter
    (fun (m : Page_table.mapping) -> Page_table.unmap t ~va:m.va m.size)
    (List.filteri (fun i _ -> i < 2) mapped);
  is_some ~msg:"unmapping gives tables back"
    (Page_table.map t ~va:(far k) System [ (0x10_0000, page) ])

(* With no pool of their own, tables come from the main pool. *)
let test_out_of_main () =
  let t, g = tables ~tables:Main () in
  let rec drain acc =
    match Page_table.palloc ~zero:false t page with
    | Some pa -> drain (pa :: acc)
    | None -> acc
  in
  let taken = drain [] in
  is_none ~msg:"no room for a table"
    (Page_table.map t ~va:(far 0) System [ (0x10_0000, page) ]);
  equal ~msg:"nothing mapped" (list entry) [] (pages g t);
  List.iter (Page_table.pfree t) taken;
  is_some ~msg:"room again"
    (Page_table.map t ~va:(far 0) System [ (0x10_0000, page) ])

(* A map that runs out of tables after making one frees it: the main pool has 8
   KiB free, room for one table, and the region needs three. *)
let test_failed_descent () =
  let t, g = tables ~tables:Main () in
  let rec drain acc =
    match Page_table.palloc ~zero:false t page with
    | Some pa -> drain (pa :: acc)
    | None -> acc
  in
  let taken = drain [] in
  let held a = List.mem a taken in
  (* Two blocks in a row between two others, so the 8 KiB has no free
     neighbour. *)
  let a =
    List.find
      (fun a -> held (a - page) && held (a + page) && held (a + (2 * page)))
      taken
  in
  Page_table.pfree t a;
  Page_table.pfree t (a + page);
  is_none ~msg:"no room for three tables"
    (Page_table.map t ~va:(far 0) System [ (0x10_0000, page) ]);
  equal ~msg:"no table but the root" (list hex)
    [ Page_table.root t ]
    (snd (walk g t));
  is_some ~msg:"the 8 KiB free again" (Page_table.palloc ~zero:false t page)

(* The tables in use are not blocks [pfree] frees. *)
let test_pfree_tables () =
  let t, g = tables () in
  ignore (require_some (Page_table.map t ~va:(far 0) System [ (0, page) ]));
  let mapped = pages g t in
  List.iter
    (fun table ->
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Page_table.pfree t table))
    (snd (walk g t));
  equal ~msg:"the tables as they were" (list entry) mapped (pages g t)

(* Map and unmap many times over, in more regions than the tables' pool holds
   tables for at once: unmapping frees the tables it empties. *)
let test_tables_freed () =
  let t, _ = tables ~tables:Pool () in
  for k = 0 to 299 do
    let va = far k in
    ignore
      (require_some ~msg:"mapped" (Page_table.map t ~va System [ (0, page) ]));
    Page_table.unmap t ~va page
  done

(* One range that maps a 2 MiB page and then a full table of 4 KiB pages. *)
let test_mixed_unmap () =
  let t, g = tables ~memory:(4 * mib) () in
  let free = capacity t in
  let ranges = [ (0, (2 * mib) + page); (0x1000_0000, (2 * mib) - page) ] in
  ignore (require_some (Page_table.map t ~va:base Gpu ranges));
  Page_table.unmap t ~va:base (4 * mib);
  equal ~msg:"no table but the root" (list hex)
    [ Page_table.root t ]
    (snd (walk g t));
  equal ~msg:"the tables went back to the pool" int free (capacity t)

(* Pages of 4 KiB at three places in two 2 MiB regions, and a 2 MiB page, in
   each of two 1 GiB regions: slots [(g, m, Some p)] and [(g, 2, None)]. Some of
   them, mapped one by one and unmapped in any order. *)
let slots =
  List.concat_map
    (fun g ->
      (g, 2, None)
      :: List.concat_map
           (fun m -> List.map (fun p -> (g, m, Some p)) [ 0; 1; 511 ])
           [ 0; 1 ])
    [ 0; 1 ]

let pp_slot ppf (g, m, p) =
  match p with
  | Some p -> Format.fprintf ppf "page %d of 2 MiB %d of 1 GiB %d" p m g
  | None -> Format.fprintf ppf "2 MiB %d of 1 GiB %d" m g

let unmap_orders =
  let pp_slots = Format.pp_print_list ~pp_sep:Format.pp_print_space pp_slot in
  let pp ppf (mapped, order) =
    Format.fprintf ppf "@[<v>mapped @[%a@]@,unmapped @[%a@]@]" pp_slots mapped
      pp_slots order
  in
  let open Gen in
  with_pp pp
    (let* mapped = subsequence ~pp:pp_slot slots in
     let+ order = permutation ~pp:pp_slot mapped in
     (mapped, order))

(* The tables that the mapped slots need: the root, a table under it, one per 1
   GiB region and one per 2 MiB region of 4 KiB pages. *)
let needed mapped =
  let count l = List.length (List.sort_uniq compare l) in
  let regions = count (List.map (fun (g, _, _) -> g) mapped) in
  let tables =
    count
      (List.filter_map (fun (g, m, p) -> Option.map (fun _ -> (g, m)) p) mapped)
  in
  1 + Int.min 1 regions + regions + tables

(* A table is freed when the last entry under it is cleared, and not before; an
   unmap reads no table whole: it touches a few entries a level. *)
let test_last_entry =
  prop "a table is freed exactly when its last entry goes" ~count:200
    unmap_orders (fun (mapped, order) ->
      let t, g = tables ~memory:(4 * mib) () in
      let free = capacity t in
      let va (gi, m, p) =
        base + (gi * gib) + (m * 2 * mib) + (Option.value p ~default:0 * page)
      in
      let size (_, _, p) = if Option.is_some p then page else 2 * mib in
      List.iteri
        (fun i s ->
          ignore
            (require_some
               (Page_table.map t ~va:(va s) System [ (i * 2 * mib, size s) ])))
        mapped;
      equal ~msg:"tables mapped" int (needed mapped)
        (List.length (snd (walk g t)));
      let rec go left = function
        | [] -> ()
        | s :: rest ->
            let left = List.filter (( <> ) s) left in
            let touches = g.touches in
            Page_table.unmap t ~va:(va s) (size s);
            less ~msg:"entries the unmap touched" int ~than:(4 * 4)
              (g.touches - touches);
            cover "an unmap that frees a table"
              (needed left < needed (s :: left));
            cover "an unmap that frees none" (needed left = needed (s :: left));
            equal ~msg:"tables left" int (needed left)
              (List.length (snd (walk g t)));
            go left rest
      in
      go mapped order;
      equal ~msg:"the tables went back to the pool" int free (capacity t))

(* The tables of an address *)

let test_tables_path =
  cases
    ~name:(fun (what, _, _, _) -> what)
    "tables"
    [
      ("1 GiB at 1 GiB", gib, gib, 2);
      ("2 MiB at 2 MiB", 2 * mib, 2 * mib, 3);
      ("4 KiB at 2 MiB", 2 * mib, page, 4);
      ("4 KiB at 4 KiB", (2 * mib) + page, page, 4);
    ]
    (fun (_, offset, n, depth) ->
      let t, g = tables () in
      let va = base + gib + offset in
      let path = require_some (Page_table.tables t ~va n) in
      equal ~msg:"depth" int depth (List.length path);
      equal ~msg:"root first" hex (Page_table.root t) (List.hd path);
      equal ~msg:"the same tables again" (list hex) path
        (require_some (Page_table.tables t ~va n));
      equal ~msg:"tables made, nothing mapped" (list entry) [] (pages g t);
      ignore (require_some (Page_table.map t ~va System [ (offset, n) ]));
      equal ~msg:"the tables map uses" (list hex) path (snd (walk g t));
      Page_table.unmap t ~va n;
      equal ~msg:"never freed" (list hex) path (snd (walk g t)))

(* Real formats *)

(* The layouts of real GPUs' tables, levels numbered from [first], and each
   level's entry count, root first, as their specifications give it: - AMD GFX9
   on: 48 bits, four levels of 9 bits over 4 KiB pages; the root, PDB2, numbered
   1, holds 512 entries (Linux amdgpu's amdgpu_vm_pt_num_entries over the 256
   TiB space gmc_v9_0 sets). - NVIDIA MMU version 2, Pascal to Ada: 49 bits; PD3
   indexes bits 48-47 and PD0 bits 28-21 (open-gpu-kernel-modules,
   kern_gmmu_fmt_gp10x.c). - NVIDIA MMU version 3, Hopper on: 57 bits; PD4
   indexes bit 56 (kern_gmmu_fmt_gh10x.c). *)
let real_formats =
  [
    ("AMD", 1, [ 12; 21; 30; 39 ], 48, [ 512; 512; 512; 512 ]);
    ("NVIDIA MMU v2", 0, [ 12; 21; 29; 38; 47 ], 49, [ 4; 512; 512; 256; 512 ]);
    ( "NVIDIA MMU v3",
      0,
      [ 12; 21; 29; 38; 47; 56 ],
      57,
      [ 2; 512; 512; 512; 256; 512 ] );
  ]

(* A map of the last page the tables reach writes the last entry of each level,
   numbered as the format numbers them; the address past it is refused. *)
let test_real_formats =
  cases
    ~name:(fun (name, _, _, _, _) -> name)
    "real formats" real_formats
    (fun (_, first, levels, bits, counts) ->
      let mem = Hashtbl.create 16 and writes = ref [] in
      let bottom = first + List.length levels - 1 in
      let set ~level ~table i e =
        writes := (level, i) :: !writes;
        Hashtbl.replace mem (table + (8 * i)) e
      in
      let fmt =
        {
          Page_table.levels;
          bits;
          first;
          get =
            (fun ~level ~table i : Page_table.entry ->
              match Hashtbl.find_opt mem (table + (8 * i)) with
              | None | Some 0 -> Invalid
              | Some _ when level = bottom -> Page
              | Some e -> Table (e land address_mask));
          set_table =
            (fun ~level ~table i ~child -> set ~level ~table i (child lor 1));
          set_page =
            (fun ~level ~table i ~pa _ ~uncached:_ ~snooped:_ ~fragment:_ ->
              set ~level ~table i (pa lor 1));
          clear = (fun ~level ~table i -> set ~level ~table i 0);
          large = (fun ~level -> level = bottom);
          zero =
            (fun pa n ->
              Hashtbl.filter_map_inplace
                (fun k e -> if k >= pa && k < pa + n then None else Some e)
                mem);
          flush = ignore;
        }
      in
      let t =
        Page_table.create ~base:0 fmt (Space.create ~base:0 gib)
          ~memory:(66 * mib) ~boot:mib ~tables:Main
          ~pages:[ (page, page) ]
      in
      Page_table.booted t;
      equal ~msg:"span" hex (1 lsl bits) (Page_table.span t);
      let last = Page_table.span t - page in
      ignore (require_some (Page_table.map t ~va:last Gpu [ (0, page) ]));
      equal ~msg:"the last entry of each level, root first"
        (list (pair int int))
        (List.mapi (fun d n -> (first + d, n - 1)) counts)
        (List.rev !writes);
      raises_match (Exn.invalid_arg ~substring:"") (fun () ->
          Page_table.map t ~va:(last + page) Gpu [ (0, page) ]))

(* Booting *)

(* Until booted, tables come from the boot pool. *)
let test_booting () =
  let t, g = tables ~tables:Pool ~booted:false () in
  ignore (require_some (Page_table.map t ~va:(far 0) System [ (0, page) ]));
  List.iter
    (fun table -> less ~msg:"a table in the boot pool" hex ~than:mib table)
    (snd (walk g t));
  let pa = require_some (Page_table.palloc t page) in
  less ~msg:"physical memory from the boot pool" hex ~than:mib pa;
  let m = require_some (Page_table.alloc t page) in
  List.iter
    (fun (pa, _) -> less ~msg:"allocations from the boot pool" hex ~than:mib pa)
    m.pages;
  Page_table.booted t;
  let pa = require_some (Page_table.palloc t page) in
  at_least ~msg:"then from the main pool, after the tables' 1 MiB" hex
    ~than:(2 * mib) pa

(* Allocations *)

let sizes ranges = List.map snd ranges
let sum = List.fold_left ( + ) 0

let test_contiguous =
  cases
    ~name:(fun (n, _) -> Printf.sprintf "%d bytes" n)
    "a contiguous allocation"
    [
      (1, page); (3 * mib, 2 * mib); (100 * kib, 64 * kib); (64 * kib, 64 * kib);
    ]
    (fun (n, align) ->
      let t, g = tables () in
      g.zeroed <- [];
      let m = require_some (Page_table.alloc ~contiguous:true t n) in
      let size = round_up n page in
      equal ~msg:"rounded up to 4 KiB" int size m.size;
      let pa, len =
        match m.pages with
        | [ p ] -> p
        | ps -> failf "%d blocks" (List.length ps)
      in
      equal ~msg:"one block" int size len;
      equal ~msg:"aligned as its largest page" hex 0 (pa land (align - 1));
      zeroed ~msg:"zeroed" g.zeroed (pa, size);
      equal ~msg:"mapped" (list placed) (expect_mapping ~base m) (pages g t))

(* Without [contiguous], 4 MiB and three pages are two 2 MiB blocks, which map
   as 2 MiB pages, and 4 KiB ones. *)
let test_blocks () =
  let t, g = tables () in
  let n = (4 * mib) + (3 * page) in
  let m = require_some (Page_table.alloc ~uncached:true t n) in
  equal ~msg:"size" int n m.size;
  equal ~msg:"blocks cover it" int n (sum (sizes m.pages));
  equal ~msg:"uncached" bool true m.uncached;
  equal ~msg:"the GPU's memory" target Gpu m.target;
  let ps = pages g t in
  equal ~msg:"entries" (list placed) (expect_mapping ~base m) ps;
  equal ~msg:"levels" (list int) [ 2; 2; 3; 3; 3 ]
    (List.map (fun e -> e.level) ps)

let test_free () =
  let t, g = tables ~length:(64 * mib) () in
  let free = capacity t and addresses = space_capacity t in
  let m = require_some (Page_table.alloc t (5 * mib)) in
  let c = require_some (Page_table.alloc ~contiguous:true t (3 * mib)) in
  Page_table.free t m;
  Page_table.free t c;
  equal ~msg:"nothing mapped" (list entry) [] (pages g t);
  equal ~msg:"memory back in the pool" int free (capacity t);
  equal ~msg:"addresses back in the space" int addresses (space_capacity t);
  let s = Page_table.space t in
  let va = require_some (Space.alloc s (2 * mib)) in
  let taken = require_some (Page_table.palloc ~zero:false t page) in
  let sys = require_some (Page_table.map t ~va System [ (taken, page) ]) in
  Page_table.free t sys;
  equal ~msg:"system memory: addresses back" int addresses (space_capacity t);
  Page_table.pfree t taken

(* Physical pools, against lists of the blocks they handed out *)

type pool = {
  lo : int;
  hi : int;
  mutable live : (int * int) list;  (** (address, bytes), newest first. *)
  mutable freed : int list;
}

let pool lo hi = { lo; hi; live = []; freed = [] }
let largest_gap p = largest_gap p.lo p.hi p.live

let apart ~msg blocks (a, n) =
  List.iter
    (fun (b, k) ->
      if not (a + n <= b || b + k <= a) then
        failf "%s: [0x%x, +0x%x) overlaps [0x%x, +0x%x)" msg a n b k)
    blocks

(* A block from the pool, aligned, apart from the pool's other blocks. *)
let in_pool ~msg p ~align (a, n) =
  equal ~msg:(msg ^ ": aligned") hex 0 (a land (align - 1));
  at_least ~msg:(msg ^ ": from the pool") hex ~than:p.lo a;
  at_most ~msg:(msg ^ ": inside the pool") hex ~than:(p.hi - n) a;
  apart ~msg p.live (a, n)

let take p (a, n) =
  p.live <- (a, n) :: p.live;
  p.freed <- List.filter (( <> ) a) p.freed

let give p a =
  p.live <- List.remove_assoc a p.live;
  p.freed <- a :: p.freed

(* A boot pool of 64 KiB, a tables' pool of 1 MiB or none, and the rest. *)
let small_memory = mib + (64 * kib) + (256 * kib)
let small_boot = 64 * kib

type pools = { boot : pool; main : pool; root : int; mutable booting : bool }

(* The root table, which the system's [create] places before the model's
   runs. *)
let made_root = ref 0

let create_pools tables =
  let main =
    small_memory - small_boot
    - match tables with Page_table.Pool -> mib | Main -> 0
  in
  let boot = pool 0 small_boot in
  boot.live <- [ (!made_root, page) ];
  {
    boot;
    main = pool (small_memory - main) small_memory;
    root = !made_root;
    booting = true;
  }

let create_tables kind =
  let t, g =
    tables ~tables:kind ~memory:small_memory ~boot:small_boot
      ~pages:[ (page, page) ]
      ~length:(1 lsl 30) ~booted:false ()
  in
  made_root := Page_table.root t;
  (t, g)

let palloc_judge align zero boot n m got =
  let p = if Option.value boot ~default:m.booting then m.boot else m.main in
  let align = Option.value align ~default:page in
  (* Past the pool, [round_up] could wrap: no block fits. *)
  let size = if n > p.hi - p.lo then max_int else round_up n page in
  let refused = n <= 0 || not (is_pow2 align) in
  let gap = largest_gap p in
  cover "a refused request" refused;
  cover "the boot pool" (p == m.boot);
  cover "the main pool by default" (p == m.main && boot = None);
  cover "a pool without room"
    (match got with Ok (None, _) -> true | _ -> false);
  cover "aligned beyond 4 KiB" (align > page);
  cover "memory handed out again after a free"
    (match got with Ok (Some a, _) -> List.mem a p.freed | _ -> false);
  match got with
  | Error (Invalid_argument _) when refused -> ()
  | Error e -> raise e
  | Ok _ when refused ->
      failf "a block for %d bytes aligned to %d, which raise Invalid_argument" n
        align
  | Ok (None, _) ->
      if fits ~gap size align then
        failf "None for %d bytes aligned to 0x%x with %d free in a row" size
          align gap
  | Ok (Some a, zs) ->
      in_pool ~msg:"block" p ~align (a, size);
      if Option.value zero ~default:true then zeroed ~msg:"zeroed" zs (a, size);
      List.iter (apart ~msg:"zeroing" (m.boot.live @ m.main.live)) zs;
      take p (a, size)

let pfree m a =
  let owns p = a <> m.root && List.mem_assoc a p.live in
  if owns m.boot then give m.boot a
  else if owns m.main then give m.main a
  else invalid_arg "Page_table.pfree"

let pools =
  abstract "t" ~invariant:(fun m (t, _) ->
      equal ~msg:"the GPU's memory" int small_memory (Page_table.memory t);
      equal ~msg:"root" hex m.root (Page_table.root t))

let blocks m =
  List.filter (( <> ) m.root) (List.map fst (m.boot.live @ m.main.live))

let block = among hex pools blocks

(* Addresses [pfree] must refuse, beside live ones: inside a block, freed
   already, the root table, past memory. *)
let not_blocks =
  among hex pools (fun m ->
      List.map (fun a -> a + page) (blocks m)
      @ m.boot.freed @ m.main.freed
      @ [ m.root; small_memory; -1 ])

(* Sizes at the fit bound of the pool [palloc] takes from by default. *)
let pool_edges =
  among int pools (fun m ->
      let g = largest_gap (if m.booting then m.boot else m.main) in
      List.filter (fun n -> n > 0) [ (g / 2) - page; (g / 2) - page + 1; g / 4 ])

let opt pp =
  Format.pp_print_option ~none:(fun ppf () -> Format.pp_print_string ppf "_") pp

let palloc_sizes =
  Gen.frequency
    [
      (4, Gen.int_range 1 (64 * kib));
      (2, Gen.map (fun k -> k * page) (Gen.int_range 1 32));
      ( 1,
        Gen.of_list ~pp:pp_hex
          [ 1; 4095; 4096; 4097; 256 * kib; 0; -1; min_int; max_int ] );
    ]

let palloc_aligns =
  Gen.of_list ~pp:(opt pp_hex)
    [
      None;
      None;
      None;
      Some 1;
      Some page;
      Some (64 * kib);
      Some (2 * mib);
      Some 0;
      Some 0x3000;
      Some (-page);
    ]

let bools =
  Gen.of_list ~pp:(opt Format.pp_print_bool) [ None; Some true; Some false ]

let palloc align zero boot n (t, g) =
  g.zeroed <- [];
  let a = Page_table.palloc ?align ?zero ?boot t n in
  (a, g.zeroed)

let palloc_commands =
  let outcome = pair (option hex) (list (pair hex int)) in
  [
    command "create"
      (Gen.of_list
         ~pp:(fun ppf t ->
           Format.pp_print_string ppf
             (match t with Page_table.Pool -> "Pool" | Main -> "Main"))
         [ Page_table.Pool; Main ]
      @-> makes pools)
      create_pools create_tables;
    command "palloc"
      (palloc_aligns @-> bools @-> bools @-> palloc_sizes @-> pools
     ^-> judges outcome)
      palloc_judge palloc;
    command "palloc"
      (pools ^-> pool_edges ^-> judges outcome)
      (fun m n got -> palloc_judge None None None n m got)
      (fun tg n -> palloc None None None n tg);
    command "pfree"
      (pools ^-> block ^-> returns unit)
      pfree
      (fun (t, _) a -> Page_table.pfree t a);
    command "pfree"
      (pools ^-> not_blocks ^-> returns unit)
      pfree
      (fun (t, _) a -> Page_table.pfree t a);
    command "booted"
      (pools ^-> returns unit)
      (fun m -> m.booting <- false)
      (fun (t, _) -> Page_table.booted t);
  ]

(* Allocations, against the blocks, addresses and mappings they took *)

(* A boot pool and a tables' pool of 1 MiB each, a main pool of 1 MiB, a space
   of 4 MiB. *)
let alloc_memory = 3 * mib
let alloc_space = 4 * mib

type allocs = {
  space : pool;
  phys : pool;
  mutable maps : Page_table.mapping list;
}

let create_allocs () =
  {
    space = pool base (base + alloc_space);
    phys = pool (alloc_memory - mib) alloc_memory;
    maps = [];
  }

let create_alloc_tables () =
  tables ~tables:Pool ~memory:alloc_memory ~boot:mib
    ~pages:[ (page, page) ]
    ~length:alloc_space ()

(* An allocation takes fresh addresses and fresh blocks of the main pool, one
   zeroed block if contiguous. [None] is accepted where the space or the pool
   holds no free range of twice the request and its alignment: the request a
   single block would make. *)
let alloc_judge contiguous uncached n m got =
  let size = round_up n page and refused = n <= 0 in
  cover "a refused request" refused;
  let space_fits =
    fits ~gap:(largest_gap m.space) size (max page (pow2_floor size))
  in
  let pool_fits = fits ~gap:(largest_gap m.phys) size page in
  cover "out of addresses" (not space_fits);
  cover "out of memory" (not pool_fits);
  cover "several blocks"
    (match got with
    | Ok (Some ((a : Page_table.mapping), _)) -> List.length a.pages > 1
    | _ -> false);
  match got with
  | Error (Invalid_argument _) when refused -> ()
  | Error e -> raise e
  | Ok _ when refused -> failf "an allocation of %d bytes" n
  | Ok None ->
      if space_fits && pool_fits then
        failf "None for %d bytes with room in the space and the pool" size
  | Ok (Some (a, zs)) ->
      equal ~msg:"size" int size a.size;
      equal ~msg:"uncached" bool uncached a.uncached;
      equal ~msg:"snooped" bool false a.snooped;
      equal ~msg:"the GPU's memory" target Gpu a.target;
      in_pool ~msg:"addresses" m.space ~align:1 (a.va, size);
      equal ~msg:"blocks cover it" int size (sum (sizes a.pages));
      if contiguous then equal ~msg:"one block" int 1 (List.length a.pages);
      List.iter
        (fun (pa, k) ->
          in_pool ~msg:"block" m.phys ~align:page (pa, k);
          take m.phys (pa, k);
          if contiguous then zeroed ~msg:"zeroed" zs (pa, k))
        a.pages;
      take m.space (a.va, size);
      m.maps <- a :: m.maps

let free_alloc m (a : Page_table.mapping) =
  m.maps <- List.filter (( != ) a) m.maps;
  give m.space a.va;
  List.iter (fun (pa, _) -> give m.phys pa) a.pages

let allocs =
  abstract "t" ~invariant:(fun m (t, g) ->
      let want =
        List.concat_map (expect_mapping ~base) m.maps
        |> List.sort (fun a b -> compare a.va b.va)
      in
      equal ~msg:"entries" (list placed) want (pages g t);
      equal ~msg:"flushed" int 0 g.unflushed)

let mapping =
  Testable.make
    ~pp:(fun ppf (m : Page_table.mapping) ->
      Format.fprintf ppf "0x%x+0x%x" m.va m.size)
    ~equal:( == )

let live_maps = among mapping allocs (fun m -> m.maps)

let alloc_edges =
  among int allocs (fun m ->
      [ (largest_gap m.phys / 2) - page; largest_gap m.space / 4 ]
      |> List.filter (fun n -> n > 0))

let alloc_sizes =
  Gen.frequency
    [
      (4, Gen.int_range 1 (64 * kib));
      (2, Gen.map (fun k -> k * page) (Gen.int_range 1 64));
      (1, Gen.of_list ~pp:pp_hex [ 1; 4096; 4097; 256 * kib; mib; 0; -1 ]);
    ]

let alloc contiguous uncached n (t, g) =
  g.zeroed <- [];
  let m = Page_table.alloc ~contiguous ~uncached t n in
  Option.map (fun m -> (m, g.zeroed)) m

let alloc_commands =
  let outcome = option (pair mapping (list (pair hex int))) in
  [
    command "create"
      (Gen.unit @-> makes allocs)
      create_allocs create_alloc_tables;
    command "alloc"
      (Gen.bool @-> Gen.bool @-> alloc_sizes @-> allocs ^-> judges outcome)
      alloc_judge alloc;
    command "alloc"
      (Gen.bool @-> allocs ^-> alloc_edges ^-> judges outcome)
      (fun contiguous m n got -> alloc_judge contiguous false n m got)
      (fun contiguous tg n -> alloc contiguous false n tg);
    command "free"
      (allocs ^-> live_maps ^-> returns unit)
      free_alloc
      (fun (t, _) a -> Page_table.free t a);
  ]

let () =
  exit
  @@ run "device_pci Page_table"
       [
         group ~timeout:patience "create"
           [
             test "names its space, base, span and root" test_create;
             test_main_pool;
             test_create_refusals;
             test_level_refusals;
             test_base_refusals;
           ];
         group ~timeout:patience "map"
           [
             test_map;
             test "1 GiB pages, and 2 MiB ones where memory is aligned to 2 MiB"
               test_gib_pages;
             test "a 2 MiB page spans ranges that follow each other"
               test_page_across_ranges;
             test_last_entry;
             test_fragments;
             test_fragment_cases;
             test "fragments align in the space's addresses"
               test_fragments_from_base;
             test_map_refusals;
             test_tables_path;
           ];
         group ~timeout:patience "table memory"
           [
             test "a full tables' pool answers None and keeps what it had"
               test_out_of_tables;
             test "a full main pool answers None" test_out_of_main;
             test "a map out of tables frees those it made" test_failed_descent;
             test "pfree refuses the tables in use" test_pfree_tables;
             test "unmapping frees the tables it empties" test_tables_freed;
             test "unmapping 2 MiB and 4 KiB pages frees their tables"
               test_mixed_unmap;
             test "booting takes tables and memory from the boot pool"
               test_booting;
           ];
         group ~timeout:patience "real formats" [ test_real_formats ];
         group ~timeout:patience "physical memory"
           [
             stateful "blocks stay apart, zeroed and within the fit bound"
               ~count:300 palloc_commands;
           ];
         group ~timeout:patience "alloc"
           [
             test_contiguous;
             test "large blocks map as large pages" test_blocks;
             test "free gives back memory and addresses" test_free;
             stateful "the tables map exactly what is allocated" ~count:200
               alloc_commands;
           ];
       ]
