(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Errors *)

let[@inline never] err_mapped va =
  invalid_argf "Page_table.map: 0x%x is mapped already" va

let[@inline never] err_unmapped va =
  invalid_argf "Page_table.unmap: 0x%x is not mapped" va

type target = Gpu | System | Peer of int

type format = {
  levels : int list;
  bits : int;
  first : int;
  set_table : level:int -> table:int -> int -> child:int -> unit;
  set_page :
    level:int ->
    table:int ->
    int ->
    pa:int ->
    target ->
    uncached:bool ->
    snooped:bool ->
    fragment:int ->
    unit;
  clear : level:int -> table:int -> int -> unit;
  large : level:int -> bool;
  zero : int -> int -> unit;
  flush : unit -> unit;
}

type tables = Pool | Main

type mapping = {
  va : int;
  size : int;
  pages : (int * int) list;
  target : target;
  uncached : bool;
  snooped : bool;
}

(* The GPU's page: what the leaf level maps, and the unit of a fragment. *)
let page_bits = 12
let page = 1 lsl page_bits

(* Tables by physical address, a multiple of a page: its page number hashes it,
   cheaper than the generic hash that a walk would pay at each table. *)
module Held = Hashtbl.Make (struct
  type t = int

  let equal = Int.equal
  let hash pa = pa lsr page_bits
end)

(* The tables as this module wrote them, a tree beside the GPU's that a walk
   follows: the GPU's copy is never read back, since each read crosses the bus.
   A table is a directory, whose entries are invalid, map a page or point to a
   table, or a leaf, which keeps a bit per entry, set where it maps a page. Each
   is known by its physical address and counts its valid entries, which tells
   that it is empty at once. *)
type table = Directory of int * directory | Leaf of int * leaf
and entry = Invalid | Page | Table of table
and directory = { mutable valid : int; entries : entry array }
and leaf = { mutable pages : int; bits : Bytes.t }

let address = function Directory (pa, _) | Leaf (pa, _) -> pa

type t = {
  fmt : format;
  space : Space.t;
  shifts : int array; (* the address bit each level indexes, root first *)
  counts : int array; (* the entries of a table, root first *)
  boot : Tlsf.t;
  tables : Tlsf.t; (* empty unless the tables have a pool *)
  main : Tlsf.t;
  pages : (int * int) list; (* block sizes and alignments, largest first *)
  held : unit Held.t; (* the tables in use, the root among them *)
  spare : entry array list array;
      (* by depth, the entries of directories freed, all invalid: a directory's
         4 KiB is dear to allocate, and tables come and go with mappings *)
  mutable booting : bool;
  root : table;
  base : int;
  memory : int;
}

(* A table pool per GPU memory of this many bytes, rounded up to
   [table_round]. *)
let table_share = 512
let table_round = 1 lsl 20
let round_up n a = (n + a - 1) / a * a
let is_pow2 n = n > 0 && n land (n - 1) = 0
let aligned x a = x land (a - 1) = 0

(* A table that cannot be allocated, inside a walk. *)
exception No_room

(* Physical memory *)

let pool t ~table =
  if t.booting then t.boot
  else if table && Tlsf.length t.tables > 0 then t.tables
  else t.main

let take t tlsf ?(align = page) ?(zero = true) n =
  if n > Tlsf.length tlsf then None
  else
    let n = round_up n page in
    match Tlsf.alloc ~align tlsf n with
    | Some pa as r ->
        if zero then t.fmt.zero pa n;
        r
    | None -> None

let palloc ?(align = page) ?zero ?boot t n =
  if n <= 0 || not (is_pow2 align) then
    invalid_argf
      "Page_table.palloc: %d bytes aligned to %d, expected more than 0 bytes \
       and a power of two"
      n align;
  let tlsf =
    match boot with
    | Some true -> t.boot
    | Some false -> t.main
    | None -> pool t ~table:false
  in
  take t tlsf ~align ?zero n

let pfree t pa =
  let refuse () = invalid_argf "Page_table.pfree: no block at 0x%x" pa in
  let inside a = pa >= Tlsf.base a && pa < Tlsf.base a + Tlsf.length a in
  if Held.mem t.held pa then refuse ();
  match List.find_opt inside [ t.boot; t.tables; t.main ] with
  | Some a -> ( try Tlsf.free a pa with Invalid_argument _ -> refuse ())
  | None -> refuse ()

(* Tables *)

(* A table's depth is its distance from the root. The format numbers its level
   from the root's, [first]. Addresses below are relative to the tables'
   base. *)
let level t d = t.fmt.first + d
let covers t d = 1 lsl t.shifts.(d)
let bottom t = Array.length t.shifts - 1

(* A table at [pa] of [n] invalid entries. *)
let empty ~leaf n pa =
  if leaf then Leaf (pa, { pages = 0; bits = Bytes.make ((n + 7) / 8) '\000' })
  else Directory (pa, { valid = 0; entries = Array.make n Invalid })

(* The table at [pa], at depth [d], whose entries are invalid. *)
let hold t d pa =
  Held.replace t.held pa ();
  match t.spare.(d) with
  | entries :: rest ->
      t.spare.(d) <- rest;
      Directory (pa, { valid = 0; entries })
  | [] -> empty ~leaf:(d = bottom t) t.counts.(d) pa

let new_table t d =
  match take t (pool t ~table:true) page with
  | None -> raise No_room
  | Some pa -> hold t d pa

let is_page l i =
  Char.code (Bytes.unsafe_get l.bits (i lsr 3)) land (1 lsl (i land 7)) <> 0

let set_bit l i on =
  let b = Char.code (Bytes.unsafe_get l.bits (i lsr 3))
  and m = 1 lsl (i land 7) in
  Bytes.unsafe_set l.bits (i lsr 3)
    (Char.unsafe_chr (if on then b lor m else b land lnot m))

(* Stops keeping [table], at depth [d], and frees it. *)
let drop t d table =
  (match table with
  | Directory (_, dir) -> t.spare.(d) <- dir.entries :: t.spare.(d)
  | Leaf _ -> ());
  Held.remove t.held (address table);
  pfree t (address table)

(* The table entry [i] of the directory [dir] at [pa], at depth [d], points to,
   made if missing. *)
let child t d pa dir i =
  match dir.entries.(i) with
  | Table c -> c
  | Page -> invalid_arg "Page_table.tables: a larger page maps the address"
  | Invalid ->
      let c = new_table t (d + 1) in
      (match
         t.fmt.set_table ~level:(level t d) ~table:pa i ~child:(address c)
       with
      | () -> ()
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          drop t (d + 1) c;
          Printexc.raise_with_backtrace e bt);
      dir.entries.(i) <- Table c;
      dir.valid <- dir.valid + 1;
      c

(* Calls [f i at lo hi] for each entry [i] of a table at depth [d], whose first
   entry maps [at], that overlaps [lo, hi): [at] is the first address the entry
   maps and [lo, hi) the part of the range inside it. *)
let each t d ~at lo hi f =
  let c = covers t d in
  if lo < hi then
    for i = (lo - at) / c to (hi - 1 - at) / c do
      let at = at + (i * c) in
      f i at (Int.max lo at) (Int.min hi (at + c))
    done

(* Raises if a page maps an address of [lo, hi). *)
let rec unmapped t d table ~at lo hi =
  match table with
  | Leaf (_, l) ->
      let c = covers t d in
      for i = (lo - at) / c to ((hi - at) / c) - 1 do
        if is_page l i then err_mapped (t.base + at + (i * c))
      done
  | Directory (_, dir) -> (
      each t d ~at lo hi @@ fun i at lo hi ->
      match dir.entries.(i) with
      | Invalid -> ()
      | Table c -> unmapped t (d + 1) c ~at lo hi
      | Page -> err_mapped (t.base + lo))

(* Raises unless pages map every address of [lo, hi), none past it. *)
let rec mapped t d table ~at lo hi =
  let c = covers t d in
  match table with
  | Leaf (_, l) ->
      for i = (lo - at) / c to ((hi - at) / c) - 1 do
        if not (is_page l i) then err_unmapped (t.base + at + (i * c))
      done
  | Directory (_, dir) -> (
      each t d ~at lo hi @@ fun i at lo hi ->
      match dir.entries.(i) with
      | Invalid -> err_unmapped (t.base + lo)
      | Table c -> mapped t (d + 1) c ~at lo hi
      | Page ->
          if lo <> at || hi <> at + c then
            invalid_argf "Page_table.unmap: the page at 0x%x is partly outside"
              (t.base + at))

(* The fragment of the page at [v] of the run [lo, hi), mapped [delta] bytes
   further: the log2 of the pages of the largest block naturally aligned in both
   address spaces, inside the run, that holds it, if at least [k]. *)
let rec fragment ~lo ~hi ~delta v k =
  let size = page lsl (k + 1) in
  let block = v land lnot (size - 1) in
  if block >= lo && block + size <= hi && aligned delta size then
    fragment ~lo ~hi ~delta v (k + 1)
  else k

(* The physical ranges a map writes, read in virtual order as one walk of the
   tables reaches them. The current run maps [lo, hi) to the addresses [delta]
   bytes further; ranges that follow each other in both address spaces are one
   run. Its pages below [until] share the fragment [frag]. The walk is at the
   entry that maps [reached]: it wrote those before it and none from it on. *)
type run = {
  target : target;
  uncached : bool;
  snooped : bool;
  mutable rest : (int * int) list;
  mutable lo : int;
  mutable hi : int;
  mutable delta : int;
  mutable frag : int;
  mutable until : int;
  mutable reached : int;
}

(* Moves [r] on to the run that holds [v]. *)
let rec seek r v =
  match r.rest with
  | (pa, n) :: rest when v >= r.hi ->
      r.rest <- rest;
      r.lo <- r.hi;
      r.hi <- r.hi + n;
      r.delta <- pa - r.lo;
      r.until <- r.lo;
      join r;
      seek r v
  | _ -> ()

(* Joins to [r]'s run the ranges that follow it, passing empty ones. *)
and join r =
  match r.rest with
  | (_, 0) :: rest ->
      r.rest <- rest;
      join r
  | (pa, n) :: rest when pa = r.hi + r.delta ->
      r.rest <- rest;
      r.hi <- r.hi + n;
      join r
  | _ -> ()

(* Maps the page at [v] of [r]'s run with entry [i] of [table], at depth [d]. A
   fragment's block is aligned in the space's addresses; the pages of a block
   share its fragment, which is found once a block. *)
let map_page t r d table i v =
  if v >= r.until then begin
    let va = t.base + v and k = t.shifts.(d) - page_bits in
    r.frag <-
      fragment ~lo:(t.base + r.lo) ~hi:(t.base + r.hi) ~delta:(r.delta - t.base)
        va k;
    r.until <- (va lor ((page lsl r.frag) - 1)) + 1 - t.base
  end;
  t.fmt.set_page ~level:(level t d) ~table i ~pa:(v + r.delta) r.target
    ~uncached:r.uncached ~snooped:r.snooped ~fragment:r.frag

(* Maps [lo, hi) to the runs of [r], each page with the largest entry that its
   run holds and both its addresses are aligned to, unless a table is there
   already, as [tables] may keep one. One walk serves every run: each table is
   visited once. The last level's entries are all whole pages. *)
let rec write t d table ~at lo hi r =
  let level = level t d and c = covers t d in
  match table with
  | Leaf (pa, l) ->
      for i = (lo - at) / c to ((hi - at) / c) - 1 do
        let v = at + (i * c) in
        r.reached <- v;
        seek r v;
        map_page t r d pa i v;
        set_bit l i true;
        l.pages <- l.pages + 1
      done
  | Directory (pa, dir) -> (
      each t d ~at lo hi @@ fun i at lo hi ->
      r.reached <- lo;
      seek r lo;
      let whole =
        lo = at && hi = at + c && hi <= r.hi && aligned (lo + r.delta) c
      in
      match dir.entries.(i) with
      | Invalid when whole && t.fmt.large ~level ->
          map_page t r d pa i lo;
          dir.entries.(i) <- Page;
          dir.valid <- dir.valid + 1
      | _ -> write t (d + 1) (child t d pa dir i) ~at lo hi r)

(* Clears the entries of [lo, hi) that map or point to something and frees the
   tables it empties: [true] iff [table] is then empty. *)
let rec clear t d table ~at lo hi =
  let level = level t d and c = covers t d in
  match table with
  | Leaf (pa, l) ->
      for i = (lo - at) / c to ((hi - at) / c) - 1 do
        if is_page l i then begin
          t.fmt.clear ~level ~table:pa i;
          set_bit l i false;
          l.pages <- l.pages - 1
        end
      done;
      l.pages = 0
  | Directory (pa, dir) ->
      let forget i =
        t.fmt.clear ~level ~table:pa i;
        dir.entries.(i) <- Invalid;
        dir.valid <- dir.valid - 1
      in
      ( each t d ~at lo hi @@ fun i at lo hi ->
        match dir.entries.(i) with
        | Invalid -> ()
        | Page -> forget i
        | Table c ->
            if clear t (d + 1) c ~at lo hi then begin
              forget i;
              drop t (d + 1) c
            end );
      dir.valid = 0

(* Page tables *)

let create ?base fmt space ~memory ~boot ~tables ~pages =
  let rec rising = function
    | a :: (b :: _ as rest) -> a < b && rising rest
    | _ -> true
  in
  if
    List.nth_opt fmt.levels 0 <> Some page_bits
    || fmt.bits >= Sys.int_size - 1
    || not (rising (fmt.levels @ [ fmt.bits ]))
  then invalid_arg "Page_table.create: levels must rise from 12 below bits";
  let table_bytes =
    match tables with
    | Pool -> round_up (memory / table_share) table_round
    | Main -> 0
  in
  if boot < 0 then invalid_argf "Page_table.create: boot %d is negative" boot;
  if boot + table_bytes > memory then
    invalid_argf
      "Page_table.create: %d boot and %d table bytes exceed the %d bytes of \
       memory"
      boot table_bytes memory;
  let rest = boot + table_bytes in
  let boot_pool = Tlsf.create ~base:0 boot in
  let root =
    match Tlsf.alloc ~align:page boot_pool page with
    | Some pa ->
        fmt.zero pa page;
        pa
    | None -> invalid_arg "Page_table.create: no boot memory for the root table"
  in
  let shifts = Array.of_list (List.rev fmt.levels) in
  let above d = if d = 0 then fmt.bits else shifts.(d - 1) in
  let base = Option.value base ~default:(Space.base space) in
  (* The largest page: aligned to it, [base] aligns pages and fragments in the
     tables as in the address space. *)
  let rec largest d =
    if d = Array.length shifts - 1 || fmt.large ~level:(fmt.first + d) then
      1 lsl shifts.(d)
    else largest (d + 1)
  in
  if base < 0 || not (aligned base (largest 0)) then
    invalid_argf "Page_table.create: base 0x%x off a page" base;
  let counts = Array.mapi (fun d s -> 1 lsl (above d - s)) shifts in
  let held = Held.create 64 in
  Held.replace held root ();
  {
    fmt;
    space;
    shifts;
    counts;
    boot = boot_pool;
    tables = Tlsf.create ~base:boot table_bytes;
    main = Tlsf.create ~base:rest (memory - rest);
    pages;
    held;
    spare = Array.make (Array.length shifts) [];
    booting = true;
    root = empty ~leaf:(Array.length shifts = 1) counts.(0) root;
    base;
    memory;
  }

let booted t = t.booting <- false
let root t = address t.root
let space t = t.space
let base t = t.base
let span t = 1 lsl t.fmt.bits
let memory t = t.memory

(* Raises unless the [n] bytes from [va] are whole pages the tables reach. *)
let check t fn ~va n =
  let v = va - t.base in
  if v < 0 || n < 0 || n > span t - v || not (aligned (v lor n) page) then
    invalid_argf
      "Page_table.%s: 0x%x bytes at 0x%x are not whole pages the tables reach"
      fn n va

let tables t ~va n =
  check t "tables" ~va n;
  let v = va - t.base in
  let held = Held.length t.held in
  (* Down to the table where [map] would write the entry of [v]. The caller may
     write entries of that table that its count does not see: one more keeps it,
     and so its path, from being freed. *)
  let rec go d table path =
    let path = address table :: path in
    match table with
    | Leaf (_, l) ->
        l.pages <- l.pages + 1;
        List.rev path
    | Directory (pa, dir) ->
        let c = covers t d in
        if t.fmt.large ~level:(level t d) && c <= n && aligned v c then begin
          dir.valid <- dir.valid + 1;
          List.rev path
        end
        else
          let i = (v lsr t.shifts.(d)) land (t.counts.(d) - 1) in
          go (d + 1) (child t d pa dir i) path
  in
  let path =
    match go 0 t.root [] with
    | p -> Ok p
    | exception e -> Error (e, Printexc.get_raw_backtrace ())
  in
  if Held.length t.held > held then t.fmt.flush ();
  match path with
  | Ok p -> Some p
  | Error (No_room, _) -> None
  | Error (e, bt) -> Printexc.raise_with_backtrace e bt

let unmap t ~va n =
  check t "unmap" ~va n;
  let lo = va - t.base in
  mapped t 0 t.root ~at:0 lo (lo + n);
  ignore (clear t 0 t.root ~at:0 lo (lo + n));
  t.fmt.flush ()

let map ?(uncached = false) ?(snooped = false) t ~va target ranges =
  List.iter
    (fun (pa, n) ->
      if pa < 0 || n < 0 || not (aligned (pa lor n) page) then
        invalid_argf
          "Page_table.map: 0x%x bytes at physical 0x%x are not whole pages" n pa)
    ranges;
  let size = List.fold_left (fun n (_, k) -> n + k) 0 ranges in
  check t "map" ~va size;
  let lo = va - t.base in
  unmapped t 0 t.root ~at:0 lo (lo + size);
  let r =
    {
      target;
      uncached;
      snooped;
      rest = ranges;
      lo;
      hi = lo;
      delta = 0;
      frag = 0;
      until = lo;
      reached = lo;
    }
  in
  match write t 0 t.root ~at:0 lo (lo + size) r with
  | () ->
      t.fmt.flush ();
      Some { va; size; pages = ranges; target; uncached; snooped }
  | exception e -> (
      let bt = Printexc.get_raw_backtrace () in
      (* Out of tables, or the format raised: clearing through the page at
         [reached] also frees the tables made on the way to it, which hold
         nothing. *)
      ignore (clear t 0 t.root ~at:0 lo (r.reached + page));
      t.fmt.flush ();
      match e with No_room -> None | e -> Printexc.raise_with_backtrace e bt)

(* Physical blocks for [n] bytes, the largest [pages] allows first, falling to
   smaller ones when the pool has none left. *)
let blocks t n =
  let pool = pool t ~table:false in
  let rec go acc left = function
    | _ when left = 0 -> Some (List.rev acc)
    | [] ->
        List.iter (fun (pa, _) -> pfree t pa) acc;
        None
    | (size, _) :: rest when size > left -> go acc left rest
    | (size, align) :: rest as sizes -> (
        match take t pool ~align ~zero:false size with
        | Some pa -> go ((pa, size) :: acc) (left - size) sizes
        | None -> go acc left rest)
  in
  go [] n t.pages

(* One block of [n] bytes, aligned as the largest block of [pages] it holds, so
   that it maps with the largest pages, when the pool has such a block. *)
let block t n =
  let pool = pool t ~table:false in
  let align =
    match List.find_opt (fun (size, _) -> size <= n) t.pages with
    | Some (_, align) -> align
    | None -> page
  in
  match take t pool ~align n with
  | None when align > page -> take t pool n
  | found -> found

let alloc ?(uncached = false) ?(contiguous = false) t n =
  if n <= 0 then
    invalid_argf "Page_table.alloc: %d bytes, expected more than 0" n;
  if n > Space.length t.space then None
  else
    let n = round_up n page in
    match Space.alloc t.space n with
    | None -> None
    | Some va -> (
        let pages =
          if contiguous then Option.map (fun pa -> [ (pa, n) ]) (block t n)
          else blocks t n
        in
        let give_back () =
          Option.iter (List.iter (fun (pa, _) -> pfree t pa)) pages;
          Space.free t.space va
        in
        match Option.bind pages (map ~uncached t ~va Gpu) with
        | Some _ as m -> m
        | None ->
            give_back ();
            None
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            give_back ();
            Printexc.raise_with_backtrace e bt)

let free t (m : mapping) =
  unmap t ~va:m.va m.size;
  Space.free t.space m.va;
  if m.target = Gpu then List.iter (fun (pa, _) -> pfree t pa) m.pages
