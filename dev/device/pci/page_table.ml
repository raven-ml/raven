(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type target = Gpu | System | Peer

type format = {
  levels : int list;
  bits : int;
  first : int;
  get : level:int -> table:int -> int -> int64;
  set : level:int -> table:int -> int -> int64 -> unit;
  encode :
    level:int ->
    table:bool ->
    target ->
    uncached:bool ->
    snooped:bool ->
    fragment:int ->
    valid:bool ->
    int ->
    int64;
  valid : int64 -> bool;
  leaf : level:int -> int64 -> bool;
  address : int64 -> int;
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

type t = {
  fmt : format;
  space : Space.t;
  covers : int array; (* bytes an entry covers, by level *)
  counts : int array; (* entries of a table, by level *)
  boot : Tlsf.t;
  tables : Tlsf.t; (* empty unless the tables have a pool *)
  main : Tlsf.t;
  pages : (int * int) list; (* block sizes and alignments, largest first *)
  mutable booting : bool;
  root : int;
  base : int;
}

let page = 0x1000

(* A table pool per GPU memory of this many bytes, rounded up to
   [table_round]. *)
let table_share = 512
let table_round = 1 lsl 20
let round_up n a = (n + a - 1) / a * a

(* A table that cannot be allocated, inside a walk. *)
exception No_room

(* Physical memory *)

let pool t ~table =
  if t.booting then t.boot
  else if table && Tlsf.length t.tables > 0 then t.tables
  else t.main

let take t tlsf ?(align = page) ?(zero = true) n =
  let n = round_up n page in
  match Tlsf.alloc ~align tlsf n with
  | Some pa as r ->
      if zero then t.fmt.zero pa n;
      r
  | None -> None

let palloc ?align ?zero ?boot t n =
  let tlsf =
    match boot with
    | Some true -> t.boot
    | Some false -> t.main
    | None -> pool t ~table:false
  in
  take t tlsf ?align ?zero n

let pfree t pa =
  let inside a = pa >= Tlsf.base a && pa < Tlsf.base a + Tlsf.length a in
  match List.find_opt inside [ t.boot; t.tables; t.main ] with
  | Some a -> (
      try Tlsf.free a pa
      with Invalid_argument _ ->
        invalid_arg (Printf.sprintf "Page_table.pfree: 0x%x" pa))
  | None -> invalid_arg (Printf.sprintf "Page_table.pfree: 0x%x" pa)

(* Walks *)

(* A walk visits the entries of a virtual range from [va] on, keeping the path
   from the root to the table it is in: each element is a table's physical
   address, its level, and the index of the next entry. *)
type walk = {
  mutable at : int; (* the virtual address, relative to the tables' base *)
  mutable path : (int * int * int) list;
  create : bool;
  free_tables : bool;
  inspect : bool;
}

let index t level va = va / t.covers.(level) mod t.counts.(level)

let walk t ?(create = false) ?(free_tables = false) ?(inspect = false) va =
  let at = va - t.base in
  let level = t.fmt.first in
  {
    at;
    path = [ (t.root, level, index t level at) ];
    create;
    free_tables;
    inspect;
  }

let top w = List.hd w.path

let invalid_entry t level =
  t.fmt.encode ~level ~table:false Gpu ~uncached:false ~snooped:false
    ~fragment:0 ~valid:false 0

(* Descends into the table the entry at the walk's position points to, creating
   it if the walk creates. *)
let down t w =
  let table, level, i = top w in
  let e = t.fmt.get ~level ~table i in
  if not (t.fmt.valid e) then begin
    if not w.create then
      invalid_arg "Page_table: an address of the range is not mapped";
    match take t (pool t ~table:true) page with
    | None -> raise No_room
    | Some pa ->
        t.fmt.set ~level ~table i
          (t.fmt.encode ~level ~table:true Gpu ~uncached:false ~snooped:false
             ~fragment:0 ~valid:true pa)
  end;
  let e = t.fmt.get ~level ~table i in
  if t.fmt.leaf ~level e then invalid_arg "Page_table: a page where a table was";
  let child = t.fmt.address e and level = level + 1 in
  w.path <- (child, level, index t level w.at) :: w.path

let empty t table level =
  let rec go i =
    i >= t.counts.(level)
    || ((not (t.fmt.valid (t.fmt.get ~level ~table i))) && go (i + 1))
  in
  go 0

(* Frees the table at the top of the path if it became empty. *)
let try_free t w =
  match w.path with
  | (table, level, _) :: (parent, plevel, pi) :: _
    when w.free_tables && empty t table level ->
      pfree t table;
      t.fmt.set ~level:plevel ~table:parent pi (invalid_entry t plevel);
      true
  | _ -> false

let rec up t w =
  let at_end () =
    let _, level, i = top w in
    i = t.counts.(level)
  in
  if try_free t w || at_end () then
    match w.path with
    | (_, level, i) :: (pt, plevel, pi) :: rest ->
        w.path <-
          (if i = t.counts.(level) then (pt, plevel, pi + 1) :: rest
           else (pt, plevel, pi) :: rest);
        up t w
    | _ -> ()

(* Visits the entries covering [size] bytes from the walk's position with [f off
   table level i n covers]: [n] entries from [i] of [table], [off] bytes into
   the range. Creating, it descends until an entry covers no more than the rest,
   pages may map at its level, and both the virtual address and the physical
   one, [pa + off], are aligned to what it covers; otherwise it descends through
   the valid tables. *)
let visit ?(pa = 0) t w size f =
  let rec descend size off =
    let table, level, i = top w in
    let covers = t.covers.(level) in
    let deeper =
      if w.create then
        covers > size
        || (not (t.fmt.large ~level))
        || w.at land (covers - 1) <> 0
        || (pa + off) land (covers - 1) <> 0
      else
        let e = t.fmt.get ~level ~table i in
        (not (t.fmt.leaf ~level e)) && (w.free_tables || t.fmt.valid e)
    in
    if deeper then begin
      down t w;
      descend size off
    end
  in
  let rec go size off =
    if size > 0 then begin
      descend size off;
      let table, level, i = top w in
      let covers = t.covers.(level) in
      let n =
        Int.max
          (Int.min (size / covers) (t.counts.(level) - i))
          (if w.inspect then 1 else 0)
      in
      if n <= 0 then invalid_arg "Page_table: a range smaller than a page";
      f off table level i n covers;
      w.at <- w.at + (n * covers);
      w.path <- (table, level, i + n) :: List.tl w.path;
      up t w;
      go (size - (n * covers)) (off + (n * covers))
    end
  in
  go size 0

(* The fragment of a run of [size] bytes mapping [pa] at [va]: the log2 of the 4
   KiB pages of the largest block naturally aligned in both address spaces that
   the run's size allows. The lowest bit set in any of the three bounds it. *)
let fragment ~va ~pa size =
  let x = va lor pa lor size in
  let rec log2 n k = if n <= 1 then k else log2 (n lsr 1) (k + 1) in
  log2 (x land -x) 0 - 12

(* Tables *)

let create ?base fmt space ~memory ~boot ~tables ~pages =
  let levels = Array.of_list (List.rev fmt.levels) in
  let msb = Array.of_list (fmt.levels @ [ fmt.bits + 1 ]) in
  let n = Array.length levels in
  let table_bytes =
    match tables with
    | Pool -> round_up (memory / table_share) table_round
    | Main -> 0
  in
  if boot + table_bytes > memory then
    invalid_arg "Page_table.create: the pools do not fit in the memory";
  let rest = boot + table_bytes in
  let boot_pool = Tlsf.create ~base:0 boot in
  let root =
    match Tlsf.alloc ~align:page boot_pool page with
    | Some pa ->
        fmt.zero pa page;
        pa
    | None -> invalid_arg "Page_table.create: no boot memory for the root table"
  in
  {
    fmt;
    space;
    covers = Array.map (fun s -> 1 lsl s) levels;
    counts = Array.init n (fun i -> 1 lsl (msb.(n - i) - msb.(n - i - 1)));
    boot = boot_pool;
    tables = Tlsf.create ~base:boot table_bytes;
    main = Tlsf.create ~base:rest (memory - rest);
    pages;
    booting = true;
    root;
    base = Option.value base ~default:(Space.base space);
  }

let booted t = t.booting <- false
let root t = t.root
let space t = t.space
let base t = t.base
let span t = 1 lsl t.fmt.bits
let memory t = Tlsf.length t.main

let tables t ~va n =
  let w = walk t ~create:true va in
  let path = ref [] in
  match
    visit t w n (fun _ _ _ _ _ _ ->
        if !path = [] then path := List.map (fun (table, _, _) -> table) w.path)
  with
  | () -> Some (List.rev !path)
  | exception No_room -> None

let clear t ~va n =
  let w = walk t ~free_tables:true va in
  visit t w n (fun _ table level i n _ ->
      for k = i to i + n - 1 do
        if not (t.fmt.valid (t.fmt.get ~level ~table k)) then
          invalid_arg (Printf.sprintf "Page_table.unmap: 0x%x is not mapped" va);
        t.fmt.set ~level ~table k (invalid_entry t level)
      done)

let unmap t ~va n =
  clear t ~va n;
  t.fmt.flush ()

let map ?(uncached = false) ?(snooped = false) t ~va target ranges =
  let size = List.fold_left (fun n (_, s) -> n + s) 0 ranges in
  let probe = walk t ~inspect:true va in
  visit t probe size (fun _ table level i n _ ->
      for k = i to i + n - 1 do
        if t.fmt.valid (t.fmt.get ~level ~table k) then
          invalid_arg
            (Printf.sprintf "Page_table.map: 0x%x is mapped already" va)
      done);
  let w = walk t ~create:true va in
  let write (pa, bytes) =
    visit ~pa t w bytes (fun off table level i n covers ->
        let fragment =
          fragment ~va:(t.base + w.at) ~pa:(pa + off) (n * covers)
        in
        for k = 0 to n - 1 do
          t.fmt.set ~level ~table (i + k)
            (t.fmt.encode ~level ~table:false target ~uncached ~snooped
               ~fragment ~valid:true
               (pa + off + (k * covers)))
        done)
  in
  match List.iter write ranges with
  | () ->
      t.fmt.flush ();
      Some { va; size; pages = ranges; target; uncached; snooped }
  | exception No_room ->
      (* The entries written so far are cleared. *)
      let mapped = t.base + w.at - va in
      if mapped > 0 then clear t ~va mapped;
      t.fmt.flush ();
      None

(* Physical blocks for [n] bytes, the largest [pages] allows first, falling to
   smaller ones when the pool has none left. *)
let blocks t n =
  let rec go acc left = function
    | _ when left = 0 -> Some (List.rev acc)
    | [] ->
        List.iter (fun (pa, _) -> pfree t pa) acc;
        None
    | (size, _) :: rest when size > left -> go acc left rest
    | (size, align) :: rest as sizes -> (
        match take t t.main ~align ~zero:false size with
        | Some pa -> go ((pa, size) :: acc) (left - size) sizes
        | None -> go acc left rest)
  in
  go [] n t.pages

(* One block of [n] bytes, aligned as the largest block of [pages] it holds, so
   that it maps with the largest pages, when the pool has such a block. *)
let block t n =
  let align =
    match List.find_opt (fun (size, _) -> size <= n) t.pages with
    | Some (_, align) -> align
    | None -> page
  in
  match take t t.main ~align n with
  | None when align > page -> take t t.main n
  | found -> found

let alloc ?(align = page) ?(uncached = false) ?(contiguous = false) t n =
  let n = round_up n page in
  match Space.alloc ~align t.space n with
  | None -> None
  | Some va -> (
      let pages =
        if contiguous then Option.map (fun pa -> [ (pa, n) ]) (block t n)
        else blocks t n
      in
      let give_back pages =
        List.iter (fun (pa, _) -> pfree t pa) pages;
        Space.free t.space va;
        None
      in
      match pages with
      | None ->
          Space.free t.space va;
          None
      | Some pages -> (
          match map ~uncached t ~va Gpu pages with
          | Some m -> Some m
          | None -> give_back pages))

let free t (m : mapping) =
  unmap t ~va:m.va m.size;
  Space.free t.space m.va;
  if m.target = Gpu then List.iter (fun (pa, _) -> pfree t pa) m.pages
