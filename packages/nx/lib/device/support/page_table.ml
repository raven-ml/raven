(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type space = Phys | Sys | Peer

module Space = struct
  type t = { tlsf : Tlsf.t; lock : Mutex.t }

  let create ~base n = { tlsf = Tlsf.create ~base n; lock = Mutex.create () }
  let base s = Tlsf.base s.tlsf
  let length s = Tlsf.length s.tlsf

  let highest_bit n =
    let rec go b = if b * 2 > n then b else go (b * 2) in
    if n <= 0 then 1 else go 1

  let alloc ?(align = 0x1000) s n =
    Mutex.protect s.lock (fun () ->
        Tlsf.alloc ~align:(Int.max (highest_bit n) align) s.tlsf n)

  let free s a = Mutex.protect s.lock (fun () -> Tlsf.free s.tlsf a)
end

type entry = {
  levels : int list;
  bits : int;
  first : int;
  get : level:int -> table:int -> int -> int64;
  set : level:int -> table:int -> int -> int64 -> unit;
  encode :
    level:int ->
    table:bool ->
    space ->
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

type mapping = {
  va : int;
  size : int;
  pages : (int * int) list;
  space : space;
  uncached : bool;
  snooped : bool;
}

type t = {
  e : entry;
  space : Space.t;
  covers : int array; (* bytes an entry covers, by level *)
  counts : int array; (* entries of a table, by level *)
  boot : Tlsf.t;
  tables : Tlsf.t;
  main : Tlsf.t;
  pages : (int * int) list; (* block sizes and alignments, largest first *)
  mutable booting : bool;
  root : int;
  base : int;
}

let round_up n a = (n + a - 1) / a * a
let page = 0x1000

(* Physical memory *)

let pool t ~table =
  if t.booting then t.boot
  else if table && Tlsf.length t.tables > 0 then t.tables
  else t.main

let take t tlsf ?(align = page) ?(zero = true) n =
  let n = round_up n page in
  match Tlsf.alloc ~align tlsf n with
  | Some pa as r ->
      if zero then t.e.zero pa n;
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

let owner t pa =
  List.find
    (fun a -> pa >= Tlsf.base a && pa < Tlsf.base a + Tlsf.length a)
    [ t.boot; t.tables; t.main ]

let pfree t pa = Tlsf.free (owner t pa) pa

(* Walks *)

(* A walk visits the entries of a virtual range, from [va] on, keeping the path
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
  let lv = t.e.first in
  { at; path = [ (t.root, lv, index t lv at) ]; create; free_tables; inspect }

let top w = List.hd w.path

let down t w =
  let table, level, i = top w in
  let e = t.e.get ~level ~table i in
  if not (t.e.valid e) then begin
    if not w.create then
      invalid_arg "Page_table: an address of the range is not mapped";
    match take t (pool t ~table:true) 0x1000 with
    | None -> failwith "Page_table: no memory for a page table"
    | Some pa ->
        t.e.set ~level ~table i
          (t.e.encode ~level ~table:true Phys ~uncached:false ~snooped:false
             ~fragment:0 ~valid:true pa)
  end;
  let e = t.e.get ~level ~table i in
  if t.e.leaf ~level e then invalid_arg "Page_table: a page where a table was";
  let child = t.e.address e and level = level + 1 in
  w.path <- (child, level, index t level w.at) :: w.path

let empty t table level =
  let rec go i =
    i >= t.counts.(level)
    || ((not (t.e.valid (t.e.get ~level ~table i))) && go (i + 1))
  in
  go 0

(* Frees the table at the top of the path if it became empty. *)
let try_free t w =
  match w.path with
  | (table, level, _) :: (parent, plevel, pi) :: _
    when w.free_tables && empty t table level ->
      pfree t table;
      t.e.set ~level:plevel ~table:parent pi
        (t.e.encode ~level:plevel ~table:false Phys ~uncached:false
           ~snooped:false ~fragment:0 ~valid:false 0);
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
  let rec go size off =
    if size > 0 then begin
      let rec descend () =
        let table, level, i = top w in
        let covers = t.covers.(level) in
        let deeper =
          if w.create then
            covers > size
            || (not (t.e.large ~level))
            || w.at land (covers - 1) <> 0
            || (pa + off) land (covers - 1) <> 0
          else
            let e = t.e.get ~level ~table i in
            (not (t.e.leaf ~level e)) && (w.free_tables || t.e.valid e)
        in
        if deeper then begin
          down t w;
          descend ()
        end
      in
      descend ();
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
   KiB pages of the largest block that is naturally aligned in both address
   spaces and that the run's size allows. The lowest bit set in any of the three
   bounds it. *)
let fragment ~va ~pa size =
  let x = va lor pa lor size in
  let rec log2 n k = if n <= 1 then k else log2 (n lsr 1) (k + 1) in
  log2 (x land -x) 0 - 12

(* Tables *)

let create ?base e space ~memory ~boot ~tables ~pages =
  let levels = Array.of_list (List.rev e.levels) in
  let msb = Array.of_list (e.levels @ [ e.bits + 1 ]) in
  let n = Array.length levels in
  let table_bytes = if tables then round_up (memory / 512) (1 lsl 20) else 0 in
  if boot + table_bytes > memory then
    invalid_arg "Page_table.create: the pools do not fit in the memory";
  let off = boot + table_bytes in
  let boot_pool = Tlsf.create ~base:0 boot in
  let root =
    match Tlsf.alloc ~align:page boot_pool page with
    | Some pa ->
        e.zero pa page;
        pa
    | None -> invalid_arg "Page_table.create: no boot memory for the root table"
  in
  {
    e;
    space;
    covers = Array.map (fun s -> 1 lsl s) levels;
    counts = Array.init n (fun i -> 1 lsl (msb.(n - i) - msb.(n - i - 1)));
    boot = boot_pool;
    tables = Tlsf.create ~base:boot table_bytes;
    main = Tlsf.create ~base:off (memory - off);
    pages;
    booting = true;
    root;
    base = Option.value base ~default:(Space.base space);
  }

let booted t = t.booting <- false
let root t = t.root
let space t = t.space
let base t = t.base
let span t = 1 lsl t.e.bits
let memory t = Tlsf.length t.main

let tables t ~va n =
  let w = walk t ~create:true va in
  let path = ref [] in
  visit t w n (fun _ _ _ _ _ _ ->
      if !path = [] then path := List.map (fun (table, _, _) -> table) w.path);
  List.rev !path

let clear t ~va n =
  let w = walk t ~free_tables:true va in
  visit t w n (fun _ table level i n _ ->
      for k = i to i + n - 1 do
        if not (t.e.valid (t.e.get ~level ~table k)) then
          invalid_arg (Printf.sprintf "Page_table.unmap: 0x%x is not mapped" va);
        t.e.set ~level ~table k
          (t.e.encode ~level ~table:false Phys ~uncached:false ~snooped:false
             ~fragment:0 ~valid:false 0)
      done)

let unmap t ~va n =
  clear t ~va n;
  t.e.flush ()

let map ?(uncached = false) ?(snooped = false) t ~va space ranges =
  let size = List.fold_left (fun n (_, s) -> n + s) 0 ranges in
  let probe = walk t ~inspect:true va in
  visit t probe size (fun _ table level i n _ ->
      for k = i to i + n - 1 do
        if t.e.valid (t.e.get ~level ~table k) then
          invalid_arg
            (Printf.sprintf "Page_table.map: 0x%x is mapped already" va)
      done);
  let w = walk t ~create:true va in
  let base = t.base in
  let write (pa, bytes) =
    visit ~pa t w bytes (fun off table level i n covers ->
        let fragment = fragment ~va:(base + w.at) ~pa:(pa + off) (n * covers) in
        for k = 0 to n - 1 do
          t.e.set ~level ~table (i + k)
            (t.e.encode ~level ~table:false space ~uncached ~snooped ~fragment
               ~valid:true
               (pa + off + (k * covers)))
        done)
  in
  (* A table that cannot be allocated leaves the entries written so far, which
     are cleared. *)
  (match List.iter write ranges with
  | () -> ()
  | exception e ->
      let mapped = base + w.at - va in
      if mapped > 0 then clear t ~va mapped;
      raise e);
  t.e.flush ();
  { va; size; pages = ranges; space; uncached; snooped }

(* Physical blocks for [n] bytes, the largest [pages] allows first, falling to
   smaller ones when the pool has none left. *)
let blocks t ~zero n =
  let rec go acc left = function
    | _ when left = 0 -> Some (List.rev acc)
    | [] ->
        List.iter (fun (pa, _) -> pfree t pa) acc;
        None
    | (size, _) :: rest when size > left -> go acc left rest
    | (size, align) :: rest as sizes -> (
        match take t t.main ~align ~zero size with
        | Some pa -> go ((pa, size) :: acc) (left - size) sizes
        | None -> go acc left rest)
  in
  go [] n t.pages

let alloc ?(align = page) ?(uncached = false) ?(contiguous = false)
    ?(zero = false) t n =
  let n = round_up n page in
  match Space.alloc ~align t.space n with
  | None -> None
  | Some va -> (
      let pages =
        if contiguous then
          (* Aligned as the largest block it holds, so that it maps with the
             largest pages, when the pool has such a block. *)
          let align =
            match List.find_opt (fun (size, _) -> size <= n) t.pages with
            | Some (_, align) -> align
            | None -> page
          in
          let block =
            match take t t.main ~align ~zero:true n with
            | None when align > page -> take t t.main ~zero:true n
            | block -> block
          in
          Option.map (fun pa -> [ (pa, n) ]) block
        else blocks t ~zero n
      in
      match pages with
      | None ->
          Space.free t.space va;
          None
      | Some pages -> (
          match map ~uncached t ~va Phys pages with
          | m -> Some m
          | exception e ->
              List.iter (fun (pa, _) -> pfree t pa) pages;
              Space.free t.space va;
              raise e))

let free t (m : mapping) =
  unmap t ~va:m.va m.size;
  Space.free t.space m.va;
  if m.space = Phys then List.iter (fun (pa, _) -> pfree t pa) m.pages
