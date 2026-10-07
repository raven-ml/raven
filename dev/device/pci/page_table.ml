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
  shifts : int array; (* the address bit each level indexes, root first *)
  counts : int array; (* the entries of a table, root first *)
  boot : Tlsf.t;
  tables : Tlsf.t; (* empty unless the tables have a pool *)
  main : Tlsf.t;
  pages : (int * int) list; (* block sizes and alignments, largest first *)
  held : (int, unit) Hashtbl.t; (* the tables in use, the root among them *)
  mutable booting : bool;
  root : int;
  base : int;
  memory : int;
}

(* The GPU's page: what the leaf level maps, and the unit of a fragment. *)
let page_bits = 12
let page = 1 lsl page_bits

(* A table pool per GPU memory of this many bytes, rounded up to
   [table_round]. *)
let table_share = 512
let table_round = 1 lsl 20
let round_up n a = (n + a - 1) / a * a
let is_power_of_two n = n > 0 && n land (n - 1) = 0
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
  if n <= 0 || not (is_power_of_two align) then
    invalid_arg
      (Printf.sprintf
         "Page_table.palloc: %d bytes aligned to %d, expected more than 0 \
          bytes and a power of two"
         n align);
  let tlsf =
    match boot with
    | Some true -> t.boot
    | Some false -> t.main
    | None -> pool t ~table:false
  in
  take t tlsf ~align ?zero n

let pfree t pa =
  let refuse () =
    invalid_arg (Printf.sprintf "Page_table.pfree: no block at 0x%x" pa)
  in
  let inside a = pa >= Tlsf.base a && pa < Tlsf.base a + Tlsf.length a in
  if Hashtbl.mem t.held pa then refuse ();
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

let new_table t =
  match take t (pool t ~table:true) page with
  | None -> raise No_room
  | Some pa ->
      Hashtbl.replace t.held pa ();
      pa

(* Whether the valid entry [e] at depth [d] maps a page: every one of the last
   level does. *)
let is_page t d e = d = bottom t || t.fmt.leaf ~level:(level t d) e

let invalid_entry t level =
  t.fmt.encode ~level ~table:false Gpu ~uncached:false ~snooped:false
    ~fragment:0 ~valid:false 0

let empty t d table =
  let level = level t d in
  let rec go i =
    i = t.counts.(d)
    || ((not (t.fmt.valid (t.fmt.get ~level ~table i))) && go (i + 1))
  in
  go 0

(* The table entry [i] of [table], at depth [d], points to, made if missing. *)
let child t d table i =
  let level = level t d in
  let e = t.fmt.get ~level ~table i in
  if not (t.fmt.valid e) then begin
    let pa = new_table t in
    t.fmt.set ~level ~table i
      (t.fmt.encode ~level ~table:true Gpu ~uncached:false ~snooped:false
         ~fragment:0 ~valid:true pa);
    pa
  end
  else if is_page t d e then
    invalid_arg "Page_table.tables: a larger page maps the address"
  else t.fmt.address e

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
  let level = level t d in
  each t d ~at lo hi @@ fun i at lo hi ->
  let e = t.fmt.get ~level ~table i in
  if not (t.fmt.valid e) then ()
  else if is_page t d e then
    invalid_arg
      (Printf.sprintf "Page_table.map: 0x%x is mapped already" (t.base + lo))
  else unmapped t (d + 1) (t.fmt.address e) ~at lo hi

(* Raises unless pages map every address of [lo, hi), none past it. *)
let rec mapped t d table ~at lo hi =
  let level = level t d and c = covers t d in
  each t d ~at lo hi @@ fun i at lo hi ->
  let e = t.fmt.get ~level ~table i in
  if not (t.fmt.valid e) then
    invalid_arg
      (Printf.sprintf "Page_table.unmap: 0x%x is not mapped" (t.base + lo))
  else if not (is_page t d e) then mapped t (d + 1) (t.fmt.address e) ~at lo hi
  else if lo <> at || hi <> at + c then
    invalid_arg
      (Printf.sprintf "Page_table.unmap: the page at 0x%x is partly outside"
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
   run. Its pages below [until] share the fragment [frag]. *)
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

(* The entry at depth [d] of the page at [v] of [r]'s run. A fragment's block is
   aligned in the space's addresses; the pages of a block share its fragment,
   which is found once a block. *)
let entry t r d v =
  if v >= r.until then begin
    let va = t.base + v and k = t.shifts.(d) - page_bits in
    r.frag <-
      fragment ~lo:(t.base + r.lo) ~hi:(t.base + r.hi) ~delta:(r.delta - t.base)
        va k;
    r.until <- (va lor ((page lsl r.frag) - 1)) + 1 - t.base
  end;
  t.fmt.encode ~level:(level t d) ~table:false r.target ~uncached:r.uncached
    ~snooped:r.snooped ~fragment:r.frag ~valid:true (v + r.delta)

(* Maps [lo, hi) to the runs of [r], each page with the largest entry that its
   run holds and both its addresses are aligned to. One walk serves every run:
   each table is read once. The last level's entries are all whole pages. *)
let rec write t d table ~at lo hi r =
  let level = level t d and c = covers t d in
  if d = bottom t then
    for i = (lo - at) / c to ((hi - at) / c) - 1 do
      let v = at + (i * c) in
      seek r v;
      t.fmt.set ~level ~table i (entry t r d v)
    done
  else
    each t d ~at lo hi @@ fun i at lo hi ->
    seek r lo;
    let whole =
      lo = at && hi = at + c && hi <= r.hi && aligned (lo + r.delta) c
    in
    if whole && t.fmt.large ~level then
      t.fmt.set ~level ~table i (entry t r d lo)
    else write t (d + 1) (child t d table i) ~at lo hi r

(* Clears the entries of [lo, hi) and frees the tables it empties. The last
   level's entries are cleared unread: each is a page or invalid. *)
let rec clear t d table ~at lo hi =
  let level = level t d and c = covers t d in
  let none = invalid_entry t level in
  if d = bottom t then
    for i = (lo - at) / c to ((hi - at) / c) - 1 do
      t.fmt.set ~level ~table i none
    done
  else
    each t d ~at lo hi @@ fun i at lo hi ->
    let e = t.fmt.get ~level ~table i in
    if not (t.fmt.valid e) then ()
    else if t.fmt.leaf ~level e then t.fmt.set ~level ~table i none
    else begin
      let child = t.fmt.address e in
      clear t (d + 1) child ~at lo hi;
      if empty t (d + 1) child then begin
        t.fmt.set ~level ~table i none;
        Hashtbl.remove t.held child;
        pfree t child
      end
    end

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
  if boot < 0 then
    invalid_arg (Printf.sprintf "Page_table.create: boot %d is negative" boot);
  if boot + table_bytes > memory then
    invalid_arg
      (Printf.sprintf
         "Page_table.create: %d boot and %d table bytes exceed the %d bytes of \
          memory"
         boot table_bytes memory);
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
    invalid_arg (Printf.sprintf "Page_table.create: base 0x%x off a page" base);
  let held = Hashtbl.create 64 in
  Hashtbl.replace held root ();
  {
    fmt;
    space;
    shifts;
    counts = Array.mapi (fun d s -> 1 lsl (above d - s)) shifts;
    boot = boot_pool;
    tables = Tlsf.create ~base:boot table_bytes;
    main = Tlsf.create ~base:rest (memory - rest);
    pages;
    held;
    booting = true;
    root;
    base;
    memory;
  }

let booted t = t.booting <- false
let root t = t.root
let space t = t.space
let base t = t.base
let span t = 1 lsl t.fmt.bits
let memory t = t.memory

(* Raises unless the [n] bytes from [va] are whole pages the tables reach. *)
let check t fn ~va n =
  let v = va - t.base in
  if v < 0 || n < 0 || n > span t - v || not (aligned (v lor n) page) then
    invalid_arg
      (Printf.sprintf
         "Page_table.%s: 0x%x bytes at 0x%x are not whole pages the tables \
          reach"
         fn n va)

let tables t ~va n =
  check t "tables" ~va n;
  let v = va - t.base in
  (* Down to the table where [map] would write the entry of [v]. *)
  let rec go d table path =
    let c = covers t d in
    let i = (v lsr t.shifts.(d)) land (t.counts.(d) - 1) in
    if d = bottom t || (t.fmt.large ~level:(level t d) && c <= n && aligned v c)
    then List.rev (table :: path)
    else go (d + 1) (child t d table i) (table :: path)
  in
  match go 0 t.root [] with path -> Some path | exception No_room -> None

let unmap t ~va n =
  check t "unmap" ~va n;
  let lo = va - t.base in
  mapped t 0 t.root ~at:0 lo (lo + n);
  clear t 0 t.root ~at:0 lo (lo + n);
  t.fmt.flush ()

let map ?(uncached = false) ?(snooped = false) t ~va target ranges =
  List.iter
    (fun (pa, n) ->
      if pa < 0 || n < 0 || not (aligned (pa lor n) page) then
        invalid_arg
          (Printf.sprintf
             "Page_table.map: 0x%x bytes at physical 0x%x are not whole pages" n
             pa))
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
    }
  in
  match write t 0 t.root ~at:0 lo (lo + size) r with
  | () ->
      t.fmt.flush ();
      Some { va; size; pages = ranges; target; uncached; snooped }
  | exception No_room ->
      (* The range was unmapped: clearing all of it removes what was written and
         the tables made for it. *)
      clear t 0 t.root ~at:0 lo (lo + size);
      t.fmt.flush ();
      None

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
    invalid_arg
      (Printf.sprintf "Page_table.alloc: %d bytes, expected more than 0" n);
  if n > Space.length t.space then None
  else
    let n = round_up n page in
    match Space.alloc t.space n with
    | None -> None
    | Some va ->
        let pages =
          if contiguous then Option.map (fun pa -> [ (pa, n) ]) (block t n)
          else blocks t n
        in
        let m = Option.bind pages (map ~uncached t ~va Gpu) in
        if Option.is_none m then begin
          Option.iter (List.iter (fun (pa, _) -> pfree t pa)) pages;
          Space.free t.space va
        end;
        m

let free t (m : mapping) =
  unmap t ~va:m.va m.size;
  Space.free t.space m.va;
  if m.target = Gpu then List.iter (fun (pa, _) -> pfree t pa) m.pages
