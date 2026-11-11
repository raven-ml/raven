(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type mask = Full | Window | Reach
type holes = Whole | Scattered | Tile

type t = {
  dtype : Nx.Dtype.any;
  lead : int;
  hq : int;
  hk : int;
  tq : int;
  page : int;
  max_pages : int;
  dk : int;
  dv : int;
  lengths : int array option;
  mask : mask;
  holes : holes;
  nans : bool;
  seed : int;
}

let mask_name = function Full -> "full" | Window -> "window" | Reach -> "reach"

let holes_name = function
  | Whole -> "whole"
  | Scattered -> "scattered"
  | Tile -> "tile"

let tk c = c.max_pages * c.page

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  Format.fprintf ppf
    "%s lead %d hq %d hk %d tq %d page %d max_pages %d dk %d dv %d lengths %s \
     mask %s holes %s nans %b seed %d"
    (Nx.Dtype.name dt) c.lead c.hq c.hk c.tq c.page c.max_pages c.dk c.dv
    (match c.lengths with
    | Some l -> String.concat "," (Array.to_list (Array.map string_of_int l))
    | None -> "none")
    (mask_name c.mask) (holes_name c.holes) c.nans c.seed

(* The width of [Window] and [Reach], in keys. *)
let width = 24L

(* The keys [Tile] hides: a whole tile of 32 or 64 keys on every target. *)
let tile_lo = 64
let tile_hi = 128

(* The pools' pages no entry names, beside every lead index's. *)
let spare = 2

let gen =
  let open Gen in
  let* dtype =
    of_list Nx.Dtype.[ Any Nx.bfloat16; Any Nx.float16; Any Nx.float32 ]
  in
  let* lead = int_range 1 2 in
  let* hq, hk = of_list [ (8, 8); (8, 2); (8, 1) ] in
  let* tq = frequency [ (1, constant 1); (2, of_list [ 7; 33; 64; 65; 100 ]) ] in
  let* page = of_list [ 1; 16 ] in
  let* keys = of_list [ 16; 64; 100; 192; 320 ] in
  let max_pages = max 1 (keys / page) in
  let tk = max_pages * page in
  let* lengths =
    frequency
      [
        (1, constant None);
        ( 2,
          map Option.some
            (array ~size:(constant lead)
               (frequency
                  [ (1, constant 0); (1, constant tk); (2, int_range 0 tk) ])) );
      ]
  in
  let* mask = of_list [ Full; Window; Reach ] in
  let* holes = of_list [ Whole; Scattered; Tile ] in
  let* nans = bool in
  let* dk = of_list [ 16; 64; 72 ] in
  let* dv = of_list [ 8; 64 ] in
  let+ seed = int_range 0 1000 in
  { dtype; lead; hq; hk; tq; page; max_pages; dk; dv; lengths; mask; holes; nans; seed }

let base =
  {
    dtype = Nx.Dtype.Any Nx.bfloat16;
    lead = 1;
    hq = 8;
    hk = 2;
    tq = 100;
    page = 1;
    max_pages = 320;
    dk = 64;
    dv = 64;
    lengths = None;
    mask = Full;
    holes = Whole;
    nans = false;
    seed = 5;
  }

let examples =
  let f32 = Nx.Dtype.Any Nx.float32 in
  [
    (* gpt-oss's decode step, 64 query heads over 8, in a window. *)
    {
      base with
      lead = 2;
      hq = 64;
      hk = 8;
      tq = 1;
      lengths = Some [| 300; 77 |];
      mask = Window;
      holes = Tile;
      nans = true;
    };
    { base with page = 16; max_pages = 20; holes = Scattered };
    {
      base with
      tq = 65;
      max_pages = 192;
      lengths = Some [| 150 |];
      mask = Reach;
      holes = Tile;
      nans = true;
    };
    (* Every length past keys 64 to 127, which the table alone hides. *)
    { base with lengths = Some [| 300 |]; holes = Tile; nans = true };
    {
      base with
      dtype = f32;
      lead = 2;
      tq = 1;
      page = 16;
      max_pages = 20;
      lengths = Some [| 200; 0 |];
      holes = Scattered;
      nans = true;
    };
    { base with dtype = f32; tq = 33; mask = Reach; holes = Tile };
    (* 8 warps of rows, keys of 272 dimensions: values of 64 overflowed a
       workgroup's memory beside tiles of 64 keys, and of 56 fill it within
       a byte a key. *)
    { base with hq = 128; hk = 1; tq = 256; max_pages = 256; dk = 272; dv = 64 };
    { base with hq = 128; hk = 1; tq = 256; max_pages = 256; dk = 272; dv = 56 };
  ]

let host_examples =
  List.filter_map
    (fun c ->
      if c.hq >= 128 then None
      else Some { c with dtype = Nx.Dtype.Any Nx.float32 })
    examples

(* Inputs *)

type float_dtype = F : (float, 's) Nx.dtype -> float_dtype

let float_dtype (Nx.Dtype.Any dt) =
  match Nx.Dtype.name dt with
  | "bfloat16" -> F Nx.bfloat16
  | "float16" -> F Nx.float16
  | _ -> F Nx.float32

let numel = Array.fold_left ( * ) 1

(* [n] floats within [[-1, 1]] rounded to [dt], held as float32's. *)
let draw (type s) (dt : (float, s) Nx.dtype) c salt shape =
  let st = Random.State.make [| c.seed; salt |] in
  let x =
    Nx.create Nx.float32 shape
      (Array.init (numel shape) (fun _ -> Random.State.float st 2. -. 1.))
  in
  Nx.to_array (Nx.cast Nx.float32 (Nx.cast dt x))

(* The table, [[lead; max_pages]] flattened: the pools' pages shuffled, then
   the case's holes, and [-1] at entries wholly past a row's length. *)
let entries c ~pages =
  let n = c.lead * c.max_pages in
  let st = Random.State.make [| c.seed; 21 |] in
  let perm = Array.init pages Fun.id in
  for i = pages - 1 downto 1 do
    let j = Random.State.int st (i + 1) in
    let t = perm.(i) in
    perm.(i) <- perm.(j);
    perm.(j) <- t
  done;
  Array.init n (fun j ->
      let b = j / c.max_pages and i = j mod c.max_pages in
      let past =
        match c.lengths with Some ls -> i * c.page >= ls.(b) | None -> false
      in
      match c.holes with
      | Whole -> perm.(j)
      | _ when past -> -1
      | Scattered when j mod 3 = 1 -> if j mod 2 = 0 then -1 else pages
      | Tile when i * c.page >= tile_lo && (i + 1) * c.page <= tile_hi -> -1
      | Scattered | Tile -> perm.(j))

(* The call over [q], its keys and values paged through [table] or dense
   with the keys [hidden] masked, as {!Nx.attention} takes them. *)
let call (type s d) c ~lengths ~table ~hidden (q : (float, s, d) Nx.t) k v =
  let place x = Nx.place (Nx.beside q) x in
  let pos (p : d Nx.pair) = Nx.add p.q (Nx.sub_s p.len (Int64.of_int p.tq)) in
  let near : (d Nx.pair -> d Nx.bool_t) option =
    match c.mask with
    | Full -> None
    | Window -> Some (fun p -> Nx.greater (Nx.add_s p.k width) (pos p))
    | Reach ->
        let w = place (Nx.create Nx.int64 [||] [| width |]) in
        Some (fun p -> Nx.greater (Nx.add p.k w) (pos p))
  in
  let hidden = Option.map place hidden in
  let mask (p : d Nx.pair) =
    let m = Nx.causal p in
    let m = match near with Some f -> Nx.logical_and m (f p) | None -> m in
    match hidden with Some h -> Nx.logical_and m (Nx.logical_not h) | None -> m
  in
  let kv : (float, s, d) Nx.kv =
    match table with
    | Some t -> Paged { k; v; table = place t }
    | None -> Dense { k; v }
  in
  Nx.attention ?lengths:(Option.map place lengths) ~mask ~acc:Nx.float32 ~kv q

let law (module S : Nx.Devices) c =
  let (F dt) = float_dtype c.dtype in
  let tk = tk c in
  let pages = (c.lead * c.max_pages) + spare in
  let table = entries c ~pages in
  let out e = e < 0 || e >= pages in
  let len b = match c.lengths with Some ls -> ls.(b) | None -> tk in
  cover "decode" (c.tq = 1);
  cover "prefill" (c.tq > 1);
  cover "pages of one key" (c.page = 1);
  cover "pages of 16 keys" (c.page = 16);
  cover (mask_name c.mask) true;
  cover (Nx.Dtype.name dt) true;
  cover "entries out of range" (Array.exists out table);
  cover "keys the table alone hides"
    (c.holes = Tile && c.mask = Full
    && List.for_all (fun b -> len b >= tile_hi) (List.init c.lead Fun.id));
  cover "NaN where no row reads" c.nans;
  (* Each pool slot some row reads: its entry's in range, its key before
     the row's length. *)
  let read = Array.make (pages * c.page) false in
  Array.iteri
    (fun j e ->
      let b = j / c.max_pages and i = j mod c.max_pages in
      if not (out e) then
        for s = 0 to c.page - 1 do
          if (i * c.page) + s < len b then read.((e * c.page) + s) <- true
        done)
    table;
  let pool salt d =
    let a = draw dt c salt [| pages; c.hk; c.page; d |] in
    if c.nans then
      Array.iteri
        (fun i _ ->
          let e = i / d / c.page / c.hk and s = i / d mod c.page in
          if not read.((e * c.page) + s) then a.(i) <- Float.nan)
        a;
    a
  in
  let pk = pool 2 c.dk and pv = pool 3 c.dv in
  (* The rows the table reads, zeros where its entry is out of range. *)
  let dense p d =
    Array.init (c.lead * c.hk * tk * d) (fun i ->
        let col = i mod d and t = i / d mod tk in
        let h = i / d / tk mod c.hk and b = i / d / tk / c.hk in
        let e = table.((b * c.max_pages) + (t / c.page)) in
        if out e then 0.
        else p.((((((e * c.hk) + h) * c.page) + (t mod c.page)) * d) + col))
  in
  let dk = dense pk c.dk and dv = dense pv c.dv in
  let host shape a = Nx.create Nx.float32 shape a in
  let qs = [| c.lead; c.hq; c.tq; c.dk |] in
  let q = host qs (draw dt c 1 qs) in
  let lengths =
    Option.map
      (fun l -> Nx.create Nx.int64 [| c.lead |] (Array.map Int64.of_int l))
      c.lengths
  in
  let holes =
    Array.init (c.lead * tk) (fun i ->
        out table.((i / tk * c.max_pages) + (i mod tk / c.page)))
  in
  let hidden =
    if Array.exists Fun.id holes then
      Some (Nx.create Nx.bool [| c.lead; 1; 1; tk |] holes)
    else None
  in
  let table =
    Nx.create Nx.int64 [| c.lead; c.max_pages |] (Array.map Int64.of_int table)
  in
  let compiled ~table ~hidden k v =
    let place x = Nx.place S.on (Nx.cast dt x) in
    let g =
      Rune.jit
        Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))
        (fun q k v -> call c ~lengths ~table ~hidden q k v)
    in
    let before = Rune.evaluations () in
    let o, lse = g (place q) (place k) (place v) in
    equal ~msg:"compiled, not evaluated" Testable.int 0 (Rune.evaluations () - before);
    let bits x =
      Array.map Int64.bits_of_float
        (Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)))
    in
    (bits o, bits lse)
  in
  let po, plse =
    compiled ~table:(Some table) ~hidden:None
      (host [| pages; c.hk; c.page; c.dk |] pk)
      (host [| pages; c.hk; c.page; c.dv |] pv)
  in
  let o, lse =
    compiled ~table:None ~hidden
      (host [| c.lead; c.hk; tk; c.dk |] dk)
      (host [| c.lead; c.hk; tk; c.dv |] dv)
  in
  let same name dense paged =
    equal ~msg:(name ^ "'s elements") Testable.int (Array.length dense)
      (Array.length paged);
    let rec first i =
      if i = Array.length dense then ()
      else if Int64.equal dense.(i) paged.(i) then first (i + 1)
      else
        equal
          ~msg:
            (Printf.sprintf "%s's bits at element %d, dense %h and paged %h"
               name i (Int64.float_of_bits dense.(i))
               (Int64.float_of_bits paged.(i)))
          Testable.int64 dense.(i) paged.(i)
    in
    first 0
  in
  same "o" o po;
  same "lse" lse plse
