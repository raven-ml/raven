(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.amd's moves on the machine's AMD GPU 0, skipping without one: a scatter
   Add as nx_amd.mli states it, at each target x + S, S the sum of the position
   tree over the updates' positions along the axis, bit for bit, for axes past
   one tile, for updates whose positions step 0 along the last axis and into its
   own operand, of no update, target or column, NaNs and infinities among the
   values, operands through strided, permuted and reversed views; every move of
   complex64 arrays aligned to their parts' 4 bytes alone, as nx.cpu's; and the
   workspace nx_amd.mli states the moves take. *)

open Windtrap
open Elements
module Support = Nx_kernels_support

let amd () =
  match Support.amd () with
  | Some b -> b
  | None -> skip ~reason:"no AMD GPU nx.amd computes on" ()

let done_ r =
  if r <> A.Done then
    failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Cases *)

(* How idx draws a slice's positions along the axis: uniformly, int64's
   extremes among those out of range; each target at most once ([Perm]); all
   at one target ([One]); log-uniformly, so that the first targets take long
   runs ([Zipf]); or, in two slices of each four, all out of range, else
   uniformly in range ([Holes]), so that tiles take no entry. Each draws now
   and then a position out of range, but [One]. *)
type dist = Uniform | Perm | One | Zipf | Holes

(* Nx_amd.bound_scatters's bounds. *)
type bound = { parts : int; groups : int; whole : int; wide : bool }

(* Updates [m; cols] along axis 0, or [cols; m] along axis 1, into [d] targets
   along the axis; with [rows], idx is a column broadcast along the last axis;
   with [in_place], dst is [into]; into, idx and the updates through the views
   [vi], [vx] and [vu]; NaNs and infinities among the values where [specials];
   the sort under [bound] where given. *)
type case = {
  dt : D.any;
  m : int;
  d : int;
  cols : int;
  axis : int;
  rows : bool;
  in_place : bool;
  dist : dist;
  bound : bound option;
  vi : view;
  vx : view;
  vu : view;
  specials : bool;
  seed : int;
}

let viewed c = List.exists (fun v -> v <> plain 2) [ c.vi; c.vx; c.vu ]

let dist_name = function
  | Uniform -> "uniform"
  | Perm -> "distinct"
  | One -> "one target"
  | Zipf -> "log-uniform"
  | Holes -> "slices out of range"

let pp_case ppf c =
  let names v =
    match view_names v with [] -> "plain" | ns -> String.concat " " ns
  in
  Format.fprintf ppf
    "%a: %d updates by %d along %d into %d, %s positions%s%s%s%s, into %s, \
     idx %s, updates %s (seed %d)"
    pp_dtype c.dt c.m c.cols c.axis c.d (dist_name c.dist)
    (if c.rows then ", idx broadcast along the last axis" else "")
    (if c.in_place then ", in place" else "")
    (if c.specials then ", NaNs and infinities" else "")
    (match c.bound with
    | None -> ""
    | Some b ->
        Printf.sprintf ", sorts bound to %d parts, %d groups, %d whole%s"
          b.parts b.groups b.whole
          (if b.wide then ", wide" else ""))
    (names c.vi) (names c.vx) (names c.vu) c.seed

let dtypes =
  D.
    [
      Any Float32;
      Any Float64;
      Any Bfloat16;
      Any Float16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
      Any Complex64;
      Any Complex128;
    ]

let gen =
  let open Gen in
  let* dt = of_list ~pp:pp_dtype dtypes in
  let* m =
    frequency
      [
        (1, constant 0);
        (6, int_range 1 40);
        (6, int_range 257 1024);
        (8, int_range 1025 6000);
      ]
  in
  let* d =
    frequency [ (1, constant 0); (12, int_range 1 6); (12, int_range 50 3000) ]
  in
  let* cols =
    frequency [ (1, constant 0); (16, int_range 1 5); (8, int_range 30 40) ]
  in
  let* axis = int_range 0 1 in
  let* rows = bool in
  let* in_place = bool in
  let* dist =
    frequency
      [ (3, constant Uniform); (1, constant Perm); (1, constant One);
        (2, constant Zipf); (1, constant Holes) ]
  in
  let* bound =
    frequency
      [
        (2, constant None);
        ( 1,
          let* parts = int_range 1 3 and+ groups = int_range 1 2 in
          let* whole = of_list [ 0; 1; 5; 64 ] and+ wide = bool in
          constant (Some { parts; groups; whole; wide }) );
      ]
  in
  let* views = bool in
  let view = if views then view_of 2 else constant (plain 2) in
  let* vi = view and+ vx = view and+ vu = view in
  let* specials = bool in
  let+ seed = int in
  {
    dt;
    m;
    d;
    cols;
    axis;
    rows = rows && axis = 0 && cols > 1;
    in_place;
    dist;
    bound;
    vi;
    vx = (if dist = Uniform then vx else plain 2);
    vu;
    specials;
    seed;
  }

let cases = Gen.with_pp pp_case gen

(* Cases that each run reaches: the regimes of the bench's rows at small
   sizes; the paths only large axes reach: tiles' entries past 256 tiles,
   the updates' runs past 2^16 positions; a part of the tiles' entries past
   256 tiles, some of them empty; wide items of the tiles' entries and of
   the updates into a float32 sum; row tiles that split their columns. *)
let examples =
  let ex ?(rows = false) ?(dist = Uniform) ?bound dt ~m ~d ~cols =
    {
      dt;
      m;
      d;
      cols;
      axis = 0;
      rows = rows && cols > 1;
      in_place = false;
      dist;
      bound;
      vi = plain 2;
      vx = plain 2;
      vu = plain 2;
      specials = false;
      seed = 3;
    }
  in
  let small = { parts = 2; groups = 1; whole = 64; wide = false } in
  D.
    [
      ex (Any Float32) ~dist:Perm ~m:5000 ~d:5000 ~cols:1;
      ex (Any Float32) ~m:5000 ~d:5000 ~cols:1;
      ex (Any Float32) ~m:20000 ~d:1250 ~cols:1 ~bound:small;
      ex (Any Float32) ~rows:true ~m:3000 ~d:2000 ~cols:36;
      ex (Any Float32) ~rows:true ~dist:Zipf ~m:3000 ~d:2000 ~cols:36;
      ex (Any Float32) ~dist:One ~m:5000 ~d:1024 ~cols:1;
      ex (Any Float64) ~dist:Zipf ~m:70000 ~d:3000 ~cols:1;
      ex (Any Float32) ~dist:One ~m:263000 ~d:2 ~cols:1;
      ex (Any Bfloat16) ~rows:true ~dist:One ~m:70000 ~d:3 ~cols:3;
      ex (Any Complex128) ~dist:Zipf ~m:3000 ~d:300 ~cols:2
        ~bound:{ small with whole = 0; wide = true };
      ex (Any Float64) ~m:2500 ~d:300 ~cols:5;
      ex (Any Float32) ~m:1500 ~d:2000 ~cols:40;
      ex (Any Float32) ~dist:Holes ~m:1500 ~d:3 ~cols:200
        ~bound:{ small with parts = 1 };
      ex (Any Float32) ~m:5000 ~d:500 ~cols:1 ~bound:{ small with wide = true };
      ex (Any Float32) ~m:5000 ~d:5000 ~cols:1
        ~bound:{ small with wide = true };
      ex (Any Float32) ~rows:true ~m:600 ~d:40 ~cols:36;
    ]

(* Values *)

let narrow (D.Any dt) = D.is D.Float dt && D.bits dt < 32
let complex (D.Any dt) = D.is D.Complex dt

(* A part's value: an integer of up to 12 bits times a power of two, -0 now and
   then, a NaN or an infinity now and then where [specials]; a narrow float's
   within its range over 3000 terms. *)
let value ?(specials = false) dt rs =
  let r = Random.State.int rs 64 in
  if r < 4 then -0.
  else if specials && r = 4 then Float.nan
  else if specials && r = 5 then Float.infinity
  else if specials && r = 6 then Float.neg_infinity
  else
    let e = if narrow dt then 3 else 12 in
    let k = Float.of_int (Random.State.int rs 4095 - 2047) in
    Float.ldexp k (Random.State.int rs ((2 * e) + 1) - e - 11)

(* Arithmetic in the format a sum computes in: float64, or float32. *)
let wide (D.Any dt) = D.bits dt = if D.is D.Complex dt then 128 else 64
let add dt a b = if wide dt then a +. b else round32 (a +. b)

(* An element's parts and back. *)
let parts dt e =
  if narrow dt then [| decode dt e |]
  else if complex dt then
    let h = String.length e / 2 in
    if h = 4 then [| get32 e 0; get32 e 4 |] else [| get64 e 0; get64 e 8 |]
  else if String.length e = 4 then [| get32 e 0 |]
  else [| get64 e 0 |]

let of_parts dt ps =
  if narrow dt then encode dt ps.(0)
  else
    String.concat ""
      (Array.to_list (Array.map (if wide dt then f64 else f32) ps))

(* A float64 part: 53 bits over 80 binades, so that sums round and an
   association other than the tree's gives other bits. *)
let wide_value ?(specials = false) rs =
  let r = Random.State.int rs 64 in
  if r < 4 || (specials && r < 7) then value ~specials (D.Any D.Float64) rs
  else Float.ldexp (Random.State.float rs 2. -. 1.) (Random.State.int rs 81 - 40)

let draw ?specials dt rs =
  let n = if complex dt then 2 else 1 in
  let part () =
    if wide dt then wide_value ?specials rs else round32 (value ?specials dt rs)
  in
  of_parts dt (Array.init n (fun _ -> part ()))

(* The reference *)

(* The position tree's sum over [items], positions below [2^l]: each half of the
   block summed apart, the left one first; [None] for none. *)
let rec tree dt l items =
  match items with
  | [] -> None
  | [ (_, v) ] -> Some v
  | _ -> (
      let left, right =
        List.partition (fun (j, _) -> (j lsr (l - 1)) land 1 = 0) items
      in
      match (tree dt (l - 1) left, tree dt (l - 1) right) with
      | Some a, Some b -> Some (add dt a b)
      | (Some _ as a), None | None, (Some _ as a) -> a
      | None, None -> None)

let rec ceil_log2 ?(b = 0) n =
  if 1 lsl b >= n then b else ceil_log2 ~b:(b + 1) n

(* The scatter's result as nx_amd.mli states it, as elements in C order, and
   whether a target's sum differs from the left-to-right sum of its updates, so
   that another association would give other bits. *)
let expected c into idx upd =
  let tgt = Array.map (fun _ -> []) into in
  for j = c.m - 1 downto 0 do
    for k = 0 to c.cols - 1 do
      let p = idx j k in
      if p >= 0 && p < c.d then begin
        let t = if c.axis = 0 then (p * c.cols) + k else (k * c.d) + p in
        tgt.(t) <- (j, upd j k) :: tgt.(t)
      end
    done
  done;
  let shows = ref false in
  let want =
    Array.mapi
      (fun t x ->
        match tgt.(t) with
        | [] -> x
        | us ->
            let xs = parts c.dt x in
            let sums =
              Array.mapi
                (fun i xi ->
                  let leaves =
                    List.map (fun (j, u) -> (j, (parts c.dt u).(i))) us
                  in
                  let s = Option.get (tree c.dt (ceil_log2 c.m) leaves) in
                  let l =
                    match leaves with
                    | [] -> s
                    | (_, v) :: rest ->
                        List.fold_left (fun a (_, v) -> add c.dt a v) v rest
                  in
                  if Float.is_finite s && s <> l then shows := true;
                  add c.dt xi s)
                xs
            in
            of_parts c.dt sums)
      into
  in
  (want, !shows)

(* The device paths a case reaches, as moves.ml plans a float Add: row tiles of
   256 positions for complex numbers, as two parts, and idx broadcast along the
   last axis, else tiles of 1024; past one tile, the tiles' entries where a
   slice has at most a tile's targets, packed in parts of at least 256 tiles,
   else the updates, sorted by target in a pass each key byte, in parts of
   tiles of 4096 narrow items or 2048 wide ones; each target's run of entries,
   the longest, and whether one past [whole] entries spans blocks of 256
   positions, so that its sum takes levels. *)
type paths = {
  tiled : bool;
  from_tiles : bool;
  packed_parts : bool;
  pack_chunks : bool;
  split_cols : bool;
  sorted_rows : bool;
  narrow : bool;
  parts : int;
  passes : int;
  longest : int;
  levels : bool;
}

let paths c idx =
  let cx = complex c.dt in
  let rows = c.rows || cx in
  let tile = if rows then 256 else 1024 in
  let run = Int.min tile (1 lsl ceil_log2 c.m) in
  let nb = if c.m = 0 then 0 else (c.m + run - 1) / run in
  let tiled = nb > 1 && c.d > 0 && c.cols > 0 in
  let from_tiles = tiled && c.d <= tile in
  let parts = if cx then 2 else 1 in
  let last = if cx then 2 else if c.rows then c.cols else 1 in
  let vb = if wide c.dt then 8 else 4 in
  let tc =
    if rows then ceil_log2 (Int.min last (Int.min 32 (32768 / (256 * vb))))
    else 0
  in
  let nc = (last + (1 lsl tc) - 1) lsr tc in
  let nu = c.m * c.cols * parts and nd = c.d * c.cols * parts in
  (* Past one tile, a tile holds one slice's [run] positions. *)
  let slices = if c.m = 0 || last = 0 then 0 else nu / c.m / last in
  let tiles = slices * nb * nc in
  let entries = if from_tiles then tiles * c.d else nu / Int.max 1 last in
  let keys = if from_tiles && rows then nd / last * nc else nd / Int.max 1 last in
  let passes = if tiled then Int.max 1 ((ceil_log2 keys + 7) / 8) else 0 in
  let bparts, bwide =
    match c.bound with Some b -> (b.parts, b.wide) | None -> (256, false)
  in
  let narrow = (not bwide) && ceil_log2 keys + ceil_log2 entries <= 64 in
  let tile_items = 256 * if narrow then 16 else 8 in
  let packs = Int.min bparts ((tiles + 255) / 256) in
  (* A run's entries' positions: an entry an update, a row of updates, or a
     tile's entries of a target; and the slices' blocks of [run] positions
     that hold an entry. *)
  let runs = Hashtbl.create 64 and full = Hashtbl.create 64 in
  for j = 0 to c.m - 1 do
    for k = 0 to c.cols - 1 do
      let p = idx j k in
      if p >= 0 && p < c.d then begin
        let t = if c.axis = 0 then (p * c.cols) + k else (k * c.d) + p in
        let key = if c.rows && not cx then p else t in
        let pos = if from_tiles then j / run else j in
        Hashtbl.replace full ((if c.rows && not cx then 0 else k), j / run) ();
        let ps =
          match Hashtbl.find_opt runs key with
          | Some ps -> ps
          | None ->
              let ps = Hashtbl.create 4 in
              Hashtbl.replace runs key ps;
              ps
        in
        Hashtbl.replace ps pos ()
      end
    done
  done;
  let whole =
    match c.bound with Some b -> b.whole | None -> 64
  in
  let longest = ref 0 and levels = ref false in
  Hashtbl.iter
    (fun _ ps ->
      let n = Hashtbl.length ps in
      longest := Int.max !longest n;
      let block = ref (-1) and apart = ref false in
      Hashtbl.iter
        (fun p () ->
          if !block >= 0 && p lsr 8 <> !block then apart := true;
          block := p lsr 8)
        ps;
      if n > whole && !apart then levels := true)
    runs;
  {
    tiled;
    from_tiles;
    packed_parts = from_tiles && packs > 1;
    pack_chunks =
      from_tiles
      && (tiles + packs - 1) / packs > 256
      && Hashtbl.length full * nc < tiles;
    split_cols = from_tiles && nc > 1;
    sorted_rows = tiled && (not from_tiles) && rows;
    narrow;
    parts = Int.min bparts ((entries + tile_items - 1) / tile_items);
    passes;
    longest = (if tiled then !longest else 0);
    levels = tiled && !levels;
  }

(* idx's positions along the axis, slice [k]'s at [j], as [c.dist] draws
   them. *)
let draw_positions c rs ~slices =
  let out () =
    if Random.State.int rs 20 = 0 then if Random.State.bool rs then -1 else c.d
    else -2
  in
  let uniform () = Random.State.int rs (c.d + 2) - 1 in
  Array.init slices (fun k ->
      match c.dist with
      | Uniform -> Array.init c.m (fun _ -> uniform ())
      | One ->
          let t = if c.d = 0 then 0 else Random.State.int rs c.d in
          Array.make c.m t
      | Perm ->
          let ps = Array.init (c.d + 2) (fun i -> i - 1) in
          for i = Array.length ps - 1 downto 1 do
            let r = Random.State.int rs (i + 1) in
            let x = ps.(i) in
            ps.(i) <- ps.(r);
            ps.(r) <- x
          done;
          Array.init c.m (fun j ->
              if j < Array.length ps then ps.(j) else uniform ())
      | Holes when k mod 4 < 2 ->
          Array.init c.m (fun _ -> if Random.State.bool rs then -1 else c.d)
      | Holes ->
          Array.init c.m (fun _ -> if c.d = 0 then 0 else Random.State.int rs c.d)
      | Zipf ->
          Array.init c.m (fun _ ->
              match out () with
              | -2 when c.d > 0 ->
                  let u = Random.State.float rs 1. in
                  Int.min (c.d - 1)
                    (Float.to_int (Float.pow (Float.of_int c.d) u) - 1)
              | -2 -> 0
              | p -> p))

(* The law *)

let law c =
  let b = amd () in
  let module K = (val b.kernels) in
  cover "an axis of one tile" (c.m > 0 && c.m <= 256);
  cover "row tiles" c.rows;
  cover "targets of several updates" (c.m > c.d);
  cover "in place" c.in_place;
  cover "a narrow float" (narrow c.dt);
  cover "no update, target or column" (c.m = 0 || c.d = 0 || c.cols = 0);
  cover "NaNs and infinities" c.specials;
  cover "views" (viewed c);
  cover "each target at most one update" (c.dist = Perm && c.m <= c.d + 2);
  cover "every update at one target" (c.dist = One);
  cover "log-uniform positions" (c.dist = Zipf);
  cover "slices out of range" (c.dist = Holes);
  let rs = Random.State.make [| c.seed |] in
  let ts = if c.axis = 0 then [| c.d; c.cols |] else [| c.cols; c.d |] in
  let us = if c.axis = 0 then [| c.m; c.cols |] else [| c.cols; c.m |] in
  let value _ = draw ~specials:c.specials c.dt rs in
  let (A.Any into_a) = operand c.dt ts c.vi value in
  let u = operand c.dt us c.vu value in
  let into = elements (A.Any into_a) and ups = elements u in
  let slices = if c.rows then 1 else c.cols in
  let table = draw_positions c rs ~slices in
  let at j k = if c.axis = 0 then (j * c.cols) + k else (k * c.m) + j in
  let idx_a =
    if c.rows then
      Option.get
        (A.move
           (M.Broadcast (Iarray.of_array us))
           (A.of_array D.Int64 [| c.m; 1 |]
              (Array.init c.m (fun j -> Int64.of_int table.(0).(j)))))
    else if c.dist = Uniform then
      (* A view of idx, drawn only with uniform positions, makes more
         elements than idx holds: each made element is drawn. *)
      let (A.Any p) = int64s us c.vx (fun () -> position rs c.d) in
      A.expect D.Int64 (A.Any p)
    else
      let lin = Array.make (c.m * c.cols) 0L in
      for j = 0 to c.m - 1 do
        for k = 0 to c.cols - 1 do
          lin.(at j k) <- Int64.of_int table.(k).(j)
        done
      done;
      let n = ref (-1) in
      let (A.Any p) =
        int64s us c.vx (fun () ->
            incr n;
            lin.(!n))
      in
      equal ~msg:"idx's elements, each drawn once" int (Array.length lin)
        (!n + 1);
      A.expect D.Int64 (A.Any p)
  in
  (* A position out of range is -1: Int64.to_int keeps int64's extremes in
     no int. *)
  let pos x =
    if Int64.compare x 0L < 0 || Int64.compare x (Int64.of_int c.d) >= 0 then -1
    else Int64.to_int x
  in
  let ids = Array.map pos (positions (A.Any idx_a)) in
  let idx j k = ids.(at j k) in
  let upd j k = ups.(at j k) in
  let want, shows = expected c into idx upd in
  let p = paths c idx in
  cover "a wide sum another association rounds otherwise" (wide c.dt && shows);
  cover "the updates sorted by target" (p.tiled && not p.from_tiles);
  cover "rows of updates sorted by target row" p.sorted_rows;
  cover "the tiles' entries sorted by target" p.from_tiles;
  cover "the tiles' entries packed in several parts" p.packed_parts;
  cover "a sort of one pass" (p.passes = 1);
  cover "a sort of two passes" (p.passes = 2);
  cover "a sort of three passes" (p.passes = 3);
  cover "a run past 64 entries" (p.longest > 64);
  cover "a run summed over levels" p.levels;
  let groups = match c.bound with Some b -> b.groups | None -> 1024 in
  cover "a sort of several parts, one group"
    (p.tiled && p.parts > 1 && groups = 1);
  cover "wide items" (p.tiled && not p.narrow);
  cover "wide items of tiles' entries" (p.from_tiles && not p.narrow);
  cover "wide items of a float32 sum" (p.tiled && (not p.narrow) && not (wide c.dt));
  cover "row tiles that split their columns" p.split_cols;
  cover "a pack's part past 256 tiles, some empty" p.pack_chunks;
  let into_a = Support.on b into_a in
  let u = Support.on b (A.expect (A.dtype into_a) u) in
  let idx_a = Support.on b idx_a in
  let dst, into_a =
    if c.in_place then
      let d = Support.on b (A.copy (Support.host into_a)) in
      (d, d)
    else (A.create b.device (A.dtype into_a) (Iarray.of_array ts), into_a)
  in
  let call () =
    done_ (K.scatter Add ~unique:false ~axis:c.axis ~dst ~into:into_a idx_a u)
  in
  (match c.bound with
  | None -> call ()
  | Some bd ->
      Fun.protect
        ~finally:(fun () ->
          Nx_amd.bound_scatters ~parts:256 ~groups:1024 ~whole:64 ~wide:false)
        (fun () ->
          Nx_amd.bound_scatters ~parts:bd.parts ~groups:bd.groups
            ~whole:bd.whole ~wide:bd.wide;
          call ()));
  equal (array (same c.dt)) want (elements (A.Any dst))

(* Complex64 aligned to 4 bytes *)

(* A complex64 array of shape [s] holding [es], on [b]'s device with its first
   element at 4 mod 8 bytes, or on the host for nx.cpu. *)
let c64 (b : Support.backend) s es =
  let h = A.expect D.Complex64 (of_elements (D.Any D.Complex64) s es) in
  if Rig.equal b.device Rig.host then h
  else
    let n = total s in
    let buf = Rig.Buffer.create b.device ((8 * n) + 8) in
    let view = Rig.Buffer.view buf ~first:4 ~length:(8 * n) in
    let a = A.v D.Complex64 (L.contiguous (Iarray.of_array s)) view in
    A.transfer ~src:h ~dst:a;
    a

let ints (b : Support.backend) s xs =
  Support.on b
    (A.of_array D.Int64 (Iarray.of_array s) (Array.map Int64.of_int xs))

let c64s rs n =
  Array.init n (fun _ ->
      let part () =
        if Random.State.int rs 8 = 0 then -0. else value (D.Any D.Float32) rs
      in
      f32 (round32 (part ())) ^ f32 (round32 (part ())))

(* Each move of complex64 arrays at 4 mod 8 bytes, by [b], as the elements of
   its result. *)
let misaligned_moves (b : Support.backend) =
  let module K = (val b.kernels) in
  let rs () = Random.State.make [| 7 |] in
  let fresh s = c64 b s (Array.make (total s) (String.make 8 '\000')) in
  let scatter c ~unique ?(in_place = false) ids =
    let r = rs () in
    let into = c64 b [| 5; 2 |] (c64s r 10) in
    let u = c64 b [| 4; 2 |] (c64s r 8) in
    let dst = if in_place then into else fresh [| 5; 2 |] in
    done_ (K.scatter c ~unique ~axis:0 ~dst ~into (ints b [| 4; 2 |] ids) u);
    dst
  in
  let same_ids = [| 1; 0; 1; 4; 1; 4; 3; -1 |] in
  let distinct = [| 0; 4; 2; 1; 4; 0; 1; 3 |] in
  [
    ( "gather",
      fun () ->
        let x = c64 b [| 6; 3 |] (c64s (rs ()) 18) and dst = fresh [| 5; 3 |] in
        let idx =
          ints b [| 5; 3 |] (Array.init 15 (fun i -> (i * 7 mod 8) - 1))
        in
        done_ (K.gather ~axis:0 ~dst idx x);
        dst );
    ( "gather_ranges",
      fun () ->
        let x = c64 b [| 8; 2 |] (c64s (rs ()) 16) and dst = fresh [| 6; 2 |] in
        done_
          (K.gather_ranges ~dst
             ~starts:(ints b [| 3 |] [| 5; 0; 2 |])
             ~ends:(ints b [| 3 |] [| 2; 3; 7 |])
             x);
        dst );
    ("scatter Set", fun () -> scatter Set ~unique:false same_ids);
    ("scatter Set ~unique", fun () -> scatter Set ~unique:true distinct);
    ("scatter Max", fun () -> scatter Max ~unique:false same_ids);
    ( "scatter Min, in place",
      fun () -> scatter Min ~unique:false ~in_place:true same_ids );
    ("scatter Add", fun () -> scatter Add ~unique:false same_ids);
    ("scatter Add ~unique", fun () -> scatter Add ~unique:true distinct);
    ( "assemble",
      fun () ->
        let r = rs () in
        let p0 = c64 b [| 2; 3 |] (c64s r 6)
        and p1 = c64 b [| 3; 2 |] (c64s r 6) in
        let dst = fresh [| 4; 4 |] in
        let g start count = { M.start; count; step = 1 } in
        done_
          (K.assemble
             ~fill:{ Complex.re = 1.5; im = -0. }
             ~dst
             [|
               (Iarray.of_array [| g 0 2; g 0 3 |], p0);
               (Iarray.of_array [| g 1 3; g 2 2 |], p1);
             |]);
        dst );
    ( "fold",
      fun () ->
        let pad =
          {
            Nx_kernel.lo = Iarray.of_array [| 1; 0 |];
            hi = Iarray.of_array [| 0; 1 |];
            interior = Iarray.of_array [| 0; 1 |];
            windows =
              Iarray.of_array
                [| { M.axis = 1; size = 2; step = 1; dilation = 2 } |];
          }
        in
        let xs = Result.get_ok (Nx_kernel.Shape.padded pad [| 3; 3 |]) in
        let x = c64 b xs (c64s (rs ()) (total xs)) and dst = fresh [| 3; 3 |] in
        done_ (K.fold pad ~dst x);
        dst );
  ]

let test_misaligned () =
  let want =
    List.map
      (fun (n, f) -> (n, elements (A.Any (f ()))))
      (misaligned_moves Support.cpu)
  in
  List.iter2
    (fun (n, w) (_, f) ->
      equal ~msg:n
        (array (same (D.Any D.Complex64)))
        w
        (elements (A.Any (f ()))))
    want
    (misaligned_moves (amd ()))

(* Rows *)

(* A move whose rows are runs: a gather or range gather of rows, a copy (a Set
   scatter of no update), or an assembly of pieces of rows, of [rows] rows of
   [n] elements, each array's first element [dst_at] or [src_at] bytes into its
   buffer; the sources' rows [extra] elements apart past [n], from column [col].
   An assembly's pieces lie in their regions along the last axis by [steps]. *)
type op = Gather | Ranges | Copy | Assemble

type row_case = {
  op : op;
  rdt : D.any;
  rows : int;
  n : int;
  dst_at : int;
  src_at : int;
  extra : int;
  col : int;
  steps : int list;
  rseed : int;
}

let op_name = function
  | Gather -> "gather"
  | Ranges -> "range gather"
  | Copy -> "copy"
  | Assemble -> "assembly"

let pp_row_case ppf c =
  Format.fprintf ppf
    "%s of %a, %d rows of %d, dst at byte %d, src at byte %d, rows %d apart \
     from column %d, steps %s (seed %d)"
    (op_name c.op) pp_dtype c.rdt c.rows c.n c.dst_at c.src_at (c.n + c.extra)
    c.col
    (String.concat " " (List.map string_of_int c.steps))
    c.rseed

(* An element's bytes and the alignment of its units. *)
let el_bytes (D.Any dt) = D.bits dt / 8

let unit_bytes (D.Any dt as d) =
  if D.is D.Complex dt then el_bytes d / 2 else el_bytes d

let row_dtypes =
  D.
    [
      Any Uint8;
      Any Int16;
      Any Bfloat16;
      Any Float32;
      Any Int64;
      Any Complex64;
      Any Complex128;
    ]

(* Runs below 256 bytes, around it, and past one chunk of 2048 bytes. Half
   the cases lay every row at one offset modulo 16 in dst and src, that
   offset drawn. *)
let row_gen =
  let open Gen in
  let* op = of_list [ Gather; Ranges; Copy; Assemble ] in
  let* rdt = of_list ~pp:pp_dtype row_dtypes in
  let es = el_bytes rdt and ub = unit_bytes rdt in
  let* bytes =
    frequency
      [ (2, int_range 1 255); (3, int_range 240 600); (3, int_range 2040 5000) ]
  in
  let* rows = int_range 1 4 in
  let* alike = bool in
  let per = Int.max 1 (16 / es) in
  let n = Int.max 1 (bytes / es) in
  let n = if alike then Int.max per (n / per * per) else n in
  let at = map (fun k -> k * ub) (int_range 0 ((16 / ub) - 1)) in
  let* shared = at in
  let* dst_at = if alike then constant shared else at in
  let* src_at = if alike then constant shared else at in
  let* extra =
    if alike then map (fun k -> k * per) (int_range 0 2) else int_range 0 5
  in
  let* col =
    if alike then map (fun k -> k * per) (int_range 0 (extra / per))
    else int_range 0 extra
  in
  let* steps =
    list ~size:(int_range 1 3)
      (frequency [ (6, constant 1); (1, constant (-1)); (1, constant 2) ])
  in
  let+ rseed = int in
  { op; rdt; rows; n; dst_at; src_at; extra; col; steps; rseed }

let row_cases = Gen.with_pp pp_row_case row_gen

(* One case for each covered regime, so that every seed reaches each. *)
let row_examples =
  let ex op rdt ~rows ~n ?(dst_at = 0) ?(src_at = 0) ?(extra = 0) ?(col = 0)
      ?(steps = [ 1 ]) () =
    { op; rdt; rows; n; dst_at; src_at; extra; col; steps; rseed = 5 }
  in
  D.
    [
      ex Gather (Any Float32) ~rows:3 ~n:1100 ();
      ex Gather (Any Bfloat16) ~rows:2 ~n:300 ~dst_at:6 ~src_at:2 ~extra:3
        ~col:1 ();
      ex Ranges (Any Float32) ~rows:4 ~n:256 ~dst_at:4 ~src_at:4 ();
      ex Copy (Any Complex64) ~rows:2 ~n:700 ~dst_at:4 ~src_at:12 ~extra:2
        ~col:1 ();
      ex Assemble (Any Float32) ~rows:4 ~n:600 ~dst_at:8 ~src_at:8
        ~steps:[ 1; -1 ] ();
      ex Assemble (Any Uint8) ~rows:3 ~n:3000 ~dst_at:3 ~src_at:9 ();
      ex Gather (Any Int64) ~rows:2 ~n:400 ~dst_at:8 ~src_at:8 ();
      ex Copy (Any Uint8) ~rows:2 ~n:1008 ~dst_at:13 ~src_at:13 ();
      ex Assemble (Any Float32) ~rows:1 ~n:1000 ~steps:[ 1 ] ();
      ex Assemble (Any Float32) ~rows:2 ~n:600 ~steps:(List.init 17 (fun _ -> 1))
        ();
    ]

(* The byte every buffer [placed] makes holds outside its array. *)
let sentinel = '\xA5'

(* An array of [dt] and shape [s] holding [es], on [b]'s device with its
   first element [at] bytes into its buffer, the buffer's other bytes
   [sentinel], or on the host for nx.cpu; and a check that those bytes
   still hold it. *)
let guarded (type v s) (b : Support.backend) (dt : (v, s) D.t) sh ~at es =
  let h = A.expect dt (of_elements (D.Any dt) sh es) in
  if Rig.equal b.device Rig.host then (h, fun () -> ())
  else
    let bytes = el_bytes (D.Any dt) * total sh in
    let size = bytes + at + 16 in
    let buf = Rig.Buffer.create b.device size in
    Rig.Buffer.fill buf sentinel;
    let view = Rig.Buffer.view buf ~first:at ~length:bytes in
    let a = A.v dt (L.contiguous (Iarray.of_array sh)) view in
    A.transfer ~src:h ~dst:a;
    let check () =
      let got = Bytes.create size in
      Rig.Buffer.blit_to_bytes buf 0 got 0 size;
      let outside =
        Bytes.sub_string got 0 at ^ Bytes.sub_string got (at + bytes) 16
      in
      equal string (String.make (at + 16) sentinel) outside
    in
    (a, check)

let placed b dt sh ~at es = fst (guarded b dt sh ~at es)

(* [rows] rows of [n] elements from column [col] of rows [n + extra] long,
   [src_at] bytes into its buffer. *)
let source b dt c rs ~rows =
  let w = c.n + c.extra in
  let es = Array.init (rows * w) (fun _ -> element c.rdt rs) in
  let x = placed b dt [| rows; w |] ~at:c.src_at es in
  let g start count = { M.start; count; step = 1 } in
  Option.get (A.move (M.Slice (Iarray.of_list [ g 0 rows; g c.col c.n ])) x)

(* The case's move by [b], as the elements of its result, whether some row
   of it reads no source (zeros), and whether it is an assembly of one
   piece of step 1 whose rows are runs, which nx.amd copies by runs. *)
let row_move (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.rdt in
  let rs = Random.State.make [| c.rseed |] in
  (* dst starts at drawn elements, so that a row the move leaves unwritten
     fails, and its buffer's bytes outside it are checked once it is done. *)
  let check = ref (fun () -> ()) in
  let fresh s =
    let rd = Random.State.make [| c.rseed; 1 |] in
    let a, ch =
      guarded b dt s ~at:c.dst_at
        (Array.init (total s) (fun _ -> element c.rdt rd))
    in
    check := ch;
    a
  in
  let dst, zeros, runs =
    match c.op with
    | Gather ->
        let d = 1 + Random.State.int rs 4 in
        let x = source b dt c rs ~rows:d in
        let ps = Array.init c.rows (fun _ -> position rs d) in
        let idx =
          Option.get
            (A.move
               (M.Broadcast (Iarray.of_list [ c.rows; c.n ]))
               (Support.on b (A.of_array D.Int64 [| c.rows; 1 |] ps)))
        in
        let dst = fresh [| c.rows; c.n |] in
        done_ (K.gather ~axis:0 ~dst idx x);
        (dst, Array.exists (fun p -> p < 0L || p >= Int64.of_int d) ps, false)
    | Ranges ->
        let d = 1 + Random.State.int rs 4 in
        let x = source b dt c rs ~rows:d in
        (* Runs of 0 to 2 rows, [c.rows] in all, from rows -1 to d. *)
        let lens = ref [] and left = ref c.rows in
        while !left > 0 do
          let l = Int.min !left (Random.State.int rs 3) in
          lens := l :: !lens;
          left := !left - l
        done;
        let lens = Array.of_list !lens in
        let k = Array.length lens in
        let ends = Array.make k 0 in
        Array.iteri
          (fun i l -> ends.(i) <- (l + if i = 0 then 0 else ends.(i - 1)))
          lens;
        let starts = Array.init k (fun _ -> Random.State.int rs (d + 2) - 1) in
        let dst = fresh [| c.rows; c.n |] in
        done_
          (K.gather_ranges ~dst ~starts:(ints b [| k |] starts)
             ~ends:(ints b [| k |] ends) x);
        let outside = ref false in
        Array.iteri
          (fun i l -> if l > 0 && (starts.(i) < 0 || starts.(i) + l > d) then
              outside := true)
          lens;
        (dst, !outside, false)
    | Copy ->
        let into = source b dt c rs ~rows:c.rows in
        let u = A.create b.device dt [| 0; c.n |] in
        let dst = fresh [| c.rows; c.n |] in
        done_
          (K.scatter Set ~unique:true ~axis:0 ~dst ~into
             (ints b [| 0; c.n |] [||])
             u);
        (dst, false, false)
    | Assemble ->
        let dst = fresh [| c.rows; c.n |] in
        (* Piece i: rows from a drawn start, the last axis from a drawn
           column by its step; [spans_runs] counts pieces of step 1 whose
           rows are runs. *)
        let spans_runs = ref 0 in
        let piece step =
          let r0 = Random.State.int rs c.rows in
          let rc = 1 + Random.State.int rs (c.rows - r0) in
          let span =
            if Random.State.int rs 3 = 0 then c.n else 1 + Random.State.int rs c.n
          in
          let cc = (span + abs step - 1) / abs step in
          let c0 = Random.State.int rs (c.n - span + 1) in
          let start = if step < 0 then c0 + span - 1 else c0 in
          if step = 1 && cc * el_bytes c.rdt >= 256 then incr spans_runs;
          let src = source b dt { c with n = cc } rs ~rows:rc in
          let rows = { M.start = r0; count = rc; step = 1 } in
          (Iarray.of_list [ rows; { start; count = cc; step } ], src)
        in
        let pieces = Array.of_list (List.map piece c.steps) in
        let one = of_elements c.rdt [| 1 |] [| element c.rdt rs |] in
        let fill = A.get (A.expect dt one) [| 0 |] in
        done_ (K.assemble ~fill ~dst (Iarray.of_array pieces));
        (dst, false, Array.length pieces = 1 && !spans_runs = 1)
  in
  let es = elements (A.Any dst) in
  !check ();
  (es, zeros, runs)

let row_law c =
  let b = amd () in
  let es = el_bytes c.rdt in
  let alike =
    (c.src_at + (c.col * es) - c.dst_at) mod 16 = 0
    && c.n * es mod 16 = 0
    && (c.n + c.extra) * es mod 16 = 0
  in
  cover (op_name c.op) true;
  cover "rows of 256 bytes or more" (c.n * es >= 256);
  cover "rows past a chunk of 2048 bytes" (c.n * es > 2048 + 16);
  cover "rows below 256 bytes" (c.n * es < 256);
  cover "every row alike modulo 16 in dst and src" (alike && c.n * es >= 256);
  cover "rows unalike modulo 16" ((not alike) && c.n * es >= 256);
  let want, zeros, runs = row_move Support.cpu c in
  let got, _, _ = row_move b c in
  cover "a row of zeros, its position outside the axis" zeros;
  cover "an assembly copied by runs" runs;
  cover "rows alike modulo 16, off a 16-byte boundary"
    (alike && c.dst_at mod 16 <> 0 && c.n * es >= 256);
  cover "an assembly of more than 16 pieces" (List.length c.steps > 16);
  equal (array string) want got

(* Set *)

(* A Set scatter without [~unique] of [sm] updates by [scols] along axis 0
   into [sd] targets, [into] the destination itself where [same]. *)
type set_case = {
  sdt : D.any;
  sm : int;
  sd : int;
  scols : int;
  same : bool;
  sseed : int;
}

let pp_set_case ppf c =
  Format.fprintf ppf "%a: %d updates by %d into %d%s (seed %d)" pp_dtype c.sdt
    c.sm c.scols c.sd
    (if c.same then ", into dst" else "")
    c.sseed

(* An element nx_amd.mli packs with its position: 1, 2 or 4 bytes, not
   complex. *)
let packed (D.Any dt) =
  List.mem (D.bits dt) [ 8; 16; 32 ] && not (D.is D.Complex dt)

(* Every dtype; targets fewer than the updates often, so that positions
   repeat. *)
let set_cases =
  let open Gen in
  with_pp pp_set_case
    (let* sdt = of_list ~pp:pp_dtype D.all in
     let* sm = int_range 1 300 in
     let* sd = frequency [ (3, int_range 1 8); (1, int_range 1 400) ] in
     let* scols = int_range 1 3 in
     let* same = bool in
     let+ sseed = int in
     { sdt; sm; sd; scols; same; sseed })

let set_examples =
  let ex sdt ~same = { sdt; sm = 200; sd = 5; scols = 2; same; sseed = 3 } in
  D.
    [
      ex (Any Float32) ~same:false; ex (Any Uint8) ~same:true;
      ex (Any Bfloat16) ~same:false; ex (Any Float64) ~same:true;
      ex (Any Complex64) ~same:false; ex (Any Int4) ~same:false;
    ]

(* The case's scatter by [b], as the elements of its result. *)
let set_scatter (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.sdt in
  let rs = Random.State.make [| c.sseed |] in
  let on sh es = Support.on b (A.expect dt (of_elements c.sdt sh es)) in
  let target = [| c.sd; c.scols |] and upd = [| c.sm; c.scols |] in
  let into = on target (Array.init (total target) (fun _ -> element c.sdt rs)) in
  let u = on upd (Array.init (total upd) (fun _ -> element c.sdt rs)) in
  let ps = Array.init (total upd) (fun _ -> position rs c.sd) in
  let idx = Support.on b (A.of_array D.Int64 (Iarray.of_array upd) ps) in
  let dst =
    if c.same then into else on target (Array.make (total target) (zero c.sdt))
  in
  done_ (K.scatter Set ~unique:false ~axis:0 ~dst ~into idx u);
  elements (A.Any dst)

(* nx_amd.mli: a Set keeps each target's last update in C order, its bits as
   nx.cpu's, an element of 1, 2 or 4 bytes packed with its position. *)
let set_law c =
  let b = amd () in
  cover "an element of 1, 2 or 4 bytes" (packed c.sdt);
  cover "an element of 8 bytes or more, or complex"
    ((not (packed c.sdt)) && el_bytes c.sdt >= 4);
  cover "a sub-byte element" (el_bytes c.sdt = 0);
  cover "into dst" c.same;
  cover "into apart from dst" (not c.same);
  cover "more updates than targets" (c.sm > c.sd);
  equal (array string) (set_scatter Support.cpu c) (set_scatter b c)

(* Workspace *)

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024

(* Runs [f] with every call taking a workspace of its own. *)
let all_fresh f =
  let restore () =
    Nx_amd.bound_fresh ~workspace:kept_workspace ~tickets:kept_tickets
  in
  Fun.protect ~finally:restore (fun () ->
      Nx_amd.bound_fresh ~workspace:0 ~tickets:0;
      f ())

(* nx_amd.mli: a call's own buffers return to the device once its work is done,
   so calls in a loop hold one call's workspace, of the bytes its section
   states, at most [bound]. [setup b] makes the call's arrays once, outside the
   count, and gives the call. *)
let held ~bound setup =
  let b = amd () in
  let call = setup b in
  all_fresh (fun () ->
      Kernels_cases.loop_held b ~bound (fun () ->
          let (A.Any dst) = call () in
          Rig.Buffer.wait (A.buffer dst) Read_write))

(* A scatter by [c] of [m] updates of [dt] along axis 0 into [d] by [cols]
   targets, with [rows] idx a column broadcast along the last axis, giving its
   destination. *)
let scatter_held (type v s) ?(rows = false) c ~unique (dt : (v, s) D.t) ~m ~d
    ~cols (b : Support.backend) =
  let module K = (val b.kernels) in
  let rs = Random.State.make [| m; d |] in
  let into = A.create b.device dt [| d; cols |]
  and dst = A.create b.device dt [| d; cols |] in
  let u = A.create b.device dt [| m; cols |] in
  let idx =
    if rows then
      Option.get
        (A.move
           (M.Broadcast (Iarray.of_array [| m; cols |]))
           (ints b [| m; 1 |] (Array.init m (fun _ -> Random.State.int rs d))))
    else
      ints b [| m; cols |]
        (Array.init (m * cols) (fun _ -> Random.State.int rs d))
  in
  fun () ->
    done_ (K.scatter c ~unique ~axis:0 ~dst ~into idx u);
    A.Any dst

(* A [Set] without [~unique]: 8 bytes a target for an element of 1, 2 or 4
   bytes, 4 for others. *)
let test_set_held () =
  let d = 4096 and cols = 3 in
  held
    ~bound:(8 * d * cols)
    (scatter_held Set ~unique:false D.Uint8 ~m:100 ~d ~cols);
  held
    ~bound:(4 * d * cols)
    (scatter_held Set ~unique:false D.Float64 ~m:100 ~d ~cols)

(* A complex [Max] or [Min]: 8 bytes a target for complex64, 16 for
   complex128. *)
let test_extreme_held () =
  let d = 2048 and cols = 2 in
  let each (type v s) (dt : (v, s) D.t) bytes =
    List.iter
      (fun c ->
        held
          ~bound:(bytes * d * cols)
          (scatter_held c ~unique:false dt ~m:300 ~d ~cols))
      Nx_kernel.[ Max; Min ]
  in
  each D.Complex64 8;
  each D.Complex128 16

(* A float Add past one tile: at most 64 + 4 w bytes an update, w the bytes of
   an element, 4 for a narrow float, and 3 KiB, through each of its paths: the
   updates or rows of them sorted, where a slice has more than a tile's
   targets, and the tiles' entries, of row tiles too; each with the sort's
   items narrow where they fit and wide always. *)
let test_add_held () =
  let each ?rows (D.Any dt) ~m ~d ~cols =
    let w = Int.max 4 (D.bits dt / 8) in
    held
      ~bound:(((64 + (4 * w)) * m * cols) + 3072)
      (scatter_held ?rows Add ~unique:false dt ~m ~d ~cols)
  in
  let dts =
    D.[ Any Float32; Any Float64; Any Bfloat16; Any Complex64; Any Complex128 ]
  in
  let paths ~wide =
    Fun.protect
      ~finally:(fun () ->
        Nx_amd.bound_scatters ~parts:256 ~groups:1024 ~whole:64 ~wide:false)
      (fun () ->
        Nx_amd.bound_scatters ~parts:256 ~groups:1024 ~whole:64 ~wide;
        List.iter
          (fun dt ->
            each dt ~m:5000 ~d:3000 ~cols:3;
            each dt ~m:5000 ~d:500 ~cols:3;
            each ~rows:true dt ~m:3000 ~d:2000 ~cols:33;
            each ~rows:true dt ~m:3000 ~d:200 ~cols:33)
          dts)
  in
  paths ~wide:false;
  paths ~wide:true

(* A range gather: 24 bytes a run and 8 for each 4096 runs of a row. *)
let test_ranges_held () =
  let rows = 3 and k = 9000 and n = 40 in
  let runs = rows * k in
  let chunks = (k + 4095) / 4096 in
  held
    ~bound:((24 * runs) + (8 * rows * chunks))
    (fun b ->
      let module K = (val b.kernels) in
      let x = A.create b.device D.Float32 [| rows; n |] in
      let dst = A.create b.device D.Float32 [| rows; n |] in
      let starts = ints b [| rows; k |] (Array.make runs 0) in
      let ends = ints b [| rows; k |] (Array.init runs (fun i -> i mod n)) in
      fun () ->
        done_ (K.gather_ranges ~dst ~starts ~ends x);
        A.Any dst)

(* Long axes *)

(* nx_amd.mli: a fold computes every case Nx_kernel.S offers, an axis of
   2^32 elements among them, past the divisors of 32 bits: int4 ones from a
   broadcast one, 2 GiB, read back at both ends. *)
let test_fold_long () =
  let b = amd () in
  let module K = (val b.kernels) in
  let n = 1 lsl 32 in
  let dst = A.create b.device D.Int4 [| n |] in
  let one = Support.on b (A.of_array D.Int4 [| 1 |] [| 1 |]) in
  let x = Option.get (A.move (M.Broadcast (Iarray.of_list [ n ])) one) in
  let pad =
    {
      Nx_kernel.lo = [| 0 |];
      hi = [| 0 |];
      interior = [| 0 |];
      windows = Iarray.of_list [];
    }
  in
  done_ (K.fold pad ~dst x);
  let buf = A.buffer dst and ends = 16 and bytes = n / 2 in
  Rig.Buffer.wait buf Read_write;
  let got = Bytes.create (2 * ends) in
  Rig.Buffer.blit_to_bytes buf 0 got 0 ends;
  Rig.Buffer.blit_to_bytes buf (bytes - ends) got ends ends;
  equal string (String.make (2 * ends) '\x11') (Bytes.to_string got)

(* Kept submissions *)

(* nx_amd.mli: a device keeps an assembly's submission whatever its launches,
   and past the bytes it keeps drops the least recently used. [pieces n]
   assembles [n] one-element pieces covering a rank-1 dst, a copy launch each
   16, so that assemblies of more pieces take more bytes. *)
let test_assemble_kept () =
  let b = amd () in
  let module K = (val b.kernels) in
  let pieces n =
    let dst = A.create b.device D.Int32 [| n |] in
    let one i =
      let x = Support.on b (A.of_array D.Int32 [| 1 |] [| Int32.of_int i |]) in
      (Iarray.of_list [ { A.Move.start = i; count = 1; step = 1 } ], x)
    in
    let ps = Iarray.init n one in
    fun () ->
      done_ (K.assemble ~fill:0l ~dst ps);
      equal (array int32)
        (Array.init n Int32.of_int)
        (A.to_array (Support.host dst))
  in
  let d = b.device in
  let one = pieces 16 and two = pieces 32 and nine = pieces ((8 * 16) + 1) in
  (* A call's bytes: those a device keeps once it makes it alone. *)
  let bytes call =
    Nx_amd.drop_kept d;
    call ();
    call ();
    equal ~msg:"kept calls" int 1 (Nx_amd.kept_calls d);
    Nx_amd.kept_bytes d
  in
  let b1 = bytes one and b2 = bytes two and b9 = bytes nine in
  less ~msg:"one launch's bytes" int ~than:b2 b1;
  less ~msg:"two launches' bytes" int ~than:b9 b2;
  let kept ~msg ~calls n =
    equal ~msg:(msg ^ ", calls") int calls (Nx_amd.kept_calls d);
    equal ~msg:(msg ^ ", bytes") int n (Nx_amd.kept_bytes d)
  in
  Fun.protect
    ~finally:(fun () -> Nx_amd.bound_kept Nx_amd.Plan.kept_submissions)
    (fun () ->
      Nx_amd.bound_kept (b1 + b9);
      Nx_amd.drop_kept d;
      one ();
      two ();
      one ();
      kept ~msg:"one and two" ~calls:2 (b1 + b2);
      nine ();
      kept ~msg:"two dropped" ~calls:2 (b1 + b9);
      one ();
      kept ~msg:"one found" ~calls:2 (b1 + b9);
      two ();
      kept ~msg:"nine dropped" ~calls:2 (b1 + b2))

let () =
  exit
    (Windtrap.run "nx.amd moves"
       [
         group "moves"
           [
             prop ~examples
               "a float Add is x + S, S the position tree's sum of the updates"
               cases law;
             test "moves complex64 aligned to 4 bytes as nx.cpu does"
               test_misaligned;
             test "assemblies are kept, the least recently used dropped"
               test_assemble_kept;
             test "a fold over an axis of 2^32 elements computes"
               test_fold_long;
             prop ~examples:row_examples
               "a move of rows gives nx.cpu's bits, its rows at any offset"
               row_cases row_law;
             prop ~examples:set_examples
               "a Set keeps each target's last update, nx.cpu's bits"
               set_cases set_law;
           ];
         group "workspace"
           [
             test "a Set holds 8 or 4 bytes a target" test_set_held;
             test "a complex Max or Min holds 8 or 16 bytes a target"
               test_extreme_held;
             test "a float Add holds its stated bytes" test_add_held;
             test "a range gather holds 24 bytes a run" test_ranges_held;
           ];
       ])
