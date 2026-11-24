(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tiles

   A group of four SIMD-groups computes a tile of outputs. Each k step it stages
   [depth] terms of a's tile rows and of b's tile columns in threadgroup memory,
   each staged row padded by 16 bytes. *)

type stage = F32 | F16 | Bf16 | I8
type tile = T64 | T32 | T16 | G32 | G16

let rows = function T64 -> 64 | T32 | G32 -> 32 | T16 | G16 -> 16
let cols = function T64 | T16 | G32 | G16 -> 64 | T32 -> 32

let depth s t =
  match (s, t) with
  | _, (T16 | G16) | (F16 | I8), _ -> 32
  | (F32 | Bf16), (T64 | T32 | G32) -> 16

let width = function F32 -> 4 | F16 | Bf16 | I8 -> 2

(* The bytes of an operand's element a tile reads, and of a sum it stores. *)
let read_width = function I8 -> 1 | s -> width s
let sum_width = function I8 -> 8 | F32 | F16 | Bf16 -> 4
let padding = 16

let shared ~nk s t =
  let e = width s in
  let p = padding / e in
  let bk = depth s t in
  let a = rows t * (bk + p) in
  let b = if nk then cols t * (bk + p) else bk * (cols t + p) in
  (a + b) * e

let threads = Kernels.contract_threads

(* The cores of the M1 Max, the family's GPU whose waves G counts. *)
let wave_cores = 32
let spread s t = wave_cores * Plan.holds ~threads (shared ~nk:false s t)

(* Plans *)

type extents = {
  batch : int;
  m : int;
  n : int;
  k : int;
  stage : stage;
  nk : bool;
  complex : bool;
  source : int;
}

type launch = {
  first : int;
  tiles : int;
  chunk : int;
  length : int;
  seconds : float;
}

type t = {
  exchanged : bool;
  rows_out : int;
  cols_out : int;
  tile : tile;
  ranges : int;
  spans : int;
  tiles_m : int;
  tiles_n : int;
  swizzle : int;
  padded : int;
  resident : int;
  launches : launch array;
  seconds : float;
}

let span = Kernels.contract_span
let most_ranges = 64
let range_spans = 8
let big = 96

(* The tiles at which a call takes 64 x 64 tiles: [big], fewer in tests. *)
let tiles_64 = Atomic.make big

let bound_big n =
  if n < 1 || n > big then invalid_arg "Contract.bound_big";
  Atomic.set tiles_64 n;
  Plan.renew ()

let swizzled = 8
let swizzle_log = 2
let ceil_div a b = (a + b - 1) / b
let rec gcd a b = if b = 0 then a else gcd b (a mod b)
let lcm a b = a / gcd a b * b

(* The fraction of the lanes' peak a tile reaches, as its inverse: the
   multiply-adds a lane and a cycle at its compute-bound bench row on the M1
   Max, rounded down to a power of two. The 64 x 64 and 32 x 32 tiles reach
   0.77-0.93, the 16 x 64 tile 0.30, bound by memory. *)
let derate = function T64 | T32 | G32 -> 2 | T16 | G16 -> 4

(* A loader's work on an element it stages, in float32 operations, an upper
   bound. The figure was measured on the converting tiles that a mixed call no
   longer takes, which read each element through a run-time switch on its dtype
   and cast it: 289-490 a staged element at 1024³ on the M1 Max, 370-441 on the
   Rows tiles, 491 at the slowest of 50 calls plus their spread. A loader reads
   its weight's dtype without the switch, so 512 errs high for it, and is kept
   until the loaders' own rows measure their charge. *)
let cast_ops = 512

(* The terms of [p]'s longest range, at most [k]. *)
let whole (e : extents) p = Int.min e.k (span * ceil_div p.spans p.ranges)

(* The terms a group of [l] takes: its chunk's, or its range's. *)
let terms e p l = if l.length = 0 then whole e p else l.length

(* The weights of a launch [l] of [p] on a GPU of [cores] cores, each of its
   groups a tile over [terms] terms. A wave of [w] groups in the call's order
   covers [down] tile rows and [across] tile columns, whose staged slices it
   reads once a k step: the GPU's caches hold a wave's slices, which the bench
   checks. A grouped tile's row may be of another group than its neighbours', so
   each such tile reads b's slice of its own. Every group writes its tile's sums
   as float32, and reads them first where it resumes a chunk. A loader reads
   [e.source] bytes an element of its weight, and casts each element it stages.
   A complex tile reads b's real form from its complex64 elements, 8 bytes for 4
   staged ones. *)
let weights ~cores e p l =
  let bm = rows p.tile and bn = cols p.tile in
  let c = terms e p l in
  let groups = l.tiles * p.ranges in
  let wave = p.resident * cores in
  let w = Int.min groups wave in
  let across = Int.min p.tiles_n (ceil_div w (1 lsl p.swizzle)) in
  let down = ceil_div w across in
  let waves = ceil_div groups wave in
  let reads_a = waves * down * bm in
  let reads_b =
    match p.tile with
    | G32 | G16 -> groups * bn
    | T64 | T32 | T16 -> waves * across * bn
  in
  let sums =
    groups * bm * bn * sum_width e.stage * if l.chunk > 0 then 2 else 1
  in
  let read = if e.source > 0 then e.source else read_width e.stage in
  let read_b = if e.complex then read / 2 else read in
  let casts = if e.source > 0 then groups * (bm + bn) * c * cast_ops else 0 in
  {
    Plan.bytes = (reads_a * c * read) + (reads_b * c * read_b) + sums;
    ops = (groups * bm * bn * c * derate p.tile) + casts;
    chain = c;
  }

let seconds e p l =
  let groups = (l.tiles, 1, p.ranges) in
  let w = weights ~cores:Plan.cores e p l in
  Plan.seconds ~threads ~groups ~resident:p.resident w

let estimate e p ~bandwidth ~cores ~op_rate l =
  Plan.estimate ~bandwidth ~cores ~op_rate ~threads
    ~groups:(l.tiles, 1, p.ranges) ~resident:p.resident (weights ~cores e p l)

(* The largest [j] in [0, most] with [fits j], for [fits] true up to some [j]
   and false past it, [fits 0] taken true. *)
let largest fits most =
  let rec go lo hi =
    if lo >= hi then lo
    else
      let mid = lo + ((hi - lo + 1) / 2) in
      if fits mid then go mid hi else go lo (mid - 1)
  in
  go 0 most

(* Launches of [p]'s [total] tiles in windows of [size], the last the rest, each
   over [chunks] chunks of [length] terms. *)
let windows e p ~total ~chunks ~length size =
  let launch first chunk =
    let tiles = Int.min size (total - first) in
    let l = { first; tiles; chunk; length; seconds = 0. } in
    { l with seconds = seconds e p l }
  in
  List.concat
    (List.init (ceil_div total size) (fun i ->
         List.init chunks (launch (i * size))))

(* The launches of [p]'s [total] tiles in units of [u]: whole ranges in windows
   of as many units as fit; else windows of one unit, each over chunks of as
   many spans as fit; else [None]. Chunks of at most [most] terms, a multiple of
   the span, are taken always: windows of as many units as fit over chunks of
   [most] terms or a range's, else one unit's over chunks of as many spans as
   fit. *)
let cut e p ~total ~allowed ?most u =
  let fits ~tiles ~chunk ~length =
    seconds e p { first = 0; tiles; chunk; length; seconds = 0. } <= allowed
  in
  let whole = whole e p in
  let spans = ceil_div whole span in
  let chunked length =
    let wide j = fits ~tiles:(Int.min total (j * u)) ~chunk:1 ~length in
    match largest wide (ceil_div total u) with
    | 0 -> None
    | j ->
        Some
          (windows e p ~total ~chunks:(ceil_div whole length) ~length (j * u))
  in
  let deep most =
    let tiles = Int.min total u in
    let fits_q q = fits ~tiles ~chunk:1 ~length:(q * span) in
    match largest fits_q (Int.min spans (most / span)) with
    | 0 -> None
    | q ->
        let length = q * span in
        let chunks = ceil_div whole length in
        Some (windows e p ~total ~chunks ~length u)
  in
  match most with
  | Some most -> (
      match chunked (Int.min most (spans * span)) with
      | Some ls -> Some ls
      | None -> deep most)
  | None ->
      let wide j = fits ~tiles:(Int.min total (j * u)) ~chunk:0 ~length:0 in
      let j = largest wide (ceil_div total u) in
      if j > 0 then Some (windows e p ~total ~chunks:1 ~length:0 (j * u))
      else deep (spans * span)

(* The tile: 16 x 64 for few rows, 64 x 64 where the call has [tiles_64] of
   them, 32 x 32 otherwise. *)
let tile_of ~batch ~rows_out ~cols_out =
  if rows_out <= 16 then T16
  else if
    batch * ceil_div rows_out 64 * ceil_div cols_out 64 >= Atomic.get tiles_64
  then T64
  else T32

(* The ranges: one where the call's tiles fill half the GPU's waves; else as
   many as fill one, each of at least [range_spans] spans. *)
let ranges_of ~tiles ~spread ~spans =
  if tiles = 0 || 2 * tiles >= spread then 1
  else
    Int.max 1
      (Int.min most_ranges
         (Int.min (spans / range_spans) (ceil_div spread tiles)))

let exchanges ~m ~n = n <= 16 && m > 16

(* [p] with the launches of its [total] tiles, in units of [lcm (G, block)],
   [block] or one tile; [None] where none fits. Where [most] is given, its value
   at [p]'s ranges bounds the chunks' terms and chunks are taken always
   ({!cut}). *)
let launched ?most e p ~total ~block =
  if e.k = 0 || total = 0 then Some p
  else
    let allowed = Plan.allowed () in
    let g = spread e.stage p.tile in
    let most = match most with Some f -> Some (f p.ranges) | None -> None in
    match
      List.find_map (cut e p ~total ~allowed ?most) [ lcm g block; block; 1 ]
    with
    | None -> None
    | Some ls ->
        let launches = Array.of_list ls in
        let seconds =
          Array.fold_left (fun s (l : launch) -> s +. l.seconds) 0. launches
        in
        Some { p with launches; seconds }

(* The plan of [e], its ranges [min s ⌈k / 32⌉] for [?ranges:s], else
   {!ranges_of}'s. Its exchange, tile and ranges are those of the call of
   [whole]'s batch, rows and columns, [e]'s own by default: a piece of a call
   keeps the call's ranges, and so its bits. *)
let planned ?ranges ?whole ?most (e : extents) =
  let wb, wm, wn =
    match whole with Some w -> w | None -> (e.batch, e.m, e.n)
  in
  let exchanged = (not e.complex) && exchanges ~m:wm ~n:wn in
  let rows_out = if exchanged then e.n else e.m in
  let cols_out = if exchanged then e.m else e.n in
  let wrows = if exchanged then wn else wm in
  let wcols = if exchanged then wm else wn in
  let tile = tile_of ~batch:wb ~rows_out:wrows ~cols_out:wcols in
  let tiles_m = ceil_div rows_out (rows tile) in
  let tiles_n = ceil_div cols_out (cols tile) in
  let spans = ceil_div e.k span in
  let g = spread e.stage tile in
  let ranges =
    match ranges with
    | Some s -> Int.max 1 (Int.min s spans)
    | None ->
        let tiles =
          wb * ceil_div wrows (rows tile) * ceil_div wcols (cols tile)
        in
        ranges_of ~tiles ~spread:g ~spans
  in
  let swizzle =
    if tiles_m >= swizzled && tiles_n >= swizzled then swizzle_log else 0
  in
  let padded = ceil_div tiles_m (1 lsl swizzle) * (tiles_n lsl swizzle) in
  let resident = Plan.holds ~threads (shared ~nk:e.nk e.stage tile) in
  let p =
    {
      exchanged;
      rows_out;
      cols_out;
      tile;
      ranges;
      spans;
      tiles_m;
      tiles_n;
      swizzle;
      padded;
      resident;
      launches = [||];
      seconds = 0.;
    }
  in
  launched ?most e p ~total:(e.batch * padded) ~block:(tiles_n lsl swizzle)

let plan e = planned e

(* The tile rows a Rows call's tiles take at most, for [rows] rows in [buckets]
   buckets, tiles of [bm] rows: a bucket of c rows takes ⌈c / bm⌉, and at most
   [min buckets rows] buckets hold rows. *)
let bound ~rows ~buckets bm =
  if rows = 0 then 0 else ceil_div rows bm + Int.min buckets rows - 1

let grouped_of ?whole ?most (e : extents) ~groups =
  let wb, wm, wn =
    match whole with Some w -> w | None -> (e.batch, e.m, e.n)
  in
  let tile = if wm >= 16 * groups then G32 else G16 in
  let tiles_rows b m =
    bound ~rows:(b * m) ~buckets:((b * groups) + 1) (rows tile)
  in
  let tiles_m = tiles_rows e.batch e.m in
  let tiles_n = ceil_div e.n (cols tile) in
  let spans = ceil_div e.k span in
  let padded = tiles_m * tiles_n in
  let ranges =
    let tiles = tiles_rows wb wm * ceil_div wn (cols tile) in
    ranges_of ~tiles ~spread:(spread e.stage tile) ~spans
  in
  let resident = Plan.holds ~threads (shared ~nk:e.nk e.stage tile) in
  let p =
    {
      exchanged = false;
      rows_out = e.m;
      cols_out = e.n;
      tile;
      ranges;
      spans;
      tiles_m;
      tiles_n;
      swizzle = 0;
      padded;
      resident;
      launches = [||];
      seconds = 0.;
    }
  in
  launched ?most e p ~total:padded ~block:tiles_n

let grouped e ~groups = grouped_of e ~groups

(* The tiles' plan of [e]: a Rows call's ([segmented]) of [groups] groups, or a
   plain one's, its chunks bounded by [most] ({!launched}). *)
let tiled ?whole ?most ~segmented ~groups e =
  if segmented then grouped_of ?whole ?most e ~groups
  else planned ?whole ?most e

(* Vector kernels

   A call of few rows over a weight of a key (contract.mli) takes the vector
   kernels: an output's spans dealt to [S] lanes, a threadgroup a column group
   and one pass of rows, the grid's x the passes, launched first: a column
   group's passes run together and share its weight in the caches. The n x k
   kernels' threadgroup is four SIMD-groups, each lane holding [c] columns and
   the [R] rows of a pass, a column group [4 (32 / S) c] columns. A plain weight
   of columns contiguous takes the k x n kernels: eight SIMD-groups, a lane
   holding 16 bytes of adjacent columns, a SIMD-group [rho'] spans of [32 /
   rho'] lanes, [rho' = min rho S], and a column block [S / rho'] SIMD-groups,
   whose sums the halving pairs through threadgroup memory. *)

module Vector = struct
  type key =
    | Dense_f32
    | Dense_f16
    | Dense_bf16
    | E2m1_by_exponent
    | Int8_by_value

  (* The M1 Max's figures of the vector kernels against the tiles at each m set
     the bounds, which read nothing of a weight's layout, so that a call's bits
     depend on its extents alone. MXFP4's 5760 x 2880 took 36 us a row by the
     vector kernels against the loaders' 400 at any m to 16, so e2m1's bound is
     11; Q8_0's 4096^2 took 114 us at 2 rows and 127 at 3 against 145, so int8's
     is 3. A Rows call's rows each read their own group, so its quantized keys
     are faster at every m to 16; f16 and bf16 to 2, f32, of one column a lane,
     at 1. *)
  let bound key ~rows =
    match key with
    | (E2m1_by_exponent | Int8_by_value) when rows -> 16
    | E2m1_by_exponent -> 11
    | Int8_by_value -> 3
    | Dense_f16 | Dense_bf16 -> 2
    | Dense_f32 -> 1

  (* The most terms of a call the vector kernels take, so that the plan of every
     call they take finds launches, and a call's bits read its extents alone,
     never the cost model. Each is the most terms whose one column group, at
     every row count to the key's bound and either layout, plans within 0.8 of
     Plan.budget at Plan's rates, rounded down to a power of two. The figures
     come from today's kernels and rates: a refit under which a call at these
     bounds finds no launches lowers them in the same change. Under each, a
     lane's term indices, to k plus a span, stay below 2^32. *)

  (* float32 k x n at 1 row: 9.2M terms. *)
  let f32_terms = 1 lsl 23

  (* float16 and bfloat16 k x n at 2 rows: 5.5M. *)
  let half_terms = 1 lsl 22

  (* e2m1 k x n at 11 rows: 0.99M; int8 k x n at 3: 1.0M. *)
  let scaled_terms = 1 lsl 19

  (* A Rows call's: int8 k x n at 16 rows, 0.39M; e2m1 0.43M. *)
  let rows_terms = 1 lsl 18

  let stated_terms key ~rows =
    match key with
    | Dense_f32 -> f32_terms
    | Dense_f16 | Dense_bf16 -> half_terms
    | E2m1_by_exponent | Int8_by_value ->
        if rows then rows_terms else scaled_terms

  let terms_bound = Atomic.make max_int

  let terms key ~rows =
    Int.min (Atomic.get terms_bound) (stated_terms key ~rows)

  let bound_terms k =
    if k < 1 then invalid_arg "Contract.Vector.bound_terms";
    Atomic.set terms_bound k;
    Plan.renew ()

  let columns = 256
  let threads = Kernels.mv_threads
  let kn_threads = Kernels.mv_kn_threads
  let groups_lanes = threads / 32
  let most_lanes = 32
  let rho = Kernels.mv_rho

  (* A lane's columns: a span's 128 bytes of f32, 64 of f16, bf16 and int8 by
     two, and the 16 of e2m1 by four, which keep a lane's live values within
     what the probes measured fast, about 72 counted 32-bit values: payload
     words, decoded terms, sums, row terms and addresses. *)
  let lane_cols = function
    | Dense_f32 -> 1
    | Dense_f16 | Dense_bf16 | Int8_by_value -> 2
    | E2m1_by_exponent -> 4

  (* A k x n lane's columns: 16 bytes of a plain key's term row; 8 of a scaled
     key's, whose decoded values and scales a column the lane holds too. *)
  let kn_cols = function
    | Dense_f32 -> 4
    | Dense_f16 | Dense_bf16 | E2m1_by_exponent | Int8_by_value -> 8

  (* A k x n pass's rows: 1 for a plain key, at one row and in a Rows call; 4
     for a scaled one past one row, which reads its weight a quarter as often
     with few values live: MXFP4's 16-row call ran 515 us at 4 rows a pass and
     858 at 1. *)
  let kn_pass key ~rows m =
    match key with
    | (E2m1_by_exponent | Int8_by_value) when (not rows) && m > 1 -> 4
    | _ -> 1

  (* A pass's rows: 1 at one row and in a Rows call, whose rows each read their
     own group; 2 for f16 and bf16 past one row. The scaled keys run 1: their
     twins' registers grow with the rows, and MXFP4's 16-row call ran 582 us at
     1 row a pass, 652 at 2 and 902 at 4, Q8_0's 2-row call 114 at 1 and 169 at
     4. *)
  let pass_rows key ~rows m =
    if rows || m <= 1 then 1
    else
      match key with
      | Dense_f16 | Dense_bf16 -> 2
      | Dense_f32 | E2m1_by_exponent | Int8_by_value -> 1

  let lanes_of spans =
    let rec up s = if 2 * s <= spans && s < most_lanes then up (2 * s) else s in
    up 1

  type t = {
    kn : bool;
    rows : int;
    cols : int;
    lanes : int;
    spans : int;
    pass : int;
    width : int;
    groups : int;
    resident : int;
    launches : launch array;
    seconds : float;
  }

  (* A key's weight bytes an element with its scale's share, which errs high for
     int8 by float16, and its decoding's operations: e2m1's code shifted out and
     masked, its sign and magnitude masked and shifted into a half and joined,
     the half widened and scaled (9), and 2 for the rest of a lane's work an
     element, measured: its span's scale test, addresses, and x's loads and
     widening shared by its 4 columns, past which MXFP4's decode row ran 3-4% at
     9; int8's quant shifted out, sign-extended, widened and scaled (4); a
     half's or a bfloat's widening. *)
  let key_bytes = function
    | Dense_f32 -> 4.
    | Dense_f16 | Dense_bf16 -> 2.
    | Int8_by_value -> 1. +. (4. /. 32.)
    | E2m1_by_exponent -> 0.5 +. (1. /. 32.)

  let key_ops = function
    | Dense_f32 -> 0
    | Dense_f16 | Dense_bf16 -> 1
    | Int8_by_value -> 4
    | E2m1_by_exponent -> 11

  (* A scaled k x n lane's operations an element, fitted so that MXFP4's 16-row
     call over codes outside [2, 240], contract_exponent an element, runs within
     0.8 of its estimate on the M1 Max: 1023 us against 1330 (522 us over codes
     within it). A term at a time, a lane's terms wait on their loads, which the
     n x k spans load together. *)
  let kn_ops = 80
  let passes v = ceil_div v.rows v.pass

  (* A k x n threadgroup's column blocks hold [32 / rho'] lanes of a SIMD-group
     and [S / rho'] SIMD-groups each. *)
  let group_cols v =
    if not v.kn then groups_lanes * (most_lanes / v.lanes) * v.width
    else
      let r = Int.min rho v.lanes in
      kn_threads / 32 / (v.lanes / r) * (most_lanes / r) * v.width

  let group_threads v = if v.kn then kn_threads else threads

  (* A k x n thread halves its sums through threadgroup memory 8 at a time. *)
  let shared v =
    if v.kn then kn_threads * Int.min (v.width * v.pass) 8 * 4 else 0

  (* The weights of [tiles] column groups of [v] in a call of [e.k] terms: each
     pass reads its columns' weight at DRAM's rate, which errs high where the
     caches keep a column group's weight for its other passes, which the M1's
     are not shown to; a pass's rows' terms and outputs' float32 sums. A lane
     decodes each of its terms once a pass and adds its product to each row's
     sum; its chain is a load and a span's fused multiply-adds for each of its
     spans, then the halving. *)
  let weights key (e : extents) v ~tiles =
    let p = passes v in
    let cols = tiles * group_cols v in
    let elements = cols * e.k in
    let weight =
      Float.to_int (Float.ceil (float_of_int elements *. key_bytes key))
    in
    let reads = (p * weight) + (tiles * v.rows * e.k * 4) in
    let sums = cols * v.rows * 4 in
    let steps = ceil_div v.spans v.lanes in
    let rec log2 s = if s <= 1 then 0 else 1 + log2 (s / 2) in
    {
      Plan.bytes = reads + sums;
      ops =
        p * elements
        * ((match key with
             | (E2m1_by_exponent | Int8_by_value) when v.kn -> kn_ops
             | _ -> key_ops key)
          + v.pass);
      chain = (steps * (span + Plan.read_links)) + log2 v.lanes;
    }

  let seconds key e v ~tiles =
    Plan.seconds ~threads:(group_threads v)
      ~groups:(passes v, tiles, 1)
      ~resident:v.resident (weights key e v ~tiles)

  let estimate key e v ~bandwidth ~cores ~op_rate (l : launch) =
    Plan.estimate ~bandwidth ~cores ~op_rate ~threads:(group_threads v)
      ~groups:(passes v, l.tiles, 1)
      ~resident:v.resident
      (weights key e v ~tiles:l.tiles)

  let plan (e : extents) ~key ~rows ~kn =
    let spans = ceil_div e.k span in
    let lanes = lanes_of spans in
    let v =
      {
        kn;
        rows = e.m;
        cols = e.n;
        lanes;
        spans;
        pass = (if kn then kn_pass key ~rows e.m else pass_rows key ~rows e.m);
        width = (if kn then kn_cols key else lane_cols key);
        groups = 0;
        resident = 0;
        launches = [||];
        seconds = 0.;
      }
    in
    let v =
      {
        v with
        groups = ceil_div e.n (group_cols v);
        resident = Plan.holds ~threads:(group_threads v) (shared v);
      }
    in
    let total = e.batch * v.groups in
    let allowed = Plan.allowed () in
    let fits tiles = seconds key e v ~tiles <= allowed in
    if total = 0 || e.m = 0 || e.k = 0 then Some v
    else
      match largest fits total with
      | 0 -> None
      | size ->
          let launch i =
            let first = i * size in
            let tiles = Int.min size (total - first) in
            {
              first;
              tiles;
              chunk = 0;
              length = 0;
              seconds = seconds key e v ~tiles;
            }
          in
          let launches = Array.init (ceil_div total size) launch in
          let seconds =
            Array.fold_left (fun s (l : launch) -> s +. l.seconds) 0. launches
          in
          Some { v with launches; seconds }
end

(* Calls

   A call's launches, as one plan:

   packs (a, b, init) --> tile windows, each over its chunks in increasing k -->
   the finish, where the sums do not go to dst

   The tiles read a ([b] where exchanged) as rows of terms, each row's terms
   contiguous, and b ([a] where exchanged) as rows of columns, each row's
   columns contiguous (k x n), or as columns of terms, each column's terms
   contiguous (n x k). An operand whose batch, row and term axes do not each
   merge into one run of strides, or whose contiguous axis is neither, is copied
   in C order into the workspace, as is an init whose axes do not merge, or that
   shares memory with dst through another layout. The tiles store their sums
   into dst, rounded where it is of the operands' 16-bit dtype, where one range
   of whole k gives them, dst is float32 or of the operands' dtype and shares no
   memory with a or b, and there is no init; else into float32 planes in the
   workspace, one a range, which the finish adds in order, then init, and rounds
   into dst. A copy keeps bits, so no layout changes a bit.

   A Rows call takes the grouped tiles after its prepass (Rows segments, below),
   b's batch elements and groups as one batch, every batch element's rows in one
   call order; the finish takes the rows of no group as ones of no products.

   A complex64 call is the float32 call of its real form: a's elements as pairs
   of terms, its terms' run doubled; b's elements as real 2 x 2 blocks, 2k x 2n
   a batch element, which the complex tiles stage from b where it lies, k x n or
   n x k, its strides doubled; and dst's or the planes' columns as pairs of real
   columns. It never exchanges a and b: exchanged, the imaginary part would sum
   ai br before ar bi. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

let stage_of (D.Any d) =
  match d with
  | D.Float32 -> Some F32
  | D.Float16 -> Some F16
  | D.Bfloat16 -> Some Bf16
  | _ -> None

type fit = Whole | Wide | Narrow

let kernel stage ~nk tile fit =
  match (stage, nk, tile, fit) with
  | F32, false, T64, Whole -> K.Contract_f32_64_kn_a
  | F32, false, T64, Wide -> K.Contract_f32_64_kn_e
  | F32, false, T64, Narrow -> K.Contract_f32_64_kn_n
  | F32, false, T32, Whole -> K.Contract_f32_32_kn_a
  | F32, false, T32, Wide -> K.Contract_f32_32_kn_e
  | F32, false, T32, Narrow -> K.Contract_f32_32_kn_n
  | F32, false, T16, Whole -> K.Contract_f32_16_kn_a
  | F32, false, T16, Wide -> K.Contract_f32_16_kn_e
  | F32, false, T16, Narrow -> K.Contract_f32_16_kn_n
  | F32, true, T64, _ -> K.Contract_f32_64_nk
  | F32, true, T32, _ -> K.Contract_f32_32_nk
  | F32, true, T16, _ -> K.Contract_f32_16_nk
  | F16, false, T64, Whole -> K.Contract_f16_64_kn_a
  | F16, false, T64, Wide -> K.Contract_f16_64_kn_e
  | F16, false, T64, Narrow -> K.Contract_f16_64_kn_n
  | F16, false, T32, Whole -> K.Contract_f16_32_kn_a
  | F16, false, T32, Wide -> K.Contract_f16_32_kn_e
  | F16, false, T32, Narrow -> K.Contract_f16_32_kn_n
  | F16, false, T16, Whole -> K.Contract_f16_16_kn_a
  | F16, false, T16, Wide -> K.Contract_f16_16_kn_e
  | F16, false, T16, Narrow -> K.Contract_f16_16_kn_n
  | F16, true, T64, _ -> K.Contract_f16_64_nk
  | F16, true, T32, _ -> K.Contract_f16_32_nk
  | F16, true, T16, _ -> K.Contract_f16_16_nk
  | Bf16, false, T64, Whole -> K.Contract_bf16_64_kn_a
  | Bf16, false, T64, Wide -> K.Contract_bf16_64_kn_e
  | Bf16, false, T64, Narrow -> K.Contract_bf16_64_kn_n
  | Bf16, false, T32, Whole -> K.Contract_bf16_32_kn_a
  | Bf16, false, T32, Wide -> K.Contract_bf16_32_kn_e
  | Bf16, false, T32, Narrow -> K.Contract_bf16_32_kn_n
  | Bf16, false, T16, Whole -> K.Contract_bf16_16_kn_a
  | Bf16, false, T16, Wide -> K.Contract_bf16_16_kn_e
  | Bf16, false, T16, Narrow -> K.Contract_bf16_16_kn_n
  | Bf16, true, T64, _ -> K.Contract_bf16_64_nk
  | Bf16, true, T32, _ -> K.Contract_bf16_32_nk
  | Bf16, true, T16, _ -> K.Contract_bf16_16_nk
  | I8, false, T64, (Whole | Wide) -> K.Contract_i8_64_kn
  | I8, false, T64, Narrow -> K.Contract_i8_64_kn_n
  | I8, false, T32, (Whole | Wide) -> K.Contract_i8_32_kn
  | I8, false, T32, Narrow -> K.Contract_i8_32_kn_n
  | I8, false, T16, (Whole | Wide) -> K.Contract_i8_16_kn
  | I8, false, T16, Narrow -> K.Contract_i8_16_kn_n
  | I8, true, T64, _ -> K.Contract_i8_64_nk
  | I8, true, T32, _ -> K.Contract_i8_32_nk
  | I8, true, T16, _ -> K.Contract_i8_16_nk
  | F32, false, G32, (Whole | Wide) -> K.Contract_rows_f32_32_kn
  | F32, false, G32, Narrow -> K.Contract_rows_f32_32_kn_n
  | F32, false, G16, (Whole | Wide) -> K.Contract_rows_f32_16_kn
  | F32, false, G16, Narrow -> K.Contract_rows_f32_16_kn_n
  | F32, true, G32, _ -> K.Contract_rows_f32_32_nk
  | F32, true, G16, _ -> K.Contract_rows_f32_16_nk
  | F16, false, G32, (Whole | Wide) -> K.Contract_rows_f16_32_kn
  | F16, false, G32, Narrow -> K.Contract_rows_f16_32_kn_n
  | F16, false, G16, (Whole | Wide) -> K.Contract_rows_f16_16_kn
  | F16, false, G16, Narrow -> K.Contract_rows_f16_16_kn_n
  | F16, true, G32, _ -> K.Contract_rows_f16_32_nk
  | F16, true, G16, _ -> K.Contract_rows_f16_16_nk
  | Bf16, false, G32, (Whole | Wide) -> K.Contract_rows_bf16_32_kn
  | Bf16, false, G32, Narrow -> K.Contract_rows_bf16_32_kn_n
  | Bf16, false, G16, (Whole | Wide) -> K.Contract_rows_bf16_16_kn
  | Bf16, false, G16, Narrow -> K.Contract_rows_bf16_16_kn_n
  | Bf16, true, G32, _ -> K.Contract_rows_bf16_32_nk
  | Bf16, true, G16, _ -> K.Contract_rows_bf16_16_nk
  | I8, _, (G32 | G16), _ -> invalid_arg "Contract.kernel: grouped tiles at I8"

let complex_kernel ~nk tile fit =
  match (nk, tile, fit) with
  | false, T64, Whole -> K.Contract_c64_64_kn_a
  | false, T64, Wide -> K.Contract_c64_64_kn_e
  | false, T64, Narrow -> K.Contract_c64_64_kn_n
  | false, T32, Whole -> K.Contract_c64_32_kn_a
  | false, T32, Wide -> K.Contract_c64_32_kn_e
  | false, T32, Narrow -> K.Contract_c64_32_kn_n
  | false, T16, Whole -> K.Contract_c64_16_kn_a
  | false, T16, Wide -> K.Contract_c64_16_kn_e
  | false, T16, Narrow -> K.Contract_c64_16_kn_n
  | true, T64, _ -> K.Contract_c64_64_nk
  | true, T32, _ -> K.Contract_c64_32_nk
  | true, T16, _ -> K.Contract_c64_16_nk
  | false, G32, (Whole | Wide) -> K.Contract_rows_c64_32_kn
  | false, G32, Narrow -> K.Contract_rows_c64_32_kn_n
  | false, G16, (Whole | Wide) -> K.Contract_rows_c64_16_kn
  | false, G16, Narrow -> K.Contract_rows_c64_16_kn_n
  | true, G32, _ -> K.Contract_rows_c64_32_nk
  | true, G16, _ -> K.Contract_rows_c64_16_nk

let fit (e : extents) p =
  let bn = cols p.tile in
  if e.k mod depth e.stage p.tile = 0 && p.cols_out mod bn = 0 then Whole
  else if p.cols_out >= bn then Wide
  else Narrow

(* Runs *)

(* The axes [axes] of [l] as one run, (extent, stride), the stride 0 along at
   most one element; [None] where they make several. An axis merges into the one
   before it where that one's stride is its stride times its extent. *)
let run l axes =
  let extent = ref 1 and stride = ref 0 and runs = ref 0 in
  Array.iter
    (fun a ->
      let e = L.dim l a in
      if e = 0 then extent := 0
      else if e > 1 then begin
        let s = L.stride l a in
        if !runs = 0 || !stride <> s * e then incr runs;
        stride := s;
        extent := !extent * e
      end)
    axes;
  if !extent = 0 then Some (0, 0)
  else if !runs <= 1 then Some (!extent, if !extent > 1 then !stride else 0)
  else None

(* An operand as the tiles read it: its slot, its first element's byte, and its
   strides in elements along the batch, its rows (a's) or its staged rows
   (b's). *)
type view = { slot : int; at : int; batch : int; major : int }

(* The index of the first of [ops] from [k] on [buffer]. *)
let rec first_on (ops : A.any array) buffer k =
  let (A.Any y) = ops.(k) in
  if A.buffer y == buffer then k else first_on ops buffer (k + 1)

(* The slot of [x]'s buffer in [b]: that of the first of [ops], the call's
   operands as the door numbers them ({!Plan.slot}), on it. *)
let slot b ops (A.Any x) = Plan.slot b (first_on ops (A.buffer x) 0)

(* Whether {!run}'s runs [rs] of the axes groups batch, outer and inner merge:
   each into one run, the inner one stepping by one. *)
let merging rs =
  match rs with
  | [| Some _; Some _; Some (ni, si) |] -> si = 1 || ni <= 1
  | _ -> false

(* [x]'s axes [groups] as a view, where they are {!merging}. *)
let merged b ops x groups =
  let l = A.layout x in
  match Array.map (run l) groups with
  | [| Some (_, sb); Some (_, so); _ |] as rs when merging rs ->
      let width = D.bits (A.dtype x) / 8 in
      Some
        {
          slot = slot b ops (A.Any x);
          at = L.offset l * width;
          batch = sb;
          major = so;
        }
  | _ -> None

(* [x]'s axes [groups] copied in C order of their concatenation into a workspace
   piece, as a view. *)
let packed b ops x groups =
  let l = A.layout x in
  let axes = Array.concat (Array.to_list groups) in
  let shape = Iarray.of_array (Array.map (L.dim l) axes) in
  let strides = Iarray.of_array (Array.map (L.stride l) axes) in
  let dt = D.Any (A.dtype x) in
  let src =
    Moves.site_at ~slot:(slot b ops (A.Any x)) ~base:0
      ~residue:(Operands.residue (A.buffer x))
      dt
      (L.v ~offset:(L.offset l) ~strides shape)
  in
  let extent g = Array.fold_left (fun n a -> n * L.dim l a) 1 g in
  let n = Iarray.fold_left ( * ) 1 shape in
  let at = Plan.take b (n * (D.bits (A.dtype x) / 8)) in
  let ws = Plan.workspace b in
  let dst =
    Moves.site_at ~slot:ws ~base:at ~residue:0 dt (L.contiguous shape)
  in
  ignore (Moves.launch b ~n ~dst src : float);
  let inner = extent groups.(2) in
  { slot = ws; at; batch = extent groups.(1) * inner; major = inner }

(* Finish *)

(* A finish thread's work in float32 operations, an estimate: its index and a
   32-bit division, the init's decode and dst's encode, and a read and an
   addition a range. *)
let finish_ops = 256
let range_ops = 16
let finish_threads = K.threads K.Contract_finish

let finish_cost ~cols ~ranges ~bytes rows =
  let gx = ceil_div cols finish_threads in
  {
    Plan.bytes = (rows * cols * bytes) + (2 * Plan.line * rows);
    ops = gx * rows * finish_threads * (finish_ops + (range_ops * ranges));
    chain = (2 * Plan.read_links) + ranges;
  }

let finish_seconds ~cols ~ranges ~bytes rows =
  Plan.seconds ~threads:finish_threads
    ~groups:(ceil_div cols finish_threads, rows, 1)
    (finish_cost ~cols ~ranges ~bytes rows)

(* Where the finish reads init or writes dst: a slot, the byte of its address,
   its dtype's code, its strides along the batch and the tiles' rows and
   columns, and for the integer finish, the position of its element at index 0
   from the address and the bit of the address's byte at which position 0
   starts. *)
type site = {
  fslot : int;
  fat : int;
  code : int;
  s : int * int * int;
  pos : int;
  phase : int;
}

(* Whether [d] is an integer dtype. *)
let integer (D.Any d) =
  match D.kind d with
  | D.Signed | D.Unsigned -> true
  | D.Float | D.Complex | D.Boolean -> false

(* A Contracting call's groups as the SIMT kernel and the finish read them: the
   prepass's first entries at [(slot, at)], the groups, the operands' batch
   elements and the cap of a group's ranges. *)
type listing = { koff : int * int; kcount : int; kbatch : int; kcap : int }

(* The finish into [acc]: an integer one reads 64-bit sums, casts init into
   [acc], and wraps there. A Rows call's finish reads [buckets], the prepass's
   bucket of each row at [(slot, at)], and takes a row of the bucket [none] as
   one of no products. A Contracting call's adds each output's group's ranges
   alone. *)
let finish b ~allowed ~acc ?buckets ?listing ?(tri = 0) ~planes ~ranges ~plane
    ~rows ~total ~cols ~init ~dst () =
  let width (s : site option) =
    match s with
    | None -> 0
    | Some s -> (
        match D.of_code s.code with
        | Some (D.Any D.Complex64) -> 4
        | Some (D.Any d) -> Int.max 1 (D.bits d / 8)
        | None -> invalid_arg "Contract.finish: a dtype code")
  in
  let kernel, sums =
    if integer acc then (K.Contract_finish_int, 8) else (K.Contract_finish, 4)
  in
  let bytes = (sums * ranges) + width init + width (Some dst) in
  let fits r = finish_seconds ~cols ~ranges ~bytes r <= allowed in
  let most = largest fits total in
  if most = 0 then false
  else begin
    let first = ref 0 in
    while !first < total do
      let count = Int.min most (total - !first) in
      let w = Plan.words kernel in
      let refs = ref [] in
      let address word slot at write =
        Plan.set w word at;
        refs := { Plan.word; slot; write } :: !refs
      in
      (match planes with
      | Some (slot, at) when ranges > 0 -> address K.finish_planes slot at false
      | Some _ | None -> ());
      Plan.set w K.finish_rows rows;
      Plan.set w K.finish_cols cols;
      Plan.set w K.finish_plane plane;
      Plan.set w K.finish_ranges ranges;
      Plan.set w K.finish_first !first;
      Plan.set w K.finish_tri tri;
      (match init with
      | None -> Plan.set w K.finish_init_dtype (-1)
      | Some i ->
          let z, r, c = i.s in
          address K.finish_init i.fslot i.fat false;
          Plan.set w K.finish_init_dtype i.code;
          Plan.set w K.finish_init_z z;
          Plan.set w K.finish_init_r r;
          Plan.set w K.finish_init_c c;
          Plan.set w K.finish_init_at i.pos;
          Plan.set w K.finish_init_phase i.phase);
      let z, r, c = dst.s in
      address K.finish_dst dst.fslot dst.fat true;
      Plan.set w K.finish_dst_dtype dst.code;
      Plan.set w K.finish_dst_z z;
      Plan.set w K.finish_dst_r r;
      Plan.set w K.finish_dst_c c;
      Plan.set w K.finish_dst_at dst.pos;
      Plan.set w K.finish_dst_phase dst.phase;
      Plan.set w K.finish_acc (match acc with D.Any d -> D.code d);
      (match buckets with
      | None -> Plan.set w K.finish_none (-1)
      | Some (slot, at, none) ->
          address K.finish_buckets slot at false;
          Plan.set w K.finish_none none);
      (match listing with
      | None -> ()
      | Some g ->
          let slot, at = g.koff in
          address K.finish_koff slot at false;
          Plan.set w K.finish_kgroups g.kcount;
          Plan.set w K.finish_batch g.kbatch;
          Plan.set w K.finish_cap g.kcap);
      Plan.launch b kernel
        ~groups:(ceil_div cols finish_threads, count, 1)
        ~cost:(finish_cost ~cols ~ranges ~bytes count)
        w
        (Array.of_list (List.rev !refs));
      first := !first + count
    done;
    true
  end

(* The call *)

let pairs_fst p = Array.map fst (Iarray.to_array p)
let pairs_snd p = Array.map snd (Iarray.to_array p)

(* The axes of [0, rank) not in [paired], in order. *)
let free rank paired =
  Array.of_list
    (List.filter (fun a -> not (Array.mem a paired)) (List.init rank Fun.id))

let extent l axes = Array.fold_left (fun n a -> n * L.dim l a) 1 axes
let buffer (A.Any x) = A.buffer x
let layout (A.Any x) = A.layout x
let code (A.Any x) = D.code (A.dtype x)
let overlaps x y = Rig.Buffer.overlaps (buffer x) (buffer y)
let staged (A.Any x) = stage_of (D.Any (A.dtype x))
let fits_32 n = n < 1 lsl 32

(* m = floor (2^32 (2^l - d) / d) + 1 for l = ceil (log2 d), in unsigned 64
   bits: 2^32 (2^l - d) reaches 2^64 for d near 2^32, past OCaml's int. m is
   below 2^32 since 2^l < 2 d. *)
let divisor d =
  if d < 1 || d > (1 lsl 32) - 1 then invalid_arg "Contract.divisor";
  let rec log l = if 1 lsl l >= d then l else log (l + 1) in
  let l = log 0 in
  let num = Int64.shift_left (Int64.of_int ((1 lsl l) - d)) 32 in
  let m = Int64.(to_int (add (unsigned_div num (of_int d)) 1L)) in
  m lor (Int.min l 1 lsl 32) lor (Int.max (l - 1) 0 lsl 40)

(* Whether [x]'s axes [groups] merge into a view. *)
let merges x groups = merging (Array.map (run (layout x)) groups)

let view_of b ops (A.Any x) groups =
  match merged b ops x groups with Some v -> v | None -> packed b ops x groups

(* An init's or dst's batch, row and column strides, where they merge. *)
let strides_of x groups =
  match Array.map (run (layout x)) groups with
  | [| Some (_, z); Some (_, r); Some (_, c) |] -> Some (z, r, c)
  | _ -> None

(* [s] along the tiles' batch, rows and columns. *)
let oriented ~exchanged (z, r, c) = if exchanged then (z, c, r) else (z, r, c)

(* A call that a step finds no launches for, which {!call} declines. *)
exception Decline

let decline () = raise_notrace Decline

(* A call that runs in pieces ({!call}): of a float32 call whose tiles would
   read an unscaled operand not of their stage's dtype, the dtypes a and b are
   cast into first; [temp] where dst does not merge its groups. *)
type recast = { a : D.any option; b : D.any option; temp : bool }

exception Recast_call of recast

(* The pieces of a call of no casts, [apart] where its dst does not merge its
   groups. *)
let pieces ~apart = { a = None; b = None; temp = apart }
let largest_outputs = (1 lsl 31) - 1
let most_outputs = Atomic.make largest_outputs

(* Whether a call of [outputs] runs in pieces of fewer. *)
let large outputs = outputs > Atomic.get most_outputs

let stage_dtype = function
  | F32 -> D.Any D.Float32
  | F16 -> D.Any D.Float16
  | Bf16 -> D.Any D.Bfloat16
  | I8 -> invalid_arg "Contract.stage_dtype: I8"

(* The dtype [x] is cast into at [stage], unscaled ([s]) and not of the stage's
   dtype. *)
let cast_into stage s (A.Any x) =
  let (D.Any d as want) = stage_dtype stage in
  if Option.is_some s || D.code (A.dtype x) = D.code d then None else Some want

(* A float view's site for the finish: [x]'s slot, first element's byte and
   dtype, at the strides [s]. *)
let site_of b ops (A.Any x) s =
  let width = D.bits (A.dtype x) / 8 in
  {
    fslot = slot b ops (A.Any x);
    fat = L.offset (A.layout x) * width;
    code = D.code (A.dtype x);
    s;
    pos = 0;
    phase = 0;
  }

(* [x] placed ({!Moves.place}), in its slot. *)
let place b ops x =
  let at, pos, phase = Moves.place x in
  (slot b ops x, at, pos, phase)

(* An integer call's site for the finish: [x] placed, at the strides [s]. *)
let placed_site b ops x s =
  let fat, pos, phase = Moves.place x in
  { fslot = slot b ops x; fat; code = code x; s; pos; phase }

(* [x]'s axes [groups], of a dtype narrower than a byte, widened into a
   workspace piece of bytes of {!Moves.twin}'s dtype in C order of their
   concatenation, as a view. *)
let widened b ops (A.Any x as ax) groups =
  let l = A.layout x in
  let axes = Array.concat (Array.to_list groups) in
  let shape = Iarray.of_array (Array.map (L.dim l) axes) in
  let strides = Iarray.of_array (Array.map (L.stride l) axes) in
  let src = L.v ~offset:(L.offset l) ~strides shape in
  let at = Plan.take b (L.numel src) in
  let ws = Plan.workspace b in
  let dst =
    Moves.site_at ~slot:ws ~base:at ~residue:0 (D.Any D.Uint8)
      (L.contiguous shape)
  in
  Moves.widen b ~dst ~slot:(slot b ops ax) ax src;
  let inner = extent l groups.(2) in
  { slot = ws; at; batch = extent l groups.(1) * inner; major = inner }

(* Scales *)

(* A scale thread's work besides its axes, an estimate: a widening's, its
   element's and its scale's reads and casts by a switch on the dtypes. *)
let scale_ops = 1024

(* A scale thread's work for each axis but the outermost: a 64-bit [%] and [/],
   which the GPU emulates, as a copy's. *)
let scale_axis_ops = 4096
let scale_threads = K.threads K.Contract_scale

(* The cost of scaling [m] elements of a layout of [rank] axes, in [groups]
   groups: the float32 written, the element's and its scale's bytes or a line
   where neighbours share none, and each thread's work. *)
let scale_cost ~m ~groups ~rank ~read =
  {
    Plan.bytes = (m * (4 + read)) + (4 * Plan.line);
    ops = groups * scale_threads * (scale_ops + ((rank - 1) * scale_axis_ops));
    chain = 0;
  }

(* [l]'s and [sl]'s axes, of one shape, as contract_scale reads them: each
   axis's extent, then its strides in each, as little-endian 64-bit words into
   [s] from word [at]. *)
let scale_axes s ~at l sl =
  for j = 0 to L.rank l - 1 do
    let word i v =
      let w = at + (K.scale_axis_words * j) + i in
      Bytes.set_int64_le s (8 * w) (Int64.of_int v)
    in
    word K.scale_axis_extent (L.dim l j);
    word K.scale_axis_src (L.stride l j);
    word K.scale_axis_scale (L.stride sl j)
  done

(* [x]'s axes [groups], each element's cast into float32 times its scale block's
   scale [s], rounded once, in a workspace piece of float32 in C order of their
   concatenation, as a view. Each axis splits into its scale's blocks and a
   block's elements, along which the scale steps by 0, so one stride a layout
   reaches the scale of every element. *)
let scaled b ops (A.Any x as ax) (s : Nx_kernel.scale) groups =
  let sx, exponent =
    match s with Exponent e -> (A.Any e, true) | Value v -> (A.Any v, false)
  in
  let l = A.layout x and sl = layout sx in
  let axes = Array.concat (Array.to_list groups) in
  let split a =
    let e = L.dim l a and blocks = L.dim sl a in
    let r = if blocks = 0 then 1 else e / blocks in
    [| (blocks, L.stride l a * r, L.stride sl a); (r, L.stride l a, 0) |]
  in
  let parts = Array.concat (Array.to_list (Array.map split axes)) in
  let shape = Iarray.of_array (Array.map (fun (e, _, _) -> e) parts) in
  let on offset strides =
    L.v ~offset ~strides:(Iarray.of_array strides) shape
  in
  let src = on (L.offset l) (Array.map (fun (_, st, _) -> st) parts) in
  let sc = on (L.offset sl) (Array.map (fun (_, _, st) -> st) parts) in
  let ls = L.coalesce [| L.contiguous shape; src; sc |] in
  let src = ls.(1) and sc = ls.(2) in
  let rank = L.rank src in
  let n = L.numel src in
  let at = Plan.take b (4 * n) in
  let ws = Plan.workspace b in
  (* Each end's bit of its index 0 from its address, as a widening finds its
     source's. *)
  let bits (A.Any z) = D.bits (A.dtype z) in
  let base y =
    let slot, byte, pos, phase = place b ops y in
    (slot, byte, phase + (bits y * pos))
  in
  let xslot, xat, xbit = base ax and sslot, sat, sbit = base sx in
  let tail =
    if rank <= K.scale_holds then [||]
    else
      let t = Bytes.create (8 * K.scale_axis_words * rank) in
      scale_axes t ~at:0 src sc;
      let t = Bytes.unsafe_to_string t in
      let c =
        Plan.constant b ~key:("scale " ^ t) ~length:(String.length t) (fun () ->
            Moves.bigbytes t)
      in
      [| { Plan.word = K.scale_axes; slot = c; write = false } |]
  in
  let codes =
    code ax + (256 * code sx) + if exponent then K.scale_exponent else 0
  in
  let read =
    let width y = Int.max 1 (bits y / 8) in
    let inner l = abs (L.stride l (rank - 1)) in
    let one y l = if inner l <= 1 then width y else Plan.line in
    one ax src + one sx sc
  in
  let fits m =
    let groups = ceil_div m scale_threads in
    Plan.seconds ~threads:scale_threads ~groups:(groups, 1, 1)
      (scale_cost ~m ~groups ~rank ~read)
    <= Plan.allowed ()
  in
  let most = largest fits n in
  if n > 0 && most = 0 then decline ();
  let first = ref 0 in
  while !first < n do
    let m = Int.min most (n - !first) in
    let w = Plan.words K.Contract_scale in
    Plan.set w K.scale_dst at;
    Plan.set w K.scale_src xat;
    Plan.set w K.scale_scale sat;
    Plan.set w K.scale_first !first;
    Plan.set w K.scale_count (!first + m);
    Plan.set w K.scale_rank rank;
    Plan.set w K.scale_codes codes;
    Plan.set w K.scale_src_bit xbit;
    Plan.set w K.scale_scale_bit sbit;
    if rank <= K.scale_holds then scale_axes w ~at:K.scale_held src sc;
    let groups = ceil_div m scale_threads in
    Plan.launch b K.Contract_scale ~groups:(groups, 1, 1)
      ~cost:(scale_cost ~m ~groups ~rank ~read)
      w
      (Array.append
         [|
           { Plan.word = K.scale_dst; slot = ws; write = true };
           { Plan.word = K.scale_src; slot = xslot; write = false };
           { Plan.word = K.scale_scale; slot = sslot; write = false };
         |]
         tail);
    first := !first + m
  done;
  let inner = extent l groups.(2) in
  { slot = ws; at; batch = extent l groups.(1) * inner; major = inner }

let twin_code (A.Any x) =
  let (D.Any d) = Moves.twin (D.Any (A.dtype x)) in
  D.code d

(* An init the finish reads in place, where its axes merge and it shares no
   memory with dst but through dst's own layout; else its copy in C order in the
   workspace, or of a dtype narrower than a byte, its widening. The integer
   finish reads it [placed]. *)
let init_site b ops ~placed ~exchanged ~dst groups (A.Any x as i) =
  let inplace = (not (overlaps i dst)) || L.equal (A.layout x) (layout dst) in
  match strides_of i groups with
  | Some s when inplace ->
      let s = oriented ~exchanged s in
      if placed then placed_site b ops i s else site_of b ops i s
  | Some _ | None ->
      let narrow = D.bits (A.dtype x) < 8 in
      let v =
        if narrow then widened b ops i groups else packed b ops x groups
      in
      {
        fslot = v.slot;
        fat = v.at;
        code = twin_code i;
        s = oriented ~exchanged (v.batch, v.major, 1);
        pos = 0;
        phase = 0;
      }

(* A grouped tile launch's words: the prepass's list and table in the workspace,
   the groups of a batch element, b's stride along them and the bucket of no
   group. *)
type grouping = { list : int; table : int; groups : int; sg : int; none : int }

(* A loader's scale of b's strides along b's batch, its groups, its columns and
   its terms, and the elements of a block along the columns and the terms, as
   the loader's words hold them (kernels.h). *)
type steps = { sz : int; sj : int; sn : int; rn : int; sk : int; rk : int }

(* Stores [lo] and [hi] as the low and high 32 bits of word [i]. The call's
   checks keep each half within 32 bits before any launch is written: the tiles'
   rows and columns below 2^32 by [fits_32] of each, the groups g and the bucket
   of no group (batch g) by [fits_32] of batch g + 1, and the scale's steps by
   {!scale_steps}; one past would alias, so it raises. *)
let set_pair w i lo hi =
  let half v = v >= -(1 lsl 31) && v < 1 lsl 32 in
  if not (half lo && half hi) then
    invalid_arg (Printf.sprintf "Contract.set_pair: %d, %d" lo hi);
  Bytes.set_int32_le w (8 * i) (Int32.of_int lo);
  Bytes.set_int32_le w ((8 * i) + 4) (Int32.of_int hi)

let tile_launch b e p ~kernel ~shared ~a ~bv ~y:(slot, at) ~ystr ~narrow ~kinds
    ~dtypes ?(phase = 0) ?(words = false) ?(blocks = (0, 0)) ?scale ?grouping
    ?(routine = 0) ?(flips = 0) ?(band = 0) l =
  let ldy, cy, sy, plane = ystr in
  let w = Plan.words kernel in
  let set = Plan.set w in
  set K.contract_a a.at;
  set K.contract_b bv.at;
  set K.contract_y at;
  set K.contract_lda a.major;
  set K.contract_ldb bv.major;
  set K.contract_sa a.batch;
  set K.contract_sb bv.batch;
  set K.contract_ldy ldy;
  set K.contract_cy cy;
  set K.contract_sy sy;
  set K.contract_plane plane;
  set K.contract_m p.rows_out;
  set K.contract_n p.cols_out;
  set K.contract_k e.k;
  set_pair w K.contract_tiles p.tiles_m p.tiles_n;
  set K.contract_padded p.padded;
  set K.contract_first l.first;
  set K.contract_ranges p.ranges;
  set K.contract_spans p.spans;
  set K.contract_length l.length;
  set K.contract_chunk l.chunk;
  set K.contract_codes
    (dtypes lor kinds lor routine lor (flips lsl K.codes_flips)
   lor (phase lsl K.codes_phase)
    lor (Bool.to_int narrow lsl K.codes_narrow)
    lor (p.swizzle lsl K.codes_swizzle)
    lor (Bool.to_int words lsl K.codes_words)
    lor ((band land 1) lsl K.codes_band_a)
    lor (((band lsr 1) land 1) lsl K.codes_band_b));
  set_pair w K.contract_bkb (fst blocks) (snd blocks);
  if fst blocks > 0 then set K.contract_dbl (divisor (fst blocks));
  let refs =
    [
      { Plan.word = K.contract_a; slot = a.slot; write = false };
      { Plan.word = K.contract_b; slot = bv.slot; write = false };
      { Plan.word = K.contract_y; slot; write = true };
    ]
  in
  let refs =
    match grouping with
    | None -> refs
    | Some gr ->
        set K.contract_list gr.list;
        set K.contract_table gr.table;
        (* The groups g of a batch element, and the bucket of no group, batch
           g. *)
        set_pair w K.contract_groups gr.groups gr.none;
        set K.contract_sg gr.sg;
        let ws = Plan.workspace b in
        { Plan.word = K.contract_list; slot = ws; write = false }
        :: { Plan.word = K.contract_table; slot = ws; write = false }
        :: refs
  in
  let refs =
    match scale with
    | None -> refs
    | Some (sslot, sat, s) ->
        set K.contract_bs sat;
        set_pair w K.contract_bsz s.sz s.sj;
        set_pair w K.contract_bsn s.sn s.rn;
        set_pair w K.contract_bsk s.sk s.rk;
        set K.contract_drn (divisor s.rn);
        set K.contract_drk (divisor s.rk);
        { Plan.word = K.contract_bs; slot = sslot; write = false } :: refs
  in
  Plan.launch b kernel ~groups:(l.tiles, 1, p.ranges) ~shared
    ~resident:p.resident
    ~cost:(weights ~cores:Plan.cores e p l)
    w (Array.of_list refs)

(* The launches a call keeps a submission of: packs, a few windows and the
   finish. *)
let kept_launches = 8

(* [b]'s call, each part of at most {!kept_launches} launches kept. *)
let finished b = Some (Plan.finish ~most:kept_launches b)

(* [v], a view of complex64 elements, as one of their float32 parts. *)
let real (v : view) = { v with batch = 2 * v.batch; major = 2 * v.major }

(* A loader's [x] in place: at its index 0's byte [at], its batch's and rows'
   strides [sb] and [so]. *)
let placed_view b ops x at sb so =
  { slot = slot b ops x; at; batch = sb; major = so }

(* An integer tile's operand: a byte dtype's view, or a narrower one's
   widening. *)
let byte_view b ops (A.Any x as ax) groups =
  if D.bits (A.dtype x) >= 8 then view_of b ops ax groups
  else widened b ops ax groups

(* A tile's operand [x] by its axes [groups]: where scaled by [s], its float32
   products with its scales; else where the tile reads [bytes], {!byte_view}'s;
   else {!view_of}'s. *)
let tile_view b ops ~bytes s x groups =
  match s with
  | Some s -> scaled b ops x s groups
  | None when bytes -> byte_view b ops x groups
  | None -> view_of b ops x groups

(* How an integer tile reads an operand's bytes. *)
let bytes_kind (A.Any x) =
  match D.kind (A.dtype x) with
  | D.Signed -> K.bytes_signed
  | D.Boolean -> K.bytes_boolean
  | _ -> K.bytes_unsigned

let dtype_bits (A.Any x) = D.bits (A.dtype x)
let integer_array (A.Any x) = integer (D.Any (A.dtype x))

(* Tiles *)

(* A call's axes: [a]'s and [b]'s batch, contracting and free axes, [dst]'s
   batch, row and column groups, and the batch elements, rows, columns and
   terms. A Rows call's b [drop]s its axis of groups from its free axes. *)
type axes = {
  ab : int array;
  bb : int array;
  ak : int array;
  bk : int array;
  af : int array;
  bf : int array;
  dgroups : int array array;
  bt : int;
  m : int;
  n : int;
  k : int;
}

let axes_of ?drop ~batch ~contracting ~dst oa ob =
  let la = layout oa and lb = layout ob and ld = layout dst in
  let ab = pairs_fst batch and bb = pairs_snd batch in
  let ak = pairs_fst contracting and bk = pairs_snd contracting in
  let af = free (L.rank la) (Array.append ab ak) in
  let dropped = Option.to_list drop |> Array.of_list in
  let bf = free (L.rank lb) (Array.concat [ bb; bk; dropped ]) in
  let nb = Array.length ab and nf = Array.length af in
  let dgroups =
    [|
      Array.init nb Fun.id;
      Array.init nf (fun i -> nb + i);
      Array.init (L.rank ld - nb - nf) (fun i -> nb + nf + i);
    |]
  in
  {
    ab;
    bb;
    ak;
    bk;
    af;
    bf;
    dgroups;
    bt = extent la ab;
    m = extent la af;
    n = extent lb bf;
    k = extent la ak;
  }

(* Rows segments

   A Rows call's prepass, contract_buckets, gives each row its bucket: the group
   of its id, of its batch element, or the last bucket, of the rows of no group.
   For the grouped tiles it also sorts the rows into buckets and gives each tile
   row its bucket and rows, at most {!bound} of them. *)

type rows = { ids : (int64, D.int64_elt) A.t; a_axis : int; b_axis : int }

let buckets_threads = K.threads K.Contract_buckets

(* The prepass's threadgroup memory: a chunk's counts and first entries, its
   SIMD-groups' partial scans, the chunks' carry and a total. *)
let buckets_shared = (8 * K.buckets_chunk) + (8 * ((buckets_threads / 32) + 2))

(* A prepass thread's work in float32 operations, an estimate: a row's bucket,
   its index's divisions, its id's read and test; a row's count or entry, an
   atomic of threadgroup memory; a chunk's scan, its barriers and prefix sums;
   and a step of a tile row's search. *)
let bucket_ops = 256
let count_ops = 64
let scan_ops = 512
let search_ops = 32

(* The links of a chunk's scan: its five barriers and two prefix sums of each of
   two levels. *)
let scan_links = 64
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)

(* The cost of a prepass of [rows] rows, ids [ids_r] elements apart, in
   [buckets] buckets, and where [sort], of a table of [bound] tile rows: each
   row's id and bucket; each chunk's two reads of the buckets; each row's entry,
   a line where no neighbour shares it; the first entries; each tile row's
   search, a line a step, and its entry. One group runs it, so its chain is its
   threads' longest. *)
let buckets_cost ~rows ~ids_r ~buckets ~bound ~sort =
  let nt = buckets_threads in
  let per = ceil_div rows nt in
  let id = if abs ids_r <= 1 then 8 else Int.min Plan.line (8 * abs ids_r) in
  let chunks = if sort then ceil_div buckets K.buckets_chunk else 0 in
  let depth = log2 buckets + 1 in
  let steps = if sort then ceil_div bound nt * depth else 0 in
  let listed =
    if sort then
      (rows * Plan.line)
      + (8 * (buckets + 1))
      + (bound * (12 + (depth * Plan.line)))
    else 0
  in
  {
    Plan.bytes = (rows * (id + 4)) + (chunks * rows * 8) + listed;
    ops =
      nt
      * ((per * bucket_ops)
        + (chunks * ((2 * per * count_ops) + scan_ops))
        + (steps * search_ops));
    chain =
      (per * Plan.read_links)
      + (chunks * ((2 * per * Plan.read_links) + scan_links))
      + (steps * Plan.read_links);
  }

(* Where the prepass put each row's bucket, its list and its table, in the
   workspace. *)
type placed_buckets = { of_rows : int; list_at : int; table_at : int }

(* Plans the prepass of [rows] rows of [m] a batch element, their ids at [(slot,
   at)] with strides [iz] along the batch and [ir] along the segmented axis, row
   r's id at index [(r / inner) mod extent] of it, into the buckets of [groups]
   groups a batch element, and where [sort], of [bound] tile rows of [bm] rows.
   Declined where it takes longer than [allowed]. *)
let buckets b ~allowed ~ids:(slot, at, iz, ir) ~rows ~m ~inner ~extent ~groups
    ~bm ~bound ~sort =
  let nb = (rows / m * groups) + 1 in
  let cost = buckets_cost ~rows ~ids_r:ir ~buckets:nb ~bound ~sort in
  if Plan.seconds ~threads:buckets_threads ~groups:(1, 1, 1) cost > allowed then
    decline ();
  let ws = Plan.workspace b in
  let w = Plan.words K.Contract_buckets in
  let refs = ref [] in
  let address word slot at write =
    Plan.set w word at;
    refs := { Plan.word; slot; write } :: !refs
  in
  let of_rows = Plan.take b (4 * rows) in
  address K.buckets_ids slot at false;
  address K.buckets_of ws of_rows true;
  let placed =
    if not sort then { of_rows; list_at = 0; table_at = 0 }
    else begin
      let list_at = Plan.take b (4 * rows) in
      let table_at = Plan.take b (12 * bound) in
      let firsts = Plan.take b (8 * (nb + 1)) in
      address K.buckets_list ws list_at true;
      address K.buckets_table ws table_at true;
      address K.buckets_firsts ws firsts true;
      { of_rows; list_at; table_at }
    end
  in
  List.iter
    (fun (word, v) -> Plan.set w word v)
    [
      (K.buckets_ids_z, iz);
      (K.buckets_ids_r, ir);
      (K.buckets_rows, rows);
      (K.buckets_m, m);
      (K.buckets_inner, inner);
      (K.buckets_extent, extent);
      (K.buckets_groups, groups);
      (K.buckets_buckets, nb);
      (K.buckets_tile, bm);
      (K.buckets_bound, bound);
      (K.buckets_sort, Bool.to_int sort);
    ];
  Plan.launch b K.Contract_buckets ~groups:(1, 1, 1) ~shared:buckets_shared
    ~cost w
    (Array.of_list (List.rev !refs));
  placed

(* A Rows call's ids as the prepass reads them: their slot, the byte of their
   first element, and their strides along the batch and the segmented axis; a
   copy in C order where their batch axes do not merge. *)
let ids_site b ops (ids : (int64, D.int64_elt) A.t) ~nb =
  let l = A.layout ids in
  let batch = Array.init nb Fun.id in
  match run l batch with
  | Some (_, sz) -> (slot b ops (A.Any ids), L.offset l * 8, sz, L.stride l nb)
  | None ->
      let v = packed b ops ids [| batch; [| nb |]; [||] |] in
      (v.slot, v.at, v.batch, v.major)

(* The prepass of a Rows call [r] of the axes [x]. *)
let call_buckets b ops ~allowed (r : rows) (x : axes) oa ~groups ~bm ~bound
    ~sort =
  let la = layout oa in
  let nf = Array.length x.af in
  let pos = ref 0 in
  Array.iteri (fun i a -> if a = r.a_axis then pos := i) x.af;
  let inner = extent la (Array.sub x.af (!pos + 1) (nf - !pos - 1)) in
  let ids = ids_site b ops r.ids ~nb:(Array.length x.ab) in
  buckets b ~allowed ~ids ~rows:(x.bt * x.m) ~m:x.m ~inner
    ~extent:(L.dim la r.a_axis) ~groups ~bm ~bound ~sort

(* b of a Rows call, its axes [groups] and its axis of [g] groups [axis], as the
   tiles read it: a view and its stride along the groups; copied with the groups
   after the batch where its axes do not merge, or widened so where its dtype is
   narrower than a byte, or scaled so where it has a [scale]. *)
let grouped_view ?scale b ops (A.Any x as ax) groups ~axis ~g =
  let gg = [| Array.append groups.(0) [| axis |]; groups.(1); groups.(2) |] in
  let narrow = D.bits (A.dtype x) < 8 in
  let apart = narrow || Option.is_some scale in
  match if apart then None else merged b ops x groups with
  | Some v -> (v, L.stride (A.layout x) axis)
  | None ->
      let v =
        match scale with
        | Some s -> scaled b ops ax s gg
        | None -> if narrow then widened b ops ax gg else packed b ops x gg
      in
      ({ v with batch = g * v.batch }, v.batch)

(* Loaders

   A loader reads b, the tiles' columns, and b's scale where they lie, a tile
   made for a key of b's dtype, b's scale and a's dtype: the scaled weights
   nx.quant makes, float4_e2m1fn by Exponent scales and int8 by float16 or
   float32 Value scales, each by float32, float16 or bfloat16. *)

(* A float operand's dtype, as the loaders and vector kernels take it. *)
type floating = Float32 | Float16 | Bfloat16

let floatings = [| Float32; Float16; Bfloat16 |]
let floating_index = function Float32 -> 0 | Float16 -> 1 | Bfloat16 -> 2

let floating_name = function
  | Float32 -> "f32"
  | Float16 -> "f16"
  | Bfloat16 -> "bf16"

(* A scaled weight's dtype and its scale's. *)
type scaled = E2m1_by_exp | I8_by_f16 | I8_by_f32

let scaleds = [| E2m1_by_exp; I8_by_f16; I8_by_f32 |]
let scaled_index = function E2m1_by_exp -> 0 | I8_by_f16 -> 1 | I8_by_f32 -> 2

let scaled_name = function
  | E2m1_by_exp -> "e2m1_by_exp"
  | I8_by_f16 -> "i8_by_f16"
  | I8_by_f32 -> "i8_by_f32"

let floating (A.Any x) =
  match A.dtype x with
  | D.Float32 -> Some Float32
  | D.Float16 -> Some Float16
  | D.Bfloat16 -> Some Bfloat16
  | _ -> None

let loader_b (A.Any x) (s : Nx_kernel.scale) =
  match (A.dtype x, s) with
  | D.Float4_e2m1fn, Exponent _ -> Some E2m1_by_exp
  | D.Int8, Value v -> (
      match A.dtype v with
      | D.Float16 -> Some I8_by_f16
      | D.Float32 -> Some I8_by_f32
      | _ -> None)
  | _ -> None

(* The kernels by name. Kernels named by parts are looked up by the parts'
   indices in tables made once: formatting the name on each call cost a vector
   call 91 of its 1,292 words. *)
let by_name =
  let t = Hashtbl.create (Array.length K.all) in
  Array.iter (fun k -> Hashtbl.replace t (K.name k) k) K.all;
  t

(* Each tile's banded twin, which reads an operand through its band: the tile's
   name then "_band", an aligned function's the wide one's. The other tiles read
   from the range's first term, without the band's select, so a call without a
   band pays nothing for it. *)
let band_twins =
  let t = Hashtbl.create 128 in
  let twin k =
    let n = K.name k in
    let n =
      if String.ends_with ~suffix:"_kn_a" n then
        String.sub n 0 (String.length n - 1) ^ "e"
      else n
    in
    match Hashtbl.find_opt by_name (n ^ "_band") with
    | Some b -> Hashtbl.replace t k b
    | None -> ()
  in
  Array.iter twin K.all;
  t

let band_kernel k =
  match Hashtbl.find_opt band_twins k with
  | Some b -> b
  | None -> invalid_arg "Contract.band_kernel: a tile of no banded twin"

(* The loaders by b's key, a's dtype, tile and layout, [None] where the metallib
   has none. *)
let loaders =
  let na = Array.length floatings in
  Array.init
    (Array.length scaleds * na * 4 * 2)
    (fun i ->
      let nk = i mod 2 = 1 and t = i / 2 mod 4 in
      let a = i / 8 mod na and b = i / 8 / na in
      let rows = if t >= 2 then "rows_" else ""
      and size = if t mod 2 = 0 then 32 else 16 in
      Hashtbl.find_opt by_name
        (Printf.sprintf "contract_%sld_%s_%s_%d_%s" rows
           (scaled_name scaleds.(b))
           (floating_name floatings.(a))
           size
           (if nk then "nk" else "kn")))

(* The loader of b's key [kb] and a's dtype [ka] at [tile], b n x k where [nk];
   none at 64 x 64. *)
let loader kb ka ~nk tile =
  let named t =
    let b = scaled_index kb and a = floating_index ka in
    loaders.((((((b * Array.length floatings) + a) * 4) + t) * 2)
             + Bool.to_int nk)
  in
  match tile with
  | T64 -> None
  | T32 -> named 0
  | T16 -> named 1
  | G32 -> named 2
  | G16 -> named 3

let signed_32 v = v >= -(1 lsl 31) && v < 1 lsl 31

(* The axes [axes] of [l], one run of b, as its scale [sl] steps along them: the
   scale's stride along the run and a block's elements, where each block is
   whole and every axis of more than one element but the run's innermost has
   blocks of one. A scale of [l]'s rank holds an axis's blocks as its extent. *)
let scale_run l sl axes =
  let live = List.filter (fun a -> L.dim l a > 1) (Array.to_list axes) in
  let whole a = L.dim l a mod L.dim sl a = 0 in
  if not (List.for_all whole live) then None
  else
    match List.rev live with
    | [] -> Some (0, 1)
    | last :: before ->
        if List.exists (fun a -> L.dim sl a <> L.dim l a) before then None
        else
          Option.map
            (fun (_, st) -> (st, L.dim l last / L.dim sl last))
            (run sl axes)

(* The scale's stride along a run of b's batch or groups: [0] where it holds one
   block of them, else where its blocks are of one element. *)
let scale_stride l sl axes =
  if Array.for_all (fun a -> L.dim sl a <= 1) axes then Some 0
  else match scale_run l sl axes with Some (st, 1) -> Some st | _ -> None

let scale_of : Nx_kernel.scale -> A.any = function
  | Exponent e -> A.Any e
  | Value v -> A.Any v

(* A loader indexes b's terms, its scale's and its blocks' by 32-bit integers,
   so b's terms number under 2^32. *)
let scale_steps ~nk ?axis l sl groups =
  let cols, terms =
    if nk then (groups.(1), groups.(2)) else (groups.(2), groups.(1))
  in
  let k = Array.fold_left (fun k a -> k * L.dim l a) 1 terms in
  let sj =
    match axis with None -> Some 0 | Some a -> scale_stride l sl [| a |]
  in
  match
    (scale_stride l sl groups.(0), sj, scale_run l sl cols, scale_run l sl terms)
  with
  | Some sz, Some sj, Some (sn, rn), Some (sk, rk)
    when fits_32 k && List.for_all signed_32 [ sz; sj; sn; rn; sk; rk ] ->
      Some { sz; sj; sn; rn; sk; rk }
  | _ -> None

(* b [x]'s axes [groups], of a dtype narrower than a byte, where a loader reads
   them in place: its index 0's bit, and its batch and staged rows' strides;
   where its axes merge, its inner run steps by one, and no stride along
   [groups] or [extra] is negative, so every element it reads lies at or past
   its index 0. *)
let in_place (A.Any x) groups extra =
  let l = A.layout x in
  let axes = Array.append (Array.concat (Array.to_list groups)) extra in
  if Array.exists (fun a -> L.stride l a < 0) axes then None
  else
    match Array.map (run l) groups with
    | [| Some (_, sb); Some (_, so); Some (ni, si) |] when si = 1 || ni <= 1 ->
        Some (L.offset l * D.bits (A.dtype x), sb, so)
    | _ -> None

(* The terms [axes] of [l] as Q8_0's blocks: two runs, the inner of stride 1, as
   the extent of the inner and the stride of the outer, each within signed 32
   bits. *)
let blocks_of l axes =
  let n = Array.length axes in
  let rec cut i =
    if i >= n then None
    else
      match (run l (Array.sub axes 0 i), run l (Array.sub axes i (n - i))) with
      | Some (no, so), Some (ni, 1) when no > 1 && ni > 1 ->
          if signed_32 ni && signed_32 so then Some (ni, so) else None
      | _ -> cut (i + 1)
  in
  cut 1

(* Where a loader reads b: a b narrower than a byte in place at its
   index 0's bit, with its batch and staged rows' strides; a b n x k of
   Q8_0's blocks in place at its index 0's element, also with the
   blocks' terms and stride; or b's view, copied where its axes do not
   merge. *)
(* Bands

   A cast operand's line that passes its share of a mixed call's buffer
   (nx_metal.mli's Contraction) is read from a band in workspace: before
   each chunk of the tiles' launches, contract_band casts the chunk's
   terms of each range [s] into the band from its term [s L], [L] the
   chunk's terms, and the tiles read the operand there. *)

type band = { a_band : int; b_band : int }

(* A banded line: [x], one batch element and line, its terms one run of [stride]
   elements, and a Rows call's [g] groups of b, [group] elements apart; the band
   takes at most [bytes]. *)
type banded = { line : A.any; stride : int; g : int; group : int; bytes : int }

let band_threads = K.threads K.Contract_band

(* A band launch's cost over [elements] terms in [groups] groups: each term's
   source bytes [read], or a line where its neighbours share none, and its cast
   written at [width] bytes, and {!cast_ops} a thread. *)
let band_cost ~elements ~groups ~read ~width =
  {
    Plan.bytes = elements * (read + width);
    ops = groups * band_threads * cast_ops;
    chain = 0;
  }

let band_read (d : banded) =
  let (A.Any x) = d.line in
  if abs d.stride > 1 then Plan.line else Int.max 1 (D.bits (A.dtype x) / 8)

let band_groups (d : banded) ~ranges length =
  (ceil_div length band_threads, ranges, d.g)

(* The most terms of a chunk of [d]'s band at [ranges] ranges: a multiple of the
   span whose band, [ranges] [g] chunks, takes at most [bytes] and whose launch
   fits a launch's time; one span at least. *)
let band_most stage (d : banded) ranges =
  let width = width stage in
  let fits q =
    let length = q * span in
    let groups = band_groups d ~ranges length in
    let x, y, z = groups in
    let elements = ranges * d.g * length in
    Plan.seconds ~threads:band_threads ~groups
      (band_cost ~elements ~groups:(x * y * z) ~read:(band_read d) ~width)
    <= Plan.allowed ()
  in
  let most = d.bytes / (ranges * d.g * width * span) in
  span * Int.max 1 (largest fits most)

(* The launch of contract_band that casts chunk [chunk] of [length] terms of
   [d]'s line into the band at workspace byte [at], for a call of [k] terms in
   [spans] spans and [ranges] ranges. *)
let band_launch b ops stage ~k ~spans ~ranges ~length ~at ~chunk (d : banded) =
  let (A.Any x) = d.line in
  let slot, byte, pos, phase = place b ops d.line in
  let (D.Any staged) = stage_dtype stage in
  let w = Plan.words K.Contract_band in
  let set = Plan.set w in
  set K.band_dst at;
  set K.band_src byte;
  set K.band_src_bit (phase + (D.bits (A.dtype x) * pos));
  set K.band_stride d.stride;
  set K.band_group d.group;
  set K.band_codes (code d.line lor (D.code staged lsl 8));
  set K.band_k k;
  set K.band_quot (spans / ranges);
  set K.band_rem (spans mod ranges);
  set K.band_ds (divisor ranges);
  set K.band_length length;
  set K.band_chunk chunk;
  let groups = band_groups d ~ranges length in
  let gx, gy, gz = groups in
  let cost =
    band_cost
      ~elements:(ranges * d.g * length)
      ~groups:(gx * gy * gz)
      ~read:(band_read d) ~width:(width stage)
  in
  Plan.launch b K.Contract_band ~groups ~cost w
    [|
      { Plan.word = K.band_dst; slot = Plan.workspace b; write = true };
      { Plan.word = K.band_src; slot; write = false };
    |]

(* [d]'s band in [b]'s workspace, [ranges] [g] chunks of [length] terms at
   [stage]'s width, and its byte there. *)
let band_take b stage ~ranges ~length = function
  | Some (d : banded) ->
      Some (d, Plan.take b (ranges * d.g * length * width stage))
  | None -> None

(* A band's view for a tile: its chunks at its byte of the workspace. *)
let band_view b = function
  | Some (_, at) -> Some { slot = Plan.workspace b; at; batch = 0; major = 0 }
  | None -> None

(* The dtype code a tile reads an operand [x] at: a band's at [stage]'s dtype, a
   scaled operand's float32, else [x]'s twin's. *)
let read_code stage band s x =
  if Option.is_some band then
    let (D.Any d) = stage_dtype stage in
    D.code d
  else if Option.is_some s then D.code D.Float32
  else twin_code x

type place =
  | Bits of { bit : int; sb : int; so : int }
  | Blocks of { at : int; sb : int; so : int; bl : int; bs : int }
  | View

(* Whether a loader's runs of b [x], narrower than a byte, all start on 4 bytes,
   so that it reads them a 32-bit word at a time: its index 0's [bit] on 4 bytes
   of its buffer, and each of the element [strides] that place a run, its batch,
   staged rows' and group's, whole words. A run starts at a multiple of its VB
   elements past these, and VB elements are whole words where the kernel reads
   words. *)
let words_of (A.Any x as ax) ~bit strides =
  let bits = dtype_bits ax in
  bit mod 8 = 0
  && Rig.Buffer.aligned (A.buffer x) ~at:(bit / 8) 4
  && List.for_all (fun s -> s * bits mod 32 = 0) strides

type loading = {
  le : extents;
  lp : t;
  lkernel : K.t;
  lscale : Nx_kernel.scale;
  lsteps : steps;
  lnk : bool;
  lgroups : int array array;
  lplace : place;
}

(* The tiles' call. An integer call ([acc]) stages [a]'s and [b]'s bytes at
   [I8]; its tiles store 64-bit sums, or into dst, where it is an integer of the
   accumulator's 32 or 64 bits, their low bits. A float32 call that reads
   [source] bytes an element of an operand not of its stage's dtype takes a
   loader, or raises {!Recast_call} for the casts of its unscaled operands. A
   Rows call ([segments]) takes the grouped tiles after its prepass; one of no
   groups has no products. A call of a loader's key whose b and scale fit takes
   the loader at every tile but 64 x 64. A piece of a call ([whole], {!call})
   takes the call's exchange, tile and ranges. An operand of [band] is a cast
   operand's line, read through its band ({!band_launch}). *)
let lower ~ops ~stage ~complex ?(source = 0) ?acc ?segments ?sa ?sb ?whole ?band
    ~batch ~contracting ~dst ~init oa ob =
  let drop = Option.map (fun r -> r.b_axis) segments in
  let x = axes_of ?drop ~batch ~contracting ~dst oa ob in
  let { bt; m; n; k; ab; bb; ak; bk; af; bf; dgroups } = x in
  let groups =
    match segments with Some r -> L.dim (layout ob) r.b_axis | None -> 0
  in
  let wb, wm, wn = match whole with Some w -> w | None -> (bt, m, n) in
  if bt * m * n = 0 then None
  else
    let segmented = Option.is_some segments in
    let exchanged = (not complex) && (not segmented) && exchanges ~m:wm ~n:wn in
    let rows_x, rows_groups, cols_x, kn, nk_form =
      if exchanged then
        (ob, [| bb; bf; bk |], oa, [| ab; ak; af |], [| ab; af; ak |])
      else (oa, [| ab; af; ak |], ob, [| bb; bk; bf |], [| bb; bf; bk |])
    in
    let rows_s, cols_s = if exchanged then (sb, sa) else (sa, sb) in
    let band_a, band_b =
      match band with
      | None -> (None, None)
      | Some { a_band; b_band } ->
          let banded side x kaxes bytes =
            if bytes = 0 then None
            else
              let stride =
                match run (layout x) kaxes with
                | Some (_, s) -> s
                | None ->
                    invalid_arg "Contract.call: a banded line of several runs"
              in
              let g, group =
                match segments with
                | Some r when side = 1 -> (groups, L.stride (layout x) r.b_axis)
                | Some _ | None -> (1, 0)
              in
              Some { line = x; stride; g; group; bytes }
          in
          (banded 0 oa ak a_band, banded 1 ob bk b_band)
    in
    let rows_bd, cols_bd =
      if exchanged then (band_b, band_a) else (band_a, band_b)
    in
    let is_banded = Option.is_some band_a || Option.is_some band_b in
    (* The tiles read a banded b's one column n x k: it clamps the columns past
       it onto it. *)
    let nk =
      Option.is_some cols_bd
      || ((not (merges cols_x kn)) && merges cols_x nk_form)
    in
    let cols_groups = if nk then nk_form else kn in
    let parts = if complex then 2 else 1 in
    let base =
      {
        batch = bt;
        m;
        n = parts * n;
        k = parts * k;
        stage;
        nk;
        complex;
        source;
      }
    in
    let whole = (wb, wm, parts * wn) in
    let most =
      if not is_banded then None
      else
        Some
          (fun ranges ->
            List.fold_left
              (fun m d -> Int.min m (band_most stage d ranges))
              max_int
              (List.filter_map Fun.id [ band_a; band_b ]))
    in
    let axis = Option.map (fun r -> r.b_axis) segments in
    (* A loader's form of b: its layout, axes and place. A b of Q8_0's blocks,
       whose terms merge into no run, reads n x k. *)
    let form () =
      let lb = layout cols_x in
      let extra = Array.of_list (Option.to_list axis) in
      if dtype_bits cols_x < 8 then
        Option.map
          (fun (bit, sb, so) -> (nk, cols_groups, Bits { bit; sb; so }))
          (in_place cols_x cols_groups extra)
      else if merges cols_x cols_groups then Some (nk, cols_groups, View)
      else
        match
          (run lb nk_form.(0), run lb nk_form.(1), blocks_of lb nk_form.(2))
        with
        | Some (_, sb), Some (_, so), Some (bl, bs) ->
            Some (true, nk_form, Blocks { at = L.offset lb; sb; so; bl; bs })
        | _ -> Some (nk, cols_groups, View)
    in
    let loading =
      match (cols_s, rows_s, acc) with
      | Some s, None, None when not complex -> (
          let ka =
            if Option.is_some rows_bd then Some Float32 else floating rows_x
          in
          match (ka, loader_b cols_x s, form ()) with
          | Some ka, Some kb, Some (lnk, lgroups, lplace) -> (
              let read =
                if Option.is_some rows_bd then width stage
                else Int.max 1 (dtype_bits rows_x / 8)
              in
              let le = { base with nk = lnk; source = read } in
              let st =
                scale_steps ~nk:lnk ?axis (layout cols_x)
                  (layout (scale_of s))
                  lgroups
              in
              match (st, tiled ~whole ?most ~segmented ~groups le) with
              | Some lsteps, Some lp ->
                  Option.map
                    (fun lkernel ->
                      {
                        le;
                        lp;
                        lkernel;
                        lscale = s;
                        lsteps;
                        lnk;
                        lgroups;
                        lplace;
                      })
                    (loader kb ka ~nk:lnk lp.tile)
              | _ -> None)
          | _ -> None)
      | _ -> None
    in
    let products = k > 0 && ((not segmented) || groups > 0) in
    let apart = Option.is_none (strides_of dst dgroups) in
    if products && source > 0 && Option.is_none loading then begin
      let cast bd s x =
        if Option.is_some bd then None else cast_into stage s x
      in
      let ca = cast band_a sa oa and cb = cast band_b sb ob in
      if (not is_banded) || Option.is_some ca || Option.is_some cb then
        raise (Recast_call { a = ca; b = cb; temp = apart })
    end;
    if apart then raise (Recast_call (pieces ~apart:true));
    if large (bt * m * n) then raise (Recast_call (pieces ~apart:false));
    let e, planned, nk, cols_groups =
      match loading with
      | Some l -> (l.le, Some l.lp, l.lnk, l.lgroups)
      | None ->
          (* The tiles read a banded operand's casts, of the stage's dtype. *)
          let e = if is_banded then { base with source = 0 } else base in
          (e, tiled ~whole ?most ~segmented ~groups e, nk, cols_groups)
    in
    match (strides_of dst dgroups, planned) with
    | None, _ | _, None -> decline ()
    | Some ds, Some p ->
        let total = if segmented then p.padded else bt * p.padded in
        if
          not
            (fits_32 total
            && fits_32 (bt * p.rows_out)
            && fits_32 p.cols_out && fits_32 p.tiles_m && fits_32 p.tiles_n
            && fits_32 ((bt * groups) + 1))
        then decline ();
        let b = Plan.make () in
        let allowed = Plan.allowed () in
        let whole = Array.for_all (fun l -> l.length = 0) p.launches in
        let acc_bits =
          match acc with Some (D.Any d) -> D.bits d | None -> 0
        in
        let _, _, dc = ds in
        (* [narrow]: the tiles store the low 32 bits of an integer sum, or round
           into the 16-bit staged dtype, where they store into dst; [into]:
           where dst's dtype lets them. *)
        let narrow, into =
          match acc with
          | Some _ ->
              ( acc_bits = 32,
                integer_array dst
                && dtype_bits dst = acc_bits
                && (acc_bits = 32 || acc_bits = 64) )
          | None when complex -> (false, dc = 1 || n = 1)
          | None ->
              let nr = stage <> F32 && staged dst = Some stage in
              (nr, code dst = D.code D.Float32 || nr)
        in
        let direct =
          products && p.ranges = 1 && whole && Option.is_none init && into
          && (not (overlaps dst oa))
          && not (overlaps dst ob)
        in
        let placed = Option.is_some acc in
        (* The finish reads init and writes dst at their positions, by their
           casts, but a float32 call's of float32, float16 or bfloat16, and a
           complex64 call's. *)
        let positioned =
          placed || not (complex || Option.is_some (staged dst))
        in
        let init =
          Option.map
            (init_site b ops ~placed:positioned ~exchanged ~dst dgroups)
            init
        in
        let outputs = bt * p.rows_out * p.cols_out in
        let planes =
          if (not products) || direct then None
          else
            Some
              ( Plan.workspace b,
                Plan.take b (sum_width stage * p.ranges * outputs) )
        in
        (* Each banded operand's band, [ranges] [g] chunks of the launches'
           terms, all of one length. *)
        let length =
          if Array.length p.launches = 0 then 0 else p.launches.(0).length
        in
        (* Rows' band, then cols': their places in the workspace. *)
        let rows_band =
          if products then band_take b stage ~ranges:p.ranges ~length rows_bd
          else None
        in
        let cols_band =
          if products then band_take b stage ~ranges:p.ranges ~length cols_bd
          else None
        in
        let grouping =
          match segments with
          | None -> None
          | Some _ when not products -> None
          | Some r ->
              let bv, sg =
                match Option.map (fun l -> l.lplace) loading with
                | _ when Option.is_some cols_band ->
                    (Option.get (band_view b cols_band), p.ranges * length)
                | _ when complex ->
                    let v, sg =
                      grouped_view b ops cols_x cols_groups ~axis:r.b_axis
                        ~g:groups
                    in
                    (real v, 2 * sg)
                | Some (Bits { bit; sb; so }) ->
                    ( placed_view b ops cols_x (bit / 8) sb so,
                      L.stride (layout cols_x) r.b_axis )
                | Some (Blocks { at; sb; so; _ }) ->
                    ( placed_view b ops cols_x (at * dtype_bits cols_x / 8) sb so,
                      L.stride (layout cols_x) r.b_axis )
                | Some View ->
                    grouped_view b ops cols_x cols_groups ~axis:r.b_axis
                      ~g:groups
                | None ->
                    grouped_view ?scale:cols_s b ops cols_x cols_groups
                      ~axis:r.b_axis ~g:groups
              in
              let at =
                call_buckets b ops ~allowed r x oa ~groups ~bm:(rows p.tile)
                  ~bound:p.tiles_m ~sort:true
              in
              Some
                ( bv,
                  at,
                  {
                    list = at.list_at;
                    table = at.table_at;
                    groups;
                    sg;
                    none = bt * groups;
                  } )
        in
        if products then begin
          let bytes = placed in
          let bv =
            match (grouping, Option.map (fun l -> l.lplace) loading) with
            | Some (bv, _, _), _ -> bv
            | None, _ when Option.is_some cols_band ->
                Option.get (band_view b cols_band)
            | None, _ when complex -> real (view_of b ops cols_x cols_groups)
            | None, Some (Bits { bit; sb; so }) ->
                placed_view b ops cols_x (bit / 8) sb so
            | None, Some (Blocks { at; sb; so; _ }) ->
                placed_view b ops cols_x (at * dtype_bits cols_x / 8) sb so
            | None, Some View -> view_of b ops cols_x cols_groups
            | None, None -> tile_view b ops ~bytes cols_s cols_x cols_groups
          in
          let a =
            match band_view b rows_band with
            | Some v -> v
            | None -> tile_view b ops ~bytes rows_s rows_x rows_groups
          in
          let a = if complex then real a else a in
          let kernel =
            match loading with
            | Some l -> l.lkernel
            | None when complex -> complex_kernel ~nk p.tile (fit e p)
            | None -> kernel stage ~nk p.tile (fit e p)
          in
          let kernel = if is_banded then band_kernel kernel else kernel in
          let shared = shared ~nk stage p.tile in
          let y, ystr =
            match planes with
            | Some (slot, at) ->
                let plane = p.rows_out * p.cols_out in
                ((slot, at), (p.cols_out, 1, plane, outputs))
            | None when complex ->
                let z, r, _ = ds in
                ( (slot b ops dst, L.offset (layout dst) * 8),
                  (2 * r, 1, 2 * z, 0) )
            | None ->
                let z, r, c = oriented ~exchanged ds in
                let width =
                  if placed then acc_bits / 8 else if narrow then 2 else 4
                in
                ((slot b ops dst, L.offset (layout dst) * width), (r, c, z, 0))
          in
          let kinds =
            (bytes_kind rows_x lsl K.codes_kind_a)
            lor (bytes_kind cols_x lsl K.codes_kind_b)
          in
          let dtypes =
            (read_code stage rows_band rows_s rows_x lsl K.codes_dtype_a)
            lor (read_code stage cols_band cols_s cols_x lsl K.codes_dtype_b)
          in
          let grouping = Option.map (fun (_, _, gr) -> gr) grouping in
          let phase, blocks, scale, words =
            match loading with
            | None -> (0, (0, 0), None, false)
            | Some l -> (
                let (A.Any sx) = scale_of l.lscale in
                let width = Int.max 1 (D.bits (A.dtype sx) / 8) in
                let at = L.offset (A.layout sx) * width in
                let scale = Some (slot b ops (A.Any sx), at, l.lsteps) in
                match l.lplace with
                | Bits { bit; sb; so } ->
                    let sg =
                      Option.fold ~none:0 ~some:(fun g -> g.sg) grouping
                    in
                    let words = words_of cols_x ~bit [ sb; so; sg ] in
                    (bit mod 8, (0, 0), scale, words)
                | Blocks { bl; bs; _ } -> (0, (bl, bs), scale, false)
                | View -> (0, (0, 0), scale, false))
          in
          let narrow = direct && narrow in
          (* A banded call runs chunk by chunk: each band's casts of the chunk,
             then every window's launch of it, which resumes the chunk before's
             sums. The launches run in order, so a band is never written under a
             read. *)
          let band =
            Bool.to_int (Option.is_some rows_band)
            lor (2 * Bool.to_int (Option.is_some cols_band))
          in
          let launches =
            if not is_banded then p.launches
            else
              let ls = Array.copy p.launches in
              Array.stable_sort
                (fun (x : launch) y -> Int.compare x.chunk y.chunk)
                ls;
              ls
          in
          let chunk = ref (-1) in
          for i = 0 to Array.length launches - 1 do
            let l = launches.(i) in
            if is_banded && l.chunk <> !chunk then begin
              chunk := l.chunk;
              let cast = function
                | Some (d, at) ->
                    band_launch b ops stage ~k:e.k ~spans:p.spans
                      ~ranges:p.ranges ~length ~at ~chunk:l.chunk d
                | None -> ()
              in
              cast rows_band;
              cast cols_band
            end;
            tile_launch b e p ~kernel ~shared ~a ~bv ~y ~ystr ~narrow ~kinds
              ~dtypes ~phase ~words ~blocks ?scale ?grouping ~band l
          done
        end;
        if not direct then begin
          let s = oriented ~exchanged ds in
          let dst =
            if positioned then placed_site b ops dst s else site_of b ops dst s
          in
          let acc =
            match acc with
            | Some acc -> acc
            | None when complex -> D.Any D.Complex64
            | None -> D.Any D.Float32
          in
          (* Rows of no group have no products: init alone, which the planes' +0
             would change where it is -0. *)
          let buckets =
            match grouping with
            | Some (_, at, _) when Option.is_some init ->
                Some (Plan.workspace b, at.of_rows, bt * groups)
            | Some _ | None -> None
          in
          if
            not
              (finish b ~allowed ~acc ?buckets ~planes
                 ~ranges:(if products then p.ranges else 0)
                 ~plane:outputs ~rows:p.rows_out ~total:(bt * p.rows_out)
                 ~cols:p.cols_out ~init ~dst ())
          then decline ()
        end;
        finished b

(* Vector calls

   A float32 call whose wider operand is of a key takes the vector kernels
   within the key's bound on its rows (contract.mli). The wider operand is read
   in place where its terms are contiguous, a plain key's also where its columns
   are, by the k x n kernels, or in blocks of a multiple of 32 contiguous terms
   (Q8_0's), a float4_e2m1fn one from a whole byte with its columns on whole
   bytes, and its scale's blocks along k a multiple of 8 terms or all of k. Else
   a plain one is copied in C order, and a scaled one's products with its scales
   are written as float32, which the float32 kernel reads: they are the products
   the key's kernel forms, so the sums keep their bits. The other operand is
   read in place where its terms are contiguous, else copied, and a scaled one
   as its products. *)

let dense_key (D.Any d) =
  match d with
  | D.Float32 -> Some Vector.Dense_f32
  | D.Float16 -> Some Vector.Dense_f16
  | D.Bfloat16 -> Some Vector.Dense_bf16
  | _ -> None

let vector_key (o : Nx_kernel.operand) =
  let (A.Any x) = o.array in
  match (A.dtype x, o.scale) with
  | (D.Float32 | D.Float16 | D.Bfloat16), None -> dense_key (D.Any (A.dtype x))
  | D.Float4_e2m1fn, Some (Exponent _) -> Some Vector.E2m1_by_exponent
  | D.Int8, Some (Value v) -> (
      match A.dtype v with
      | D.Float16 | D.Float32 -> Some Vector.Int8_by_value
      | _ -> None)
  | _ -> None

(* The other operand's dtype in a kernel's name: its own, float32 for its
   products; [None] where the kernels take no such operand. *)
let vector_other (o : Nx_kernel.operand) =
  match o.scale with Some _ -> Some Float32 | None -> floating o.array

(* A vector kernel's key: its wider operand's dtype, or its scaled weight's. *)
type mv_key = Dense of floating | Scaled of scaled

(* The kernel's key for a wider operand of [key] scaled by [scale]. *)
let mv_key (key : Vector.key) scale =
  match (key, scale) with
  | Vector.Dense_f32, _ -> Dense Float32
  | Vector.Dense_f16, _ -> Dense Float16
  | Vector.Dense_bf16, _ -> Dense Bfloat16
  | Vector.E2m1_by_exponent, _ -> Scaled E2m1_by_exp
  | Vector.Int8_by_value, Some (Nx_kernel.Value v) -> (
      match A.dtype v with
      | D.Float16 -> Scaled I8_by_f16
      | _ -> Scaled I8_by_f32)
  | Vector.Int8_by_value, _ -> Scaled I8_by_f32

let mv_keys =
  [|
    Dense Float32;
    Dense Float16;
    Dense Bfloat16;
    Scaled E2m1_by_exp;
    Scaled I8_by_f16;
    Scaled I8_by_f32;
  |]

let mv_key_index = function
  | Dense f -> floating_index f
  | Scaled s -> 3 + scaled_index s

let mv_key_name = function
  | Dense f -> floating_name f
  | Scaled E2m1_by_exp -> "e2m1"
  | Scaled I8_by_f16 -> "i8f16"
  | Scaled I8_by_f32 -> "i8f32"

let vector_passes = [| 1; 2; 4 |]

let pass_index what = function
  | 1 -> 0
  | 2 -> 1
  | 4 -> 2
  | p -> invalid_arg (what ^ ": pass " ^ string_of_int p)

let vector_name key ~other ~pass ~kn ~typed =
  Printf.sprintf "contract_mv_%s_%s_r%d_%s%s" (mv_key_name key)
    (floating_name other) pass
    (if kn then "kn" else "nk")
    (if typed then "_typed" else "")

(* The vector kernels by key, other operand, pass, layout and typed store,
   [None] where the metallib has none. *)
let vector_kernels =
  let no = Array.length floatings and np = Array.length vector_passes in
  Array.init
    (Array.length mv_keys * no * np * 4)
    (fun i ->
      let typed = i mod 2 = 1 and kn = i / 2 mod 2 = 1 in
      let p = i / 4 mod np and o = i / 4 / np mod no and k = i / 4 / np / no in
      Hashtbl.find_opt by_name
        (vector_name mv_keys.(k) ~other:floatings.(o) ~pass:vector_passes.(p)
           ~kn ~typed))

(* The twins (kernels.h's NX_CONTRACT_MV_TWIN) by key, other operand and typed
   store, [None] where the metallib has none. *)
let vector_twins =
  let no = Array.length floatings in
  Array.init
    (Array.length mv_keys * no * 2)
    (fun i ->
      let name =
        vector_name
          mv_keys.(i / 2 / no)
          ~other:floatings.(i / 2 mod no)
          ~pass:1 ~kn:false
          ~typed:(i mod 2 = 1)
      in
      Hashtbl.find_opt by_name (name ^ "_w32"))

let vector_kernel key ~other ~pass ~kn ~typed =
  let what = "Contract.vector_kernel" in
  let k = mv_key_index key
  and o = floating_index other
  and p = pass_index what pass in
  let no = Array.length floatings and np = Array.length vector_passes in
  match
    vector_kernels.((((((k * no) + o) * np) + p) * 4)
                    + (Bool.to_int kn * 2)
                    + Bool.to_int typed)
  with
  | Some k -> k
  | None -> invalid_arg (what ^ ": " ^ vector_name key ~other ~pass ~kn ~typed)

(* Which operand is the wider, [Some exchanged] where a call takes the vector
   kernels, and its key. *)
let vector_pick ?(cast = (None, None)) ~rows ~m ~n ~k oa ob =
  let ca, cb = cast in
  (* An operand cast into [c] is of [c]'s dense key, unscaled. *)
  let key c o = match c with Some d -> dense_key d | None -> vector_key o in
  let side c o =
    match c with
    | Some d -> Option.is_some (dense_key d)
    | None -> Option.is_some (vector_other o)
  in
  if k = 0 then None
  else
    match key cb ob with
    | Some kb
      when side ca oa
           && m <= Vector.bound kb ~rows
           && n >= Vector.columns
           && k <= Vector.terms kb ~rows ->
        Some (false, kb)
    | _ -> (
        match key ca oa with
        | Some ka
          when (not rows) && side cb ob
               && n <= Vector.bound ka ~rows
               && m >= Vector.columns
               && k <= Vector.terms ka ~rows ->
            Some (true, ka)
        | _ -> None)

(* The wider operand [o] of [key], its axes [groups] and a Rows call's axis of
   groups [axis], where the kernel reads it in place: its view, its stride along
   the groups, its blocks of terms, its scale's slot, byte and steps, and
   whether its columns are contiguous, its view's [major] then its terms'
   stride: a plain key's k x n form. *)
let vector_placed b ops (key : Vector.key) (o : Nx_kernel.operand) groups ~axis
    ~k =
  let l = layout o.array in
  let width = Int.max 1 (dtype_bits o.array / 8) in
  let sj = match axis with Some a -> L.stride l a | None -> 0 in
  (* Where the kernel reads [o], with no slot taken: its index 0's byte, its
     batch and columns' strides, its blocks, and whether k x n. *)
  let merged_at groups =
    match Array.map (run l) groups with
    | [| Some (_, sb); Some (_, so); Some (ni, si) |] when si = 1 || ni <= 1 ->
        Some (L.offset l * width, sb, so)
    | _ -> None
  in
  (* The scale's steps: n x k, of blocks a whole number of a span's 8-term steps
     or all of k; k x n, of any blocks, which a lane finds by a counter. *)
  let steps ~kn s =
    let (A.Any sx) = scale_of s in
    match scale_steps ~nk:true ?axis l (A.layout sx) groups with
    | Some st when kn || st.rk mod 8 = 0 || st.rk >= k -> Some (A.Any sx, st)
    | Some _ | None -> None
  in
  let kn_groups = [| groups.(0); groups.(2); groups.(1) |] in
  let place =
    match (key, o.scale) with
    | (Vector.Dense_f32 | Vector.Dense_f16 | Vector.Dense_bf16), None -> (
        match merged_at groups with
        | Some (at, sb, so) -> Some ((at, sb, so, (0, 0), false), None)
        | None ->
            Option.map
              (fun (at, sb, so) -> ((at, sb, so, (0, 0), true), None))
              (merged_at kn_groups))
    | Vector.E2m1_by_exponent, Some s -> (
        (* Whole bytes: index 0 at a byte, every stride whole bytes. *)
        let even v = v mod 2 = 0 in
        let extra = Array.of_list (Option.to_list axis) in
        let bytes (bit, sb, so) =
          bit mod 8 = 0 && even sb && even so && even sj
        in
        match in_place o.array groups extra with
        | Some ((bit, sb, so) as p) when bytes p ->
            Option.map
              (fun sc -> ((bit / 8, sb, so, (0, 0), false), Some sc))
              (steps ~kn:false s)
        | _ -> (
            match in_place o.array kn_groups extra with
            | Some ((bit, sb, so) as p) when bytes p ->
                Option.map
                  (fun sc -> ((bit / 8, sb, so, (0, 0), true), Some sc))
                  (steps ~kn:true s)
            | _ -> None))
    | Vector.Int8_by_value, Some s -> (
        let blocks =
          match merged_at groups with
          | Some (at, sb, so) ->
              Some ((at, sb, so, (0, 0), false), steps ~kn:false s)
          | None -> (
              match
                (run l groups.(0), run l groups.(1), blocks_of l groups.(2))
              with
              | Some (_, sb), Some (_, so), Some (bl, bs) when bl mod span = 0
                ->
                  Some ((L.offset l, sb, so, (bl, bs), false), steps ~kn:false s)
              | _ ->
                  Option.map
                    (fun (at, sb, so) ->
                      ((at, sb, so, (0, 0), true), steps ~kn:true s))
                    (merged_at kn_groups))
        in
        match blocks with Some (p, Some sc) -> Some (p, Some sc) | _ -> None)
    | _ -> None
  in
  (* The slots, once the kernel reads [o] where it lies. *)
  Option.map
    (fun ((at, sb, so, bl, kn), sc) ->
      let v = { slot = slot b ops o.array; at; batch = sb; major = so } in
      let scale =
        Option.map
          (fun (A.Any sx, st) ->
            let w = Int.max 1 (D.bits (A.dtype sx) / 8) in
            (slot b ops (A.Any sx), L.offset (A.layout sx) * w, st))
          sc
      in
      (v, sj, bl, scale, kn))
    place

(* Whether axis [i] is a side of one of [pairs] from the [j]th, the first sides
   where [first]. *)
let rec paired pairs ~first i j =
  j < Iarray.length pairs
  &&
  let x, y = Iarray.get pairs j in
  (if first then x else y) = i || paired pairs ~first i (j + 1)

(* The extent of [l]'s axes in no pair of [batch] or [contracting], nor [drop];
   or, where [terms], of its axes in [contracting]. *)
let free_extent l ~first ~terms ~drop batch contracting =
  let n = ref 1 in
  for i = 0 to L.rank l - 1 do
    let term = paired contracting ~first i 0 in
    let free = (not term) && (not (paired batch ~first i 0)) && i <> drop in
    if (terms && term) || ((not terms) && free) then n := !n * L.dim l i
  done;
  !n

(* Whether a call takes the vector kernels, by its extents and keys alone,
   allocating nothing: every call asks it first. A piece of a call ([whole])
   takes them where the call does. *)
let vector_takes ?cast ?segments ?whole ~batch ~contracting
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let la = layout oa.array and lb = layout ob.array in
  let drop = match segments with Some r -> r.b_axis | None -> -1 in
  let m =
    free_extent la ~first:true ~terms:false ~drop:(-1) batch contracting
  in
  let n = free_extent lb ~first:false ~terms:false ~drop batch contracting in
  let k = free_extent la ~first:true ~terms:true ~drop:(-1) batch contracting in
  let _, wm, wn = match whole with Some w -> w | None -> (0, m, n) in
  m * n > 0
  && Option.is_some
       (vector_pick ?cast ~rows:(Option.is_some segments) ~m:wm ~n:wn ~k oa ob)

(* The loads of the vector kernel's whole spans, one for every lane (the
   kernels' [form] word): 2, by 16-byte words, where every span of the wider
   operand's columns and of the other's rows starts on 16 bytes; 1, by 2-byte
   words, where the wider operand's start on 2 bytes, as Q8_0's 34-byte blocks
   do, and the other's on 16; else 0. Blocks and scale blocks are a span each or
   hold all of k, so that the kernels find a span's without dividing; or 3, by
   16-byte words as 2, where int8's scale blocks are of 16 terms, two a span, as
   Q6_K's are, and its weight has no blocks. e2m1's span forms read its weight
   without blocks, so a blocked one takes 0. Only views in place, [wbuf] and
   [abuf], have addresses whose alignment their buffers state: a copy takes
   0. *)
let vector_form (key : Vector.key) ~k ~(wv : view) ~sj ~blocks:(bl, bs) ~rk wbuf
    (ov : view) ~abytes abuf =
  match (wbuf, abuf) with
  | Some wb, Some ab ->
      let half =
        match key with
        | Vector.Dense_f32 -> 8
        | Dense_f16 | Dense_bf16 -> 4
        | Int8_by_value -> 2
        | E2m1_by_exponent -> 1
      in
      let starts n =
        Rig.Buffer.aligned wb ~at:wv.at n
        && List.for_all
             (fun s -> s * half / 2 mod n = 0)
             [ wv.major; wv.batch; sj; (if bl > 0 then bs else 0) ]
      in
      let rows =
        Rig.Buffer.aligned ab ~at:ov.at 16
        && ov.major * abytes mod 16 = 0
        && ov.batch * abytes mod 16 = 0
      in
      let whole b = b = span || b >= k in
      let scales = match rk with None -> true | Some rk -> whole rk in
      let halves =
        key = Vector.Int8_by_value && rk = Some (span / 2) && bl = 0
      in
      let blocked_e2m1 = key = Vector.E2m1_by_exponent && bl > 0 in
      if not rows then 0
      else if halves then if starts 16 then 3 else 0
      else if (not (scales && (bl = 0 || whole bl))) || blocked_e2m1 then 0
      else if starts 16 then 2
      else if starts 2 then 1
      else 0
  | _ -> 0

(* The form of {!vector_form} a key's twins run (kernels.h's
   NX_CONTRACT_MV_TWINS), 0 where it has none. *)
let twin_form = function
  | Scaled E2m1_by_exp -> 2
  | Scaled I8_by_f16 -> 1
  | Scaled I8_by_f32 -> 3
  | Dense _ -> 0

(* The kernel of a vector call of [key], as vector_kernel's, of whole spans'
   loads [form] (vector_form's) and [lanes] lanes: the key's twin where its one
   path takes every span, at the twin's form, k a multiple of 32 and 32 lanes;
   else the key's kernel. *)
let vector_launched key ~other ~pass ~kn ~typed ~form ~k ~lanes =
  let twin = twin_form key in
  if
    not
      (twin > 0 && pass = 1 && (not kn) && form = twin
      && k mod span = 0
      && lanes = 32)
  then vector_kernel key ~other ~pass ~kn ~typed
  else
    let no = Array.length floatings in
    let i =
      (((mv_key_index key * no) + floating_index other) * 2) + Bool.to_int typed
    in
    match vector_twins.(i) with
    | Some k -> k
    | None ->
        invalid_arg
          ("Contract.vector_launched: no twin of "
          ^ vector_name key ~other ~pass ~kn ~typed)

let vector_call ~ops ?segments ?whole ~batch ~contracting ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let drop = Option.map (fun r -> r.b_axis) segments in
  let x = axes_of ?drop ~batch ~contracting ~dst oa.array ob.array in
  let { bt; m; n; k; ab; bb; ak; bk; af; bf; dgroups } = x in
  let rows = Option.is_some segments in
  let _, wm, wn = match whole with Some w -> w | None -> (bt, m, n) in
  let pick =
    if bt * m * n = 0 then None else vector_pick ~rows ~m:wm ~n:wn ~k oa ob
  in
  match (pick, strides_of dst dgroups) with
  | None, _ -> None
  | Some _, None -> raise (Recast_call (pieces ~apart:true))
  | Some _, Some _ when large (bt * m * n) ->
      raise (Recast_call (pieces ~apart:false))
  | Some (exchanged, key), Some ds ->
      let wide, wgroups, other, ogroups =
        if exchanged then (oa, [| ab; af; ak |], ob, [| bb; bf; bk |])
        else (ob, [| bb; bf; bk |], oa, [| ab; af; ak |])
      in
      let rows_out, cols_out = if exchanged then (n, m) else (m, n) in
      let groups =
        match segments with
        | Some r -> L.dim (layout ob.array) r.b_axis
        | None -> 0
      in
      let axis = Option.map (fun r -> r.b_axis) segments in
      let b = Plan.make () in
      let allowed = Plan.allowed () in
      (* The wider operand: in place, else copied or as its products, with a
         Rows call's groups after its batch. *)
      let kkey, wv, sj, blocks, scale, kn, wbuf =
        match vector_placed b ops key wide wgroups ~axis ~k with
        | Some (v, sj, bl, sc, kn) ->
            (key, v, sj, bl, sc, kn, Some (buffer wide.array))
        | None ->
            let kkey =
              if Option.is_some wide.scale then Vector.Dense_f32 else key
            in
            let v, sj =
              match (axis, wide.scale) with
              | Some a, scale ->
                  grouped_view ?scale b ops wide.array wgroups ~axis:a ~g:groups
              | None, Some s -> (scaled b ops wide.array s wgroups, 0)
              | None, None -> (view_of b ops wide.array wgroups, 0)
            in
            (kkey, v, sj, (0, 0), None, false, None)
      in
      let ov =
        match other.scale with
        | Some s -> scaled b ops other.array s ogroups
        | None -> view_of b ops other.array ogroups
      in
      let abuf =
        match other.scale with
        | None when merges other.array ogroups -> Some (buffer other.array)
        | _ -> None
      in
      let form =
        let (A.Any o) = other.array in
        vector_form kkey ~k ~wv ~sj ~blocks
          ~rk:(Option.map (fun (_, _, (st : steps)) -> st.rk) scale)
          wbuf ov
          ~abytes:(D.bits (A.dtype o) / 8)
          abuf
      in
      let e =
        {
          batch = bt;
          m = rows_out;
          n = cols_out;
          k;
          stage = F32;
          nk = true;
          complex = false;
          source = 0;
        }
      in
      let v =
        match Vector.plan e ~key:kkey ~rows ~kn with
        | Some v -> v
        | None -> decline ()
      in
      if
        not
          (fits_32 (bt * v.Vector.groups)
          && fits_32 (bt * rows_out)
          && fits_32 cols_out
          && fits_32 ((bt * groups) + 1))
      then decline ();
      let outputs = bt * rows_out * cols_out in
      (* A kernel stores float32 sums, or a float16 or bfloat16 other operand's
         typed kernel sums of its dtype, rounded as the finish rounds them. *)
      let typed =
        match vector_other other with
        | Some Float16 -> code dst = D.code D.Float16
        | Some Bfloat16 -> code dst = D.code D.Bfloat16
        | Some Float32 | None -> false
      in
      let direct =
        Option.is_none init
        && (code dst = D.code D.Float32 || typed)
        && (not (overlaps dst oa.array))
        && not (overlaps dst ob.array)
      in
      let positioned = Option.is_none (staged dst) in
      let init =
        Option.map
          (init_site b ops ~placed:positioned ~exchanged ~dst dgroups)
          init
      in
      (* The workspace's slot only where a launch names it. *)
      let y, (yz, yr, yc) =
        if direct then
          ( (slot b ops dst, L.offset (layout dst) * (dtype_bits dst / 8)),
            oriented ~exchanged ds )
        else
          ( (Plan.workspace b, Plan.take b (4 * outputs)),
            (rows_out * cols_out, cols_out, 1) )
      in
      (* Rows of no group have no products: init alone, which the plane's +0
         would change where it is -0, so the kernel writes each row's bucket for
         the finish. *)
      let buckets =
        if rows && Option.is_some init then
          Some (Plan.workspace b, Plan.take b (4 * bt * rows_out), bt * groups)
        else None
      in
      let ids =
        Option.map
          (fun r ->
            let la = layout oa.array in
            let nf = Array.length af in
            let pos = ref 0 in
            Array.iteri (fun i a -> if a = r.a_axis then pos := i) af;
            let inner = extent la (Array.sub af (!pos + 1) (nf - !pos - 1)) in
            ( ids_site b ops r.ids ~nb:(Array.length ab),
              inner,
              L.dim la r.a_axis ))
          segments
      in
      let other_dtype = Option.value (vector_other other) ~default:Float32 in
      let wscale = if kkey = Vector.Dense_f32 then None else wide.scale in
      let kernel =
        vector_launched (mv_key kkey wscale) ~other:other_dtype
          ~pass:v.Vector.pass ~kn ~typed:(direct && typed) ~form ~k
          ~lanes:v.Vector.lanes
      in
      let launch (l : launch) =
        let w = Plan.words kernel in
        let set = Plan.set w in
        let refs = ref [] in
        let address word slot at write =
          set word at;
          refs := { Plan.word; slot; write } :: !refs
        in
        address K.mv_a ov.slot ov.at false;
        address K.mv_b wv.slot wv.at false;
        address K.mv_y (fst y) (snd y) true;
        set K.mv_lda ov.major;
        set K.mv_sa ov.batch;
        set K.mv_ldb wv.major;
        set K.mv_sb wv.batch;
        set K.mv_sj sj;
        set_pair w K.mv_blocks (fst blocks) (snd blocks);
        set K.mv_ldy yr;
        set K.mv_cy yc;
        set K.mv_sy yz;
        set K.mv_m rows_out;
        set K.mv_n cols_out;
        set K.mv_k k;
        set K.mv_lanes v.Vector.lanes;
        set K.mv_groups_n v.Vector.groups;
        set K.mv_first l.first;
        set K.mv_form form;
        (match scale with
        | None ->
            set_pair w K.mv_bsn 0 1;
            set_pair w K.mv_bsk 0 1
        | Some (slot, at, (st : steps)) ->
            address K.mv_bs slot at false;
            set_pair w K.mv_bsz st.sz st.sj;
            set_pair w K.mv_bsn st.sn st.rn;
            set_pair w K.mv_bsk st.sk st.rk);
        (match ids with
        | None ->
            set K.mv_inner 1;
            set K.mv_extent 1
        | Some ((slot, at, iz, ir), inner, ext) -> (
            address K.mv_ids slot at false;
            set K.mv_ids_z iz;
            set K.mv_ids_r ir;
            set K.mv_inner inner;
            set K.mv_extent ext;
            set_pair w K.mv_groups groups (bt * groups);
            match buckets with
            | None -> set K.mv_rows 1
            | Some (slot, at, _) ->
                set K.mv_rows 3;
                address K.mv_of slot at true));
        Plan.launch b kernel
          ~groups:(Vector.passes v, l.tiles, 1)
          ~shared:(Vector.shared v) ~resident:v.Vector.resident
          ~cost:(Vector.weights kkey e v ~tiles:l.tiles)
          w
          (Array.of_list (List.rev !refs))
      in
      Array.iter launch v.Vector.launches;
      if not direct then begin
        let s = oriented ~exchanged ds in
        let dst =
          if positioned then placed_site b ops dst s else site_of b ops dst s
        in
        if
          not
            (finish b ~allowed ~acc:(D.Any D.Float32) ?buckets ~planes:(Some y)
               ~ranges:1 ~plane:outputs ~rows:rows_out ~total:(bt * rows_out)
               ~cols:cols_out ~init ~dst ())
        then decline ()
      end;
      Some (finished b)

(* SIMT

   An integer call whose operands the tiles do not take, wider than a byte or
   floats, and every call of Contracting segments: a thread an output and a
   range of k, as the tiles' ranges, which reads each element at its own dtype
   and takes it into the accumulator by its cast. An integer sum is 64-bit and
   wraps; a float32 one is a chain of fused multiply-adds; a complex64 one's
   thread computes a column of the real form. Then the finish. *)

let simt_threads = K.threads K.Contract_simt

(* A SIMT thread's work in float32 operations, an estimate: its index, two
   32-bit divisions and its range's bounds; and a term's, two reads by dtype,
   each a switch and a cast or a sign extension, and a 64-bit multiply-add. *)
let simt_index_ops = 256
let simt_ops = 64
let simt_resident = Plan.holds ~threads:simt_threads 0

(* [G] of the SIMT kernel: the groups the GPU's waves hold. *)
let simt_spread = wave_cores * simt_resident

(* A launch of the outputs from [from], over each range's chunk [chunk] of
   [length] terms, [0] for whole ranges. *)
type sweep = { from : int; count : int; chunk : int; length : int }

(* The cost of [count] outputs, each a thread for each of [ranges] ranges over
   [terms] terms, of [n] columns and elements of [wa] and [wb] bytes: each
   group's reads a term, a row of b's elements and an element of each of its
   rows of a, and where [listed] its entry of a list, and its sums, read first
   where it resumes a chunk. *)
let simt_cost ?(listed = false) ~n ~wa ~wb ~ranges ~terms ~chunk count =
  let groups = ceil_div count simt_threads * ranges in
  let rows = Int.min simt_threads (ceil_div simt_threads n + 1) in
  let sums = if chunk > 0 then 16 else 8 in
  let entry = if listed then 4 else 0 in
  {
    Plan.bytes =
      (groups * terms * ((simt_threads * wb) + (rows * wa) + entry))
      + (groups * simt_threads * sums);
    ops = groups * simt_threads * (simt_index_ops + (terms * simt_ops));
    chain = (2 * terms) + Plan.read_links;
  }

(* The ranges of a sum of [spans] spans, of a call of [outputs] outputs, as the
   tiles': [T] the groups of its outputs. *)
let simt_ranges ~outputs ~spans =
  ranges_of ~tiles:(ceil_div outputs simt_threads) ~spread:simt_spread ~spans

(* The most ranges a Contracting call's group takes, its [cap]: as many as fill
   the waves, or 1 where its outputs fill half of them. A group of [spans] spans
   takes [max 1 (min cap (spans / 8))], {!simt_ranges}. *)
let simt_cap ~outputs = simt_ranges ~outputs ~spans:(most_ranges * range_spans)

(* The spans a range takes at most, of the [spans] spans of a group of a
   Contracting call under [cap]: a group of fewer than 16 spans takes one range,
   of up to 15, and one of more takes ranges of at least 8, at most as long as
   the longest group's. *)
let longest ~cap spans =
  let ranges s = Int.max 1 (Int.min cap (s / range_spans)) in
  Int.max
    (ceil_div spans (ranges spans))
    (Int.min spans ((2 * range_spans) - 1))

(* The ranges, as the tiles', of a call of [outputs] outputs and [k] terms,
   spans of [span] terms, and its launches: whole ranges in windows of the most
   groups that fit [allowed]; else windows of one group, each over chunks of the
   most spans that fit; else [None]. A Contracting call's ([listed]) groups take
   ranges of their own, at most those of [k] terms, each of up to {!longest}
   spans. *)
let simt_plan ?(listed = false) ~outputs ~n ~k ~span ~wa ~wb ~allowed () =
  let spans = ceil_div k span in
  let ranges = simt_ranges ~outputs ~spans in
  let whole =
    if listed then Int.min k (span * longest ~cap:(simt_cap ~outputs) spans)
    else Int.min k (span * ceil_div spans ranges)
  in
  let fits ~length ~chunk count =
    let terms = if length = 0 then whole else length in
    Plan.seconds ~threads:simt_threads
      ~groups:(ceil_div count simt_threads, 1, ranges)
      ~resident:simt_resident
      (simt_cost ~listed ~n ~wa ~wb ~ranges ~terms ~chunk count)
    <= allowed
  in
  let sweeps size ~chunks ~length =
    List.concat
      (List.init (ceil_div outputs size) (fun i ->
           let from = i * size in
           List.init chunks (fun chunk ->
               { from; count = Int.min size (outputs - from); chunk; length })))
  in
  let cap j = Int.min outputs (j * simt_threads) in
  let groups = ceil_div outputs simt_threads in
  match largest (fun j -> fits ~length:0 ~chunk:0 (cap j)) groups with
  | j when j > 0 ->
      Some (ranges, spans, whole, sweeps (cap j) ~chunks:1 ~length:0)
  | _ -> (
      let size = cap 1 in
      let deep q = fits ~length:(q * span) ~chunk:1 size in
      match largest deep (ceil_div whole span) with
      | 0 -> None
      | q ->
          let length = q * span in
          let chunks = ceil_div whole length in
          Some (ranges, spans, whole, sweeps size ~chunks ~length))

(* [x]'s axes [groups] as the SIMT kernel reads them: placed where they merge,
   else a copy in C order, or of a dtype narrower than a byte, its widening, or
   with a [scale], its products with its scales in float32; their strides; the
   dtype code it reads them at; and for a Rows call's b, [group]'s axis of
   groups and their count, its stride along them, the groups copied after the
   batch. *)
let simt_operand ?group ?scale b ops (A.Any x as ax) groups =
  match (strides_of ax groups, scale) with
  | Some s, None ->
      let sg =
        match group with
        | Some (axis, _) -> L.stride (A.layout x) axis
        | None -> 0
      in
      (place b ops ax, s, code ax, sg)
  | _ ->
      let groups, g =
        match group with
        | Some (axis, g) ->
            let gg = Array.append groups.(0) [| axis |] in
            ([| gg; groups.(1); groups.(2) |], g)
        | None -> (groups, 1)
      in
      let narrow = D.bits (A.dtype x) < 8 in
      let v, read =
        match scale with
        | Some s -> (scaled b ops ax s groups, D.code D.Float32)
        | None when narrow -> (widened b ops ax groups, twin_code ax)
        | None -> (packed b ops x groups, twin_code ax)
      in
      ((v.slot, v.at, 0, 0), (g * v.batch, v.major, 1), read, v.batch)

(* Contracting segments

   A Contracting call's prepass, contract_compact, lists each group's contracted
   indices in increasing order, a batch element's groups one after another, and
   each group's first entry; each output then sums its group's terms through the
   list, in its group's ranges. *)

type kseg = { kids : (int64, D.int64_elt) A.t; kgroups : int; kpair : int }

let compact_threads = K.threads K.Contract_compact
let compact_shared = K.compact_shared

(* The least terms of a prepass part, and the most counts of a group a part the
   parts take together. *)
let compact_part = 4096
let compact_counts = 1 lsl 22

(* A prepass thread's work in float32 operations, an estimate: a term's index's
   32-bit divisions, its id's test and an atomic; a word zeroed or a count
   scanned; and a lane's place among 32, 32 shuffles and compares. A chunk of
   the scan is the Rows prepass's: its barriers and prefix sums. A listing step
   reads its groups' next entries, a line, and writes its terms' entries. *)
let compact_ops = 128
let step_ops = 16
let place_ops = 160

(* Where the prepass put each group's first entry and the list. *)
type compacted = { offsets : int; entries : int }

(* Plans the prepass of [batch] batch elements of [k] terms, their ids at
   [(slot, at)] with strides [iz] along the batch and [ir] along the segmented
   axis, term q's id at index [(q / inner) mod extent] of it, into [groups]
   groups. Declined where a launch takes longer than [allowed]. *)
let compact b ~allowed ~ids:(slot, at, iz, ir) ~batch ~k ~inner ~extent ~groups
    =
  let parts =
    Int.max 1
      (Int.min (ceil_div k compact_part)
         (compact_counts / Int.max 1 (batch * groups)))
  in
  let part = ceil_div k parts in
  let ws = Plan.workspace b in
  let counts = Plan.take b (4 * Int.max 1 (batch * parts * groups)) in
  let offsets = Plan.take b (8 * batch * (groups + 1)) in
  let list = Plan.take b (4 * Int.max 1 (batch * k)) in
  let id = if abs ir <= 1 then 8 else Int.min Plan.line (8 * abs ir) in
  let nt = compact_threads in
  let over n = Int.max 1 (ceil_div n nt) in
  let per = ceil_div part nt and steps = ceil_div part 32 in
  let words = batch * parts * groups in
  let chunks = ceil_div (parts * groups) K.compact_chunk in
  let cper = K.compact_chunk / nt in
  let phases =
    [
      ( over words,
        {
          Plan.bytes = 4 * words;
          ops = over words * nt * step_ops;
          chain = Plan.read_links;
        } );
      ( batch * parts,
        {
          Plan.bytes = batch * k * (id + 4);
          ops = batch * parts * nt * per * compact_ops;
          chain = per * Plan.read_links;
        } );
      ( batch,
        {
          Plan.bytes = (8 * words) + (8 * batch * (groups + 1));
          ops = batch * nt * chunks * ((2 * cper * step_ops) + scan_ops);
          chain = chunks * ((2 * cper * Plan.read_links) + scan_links);
        } );
      ( ceil_div (batch * parts) (nt / 32),
        {
          Plan.bytes =
            (batch * k * (id + 4)) + (batch * parts * steps * Plan.line);
          ops = ceil_div (batch * parts) (nt / 32) * nt * steps * place_ops;
          chain = steps * ((2 * Plan.read_links) + 64);
        } );
    ]
  in
  let resident = Plan.holds ~threads:nt compact_shared in
  List.iter
    (fun (gx, cost) ->
      if Plan.seconds ~threads:nt ~groups:(gx, 1, 1) ~resident cost > allowed
      then decline ())
    phases;
  List.iteri
    (fun phase (gx, cost) ->
      let w = Plan.words K.Contract_compact in
      let refs = ref [] in
      let address word slot at write =
        Plan.set w word at;
        refs := { Plan.word; slot; write } :: !refs
      in
      address K.compact_ids slot at false;
      address K.compact_counts ws counts true;
      address K.compact_offsets ws offsets true;
      address K.compact_list ws list true;
      List.iter
        (fun (word, v) -> Plan.set w word v)
        [
          (K.compact_ids_z, iz);
          (K.compact_ids_r, ir);
          (K.compact_batch, batch);
          (K.compact_terms, k);
          (K.compact_inner, inner);
          (K.compact_extent, extent);
          (K.compact_groups, groups);
          (K.compact_parts, parts);
          (K.compact_phase, phase);
        ];
      Plan.launch b K.Contract_compact ~groups:(gx, 1, 1) ~shared:compact_shared
        ~resident ~cost w
        (Array.of_list (List.rev !refs)))
    phases;
  { offsets; entries = list }

(* The prepass of a Contracting call [ks] of the axes [x]. *)
let call_compact b ops ~allowed (ks : kseg) (x : axes) oa =
  let la = layout oa in
  let pa = x.ak.(ks.kpair) in
  let nk = Array.length x.ak in
  let inner = extent la (Array.sub x.ak (ks.kpair + 1) (nk - ks.kpair - 1)) in
  let ids = ids_site b ops ks.kids ~nb:(Array.length x.ab) in
  compact b ~allowed ~ids ~batch:x.bt ~k:x.k ~inner ~extent:(L.dim la pa)
    ~groups:ks.kgroups

(* The call. A Rows call's b reads its row's bucket's group; a Contracting
   call's outputs lead with its groups, each output summing its group's terms in
   its group's ranges. *)
let simt ~ops ~acc ?rows ?kseg ?sa ?sb ~batch ~contracting ~dst ~init oa ob =
  let drop = Option.map (fun r -> r.b_axis) rows in
  let x = axes_of ?drop ~batch ~contracting ~dst oa ob in
  let { bt; m; n; k; ab; bb; ak; bk; af; bf; dgroups } = x in
  let groups =
    match rows with Some r -> L.dim (layout ob) r.b_axis | None -> 0
  in
  let kcount = match kseg with Some ks -> ks.kgroups | None -> 1 in
  let complex = acc = D.Any D.Complex64 in
  let cols = if complex then 2 * n else n in
  (* A Contracting call's dst leads with its groups, before its batch. *)
  let dgroups =
    match kseg with
    | None -> dgroups
    | Some _ ->
        let nb = Array.length ab and nf = Array.length af in
        let rank = L.rank (layout dst) in
        [|
          Array.init (nb + 1) Fun.id;
          Array.init nf (fun i -> nb + 1 + i);
          Array.init (rank - nb - 1 - nf) (fun i -> nb + 1 + nf + i);
        |]
  in
  let outputs = kcount * bt * m * cols in
  if outputs = 0 then None
  else
    let ds =
      match (strides_of dst dgroups, kseg) with
      | Some s, _ -> s
      | None, None -> raise (Recast_call (pieces ~apart:true))
      (* A Contracting call's pieces would cut its groups, along which no
         operand is sliced. *)
      | None, Some _ -> decline ()
    in
    if Option.is_none kseg && large (bt * m * n) then
      raise (Recast_call (pieces ~apart:false));
    if
      not
        (fits_32 outputs
        && fits_32 ((bt * groups) + 1)
        && (Option.is_none kseg || fits_32 k))
    then decline ();
    let products = k > 0 && (Option.is_none rows || groups > 0) in
    let b = Plan.make () in
    let allowed = Plan.allowed () in
    let acc_code = match acc with D.Any d -> D.code d in
    let whole = integer acc in
    (* The finish reads init and writes dst at their positions, by their casts,
       but a float32 call's of float32, float16 or bfloat16, and a complex64
       call's. *)
    let positioned = whole || not (complex || Option.is_some (staged dst)) in
    let site =
      init_site b ops ~placed:positioned ~exchanged:false ~dst dgroups
    in
    let init = Option.map site init in
    let width (o : Nx_kernel.scale option) x =
      if Option.is_some o then 4 else Int.max 1 (dtype_bits x / 8)
    in
    let wa = width sa oa and wb = width sb ob in
    let span = if complex then span / 2 else span in
    let listed = Option.is_some kseg in
    let cap = simt_cap ~outputs in
    let compacted =
      if products then
        Option.map (fun ks -> call_compact b ops ~allowed ks x oa) kseg
      else None
    in
    let listing =
      Option.map
        (fun (c : compacted) ->
          {
            koff = (Plan.workspace b, c.offsets);
            kcount;
            kbatch = bt;
            kcap = cap;
          })
        compacted
    in
    let ranges, planes =
      if not products then (0, None)
      else
        match
          simt_plan ~listed ~outputs ~n:cols ~k ~span ~wa ~wb ~allowed ()
        with
        | None -> decline ()
        | Some (ranges, spans, whole_terms, sweeps) ->
            let (sa', a_at, a_pos, a_phase), (az, ar, ak'), a_code, _ =
              simt_operand ?scale:sa b ops oa [| ab; af; ak |]
            in
            let group = Option.map (fun r -> (r.b_axis, groups)) rows in
            let (sb', b_at, b_pos, b_phase), (bz, bk', bc), b_code, b_g =
              simt_operand ?group ?scale:sb b ops ob [| bb; bk; bf |]
            in
            let marks =
              Option.map
                (fun r ->
                  (call_buckets b ops ~allowed r x oa ~groups ~bm:1 ~bound:0
                     ~sort:false)
                    .of_rows)
                rows
            in
            let sums = if whole then 8 else 4 in
            let y = Plan.take b (sums * ranges * outputs) in
            let ws = Plan.workspace b in
            List.iter
              (fun sw ->
                let w = Plan.words K.Contract_simt in
                List.iter
                  (fun (word, v) -> Plan.set w word v)
                  [
                    (K.simt_a, a_at);
                    (K.simt_b, b_at);
                    (K.simt_y, y);
                    (K.simt_a_at, a_pos);
                    (K.simt_a_phase, a_phase);
                    (K.simt_a_dtype, a_code);
                    (K.simt_a_z, az);
                    (K.simt_a_r, ar);
                    (K.simt_a_k, ak');
                    (K.simt_b_at, b_pos);
                    (K.simt_b_phase, b_phase);
                    (K.simt_b_dtype, b_code);
                    (K.simt_b_z, bz);
                    (K.simt_b_k, bk');
                    (K.simt_b_c, bc);
                    (K.simt_acc, acc_code);
                    (K.simt_m, m);
                    (K.simt_n, cols);
                    (K.simt_k, k);
                    (K.simt_plane, outputs);
                    (K.simt_first, sw.from);
                    (K.simt_last, sw.from + sw.count);
                    (K.simt_ranges, ranges);
                    (K.simt_spans, spans);
                    (K.simt_length, sw.length);
                    (K.simt_chunk, sw.chunk);
                    (K.simt_groups, groups);
                    (K.simt_b_g, b_g);
                    (K.simt_segmented, Bool.to_int (Option.is_some marks));
                  ];
                let refs =
                  [
                    { Plan.word = K.simt_a; slot = sa'; write = false };
                    { Plan.word = K.simt_b; slot = sb'; write = false };
                    { Plan.word = K.simt_y; slot = ws; write = true };
                  ]
                in
                let refs =
                  match marks with
                  | None -> refs
                  | Some at ->
                      Plan.set w K.simt_buckets at;
                      { Plan.word = K.simt_buckets; slot = ws; write = false }
                      :: refs
                in
                let refs =
                  match compacted with
                  | None -> refs
                  | Some c ->
                      Plan.set w K.simt_koff c.offsets;
                      Plan.set w K.simt_klist c.entries;
                      Plan.set w K.simt_kgroups kcount;
                      Plan.set w K.simt_batch bt;
                      Plan.set w K.simt_cap cap;
                      { Plan.word = K.simt_koff; slot = ws; write = false }
                      :: { Plan.word = K.simt_klist; slot = ws; write = false }
                      :: refs
                in
                let terms = if sw.length = 0 then whole_terms else sw.length in
                Plan.launch b K.Contract_simt
                  ~groups:(ceil_div sw.count simt_threads, 1, ranges)
                  ~resident:simt_resident
                  ~cost:
                    (simt_cost ~listed ~n:cols ~wa ~wb ~ranges ~terms
                       ~chunk:sw.chunk sw.count)
                  w (Array.of_list refs))
              sweeps;
            (ranges, Some (ws, y))
    in
    let s = ds in
    let dst =
      if positioned then placed_site b ops dst s else site_of b ops dst s
    in
    if
      not
        (finish b ~allowed ~acc ?listing ~planes ~ranges ~plane:outputs ~rows:m
           ~total:(kcount * bt * m)
           ~cols ~init ~dst ())
    then decline ();
    finished b

(* Calls *)

let complex64 x = code x = D.code D.Complex64

(* Whether [x] is of a dtype Apple7 computes in: neither float64 nor
   complex128. *)
let hosted (A.Any x) =
  match A.dtype x with D.Float64 | D.Complex128 -> false | _ -> true

(* Whether [o]'s scale, where it has one, is {!hosted}. *)
let scale_hosted (o : Nx_kernel.operand) =
  match o.scale with
  | Some (Exponent e) -> hosted (A.Any e)
  | Some (Value v) -> hosted (A.Any v)
  | None -> true

(* Operands the integer tiles take: integers of 8 bits or fewer and booleans. *)
let bytewise (A.Any x) =
  let d = A.dtype x in
  D.bits d <= 8
  &&
  match D.kind d with
  | D.Signed | D.Unsigned | D.Boolean -> true
  | D.Float | D.Complex -> false

(* Whether float16, and bfloat16, holds every value of [x]'s dtype. *)
let half_exact (A.Any x) =
  match A.dtype x with
  | D.Float16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn | D.Int8
  | D.Uint8 | D.Int4 | D.Uint4 | D.Bool | D.Bit ->
      true
  | _ -> false

let bfloat_exact (A.Any x) =
  match A.dtype x with
  | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 | D.Float4_e2m1fn | D.Int8
  | D.Uint8 | D.Int4 | D.Uint4 | D.Bool | D.Bit ->
      true
  | _ -> false

(* The stage of a float32 call of operands [a] and [b] not of one staged dtype:
   float16 where it holds both operands' values, else bfloat16 where it does,
   else float32, which holds them or their casts. *)
let converted a b =
  if half_exact a && half_exact b then F16
  else if bfloat_exact a && bfloat_exact b then Bf16
  else F32

(* The bytes of [x]'s element as a loader reads it, widened where narrower than
   a byte. *)
let source_width (A.Any x) =
  let (D.Any t) = Moves.twin (D.Any (A.dtype x)) in
  D.bits t / 8

type planned = Planned of Plan.t option | Recast of recast

let rows_of (segments : _ Nx_kernel.segments option) =
  match segments with
  | Some (Rows r) -> Some { ids = r.ids; a_axis = r.a_axis; b_axis = r.b_axis }
  | Some (Contracting _) | None -> None

let bandable ~contracting (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let one x axes = Option.is_some (run (layout x) axes) in
  (one oa.array (pairs_fst contracting), one ob.array (pairs_snd contracting))

let vectored ~whole ~batch ~contracting ~segments cast oa ob =
  vector_takes ~cast ?segments:(rows_of segments) ~whole ~batch ~contracting oa
    ob

let plan_call ?whole ?band ~ops ~batch ~contracting
    ~(segments : _ Nx_kernel.segments option) ~acc ~dst ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let all = oa.array :: ob.array :: dst :: Option.to_list init in
  let unsegmented = Option.is_none segments in
  let kseg =
    match segments with
    | Some (Contracting c) ->
        Some { kids = c.ids; kgroups = c.groups; kpair = c.pair }
    | Some (Rows _) | None -> None
  in
  let segments = rows_of segments in
  (* Whether the arrays, and the operands' scales, are of dtypes the kernels
     take. *)
  let hosted_all = List.for_all hosted all in
  let hosted_scaled = hosted_all && scale_hosted oa && scale_hosted ob in
  (* A scaled operand's tiles read float32: its products with its scales. *)
  let read_f32 (o : Nx_kernel.operand) =
    Option.is_some o.scale || code o.array = D.code D.Float32
  in
  let width (o : Nx_kernel.operand) =
    if Option.is_some o.scale then 4 else source_width o.array
  in
  let simt_k kseg =
    Ok
      (simt ~ops ~acc ~kseg ?sa:oa.scale ?sb:ob.scale ~batch ~contracting ~dst
         ~init oa.array ob.array)
  in
  try
    match (kseg, oa.scale, ob.scale, acc) with
    | Some ks, _, _, D.Any D.Float32 when hosted_scaled -> simt_k ks
    | Some ks, None, None, D.Any D.Complex64
      when complex64 oa.array && complex64 ob.array && complex64 dst ->
        simt_k ks
    | Some ks, None, None, acc when integer acc && hosted_all -> simt_k ks
    | Some _, _, _, _ -> Error A.Declined
    | None, _, _, _ -> (
        (* A float32 call of few rows over a weight of a key takes the vector
           kernels, which decline it whole where they find no launches. A banded
           call is one they do not take ({!vectored}). *)
        let vector =
          match acc with
          | D.Any D.Float32
            when Option.is_none band
                 && vector_takes ?segments ?whole ~batch ~contracting oa ob
                 && hosted_scaled ->
              vector_call ~ops ?segments ?whole ~batch ~contracting ~dst ~init
                oa ob
          | _ -> None
        in
        match vector with
        | Some p -> Ok p
        | None -> (
            match
              (oa.scale, ob.scale, acc, staged oa.array, staged ob.array)
            with
            | Some _, _, D.Any D.Float32, _, _
            | _, Some _, D.Any D.Float32, _, _
              when hosted_scaled ->
                let source =
                  if read_f32 oa && read_f32 ob then 0
                  else Int.max (width oa) (width ob)
                in
                Ok
                  (lower ~ops ~stage:F32 ~complex:false ~source ?segments
                     ?sa:oa.scale ?sb:ob.scale ?whole ?band ~batch ~contracting
                     ~dst ~init oa.array ob.array)
            | None, None, D.Any D.Float32, Some sa, Some sb
              when sa = sb && hosted dst
                   && Option.fold ~none:true ~some:hosted init ->
                Ok
                  (lower ~ops ~stage:sa ~complex:false ?segments ?whole ~batch
                     ~contracting ~dst ~init oa.array ob.array)
            | None, None, D.Any D.Float32, _, _ when hosted_all ->
                let a = oa.array and b = ob.array in
                let source = Int.max (source_width a) (source_width b) in
                Ok
                  (lower ~ops ~stage:(converted a b) ~complex:false ~source
                     ?segments ?whole ?band ~batch ~contracting ~dst ~init a b)
            | None, None, D.Any D.Complex64, _, _
              when complex64 oa.array && complex64 ob.array && complex64 dst ->
                Ok
                  (lower ~ops ~stage:F32 ~complex:true ?segments ?whole ~batch
                     ~contracting ~dst ~init oa.array ob.array)
            | None, None, acc, _, _ when integer acc && hosted_all ->
                if unsegmented && bytewise oa.array && bytewise ob.array then
                  Ok
                    (lower ~ops ~stage:I8 ~complex:false ~acc ?whole ~batch
                       ~contracting ~dst ~init oa.array ob.array)
                else
                  Ok
                    (simt ~ops ~acc ?rows:segments ~batch ~contracting ~dst
                       ~init oa.array ob.array)
            | _ -> Error A.Declined))
  with Decline -> Error A.Declined

let call ?whole ?band ~ops ~batch ~contracting ~segments ~acc ~dst ~init oa ob =
  match
    plan_call ?whole ?band ~ops ~batch ~contracting ~segments ~acc ~dst ~init oa
      ob
  with
  | Ok p -> Ok (Planned p)
  | Error e -> Error e
  | exception Recast_call r -> Ok (Recast r)

(* The matrix routines' products *)

type where = Operand of int | Work

(* A matrix a product reads or writes: where, the byte its layout's offset
   counts from, its dtype and its layout. *)
type matrix = { where : where; base : int; mdt : D.any; ml : L.t }

let operand k dt l = { where = Operand k; base = 0; mdt = dt; ml = l }
let array k (A.Any x) = operand k (D.Any (A.dtype x)) (A.layout x)
let work ~at dt l = { where = Work; base = at; mdt = dt; ml = l }

type options = {
  init_first : bool;
  neg : bool;
  conj_a : bool;
  conj_b : bool;
  tri : [ `Lower | `Upper ] option;
  ranges : int option;
}

let plain =
  {
    init_first = false;
    neg = false;
    conj_a = false;
    conj_b = false;
    tri = None;
    ranges = None;
  }

let ranges ~rows ~cols ~k =
  ranges_of
    ~tiles:(ceil_div rows 64 * ceil_div cols 64)
    ~spread:(spread F32 T64) ~spans:(ceil_div k span)

let m_slot b x =
  match x.where with Operand k -> Plan.slot b k | Work -> Plan.workspace b

let m_width x =
  let (D.Any d) = x.mdt in
  D.bits d / 8

let m_at x = x.base + (L.offset x.ml * m_width x)
let m_dim x i = L.dim x.ml i
let m_stride x i = L.stride x.ml i
let m_code x = match x.mdt with D.Any d -> D.code d

(* Whether [x]'s axis [i] steps by one element, or holds at most one. *)
let unit x i = m_dim x i <= 1 || m_stride x i = 1

(* The parts of a routine's element: 1 for float32, 2 for complex64. *)
let parts_of what (D.Any d) =
  if D.code d = D.code D.Float32 then 1
  else if D.code d = D.code D.Complex64 then 2
  else invalid_arg (what ^ ": a dtype")

let copy_ops = 64
let copy_threads = K.threads K.Contract_copy

(* The cost of copying [rows] rows of [cols] elements of [parts] float32 parts,
   their columns [sc] elements apart: each part read, a line where neighbours
   share none, and written, and two lines at each row's ends. *)
let copy_cost ~cols ~parts ~sc rows =
  let gx = ceil_div cols copy_threads in
  let e = 4 * parts in
  let read = if abs sc <= 1 then e else Int.min Plan.line (abs sc * e) in
  {
    Plan.bytes = (rows * cols * (read + e)) + (4 * Plan.line * rows);
    ops = gx * rows * copy_threads * copy_ops;
    chain = Plan.read_links;
  }

let copy b ~src ~dst =
  let parts = parts_of "Contract.copy" src.mdt in
  let shaped x =
    L.rank x.ml = 3
    && Array.for_all (fun i -> m_dim x i = m_dim src i) [| 0; 1; 2 |]
  in
  if L.rank src.ml <> 3 || m_code dst <> m_code src || not (shaped dst) then
    invalid_arg "Contract.copy: a shape or dtype";
  let rows = m_dim src 1 and cols = m_dim src 2 in
  let total = m_dim src 0 * rows in
  if total * cols > 0 then begin
    let sc = m_stride src 2 in
    let span ~start g = Int.min (g * copy_threads) (cols - start) in
    let fits ~g r =
      Plan.seconds ~threads:copy_threads ~groups:(g, r, 1)
        (copy_cost ~cols:(span ~start:0 g) ~parts ~sc r)
      <= Plan.allowed ()
    in
    let launch ~first ~count ~start g =
      let w = Plan.words K.Contract_copy in
      Plan.set w K.ccopy_src (m_at src);
      Plan.set w K.ccopy_dst (m_at dst);
      Plan.set w K.ccopy_sz (m_stride src 0);
      Plan.set w K.ccopy_sr (m_stride src 1);
      Plan.set w K.ccopy_sc sc;
      Plan.set w K.ccopy_dz (m_stride dst 0);
      Plan.set w K.ccopy_dr (m_stride dst 1);
      Plan.set w K.ccopy_dc (m_stride dst 2);
      Plan.set w K.ccopy_rows rows;
      Plan.set w K.ccopy_cols cols;
      Plan.set w K.ccopy_parts parts;
      Plan.set w K.ccopy_first first;
      Plan.set w K.ccopy_start start;
      Plan.launch b K.Contract_copy ~groups:(g, count, 1)
        ~cost:(copy_cost ~cols:(span ~start g) ~parts ~sc count)
        w
        [|
          { Plan.word = K.ccopy_src; slot = m_slot b src; write = false };
          { Plan.word = K.ccopy_dst; slot = m_slot b dst; write = true };
        |]
    in
    (* contract_copy takes a row's index in 32 bits. *)
    if not (fits_32 total) then decline ();
    let groups = ceil_div cols copy_threads in
    let most = largest (fits ~g:groups) total in
    if most > 0 then begin
      let first = ref 0 in
      while !first < total do
        let count = Int.min most (total - !first) in
        launch ~first:!first ~count ~start:0 groups;
        first := !first + count
      done
    end
    else
      (* A row past a launch: each row's columns in runs of [most] groups. *)
      match Plan.most ~fits:(fun g -> fits ~g 1) groups with
      | None -> decline ()
      | Some most ->
          for y = 0 to total - 1 do
            let start = ref 0 in
            while !start < cols do
              let g = Int.min most (ceil_div (cols - !start) copy_threads) in
              launch ~first:y ~count:1 ~start:!start g;
              start := !start + (g * copy_threads)
            done
          done
  end

(* [x] copied into a piece of the workspace, its axis [inner], 1 or 2, stepping
   by one element. *)
let copied b x ~inner =
  let z = m_dim x 0 and r = m_dim x 1 and c = m_dim x 2 in
  let strides = if inner = 2 then [| r * c; c; 1 |] else [| r * c; 1; r |] in
  let at = Plan.take b (z * r * c * m_width x) in
  let l =
    L.v ~strides:(Iarray.of_array strides) (Iarray.of_array [| z; r; c |])
  in
  let w = work ~at x.mdt l in
  copy b ~src:x ~dst:w;
  w

(* [x] as a tile's operand: its batch's stride and [major]'s, in elements. *)
let m_view b x ~major =
  {
    slot = m_slot b x;
    at = m_at x;
    batch = m_stride x 0;
    major = m_stride x major;
  }

(* The routines' float32 tiles, which read a routine's options: plain, and
   negated. *)
let routined ~nk tile fit =
  match (nk, tile, fit) with
  | false, T64, Whole -> K.Contract_f32_rt_64_kn_a
  | false, T64, Wide -> K.Contract_f32_rt_64_kn_e
  | false, T64, Narrow -> K.Contract_f32_rt_64_kn_n
  | false, T32, Whole -> K.Contract_f32_rt_32_kn_a
  | false, T32, Wide -> K.Contract_f32_rt_32_kn_e
  | false, T32, Narrow -> K.Contract_f32_rt_32_kn_n
  | false, T16, Whole -> K.Contract_f32_rt_16_kn_a
  | false, T16, Wide -> K.Contract_f32_rt_16_kn_e
  | false, T16, Narrow -> K.Contract_f32_rt_16_kn_n
  | true, T64, _ -> K.Contract_f32_rt_64_nk
  | true, T32, _ -> K.Contract_f32_rt_32_nk
  | true, T16, _ -> K.Contract_f32_rt_16_nk
  | _, (G32 | G16), _ -> invalid_arg "Contract.routined: a grouped tile"

let negated ~nk tile fit =
  match (nk, tile, fit) with
  | false, T64, Whole -> K.Contract_f32_neg_64_kn_a
  | false, T64, Wide -> K.Contract_f32_neg_64_kn_e
  | false, T64, Narrow -> K.Contract_f32_neg_64_kn_n
  | false, T32, Whole -> K.Contract_f32_neg_32_kn_a
  | false, T32, Wide -> K.Contract_f32_neg_32_kn_e
  | false, T32, Narrow -> K.Contract_f32_neg_32_kn_n
  | false, T16, Whole -> K.Contract_f32_neg_16_kn_a
  | false, T16, Wide -> K.Contract_f32_neg_16_kn_e
  | false, T16, Narrow -> K.Contract_f32_neg_16_kn_n
  | true, T64, _ -> K.Contract_f32_neg_64_nk
  | true, T32, _ -> K.Contract_f32_neg_32_nk
  | true, T16, _ -> K.Contract_f32_neg_16_nk
  | _, (G32 | G16), _ -> invalid_arg "Contract.negated: a grouped tile"

(* The tiles' operands of a real product: a's rows of contiguous terms, and b k
   x n or n x k, each read in place where it lies so, else copied; where
   exchanged, a is m transposed and b a transposed. *)
let real_operands b ~exchanged a m =
  if not exchanged then
    let a = if unit a 2 then a else copied b a ~inner:2 in
    let va = m_view b a ~major:1 in
    if unit m 2 then (va, m_view b m ~major:1, false)
    else if unit m 1 then (va, m_view b m ~major:2, true)
    else (va, m_view b (copied b m ~inner:2) ~major:1, false)
  else
    let m = if unit m 1 then m else copied b m ~inner:1 in
    let va = m_view b m ~major:2 in
    if unit a 1 then (va, m_view b a ~major:2, false)
    else if unit a 2 then (va, m_view b a ~major:1, true)
    else (va, m_view b (copied b a ~inner:2) ~major:1, true)

(* A complex product's operands as their float32 parts: a's rows of contiguous
   terms, and m k x n or n x k, each read in place where it lies so, else
   copied; the complex tiles stage m's real blocks. *)
let complex_operands b a m =
  let a = if unit a 2 then a else copied b a ~inner:2 in
  let va = real (m_view b a ~major:1) in
  if unit m 2 then (va, real (m_view b m ~major:1), false)
  else if unit m 1 then (va, real (m_view b m ~major:2), true)
  else (va, real (m_view b (copied b m ~inner:2) ~major:1), false)

(* The complex tiles' sign flips of [o] (kernels.h's [codes] field
   NX_CODES_FLIPS). *)
let flips o =
  Bool.to_int o.conj_b
  lor (Bool.to_int o.conj_a lsl 1)
  lor (Bool.to_int o.neg lsl 2)

let product b o ~acc ~init ~y a m =
  let complex = parts_of "Contract.product" acc = 2 in
  let code = match acc with D.Any d -> D.code d in
  let dims x = Array.init (L.rank x.ml) (m_dim x) in
  let batch, rows, cols, k =
    match (dims y, dims a, dims m) with
    | [| z; r; c |], [| za; ra; k |], [| zm; km; cm |]
      when za = z && ra = r && zm = z && km = k && cm = c ->
        (z, r, c, k)
    | _ -> invalid_arg "Contract.product: a shape"
  in
  if List.exists (fun x -> m_code x <> code) [ y; a; m ] then
    invalid_arg "Contract.product: a dtype";
  let outputs = batch * rows * cols in
  let skip = outputs = 0 || (k = 0 && init) in
  if not skip then begin
    let mark = Plan.mark b in
    let exchanged = (not complex) && exchanges ~m:rows ~n:cols in
    (* The tiles' operands, where there are terms: slots that no launch reads
       would fail the call. *)
    let operands =
      if k = 0 then None
      else if complex then Some (complex_operands b a m)
      else Some (real_operands b ~exchanged a m)
    in
    let nk = match operands with Some (_, _, nk) -> nk | None -> false in
    let parts = if complex then 2 else 1 in
    let e =
      {
        batch;
        m = rows;
        n = parts * cols;
        k = parts * k;
        stage = F32;
        nk;
        complex;
        source = 0;
      }
    in
    let p =
      match planned ?ranges:o.ranges e with
      | Some p
        when fits_32 (batch * p.padded)
             && fits_32 (batch * p.rows_out)
             && fits_32 p.cols_out ->
          p
      | _ -> decline ()
    in
    let ys = (m_stride y 0, m_stride y 1, m_stride y 2) in
    let sz, sr, sc = oriented ~exchanged ys in
    let first = o.init_first && init in
    (* A triangle as the tiles see it: exchanged, y's columns are their rows. *)
    let tri =
      match (o.tri, exchanged) with
      | None, _ -> 0
      | Some `Lower, false | Some `Upper, true -> 1
      | Some `Upper, false | Some `Lower, true -> 2
    in
    let pair = if complex && tri <> 0 then 4 else 0 in
    let direct =
      k > 0 && p.ranges = 1 && ((not init) || first)
      && ((not complex) || unit y 2)
    in
    let plane = p.rows_out * p.cols_out in
    let planes =
      if direct || k = 0 then None
      else Some (Plan.take b (4 * p.ranges * batch * plane))
    in
    (* Range 0 from y, in the first plane where it is not y: y copied there in
       the tiles' order. *)
    (match planes with
    | Some at when first ->
        let pz = plane / parts and pr = p.cols_out / parts in
        let strides = if exchanged then [| pz; 1; pr |] else [| pz; pr; 1 |] in
        let l =
          L.v ~strides:(Iarray.of_array strides)
            (Iarray.of_array [| batch; rows; cols |])
        in
        copy b ~src:y ~dst:(work ~at y.mdt l)
    | _ -> ());
    Option.iter
      (fun (va, bv, _) ->
        let y', ystr =
          match planes with
          | Some at ->
              ((Plan.workspace b, at), (p.cols_out, 1, plane, batch * plane))
          | None when complex -> ((m_slot b y, m_at y), (2 * sr, 1, 2 * sz, 0))
          | None -> ((m_slot b y, m_at y), (sr, sc, sz, 0))
        in
        let fit = fit e p in
        let routine =
          (Bool.to_int first lsl K.codes_first)
          lor (tri lsl K.codes_tri)
          lor (Bool.to_int (pair > 0) lsl K.codes_pair)
        in
        (* A product of no routine option takes the plain tiles. *)
        let kernel =
          if complex then complex_kernel ~nk p.tile fit
          else if o.neg then negated ~nk p.tile fit
          else if routine <> 0 then routined ~nk p.tile fit
          else kernel F32 ~nk p.tile fit
        in
        let shared = shared ~nk F32 p.tile in
        let flips = if complex then flips o else 0 in
        Array.iter
          (tile_launch b e p ~kernel ~shared ~a:va ~bv ~y:y' ~ystr ~narrow:false
             ~kinds:0 ~dtypes:0 ~routine ~flips)
          p.launches)
      operands;
    if not direct then begin
      let site =
        {
          fslot = m_slot b y;
          fat = m_at y;
          code;
          s = (sz, sr, sc);
          pos = 0;
          phase = 0;
        }
      in
      let planes = Option.map (fun at -> (Plan.workspace b, at)) planes in
      if
        not
          (finish b ~allowed:(Plan.allowed ()) ~acc ~tri:(tri lor pair) ~planes
             ~ranges:(if k > 0 then p.ranges else 0)
             ~plane:(batch * plane) ~rows:p.rows_out ~total:(batch * p.rows_out)
             ~cols:p.cols_out
             ~init:(if init && not first then Some site else None)
             ~dst:site ())
      then decline ()
    end;
    Plan.release b mark
  end
