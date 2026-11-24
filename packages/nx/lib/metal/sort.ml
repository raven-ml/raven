(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts and numberings: the path of a call, its workspace, its launches' words
   and their estimates (kernels.h, Sorts and numberings).

   sort: n <= cap --> sort_block keep <= cap / 4 --> zero, sort_round a digit,
   sort_gather, sort_block of the candidates otherwise --> zero, sort_hist, then
   a key byte at a time sort_count (where a slice has several parts) and
   sort_scatter unique: zero; a word sort (zero, sort_hist, sort_count and
   sort_scatter a byte) a word from the last; unique_count, unique_heads,
   unique_rank, unique_ids

   Each kernel's launch takes units, blocks, chunks or parts, one a group; a
   kernel whose units take longer than a launch allows runs as launches of
   ranges of them (units). *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

let ceil_div a b = (a + b - 1) / b
let set = Plan.set
let set64 w i v = Bytes.set_int64_le w (8 * i) v
let line = Plan.line
let align8 x = (x + 7) land lnot 7

(* Bounds for tests: the most parts a pass cuts a slice into, and the bits from
   which a slice's positions and counts take 64 bits. *)
let parts_bound = Atomic.make K.sort_parts
let wide_bound = Atomic.make 32

let bound_sorts ~parts ~wide =
  if parts < 1 || parts > K.sort_parts then
    invalid_arg "Nx_metal.bound_sorts: parts";
  if wide < 1 || wide > 32 then invalid_arg "Nx_metal.bound_sorts: wide";
  Atomic.set parts_bound parts;
  Atomic.set wide_bound wide;
  Plan.renew ()

type outer = Estimate | Strided | Transposed

(* Bounds for tests: a strided slice's plan, and the least bytes of a select's
   positions. *)
let outer_bound = Atomic.make Estimate
let positions_bound = Atomic.make 1

let bound_outer o =
  Atomic.set outer_bound o;
  Plan.renew ()

let bound_positions p =
  if p < 1 || p > 8 then invalid_arg "Nx_metal.Sort.bound_positions";
  Atomic.set positions_bound p;
  Plan.renew ()

(* A byte [at] into the buffer of slot [slot]. *)
type place = { slot : int; at : int }

let read word p = { Plan.word; slot = p.slot; write = false }
let write word p = { Plan.word; slot = p.slot; write = true }

(* Weights

   A launch's work in float32 operations an item, and its chain in links. A
   kernel's work weight is 1.5 times the most time an item its launches took on
   the M1 Max, at 2^20 to 2^24 elements of random, few and equal keys, counted
   as work at the Max's float32 rate (32 cores of 128 lanes at 1.296 GHz). The
   M1's 7 cores take at most 4.64 times as long, so the work alone bounds the
   M1; reads and writes are charged on top. *)

(* An element's load from x and its key, its slice's offsets aside. *)
let w_load = 64

(* sort_block's pass, an element: its digit's rank, and each 4-byte word of it
   moved through the exchange. Measured 882 a pass at 4 bytes, 802 at 8 and 1188
   at 16. *)
let w_block_rank = 1180
let w_block_move = 155

(* sort_scatter's item: its digit's rank, and each 4-byte word of it and its
   position moved through the exchange and stored. Measured 2115 at 4 bytes,
   2971 at 8 and 4547 at 16, each in a sort's last pass, which stores values and
   8-byte positions. *)
let w_scatter_rank = 540
let w_scatter_move = 1290

(* A key byte's ballots and add, in sort_hist. Measured 691 an element of 4 key
   bytes, 927 of 8 and 1678 of 16. *)
let w_hist = 250

(* A digit's ballots and add, in sort_count. Measured 287. *)
let w_count = 430

(* A select's digits against the prefix, then the digit's ballots and add, in
   sort_round; the comparison and the slot, in sort_gather. Measured 325 and
   248, at keys all equal. *)
let w_round = 500
let w_gather = 380

(* A numbering's row: its words against the row before it, and its writes. Its
   reads, which go through the sort's order, take its time and are charged as
   [gather_lines] lines a word. *)
let w_unique = 256

(* unique_rank's bit word: loaded and counted before its part's scan, and again
   after it. Its chain is its part's first bit, its words and their second
   reads, and the scan. *)
let unique_rank_ops = 128

(* A pass's barriers and ranks in one group, and a combine's 64-bit scan of 256
   counts, 16 barriers. *)
let rank_links = 256
let combine_links = 64

(* The lines a read through an order is charged: its line lies anywhere. The M1
   Max reads such lines at 165 GB/s, 0.41 of its 400 GB/s, where sort_count
   reads a column through a numbering's order. The M1 at that share reads them
   at 28 GB/s, about half [Plan.bandwidth], so a line counts twice, and with the
   1.5 margin three times. *)
let gather_lines = 3

(* Lines

   The lines of a read or write of [n] elements of [bits] bits, [stride]
   elements apart: one where they all lie at one place, one an element where
   neighbours share none, and the span's lines and one at each end otherwise. *)
let lines ~n ~bits ~stride =
  if n <= 0 then 0
  else if stride = 0 then 1
  else
    let step = abs stride * bits in
    if step >= 8 * line then n
    else ceil_div (((n - 1) * step) + bits) (8 * line) + 1

let run_lines bytes = if bytes <= 0 then 0 else ceil_div bytes line + 1

(* Keys *)

(* How an element's bits become its key: its format, its width, its key's bytes,
   a float's greatest magnitude not NaN, and its kernels' width, 4 for elements
   of at most 32 bits, 8 or 16. *)
type format = {
  kind : int;
  ebits : int;
  kbytes : int;
  nan_abs : int64;
  width : int;
}

let format (type v s) (dt : (v, s) D.t) =
  let width ebits = if ebits <= 32 then 4 else ebits / 8 in
  let make kind ebits nan_abs =
    {
      kind;
      ebits;
      kbytes = Int.max 1 (ebits / 8);
      nan_abs;
      width = width ebits;
    }
  in
  let float = make K.sort_float and int k e = make k e (-1L) in
  match dt with
  | D.Float64 -> float 64 0x7FF0000000000000L
  | D.Float32 -> float 32 0x7F800000L
  | D.Float16 -> float 16 0x7C00L
  | D.Bfloat16 -> float 16 0x7F80L
  | D.Float8_e4m3fn -> float 8 0x7EL
  | D.Float8_e5m2 -> float 8 0x7CL
  | D.Float4_e2m1fn -> float 4 0x7L
  | D.Int64 -> int K.sort_signed 64
  | D.Uint64 -> int K.sort_unsigned 64
  | D.Int32 -> int K.sort_signed 32
  | D.Uint32 -> int K.sort_unsigned 32
  | D.Int16 -> int K.sort_signed 16
  | D.Uint16 -> int K.sort_unsigned 16
  | D.Int8 -> int K.sort_signed 8
  | D.Uint8 -> int K.sort_unsigned 8
  | D.Int4 -> int K.sort_signed 4
  | D.Uint4 -> int K.sort_unsigned 4
  | D.Complex128 -> make K.sort_complex 128 0x7FF0000000000000L
  | D.Complex64 -> make K.sort_complex 64 0x7F800000L
  | D.Bool -> int K.sort_bool 8
  | D.Bit -> int K.sort_unsigned 1

(* The bytes of a pass's buffer element: the element's, one for a sub-byte
   one. *)
let ebytes f = Int.max 1 (f.ebits / 8)

(* Writes [f]'s words: a real element's NaN key and the flips of its sign, and
   the bits every key is xored with. *)
let set_key w f ~descending =
  let bits = Int.min f.ebits 64 in
  let mask = if bits >= 64 then -1L else Int64.(pred (shift_left 1L bits)) in
  let sign = Int64.shift_left 1L (bits - 1) in
  let float = f.kind = K.sort_float and signed = f.kind = K.sort_signed in
  let nan_abs = if float || f.kind = K.sort_complex then f.nan_abs else -1L in
  let dmask_lo =
    if not descending then 0L
    else if f.kbytes >= 8 then -1L
    else Int64.(pred (shift_left 1L (8 * f.kbytes)))
  in
  set w K.sort_kind f.kind;
  set w K.sort_ebits f.ebits;
  set w K.sort_kbytes f.kbytes;
  set64 w K.sort_nan_abs nan_abs;
  set64 w K.sort_kmask mask;
  set64 w K.sort_flip (if float || signed then sign else 0L);
  set64 w K.sort_flip_signed
    (if float then mask else if signed then sign else 0L);
  set64 w K.sort_dmask_lo dmask_lo;
  set64 w K.sort_dmask_hi (if descending && f.kbytes = 16 then -1L else 0L)

(* Kernels by width *)

type kernels = {
  block : K.t;
  hist : K.t;
  count : K.t;
  scatter : K.t;
  round : K.t;
  gather : K.t;
  block_items : int;
  tile_items : int;
}

let kernels f ~wide =
  match f.width with
  | 4 ->
      {
        block = K.Sort_block_4;
        hist = K.Sort_hist_4;
        count = K.Sort_count_4;
        scatter = (if wide then K.Sort_scatter_wide_4 else K.Sort_scatter_4);
        round = K.Sort_round_4;
        gather = K.Sort_gather_4;
        block_items = K.sort_block_items;
        tile_items = K.sort_tile_items4;
      }
  | 8 ->
      {
        block = K.Sort_block_8;
        hist = K.Sort_hist_8;
        count = K.Sort_count_8;
        scatter = (if wide then K.Sort_scatter_wide_8 else K.Sort_scatter_8);
        round = K.Sort_round_8;
        gather = K.Sort_gather_8;
        block_items = K.sort_block_items;
        tile_items = K.sort_tile_items8;
      }
  | _ ->
      {
        block = K.Sort_block_16;
        hist = K.Sort_hist_16;
        count = K.Sort_count_16;
        scatter = (if wide then K.Sort_scatter_wide_16 else K.Sort_scatter_16);
        round = K.Sort_round_16;
        gather = K.Sort_gather_16;
        block_items = K.sort_block_items16;
        tile_items = K.sort_tile_items16;
      }

(* The longest slice sort_block sorts. *)
let cap ks = K.sort_threads * ks.block_items

(* Threadgroup memory a group of each kernel takes (sort.metal). *)
let block_shared f ks = (cap ks * 6) + K.sort_rank_bytes + (8 * (f.width / 4))
let hist_shared f = (4 * 256 * f.kbytes) + (8 * K.sort_threads) + 8
let count_shared = (4 * 256 * (K.sort_threads / 32)) + 8
let round_shared = (4 * 256) + 32 + (8 * K.sort_threads) + 8
let gather_shared = 32

(* A scatter's fields an item, 32-bit words of its element and position, and
   those its exchange holds at once. *)
let fields f ~wide = (f.width / 4) + if wide then 2 else 1

let held f ks ~wide =
  let tile = K.sort_threads * ks.tile_items in
  Int.min (fields f ~wide)
    ((Plan.group_memory - K.sort_scatter_rest) / (tile * 4))

let scatter_shared f ks ~wide =
  (K.sort_threads * ks.tile_items * 4 * held f ks ~wide) + K.sort_scatter_rest

(* The groups of [k] a core holds at once: {!Plan.holds}, at most its registers'
   share, [holds k] threads, and at least one. *)
let resident ~holds k shared =
  let threads = K.threads k in
  Int.max 1 (Int.min (Plan.holds ~threads shared) (holds k / threads))

(* Launches

   A call's launches, and their estimated seconds together, by which a call
   chooses between plans; [split] once a kernel's units take several launches,
   so that the call's launches grow with its extents. *)

exception Too_long

type call = {
  b : Plan.builder;
  holds : K.t -> int;
  mutable seconds : float;
  mutable split : bool;
}

let call ~holds = { b = Plan.make (); holds; seconds = 0.; split = false }

(* Adds the launches of [k] over [total] units, as many a launch as its estimate
   allows within {!Plan.allowed}: [cost m] a launch of [m] units, [words ~first
   ~stop] its words for units [first] to [stop - 1], and [refs ~first ~stop] its
   refs. Raises [Too_long] where one unit takes longer than a launch allows. *)
let units c k ~total ~shared ~cost ~words ~refs =
  let threads = K.threads k and resident = resident ~holds:c.holds k shared in
  let seconds m = Plan.seconds ~threads ~groups:(m, 1, 1) ~resident (cost m) in
  let allowed = Plan.allowed () in
  if seconds 1 > allowed then raise Too_long;
  let total' = Int.min total Plan.most_groups in
  let most =
    if seconds total' <= allowed then total'
    else
      let rec go lo hi =
        if hi - lo <= 1 then lo
        else
          let mid = lo + ((hi - lo) / 2) in
          if seconds mid <= allowed then go mid hi else go lo mid
      in
      go 1 total'
  in
  if most < total then c.split <- true;
  let first = ref 0 in
  while !first < total do
    let m = Int.min most (total - !first) in
    let stop = !first + m in
    c.seconds <- c.seconds +. seconds m;
    Plan.launch c.b k ~groups:(m, 1, 1) ~shared ~resident ~cost:(cost m)
      (words ~first:!first ~stop)
      (refs ~first:!first ~stop);
    first := stop
  done

(* Zeroes [bytes] of the workspace from [p]. *)
let zero c p bytes =
  if bytes > 0 then begin
    c.seconds <- c.seconds +. (float (bytes + (2 * line)) /. Plan.bandwidth);
    Plan.zero c.b ~slot:p.slot ~at:p.at (4 * ceil_div bytes 4)
  end

(* Sorts *)

(* A sort of [count] slices of [n] elements of x, along which x, the values and
   the positions step [sx], [sv] and [sp] elements from [off_x], [off_v] and
   [off_p], each of the [kept] axes an extent and those three strides, keeping
   [keep] of each slice. The first pass reads x through [order] where given, and
   the values are stored where given, their first byte [skew] bits into its
   32-bit word. Counts and positions take 64 bits where [wide], their low words
   carrying at bit [carry]; the slices' offsets take 64-bit divisions where
   [huge]. *)
type sort = {
  f : format;
  descending : bool;
  n : int;
  keep : int;
  count : int;
  sx : int;
  sv : int;
  sp : int;
  off_x : int;
  off_v : int;
  off_p : int;
  kept : (int * int * int * int) array;
  x : place;
  values : place option;
  positions : place;
  order : place option;
  skew : int;
  wide : bool;
  carry : int;
  huge : bool;
  axes : place option;
}

let flags t =
  (if Option.is_some t.values then K.sort_store else 0)
  lor (if Option.is_some t.order then K.sort_ordered else 0)
  lor (if t.wide then K.sort_wide else 0)
  lor if t.huge then K.sort_huge else 0

(* The kept axes' words, an axis's extent, divisor and three strides. *)
let axis_words kept =
  let w = Bytes.make (8 * K.sort_axis_words * Array.length kept) '\000' in
  Array.iteri
    (fun i (e, sx, sv, sp) ->
      let at = K.sort_axis_words * i in
      set w (at + K.sort_axis_extent) e;
      set w (at + K.sort_axis_div) (Plan.divisor e);
      set w (at + K.sort_axis_sx) sx;
      set w (at + K.sort_axis_sv) sv;
      set w (at + K.sort_axis_sp) sp)
    kept;
  w

let bigbytes s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set b i (Char.code c)) s;
  b

(* The kept axes' constant, past what a launch's words hold. *)
let axes_constant b kept =
  if Array.length kept <= K.sort_holds then None
  else
    let t = Bytes.unsafe_to_string (axis_words kept) in
    let slot =
      Plan.constant b ~key:("sort " ^ t) ~length:(String.length t) (fun () ->
          bigbytes t)
    in
    Some { slot; at = 0 }

(* A launch's words of [k] for [t]: its key, its slices and its flags, [extra]
   more. *)
let words k t ~extra =
  let w = Plan.words k in
  set_key w t.f ~descending:t.descending;
  set w K.sort_x t.x.at;
  Option.iter (fun v -> set w K.sort_values v.at) t.values;
  set w K.sort_positions t.positions.at;
  Option.iter (fun o -> set w K.sort_order o.at) t.order;
  set w K.sort_keep t.keep;
  set w K.sort_flags (flags t lor extra);
  set w K.sort_carry t.carry;
  set w K.sort_skew t.skew;
  set w K.sort_n t.n;
  set w K.sort_n_div (Plan.divisor t.n);
  set w K.sort_count t.count;
  set w K.sort_sx t.sx;
  set w K.sort_sv t.sv;
  set w K.sort_sp t.sp;
  set w K.sort_off_x t.off_x;
  set w K.sort_off_v t.off_v;
  set w K.sort_off_p t.off_p;
  set w K.sort_kept (Array.length t.kept);
  if Array.length t.kept <= K.sort_holds then
    Bytes.blit (axis_words t.kept) 0 w (8 * K.sort_held)
      (8 * K.sort_axis_words * Array.length t.kept);
  w

(* [w] for units [first] to [stop - 1]. *)
let ranged w ~first ~stop =
  let w = Bytes.copy w in
  set w K.sort_first first;
  set w K.sort_end stop;
  w

(* The refs every launch of [t] that reads x takes. *)
let x_refs t =
  read K.sort_x t.x
  :: (match t.order with Some o -> [ read K.sort_order o ] | None -> [])
  @ match t.axes with Some a -> [ read K.sort_axes a ] | None -> []

(* The refs of a launch that writes [t]'s values and positions. *)
let out_refs t =
  write K.sort_positions t.positions
  :: (match t.values with Some v -> [ write K.sort_values v ] | None -> [])

(* The lines of x's elements [n] of a slice: [gather_lines] an element where
   they are read through an order, which scatters them. *)
let x_lines t n =
  if Option.is_some t.order then gather_lines * n
  else lines ~n ~bits:t.f.ebits ~stride:t.sx

(* The lines of [n] kept values and positions of a slice. *)
let out_lines t n =
  let n = Int.min n t.keep in
  (if Option.is_some t.values then lines ~n ~bits:t.f.ebits ~stride:t.sv else 0)
  + lines ~n ~bits:64 ~stride:t.sp

(* A select's candidates: each slice's elements, then its 64-bit positions. *)
type select = { sel : place; cand : place; cand_p : place; pbytes : int }

(* The block sort of [t]'s slices, [per] a group, or with [select] each slice's
   candidates. *)
let block ?select c t ks =
  let cap = cap ks in
  let per, blocks, extra, pb =
    match select with
    | None ->
        let per = Int.min 256 (cap / t.n) in
        (per, ceil_div t.count per, 0, 0)
    | Some s -> (1, t.count, K.sort_candidates, s.pbytes)
  in
  let w = words ks.block t ~extra in
  set w K.sort_per per;
  set w K.sort_pbytes pb;
  let refs =
    match select with
    | None -> x_refs t @ out_refs t
    | Some s ->
        set w K.sort_hist s.sel.at;
        set w K.sort_cand s.cand.at;
        set w K.sort_cand_p s.cand_p.at;
        (match t.axes with Some a -> [ read K.sort_axes a ] | None -> [])
        @ [
            read K.sort_hist s.sel;
            read K.sort_cand s.cand;
            read K.sort_cand_p s.cand_p;
          ]
        @ out_refs t
  in
  let passes = pb + t.f.kbytes + if per > 1 then 1 else 0 in
  let phases = t.f.width / 4 in
  let eb = ebytes t.f in
  let cost m =
    let slices = Int.min (per * m) t.count in
    let reads =
      match select with
      | None -> slices * x_lines t t.n
      | Some _ -> m * (run_lines (cap * eb) + ((pb + 1) * cap))
    in
    let n = match select with None -> t.n | Some _ -> t.keep in
    {
      Plan.bytes = (reads + (slices * out_lines t n)) * line;
      ops =
        m * cap * (w_load + (passes * (w_block_rank + (phases * w_block_move))));
      chain = (2 * Plan.read_links) + (passes * rank_links);
    }
  in
  units c ks.block ~total:blocks ~shared:(block_shared t.f ks) ~cost
    ~words:(ranged w) ~refs:(fun ~first:_ ~stop:_ -> Array.of_list refs)

(* The bytes of position a select's digits take: those of [n - 1]. *)
let pbytes n =
  let rec go b = if b < 8 && (n - 1) lsr (8 * b) <> 0 then go (b + 1) else b in
  go (Atomic.get positions_bound)

(* The ticket words of a launch of units [first] to [stop - 1], [units] a slice:
   its slices'. *)
let ticket_words ~units ~first ~stop =
  ((stop - 1) / units) - (first / units) + 1

(* A select of each slice's [keep] first elements, then their block sort. *)
let select c ws t ks =
  let b = c.b in
  let cap = cap ks and pb = pbytes t.n and eb = ebytes t.f in
  let state =
    4 * if t.wide then K.sort_select_words_wide else K.sort_select_words
  in
  let sel = { slot = ws; at = Plan.take b (t.count * state) } in
  let cand = { slot = ws; at = Plan.take b (t.count * cap * eb) } in
  let cand_p = { slot = ws; at = Plan.take b (t.count * cap * 8) } in
  zero c sel (t.count * state);
  let chunks = ceil_div t.n K.sort_chunk in
  let chunk = Int.min t.n K.sort_chunk in
  let tickets = Plan.tickets_slot b in
  let bins = 256 * if t.wide then 8 else 4 in
  let launch k ~pass ~shared ~ops ~bytes ~chain refs =
    let w = words k t ~extra:0 in
    set w K.sort_hist sel.at;
    set w K.sort_cand cand.at;
    set w K.sort_cand_p cand_p.at;
    set w K.sort_pass pass;
    set w K.sort_units chunks;
    set w K.sort_units_div (Plan.divisor chunks);
    set w K.sort_pbytes pb;
    let cost m = { Plan.bytes = m * bytes; ops = m * ops; chain } in
    units c k ~total:(t.count * chunks) ~shared ~cost
      ~words:(fun ~first ~stop ->
        let w = ranged w ~first ~stop in
        set w K.sort_tickets
          (Plan.tickets b (ticket_words ~units:chunks ~first ~stop));
        w)
      ~refs:(fun ~first:_ ~stop:_ -> Array.of_list (x_refs t @ refs))
  in
  let reads = x_lines t chunk * line in
  for r = 0 to t.f.kbytes + pb - 1 do
    launch ks.round ~pass:r ~shared:round_shared ~ops:(K.sort_chunk * w_round)
      ~bytes:(reads + bins)
      ~chain:(Plan.read_links + combine_links)
      [
        write K.sort_hist sel;
        { Plan.word = K.sort_tickets; slot = tickets; write = true };
      ]
  done;
  launch ks.gather ~pass:0 ~shared:gather_shared ~ops:(K.sort_chunk * w_gather)
    ~bytes:(reads + (run_lines (cap * (eb + 8)) * line))
    ~chain:Plan.read_links
    [
      write K.sort_hist sel; write K.sort_cand cand; write K.sort_cand_p cand_p;
    ];
  block c t ks ~select:{ sel; cand; cand_p; pbytes = pb }

(* The parts a pass cuts a slice of [tiles] tiles into, of [span] tiles each,
   with parts enough for a few groups a core across the slices, at most the
   bound's a slice. *)
let parts ~count ~tiles =
  let most = Int.max 1 (ceil_div (Atomic.get parts_bound) count) in
  let span = ceil_div tiles (Int.min tiles most) in
  (ceil_div tiles span, span)

(* A pass's buffers: elements, then positions. *)
type buffer = { e : place; p : place }

(* A radix sort of [t], which takes its workspace from [b]'s. [reuse] holds the
   positions' place where they are runs of each slice's [n] in slice order, of 8
   bytes an element, so that a pass's buffer 0 may wait there. *)
let radix c ws t ks ~reuse =
  let b = c.b in
  let kb = t.f.kbytes and eb = ebytes t.f in
  let tile = K.sort_threads * ks.tile_items in
  let tiles = ceil_div t.n tile in
  let parts, span = parts ~count:t.count ~tiles in
  let mark = Plan.mark b in
  let bin = 4 * if t.wide then K.sort_bin_wide else K.sort_bin in
  let entry = if t.wide then 8 else 4 in
  let pbuf = if t.wide then 8 else 4 in
  let hist = { slot = ws; at = Plan.take b (t.count * kb * 256 * bin) } in
  zero c hist (t.count * kb * 256 * bin);
  let meta = { slot = ws; at = Plan.take b (t.count * 4) } in
  let counts =
    if parts = 1 then None
    else Some { slot = ws; at = Plan.take b (t.count * parts * 256 * entry) }
  in
  (* A slice's run of a pass buffer: its elements, then its positions from the
     first multiple of their width past them, [stride] bytes from one slice's
     run to the next. *)
  let pos_at = ceil_div (t.n * eb) pbuf * pbuf in
  let stride = align8 (pos_at + (t.n * pbuf)) in
  let take () =
    let e = { slot = ws; at = Plan.take b (t.count * stride) } in
    { e; p = { e with at = e.at + pos_at } }
  in
  let buf1 = if kb < 2 then None else Some (take ()) in
  (* Buffer 0 waits in the positions where each slice's run lies in its own
     positions' 8n bytes: slices whose passes differ end at different passes,
     and one's final pass writes its positions while another's still reads its
     buffer 0. *)
  let own = stride = 8 * t.n || (t.count = 1 && stride <= 8 * t.n) in
  let buf0 =
    if kb < 3 then None
    else
      match reuse with
      | Some p when own -> Some { e = p; p = { p with at = p.at + pos_at } }
      | Some _ | None -> Some (take ())
  in
  let tickets = Plan.tickets_slot b in
  let ticket_ref =
    { Plan.word = K.sort_tickets; slot = tickets; write = true }
  in

  (* sort_hist *)
  let chunks = ceil_div t.n K.sort_chunk in
  let h = words ks.hist t ~extra:0 in
  set h K.sort_hist hist.at;
  set h K.sort_meta meta.at;
  set h K.sort_units chunks;
  set h K.sort_units_div (Plan.divisor chunks);
  let chunk = Int.min t.n K.sort_chunk in
  let hcost m =
    {
      Plan.bytes = m * ((x_lines t chunk * line) + (kb * 256 * bin));
      ops = m * K.sort_chunk * (w_load + (kb * w_hist));
      chain = Plan.read_links + (kb * combine_links);
    }
  in
  units c ks.hist ~total:(t.count * chunks) ~shared:(hist_shared t.f)
    ~cost:hcost
    ~words:(fun ~first ~stop ->
      let w = ranged h ~first ~stop in
      set w K.sort_tickets
        (Plan.tickets b (ticket_words ~units:chunks ~first ~stop));
      w)
    ~refs:(fun ~first:_ ~stop:_ ->
      Array.of_list
        (x_refs t
        @ [ write K.sort_hist hist; write K.sort_meta meta; ticket_ref ]));

  (* A pass's words: its buffers, counts and parts. *)
  let w = words ks.scatter t ~extra:0 in
  set w K.sort_hist hist.at;
  set w K.sort_meta meta.at;
  Option.iter (fun p -> set w K.sort_counts p.at) counts;
  let bufs word_e word_p = function
    | None -> []
    | Some buf ->
        set w word_e buf.e.at;
        set w word_p buf.p.at;
        [ (word_e, buf.e); (word_p, buf.p) ]
  in
  let buffers = bufs K.sort_e0 K.sort_p0 buf0 @ bufs K.sort_e1 K.sort_p1 buf1 in
  set w K.sort_tiles tiles;
  set w K.sort_span span;
  set w K.sort_stride stride;
  set w K.sort_units parts;
  set w K.sort_units_div (Plan.divisor parts);
  let items = span * tile in
  let fields = fields t.f ~wide:t.wide in
  let reads ~elements = Int.max (x_lines t items * line) (items * elements) in
  let ccost m =
    let slices = ceil_div m parts + 1 in
    {
      Plan.bytes =
        (m * (reads ~elements:eb + (256 * entry)))
        + (slices * parts * 256 * entry * 2);
      ops = m * ((items * w_count) + (256 * 8));
      chain =
        Plan.read_links + (ceil_div parts 8 * Plan.read_links) + parts + 16;
    }
  in
  let edges = 2 * 256 * line * 2 in
  let scost m =
    let writes =
      Int.max
        ((items * (eb + pbuf)) + (span * edges))
        ((out_lines t items * line) + (span * edges))
    in
    {
      Plan.bytes = m * (reads ~elements:(eb + pbuf) + writes);
      ops = m * items * (w_load + w_scatter_rank + (fields * w_scatter_move));
      chain = Plan.read_links + (span * rank_links);
    }
  in
  let common = x_refs t @ [ read K.sort_hist hist; read K.sort_meta meta ] in
  for p = 0 to kb - 1 do
    let w = Bytes.copy w in
    set w K.sort_pass p;
    Option.iter
      (fun cs ->
        units c ks.count ~total:(t.count * parts) ~shared:count_shared
          ~cost:ccost
          ~words:(fun ~first ~stop ->
            let w = ranged w ~first ~stop in
            set w K.sort_tickets
              (Plan.tickets b (ticket_words ~units:parts ~first ~stop));
            w)
          ~refs:(fun ~first:_ ~stop:_ ->
            Array.of_list
              ((write K.sort_counts cs :: ticket_ref :: common)
              @ List.map (fun (word, p) -> read word p) buffers)))
      counts;
    units c ks.scatter ~total:(t.count * parts)
      ~shared:(scatter_shared t.f ks ~wide:t.wide) ~cost:scost ~words:(ranged w)
      ~refs:(fun ~first:_ ~stop:_ ->
        Array.of_list
          (out_refs t
          @ (match counts with
            | Some cs -> [ read K.sort_counts cs ]
            | None -> [])
          @ common
          @ List.map (fun (word, p) -> write word p) buffers))
  done;
  Plan.release b mark

(* The kept axes of a sort along [axis] of [x] into [values] and [positions]:
   each axis of more than one element as its extent and the three strides, an
   axis merged into the one before where all three step along both as one. *)
let kept ~axis x v p =
  let rank = L.rank x in
  let rows = ref [] in
  for i = 0 to rank - 1 do
    let e = L.dim x i in
    if i <> axis && e > 1 then
      let sx = L.stride x i and sv = L.stride v i and sp = L.stride p i in
      match !rows with
      | (e0, sx0, sv0, sp0) :: rest
        when sx0 = sx * e && sv0 = sv * e && sp0 = sp * e ->
          rows := (e0 * e, sx, sv, sp) :: rest
      | rs -> rows := (e, sx, sv, sp) :: rs
  done;
  Array.of_list (List.rev !rows)

(* Finishes [c]'s call, kept where no kernel split and [kept] holds. *)
let finish c ~kept =
  let most = if c.split || not kept then 0 else max_int in
  Ok (Some (Plan.finish ~most c.b))

let huge ~count kept =
  count >= 1 lsl 32 || Array.exists (fun (e, _, _, _) -> e >= 1 lsl 32) kept

(* Copies x's slices along [axis], [o]'s elements at [lx], contiguous into the
   workspace, in C order of x's other axes then [axis]; is the copy's byte in
   the workspace and x's layout there. *)
let transpose c (o : Operands.operand) lx axis =
  let b = c.b in
  let rank = L.rank lx in
  let order =
    Array.of_list (List.filter (( <> ) axis) (List.init rank Fun.id) @ [ axis ])
  in
  let shape = Iarray.of_array (Array.map (L.dim lx) order) in
  let strides = Iarray.of_array (Array.map (L.stride lx) order) in
  let (D.Any dt) = o.dtype in
  let src =
    Moves.site_at ~slot:(Plan.slot b o.buffer) ~base:0 ~residue:o.residue
      o.dtype
      (L.v ~offset:(L.offset lx) ~strides shape)
  in
  let n = L.numel lx in
  let at = Plan.take b (n * (D.bits dt / 8)) in
  let dst =
    Moves.site_at ~slot:(Plan.workspace b) ~base:at ~residue:0 o.dtype
      (L.contiguous shape)
  in
  let seconds = Moves.launch b ~n ~dst src in
  c.seconds <- c.seconds +. seconds;
  if seconds > Plan.allowed () then c.split <- true;
  let packed = L.strides (L.contiguous shape) in
  let strides = Array.make rank 0 in
  Array.iteri (fun j a -> strides.(a) <- Iarray.get packed j) order;
  (at, L.v ~strides:(Iarray.of_array strides) (L.shape lx))

let of_facts ~holds ~axis ~descending ~shares ~(values : Operands.operand)
    ~(positions : Operands.operand) (x : Operands.operand) =
  let lx = x.layout and lv = values.layout and lp = positions.layout in
  let n = L.dim lx axis and keep = L.dim lv axis in
  let count = L.numel lx / Int.max 1 n in
  let f =
    let (D.Any dt) = x.dtype in
    format dt
  in
  let k = Atomic.get wide_bound in
  let wide = n >= 1 lsl k in
  let ks = kernels f ~wide in
  let cap = cap ks in
  let radix_path = n > cap && keep > cap / 4 in
  (* A copy of x's slices contiguous first: forced, or where no two elements of
     x along any axis share a line and its estimate is less. Where neighbouring
     slices share x's lines, their groups read each line about once between
     them, which the estimate's line an element overcharges: along the first
     axis of a 16384 x 1024 float32 matrix, the radix sort read in place takes
     20.6 ms on the M1 Max and the copy then sort 25.5 ms, though the estimate
     is less with the copy. *)
  let sx = L.stride lx axis in
  let copies = f.ebits >= 8 && n > 1 && abs sx <> 1 in
  let apart i = L.dim lx i = 1 || abs (L.stride lx i) * f.ebits >= 8 * line in
  let all_apart = List.for_all apart (List.init (L.rank lx) Fun.id) in
  let plans =
    match Atomic.get outer_bound with
    | Strided -> [ false ]
    | Transposed -> [ copies ]
    | Estimate when copies && all_apart -> [ false; true ]
    | Estimate -> [ false ]
  in
  let plan copied =
    let c = call ~holds in
    let b = c.b in
    let place (o : Operands.operand) =
      { slot = Plan.slot b o.buffer; at = 0 }
    in
    let xp, lx =
      if not copied then (place x, lx)
      else
        let at, lx = transpose c x lx axis in
        ({ slot = Plan.workspace b; at }, lx)
    in
    let kept = kept ~axis lx lv lp in
    let t =
      {
        f;
        descending;
        n;
        keep;
        count;
        sx = L.stride lx axis;
        sv = L.stride lv axis;
        sp = L.stride lp axis;
        off_x = L.offset lx;
        off_v = L.offset lv;
        off_p = L.offset lp;
        kept;
        x = xp;
        (* Sub-byte values are stored by values.h's store_bits, which holds
           where three facts do: its word is aligned, as the values are
           addressed from their buffer's first byte [skew] bits into the word
           below it; no launch stores a whole word or vector of the values, so
           none writes another array's bytes; and no launch that stores the
           values reads them, nor reads x where they share a byte (the decline
           below). *)
        values = Some (place values);
        positions = place positions;
        order = None;
        skew = 8 * values.skew;
        wide;
        carry = k;
        huge = huge ~count kept;
        axes = axes_constant b kept;
      }
    in
    match
      if n <= cap then block c t ks
      else
        let ws = Plan.workspace b in
        if not radix_path then select c ws t ks
        else
          (* Each slice's positions a run of n, in slice order. *)
          let runs =
            count = 1
            || match kept with [| (_, _, _, ps) |] -> ps = n | _ -> false
          in
          let reuse =
            if keep = n && t.sp = 1 && runs then
              Some { t.positions with at = 8 * t.off_p }
            else None
          in
          radix c ws t ks ~reuse
    with
    | () -> Some c
    | exception Too_long -> None
  in
  let least best c =
    match best with Some b when b.seconds <= c.seconds -> best | _ -> Some c
  in
  if n = 0 || keep = 0 || count = 0 then Ok None
  else if shares && (n <= cap || radix_path) then Error A.Declined
  else
    match List.fold_left least None (List.filter_map plan plans) with
    | Some c -> finish c ~kept:true
    | None -> Error A.Declined

let sort ~holds ~axis ~descending ~values ~positions x =
  let xs =
    Operands.v
      ~dsts:(Iarray.of_array [| A.Any values; A.Any positions |])
      (Iarray.of_array [| Nx_kernel.Plain x |])
  in
  (* A block sort or a radix pass reads x and stores the values and positions in
     one launch, so where they share a byte, as where the values are x, a store
     would race with another group's read of x. Where they share only a word, no
     launch stores x's bytes and store_bits changes only its own element's bits,
     so a read of x's bytes of the word sees them unchanged. A select's last
     launch reads only its candidates. *)
  let shares =
    let bx = A.buffer x in
    Rig.Buffer.share_byte bx (A.buffer values)
    || Rig.Buffer.share_byte bx (A.buffer positions)
  in
  of_facts ~holds ~axis ~descending ~shares ~values:xs.dsts.(0)
    ~positions:xs.dsts.(1) xs.loads.(0)

(* Numberings *)

(* The most parts a numbering's rows take, NX_UNIQUE_TILE rows or more each. *)
let unique_parts = 256

let unique_shared = function
  | K.Unique_count -> (8 * K.sort_threads) + 8
  | K.Unique_heads -> (8 * K.sort_threads) + 8 + (4 * (K.sort_threads / 32))
  | K.Unique_rank -> 4 * (K.sort_threads / 32)
  | _ -> 0

let unique ~holds ~ids ~first ~counts ~count keys =
  let lk = A.layout keys in
  let n = L.dim lk 0 and w = L.dim lk 1 in
  let m = L.dim (A.layout first) 0 in
  let s0 = L.stride lk 0 and s1 = L.stride lk 1 in
  let xs =
    Operands.v
      ~dsts:
        (Iarray.of_array
           [| A.Any ids; A.Any first; A.Any counts; A.Any count |])
      (Iarray.of_array [| Nx_kernel.Plain keys |])
  in
  let c = call ~holds in
  let b = c.b in
  let place (o : Operands.operand) =
    { slot = Plan.slot b o.buffer; at = 8 * L.offset o.layout }
  in
  let ids = place xs.dsts.(0) and first = place xs.dsts.(1) in
  let counts = place xs.dsts.(2) and count = place xs.dsts.(3) in
  (* The keys' slot, which a call of no row or no word never reads. *)
  let keys_place () = { slot = Plan.slot b xs.loads.(0).buffer; at = 0 } in
  let u = Plan.words K.Unique_ids in
  set u K.unique_ids ids.at;
  set u K.unique_first first.at;
  set u K.unique_counts counts.at;
  set u K.unique_count count.at;
  set u K.unique_m m;
  set u K.unique_n n;
  let uw word p = { Plan.word; slot = p.slot; write = true } in
  let ur word p = { Plan.word; slot = p.slot; write = false } in
  let outs =
    [
      uw K.unique_ids ids;
      uw K.unique_first first;
      uw K.unique_counts counts;
      uw K.unique_count count;
    ]
  in
  (* Each launch of [k] over [total] units: [per] the cost of one. *)
  let launch k ~total ~(per : Plan.cost) refs =
    let cost m =
      { Plan.bytes = m * per.bytes; ops = m * per.ops; chain = per.chain }
    in
    units c k ~total ~shared:(unique_shared k) ~cost
      ~words:(fun ~first ~stop ->
        let u = Bytes.copy u in
        set u K.unique_first_unit first;
        set u K.unique_end stop;
        u)
      ~refs:(fun ~first:_ ~stop:_ -> Array.of_list refs)
  in
  let ids_launch refs =
    launch K.Unique_ids
      ~total:(Int.max 1 (ceil_div (Int.max n m) K.sort_threads))
      ~per:
        {
          Plan.bytes = K.sort_threads * 6 * line;
          ops = K.sort_threads * w_unique;
          chain = 4 * Plan.read_links;
        }
      (outs @ refs)
  in
  match
    if n = 0 then ids_launch []
    else begin
      let sorts = w > 0 in
      let words = ceil_div n 32 in
      let parts = Int.min unique_parts (ceil_div n K.unique_tile) in
      let span = ceil_div n parts in
      let parts = ceil_div n span in
      let wspan = ceil_div words parts in
      let wparts = ceil_div words wspan in
      let ws = Plan.workspace b in
      let take bytes = { slot = ws; at = Plan.take b bytes } in
      (* The bit words and each part's count of them, zeroed at once. *)
      let bits = take (align8 (words * 4) + (wparts * 8)) in
      let set_bits = { bits with at = bits.at + align8 (words * 4) } in
      zero c bits (align8 (words * 4) + (wparts * 8));
      (* The sorts' order, which no launch reads where no row is sorted. *)
      let order0 = if sorts then take (n * 8) else { slot = ws; at = 0 } in
      let order1 = if sorts && w > 1 then take (n * 8) else order0 in
      let skeys = if sorts && w = 1 then Some (take (n * 8)) else None in
      let gid = take (n * 8) and gfirst = take (n * 8) in
      let gstart = take ((n + 1) * 8) and below = take (words * 8) in
      let heads = take (parts * 8) and k = take 8 in
      let f = format D.Uint64 in
      (* Word [j]'s sort, read through the order of word [j + 1]'s. *)
      let word j =
        let t =
          {
            f;
            descending = false;
            n;
            keep = n;
            count = 1;
            sx = s0;
            sv = 1;
            sp = 1;
            off_x = L.offset lk + (j * s1);
            off_v = 0;
            off_p = 0;
            kept = [||];
            x = keys_place ();
            values = skeys;
            positions = (if j land 1 = 0 then order0 else order1);
            order =
              (if j = w - 1 then None
               else Some (if j land 1 = 0 then order1 else order0));
            skew = 0;
            wide = n >= 1 lsl Atomic.get wide_bound;
            carry = Atomic.get wide_bound;
            huge = false;
            axes = None;
          }
        in
        let ks = kernels f ~wide:t.wide in
        if n <= cap ks then block c t ks else radix c ws t ks ~reuse:None
      in
      if sorts then
        for j = w - 1 downto 0 do
          word j
        done;
      let flags =
        (if sorts then K.unique_by_order else 0)
        lor if sorts && w = 1 then K.unique_by_skeys else 0
      in
      set u K.unique_keys (8 * L.offset lk);
      Option.iter (fun s -> set u K.unique_skeys s.at) skeys;
      set u K.unique_w w;
      set u K.unique_s0 s0;
      set u K.unique_s1 s1;
      set u K.unique_heads heads.at;
      set u K.unique_tickets (Plan.tickets b 1);
      set u K.unique_span span;
      set u K.unique_order order0.at;
      set u K.unique_gid gid.at;
      set u K.unique_gfirst gfirst.at;
      set u K.unique_gstart gstart.at;
      set u K.unique_k k.at;
      set u K.unique_flags flags;
      set u K.unique_bits bits.at;
      set u K.unique_below below.at;
      set u K.unique_set set_bits.at;
      set u K.unique_wspan wspan;
      let tickets =
        {
          Plan.word = K.unique_tickets;
          slot = Plan.tickets_slot b;
          write = true;
        }
      in
      (* The rows' words as differs reads them: sorted keys, or keys through the
         order of the sorts. *)
      let order = if sorts then [ ur K.unique_order order0 ] else [] in
      let key_words =
        match skeys with
        | Some s -> [ ur K.unique_skeys s ]
        | None when w = 0 -> []
        | None -> [ ur K.unique_keys (keys_place ()) ]
      in
      let rows =
        if Option.is_some skeys then key_words else key_words @ order
      in
      (* A row's bytes: its sorted key, or its order and each word of it and of
         the row before, read through the order. *)
      let row =
        if Option.is_some skeys then 16 else 8 + (2 * w * gather_lines * line)
      in
      let per_part ~extra =
        {
          Plan.bytes = span * (row + extra);
          ops = span * w_unique;
          chain = Plan.read_links + combine_links;
        }
      in
      launch K.Unique_count ~total:parts ~per:(per_part ~extra:0)
        (uw K.unique_heads heads :: tickets :: uw K.unique_k k
       :: uw K.unique_gstart gstart :: rows);
      launch K.Unique_heads ~total:parts
        ~per:(per_part ~extra:(3 * line))
        (ur K.unique_heads heads :: tickets :: uw K.unique_gid gid
         :: uw K.unique_gfirst gfirst :: uw K.unique_gstart gstart
         :: uw K.unique_bits bits :: uw K.unique_set set_bits :: key_words
        @ order);
      launch K.Unique_rank ~total:wparts
        ~per:
          {
            Plan.bytes = wspan * 12;
            ops = wspan * unique_rank_ops;
            chain = (3 * Plan.read_links) + combine_links;
          }
        [
          ur K.unique_bits bits;
          uw K.unique_below below;
          ur K.unique_set set_bits;
        ];
      ids_launch
        ([
           ur K.unique_gid gid;
           ur K.unique_gfirst gfirst;
           ur K.unique_gstart gstart;
           ur K.unique_bits bits;
           ur K.unique_below below;
           ur K.unique_k k;
         ]
        @ order)
    end
  with
  | () -> finish c ~kept:(w <= 8)
  | exception Too_long -> Error A.Declined
