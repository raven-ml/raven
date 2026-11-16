(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts and numberings: the path of a call, its workspace and its
   launches' words (kernels.h, Sorts and numberings).

     sort: n <= cap --> sort_block
           keep <= cap / 4 --> zero, sort_round a digit, sort_gather,
                               sort_block of the candidates
           otherwise --> zero, sort_hist, then a key byte at a time
                         sort_count (where a slice has several parts)
                         and sort_scatter
     unique: zero; a word sort (sort_hist, sort_count and sort_scatter
             a byte) a word from the last; unique_count, unique_heads,
             unique_rank, unique_ids *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

let ceil_div a b = (a + b - 1) / b
let set = Plan.set
let set64 w i v = Bytes.set_int64_le w (8 * i) v

(* Bounds for tests: the most parts a pass cuts a slice into, and the most
   groups of a sort's launch. *)
let parts_bound = Atomic.make K.sort_parts
let groups_bound = Atomic.make Plan.stride_groups

let bound_sorts ~parts ~groups =
  if parts < 1 || parts > K.sort_parts then
    invalid_arg "Nx_amd.bound_sorts: parts";
  if groups < 1 || groups > Plan.stride_groups then
    invalid_arg "Nx_amd.bound_sorts: groups";
  Atomic.set parts_bound parts;
  Atomic.set groups_bound groups

(* The groups of a launch of [k] whose groups take [units] units in turn. *)
let groups k units =
  Int.min (Atomic.get groups_bound) (Plan.grid ~per:1 k (Int.max 1 units))

(* A byte [at] into the buffer of slot [slot]. *)
type place = { slot : int; at : int }

let read word p = { Plan.word; slot = p.slot; write = false }
let write word p = { Plan.word; slot = p.slot; write = true }

(* Keys *)

(* How an element's bits become its key: its format, its width, its key's
   bytes and a float's greatest magnitude not NaN. *)
type format = { kind : int; ebits : int; kbytes : int; nan_abs : int64 }

let format (type v s) (dt : (v, s) D.t) =
  let bytes ebits = Int.max 1 (ebits / 8) in
  let float ebits nan_abs =
    { kind = K.sort_float; ebits; kbytes = bytes ebits; nan_abs }
  in
  let int kind ebits = { kind; ebits; kbytes = bytes ebits; nan_abs = -1L } in
  let complex ebits nan_abs =
    { kind = K.sort_complex; ebits; kbytes = bytes ebits; nan_abs }
  in
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
  | D.Complex128 -> complex 128 0x7FF0000000000000L
  | D.Complex64 -> complex 64 0x7F800000L
  | D.Bool -> int K.sort_bool 8
  | D.Bit -> int K.sort_unsigned 1

(* Writes [f]'s words from word [at] of [w]: a real element's NaN key and
   the flips of its sign, and the bits every key is xored with. *)
let set_key w at f ~descending =
  let mask =
    if f.ebits >= 64 then -1L else Int64.(pred (shift_left 1L f.ebits))
  in
  let sign = Int64.shift_left 1L (Int.min f.ebits 64 - 1) in
  let float = f.kind = K.sort_float and signed = f.kind = K.sort_signed in
  let nan_abs = if float || f.kind = K.sort_complex then f.nan_abs else -1L in
  let dmask_lo =
    if not descending then 0L
    else if f.kbytes >= 8 then -1L
    else Int64.(pred (shift_left 1L (8 * f.kbytes)))
  in
  set w (at + K.sort_key_kind) f.kind;
  set w (at + K.sort_key_ebits) f.ebits;
  set w (at + K.sort_key_kbytes) f.kbytes;
  set64 w (at + K.sort_key_nan_abs) nan_abs;
  set64 w (at + K.sort_key_kmask) mask;
  set64 w (at + K.sort_key_flip) (if float || signed then sign else 0L);
  set64 w (at + K.sort_key_flip_signed)
    (if float then mask else if signed then sign else 0L);
  set64 w (at + K.sort_key_dmask_lo) dmask_lo;
  set64 w (at + K.sort_key_dmask_hi)
    (if descending && f.kbytes = 16 then -1L else 0L)

(* Sorts *)

(* A sort of [count] slices of [n] elements of x, along which x, the values
   and the positions step [sx], [sv] and [sp] elements from [off_x],
   [off_v] and [off_p], each of the [kept] axes an extent and those three
   strides, keeping [keep] of each slice. The first pass reads x through
   [order] where given, and the values are stored where given. *)
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
}

(* Writes [t]'s key from word [key] of [w] and its slices from word
   [slices], its last words, and is the count of [w]'s words used. *)
let set_sort w t ~key ~slices =
  set_key w key t.f ~descending:t.descending;
  Plan.axis w (slices + K.sort_slices_n) t.n;
  let small = Plan.small (t.count * t.n) in
  set w (slices + K.sort_slices_small) (Bool.to_int small);
  set w (slices + K.sort_slices_count) t.count;
  set w (slices + K.sort_slices_sx) t.sx;
  set w (slices + K.sort_slices_sv) t.sv;
  set w (slices + K.sort_slices_sp) t.sp;
  set w (slices + K.sort_slices_off_x) t.off_x;
  set w (slices + K.sort_slices_off_v) t.off_v;
  set w (slices + K.sort_slices_off_p) t.off_p;
  set w (slices + K.sort_slices_kept) (Array.length t.kept);
  Array.iteri
    (fun i (e, sx, sv, sp) ->
      let at = slices + K.sort_slices_axes + (K.sort_slices_axis * i) in
      Plan.axis w at e;
      set w (at + K.axis_strides) sx;
      set w (at + K.axis_strides + 1) sv;
      set w (at + K.axis_strides + 2) sp)
    t.kept;
  slices + K.sort_slices_axes + (K.sort_slices_axis * Array.length t.kept)

let flags t =
  (if Option.is_some t.values then K.sort_values else 0)
  lor if Option.is_some t.order then K.sort_order else 0

(* [refs] with [f word p] for [p] where given. *)
let some f word p refs = match p with None -> refs | Some p -> f word p :: refs

(* The longest slice sort_block sorts, and its group's dynamic LDS: each
   key's words and a 16-bit index an element. *)
let cap f = if f.kbytes = 16 then K.sort_cap16 else K.sort_cap
let block_shared f = cap f * ((((f.kbytes + 3) / 4) * 4) + 2)

(* The block sort of [t]'s slices, [per] a group, or with [select] each
   slice's candidates of its select [sel] at [cand], positions of [pbytes]
   bytes. *)
let block ?select b t =
  let w = Plan.words K.Sort_block in
  let cap = cap t.f in
  let per, blocks, refs, flags =
    match select with
    | None ->
        let per = Int.min 256 (cap / t.n) in
        (per, ceil_div t.count per, [], flags t)
    | Some (sel, cand, pbytes) ->
        set w K.sort_block_sel sel.at;
        set w K.sort_block_cand cand.at;
        set w K.sort_block_pbytes pbytes;
        ( 1,
          t.count,
          [ read K.sort_block_sel sel; read K.sort_block_cand cand ],
          flags t lor K.sort_candidates )
  in
  let groups = groups K.Sort_block blocks in
  set w K.sort_block_x t.x.at;
  Option.iter (fun v -> set w K.sort_block_values v.at) t.values;
  set w K.sort_block_positions t.positions.at;
  set w K.sort_block_keep t.keep;
  set w K.sort_block_flags flags;
  set w K.sort_block_groups groups;
  set w K.sort_block_cap cap;
  set w K.sort_block_per per;
  let used =
    set_sort w t ~key:K.sort_block_key ~slices:K.sort_block_slices
  in
  let refs =
    read K.sort_block_x t.x
    :: write K.sort_block_positions t.positions
    :: some write K.sort_block_values t.values refs
  in
  Plan.launch b K.Sort_block ~groups:(groups, 1, 1)
    ~shared:(block_shared t.f) ~used w (Array.of_list refs)

(* The bytes of position a select's digits take: those of [n - 1]. *)
let pbytes n =
  let rec go b = if b < 8 && (n - 1) lsr (8 * b) <> 0 then go (b + 1) else b in
  go 1

(* A select of each slice's [keep] first elements, then their block
   sort. *)
let select b t =
  let ws = Plan.workspace b in
  let cap = cap t.f and pb = pbytes t.n in
  let hblocks = ceil_div t.n K.sort_chunk in
  let sel = { slot = ws; at = Plan.take b (t.count * K.sort_select_bytes) } in
  let cand = { slot = ws; at = Plan.take b (t.count * cap * 24) } in
  let tickets = { slot = Plan.tickets_slot b; at = Plan.tickets b t.count } in
  Plan.zero b ~slot:ws ~at:sel.at (t.count * K.sort_select_bytes);
  let w = Plan.words K.Sort_round in
  set w K.sort_select_x t.x.at;
  set w K.sort_select_sel sel.at;
  set w K.sort_select_tickets tickets.at;
  set w K.sort_select_keep t.keep;
  set w K.sort_select_cand cand.at;
  Plan.axis w K.sort_select_hblocks hblocks;
  set w K.sort_select_cap cap;
  set w K.sort_select_pbytes pb;
  let used = set_sort w t ~key:K.sort_select_key ~slices:K.sort_select_slices in
  let launch k ~pass refs =
    let groups = groups k (t.count * hblocks) in
    let w = Bytes.copy w in
    set w K.sort_select_pass pass;
    set w K.sort_select_groups groups;
    Plan.launch b k ~groups:(groups, 1, 1) ~used w (Array.of_list refs)
  in
  for r = 0 to t.f.kbytes + pb - 1 do
    launch K.Sort_round ~pass:r
      [
        read K.sort_select_x t.x;
        write K.sort_select_sel sel;
        write K.sort_select_tickets tickets;
      ]
  done;
  launch K.Sort_gather ~pass:0
    [
      read K.sort_select_x t.x;
      write K.sort_select_sel sel;
      write K.sort_select_cand cand;
    ];
  block b t ~select:(sel, cand, pb)

(* A pass's items: 8 bytes for keys of 32 bits or fewer, 12 for elements
   of 64 bits or fewer, 24 for the others, and for slices past 2^32
   elements, whose positions take 64 bits. Each with its kernels and the
   items a work-item ranks. *)
let item f n =
  if n > 1 lsl 32 then (K.Sort_count24, K.Sort_scatter24, 24, K.sort_items24)
  else if f.ebits <= 32 then
    (K.Sort_count8, K.Sort_scatter8, 8, K.sort_items8)
  else if f.ebits <= 64 then
    (K.Sort_count12, K.Sort_scatter12, 12, K.sort_items12)
  else (K.Sort_count24, K.Sort_scatter24, 24, K.sort_items24)

(* The parts a pass cuts a slice of [tiles] tiles into, of [span] tiles
   each, with parts enough for a few groups a compute unit across the
   slices, at most {!K.sort_parts} a slice. *)
let parts ~count ~tiles =
  let most = Int.max 1 (ceil_div (Atomic.get parts_bound) count) in
  let span = ceil_div tiles (Int.min tiles most) in
  (ceil_div tiles span, span)

(* A radix sort of [t], whose counts [hist] are zero, which takes the rest
   of its workspace from [b]'s. [reuse] holds the positions' place where
   they are runs of each slice's [n] in slice order that 8-byte items fit
   in, so that a pass's items may wait there. *)
let radix b ws t ~hist ~reuse =
  let count_k, scatter_k, bytes, items = item t.f t.n in
  let kb = t.f.kbytes in
  let tiles = ceil_div t.n (K.sort_threads * items) in
  let parts, span = parts ~count:t.count ~tiles in
  let mark = Plan.mark b in
  let meta = { slot = ws; at = Plan.take b (t.count * 4) } in
  let counts =
    if parts = 1 then None
    else Some { slot = ws; at = Plan.take b (t.count * parts * 256 * 8) }
  in
  let items = t.count * t.n * bytes in
  let buf1 =
    if kb < 2 then None else Some { slot = ws; at = Plan.take b items }
  in
  let buf0 =
    if kb < 3 then None
    else if bytes = 8 then
      match reuse with
      | Some p -> Some p
      | None -> Some { slot = ws; at = Plan.take b items }
    else Some { slot = ws; at = Plan.take b items }
  in
  let tickets = { slot = Plan.tickets_slot b; at = Plan.tickets b t.count } in
  let hblocks = ceil_div t.n K.sort_chunk in
  let h = Plan.words K.Sort_hist in
  let hgroups = groups K.Sort_hist (t.count * hblocks) in
  set h K.sort_hist_x t.x.at;
  set h K.sort_hist_hist hist.at;
  set h K.sort_hist_meta meta.at;
  set h K.sort_hist_tickets tickets.at;
  set h K.sort_hist_groups hgroups;
  Plan.axis h K.sort_hist_hblocks hblocks;
  let used = set_sort h t ~key:K.sort_hist_key ~slices:K.sort_hist_slices in
  Plan.launch b K.Sort_hist ~groups:(hgroups, 1, 1) ~used h
    [|
      read K.sort_hist_x t.x;
      write K.sort_hist_hist hist;
      write K.sort_hist_meta meta;
      write K.sort_hist_tickets tickets;
    |];
  let w = Plan.words count_k in
  let at p = Option.iter (fun p' -> set w p p'.at) in
  set w K.sort_pass_x t.x.at;
  at K.sort_pass_values t.values;
  set w K.sort_pass_positions t.positions.at;
  set w K.sort_pass_tickets tickets.at;
  at K.sort_pass_order t.order;
  set w K.sort_pass_hist hist.at;
  set w K.sort_pass_meta meta.at;
  at K.sort_pass_counts counts;
  at K.sort_pass_buf0 buf0;
  at K.sort_pass_buf1 buf1;
  set w K.sort_pass_keep t.keep;
  set w K.sort_pass_flags (flags t);
  set w K.sort_pass_tiles tiles;
  Plan.axis w K.sort_pass_parts parts;
  set w K.sort_pass_span span;
  let used = set_sort w t ~key:K.sort_pass_key ~slices:K.sort_pass_slices in
  (* Each kernel's grid of its own: they differ in VGPRs, and so in
     whether a count lies in the slow band. *)
  let launch k p refs =
    let groups = groups k (t.count * parts) in
    let w = Bytes.copy w in
    set w K.sort_pass_pass p;
    set w K.sort_pass_groups groups;
    Plan.launch b k ~groups:(groups, 1, 1) ~used w (Array.of_list refs)
  in
  let common =
    read K.sort_pass_x t.x :: read K.sort_pass_hist hist
    :: read K.sort_pass_meta meta
    :: some read K.sort_pass_order t.order []
  in
  for p = 0 to kb - 1 do
    Option.iter
      (fun c ->
        launch count_k p
          (write K.sort_pass_counts c :: write K.sort_pass_tickets tickets
          :: some read K.sort_pass_buf0 buf0
               (some read K.sort_pass_buf1 buf1 common)))
      counts;
    launch scatter_k p
      (write K.sort_pass_positions t.positions
      :: some write K.sort_pass_values t.values
           (some read K.sort_pass_counts counts
              (some write K.sort_pass_buf0 buf0
                 (some write K.sort_pass_buf1 buf1 common))))
  done;
  Plan.release b mark

(* The kept axes of a sort along [axis] of [x] into [values] and
   [positions]: each axis of more than one element as its extent and the
   three strides, an axis merged into the one before where all three step
   along both as one. *)
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

let sort (type v s) ~axis ~descending ~(values : (v, s) A.t)
    ~(positions : (int64, D.int64_elt) A.t) (x : (v, s) A.t) =
  let lx = A.layout x and lv = A.layout values and lp = A.layout positions in
  let n = L.dim lx axis and keep = L.dim lv axis in
  let count = L.numel lx / Int.max 1 n in
  if n = 0 || keep = 0 || count = 0 then Ok None
  else
    let b = Plan.make () in
    let place a = { slot = Plan.slot b (A.buffer a); at = 0 } in
    let kept = kept ~axis lx lv lp in
    let t =
      {
        f = format (A.dtype x);
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
        x = place x;
        values = Some (place values);
        positions = place positions;
        order = None;
      }
    in
    let cap = cap t.f in
    if n <= cap then block b t
    else if keep <= cap / 4 then select b t
    else begin
      let ws = Plan.workspace b in
      let bytes = t.count * t.f.kbytes * 256 * 8 in
      let hist = { slot = ws; at = Plan.take b bytes } in
      Plan.zero b ~slot:ws ~at:hist.at bytes;
      (* Each slice's positions a run of n: 8-byte items fit there. *)
      let runs =
        count = 1
        || match kept with [| (_, _, _, ps) |] -> ps = n | _ -> false
      in
      let reuse =
        if keep = n && t.sp = 1 && runs then
          Some { t.positions with at = 8 * t.off_p }
        else None
      in
      radix b ws t ~hist ~reuse
    end;
    Ok (Some (Plan.finish b))

(* Numberings *)

(* Pieces of [sizes] bytes at offsets from 0, each on 256 bytes, and their
   end. *)
let pieces sizes =
  let up x = (x + 255) land lnot 255 in
  let ats = Array.make (Array.length sizes) 0 in
  let fin =
    Array.fold_left
      (fun (i, at) s ->
        ats.(i) <- at;
        (i + 1, at + up s))
      (0, 0) sizes
    |> snd
  in
  (ats, fin)

let unique ~ids ~first ~counts ~count keys =
  let lk = A.layout keys in
  let n = L.dim lk 0 and w = L.dim lk 1 in
  let m = L.dim (A.layout first) 0 in
  let s0 = L.stride lk 0 and s1 = L.stride lk 1 in
  let b = Plan.make () in
  let place a = { slot = Plan.slot b (A.buffer a); at = 8 * L.offset (A.layout a) } in
  let ids = place ids and first = place first in
  let counts = place counts and count = place count in
  let u = Plan.words K.Unique_ids in
  set u K.unique_ids ids.at;
  set u K.unique_first first.at;
  set u K.unique_counts counts.at;
  set u K.unique_count count.at;
  set u K.unique_m m;
  set u K.unique_n n;
  let ids_launch refs =
    let groups =
      Int.min (Atomic.get groups_bound) (Plan.grid K.Unique_ids (Int.max n m))
    in
    let u = Bytes.copy u in
    set u K.unique_groups groups;
    Plan.launch b K.Unique_ids ~groups:(groups, 1, 1) u
      (Array.of_list
         (write K.unique_ids ids :: write K.unique_first first
         :: write K.unique_counts counts :: write K.unique_count count :: refs))
  in
  if n = 0 then begin
    ids_launch [];
    Ok (Some (Plan.finish b))
  end
  else
    let sorts = w > 0 in
    let words = ceil_div n 64 in
    let most_parts = Plan.grid ~per:K.unique_tile K.Unique_heads n in
    let span = ceil_div n most_parts and wspan = ceil_div words most_parts in
    let parts = ceil_div n span and wparts = ceil_div words wspan in
    let ws = Plan.workspace b in
    let hist = 8 * 256 * 8 in
    (* The zero pieces first: the bits, each part's bits set, each word
       sort's counts. *)
    let zats, zbytes =
      pieces (Array.append [| words * 8; wparts * 8 |]
                (Array.make (if sorts then w else 0) hist))
    in
    let z = Plan.take b zbytes in
    let bits = { slot = ws; at = z + zats.(0) } in
    let set_bits = { slot = ws; at = z + zats.(1) } in
    let hist j = { slot = ws; at = z + zats.(2 + j) } in
    let take bytes = { slot = ws; at = Plan.take b bytes } in
    (* The sorts' order, which no launch reads where no row is sorted. *)
    let order0 = { slot = ws; at = (if sorts then Plan.take b (n * 8) else 0) } in
    let order1 = if sorts && w > 1 then take (n * 8) else order0 in
    let skeys = if sorts && w = 1 then Some (take (n * 8)) else None in
    let gid = take (n * 8) and gfirst = take (n * 8) in
    let gstart = take ((n + 1) * 8) and below = take (words * 8) in
    let heads = take (parts * 8) and k = take 8 in
    Plan.zero b ~slot:ws ~at:z zbytes;
    let key_at = 8 * L.offset lk in
    (* Word [j]'s sort, read through the order of word [j + 1]'s. *)
    let word j =
      let t =
        {
          f = format D.Uint64;
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
          x = { (place keys) with at = 0 };
          values = skeys;
          positions = (if j land 1 = 0 then order0 else order1);
          order =
            (if j = w - 1 then None
             else Some (if j land 1 = 0 then order1 else order0));
        }
      in
      radix b ws t ~hist:(hist j) ~reuse:None
    in
    if sorts then
      for j = w - 1 downto 0 do
        word j
      done;
    let flags =
      (if sorts then K.unique_by_order else 0)
      lor if sorts && w = 1 then K.unique_by_skeys else 0
    in
    set u K.unique_keys key_at;
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
    let tickets = { slot = Plan.tickets_slot b; at = 0 } in
    (* The rows' words as differs reads them: sorted keys, or keys through
       the order of the sorts. *)
    let order = if sorts then [ read K.unique_order order0 ] else [] in
    let key_words =
      match skeys with
      | Some s -> [ read K.unique_skeys s ]
      | None when w = 0 -> []
      | None -> [ read K.unique_keys (place keys) ]
    in
    let rows = if Option.is_some skeys then key_words else key_words @ order in
    let launch k units refs =
      let groups = groups k units in
      let u = Bytes.copy u in
      set u K.unique_groups groups;
      Plan.launch b k ~groups:(groups, 1, 1) u (Array.of_list refs)
    in
    launch K.Unique_count parts
      (write K.unique_heads heads :: write K.unique_tickets tickets
      :: write K.unique_k k :: write K.unique_gstart gstart :: rows);
    launch K.Unique_heads parts
      (read K.unique_heads heads :: write K.unique_tickets tickets
      :: write K.unique_gid gid :: write K.unique_gfirst gfirst
      :: write K.unique_gstart gstart :: write K.unique_bits bits
      :: write K.unique_set set_bits :: key_words @ order);
    launch K.Unique_rank wparts
      [
        read K.unique_bits bits;
        write K.unique_below below;
        read K.unique_set set_bits;
      ];
    ids_launch
      ([
         read K.unique_gid gid;
         read K.unique_gfirst gfirst;
         read K.unique_gstart gstart;
         read K.unique_bits bits;
         read K.unique_below below;
         read K.unique_k k;
       ]
      @ order);
    Ok (Some (Plan.finish b))
