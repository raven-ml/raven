(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorts and numberings: the path of a call, its workspace and its
   launches' parameters (kernels.h, Sorts and numberings).

     sort: n <= cap --> sort_block
           keep <= cap / 4 --> clear, select_round per digit, select_gather,
                               sort_block of the candidates
           otherwise --> clear, sort_hist, sort_pass per key byte
     unique: per word from the last, a word sort (clear, sort_hist, eight
             sort_pass), then clear, unique_heads, unique_rank, unique_ids

   Every launch of a call reads one struct whose fields differ only in
   [pass]; the buffers' fields hold byte offsets that rig turns into
   addresses. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels
module SP = K.Sort_params
module UP = K.Unique_params
module CP = K.Clear_params

(* Kernels *)

type kernel =
  | Clear
  | Block
  | Hist
  | Pass8
  | Pass12
  | Pass24
  | Round
  | Gather
  | Heads
  | Rank
  | Ids

let name = function
  | Clear -> "sort_clear"
  | Block -> "sort_block"
  | Hist -> "sort_hist"
  | Pass8 -> "sort_pass8"
  | Pass12 -> "sort_pass12"
  | Pass24 -> "sort_pass24"
  | Round -> "select_round"
  | Gather -> "select_gather"
  | Heads -> "unique_heads"
  | Rank -> "unique_rank"
  | Ids -> "unique_ids"

type mode = Sorting | Numbering

(* The workspace's slot in each mode. *)
let work_slot = function Sorting -> 3 | Numbering -> 1

(* A sort's fields that address the workspace. *)
let sort_work =
  [
    SP.hist;
    SP.meta;
    SP.status;
    SP.status + 8;
    SP.counter;
    SP.tickets;
    SP.sel;
    SP.cand;
    SP.buf;
    SP.buf + 8;
  ]

let unique_work =
  [
    UP.order;
    UP.skeys;
    UP.gid;
    UP.gfirst;
    UP.gstart;
    UP.bits;
    UP.below;
    UP.status;
    UP.counter;
    UP.k;
  ]

(* The refs of [kernel]'s parameters in a call of [mode]: triples of a
   field's byte, its slot and [1] where the launch writes it. *)
let refs mode kernel =
  let w = work_slot mode in
  let read s at = [ at; s; 0 ] and write s at = [ at; s; 1 ] in
  let work = List.concat_map (write w) in
  Array.of_list
    (match (kernel, mode) with
    | Clear, _ -> write w CP.p
    | (Heads | Rank | Ids), _ ->
        read 0 UP.keys @ write 2 UP.ids @ write 3 UP.first @ write 4 UP.counts
        @ write 5 UP.count @ work unique_work
    | _, Sorting ->
        read 0 SP.x @ write 1 SP.values @ write 2 SP.positions @ work sort_work
    | _, Numbering ->
        read 0 SP.x @ work (SP.values :: SP.positions :: SP.order :: sort_work))

let kernels =
  [|
    Clear; Block; Hist; Pass8; Pass12; Pass24; Round; Gather; Heads; Rank; Ids;
  |]

let ord = function
  | Clear -> 0
  | Block -> 1
  | Hist -> 2
  | Pass8 -> 3
  | Pass12 -> 4
  | Pass24 -> 5
  | Round -> 6
  | Gather -> 7
  | Heads -> 8
  | Rank -> 9
  | Ids -> 10

(* Each kernel's index in {!Kernels.kernels}, and its refs in each mode. *)
let indices = Array.map (fun k -> Launch.kernel (name k)) kernels
let sorting = Array.map (refs Sorting) kernels
let numbering = Array.map (refs Numbering) kernels

(* A launch of [kernel] in a call of [mode] on [groups] groups, of
   parameters [p]. *)
let launch_of mode ?(shared = 0) kernel groups p : Bytes.t Launch.t =
  let k = ord kernel in
  let refs = match mode with Sorting -> sorting | Numbering -> numbering in
  {
    kernel = indices.(k);
    size = Bytes.length p;
    base = 0;
    refs = refs.(k);
    gx = groups;
    gy = 1;
    gz = 1;
    threads = K.sort_threads;
    shared;
    params = p;
  }

let rec fill a i = function
  | [] -> i
  | l :: rest ->
      a.(i) <- l;
      fill a (i + 1) rest

let rec fill_lists a i = function
  | [] -> ()
  | ls :: rest -> fill_lists a (fill a i ls) rest

let rec count n = function
  | [] -> n
  | ls :: rest -> count (n + List.length ls) rest

(* An array of [n] launches, each the first of [lists]. *)
let rec array n = function
  | [] -> [||]
  | [] :: rest -> array n rest
  | (l :: _) :: _ -> Array.make n l

(* The call of the launches of [lists], in order, over [buffers],
   [workspace] bytes of the device's workspace where they name it. *)
let call ~workspace buffers lists =
  let launches = array (count 0 lists) lists in
  fill_lists launches 0 lists;
  { Launch.launches; buffers; workspace; tickets = 0; kept = true; frees = [] }

(* The most groups a launch runs. *)
let max_groups = (1 lsl 31) - 1
let ceil_div a b = (a + b - 1) / b

(* [b] bytes rounded up to a whole number of 256-byte lines. *)
let up b = (b + 255) land lnot 255
let set64 p at x = Bytes.set_int64_le p at (Int64.of_int x)
let set32 p at x = Bytes.set_int32_le p at (Int32.of_int x)

(* A clear of [bytes] bytes of the workspace from [at]. *)
let clear mode ~at bytes =
  let p = Bytes.make CP.size '\000' in
  set64 p CP.p at;
  set64 p CP.words (bytes / 8);
  let groups = Int.max 1 (Int.min 4096 (ceil_div (bytes / 8) K.sort_threads)) in
  launch_of mode Clear groups p

(* [p] with [pass] set, launched as [kernel] on [groups] groups. *)
let launch mode ?shared ?pass kernel groups p =
  let p =
    match pass with
    | None -> p
    | Some x ->
        let p = Bytes.copy p in
        set32 p SP.pass x;
        p
  in
  launch_of mode ?shared kernel groups p

(* Keys *)

(* How an element's bits become its key: its format, its width, its key's
   bytes and a float's greatest magnitude not NaN. *)
type format = { kind : int; ebits : int; kbytes : int; nan_abs : int64 }

let format (type v s) (dt : (v, s) D.t) =
  let bytes ebits = Int.max 1 (ebits / 8) in
  let float ebits nan_abs =
    { kind = K.key_float; ebits; kbytes = bytes ebits; nan_abs }
  in
  let int kind ebits = { kind; ebits; kbytes = bytes ebits; nan_abs = -1L } in
  let complex ebits nan_abs =
    { kind = K.key_complex; ebits; kbytes = bytes ebits; nan_abs }
  in
  match dt with
  | D.Float64 -> float 64 0x7FF0000000000000L
  | D.Float32 -> float 32 0x7F800000L
  | D.Float16 -> float 16 0x7C00L
  | D.Bfloat16 -> float 16 0x7F80L
  | D.Float8_e4m3fn -> float 8 0x7EL
  | D.Float8_e5m2 -> float 8 0x7CL
  | D.Float4_e2m1fn -> float 4 0x7L
  | D.Int64 -> int K.key_signed 64
  | D.Uint64 -> int K.key_unsigned 64
  | D.Int32 -> int K.key_signed 32
  | D.Uint32 -> int K.key_unsigned 32
  | D.Int16 -> int K.key_signed 16
  | D.Uint16 -> int K.key_unsigned 16
  | D.Int8 -> int K.key_signed 8
  | D.Uint8 -> int K.key_unsigned 8
  | D.Int4 -> int K.key_signed 4
  | D.Uint4 -> int K.key_unsigned 4
  | D.Complex128 -> complex 128 0x7FF0000000000000L
  | D.Complex64 -> complex 64 0x7F800000L
  | D.Bool -> int K.key_bool 8
  | D.Bit -> int K.key_unsigned 1

(* Writes [f]'s fields into [p]: a real element's NaN key and the flips of
   its sign, and the bits every key is xored with. *)
let set_format p f ~descending =
  let mask =
    if f.ebits >= 64 then -1L else Int64.(pred (shift_left 1L f.ebits))
  in
  let sign = Int64.shift_left 1L (Int.min f.ebits 64 - 1) in
  let float = f.kind = K.key_float and signed = f.kind = K.key_signed in
  let nan_abs = if float || f.kind = K.key_complex then f.nan_abs else -1L in
  let dmask0 =
    if not descending then 0L
    else if f.kbytes >= 8 then -1L
    else Int64.(pred (shift_left 1L (8 * f.kbytes)))
  in
  Bytes.set_int64_le p SP.nan_abs nan_abs;
  Bytes.set_int64_le p SP.kmask mask;
  Bytes.set_int64_le p SP.flip (if float || signed then sign else 0L);
  Bytes.set_int64_le p (SP.flip + 8)
    (if float then mask else if signed then sign else 0L);
  Bytes.set_int64_le p SP.dmask dmask0;
  Bytes.set_int64_le p (SP.dmask + 8)
    (if descending && f.kbytes = 16 then -1L else 0L);
  set32 p SP.kind f.kind;
  set32 p SP.ebits f.ebits;
  set32 p SP.kbytes f.kbytes

(* Slices *)

(* The kept axes of a sort along [axis] of [x] into [values] and
   [positions]: each axis of more than one element as its extent and
   the three strides, an axis merged into the one before where all three
   step along both as one. *)
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

(* Paths *)

(* The longest slice sort_block sorts, and its shared bytes: each key's
   words and a 16-bit index an element. *)
let cap f = if f.kbytes = 16 then K.sort_cap16 else K.sort_cap
let block_shared f = cap f * (((f.kbytes + 3) / 4 * 4) + 2)

(* A one-sweep pass's items: 8 bytes for keys of 32 bits or fewer, 12 for
   elements of 64 bits or fewer, 24 for the others, and for slices past
   2^32 elements, whose positions take 64 bits. *)
let item f n =
  if n > 1 lsl 32 then (Pass24, 24, K.sort_items24)
  else if f.ebits <= 32 then (Pass8, 8, K.sort_items8)
  else if f.ebits <= 64 then (Pass12, 12, K.sort_items12)
  else (Pass24, 24, K.sort_items24)

(* The bytes of position a select's digits take: those of [n - 1]. *)
let pbytes n =
  let rec go b = if b < 8 && (n - 1) lsr (8 * b) <> 0 then go (b + 1) else b in
  go 1

(* A one-sweep sort's pass kernel, tiles a slice and workspace from [base]:
   the regions the clear zeroes first, then the rest. Keys of one byte take
   one pass and no item buffer, of two bytes one buffer, and wider keys
   two, the first of which may be the positions' destination ([reuse]). *)
type sweep = {
  pass : kernel;
  tiles : int;
  hist : int;
  counter : int;
  tickets : int;
  status0 : int;
  zero : int; (* the bytes from [base] the clear zeroes *)
  status1 : int;
  meta : int;
  buf1 : int;
  buf0 : int;
  fin : int; (* the end *)
}

let sweep f ~base ~slices ~n ~reuse =
  let pass, bytes, items = item f n in
  let tiles = ceil_div n (K.sort_threads * items) in
  let total = slices * n in
  let hist = base in
  let counter = hist + up (slices * f.kbytes * 256 * 8) in
  let tickets = counter + up 64 in
  let status0 = tickets + up (slices * 4) in
  let status = up (slices * tiles * 256 * 8) in
  let status1 = status0 + status in
  let meta = status1 + status in
  let buf1 = meta + up (slices * 4) in
  let items_bytes = up (total * bytes) in
  let buf0 = buf1 + if f.kbytes >= 2 then items_bytes else 0 in
  let fin = buf0 + if f.kbytes >= 3 && not reuse then items_bytes else 0 in
  {
    pass;
    tiles;
    hist;
    counter;
    tickets;
    status0;
    zero = status1 - base;
    status1;
    meta;
    buf1;
    buf0;
    fin;
  }

(* Writes [s]'s regions into [p]. *)
let set_sweep p s ~n =
  set64 p SP.hist s.hist;
  set64 p SP.counter s.counter;
  set64 p SP.tickets s.tickets;
  set64 p SP.status s.status0;
  set64 p (SP.status + 8) s.status1;
  set64 p SP.meta s.meta;
  set64 p SP.buf s.buf0;
  set64 p (SP.buf + 8) s.buf1;
  set64 p SP.tiles s.tiles;
  set64 p SP.hblocks (ceil_div n K.sort_chunk)

(* The launches of a one-sweep sort of [s] from the parameters [p]. *)
let sweep_launches mode f s p ~base ~slices ~n =
  let hblocks = ceil_div n K.sort_chunk in
  clear mode ~at:base s.zero
  :: launch mode Hist (slices * hblocks) p
  :: List.init f.kbytes (fun q ->
      launch mode ~pass:q s.pass (slices * s.tiles) p)

let sort (type v s) ~axis ~descending ~(values : (v, s) A.t)
    ~(positions : (int64, D.int64_elt) A.t) (x : (v, s) A.t) =
  let lx = A.layout x and lv = A.layout values and lp = A.layout positions in
  let n = L.dim lx axis and keep = L.dim lv axis in
  let slices = L.numel lx / Int.max 1 n in
  if n = 0 || keep = 0 || slices = 0 then Ok []
  else
    let f = format (A.dtype x) in
    let rows = kept ~axis lx lv lp in
    let p = Bytes.make SP.size '\000' in
    set64 p SP.n n;
    set64 p SP.keep keep;
    set64 p SP.slices slices;
    set64 p SP.sx (L.stride lx axis);
    set64 p SP.sv (L.stride lv axis);
    set64 p SP.sp (L.stride lp axis);
    set64 p SP.off (L.offset lx);
    set64 p (SP.off + 8) (L.offset lv);
    set64 p (SP.off + 16) (L.offset lp);
    set32 p SP.nkept (Array.length rows);
    Array.iteri
      (fun i (e, sx, sv, sp) ->
        let at = SP.kept + (32 * i) in
        set64 p at e;
        set64 p (at + 8) sx;
        set64 p (at + 16) sv;
        set64 p (at + 24) sp)
      rows;
    set_format p f ~descending;
    let cap = cap f in
    set32 p SP.cap cap;
    (* The flags before the launches, which copy [p]. *)
    set32 p SP.flags K.sort_values;
    let call bytes launches =
      let buffers =
        [| A.buffer x; A.buffer values; A.buffer positions; Launch.workspace |]
      in
      Ok [ call ~workspace:bytes buffers [ launches ] ]
    in
    let hblocks = ceil_div n K.sort_chunk in
    if n <= cap then begin
      let per_block = Int.min 256 (cap / n) in
      let groups = ceil_div slices per_block in
      if groups > max_groups then Error A.Declined
      else begin
        set32 p SP.per_block per_block;
        call 0 [ launch Sorting ~shared:(block_shared f) Block groups p ]
      end
    end
    else if keep <= cap / 4 then begin
      let pb = pbytes n in
      if slices * hblocks > max_groups then Error A.Declined
      else begin
        let sel = 0 and tickets = up (slices * K.Sort_select.size) in
        let cand = tickets + up (slices * 4) in
        set32 p SP.pbytes pb;
        set32 p SP.per_block 1;
        set64 p SP.hblocks hblocks;
        set64 p SP.sel sel;
        set64 p SP.tickets tickets;
        set64 p SP.cand cand;
        let rounds = f.kbytes + pb in
        let gathered = Bytes.copy p in
        set32 gathered SP.flags (K.sort_values lor K.sort_candidates);
        let round r = launch Sorting ~pass:r Round (slices * hblocks) p in
        call
          (cand + up (slices * cap * 24))
          ((clear Sorting ~at:0 cand :: List.init rounds round)
          @ [
              launch Sorting Gather (slices * hblocks) p;
              launch Sorting ~shared:(block_shared f) Block slices gathered;
            ])
      end
    end
    else begin
      let pass, _, _ = item f n in
      let sp = L.stride lp axis in
      (* Each slice's positions a run of n: 8-byte items fit there. *)
      let runs =
        slices = 1
        || match rows with [| (_, _, _, ps) |] -> ps = n | _ -> false
      in
      let reuse = pass = Pass8 && keep = n && sp = 1 && runs in
      let s = sweep f ~base:0 ~slices ~n ~reuse in
      if slices * hblocks > max_groups || slices * s.tiles > max_groups then
        Error A.Declined
      else begin
        if reuse then set32 p SP.flags (K.sort_values lor K.sort_reuse);
        set_sweep p s ~n;
        call s.fin (sweep_launches Sorting f s p ~base:0 ~slices ~n)
      end
    end

(* Numberings *)

(* The most words a numbering's submission sorts. A submission is cached by
   its kernels, so its words are bounded: with keys of at most this many
   words, each numbering's submission is one of a few per item format.
   Wider keys take several submissions, each of up to this many words. *)
let max_words = 8
let run_words = Atomic.make max_words
let set_run_words n = Atomic.set run_words (Int.max 1 (Int.min max_words n))

(* [l] cut into lists of [c] elements, the last one shorter. *)
let rec chunks c l =
  if List.compare_length_with l c <= 0 then [ l ]
  else
    let rec take k acc l =
      if k = 0 then (List.rev acc, l)
      else
        match l with
        | x :: r -> take (k - 1) (x :: acc) r
        | [] -> (List.rev acc, [])
    in
    let first, rest = take c [] l in
    first :: chunks c rest

let unique ~ids ~first ~counts ~count keys =
  let lk = A.layout keys in
  let n = L.dim lk 0 and w = L.dim lk 1 in
  let m = L.dim (A.layout first) 0 in
  let s0 = L.stride lk 0 and s1 = L.stride lk 1 in
  let words = ceil_div n 64 in
  let ht = ceil_div n K.unique_tile and bt = ceil_div words K.unique_tile in
  (* The zero regions first: the bits, the look-back words, the counters;
     then two orders, the sorted words, gid, gfirst, below, gstart, k;
     then a word sort's workspace. *)
  let bits = 0 in
  let status = bits + up (words * 8) in
  let counter = status + up ((ht + bt) * 8) in
  let zero = counter + up 64 in
  let order0 = zero in
  let order1 = order0 + up (n * 8) in
  let skeys = order1 + up (n * 8) in
  let gid = skeys + up (n * 8) in
  let gfirst = gid + up (n * 8) in
  let below = gfirst + up (n * 8) in
  let gstart = below + up (words * 8) in
  let k = gstart + up ((n + 1) * 8) in
  let sorted = k + up 64 in
  let f = format D.Uint64 in
  let s = sweep f ~base:sorted ~slices:1 ~n ~reuse:false in
  let sorts = n > 0 && w > 0 in
  if ht > max_groups || (sorts && s.tiles > max_groups) then Error A.Declined
  else
    let per_run = Atomic.get run_words in
    (* Keys of more words than one submission sorts take submissions one
       after another, with the state between them in a workspace of the
       call's own, which no other call writes in between. *)
    let bytes = if sorts then s.fin else sorted in
    let ws, workspace =
      if w > per_run then (Rig.Buffer.create (A.device keys) bytes, 0)
      else (Launch.workspace, bytes)
    in
    (* Word [j]'s sort, read through the order of word [j + 1]'s. *)
    let word j =
      let p = Bytes.make SP.size '\000' in
      set64 p SP.n n;
      set64 p SP.keep n;
      set64 p SP.slices 1;
      set64 p SP.sx s0;
      set64 p SP.sv 1;
      set64 p SP.sp 1;
      set64 p SP.off (L.offset lk + (j * s1));
      set64 p SP.values skeys;
      set64 p SP.positions (if j land 1 = 0 then order0 else order1);
      set64 p SP.order (if j land 1 = 0 then order1 else order0);
      set_format p f ~descending:false;
      set32 p SP.cap K.sort_cap;
      set_sweep p s ~n;
      let flags =
        (if w = 1 then K.sort_values else 0)
        lor if j < w - 1 then K.sort_order else 0
      in
      set32 p SP.flags flags;
      sweep_launches Numbering f s p ~base:sorted ~slices:1 ~n
    in
    let u = Bytes.make UP.size '\000' in
    let byte a = 8 * L.offset (A.layout a) in
    set64 u UP.keys (byte keys);
    set64 u UP.ids (byte ids);
    set64 u UP.first (byte first);
    set64 u UP.counts (byte counts);
    set64 u UP.count (byte count);
    set64 u UP.order order0;
    set64 u UP.skeys skeys;
    set64 u UP.gid gid;
    set64 u UP.gfirst gfirst;
    set64 u UP.gstart gstart;
    set64 u UP.bits bits;
    set64 u UP.below below;
    set64 u UP.status status;
    set64 u UP.counter counter;
    set64 u UP.k k;
    set64 u UP.n n;
    set64 u UP.w w;
    set64 u UP.s0 s0;
    set64 u UP.s1 s1;
    set64 u UP.m m;
    set32 u UP.flags
      ((if sorts then K.unique_order else 0)
      lor if sorts && w = 1 then K.unique_skeys else 0);
    let ids_groups =
      Int.min 4096 ((Int.min (Int.max n m) (1 lsl 20) / K.sort_threads) + 1)
    in
    let numbering =
      clear Numbering ~at:0 zero
      ::
      (if n > 0 then
         [ launch_of Numbering Heads ht u; launch_of Numbering Rank bt u ]
       else [])
      @ [ launch_of Numbering Ids ids_groups u ]
    in
    let words = if sorts then List.init w (fun i -> word (w - 1 - i)) else [] in
    let call words last =
      (* A call of word sorts alone names the keys and the workspace. *)
      let buffers =
        if last = [] then [| A.buffer keys; ws |]
        else
          [|
            A.buffer keys;
            ws;
            A.buffer ids;
            A.buffer first;
            A.buffer counts;
            A.buffer count;
          |]
      in
      call ~workspace buffers (words @ [ last ])
    in
    let rec calls = function
      | [] -> []
      | [ ws ] -> [ call ws numbering ]
      | ws :: rest -> call ws [] :: calls rest
    in
    let cs = calls (chunks per_run words) in
    if ws == Launch.workspace then Ok cs
    else
      match List.rev cs with
      | [] -> Ok cs
      | c :: rest -> Ok (List.rev ({ c with frees = [ ws ] } :: rest))
