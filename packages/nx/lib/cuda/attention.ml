(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module K = Kernels

let attention_kernel = Launch.kernel "attention"
let merge_kernel = Launch.kernel "attention_merge"
let ceil_div a b = (a + b - 1) / b
let shape_of a = Iarray.to_array (L.shape (A.layout a))

(* The most shared memory a group takes: sm_89's 99 KiB. *)
let shared_most = 99 * 1024

(* The groups a call aims to fill before it cuts its keys into parts: about
   two per multiprocessor of the largest sm_89 parts, a constant so that a
   call's association follows its extents alone. *)
let parts_aim = 256

(* Paths *)

(* A path's rows of a tile and columns of o a group. *)
let rows_of path =
  if path = K.att_f32_r16 then 16 else if path = K.att_f64 then 64 else 128

let columns_of path = if path = K.att_f64 then 64 else K.att_columns

(* A path's shared bytes past the tapes' planes (attention.cu's layouts:
   Simt, and the mma kernel's MMA_OFF_X on, the classes last), and the
   bytes of its staging region, which an FFMA path lays over the planes: no
   tape runs while it stages. *)
let kernel_bytes path =
  let keys = K.att_keys and cls = K.att_class_run in
  if path = K.att_mma then
    let qs = K.att_mma_dk + 8 and vs = K.att_columns + 8 in
    let kv = (2 * keys * qs) + (2 * keys * vs) in
    let off_b = Int.max (2 * 128 * qs) kv in
    (off_b + kv + (2 * 8 * 128) + (2 * 2 * 8 * keys) + cls, 0)
  else
    let rt = rows_of path and dvc = columns_of path in
    let a = if path = K.att_f64 then 8 else 4 in
    let pad = if a = 4 then 4 else 2 in
    let qs = rt + pad and ks = keys + pad and ps = rt + pad in
    let g = K.att_threads / rt in
    let stage = Int.max ((32 * qs) + (32 * ks)) (32 * dvc) in
    let n_a = stage + (keys * ps) + (4 * rt) + (g * rt) + (8 * rt) in
    ((a * n_a) + (8 * rt) + (16 * keys) + (8 * rt) + cls, a * stage)

(* A path's byte where its own shared memory starts past [planes] bytes of
   the tapes' planes, and its shared bytes in all. *)
let layout path ~planes =
  let bytes, stage = kernel_bytes path in
  let start = Int.max 0 (planes - stage + 15) land lnot 15 in
  (start, start + bytes)

let is_narrow = function
  | D.Any (D.Float16 | D.Bfloat16) -> true
  | D.Any _ -> false

(* Tapes *)

let operands_of : P.node -> int list = function
  | P.In _ | Coord _ | Const _ -> []
  | Op1 (_, _, a) -> [ a ]
  | Op2 (_, a, b) -> [ a; b ]
  | Op3 (_, a, b, c) -> [ a; b; c ]

(* Whether each node of [p] is one its outputs [roots] reach. *)
let reached (p : P.t) roots =
  let n = Iarray.length p.nodes in
  let live = Array.make n false in
  List.iter (fun k -> live.(Iarray.get p.outs k) <- true) roots;
  for i = n - 1 downto 0 do
    if live.(i) then
      List.iter (fun a -> live.(a) <- true) (operands_of (Iarray.get p.nodes i))
  done;
  live

(* The leaves of [p] that the outputs [roots] reach: its [In] and [Coord]
   nodes, each once. *)
let leaves (p : P.t) roots =
  let live = reached p roots in
  let seen = Hashtbl.create 8 and acc = ref [] in
  Iarray.iteri
    (fun i x ->
      match x with
      | (P.In _ | Coord _) when live.(i) && not (Hashtbl.mem seen x) ->
          Hashtbl.add seen x ();
          acc := (x, P.dtype p i) :: !acc
      | _ -> ())
    p.nodes;
  List.rev !acc

let is_true = function
  | P.Const (D.Any D.Bool, b) -> String.length b > 0 && b.[0] <> '\000'
  | _ -> false

let is_false = function
  | P.Const (D.Any D.Bool, b) -> String.length b > 0 && b.[0] = '\000'
  | _ -> false

(* A tape: its words, its constants then its body, run as one; its given
   planes, each entry's kind, index and first plane; its outputs' first
   planes; and the planes it takes. The tapes' planes share their bytes, so
   the call takes the most one tape takes. *)
type tape = {
  words : Bytes.t;
  given : (int * int * int) list;
  outs : int array;
  planes : int;
}

let no_tape = { words = Bytes.empty; given = []; outs = [| 0; 0 |]; planes = 0 }

(* The tape of [p]'s outputs [roots], each leaf [x] of dtype [dt] given in
   planes of [entry x dt]'s dtype, as its kind and index. Raises
   [Loop.Too_wide] past the planes an instruction names. *)
let lower_tape (p : P.t) roots ~entry =
  let t = Loop.tape () in
  let planes = Hashtbl.create 8 in
  let given =
    List.map
      (fun (x, dt) ->
        let kind, i, dt = entry x dt in
        let q = Loop.given_plane t dt in
        Hashtbl.replace planes x q;
        (kind, i, q))
      (leaves p roots)
  in
  let outs = [| 0; 0 |] in
  let first, n =
    Loop.lower_given t p ~roots ~given:(Hashtbl.find_opt planes)
      ~out:(fun k q -> outs.(k) <- q)
  in
  let words =
    Bytes.cat (Loop.prologue t) (Bytes.sub (Loop.body t) (8 * first) (8 * n))
  in
  { words; given; outs; planes = Loop.planes_taken t }

(* Whether attention.cu's tape runs each instruction of [words]: a kind
   kernels.h's NX_ATT_KINDS lists, a constant, a cast among float32,
   float64, int64 and bool, a selection of one or two words or a truth. *)
let in_line words =
  let code dt = D.code dt in
  let castable c =
    c = code D.Float32
    || c = code D.Float64
    || c = code D.Int64
    || c = code D.Bool
  in
  let field x s =
    Int64.to_int (Int64.logand (Int64.shift_right_logical x s) 0xfffL)
  in
  let rec go i =
    8 * i >= Bytes.length words
    ||
    let x = Bytes.get_int64_le words (8 * i) in
    let op = Int64.to_int (Int64.logand x 0xffffL) in
    if op = K.loop_const then go (i + 3)
    else
      (if op < K.loop_load then List.mem op K.att_kinds
       else if op = K.loop_cast then
         castable (field x 40) && castable (field x 52)
       else op = K.loop_where1 || op = K.loop_where2 || op = K.loop_truth)
      && go (i + 1)
  in
  go 0

type tapes = { flags : int; mask : tape; score : tape; ranges : tape }

(* The bytes of [tp]'s planes, the pairs' tapes of [e] elements and the
   ranges' of a block an element. *)
let plane_bytes tp e =
  4
  * Int.max
      (e * Int.max tp.mask.planes tp.score.planes)
      (K.att_class_run * tp.ranges.planes)

(* The tapes of [p] over a loop of [rank] axes and a score in [acc]: output
   0 on pairs where it is not true everywhere, output 1 where it is not the
   score, and, where [ranges], output 0's ranges over a block's box where
   they decide some block. Raises [Loop.Too_wide] as {!lower_tape}. *)
let tapes (p : P.t) ~rank ~acc ~ranges:want =
  let out0 = Iarray.get p.nodes (Iarray.get p.outs 0) in
  let out1 = Iarray.get p.nodes (Iarray.get p.outs 1) in
  let ranges =
    if is_true out0 || not want then None
    else
      match P.ranges ~axes:rank ~outs:[| 0 |] p with
      | exception Invalid_argument _ -> None
      | r ->
          let lo = Iarray.get r.nodes (Iarray.get r.outs 0) in
          let hi = Iarray.get r.nodes (Iarray.get r.outs 1) in
          if is_true lo then Some `Shown
          else if is_false lo && is_true hi then None
          else Some (`Ranges r)
  in
  let mask = (not (is_true out0)) && ranges <> Some `Shown in
  let score = out1 <> P.In 0 in
  let pair x dt =
    match x with
    | P.In 0 -> (K.att_give_score, 0, acc)
    | P.In k -> (K.att_give_load, k - 1, dt)
    | P.Coord a -> (K.att_give_coord, a, dt)
    | _ -> assert false
  in
  (* Operands 2i and 2i + 1 of output 0's ranges bound p's operand i, the
     score's never read; then each coordinate's from 2n. *)
  let n = Iarray.length p.ins in
  let bound x dt =
    match x with
    | P.In j when j < 2 * n ->
        ( (if j mod 2 = 0 then K.att_give_load_lo else K.att_give_load_hi),
          (j / 2) - 1,
          dt )
    | P.In j ->
        ( (if j mod 2 = 0 then K.att_give_coord_lo else K.att_give_coord_hi),
          (j - (2 * n)) / 2,
          dt )
    | _ -> invalid_arg "Attention: a coordinate in output 0's ranges"
  in
  let rg =
    match ranges with
    | Some (`Ranges r) when mask -> Some (lower_tape r [ 0; 1 ] ~entry:bound)
    | _ -> None
  in
  {
    flags =
      ((if mask then K.att_mask else 0)
      lor (if score then K.att_score else 0)
      lor if Option.is_some rg then K.att_ranges else 0);
    mask = (if mask then lower_tape p [ 0 ] ~entry:pair else no_tape);
    score = (if score then lower_tape p [ 1 ] ~entry:pair else no_tape);
    ranges = Option.value rg ~default:no_tape;
  }

(* Words *)

exception Too_many

(* A launch's parameter words as they grow: [n] of [b]'s, little-endian,
   whose setters box nothing. *)
type words = { b : Bytes.t; mutable n : int }

let set ws i x = Bytes.set_int64_le ws.b (8 * i) (Int64.of_int x)

let push_int ws x =
  if ws.n >= K.att_words then raise Too_many;
  set ws ws.n x;
  ws.n <- ws.n + 1

(* An array's descriptor, of dtype [dt] laid out by [l]: its slot and
   dtype's code, its first element and the strides at [axes] of its
   own. *)
let descriptor ws ~slot (D.Any dt) l axes =
  let at = ws.n in
  let st = Iarray.to_array (L.strides l) in
  push_int ws (slot lor (D.code dt lsl 16));
  push_int ws (L.offset l);
  Array.iter (fun a -> push_int ws st.(a)) axes;
  at

let range a b = Array.init (b - a) (fun i -> a + i)

(* Calls *)

(* A load the kernel reads: its dtype, its layout over the loop's axes and
   its buffer, made once a plan is taken. *)
type load = { dt : D.any; layout : L.t; buffer : Rig.Buffer.t Lazy.t }

let load_of (A.Any x) =
  {
    dt = D.Any (A.dtype x);
    layout = A.layout x;
    buffer = Lazy.from_val (A.buffer x);
  }

(* A call's arrays and extents, its program aside. *)
type geometry = {
  rank : int;
  nl : int;
  lead : int array;
  nb : int;
  hq : int;
  hk : int;
  tq : int;
  tk : int;
  dk : int;
  dv : int;
  page : int;
  pages : int;
  acc : D.any;
  dts : int; (* q's dtype's code, then o's << 8 *)
  scale : float;
  q : load;
  k : load;
  v : load;
  table : load option;
  lengths : load option;
  o : load;
  lse : load;
  paths : int list;
}

(* A call's launches by [g] on [path] with [ve] elements a thread of a
   tape, of the tapes [tp] over [loads], whose keys go in parts, their
   states in the workspace; with [scores], those of the pass that stores
   the scores there instead, its keys in one part. Raises [Too_many] where
   its words do not fit a launch. *)
let launches g ~path ~ve ~(tp : tapes) ~loads ~scores =
  let planes = plane_bytes tp (ve * K.att_threads) in
  let dv = if Option.is_some scores then 0 else g.dv in
  let gq = g.hq / g.hk in
  (* A path's heads of a group and rows of a head a unit, its chunks of
     the group's heads, its tiles of the rows and its chunks of o's
     columns. *)
  let shape path =
    let rt = rows_of path in
    let hg = Int.min gq (Int.max 1 (rt / Int.min g.tq rt)) in
    let tr = Int.min g.tq (rt / hg) in
    let chunks = ceil_div gq hg and tiles = ceil_div g.tq tr in
    let ncols = Int.max 1 (ceil_div dv (columns_of path)) in
    (hg, tr, chunks, tiles, ncols)
  in
  let hg, tr, chunks, tiles, ncols = shape path in
  let units = g.nb * g.hk * chunks * tiles in
  let nblk = ceil_div g.tk K.att_keys in
  (* The parts follow the groups of the call's first path, whichever path
     runs: a tile of 16 rows, taken where a program leaves a tile of 128
     too little shared memory, keeps the call's association. *)
  let groups =
    let _, _, chunks, tiles, ncols = shape (List.hd g.paths) in
    g.nb * g.hk * chunks * tiles * ncols
  in
  let parts, pblocks =
    if Option.is_some scores || groups >= parts_aim || nblk < 8 then (1, nblk)
    else
      let want =
        Int.min (ceil_div parts_aim groups) (Int.min (nblk / 4) 65535)
      in
      if want < 2 then (1, nblk)
      else
        let pb = ceil_div nblk want in
        (ceil_div nblk pb, pb)
  in
  let accb = if g.acc = D.Any D.Float64 then 8 else 4 in
  let rows = g.nb * g.hq * g.tq in
  (* Slots: the arrays read, then o, lse and the workspace or the
     scores. *)
  let reads =
    [ g.q; g.k; g.v ] @ Option.to_list g.table @ Option.to_list g.lengths
    @ Array.to_list loads
  in
  let nreads = List.length reads in
  let work =
    if Option.is_some scores then scores
    else if parts > 1 then Some (lazy Launch.workspace)
    else None
  in
  let work_bytes =
    if Option.is_none scores && parts > 1 then parts * rows * (g.dv + 2) * accb
    else 0
  in
  let writes = [ Lazy.force g.o.buffer; Lazy.force g.lse.buffer ] in
  let nslots = nreads + 2 + Option.fold ~none:0 ~some:(fun _ -> 1) work in
  let ws = { b = Bytes.make (8 * K.att_words) '\000'; n = 0 } in
  for _ = 0 to K.att_header + nslots - 1 do
    push_int ws 0
  done;
  let set i x = set ws i x in
  set K.att_kernel path;
  set K.att_nlead g.nl;
  set K.att_lead ws.n;
  Array.iter (push_int ws) g.lead;
  set K.att_b g.nb;
  set K.att_hq g.hq;
  set K.att_hk g.hk;
  set K.att_tq g.tq;
  set K.att_tk g.tk;
  set K.att_dk g.dk;
  set K.att_dv dv;
  set K.att_dt g.dts;
  let s =
    if g.acc = D.Any D.Float64 then g.scale
    else Int32.float_of_bits (Int32.bits_of_float g.scale)
  in
  Bytes.set_int64_le ws.b (8 * K.att_scale) (Int64.bits_of_float s);
  set K.att_page g.page;
  set K.att_pages g.pages;
  let four = range g.nl (g.nl + 3) and lead_axes = range 0 g.nl in
  let desc ~slot (x : load) axes = descriptor ws ~slot x.dt x.layout axes in
  set K.att_q (desc ~slot:0 g.q (range 0 g.rank));
  let kv_axes = if g.page > 0 then range 0 4 else range 0 (g.nl + 3) in
  set K.att_k (desc ~slot:1 g.k kv_axes);
  set K.att_values (desc ~slot:2 g.v kv_axes);
  let next = ref 3 in
  Option.iter
    (fun t ->
      set K.att_table (desc ~slot:!next t (range 0 (g.nl + 1)));
      incr next)
    g.table;
  Option.iter
    (fun l ->
      set K.att_lengths (desc ~slot:!next l lead_axes);
      incr next)
    g.lengths;
  let load_slot = !next in
  set K.att_o (desc ~slot:nreads g.o (range 0 g.rank));
  set K.att_lse (desc ~slot:(nreads + 1) g.lse (range 0 (g.rank - 1)));
  set K.att_work (nreads + 2);
  set K.att_hg hg;
  set K.att_tr tr;
  set K.att_chunks chunks;
  set K.att_tiles tiles;
  set K.att_units units;
  set K.att_parts parts;
  set K.att_pblocks pblocks;
  set K.att_v ve;
  set K.att_shared (fst (layout path ~planes));
  set K.att_flags
    (tp.flags lor if Option.is_some scores then K.att_scores else 0);
  (* The loads' table, then their descriptors. *)
  let table_at = ws.n in
  set K.att_loads table_at;
  Array.iter (fun _ -> push_int ws 0) loads;
  Array.iteri
    (fun j x ->
      set (table_at + j)
        (desc ~slot:(load_slot + j) x (Array.append lead_axes four)))
    loads;
  let table g =
    let at = ws.n in
    push_int ws (List.length g);
    List.iter
      (fun (kind, i, q) -> push_int ws (kind lor (i lsl 8) lor (q lsl 32)))
      g;
    at
  in
  let place (x : tape) ~at ~n ~given =
    set given (table x.given);
    set at ws.n;
    let n' = Bytes.length x.words / 8 in
    set n n';
    if ws.n + n' > K.att_words then raise Too_many;
    Bytes.blit x.words 0 ws.b (8 * ws.n) (8 * n');
    ws.n <- ws.n + n'
  in
  place tp.mask ~at:K.att_mask_at ~n:K.att_mask_n ~given:K.att_mask_given;
  place tp.score ~at:K.att_score_at ~n:K.att_score_n ~given:K.att_score_given;
  place tp.ranges ~at:K.att_ranges_at ~n:K.att_ranges_n
    ~given:K.att_ranges_given;
  set K.att_mask_out tp.mask.outs.(0);
  set K.att_score_out tp.score.outs.(1);
  set K.att_ranges_lo tp.ranges.outs.(0);
  set K.att_ranges_hi tp.ranges.outs.(1);
  let params = ws.b in
  (* The buffers once the words fit. *)
  let reads = List.map (fun (x : load) -> Lazy.force x.buffer) reads in
  let writes = writes @ Option.to_list (Option.map Lazy.force work) in
  let refs =
    Launch.slot_refs ~header:K.att_header ~reads:(List.length reads)
      ~writes:(List.length writes)
  in
  let launch kernel ~gx ~gy ~gz ~shared =
    {
      Launch.kernel;
      size = 8 * K.att_words;
      base = 0;
      refs;
      gx;
      gy;
      gz;
      threads = K.att_threads;
      shared;
      params;
    }
  in
  let main =
    launch attention_kernel ~gx:(Int.min units 0x7fffffff)
      ~gy:(Int.min ncols 65535) ~gz:parts
      ~shared:(snd (layout path ~planes))
  in
  let launches =
    if parts = 1 then [| main |]
    else
      let merged = Int.min (ceil_div (rows * (g.dv + 1)) K.att_threads) 65535 in
      [| main; launch merge_kernel ~gx:merged ~gy:1 ~gz:1 ~shared:0 |]
  in
  {
    Launch.launches;
    buffers = Array.of_list (reads @ writes);
    workspace = work_bytes;
    tickets = 0;
    kept = true;
    frees = [];
  }

(* The call of [p] over [loads] on the first of [g]'s paths whose tapes'
   planes fit beside it, with output 0's ranges, else without them. [None]
   where a tape holds an instruction the kernel does not run, or does not
   fit a launch's planes or parameters. *)
let fit g (p : P.t) loads =
  let attempt ~ranges =
    match tapes p ~rank:g.rank ~acc:g.acc ~ranges with
    | exception Loop.Too_wide -> None
    | tp
      when not
             (List.for_all
                (fun t -> in_line t.words)
                [ tp.mask; tp.score; tp.ranges ]) ->
        None
    | tp -> (
        let fits path v =
          snd (layout path ~planes:(plane_bytes tp (v * K.att_threads)))
          <= shared_most
        in
        let chosen =
          List.find_map
            (fun path ->
              Option.map
                (fun v -> (path, v))
                (List.find_opt (fits path) [ K.att_most_v; 2; 1 ]))
            g.paths
        in
        match chosen with
        | None -> None
        | Some (path, ve) -> (
            match launches g ~path ~ve ~tp ~loads ~scores:None with
            | exception Too_many -> None
            | c -> Some c))
  in
  match attempt ~ranges:true with
  | Some c -> Some c
  | None -> attempt ~ranges:false

(* Programs past the kernel

   A program whose tapes do not fit the kernel's planes or parameters
   computes in part through map, into temporaries the kernel reads as
   loads, the first plan that fits of: output 0 into a temporary; each
   node of output 1 that reads no score but that a node reading it reads;
   both; output 1 itself from the scores a first pass stores, with output
   0 in the kernel or a temporary. A temporary spans the axes its node's
   leaves vary along, a load's of more than one element and a stride and
   a coordinate's, and is broadcast along the others. *)

(* A temporary: its node of the call's program, its dtype and whether it
   varies along each axis. *)
type temp = { node : int; tdt : D.any; along : bool array }

(* The axes along which each node of [p] over [loads] varies. *)
let varying (p : P.t) (loads : load array) rank =
  let n = Iarray.length p.nodes in
  let along = Array.make_matrix n rank false in
  Iarray.iteri
    (fun i nd ->
      match nd with
      | P.In 0 -> Array.fill along.(i) 0 rank true
      | P.In k ->
          let l = loads.(k - 1).layout in
          for a = 0 to rank - 1 do
            along.(i).(a) <- L.dim l a > 1 && L.stride l a <> 0
          done
      | Coord a -> along.(i).(a) <- true
      | Const _ -> ()
      | Op1 _ | Op2 _ | Op3 _ ->
          List.iter
            (fun o ->
              Array.iteri (fun a x -> if x then along.(i).(a) <- true) along.(o))
            (operands_of nd))
    p.nodes;
  along

(* Whether each node of [p] reads the score, through its operands. *)
let scored (p : P.t) =
  let s = Array.make (Iarray.length p.nodes) false in
  Iarray.iteri
    (fun i nd ->
      s.(i) <-
        (match nd with
        | P.In 0 -> true
        | _ -> List.exists (fun a -> s.(a)) (operands_of nd)))
    p.nodes;
  s

let computed : P.node -> bool = function
  | P.Op1 _ | Op2 _ | Op3 _ -> true
  | P.In _ | Coord _ | Const _ -> false

(* The nodes of output 1 that read no score and that a node reading it
   reads, or output 1 itself where it reads none, computed. *)
let frontier (p : P.t) =
  let s = scored p and live = reached p [ 1 ] in
  let out1 = Iarray.get p.outs 1 in
  let marked = Array.make (Iarray.length p.nodes) false in
  if (not s.(out1)) && computed (Iarray.get p.nodes out1) then
    marked.(out1) <- true;
  Iarray.iteri
    (fun i nd ->
      if live.(i) && s.(i) then
        List.iter
          (fun a ->
            if (not s.(a)) && computed (Iarray.get p.nodes a) then
              marked.(a) <- true)
          (operands_of nd))
    p.nodes;
  List.filter (fun i -> marked.(i)) (List.init (Iarray.length p.nodes) Fun.id)

(* [p] with output 1 read through operand [subs i] in place of each node
   [i] it maps, its other nodes reading the score copied past [p]'s, and
   with output 0 the operand [out0] where given, over operands [ins]. *)
let rewrite (p : P.t) ~ins ~subs ~out0 =
  let nodes = ref (List.rev (Iarray.to_list p.nodes)) in
  let n = ref (Iarray.length p.nodes) in
  let add x =
    nodes := x :: !nodes;
    incr n;
    !n - 1
  in
  let s = scored p and live = reached p [ 1 ] in
  let copy = Array.init (Iarray.length p.nodes) Fun.id in
  Iarray.iteri
    (fun i nd ->
      if live.(i) then
        match subs i with
        | Some j -> copy.(i) <- add (P.In j)
        | None when s.(i) && computed nd ->
            let c = copy in
            copy.(i) <-
              add
                (match nd with
                | P.Op1 (k, dt, a) -> P.Op1 (k, dt, c.(a))
                | Op2 (k, a, b) -> Op2 (k, c.(a), c.(b))
                | Op3 (k, a, b, d) -> Op3 (k, c.(a), c.(b), c.(d))
                | x -> x)
        | None -> ())
    p.nodes;
  let o1 = copy.(Iarray.get p.outs 1) in
  let o0 =
    match out0 with Some j -> add (P.In j) | None -> Iarray.get p.outs 0
  in
  P.v ~ins (Iarray.of_list (List.rev !nodes)) ~outs:(Iarray.of_list [ o0; o1 ])

(* [p] over its operands that its outputs read, the score kept, and those
   operands' loads. *)
let compact (p : P.t) (loads : load array) =
  let live = reached p [ 0; 1 ] in
  let used = Array.make (Iarray.length p.ins) false in
  used.(0) <- true;
  Iarray.iteri
    (fun i nd ->
      match nd with P.In k when live.(i) -> used.(k) <- true | _ -> ())
    p.nodes;
  let index = Array.make (Array.length used) (-1) and m = ref 0 in
  Array.iteri
    (fun k u ->
      if u then (
        index.(k) <- !m;
        incr m))
    used;
  let kept =
    List.filter (fun k -> used.(k)) (List.init (Array.length used) Fun.id)
  in
  let nodes =
    Iarray.map
      (function
        | P.In k when index.(k) >= 0 -> P.In index.(k)
        | P.In k ->
            let (D.Any d) = Iarray.get p.ins k in
            P.Const (D.Any d, String.make (D.bytes d 1) '\000')
        | nd -> nd)
      p.nodes
  in
  ( P.v
      ~ins:(Iarray.of_list (List.map (Iarray.get p.ins) kept))
      nodes ~outs:p.outs,
    Array.of_list
      (List.filter_map
         (fun k -> if k = 0 then None else Some loads.(k - 1))
         kept) )

(* The contiguous strides of [shape], [0] along an axis of one element. *)
let strides shape =
  let st = Array.make (Array.length shape) 0 and m = ref 1 in
  for a = Array.length shape - 1 downto 0 do
    if shape.(a) > 1 then st.(a) <- !m;
    m := !m * shape.(a)
  done;
  st

(* [x] read over [shape], its offset and strides kept: where a
   temporary's node does not vary along an axis, the index 0 there stands
   for every other. *)
let over shape (x : load) =
  let (D.Any d) = x.dt in
  A.Any
    (A.v d
       (L.v ~offset:(L.offset x.layout) ~strides:(L.strides x.layout)
          (Iarray.of_array shape))
       (Lazy.force x.buffer))

(* The loop's shape. *)
let loop_shape g = Array.concat [ g.lead; [| g.hq; g.tq; g.tk |] ]

(* The temporary of [t]: its load over the loop and its shape, one element
   along the axes it does not vary along. *)
let temp_load g dev t =
  let full = loop_shape g in
  let shape = Array.mapi (fun a x -> if t.along.(a) then x else 1) full in
  let (D.Any d) = t.tdt in
  let n = Array.fold_left ( * ) 1 shape in
  let buffer = lazy (Rig.Buffer.create dev (D.bytes d n)) in
  let layout =
    L.v ~strides:(Iarray.of_array (strides shape)) (Iarray.of_array full)
  in
  ({ dt = t.tdt; layout; buffer }, shape)

let ( let* ) = Result.bind

(* The calls of a map of [p]'s nodes [ts] into their temporaries [xs],
   one map for each shape. *)
let map_temps (p : P.t) ~user ts xs =
  (* p's nodes over the user's loads alone: operand 0, the score, is read
     by none of them. *)
  let ins = Iarray.sub p.ins ~pos:1 ~len:(Iarray.length p.ins - 1) in
  let nodes =
    Iarray.map
      (function
        | P.In 0 ->
            let (D.Any d) = Iarray.get p.ins 0 in
            P.Const (D.Any d, String.make (D.bytes d 1) '\000')
        | P.In k -> P.In (k - 1)
        | nd -> nd)
      p.nodes
  in
  let shapes = List.sort_uniq compare (List.map (fun (_, s) -> s) xs) in
  List.fold_left
    (fun acc shape ->
      let* steps = acc in
      let pairs =
        List.filter (fun (_, (_, s)) -> s = shape) (List.combine ts xs)
      in
      let outs =
        Iarray.of_list (List.map (fun ((t : temp), _) -> t.node) pairs)
      in
      let dsts =
        Iarray.of_list
          (List.map
             (fun (_, ((x : load), _)) ->
               let (D.Any d) = x.dt in
               A.Any
                 (A.v d
                    (L.contiguous (Iarray.of_array shape))
                    (Lazy.force x.buffer)))
             pairs)
      in
      let loads =
        Iarray.init (Array.length user) (fun i ->
            let (A.Any y) = over shape user.(i) in
            Nx_kernel.Plain y)
      in
      let* cs = Loop.map (P.v ~ins nodes ~outs) ~dsts loads in
      Ok (steps @ cs))
    (Ok []) shapes

let attention (type v s w r c) (p : P.t) ~scale ~(o : (w, r) A.t)
    ~(lse : (float, c) A.t) ?lengths (q : (v, s) A.t) kv (loads : A.any iarray)
    =
  let qs = shape_of q in
  let rank = Array.length qs in
  let nl = rank - 3 in
  let lead = Array.sub qs 0 nl in
  let nb = Array.fold_left ( * ) 1 lead in
  let hq = qs.(nl) and tq = qs.(nl + 1) and dk = qs.(nl + 2) in
  let k, v, table =
    match kv with
    | Nx_kernel.Dense { k; v } -> (k, v, None)
    | Paged { k; v; table } -> (k, v, Some table)
  in
  let ks = shape_of k and vs = shape_of v in
  let paged = Option.is_some table in
  let hk = if paged then ks.(1) else ks.(nl) in
  let dv = vs.(Array.length vs - 1) in
  let tk, page, pages =
    match table with
    | None -> (ks.(nl + 1), 0, 0)
    | Some t -> ((shape_of t).(nl) * ks.(2), ks.(2), ks.(0))
  in
  if nb * hq * tq = 0 then Ok []
  else
    let acc = D.Any (A.dtype lse) and dt = D.Any (A.dtype q) in
    let f64 = acc = D.Any D.Float64 in
    let gq = hq / hk in
    (* The path by the dtypes and the extents; on fused multiply-adds, a
       tile of 16 rows where a tile of 128 leaves its tapes' planes too
       little room, which changes no bit. *)
    let paths =
      if f64 then [ K.att_f64 ]
      else if is_narrow dt && dk <= K.att_mma_dk && gq * tq > 16 then
        [ K.att_mma ]
      else if gq * tq <= 16 then [ K.att_f32_r16 ]
      else [ K.att_f32; K.att_f32_r16 ]
    in
    let g =
      {
        rank;
        nl;
        lead;
        nb;
        hq;
        hk;
        tq;
        tk;
        dk;
        dv;
        page;
        pages;
        acc;
        dts = D.code (A.dtype q) lor (D.code (A.dtype o) lsl 8);
        scale;
        q = load_of (A.Any q);
        k = load_of (A.Any k);
        v = load_of (A.Any v);
        table = Option.map (fun t -> load_of (A.Any t)) table;
        lengths = Option.map (fun l -> load_of (A.Any l)) lengths;
        o = load_of (A.Any o);
        lse = load_of (A.Any lse);
        paths;
      }
    in
    let user = Array.of_list (List.map load_of (Iarray.to_list loads)) in
    let dev = A.device q in
    let nins = Iarray.length p.ins in
    let out0 = Iarray.get p.nodes (Iarray.get p.outs 0) in
    let out1 = Iarray.get p.nodes (Iarray.get p.outs 1) in
    (* Without keys no pair runs the program. *)
    let p, user =
      if tk = 0 then
        ( P.v ~ins:(Iarray.of_list [ acc ])
            (Iarray.of_list [ P.In 0; P.Const (D.Any D.Bool, "\001") ])
            ~outs:(Iarray.of_list [ 1; 0 ]),
          [||] )
      else (p, user)
    in
    let along = varying p user rank in
    let temp i = { node = i; tdt = P.dtype p i; along = along.(i) } in
    (* Output 0 in the kernel or a temporary; output 1 in the kernel, its
       nodes reading no score in temporaries, or from the scores. *)
    let front = if tk = 0 then [] else frontier p in
    let masks =
      if tk = 0 || is_true out0 || not (computed out0) then [ `Kernel ]
      else [ `Kernel; `Temp ]
    in
    let rescores =
      if tk = 0 || out1 = P.In 0 then [ `Kernel ]
      else if front = [] then [ `Kernel; `Scores ]
      else [ `Kernel; `Frontier; `Scores ]
    in
    let order =
      List.concat_map (fun s -> List.map (fun m -> (m, s)) masks) rescores
    in
    let accb = if f64 then 8 else 4 in
    let plan (mask, score) =
      let ts = if score = `Frontier then List.map temp front else [] in
      let o0 = Iarray.get p.outs 0 in
      let ts =
        if mask = `Temp && not (List.exists (fun (t : temp) -> t.node = o0) ts)
        then ts @ [ temp o0 ]
        else ts
      in
      let xs = List.map (temp_load g dev) ts in
      let scores =
        if score <> `Scores then None
        else
          let n = nb * hq * tq * tk in
          let buffer = lazy (Rig.Buffer.create dev (accb * n)) in
          let layout = L.contiguous (Iarray.of_array (loop_shape g)) in
          Some { dt = acc; layout; buffer }
      in
      let extra = List.map fst xs @ Option.to_list scores in
      let index = Hashtbl.create 8 in
      List.iteri
        (fun j (t : temp) -> Hashtbl.replace index t.node (nins + j))
        ts;
      let subs i =
        if score = `Scores && i = Iarray.get p.outs 1 then
          Some (nins + List.length ts)
        else Hashtbl.find_opt index i
      in
      let out0 = if mask = `Temp then Hashtbl.find_opt index o0 else None in
      let ins =
        Iarray.append p.ins
          (Iarray.of_list (List.map (fun (x : load) -> x.dt) extra))
      in
      let p' = if extra = [] then p else rewrite p ~ins ~subs ~out0 in
      let p', kl = compact p' (Array.append user (Array.of_list extra)) in
      match fit g p' kl with None -> None | Some c -> Some (ts, xs, scores, c)
    in
    match List.find_map plan order with
    | None ->
        (* The last plan reads two loads through tapes of no node. *)
        assert false
    | Some (ts, xs, scores, c) ->
        let* first =
          match scores with
          | None -> Ok []
          | Some s ->
              (* The scores, then output 1 over them in their place. *)
              let path = List.hd paths in
              let pass =
                launches g ~path ~ve:1
                  ~tp:
                    {
                      flags = 0;
                      mask = no_tape;
                      score = no_tape;
                      ranges = no_tape;
                    }
                  ~loads:[||] ~scores:(Some s.buffer)
              in
              let (D.Any d) = acc in
              let sa = A.v d s.layout (Lazy.force s.buffer) in
              let ld =
                Iarray.init
                  (Iarray.length loads + 1)
                  (fun i ->
                    if i = 0 then Nx_kernel.Plain sa
                    else
                      let (A.Any y) = Iarray.get loads (i - 1) in
                      Nx_kernel.Plain y)
              in
              let p1 =
                P.v ~ins:p.ins p.nodes
                  ~outs:(Iarray.of_list [ Iarray.get p.outs 1 ])
              in
              let* cs = Loop.map p1 ~dsts:[| A.Any sa |] ld in
              Ok (pass :: cs)
        in
        let* maps = if ts = [] then Ok [] else map_temps p ~user ts xs in
        (* The temporaries and the scores the plan made. *)
        let made =
          List.filter_map
            (fun (x : load) ->
              if Lazy.is_val x.buffer then Some (Lazy.force x.buffer) else None)
            (List.map fst xs @ Option.to_list scores)
        in
        let c = if made = [] then c else { c with frees = made } in
        Ok (first @ maps @ [ c ])
