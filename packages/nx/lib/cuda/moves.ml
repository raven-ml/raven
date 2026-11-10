(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Moves and their adjoints: each call planned as launches of moves.cu,
   their parameters written as kernels.h lays them out.

     arrays --> launches: a kernel, its parameters, a buffer a slot

   Each kernel names a fixed set of slots, every pointer of its parameters
   one of them, so a submission's refs follow its kernels alone: a launch's
   slots follow the slots of the launches before it. A slot a launch does
   not read takes the buffer of its first.

   A call's launches that share the workspace run as one submission, so
   that no other call's work comes between them: every call but an
   assembly is one submission. An assembly, which reads no workspace,
   submits each launch apart, ordered by its destination. So the kernel
   sequences a device keeps submissions of are few: a move; a range
   gather's 1, 2 or 4 launches; a fill or a copy of pieces; a fold; and a
   scatter's copy of [into] or none, then one pass, a zeroing and one or
   two passes, a tile pass, or a zeroing, a tile pass of 4 kinds, the
   spill, 1 to 8 sort passes of 3 launches and the two tree launches:
   under 100 in all.

   A range gather takes two more launches past 4096 runs a row, and a
   tiled float Add a sort pass for each byte of its keys, of
   ⌈log₂ targets⌉ + ⌈log₂ tiles⌉ bits, so their words grow with those.

   A launch's grid follows its work alone: every kernel strides over its
   work, so its bits do not depend on the grid.

   A float Add sums in float32 or float64. A complex one runs as the float
   Add of its parts: the arrays read as floats with a last axis of the two
   parts, along which idx steps 0, so that the row tiles sum both parts of
   a target together, each part by its own position tree. *)

module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module K = Kernels

(* Kernels *)

let kernels = Array.length K.kernels

let index name =
  let rec find i =
    if i = kernels then invalid_arg ("Moves: no kernel " ^ name)
    else if fst K.kernels.(i) = name then i
    else find (i + 1)
  in
  find 0

(* The size of each family's parameters, whether a launch writes each of
   its slots, and its pointer fields with their slots. *)
let layout = function
  | K.Move ->
      let module M = K.Move_params in
      ( M.size,
        [| true; false; false; false |],
        [|
          (M.dst, 0);
          (M.src, 1);
          (M.idx, 2);
          (M.starts, 3);
          (M.ends, 3);
          (M.maxima, 3);
        |] )
  | Runs _ ->
      let module R = K.Runs_params in
      ( R.size,
        [| false; false; true |],
        [|
          (R.in_s, 0);
          (R.in_e, 1);
          (R.starts, 2);
          (R.ends, 2);
          (R.maxima, 2);
          (R.carry, 2);
        |] )
  | Assemble ->
      let module P = K.Assemble_params in
      ( P.size,
        Array.init (1 + K.assemble_pieces) (fun i -> i = 0),
        Array.append
          [| (P.dst, 0) |]
          (Array.init K.assemble_pieces (fun i -> (P.src + (8 * i), 1 + i))) )
  | Fold ->
      let module F = K.Fold_params in
      (F.size, [| true; false |], [| (F.dst, 0); (F.x, 1) |])
  | Scatter | Scatter_add _ | Scatter_sort _ ->
      let module S = K.Scatter_params in
      let ws =
        [
          S.owner;
          S.locks;
          S.tiles_of;
          S.ent_g;
          S.ent_v;
          S.ent_n;
          S.keys;
          S.keys + 8;
          S.index;
          S.index + 8;
          S.piece;
          S.hist;
          S.ctl;
        ]
      in
      ( S.size,
        [| true; false; false; true |],
        Array.append
          [| (S.dst, 0); (S.idx, 1); (S.upd, 2) |]
          (Array.of_list (List.map (fun at -> (at, 3)) ws)) )
  | Pack | Rows | Ks | Gemm _ | Gemv _ | Mma _ | Loop | Linalg _ | Linalg_any _
  | Spectral _ | Spectral_any _ | Spectral_real _ | Clear | Sort | Unique
  | Fft _ | Fft_real _ | Attention | Attention_merge ->
      invalid_arg "Moves: not a move's kernel"

let layout_of k = layout (snd K.kernels.(k))

(* Each move's kernel's refs into its own slots, from the first. *)
let refs =
  Array.init kernels (fun k ->
      match layout_of k with
      | exception Invalid_argument _ -> [||]
      | _, writes, fields ->
          Array.concat
            (List.map
               (fun (at, s) -> [| at; s; Bool.to_int writes.(s) |])
               (Array.to_list fields)))

(* A launch's parameters as they are written: its kernel, their bytes and
   the buffer of each slot named so far. *)
type params = { k : int; b : Bytes.t; slots : Rig.Buffer.t option array }

let params name =
  let k = index name in
  let size, writes, _ = layout_of k in
  {
    k;
    b = Bytes.make size '\000';
    slots = Array.make (Array.length writes) None;
  }

let again p = { p with b = Bytes.copy p.b; slots = Array.copy p.slots }

(* [p] as the parameters of kernel [name], of the same family. *)
let as_kernel name p = { (again p) with k = index name }
let i64 p at v = Bytes.set_int64_le p.b at (Int64.of_int v)
let i32 p at v = Bytes.set_int32_le p.b at (Int32.of_int v)
let u64 p at v = Bytes.set_int64_le p.b at v

(* The pointer at [at] to byte [byte] of [buf], whose slot the kernel's
   layout gives. *)
let ptr p at ?(byte = 0) buf =
  let _, _, fields = layout_of p.k in
  let rec slot i =
    if fst fields.(i) = at then snd fields.(i) else slot (i + 1)
  in
  i64 p at byte;
  p.slots.(slot 0) <- Some buf

(* The groups of a grid that strides over [work] elements. *)
let groups work =
  let g = (work + K.move_threads - 1) / K.move_threads in
  Int.max 1 (Int.min g 8192)

(* A call's launches and the buffers of their slots as they grow, in
   reverse, the slots taken and the workspace's bytes. Each launch's slots
   follow the one before's, or, [apart], start a call of their own. *)
type plan = {
  apart : bool;
  mutable launches : Bytes.t Launch.t list;
  mutable buffers : Rig.Buffer.t array list;
  mutable count : int;
  mutable slots : int;
  mutable workspace : int;
}

let plan ?(apart = false) () =
  { apart; launches = []; buffers = []; count = 0; slots = 0; workspace = 0 }

(* The workspace, of at least [bytes] for [pl]. *)
let workspace pl bytes =
  pl.workspace <- Int.max pl.workspace bytes;
  Launch.workspace

let launch pl (p : params) ~work =
  let buffers = Array.make (Array.length p.slots) (Option.get p.slots.(0)) in
  for i = 1 to Array.length buffers - 1 do
    match p.slots.(i) with Some b -> buffers.(i) <- b | None -> ()
  done;
  let base = if pl.apart then 0 else pl.slots in
  pl.slots <- base + Array.length buffers;
  pl.count <- pl.count + 1;
  pl.buffers <- buffers :: pl.buffers;
  pl.launches <-
    {
      kernel = p.k;
      size = Bytes.length p.b;
      base;
      refs = refs.(p.k);
      gx = groups work;
      gy = 1;
      gz = 1;
      threads = K.move_threads;
      shared = 0;
      params = p.b;
    }
    :: pl.launches

let call pl launches buffers =
  {
    Launch.launches;
    buffers;
    workspace = pl.workspace;
    tickets = 0;
    kept = true;
    frees = [];
  }

(* [pl]'s launches and their buffers in one call, filled from the last. *)
let one_call pl =
  let launches = Array.make pl.count (List.hd pl.launches) in
  List.iteri (fun i l -> launches.(pl.count - 1 - i) <- l) pl.launches;
  let buffers = Array.make pl.slots Launch.workspace in
  ignore
    (List.fold_left
       (fun at b ->
         let at = at - Array.length b in
         Array.blit b 0 buffers at (Array.length b);
         at)
       pl.slots pl.buffers);
  call pl launches buffers

(* The plan as one call, or, [apart], a call a launch. *)
let finish pl : Bytes.t Launch.plan =
  match pl.launches with
  | [] -> Ok []
  | ls when pl.apart ->
      Ok (List.rev (List.map2 (fun l b -> call pl [| l |] b) ls pl.buffers))
  | _ -> Ok [ one_call pl ]

(* Arrays *)

let shape_of a = Iarray.to_array (L.shape (A.layout a))
let strides_of a = Iarray.to_array (L.strides (A.layout a))
let offset_of a = L.offset (A.layout a)
let numel s = Array.fold_left ( * ) 1 s
let code a = D.code (A.dtype a)

(* An element's units, of its dtype's alignment, and their count: two for
   a complex number, a part each. *)
let unit_count a =
  let dt = A.dtype a in
  if D.is D.Complex dt then (D.bits dt / 2, 2) else (D.bits dt, 1)

(* C-order strides of [s]. *)
let c_strides s =
  let r = Array.length s in
  let st = Array.make r 1 in
  for i = r - 2 downto 0 do
    st.(i) <- st.(i + 1) * s.(i + 1)
  done;
  st

let small n = if n < 1 lsl 32 then 1 else 0

let rec ceil_log2 ?(b = 0) n =
  if 1 lsl b >= n then b else ceil_log2 ~b:(b + 1) n

(* A sub-byte destination's work: its 32-bit words. *)
let work_of ~unit n = if unit < 8 then (n * unit / 32) + 2 else n

(* Moves *)

(* [dst], C-contiguous over [shape], gets [mode]'s elements: [axes a] is
   axis [a]'s strides of the source and of the positions. *)
let move pl ~dst ~dst_byte ~dst_off ~unit:(unit, count) ~mode ~shape ~axes
    ?(axis = -1) ?src ?(src_off = 0) ?idx ?(idx_off = 0) ?runs ?(d = 0)
    ?(sa = 0) ?(k = 0) () =
  let module M = K.Move_params in
  let n = numel shape in
  if n > 0 then begin
    let p = params "move" in
    ptr p M.dst ~byte:dst_byte dst;
    Option.iter (fun b -> ptr p M.src b) src;
    Option.iter (fun b -> ptr p M.idx b) idx;
    Option.iter
      (fun (b, s, e, m) ->
        ptr p M.starts ~byte:s b;
        ptr p M.ends ~byte:e b;
        ptr p M.maxima ~byte:m b)
      runs;
    i64 p M.numel n;
    i64 p M.dst_off dst_off;
    i64 p M.src_off src_off;
    i64 p M.idx_off idx_off;
    i64 p M.d d;
    i64 p M.sa sa;
    i64 p M.k k;
    let r = Array.length shape in
    i32 p M.rank r;
    i32 p M.axis axis;
    i32 p M.mode mode;
    i32 p M.small (small n);
    i32 p M.unit unit;
    i32 p M.count count;
    let cs = c_strides shape in
    for a = 0 to r - 1 do
      let at = M.ax + (a * K.Move_axis.size) in
      let ss, si = axes a in
      i64 p (at + K.Move_axis.n) shape.(a);
      i64 p (at + K.Move_axis.sd) cs.(a);
      i64 p (at + K.Move_axis.ss) ss;
      i64 p (at + K.Move_axis.si) si
    done;
    launch pl p ~work:(work_of ~unit n)
  end

(* Zeroes [words] 32-bit words of [buf] from byte [byte]. *)
let zero pl buf ~byte ~words =
  move pl ~dst:buf ~dst_byte:byte ~dst_off:0 ~unit:(32, 1) ~mode:K.move_fill
    ~shape:[| words |]
    ~axes:(fun _ -> (0, 0))
    ()

(* Copies [src] into [dst], C-contiguous of its shape. *)
let copy pl ~dst src =
  let ss = strides_of src in
  move pl ~dst:(A.buffer dst) ~dst_byte:0 ~dst_off:(offset_of dst)
    ~unit:(unit_count dst) ~mode:K.move_copy ~shape:(shape_of src)
    ~axes:(fun a -> (ss.(a), 0))
    ~src:(A.buffer src) ~src_off:(offset_of src) ()

let gather ~axis ~dst idx x =
  let pl = plan () in
  let xs = strides_of x and is = strides_of idx in
  move pl ~dst:(A.buffer dst) ~dst_byte:0 ~dst_off:(offset_of dst)
    ~unit:(unit_count dst) ~mode:K.move_gather ~shape:(shape_of dst) ~axis
    ~axes:(fun a -> ((if a = axis then 0 else xs.(a)), is.(a)))
    ~src:(A.buffer x) ~src_off:(offset_of x) ~idx:(A.buffer idx)
    ~idx_off:(offset_of idx)
    ~d:(shape_of x).(axis)
    ~sa:xs.(axis) ();
  finish pl

(* Range gathers *)

(* The runs' launches: each row's starts, ends and their running maximum
   into [ws] from byte 0, [rows] rows of [k], each chunk's maximum
   after. *)
let runs pl ws ~rows ~k ~chunks starts ends =
  let module R = K.Runs_params in
  let lead = L.rank (A.layout starts) - 1 in
  let ss = strides_of starts and es = strides_of ends in
  let sh = shape_of starts in
  let row_bytes = 8 * rows * k in
  let p = params "runs_chunk" in
  ptr p R.in_s (A.buffer starts);
  ptr p R.in_e (A.buffer ends);
  ptr p R.starts ws;
  ptr p R.ends ~byte:row_bytes ws;
  ptr p R.maxima ~byte:(2 * row_bytes) ws;
  ptr p R.carry ~byte:(3 * row_bytes) ws;
  i64 p R.rows rows;
  i64 p R.k k;
  i64 p R.chunks chunks;
  i64 p R.s_k ss.(lead);
  i64 p R.e_k es.(lead);
  i64 p R.s_off (offset_of starts);
  i64 p R.e_off (offset_of ends);
  i32 p R.lead lead;
  i32 p R.small (small rows);
  for a = 0 to lead - 1 do
    let ax = R.ax + (a * K.Runs_axis.size) in
    i64 p (ax + K.Runs_axis.n) sh.(a);
    i64 p (ax + K.Runs_axis.ss) ss.(a);
    i64 p (ax + K.Runs_axis.se) es.(a)
  done;
  launch pl p ~work:(rows * chunks * K.move_threads);
  if chunks > 1 then begin
    launch pl (as_kernel "runs_carry" p) ~work:(rows * K.move_threads);
    launch pl (as_kernel "runs_merge" p) ~work:(rows * k)
  end

let gather_ranges ~dst ~starts ~ends x =
  let pl = plan () in
  let sh = shape_of starts in
  let lead = Array.length sh - 1 in
  let k = sh.(lead) in
  let rows = numel (Array.sub sh 0 lead) in
  let chunks = (k + K.runs_chunk - 1) / K.runs_chunk in
  let row_bytes = 8 * rows * k in
  let ws = workspace pl ((3 * row_bytes) + (8 * rows * chunks)) in
  if rows * k > 0 then runs pl ws ~rows ~k ~chunks starts ends;
  (* Position j of a row's runs is word (row k + j) of each table. *)
  let row_st = c_strides (Array.append (Array.sub sh 0 lead) [| k |]) in
  let xs = strides_of x in
  move pl ~dst:(A.buffer dst) ~dst_byte:0 ~dst_off:(offset_of dst)
    ~unit:(unit_count dst) ~mode:K.move_ranges ~shape:(shape_of dst) ~axis:lead
    ~axes:(fun a ->
      ((if a = lead then 0 else xs.(a)), if a < lead then row_st.(a) else 0))
    ~src:(A.buffer x) ~src_off:(offset_of x)
    ~runs:(ws, 0, row_bytes, 2 * row_bytes)
    ~d:(shape_of x).(lead)
    ~sa:xs.(lead) ~k ();
  finish pl

(* Assemblies *)

(* The bits of [v], an element of [dt], as two little-endian words. *)
let fill_bits (type v s) (dt : (v, s) D.t) (v : v) =
  let s = Nx_kernel.Prog.bits dt v in
  let b = Bytes.make 16 '\000' in
  Bytes.blit_string s 0 b 0 (String.length s);
  (Bytes.get_int64_le b 0, Bytes.get_int64_le b 8)

(* The least and greatest element of [r], and its step's size. *)
let span (r : A.Move.range) =
  let last = r.start + ((r.count - 1) * r.step) in
  (Int.min r.start last, Int.max r.start last, abs r.step)

(* Whether two ranges may share an element: their spans overlap and, where
   their steps are equal, they are of one residue. *)
let may_meet r r' =
  let lo, hi, s = span r and lo', hi', s' = span r' in
  lo <= hi' && lo' <= hi && (s <> s' || (r.start - r'.start) mod s = 0)

let regions_meet a b =
  let ok = ref true in
  Iarray.iteri
    (fun i r -> if not (may_meet r (Iarray.get b i)) then ok := false)
    a;
  !ok

let whole ys (rs : A.Move.range iarray) =
  let ok = ref true in
  Iarray.iteri
    (fun i (r : A.Move.range) ->
      if r.count <> ys.(i) || abs r.step <> 1 then ok := false)
    rs;
  !ok

let assemble ~fill ~dst pieces =
  let pl = plan ~apart:true () in
  let ys = shape_of dst in
  let r = Array.length ys and n = numel ys in
  let module P = K.Assemble_params in
  let pieces =
    Array.of_list
      (List.filter
         (fun (_, x) -> numel (shape_of x) > 0)
         (Iarray.to_list pieces))
  in
  let np = Array.length pieces in
  let rec_words = K.assemble_head + (r * K.assemble_axis) in
  let per = Int.min K.assemble_pieces (K.assemble_words / rec_words) in
  let cs = c_strides ys in
  let unit, count = unit_count dst in
  let disjoint = ref true and total = ref 0 in
  for i = 0 to np - 1 do
    total := !total + numel (shape_of (snd pieces.(i)));
    for j = i + 1 to np - 1 do
      if regions_meet (fst pieces.(i)) (fst pieces.(j)) then disjoint := false
    done
  done;
  let covered =
    Array.exists (fun (rs, _) -> whole ys rs) pieces || (!disjoint && !total = n)
  in
  (* The parameters of [npieces] pieces from [first]. *)
  let table mode first npieces =
    let p = params "assemble" in
    ptr p P.dst (A.buffer dst);
    i64 p P.dst_off (offset_of dst);
    Array.iteri (fun a e -> i64 p (P.dn + (8 * a)) e) ys;
    let f0, f1 = fill_bits (A.dtype dst) fill in
    (* A complex64 fill is two units of 32 bits, a word each. *)
    let f0, f1 =
      if unit = 32 && count = 2 then
        (Int64.logand f0 0xFFFFFFFFL, Int64.shift_right_logical f0 32)
      else (f0, f1)
    in
    u64 p P.fill f0;
    u64 p (P.fill + 8) f1;
    i32 p P.rank r;
    i32 p P.pieces npieces;
    i32 p P.small (small (Int.max n !total));
    i32 p P.unit unit;
    i32 p P.count count;
    i32 p P.mode mode;
    let prefix = ref 0 in
    for i = 0 to npieces - 1 do
      let rs, x = pieces.(first + i) in
      ptr p (P.src + (8 * i)) (A.buffer x);
      let at = P.tab + (8 * i * rec_words) in
      let base = ref 0 in
      Iarray.iteri
        (fun a (g : A.Move.range) -> base := !base + (g.start * cs.(a)))
        rs;
      let later = ref 0 in
      for j = i + 1 to npieces - 1 do
        if regions_meet rs (fst pieces.(first + j)) then
          later := !later lor (1 lsl j)
      done;
      i64 p at (offset_of x);
      i64 p (at + 8) !base;
      i64 p (at + 16) !prefix;
      i64 p (at + 24) !later;
      let xs = shape_of x and xst = strides_of x in
      Iarray.iteri
        (fun a (g : A.Move.range) ->
          let ax = at + (8 * (K.assemble_head + (a * K.assemble_axis))) in
          i64 p ax xs.(a);
          i64 p (ax + 8) xst.(a);
          i64 p (ax + 16) g.start;
          i64 p (ax + 24) g.step;
          i64 p (ax + 32) (g.step * cs.(a)))
        rs;
      prefix := !prefix + numel xs
    done;
    i64 p P.total !prefix;
    (p, !prefix)
  in
  if n > 0 && not covered then begin
    let p, _ = table K.assemble_fill 0 (Int.min np per) in
    i32 p P.check (if np <= per then 1 else 0);
    launch pl p ~work:(work_of ~unit n)
  end;
  let first = ref 0 in
  while !first < np do
    let c = Int.min per (np - !first) in
    let p, elements = table K.assemble_copy !first c in
    launch pl p ~work:elements;
    first := !first + c
  done;
  finish pl

(* Folds *)

let fold (pad : Nx_kernel.pad) ~dst x =
  let pl = plan () in
  let module F = K.Fold_params in
  let module X = K.Fold_axis in
  let ys = shape_of dst and xs = strides_of x in
  let r = Array.length ys and n = numel ys in
  let unit, count = unit_count dst in
  let p = params "fold" in
  ptr p F.dst (A.buffer dst);
  ptr p F.x (A.buffer x);
  i64 p F.numel n;
  i64 p F.dst_off (offset_of dst);
  i64 p F.x_off (offset_of x);
  i32 p F.rank r;
  i32 p F.dtype (code dst);
  i32 p F.small (small n);
  i32 p F.unit unit;
  i32 p F.count count;
  let xshape = shape_of x in
  for a = 0 to r - 1 do
    let at = F.ax + (a * X.size) in
    let lo = Iarray.get pad.lo a and hi = Iarray.get pad.hi a in
    let inter = Iarray.get pad.interior a in
    let pext =
      lo + hi + ys.(a) + if ys.(a) > 0 then (ys.(a) - 1) * inter else 0
    in
    i64 p (at + X.n) ys.(a);
    i64 p (at + X.lo) lo;
    i64 p (at + X.inter) inter;
    i64 p (at + X.pext) pext;
    i64 p (at + X.sx) xs.(a);
    i32 p (at + X.win) (-1)
  done;
  Iarray.iteri
    (fun w (win : A.Move.window) ->
      let at = F.ax + (win.axis * X.size) in
      i64 p (at + X.taps) win.size;
      i64 p (at + X.step) win.step;
      i64 p (at + X.dil) win.dilation;
      i64 p (at + X.nwin) xshape.(win.axis);
      i64 p (at + X.sw) xs.(win.axis);
      i64 p (at + X.st) xs.(r + w);
      i32 p (at + X.win) w)
    pad.windows;
  if n > 0 then launch pl p ~work:n;
  finish pl

(* Scatters *)

(* A scatter's geometry: the updates' shape and strides, idx's strides,
   dst's shape, each array's first element, and the dtype code, unit and
   count of an element. *)
type geometry = {
  us : int array;
  ust : int array;
  ist : int array;
  ys : int array;
  dst_off : int;
  idx_off : int;
  upd_off : int;
  dtype : int;
  unit : int;
  count : int;
}

let geometry ~dst idx u =
  let unit, count = unit_count dst in
  {
    us = shape_of u;
    ust = strides_of u;
    ist = strides_of idx;
    ys = shape_of dst;
    dst_off = offset_of dst;
    idx_off = offset_of idx;
    upd_off = offset_of u;
    dtype = code dst;
    unit;
    count;
  }

(* [g], of complex numbers, as floats: a last axis of their two parts,
   along which idx steps 0. *)
let as_parts g =
  let last a x = Array.append a [| x |] in
  {
    us = last g.us 2;
    ust = last (Array.map (fun s -> 2 * s) g.ust) 1;
    ist = last g.ist 0;
    ys = last g.ys 2;
    dst_off = 2 * g.dst_off;
    idx_off = g.idx_off;
    upd_off = 2 * g.upd_off;
    dtype = (if g.unit = 64 then D.code D.Float64 else D.code D.Float32);
    unit = g.unit;
    count = 1;
  }

(* A scatter kernel's parameters, every field [g] gives. *)
let scatter_params name ~axis ~dst idx u g =
  let module S = K.Scatter_params in
  let p = params name in
  let cs = c_strides g.ys in
  let nu = numel g.us and nd = numel g.ys in
  ptr p S.dst (A.buffer dst);
  ptr p S.idx (A.buffer idx);
  ptr p S.upd (A.buffer u);
  i64 p S.numel_u nu;
  i64 p S.numel_d nd;
  i64 p S.dst_off g.dst_off;
  i64 p S.idx_off g.idx_off;
  i64 p S.upd_off g.upd_off;
  i64 p S.d g.ys.(axis);
  i64 p S.m g.us.(axis);
  i64 p S.sda cs.(axis);
  i32 p S.rank (Array.length g.us);
  i32 p S.axis axis;
  i32 p S.dtype g.dtype;
  i32 p S.small (small (Int.max nu nd));
  i32 p S.unit g.unit;
  i32 p S.count g.count;
  Array.iteri
    (fun a e ->
      let at = S.ax + (a * K.Scatter_axis.size) in
      i64 p (at + K.Scatter_axis.n) e;
      i64 p (at + K.Scatter_axis.sd) cs.(a);
      i64 p (at + K.Scatter_axis.si) g.ist.(a);
      i64 p (at + K.Scatter_axis.su) g.ust.(a))
    g.us;
  p

(* A float Add by the position tree (moves.cu): row tiles where idx steps 0
   along the last axis off the scattered one. It is [false] where it
   declines: its spilled keys would pass 64 bits. *)
let tree_add pl ~axis ~dst idx u g =
  let module S = K.Scatter_params in
  let us = g.us and r = Array.length g.us in
  (* A value of the sum's format: float64, or float32 for the narrow floats
     too. *)
  let vb = if g.unit = 64 then 8 else 4 in
  let real = if vb = 8 then "f64" else "f32" in
  let m = us.(axis) and nd = numel g.ys and d = g.ys.(axis) in
  let rows = axis < r - 1 && g.ist.(r - 1) = 0 && us.(r - 1) > 1 in
  let tile = if rows then K.scatter_rows else K.scatter_tile in
  let last = if rows then us.(r - 1) else 1 in
  let lcols =
    if rows then
      ceil_log2
        (Int.min last
           (Int.min 32 (K.scatter_row_bytes / (K.scatter_rows * vb))))
    else 0
  in
  let nc = (last + (1 lsl lcols) - 1) lsr lcols in
  let run = Int.min tile (1 lsl ceil_log2 m) in
  let group = tile / run in
  let nb = (m + run - 1) / run in
  let slices = numel us / m / last in
  let nsg = (slices + group - 1) / group in
  let tiles = nsg * nb * nc in
  (* A spilled entry's key is its target's index in dst, below [nd], then
     its tile's along the axis, below [nb]: [key_bits] bits. Past 64 the
     call declines. There [nd nb > 2^63]; [into] and dst hold [nd]
     elements each and the updates [nb - 1] tiles of 256 positions or
     more, all of 4 bits or more: at least [nd + 128 nb] bytes, which is
     [2 sqrt(128 2^63) = 2^36] at the least, 64 GiB, past every sm_89
     device's memory, 48 GB at most. *)
  let kb_bits = ceil_log2 nb in
  let key_bits = ceil_log2 nd + kb_bits in
  if nb > 1 && key_bits > 64 then false
  else begin
    let tiler =
      Printf.sprintf "scatter_%s_%s" (if rows then "rows" else "tile") real
    in
    let p = scatter_params tiler ~axis ~dst idx u g in
    i64 p S.slices slices;
    i64 p S.run run;
    i64 p S.group group;
    i64 p S.nb nb;
    i64 p S.nsg nsg;
    i64 p S.tiles tiles;
    i64 p S.last last;
    i64 p S.nc nc;
    i32 p S.lcols lcols;
    if nb = 1 then launch pl p ~work:(tiles * K.move_threads)
    else begin
      (* The workspace's regions, each at a multiple of 8 bytes: the flags
         and the counts, which start zero, then the tiles' entries, [est]
         a tile, the sort's keys and indices, the pieces' sums and the
         histograms. nx_cuda.mli's bound: an entry takes 41 bytes and two
         rows of [2^lcols] values; past one tile the tiles hold fewer than
         twice a slice's positions, so fewer than [2 slices nc m] entries,
         at most [82 + 4 w] bytes an update whether a row holds one value,
         [last] (complex parts included) or [2^lcols] of more; [ent_n]
         adds under a byte an update, the counts, the histograms' rounding
         and the alignment 5 KiB at most. *)
      let est = Int.min tile (group * d) in
      let ents = tiles * est in
      let rowb = vb lsl lcols in
      let flags = if rows then nd / last * nc else nd in
      let up8 x = (x + 7) land lnot 7 in
      let regions =
        [
          (S.tiles_of, up8 (4 * flags));
          (S.ctl, 8 * 257);
          (S.ent_g, 8 * ents);
          (S.ent_v, up8 (rowb * ents));
          (S.ent_n, up8 (4 * tiles));
          (S.keys, 8 * ents);
          (S.keys + 8, 8 * ents);
          (S.index, 8 * ents);
          (S.index + 8, 8 * ents);
          (S.piece, up8 (rowb * ents));
          (S.hist, 8 * 256 * ((ents + K.scatter_chunk - 1) / K.scatter_chunk));
        ]
      in
      let size = List.fold_left (fun o (_, bytes) -> o + bytes) 0 regions in
      let ws = workspace pl size in
      ignore
        (List.fold_left
           (fun o (field, bytes) ->
             ptr p field ~byte:o ws;
             o + bytes)
           0 regions);
      i64 p S.est est;
      i32 p S.kb_bits kb_bits;
      i32 p S.key_bits key_bits;
      zero pl ws ~byte:0 ~words:((up8 (4 * flags) + (8 * 257)) / 4);
      launch pl p ~work:(tiles * K.move_threads);
      launch pl
        (as_kernel ("scatter_spill_" ^ real) p)
        ~work:(tiles * K.move_threads);
      let chunks = (ents + K.scatter_chunk - 1) / K.scatter_chunk in
      let shift = ref 0 in
      while !shift < key_bits do
        let q = again p in
        i32 q S.shift !shift;
        launch pl (as_kernel "scatter_hist" q) ~work:(chunks * K.move_threads);
        launch pl (as_kernel "scatter_scan" q) ~work:(256 * K.move_threads);
        launch pl (as_kernel "scatter_place" q) ~work:(chunks * K.move_threads);
        shift := !shift + 8
      done;
      launch pl (as_kernel ("scatter_tree1_" ^ real) p) ~work:(ents lsl lcols);
      launch pl (as_kernel ("scatter_tree2_" ^ real) p) ~work:(ents lsl lcols)
    end;
    true
  end

let scatter (c : Nx_kernel.combine) ~unique ~axis ~dst ~into idx u =
  let module S = K.Scatter_params in
  let pl = plan () in
  let identical =
    A.buffer into == A.buffer dst && L.equal (A.layout into) (A.layout dst)
  in
  if not identical then copy pl ~dst into;
  let g = geometry ~dst idx u in
  let nu = numel g.us and nd = numel g.ys in
  let float = D.is D.Float (A.dtype dst) || D.is D.Complex (A.dtype dst) in
  let pass ?(f = ignore) op =
    let p = scatter_params "scatter" ~axis ~dst idx u g in
    f p;
    i32 p S.op op;
    launch pl p ~work:nu
  in
  (* A buffer of [words] zero 32-bit words in the workspace, for each
     target's owner or lock. *)
  let zeros words =
    let ws = workspace pl (4 * words) in
    zero pl ws ~byte:0 ~words;
    ws
  in
  let computed =
    nu = 0 || nd = 0
    ||
    match c with
    | Set when unique ->
        pass K.scatter_set;
        true
    | Set ->
        (* An owner is a position plus one, 0 for none: 32 bits hold it
           up to 2^32 - 1 positions. *)
        let wide = g.us.(axis) > 0xFFFFFFFF in
        let ws = zeros (if wide then 2 * nd else nd) in
        let own p =
          ptr p S.owner ws;
          i32 p S.wide (if wide then 1 else 0)
        in
        pass ~f:own K.scatter_own;
        pass ~f:own K.scatter_owned;
        true
    | Max | Min ->
        let op = if c = Max then K.scatter_max else K.scatter_min in
        if g.count = 2 then begin
          let ws = zeros nd in
          pass ~f:(fun p -> ptr p S.locks ws) op
        end
        else pass op;
        true
    | Add when not float ->
        pass K.scatter_add_int;
        true
    | Add when unique ->
        pass K.scatter_add_unique;
        true
    | Add ->
        tree_add pl ~axis ~dst idx u (if g.count = 2 then as_parts g else g)
  in
  if computed then finish pl else Error A.Declined
