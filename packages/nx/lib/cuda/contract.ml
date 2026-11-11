(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A contraction's launches of contract.cu and mma.cuh, as one submission:

     packs (a, b, init, ids) --> per window of one launch's bounds:
     a segmented call's prepass, then the contraction --> a pack of y out
     of the workspace

   The plan reads each operand's axes as runs, one run of strides for its
   batch axes, one for its free ones and one for its contracted ones,
   where its layout merges them. It packs into the workspace:

   - an operand whose axes of one kind do not form one run of strides, of
     a sub-byte dtype, scaled, a float8 one of the matrix units, or one of
     the accumulator's own dtype or of the matrix units whose rows are not
     16-byte vectors along a contiguous axis;
   - an unsegmented integer call's operand of another dtype than the
     native tiles read, where the call has more rows than gemv takes, cast
     once into the compute type;
   - a matrix routine's negated a, as its values negated;
   - a complex call's a not of the accumulator's dtype with its
     contracted axes contiguous, and its b, as each element's 2 x 2 real
     block [[re, -im], [im, re]];
   - an init or a y of another dtype than the accumulator's, but bfloat16
     and float16 on the matrix units and a y's integer dtype no wider than
     an integer accumulator, or of axes that do not form one run of
     strides, or a complex one whose columns are apart.

   An operand is copied in C order of its groups (a sub-byte dtype one
   element a byte), or, for a scale, multiplied into the accumulator's
   dtype; a y is computed into the workspace and packed out. Every pack
   copies values exactly or rounds as the call would, so packs change no
   bit. int8 operands into int32 sum on the matrix units, exactly.

   The association is fixed by the dtypes and the extents alone: the
   kernel's family by the dtypes (and scales and segments), its tiles and
   its ranges of k by the model below, which reads the extents and the
   dtypes' widths, never a layout. A layout only chooses whether an
   operand loads in place or through a pack. *)

module K = Kernels
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Sub = Rig.Submission
module Run = Rig.Submission.Run

type call = Bytes.t Launch.call

(* Dtypes *)

let code (D.Any d) = D.code d
let bits (D.Any d) = D.bits d
let sub_byte dt = bits dt < 8

(* The bytes an element takes in the kernels' operands: a sub-byte one's,
   one a byte, 1. *)
let width dt = Int.max 1 (bits dt / 8)
let is_complex (D.Any d) = D.is Complex d
let is_f8 = function D.Any (Float8_e4m3fn | Float8_e5m2) -> true | _ -> false

(* The FMA kernels' compute type of an accumulator, and whether it is
   complex: narrower integers wrap in 32 bits, then to their width. *)
let ct_of (D.Any d) : K.ct * bool =
  match d with
  | Float32 -> (F32, false)
  | Complex64 -> (F32, true)
  | Float64 -> (F64, false)
  | Complex128 -> (F64, true)
  | Int64 | Uint64 -> (U64, false)
  | _ -> (U32, false)

let ct_bytes : K.ct -> int = function F32 | U32 -> 4 | F64 | U64 -> 8
let ct_float : K.ct -> bool = function F32 | F64 -> true | U32 | U64 -> false

let ct_dtype : K.ct -> D.any = function
  | F32 -> D.Any Float32
  | F64 -> D.Any Float64
  | U32 -> D.Any Uint32
  | U64 -> D.Any Uint64

(* Whether an operand of [dt] loads natively into [ct]: its own bits. *)
let native ct (D.Any d) =
  match ((ct : K.ct), d) with
  | F32, Float32 | F64, Float64 -> true
  | U32, (Int32 | Uint32) | U64, (Int64 | Uint64) -> true
  | _ -> false

(* Runs *)

(* The axes [axes] of [l] as one run, (extent, stride), the stride 0
   along at most one element; [None] where they make several. An axis
   merges into the one before it where that one's stride is its stride
   times its extent. *)
let run l (axes : int array) =
  let extent = ref 1 and stride = ref 0 and runs = ref 0 in
  for i = 0 to Array.length axes - 1 do
    let e = L.dim l axes.(i) in
    if e = 0 then extent := 0
    else if e > 1 then begin
      let s = L.stride l axes.(i) in
      if !runs = 0 || !stride <> s * e then incr runs;
      stride := s;
      extent := !extent * e
    end
  done;
  if !extent = 0 then Some (0, 0)
  else if !runs <= 1 then Some (!extent, if !extent > 1 then !stride else 0)
  else None

(* An array's axes of its three groups, in order, as a pack walks them:
   their extents, strides and groups (0 to 2, and 3 for a Contracting
   y's groups axis). *)
type axes = { ext : int array; str : int array; grp : int array }

let axes_of l (groups : int array array) =
  let all = Array.concat (Array.to_list groups) in
  let grp =
    Array.concat
      (Array.to_list
         (Array.mapi (fun g a -> Array.make (Array.length a) g) groups))
  in
  { ext = Array.map (L.dim l) all; str = Array.map (L.stride l) all; grp }

(* The elements of [a] where [keep] holds. *)
let kept keep a =
  Array.of_list (List.filteri (fun i _ -> keep.(i)) (Array.to_list a))

(* Views *)

(* Where a kernel reads an array: its own slot and buffer, or the
   workspace. *)
type where = Own of { slot : int; buf : Rig.Buffer.t } | Work

(* An array as a kernel reads it: where, its first element's byte, its
   strides along its three groups in elements, its dtype, and, for a Rows
   call's b packed into the workspace, the stride of its slices. *)
type view = { where : where; at : int; s : int array; dt : D.any; sg : int }

(* The plan's call: the extents, and for each array its layout's runs, or
   [None] where a group does not merge. *)
type side = {
  arr : A.any;
  slot : int;
  groups : int array array; (* batch, free, contracted (y: batch, rows, cols) *)
  runs : (int * int) option array;
}

let side_of (A.Any x as arr) slot groups =
  let l = A.layout x in
  { arr; slot; groups; runs = Array.map (run l) groups }

let layout (A.Any x) = A.layout x
let dtype_of (A.Any x) = D.Any (A.dtype x)
let buffer (A.Any x) = A.buffer x
let grouped s = Array.for_all Option.is_some s.runs
let stride_of s g = match s.runs.(g) with Some (_, st) -> st | None -> 0

let view_of s =
  let l = layout s.arr in
  {
    where = Own { slot = s.slot; buf = buffer s.arr };
    at = L.offset l * width (dtype_of s.arr);
    s = Array.init 3 (stride_of s);
    dt = dtype_of s.arr;
    sg = 0;
  }

(* The builder of a call: its slots, launches and workspace, and the
   buffers it made. *)
type builder = {
  mutable slots : Rig.Buffer.t list;
  mutable frees : Rig.Buffer.t list;
  mutable nslots : int;
  mutable launches : Bytes.t Launch.t list;
  ws : Arena.t;
  mutable ticket_bytes : int;
  mutable windowed : bool;
}

let builder () =
  {
    slots = [];
    frees = [];
    nslots = 0;
    launches = [];
    ws = Arena.make ();
    ticket_bytes = 0;
    windowed = false;
  }

(* A buffer's slot: the one it already has, so that a call of many
   launches over a few arrays, a routine's, names each once. *)
let slot b buf =
  let rec find i = function
    | [] -> None
    | x :: rest -> if x == buf then Some i else find (i - 1) rest
  in
  match find (b.nslots - 1) b.slots with
  | Some s -> s
  | None ->
      b.slots <- buf :: b.slots;
      b.nslots <- b.nslots + 1;
      b.nslots - 1

let fresh b d n =
  let buf = Rig.Buffer.create d n in
  b.frees <- buf :: b.frees;
  buf

(* The workspace and the tickets: slots [-1] and [-2] until the call is
   finished, which knows their bytes. *)
let ws_slot = -1
let tickets_slot = -2
let take b bytes = Arena.take b.ws bytes
let mark b = Arena.mark b.ws
let release b m = Arena.release b.ws m
let ceil_div a b = (a + b - 1) / b

(* A scale as a pack reads it: its slot, first element, strides and the
   ratios of its operand's extents to its own along each of the operand's
   axes, its dtype, and whether it is an Exponent. *)
type scale = {
  sslot : int;
  sfirst : int;
  cs : int array;
  ratio : int array;
  sdt : D.any;
  exponent : bool;
}

let scale_of sslot (A.Any x) (s : Nx_kernel.scale) =
  let arr, sdt, exponent =
    match s with
    | Exponent e -> (A.Any e, D.Any D.Uint8, true)
    | Value v -> (A.Any v, D.Any (A.dtype v), false)
  in
  let l = layout arr and lx = A.layout x in
  {
    sslot;
    sfirst = L.offset l;
    cs = Array.init (L.rank l) (L.stride l);
    ratio =
      Array.init (L.rank l) (fun i ->
          let d = L.dim l i in
          if d = 0 then 1 else L.dim lx i / d);
    sdt;
    exponent;
  }

(* Parameter words *)

(* Parameters live in bytes, whose setters box nothing: a call allocates
   as many words whatever its extents. *)
let params_of size = Bytes.make size '\000'
let set64 w at v = Bytes.set_int64_le w at (Int64.of_int v)
let set32 w at v = Bytes.set_int32_le w at (Int32.of_int v)

(* A launch's address field at [at]: the byte [off] of [where]'s slot. *)
let address refs w at slot off ~write =
  set64 w at off;
  (* Reversed: [launch] reverses the list into (at, slot, write) triples. *)
  refs := (if write then 1 else 0) :: slot :: at :: !refs

let slot_of = function Own o -> o.slot | Work -> ws_slot

let launch b ~kernel ?(gy = 1) ?(gz = 1) ~gx ~threads ~shared params refs =
  b.launches <-
    {
      Launch.kernel;
      size = Bytes.length params;
      base = 0;
      refs = Array.of_list (List.rev !refs);
      gx;
      gy;
      gz;
      threads;
      shared;
      params;
    }
    :: b.launches

let count = Array.length K.kernels
let pack_kernel = Launch.kernel "contract_pack"
let rows_kernel = Launch.kernel "contract_rows"
let ks_kernel = Launch.kernel "contract_ks"

(* Packs *)

(* A pack of the axes [ax] of [src_slot], of [src_dt] from its element
   [src_first], into [dst_where] from its byte [dst_at] and its element
   [dst_first], of [dst_dt], [dst_str] the strides of each axis there.
   [scale] is a scale, with its strides and ratios along [ax]. *)
let pack b ~mode ~src_slot ~src_dt ~src_first ~dst_where ~dst_at ~dst_dt
    ~dst_first ~acc (ax : axes) dst_str ?scale ?(block = 0) () =
  let numel = Array.fold_left ( * ) 1 ax.ext in
  if numel > 0 then begin
    let module P = K.Pack_params in
    let w = params_of P.size and refs = ref [] in
    address refs w P.src src_slot 0 ~write:false;
    address refs w P.dst (slot_of dst_where) dst_at ~write:true;
    set64 w P.src_first src_first;
    set64 w P.dst_first dst_first;
    set64 w P.numel numel;
    set32 w P.src_dtype (code src_dt);
    set32 w P.dst_dtype (code dst_dt);
    set32 w P.scale_dtype (-1);
    set32 w P.acc_dtype (code acc);
    set64 w P.block block;
    (* The axes of more than one element, in order. *)
    let r = ref 0 in
    for i = 0 to Array.length ax.ext - 1 do
      if ax.ext.(i) > 1 then begin
        let at = !r in
        set64 w (P.n + (8 * at)) ax.ext.(i);
        set64 w (P.ss + (8 * at)) ax.str.(i);
        set64 w (P.ds + (8 * at)) dst_str.(i);
        (match scale with
        | None -> set32 w (P.ratio + (4 * at)) 1
        | Some (_, (cs : int array), (ratio : int array)) ->
            set64 w (P.cs + (8 * at)) cs.(i);
            set32 w (P.ratio + (4 * at)) ratio.(i));
        incr r
      end
    done;
    set32 w P.rank !r;
    let mode =
      match scale with
      | None -> mode
      | Some (sc, _, _) ->
          address refs w P.scale sc.sslot 0 ~write:false;
          set64 w P.scale_first sc.sfirst;
          set32 w P.scale_dtype (code sc.sdt);
          if sc.exponent then mode lor K.pack_exponent else mode
    in
    set32 w P.mode mode;
    let gx = Int.max 1 (Int.min 65535 (ceil_div numel 256)) in
    launch b ~kernel:pack_kernel ~gx ~threads:256 ~shared:0 w refs
  end

(* The strides of a contiguous layout of [ax] whose groups have the
   strides [gs] (batch, free, k, and the groups axis), each group's axes
   in C order within it. *)
let packed_strides (ax : axes) (gs : int array) =
  let n = Array.length ax.ext in
  let st = Array.make n 0 and inner = Array.make 4 1 in
  for i = n - 1 downto 0 do
    let g = ax.grp.(i) in
    st.(i) <- gs.(g) * inner.(g);
    inner.(g) <- inner.(g) * ax.ext.(i)
  done;
  st

(* The copies of an element a packed row holds where the operand repeats
   it along k: the most k a k-tile of any kernel reads, an mma tile's
   bytes of int8, so that a kernel reads them from the row's start at
   every k, the row's k stride 0. *)
let k_copies = Array.fold_left (fun n t -> Int.max n (K.shape t).bkb) 16 K.tiles

(* Whether [s]'s group [g] repeats one element: its axes of more than one
   element are all of stride 0, and [scale], where given, has one element
   along them. *)
let repeats s ?scale g =
  match s.runs.(g) with
  | Some (e, 0) when e > 1 -> (
      match scale with
      | None -> true
      | Some sc ->
          let l = layout s.arr in
          Array.for_all
            (fun ax -> L.dim l ax <= 1 || sc.ratio.(ax) = L.dim l ax)
            s.groups.(g))
  | _ -> false

(* An operand a or b packed into the workspace as [into], [batch] x
   [rows] x [lead] elements, [lead] the contracted extent [k] padded to
   16 bytes, after the slices of a Rows call's b along its fourth group;
   with [value], its values cast into [acc], times its scale. A group that
   repeats one element is packed once, its stride 0: batch elements or
   rows as one, and a k past [k_copies] as [k_copies] copies. *)
let pack_operand b s ~into ~batch ~rows ~k ~acc ~value ?scale ?(neg = false) ()
    =
  let w = width into in
  let one g = repeats s ?scale g in
  (* A complex operand's parts lie along k, so its k is never one copy. *)
  let rb = one 0
  and rr = one 1
  and rk = one 2 && k > k_copies && not (is_complex into) in
  let lead = if rk then k_copies else ceil_div k (16 / w) * (16 / w) in
  let batch' = if rb then 1 else batch and rows' = if rr then 1 else rows in
  let ax = axes_of (layout s.arr) s.groups in
  let slices = ref 1 in
  Array.iteri (fun i g -> if g = 3 then slices := !slices * ax.ext.(i)) ax.grp;
  let sg = batch' * rows' * lead in
  let at = take b (!slices * sg * w) in
  let all = Array.concat (Array.to_list s.groups) in
  (* The axes the pack walks: the operand's but those of a group packed
     once, and for a repeated k one axis of its copies, read at stride 0. *)
  let keep =
    Array.map
      (fun g -> not ((g = 0 && rb) || (g = 1 && rr) || (g = 2 && rk)))
      ax.grp
  in
  let pick = kept keep in
  let copies a x = if rk then Array.append a [| x |] else a in
  let walked =
    {
      ext = copies (pick ax.ext) k_copies;
      str = copies (pick ax.str) 0;
      grp = copies (pick ax.grp) 2;
    }
  in
  let scale =
    Option.map
      (fun sc ->
        ( sc,
          copies (pick (Array.map (fun i -> sc.cs.(i)) all)) 0,
          copies (pick (Array.map (fun i -> sc.ratio.(i)) all)) 1 ))
      scale
  in
  let src_dt = dtype_of s.arr in
  let mode =
    (if value then K.pack_value else 0)
    lor (if neg then K.pack_neg else 0)
    lor if sub_byte src_dt && not value then K.pack_spread_dst else 0
  in
  let gs =
    [| (if rb then 0 else rows' * lead); (if rr then 0 else lead); 1; sg |]
  in
  pack b ~mode ~src_slot:s.slot ~src_dt
    ~src_first:(L.offset (layout s.arr))
    ~dst_where:Work ~dst_at:at ~dst_dt:into ~dst_first:0 ~acc walked
    (packed_strides walked gs) ?scale ();
  {
    where = Work;
    at;
    s = [| gs.(0); gs.(1); (if rk then 0 else 1) |];
    dt = into;
    sg;
  }

(* A complex call as the real contraction of its parts: a's (re, im) at
   k' = 2p, 2p + 1, b's as the 2 x 2 block [[re, -im], [im, re]] of rows
   2j, 2j + 1 and k' = 2p, 2p + 1, y's and init's parts neighbouring
   columns, so that the sums take the contract's numbered terms in order.
   [real v] is the real view of a complex [v] whose last axis is
   contiguous, of the real dtype [rdt]. *)
let real (v : view) rdt =
  { v with s = [| 2 * v.s.(0); 2 * v.s.(1); 1 |]; dt = rdt }

(* b of a complex call as its real blocks in the workspace, [2 rows] x
   [lead] reals a batch element after the slices of a Rows call's b, [lead]
   [2 k] padded to 16 bytes, each element's value cast into [acc]. *)
let pack_block ?(mode = 0) b s ~rdt ~batch ~rows ~k ~acc =
  let per = 16 / width rdt in
  let lead = ceil_div (2 * k) per * per in
  let ax = axes_of (layout s.arr) s.groups in
  let slices = ref 1 in
  Array.iteri (fun i g -> if g = 3 then slices := !slices * ax.ext.(i)) ax.grp;
  let sg = batch * 2 * rows * lead in
  let at = take b (!slices * sg * width rdt) in
  pack b
    ~mode:(K.pack_value lor K.pack_block lor mode)
    ~src_slot:s.slot ~src_dt:(dtype_of s.arr)
    ~src_first:(L.offset (layout s.arr))
    ~dst_where:Work ~dst_at:at ~dst_dt:rdt ~dst_first:0 ~acc ax
    (packed_strides ax [| 2 * rows * lead; 2 * lead; 2; sg |])
    ~block:lead ();
  { where = Work; at; s = [| 2 * rows * lead; lead; 1 |]; dt = rdt; sg }

(* An init or y as a contiguous [groups; batch; rows; cols] in the
   workspace, its dtype's elements of a byte or more, a sub-byte one's one
   a byte. *)
let out_strides ~batch ~m ~n = [| m * n; n; 1; batch * m * n |]

(* The model *)

(* An instance as the model reads it: its kernel, compute type, load
   (gemv's 3), tile, threads, the blocks an SM holds and shared bytes. *)
type inst = {
  kernel : int;
  t : K.ct;
  ld : int;
  bm : int;
  bn : int;
  bk : int;
  tm : int;
  tn : int;
  threads : int;
  occ : int;
  smem : int;
}

let load_code : K.load -> int = function Native -> 0 | Convert -> 1

let insts =
  let l = ref [] in
  Array.iteri
    (fun i (_, inst) ->
      match inst with
      | K.Gemm (g : K.gemm) ->
          let v = 16 / ct_bytes g.t in
          l :=
            {
              kernel = i;
              t = g.t;
              ld = load_code g.load;
              bm = g.bm;
              bn = g.bn;
              bk = g.bk;
              tm = g.tm;
              tn = g.tn;
              threads = g.wm * g.wn * 32;
              occ = g.occ;
              smem = 2 * g.bk * (g.bm + v + g.bn + v) * ct_bytes g.t;
            }
            :: !l
      | Gemv t ->
          (* The blocks an SM holds by ptxas's registers, 104 to 128;
             gemv_ranges weighs its shared memory. *)
          l :=
            {
              kernel = i;
              t;
              ld = 3;
              bm = K.gemv_rows;
              bn = K.gemv_threads;
              bk = 16;
              tm = K.gemv_rows;
              tn = 1;
              threads = K.gemv_threads;
              occ = 2;
              smem = 0;
            }
            :: !l
      | _ -> ())
    K.kernels;
  Array.of_list (List.rev !l)

let sms = 100

let ranges ~rows ~cols ~k =
  let tiles = ceil_div rows 64 * ceil_div cols 64 in
  if tiles < sms && k >= 2048 then
    Int.min (ceil_div k 512) (ceil_div (2 * sms) tiles)
  else 1

(* The time, in units of an SM's multiply-adds, of [n1] blocks of work
   [w1] then [n2] of [w2] on [sms] SMs of [c] blocks each, an SM's rate
   shared by its blocks and reached from [sat] of them: whole waves of the
   first, then the rest spread over the SMs. *)
(* An SM's rate holding [a] blocks, reached from [sat] of them. *)
let[@inline] share sat a = Float.min 1. (Float.of_int a /. sat)

let[@inline] makespan n1 w1 n2 w2 c sat =
  let cap = sms * c in
  let waves = n1 / cap in
  let t = Float.of_int (waves * c) *. w1 /. share sat c in
  let r1 = n1 - (waves * cap) in
  let blocks = r1 + n2 in
  if blocks = 0 then t
  else
    let work = (Float.of_int r1 *. w1) +. (Float.of_int n2 *. w2) in
    let wmax =
      Float.max (if r1 > 0 then w1 else 0.) (if n2 > 0 then w2 else 0.)
    in
    if blocks <= cap then
      let a = ceil_div blocks sms in
      t
      +. Float.max
           (Float.of_int a *. work /. Float.of_int blocks /. share sat a)
           (wmax /. share sat 1)
    else
      t
      +. (work /. (Float.of_int sms *. share sat c))
      +. (0.5 *. wmax /. share sat 1)

(* An SM's multiply-adds a microsecond. DFMA: 2 a clock at 2.4 GHz.
   FFMA: 128 a clock at an effective 1.65 GHz, times the instance's share
   of its loop: a k-tile's FMAs against its fragment loads and some 80
   other instructions, more for a converting load. Integers: 64 a clock, a
   quarter for 64 bits. *)
let[@inline] sm_rate i =
  let b = ct_bytes i.t and fl = ct_float i.t in
  let v = 16 / b in
  let rm = Int.min i.tm v and cn = Int.min i.tn v in
  let fma = Float.of_int (i.tm * i.tn * i.bk) in
  let lds = Float.of_int (i.bk * (ceil_div i.tm rm + ceil_div i.tn cn)) in
  let other =
    80.
    +.
    if i.ld > 0 then
      3. *. Float.of_int ((i.bm + i.bn) * i.bk) /. Float.of_int i.threads
    else 0.
  in
  let share = fma /. (fma +. lds +. other) in
  if not fl then 64. *. 1650. *. share *. if b = 8 then 0.25 else 1.
  else if b = 8 then 2. *. 2400.
  else 128. *. 1650. *. share

(* The extents and widths the model reads: a call's batch, rows, its
   compute columns and terms (doubled for a complex accumulator), and its
   a's, b's and y's bytes an element. *)
type shape = {
  batch : int;
  m : int;
  n : int;
  k : int;
  cx : bool;
  wa : float;
  wb : float;
  wy : float;
}

(* The predicted microseconds of [c] on [i] in [tm] x [tn] tiles a batch
   element, its first [main] tiles in [s1] ranges and the others in
   [s2]. *)
let[@inline] cost c i ~tm ~tn ~main ~s1 ~s2 =
  let tiles = c.batch * tm * tn in
  let kk = Float.of_int (Int.max c.k 1) in
  let w = Float.of_int (i.bm * i.bn) *. kk in
  let tb = ct_bytes i.t in
  let sat =
    Float.max 1.
      ((if tb = 8 && ct_float i.t then 128. else 256.) /. Float.of_int i.threads)
  in
  let rate = sm_rate i in
  let fixed = 0.5 *. rate /. Float.of_int i.occ in
  let t =
    makespan (main * s1)
      ((w /. Float.of_int s1) +. fixed)
      ((tiles - main) * s2)
      ((w /. Float.of_int s2) +. fixed)
      i.occ sat
    /. rate
  in
  let smax =
    Int.max (if main > 0 then s1 else 1) (if tiles > main then s2 else 1)
  in
  let t =
    t +. 1.5
    +. if smax > 1 then Float.of_int (smax * i.bm * i.bn * tb) /. 1.6e5 else 0.
  in
  let kr = Float.of_int (if c.cx then c.k / 2 else c.k) in
  let nr = Float.of_int (if c.cx then c.n / 2 else c.n) in
  let batch = Float.of_int c.batch and m = Float.of_int c.m in
  let ab =
    (c.wa *. batch *. m *. kr)
    +. (c.wb *. batch *. nr *. kr)
    +. (c.wy *. batch *. m *. nr)
  in
  let part =
    2.
    *. Float.of_int (tb * i.bm * i.bn)
    *. Float.of_int
         ((if s1 > 1 then main * s1 else 0)
         + if s2 > 1 then (tiles - main) * s2 else 0)
  in
  let dram =
    if ab <= 48e6 then ab /. 2.6e6
    else if ab <= 72e6 then ab /. 1.0e6
    else ab /. 0.53e6
  in
  let bn = Float.of_int (if c.cx then i.bn / 2 else i.bn) in
  let l2 =
    ((Float.of_int tiles *. kr *. ((Float.of_int i.bm *. c.wa) +. (bn *. c.wb)))
    +. part)
    /. 2.6e6
  in
  Float.max t (Float.max dram l2 +. 1.5) +. 1.5

(* A plan of the FMA kernels: the instance, its tiles a batch element, and
   its schedule. *)
type plan = {
  i : inst;
  tiles_m : int;
  tiles_n : int;
  tiles : int;
  main : int;
  s1 : int;
  s2 : int;
}

let blocks p = (p.main * p.s1) + ((p.tiles - p.main) * p.s2)

(* NX_GEMV_STAGE: gemv's bytes a warp for b's values in a compute type of
   [t] bytes, from nx_cuda_stubs.c. *)
external gemv_stage : int -> int = "nx_cuda_gemv_stage" [@@noalloc]

(* gemv's shared bytes for [m] rows: a's stretch, and each warp's rows of
   b's values. *)
let gemv_shared ct ~m =
  (ceil_div (m * K.gemv_stretch * ct_bytes ct) 16 * 16)
  + (K.gemv_threads / 32 * gemv_stage (ct_bytes ct))

(* gemv's ranges of k for [tiles] tiles of [runs] runs on [i]: one wave of
   the blocks the SMs hold at once, by registers ([i.occ]) and shared
   memory (100 KiB an SM, 1 KiB of it a block's), but where b, past 128 MB,
   loads less than a 128-byte line of a column a step: there as many as
   the runs allow up to 64, so that the blocks an SM holds at once read
   the lines of few columns, which keeps memory's pages open. Measured on
   the RTX 5000 Ada: float32 1x5120x2880 21.8 us at 5 and 10 ranges, 35.7
   at 64; float64 8x5120x2880, one block an SM, 282 us at 5, 371 at 7;
   20480x1x2880 (on gemv transposed) 752 us at 5, 485 at 64. *)
let gemv_ranges i c ~tiles ~runs =
  let bytes = c.wb *. Float.of_int (c.batch * c.n) *. Float.of_int c.k in
  let dram = bytes > 128e6 && c.wb *. 16. < 128. in
  let occ = Int.min i.occ (102400 / (gemv_shared i.t ~m:c.m + 1024)) in
  Int.max 1
    (Int.min 64
       (Int.min runs (if dram then runs else sms * Int.max 1 occ / tiles)))

(* The plan of [ct]'s gemv where [gemv], in gemv_ranges' ranges; else the
   plan of least predicted time among the instances of [ld] of [ct]: tiles
   of fewer than 16 rows for as few rows alone, and calls of 8 rows or
   fewer on those where the load has them; for each, k cut into s ranges,
   then the tiles past the last whole wave of split tiles cut again to
   fill a wave. A function of the shape and the dtypes alone. With [only],
   that instance alone. *)
let choose ?only ct ld ~gemv c =
  (* The best so far as unboxed numbers: a float array's float and ints,
     so that the search allocates nothing however many plans it weighs. *)
  let least = Float.Array.make 1 Float.infinity in
  let best = ref (-1)
  and best_s = ref 1
  and best_main = ref 0
  and best_s2 = ref 1 in
  let skinny i' = i'.ld = ld && i'.t = ct && i'.bm <= 8 in
  let has_skinny = Array.exists skinny insts in
  let runs = Int.max 1 (ceil_div c.k K.contract_run) in
  for x = 0 to Array.length insts - 1 do
    let i = insts.(x) in
    let ok =
      match only with
      | Some o -> o == i
      | None ->
          if i.ld = 3 then gemv && i.t = ct
          else
            (not gemv) && i.ld = ld && i.t = ct
            && (i.bm >= 16 || c.m <= i.bm)
            && not (c.m <= 8 && i.bm > 8 && has_skinny)
    in
    if ok && i.ld = 3 then begin
      let tiles = c.batch * ceil_div c.n i.bn in
      best := x;
      best_s := gemv_ranges i c ~tiles ~runs;
      best_main := tiles;
      best_s2 := 1
    end
    else if ok then begin
      let tm = ceil_div c.m i.bm and tn = ceil_div c.n i.bn in
      let tiles = c.batch * tm * tn in
      let cap = sms * i.occ in
      for s = 1 to Int.min 64 runs do
        let waves = tiles * s / cap in
        let main = waves * cap / s in
        let rest = tiles - main in
        let s2 =
          if rest > 0 then Int.min 64 (Int.min runs (Int.max s (cap / rest)))
          else s
        in
        let main, s2 = if rest = 0 || s2 = s then (tiles, 1) else (main, s2) in
        let t = cost c i ~tm ~tn ~main ~s1:s ~s2 in
        if t < Float.Array.get least 0 then begin
          Float.Array.set least 0 t;
          best := x;
          best_s := s;
          best_main := main;
          best_s2 := s2
        end
      done
    end
  done;
  if !best < 0 then invalid_arg "Contract.choose: no instance";
  let i = insts.(!best) in
  let tiles_m = ceil_div c.m i.bm and tiles_n = ceil_div c.n i.bn in
  {
    i;
    tiles_m;
    tiles_n;
    tiles = c.batch * tiles_m * tiles_n;
    main = !best_main;
    s1 = !best_s;
    s2 = !best_s2;
  }

(* The matrix units *)

(* The mma kind of operands of [a] and [b] into [acc], if one sums them.
   float8 operands decode exactly to bfloat16 and sum on its unit: Ada's
   float8 unit keeps 13 bits of its sums. *)
let mma_kind a b acc : K.kind option =
  match (acc, a, b) with
  | ( D.Any Float32,
      D.Any (Bfloat16 | Float8_e4m3fn | Float8_e5m2),
      D.Any (Bfloat16 | Float8_e4m3fn | Float8_e5m2) ) ->
      Some Bf16
  | D.Any Float32, D.Any Float16, D.Any Float16 -> Some F16
  | D.Any Int32, D.Any Int8, D.Any Int8 -> Some S8
  | _ -> None

let rec find_mma_from k kind a b t =
  if k = count then -1
  else
    match snd K.kernels.(k) with
    | Mma (kind', a', b', t')
      when (kind' = kind || kind' = K.Any) && a' = a && b' = b && t' = t ->
        k
    | _ -> find_mma_from (k + 1) kind a b t

let find_mma kind a b t = find_mma_from 0 kind a b t

(* The mma tiles' efficiency, against the 128 x 256 tile's, on sm_89. *)
let efficiency : K.tile -> int = function
  | T128x128 -> 90
  | T128x256 -> 100
  | T64x64 -> 60
  | T16x64 -> 30

(* The mma tile of least cost among those [kind] has an instance of with
   k contiguous, by the extents alone, each tile [splits] blocks; 16 rows
   or fewer take the 16-row tile, which every kind has. *)
let mma_tile ~batch ~m ~n ~splits kind =
  if m <= 16 then K.T16x64
  else begin
    (* A float array's float: the search allocates nothing. *)
    let best = ref K.T16x64 and least = Float.Array.make 1 Float.infinity in
    for x = 0 to Array.length K.tiles - 1 do
      let t = K.tiles.(x) in
      if find_mma kind K.K K.K t >= 0 then begin
        let s = K.shape t in
        let blocks = batch * ceil_div m s.bm * ceil_div n s.bn * splits in
        let c =
          Float.of_int (ceil_div blocks sms)
          *. Float.of_int (s.bm * s.bn)
          /. Float.of_int (efficiency t)
        in
        if c < Float.Array.get least 0 then begin
          Float.Array.set least 0 c;
          best := t
        end
      end
    done;
    !best
  end

(* The ranges of k an mma output sums apart: one a [least] runs of
   NX_MMA_RUN terms, at least one and at most 16; few rows or columns, or
   few outputs (bfloat16 256x256x65536: 64 tiles), which fill the GPU only
   with their k cut, take more. *)
let mma_splits ~batch ~m ~n ~k =
  let least =
    if m <= 4 || n <= 4 || batch * m * n <= 1 lsl 18 then 16 else 256
  in
  Int.max 1 (Int.min 16 (ceil_div k K.mma_run / least))

(* Bounds *)

(* The most rows and columns a launch takes, and terms an mma launch
   takes: each coordinate the kernels form fits an int32. *)
let max_extent = 1 lsl 30
let max_int32 = 0x7fff_ffff

(* The most batch elements an mma launch takes: the grid's z. *)
let max_batch = 65535

(* Contraction parameters *)

let write_operand w at v ~rows ~flags refs =
  let module O = K.Contract_operand in
  address refs w (at + O.p) (slot_of v.where) v.at ~write:false;
  set64 w (at + O.s) v.s.(0);
  set64 w (at + O.s + 8) v.s.(1);
  set64 w (at + O.s + 16) v.s.(2);
  set32 w (at + O.rows) (Int.max rows 1);
  set32 w (at + O.dtype) (code v.dt);
  set32 w (at + O.flags) flags

(* Whether [v]'s first element lies on 16 bytes, as the device addresses
   it: a workspace piece's does. *)
let aligned16 (v : view) =
  match v.where with
  | Work -> v.at mod 16 = 0
  | Own o -> Rig.Buffer.aligned o.buf ~at:v.at 16

(* Whether rows along [v]'s contiguous axis load as 16-byte vectors into a
   tile of [rows]: k contiguous, or the free axis and the tile whole
   vectors; every other stride whole vectors, the first element on a
   vector. *)
let vectors (v : view) ~bytes ~rows =
  let per = 16 / bytes in
  let s = v.s in
  aligned16 v
  && s.(0) mod per = 0
  &&
  if s.(2) = 1 then s.(1) mod per = 0
  else s.(1) = 1 && rows mod per = 0 && s.(2) mod per = 0

let free_contiguous (v : view) = v.s.(1) = 1 && v.s.(2) <> 1

(* The plan *)

(* What a matrix routine's product asks beyond {!Nx_kernel.S.contract}
   (kernels.h): its first range of k summing from init, a negated, a
   conjugated a or b, a triangle of tiles, and [ranges] of k for every
   tile, a function of each matrix's extents alone, in place of the
   model's. *)
type options = {
  init_first : bool;
  neg : bool;
  conj_a : bool;
  conj_b : bool;
  tri : int;
  ranges : int option;
}

let plain =
  {
    init_first = false;
    neg = false;
    conj_a = false;
    conj_b = false;
    tri = 0;
    ranges = None;
  }

type segments =
  | No_segments
  | Rows_seg of {
      ids : side;
      groups : int;
      sbg : int;
      inner : int;
      extent : int;
    }
  | K_seg of { ids : side; groups : int; inner : int; extent : int }

(* The call as the planner reads it. *)
type c = {
  a : side;
  b : side;
  y : side;
  init : side option;
  sa : scale option;
  sb : scale option;
  seg : segments;
  acc : D.any;
  batch : int;
  m : int;
  n : int;
  k : int;
}

(* init and y *)

(* The groups a Contracting call adds to a batch element's outputs: its
   groups, else one. *)
let kgroups c = match c.seg with K_seg g -> g.groups | _ -> 1

(* The dtypes a kernel stores y in and reads init in: [own_y dt] and
   [own_init dt] hold for those it writes and reads itself, [wdt] the one
   it takes in the workspace for any other. The FMA kernels' are the
   accumulator's, a complex one's, and integers of the compute type's
   width for init, no wider than an integer accumulator for y; the mma
   kernels' float32's, bfloat16's and float16's, or int32's. *)
type store = { own_y : D.any -> bool; own_init : D.any -> bool; wdt : D.any }

let fma_store c =
  let ct, cx = ct_of c.acc in
  let integer (D.Any d) =
    (D.is D.Signed d || D.is D.Unsigned d) && D.bits d >= 8
  in
  (* An integer init of the compute type's width adds as its bits do,
     modulo the accumulator's width; an integer y keeps a sum's low bits
     where it is no wider than the accumulator. *)
  let own ~y (dt : D.any) =
    if cx then dt = c.acc
    else
      match (ct, dt, c.acc) with
      | F32, D.Any Float32, D.Any Float32 | F64, D.Any Float64, D.Any Float64 ->
          true
      | (U32 | U64), _, _ ->
          integer dt && integer c.acc
          && if y then bits dt <= bits c.acc else width dt = ct_bytes ct
      | _ -> false
  in
  {
    own_y = own ~y:true;
    own_init = own ~y:false;
    wdt = (if cx then c.acc else ct_dtype ct);
  }

let mma_store (kind : K.kind) =
  match kind with
  | S8 ->
      let own = function D.Any (Int32 | Uint32) -> true | _ -> false in
      { own_y = own; own_init = own; wdt = D.Any D.Int32 }
  | Bf16 | F16 | Any ->
      let own = function
        | D.Any (Float32 | Bfloat16 | Float16) -> true
        | _ -> false
      in
      { own_y = own; own_init = own; wdt = D.Any D.Float32 }

(* init as the kernels read it, and its groups' stride: in place where its
   axes merge and its dtype is the kernel's own, else cast into [st.wdt]
   in the workspace, a group that repeats one element packed once, its
   stride 0. *)
let init_view b c st =
  match c.init with
  | None -> None
  | Some s ->
      let dt = dtype_of s.arr in
      let own =
        match c.seg with K_seg _ -> L.stride (layout s.arr) 0 | _ -> 0
      in
      (* A complex call reads its columns' parts as neighbours. *)
      let cx = snd (ct_of c.acc) in
      let cols = (not cx) || c.n <= 1 || stride_of s 2 = 1 in
      if grouped s && st.own_init dt && cols then Some (view_of s, own)
      else begin
        let one g =
          (g < 2 || not cx)
          && match s.runs.(g) with Some (e, 0) -> e > 1 | _ -> false
        in
        let nb = if one 0 then 1 else c.batch
        and nm = if one 1 then 1 else c.m in
        let nn = if one 2 then 1 else c.n in
        let gs =
          [|
            (if one 0 then 0 else nm * nn);
            (if one 1 then 0 else nn);
            (if one 2 then 0 else 1);
            nb * nm * nn;
          |]
        in
        let at = take b (kgroups c * gs.(3) * width st.wdt) in
        let ax = axes_of (layout s.arr) s.groups in
        let keep = Array.map (fun g -> g = 3 || not (one g)) ax.grp in
        let pick = kept keep in
        let walked =
          { ext = pick ax.ext; str = pick ax.str; grp = pick ax.grp }
        in
        pack b ~mode:K.pack_value ~src_slot:s.slot ~src_dt:dt
          ~src_first:(L.offset (layout s.arr))
          ~dst_where:Work ~dst_at:at ~dst_dt:st.wdt ~dst_first:0 ~acc:c.acc
          walked (packed_strides walked gs) ();
        Some
          ( { where = Work; at; s = Array.sub gs 0 3; dt = st.wdt; sg = 0 },
            gs.(3) )
      end

(* y as the kernels write it, its groups' stride, and whether it lies in
   the workspace: in place where its axes merge and its dtype is the
   kernel's own, else in [st.wdt] in the workspace. *)
let y_view b c st =
  let s = c.y in
  let dt = dtype_of s.arr in
  let own = match c.seg with K_seg _ -> L.stride (layout s.arr) 0 | _ -> 0 in
  let cols = (not (snd (ct_of c.acc))) || c.n <= 1 || stride_of s 2 = 1 in
  if grouped s && st.own_y dt && cols then (view_of s, own, false)
  else
    let at = take b (kgroups c * c.batch * c.m * c.n * width st.wdt) in
    ( { where = Work; at; s = [| c.m * c.n; c.n; 1 |]; dt = st.wdt; sg = 0 },
      c.batch * c.m * c.n,
      true )

(* y cast out of the workspace, where [y_view] put it. *)
let unpack b c (v : view) =
  let s = c.y in
  let ax = axes_of (layout s.arr) s.groups in
  let src = packed_strides ax (out_strides ~batch:c.batch ~m:c.m ~n:c.n) in
  pack b ~mode:K.pack_cast ~src_slot:ws_slot ~src_dt:v.dt
    ~src_first:(v.at / width v.dt)
    ~dst_where:(Own { slot = s.slot; buf = buffer s.arr })
    ~dst_at:0 ~dst_dt:(dtype_of s.arr)
    ~dst_first:(L.offset (layout s.arr))
    ~acc:c.acc { ax with str = src } ax.str ()

(* Contraction parameters *)

(* The parameters of a contraction kernel common to the FMA and the mma
   kernels. *)
let params c ~a ~b:bv ~fa ~fb ~init ~(y : view) ~flags ~ct_k ~n refs =
  let module P = K.Contract_params in
  let w = params_of P.size in
  write_operand w P.a a ~rows:c.m ~flags:fa refs;
  write_operand w P.b bv ~rows:n ~flags:fb refs;
  (match init with
  | None -> set32 w P.init_dtype (code (dtype_of c.y.arr))
  | Some ((v : view), _) ->
      address refs w P.init (slot_of v.where) v.at ~write:false;
      set64 w P.si v.s.(0);
      set64 w (P.si + 8) v.s.(1);
      set64 w (P.si + 16) v.s.(2);
      set32 w P.init_dtype (code v.dt));
  address refs w P.y (slot_of y.where) y.at ~write:true;
  set64 w P.sy y.s.(0);
  set64 w (P.sy + 8) y.s.(1);
  set64 w (P.sy + 16) y.s.(2);
  set32 w P.y_dtype (code y.dt);
  set32 w P.acc_dtype (code c.acc);
  set32 w P.flags flags;
  set32 w P.batch c.batch;
  set32 w P.m c.m;
  set32 w P.n n;
  set64 w P.k ct_k;
  w

(* A view of [v]'s window from batch element [z], row [i] (a's, or b's
   column, or y's row and column [j]). *)
let shift (v : view) ~z ~i ~j =
  {
    v with
    at = v.at + (((z * v.s.(0)) + (i * v.s.(1)) + (j * v.s.(2))) * width v.dt);
  }

(* Segments *)

(* A segmented call's ids as ROWS and KS read them: where, the byte of
   their first element, and their strides along batch and the segmented
   axis; packed, int64 [batch; extent], where their batch axes do not
   merge. *)
let ids_view b c (s : side) ~extent =
  match s.runs.(0) with
  | Some (_, sz) ->
      let l = layout s.arr in
      ( Own { slot = s.slot; buf = buffer s.arr },
        8 * L.offset l,
        sz,
        L.stride l (L.rank l - 1) )
  | None ->
      let dt = D.Any D.Int64 in
      let at = take b (8 * c.batch * extent) in
      let ax = axes_of (layout s.arr) s.groups in
      pack b ~mode:0 ~src_slot:s.slot ~src_dt:dt
        ~src_first:(L.offset (layout s.arr))
        ~dst_where:Work ~dst_at:at ~dst_dt:dt ~dst_first:0 ~acc:dt ax
        (packed_strides ax [| extent; 1; 0; 0 |])
        ();
      (Work, at, extent, 1)

(* The launches of ROWS or KS, each phase's threads and the elements its
   threads stride over: [ids] as [ids_view] gives them, each table a
   workspace byte, [-1] for none, KS's [perm] of int64 words where
   [wide]. *)
let segment_launches ?(wide = false) b ~kernel ~phases ~ids:(where, at, sz, sx)
    ~counts ~cursor ~perm ~desc ~ntiles ~batch ~count ~first ~extent ~inner
    ~groups ~bm =
  let module P = K.Segment_params in
  List.iteri
    (fun phase (threads, gx) ->
      let w = params_of P.size and refs = ref [] in
      address refs w P.ids (slot_of where) at ~write:false;
      List.iter
        (fun (f, at) -> if at >= 0 then address refs w f ws_slot at ~write:true)
        [
          (P.counts, counts);
          (P.cursor, cursor);
          (P.perm, perm);
          (P.desc, desc);
          (P.ntiles, ntiles);
        ];
      set64 w P.sz sz;
      set64 w P.sx sx;
      set64 w P.inner inner;
      set64 w P.count count;
      set64 w P.extent extent;
      set64 w P.first first;
      set32 w P.batch batch;
      set32 w P.groups groups;
      set32 w P.bm bm;
      set32 w P.phase phase;
      set32 w P.wide (Bool.to_int wide);
      launch b ~kernel ~gx ~threads ~shared:0 w refs)
    phases

(* Blocks of 1024 threads that stride over [n] elements: at most 1024. *)
let strided n = Int.max 1 (Int.min 1024 (ceil_div n 1024))

(* The FMA kernels *)

(* Windows of a call past one launch's bounds, [(z, i, j, batch, m, n)]:
   its batch, then its rows, then its columns halved until each fits;
   [None] where a window of one output does not. *)
let windows ~fits w =
  let rec go ((z, i, j, nb, nm, nn) as w) acc =
    match acc with
    | None -> None
    | Some l ->
        if fits ~batch:nb ~m:nm ~n:nn then Some (w :: l)
        else if nb > 1 then
          let h = nb / 2 in
          go (z, i, j, h, nm, nn) (go (z + h, i, j, nb - h, nm, nn) acc)
        else if nm > 1 then
          let h = nm / 2 in
          go (z, i, j, nb, h, nn) (go (z, i + h, j, nb, nm - h, nn) acc)
        else if nn > 1 then
          let h = nn / 2 in
          go (z, i, j, nb, nm, h) (go (z, i, j + h, nb, nm, nn - h) acc)
        else None
  in
  go w (Some [])

(* The most rows (Rows) or contracted indices (Contracting) of a
   segmented window's tables, 4 bytes each: 64 MiB. A window of one batch
   element holds all its contracted indices. *)
let seg_window = 1 lsl 24

(* The contracted extent past which a Contracting call's tables hold int64
   indices, which take twice the memory of int32 ones. *)
let wide_k = Atomic.make max_int32
let set_wide_k n = Atomic.set wide_k (Int.max 0 (Int.min max_int32 n))

(* A Contracting call's prepass counts each group's k in parts of at least
   [ks_part] k, at most [ks_counts] counts a window. *)
let ks_part = 4096
let ks_counts = 1 lsl 22

(* NX_CONTRACT_Y_VECTORS for the FMA kernels: y of the compute type's
   width, its columns contiguous and its rows, batch elements and
   groups whole vectors from a first element on one. *)
let fma_flags ct ~(y : view) ~ystep =
  let per = 16 / ct_bytes ct in
  if
    width y.dt = ct_bytes ct
    && y.s.(2) = 1
    && y.s.(1) mod per = 0
    && y.s.(0) mod per = 0
    && ystep mod per = 0
    && aligned16 y
  then K.contract_y_vectors
  else 0

let operand_flags (v : view) = if free_contiguous v then K.op_free else 0

(* a or b as an FMA kernel reads it: scaled, the scale multiplied into the
   accumulator's dtype; of a sub-byte dtype or axes that do not merge,
   packed as it is. *)
let fma_operand b c s scale ~rows ~ct =
  let dt = dtype_of s.arr in
  match scale with
  | Some sc ->
      pack_operand b s ~into:(ct_dtype ct) ~batch:c.batch ~rows ~k:c.k
        ~acc:c.acc ~value:true ~scale:sc ()
  | None when sub_byte dt || not (grouped s) ->
      pack_operand b s ~into:dt ~batch:c.batch ~rows ~k:c.k ~acc:c.acc
        ~value:false ()
  | None -> view_of s

(* [v], in place, packed as it is where a native load cannot read it:
   rows that are not 16-byte vectors, or, for a Rows call's a, rows that
   are not along k. A packed view is laid out for every kernel. *)
let native_operand b c s (v : view) ~rows ~tile ~bytes ~k_rows ~shift =
  match v.where with
  | Work -> v
  | Own _ ->
      if
        vectors v ~bytes ~rows:tile
        && ((not k_rows) || v.s.(2) = 1)
        && shift mod (16 / bytes) = 0
      then v
      else
        pack_operand b s ~into:v.dt ~batch:c.batch ~rows ~k:c.k ~acc:c.acc
          ~value:false ()

(* The launches of [plans], each window's plan: for a segmented call its
   prepass, then the contraction. The windows run one after another in the
   call's submission, so they share the tables and partials, sized by the
   largest. *)
let fma_launches b c ~o ~ct ~cx ~k' ~a ~bv ~init ~y ~ystep plans =
  (* A complex call's columns, real ones. *)
  let jm = if cx then 2 else 1 in
  let most f = List.fold_left (fun n x -> Int.max n (f x)) 0 plans in
  let nb_most = most (fun ((_, _, _, nb, _, _), _) -> nb) in
  let kwide =
    match c.seg with K_seg _ -> c.k > Atomic.get wide_k | _ -> false
  in
  (* The segment tables, and each window's prepass and fields. *)
  let seg =
    match c.seg with
    | No_segments -> fun _ _ _ _ -> ()
    | Rows_seg r ->
        let g = r.groups + 1 in
        let counts = take b (4 * ((2 * nb_most * g) + 1)) in
        let cursor = take b (4 * nb_most * g) in
        let perm =
          take b (4 * most (fun ((_, _, _, nb, nm, _), _) -> nb * nm))
        in
        let desc = take b (16 * most (fun (_, p) -> p.tiles_m)) in
        let ntiles = take b 4 in
        let where, at, sz, sx = ids_view b c r.ids ~extent:r.extent in
        fun (z, i, _, nb, nm, _) (p : plan) w refs ->
          let rows = strided (Int.max (nb * nm) p.tiles_m) in
          segment_launches b ~kernel:rows_kernel
            ~phases:
              [
                (1024, strided (nb * g));
                (1024, strided (nb * nm));
                (32, 1);
                (1024, rows);
              ]
            ~ids:(where, at + (8 * z * sz), sz, sx)
            ~counts ~cursor ~perm ~desc ~ntiles ~batch:nb ~count:nm ~first:i
            ~extent:r.extent ~inner:r.inner ~groups:r.groups ~bm:p.i.bm;
          let module P = K.Contract_params in
          address refs w P.perm ws_slot perm ~write:false;
          address refs w P.desc ws_slot desc ~write:false;
          address refs w P.ntiles ws_slot ntiles ~write:false;
          set64 w P.sbg (match bv.where with Work -> bv.sg | Own _ -> r.sbg);
          set32 w P.groups r.groups
    | K_seg g ->
        let gg = g.groups + 1 in
        (* A batch element's k in parts of at least [ks_part], whose
           groups' counts take at most [ks_counts] words. *)
        let parts nb =
          Int.max 1
            (Int.min (ceil_div c.k ks_part)
               (ks_counts / Int.max 1 (nb * g.groups)))
        in
        let koff = take b (8 * nb_most * gg) in
        let cursor = take b (8 * Int.max 1 (nb_most * g.groups)) in
        let counts =
          take b
            (8
            * Int.max 1
                (most (fun ((_, _, _, nb, _, _), _) -> nb * parts nb * g.groups))
            )
        in
        let kperm =
          take b ((if kwide then 8 else 4) * Int.max 1 (nb_most * c.k))
        in
        let where, at, sz, sx = ids_view b c g.ids ~extent:g.extent in
        fun (z, _, _, nb, _, _) _ w refs ->
          let p = parts nb in
          segment_launches ~wide:kwide b ~kernel:ks_kernel
            ~phases:
              [
                (1024, strided (nb * p * g.groups));
                (1024, nb * p);
                (1024, strided (nb * g.groups));
                (1024, strided nb);
                (32, nb * p);
              ]
            ~ids:(where, at + (8 * z * sz), sz, sx)
            ~counts:koff ~cursor ~perm:kperm ~desc:counts ~ntiles:(-1) ~batch:nb
            ~count:c.k ~first:0 ~extent:g.extent ~inner:g.inner ~groups:g.groups
            ~bm:p;
          let module P = K.Contract_params in
          address refs w P.kperm ws_slot kperm ~write:false;
          address refs w P.koff ws_slot koff ~write:false;
          set32 w P.kgroups g.groups;
          set64 w P.syg ystep;
          set64 w P.sig_ (match init with Some (_, st) -> st | None -> 0)
  in
  (* The partials, a tile's sums for each block, where a call splits or
     a float range passes a block (combine.cuh's fold), the most any window
     takes, and the split tiles' tickets. *)
  let bytes = ct_bytes ct in
  let blocked = ct_float ct && (not o.init_first) && k' > K.contract_block in
  let split_of (p : plan) =
    let splits = p.s1 > 1 || p.s2 > 1 in
    ( (if splits || blocked then blocks p * p.i.bm * p.i.bn * bytes else 0),
      if splits then 4 * p.tiles else 0 )
  in
  let part_bytes = most (fun (_, p) -> fst (split_of p)) in
  let partials = if part_bytes > 0 then take b part_bytes else 0 in
  b.ticket_bytes <-
    Int.max b.ticket_bytes (most (fun (_, p) -> snd (split_of p)));
  b.windowed <- b.windowed || List.length plans > 1;
  let fa = operand_flags a in
  let fb = operand_flags bv in
  let kflags =
    (match c.seg with K_seg _ when cx -> K.contract_k_pairs | _ -> 0)
    lor if kwide then K.contract_k_wide else 0
  in
  (* A triangle's tiles are those of the whole call's outputs. *)
  let extra =
    (if o.init_first then K.contract_init_first else 0)
    lor if List.length plans = 1 then o.tri else 0
  in
  List.iter
    (fun (((z, i, j, nb, nm, nn) as win), p) ->
      let refs = ref [] in
      let j = jm * j in
      let yw = shift y ~z ~i ~j in
      let initw = Option.map (fun (v, st) -> (shift v ~z ~i ~j, st)) init in
      let cw = { c with batch = nb; m = nm; n = nn } in
      let w =
        params cw ~a:(shift a ~z ~i ~j:0) ~b:(shift bv ~z ~i:j ~j:0) ~fa ~fb
          ~init:initw ~y:yw
          ~flags:(fma_flags ct ~y:yw ~ystep lor kflags lor extra)
          ~ct_k:k' ~n:(jm * nn) refs
      in
      let module P = K.Contract_params in
      set32 w P.tiles_m p.tiles_m;
      set32 w P.tiles_n p.tiles_n;
      set32 w P.main p.main;
      set32 w P.s1 p.s1;
      set32 w P.s2 p.s2;
      let part, tickets = split_of p in
      if part > 0 then address refs w P.partials ws_slot partials ~write:true;
      if tickets > 0 then address refs w P.tickets tickets_slot 0 ~write:true;
      seg win p w refs;
      let shared = if p.i.ld = 3 then gemv_shared ct ~m:nm else p.i.smem in
      launch b ~kernel:p.i.kernel ~gx:(blocks p) ~threads:p.i.threads ~shared w
        refs)
    plans

(* [c] with a and b, and y's and init's rows and columns, exchanged: the
   same sums, each product's factors exchanged, which a fused
   multiply-add rounds alike. *)
let transposed c =
  let swap s =
    {
      s with
      groups = [| s.groups.(0); s.groups.(2); s.groups.(1) |];
      runs = [| s.runs.(0); s.runs.(2); s.runs.(1) |];
    }
  in
  {
    c with
    a = c.b;
    b = c.a;
    sa = c.sb;
    sb = c.sa;
    m = c.n;
    n = c.m;
    y = swap c.y;
    init = Option.map swap c.init;
  }

(* Whether [c] has gemv's few rows only once transposed: few columns of
   an a that gemv reads as its b. *)
let few_columns ~o c =
  let unseg = match c.seg with No_segments -> true | _ -> false in
  let at = dtype_of c.a.arr in
  o == plain && unseg && c.n <= K.gemv_rows && c.m > K.gemv_rows
  && (not (snd (ct_of c.acc)))
  && (not (is_complex at))
  && width at <= 8

let rec fma ?(o = plain) b c =
  if few_columns ~o c then fma ~o b (transposed c)
  else
    let ct, cx = ct_of c.acc in
    let st = fma_store c in
    let kseg = match c.seg with K_seg _ -> true | _ -> false in
    let unseg = match c.seg with No_segments -> true | _ -> false in
    let init = init_view b c st in
    let y, ystep, packed_y = y_view b c st in
    let rdt = ct_dtype ct in
    (* A negated a, a routine's, is a copy of its values negated, so that
     no load tests for it. *)
    let negated into =
      pack_operand b c.a ~into ~batch:c.batch ~rows:c.m ~k:c.k ~acc:c.acc
        ~value:true ~neg:true ()
    in
    (* A complex call's real views: init and y in place or as [init_view] and
     [y_view] laid them out, a cast into the accumulator, b's blocks. *)
    let init, ystep, yr, a, bv =
      if not cx then
        ( init,
          ystep,
          y,
          (if o.neg then negated (ct_dtype ct)
           else fma_operand b c c.a c.sa ~rows:c.m ~ct),
          fma_operand b c c.b c.sb ~rows:c.n ~ct )
      else
        let init = Option.map (fun (v, st) -> (real v rdt, 2 * st)) init in
        let a =
          if o.neg then real (negated c.acc) rdt
          else if
            dtype_of c.a.arr = c.acc
            && grouped c.a
            && (c.k <= 1 || stride_of c.a 2 = 1)
          then real (view_of c.a) rdt
          else
            real
              (pack_operand b c.a ~into:c.acc ~batch:c.batch ~rows:c.m ~k:c.k
                 ~acc:c.acc ~value:true ())
              rdt
        in
        let mode =
          (if o.conj_b then K.pack_conj else 0)
          lor if o.conj_a then K.pack_flip else 0
        in
        let bv =
          pack_block ~mode b c.b ~rdt ~batch:c.batch ~rows:c.n ~k:c.k ~acc:c.acc
        in
        (init, 2 * ystep, real y rdt, a, bv)
    in
    let ld = if native ct a.dt && native ct bv.dt && not kseg then 0 else 1 in
    (* An integer call of more rows than gemv takes whose operands convert:
     each operand cast once into the compute type in the workspace, as the
     converting loads would cast it, then the native tiles, whose loads do
     not decode every element once a tile. int16 1024x1024x1024 into int64:
     633 us converting, 454 us cast first. *)
    let a, bv, ld =
      if
        ld = 1 && unseg && o == plain && (not cx)
        && (not (ct_float ct))
        && c.m > K.gemv_rows
      then
        let cast s (v : view) ~rows =
          if native ct v.dt then v
          else
            pack_operand b s ~into:(ct_dtype ct) ~batch:c.batch ~rows ~k:c.k
              ~acc:c.acc ~value:true ()
        in
        (cast c.a a ~rows:c.m, cast c.b bv ~rows:c.n, 0)
      else (a, bv, ld)
    in
    let gemv =
      o == plain && unseg && c.m <= K.gemv_rows
      && (not (is_complex bv.dt))
      && width bv.dt <= 8
    in
    let k' = if cx then 2 * c.k else c.k in
    let shape ~batch ~m ~n =
      {
        batch = batch * kgroups c;
        m;
        n = (if cx then 2 * n else n);
        k = k';
        cx;
        wa = Float.of_int (bits (dtype_of c.a.arr)) /. 8.;
        wb = Float.of_int (bits (dtype_of c.b.arr)) /. 8.;
        wy = Float.of_int (bits (dtype_of c.y.arr)) /. 8.;
      }
    in
    (* A routine's ranges, whatever the instance: its bits follow them. *)
    let ranged (p : plan) =
      match o.ranges with
      | None -> p
      | Some s ->
          {
            p with
            main = p.tiles;
            s1 = Int.max 1 (Int.min s (ceil_div k' K.contract_run));
            s2 = 1;
          }
    in
    let whole =
      ranged (choose ct ld ~gemv (shape ~batch:c.batch ~m:c.m ~n:c.n))
    in
    (* A native load needs 16-byte vectors along a contiguous axis. *)
    let a, bv =
      if whole.i.ld <> 0 then (a, bv)
      else if cx then
        (* a in place, else cast into the accumulator; b's blocks are, and
         a packed a. *)
        let packed = match a.where with Work -> true | Own _ -> false in
        if packed || vectors a ~bytes:(ct_bytes ct) ~rows:whole.i.bm then (a, bv)
        else
          ( real
              (pack_operand b c.a ~into:c.acc ~batch:c.batch ~rows:c.m ~k:c.k
                 ~acc:c.acc ~value:true ())
              rdt,
            bv )
      else
        let bytes = ct_bytes ct in
        let rows_seg = match c.seg with Rows_seg _ -> true | _ -> false in
        (* A Rows call's kernel steps b in place by its stride along
         b_axis from one group to the next, so that stride too must be
         whole vectors; a packed b steps by its pack's. *)
        let shift =
          match (c.seg, bv.where) with Rows_seg r, Own _ -> r.sbg | _ -> 0
        in
        ( native_operand b c c.a a ~rows:c.m ~tile:whole.i.bm ~bytes
            ~k_rows:rows_seg ~shift:0,
          native_operand b c c.b bv ~rows:c.n ~tile:whole.i.bn ~bytes
            ~k_rows:false ~shift )
    in
    (* A window's plan: the instance's for its extents, every tile of a
     segmented call in the same ranges of k, a Rows call's row tiles its
     groups' rows, at most one a row. *)
    let plan_of ~batch ~m ~n =
      let p = ranged (choose ~only:whole.i ct ld ~gemv (shape ~batch ~m ~n)) in
      match c.seg with
      | No_segments -> p
      | Rows_seg r ->
          let tiles_m =
            batch * (ceil_div m p.i.bm + Int.min (r.groups + 1) m)
          in
          let tiles = tiles_m * p.tiles_n in
          { p with tiles_m; tiles; main = tiles; s2 = 1 }
      | K_seg _ -> { p with main = p.tiles; s2 = 1 }
    in
    (* A window fits a launch where its coordinates fit an int32, and a
     segmented one's tables hold at most [seg_window] rows (Rows) or
     contracted indices (Contracting), but one batch element's. *)
    let fits ~batch ~m ~n =
      m <= max_extent
      && (if cx then 2 * n else n) <= max_extent
      && (match c.seg with
        | No_segments -> true
        | Rows_seg _ -> batch * m <= seg_window
        | K_seg _ -> batch = 1 || batch * c.k <= seg_window)
      && blocks (plan_of ~batch ~m ~n) <= max_int32
    in
    match windows ~fits (0, 0, 0, c.batch, c.m, c.n) with
    | Some ws ->
        let plans =
          List.map
            (fun ((_, _, _, nb, nm, nn) as w) ->
              let p =
                if unseg && nb = c.batch && nm = c.m && nn = c.n then whole
                else plan_of ~batch:nb ~m:nm ~n:nn
              in
              (w, p))
            ws
        in
        fma_launches b c ~o ~ct ~cx ~k' ~a ~bv ~init ~y:yr ~ystep plans;
        if packed_y then unpack b c y;
        Ok ()
    | _ -> Error A.Declined

(* The matrix units' launches *)

(* NX_CONTRACT_Y_VECTORS for the mma kernels: a float32 sum into float32,
   bfloat16 or float16, an int32 one into a 32-bit integer, y's columns
   contiguous and its rows and batch elements on 16 bytes. Over no terms,
   bfloat16 and float16 store element by element, whose cast keeps a NaN
   init's sign and payload as nx_dtype.h's encoders do. *)
let mma_flags kind ~k ~(y : view) =
  let own =
    match ((kind : K.kind), y.dt) with
    | S8, D.Any (Int32 | Uint32) -> true
    | (Bf16 | F16 | Any), D.Any Float32 -> true
    | (Bf16 | F16 | Any), D.Any (Bfloat16 | Float16) -> k > 0
    | _ -> false
  in
  let wy = width y.dt in
  if
    own
    && y.s.(2) = 1
    && y.s.(1) * wy mod 16 = 0
    && y.s.(0) * wy mod 16 = 0
    && aligned16 y
  then K.contract_y_vectors
  else 0

let mma b c (kind : K.kind) =
  let st = mma_store kind in
  let init = init_view b c st in
  let y, _, packed_y = y_view b c st in
  let splits = mma_splits ~batch:c.batch ~m:c.m ~n:c.n ~k:c.k in
  let t = mma_tile ~batch:c.batch ~m:c.m ~n:c.n ~splits kind in
  let sh = K.shape t in
  let into : D.any =
    match kind with
    | S8 -> D.Any Int8
    | F16 -> D.Any Float16
    | Bf16 | Any -> D.Any Bfloat16
  in
  let es = width into in
  (* An operand the unit reads in place: of its dtype, its axes merged,
     its rows 16-byte vectors along k or along its free axis where a tile
     has an instance of that layout, that axis's extent whole vectors;
     packed with k contiguous otherwise, a float8 one into bfloat16. A
     16-byte copy of a row's last part reads its whole 16 bytes, past the
     array's last element where the extent is not whole vectors. *)
  let operand s ~rows ~tile_rows ~other =
    let dt = dtype_of s.arr in
    let packed () =
      ( pack_operand b s ~into ~batch:c.batch ~rows ~k:c.k ~acc:c.acc
          ~value:(is_f8 dt) (),
        K.K )
    in
    if is_f8 dt || not (grouped s) then packed ()
    else
      let v = view_of s in
      let lay = if free_contiguous v then other else K.K in
      let along = if lay = K.K then c.k else rows in
      if vectors v ~bytes:es ~rows:tile_rows && along mod (16 / es) = 0 then
        (v, lay)
      else packed ()
  in
  let a, la = operand c.a ~rows:c.m ~tile_rows:sh.bm ~other:K.M in
  let a, la =
    if la <> K.K && find_mma kind la K.K t < 0 then
      ( pack_operand b c.a ~into ~batch:c.batch ~rows:c.m ~k:c.k ~acc:c.acc
          ~value:false (),
        K.K )
    else (a, la)
  in
  let bv, lb = operand c.b ~rows:c.n ~tile_rows:sh.bn ~other:K.N in
  let bv, lb =
    if lb <> K.K && find_mma kind la lb t < 0 then
      ( pack_operand b c.b ~into ~batch:c.batch ~rows:c.n ~k:c.k ~acc:c.acc
          ~value:false (),
        K.K )
    else (bv, lb)
  in
  let kernel = find_mma kind la lb t in
  let fits ~batch ~m ~n =
    batch <= max_batch && m <= max_extent && n <= max_extent
    && ceil_div m sh.bm * ceil_div n sh.bn <= max_int32
  in
  (* A window of one output fits. *)
  let ws = Option.get (windows ~fits (0, 0, 0, c.batch, c.m, c.n)) in
  b.windowed <- b.windowed || List.length ws > 1;
  let tiles (_, _, _, nb, nm, nn) =
    nb * ceil_div nm sh.bm * ceil_div nn sh.bn
  in
  let most = List.fold_left (fun t w -> Int.max t (tiles w)) 0 ws in
  (* A tile's sums for each range, where the call splits or a float range
     passes a block (combine.cuh's fold). *)
  let blocked = kind <> K.S8 && c.k > K.contract_block in
  let partials =
    if splits > 1 || blocked then take b (most * splits * sh.bm * sh.bn * 4)
    else 0
  in
  if splits > 1 then b.ticket_bytes <- Int.max b.ticket_bytes (4 * most);
  let threads = sh.bm / sh.wm * (sh.bn / sh.wn) * 32 in
  List.iter
    (fun (z, i, j, nb, nm, nn) ->
      let refs = ref [] in
      let yw = shift y ~z ~i ~j in
      let initw = Option.map (fun (v, st) -> (shift v ~z ~i ~j, st)) init in
      let cw = { c with batch = nb; m = nm; n = nn } in
      let w =
        params cw ~a:(shift a ~z ~i ~j:0) ~b:(shift bv ~z ~i:j ~j:0) ~fa:0 ~fb:0
          ~init:initw ~y:yw
          ~flags:(mma_flags kind ~k:c.k ~y:yw)
          ~ct_k:c.k ~n:nn refs
      in
      let module P = K.Contract_params in
      let tm = ceil_div nm sh.bm and tn = ceil_div nn sh.bn in
      set32 w P.tiles_m tm;
      set32 w P.tiles_n tn;
      set32 w P.main (nb * tm * tn);
      set32 w P.s1 splits;
      set32 w P.s2 1;
      if splits > 1 || blocked then
        address refs w P.partials ws_slot partials ~write:true;
      if splits > 1 then address refs w P.tickets tickets_slot 0 ~write:true;
      launch b ~kernel ~gx:(tm * tn) ~gy:splits ~gz:nb ~threads
        ~shared:(sh.stages * (sh.bm + sh.bn) * sh.bkb)
        w refs)
    ws;
  if packed_y then unpack b c y

(* The call *)

let free rank (used : int array) =
  Array.of_list
    (List.filter (fun x -> not (Array.mem x used)) (List.init rank Fun.id))

let product l axes = Array.fold_left (fun n ax -> n * L.dim l ax) 1 axes

(* The call [c] read from the arrays, its slots added to [b]: a, b and y,
   then init, the ids, and a's and b's scales where given. *)
let read_call b ~batch ~contracting ~segments ~acc ~(dst : A.any) ~init
    (oa : Nx_kernel.operand) (ob : Nx_kernel.operand) =
  let la = layout oa.array and lb = layout ob.array in
  let bp = Iarray.to_array batch and cp = Iarray.to_array contracting in
  let nb = Array.length bp in
  let ab = Array.map fst bp and bb = Array.map snd bp in
  let ac = Array.map fst cp and bc = Array.map snd cp in
  let afree = free (L.rank la) (Array.append ab ac) in
  let bfree = free (L.rank lb) (Array.append bb bc) in
  let bcols =
    match (segments : _ Nx_kernel.segments option) with
    | Some (Rows r) ->
        Array.of_list (List.filter (( <> ) r.b_axis) (Array.to_list bfree))
    | _ -> bfree
  in
  let fa = Array.length afree and fb = Array.length bcols in
  let lead = match segments with Some (Contracting _) -> 1 | _ -> 0 in
  let yb = Array.init nb (fun i -> lead + i) in
  let yr = Array.init fa (fun i -> lead + nb + i) in
  let yc = Array.init fb (fun i -> lead + nb + fa + i) in
  let ygroups =
    if lead = 1 then [| yb; yr; yc; [| 0 |] |] else [| yb; yr; yc |]
  in
  let slot (x : A.any) = slot b (buffer x) in
  let a = side_of oa.array (slot oa.array) [| ab; afree; ac |] in
  (* Rows' b_axis is a fourth group, which a pack keeps. *)
  let bgroups =
    match segments with
    | Some (Rows r) -> [| bb; bcols; bc; [| r.b_axis |] |]
    | _ -> [| bb; bcols; bc |]
  in
  let bs = side_of ob.array (slot ob.array) bgroups in
  let y = side_of dst (slot dst) ygroups in
  let init = Option.map (fun i -> side_of i (slot i) ygroups) init in
  let ids_side ids =
    let x = A.Any ids in
    side_of x (slot x) [| Array.init nb Fun.id; [| nb |] |]
  in
  let seg =
    match segments with
    | None -> No_segments
    | Some (Rows r) ->
        let pos = ref 0 in
        Array.iteri (fun i ax -> if ax = r.a_axis then pos := i) afree;
        let inner = product la (Array.sub afree (!pos + 1) (fa - !pos - 1)) in
        Rows_seg
          {
            ids = ids_side r.ids;
            groups = L.dim lb r.b_axis;
            sbg = L.stride lb r.b_axis;
            inner;
            extent = L.dim la r.a_axis;
          }
    | Some (Contracting g) ->
        let nk = Array.length ac in
        let inner = product la (Array.sub ac (g.pair + 1) (nk - g.pair - 1)) in
        K_seg
          {
            ids = ids_side g.ids;
            groups = g.groups;
            inner;
            extent = L.dim la ac.(g.pair);
          }
  in
  let scale (o : Nx_kernel.operand) =
    match o.scale with
    | None -> None
    | Some s ->
        let arr = match s with Exponent e -> A.Any e | Value v -> A.Any v in
        Some (scale_of (slot arr) o.array s)
  in
  let sa = scale oa and sb = scale ob in
  {
    a;
    b = bs;
    y;
    init;
    sa;
    sb;
    seg;
    acc;
    batch = product la ab;
    m = product la afree;
    n = product lb bcols;
    k = product la ac;
  }

(* The call's launches and slots, the workspace's and the tickets' last,
   kept where it has at most [most] launches and no windows. *)
let finish ?(most = max_int) b =
  let launches = Array.of_list (List.rev b.launches) in
  let own = Array.of_list (List.rev b.slots) in
  let n = Array.length own in
  (* The slots a launch addresses, whatever their bytes: an array a call
     reads only through a pack of no element is in none, and a submission
     takes no slot no part names. Index [n] is the workspace's, [n + 1] the
     tickets'. *)
  let used = Array.make (n + 2) false in
  Array.iter
    (fun (l : _ Launch.t) ->
      for r = 0 to (Array.length l.refs / 3) - 1 do
        let s = l.refs.((3 * r) + 1) in
        used.(if s = ws_slot then n else if s = tickets_slot then n + 1 else s) <-
          true
      done)
    launches;
  let renumber = Array.make (n + 2) (-1) and next = ref 0 in
  Array.iteri
    (fun s u ->
      if u then begin
        renumber.(s) <- !next;
        incr next
      end)
    used;
  let buffers =
    Array.init !next (fun i ->
        let s = ref 0 in
        Array.iteri (fun s' r -> if r = i then s := s') renumber;
        if !s = n then Launch.workspace
        else if !s = n + 1 then Launch.tickets
        else own.(!s))
  in
  (* Each launch's refs are its own, made by [launch]. *)
  Array.iter
    (fun (l : _ Launch.t) ->
      let refs = l.refs in
      for r = 0 to (Array.length refs / 3) - 1 do
        let s = refs.((3 * r) + 1) in
        refs.((3 * r) + 1) <-
          renumber.(if s = ws_slot then n
                    else if s = tickets_slot then n + 1
                    else s)
      done)
    launches;
  {
    Launch.launches;
    buffers;
    workspace = Int.max 1 (Arena.peak b.ws);
    tickets = b.ticket_bytes;
    kept = not (b.windowed || Array.length launches > most);
    frees = b.frees;
  }

let plan ~batch ~contracting ~segments ~acc ~dst ~init oa ob =
  let b = builder () in
  let c = read_call b ~batch ~contracting ~segments ~acc ~dst ~init oa ob in
  let _, cx = ct_of acc in
  let scaled = Option.is_some c.sa || Option.is_some c.sb in
  if scaled && (cx || not (ct_float (fst (ct_of acc)))) then Error A.Declined
  else if L.numel (layout dst) = 0 then Ok []
  else
    (* float8 operands of 8 rows or columns or fewer sum on gemv, which
       decodes them as it loads: the matrix units would take them through
       a pack into bfloat16 that costs more than their sums. *)
    let f8 = is_f8 (dtype_of c.a.arr) || is_f8 (dtype_of c.b.arr) in
    let kind =
      match c.seg with
      | No_segments
        when (not scaled) && c.k <= max_extent
             && not (f8 && Int.min c.m c.n <= K.gemv_rows) ->
          mma_kind (dtype_of c.a.arr) (dtype_of c.b.arr) acc
      | _ -> None
    in
    let result =
      match kind with
      | Some kind ->
          mma b c kind;
          Ok ()
      | None -> fma b c
    in
    match result with Error e -> Error e | Ok () -> Ok [ finish b ]

(* The matrix routines' launches *)

let operand x = { Nx_kernel.array = x; scale = None }
let one_batch = Iarray.of_array [| (0, 0) |]
let one_k = Iarray.of_array [| (2, 1) |]

let product b o ~acc ~init ~y a bm =
  if L.numel (layout y) > 0 then
    let c =
      read_call b ~batch:one_batch ~contracting:one_k ~segments:None ~acc ~dst:y
        ~init:(if init then Some y else None)
        (operand a) (operand bm)
    in
    match fma ~o b c with
    | Ok () -> ()
    | Error _ -> invalid_arg "Contract.product: an output past a launch"

let copy b ~src:(A.Any x) ~dst:(A.Any y) =
  let l = A.layout x and ly = A.layout y in
  let r = L.rank l in
  let ax =
    {
      ext = Array.init r (L.dim l);
      str = Array.init r (L.stride l);
      grp = Array.make r 0;
    }
  in
  let dt = D.Any (A.dtype x) in
  let dst_where = Own { slot = slot b (A.buffer y); buf = A.buffer y } in
  pack b ~mode:0
    ~src_slot:(slot b (A.buffer x))
    ~src_dt:dt ~src_first:(L.offset l) ~dst_where ~dst_at:0 ~dst_dt:dt
    ~dst_first:(L.offset ly) ~acc:dt ax
    (Array.init r (L.stride ly))
    ()
