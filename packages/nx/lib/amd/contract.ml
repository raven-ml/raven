(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A contraction's launches of contract.hip, as one call:

     packs (a, b, init, ids) --> per window of one launch's bounds:
     a segmented call's prepass, then the contraction --> a pack of y out
     of the workspace

   The plan reads each operand's axes as runs, one run of strides for its
   batch axes, one for its free ones and one for its contracted ones,
   where its layout merges them. It packs into the workspace:

   - an operand whose axes of one kind do not form one run of strides, of
     a sub-byte dtype, scaled, or of the compute type's own dtype whose
     rows are not 16-byte vectors along a contiguous axis;
   - an integer call's operand of another dtype than the native tiles
     read, where the call has more rows than gemv takes, cast once into
     the compute type;
   - a complex call's a not of the accumulator's dtype with its
     contracted axes contiguous, and its b, as each element's 2 x 2 real
     block [[re, -im], [im, re]];
   - an init or a y of another dtype than the accumulator's, but a y's
     integer dtype no wider than an integer accumulator, or of axes that
     do not form one run of strides, or a complex one whose columns are
     apart.

   An operand is copied in C order of its groups (a sub-byte dtype one
   element a byte), or, for a scale, multiplied into the accumulator's
   dtype; a y is computed into the workspace and packed out. Every pack
   copies values exactly or rounds as the call would, so packs change no
   bit.

   The association is fixed by the dtypes and the extents alone: the
   kernel's family by the dtypes (and scales and segments), its tiles and its ranges of k
   by the model below, which reads the extents and the dtypes' widths,
   never a layout. A layout only chooses whether an operand loads in
   place or through a pack. *)

module K = Kernels
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* Dtypes *)

let code (D.Any d) = D.code d
let bits (D.Any d) = D.bits d
let sub_byte dt = bits dt < 8

(* The bytes an element takes in the kernels' operands: a sub-byte one's,
   one a byte, 1. *)
let width dt = Int.max 1 (bits dt / 8)
let is_complex (D.Any d) = D.is Complex d

(* The SIMT kernels' compute type of an accumulator, and whether it is
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
   their extents, strides and groups (0 to 2, and 3 for a Rows call's b's
   slices or a Contracting y's groups axis). *)
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

(* Where a kernel reads an array: its own buffer, or the workspace. A
   buffer takes its slot when a launch first names it: a submission takes
   no slot no launch names, as an operand's of no element. *)
type where = Own of Rig.Buffer.t | Work

(* An array as a kernel reads it: where, its first element's byte, its
   strides along its three groups in elements, its dtype, for a Rows
   call's b packed into the workspace the stride of its slices, and
   whether this plan packed it, laid out for every kernel. A caller's
   matrix in the workspace is not packed. *)
type view = {
  where : where;
  at : int;
  s : int array;
  dt : D.any;
  sg : int;
  packed : bool;
}

(* A matrix a product reads or writes: where, the byte its layout's
   offset counts from, its dtype and its layout. *)
type matrix = { where : where; base : int; dt : D.any; l : L.t }

let array (A.Any x) =
  { where = Own (A.buffer x); base = 0; dt = D.Any (A.dtype x); l = A.layout x }

let work ~at dt l = { where = Work; base = at; dt; l }

(* The call's matrices: each one's layout's runs, or [None] where a group
   does not merge. *)
type side = {
  mat : matrix;
  groups : int array array; (* batch, free, contracted (y: batch, rows, cols) *)
  runs : (int * int) option array;
}

let side_of mat groups = { mat; groups; runs = Array.map (run mat.l) groups }
let layout (A.Any x) = A.layout x
let buffer (A.Any x) = A.buffer x
let grouped s = Array.for_all Option.is_some s.runs
let stride_of s g = match s.runs.(g) with Some (_, st) -> st | None -> 0

let view_of s =
  let l = s.mat.l in
  {
    where = s.mat.where;
    at = s.mat.base + (L.offset l * width s.mat.dt);
    s = Array.init 3 (stride_of s);
    dt = s.mat.dt;
    sg = 0;
    packed = false;
  }

let slot_of b = function Own x -> Plan.slot b x | Work -> Plan.workspace b
let ceil_div a b = (a + b - 1) / b

(* A scale as a pack reads it: its buffer, first element, strides and the
   ratios of its operand's extents to its own along each of the operand's
   axes, its dtype, and whether it is an Exponent. *)
type scale = {
  sbuf : Rig.Buffer.t;
  sfirst : int;
  cs : int array;
  ratio : int array;
  sdt : D.any;
  exponent : bool;
}

let scale_of (A.Any x) (s : Nx_kernel.scale) =
  let arr, sdt, exponent =
    match s with
    | Exponent e -> (A.Any e, D.Any D.Uint8, true)
    | Value v -> (A.Any v, D.Any (A.dtype v), false)
  in
  let l = layout arr and lx = A.layout x in
  {
    sbuf = buffer arr;
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

(* Word [word] of [w]: the byte [off] of [slot]'s buffer, a ref of [refs]. *)
let address refs w word slot off ~write =
  Plan.set w word off;
  refs := { Plan.word; slot; write } :: !refs

let launch b kernel ~groups ~shared w refs =
  Plan.launch b kernel ~groups:(groups, 1, 1) ~shared w
    (Array.of_list (List.rev !refs))

(* Packs *)

(* A pack of the axes [ax] of [src], of [src_dt] from its element
   [src_first], into [dst_where] from its byte [dst_at] and its element
   [dst_first], of [dst_dt], [dst_str] the strides of each axis there.
   [scale] is a scale, with its strides and ratios along [ax]. *)
let pack b ~mode ~src ?(src_at = 0) ~src_dt ~src_first ~dst_where ~dst_at
    ~dst_dt ~dst_first ~acc (ax : axes) dst_str ?scale ?(block = 0) () =
  let numel = Array.fold_left ( * ) 1 ax.ext in
  if numel > 0 then begin
    let w = Plan.words K.Contract_pack and refs = ref [] in
    address refs w K.pack_src (slot_of b src) src_at ~write:false;
    address refs w K.pack_dst (slot_of b dst_where) dst_at ~write:true;
    Plan.set w K.pack_src_first src_first;
    Plan.set w K.pack_dst_first dst_first;
    Plan.set w K.pack_numel numel;
    Plan.set w K.pack_src_dtype (code src_dt);
    Plan.set w K.pack_dst_dtype (code dst_dt);
    Plan.set w K.pack_scale_dtype (-1);
    Plan.set w K.pack_acc_dtype (code acc);
    Plan.set w K.pack_block_word block;
    Plan.set w K.pack_small (Bool.to_int (Plan.small numel));
    (* The axes of more than one element, in order. *)
    let r = ref 0 in
    for i = 0 to Array.length ax.ext - 1 do
      if ax.ext.(i) > 1 then begin
        let at = K.pack_ax + (!r * K.pack_axis) in
        Plan.axis w at ax.ext.(i);
        Plan.set w (at + K.pack_axis_ss) ax.str.(i);
        Plan.set w (at + K.pack_axis_ds) dst_str.(i);
        (match scale with
        | None -> Plan.axis w (at + K.pack_axis_ratio) 1
        | Some (_, (cs : int array), (ratio : int array)) ->
            Plan.set w (at + K.pack_axis_cs) cs.(i);
            Plan.axis w (at + K.pack_axis_ratio) ratio.(i));
        incr r
      end
    done;
    Plan.set w K.pack_rank !r;
    let mode =
      match scale with
      | None -> mode
      | Some (sc, _, _) ->
          address refs w K.pack_scale (Plan.slot b sc.sbuf) 0 ~write:false;
          Plan.set w K.pack_scale_first sc.sfirst;
          Plan.set w K.pack_scale_dtype (code sc.sdt);
          if sc.exponent then mode lor K.pack_exponent else mode
    in
    Plan.set w K.pack_mode mode;
    let groups = Plan.grid K.Contract_pack numel in
    Plan.set w K.pack_groups groups;
    launch b K.Contract_pack ~groups ~shared:0 w refs
  end

(* The strides of a contiguous layout of [ax] whose groups have the
   strides [gs] (batch, free, k, and the fourth group), each group's axes
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
   it along k: the most k a k-tile of any kernel reads, so that a kernel
   reads them from the row's start at every k, the row's k stride 0. *)
let k_copies =
  Array.fold_left
    (fun n k -> match K.tile k with Some t -> Int.max n t.bk | None -> n)
    16 K.all

(* Whether [s]'s group [g] repeats one element: its axes of more than one
   element are all of stride 0, and [scale], where given, has one element
   along them. *)
let repeats s ?scale g =
  match s.runs.(g) with
  | Some (e, 0) when e > 1 -> (
      match scale with
      | None -> true
      | Some sc ->
          let l = s.mat.l in
          Array.for_all
            (fun ax -> L.dim l ax <= 1 || sc.ratio.(ax) = L.dim l ax)
            s.groups.(g))
  | _ -> false

(* An operand a or b packed into the workspace as [into], [batch] x
   [rows] x [lead] elements, [lead] the contracted extent [k] padded to
   16 bytes, after the slices of a Rows call's b along its fourth group;
   with [value], its values cast into [acc], times its scale. A
   group that repeats one element is packed once, its stride 0: batch
   elements or rows as one, and a k past [k_copies] as [k_copies]
   copies. *)
let pack_operand b s ~into ~batch ~rows ~k ~acc ~value ?scale ?(neg = false)
    ?(k_copies = k_copies) () =
  let w = width into in
  let one g = repeats s ?scale g in
  (* A complex operand's parts lie along k, so its k is never one copy. *)
  let rb = one 0
  and rr = one 1
  and rk = one 2 && k > k_copies && not (is_complex into) in
  let lead = if rk then k_copies else ceil_div k (16 / w) * (16 / w) in
  let batch' = if rb then 1 else batch and rows' = if rr then 1 else rows in
  let ax = axes_of s.mat.l s.groups in
  let slices = ref 1 in
  Array.iteri (fun i g -> if g = 3 then slices := !slices * ax.ext.(i)) ax.grp;
  let sg = batch' * rows' * lead in
  let at = Plan.take b (!slices * sg * w) in
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
  let src_dt = s.mat.dt in
  let mode =
    (if value then K.pack_value else 0)
    lor (if neg then K.pack_neg else 0)
    lor if sub_byte src_dt && not value then K.pack_spread_dst else 0
  in
  let gs =
    [| (if rb then 0 else rows' * lead); (if rr then 0 else lead); 1; sg |]
  in
  pack b ~mode ~src:s.mat.where ~src_at:s.mat.base ~src_dt
    ~src_first:(L.offset s.mat.l) ~dst_where:Work ~dst_at:at ~dst_dt:into
    ~dst_first:0 ~acc walked (packed_strides walked gs) ?scale ();
  {
    where = Work;
    at;
    s = [| gs.(0); gs.(1); (if rk then 0 else 1) |];
    dt = into;
    sg;
    packed = true;
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
   [lead] reals a batch element after the slices of a Rows call's b,
   [lead] [2 k] padded to 16 bytes, each element's value cast into
   [acc]. *)
let pack_block ?(mode = 0) b s ~rdt ~batch ~rows ~k ~acc =
  let per = 16 / width rdt in
  let lead = ceil_div (2 * k) per * per in
  let ax = axes_of s.mat.l s.groups in
  let slices = ref 1 in
  Array.iteri (fun i g -> if g = 3 then slices := !slices * ax.ext.(i)) ax.grp;
  let sg = batch * 2 * rows * lead in
  let at = Plan.take b (!slices * sg * width rdt) in
  pack b
    ~mode:(K.pack_value lor K.pack_block lor mode)
    ~src:s.mat.where ~src_at:s.mat.base ~src_dt:s.mat.dt
    ~src_first:(L.offset s.mat.l) ~dst_where:Work ~dst_at:at ~dst_dt:rdt
    ~dst_first:0 ~acc ax
    (packed_strides ax [| 2 * rows * lead; 2 * lead; 2; sg |])
    ~block:lead ();
  {
    where = Work;
    at;
    s = [| 2 * rows * lead; lead; 1 |];
    dt = rdt;
    sg;
    packed = true;
  }

(* An init or y as a contiguous [groups; batch; rows; cols] in the
   workspace, its dtype's elements of a byte or more, a sub-byte one's one
   a byte. *)
let out_strides ~batch ~m ~n = [| m * n; n; 1; batch * m * n |]

(* The model *)

(* An instance as the model reads it: its kernel, compute type, load
   (0 native, 1 converting, 3 gemv), tile, threads, the groups a CU holds
   and shared bytes. *)
type inst = {
  kernel : K.t;
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

(* The CUs of the R9700 (gfx1201) the model is fitted to, and the bytes
   of LDS a CU holds. *)
let cus = 64
let cu_lds = 65536

(* The groups a CU holds at once: by its LDS, and by 16 waves of 32 lanes
   a CU, the waves its registers hold at some 128 VGPRs a lane. *)
let occupancy ~threads ~smem =
  Int.max 1 (Int.min (cu_lds / Int.max 1 smem) (512 / threads))

let gemv_ct : K.t -> K.ct option = function
  | Contract_gemv_f32 -> Some F32
  | Contract_gemv_f64 -> Some F64
  | Contract_gemv_u32 -> Some U32
  | Contract_gemv_u64 -> Some U64
  | _ -> None

let insts =
  let of_kernel k =
    match (K.tile k, gemv_ct k) with
    | Some g, _ ->
        let threads = g.wm * g.wn * 32 in
        Some
          {
            kernel = k;
            t = g.t;
            ld = (match g.load with Native -> 0 | Convert -> 1);
            bm = g.bm;
            bn = g.bn;
            bk = g.bk;
            tm = g.tm;
            tn = g.tn;
            threads;
            occ = occupancy ~threads ~smem:g.lds;
            smem = g.lds;
          }
    | None, Some t ->
        (* gemv_ranges weighs its shared memory. *)
        Some
          {
            kernel = k;
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
    | None, None -> None
  in
  Array.of_list (List.filter_map of_kernel (Array.to_list K.all))

(* A CU's rate holding [a] groups, reached from [sat] of them. *)
let[@inline] share sat a = Float.min 1. (Float.of_int a /. sat)

(* The time, in units of a CU's multiply-adds, of [n1] groups of work
   [w1] then [n2] of [w2] on [cus] CUs of [c] groups each, a CU's rate
   shared by its groups and reached from [sat] of them: whole waves of
   the first, then the rest spread over the CUs. *)
let[@inline] makespan n1 w1 n2 w2 c sat =
  let cap = cus * c in
  let waves = n1 / cap in
  let t = Float.of_int (waves * c) *. w1 /. share sat c in
  let r1 = n1 - (waves * cap) in
  let groups = r1 + n2 in
  if groups = 0 then t
  else
    let work = (Float.of_int r1 *. w1) +. (Float.of_int n2 *. w2) in
    let wmax =
      Float.max (if r1 > 0 then w1 else 0.) (if n2 > 0 then w2 else 0.)
    in
    if groups <= cap then
      let a = ceil_div groups cus in
      t
      +. Float.max
           (Float.of_int a *. work /. Float.of_int groups /. share sat a)
           (wmax /. share sat 1)
    else
      t
      +. (work /. (Float.of_int cus *. share sat c))
      +. (0.5 *. wmax /. share sat 1)

(* A CU's multiply-adds a microsecond, at 2.9 GHz: f32, 64 a clock times
   the instance's share of its loop, a k-tile's FMAs against its fragment
   loads and some 80 other instructions, more for a converting load; f64,
   2 a clock; 32-bit integers, a multiply at a quarter rate and an add,
   12 a clock, a quarter for 64 bits. A model to fit by timing. *)
let[@inline] cu_rate i =
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
  if not fl then 12. *. 2900. *. share *. if b = 8 then 0.25 else 1.
  else if b = 8 then 2. *. 2900.
  else 64. *. 2900. *. share

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
   element, its first [main] tiles in [s1] ranges and the others in [s2];
   memory at 640 GB/s, the L2 at some 2 TB/s. *)
let[@inline] cost c i ~tm ~tn ~main ~s1 ~s2 =
  let tiles = c.batch * tm * tn in
  let kk = Float.of_int (Int.max c.k 1) in
  let w = Float.of_int (i.bm * i.bn) *. kk in
  let tb = ct_bytes i.t in
  let sat =
    Float.max 1.
      ((if tb = 8 && ct_float i.t then 128. else 256.) /. Float.of_int i.threads)
  in
  let rate = cu_rate i in
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
  let dram = ab /. 6.4e5 in
  let bn = Float.of_int (if c.cx then i.bn / 2 else i.bn) in
  let l2 =
    ((Float.of_int tiles *. kr *. ((Float.of_int i.bm *. c.wa) +. (bn *. c.wb)))
    +. part)
    /. 2.0e6
  in
  Float.max t (Float.max dram l2 +. 1.5) +. 1.5

(* A plan of the SIMT kernels: the instance, its tiles a batch element,
   and its schedule. *)
type plan = {
  i : inst;
  tiles_m : int;
  tiles_n : int;
  tiles : int;
  main : int;
  s1 : int;
  s2 : int;
}

let groups p = (p.main * p.s1) + ((p.tiles - p.main) * p.s2)

(* gemv's shared bytes for [m] rows: a's stretch, and each wave's rows of
   b's values (NX_GEMV_STAGE). *)
let gemv_shared ct ~m =
  let stage = if ct_bytes ct = 4 then K.gemv_stage_4 else K.gemv_stage_8 in
  (ceil_div (m * K.gemv_stretch * ct_bytes ct) 16 * 16)
  + (K.gemv_threads / 32 * stage)

(* gemv's ranges of k for [tiles] tiles of [runs] runs on [i]: one wave of
   the groups the CUs hold at once, by [i.occ] and LDS, at most 64. *)
let gemv_ranges i c ~tiles ~runs =
  let occ = Int.min i.occ (cu_lds / (gemv_shared i.t ~m:c.m + 1024)) in
  Int.max 1 (Int.min 64 (Int.min runs (cus * Int.max 1 occ / tiles)))

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
      let cap = cus * i.occ in
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

(* Bounds *)

(* The most rows and columns a launch takes: each coordinate the kernels
   form fits an int32. *)
let max_extent = 1 lsl 30
let max_int32 = 0x7fff_ffff

(* Contraction parameters *)

let write_operand b w at (v : view) ~rows ~flags refs =
  address refs w (at + K.operand_p) (slot_of b v.where) v.at ~write:false;
  Plan.set w (at + K.operand_s) v.s.(0);
  Plan.set w (at + K.operand_s + 1) v.s.(1);
  Plan.set w (at + K.operand_s + 2) v.s.(2);
  Plan.set w (at + K.operand_rows) (Int.max rows 1);
  Plan.set w (at + K.operand_dtype) (code v.dt);
  Plan.set w (at + K.operand_flags) flags

(* Whether [v]'s first element lies on 16 bytes, as the device addresses
   it: a workspace piece's does. *)
let aligned16 (v : view) =
  match v.where with
  | Work -> v.at mod 16 = 0
  | Own x -> Rig.Buffer.aligned x ~at:v.at 16

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

(* What a matrix routine's product asks beyond {!Nx_kernel.S.contract}:
   its first range of k summing from init, a negated, a conjugated a or b,
   a triangle of tiles, and [ranges] of k for every tile in place of the
   model's. *)
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

(* A routine's ranges for an output of [rows] x [cols] and [k] terms, a
   function of these extents alone: several where its tiles do not fill
   the CUs and [k] is long. *)
let ranges ~rows ~cols ~k =
  let tiles = ceil_div rows 64 * ceil_div cols 64 in
  if tiles < cus && k >= 2048 then
    Int.min (ceil_div k 512) (ceil_div (2 * cus) tiles)
  else 1

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
   it takes in the workspace for any other: the accumulator's, a complex
   one's, and integers of the compute type's width for init, no wider than
   an integer accumulator for y. *)
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

(* init as the kernels read it, and its groups' stride: in place where its
   axes merge and its dtype is the kernel's own, else cast into [st.wdt]
   in the workspace, a group that repeats one element packed once, its
   stride 0. *)
let init_view b c st =
  match c.init with
  | None -> None
  | Some s ->
      let dt = s.mat.dt in
      let own = match c.seg with K_seg _ -> L.stride s.mat.l 0 | _ -> 0 in
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
        let at = Plan.take b (kgroups c * gs.(3) * width st.wdt) in
        let ax = axes_of s.mat.l s.groups in
        let keep = Array.map (fun g -> g = 3 || not (one g)) ax.grp in
        let pick = kept keep in
        let walked =
          { ext = pick ax.ext; str = pick ax.str; grp = pick ax.grp }
        in
        pack b ~mode:K.pack_value ~src:s.mat.where ~src_at:s.mat.base ~src_dt:dt
          ~src_first:(L.offset s.mat.l) ~dst_where:Work ~dst_at:at
          ~dst_dt:st.wdt ~dst_first:0 ~acc:c.acc walked
          (packed_strides walked gs) ();
        Some
          ( {
              where = Work;
              at;
              s = Array.sub gs 0 3;
              dt = st.wdt;
              sg = 0;
              packed = true;
            },
            gs.(3) )
      end

(* y as the kernels write it, its groups' stride, and whether it lies in
   the workspace: in place where its axes merge and its dtype is the
   kernel's own, else in [st.wdt] in the workspace. *)
let y_view b c st =
  let s = c.y in
  let dt = s.mat.dt in
  let own = match c.seg with K_seg _ -> L.stride s.mat.l 0 | _ -> 0 in
  let cols = (not (snd (ct_of c.acc))) || c.n <= 1 || stride_of s 2 = 1 in
  if grouped s && st.own_y dt && cols then (view_of s, own, false)
  else
    let at = Plan.take b (kgroups c * c.batch * c.m * c.n * width st.wdt) in
    ( {
        where = Work;
        at;
        s = [| c.m * c.n; c.n; 1 |];
        dt = st.wdt;
        sg = 0;
        packed = true;
      },
      c.batch * c.m * c.n,
      true )

(* y's values into the workspace where [y_view] put it, so that the tiles
   a triangle skips keep them. *)
let keep_y b c (v : view) =
  let s = c.y in
  let ax = axes_of s.mat.l s.groups in
  let dst = packed_strides ax (out_strides ~batch:c.batch ~m:c.m ~n:c.n) in
  pack b ~mode:K.pack_value ~src:s.mat.where ~src_at:s.mat.base ~src_dt:s.mat.dt
    ~src_first:(L.offset s.mat.l) ~dst_where:Work ~dst_at:v.at ~dst_dt:v.dt
    ~dst_first:0 ~acc:c.acc ax dst ()

(* y cast out of the workspace, where [y_view] put it. *)
let unpack b c (v : view) =
  let s = c.y in
  let ax = axes_of s.mat.l s.groups in
  let src = packed_strides ax (out_strides ~batch:c.batch ~m:c.m ~n:c.n) in
  pack b ~mode:K.pack_cast ~src:Work ~src_dt:v.dt
    ~src_first:(v.at / width v.dt)
    ~dst_where:s.mat.where ~dst_at:s.mat.base ~dst_dt:s.mat.dt
    ~dst_first:(L.offset s.mat.l) ~acc:c.acc { ax with str = src } ax.str ()

(* The words of a launch of [kernel] over [c]'s operands, init and y. *)
let params b kernel c ~a ~b:bv ~fa ~fb ~init ~(y : view) ~flags ~ct_k ~n refs =
  let w = Plan.words kernel in
  write_operand b w K.contract_a a ~rows:c.m ~flags:fa refs;
  write_operand b w K.contract_b bv ~rows:n ~flags:fb refs;
  (match init with
  | None -> Plan.set w K.contract_init_dtype (code c.y.mat.dt)
  | Some ((v : view), _) ->
      address refs w K.contract_init (slot_of b v.where) v.at ~write:false;
      Plan.set w K.contract_si v.s.(0);
      Plan.set w (K.contract_si + 1) v.s.(1);
      Plan.set w (K.contract_si + 2) v.s.(2);
      Plan.set w K.contract_init_dtype (code v.dt));
  address refs w K.contract_y (slot_of b y.where) y.at ~write:true;
  Plan.set w K.contract_sy y.s.(0);
  Plan.set w (K.contract_sy + 1) y.s.(1);
  Plan.set w (K.contract_sy + 2) y.s.(2);
  Plan.set w K.contract_y_dtype (code y.dt);
  Plan.set w K.contract_acc_dtype (code c.acc);
  Plan.set w K.contract_flags flags;
  Plan.set w K.contract_batch c.batch;
  Plan.set w K.contract_m c.m;
  Plan.set w K.contract_n n;
  Plan.set w K.contract_k ct_k;
  w

(* A view of [v]'s window from batch element [z], row [i] (a's, or b's
   column, or y's row and column [j]). *)
let shift (v : view) ~z ~i ~j =
  {
    v with
    at = v.at + (((z * v.s.(0)) + (i * v.s.(1)) + (j * v.s.(2))) * width v.dt);
  }

(* Segments *)

(* A segmented call's ids as the prepasses read them: where, the byte of
   their first element, and their strides along batch and the segmented
   axis; packed, int64 [batch; extent], where their batch axes do not
   merge. *)
let ids_view b c (s : side) ~extent =
  match s.runs.(0) with
  | Some (_, sz) ->
      let l = s.mat.l in
      (s.mat.where, s.mat.base + (8 * L.offset l), sz, L.stride l (L.rank l - 1))
  | None ->
      let dt = D.Any D.Int64 in
      let at = Plan.take b (8 * c.batch * extent) in
      let ax = axes_of s.mat.l s.groups in
      pack b ~mode:0 ~src:s.mat.where ~src_at:s.mat.base ~src_dt:dt
        ~src_first:(L.offset s.mat.l) ~dst_where:Work ~dst_at:at ~dst_dt:dt
        ~dst_first:0 ~acc:dt ax
        (packed_strides ax [| extent; 1; 0; 0 |])
        ();
      (Work, at, extent, 1)

(* The launches of [kernel], contract_rows or contract_ks, one a phase
   over its groups [phases]: [ids] as [ids_view] gives them, each table a
   workspace byte, [-1] for none, contract_ks's [perm] of int64 words
   where [wide]. *)
let segment_launches ?(wide = false) b kernel ~phases ~ids:(where, at, sz, sx)
    ~counts ~cursor ~perm ~desc ~ntiles ~batch ~count ~first ~extent ~inner
    ~groups ~bm =
  List.iteri
    (fun phase grid ->
      let w = Plan.words kernel and refs = ref [] in
      address refs w K.seg_ids (slot_of b where) at ~write:false;
      List.iter
        (fun (f, at) ->
          if at >= 0 then address refs w f (Plan.workspace b) at ~write:true)
        [
          (K.seg_counts, counts);
          (K.seg_cursor, cursor);
          (K.seg_perm, perm);
          (K.seg_desc, desc);
          (K.seg_ntiles, ntiles);
        ];
      Plan.set w K.seg_sz sz;
      Plan.set w K.seg_sx sx;
      Plan.set w K.seg_inner inner;
      Plan.set w K.seg_count count;
      Plan.set w K.seg_extent extent;
      Plan.set w K.seg_first first;
      Plan.set w K.seg_batch batch;
      Plan.set w K.seg_groups groups;
      Plan.set w K.seg_bm bm;
      Plan.set w K.seg_phase phase;
      Plan.set w K.seg_wide (Bool.to_int wide);
      Plan.set w K.seg_grid grid;
      launch b kernel ~groups:grid ~shared:0 w refs)
    phases

(* The SIMT kernels *)

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
   indices, which take twice the memory of int32 ones: 2^31 - 1, lower
   only in tests. *)
let wide_k = Atomic.make max_int32

let contract_wide_k n =
  if n < 0 || n > max_int32 then invalid_arg "Nx_amd.contract_wide_k";
  Atomic.set wide_k n

(* A Contracting call's prepass counts each group's k in parts of at least
   [ks_part] k, at most [ks_counts] counts a window. *)
let ks_part = 4096
let ks_counts = 1 lsl 22

(* NX_CONTRACT_Y_VECTORS: y of the compute type's width, its columns
   contiguous and its rows, batch elements and groups whole vectors from
   a first element on one. *)
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

(* a or b as a SIMT kernel reads it: scaled, the scale multiplied into the
   accumulator's dtype; of a sub-byte dtype or axes that do not merge,
   packed as it is. *)
let fma_operand b c s scale ~rows ~ct =
  let dt = s.mat.dt in
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
  if v.packed then v
  else if
    vectors v ~bytes ~rows:tile
    && ((not k_rows) || v.s.(2) = 1)
    && shift mod (16 / bytes) = 0
  then v
  else
    pack_operand b s ~into:v.dt ~batch:c.batch ~rows ~k:c.k ~acc:c.acc
      ~value:false ()

(* The launches of [plans], each window's plan: for a segmented call its
   prepass, then the contraction. The windows run one after another in the
   call, so they share the tables and partials, sized by the largest. *)
let fma_launches b c ~o ~ct ~cx ~k' ~(a : view) ~(bv : view) ~init ~(y : view)
    ~ystep plans =
  (* A complex call's columns, real ones. *)
  let jm = if cx then 2 else 1 in
  let most f = List.fold_left (fun n x -> Int.max n (f x)) 0 plans in
  let nb_most = most (fun ((_, _, _, nb, _, _), _) -> nb) in
  let kwide =
    match c.seg with K_seg _ -> c.k > Atomic.get wide_k | _ -> false
  in
  let grid = Plan.grid K.Contract_rows in
  (* The segment tables, and each window's prepass and words. *)
  let seg =
    match c.seg with
    | No_segments -> fun _ _ _ _ -> ()
    | Rows_seg r ->
        let g = r.groups + 1 in
        let counts = Plan.take b (4 * ((2 * nb_most * g) + 1)) in
        let cursor = Plan.take b (4 * nb_most * g) in
        let perm =
          Plan.take b (4 * most (fun ((_, _, _, nb, nm, _), _) -> nb * nm))
        in
        let desc = Plan.take b (16 * most (fun (_, p) -> p.tiles_m)) in
        let ntiles = Plan.take b 4 in
        let where, at, sz, sx = ids_view b c r.ids ~extent:r.extent in
        fun (z, i, _, nb, nm, _) (p : plan) w refs ->
          segment_launches b K.Contract_rows
            ~phases:
              [
                grid (nb * g);
                grid (nb * nm);
                1;
                grid (Int.max (nb * nm) p.tiles_m);
              ]
            ~ids:(where, at + (8 * z * sz), sz, sx)
            ~counts ~cursor ~perm ~desc ~ntiles ~batch:nb ~count:nm ~first:i
            ~extent:r.extent ~inner:r.inner ~groups:r.groups ~bm:p.i.bm;
          let ws = Plan.workspace b in
          address refs w K.contract_perm ws perm ~write:false;
          address refs w K.contract_desc ws desc ~write:false;
          address refs w K.contract_ntiles ws ntiles ~write:false;
          Plan.set w K.contract_sbg (if bv.packed then bv.sg else r.sbg);
          Plan.set w K.contract_groups r.groups
    | K_seg g ->
        let gg = g.groups + 1 in
        (* A batch element's k in parts of at least [ks_part], whose
           groups' counts take at most [ks_counts] words. *)
        let parts nb =
          Int.max 1
            (Int.min (ceil_div c.k ks_part)
               (ks_counts / Int.max 1 (nb * g.groups)))
        in
        let koff = Plan.take b (8 * nb_most * gg) in
        let cursor = Plan.take b (8 * Int.max 1 (nb_most * g.groups)) in
        let counts =
          Plan.take b
            (8
            * Int.max 1
                (most (fun ((_, _, _, nb, _, _), _) -> nb * parts nb * g.groups))
            )
        in
        let kperm =
          Plan.take b ((if kwide then 8 else 4) * Int.max 1 (nb_most * c.k))
        in
        let where, at, sz, sx = ids_view b c g.ids ~extent:g.extent in
        fun (z, _, _, nb, _, _) _ w refs ->
          let p = parts nb in
          segment_launches ~wide:kwide b K.Contract_ks
            ~phases:
              [
                grid (nb * p * g.groups);
                nb * p;
                grid (nb * g.groups);
                grid nb;
                nb * p;
              ]
            ~ids:(where, at + (8 * z * sz), sz, sx)
            ~counts:koff ~cursor ~perm:kperm ~desc:counts ~ntiles:(-1) ~batch:nb
            ~count:c.k ~first:0 ~extent:g.extent ~inner:g.inner ~groups:g.groups
            ~bm:p;
          let ws = Plan.workspace b in
          address refs w K.contract_kperm ws kperm ~write:false;
          address refs w K.contract_koff ws koff ~write:false;
          Plan.set w K.contract_kgroups g.groups;
          Plan.set w K.contract_syg ystep;
          Plan.set w K.contract_sig
            (match init with Some (_, st) -> st | None -> 0)
  in
  (* The split tiles' partials, the most any window takes, and their
     tickets. *)
  let bytes = ct_bytes ct in
  let split_of (p : plan) =
    (if p.s1 > 1 then p.main * p.s1 else 0)
    + if p.s2 > 1 then (p.tiles - p.main) * p.s2 else 0
  in
  let part_bytes = most (fun (_, p) -> split_of p * p.i.bm * p.i.bn * bytes) in
  let partials = if part_bytes > 0 then Plan.take b part_bytes else 0 in
  let fa = operand_flags a and fb = operand_flags bv in
  let kflags =
    (match c.seg with K_seg _ when cx -> K.contract_k_pairs | _ -> 0)
    lor if kwide then K.contract_k_wide else 0
  in
  (* A triangle's tiles are those of the whole call's outputs. *)
  let tri =
    match o.tri with
    | Some `Lower when List.length plans = 1 -> K.contract_lower
    | Some `Upper when List.length plans = 1 -> K.contract_upper
    | _ -> 0
  in
  let extra = (if o.init_first then K.contract_init_first else 0) lor tri in
  List.iter
    (fun (((z, i, j, nb, nm, nn) as win), p) ->
      let refs = ref [] in
      let j = jm * j in
      let yw = shift y ~z ~i ~j in
      let initw = Option.map (fun (v, st) -> (shift v ~z ~i ~j, st)) init in
      let cw = { c with batch = nb; m = nm; n = nn } in
      let w =
        params b p.i.kernel cw ~a:(shift a ~z ~i ~j:0)
          ~b:(shift bv ~z ~i:j ~j:0) ~fa ~fb ~init:initw ~y:yw
          ~flags:(fma_flags ct ~y:yw ~ystep lor kflags lor extra)
          ~ct_k:k' ~n:(jm * nn) refs
      in
      Plan.set w K.contract_tiles_m p.tiles_m;
      Plan.set w K.contract_tiles_n p.tiles_n;
      Plan.set w K.contract_main p.main;
      Plan.set w K.contract_s1 p.s1;
      Plan.set w K.contract_s2 p.s2;
      seg win p w refs;
      if split_of p > 0 then begin
        address refs w K.contract_partials (Plan.workspace b) partials
          ~write:true;
        address refs w K.contract_tickets (Plan.tickets_slot b)
          (Plan.tickets b p.tiles) ~write:true
      end;
      let shared = if p.i.ld = 3 then gemv_shared ct ~m:nm else p.i.smem in
      launch b p.i.kernel ~groups:(groups p) ~shared w refs)
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
  let at = c.a.mat.dt in
  o = plain && unseg && c.n <= K.gemv_rows && c.m > K.gemv_rows
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
    if packed_y && o.tri <> None then keep_y b c y;
    let rdt = ct_dtype ct in
    (* A negated a, a routine's, is a copy of its values negated, so that
       no load tests for it. *)
    let negated into =
      pack_operand b c.a ~into ~batch:c.batch ~rows:c.m ~k:c.k ~acc:c.acc
        ~value:true ~neg:true ()
    in
    (* A complex call's real views: init and y in place or as [init_view]
       and [y_view] laid them out, a cast into the accumulator, b's
       blocks. *)
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
            c.a.mat.dt = c.acc && grouped c.a
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
    (* An unsegmented integer call of more rows than gemv takes whose
       operands convert: each operand cast once into the compute type in
       the workspace, as the converting loads would cast it, then the
       native tiles, whose loads do not decode every element once a
       tile. *)
    let a, bv, ld =
      if
        ld = 1 && unseg && o = plain && (not cx)
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
      o = plain && unseg && c.m <= K.gemv_rows
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
        wa = Float.of_int (bits c.a.mat.dt) /. 8.;
        wb = Float.of_int (bits c.b.mat.dt) /. 8.;
        wy = Float.of_int (bits c.y.mat.dt) /. 8.;
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
        if a.packed || vectors a ~bytes:(ct_bytes ct) ~rows:whole.i.bm then
          (a, bv)
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
          match c.seg with Rows_seg r when not bv.packed -> r.sbg | _ -> 0
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
      && groups (plan_of ~batch ~m ~n) <= max_int32
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
    | None -> Error A.Declined

(* The matrix units *)

(* The dtypes the matrix units read the operands [a] and [b] of a sum into [acc]
   in, if they sum it: float16, bfloat16 or float8 pairs into float32, a float8
   operand beside a float16 or bfloat16 one decoded into its partner's dtype,
   which holds it exactly; 8-bit integers into a 32-bit integer, a 4-bit one
   widened to 8 bits. *)
let unit_dtypes ~acc a b =
  let f8 = function
    | D.Any (Float8_e4m3fn | Float8_e5m2) -> true
    | _ -> false
  in
  let wide = function
    | D.Any (Int8 | Int4) -> Some (D.Any D.Int8)
    | D.Any (Uint8 | Uint4) -> Some (D.Any D.Uint8)
    | _ -> None
  in
  let half = function D.Any (Float16 | Bfloat16) -> true | _ -> false in
  match (acc, a, b) with
  | D.Any Float32, a, b when half a && (b = a || f8 b) -> Some (a, a)
  | D.Any Float32, a, b when f8 a && half b -> Some (b, b)
  | D.Any Float32, a, b when f8 a && f8 b -> Some (a, b)
  | D.Any (Int32 | Uint32), a, b -> (
      match (wide a, wide b) with Some a, Some b -> Some (a, b) | _ -> None)
  | _ -> None

(* The tile of the units' kernel [kind] (or of a kind summing every pair), its
   kernel, and the kernel's dynamic LDS: its two stages. *)
let unit_tile kind =
  let found = ref None in
  Array.iter
    (fun k ->
      match K.wtile k with
      | Some t when t.kind = kind && !found = None -> found := Some (k, t)
      | _ -> ())
    K.all;
  Option.get !found

(* The groups of the units' kernel [k], of tile [t], the CUs hold at once. *)
let unit_resident k (t : K.wtile) =
  cus * occupancy ~threads:(K.threads k) ~smem:t.lds

(* The ranges of k an output sums apart on [tiles] tiles of which [resident]
   groups run at once: one a [least] runs of NX_WMMA_RUN terms, few rows or
   columns, or few outputs, which fill the GPU only with their k cut, taking
   more; at least one and at most 64, and no more than the tiles' groups
   fill the resident ones with, which bounds the ranges' sums held. *)
let unit_splits ~batch ~m ~n ~k ~tiles ~resident =
  let least =
    if m <= 4 || n <= 4 || batch * m * n <= 1 lsl 18 then 16 else 256
  in
  let fill = Int.min 64 (resident / tiles) in
  Int.max 1 (Int.min fill (ceil_div k K.wmma_run / least))

(* The rows of the tile every call on the units takes, 16 or 128, or 0 for
   unit_kernel's choice. *)
let unit_rows = Atomic.make 0

let contract_unit_tile rows =
  match rows with
  | None -> Atomic.set unit_rows 0
  | Some ((16 | 128) as r) -> Atomic.set unit_rows r
  | Some _ -> invalid_arg "Nx_amd.contract_unit_tile: rows"

(* A tile's time against the 128 x 128 tile's for the same outputs: nx.cuda's
   ratio of its 16 x 64 tile to its 128 x 128 one on sm_89, until the R9700's
   are timed. *)
let unit_weight (t : K.wtile) = if t.bm >= 128 then 1 else 3

(* The units' kernel of operands of [a]'s dtype for [batch] outputs of [m] x
   [n] and [k] terms, its tile and its ranges: 16 rows or fewer the 16-row
   tile, which sums every pair; otherwise the tile of least modelled time,
   its groups dealt over the CUs, the larger on a tie. A function of the
   dtypes and extents alone. *)
let unit_kernel ~batch ~m ~n ~k (a : D.any) =
  let forced = Atomic.get unit_rows in
  let plan (kernel, t) =
    let tiles = batch * ceil_div m t.K.bm * ceil_div n t.bn in
    let resident = unit_resident kernel t in
    let s = unit_splits ~batch ~m ~n ~k ~tiles ~resident in
    let cost = ceil_div (tiles * s) cus * t.bm * t.bn * unit_weight t in
    (kernel, t, s, cost)
  in
  let ((_, _, _, small_cost) as small) = plan (unit_tile K.Any) in
  if forced = 16 || (m <= 16 && forced <> 128) then small
  else
    let kind =
      match a with
      | D.Any D.Float16 -> K.F16
      | D.Any D.Bfloat16 -> K.Bf16
      | _ -> K.Bits8
    in
    let ((_, _, _, large_cost) as large) = plan (unit_tile kind) in
    if small_cost < large_cost && forced = 0 then small else large

(* The most batch elements a units launch takes: the grid's z. *)
let max_batch = 65535

(* The most a k-tile of the units reads of a row repeated along k: a
   k-tile reads [bkb] bytes of a row, so at most [bkb] elements. *)
let unit_copies =
  Array.fold_left
    (fun n k -> match K.wtile k with Some t -> Int.max n t.bkb | None -> n)
    k_copies K.all

let unit_store ~integer =
  let own = function
    | D.Any (Int32 | Uint32) -> integer
    | D.Any (Float32 | Bfloat16 | Float16) -> not integer
    | _ -> false
  in
  {
    own_y = own;
    own_init = own;
    wdt = (if integer then D.Any D.Int32 else D.Any D.Float32);
  }

(* Whether [c] sums on the units: unsegmented, unscaled, of products, at most
   2^30 of them, of operands [unit_dtypes] takes. *)
let on_units c =
  let unseg = match c.seg with No_segments -> true | _ -> false in
  unseg && c.sa = None && c.sb = None && c.k > 0 && c.k <= max_extent
  && Option.is_some (unit_dtypes ~acc:c.acc c.a.mat.dt c.b.mat.dt)

let rec units b c =
  if c.n <= 16 && c.m > 16 then units b (transposed c)
  else
    let da, db = Option.get (unit_dtypes ~acc:c.acc c.a.mat.dt c.b.mat.dt) in
    let integer =
      match c.acc with D.Any (Int32 | Uint32) -> true | _ -> false
    in
    let st = unit_store ~integer in
    let init = init_view b c st in
    let y, _, packed_y = y_view b c st in
    let kernel, t, splits, _ =
      unit_kernel ~batch:c.batch ~m:c.m ~n:c.n ~k:c.k da
    in
    let es = width da in
    (* An operand the units read in place: of its dtype, its axes merged, its
       rows 16-byte vectors along k or along its free axis; packed with k
       contiguous otherwise. A tile's part past the operand's extents loads
       element by element. *)
    let operand s ~into ~rows ~tile_rows =
      let packed () =
        pack_operand b s ~into ~batch:c.batch ~rows ~k:c.k ~acc:c.acc
          ~value:(s.mat.dt <> into) ~k_copies:unit_copies ()
      in
      if s.mat.dt <> into || not (grouped s) then packed ()
      else
        let v = view_of s in
        if vectors v ~bytes:es ~rows:tile_rows then v else packed ()
    in
    let a = operand c.a ~into:da ~rows:c.m ~tile_rows:t.bm in
    let bv = operand c.b ~into:db ~rows:c.n ~tile_rows:t.bn in
    let fits ~batch ~m ~n =
      batch <= max_batch && m <= max_extent && n <= max_extent
      && ceil_div m t.bm * ceil_div n t.bn <= max_int32
    in
    (* A window of one output fits. *)
    let ws = Option.get (windows ~fits (0, 0, 0, c.batch, c.m, c.n)) in
    let tiles (_, _, _, nb, nm, nn) =
      nb * ceil_div nm t.bm * ceil_div nn t.bn
    in
    let most = List.fold_left (fun n w -> Int.max n (tiles w)) 0 ws in
    let partials =
      if splits > 1 then Plan.take b (most * splits * t.bm * t.bn * 4) else 0
    in
    List.iter
      (fun ((z, i, j, nb, nm, nn) as win) ->
        let refs = ref [] in
        let yw = shift y ~z ~i ~j in
        let initw = Option.map (fun (v, st) -> (shift v ~z ~i ~j, st)) init in
        let cw = { c with batch = nb; m = nm; n = nn } in
        let w =
          params b kernel cw ~a:(shift a ~z ~i ~j:0) ~b:(shift bv ~z ~i:j ~j:0)
            ~fa:(operand_flags a) ~fb:(operand_flags bv) ~init:initw ~y:yw
            ~flags:0 ~ct_k:c.k ~n:nn refs
        in
        let tm = ceil_div nm t.bm and tn = ceil_div nn t.bn in
        Plan.set w K.contract_tiles_m tm;
        Plan.set w K.contract_tiles_n tn;
        Plan.set w K.contract_s1 splits;
        if splits > 1 then begin
          address refs w K.contract_partials (Plan.workspace b) partials
            ~write:true;
          address refs w K.contract_tickets (Plan.tickets_slot b)
            (Plan.tickets b (tiles win))
            ~write:true
        end;
        Plan.launch b kernel
          ~groups:(tm * tn, splits, nb)
          ~shared:t.lds w
          (Array.of_list (List.rev !refs)))
      ws;
    if packed_y then unpack b c y

(* The call *)

let free rank (used : int array) =
  Array.of_list
    (List.filter (fun x -> not (Array.mem x used)) (List.init rank Fun.id))

let product l axes = Array.fold_left (fun n ax -> n * L.dim l ax) 1 axes

(* The call read from the arrays. *)
let read_call ~batch ~contracting ~segments ~acc ~(dst : A.any) ~init
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
  let a = side_of (array oa.array) [| ab; afree; ac |] in
  (* Rows' b_axis is a fourth group, which a pack keeps. *)
  let bgroups =
    match segments with
    | Some (Rows r) -> [| bb; bcols; bc; [| r.b_axis |] |]
    | _ -> [| bb; bcols; bc |]
  in
  let bs = side_of (array ob.array) bgroups in
  let y = side_of (array dst) ygroups in
  let init = Option.map (fun i -> side_of (array i) ygroups) init in
  let ids_side ids =
    side_of (array (A.Any ids)) [| Array.init nb Fun.id; [| nb |] |]
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
    match o.scale with None -> None | Some s -> Some (scale_of o.array s)
  in
  {
    a;
    b = bs;
    y;
    init;
    sa = scale oa;
    sb = scale ob;
    seg;
    acc;
    batch = product la ab;
    m = product la afree;
    n = product lb bcols;
    k = product la ac;
  }

let plan ~batch ~contracting ~segments ~acc ~dst ~init oa ob =
  let ct, cx = ct_of acc in
  let scaled =
    Option.is_some oa.Nx_kernel.scale || Option.is_some ob.Nx_kernel.scale
  in
  if scaled && (cx || not (ct_float ct)) then Error A.Declined
  else if L.numel (layout dst) = 0 then Ok None
  else
    let b = Plan.make () in
    let c = read_call ~batch ~contracting ~segments ~acc ~dst ~init oa ob in
    let planned = if on_units c then Ok (units b c) else fma b c in
    Result.map (fun () -> Plan.finish_some b) planned

(* The matrix routines' launches *)

let product b o ~acc ~init ~y a m =
  if L.numel y.l > 0 then begin
    let groups3 = [| [| 0 |]; [| 1 |]; [| 2 |] |] in
    let ys = side_of y groups3 in
    let c =
      {
        a = side_of a groups3;
        b = side_of m [| [| 0 |]; [| 2 |]; [| 1 |] |];
        y = ys;
        init = (if init then Some ys else None);
        sa = None;
        sb = None;
        seg = No_segments;
        acc;
        batch = L.dim a.l 0;
        m = L.dim a.l 1;
        n = L.dim m.l 2;
        k = L.dim a.l 2;
      }
    in
    match fma ~o b c with
    | Ok () -> ()
    | Error _ -> invalid_arg "Contract.product: an output past a launch"
  end

let copy b ~src ~dst =
  let l = src.l in
  let r = L.rank l in
  let ax =
    {
      ext = Array.init r (L.dim l);
      str = Array.init r (L.stride l);
      grp = Array.make r 0;
    }
  in
  pack b ~mode:0 ~src:src.where ~src_at:src.base ~src_dt:src.dt
    ~src_first:(L.offset l) ~dst_where:dst.where ~dst_at:dst.base ~dst_dt:src.dt
    ~dst_first:(L.offset dst.l) ~acc:src.dt ax
    (Array.init r (L.stride dst.l))
    ()
