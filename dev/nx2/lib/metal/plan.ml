(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call's launches: which kernel it runs, over which threadgroups, with
   which parameters. Each output's association is fixed by k, the dtypes
   and the order of summation the call names, so that any window of a
   result computed alone has the bits the whole gives it: the extents of
   the batch, the rows and the columns choose only the kernel's tile and
   grid, never how an output adds its terms. In chain order, products run
   on the dense tiles, which all sum an output alike; in lane order, float
   products run on the skinny kernels, whatever their rows and columns.
   No float sum splits along k.

   A plan declines a call, and nx's expansion computes it, when a and b differ
   in dtype; the call is neither a float contraction (float32, float16 or
   bfloat16 operands, result and init, a float32 accumulator) nor an integer
   one (8- to 64-bit integers throughout); or k passes 2^31 - 1: a function of
   the dtypes and k alone. Every layout is read where it lies: in chain
   order, operands with no axis of unit stride, or with offsets past 2^31
   within a batch element, on the general dense instances. Batches and rows
   past 2^30, and columns past 2^26, run in pieces of those extents. A call
   whose axes do not group answers what Contract_view.v answers. *)

module K = Kernels
module V = Nx_kernel.Contract_view
module D = Nx_array.Dtype
module Run = Rig.Submission.Run
module Sub = Rig.Submission

(* Dtypes *)

let code (D.Any d) = D.code d
let bytes (D.Any d) = D.bits d / 8
let dtype_of x = D.Any (Nx_array.dtype x)
let f32 = D.Any D.Float32

let integer : D.any -> bool = function
  | Any Int8 | Any Uint8 | Any Int16 | Any Uint16 | Any Int32 | Any Uint32
  | Any Int64 | Any Uint64 ->
      true
  | Any _ -> false

(* The dense kernels' dtype of [d], if it is one. *)
let dense : D.any -> K.dtype option = function
  | Any Float32 -> Some F32
  | Any Float16 -> Some F16
  | Any Bfloat16 -> Some Bf16
  | Any _ -> None

let same (D.Any a) (D.Any b) = D.equal a b

(* Instances *)

let count = Array.length K.kernels

(* The index of [instance], at most once per kernel and plan. *)
let rec index_from k instance =
  if snd K.kernels.(k) = instance then k else index_from (k + 1) instance

let index instance = index_from 0 instance

(* The instances by dtype and layout, in [K.kernels]' indices, made once. *)
let dtypes = [| K.F32; F16; Bf16 |]
let layouts = [| K.nn; K.nt; K.tn; K.tt |]
let large = Array.map (fun d -> Array.map (fun o -> index (Large (d, o))) layouts) dtypes
let wide = Array.map (fun d -> Array.map (fun o -> index (Wide (d, o))) layouts) dtypes
let narrow = Array.map (fun d -> Array.map (fun o -> index (Narrow (d, o))) layouts) dtypes
let short = Array.map (fun d -> Array.map (fun o -> index (Short (d, o))) layouts) dtypes
let small = Array.map (fun d -> index (Small d)) dtypes
let general = Array.map (fun d -> index (General d)) dtypes

(* Each dtype's checked large instance, if it has one. *)
let checked =
  Array.map
    (fun d ->
      if Array.exists (fun (_, i) -> i = K.Checked d) K.kernels then
        Some (index (Checked d))
      else None)
    dtypes

let int8 = Array.map (fun o -> index (Int8 o)) layouts

(* Each dtype's skinny instances of [rows] rows a threadgroup: b read along
   n, then k. *)
let skinny rows =
  Array.map
    (fun d -> [| index (Skinny (d, false, rows)); index (Skinny (d, true, rows)) |])
    dtypes

let skinny1 = skinny 1
let skinny4 = skinny K.skinny_rows

let int_kernel = index Int
let dtype_index : K.dtype -> int = function F32 -> 0 | F16 -> 1 | Bf16 -> 2

(* Plans *)

let ceil_div a b = (a + b - 1) / b

(* The largest k the kernels count in 32 bits with room for a run of 256
   terms past it. *)
let max_k = (1 lsl 31) - 1

(* The extent of a piece of a batch or rows past it: a multiple of every
   tile's side, so that pieces of whole tiles stay whole, and small enough
   that an extent plus a tile counts in 32 bits. Columns run in pieces of
   [column_piece]: a grid counts its threads along x in 32 bits, and a
   skinny grid takes 32 threads a column. At 2^30 columns the integer
   kernel's 4 threads a column passed 2^32, and the piece's last columns
   went unwritten. *)
let piece = 1 lsl 30
let column_piece = 1 lsl 26

(* The largest offset the dense instances but the general ones count, in
   32 bits. *)
let max_offset = (1 lsl 31) - 1

(* The column of tiles that consecutive threadgroups take, as a power of two:
   4 tiles reading one tile of b share it in the GPU's cache. *)
let swizzle = 2

(* A float product of fewer large tiles than [small_tiles] runs on small
   ones, which give the GPU's cores 4 times as many threadgroups; one of at
   most [wide_rows] rows runs on wide ones, which waste fewer of the matrix
   units' rows. *)
let small_tiles = 128
let wide_rows = 16

(* A product of fewer wide tiles than [narrow_tiles] runs on narrow ones,
   twice as many threadgroups, or, of at most [K.short_m] rows, short ones:
   with as many threadgroups as wide tiles, the GPU's cores have too few to
   cover its memory's latency. Measured by 2880 by 5120, wide tiles take
   1.1-1.3 times narrow ones' time at 3 to 16 rows; short ones 0.46-0.68
   of it at 3 and 8 rows for the half types and 0.93-0.97 for float32, and
   up to 1.9 times at 16 rows; tiles of 16 columns 1.2-1.25 times at 48
   rows by 5120 by 2880. *)
let narrow_tiles = 256

(* Checked tiles are large ones that read past the matrix. *)
type size = Large | Small | Wide | Narrow | Short | Checked

let tile_rows = function
  | Large | Checked -> K.large
  | Small -> K.small
  | Wide | Narrow -> K.wide_m
  | Short -> K.short_m

let tile_cols = function
  | Large | Checked -> K.large
  | Small -> K.small
  | Wide -> K.wide_n
  | Narrow | Short -> K.narrow_n

(* An operand as the kernels read it: its buffer, its dtype, its first
   element, bytes into the buffer as the device addresses it, and its strides
   in elements: a (batch, m, k), b (batch, k, n), init (batch, m, n). *)
type operand = {
  buffer : Rig.Buffer.t;
  dtype : D.any;
  first : int;
  s : int array;
}

(* A call as the rules read it; y is C-contiguous. *)
type call = {
  a : operand;
  b : operand;
  y : Rig.Buffer.t;
  y_dtype : D.any;
  y_first : int;
  init : operand option;
  acc : D.any;
  order : Nx_kernel.order;
  batch : int;
  m : int;
  n : int;
  k : int;
}

(* How a kernel's threadgroups cover a piece's outputs: dense tiles of a
   size, the integer kernel's tiles, or skinny threadgroups of [rows] rows,
   reading b along k (t) or along n. *)
type grid = Tiles of size | Int_tiles | Skinny_t of int | Skinny_n of int

(* A call's launch: its kernel, the grid of each piece and the layout bits
   of a and b. *)
type launch = { call : call; kernel : int; grid : grid; layout : int }
type verdict = Nothing | Launch of launch

let declined = Error Nx_array.Declined

(* The tiles of [size] that cover a product's outputs. *)
let tiles c size =
  ceil_div c.m (tile_rows size) * ceil_div c.n (tile_cols size) * c.batch

(* The steps of k the large tile stages for a's dtype. *)
let large_k c = match c.a.dtype with Any Float16 -> K.bk_half | Any _ -> K.bk

(* Whether tiles of [size] leave at most an eighth of a product's rows
   empty. *)
let fills c size =
  let rows = tile_rows size in
  let covered = ceil_div c.m rows * rows in
  8 * (covered - c.m) <= covered

(* The tile a product's dtype and shape pick: wide, or narrow or short in
   few tiles, for few rows; large for products of whole large tiles, enough of
   them, and whole steps of k; small for the other products of whole tiles.
   Past whole tiles, the largest of the dtype's tiles that leaves at most an
   eighth of the rows empty: checked large ones, for the dtypes that have
   them, given enough of them; small ones; wide or narrow ones for the half
   types, small ones for float32. Every tile sums
   an output alike, so the choice moves no bit. Measured on the 1000 cube,
   checked large tiles run float32 and bfloat16 nt 0.92 of small ones and nn
   1.04 and 0.99, float16 nn 1.06; on 48 rows, wide tiles run the half types
   1.2 to 1.7 times faster than small ones, and float32 nt 1.3 times
   slower. *)
let few_rows c =
  if tiles c Wide >= narrow_tiles then Wide
  else if c.m <= K.short_m then Short
  else Narrow

let tile_of c d =
  if c.m <= wide_rows then few_rows c
  else
    let side = K.large in
    let whole = c.m mod side = 0 && c.n mod side = 0 in
    let large = tiles c Large in
    if whole && large >= small_tiles && c.k mod large_k c = 0 then Large
    else if whole then Small
    else if
      Option.is_some checked.(dtype_index d)
      && large >= small_tiles && fills c Checked
    then Checked
    else if same c.a.dtype f32 || fills c Small then Small
    else few_rows c

(* Whether an operand of rows × cols with strides [s1] and [s2] keeps
   |s1| · rows + |s2| · cols within [max_offset], the extents rounded up to
   whole tiles of 64, whose reads the dense instances count past them:
   then every offset they form fits in 32 bits. *)
let near s1 s2 rows cols =
  let whole e = ceil_div e K.large * K.large in
  (abs s1 * whole rows) + (abs s2 * whole cols) <= max_offset

(* Whether every operand of [c] is [near], y with the strides of a
   C-contiguous result: the dense instances' offsets fit in 32 bits. *)
let near_all c =
  let sa = c.a.s and sb = c.b.s in
  near sa.(1) sa.(2) c.m c.k && near sb.(1) sb.(2) c.k c.n && near c.n 1 c.m c.n
  && match c.init with None -> true | Some i -> near i.s.(1) i.s.(2) c.m c.n

(* Whether an operand's stored rows start on 16-byte boundaries: its first
   element, row stride and, past one batch element, batch stride. *)
let aligned vector_first row batch count =
  vector_first && row mod 16 = 0 && (count = 1 || batch mod 16 = 0)

let layout_index layout =
  (Bool.to_int (layout land K.a_t <> 0) * 2)
  + Bool.to_int (layout land K.b_t <> 0)

(* Products on the matrix units read each operand where it lies, as the
   layout bits say. Every output adds its terms in increasing k, 8 at a
   time, in every tile: the tile only shapes the grid. *)
let tiled c d ~layout =
  let size = tile_of c d in
  let o = layout_index layout and i = dtype_index d in
  let kernel =
    match size with
    | Large -> large.(i).(o)
    | Small -> small.(i)
    | Checked -> Option.get checked.(i)
    | Wide -> wide.(i).(o)
    | Narrow -> narrow.(i).(o)
    | Short -> short.(i).(o)
  in
  { call = c; kernel; grid = Tiles size; layout }

(* A float product in lane order: b read along n where its columns are
   adjacent, its rows not, and it has more columns than a t threadgroup
   computes; along k otherwise, at any strides; one row a threadgroup, or
   [K.skinny_rows], reading b once for them. Every form sums an output
   alike. *)
let skinny_of c d =
  let sb = c.b.s in
  let along_n = sb.(2) = 1 && sb.(1) <> 1 && c.n > K.skinny_t in
  let rows = if c.m = 1 then 1 else K.skinny_rows in
  let kernels = if rows = 1 then skinny1 else skinny4 in
  let kernel = kernels.(dtype_index d).(Bool.to_int (not along_n)) in
  let grid = if along_n then Skinny_n rows else Skinny_t rows in
  { call = c; kernel; grid; layout = 0 }

(* The rules of nx.metal's contraction, over [c] and whether a's and b's
   first elements lie on 16-byte boundaries. *)
let rules c ~a_vector ~b_vector =
  (* Without init, [i] is y, which the families constrain alike. *)
  let a = c.a.dtype and y = c.y_dtype in
  let i = match c.init with None -> y | Some i -> i.dtype in
  let ints = integer a && integer c.acc && integer y && integer i in
  let floats =
    same c.acc f32 && Option.is_some (dense a) && Option.is_some (dense y)
    && Option.is_some (dense i)
  in
  let sa = c.a.s and sb = c.b.s in
  if (not (same a c.b.dtype)) || not (ints || floats) then declined
  else if c.k > max_k then declined
  else if c.batch = 0 || c.m = 0 || c.n = 0 then Ok Nothing
  else begin
    (* The dense kernels read an operand along an axis of unit stride, t
       naming the operands they read along their other axis, in offsets of
       32 bits; the general ones, any operand at its strides in offsets of
       64 bits. Contract_view's strides along an axis of one element are its
       own, so whether a window runs on the general instances may differ
       from the whole: the bits do not. *)
    let a_t = sa.(2) <> 1 && c.k > 1 and b_t = sb.(2) <> 1 in
    let ordered =
      ((not a_t) || sa.(1) = 1)
      && ((not b_t) || sb.(1) = 1 || c.k <= 1)
      && near_all c
    in
    let layout = (if a_t then K.a_t else 0) lor if b_t then K.b_t else 0 in
    (* int8 into 32 bits runs on the matrix units, exactly, in whole tiles
       whose stored rows start on 16-byte boundaries, which its 16-byte loads
       need; every other integer contraction on the SIMD units. Both are
       exact: the choice moves no bit. *)
    let side = K.large in
    let i8 =
      ints
      && (match a with Any Int8 -> true | Any _ -> false)
      && (match c.acc with Any Int32 | Any Uint32 -> true | Any _ -> false)
      && ordered && c.m mod side = 0 && c.n mod side = 0
      && c.k mod K.bk_half = 0
      && (c.k = 0
         || aligned a_vector (if a_t then sa.(2) else sa.(1)) sa.(0) c.batch
            && aligned b_vector (if b_t then sb.(2) else sb.(1)) sb.(0) c.batch)
    in
    let int8 = int8.(layout_index layout) in
    if i8 then
      Ok (Launch { call = c; kernel = int8; grid = Tiles Large; layout })
    else if ints then
      Ok (Launch { call = c; kernel = int_kernel; grid = Int_tiles; layout })
    else
      match (dense a, c.order) with
      | Some d, Nx_kernel.Lane -> Ok (Launch (skinny_of c d))
      | Some d, Chain when ordered -> Ok (Launch (tiled c d ~layout))
      | Some d, Chain ->
          let kernel = general.(dtype_index d) in
          Ok (Launch { call = c; kernel; grid = Tiles Small; layout = 0 })
      | None, _ -> declined
  end

(* [x] as the operand [which] of [v], over the axes [r] and [c]. *)
let operand v which (r : V.axis) (c : V.axis) x =
  let dtype = dtype_of x in
  {
    buffer = Nx_array.buffer x;
    dtype;
    first = V.offset v which * bytes dtype;
    s = [| V.stride v which Batch; V.stride v which r; V.stride v which c |];
  }

let choose ~order ~batch ~contracting ~acc ~dst ~init a b =
  match V.v ~batch ~contracting ~dst ~init a b with
  | Error answer -> Error answer
  | Ok v ->
      let y_dtype = dtype_of dst in
      let c =
        {
          a = operand v A Row Contracted a;
          b = operand v B Contracted Column b;
          y = Nx_array.buffer dst;
          y_dtype;
          y_first = V.offset v Dst * bytes y_dtype;
          init = Option.map (operand v Init Row Column) init;
          acc;
          order;
          batch = V.extent v Batch;
          m = V.extent v Row;
          n = V.extent v Column;
          k = V.extent v Contracted;
        }
      in
      let vector o = Rig.Buffer.aligned o.buffer ~at:o.first 16 in
      rules c ~a_vector:(vector c.a) ~b_vector:(vector c.b)

(* Pieces *)

let pieces_of e = ceil_div e piece
let column_pieces e = ceil_div e column_piece

let pieces l =
  pieces_of l.call.batch * pieces_of l.call.m * column_pieces l.call.n

(* Sequences *)

(* The submission's key: its kernel, then whether it has an init, dense over
   [count] kernels. *)
let sequences = count * 2

let sequence l = l.kernel + (count * Bool.to_int (Option.is_some l.call.init))

(* The slots: [a] 0, [b] 1 and [y] 2, then [init] where the sequence has
   one. *)
let init_slot = 3

let buffers l =
  let c = l.call in
  match c.init with
  | None -> [| c.a.buffer; c.b.buffer; c.y |]
  | Some i -> [| c.a.buffer; c.b.buffer; c.y; i.buffer |]

let parts key image =
  if key < 0 || key >= sequences then
    invalid_arg "Plan.parts: a key past sequences";
  let kernel = key mod count and has_init = key / count = 1 in
  let module P = K.Contract_params in
  let read slot at = Sub.read slot ~at and write slot at = Sub.write slot ~at in
  let init = if has_init then [ read init_slot P.init ] else [] in
  let refs =
    Array.of_list ([ read 0 P.a; read 1 P.b ] @ init @ [ write 2 P.out ])
  in
  [| Sub.launch ~refs image (fst K.kernels.(kernel)) ~params:P.size |]

(* Writing a run *)

(* The swizzle and threadgroups of [grid] over a piece of [m] rows and [n]
   columns, functions of ints that allocate nothing, so that every piece of
   a call allocates alike. *)
let swizzle_of grid m =
  match grid with
  | Tiles size when ceil_div m (tile_rows size) >= 2 * (1 lsl swizzle) -> swizzle
  | Tiles _ | Int_tiles | Skinny_t _ | Skinny_n _ -> 0

let groups_x grid m n =
  match grid with
  | Tiles size -> ceil_div n (tile_cols size) lsl swizzle_of grid m
  | Int_tiles -> ceil_div n K.int_tile
  | Skinny_t _ -> ceil_div n K.skinny_t
  | Skinny_n _ -> ceil_div n K.skinny_n

let groups_y grid m =
  match grid with
  | Tiles size -> ceil_div (ceil_div m (tile_rows size)) (1 lsl swizzle_of grid m)
  | Int_tiles -> ceil_div m K.int_tile
  | Skinny_t rows | Skinny_n rows -> ceil_div m rows

(* The byte of [o]'s element (p, r, q) as the device addresses it. *)
let element o p r q = o.first + ((p * o.s.(0)) + (r * o.s.(1)) + (q * o.s.(2))) * bytes o.dtype

(* Stores every field of the struct, and test_metal_kernels.ml shows that
   kernels.ml's fields tile it: every parameter byte is stored before each
   submit, so bytes another submission left in the run never reach a
   kernel. Piece [p] covers the batch elements, rows and columns from b0,
   m0 and n0, in C order of the pieces along batch, rows and columns. *)
let write l ~piece:p run sub =
  let module P = K.Contract_params in
  let c = l.call in
  let pm = pieces_of c.m and pn = column_pieces c.n in
  let b0 = p / (pm * pn) * piece and m0 = p / pn mod pm * piece in
  let n0 = p mod pn * column_piece in
  let batch = Int.min piece (c.batch - b0) and m = Int.min piece (c.m - m0) in
  let n = Int.min column_piece (c.n - n0) in
  let threads = match l.grid with Int_tiles -> K.int_threads | _ -> K.threads in
  let sa = c.a.s and sb = c.b.s in
  let at = Rig.Submission.block sub 0 in
  Run.groups run at (groups_x l.grid m n) (groups_y l.grid m) batch;
  Run.threads run at threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.a (element c.a b0 m0 0);
  Run.int64 run at P.b (element c.b b0 0 n0);
  Run.int64 run at P.out
    (c.y_first + ((((b0 * c.m) + m0) * c.n) + n0) * bytes c.y_dtype);
  Run.int64 run at P.a_batch sa.(0);
  Run.int64 run at P.a_m sa.(1);
  Run.int64 run at P.a_k sa.(2);
  Run.int64 run at P.b_batch sb.(0);
  Run.int64 run at P.b_k sb.(1);
  Run.int64 run at P.b_n sb.(2);
  begin match c.init with
  | Some i ->
      Run.int64 run at P.init (element i b0 m0 n0);
      Run.int64 run at P.init_batch i.s.(0);
      Run.int64 run at P.init_m i.s.(1);
      Run.int64 run at P.init_n i.s.(2);
      Run.int32 run at P.init_dtype (code i.dtype)
  | None ->
      Run.int64 run at P.init 0;
      Run.int64 run at P.init_batch 0;
      Run.int64 run at P.init_m 0;
      Run.int64 run at P.init_n 0;
      Run.int32 run at P.init_dtype K.no_init
  end;
  Run.int64 run at P.out_batch (c.m * c.n);
  Run.int64 run at P.out_m c.n;
  Run.int32 run at P.batch batch;
  Run.int32 run at P.m m;
  Run.int32 run at P.n n;
  Run.int32 run at P.k c.k;
  Run.int32 run at P.out_dtype (code c.y_dtype);
  Run.int32 run at P.swizzle (swizzle_of l.grid m);
  Run.int32 run at P.dtype (code c.a.dtype);
  Run.int32 run at P.acc (code c.acc);
  Run.int32 run at P.order l.layout
