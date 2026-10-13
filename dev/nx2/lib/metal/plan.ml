(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A call's launches: which kernels it runs, over which threadgroups, with
   which parameters. A pure function of the dtypes, the shapes and the
   strides: no association depends on the GPU's size or load.

   A plan declines a call, and nx's expansion computes it, when a and b differ
   in dtype; the call is neither a float contraction (float32, float16 or
   bfloat16 operands, result and init, a float32 accumulator) nor an integer
   one (8- to 64-bit integers throughout); an extent passes 2^32 - 1, or an
   operand's elements within a batch element lie 2^32 elements or more apart;
   or a float operand has neither axis of unit stride. A call whose axes do
   not group never reaches the plan: Contract_view.fill answers false
   first. *)

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

(* The instances by dtype and order, in [K.kernels]' indices, made once. *)
let dtypes = [| K.F32; F16; Bf16 |]
let orders = [| K.nn; K.nt; K.tn; K.tt |]
let large = Array.map (fun d -> Array.map (fun o -> index (Large (d, o))) orders) dtypes
let wide = Array.map (fun d -> Array.map (fun o -> index (Wide (d, o))) orders) dtypes
let small = Array.map (fun d -> index (Small d)) dtypes

(* Each dtype's checked large instance, if it has one. *)
let checked =
  Array.map
    (fun d ->
      if Array.exists (fun (_, i) -> i = K.Checked d) K.kernels then
        Some (index (Checked d))
      else None)
    dtypes

let int8 = Array.map (fun o -> index (Int8 o)) orders

let skinny =
  Array.map (fun d -> [| index (Skinny (d, false)); index (Skinny (d, true)) |]) dtypes

let combine_kernel = index Combine
let int_kernel = index Int
let dtype_index : K.dtype -> int = function F32 -> 0 | F16 -> 1 | Bf16 -> 2

(* Plans *)

let max_uint32 = 0xffff_ffff
let ceil_div a b = (a + b - 1) / b

(* The column of tiles that consecutive threadgroups take, as a power of two:
   4 tiles reading one tile of b share it in the GPU's cache. *)
let swizzle = 2

(* A float product of fewer large tiles than [small_tiles] runs on small
   ones, which give the GPU's cores 4 times as many threadgroups; one of at
   most [wide_rows] rows runs on wide ones, which waste fewer of the matrix
   units' rows. *)
let small_tiles = 128
let wide_rows = 16

(* A product of fewer tiles than [split_tiles] splits along k into parts
   whose sums contract_combine adds, so that the GPU's cores have work and
   each threadgroup a shorter chain of steps: up to [max_parts] parts of at
   least [min_part_k] terms each, since shorter parts cost more in their
   partial sums than they save. A function of the shape, as every
   association is. *)
let split_tiles = 256
let max_parts = 16
let min_part_k = 256

(* The combine's threads per threadgroup. *)
let combine_threads = 256

(* Checked tiles are large ones that read past the matrix. *)
type size = Large | Small | Wide | Checked

let tile_rows = function
  | Large | Checked -> K.large
  | Small -> K.small
  | Wide -> K.wide_m

let tile_cols = function
  | Large | Checked -> K.large
  | Small -> K.small
  | Wide -> K.wide_n


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
  batch : int;
  m : int;
  n : int;
  k : int;
}

(* A call's launches: the contraction kernel's grid and parameters, the parts
   of a split sum (1 for none), and the workspace bytes the parts' sums
   take. *)
type launch = {
  call : call;
  kernel : int;
  groups_x : int;
  groups_y : int;
  groups_z : int;
  threads : int;
  swizzle : int;
  order : int;
  parts : int;
  used : int;
}

type verdict = Declined | Nothing | Launch of launch

(* The strides of an absent init. *)
let no_strides = [| 0; 0; 0 |]

let fits32 x = x >= 0 && x <= max_uint32

(* Whether every element of a rows × cols matrix with strides [s] (over the
   call's three axes) lies within 2^32 elements of the batch's first: the
   kernels index within a batch in 32 bits. *)
let spans32 s rows cols =
  fits32 s.(1) && fits32 s.(2)
  && (rows = 0 || cols = 0 || fits32 (((rows - 1) * s.(1)) + ((cols - 1) * s.(2))))

(* How many parts a float product of [tiles] tiles splits into along k: only
   a batch of one splits. *)
let split c tiles =
  if c.batch <> 1 then 1
  else begin
    let parts = ref 1 in
    while
      tiles * !parts < split_tiles
      && 2 * !parts <= max_parts
      && c.k mod (2 * !parts) = 0
      && c.k / (2 * !parts) >= min_part_k
    do
      parts := 2 * !parts
    done;
    !parts
  end

(* The tiles of [size] that cover a product's outputs. *)
let tiles c size =
  ceil_div c.m (tile_rows size) * ceil_div c.n (tile_cols size) * c.batch

(* The steps of k the dense kernel stages for a dtype and tile. *)
let tile_k c size =
  match size with
  | Wide -> K.bk_wide
  | Large | Small | Checked -> (
      match c.a.dtype with Any Float16 -> K.bk_half | Any _ -> K.bk)

(* Whether tiles of [size] leave at most an eighth of a product's rows
   empty. *)
let fills c size =
  let rows = tile_rows size in
  let covered = ceil_div c.m rows * rows in
  8 * (covered - c.m) <= covered

(* The tile a product's dtype and shape pick: wide for few rows; large for
   products of whole large tiles, enough of them, and whole steps of k in each
   part; small for the other products of whole tiles. Past whole tiles, the
   largest of the dtype's tiles that leaves at most an eighth of the rows
   empty: checked large ones, for the dtypes that have them, given enough of
   them; small ones; wide ones for the half types, small ones for float32.
   Measured on the 1000 cube, checked large tiles run float32 and bfloat16 nt
   0.92 of small ones and nn 1.04 and 0.99, float16 nn 1.06; on 48 rows, wide
   tiles run the half types 1.2 to 1.7 times faster than small ones, and
   float32 nt 1.3 times slower. *)
let tile_of c d =
  if c.m <= wide_rows then Wide
  else
    let side = K.large in
    let whole = c.m mod side = 0 && c.n mod side = 0 in
    let large = tiles c Large in
    if whole && large >= small_tiles
       && c.k / split c large mod tile_k c Large = 0
    then Large
    else if whole then Small
    else if
      Option.is_some checked.(dtype_index d)
      && large >= small_tiles && fills c Checked
    then Checked
    else if same c.a.dtype f32 || fills c Small then Small
    else Wide

(* Whether an operand's stored rows start on 16-byte boundaries: its first
   element, row stride and, past one batch element, batch stride. *)
let aligned vector_first row batch count =
  vector_first && row mod 16 = 0 && (count = 1 || batch mod 16 = 0)

let order_index order =
  (Bool.to_int (order land K.a_t <> 0) * 2)
  + Bool.to_int (order land K.b_t <> 0)

(* An integer contraction on the SIMD units: any strides, operands widened to
   64 bits. *)
let plan_integer c ~order =
  {
    call = c;
    kernel = int_kernel;
    groups_x = ceil_div c.n K.int_tile;
    groups_y = ceil_div c.m K.int_tile;
    groups_z = c.batch;
    threads = K.int_threads;
    swizzle = 0;
    order;
    parts = 1;
    used = 0;
  }

(* A float product of one row of a. *)
let plan_skinny c d ~order =
  let b_t = order land K.b_t <> 0 in
  let per = if b_t then K.skinny_t else K.skinny_n in
  {
    call = c;
    kernel = skinny.(dtype_index d).(Bool.to_int b_t);
    groups_x = ceil_div c.n per;
    groups_y = 1;
    groups_z = c.batch;
    threads = K.threads;
    swizzle = 0;
    order;
    parts = 1;
    used = 0;
  }

(* Products on the matrix units read each operand where it lies, as the
   order says. The tile the shape picks fixes the split, so every output's
   association is the shape's. A float product of few tiles splits along k
   into parts of equal length: a batch of float32 contractions, part q
   reading k from q · part_k, whose outputs contract_combine adds in order,
   then init. Integers sum in chunks within one threadgroup: they never
   split. *)

(* [kernel] on the grid of [size] tiles over the product, a column of
   [1 lsl swizzle] tiles to consecutive threadgroups, in [parts] parts along
   k, whose float32 sums the workspace holds. *)
let gridded c size ~order ~kernel ~parts =
  let tiles_m = ceil_div c.m (tile_rows size) in
  let tiles_n = ceil_div c.n (tile_cols size) in
  let swizzle = if tiles_m >= 2 * (1 lsl swizzle) then swizzle else 0 in
  let column = 1 lsl swizzle in
  let split = parts > 1 in
  {
    call = c;
    kernel;
    groups_x = tiles_n * column;
    groups_y = ceil_div tiles_m column;
    groups_z = (if split then parts else c.batch);
    threads = K.threads;
    swizzle;
    order;
    parts;
    used = (if split then ceil_div (parts * c.m * c.n * 4) 16 * 16 else 0);
  }

(* int8 into 32 bits: whole large tiles, which never split. *)
let plan_int8 c ~order =
  gridded c Large ~order ~kernel:int8.(order_index order) ~parts:1

(* Floats of the dense dtype [d]. *)
let plan_float c d ~order =
  let size = tile_of c d in
  let o = order_index order and i = dtype_index d in
  let kernel =
    match size with
    | Large -> large.(i).(o)
    | Small -> small.(i)
    | Checked -> Option.get checked.(i)
    | Wide -> wide.(i).(o)
  in
  gridded c size ~order ~kernel ~parts:(split c (tiles c size))

(* The rules of nx.metal's contraction, over [c] and whether a's and b's
   first elements lie on 16-byte boundaries. *)
let rules c ~a_vector ~b_vector =
  (* Without init, [i] is y, which the families constrain alike. *)
  let a = c.a.dtype and y = c.y_dtype in
  let i, si =
    match c.init with None -> (y, no_strides) | Some i -> (i.dtype, i.s)
  in
  let ints = integer a && integer c.acc && integer y && integer i in
  let floats =
    same c.acc f32 && Option.is_some (dense a) && Option.is_some (dense y)
    && Option.is_some (dense i)
  in
  let sa = c.a.s and sb = c.b.s in
  if (not (same a c.b.dtype)) || not (ints || floats) then Declined
  else if not (fits32 c.batch && fits32 c.m && fits32 c.n && fits32 c.k) then
    Declined
  else if
    (not (spans32 sa c.m c.k))
    || (not (spans32 sb c.k c.n))
    || not (spans32 si c.m c.n)
  then Declined
  else if c.batch = 0 || c.m = 0 || c.n = 0 then Nothing
  else begin
    (* The kernels read an operand with one axis of unit stride, or of one
       element; t names the operands whose other axis it is. *)
    let a_t = sa.(2) <> 1 && c.k > 1 and b_t = sb.(2) <> 1 && c.n > 1 in
    let ordered =
      (not (a_t && sa.(1) <> 1 && c.m > 1))
      && not (b_t && sb.(1) <> 1 && c.k > 1)
    in
    let order = (if a_t then K.a_t else 0) lor if b_t then K.b_t else 0 in
    (* int8 into 32 bits runs on the matrix units, exactly, in whole tiles
       whose stored rows start on 16-byte boundaries, which its 16-byte loads
       need; every other integer contraction on the SIMD units. *)
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
    if ints && not i8 then Launch (plan_integer c ~order)
    else if not ordered then Declined
    else
      match dense a with
      | _ when i8 -> Launch (plan_int8 c ~order)
      | Some d when c.m = 1 -> Launch (plan_skinny c d ~order)
      | Some d -> Launch (plan_float c d ~order)
      | None -> Declined
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

let choose v ~acc ~dst ~init a b =
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
      batch = V.extent v Batch;
      m = V.extent v Row;
      n = V.extent v Column;
      k = V.extent v Contracted;
    }
  in
  let vector o = Rig.Buffer.aligned o.buffer ~at:o.first 16 in
  rules c ~a_vector:(vector c.a) ~b_vector:(vector c.b)

(* Sequences *)

(* The submission's key: its contraction kernel, then whether it splits and
   whether it has an init, dense over [count] kernels. *)
let flags = 4
let sequences = count * flags
let split_flag = 1
let init_flag = 2

let sequence l =
  let has_init = Option.is_some l.call.init in
  let split = if l.parts > 1 then split_flag else 0 in
  l.kernel + (count * (split lor if has_init then init_flag else 0))

(* The workspace is a slot of the sequence whenever it splits, whatever its
   bytes. *)
let workspace l = if l.parts > 1 then Int.max 1 l.used else 0

(* The slots: [a] 0, [b] 1 and [y] 2, then [init] and the workspace where the
   sequence has them. *)
let y_slot = 2
let init_slot = 3

let buffers l ~workspace =
  let c = l.call in
  match (c.init, l.parts > 1) with
  | None, false -> [| c.a.buffer; c.b.buffer; c.y |]
  | Some i, false -> [| c.a.buffer; c.b.buffer; c.y; i.buffer |]
  | None, true -> [| c.a.buffer; c.b.buffer; c.y; workspace |]
  | Some i, true -> [| c.a.buffer; c.b.buffer; c.y; i.buffer; workspace |]

(* The launches of the sequence [key] in order: each kernel, its parameter
   bytes and its refs into the slots [buffers] lists. *)
let launches key =
  if key < 0 || key >= sequences then
    invalid_arg "Plan.parts: a key past sequences";
  let kernel = key mod count and flags = key / count in
  let has_init = flags land init_flag <> 0 in
  let ws = init_slot + Bool.to_int has_init in
  let read slot at = Sub.read slot ~at and write slot at = Sub.write slot ~at in
  let init at = if has_init then [ read init_slot at ] else [] in
  let module P = K.Contract_params in
  let module Q = K.Combine_params in
  let split = flags land split_flag <> 0 in
  let contract =
    ( kernel,
      P.size,
      Array.of_list
        ([ read 0 P.a; read 1 P.b ] @ init P.init
        @ [ write (if split then ws else y_slot) P.out ]) )
  in
  if not split then [ contract ]
  else
    [
      contract;
      ( combine_kernel,
        Q.size,
        Array.of_list
          ([ write y_slot Q.out; write ws Q.parts ] @ init Q.init) );
    ]

let parts key image =
  let part (kernel, params, refs) =
    let kernel = fst K.kernels.(kernel) in
    Sub.launch ~refs image kernel ~params
  in
  Array.of_list (List.map part (launches key))

(* Writing a run *)

(* Stores every field of each launch's struct, and test_metal_kernels.ml
   shows that kernels.ml's fields tile each struct: every parameter byte is
   stored before each submit, so bytes another submission left in the run
   never reach a kernel. *)
let write l run sub =
  let module P = K.Contract_params in
  let c = l.call in
  let split = l.parts > 1 in
  let part_k = c.k / l.parts in
  let sa = c.a.s and sb = c.b.s in
  let i_first, si, i_code =
    match c.init with
    | Some i -> (i.first, i.s, code i.dtype)
    | None -> (0, no_strides, K.no_init)
  in
  let at = Rig.Submission.block sub 0 in
  Run.groups run at l.groups_x l.groups_y l.groups_z;
  Run.threads run at l.threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.a c.a.first;
  Run.int64 run at P.b c.b.first;
  Run.int64 run at P.init i_first;
  Run.int64 run at P.out (if split then 0 else c.y_first);
  Run.int64 run at P.a_batch (if split then part_k * sa.(2) else sa.(0));
  Run.int64 run at P.b_batch (if split then part_k * sb.(1) else sb.(0));
  Run.int64 run at P.init_batch si.(0);
  Run.int32 run at P.a_m sa.(1);
  Run.int32 run at P.a_k sa.(2);
  Run.int32 run at P.b_k sb.(1);
  Run.int32 run at P.b_n sb.(2);
  Run.int32 run at P.init_m si.(1);
  Run.int32 run at P.init_n si.(2);
  Run.int32 run at P.batch (if split then l.parts else c.batch);
  Run.int32 run at P.m c.m;
  Run.int32 run at P.n c.n;
  Run.int32 run at P.k part_k;
  Run.int32 run at P.init_dtype (if split then K.no_init else i_code);
  Run.int32 run at P.out_dtype (if split then code f32 else code c.y_dtype);
  Run.int32 run at P.swizzle l.swizzle;
  Run.int32 run at P.dtype (code c.a.dtype);
  Run.int32 run at P.acc (code c.acc);
  Run.int32 run at P.order l.order;
  if split then begin
    let module Q = K.Combine_params in
    let at = Rig.Submission.block sub 1 in
    Run.groups run at (ceil_div (c.m * c.n) combine_threads) 1 1;
    Run.threads run at combine_threads 1 1;
    Run.shared run at 0;
    Run.int64 run at Q.out c.y_first;
    Run.int64 run at Q.parts 0;
    Run.int64 run at Q.init i_first;
    Run.int64 run at Q.init_batch si.(0);
    Run.int32 run at Q.init_m si.(1);
    Run.int32 run at Q.init_n si.(2);
    Run.int32 run at Q.batch 1;
    Run.int32 run at Q.m c.m;
    Run.int32 run at Q.n c.n;
    Run.int32 run at Q.split l.parts;
    Run.int32 run at Q.init_dtype i_code;
    Run.int32 run at Q.out_dtype (code c.y_dtype)
  end
