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
module V = Nx_kernel.Spec.Contract_view
module S = Nx_kernel.Spec
module D = Nx_array.Dtype
module Run = Rig.Submission.Run

(* Dtypes *)

let code (D.Any d) = D.code d
let bytes (D.Any d) = D.bits d / 8
let dtype_of (Nx_array.Any x) = D.Any (Nx_array.dtype x)
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

(* The instances by dtype and order, in [K.kernels]' indices: tables made
   once, so that a plan allocates nothing. *)
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

type t = {
  mutable batch : int;
  mutable m : int;
  mutable n : int;
  mutable k : int;
  mutable acc : D.any;
  mutable init : bool;
  mutable a_dtype : D.any;
  mutable b_dtype : D.any;
  mutable y_dtype : D.any;
  mutable i_dtype : D.any;
  (* Each operand's first element, bytes into its slot, as the device
     addresses it, and its strides: a (batch, m, k), b (batch, k, n), init
     (batch, m, n); y is C-contiguous. *)
  mutable a_first : int;
  mutable b_first : int;
  mutable y_first : int;
  mutable i_first : int;
  mutable a_address : int;
  mutable b_address : int;
  sa : int array;
  sb : int array;
  si : int array;
  (* The plan: the contraction kernel's launch, the parts of a split sum (1
     for none), and the workspace bytes the parts' sums take. *)
  mutable kernel : int;
  mutable groups_x : int;
  mutable groups_y : int;
  mutable groups_z : int;
  mutable threads : int;
  mutable swizzle : int;
  mutable order : int;
  mutable parts : int;
  mutable used : int;
}

let make () =
  {
    batch = 0;
    m = 0;
    n = 0;
    k = 0;
    acc = f32;
    init = false;
    a_dtype = f32;
    b_dtype = f32;
    y_dtype = f32;
    i_dtype = f32;
    a_first = 0;
    b_first = 0;
    y_first = 0;
    i_first = 0;
    a_address = 0;
    b_address = 0;
    sa = Array.make 3 0;
    sb = Array.make 3 0;
    si = Array.make 3 0;
    kernel = 0;
    groups_x = 0;
    groups_y = 0;
    groups_z = 0;
    threads = 0;
    swizzle = 0;
    order = 0;
    parts = 1;
    used = 0;
  }

type verdict = Declined | Nothing | Launches

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
      match c.a_dtype with Any Float16 -> K.bk_half | Any _ -> K.bk)

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
    else if same c.a_dtype f32 || fills c Small then Small
    else Wide

(* Whether an operand's stored rows start on 16-byte boundaries: its address,
   row stride and, past one batch element, batch stride. *)
let aligned address row batch count =
  address mod 16 = 0 && row mod 16 = 0 && (count = 1 || batch mod 16 = 0)

let order_index c = (Bool.to_int (c.order land K.a_t <> 0) * 2) + Bool.to_int (c.order land K.b_t <> 0)

(* An integer contraction on the SIMD units: any strides, operands widened to
   64 bits. *)
let plan_integer c =
  c.kernel <- int_kernel;
  c.groups_x <- ceil_div c.n K.int_tile;
  c.groups_y <- ceil_div c.m K.int_tile;
  c.groups_z <- c.batch;
  c.threads <- K.int_threads

(* A float product of one row of a. *)
let plan_skinny c d =
  let b_t = c.order land K.b_t <> 0 in
  let per = if b_t then K.skinny_t else K.skinny_n in
  c.kernel <- skinny.(dtype_index d).(Bool.to_int b_t);
  c.groups_x <- ceil_div c.n per;
  c.groups_y <- 1;
  c.groups_z <- c.batch;
  c.threads <- K.threads

(* Products on the matrix units read each operand where it lies, as the
   order says. The tile the shape picks fixes the split, so every output's
   association is the shape's. A float product of few tiles splits along k
   into parts of equal length: a batch of float32 contractions, part q
   reading k from q · part_k, whose outputs contract_combine adds in order,
   then init. Integers sum in chunks within one threadgroup: they never
   split. *)

(* The grid of [size] tiles over the product, a column of [1 lsl swizzle]
   tiles to consecutive threadgroups. *)
let grid c size =
  let tiles_m = ceil_div c.m (tile_rows size) in
  let tiles_n = ceil_div c.n (tile_cols size) in
  c.swizzle <- (if tiles_m >= 2 * (1 lsl swizzle) then swizzle else 0);
  let column = 1 lsl c.swizzle in
  c.groups_x <- tiles_n * column;
  c.groups_y <- ceil_div tiles_m column;
  c.groups_z <- c.batch;
  c.threads <- K.threads

(* int8 into 32 bits: whole large tiles, which never split. *)
let plan_int8 c =
  grid c Large;
  c.kernel <- int8.(order_index c)

(* Floats of the dense dtype [d]; the workspace holds the parts' float32
   sums. *)
let plan_float c d =
  let size = tile_of c d in
  grid c size;
  let o = order_index c and i = dtype_index d in
  c.kernel <-
    (match size with
    | Large -> large.(i).(o)
    | Small -> small.(i)
    | Checked -> Option.get checked.(i)
    | Wide -> wide.(i).(o));
  c.parts <- split c (tiles c size);
  if c.parts > 1 then begin
    c.used <- ceil_div (c.parts * c.m * c.n * 4) 16 * 16;
    c.groups_z <- c.parts
  end

(* The plan of the call whose fields [choose] read: the rules of nx.metal's
   contraction, reading nothing but [c]. *)
let rules c =
  let a = c.a_dtype and y = c.y_dtype and i = c.i_dtype in
  let ints =
    integer a && integer c.acc && integer y && ((not c.init) || integer i)
  in
  let floats =
    same c.acc f32 && Option.is_some (dense a) && Option.is_some (dense y)
    && ((not c.init) || Option.is_some (dense i))
  in
  if (not (same a c.b_dtype)) || not (ints || floats) then Declined
  else if
    not (fits32 c.batch && fits32 c.m && fits32 c.n && fits32 c.k)
  then Declined
  else if
    (not (spans32 c.sa c.m c.k))
    || (not (spans32 c.sb c.k c.n))
    || not (spans32 c.si c.m c.n)
  then Declined
  else if c.batch = 0 || c.m = 0 || c.n = 0 then Nothing
  else begin
    c.used <- 0;
    c.parts <- 1;
    c.swizzle <- 0;
    (* The kernels read an operand with one axis of unit stride, or of one
       element; t names the operands whose other axis it is. *)
    let a_t = c.sa.(2) <> 1 && c.k > 1 and b_t = c.sb.(2) <> 1 && c.n > 1 in
    let ordered =
      (not (a_t && c.sa.(1) <> 1 && c.m > 1))
      && not (b_t && c.sb.(1) <> 1 && c.k > 1)
    in
    c.order <- (if a_t then K.a_t else 0) lor if b_t then K.b_t else 0;
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
         || aligned c.a_address
              (if a_t then c.sa.(2) else c.sa.(1))
              c.sa.(0) c.batch
            && aligned c.b_address
                 (if b_t then c.sb.(2) else c.sb.(1))
                 c.sb.(0) c.batch)
    in
    if ints && not i8 then begin
      plan_integer c;
      Launches
    end
    else if not ordered then Declined
    else
      match dense a with
      | _ when i8 ->
          plan_int8 c;
          Launches
      | Some d when c.m = 1 ->
          plan_skinny c d;
          Launches
      | Some d ->
          plan_float c d;
          Launches
      | None -> Declined
  end

let strides into v which (x : V.axis) (y : V.axis) =
  into.(0) <- V.stride v which Batch;
  into.(1) <- V.stride v which x;
  into.(2) <- V.stride v which y

let choose c v s ~dst ops =
  c.batch <- V.extent v Batch;
  c.m <- V.extent v Row;
  c.n <- V.extent v Column;
  c.k <- V.extent v Contracted;
  c.init <- S.init s;
  c.acc <- S.acc s;
  let (Nx_array.Any a) = ops.(0) in
  let (Nx_array.Any b) = ops.(1) in
  c.a_dtype <- dtype_of ops.(0);
  c.b_dtype <- dtype_of ops.(1);
  c.y_dtype <- dtype_of dst;
  c.a_first <- V.offset v A * bytes c.a_dtype;
  c.b_first <- V.offset v B * bytes c.b_dtype;
  c.y_first <- V.offset v Dst * bytes c.y_dtype;
  c.a_address <- Rig.Buffer.address (Nx_array.buffer a) + c.a_first;
  c.b_address <- Rig.Buffer.address (Nx_array.buffer b) + c.b_first;
  strides c.sa v A Row Contracted;
  strides c.sb v B Contracted Column;
  if c.init then begin
    c.i_dtype <- dtype_of ops.(2);
    c.i_first <- V.offset v Init * bytes c.i_dtype;
    strides c.si v Init Row Column
  end
  else begin
    c.i_dtype <- c.y_dtype;
    Array.fill c.si 0 3 0
  end;
  rules c

(* Sequences *)

(* The submission's key: its contraction kernel, then whether it splits and
   whether it has an init, dense over [count] kernels. *)
let flags = 4
let sequences = count * flags

let sequence c =
  c.kernel + (count * (Bool.to_int (c.parts > 1) lor (Bool.to_int c.init lsl 1)))

let reads c = if c.init then 3 else 2
let writes c = if c.parts > 1 then 2 else 1
let workspace c = if c.parts > 1 then c.used else 0

(* [c]'s launches in order: each kernel, its parameter bytes and its refs
   into the slots [reads] and [writes] count. *)
let launches c =
  let y = reads c in
  let ws = y + 1 in
  let ref at slot = { Rig.Submission.at; slot } in
  let init at = if c.init then [ ref at 2 ] else [] in
  let module P = K.Contract_params in
  let module Q = K.Combine_params in
  let split = c.parts > 1 in
  let contract =
    ( c.kernel,
      P.size,
      Array.of_list
        ([ ref P.a 0; ref P.b 1 ] @ init P.init
        @ [ ref P.out (if split then ws else y) ]) )
  in
  if not split then [ contract ]
  else
    [
      contract;
      ( combine_kernel,
        Q.size,
        Array.of_list ([ ref Q.out y; ref Q.parts ws ] @ init Q.init) );
    ]

let parts c image ~queue =
  let part (kernel, params, refs) =
    let kernel = fst K.kernels.(kernel) in
    {
      Rig.Submission.queue;
      after = [||];
      work = Launch { image; kernel; params; refs };
    }
  in
  Array.of_list (List.map part (launches c))

(* Writing a run *)

(* Stores every field of each launch's struct, and test_metal_kernels.ml
   shows that kernels.ml's fields tile each struct: every parameter byte is
   stored before each submit, so bytes another submission left in the run
   never reach a kernel. *)
let write run sub c =
  let module P = K.Contract_params in
  let split = c.parts > 1 in
  let part_k = c.k / c.parts in
  let at = Rig.Submission.block sub 0 in
  Run.groups run at c.groups_x c.groups_y c.groups_z;
  Run.threads run at c.threads 1 1;
  Run.shared run at 0;
  Run.int64 run at P.a c.a_first;
  Run.int64 run at P.b c.b_first;
  Run.int64 run at P.init (if c.init then c.i_first else 0);
  Run.int64 run at P.out (if split then 0 else c.y_first);
  Run.int64 run at P.a_batch (if split then part_k * c.sa.(2) else c.sa.(0));
  Run.int64 run at P.b_batch (if split then part_k * c.sb.(1) else c.sb.(0));
  Run.int64 run at P.init_batch c.si.(0);
  Run.int32 run at P.a_m c.sa.(1);
  Run.int32 run at P.a_k c.sa.(2);
  Run.int32 run at P.b_k c.sb.(1);
  Run.int32 run at P.b_n c.sb.(2);
  Run.int32 run at P.init_m c.si.(1);
  Run.int32 run at P.init_n c.si.(2);
  Run.int32 run at P.batch (if split then c.parts else c.batch);
  Run.int32 run at P.m c.m;
  Run.int32 run at P.n c.n;
  Run.int32 run at P.k part_k;
  Run.int32 run at P.init_dtype
    (if c.init && not split then code c.i_dtype else K.no_init);
  Run.int32 run at P.out_dtype (if split then code f32 else code c.y_dtype);
  Run.int32 run at P.swizzle c.swizzle;
  Run.int32 run at P.dtype (code c.a_dtype);
  Run.int32 run at P.acc (code c.acc);
  Run.int32 run at P.order c.order;
  if split then begin
    let module Q = K.Combine_params in
    let at = Rig.Submission.block sub 1 in
    Run.groups run at (ceil_div (c.m * c.n) combine_threads) 1 1;
    Run.threads run at combine_threads 1 1;
    Run.shared run at 0;
    Run.int64 run at Q.out c.y_first;
    Run.int64 run at Q.parts 0;
    Run.int64 run at Q.init (if c.init then c.i_first else 0);
    Run.int64 run at Q.init_batch c.si.(0);
    Run.int32 run at Q.init_m c.si.(1);
    Run.int32 run at Q.init_n c.si.(2);
    Run.int32 run at Q.batch 1;
    Run.int32 run at Q.m c.m;
    Run.int32 run at Q.n c.n;
    Run.int32 run at Q.split c.parts;
    Run.int32 run at Q.init_dtype (if c.init then code c.i_dtype else K.no_init);
    Run.int32 run at Q.out_dtype (code c.y_dtype)
  end
