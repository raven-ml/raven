(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A reduction's axes split into kept and reduced ones, each list coalesced
   apart in its own order, so that the terms keep their C order:

     x [kept..., reduced...] --> fold_rows or fold_cols --> y, or the
     ranges' values --> fold_tree --> y

   A scan's axis is its one reduced axis, the others kept:

     x --> scan_totals (where a slice has several chunks) --> scan_rescan --> y

   fold_rows takes outputs whose terms step through x no more than the
   outputs do, with 64 terms or more; fold_cols the others. Each unit folds
   an aligned range of blocks, sized so that a GPU of kimchi's size gets
   enough of them; the size never changes a bit. *)

module K = Kernels
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Run = Rig.Submission.Run
module Sub = Rig.Submission

(* The dtype code of the program's one operand where its one output is that
   operand unchanged, else -1: nx_cuda_stubs.c. *)
external operand : Nx_kernel.Prog.t -> (int[@untagged])
  = "nx_cuda_operand_byte" "nx_cuda_operand"
[@@noalloc]

let rank = K.fold_rank
let index name =
  let rec go k = if fst K.kernels.(k) = name then k else go (k + 1) in
  go 0

let fold_rows = index "fold_rows"
let fold_cols = index "fold_cols"
let fold_tree = index "fold_tree"
let scan_totals = index "scan_totals"
let scan_rescan = index "scan_rescan"

type verdict = Declined | Refused of A.answer | Nothing | Launches

type t = {
  mutable scan : bool;
  mutable monoid : int;
  mutable dtype : int;
  mutable width : int;
  mutable outputs : int;
  mutable terms : int;
  mutable blocks : int;
  mutable groups : int;
  mutable full : int;
  mutable span : int;
  mutable tree_span : int;  (** fold_tree's values a thread. *)
  mutable nkept : int;
  mutable nred : int;
  kept : int array;  (** [rank] triples: extent, x's stride, y's stride. *)
  red : int array;
  ys : int array;  (** y's strides by axis, while choosing. *)
  axis : int array;  (** A scan's one axis, as the axes it reduces. *)
  mutable x_first : int;  (** Bytes into x's buffer. *)
  mutable y_first : int;
  mutable body : int;  (** The first pass: fold_rows or fold_cols. *)
  mutable threads : int;
  mutable second : bool;  (** fold_tree after the body, or scan_totals first. *)
}

let make () =
  {
    scan = false;
    monoid = 0;
    dtype = 0;
    width = 0;
    outputs = 0;
    terms = 0;
    blocks = 0;
    groups = 0;
    full = 0;
    span = 0;
    tree_span = 0;
    nkept = 0;
    nred = 0;
    kept = Array.make (3 * rank) 0;
    red = Array.make (3 * rank) 0;
    ys = Array.make rank 0;
    axis = Array.make 1 0;
    x_first = 0;
    y_first = 0;
    body = 0;
    threads = 0;
    second = false;
  }

(* Axes *)

(* Appends the axis (e, xs, ys) to the [n] triples of [dims], merged into the
   last where both arrays lay the two out as one run; an axis of one element
   is dropped. The new count. *)
let push dims n e xs ys =
  if e = 1 then n
  else if
    n > 0
    && dims.((3 * (n - 1)) + 1) = xs * e
    && dims.((3 * (n - 1)) + 2) = ys * e
  then begin
    dims.(3 * (n - 1)) <- dims.(3 * (n - 1)) * e;
    dims.((3 * (n - 1)) + 1) <- xs;
    dims.((3 * (n - 1)) + 2) <- ys;
    n
  end
  else begin
    dims.(3 * n) <- e;
    dims.((3 * n) + 1) <- xs;
    dims.((3 * n) + 2) <- ys;
    n + 1
  end

let rec extent dims n i acc = if i = n then acc else extent dims n (i + 1) (acc * dims.(3 * i))

(* Whether axis [i] is among [axes] from [j], increasing. *)
let rec reduced axes i j =
  j < Array.length axes
  && (axes.(j) = i || (axes.(j) < i && reduced axes i (j + 1)))

(* Splits x's axes into kept and reduced lists; y's strides are [c.ys]. *)
let split c axes l =
  c.nkept <- 0;
  c.nred <- 0;
  for i = 0 to L.rank l - 1 do
    let e = L.dim l i and xs = L.stride l i and ys = c.ys.(i) in
    if reduced axes i 0 then c.nred <- push c.red c.nred e xs ys
    else c.nkept <- push c.kept c.nkept e xs ys
  done

(* Whether [axes] from [j] are strictly increasing axes of a rank [r]. *)
let rec axes_fit axes r j =
  j = Array.length axes
  || axes.(j) >= 0
     && axes.(j) < r
     && (j = 0 || axes.(j - 1) < axes.(j))
     && axes_fit axes r (j + 1)

(* Plans *)

let ceil_div a b = (a + b - 1) / b
let rec pow2_above n p = if p > n then p else pow2_above n (2 * p)

(* Units the first pass aims for on kimchi's 100 SMs: fold_rows' blocks of
   up to 256 threads, fold_cols' threads. *)
let row_units = 800
let col_units = 32768

(* The most blocks a unit folds: the counter's depth. *)
let max_span = 1 lsl 14

let plan_rows c =
  let q = Int.min 16 (pow2_above (c.blocks - 1) 1) in
  let span = ref 1 in
  while
    !span < max_span && c.outputs * ceil_div c.blocks (q * 2 * !span) >= row_units
  do
    span := 2 * !span
  done;
  c.body <- fold_rows;
  c.threads <- 16 * q;
  c.span <- !span;
  c.groups <- ceil_div c.blocks (q * !span);
  c.full <- c.blocks / (q * !span)

let plan_cols c =
  let span = ref 1 in
  while
    !span < max_span && c.outputs * ceil_div c.blocks (2 * !span) >= col_units
  do
    span := 2 * !span
  done;
  c.body <- fold_cols;
  c.threads <- 256;
  c.span <- !span;
  c.groups <- Int.max 1 (ceil_div c.blocks !span);
  c.full <- c.blocks / !span

(* Whether terms step through x no more than outputs do. *)
let along_terms c =
  c.nred > 0
  && (c.nkept = 0
     || Int.abs c.red.((3 * (c.nred - 1)) + 1)
        < Int.abs c.kept.((3 * (c.nkept - 1)) + 1))

let plan_reduce c =
  c.blocks <- ceil_div c.terms K.fold_block;
  if c.terms >= 64 && along_terms c then plan_rows c else plan_cols c;
  c.second <- c.groups > 1;
  (* Thread j of fold_tree folds the values from j tree_span: 256 threads
     cover the whole ranges. *)
  c.tree_span <- pow2_above (c.full / 256) 1

let plan_scan c =
  c.blocks <- ceil_div c.terms K.scan_chunk;
  c.body <- scan_rescan;
  c.threads <- 256;
  c.second <- c.blocks > 1

(* The dtypes nx.cuda folds: float32, float64, the 8- to 64-bit integers;
   bool for the extremes. *)
let folds (type v s) (dt : (v, s) D.t) monoid =
  match dt with
  | D.Float32 | D.Float64 | D.Int8 | D.Uint8 | D.Int16 | D.Uint16 | D.Int32
  | D.Uint32 | D.Int64 | D.Uint64 ->
      true
  | D.Bool -> monoid = K.monoid_max || monoid = K.monoid_min
  | _ -> false

(* fold.cu's code of [m], or -1 for a monoid it does not fold. *)
let code : Nx_kernel.monoid -> int = function
  | Sum -> K.monoid_sum
  | Prod -> K.monoid_prod
  | Max -> K.monoid_max
  | Min -> K.monoid_min
  | Logsumexp -> -1

(* y's C-contiguous strides over the result: x's shape without [axes], or
   with them for a scan, into [c.ys] by x's axes; false if y's shape is
   another. *)
let result_strides c axes l (y : L.t) =
  let rec go i yi acc =
    if i < 0 then yi < 0
    else if (not c.scan) && reduced axes i 0 then begin
      c.ys.(i) <- 0;
      go (i - 1) yi acc
    end
    else if yi < 0 || L.dim y yi <> L.dim l i then false
    else begin
      c.ys.(i) <- acc;
      go (i - 1) (yi - 1) (acc * L.dim l i)
    end
  in
  go (L.rank l - 1) (L.rank y - 1) 1

(* Whether a reduction by [m] along [axes] of x's shape has a result: one
   with outputs and no term has none for an extreme. *)
let has_result m ~axes l =
  match Nx_kernel.Shape.reduce ~axes [| Nx_kernel.Monoid m |] (L.shape l) with
  | Ok _ -> true
  | Error _ -> false

let choose c ~axes m p ~dst x =
  let (A.Any ya) = dst in
  let code = code m and dt = operand p in
  if code < 0 || dt < 0 then Declined
  else if D.code (A.dtype x) <> dt then Refused A.Wrong_dtype
  else if D.code (A.dtype ya) <> dt || not (folds (A.dtype x) code) then
    Declined
  else
    let l = A.layout x and y = A.layout ya in
    if (not (axes_fit axes (L.rank l) 0)) || not (result_strides c axes l y)
    then Refused A.Shape_mismatch
    else begin
      c.monoid <- code;
      c.dtype <- dt;
      c.width <- D.bits (A.dtype x) / 8;
      split c axes l;
      c.outputs <- extent c.kept c.nkept 0 1;
      c.terms <- extent c.red c.nred 0 1;
      c.x_first <- L.offset l * c.width;
      c.y_first <- L.offset y * c.width;
      if
        (not c.scan) && c.terms = 0 && c.outputs > 0
        && not (has_result m ~axes l)
      then Refused A.Shape_mismatch
      else if c.outputs = 0 || (c.scan && c.terms = 0) then Nothing
      else begin
        if c.scan then begin
          (* The scan's axis, though of one element, is its red[0]. *)
          let a = axes.(0) in
          c.nred <- 1;
          c.red.(0) <- L.dim l a;
          c.red.(1) <- L.stride l a;
          c.red.(2) <- c.ys.(a);
          plan_scan c
        end
        else plan_reduce c;
        Launches
      end
    end

let reduce c ~axes m p ~dst x =
  c.scan <- false;
  choose c ~axes m p ~dst x

let scan c ~axis m p ~dst x =
  c.scan <- true;
  c.axis.(0) <- axis;
  choose c ~axes:c.axis m p ~dst x

(* Sequences *)

(* Keys: fold_rows, then with fold_tree; fold_cols, then with fold_tree;
   scan_rescan, then after scan_totals. *)
let sequences = 6

let sequence c =
  let pair = Bool.to_int c.second in
  if c.scan then 4 + pair else if c.body = fold_rows then pair else 2 + pair

let workspace c =
  if not c.second then 0
  else if c.scan then 16 * c.outputs * (c.blocks - 1)
  else 16 * c.outputs * c.groups

let access c =
  let module B = Rig.Buffer in
  if c.second then [| Rig.Read; Rig.Read_write; Rig.Read_write |]
  else [| Rig.Read; Rig.Read_write |]

(* [c]'s launches in order: each kernel and its refs into the slots: x, y,
   then the workspace. *)
let launches c =
  let module P = K.Fold_params in
  let acc = access c in
  let ref at slot =
    match acc.(slot) with
    | Rig.Read -> Sub.read slot ~at
    | Read_write -> Sub.write slot ~at
  in
  let x = ref P.x 0 and y = ref P.y 1 in
  if not c.second then
    [ ((if c.scan then scan_rescan else c.body), [| x; y |]) ]
  else
    let refs = [| x; y; ref P.partials 2 |] in
    if c.scan then [ (scan_totals, refs); (scan_rescan, refs) ]
    else [ (c.body, refs); (fold_tree, refs) ]

let parts c image =
  let part (kernel, refs) =
    let kernel = fst K.kernels.(kernel) in
    Sub.launch ~refs image kernel ~params:K.Fold_params.size
  in
  Array.of_list (List.map part (launches c))

(* Writing a run *)

let write_dims run at field dims n =
  for i = 0 to (3 * n) - 1 do
    Run.int64 run at (field + (8 * i)) dims.(i)
  done

let write_params run at c ~span =
  let module P = K.Fold_params in
  Run.shared run at 0;
  Run.int64 run at P.x c.x_first;
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials 0;
  Run.int64 run at P.outputs c.outputs;
  Run.int64 run at P.terms c.terms;
  Run.int64 run at P.blocks c.blocks;
  Run.int64 run at P.groups c.groups;
  Run.int64 run at P.full c.full;
  Run.int64 run at P.span span;
  Run.int32 run at P.monoid c.monoid;
  Run.int32 run at P.dtype c.dtype;
  Run.int32 run at P.nkept c.nkept;
  Run.int32 run at P.nred c.nred;
  write_dims run at P.kept c.kept c.nkept;
  write_dims run at P.red c.red c.nred

let grid run at units threads =
  Run.groups run at (Int.max 1 (ceil_div units threads)) 1 1;
  Run.threads run at threads 1 1

let write run sub c =
  let block = Rig.Submission.block sub in
  if c.scan then begin
    let last = if c.second then 1 else 0 in
    if c.second then begin
      let at = block 0 in
      grid run at (c.outputs * (c.blocks - 1)) 256;
      write_params run at c ~span:0
    end;
    let at = block last in
    grid run at (c.outputs * c.blocks) 256;
    write_params run at c ~span:0
  end
  else begin
    let at = block 0 in
    if c.body = fold_rows then begin
      Run.groups run at (c.outputs * c.groups) 1 1;
      Run.threads run at c.threads 1 1
    end
    else grid run at (c.outputs * c.groups) 256;
    write_params run at c ~span:c.span;
    if c.second then begin
      let at = block 1 in
      Run.groups run at c.outputs 1 1;
      Run.threads run at 256 1 1;
      write_params run at c ~span:c.tree_span
    end
  end
