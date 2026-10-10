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

(* The descriptor as fold.cu's core reads it: nx_cuda_stubs.c. *)
external core : _ Nx_kernel.Spec.t -> (int[@untagged])
  = "nx_cuda_core_byte" "nx_cuda_core"
[@@noalloc]

external core_dtype : _ Nx_kernel.Spec.t -> (int[@untagged])
  = "nx_cuda_core_dtype_byte" "nx_cuda_core_dtype"
[@@noalloc]

external naxes : _ Nx_kernel.Spec.t -> (int[@untagged])
  = "nx_cuda_naxes_byte" "nx_cuda_naxes"
[@@noalloc]

external axis : _ Nx_kernel.Spec.t -> (int[@untagged]) -> (int[@untagged])
  = "nx_cuda_axis_byte" "nx_cuda_axis"
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

(* Whether axis [i] is among [s]'s [k] axes from [j], increasing. *)
let rec reduced s i j k =
  j < k && (axis s j = i || (axis s j < i && reduced s i (j + 1) k))

(* Splits x's axes into kept and reduced lists; y's strides are [c.ys]. *)
let split c s l =
  let r = L.rank l and k = naxes s in
  c.nkept <- 0;
  c.nred <- 0;
  for i = 0 to r - 1 do
    let e = L.dim l i and xs = L.stride l i and ys = c.ys.(i) in
    if reduced s i 0 k then c.nred <- push c.red c.nred e xs ys
    else c.nkept <- push c.kept c.nkept e xs ys
  done

(* Whether [s]'s axes are increasing below [r]: a scan's one. *)
let rec axes_fit s r j k = j = k || (axis s j < r && axes_fit s r (j + 1) k && (j = 0 || axis s (j - 1) < axis s j))

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

(* y's C-contiguous strides over the result: x's shape without the reduced
   axes, or with them for a scan, into [c.ys] by x's axes; false if y's
   shape is another. *)
let result_strides c s l (y : L.t) =
  let r = L.rank l and k = naxes s in
  let rec go i yi acc =
    if i < 0 then yi < 0
    else if (not c.scan) && reduced s i 0 k then begin
      c.ys.(i) <- 0;
      go (i - 1) yi acc
    end
    else if yi < 0 || L.dim y yi <> L.dim l i then false
    else begin
      c.ys.(i) <- acc;
      go (i - 1) (yi - 1) (acc * L.dim l i)
    end
  in
  go (r - 1) (L.rank y - 1) 1

let choose c family s ~dst x =
  let (A.Any xa) = x in
  let (A.Any ya) = dst in
  let m = core s in
  c.scan <- family = `Scan;
  if m < 0 then Declined
  else
    let dt = core_dtype s in
    if D.code (A.dtype xa) <> dt || D.code (A.dtype ya) <> dt then
      Refused A.Wrong_dtype
    else if not (folds (A.dtype xa) m) then Declined
    else
      let l = A.layout xa and y = A.layout ya in
      let k = naxes s in
      if (not (axes_fit s (L.rank l) 0 k)) || not (result_strides c s l y) then
        Refused A.Shape_mismatch
      else begin
        c.monoid <- m;
        c.dtype <- dt;
        c.width <- D.bits (A.dtype xa) / 8;
        split c s l;
        c.outputs <- extent c.kept c.nkept 0 1;
        c.terms <- extent c.red c.nred 0 1;
        c.x_first <- L.offset l * c.width;
        c.y_first <- L.offset y * c.width;
        if
          (not c.scan) && c.terms = 0 && c.outputs > 0
          && (m = K.monoid_max || m = K.monoid_min)
        then Refused A.Shape_mismatch
        else if c.outputs = 0 || (c.scan && c.terms = 0) then Nothing
        else begin
          if c.scan then begin
            (* The scan's axis, though of one element, is its red[0]. *)
            let a = axis s 0 in
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
  if c.second then [| B.Read; B.Read_write; B.Read_write |]
  else [| B.Read; B.Read_write |]

(* [c]'s launches in order: each kernel and its refs into the slots: x, y,
   then the workspace. *)
let launches c =
  let module P = K.Fold_params in
  let ref at slot = { Rig.Submission.at; slot } in
  let plain = [| ref P.x 0; ref P.y 1 |] in
  let with_ws = [| ref P.x 0; ref P.y 1; ref P.partials 2 |] in
  if c.scan then
    if c.second then [ (scan_totals, with_ws); (scan_rescan, with_ws) ]
    else [ (scan_rescan, plain) ]
  else if c.second then [ (c.body, with_ws); (fold_tree, with_ws) ]
  else [ (c.body, plain) ]

let parts c image ~queue =
  let part (kernel, refs) =
    let kernel = fst K.kernels.(kernel) in
    {
      Rig.Submission.queue;
      after = [||];
      work = Launch { image; kernel; params = K.Fold_params.size; refs };
    }
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
