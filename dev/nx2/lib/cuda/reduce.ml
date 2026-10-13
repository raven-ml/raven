(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A reduction's axes split into kept and reduced ones, each list coalesced
   apart in its own order, so that the terms keep their C order:

     x [kept..., reduced...] --> reduce_rows or reduce_cols --> y, or the
     ranges' values --> reduce_tree --> y

   A scan's axis is its one reduced axis, the others kept:

     x --> scan_totals (where a slice has several chunks) --> scan_rescan --> y

   reduce_rows takes outputs whose terms step through x no more than the
   outputs do, with 64 terms or more; reduce_cols the others. Each unit folds
   an aligned range of blocks, sized so that a GPU of kimchi's size gets
   enough of them; the size never changes a bit. *)

module K = Kernels
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module Run = Rig.Submission.Run
module Sub = Rig.Submission

(* The dtype code of the program's one operand where its one output is that
   operand unchanged, else -1. *)
let operand (p : Nx_kernel.Prog.t) =
  if not (Nx_kernel.Prog.is_identity p) then -1
  else
    let (D.Any dt) = Iarray.get p.ins 0 in
    D.code dt

let index name =
  let rec go k = if fst K.kernels.(k) = name then k else go (k + 1) in
  go 0

let reduce_rows = index "reduce_rows"
let reduce_cols = index "reduce_cols"
let reduce_tree = index "reduce_tree"
let scan_totals = index "scan_totals"
let scan_rescan = index "scan_rescan"

(* A call: its operand [x] and result [y]; the kept and reduced axes, [nkept]
   and [nred] triples of extent, x's stride and y's stride; their outputs and
   terms; and the first elements, bytes into x's and y's buffers. *)
type call = {
  x : Rig.Buffer.t;
  y : Rig.Buffer.t;
  monoid : int;
  dtype : int;
  nkept : int;
  nred : int;
  kept : int array;
  red : int array;
  outputs : int;
  terms : int;
  x_first : int;
  y_first : int;
}

(* A reduction's first pass: [body] over the call's blocks of terms in
   [groups] of [span] blocks, [full] of them whole, of [threads] threads. *)
type pass = { body : int; groups : int; full : int; span : int; threads : int }

(* A reduction is followed by reduce_tree where its first pass has several
   groups; a scan follows scan_totals where a slice has several chunks. *)
type launch = Reduce of call * pass | Scan of call
type verdict = Declined | Nothing | Launch of launch

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

(* Splits x's axes into the kept and reduced lists [kept] and [red], over
   y's strides [ys]: their counts. *)
let split axes l ys kept red =
  let nkept = ref 0 and nred = ref 0 in
  for i = 0 to L.rank l - 1 do
    let e = L.dim l i and xs = L.stride l i in
    if reduced axes i 0 then nred := push red !nred e xs ys.(i)
    else nkept := push kept !nkept e xs ys.(i)
  done;
  (!nkept, !nred)

(* Plans *)

let ceil_div a b = (a + b - 1) / b
let rec pow2_above n p = if p > n then p else pow2_above n (2 * p)

(* Units the first pass aims for on kimchi's 100 SMs: reduce_rows' blocks of
   up to 256 threads, reduce_cols' threads. *)
let row_units = 800
let col_units = 32768

(* The most blocks a unit folds: the counter's depth. *)
let max_span = 1 lsl 14

(* The blocks of terms a call's first pass takes. *)
let blocks = function
  | Reduce (c, _) -> ceil_div c.terms K.reduce_block
  | Scan c -> ceil_div c.terms K.scan_chunk

let rows ~outputs ~blocks =
  let q = Int.min 16 (pow2_above (blocks - 1) 1) in
  let span = ref 1 in
  while
    !span < max_span && outputs * ceil_div blocks (q * 2 * !span) >= row_units
  do
    span := 2 * !span
  done;
  let span = !span in
  {
    body = reduce_rows;
    threads = 16 * q;
    span;
    groups = ceil_div blocks (q * span);
    full = blocks / (q * span);
  }

let cols ~outputs ~blocks =
  let span = ref 1 in
  while
    !span < max_span && outputs * ceil_div blocks (2 * !span) >= col_units
  do
    span := 2 * !span
  done;
  let span = !span in
  {
    body = reduce_cols;
    threads = 256;
    span;
    groups = Int.max 1 (ceil_div blocks span);
    full = blocks / span;
  }

(* Whether terms step through x no more than outputs do. *)
let along_terms c =
  c.nred > 0
  && (c.nkept = 0
     || Int.abs c.red.((3 * (c.nred - 1)) + 1)
        < Int.abs c.kept.((3 * (c.nkept - 1)) + 1))

(* The dtypes nx.cuda reduces: float32, float64, the 8- to 64-bit integers;
   bool for the extremes. *)
let reduces (type v s) (dt : (v, s) D.t) monoid =
  match dt with
  | D.Float32 | D.Float64 | D.Int8 | D.Uint8 | D.Int16 | D.Uint16 | D.Int32
  | D.Uint32 | D.Int64 | D.Uint64 ->
      true
  | D.Bool -> monoid = K.monoid_max || monoid = K.monoid_min
  | _ -> false

(* reduce.cu's code of [m], or -1 for a monoid it does not fold. *)
let code : Nx_kernel.monoid -> int = function
  | Sum -> K.monoid_sum
  | Prod -> K.monoid_prod
  | Max -> K.monoid_max
  | Min -> K.monoid_min
  | Logsumexp -> -1

(* y's C-contiguous strides over the result, by x's axes: 0 along a reduced
   axis but for a scan, whose y has x's shape. *)
let result_strides ~scan axes l =
  let ys = Array.make (L.rank l) 0 in
  let acc = ref 1 in
  for i = L.rank l - 1 downto 0 do
    if scan || not (reduced axes i 0) then begin
      ys.(i) <- !acc;
      acc := !acc * L.dim l i
    end
  done;
  ys

(* reduce.cu holds a rank of triples of each list: one per axis of x at
   most. *)
let () = assert (L.max_rank <= K.reduce_rank)

(* The call of [x] into [dst] along [axes], or [None] where the kernels do
   not compute it. *)
let call ~scan ~axes m p ~dst x =
  let (A.Any ya) = dst in
  let code = code m and dt = operand p in
  if code < 0 || dt < 0 then None
  else if D.code (A.dtype ya) <> dt || not (reduces (A.dtype x) code) then None
  else
    let l = A.layout x and y = A.layout ya in
    let rank = L.rank l in
    let ys = result_strides ~scan axes l in
    let kept = Array.make (3 * rank) 0 and red = Array.make (3 * rank) 0 in
    let nkept, nred = split axes l ys kept red in
    let outputs = extent kept nkept 0 1 and terms = extent red nred 0 1 in
    let width = D.bits (A.dtype x) / 8 in
    let nred =
      if not scan then nred
      else begin
        (* The scan's axis, though of one element, is its red[0]. *)
        let a = axes.(0) in
        red.(0) <- L.dim l a;
        red.(1) <- L.stride l a;
        red.(2) <- ys.(a);
        1
      end
    in
    Some
      {
        x = A.buffer x;
        y = A.buffer ya;
        monoid = code;
        dtype = dt;
        nkept;
        nred;
        kept;
        red;
        outputs;
        terms;
        x_first = L.offset l * width;
        y_first = L.offset y * width;
      }

let reduce ~axes m p ~dst x =
  match call ~scan:false ~axes m p ~dst x with
  | None -> Declined
  | Some c when c.outputs = 0 -> Nothing
  | Some c ->
      let blocks = ceil_div c.terms K.reduce_block in
      let outputs = c.outputs in
      let pass =
        if c.terms >= 64 && along_terms c then rows ~outputs ~blocks
        else cols ~outputs ~blocks
      in
      Launch (Reduce (c, pass))

let scan ~axis m p ~dst x =
  match call ~scan:true ~axes:[| axis |] m p ~dst x with
  | None -> Declined
  | Some c when c.outputs = 0 || c.terms = 0 -> Nothing
  | Some c -> Launch (Scan c)

(* Sequences *)

(* Whether a second pass follows the first. *)
let second l =
  match l with Reduce (_, p) -> p.groups > 1 | Scan _ -> blocks l > 1

(* Keys: reduce_rows, then with reduce_tree; reduce_cols, then with
   reduce_tree; scan_rescan, then after scan_totals. *)
let sequences = 6

let sequence l =
  let pair = Bool.to_int (second l) in
  match l with
  | Scan _ -> 4 + pair
  | Reduce (_, p) -> if p.body = reduce_rows then pair else 2 + pair

let workspace l =
  if not (second l) then 0
  else
    match l with
    | Scan c -> 16 * c.outputs * (blocks l - 1)
    | Reduce (c, p) -> 16 * c.outputs * p.groups

let buffers l ~workspace =
  let c = match l with Reduce (c, _) | Scan c -> c in
  if second l then [| c.x; c.y; workspace |] else [| c.x; c.y |]

(* The kernels of the sequence [key], in order, with their refs into the
   slots: x, read, y, then the workspace, read and written. *)
let launches key =
  let module P = K.Reduce_params in
  let x = Sub.read 0 ~at:P.x and y = Sub.write 1 ~at:P.y in
  let refs = [| x; y; Sub.write 2 ~at:P.partials |] in
  match key with
  | 0 -> [ (reduce_rows, [| x; y |]) ]
  | 1 -> [ (reduce_rows, refs); (reduce_tree, refs) ]
  | 2 -> [ (reduce_cols, [| x; y |]) ]
  | 3 -> [ (reduce_cols, refs); (reduce_tree, refs) ]
  | 4 -> [ (scan_rescan, [| x; y |]) ]
  | 5 -> [ (scan_totals, refs); (scan_rescan, refs) ]
  | _ -> invalid_arg "Reduce.parts: a key past sequences"

let parts key image =
  let part (kernel, refs) =
    let kernel = fst K.kernels.(kernel) in
    Sub.launch ~refs image kernel ~params:K.Reduce_params.size
  in
  Array.of_list (List.map part (launches key))

(* Writing a run *)

let write_dims run at field dims n =
  for i = 0 to (3 * n) - 1 do
    Run.int64 run at (field + (8 * i)) dims.(i)
  done

let write_params run at c ~blocks ~groups ~full ~span =
  let module P = K.Reduce_params in
  Run.shared run at 0;
  Run.int64 run at P.x c.x_first;
  Run.int64 run at P.y c.y_first;
  Run.int64 run at P.partials 0;
  Run.int64 run at P.outputs c.outputs;
  Run.int64 run at P.terms c.terms;
  Run.int64 run at P.blocks blocks;
  Run.int64 run at P.groups groups;
  Run.int64 run at P.full full;
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

let write l run sub =
  let block = Rig.Submission.block sub in
  let blocks = blocks l and second = second l in
  match l with
  | Scan c ->
      let last = if second then 1 else 0 in
      if second then begin
        let at = block 0 in
        grid run at (c.outputs * (blocks - 1)) 256;
        write_params run at c ~blocks ~groups:0 ~full:0 ~span:0
      end;
      let at = block last in
      grid run at (c.outputs * blocks) 256;
      write_params run at c ~blocks ~groups:0 ~full:0 ~span:0
  | Reduce (c, p) ->
      let at = block 0 in
      if p.body = reduce_rows then begin
        Run.groups run at (c.outputs * p.groups) 1 1;
        Run.threads run at p.threads 1 1
      end
      else grid run at (c.outputs * p.groups) 256;
      let { groups; full; span; _ } = p in
      write_params run at c ~blocks ~groups ~full ~span;
      if second then begin
        let at = block 1 in
        Run.groups run at c.outputs 1 1;
        Run.threads run at 256 1 1;
        (* Thread j of reduce_tree folds the values from j tree_span: 256
           threads cover the whole ranges. *)
        let tree_span = pow2_above (full / 256) 1 in
        write_params run at c ~blocks ~groups ~full ~span:tree_span
      end
