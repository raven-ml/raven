(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The OCaml binding for the default CPU backend.

   A thin veneer over the C engine: no per-op materialization, no re-validation
   of frontend guarantees, no broadcast copies. Zero-stride (broadcast) views go
   straight to C — the engine handles stride 0. The frontend (Make_frontend)
   pre-broadcasts, promotes dtypes, and validates parameters, so a compute op
   here only allocates its C-contiguous output and hands the operands to the
   engine funnel; movement ops are pure View metadata manipulation.

   Externals are grouped by family. Every op in Backend_intf.S is wired to its C
   kernel; none remain stubbed. *)

open Nx_core

type ('a, 'b) buffer = ('a, 'b) Nx_buffer.t
type context = unit

let create_context () = ()

(* ── The tensor handle ─────────────────────────────────────────────────────

   FIELD ORDER IS ABI: [t] is passed to C directly, no per-call FFI record. The
   engine reads an operand at fixed record slots (nx_c.h NX_C_FFI_ markers): slot
   0 buffer (the bigarray), 1 shape, 2 strides, 3 offset — strides and offset in
   ELEMENT units, exactly as View provides. C never touches slot 4 (dtype, which
   it derives from the bigarray kind) or slot 5 (context). Reordering these six
   fields silently misreads every operand; the layout is pinned by the
   ABI echo test in test/test_backend_c.ml, not by convention. This
   declaration order MUST match {buffer; shape; strides; offset; dtype;
   context}. *)
type ('a, 'b) t = {
  buffer : ('a, 'b) buffer;
  shape : int array;
  strides : int array;
  offset : int;
  dtype : ('a, 'b) Dtype.t;
  context : context;
}

(* ── Accessors ─────────────────────────────────────────────────────────────*)

let view (t : ('a, 'b) t) = View.create ~offset:t.offset ~strides:t.strides t.shape
let dtype (t : ('a, 'b) t) = t.dtype
let context (t : ('a, 'b) t) = t.context
let to_host (t : ('a, 'b) t) = t.buffer

(* ── Creation ──────────────────────────────────────────────────────────────*)

let create_tensor ctx dtype shape =
  let size = Array.fold_left ( * ) 1 shape in
  let buffer = Nx_buffer.create dtype size in
  { buffer; shape; strides = Shape.c_contiguous_strides shape; offset = 0; dtype; context = ctx }

let buffer ctx dtype shape = create_tensor ctx dtype shape

let full ctx dtype shape value =
  let t = create_tensor ctx dtype shape in
  Nx_buffer.fill t.buffer value;
  t

let from_host ctx buf =
  let dtype = Nx_buffer.kind buf in
  let n = Nx_buffer.length buf in
  { buffer = buf; shape = [| n |]; strides = [| 1 |]; offset = 0; dtype; context = ctx }

(* ── Movement (pure View metadata) ─────────────────────────────────────────

   Each op runs the View transformation and reads shape/strides/offset back into
   a fresh handle sharing the buffer. Broadcast (expand) yields zero strides that
   go straight to C. pad/cat allocate and copy — they are C ops (move family). *)

let of_view (t : ('a, 'b) t) v =
  { t with shape = View.shape v; strides = View.strides v; offset = View.offset v }

let expand t shape = of_view t (View.expand (view t) shape)
let reshape t shape = of_view t (View.reshape (view t) shape)
let permute t axes = of_view t (View.permute (view t) axes)
let shrink t bounds = of_view t (View.shrink (view t) bounds)
let flip t axes = of_view t (View.flip (view t) axes)

let is_c_contiguous (t : ('a, 'b) t) =
  View.is_c_contiguous (view t) && t.offset = 0

(* [(before, after); ...] -> flat [before0; after0; before1; after1; ...], the
   window ops' padding ABI (nx_c_move.c reads pad_before/after at 2*d / 2*d+1). *)
let flatten_pairs pairs =
  Array.init
    (2 * Array.length pairs)
    (fun i ->
      let before, after = pairs.(i / 2) in
      if i mod 2 = 0 then before else after)

(* map family (nx_c_map.c): elementwise ops allocate a C-contiguous output the
   same shape as the input (comparisons output bool; cast the target dtype) and
   hand the strided operands to the map funnel — broadcast inputs (zero strides)
   go straight through. fdiv/idiv are separate primitives — the frontend selects
   by dtype, the backend never inspects it. The funnel keys binary/unary ops on
   the output dtype, comparisons on the input (bool output), cast on (src, dst). *)
external caml_neg : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_neg"
external caml_recip : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_recip"
external caml_abs : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_abs"
external caml_sign : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sign"
external caml_sqrt : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sqrt"
external caml_exp : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_exp"
external caml_log : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_log"
external caml_sin : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sin"
external caml_cos : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cos"
external caml_tan : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_tan"
external caml_asin : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_asin"
external caml_acos : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_acos"
external caml_atan : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_atan"
external caml_sinh : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sinh"
external caml_cosh : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cosh"
external caml_tanh : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_tanh"
external caml_trunc : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_trunc"
external caml_ceil : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_ceil"
external caml_floor : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_floor"
external caml_round : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_round"
external caml_erf : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_erf"

external caml_add : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_add"
external caml_sub : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_sub"
external caml_mul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_mul"
external caml_idiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_idiv"
external caml_fdiv : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_fdiv"
external caml_mod : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_mod"
external caml_pow : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_pow"
external caml_atan2 : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_atan2"
external caml_max : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_max"
external caml_min : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_min"
external caml_xor : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_xor"
external caml_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_or"
external caml_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_and"

external caml_cmpeq :
  (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cmpeq"

external caml_cmpne :
  (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cmpne"

external caml_cmplt :
  (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cmplt"

external caml_cmple :
  (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_cmple"

external caml_where :
  ('a, 'b) t -> (bool, Dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_where"

external caml_cast : ('c, 'd) t -> ('a, 'b) t -> unit = "caml_nx_c_cast"

let unary caml_op x =
  let out = create_tensor x.context x.dtype x.shape in
  caml_op out x;
  out

let binary caml_op x y =
  let out = create_tensor x.context x.dtype x.shape in
  caml_op out x y;
  out

let comparison caml_op x y =
  let out = create_tensor x.context Dtype.Bool x.shape in
  caml_op out x y;
  out

let neg x = unary caml_neg x
let recip x = unary caml_recip x
let abs x = unary caml_abs x
let sign x = unary caml_sign x
let sqrt x = unary caml_sqrt x
let exp x = unary caml_exp x
let log x = unary caml_log x
let sin x = unary caml_sin x
let cos x = unary caml_cos x
let tan x = unary caml_tan x
let asin x = unary caml_asin x
let acos x = unary caml_acos x
let atan x = unary caml_atan x
let sinh x = unary caml_sinh x
let cosh x = unary caml_cosh x
let tanh x = unary caml_tanh x
let trunc x = unary caml_trunc x
let ceil x = unary caml_ceil x
let floor x = unary caml_floor x
let round x = unary caml_round x
let erf x = unary caml_erf x
let add x y = binary caml_add x y
let sub x y = binary caml_sub x y
let mul x y = binary caml_mul x y
let mod_ x y = binary caml_mod x y
let pow x y = binary caml_pow x y
let atan2 x y = binary caml_atan2 x y
let max x y = binary caml_max x y
let min x y = binary caml_min x y
let xor x y = binary caml_xor x y
let or_ x y = binary caml_or x y
let and_ x y = binary caml_and x y

let fdiv x y = binary caml_fdiv x y
let idiv x y = binary caml_idiv x y

let cmpeq x y = comparison caml_cmpeq x y
let cmpne x y = comparison caml_cmpne x y
let cmplt x y = comparison caml_cmplt x y
let cmple x y = comparison caml_cmple x y

let where cond if_true if_false =
  let out = create_tensor if_true.context if_true.dtype if_true.shape in
  caml_where out cond if_true if_false;
  out

let cast ~dtype x =
  let out = create_tensor x.context dtype x.shape in
  caml_cast out x;
  out

(* fold family (nx_c_fold.c): [reduce] preserves the input dtype and drops the
   reduced axes — keepdims is a frontend concern (it reinserts the size-1 axes),
   so the interface's [reduce] never carries it. argreduce writes int32; scan
   preserves shape. The binding allocates the squeezed output and passes sorted
   axes. `Max/`Min have no identity over an empty axis — the binding rejects that
   before C (the fold driver does not know the op's identity). *)
external caml_reduce_sum : ('a, 'b) t -> ('a, 'b) t -> int array -> unit
  = "caml_nx_c_reduce_sum"

external caml_reduce_prod : ('a, 'b) t -> ('a, 'b) t -> int array -> unit
  = "caml_nx_c_reduce_prod"

external caml_reduce_max : ('a, 'b) t -> ('a, 'b) t -> int array -> unit
  = "caml_nx_c_reduce_max"

external caml_reduce_min : ('a, 'b) t -> ('a, 'b) t -> int array -> unit
  = "caml_nx_c_reduce_min"

external caml_argmax : (int32, Dtype.int32_elt) t -> ('a, 'b) t -> int -> unit
  = "caml_nx_c_argmax"

external caml_argmin : (int32, Dtype.int32_elt) t -> ('a, 'b) t -> int -> unit
  = "caml_nx_c_argmin"

external caml_cumsum : ('a, 'b) t -> ('a, 'b) t -> int -> unit = "caml_nx_c_cumsum"
external caml_cumprod : ('a, 'b) t -> ('a, 'b) t -> int -> unit
  = "caml_nx_c_cumprod"
external caml_cummax : ('a, 'b) t -> ('a, 'b) t -> int -> unit = "caml_nx_c_cummax"
external caml_cummin : ('a, 'b) t -> ('a, 'b) t -> int -> unit = "caml_nx_c_cummin"

let sorted_axes axes =
  let a = Array.copy axes in
  Array.sort Stdlib.compare a;
  a

let reduce ~op ~axes x =
  let caml_op, extreme =
    match op with
    | `Sum -> (caml_reduce_sum, None)
    | `Prod -> (caml_reduce_prod, None)
    | `Max -> (caml_reduce_max, Some "reduce_max")
    | `Min -> (caml_reduce_min, Some "reduce_min")
  in
  let axes = sorted_axes axes in
  (match extreme with
  | Some name ->
      Array.iter
        (fun ax ->
          if x.shape.(ax) = 0 then
            invalid_arg
              (name ^ ": reduction over an empty axis has no identity"))
        axes
  | None -> ());
  let out =
    create_tensor x.context x.dtype
      (Shape.reduce_output_shape x.shape axes false)
  in
  caml_op out x axes;
  out

let argreduce op caml_op ~axis ~keepdims x =
  if x.shape.(axis) = 0 then
    invalid_arg (op ^ ": argument reduction over an empty axis");
  let out =
    create_tensor x.context Dtype.Int32
      (Shape.reduce_output_shape x.shape [| axis |] keepdims)
  in
  caml_op out x axis;
  out

let argmax ~axis ~keepdims x = argreduce "argmax" caml_argmax ~axis ~keepdims x
let argmin ~axis ~keepdims x = argreduce "argmin" caml_argmin ~axis ~keepdims x

let associative_scan ~axis ~op x =
  let caml_op =
    match op with
    | `Sum -> caml_cumsum
    | `Prod -> caml_cumprod
    | `Max -> caml_cummax
    | `Min -> caml_cummin
  in
  let out = create_tensor x.context x.dtype x.shape in
  caml_op out x axis;
  out

(* sort family (nx_c_sort.c): both allocate a fresh contiguous output the same
   shape as the input (argsort's is int32) and hand the strided input to C. *)
external caml_sort : ('a, 'b) t -> ('a, 'b) t -> int -> bool -> unit
  = "caml_nx_c_sort"

external caml_argsort :
  (int32, Dtype.int32_elt) t -> ('a, 'b) t -> int -> bool -> unit
  = "caml_nx_c_argsort"

let sort ~axis ~descending x =
  let out = create_tensor x.context x.dtype x.shape in
  caml_sort out x axis descending;
  out

let argsort ~axis ~descending x =
  let out = create_tensor x.context Dtype.Int32 x.shape in
  caml_argsort out x axis descending;
  out

(* move family (nx_c_move.c): copy runs the strided copy kernel, so it serves
   copy (always a fresh buffer), contiguous's materialize path, and assign
   (writing the source through the destination's strides). pad/cat/gather/scatter
   and the window ops allocate their C-contiguous output and hand C the strided
   operands; scatter seeds the output from the template before the scatter walk.

   [contiguous] returns an already-contiguous, offset-0 tensor unchanged — the
   interface fast path, and the read path Make_frontend hits for every contiguous
   result — else it materializes through copy. *)
external caml_copy : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_copy"

let copy x =
  let out = create_tensor x.context x.dtype x.shape in
  caml_copy out x;
  out

let contiguous x = if is_c_contiguous x then x else copy x
let assign dst src = caml_copy dst src

external caml_pad :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> int array -> unit = "caml_nx_c_pad"

let pad x padding fill_value =
  let out_shape =
    Array.mapi
      (fun i d ->
        let before, after = padding.(i) in
        d + before + after)
      x.shape
  in
  let out = create_tensor x.context x.dtype out_shape in
  let fill = full x.context x.dtype [||] fill_value in
  caml_pad out x fill (Array.map fst padding);
  out

(* C reads the members as an array (Wosize_val/Field), so pass one, not a list. *)
external caml_cat : ('a, 'b) t -> ('a, 'b) t array -> int -> unit
  = "caml_nx_c_cat"

let cat tensors ~axis =
  match tensors with
  | [] -> invalid_arg "cat: empty tensor list"
  | first :: _ ->
      let ndim = Array.length first.shape in
      let axis = if axis < 0 then axis + ndim else axis in
      let total =
        List.fold_left (fun acc t -> acc + t.shape.(axis)) 0 tensors
      in
      let out_shape =
        Array.mapi (fun i d -> if i = axis then total else d) first.shape
      in
      let out = create_tensor first.context first.dtype out_shape in
      caml_cat out (Array.of_list tensors) axis;
      out

external caml_gather :
  ('a, 'b) t -> ('a, 'b) t -> (int32, Dtype.int32_elt) t -> int -> unit
  = "caml_nx_c_gather"

let gather data indices ~axis =
  let out = create_tensor data.context data.dtype indices.shape in
  caml_gather out data indices axis;
  out

external caml_scatter :
  ('a, 'b) t ->
  (int32, Dtype.int32_elt) t ->
  ('a, 'b) t ->
  int ->
  int ->
  unit = "caml_nx_c_scatter"

let scatter ~mode ~unique_indices:_ template ~indices ~updates ~axis =
  let out = copy template in
  let mode_int = match mode with `Set -> 0 | `Add -> 1 in
  caml_scatter out indices updates axis mode_int;
  out

external caml_unfold :
  ('a, 'b) t ->
  ('a, 'b) t ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_c_unfold_bc" "caml_nx_c_unfold"

let unfold x ~kernel_size ~stride ~dilation ~padding =
  let k = Array.length kernel_size in
  let leading_ndim = Array.length x.shape - k in
  let leading = Array.sub x.shape 0 leading_ndim in
  let spatial = Array.sub x.shape leading_ndim k in
  let out_spatial =
    Array.init k (fun i ->
        let before, after = padding.(i) in
        let padded = spatial.(i) + before + after in
        let extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
        ((padded - extent) / stride.(i)) + 1)
  in
  let kernel_prod = Array.fold_left ( * ) 1 kernel_size in
  let l = Array.fold_left ( * ) 1 out_spatial in
  let out_shape = Array.concat [ leading; [| kernel_prod; l |] ] in
  let padding_flat = flatten_pairs padding in
  let out = create_tensor x.context x.dtype out_shape in
  caml_unfold out x kernel_size stride dilation padding_flat;
  out

external caml_fold_window :
  ('a, 'b) t ->
  ('a, 'b) t ->
  int array ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_c_fold_bc" "caml_nx_c_fold"

let fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
  let leading_ndim = Array.length x.shape - 2 in
  let leading = Array.sub x.shape 0 leading_ndim in
  let out_shape = Array.concat [ leading; output_size ] in
  let padding_flat = flatten_pairs padding in
  let out = create_tensor x.context x.dtype out_shape in
  caml_fold_window out x output_size kernel_size stride dilation padding_flat;
  out

(* random family (nx_c_random.c): threefry hashes the counter, keeping its shape
   and int32 dtype. *)
external caml_threefry :
  (int32, Dtype.int32_elt) t ->
  (int32, Dtype.int32_elt) t ->
  (int32, Dtype.int32_elt) t ->
  unit = "caml_nx_c_threefry"

let threefry key counter =
  let out = create_tensor counter.context Dtype.Int32 counter.shape in
  caml_threefry out key counter;
  out

(* matmul (nx_c_matmul.c): allocate the contiguous batched output; the GEMM reads
   both operands at arbitrary strides (offset, batch, row, col), so no
   materialization — a transposed input is just distinct row/col strides. *)
external caml_matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_matmul"

let matmul x y =
  let xs = x.shape and ys = y.shape in
  let xnd = Array.length xs and ynd = Array.length ys in
  let m = xs.(xnd - 2) and n = ys.(ynd - 1) in
  let max_nd = Int.max xnd ynd in
  let batch_nd = max_nd - 2 in
  let batch =
    Array.init batch_nd (fun i ->
        let ai = i - (max_nd - xnd) and bi = i - (max_nd - ynd) in
        let sa = if ai >= 0 then xs.(ai) else 1 in
        let sb = if bi >= 0 then ys.(bi) else 1 in
        Int.max sa sb)
  in
  let out = create_tensor x.context x.dtype (Array.append batch [| m; n |]) in
  caml_matmul out x y;
  out

(* fft (nx_c_fft.c): the backend transforms are UNNORMALIZED (the frontend applies
   1/n). fft/ifft preserve the complex shape; rfft halves the last transformed
   axis to n/2+1; irfft restores it to `s` (or the inferred 2*(half-1)). The
   binding owns the output shape/dtype; C reads only the last `s` entry. Axes are
   frontend-guaranteed non-negative and in range. *)
external caml_fft : (Complex.t, 'b) t -> (Complex.t, 'b) t -> int array -> unit
  = "caml_nx_c_fft"

external caml_ifft : (Complex.t, 'b) t -> (Complex.t, 'b) t -> int array -> unit
  = "caml_nx_c_ifft"

external caml_rfft : (Complex.t, 'b) t -> (float, 'a) t -> int array -> unit
  = "caml_nx_c_rfft"

external caml_irfft :
  (float, 'b) t -> (Complex.t, 'a) t -> int array -> int array -> unit
  = "caml_nx_c_irfft"

let fft x ~axes =
  let out = create_tensor x.context x.dtype x.shape in
  caml_fft out x axes;
  out

let ifft x ~axes =
  let out = create_tensor x.context x.dtype x.shape in
  caml_ifft out x axes;
  out

let rfft x ~dtype ~axes =
  let last = axes.(Array.length axes - 1) in
  let out_shape = Array.copy x.shape in
  out_shape.(last) <- (x.shape.(last) / 2) + 1;
  let out = create_tensor x.context dtype out_shape in
  caml_rfft out x axes;
  out

let irfft ?s x ~dtype ~axes =
  let last_idx = Array.length axes - 1 in
  let last = axes.(last_idx) in
  let size =
    match s with Some sizes -> sizes.(last_idx) | None -> (x.shape.(last) - 1) * 2
  in
  let out_shape = Array.copy x.shape in
  out_shape.(last) <- size;
  let out = create_tensor x.context dtype out_shape in
  caml_irfft out x axes (match s with Some sizes -> sizes | None -> [||]);
  out

(* linalg tier 1 (nx_c_linalg.c): cholesky, triangular_solve, qr. Each allocates
   its output(s) — cholesky/trsm mirror the input/rhs shape, qr the reduced or
   full factor shapes — and hands C the operands. triangular_solve packs its
   three booleans into one int (bit 0 upper, 1 transpose, 2 unit-diagonal) so the
   stub stays at four args. eig is the later tier (nx_c_eig.c, wired below); svd's
   own factorization (nx_c_linalg.c) is under rewrite but already wired here. *)
external caml_cholesky : ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_cholesky"

external caml_triangular_solve :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> int -> unit
  = "caml_nx_c_triangular_solve"

external caml_qr : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_qr"

(* eigh (tier 2): eigenvalues always float64, eigenvectors in the input dtype.
   The stub extracts the eigenvector slot only when vectors=true, so vectors=false
   passes the input there again — no dummy allocation. *)
external caml_eigh :
  (float, Dtype.float64_elt) t -> ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_eigh"

(* Numeric linalg failures cross the FFI as [Failure "<op>: <reason>"] from the C
   funnel (nx_c_raise -> caml_failwith); shape/dtype preconditions cross as
   [Invalid_argument] and are left to propagate (the interface keeps those as-is).
   Lift the three recognized numeric reasons to [Linalg_error] so callers can
   match on the failure kind — the established pattern from backend_c's
   [reraise_linalg]. The matched suffixes are the exact static status strings
   raised in nx_c_linalg.c / nx_c_eig.c (LA_ERR_NOT_PD, LA_ERR_SINGULAR,
   LA_ERR_NO_CONVERGE / EIG_ERR_NO_CONVERGE); they must stay in sync with them. *)
let reraise_linalg ~op f =
  try f ()
  with Failure msg as e ->
    let ends suffix = String.ends_with ~suffix msg in
    if ends "matrix is not positive definite" then
      raise (Backend_intf.Linalg_error { op; kind = `Not_positive_definite })
    else if ends "triangular matrix is singular" then
      raise (Backend_intf.Linalg_error { op; kind = `Singular })
    else if ends "eigenvalue iteration did not converge" then
      raise (Backend_intf.Linalg_error { op; kind = `No_convergence })
    else raise e

let cholesky ~upper x =
  let out = create_tensor x.context x.dtype x.shape in
  reraise_linalg ~op:"cholesky" (fun () -> caml_cholesky out x upper);
  out

let triangular_solve ~upper ~transpose ~unit_diag a b =
  let vector_rhs = Array.length b.shape = Array.length a.shape - 1 in
  let b_matrix =
    if vector_rhs then reshape b (Array.append b.shape [| 1 |]) else b
  in
  let out_matrix = create_tensor b.context b.dtype b_matrix.shape in
  let flags =
    (if upper then 1 else 0)
    lor (if transpose then 2 else 0)
    lor (if unit_diag then 4 else 0)
  in
  reraise_linalg ~op:"triangular_solve" (fun () ->
      caml_triangular_solve out_matrix a b_matrix flags);
  if vector_rhs then reshape out_matrix b.shape else out_matrix

let qr ~reduced x =
  let s = x.shape in
  let nd = Array.length s in
  let m = s.(nd - 2) and n = s.(nd - 1) in
  let k = Int.min m n in
  let q_shape = Array.copy s and r_shape = Array.copy s in
  if reduced then (
    q_shape.(nd - 1) <- k;
    r_shape.(nd - 2) <- k)
  else q_shape.(nd - 1) <- m;
  let q = create_tensor x.context x.dtype q_shape in
  let r = create_tensor x.context x.dtype r_shape in
  reraise_linalg ~op:"qr" (fun () -> caml_qr q r x reduced);
  (q, r)

(* eigvalsh/eigh both drive caml_nx_c_eigh; eigenvalues always float64,
   eigenvectors in the input dtype. eigvalsh takes the cheaper values-only path
   (vectors=false); the stub then ignores the eigenvector slot, so it reuses x
   there — no dummy allocation. *)
let eigh_values x =
  let s = x.shape in
  let nd = Array.length s in
  let n = s.(nd - 1) in
  (* eigenvalues drop the trailing matrix dim: batch... x n. *)
  create_tensor x.context Dtype.Float64
    (Array.append (Array.sub s 0 (nd - 2)) [| n |])

let eigvalsh x =
  let w = eigh_values x in
  reraise_linalg ~op:"eigvalsh" (fun () -> caml_eigh w x x false);
  w

let eigh x =
  let w = eigh_values x in
  let v = create_tensor x.context x.dtype x.shape in
  reraise_linalg ~op:"eigh" (fun () -> caml_eigh w v x true);
  (w, v)

(* svd (tier 3): S is always float64; U/Vᴴ are the input dtype. full_matrices is
   encoded in the U/Vᴴ shapes the binding allocates (no separate C argument):
   thin gives U m×k, Vᴴ k×n; full gives U m×m, Vᴴ n×n; k = min(m, n). *)
external caml_svd :
  ('a, 'b) t ->
  (float, Dtype.float64_elt) t ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  unit = "caml_nx_c_svd"

let svd ~full_matrices x =
  let sh = x.shape in
  let nd = Array.length sh in
  let m = sh.(nd - 2) and n = sh.(nd - 1) in
  let k = Int.min m n in
  let batch = Array.sub sh 0 (nd - 2) in
  let u_shape =
    Array.append batch (if full_matrices then [| m; m |] else [| m; k |])
  in
  let vt_shape =
    Array.append batch (if full_matrices then [| n; n |] else [| k; n |])
  in
  let u = create_tensor x.context x.dtype u_shape in
  let s = create_tensor x.context Dtype.Float64 (Array.append batch [| k |]) in
  let vt = create_tensor x.context x.dtype vt_shape in
  reraise_linalg ~op:"svd" (fun () -> caml_svd u s vt x);
  (u, s, vt)

(* eigvals/eig (tier 3, nx_c_eig.c): the general nonsymmetric eigensolver.
   Eigenvalues and eigenvectors are always complex128 regardless of input dtype;
   the eigenvector slot is extracted only when vectors=true, so the values-only
   eigvals passes vectors=false and reuses w in that slot — the stub never
   touches it. *)
external caml_eig :
  (Complex.t, Dtype.complex64_elt) t ->
  (Complex.t, Dtype.complex64_elt) t ->
  ('a, 'b) t ->
  bool ->
  unit = "caml_nx_c_eig"

let eig_values x =
  let sh = x.shape in
  let nd = Array.length sh in
  let n = sh.(nd - 1) in
  create_tensor x.context Dtype.Complex128
    (Array.append (Array.sub sh 0 (nd - 2)) [| n |])

let eigvals x =
  let w = eig_values x in
  reraise_linalg ~op:"eigvals" (fun () -> caml_eig w w x false);
  w

let eig x =
  let w = eig_values x in
  let v = create_tensor x.context Dtype.Complex128 x.shape in
  reraise_linalg ~op:"eig" (fun () -> caml_eig w v x true);
  (w, v)
