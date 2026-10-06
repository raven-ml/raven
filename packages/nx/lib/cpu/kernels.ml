(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.cpu's kernels: the C engine over arrays in host memory.

   Each kernel hands its operands and the destination nx allocated to the
   engine's funnel, which reads the arrays at the slots nx_c.h names; zero
   strides (broadcasts) go straight through. nx gives operands the shapes and
   dtypes the operation takes, so a kernel checks only what the engine cannot
   compute. *)

open Nx_array

type ('a, 'b) t = ('a, 'b) Nx_array.t

external device_pool : unit -> nativeint = "caml_nx_device_pool"
external set_pool : nativeint -> unit = "caml_nx_c_set_pool"

(* The kernels run on the host's one thread pool, nx.device's. *)
let () = set_pool (device_pool ())
let name = "nx.cpu"
let runs_on = Nx_device.runs_on_host
let owns = runs_on
let shape (t : ('a, 'b) t) = View.shape t.view
let of_view (t : ('a, 'b) t) view = { t with view }

(* [(before, after); ...] -> flat [before0; after0; before1; after1; ...], the
   window ops' padding ABI (nx_c_move.c reads pad_before/after at 2*d /
   2*d+1). *)
let flatten_pairs pairs =
  Array.init
    (2 * Array.length pairs)
    (fun i ->
      let before, after = pairs.(i / 2) in
      if i mod 2 = 0 then before else after)

(* Map family (nx_c_map.c). The funnel keys binary and unary kernels on the
   output dtype, comparisons on the input, casts on both. *)

external caml_neg : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_neg"
external caml_recip : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_recip"
external caml_abs : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_abs"
external caml_sign : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sign"
external caml_sqrt : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_sqrt"
external caml_exp : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_exp"
external caml_log : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_log"
external caml_log1p : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_log1p"
external caml_expm1 : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_expm1"
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

external caml_or : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_or"

external caml_and : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_and"

external caml_cmpeq :
  (bool, Nx_dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_cmpeq"

external caml_cmpne :
  (bool, Nx_dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_cmpne"

external caml_cmplt :
  (bool, Nx_dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_cmplt"

external caml_cmple :
  (bool, Nx_dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_cmple"

external caml_fma : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_fma"

external caml_where :
  ('a, 'b) t -> (bool, Nx_dtype.bool_elt) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_where"

external caml_cast : ('c, 'd) t -> ('a, 'b) t -> unit = "caml_nx_c_cast"

(* 4-bit integers compute at the byte width: each operand is widened to int8 or
   uint8 by the cast, which sign-extends int4, the byte kernel runs, and a
   result of the operands' dtype is cast back, which keeps its low 4 bits: the
   result modulo 16. An axis an operand broadcasts stays broadcast. *)

type _ wide = Wide : (int, 'c) Nx_dtype.t -> int wide

let wide (type a b) (dt : (a, b) Nx_dtype.t) : a wide option =
  match dt with
  | Int4 -> Some (Wide Int8)
  | UInt4 -> Some (Wide UInt8)
  | _ -> None

let fresh dtype shape =
  let n = Array.fold_left ( * ) 1 shape in
  { dtype; view = View.create shape; buffer = Elements.create dtype n }

let widen w (x : (int, 'b) t) =
  let s = shape x in
  let held i n = if View.stride i x.view = 0 then min n 1 else n in
  let distinct = Array.mapi held s in
  let y = fresh w distinct in
  caml_cast y
    (of_view x (View.shrink x.view (Array.map (fun n -> (0, n)) distinct)));
  of_view y (View.expand y.view s)

let unary_at (k : Nx_backend.unary) x ~dst =
  match k with
  | Neg -> caml_neg dst x
  | Recip -> caml_recip dst x
  | Abs -> caml_abs dst x
  | Sqrt -> caml_sqrt dst x
  | Sign -> caml_sign dst x
  | Exp -> caml_exp dst x
  | Log -> caml_log dst x
  | Log1p -> caml_log1p dst x
  | Expm1 -> caml_expm1 dst x
  | Sin -> caml_sin dst x
  | Cos -> caml_cos dst x
  | Tan -> caml_tan dst x
  | Asin -> caml_asin dst x
  | Acos -> caml_acos dst x
  | Atan -> caml_atan dst x
  | Sinh -> caml_sinh dst x
  | Cosh -> caml_cosh dst x
  | Tanh -> caml_tanh dst x
  | Trunc -> caml_trunc dst x
  | Ceil -> caml_ceil dst x
  | Floor -> caml_floor dst x
  | Round -> caml_round dst x
  | Erf -> caml_erf dst x

let unary (type a b) k (x : (a, b) t) ~(dst : (a, b) t) =
  match wide x.dtype with
  | None -> unary_at k x ~dst
  | Some (Wide w) ->
      let y = fresh w (shape dst) in
      unary_at k (widen w x) ~dst:y;
      caml_cast dst y

let binary_at (k : Nx_backend.binary) x y ~dst =
  match k with
  | Add -> caml_add dst x y
  | Sub -> caml_sub dst x y
  | Mul -> caml_mul dst x y
  | Fdiv -> caml_fdiv dst x y
  | Idiv -> caml_idiv dst x y
  | Mod -> caml_mod dst x y
  | Pow -> caml_pow dst x y
  | Atan2 -> caml_atan2 dst x y
  | Maximum -> caml_max dst x y
  | Minimum -> caml_min dst x y
  | And -> caml_and dst x y
  | Or -> caml_or dst x y
  | Xor -> caml_xor dst x y

let binary (type a b) k (x : (a, b) t) y ~(dst : (a, b) t) =
  match wide x.dtype with
  | None -> binary_at k x y ~dst
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      binary_at k (widen w x) (widen w y) ~dst:z;
      caml_cast dst z

let compare_at (k : Nx_backend.compare) x y ~dst =
  match k with
  | Equal -> caml_cmpeq dst x y
  | Not_equal -> caml_cmpne dst x y
  | Less -> caml_cmplt dst x y
  | Less_equal -> caml_cmple dst x y

let compare (type a b) k (x : (a, b) t) y ~dst =
  match wide x.dtype with
  | None -> compare_at k x y ~dst
  | Some (Wide w) -> compare_at k (widen w x) (widen w y) ~dst

let fma (type a b) (a : (a, b) t) b c ~(dst : (a, b) t) =
  match wide a.dtype with
  | None -> caml_fma dst a b c
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      caml_fma z (widen w a) (widen w b) (widen w c);
      caml_cast dst z

let where (type a b) c (x : (a, b) t) y ~(dst : (a, b) t) =
  match wide x.dtype with
  | None -> caml_where dst c x y
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      caml_where z c (widen w x) (widen w y);
      caml_cast dst z

let cast x ~dst = caml_cast dst x

(* Fold family (nx_c_fold.c). The engine takes sorted axes. The kernels of the
   families that combine elements take the plan's thread count last; 0 leaves it
   to the engine's policy. *)

let policy = 0

external caml_reduce_sum : ('a, 'b) t -> ('a, 'b) t -> int array -> int -> unit
  = "caml_nx_c_reduce_sum"

external caml_reduce_prod : ('a, 'b) t -> ('a, 'b) t -> int array -> int -> unit
  = "caml_nx_c_reduce_prod"

external caml_reduce_max : ('a, 'b) t -> ('a, 'b) t -> int array -> int -> unit
  = "caml_nx_c_reduce_max"

external caml_reduce_min : ('a, 'b) t -> ('a, 'b) t -> int array -> int -> unit
  = "caml_nx_c_reduce_min"

external caml_argmax :
  (int64, Nx_dtype.int64_elt) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_argmax"

external caml_argmin :
  (int64, Nx_dtype.int64_elt) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_argmin"

external caml_cumsum : ('a, 'b) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_cumsum"

external caml_cumprod : ('a, 'b) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_cumprod"

external caml_cummax : ('a, 'b) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_cummax"

external caml_cummin : ('a, 'b) t -> ('a, 'b) t -> int -> int -> unit
  = "caml_nx_c_cummin"

let reduce_at (k : Nx_backend.reduce) axes x ~dst =
  match k with
  | Sum -> caml_reduce_sum dst x axes policy
  | Prod -> caml_reduce_prod dst x axes policy
  | Max -> caml_reduce_max dst x axes policy
  | Min -> caml_reduce_min dst x axes policy

(* A 4-bit reduction accumulates at the byte width and narrows once. *)
let reduce (type a b) k ~axes (x : (a, b) t) ~(dst : (a, b) t) =
  let axes = Array.copy axes in
  Array.sort Stdlib.compare axes;
  match wide x.dtype with
  | None -> reduce_at k axes x ~dst
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      reduce_at k axes (widen w x) ~dst:z;
      caml_cast dst z

let arg_reduce_at (k : Nx_backend.arg_reduce) axis x ~dst =
  match k with
  | Argmax -> caml_argmax dst x axis policy
  | Argmin -> caml_argmin dst x axis policy

let arg_reduce (type a b) k ~axis (x : (a, b) t) ~dst =
  match wide x.dtype with
  | None -> arg_reduce_at k axis x ~dst
  | Some (Wide w) -> arg_reduce_at k axis (widen w x) ~dst

let scan_at (k : Nx_backend.reduce) axis x ~dst =
  match k with
  | Sum -> caml_cumsum dst x axis policy
  | Prod -> caml_cumprod dst x axis policy
  | Max -> caml_cummax dst x axis policy
  | Min -> caml_cummin dst x axis policy

let scan (type a b) k ~axis (x : (a, b) t) ~(dst : (a, b) t) =
  match wide x.dtype with
  | None -> scan_at k axis x ~dst
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      scan_at k axis (widen w x) ~dst:z;
      caml_cast dst z

(* Sort family (nx_c_sort.c) *)

external caml_sort : ('a, 'b) t -> ('a, 'b) t -> int -> bool -> int -> unit
  = "caml_nx_c_sort"

external caml_argsort :
  (int64, Nx_dtype.int64_elt) t -> ('a, 'b) t -> int -> bool -> int -> unit
  = "caml_nx_c_argsort"

let sort ~descending ~axis x ~dst = caml_sort dst x axis descending policy
let argsort ~descending ~axis x ~dst = caml_argsort dst x axis descending policy

(* Group (nx_c_group.c) *)

external caml_group :
  (int64, Nx_dtype.int64_elt) t -> (int64, Nx_dtype.uint64_elt) t -> int -> unit
  = "caml_nx_c_group"

let group x ~dst = caml_group dst x policy

(* Move family (nx_c_move.c): the strided copy, and the kernels that write their
   operands into the destination. The pad value crosses to C as a one-element
   array. *)

external caml_copy : ('a, 'b) t -> ('a, 'b) t -> unit = "caml_nx_c_copy"

external caml_pad : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> int array -> unit
  = "caml_nx_c_pad"

external caml_cat : ('a, 'b) t -> ('a, 'b) t array -> int -> unit
  = "caml_nx_c_cat"

external caml_gather :
  ('a, 'b) t -> ('a, 'b) t -> (int64, Nx_dtype.int64_elt) t -> int -> unit
  = "caml_nx_c_gather"

external caml_scatter :
  ('a, 'b) t ->
  (int64, Nx_dtype.int64_elt) t ->
  ('a, 'b) t ->
  int ->
  int ->
  unit = "caml_nx_c_scatter"

let contiguous x ~dst = caml_copy dst x

let pad padding v (x : ('a, 'b) t) ~dst =
  let fill =
    {
      dtype = x.dtype;
      view = View.create [||];
      buffer = Elements.create x.dtype 1;
    }
  in
  Elements.fill x.dtype fill.buffer v;
  caml_pad dst x fill (Array.map fst padding)

(* C reads the members as an array (Wosize_val/Field). *)
let cat ~axis xs ~dst = caml_cat dst (Array.of_list xs) axis
let gather ~axis indices x ~dst = caml_gather dst x indices axis

(* The scatter walk writes into a copy of [x]. The last argument packs the mode
   in its two low bits and [unique] in the third. *)
let scatter_at ~mode ~unique ~axis ~indices ~updates x ~dst =
  caml_copy dst x;
  caml_scatter dst indices updates axis
    ((match mode with `Set -> 0 | `Add -> 1 | `Max -> 2 | `Min -> 3)
    lor if unique then 4 else 0)

(* A 4-bit [`Set] moves elements; the other modes combine them. *)
let scatter (type a b) ~mode ~unique ~axis ~indices ~(updates : (a, b) t)
    (x : (a, b) t) ~(dst : (a, b) t) =
  match (mode, wide x.dtype) with
  | `Set, _ | _, None -> scatter_at ~mode ~unique ~axis ~indices ~updates x ~dst
  | (`Add | `Max | `Min), Some (Wide w) ->
      let z = fresh w (shape dst) in
      scatter_at ~mode ~unique ~axis ~indices ~updates:(widen w updates)
        (widen w x) ~dst:z;
      caml_cast dst z

(* The window write is the strided copy: [x] copied, then [v] written through a
   shrunk view of the copy. The packed copy writes a window's whole words and
   merges its partial ones, keeping the elements around it. *)
let update (x : ('a, 'b) t) ~(starts : Nx_backend.index_array) v ~dst =
  caml_copy dst x;
  let positions =
    match Nx_device.Buffer.borrow Nx_device.host starts.buffer with
    | Ok b -> b
    | Error why -> invalid_arg ("Nx_cpu.update: " ^ why)
  in
  let start = Elements.get Nx_dtype.int64 positions in
  let offset = View.offset starts.view and stride = View.stride 0 starts.view in
  let bounds =
    Array.init
      (Array.length (shape x))
      (fun i ->
        let c = Int64.to_int (start (offset + (i * stride))) in
        (c, c + (shape v).(i)))
  in
  caml_copy (of_view dst (View.shrink dst.view bounds)) v

external caml_unfold :
  ('a, 'b) t ->
  ('a, 'b) t ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_c_unfold_bc" "caml_nx_c_unfold"

external caml_fold_window :
  ('a, 'b) t ->
  ('a, 'b) t ->
  int array ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_c_fold_bc" "caml_nx_c_fold"

let unfold ~kernel_size ~stride ~dilation ~padding x ~dst =
  caml_unfold dst x kernel_size stride dilation (flatten_pairs padding)

(* fold sums the windows that overlap. *)
let fold (type a b) ~output_size ~kernel_size ~stride ~dilation ~padding
    (x : (a, b) t) ~(dst : (a, b) t) =
  let padding = flatten_pairs padding in
  match wide x.dtype with
  | None ->
      caml_fold_window dst x output_size kernel_size stride dilation padding
  | Some (Wide w) ->
      let z = fresh w (shape dst) in
      caml_fold_window z (widen w x) output_size kernel_size stride dilation
        padding;
      caml_cast dst z

(* Random family (nx_c_random.c) *)

external caml_threefry :
  (int32, Nx_dtype.int32_elt) t ->
  (int32, Nx_dtype.int32_elt) t ->
  (int32, Nx_dtype.int32_elt) t ->
  unit = "caml_nx_c_threefry"

let threefry key counter ~dst = caml_threefry dst key counter

(* Matmul (nx_c_matmul.c): the GEMM reads both operands at any strides. An empty
   product has nothing to compute. *)

external caml_matmul : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> unit
  = "caml_nx_c_matmul"

let matmul (type a b) (x : (a, b) t) y ~(dst : (a, b) t) =
  if not (Array.exists (( = ) 0) (shape dst)) then
    match wide x.dtype with
    | None -> caml_matmul dst x y
    | Some (Wide w) ->
        let z = fresh w (shape dst) in
        caml_matmul z (widen w x) (widen w y);
        caml_cast dst z

(* FFT (nx_c_fft.c): unnormalized transforms. C reads the output size of the
   last transformed axis from [s] when there is one. *)

external caml_fft : (Complex.t, 'b) t -> (Complex.t, 'b) t -> int array -> unit
  = "caml_nx_c_fft"

external caml_ifft : (Complex.t, 'b) t -> (Complex.t, 'b) t -> int array -> unit
  = "caml_nx_c_ifft"

external caml_rfft : (Complex.t, 'b) t -> (float, 'a) t -> int array -> unit
  = "caml_nx_c_rfft"

external caml_irfft :
  (float, 'b) t -> (Complex.t, 'a) t -> int array -> int array -> unit
  = "caml_nx_c_irfft"

let fft ~inverse ~axes x ~dst =
  if inverse then caml_ifft dst x axes else caml_fft dst x axes

let rfft ~axes x ~dst = caml_rfft dst x axes

let irfft ~axes ~s x ~dst =
  caml_irfft dst x axes (match s with Some sizes -> sizes | None -> [||])

(* Linear algebra (nx_c_linalg.c, nx_c_eig.c). solve_triangular packs its three
   flags into one int (bit 0 upper, 1 transpose, 2 unit diagonal). The
   eigensolvers extract the vectors slot only when asked, so the values-only
   paths pass another array there. *)

external caml_cholesky : ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_cholesky"

external caml_solve_triangular :
  ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> int -> unit
  = "caml_nx_c_solve_triangular"

external caml_qr : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_qr"

external caml_eigh :
  (float, Nx_dtype.float64_elt) t -> ('a, 'b) t -> ('a, 'b) t -> bool -> unit
  = "caml_nx_c_eigh"

external caml_lu :
  ('a, 'b) t ->
  (int64, Nx_dtype.int64_elt) t ->
  (int64, Nx_dtype.int64_elt) t ->
  ('a, 'b) t ->
  unit = "caml_nx_c_lu"

external caml_svd :
  ('a, 'b) t ->
  (float, Nx_dtype.float64_elt) t ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  unit = "caml_nx_c_svd"

external caml_eig :
  (Complex.t, Nx_dtype.complex64_elt) t ->
  (Complex.t, Nx_dtype.complex64_elt) t ->
  ('a, 'b) t ->
  bool ->
  unit = "caml_nx_c_eig"

let cholesky ~upper x ~dst = caml_cholesky dst x upper

(* A vector right-hand side is solved as a one-column matrix. *)
let solve_triangular ~upper ~transpose ~unit_diag a b ~dst =
  let column (t : ('a, 'b) t) =
    if Array.length (shape t) = Array.length (shape a) - 1 then
      of_view t (View.reshape t.view (Array.append (shape t) [| 1 |]))
    else t
  in
  let flags =
    (if upper then 1 else 0)
    lor (if transpose then 2 else 0)
    lor if unit_diag then 4 else 0
  in
  caml_solve_triangular (column dst) a (column b) flags

let qr ~reduced x ~q ~r = caml_qr q r x reduced
let lu x ~lu ~pivots ~perm = caml_lu lu pivots perm x
let svd x ~u ~s ~vt = caml_svd u s vt x

let eig x ~values ~vectors =
  match vectors with
  | None -> caml_eig values values x false
  | Some v -> caml_eig values v x true

let eigh x ~values ~vectors =
  match vectors with
  | None -> caml_eigh values x x false
  | Some v -> caml_eigh values v x true
