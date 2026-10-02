(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array

(* Operations

   Every operation nx performs is a constructor of [t]: the computing ones,
   which a backend's kernels answer, and the movements, placing, reading and
   checking, which nx answers itself. A kind names the function among the
   operations of one constructor. *)

(* The two dtype conversions: [Cast] converts values, and [Bitcast] reads the
   elements' bytes in row-major order as elements of another dtype, consuming a
   last axis of [k] when it is [k] times wider and adding one when it is [k]
   times narrower. *)
type conversion = Cast | Bitcast

type move = Move.t =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Window of { axis : int; size : int; step : int }

type int32_t = (int32, Nx_dtype.int32_elt) Value.t
type int64_t = (int64, Nx_dtype.int64_elt) Value.t

type _ t =
  | Unary : Nx_backend.unary * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Binary :
      Nx_backend.binary * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Compare :
      Nx_backend.compare * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> (bool, Nx_dtype.bool_elt) Value.t t
  | Where :
      (bool, Nx_dtype.bool_elt) Value.t * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Fma :
      ('a, 'b) Value.t * ('a, 'b) Value.t * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Reduce :
      Nx_backend.reduce * int array * ('a, 'b) Value.t
      -> ('a, 'b) Value.t t
  | Scan : Nx_backend.reduce * int * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Arg_reduce : Nx_backend.arg_reduce * int * ('a, 'b) Value.t -> int64_t t
  | Sort : {
      descending : bool;
      axis : int;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Argsort : {
      descending : bool;
      axis : int;
      x : ('a, 'b) Value.t;
    }
      -> int64_t t
  | Group : {
      by : string;
      x : (int64, Nx_dtype.uint64_elt) Value.t;
    }
      -> int64_t t
  | Pad : (int * int) array * 'a * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Cat : int * ('a, 'b) Value.t list -> ('a, 'b) Value.t t
  | Convert :
      conversion * ('c, 'd) Nx_dtype.t * ('a, 'b) Value.t
      -> ('c, 'd) Value.t t
  | Threefry : int32_t * int32_t -> int32_t t
  | Gather : int * int64_t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Scatter : {
      mode : Nx_backend.scatter;
      unique : bool;
      axis : int;
      indices : int64_t;
      updates : ('a, 'b) Value.t;
      into : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Update : ('a, 'b) Value.t * int64_t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Unfold : {
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Fold : {
      output_size : int array;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
      x : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Matmul : ('a, 'b) Value.t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Fft : {
      inverse : bool;
      axes : int array;
      x : (Complex.t, 'b) Value.t;
    }
      -> (Complex.t, 'b) Value.t t
  | Rfft : {
      dtype : (Complex.t, 'c) Nx_dtype.t;
      axes : int array;
      x : (float, 'b) Value.t;
    }
      -> (Complex.t, 'c) Value.t t
  | Irfft : {
      dtype : (float, 'c) Nx_dtype.t;
      axes : int array;
      s : int array option;
      x : (Complex.t, 'b) Value.t;
    }
      -> (float, 'c) Value.t t
  | Contiguous : ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Cholesky : { upper : bool; x : ('a, 'b) Value.t } -> ('a, 'b) Value.t t
  | Qr : {
      reduced : bool;
      x : ('a, 'b) Value.t;
    }
      -> (('a, 'b) Value.t * ('a, 'b) Value.t) t
  | Lu : ('a, 'b) Value.t -> (('a, 'b) Value.t * int64_t * int64_t) t
  | Svd : {
      full_matrices : bool;
      x : ('a, 'b) Value.t;
    }
      -> (('a, 'b) Value.t
         * (float, Nx_dtype.float64_elt) Value.t
         * ('a, 'b) Value.t)
         t
  | Eig : {
      vectors : bool;
      x : ('a, 'b) Value.t;
    }
      -> ((Complex.t, Nx_dtype.complex64_elt) Value.t
         * (Complex.t, Nx_dtype.complex64_elt) Value.t option)
         t
  | Eigh : {
      vectors : bool;
      x : ('a, 'b) Value.t;
    }
      -> ((float, Nx_dtype.float64_elt) Value.t * ('a, 'b) Value.t option) t
  | Solve_triangular : {
      upper : bool;
      transpose : bool;
      unit_diag : bool;
      a : ('a, 'b) Value.t;
      b : ('a, 'b) Value.t;
    }
      -> ('a, 'b) Value.t t
  | Move : ('a, 'b) Value.t * move -> ('a, 'b) Value.t t
  | Place : Placement.t * ('a, 'b) Value.t -> ('a, 'b) Value.t t
  | Read : { by : string; x : ('a, 'b) Value.t } -> Nx_device.Buffer.t t
  | Check : {
      ok : (bool, Nx_dtype.bool_elt) Value.t;
      msg : int array -> string;
    }
      -> unit t

let name : type r. r t -> string =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (k, _) -> (
      match k with
      | Neg -> "neg"
      | Recip -> "recip"
      | Abs -> "abs"
      | Sqrt -> "sqrt"
      | Sign -> "sign"
      | Exp -> "exp"
      | Log -> "log"
      | Log1p -> "log1p"
      | Expm1 -> "expm1"
      | Sin -> "sin"
      | Cos -> "cos"
      | Tan -> "tan"
      | Asin -> "asin"
      | Acos -> "acos"
      | Atan -> "atan"
      | Sinh -> "sinh"
      | Cosh -> "cosh"
      | Tanh -> "tanh"
      | Trunc -> "trunc"
      | Ceil -> "ceil"
      | Floor -> "floor"
      | Round -> "round"
      | Erf -> "erf")
  | Binary (k, _, _) -> (
      match k with
      | Add -> "add"
      | Sub -> "sub"
      | Mul -> "mul"
      | Fdiv | Idiv -> "div"
      | Mod -> "mod"
      | Pow -> "pow"
      | Atan2 -> "atan2"
      | Maximum -> "maximum"
      | Minimum -> "minimum"
      | And -> "bitwise_and"
      | Or -> "bitwise_or"
      | Xor -> "bitwise_xor")
  | Compare (k, _, _) -> (
      match k with
      | Equal -> "equal"
      | Not_equal -> "not_equal"
      | Less -> "less"
      | Less_equal -> "less_equal")
  | Where _ -> "where"
  | Fma _ -> "fma"
  | Reduce (k, _, _) -> (
      match k with Sum -> "sum" | Prod -> "prod" | Max -> "max" | Min -> "min")
  | Scan (k, _, _) -> (
      match k with
      | Sum -> "cumsum"
      | Prod -> "cumprod"
      | Max -> "cummax"
      | Min -> "cummin")
  | Arg_reduce (k, _, _) -> (
      match k with Argmax -> "argmax" | Argmin -> "argmin")
  | Sort _ -> "sort"
  | Argsort _ -> "argsort"
  | Group _ -> "group"
  | Pad _ -> "pad"
  | Cat _ -> "concatenate"
  | Convert (k, _, _) -> ( match k with Cast -> "cast" | Bitcast -> "bitcast")
  | Threefry _ -> "threefry"
  | Gather _ -> "take_along_axis"
  | Scatter _ -> "scatter"
  | Update _ -> "set"
  | Unfold _ -> "unfold"
  | Fold _ -> "fold"
  | Matmul _ -> "matmul"
  | Fft { inverse; _ } -> if inverse then "ifft" else "fft"
  | Rfft _ -> "rfft"
  | Irfft _ -> "irfft"
  | Contiguous _ -> "contiguous"
  | Cholesky _ -> "cholesky"
  | Qr _ -> "qr"
  | Lu _ -> "lu"
  | Svd _ -> "svd"
  | Eig { vectors; _ } -> if vectors then "eig" else "eigvals"
  | Eigh { vectors; _ } -> if vectors then "eigh" else "eigvalsh"
  | Solve_triangular _ -> "solve_triangular"
  | Move (_, m) -> (
      match m with
      | Reshape _ -> "reshape"
      | Expand _ -> "expand"
      | Permute _ -> "permute"
      | Shrink _ -> "shrink"
      | Flip _ -> "flip"
      | Window _ -> "sliding_window")
  | Place _ -> "place"
  | Read _ -> "read"
  | Check _ -> "check"

let operands : type r. r t -> Value.packed list =
 fun op ->
  match[@warning "@4@8"] op with
  | Unary (_, x) -> [ Value.P x ]
  | Binary (_, a, b) -> [ Value.P a; Value.P b ]
  | Compare (_, a, b) -> [ Value.P a; Value.P b ]
  | Where (c, a, b) -> [ Value.P c; Value.P a; Value.P b ]
  | Fma (a, b, c) -> [ Value.P a; Value.P b; Value.P c ]
  | Reduce (_, _, x) -> [ Value.P x ]
  | Scan (_, _, x) -> [ Value.P x ]
  | Arg_reduce (_, _, x) -> [ Value.P x ]
  | Sort { x; _ } -> [ Value.P x ]
  | Argsort { x; _ } -> [ Value.P x ]
  | Group { x; _ } -> [ Value.P x ]
  | Pad (_, _, x) -> [ Value.P x ]
  | Cat (_, xs) -> List.map (fun x -> Value.P x) xs
  | Convert (_, _, x) -> [ Value.P x ]
  | Threefry (key, ctr) -> [ Value.P key; Value.P ctr ]
  | Gather (_, indices, x) -> [ Value.P x; Value.P indices ]
  | Scatter { indices; updates; into; _ } ->
      [ Value.P into; Value.P indices; Value.P updates ]
  | Update (x, starts, v) -> [ Value.P x; Value.P starts; Value.P v ]
  | Unfold { x; _ } -> [ Value.P x ]
  | Fold { x; _ } -> [ Value.P x ]
  | Matmul (a, b) -> [ Value.P a; Value.P b ]
  | Fft { x; _ } -> [ Value.P x ]
  | Rfft { x; _ } -> [ Value.P x ]
  | Irfft { x; _ } -> [ Value.P x ]
  | Contiguous x -> [ Value.P x ]
  | Cholesky { x; _ } -> [ Value.P x ]
  | Qr { x; _ } -> [ Value.P x ]
  | Lu x -> [ Value.P x ]
  | Svd { x; _ } -> [ Value.P x ]
  | Eig { x; _ } -> [ Value.P x ]
  | Eigh { x; _ } -> [ Value.P x ]
  | Solve_triangular { a; b; _ } -> [ Value.P a; Value.P b ]
  | Move (x, _) -> [ Value.P x ]
  | Place (_, x) -> [ Value.P x ]
  | Read { x; _ } -> [ Value.P x ]
  | Check { ok; _ } -> [ Value.P ok ]

type mapper = { f : 'a 'b. ('a, 'b) Value.t -> ('a, 'b) Value.t }

let map_operands : type r. mapper -> r t -> r t =
 fun o op ->
  let f = o.f in
  match[@warning "@4@8"] op with
  | Unary (k, x) -> Unary (k, f x)
  | Binary (k, a, b) -> Binary (k, f a, f b)
  | Compare (k, a, b) -> Compare (k, f a, f b)
  | Where (c, a, b) -> Where (f c, f a, f b)
  | Fma (a, b, c) -> Fma (f a, f b, f c)
  | Reduce (k, axes, x) -> Reduce (k, axes, f x)
  | Scan (k, axis, x) -> Scan (k, axis, f x)
  | Arg_reduce (k, axis, x) -> Arg_reduce (k, axis, f x)
  | Sort s -> Sort { s with x = f s.x }
  | Argsort s -> Argsort { s with x = f s.x }
  | Group g -> Group { g with x = f g.x }
  | Pad (padding, v, x) -> Pad (padding, v, f x)
  | Cat (axis, xs) -> Cat (axis, List.map f xs)
  | Convert (c, dtype, x) -> Convert (c, dtype, f x)
  | Threefry (key, ctr) -> Threefry (f key, f ctr)
  | Gather (axis, indices, data) -> Gather (axis, f indices, f data)
  | Scatter s ->
      Scatter
        { s with indices = f s.indices; updates = f s.updates; into = f s.into }
  | Update (x, starts, v) -> Update (f x, f starts, f v)
  | Unfold u -> Unfold { u with x = f u.x }
  | Fold u -> Fold { u with x = f u.x }
  | Matmul (a, b) -> Matmul (f a, f b)
  | Fft t -> Fft { t with x = f t.x }
  | Rfft t -> Rfft { t with x = f t.x }
  | Irfft t -> Irfft { t with x = f t.x }
  | Contiguous x -> Contiguous (f x)
  | Cholesky c -> Cholesky { c with x = f c.x }
  | Qr q -> Qr { q with x = f q.x }
  | Lu x -> Lu (f x)
  | Svd d -> Svd { d with x = f d.x }
  | Eig d -> Eig { d with x = f d.x }
  | Eigh d -> Eigh { d with x = f d.x }
  | Solve_triangular t -> Solve_triangular { t with a = f t.a; b = f t.b }
  | Move (x, m) -> Move (f x, m)
  | Place (p, x) -> Place (p, f x)
  | Read r -> Read { r with x = f r.x }
  | Check c -> Check { c with ok = f c.ok }

let pp ppf op =
  let operand ppf (Value.P x) =
    Format.fprintf ppf "%s%s"
      (Nx_dtype.to_string (Value.dtype x))
      (Shape.to_string (View.shape (Value.view x)))
  in
  let space ppf () = Format.pp_print_char ppf ' ' in
  Format.fprintf ppf "%s %a" (name op)
    (Format.pp_print_list ~pp_sep:space operand)
    (operands op)

(* Results' metadata

   The shape and dtype of an operation's result, from its operands', without
   computing it: what nx allocates before a kernel writes it. *)

let pad_shape padding s =
  Array.mapi
    (fun i d ->
      let before, after = padding.(i) in
      d + before + after)
    s

(* The shape of [shape]'s elements of [src] read as [dst]: a [k] times wider
   [dst] consumes the last axis, of [k], and a [k] times narrower one adds a
   last axis of [k]. *)
let bitcast_shape src dst shape =
  let w = Nx_dtype.itemsize src and w' = Nx_dtype.itemsize dst in
  if w' > w then Array.sub shape 0 (Array.length shape - 1)
  else if w' < w then Array.append shape [| w / w' |]
  else shape

let cat_shape axis = function
  | [] -> invalid_arg "Nx.concatenate: no value to concatenate"
  | s :: _ as shapes ->
      let total = List.fold_left (fun n s -> n + s.(axis)) 0 shapes in
      Array.mapi (fun i d -> if i = axis then total else d) s

(* The windows along each spatial axis of extents [spatial]: none where the
   dilated kernel is longer than the padded extent. *)
let window_counts kernel_size stride dilation padding spatial =
  Array.mapi
    (fun i n ->
      let before, after = padding.(i) in
      let extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
      let padded = n + before + after in
      if padded < extent then 0 else ((padded - extent) / stride.(i)) + 1)
    spatial

let unfold_shape kernel_size stride dilation padding s =
  let k = Array.length kernel_size in
  let lead = Array.length s - k in
  let windows =
    window_counts kernel_size stride dilation padding (Array.sub s lead k)
  in
  Array.append (Array.sub s 0 lead)
    [| Array.fold_left ( * ) 1 kernel_size; Array.fold_left ( * ) 1 windows |]

let fold_shape output_size s =
  Array.append (Array.sub s 0 (Array.length s - 2)) output_size

(* Leading batch axes broadcast; the product of [m, k] and [k, n] is [m, n]. *)
let matmul_shape sa sb =
  let na = Array.length sa and nb = Array.length sb in
  let r = Int.max na nb - 2 in
  let batch =
    Array.init r (fun i ->
        let da = if i - (r + 2 - na) >= 0 then sa.(i - (r + 2 - na)) else 1 in
        let db = if i - (r + 2 - nb) >= 0 then sb.(i - (r + 2 - nb)) else 1 in
        if da = 1 then db else da)
  in
  Array.append batch [| sa.(na - 2); sb.(nb - 1) |]

(* The last transformed axis of a real transform holds [n / 2 + 1] complex
   values, and its inverse [s]'s last size, or [2 (n - 1)]. *)
let rfft_shape axes s =
  let s = Array.copy s in
  let last = axes.(Array.length axes - 1) in
  s.(last) <- (s.(last) / 2) + 1;
  s

let irfft_shape axes sizes s =
  let s = Array.copy s in
  let n = Array.length axes - 1 in
  let last = axes.(n) in
  s.(last) <-
    (match sizes with Some sizes -> sizes.(n) | None -> (s.(last) - 1) * 2);
  s

let shape : type a b. (a, b) Value.t t -> int array =
 fun op ->
  let s x = View.shape (Value.view x) in
  match op with
  | Unary (_, x) -> s x
  | Binary (_, a, _) -> s a
  | Compare (_, a, _) -> s a
  | Where (_, a, _) -> s a
  | Fma (a, _, _) -> s a
  | Reduce (_, axes, x) -> Shape.reduce_output_shape (s x) axes false
  | Scan (_, _, x) -> s x
  | Arg_reduce (_, axis, x) -> Shape.reduce_output_shape (s x) [| axis |] false
  | Sort { x; _ } -> s x
  | Argsort { x; _ } -> s x
  | Group { x; _ } -> [| (s x).(0) |]
  | Pad (padding, _, x) -> pad_shape padding (s x)
  | Cat (axis, xs) -> cat_shape axis (List.map s xs)
  | Convert (Cast, _, x) -> s x
  | Convert (Bitcast, dt, x) -> bitcast_shape (Value.dtype x) dt (s x)
  | Threefry (_, ctr) -> s ctr
  | Gather (_, indices, _) -> s indices
  | Scatter { into; _ } -> s into
  | Update (x, _, _) -> s x
  | Unfold { kernel_size; stride; dilation; padding; x } ->
      unfold_shape kernel_size stride dilation padding (s x)
  | Fold { output_size; x; _ } -> fold_shape output_size (s x)
  | Matmul (a, b) -> matmul_shape (s a) (s b)
  | Fft { x; _ } -> s x
  | Rfft { axes; x; _ } -> rfft_shape axes (s x)
  | Irfft { axes; s = sizes; x; _ } -> irfft_shape axes sizes (s x)
  | Contiguous x -> s x
  | Cholesky { x; _ } -> s x
  | Solve_triangular { b; _ } -> s b
  | Move (x, m) -> View.shape (Move.view (Value.view x) m)
  | Place (_, x) -> s x
  (* A read gives a buffer, whose abstract type the checker cannot tell from a
     value's. *)
  | Read _ -> assert false

let dtype : type a b. (a, b) Value.t t -> (a, b) Nx_dtype.t =
 fun op ->
  match op with
  | Unary (_, x) -> Value.dtype x
  | Binary (_, a, _) -> Value.dtype a
  | Compare _ -> Nx_dtype.Bool
  | Where (_, a, _) -> Value.dtype a
  | Fma (a, _, _) -> Value.dtype a
  | Reduce (_, _, x) -> Value.dtype x
  | Scan (_, _, x) -> Value.dtype x
  | Arg_reduce _ -> Nx_dtype.Int64
  | Sort { x; _ } -> Value.dtype x
  | Argsort _ -> Nx_dtype.Int64
  | Group _ -> Nx_dtype.Int64
  | Pad (_, _, x) -> Value.dtype x
  | Cat (_, x :: _) -> Value.dtype x
  | Cat (_, []) -> invalid_arg "Nx.concatenate: no value to concatenate"
  | Convert (_, dt, _) -> dt
  | Threefry _ -> Nx_dtype.Int32
  | Gather (_, _, data) -> Value.dtype data
  | Scatter { into; _ } -> Value.dtype into
  | Update (x, _, _) -> Value.dtype x
  | Unfold { x; _ } -> Value.dtype x
  | Fold { x; _ } -> Value.dtype x
  | Matmul (a, _) -> Value.dtype a
  | Fft { x; _ } -> Value.dtype x
  | Rfft { dtype; _ } -> dtype
  | Irfft { dtype; _ } -> dtype
  | Contiguous x -> Value.dtype x
  | Cholesky { x; _ } -> Value.dtype x
  | Solve_triangular { b; _ } -> Value.dtype b
  | Move (x, _) -> Value.dtype x
  | Place (_, x) -> Value.dtype x
  | Read _ -> assert false
