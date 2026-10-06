(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
module B = Entry

let err op fmt =
  Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

(* [float_text s x] is the fewest significant digits that round to [x] in the
   format [s]: the float32 [0.1] is [0.1], not [0.100000001]. It
   is written in full for decimal exponents from -4 to 15, as [1234567900] or
   [0.0001], and with an exponent beyond, as [1e+16] or [1.5e-05]. NaN is [nan]
   whatever its sign, which C libraries print differently. *)
let float_text s x =
  let round =
    match (s : Nx_dtype.Scalar.t) with
    | Float64 -> Fun.id
    | Float32 -> fun f -> Int32.float_of_bits (Int32.bits_of_float f)
    | s ->
        let open Nx_dtype.Scalar in
        (* The float8 formats store a value past their largest finite one as
           that value, but only those within half a step above it round to it:
           e4m3's 448 prints as [450], not [500]. *)
        let top = decode s (encode s Float.max_float) in
        let limit =
          if Float.is_finite top then
            top +. ((top -. decode s (encode s top - 1)) /. 2.)
          else Float.infinity
        in
        fun f -> if Float.abs f > limit then Float.nan else decode s (encode s f)
  in
  let sign = if Float.sign_bit x then "-" else "" in
  let a = Float.abs x in
  let reads (m, e) = round (float_of_string (Printf.sprintf "%de%d" m e)) = a in
  let rec pow10 p = if p = 0 then 1 else 10 * pow10 (p - 1) in
  (* The [p] digits [m] and the exponent [e] of [m 10^e] nearest [a], or the
     next decimal above or below it: the values that read back to [a] are an
     interval around it, which holds a [p]-digit decimal only if it holds one of
     the two around [a]. Seventeen digits read back to any float. *)
  let rec shortest p =
    let t = Printf.sprintf "%.*e" (p - 1) a in
    let i = String.index t 'e' in
    let m =
      int_of_string
        (String.concat "" (String.split_on_char '.' (String.sub t 0 i)))
    in
    let e =
      int_of_string (String.sub t (i + 1) (String.length t - i - 1)) - (p - 1)
    in
    let below =
      if m - 1 < pow10 (p - 1) then ((10 * m) - 1, e - 1) else (m - 1, e)
    in
    match List.find_opt reads [ (m, e); (m + 1, e); below ] with
    | Some c -> c
    | None -> if p >= 17 then (m, e) else shortest (p + 1)
  in
  if Float.is_nan x then "nan"
  else if a = Float.infinity then sign ^ "inf"
  else if a = 0. then sign ^ "0"
  else
    let m, e = shortest 1 in
    let digits = string_of_int m in
    let exp = e + String.length digits - 1 in
    (* [m] ends in zeros after a carry, as [m + 1] at [99] gives [100]. *)
    let p = ref (String.length digits) in
    while digits.[!p - 1] = '0' do
      decr p
    done;
    let p = !p in
    let d = String.sub digits 0 p in
    sign
    ^
    if exp < -4 || exp >= 16 then
      let fraction = if p > 1 then "." ^ String.sub d 1 (p - 1) else "" in
      Printf.sprintf "%c%se%c%02d" d.[0] fraction
        (if exp < 0 then '-' else '+')
        (Int.abs exp)
    else if exp >= p - 1 then d ^ String.make (exp - p + 1) '0'
    else if exp >= 0 then
      String.sub d 0 (exp + 1) ^ "." ^ String.sub d (exp + 1) (p - exp - 1)
    else "0." ^ String.make (-exp - 1) '0' ^ d

(* ───── Core Types ───── *)

type ('a, 'b) t = ('a, 'b) Value.t
type context = Value.context
type float16_elt = Nx_dtype.float16_elt
type float32_elt = Nx_dtype.float32_elt
type float64_elt = Nx_dtype.float64_elt
type bfloat16_elt = Nx_dtype.bfloat16_elt
type float8_e4m3_elt = Nx_dtype.float8_e4m3_elt
type float8_e5m2_elt = Nx_dtype.float8_e5m2_elt
type int4_elt = Nx_dtype.int4_elt
type uint4_elt = Nx_dtype.uint4_elt
type int8_elt = Nx_dtype.int8_elt
type uint8_elt = Nx_dtype.uint8_elt
type int16_elt = Nx_dtype.int16_elt
type uint16_elt = Nx_dtype.uint16_elt
type int32_elt = Nx_dtype.int32_elt
type uint32_elt = Nx_dtype.uint32_elt
type int64_elt = Nx_dtype.int64_elt
type uint64_elt = Nx_dtype.uint64_elt
type complex32_elt = Nx_dtype.complex32_elt
type complex64_elt = Nx_dtype.complex64_elt
type bool_elt = Nx_dtype.bool_elt
type bit_elt = Nx_dtype.bit_elt

type ('a, 'b) dtype = ('a, 'b) Nx_dtype.t =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | BFloat16 : (float, bfloat16_elt) dtype
  | Float8_e4m3 : (float, float8_e4m3_elt) dtype
  | Float8_e5m2 : (float, float8_e5m2_elt) dtype
  | Int4 : (int, int4_elt) dtype
  | UInt4 : (int, uint4_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | UInt32 : (int32, uint32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt64 : (int64, uint64_elt) dtype
  | Complex64 : (Complex.t, complex32_elt) dtype
  | Complex128 : (Complex.t, complex64_elt) dtype
  | Bool : (bool, bool_elt) dtype
  | Bit : (bool, bit_elt) dtype

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type uint32_t = (int32, uint32_elt) t
type uint64_t = (int64, uint64_elt) t
type complex64_t = (Complex.t, complex32_elt) t
type complex128_t = (Complex.t, complex64_elt) t
type bool_t = (bool, bool_elt) t
type bit_t = (bool, bit_elt) t

let float16 = Float16
let float32 = Float32
let float64 = Float64
let bfloat16 = BFloat16
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2
let int4 = Int4
let uint4 = UInt4
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let uint32 = UInt32
let int64 = Int64
let uint64 = UInt64
let complex64 = Complex64
let complex128 = Complex128
let bool = Bool
let bit = Bit

type index =
  | I of int
  | L of int list
  | R of int * int
  | Rs of int * int * int
  | A
  | M of (bool, bool_elt) t
  | N
  | D of (int64, Nx_dtype.int64_elt) t * int

(* ───── Tensor Properties ───── *)

let shape x = View.shape (Value.view x)
let dtype x = Value.dtype x

let dim i x =
  let shape = View.shape (Value.view x) in
  let ndim = Array.length shape in
  let i = if i < 0 then i + ndim else i in
  if i < 0 || i >= ndim then
    err "dim" "axis %d out of bounds for %dD tensor" i ndim
  else shape.(i)

let ndim x = View.ndim (Value.view x)
let size x = View.numel (Value.view x)
let numel x = size x

let nbytes x =
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype (Value.dtype x))) in
  ((numel x * bits) + 7) / 8

let is_c_contiguous x = View.is_c_contiguous (Value.view x)

(* ───── Internal Utilities ───── *)

let array_prod arr = Array.fold_left ( * ) 1 arr

module IntSet = Set.Make (Int)

(* 2^shift_val for integer dtypes, used by lshift/rshift. *)
let power_of_two : type a b. (a, b) Nx_dtype.t -> int -> a =
 fun dtype shift_val ->
  if shift_val < 0 then
    err "power_of_two" "shift_val must be >= 0, got %d" shift_val;
  match dtype with
  | Int4 -> 1 lsl shift_val
  | UInt4 -> (1 lsl shift_val) land 0xF
  | Int8 -> 1 lsl shift_val
  | UInt8 -> (1 lsl shift_val) land 0xFF
  | Int16 -> 1 lsl shift_val
  | UInt16 -> (1 lsl shift_val) land 0xFFFF
  | Int32 -> Int32.shift_left Int32.one shift_val
  | UInt32 -> Int32.shift_left Int32.one shift_val
  | Int64 -> Int64.shift_left Int64.one shift_val
  | UInt64 -> Int64.shift_left Int64.one shift_val
  | _ ->
      err "power_of_two" "dtype %s, not an integer type"
        (Nx_dtype.to_string dtype)

let ensure_float_dtype fname x =
  if not (Nx_dtype.is_float (dtype x)) then
    err fname "dtype %s, expected float type (Float16, Float32, or Float64)"
      (Nx_dtype.to_string (dtype x))

let ensure_int_dtype fname x =
  if not (Nx_dtype.is_int (dtype x)) then
    invalid_arg (fname ^ ": dtype must be an integer type")

let resolve_axis ?ndim_opt x (axis_opt : int option) =
  let ndim = match ndim_opt with Some n -> n | None -> ndim x in
  match axis_opt with
  | None -> Array.init ndim Fun.id
  | Some a ->
      let resolved_a = if a < 0 then a + ndim else a in
      [| resolved_a |]

let resolve_single_axis ?ndim_opt x axis : int =
  let ndim = match ndim_opt with Some n -> n | None -> ndim x in
  if axis < 0 then axis + ndim else axis

(* Normalize negative axes, validate bounds, sort, and deduplicate. *)
let normalize_and_dedup_axes ~op ndim axes =
  let normalized =
    List.map
      (fun ax ->
        let axis = if ax < 0 then ndim + ax else ax in
        if axis < 0 || axis >= ndim then
          err op "axis %d out of bounds for %dD tensor" ax ndim;
        axis)
      axes
  in
  List.sort_uniq compare normalized

(* Count elements across reduction axes. *)
let reduction_element_count input_shape ?axes () =
  let rank = Array.length input_shape in
  let axes_arr =
    match axes with
    | None -> Array.init rank Fun.id
    | Some ax_list ->
        Array.of_list
          (List.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list)
  in
  if Array.length axes_arr = 0 then 1
  else array_prod (Array.map (fun ax -> input_shape.(ax)) axes_arr)

(* ───── Shape Manipulation Helpers ───── *)

let reshape shape_spec x =
  let current_shape = shape x in
  (* Resolve -1 dimensions *)
  let infer_count = ref 0 in
  Array.iter (fun d -> if d = -1 then incr infer_count) shape_spec;
  if !infer_count > 1 then
    invalid_arg
      "reshape: shape specification, multiple -1 dimensions, can only \
       specify one unknown dimension";
  let target_shape =
    if !infer_count = 0 then shape_spec
    else
      let old_numel = array_prod current_shape in
      let known_numel = ref 1 in
      Array.iter
        (fun d -> if d <> -1 then known_numel := !known_numel * d)
        shape_spec;
      if !known_numel = 0 || old_numel mod !known_numel <> 0 then
        err "reshape" "cannot infer dimension: %d elements into shape %s"
          old_numel
          (Shape.to_string shape_spec);
      let inferred = old_numel / !known_numel in
      Array.map (fun d -> if d = -1 then inferred else d) shape_spec
  in
  Array.iter
    (fun d ->
      if d < 0 then err "reshape" "shape specification, dimension %d < -1" d)
    target_shape;
  if Shape.equal current_shape target_shape then x
  else B.reshape x target_shape

let broadcast_shapes shape_a shape_b =
  let rank_a = Array.length shape_a in
  let rank_b = Array.length shape_b in
  let rank_out = max rank_a rank_b in
  let result = Array.make rank_out 1 in
  for i = 0 to rank_out - 1 do
    let idx_a = rank_a - rank_out + i in
    let idx_b = rank_b - rank_out + i in
    let dim_a = if idx_a >= 0 then shape_a.(idx_a) else 1 in
    let dim_b = if idx_b >= 0 then shape_b.(idx_b) else 1 in
    result.(i) <-
      (if dim_a = dim_b then dim_a
       else if dim_a = 1 then dim_b
       else if dim_b = 1 then dim_a
       else
         err "broadcast"
           "cannot broadcast %s with %s (dim %d: %d\xe2\x89\xa0%d)"
           (Shape.to_string shape_a) (Shape.to_string shape_b) i dim_a dim_b)
  done;
  result

let broadcast_to new_shape x =
  Array.iter
    (fun dim ->
      if dim < 0 then err "broadcast_to" "target shape, dimension %d < 0" dim)
    new_shape;
  let current_shape = shape x in
  if Shape.equal current_shape new_shape then x
  else
    let rank_current = Array.length current_shape in
    let rank_target = Array.length new_shape in
    if rank_current > rank_target then
      err "broadcast_to"
        "rank mismatch: source rank %d exceeds target rank %d, target shape \
         must have at least as many dimensions as source"
        rank_current rank_target
    else
      let pad_count = rank_target - rank_current in
      let padded_shape =
        if pad_count <= 0 then current_shape
        else
          let arr = Array.make rank_target 1 in
          Array.blit current_shape 0 arr pad_count rank_current;
          arr
      in
      for i = 0 to rank_target - 1 do
        let curr_dim = padded_shape.(i) in
        let target_dim = new_shape.(i) in
        if curr_dim <> target_dim && curr_dim <> 1 then
          err "broadcast_to"
            "cannot broadcast %s to %s (dim %d: %d\xe2\x89\xa0%d)"
            (Shape.to_string padded_shape)
            (Shape.to_string new_shape)
            i curr_dim target_dim
      done;
      let x_aligned =
        if pad_count <= 0 then x else B.reshape x padded_shape
      in
      if shape x_aligned = new_shape then x_aligned
      else B.expand x_aligned new_shape

let broadcasted ?(reverse = false) x y =
  let a, b = if reverse then (y, x) else (x, y) in
  let sa = shape a and sb = shape b in
  if Shape.equal sa sb then (a, b)
  else
    let s = broadcast_shapes sa sb in
    (broadcast_to s a, broadcast_to s b)

(* Like [broadcast_to] but [-1] keeps the original dimension. *)
let expand shape_spec x =
  let current_shape = shape x in
  let rank_current = Array.length current_shape in
  let rank_spec = Array.length shape_spec in
  let rank_new = max rank_current rank_spec in
  let current_aligned = Array.make rank_new 1 in
  Array.blit current_shape 0 current_aligned (rank_new - rank_current)
    rank_current;
  let target_shape =
    Array.init rank_new (fun i ->
        let spec_idx = i - (rank_new - rank_spec) in
        let spec_dim = if spec_idx < 0 then -1 else shape_spec.(spec_idx) in
        if spec_dim = -1 then current_aligned.(i)
        else if spec_dim < -1 then
          err "expand" "dimension %d, negative size %d" i spec_dim
        else spec_dim)
  in
  broadcast_to target_shape x

(* ───── Type Conversion and Tensor Creation ───── *)

let cast (type a b c d) (dt : (c, d) Nx_dtype.t) (x : (a, b) t) : (c, d) t =
  match Nx_dtype.equal_witness (dtype x) dt with
  | Some Equal -> x
  | None -> B.cast dt x

let astype dt x = cast dt x

let bitcast (type a b c d) (dt : (c, d) Nx_dtype.t) (x : (a, b) t) : (c, d) t
    =
  let src = dtype x in
  let refuse reason =
    err "bitcast" "cannot reinterpret %s as %s, %s" (Nx_dtype.to_string src)
      (Nx_dtype.to_string dt) reason
  in
  let unfit (type e f) (d : (e, f) Nx_dtype.t) =
    match d with Nx_dtype.Bool -> Some "bool holds only 0 and 1" | _ -> None
  in
  (match (unfit src, unfit dt) with
  | Some reason, _ | None, Some reason -> refuse reason
  | None, None -> ());
  let bits d = Nx_dtype.Scalar.(bitsize (of_dtype d)) in
  let w = bits src and w' = bits dt in
  (if w' > w then
     let k = w' / w and s = shape x in
     let r = Array.length s in
     if r = 0 then
       refuse
         (Printf.sprintf "a %s reads a last axis of %d and a scalar has none"
            (Nx_dtype.to_string dt) k)
     else if s.(r - 1) <> k then
       refuse
         (Printf.sprintf "a %s reads a last axis of %d, not %d"
            (Nx_dtype.to_string dt) k
            s.(r - 1)));
  B.bitcast dt x

let contiguous x = B.contiguous x
let copy x = B.copy x

(* [f] applied to the reader of [x]'s elements in C order, from index 0, read by
   the surface function [by]. The memory is under a read claim until [f]
   returns, so that no compiled call lends it meanwhile. *)
let reading ~by x f =
  let buf = B.read ~by x in
  Nx_device.Buffer.Claim.read buf;
  match f (Elements.get (Value.dtype x) buf) with
  | v ->
      Nx_device.Buffer.Claim.release buf;
      v
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Nx_device.Buffer.Claim.release buf;
      Printexc.raise_with_backtrace e bt

(* [x]'s elements in C order, read by [by]. *)
let read_array ~by x = reading ~by x (Array.init (numel x))

(* The one element of [t], read by [by]. *)
let read_item ~by t = reading ~by t (fun element -> element 0)

let check_shape op shape =
  if Array.exists (fun d -> d < 0) shape then
    err op "shape %s, dimensions must be >= 0" (Shape.to_string shape)

let create ctx dtype shape arr =
  check_shape "create" shape;
  let n = Array.fold_left ( * ) 1 shape in
  if Array.length arr <> n then
    err "create" "array size, got %d elements, expected %d" (Array.length arr)
      n;
  let buf = Elements.create dtype n in
  Array.iteri (Elements.set dtype buf) arr;
  let tensor_1d = B.from_host ctx dtype buf in
  if Array.length shape = 1 && shape.(0) = n then tensor_1d
  else B.reshape tensor_1d shape

let init ctx dtype shape f =
  check_shape "init" shape;
  let size = Array.fold_left ( * ) 1 shape in
  let arr = Array.init size (fun i -> f (Shape.unravel_index i shape)) in
  create ctx dtype shape arr

let scalar ctx dt value = B.full ctx dt [||] value
let scalar_like x_ref value =
  scalar (Value.context x_ref) (Value.dtype x_ref) value

let empty ctx dtype shape_arr =
  check_shape "empty" shape_arr;
  B.full ctx dtype shape_arr (Nx_dtype.zero dtype)

let full ctx dt target_shape fill_value =
  check_shape "full" target_shape;
  B.full ctx dt target_shape fill_value

let zeros ctx dtype shape_arr = full ctx dtype shape_arr (Nx_dtype.zero dtype)
let ones ctx dtype shape_arr = full ctx dtype shape_arr (Nx_dtype.one dtype)

let create_like x_ref fill_fn =
  fill_fn (Value.context x_ref) (Value.dtype x_ref) (shape x_ref)

let empty_like x_ref = create_like x_ref empty

let full_like x_ref fill_value =
  create_like x_ref (fun ctx dt sh -> full ctx dt sh fill_value)

let zeros_like x = full_like x (Nx_dtype.zero (Value.dtype x))
let fill value x = full_like x value
let ones_like x = full_like x (Nx_dtype.one (Value.dtype x))

let to_bigarray x =
  match Nx_dtype.to_bigarray_kind (Value.dtype x) with
  | None ->
      err "to_bigarray" "Bigarray has no %s kind"
        (Nx_dtype.to_string (Value.dtype x))
  | Some k ->
      let ba =
        Nx_device.Buffer.bigarray k (B.read ~by:"Nx.to_bigarray" (copy x))
      in
      Bigarray.reshape (Bigarray.genarray_of_array1 ba) (shape x)

let of_bigarray (type a b) ctx
    (ba : (a, b, Bigarray.c_layout) Bigarray.Genarray.t) =
  let dtype : (a, b) Nx_dtype.t =
    match Bigarray.Genarray.kind ba with
    | Bigarray.Char -> err "of_bigarray" "a char bigarray has no dtype"
    | Bigarray.Int -> err "of_bigarray" "an int bigarray has no dtype"
    | Bigarray.Nativeint ->
        err "of_bigarray" "a nativeint bigarray has no dtype"
    | k -> Nx_dtype.of_bigarray_kind k
  in
  let shape = Bigarray.Genarray.dims ba in
  let flat = Bigarray.reshape_1 ba (Array.fold_left ( * ) 1 shape) in
  reshape shape (B.from_host ctx dtype (Nx_device.Buffer.of_bigarray flat))

let to_array x = read_array ~by:"Nx.to_array" x

(* ───── Element-wise Binary Operations ───── *)

(* Operands of one shape are passed as they are. *)
let binop k a b =
  let sa = shape a and sb = shape b in
  if Shape.equal sa sb then B.binary k a b
  else
    let s = broadcast_shapes sa sb in
    B.binary k (broadcast_to s a) (broadcast_to s b)

let cmpop k a b =
  let sa = shape a and sb = shape b in
  if Shape.equal sa sb then B.cmp k a b
  else
    let s = broadcast_shapes sa sb in
    B.cmp k (broadcast_to s a) (broadcast_to s b)

let add a b = binop Add a b
let add_s t s = add t (scalar_like t s)
let sub a b = binop Sub a b
let sub_s t s = sub t (scalar_like t s)
let rsub_s s t = sub (scalar_like t s) t
let mul a b = binop Mul a b
let mul_s t s = mul t (scalar_like t s)

let div a b =
  let dt = Value.dtype a in
  if Nx_dtype.is_int dt || Nx_dtype.is_uint dt then binop Idiv a b
  else binop Fdiv a b

let div_s t s = div t (scalar_like t s)
let rdiv_s s t = div (scalar_like t s) t
let pow a b = binop Pow a b
let pow_s t s = pow t (scalar_like t s)
let rpow_s s t = pow (scalar_like t s) t
let maximum a b = binop Maximum a b
let maximum_s t s = maximum t (scalar_like t s)
let minimum a b = binop Minimum a b
let minimum_s t s = minimum t (scalar_like t s)
let mod_ a b = binop Mod a b
let mod_s t s = mod_ t (scalar_like t s)
let rmod_s s t = mod_ (scalar_like t s) t
let bitwise_xor a b = binop Xor a b
let bitwise_or a b = binop Or a b
let bitwise_and a b = binop And a b

(* ───── Logical and Comparison Operations ───── *)

(* A logical operation reads non-zero as true and gives zero or one of the
   operands' dtype. On [bool], whose only values are 0 and 1, it is the bitwise
   operation. *)
let truth x =
  cmpop Not_equal x (scalar_like x (Nx_dtype.zero (dtype x)))
let logical (type a b) op (a : (a, b) t) (b : (a, b) t) : (a, b) t =
  match dtype a with
  | Nx_dtype.Bool -> binop op a b
  | Nx_dtype.Bit -> binop op a b
  | _ -> cast (dtype a) (binop op (truth a) (truth b))
let logical_and a b = logical And a b
let logical_or a b = logical Or a b
let logical_xor a b = logical Xor a b

let bitwise_not x =
  let dt = dtype x in
  binop Xor x
    (broadcast_to (shape x)
       (B.full (Value.context x) dt [||] (Nx_dtype.minus_one dt)))

let logical_not (type a b) (x : (a, b) t) : (a, b) t =
  match dtype x with
  | Nx_dtype.Bit -> bitwise_not x
  | dt -> cast dt (cmpop Equal x (scalar_like x (Nx_dtype.zero dt)))

let cmpeq a b = cmpop Equal a b
let cmpne a b = cmpop Not_equal a b
let cmplt a b = cmpop Less a b
let cmple a b = cmpop Less_equal a b
let cmpgt a b = cmplt b a
let cmpge a b = cmple b a
let less = cmplt
let less_equal = cmple
let greater = cmpgt
let greater_equal = cmpge
let equal = cmpeq
let not_equal = cmpne
let equal_s a s = equal a (scalar_like a s)
let not_equal_s a s = not_equal a (scalar_like a s)
let less_s a s = less a (scalar_like a s)
let greater_s a s = greater a (scalar_like a s)
let less_equal_s a s = less_equal a (scalar_like a s)
let greater_equal_s a s = greater_equal a (scalar_like a s)

(* ───── Element-wise Unary Operations ───── *)

let neg x = B.unary Neg x
let sin x = B.unary Sin x
let cos x = B.unary Cos x
let sqrt x = B.unary Sqrt x
let recip x = B.unary Recip x
let log x = B.unary Log x
let exp x = B.unary Exp x
let abs x = B.unary Abs x

(* A function composed of several operations computes a narrow float at
   float32 and rounds once, as an operation of the backend does. *)
type composite = { f : 'a 'b. ('a, 'b) t -> ('a, 'b) t }

(* The floats narrower than float32. *)
let narrow dt = Nx_dtype.is_float dt && Nx_dtype.itemsize dt < 4

let at_float32 c x =
  if narrow (dtype x) then cast (dtype x) (c.f (cast Nx_dtype.float32 x))
  else c.f x

(* The same for a function of floats alone. *)
type real = { r : 'b. (float, 'b) t -> (float, 'b) t }

let real_at_float32 c x =
  if narrow (dtype x) then cast (dtype x) (c.r (cast Nx_dtype.float32 x))
  else c.r x

let log2 x =
  at_float32
    {
      f =
        (fun x ->
          mul (log x)
            (broadcast_to (shape x)
               (scalar (Value.context x) (dtype x)
                  (Nx_dtype.of_float (dtype x) (1.0 /. Stdlib.log 2.0)))));
    }
    x

let exp2 x = rpow_s (Nx_dtype.of_float (dtype x) 2.0) x
let tan x = B.unary Tan x
let square x = mul x x
let sign x = B.unary Sign x

(* [exp] only ever sees [-|x|], so it cannot overflow, and a negative [x]
   gives [e / (1 + e)], which keeps the subnormal tail where [1 / (1 + e)]
   would need [e] beyond the largest float. [x] is negated by selection, not
   by [abs], so the gradient at zero is that of one side. *)
let sigmoid x =
  at_float32
    {
      f =
        (fun x ->
          let negative = cmplt x (scalar_like x (Nx_dtype.zero (dtype x))) in
          let e = exp (B.where negative x (neg x)) in
          let r = recip (add_s e (Nx_dtype.one (dtype x))) in
          B.where negative (mul e r) r);
    }
    x

let rsqrt x = at_float32 { f = (fun x -> recip (sqrt x)) } x
let asin x = B.unary Asin x
let acos x = B.unary Acos x
let atan x = B.unary Atan x
let sinh x = B.unary Sinh x
let cosh x = B.unary Cosh x
let tanh x = B.unary Tanh x
let trunc x = B.unary Trunc x
let ceil x = B.unary Ceil x
let floor x = B.unary Floor x
let round x = B.unary Round x

let isinf x =
  if not (Nx_dtype.is_float (dtype x)) then
    zeros (Value.context x) Nx_dtype.bool (shape x)
  else
    let dt = dtype x in
    let pos_inf =
      broadcast_to (shape x)
        (B.full (Value.context x) dt [||] (Nx_dtype.of_float dt Float.infinity))
    in
    let neg_inf =
      broadcast_to (shape x)
        (B.full (Value.context x) dt [||]
           (Nx_dtype.of_float dt Float.neg_infinity))
    in
    logical_or (cmpeq x pos_inf) (cmpeq x neg_inf)

let isnan x =
  if not (Nx_dtype.is_float (dtype x)) then
    zeros (Value.context x) Nx_dtype.bool (shape x)
  else cmpne x x

let isfinite x =
  if not (Nx_dtype.is_float (dtype x)) then
    ones (Value.context x) Nx_dtype.bool (shape x)
  else logical_not (logical_or (isinf x) (isnan x))

let lerp start_tensor end_tensor weight =
  add start_tensor (mul (sub end_tensor start_tensor) weight)

let shift_op ~op ~apply x shift_val =
  let dt = dtype x in
  if not (Nx_dtype.is_int dt) then
    err op "dtype %s, expected integer type" (Nx_dtype.to_string dt);
  if shift_val < 0 then err op "shift_val must be >= 0, got %d" shift_val;
  if shift_val = 0 then x
  else
    apply x
      (broadcast_to (shape x)
         (B.full (Value.context x) dt [||] (power_of_two dt shift_val)))

(* [x * 2^n] modulo the width, so 0 once [n] reaches it. *)
let lshift x n =
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype (dtype x))) in
  if n >= bits && Nx_dtype.is_int (dtype x) then zeros_like x
  else shift_op ~op:"lshift" ~apply:mul x n

let clamp ?min ?max x =
  let x = match min with None -> x | Some min_v -> maximum_s x min_v in
  match max with None -> x | Some max_v -> minimum_s x max_v

let clip = clamp

(* ───── Ternary Operations ───── *)

let where cond if_true if_false =
  let sc = shape cond and st = shape if_true and sf = shape if_false in
  if Shape.equal sc st && Shape.equal st sf then B.where cond if_true if_false
  else
    let target = Shape.broadcast (Shape.broadcast st sf) sc in
    B.where (broadcast_to target cond) (broadcast_to target if_true)
      (broadcast_to target if_false)

let fma a b c =
  let dt = dtype a in
  if
    Nx_dtype.is_complex dt
    || Nx_dtype.equal dt Nx_dtype.bool
    || Nx_dtype.equal dt Nx_dtype.bit
  then
    err "fma" "dtype %s, expected a float or integer dtype"
      (Nx_dtype.to_string dt);
  let target =
    Shape.broadcast (Shape.broadcast (shape a) (shape b)) (shape c)
  in
  B.fma (broadcast_to target a) (broadcast_to target b) (broadcast_to target c)

let check ok msg = B.check ok msg

(* An arithmetic shift: [t / 2^n] rounded toward negative infinity. *)
let rshift x n =
  let dt = dtype x in
  let zero = scalar_like x (Nx_dtype.zero dt) in
  let one = scalar_like x (Nx_dtype.one dt) in
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype dt)) in
  if
    Nx_dtype.is_int dt
    && (not (Nx_dtype.is_uint dt))
    && n >= bits - 1
    && n >= 0
  then where (cmplt x zero) (sub zero one) (broadcast_to (shape x) zero)
  else if Nx_dtype.is_uint dt && n >= bits then zeros_like x
  else
    shift_op ~op:"rshift"
      ~apply:(fun x p ->
        (* [div] truncates toward zero: step down where it rounded up *)
        let q = div x p in
        where (cmplt (sub x (mul q p)) zero) (sub q one) q)
      x n

let float_only op x =
  if not (Nx_dtype.is_float (dtype x)) then
    err op "dtype %s, expected a float dtype" (Nx_dtype.to_string (dtype x))

let log1p x =
  float_only "log1p" x;
  B.unary Log1p x

let expm1 x =
  float_only "expm1" x;
  B.unary Expm1 x

(* Past [2^(p/2)], [p] the precision in bits, [sqrt (a^2 + 1)] rounds to [a],
   and [a^2] would overflow a narrow float. *)
let large_argument : type a b. (a, b) Nx_dtype.t -> float = function
  | Float8_e5m2 -> 2.
  | Float8_e4m3 -> 4.
  | BFloat16 -> 16.
  | Float16 -> 32.
  | Float32 -> 4096.
  | _ -> 0x1p26

let asinh x =
  at_float32
    {
      f =
        (fun x ->
          let dt = dtype x in
          let of_float = Nx_dtype.of_float dt and one = Nx_dtype.one dt in
          let a = abs x in
          let near =
            B.unary Log1p
              (add a
                 (div (square a) (add_s (sqrt (add_s (square a) one)) one)))
          in
          let far = add_s (log a) (of_float (Stdlib.log 2.)) in
          let r =
            where
              (cmpgt a (scalar_like x (of_float (large_argument dt))))
              far near
          in
          where (cmplt x (zeros_like x)) (neg r) r);
    }
    x

let acosh x =
  at_float32
    {
      f =
        (fun x ->
          let dt = dtype x in
          let of_float = Nx_dtype.of_float dt and one = Nx_dtype.one dt in
          let t = sub_s x one in
          let near = B.unary Log1p (add t (sqrt (add (add t t) (square t)))) in
          let far = add_s (log x) (of_float (Stdlib.log 2.)) in
          let r =
            where
              (cmpgt x (scalar_like x (of_float (large_argument dt))))
              far near
          in
          where
            (cmplt x (scalar_like x one))
            (scalar_like x (of_float Float.nan))
            r);
    }
    x

let atanh x =
  at_float32
    {
      f =
        (fun x ->
          let dt = dtype x in
          let of_float = Nx_dtype.of_float dt and one = Nx_dtype.one dt in
          let a = abs x in
          let r =
            mul_s (B.unary Log1p (div (add a a) (rsub_s one a))) (of_float 0.5)
          in
          where (cmplt x (zeros_like x)) (neg r) r);
    }
    x

(* ───── Binary Mathematical Functions ───── *)

let atan2 y x = binop Atan2 y x

(* sqrt(x² + y²) with overflow protection via max * sqrt(1 + (min/max)²) *)
let hypot_at x' y' =
  let dt = dtype x' in
  let x_abs = abs x' in
  let y_abs = abs y' in
  let max_val = maximum x_abs y_abs in
  let min_val = minimum x_abs y_abs in
  let both_zero =
    logical_and
      (equal_s x_abs (Nx_dtype.zero dt))
      (equal_s y_abs (Nx_dtype.zero dt))
  in
  let zero = scalar_like x' (Nx_dtype.zero dt) in
  let ratio = where both_zero zero (div min_val max_val) in
  let result = mul max_val (sqrt (add_s (square ratio) (Nx_dtype.one dt))) in
  let result = where both_zero zero result in
  (* An infinite side makes the length infinite, even beside a NaN. *)
  if not (Nx_dtype.is_float dt) then result
  else
    where
      (logical_or (isinf x') (isinf y'))
      (scalar_like x' (Nx_dtype.of_float dt Float.infinity))
      result

let hypot x y =
  let x, y = broadcasted x y in
  if narrow (dtype x) then
    let f32 = cast Nx_dtype.float32 in
    cast (dtype x) (hypot_at (f32 x) (f32 y))
  else hypot_at x y

(* ───── Reduction Operations ───── *)

let reduce_op op ?axes ?(keepdims = false) x =
  let input_shape = shape x in
  let rank = Array.length input_shape in
  let axes_to_reduce =
    match axes with
    | None -> Array.init rank Fun.id
    | Some ax_list ->
        Array.of_list
          (List.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list)
  in
  Array.iter
    (fun ax ->
      if ax < 0 || ax >= rank then
        err "reduce" "axis %d out of bounds for %dD tensor" ax rank)
    axes_to_reduce;
  (* An extreme over no element has no value. *)
  (match op with
  | Nx_backend.Max | Min ->
      Array.iter
        (fun ax ->
          if input_shape.(ax) = 0 then
            err "reduce" "axis %d is empty: its extreme has no value" ax)
        axes_to_reduce
  | Sum | Prod -> ());
  let reduced = B.reduce op ~axes:axes_to_reduce x in
  (* The backend drops the reduced axes; reinsert them as size 1 on
     request. *)
  if keepdims then
    reshape
      (Shape.reduce_output_shape input_shape axes_to_reduce true)
      reduced
  else reduced

let sum ?axes ?(keepdims = false) x = reduce_op Nx_backend.Sum ?axes ~keepdims x
let max ?axes ?(keepdims = false) x = reduce_op Nx_backend.Max ?axes ~keepdims x
let min ?axes ?(keepdims = false) x = reduce_op Nx_backend.Min ?axes ~keepdims x
let prod ?axes ?(keepdims = false) x =
  reduce_op Nx_backend.Prod ?axes ~keepdims x

let scan_along ~axis (op : Nx_backend.reduce) x =
  let name =
    match op with
    | Sum -> "cumsum"
    | Prod -> "cumprod"
    | Max -> "cummax"
    | Min -> "cummin"
  in
  let x_shape = shape x in
  let rank = Array.length x_shape in
  if rank = 0 then
    let a = if axis < 0 then axis + 1 else axis in
    if a = 0 then x
    else
      err name "axis %d out of bounds for rank 0 tensor (only axis 0 valid)"
        axis
  else
    let a = if axis < 0 then axis + rank else axis in
    if a < 0 || a >= rank then
      err name "axis %d out of bounds for %dD tensor" axis rank
    else B.scan op ~axis:a x

let flatten ?(start_dim = 0) ?(end_dim = -1) x =
  let sh = shape x in
  let r = Array.length sh in
  let s = if start_dim < 0 then start_dim + r else start_dim in
  let e = if end_dim < 0 then end_dim + r else end_dim in
  if
    not
      ((s >= 0 && s < r && e >= 0 && e < r)
      || (r = 0 && (s = 0 || start_dim = 0) && (e = -1 || end_dim = -1)))
  then
    err "flatten" "start_dim %d or end_dim %d, out of bounds for rank %d"
      start_dim end_dim r;
  if r > 0 && s > e then
    invalid_arg "flatten: dimensions, start_dim must be <= end_dim";
  let target =
    if r = 0 then [| 1 |]
    else
      Array.concat
        [
          Array.sub sh 0 s;
          [| array_prod (Array.sub sh s (e - s + 1)) |];
          Array.sub sh (e + 1) (r - (e + 1));
        ]
  in
  reshape target x

let cumulative_scan ?axis op x =
  let orig_shape = shape x in
  match axis with
  | Some axis -> scan_along ~axis op x
  | None ->
      let flat = flatten x in
      let scanned = scan_along ~axis:0 op flat in
      if Array.length orig_shape = 0 then reshape [||] scanned
      else reshape orig_shape scanned

let cumsum ?axis x = cumulative_scan ?axis Nx_backend.Sum x
let cumprod ?axis x = cumulative_scan ?axis Nx_backend.Prod x
let cummax ?axis x = cumulative_scan ?axis Nx_backend.Max x
let cummin ?axis x = cumulative_scan ?axis Nx_backend.Min x

(* The mean of [n] integers of [b] bits rounded toward zero, exact in a 64-bit
   accumulator [acc]. Below 64 bits, the sum of fewer than [2^(64 - b)] of them
   holds in [acc]. At 64 bits, with [x = q n + r] and [0 <= r < n], the sum is
   [n Q + R] for [Q] the sum of the quotients and [R] that of the remainders,
   so its floor is [Q + R / n]. The floor lies between the least and greatest
   element, so [Q] may wrap on the way; [R] stays below [n^2], below 2^62 for
   fewer than 2^31 elements. *)
let int_mean (type a b d) ?axes ~keepdims (acc : (int64, d) Nx_dtype.t)
    (x : (a, b) t) n : (a, b) t =
  let dt = Value.dtype x in
  let bits = Nx_dtype.Scalar.(bitsize (of_dtype dt)) in
  let exact = if bits = 64 then 1 lsl 31 else 1 lsl (64 - bits) in
  if n >= exact then
    err "mean" "%d elements of dtype %s, an integer mean takes fewer than %d" n
      (Nx_dtype.to_string dt) exact;
  let w = cast acc x in
  let n = Int64.of_int n in
  if bits < 64 then cast dt (div_s (sum ?axes ~keepdims w) n)
  else
    let q = div_s w n in
    let r = sub w (mul_s q n) in
    (* [div] truncates toward zero: step a negative remainder up *)
    let negative = less_s r 0L in
    let q = where negative (sub_s q 1L) q
    and r = where negative (add_s r n) r in
    let rs = sum ?axes ~keepdims r in
    let floor = add (sum ?axes ~keepdims q) (div_s rs n) in
    let inexact =
      logical_and (less_s floor 0L) (not_equal_s (mod_s rs n) 0L)
    in
    cast dt (where inexact (add_s floor 1L) floor)

let mean ?axes ?(keepdims = false) x =
  let dt = Value.dtype x in
  let n = reduction_element_count (shape x) ?axes () in
  (* The mean of nothing is 0 / 0: NaN, which an integer does not hold. *)
  if n = 0 && not (Nx_dtype.is_float dt || Nx_dtype.is_complex dt) then
    err "mean" "dtype %s, the mean of an empty axis has no value"
      (Nx_dtype.to_string dt);
  if Nx_dtype.is_uint dt then int_mean ?axes ~keepdims Nx_dtype.uint64 x n
  else if Nx_dtype.is_int dt then int_mean ?axes ~keepdims Nx_dtype.int64 x n
  else
    let s = sum ?axes ~keepdims x in
    let divisor =
      broadcast_to (shape s)
        (scalar (Value.context x) dt (Nx_dtype.of_float dt (float_of_int n)))
    in
    div s divisor

(* The variance of integers is a fraction that can outgrow their dtype. *)
let refuse_int op x =
  if Nx_dtype.is_int (dtype x) then
    err op "dtype %s, expected a float or complex dtype"
      (Nx_dtype.to_string (dtype x))

let var ?axes ?(keepdims = false) ?(ddof = 0) x =
  refuse_int "var" x;
  let dt = Value.dtype x in
  let mean_x = mean ?axes ~keepdims:true x in
  let sum_sq = sum ?axes ~keepdims (square (sub x mean_x)) in
  let n = reduction_element_count (shape x) ?axes () in
  if ddof >= n then err "var" "ddof %d, must be below the count %d" ddof n;
  let n_corr = float_of_int (n - ddof) in
  let divisor =
    broadcast_to (shape sum_sq)
      (scalar (Value.context x) dt (Nx_dtype.of_float dt n_corr))
  in
  div sum_sq divisor

let std ?axes ?(keepdims = false) ?(ddof = 0) x =
  refuse_int "std" x;
  sqrt (var ?axes ~keepdims ~ddof x)

(* A [bool] or [bit] tensor reduces as it is; any other is compared with zero
   first. *)
let truth_reduce (type a b) op axes (x : (a, b) t) : bool_t =
  match dtype x with
  | Bool -> B.reduce op ~axes x
  | Bit -> cast Bool (B.reduce op ~axes x)
  | dt -> B.reduce op ~axes (not_equal_s x (Nx_dtype.zero dt))

let logical_reduce ~op_name ~op ~identity ?axes ?(keepdims = false) x =
  let input_shape = shape x in
  let rank = Array.length input_shape in
  let axes_to_reduce =
    normalize_and_dedup_axes ~op:op_name rank
      (match axes with
      | None -> List.init rank Fun.id
      | Some values -> values)
    |> Array.of_list
  in
  if Array.exists (fun axis -> input_shape.(axis) = 0) axes_to_reduce then
    full (Value.context x) Nx_dtype.bool
      (Shape.reduce_output_shape input_shape axes_to_reduce keepdims)
      identity
  else
    let reduced = truth_reduce op axes_to_reduce x in
    if keepdims then
      reshape
        (Shape.reduce_output_shape input_shape axes_to_reduce true)
        reduced
    else reduced

let all ?axes ?(keepdims = false) x =
  logical_reduce ~op_name:"all" ~op:Nx_backend.Min ~identity:true ?axes
    ~keepdims x

let any ?axes ?(keepdims = false) x =
  logical_reduce ~op_name:"any" ~op:Nx_backend.Max ~identity:false ?axes
    ~keepdims x

let array_equal x y =
  if not (Array.equal Int.equal (shape x) (shape y)) then
    zeros (Value.context x) Nx_dtype.bool [||]
  else all (equal x y)

(* ───── Shape Manipulation ───── *)

let pad padding_config fill_value x =
  Array.iter
    (fun (before, after) ->
      if before < 0 || after < 0 then
        invalid_arg
          "pad: padding values, negative values not allowed, use shrink or \
           slice to remove elements")
    padding_config;
  B.pad padding_config fill_value x

let shrink shrink_args x = B.shrink x shrink_args

let unflatten dim sizes x =
  let dim = resolve_single_axis x dim in
  let current_shape = shape x in
  let dim_size = current_shape.(dim) in
  let sizes = Array.copy sizes in
  let neg_one_count =
    Array.fold_left (fun acc s -> if s = -1 then acc + 1 else acc) 0 sizes
  in
  if neg_one_count > 1 then
    invalid_arg
      "unflatten: sizes, can only specify one unknown dimension (using -1)";
  if neg_one_count = 1 then begin
    let known_product =
      Array.fold_left (fun acc s -> if s = -1 then acc else acc * s) 1 sizes
    in
    if known_product = 0 || dim_size mod known_product <> 0 then
      err "unflatten"
        "cannot infer dimension from total size %d to known product %d, %d \
         not divisible by %d, ensure total size is divisible by product of \
         known dimensions"
        dim_size known_product dim_size known_product;
    let inferred = dim_size / known_product in
    Array.iteri (fun i s -> if s = -1 then sizes.(i) <- inferred) sizes
  end;
  let sizes_product = Array.fold_left ( * ) 1 sizes in
  if sizes_product <> dim_size then
    err "unflatten" "sizes, product %d does not match dimension size %d"
      sizes_product dim_size;
  reshape
    (Array.concat
       [
         Array.sub current_shape 0 dim;
         sizes;
         Array.sub current_shape (dim + 1)
           (Array.length current_shape - dim - 1);
       ])
    x

let ravel x = reshape [| numel x |] x

let squeeze ?axes x =
  let sh = shape x in
  let r = Array.length sh in
  let reshape_or_id new_sh =
    if Array.length new_sh = 0 && r > 0 then reshape [||] x
    else if Array.length new_sh = 0 then x
    else reshape new_sh x
  in
  match axes with
  | None ->
      reshape_or_id
        (Array.of_list (List.filter (( <> ) 1) (Array.to_list sh)))
  | Some axes_list ->
      let normalized =
        List.map (fun ax -> if ax < 0 then ax + r else ax) axes_list
      in
      let seen = Array.make r false in
      List.iter
        (fun ax ->
          if ax < 0 || ax >= r then
            err "squeeze" "axis %d out of bounds for %dD tensor" ax r;
          if seen.(ax) then err "squeeze" "axis %d, duplicate axis" ax;
          seen.(ax) <- true)
        normalized;
      List.iter
        (fun ax ->
          if sh.(ax) <> 1 then
            err "squeeze"
              "cannot remove dimension at axis %d (size %d), size %d≠1" ax
              sh.(ax) sh.(ax))
        normalized;
      let axes_set =
        List.fold_left (fun s ax -> IntSet.add ax s) IntSet.empty normalized
      in
      reshape_or_id
        (Array.of_list
           (List.filteri
              (fun i _ -> not (IntSet.mem i axes_set))
              (Array.to_list sh)))

let unsqueeze ?axes x =
  let sh = shape x in
  let r = Array.length sh in
  let axes_list =
    match axes with
    | None -> invalid_arg "unsqueeze: axes must be specified"
    | Some lst -> lst
  in
  if List.length axes_list = 0 then x
  else
    let output_rank = r + List.length axes_list in
    let normalized =
      List.map (fun ax -> if ax < 0 then ax + output_rank else ax) axes_list
    in
    let seen = Array.make output_rank false in
    List.iter
      (fun ax ->
        if ax < 0 || ax >= output_rank then
          err "unsqueeze"
            "axis %d, out of bounds for output rank %d, valid range is [%d, \
             %d)"
            ax output_rank (-output_rank) output_rank;
        if seen.(ax) then err "unsqueeze" "axis %d, duplicate axis" ax;
        seen.(ax) <- true)
      normalized;
    let axes_set =
      List.fold_left (fun s ax -> IntSet.add ax s) IntSet.empty normalized
    in
    let new_shape = ref [] in
    let input_idx = ref 0 in
    for output_idx = 0 to output_rank - 1 do
      if IntSet.mem output_idx axes_set then new_shape := 1 :: !new_shape
      else if !input_idx < r then begin
        new_shape := sh.(!input_idx) :: !new_shape;
        incr input_idx
      end
    done;
    reshape (Array.of_list (List.rev !new_shape)) x

let expand_dims axes x = unsqueeze ~axes x

let transpose ?axes x =
  let r = ndim x in
  let resolved =
    match axes with
    | None -> Array.init r (fun i -> r - 1 - i)
    | Some ax_list ->
        if List.length ax_list <> r then
          err "transpose"
            "axes (length %d), expected rank %d, got %d, provide exactly one \
             axis per dimension"
            (List.length ax_list) r (List.length ax_list);
        let seen = Array.make r false in
        List.iter
          (fun ax_val ->
            let ax = if ax_val < 0 then ax_val + r else ax_val in
            if ax < 0 || ax >= r then
              err "transpose" "axis %d out of bounds for %dD tensor" ax_val r;
            if seen.(ax) then err "transpose" "axis %d, repeated" ax_val;
            seen.(ax) <- true)
          ax_list;
        if not (Array.for_all Fun.id seen) then
          invalid_arg "transpose: axes do not form a permutation";
        Array.of_list (List.map (fun v -> if v < 0 then v + r else v) ax_list)
  in
  B.permute x resolved

let flip ?axes x =
  let r = ndim x in
  let flip_bools = Array.make r false in
  (match axes with
  | None -> Array.fill flip_bools 0 r true
  | Some ax_list ->
      List.iter
        (fun ax_val ->
          let ax = if ax_val < 0 then ax_val + r else ax_val in
          if ax < 0 || ax >= r then
            err "flip" "axis %d out of bounds for %dD tensor" ax_val r;
          flip_bools.(ax) <- true)
        ax_list);
  B.flip x flip_bools

let moveaxis src dst x =
  let r = ndim x in
  let s = if src < 0 then src + r else src in
  let d = if dst < 0 then dst + r else dst in
  if s < 0 || s >= r || d < 0 || d >= r then
    err "moveaxis" "source %d or destination %d, out of bounds for shape %s"
      src dst
      (Shape.to_string (shape x));
  if s = d then x
  else
    let axes = Array.to_list (Array.init r Fun.id) in
    let without = List.filter (( <> ) s) axes in
    let rec insert_at idx item = function
      | [] -> [ item ]
      | hd :: tl ->
          if idx = 0 then item :: hd :: tl
          else hd :: insert_at (idx - 1) item tl
    in
    B.permute x (Array.of_list (insert_at d s without))

let swapaxes axis1 axis2 x =
  let r = ndim x in
  let a1 = if axis1 < 0 then axis1 + r else axis1 in
  let a2 = if axis2 < 0 then axis2 + r else axis2 in
  if a1 < 0 || a1 >= r || a2 < 0 || a2 >= r then
    err "swapaxes" "axes (%d, %d), out of bounds for shape %s" axis1 axis2
      (Shape.to_string (shape x));
  if a1 = a2 then x
  else
    let axes = Array.init r Fun.id in
    axes.(a1) <- a2;
    axes.(a2) <- a1;
    B.permute x axes

let cat_tensors ~axis tensors =
  match tensors with
  | [] ->
      invalid_arg
        "concatenate: tensor list cannot be empty, provide at least one \
         tensor"
  | _ -> B.cat ~axis tensors

let roll ?axis shift x =
  let original_shape = shape x in
  let x, ax_idx =
    match axis with
    | None -> (flatten x, 0)
    | Some a ->
        let r = ndim x in
        let norm = if a < 0 then a + r else a in
        if norm < 0 || norm >= r then
          err "roll" "axis %d out of bounds for %dD tensor" a r;
        (x, norm)
  in
  let sh = shape x in
  let rolled =
    if ndim x = 0 || sh.(ax_idx) = 0 then x
    else
      let dim_size = sh.(ax_idx) in
      let actual = ((shift mod dim_size) + dim_size) mod dim_size in
      if actual = 0 then x
      else
        let ranges_p1 =
          Array.mapi
            (fun i d -> if i = ax_idx then (dim_size - actual, d) else (0, d))
            sh
        in
        let ranges_p2 =
          Array.mapi
            (fun i d -> if i = ax_idx then (0, dim_size - actual) else (0, d))
            sh
        in
        cat_tensors ~axis:ax_idx [ shrink ranges_p1 x; shrink ranges_p2 x ]
  in
  if axis = None then reshape original_shape rolled else rolled

let tile reps x =
  let t_shape = shape x in
  let t_ndim = ndim x in
  let reps_len = Array.length reps in
  if reps_len < t_ndim then
    invalid_arg "tile: reps length must be >= tensor rank";
  let x_promoted, promoted_shape =
    if reps_len > t_ndim then (
      let new_shape = Array.make reps_len 1 in
      Array.blit t_shape 0 new_shape (reps_len - t_ndim) t_ndim;
      (reshape new_shape x, new_shape))
    else (x, t_shape)
  in
  Array.iteri
    (fun i r ->
      if r < 0 then
        err "tile"
          "reps[%d], negative (%d<0), use positive integers (or 0 for empty \
           result)"
          i r)
    reps;
  if Array.for_all (( = ) 1) reps then B.copy x_promoted
  else if Array.exists (( = ) 0) reps || Array.exists (( = ) 0) promoted_shape
  then
    empty (Value.context x) (dtype x)
      (Array.mapi (fun i s -> s * reps.(i)) promoted_shape)
  else
    let rec tile_axis curr axis =
      if axis >= reps_len then curr
      else if reps.(axis) = 1 then tile_axis curr (axis + 1)
      else
        tile_axis
          (cat_tensors ~axis (List.init reps.(axis) (fun _ -> curr)))
          (axis + 1)
    in
    tile_axis x_promoted 0

let repeat ?axis count x =
  if count < 0 then err "repeat" "count must be >= 0, got %d" count;
  let x, ax_idx =
    match axis with
    | None -> (flatten x, 0)
    | Some a ->
        let r = ndim x in
        let norm = if a < 0 then a + r else a in
        if norm < 0 || norm >= r then
          err "repeat" "axis %d out of bounds for %dD tensor" a r;
        (x, norm)
  in
  let t_shape = shape x in
  let t_ndim = ndim x in
  if count = 0 then begin
    let s = Array.copy t_shape in
    if t_ndim > 0 then s.(ax_idx) <- 0;
    empty (Value.context x) (dtype x) (if axis = None then [| 0 |] else s)
  end
  else if count = 1 then B.copy x
  else if t_ndim = 0 then
    let repeated = expand [| count |] (reshape [| 1 |] x) in
    if axis = None then repeated else reshape (shape x) repeated
  else
    (* Each element along the axis, broadcast [count] times on a new axis
       after it, then merged into it. *)
    let wide =
      Array.init (t_ndim + 1) (fun d ->
          if d <= ax_idx then t_shape.(d)
          else if d = ax_idx + 1 then count
          else t_shape.(d - 1))
    in
    let merged = Array.copy t_shape in
    merged.(ax_idx) <- t_shape.(ax_idx) * count;
    reshape merged (expand wide (unsqueeze ~axes:[ ax_idx + 1 ] x))

(* ───── Concatenation and Stacking ───── *)

let check_dtypes_match ~op ts =
  let first_dtype = dtype (List.hd ts) in
  List.iter
    (fun x ->
      let d = dtype x in
      if not (Nx_dtype.equal first_dtype d) then
        err op "expected dtype %s, got %s"
          (Nx_dtype.to_string first_dtype)
          (Nx_dtype.to_string d))
    (List.tl ts)

let concatenate ~axis ts =
  match ts with
  | [] ->
      invalid_arg
        "concatenate: tensor list cannot be empty, provide at least one \
         tensor"
  | first :: rest -> (
      let first_ndim = ndim first in
      let axis = if axis < 0 then axis + first_ndim else axis in
      if axis < 0 || axis >= first_ndim then
        err "concatenate" "axis %d out of bounds for %dD tensor" axis
          first_ndim;
      match rest with
      | [] -> copy first
      | _ ->
          check_dtypes_match ~op:"concatenate" ts;
          if not (List.for_all (fun x -> ndim x = first_ndim) ts) then
            invalid_arg
              "concatenate: arrays must have same number of dimensions";
          let first_shape = shape first in
          List.iter
            (fun x ->
              let s = shape x in
              Array.iteri
                (fun i d ->
                  if i <> axis && d <> first_shape.(i) then
                    err "concatenate" "dimension %d, size %d≠%d" i d
                      first_shape.(i))
                s)
            rest;
          cat_tensors ~axis ts)

let stack ?axis ts =
  match ts with
  | [] -> invalid_arg "stack: tensor list cannot be empty"
  | _ ->
      let first_ndim = Array.length (shape (List.hd ts)) in
      let axis =
        match axis with
        | None -> 0
        | Some a ->
            let a = if a < 0 then a + first_ndim + 1 else a in
            if a < 0 || a > first_ndim then
              err "stack" "axis %d out of bounds for %dD tensor" a first_ndim;
            a
      in
      concatenate ~axis (List.map (fun x -> unsqueeze ~axes:[ axis ] x) ts)

let ensure_ndim n x =
  let s = shape x in
  let nd = Array.length s in
  if nd >= n then x
  else
    let new_shape = Array.make n 1 in
    Array.blit s 0 new_shape 0 nd;
    reshape new_shape x

let broadcast_arrays ts =
  match ts with
  | [] -> []
  | [ x ] -> [ x ]
  | _ ->
      let target =
        List.fold_left
          (fun acc x -> Shape.broadcast acc (shape x))
          (shape (List.hd ts))
          (List.tl ts)
      in
      List.map (fun x -> broadcast_to target x) ts

(* ───── Array Creation ───── *)

let eye ctx ?m ?k dtype n =
  let cols = Option.value m ~default:n and k = Option.value k ~default:0 in
  check_shape "eye" [| n; cols |];
  let arr = Array.make (n * cols) (Nx_dtype.zero dtype) in
  let one = Nx_dtype.one dtype in
  for i = 0 to n - 1 do
    let j = i + k in
    if j >= 0 && j < cols then arr.((i * cols) + j) <- one
  done;
  create ctx dtype [| n; cols |] arr

(* The integers [dtype] holds: its range for an integer dtype, 0 and 1 for
   [bool], and for a float or complex dtype those whose magnitude is at most its
   largest finite value. Bounds beyond OCaml's ints are clipped to them. *)
let held_integers (type a b) (dtype : (a, b) Nx_dtype.t) =
  let finite dt =
    let m = Float.to_int (Nx_dtype.max_finite dt) in
    (-m, m)
  in
  match dtype with
  | Nx_dtype.Bool -> (0, 1)
  | Nx_dtype.Bit -> (0, 1)
  | Nx_dtype.Int4 -> (-8, 7)
  | Nx_dtype.UInt4 -> (0, 15)
  | Nx_dtype.Int8 -> (-128, 127)
  | Nx_dtype.UInt8 -> (0, 255)
  | Nx_dtype.Int16 -> (-32768, 32767)
  | Nx_dtype.UInt16 -> (0, 65535)
  | Nx_dtype.Int32 -> (-0x8000_0000, 0x7fff_ffff)
  | Nx_dtype.UInt32 -> (0, 0xffff_ffff)
  | Nx_dtype.UInt64 -> (0, max_int)
  | Nx_dtype.Float16 -> finite Nx_dtype.float16
  | Nx_dtype.Float8_e4m3 -> finite Nx_dtype.float8_e4m3
  | Nx_dtype.Float8_e5m2 -> finite Nx_dtype.float8_e5m2
  | Nx_dtype.Int64 | Nx_dtype.BFloat16 | Nx_dtype.Float32 | Nx_dtype.Float64
  | Nx_dtype.Complex64 | Nx_dtype.Complex128 ->
      (min_int, max_int)

(* The count of the values [start + i * step] on [start]'s side of [stop]. In
   int64, [stop - start] does not overflow. *)
let arange_length start stop step =
  if (step > 0 && start >= stop) || (step < 0 && start <= stop) then 0
  else
    let span = Int64.(abs (sub (of_int stop) (of_int start))) in
    let n = Int64.(succ (div (pred span) (abs (of_int step)))) in
    if Int64.compare n (Int64.of_int max_int) > 0 then
      err "arange" "%d to %d by %d, more than max_int values" start stop step;
    Int64.to_int n

(* [arange_int64 ctx n start step] is the [n] values [start + i * step] in
   int64. Up to [arange_run] values, they are [start - step] plus the running
   sum of copies of [step]. Past it, they are rows of [arange_run] values: a
   column of the rows' starts plus a row of offsets, one elementwise pass over
   two short ranges. A partial sum leaves int64 only where the values span more
   than 2{^62}; int64 arithmetic wraps (nx.cpu builds with -fwrapv), so the
   values are still exact. *)
let arange_run = 1024

let rec arange_int64 ctx n start step =
  let int64 v = scalar ctx Nx_dtype.int64 v in
  if n <= arange_run then
    add
      (cumsum ~axis:0 (broadcast_to [| n |] (int64 step)))
      (int64 (Int64.sub start step))
  else
    let b = arange_run in
    let a = (n + b - 1) / b in
    let starts = arange_int64 ctx a start (Int64.mul step (Int64.of_int b)) in
    let offsets = arange_int64 ctx b 0L step in
    shrink
      [| (0, n) |]
      (reshape [| a * b |]
         (add (reshape [| a; 1 |] starts) (reshape [| 1; b |] offsets)))

let arange (type a b) ctx (dtype : (a, b) Nx_dtype.t) start stop step : (a, b) t
    =
  if step = 0 then invalid_arg "arange: step cannot be zero";
  let n = arange_length start stop step in
  if n = 0 then empty ctx dtype [| 0 |]
  else
    (* The last value lies between [start] and [stop], so the wrapping int
       arithmetic computes it exactly. *)
    let last = start + ((n - 1) * step) in
    let lo = Int.min start last and hi = Int.max start last in
    let least, greatest = held_integers dtype in
    if lo < least || hi > greatest then
      err "arange" "values %d to %d, %s holds %d to %d" lo hi
        (Nx_dtype.to_string dtype) least greatest;
    cast dtype (arange_int64 ctx n (Int64.of_int start) (Int64.of_int step))

let arange_f ctx dtype start_f stop_f step_f =
  if step_f = 0. then invalid_arg "arange_f: step cannot be zero";
  let num_exact_steps = (stop_f -. start_f) /. step_f in
  let eps = 1e-9 in
  let num_elements =
    if
      (step_f > 0. && stop_f <= start_f +. (eps *. Float.abs step_f))
      || (step_f < 0. && stop_f >= start_f +. (eps *. Float.abs step_f))
      || (Float.abs num_exact_steps < eps && num_exact_steps <= 0.)
    then 0
    else
      let corrected =
        num_exact_steps -. Float.copy_sign eps num_exact_steps
      in
      int_of_float (Float.floor corrected +. 1.)
  in
  let n = Stdlib.max 0 num_elements in
  if n <= 0 then empty ctx dtype [| 0 |]
  else
    init ctx dtype [| n |] (fun idx ->
        start_f +. (float_of_int idx.(0) *. step_f))

let linspace ctx dtype ?(endpoint = true) start_f stop_f count =
  if count < 0 then
    err "linspace" "count %d, negative count, use count >= 0" count;
  if count = 0 then empty ctx dtype [| 0 |]
  else if count = 1 then
    full ctx dtype [| 1 |] (Nx_dtype.of_float dtype start_f)
  else
    (* [stop] is point [last], past the points when [endpoint] is false. *)
    let last = if endpoint then count - 1 else count in
    let span = float_of_int last in
    (* [stop - start] can overflow where its share of a step does not. *)
    let step =
      let s = (stop_f -. start_f) /. span in
      if Float.is_finite s then s else (stop_f /. span) -. (start_f /. span)
    in
    (* Each point counts from the nearer end, so both ends are exact. *)
    let point i =
      if i = 0 then start_f
      else if i = last then stop_f
      else if 2 * i < last then start_f +. (float_of_int i *. step)
      else stop_f -. (float_of_int (last - i) *. step)
    in
    init ctx dtype [| count |] (fun idx ->
        Nx_dtype.of_float dtype (point idx.(0)))

let logspace ctx dtype ?(endpoint = true) ?(base = 10.0) start_exp stop_exp
    count =
  if count < 0 then err "logspace" "count must be >= 0, got %d" count;
  if count = 0 then empty ctx dtype [| 0 |]
  else
    let exponents = linspace ctx dtype ~endpoint start_exp stop_exp count in
    if base = Float.exp 1.0 then exp exponents
    else
      let log2_base = Stdlib.log base /. Stdlib.log 2.0 in
      let log2_base_t =
        broadcast_to (shape exponents) (scalar ctx dtype log2_base)
      in
      exp2 (mul exponents log2_base_t)

let geomspace ctx dtype ?(endpoint = true) start_f stop_f count =
  if start_f <= 0. || stop_f <= 0. then
    err "geomspace" "start %s and stop %s, both must be positive"
      (float_text Float64 start_f) (float_text Float64 stop_f);
  if count < 0 then err "geomspace" "count must be >= 0, got %d" count;
  if count = 0 then empty ctx dtype [| 0 |]
  else if count = 1 then full ctx dtype [| 1 |] start_f
  else
    exp
      (linspace ctx dtype ~endpoint (Stdlib.log start_f) (Stdlib.log stop_f)
         count)

let meshgrid ?(indexing = `xy) x y =
  let x_shape = shape x in
  let y_shape = shape y in
  if Array.length x_shape <> 1 then invalid_arg "meshgrid: x must be 1D";
  if Array.length y_shape <> 1 then invalid_arg "meshgrid: y must be 1D";
  let nx = x_shape.(0) in
  let ny = y_shape.(0) in
  match indexing with
  | `xy ->
      ( broadcast_to [| ny; nx |] (reshape [| 1; nx |] x),
        broadcast_to [| ny; nx |] (reshape [| ny; 1 |] y) )
  | `ij ->
      ( broadcast_to [| nx; ny |] (reshape [| nx; 1 |] x),
        broadcast_to [| nx; ny |] (reshape [| 1; ny |] y) )

(* Triangular mask: tril uses (>=), triu uses (<=) *)
let triangular_mask ~op ~cmp ?k x =
  let k_val = match k with Some v -> v | None -> 0 in
  let sh = shape x in
  let nd = Array.length sh in
  if nd < 2 then err op "input requires at least 2D tensor";
  let rows = sh.(nd - 2) in
  let cols = sh.(nd - 1) in
  let row_idx =
    reshape [| rows; 1 |] (arange (Value.context x) int64 0 rows 1)
  in
  let col_idx =
    reshape [| 1; cols |] (arange (Value.context x) int64 0 cols 1)
  in
  let k_offset =
    sub col_idx (scalar (Value.context x) int64 (Int64.of_int k_val))
  in
  let mask = cmp row_idx k_offset in
  where mask x (scalar_like x (Nx_dtype.zero (dtype x)))

let tril ?k x = triangular_mask ~op:"tril" ~cmp:greater_equal ?k x
let triu ?k x = triangular_mask ~op:"triu" ~cmp:less_equal ?k x

(* ───── Take Operations ───── *)

let take ?axis ~indices t =
  match axis with
  | None ->
      let flat = reshape [| numel indices |] indices in
      reshape (shape indices) (B.gather ~axis:0 flat (flatten t))
  | Some axis ->
      let t_shape = shape t in
      let axis = resolve_single_axis t axis in
      let idx = indices in
      let n_idx = numel idx in
      (* Reshape indices for broadcasting: [1,...,1,n_idx,1,...,1] *)
      let expanded_shape =
        Array.init (Array.length t_shape) (fun i ->
            if i = axis then n_idx else 1)
      in
      let broadcast_shape = Array.copy t_shape in
      broadcast_shape.(axis) <- n_idx;
      let idx_broadcast =
        broadcast_to broadcast_shape (reshape expanded_shape idx)
      in
      let out = B.gather ~axis idx_broadcast t in
      let out_shape = Array.copy t_shape in
      out_shape.(axis) <- n_idx;
      reshape out_shape out

let take_along_axis ~axis ~indices t =
  let axis = resolve_single_axis t axis in
  let t_shape = shape t in
  let idx_shape = shape indices in
  if Array.length t_shape <> Array.length idx_shape then
    err "take_along_axis" "cannot reshape %s to %s"
      (Shape.to_string idx_shape)
      (Shape.to_string t_shape);
  Array.iteri
    (fun i dim ->
      if i <> axis && dim <> idx_shape.(i) then
        err "take_along_axis"
          "shape, dimension %d: indices has %d but tensor has %d" i
          idx_shape.(i) dim)
    t_shape;
  B.gather ~axis indices t

(* ───── Counting ───── *)

(* The set bits of each byte. *)
let popcounts ctx =
  let rec bits n =
    if n = 0 then 0L else Int64.add (Int64.of_int (n land 1)) (bits (n lsr 1))
  in
  create ctx Int64 [| 256 |] (Array.init 256 bits)

(* A [bit] mask counts its whole bytes by a table of their set bits, then its
   last elements, fewer than 8, as booleans. *)
let count_bits (m : bit_t) : int64_t =
  let flat = reshape [| numel m |] m in
  let n = numel flat in
  let whole = n - (n mod 8) in
  let tail = sum (cast Int64 (cast Bool (shrink [| (whole, n) |] flat))) in
  if whole = 0 then tail
  else
    let bytes =
      bitcast UInt8 (reshape [| whole / 8; 8 |] (shrink [| (0, whole) |] flat))
    in
    let ones = take ~indices:(cast Int64 bytes) (popcounts (Value.context m)) in
    add (sum ones) tail

let count (type b) ?axes ?(keepdims = false) (m : (bool, b) t) : int64_t =
  match dtype m with
  | Bool -> sum ?axes ~keepdims (cast Int64 m)
  | Bit ->
      let rank = ndim m in
      let reduced =
        match axes with
        | None -> rank
        | Some axes ->
            List.length (normalize_and_dedup_axes ~op:"count" rank axes)
      in
      if reduced < rank then sum ?axes ~keepdims (cast Int64 (cast Bool m))
      else
        let c = count_bits m in
        if keepdims then reshape (Array.make rank 1) c else c

(* ───── Indexing and Slicing ───── *)

let normalize_index dim_size idx = if idx < 0 then dim_size + idx else idx

let normalize_and_check_index ~op dim_size idx =
  let idx' = if idx < 0 then dim_size + idx else idx in
  if idx' < 0 || idx' >= dim_size then
    err op "index %d out of bounds [0, %d)" idx dim_size;
  idx'

type dim_op =
  | View of { start : int; stop : int; step : int; dim_len : int }
  | Squeeze of { idx : int }
  | Gather of int array
  | New_axis
  | Window of { start : (int64, Nx_dtype.int64_elt) t; len : int }

let normalize_slice_spec ~by ~axis dim_size = function
  | I idx ->
      Squeeze { idx = normalize_and_check_index ~op:"slice" dim_size idx }
  | A -> View { start = 0; stop = dim_size; step = 1; dim_len = dim_size }
  | R (start, stop) ->
      let s = Int.max 0 (Int.min (normalize_index dim_size start) dim_size) in
      let e = Int.max 0 (Int.min (normalize_index dim_size stop) dim_size) in
      View { start = s; stop = e; step = 1; dim_len = Int.max 0 (e - s) }
  | Rs (start, stop, step) ->
      if step = 0 then
        invalid_arg
          "slice: step cannot be zero, use positive step for forward slicing \
           or negative for reverse";
      (* Both bounds clamp into the axis, as a Python slice's do. *)
      let lo, hi = if step > 0 then (0, dim_size) else (-1, dim_size - 1) in
      let clamp i = Int.max lo (Int.min hi (normalize_index dim_size i)) in
      let s = clamp start and e = clamp stop in
      let len =
        if step > 0 then if s >= e then 0 else ((e - 1 - s) / step) + 1
        else if s <= e then 0
        else ((s - e - 1) / -step) + 1
      in
      View { start = s; stop = e; step; dim_len = len }
  | L indices ->
      Gather
        (Array.map
           (normalize_and_check_index ~op:"slice" dim_size)
           (Array.of_list indices))
  | N -> New_axis
  | M mask ->
      if ndim mask <> 1 then
        err "slice" "axis %d, boolean mask must be rank 1 but has rank %d"
          axis (ndim mask);
      let mask_len = numel mask in
      if mask_len <> dim_size then
        err "slice" "axis %d, boolean mask length %d, expected %d" axis
          mask_len dim_size;
      let bits = read_array ~by mask in
      let positions = ref [] in
      for i = mask_len - 1 downto 0 do
        if bits.(i) then positions := i :: !positions
      done;
      Gather (Array.of_list !positions)
  | D (start, len) ->
      if numel start <> 1 then
        err "slice" "axis %d, window start must be a scalar tensor" axis;
      if len < 0 || len > dim_size then
        err "slice" "axis %d, window of %d does not fit in %d" axis len
          dim_size;
      (* clamp the corner so the window always fits; a tensor operation, so a
         traced start stays traced *)
      let ctx = Value.context start in
      let start = reshape [||] start in
      let start =
        minimum
          (maximum start (scalar ctx Nx_dtype.int64 0L))
          (scalar ctx Nx_dtype.int64 (Int64.of_int (dim_size - len)))
      in
      Window { start; len }

(* Parse specs into one op per input axis, [New_axis] entries interleaved,
   padding unspecified trailing axes with [A]. *)
let parse_specs ~by specs input_shape =
  let ndim_in = Array.length input_shape in
  let ops, consumed =
    List.fold_left
      (fun (acc, dim) spec ->
        match spec with
        | N -> (New_axis :: acc, dim)
        | _ ->
            if dim >= ndim_in then invalid_arg "slice: too many indices";
            ( normalize_slice_spec ~by ~axis:dim input_shape.(dim) spec :: acc,
              dim + 1 ))
      ([], 0) specs
  in
  let rec pad_trailing acc dim =
    if dim >= ndim_in then List.rev acc
    else
      pad_trailing
        (normalize_slice_spec ~by ~axis:dim input_shape.(dim) A :: acc)
        (dim + 1)
  in
  pad_trailing ops consumed

let slice specs x =
  let ops = parse_specs ~by:"Nx.slice" specs (shape x) in
  let gather_axis axis indices t =
    let idx_t =
      create (Value.context t) Nx_dtype.int64
        [| Array.length indices |]
        (Array.map Int64.of_int indices)
    in
    take ~axis ~indices:idx_t t
  in
  let shrink_axis axis start stop t =
    B.shrink t
      (Array.mapi
         (fun i dim -> if i = axis then (start, stop) else (0, dim))
         (shape t))
  in
  let rec apply current axis sq_axes = function
    | [] -> (current, sq_axes)
    | New_axis :: rest ->
        apply (unsqueeze ~axes:[ axis ] current) (axis + 1) sq_axes rest
    | Squeeze { idx } :: rest ->
        apply
          (shrink_axis axis idx (idx + 1) current)
          (axis + 1) (axis :: sq_axes) rest
    | Gather indices :: rest ->
        apply (gather_axis axis indices current) (axis + 1) sq_axes rest
    | Window { start; len } :: rest ->
        let current' =
          if len = 0 then shrink_axis axis 0 0 current
          else
            let idx =
              add (arange (Value.context current) Nx_dtype.int64 0 len 1) start
            in
            take ~axis ~indices:idx current
        in
        apply current' (axis + 1) sq_axes rest
    | View { start; step; dim_len; _ } :: rest ->
        let current' =
          if step = 1 then shrink_axis axis start (start + dim_len) current
          else if step = -1 then (
            if dim_len = 0 then shrink_axis axis 0 0 current
            else
              let sliced =
                shrink_axis axis (start - dim_len + 1) (start + 1) current
              in
              let fb = Array.make (ndim sliced) false in
              fb.(axis) <- true;
              B.flip sliced fb)
          else
            gather_axis axis
              (Array.init dim_len (fun i -> start + (i * step)))
              current
        in
        apply current' (axis + 1) sq_axes rest
  in
  let result, sq_axes = apply x 0 [] ops in
  match List.sort_uniq compare sq_axes with
  | [] -> result
  | axes -> squeeze ~axes result

let get indices x =
  let x_shape = shape x in
  let checked =
    List.mapi
      (fun dim idx ->
        if dim >= Array.length x_shape then
          err "get" "indices, too many for shape %s" (Shape.to_string x_shape);
        let idx' = normalize_index x_shape.(dim) idx in
        if idx' < 0 || idx' >= x_shape.(dim) then
          err "get"
            "index [%s] out of bounds for shape %s, index %d at dim %d: %d \
             not in [0, %d)"
            (String.concat "," (List.map string_of_int indices))
            (Shape.to_string x_shape) dim dim idx' x_shape.(dim);
        idx')
      indices
  in
  slice (List.map (fun i -> I i) checked) x

let item indices t =
  let s = shape t in
  if List.length indices <> Array.length s then
    invalid_arg
      (Printf.sprintf "item: need %d indices for %d-d tensor, got %d"
         (Array.length s) (Array.length s) (List.length indices));
  read_item ~by:"Nx.item" (get indices t)

let scatter ?(mode = `Set) ?(unique_indices = false) ~axis ~indices ~values t
    =
  let axis = resolve_single_axis t axis in
  let t_shape = shape t in
  let idx_shape = shape indices in
  if Array.length t_shape <> Array.length idx_shape then
    err "scatter" "cannot reshape %s to %s"
      (Shape.to_string idx_shape)
      (Shape.to_string t_shape);
  Array.iteri
    (fun i dim ->
      if i <> axis && dim <> idx_shape.(i) then
        err "scatter" "shape, dimension %d: indices has %d but tensor has %d"
          i idx_shape.(i) dim)
    t_shape;
  (match mode with
  | (`Max | `Min) when Nx_dtype.is_complex (dtype t) ->
      err "scatter" "complex numbers are not ordered"
  | `Set | `Add | `Max | `Min -> ());
  let values =
    if shape values = idx_shape then values else broadcast_to idx_shape values
  in
  B.scatter ~mode ~unique:unique_indices ~axis ~indices ~updates:values t

(* ───── Functional update ───── *)

(* [set specs v x] is [x] with [v], broadcast to the selection, at the
   positions [specs] select. One value-carrying operation on [x], chosen from
   the spec syntax: a mask alone is a [where]; a window (single indices,
   unit-step ranges, run-time runs, whole axes) is a backend [update]; any
   gather (a list, a stepped range, a mask beside other specs) is a flat
   [scatter] over a contiguous copy. *)
let set specs v x =
  let x_shape = shape x in
  let nd = Array.length x_shape in
  let ctx = Value.context x in
  let specs_full =
    let consumed = List.length (List.filter (fun s -> s <> N) specs) in
    if consumed > nd then invalid_arg "set: too many indices";
    specs @ List.init (nd - consumed) (fun _ -> A)
  in
  let mask_alone =
    match List.filter (fun s -> s <> A) specs_full with
    | [ M mask ] ->
        let k =
          let rec find i = function
            | M _ :: _ -> i
            | _ :: rest -> find (i + 1) rest
            | [] -> assert false
          in
          find 0 specs_full
        in
        if ndim mask <> 1 then
          err "set" "axis %d, boolean mask must be rank 1 but has rank %d" k
            (ndim mask);
        if numel mask <> x_shape.(k) then
          err "set" "axis %d, boolean mask length %d, expected %d" k
            (numel mask) x_shape.(k);
        (* [v] must broadcast against [x] with no extent on the mask axis;
           otherwise it is selection-shaped and goes through the gather. *)
        let vr = ndim v in
        let vk = k - (nd - vr) in
        if vk < 0 || (shape v).(vk) = 1 then Some (k, mask) else None
    | _ -> None
  in
  match mask_alone with
  | Some (k, mask) ->
      let mshape = Array.make nd 1 in
      mshape.(k) <- x_shape.(k);
      where
        (broadcast_to x_shape (reshape mshape mask))
        (broadcast_to x_shape v) x
  | None ->
      let ops = parse_specs ~by:"Nx.set" specs_full x_shape in
      let sel_shape =
        Array.of_list
          (List.filter_map
             (function
               | New_axis -> Some 1
               | Squeeze _ -> None
               | Gather idx -> Some (Array.length idx)
               | View { dim_len; _ } -> Some dim_len
               | Window { len; _ } -> Some len)
             ops)
      in
      let v = broadcast_to sel_shape v in
      let axis_ops = List.filter (fun op -> op <> New_axis) ops in
      let is_window =
        List.for_all
          (function
            | Squeeze _ | Window _ -> true
            | View { step; _ } -> step = 1 || step = -1
            | Gather _ | New_axis -> false)
          axis_ops
      in
      if array_prod sel_shape = 0 then x
      else if is_window then begin
        let window_shape =
          Array.of_list
            (List.map
               (function
                 | Squeeze _ -> 1
                 | View { dim_len; _ } -> dim_len
                 | Window { len; _ } -> len
                 | Gather _ | New_axis -> assert false)
               axis_ops)
        in
        let corners =
          List.map
            (function
              | Squeeze { idx } -> `Int idx
              | View { start; step; dim_len; _ } ->
                  `Int (if step = 1 then start else start - dim_len + 1)
              | Window { start; _ } -> `Tensor start
              | Gather _ | New_axis -> assert false)
            axis_ops
        in
        let starts =
          if
            List.for_all
              (function `Int _ -> true | `Tensor _ -> false)
              corners
          then
            create ctx Nx_dtype.int64 [| nd |]
              (Array.of_list
                 (List.map
                    (function
                      | `Int i -> Int64.of_int i | `Tensor _ -> assert false)
                    corners))
          else
            stack ~axis:0
              (List.map
                 (function
                   | `Int i -> scalar ctx Nx_dtype.int64 (Int64.of_int i)
                   | `Tensor s -> s)
                 corners)
        in
        let v = reshape window_shape v in
        let flips =
          Array.of_list
            (List.map
               (function View { step = -1; _ } -> true | _ -> false)
               axis_ops)
        in
        let v = if Array.exists Fun.id flips then B.flip v flips else v in
        B.update x ~starts v
      end
      else begin
        let strides = Shape.c_contiguous_strides x_shape in
        let dims_info =
          List.map
            (function
              | Squeeze { idx } ->
                  (true, scalar ctx Nx_dtype.int64 (Int64.of_int idx))
              | View { start; stop; step; _ } ->
                  (false, arange ctx Nx_dtype.int64 start stop step)
              | Gather indices ->
                  let seen = Hashtbl.create (Array.length indices) in
                  Array.iter
                    (fun i ->
                      if Hashtbl.mem seen i then
                        err "set" "index %d is listed twice" i;
                      Hashtbl.replace seen i ())
                    indices;
                  ( false,
                    create ctx Nx_dtype.int64
                      [| Array.length indices |]
                      (Array.map Int64.of_int indices) )
              | Window { start; len } ->
                  (false, add (arange ctx Nx_dtype.int64 0 len 1) start)
              | New_axis -> assert false)
            axis_ops
        in
        let target_shape =
          Array.of_list
            (List.filter_map
               (fun (sq, t) -> if sq then None else Some (numel t))
               dims_info)
        in
        let target_rank = Array.length target_shape in
        let flat_idx = ref (scalar ctx Nx_dtype.int64 0L) in
        let tdim = ref 0 in
        List.iteri
          (fun i (squeezed, idx_t) ->
            let stride = Int64.of_int strides.(i) in
            let weighted =
              if stride = 1L then idx_t
              else mul idx_t (scalar ctx Nx_dtype.int64 stride)
            in
            if squeezed then flat_idx := add !flat_idx weighted
            else begin
              let rs = Array.make target_rank 1 in
              rs.(!tdim) <- numel idx_t;
              flat_idx := add !flat_idx (reshape rs weighted);
              incr tdim
            end)
          dims_info;
        let x_flat = reshape [| numel x |] x in
        let y_flat =
          reshape [| array_prod target_shape |] (reshape target_shape v)
        in
        let result =
          B.scatter ~mode:`Set ~unique:true x_flat
            ~indices:(reshape [| numel !flat_idx |] !flat_idx)
            ~updates:y_flat ~axis:0
        in
        reshape x_shape result
      end

(* Lengths that depend on values *)

(* [positions_of ~by c] is [positions c] for boolean or integer counts [c], its
   length read by the surface function [by]. A boolean reads its total. Integer counts also read their least count
   and their least running total: with every count in [0, 2^63), the first
   running total past int64's range is negative, so the two catch a negative
   count and a sum that wraps. *)
let positions_of (type a b) ~by (c : (a, b) t) : int64_t =
  let dt = dtype c in
  if ndim c <> 1 then
    err "positions" "counts of shape %s, not 1-D" (Shape.to_string (shape c));
  (match dt with
  | Bool -> ()
  | _ when Nx_dtype.is_int dt -> ()
  | _ ->
      err "positions" "counts of dtype %s, not boolean or integer"
        (Nx_dtype.to_string dt));
  let ctx = Value.context c and n = dim 0 c in
  if n = 0 then empty ctx Int64 [| 0 |]
  else
    let counts = cast Int64 c in
    let ends = cumsum counts in
    let last = shrink [| (n - 1, n) |] ends in
    let total =
      match dt with
      | Bool -> (read_array ~by last).(0)
      | _ ->
          let least x = reshape [| 1 |] (min x) in
          let read =
            read_array ~by
              (concatenate ~axis:0 [ last; least counts; least ends ])
          in
          (if read.(1) < 0L then
             match dt with
             | UInt64 -> err "positions" "a count is past int64's range"
             | _ -> err "positions" "count %Ld is negative" read.(1));
          if read.(2) < 0L then
            err "positions" "the counts sum past int64's range";
          read.(0)
    in
    if total = 0L then empty ctx Int64 [| 0 |]
    else
      (* Index [i] with a positive count lands at its run's start; every other
         index lands at [total], outside the result, and is dropped, so no two
         updates meet. A run longer than one is filled by the running
         maximum. *)
      let positive : bool_t =
        match dt with Bool -> c | _ -> greater_s counts 0L
      in
      let at = where positive (sub ends counts) (scalar ctx Int64 total) in
      let placed =
        scatter ~unique_indices:true ~axis:0 ~indices:at
          ~values:(arange ctx Int64 0 n 1)
          (zeros ctx Int64 [| Int64.to_int total |])
      in
      match dt with Bool -> placed | _ -> cummax placed

(* A [bit] mask reads its positions as [bool]. *)
let positions' (type a b) ~by (c : (a, b) t) : int64_t =
  match dtype c with
  | Bit -> positions_of ~by (cast Bool c)
  | _ -> positions_of ~by c

let positions c = positions' ~by:"Nx.positions" c

let compress ?axis ~condition t =
  if ndim condition <> 1 then
    err "compress" "condition of shape %s, not 1-D"
      (Shape.to_string (shape condition));
  let n =
    match axis with
    | None -> numel t
    | Some a -> dim (resolve_single_axis t a) t
  in
  if dim 0 condition <> n then
    err "compress" "condition of %d elements for %d" (dim 0 condition) n;
  take ?axis ~indices:(positions' ~by:"Nx.compress" condition) t

let extract ~condition t =
  if numel condition <> numel t then
    err "extract" "condition of %d elements, tensor of %d" (numel condition)
      (numel t);
  take ~indices:(positions' ~by:"Nx.extract" (flatten condition)) t

(* The flat positions, in C order, of [t]'s non-zero elements. *)
let flat_nonzero (type a b) ~by (t : (a, b) t) =
  let mask : bool_t =
    match dtype t with
    | Bool -> t
    | Bit -> cast Bool t
    | _ -> not_equal t (zeros_like t)
  in
  positions' ~by (flatten mask)

(* The coordinates in [shape] of the flat positions [p], one tensor per axis. *)
let coordinates shape p =
  let r = Array.length shape in
  let c = Array.make r p in
  let rest = ref p in
  for d = r - 1 downto 1 do
    let extent = Int64.of_int shape.(d) in
    c.(d) <- mod_s !rest extent;
    rest := div_s !rest extent
  done;
  c.(0) <- !rest;
  c

let nonzero t =
  if ndim t = 0 then [||]
  else coordinates (shape t) (flat_nonzero ~by:"Nx.nonzero" t)

let argwhere t =
  let by = "Nx.argwhere" in
  if ndim t = 0 then
    empty (Value.context t) Int64 [| dim 0 (flat_nonzero ~by t); 0 |]
  else
    stack ~axis:1 (Array.to_list (coordinates (shape t) (flat_nonzero ~by t)))

(* ───── Splitting ───── *)

let array_split ~axis sections x =
  let nd = ndim x in
  let axis = resolve_single_axis x axis in
  let axis_size = dim axis x in
  let make_slice start stop =
    if start < stop then
      slice (List.init nd (fun j -> if j = axis then R (start, stop) else A)) x
    else
      let s = Array.copy (shape x) in
      s.(axis) <- 0;
      empty (Value.context x) (dtype x) s
  in
  match sections with
  | `Indices indices ->
      (* An index counts from the end when negative, and clamps into the axis,
         as a range's bounds do. *)
      let idx =
        Array.of_list
          (List.map
             (fun i ->
               Int.max 0
                 (Int.min axis_size (if i < 0 then i + axis_size else i)))
             indices)
      in
      let n = Array.length idx + 1 in
      let bounds = Array.make (n + 1) 0 in
      Array.iteri (fun i v -> bounds.(i + 1) <- v) idx;
      bounds.(n) <- axis_size;
      Array.to_list
        (Array.init n (fun i -> make_slice bounds.(i) bounds.(i + 1)))
  | `Count n ->
      if n <= 0 then err "array_split" "sections must be >= 1, got %d" n;
      let base = axis_size / n in
      let rem = axis_size mod n in
      let splits = Array.make n x in
      let start = ref 0 in
      for i = 0 to n - 1 do
        let sz = base + if i < rem then 1 else 0 in
        splits.(i) <- make_slice !start (!start + sz);
        start := !start + sz
      done;
      Array.to_list splits

let split ~axis sections x =
  let axis = resolve_single_axis x axis in
  let axis_size = dim axis x in
  if sections < 1 then err "split" "sections must be >= 1, got %d" sections;
  if axis_size mod sections <> 0 then
    err "split"
      "cannot divide evenly axis %d (size %d) to %d sections, %d %% %d = %d, \
       use array_split for uneven division"
      axis axis_size sections axis_size sections (axis_size mod sections);
  array_split ~axis (`Count sections) x

(* ───── Sorting and Searching ───── *)

let sort_axis op x axis =
  let r = ndim x in
  let axis = if axis < 0 then axis + r else axis in
  if axis < 0 || axis >= r then
    err op "axis %d out of bounds for %dD tensor" axis r;
  axis

(* The sort kernels take no packed dtype: [int4], [uint4] and [bit] sort as the
   8-bit integers they widen to exactly. *)
let sort (type a b) ?(descending = false) ?(axis = -1) (x : (a, b) t) =
  if ndim x = 0 then (x, scalar (Value.context x) Nx_dtype.int64 0L)
  else
    let axis = sort_axis "sort" x axis in
    let sorted (type c d) (w : (c, d) t) =
      (B.sort ~descending ~axis w, B.argsort ~descending ~axis w)
    in
    match dtype x with
    | Int4 ->
        let v, i = sorted (cast Int8 x) in
        (cast Int4 v, i)
    | UInt4 ->
        let v, i = sorted (cast UInt8 x) in
        (cast UInt4 v, i)
    | Bit ->
        let v, i = sorted (cast UInt8 x) in
        (cast Bit v, i)
    | _ -> sorted x

let argsort (type a b) ?(descending = false) ?(axis = -1) (x : (a, b) t) =
  if ndim x = 0 then scalar (Value.context x) Nx_dtype.int64 0L
  else
    let axis = sort_axis "argsort" x axis in
    match dtype x with
    | Int4 -> B.argsort ~descending ~axis (cast Int8 x)
    | UInt4 -> B.argsort ~descending ~axis (cast UInt8 x)
    | Bit -> B.argsort ~descending ~axis (cast UInt8 x)
    | _ -> B.argsort ~descending ~axis x

(* Quantiles *)

let check_probabilities op qs =
  Array.iter
    (fun q ->
      if not (q >= 0. && q <= 1.) then
        err op "probability %s is outside [0, 1]" (float_text Float64 q))
    qs

(* [interpolate a b f] is [a + f * (b - a)] between the order statistics [a] and
   [b], and [a] itself where [f] is zero or [a] equals [b]. Narrow floats
   interpolate in float32 and round once. *)
let interpolate (type b) (a : (float, b) t) (b : (float, b) t) (f : float64_t) :
    (float, b) t =
  let exact = equal_s f 0. in
  let lerp (type c) (a : (float, c) t) (b : (float, c) t) =
    let f = cast (dtype a) f in
    where (logical_or exact (equal a b)) a (add a (mul f (sub b a)))
  in
  match dtype a with
  | Float32 | Float64 -> lerp a b
  | Float16 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      cast (dtype a) (lerp (cast Float32 a) (cast Float32 b))

let quantile ?axis qs x =
  check_probabilities "quantile" qs;
  let x, axis =
    match axis with
    | None -> (flatten x, 0)
    | Some axis -> (x, sort_axis "quantile" x axis)
  in
  let n = dim axis x in
  if n = 0 then err "quantile" "axis %d is empty: it has no quantile" axis;
  let ctx = Value.context x and k = Array.length qs in
  let sorted = B.sort ~descending:false ~axis x in
  let at = Array.map (fun q -> q *. float_of_int (n - 1)) qs in
  let lo = Array.map (fun h -> Int64.of_float (Float.floor h)) at in
  let hi =
    Array.map (fun i -> Int64.min (Int64.succ i) (Int64.of_int (n - 1))) lo
  in
  let order i =
    moveaxis axis 0 (take ~axis ~indices:(create ctx Int64 [| k |] i) sorted)
  in
  let f = Array.map (fun h -> h -. Float.floor h) at in
  let f =
    reshape
      (Array.init (ndim x) (fun d -> if d = 0 then k else 1))
      (create ctx Float64 [| k |] f)
  in
  interpolate (order lo) (order hi) f

(* The tensor and axis an arg-reduction runs along. *)
let arg_axis op ?axis x =
  let r = ndim x in
  let a =
    Option.map
      (fun a ->
        let a = resolve_single_axis ~ndim_opt:r x a in
        if a < 0 || a >= r then
          err op "axis %d out of bounds for %dD tensor" a r;
        a)
      axis
  in
  let n = match a with None -> numel x | Some a -> (shape x).(a) in
  if n = 0 then err op "an empty axis has no extreme";
  match a with None -> (flatten x, 0) | Some a -> (x, a)

(* [y], an argument reduction of [x] along [axis], with that axis kept as one
   when [keepdims]. *)
let keep_axis ~keepdims ~axis x y =
  if keepdims then
    reshape (Shape.reduce_output_shape (shape x) [| axis |] true) y
  else y

let argmax ?axis ?(keepdims = false) x =
  let x', axis = arg_axis "argmax" ?axis x in
  keep_axis ~keepdims ~axis x' (B.arg_reduce Argmax ~axis x')

let argmin ?axis ?(keepdims = false) x =
  let x', axis = arg_axis "argmin" ?axis x in
  keep_axis ~keepdims ~axis x' (B.arg_reduce Argmin ~axis x')

(* The identity of [op] on [dt], for the function [name], which refuses a sum of
   booleans and the extremes of complex numbers. *)
let identity_of (type a b) name op (dt : (a, b) dtype) : a =
  match (op, dt) with
  | `Add, (Bool | Bit) -> err name "booleans have no sum"
  | `Add, _ -> Nx_dtype.zero dt
  | (`Max | `Min), _ when Nx_dtype.is_complex dt ->
      err name "complex numbers are not ordered"
  | `Max, _ -> Nx_dtype.min_value dt
  | `Min, _ -> Nx_dtype.max_value dt

let reduce_segments op ~segments ids x =
  let dt = dtype x in
  if segments < 0 then err "reduce_segments" "%d segments" segments;
  if ndim x = 0 then err "reduce_segments" "x is a scalar, which has no rows";
  if ndim ids <> 1 || dim 0 ids <> dim 0 x then
    err "reduce_segments" "ids of shape %s for %d rows"
      (Shape.to_string (shape ids))
      (dim 0 x);
  let identity = identity_of "reduce_segments" op dt in
  let along = Array.init (ndim x) (fun d -> if d = 0 then dim 0 x else 1) in
  let into = Array.copy (shape x) in
  into.(0) <- segments;
  scatter
    ~mode:(op :> Nx_backend.scatter)
    ~axis:0
    ~indices:(broadcast_to (shape x) (reshape along ids))
    ~values:x
    (full (Value.context x) dt into identity)

(* Maps over segments

   [map_segments] sorts each group's positions by id, pads each id's run to
   whole blocks of [c] rows, calls [f] once on every block, and gathers each
   position's row of [f]'s result back. A block's pad rows repeat a row the
   block holds, so [f] meets only pairs the per-position map meets: a row of
   zeros is a pair that map never evaluates, where [f] may not be finite.

   The helpers below take a group's ids [[| r; j |]], [r] groups of [j]
   positions, and their rows [[| r; j; row... |]]. [call ~g ~c owners rows] is
   [f]'s result and its row shape, after checking its leading axes. *)

(* A block holds as many rows as its padding allows, up to 16: padding costs at
   most [c - 1] rows per segment, which stay below half the positions. Larger
   blocks read each segment's data fewer times, and 16 rows are the tile a GPU's
   tensor core multiplies at once. gpt-oss-20b's gate and up product of 512
   tokens takes 2.9 ms in blocks of 16 on an RTX 5000 Ada, 4.1 ms in blocks of
   32 and 7.5 ms in blocks of 8; of 64 tokens on the host, whose rows share a
   decoded weight four at a time, 220 ms in blocks of 4, 289 ms in blocks of 8
   and 440 ms in blocks of 16. *)
let largest_block = 16

(* [block ~segments j] is the rows of a block of [j] positions over [segments]
   segments, 1 when no block of 2 or more keeps the padding below half of them:
   then each position is its own block and nothing is sorted. *)
let block ~segments j =
  let rec go c =
    if c < 2 || 2 * (c - 1) * segments < j then c else go (c / 2)
  in
  (* With as many segments as positions no block of 2 or more qualifies,
     since [2 (c - 1) segments >= 2 j > j]. Deciding it first keeps the
     product below [30 j], which [segments] alone could overflow. *)
  if segments >= j then 1 else go largest_block

(* [windows what ~positional t] is the number of devices' windows that split
   [t]'s first axis, which must be a position axis. *)
let windows what ~positional t =
  List.fold_left
    (fun r (a, n) ->
      if a <> 0 || not positional then
        err "map_segments" "%s is split over devices along axis %d" what a;
      r * n)
    1
    (Placement.cuts (Value.placement t))

let units t = Array.make (Array.length t) 1

(* [first ~axes named] is, per group of [named] along [axes], its first
   in-range position as a position of the group, and whether it has one. [named]
   has a group's shape, [[| r; ... |]] for [r] groups of [j] positions; with no
   in-range position the first is [j], which a gather reads as zero. One [max]
   finds it: the first in-range position has the most positions after it. *)
let first ~axes named =
  let s = shape named in
  let r = s.(0) in
  let j = numel named / r in
  let last = Int64.of_int (j - 1) in
  let after =
    reshape s
      (broadcast_to [| r; j |]
         (reshape [| 1; j |]
            (sub
               (scalar (Value.context named) Int64 last)
               (arange (Value.context named) Int64 0 j 1))))
  in
  let most =
    max ~axes ~keepdims:true (where named after (scalar_like after (-1L)))
  in
  (sub (scalar_like most last) most, greater_equal_s most 0L)

(* [in_range ~segments owners] is [owners], which are in range already: the
   clamp writes that range into the program, where the gathers by them read
   it. *)
let in_range ~segments owners =
  clamp ~min:0L ~max:(Int64.of_int (segments - 1)) owners

(* [direct ~segments ~call ~named ~gs ids x] gives each position a block of its
   own. [gs] is a group's positions as axes, [[| r; ... |]], and [x] holds their
   rows in that shape, with an axis of length 1 where positions share a row. An
   out-of-range position keeps its row and takes the owner of the first
   in-range position that reads it. A row that no in-range position reads takes
   the group's first in-range position and its row, or zeros. Rows then vary
   only along the axes [x] varies along: at one token over a few experts, every
   block reads the token's row. *)
let direct ~segments ~call ~named ~gs ids x =
  let r = dim 0 ids and j = dim 1 ids in
  let n = Array.length gs in
  let lead = Array.sub (shape x) 0 n and row = Array.sub (shape x) n (ndim x - n) in
  let inner = List.init (n - 1) succ in
  let varies = List.exists (fun a -> lead.(a) > 1) inner in
  let shared = List.filter (fun a -> lead.(a) = 1 && gs.(a) > 1) inner in
  let at, any = first ~axes:[ 1 ] named in
  let id = take_along_axis ~axis:1 ~indices:at ids in
  let per_group t = reshape (Array.append [| r |] (Array.make (n - 1) 1)) t in
  let by_row t = reshape (Array.append (shape t) (units row)) t in
  (* Each row's owners, and whether an in-range position reads the row. *)
  let owners, read =
    if not varies then (where named ids id, per_group any)
    else if shared = [] then (where named ids id, reshape gs named)
    else
      let at_row, read = first ~axes:shared (reshape gs named) in
      let flat t = reshape [| r; j |] (broadcast_to gs t) in
      let owner = take_along_axis ~axis:1 ~indices:(flat at_row) ids in
      (where named ids (where (flat read) owner id), read)
  in
  let fallback =
    if not varies then scalar_like x (Nx_dtype.zero (dtype x))
    else
      let rows = reshape (Array.append [| r; j |] row) (broadcast_to (Array.append gs row) x) in
      reshape
        (Array.append (shape (per_group at)) row)
        (take_along_axis ~axis:1
           ~indices:(broadcast_to (Array.append [| r; 1 |] row) (by_row at))
           rows)
  in
  let rows = where (by_row read) x fallback in
  let y, out =
    call ~g:(r * j) ~c:1
      (reshape [| r * j |] (in_range ~segments owners))
      (reshape
         (Array.append [| r * j; 1 |] row)
         (broadcast_to (Array.append gs row) rows))
  in
  (reshape (Array.append [| r; j |] out) y, out)

(* [grouped ~segments ~c ~call ~named ids x] sorts each group's positions by id
   into blocks of [c] rows. *)
let grouped ~segments ~c ~call ~named ids x =
  let r = dim 0 ids and j = dim 1 ids in
  let row = Array.sub (shape x) 2 (ndim x - 2) in
  let ctx = Value.context ids and k = Int64.of_int in
  let along t indices = take_along_axis ~axis:1 ~indices t in
  let at, any = first ~axes:[ 1 ] named in
  let id = along ids at in
  (* Out-of-range positions sort last, as segment [segments], and take no
     slot. *)
  let key = where named ids (scalar_like ids (k segments)) in
  let order = argsort ~axis:1 key in
  (* Each segment's run in sorted order, [[| r; segments |]]: its length, first
     position, and slots padded to whole blocks. *)
  let count =
    scatter ~mode:`Add ~axis:1 ~indices:key ~values:(ones_like key)
      (zeros ctx Int64 [| r; segments |])
  in
  let first = sub (cumsum ~axis:1 count) count in
  let padded = mul_s (div_s (add_s count (k (c - 1))) (k c)) (k c) in
  let ends = cumsum ~axis:1 padded in
  let starts = sub ends padded in
  (* Blocks: a bound of the padded runs, whatever the ids. *)
  let blocks = (j + (Stdlib.min segments j * (c - 1)) + c - 1) / c in
  let slots = blocks * c in
  (* A block's owner is the number of runs that end at or before its first slot;
     a block past the last run takes the fallback id. *)
  let ended =
    cast Int64
      (sum ~axes:[ 2 ]
         (cast Int32
            (less_equal
               (reshape [| r; 1; segments |] ends)
               (reshape [| 1; blocks; 1 |] (arange ctx Int64 0 slots c)))))
  in
  let owners =
    in_range ~segments (where (less_s ended (k segments)) ended id)
  in
  let slot_owner =
    reshape [| r; slots |]
      (broadcast_to [| r; blocks; c |] (reshape [| r; blocks; 1 |] owners))
  in
  (* A slot past its run repeats the run's last row, which its block holds. *)
  let offset =
    sub
      (reshape [| 1; slots |] (arange ctx Int64 0 slots 1))
      (along starts slot_owner)
  in
  let held = minimum offset (sub_s (along count slot_owner) 1L) in
  let position = along order (add (along first slot_owner) held) in
  let position = where any position (scalar_like position (-1L)) in
  let by_row t = reshape (Array.append (shape t) (units row)) t in
  let rows =
    along x (broadcast_to (Array.append [| r; slots |] row) (by_row position))
  in
  let y, out =
    call ~g:(r * blocks) ~c
      (reshape [| r * blocks |] owners)
      (reshape (Array.append [| r * blocks; c |] row) rows)
  in
  (* Each position's slot: its run's first slot and its rank in the run, and -1
     out of range, which a scatter drops. *)
  let rank =
    scatter ~unique_indices:true ~axis:1 ~indices:order
      ~values:(broadcast_to [| r; j |] (arange ctx Int64 0 j 1))
      (zeros ctx Int64 [| r; j |])
  in
  let slot = add (along starts key) (sub rank (along first key)) in
  let slot = where named slot (scalar_like slot (-1L)) in
  let by_out t = reshape (Array.append (shape t) (units out)) t in
  ( along
      (reshape (Array.append [| r; slots |] out) y)
      (broadcast_to (Array.append [| r; j |] out) (by_out slot)),
    out )

let map_segments ~segments ids f x =
  let op = "map_segments" in
  if segments < 0 then err op "%d segments" segments;
  let s = shape ids and xs = shape x in
  let ns = Array.length s in
  if
    Array.length xs < ns
    || not (Array.for_all2 (fun d e -> d = e || d = 1) (Array.sub xs 0 ns) s)
  then
    err op "x of shape %s does not broadcast to ids of shape %s"
      (Shape.to_string xs) (Shape.to_string s);
  let row = Array.sub xs ns (Array.length xs - ns) in
  let positional = ns > 0 in
  let r =
    match (windows "ids" ~positional ids, windows "x" ~positional x) with
    | 1, r | r, 1 -> r
    | a, b when a = b -> a
    | a, b -> err op "ids are split in %d windows and x in %d" a b
  in
  let call ~g ~c owners rows =
    let y = f owners rows in
    let ys = shape y in
    if Array.length ys < 2 || ys.(0) <> g || ys.(1) <> c then
      err op "f gave shape %s for %d blocks of %d rows" (Shape.to_string ys) g c;
    (y, Array.sub ys 2 (Array.length ys - 2))
  in
  let full = broadcast_to (Array.append s row) x in
  let p = Array.fold_left ( * ) 1 s in
  if segments = 0 || p = 0 then
    let y, out =
      call ~g:0 ~c:1
        (zeros (Value.context ids) Int64 [| 0 |])
        (zeros (Value.context x) (dtype x) (Array.append [| 0; 1 |] row))
    in
    zeros (Value.context y) (dtype y) (Array.append s out)
  else
    let j = p / r in
    let ids = reshape [| r; j |] ids in
    let named =
      logical_and (greater_equal_s ids 0L) (less_s ids (Int64.of_int segments))
    in
    let c = block ~segments j in
    let y, out =
      if c = 1 then
        (* A group's positions as axes, and [x]'s rows in that shape, unbroadcast. *)
        let gs, lead =
          if not positional then ([| 1; 1 |], [| 1; 1 |])
          else
            let split d = if d = 1 then [| 1; 1 |] else [| r; d / r |] in
            let rest t = Array.sub t 1 (ns - 1) in
            ( Array.append (split s.(0)) (rest s),
              Array.append (split xs.(0)) (rest xs) )
        in
        direct ~segments ~call ~named ~gs ids (reshape (Array.append lead row) x)
      else
        grouped ~segments ~c ~call ~named ids
          (reshape (Array.append [| r; j |] row) full)
    in
    let named = reshape (Array.append [| r; j |] (units out)) named in
    reshape (Array.append s out) (where named y (zeros_like y))

(* Scans of structures

   [associative_scan] is the odd/even recursion: it combines neighbouring pairs,
   scans the pairs, which gives the outputs at odd indices, and combines each of
   those with the element after it, which gives the even ones. Every output's
   association is fixed by its index, never by the length. *)

let associative_scan ?(axis = 0) st f x =
  let along t =
    let r = ndim t in
    let a = if axis < 0 then axis + r else axis in
    if a < 0 || a >= r then
      err "associative_scan" "axis %d out of bounds for %dD tensor" axis r;
    a
  in
  (* [t]'s elements along [a] in [r], an index specification. *)
  let along_axis a r t = slice (List.init a (fun _ -> A) @ [ r ]) t in
  let cut lo hi a t = along_axis a (R (lo, hi)) t in
  let part lo hi x = Ptree.map st (fun _ t -> cut lo hi (along t) t) x in
  let every_other start x =
    Ptree.map st
      (fun _ t ->
        let a = along t in
        along_axis a (Rs (start, dim a t, 2)) t)
      x
  in
  (* [e] and [o] alternated into [n] elements, [e] first. *)
  let weave n a e o =
    let o = if n mod 2 = 0 then o else concatenate ~axis:a [ o; cut 0 1 a e ] in
    let s = Array.copy (shape e) in
    s.(a) <- 2 * s.(a);
    cut 0 n a (reshape s (stack ~axis:(a + 1) [ e; o ]))
  in
  let rec scan n x =
    if n < 2 then x
    else
      let even = every_other 0 x and odd = every_other 1 x in
      let m = (n + 1) / 2 in
      let odds = scan (n / 2) (f (part 0 (n / 2) even) odd) in
      let rest = f (part 0 (m - 1) odds) (part 1 m even) in
      let evens =
        Ptree.map2 st
          (fun _ e r -> concatenate ~axis:(along e) [ cut 0 1 (along e) e; r ])
          even rest
      in
      Ptree.map2 st (fun _ e o -> weave n (along e) e o) evens odds
  in
  match Ptree.fold st (fun p t acc -> (p, dim (along t) t) :: acc) x [] with
  | [] -> x
  | (p, n) :: rest ->
      List.iter
        (fun (q, m) ->
          if m <> n then
            err "associative_scan" "%s has %d elements along the axis, %s %d"
              (Ptree.Path.to_string q) m (Ptree.Path.to_string p) n)
        rest;
      scan n x

let ewma ?axis ~alpha x =
  if not (alpha > 0. && alpha <= 1.) then
    err "ewma" "alpha %s, expected in (0, 1]" (float_text Float64 alpha);
  let flat, axis =
    match axis with
    | None -> (flatten x, 0)
    | Some a -> (x, resolve_single_axis x a)
  in
  let n = dim axis flat in
  if alpha = 1. || n = 0 then x
  else
    (* Element i is the map [y -> a y + b], where [a] is 0 at 0 and [1 - alpha]
       after, and [b] is [x0] at 0 and [alpha xi] after. The result is the [b]
       of their running composition. An [a] that underflowed to 0 still carries
       an infinite [b], as the recurrence would. *)
    let smooth x =
      let dt = dtype x and s = shape x in
      let along lo hi =
        Array.mapi (fun i d -> if i = axis then (lo, hi) else (0, d)) s
      in
      let head = shrink (along 0 1) x and tail = shrink (along 1 n) x in
      let a =
        pad
          (Array.mapi (fun i _ -> if i = axis then (1, 0) else (0, 0)) s)
          (Nx_dtype.zero dt)
          (broadcast_to (shape tail)
             (scalar_like x (Nx_dtype.of_float dt (1. -. alpha))))
      in
      let b =
        concatenate ~axis [ head; mul_s tail (Nx_dtype.of_float dt alpha) ]
      in
      let compose (a1, b1) (a2, b2) =
        (mul a1 a2, where (isinf b1) (add b1 b2) (fma a2 b1 b2))
      in
      snd (associative_scan ~axis Ptree.(pair tensor tensor) compose (a, b))
    in
    reshape (shape x) (at_float32 { f = smooth } flat)

(* Ranges

   [reduce_ranges] answers each range from levels of chunks. Level [k] holds, at
   each row, the combination of its chunk of [2^k] rows from that row to the
   chunk's end, its suffix, and from the chunk's start to that row, its prefix.
   Level [k] comes from level [k - 1]: a suffix in a chunk's first half goes on
   through the second half's total, and a prefix in its second half starts from
   the first half's total. A range whose first and last rows lie in neighbouring
   chunks of a level is the first row's suffix, then the last row's prefix. At
   every such level the boundary between the two chunks is the same row, and a
   higher level only adds the identity to the suffix and the prefix, so the
   range's bits are those of its bounds whichever level answers it. A range of
   [L] rows has such a level once its chunks hold [L - 1] rows, and no range
   holds more than [x]'s rows, so the levels are fixed by [x]'s shape and
   nothing is read. *)

(* The extreme [op] of [a] and [b] as a selection of one of them, so that a
   range's extreme is one of its rows. Its values are {!maximum}'s and
   {!minimum}'s: a NaN [a] first, then a NaN [b], and [-0.] below [0.]; at any
   other tie the selection takes [b]. *)
let selected_extreme op a b =
  let ahead = match op with `Max -> cmplt b a | `Min -> cmplt a b in
  let a_wins =
    if not (Nx_dtype.is_float (dtype a)) then ahead
    else
      let zero = scalar_like a (Nx_dtype.zero (dtype a)) in
      let negative_zero x = logical_and (cmpeq x zero) (cmplt (recip x) zero) in
      let zeros = logical_and (cmpeq a zero) (cmpeq b zero) in
      let signed =
        match op with
        | `Max -> logical_and (negative_zero b) (logical_not (negative_zero a))
        | `Min -> logical_and (negative_zero a) (logical_not (negative_zero b))
      in
      logical_or (isnan a)
        (logical_and
           (logical_not (isnan b))
           (logical_or ahead (logical_and zeros signed)))
  in
  where a_wins a b

let reduce_ranges op ~lo ~hi x =
  if ndim x = 0 then err "reduce_ranges" "x is a scalar, which has no rows";
  if ndim lo <> 1 || not (Shape.equal (shape lo) (shape hi)) then
    err "reduce_ranges" "bounds of shapes %s and %s, not one of each per range"
      (Shape.to_string (shape lo))
      (Shape.to_string (shape hi));
  let n = dim 0 x and m = dim 0 lo in
  let clip b = clamp ~min:0L ~max:(Int64.of_int n) b in
  let lo = clip lo and hi = clip hi in
  let rows = Array.sub (shape x) 1 (ndim x - 1) in
  let ranges x =
    let id = identity_of "reduce_ranges" op (dtype x) in
    let none =
      full (Value.context x) (dtype x) (Array.append [| m |] rows) id
    in
    let longest = if m = 0 then 0 else n in
    if longest <= 0 then none
    else
      let combine =
        match op with
        | `Add -> add
        | `Max -> selected_extreme `Max
        | `Min -> selected_extreme `Min
      in
      let rec level k = if 1 lsl k >= longest - 1 then k else level (k + 1) in
      let top = level 0 in
      let chunk = 1 lsl top in
      let padded = (n + chunk - 1) / chunk * chunk in
      let whole = Array.map (fun d -> (0, d)) rows
      and untouched = Array.map (fun _ -> (0, 0)) rows in
      let x =
        if padded = n then x
        else pad (Array.append [| (0, padded - n) |] untouched) id x
      in
      let at indices t = take ~axis:0 ~indices t in
      let per_range b =
        reshape (Array.append [| m |] (Array.map (fun _ -> 1) rows)) b
      in
      let double k suf pre =
        let h = 1 lsl (k - 1) in
        let blocks = padded / (2 * h) in
        let halves t = reshape (Array.append [| blocks; 2; h |] rows) t in
        let row half i t =
          shrink
            (Array.append [| (0, blocks); (half, half + 1); (i, i + 1) |] whole)
            t
        in
        let around before after t =
          pad
            (Array.append [| (0, 0); (before, after); (0, 0) |] untouched)
            id t
        in
        let flat t = reshape (Array.append [| padded |] rows) t in
        (* A half's total is its first row's suffix and its last row's
           prefix. *)
        let s = halves suf and p = halves pre in
        ( flat (combine s (around 0 1 (row 1 0 s))),
          flat (combine (around 1 0 (row 0 (h - 1) p)) p) )
      in
      let l = lo and r = sub_s hi 1L in
      (* [ls] and [rs] are the chunks of [l] and [r] at level [k]. *)
      let rec up k (suf, pre) ls rs res =
        let here = equal_s (sub rs ls) 1L in
        let res = where (per_range here) (combine (at l suf) (at r pre)) res in
        if k = top then res
        else up (k + 1) (double (k + 1) suf pre) (div_s ls 2L) (div_s rs 2L) res
      in
      let one_row = where (per_range (equal l r)) (at r x) none in
      let res = up 0 (x, x) l r one_row in
      (* A sum is [0.] plus its terms, as [sum]'s: [-0.] terms give [0.]. *)
      match op with
      | `Add -> add_s res (Nx_dtype.zero (dtype x))
      | _ -> res
  in
  match op with `Add -> at_float32 { f = ranges } x | _ -> ranges x

(* Above this many entries [top_k] stops taking one greatest entry per pass
   over the axis, each pass waiting on the one before it, and selects them by
   radix instead. *)
let top_k_rounds = 8

(* Up to this many entries along the axis, [top_k] sorts the keys instead of
   selecting them. Compiled on Metal (M1 Max, float32, 1 and 64 rows, k from
   64 to 512), sorting 2048 entries costs 0.55 to 1.2 ms per call and
   selecting 0.65 to 1.1 ms; from 4096 on, selection wins, 2.5 to 5 times at
   64 rows. Eagerly a sort is one C call and cheaper at every length: 0.3 ms
   against 2.9 ms for 512 of 4096, 3.5 against 10.5 for 512 of 32768. *)
let top_k_sorted = 2048

(* Up to this many comparisons, [k * k] per row, the selected entries are
   ordered by counting, for each, the entries that precede it: one pass, where
   a sort is a network of them. Beyond, an eager backend would hold too many
   at once, and they are sorted. *)
let top_k_counted = 1 lsl 24

(* Up to this many entries along the axis, [top_k] ranks every entry by
   counting the entries that precede it: [n * n] comparisons per row and no
   pass waiting on another, where taking one entry per pass costs a kernel per
   pass when compiled. Compiled for the host, the indices of 2 of 4 entries
   take 2 kernels instead of 6, 4 of 32 take 2 instead of 14, and 16 of 32
   (a sort before) 2 instead of 22. Eagerly the comparisons cost more than
   the passes: 0.77 ms against 0.20 for 4 of 32 over 512 rows. *)
let top_k_compared = 32

(* A radix round decides [radix_bits] bits of the threshold's key at once,
   with one count per nonzero digit, all held at once by an eager backend:
   2^bits - 1 counts per entry. Rows of up to [radix_wide] entries in all take
   4 bits, 8 rounds for a 32-bit key; more take 2, 16 rounds, with a fifth of
   the counts. On Metal (M1 Max, float32, k = 512) the two cost the same
   within noise at 64 rows of 32768 (1.9 to 2.2 ms) and of 131072 (9.0 ms);
   eagerly, 64 rows of 131072 peak at 476 MB instead of 860 MB and take 90 ms
   instead of 128 ms for 64 rows of 32768. What remains of the peak is the
   compaction's, about 36 bytes per entry held at once (the keys, the masks,
   the running counts, the slots and the scatter's template and output); the
   2-bit counts take about 9. *)
let radix_wide = 1 lsl 20
let radix_bits ~entries = if entries <= radix_wide then 4 else 2

(* The rounds count entries per chunk of this many, reduced across chunks
   first: neighbouring lanes then read neighbouring entries. *)
let radix_chunk = 512

(* The element of a signed key dtype that an int64 in its range stands for. *)
let key_of_int64 (type c d) (dt : (c, d) Nx_dtype.t) (v : int64) : c =
  match dt with
  | Nx_dtype.Int8 -> Int64.to_int v
  | Nx_dtype.Int16 -> Int64.to_int v
  | Nx_dtype.Int32 -> Int64.to_int32 v
  | Nx_dtype.Int64 -> v
  | _ -> invalid_arg "key_of_int64: not a signed key dtype"

(* [radix_select_in cd ~k keys] is the positions of the [k] greatest entries of
   each row of the signed integer [keys], shaped [b; n] with [n] above
   [top_k_sorted], in the order of a stable descending sort.

   The threshold, the [k]th greatest key, is found by radix select: each round
   extends its known prefix by the [radix_bits] greatest bits that leave at
   least [k] keys at or above. The keys above the threshold and the first of
   those equal to it fill the [k] slots, compacted in order by a running count
   and a scatter, and are then put in order.

   Counts and positions are computed in [cd], which holds every running count,
   at most [k * (n + 1)]. *)
let radix_select_in (type c d p q) (cd : (p, q) Nx_dtype.t) ~k (keys : (c, d) t)
    =
  let ctx = Value.context keys in
  let kd = dtype keys in
  let b = dim 0 keys and n = dim 1 keys in
  let width = 8 * Nx_dtype.itemsize kd in
  let kk = scalar ctx cd (Nx_dtype.of_float cd (float_of_int k)) in
  let rows =
    let g = (n + radix_chunk - 1) / radix_chunk in
    (* Padding holds the least key, which only a count every key already
       passes can include. *)
    pad [| (0, 0); (0, (g * radix_chunk) - n) |] (Nx_dtype.min_value kd) keys
    |> reshape [| b; 1; g; radix_chunk |]
  in
  let radix_bits = radix_bits ~entries:(b * n) in
  let digits = (1 lsl radix_bits) - 1 in
  let lowest = Int64.shift_left (-1L) (width - 1) in
  let constants values =
    create ctx kd [| 1; digits |] (Array.map (key_of_int64 kd) values)
  in
  let rec search prefix top =
    if top = 0 then prefix
    else
      let shift = top - radix_bits in
      let offsets =
        Array.init digits (fun j ->
            Int64.shift_left (Int64.of_int (j + 1)) shift)
      in
      let trials =
        if top = width then constants (Array.map (Int64.add lowest) offsets)
        else add prefix (constants offsets)
      in
      let above =
        greater_equal rows (reshape [| -1; digits; 1; 1 |] trials)
      in
      (* A chunk lane counts at most one key per chunk, so its count fits
         int16 below 32768 chunks: half what int32 would hold at once. *)
      let count (type r s) (dt : (r, s) Nx_dtype.t) =
        sum ~axes:[ 2 ] (cast dt above)
        |> contiguous |> cast cd |> sum ~axes:[ 2 ] |> contiguous
      in
      let counts =
        if dim 2 rows < 32768 then count Nx_dtype.int16
        else count Nx_dtype.int32
      in
      let kept = where (greater_equal counts kk) trials prefix in
      search (max ~axes:[ 1 ] ~keepdims:true kept) shift
  in
  let threshold =
    search (full ctx kd [| b; 1 |] (Nx_dtype.min_value kd)) width
  in
  let above = greater keys threshold and at = equal keys threshold in
  let n_above = sum ~axes:[ 1 ] ~keepdims:true (cast cd above) in
  let room = sub kk n_above in
  (* One running count carries both, as [above + k * at]: fewer than [k] keys
     are above. *)
  let running = cumsum ~axis:1 (add (cast cd above) (mul (cast cd at) kk)) in
  let before_above = mod_ running kk and before_at = div running kk in
  let position =
    broadcast_to [| b; n |] (reshape [| 1; n |] (arange ctx cd 0 n 1))
  in
  (* A stable partition: the keys above, then the keys taken at the threshold,
     then the rest, each in position order. *)
  let slot =
    where above
      (sub_s before_above (Nx_dtype.one cd))
      (where
         (logical_and at (less_equal before_at room))
         (add n_above (sub_s before_at (Nx_dtype.one cd)))
         (add (sub (sub position before_above) (minimum before_at room)) kk))
  in
  let chosen =
    scatter ~unique_indices:true ~axis:1 ~indices:(cast Nx_dtype.int64 slot)
      ~values:position
      (zeros ctx cd [| b; n |])
    |> shrink [| (0, b); (0, k) |]
    |> cast Nx_dtype.int64
  in
  let chosen_keys = take_along_axis ~axis:1 ~indices:chosen keys in
  if b * k * k > top_k_counted then
    take_along_axis ~axis:1
      ~indices:(argsort ~descending:true ~axis:1 chosen_keys)
      chosen
  else
    (* A key's place is the number of keys before it: greater ones, and equal
       ones in an earlier slot. *)
    let mine = reshape [| b; 1; k |] chosen_keys in
    let other = reshape [| b; k; 1 |] chosen_keys in
    let slots = arange ctx Nx_dtype.int32 0 k 1 in
    let earlier =
      less (reshape [| k; 1 |] slots) (reshape [| 1; k |] slots)
    in
    let precedes =
      logical_or (greater other mine) (logical_and earlier (equal other mine))
    in
    let place = sum ~axes:[ 1 ] (cast cd precedes) in
    scatter ~unique_indices:true ~axis:1
      ~indices:(cast Nx_dtype.int64 place)
      ~values:chosen
      (zeros ctx Nx_dtype.int64 [| b; k |])

(* [select_by_counting ~k keys] is the positions of the [k] greatest entries
   of each row of [keys], shaped [b; n], in the order of a stable descending
   sort. An entry's place is the number of entries that precede it, greater
   ones and equal ones at earlier positions, and slot [r] holds the position
   whose place is [r]: [n * n] comparisons per row, none waiting on another. *)
let select_by_counting ~k keys =
  let ctx = Value.context keys in
  let b = dim 0 keys and n = dim 1 keys in
  let position = arange ctx Nx_dtype.int32 0 n 1 in
  let mine = reshape [| b; 1; n |] keys in
  let other = reshape [| b; n; 1 |] keys in
  let earlier =
    less (reshape [| n; 1 |] position) (reshape [| 1; n |] position)
  in
  let precedes =
    where earlier (greater_equal other mine) (greater other mine)
  in
  let place = sum ~axes:[ 1 ] (cast Nx_dtype.int32 precedes) in
  let slot = reshape [| 1; k; 1 |] (arange ctx Nx_dtype.int32 0 k 1) in
  let at = equal (reshape [| b; 1; n |] place) slot in
  let none = scalar ctx Nx_dtype.int32 0l in
  cast Nx_dtype.int64
    (sum ~axes:[ 2 ] (where at (reshape [| 1; 1; n |] position) none))

(* [radix_select ~k keys] is [radix_select_in] in the narrowest of [int32] and
   [int64] that holds its counts. *)
let radix_select ~k keys =
  if k * (dim 1 keys + 1) <= Int32.to_int Int32.max_int then
    radix_select_in Nx_dtype.int32 ~k keys
  else radix_select_in Nx_dtype.int64 ~k keys

(* [counted ~b ~n] is [true] iff [b] rows of [n] entries are ranked by
   counting. *)
let counted ~b ~n = n <= top_k_compared && b * n * n <= top_k_counted

(* [select ~k keys] is [radix_select ~k keys], or the first [k] positions of a
   stable descending sort of [keys] when their rows are short, or their ranks
   counted when they are shorter. *)
let select (type c d) ~k (keys : (c, d) t) =
  if counted ~b:(dim 0 keys) ~n:(dim 1 keys) then select_by_counting ~k keys
  else if dim 1 keys <= top_k_sorted then
    shrink
      [| (0, dim 0 keys); (0, k) |]
      (argsort ~descending:true ~axis:1 keys)
  else radix_select ~k keys

(* The key of a float, a signed integer of its width that orders as the sort
   order does: the bits, with the other bits of a negative float flipped, and
   NaN above every number. No number keys to the greatest integer, whose bits
   are a NaN's. -0, a negative float, keys just below +0. *)
let float_key (type a b c d) (kd : (c, d) Nx_dtype.t) (x : (a, b) t) =
  let bits = bitcast kd x in
  let flipped =
    where
      (less bits (zeros_like bits))
      (bitwise_xor bits (full_like bits (Nx_dtype.max_value kd)))
      bits
  in
  where (isnan x) (full_like bits (Nx_dtype.max_value kd)) flipped

(* The key of an unsigned integer: its bits with the top one flipped, read
   signed. *)
let unsigned_key (type a b c d) (kd : (c, d) Nx_dtype.t) (top : a)
    (x : (a, b) t) =
  bitcast kd (bitwise_xor x (full_like x top))

(* [select_by_passes ~k ~axis keys] is the positions of the [k] greatest [keys]
   along [axis], in the order of a stable descending sort, one pass over the
   axis for each. *)
let select_by_passes (type c d) ~k ~axis (keys : (c, d) t) =
  let n = dim axis keys in
  let along = Array.make (ndim keys) 1 in
  along.(axis) <- n;
  let position =
    reshape along (arange (Value.context keys) Nx_dtype.int64 0 n 1)
  in
  let low = full_like keys (Nx_dtype.min_value (dtype keys)) in
  (* One round picks the first free entry that a descending sort would place
     next. Comparing against the greatest key, under the free mask, keeps an
     entry equal to [low] distinct from one already taken. *)
  let pick free =
    let greatest = max ~axes:[ axis ] ~keepdims:true (where free keys low) in
    argmax ~axis ~keepdims:true
      (cast Nx_dtype.int32 (logical_and free (equal keys greatest)))
  in
  let rec rounds i free acc =
    if i = k then concatenate ~axis (List.rev acc)
    else
      let index = pick free in
      rounds (i + 1)
        (logical_and free (not_equal position index))
        (index :: acc)
  in
  rounds 0 (ones (Value.context keys) Nx_dtype.bool (shape keys)) []

let top_k (type a b) ~k ?(axis = -1) (x : (a, b) t) =
  let r = ndim x in
  if r = 0 then err "top_k" "requires at least one dimension";
  let axis = if axis < 0 then axis + r else axis in
  if axis < 0 || axis >= r then
    err "top_k" "axis %d out of bounds for %dD tensor" axis r;
  let n = dim axis x in
  if k < 1 || k > n then err "top_k" "k = %d is outside [1, %d]" k n;
  let dt = dtype x in
  if Nx_dtype.is_complex dt then
    err "top_k" "complex numbers have no selection key";
  let rows = Array.fold_left ( * ) 1 (shape x) / n in
  let positions (type c d) (keys : (c, d) t) =
    if k <= top_k_rounds && not (counted ~b:rows ~n) then
      select_by_passes ~k ~axis keys
    else
      let last = moveaxis axis (-1) keys in
      let batch = Array.sub (shape last) 0 (r - 1) in
      let b = Array.fold_left ( * ) 1 batch in
      let chosen =
        if b = 0 then zeros (Value.context x) Nx_dtype.int64 [| 0; k |]
        else
          select ~k (reshape [| b; n |] last)
      in
      moveaxis (-1) axis (reshape (Array.append batch [| k |]) chosen)
  in
  let indices =
    match dt with
    | Nx_dtype.Float16 -> positions (float_key Nx_dtype.int16 x)
    | Nx_dtype.BFloat16 -> positions (float_key Nx_dtype.int16 x)
    | Nx_dtype.Float32 -> positions (float_key Nx_dtype.int32 x)
    | Nx_dtype.Float64 -> positions (float_key Nx_dtype.int64 x)
    | Nx_dtype.Float8_e4m3 ->
        positions (float_key Nx_dtype.int16 (cast Nx_dtype.float16 x))
    | Nx_dtype.Float8_e5m2 ->
        positions (float_key Nx_dtype.int16 (cast Nx_dtype.float16 x))
    | Nx_dtype.Int4 -> positions (cast Nx_dtype.int8 x)
    | Nx_dtype.Int8 -> positions x
    | Nx_dtype.Int16 -> positions x
    | Nx_dtype.Int32 -> positions x
    | Nx_dtype.Int64 -> positions x
    | Nx_dtype.UInt4 | Nx_dtype.Bool | Nx_dtype.Bit ->
        positions (unsigned_key Nx_dtype.int8 0x80 (cast Nx_dtype.uint8 x))
    | Nx_dtype.UInt8 -> positions (unsigned_key Nx_dtype.int8 0x80 x)
    | Nx_dtype.UInt16 -> positions (unsigned_key Nx_dtype.int16 0x8000 x)
    | Nx_dtype.UInt32 ->
        positions (unsigned_key Nx_dtype.int32 Int32.min_int x)
    | Nx_dtype.UInt64 ->
        positions (unsigned_key Nx_dtype.int64 Int64.min_int x)
    | Nx_dtype.Complex64 | Nx_dtype.Complex128 -> assert false
  in
  (take_along_axis ~axis ~indices x, indices)

(* Keys *)

(* The key of a signed integer is its value in the signed integers [s] of the
   key's width with the sign bit flipped. A float takes the bits of its value in
   the float [f] of the key's width, read as [s], all flipped when the sign is
   set and only the sign bit otherwise, and every NaN keys to all ones. *)
let order_key (type a b c d) (kd : (c, d) Nx_dtype.t) (x : (a, b) t) : (c, d) t
    =
  let signed s =
    let v = cast s x in
    bitcast kd (bitwise_xor v (scalar_like v (Nx_dtype.min_value s)))
  in
  let float f s =
    let x = cast f x in
    let b = bitcast s x in
    let ones = scalar_like b (Nx_dtype.minus_one s) in
    let sign = scalar_like b (Nx_dtype.min_value s) in
    let flip = where (less_s b (Nx_dtype.zero s)) ones sign in
    bitcast kd (where (isnan x) ones (bitwise_xor b flip))
  in
  match (dtype x, kd) with
  | (Bool | Bit | UInt4 | UInt8), (UInt8 | UInt16 | UInt32 | UInt64)
  | UInt16, (UInt16 | UInt32 | UInt64)
  | UInt32, (UInt32 | UInt64)
  | UInt64, UInt64 ->
      cast kd x
  | (Int4 | Int8), UInt8 -> signed Int8
  | (Int4 | Int8 | Int16), UInt16 -> signed Int16
  | (Int4 | Int8 | Int16 | Int32), UInt32 -> signed Int32
  | (Int4 | Int8 | Int16 | Int32 | Int64), UInt64 -> signed Int64
  | Float8_e4m3, UInt8 -> float Float8_e4m3 Int8
  | Float8_e5m2, UInt8 -> float Float8_e5m2 Int8
  | BFloat16, UInt16 -> float BFloat16 Int16
  | (Float8_e4m3 | Float8_e5m2 | Float16), UInt16 -> float Float16 Int16
  | (Float8_e4m3 | Float8_e5m2 | Float16 | BFloat16 | Float32), UInt32 ->
      float Float32 Int32
  | (Float8_e4m3 | Float8_e5m2 | Float16 | BFloat16 | Float32 | Float64), UInt64
    ->
      float Float64 Int64
  | (Complex64 | Complex128), _ ->
      err "order_key" "complex numbers have no order key"
  | dt, _ ->
      err "order_key"
        "no %s key for %s: keys are uint8, uint16, uint32 or uint64, at least \
         as wide as the elements"
        (Nx_dtype.to_string kd) (Nx_dtype.to_string dt)

(* [check_keys ~op keys] refuses what [op] does not read as keys. *)
let check_keys ~op keys =
  let r = ndim keys in
  if r <> 1 && r <> 2 then
    err op "keys of shape %s, not 1-D or 2-D" (Shape.to_string (shape keys));
  if Nx_dtype.is_complex (dtype keys) then
    err op "complex numbers have no order"

let lexsort keys =
  check_keys ~op:"lexsort" keys;
  if ndim keys = 1 then argsort keys
  else
    let n = dim 0 keys and w = dim 1 keys in
    let column j = slice [ A; I j ] keys in
    (* Columns from the last: each stable sort keeps the order of the columns
       after it among its ties. *)
    let rec sorted j perm =
      if j < 0 then perm
      else
        sorted (j - 1)
          (take ~indices:(argsort (take ~indices:perm (column j))) perm)
    in
    if w = 0 then arange (Value.context keys) Int64 0 n 1
    else sorted (w - 2) (argsort (column (w - 1)))

(* [numeric_key x] is [order_key UInt64 x] with [-0.]'s key moved to [0.]'s, so
   that numbers compare as [less] compares them. *)
let numeric_key (type a b) (x : (a, b) t) =
  let k = order_key UInt64 x in
  if Nx_dtype.is_float (dtype x) then
    where (equal_s k Int64.max_int) (scalar_like k Int64.min_int) k
  else k

(* Whether each row of the keys [a] comes before the row of [b] at its index, or
   before or at it unless [strict], column 0 first. *)
let lexicographic ~strict a b =
  let w = dim 1 a in
  let col x j = slice [ A; I j ] x in
  let rec from j =
    let a = col a j and b = col b j in
    if j = w - 1 then if strict then less a b else less_equal a b
    else logical_or (less a b) (logical_and (equal a b) (from (j + 1)))
  in
  if w = 0 then full (Value.context a) Bool [| dim 0 a |] (not strict)
  else from 0

let bit_length n =
  let rec go n b = if n = 0 then b else go (n lsr 1) (b + 1) in
  go n 0

let searchsorted (type a b) ~side (s : (a, b) t) (v : (a, b) t) =
  check_keys ~op:"searchsorted" s;
  let rows = ndim s = 2 in
  if rows && (ndim v <> 2 || dim 1 v <> dim 1 s) then
    err "searchsorted" "keys of shape %s among rows of shape %s"
      (Shape.to_string (shape v))
      (Shape.to_string (shape s));
  let ctx = Value.context v and m = dim 0 s in
  let n = if rows then dim 0 v else numel v in
  let result = if rows then [| n |] else shape v in
  if m = 0 || n = 0 then zeros ctx Int64 result
  else
    let q =
      if rows then numeric_key v
      else reshape [| n |] (numeric_key v)
    and rounds = bit_length m in
    (* All-ones keys past [m], which no query precedes: no round reads outside
       the table. *)
    let table =
      let tail = (1 lsl rounds) - 1 - m in
      pad
        (if rows then [| (0, tail); (0, 0) |] else [| (0, tail) |])
        (-1L) (numeric_key s)
    in
    let strict = match side with `Left -> true | `Right -> false in
    let before row =
      if rows then lexicographic ~strict row q
      else if strict then less row q
      else less_equal row q
    in
    (* [last] is the last position whose key comes before, or -1. *)
    let rec bisect step last =
      if step = 0 then last
      else
        let at = add_s last (Int64.of_int step) in
        bisect (step / 2)
          (where (before (take ~axis:0 ~indices:at table)) at last)
    in
    let count =
      add_s (bisect (1 lsl (rounds - 1)) (full ctx Int64 [| n |] (-1L))) 1L
    in
    (* An all-ones query also counts the padding under [`Right]. *)
    reshape result (if strict then count else minimum_s count (Int64.of_int m))

type groups = { ids : int64_t; first : int64_t; counts : int64_t }

let unique keys =
  check_keys ~op:"unique" keys;
  let ctx = Value.context keys and n = dim 0 keys in
  if n = 0 then
    let none = empty ctx Int64 [| 0 |] in
    { ids = none; first = none; counts = none }
  else
    let k = order_key UInt64 keys in
    let ids =
      B.group ~by:"Nx.unique" (if ndim k = 1 then reshape [| n; 1 |] k else k)
    in
    (* Groups are numbered in order of first appearance, from 0. *)
    let segments = Int64.to_int (read_item ~by:"Nx.unique" (max ids)) + 1 in
    (* [`Set] keeps the last update in index order, so over the rows reversed
       it keeps each group's first. *)
    let first =
      scatter ~axis:0 ~indices:(flip ids)
        ~values:(flip (arange ctx Int64 0 n 1))
        (zeros ctx Int64 [| segments |])
    and counts =
      reduce_segments `Add ~segments ids
        (broadcast_to [| n |] (scalar ctx Int64 1L))
    in
    { ids; first; counts }

(* A point's cell is its bin along each dimension in C order: the edges at or
   below it less one, and the last bin for the last edge. A point outside the
   edges along any dimension has cell -1, which the sum drops. *)
let histogram (type b) ?weights (dims : ((float, b) t * (float, b) t) list) :
    (float, b) t =
  let points, dt, ctx =
    match dims with
    | [] -> err "histogram" "no dimension"
    | (_, x) :: _ -> (shape x, dtype x, Value.context x)
  in
  let n = Shape.numel points in
  List.iteri
    (fun d (e, x) ->
      if ndim e <> 1 || dim 0 e < 2 then
        err "histogram" "edges %d of shape %s, expected at least two in a row" d
          (Shape.to_string (shape e));
      if not (Shape.equal (shape x) points) then
        err "histogram" "points %d of shape %s, the first are of shape %s" d
          (Shape.to_string (shape x))
          (Shape.to_string points))
    dims;
  let bins = List.map (fun (e, _) -> dim 0 e - 1) dims in
  let cell, inside =
    List.fold_left2
      (fun (cell, inside) (e, x) k ->
        let x = reshape [| n |] x in
        let last = equal x (broadcast_to [| n |] (slice [ I k ] e)) in
        let b = sub_s (searchsorted ~side:`Right e x) 1L in
        let b = where last (full ctx Int64 [| n |] (Int64.of_int (k - 1))) b in
        let within =
          logical_and (greater_equal_s b 0L) (less_s b (Int64.of_int k))
        in
        (add (mul_s cell (Int64.of_int k)) b, logical_and inside within))
      (zeros ctx Int64 [| n |], full ctx Bool [| n |] true)
      dims bins
  in
  let ids = where inside cell (full ctx Int64 [| n |] (-1L)) in
  let segments = List.fold_left ( * ) 1 bins in
  let sums =
    match weights with
    | None ->
        cast dt
          (reduce_segments `Add ~segments ids
             (broadcast_to [| n |] (scalar ctx Int64 1L)))
    | Some w -> (
        if not (Shape.equal (shape w) points) then
          err "histogram" "weights of shape %s for points of shape %s"
            (Shape.to_string (shape w))
            (Shape.to_string points);
        let sum (type c) (acc : (float, c) dtype) =
          cast dt
            (reduce_segments `Add ~segments ids (cast acc (reshape [| n |] w)))
        in
        match dt with Float64 -> sum Float64 | _ -> sum Float32)
  in
  reshape (Array.of_list bins) sums

(* ───── Linear Algebra ───── *)

let matmul_with_alloc a_shape b_shape a b =
  let a_axis = Array.length a_shape - 1 in
  let b_axis = Array.length b_shape - 2 in
  let a_contract = a_shape.(a_axis) and b_contract = b_shape.(b_axis) in
  if a_contract <> b_contract then
    err "dot"
      "cannot contract %s (last axis: %d) to %s (axis %d: %d) (size %d≠%d)"
      (Shape.to_string a_shape) a_contract (Shape.to_string b_shape) b_axis
      b_contract a_contract b_contract;
  B.matmul a b

let matmul a_orig b_orig =
  let a_shape = shape a_orig and b_shape = shape b_orig in
  let a_ndim = Array.length a_shape and b_ndim = Array.length b_shape in
  if a_ndim = 0 || b_ndim = 0 then
    invalid_arg "matmul: inputs cannot be 0-D (scalars)";
  if a_ndim >= 2 && b_ndim >= 2 then
    matmul_with_alloc a_shape b_shape a_orig b_orig
  else
    let a, a_shape, b, b_shape =
      match (a_ndim, b_ndim) with
      | 1, 1 ->
          ( unsqueeze ~axes:[ 0 ] a_orig,
            Array.append [| 1 |] a_shape,
            unsqueeze ~axes:[ 1 ] b_orig,
            Array.append b_shape [| 1 |] )
      | 1, _ ->
          ( unsqueeze ~axes:[ 0 ] a_orig,
            Array.append [| 1 |] a_shape,
            b_orig,
            b_shape )
      | _ ->
          ( a_orig,
            a_shape,
            unsqueeze ~axes:[ 1 ] b_orig,
            Array.append b_shape [| 1 |] )
    in
    let r = matmul_with_alloc a_shape b_shape a b in
    if a_ndim = 1 && b_ndim = 1 then squeeze r
    else if a_ndim = 1 then squeeze ~axes:[ ndim r - 2 ] r
    else squeeze ~axes:[ ndim r - 1 ] r

(* A vector operand makes [dot] the matrix product. *)
let diagonal ?(offset = 0) ?axis1 ?axis2 x =
  let nd = ndim x in
  let ax1 =
    let a = Option.value axis1 ~default:(nd - 2) in
    if a < 0 then nd + a else a
  in
  let ax2 =
    let a = Option.value axis2 ~default:(nd - 1) in
    if a < 0 then nd + a else a
  in
  if ax1 = ax2 then invalid_arg "diagonal: axes must be different";
  let perm =
    let others =
      List.filter (fun a -> a <> ax1 && a <> ax2) (List.init nd Fun.id)
    in
    others @ [ ax1; ax2 ]
  in
  let x_trans = transpose ~axes:perm x in
  let d1 = dim (nd - 2) x_trans in
  let d2 = dim (nd - 1) x_trans in
  let diag_len =
    if offset >= 0 then Stdlib.max 0 (Stdlib.min d1 (d2 - offset))
    else Stdlib.max 0 (Stdlib.min (d1 + offset) d2)
  in
  if diag_len = 0 then
    empty (Value.context x) (dtype x)
      (Array.append (Array.sub (shape x_trans) 0 (nd - 2)) [| 0 |])
  else
    let prefix = Array.sub (shape x_trans) 0 (nd - 2) in
    let x_flat = reshape (Array.append prefix [| d1 * d2 |]) x_trans in
    (* Diagonal indices: start + i*(d2+1) for i in 0..diag_len-1 *)
    let start = if offset >= 0 then offset else -offset * d2 in
    let step = d2 + 1 in
    let ctx = Value.context x in
    let idx =
      add
        (mul
           (arange ctx Nx_dtype.int64 0 diag_len 1)
           (scalar ctx Nx_dtype.int64 (Int64.of_int step)))
        (scalar ctx Nx_dtype.int64 (Int64.of_int start))
    in
    take ~axis:(nd - 2) ~indices:idx x_flat

(* [diag] for a 1-D [v]: [v] on the [k]-th diagonal of a zero [s × s] matrix,
   [s = n + |k|], by scattering into a zero template at the diagonal
   coordinates. Everything here is graph ops, so [diag] traces under jit and
   differentiates through scatter — the previous implementation read [v] back
   to the host with [to_array] and rebuilt the matrix with [init], which
   severed the tape and refused to trace. *)
let diag_construct k v =
  let n = (shape v).(0) in
  let s = n + Int.abs k in
  let ctx = Value.context v in
  let dt = dtype v in
  if n = 0 then zeros ctx dt [| s; s |]
  else
    let template = zeros ctx dt [| s; s |] in
    let i = arange ctx Nx_dtype.int64 0 n 1 in
    (* [B.scatter] needs [indices]' non-axis dimensions to match the
       template's, so [s - n] dummy rows/columns ride along scattering zeros —
       every written cell is distinct, and the dummies land where the template
       is zero anyway, so they are inert. *)
    let pad = s - n in
    let extend ~axis ~dtype tail t =
      if pad = 0 then t else concatenate ~axis [ t; zeros ctx dtype tail ]
    in
    if k >= 0 then
      (* [v_i] at [i, i+k]: one index per row along axis 1. *)
      B.scatter ~mode:`Set ~unique:true template
        ~indices:
          (extend ~axis:0 ~dtype:Nx_dtype.int64 [| pad; 1 |]
             (reshape [| n; 1 |]
                (add i (scalar ctx Nx_dtype.int64 (Int64.of_int k)))))
        ~updates:
          (extend ~axis:0 ~dtype:dt [| pad; 1 |] (reshape [| n; 1 |] v))
        ~axis:1
    else
      (* [v_j] at [j+|k|, j]: one index per column along axis 0. *)
      B.scatter ~mode:`Set ~unique:true template
        ~indices:
          (extend ~axis:1 ~dtype:Nx_dtype.int64 [| 1; pad |]
             (reshape [| 1; n |]
                (add i (scalar ctx Nx_dtype.int64 (Int64.of_int (-k))))))
        ~updates:
          (extend ~axis:1 ~dtype:dt [| 1; pad |] (reshape [| 1; n |] v))
        ~axis:0

let diag ?(k = 0) v =
  match ndim v with
  | 1 -> diag_construct k v
  | 2 -> diagonal ~offset:k v
  | n -> err "diag" "input, expected 1D or 2D array, got %dD" n

let matrix_transpose x =
  let nd = ndim x in
  if nd < 2 then x else swapaxes (nd - 2) (nd - 1) x

(* ───── Complex ───── *)

(* These compose element-wise operations rather than reading elements back to
   the host, so effectful backends can trace them and no intermediate boxes a
   [Complex.t].

   [real], [imag], [complex] and [conjugate] do no arithmetic on a component, so
   infinities, NaN and signed zeros survive: a bitcast reads a complex tensor's
   storage as the floats of its components along a last axis of two, real part
   first, and [complex] stacks two float tensors along such an axis and bitcasts
   the pairs back. Rotating the imaginary part into the real one with a complex
   multiply would poison a finite component with a non-finite one, as [inf * 0]
   is NaN. *)

(* [component k dt z] is component [k] of each element of [z], 0 for the real
   part and 1 for the imaginary one, cast to [dt]. At the float of [z]'s
   components it is a view of [z]'s storage. *)
let component (type b c) k (dt : (float, b) Nx_dtype.t) (z : (Complex.t, c) t) :
    (float, b) t =
  let lane (type e) (part : (float, e) Nx_dtype.t) =
    let pairs = bitcast part z in
    let s = shape pairs in
    let last = Array.length s - 1 in
    let one =
      Array.mapi (fun d n -> if d = last then (k, k + 1) else (0, n)) s
    in
    cast dt (reshape (Array.sub s 0 last) (shrink one pairs))
  in
  match dtype z with
  | Complex64 -> lane Nx_dtype.float32
  | Complex128 -> lane Nx_dtype.float64

let real dt z = component 0 dt z
let imag dt z = component 1 dt z
let magnitude dt (z : (Complex.t, _) t) = cast dt (abs z)
let angle dt (z : (Complex.t, _) t) = atan2 (imag dt z) (real dt z)

let complex (type c) (dt : (Complex.t, c) Nx_dtype.t) ~re ~im =
  let re, im = broadcasted re im in
  let pairs (type e) (part : (float, e) Nx_dtype.t) : (Complex.t, c) t =
    bitcast dt (stack ~axis:(-1) [ cast part re; cast part im ])
  in
  match dt with
  | Complex64 -> pairs Nx_dtype.float32
  | Complex128 -> pairs Nx_dtype.float64

let conjugate (type a b) (x : (a, b) t) : (a, b) t =
  let negate_imag (type c e) (part : (float, e) Nx_dtype.t)
      (z : (Complex.t, c) t) : (Complex.t, c) t =
    complex (dtype z) ~re:(real part z) ~im:(neg (imag part z))
  in
  match dtype x with
  | Complex64 -> negate_imag Nx_dtype.float32 x
  | Complex128 -> negate_imag Nx_dtype.float64 x
  | _ -> x

(* ───── Dot Products and Tensor Contractions ───── *)

let tensordot ?axes a b =
  match axes with
  | None -> matmul a b
  | Some (axes_a, axes_b) ->
      let n_axes = List.length axes_a in
      if n_axes <> List.length axes_b then
        invalid_arg "tensordot: axes lists must have same length";
      let ndim_a = ndim a in
      let ndim_b = ndim b in
      let axes_a =
        Array.of_list
          (List.map (fun ax -> if ax < 0 then ndim_a + ax else ax) axes_a)
      in
      let axes_b =
        Array.of_list
          (List.map (fun ax -> if ax < 0 then ndim_b + ax else ax) axes_b)
      in
      let sa = shape a in
      let sb = shape b in
      Array.iter2
        (fun ax_a ax_b ->
          if sa.(ax_a) <> sb.(ax_b) then
            invalid_arg "tensordot: axes have different sizes")
        axes_a axes_b;
      let axes_a_set =
        Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_a
      in
      let axes_b_set =
        Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_b
      in
      let free_a =
        Array.of_list
          (List.filter
             (fun i -> not (IntSet.mem i axes_a_set))
             (List.init ndim_a Fun.id))
      in
      let free_b =
        Array.of_list
          (List.filter
             (fun i -> not (IntSet.mem i axes_b_set))
             (List.init ndim_b Fun.id))
      in
      let perm_a = Array.append free_a axes_a in
      let perm_b = Array.append axes_b free_b in
      let do_transpose perm t =
        if Array.length perm > 1 then
          contiguous (transpose ~axes:(Array.to_list perm) t)
        else t
      in
      let at = do_transpose perm_a a in
      let bt = do_transpose perm_b b in
      let sat = shape at in
      let sbt = shape bt in
      let nfa = Array.length free_a in
      let nfb = Array.length free_b in
      let prod arr = Array.fold_left ( * ) 1 arr in
      let free_size_a = if nfa = 0 then 1 else prod (Array.sub sat 0 nfa) in
      let free_size_b =
        if nfb = 0 then 1 else prod (Array.sub sbt n_axes (ndim_b - n_axes))
      in
      let contract_size = prod (Array.sub sat nfa n_axes) in
      let r =
        matmul
          (reshape [| free_size_a; contract_size |] at)
          (reshape [| contract_size; free_size_b |] bt)
      in
      let result_shape =
        Array.append
          (if nfa = 0 then [||] else Array.sub sat 0 nfa)
          (if nfb = 0 then [||] else Array.sub sbt n_axes (ndim_b - n_axes))
      in
      if Array.length result_shape = 0 then squeeze r
      else reshape result_shape r

let dot x w =
  let x_ndim = ndim x and w_ndim = ndim w in
  if not (x_ndim > 0 && w_ndim > 0) then
    invalid_arg "dot: tensors, both must be at least 1D";
  (* The last axis of [x] against the only axis of [w], or its second to last;
     the other axes of both are kept, [x]'s first. *)
  tensordot ~axes:([ x_ndim - 1 ], [ Stdlib.max 0 (w_ndim - 2) ]) x w

module Einsum = struct
  type token = Axis of char | Ellipsis

  let parse_operand str =
    let len = String.length str in
    if len = 0 then []
    else
      let rec loop idx acc ell =
        if idx >= len then List.rev acc
        else
          match str.[idx] with
          | '.' ->
              if
                idx + 2 >= len || str.[idx + 1] <> '.' || str.[idx + 2] <> '.'
              then invalid_arg "einsum: ellipsis must be '...'";
              if ell then invalid_arg "einsum: multiple ellipsis in operand";
              loop (idx + 3) (Ellipsis :: acc) true
          | c
            when (c >= 'a' && c <= 'z')
                 || (c >= 'A' && c <= 'Z')
                 || (c >= '0' && c <= '9')
                 || c = '_' ->
              loop (idx + 1) (Axis c :: acc) ell
          | c ->
              invalid_arg (Printf.sprintf "einsum: invalid character '%c'" c)
      in
      loop 0 [] false

  let parse_equation subscripts =
    let parts = String.split_on_char '-' subscripts in
    match parts with
    | [ lhs; rhs ] when String.length rhs > 0 && rhs.[0] = '>' ->
        let inputs =
          String.split_on_char ',' lhs
          |> List.map String.trim
          |> List.filter (( <> ) "")
        in
        let output = String.trim (String.sub rhs 1 (String.length rhs - 1)) in
        ( Array.of_list (List.map parse_operand inputs),
          Some (parse_operand output) )
    | [ lhs ] ->
        let inputs =
          String.split_on_char ',' lhs
          |> List.map String.trim
          |> List.filter (( <> ) "")
        in
        (Array.of_list (List.map parse_operand inputs), None)
    | _ -> invalid_arg "einsum: invalid format, expected inputs->output"

  let handle_repeated_indices tensor tokens =
    let rec find_dups acc idx = function
      | [] -> None
      | Axis c :: rest -> (
          match List.find_opt (fun (ch, _) -> ch = c) acc with
          | Some (_, prev) -> Some (prev, idx, c)
          | None -> find_dups ((c, idx) :: acc) (idx + 1) rest)
      | Ellipsis :: rest -> find_dups acc (idx + 1) rest
    in
    let rec process t toks =
      match find_dups [] 0 toks with
      | None -> (t, toks)
      | Some (ax1, ax2, c) ->
          let s = shape t in
          if s.(ax1) <> s.(ax2) then
            invalid_arg
              (Printf.sprintf
                 "einsum: index var '%c' must have consistent dimensions (%d \
                  vs %d)"
                 c s.(ax1) s.(ax2));
          let t' = diagonal ~axis1:ax1 ~axis2:ax2 t in
          (* [diagonal] appends the diagonal axis after the remaining axes. *)
          let toks' =
            List.filteri (fun i _ -> i <> ax1 && i <> ax2) toks @ [ Axis c ]
          in
          process t' toks'
    in
    process tensor tokens

  type tensor_info = { id : int; shape : int array; axis_labels : char list }

  type contraction_path =
    | Leaf of int
    | Node of contraction_path * contraction_path * tensor_info

  let estimate_cost (t1 : tensor_info) (t2 : tensor_info) common_chars =
    let dim_map = Hashtbl.create 16 in
    List.iteri
      (fun i c -> Hashtbl.replace dim_map c t1.shape.(i))
      t1.axis_labels;
    List.iteri
      (fun i c -> Hashtbl.replace dim_map c t2.shape.(i))
      t2.axis_labels;
    let all = List.sort_uniq Char.compare (t1.axis_labels @ t2.axis_labels) in
    let output_size =
      List.fold_left
        (fun acc c ->
          if List.mem c common_chars then acc
          else acc * Hashtbl.find dim_map c)
        1 all
    in
    let op_cost =
      List.fold_left (fun acc c -> acc * Hashtbl.find dim_map c) 1 all
    in
    (float_of_int op_cost, float_of_int output_size)

  let optimize_path inputs output_chars =
    let workset = ref (List.mapi (fun i t -> (Leaf i, t)) inputs) in
    let contract_info (p1, t1) (p2, t2) =
      let common =
        List.filter (fun c -> List.mem c t2.axis_labels) t1.axis_labels
      in
      let new_labels =
        let all =
          List.sort_uniq Char.compare (t1.axis_labels @ t2.axis_labels)
        in
        List.filter
          (fun c -> (not (List.mem c common)) || List.mem c output_chars)
          all
      in
      let find_index x lst =
        let rec aux i = function
          | [] -> raise Not_found
          | h :: _ when h = x -> i
          | _ :: t -> aux (i + 1) t
        in
        aux 0 lst
      in
      let get_dim c =
        if List.mem c t1.axis_labels then
          t1.shape.(find_index c t1.axis_labels)
        else t2.shape.(find_index c t2.axis_labels)
      in
      let new_shape = Array.of_list (List.map get_dim new_labels) in
      let info = { id = -1; shape = new_shape; axis_labels = new_labels } in
      let cost, size =
        estimate_cost t1 t2
          (List.filter (fun c -> not (List.mem c new_labels)) common)
      in
      (cost, size, Node (p1, p2, info), info)
    in
    while List.length !workset > 1 do
      let items = !workset in
      let best = ref None in
      let min_cost = ref Float.infinity in
      let rec iter_pairs = function
        | [] -> ()
        | x :: rest ->
            List.iter
              (fun y ->
                let cost, _, path, info = contract_info x y in
                if cost < !min_cost then (
                  min_cost := cost;
                  best := Some (x, y, path, info)))
              rest;
            iter_pairs rest
      in
      iter_pairs items;
      match !best with
      | None -> err "einsum" "could not find valid contraction"
      | Some (i1, i2, new_path, new_info) ->
          workset :=
            (new_path, new_info)
            :: List.filter (fun x -> x != i1 && x != i2) items
    done;
    match !workset with
    | [ (p, _) ] -> p
    | _ -> err "einsum" "optimization failed"

  let contract_pair op_a str_a op_b str_b result_str =
    let sa = shape op_a in
    let sb = shape op_b in
    let chars_a = String.to_seq str_a |> List.of_seq in
    let chars_b = String.to_seq str_b |> List.of_seq in
    let chars_out = String.to_seq result_str |> List.of_seq in
    let batch_chars =
      List.filter
        (fun c -> List.mem c chars_b && List.mem c chars_out)
        chars_a
    in
    let contract_chars =
      List.filter
        (fun c -> List.mem c chars_b && not (List.mem c chars_out))
        chars_a
    in
    let a_free = List.filter (fun c -> not (List.mem c chars_b)) chars_a in
    let b_free = List.filter (fun c -> not (List.mem c chars_a)) chars_b in
    let get_axes source target =
      List.map
        (fun c ->
          let rec find i = function
            | [] -> err "einsum" "index %c not found in operand subscripts" c
            | x :: _ when x = c -> i
            | _ :: xs -> find (i + 1) xs
          in
          find 0 source)
        target
    in
    let perm_a = get_axes chars_a (batch_chars @ a_free @ contract_chars) in
    let perm_b = get_axes chars_b (batch_chars @ contract_chars @ b_free) in
    let is_identity perm n =
      let rec check i = function
        | [] -> i = n
        | x :: xs -> x = i && check (i + 1) xs
      in
      check 0 perm
    in
    let at =
      if is_identity perm_a (String.length str_a) then op_a
      else contiguous (transpose ~axes:perm_a op_a)
    in
    let bt =
      if is_identity perm_b (String.length str_b) then op_b
      else contiguous (transpose ~axes:perm_b op_b)
    in
    let prod dims = Array.fold_left ( * ) 1 dims in
    let pa = Array.of_list perm_a in
    let pb = Array.of_list perm_b in
    let nb = List.length batch_chars in
    let naf = List.length a_free in
    let nc = List.length contract_chars in
    let nbf = List.length b_free in
    let batch_dims =
      Array.init nb (fun i ->
          let da = sa.(pa.(i)) in
          let db = sb.(pb.(i)) in
          if da = db then da
          else if da = 1 then db
          else if db = 1 then da
          else
            invalid_arg
              (Printf.sprintf
                 "einsum: incompatible broadcast dimensions (%d vs %d)" da db))
    in
    let a_free_dims = Array.init naf (fun i -> sa.(pa.(nb + i))) in
    let contract_dims = Array.init nc (fun i -> sa.(pa.(nb + naf + i))) in
    let b_free_dims = Array.init nbf (fun i -> sb.(pb.(nb + nc + i))) in
    let bs = prod batch_dims in
    let m = prod a_free_dims in
    let k = prod contract_dims in
    let n = prod b_free_dims in
    let broadcast_batch tensor parr src_shape =
      if nb = 0 then tensor
      else
        let needs = ref false in
        let target =
          Array.init (ndim tensor) (fun i ->
              if i < nb then (
                let src = src_shape.(parr.(i)) in
                let tgt = batch_dims.(i) in
                if src <> tgt then needs := true;
                tgt)
              else src_shape.(parr.(i)))
        in
        if !needs then broadcast_to target tensor else tensor
    in
    let at = broadcast_batch at pa sa in
    let bt = broadcast_batch bt pb sb in
    let r = matmul (reshape [| bs; m; k |] at) (reshape [| bs; k; n |] bt) in
    let intermediate =
      reshape (Array.concat [ batch_dims; a_free_dims; b_free_dims ]) r
    in
    let inter_chars = batch_chars @ a_free @ b_free in
    if inter_chars = chars_out then intermediate
    else transpose ~axes:(get_axes inter_chars chars_out) intermediate

  let calculate subscripts operands =
    let n_ops = Array.length operands in
    if n_ops = 0 then invalid_arg "einsum: no input operands";
    match (subscripts, n_ops) with
    | "i,i->", 2 -> dot operands.(0) operands.(1)
    | "ij,jk->ik", 2 -> matmul operands.(0) operands.(1)
    | "ij->ji", 1 -> transpose operands.(0)
    | _ ->
        let input_tokens, output_opt = parse_equation subscripts in
        if Array.length input_tokens <> n_ops then
          invalid_arg "einsum: number of inputs must equal number of operands";
        let ell_rank =
          let max_rank = ref 0 in
          for i = 0 to n_ops - 1 do
            let n_named =
              List.length
                (List.filter
                   (function Axis _ -> true | _ -> false)
                   input_tokens.(i))
            in
            let r = ndim operands.(i) - n_named in
            if r < 0 then
              invalid_arg "einsum: operand rank too small for subscripts";
            if r > !max_rank then max_rank := r
          done;
          !max_rank
        in
        let get_ell_char i = char_of_int (200 + i) in
        let normalized_inputs =
          Array.mapi
            (fun i tokens ->
              let op = operands.(i) in
              let n_named =
                List.length
                  (List.filter
                     (function Axis _ -> true | _ -> false)
                     tokens)
              in
              let ell_dim = ndim op - n_named in
              let expanded =
                List.concat_map
                  (function
                    | Axis c -> [ Axis c ]
                    | Ellipsis ->
                        List.init ell_dim (fun k ->
                            Axis (get_ell_char (ell_rank - ell_dim + k))))
                  tokens
              in
              let op_diag, final = handle_repeated_indices op expanded in
              let chars =
                List.map (function Axis c -> c | _ -> assert false) final
              in
              ({ id = i; shape = shape op_diag; axis_labels = chars }, op_diag))
            input_tokens
        in
        let ops_info = Array.map fst normalized_inputs in
        let ops_tensors = Array.map snd normalized_inputs in
        (* Validate dimension consistency *)
        let char_dims = Hashtbl.create 16 in
        Array.iter
          (fun info ->
            List.iteri
              (fun idx c ->
                let d = info.shape.(idx) in
                match Hashtbl.find_opt char_dims c with
                | None -> Hashtbl.add char_dims c d
                | Some prev ->
                    if prev <> d && prev <> 1 && d <> 1 then
                      invalid_arg
                        (Printf.sprintf
                           "einsum: index var '%c' must have consistent \
                            dimensions (%d vs %d)"
                           c prev d)
                    else if d > prev then Hashtbl.replace char_dims c d)
              info.axis_labels)
          ops_info;
        let inputs_have_ell =
          Array.exists
            (fun toks -> List.exists (( = ) Ellipsis) toks)
            input_tokens
        in
        let target_chars =
          match output_opt with
          | Some tokens ->
              if List.exists (( = ) Ellipsis) tokens && not inputs_have_ell
              then
                invalid_arg
                  "einsum: output ellipsis requires ellipsis in inputs";
              List.concat_map
                (function
                  | Axis c -> [ c ]
                  | Ellipsis -> List.init ell_rank (fun k -> get_ell_char k))
                tokens
          | None ->
              let all_chars =
                List.concat
                  (Array.to_list
                     (Array.map
                        (fun toks ->
                          List.filter_map
                            (function Axis c -> Some c | Ellipsis -> None)
                            toks)
                        input_tokens))
              in
              let counts = Hashtbl.create 16 in
              List.iter
                (fun c ->
                  Hashtbl.replace counts c
                    (1
                    + (Hashtbl.find_opt counts c |> Option.value ~default:0)))
                all_chars;
              let ell_chars = List.init ell_rank (fun k -> get_ell_char k) in
              let named =
                List.filter (fun c -> int_of_char c < 200) all_chars
                |> List.sort_uniq Char.compare
                |> List.filter (fun c -> Hashtbl.find counts c = 1)
              in
              ell_chars @ named
        in
        let all_input_chars =
          Array.fold_left (fun acc info -> acc @ info.axis_labels) [] ops_info
        in
        List.iter
          (fun c ->
            if not (List.mem c all_input_chars) then
              invalid_arg
                (Printf.sprintf
                   "einsum: output index '%c' not found in inputs" c))
          target_chars;
        (* Pre-reduce single-operand axes absent from output *)
        Array.iteri
          (fun i info ->
            let reduce_axes = ref [] in
            let new_labels = ref [] in
            let char_count = Hashtbl.create 16 in
            Array.iter
              (fun inf ->
                List.iter
                  (fun c ->
                    Hashtbl.replace char_count c
                      (1
                      + (Hashtbl.find_opt char_count c
                        |> Option.value ~default:0)))
                  inf.axis_labels)
              ops_info;
            List.iteri
              (fun axis_idx c ->
                if
                  Hashtbl.find char_count c = 1
                  && not (List.mem c target_chars)
                then reduce_axes := axis_idx :: !reduce_axes
                else new_labels := c :: !new_labels)
              info.axis_labels;
            match !reduce_axes with
            | [] -> ()
            | axes ->
                ops_tensors.(i) <- sum ~axes:(List.rev axes) ops_tensors.(i);
                ops_info.(i) <-
                  {
                    info with
                    shape = shape ops_tensors.(i);
                    axis_labels = List.rev !new_labels;
                  })
          ops_info;
        let finalize result current_chars =
          let reduce =
            List.filter_map
              (fun (i, c) ->
                if not (List.mem c target_chars) then Some i else None)
              (List.mapi (fun i c -> (i, c)) current_chars)
          in
          let result =
            if reduce = [] then result else sum ~axes:reduce result
          in
          let final =
            List.filter (fun c -> List.mem c target_chars) current_chars
          in
          if final = target_chars then result
          else
            let perm =
              List.map
                (fun c ->
                  let rec find i = function
                    | [] -> 0
                    | x :: xs -> if x = c then i else find (i + 1) xs
                  in
                  find 0 final)
                target_chars
            in
            transpose ~axes:perm result
        in
        if n_ops = 1 then finalize ops_tensors.(0) ops_info.(0).axis_labels
        else if n_ops = 2 then
          let ia = ops_info.(0) in
          let ib = ops_info.(1) in
          let stra = ia.axis_labels |> List.to_seq |> String.of_seq in
          let strb = ib.axis_labels |> List.to_seq |> String.of_seq in
          let common =
            List.filter (fun c -> List.mem c ib.axis_labels) ia.axis_labels
          in
          let result_labels =
            List.sort_uniq Char.compare (ia.axis_labels @ ib.axis_labels)
            |> List.filter (fun c ->
                (not (List.mem c common)) || List.mem c target_chars)
          in
          let str_out = result_labels |> List.to_seq |> String.of_seq in
          finalize
            (contract_pair ops_tensors.(0) stra ops_tensors.(1) strb str_out)
            result_labels
        else
          let plan = optimize_path (Array.to_list ops_info) target_chars in
          let rec execute = function
            | Leaf idx ->
                ( ops_tensors.(idx),
                  ops_info.(idx).axis_labels |> List.to_seq |> String.of_seq
                )
            | Node (left, right, info) ->
                let ra, sa = execute left in
                let rb, sb = execute right in
                let so = info.axis_labels |> List.to_seq |> String.of_seq in
                (contract_pair ra sa rb sb so, so)
          in
          let result, rstr = execute plan in
          finalize result (String.to_seq rstr |> List.of_seq)
end

let vdot (type a b) (a : (a, b) t) (b : (a, b) t) =
  let a', b' =
    try
      let bc = broadcast_arrays [ a; b ] in
      (contiguous (List.nth bc 0), contiguous (List.nth bc 1))
    with _ -> (a, b)
  in
  let fa = flatten a' in
  let fb = flatten b' in
  if numel fa <> numel fb then
    invalid_arg "vdot: different number of elements";
  match dtype a with
  | (Complex64 | Complex128) when dtype a = dtype b ->
      matmul (conjugate fa) fb
  | _ -> matmul fa fb

(* Each pair of vectors along [axis] is a row times a column, so the broadcast
   operands contract as a batch of matrix products. *)
let vecdot ?axis x1 x2 =
  let ax =
    match axis with
    | None -> ndim x1 - 1
    | Some a -> if a < 0 then ndim x1 + a else a
  in
  let target = Shape.broadcast (shape x1) (shape x2) in
  let n = Array.length target in
  let ax = ax + n - ndim x1 in
  let x1 = broadcast_to target x1 and x2 = broadcast_to target x2 in
  let row = unsqueeze ~axes:[ n - 1 ] (moveaxis ax (-1) x1) in
  let column = unsqueeze ~axes:[ n ] (moveaxis ax (-1) x2) in
  squeeze ~axes:[ n - 1; n ] (matmul row column)

let inner a b =
  if (shape a).(ndim a - 1) <> (shape b).(ndim b - 1) then
    invalid_arg "inner: last dimensions differ";
  tensordot ~axes:([ ndim a - 1 ], [ ndim b - 1 ]) a b

let outer a b =
  let fa = if ndim a = 0 then reshape [| 1 |] a else flatten a in
  let fb = if ndim b = 0 then reshape [| 1 |] b else flatten b in
  let r =
    matmul (reshape [| numel fa; 1 |] fa) (reshape [| 1; numel fb |] fb)
  in
  let r = if ndim a = 0 then squeeze ~axes:[ 0 ] r else r in
  if ndim b = 0 then squeeze ~axes:[ (if ndim a = 0 then 0 else 1) ] r else r

let einsum subscripts operands = Einsum.calculate subscripts operands

let kron a b =
  let sa = shape a in
  let sb = shape b in
  let a2 = if ndim a = 1 then reshape [| sa.(0); 1 |] a else a in
  let b2 = if ndim b = 1 then reshape [| sb.(0); 1 |] b else b in
  let sa2 = shape a2 in
  let sb2 = shape b2 in
  let r =
    mul
      (reshape [| sa2.(0); 1; sa2.(1); 1 |] a2)
      (reshape [| 1; sb2.(0); 1; sb2.(1) |] b2)
  in
  let flat = reshape [| sa2.(0) * sb2.(0); sa2.(1) * sb2.(1) |] r in
  if ndim a = 1 && ndim b = 1 then flatten flat else flat

let multi_dot arrays =
  match arrays with
  | [||] -> invalid_arg "multi_dot: empty array"
  | [| arr |] -> arr
  | _ ->
      let n = Array.length arrays in
      let dims = Array.make (n + 1) 0 in
      let matrix_dims idx =
        let t = arrays.(idx) in
        match ndim t with
        | 1 ->
            let len = (shape t).(0) in
            if idx = 0 then (1, len)
            else if idx = n - 1 then (len, 1)
            else
              invalid_arg
                "multi_dot: only first and last arguments may be 1D vectors"
        | 2 ->
            let s = shape t in
            (s.(0), s.(1))
        | _ ->
            invalid_arg
              (Printf.sprintf
                 "multi_dot: argument %d must be 1D (endpoints) or 2D matrix"
                 idx)
      in
      for i = 0 to n - 1 do
        let rows, cols = matrix_dims i in
        if i = 0 then dims.(0) <- rows
        else if dims.(i) <> rows then
          invalid_arg
            (Printf.sprintf
               "multi_dot: shapes not aligned between arguments %d and %d \
                (%d <> %d)"
               (i - 1) i dims.(i) rows);
        dims.(i + 1) <- cols
      done;
      (* MCM dynamic programming *)
      let d64 = Array.map Int64.of_int dims in
      let cost = Array.make_matrix n n Int64.zero in
      let split = Array.make_matrix n n 0 in
      for len = 2 to n do
        for i = 0 to n - len do
          let j = i + len - 1 in
          let best_c = ref Int64.max_int in
          let best_s = ref i in
          for k = i to j - 1 do
            let c =
              Int64.(
                add
                  cost.(i).(k)
                  (add
                     cost.(k + 1).(j)
                     (mul d64.(i) (mul d64.(k + 1) d64.(j + 1)))))
            in
            if c < !best_c then (
              best_c := c;
              best_s := k)
          done;
          cost.(i).(j) <- !best_c;
          split.(i).(j) <- !best_s
        done
      done;
      let memo = Array.init n (fun _ -> Array.make n None) in
      let rec compute i j =
        match memo.(i).(j) with
        | Some t -> t
        | None ->
            let r =
              if i = j then arrays.(i)
              else
                matmul
                  (compute i split.(i).(j))
                  (compute (split.(i).(j) + 1) j)
            in
            memo.(i).(j) <- Some r;
            r
      in
      compute 0 (n - 1)

let cross ?axis a b =
  let axis =
    let ax = Option.value axis ~default:(-1) in
    if ax < 0 then ndim a + ax else ax
  in
  if axis >= ndim a then invalid_arg "cross: axis out of bounds";
  if (shape a).(axis) <> 3 then invalid_arg "cross: axis dim not 3";
  if (shape b).(axis) <> 3 then invalid_arg "cross: axis dim not 3";
  let at i t =
    squeeze ~axes:[ axis ]
      (slice
         (Array.to_list
            (Array.init (ndim t) (fun j ->
                 if j = axis then R (i, i + 1) else A)))
         t)
  in
  let c1 = sub (mul (at 1 a) (at 2 b)) (mul (at 2 a) (at 1 b)) in
  let c2 = sub (mul (at 2 a) (at 0 b)) (mul (at 0 a) (at 2 b)) in
  let c3 = sub (mul (at 0 a) (at 1 b)) (mul (at 1 a) (at 0 b)) in
  stack ~axis [ c1; c2; c3 ]

(* ───── Matrix Decompositions and Solving ───── *)

let check_square ~op a =
  let sh = shape a in
  let n = Array.length sh in
  if n < 2 then err op "input requires at least 2D array";
  if sh.(n - 1) <> sh.(n - 2) then
    invalid_arg (Printf.sprintf "%s: coefficient matrix must be square" op)

let check_float_or_complex (type a b) ~op (a : (a, b) t) =
  match dtype a with
  | Float16 | Float32 | Float64 | Complex64 | Complex128 -> ()
  | _ -> err op "dtype must be float or complex"

let cholesky ?upper a =
  check_square ~op:"cholesky" a;
  check_float_or_complex ~op:"cholesky" a;
  B.cholesky ~upper:(Option.value upper ~default:false) a

let qr ?mode a =
  check_float_or_complex ~op:"qr" a;
  let reduced =
    match mode with None | Some `Reduced -> true | Some `Complete -> false
  in
  B.qr ~reduced a

let lu a =
  check_float_or_complex ~op:"lu" a;
  let sh = shape a in
  let nd = Array.length sh in
  if nd < 2 then err "lu" "input requires at least 2D array";
  let m = sh.(nd - 2) and n = sh.(nd - 1) in
  let k = Int.min m n in
  let packed, _, perm = B.lu a in
  let ctx = Value.context a and dt = dtype a in
  let batch = List.init (nd - 2) (fun _ -> A) in
  let l =
    add
      (tril ~k:(-1) (slice (batch @ [ A; R (0, k) ]) packed))
      (eye ctx ~m:k dt m)
  in
  let u = triu (slice (batch @ [ R (0, k); A ]) packed) in
  (perm, l, u)

let svd ?full_matrices a =
  check_float_or_complex ~op:"svd" a;
  B.svd ~full_matrices:(Option.value full_matrices ~default:false) a

let svdvals a =
  check_float_or_complex ~op:"svdvals" a;
  let _, s, _ = B.svd ~full_matrices:false a in
  s

let eig a =
  check_square ~op:"eig" a;
  check_float_or_complex ~op:"eig" a;
  B.eig a

(* The kernels read the lower triangle. The lower triangle of aᴴ is the
   Hermitian matrix that a's upper triangle names. *)
let hermitian_lower ?(uplo = `L) a =
  match uplo with `L -> a | `U -> conjugate (matrix_transpose a)

let eigh ?uplo a =
  check_square ~op:"eigh" a;
  check_float_or_complex ~op:"eigh" a;
  B.eigh (hermitian_lower ?uplo a)

let eigvals a =
  check_square ~op:"eigvals" a;
  check_float_or_complex ~op:"eigvals" a;
  B.eigvals a

let eigvalsh ?uplo a =
  check_square ~op:"eigvalsh" a;
  check_float_or_complex ~op:"eigvalsh" a;
  B.eigvalsh (hermitian_lower ?uplo a)

let norm (type a b) ?ord ?axes ?keepdims (x : (a, b) t) =
  let keepdims = Option.value keepdims ~default:false in
  match (ord, axes) with
  | None, None -> sqrt (sum (square (abs x)) ~keepdims)
  | None, Some _ | Some `Fro, _ -> sqrt (sum (square (abs x)) ?axes ~keepdims)
  | Some `One, None ->
      max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
  | Some `NegOne, None ->
      if ndim x = 1 then min (abs x) ~keepdims
      else min (sum (abs x) ~axes:[ ndim x - 2 ]) ~keepdims
  | Some `Two, None when ndim x = 1 -> sqrt (sum (square (abs x)) ~keepdims)
  | Some `Two, None -> max (svdvals x |> cast (dtype x)) ~keepdims
  | Some `NegTwo, None -> min (svdvals x |> cast (dtype x)) ~keepdims
  | Some `Inf, None ->
      if ndim x = 1 then max (abs x) ~keepdims
      else max (sum (abs x) ~axes:[ ndim x - 1 ] ~keepdims) ~keepdims
  | Some `NegInf, None ->
      if ndim x = 1 then min (abs x) ~keepdims
      else min (sum (abs x) ~axes:[ ndim x - 1 ]) ~keepdims
  | Some `Nuc, None ->
      if ndim x < 2 then
        invalid_arg "norm: input, nuclear norm defined for matrices";
      sum (svdvals x |> cast (dtype x)) ~keepdims
  | Some `NegOne, _ | Some `NegTwo, _ | Some `NegInf, _ | Some `Nuc, _ ->
      invalid_arg "norm: this combination of ord and axis not implemented"
  | Some (`P p), _ ->
      if p = 1.0 && axes = None && ndim x = 2 then
        max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
      else
        let p_t =
          full (Value.context x) (dtype x) [||] (Nx_dtype.of_float (dtype x) p)
        in
        let inv_p =
          div
            (full (Value.context x) (dtype x) [||] (Nx_dtype.one (dtype x)))
            p_t
        in
        pow (sum (pow (abs x) p_t) ?axes ~keepdims) inv_p
  | _ -> invalid_arg "norm: this combination of ord and axis not implemented"

(* +1 for a pivot that kept its row, -1 for one that exchanged it. *)
let pivot_signs dt pivots =
  let one = ones (Value.context pivots) dt (shape pivots) in
  let steps = arange (Value.context pivots) int64 0 (dim (-1) pivots) 1 in
  where (not_equal pivots steps) (neg one) one

let det a =
  check_square ~op:"det" a;
  check_float_or_complex ~op:"det" a;
  at_float32
    {
      f =
        (fun a ->
          let packed, pivots, _ = B.lu a in
          mul
            (prod (diagonal packed) ~axes:[ -1 ])
            (prod (pivot_signs (dtype a) pivots) ~axes:[ -1 ]));
    }
    a

let slogdet (type a b) (a : (a, b) t) : (a, b) t * (float, float64_elt) t =
  check_square ~op:"slogdet" a;
  check_float_or_complex ~op:"slogdet" a;
  let factored (type c d) (a : (c, d) t) =
    let packed, pivots, _ = B.lu a in
    let d = diagonal packed in
    let mag = abs d in
    let unit =
      div d (where (cmpeq mag (zeros_like mag)) (ones_like mag) mag)
    in
    let sign =
      mul (prod unit ~axes:[ -1 ])
        (prod (pivot_signs (dtype a) pivots) ~axes:[ -1 ])
    in
    (sign, sum (log (cast float64 mag)) ~axes:[ -1 ])
  in
  match dtype a with
  | Float16 ->
      let sign, logabs = factored (cast float32 a) in
      (cast (dtype a) sign, logabs)
  | _ -> factored a

let matrix_rank' ~by ?tol ?rtol ?hermitian a =
  check_float_or_complex ~op:"matrix_rank" a;
  (match hermitian with
  | Some true -> check_square ~op:"matrix_rank" a
  | None | Some false -> ());
  let s =
    match hermitian with Some true -> abs (B.eigvalsh a) | _ -> svdvals a
  in
  let max_s = max s |> read_item ~by in
  let sh = shape a in
  let m = sh.(Array.length sh - 2) in
  let n = sh.(Array.length sh - 1) in
  let eps =
    let dt = dtype a in
    if
      Nx_dtype.equal dt Nx_dtype.float32
      || Nx_dtype.equal dt Nx_dtype.complex64
    then 1.2e-7
    else if
      Nx_dtype.equal dt Nx_dtype.float64
      || Nx_dtype.equal dt Nx_dtype.complex128
    then 2.2e-16
    else 1e-15
  in
  let tol =
    match (tol, rtol) with
    | Some t, _ -> t
    | None, Some r -> r *. max_s
    | None, None -> float_of_int (Stdlib.max m n) *. eps *. max_s
  in
  let mask = greater s (scalar (Value.context a) (dtype s) tol) in
  int_of_float (Float.round (sum (cast (dtype s) mask) |> read_item ~by))

let matrix_rank ?tol ?rtol ?hermitian a =
  matrix_rank' ~by:"Nx.matrix_rank" ?tol ?rtol ?hermitian a

let trace ?offset a =
  if ndim a < 2 then invalid_arg "trace: input requires at least 2D array";
  sum
    (diagonal ~offset:(Option.value offset ~default:0) a)
    ~axes:[ -1 ] ~keepdims:false

(* [solve_triangular ?upper ?transpose ?unit_diag a b] exploits triangularity
   instead of factoring: only the [upper] triangle of [a] is read (the other
   one is silently ignored, not checked), [transpose] solves the transposed
   system (conjugate transpose for complex [a]), and [unit_diag] assumes a
   diagonal of ones without reading it. *)
let solve_triangular ?(upper = false) ?(transpose = false)
    ?(unit_diag = false) a b =
  let op = "solve_triangular" in
  check_square ~op a;
  check_float_or_complex ~op a;
  check_float_or_complex ~op b;
  if not (Nx_dtype.equal (dtype a) (dtype b)) then
    err op "a and b must have the same dtype";
  let sh_a = shape a in
  let n = sh_a.(Array.length sh_a - 1) in
  let batch = Array.sub sh_a 0 (Array.length sh_a - 2) in
  let sb = shape b in
  (* [leading] is how many leading dimensions of [b] are batch. *)
  let check_batch what leading =
    if leading <> Array.length batch then
      err op "%s must share the batch dimensions of a" what;
    Array.iteri
      (fun i v ->
        if sb.(i) <> v then
          err op "%s batch dimension %d does not match a" what i)
      batch
  in
  if Array.length sb = Array.length sh_a - 1 then begin
    (* Vector right-hand side [batch, n]. *)
    check_batch "vector right-hand side" (Array.length sb - 1);
    if sb.(Array.length sb - 1) <> n then
      err op "vector right-hand side length does not match a"
  end
  else begin
    (* Matrix right-hand side [batch, n, nrhs]. *)
    if Array.length sb <> Array.length sh_a then
      err op "b must be a [.., n] vector or a [.., n, nrhs] matrix";
    check_batch "matrix right-hand side" (Array.length sb - 2);
    if sb.(Array.length sb - 2) <> n then
      err op "matrix right-hand side row count does not match a"
  end;
  B.solve_triangular ~upper ~transpose ~unit_diag a b

let solve a b =
  check_square ~op:"solve" a;
  check_float_or_complex ~op:"solve" a;
  check_float_or_complex ~op:"solve" b;
  let b_expanded =
    if ndim a > 2 && ndim b = 2 then
      let sa = shape a in
      let sb = shape b in
      let batch = array_prod (Array.sub sa 0 (ndim a - 2)) in
      if sb.(0) = batch && sb.(1) = sa.(ndim a - 2) then expand_dims [ -1 ] b
      else b
    else b
  in
  let packed, _, perm = B.lu a in
  (* Row i of L U is row perm[i] of a, so L U x = b[perm]. A right-hand side
     of one axis is a vector; any other shares a's batch as a stack of
     matrices. *)
  let sa = shape a in
  let n = sa.(Array.length sa - 1) in
  let batch = Array.sub sa 0 (Array.length sa - 2) in
  let vector = ndim b_expanded = 1 in
  let rows = if vector then [| n |] else [| n; dim (-1) b_expanded |] in
  let target = Array.append batch rows in
  let pb =
    take_along_axis
      ~axis:(if vector then -1 else -2)
      ~indices:
        (broadcast_to target
           (if vector then perm else expand_dims [ -1 ] perm))
      (broadcast_to target b_expanded)
  in
  (* A pivot below tolerance makes the system singular. Zeroing its row of U
     keeps the check in the graph: the triangular solve then reports
     [`Singular] itself, and a compiled program yields infinities instead. *)
  let u =
    let pivots = abs (diagonal packed) |> cast Nx_dtype.float64 in
    let m = dim (-2) a in
    let eps =
      if Nx_dtype.equal (dtype a) Nx_dtype.float32 then 1e-6 else 1e-12
    in
    let tol_t =
      full (Value.context pivots) Nx_dtype.float64 (shape pivots)
        (eps *. float_of_int m)
    in
    where (expand_dims [ -1 ] (less pivots tol_t)) (zeros_like packed) packed
  in
  let result =
    try
      B.solve_triangular ~upper:true ~transpose:false ~unit_diag:false u
        (B.solve_triangular ~upper:false ~transpose:false ~unit_diag:true
           packed pb)
    with Nx_backend.Linalg_error { kind; _ } ->
      raise (Nx_backend.Linalg_error { op = "solve"; kind })
  in
  if b_expanded != b then squeeze ~axes:[ ndim result - 1 ] result else result

let pinv' (type a b) ~by ?rtol ?hermitian (a : (a, b) t) =
  check_float_or_complex ~op:"pinv" a;
  (match hermitian with
  | Some true -> check_square ~op:"pinv" a
  | None | Some false -> ());
  let sh = shape a in
  let m = sh.(Array.length sh - 2) in
  let n = sh.(Array.length sh - 1) in
  let dtype_a = dtype a in
  let eps =
    if
      Nx_dtype.equal dtype_a Nx_dtype.float32
      || Nx_dtype.equal dtype_a Nx_dtype.complex64
    then 1.2e-7
    else if
      Nx_dtype.equal dtype_a Nx_dtype.float64
      || Nx_dtype.equal dtype_a Nx_dtype.complex128
    then 2.2e-16
    else 1e-15
  in
  let max_dim = float_of_int (Stdlib.max m n) in
  let cutoff ~max_s =
    match rtol with
    | Some r -> r *. max_s *. max_dim
    | None -> max_dim *. eps *. max_s
  in
  let pinv_from_factors u s vh =
    let max_s = max s |> read_item ~by in
    let cutoff = cutoff ~max_s in
    let ones_s = ones (Value.context s) (dtype s) (shape s) in
    let threshold = scalar (Value.context s) (dtype s) cutoff in
    let mask = greater s threshold in
    let s_inv =
      mul (div ones_s (where mask s ones_s)) (cast (dtype s) mask)
      |> cast dtype_a
    in
    let v =
      if Nx_dtype.is_complex dtype_a then matrix_transpose (conjugate vh)
      else matrix_transpose vh
    in
    (* Scale V's columns. The singleton belongs immediately before the
       singular-value axis so batched factors [..., n, k] and [..., k]
       broadcast as [..., n, k]. *)
    let vs = mul v (expand_dims [ -2 ] s_inv) in
    if Nx_dtype.is_complex dtype_a then
      matmul vs (matrix_transpose (conjugate u))
    else matmul vs (matrix_transpose u)
  in
  let pinv_via_svd () =
    let u, s, vh = B.svd ~full_matrices:false a in
    pinv_from_factors u s vh
  in
  match hermitian with
  | Some true ->
      let vals, vecs = B.eigh a in
      let abs_vals = abs vals in
      let sign_vals = sign vals in
      let o = ones (Value.context vals) (dtype vals) (shape vals) in
      let z = zeros (Value.context vals) (dtype vals) (shape vals) in
      let sign_fixed = where (cmpeq sign_vals z) o sign_vals in
      let vecs_h =
        if Nx_dtype.is_complex dtype_a then matrix_transpose (conjugate vecs)
        else matrix_transpose vecs
      in
      let vh = mul (expand_dims [ -1 ] (cast dtype_a sign_fixed)) vecs_h in
      pinv_from_factors vecs abs_vals vh
  | _ -> pinv_via_svd ()

let pinv ?rtol ?hermitian a = pinv' ~by:"Nx.pinv" ?rtol ?hermitian a

let lstsq ?rcond a b =
  let by = "Nx.lstsq" in
  check_float_or_complex ~op:"lstsq" a;
  check_float_or_complex ~op:"lstsq" b;
  let sh = shape a in
  let m = sh.(Array.length sh - 2) in
  let n = sh.(Array.length sh - 1) in
  let rcond_value =
    match rcond with
    | Some v -> v
    | None ->
        let eps =
          if Nx_dtype.equal (dtype a) Nx_dtype.float32 then 1.2e-7
          else if Nx_dtype.equal (dtype a) Nx_dtype.float64 then 2.2e-16
          else 1e-15
        in
        float_of_int (Stdlib.max m n)
        *. eps
        *. (max (svdvals a) |> read_item ~by)
  in
  let x =
    if m >= n then
      let q, r = B.qr ~reduced:true a in
      let y = matmul (matrix_transpose q) b in
      let r_sq =
        if ndim r = 2 then slice [ R (0, n); R (0, n) ] r
        else slice [ A; R (0, n); R (0, n) ] r
      in
      let y_top =
        if ndim y = 2 then slice [ R (0, n); A ] y
        else if ndim y = 1 then slice [ R (0, n) ] y
        else slice [ A; R (0, n); A ] y
      in
      B.solve_triangular ~upper:true ~transpose:false ~unit_diag:false r_sq
        y_top
    else matmul (pinv' ~by a ~rtol:rcond_value) b
  in
  let residuals =
    if m > n then
      let res = sub b (matmul a x) in
      sum (square res) ~axes:[ ndim res - 2 ] ~keepdims:false
    else zeros (Value.context a) (dtype b) [||]
  in
  (x, residuals, matrix_rank' ~by a, svdvals a)

let inv a =
  check_square ~op:"inv" a;
  check_float_or_complex ~op:"inv" a;
  let sh = shape a in
  let n = sh.(Array.length sh - 1) in
  let batch = Array.sub sh 0 (Array.length sh - 2) in
  let i =
    broadcast_to
      (Array.append batch [| n; n |])
      (eye (Value.context a) (dtype a) n)
  in
  try solve a i with
  | Invalid_argument msg when String.sub msg 0 5 = "solve" ->
      invalid_arg ("inv" ^ String.sub msg 5 (String.length msg - 5))
  | Nx_backend.Linalg_error { kind; _ } ->
      raise (Nx_backend.Linalg_error { op = "inv"; kind })

let matrix_power a n =
  let sh = shape a in
  let rank = Array.length sh in
  if rank < 2 then
    invalid_arg "matrix_power: input requires at least 2D array";
  if sh.(rank - 2) <> sh.(rank - 1) then
    err "matrix_power" "matrix must be square, got %dx%d"
      sh.(rank - 2)
      sh.(rank - 1);
  let rec power acc base exp =
    if exp = 0 then acc
    else if exp mod 2 = 0 then power acc (matmul base base) (exp / 2)
    else power (matmul acc base) (matmul base base) (exp / 2)
  in
  if n = 0 then eye (Value.context a) (dtype a) sh.(rank - 1)
  else if n > 0 then power a a (n - 1)
  else
    try
      let ia = inv a in
      if -n = 1 then ia else power ia ia (-n - 1)
    with Nx_backend.Linalg_error { kind; _ } ->
      raise (Nx_backend.Linalg_error { op = "matrix_power"; kind })

let cond ?p x =
  check_square ~op:"cond" x;
  check_float_or_complex ~op:"cond" x;
  match p with
  | None | Some `Two ->
      let s = svdvals x in
      let ds = dtype s in
      let mx = max s in
      let max_v = mx |> read_item ~by:"Nx.cond" in
      let eps =
        if Nx_dtype.equal ds Nx_dtype.float32 then 1.2e-7
        else if Nx_dtype.equal ds Nx_dtype.float64 then 2.2e-16
        else 1e-15
      in
      let tol_t = scalar (Value.context x) ds (eps *. max_v) in
      let safe_s = where (greater s tol_t) s tol_t in
      let mn =
        if ndim safe_s > 1 then min safe_s ~axes:[ -1 ] ~keepdims:false
        else min safe_s
      in
      cast (dtype x) (div mx mn)
  | Some `One -> mul (norm ~ord:`One x) (norm ~ord:`One (inv x))
  | Some `Inf -> mul (norm ~ord:`Inf x) (norm ~ord:`Inf (inv x))
  | _ -> invalid_arg "cond: unsupported norm"

let tensorsolve ?axes a b =
  check_float_or_complex ~op:"tensorsolve" a;
  check_float_or_complex ~op:"tensorsolve" b;
  let sa = shape a in
  let sb = shape b in
  let ra = Array.length sa in
  let rb = Array.length sb in
  if rb = 0 then invalid_arg "tensorsolve: b must have at least one dimension";
  if ra < rb then invalid_arg "tensorsolve: a, rank must be >= rank of b";
  let axes_for_b =
    match axes with
    | None -> Array.init rb Fun.id
    | Some axes ->
        if List.length axes <> rb then
          err "tensorsolve" "axes, expected %d entries, got %d" rb
            (List.length axes);
        let seen = Array.make ra false in
        Array.map
          (fun ax ->
            let axis = if ax < 0 then ax + ra else ax in
            if axis < 0 || axis >= ra then
              err "tensorsolve" "axis %d out of bounds for %dD tensor" ax ra;
            if seen.(axis) then err "tensorsolve" "axis %d, repeated" ax;
            seen.(axis) <- true;
            axis)
          (Array.of_list axes)
  in
  let selected = Array.make ra false in
  Array.iter (fun ax -> selected.(ax) <- true) axes_for_b;
  let free =
    Array.of_list
      (List.filter (fun ax -> not selected.(ax)) (List.init ra Fun.id))
  in
  let perm = Array.append free axes_for_b in
  let a_perm =
    let rec is_id i =
      if i = ra then true else if perm.(i) <> i then false else is_id (i + 1)
    in
    if is_id 0 then a else transpose ~axes:(Array.to_list perm) a
  in
  let ps = shape a_perm in
  let nf = Array.length free in
  let free_shape = Array.sub ps 0 nf in
  let rhs_shape = Array.sub ps nf rb in
  if rhs_shape <> sb then
    err "tensorsolve" "cannot reshape %s to %s"
      (Shape.to_string rhs_shape)
      (Shape.to_string sb);
  let rows = array_prod free_shape in
  let cols = array_prod rhs_shape in
  if rows <> cols then
    invalid_arg
      "tensorsolve: a, leading dimensions must match trailing dimensions";
  let a_mat = reshape [| rows; cols |] a_perm in
  let b_vec = reshape [| rows |] b in
  let solution =
    try solve a_mat b_vec
    with Nx_backend.Linalg_error { kind = `Singular; _ } ->
      let x_col =
        matmul (pinv' ~by:"Nx.tensorsolve" a_mat) (reshape [| rows; 1 |] b_vec)
      in
      reshape [| cols |] x_col
  in
  reshape free_shape solution

let tensorinv ?ind a =
  check_float_or_complex ~op:"tensorinv" a;
  let sh = shape a in
  let rank = Array.length sh in
  if rank = 0 then
    invalid_arg "tensorinv: input must have at least one dimension";
  let ind = Option.value ind ~default:(rank / 2) in
  if ind <= 0 || ind >= rank then
    invalid_arg
      "tensorinv: ind must split dimensions into two non-empty groups";
  let left = Array.sub sh 0 ind in
  let right = Array.sub sh ind (rank - ind) in
  let ls = array_prod left in
  let rs = array_prod right in
  if ls <> rs then
    invalid_arg
      "tensorinv: input, leading and trailing dimensions must have equal \
       product";
  let inv_mat =
    try inv (reshape [| ls; rs |] a)
    with Nx_backend.Linalg_error { kind = `Singular; _ } ->
      pinv' ~by:"Nx.tensorinv" (reshape [| ls; rs |] a)
  in
  reshape (Array.append right left) inv_mat

(* ───── FFT ───── *)

type fft_norm = [ `Backward | `Forward | `Ortho ]

let pad_or_truncate_for_fft x axes s =
  match s with
  | None -> x
  | Some sizes ->
      let s_arr = Array.of_list sizes in
      let acc = ref x in
      List.iteri
        (fun i ax ->
          let ax = if ax < 0 then ndim !acc + ax else ax in
          let cur = dim ax !acc in
          let target = s_arr.(i) in
          if target > cur then (
            let pad_config = Array.make (ndim !acc) (0, 0) in
            pad_config.(ax) <- (0, target - cur);
            acc := B.pad pad_config (Nx_dtype.zero (dtype !acc)) !acc)
          else if target < cur then
            acc :=
              B.shrink !acc
                (Array.init (ndim !acc) (fun idx ->
                     if idx = ax then (0, target) else (0, dim idx !acc))))
        axes;
      !acc

(* The factors that divide a transform of [n] points by [n] and by its square
   root. A transform of no points has nothing to scale: each of its outputs is
   an empty sum, zero, under every norm. *)
let per_length n = if n = 0 then 1.0 else 1.0 /. float_of_int n
let per_sqrt_length n =
  if n = 0 then 1.0 else 1.0 /. Stdlib.sqrt (float_of_int n)

let fft_norm_scale norm axes_list x =
  match norm with
  | `Backward -> 1.0
  | `Forward ->
      let n = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list in
      per_length n
  | `Ortho ->
      let n = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list in
      per_sqrt_length n

(* Inverse: Backward↔Forward swapped, Ortho unchanged *)
let ifft_norm_scale norm axes_list x =
  match norm with
  | `Backward ->
      let n = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list in
      per_length n
  | `Forward -> 1.0
  | `Ortho ->
      let n = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list in
      per_sqrt_length n

let apply_fft_scale (type a) scale (result : (Complex.t, a) t) :
    (Complex.t, a) t =
  if scale <> 1.0 then
    let sv =
      match Value.dtype result with
      | Complex64 | Complex128 -> Complex.{ re = scale; im = 0.0 }
    in
    mul result (scalar (Value.context result) (Value.dtype result) sv)
  else result

let fftn (type a) ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t) :
    (Complex.t, a) t =
  let nd = ndim x in
  let axes_list =
    match axes with
    | None -> List.init nd Fun.id
    | Some a -> List.map (fun ax -> if ax < 0 then nd + ax else ax) a
  in
  (match s with
  | Some sizes when List.length sizes <> List.length axes_list ->
      invalid_arg "fft: s parameter must have same length as axes"
  | _ -> ());
  let xp = pad_or_truncate_for_fft x axes_list s in
  let scale = fft_norm_scale norm axes_list xp in
  let r = B.fft ~inverse:false ~axes:(Array.of_list axes_list) xp in
  apply_fft_scale scale r

let ifftn (type a) ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t) :
    (Complex.t, a) t =
  let nd = ndim x in
  let axes_list =
    match axes with
    | None -> List.init nd Fun.id
    | Some a -> List.map (fun ax -> if ax < 0 then nd + ax else ax) a
  in
  (match s with
  | Some sizes when List.length sizes <> List.length axes_list ->
      invalid_arg "ifft: s parameter must have same length as axes"
  | _ -> ());
  let xp = pad_or_truncate_for_fft x axes_list s in
  let scale = ifft_norm_scale norm axes_list xp in
  let r = B.fft ~inverse:true ~axes:(Array.of_list axes_list) xp in
  apply_fft_scale scale r

let rfftn dtype ?axes ?s ?(norm = `Backward) x =
  let nd = ndim x in
  let axes_list =
    match axes with
    | None -> [ nd - 1 ]
    | Some values ->
        List.map (fun axis -> if axis < 0 then nd + axis else axis) values
  in
  (match s with
  | Some sizes when List.length sizes <> List.length axes_list ->
      invalid_arg "rfft: s parameter must have same length as axes"
  | _ -> ());
  let xp = pad_or_truncate_for_fft x axes_list s in
  let scale = fft_norm_scale norm axes_list xp in
  let r = B.rfft dtype ~axes:(Array.of_list axes_list) xp in
  apply_fft_scale scale r

(* The backend's real transforms read and write float32 and float64. A
   narrower float widens to float32 exactly on the way in, and the inverse
   works at float32 and rounds once on the way out. *)
let rfftn (type a) dtype ?axes ?s ?norm (x : (float, a) t) =
  match Value.dtype x with
  | Float32 | Float64 -> rfftn dtype ?axes ?s ?norm x
  | _ -> rfftn dtype ?axes ?s ?norm (cast Nx_dtype.float32 x)

let irfftn dtype ?axes ?s ?(norm = `Backward) x =
  let nd = ndim x in
  let axes_list =
    match axes with
    | None -> [ nd - 1 ]
    | Some values ->
        List.map (fun axis -> if axis < 0 then nd + axis else axis) values
  in
  (match s with
  | Some sizes when List.length sizes <> List.length axes_list ->
      invalid_arg "irfft: s parameter must have same length as axes"
  | _ -> ());
  let input_shape = shape x in
  let output_sizes =
    match s with
    | Some sizes -> sizes
    | None ->
        List.mapi
          (fun i axis ->
            if i = List.length axes_list - 1 then (input_shape.(axis) - 1) * 2
            else input_shape.(axis))
          axes_list
  in
  let norm_scale =
    let n = List.fold_left ( * ) 1 output_sizes in
    match norm with
    | `Backward -> per_length n
    | `Forward -> 1.0
    | `Ortho -> per_sqrt_length n
  in
  let s_param =
    match s with None -> None | Some _ -> Some (Array.of_list output_sizes)
  in
  (* [s] names output lengths along every transformed axis. Crop or zero-pad
     the leading, complex axes to those lengths, as [ifftn] would, and the
     last axis to the s/2 + 1 bins its length supports. The resize policy
     lives here alone: the backend always sees a spectrum whose bins match the
     requested output. *)
  let x =
    match s with
    | None -> x
    | Some sizes ->
        let last = List.length axes_list - 1 in
        let targets =
          List.mapi
            (fun i size -> if i = last then (size / 2) + 1 else size)
            sizes
        in
        pad_or_truncate_for_fft x axes_list (Some targets)
  in
  let r = B.irfft ?s:s_param dtype ~axes:(Array.of_list axes_list) x in
  if norm_scale <> 1.0 then
    mul r (scalar (Value.context r) (Value.dtype r) norm_scale)
  else r

let irfftn (type b) (dtype : (float, b) Nx_dtype.t) ?axes ?s ?norm x :
    (float, b) t =
  match dtype with
  | Float32 | Float64 -> irfftn dtype ?axes ?s ?norm x
  | _ -> cast dtype (irfftn Nx_dtype.float32 ?axes ?s ?norm x)

(* 1D FFT convenience *)
let fft ?(axis = -1) ?n ?(norm = `Backward) x =
  let s = match n with None -> None | Some sz -> Some [ sz ] in
  fftn x ~axes:[ axis ] ?s ~norm

let ifft ?(axis = -1) ?n ?(norm = `Backward) x =
  let s = match n with None -> None | Some sz -> Some [ sz ] in
  ifftn x ~axes:[ axis ] ?s ~norm

let rfft dtype ?(axis = -1) ?n ?(norm = `Backward) x =
  let s = match n with None -> None | Some sz -> Some [ sz ] in
  rfftn dtype x ~axes:[ axis ] ?s ~norm

let irfft dtype ?(axis = -1) ?n ?(norm = `Backward) x =
  let s = match n with None -> None | Some sz -> Some [ sz ] in
  irfftn dtype x ~axes:[ axis ] ?s ~norm

(* 2D FFT *)

let check_fft2 ~op x axes =
  let n = ndim x in
  if n < 2 then err op "input requires at least 2D array, got %dD" n;
  let axes_list =
    match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
  in
  if List.length axes_list <> 2 then err op "axes must specify exactly 2 axes";
  axes_list

let fft2 ?axes ?s ?(norm = `Backward) x =
  let axes_list = check_fft2 ~op:"fft2" x axes in
  fftn x ~axes:axes_list ?s ~norm

let ifft2 ?axes ?s ?(norm = `Backward) x =
  let axes_list = check_fft2 ~op:"ifft2" x axes in
  ifftn x ~axes:axes_list ?s ~norm

(* N-dimensional FFT public wrappers *)
let fftn ?axes ?s ?(norm = `Backward) x =
  fftn x
    ~axes:
      (match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
    ?s ~norm

let ifftn ?axes ?s ?(norm = `Backward) x =
  ifftn x
    ~axes:
      (match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
    ?s ~norm

let rfft2 dtype ?axes ?s ?(norm = `Backward) x =
  let axes_list = check_fft2 ~op:"rfft2" x axes in
  rfftn dtype x ~axes:axes_list ?s ~norm

let irfft2 dtype ?axes ?s ?(norm = `Backward) x =
  let axes_list = check_fft2 ~op:"irfft2" x axes in
  irfftn dtype x ~axes:axes_list ?s ~norm

let rfftn dtype ?axes ?s ?(norm = `Backward) x =
  rfftn dtype x
    ~axes:
      (match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
    ?s ~norm

let irfftn dtype ?axes ?s ?(norm = `Backward) x =
  irfftn dtype x
    ~axes:
      (match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
    ?s ~norm

(* Hermitian FFT. The forward transform of the Hermitian signal whose half is
   [x] is the unscaled inverse transform of its conjugate, so each direction
   is the other real transform with the norm's scaling swapped. *)
let swap_norm = function
  | `Backward -> `Forward
  | `Forward -> `Backward
  | `Ortho -> `Ortho

let hfft dtype ?(axis = -1) ?n ?(norm = `Backward) x =
  let n = match n with None -> 2 * (dim axis x - 1) | Some n -> n in
  let axis = resolve_single_axis x axis in
  irfftn dtype (conjugate x) ~axes:[ axis ] ~s:[ n ] ~norm:(swap_norm norm)

let ihfft dtype ?(axis = -1) ?n ?(norm = `Backward) x =
  let n = match n with None -> dim axis x | Some n -> n in
  let axis = resolve_single_axis x axis in
  conjugate (rfftn dtype x ~axes:[ axis ] ~s:[ n ] ~norm:(swap_norm norm))

(* FFT helpers *)
let fftfreq ctx dt ?(d = 1.0) n =
  let v = 1.0 /. (float_of_int n *. d) in
  let freqs =
    if n mod 2 = 0 then
      concatenate ~axis:0
        [
          cast dt (arange ctx Nx_dtype.int64 0 (n / 2) 1);
          cast dt (arange ctx Nx_dtype.int64 (-(n / 2)) 0 1);
        ]
    else
      concatenate ~axis:0
        [
          cast dt (arange ctx Nx_dtype.int64 0 ((n + 1) / 2) 1);
          cast dt (arange ctx Nx_dtype.int64 (-((n - 1) / 2)) 0 1);
        ]
  in
  mul_s freqs v

let rfftfreq ctx dt ?(d = 1.0) n =
  let v = 1.0 /. (float_of_int n *. d) in
  mul (cast dt (arange ctx Nx_dtype.int64 0 ((n / 2) + 1) 1)) (scalar ctx dt v)

let fftshift ?axes x =
  let sh = shape x in
  let axes_list =
    match axes with
    | None -> List.init (Array.length sh) Fun.id
    | Some ax -> ax
  in
  List.fold_left
    (fun acc axis ->
      let axis = resolve_single_axis acc axis in
      roll (sh.(axis) / 2) acc ~axis)
    x axes_list

let ifftshift ?axes x =
  let sh = shape x in
  let axes_list =
    match axes with
    | None -> List.init (Array.length sh) Fun.id
    | Some ax -> ax
  in
  List.fold_left
    (fun acc axis ->
      let axis = resolve_single_axis acc axis in
      roll (-(sh.(axis) / 2)) acc ~axis)
    x axes_list

(* Short-time Fourier analysis.

   Framing is a pure view: the frames alias the signal through overlapping
   strides and are never materialized, so a spectrogram costs one transform
   pass and no framed copy. The inverse is weighted overlap-add — each frame
   is windowed again and summed back at the position it came from, then
   divided by the overlap envelope the windows themselves produce. Dividing by
   the measured envelope rather than assuming one makes reconstruction exact
   for every [step] that covers the signal, instead of only for the hops where
   the window happens to sum to a constant — which is also what lets both
   directions default their taper without pinning the hop. *)

let hann ctx dt n =
  if n < 1 then err "hann" "length must be >= 1, got %d" n;
  (* Periodic form: the sample that would open the next period is dropped
     rather than duplicated, which is what lets shifted copies sum flat. The
     symmetric variant divides by [n - 1] and does not. *)
  let phase =
    mul_s
      (arange_f ctx dt 0. (float_of_int n) 1.)
      (2.0 *. Float.pi /. float_of_int n)
  in
  add_s (mul_s (cos phase) (-0.5)) 0.5

let stft_step ~window = function
  | Some s -> s
  | None -> Stdlib.max 1 (window / 4)

(* Framing with no taper multiplies the signal by a rectangle, and a
   rectangle's own spectrum smears every component across every bin, so the
   default tapers rather than leaving the leakage to whoever did not know to
   ask. An explicit [ones] opts back out. *)
let taper op ctx dt ~window = function
  | None -> hann ctx dt window
  | Some w ->
      if shape w <> [| window |] then
        err op "win has shape %s, expected [%d]"
          (Shape.to_string (shape w))
          window;
      w

let stft (cdt : (Complex.t, 'c) Nx_dtype.t) ~window ?step ?win x :
    (Complex.t, 'c) t =
  let step = stft_step ~window step in
  let r = ndim x in
  if r < 1 then err "stft" "input must have at least 1 dimension";
  if window < 1 then err "stft" "window must be >= 1, got %d" window;
  if step < 1 then err "stft" "step must be >= 1, got %d" step;
  let n = (shape x).(r - 1) in
  if window > n then
    err "stft" "window %d exceeds the %d samples on the last axis" window n;
  let w = taper "stft" (Value.context x) (Value.dtype x) ~window win in
  let frames = B.sliding_window x ~axis:(r - 1) ~window ~step in
  rfft cdt (mul frames w) ~axis:(-1)

let istft (dt : (float, 'a) Nx_dtype.t) ~window ?step ?win ?length z :
    (float, 'a) t =
  let step = stft_step ~window step in
  let r = ndim z in
  if r < 2 then err "istft" "input must have at least 2 dimensions";
  if window < 1 then err "istft" "window must be >= 1, got %d" window;
  if step < 1 || step > window then
    err "istft" "step must be in [1, %d] to cover the signal, got %d" window
      step;
  let z_shape = shape z in
  let bins = (window / 2) + 1 in
  if z_shape.(r - 1) <> bins then
    err "istft" "last axis has %d bins, expected %d for window %d"
      z_shape.(r - 1)
      bins window;
  let frames = z_shape.(r - 2) in
  let out_len = ((frames - 1) * step) + window in
  let ctx = Value.context z in
  let w = taper "istft" ctx dt ~window win in
  let taper_sq = mul w w in
  (* [fold] sums the taps landing on each output position, which is exactly
     overlap-add, and reads its operand as [(leading…, window, frames)]. *)
  let swap_last_two rank =
    List.init rank (fun i ->
        if i < rank - 2 then i
        else if i = rank - 2 then rank - 1
        else rank - 2)
  in
  let overlap_add t =
    let rank = ndim t in
    B.fold
      (transpose t ~axes:(swap_last_two rank))
      ~output_size:[| out_len |] ~kernel_size:[| window |] ~stride:[| step |]
      ~dilation:[| 1 |]
      ~padding:[| (0, 0) |]
  in
  let windowed = mul (irfft dt z ~axis:(-1) ~n:window) w in
  let signal = overlap_add windowed in
  let envelope =
    overlap_add
      (broadcast_to [| frames; window |] (reshape [| 1; window |] taper_sq))
  in
  (* A sample the analysis window zeroed carries no information: both its
     envelope and its numerator are zero, so dividing by one returns the
     honest 0 rather than a NaN. *)
  let denom =
    where
      (equal envelope (scalar_like envelope (Nx_dtype.zero dt)))
      (scalar_like envelope (Nx_dtype.one dt))
      envelope
  in
  let y = div signal denom in
  match length with
  | None -> y
  | Some l ->
      if l < 1 then err "istft" "length must be >= 1, got %d" l;
      let rank = ndim y in
      let axis_of i = (0, (shape y).(i)) in
      if l < out_len then
        B.shrink y
          (Array.init rank (fun i ->
               if i = rank - 1 then (0, l) else axis_of i))
      else if l > out_len then
        pad
          (Array.init rank (fun i ->
               if i = rank - 1 then (0, l - out_len) else (0, 0)))
          (Nx_dtype.zero dt) y
      else y

(* Discrete cosine and sine transforms. These are expressed in terms of the
   complex FFT so every backend, including effectful ones, gets the same
   implementation without extending the backend interface. *)

let real_transform_slice_last spec x =
  slice (List.init (ndim x - 1) (fun _ -> A) @ [ spec ]) x

let real_transform_scale_first factor x =
  concatenate ~axis:(-1)
    [
      mul_s (real_transform_slice_last (R (0, 1)) x) factor;
      real_transform_slice_last (R (1, dim (-1) x)) x;
    ]

let real_transform_scale_last factor x =
  let n = dim (-1) x in
  concatenate ~axis:(-1)
    [
      real_transform_slice_last (R (0, n - 1)) x;
      mul_s (real_transform_slice_last (R (n - 1, n)) x) factor;
    ]

let real_transform_alternating_signs (type a) (dtype : (float, a) Nx_dtype.t)
    ctx n =
  let indices = arange ctx Nx_dtype.int64 0 n 1 in
  let even = equal_s (mod_s indices 2L) 0L in
  where even (ones ctx dtype [| n |]) (full ctx dtype [| n |] (-1.0))

let real_transform_phase (type a b) (float_dtype : (float, a) Nx_dtype.t)
    (complex_dtype : (Complex.t, b) Nx_dtype.t) ctx n =
  let indices = cast float_dtype (arange ctx Nx_dtype.int64 0 n 1) in
  let angles = mul_s indices (-.Float.pi /. (2.0 *. float_of_int n)) in
  add
    (cast complex_dtype (cos angles))
    (mul_s (cast complex_dtype (sin angles)) Complex.{ re = 0.0; im = 1.0 })

let dct_raw_last (type a b) ~type_ (float_dtype : (float, a) Nx_dtype.t)
    (complex_dtype : (Complex.t, b) Nx_dtype.t) (x : (float, a) t) =
  let ctx = Value.context x in
  let n = dim (-1) x in
  let dct_2 x =
    let n = dim (-1) x in
    let even = real_transform_slice_last (Rs (0, n, 2)) x in
    let odd =
      flip ~axes:[ -1 ] (real_transform_slice_last (Rs (1, n, 2)) x)
    in
    let reordered = concatenate ~axis:(-1) [ even; odd ] in
    let spectrum = fft (cast complex_dtype reordered) in
    mul_s
      (cast float_dtype
         (mul spectrum (real_transform_phase float_dtype complex_dtype ctx n)))
      2.0
  in
  let dct_3 x =
    let zero_shape = Array.copy (shape x) in
    zero_shape.(Array.length zero_shape - 1) <- (2 * n) + 1;
    let tail = flip ~axes:[ -1 ] (real_transform_slice_last (R (1, n)) x) in
    let extended =
      concatenate ~axis:(-1) [ x; zeros ctx float_dtype zero_shape; tail ]
    in
    let spectrum = fft (cast complex_dtype extended) in
    let odd_indices = arange ctx Nx_dtype.int64 1 (2 * n) 2 in
    cast float_dtype (take spectrum ~axis:(-1) ~indices:odd_indices)
  in
  match type_ with
  | 1 ->
      let interior =
        flip ~axes:[ -1 ] (real_transform_slice_last (R (1, n - 1)) x)
      in
      let spectrum =
        fft (cast complex_dtype (concatenate ~axis:(-1) [ x; interior ]))
      in
      cast float_dtype (real_transform_slice_last (R (0, n)) spectrum)
  | 2 -> dct_2 x
  | 3 -> dct_3 x
  | 4 ->
      let zero_shape = Array.copy (shape x) in
      zero_shape.(Array.length zero_shape - 1) <- n;
      let padded =
        concatenate ~axis:(-1) [ x; zeros ctx float_dtype zero_shape ]
      in
      let transformed = dct_2 padded in
      take transformed ~axis:(-1)
        ~indices:(arange ctx Nx_dtype.int64 1 (2 * n) 2)
  | _ -> assert false

let dst_raw_last (type a b) ~type_ (float_dtype : (float, a) Nx_dtype.t)
    (complex_dtype : (Complex.t, b) Nx_dtype.t) (x : (float, a) t) =
  let ctx = Value.context x in
  let n = dim (-1) x in
  let signs = real_transform_alternating_signs float_dtype ctx n in
  match type_ with
  | 1 ->
      let zero_shape = Array.copy (shape x) in
      zero_shape.(Array.length zero_shape - 1) <- 1;
      let zero = zeros ctx float_dtype zero_shape in
      let extended =
        concatenate ~axis:(-1) [ zero; x; zero; neg (flip ~axes:[ -1 ] x) ]
      in
      let spectrum = fft (cast complex_dtype extended) in
      let frequencies = real_transform_slice_last (R (1, n + 1)) spectrum in
      cast float_dtype (mul_s frequencies Complex.{ re = 0.0; im = 1.0 })
  | 2 ->
      flip ~axes:[ -1 ]
        (dct_raw_last ~type_:2 float_dtype complex_dtype (mul x signs))
  | 3 ->
      mul signs
        (dct_raw_last ~type_:3 float_dtype complex_dtype (flip ~axes:[ -1 ] x))
  | 4 ->
      flip ~axes:[ -1 ]
        (dct_raw_last ~type_:4 float_dtype complex_dtype (mul x signs))
  | _ -> assert false

let validate_real_transform ~op ~type_ (type a) (x : (float, a) t) =
  (match dtype x with
  | Float32 | Float64 -> ()
  | dtype ->
      err op "dtype, expected float32 or float64, got %s"
        (Nx_dtype.to_string dtype));
  if type_ < 1 || type_ > 4 then
    err op "type_, expected one of 1, 2, 3, or 4, got %d" type_

let resolve_real_transform_axis ~op x axis =
  let rank = ndim x in
  let resolved = if axis < 0 then axis + rank else axis in
  if resolved < 0 || resolved >= rank then
    err op "axis %d out of bounds for %dD tensor" axis rank;
  resolved

let normalize_real_transform_axes ~op x axes =
  let rank = ndim x in
  let seen = Array.make rank false in
  List.map
    (fun axis ->
      let resolved = if axis < 0 then axis + rank else axis in
      if resolved < 0 || resolved >= rank then
        err op "axis %d out of bounds for %dD tensor" axis rank;
      if seen.(resolved) then err op "axis %d is repeated" axis;
      seen.(resolved) <- true;
      resolved)
    axes

let real_transform_length family type_ n =
  match (family, type_) with
  | `Dct, 1 -> 2 * (n - 1)
  | `Dst, 1 -> 2 * (n + 1)
  | (`Dct | `Dst), (2 | 3 | 4) -> 2 * n
  | _ -> assert false

let real_transform_ortho_input family type_ x =
  let sqrt_two = Stdlib.sqrt 2.0 in
  match (family, type_) with
  | `Dct, 1 ->
      x
      |> real_transform_scale_first sqrt_two
      |> real_transform_scale_last sqrt_two
  | `Dct, 3 -> real_transform_scale_first sqrt_two x
  | `Dst, 3 -> real_transform_scale_last sqrt_two x
  | (`Dct | `Dst), (1 | 2 | 4) -> x
  | _ -> assert false

let real_transform_ortho_output family type_ x =
  let inv_sqrt_two = 1.0 /. Stdlib.sqrt 2.0 in
  match (family, type_) with
  | `Dct, 1 ->
      x
      |> real_transform_scale_first inv_sqrt_two
      |> real_transform_scale_last inv_sqrt_two
  | `Dct, 2 -> real_transform_scale_first inv_sqrt_two x
  | `Dst, 2 -> real_transform_scale_last inv_sqrt_two x
  | (`Dct | `Dst), (1 | 3 | 4) -> x
  | _ -> assert false

let real_transform (type a) ~op ~family ~inverse ~type_ ~axis ~norm
    (x : (float, a) t) : (float, a) t =
  validate_real_transform ~op ~type_ x;
  if ndim x = 0 then err op "input must have at least one dimension";
  let axis = resolve_real_transform_axis ~op x axis in
  let x = moveaxis axis (-1) x in
  let n = dim (-1) x in
  if n = 0 then err op "input size along axis %d must be positive" axis;
  if family = `Dct && type_ = 1 && n = 1 then
    err op "type 1 requires an input size greater than 1";
  let raw_type =
    if inverse then match type_ with 2 -> 3 | 3 -> 2 | value -> value
    else type_
  in
  let x =
    match norm with
    | `Ortho -> real_transform_ortho_input family raw_type x
    | `Backward | `Forward -> x
  in
  let transformed : (float, a) t =
    match dtype x with
    | Float32 -> (
        match family with
        | `Dct -> dct_raw_last ~type_:raw_type Float32 Complex64 x
        | `Dst -> dst_raw_last ~type_:raw_type Float32 Complex64 x)
    | Float64 -> (
        match family with
        | `Dct -> dct_raw_last ~type_:raw_type Float64 Complex128 x
        | `Dst -> dst_raw_last ~type_:raw_type Float64 Complex128 x)
    | _ -> assert false
  in
  let transformed =
    match norm with
    | `Ortho -> real_transform_ortho_output family raw_type transformed
    | `Backward | `Forward -> transformed
  in
  let length = real_transform_length family type_ n in
  let scale =
    match (inverse, norm) with
    | false, `Backward | true, `Forward -> 1.0
    | false, `Forward | true, `Backward -> per_length length
    | _, `Ortho -> per_sqrt_length length
  in
  let transformed =
    if scale = 1.0 then transformed else mul_s transformed scale
  in
  moveaxis (-1) axis transformed

let real_transformn ~op ~family ~inverse ~type_ ~axes ~norm x =
  validate_real_transform ~op ~type_ x;
  let axes =
    normalize_real_transform_axes ~op x
      (match axes with
      | None -> List.init (ndim x) Fun.id
      | Some axes -> axes)
  in
  List.fold_left
    (fun acc axis ->
      real_transform ~op ~family ~inverse ~type_ ~axis ~norm acc)
    x axes

let dct ?(type_ = 2) ?(axis = -1) ?(norm = `Backward) x =
  real_transform ~op:"dct" ~family:`Dct ~inverse:false ~type_ ~axis ~norm x

let idct ?(type_ = 2) ?(axis = -1) ?(norm = `Backward) x =
  real_transform ~op:"idct" ~family:`Dct ~inverse:true ~type_ ~axis ~norm x

let dctn ?(type_ = 2) ?axes ?(norm = `Backward) x =
  real_transformn ~op:"dctn" ~family:`Dct ~inverse:false ~type_ ~axes ~norm x

let idctn ?(type_ = 2) ?axes ?(norm = `Backward) x =
  real_transformn ~op:"idctn" ~family:`Dct ~inverse:true ~type_ ~axes ~norm x

let dst ?(type_ = 2) ?(axis = -1) ?(norm = `Backward) x =
  real_transform ~op:"dst" ~family:`Dst ~inverse:false ~type_ ~axis ~norm x

let idst ?(type_ = 2) ?(axis = -1) ?(norm = `Backward) x =
  real_transform ~op:"idst" ~family:`Dst ~inverse:true ~type_ ~axis ~norm x

let dstn ?(type_ = 2) ?axes ?(norm = `Backward) x =
  real_transformn ~op:"dstn" ~family:`Dst ~inverse:false ~type_ ~axes ~norm x

let idstn ?(type_ = 2) ?axes ?(norm = `Backward) x =
  real_transformn ~op:"idstn" ~family:`Dst ~inverse:true ~type_ ~axes ~norm x

(* ───── Normalisations ───── *)

(* [scale * (x - m)] along [axes], with [m] the maximum of [x] for a
   non-negative scale and its minimum for a negative one: no element is
   positive, so its [exp] cannot overflow. *)
let scaled_shift ~axes ~scale x =
  let extreme = if scale >= 0.0 then max else min in
  let shifted = sub x (extreme x ~axes ~keepdims:true) in
  if scale = 1.0 then shifted
  else mul (scalar_like x (Nx_dtype.of_float (dtype x) scale)) shifted

let softmax ?(axes = [ -1 ]) ?(scale = 1.0) x =
  let axes = normalize_and_dedup_axes ~op:"softmax" (ndim x) axes in
  let e = exp (scaled_shift ~axes ~scale x) in
  div e (sum e ~axes ~keepdims:true)

let log_softmax ?(axes = [ -1 ]) ?(scale = 1.0) x =
  let axes = normalize_and_dedup_axes ~op:"log_softmax" (ndim x) axes in
  if axes = [] then zeros_like x
  else
    let scaled = scaled_shift ~axes ~scale x in
    sub scaled (log (sum (exp scaled) ~axes ~keepdims:true))

let logsumexp ?axes ?(keepdims = false) x =
  let axes_norm =
    match axes with
    | None -> List.init (ndim x) Fun.id
    | Some lst -> normalize_and_dedup_axes ~op:"logsumexp" (ndim x) lst
  in
  if axes_norm = [] then x
  else
    let max_x = max x ~axes:axes_norm ~keepdims:true in
    let log_sum =
      add (log (sum (exp (sub x max_x)) ~axes:axes_norm ~keepdims:true)) max_x
    in
    if keepdims then log_sum else squeeze ~axes:(List.rev axes_norm) log_sum

let logmeanexp ?axes ?(keepdims = false) x =
  let axes_norm =
    match axes with
    | None -> List.init (ndim x) Fun.id
    | Some lst -> normalize_and_dedup_axes ~op:"logmeanexp" (ndim x) lst
  in
  if axes_norm = [] then x
  else
    let log_sum = logsumexp ~axes:axes_norm ~keepdims:true x in
    let count = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_norm in
    let log_mean =
      sub log_sum
        (log
           (scalar_like log_sum
              (Nx_dtype.of_float (dtype x) (float_of_int count))))
    in
    if keepdims then log_mean else squeeze ~axes:(List.rev axes_norm) log_mean

let standardize ?axes ?mean:mean_param ?variance:variance_param
    ?(epsilon = 1e-5) x =
  let nd = ndim x in
  let axes_norm =
    match axes with
    | None -> List.init nd Fun.id
    | Some lst -> normalize_and_dedup_axes ~op:"standardize" nd lst
  in
  let x_shape = shape x in
  let keep_shape =
    Array.mapi
      (fun idx d -> if List.exists (( = ) idx) axes_norm then 1 else d)
      x_shape
  in
  let unaffected =
    List.filter
      (fun idx -> not (List.exists (( = ) idx) axes_norm))
      (List.init nd Fun.id)
  in
  let core_shape =
    Array.of_list (List.map (fun idx -> x_shape.(idx)) unaffected)
  in
  let broadcast_param name param =
    let ps = shape param in
    if ps = x_shape || ps = keep_shape then param
    else if ps = core_shape then reshape keep_shape param
    else err "standardize" "%s, shape must match normalized axes" name
  in
  let mean_tensor =
    match mean_param with
    | Some m -> broadcast_param "mean" m
    | None ->
        if axes_norm = [] then x else mean x ~axes:axes_norm ~keepdims:true
  in
  let variance_tensor =
    match variance_param with
    | Some v -> broadcast_param "variance" v
    | None ->
        refuse_int "standardize" x;
        if axes_norm = [] then zeros_like x
        else var x ~axes:axes_norm ~keepdims:true
  in
  div (sub x mean_tensor)
    (sqrt
       (add variance_tensor
          (scalar_like x (Nx_dtype.of_float (dtype x) epsilon))))

let erf (x : (float, 'b) t) = B.unary Erf x

let sliding_window ?axis ~window ?(step = 1) x =
  let r = ndim x in
  let ax = match axis with Some a -> a | None -> -1 in
  let axis = resolve_single_axis x ax in
  if axis < 0 || axis >= r then
    err "sliding_window" "axis %d out of bounds for %dD tensor" ax r;
  if window < 1 then err "sliding_window" "window must be >= 1, got %d" window;
  if step < 1 then err "sliding_window" "step must be >= 1, got %d" step;
  let size = (shape x).(axis) in
  if window > size then
    err "sliding_window"
      "cannot slide window %d along axis %d in shape %s (%d>%d)" window axis
      (Shape.to_string (shape x))
      window size;
  B.sliding_window x ~axis ~window ~step

(* A window's size, step and dilation along each of its axes are positive, and
   its padding is not negative. *)
let check_window op ~kernel_size ~stride ~dilation ~padding =
  let k = Array.length kernel_size in
  if k = 0 then err op "kernel_size has no axis";
  if
    Array.length stride <> k
    || Array.length dilation <> k
    || Array.length padding <> k
  then
    err op "stride, dilation and padding need one entry per kernel axis (%d)" k;
  let positive name a =
    Array.iter (fun n -> if n < 1 then err op "%s %d is not positive" name n) a
  in
  positive "kernel_size" kernel_size;
  positive "stride" stride;
  positive "dilation" dilation;
  Array.iter
    (fun (before, after) ->
      if before < 0 || after < 0 then
        err op "padding (%d, %d) is negative" before after)
    padding

let extract_patches ~kernel_size ~stride ~dilation ~padding x =
  check_window "extract_patches" ~kernel_size ~stride ~dilation ~padding;
  let k = Array.length kernel_size in
  if ndim x < k then
    err "extract_patches" "%d kernel axes for a rank %d tensor" k (ndim x);
  B.unfold x ~kernel_size ~stride ~dilation ~padding

let combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding x =
  let op = "combine_patches" in
  check_window op ~kernel_size ~stride ~dilation ~padding;
  let k = Array.length kernel_size in
  if Array.length output_size <> k then
    err op "output_size has %d axes, kernel_size %d" (Array.length output_size) k;
  Array.iter
    (fun n -> if n < 0 then err op "output_size %d is negative" n)
    output_size;
  if ndim x < 2 then err op "a rank %d tensor holds no patches" (ndim x);
  let patch = Array.fold_left ( * ) 1 kernel_size
  and windows =
    Array.fold_left ( * ) 1
      (Op.window_counts kernel_size stride dilation padding output_size)
  in
  if dim (-2) x <> patch || dim (-1) x <> windows then
    err op "patches of shape (%d, %d), where the geometry gives (%d, %d)"
      (dim (-2) x) (dim (-1) x) patch windows;
  B.fold x ~output_size ~kernel_size ~stride ~dilation ~padding

(* Correlation and convolution *)

(* The zeros around each spatial axis of [x], of size [n], against a kernel axis
   of size [k]: those whose windows are the part of the full correlation that
   [mode] keeps. A kernel longer than [x] is centred as if the two were
   swapped, and a convolution, whose kernel is flipped, keeps the mirrored
   part. *)
let correlate_padding ~flipped ~mode input k_shape =
  Array.map2
    (fun n k ->
      match mode with
      | `Full -> (k - 1, k - 1)
      | `Valid -> if k <= n then (0, 0) else (k - n, k - n)
      | `Same when k <= n -> (k / 2, k - 1 - (k / 2))
      | `Same ->
          let before = k - 1 - (n / 2) and after = k - n + (n / 2) in
          if flipped then (after, before) else (before, after))
    input k_shape

let correlation ~flipped padding x kernel =
  let kr = ndim kernel in
  let xr = ndim x in
  if xr < kr then err "correlate" "input rank %d < kernel rank %d" xr kr;
  let ks = shape kernel in
  let input_spatial = Array.sub (shape x) (xr - kr) kr in
  let pad_pairs = correlate_padding ~flipped ~mode:padding input_spatial ks in
  let ones_arr = Array.make kr 1 in
  let x_unf =
    B.unfold x ~kernel_size:ks ~stride:ones_arr ~dilation:ones_arr
      ~padding:pad_pairs
  in
  let und = ndim x_unf in
  let kp = (shape x_unf).(und - 2) in
  let result =
    sum (mul x_unf (reshape [| kp; 1 |] kernel)) ~axes:[ und - 2 ]
  in
  let leading = Array.sub (shape x) 0 (xr - kr) in
  let out_spatial =
    Array.init kr (fun i ->
        input_spatial.(i) + fst pad_pairs.(i) + snd pad_pairs.(i) - ks.(i) + 1)
  in
  reshape (Array.concat [ leading; out_spatial ]) result

let correlate ?(padding = `Valid) x kernel =
  correlation ~flipped:false padding x kernel

let convolve ?(padding = `Valid) x kernel =
  correlation ~flipped:true padding x
    (flip ~axes:(List.init (ndim kernel) Fun.id) kernel)

(* Sliding window filters *)

let sliding_filter ~reduce_fn ~kernel_size ?stride x =
  let kr = Array.length kernel_size in
  let stride = match stride with Some s -> s | None -> kernel_size in
  let ones_arr = Array.make kr 1 in
  let zeros_arr = Array.make kr (0, 0) in
  let x_unf =
    B.unfold x ~kernel_size ~stride ~dilation:ones_arr ~padding:zeros_arr
  in
  let und = ndim x_unf in
  let reduced = reduce_fn x_unf ~axes:[ und - 2 ] ~keepdims:false in
  let xr = ndim x in
  let leading = Array.sub (shape x) 0 (xr - kr) in
  let input_spatial = Array.sub (shape x) (xr - kr) kr in
  let out_spatial =
    Array.init kr (fun i ->
        ((input_spatial.(i) - kernel_size.(i)) / stride.(i)) + 1)
  in
  reshape (Array.concat [ leading; out_spatial ]) reduced

let maximum_filter ~kernel_size ?stride x =
  sliding_filter
    ~reduce_fn:(fun x ~axes ~keepdims -> max x ~axes ~keepdims)
    ~kernel_size ?stride x

let minimum_filter ~kernel_size ?stride x =
  sliding_filter
    ~reduce_fn:(fun x ~axes ~keepdims -> min x ~axes ~keepdims)
    ~kernel_size ?stride x

let uniform_filter ~kernel_size ?stride x =
  sliding_filter
    ~reduce_fn:(fun x ~axes ~keepdims:_ -> mean x ~axes)
    ~kernel_size ?stride x

let one_hot ~num_classes index_tensor =
  let dt = dtype index_tensor in
  if not (Nx_dtype.is_int dt || Nx_dtype.is_uint dt) then
    err "one_hot" "dtype %s, indices must be integer type"
      (Nx_dtype.to_string dt);
  if num_classes <= 0 then
    err "one_hot" "num_classes %d, must be positive" num_classes;
  (* Compared as int64: the index dtype may not hold every class. *)
  let idx_exp =
    unsqueeze (cast Nx_dtype.int64 index_tensor) ~axes:[ ndim index_tensor ]
  in
  let nd_exp = ndim idx_exp in
  let s = Array.make nd_exp 1 in
  s.(nd_exp - 1) <- num_classes;
  let arange_b =
    reshape s
      (arange (Value.context index_tensor) Nx_dtype.int64 0 num_classes 1)
  in
  cast Nx_dtype.uint8 (cmpeq idx_exp arange_b)

(* ───── Display and Formatting ───── *)

let pp_shape = Shape.pp
let pp_dtype ppf dtype = Format.pp_print_string ppf (Nx_dtype.to_string dtype)

let pp' (type a b) ~by fmt (x : (a, b) t) =
  let open Format in
  reading ~by x @@ fun element ->
  let dtype = dtype x in
  let shape = shape x in
  let ndim = Array.length shape in
  let sz = numel x in
  let real s fmt x = pp_print_string fmt (float_text s x) in
  let complex s fmt (z : Complex.t) =
    let im = float_text s z.im in
    let sign = if im.[0] = '-' then "" else "+" in
    fprintf fmt "(%a%s%si)" (real s) z.re sign im
  in
  let pp_element fmt (elt : a) =
    match dtype with
    | Float16 -> real Float16 fmt elt
    | Float32 -> real Float32 fmt elt
    | Float64 -> real Float64 fmt elt
    | BFloat16 -> real BFloat16 fmt elt
    | Float8_e4m3 -> real Float8_e4m3 fmt elt
    | Float8_e5m2 -> real Float8_e5m2 fmt elt
    | Int8 -> fprintf fmt "%d" elt
    | Int16 -> fprintf fmt "%d" elt
    | Int32 -> fprintf fmt "%ld" elt
    | Int64 -> fprintf fmt "%Ld" elt
    | UInt8 -> fprintf fmt "%d" elt
    | UInt16 -> fprintf fmt "%d" elt
    | UInt32 -> fprintf fmt "%ld" elt
    | UInt64 -> fprintf fmt "%Ld" elt
    | Int4 -> fprintf fmt "%d" elt
    | UInt4 -> fprintf fmt "%d" elt
    | Bool -> fprintf fmt "%b" elt
    | Bit -> fprintf fmt "%b" elt
    | Complex64 -> complex Float32 fmt elt
    | Complex128 -> complex Float64 fmt elt
  in
  let edge = 2 in
  if ndim = 0 then pp_element fmt (element 0)
  else
    let strides = Shape.c_contiguous_strides shape in
    let sep fmt axis first =
      if not first then (
        fprintf fmt ",";
        if axis = ndim - 1 then fprintf fmt " " else pp_print_cut fmt ())
    in
    let rec pp_slice fmt indices =
      let depth = List.length indices in
      if depth = ndim then
        let md_index = Array.of_list indices in
        pp_element fmt (element (Shape.ravel_index md_index strides))
      else
        let axis = depth in
        let dim_size = shape.(axis) in
        let truncate = dim_size > edge * 2 in
        fprintf fmt "[";
        if dim_size > 0 then (
          if axis < ndim - 1 then pp_open_vbox fmt 0 else pp_open_hbox fmt ();
          if truncate then (
            for i = 0 to edge - 1 do
              sep fmt axis (i = 0);
              pp_slice fmt (indices @ [ i ])
            done;
            fprintf fmt ",";
            if axis = ndim - 1 then fprintf fmt " ..., "
            else (
              pp_print_cut fmt ();
              fprintf fmt "...";
              pp_print_cut fmt ());
            for i = dim_size - edge to dim_size - 1 do
              sep fmt axis (i = dim_size - edge);
              pp_slice fmt (indices @ [ i ])
            done)
          else
            for i = 0 to dim_size - 1 do
              sep fmt axis (i = 0);
              pp_slice fmt (indices @ [ i ])
            done;
          pp_close_box fmt ());
        fprintf fmt "]"
    in
    (* Print shape and dtype header for non-trivial tensors *)
    if ndim > 1 || sz > edge * 2 then (
      fprintf fmt "%a %a " pp_dtype dtype pp_shape shape;
      pp_print_cut fmt ());
    if sz = 0 then fprintf fmt "[]" else pp_slice fmt []

let pp fmt x = pp' ~by:"Nx.pp" fmt x
let to_string x = Format.asprintf "%a" (pp' ~by:"Nx.to_string") x
let print x = Format.printf "%a@." (pp' ~by:"Nx.print") x

(* ───── Higher-order Functions ───── *)

let map_item f x =
  let sz = size x in
  reading ~by:"Nx.map_item" x @@ fun src ->
  let dst = Elements.create (dtype x) sz in
  let set = Elements.set (dtype x) dst in
  for i = 0 to sz - 1 do
    set i (f (src i))
  done;
  reshape (shape x) (B.from_host (Value.context x) (dtype x) dst)

let iter_item f x =
  reading ~by:"Nx.iter_item" x @@ fun src ->
  for i = 0 to size x - 1 do
    f (src i)
  done

let fold_item f init x =
  reading ~by:"Nx.fold_item" x @@ fun src ->
  let acc = ref init in
  for i = 0 to size x - 1 do
    acc := f !acc (src i)
  done;
  !acc

(* ───── Infix Operators ───── *)

module Infix = struct
  let ( + ) a b = add a b
  let ( +$ ) a s = add_s a s
  let ( - ) a b = sub a b
  let ( -$ ) a s = sub_s a s
  let ( ~- ) x = neg x
  let ( * ) a b = mul a b
  let ( *$ ) a s = mul_s a s
  let ( / ) a b = div a b
  let ( /$ ) a s = div_s a s
  let ( ** ) a b = pow a b
  let ( **$ ) a s = pow_s a s
  let ( % ) a b = mod_ a b
  let ( mod ) a b = mod_ a b
  let ( %$ ) a s = mod_s a s
  let ( lxor ) a b = bitwise_xor a b
  let ( lor ) a b = bitwise_or a b
  let ( land ) a b = bitwise_and a b
  let ( && ) a b = logical_and a b
  let ( || ) a b = logical_or a b
  let ( < ) a b = less a b
  let ( <$ ) a b = less_s a b
  let ( <> ) a b = not_equal a b
  let ( <>$ ) a b = not_equal_s a b
  let ( = ) a b = equal a b
  let ( =$ ) a b = equal_s a b
  let ( > ) a b = greater a b
  let ( >$ ) a b = greater_s a b
  let ( <= ) a b = less_equal a b
  let ( <=$ ) a b = less_equal_s a b
  let ( >= ) a b = greater_equal a b
  let ( >=$ ) a b = greater_equal_s a b
  let ( *@ ) a b = matmul a b
  let ( /@ ) = solve
  let ( **@ ) = matrix_power
  let ( .%{} ) x indices = get indices x
  let ( .${} ) x slice_def = slice slice_def x
end
