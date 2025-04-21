type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

type ('a, 'b) dtype =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Complex32 : (Complex.t, complex32_elt) dtype
  | Complex64 : (Complex.t, complex64_elt) dtype

type layout = C_contiguous | Strided

type ('a, 'b) descriptor = {
  dtype : ('a, 'b) dtype;
  shape : int array;
  layout : layout;
  strides : int array;
  offset : int;
}

let float16 = Float16
let float32 = Float32
let float64 = Float64
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let int64 = Int64
let complex32 = Complex32
let complex64 = Complex64

let zero : type a b. (a, b) dtype -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 0.0
  | Float32 -> 0.0
  | Float64 -> 0.0
  | Int8 -> 0
  | UInt8 -> 0
  | Int16 -> 0
  | UInt16 -> 0
  | Int32 -> 0l
  | Int64 -> 0L
  | Complex32 -> Complex.zero
  | Complex64 -> Complex.zero

let one : type a b. (a, b) dtype -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 1.0
  | Float32 -> 1.0
  | Float64 -> 1.0
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 1
  | UInt16 -> 1
  | Int32 -> 1l
  | Int64 -> 1L
  | Complex32 -> Complex.one
  | Complex64 -> Complex.one

let pp_int_array fmt arr =
  let open Format in
  let n = Array.length arr in
  fprintf fmt "[";
  pp_open_hovbox fmt 1;
  Array.iteri
    (fun i x ->
      pp_print_int fmt x;
      if i < n - 1 then (
        pp_print_string fmt ";";
        pp_print_space fmt ()))
    arr;
  pp_close_box fmt ();
  fprintf fmt "]"

let compute_c_strides shape =
  let n = Array.length shape in
  if n = 0 then [||]
  else
    let strides = Array.make n 0 in
    strides.(n - 1) <- 1;
    for k = n - 2 downto 0 do
      strides.(k) <- strides.(k + 1) * shape.(k + 1)
    done;
    strides

let check_c_contiguity_from_shape_strides shape strides =
  let n = Array.length shape in
  if n = 0 then true
  else if Array.length strides <> n then false
  else
    let expected_stride = ref 1 in
    let is_contig = ref true in
    for i = n - 1 downto 0 do
      let dim_size = shape.(i) in
      if dim_size = 0 then expected_stride := 0
      else if dim_size = 1 then (
        if strides.(i) <> !expected_stride then is_contig := false;
        expected_stride := !expected_stride * 1)
      else (
        if strides.(i) <> !expected_stride then is_contig := false;
        expected_stride := !expected_stride * dim_size)
    done;
    !is_contig

let is_c_contiguous desc =
  desc.offset = 0
  && check_c_contiguity_from_shape_strides desc.shape desc.strides

let is_scalar desc =
  let n = Array.length desc.shape in
  if n = 0 then true else if n = 1 && desc.shape.(0) = 1 then true else false

let linear_to_md_c_contig k shape =
  let n = Array.length shape in
  let md_index = Array.make n 0 in
  let temp = ref k in
  for i = n - 1 downto 1 do
    md_index.(i) <- !temp mod shape.(i);
    temp := !temp / shape.(i)
  done;
  md_index.(0) <- !temp;
  md_index

let md_to_linear md_index strides =
  let n = Array.length md_index in
  let idx = ref 0 in
  for i = 0 to n - 1 do
    idx := !idx + (md_index.(i) * strides.(i))
  done;
  !idx

let compute_broadcast_index target_multi_idx source_shape =
  let target_ndim = Array.length target_multi_idx in
  let source_ndim = Array.length source_shape in
  let source_multi_idx = Array.make source_ndim 0 in
  for i = 0 to source_ndim - 1 do
    let target_idx_pos = target_ndim - source_ndim + i in
    let source_idx_pos = i in
    if source_idx_pos < 0 || target_idx_pos < 0 then ()
    else if source_shape.(source_idx_pos) = 1 then
      source_multi_idx.(source_idx_pos) <- 0
    else source_multi_idx.(source_idx_pos) <- target_multi_idx.(target_idx_pos)
  done;
  source_multi_idx

let multi_index_from_linear linear_idx shape =
  let ndim = Array.length shape in
  let index = Array.make ndim 0 in
  if ndim = 0 then index
  else if Array.exists (( = ) 0) shape then index
  else
    let current_linear_idx = ref linear_idx in
    let strides = Array.make ndim 1 in
    for i = ndim - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * shape.(i + 1)
    done;

    for i = 0 to ndim - 1 do
      if strides.(i) = 0 then index.(i) <- 0
      else (
        index.(i) <- !current_linear_idx / strides.(i);
        current_linear_idx := !current_linear_idx mod strides.(i))
    done;
    index

(* Helper to iterate through multi-dimensional indices based on shape *)
let iter_multi_indices shape f =
  let ndim = Array.length shape in
  if ndim = 0 then (if Array.fold_left ( * ) 1 shape > 0 then f [||])
  else
    let current_indices = Array.make ndim 0 in
    let sizes = shape in
    let rec loop dim_idx =
      if dim_idx = ndim then f (Array.copy current_indices)
      else
        let current_dim_size = sizes.(dim_idx) in
        if current_dim_size = 0 then ()
        else
          for i = 0 to current_dim_size - 1 do
            current_indices.(dim_idx) <- i;
            loop (dim_idx + 1)
          done
    in
    if Array.exists (( = ) 0) shape then () else loop 0

let add_dtype : type a b. (a, b) dtype -> a -> a -> a =
 fun dtype x y ->
  match dtype with
  | Float16 -> x +. y
  | Float32 -> x +. y
  | Float64 -> x +. y
  | Int8 -> x + y
  | UInt8 -> x + y
  | Int16 -> x + y
  | UInt16 -> x + y
  | Int32 -> Int32.add x y
  | Int64 -> Int64.add x y
  | Complex32 -> Complex.add x y
  | Complex64 -> Complex.add x y

let mul_dtype : type a b. (a, b) dtype -> a -> a -> a =
 fun dtype x y ->
  match dtype with
  | Float16 -> x *. y
  | Float32 -> x *. y
  | Float64 -> x *. y
  | Int8 -> x * y
  | UInt8 -> x * y
  | Int16 -> x * y
  | UInt16 -> x * y
  | Int32 -> Int32.mul x y
  | Int64 -> Int64.mul x y
  | Complex32 -> Complex.mul x y
  | Complex64 -> Complex.mul x y

let sub_dtype : type a b. (a, b) dtype -> a -> a -> a =
 fun dtype x y ->
  match dtype with
  | Float16 -> x -. y
  | Float32 -> x -. y
  | Float64 -> x -. y
  | Int8 -> x - y
  | UInt8 -> x - y
  | Int16 -> x - y
  | UInt16 -> x - y
  | Int32 -> Int32.sub x y
  | Int64 -> Int64.sub x y
  | Complex32 -> Complex.sub x y
  | Complex64 -> Complex.sub x y

let div_dtype : type a b. (a, b) dtype -> a -> a -> a =
 fun dtype x y ->
  match dtype with
  | Float16 -> x /. y
  | Float32 -> x /. y
  | Float64 -> x /. y
  | Int8 -> x / y
  | UInt8 -> x / y
  | Int16 -> x / y
  | UInt16 -> x / y
  | Int32 -> Int32.div x y
  | Int64 -> Int64.div x y
  | Complex32 -> Complex.div x y
  | Complex64 -> Complex.div x y

let fma_dtype (type a b) (dtype : (a, b) dtype) (a : a) (b : a) (c : a) : a =
  match dtype with
  | Float16 -> c +. (a *. b)
  | Float32 -> c +. (a *. b)
  | Float64 -> c +. (a *. b)
  | Int8 -> c + (a * b)
  | Int16 -> c + (a * b)
  | Int32 -> Int32.add c (Int32.mul a b)
  | Int64 -> Int64.add c (Int64.mul a b)
  | UInt8 -> c + (a * b)
  | UInt16 -> c + (a * b)
  | Complex32 -> Complex.add c (Complex.mul a b)
  | Complex64 -> Complex.add c (Complex.mul a b)

let shape desc = desc.shape
let dtype desc = desc.dtype
let strides desc = desc.strides
let offset desc = desc.offset
let layout desc = desc.layout

let size desc =
  let n = Array.length desc.shape in
  if n = 0 then 1 else Array.fold_left ( * ) 1 desc.shape

let stride axis desc =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length desc.strides then invalid_arg "axis out of bounds";
  let stride = Array.unsafe_get desc.strides axis in
  stride

let dims desc = desc.shape

let dim axis desc =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length desc.shape then invalid_arg "axis out of bounds";
  let dim = Array.unsafe_get desc.shape axis in
  dim

let ndim desc = Array.length desc.shape

let itemsize : type a b. (a, b) descriptor -> int =
 fun desc ->
  match desc.dtype with
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 4
  | Int64 -> 8
  | Complex32 -> 8
  | Complex64 -> 16

let nbytes : type a b. (a, b) descriptor -> int =
 fun desc ->
  let n = Array.length desc.shape in
  if n = 0 then 0
  else
    let size = Array.fold_left ( * ) 1 desc.shape in
    size * itemsize desc

let broadcast_shapes shape1 shape2 =
  let ndim1 = Array.length shape1 in
  let ndim2 = Array.length shape2 in
  let max_ndim = max ndim1 ndim2 in
  let padded_shape1 = Array.make max_ndim 1 in
  let padded_shape2 = Array.make max_ndim 1 in
  Array.blit shape1 0 padded_shape1 (max_ndim - ndim1) ndim1;
  Array.blit shape2 0 padded_shape2 (max_ndim - ndim2) ndim2;

  let out_shape = Array.make max_ndim 0 in
  for i = 0 to max_ndim - 1 do
    let d1 = padded_shape1.(i) in
    let d2 = padded_shape2.(i) in
    if d1 = d2 then out_shape.(i) <- d1
    else if d1 = 1 then out_shape.(i) <- d2
    else if d2 = 1 then out_shape.(i) <- d1
    else
      failwith
        (Format.asprintf "Shapes %a and %a cannot be broadcast together"
           pp_int_array padded_shape1 pp_int_array padded_shape2)
  done;
  out_shape
