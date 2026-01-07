(* Extended Bigarray module with additional types *)

(* Re-export all standard bigarray types first *)
include Stdlib.Bigarray

(* Additional element types - following Bigarray naming convention *)
type bfloat16_elt = Bfloat16_elt
type bool_elt = Bool_elt
type int4_signed_elt = Int4_signed_elt
type int4_unsigned_elt = Int4_unsigned_elt
type float8_e4m3_elt = Float8_e4m3_elt (* 4 exponent bits, 3 mantissa bits *)
type float8_e5m2_elt = Float8_e5m2_elt (* 5 exponent bits, 2 mantissa bits *)
type uint32_elt = Uint32_elt
type uint64_elt = Uint64_elt

(* Shadow the kind type to include our new types *)
type ('a, 'b) kind =
  | Float32 : (float, float32_elt) kind
  | Float64 : (float, float64_elt) kind
  | Int8_signed : (int, int8_signed_elt) kind
  | Int8_unsigned : (int, int8_unsigned_elt) kind
  | Int16_signed : (int, int16_signed_elt) kind
  | Int16_unsigned : (int, int16_unsigned_elt) kind
  | Int32 : (int32, int32_elt) kind
  | Int64 : (int64, int64_elt) kind
  | Int : (int, int_elt) kind
  | Nativeint : (nativeint, nativeint_elt) kind
  | Complex32 : (Complex.t, complex32_elt) kind
  | Complex64 : (Complex.t, complex64_elt) kind
  | Char : (char, int8_unsigned_elt) kind
  | Float16 : (float, float16_elt) kind
  | Bfloat16 : (float, bfloat16_elt) kind
  | Bool : (bool, bool_elt) kind
  | Int4_signed : (int, int4_signed_elt) kind
  | Int4_unsigned : (int, int4_unsigned_elt) kind
  | Float8_e4m3 : (float, float8_e4m3_elt) kind
  | Float8_e5m2 : (float, float8_e5m2_elt) kind
  | Uint32 : (int32, uint32_elt) kind
  | Uint64 : (int64, uint64_elt) kind

(* Shadow the value constructors *)
let float32 = Float32
let float64 = Float64
let int8_signed = Int8_signed
let int8_unsigned = Int8_unsigned
let int16_signed = Int16_signed
let int16_unsigned = Int16_unsigned
let int32 = Int32
let int64 = Int64
let int = Int
let nativeint = Nativeint
let complex32 = Complex32
let complex64 = Complex64
let char = Char
let float16 = Float16
let bfloat16 = Bfloat16
let bool = Bool
let int4_signed = Int4_signed
let int4_unsigned = Int4_unsigned
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2
let uint32 = Uint32
let uint64 = Uint64

(* Shadow kind_size_in_bytes to handle new types *)
let kind_size_in_bytes : type a b. (a, b) kind -> int = function
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | Int8_signed -> 1
  | Int8_unsigned -> 1
  | Int16_signed -> 2
  | Int16_unsigned -> 2
  | Int32 -> 4
  | Int64 -> 8
  | Int -> Sys.word_size / 8
  | Nativeint -> Sys.word_size / 8
  | Complex32 -> 8
  | Complex64 -> 16
  | Char -> 1
  | Bfloat16 -> 2
  | Bool -> 1
  | Int4_signed -> 1 (* 2 values packed per byte *)
  | Int4_unsigned -> 1 (* 2 values packed per byte *)
  | Float8_e4m3 -> 1
  | Float8_e5m2 -> 1
  | Uint32 -> 4
  | Uint64 -> 8

(* Convert our extended kind to stdlib kind for fallback *)
let to_stdlib_kind : type a b. (a, b) kind -> (a, b) Stdlib.Bigarray.kind option
    = function
  | Float32 -> Some Stdlib.Bigarray.Float32
  | Float64 -> Some Stdlib.Bigarray.Float64
  | Int8_signed -> Some Stdlib.Bigarray.Int8_signed
  | Int8_unsigned -> Some Stdlib.Bigarray.Int8_unsigned
  | Int16_signed -> Some Stdlib.Bigarray.Int16_signed
  | Int16_unsigned -> Some Stdlib.Bigarray.Int16_unsigned
  | Int32 -> Some Stdlib.Bigarray.Int32
  | Int64 -> Some Stdlib.Bigarray.Int64
  | Int -> Some Stdlib.Bigarray.Int
  | Nativeint -> Some Stdlib.Bigarray.Nativeint
  | Complex32 -> Some Stdlib.Bigarray.Complex32
  | Complex64 -> Some Stdlib.Bigarray.Complex64
  | Char -> Some Stdlib.Bigarray.Char
  | Float16 -> Some Stdlib.Bigarray.Float16
  | Bfloat16 -> None
  | Bool -> None
  | Int4_signed -> None
  | Int4_unsigned -> None
  | Float8_e4m3 -> None
  | Float8_e5m2 -> None
  | Uint32 -> None
  | Uint64 -> None

(* External functions for creating arrays with new types *)
external create_bfloat16_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t
  = "caml_nx_ba_create_bfloat16"

external create_bool_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t = "caml_nx_ba_create_bool"

external create_int4_signed_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t
  = "caml_nx_ba_create_int4_signed"

external create_int4_unsigned_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t
  = "caml_nx_ba_create_int4_unsigned"

external create_float8_e4m3_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t
  = "caml_nx_ba_create_float8_e4m3"

external create_float8_e5m2_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t
  = "caml_nx_ba_create_float8_e5m2"

external create_uint32_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t = "caml_nx_ba_create_uint32"

external create_uint64_genarray :
  'c layout -> int array -> ('a, 'b, 'c) Genarray.t = "caml_nx_ba_create_uint64"

(* External functions for get/set operations on extended types *)
external nx_ba_get_generic : ('a, 'b, 'c) Genarray.t -> int array -> 'a
  = "caml_nx_ba_get_generic"

external nx_ba_set_generic : ('a, 'b, 'c) Genarray.t -> int array -> 'a -> unit
  = "caml_nx_ba_set_generic"

(* External function to get extended kind - needs C stub implementation *)
external nx_ba_kind : ('a, 'b, 'c) Genarray.t -> ('a, 'b) kind
  = "caml_nx_ba_kind"

(* Shadow the Genarray module *)
module Genarray = struct
  include Stdlib.Bigarray.Genarray

  (* Shadow create to handle new types *)
  let create : type a b c. (a, b) kind -> c layout -> int array -> (a, b, c) t =
   fun kind layout dims ->
    match kind with
    | Bfloat16 -> create_bfloat16_genarray layout dims
    | Bool -> create_bool_genarray layout dims
    | Int4_signed -> create_int4_signed_genarray layout dims
    | Int4_unsigned -> create_int4_unsigned_genarray layout dims
    | Float8_e4m3 -> create_float8_e4m3_genarray layout dims
    | Float8_e5m2 -> create_float8_e5m2_genarray layout dims
    | Uint32 -> create_uint32_genarray layout dims
    | Uint64 -> create_uint64_genarray layout dims
    | _ -> (
        match to_stdlib_kind kind with
        | Some k -> Stdlib.Bigarray.Genarray.create k layout dims
        | None -> failwith "Internal error: unhandled kind")

  (* Override kind to return Bigarray_ext.kind *)
  let kind : type a b c. (a, b, c) t -> (a, b) kind = nx_ba_kind

  (* Shadow get to handle extended types *)
  let get arr idx = nx_ba_get_generic arr idx

  (* Shadow set to handle extended types *)
  let set arr idx value = nx_ba_set_generic arr idx value

  (* Shadow init function *)
  let init (type t) kind (layout : t layout) dims f =
    let arr = create kind layout dims in
    let dlen = Array.length dims in
    match layout with
    | C_layout ->
        let rec cloop arr idx f col max =
          if col = Array.length idx then set arr idx (f idx)
          else
            for j = 0 to pred max.(col) do
              idx.(col) <- j;
              cloop arr idx f (succ col) max
            done
        in
        cloop arr (Array.make dlen 0) f 0 dims;
        arr
    | Fortran_layout ->
        let rec floop arr idx f col max =
          if col < 0 then set arr idx (f idx)
          else
            for j = 1 to max.(col) do
              idx.(col) <- j;
              floop arr idx f (pred col) max
            done
        in
        floop arr (Array.make dlen 1) f (pred dlen) dims;
        arr

  (* size_in_bytes needs to use our extended kind_size_in_bytes *)
  let size_in_bytes arr =
    (* We can't get the extended kind from the array, so we keep the original *)
    Stdlib.Bigarray.Genarray.size_in_bytes arr

  (* Override blit to handle extended types *)
  external nx_ba_blit_genarray : ('a, 'b, 'c) t -> ('a, 'b, 'c) t -> unit
    = "caml_nx_ba_blit"

  let blit = nx_ba_blit_genarray

  (* Override fill for extended types *)
  external nx_ba_fill : ('a, 'b, 'c) t -> 'a -> unit = "caml_nx_ba_fill"

  let fill = nx_ba_fill

  external unsafe_blit_from_bytes :
    bytes -> int -> ('a, 'b, 'c) t -> int -> int -> unit
    = "caml_nx_ba_blit_from_bytes"

  external unsafe_blit_to_bytes :
    ('a, 'b, 'c) t -> int -> bytes -> int -> int -> unit
    = "caml_nx_ba_blit_to_bytes"

  let element_size arr = kind_size_in_bytes (kind arr)
  let num_elements arr = Array.fold_left ( * ) 1 (dims arr)

  let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes arr =
    let elem_size = element_size arr in
    let total = num_elements arr in
    let len = match len with Some len -> len | None -> total - dst_off in
    if len < 0 then invalid_arg "blit_from_bytes: negative length";
    unsafe_blit_from_bytes bytes (src_off * elem_size) arr (dst_off * elem_size)
      (len * elem_size)

  let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len arr bytes =
    let elem_size = element_size arr in
    let total = num_elements arr in
    let len = match len with Some len -> len | None -> total - src_off in
    if len < 0 then invalid_arg "blit_to_bytes: negative length";
    unsafe_blit_to_bytes arr (src_off * elem_size) bytes (dst_off * elem_size)
      (len * elem_size)
end

(* Shadow Array0 module *)
module Array0 = struct
  include Stdlib.Bigarray.Array0

  let create : type a b c. (a, b) kind -> c layout -> (a, b, c) t =
   fun kind layout -> array0_of_genarray (Genarray.create kind layout [||])

  (* Override get and set to use our extended functions *)
  let get arr = Genarray.get (genarray_of_array0 arr) [||]
  let set arr v = Genarray.set (genarray_of_array0 arr) [||] v

  let init kind layout f =
    let a = create kind layout in
    Genarray.set (genarray_of_array0 a) [||] (f ());
    a

  let of_value = init

  let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes arr =
    Genarray.blit_from_bytes ~src_off ~dst_off ?len bytes
      (genarray_of_array0 arr)

  let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len arr bytes =
    Genarray.blit_to_bytes ~src_off ~dst_off ?len (genarray_of_array0 arr) bytes
end

(* Shadow Array1 module *)
module Array1 = struct
  include Stdlib.Bigarray.Array1

  let create : type a b c. (a, b) kind -> c layout -> int -> (a, b, c) t =
   fun kind layout dim ->
    array1_of_genarray (Genarray.create kind layout [| dim |])

  (* Override kind to return Bigarray_ext.kind *)
  let kind : type a b c. (a, b, c) t -> (a, b) kind =
   fun arr -> Genarray.kind (genarray_of_array1 arr)

  (* Override get and set to use our extended functions *)
  let get arr i = Genarray.get (genarray_of_array1 arr) [| i |]
  let set arr i v = Genarray.set (genarray_of_array1 arr) [| i |] v

  (* unsafe versions just call the safe versions for extended types *)
  let unsafe_get arr i = get arr i
  let unsafe_set arr i v = set arr i v

  (* Override blit to handle extended types *)
  external nx_ba_blit : ('a, 'b, 'c) t -> ('a, 'b, 'c) t -> unit
    = "caml_nx_ba_blit"

  let blit = nx_ba_blit

  (* Override fill for extended types *)
  external nx_ba_fill : ('a, 'b, 'c) t -> 'a -> unit = "caml_nx_ba_fill"

  let fill = nx_ba_fill

  let init (type t) kind (layout : t layout) dim f =
    let arr = create kind layout dim in
    match layout with
    | C_layout ->
        for i = 0 to pred dim do
          unsafe_set arr i (f i)
        done;
        arr
    | Fortran_layout ->
        for i = 1 to dim do
          unsafe_set arr i (f i)
        done;
        arr

  let of_array (type t) kind (layout : t layout) data =
    let ba = create kind layout (Array.length data) in
    let ofs = match layout with C_layout -> 0 | Fortran_layout -> 1 in
    for i = 0 to Array.length data - 1 do
      unsafe_set ba (i + ofs) data.(i)
    done;
    ba

  let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes arr =
    Genarray.blit_from_bytes ~src_off ~dst_off ?len bytes
      (genarray_of_array1 arr)

  let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len arr bytes =
    Genarray.blit_to_bytes ~src_off ~dst_off ?len (genarray_of_array1 arr) bytes
end

(* Shadow Array2 module *)
module Array2 = struct
  include Stdlib.Bigarray.Array2

  let create : type a b c. (a, b) kind -> c layout -> int -> int -> (a, b, c) t
      =
   fun kind layout dim1 dim2 ->
    array2_of_genarray (Genarray.create kind layout [| dim1; dim2 |])

  (* Override kind to return Bigarray_ext.kind *)
  let kind : type a b c. (a, b, c) t -> (a, b) kind =
   fun arr -> Genarray.kind (genarray_of_array2 arr)

  (* Override get and set to use our extended functions *)
  let get arr i j = Genarray.get (genarray_of_array2 arr) [| i; j |]
  let set arr i j v = Genarray.set (genarray_of_array2 arr) [| i; j |] v

  (* unsafe versions just call the safe versions for extended types *)
  let unsafe_get arr i j = get arr i j
  let unsafe_set arr i j v = set arr i j v

  (* Override blit to handle extended types *)
  external nx_ba_blit : ('a, 'b, 'c) t -> ('a, 'b, 'c) t -> unit
    = "caml_nx_ba_blit"

  let blit = nx_ba_blit

  (* Override fill for extended types *)
  external nx_ba_fill : ('a, 'b, 'c) t -> 'a -> unit = "caml_nx_ba_fill"

  let fill = nx_ba_fill

  let init (type t) kind (layout : t layout) dim1 dim2 f =
    let arr = create kind layout dim1 dim2 in
    match layout with
    | C_layout ->
        for i = 0 to pred dim1 do
          for j = 0 to pred dim2 do
            unsafe_set arr i j (f i j)
          done
        done;
        arr
    | Fortran_layout ->
        for j = 1 to dim2 do
          for i = 1 to dim1 do
            unsafe_set arr i j (f i j)
          done
        done;
        arr

  let of_array (type t) kind (layout : t layout) data =
    let dim1 = Array.length data in
    let dim2 = if dim1 = 0 then 0 else Array.length data.(0) in
    let ba = create kind layout dim1 dim2 in
    let ofs = match layout with C_layout -> 0 | Fortran_layout -> 1 in
    for i = 0 to dim1 - 1 do
      let row = data.(i) in
      if Array.length row <> dim2 then
        invalid_arg "Bigarray_ext.Array2.of_array: non-rectangular data";
      for j = 0 to dim2 - 1 do
        unsafe_set ba (i + ofs) (j + ofs) row.(j)
      done
    done;
    ba

  let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes arr =
    Genarray.blit_from_bytes ~src_off ~dst_off ?len bytes
      (genarray_of_array2 arr)

  let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len arr bytes =
    Genarray.blit_to_bytes ~src_off ~dst_off ?len (genarray_of_array2 arr) bytes
end

(* Shadow Array3 module *)
module Array3 = struct
  include Stdlib.Bigarray.Array3

  let create : type a b c.
      (a, b) kind -> c layout -> int -> int -> int -> (a, b, c) t =
   fun kind layout dim1 dim2 dim3 ->
    array3_of_genarray (Genarray.create kind layout [| dim1; dim2; dim3 |])

  (* Override kind to return Bigarray_ext.kind *)
  let kind : type a b c. (a, b, c) t -> (a, b) kind =
   fun arr -> Genarray.kind (genarray_of_array3 arr)

  (* Override get and set to use our extended functions *)
  let get arr i j k = Genarray.get (genarray_of_array3 arr) [| i; j; k |]
  let set arr i j k v = Genarray.set (genarray_of_array3 arr) [| i; j; k |] v

  (* unsafe versions just call the safe versions for extended types *)
  let unsafe_get arr i j k = get arr i j k
  let unsafe_set arr i j k v = set arr i j k v

  (* Override blit to handle extended types *)
  external nx_ba_blit : ('a, 'b, 'c) t -> ('a, 'b, 'c) t -> unit
    = "caml_nx_ba_blit"

  let blit = nx_ba_blit

  (* Override fill for extended types *)
  external nx_ba_fill : ('a, 'b, 'c) t -> 'a -> unit = "caml_nx_ba_fill"

  let fill = nx_ba_fill

  let init (type t) kind (layout : t layout) dim1 dim2 dim3 f =
    let arr = create kind layout dim1 dim2 dim3 in
    match layout with
    | C_layout ->
        for i = 0 to pred dim1 do
          for j = 0 to pred dim2 do
            for k = 0 to pred dim3 do
              unsafe_set arr i j k (f i j k)
            done
          done
        done;
        arr
    | Fortran_layout ->
        for k = 1 to dim3 do
          for j = 1 to dim2 do
            for i = 1 to dim1 do
              unsafe_set arr i j k (f i j k)
            done
          done
        done;
        arr

  let of_array (type t) kind (layout : t layout) data =
    let dim1 = Array.length data in
    let dim2 = if dim1 = 0 then 0 else Array.length data.(0) in
    let dim3 = if dim2 = 0 then 0 else Array.length data.(0).(0) in
    let ba = create kind layout dim1 dim2 dim3 in
    let ofs = match layout with C_layout -> 0 | Fortran_layout -> 1 in
    for i = 0 to dim1 - 1 do
      let row = data.(i) in
      if Array.length row <> dim2 then
        invalid_arg "Bigarray_ext.Array3.of_array: non-cubic data";
      for j = 0 to dim2 - 1 do
        let col = row.(j) in
        if Array.length col <> dim3 then
          invalid_arg "Bigarray_ext.Array3.of_array: non-cubic data";
        for k = 0 to dim3 - 1 do
          unsafe_set ba (i + ofs) (j + ofs) (k + ofs) col.(k)
        done
      done
    done;
    ba

  let blit_from_bytes ?(src_off = 0) ?(dst_off = 0) ?len bytes arr =
    Genarray.blit_from_bytes ~src_off ~dst_off ?len bytes
      (genarray_of_array3 arr)

  let blit_to_bytes ?(src_off = 0) ?(dst_off = 0) ?len arr bytes =
    Genarray.blit_to_bytes ~src_off ~dst_off ?len (genarray_of_array3 arr) bytes
end
