(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

let err_void_bounds = "void has no numeric bounds"

type t =
  | Void
  | Weakint
  | Bool
  | Int8
  | Int16
  | Int32
  | Int64
  | Uint8
  | Uint16
  | Uint32
  | Uint64
  | Weakfloat
  | Fp8e4m3
  | Fp8e5m2
  | Fp8e4m3fnuz
  | Fp8e5m2fnuz
  | Float16
  | Bfloat16
  | Float32
  | Float64

type addr_space = Global | Local | Reg | Alu

(* Properties *)

let bitsize = function
  | Void -> 0
  | Bool -> 1
  | Int8 | Uint8 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> 8
  | Int16 | Uint16 | Float16 | Bfloat16 -> 16
  | Int32 | Uint32 | Float32 -> 32
  | Int64 | Uint64 | Float64 -> 64
  | Weakint | Weakfloat -> 800

let itemsize dt = (bitsize dt + 7) / 8

let priority = function
  | Void -> -1
  | Weakint | Bool -> 0
  | Int8 -> 1
  | Uint8 -> 2
  | Int16 -> 3
  | Uint16 -> 4
  | Int32 -> 5
  | Uint32 -> 6
  | Int64 -> 7
  | Uint64 -> 8
  | Weakfloat -> 9
  | Fp8e4m3 | Fp8e4m3fnuz -> 10
  | Fp8e5m2 | Fp8e5m2fnuz -> 11
  | Float16 -> 12
  | Bfloat16 -> 13
  | Float32 -> 14
  | Float64 -> 15

(* Comparison *)

let compare (a : t) (b : t) =
  let c = Int.compare (priority a) (priority b) in
  if c <> 0 then c
  else
    let c = Int.compare (bitsize a) (bitsize b) in
    if c <> 0 then c else Stdlib.compare a b

let equal (a : t) (b : t) = a = b

(* Predicates *)

let is_float = function
  | Float16 | Bfloat16 | Float32 | Float64
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz | Weakfloat -> true
  | _ -> false

let is_fp8 = function
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> true
  | _ -> false

let is_int = function
  | Int8 | Int16 | Int32 | Int64
  | Uint8 | Uint16 | Uint32 | Uint64 | Weakint -> true
  | _ -> false

let is_unsigned = function
  | Uint8 | Uint16 | Uint32 | Uint64 -> true
  | _ -> false

let is_bool = function Bool -> true | _ -> false
let is_weak = function Weakint | Weakfloat -> true | _ -> false

(* Environment-configured dtypes *)

let default_float_ref = ref Float32
let default_int_ref = ref Int32

let of_string s =
  match String.lowercase_ascii s with
  | "void" -> Some Void
  | "weakint" -> Some Weakint
  | "weakfloat" -> Some Weakfloat
  | "bool" -> Some Bool
  | "int8" | "char" -> Some Int8
  | "int16" | "short" -> Some Int16
  | "int32" | "int" -> Some Int32
  | "int64" | "long" -> Some Int64
  | "uint8" | "uchar" -> Some Uint8
  | "uint16" | "ushort" -> Some Uint16
  | "uint32" | "uint" -> Some Uint32
  | "uint64" | "ulong" -> Some Uint64
  | "float16" | "half" -> Some Float16
  | "bfloat16" -> Some Bfloat16
  | "float32" | "float" -> Some Float32
  | "float64" | "double" -> Some Float64
  | "default_float" -> Some !default_float_ref
  | "default_int" -> Some !default_int_ref
  | "fp8e4m3" -> Some Fp8e4m3
  | "fp8e5m2" -> Some Fp8e5m2
  | "fp8e4m3fnuz" -> Some Fp8e4m3fnuz
  | "fp8e5m2fnuz" -> Some Fp8e5m2fnuz
  | _ -> None

let env_dtype ~key ~default ~accept =
  match Sys.getenv_opt key with
  | None | Some "" -> default
  | Some s -> (
      match of_string s with
      | Some dt when accept dt -> dt
      | _ -> invalid_arg (strf "%s: invalid dtype %S" key s))

let default_float () =
  let dt = env_dtype ~key:"DEFAULT_FLOAT" ~default:Float32 ~accept:is_float in
  default_float_ref := dt;
  dt

let default_int () =
  let dt = env_dtype ~key:"DEFAULT_INT" ~default:Int32 ~accept:is_int in
  default_int_ref := dt;
  dt

let sum_dtype () =
  env_dtype ~key:"SUM_DTYPE" ~default:Float32 ~accept:(fun _ -> true)

(* Named dtypes *)

let void = Void
let bool = Bool
let int8 = Int8
let int16 = Int16
let int32 = Int32
let int64 = Int64
let uint8 = Uint8
let uint16 = Uint16
let uint32 = Uint32
let uint64 = Uint64
let float16 = Float16
let bfloat16 = Bfloat16
let float32 = Float32
let float64 = Float64
let fp8e4m3 = Fp8e4m3
let fp8e5m2 = Fp8e5m2
let fp8e4m3fnuz = Fp8e4m3fnuz
let fp8e5m2fnuz = Fp8e5m2fnuz
let weakint = Weakint
let weakfloat = Weakfloat
let default_float = default_float ()
let default_int = default_int ()

let strong_dtype = function
  | Weakint -> default_int
  | Weakfloat -> default_float
  | dt -> dt

let weak_dtype dt =
  if is_float dt then Weakfloat else if is_int dt then Weakint else dt

(* Promotion lattice *)

let promo_lattice =
  [ Bool, [ Weakint ];
    Weakint, [ Int8; Uint8 ];
    Int8, [ Int16 ];       Int16, [ Int32 ];
    Int32, [ Int64 ];      Int64, [ Weakfloat ];
    Uint8, [ Int16; Uint16 ];
    Uint16, [ Int32; Uint32 ];
    Uint32, [ Int64; Uint64 ];
    Uint64, [ Weakfloat ];
    Weakfloat, [ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ];
    Fp8e4m3, [ Float16; Bfloat16 ];
    Fp8e5m2, [ Float16; Bfloat16 ];
    Fp8e4m3fnuz, [ Float16; Bfloat16 ];
    Fp8e5m2fnuz, [ Float16; Bfloat16 ];
    Float16, [ Float32 ];  Bfloat16, [ Float32 ];
    Float32, [ Float64 ] ]

module Dtype_set = Set.Make (struct
  type nonrec t = t
  let compare = Stdlib.compare
end)

let ancestor_cache : (t, Dtype_set.t) Hashtbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Hashtbl.create 16)

let rec ancestors s =
  let ancestor_cache = Domain.DLS.get ancestor_cache in
  match Hashtbl.find_opt ancestor_cache s with
  | Some set -> set
  | None ->
      let parents = Option.value ~default:[] (List.assoc_opt s promo_lattice) in
      let set =
        List.fold_left
          (fun acc p -> Dtype_set.union acc (ancestors p))
          (Dtype_set.singleton s) parents
      in
      Hashtbl.add ancestor_cache s set;
      set

let min_by_priority dtypes =
  Dtype_set.fold
    (fun s best ->
      match best with
      | None -> Some s
      | Some b when compare s b < 0 -> Some s
      | _ -> best)
    dtypes None

let least_upper_dtype dts =
  match dts with
  | [] -> invalid_arg "Dtype.least_upper_dtype: empty list"
  | [ d ] -> d
  | first :: rest ->
      let intersection =
        List.fold_left
          (fun acc d -> Dtype_set.inter acc (ancestors d))
          (ancestors first) rest
      in
      (match min_by_priority intersection with
      | Some s -> s
      | None -> invalid_arg "Dtype.least_upper_dtype: no common ancestor")

let least_upper_float dt =
  if dt = Weakint then Weakfloat
  else if is_float dt then dt
  else least_upper_dtype [ dt; default_float ]

let can_lossless_cast (dt0 : t) (dt1 : t) =
  dt0 = dt1 || dt0 = Bool
  ||
  match dt1 with
  | Weakint ->
      List.mem dt0 [ Uint8; Uint16; Uint32; Uint64; Int8; Int16; Int32; Int64 ]
  | Float64 ->
      List.mem dt0
        [ Float32; Float16; Bfloat16; Fp8e4m3; Fp8e5m2;
          Fp8e4m3fnuz; Fp8e5m2fnuz;
          Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Float32 ->
      List.mem dt0
        [ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz;
          Uint16; Uint8; Int16; Int8 ]
  | Float16 ->
      List.mem dt0
        [ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz; Uint8; Int8 ]
  | Uint64 -> List.mem dt0 [ Uint32; Uint16; Uint8 ]
  | Uint32 -> List.mem dt0 [ Uint16; Uint8 ]
  | Uint16 -> dt0 = Uint8
  | Int64 -> List.mem dt0 [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Int32 -> List.mem dt0 [ Uint16; Uint8; Int16; Int8 ]
  | Int16 -> List.mem dt0 [ Uint8; Int8 ]
  | _ -> false

let sum_acc_dtype dt =
  if is_unsigned dt then least_upper_dtype [ dt; uint32 ]
  else if is_int dt || is_bool dt then least_upper_dtype [ dt; int32 ]
  else least_upper_dtype [ dt; sum_dtype () ]

(* Bounds *)

type bound = [ `Bool of bool | `Int of Z.t | `Float of float ]

let max (dt : t) =
  match dt with
  | Bool -> `Bool true
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      `Int (Z.pred (Z.shift_left Z.one (bitsize dt)))
  | Int8 | Int16 | Int32 | Int64 | Weakint ->
      `Int (Z.pred (Z.shift_left Z.one (bitsize dt - 1)))
  | Fp8e4m3 -> `Float 448.
  | Fp8e4m3fnuz -> `Float 240.
  | Fp8e5m2fnuz -> `Float 57344.
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e5m2 | Weakfloat ->
      `Float infinity
  | Void -> invalid_arg err_void_bounds

let min (dt : t) =
  match max dt with
  | `Bool _ -> `Bool false
  | `Int n -> `Int (if is_unsigned dt then Z.zero else Z.neg (Z.succ n))
  | `Float f -> `Float (-. f)

let finfo = function
  | Float16 -> 5, 10
  | Bfloat16 -> 8, 7
  | Float32 -> 8, 23
  | Float64 -> 11, 52
  | Fp8e5m2 | Fp8e5m2fnuz -> 5, 2
  | Fp8e4m3 | Fp8e4m3fnuz -> 4, 3
  | _ -> invalid_arg "finfo: not a floating-point dtype"

(* Formatting *)

let to_string = function
  | Void -> "void"   | Bool -> "bool"   | Weakint -> "weakint"
  | Weakfloat -> "weakfloat"
  | Int8 -> "i8"     | Int16 -> "i16"   | Int32 -> "i32"   | Int64 -> "i64"
  | Uint8 -> "u8"    | Uint16 -> "u16"  | Uint32 -> "u32"  | Uint64 -> "u64"
  | Float16 -> "f16" | Bfloat16 -> "bf16"
  | Float32 -> "f32" | Float64 -> "f64"
  | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"
  | Fp8e4m3fnuz -> "fp8e4m3fnuz" | Fp8e5m2fnuz -> "fp8e5m2fnuz"

let repr_name = function
  | Void -> "void" | Weakint -> "weakint"
  | Weakfloat -> "weakfloat" | Bool -> "bool"
  | Int8 -> "char" | Int16 -> "short" | Int32 -> "int" | Int64 -> "long"
  | Uint8 -> "uchar" | Uint16 -> "ushort" | Uint32 -> "uint"
  | Uint64 -> "ulong"
  | Float16 -> "half" | Bfloat16 -> "bfloat16"
  | Float32 -> "float" | Float64 -> "double"
  | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"
  | Fp8e4m3fnuz -> "fp8e4m3fnuz"
  | Fp8e5m2fnuz -> "fp8e5m2fnuz"

let repr dt = strf "dtypes.%s" (repr_name dt)
let pp fmt dt = Format.pp_print_string fmt (to_string dt)

let addr_space_to_string = function
  | Global -> "global" | Local -> "local" | Reg -> "reg" | Alu -> "alu"

let pp_addr_space fmt a = Format.pp_print_string fmt (addr_space_to_string a)

(* Storage formats *)

let to_scalar : t -> Nx_dtype.Scalar.t option = function
  | Bool -> Some Bool
  | Int8 -> Some Int8
  | Int16 -> Some Int16
  | Int32 -> Some Int32
  | Int64 -> Some Int64
  | Uint8 -> Some UInt8
  | Uint16 -> Some UInt16
  | Uint32 -> Some UInt32
  | Uint64 -> Some UInt64
  | Fp8e4m3 -> Some Float8_e4m3
  | Fp8e5m2 -> Some Float8_e5m2
  | Fp8e4m3fnuz -> Some Float8_e4m3fnuz
  | Fp8e5m2fnuz -> Some Float8_e5m2fnuz
  | Float16 -> Some Float16
  | Bfloat16 -> Some BFloat16
  | Float32 -> Some Float32
  | Float64 -> Some Float64
  | Void | Weakint | Weakfloat -> None

let of_scalar : Nx_dtype.Scalar.t -> t option = function
  | Bool -> Some Bool
  | Int8 -> Some Int8
  | Int16 -> Some Int16
  | Int32 -> Some Int32
  | Int64 -> Some Int64
  | UInt8 -> Some Uint8
  | UInt16 -> Some Uint16
  | UInt32 -> Some Uint32
  | UInt64 -> Some Uint64
  | Float8_e4m3 -> Some Fp8e4m3
  | Float8_e5m2 -> Some Fp8e5m2
  | Float8_e4m3fnuz -> Some Fp8e4m3fnuz
  | Float8_e5m2fnuz -> Some Fp8e5m2fnuz
  | Float16 -> Some Float16
  | BFloat16 -> Some Bfloat16
  | Float32 -> Some Float32
  | Float64 -> Some Float64
  | Int4 | UInt4 | Complex64 | Complex128 -> None

let scalar dt =
  match to_scalar dt with
  | Some s -> s
  | None -> invalid_arg (strf "%s has no storage format" (to_string dt))

(* Rounding *)

let truncate_float (dt : t) x =
  match dt with
  | Float64 | Weakfloat -> x
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float x)
  | Float16 | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      let s = scalar dt in
      Nx_dtype.Scalar.decode s (Nx_dtype.Scalar.encode s x)
  | _ -> invalid_arg "truncate_float: not a floating-point dtype"

let truncate_int (dt : t) x =
  let b = bitsize dt in
  match dt with
  | Bool -> if x <> 0 then 1 else 0
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= Sys.int_size then x else x land ((1 lsl b) - 1)
  | Int8 | Int16 | Int32 | Int64 | Weakint ->
      if b >= Sys.int_size then x
      else
        let mask = (1 lsl b) - 1 in
        let unsigned = x land mask in
        if unsigned land (1 lsl (b - 1)) <> 0 then unsigned lor lnot mask
        else unsigned
  | _ -> invalid_arg "truncate_int: not an integer or bool dtype"

let truncate_integer (dt : t) x =
  if dt = Weakint then x
  else if dt = Bool then if Z.equal x Z.zero then Z.zero else Z.one
  else if is_unsigned dt then Z.extract x 0 (bitsize dt)
  else if is_int dt then Z.signed_extract x 0 (bitsize dt)
  else invalid_arg "truncate_integer: not an integer or bool dtype"

(* Storage conversion *)

type storage_scalar = [ `Bool of bool | `Float of float | `Int of int64 ]

let storage_fmt_for_dtype (dt : t) =
  match dt with
  | Bool -> Some '?'
  | Int8 -> Some 'b'
  | Uint8 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> Some 'B'
  | Int16 -> Some 'h'
  | Uint16 | Bfloat16 -> Some 'H'
  | Int32 -> Some 'i'
  | Uint32 -> Some 'I'
  | Int64 -> Some 'q'
  | Uint64 -> Some 'Q'
  | Float16 -> Some 'e'
  | Float32 -> Some 'f'
  | Float64 -> Some 'd'
  | Void | Weakint | Weakfloat -> None

let storage_bool = function
  | `Bool b -> b
  | `Int n -> n <> 0L
  | `Float f -> f <> 0.0

let storage_float = function
  | `Bool b -> if b then 1.0 else 0.0
  | `Int n -> Int64.to_float n
  | `Float f -> f

let storage_int64 = function
  | `Bool b -> if b then 1L else 0L
  | `Int n -> n
  | `Float f -> Int64.of_float f

let truncate_int64 (dt : t) x =
  let b = bitsize dt in
  match dt with
  | Bool -> if x <> 0L then 1L else 0L
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= 64 then x else Int64.logand x Int64.(sub (shift_left 1L b) 1L)
  | Int8 | Int16 | Int32 | Int64 | Weakint ->
      if b >= 64 then x
      else
        let mask = Int64.(sub (shift_left 1L b) 1L) in
        let unsigned = Int64.logand x mask in
        if Int64.logand unsigned (Int64.shift_left 1L (b - 1)) <> 0L then
          Int64.logor unsigned (Int64.lognot mask)
        else unsigned
  | _ -> invalid_arg "truncate: not an integer or bool dtype"

let to_storage_scalar (dt : t) x =
  match dt with
  | Bool -> `Bool (storage_bool x)
  | Float16 -> `Float (truncate_float dt (storage_float x))
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Int (Int64.of_int (Nx_dtype.Scalar.encode (scalar dt) (storage_float x)))
  | Float32 | Float64 | Weakfloat -> `Float (storage_float x)
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Weakint -> `Int (storage_int64 x)
  | Void -> invalid_arg "to_storage_scalar: void has no storage scalar"

let from_storage_scalar x (dt : t) =
  match dt with
  | Bool -> `Bool (storage_bool x)
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      let mask = Int64.pred (Int64.shift_left 1L (bitsize dt)) in
      let bits = Int64.to_int (Int64.logand (storage_int64 x) mask) in
      `Float (Nx_dtype.Scalar.decode (scalar dt) bits)
  | Float16 | Float32 | Float64 | Weakfloat -> `Float (storage_float x)
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Weakint -> `Int (storage_int64 x)
  | Void -> invalid_arg "from_storage_scalar: void has no storage scalar"

let truncate (dt : t) x =
  match dt with
  | Bool -> `Bool (storage_bool x)
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz
  | Fp8e5m2fnuz | Weakfloat -> `Float (truncate_float dt (storage_float x))
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Weakint -> `Int (truncate_int64 dt (storage_int64 x))
  | Void -> invalid_arg "truncate: void has no storage scalar"
