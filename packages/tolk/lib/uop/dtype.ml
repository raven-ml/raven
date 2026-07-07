(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

let err_void_bounds = "void has no numeric bounds"
let err_private_wide_repr name = strf "Dtype.repr: private dtype %s has no public repr" name
let err_ptr_vcount n = strf "pointer v must be >= 1 (got %d)" n
let err_ptr_unknown_size = "can't get nbytes of a pointer with unlimited size"
let err_not_image = "image operation on non-image pointer dtype"

type scalar =
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
  | Uint128
  | Uint256
  | Float16
  | Bfloat16
  | Float32
  | Float64
  | Fp8e4m3
  | Fp8e5m2
  | Fp8e4m3fnuz
  | Fp8e5m2fnuz

type addr_space = Global | Local | Reg | Alu
type image_kind = Imageh | Imagef

(* Shared scalar-level functions *)

let scalar_bitsize = function
  | Void -> 0
  | Bool -> 1
  | Int8 | Uint8 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> 8
  | Int16 | Uint16 | Float16 | Bfloat16 -> 16
  | Int32 | Uint32 | Float32 -> 32
  | Int64 | Uint64 | Float64 -> 64
  | Uint128 -> 128
  | Uint256 -> 256
  | Weakint -> 800

let scalar_priority = function
  | Void -> -1
  | Weakint | Bool -> 0
  | Int8 -> 1 | Uint8 -> 2
  | Int16 -> 3 | Uint16 -> 4
  | Int32 -> 5 | Uint32 -> 6
  | Int64 -> 7 | Uint64 | Uint128 | Uint256 -> 8
  | Fp8e4m3 | Fp8e4m3fnuz -> 9 | Fp8e5m2 | Fp8e5m2fnuz -> 10
  | Float16 -> 11 | Bfloat16 -> 12
  | Float32 -> 13 | Float64 -> 14

let scalar_compare a b =
  let c = Int.compare (scalar_priority a) (scalar_priority b) in
  if c <> 0 then c
  else
    let c = Int.compare (scalar_bitsize a) (scalar_bitsize b) in
    if c <> 0 then c else Stdlib.compare a b

let scalar_is_float = function
  | Float16 | Bfloat16 | Float32 | Float64
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> true
  | _ -> false

let scalar_is_fp8 = function
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> true
  | _ -> false

let scalar_is_fp8_fnuz = function
  | Fp8e4m3fnuz | Fp8e5m2fnuz -> true
  | _ -> false

let scalar_is_int = function
  | Int8 | Int16 | Int32 | Int64
  | Uint8 | Uint16 | Uint32 | Uint64 | Weakint -> true
  | _ -> false

let scalar_is_unsigned = function
  | Uint8 | Uint16 | Uint32 | Uint64 -> true
  | _ -> false

let scalar_is_bool = function Bool -> true | _ -> false
let default_float_scalar_ref = ref Float32

let scalar_of_string s =
  match String.lowercase_ascii s with
  | "void" -> Some Void
  | "weakint" -> Some Weakint
  | "bool" -> Some Bool
  | "int8" | "char" -> Some Int8
  | "int16" | "short" -> Some Int16
  | "int32" | "int" -> Some Int32
  | "int64" | "long" -> Some Int64
  | "uint8" | "uchar" -> Some Uint8
  | "uint16" | "ushort" -> Some Uint16
  | "uint32" | "uint" -> Some Uint32
  | "uint64" | "ulong" -> Some Uint64
  | "_uint128" -> Some Uint128
  | "_uint256" -> Some Uint256
  | "float16" | "half" -> Some Float16
  | "bfloat16" -> Some Bfloat16
  | "float32" | "float" -> Some Float32
  | "float64" | "double" -> Some Float64
  | "default_float" -> Some !default_float_scalar_ref
  | "default_int" -> Some Int32
  | "fp8e4m3" -> Some Fp8e4m3
  | "fp8e5m2" -> Some Fp8e5m2
  | "fp8e4m3fnuz" -> Some Fp8e4m3fnuz
  | "fp8e5m2fnuz" -> Some Fp8e5m2fnuz
  | _ -> None

let env_scalar ~key ~default ~accept =
  match Sys.getenv_opt key with
  | None | Some "" -> default
  | Some s -> (
      match scalar_of_string s with
      | Some scalar when accept scalar -> scalar
      | _ -> invalid_arg (strf "%s: invalid dtype %S" key s))

let default_float_scalar () =
  let scalar =
    env_scalar ~key:"DEFAULT_FLOAT" ~default:Float32 ~accept:scalar_is_float
  in
  default_float_scalar_ref := scalar;
  scalar

let sum_dtype_scalar () =
  env_scalar ~key:"SUM_DTYPE" ~default:Float32 ~accept:(fun _ -> true)

(* Promotion lattice *)

let promo_lattice =
  [ Bool, [ Weakint ];
    Weakint, [ Int8; Uint8 ];
    Int8, [ Int16 ];       Int16, [ Int32 ];
    Int32, [ Int64 ];      Int64, [ Uint64 ];
    Uint8, [ Int16; Uint16 ];
    Uint16, [ Int32; Uint32 ];
    Uint32, [ Int64; Uint64 ];
    Uint64, [ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ];
    Fp8e4m3, [ Float16; Bfloat16 ];
    Fp8e5m2, [ Float16; Bfloat16 ];
    Fp8e4m3fnuz, [ Float16; Bfloat16 ];
    Fp8e5m2fnuz, [ Float16; Bfloat16 ];
    Float16, [ Float32 ];  Bfloat16, [ Float32 ];
    Float32, [ Float64 ] ]

module Scalar_set = Set.Make (struct
  type t = scalar
  let compare = Stdlib.compare
end)

let ancestor_cache : (scalar, Scalar_set.t) Hashtbl.t = Hashtbl.create 16

let rec scalar_ancestors s =
  match Hashtbl.find_opt ancestor_cache s with
  | Some set -> set
  | None ->
      let parents = Option.value ~default:[] (List.assoc_opt s promo_lattice) in
      let set =
        List.fold_left
          (fun acc p -> Scalar_set.union acc (scalar_ancestors p))
          (Scalar_set.singleton s) parents
      in
      Hashtbl.add ancestor_cache s set;
      set

let min_by_priority scalars =
  Scalar_set.fold
    (fun s best ->
      match best with
      | None -> Some s
      | Some b when scalar_compare s b < 0 -> Some s
      | _ -> best)
    scalars None

(* Val module *)

module Val = struct
  type t = { scalar : scalar; count : int }

  let scalar dt = dt.scalar
  let count dt = dt.count
  let of_scalar s = { scalar = s; count = 1 }
  let void = of_scalar Void
  let bool = of_scalar Bool
  let int8 = of_scalar Int8
  let int16 = of_scalar Int16
  let int32 = of_scalar Int32
  let int64 = of_scalar Int64
  let uint8 = of_scalar Uint8
  let uint16 = of_scalar Uint16
  let uint32 = of_scalar Uint32
  let uint64 = of_scalar Uint64
  let float16 = of_scalar Float16
  let bfloat16 = of_scalar Bfloat16
  let float32 = of_scalar Float32
  let float64 = of_scalar Float64
  let fp8e4m3 = of_scalar Fp8e4m3
  let fp8e5m2 = of_scalar Fp8e5m2
  let fp8e4m3fnuz = of_scalar Fp8e4m3fnuz
  let fp8e5m2fnuz = of_scalar Fp8e5m2fnuz
  let uint128 = of_scalar Uint128
  let uint256 = of_scalar Uint256
  let weakint = of_scalar Weakint
  let default_float = of_scalar (default_float_scalar ())
  let default_int = int32

  let scalarize dt = if dt.count = 1 then dt else { dt with count = 1 }

  let vec n dt =
    if dt.count <> 1 then
      invalid_arg (strf "Val.vec: already vectorized (count = %d)" dt.count);
    if n < 0 then
      invalid_arg (strf "Val.vec: size must be non-negative (got %d)" n);
    if n = 1 || dt.scalar = Void then dt else { dt with count = n }

  let with_scalar s dt = { dt with scalar = s }

  let is_float dt = scalar_is_float dt.scalar
  let is_int dt = scalar_is_int dt.scalar
  let is_unsigned dt = scalar_is_unsigned dt.scalar
  let is_bool dt = scalar_is_bool dt.scalar
  let is_fp8 dt = scalar_is_fp8 dt.scalar
  let bitsize dt = scalar_bitsize dt.scalar * dt.count
  let itemsize dt = (bitsize dt + 7) / 8
  let priority dt = scalar_priority dt.scalar

  let least_upper_dtype dts =
    match dts with
    | [] -> invalid_arg "Val.least_upper_dtype: empty list"
    | [ d ] -> scalarize d
    | first :: rest ->
        let intersection =
          List.fold_left
            (fun acc d -> Scalar_set.inter acc (scalar_ancestors d.scalar))
            (scalar_ancestors first.scalar) rest
        in
        (match min_by_priority intersection with
        | Some s -> of_scalar s
        | None -> invalid_arg "Val.least_upper_dtype: no common ancestor")

  let least_upper_float dt =
    if scalar_is_float dt.scalar then scalarize dt
    else least_upper_dtype [ scalarize dt; default_float ]

  let can_lossless_cast dt0 dt1 =
    let s0 = dt0.scalar and s1 = dt1.scalar in
    s0 = s1 || s0 = Bool ||
    match s1 with
    | Weakint ->
        List.mem s0 [ Uint8; Uint16; Uint32; Uint64; Int8; Int16; Int32; Int64 ]
    | Float64 ->
        List.mem s0
          [ Float32; Float16; Bfloat16; Fp8e4m3; Fp8e5m2;
            Fp8e4m3fnuz; Fp8e5m2fnuz;
            Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
    | Float32 ->
        List.mem s0
          [ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz;
            Uint16; Uint8; Int16; Int8 ]
    | Float16 ->
        List.mem s0
          [ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz; Uint8; Int8 ]
    | Uint64 -> List.mem s0 [ Uint32; Uint16; Uint8 ]
    | Uint32 -> List.mem s0 [ Uint16; Uint8 ]
    | Uint16 -> s0 = Uint8
    | Int64 -> List.mem s0 [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
    | Int32 -> List.mem s0 [ Uint16; Uint8; Int16; Int8 ]
    | Int16 -> List.mem s0 [ Uint8; Int8 ]
    | _ -> false

  let sum_acc_dtype dt =
    let dt = scalarize dt in
    if scalar_is_unsigned dt.scalar then least_upper_dtype [ dt; uint32 ]
    else if scalar_is_int dt.scalar || scalar_is_bool dt.scalar then
      least_upper_dtype [ dt; int32 ]
    else least_upper_dtype [ dt; of_scalar (sum_dtype_scalar ()) ]

  let equal a b = a.scalar = b.scalar && a.count = b.count

  let compare a b =
    let c = scalar_compare a.scalar b.scalar in
    if c <> 0 then c else Int.compare a.count b.count

  let to_string t =
    let s = match t.scalar with
      | Void -> "void"   | Bool -> "bool"   | Weakint -> "weakint"
      | Int8 -> "i8"     | Int16 -> "i16"   | Int32 -> "i32"   | Int64 -> "i64"
      | Uint8 -> "u8"    | Uint16 -> "u16"  | Uint32 -> "u32"  | Uint64 -> "u64"
      | Uint128 -> "u128" | Uint256 -> "u256"
      | Float16 -> "f16" | Bfloat16 -> "bf16"
      | Float32 -> "f32" | Float64 -> "f64"
      | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"
      | Fp8e4m3fnuz -> "fp8e4m3fnuz" | Fp8e5m2fnuz -> "fp8e5m2fnuz"
    in
    if t.count = 1 then s else strf "%s×%d" s t.count

  let repr_name = function
    | Void -> "void" | Weakint -> "weakint" | Bool -> "bool"
    | Int8 -> "char" | Int16 -> "short" | Int32 -> "int" | Int64 -> "long"
    | Uint8 -> "uchar" | Uint16 -> "ushort" | Uint32 -> "uint"
    | Uint64 -> "ulong"
    | Uint128 -> invalid_arg (err_private_wide_repr "_uint128")
    | Uint256 -> invalid_arg (err_private_wide_repr "_uint256")
    | Float16 -> "half" | Bfloat16 -> "bfloat16"
    | Float32 -> "float" | Float64 -> "double"
    | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"
    | Fp8e4m3fnuz -> "fp8e4m3fnuz"
    | Fp8e5m2fnuz -> "fp8e5m2fnuz"

  let repr dt =
    let base = strf "dtypes.%s" (repr_name dt.scalar) in
    if dt.count = 1 then base else strf "%s.vec(%d)" base dt.count

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

(* Ptr module *)

module Ptr = struct
  type image = { kind : image_kind; shape : int list }

  type t = {
    scalar : scalar;
    count : int;
    addrspace : addr_space;
    v : int;
    size : int;
    image : image option;
  }

  let scalar p = p.scalar
  let count p = p.count
  let addrspace p = p.addrspace
  let v p = p.v
  let size p = p.size
  let base p : Val.t = { scalar = p.scalar; count = p.count }
  let value p =
    Val.vec (p.count * p.v) (Val.scalarize (base p))

  let image_kind p = Option.map (fun image -> image.kind) p.image
  let image_shape p = Option.map (fun image -> image.shape) p.image
  let is_image p = Option.is_some p.image

  let create (base : Val.t) ~addrspace ~size =
    { scalar = base.scalar; count = base.count; addrspace; v = 1; size;
      image = None }

  let create_v (base : Val.t) ~addrspace ~size ~v =
    if v < 1 then invalid_arg (err_ptr_vcount v);
    { scalar = base.scalar; count = base.count; addrspace; v; size;
      image = None }

  let scalarize p =
    if p.v = 1 && p.count = 1 then p
    else { p with count = 1; v = 1 }

  let vec n p =
    if n < 1 then invalid_arg (err_ptr_vcount n);
    if p.v <> 1 then invalid_arg (err_ptr_vcount n);
    { p with v = n }

  let with_base (dt : Val.t) p =
    { p with scalar = dt.scalar; count = dt.count; image = None }

  let with_size n p =
    if Option.is_some p.image && n <> p.size then
      { p with size = n; image = None }
    else { p with size = n }

  let image_bitsize = function Imageh -> 16 | Imagef -> 32
  let image_name = function Imageh -> "imageh" | Imagef -> "imagef"
  let image_priority = 100

  let create_image kind shape =
    let size = List.fold_left ( * ) 1 shape in
    { scalar = Float32; count = 1; addrspace = Global; v = 1; size;
      image = Some { kind; shape } }

  let bitsize p =
    match p.image with
    | None -> scalar_bitsize p.scalar * p.count
    | Some image -> image_bitsize image.kind * p.count

  let itemsize p = (bitsize p + 7) / 8

  let nbytes p =
    if p.size = -1 then invalid_arg err_ptr_unknown_size;
    p.size * itemsize p

  let priority p =
    match p.image with
    | None -> scalar_priority p.scalar
    | Some _ -> image_priority

  let equal a b =
    a.scalar = b.scalar && a.count = b.count
    && a.addrspace = b.addrspace && a.v = b.v && a.size = b.size
    && a.image = b.image

  let compare a b =
    let ( |? ) c f = if c <> 0 then c else f () in
    Int.compare (priority a) (priority b) |? fun () ->
    scalar_compare a.scalar b.scalar |? fun () ->
    Int.compare a.count b.count |? fun () ->
    Stdlib.compare a.addrspace b.addrspace |? fun () ->
    Int.compare a.v b.v |? fun () ->
    Int.compare a.size b.size |? fun () -> Stdlib.compare a.image b.image

  let to_string p =
    let vec = if p.v = 1 then "" else strf ".vec(%d)" p.v in
    match p.image with
    | Some image ->
        let shape = String.concat "," (List.map string_of_int image.shape) in
        strf "%s(%s)%s" (image_name image.kind) shape vec
    | None ->
        let base = Val.to_string { Val.scalar = p.scalar; count = p.count } in
        let space = match p.addrspace with
          | Global -> "global" | Local -> "local" | Reg -> "reg" | Alu -> "alu"
        in
        strf "%s*%s [%s]" base vec space

  let addr_space_repr = function
    | Global -> "AddrSpace.GLOBAL"
    | Local -> "AddrSpace.LOCAL"
    | Reg -> "AddrSpace.REG"
    | Alu -> "AddrSpace.ALU"

  let tuple_repr = function
    | [] -> "()"
    | [ x ] -> strf "(%d,)" x
    | xs -> strf "(%s)" (String.concat ", " (List.map string_of_int xs))

  let repr p =
    let vec = if p.v = 1 then "" else strf ".vec(%d)" p.v in
    match p.image with
    | Some image -> strf "dtypes.%s(%s)%s" (image_name image.kind)
        (tuple_repr image.shape) vec
    | None ->
        let base = Val.repr { Val.scalar = p.scalar; count = p.count } in
        let addrspace =
          if p.addrspace = Global then "" else strf ", %s" (addr_space_repr p.addrspace)
        in
        strf "%s.ptr(%d%s)%s" base p.size addrspace vec

  let pp fmt p = Format.pp_print_string fmt (to_string p)
end

(* Image module *)

module Image = struct
  type kind = image_kind = Imageh | Imagef
  type t = Ptr.t

  let create = Ptr.create_image
  let imageh shape = create Imageh shape
  let imagef shape = create Imagef shape

  let require_image p =
    match Ptr.image_kind p with
    | Some _ -> ()
    | None -> invalid_arg err_not_image

  let kind p =
    match Ptr.image_kind p with
    | Some kind -> kind
    | None -> invalid_arg err_not_image

  let shape p =
    match Ptr.image_shape p with
    | Some shape -> shape
    | None -> invalid_arg err_not_image

  let base p =
    require_image p;
    Ptr.base p

  let size p =
    require_image p;
    Ptr.size p

  let is_macos () =
    let ostype =
      Sys.getenv_opt "OSTYPE"
      |> Option.value ~default:""
      |> String.lowercase_ascii
    in
    String.starts_with ~prefix:"darwin" ostype
    || (String.equal Sys.os_type "Unix"
        && Sys.file_exists "/System/Library/CoreServices/SystemVersion.plist")

  let round_up n align = (n + align - 1) / align * align

  let pitch p =
    require_image p;
    match shape p with
    | _ :: width :: _ ->
        let width = if is_macos () then round_up width 256 else width in
        width * 4 * Ptr.itemsize p
    | _ -> invalid_arg "image pitch requires at least two dimensions"

  let addrspace p =
    require_image p;
    Ptr.addrspace p

  let vec n p =
    require_image p;
    Ptr.vec n p
end

(* Unified type *)

type t = Val of Val.t | Ptr of Ptr.t

(* Dispatching accessors *)

let scalar = function Val v -> Val.scalar v | Ptr p -> Ptr.scalar p
let count = function Val v -> Val.count v | Ptr p -> Ptr.count p
let vcount = function Val v -> Val.count v | Ptr p -> Ptr.v p
let is_ptr = function Ptr _ -> true | Val _ -> false
let val_of = function Val v -> v | Ptr p -> Ptr.base p

(* Dispatching transformers *)

let scalarize = function
  | Val v -> Val (Val.scalarize v)
  | Ptr p -> Ptr (Ptr.scalarize p)

let vec n = function
  | Val v -> Val (Val.vec n v)
  | Ptr p -> Ptr (Ptr.vec n p)

(* Predicates *)

let is_float = function Val v -> Val.is_float v | Ptr p -> Ptr.is_image p
let is_int = function Val v -> Val.is_int v | Ptr _ -> false
let is_unsigned = function Val v -> Val.is_unsigned v | Ptr _ -> false
let is_bool = function Val v -> Val.is_bool v | Ptr _ -> false
let is_fp8 = function Val v -> Val.is_fp8 v | Ptr _ -> false

(* Promotion *)

let least_upper_dtype dts =
  let rec scan first_image vals = function
    | [] -> (
        match first_image with
        | Some image -> Ptr image
        | None -> Val (Val.least_upper_dtype (List.rev vals)))
    | Val v :: rest -> scan first_image (v :: vals) rest
    | Ptr p :: rest ->
        if Ptr.is_image p then
          let first_image = match first_image with
            | Some _ as image -> image
            | None -> Some p
          in
          scan first_image vals rest
        else invalid_arg "Dtype.least_upper_dtype: pointer dtype"
  in
  scan None [] dts

let least_upper_float = function
  | Ptr p when Ptr.is_image p -> Ptr p
  | Ptr _ -> invalid_arg "Dtype.least_upper_float: pointer dtype"
  | Val v when Val.is_float v -> Val v
  | Val v -> Val (Val.least_upper_dtype [ Val.scalarize v; Val.default_float ])

(* Properties *)

let bitsize = function
  | Val v -> Val.bitsize v
  | Ptr p -> Ptr.bitsize p

let itemsize dt = (bitsize dt + 7) / 8

let priority = function
  | Val v -> Val.priority v
  | Ptr p -> Ptr.priority p

(* Bounds *)

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]

let min_scalar s =
  let b = scalar_bitsize s in
  match s with
  | Bool -> `Bool false
  | Uint8 | Uint16 | Uint32 | Uint64 -> `UInt 0L
  | Uint128 | Uint256 -> `Bool false
  | Weakint -> `SInt Int64.min_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b >= 64 then `SInt Int64.min_int
      else `SInt Int64.(neg (shift_left 1L (b - 1)))
  | Float16 | Bfloat16 | Float32 | Float64
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Float neg_infinity
  | Void -> invalid_arg err_void_bounds

let max_scalar s =
  let b = scalar_bitsize s in
  match s with
  | Bool -> `Bool true
  | Uint8 | Uint16 | Uint32 ->
      `UInt Int64.(sub (shift_left 1L b) 1L)
  | Uint64 ->
      `UInt Int64.minus_one
  | Uint128 | Uint256 -> `Bool true
  | Weakint -> `SInt Int64.max_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b >= 64 then `SInt Int64.max_int
      else `SInt Int64.(sub (shift_left 1L (b - 1)) 1L)
  | Float16 | Bfloat16 | Float32 | Float64
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Float infinity
  | Void -> invalid_arg err_void_bounds

let min = function
  | Val v -> min_scalar (Val.scalar v)
  | Ptr p -> if Ptr.is_image p then `Float neg_infinity else `Bool false

let max = function
  | Val v -> max_scalar (Val.scalar v)
  | Ptr p -> if Ptr.is_image p then `Float infinity else `Bool true

let finfo = function
  | Ptr _ -> invalid_arg "finfo: not a floating-point dtype"
  | Val dt -> (
      match Val.scalar dt with
      | Float16 -> 5, 10   | Bfloat16 -> 8, 7
      | Float32 -> 8, 23   | Float64 -> 11, 52
      | Fp8e5m2 | Fp8e5m2fnuz -> 5, 2
      | Fp8e4m3 | Fp8e4m3fnuz -> 4, 3
      | _ -> invalid_arg "finfo: not a floating-point dtype")

(* Comparison *)

let equal a b =
  match a, b with
  | Val a, Val b -> Val.equal a b
  | Ptr a, Ptr b -> Ptr.equal a b
  | _ -> false

let compare a b =
  match a, b with
  | Val a, Val b -> Val.compare a b
  | Ptr a, Ptr b -> Ptr.compare a b
  | Val _, Ptr _ -> -1
  | Ptr _, Val _ -> 1

(* Formatting *)

let to_string = function Val v -> Val.to_string v | Ptr p -> Ptr.to_string p
let repr = function Val v -> Val.repr v | Ptr p -> Ptr.repr p
let pp fmt dt = Format.pp_print_string fmt (to_string dt)

(* Scalar formatting *)

let scalar_to_string = function
  | Void -> "void"   | Bool -> "bool"   | Weakint -> "weakint"
  | Int8 -> "i8"     | Int16 -> "i16"   | Int32 -> "i32"   | Int64 -> "i64"
  | Uint8 -> "u8"    | Uint16 -> "u16"  | Uint32 -> "u32"  | Uint64 -> "u64"
  | Uint128 -> "u128" | Uint256 -> "u256"
  | Float16 -> "f16" | Bfloat16 -> "bf16"
  | Float32 -> "f32" | Float64 -> "f64"
  | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"
  | Fp8e4m3fnuz -> "fp8e4m3fnuz" | Fp8e5m2fnuz -> "fp8e5m2fnuz"

let pp_scalar fmt s = Format.pp_print_string fmt (scalar_to_string s)

let addr_space_to_string = function
  | Global -> "global" | Local -> "local" | Reg -> "reg" | Alu -> "alu"

let pp_addr_space fmt a = Format.pp_print_string fmt (addr_space_to_string a)

(* Convenience constructors — wrapped as Dtype.t *)

let of_scalar s = Val (Val.of_scalar s)
let void = Val Val.void
let bool = Val Val.bool
let int8 = Val Val.int8
let int16 = Val Val.int16
let int32 = Val Val.int32
let int64 = Val Val.int64
let uint8 = Val Val.uint8
let uint16 = Val Val.uint16
let uint32 = Val Val.uint32
let uint64 = Val Val.uint64
let float16 = Val Val.float16
let bfloat16 = Val Val.bfloat16
let float32 = Val Val.float32
let float64 = Val Val.float64
let fp8e4m3 = Val Val.fp8e4m3
let fp8e5m2 = Val Val.fp8e5m2
let fp8e4m3fnuz = Val Val.fp8e4m3fnuz
let fp8e5m2fnuz = Val Val.fp8e5m2fnuz
let weakint = Val Val.weakint
let default_float = Val Val.default_float
let default_int = Val Val.default_int
let imageh shape = Ptr (Image.imageh shape)
let imagef shape = Ptr (Image.imagef shape)

(* FP conversion.

   These routines round binary64 values to narrower IEEE-754 encodings
   (fp16, bfloat16, fp8e4m3, fp8e5m2) using round-to-nearest-even. They
   walk the exponent and mantissa bit-by-bit rather than delegating to a
   hardware conversion because (a) not every host supports every format
   and (b) constant folding must be bit-identical across backends. *)

let float_to_fp16 x =
  if Float.is_nan x then Float.nan
  else if Float.is_infinite x then x
  else if x = 0.0 then x
  else
    let bits = Int64.bits_of_float x in
    let sign = Int64.logand (Int64.shift_right_logical bits 63) 1L in
    let exp =
      Int64.to_int (Int64.logand (Int64.shift_right_logical bits 52) 0x7FFL)
    in
    let mant = Int64.logand bits 0xFFFFFFFFFFFFFL in
    let unbiased = exp - 1023 in
    if unbiased > 15 then
      if sign = 1L then Float.neg Float.infinity else Float.infinity
    else if unbiased < -24 then if sign = 1L then -0.0 else 0.0
    else
      let fp16_sign = Int64.shift_left sign 15 in
      let fp16_bits =
        if unbiased < -14 then begin
          let shift = -14 - unbiased in
          let full_mant = Int64.logor mant 0x10000000000000L in
          let total_shift = 42 + shift in
          let shifted = Int64.shift_right_logical full_mant total_shift in
          let round_bit =
            Int64.to_int
              (Int64.logand
                 (Int64.shift_right_logical full_mant (total_shift - 1))
                 1L)
          in
          let sticky =
            let mask = Int64.sub (Int64.shift_left 1L (total_shift - 1)) 1L in
            if Int64.logand full_mant mask <> 0L then 1 else 0
          in
          let rounded =
            if round_bit = 1 && (sticky = 1 || Int64.logand shifted 1L <> 0L)
            then Int64.add shifted 1L
            else shifted
          in
          Int64.logor fp16_sign rounded
        end
        else begin
          let biased16 = unbiased + 15 in
          let shifted_mant = Int64.shift_right_logical mant 42 in
          let round_bit =
            Int64.to_int (Int64.logand (Int64.shift_right_logical mant 41) 1L)
          in
          let sticky =
            if Int64.logand mant 0x1FFFFFFFFFFL <> 0L then 1 else 0
          in
          let rounded =
            if
              round_bit = 1 && (sticky = 1 || Int64.logand shifted_mant 1L <> 0L)
            then Int64.add shifted_mant 1L
            else shifted_mant
          in
          let final_exp, final_mant =
            if rounded > 0x3FFL then (biased16 + 1, 0L) else (biased16, rounded)
          in
          if final_exp > 30 then Int64.logor fp16_sign 0x7C00L
          else
            Int64.logor fp16_sign
              (Int64.logor (Int64.of_int (final_exp lsl 10)) final_mant)
        end
      in
      let fp16_exp =
        Int64.to_int
          (Int64.logand (Int64.shift_right_logical fp16_bits 10) 0x1FL)
      in
      let fp16_mant = Int64.logand fp16_bits 0x3FFL in
      let f =
        if fp16_exp = 0x1F then
          if fp16_mant = 0L then Float.infinity else Float.nan
        else if fp16_exp = 0 then Float.ldexp (Int64.to_float fp16_mant) (-24)
        else
          Float.ldexp
            (Int64.to_float (Int64.logor fp16_mant 0x400L))
            (fp16_exp - 25)
      in
      if sign = 1L then Float.neg f else f

let float_to_bf16 x =
  if not (Float.is_finite x) then x
  else
    let u = Int32.bits_of_float x in
    let u =
      Int32.logand
        (Int32.add u
           (Int32.add 0x7FFFl
              (Int32.logand (Int32.shift_right_logical u 16) 1l)))
        0xFFFF_0000l
    in
    Int32.float_of_bits u

type fp8_params = {
  exp_bias : int; sig_bits : int; mantissa_mask : int;
  mindenorm_o2 : int64; overflow_threshold : int64;
  maxnorm : int; minnorm : int64;
}

let fp8e4m3_params =
  { exp_bias = 7; sig_bits = 4; mantissa_mask = 0x7;
    mindenorm_o2 = 0x3F50000000000000L; overflow_threshold = 0x407D000000000000L;
    maxnorm = 0x7E; minnorm = 0x3F90000000000000L }

let fp8e5m2_params =
  { exp_bias = 15; sig_bits = 3; mantissa_mask = 0x3;
    mindenorm_o2 = 0x3EE0000000000000L;
    overflow_threshold = Int64.sub 0x40EE000000000000L 1L;
    maxnorm = 0x7B; minnorm = 0x3F10000000000000L }

let fp8e4m3fnuz_params =
  { exp_bias = 8; sig_bits = 4; mantissa_mask = 0x7;
    mindenorm_o2 = 0x3F40000000000000L;
    overflow_threshold = Int64.sub 0x406F000000000000L 1L;
    maxnorm = 0x7F; minnorm = 0x3F80000000000000L }

let fp8e5m2fnuz_params =
  { exp_bias = 16; sig_bits = 3; mantissa_mask = 0x3;
    mindenorm_o2 = 0x3ED0000000000000L;
    overflow_threshold = Int64.sub 0x40EE000000000000L 1L;
    maxnorm = 0x7F; minnorm = 0x3F00000000000000L }

(* Pack a binary64 into an fp8 byte. Signs, subnormals, overflow, and
   round-to-nearest-even are handled case-by-case. *)
let float_to_fp8 scalar x =
  match scalar with
  | (Fp8e4m3fnuz | Fp8e5m2fnuz) when not (Float.is_finite x) -> 0x80
  | (Fp8e4m3fnuz | Fp8e5m2fnuz) when x = 0.0 -> 0x00
  | Fp8e4m3 when not (Float.is_finite x) ->
      if Float.copy_sign 1.0 x > 0.0 then 0x7f else 0xff
  | Fp8e5m2 when not (Float.is_finite x) ->
      let sign = if Float.copy_sign 1.0 x > 0.0 then 0 else 0x80 in
      sign lor if Float.is_infinite x then 0x7c else 0x7f
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      let p = match scalar with
        | Fp8e4m3 -> fp8e4m3_params
        | Fp8e5m2 -> fp8e5m2_params
        | Fp8e4m3fnuz -> fp8e4m3fnuz_params
        | _ -> fp8e5m2fnuz_params
      in
      let xbits = Int64.bits_of_float x in
      let half_ulp = Int64.shift_left 1L (53 - p.sig_bits - 1) in
      let sign =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 63) 1L)
        lsl 7
      in
      let raw_exp =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 52) 0x7FFL)
      in
      let exp = raw_exp - 1023 + p.exp_bias in
      let mantissa =
        Int64.to_int
          (Int64.logand
             (Int64.shift_right_logical xbits (53 - p.sig_bits))
             (Int64.of_int p.mantissa_mask))
      in
      let absx = Int64.logand xbits 0x7FFFFFFFFFFFFFFFL in
      let res =
        if Int64.compare absx p.mindenorm_o2 <= 0 then 0
        else if Int64.compare absx 0x7FF0000000000000L > 0 then
          if scalar = Fp8e4m3 then 0x7F else 0x7E lor mantissa
        else if Int64.compare absx p.overflow_threshold > 0 then p.maxnorm
        else if Int64.compare absx p.minnorm >= 0 then begin
          let base = (exp lsl (p.sig_bits - 1)) lor mantissa in
          let round_mask = Int64.sub (Int64.shift_left half_ulp 1) 1L in
          let round_bits = Int64.logand xbits round_mask in
          if Int64.compare round_bits half_ulp > 0
             || (round_bits = half_ulp && mantissa land 1 <> 0)
          then base + 1 else base
        end
        else begin
          let shift = 1 - exp in
          let mant_with_implicit = mantissa lor (1 lsl (p.sig_bits - 1)) in
          let base = mant_with_implicit asr shift in
          let round_bits =
            Int64.logand
              (Int64.logor xbits (Int64.shift_left 1L 52))
              (Int64.sub (Int64.shift_left half_ulp (shift + 1)) 1L)
          in
          let threshold = Int64.shift_left half_ulp shift in
          if Int64.compare round_bits threshold > 0
             || (round_bits = threshold && base land 1 <> 0)
          then base + 1 else base
        end
      in
      (* fnuz types have no negative zero. *)
      if scalar_is_fp8_fnuz scalar && res = 0 then 0 else res lor sign
  | _ -> invalid_arg "float_to_fp8: not an fp8 dtype"

let fp8_to_float scalar x =
  match scalar with
  | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      if x = 0x80 then Float.nan
      else if x land 0x7F = 0 then 0.0
      else
        let p = match scalar with
          | Fp8e4m3fnuz -> fp8e4m3fnuz_params | _ -> fp8e5m2fnuz_params
        in
        let mant_bits = p.sig_bits - 1 in
        let exp_bits = 8 - p.sig_bits in
        let exp_max = (1 lsl exp_bits) - 1 in
        let mant_max = (1 lsl mant_bits) - 1 in
        let sign = (x lsr 7) land 1 in
        let exp = (x lsr mant_bits) land exp_max in
        let mantissa = x land mant_max in
        let frac = Float.of_int mantissa /. Float.of_int (mant_max + 1) in
        let v =
          if exp = 0 then frac *. (2. ** Float.of_int (1 - p.exp_bias))
          else (1. +. frac) *. (2. ** Float.of_int (exp - p.exp_bias))
        in
        if sign = 1 then Float.neg v else v
  | Fp8e4m3 | Fp8e5m2 ->
      let ur = x lsl 8 in
      let ur =
        if scalar = Fp8e5m2 && ur land 0x7FFF > 0x7C00 then 0x7FFF
        else if scalar = Fp8e4m3 then begin
          let sign = ur land 0x8000 in
          let exponent = ((ur land 0x7800) asr 1) + 0x2000 in
          let mantissa_init = (ur land 0x0700) asr 1 in
          let absx = x land 0x7F in
          if absx = 0x7F then 0x7FFF
          else if exponent = 0x2000 then begin
            if mantissa_init <> 0 then begin
              let rec normalize m e =
                if m land 0x0400 <> 0 then (m, e)
                else normalize (m lsl 1) (e - 0x0400)
              in
              let m, e = normalize (mantissa_init lsl 1) exponent in
              sign lor e lor (m land 0x03FF)
            end
            else sign
          end
          else sign lor exponent lor mantissa_init
        end
        else ur
      in
      let fp16_sign = (ur asr 15) land 1 in
      let fp16_exp = (ur asr 10) land 0x1F in
      let fp16_mant = ur land 0x3FF in
      let f =
        if fp16_exp = 0x1F then
          if fp16_mant = 0 then Float.infinity else Float.nan
        else if fp16_exp = 0 then Float.ldexp (Float.of_int fp16_mant) (-24)
        else Float.ldexp (Float.of_int (fp16_mant + 1024)) (fp16_exp - 25)
      in
      if fp16_sign = 1 then Float.neg f else f
  | _ -> invalid_arg "fp8_to_float: not an fp8 dtype"

let truncate_float (dt : Val.t) x =
  match dt.scalar with
  | Float64 -> x
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float x)
  | Float16 -> float_to_fp16 x
  | Bfloat16 -> float_to_bf16 x
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      fp8_to_float dt.scalar (float_to_fp8 dt.scalar x)
  | _ -> invalid_arg "truncate_float: not a floating-point dtype"

let truncate_int (dt : Val.t) x =
  let b = scalar_bitsize dt.scalar in
  match dt.scalar with
  | Bool -> if x <> 0 then 1 else 0
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= Sys.int_size then x else x land ((1 lsl b) - 1)
  | Uint128 | Uint256 ->
      (* Virtual wider-than-native types: no OCaml int truncation. *)
      x
  | Int8 | Int16 | Int32 | Int64 | Weakint ->
      if b >= Sys.int_size then x
      else
        let mask = (1 lsl b) - 1 in
        let unsigned = x land mask in
        if unsigned land (1 lsl (b - 1)) <> 0 then unsigned lor lnot mask
        else unsigned
  | _ -> invalid_arg "truncate_int: not an integer or bool dtype"

type storage_scalar = [ `Bool of bool | `Float of float | `Int of int64 ]

let storage_fmt_for_dtype (dt : Val.t) =
  match dt.scalar with
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
  | Void | Weakint | Uint128 | Uint256 -> None

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

let truncate_int64 scalar x =
  let b = scalar_bitsize scalar in
  match scalar with
  | Bool -> if x <> 0L then 1L else 0L
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= 64 then x else Int64.logand x Int64.(sub (shift_left 1L b) 1L)
  | Uint128 | Uint256 -> x
  | Int8 | Int16 | Int32 | Int64 | Weakint ->
      if b >= 64 then x
      else
        let mask = Int64.(sub (shift_left 1L b) 1L) in
        let unsigned = Int64.logand x mask in
        if Int64.logand unsigned (Int64.shift_left 1L (b - 1)) <> 0L then
          Int64.logor unsigned (Int64.lognot mask)
        else unsigned
  | _ -> invalid_arg "truncate: not an integer or bool dtype"

let to_storage_scalar (dt : Val.t) x =
  match dt.scalar with
  | Bool -> `Bool (storage_bool x)
  | Float16 -> `Float (float_to_fp16 (storage_float x))
  | Bfloat16 ->
      let bits = Int32.bits_of_float (float_to_bf16 (storage_float x)) in
      `Int
        (Int64.of_int
           (Int32.to_int
              (Int32.logand (Int32.shift_right_logical bits 16) 0xFFFFl)))
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Int (Int64.of_int (float_to_fp8 dt.scalar (storage_float x)))
  | Float32 | Float64 -> `Float (storage_float x)
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 | Uint128
  | Uint256 | Weakint -> `Int (storage_int64 x)
  | Void -> invalid_arg "to_storage_scalar: void has no storage scalar"

let from_storage_scalar x (dt : Val.t) =
  match dt.scalar with
  | Bool -> `Bool (storage_bool x)
  | Bfloat16 ->
      let lo = Int64.to_int (Int64.logand (storage_int64 x) 0xFFFFL) in
      `Float (Int32.float_of_bits (Int32.shift_left (Int32.of_int lo) 16))
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Float (fp8_to_float dt.scalar (Int64.to_int (storage_int64 x)))
  | Float16 | Float32 | Float64 -> `Float (storage_float x)
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 | Uint128
  | Uint256 | Weakint -> `Int (storage_int64 x)
  | Void -> invalid_arg "from_storage_scalar: void has no storage scalar"

let truncate (dt : Val.t) x =
  match dt.scalar with
  | Bool -> `Bool (storage_bool x)
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz
  | Fp8e5m2fnuz -> `Float (truncate_float dt (storage_float x))
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 | Uint128
  | Uint256 | Weakint -> `Int (truncate_int64 dt.scalar (storage_int64 x))
  | Void -> invalid_arg "truncate: void has no storage scalar"
