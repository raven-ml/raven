open Windtrap
open Tolk_uop

let dtype = testable ~pp:Dtype.Val.pp ~equal:Dtype.Val.equal ()
let any_dtype = testable ~pp:Dtype.pp ~equal:Dtype.equal ()
let ptr_dtype = testable ~pp:Dtype.Ptr.pp ~equal:Dtype.Ptr.equal ()

let bound =
  let pp fmt = function
    | `Bool b -> Format.fprintf fmt "`Bool %b" b
    | `SInt n -> Format.fprintf fmt "`SInt %Ld" n
    | `UInt n -> Format.fprintf fmt "`UInt %Ld" n
    | `Float f -> Format.fprintf fmt "`Float %h" f
  in
  let equal a b =
    match a, b with
    | `Bool a, `Bool b -> Bool.equal a b
    | `SInt a, `SInt b -> Int64.equal a b
    | `UInt a, `UInt b -> Int64.equal a b
    | `Float a, `Float b ->
        Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
    | _ -> false
  in
  testable ~pp ~equal ()

let storage_scalar =
  let pp fmt = function
    | `Bool b -> Format.fprintf fmt "`Bool %b" b
    | `Int n -> Format.fprintf fmt "`Int %Ld" n
    | `Float f -> Format.fprintf fmt "`Float %h" f
  in
  let equal a b =
    match a, b with
    | `Bool a, `Bool b -> Bool.equal a b
    | `Int a, `Int b -> Int64.equal a b
    | `Float a, `Float b ->
        Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
    | _ -> false
  in
  testable ~pp ~equal ()

let char_option =
  let pp fmt = function
    | None -> Format.pp_print_string fmt "None"
    | Some c -> Format.fprintf fmt "Some %C" c
  in
  testable ~pp ~equal:( = ) ()

let float_bits =
  let pp fmt f = Format.fprintf fmt "%h" f in
  let equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  testable ~pp ~equal ()

let const = testable ~pp:Const.pp ~equal:Const.equal ()

let raises_invalid f =
  try
    ignore (f ());
    false
  with Invalid_argument _ -> true

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

let exit_bool ok = exit (if ok then 0 else 1)

let run_env_case = function
  | "default_float_half" ->
      exit_bool (Dtype.Val.equal Dtype.Val.float16 Dtype.Val.default_float)
  | "default_float_float" ->
      exit_bool (Dtype.Val.equal Dtype.Val.float32 Dtype.Val.default_float)
  | "sum_dtype_double" ->
      exit_bool
        (Dtype.Val.equal Dtype.Val.float64
           (Dtype.Val.sum_acc_dtype Dtype.Val.float16))
  | "sum_dtype_bfloat16" ->
      exit_bool
        (Dtype.Val.equal Dtype.Val.float32
           (Dtype.Val.sum_acc_dtype Dtype.Val.float16))
  | "sum_dtype_uchar" ->
      exit_bool
        (Dtype.Val.equal Dtype.Val.float16
           (Dtype.Val.sum_acc_dtype Dtype.Val.float16))
  | "sum_dtype_default_float" ->
      exit_bool
        (Dtype.Val.equal Dtype.Val.float16
           (Dtype.Val.sum_acc_dtype Dtype.Val.float16))
  | "sum_dtype_rejected" ->
      exit_bool
        (raises_invalid (fun () ->
             Dtype.Val.sum_acc_dtype Dtype.Val.float16))
  | _ -> exit 2

let env_assignment name value = name ^ "=" ^ Filename.quote value

let run_with_env ?default_float ?sum_dtype case =
  let assignments =
    [ Some (env_assignment "TOLK_DTYPE_ENV_CASE" case);
      Option.map (env_assignment "DEFAULT_FLOAT") default_float;
      Option.map (env_assignment "SUM_DTYPE") sum_dtype ]
    |> List.filter_map Fun.id
  in
  let command =
    String.concat " "
      (assignments
      @ [ Filename.quote Sys.executable_name; ">/dev/null"; "2>&1" ])
  in
  Sys.command command

let expect_env_success ?default_float ?sum_dtype case =
  equal int 0 (run_with_env ?default_float ?sum_dtype case)

let expect_env_failure ?default_float ?sum_dtype case =
  is_true (run_with_env ?default_float ?sum_dtype case <> 0)

let storage_formats () =
  equal char_option (Some '?') (Dtype.storage_fmt_for_dtype Dtype.Val.bool);
  equal char_option (Some 'i') (Dtype.storage_fmt_for_dtype Dtype.Val.int32);
  equal char_option (Some 'Q') (Dtype.storage_fmt_for_dtype Dtype.Val.uint64);
  equal char_option (Some 'H') (Dtype.storage_fmt_for_dtype Dtype.Val.bfloat16);
  equal char_option (Some 'B') (Dtype.storage_fmt_for_dtype Dtype.Val.fp8e4m3);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.Val.weakint);
  equal char_option None
    (Dtype.storage_fmt_for_dtype (Dtype.Val.of_scalar Dtype.Uint128))

let truncation_surface () =
  equal storage_scalar (`Bool false) (Dtype.truncate Dtype.Val.bool (`Int 0L));
  equal storage_scalar (`Bool true) (Dtype.truncate Dtype.Val.bool (`Float Float.nan));
  equal storage_scalar (`Int 0L) (Dtype.truncate Dtype.Val.uint8 (`Int 256L));
  equal storage_scalar (`Int 255L) (Dtype.truncate Dtype.Val.uint8 (`Int (-1L)));
  equal storage_scalar (`Int (-128L)) (Dtype.truncate Dtype.Val.int8 (`Int 128L));
  equal storage_scalar (`Float 448.0)
    (Dtype.truncate Dtype.Val.fp8e4m3 (`Float 500.0));
  equal storage_scalar (`Float 1232.0)
    (Dtype.truncate Dtype.Val.bfloat16 (`Float 1234.0))

let storage_roundtrips () =
  let bf16_storage =
    Dtype.to_storage_scalar Dtype.Val.bfloat16 (`Float 1234.0)
  in
  equal storage_scalar (`Int 17562L) bf16_storage;
  equal storage_scalar (`Float 1232.0)
    (Dtype.from_storage_scalar bf16_storage Dtype.Val.bfloat16);
  let fp8_storage = Dtype.to_storage_scalar Dtype.Val.fp8e4m3 (`Float 448.0) in
  equal storage_scalar (`Int 126L) fp8_storage;
  equal storage_scalar (`Float 448.0)
    (Dtype.from_storage_scalar fp8_storage Dtype.Val.fp8e4m3);
  equal storage_scalar (`Int 127L)
    (Dtype.to_storage_scalar Dtype.Val.fp8e5m2 (`Float Float.nan));
  let neg_zero_storage =
    Dtype.to_storage_scalar Dtype.Val.fp8e4m3 (`Float (-0.0))
  in
  equal storage_scalar (`Int 128L) neg_zero_storage;
  equal float_bits (-0.0)
    (match Dtype.from_storage_scalar neg_zero_storage Dtype.Val.fp8e4m3 with
    | `Float f -> f
    | _ -> assert false)

let defaults () =
  is_true (Dtype.Val.is_float Dtype.Val.default_float);
  equal dtype
    (Dtype.Val.least_upper_dtype [ Dtype.Val.int8; Dtype.Val.default_float ])
    (Dtype.Val.least_upper_float Dtype.Val.int8);
  equal dtype Dtype.Val.float64 (Dtype.Val.sum_acc_dtype Dtype.Val.float64)

let env_dtype_parsing () =
  expect_env_success ~default_float:"half" "default_float_half";
  expect_env_success ~default_float:"float" "default_float_float";
  expect_env_success ~default_float:"default_float" "default_float_float";
  expect_env_failure ~default_float:"f32" "default_float_float";
  expect_env_success ~default_float:"float" ~sum_dtype:"double"
    "sum_dtype_double";
  expect_env_success ~default_float:"float" ~sum_dtype:"bfloat16"
    "sum_dtype_bfloat16";
  expect_env_success ~default_float:"float" ~sum_dtype:"uchar"
    "sum_dtype_uchar";
  expect_env_success ~default_float:"half" ~sum_dtype:"default_float"
    "sum_dtype_default_float";
  List.iter
    (fun alias ->
      expect_env_success ~default_float:"float" ~sum_dtype:alias
        "sum_dtype_rejected")
    [ "i8"; "u8"; "f16"; "f32"; "bf16"; " half "; "uint128"; "u128";
      "uint256"; "u256" ]

let address_space () =
  equal string "alu" (Dtype.addr_space_to_string Dtype.Alu);
  let ptr = Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Alu ~size:(-1) in
  equal string "i32* [alu]" (Dtype.Ptr.to_string ptr)

let pointer_predicates () =
  let float_ptr =
    Dtype.Ptr (Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:4)
  in
  let int_ptr =
    Dtype.Ptr (Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Global ~size:4)
  in
  let bool_ptr =
    Dtype.Ptr (Dtype.Ptr.create Dtype.Val.bool ~addrspace:Dtype.Global ~size:4)
  in
  is_false (Dtype.is_float float_ptr);
  is_false (Dtype.is_int int_ptr);
  is_false (Dtype.is_unsigned int_ptr);
  is_false (Dtype.is_bool bool_ptr);
  is_false (Dtype.is_fp8 float_ptr)

let private_wide_helpers () =
  let uint128 = Dtype.Val.of_scalar Dtype.Uint128 in
  let uint256 = Dtype.Val.of_scalar Dtype.Uint256 in
  is_false (Dtype.Val.is_int uint128);
  is_false (Dtype.Val.is_unsigned uint128);
  is_false (Dtype.is_int (Dtype.Val uint256));
  is_false (Dtype.is_unsigned (Dtype.Val uint256));
  equal bound (`Bool false) (Dtype.min (Dtype.Val uint128));
  equal bound (`Bool true) (Dtype.max (Dtype.Val uint256));
  equal char_option None (Dtype.storage_fmt_for_dtype uint128);
  is_true
    ~msg:"private wide helper dtypes have no tinygrad public repr"
    (raises_invalid (fun () -> Dtype.Val.repr uint128));
  is_true
    ~msg:"private wide helper dtype pointers have no tinygrad public repr"
    (raises_invalid (fun () ->
         Dtype.Ptr.repr
           (Dtype.Ptr.create uint256 ~addrspace:Dtype.Global ~size:1)))

let vector_edge_surface () =
  let int0 = Dtype.Val.vec 0 Dtype.Val.int32 in
  equal int 0 (Dtype.Val.count int0);
  equal int 0 (Dtype.Val.bitsize int0);
  equal int 0 (Dtype.Val.itemsize int0);
  equal string "i32×0" (Dtype.Val.to_string int0);
  equal string "dtypes.int.vec(0)" (Dtype.Val.repr int0);
  equal dtype Dtype.Val.int32 (Dtype.Val.scalarize int0);
  equal any_dtype Dtype.void (Dtype.vec 0 Dtype.void);
  equal dtype Dtype.Val.void (Dtype.Val.vec 0 Dtype.Val.void);
  is_true ~msg:"negative vector widths still reject"
    (raises_invalid (fun () -> Dtype.Val.vec (-1) Dtype.Val.int32));
  is_true ~msg:"already-vectorized zero-lane dtype still rejects"
    (raises_invalid (fun () -> Dtype.Val.vec 2 int0))

let dtype_identity_surface () =
  equal dtype Dtype.Val.int32 (Dtype.Val.of_scalar Dtype.Int32);
  equal int 0 (Dtype.Val.compare Dtype.Val.int32 (Dtype.Val.of_scalar Dtype.Int32));
  let ptr_a =
    Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:8
  in
  let ptr_b =
    Dtype.Ptr.create (Dtype.Val.of_scalar Dtype.Float32)
      ~addrspace:Dtype.Global ~size:8
  in
  equal ptr_dtype ptr_a ptr_b;
  equal int 0 (Dtype.Ptr.compare ptr_a ptr_b);
  equal any_dtype (Dtype.imageh [ 2; 3; 4 ])
    (Dtype.Ptr (Dtype.Image.create Dtype.Imageh [ 2; 3; 4 ]))

let tinygrad_repr_surface () =
  equal string "dtypes.int" (Dtype.Val.repr Dtype.Val.int32);
  equal string "dtypes.float.vec(4)"
    (Dtype.Val.repr (Dtype.Val.vec 4 Dtype.Val.float32));
  let global_ptr =
    Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:(-1)
  in
  let local_ptr =
    Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Local ~size:4
    |> Dtype.Ptr.vec 2
  in
  equal string "dtypes.float.ptr(-1)" (Dtype.Ptr.repr global_ptr);
  equal string "dtypes.int.ptr(4, AddrSpace.LOCAL).vec(2)"
    (Dtype.Ptr.repr local_ptr);
  equal string "dtypes.imageh((8, 16, 4))"
    (Dtype.repr (Dtype.imageh [ 8; 16; 4 ]))

let image_dtype_surface () =
  let imageh = Dtype.Image.imageh [ 3; 5; 4 ] in
  let imagef = Dtype.Image.imagef [ 2; 7; 4 ] in
  equal dtype Dtype.Val.float32 (Dtype.Image.base imageh);
  equal int 60 (Dtype.Image.size imageh);
  equal int 56 (Dtype.Image.size imagef);
  equal int 16 (Dtype.Ptr.bitsize imageh);
  equal int 32 (Dtype.Ptr.bitsize imagef);
  equal int 2 (Dtype.Ptr.itemsize imageh);
  equal int 4 (Dtype.Ptr.itemsize imagef);
  equal int 120 (Dtype.Ptr.nbytes imageh);
  equal int 2 (Dtype.itemsize (Dtype.Ptr imageh));
  equal int 100 (Dtype.Ptr.priority imageh);
  equal int 100 (Dtype.priority (Dtype.Ptr imagef));
  equal int 60 (Dtype.Ptr.size imageh);
  equal string "imageh(3,5,4)" (Dtype.Ptr.to_string imageh);
  is_true (Dtype.Ptr.is_image imageh);
  is_true (Dtype.is_ptr (Dtype.Ptr imageh));
  is_true (Dtype.is_float (Dtype.Ptr imageh));
  is_false (Dtype.is_int (Dtype.Ptr imageh));
  is_false (Dtype.is_unsigned (Dtype.Ptr imageh));
  is_false (Dtype.is_bool (Dtype.Ptr imageh));
  equal any_dtype (Dtype.Ptr imagef) (Dtype.imagef [ 2; 7; 4 ]);
  equal ptr_dtype imageh
    (Dtype.Image.create Dtype.Imageh [ 3; 5; 4 ]);
  let pitch_width = if is_macos () then round_up 5 256 else 5 in
  equal int (pitch_width * 4 * Dtype.Ptr.itemsize imageh)
    (Dtype.Image.pitch imageh)

let pointer_nbytes () =
  let ptr = Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:4 in
  let unbounded =
    Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:(-1)
  in
  equal int 16 (Dtype.Ptr.nbytes ptr);
  is_true ~msg:"Ptr.nbytes rejects unbounded pointers"
    (raises_invalid (fun () -> Dtype.Ptr.nbytes unbounded))

let image_vectorization_preserves_shape () =
  let image = Dtype.Image.imageh [ 8; 16; 4 ] in
  let ptr_vec = Dtype.Ptr.vec 3 image in
  let image_vec = Dtype.Image.vec 3 image in
  equal ptr_dtype ptr_vec image_vec;
  equal int 3 (Dtype.Ptr.v ptr_vec);
  equal int 3 (Dtype.vcount (Dtype.Ptr ptr_vec));
  equal (list int) [ 8; 16; 4 ] (Dtype.Image.shape ptr_vec);
  equal int 512 (Dtype.Ptr.size ptr_vec);
  equal string "imageh(8,16,4).vec(3)" (Dtype.Ptr.to_string ptr_vec)

let pointer_revectorize_rejected () =
  let ptr = Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Global ~size:4 in
  let vec = Dtype.Ptr.vec 2 ptr in
  equal int 2 (Dtype.Ptr.v vec);
  let rejected =
    raises_invalid (fun () -> Dtype.Ptr.vec 4 vec)
  in
  is_true ~msg:"Ptr.vec rejects already-vectorized pointer" rejected

let pointer_bounds_do_not_use_pointee () =
  let int_ptr =
    Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Global ~size:4
  in
  let float_ptr =
    Dtype.Ptr.create Dtype.Val.float32 ~addrspace:Dtype.Global ~size:4
  in
  equal bound (`Bool false) (Dtype.min (Dtype.Ptr int_ptr));
  equal bound (`Bool true) (Dtype.max (Dtype.Ptr float_ptr));
  is_true
    ~msg:"finfo rejects ordinary pointers"
    (raises_invalid (fun () -> Dtype.finfo (Dtype.Ptr float_ptr)))

let image_bounds_are_float_like () =
  let image = Dtype.Image.imageh [ 2; 3; 4 ] in
  equal bound (`Float neg_infinity) (Dtype.min (Dtype.Ptr image));
  equal bound (`Float infinity) (Dtype.max (Dtype.Ptr image));
  is_true
    ~msg:"finfo rejects image pointers"
    (raises_invalid (fun () -> Dtype.finfo (Dtype.Ptr image)))

let top_level_promotion () =
  let imageh = Dtype.Image.imageh [ 2; 3; 4 ] in
  let imagef = Dtype.Image.imagef [ 4; 5; 4 ] in
  let ptr = Dtype.Ptr.create Dtype.Val.int32 ~addrspace:Dtype.Global ~size:4 in
  equal any_dtype Dtype.float32
    (Dtype.least_upper_dtype [ Dtype.int8; Dtype.float32 ]);
  equal any_dtype (Dtype.Ptr imageh)
    (Dtype.least_upper_dtype [ Dtype.float32; Dtype.Ptr imageh ]);
  equal any_dtype (Dtype.Ptr imagef)
    (Dtype.least_upper_dtype [ Dtype.Ptr imagef; Dtype.Ptr imageh ]);
  equal any_dtype Dtype.float32 (Dtype.least_upper_float Dtype.int8);
  equal any_dtype (Dtype.vec 4 Dtype.float32)
    (Dtype.least_upper_float (Dtype.vec 4 Dtype.float32));
  equal any_dtype (Dtype.Ptr imageh)
    (Dtype.least_upper_float (Dtype.Ptr imageh));
  is_true
    ~msg:"least_upper_dtype rejects ordinary pointers"
    (raises_invalid (fun () ->
         Dtype.least_upper_dtype [ Dtype.Ptr imageh; Dtype.Ptr ptr ]));
  is_true
    ~msg:"least_upper_float rejects ordinary pointers"
    (raises_invalid (fun () -> Dtype.least_upper_float (Dtype.Ptr ptr)))

let const_float_identity () =
  equal const (Const.float Dtype.Val.float32 Float.nan)
    (Const.float Dtype.Val.float32 Float.nan);
  is_false
    (Const.equal (Const.float Dtype.Val.float32 0.0)
       (Const.float Dtype.Val.float32 (-0.0)));
  equal float_bits 0.0
    (match Const.view (Const.float Dtype.Val.float32 0.0) with
    | Const.Float f -> f
    | _ -> assert false);
  equal float_bits (-0.0)
    (match Const.view (Const.float Dtype.Val.float32 (-0.0)) with
    | Const.Float f -> f
    | _ -> assert false)

let const_uop_float_identity () =
  let nan_a = Uop.const (Const.float Dtype.Val.float32 Float.nan) in
  let nan_b = Uop.const (Const.float Dtype.Val.float32 Float.nan) in
  is_true ~msg:"NaN constants hash-cons to one UOp"
    (Uop.equal nan_a nan_b);
  let pos_zero = Uop.const (Const.float Dtype.Val.float32 0.0) in
  let neg_zero = Uop.const (Const.float Dtype.Val.float32 (-0.0)) in
  is_false ~msg:"signed-zero constants remain distinct UOps"
    (Uop.equal pos_zero neg_zero)

let const_dtype_directed () =
  equal const (Const.float Dtype.Val.float32 1.0)
    (Const.of_scalar Dtype.Val.float32 (`Bool true));
  let vec0 = Dtype.Val.vec 0 Dtype.Val.int32 in
  let vec0_const = Const.of_scalar vec0 (`Int 7L) in
  equal dtype vec0 (Const.dtype vec0_const);
  is_true
    ~msg:"zero-lane vector const keeps scalar-broadcast payload"
    (match Const.view vec0_const with
    | Const.Int 7L -> true
    | _ -> false);
  equal const (Const.bool true)
    (Const.of_scalar Dtype.Val.bool (`Float Float.nan));
  equal const (Const.int64 Dtype.Val.int32 7L)
    (Const.of_scalar Dtype.Val.int32 (`Float 7.9));
  equal const
    (Const.invalid ~dtype:Dtype.Val.float32 ())
    (Const.of_view Dtype.Val.float32 Const.Invalid);
  let private_const =
    Const.of_view (Dtype.Val.of_scalar Dtype.Uint128) (Const.Float 2.5)
  in
  equal dtype (Dtype.Val.of_scalar Dtype.Uint128) (Const.dtype private_const);
  equal bool true
    (match Const.view private_const with
    | Const.Int 2L -> true
    | _ -> false)

let dtype_surface_tests =
  [
    group "Dtype"
      [
        test "storage format surface" storage_formats;
        test "generic truncation surface" truncation_surface;
        test "storage scalar round-trips" storage_roundtrips;
        test "default dtype policy" defaults;
        test "env dtype parsing follows tinygrad aliases" env_dtype_parsing;
        test "ALU address space" address_space;
        test "pointer predicates are not scalar predicates"
          pointer_predicates;
        test "private wide helpers follow tinygrad classification"
          private_wide_helpers;
        test "value vector edge surface" vector_edge_surface;
        test "structural dtype identity surface" dtype_identity_surface;
        test "tinygrad dtype repr" tinygrad_repr_surface;
        test "image dtype surface" image_dtype_surface;
        test "pointer nbytes" pointer_nbytes;
        test "image vectorization preserves shape"
          image_vectorization_preserves_shape;
        test "pointer re-vectorization is rejected" pointer_revectorize_rejected;
        test "pointer bounds do not use pointee scalar"
          pointer_bounds_do_not_use_pointee;
        test "image bounds are float-like" image_bounds_are_float_like;
        test "top-level promotion" top_level_promotion;
      ];
    group "Const"
      [
        test "NaN and signed-zero identity" const_float_identity;
        test "NaN and signed-zero UOp identity" const_uop_float_identity;
        test "dtype-directed constants" const_dtype_directed;
      ];
  ]

(* Legacy IR dtype coverage, consolidated under the Dtype UOp suite. *)

open Windtrap
open Tolk_uop

let dtype = testable ~pp:Dtype.Val.pp ~equal:Dtype.Val.equal ()

let bound =
  let pp fmt = function
    | `Bool b -> Format.fprintf fmt "`Bool %b" b
    | `SInt n -> Format.fprintf fmt "`SInt %Ld" n
    | `UInt n -> Format.fprintf fmt "`UInt %Ld" n
    | `Float f -> Format.fprintf fmt "`Float %g" f
  in
  let equal a b =
    match (a, b) with
    | `Bool a, `Bool b -> a = b
    | `SInt a, `SInt b -> Int64.equal a b
    | `UInt a, `UInt b -> Int64.equal a b
    | `Float a, `Float b ->
        (Float.is_nan a && Float.is_nan b) || Float.equal a b
    | _ -> false
  in
  testable ~pp ~equal ()

let int_pair =
  let pp fmt (a, b) = Format.fprintf fmt "(%d, %d)" a b in
  testable ~pp ~equal:( = ) ()

let raises_invalid (f : unit -> _) =
  raises_match (function Invalid_argument _ -> true | _ -> false) f

(* Dtypes that participate in promotion (excludes Void and Index). *)
let promotable_dtypes =
  Dtype.Val.
    [
      bool; int8; int16; int32; int64; uint8; uint16; uint32; uint64; float16;
      bfloat16; float32; float64; fp8e4m3; fp8e5m2;
    ]

let promotable_dtype =
  let gen = Gen.oneofl promotable_dtypes in
  testable ~pp:Dtype.Val.pp ~equal:Dtype.Val.equal ~gen ()

(* Integer dtypes suitable for truncate_int (excludes Index). *)
let int_dtypes = Dtype.Val.[ bool; int8; int16; int32; uint8; uint16; uint32 ]

let int_dtype =
  let gen = Gen.oneofl int_dtypes in
  testable ~pp:Dtype.Val.pp ~equal:Dtype.Val.equal ~gen ()

let fp8_byte =
  let gen = Gen.int_range 0 255 in
  testable ~pp:Format.pp_print_int ~equal:Int.equal ~gen ()

let lub = Dtype.Val.least_upper_dtype

let dtype_legacy_tests =
    [
      group "Type Promotion"
        [
          test "lattice edges" (fun () ->
            equal dtype Dtype.Val.int8 (lub [ Dtype.Val.bool; Dtype.Val.int8 ]);
            equal dtype Dtype.Val.int16 (lub [ Dtype.Val.int8; Dtype.Val.uint8 ]);
            equal dtype Dtype.Val.int32 (lub [ Dtype.Val.int16; Dtype.Val.uint16 ]);
            equal dtype Dtype.Val.int64 (lub [ Dtype.Val.int32; Dtype.Val.uint32 ]);
            (* Cross-category: int through float. *)
            equal dtype Dtype.Val.float16 (lub [ Dtype.Val.float16; Dtype.Val.int64 ]);
            (* FP8 siblings meet at float16. *)
            equal dtype Dtype.Val.float16 (lub [ Dtype.Val.fp8e4m3; Dtype.Val.fp8e5m2 ]);
            (* Float16 and bfloat16 are incomparable; they meet at float32. *)
            equal dtype Dtype.Val.float32 (lub [ Dtype.Val.float16; Dtype.Val.bfloat16 ]));
          test "strips vectorization" (fun () ->
            let vec4 = Dtype.Val.vec 4 Dtype.Val.int8 in
            equal dtype Dtype.Val.int16 (lub [ vec4; Dtype.Val.uint8 ]));
          test "errors" (fun () ->
            raises_invalid_arg "Val.least_upper_dtype: empty list"
              (fun () -> lub []);
            equal dtype Dtype.Val.weakint (lub [ Dtype.Val.weakint ]));
          prop2 "commutative" promotable_dtype promotable_dtype (fun a b ->
            Dtype.Val.equal (lub [ a; b ]) (lub [ b; a ]));
          prop "idempotent" promotable_dtype (fun a ->
            Dtype.Val.equal (lub [ a; a ]) (Dtype.Val.scalarize a));
        ];
      group "Lossless Cast"
        [
          test "widening" (fun () ->
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.int8 Dtype.Val.int16);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.int16 Dtype.Val.int32);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.uint8 Dtype.Val.uint16);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.float16 Dtype.Val.float32);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.float32 Dtype.Val.float64);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.fp8e4m3 Dtype.Val.float16);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.fp8e5m2 Dtype.Val.float16));
          test "narrowing fails" (fun () ->
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.int32 Dtype.Val.int16);
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.float64 Dtype.Val.float32);
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.float16 Dtype.Val.fp8e4m3));
          test "cross-sign" (fun () ->
            (* uint8 fits in int16 (wider signed). *)
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.uint8 Dtype.Val.int16);
            (* int8 doesn't fit in uint8 (loses negatives). *)
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.int8 Dtype.Val.uint8);
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.int16 Dtype.Val.uint16));
          test "to index" (fun () ->
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.int32 Dtype.Val.weakint);
            is_true (Dtype.Val.can_lossless_cast Dtype.Val.uint64 Dtype.Val.weakint);
            is_false (Dtype.Val.can_lossless_cast Dtype.Val.float32 Dtype.Val.weakint));
          prop "reflexive" promotable_dtype (fun a ->
            Dtype.Val.can_lossless_cast a a);
        ];
      group "Sum Accumulator"
        [
          test "all categories" (fun () ->
            (* Unsigned widens to at least uint32. *)
            equal dtype Dtype.Val.uint32 (Dtype.Val.sum_acc_dtype Dtype.Val.uint8);
            equal dtype Dtype.Val.uint32 (Dtype.Val.sum_acc_dtype Dtype.Val.uint32);
            equal dtype Dtype.Val.uint64 (Dtype.Val.sum_acc_dtype Dtype.Val.uint64);
            (* Signed widens to at least int32. *)
            equal dtype Dtype.Val.int32 (Dtype.Val.sum_acc_dtype Dtype.Val.int8);
            equal dtype Dtype.Val.int64 (Dtype.Val.sum_acc_dtype Dtype.Val.int64);
            (* Bool accumulates as int32. *)
            equal dtype Dtype.Val.int32 (Dtype.Val.sum_acc_dtype Dtype.Val.bool);
            (* Floats widen to at least float32. *)
            equal dtype Dtype.Val.float32 (Dtype.Val.sum_acc_dtype Dtype.Val.float16);
            equal dtype Dtype.Val.float64 (Dtype.Val.sum_acc_dtype Dtype.Val.float64);
            (* Weakint accumulates as the default integer dtype. *)
            equal dtype Dtype.Val.int32 (Dtype.Val.sum_acc_dtype Dtype.Val.weakint));
          prop "idempotent" promotable_dtype (fun a ->
            Dtype.Val.equal
              (Dtype.Val.sum_acc_dtype (Dtype.Val.sum_acc_dtype a))
              (Dtype.Val.sum_acc_dtype a));
        ];
      group "FP16 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
            eq 1.0 (Dtype.float_to_fp16 1.0);
            eq (-1.0) (Dtype.float_to_fp16 (-1.0));
            eq 0.0 (Dtype.float_to_fp16 0.0);
            eq (-0.0) (Dtype.float_to_fp16 (-0.0));
            (* Max representable. *)
            eq 65504.0 (Dtype.float_to_fp16 65504.0);
            (* Overflow to infinity. *)
            eq infinity (Dtype.float_to_fp16 65520.0);
            eq neg_infinity (Dtype.float_to_fp16 (-65520.0));
            (* Underflow to zero. *)
            eq 0.0 (Dtype.float_to_fp16 1e-8);
            (* Non-finite passthrough. *)
            eq infinity (Dtype.float_to_fp16 infinity);
            eq neg_infinity (Dtype.float_to_fp16 neg_infinity);
            is_true (Float.is_nan (Dtype.float_to_fp16 Float.nan)));
          test "denormal range" (fun () ->
            (* Smallest positive fp16 denormal: 2^-24 *)
            let x = Float.ldexp 1.0 (-24) in
            equal (float 0.0) x (Dtype.float_to_fp16 x);
            (* Largest fp16 denormal: just below 2^-14. *)
            let x = Float.ldexp 1.0 (-14) -. Float.ldexp 1.0 (-24) in
            let r = Dtype.float_to_fp16 x in
            is_true ~msg:"denormal round-trips to finite" (Float.is_finite r);
            is_true ~msg:"denormal non-zero" (r > 0.0));
          prop "idempotent" (float 0.0) (fun x ->
            let r = Dtype.float_to_fp16 x in
            if Float.is_nan r then Float.is_nan (Dtype.float_to_fp16 r)
            else Float.equal r (Dtype.float_to_fp16 r));
        ];
      group "BF16 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
            eq 1.0 (Dtype.float_to_bf16 1.0);
            eq 0.0 (Dtype.float_to_bf16 0.0);
            (* 128.0 = 1.0 * 2^7, exactly representable. *)
            eq 128.0 (Dtype.float_to_bf16 128.0);
            (* 1234.0 needs 10 mantissa bits, rounds to 1232.0 in bf16's 7. *)
            eq 1232.0 (Dtype.float_to_bf16 1234.0);
            (* Non-finite passthrough. *)
            eq infinity (Dtype.float_to_bf16 infinity);
            eq neg_infinity (Dtype.float_to_bf16 neg_infinity);
            is_true (Float.is_nan (Dtype.float_to_bf16 Float.nan)));
          prop "idempotent" (float 0.0) (fun x ->
            let r = Dtype.float_to_bf16 x in
            if Float.is_nan r then Float.is_nan (Dtype.float_to_bf16 r)
            else Float.equal r (Dtype.float_to_bf16 r));
        ];
      group "FP8 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
            equal int 0 (Dtype.float_to_fp8 Fp8e4m3 0.0);
            equal int 0 (Dtype.float_to_fp8 Fp8e5m2 0.0);
            eq 0.0 (Dtype.fp8_to_float Fp8e4m3 0);
            eq 0.0 (Dtype.fp8_to_float Fp8e5m2 0);
            (* E4m3 max normal: 448.0. *)
            eq 448.0
              (Dtype.fp8_to_float Fp8e4m3
                 (Dtype.float_to_fp8 Fp8e4m3 448.0));
            (* E4m3 is saturating: infinity -> NaN, above-max -> maxnorm. *)
            is_true
              (Float.is_nan
                 (Dtype.fp8_to_float Fp8e4m3
                    (Dtype.float_to_fp8 Fp8e4m3 infinity)));
            eq 448.0
              (Dtype.fp8_to_float Fp8e4m3
                 (Dtype.float_to_fp8 Fp8e4m3 500.0));
            (* E5m2 max normal: 57344.0. *)
            eq 57344.0
              (Dtype.fp8_to_float Fp8e5m2
                 (Dtype.float_to_fp8 Fp8e5m2 57344.0));
            (* E5m2 is IEEE-like: infinity -> infinity, NaN -> NaN. *)
            eq infinity
              (Dtype.fp8_to_float Fp8e5m2
                 (Dtype.float_to_fp8 Fp8e5m2 infinity));
            equal int 0x7f (Dtype.float_to_fp8 Fp8e5m2 Float.nan);
            let neg_nan =
              Int64.float_of_bits
                (Int64.logor Int64.min_int 0x7FF8000000000000L)
            in
            equal int 0xff (Dtype.float_to_fp8 Fp8e5m2 neg_nan);
            is_true
              (Float.is_nan
                 (Dtype.fp8_to_float Fp8e5m2
                    (Dtype.float_to_fp8 Fp8e5m2 Float.nan)));
            raises_invalid (fun () -> Dtype.float_to_fp8 Int8 1.0);
            raises_invalid (fun () -> Dtype.fp8_to_float Int8 0));
          prop "byte round-trip stable" fp8_byte (fun byte ->
            List.for_all
              (fun s ->
                let f = Dtype.fp8_to_float s byte in
                let byte' = Dtype.float_to_fp8 s f in
                let f' = Dtype.fp8_to_float s byte' in
                (Float.is_nan f && Float.is_nan f') || Float.equal f f')
              [ Fp8e4m3; Fp8e5m2 ]);
        ];
      group "Integer Truncation"
        [
          test "boundaries" (fun () ->
            (* In-range identity. *)
            equal int 42 (Dtype.truncate_int Dtype.Val.int8 42);
            equal int (-1) (Dtype.truncate_int Dtype.Val.int8 (-1));
            (* Unsigned wrap. *)
            equal int 0 (Dtype.truncate_int Dtype.Val.uint8 256);
            equal int 255 (Dtype.truncate_int Dtype.Val.uint8 255);
            equal int 0 (Dtype.truncate_int Dtype.Val.uint16 65536);
            (* Signed wrap with sign extension. *)
            equal int (-128) (Dtype.truncate_int Dtype.Val.int8 128);
            equal int (-1) (Dtype.truncate_int Dtype.Val.int8 255);
            equal int (-1) (Dtype.truncate_int Dtype.Val.int16 65535);
            (* Bool: 0 -> 0, nonzero -> 1. *)
            equal int 0 (Dtype.truncate_int Dtype.Val.bool 0);
            equal int 1 (Dtype.truncate_int Dtype.Val.bool 1);
            equal int 1 (Dtype.truncate_int Dtype.Val.bool 2);
            raises_invalid (fun () -> Dtype.truncate_int Dtype.Val.float32 1));
          prop "idempotent" (pair int_dtype int) (fun (dt, x) ->
            let r = Dtype.truncate_int dt x in
            r = Dtype.truncate_int dt r);
        ];
      group "Vec"
        [
          test "operations" (fun () ->
            let v = Dtype.Val.vec 4 Dtype.Val.int32 in
            equal dtype (Dtype.Val.vec 4 Dtype.Val.int32) v;
            (* Count=1 is identity. *)
            equal dtype Dtype.Val.int32 (Dtype.Val.vec 1 Dtype.Val.int32);
            (* Void ignores count. *)
            equal dtype Dtype.Val.void (Dtype.Val.vec 4 Dtype.Val.void);
            (* index.vec(0) for empty shape vectors. *)
            equal int 0 (Dtype.Val.count (Dtype.Val.vec 0 Dtype.Val.weakint));
            (* tinygrad's raw dtype layer permits zero-lane vectors for any
               non-void scalar. *)
            let int0 = Dtype.Val.vec 0 Dtype.Val.int32 in
            equal int 0 (Dtype.Val.count int0);
            equal int 0 (Dtype.Val.bitsize int0);
            equal dtype Dtype.Val.int32 (Dtype.Val.scalarize int0);
            (* scalar_of strips count. *)
            equal dtype Dtype.Val.int32 (Dtype.Val.scalarize v);
            equal dtype Dtype.Val.float64 (Dtype.Val.scalarize Dtype.Val.float64));
          test "errors" (fun () ->
            raises_invalid (fun () -> Dtype.Val.vec 2 (Dtype.Val.vec 4 Dtype.Val.int32));
            raises_invalid (fun () -> Dtype.Val.vec (-1) Dtype.Val.int32));
        ];
      group "Bounds"
        [
          test "spot checks" (fun () ->
            equal bound (`Bool false) (Dtype.min (Dtype.Val Dtype.Val.bool));
            equal bound (`Bool true) (Dtype.max (Dtype.Val Dtype.Val.bool));
            equal bound (`SInt (-128L)) (Dtype.min (Dtype.Val Dtype.Val.int8));
            equal bound (`SInt 127L) (Dtype.max (Dtype.Val Dtype.Val.int8));
            equal bound (`UInt 0L) (Dtype.min (Dtype.Val Dtype.Val.uint8));
            equal bound (`UInt 255L) (Dtype.max (Dtype.Val Dtype.Val.uint8));
            equal bound (`SInt Int64.min_int) (Dtype.min (Dtype.Val Dtype.Val.int64));
            equal bound (`SInt Int64.max_int) (Dtype.max (Dtype.Val Dtype.Val.int64));
            equal bound (`UInt Int64.minus_one) (Dtype.max (Dtype.Val Dtype.Val.uint64));
            equal bound (`Float neg_infinity) (Dtype.min (Dtype.Val Dtype.Val.float32));
            equal bound (`Float infinity) (Dtype.max (Dtype.Val Dtype.Val.float64));
            (* Vec inherits scalar bounds. *)
            equal bound (`SInt (-128L)) (Dtype.min (Dtype.Val (Dtype.Val.vec 4 Dtype.Val.int8)));
            raises_invalid_arg "void has no numeric bounds" (fun () ->
              Dtype.min (Dtype.Val Dtype.Val.void)));
        ];
      group "Float Info"
        [
          test "all types" (fun () ->
            equal int_pair (5, 10) (Dtype.finfo (Dtype.Val Dtype.Val.float16));
            equal int_pair (8, 7) (Dtype.finfo (Dtype.Val Dtype.Val.bfloat16));
            equal int_pair (8, 23) (Dtype.finfo (Dtype.Val Dtype.Val.float32));
            equal int_pair (11, 52) (Dtype.finfo (Dtype.Val Dtype.Val.float64));
            equal int_pair (4, 3) (Dtype.finfo (Dtype.Val Dtype.Val.fp8e4m3));
            equal int_pair (5, 2) (Dtype.finfo (Dtype.Val Dtype.Val.fp8e5m2));
            raises_invalid_arg "finfo: not a floating-point dtype" (fun () ->
              Dtype.finfo (Dtype.Val Dtype.Val.int32)));
        ];
    ]

let () =
  match Sys.getenv_opt "TOLK_DTYPE_ENV_CASE" with
  | Some case -> run_env_case case
  | None -> run "tolk.uop.dtype" (dtype_surface_tests @ dtype_legacy_tests)
