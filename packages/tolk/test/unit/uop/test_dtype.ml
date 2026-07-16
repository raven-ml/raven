open Windtrap
open Tolk_uop

let dtype = testable ~pp:Dtype.pp ~equal:Dtype.equal ()

let bound =
  let pp fmt = function
    | `Bool b -> Format.fprintf fmt "`Bool %b" b
    | `SInt n -> Format.fprintf fmt "`SInt %Ld" n
    | `UInt n -> Format.fprintf fmt "`UInt %Ld" n
    | `Float f -> Format.fprintf fmt "`Float %h" f
  in
  let equal a b =
    match (a, b) with
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
    match (a, b) with
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

let int_pair =
  let pp fmt (a, b) = Format.fprintf fmt "(%d, %d)" a b in
  testable ~pp ~equal:( = ) ()

let const = testable ~pp:Const.pp ~equal:Const.equal ()

let raises_invalid f =
  try
    ignore (f ());
    false
  with Invalid_argument _ -> true

(* Oracle tables generated from the tinygrad clone at parity time. Each entry
   is the ground-truth expected result of the corresponding function; the tests
   below assert Tolk reproduces them exactly across the whole surface. *)

let lud_order =
  Dtype.
    [|
      weakint; bool; int8; int16; int32; int64; uint8; uint16; uint32; uint64;
      weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16;
      float32; float64;
    |]

(* lud_expected.(i).(j) = least_upper_dtype [lud_order.(i); lud_order.(j)]. *)
let lud_expected =
  Dtype.[|
    [| weakint; weakint; int8; int16; int32; int64; uint8; uint16; uint32; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| weakint; bool; int8; int16; int32; int64; uint8; uint16; uint32; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| int8; int8; int8; int16; int32; int64; int16; int32; int64; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| int16; int16; int16; int16; int32; int64; int16; int32; int64; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| int32; int32; int32; int32; int32; int64; int32; int32; int64; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| int64; int64; int64; int64; int64; int64; int64; int64; int64; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| uint8; uint8; int16; int16; int32; int64; uint8; uint16; uint32; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| uint16; uint16; int32; int32; int32; int64; uint16; uint16; uint32; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| uint32; uint32; int64; int64; int64; int64; uint32; uint32; uint32; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| uint64; uint64; uint64; uint64; uint64; uint64; uint64; uint64; uint64; uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; fp8e4m3; float16; float16; float16; float16; bfloat16; float32; float64 |];
    [| fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; fp8e5m2; float16; fp8e5m2; float16; float16; float16; bfloat16; float32; float64 |];
    [| fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; fp8e4m3fnuz; float16; float16; fp8e4m3fnuz; float16; float16; bfloat16; float32; float64 |];
    [| fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; fp8e5m2fnuz; float16; float16; float16; fp8e5m2fnuz; float16; bfloat16; float32; float64 |];
    [| float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float16; float32; float32; float64 |];
    [| bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; bfloat16; float32; bfloat16; float32; float64 |];
    [| float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float32; float64 |];
    [| float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64; float64 |];
  |]

let clc_order =
  Dtype.
    [|
      index; weakint; bool; int8; int16; int32; int64; uint8; uint16; uint32;
      uint64; weakfloat; fp8e4m3; fp8e5m2; fp8e4m3fnuz; fp8e5m2fnuz; float16;
      bfloat16; float32; float64;
    |]

(* clc_expected.(i).(j) = can_lossless_cast clc_order.(i) clc_order.(j). *)
let clc_expected =
  [|
    [| 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 |];
    [| 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 |];
    [| 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1 |];
    [| 1; 1; 0; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 1; 1 |];
    [| 1; 1; 0; 0; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1 |];
    [| 1; 1; 0; 0; 0; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1 |];
    [| 1; 1; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 |];
    [| 1; 1; 0; 0; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 1; 0; 1; 1 |];
    [| 1; 1; 0; 0; 0; 1; 1; 0; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 1; 1 |];
    [| 1; 1; 0; 0; 0; 0; 1; 0; 0; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 1 |];
    [| 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 1; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 1; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 1; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 1 |];
    [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1 |];
  |]

(* Predicates and identity *)

let predicates () =
  equal int 0 (Dtype.priority Dtype.index);
  equal int 9 (Dtype.priority Dtype.weakfloat);
  equal int 800 (Dtype.bitsize Dtype.index);
  equal int 800 (Dtype.bitsize Dtype.weakfloat);
  is_true ~msg:"index is an integer" (Dtype.is_int Dtype.index);
  is_false ~msg:"index is not unsigned" (Dtype.is_unsigned Dtype.index);
  is_false ~msg:"index is not weak" (Dtype.is_weak Dtype.index);
  is_false ~msg:"index is not float" (Dtype.is_float Dtype.index);
  is_true ~msg:"weakfloat is a float" (Dtype.is_float Dtype.weakfloat);
  is_true ~msg:"weakfloat is weak" (Dtype.is_weak Dtype.weakfloat);
  is_true ~msg:"weakint is weak" (Dtype.is_weak Dtype.weakint);
  is_true ~msg:"weakint is an integer" (Dtype.is_int Dtype.weakint);
  is_false ~msg:"concrete floats are not weak" (Dtype.is_weak Dtype.float32);
  is_true (Dtype.is_fp8 Dtype.fp8e4m3);
  is_false (Dtype.is_fp8 Dtype.float16)

(* Private wide storage helpers are not classified as integers and carry no
   public representation. *)
let private_wide_helpers () =
  is_false (Dtype.is_int Dtype.Uint128);
  is_false (Dtype.is_unsigned Dtype.Uint128);
  is_false (Dtype.is_int Dtype.Uint256);
  is_false (Dtype.is_unsigned Dtype.Uint256);
  equal bound (`Bool false) (Dtype.min Dtype.Uint128);
  equal bound (`Bool true) (Dtype.max Dtype.Uint256);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.Uint128);
  is_true ~msg:"private wide helper dtypes have no public repr"
    (raises_invalid (fun () -> Dtype.repr Dtype.Uint128))

let repr_surface () =
  equal string "dtypes.int" (Dtype.repr Dtype.int32);
  equal string "dtypes.char" (Dtype.repr Dtype.int8);
  equal string "dtypes.uchar" (Dtype.repr Dtype.uint8);
  equal string "dtypes.long" (Dtype.repr Dtype.int64);
  equal string "dtypes.ulong" (Dtype.repr Dtype.uint64);
  equal string "dtypes.float" (Dtype.repr Dtype.float32);
  equal string "dtypes.half" (Dtype.repr Dtype.float16);
  equal string "dtypes.bfloat16" (Dtype.repr Dtype.bfloat16);
  equal string "dtypes.fp8e4m3" (Dtype.repr Dtype.fp8e4m3);
  equal string "dtypes.bool" (Dtype.repr Dtype.bool);
  equal string "dtypes.weakint" (Dtype.repr Dtype.weakint);
  equal string "dtypes.index" (Dtype.repr Dtype.index);
  equal string "dtypes.weakfloat" (Dtype.repr Dtype.weakfloat)

let address_space () =
  equal string "global" (Dtype.addr_space_to_string Dtype.Global);
  equal string "local" (Dtype.addr_space_to_string Dtype.Local);
  equal string "reg" (Dtype.addr_space_to_string Dtype.Reg);
  equal string "alu" (Dtype.addr_space_to_string Dtype.Alu)

(* Promotion *)

let lub = Dtype.least_upper_dtype

let promotable_dtypes =
  Dtype.
    [
      bool; int8; int16; int32; int64; uint8; uint16; uint32; uint64; weakfloat;
      float16; bfloat16; float32; float64; fp8e4m3; fp8e5m2;
    ]

let promotable_dtype =
  let gen = Gen.oneofl promotable_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

let promotion_matrix () =
  Array.iteri
    (fun i a ->
      Array.iteri
        (fun j b -> equal dtype lud_expected.(i).(j) (lub [ a; b ]))
        lud_order)
    lud_order

let promotion_edges () =
  equal dtype Dtype.int8 (lub [ Dtype.bool; Dtype.int8 ]);
  equal dtype Dtype.int16 (lub [ Dtype.int8; Dtype.uint8 ]);
  (* Cross-category: int promotes through weakfloat/fp8 into float. *)
  equal dtype Dtype.float16 (lub [ Dtype.float16; Dtype.int64 ]);
  (* uint64 reaches the floats via weakfloat. *)
  equal dtype Dtype.weakfloat (lub [ Dtype.int64; Dtype.uint64 ]);
  (* FP8 siblings meet at float16. *)
  equal dtype Dtype.float16 (lub [ Dtype.fp8e4m3; Dtype.fp8e5m2 ]);
  (* Float16 and bfloat16 are incomparable; they meet at float32. *)
  equal dtype Dtype.float32 (lub [ Dtype.float16; Dtype.bfloat16 ])

let promotion_errors () =
  raises_invalid_arg "Dtype.least_upper_dtype: empty list" (fun () -> lub []);
  equal dtype Dtype.weakint (lub [ Dtype.weakint ]);
  is_true ~msg:"index is outside the lattice"
    (raises_invalid (fun () -> lub [ Dtype.index; Dtype.int32 ]));
  is_true ~msg:"void is outside the lattice"
    (raises_invalid (fun () -> lub [ Dtype.void; Dtype.int32 ]))

let least_upper_float_cases () =
  equal dtype Dtype.float32 (Dtype.least_upper_float Dtype.int8);
  equal dtype Dtype.float16 (Dtype.least_upper_float Dtype.float16);
  equal dtype Dtype.weakfloat (Dtype.least_upper_float Dtype.weakfloat)

(* Lossless cast *)

let lossless_matrix () =
  Array.iteri
    (fun i a ->
      Array.iteri
        (fun j b ->
          equal bool (clc_expected.(i).(j) = 1) (Dtype.can_lossless_cast a b))
        clc_order)
    clc_order

let lossless_to_index () =
  is_true (Dtype.can_lossless_cast Dtype.int32 Dtype.index);
  is_true (Dtype.can_lossless_cast Dtype.uint64 Dtype.index);
  is_true (Dtype.can_lossless_cast Dtype.int32 Dtype.weakint);
  is_false (Dtype.can_lossless_cast Dtype.float32 Dtype.index);
  is_false (Dtype.can_lossless_cast Dtype.float32 Dtype.weakint)

(* Sum accumulator *)

let sum_acc () =
  equal dtype Dtype.uint32 (Dtype.sum_acc_dtype Dtype.uint8);
  equal dtype Dtype.uint32 (Dtype.sum_acc_dtype Dtype.uint32);
  equal dtype Dtype.uint64 (Dtype.sum_acc_dtype Dtype.uint64);
  equal dtype Dtype.int32 (Dtype.sum_acc_dtype Dtype.int8);
  equal dtype Dtype.int64 (Dtype.sum_acc_dtype Dtype.int64);
  equal dtype Dtype.int32 (Dtype.sum_acc_dtype Dtype.bool);
  equal dtype Dtype.float32 (Dtype.sum_acc_dtype Dtype.float16);
  equal dtype Dtype.float64 (Dtype.sum_acc_dtype Dtype.float64);
  equal dtype Dtype.int32 (Dtype.sum_acc_dtype Dtype.weakint)

(* FP conversion *)

let fp16_conversion () =
  let eq = equal (float 0.0) in
  eq 1.0 (Dtype.float_to_fp16 1.0);
  eq (-1.0) (Dtype.float_to_fp16 (-1.0));
  eq 0.0 (Dtype.float_to_fp16 0.0);
  eq (-0.0) (Dtype.float_to_fp16 (-0.0));
  eq 65504.0 (Dtype.float_to_fp16 65504.0);
  eq infinity (Dtype.float_to_fp16 65520.0);
  eq neg_infinity (Dtype.float_to_fp16 (-65520.0));
  eq 0.0 (Dtype.float_to_fp16 1e-8);
  eq infinity (Dtype.float_to_fp16 infinity);
  eq neg_infinity (Dtype.float_to_fp16 neg_infinity);
  is_true (Float.is_nan (Dtype.float_to_fp16 Float.nan))

let bf16_conversion () =
  let eq = equal (float 0.0) in
  eq 1.0 (Dtype.float_to_bf16 1.0);
  eq 0.0 (Dtype.float_to_bf16 0.0);
  eq 128.0 (Dtype.float_to_bf16 128.0);
  eq 1232.0 (Dtype.float_to_bf16 1234.0);
  eq infinity (Dtype.float_to_bf16 infinity);
  eq neg_infinity (Dtype.float_to_bf16 neg_infinity);
  is_true (Float.is_nan (Dtype.float_to_bf16 Float.nan))

let fp8_conversion () =
  let eq = equal (float 0.0) in
  equal int 0 (Dtype.float_to_fp8 Dtype.fp8e4m3 0.0);
  equal int 0 (Dtype.float_to_fp8 Dtype.fp8e5m2 0.0);
  eq 0.0 (Dtype.fp8_to_float Dtype.fp8e4m3 0);
  eq 0.0 (Dtype.fp8_to_float Dtype.fp8e5m2 0);
  (* E4m3 max normal 448.0; saturating (infinity -> NaN, above-max -> maxnorm). *)
  eq 448.0
    (Dtype.fp8_to_float Dtype.fp8e4m3 (Dtype.float_to_fp8 Dtype.fp8e4m3 448.0));
  is_true
    (Float.is_nan
       (Dtype.fp8_to_float Dtype.fp8e4m3
          (Dtype.float_to_fp8 Dtype.fp8e4m3 infinity)));
  eq 448.0
    (Dtype.fp8_to_float Dtype.fp8e4m3 (Dtype.float_to_fp8 Dtype.fp8e4m3 500.0));
  (* E5m2 max normal 57344.0; IEEE-like (infinity -> infinity, NaN -> NaN). *)
  eq 57344.0
    (Dtype.fp8_to_float Dtype.fp8e5m2
       (Dtype.float_to_fp8 Dtype.fp8e5m2 57344.0));
  eq infinity
    (Dtype.fp8_to_float Dtype.fp8e5m2
       (Dtype.float_to_fp8 Dtype.fp8e5m2 infinity));
  equal int 0x7f (Dtype.float_to_fp8 Dtype.fp8e5m2 Float.nan);
  let neg_nan =
    Int64.float_of_bits (Int64.logor Int64.min_int 0x7FF8000000000000L)
  in
  equal int 0xff (Dtype.float_to_fp8 Dtype.fp8e5m2 neg_nan);
  is_true
    (Float.is_nan
       (Dtype.fp8_to_float Dtype.fp8e5m2
          (Dtype.float_to_fp8 Dtype.fp8e5m2 Float.nan)));
  is_true ~msg:"float_to_fp8 rejects non-fp8"
    (raises_invalid (fun () -> Dtype.float_to_fp8 Dtype.int8 1.0));
  is_true ~msg:"fp8_to_float rejects non-fp8"
    (raises_invalid (fun () -> Dtype.fp8_to_float Dtype.int8 0))

(* Integer truncation *)

let int_dtypes =
  Dtype.[ bool; int8; int16; int32; uint8; uint16; uint32 ]

let int_dtype =
  let gen = Gen.oneofl int_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

let fp8_byte =
  let gen = Gen.int_range 0 255 in
  testable ~pp:Format.pp_print_int ~equal:Int.equal ~gen ()

let integer_truncation () =
  equal int 42 (Dtype.truncate_int Dtype.int8 42);
  equal int (-1) (Dtype.truncate_int Dtype.int8 (-1));
  equal int 0 (Dtype.truncate_int Dtype.uint8 256);
  equal int 255 (Dtype.truncate_int Dtype.uint8 255);
  equal int 0 (Dtype.truncate_int Dtype.uint16 65536);
  equal int (-128) (Dtype.truncate_int Dtype.int8 128);
  equal int (-1) (Dtype.truncate_int Dtype.int8 255);
  equal int (-1) (Dtype.truncate_int Dtype.int16 65535);
  equal int 0 (Dtype.truncate_int Dtype.bool 0);
  equal int 1 (Dtype.truncate_int Dtype.bool 1);
  equal int 1 (Dtype.truncate_int Dtype.bool 2);
  is_true ~msg:"truncate_int rejects floats"
    (raises_invalid (fun () -> Dtype.truncate_int Dtype.float32 1))

(* Storage boundary *)

let storage_formats () =
  equal char_option (Some '?') (Dtype.storage_fmt_for_dtype Dtype.bool);
  equal char_option (Some 'i') (Dtype.storage_fmt_for_dtype Dtype.int32);
  equal char_option (Some 'Q') (Dtype.storage_fmt_for_dtype Dtype.uint64);
  equal char_option (Some 'H') (Dtype.storage_fmt_for_dtype Dtype.bfloat16);
  equal char_option (Some 'B') (Dtype.storage_fmt_for_dtype Dtype.fp8e4m3);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.weakint);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.index);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.weakfloat);
  equal char_option None (Dtype.storage_fmt_for_dtype Dtype.Uint128)

let truncation_surface () =
  equal storage_scalar (`Bool false) (Dtype.truncate Dtype.bool (`Int 0L));
  equal storage_scalar (`Bool true) (Dtype.truncate Dtype.bool (`Float Float.nan));
  equal storage_scalar (`Int 0L) (Dtype.truncate Dtype.uint8 (`Int 256L));
  equal storage_scalar (`Int 255L) (Dtype.truncate Dtype.uint8 (`Int (-1L)));
  equal storage_scalar (`Int (-128L)) (Dtype.truncate Dtype.int8 (`Int 128L));
  equal storage_scalar (`Float 448.0)
    (Dtype.truncate Dtype.fp8e4m3 (`Float 500.0));
  equal storage_scalar (`Float 1232.0)
    (Dtype.truncate Dtype.bfloat16 (`Float 1234.0))

let storage_roundtrips () =
  let bf16_storage = Dtype.to_storage_scalar Dtype.bfloat16 (`Float 1234.0) in
  equal storage_scalar (`Int 17562L) bf16_storage;
  equal storage_scalar (`Float 1232.0)
    (Dtype.from_storage_scalar bf16_storage Dtype.bfloat16);
  let fp8_storage = Dtype.to_storage_scalar Dtype.fp8e4m3 (`Float 448.0) in
  equal storage_scalar (`Int 126L) fp8_storage;
  equal storage_scalar (`Float 448.0)
    (Dtype.from_storage_scalar fp8_storage Dtype.fp8e4m3);
  equal storage_scalar (`Int 127L)
    (Dtype.to_storage_scalar Dtype.fp8e5m2 (`Float Float.nan));
  let neg_zero_storage = Dtype.to_storage_scalar Dtype.fp8e4m3 (`Float (-0.0)) in
  equal storage_scalar (`Int 128L) neg_zero_storage;
  equal float_bits (-0.0)
    (match Dtype.from_storage_scalar neg_zero_storage Dtype.fp8e4m3 with
    | `Float f -> f
    | _ -> assert false)

(* Bounds and float info *)

let bounds () =
  equal bound (`Bool false) (Dtype.min Dtype.bool);
  equal bound (`Bool true) (Dtype.max Dtype.bool);
  equal bound (`SInt (-128L)) (Dtype.min Dtype.int8);
  equal bound (`SInt 127L) (Dtype.max Dtype.int8);
  equal bound (`UInt 0L) (Dtype.min Dtype.uint8);
  equal bound (`UInt 255L) (Dtype.max Dtype.uint8);
  equal bound (`SInt Int64.min_int) (Dtype.min Dtype.int64);
  equal bound (`SInt Int64.max_int) (Dtype.max Dtype.int64);
  equal bound (`UInt Int64.minus_one) (Dtype.max Dtype.uint64);
  equal bound (`Float neg_infinity) (Dtype.min Dtype.float32);
  equal bound (`Float infinity) (Dtype.max Dtype.float64);
  (* Weak and index integers report the Int64 range as an approximation. *)
  equal bound (`SInt Int64.min_int) (Dtype.min Dtype.index);
  equal bound (`SInt Int64.max_int) (Dtype.max Dtype.weakint);
  equal bound (`Float neg_infinity) (Dtype.min Dtype.weakfloat);
  raises_invalid_arg "void has no numeric bounds" (fun () -> Dtype.min Dtype.void)

let float_info () =
  equal int_pair (5, 10) (Dtype.finfo Dtype.float16);
  equal int_pair (8, 7) (Dtype.finfo Dtype.bfloat16);
  equal int_pair (8, 23) (Dtype.finfo Dtype.float32);
  equal int_pair (11, 52) (Dtype.finfo Dtype.float64);
  equal int_pair (4, 3) (Dtype.finfo Dtype.fp8e4m3);
  equal int_pair (5, 2) (Dtype.finfo Dtype.fp8e5m2);
  raises_invalid_arg "finfo: not a floating-point dtype" (fun () ->
      Dtype.finfo Dtype.int32);
  is_true ~msg:"finfo rejects weakfloat"
    (raises_invalid (fun () -> Dtype.finfo Dtype.weakfloat))

(* Defaults and environment configuration *)

let defaults () =
  is_true (Dtype.is_float Dtype.default_float);
  equal dtype
    (Dtype.least_upper_dtype [ Dtype.int8; Dtype.default_float ])
    (Dtype.least_upper_float Dtype.int8);
  equal dtype Dtype.float64 (Dtype.sum_acc_dtype Dtype.float64)

let exit_bool ok = exit (if ok then 0 else 1)

let run_env_case = function
  | "default_float_half" ->
      exit_bool (Dtype.equal Dtype.float16 Dtype.default_float)
  | "default_float_float" ->
      exit_bool (Dtype.equal Dtype.float32 Dtype.default_float)
  | "sum_dtype_double" ->
      exit_bool (Dtype.equal Dtype.float64 (Dtype.sum_acc_dtype Dtype.float16))
  | "sum_dtype_bfloat16" ->
      exit_bool (Dtype.equal Dtype.float32 (Dtype.sum_acc_dtype Dtype.float16))
  | "sum_dtype_uchar" ->
      exit_bool (Dtype.equal Dtype.float16 (Dtype.sum_acc_dtype Dtype.float16))
  | "sum_dtype_default_float" ->
      exit_bool (Dtype.equal Dtype.float16 (Dtype.sum_acc_dtype Dtype.float16))
  | "sum_dtype_rejected" ->
      exit_bool (raises_invalid (fun () -> Dtype.sum_acc_dtype Dtype.float16))
  | _ -> exit 2

let env_assignment name value = name ^ "=" ^ Filename.quote value

let run_with_env ?default_float ?sum_dtype case =
  let assignments =
    [
      Some (env_assignment "TOLK_DTYPE_ENV_CASE" case);
      Option.map (env_assignment "DEFAULT_FLOAT") default_float;
      Option.map (env_assignment "SUM_DTYPE") sum_dtype;
    ]
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

let env_dtype_parsing () =
  expect_env_success ~default_float:"half" "default_float_half";
  expect_env_success ~default_float:"float" "default_float_float";
  expect_env_success ~default_float:"default_float" "default_float_float";
  expect_env_failure ~default_float:"f32" "default_float_float";
  expect_env_success ~default_float:"float" ~sum_dtype:"double" "sum_dtype_double";
  expect_env_success ~default_float:"float" ~sum_dtype:"bfloat16"
    "sum_dtype_bfloat16";
  expect_env_success ~default_float:"float" ~sum_dtype:"uchar" "sum_dtype_uchar";
  expect_env_success ~default_float:"half" ~sum_dtype:"default_float"
    "sum_dtype_default_float";
  List.iter
    (fun alias ->
      expect_env_success ~default_float:"float" ~sum_dtype:alias
        "sum_dtype_rejected")
    [
      "i8"; "u8"; "f16"; "f32"; "bf16"; " half "; "uint128"; "u128"; "uint256";
      "u256";
    ]

(* ConstFloat identity: NaN is canonicalized (nan = nan) and signed zero is
   preserved (-0.0 <> 0.0). *)

let const_float_identity () =
  equal const
    (Const.float Dtype.float32 Float.nan)
    (Const.float Dtype.float32 Float.nan);
  is_false
    (Const.equal
       (Const.float Dtype.float32 0.0)
       (Const.float Dtype.float32 (-0.0)));
  equal float_bits 0.0
    (match Const.view (Const.float Dtype.float32 0.0) with
    | Const.Float f -> f
    | _ -> assert false);
  equal float_bits (-0.0)
    (match Const.view (Const.float Dtype.float32 (-0.0)) with
    | Const.Float f -> f
    | _ -> assert false)

let const_uop_float_identity () =
  let nan_a = Uop.const (Const.float Dtype.float32 Float.nan) in
  let nan_b = Uop.const (Const.float Dtype.float32 Float.nan) in
  is_true ~msg:"NaN constants hash-cons to one UOp" (Uop.equal nan_a nan_b);
  let pos_zero = Uop.const (Const.float Dtype.float32 0.0) in
  let neg_zero = Uop.const (Const.float Dtype.float32 (-0.0)) in
  is_false ~msg:"signed-zero constants remain distinct UOps"
    (Uop.equal pos_zero neg_zero)

(* Property-based invariants *)

let properties =
  [
    prop2 "promotion commutative" promotable_dtype promotable_dtype
      (fun a b -> Dtype.equal (lub [ a; b ]) (lub [ b; a ]));
    prop "promotion idempotent" promotable_dtype (fun a ->
        Dtype.equal (lub [ a; a ]) a);
    prop "lossless reflexive" promotable_dtype (fun a ->
        Dtype.can_lossless_cast a a);
    prop "sum_acc idempotent" promotable_dtype (fun a ->
        Dtype.equal
          (Dtype.sum_acc_dtype (Dtype.sum_acc_dtype a))
          (Dtype.sum_acc_dtype a));
    prop "fp16 idempotent" (float 0.0) (fun x ->
        let r = Dtype.float_to_fp16 x in
        if Float.is_nan r then Float.is_nan (Dtype.float_to_fp16 r)
        else Float.equal r (Dtype.float_to_fp16 r));
    prop "bf16 idempotent" (float 0.0) (fun x ->
        let r = Dtype.float_to_bf16 x in
        if Float.is_nan r then Float.is_nan (Dtype.float_to_bf16 r)
        else Float.equal r (Dtype.float_to_bf16 r));
    prop "fp8 byte round-trip stable" fp8_byte (fun byte ->
        List.for_all
          (fun s ->
            let f = Dtype.fp8_to_float s byte in
            let byte' = Dtype.float_to_fp8 s f in
            let f' = Dtype.fp8_to_float s byte' in
            (Float.is_nan f && Float.is_nan f') || Float.equal f f')
          Dtype.[ fp8e4m3; fp8e5m2 ]);
    prop "truncate_int idempotent" (pair int_dtype int) (fun (dt, x) ->
        let r = Dtype.truncate_int dt x in
        r = Dtype.truncate_int dt r);
  ]

let tests =
  [
    group "Dtype"
      [
        test "predicates" predicates;
        test "private wide helpers" private_wide_helpers;
        test "repr" repr_surface;
        test "address space" address_space;
      ];
    group "Promotion"
      [
        test "full lattice matrix" promotion_matrix;
        test "representative edges" promotion_edges;
        test "errors and singletons" promotion_errors;
        test "least_upper_float" least_upper_float_cases;
      ];
    group "Lossless Cast"
      [
        test "full matrix" lossless_matrix; test "to weak and index" lossless_to_index;
      ];
    group "Sum Accumulator" [ test "all categories" sum_acc ];
    group "FP Conversion"
      [
        test "fp16" fp16_conversion;
        test "bf16" bf16_conversion;
        test "fp8" fp8_conversion;
      ];
    group "Integer Truncation" [ test "boundaries" integer_truncation ];
    group "Storage"
      [
        test "formats" storage_formats;
        test "truncation" truncation_surface;
        test "round-trips" storage_roundtrips;
      ];
    group "Bounds" [ test "spot checks" bounds; test "float info" float_info ];
    group "Defaults"
      [
        test "policy" defaults;
        test "env dtype parsing" env_dtype_parsing;
      ];
    group "Const"
      [
        test "NaN and signed-zero identity" const_float_identity;
        test "NaN and signed-zero UOp identity" const_uop_float_identity;
      ];
    group "Properties" properties;
  ]

let () =
  match Sys.getenv_opt "TOLK_DTYPE_ENV_CASE" with
  | Some case -> run_env_case case
  | None -> run "tolk.uop.dtype" tests
