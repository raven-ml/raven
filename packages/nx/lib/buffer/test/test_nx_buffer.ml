(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_buffer
open Windtrap

let invalid_argument = function Invalid_argument _ -> true | _ -> false

(* Test creation of different buffer types *)
let test_create_bfloat16 () =
  let buf = create bfloat16 10 in
  equal ~msg:"bfloat16 buffer size" int 10 (length buf);
  set buf 0 1.0;
  set buf 5 2.5;
  equal ~msg:"bfloat16 get" (float 0.1) 1.0 (get buf 0);
  equal ~msg:"bfloat16 get" (float 0.1) 2.5 (get buf 5)

let test_create_bool () =
  let buf = create Nx_buffer.bool 8 in
  equal ~msg:"bool buffer size" int 8 (length buf);
  set buf 0 true;
  set buf 1 false;
  set buf 7 true;
  equal ~msg:"bool get" bool true (get buf 0);
  equal ~msg:"bool get" bool false (get buf 1);
  equal ~msg:"bool get" bool true (get buf 7)

let test_create_int4 () =
  let buf = create int4 16 in
  equal ~msg:"int4 buffer size" int 16 (length buf);
  set buf 0 (-8);
  set buf 1 7;
  set buf 2 0;
  equal ~msg:"int4 get" int (-8) (get buf 0);
  equal ~msg:"int4 get" int 7 (get buf 1);
  equal ~msg:"int4 get" int 0 (get buf 2)

let test_create_uint4 () =
  let buf = create uint4 16 in
  equal ~msg:"uint4 buffer size" int 16 (length buf);
  set buf 0 0;
  set buf 1 15;
  set buf 2 8;
  equal ~msg:"uint4 get" int 0 (get buf 0);
  equal ~msg:"uint4 get" int 15 (get buf 1);
  equal ~msg:"uint4 get" int 8 (get buf 2)

let test_create_float8_e4m3 () =
  let buf = create float8_e4m3 10 in
  equal ~msg:"float8_e4m3 buffer size" int 10 (length buf);
  set buf 0 0.0;
  set buf 1 1.0;
  set buf 2 (-1.5);
  equal ~msg:"float8_e4m3 get" (float 0.1) 0.0 (get buf 0);
  equal ~msg:"float8_e4m3 get" (float 0.1) 1.0 (get buf 1);
  equal ~msg:"float8_e4m3 get" (float 0.1) (-1.5) (get buf 2)

let test_create_float8_e5m2 () =
  let buf = create float8_e5m2 10 in
  equal ~msg:"float8_e5m2 buffer size" int 10 (length buf);
  set buf 0 0.0;
  set buf 1 2.0;
  set buf 2 (-0.5);
  equal ~msg:"float8_e5m2 get" (float 0.1) 0.0 (get buf 0);
  equal ~msg:"float8_e5m2 get" (float 0.1) 2.0 (get buf 1);
  equal ~msg:"float8_e5m2 get" (float 0.1) (-0.5) (get buf 2)

(* Test genarray creation *)
let test_genarray_creation () =
  let dims = [| 2; 3; 4 |] in
  let ga_bf16 = genarray_create bfloat16 Bigarray.c_layout dims in
  let ga_bool = genarray_create Nx_buffer.bool Bigarray.c_layout dims in
  let ga_fp8 = genarray_create float8_e4m3 Bigarray.c_layout dims in
  equal ~msg:"Genarray bfloat16 dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_bf16));
  equal ~msg:"Genarray bool dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_bool));
  equal ~msg:"Genarray float8 dims" int 3
    (Array.length (Bigarray.Genarray.dims ga_fp8));
  equal ~msg:"Genarray dim 0" int 2 (Bigarray.Genarray.nth_dim ga_bf16 0);
  equal ~msg:"Genarray dim 1" int 3 (Bigarray.Genarray.nth_dim ga_bf16 1);
  equal ~msg:"Genarray dim 2" int 4 (Bigarray.Genarray.nth_dim ga_bf16 2)

(* The C stub maps storage flag bits back to GADT constructor indices by
   declaration order; a mismatch shows up as [kind] returning the wrong
   constructor. Check the round-trip for every constructor. *)
let test_kind_roundtrip () =
  let check : type a b. (a, b) kind -> unit =
   fun k ->
    equal
      ~msg:(kind_name k ^ " kind round-trip")
      bool true
      (kind (create k 2) = k)
  in
  check Nx_buffer.float16;
  check Nx_buffer.float32;
  check Nx_buffer.float64;
  check Nx_buffer.bfloat16;
  check Nx_buffer.float8_e4m3;
  check Nx_buffer.float8_e5m2;
  check Nx_buffer.int4;
  check Nx_buffer.uint4;
  check Nx_buffer.int8;
  check Nx_buffer.uint8;
  check Nx_buffer.int16;
  check Nx_buffer.uint16;
  check Nx_buffer.int32;
  check Nx_buffer.uint32;
  check Nx_buffer.int64;
  check Nx_buffer.uint64;
  check Nx_buffer.complex64;
  check Nx_buffer.complex128;
  check Nx_buffer.bool

(* Test kind_size_in_bytes *)
let test_kind_sizes () =
  equal ~msg:"bfloat16 size" int 2 (kind_size_in_bytes bfloat16);
  equal ~msg:"bool size" int 1 (kind_size_in_bytes Nx_buffer.bool);
  equal ~msg:"int4 size" int 1 (kind_size_in_bytes int4);
  equal ~msg:"uint4 size" int 1 (kind_size_in_bytes uint4);
  equal ~msg:"float8_e4m3 size" int 1 (kind_size_in_bytes float8_e4m3);
  equal ~msg:"float8_e5m2 size" int 1 (kind_size_in_bytes float8_e5m2);
  equal ~msg:"uint32 size" int 4 (kind_size_in_bytes uint32);
  equal ~msg:"uint64 size" int 8 (kind_size_in_bytes uint64);
  equal ~msg:"float32 size" int 4 (kind_size_in_bytes float32);
  equal ~msg:"float64 size" int 8 (kind_size_in_bytes float64);
  equal ~msg:"int32 size" int 4 (kind_size_in_bytes Nx_buffer.int32)

(* Test blit *)
let test_blit () =
  let src = create float32 4 in
  let dst = create float32 4 in
  set src 0 1.0;
  set src 1 2.0;
  set src 2 3.0;
  set src 3 4.0;
  blit ~src ~dst;
  equal ~msg:"blit[0]" (float 1e-6) 1.0 (get dst 0);
  equal ~msg:"blit[3]" (float 1e-6) 4.0 (get dst 3)

(* Test fill *)
let test_fill () =
  let buf = create float32 4 in
  fill buf 7.0;
  equal ~msg:"fill[0]" (float 1e-6) 7.0 (get buf 0);
  equal ~msg:"fill[3]" (float 1e-6) 7.0 (get buf 3)

(* Conversion semantics: store a float through the packed representation and
   read it back. These vectors pin round-to-nearest-even, saturation, NaN and
   subnormal handling; the JavaScript stubs mirror the same algorithms. *)
let store_get k v =
  let buf = create k 1 in
  set buf 0 v;
  get buf 0

let test_bfloat16_semantics () =
  let rt = store_get bfloat16 in
  equal ~msg:"bf16 1.0" (float 0.0) 1.0 (rt 1.0);
  (* 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even, down. *)
  equal ~msg:"bf16 tie to even down" (float 0.0) 1.0 (rt 1.00390625);
  (* 1 + 2^-7 + 2^-8 is halfway with an odd mantissa: ties to even, up. *)
  equal ~msg:"bf16 tie to even up" (float 0.0) 1.015625 (rt 1.01171875);
  equal ~msg:"bf16 inf" (float 0.0) Float.infinity (rt Float.infinity);
  (* Max finite float32 is beyond the max finite bfloat16: rounds to inf. *)
  equal ~msg:"bf16 overflow" (float 0.0) Float.infinity (rt 3.4028234e38);
  equal ~msg:"bf16 nan" bool true (Float.is_nan (rt Float.nan))

let test_float8_e4m3_semantics () =
  let rt = store_get float8_e4m3 in
  equal ~msg:"e4m3 1.0" (float 0.0) 1.0 (rt 1.0);
  (* Exponent 15 is a normal binade in e4m3fn: 256..448 are representable. *)
  equal ~msg:"e4m3 256" (float 0.0) 256.0 (rt 256.0);
  equal ~msg:"e4m3 max finite" (float 0.0) 448.0 (rt 448.0);
  equal ~msg:"e4m3 300 rounds to 288" (float 0.0) 288.0 (rt 300.0);
  (* No infinities: overflow and infinities convert to NaN, matching the
     ml_dtypes and PyTorch e4m3fn casts. 464 is the round-to-nearest boundary
     past 448 and ties to the even finite value. *)
  equal ~msg:"e4m3 460 rounds back to 448" (float 0.0) 448.0 (rt 460.0);
  equal ~msg:"e4m3 tie at 464 stays finite" (float 0.0) 448.0 (rt 464.0);
  equal ~msg:"e4m3 overflow is nan" bool true (Float.is_nan (rt 465.0));
  equal ~msg:"e4m3 512 is nan" bool true (Float.is_nan (rt 512.0));
  equal ~msg:"e4m3 inf is nan" bool true (Float.is_nan (rt Float.infinity));
  equal ~msg:"e4m3 -inf is nan" bool true (Float.is_nan (rt Float.neg_infinity));
  (* Subnormals: min subnormal is 2^-9. *)
  equal ~msg:"e4m3 min subnormal" (float 0.0) 0x1p-9 (rt 0x1p-9);
  equal ~msg:"e4m3 subnormal rounds up" (float 0.0) 0x1p-9 (rt 0x1.8p-10);
  (* Half the min subnormal ties to even: zero. *)
  equal ~msg:"e4m3 underflow" (float 0.0) 0.0 (rt 0x1p-10);
  equal ~msg:"e4m3 nan" bool true (Float.is_nan (rt Float.nan))

let test_float8_e5m2_semantics () =
  let rt = store_get float8_e5m2 in
  equal ~msg:"e5m2 1.0" (float 0.0) 1.0 (rt 1.0);
  equal ~msg:"e5m2 max finite" (float 0.0) 57344.0 (rt 57344.0);
  equal ~msg:"e5m2 below tie stays finite" (float 0.0) 57344.0 (rt 61439.0);
  (* 61440 is halfway between 57344 and 65536: ties to even, to inf. *)
  equal ~msg:"e5m2 tie overflows to inf" (float 0.0) Float.infinity (rt 61440.0);
  equal ~msg:"e5m2 inf" (float 0.0) Float.infinity (rt Float.infinity);
  (* Subnormals: min subnormal is 2^-16. *)
  equal ~msg:"e5m2 min subnormal" (float 0.0) 0x1p-16 (rt 0x1p-16);
  (* 1.5 * 2^-16 is halfway between 2^-16 and 2^-15: ties to even, up. *)
  equal ~msg:"e5m2 subnormal tie to even" (float 0.0) 0x1p-15 (rt 0x1.8p-16);
  (* Half the min subnormal ties to even: zero. *)
  equal ~msg:"e5m2 underflow" (float 0.0) 0.0 (rt 0x1p-17);
  equal ~msg:"e5m2 nan" bool true (Float.is_nan (rt Float.nan))

let test_int4_clamping () =
  equal ~msg:"int4 clamps high" int 7 (store_get int4 9);
  equal ~msg:"int4 clamps low" int (-8) (store_get int4 (-9));
  equal ~msg:"uint4 clamps high" int 15 (store_get uint4 99);
  equal ~msg:"uint4 clamps low" int 0 (store_get uint4 (-1))

let test_uint64_roundtrip () =
  equal ~msg:"uint64 all ones" int64 (-1L) (store_get uint64 (-1L));
  equal ~msg:"uint64 max int64 + 1" int64 Int64.min_int
    (store_get uint64 Int64.min_int)

(* Int4 packs two elements per byte: the bytes blits move whole bytes, so
   element offsets must be even and map to byte offset [off / 2]. *)
let test_int4_bytes_blit_roundtrip () =
  let src = create int4 8 in
  for i = 0 to 7 do
    set src i (i - 4)
  done;
  let bytes = Bytes.create 4 in
  blit_to_bytes src bytes;
  let dst = create int4 8 in
  blit_from_bytes bytes dst;
  for i = 0 to 7 do
    equal ~msg:(Printf.sprintf "int4 roundtrip[%d]" i) int (i - 4) (get dst i)
  done

let test_int4_bytes_blit_offsets () =
  let src = create int4 8 in
  for i = 0 to 7 do
    set src i (i mod 8)
  done;
  (* Elements 4..7 of [src], packed into two bytes. *)
  let bytes = Bytes.create 2 in
  blit_to_bytes ~src_off:4 ~len:4 src bytes;
  (* Into elements 2..5 of [dst]. *)
  let dst = create int4 8 in
  blit_from_bytes ~dst_off:2 ~len:4 bytes dst;
  equal ~msg:"int4 offset dst[1]" int 0 (get dst 1);
  equal ~msg:"int4 offset dst[2]" int 4 (get dst 2);
  equal ~msg:"int4 offset dst[5]" int 7 (get dst 5);
  equal ~msg:"int4 offset dst[6]" int 0 (get dst 6)

let test_int4_bytes_blit_odd_raises () =
  let buf = create int4 8 in
  let bytes = Bytes.create 4 in
  raises_match ~msg:"odd src_off to bytes" invalid_argument (fun () ->
      blit_to_bytes ~src_off:1 ~len:2 buf bytes);
  raises_match ~msg:"odd dst_off from bytes" invalid_argument (fun () ->
      blit_from_bytes ~dst_off:1 ~len:2 bytes buf);
  raises_match ~msg:"odd len not reaching the end" invalid_argument (fun () ->
      blit_from_bytes ~len:3 bytes buf);
  (* An odd length is fine when the copy reaches the end of the buffer: the
     trailing nibble is padding. *)
  let tail = create int4 5 in
  blit_from_bytes ~dst_off:2 ~len:3 bytes tail

(* Test bigarray conversions *)
let test_bigarray_roundtrip () =
  let buf = create float32 3 in
  set buf 0 1.0;
  set buf 1 2.0;
  set buf 2 3.0;
  let ba1 = to_bigarray1 buf in
  equal ~msg:"to_bigarray1 dim" int 3 (Bigarray.Array1.dim ba1);
  let buf2 = of_bigarray1 ba1 in
  equal ~msg:"roundtrip[0]" (float 1e-6) 1.0 (get buf2 0);
  equal ~msg:"roundtrip[2]" (float 1e-6) 3.0 (get buf2 2)

let test_genarray_roundtrip () =
  let buf = create float32 6 in
  for i = 0 to 5 do
    set buf i (float_of_int i)
  done;
  let ga = to_genarray buf [| 2; 3 |] in
  equal ~msg:"genarray dims" (array int) [| 2; 3 |] (Bigarray.Genarray.dims ga);
  let buf2 = of_genarray ga in
  equal ~msg:"genarray roundtrip length" int 6 (length buf2);
  equal ~msg:"genarray roundtrip[0]" (float 1e-6) 0.0 (get buf2 0);
  equal ~msg:"genarray roundtrip[5]" (float 1e-6) 5.0 (get buf2 5)

(* Extended kinds have no faithful Bigarray.kind: viewing one as a bigarray
   would let stdlib operations misread it. *)
let test_to_bigarray1_extended_raises () =
  let buf = create bfloat16 4 in
  raises_match ~msg:"to_bigarray1 on bfloat16" invalid_argument (fun () ->
      to_bigarray1 buf)

(* Extended-kind genarrays stay usable through the genarray bridge. *)
let test_genarray_extended_roundtrip () =
  let buf = create bfloat16 6 in
  for i = 0 to 5 do
    set buf i (float_of_int i)
  done;
  let ga = to_genarray buf [| 2; 3 |] in
  equal ~msg:"extended genarray kind" bool true (genarray_kind ga = BFloat16);
  let buf2 = of_genarray ga in
  equal ~msg:"extended genarray roundtrip[5]" (float 1e-6) 5.0 (get buf2 5)

let test_of_bigarray1_unsupported_raises () =
  let ba_int = Bigarray.Array1.create Bigarray.int Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on int" invalid_argument (fun () ->
      of_bigarray1 ba_int);
  let ba_char = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on char" invalid_argument (fun () ->
      of_bigarray1 ba_char);
  let ba_nat = Bigarray.Array1.create Bigarray.nativeint Bigarray.c_layout 4 in
  raises_match ~msg:"of_bigarray1 on nativeint" invalid_argument (fun () ->
      of_bigarray1 ba_nat)

(* Test suite *)
let () =
  run "Nx_buffer tests"
    [
      group "creation"
        [
          test "create bfloat16" test_create_bfloat16;
          test "create bool" test_create_bool;
          test "create int4" test_create_int4;
          test "create uint4" test_create_uint4;
          test "create float8_e4m3" test_create_float8_e4m3;
          test "create float8_e5m2" test_create_float8_e5m2;
        ];
      group "genarray" [ test "genarray creation" test_genarray_creation ];
      group "properties"
        [
          test "kind round-trip" test_kind_roundtrip;
          test "kind sizes" test_kind_sizes;
        ];
      group "semantics"
        [
          test "bfloat16" test_bfloat16_semantics;
          test "float8 e4m3" test_float8_e4m3_semantics;
          test "float8 e5m2" test_float8_e5m2_semantics;
          test "int4 clamping" test_int4_clamping;
          test "uint64 roundtrip" test_uint64_roundtrip;
        ];
      group "operations"
        [
          test "blit" test_blit;
          test "fill" test_fill;
          test "int4 bytes blit roundtrip" test_int4_bytes_blit_roundtrip;
          test "int4 bytes blit offsets" test_int4_bytes_blit_offsets;
          test "int4 bytes blit odd offsets raise"
            test_int4_bytes_blit_odd_raises;
        ];
      group "conversions"
        [
          test "bigarray roundtrip" test_bigarray_roundtrip;
          test "genarray roundtrip" test_genarray_roundtrip;
          test "to_bigarray1 rejects extended kinds"
            test_to_bigarray1_extended_raises;
          test "extended genarray bridge" test_genarray_extended_roundtrip;
          test "of_bigarray1 rejects unsupported kinds"
            test_of_bigarray1_unsupported_raises;
        ];
    ]
