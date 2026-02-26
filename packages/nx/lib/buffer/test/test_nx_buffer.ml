(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_buffer
open Windtrap

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

let test_create_int4_signed () =
  let buf = create int4_signed 16 in
  equal ~msg:"int4_signed buffer size" int 16 (length buf);
  set buf 0 (-8);
  set buf 1 7;
  set buf 2 0;
  equal ~msg:"int4_signed get" int (-8) (get buf 0);
  equal ~msg:"int4_signed get" int 7 (get buf 1);
  equal ~msg:"int4_signed get" int 0 (get buf 2)

let test_create_int4_unsigned () =
  let buf = create int4_unsigned 16 in
  equal ~msg:"int4_unsigned buffer size" int 16 (length buf);
  set buf 0 0;
  set buf 1 15;
  set buf 2 8;
  equal ~msg:"int4_unsigned get" int 0 (get buf 0);
  equal ~msg:"int4_unsigned get" int 15 (get buf 1);
  equal ~msg:"int4_unsigned get" int 8 (get buf 2)

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

(* Test kind_size_in_bytes *)
let test_kind_sizes () =
  equal ~msg:"bfloat16 size" int 2 (kind_size_in_bytes bfloat16);
  equal ~msg:"bool size" int 1 (kind_size_in_bytes Nx_buffer.bool);
  equal ~msg:"int4_signed size" int 1 (kind_size_in_bytes int4_signed);
  equal ~msg:"int4_unsigned size" int 1 (kind_size_in_bytes int4_unsigned);
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

(* Test suite *)
let () =
  run "Nx_buffer tests"
    [
      group "creation"
        [
          test "create bfloat16" test_create_bfloat16;
          test "create bool" test_create_bool;
          test "create int4_signed" test_create_int4_signed;
          test "create int4_unsigned" test_create_int4_unsigned;
          test "create float8_e4m3" test_create_float8_e4m3;
          test "create float8_e5m2" test_create_float8_e5m2;
        ];
      group "genarray" [ test "genarray creation" test_genarray_creation ];
      group "properties" [ test "kind sizes" test_kind_sizes ];
      group "operations" [ test "blit" test_blit; test "fill" test_fill ];
      group "conversions"
        [
          test "bigarray roundtrip" test_bigarray_roundtrip;
          test "genarray roundtrip" test_genarray_roundtrip;
        ];
    ]
