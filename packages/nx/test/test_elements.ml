(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Typed access to host buffers: each dtype's storage representation, fills that
   stay inside their buffer, and gathers that keep every bit. *)

open Windtrap
module B = Nx_device.Buffer
module E = Nx_core.Elements
module S = Nx_dtype.Scalar
module V = Nx_core.View

let buffer dt n = B.create Nx_device.host (S.of_dtype dt) n
let bytes b = B.bigarray Bigarray.int8_unsigned b

let byte_list b =
  let ba = bytes b in
  List.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let of_bytes l =
  B.of_bigarray
    (Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
       (List.length l) (List.nth l))

let inv = Exn.invalid_arg ~substring:""

(* Round trips *)

type case = Case : ('a, 'b) Nx_dtype.t * 'a list -> case

let cases =
  Nx_dtype.
    [
      Case (float16, [ 1.5; -2.; 0.25 ]);
      Case (float32, [ 1.5; -2.; 0x1p-100 ]);
      Case (float64, [ 1.5; -2.; 1e-300 ]);
      Case (bfloat16, [ 1.5; -2.; 256. ]);
      Case (float8_e4m3, [ 1.5; -2.; 448. ]);
      Case (float8_e5m2, [ 1.5; -2.; 57344. ]);
      Case (int4, [ -8; 7; 0; -1; 3 ]);
      Case (uint4, [ 0; 15; 9 ]);
      Case (int8, [ -128; 127; 0 ]);
      Case (uint8, [ 0; 255; 7 ]);
      Case (int16, [ -32768; 32767; 5 ]);
      Case (uint16, [ 0; 65535; 5 ]);
      Case (int32, [ Int32.min_int; Int32.max_int; 5l ]);
      Case (uint32, [ -1l; 0l; 5l ]);
      Case (int64, [ Int64.min_int; Int64.max_int; 5L ]);
      Case (uint64, [ -1L; 0L; 5L ]);
      Case (complex64, [ { Complex.re = 1.5; im = -2. } ]);
      Case (complex128, [ { Complex.re = 1e-300; im = 3. } ]);
      Case (bool, [ true; false; true ]);
    ]

let test_round_trips () =
  List.iter
    (fun (Case (dt, values)) ->
      let n = List.length values in
      let b = buffer dt n in
      let set = E.set dt b and get = E.get dt b in
      List.iteri set values;
      is_true ~msg:(Nx_dtype.to_string dt) (List.init n get = values);
      let f = buffer dt n in
      E.fill dt f (List.hd values);
      is_true
        ~msg:(Nx_dtype.to_string dt ^ " fill")
        (List.init n (E.get dt f) = List.init n (fun _ -> List.hd values)))
    cases

(* Storage *)

let test_storage () =
  let i4 = buffer Nx_dtype.int4 3 in
  List.iteri (E.set Nx_dtype.int4 i4) [ 1; -2; 3 ];
  equal ~msg:"int4, the first in the low nibble" (list int) [ 0xe1; 0x03 ]
    (List.mapi (fun i b -> if i = 1 then b land 0xf else b) (byte_list i4));
  let u4 = buffer Nx_dtype.uint4 2 in
  E.set Nx_dtype.uint4 u4 0 20;
  E.set Nx_dtype.uint4 u4 1 (-3);
  equal ~msg:"uint4 stores clamp" (list int) [ 15; 0 ]
    (List.init 2 (E.get Nx_dtype.uint4 u4));
  E.set Nx_dtype.int4 i4 0 (-9);
  equal ~msg:"int4 stores clamp" int (-8) (E.get Nx_dtype.int4 i4 0);
  let bf = buffer Nx_dtype.bfloat16 1 in
  E.set Nx_dtype.bfloat16 bf 0 1.;
  equal ~msg:"bfloat16 bits" (list int) [ 0x80; 0x3f ] (byte_list bf);
  let u32 = buffer Nx_dtype.uint32 1 in
  E.set Nx_dtype.uint32 u32 0 (-1l);
  equal ~msg:"uint32 bits" (list int) [ 0xff; 0xff; 0xff; 0xff ] (byte_list u32);
  let bo = of_bytes [ 0; 1; 7 ] in
  let bo = B.view bo ~offset:0 S.Bool 3 in
  equal ~msg:"bool reads a nonzero byte as true" (list bool)
    [ false; true; true ]
    (List.init 3 (E.get Nx_dtype.bool bo));
  E.set Nx_dtype.bool bo 2 true;
  equal ~msg:"bool stores 1" (list int) [ 0; 1; 1 ] (byte_list bo)

let test_fill_stays_inside () =
  let mem = of_bytes [ 0x00; 0xa0 ] in
  let three = B.view mem ~offset:0 S.Int4 3 in
  E.fill Nx_dtype.int4 three 5;
  equal ~msg:"the neighbour's nibble is kept" (list int) [ 0x55; 0xa5 ]
    (byte_list mem)

let test_refusals () =
  let b = buffer Nx_dtype.float32 2 in
  raises_match inv (fun () -> E.get Nx_dtype.int32 b);
  raises_match inv (fun () -> E.get Nx_dtype.float32 b 2);
  raises_match inv (fun () -> E.set Nx_dtype.int4 (buffer Nx_dtype.int4 3) 3 0)

(* Gathering *)

let view ?(offset = 0) strides shape = V.create ~offset ~strides shape

let test_gather_words () =
  let src = buffer Nx_dtype.int32 6 in
  List.iteri (E.set Nx_dtype.int32 src) [ 0l; 1l; 2l; 3l; 4l; 5l ];
  let got v = List.init (V.numel v) (E.get Nx_dtype.int32 (E.gather src v)) in
  equal ~msg:"transposed" (list int32) [ 0l; 3l; 1l; 4l; 2l; 5l ]
    (got (view [| 1; 3 |] [| 3; 2 |]));
  equal ~msg:"reversed" (list int32) [ 5l; 4l; 3l ]
    (got (view ~offset:5 [| -1 |] [| 3 |]));
  equal ~msg:"broadcast" (list int32) [ 2l; 2l; 2l; 2l ]
    (got (view ~offset:2 [| 0; 0 |] [| 2; 2 |]));
  equal ~msg:"rows of a slice" (list int32) [ 1l; 2l; 4l; 5l ]
    (got (view ~offset:1 [| 3; 1 |] [| 2; 2 |]));
  equal ~msg:"a scalar" (list int32) [ 4l ] (got (view ~offset:4 [||] [||]));
  equal ~msg:"empty" (list int32) [] (got (view [| 1 |] [| 0 |]));
  raises_match inv (fun () -> E.gather src (view ~offset:4 [| 1 |] [| 3 |]));
  raises_match inv (fun () -> E.gather src (view ~offset:1 [| -1 |] [| 3 |]))

let test_gather_bits () =
  let src = buffer Nx_dtype.float32 2 in
  let bits = B.bigarray Bigarray.int32 src in
  bits.{0} <- 0x7f800001l;
  bits.{1} <- 0xffc00001l;
  let dst = E.gather src (view ~offset:1 [| -1 |] [| 2 |]) in
  let out = B.bigarray Bigarray.int32 dst in
  equal ~msg:"a signalling NaN keeps its bits" (list int32)
    [ 0xffc00001l; 0x7f800001l ]
    [ out.{0}; out.{1} ];
  let c = buffer Nx_dtype.complex128 3 in
  List.iteri
    (E.set Nx_dtype.complex128 c)
    [ Complex.one; Complex.i; { Complex.re = 2.; im = 3. } ];
  let g = E.gather c (view ~offset:2 [| -2 |] [| 2 |]) in
  equal ~msg:"16-byte elements"
    (list (pair float_exact float_exact))
    [ (2., 3.); (1., 0.) ]
    (List.init 2 (fun i ->
         let z = E.get Nx_dtype.complex128 g i in
         (z.Complex.re, z.im)))

let test_gather_nibbles () =
  let src = buffer Nx_dtype.int4 7 in
  List.iteri (E.set Nx_dtype.int4 src) [ 0; -1; 2; -3; 4; -5; 6 ];
  let g = E.gather src (view ~offset:1 [| 2 |] [| 3 |]) in
  equal ~msg:"odd offsets" (list int) [ -1; -3; -5 ]
    (List.init 3 (E.get Nx_dtype.int4 g));
  let r = E.gather src (view ~offset:6 [| -1 |] [| 7 |]) in
  equal ~msg:"reversed" (list int) [ 6; -5; 4; -3; 2; -1; 0 ]
    (List.init 7 (E.get Nx_dtype.int4 r))

let () =
  exit
    (run "Nx_core.Elements"
       [
         group "access"
           [
             test "every dtype round trips" test_round_trips;
             test "storage representations" test_storage;
             test "a fill stays inside its buffer" test_fill_stays_inside;
             test "refusals" test_refusals;
           ];
         group "gather"
           [
             test "strided views of words" test_gather_words;
             test "bits are kept" test_gather_bits;
             test "4-bit elements" test_gather_nibbles;
           ];
       ])
