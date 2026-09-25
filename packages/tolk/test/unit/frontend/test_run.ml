(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let bufferized_call sink =
  let sink, map = Tolk.Bufferize.run sink in
  Tolk.Callify.transform_to_call sink, map


(* Numeric end-to-end tests on the process-wide default device (DEV selects
   the backend), with expectations derived from tinygrad. *)

open Windtrap
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module El = Tolk_frontend.Elementwise
module Rd = Tolk_frontend.Reduce
module Op = Tolk_frontend.Op
module Dt = Tolk_frontend.Dtype_ops
module Creation = Tolk_frontend.Creation
module Run = Tolk_frontend.Run
module U = Tolk_uop.Uop

let fa ~shape data = Run.of_float_array ~shape data
let vec data = Run.of_float_array ~shape:[ Array.length data ] data

let close a b = Float.abs (a -. b) < 1e-4

let check_floats expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if not (close e got.(i)) then
        failf "element %d: expected %g, got %g" i e got.(i))
    expected

let check_ints expected t =
  equal (array int) expected (Run.to_int_array t)

let count_kernels t =
  let sink = U.sink [ U.contiguous ~src:(T.uop t) () ] in
  List.length
    (List.filter
       (fun u -> U.op u = Tolk_uop.Ops.Call)
       (U.toposort (Tolk.Rangeify.get_kernel_graph sink)))

let elementwise_tests =
  group "elementwise"
    [
      test "add" (fun () ->
          check_floats [| 11.; 22.; 33. |]
            (El.add (vec [| 1.; 2.; 3. |]) (vec [| 10.; 20.; 30. |])));
      test "mul then relu" (fun () ->
          check_floats [| 0.; 2.; 0.; 4. |]
            (El.relu (vec [| -1.; 2.; -3.; 4. |])));
      test "exp" (fun () ->
          check_floats [| 1.; 2.718282; 7.389056 |] (El.exp (vec [| 0.; 1.; 2. |])));
      test "pow" (fun () ->
          check_floats [| 4.; 9. |] (El.pow (vec [| 2.; 3. |]) (T.f 2.0)));
      test "broadcast add scalar" (fun () ->
          check_floats [| 6.; 7.; 8. |] (El.add (vec [| 1.; 2.; 3. |]) (T.f 5.0)));
      test "contiguous preserves values" (fun () ->
          check_floats [| 2.; 4.; 6. |]
            (El.contiguous (El.add (vec [| 1.; 2.; 3. |]) (vec [| 1.; 2.; 3. |]))));
    ]

let bitcast_tests =
  let module D = Tolk_uop.Dtype in
  let bytes = Bytes.init 16 (fun i -> Char.chr ((i * 37 + 129) land 255)) in
  group "bitcast"
    [
      test "size-changing bitcasts preserve every byte" (fun () ->
          let dtypes =
            [ D.uint8; D.uint16; D.uint32; D.uint64; D.int32; D.float32 ]
          in
          List.iter
            (fun src_dtype ->
              List.iter
                (fun dst_dtype ->
                  let source =
                    Run.of_bytes ~dtype:src_dtype
                      ~shape:[ 2; 8 / D.itemsize src_dtype ] bytes
                  in
                  let result = Dt.bitcast source dst_dtype in
                  equal (list int) [ 2; 8 / D.itemsize dst_dtype ] (T.shape result);
                  equal string (Bytes.to_string bytes)
                    (Bytes.to_string (Run.data result)))
                dtypes)
            dtypes);
      test "size-changing bitcast follows noncontiguous element order" (fun () ->
          let source = Run.of_int_array ~shape:[ 2; 2 ] [| 1; 2; 3; 4 |] in
          let result = Dt.bitcast (Mv.permute source [ 1; 0 ]) D.uint8 in
          let expected = Bytes.make 16 '\000' in
          List.iteri
            (fun i n -> Bytes.set_int32_le expected (4 * i) (Int32.of_int n))
            [ 1; 3; 2; 4 ];
          equal (list int) [ 2; 8 ] (T.shape result);
          equal string (Bytes.to_string expected)
            (Bytes.to_string (Run.data result)));
      test "subword slices can be repacked at an unaligned offset" (fun () ->
          let source =
            Run.of_int_array ~shape:[ 2 ] [| 0x04030201; 0x08070605 |]
          in
          let part = Mv.shrink (Dt.bitcast source D.uint8) [ (1, 3) ] in
          let result = Dt.bitcast part D.uint16 in
          equal (list int) [ 1 ] (T.shape result);
          equal string "\002\003" (Bytes.to_string (Run.data result)));
      test "size-changing bitcast supports an empty last axis" (fun () ->
          let source = Run.of_bytes ~dtype:D.uint32 ~shape:[ 2; 0 ] Bytes.empty in
          let result = Dt.bitcast source D.uint8 in
          equal (list int) [ 2; 0 ] (T.shape result);
          equal string "" (Bytes.to_string (Run.data result)));
      test "size-changing bitcast rejects an incomplete destination element" (fun () ->
          let source =
            Run.of_bytes ~dtype:D.uint8 ~shape:[ 3 ] (Bytes.make 3 '\000')
          in
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () -> T.shape (Dt.bitcast source D.uint32)));
    ]

let reduce_tests =
  group "reduce"
    [
      test "padding an empty slice preserves its rank" (fun () ->
          let empty = Mv.shrink (vec [| 7. |]) [ (0, 0) ] in
          check_floats [| 0.; 0.; 0. |] (Mv.pad empty [ (1, 2) ]));
      test "reducing an empty axis preserves the remaining axes" (fun () ->
          let empty = Mv.shrink (fa ~shape:[ 1; 2 ] [| 7.; 8. |])
              [ (0, 0); (0, 2) ] in
          check_floats [| 0.; 0.; 0.; 0. |]
            (Mv.pad (Rd.sum ~axis:[ 0 ] empty) [ (1, 1) ]));
      test "weak reductions preserve values beyond int32" (fun () ->
          let large = 1 lsl 40 in
          let input = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 3 ] in
          let check expected tensor =
            let bytes = Run.data tensor in
            equal int 8 (Bytes.length bytes);
            equal int64 (Int64.of_int expected) (Bytes.get_int64_le bytes 0)
          in
          check (3 * large) (Rd.sum input);
          check large (Rd.max input);
          check large (Rd.prod (Mv.reshape (T.i large) [ 1 ])));
      test "weak integer mean preserves values beyond int32" (fun () ->
          let large = 1 lsl 40 in
          let input = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 3 ] in
          check_floats [| float_of_int large |] (Op.mean input));
      test "sum all" (fun () ->
          check_floats [| 21. |] (Rd.sum (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "sum axis 0" (fun () ->
          check_floats [| 5.; 7.; 9. |]
            (Rd.sum ~axis:[ 0 ] (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "mean" (fun () ->
          check_floats [| 2.5 |] (Op.mean (vec [| 1.; 2.; 3.; 4. |])));
      test "max" (fun () -> check_floats [| 5. |] (Rd.max (vec [| 1.; 5.; 3. |])));
      test "argmax and argmin take the first of equal extremes" (fun () ->
          check_ints [| 1 |] (Op.argmax (vec [| 1.; 3.; 3.; 0. |]));
          check_ints [| 0 |] (Op.argmin (vec [| 0.; 3.; 3.; 0. |])));
    ]

let matmul_tests =
  group "matmul"
    [
      test "float matmul" (fun () ->
          check_floats [| 4.; 5.; 10.; 11. |]
            (Op.matmul
               (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])
               (fa ~shape:[ 3; 2 ] [| 1.; 0.; 0.; 1.; 1.; 1. |])));
      test "int matmul" (fun () ->
          check_ints [| 1; 2; 3; 4 |]
            (Op.matmul
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 2; 3; 4 |])
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 0; 1 |])));
    ]

let scan_tests =
  (* Axes longer than 512 scan in chunks of 256; the lengths straddle that
     threshold and leave a partial last chunk. *)
  let values n = Array.init n (fun i -> ((i * 7) + 3) mod 11 - 5) in
  let prefix f xs =
    let acc = ref None in
    Array.map
      (fun x ->
        let y = match !acc with None -> x | Some a -> f a x in
        acc := Some y;
        y)
      xs
  in
  let floats xs = Array.map float_of_int xs in
  let param n =
    T.of_uop
      (U.param ~slot:0 ~dtype:Tolk_uop.Dtype.float32 ~shape:(T.shape_uop [ n ])
         ~device:(U.Single "CPU") ())
  in
  group "scan"
    [
      test "cumsum 1d" (fun () ->
          check_floats [| 1.; 3.; 6.; 10. |] (Op.cumsum (vec [| 1.; 2.; 3.; 4. |])));
      test "cumsum 2d axis 1" (fun () ->
          check_floats [| 1.; 3.; 6.; 4.; 9.; 15. |]
            (Op.cumsum ~axis:1 (fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |])));
      test "long cumsum around the split threshold" (fun () ->
          List.iter
            (fun n ->
              let xs = values n in
              check_floats (floats (prefix ( + ) xs)) (Op.cumsum (vec (floats xs)));
              check_ints (prefix ( + ) xs)
                (Op.cumsum (Run.of_int_array ~shape:[ n ] xs)))
            [ 512; 513; 1000; 1024; 3000 ]);
      test "long cumsum of int8 accumulates in int32" (fun () ->
          let xs = Array.make 1000 100 in
          check_ints (prefix ( + ) xs)
            (Op.cumsum (Dt.cast (Run.of_int_array ~shape:[ 1000 ] xs) Tolk_uop.Dtype.int8)));
      test "long cumsum along a leading axis" (fun () ->
          let n = 700 in
          let xs = values (n * 3) in
          let column c = Array.init n (fun i -> xs.((i * 3) + c)) in
          let sums = Array.map (fun c -> prefix ( + ) (column c)) [| 0; 1; 2 |] in
          check_ints
            (Array.init (n * 3) (fun k -> sums.(k mod 3).(k / 3)))
            (Op.cumsum ~axis:0 (Run.of_int_array ~shape:[ n; 3 ] xs)));
      test "long cumprod" (fun () ->
          let xs = Array.init 600 (fun i -> if i mod 97 = 0 then -1 else 1) in
          check_ints (prefix ( * ) xs) (Op.cumprod (Run.of_int_array ~shape:[ 600 ] xs)));
      test "long cumprod of 8-bit floats" (fun () ->
          (* The chunked scan of an emulated 8-bit float failed in the C and
             Metal compilers. *)
          let xs = Array.init 600 (fun i -> if i mod 97 = 0 then -1 else 1) in
          List.iter
            (fun dtype ->
              check_floats (floats (prefix ( * ) xs))
                (Dt.cast (Op.cumprod (Dt.cast (vec (floats xs)) dtype))
                   Tolk_uop.Dtype.float32))
            [ Tolk_uop.Dtype.fp8e4m3; Tolk_uop.Dtype.fp8e5m2 ]);
      test "long cummax" (fun () ->
          let xs = Array.init 1000 (fun i -> ((i * 37) mod 1009) - (i mod 3)) in
          let values, indices = Op.cummax (Run.of_int_array ~shape:[ 1000 ] xs) in
          check_ints (prefix max xs) values;
          let best = ref 0 in
          check_ints
            (Array.mapi
               (fun i x ->
                 if x > xs.(!best) then best := i;
                 !best)
               xs)
            indices);
      test "long scans split into chunk, total and combine kernels" (fun () ->
          equal int 1 (count_kernels (Op.cumsum (param 512)));
          equal int 3 (count_kernels (Op.cumsum (param 513)));
          equal int 3 (count_kernels (Op.cumprod (param 1000)));
          equal int 3 (count_kernels (fst (Op.cummax (param 1000)))));
    ]

let logspace_tests =
  group "logspace"
    [
      test "softmax" (fun () ->
          check_floats [| 0.0900306; 0.2447284; 0.6652409 |]
            (Op.softmax (vec [| 1.; 2.; 3. |])));
      test "logsumexp" (fun () ->
          check_floats [| 3.4076059 |] (Op.logsumexp (vec [| 1.; 2.; 3. |])));
      (* A sink is a key of value zero with no column: its logit joins the
         shift and the normaliser. It is finite, so a row with every key masked
         divides 0 by 1. *)
      test "a sink keeps a fully masked softmax row at zero" (fun () ->
          let scores = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 1.; 2.; 3. |] in
          let seen =
            El.gt (fa ~shape:[ 2; 3 ] [| 1.; 1.; 1.; 0.; 0.; 0. |]) (T.f 0.5)
          in
          let sink = fa ~shape:[ 1; 1 ] [| 0. |] in
          let scores = El.where seen scores (T.f Float.neg_infinity) in
          let top = El.maximum (Rd.max ~axis:[ 1 ] ~keepdim:true scores) sink in
          let e = El.exp (El.sub scores top) in
          let total =
            El.add (Rd.sum ~axis:[ 1 ] ~keepdim:true e) (El.exp (El.sub sink top))
          in
          check_floats
            [| 0.087144; 0.236883; 0.643914; 0.; 0.; 0. |]
            (El.div e total));
    ]

let getitem_tests =
  let base () = fa ~shape:[ 3; 4 ] (Array.init 12 float_of_int) in
  group "getitem"
    [
      test "slice" (fun () ->
          check_floats [| 5.; 6.; 9.; 10. |]
            (Op.getitem (base ())
               [ Mv.R (Some 1, Some 3, None); Mv.R (Some 1, Some 3, None) ]));
      test "int index" (fun () ->
          check_floats [| 4.; 5.; 6.; 7. |] (Op.getitem (base ()) [ Mv.I 1 ]));
      test "strided" (fun () ->
          check_floats [| 0.; 2. |]
            (Op.getitem (base ()) [ Mv.I 0; Mv.R (None, None, Some 2) ]));
      test "tensor index" (fun () ->
          check_floats [| 8.; 9.; 10.; 11.; 0.; 1.; 2.; 3. |]
            (Op.getitem (base ()) [ Mv.T (Run.of_int_array ~shape:[ 2 ] [| 2; 0 |]) ]));
      test "non-consecutive tensor indices select their elements" (fun () ->
          (* x[i, :, j] reads x[2, :, 0], both -0, beside a NaN in a row it
             does not read; a sum of masked elements gives +0, or NaN when a
             masked NaN is multiplied by zero. *)
          let module D = Tolk_uop.Dtype in
          let bytes = Bytes.create (4 * 256 * 2 * 256) in
          for k = 0 to (256 * 2 * 256) - 1 do
            Bytes.set_int32_le bytes (4 * k)
              (if k mod 256 = 0 then 0x80000000l else 0x3F800000l)
          done;
          Bytes.set_int32_le bytes (4 * 3 * 2 * 256) 0x7FC00000l;
          let x = Run.of_bytes ~dtype:D.float32 ~shape:[ 256; 2; 256 ] bytes in
          let at v = Run.of_int_array ~shape:[ 1 ] [| v |] in
          let got = Run.data (Op.getitem x [ Mv.T (at 2); Mv.All; Mv.T (at 0) ]) in
          equal (array int32) [| 0x80000000l; 0x80000000l |]
            (Array.init 2 (fun k -> Bytes.get_int32_le got (4 * k))));
    ]

(* The lowered kernel of [t] loads int32 indices, each under a gate. *)
let check_index_loads_gated t =
  let graph =
    Tolk.Rangeify.get_kernel_graph (U.sink [ U.contiguous ~src:(T.uop t) () ])
  in
  let kernel =
    List.find_map
      (fun u ->
        match U.as_call u with Some { body; _ } -> Some body | None -> None)
      (U.toposort graph)
    |> Option.get
  in
  let program =
    Tolk.Linearizer.linearize
      (Tolk.Codegen.full_rewrite_to_sink
         (Tolk.Cstyle.clang_no_abi Tolk.Gpu_target.X86_64)
         kernel)
  in
  let index_loads =
    List.filter_map
      (fun u ->
        match U.as_load u with
        | Some { gate; _ }
          when Tolk_uop.Dtype.equal (U.dtype u) Tolk_uop.Dtype.int32 ->
            Some (Option.is_some gate)
        | _ -> None)
      program
  in
  is_true ~msg:"the kernel loads the indices" (index_loads <> []);
  is_true ~msg:"every index load is gated" (List.for_all Fun.id index_loads)

let param slot dtype dims =
  T.of_uop
    (U.param ~slot ~dtype ~shape:(T.shape_uop dims) ~device:(U.Single "CPU") ())

let gather_params () =
  (param 0 Tolk_uop.Dtype.float32 [ 100 ], param 1 Tolk_uop.Dtype.int32 [ 50 ])

let large_gather_tests =
  let rows = 65_536 in
  group "large gather"
    [
      test "gather over 65536 rows schedules to one kernel" (fun () ->
          let table = param 0 Tolk_uop.Dtype.float32 [ rows; 4 ] in
          let index = param 1 Tolk_uop.Dtype.int32 [ 8; 4 ] in
          equal int 1 (count_kernels (Op.gather table ~dim:0 index)));
      test "gather of a comparison on a narrowed int64" (fun () ->
          (* Reduce collapse lifted the subtraction out of the comparison and
             back in again until it detected a rewrite cycle. *)
          let x =
            El.contiguous
              (Dt.cast (Run.of_int_array ~shape:[ 2 ] [| 5; 3 |]) Tolk_uop.Dtype.int64)
          in
          let s =
            Dt.cast
              (El.sub x (Creation.const_like x (T.Sint (1 lsl 31))))
              Tolk_uop.Dtype.int32
          in
          check_floats [| 0. |]
            (Op.gather
               (Dt.cast (El.eq s (Creation.const_like s (T.Sint 7))) Tolk_uop.Dtype.float32)
               ~dim:0
               (Run.of_int_array ~shape:[ 1 ] [| 1 |])));
      test "gather over 65536 rows reads the indexed rows" (fun () ->
          let table =
            fa ~shape:[ rows; 2 ]
              (Array.init (rows * 2) (fun i -> float_of_int (i mod 1000)))
          in
          let index =
            Run.of_int_array ~shape:[ 3; 2 ]
              [| 0; 65_535; 40_000; 1; 123; 32_768 |]
          in
          check_floats [| 0.; 71.; 0.; 3.; 246.; 537. |]
            (Op.gather table ~dim:0 index));
      (* The index load sits in the address of the gathered load, and both
         are gated by where the gather lands. Simplified under the gathered
         load's gate, the index load lost its own and read the indices
         outside: a sort's compiled join of values segfaulted there. *)
      test "a padded gather loads its indices under the pad's gate" (fun () ->
          let table, index = gather_params () in
          check_index_loads_gated
            (Mv.pad (Op.gather table ~dim:0 index) [ (10, 10) ]));
      test "a gather between other pieces loads its indices under their gate"
        (fun () ->
          let table, index = gather_params () in
          let a = param 2 Tolk_uop.Dtype.float32 [ 30 ] in
          check_index_loads_gated
            (Op.cat a [ Op.gather table ~dim:0 index; a ]));
    ]

let conv_tests =
  group "conv"
    [
      test "conv2d 3x3 with 2x2 kernel" (fun () ->
          let x = fa ~shape:[ 1; 1; 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))) in
          let w = fa ~shape:[ 1; 1; 2; 2 ] [| 1.; 0.; 0.; 1. |] in
          check_floats [| 6.; 8.; 12.; 14. |] (Op.conv2d x w));
    ]

let select_tests =
  let mask ~shape bits = Dt.bool (Run.of_int_array ~shape bits) in
  group "select"
    [
      test "masked_select 2d with fill" (fun () ->
          check_floats [| 0.; 2.; 4.; 8.; -1.; -1. |]
            (Op.masked_select ~fill_value:(T.Sfloat (-1.))
               (fa ~shape:[ 3; 3 ] (Array.init 9 float_of_int))
               (mask ~shape:[ 3; 3 ] [| 1; 0; 1; 0; 1; 0; 0; 0; 1 |])
               ~size:6));
      test "masked_select 1d" (fun () ->
          check_floats [| 1.; 3.; 5. |]
            (Op.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])
               ~size:3));
      test "masked_select truncates overflow" (fun () ->
          check_floats [| 1.; 3. |]
            (Op.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])
               ~size:2));
      test "nonzero 1d" (fun () ->
          check_ints [| 0; 2; 4 |]
            (Op.nonzero ~fill_value:(T.Sint (-1))
               (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 2; 0; 3 |])
               ~size:3));
      test "nonzero 2d" (fun () ->
          check_ints [| 0; 0; 1; 1 |]
            (Op.nonzero
               (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 0; 2 |])
               ~size:2));
      test "nonzero pads with fill" (fun () ->
          check_ints [| 0; 4; -1; -1 |]
            (Op.nonzero ~fill_value:(T.Sint (-1))
               (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 0; 0; 5 |])
               ~size:4));
    ]

let dynamic_select_tests =
  let mask ~shape bits = Dt.bool (Run.of_int_array ~shape bits) in
  group "dynamic_select"
    [
      test "masked_select dynamic size" (fun () ->
          check_floats [| 1.; 3.; 5. |]
            (Run.masked_select
               (vec [| 1.; 2.; 3.; 4.; 5. |])
               (mask ~shape:[ 5 ] [| 1; 0; 1; 0; 1 |])));
      test "nonzero dynamic size" (fun () ->
          check_ints [| 0; 2; 4 |]
            (Run.nonzero (Run.of_int_array ~shape:[ 5 ] [| 1; 0; 2; 0; 3 |])));
      test "list-style advanced index" (fun () ->
          check_floats [| 8.; 9.; 10.; 11.; 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |]
            (Op.getitem
               (fa ~shape:[ 3; 4 ] (Array.init 12 float_of_int))
               [ Mv.T (Run.of_int_array ~shape:[ 3 ] [| 2; 0; 1 |]) ]));
    ]

let scatter_tests =
  let fi ~shape data = Run.of_int_array ~shape data in
  let src10 = fa ~shape:[ 2; 5 ] (Array.init 10 (fun i -> float_of_int (i + 1))) in
  let zeros35 = fa ~shape:[ 3; 5 ] (Array.make 15 0.) in
  let idx0 () = fi ~shape:[ 2; 5 ] (Array.make 10 0) in
  group "scatter"
    [
      test "scatter along dim 0" (fun () ->
          check_floats
            [| 1.; 0.; 0.; 4.; 0.; 0.; 2.; 0.; 0.; 0.; 0.; 0.; 3.; 0.; 0. |]
            (Op.scatter zeros35 ~dim:0 (fi ~shape:[ 1; 4 ] [| 0; 1; 2; 0 |]) src10));
      test "scatter along dim 1" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 0.; 0.; 6.; 7.; 0.; 0.; 8.; 0.; 0.; 0.; 0.; 0. |]
            (Op.scatter zeros35 ~dim:1
               (fi ~shape:[ 2; 3 ] [| 0; 1; 2; 0; 1; 4 |])
               (fa ~shape:[ 2; 5 ] (Array.init 10 (fun i -> float_of_int (i + 1))))));
      test "scatter_reduce sum" (fun () ->
          check_floats [| 8.; 10.; 12.; 14.; 16. |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Sum ()));
      test "scatter_reduce prod" (fun () ->
          check_floats [| 6.; 14.; 24.; 36.; 50. |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Prod ()));
      test "scatter_reduce amax" (fun () ->
          check_floats [| 6.; 20.; 8.; 9.; 10. |]
            (Op.scatter_reduce
               (fa ~shape:[ 1; 5 ] [| -10.; 20.; 0.; 5.; 10. |])
               ~dim:0 (idx0 ()) src10 ~reduce:`Amax ()));
      test "scatter_reduce amin" (fun () ->
          check_floats [| -10.; 2.; 0.; 4.; 5. |]
            (Op.scatter_reduce
               (fa ~shape:[ 1; 5 ] [| -10.; 20.; 0.; 5.; 10. |])
               ~dim:0 (idx0 ()) src10 ~reduce:`Amin ()));
      test "scatter_reduce mean excluding self" (fun () ->
          check_floats [| 3.5; 4.5; 5.5; 6.5; 7.5 |]
            (Op.scatter_reduce (fa ~shape:[ 1; 5 ] (Array.make 5 1.)) ~dim:0
               (idx0 ()) src10 ~reduce:`Mean ~include_self:false ()));
    ]

let scatter_indexed_tests =
  let fi ~shape data = Run.of_int_array ~shape data in
  let iota shape =
    let n = List.fold_left ( * ) 1 shape in
    fa ~shape (Array.init n (fun i -> float_of_int (i + 1)))
  in
  let zeros shape = fa ~shape (Array.make (List.fold_left ( * ) 1 shape) 0.) in
  let indexed ?(unique = false) mode t ~dim index src =
    let device = List.find_map T.device [ t; index; src ] in
    Op.scatter_indexed (Creation.clone ?device t) ~dim index src ~mode ~unique
  in
  let same_as_reference name ~dim t index src =
    [
      test (name ^ ", set") (fun () ->
          check_floats
            (Run.to_float_array (Op.scatter (t ()) ~dim (index ()) (src ())))
            (indexed `Set (t ()) ~dim (index ()) (src ())));
      test (name ^ ", add") (fun () ->
          check_floats
            (Run.to_float_array
               (Op.scatter_reduce (t ()) ~dim (index ()) (src ()) ~reduce:`Sum
                  ()))
            (indexed `Add (t ()) ~dim (index ()) (src ())));
    ]
  in
  group "scatter_indexed"
    (same_as_reference "rows with a repeated index" ~dim:0
       (fun () -> iota [ 4; 3 ])
       (fun () -> fi ~shape:[ 3; 3 ] [| 2; 0; 1; 2; 3; 1; 0; 0; 1 |])
       (fun () -> iota [ 3; 3 ])
    @ same_as_reference "every update in one lane aims at one cell" ~dim:1
        (fun () -> iota [ 2; 4 ])
        (fun () -> fi ~shape:[ 2; 3 ] [| 1; 1; 1; 3; 3; 3 |])
        (fun () -> iota [ 2; 3 ])
    @ same_as_reference "middle axis" ~dim:1
        (fun () -> iota [ 2; 4; 3 ])
        (fun () ->
          fi ~shape:[ 2; 2; 3 ] [| 3; 0; 1; 3; 2; 1; 0; 0; 0; 1; 2; 3 |])
        (fun () -> iota [ 2; 2; 3 ])
    @ [
        test "an index outside the axis writes nothing" (fun () ->
            let index () = fi ~shape:[ 4; 1 ] [| -1; 4; 1; -7 |] in
            check_floats [| 0.; 0.; 5.; 6.; 0.; 0.; 0.; 0. |]
              (indexed `Set (zeros [ 4; 2 ]) ~dim:0 (index ()) (iota [ 4; 2 ]));
            check_floats [| 1.; 2.; 8.; 10.; 5.; 6.; 7.; 8. |]
              (indexed `Add (iota [ 4; 2 ]) ~dim:0 (index ()) (iota [ 4; 2 ])));
        test "an index outside the axis writes nothing beside unit axes"
          (fun () ->
            (* When no range of the destination's address meets the updates,
               one update, or a scatter axis of extent 1, the bound on the
               loaded index is its store's only gate. *)
            let devices =
              U.Single "CPU"
              :: Option.to_list (T.device (fi ~shape:[ 1 ] [| 0 |]))
              |> List.sort_uniq compare
            in
            let column = [ 3; 1 ] and one = [ 1; 1 ] and row = [ 1; 3 ] in
            let cases =
              [
                (column, [ 1; 1 ], [| -1 |], None, [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
                (column, [ 1; 1 ], [| 3 |], None, [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
                (column, [ 2; 1 ], [| -1; 1 |], None, [| 1.; 8.; 3. |], [| 1.; 10.; 3. |]);
                (column, [ 2; 1 ], [| 1; 3 |], None, [| 1.; 9.; 3. |], [| 1.; 11.; 3. |]);
                (one, [ 2; 1 ], [| -1; -1 |], None, [| 1. |], [| 1. |]);
                (row, [ 2; 1 ], [| -1; -1 |], Some [ 2; 3 ], [| 1.; 2.; 3. |], [| 1.; 2.; 3. |]);
              ]
            in
            List.iter
              (fun device ->
                let on t = Creation.clone ~device t in
                List.iter
                  (fun (dest, ishape, ids, expanded, set, add) ->
                    let index () =
                      let i = on (fi ~shape:ishape ids) in
                      match expanded with None -> i | Some sh -> Mv.expand i sh
                    in
                    let sshape = Option.value expanded ~default:ishape in
                    let n = List.fold_left ( * ) 1 sshape in
                    let src () =
                      on (fa ~shape:sshape (Array.init n (fun j -> 9. -. float_of_int j)))
                    in
                    List.iter
                      (fun unique ->
                        check_floats set
                          (indexed ~unique `Set (on (iota dest)) ~dim:0
                             (index ()) (src ()));
                        check_floats add
                          (indexed ~unique `Add (on (iota dest)) ~dim:0
                             (index ()) (src ())))
                      [ false; true ])
                  cases)
              devices);
        test "an index broadcast off the axis is read once per row" (fun () ->
            let index = fi ~shape:[ 3; 1 ] [| 2; 0; 2 |] in
            let broadcast = Mv.expand index [ 3; 2 ] in
            check_floats [| 3.; 4.; 0.; 0.; 5.; 6.; 0.; 0. |]
              (indexed `Set (zeros [ 4; 2 ]) ~dim:0 broadcast (iota [ 3; 2 ]));
            check_floats [| 3.; 4.; 0.; 0.; 6.; 8.; 0.; 0. |]
              (indexed `Add (zeros [ 4; 2 ]) ~dim:0 broadcast (iota [ 3; 2 ])));
        test "unique indices" (fun () ->
            check_floats [| 0.; 0.; 1.; 2.; 0.; 0.; 3.; 4. |]
              (indexed ~unique:true `Set (zeros [ 4; 2 ]) ~dim:0
                 (fi ~shape:[ 2; 1 ] [| 1; 3 |])
                 (iota [ 2; 2 ])));
        test "unique indices broken at one row leave every other row exact"
          (fun () ->
            let got =
              Run.to_float_array
                (indexed ~unique:true `Set (zeros [ 4; 3 ]) ~dim:0
                   (fi ~shape:[ 4; 1 ] [| 1; 3; 0; 3 |])
                   (iota [ 4; 3 ]))
            in
            let expected = [| 7.; 8.; 9.; 1.; 2.; 3.; 0.; 0.; 0. |] in
            Array.iteri
              (fun i e ->
                is_true ~msg:(Printf.sprintf "element %d" i) (close e got.(i)))
              expected;
            for j = 0 to 2 do
              let v = got.(9 + j) in
              is_true
                ~msg:(Printf.sprintf "the repeated row holds one update at %d" j)
                (close v (float_of_int (4 + j)) || close v (float_of_int (10 + j)))
            done);
        test "a cloned destination keeps its value" (fun () ->
            let t = iota [ 3 ] in
            check_floats [| 1.; 9.; 3. |]
              (indexed `Set t ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                 (fa ~shape:[ 1 ] [| 9. |]));
            check_floats [| 1.; 2.; 3. |] t);
        test "the write lands in the destination's storage" (fun () ->
            let t = iota [ 3 ] in
            let written =
              Run.realize
                (Op.scatter_indexed t ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                   (fa ~shape:[ 1 ] [| 9. |])
                   ~mode:`Set ~unique:false)
            in
            check_floats [| 1.; 9.; 3. |] written;
            check_floats [| 1.; 9.; 3. |] t);
        test "a destination that is not storage is refused" (fun () ->
            raises (Invalid_argument "Op.scatter_indexed: self must be storage")
              (fun () ->
                Op.scatter_indexed
                  (El.add (iota [ 3 ]) (iota [ 3 ]))
                  ~dim:0 (fi ~shape:[ 1 ] [| 1 |])
                  (fa ~shape:[ 1 ] [| 9. |])
                  ~mode:`Set ~unique:false));
        test "constant destination and source" (fun () ->
            check_floats [| 0.; 0.; 2.5; 2.5; 0.; 0. |]
              (indexed `Add
                 (Creation.full ~buffer:false [ 3; 2 ] (T.Sfloat 0.0))
                 ~dim:0
                 (fi ~shape:[ 1; 2 ] [| 1; 1 |])
                 (Creation.full ~buffer:false [ 1; 2 ] (T.Sfloat 2.5))));
        test "a source that is a view of the destination" (fun () ->
            let x = iota [ 2; 3 ] in
            check_floats [| 2.; 2.; 1.; 4.; 5.; 6. |]
              (indexed `Set x ~dim:1
                 (fi ~shape:[ 2; 2 ] [| 2; 0; 1; 1 |])
                 (Mv.shrink x [ (0, 2); (0, 2) ])));
        test "int payload" (fun () ->
            check_ints [| 7; 0; 12 |]
              (indexed `Add
                 (Run.of_int_array ~shape:[ 3 ] [| 0; 0; 0 |])
                 ~dim:0
                 (fi ~shape:[ 3 ] [| 2; 0; 2 |])
                 (Run.of_int_array ~shape:[ 3 ] [| 5; 7; 7 |])));
        test "2048 updates schedule to a fill and a write" (fun () ->
            let param slot dtype dims =
              T.of_uop
                (U.param ~slot ~dtype ~shape:(T.shape_uop dims)
                   ~device:(U.Single "CPU") ())
            in
            let t = param 0 Tolk_uop.Dtype.float32 [ 4096; 8 ] in
            let index = param 1 Tolk_uop.Dtype.int32 [ 2048; 8 ] in
            let src = param 2 Tolk_uop.Dtype.float32 [ 2048; 8 ] in
            let written = T.uop (indexed `Set t ~dim:0 index src) in
            let graph = Tolk.Rangeify.get_kernel_graph (U.sink [ written ]) in
            let calls =
              List.filter
                (fun u -> U.op u = Tolk_uop.Ops.Call)
                (U.toposort graph)
            in
            equal int 2 (List.length calls));
        test "only a unique scatter reaches the optimizer" (fun () ->
            let param slot dtype dims =
              T.of_uop
                (U.param ~slot ~dtype ~shape:(T.shape_uop dims)
                   ~device:(U.Single "CPU") ())
            in
            let applied ~unique =
              let t = param 0 Tolk_uop.Dtype.float32 [ 4096; 8 ] in
              let index = param 1 Tolk_uop.Dtype.int32 [ 2048; 8 ] in
              let src = param 2 Tolk_uop.Dtype.float32 [ 2048; 8 ] in
              let written = T.uop (indexed ~unique `Set t ~dim:0 index src) in
              let graph = Tolk.Rangeify.get_kernel_graph (U.sink [ written ]) in
              let scatter =
                List.find_map
                  (fun u ->
                    match U.as_call u with
                    | Some { body; _ } -> (
                        match U.as_kernel_info body with
                        | Some ki
                          when String.starts_with ~prefix:"scatter_" ki.name ->
                            Some body
                        | _ -> None)
                    | None -> None)
                  (U.toposort graph)
                |> Option.get
              in
              let gpu =
                Tolk.Renderer.make ~name:"test" ~device:"TEST" ~has_local:true
                  ~has_shared:true ~shared_max:32768
                  ~render:(fun ?name:_ _ -> "")
                  ()
              in
              let optimized =
                Tolk.Postrange.apply_opts
                  ~hand_coded_optimizations:Tolk.Heuristic.hand_coded_optimizations
                  scatter gpu
              in
              (Option.get (U.as_kernel_info optimized)).applied_opts
            in
            is_true ~msg:"a unique scatter is laid out"
              (applied ~unique:true <> []);
            equal int 0
              ~msg:"a scatter without the promise keeps its kernel as built"
              (List.length (applied ~unique:false)));
      ])

let clone_tests =
  let param device =
    T.of_uop
      (U.param ~slot:0 ~dtype:Tolk_uop.Dtype.float32 ~shape:(T.shape_uop [ 4 ])
         ~device:(U.Single device) ())
  in
  let copies t =
    List.length
      (List.filter
         (fun u -> U.op u = Tolk_uop.Ops.Copy)
         (U.toposort (T.uop t)))
  in
  group "clone"
    [
      test "a clone on another device copies its source across" (fun () ->
          let t = Creation.clone ~device:(U.Single "METAL") (param "CPU") in
          equal int 1 (copies t);
          is_true ~msg:"placed on the device asked for"
            (T.device t = Some (U.Single "METAL")));
      test "a clone on its source's device copies nothing across" (fun () ->
          equal int 0 (copies (Creation.clone (param "CPU")));
          equal int 0
            (copies (Creation.clone ~device:(U.Single "CPU") (param "CPU"))));
      test "empty storage is placed on the device asked for" (fun () ->
          is_true ~msg:"device"
            (T.device (Creation.empty ~device:(U.Single "CPU") [ 2; 2 ])
            = Some (U.Single "CPU")));
    ]

let sort_tests =
  group "sort"
    [
      test "sort ascending values and indices" (fun () ->
          let v, i = Op.sort (vec [| 3.; 1.; 2.; 5.; 4. |]) in
          check_floats [| 1.; 2.; 3.; 4.; 5. |] v;
          check_ints [| 1; 2; 0; 4; 3 |] i);
      test "sort descending" (fun () ->
          let v, i = Op.sort ~descending:true (vec [| 3.; 1.; 2.; 5.; 4. |]) in
          check_floats [| 5.; 4.; 3.; 2.; 1. |] v;
          check_ints [| 3; 4; 0; 2; 1 |] i);
      test "argsort" (fun () ->
          check_ints [| 1; 2; 0; 4; 3 |] (Op.argsort (vec [| 3.; 1.; 2.; 5.; 4. |])));
      test "sort 2d axis 1" (fun () ->
          let v, i = Op.sort ~dim:1 (fa ~shape:[ 2; 3 ] [| 3.; 1.; 2.; 6.; 5.; 4. |]) in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] v;
          check_ints [| 1; 2; 0; 2; 1; 0 |] i);
      test "sort keeps ties stable" (fun () ->
          let v, i = Op.sort (vec [| 2.; 1.; 2.; 1. |]) in
          check_floats [| 1.; 1.; 2.; 2. |] v;
          check_ints [| 1; 3; 0; 2 |] i);
      test "sort length not a power of two" (fun () ->
          let _, i = Op.sort (vec [| 8.; 3.; 5.; 1.; 7.; 2.; 6.; 4. |]) in
          check_ints [| 3; 5; 1; 7; 2; 6; 4; 0 |] i);
      test "sort int input" (fun () ->
          let v, i = Op.sort (Run.of_int_array ~shape:[ 5 ] [| 3; 1; 2; 5; 4 |]) in
          check_ints [| 1; 2; 3; 4; 5 |] v;
          check_ints [| 1; 2; 0; 4; 3 |] i);
      test "topk largest" (fun () ->
          let v, i = Op.topk (vec [| 1.; 5.; 3.; 4.; 2. |]) 3 in
          check_floats [| 5.; 4.; 3. |] v;
          check_ints [| 1; 3; 2 |] i);
      test "topk smallest" (fun () ->
          let v, i = Op.topk ~largest:false (vec [| 1.; 5.; 3.; 4.; 2. |]) 2 in
          check_floats [| 1.; 2. |] v;
          check_ints [| 0; 4 |] i);
    ]

(* Values may be infinite; [close] is NaN on two infinities of the same
   sign, so compare those for equality instead. *)
let check_floats_inf expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      let ok =
        if Float.is_finite e then close e got.(i) else e = got.(i)
      in
      if not ok then failf "element %d: expected %g, got %g" i e got.(i))
    expected

let stack_tests =
  group "stack"
    [
      test "stack along new leading axis" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] (Mv.stack a [ b ]));
      test "stack along inner axis" (fun () ->
          let a = vec [| 1.; 2. |] and b = vec [| 3.; 4. |] in
          check_floats [| 1.; 3.; 2.; 4. |] (Mv.stack ~dim:1 a [ b ]));
    ]

(* [cat] takes one of two lowerings: equal extents on the joined axis stack
   and merge the new axis, anything else pads and selects. Both are covered on
   every axis, since only the equal-extent case reaches the stack. *)
let cat_tests =
  group "cat"
    [
      test "equal extents along axis 0" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] (Op.cat a [ b ]));
      test "equal extents along axis 1" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 5.; 6.; 7.; 8. |] in
          check_floats
            [| 1.; 2.; 5.; 6.; 3.; 4.; 7.; 8. |]
            (Op.cat ~dim:1 a [ b ]));
      test "equal extents, three operands" (fun () ->
          let a = vec [| 1.; 2. |] and b = vec [| 3.; 4. |] in
          let c = vec [| 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat a [ b; c ]));
      test "unequal extents along axis 0" (fun () ->
          let a = fa ~shape:[ 1; 2 ] [| 1.; 2. |] in
          let b = fa ~shape:[ 2; 2 ] [| 3.; 4.; 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat a [ b ]));
      test "unequal extents along axis 1" (fun () ->
          let a = fa ~shape:[ 2; 1 ] [| 1.; 4. |] in
          let b = fa ~shape:[ 2; 2 ] [| 2.; 3.; 5.; 6. |] in
          check_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] (Op.cat ~dim:1 a [ b ]));
      test "unequal extents keep every bit" (fun () ->
          (* Both zeros, NaNs of either sign with payloads, signalling ones,
             and subnormals, which a sum of zero-padded pieces changes. *)
          let module D = Tolk_uop.Dtype in
          let words =
            [|
              0x80000000l; 0x7F800001l; 0xFFC00123l; 0x00000001l; 0x807FFFFFl;
              0x7FA00000l; 0x80000000l; 0x3F800000l; 0x80000001l; 0x00000000l;
            |]
          in
          let bytes = Bytes.create 40 in
          Array.iteri (fun i w -> Bytes.set_int32_le bytes (4 * i) w) words;
          let words_of t =
            let b = Run.data t in
            Array.init (Bytes.length b / 4) (fun i -> Bytes.get_int32_le b (4 * i))
          in
          let devices =
            U.Single "CPU" :: Option.to_list (T.device (vec [| 0. |]))
            |> List.sort_uniq compare
          in
          List.iter
            (fun device ->
              let x =
                Creation.clone ~device
                  (Run.of_bytes ~dtype:D.float32 ~shape:[ 10 ] bytes)
              in
              let piece lo hi = Mv.shrink x [ (lo, hi) ] in
              equal (array int32) words
                (words_of (Op.cat (piece 0 3) [ piece 3 8; piece 8 10 ]));
              let rows = Mv.reshape x [ 2; 5 ] in
              let columns lo hi = Mv.shrink rows [ (0, 2); (lo, hi) ] in
              equal (array int32) words
                (words_of
                   (Op.cat ~dim:1 (columns 0 1) [ columns 1 3; columns 3 5 ])))
            devices);
      test "a single operand is the identity" (fun () ->
          check_floats [| 1.; 2.; 3. |] (Op.cat (vec [| 1.; 2.; 3. |]) []));
      test "negative axis counts from the end" (fun () ->
          let a = fa ~shape:[ 2; 1 ] [| 1.; 3. |] in
          let b = fa ~shape:[ 2; 1 ] [| 2.; 4. |] in
          check_floats [| 1.; 2.; 3.; 4. |] (Op.cat ~dim:(-1) a [ b ]));
      test "booleans keep their values" (fun () ->
          let b v = Dt.bool (vec v) in
          check_ints [| 1; 0; 0; 1 |]
            (Dt.int (Op.cat (b [| 1.; 0. |]) [ b [| 0.; 1. |] ])));
      test "operands are promoted to a common dtype" (fun () ->
          let a = Dt.int (vec [| 1.; 2. |]) and b = vec [| 3.5; 4.5 |] in
          check_floats [| 1.; 2.; 3.5; 4.5 |] (Op.cat a [ b ]));
      test "unequal extents convert each operand once" (fun () ->
          (* 16777217 rounds to float32 as 16777216; through float16 first,
             as a join of int32 and float16 would take it, it overflows. *)
          let i = Run.of_int_array ~shape:[ 2 ] [| 16777217; 3 |] in
          let h = Dt.half (vec [| 1.; 2.; 4. |]) in
          check_floats [| 16777216.; 3.; 1.; 2.; 4.; 5. |]
            (Op.cat i [ h; vec [| 5. |] ]));
      test "shapes must match off the joined axis" (fun () ->
          let a = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          let b = fa ~shape:[ 2; 3 ] (Array.make 6 0.) in
          raises
            (Invalid_argument
               "Op.cat: shapes must match off the concatenated axis")
            (fun () -> ignore (Op.cat a [ b ])));
    ]

let triu_tests =
  group "triu"
    [
      test "triu main diagonal" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 0.; 5.; 6.; 0.; 0.; 9. |]
            (Op.triu (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "triu positive diagonal" (fun () ->
          check_floats
            [| 0.; 2.; 3.; 0.; 0.; 6.; 0.; 0.; 0. |]
            (Op.triu ~diagonal:1
               (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "triu negative diagonal" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 4.; 5.; 6.; 0.; 8.; 9. |]
            (Op.triu ~diagonal:(-1)
               (fa ~shape:[ 3; 3 ] (Array.init 9 (fun i -> float_of_int (i + 1))))));
      test "full neg-infinity causal mask" (fun () ->
          let mask =
            Op.triu ~diagonal:1
              (Creation.full [ 2; 2 ] (T.Sfloat Float.neg_infinity))
          in
          check_floats_inf
            [| 1.; Float.neg_infinity; 3.; 4. |]
            (El.add (fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |]) mask));
    ]

let assign_tests =
  group "assign"
    [
      test "assign through a bitcast updates the original tensor" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let view = Dt.bitcast base Tolk_uop.Dtype.int32 in
          let values =
            Run.of_int_array ~shape:[ 4 ]
              [| 0x40800000; 0x40400000; 0x40000000; 0x3f800000 |]
          in
          ignore (Op.assign view values);
          check_floats [| 4.; 3.; 2.; 1. |] base);
      test "assign through a shrunk and twice-bitcast view keeps other elements" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let view =
            Mv.shrink base [ (0, 2) ]
            |> fun x -> Dt.bitcast x Tolk_uop.Dtype.uint32
            |> fun x -> Dt.bitcast x Tolk_uop.Dtype.int32
          in
          let values =
            Run.of_int_array ~shape:[ 2 ] [| 0x40800000; 0x40400000 |]
          in
          ignore (Op.assign view values);
          check_floats [| 4.; 3.; 3.; 4. |] base);
      test "assign through a wider bitcast updates the original bytes" (fun () ->
          let base =
            Run.of_bytes ~dtype:Tolk_uop.Dtype.uint8
              ~shape:[ 8 ] (Bytes.make 8 '\000')
          in
          let view = Dt.bitcast base Tolk_uop.Dtype.int64 in
          let bytes = Bytes.make 8 '\000' in
          Bytes.set_int64_le bytes 0 12345L;
          let values =
            Run.of_bytes ~dtype:Tolk_uop.Dtype.int64 ~shape:[ 1 ] bytes
          in
          ignore (Op.assign view values);
          equal string (Bytes.to_string bytes) (Bytes.to_string (Run.data base)));
      test "assign to a pending value discards its old computation" (fun () ->
          let input = vec [| 1.; 2.; 3. |] in
          let destination = El.add input (T.f 1.) in
          let pending = T.uop destination in
          ignore (Op.assign destination (vec [| 7.; 8.; 9. |]));
          is_true
            (not (List.exists (( == ) pending) (U.toposort (T.uop destination))));
          check_floats [| 7.; 8.; 9. |] destination;
          check_floats [| 1.; 2.; 3. |] input);
      test "assign to a pending contiguous value discards its old computation" (fun () ->
          let destination =
            El.contiguous (El.add (vec [| 1.; 2.; 3. |]) (T.f 1.))
          in
          let pending = T.uop destination in
          ignore (Op.assign destination (vec [| 7.; 8.; 9. |]));
          is_true
            (not (List.exists (( == ) pending) (U.toposort (T.uop destination))));
          check_floats [| 7.; 8.; 9. |] destination);
      test "initialization by assignment does not alias its source" (fun () ->
          let source = El.add (vec [| 1.; 2.; 3. |]) (T.f 1.) in
          let destination = El.mul (vec [| 4.; 5.; 6. |]) (T.f 2.) in
          ignore (Op.assign destination source);
          ignore (Op.assign destination (T.f 7.));
          check_floats [| 7.; 7.; 7. |] destination;
          check_floats [| 2.; 3.; 4. |] source);
      test "partial assign materializes a pending contiguous value for its aliases" (fun () ->
          let input = vec [| 1.; 2.; 3.; 4. |] in
          let base = El.contiguous (El.mul input (T.f 2.)) in
          let view = Mv.shrink base [ (1, 3) ] in
          ignore (Op.assign view (T.f 7.));
          check_floats [| 2.; 7.; 7.; 8. |] base;
          check_floats [| 7.; 7. |] view;
          check_floats [| 1.; 2.; 3.; 4. |] input);
      test "assign an identity computation leaves storage unchanged" (fun () ->
          let t = vec [| 1.; 2.; 3. |] in
          ignore (Op.assign t (El.add t (T.f 0.)));
          check_floats [| 1.; 2.; 3. |] t);
      test "reads execute pending assignments exactly once" (fun () ->
          let t = vec [| 1.; 2.; 3. |] in
          ignore (Op.assign t (El.add t (T.f 1.)));
          check_floats [| 2.; 3.; 4. |] t;
          check_floats [| 2.; 3.; 4. |] t;
          ignore (Op.assign t (El.add t (T.f 1.)));
          check_floats [| 3.; 4.; 5. |] t;
          check_floats [| 3.; 4.; 5. |] t);
      test "assign weak scalars into concrete storage" (fun () ->
          let floats = vec [| 0.; 0.; 0. |] in
          ignore (Op.assign floats (T.f 2.5));
          check_floats [| 2.5; 2.5; 2.5 |] floats;
          let ints = Run.of_int_array ~shape:[ 3 ] [| 0; 0; 0 |] in
          ignore (Op.assign ints (T.i 7));
          check_ints [| 7; 7; 7 |] ints);
      test "assign weak scalar through a view updates the base" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let view = Mv.shrink base [ (1, 3) ] in
          ignore (Op.assign view (T.i 7));
          check_floats [| 1.; 7.; 7.; 4. |] base);
      test "assign commits a weak destination without narrowing" (fun () ->
          let large = 1 lsl 40 in
          let dst = T.i large in
          ignore (Op.assign dst (T.i (large + 1)));
          let dtype =
            Testable.make ~pp:Tolk_uop.Dtype.pp ~equal:Tolk_uop.Dtype.equal
          in
          equal dtype Tolk_uop.Dtype.int64 (T.dtype dst);
          let bytes = Run.data dst in
          equal int 8 (Bytes.length bytes);
          equal int64 (Int64.of_int (large + 1)) (Bytes.get_int64_le bytes 0));
      test "assign and indexed scatter reject weak-float values for integer storage" (fun () ->
          let destination () = Run.of_int_array ~shape:[ 1 ] [| 0 |] in
          let source = Mv.reshape (T.f 1.5) [ 1 ] in
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Op.assign (destination ()) source);
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.scatter_indexed (destination ()) ~dim:0
                (Run.of_int_array ~shape:[ 1 ] [| 0 |]) source ~mode:`Set
                ~unique:true));
      test "assign whole tensor in place" (fun () ->
          let t = vec [| 1.; 2.; 3. |] in
          ignore (Run.realize t);
          ignore (Run.realize (Op.assign t (vec [| 4.; 5.; 6. |])));
          check_floats [| 4.; 5.; 6. |] t);
      test "assign broadcasts the value" (fun () ->
          let t = fa ~shape:[ 2; 3 ] (Array.make 6 0.) in
          ignore (Run.realize (Op.assign t (vec [| 1.; 2.; 3. |])));
          check_floats [| 1.; 2.; 3.; 1.; 2.; 3. |] t);
      test "assign dtype mismatch raises" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.assign (vec [| 1. |]) (Run.of_int_array ~shape:[ 1 ] [| 1 |])));
      test "assign to shrunk view then read back (kv cache)" (fun () ->
          (* cache is (2, bsz=1, ctx=4, heads=2, head_dim=3); write positions
             1..2 with the stack of xk and xv, as a transformer kv-cache
             update does, then read the whole cache and a keys slice back. *)
          let cache = fa ~shape:[ 2; 1; 4; 2; 3 ] (Array.make 48 0.) in
          ignore (Run.realize cache);
          let xk =
            fa ~shape:[ 1; 2; 2; 3 ] (Array.init 12 (fun i -> float_of_int (i + 1)))
          in
          let xv =
            fa ~shape:[ 1; 2; 2; 3 ]
              (Array.init 12 (fun i -> float_of_int (i + 13)))
          in
          let view =
            Op.getitem cache
              [ Mv.All; Mv.All; Mv.R (Some 1, Some 3, None); Mv.All; Mv.All ]
          in
          ignore (Run.realize (Op.assign view (Mv.stack xk [ xv ])));
          let expected = Array.make 48 0. in
          for pos = 0 to 1 do
            for h = 0 to 1 do
              for c = 0 to 2 do
                let src = (((pos * 2) + h) * 3) + c in
                expected.((((1 + pos) * 2 + h) * 3) + c) <-
                  float_of_int (src + 1);
                expected.((((4 + 1 + pos) * 2 + h) * 3) + c) <-
                  float_of_int (src + 13)
              done
            done
          done;
          check_floats expected cache;
          (* keys = cache[0][:, :3, :, :]: position 0 still zero, then xk. *)
          let keys =
            Op.getitem
              (Op.getitem cache [ Mv.I 0 ])
              [ Mv.All; Mv.R (None, Some 3, None); Mv.All; Mv.All ]
          in
          let expected_keys = Array.make 18 0. in
          Array.blit expected 6 expected_keys 6 12;
          check_floats expected_keys keys);
      test "sequential view assigns accumulate" (fun () ->
          let cache = fa ~shape:[ 2; 1; 4; 2; 3 ] (Array.make 48 0.) in
          ignore (Run.realize cache);
          let step start v =
            let xk = fa ~shape:[ 1; 1; 2; 3 ] (Array.make 6 v) in
            let xv = fa ~shape:[ 1; 1; 2; 3 ] (Array.make 6 (v +. 0.5)) in
            let view =
              Op.getitem cache
                [
                  Mv.All; Mv.All;
                  Mv.R (Some start, Some (start + 1), None);
                  Mv.All; Mv.All;
                ]
            in
            ignore (Run.realize (Op.assign view (Mv.stack xk [ xv ])))
          in
          step 0 1.;
          step 1 2.;
          let expected = Array.make 48 0. in
          for h = 0 to 5 do
            expected.(h) <- 1.;
            expected.(6 + h) <- 2.;
            expected.(24 + h) <- 1.5;
            expected.(30 + h) <- 2.5
          done;
          check_floats expected cache);
    ]

let attention_tests =
  let q () = fa ~shape:[ 2; 3 ] [| 0.1; 0.2; 0.3; -0.1; 0.4; 0.5 |] in
  let k () = fa ~shape:[ 2; 3 ] [| 0.5; 0.1; -0.2; 0.3; 0.9; 0.4 |] in
  let v () = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  group "attention"
    [
      test "sdpa matches reference" (fun () ->
          check_floats
            [| 2.638171; 3.638171; 4.638171; 2.774017; 3.774017; 4.774017 |]
            (Op.scaled_dot_product_attention (q ()) (k ()) (v ())));
      test "sdpa causal matches reference" (fun () ->
          check_floats
            [| 1.; 2.; 3.; 2.774017; 3.774017; 4.774017 |]
            (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
               (v ())));
      test "sdpa additive neg-infinity mask equals causal" (fun () ->
          let mask =
            Op.triu ~diagonal:1
              (Creation.full [ 2; 2 ] (T.Sfloat Float.neg_infinity))
          in
          check_floats
            (Run.to_float_array
               (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
                  (v ())))
            (Op.scaled_dot_product_attention ~attn_mask:mask (q ()) (k ())
               (v ())));
      test "sdpa boolean mask equals causal" (fun () ->
          let mask = Dt.bool (Run.of_int_array ~shape:[ 2; 2 ] [| 1; 0; 1; 1 |]) in
          check_floats
            (Run.to_float_array
               (Op.scaled_dot_product_attention ~is_causal:true (q ()) (k ())
                  (v ())))
            (Op.scaled_dot_product_attention ~attn_mask:mask (q ()) (k ())
               (v ())));
      test "sdpa rejects mask with is_causal" (fun () ->
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Op.scaled_dot_product_attention ~is_causal:true
                ~attn_mask:(fa ~shape:[ 2; 2 ] (Array.make 4 0.))
                (q ()) (k ()) (v ())));
      test "layernorm matches reference" (fun () ->
          check_floats
            [| -1.341635; -0.447212; 0.447212; 1.341635 |]
            (Op.layernorm (vec [| 1.; 2.; 3.; 4. |])));
    ]

let gpt2_getitem_tests =
  group "gpt2_getitem"
    [
      test "integer index in the middle of the rank" (fun () ->
          (* xqkv[:, :, i, :, :] on a (1, 2, 3, 2, 2) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 1; 2; 3; 2; 2 ] data in
          let expected i =
            Array.init 8 (fun j ->
                let s = j / 4 and h = j mod 4 / 2 and d = j mod 2 in
                data.((((s * 3) + i) * 2 + h) * 2 + d))
          in
          check_floats (expected 1)
            (Op.getitem t [ Mv.All; Mv.All; Mv.I 1; Mv.All; Mv.All ]);
          check_floats (expected 2)
            (Op.getitem t [ Mv.All; Mv.All; Mv.I 2; Mv.All; Mv.All ]));
      test "negative integer index selects the last row" (fun () ->
          (* logits[:, -1, :] on a (2, 3, 4) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 2; 3; 4 ] data in
          check_floats
            (Array.init 8 (fun j -> data.((j / 4 * 3 + 2) * 4 + (j mod 4))))
            (Op.getitem t [ Mv.All; Mv.I (-1); Mv.All ]));
      test "leading index then open-ended slice" (fun () ->
          (* cache_kv[0][:, :2, :, :] on a (2, 1, 3, 2, 2) tensor. *)
          let data = Array.init 24 float_of_int in
          let t = fa ~shape:[ 2; 1; 3; 2; 2 ] data in
          check_floats
            (Array.sub data 0 8)
            (Op.getitem
               (Op.getitem t [ Mv.I 0 ])
               [ Mv.All; Mv.R (None, Some 2, None); Mv.All; Mv.All ]));
    ]

(* Symbolic shapes: a shape dimension may be an expression over a bound
   variable. One traced graph then serves every bound value; the value is
   passed to the kernel at launch. *)


let bound_var ?(min_val = 0) name ~max_val value =
  let var = U.variable ~name ~min_val ~max_val () in
  U.bind ~var ~value:(U.const_int value)

let plus1 u = U.O.(u + U.const_int 1)

let symbolic_tests =
  group "symbolic"
    [
      (* A write whose length is a variable in [0, 1] loops that many times:
         at length 0 it writes nothing, at the start or at an offset. *)
      test "a store of symbolic length at most 1 writes only when it is 1"
        (fun () ->
          List.iter
            (fun (offset, length) ->
              let cache = vec [| 0.; 0.; 0. |] in
              ignore (Run.realize cache);
              let len = bound_var "store_len" ~max_val:1 length in
              let start = U.const_int offset in
              let view =
                Mv.symbolic_shrink cache [ Some (start, U.O.(start + len)) ]
              in
              let src =
                Mv.symbolic_shrink (vec [| 9.; 9.; 9. |])
                  [ Some (U.const_int 0, len) ]
              in
              ignore (Run.realize (Op.assign view src));
              check_floats
                (Array.init 3 (fun i ->
                     if i = offset && length = 1 then 9. else 0.))
                cache)
            [ (0, 0); (0, 1); (1, 0); (1, 1) ]);
      test "advanced indexing retains symbolic unindexed dimensions" (fun () ->
          let index values = Mv.T (Run.of_int_array ~shape:[ 2 ] values) in
          List.iter
            (fun length ->
              let bound = bound_var "advanced_len" ~min_val:1 ~max_val:3 length in
              let matrix =
                fa ~shape:[ 3; 4 ] (Array.init 12 (fun i -> float_of_int (i + 1)))
              in
              let matrix = Mv.symbolic_shrink matrix
                  [ Some (U.const_int 0, bound); None ] in
              let columns = Op.getitem matrix [ Mv.All; index [| 2; 0 |] ] in
              check_floats [| float_of_int (4 * length * length) |] (Rd.sum columns);
              let cube () =
                fa ~shape:[ 2; 3; 3 ] (Array.init 18 (fun i -> float_of_int (i + 1)))
              in
              let middle = Mv.symbolic_shrink (cube ())
                  [ None; Some (U.const_int 0, bound); None ] in
              let separated = Op.getitem middle
                  [ index [| 1; 0 |]; Mv.All; index [| 2; 0 |] ] in
              check_floats [| float_of_int (3 * length * length + 10 * length) |]
                (Rd.sum separated);
              let trailing = Mv.symbolic_shrink (cube ())
                  [ None; None; Some (U.const_int 0, bound) ] in
              let consecutive =
                Op.getitem trailing [ index [| 1; 0 |]; index [| 2; 0 |] ]
              in
              check_floats [| float_of_int (length * length + 16 * length) |]
                (Rd.sum consecutive))
            [ 1; 2; 3 ]);
      test "advanced index tensors can have a symbolic length" (fun () ->
          List.iter
            (fun (length, expected) ->
              let bound =
                bound_var "index_tensor_len" ~min_val:1 ~max_val:3 length
              in
              let index = Run.of_int_array ~shape:[ 3 ] [| -1; 0; 2 |] in
              let index =
                Mv.symbolic_shrink index [ Some (U.const_int 0, bound) ]
              in
              let result = Op.getitem (vec [| 1.; 2.; 3.; 4. |]) [ Mv.T index ] in
              check_floats [| expected |] (Rd.sum result))
            [ 1, 4.; 2, 5.; 3, 8. ]);
      test "integer getitem resolves against a symbolic axis" (fun () ->
          List.iter
            (fun length ->
              let base = fa ~shape:[ 2; 4 ] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
              let bound = bound_var "index_len" ~max_val:4 length in
              let source = Mv.symbolic_shrink base
                  [ None; Some (U.const_int 0, bound) ] in
              check_floats [| float_of_int length; float_of_int (4 + length) |]
                (Op.getitem source [ Mv.All; Mv.I (-1) ]);
              check_floats [| float_of_int (length * (length + 1) / 2) |]
                (Rd.sum (Op.getitem source [ Mv.I 0; Mv.All ])))
            [ 1; 3; 4 ]);
      test "negative slice bounds use the symbolic length" (fun () ->
          List.iter
            (fun length ->
              let base = vec [| 1.; 2.; 3.; 4. |] in
              let bound = bound_var "slice_len" ~min_val:2 ~max_val:4 length in
              let source = Mv.symbolic_shrink base [ Some (U.const_int 0, bound) ] in
              check_floats [| float_of_int (length - 1); float_of_int length |]
                (Op.getitem source [ Mv.New; Mv.Ellipsis; Mv.R (Some (-2), None, None) ]);
              check_floats [| float_of_int (length * (length - 1) / 2) |]
                (Rd.sum (Op.getitem source [ Mv.R (None, Some (-1), None) ])))
            [ 2; 3; 4 ]);
      test "symbolic slicing rejects unproved lengths and non-unit steps" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let bound = bound_var "unknown_slice_len" ~max_val:4 3 in
          let source = Mv.symbolic_shrink base [ Some (U.const_int 0, bound) ] in
          List.iter
            (fun index ->
              raises_match (function Invalid_argument _ -> true | _ -> false)
                (fun () -> Op.getitem source [ index ]))
            [ Mv.R (None, Some (-1), None); Mv.R (None, None, Some (-1));
              Mv.R (None, None, Some 2); Mv.R (None, None, Some 0) ]);
      test "raw byte reads reject a symbolic logical shape" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let bound = bound_var "read_len" ~max_val:4 3 in
          let view = Mv.symbolic_shrink base [ Some (U.const_int 0, bound) ] in
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Run.data view));
      test "cloning a symbolic view preserves its values and storage independence" (fun () ->
          List.iter
            (fun length ->
              let base = fa ~shape:[ 2; 4 ] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
              let bound = bound_var "clone_len" ~max_val:4 length in
              let source = Mv.symbolic_shrink base
                  [ None; Some (U.const_int 0, bound) ] in
              let cloned = Creation.clone source in
              let shape = T.symbolic_shape source in
              is_true (List.equal U.equal shape (T.symbolic_shape cloned));
              let expected =
                float_of_int (length * (length + 1) + 4 * length)
              in
              check_floats [| expected |] (Rd.sum cloned);
              is_true (List.equal U.equal shape (T.symbolic_shape cloned));
              ignore (Op.assign cloned (T.f 9.));
              check_floats [| float_of_int (18 * length) |] (Rd.sum cloned);
              check_floats [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] base)
            [ 0; 1; 3; 4 ]);
      test "initializing a symbolic pending value preserves its logical shape" (fun () ->
          let base = vec [| 1.; 2.; 3.; 4. |] in
          let bound = bound_var "assign_len" ~max_val:4 3 in
          let source = Mv.symbolic_shrink base [ Some (U.const_int 0, bound) ] in
          let destination = El.add source (T.f 1.) in
          ignore (Op.assign destination (T.f 7.));
          check_floats [| 21. |] (Rd.sum destination));
      test "shrink + sum matches concrete for several bind values" (fun () ->
          let data = Array.init 8 (fun i -> float_of_int (i + 1)) in
          List.iter
            (fun pos ->
              let t = vec data in
              let bound = bound_var "pos" ~max_val:6 pos in
              let s =
                Mv.symbolic_shrink t [ Some (U.const_int 0, plus1 bound) ]
              in
              let expected =
                Array.fold_left ( +. ) 0. (Array.sub data 0 (pos + 1))
              in
              check_floats [| expected |] (Rd.sum s))
            [ 2; 5; 0 ]);
      test "attention with symbolic key/value length matches concrete"
        (fun () ->
          let qdata = [| 0.1; 0.2; 0.3; -0.1 |] in
          let kdata = Array.init 24 (fun i -> float_of_int (i mod 5) /. 5.) in
          let vdata = Array.init 24 (fun i -> float_of_int (i + 1)) in
          List.iter
            (fun pos ->
              let bound = bound_var "apos" ~max_val:4 pos in
              let sym_bounds = [ Some (U.const_int 0, plus1 bound); None ] in
              let keys =
                Mv.symbolic_shrink (fa ~shape:[ 6; 4 ] kdata) sym_bounds
              in
              let values =
                Mv.symbolic_shrink (fa ~shape:[ 6; 4 ] vdata) sym_bounds
              in
              let attend q k v =
                Op.matmul
                  (Op.softmax ~axis:(-1)
                     (Op.matmul q (Mv.transpose ~dim0:(-2) ~dim1:(-1) k)))
                  v
              in
              let got =
                attend (fa ~shape:[ 1; 4 ] qdata) keys values
              in
              let concrete_bounds = [ (0, pos + 1); (0, 4) ] in
              let expected =
                attend
                  (fa ~shape:[ 1; 4 ] qdata)
                  (Mv.shrink (fa ~shape:[ 6; 4 ] kdata) concrete_bounds)
                  (Mv.shrink (fa ~shape:[ 6; 4 ] vdata) concrete_bounds)
              in
              check_floats (Run.to_float_array expected) got)
            [ 1; 3 ]);
      test "kv-cache: assign at symbolic position, read symbolic prefix"
        (fun () ->
          (* cache is (ctx=4, width=2); each decode step writes one row at a
             symbolic position and reads back rows [0, pos+1). *)
          let cache = fa ~shape:[ 4; 2 ] (Array.make 8 0.) in
          ignore (Run.realize cache);
          let step pos value expected_sum =
            let bound = bound_var "cpos" ~max_val:3 pos in
            let row =
              fa ~shape:[ 1; 2 ] [| value; value +. 0.5 |]
            in
            let view =
              Mv.symbolic_shrink cache [ Some (bound, plus1 bound); None ]
            in
            ignore (Run.realize (Op.assign view row));
            let keys =
              Mv.symbolic_shrink cache
                [ Some (U.const_int 0, plus1 bound); None ]
            in
            check_floats [| expected_sum |] (Rd.sum keys)
          in
          step 0 1. 2.5;
          step 1 2. 7.;
          step 2 3. 13.5;
          check_floats [| 1.; 1.5; 2.; 2.5; 3.; 3.5; 0.; 0. |] cache);
      test "one schedule serves every bind value" (fun () ->
          (* The bound value is stripped from the schedule cache key, so the
             second value must reuse the first schedule instead of scheduling
             (and later compiling) a new kernel. *)
          let lowered = ref 0 in
          let get_kernel_graph sink =
            incr lowered;
            Tolk.Rangeify.get_kernel_graph sink
          in
          let schedule pos =
            let data = Array.init 8 (fun i -> float_of_int (i + 1)) in
            let t = vec data in
            let bound = bound_var "spos" ~max_val:6 pos in
            let s =
              Mv.symbolic_shrink t [ Some (U.const_int 0, plus1 bound) ]
            in
            let out = Rd.sum (El.mul s (T.f 2.5)) in
            let sink =
              U.sink [ U.contiguous ~src:(T.uop out) () ]
            in
            let call, _ = bufferized_call sink in
            ignore
              (U.graph_rewrite ~enter_calls:true
                 (fun node ->
                   Tolk.Schedule.lower_sink_to_linear ~get_kernel_graph node)
                 call)
          in
          schedule 2;
          equal int ~msg:"first value schedules" 1 !lowered;
          schedule 5;
          equal int ~msg:"second value hits the schedule cache" 1 !lowered);
    ]

(* A contiguous view of a realized buffer realizes as an alias of that buffer —
   the source memory viewed at the slice offset, with no copy — mirroring the
   reference, where such a view resolves lazily to [base.view(offset)]. *)
let aliasing_tests =
  let module Device = Tolk.Device in
  let buffer_of t =
    match Run.buffer_of_node (T.uop t) with
    | Some b -> b
    | None -> fail "tensor has no backing buffer after realize"
  in
  group "aliasing"
    [
      test "contiguous slice aliases its source buffer" (fun () ->
          let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
          ignore (Run.realize x);
          let xbuf = buffer_of x in
          let y = Op.getitem x [ Mv.R (Some 2, Some 6, None) ] in
          ignore (Run.realize y);
          let ybuf = buffer_of y in
          (* Same underlying allocation, viewed at the slice's byte offset (2
             elements * 4 bytes), as a distinct view — not a fresh buffer. *)
          equal int
            ~msg:"slice shares the source allocation"
            (Device.Buffer.base_id xbuf) (Device.Buffer.base_id ybuf);
          equal int ~msg:"view starts at the slice offset" (2 * 4)
            (Device.Buffer.offset ybuf);
          if Device.Buffer.id xbuf = Device.Buffer.id ybuf then
            fail "slice must be a distinct view, not the base buffer itself";
          check_floats [| 2.; 3.; 4.; 5. |] y);
      test "explicit contiguous on a slice still aliases" (fun () ->
          let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
          ignore (Run.realize x);
          let xbuf = buffer_of x in
          let y = El.contiguous (Op.getitem x [ Mv.R (Some 3, Some 7, None) ]) in
          ignore (Run.realize y);
          let ybuf = buffer_of y in
          equal int
            ~msg:"contiguous view shares the source allocation"
            (Device.Buffer.base_id xbuf) (Device.Buffer.base_id ybuf);
          equal int ~msg:"view starts at the slice offset" (3 * 4)
            (Device.Buffer.offset ybuf);
          check_floats [| 3.; 4.; 5.; 6. |] y);
      test "symbolic leading views retain their storage and byte offset" (fun () ->
          List.iter (fun size ->
              let x = fa ~shape:[3; 2] [|1.; 2.; 3.; 4.; 5.; 6.|] in
              let n = U.variable ~name:"view_rows" ~min_val:1 ~max_val:2 () in
              let bound = U.bind ~var:n ~value:(U.const_int size) in
              let view = U.shrink ~src:(T.uop x)
                  ~offset:(U.stack [U.const_int 1; U.const_int 0])
                  ~size:(U.stack [bound; U.const_int 2]) |> T.of_uop |> El.contiguous in
              ignore (Run.realize view);
              let base = buffer_of x and selected = buffer_of view in
              equal int (Device.Buffer.base_id base) (Device.Buffer.base_id selected);
              equal int 8 (Device.Buffer.offset selected);
              let sum = Rd.sum view in
              equal float_exact (if size = 1 then 7. else 18.) (Run.item_float sum)) [1; 2]);
      test "subword bitcast views alias the original bytes" (fun () ->
          let x = Run.of_int_array ~shape:[2] [|0x04030201; 0x08070605|] in
          let part = Mv.shrink (Dt.bitcast x Tolk_uop.Dtype.uint8) [(1, 3)] in
          let view = Dt.bitcast part Tolk_uop.Dtype.uint16 |> El.contiguous in
          ignore (Run.realize view);
          let base = buffer_of x and selected = buffer_of view in
          equal int (Device.Buffer.base_id base) (Device.Buffer.base_id selected);
          equal int 1 (Device.Buffer.offset selected);
          equal string "\002\003" (Bytes.to_string (Run.data view)));
      test "a shrunk trailing axis is not a contiguous range" (fun () ->
          let x = fa ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          check_floats [| 1.; 2.; 4.; 5. |]
            (El.contiguous (Mv.shrink x [ (0, 2); (0, 2) ]));
          check_floats [| 2.; 3.; 5.; 6. |]
            (El.contiguous (Mv.shrink x [ (0, 2); (1, 3) ])));
      test "a shrunk leading axis is a contiguous range" (fun () ->
          let x = fa ~shape:[ 3; 2 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          check_floats [| 3.; 4.; 5.; 6. |]
            (El.contiguous (Mv.shrink x [ (1, 3); (0, 2) ])));
    ]

(* An expression whose graph folds to a pure constant owns no storage and is
   placed on no device. Reading it must materialize it rather than fail. *)

let constant_tests =
  group "constant"
    [
      test "broadcast constant reads back" (fun () ->
          check_floats [| 3.; 3.; 3.; 3. |]
            (Creation.full ~buffer:false [ 4 ] (T.Sfloat 3.0)));
      test "expression folding to a constant reads back" (fun () ->
          check_floats [| 0.; 0.; 0. |]
            (El.mul (vec [| 1.; 2.; 3. |]) (T.f 0.0)));
      test "constant scalar reads back" (fun () ->
          equal float_exact 5.0 (Run.item_float (T.f 5.0)));
      test "constant realizes into a buffer only once" (fun () ->
          let t = Creation.full ~buffer:false [ 2; 3 ] (T.Sfloat 1.5) in
          check_floats [| 1.5; 1.5; 1.5; 1.5; 1.5; 1.5 |] t;
          check_floats [| 1.5; 1.5; 1.5; 1.5; 1.5; 1.5 |] t);
    ]

let numerical_edge_tests =
  let int64s values =
    let bytes = Bytes.create (Array.length values * 8) in
    Array.iteri (fun i n -> Bytes.set_int64_le bytes (8 * i) n) values;
    Run.of_bytes ~dtype:Tolk_uop.Dtype.int64 ~shape:[ Array.length values ] bytes
  in
  let check_int64s values tensor =
    let bytes = Run.data tensor in
    equal int (8 * Array.length values) (Bytes.length bytes);
    Array.iteri (fun i n -> equal int64 n (Bytes.get_int64_le bytes (8 * i))) values
  in
  let check expected t =
    let actual = Run.to_float_array t in
    equal int (Array.length expected) (Array.length actual);
    Array.iteri
      (fun i x ->
        let y = actual.(i) in
        if Float.is_nan x then is_true (Float.is_nan y)
        else if Float.is_infinite x then equal float_exact x y
        else if not (Float.abs (x -. y) <= 1e-4 *. max 1. (Float.abs x)) then
          failf "element %d: expected %g, got %g" i x y)
      expected
  in
  group "numerical edges"
    [
      test "staged weak arithmetic preserves its storage width" (fun () ->
          let large = 1 lsl 40 in
          let source = Dt.cast (int64s [| Int64.of_int large |]) Tolk_uop.Dtype.weakint in
          let expression = El.add source (T.i 1) in
          let staged = T.of_uop (U.contiguous ~src:(T.uop expression) ()) in
          check_int64s [| Int64.of_int (large + 1) |] (Dt.long staged));
      test "raw host bytes require a concrete dtype" (fun () ->
          List.iter (fun dtype ->
              raises_match (function Invalid_argument _ -> true | _ -> false)
                (fun () -> Run.of_bytes ~dtype ~shape:[ 1 ] (Bytes.make 100 '\000')))
            Tolk_uop.Dtype.[ weakint; weakfloat ]);
      test "weak constants and their clones read at a concrete width" (fun () ->
          let large = 1 lsl 40 in
          let t = T.i large in
          check_int64s [| Int64.of_int large |] t;
          check_int64s [| Int64.of_int large |] (Creation.clone t));
      test "fills infer the width of large integers" (fun () ->
          let large = 1 lsl 40 in
          let expected = Array.make 2 (Int64.of_int large) in
          check_int64s expected (Creation.full [ 2 ] (T.Sint large));
          check_int64s expected (Creation.full ~buffer:false [ 2 ] (T.Sint large));
          check_int64s expected
            (Creation.full_like (Mv.expand (Mv.reshape (T.i 0) [ 1 ]) [ 2 ])
               (T.Sint large)));
      test "weak scans select a finite storage identity" (fun () ->
          let large = 1 lsl 40 in
          let t = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 3 ] in
          check_int64s (Array.make 3 (Int64.of_int large)) (fst (Op.cummax t)));
      test "weak scatter reductions select a finite storage identity" (fun () ->
          let large = 1 lsl 40 in
          let input = Mv.expand (Mv.reshape (T.i large) [ 1 ]) [ 2 ] in
          let source = Mv.reshape (T.i (large + 1)) [ 1 ] in
          let index = Run.of_int_array ~shape:[ 1 ] [| 0 |] in
          List.iter (fun reduce ->
              check_int64s [| Int64.of_int (large + 1); Int64.of_int large |]
                (Op.scatter_reduce input ~dim:0 index source ~reduce
                   ~include_self:false ())) [ `Amax; `Amin ]);
      test "weak max pooling pads at a concrete integer minimum" (fun () ->
          let input = Mv.expand (Mv.reshape (T.i (-5)) [ 1; 1; 1; 1 ])
              [ 1; 1; 2; 2 ] in
          check_ints (Array.make 4 (-5)) (Op.max_pool2d ~padding:[ 1 ] input));
      test "weak promotion preserves padding and movement" (fun () ->
          let padded = Mv.pad (Mv.expand (Mv.reshape (T.i 5) [ 1; 1 ]) [ 1; 2 ])
              [ (0, 2); (0, 0) ] in
          check_floats [| 6.; 6.; 2.; 2.; 3.; 3. |]
            (El.add padded (fa ~shape:[ 3; 1 ] [| 1.; 2.; 3. |])));
      test "int64 scan, sort and scatter retain full-width identities" (fun () ->
          let lo = Int64.min_int and hi = Int64.max_int in
          let ascending = [| lo; Int64.succ lo; Int64.add lo 2L |] in
          check_int64s ascending (fst (Op.cummax (int64s ascending)));
          let unordered = [| hi; Int64.sub hi 2L; Int64.pred hi |] in
          check_int64s [| Int64.sub hi 2L; Int64.pred hi; hi |]
            (fst (Op.sort (int64s unordered)));
          check_int64s [| hi; Int64.sub hi 2L; hi |]
            (Op.scatter_reduce (int64s [| hi; hi; hi |]) ~dim:0
               (Run.of_int_array ~shape:[ 1 ] [| 1 |])
               (int64s [| Int64.sub hi 2L |]) ~reduce:`Amin ~include_self:false ()));
      test "small-dtype arange accumulates before narrowing" (fun () ->
          let module D = Tolk_uop.Dtype in
          List.iter
            (fun dtype ->
              List.iter
                (fun (start, stop, step) ->
                  let expected =
                    Dt.float (Dt.cast (Op.arange ~stop ~step start) dtype)
                    |> Run.to_float_array
                  in
                  check expected (Dt.float (Op.arange ~dtype ~stop ~step start)))
                ([ (0, 10, 3); (0, 200, 1) ]
                 @ if D.is_fp8 dtype then [] else [ (0, 2560, 1) ]))
            [ D.float16; D.bfloat16; D.fp8e4m3; D.fp8e5m2 ]);
      test "pad_to preserves nonzero fill" (fun () ->
          let input = fa ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          check [| 1.; 2.; -7.; 3.; 4.; -7.; -7.; -7.; -7. |]
            (Op.pad_to ~value:(T.Sfloat (-7.)) input [ Some 3; Some 3 ]);
          check [| 1.; 2.; 0.; 3.; 4.; 0. |]
            (Op.pad_to input [ None; Some 3 ]);
          is_true
            (T.uop (Op.pad_to ~value:(T.Sfloat (-7.)) input [ None; None ])
             == T.uop input));
      test "asinh handles large negative inputs" (fun () ->
          let values = [| -10000.; -1.; 0.; 1.; 10000. |] in
          check (Array.map Float.asinh values) (El.asinh (vec values)));
      test "logaddexp handles infinities" (fun () ->
          check [| neg_infinity; infinity; infinity; 0.; nan |]
            (El.logaddexp
               (vec [| neg_infinity; infinity; infinity; neg_infinity; nan |])
               (vec [| neg_infinity; infinity; 0.; 0.; 1. |])));
      test "logsumexp handles infinite rows" (fun () ->
          check [| neg_infinity; infinity; Float.log 3. |]
            (Op.logsumexp ~axis:1
               (fa ~shape:[ 3; 2 ]
                  [| neg_infinity; neg_infinity; infinity; 0.; 0.; Float.log 2. |])));
      test "logcumsumexp handles infinite prefixes" (fun () ->
          check [| neg_infinity; neg_infinity; 0.; infinity |]
            (Op.logcumsumexp (vec [| neg_infinity; neg_infinity; 0.; infinity |])));
      test "bounded activations saturate at large inputs" (fun () ->
          let input = vec [| 1e20; infinity; neg_infinity; 0. |] in
          check [| 6.; 6.; 0.; 0. |] (El.relu6 input);
          check [| 1.; 1.; 0.; 0.5 |] (El.hardsigmoid input));
      test "integer variance retains fractional squares" (fun () ->
          check [| 0.25 |]
            (Op.var ~correction:0 (Run.of_int_array ~shape:[ 2 ] [| 0; 1 |])));
      test "weak promotion preserves padding" (fun () ->
          let padded = Mv.pad (Mv.reshape (T.i 3) [ 1 ]) [ (1, 1) ] in
          check [| 1.; 4.; 1. |] (El.add padded (vec [| 1.; 1.; 1. |])));
      test "integer max pooling pads with the dtype minimum" (fun () ->
          let input = Run.of_int_array ~shape:[ 1; 1; 2; 2 ] [| -4; -3; -2; -1 |] in
          check_ints [| -4; -3; -3; -2; -1; -1; -2; -1; -1 |]
            (Op.max_pool2d ~stride:[ 1; 1 ] ~padding:[ 1 ] input));
      test "int64 max pooling preserves a full-width padding value" (fun () ->
          let data = Bytes.create 8 in
          Bytes.set_int64_le data 0 Int64.min_int;
          let input = Run.of_bytes ~dtype:Tolk_uop.Dtype.int64 ~shape:[ 1; 1; 1; 1 ] data in
          let out = Op.max_pool2d ~stride:[ 1; 1 ] ~padding:[ 1 ] input in
          let actual = Run.data out in
          equal int 32 (Bytes.length actual);
          for i = 0 to 3 do equal int64 Int64.min_int (Bytes.get_int64_le actual (8 * i)) done);
    ]

let lifetime_tests =
  group "lifetime"
    [
      test "empty host inputs need no native allocation" (fun () ->
          let floats = Run.of_float_array ~shape:[ 0; 3 ] [||] in
          let ints = Run.of_int_array ~shape:[ 2; 0 ] [||] in
          Run.realize_many [ floats; ints ];
          equal int 0 (Array.length (Run.to_float_array floats));
          equal int 0 (Array.length (Run.to_int_array ints));
          equal int 0 (Bytes.length (Run.data floats));
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> Run.of_float_array ~shape:[ 3 ] [||]));
      test "unreachable input and realized storage are collectible" (fun () ->
          let nodes = Stdlib.Weak.create 2 and buffers = Stdlib.Weak.create 2 in
          let[@inline never] populate () =
            let input = vec [| 1.; 2.; 3. |] in
            let output = Run.realize (El.add input (T.f 1.)) in
            List.iteri
              (fun i tensor ->
                let node = U.buf_uop (T.uop tensor) in
                Stdlib.Weak.set nodes i (Some node);
                Stdlib.Weak.set buffers i (Run.buffer_of_node node))
              [ input; output ]
          in
          populate ();
          Gc.full_major ();
          Gc.full_major ();
          for i = 0 to 1 do
            is_false ~msg:"unreachable node retained" (Stdlib.Weak.check nodes i);
            is_false ~msg:"unreachable buffer retained" (Stdlib.Weak.check buffers i)
          done);
      test "live slice retains its backing storage" (fun () ->
          let[@inline never] make_view () =
            let input = vec [| 1.; 2.; 3.; 4. |] in
            Mv.shrink input [ (1, 3) ]
          in
          let view = make_view () in
          Gc.full_major ();
          check_floats [| 2.; 3. |] view);
    ]

let constant_integer_division () =
  let module D = Tolk_uop.Dtype in
  Tolk.Helpers.Context_var.with_context
    [ B (Tolk.Helpers.disable_fast_idiv, 0) ] (fun () ->
      List.iter (fun (dtype, values) ->
          let bytes = Bytes.create (4 * Array.length values) in
          Array.iteri (fun i value -> Bytes.set_int32_le bytes (4 * i) (Int32.of_int value)) values;
          let input = Run.of_bytes ~dtype ~shape:[ Array.length values ] bytes in
          List.iter (fun divisor ->
              let d = T.i divisor in
              check_ints (Array.map (fun x -> x / divisor) values) (El.cdiv input d);
              check_ints (Array.map (fun x -> x mod divisor) values) (El.fmod input d);
              let quotient x = x / divisor - (if x mod divisor < 0 then 1 else 0) in
              check_ints (Array.map quotient values) (El.floordiv input d);
              check_ints (Array.map (fun x -> x - divisor * quotient x) values) (El.mod_ input d))
            [ 2; 3; 4; 6; 7; 8; 19; 65537; 2147483647 ])
        [ D.uint32, [| 0; 1; 2; 3; 6; 7; 18; 19; 20; 65536; 65537;
                       2147483647; 2147483648; 4294967294; 4294967295 |];
          D.int32, [| -2147483648; -65537; -20; -19; -7; -1; 0; 1; 7; 2147483647 |] ])

let sharding_tests =
  group "sharding"
    [
      test "partition, compute and gather" (fun () ->
          let device = Run.device_name () in
          let devices = [device; device] in
          let input = fa ~shape:[2; 4] [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|] in
          let sharded = Creation.shard ~axis:(-1) ~devices input in
          equal (list int) [2; 4] (T.shape sharded);
          equal (list int) [2; 2] (U.max_shard_shape (T.uop sharded));
          let result = Rd.sum ~axis:[1] (El.mul sharded sharded) in
          check_floats [|30.; 174.|] (Creation.clone ~device:(U.Single device) result));
      test "sharded and replicated tensors realize" (fun () ->
          let device = Run.device_name () in
          let devices = [device; device] in
          let sharded = Creation.shard ~axis:0 ~devices
              (fa ~shape:[2; 4] [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|]) in
          let total = Rd.sum ~axis:[0] sharded in
          Run.realize_many [sharded; total];
          check_floats [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|]
            (Creation.clone ~device:(U.Single device) sharded);
          check_floats [|6.; 8.; 10.; 12.|]
            (Creation.clone ~device:(U.Single device) total));
      test "replication preserves shape and values" (fun () ->
          let device = Run.device_name () in
          let input = vec [|1.; 2.; 3.|] in
          let replicated = Creation.shard ~devices:[device; device] input in
          is_true (U.axis (T.uop replicated) = None);
          check_floats [|2.; 4.; 6.|]
            (Creation.clone ~device:(U.Single device) (El.add replicated replicated)));
      test "single and device-less tensors need no partition" (fun () ->
          let input = vec [|1.; 2.|] in
          is_true (Creation.shard ~devices:[Run.device_name ()] input == input);
          let constant = T.i 7 in
          is_true (Creation.shard ~devices:["CPU:1"; "CPU:2"] constant == constant));
      test "invalid partitions fail before realization" (fun () ->
          let input = vec [|1.; 2.; 3.|] in
          let devices = [Run.device_name (); Run.device_name ()] in
          let invalid f = raises_match (function Invalid_argument _ -> true | _ -> false) f in
          invalid (fun () -> ignore (Creation.shard ~devices:[] input));
          invalid (fun () -> ignore (Creation.shard ~axis:1 ~devices input));
          invalid (fun () -> ignore (Creation.shard ~axis:0 ~devices input));
          let replicated = Creation.shard ~devices input in
          invalid (fun () ->
              ignore (Creation.shard ~devices:(Run.device_name () :: devices) replicated));
          let split = Creation.shard ~axis:0 ~devices (fa ~shape:[2; 2] [|1.; 2.; 3.; 4.|]) in
          invalid (fun () -> ignore (Creation.shard ~axis:0 ~devices split)));
      test "a replicated tensor splits where it lives" (fun () ->
          let device = Run.device_name () in
          let devices = [device; device] in
          let replicated =
            Creation.shard ~devices (fa ~shape:[2; 4] [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|]) in
          is_true (Creation.shard ~devices replicated == replicated);
          let split = Creation.shard ~axis:0 ~devices replicated in
          equal (list int) [2; 4] (T.shape split);
          equal (list int) [1; 4] (U.max_shard_shape (T.uop split));
          let shrink = (U.src (T.uop split)).(0) in
          is_true ~msg:"each device shrinks its own replica" ((U.src shrink).(0) == T.uop replicated);
          check_floats [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.|]
            (Creation.clone ~device:(U.Single device) split));
    ]

let () =
  run "Tolk_frontend_run"
    [
      test "DEV targets select the default device within a context" (fun () ->
          let original = Run.device_name () in
          Tolk.Helpers.Context_var.with_context
            [ B (Tolk.Helpers.dev,
                 [ Tolk_uop.Target.of_string "CPU:CLANG";
                   Tolk_uop.Target.of_string "PCI+NV" ]) ] (fun () ->
                equal string "CPU" (Run.device_name ());
                equal (array (float 1e-6)) [| 4.; 6. |]
                  (Run.to_float_array (El.add (vec [| 1.; 2. |]) (vec [| 3.; 4. |]))));
          equal string original (Run.device_name ()));
      test "constant integer division and modulo preserve boundary values"
        constant_integer_division;
      sharding_tests;
      aliasing_tests;
      lifetime_tests;
      numerical_edge_tests;
      constant_tests;
      elementwise_tests;
      select_tests;
      dynamic_select_tests;
      scatter_tests;
      scatter_indexed_tests;
      clone_tests;
      sort_tests;
      bitcast_tests;
      reduce_tests;
      matmul_tests;
      scan_tests;
      logspace_tests;
      getitem_tests;
      gpt2_getitem_tests;
      large_gather_tests;
      conv_tests;
      stack_tests;
      cat_tests;
      triu_tests;
      assign_tests;
      attention_tests;
      symbolic_tests;
    ]
