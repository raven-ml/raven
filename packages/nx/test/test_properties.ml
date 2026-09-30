(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Properties, conversions and copies. *)

open Windtrap
open Nx_test

let same = tensor int32
let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
let iota s = Array.init (Ref.numel s) (fun i -> Int32.of_int (i - 7))

let viewed =
  Gen.map
    (fun (s, steps) -> lay_out steps (Nx.create Nx.int32 s (iota s)))
    (Gen.pair shape layout)

(* A layout, printed as the steps that made it. *)
let laid_out =
  Gen.with_pp
    (fun ppf (steps, t) ->
      Format.fprintf ppf "%a, of shape %a" pp_layout steps pp_shape (Nx.shape t))
    (Gen.map
       (fun (s, steps) ->
         (steps, lay_out steps (Nx.create Nx.int32 s (iota s))))
       (Gen.pair shape layout))

let transposed_row =
  ([ List.nth layout_steps 0 ], Nx.transpose (Nx.zeros Nx.int32 [| 1; 3 |]))

let without_first =
  ( [ List.nth layout_steps 3 ],
    Nx.slice [ R (1, 3) ] (Nx.zeros Nx.int32 [| 3; 2 |]) )

let properties =
  group "properties"
    [
      prop "shape, ndim, dim and numel describe the shape" shape (fun s ->
          let t = Nx.zeros Nx.int32 s in
          equal (array int) s (Nx.shape t);
          equal int (Array.length s) (Nx.ndim t);
          equal int (Ref.numel s) (Nx.numel t);
          Array.iteri (fun i d -> equal int d (Nx.dim i t)) s);
      cases
        "itemsize is the width of the dtype, and nbytes counts every element"
        ~name:(fun (name, _, _) -> name)
        (let sizes dtype () =
           let t = Nx.zeros dtype [| 3 |] in
           (Nx.itemsize t, Nx.nbytes t)
         in
         [
           ("float16", 2, sizes Nx.float16);
           ("bfloat16", 2, sizes Nx.bfloat16);
           ("float32", 4, sizes Nx.float32);
           ("float64", 8, sizes Nx.float64);
           ("float8_e4m3", 1, sizes Nx.float8_e4m3);
           ("int8", 1, sizes Nx.int8);
           ("uint16", 2, sizes Nx.uint16);
           ("int32", 4, sizes Nx.int32);
           ("uint64", 8, sizes Nx.uint64);
           ("complex64", 8, sizes Nx.complex64);
           ("complex128", 16, sizes Nx.complex128);
           ("bool", 1, sizes Nx.bool);
         ])
        (fun (_, width, sizes) ->
          equal (pair int int) (width, 3 * width) (sizes ()));
      test "dim refuses an axis out of bounds" (fun () ->
          raises_invalid_arg (fun () -> Nx.dim 2 (Nx.zeros Nx.int32 [| 2; 2 |]));
          raises_invalid_arg (fun () ->
              Nx.dim (-3) (Nx.zeros Nx.int32 [| 2; 2 |])));
      cases "is_c_contiguous says whether elements run in row-major order"
        ~name:fst
        [
          ("a created tensor", (true, Fun.id));
          ("a transpose", (false, fun t -> Nx.transpose t));
          ( "a transpose transposed back",
            (true, fun t -> Nx.transpose (Nx.transpose t)) );
          ("a broadcast", (false, fun t -> Nx.broadcast_to [| 2; 2; 3 |] t));
          ("a flip", (false, fun t -> Nx.flip t));
        ]
        (fun (_, (expected, view)) ->
          equal bool expected
            (Nx.is_c_contiguous (view (Nx.zeros Nx.int32 [| 2; 3 |]))));
      prop ~examples:[ transposed_row ]
        "is_c_contiguous holds exactly when the elements follow each other in \
         the buffer"
        laid_out (fun (_, t) ->
          equal bool (consecutive t) (Nx.is_c_contiguous t));
    ]

let conversions =
  group "conversions"
    [
      prop "to_array lists the elements in row-major order" viewed (fun t ->
          equal (array int32) (Ref.of_layout t).data (Nx.to_array t));
      prop "to_bigarray copies, and of_bigarray reads it back" viewed
        (Law.round_trip same
           (Testable.contramap Nx.of_bigarray same)
           Nx.to_bigarray Nx.of_bigarray);
      prop "copy has the values and storage of its own" viewed (fun t ->
          let c = Nx.copy t in
          equal same t c;
          is_true ~msg:"copy is contiguous" (Nx.is_c_contiguous c);
          is_false ~msg:"copy shares no memory"
            (share_memory (storage c) (storage t)));
      prop "contiguous has the values, in a contiguous layout" viewed (fun t ->
          let c = Nx.contiguous t in
          equal same t c;
          is_true (Nx.is_c_contiguous c));
      test "contiguous of a contiguous tensor shares its storage" (fun () ->
          let t = Nx.zeros Nx.int32 [| 2; 3 |] in
          is_true (storage (Nx.contiguous t) == storage t));
      prop ~examples:[ without_first ]
        "contiguous shares the storage of a C-contiguous tensor at offset 0, \
         and copies one past it"
        laid_out (fun (_, t) ->
          if Nx.is_c_contiguous t then
            equal bool
              (Nx_array.View.offset (view t) = 0)
              (storage (Nx.contiguous t) == storage t));
      test "to_bigarray copies, so writing the bigarray leaves the tensor"
        (fun () ->
          let t = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
          Bigarray.Genarray.set (Nx.to_bigarray t) [| 0 |] 99l;
          equal same (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]) t);
      test "to_bigarray refuses a dtype Bigarray has no kind for" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.to_bigarray (Nx.zeros Nx.bfloat16 [| 2 |])));
      test "of_bigarray is over the bigarray's memory" (fun () ->
          let ba =
            Bigarray.Array1.init Bigarray.int32 Bigarray.c_layout 6 Int32.of_int
          in
          let t = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          equal nativeint
            (Nx_device.Buffer.address (Nx_device.Buffer.of_bigarray ba))
            (Nx_device.Buffer.address (storage t)));
      test "of_bigarray refuses the kinds that are no dtype" (fun () ->
          let refuses (type a b) (k : (a, b) Bigarray.kind) =
            raises_invalid_arg (fun () ->
                Nx.of_bigarray
                  (Bigarray.Genarray.create k Bigarray.c_layout [| 1 |]))
          in
          refuses Bigarray.char;
          refuses Bigarray.int;
          refuses Bigarray.nativeint);
    ]

let iteration =
  group "iteration"
    [
      prop "map_item maps and fold_item folds each element in row-major order"
        viewed (fun t ->
          let r = Ref.of_layout t in
          equal (Ref.witness int32)
            (Ref.map (Int32.mul 3l) r)
            (Ref.of_nx (Nx.map_item (Int32.mul 3l) t));
          equal (list int32) (Array.to_list r.data)
            (List.rev (Nx.fold_item (fun acc x -> x :: acc) [] t)));
      prop
        "iter_item visits each element in row-major order (nx.mli is silent on \
         the order)"
        viewed (fun t ->
          let seen = ref [] in
          Nx.iter_item (fun x -> seen := x :: !seen) t;
          equal (list int32) (Array.to_list (Nx.to_array t)) (List.rev !seen));
    ]

let bits = Gen.map Int32.float_of_bits Gen.int32

let casts =
  group "casts"
    [
      prop "cast to float64 and back is the identity on int32" viewed
        (Law.round_trip same (tensor float_exact) (Nx.cast Nx.float64)
           (Nx.cast Nx.int32));
      prop "cast from a float to an integer truncates toward zero"
        (Gen.array ~size:(Gen.int_range 0 8) (Gen.float_range (-1e6) 1e6))
        (fun xs ->
          equal (array int32)
            (Array.map Int32.of_float xs)
            (Nx.to_array
               (Nx.cast Nx.int32
                  (Nx.create Nx.float64 [| Array.length xs |] xs))));
      test "cast to the tensor's own dtype is the tensor" (fun () ->
          let t = Nx.zeros Nx.float32 [| 3 |] in
          is_true (Nx.cast Nx.float32 t == t));
      prop
        "bitcast from float32 to int32 reads each element's bits, in its place"
        (Gen.array ~size:(Gen.constant 6) bits)
        (fun xs ->
          assume (Array.for_all (fun x -> not (Float.is_nan x)) xs);
          let t = Nx.transpose (Nx.create Nx.float32 [| 2; 3 |] xs) in
          equal (tensor int32)
            (Nx.transpose
               (Nx.create Nx.int32 [| 2; 3 |]
                  (Array.map Int32.bits_of_float xs)))
            (Nx.bitcast Nx.int32 t));
      prop "bitcast to int32 and back keeps every float32, NaN included"
        (Gen.array ~size:(Gen.int_range 0 6) bits)
        (fun xs ->
          Law.round_trip (tensor float_exact) (tensor int32)
            (Nx.bitcast Nx.int32) (Nx.bitcast Nx.float32)
            (Nx.create Nx.float32 [| Array.length xs |] xs));
      test "bitcast refuses dtypes of other widths, bool, and packed int4"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.bitcast Nx.int16 (Nx.zeros Nx.float32 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.bitcast Nx.bool (Nx.zeros Nx.uint8 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.bitcast Nx.int8 (Nx.zeros Nx.int4 [| 2 |])));
    ]

(* A float truncated toward zero, held at the ends of the range, NaN at 0. *)
let saturate ~bits ~signed x =
  let lo, hi = int_range ~bits ~signed in
  let t = Float.trunc x in
  let two_63 = 0x1p63 in
  if Float.is_nan x then 0L
  else if signed then
    if t <= Int64.to_float lo then lo
    else if t >= if bits = 64 then two_63 else Int64.to_float hi then hi
    else Int64.of_float t
  else if t <= 0. then 0L
  else if bits = 64 then
    if t >= 0x1p64 then hi
    else if t >= two_63 then
      Int64.add (Int64.of_float (t -. two_63)) Int64.min_int
    else Int64.of_float t
  else if t >= Int64.to_float hi then hi
  else Int64.of_float t

let integer_casts =
  group "casts to integers"
    (List.map
       (fun (Int_dtype d) ->
         prop
           ("cast from float64 to " ^ d.name
          ^ " truncates, holds at the range and takes NaN to 0 (nx.mli is \
             silent)")
           (Gen.array ~size:(Gen.int_range 0 8)
              (Gen.one_of [ Gen.any_float; Gen.float_range (-300.) 300. ]))
           (fun xs ->
             equal (array d.exact)
               (Array.map
                  (fun x -> d.of_i64 (saturate ~bits:d.bits ~signed:d.signed x))
                  xs)
               (Nx.to_array
                  (Nx.cast d.dtype
                     (Nx.create Nx.float64 [| Array.length xs |] xs)))))
       int_dtypes
    @ [
        prop "int4 and uint4 hold every value of their range through a cast"
          (Gen.pair
             (Gen.array ~size:(Gen.int_range 0 8) (Gen.int_range (-8) 7))
             (Gen.array ~size:(Gen.int_range 0 8) (Gen.int_range 0 15)))
          (fun (s, u) ->
            let through narrow wide v =
              Nx.to_array
                (Nx.cast wide
                   (Nx.cast narrow (Nx.create wide [| Array.length v |] v)))
            in
            equal (array int) s (through Nx.int4 Nx.int8 s);
            equal (array int) u (through Nx.uint4 Nx.uint8 u));
        test
          "a real value is complex with no imaginary part, and back it drops \
           that part" (fun () ->
            let z =
              Nx.cast Nx.complex128
                (Nx.create Nx.float64 [| 2 |] [| 1.5; -2. |])
            in
            equal
              (array (pair float_exact float_exact))
              [| (1.5, 0.); (-2., 0.) |]
              (Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array z));
            let r =
              Nx.cast Nx.float64
                (Nx.create Nx.complex128 [| 1 |] [| { re = 3.; im = 9. } |])
            in
            equal (array float_exact) [| 3. |] (Nx.to_array r));
      ])

(* A bitcast to a dtype of the same width and back keeps every bit. *)
type same_width = W : string * ('a, 'b) Nx.dtype -> same_width

let bitcasts =
  let round_trip (type a b) name (source : (a, b) Nx.dtype) (w : a testable)
      (value : a Gen.t) widths =
    List.map
      (fun (W (dname, dt)) ->
        prop
          (Printf.sprintf "bitcast from %s to %s and back keeps every bit" name
             dname)
          (Gen.array ~size:(Gen.int_range 0 8) value)
          (fun xs ->
            let t = Nx.create source [| Array.length xs |] xs in
            equal (tensor w) t (Nx.bitcast source (Nx.bitcast dt t))))
      widths
  in
  group "bitcasts"
    (round_trip "uint8" Nx.uint8 int (Gen.int_range 0 255)
       [
         W ("int8", Nx.int8);
         W ("float8_e4m3", Nx.float8_e4m3);
         W ("float8_e5m2", Nx.float8_e5m2);
       ]
    @ round_trip "uint16" Nx.uint16 int (Gen.int_range 0 65535)
        [
          W ("int16", Nx.int16);
          W ("float16", Nx.float16);
          W ("bfloat16", Nx.bfloat16);
        ]
    @ round_trip "uint32" Nx.uint32 int32 Gen.int32
        [ W ("int32", Nx.int32); W ("float32", Nx.float32) ]
    @ round_trip "uint64" Nx.uint64 int64 Gen.int64
        [
          W ("int64", Nx.int64);
          W ("float64", Nx.float64);
          W ("complex64", Nx.complex64);
        ])

let () =
  exit
    (run "nx properties"
       [ properties; conversions; iteration; casts; integer_casts; bitcasts ])
