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
      xfail
        ~reason:
          "is_c_contiguous compares every stride with its row-major value, so \
           it says false for a unit axis of another stride, as in the \
           transpose of a [1; 3], and for a tensor with no elements, even a \
           created [0]"
        (prop ~examples:[ transposed_row ]
           "is_c_contiguous holds exactly when the elements follow each other \
            in the buffer"
           laid_out (fun (_, t) ->
             equal bool (consecutive t) (Nx.is_c_contiguous t)));
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
      prop "to_buffer and of_buffer round trip through the shape" viewed
        (fun t ->
          Law.round_trip same pass Nx.to_buffer
            (Nx.of_buffer ~shape:(Nx.shape t))
            t);
      prop "copy has the values and storage of its own" viewed (fun t ->
          let c = Nx.copy t in
          equal same t c;
          is_true ~msg:"copy is contiguous" (Nx.is_c_contiguous c);
          is_false ~msg:"copy shares no storage"
            (Nx.numel t > 0 && Nx.data c == Nx.data t));
      prop "contiguous has the values, in a contiguous layout" viewed (fun t ->
          let c = Nx.contiguous t in
          equal same t c;
          is_true (Nx.is_c_contiguous c));
      test "contiguous of a contiguous tensor shares its storage" (fun () ->
          let t = Nx.zeros Nx.int32 [| 2; 3 |] in
          is_true (Nx.data (Nx.contiguous t) == Nx.data t));
      xfail
        ~reason:
          "contiguous copies a C-contiguous view whose offset is not 0, such \
           as a tensor without its first row"
        (prop ~examples:[ without_first ]
           "contiguous shares the storage of every C-contiguous tensor" laid_out
           (fun (_, t) ->
             if Nx.is_c_contiguous t then
               is_true (Nx.data (Nx.contiguous t) == Nx.data t)));
      xfail
        ~reason:
          "to_bigarray wraps the tensor's own buffer when it is contiguous \
           from offset 0"
        (test "to_bigarray copies, so writing the bigarray leaves the tensor"
           (fun () ->
             let t = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
             Bigarray.Genarray.set (Nx.to_bigarray t) [| 0 |] 99l;
             equal same (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]) t));
      test "to_bigarray refuses a dtype Bigarray has no kind for" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.to_bigarray (Nx.zeros Nx.bfloat16 [| 2 |])));
      prop "to_buffer holds exactly the elements" viewed (fun t ->
          equal int (Nx.numel t) (Nx_buffer.length (Nx.to_buffer t)));
      test
        "to_buffer is the tensor's own buffer when it is contiguous from \
         offset 0" (fun () ->
          let t = Nx.zeros Nx.int32 [| 2; 3 |] in
          is_true (Nx.to_buffer t == Nx.data t));
      test "of_buffer refuses a shape of another size (nx.mli states no error)"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.of_buffer
                (Nx.to_buffer (Nx.zeros Nx.int32 [| 4 |]))
                ~shape:[| 3 |]));
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

let () = exit (run "nx properties" [ properties; conversions; casts ])
