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
        prop "a cast reads a strided view of int4 and of uint4"
          (Gen.array ~size:(Gen.int_range 0 9) (Gen.int_range (-8) 7))
          (fun v ->
            let n = Array.length v in
            let read narrow wide v =
              let every_other =
                Nx.slice
                  [ Rs (0, n, 2) ]
                  (Nx.flip (Nx.cast narrow (Nx.create wide [| n |] v)))
              in
              Nx.to_array (Nx.cast wide every_other)
            in
            let expected v =
              Array.of_list
                (List.filteri
                   (fun i _ -> i mod 2 = 0)
                   (List.rev (Array.to_list v)))
            in
            equal (array int) (expected v) (read Nx.int4 Nx.int8 v);
            let u = Array.map (fun x -> x + 8) v in
            equal (array int) (expected u) (read Nx.uint4 Nx.uint8 u));
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

(* Bitcasts. Their law is on bytes: the result's elements in row-major order
   hold the operand's bytes in row-major order, so a round trip in either
   direction gives the operand back. *)

type target = T : ('a, 'b) Nx.dtype -> target

let targets =
  [
    T Nx.uint8;
    T Nx.uint16;
    T Nx.float16;
    T Nx.uint32;
    T Nx.float32;
    T Nx.uint64;
    T Nx.float64;
    T Nx.complex128;
  ]

(* Rows of [k] elements from the elements of [k] of [tensors]'s draws, about as
   many rows as a draw has elements, under a layout that keeps each row
   whole. *)
let rows_of k tensors =
  let drawn =
    let open Gen in
    let* ts = list ~size:(constant k) tensors in
    let+ steps = row_layout in
    let t = Nx.concatenate ~axis:0 (List.map Nx.flatten ts) in
    let n = Nx.numel t / k in
    lay_out steps (Nx.reshape [| n; k |] (Nx.slice [ R (0, n * k) ] t))
  in
  Gen.with_pp (fun ppf t -> Stored.pp_packed ppf (Nx.P t)) drawn

let reads_the_bytes (Stored.Case c) (T dt) =
  let w = Nx_dtype.itemsize c.dtype and w' = Nx_dtype.itemsize dt in
  let tensors = if w' > w then rows_of (w' / w) c.tensors else c.tensors in
  prop
    (Printf.sprintf "bitcast from %s to %s keeps the bytes in row-major order"
       c.name (Nx_dtype.to_string dt))
    tensors
    (fun t ->
      let _, shape, bytes = Stored.storage (Nx.P t) in
      let r = Array.length shape in
      let expected =
        if w' > w then Array.sub shape 0 (r - 1)
        else if w' < w then Array.append shape [| w / w' |]
        else shape
      in
      if w' > w then
        cover "a widening of two rows or more" (Nx.numel t >= 2 * (w' / w));
      let _, shape', bytes' = Stored.storage (Nx.P (Nx.bitcast dt t)) in
      equal ~msg:"shape" (array int) expected shape';
      equal ~msg:"bytes" string bytes bytes')

let shares t u = Nx_test.share_memory (storage t) (storage u)

let bitcasts =
  group "bitcasts"
    (List.concat_map
       (fun (Stored.Case c as case) ->
         if c.name = "bool" then [] else List.map (reads_the_bytes case) targets)
       Stored.every
    @ [
        test "a bitcast reads a C-contiguous value in place" (fun () ->
            let t = Nx.create Nx.uint8 [| 3; 8 |] (Array.init 24 Fun.id) in
            is_true ~msg:"the whole matrix" (shares (Nx.bitcast Nx.uint64 t) t);
            let rows = Nx.slice [ R (1, 3) ] t in
            let words = Nx.bitcast Nx.uint64 rows in
            is_true ~msg:"its last two rows" (shares words rows);
            equal ~msg:"their words" (array int64)
              [| 0x0f0e0d0c0b0a0908L; 0x1716151413121110L |]
              (Nx.to_array words);
            let f = Nx.transpose (Nx.ones Nx.float32 [| 2; 3 |]) in
            is_true ~msg:"a narrowing of a transposed value"
              (shares (Nx.bitcast Nx.uint8 f) f));
        test
          "a bitcast copies a widening that is not C-contiguous from an \
           aligned address" (fun () ->
            let t = Nx.create Nx.uint8 [| 3; 16 |] (Array.init 48 Fun.id) in
            let cols =
              Nx.squeeze ~axes:[ -1 ]
                (Nx.sliding_window ~axis:1 ~window:1 ~step:2 t)
            in
            is_false ~msg:"every other column"
              (shares (Nx.bitcast Nx.uint64 cols) cols);
            let bytes = Nx.create Nx.uint8 [| 16 |] (Array.init 16 Fun.id) in
            let unaligned = Nx.slice [ R (3, 11) ] bytes in
            let word = Nx.bitcast Nx.uint64 unaligned in
            is_false ~msg:"eight bytes from the third" (shares word unaligned);
            equal ~msg:"their word" int64 0x0a09080706050403L (Nx.item [] word));
        test "a widening is in place exactly from an aligned first element"
          (fun () ->
            let ba =
              Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout 19
            in
            for i = 0 to 18 do
              ba.{i} <- i
            done;
            let t =
              Nx.of_bigarray
                (Bigarray.genarray_of_array1 (Bigarray.Array1.sub ba 3 16))
            in
            let odd = Nx.reshape [| 2; 8 |] t in
            let words = Nx.bitcast Nx.uint64 odd in
            is_false ~msg:"from an odd address" (shares words odd);
            equal ~msg:"their words" (array int64)
              [| 0x0a09080706050403L; 0x1211100f0e0d0c0bL |]
              (Nx.to_array words);
            let even = Nx.slice [ R (5, 13) ] t in
            is_true ~msg:"from an aligned element at an odd offset"
              (shares (Nx.bitcast Nx.uint64 even) even));
        test
          "bitcast refuses bool, packed int4, and a widening without a last \
           axis of the ratio" (fun () ->
            raises_invalid_arg (fun () ->
                Nx.bitcast Nx.int64 (Nx.zeros Nx.float32 [| 3 |]));
            raises_invalid_arg (fun () ->
                Nx.bitcast Nx.int64 (Nx.scalar Nx.float32 0.));
            raises_invalid_arg (fun () ->
                Nx.bitcast Nx.uint64 (Nx.zeros Nx.uint8 [| 2; 4 |]));
            raises_invalid_arg (fun () ->
                Nx.bitcast Nx.bool (Nx.zeros Nx.uint8 [| 2 |]));
            raises_invalid_arg (fun () ->
                Nx.bitcast Nx.int8 (Nx.zeros Nx.int4 [| 2 |])));
      ])

let () =
  exit
    (run "nx properties"
       [ properties; conversions; iteration; casts; integer_casts; bitcasts ])
