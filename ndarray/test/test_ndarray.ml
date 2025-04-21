open Ndarray

let test _test_name operation_desc f =
  (* Printf.printf "Test: %s\n" test_name; *)
  Printf.printf "> %s\n" operation_desc;
  let t = f () in
  print t;
  Printf.printf "\n"

let test_custom _test_name operation_desc f =
  (* Printf.printf "Test: %s\n" test_name; *)
  Printf.printf "> %s\n" operation_desc;
  f ();
  Printf.printf "\n"

let test_f _test_name operation_desc f =
  (* Printf.printf "Test: %s\n" test_name; *)
  Printf.printf "> %s\n" operation_desc;
  try
    let _ = f () in
    Printf.printf "Completed without error.\n\n"
  with
  | Failure msg | Invalid_argument msg -> Printf.printf "%s\n\n" msg
  | e -> Printf.printf "%s\n\n" (Printexc.to_string e)

let () = Printexc.record_backtrace true

let () =
  (* === Creation Tests === *)
  test "Create 2x2 float32 ndarray"
    "create float32 [2, 2] [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]);

  test "Create 1D int32 ndarray" "create int32 [3] [1, 2, 3]" (fun () ->
      create int32 [| 3 |] [| 1l; 2l; 3l |]);

  test "Create empty float32 ndarray" "create float32 [0] []" (fun () ->
      create float32 [| 0 |] [||]);

  test "Create 2x2x2 high-dimensional ndarray"
    "create float32 [2, 2, 2] [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, \
     7.0]]]" (fun () ->
      create float32 [| 2; 2; 2 |] (Array.init 8 float_of_int));

  test "Create scalar float32" "scalar float32 42.0" (fun () ->
      scalar float32 42.0);

  test "Create scalar int64" "scalar int64 100L" (fun () -> scalar int64 100L);

  test_custom "Convert to Bigarray" "to_bigarray float32 [2, 2]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let ba = to_bigarray t in
      Printf.printf "Converted Bigarray:\n%f %f\n%f %f\n"
        (Bigarray.Genarray.get ba [| 0; 0 |])
        (Bigarray.Genarray.get ba [| 0; 1 |])
        (Bigarray.Genarray.get ba [| 1; 0 |])
        (Bigarray.Genarray.get ba [| 1; 1 |]);
      set_item [| 0; 0 |] 55.0 t;
      Printf.printf "Original modified, Bigarray view:\n%f %f\n%f %f\n"
        (Bigarray.Genarray.get ba [| 0; 0 |])
        (Bigarray.Genarray.get ba [| 0; 1 |])
        (Bigarray.Genarray.get ba [| 1; 0 |])
        (Bigarray.Genarray.get ba [| 1; 1 |]));

  test_custom "Copy ndarray" "copy [1.0, 2.0, 3.0]" (fun () ->
      let original = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let copy_arr = copy original in
      set_item [| 0 |] 10.0 original;
      Printf.printf "Original after modification: ";
      print original;
      Printf.printf "Copy (should be unchanged): ";
      print copy_arr);

  test "Create empty (uninitialized) 2x2 float32" "empty float32 [2, 2]"
    (fun () ->
      (* Contents are undefined, just check shape/type *)
      empty float32 [| 2; 2 |]);

  test "Fill 2x2 float32 with 7.0" "fill 7.0 (empty float32 [2, 2])" (fun () ->
      let t = empty float32 [| 2; 2 |] in
      fill 7.0 t;
      t);

  test_custom "Blit 1D to 1D" "blit [1., 2.] to [0., 0.]" (fun () ->
      let src = create float32 [| 2 |] [| 1.0; 2.0 |] in
      let dst = zeros float32 [| 2 |] in
      Printf.printf "Destination before blit: ";
      print dst;
      blit src dst;
      Printf.printf "Destination after blit: ";
      print dst);

  test_f "Blit incompatible shapes" "blit [1., 2.] to [0., 0., 0.]" (fun () ->
      let src = create float32 [| 2 |] [| 1.0; 2.0 |] in
      let dst = zeros float32 [| 3 |] in
      blit src dst);

  test "Create full 2x3 float32 with 5.5" "full float32 [2, 3] 5.5" (fun () ->
      full float32 [| 2; 3 |] 5.5);

  test "Create full_like 2x2 from int32" "full_like 10l [[1l, 2l], [3l, 4l]]"
    (fun () ->
      let t_ref = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      full_like 10l t_ref);

  test "Create empty_like from 2x2 float64" "empty_like [[1., 2.], [3., 4.]]"
    (fun () ->
      let t_ref = create float64 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      empty_like t_ref);

  test "Create zeros 2x2 float32" "zeros float32 [2, 2]" (fun () ->
      zeros float32 [| 2; 2 |]);

  test "Create zeros_like from 2x2 array" "zeros_like [[1.0, 2.0], [3.0, 4.0]]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      zeros_like t);

  test "Create ones 2x2 float32" "ones float32 [2, 2]" (fun () ->
      ones float32 [| 2; 2 |]);

  test "Create ones_like from 2x2 array" "ones_like [[1l, 2l], [3l, 4l]]"
    (fun () ->
      let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      ones_like t);

  test "Create identity 3x3 float32" "identity float32 3" (fun () ->
      identity float32 3);

  test "Create identity 1x1 int32" "identity int32 1" (fun () ->
      identity int32 1);

  test "Create eye 2x2 float32" "eye float32 2" (fun () -> eye float32 2);

  test "Create eye 3x4 float32" "eye ~m:3 float32 4" (fun () ->
      eye ~m:3 float32 4);

  test "Create eye 4x3 k=1 float32" "eye ~m:4 ~k:1 float32 3" (fun () ->
      eye ~m:4 ~k:1 float32 3);

  test "Create eye 3x3 k=-1 int32" "eye ~k:(-1) int32 3" (fun () ->
      eye ~k:(-1) int32 3);

  test "Create arange 0 to 10 step 2 int32" "arange int32 0 10 2" (fun () ->
      arange int32 0 10 2);

  test "Create arange_f 0.0 to 5.0 step 0.5 float32"
    "arange_f float32 0.0 5.0 0.5" (fun () -> arange_f float32 0.0 5.0 0.5);

  test "Create linspace 2.0 to 3.0 with 5 points float32"
    "linspace float32 2.0 3.0 5" (fun () -> linspace float32 2.0 3.0 5);

  test "Create linspace 0 to 4 with 5 points no endpoint int32"
    "linspace ~endpoint:false int32 0.0 4.0 5" (fun () ->
      linspace ~endpoint:false int32 0.0 4.0 5);

  test "Create logspace base 10 from 2.0 to 3.0 with 4 points float32"
    "logspace float32 ~base:10.0 2.0 3.0 4" (fun () ->
      logspace float32 ~base:10.0 2.0 3.0 4);

  test "Create logspace base 2 from 0 to 4 with 5 points no endpoint float64"
    "logspace ~endpoint:false ~base:2.0 float64 0.0 4.0 5" (fun () ->
      logspace ~endpoint:false ~base:2.0 float64 0.0 4.0 5);

  test "Create logspace default base 1 to 3 with 3 points float32"
    "logspace float32 1.0 3.0 3" (fun () -> logspace float32 1.0 3.0 3);

  test "Create geomspace 2.0 to 32.0 with 5 points float32"
    "geomspace float32 2.0 32.0 5" (fun () -> geomspace float32 2.0 32.0 5);

  test "Create geomspace 1 to 256 with 9 points no endpoint float64"
    "geomspace ~endpoint:false float64 1.0 256.0 9" (fun () ->
      geomspace ~endpoint:false float64 1.0 256.0 9);

  (* === Property Tests === *)
  test_custom "Get shape of 2x3 ndarray" "shape [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      let s = shape t in
      Printf.printf "[%s]\n"
        (String.concat ", " (Array.map string_of_int s |> Array.to_list)));

  test_custom "Get strides of 2x3 float32 ndarray" "strides float32 [2, 3]"
    (fun () ->
      let t = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      let st = strides t in
      Printf.printf "[%s]\n"
        (String.concat ", " (Array.map string_of_int st |> Array.to_list)));

  test_custom "Get stride dim 0 of 2x3 float32 ndarray"
    "stride 0 float32 [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      Printf.printf "%d\n" (stride 0 t));

  test_custom "Get stride dim 1 of 2x3 float32 ndarray"
    "stride 1 float32 [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      Printf.printf "%d\n" (stride 1 t));

  test_custom "Get strides of 2x3 int64 ndarray" "strides int64 [2, 3]"
    (fun () ->
      let t = create int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
      let st = strides t in
      Printf.printf "[%s]\n"
        (String.concat ", " (Array.map string_of_int st |> Array.to_list)));

  test_custom "Check dtype of 2x2 float32 ndarray" "dtype float32 [2, 2]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%s\n" (match dtype t with Float32 -> "Float32"));

  test_custom "Check dtype of 1D complex64 ndarray" "dtype complex64 [3]"
    (fun () ->
      let t =
        create complex64 [| 3 |]
          [|
            Complex.{ re = 1.; im = 2. };
            { re = 3.; im = 4. };
            { re = 5.; im = 6. };
          |]
      in
      Printf.printf "%s\n" (match dtype t with Complex64 -> "Complex64"));

  test_custom "Check itemsize of 2x2 float32 ndarray" "itemsize float32 [2, 2]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%d\n" (itemsize t));

  test_custom "Check itemsize of 2x2 int64 ndarray" "itemsize int64 [2, 2]"
    (fun () ->
      let t = create int64 [| 2; 2 |] [| 1L; 2L; 3L; 4L |] in
      Printf.printf "%d\n" (itemsize t));

  test_custom "Get ndim of scalar" "ndim []" (fun () ->
      let t = scalar float32 1.0 in
      Printf.printf "%d\n" (ndim t));

  test_custom "Get ndim of 2x2 array" "ndim [2, 2]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%d\n" (ndim t));

  test_custom "Get dim 0 of 2x3 array" "dim 0 [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      Printf.printf "%d\n" (dim 0 t));

  test_f "Get dim invalid of 2x3 array" "dim 2 [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      dim 2 t);

  test_custom "Get dims of 2x3 array" "dims [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      let d = dims t in
      Printf.printf "[%s]\n"
        (String.concat ", " (Array.to_list (Array.map string_of_int d))));

  test_custom "Get nbytes of 2x2 float32 array" "nbytes float32 [2, 2]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%d\n" (nbytes t));

  test_custom "Get nbytes of 2x3 int64 array" "nbytes int64 [2, 3]" (fun () ->
      let t = create int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
      Printf.printf "%d\n" (nbytes t));

  test_custom "Get size of 2x3 array" "size [2, 3]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      Printf.printf "%d\n" (size t));

  test_custom "Get size of scalar" "size []" (fun () ->
      let t = scalar float32 10.0 in
      Printf.printf "%d\n" (size t));

  test_custom "Get offset of basic array" "offset basic [2, 2]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%d\n" (offset t));

  test_custom "Get offset of slice" "offset slice [1:, 1:] of [3, 3]" (fun () ->
      let t = create float32 [| 3; 3 |] (Array.init 9 float_of_int) in
      let s = slice [| 1; -1 |] [| 1; -1 |] t in
      (* Assuming -1 means end *)
      Printf.printf "Original:\n";
      print t;
      Printf.printf "Slice:\n";
      print s;
      Printf.printf "Offset of slice: %d\n" (offset s));

  test_custom "Underlying data buffer view" "data view check" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let d = data t in
      Printf.printf "Original: ";
      print t;
      Bigarray.Array1.set d 0 99.0;
      Printf.printf "After modifying data buffer view: ";
      print t);

  (* === Element Access Tests === *)
  test_custom "Get item from 2x2 ndarray"
    "get_item [[1.0, 2.0], [3.0, 4.0]] [0, 1]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let e = get_item [| 0; 1 |] t in
      Printf.printf "%f\n" e);

  (* This test might be misleading as get_item expects indices matching ndim *)
  (* test_custom "Get item with flat index" "get [1.0, 2.0, 3.0, 4.0] [1]"
    (fun () ->
      let t = create float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%f" (get_item [| 1 |] t)); *)
  test_custom "Get item with multi-dim index"
    "get_item [[1.0, 2.0], [3.0, 4.0]] [0, 1]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Printf.printf "%f\n" (get_item [| 0; 1 |] t));

  test "Set element in 2x2 ndarray"
    "set_item [[1.0, 2.0], [3.0, 4.0]] [1, 0] 5.0" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      set_item [| 1; 0 |] 5.0 t;
      t);

  test_f "Out-of-bounds get_item" "get_item [[1.0, 2.0], [3.0, 4.0]] [2, 0]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      get_item [| 2; 0 |] t);

  test_f "Out-of-bounds set_item" "set_item [[1.0, 2.0], [3.0, 4.0]] [0, 2] 5.0"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      set_item [| 0; 2 |] 5.0 t);

  test "Get view with index" "get [0] from [[1,2],[3,4]]" (fun () ->
      let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      get [| 0 |] t);

  test "Get view scalar" "get [1, 1] from [[1,2],[3,4]]" (fun () ->
      let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      get [| 1; 1 |] t);

  test_custom "Set view with index" "set [0] of [[1,2],[3,4]] to [8,9]"
    (fun () ->
      let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      let v = create int32 [| 2 |] [| 8l; 9l |] in
      Printf.printf "Original:\n";
      print t;
      set [| 0 |] v t;
      Printf.printf "After set [0]:\n";
      print t);

  test_custom "Set view scalar" "set [1, 0] of [[1,2],[3,4]] to 99" (fun () ->
      let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      let v = scalar int32 99l in
      Printf.printf "Original:\n";
      print t;
      set [| 1; 0 |] v t;
      Printf.printf "After set [1, 0]:\n";
      print t);

  (* === Array Manipulation === *)
  test "Flatten 2x2 array" "flatten [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      flatten t);

  test_custom "Ravel contiguous array (view)" "ravel [[1.,2.],[3.,4.]]"
    (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let r = ravel t in
      Printf.printf "Raveled:\n";
      print r;
      set_item [| 0; 0 |] 99.0 t;
      (* Modify original *)
      Printf.printf "Original modified:\n";
      print t;
      Printf.printf "Raveled view (should reflect change):\n";
      print r);

  test_custom "Ravel non-contiguous array (copy)"
    "ravel (transpose [[1.,2.],[3.,4.]])" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let tr = transpose t in
      let r = ravel tr in
      Printf.printf "Original Transposed:\n";
      print tr;
      Printf.printf "Raveled:\n";
      print r;
      set_item [| 0; 0 |] 99.0 t;
      (* Modify original, affects transpose *)
      Printf.printf "Original modified, Transposed view updated:\n";
      print tr;
      Printf.printf "Raveled copy (should NOT reflect change):\n";
      print r);

  test "Reshape 1D to 2x2" "reshape [1.0, 2.0, 3.0, 4.0] [2, 2]" (fun () ->
      let t = create float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      reshape [| 2; 2 |] t);

  test_custom "Reshape view check" "reshape [4] to [2, 2] (view?)" (fun () ->
      let t = create float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let r = reshape [| 2; 2 |] t in
      Printf.printf "Reshaped:\n";
      print r;
      set_item [| 0 |] 99.0 t;
      (* Modify original *)
      Printf.printf "Original modified:\n";
      print t;
      Printf.printf "Reshaped view (should reflect change if view):\n";
      print r);

  test "Reshape to vector" "reshape [1.0, 2.0, 3.0, 4.0] [4, 1]" (fun () ->
      let t = create float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      reshape [| 4; 1 |] t);

  test_f "Reshape incompatible" "reshape [1.0, 2.0, 3.0] [2, 2]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      reshape [| 2; 2 |] t);

  test "Squeeze basic 1x2x1x3 array" "squeeze [1, 2, 1, 3]" (fun () ->
      let t =
        create float32 [| 1; 2; 1; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
      in
      squeeze t);

  test "Squeeze specific axis" "squeeze ~axes:[0, 2] [1, 2, 1, 3]" (fun () ->
      let t =
        create float32 [| 1; 2; 1; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
      in
      squeeze ~axes:[| 0; 2 |] t);

  (* todo *)
  (* test "Squeeze axis not size 1" "squeeze ~axes:[1] [1, 2, 1, 3]" (fun () ->
      let t =
        create float32 [| 1; 2; 1; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
      in
      squeeze ~axes:[| 1 |] t); *)
  test "Squeeze multiple 1x1x1 array" "squeeze [1, 1, 1]" (fun () ->
      let t = create float32 [| 1; 1; 1 |] [| 42.0 |] in
      squeeze t);

  test "Squeeze no-op 2x2 array" "squeeze [2, 2]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      squeeze t);

  test "Slice 3x4 array, rows 1-2, cols 0-3" "slice [1, 3] [0, 4] from 3x4"
    (fun () ->
      let t = create float32 [| 3; 4 |] (Array.init 12 float_of_int) in
      Printf.printf "Original:\n";
      print t;
      Printf.printf "Slice view (should reflect change):\n";
      slice [| 1; 0 |] [| 3; 4 |] t);

  (* OCaml slicing might be different *)
  test "Slice 3x4 array, rows 0-3 step 2, cols 0-4 step 2"
    "slice ~steps:[2, 2] [0, 3] [0, 4] from 3x4" (fun () ->
      let t = create float32 [| 3; 4 |] (Array.init 12 float_of_int) in
      Printf.printf "Original:\n";
      print t;
      Printf.printf "Slice view (should reflect change):\n";
      slice ~steps:[| 2; 2 |] [| 0; 0 |] [| 3; 4 |] t);

  test_custom "Slice view check" "slice [1:2, :] from 3x2 (view?)" (fun () ->
      let t = create float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      let s = slice [| 1; 0 |] [| 2; 2 |] t in
      Printf.printf "Original:\n";
      print t;
      Printf.printf "Slice:\n";
      print s;
      set_item [| 1; 0 |] 99.0 t;
      (* Modify original in the slice region *)
      Printf.printf "Original modified:\n";
      print t;
      Printf.printf "Slice view (should reflect change):\n";
      print s);

  (* todo *)
  (* test "Pad 1D array" "pad [(1, 2)] 0.0 [1., 2., 3.]" (fun () -> let t =
     create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in pad [| (1, 2) |] 0.0 t); *)

  (* todo *)
  (* test "Pad 2D array" "pad [(1, 1), (2, 0)] 9l [[1l, 2l], [3l, 4l]]" (fun ()
     -> let t = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in pad [| (1, 1);
     (2, 0) |] 9l t); *)
  test "Expand dims axis 0" "expand_dims 0 [1.0, 2.0, 3.0]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      expand_dims 0 t);

  test "Expand dims axis 1" "expand_dims 1 [1.0, 2.0, 3.0]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      expand_dims 1 t);

  test "Expand dims multiple" "expand_dims 0 then 2 for [1.0, 2.0, 3.0]"
    (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let e1 = expand_dims 0 t in
      expand_dims 2 e1);

  test_f "Expand dims invalid axis" "expand_dims 2 [1.0, 2.0, 3.0]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      expand_dims 2 t);

  test "Broadcast 1D to 3x3" "broadcast_to [1.0, 2.0, 3.0] [3, 3]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      broadcast_to [| 3; 3 |] t);

  test "Broadcast 2x1 to 2x3" "broadcast_to [[1.], [2.]] [2, 3]" (fun () ->
      let t = create float32 [| 2; 1 |] [| 1.0; 2.0 |] in
      broadcast_to [| 2; 3 |] t);

  test "Broadcast scalar to 2x2" "broadcast_to 5.0 [2, 2]" (fun () ->
      let t = scalar float32 5.0 in
      broadcast_to [| 2; 2 |] t);

  test_f "Broadcast incompatible" "broadcast_to [1.0, 2.0] [3, 3]" (fun () ->
      let t = create float32 [| 2 |] [| 1.0; 2.0 |] in
      broadcast_to [| 3; 3 |] t);

  test_f "Broadcast incompatible dim" "broadcast_to [1.,2.,3.] [2,2]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      broadcast_to [| 2; 2 |] t);

  (* === Conversion Tests === *)
  test_custom "Convert ndarray to array" "to_array [1, 2, 3]" (fun () ->
      let t = create int32 [| 3 |] [| 1l; 2l; 3l |] in
      let a = to_array t in
      Printf.printf "[%s]\n"
        (String.concat ";" (Array.map Int32.to_string a |> Array.to_list)));

  test "Convert float32 to int32" "astype int32 [1.1, 2.9, -3.3]" (fun () ->
      let t = create float32 [| 3 |] [| 1.1; 2.9; -3.3 |] in
      astype int32 t);

  test "Convert int32 to float64" "astype float64 [1l, 2l, 3l]" (fun () ->
      let t = create int32 [| 3 |] [| 1l; 2l; 3l |] in
      astype float64 t);

  test "Convert float32 to complex32" "astype complex32 [1.0, 2.0]" (fun () ->
      let t = create float32 [| 2 |] [| 1.0; 2.0 |] in
      astype complex32 t);

  test "Convert complex64 to float64 (real part)"
    "astype float64 [{1.,2.}, {3.,4.}]" (fun () ->
      let t =
        create complex64 [| 2 |]
          [| Complex.{ re = 1.; im = 2. }; { re = 3.; im = 4. } |]
      in
      astype float64 t);

  (* === Element-wise Operations === *)
  test "Add two 2x2 float32 ndarrays"
    "add [[1.0, 2.0], [3.0, 4.0]] [[5.0, 6.0], [7.0, 8.0]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      add t1 t2);

  test_custom "Add inplace 2x2 float32"
    "add_inplace t1 [[5.0, 6.0], [7.0, 8.0]] where t1=[[1..4]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      Printf.printf "t1 before:\n";
      print t1;
      let t_res = add_inplace t1 t2 in
      Printf.printf "t1 after:\n";
      print t1;
      Printf.printf "Returned tensor (should be t1):\n";
      print t_res;
      assert (t1 == t_res));

  test "Multiply two 2x2 ndarrays"
    "mul [[1.0, 2.0], [3.0, 4.0]] [[5.0, 6.0], [7.0, 8.0]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      mul t1 t2);

  test_custom "Multiply inplace 1D int32"
    "mul_inplace t1 [5, 6, 7] where t1=[1, 2, 3]" (fun () ->
      let t1 = create int32 [| 3 |] [| 1l; 2l; 3l |] in
      let t2 = create int32 [| 3 |] [| 5l; 6l; 7l |] in
      Printf.printf "t1 before:\n";
      print t1;
      let t_res = mul_inplace t1 t2 in
      Printf.printf "t1 after:\n";
      print t1;
      assert (t1 == t_res));

  test "Subtract two 2x2 ndarrays"
    "sub [[5.0, 6.0], [7.0, 8.0]] [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      sub t1 t2);

  test_custom "Subtract inplace 1D->2D broadcast"
    "sub_inplace t1 [1., 2.] where t1=[[10., 11.], [12., 13.]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 10.0; 11.0; 12.0; 13.0 |] in
      let t2 = create float32 [| 2 |] [| 1.0; 2.0 |] in
      Printf.printf "t1 before:\n";
      print t1;
      let t_res = sub_inplace t1 t2 in
      Printf.printf "t1 after:\n";
      print t1;
      assert (t1 == t_res));

  test "Divide two 2x2 ndarrays"
    "div [[5.0, 6.0], [7.0, 8.0]] [[1.0, 2.0], [4.0, 5.0]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 1.0; 2.0; 4.0; 5.0 |] in
      div t1 t2);

  test_custom "Divide inplace scalar broadcast"
    "div_inplace t1 2.0 where t1=[10., 20., 30.]" (fun () ->
      let t1 = create float32 [| 3 |] [| 10.0; 20.0; 30.0 |] in
      let t2 = scalar float32 2.0 in
      Printf.printf "t1 before:\n";
      print t1;
      let t_res = div_inplace t1 t2 in
      Printf.printf "t1 after:\n";
      print t1;
      assert (t1 == t_res));

  test "Remainder int32" "remainder [10, 11, 12] [3, 5, 4]" (fun () ->
      let t1 = create int32 [| 3 |] [| 10l; 11l; 12l |] in
      let t2 = create int32 [| 3 |] [| 3l; 5l; 4l |] in
      rem t1 t2);

  test "Power float32" "pow [2., 3., 4.] [3., 2., 0.5]" (fun () ->
      let t1 = create float32 [| 3 |] [| 2.0; 3.0; 4.0 |] in
      let t2 = create float32 [| 3 |] [| 3.0; 2.0; 0.5 |] in
      pow t1 t2);

  test "Add two 2x2 int32 ndarrays" "add [[1, 2], [3, 4]] [[5, 6], [7, 8]]"
    (fun () ->
      let t1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      let t2 = create int32 [| 2; 2 |] [| 5l; 6l; 7l; 8l |] in
      add t1 t2);

  test "Divide two 2x2 int32 ndarrays"
    "div [[10, 21], [30, 40]] [[3, 5], [4, 4]]" (fun () ->
      let t1 = create int32 [| 2; 2 |] [| 10l; 21l; 30l; 40l |] in
      let t2 = create int32 [| 2; 2 |] [| 3l; 5l; 4l; 4l |] in
      div t1 t2);

  test "Exponential of array" "exp [0.0, 1.0, 2.0]" (fun () ->
      let t = create float32 [| 3 |] [| 0.0; 1.0; 2.0 |] in
      exp t);

  test "Log of array" "log [1.0, 2.718, 7.389]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.71828; 7.38905 |] in
      log t);

  test_f "Log of non-positive" "log [1.0, 0.0, -1.0]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 0.0; -1.0 |] in
      log t);

  test "Absolute value int32" "abs [-1, 0, 5, -10]" (fun () ->
      let t = create int32 [| 4 |] [| -1l; 0l; 5l; -10l |] in
      abs t);

  test "Absolute value float32" "abs [-1.5, 0.0, 5.2]" (fun () ->
      let t = create float32 [| 3 |] [| -1.5; 0.0; 5.2 |] in
      abs t);

  test "Negation float64" "neg [1.0, -2.0, 0.0]" (fun () ->
      let t = create float64 [| 3 |] [| 1.0; -2.0; 0.0 |] in
      neg t);

  test "Sign float32" "sign [-5.0, 0.0, 3.2, -0.0]" (fun () ->
      let t = create float32 [| 4 |] [| -5.0; 0.0; 3.2; -0.0 |] in
      sign t);

  test "Sign int32" "sign [-5l, 0l, 3l]" (fun () ->
      let t = create int32 [| 3 |] [| -5l; 0l; 3l |] in
      sign t);

  test "Square root of array" "sqrt [4.0, 9.0, 16.0]" (fun () ->
      let t = create float32 [| 3 |] [| 4.0; 9.0; 16.0 |] in
      sqrt t);

  test_f "Square root of negative" "sqrt [4.0, -9.0, 16.0]" (fun () ->
      let t = create float32 [| 3 |] [| 4.0; -9.0; 16.0 |] in
      sqrt t);

  test "Maximum of two arrays" "maximum [1.0, 3.0, 2.0] [2.0, 1.0, 4.0]"
    (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
      let t2 = create float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
      maximum t1 t2);

  test "Minimum of two arrays int32" "minimum [1, 3, 4] [2, 1, 4]" (fun () ->
      let t1 = create int32 [| 3 |] [| 1l; 3l; 4l |] in
      let t2 = create int32 [| 3 |] [| 2l; 1l; 4l |] in
      minimum t1 t2);

  test "Equal comparison float32" "equal [1., 2., 3.] [1., 5., 3.]" (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let t2 = create float32 [| 3 |] [| 1.0; 5.0; 3.0 |] in
      equal t1 t2);

  test "Greater comparison of two arrays"
    "greater [1.0, 3.0, 2.0] [2.0, 1.0, 4.0]" (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
      let t2 = create float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
      greater t1 t2);

  test "Greater equal comparison int32" "greater_equal [1, 3, 4] [2, 3, 3]"
    (fun () ->
      let t1 = create int32 [| 3 |] [| 1l; 3l; 4l |] in
      let t2 = create int32 [| 3 |] [| 2l; 3l; 3l |] in
      greater_equal t1 t2);

  test "Less comparison float32" "less [1.0, 3.0, 2.0] [2.0, 1.0, 4.0]"
    (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 3.0; 2.0 |] in
      let t2 = create float32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
      less t1 t2);

  test "Less equal comparison int32" "less_equal [1, 3, 4] [2, 3, 3]" (fun () ->
      let t1 = create int32 [| 3 |] [| 1l; 3l; 4l |] in
      let t2 = create int32 [| 3 |] [| 2l; 3l; 3l |] in
      less_equal t1 t2);

  test "Square float32" "square [1., 2., -3., 0.]" (fun () ->
      let t = create float32 [| 4 |] [| 1.0; 2.0; -3.0; 0.0 |] in
      square t);

  (* --- Trig / Hyperbolic --- *)
  let trig_input =
    create float32 [| 4 |]
      [| 0.0; Float.pi /. 6.0; Float.pi /. 4.0; Float.pi /. 2.0 |]
  in
  test "Sine" "sin [0, pi/6, pi/4, pi/2]" (fun () -> sin trig_input);
  test "Cosine" "cos [0, pi/6, pi/4, pi/2]" (fun () -> cos trig_input);
  test "Tangent" "tan [0, pi/6, pi/4, pi/2]" (fun () -> tan trig_input);

  let asin_input = create float32 [| 3 |] [| 0.0; 0.5; 1.0 |] in
  test "Arcsine" "asin [0, 0.5, 1.0]" (fun () -> asin asin_input);

  let acos_input = create float32 [| 3 |] [| 1.0; 0.5; 0.0 |] in
  test "Arccosine" "acos [1.0, 0.5, 0.0]" (fun () -> acos acos_input);

  let atan_input = create float32 [| 3 |] [| 0.0; 1.0; -1.0 |] in
  test "Arctangent" "atan [0, 1.0, -1.0]" (fun () -> atan atan_input);

  let hyp_input = create float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  test "Hyperbolic Sine" "sinh [-1, 0, 1]" (fun () -> sinh hyp_input);
  test "Hyperbolic Cosine" "cosh [-1, 0, 1]" (fun () -> cosh hyp_input);
  test "Hyperbolic Tangent" "tanh [-1, 0, 1]" (fun () -> tanh hyp_input);

  let asinh_input = create float32 [| 3 |] [| -1.0; 0.0; 2.0 |] in
  test "Inverse Hyperbolic Sine" "asinh [-1, 0, 2]" (fun () ->
      asinh asinh_input);

  let acosh_input = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  test "Inverse Hyperbolic Cosine" "acosh [1, 2, 3]" (fun () ->
      acosh acosh_input);

  let atanh_input = create float32 [| 3 |] [| -0.5; 0.0; 0.5 |] in
  test "Inverse Hyperbolic Tangent" "atanh [-0.5, 0, 0.5]" (fun () ->
      atanh atanh_input);

  (* === Broadcasting Tests === *)
  test "Broadcasting: Add 1D to 2D"
    "add [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] [10., 20., 30.]" (fun () ->
      let t2d = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      let t1d = create float32 [| 3 |] [| 10.0; 20.0; 30.0 |] in
      add t2d t1d);

  test "Broadcasting: Add 2x1 to 2x3"
    "add [[1.], [2.]] [[10., 11., 12.], [20., 21., 22.]]" (fun () ->
      let t21 = create float32 [| 2; 1 |] [| 1.0; 2.0 |] in
      let t23 = create float32 [| 2; 3 |] [| 10.; 11.; 12.; 20.; 21.; 22. |] in
      add t21 t23);

  test "Broadcasting: Multiply (4, 1) * (3,)" "mul (shape 4,1) (shape 3,)"
    (fun () ->
      let t41 =
        reshape [| 4; 1 |] (create float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
      in
      let t3 = create float32 [| 3 |] [| 10.; 100.; 1000. |] in
      Printf.printf "t1 (4,1):\n";
      print t41;
      Printf.printf "t2 (3,):\n";
      print t3;
      mul t41 t3);

  test "Add scalar to 1D array" "add 2.0 [1.0, 2.0, 3.0]" (fun () ->
      let s = scalar float32 2.0 in
      let t1d = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      add s t1d);

  (* === Reduction Tests === *)
  test "Sum of 2x2 array (all)" "sum [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      sum t);

  test "Sum of 2x3 array axis=0" "sum ~axes:[0] [[1..3],[4..6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      sum ~axes:[| 0 |] t);

  test "Sum of 2x3 array axis=1 keepdims"
    "sum ~axes:[1] ~keepdims:true [[1..3],[4..6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      sum ~axes:[| 1 |] ~keepdims:true t);

  test "Product of 1D array int32" "prod [1, 2, 3, 4]" (fun () ->
      let t = create int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
      prod t);

  test "Product of 2x3 axis=1" "prod ~axes:[1] [[1,2,3],[4,5,6]]" (fun () ->
      let t = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
      prod ~axes:[| 1 |] t);

  test "Mean of 2x2 array (all)" "mean [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      mean t);

  test "Mean of 2x3 array axis=0 keepdims"
    "mean ~axes:[0] ~keepdims:true [[1..3],[4..6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      mean ~axes:[| 0 |] ~keepdims:true t);

  test "Max of 2x2 array (all)" "max [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      max t);

  test "Max of 2x3 array axis=1" "max ~axes:[1] [[1,5,3],[4,2,6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 5.; 3.; 4.; 2.; 6. |] in
      max ~axes:[| 1 |] t);

  test "Min of 2x2 array (all)" "min [[1.0, 2.0], [3.0, 4.0]]" (fun () ->
      let t = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      min t);

  test "Min of 2x3 array axis=0 keepdims"
    "min ~axes:[0] ~keepdims:true [[1,5,3],[4,2,6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 5.; 3.; 4.; 2.; 6. |] in
      min ~axes:[| 0 |] ~keepdims:true t);

  test "Variance of 1D array" "var [1., 2., 3., 4., 5.]" (fun () ->
      let t = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
      var t);

  test "Variance of 2x3 axis=1" "var ~axes:[1] [[1,2,3],[4,5,6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      var ~axes:[| 1 |] t);

  test "Standard Deviation of 1D array" "std [1., 2., 3., 4., 5.]" (fun () ->
      let t = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
      std t);

  test "Standard Deviation of 2x3 axis=0 keepdims"
    "std ~axes:[0] ~keepdims:true [[1,2,3],[4,5,6]]" (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      std ~axes:[| 0 |] ~keepdims:true t);

  (* === Matrix Operations === *)
  test "Dot product 1D x 1D (Inner Product)" "dot [1, 2, 3] [4, 5, 6]"
    (fun () ->
      let t1 = create int32 [| 3 |] [| 1l; 2l; 3l |] in
      let t2 = create int32 [| 3 |] [| 4l; 5l; 6l |] in
      dot t1 t2);

  test "Dot product 2D x 2D (Matrix Multiply)"
    "dot [[1.0, 2.0], [3.0, 4.0]] [[5.0, 6.0], [7.0, 8.0]]" (fun () ->
      let t1 = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let t2 = create float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
      dot t1 t2);

  test "Dot product 2D x 1D" "dot [[1, 2], [3, 4]] [10, 20]" (fun () ->
      let t1 = create int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
      let t2 = create int32 [| 2 |] [| 10l; 20l |] in
      dot t1 t2);

  (* todo *)
  (* test "Dot product scalar x 1D" "dot 5.0 [1., 2., 3.]" (fun () -> let s =
     scalar float32 5.0 in let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
     dot s t); *)

  (* todo *)
  (* test "Dot product 1D x scalar" "dot [1., 2., 3.] 5.0" (fun () -> let s =
     scalar float32 5.0 in let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
     dot t s); *)
  test "Dot product 3D x 2D" "dot (2,2,3) x (3,2)" (fun () ->
      let t1 = create float32 [| 2; 2; 3 |] (Array.init 12 float_of_int) in
      let t2 = create float32 [| 3; 2 |] (Array.init 6 float_of_int) in
      Printf.printf "t1 (2,2,3):\n";
      print t1;
      Printf.printf "t2 (3,2):\n";
      print t2;
      dot t1 t2);

  test "Transpose 2D array" "transpose [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"
    (fun () ->
      let t = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      transpose t);

  test "Transpose 3D array with axes" "transpose ~axes:[1, 2, 0] (2,3,4)"
    (fun () ->
      let t = create float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
      Printf.printf "Original (2,3,4):\n";
      print t;
      transpose ~axes:[| 1; 2; 0 |] t);

  test "Transpose 1D array" "transpose [1.0, 2.0, 3.0]" (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      transpose t);

  test "Transpose scalar" "transpose 42.0" (fun () ->
      let t = scalar float32 42.0 in
      transpose t);

  test "Matrix multiply 2x3 with 3x2"
    "matmul [[1.0..3.0], [4.0..6.0]] [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]"
    (fun () ->
      let t1 = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      let t2 =
        create float32 [| 3; 2 |] [| 7.0; 8.0; 9.0; 10.0; 11.0; 12.0 |]
      in
      matmul t1 t2);

  (* todo *)
  (* test "Matrix multiply broadcasting (2, 1, 2, 3) @ (1, 3, 3, 4)" "matmul
     (2,1,2,3) (1,3,3,4)" (fun () -> let t1 = create float32 [| 2; 1; 2; 3 |]
     (Array.init 12 float_of_int) in let t2 = create float32 [| 1; 3; 3; 4 |]
     (Array.init 36 float_of_int) in Printf.printf "t1 (2,1,2,3):\n"; print t1;
     Printf.printf "t2 (1,3,3,4):\n"; print t2; matmul t1 t2); *)
  test_f "Matmul incompatible shapes" "matmul (2,3) (2,3)" (fun () ->
      let t1 = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      let t2 = create float32 [| 2; 3 |] (Array.init 6 float_of_int) in
      matmul t1 t2);

  test "Convolve 1D array" "convolve [1., 2., 3., 4.] [1., 1.]" (fun () ->
      let signal = create float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let kernel = create float32 [| 2 |] [| 1.0; 1.0 |] in
      convolve1d signal kernel);

  (* todo *)
  (* test "Convolve 2D array" "convolve [[1..4],[5..8]] [[1,1],[1,1]]" (fun ()
     -> let signal = create float32 [| 2; 4 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.
     |] in let kernel = create float32 [| 2; 2 |] [| 1.; 1.; 1.; 1. |] in
     Printf.printf "Signal:\n"; print signal; Printf.printf "Kernel:\n"; print
     kernel; convolve signal kernel); *)

  (* === Logic Functions === *)
  test_custom "Check equal ndarrays"
    "array_equal [1.0, 2.0, 3.0] [1.0, 2.0, 3.0]" (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let t2 = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      Printf.printf "%b\n" (array_equal t1 t2));

  test_custom "Check not equal ndarrays (value)"
    "array_equal [1.0, 2.0, 3.0] [1.0, 2.0, 4.0]" (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let t2 = create float32 [| 3 |] [| 1.0; 2.0; 4.0 |] in
      Printf.printf "%b\n" (array_equal t1 t2));

  test_custom "Check not equal ndarrays (shape)"
    "array_equal [1.0, 2.0, 3.0] [[1.0, 2.0, 3.0]]" (fun () ->
      let t1 = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let t2 = create float32 [| 1; 3 |] [| 1.0; 2.0; 3.0 |] in
      Printf.printf "%b\n" (array_equal t1 t2));

  (* === Map and Fold === *)
  test "Map function (x * 10)" "map (fun x -> x * 10) [1, 2, 3]" (fun () ->
      let t = create int32 [| 3 |] [| 1l; 2l; 3l |] in
      map (fun x -> Int32.mul x 10l) t);

  test_custom "Iter function (summing)" "iter (summing ref) [1., 2., 3.]"
    (fun () ->
      let t = create float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let sum_ref = ref 0.0 in
      iter (fun x -> sum_ref := !sum_ref +. x) t;
      Printf.printf "Sum computed via iter: %f\n" !sum_ref);

  test_custom "Fold function (sum)" "fold (+) 0 [1, 2, 3, 4, 5, 6]" (fun () ->
      let t = create int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
      let sum = fold Int32.add 0l t in
      Printf.printf "Fold sum: %ld\n" sum);

  (* === Indexing === *)
  test "Where condition" "where [1,0,1] [10,20,30] [40,50,60]" (fun () ->
      let cond = create uint8 [| 3 |] [| 1; 0; 1 |] in
      let t1 = create int32 [| 3 |] [| 10l; 20l; 30l |] in
      let t2 = create int32 [| 3 |] [| 40l; 50l; 60l |] in
      where cond t1 t2);

  test "Where with broadcasting" "where [[1],[0]] [10,20] [[100,200],[300,400]]"
    (fun () ->
      let cond = create uint8 [| 2; 1 |] [| 1; 0 |] in
      let t1 = create int32 [| 2 |] [| 10l; 20l |] in
      (* Broadcasts to (2,2) *)
      let t2 = create int32 [| 2; 2 |] [| 100l; 200l; 300l; 400l |] in
      Printf.printf "Condition (2,1):\n";
      print cond;
      Printf.printf "T1 (2,):\n";
      print t1;
      Printf.printf "T2 (2,2):\n";
      print t2;
      where cond t1 t2);

  (* === Edge Case Tests === *)
  test "Division by zero float" "div [1.0] [0.0]" (fun () ->
      let t1 = scalar float32 1.0 in
      let t2 = scalar float32 0.0 in
      div t1 t2);

  (* todo *)
  (* test_f "Division by zero int" "div [1] [0]" (fun () -> let t1 = scalar
     int32 1l in let t2 = scalar int32 0l in div t1 t2); *)
  test_f "Invalid creation (mismatched size)"
    "create float32 [2, 2] [1.0, 2.0, 3.0]" (fun () ->
      create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0 |]);

  test "Add two empty arrays" "add (empty) (empty)" (fun () ->
      let t1 = create float32 [| 0 |] [||] in
      let t2 = create float32 [| 0 |] [||] in
      add t1 t2);

  ()
