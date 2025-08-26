open Safetensors

let bytes_equal b1 b2 = String.equal b1 b2

let test_dtype_bitsize =
  Alcotest.test_case "dtype bitsize" `Quick (fun () ->
      Alcotest.(check int) "F4 bitsize" 4 (bitsize F4);
      Alcotest.(check int) "F6_E2M3 bitsize" 6 (bitsize F6_E2M3);
      Alcotest.(check int) "U8 bitsize" 8 (bitsize U8);
      Alcotest.(check int) "I16 bitsize" 16 (bitsize I16);
      Alcotest.(check int) "F32 bitsize" 32 (bitsize F32);
      Alcotest.(check int) "F64 bitsize" 64 (bitsize F64))

let test_dtype_string_conversion =
  Alcotest.test_case "dtype string conversion" `Quick (fun () ->
      let dtypes =
        [
          BOOL;
          F4;
          F6_E2M3;
          F6_E3M2;
          U8;
          I8;
          F8_E5M2;
          F8_E4M3;
          F8_E8M0;
          I16;
          U16;
          F16;
          BF16;
          I32;
          U32;
          F32;
          F64;
          I64;
          U64;
        ]
      in
      List.iter
        (fun dt ->
          let str = dtype_to_string dt in
          match dtype_of_string str with
          | Some dt' ->
              Alcotest.(check bool) ("roundtrip " ^ str) true (dt = dt')
          | None -> Alcotest.fail ("Failed to parse " ^ str))
        dtypes)

let test_tensor_view_creation =
  Alcotest.test_case "tensor view creation" `Quick (fun () ->
      let data = String.make 16 '\x00' in
      match tensor_view_new ~dtype:F32 ~shape:[ 2; 2 ] ~data with
      | Ok tv ->
          Alcotest.(check bool) "dtype" true (tv.dtype = F32);
          Alcotest.(check bool) "shape" true (tv.shape = [ 2; 2 ]);
          Alcotest.(check int) "length" 16 tv.length
      | Error e -> Alcotest.fail (string_of_error e))

let test_tensor_view_misaligned =
  Alcotest.test_case "tensor view misaligned" `Quick (fun () ->
      let data = String.make 2 '\x00' in
      match tensor_view_new ~dtype:F4 ~shape:[ 1; 3 ] ~data with
      | Ok _ -> Alcotest.fail "Should have failed with misaligned slice"
      | Error Misaligned_slice -> ()
      | Error e -> Alcotest.fail ("Wrong error: " ^ string_of_error e))

let test_serialization_empty =
  Alcotest.test_case "serialization empty" `Quick (fun () ->
      match serialize [] None with
      | Ok serialized ->
          (* Expected: 8 bytes for header size + "{}" + padding to 8 bytes *)
          let expected =
            let header = "{}" in
            let padded_len = 8 in
            (* next multiple of 8 *)
            let buf = Bytes.create (8 + padded_len) in
            write_u64_le buf 0 (Int64.of_int padded_len);
            Bytes.blit_string header 0 buf 8 (String.length header);
            for i = 8 + String.length header to 8 + padded_len - 1 do
              Bytes.set buf i ' '
            done;
            Bytes.to_string buf
          in
          Alcotest.(check bool)
            "empty serialization" true
            (bytes_equal serialized expected)
      | Error e -> Alcotest.fail (string_of_error e))

let test_roundtrip_simple =
  Alcotest.test_case "roundtrip simple" `Quick (fun () ->
      (* Create a simple F32 tensor with data 0.0, 1.0, 2.0, 3.0 *)
      let float_to_bytes f =
        let buf = Bytes.create 4 in
        let bits = Int32.bits_of_float f in
        Bytes.set buf 0 (Char.chr (Int32.to_int (Int32.logand bits 0xFFl)));
        Bytes.set buf 1
          (Char.chr
             (Int32.to_int (Int32.logand (Int32.shift_right bits 8) 0xFFl)));
        Bytes.set buf 2
          (Char.chr
             (Int32.to_int (Int32.logand (Int32.shift_right bits 16) 0xFFl)));
        Bytes.set buf 3
          (Char.chr
             (Int32.to_int (Int32.logand (Int32.shift_right bits 24) 0xFFl)));
        Bytes.to_string buf
      in

      let data =
        String.concat "" (List.map float_to_bytes [ 0.0; 1.0; 2.0; 3.0 ])
      in

      match tensor_view_new ~dtype:F32 ~shape:[ 2; 2 ] ~data with
      | Error e ->
          Alcotest.fail ("Failed to create tensor: " ^ string_of_error e)
      | Ok tv -> (
          let tensors = [ ("test", tv) ] in
          match serialize tensors None with
          | Error e ->
              Alcotest.fail ("Failed to serialize: " ^ string_of_error e)
          | Ok serialized -> (
              match deserialize serialized with
              | Error e ->
                  Alcotest.fail ("Failed to deserialize: " ^ string_of_error e)
              | Ok st -> (
                  Alcotest.(check int) "tensor count" 1 (len st);
                  match tensor st "test" with
                  | Error e ->
                      Alcotest.fail
                        ("Failed to get tensor: " ^ string_of_error e)
                  | Ok tv' ->
                      Alcotest.(check bool) "dtype match" true (tv'.dtype = F32);
                      Alcotest.(check bool)
                        "shape match" true
                        (tv'.shape = [ 2; 2 ]);
                      let data' = String.sub tv'.data tv'.offset tv'.length in
                      Alcotest.(check bool)
                        "data match" true (bytes_equal data data')))))

let test_multiple_tensors =
  Alcotest.test_case "multiple tensors" `Quick (fun () ->
      let data1 = String.make 4 '\x01' in
      let data2 = String.make 8 '\x02' in

      match
        ( tensor_view_new ~dtype:U8 ~shape:[ 4 ] ~data:data1,
          tensor_view_new ~dtype:U8 ~shape:[ 2; 4 ] ~data:data2 )
      with
      | Ok tv1, Ok tv2 -> (
          let tensors = [ ("tensor1", tv1); ("tensor2", tv2) ] in
          match serialize tensors None with
          | Error e ->
              Alcotest.fail ("Failed to serialize: " ^ string_of_error e)
          | Ok serialized -> (
              match deserialize serialized with
              | Error e ->
                  Alcotest.fail ("Failed to deserialize: " ^ string_of_error e)
              | Ok st ->
                  Alcotest.(check int) "tensor count" 2 (len st);
                  Alcotest.(check bool)
                    "has tensor1" true
                    (List.mem "tensor1" (names st));
                  Alcotest.(check bool)
                    "has tensor2" true
                    (List.mem "tensor2" (names st))))
      | Error e, _ | _, Error e -> Alcotest.fail (string_of_error e))

let test_metadata =
  Alcotest.test_case "metadata" `Quick (fun () ->
      let data = String.make 4 '\x00' in
      match tensor_view_new ~dtype:U8 ~shape:[ 4 ] ~data with
      | Error e ->
          Alcotest.fail ("Failed to create tensor: " ^ string_of_error e)
      | Ok tv -> (
          let metadata = Some [ ("framework", "ocaml"); ("version", "1.0") ] in
          let tensors = [ ("test", tv) ] in
          match serialize tensors metadata with
          | Error e ->
              Alcotest.fail ("Failed to serialize: " ^ string_of_error e)
          | Ok serialized -> (
              match deserialize serialized with
              | Error e ->
                  Alcotest.fail ("Failed to deserialize: " ^ string_of_error e)
              | Ok st -> (
                  match st.metadata.metadata_kv with
                  | None -> Alcotest.fail "Metadata not preserved"
                  | Some kv ->
                      Alcotest.(check bool)
                        "has framework" true
                        (List.mem_assoc "framework" kv);
                      Alcotest.(check string)
                        "framework value" "ocaml"
                        (List.assoc "framework" kv)))))

let test_slicing_basic =
  Alcotest.test_case "slicing basic" `Quick (fun () ->
      let data = String.make 24 '\x00' in
      (* 6 floats * 4 bytes *)
      match tensor_view_new ~dtype:F32 ~shape:[ 2; 3 ] ~data with
      | Error e ->
          Alcotest.fail ("Failed to create tensor: " ^ string_of_error e)
      | Ok tv -> (
          let open Slice in
          (* Slice first row [0:1, :] *)
          match
            make tv [ included 0 // excluded 1; unbounded // unbounded ]
          with
          | Error _ -> Alcotest.fail "Failed to create slice"
          | Ok iter ->
              Alcotest.(check bool) "newshape" true (newshape iter = [ 1; 3 ]);
              Alcotest.(check int)
                "remaining bytes" 12 (remaining_byte_len iter)))

let test_error_handling =
  Alcotest.test_case "error handling" `Quick (fun () ->
      (* Test header too small *)
      let buffer = "short" in
      match deserialize buffer with
      | Ok _ -> Alcotest.fail "Should have failed with HeaderTooSmall"
      | Error Header_too_small -> ()
      | Error e -> Alcotest.fail ("Wrong error: " ^ string_of_error e))

let () =
  let open Alcotest in
  run "Safetensors"
    [
      ("dtype", [ test_dtype_bitsize; test_dtype_string_conversion ]);
      ("tensor_view", [ test_tensor_view_creation; test_tensor_view_misaligned ]);
      ( "serialization",
        [
          test_serialization_empty;
          test_roundtrip_simple;
          test_multiple_tensors;
          test_metadata;
        ] );
      ("slicing", [ test_slicing_basic ]);
      ("errors", [ test_error_handling ]);
    ]
