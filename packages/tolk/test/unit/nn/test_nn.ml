(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* End-to-end tests for the nn layers and state loading: build small layers,
   realize them, and assert the computed values against expectations derived
   from tinygrad. *)

open Windtrap
module T = Tolk_frontend.Tensor
module Mv = Tolk_frontend.Movement
module Run = Tolk_frontend.Run
module Embedding = Tolk_nn.Embedding
module Linear = Tolk_nn.Linear
module Layer_norm = Tolk_nn.Layer_norm
module State = Tolk_nn.State
module D = Tolk_uop.Dtype

let close a b = Float.abs (a -. b) < 1e-4

let check_floats expected t =
  let got = Run.to_float_array t in
  equal int (Array.length expected) (Array.length got);
  Array.iteri
    (fun i e ->
      if not (close e got.(i)) then
        failf "element %d: expected %g, got %g" i e got.(i))
    expected

let embedding_tests =
  group "embedding"
    [
      test "looks up rows" (fun () ->
          let e : Embedding.t =
            {
              weight =
                Run.of_float_array ~shape:[ 4; 2 ]
                  [| 0.; 1.; 10.; 11.; 20.; 21.; 30.; 31. |];
            }
          in
          let idx = Run.of_int_array ~shape:[ 1; 3 ] [| 2; 0; 2 |] in
          let out = Embedding.apply e idx in
          equal (list int) [ 1; 3; 2 ] (T.shape out);
          check_floats [| 20.; 21.; 0.; 1.; 20.; 21. |] out);
      test "rejects float indices" (fun () ->
          let e = Embedding.create 4 2 in
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Embedding.apply e (Run.of_float_array ~shape:[ 1 ] [| 0. |])));
    ]

let linear_tests =
  group "linear"
    [
      test "applies weight transpose and bias" (fun () ->
          let l : Linear.t =
            {
              (* weight is (out=2, in=3) *)
              weight =
                Run.of_float_array ~shape:[ 2; 3 ] [| 1.; 0.; 0.; 0.; 1.; 1. |];
              bias = Some (Run.of_float_array ~shape:[ 2 ] [| 0.5; -0.5 |]);
            }
          in
          let x = Run.of_float_array ~shape:[ 1; 3 ] [| 1.; 2.; 3. |] in
          let out = Linear.apply l x in
          equal (list int) [ 1; 2 ] (T.shape out);
          check_floats [| 1.5; 4.5 |] out);
      test "no bias" (fun () ->
          let l : Linear.t =
            {
              weight = Run.of_float_array ~shape:[ 1; 2 ] [| 2.; 3. |];
              bias = None;
            }
          in
          check_floats [| 8. |]
            (Linear.apply l (Run.of_float_array ~shape:[ 1; 2 ] [| 1.; 2. |])));
    ]

let layer_norm_tests =
  group "layer_norm"
    [
      test "normalizes and applies affine" (fun () ->
          let ln = Layer_norm.create 2 in
          let x = Run.of_float_array ~shape:[ 1; 2 ] [| 1.; 3. |] in
          (* mean 2, biased var 1: normalized to (-1, 1); identity affine. *)
          check_floats [| -0.99999; 0.99999 |] (Layer_norm.apply ln x));
      test "rejects wrong last axis" (fun () ->
          let ln = Layer_norm.create 3 in
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              Layer_norm.apply ln (Run.of_float_array ~shape:[ 1; 2 ] [| 1.; 2. |])));
    ]

(* State loading: write a small safetensors file, read it back, and bind it
   into parameter handles. *)

let write_safetensors path tensors =
  let headers =
    List.map
      (fun (name, dtype, shape, off0, off1) ->
        Printf.sprintf
          {|"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}|} name dtype
          (String.concat "," (List.map string_of_int shape))
          off0 off1)
      tensors
  in
  let header =
    Printf.sprintf {|{"__metadata__":{"format":"pt"},%s}|}
      (String.concat "," headers)
  in
  let oc = open_out_bin path in
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int (String.length header));
  output_bytes oc b;
  output_string oc header;
  oc

let output_floats oc values =
  List.iter
    (fun v ->
      let b = Bytes.create 4 in
      Bytes.set_int32_le b 0 (Int32.bits_of_float v);
      output_bytes oc b)
    values

let state_tests =
  group "state"
    [
      test "safe_load reads tensors" (fun () ->
          let path = Filename.temp_file "tolk_nn" ".safetensors" in
          let oc =
            write_safetensors path
              [ ("a", "F32", [ 2; 2 ], 0, 16); ("b", "F32", [ 3 ], 16, 28) ]
          in
          output_floats oc [ 1.; 2.; 3.; 4.; 5.; 6.; 7. ];
          close_out oc;
          let sd = State.safe_load path in
          Sys.remove path;
          equal (list string) [ "a"; "b" ] (List.map fst sd);
          equal (list int) [ 2; 2 ] (T.shape (List.assoc "a" sd));
          check_floats [| 1.; 2.; 3.; 4. |] (List.assoc "a" sd);
          check_floats [| 5.; 6.; 7. |] (List.assoc "b" sd));
      test "safe_load supports fp8" (fun () ->
          let path = Filename.temp_file "tolk_nn" ".safetensors" in
          let oc =
            write_safetensors path
              [ ("e4m3", "F8_E4M3", [ 2 ], 0, 2); ("e5m2", "F8_E5M2", [ 3 ], 2, 5) ]
          in
          List.iter (output_byte oc) [ 0x00; 0x38; 0x01; 0x02; 0x03 ];
          close_out oc;
          let sd = State.safe_load path in
          Sys.remove path;
          is_true (D.equal (T.dtype (List.assoc "e4m3" sd)) D.fp8e4m3);
          is_true (D.equal (T.dtype (List.assoc "e5m2" sd)) D.fp8e5m2);
          equal (list int) [ 2 ] (T.shape (List.assoc "e4m3" sd));
          equal (list int) [ 3 ] (T.shape (List.assoc "e5m2" sd)));
      test "load_state_dict rebinds parameters" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 2; 2 ] in
          let v = Run.of_float_array ~shape:[ 2; 2 ] [| 1.; 2.; 3.; 4. |] in
          State.load_state_dict [ ("p", p) ] [ ("p", v) ];
          check_floats [| 1.; 2.; 3.; 4. |] p);
      test "load_state_dict materialises views" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 3; 2 ] in
          let v = Run.of_float_array ~shape:[ 2; 3 ] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
          State.load_state_dict [ ("p", p) ] [ ("p", Mv.transpose v) ];
          check_floats [| 1.; 4.; 2.; 5.; 3.; 6. |] p);
      test "load_state_dict reshapes a scalar to a one-vector" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 1 ] in
          let v = Run.of_float_array ~shape:[] [| 5. |] in
          State.load_state_dict [ ("p", p) ] [ ("p", v) ];
          check_floats [| 5. |] p);
      test "load_state_dict rejects a non-scalar one-element mismatch" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 1; 1 ] in
          let v = Run.of_float_array ~shape:[ 1 ] [| 5. |] in
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () -> State.load_state_dict [ ("p", p) ] [ ("p", v) ]));
      test "strict load fails on a missing key" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 1 ] in
          raises_match
            (function Invalid_argument _ -> true | _ -> false)
            (fun () ->
              State.load_state_dict [ ("p", p) ] []));
      test "non-strict load skips a missing key" (fun () ->
          let p = Tolk_frontend.Creation.zeros [ 1 ] in
          let before = T.uop p in
          State.load_state_dict ~strict:false [ ("p", p) ] [];
          is_true (T.uop p == before));
    ]

let () = run "Tolk_nn" [ embedding_tests; linear_tests; layer_norm_tests; state_tests ]
