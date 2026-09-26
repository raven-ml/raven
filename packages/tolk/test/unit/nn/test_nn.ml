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

let with_safetensors_header header data f =
  let path = Filename.temp_file "tolk-header" ".safetensors" in
  Fun.protect ~finally:(fun () -> Sys.remove path) (fun () ->
      Out_channel.with_open_bin path (fun oc ->
          let prefix = Bytes.create 8 in
          Bytes.set_int64_le prefix 0 (Int64.of_int (String.length header));
          output_bytes oc prefix;
          output_string oc header;
          output_string oc data);
      f path)

let header_tests =
  group "safetensors headers"
    [
      test "decodes Unicode and every JSON string escape" (fun () ->
          let header = {|{"\u0061\u00e9\ud83d\ude80\"\\\/\b\f\r\n\t":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|} in
          with_safetensors_header header "\001\000\000\000" (fun path ->
              let tensors = State.safe_load path in
              equal (list string) [ "aé🚀\"\\/\b\012\r\n\t" ] (List.map fst tensors);
              equal (array int) [| 1 |] (Run.to_int_array (snd (List.hd tensors)))));
      test "accepts unknown fields with general JSON values" (fun () ->
          let header = {|{"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4],"extra":[true,false,null,1.25e+2]}}|} in
          with_safetensors_header header "\042\000\000\000" (fun path ->
              equal (array int) [| 42 |]
                (Run.to_int_array (List.assoc "a" (State.safe_load path)))));
      test "preserves integer shape dimensions above 2^53" (fun () ->
          let header = {|{"empty":{"dtype":"U8","shape":[9007199254740993,0],"data_offsets":[0,0]}}|} in
          with_safetensors_header header "" (fun path ->
              equal (list int) [ 9007199254740993; 0 ]
                (T.shape (List.assoc "empty" (State.safe_load path)))));
      test "rejects invalid headers before loading tensors" (fun () ->
          List.iter
            (fun header ->
              with_safetensors_header header "\000\000\000\000" (fun path ->
                  raises_match (function Invalid_argument _ -> true | _ -> false)
                    (fun () -> State.safe_load path)))
            [
              {|{"\ud800":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|};
              {|{"\udc00":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|};
              {|{"\ud800\u0041":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|};
              {|{"\q":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[01],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[1,],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4],}}|};
              "{\"a\001b\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
              "{\"\255\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
              {|{"__metadata__":{"format":1},"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[-1],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[1.0],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[1],"data_offsets":[4,0]}}|};
              {|{"a":{"dtype":"I32","shape":[1],"data_offsets":[0,8]}}|};
              {|{"a":{"dtype":"I32","shape":[2],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[1],"shape":[1],"data_offsets":[0,4]}}|};
              {|{"a":{"dtype":"I32","shape":[4611686018427387903,4],"data_offsets":[0,4]}}|};
            ]);
    ]

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
      test "load_state_dict preserves the parameter device" (fun () ->
          let module U = Tolk_uop.Uop in
          let empty device = Tolk_frontend.Creation.empty
              ~device:(U.Single device) [2; 2] in
          let p = empty "CPU:1" and v = empty "CPU" in
          State.load_state_dict ~realize:false [("p", p)] [("p", v)];
          equal (option string) (Some "CPU:1")
            (match T.device p with Some (U.Single d) -> Some d | _ -> None);
          is_true ~msg:"placed checkpoint values are copied to the parameter device"
            (U.op (T.uop p) = Tolk_uop.Ops.Copy));
      test "load_state_dict shards checkpoint values along the parameter axis" (fun () ->
          let module U = Tolk_uop.Uop in
          let devices = ["CPU:1"; "CPU:2"] in
          let empty () = Tolk_frontend.Creation.empty ~device:(U.Single "CPU") [2; 4] in
          let p = Tolk_frontend.Creation.shard ~axis:1 ~devices (empty ()) in
          State.load_state_dict ~realize:false [("p", p)] [("p", empty ())];
          equal (option (list string)) (Some devices)
            (match T.device p with Some (U.Multi ds) -> Some ds | _ -> None);
          equal (option int) (Some 1) (U.axis (T.uop p));
          equal (list int) [2; 2] (U.max_shard_shape (T.uop p));
          equal (list int) [2; 4] (T.shape p));
      test "load_state_dict retains an already sharded checkpoint" (fun () ->
          let module U = Tolk_uop.Uop in
          let empty () = Tolk_frontend.Creation.empty ~device:(U.Single "CPU") [4; 4] in
          let p = Tolk_frontend.Creation.shard ~axis:1
              ~devices:["CPU:1"; "CPU:2"] (empty ()) in
          let v = Tolk_frontend.Creation.shard ~axis:0
              ~devices:["CPU:3"; "CPU:4"] (empty ()) in
          State.load_state_dict ~realize:false [("p", p)] [("p", v)];
          is_true ~msg:"target state loading retains existing checkpoint partitioning"
            (U.equal (T.uop p) (T.uop v)));
      test "load_state_dict leaves device-less checkpoint values virtual" (fun () ->
          let module U = Tolk_uop.Uop in
          let p = Tolk_frontend.Creation.empty ~device:(U.Single "CPU:1") [2] in
          let v = Tolk_frontend.Creation.zeros ~buffer:false [2] in
          is_true ~msg:"checkpoint fixture has no device" (T.device v = None);
          State.load_state_dict ~realize:false [("p", p)] [("p", v)];
          is_true ~msg:"device-less values follow target to() semantics"
            (U.equal (T.uop p) (T.uop v)));
      test "load_state_dict rejects unsupported disk transfers before rebinding" (fun () ->
          let module U = Tolk_uop.Uop in
          let empty device = Tolk_frontend.Creation.empty
              ~device:(U.Single device) [1] in
          let p = empty "DISK:/unused-state-destination" and v = empty "CPU" in
          let before = T.uop p in
          raises_match (function Invalid_argument _ -> true | _ -> false)
            (fun () -> State.load_state_dict ~realize:false [("p", p)] [("p", v)]);
          is_true ~msg:"failed placement leaves the parameter unchanged"
            (U.equal before (T.uop p)));
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

let () = exit (run "Tolk_nn" [ embedding_tests; linear_tests; layer_norm_tests; state_tests; header_tests ])
