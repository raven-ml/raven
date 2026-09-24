(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

external setup : unit -> unit
  = "caml_test_cuda_abi_setup"
external captured : unit -> bytes
  = "caml_test_cuda_abi_captured"
external create : nativeint -> string -> int -> int array -> nativeint
  = "caml_tolk_cuda_program_create"
external free : nativeint -> unit
  = "caml_tolk_cuda_program_free"
external launch :
  nativeint -> nativeint array -> int64 array -> int array -> int array -> bool ->
  float option
  = "caml_tolk_cuda_launch_kernel_bc" "caml_tolk_cuda_launch_kernel"
external graph_create : int -> nativeint
  = "caml_tolk_cuda_graph_create"
external graph_add :
  nativeint -> nativeint -> int array -> int array -> nativeint array ->
  int64 array -> int array -> int
  = "caml_tolk_cuda_graph_add_kernel_bc" "caml_tolk_cuda_graph_add_kernel"
external graph_instantiate : nativeint -> unit
  = "caml_tolk_cuda_graph_instantiate"
external graph_set_buf : nativeint -> int -> int -> nativeint -> unit
  = "caml_tolk_cuda_graph_set_buf"
external graph_set_val : nativeint -> int -> int -> int64 -> unit
  = "caml_tolk_cuda_graph_set_val"
external graph_set_params : nativeint -> int -> unit
  = "caml_tolk_cuda_graph_set_params"
external graph_destroy : nativeint -> unit
  = "caml_tolk_cuda_graph_destroy"

let rejects f =
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (f ()))

let typed_arguments () =
  setup ();
  let layout = [| 1; 0; 8; 0; 8; 8; 3; 16; 1; 2; 24; 8; 4; 32; 2; 5; 36; 4 |] in
  let prg = create 0n "typed" 2 layout in
  let bufs = [| 0x100002000n; 0x300004000n |] in
  let vals = [| Int64.min_int; -7L; 300L; 0x123456789L |] in
  let expected = Bytes.of_string
      "\x00\x40\x00\x00\x03\x00\x00\x00\
       \x00\x20\x00\x00\x01\x00\x00\x00\
       \xf9\x00\x00\x00\x00\x00\x00\x00\
       \x00\x00\x00\x00\x00\x00\x00\x80\
       \x2c\x01\x00\x00\x89\x67\x45\x23" in
  let dims = [| 1; 1; 1 |] in
  ignore (launch prg bufs vals dims dims false);
  equal bytes expected (captured ());
  rejects (fun () -> launch prg [||] vals dims dims false);
  rejects (fun () -> create 0n "bad" 2 [| 0; 0; 8; 0; 8; 8 |]);
  let graph = graph_create 1 in
  ignore (graph_add graph prg dims dims bufs vals [||]);
  equal bytes expected (captured ());
  graph_instantiate graph;
  free prg;
  graph_set_buf graph 0 1 0x900008000n;
  graph_set_val graph 0 0 0x200000003L;
  graph_set_val graph 0 1 11L;
  graph_set_params graph 0;
  Bytes.set_int64_le expected 0 0x900008000L;
  Bytes.set_int64_le expected 24 0x200000003L;
  Bytes.set_uint8 expected 16 11;
  equal bytes expected (captured ());
  graph_destroy graph

let () =
  run "CUDA native ABI"
    [ test "dispatch and replay preserve signature slots and scalar widths"
        typed_arguments ]
