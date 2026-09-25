(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

external setup : unit -> nativeint
  = "caml_test_cuda_abi_setup"
external captured : unit -> bytes
  = "caml_test_cuda_abi_captured"
external create : nativeint -> string -> int -> int array -> nativeint
  = "caml_tolk_cuda_program_create"
external free : nativeint -> unit
  = "caml_tolk_cuda_program_free"
external launch :
  nativeint -> nativeint -> nativeint array -> int64 array -> int array -> int array -> bool ->
  float option
  = "caml_tolk_cuda_launch_kernel_bc" "caml_tolk_cuda_launch_kernel"
external function_ : nativeint -> nativeint = "caml_tolk_cuda_program_function"
external submit : nativeint -> bytes -> unit = "caml_test_cuda_abi_submit"
external handoffs : unit -> int = "caml_test_cuda_abi_handoffs"

let rejects f =
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (f ()))

let typed_arguments () =
  let queue = setup () in
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
  ignore (launch queue prg bufs vals dims dims false);
  equal bytes expected (captured ());
  rejects (fun () -> launch queue prg [||] vals dims dims false);
  rejects (fun () -> create 0n "bad" 2 [| 0; 0; 8; 0; 8; 8 |]);
  let fn = function_ prg in
  submit fn expected;
  equal bytes expected (captured ());
  equal int 1 (handoffs ());
  bufs.(1) <- 0x900008000n;
  vals.(0) <- 0x200000003L;
  vals.(1) <- 11L;
  Bytes.set_int64_le expected 0 0x900008000L;
  Bytes.set_int64_le expected 24 0x200000003L;
  Bytes.set_uint8 expected 16 11;
  ignore (launch queue prg bufs vals dims dims false);
  equal bytes expected (captured ());
  equal int 2 (handoffs ());
  free prg;
  submit fn expected;
  equal bytes expected (captured ());
  equal int 3 (handoffs ())

let () =
  run "CUDA native ABI"
    [ test "direct and compiled submission preserve typed arguments and handoffs"
        typed_arguments ]
