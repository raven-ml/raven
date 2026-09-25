(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

external setup : unit -> nativeint
  = "caml_test_cuda_abi_setup"
external captured : unit -> bytes
  = "caml_test_cuda_abi_captured"
external load : bytes -> nativeint = "caml_tolk_cuda_module_load"
external function_ : nativeint -> string -> nativeint = "caml_tolk_cuda_module_function"
external unload : nativeint -> unit = "caml_tolk_cuda_module_unload"
external live_modules : unit -> int = "caml_test_cuda_live_modules"
external copy : nativeint -> nativeint -> nativeint -> int -> unit
  = "caml_tolk_cuda_memcpy_htod_async"
external submit : nativeint -> bytes -> unit = "caml_test_cuda_abi_submit"
external handoffs : unit -> int = "caml_test_cuda_abi_handoffs"
external registration_status : int -> unit = "caml_test_cuda_registration_status"
external register_host : nativeint -> int -> int = "caml_tolk_cuda_mem_host_register"
external timestamp : unit -> int64 = "caml_test_cuda_timestamp"
external init : unit -> unit = "caml_tolk_cuda_init"
external init_counts : unit -> int * int = "caml_test_cuda_init_counts"
external shutdown_setup : bool -> nativeint = "caml_test_cuda_shutdown_setup"
external shutdown_steps : unit -> int = "caml_test_cuda_shutdown_steps"
external queue_destroy : nativeint -> unit = "caml_tolk_cuda_hcq_destroy"
external context_destroy : nativeint -> unit = "caml_tolk_cuda_ctx_destroy"

let concurrent_initialization () =
  let missing = Sys.getenv_opt "TOLK_TEST_CUDA_MISSING" <> None in
  let start = Atomic.make false in
  let domains = List.init 4 (fun _ -> Domain.spawn (fun () ->
      while not (Atomic.get start) do Domain.cpu_relax () done;
      for _ = 1 to 10 do
        if missing then
          raises (Failure "CUDA driver is missing cuMemAlloc_v2") init
        else init ()
      done)) in
  Atomic.set start true;
  List.iter Domain.join domains;
  let initialized, symbols = init_counts () in
  equal int (if missing then 0 else 1) initialized;
  is_true (symbols > 0);
  if not missing then init ();
  equal (pair int int) (initialized, symbols) (init_counts ())

let failed_initialization () =
  let binary = Sys.executable_name in
  let env = Array.append (Unix.environment ()) [| "TOLK_TEST_CUDA_MISSING=1" |] in
  let pid = Unix.create_process_env binary
      [| binary; "--filter"; "concurrent initialization" |] env
      Unix.stdin Unix.stdout Unix.stderr in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | _ -> failf "missing-symbol initialization did not fail consistently"

let shutdown_after_failure () =
  List.iter (fun failure ->
      ignore (setup ());
      let module_ = load (Bytes.of_string "ptx") in
      ignore (function_ module_ "typed");
      equal int 1 (live_modules ());
      let queue = shutdown_setup failure in
      let destroy () = Fun.protect ~finally:(fun () -> context_destroy 0x4n)
          (fun () -> queue_destroy queue) in
      if failure then
        raises (Failure "CUDA Error 719, injected synchronization failure") destroy
      else destroy ();
      equal int 5 (shutdown_steps ());
      equal ~msg:"context retirement releases its cached modules" int 0
        (live_modules ())) [ false; true ]

let compiled_arguments () =
  let queue = setup () in
  let module_ = load (Bytes.of_string "ptx") in
  let fn = function_ module_ "typed" in
  equal nativeint 0xcafen fn;
  equal int 1 (live_modules ());
  let expected = Bytes.of_string
      "\x00\x40\x00\x00\x03\x00\x00\x00\
       \x00\x20\x00\x00\x01\x00\x00\x00\
       \xf9\x00\x00\x00\x00\x00\x00\x00\
       \x00\x00\x00\x00\x00\x00\x00\x80\
       \x2c\x01\x00\x00\x89\x67\x45\x23" in
  copy queue 0x100002000n 0x300004000n 40;
  equal int 0 (handoffs ());
  submit fn expected;
  equal bytes expected (captured ());
  equal int 1 (handoffs ());
  Bytes.set_int64_le expected 0 0x900008000L;
  Bytes.set_int64_le expected 24 0x200000003L;
  Bytes.set_uint8 expected 16 11;
  copy queue 0x100002000n 0x300004000n 40;
  equal int 2 (handoffs ());
  submit fn expected;
  equal bytes expected (captured ());
  equal int 3 (handoffs ());
  equal int 1 (live_modules ());
  unload module_;
  equal int 0 (live_modules ())

let () =
  run "CUDA native ABI"
    [ test "timestamp callback runs only while the queue is healthy" (fun () ->
        is_true (timestamp () > 0L));
      test "concurrent initialization publishes a complete driver table" concurrent_initialization;
      test "missing driver symbols stay failed across callers" failed_initialization;
      test "compiled submission preserves packed arguments and allocator copy handoffs" compiled_arguments;
      test "host registration separates unsupported mappings from driver faults" (fun () ->
        List.iter (fun (status, expected) -> registration_status status;
            equal int expected (register_host 0x10000n 16)) [0, 1; 712, 0; 1, -1; 801, -1];
        equal int (-1) (register_host 0x10001n 16);
        registration_status 2;
        raises (Failure "CUDA Error 2, injected synchronization failure")
          (fun () -> ignore (register_host 0x10000n 16)));
      test "shutdown releases all resources after synchronization failure" shutdown_after_failure ]
