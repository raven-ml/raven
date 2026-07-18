(* Runner for the engine self-test (nx_c_selftest.c). Drives coalescing, the
   iteration ladder, dispatch, the thread pool, and the fold/argreduce/scan
   protocols through the real engine with trivial f64 kernels, then exercises
   the FFI funnel end-to-end through a macro-generated map stub — which also
   pins the FFI record slot order.

   The externals below bind the engine's test-only C entry points
   (nx_c_selftest.c symbols) directly rather than through Nx_backend, isolating
   the engine from the OCaml binding. *)

open Bigarray
open Windtrap

(* FFI operand mirroring the four slots the engine reads (the nx_c.h NX_C_FFI
   markers): buffer, shape, strides, offset at record slots 0-3. Declaration
   order IS that slot order; the echo_neg round-trip below pins it. *)
type ffi = {
  buffer : (float, float64_elt, c_layout) Array1.t;
  shape : int array;
  strides : int array;
  offset : int;
}

external selftest : unit -> int = "caml_nx_c_selftest"
external parallel_probe : unit -> int = "caml_nx_c_selftest_parallel_probe"
external echo_neg : ffi -> ffi -> unit = "caml_nx_c_echo_neg"
external echo_sum : ffi -> ffi -> int array -> unit = "caml_nx_c_echo_sum"
external echo_argmax : ffi -> ffi -> int -> unit = "caml_nx_c_echo_argmax"

let check cond name = is_true ~msg:name cond

(* Build an ffi record over a float64 buffer; shape defaults to the whole
   buffer, strides to contiguous. *)
let mk ?(offset = 0) ?strides ?shape buffer =
  let shape =
    match shape with Some s -> s | None -> [| Array1.dim buffer |]
  in
  let strides = match strides with Some s -> s | None -> [| 1 |] in
  { buffer; shape; strides; offset }

let test_engine () =
  (* Driver-level self-test (builds nx_c_ndarray structs directly in C). *)
  let n = selftest () in
  check (n = 0) (Printf.sprintf "%d driver self-test failure(s)" n);

  (* FFI funnel: neg through a macro-generated stub over real FFI records. *)
  let len = 64 in
  let inb = Array1.create float64 c_layout len in
  let outb = Array1.create float64 c_layout len in
  for i = 0 to len - 1 do
    inb.{i} <- float_of_int (i + 1);
    outb.{i} <- 0.0
  done;
  echo_neg (mk outb) (mk inb);
  let ok = ref true in
  for i = 0 to len - 1 do
    if outb.{i} <> -.float_of_int (i + 1) then ok := false
  done;
  check !ok "FFI funnel neg (macro stub, slot order)";

  (* The funnel raises on an aliased (0-stride) output. *)
  let raised =
    try
      echo_neg (mk ~strides:[| 0 |] outb) (mk inb);
      false
    with Invalid_argument _ -> true
  in
  check raised "FFI funnel raises Invalid_argument on aliased output";

  (* Fold funnel with keepdims=true: reduce [2,3] over axis 1 into [2,1]. The
     output is full-rank with the reduced axis size 1, so nx_c_squeeze_out takes
     its keepdims path and rebuilds the kept-axis descriptor. *)
  let src = Array1.create float64 c_layout 6 in
  for i = 0 to 5 do
    src.{i} <- float_of_int (i + 1)
  done;
  (* src as [2,3]: rows (1,2,3) and (4,5,6) *)
  let dst = Array1.create float64 c_layout 2 in
  dst.{0} <- 0.0;
  dst.{1} <- 0.0;
  echo_sum
    (mk ~shape:[| 2; 1 |] ~strides:[| 1; 1 |] dst)
    (mk ~shape:[| 2; 3 |] ~strides:[| 3; 1 |] src)
    [| 1 |];
  check (dst.{0} = 6.0 && dst.{1} = 15.0) "fold funnel keepdims=true squeeze";

  (* Fold funnel rejects a duplicate axis (nx_c_squeeze_out mask
     dup-detection). *)
  let dup_raised =
    try
      echo_sum
        (mk ~shape:[| 2; 1 |] ~strides:[| 1; 1 |] dst)
        (mk ~shape:[| 2; 3 |] ~strides:[| 3; 1 |] src)
        [| 1; 1 |];
      false
    with Invalid_argument _ -> true
  in
  check dup_raised "fold funnel raises on duplicate axis";

  (* Argreduce funnel raise path: empty axis -> Invalid_argument (output never
     written, so a scalar float64 stand-in is fine). *)
  let empty_in = Array1.create float64 c_layout 1 in
  let scalar_out = Array1.create float64 c_layout 1 in
  let arg_raised =
    try
      echo_argmax
        (mk ~shape:[||] ~strides:[||] scalar_out)
        (mk ~shape:[| 0 |] ~strides:[| 1 |] empty_in)
        0;
      false
    with Invalid_argument _ -> true
  in
  check arg_raised "argreduce funnel raises on empty axis"

let test_pool_after_fork () =
  check (parallel_probe () = 0) "worker pool before fork";
  let pid = Unix.fork () in
  let status =
    if pid = 0 then Unix._exit (parallel_probe ())
    else
      let deadline = Unix.gettimeofday () +. 5.0 in
      let rec wait () =
        match Unix.waitpid [ Unix.WNOHANG ] pid with
        | 0, _ when Unix.gettimeofday () < deadline ->
            Unix.sleepf 0.001;
            wait ()
        | 0, _ ->
            Unix.kill pid Sys.sigkill;
            ignore (Unix.waitpid [] pid);
            3
        | _, Unix.WEXITED code -> if code = 0 then 0 else 10 + code
        | _, Unix.WSIGNALED signal -> 100 + signal
        | _, Unix.WSTOPPED signal -> 200 + signal
      in
      wait ()
  in
  check (status = 0)
    (Printf.sprintf "worker pool after fork (status %d)" status);
  check (parallel_probe () = 0) "parent worker pool after child exit"

let () =
  Windtrap.run "nx C backend engine"
    [
      group "engine-invariants"
        [
          test "driver and FFI funnels" test_engine;
          test "worker pool is rebuilt after fork" test_pool_after_fork;
        ];
    ]
