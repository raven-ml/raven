(* Copyright (c) 2026 The Raven authors. ISC License. *)

open Windtrap

external setup : bool -> nativeint = "caml_test_metal_completion_setup"
external complete : int64 -> string -> bool -> unit = "caml_test_metal_complete"
external poll : unit -> int64 = "caml_test_metal_completion_poll"
external stamps : unit -> int64 * int64 = "caml_test_metal_completion_stamps"
external wait : nativeint -> int64 -> unit = "caml_tolk_metal_hcq_wait"
external failed_submit : unit -> unit = "caml_test_metal_failed_submit"

let successful_completion () =
  List.iter (fun profile ->
      let context = setup profile in
      equal int64 0L (poll ());
      wait context 0L;
      let worker = Domain.spawn (fun () -> complete 1L "" profile) in
      Fun.protect ~finally:(fun () -> Domain.join worker) (fun () -> wait context 1L);
      equal int64 1L (poll ());
      equal (pair int64 int64)
        (if profile then 1_000_000_000L, 2_000_000_000L else 0L, 0L)
        (stamps ())) [false; true]

let failed_completion () =
  let context = setup false in
  let error = Failure "Metal queue: injected GPU fault" in
  let worker = Domain.spawn (fun () -> complete 1L "injected GPU fault" false) in
  Fun.protect ~finally:(fun () -> Domain.join worker) (fun () ->
      raises error (fun () -> wait context 1L));
  equal ~msg:"native fence loops unblock on failure" int64 Int64.minus_one (poll ());
  raises ~msg:"preparation reports the latched error" error (fun () -> wait context 0L);
  failed_submit ();
  complete 2L "later error" false;
  raises ~msg:"preserve the first fault" error (fun () -> wait context 2L)

let () = run __FILE__ [
  test "host waits observe completion and profiling writes" successful_completion;
  test "GPU failures stop submissions and reach host waits" failed_completion;
]
