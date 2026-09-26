(* Copyright (c) 2026 The Raven authors. ISC License. *)

open Windtrap

external setup : unit -> nativeint = "caml_test_metal_completion_setup"
external enqueue : nativeint -> int64 -> int -> nativeint = "caml_test_metal_enqueue"
external complete : nativeint -> nativeint -> string -> unit = "caml_test_metal_complete"
external poll : nativeint -> int64 = "caml_test_metal_completion_poll"
external stamps : nativeint -> int -> int64 * int64 = "caml_test_metal_completion_stamps"
external released : nativeint -> int = "caml_test_metal_released"
external wait : nativeint -> int64 -> unit = "caml_tolk_metal_hcq_wait"
external release : nativeint -> unit = "caml_tolk_metal_hcq_release"
external failed_submit : nativeint -> unit = "caml_test_metal_failed_submit"
external cleanup : nativeint -> unit = "caml_test_metal_completion_cleanup"

let with_context f =
  let context = setup () in
  Fun.protect ~finally:(fun () -> cleanup context) (fun () -> f context)

let completed_stamps = 1_000_000_000L, 2_000_000_000L
let no_stamps = 0L, 0L

let successful_completion () =
  List.iter (fun profile -> with_context (fun context ->
      let command = enqueue context 1L (if profile then 0 else -1) in
      equal int64 0L (poll context);
      wait context 0L;
      let worker = Domain.spawn (fun () -> complete context command "") in
      Fun.protect ~finally:(fun () -> Domain.join worker) (fun () -> wait context 1L);
      equal int64 1L (poll context);
      equal (pair int64 int64) (if profile then completed_stamps else no_stamps)
        (stamps context 0);
      equal int 1 (released context))) [false; true]

let completed_before_later_profile () = with_context (fun context ->
  let first = enqueue context 1L (-1) in
  complete context first "";
  equal int64 1L (poll context);
  let later = enqueue context 2L 0 in
  equal ~msg:"later profiling cannot hide an already completed submission"
    int64 1L (poll context);
  wait context 1L;
  equal (pair int64 int64) no_stamps (stamps context 0);
  complete context later "";
  equal int64 2L (poll context);
  equal (pair int64 int64) completed_stamps (stamps context 0))

let ordered_collection () = with_context (fun context ->
  let first = enqueue context 0L 0 in
  let last = enqueue context 1L 1 in
  complete context last "";
  equal ~msg:"a later ready command cannot publish the submission" int64 0L (poll context);
  equal int 0 (released context);
  equal (pair int64 int64) no_stamps (stamps context 1);
  complete context first "";
  equal int64 1L (poll context);
  List.iter (fun slot -> equal (pair int64 int64) completed_stamps (stamps context slot)) [0; 1];
  equal int 2 (released context))

let failed_completion () = with_context (fun context ->
  let first = enqueue context 1L 0 in
  let later = enqueue context 2L 1 in
  complete context later "";
  complete context first "injected GPU fault";
  let error = Failure "Metal queue: injected GPU fault" in
  raises error (fun () -> wait context 1L);
  equal ~msg:"native fence loops unblock on failure" int64 Int64.minus_one (poll context);
  raises ~msg:"preparation reports the latched error" error (fun () -> wait context 0L);
  failed_submit context;
  complete context later "later error";
  raises ~msg:"preserve the first fault" error (fun () -> wait context 2L);
  equal ~msg:"failure retains command ownership" int 0 (released context);
  raises (Failure "Metal queue has unretired commands") (fun () -> release context))

let context_retirement () = with_context (fun context ->
  let command = enqueue context 1L (-1) in
  raises (Failure "Metal queue has unretired commands") (fun () -> release context);
  equal int 0 (released context);
  complete context command "";
  wait context 1L;
  equal ~msg:"successful wait retires native command ownership" int 1 (released context))

external shared_encode : bool -> string = "caml_test_metal_shared_encode"

let shared_encode_order profile () =
  let icb_before, icb_after = if profile then "I0:1;I1:1;", "I4:1;I5:1;"
    else "I0:2;", "I4:2;" in
  let pass extent = Printf.sprintf "%sB;A512:0;D%d/2;B;B;A768:0;D%d/2;B;%s"
      icb_before extent extent icb_after in
  equal string (pass 7 ^ "|" ^ pass 11) (shared_encode profile)

let () = exit (run __FILE__ [
  test "shared submission keeps ICB and direct dispatch order" (shared_encode_order false);
  test "profiled direct dispatch reads updated launch sizes" (shared_encode_order true);
  test "host waits observe completion and profiling writes" successful_completion;
  test "completed work does not wait for later profiling" completed_before_later_profile;
  test "completion is collected in submission order" ordered_collection;
  test "GPU failures stop submissions and retain command ownership" failed_completion;
  test "context retirement rejects unfinished commands" context_retirement;
])
