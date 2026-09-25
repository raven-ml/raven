(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

let fixture ?(profile_offset = fun () -> 0.) synchronize =
  let host = Tolk_cpu.create "CPU:profile-host" in
  let allocator = Device.Allocator.Pack (Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let renderer_set = Device.Renderer_set.make ~device:"CPU:profile"
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let queue = Device.{timestamp_divider = 10.; profile_offset; completion = (fun () () -> ()); prepare = (fun () -> ()); host = Device.name host; max_kernel_bindings = None;
    copy = (fun _ -> None); encode = (fun _ -> None); lower = (fun _ -> None);
    compile = (fun _ -> fail "no compilation expected")} in
  let device = Device.make ~name:"CPU:profile" ~allocator ~renderer_set
      ~runtime:(Device.runtime host) ~synchronize ~queue () in
  let buffer = Device.create_buffer ~size:2 ~dtype:Tolk_uop.Dtype.uint64 device in
  Device.Buffer.ensure_allocated buffer;
  device, buffer

let stamps buffer first last =
  let bytes = Bytes.create 16 in
  Bytes.set_int64_le bytes 0 first;
  Bytes.set_int64_le bytes 8 last;
  Device.Buffer.copyin buffer bytes

let register device buffer name =
  Device.record_timing device ~name ~queue:"COMPUTE:0" ~buffer ~first:0 ~last:1

let render events =
  let path = Filename.temp_file "tolk-profile" ".json" in
  Fun.protect ~finally:(fun () -> Sys.remove path) (fun () ->
      Out_channel.with_open_bin path (fun channel -> Profile.output channel events);
      In_channel.with_open_bin path In_channel.input_all)

let () = run "Profile" [
  test "clock calibration rejects outliers and invalid samples" (fun () ->
      let samples = [|1e6; 1e15; 1e6; -1e15; 1e6|] and count = ref 0 in
      let before = Unix.gettimeofday () *. 1e6 -. 1e6 in
      let offset = Profile.calibrate (fun () () -> let value = samples.(!count) in incr count; value) in
      let after = Unix.gettimeofday () *. 1e6 -. 1e6 in
      equal int 5 !count;
      is_true (before <= offset && offset <= after);
      raises_match (function Invalid_argument _ -> true | _ -> false)
        (fun () -> Profile.calibrate (fun () () -> nan)));
  test "calibration shifts starts, preserves durations and retains failed collections" (fun () ->
      let failed = ref true in
      let profile_offset () = if !failed then failwith "clock unavailable" else 234. in
      let device, buffer = fixture ~profile_offset (fun () -> ()) in
      stamps buffer 100L 250L; register device buffer "pending";
      raises_match (Exn.failure ~substring:"clock unavailable") (fun () -> Device.profile device);
      failed := false;
      let events = Device.profile device in
      equal int 1 (List.length events);
      equal float_exact 244. (List.hd events).Profile.start_us;
      equal float_exact 15. (List.hd events).Profile.duration_us;
      equal int 0 (List.length (Device.profile device)));
  test "asynchronous records wait for synchronization and drain once" (fun () ->
      let syncs = ref 0 in
      let device, buffer = fixture (fun () -> incr syncs) in
      register device buffer "old";
      stamps buffer 100L 250L;
      register device buffer "latest";
      equal int 0 !syncs;
      let events = Device.profile device in
      equal int 1 !syncs;
      equal int 1 (List.length events);
      let event = List.hd events in
      equal string "latest" event.Profile.name;
      equal float_exact 10. event.start_us;
      equal float_exact 15. event.duration_us;
      equal int 0 (List.length (Device.profile device)));
  test "independent batches retain separate records" (fun () ->
      let device, first = fixture (fun () -> ()) in
      let second = Device.create_buffer ~size:2 ~dtype:Tolk_uop.Dtype.uint64 device in
      Device.Buffer.ensure_allocated second;
      stamps first 10L 20L; stamps second 30L 50L;
      register device first "first"; register device second "second";
      let events = Device.profile device in
      equal (list string) ["first"; "second"]
        (List.map (fun e -> e.Profile.name) events |> List.sort String.compare));
  test "failed synchronization retains pending records" (fun () ->
      let failed = ref true in
      let device, buffer = fixture (fun () -> if !failed then failwith "not ready") in
      stamps buffer 40L 70L;
      register device buffer "pending";
      raises_match (Exn.failure ~substring:"not ready") (fun () -> Device.profile device);
      failed := false;
      equal string "pending" (List.hd (Device.profile device)).Profile.name);
  test "pending records retain buffers only until collection" (fun () ->
      let device, weak =
        let device, buffer = fixture (fun () -> ()) in
        stamps buffer 10L 20L;
        register device buffer "owned";
        let weak = Weak.create 1 in
        Weak.set weak 0 (Some buffer);
        device, weak in
      Gc.full_major ();
      is_true (Weak.check weak 0);
      equal int 1 (List.length (Device.profile device));
      Gc.full_major ();
      is_false (Weak.check weak 0));
  test "trace JSON escapes names and separates queue lanes" (fun () ->
      let event = Profile.{device = "NV\"0"; queue = "COPY:0"; name = "copy\n\\\001";
        start_us = 42.; duration_us = 3.5} in
      let result = render [event; {event with queue = "COMPUTE:0"; name = "kernel";
        start_us = 43.; duration_us = 8.}] in
      equal string
        "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"NV\\\"0\"}},{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":0,\"tid\":0,\"args\":{\"name\":\"COMPUTE:0\"}},{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":0,\"tid\":1,\"args\":{\"name\":\"COPY:0\"}},{\"ph\":\"X\",\"cat\":\"queue\",\"name\":\"copy\\n\\\\\\u0001\",\"pid\":0,\"tid\":1,\"ts\":0,\"dur\":3.5},{\"ph\":\"X\",\"cat\":\"queue\",\"name\":\"kernel\",\"pid\":0,\"tid\":0,\"ts\":1,\"dur\":8}]}\n"
        result);
  test "trace rejects invalid durations before writing" (fun () ->
      List.iter (fun duration_us ->
          raises_match (function Invalid_argument _ -> true | _ -> false) (fun () ->
              render [Profile.{device = "NV"; queue = "COMPUTE:0"; name = "k";
                start_us = 0.; duration_us}])) [nan; infinity; -1.]);
  test "trace keeps one origin across devices" (fun () ->
      let event = Profile.{device = "AMD"; queue = "COMPUTE:0"; name = "kernel";
        start_us = 42.; duration_us = 3.} in
      let fields = render [event; {event with device = "NV"; start_us = 52.}]
        |> String.split_on_char ',' in
      is_true (List.mem "\"ts\":0" fields);
      is_true (List.mem "\"ts\":10" fields));
]
