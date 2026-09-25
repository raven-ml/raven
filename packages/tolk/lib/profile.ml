(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type event = {
  device : string;
  queue : string;
  name : string;
  start_us : float;
  duration_us : float;
}

let calibrate sample =
  let offsets = Array.init 5 (fun _ ->
      let read = sample () in
      let before = Unix.gettimeofday () *. 1e6 in
      let device = read () in
      let after = Unix.gettimeofday () *. 1e6 in
      if not (Float.is_finite device) then invalid_arg "Profile.calibrate: invalid clock sample";
      before +. (after -. before) /. 2. -. device) in
  Array.sort Float.compare offsets;
  offsets.(2)

let string channel value =
  output_char channel '"';
  String.iter (function
    | '"' -> output_string channel "\\\""
    | '\\' -> output_string channel "\\\\"
    | '\n' -> output_string channel "\\n"
    | '\r' -> output_string channel "\\r"
    | '\t' -> output_string channel "\\t"
    | c when Char.code c < 0x20 -> Printf.fprintf channel "\\u%04x" (Char.code c)
    | c -> output_char channel c) value;
  output_char channel '"'

let output channel events =
  let devices = List.map (fun e -> e.device) events |> List.sort_uniq String.compare in
  let origin = ref infinity in
  List.iter (fun event ->
      if not (Float.is_finite event.start_us && Float.is_finite event.duration_us)
         || event.start_us < 0. || event.duration_us < 0. then
        invalid_arg "Profile.output: invalid timing";
      if not (List.for_all String.is_valid_utf_8 [event.device; event.queue; event.name]) then
        invalid_arg "Profile.output: names must be UTF-8";
      origin := min !origin event.start_us) events;
  let ids = List.mapi (fun id device -> device, id) devices in
  let queues = List.map (fun e -> e.device, e.queue) events |> List.sort_uniq compare
    |> List.mapi (fun id key -> key, id) in
  output_string channel "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[";
  let first = ref true in
  let separator () = if !first then first := false else output_char channel ',' in
  List.iter (fun (device, id) ->
      separator ();
      Printf.fprintf channel "{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"tid\":0,\"args\":{\"name\":" id;
      string channel device;
      output_string channel "}}") ids;
  List.iter (fun ((device, queue), id) ->
      separator ();
      Printf.fprintf channel "{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,\"args\":{\"name\":"
        (List.assoc device ids) id;
      string channel queue;
      output_string channel "}}") queues;
  List.iter (fun event ->
      separator ();
      output_string channel "{\"ph\":\"X\",\"cat\":\"queue\",\"name\":";
      string channel event.name;
      Printf.fprintf channel ",\"pid\":%d,\"tid\":%d,\"ts\":%.17g,\"dur\":%.17g}"
        (List.assoc event.device ids) (List.assoc (event.device, event.queue) queues)
        (event.start_us -. !origin)
        event.duration_us) events;
  output_string channel "]}\n"
