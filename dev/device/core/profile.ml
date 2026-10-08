(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type event = Def.event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { program : program; binary : string; time : int }
  | Counters of {
      device : device;
      name : string;
      start : int;
      stop : int;
      counters : (string * int array) list;
    }
  | Trace of {
      device : device;
      name : string;
      start : int;
      stop : int;
      part : int;
      data : string;
    }
  | Overwritten of { device : device; time : int; runs : int }
  | Copy of { src : device; dst : device; bytes : int; start : int; stop : int }

external timestamp : unit -> nativeint = "caml_device_core_timestamp"
external load64 : int -> int = "caml_device_core_load64"

let timestamp = timestamp ()
let now = Prof.now
let enabled = Prof.enabled

let time = function
  | Span s -> s.start
  | Allocation a -> a.time
  | Load l -> l.time
  | Counters c -> c.start
  | Trace t -> t.start
  | Overwritten o -> o.time
  | Copy c -> c.start

let duration = function
  | Span s -> s.stop - s.start
  | Counters c -> c.stop - c.start
  | Trace t -> t.stop - t.start
  | Copy c -> c.stop - c.start
  | Allocation _ | Load _ | Overwritten _ -> 0

(* Time order; at equal times longest first, then as recorded. *)
let order (n, e) (n', e') =
  match Int.compare (time e) (time e') with
  | 0 -> (
      match Int.compare (duration e') (duration e) with
      | 0 -> Int.compare n n'
      | c -> c)
  | c -> c

let counters () =
  List.fold_left
    (fun acc (p : Prof.t) ->
      List.fold_left
        (fun acc c -> if List.mem c acc then acc else acc @ [ c ])
        acc p.counters)
    [] (Prof.active ())

let traced () = List.exists (fun (p : Prof.t) -> p.trace) (Prof.active ())

(* Waits for the points whose events are still to be read; a device lost
   meanwhile has them dropped. *)
let read_pending () =
  Array.iteri
    (fun i d ->
      if i = d.index && d.afters <> [] then
        let v = List.fold_left (fun m (v, _) -> Int.max m v) 0 d.afters in
        try Dev.wait d v
        with Dev.Lost _ -> Mutex.protect d.lock (fun () -> d.afters <- []))
    (Dev.all ())

let take ?(counters = []) ?(trace = false) f =
  let rec dup = function
    | [] -> ()
    | c :: cs ->
        if List.mem c cs then
          invalid_argf "Device_core.Profile.take: counter %S is named twice" c;
        dup cs
  in
  dup counters;
  let p = Prof.start ~counters ~trace in
  match f () with
  | r ->
      Fun.protect ~finally:(fun () -> Prof.stop p) read_pending;
      let events = Mutex.protect p.lock (fun () -> p.events) in
      (r, List.map snd (List.stable_sort order events))
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Prof.stop p;
      Printexc.raise_with_backtrace e bt

let span name f =
  match Prof.active () with
  | [] -> f ()
  | ps ->
      let lane = Printf.sprintf "domain %d" (Domain.self () :> int) in
      let start = now () in
      let finish () =
        Prof.add ps
          (Span { device = Dev.host; lane; name; start; stop = now () })
      in
      Fun.protect ~finally:finish f

let after p f =
  match Prof.active () with
  | [] -> ()
  | ps ->
      let d = Dev.of_index (Point.index p) in
      Dev.after d (Point.value p) (fun () -> Prof.add_all ps (f ()))

let record p ~lane ~name stamps =
  if
    Buffer.nbytes stamps <> 32
    || stamps.mem.host < 0
    || (stamps.mem.host + stamps.offset) mod 8 <> 0
  then
    invalid_arg
      "Device_core.Profile.record: the stamps are not 32 aligned bytes of host \
       memory";
  let device = Dev.of_index (Point.index p) in
  let at = stamps.mem.host + stamps.offset in
  after p (fun () ->
      ignore (Sys.opaque_identity stamps);
      [
        Span
          {
            device;
            lane;
            name;
            start = load64 (at + 8);
            stop = load64 (at + 24);
          };
      ])

(* Chrome's trace event format *)

(* [s] as a JSON string; malformed UTF-8 becomes U+FFFD. *)
let string oc s =
  output_char oc '"';
  let rec go i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let n = Uchar.utf_decode_length d in
      (if not (Uchar.utf_decode_is_valid d) then output_string oc "\u{FFFD}"
       else
         match s.[i] with
         | '"' -> output_string oc "\\\""
         | '\\' -> output_string oc "\\\\"
         | '\n' -> output_string oc "\\n"
         | '\r' -> output_string oc "\\r"
         | '\t' -> output_string oc "\\t"
         | c when Char.code c < 0x20 ->
             Printf.fprintf oc "\\u%04x" (Char.code c)
         | _ -> output_substring oc s i n);
      go (i + n)
    end
  in
  go 0;
  output_char oc '"'

(* [ns] nanoseconds in microseconds. *)
let micros oc ns =
  if ns < 0 then output_char oc '-';
  Printf.fprintf oc "%d.%03d" (abs ns / 1000) (abs ns mod 1000)

let device_of = function
  | Span s -> s.device
  | Allocation a -> a.device
  | Load l -> l.program.pdev
  | Counters c -> c.device
  | Trace t -> t.device
  | Overwritten o -> o.device
  | Copy c -> c.src

let output_chrome_trace oc events =
  let events =
    List.mapi (fun i e -> (i, e)) events
    |> List.stable_sort order |> List.map snd
  in
  let origin = match events with [] -> 0 | e :: _ -> time e in
  let pids = Hashtbl.create 8 and tids = Hashtbl.create 8 in
  let first = ref true in
  let next () = if !first then first := false else output_string oc ",\n" in
  let meta ~pid ~tid what name =
    next ();
    Printf.fprintf oc
      "{\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"name\":\"%s\",\"args\":{\"name\":"
      pid tid what;
    string oc name;
    output_string oc "}}"
  in
  let pid d =
    match Hashtbl.find_opt pids d.index with
    | Some pid -> pid
    | None ->
        let pid = Hashtbl.length pids + 1 in
        Hashtbl.add pids d.index pid;
        meta ~pid ~tid:0 "process_name" d.name;
        pid
  in
  let tid d pid lane =
    match Hashtbl.find_opt tids (d.index, lane) with
    | Some tid -> tid
    | None ->
        let tid = Hashtbl.length tids + 1 in
        Hashtbl.add tids (d.index, lane) tid;
        meta ~pid ~tid "thread_name" lane;
        tid
  in
  output_string oc "{\"traceEvents\":[\n";
  List.iter
    (fun e ->
      let d = device_of e in
      let pid = pid d in
      let tid =
        match e with
        | Span s -> tid d pid s.lane
        | Counters _ -> tid d pid "counters"
        | Copy _ -> tid d pid "copy"
        | Allocation _ | Load _ | Trace _ | Overwritten _ -> 0
      in
      next ();
      let ph =
        match e with
        | Span _ | Counters _ | Copy _ -> "X"
        | Allocation _ -> "C"
        | Load _ | Trace _ | Overwritten _ -> "i"
      in
      Printf.fprintf oc "{\"ph\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":" ph pid
        tid;
      micros oc (time e - origin);
      (match e with
      | Span s ->
          output_string oc ",\"dur\":";
          micros oc (s.stop - s.start);
          output_string oc ",\"name\":";
          string oc s.name
      | Counters c ->
          output_string oc ",\"dur\":";
          micros oc (c.stop - c.start);
          output_string oc ",\"name\":";
          string oc c.name;
          output_string oc ",\"args\":{";
          List.iteri
            (fun i (name, values) ->
              if i > 0 then output_char oc ',';
              string oc name;
              Printf.fprintf oc ":%d" (Array.fold_left ( + ) 0 values))
            c.counters;
          output_char oc '}'
      | Copy c ->
          output_string oc ",\"dur\":";
          micros oc (c.stop - c.start);
          output_string oc ",\"name\":\"copy\",\"args\":{\"to\":";
          string oc c.dst.name;
          Printf.fprintf oc ",\"bytes\":%d}" c.bytes
      | Trace t ->
          output_string oc ",\"s\":\"p\",\"name\":";
          string oc t.name;
          Printf.fprintf oc ",\"args\":{\"part\":%d,\"bytes\":%d}" t.part
            (String.length t.data)
      | Overwritten o ->
          Printf.fprintf oc
            ",\"s\":\"p\",\"name\":\"overwritten\",\"args\":{\"runs\":%d}"
            o.runs
      | Allocation a ->
          Printf.fprintf oc ",\"name\":\"memory\",\"args\":{\"allocated\":%d}"
            a.allocated
      | Load _ -> output_string oc ",\"s\":\"p\",\"name\":\"load\"");
      output_char oc '}')
    events;
  output_string oc "\n]}\n"
