(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf

type event = Def.event =
  | Span of {
      device : device;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : device; time : int; allocated : int }
  | Load of { image : image; binary : string; time : int }
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

(* Taking *)

type t = {
  events : (int * event) list option Atomic.t;
      (* [None] is closed. Closing detaches history from pending callbacks. *)
  pending : intake Atomic.t;
}

(* A profile's entries: it takes them while [take]'s function runs, then
   only gives up those it took to [take]'s reads, until its stop. *)
and intake = Open of pending list | Reading of pending list | Closed

and pending = {
  device : device;
  value : int;
  read : t -> unit;
  drop : unit -> unit;
}

(* The host's clock, [rig_now_ns] of [rig_edge.h]. *)
external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

let profiles : t list Atomic.t = Atomic.make []
let numbers = Atomic.make 0
let active () = Atomic.get profiles
let is_open p = Option.is_some (Atomic.get p.events)

let enabled () =
  match Atomic.get profiles with [] -> false | ps -> List.exists is_open ps

(* Drops [p]'s pending entries one at a time, then takes no more. Each
   entry leaves [p] before its drop, with nothing that allocates or polls
   between, so an exception between two drops leaves the rest in [p] and
   none is dropped twice. *)
let rec close p =
  match Atomic.get p.pending with
  | (Open [] | Reading []) as before ->
      if not (Atomic.compare_and_set p.pending before Closed) then close p
  | (Open (e :: l) | Reading (e :: l)) as before ->
      if Atomic.compare_and_set p.pending before (Reading l) then e.drop ();
      close p
  | Closed -> ()

let close_stopped p = if not (is_open p) then close p

(* Sets the list to [f] of it without the stopped profiles, once their
   entries are dropped: a stop that an exception ended leaves its profile in
   the list with the entries it did not drop. *)
let rec change f =
  let ps = Atomic.get profiles in
  List.iter close_stopped ps;
  if not (Atomic.compare_and_set profiles ps (f (List.filter is_open ps))) then
    change f

let start () =
  let p = { events = Atomic.make (Some []); pending = Atomic.make (Open []) } in
  change (fun ps -> ps @ [ p ]);
  p

(* The exchange is the stop: an exception that ends the call after it, at an
   allocation or a poll, leaves [p] in the list stopped, with the entries it
   did not drop, and the next change drops them and [p]. *)
let stop p =
  let events = Atomic.exchange p.events None in
  close p;
  change Fun.id;
  match events with None -> [] | Some events -> events

let rec pend p e =
  match Atomic.get p.pending with
  | Open l as before ->
      if not (Atomic.compare_and_set p.pending before (Open (e :: l))) then
        pend p e
  | Reading _ | Closed -> e.drop ()

(* Takes no more entries into [p]: [take]'s function returned. *)
let rec seal p =
  match Atomic.get p.pending with
  | Open l as before ->
      if not (Atomic.compare_and_set p.pending before (Reading l)) then seal p
  | Reading _ | Closed -> ()

(* What [pop] answers when [p] has no pending entry. *)
let no_pending = { device = Dev.host; value = 0; read = ignore; drop = ignore }

let rec pop p =
  match Atomic.get p.pending with
  | Reading (e :: l) as before ->
      if Atomic.compare_and_set p.pending before (Reading l) then e else pop p
  | Open _ | Reading [] | Closed -> no_pending

let rec push a x =
  match Atomic.get a with
  | None -> ()
  | Some l as before ->
      if not (Atomic.compare_and_set a before (Some (x :: l))) then push a x

let add ps e =
  let n = Atomic.fetch_and_add numbers 1 in
  List.iter (fun p -> push p.events (n, e)) ps

let record e = match active () with [] -> () | ps -> add ps e

(* Events *)

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

(* Time order; at equal times longest first. *)
let order e e' =
  match Int.compare (time e) (time e') with
  | 0 -> Int.compare (duration e') (duration e)
  | c -> c

let recorded_order (n, e) (n', e') =
  match order e e' with 0 -> Int.compare n n' | c -> c

(* Reads the entries [p] took into it, newest first, which waits for the
   last value first. Each waits for its value and reads its events, or none
   if its device is lost meanwhile. An entry leaves [p] as its wait starts
   and is dropped however its read ends, so an exception, such as
   [Sys.Break] in the wait, leaves the rest in [p] for [stop] to drop. *)
let rec read_all p =
  let e = pop p in
  if e != no_pending then begin
    (match
       match Dev.wait e.device e.value with
       | () -> e.read p
       | exception Dev.Lost _ -> ()
     with
    | () -> e.drop ()
    | exception x ->
        e.drop ();
        Printexc.raise_with_backtrace x (Printexc.get_raw_backtrace ()));
    read_all p
  end

let take f =
  let p = start () in
  match
    let r = f () in
    seal p;
    read_all p;
    r
  with
  | r ->
      let events = stop p in
      (r, List.map snd (List.stable_sort recorded_order events))
  | exception e ->
      (* Stopped before the backtrace is taken, whose allocation an
         asynchronous exception may interrupt. *)
      ignore (stop p);
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let span name f =
  match if enabled () then active () else [] with
  | [] -> f ()
  | ps ->
      let lane = strf "domain %d" (Domain.self () :> int) in
      let start = now () in
      let finish () =
        add ps
          (Span { device = Dev.host; lane; name; start; stop = now () })
      in
      Fun.protect ~finally:finish f

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
  | Load l -> l.image.idev
  | Counters c -> c.device
  | Trace t -> t.device
  | Overwritten o -> o.device
  | Copy c -> c.src

let output_chrome_trace oc events =
  let events = List.stable_sort order events in
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
    match Hashtbl.find_opt pids d.opened with
    | Some pid -> pid
    | None ->
        let pid = Hashtbl.length pids + 1 in
        Hashtbl.add pids d.opened pid;
        meta ~pid ~tid:0 "process_name" d.name;
        pid
  in
  let tid d pid lane =
    match Hashtbl.find_opt tids (d.opened, lane) with
    | Some tid -> tid
    | None ->
        let tid = Hashtbl.length tids + 1 in
        Hashtbl.add tids (d.opened, lane) tid;
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
