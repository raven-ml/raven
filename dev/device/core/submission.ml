(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type work =
  | Words of buffer
  | Fill of {
      fill : nativeint;
      arg : buffer;
      ring_units : int;
      segment_bytes : int;
    }
  | Copy of { src : buffer; dst : buffer }

type part = { queue : string; after : int array; work : work }

(* The C form *)

(* The C form, which a custom block holds and frees once collected. *)
type c

external sub_new : int -> int -> int -> int -> int -> int -> int -> c
  = "caml_device_core_sub_new_byte" "caml_device_core_sub_new"

external sub_part : c -> int -> int -> int array -> int -> unit
  = "caml_device_core_sub_part"

external sub_words : c -> int -> int -> int -> unit
  = "caml_device_core_sub_words"

external sub_fill : c -> int -> nativeint -> int -> int -> int -> unit
  = "caml_device_core_sub_fill_byte" "caml_device_core_sub_fill"

external sub_copy : c -> int -> nativeint * int * nativeint * int * int -> unit
  = "caml_device_core_sub_copy"

external sub_fixed : c -> int -> int -> nativeint -> bool -> unit
  = "caml_device_core_sub_fixed"

external sub_slot : c -> int -> int -> nativeint -> unit
  = "caml_device_core_sub_slot"
[@@noalloc]

external sub_wait_slot : c -> int -> int -> unit
  = "caml_device_core_sub_wait_slot"
[@@noalloc]

external sub_hold : c -> int -> unit = "caml_device_core_sub_hold" [@@noalloc]
external sub_collect : c -> int = "caml_device_core_sub_collect"
external sub_point : c -> int -> int = "caml_device_core_sub_point" [@@noalloc]

external sub_wait : c -> int -> int -> int -> int -> unit
  = "caml_device_core_sub_wait"

external sub_clear : c -> unit = "caml_device_core_sub_clear" [@@noalloc]
external sub_value : c -> int = "caml_device_core_sub_value" [@@noalloc]

external sub_no_room_at : c -> int = "caml_device_core_sub_no_room_at"
[@@noalloc]

external sub_producer : c -> int = "caml_device_core_sub_producer" [@@noalloc]
external sub_claims : c -> int array = "caml_device_core_sub_claims"
external c_submit : c -> int = "caml_device_core_submit"
external ensure_record : int -> int -> unit = "caml_device_core_ensure_record"

type t = {
  dev : device;
  c : c;
  parts : part array;  (** Its buffers are checked live at each submit. *)
  slots : buffer array;  (** The read slots, then the write slots. *)
  nreads : int;
  nwaits : int;
  hold : hold option;
}

(* The slot no submit has set. *)
let unset =
  let mem = Memory.make Dev.host 0 Memory.no_entry in
  { mem; offset = 0; length = 0; generation = -1 }

let queue_index d fn q =
  let rec go i =
    if i = Array.length d.queues then
      invalid_argf "Device_core.%s: %s has no queue %S" fn d.name q
    else if d.queues.(i) = q then i
    else go (i + 1)
  in
  go 0

(* The record of [b]'s memory, which holds the stamps a submission raises. *)
let entry_of b =
  let m = b.mem.root in
  if m.entry == Memory.no_entry then Memory.ensure_entry m;
  m.entry

let host_address fn b =
  if b.mem.host < 0 then
    invalid_argf "Device_core.%s: the buffer is not host memory" fn;
  b.mem.host + b.offset

let make ?hold ~reads ~writes ~waits d parts =
  let fn = "Submission.make" in
  if reads < 0 || writes < 0 || waits < 0 then
    invalid_argf "Device_core.%s: a slot count is negative" fn;
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_host d || Dev.is_io d then
    invalid_argf "Device_core.%s: %s runs no submitted work" fn d.name;
  let hold_stamps = match hold with Some h -> h.hstamps | None -> 0 in
  let check_buffer b =
    Buffer.check_live fn b;
    let e = b.mem.root.entry in
    if e.held && e.stamps <> hold_stamps then
      invalid_argf "Device_core.%s: a part names memory of another hold" fn
  in
  let nafter = ref 0 and nfixed = ref 0 in
  Array.iteri
    (fun i p ->
      Array.iter
        (fun j ->
          if j < 0 || j >= i then
            invalid_argf "Device_core.%s: part %d's after names part %d" fn i j)
        p.after;
      nafter := !nafter + Array.length p.after;
      ignore (queue_index d fn p.queue);
      match p.work with
      | Words w ->
          check_buffer w;
          ignore (host_address fn w);
          incr nfixed
      | Fill f ->
          check_buffer f.arg;
          ignore (host_address fn f.arg);
          incr nfixed
      | Copy { src; dst } ->
          (* A driver that lists no copy queue runs no copy. *)
          if not (Dev.copies d) then
            invalid_argf "Device_core.%s: %s runs no copies" fn d.name;
          check_buffer src;
          check_buffer dst;
          if Buffer.length src <> Buffer.length dst then
            invalid_argf "Device_core.%s: a copy's buffers differ in size" fn;
          if src.mem.dev != d || dst.mem.dev != d then
            invalid_argf "Device_core.%s: a copy's buffers are not %s's memory"
              fn d.name;
          nfixed := !nfixed + 2)
    parts;
  let c = sub_new d.c (Array.length parts) !nafter !nfixed reads writes waits in
  let at = ref 0 and k = ref 0 in
  let fixed b write =
    sub_fixed c !k (entry_of b).stamps b.mem.handle write;
    incr k
  in
  Array.iteri
    (fun i p ->
      sub_part c i (queue_index d fn p.queue) p.after !at;
      at := !at + Array.length p.after;
      match p.work with
      | Words w ->
          sub_words c i (host_address fn w) (Buffer.length w / 4);
          fixed w false
      | Fill f ->
          sub_fill c i f.fill (host_address fn f.arg) f.ring_units
            f.segment_bytes;
          fixed f.arg false
      | Copy { src; dst } ->
          sub_copy c i
            ( dst.mem.handle,
              dst.offset,
              src.mem.handle,
              src.offset,
              Buffer.length src );
          fixed src false;
          fixed dst true)
    parts;
  {
    dev = d;
    c;
    parts;
    slots = Array.make (reads + writes) unset;
    nreads = reads;
    nwaits = waits;
    hold;
  }

let set fn s k b =
  let e = entry_of b in
  if e.held then
    invalid_argf "Device_core.%s: the buffer's memory is in a hold" fn;
  sub_slot s.c k e.stamps b.mem.handle;
  s.slots.(k) <- b

let read s i b =
  if i < 0 || i >= s.nreads then
    invalid_argf "Device_core.Submission.read: %d is no read slot" i;
  set "Submission.read" s i b

let write s i b =
  if i < 0 || i >= Array.length s.slots - s.nreads then
    invalid_argf "Device_core.Submission.write: %d is no write slot" i;
  set "Submission.write" s (s.nreads + i) b

let wait_for s i p =
  if i < 0 || i >= s.nwaits then
    invalid_argf "Device_core.Submission.wait_for: %d is no wait slot" i;
  sub_wait_slot s.c i p

let clear s =
  sub_clear s.c;
  Array.fill s.slots 0 (Array.length s.slots) unset

(* In-queue waits *)

let nx_word = 0
let nx_object = 1
let host_wait = -1

(* How [d] waits for [p]'s values, decided once per pair: [host_wait], or the
   address of [p]'s word as [d] maps it. *)
let decide d p =
  let waits =
    match p.completion with
    | Store -> d.waits_store
    | Object _ -> d.waits_object && d.key = p.key
    | Host_writes -> d.waits_host
  in
  if (not waits) || p.machine <> d.machine then host_wait
  else
    match (p.completion, p.word_region) with
    | Object o, _ -> o
    | _, None -> host_wait
    | _, Some word_region -> (
        (* A device of the producer's driver maps its word region; another maps
           the host page that holds the word. *)
        let start = p.word - (p.word mod Memory.page) in
        let mapped, skip =
          if d.key = p.key then (Memory.map_peer_region d word_region, 0)
          else (Memory.map_host_range d start Memory.page, p.word - start)
        in
        match mapped with
        | None -> host_wait
        | Some r ->
            Mutex.protect d.lock (fun () -> d.pair_maps <- r :: d.pair_maps);
            let at, _, _ = Memory.region_info r in
            if at < 0 then host_wait else at + skip)

let pair d p =
  let i = p.index in
  let known =
    Mutex.protect d.lock (fun () ->
        if i < Array.length d.pairs then d.pairs.(i) else 0)
  in
  if known <> 0 then known
  else begin
    let way = decide d p in
    Mutex.protect d.lock (fun () ->
        if i >= Array.length d.pairs then begin
          let a = Array.make (Int.max (i + 1) (2 * Array.length d.pairs)) 0 in
          Array.blit d.pairs 0 a 0 (Array.length d.pairs);
          d.pairs <- a
        end;
        d.pairs.(i) <- way);
    way
  end

(* Submit *)

let ok = 0
let busy = 1
let no_room = 2
let never = 3
let producer_lost = 6
let failed = 7
let need_record = 8
let stop_claimed = 16
let fn = "submit"

let check_part p =
  match p.work with
  | Words b -> Buffer.check_live fn b
  | Fill f -> Buffer.check_live fn f.arg
  | Copy { src; dst } ->
      Buffer.check_live fn src;
      Buffer.check_live fn dst

(* Checks the slots are set and every buffer the work names is live, and gives
   the C form the hold's stamps, which the hold keeps while the submission holds
   it. *)
let check_slots s =
  for k = 0 to Array.length s.parts - 1 do
    check_part s.parts.(k)
  done;
  (match s.hold with Some h -> sub_hold s.c h.hstamps | None -> ());
  for k = 0 to Array.length s.slots - 1 do
    let b = s.slots.(k) in
    if b == unset then
      invalid_argf "Device_core.%s: a read or write slot is unset" fn;
    Buffer.check_live fn b
  done

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, adds the others to [s]'s waits, and is their number. *)
let rec waits s n i count =
  if i = n then count
  else
    let p = sub_point s.c i in
    let producer = Dev.of_index (Point.index p) and v = Point.value p in
    if v > Dev.submitted producer then
      invalid_argf "Device_core.%s: %s's value %d is not submitted" fn
        producer.name v;
    if Dev.is_lost producer then Dev.raise_lost producer;
    if Dev.point_reached p then waits s n (i + 1) count
    else
      let way = pair s.dev producer in
      if way = host_wait then begin
        Dev.wait producer v;
        waits s n (i + 1) count
      end
      else begin
        let kind =
          match producer.completion with Object _ -> nx_object | _ -> nx_word
        in
        sub_wait s.c producer.index way v kind;
        waits s n (i + 1) (count + 1)
      end

let rec hand_over s nwaits =
  let d = s.dev in
  let r = c_submit s.c in
  if r land stop_claimed <> 0 then Dev.stop d;
  let r = r land (stop_claimed - 1) in
  if r = ok then Point.make d.index (sub_value s.c)
  else if r = busy then hand_over s nwaits
  else if r = no_room then begin
    let at = sub_no_room_at s.c in
    let w = Dev.signaled d in
    if w < at then Dev.wait d (w + 1);
    hand_over s nwaits
  end
  else if r = need_record then begin
    ensure_record d.c nwaits;
    hand_over s nwaits
  end
  else if r = never then
    invalid_argf
      "Device_core.%s: the parts never fit %s's queues, or name work its \
       driver does not run"
      fn d.name
  else if r = producer_lost then
    Dev.raise_lost (Dev.of_index (sub_producer s.c))
  else if r = failed then begin
    Array.iter (fun i -> Dev.stop (Dev.of_index i)) (sub_claims s.c);
    Dev.raise_lost d
  end
  else Dev.raise_lost d

let submit s =
  match
    check_slots s;
    let nwaits = waits s (sub_collect s.c) 0 0 in
    if nwaits > 0 then ensure_record s.dev.c nwaits;
    hand_over s nwaits
  with
  | p ->
      clear s;
      p
  | exception e ->
      clear s;
      raise e
