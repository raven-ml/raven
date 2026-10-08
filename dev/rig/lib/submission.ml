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
  = "caml_rig_sub_new_byte" "caml_rig_sub_new"

external sub_part : c -> int -> int -> int array -> int -> unit
  = "caml_rig_sub_part"

external sub_words : c -> int -> int -> int -> unit = "caml_rig_sub_words"

external sub_fill : c -> int -> nativeint -> int -> int -> int -> unit
  = "caml_rig_sub_fill_byte" "caml_rig_sub_fill"

external sub_copy : c -> int -> nativeint * int * nativeint * int * int -> unit
  = "caml_rig_sub_copy"

external sub_fixed : c -> int -> int -> nativeint -> bool -> unit
  = "caml_rig_sub_fixed"

external sub_slot : c -> int -> int -> nativeint -> unit = "caml_rig_sub_slot"
[@@noalloc]

external sub_wait_slot : c -> int -> int -> unit = "caml_rig_sub_wait_slot"
[@@noalloc]

external sub_hold : c -> int -> unit = "caml_rig_sub_hold" [@@noalloc]
external sub_collect : c -> int = "caml_rig_sub_collect"
external sub_point : c -> int -> int = "caml_rig_sub_point" [@@noalloc]
external sub_wait : c -> int -> int -> int -> int -> unit = "caml_rig_sub_wait"
external sub_clear : c -> unit = "caml_rig_sub_clear" [@@noalloc]
external sub_value : c -> int = "caml_rig_sub_value" [@@noalloc]
external sub_no_room_at : c -> int = "caml_rig_sub_no_room_at" [@@noalloc]
external sub_producer : c -> int = "caml_rig_sub_producer" [@@noalloc]
external sub_claims : c -> int array = "caml_rig_sub_claims"
external c_submit : c -> int = "caml_rig_submit"
external sub_take : c -> unit = "caml_rig_sub_take"
external sub_give : c -> unit = "caml_rig_sub_give" [@@noalloc]
external ensure_record : int -> int -> unit = "caml_rig_ensure_record"

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
      invalid_argf "Rig.%s: %s has no queue %S" fn d.name q
    else if d.queues.(i) = q then i
    else go (i + 1)
  in
  go 0

(* The record of [b]'s memory, which holds the stamps a submission raises. *)
let[@inline] entry_of b =
  let m = b.mem.root in
  if m.entry == Memory.no_entry then Memory.ensure_entry m;
  m.entry

let host_address fn b =
  if b.mem.host < 0 then invalid_argf "Rig.%s: the buffer is not host memory" fn;
  b.mem.host + b.offset

let hold_stamps = function Some h -> h.hstamps | None -> 0

(* Refuses a part's buffer that is dead or in a hold other than the
   submission's, whose stamps are [hold_stamps]. *)
let check_buffer fn hold_stamps b =
  Buffer.check_live fn b;
  let e = b.mem.root.entry in
  if e.held && e.stamps <> hold_stamps then
    invalid_argf "Rig.%s: a part names memory of another hold" fn

let make ?hold ~reads ~writes ~waits d parts =
  let fn = "Submission.make" in
  if reads < 0 || writes < 0 || waits < 0 then
    invalid_argf "Rig.%s: a slot count is negative" fn;
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_host d || Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name;
  let hold_stamps = hold_stamps hold in
  let check_buffer = check_buffer fn hold_stamps in
  let nafter = ref 0 and nfixed = ref 0 in
  Array.iteri
    (fun i p ->
      Array.iter
        (fun j ->
          if j < 0 || j >= i then
            invalid_argf "Rig.%s: part %d's after names part %d" fn i j)
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
          if d.copy_queue = None then
            invalid_argf "Rig.%s: %s runs no copies" fn d.name;
          check_buffer src;
          check_buffer dst;
          if Buffer.length src <> Buffer.length dst then
            invalid_argf "Rig.%s: a copy's buffers differ in size" fn;
          if src.mem.dev != d || dst.mem.dev != d then
            invalid_argf "Rig.%s: a copy's buffers are not %s's memory" fn
              d.name;
          nfixed := !nfixed + 2)
    parts;
  let c = sub_new d.c (Array.length parts) !nafter !nfixed reads writes waits in
  (* The hold keeps its stamps while the submission holds it. *)
  Option.iter (fun h -> sub_hold c h.hstamps) hold;
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
  (* No queue reaches io memory itself, only a borrow of its pages. *)
  if Dev.is_io b.mem.dev then
    invalid_argf "Rig.%s: %s's memory is io memory, which no queue reaches" fn
      b.mem.dev.name;
  let e = entry_of b in
  if e.held then invalid_argf "Rig.%s: the buffer's memory is in a hold" fn;
  sub_slot s.c k e.stamps b.mem.handle;
  s.slots.(k) <- b

let read s i b =
  if i < 0 || i >= s.nreads then
    invalid_argf "Rig.Submission.read: %d is no read slot" i;
  set "Submission.read" s i b

let write s i b =
  if i < 0 || i >= Array.length s.slots - s.nreads then
    invalid_argf "Rig.Submission.write: %d is no write slot" i;
  set "Submission.write" s (s.nreads + i) b

let wait_for s i p =
  if i < 0 || i >= s.nwaits then
    invalid_argf "Rig.Submission.wait_for: %d is no wait slot" i;
  sub_wait_slot s.c i p

let clear s =
  sub_clear s.c;
  Array.fill s.slots 0 (Array.length s.slots) unset

(* In-queue waits *)

let rig_word = 0
let rig_object = 1
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
  if (not waits) || not (Dev.same_machine p d) then host_wait
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
            Dev.protect d (fun () -> d.pair_maps <- r :: d.pair_maps);
            let at, _, _ = Memory.region_info r in
            if at < 0 then host_wait else at + skip)

(* [d.pairs] is replaced whole under [d]'s lock and its cells are ints, so a
   read takes no lock: only an undecided pair does. *)
let pair d p =
  let i = p.index in
  let pairs = d.pairs in
  let known = if i < Array.length pairs then pairs.(i) else 0 in
  if known <> 0 then known
  else begin
    let way = decide d p in
    Dev.protect d (fun () ->
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

(* Checks a part's buffers: live and, once a hold exists, in no hold but the
   submission's, whose stamps are [st]. *)
let check_held held st b =
  if held then check_buffer fn st b else Buffer.check_live fn b

let check_part held st p =
  match p.work with
  | Words b -> check_held held st b
  | Fill f -> check_held held st f.arg
  | Copy { src; dst } ->
      check_held held st src;
      check_held held st dst

(* Checks the slots are set and every buffer the work names is live and in no
   hold but the submission's: memory put in a hold after it was named must be
   named with the hold. A process that never made a hold holds no memory, and
   checks liveness only. *)
let check_slots s =
  let held = Atomic.get Memory.any_held in
  let st = hold_stamps s.hold in
  for k = 0 to Array.length s.parts - 1 do
    check_part held st s.parts.(k)
  done;
  for k = 0 to Array.length s.slots - 1 do
    let b = s.slots.(k) in
    if b == unset then invalid_argf "Rig.%s: a read or write slot is unset" fn;
    Buffer.check_live fn b;
    if held && b.mem.root.entry.held then
      invalid_argf "Rig.%s: the buffer's memory is in a hold" fn
  done

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, or that its queue has no room for, adds the others to [s]'s waits, and
   is their number. *)
let rec waits s n i count =
  if i = n then count
  else
    let p = sub_point s.c i in
    let producer = Dev.of_index (Point.index p) and v = Point.value p in
    if v > Dev.submitted producer then
      invalid_argf "Rig.%s: %s's value %d is not submitted" fn producer.name v;
    if Dev.is_lost producer then Dev.raise_lost producer;
    if Dev.point_reached p then waits s n (i + 1) count
    else
      let way =
        if count >= s.dev.max_waits then host_wait else pair s.dev producer
      in
      if way = host_wait then begin
        Dev.wait producer v;
        waits s n (i + 1) count
      end
      else begin
        let kind =
          match producer.completion with
          | Object _ -> rig_object
          | _ -> rig_word
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
    let w = Dev.word d in
    if w < at then Dev.wait d (w + 1);
    hand_over s nwaits
  end
  else if r = need_record then begin
    ensure_record d.c nwaits;
    hand_over s nwaits
  end
  else if r = never then
    invalid_argf
      "Rig.%s: the parts never fit %s's queues, or name work its driver does \
       not run"
      fn d.name
  else if r = producer_lost then
    Dev.raise_lost (Dev.of_index (sub_producer s.c))
  else if r = failed then begin
    Array.iter (fun i -> Dev.stop (Dev.of_index i)) (sub_claims s.c);
    Dev.raise_lost d
  end
  else Dev.raise_lost d

(* A submit holds the submission's guard throughout, so two domains' submits of
   it take turns. A forked child never waits on a guard its parent's thread may
   hold: every submission made before the fork is on a device the child
   inherited, which raises first. *)
let submit s =
  if Dev.inherited s.dev then Dev.raise_lost s.dev;
  sub_take s.c;
  match
    check_slots s;
    hand_over s (waits s (sub_collect s.c) 0 0)
  with
  | p ->
      clear s;
      sub_give s.c;
      p
  | exception e ->
      clear s;
      sub_give s.c;
      raise e
