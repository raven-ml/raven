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

external sub_new : int -> int -> int -> int -> int -> int -> c
  = "caml_rig_sub_new_byte" "caml_rig_sub_new"

external sub_part : c -> int -> int -> int array -> int -> unit
  = "caml_rig_sub_part"

external sub_words : c -> int -> int -> int -> unit = "caml_rig_sub_words"

external sub_fill : c -> int -> nativeint -> int -> int -> int -> unit
  = "caml_rig_sub_fill_byte" "caml_rig_sub_fill"

external sub_copy :
  c -> int -> nativeint * int * nativeint * int * int * int -> unit
  = "caml_rig_sub_copy"

external sub_fixed : c -> int -> int -> nativeint -> bool -> unit
  = "caml_rig_sub_fixed"

external sub_slot : c -> int -> int -> nativeint -> unit = "caml_rig_sub_slot"
[@@noalloc]

external sub_hold : c -> int -> unit = "caml_rig_sub_hold" [@@noalloc]
external sub_collect : c -> int array -> int = "caml_rig_sub_collect"
external sub_point : c -> int -> int = "caml_rig_sub_point" [@@noalloc]
external sub_wait : c -> int -> int -> int -> int -> unit = "caml_rig_sub_wait"
external sub_clear : c -> unit = "caml_rig_sub_clear" [@@noalloc]
external sub_value : c -> int = "caml_rig_sub_value" [@@noalloc]
external sub_no_room_at : c -> int = "caml_rig_sub_no_room_at" [@@noalloc]
external sub_producer : c -> int = "caml_rig_sub_producer" [@@noalloc]
external c_submit : c -> int = "caml_rig_submit"
external sub_take : c -> unit = "caml_rig_sub_take"
external sub_give : c -> unit = "caml_rig_sub_give" [@@noalloc]
external ensure_record : int -> int -> unit = "caml_rig_ensure_record"

(* The hold whose memory a submission's parts may name, and whose stamps it
   raises: a {!Hold.t}, which the submission keeps reachable, as its release
   frees what the parts run; or, for the library's own copy, the stamps of the
   hold its memory is in, as the copy runs none of the hold's work. *)
type named = No_hold | Hold of hold | Stamps of int

let named_stamps = function
  | No_hold -> 0
  | Hold h -> h.hstamps
  | Stamps st -> st

type t = {
  dev : device;
  c : c;
  parts : part array;  (** Its buffers are checked live at each submit. *)
  nreads : int;  (** The buffers each run reads. *)
  nwrites : int;  (** The buffers each run writes. *)
  named : named;  (** The hold whose memory its parts may name. *)
}

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

(* Memory this process's host addresses, which a copy on a device of another
   machine names by its host address ([rig_edge.h]'s [copy_local]). *)
let local b = Option.is_none b.mem.dev.machine && b.mem.host >= 0
let local_none = 0
let local_src = 1
let local_dst = 2

(* Which side of a copy on [d] is this process's memory: on a device of another
   machine, one side may be. *)
let copy_local d src dst =
  let far = Option.is_some d.machine in
  if src.mem.dev == d && dst.mem.dev == d then Some local_none
  else if far && local src && dst.mem.dev == d then Some local_src
  else if far && src.mem.dev == d && local dst then Some local_dst
  else None

(* Refuses a part's buffer that is dead or in a hold other than the one whose
   stamps are [hold_stamps]: held memory's stamps are its hold's. *)
let check_buffer fn hold_stamps b =
  Buffer.check_live fn b;
  let e = b.mem.root.entry in
  if e.held && e.stamps <> hold_stamps then
    invalid_argf "Rig.%s: a part names memory of another hold" fn

let build named ~reads ~writes d parts =
  let fn = "Submission.make" in
  if reads < 0 || writes < 0 then invalid_argf "Rig.%s: a count is negative" fn;
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_host d || Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name;
  let hold_stamps = named_stamps named in
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
          if copy_local d src dst = None then
            invalid_argf "Rig.%s: a copy's buffers are not %s's memory" fn
              d.name;
          if Buffer.access dst = Read then
            invalid_argf "Rig.%s: a copy's dst admits only reads" fn;
          nfixed := !nfixed + 2)
    parts;
  let c = sub_new d.c (Array.length parts) !nafter !nfixed reads writes in
  (* The hold keeps its stamps while the submission holds it. *)
  if hold_stamps <> 0 then sub_hold c hold_stamps;
  let at = ref 0 and k = ref 0 in
  (* Only [d]'s own memory names a handle of its driver ([rig_edge.h]'s
     [handles]): another's, such as this process's memory a copy on another
     machine's device names, keeps its stamps alone. *)
  let fixed b write =
    let handle = if b.mem.dev == d then b.mem.handle else 0n in
    sub_fixed c !k (entry_of b).stamps handle write;
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
          let side = Option.get (copy_local d src dst) in
          let handle b k =
            if side = k then Nativeint.of_int b.mem.host else b.mem.handle
          in
          sub_copy c i
            ( handle dst local_dst,
              dst.offset,
              handle src local_src,
              src.offset,
              Buffer.length src,
              side );
          fixed src false;
          fixed dst true)
    parts;
  { dev = d; c; parts; nreads = reads; nwrites = writes; named }

let make ?hold ~reads ~writes d parts =
  let named = match hold with Some h -> Hold h | None -> No_hold in
  build named ~reads ~writes d parts

(* In-queue waits *)

let rig_word = 0
let rig_object = 1
let host_wait = -1

(* How [d] waits for [p]'s values, decided once per pair: [host_wait], or for a
   queue that waits in it, the producer's object for an object completion, or
   the address of [p]'s word as [d] maps it. *)
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
            Dev.protect d (fun () -> d.pair_maps <- (p.index, r) :: d.pair_maps);
            let at, _, _ = Memory.region_info r in
            if at < 0 then host_wait else at + skip)

(* An undecided pair: no way is negative but [host_wait], and an address or an
   object fits in 62 bits. *)
let undecided = min_int

(* [d.pairs] is replaced whole under [d]'s lock and its cells are ints, so a
   read takes no lock: only an undecided pair does. *)
let pair d p =
  let i = p.index in
  let pairs = d.pairs in
  let known = if i < Array.length pairs then pairs.(i) else undecided in
  if known <> undecided then known
  else begin
    let way = decide d p in
    Dev.protect d (fun () ->
        if i >= Array.length d.pairs then begin
          let a =
            Array.make (Int.max (i + 1) (2 * Array.length d.pairs)) undecided
          in
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
let need_record = 8
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

(* Checks the parts' buffers are live and, once a hold exists, in no hold but
   the submission's: memory put in a hold after it was named must be named with
   the hold. A process that never made a hold holds no memory, and checks
   liveness only. *)
let check_parts s held =
  for k = 0 to Array.length s.parts - 1 do
    check_part held (named_stamps s.named) s.parts.(k)
  done

let counted n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")

let check_counts s reads writes =
  let nr = Array.length reads and nw = Array.length writes in
  if nr <> s.nreads || nw <> s.nwrites then
    invalid_argf "Rig.%s: %s and %s for a submission of %s and %s" fn
      (counted nr "read") (counted nw "write") (counted s.nreads "read")
      (counted s.nwrites "write")

(* Refuses [b], element [i] of the run's array of [access] ([reads] or
   [writes]), unless it is live, on [s]'s device, once a hold exists in no hold,
   and, written, of memory that admits writes; and hands its stamps and handle
   to the C slot [k]. Memory of another device that is lost raises its loss;
   [s]'s device's own loss is the hand-over's. *)
let name_one s held (access : access) i k b =
  let what = match access with Read -> "reads" | Read_write -> "writes" in
  if not (Buffer.is_live b) then
    invalid_argf "Rig.%s: %s.(%d) is dead: %s" fn what i b.mem.claim.why;
  (* A queue reaches other memory only through a borrow on its device. *)
  if b.mem.dev != s.dev then
    invalid_argf "Rig.%s: %s.(%d) is on %s, not on %s: borrow it" fn what i
      b.mem.dev.name s.dev.name;
  let m = b.mem.root in
  if m != b.mem && m.dev != s.dev && Dev.is_lost m.dev then Dev.raise_lost m.dev;
  if m.entry == Memory.no_entry then Memory.ensure_entry m;
  let e = m.entry in
  if held && e.held then
    invalid_argf "Rig.%s: %s.(%d)'s memory is in a hold" fn what i;
  if access = Read_write && e.access = Read then
    invalid_argf "Rig.%s: %s.(%d)'s memory admits only reads" fn what i;
  sub_slot s.c k e.stamps b.mem.handle

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, or that its queue has no room for, adds the others to [s]'s waits, and
   is their number. *)
let rec wait_points s n i count =
  if i = n then count
  else
    let p = sub_point s.c i in
    let producer = Dev.of_index (Point.index p) and v = Point.value p in
    if Dev.is_lost producer then begin
      Dev.check p;
      wait_points s n (i + 1) count
    end
    else if Dev.is_done p then wait_points s n (i + 1) count
    else
      let way =
        if count >= s.dev.max_waits then host_wait else pair s.dev producer
      in
      if way = host_wait then begin
        Dev.wait producer v;
        wait_points s n (i + 1) count
      end
      else begin
        let kind =
          match producer.completion with
          | Object _ -> rig_object
          | _ -> rig_word
        in
        sub_wait s.c producer.index way v kind;
        wait_points s n (i + 1) (count + 1)
      end

let rec hand_over s nwaits =
  let d = s.dev in
  let r = c_submit s.c in
  Dev.run_owed ();
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
  else Dev.raise_lost d

(* Names the run's [k]th buffer, a read below [s.nreads], else a write: the
   buffer. *)
let name s held reads writes k =
  let nr = s.nreads in
  if k < nr then begin
    let b = Array.unsafe_get reads k in
    name_one s held Read k k b;
    b
  end
  else begin
    let b = Array.unsafe_get writes (k - nr) in
    name_one s held Read_write (k - nr) k b;
    b
  end

(* Names the run's buffers from the [k]th on, then hands the work over. Each
   buffer is read once from its array and stays reachable in a frame until the
   hand-over returned: the C slots hold its stamps without a reference, and the
   caller's array may change meanwhile. A frame keeps four buffers, so the
   rooting costs a call per four. *)
let rec run s held reads writes waits k =
  let n = s.nreads + s.nwrites in
  if k >= n then hand_over s (wait_points s (sub_collect s.c waits) 0 0)
  else
    let b0 = name s held reads writes k in
    let b1 = if k + 1 < n then name s held reads writes (k + 1) else b0 in
    let b2 = if k + 2 < n then name s held reads writes (k + 2) else b0 in
    let b3 = if k + 3 < n then name s held reads writes (k + 3) else b0 in
    let p = run s held reads writes waits (k + 4) in
    ignore (Sys.opaque_identity b0);
    ignore (Sys.opaque_identity b1);
    ignore (Sys.opaque_identity b2);
    ignore (Sys.opaque_identity b3);
    p

(* A forked child never waits on a guard its parent's thread may hold: every
   submission made before the fork is on a device the child inherited, which is
   lost. *)
let submit s ~reads ~writes ~waits =
  if Dev.is_lost s.dev then Dev.raise_lost s.dev;
  check_counts s reads writes;
  sub_take s.c;
  match
    let held = Atomic.get Memory.any_held in
    check_parts s held;
    run s held reads writes waits 0
  with
  | p ->
      sub_clear s.c;
      sub_give s.c;
      p
  | exception e ->
      sub_clear s.c;
      sub_give s.c;
      raise e

let copy ~hold_stamps d queue ~src ~dst =
  let part = { queue; after = [||]; work = Copy { src; dst } } in
  submit
    (build (Stamps hold_stamps) ~reads:0 ~writes:0 d [| part |])
    ~reads:[||] ~writes:[||] ~waits:[||]
