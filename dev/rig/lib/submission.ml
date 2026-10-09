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

external sub_copy : c -> int -> nativeint * int * nativeint * int * int -> unit
  = "caml_rig_sub_copy"

external sub_copy_local : c -> int -> int -> unit = "caml_rig_sub_copy_local"
[@@noalloc]

external sub_fixed : c -> int -> int -> nativeint -> bool -> unit
  = "caml_rig_sub_fixed"

external ensure_record : int -> int -> unit = "caml_rig_ensure_record"

(* Runs *)

(* A run's state, C memory a custom block holds and frees once collected. *)
type run

external run_new : unit -> run = "caml_rig_run_new"
external run_take : run -> c -> int = "caml_rig_run_take" [@@noalloc]
external run_fit : run -> c -> unit = "caml_rig_run_fit"
external run_give : run -> unit = "caml_rig_run_give" [@@noalloc]

(* Takes [run] for a submit of [c], fitting it to [c] unless it served [c]
   last: whether no other submit held it. *)
let take run c =
  match run_take run c with
  | 0 -> false
  | 1 -> true
  | _ ->
      run_fit run c;
      true

external run_slot : run -> int -> int -> nativeint -> unit = "caml_rig_run_slot"
[@@noalloc]

external run_collect : c -> run -> int array -> int -> int
  = "caml_rig_run_collect"

external run_point : run -> int -> int = "caml_rig_run_point" [@@noalloc]

external run_wait : run -> int -> int -> int -> int -> unit
  = "caml_rig_run_wait"

external run_value : run -> int = "caml_rig_run_value" [@@noalloc]
external run_no_room_at : run -> int = "caml_rig_run_no_room_at" [@@noalloc]
external run_producer : run -> int = "caml_rig_run_producer" [@@noalloc]
external c_submit : c -> run -> int = "caml_rig_submit"

module Run = struct
  type t = run

  let make = run_new
end

type t = {
  dev : device;
  c : c;
  parts : part array;  (** Its buffers are checked live at each submit. *)
  nreads : int;  (** The buffers each run reads. *)
  nwrites : int;  (** The buffers each run writes. *)
  hold : hold option;
      (** The hold whose stamps each run raises, kept reachable: its release
          frees what the parts run. *)
}

let queue_index d fn q =
  let rec go i =
    if i = Array.length d.queues then
      invalid_argf "Rig.%s: %s has no queue %S" fn d.name q
    else if String.equal d.queues.(i).name q then i
    else go (i + 1)
  in
  go 0

let kind_of = function
  | Words _ -> Rig_edge.Words
  | Fill _ -> Fill
  | Copy _ -> Copy

let kind_name = function
  | Rig_edge.Words -> "words"
  | Fill -> "fills"
  | Copy -> "copies"

(* Refuses a part of a kind its queue does not run. *)
let check_runs d fn q work =
  let k = kind_of work in
  if not (List.mem k d.queues.(q).runs) then
    invalid_argf "Rig.%s: %s's queue %S runs no %s" fn d.name d.queues.(q).name
      (kind_name k)

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

(* Which side of a copy on [d] is this process's memory, [-1] for a copy [d]
   cannot run: on a device of another machine, one side may be. An int, so a
   local copy allocates nothing for it. *)
let copy_local d src dst =
  let far = Option.is_some d.machine in
  if src.mem.dev == d && dst.mem.dev == d then local_none
  else if far && local src && dst.mem.dev == d then local_src
  else if far && src.mem.dev == d && local dst then local_dst
  else -1

(* The handle a copy's side passes: the host address of the memory on the side
   [copy_local] names. *)
let copy_handle side k b =
  if side = k then Nativeint.of_int b.mem.host else b.mem.handle

(* Hands the C form [c] its [!k]th fixed buffer [b], and counts it. Only [d]'s
   own memory names a handle of its driver ([rig_edge.h]'s [handles]):
   another's, such as this process's memory a copy on another machine's device
   names, keeps its stamps alone. *)
let fix c k d b write =
  let handle = if b.mem.dev == d then b.mem.handle else 0n in
  sub_fixed c !k (entry_of b).stamps handle write;
  incr k

let build hold ~reads ~writes d parts =
  let fn = "Submission.make" in
  if reads < 0 || writes < 0 then invalid_argf "Rig.%s: a count is negative" fn;
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_host d || Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name;
  let check_buffer = Buffer.check_live fn in
  let nafter = ref 0 and nfixed = ref 0 in
  Array.iteri
    (fun i p ->
      Array.iter
        (fun j ->
          if j < 0 || j >= i then
            invalid_argf "Rig.%s: part %d's after names part %d" fn i j)
        p.after;
      nafter := !nafter + Array.length p.after;
      check_runs d fn (queue_index d fn p.queue) p.work;
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
          check_buffer src;
          check_buffer dst;
          if Buffer.length src <> Buffer.length dst then
            invalid_argf "Rig.%s: a copy's buffers differ in size" fn;
          if copy_local d src dst < 0 then
            invalid_argf "Rig.%s: a copy's buffers are not %s's memory" fn
              d.name;
          if Buffer.access dst = Read then
            invalid_argf "Rig.%s: a copy's dst admits only reads" fn;
          nfixed := !nfixed + 2)
    parts;
  let c = sub_new d.c (Array.length parts) !nafter !nfixed reads writes in
  let at = ref 0 and k = ref 0 in
  Array.iteri
    (fun i p ->
      sub_part c i (queue_index d fn p.queue) p.after !at;
      at := !at + Array.length p.after;
      match p.work with
      | Words w ->
          sub_words c i (host_address fn w) (Buffer.length w / 4);
          fix c k d w false
      | Fill f ->
          sub_fill c i f.fill (host_address fn f.arg) f.ring_units
            f.segment_bytes;
          fix c k d f.arg false
      | Copy { src; dst } ->
          let side = copy_local d src dst in
          sub_copy c i
            ( copy_handle side local_dst dst,
              dst.offset,
              copy_handle side local_src src,
              src.offset,
              Buffer.length src );
          if side <> local_none then sub_copy_local c i side;
          fix c k d src false;
          fix c k d dst true)
    parts;
  { dev = d; c; parts; nreads = reads; nwrites = writes; hold }

let make ?hold ~reads ~writes d parts = build hold ~reads ~writes d parts

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
    | Store -> d.waits.stores
    | Object _ -> d.waits.objects && d.key = p.key
    | Host_writes -> d.waits.hosts
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

let fn = "submit"

let check_part p =
  match p.work with
  | Words b -> Buffer.check_live fn b
  | Fill f -> Buffer.check_live fn f.arg
  | Copy { src; dst } ->
      Buffer.check_live fn src;
      Buffer.check_live fn dst

let check_parts s =
  for k = 0 to Array.length s.parts - 1 do
    check_part s.parts.(k)
  done

let counted n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")

let check_counts s reads writes =
  let nr = Array.length reads and nw = Array.length writes in
  if nr <> s.nreads || nw <> s.nwrites then
    invalid_argf "Rig.%s: %s and %s for a submission of %s and %s" fn
      (counted nr "read") (counted nw "write") (counted s.nreads "read")
      (counted s.nwrites "write")

(* Refuses [b], element [i] of the run's array of [access] ([reads] or
   [writes]), unless it is live, on [s]'s device and, written, of memory that
   admits writes; and hands its stamps and handle to the run's slot [k].
   Memory of another device that is lost raises its loss; [s]'s device's own
   loss is the hand-over's. *)
let name_one s run (access : access) i k b =
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
  if access = Read_write && e.access = Read then
    invalid_argf "Rig.%s: %s.(%d)'s memory admits only reads" fn what i;
  run_slot run k e.stamps b.mem.handle

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, or that its queue has no room for, adds the others to the run's waits
   after committing their producers' work, and is their number. *)
let rec wait_points s run n i count =
  if i = n then count
  else
    let p = run_point run i in
    let producer = Dev.of_index (Point.index p) and v = Point.value p in
    if Dev.is_lost producer then begin
      Dev.check p;
      wait_points s run n (i + 1) count
    end
    else if Dev.is_done p then wait_points s run n (i + 1) count
    else
      let way =
        if count >= s.dev.waits.most then host_wait else pair s.dev producer
      in
      if way = host_wait then begin
        Dev.wait producer v;
        wait_points s run n (i + 1) count
      end
      else begin
        Dev.commit producer v;
        let kind =
          match producer.completion with
          | Object _ -> rig_object
          | _ -> rig_word
        in
        run_wait run producer.index way v kind;
        wait_points s run n (i + 1) (count + 1)
      end

let rec hand_over s run nwaits =
  let d = s.dev in
  let r = c_submit s.c run in
  Dev.run_owed ();
  if r = Dev.Answer.ok then Point.make d.index (run_value run)
  else if r = Dev.Answer.busy then hand_over s run nwaits
  else if r = Dev.Answer.no_room then begin
    let at = run_no_room_at run in
    let w = Dev.word d in
    if w < at then Dev.wait d (w + 1);
    hand_over s run nwaits
  end
  else if r = Dev.Answer.need_record then begin
    ensure_record d.c nwaits;
    hand_over s run nwaits
  end
  else if r = Dev.Answer.never then
    invalid_argf
      "Rig.%s: the parts never fit %s's queues, or name work its driver does \
       not run"
      fn d.name
  else if r = Dev.Answer.producer_lost then
    Dev.raise_lost (Dev.of_index (run_producer run))
  else Dev.raise_lost d

(* Names the run's [k]th buffer, a read below [s.nreads], else a write: the
   buffer. *)
let name s run reads writes k =
  let nr = s.nreads in
  if k < nr then begin
    let b = Array.unsafe_get reads k in
    name_one s run Read k k b;
    b
  end
  else begin
    let b = Array.unsafe_get writes (k - nr) in
    name_one s run Read_write (k - nr) k b;
    b
  end

(* Names the run's buffers from the [k]th on, then hands the work over. Each
   buffer is read once from its array and stays reachable in a frame until the
   hand-over returned: the C slots hold its stamps without a reference, and the
   caller's array may change meanwhile. A frame keeps four buffers, so the
   rooting costs a call per four. *)
let rec go s run reads writes waits k =
  let n = s.nreads + s.nwrites in
  if k >= n then
    let hold = match s.hold with Some h -> h.hstamps | None -> 0 in
    hand_over s run (wait_points s run (run_collect s.c run waits hold) 0 0)
  else
    let b0 = name s run reads writes k in
    let b1 = if k + 1 < n then name s run reads writes (k + 1) else b0 in
    let b2 = if k + 2 < n then name s run reads writes (k + 2) else b0 in
    let b3 = if k + 3 < n then name s run reads writes (k + 3) else b0 in
    let p = go s run reads writes waits (k + 4) in
    ignore (Sys.opaque_identity b0);
    ignore (Sys.opaque_identity b1);
    ignore (Sys.opaque_identity b2);
    ignore (Sys.opaque_identity b3);
    p

(* Submits [s] with [run], which the caller took. [s] stays reachable until
   the run is given back: the C submit reads it with the runtime released. *)
let taken s run reads writes waits =
  match
    check_parts s;
    go s run reads writes waits 0
  with
  | p ->
      run_give run;
      ignore (Sys.opaque_identity s);
      p
  | exception e ->
      run_give run;
      raise e

let submit s ~run ~reads ~writes ~waits =
  if Dev.is_lost s.dev then Dev.raise_lost s.dev;
  check_counts s reads writes;
  if not (take run s.c) then
    invalid_argf "Rig.%s: another submit is using the run" fn;
  taken s run reads writes waits

(* Each domain's run for copies; one another thread of the domain holds is
   replaced by a fresh one for the copy. *)
let copy_runs = Domain.DLS.new_key run_new

let copy d queue ~src ~dst =
  let part = { queue; after = [||]; work = Copy { src; dst } } in
  let s = build None ~reads:0 ~writes:0 d [| part |] in
  let run = Domain.DLS.get copy_runs in
  let run =
    if take run s.c then run
    else
      let fresh = run_new () in
      ignore (take fresh s.c : bool);
      fresh
  in
  taken s run [||] [||] [||]
