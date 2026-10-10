(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type ref = { at : int; slot : int }

type work =
  | Words of buffer
  | Fill of {
      fill : nativeint;
      arg : buffer;
      ring_units : int;
      segment_bytes : int;
    }
  | Copy of { src : buffer; dst : buffer }
  | Launch of { image : image; kernel : string; params : int; refs : ref array }

type part = { queue : string; after : int array; work : work }

(* The C form *)

(* The C form, which a custom block holds and frees once collected. *)
type c

external sub_new : int -> int -> int -> int -> access array -> int -> c
  = "caml_rig_sub_new_byte" "caml_rig_sub_new"

external sub_launch :
  c -> int -> int -> nativeint -> int -> ref array -> int -> int
  = "caml_rig_sub_launch_byte" "caml_rig_sub_launch"

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
  | 2 ->
      run_fit run c;
      true
  | _ -> invalid_arg "Rig.submit: the run holds no block of the last launch"

external run_slot : run -> int -> int -> nativeint -> int -> unit
  = "caml_rig_run_slot"
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
external run_timed : run -> bool -> unit = "caml_rig_run_timed" [@@noalloc]
external run_pair : run -> int = "caml_rig_run_pair" [@@noalloc]
external times_read : int -> int -> int * int = "caml_rig_times_read"

type block = int

module Run = struct
  type t = run

  let make = run_new

  external groups :
    t ->
    (block[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"

  external threads :
    t ->
    (block[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"

  external shared : t -> (block[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_shared_byte" "caml_rig_run_shared"

  external int32 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int32_byte" "caml_rig_run_int32"

  external int64 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"

  external float32 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"

  external float64 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
end

type t = {
  dev : device;
  c : c;
  fixed : buffer array;
      (** The buffers [c] names, checked live at each submit and kept
          reachable: [c] holds their stamps without a reference. *)
  images : image list;
      (** The images of [c]'s launches, kept loaded: [c] names their
          templates, which live while the image is loaded. *)
  access : access array;  (** Each run buffer's access, by slot. *)
  hold : hold option;
      (** The hold whose stamps each run raises, kept reachable: its release
          frees what the parts run. *)
  blocks : int array;  (** Each part's block, [-1] for a part no launch. *)
  lane : string;
      (** The lane of a span of its work: its first part's queue, [""] for no
          part. *)
  span : string;
      (** The name of a span of its work: its launches' functions, or its first
          part's kind. *)
}

let no_block = -1

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
  | Launch _ -> Launch

let kind_name = function
  | Rig_edge.Words -> "words"
  | Fill -> "fills"
  | Copy -> "copies"
  | Launch -> "launches"

(* Refuses a part of the kind [k] on the queue [q] if [q] does not run it. *)
let check_runs d fn q k =
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

(* Hands the C form [c] its [k]th fixed buffer [b]. Only [d]'s own memory
   names a handle of its driver ([rig_edge.h]'s [handles]): another's, such as
   this process's memory a copy on another machine's device names, keeps its
   stamps alone. *)
let fix c k d b write =
  let handle = if b.mem.dev == d then b.mem.handle else 0n in
  sub_fixed c k (entry_of b).stamps handle write

(* Refuses a device that runs no submitted work. *)
let check_device fn d =
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_host d || Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name

(* Refuses a copy on [d] of [src] into [dst] unless both are live and of one
   size, [d] reaches both, and [dst] admits writes. *)
let check_copy fn d src dst =
  Buffer.check_live fn src;
  Buffer.check_live fn dst;
  if Buffer.length src <> Buffer.length dst then
    invalid_argf "Rig.%s: a copy's buffers differ in size" fn;
  if copy_local d src dst < 0 then
    invalid_argf "Rig.%s: a copy's buffers are not %s's memory" fn d.name;
  if Buffer.access dst = Read then
    invalid_argf "Rig.%s: a copy's dst admits only reads" fn

(* Makes part [i] of [c] a copy on [d] of [src] into [dst]. *)
let set_copy c d i src dst =
  let side = copy_local d src dst in
  sub_copy c i
    ( copy_handle side local_dst dst,
      dst.offset,
      copy_handle side local_src src,
      src.offset,
      Buffer.length src );
  if side <> local_none then sub_copy_local c i side

(* The most parameter bytes of a launch, [rig_edge.h]'s RIG_PARAMS. *)
let max_params = 4096

(* Refuses part [i]'s launch on [d] of a submission whose runs name [slots]
   buffers unless its image is [d]'s, its refs lie among its parameters and name
   a run buffer, and its image has its function: the entry. *)
let launch_entry fn d slots i image kernel params refs =
  if image.idev != d then
    invalid_argf "Rig.%s: part %d's image is not loaded on %s" fn i d.name;
  if params < 0 || params > max_params then
    invalid_argf "Rig.%s: part %d has %d parameter bytes, not 0 to %d" fn i
      params max_params;
  Array.iter
    (fun { at; slot } ->
      if at < 0 || at mod 8 <> 0 || at + 8 > params then
        invalid_argf
          "Rig.%s: part %d's ref at %d is not 8 aligned bytes of its %d" fn i at
          params;
      if slot < 0 || slot >= slots then
        invalid_argf "Rig.%s: part %d's ref names slot %d of %d" fn i slot slots)
    refs;
  let ats = Array.map (fun r -> r.at) refs in
  Array.sort Int.compare ats;
  for k = 1 to Array.length ats - 1 do
    if ats.(k) = ats.(k - 1) then
      invalid_argf "Rig.%s: part %d has two refs at %d" fn i ats.(k)
  done;
  match Memory.kernel_entry image kernel with
  | Some e -> e
  | None ->
      invalid_argf "Rig.%s: part %d's image has no function %S" fn i kernel

(* Refuses fixed memory that is dead, not [d]'s or, written, admits only
   reads. *)
let check_memory fn d (b, (access : access)) =
  Buffer.check_live fn b;
  if b.mem.dev != d then
    invalid_argf "Rig.%s: a fixed buffer is on %s, not on %s: borrow it" fn
      b.mem.dev.name d.name;
  if access = Read_write && Buffer.access b = Read then
    invalid_argf "Rig.%s: a fixed buffer written admits only reads" fn

(* [p] with arrays of its own. [make] copies the caller's parts and access
   once, then checks, sizes and compiles the copy alone: a change to the
   caller's arrays while it runs, from another domain or from a thread while a
   driver call blocks, changes nothing. *)
let own p =
  let work =
    match p.work with
    | Launch l -> Launch { l with refs = Array.copy l.refs }
    | w -> w
  in
  { p with after = Array.copy p.after; work }

let build hold ~fixed ~access d parts =
  let parts = Array.map own parts and access = Array.copy access in
  let fn = "Submission.make" in
  check_device fn d;
  let check_buffer = Buffer.check_live fn in
  let nafter = ref 0 and nfixed = ref 0 and nrefs = ref 0 in
  let entries = Array.make (Array.length parts) None in
  Array.iteri
    (fun i p ->
      Array.iter
        (fun j ->
          if j < 0 || j >= i then
            invalid_argf "Rig.%s: part %d's after names part %d" fn i j)
        p.after;
      nafter := !nafter + Array.length p.after;
      check_runs d fn (queue_index d fn p.queue) (kind_of p.work);
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
          check_copy fn d src dst;
          nfixed := !nfixed + 2
      | Launch l ->
          entries.(i) <-
            Some
              (launch_entry fn d (Array.length access) i l.image l.kernel
                 l.params l.refs);
          nrefs := !nrefs + Array.length l.refs)
    parts;
  let memory = Array.of_list fixed in
  Array.iter (check_memory fn d) memory;
  nfixed := !nfixed + Array.length memory;
  let c = sub_new d.c (Array.length parts) !nafter !nfixed access !nrefs in
  let blocks = Array.make (Array.length parts) no_block in
  let at = ref 0 and k = ref 0 and r = ref 0 in
  let fixed = ref [] and images = ref [] in
  let fix_next b write =
    fix c !k d b write;
    incr k;
    fixed := b :: !fixed
  in
  Array.iteri
    (fun i p ->
      sub_part c i (queue_index d fn p.queue) p.after !at;
      at := !at + Array.length p.after;
      match p.work with
      | Words w ->
          sub_words c i (host_address fn w) (Buffer.length w / 4);
          fix_next w false
      | Fill f ->
          sub_fill c i f.fill (host_address fn f.arg) f.ring_units
            f.segment_bytes;
          fix_next f.arg false
      | Copy { src; dst } ->
          set_copy c d i src dst;
          fix_next src false;
          fix_next dst true
      | Launch l ->
          let e = Option.get entries.(i) in
          blocks.(i) <- sub_launch c i e.code e.launch l.params l.refs !r;
          r := !r + Array.length l.refs;
          images := l.image :: !images)
    parts;
  Array.iter (fun (b, access) -> fix_next b (access = Read_write)) memory;
  let kernels =
    Array.to_list parts
    |> List.filter_map (fun p ->
           match p.work with Launch l -> Some l.kernel | _ -> None)
  in
  let lane, span =
    if Array.length parts = 0 then ("", "")
    else
      ( parts.(0).queue,
        if kernels = [] then kind_name (kind_of parts.(0).work)
        else String.concat ", " kernels )
  in
  {
    dev = d;
    c;
    fixed = Array.of_list !fixed;
    images = !images;
    access;
    hold;
    blocks;
    lane;
    span;
  }

let make ?hold ?(fixed = []) ?(access = [||]) d parts =
  build hold ~fixed ~access d parts

let block s i =
  if i < 0 || i >= Array.length s.blocks || s.blocks.(i) = no_block then
    invalid_argf "Rig.Submission.block: part %d is no launch" i;
  s.blocks.(i)

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

let check_fixed s =
  for k = 0 to Array.length s.fixed - 1 do
    Buffer.check_live fn (Array.unsafe_get s.fixed k)
  done

let counted n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")

let check_count s buffers =
  let n = Array.length buffers and slots = Array.length s.access in
  if n <> slots then
    invalid_argf "Rig.%s: %s for a submission of %s" fn (counted n "buffer")
      (counted slots "buffer")

(* Refuses [b], the run's buffer [k], used with [access], unless it is live, on
   [s]'s device and, written, of memory that admits writes; and hands its
   stamps and handle to the run's slot [k]. Memory of another device that is
   lost raises its loss; [s]'s device's own loss is the hand-over's. *)
let name_one s run (access : access) k b =
  if not (Buffer.is_live b) then
    invalid_argf "Rig.%s: buffers.(%d) is dead: %s" fn k b.mem.claim.why;
  (* A queue reaches other memory only through a borrow on its device. *)
  if b.mem.dev != s.dev then
    invalid_argf "Rig.%s: buffers.(%d) is on %s, not on %s: borrow it" fn k
      b.mem.dev.name s.dev.name;
  let m = b.mem.root in
  if m != b.mem && m.dev != s.dev && Dev.is_lost m.dev then Dev.raise_lost m.dev;
  if m.entry == Memory.no_entry then Memory.ensure_entry m;
  let e = m.entry in
  if access = Read_write && e.access = Read then
    invalid_argf
      "Rig.%s: buffers.(%d) is written and its memory admits only reads" fn k;
  (* -1 for memory with no address, which the collect refuses a ref to. *)
  let address = if b.mem.address < 0 then -1 else b.mem.address + b.offset in
  run_slot run k e.stamps b.mem.handle address

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
      "Rig.%s: the parts never fit %s's queues, name work its driver does not \
       run, or hold a launch whose block it refuses"
      fn d.name
  else if r = Dev.Answer.producer_lost then
    Dev.raise_lost (Dev.of_index (run_producer run))
  else Dev.raise_lost d

(* Names the run's [k]th buffer with its access: the buffer. *)
let name s run buffers k =
  let b = Array.unsafe_get buffers k in
  name_one s run (Array.unsafe_get s.access k) k b;
  b

(* Names the run's buffers from the [k]th on, then hands the work over. Each
   buffer is read once from its array and stays reachable in a frame until the
   hand-over returned: the C slots hold its stamps without a reference, and the
   caller's array may change meanwhile. A frame keeps four buffers, so the
   rooting costs a call per four. *)
let rec go s run buffers waits k =
  let n = Array.length s.access in
  if k >= n then
    let hold = match s.hold with Some h -> h.hstamps | None -> 0 in
    hand_over s run (wait_points s run (run_collect s.c run waits hold) 0 0)
  else
    let b0 = name s run buffers k in
    let b1 = if k + 1 < n then name s run buffers (k + 1) else b0 in
    let b2 = if k + 2 < n then name s run buffers (k + 2) else b0 in
    let b3 = if k + 3 < n then name s run buffers (k + 3) else b0 in
    let p = go s run buffers waits (k + 4) in
    ignore (Sys.opaque_identity b0);
    ignore (Sys.opaque_identity b1);
    ignore (Sys.opaque_identity b2);
    ignore (Sys.opaque_identity b3);
    p

(* Records, in the profiles [ps], the span of [s]'s value [p] once [p] is
   reached, from the time pair [pair] its driver wrote: none if the driver
   left it 0. The function holds only [s]'s names: a lost device keeps its
   functions for good. *)
let record s ps p pair =
  let device = s.dev and lane = s.lane and name = s.span in
  Dev.after device (Point.value p) (fun () ->
      match times_read device.c pair with
      | 0, 0 -> ()
      | start, stop ->
          Prof.add_all ps [ Span { device; lane; name; start; stop } ])

(* Submits [s] with [run], which the caller took. The C submit reads [s]'s C
   form, and the stamps and templates it names without a reference, with the
   runtime released: their owners stay reachable until the run is given
   back. While a profile is taken, a submission with parts times its value. *)
let taken s run buffers waits =
  let ps = if s.lane = "" then [] else Prof.active () in
  run_timed run (ps <> []);
  match
    check_fixed s;
    go s run buffers waits 0
  with
  | p ->
      let pair = run_pair run in
      run_give run;
      if pair >= 0 then record s ps p pair;
      ignore (Sys.opaque_identity s.c);
      ignore (Sys.opaque_identity s.fixed);
      ignore (Sys.opaque_identity s.images);
      ignore (Sys.opaque_identity s.hold);
      p
  | exception e ->
      run_give run;
      raise e

let submit s ~run ~buffers ~waits =
  if Dev.is_lost s.dev then Dev.raise_lost s.dev;
  check_count s buffers;
  if not (take run s.c) then
    invalid_argf "Rig.%s: another submit is using the run" fn;
  taken s run buffers waits

(* Each domain's run for copies; one another thread of the domain holds is
   replaced by a fresh one for the copy. *)
let copy_runs = Domain.DLS.new_key run_new

(* The submission [build] makes of the one part
   [{ queue; after = [||]; work = Copy { src; dst } }], compiled directly: each
   [Buffer.copy] through a queue makes one, and it has no after, launch or fixed
   memory for [build]'s copies, loops and blocks to handle. *)
let copy d queue ~src ~dst =
  let fn = "Submission.make" in
  check_device fn d;
  let q = queue_index d fn queue in
  check_runs d fn q Rig_edge.Copy;
  check_copy fn d src dst;
  let c = sub_new d.c 1 0 2 [||] 0 in
  sub_part c 0 q [||] 0;
  set_copy c d 0 src dst;
  fix c 0 d src false;
  fix c 1 d dst true;
  let s =
    {
      dev = d;
      c;
      fixed = [| src; dst |];
      images = [];
      access = [||];
      hold = None;
      blocks = [||];
      lane = queue;
      span = kind_name Copy;
    }
  in
  let run = Domain.DLS.get copy_runs in
  let run =
    if take run s.c then run
    else
      let fresh = run_new () in
      ignore (take fresh s.c : bool);
      fresh
  in
  taken s run [||] [||]
