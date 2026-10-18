(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* How work uses a slot or fixed memory, in [rig_stubs.h]'s [enum rig_use]
   order: C reads a constructor's index. *)
type use = Use_read | Use_write | Use_unordered

(* A buffer ref. C reads [at] and [slot], the first two fields. *)
type slot_ref = { at : int; slot : int; use : use }

type ref =
  | Slot_ref of slot_ref
  | Entry of { at : int; fn : func }

(* Each constructor refuses what its arguments show, naming itself; [make]
   refuses what needs the device or several parts. *)

(* Refuses a ref's [at] that is negative or no multiple of 8. *)
let check_at fn at =
  if at < 0 || at mod 8 <> 0 then
    invalid_argf "Rig.Submission.%s: a ref at %d, not a multiple of 8 from 0"
      fn at

let slot_ref fn use slot ~at =
  if slot < 0 then invalid_argf "Rig.Submission.%s: slot %d" fn slot;
  check_at fn at;
  Slot_ref { at; slot; use }

let read slot ~at = slot_ref "read" Use_read slot ~at
let write slot ~at = slot_ref "write" Use_write slot ~at
let unordered slot ~at = slot_ref "unordered" Use_unordered slot ~at

let entry f ~at =
  check_at "entry" at;
  Entry { at; fn = f }

let ref_at = function Slot_ref { at; _ } | Entry { at; _ } -> at

type side = Slot of int | Buffer of buffer

type part =
  | Words of buffer
  | Fill of { fill : nativeint; params : int; refs : ref array }
  | Copy of { src : side; dst : side }
  | Set of { dst : side; byte : char }
  | Launch of { fn : func; params : int; refs : ref array }

(* The most parameter bytes of a launch or a fill: a block counts 16 bytes
   plus its parameters rounded up to 16 against the [2^31 - 1] bytes a run
   holds. *)
let most_params = (1 lsl 31) - 32

(* Refuses a part's parameters outside 0 to [most_params], and refs that share
   an [at] or whose 8 bytes end past the parameters. A ref's end is checked as
   [at > params - 8], since [at + 8] wraps for an [at] near [max_int]. *)
let check_args fn params refs =
  if params < 0 || params > most_params then
    invalid_argf "Rig.Submission.%s: %d parameter bytes, outside 0 to %d" fn
      params most_params;
  Array.iter
    (fun r ->
      let at = ref_at r in
      if at > params - 8 then
        invalid_argf
          "Rig.Submission.%s: a ref at %d past its %d parameter bytes" fn at
          params)
    refs;
  let ats = Array.map ref_at refs in
  Array.sort Int.compare ats;
  for k = 1 to Array.length ats - 1 do
    if ats.(k) = ats.(k - 1) then
      invalid_argf "Rig.Submission.%s: two refs at %d" fn ats.(k)
  done

(* The caller's array is copied: a part is fixed once made. *)
let fill ?(refs = [||]) fill ~params =
  let refs = Array.copy refs in
  check_args "fill" params refs;
  Fill { fill; params; refs }

(* Refuses fewer parameters than [f] reads, or a ref past them, where the
   driver states them. *)
let check_reads (f : func) params refs =
  match f.fentry.params with
  | None -> ()
  | Some takes ->
      if params < takes then
        invalid_argf
          "Rig.Submission.launch: %s takes %d bytes of parameters; the launch \
           gives %d"
          f.fname takes params;
      Array.iter
        (fun r ->
          let at = ref_at r in
          if at > takes - 8 then
            invalid_argf
              "Rig.Submission.launch: a ref at %d lies past the %d bytes of \
               parameters %s takes"
              at takes f.fname)
        refs

let launch ?(refs = [||]) f ~params =
  let refs = Array.copy refs in
  check_args "launch" params refs;
  check_reads f params refs;
  Launch { fn = f; params; refs }

(* Refuses a copy whose ends are one slot or a negative one, whose buffer
   end is dead or whose [dst] buffer admits only reads, or whose two buffer
   ends differ in length or share a byte. *)
let copy ~src ~dst =
  let fn = "Submission.copy" in
  let check = function
    | Slot k when k < 0 -> invalid_argf "Rig.%s: slot %d" fn k
    | Slot _ -> ()
    | Buffer b -> Buffer.check_live fn b
  in
  check src;
  check dst;
  (match (src, dst) with
  | Slot k, Slot k' when k = k' ->
      invalid_argf "Rig.%s: slot %d into itself" fn k
  | Buffer a, Buffer b ->
      if Buffer.length a <> Buffer.length b then
        invalid_argf "Rig.%s: buffers of %d and %d bytes" fn (Buffer.length a)
          (Buffer.length b);
      if Buffer.overlaps a b then invalid_argf "Rig.%s: the buffers overlap" fn
  | _ -> ());
  (match dst with
  | Buffer b when Buffer.access b = Def.Read ->
      invalid_argf "Rig.%s: into a buffer that admits only reads" fn
  | Slot _ | Buffer _ -> ());
  Copy { src; dst }

(* Internal: rig's own fills, whose [dst] is checked where they are made. *)
let set ~dst byte = Set { dst; byte }

let words b =
  let fn = "Submission.words" in
  Buffer.check_live fn b;
  if b.mem.host < 0 then
    invalid_argf "Rig.%s: the buffer is not host memory" fn;
  if Buffer.length b mod 4 <> 0 then
    invalid_argf "Rig.%s: words of %d bytes, not a multiple of 4" fn
      (Buffer.length b);
  Words b

let use_of : Def.access -> use = function
  | Read -> Use_read
  | Read_write -> Use_write

(* The C form *)

(* The C form, which a custom block holds and frees once collected. *)
type c

external sub_new :
  Def.c -> int -> int -> int -> use array -> int -> int -> int -> c
  = "caml_rig_sub_new_byte" "caml_rig_sub_new"

external sub_launch :
  c -> int -> int -> nativeint -> int -> slot_ref array -> int -> int
  = "caml_rig_sub_launch_byte" "caml_rig_sub_launch"

external sub_part : c -> int -> int -> int array -> int -> unit
  = "caml_rig_sub_part"

external sub_words : c -> int -> int -> int -> unit = "caml_rig_sub_words"

external sub_fill :
  c -> int -> nativeint -> int -> slot_ref array -> int -> int
  = "caml_rig_sub_fill_byte" "caml_rig_sub_fill"

external sub_code : c -> int -> int -> int -> int -> unit
  = "caml_rig_sub_code"

external sub_copy : c -> int -> int -> int -> int -> unit = "caml_rig_sub_copy"

external sub_set : c -> int -> int -> int -> char -> unit = "caml_rig_sub_set"

external sub_side : c -> int -> bool -> buffer -> bool -> bool -> unit
  = "caml_rig_sub_side_byte" "caml_rig_sub_side"

external sub_seal : c -> unit = "caml_rig_sub_seal"

external reraise : exn -> 'a = "%reraise"

external sub_id : c -> int = "caml_rig_sub_id" [@@noalloc]

external sub_fixed : c -> int -> int -> use -> unit
  = "caml_rig_sub_fixed"

(* Calls *)

(* The state of one submit, C memory that one submit at a time takes from
   every call made, as the pointer [submit_begin] answers. *)
type call = int

external call_slot : call -> int -> int -> int -> unit = "caml_rig_call_slot"
[@@noalloc]

external call_side : call -> int -> buffer -> bool -> bool -> unit
  = "caml_rig_call_side"
[@@noalloc]

external call_point : call -> int -> int = "caml_rig_call_point" [@@noalloc]

external call_wait : call -> int -> int -> int -> unit = "caml_rig_call_wait"
external call_check : call -> unit = "caml_rig_call_check" [@@noalloc]
external call_value : call -> int = "caml_rig_call_value" [@@noalloc]

external call_no_room_at : call -> int = "caml_rig_call_no_room_at"
[@@noalloc]

external call_producer : call -> int = "caml_rig_call_producer" [@@noalloc]
external call_timed : call -> bool -> unit = "caml_rig_call_timed" [@@noalloc]
external call_pair : call -> int = "caml_rig_call_pair" [@@noalloc]
external times_read : Def.c -> int -> int * int = "caml_rig_times_read"
external times_free : Def.c -> int -> unit = "caml_rig_times_free" [@@noalloc]

external times_owe : Def.c -> int -> int -> unit = "caml_rig_times_owe"
[@@noalloc]

external times_due : Def.c -> int = "caml_rig_times_due" [@@noalloc]
external times_give : Def.c -> int -> unit = "caml_rig_times_give" [@@noalloc]

(* A launch's or a fill's block, as the setters' C reads it: the id of its
   submission's C form, where it starts in a run, its parameter bytes, and the
   most threads a group of its function has, [0] for a fill or where the
   driver states none, which [Run.threads] checks, with the function's name
   for its refusal. *)
type block = int * int * int * int * string

(* What names a run's slot: refs, copies or both. A slot only copies name may
   be this process's memory on a device of another machine, whose driver
   carries its bytes. *)
type naming = Refs | Copies | Both

type t = {
  dev : device;
  c : c;
  fixed : buffer array;
      (** The buffers [c] names, held ([held]) and kept reachable: [c] holds
          their stamps without a reference. *)
  addressable : int;
      (** The first [addressable] buffers of [fixed] are the caller's fixed
          memory; the others belong to parts. *)
  images : image list;
      (** The images of [c]'s launches, kept loaded: [c] names their templates,
          which live while the image is loaded. *)
  uses : use array;  (** Each slot's use. *)
  naming : naming array;  (** What names each slot. *)
  kept : token option;
      (** The token of the value the submission keeps, kept reachable: its
          collection lets the value go once the work is done. *)
  held : Claim.hold;
      (** The hold of [fixed], which every run reads, released once the
          submission is collected. *)
  blocks : block array;
      (** Each part's block, [no_block] for no launch or fill. *)
  span : (string * string) option;
      (** The first part's queue and the launches' functions, or the first
          part's kind. [None] for no parts. *)
  copies : (side * side) array;
      (** The sides of the copy parts that read memory, in order. *)
}

let no_block = (0, -1, 0, 0, "")

let kind_of = function
  | Words _ -> Rig_edge.Words
  | Fill _ -> Fill
  | Copy _ | Set _ -> Copy
  | Launch _ -> Launch

let kind_name = function
  | Rig_edge.Words -> "words"
  | Fill -> "fills"
  | Copy -> "copies"
  | Launch -> "launches"

(* The record of [b]'s memory, which holds the stamps a submission raises. *)
let[@inline] entry_of b =
  let a = b.mem.alloc in
  if a.entry == Memory.no_entry then Memory.ensure_entry a;
  a.entry

let host_address fn b =
  if b.mem.host < 0 then invalid_argf "Rig.%s: the buffer is not host memory" fn;
  b.mem.host + b.offset

(* Memory this process's host addresses, which a copy on a device of another
   machine names by its host address ([rig_edge.h]'s [copy_local]). *)
let local b = Dev.same_machine b.mem.dev Dev.host && b.mem.host >= 0

(* Whether a copy on [d] takes [b] as this process's memory, which the driver of
   [d], a device of another machine, carries. *)
let carried d b = b.mem.dev != d && local b && not (Dev.same_machine d Dev.host)

(* Only the original allocation says whose memory it is and of which kind:
   a peer borrow's device and handle describe the receiving device's mapping. *)
let owns_device_memory d b =
  let a = b.mem.alloc in
  a.owner == d
  && match a.entry.backing with
     | Driver_memory { memory = Device; _ } -> true
     | Driver_memory _ | Io_memory _ | Kept _ -> false

(* Hands the C form [c] its [k]th fixed buffer's stamps and access. *)
let fix c k b use = sub_fixed c k (entry_of b).stamps use

(* Refuses part [i]'s copy end on [d] unless its buffer is live and [d]'s,
   or carried. A slot end's buffer only a run gives, which the submit
   checks. *)
let check_end fn d i = function
  | Slot _ -> ()
  | Buffer b ->
      Buffer.check_live fn b;
      if b.mem.dev != d && not (carried d b) then
        invalid_argf "Rig.%s: part %d copies a buffer on %s, not on %s: \
                      borrow it"
          fn i b.mem.dev.name d.name

(* Refuses part [i]'s copy on [d] unless each end passes [check_end], and
   two buffer ends are not both carried. *)
let check_ends fn d i src dst =
  check_end fn d i src;
  check_end fn d i dst;
  match (src, dst) with
  | Buffer a, Buffer b when carried d a && carried d b ->
      invalid_argf "Rig.%s: neither of part %d's buffers is %s's memory" fn i
        d.name
  | _ -> ()

(* Refuses a device that runs no submitted work. *)
let check_device fn d =
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name

(* Refuses part [i]'s launch of [f] on [d] unless [f]'s image is [d]'s. *)
let check_launch fn d i (f : func) =
  if f.fimage.idev != d then
    invalid_argf "Rig.%s: part %d's image is not loaded on %s" fn i d.name

(* Refuses fixed memory that is dead, not [d]'s or, written, admits only
   reads. *)
let check_memory fn d (b, (access : Def.access)) =
  Buffer.check_live fn b;
  if b.mem.dev != d then
    invalid_argf "Rig.%s: a fixed buffer is on %s, not on %s: borrow it" fn
      b.mem.dev.name d.name;
  if access = Read_write && Buffer.access b = Def.Read then
    invalid_argf "Rig.%s: a fixed buffer written admits only reads" fn

(* The value of part [i]'s entry ref to [f] on [d]. *)
let entry_code fn d i (f : func) =
  if f.fimage.idev != d then
    invalid_argf "Rig.%s: part %d's entry ref names an image not loaded on %s"
      fn i d.name;
  f.fentry.code

let refs_of = function
  | Fill { refs; _ } | Launch { refs; _ } -> refs
  | Words _ | Copy _ | Set _ -> [||]

(* The buffer refs of [refs], which C places. *)
let slot_refs refs =
  Array.to_seq refs
  |> Seq.filter_map (function Slot_ref r -> Some r | Entry _ -> None)
  |> Array.of_seq

(* Slots *)

(* The slots part [i] names, how its work uses each, and by what: a copy reads
   its [src] and writes its [dst]. *)
let slots_of = function
  | Copy { src; dst } ->
      let named use = function
        | Slot k -> [ (k, use, Copies) ]
        | Buffer _ -> []
      in
      named Use_read src @ named Use_write dst
  | Set { dst = Slot k; _ } -> [ (k, Use_write, Copies) ]
  | Set { dst = Buffer _; _ } -> []
  | p ->
      Array.to_list
        (Array.map (fun r -> (r.slot, r.use, Refs)) (slot_refs (refs_of p)))

(* Each slot's use and what names it, from the refs and copies that name it:
   [Use_write] if one writes it. The slots are 0 to the highest one named,
   each named, and none used both unordered and ordered: a gap or a mix is a
   mistaken index. *)
let slot_uses fn parts =
  let named = Array.map slots_of parts in
  let n =
    Array.fold_left
      (List.fold_left (fun n (k, _, _) -> max n (k + 1)))
      0 named
  in
  let uses = Array.make n None and naming = Array.make n None in
  let name (k, use, by) =
    uses.(k) <-
      (match (uses.(k), use) with
      | None, u -> Some u
      | Some Use_unordered, Use_unordered -> Some Use_unordered
      | Some Use_unordered, _ | Some _, Use_unordered ->
          invalid_argf "Rig.%s: slot %d is used both unordered and ordered" fn
            k
      | Some Use_write, _ | _, Use_write -> Some Use_write
      | Some Use_read, Use_read -> Some Use_read);
    naming.(k) <-
      (match naming.(k) with
      | None -> Some by
      | Some prev when prev = by -> Some by
      | Some _ -> Some Both)
  in
  Array.iter (List.iter name) named;
  let get k = function
    | Some x -> x
    | None -> invalid_argf "Rig.%s: no ref or copy names slot %d" fn k
  in
  (Array.mapi get uses, Array.mapi get naming)

(* Placement and order *)

(* What a part reaches, and whether it writes it: a slot, whose buffer only a
   run knows ([bytes = None]), or the bytes of a buffer it names, as
   [(space, first, past)] at each of its places ({!Buffer.places}). *)
type reach = { bytes : (int * int * int) list option; writes : bool }

let named writes b =
  let n = Buffer.length b in
  let bytes = List.map (fun (s, a) -> (s, a, a + n)) (Buffer.places b) in
  { bytes = Some bytes; writes }

let slot writes = { bytes = None; writes }

let reaches = function
  | Words w -> [ named false w ]
  | Copy { src; dst } ->
      let side writes = function
        | Slot _ -> slot writes
        | Buffer b -> named writes b
      in
      [ side false src; side true dst ]
  | Set { dst = Slot _; _ } -> [ slot true ]
  | Set { dst = Buffer b; _ } -> [ named true b ]
  | Fill { refs; _ } | Launch { refs; _ } ->
      Array.fold_right
        (fun r acc ->
          match r with
          | Entry _ | Slot_ref { use = Use_unordered; _ } -> acc
          | Slot_ref { use = Use_read; _ } -> slot false :: acc
          | Slot_ref { use = Use_write; _ } -> slot true :: acc)
        refs []

(* The parts of one queue that reached bytes of one space: a segment tree over
   [cuts], the sorted bytes where a reached range of the space starts or ends.
   Node [k] spans cuts [lo] to [hi], its children [2k] and [2k + 1] the
   halves, the root [1] all. [all.(k)] is the last part that reached every
   byte of node [k], [some.(k)] the last that reached one. Parts reach in
   order, so the last part marked is the latest. *)
type ranges = { all : int array; some : int array }

let rec mark t k lo hi a b i =
  if b <= lo || hi <= a then ()
  else if a <= lo && hi <= b then begin
    t.all.(k) <- i;
    t.some.(k) <- i
  end
  else begin
    let mid = (lo + hi) / 2 in
    mark t (2 * k) lo mid a b i;
    mark t ((2 * k) + 1) mid hi a b i;
    t.some.(k) <- i
  end

(* The last part that reached a byte of cuts [a] to [b], or [-1]. *)
let rec latest t k lo hi a b =
  if b <= lo || hi <= a then -1
  else if a <= lo && hi <= b then t.some.(k)
  else
    let mid = (lo + hi) / 2 in
    let l = latest t (2 * k) lo mid a b
    and r = latest t ((2 * k) + 1) mid hi a b in
    Int.max t.all.(k) (Int.max l r)

(* A space's cuts and, per queue [q], the ranges its parts wrote ([2q]) and
   used at all ([2q + 1]), made when a part first reaches the space. *)
type space = { cuts : int array; ranges : ranges option array }

(* The index of byte [x] in [cuts], which holds it. *)
let cut cuts x =
  let rec go lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if cuts.(mid) < x then go (mid + 1) hi else go lo mid
  in
  go 0 (Array.length cuts)

(* The last part of each queue that reached memory, by how: per queue, the
   last that wrote anything, used anything, wrote a slot and used a slot, and
   per space, the last that wrote or used each byte. A slot may be any buffer,
   so it meets every use. *)
type order = {
  wrote : int array;
  used : int array;
  wrote_slot : int array;
  used_slot : int array;
  spaces : (int, space) Hashtbl.t;
}

let order_of queues reach fixed =
  let cuts = Hashtbl.create 8 in
  let add r =
    Option.iter
      (List.iter (fun (s, a, b) ->
           let l = Option.value (Hashtbl.find_opt cuts s) ~default:[] in
           Hashtbl.replace cuts s (a :: b :: l)))
      r.bytes
  in
  Array.iter (List.iter add) reach;
  List.iter add fixed;
  let spaces = Hashtbl.create (Hashtbl.length cuts) in
  Hashtbl.iter
    (fun s l ->
      Hashtbl.replace spaces s
        {
          cuts = Array.of_list (List.sort_uniq Int.compare l);
          ranges = Array.make (2 * queues) None;
        })
    cuts;
  let none () = Array.make queues (-1) in
  {
    wrote = none ();
    used = none ();
    wrote_slot = none ();
    used_slot = none ();
    spaces;
  }

(* Raises [last.(q)] to the last part of queue [q] whose use may share memory
   with [r] while one of them writes. *)
let follow o last r =
  for q = 0 to Array.length last - 1 do
    let i =
      match r.bytes with
      | None -> if r.writes then o.used.(q) else o.wrote.(q)
      | Some places ->
          List.fold_left
            (fun i (s, a, b) ->
              let sp = Hashtbl.find o.spaces s in
              match sp.ranges.((2 * q) + Bool.to_int r.writes) with
              | None -> i
              | Some t ->
                  let m = Array.length sp.cuts - 1 in
                  Int.max i (latest t 1 0 m (cut sp.cuts a) (cut sp.cuts b)))
            (if r.writes then o.used_slot.(q) else o.wrote_slot.(q))
            places
    in
    last.(q) <- Int.max last.(q) i
  done

(* Records that part [i] on queue [q] reached [r]. *)
let reached o q i r =
  o.used.(q) <- i;
  if r.writes then o.wrote.(q) <- i;
  match r.bytes with
  | None ->
      o.used_slot.(q) <- i;
      if r.writes then o.wrote_slot.(q) <- i
  | Some places ->
      List.iter
        (fun (s, a, b) ->
          let sp = Hashtbl.find o.spaces s in
          let m = Array.length sp.cuts - 1 in
          let a = cut sp.cuts a and b = cut sp.cuts b in
          let range k =
            let t =
              match sp.ranges.(k) with
              | Some t -> t
              | None ->
                  let n = 4 * Array.length sp.cuts in
                  let t =
                    { all = Array.make n (-1); some = Array.make n (-1) }
                  in
                  sp.ranges.(k) <- Some t;
                  t
            in
            mark t 1 0 m a b i
          in
          range ((2 * q) + 1);
          if r.writes then range (2 * q))
        places

let first_runs fn d k =
  let rec go i =
    if i = Array.length d.live.queues then
      invalid_argf "Rig.%s: %s runs no %s" fn d.name (kind_name k)
    else if List.mem k d.live.queues.(i).runs then i
    else go (i + 1)
  in
  go 0

(* A part's queue, given [last], the last earlier part of each queue it
   follows: the first that runs its kind, except for words and copies. Words
   go on the first queue that runs launches and words, so the launches they
   prepare follow them, else on the first that runs words. A copy goes on the
   queue of the last earlier part it follows, where that queue runs copies, so
   dependent work stays on one queue; otherwise on the copy engine. *)
let place fn d last = function
  | Words _ -> (
      let prepares (q : Rig_edge.queue) =
        List.mem Rig_edge.Launch q.runs && List.mem Rig_edge.Words q.runs
      in
      match Array.find_index prepares d.live.queues with
      | Some q -> q
      | None -> first_runs fn d Rig_edge.Words)
  | Copy _ | Set _ -> (
      let q = ref (-1) in
      Array.iteri
        (fun q' i -> if i >= 0 && (!q < 0 || i > last.(!q)) then q := q')
        last;
      match !q with
      | q
        when q >= 0 && List.mem Rig_edge.Copy d.live.queues.(q).runs ->
          q
      | _ -> (
          match d.live.copy_engine with
          | Some q -> q
          | None -> invalid_argf "Rig.%s: %s runs no copies" fn d.name))
  | p -> first_runs fn d (kind_of p)

(* The parts a part on queue [q] follows on other queues, in order: per queue,
   the last earlier one it follows, since a queue runs its parts in order. *)
let after q last =
  let found = ref [] in
  Array.iteri (fun q' i -> if q' <> q && i >= 0 then found := i :: !found) last;
  Array.of_list (List.sort Int.compare !found)

(* Each part's queue and the parts it follows on other queues. Part [j] follows
   part [i] where they may share memory while one of them writes: every part
   also uses the [fixed] memory. Words, whose effect rig cannot see, also
   follow every earlier part; later parts follow them by memory, or by running
   on their queue. One pass keeps, per queue, the last part that reached each
   memory and the last part of all, so the cost grows with the uses, not their
   pairs. *)
let order fn d reach fixed parts =
  let n = Array.length parts in
  let queues = Array.make n 0 and afters = Array.make n [||] in
  let nq = Array.length d.live.queues in
  let last = Array.make nq (-1) and latest = Array.make nq (-1) in
  let o = order_of nq reach fixed in
  for j = 0 to n - 1 do
    Array.fill last 0 nq (-1);
    List.iter (follow o last) reach.(j);
    List.iter (follow o last) fixed;
    (match parts.(j) with
    | Words _ -> Array.blit latest 0 last 0 nq
    | Fill _ | Copy _ | Set _ | Launch _ -> ());
    let q = place fn d last parts.(j) in
    queues.(j) <- q;
    afters.(j) <- after q last;
    latest.(q) <- j;
    List.iter (reached o q j) reach.(j);
    List.iter (reached o q j) fixed
  done;
  (queues, afters)

(* A token that keeps [v] reachable until the submission is collected and its
   work done: its record lands on [d]'s release list, and [d]'s drain defers
   it after the work submitted to [d] by then, which covers every submit of the
   submission. The token keeps [d]'s slot until then. *)
let keep_token d v =
  let t = Memory.token d.live.release (Release (Held v)) 0 max_int (-1) in
  if t == Memory.no_token then Dev.raise_lost d;
  t

(* The buffers every run reads: the fixed memory, the words and the copies'
   buffer sides. *)
let named_buffers fixed parts =
  Array.fold_right
    (fun p bs ->
      match p with
      | Words w -> w :: bs
      | Copy { src; dst } ->
          let side s bs = match s with Buffer b -> b :: bs | Slot _ -> bs in
          side src (side dst bs)
      | Set { dst = Buffer b; _ } -> b :: bs
      | Set { dst = Slot _; _ } | Fill _ | Launch _ -> bs)
    parts (List.map fst fixed)

let build keep ~fixed ~held d parts =
  let fn = "Submission.make" in
  check_device fn d;
  let kept = Option.map (keep_token d) keep in
  let uses, naming = slot_uses fn parts in
  let check_buffer = Buffer.check_live fn in
  let nfixed = ref 0 and nrefs = ref 0 and ncodes = ref 0 in
  let ncopies = ref 0 and nmoves = ref 0 in
  let slots = Array.map (fun p -> slot_refs (refs_of p)) parts in
  (* Each part's entry refs: where they lie in its parameters, their value
     and their image. *)
  let codes =
    Array.mapi
      (fun i p ->
        Array.to_list (refs_of p)
        |> List.filter_map (function
          | Slot_ref _ -> None
          | Entry { at; fn = f } -> Some (at, entry_code fn d i f, f.fimage)))
      parts
  in
  Array.iteri
    (fun i p ->
      match p with
      | Words w ->
          check_buffer w;
          incr nfixed
      | Fill _ ->
          nrefs := !nrefs + Array.length slots.(i);
          ncodes := !ncodes + List.length codes.(i)
      | Copy { src; dst } ->
          check_ends fn d i src dst;
          let buffer = function Buffer _ -> 1 | Slot _ -> 0 in
          nfixed := !nfixed + buffer src + buffer dst;
          incr ncopies;
          incr nmoves
      | Set { dst; _ } ->
          check_end fn d i dst;
          (match dst with
          | Buffer b when carried d b ->
              invalid_argf "Rig.%s: part %d fills this process's memory" fn i
          | Buffer _ -> incr nfixed
          | Slot _ -> ());
          incr ncopies
      | Launch l ->
          check_launch fn d i l.fn;
          nrefs := !nrefs + Array.length slots.(i);
          ncodes := !ncodes + List.length codes.(i))
    parts;
  let memory = Array.of_list fixed in
  Array.iter (check_memory fn d) memory;
  nfixed := !nfixed + Array.length memory;
  let fixed_reach =
    List.map (fun (b, (a : Def.access)) -> named (a = Read_write) b) fixed
  in
  let queues, afters = order fn d (Array.map reaches parts) fixed_reach parts in
  let nafter = Array.fold_left (fun n a -> n + Array.length a) 0 afters in
  let c =
    sub_new d.live.c (Array.length parts) nafter !nfixed uses !nrefs
      !ncodes !ncopies
  in
  let blocks = Array.make (Array.length parts) no_block in
  let copies = Array.make !nmoves (Slot 0, Slot 0) in
  let at = ref 0 and k = ref 0 and r = ref 0 and j = ref 0 and m = ref 0 in
  let n = ref 0 in
  let fixed = ref [] and images = ref [] in
  let fix_next b use =
    fix c !k b use;
    incr k;
    fixed := b :: !fixed
  in
  let slot_of = function Slot k -> k | Buffer _ -> -1 in
  let copy_side ~dst use = function
    | Slot _ -> ()
    | Buffer b ->
        sub_side c !m dst b (carried d b) (owns_device_memory d b);
        fix_next b use
  in
  let place_codes i place =
    List.iter
      (fun (at, code, image) ->
        sub_code c !j place at code;
        incr j;
        images := image :: !images)
      codes.(i)
  in
  Array.iteri
    (fun i p ->
      sub_part c i queues.(i) afters.(i) !at;
      at := !at + Array.length afters.(i);
      match p with
      | Words w ->
          sub_words c i (host_address fn w) (Buffer.length w / 4);
          fix_next w Use_read
      | Fill f ->
          let place = sub_fill c i f.fill f.params slots.(i) !r in
          place_codes i place;
          blocks.(i) <- (sub_id c, place, f.params, 0, "");
          r := !r + Array.length slots.(i)
      | Copy { src; dst } ->
          sub_copy c !m i (slot_of src) (slot_of dst);
          copies.(!n) <- (src, dst);
          copy_side ~dst:false Use_read src;
          copy_side ~dst:true Use_write dst;
          incr m;
          incr n
      | Set { dst; byte } ->
          sub_set c !m i (slot_of dst) byte;
          copy_side ~dst:true Use_write dst;
          incr m
      | Launch l ->
          let e = l.fn.fentry in
          let place = sub_launch c i e.code e.launch l.params slots.(i) !r in
          place_codes i place;
          let most = Option.value e.threads ~default:0 in
          blocks.(i) <- (sub_id c, place, l.params, most, l.fn.fname);
          r := !r + Array.length slots.(i);
          images := l.fn.fimage :: !images)
    parts;
  Array.iter (fun (b, a) -> fix_next b (use_of a)) memory;
  sub_seal c;
  let kernels =
    Array.to_list parts
    |> List.filter_map (function Launch l -> Some l.fn.fname | _ -> None)
  in
  let span =
    if Array.length parts = 0 then None
    else
      Some
        ( d.live.queues.(queues.(0)).name,
          if kernels = [] then kind_name (kind_of parts.(0))
          else String.concat ", " kernels )
  in
  {
    dev = d;
    c;
    fixed = Array.of_list !fixed;
    addressable = Array.length memory;
    images = !images;
    uses;
    naming;
    kept;
    held;
    blocks;
    span;
    copies;
  }

(* The memory is held first, so that no free returns it while [build] reads
   it, and given back at once if [build] raises. *)
let make ?keep ?(fixed = []) d parts =
  let fn = "Submission.make" in
  Dev.check_held fn;
  let parts = Array.copy parts in
  let held = Claim.held fn (named_buffers fixed parts) in
  match build keep ~fixed ~held d parts with
  | s -> s
  | exception e ->
      Claim.release held;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let block s i =
  if i < 0 || i >= Array.length s.blocks || s.blocks.(i) == no_block then
    invalid_argf "Rig.Submission.block: part %d is no launch or fill" i;
  s.blocks.(i)

(* Whether the bytes of [b] lie in those of [f]. *)
let within b f =
  b.mem.alloc == f.mem.alloc && f.offset <= b.offset
  && b.length <= f.length - (b.offset - f.offset)

let address s b =
  let fn = "Submission.address" in
  Buffer.check_live fn b;
  let rec find i =
    if i = s.addressable then
      invalid_argf "Rig.%s: no fixed memory of the submission holds the buffer"
        fn;
    let f = s.fixed.(i) in
    if within b f then Buffer.address fn f + (b.offset - f.offset)
    else find (i + 1)
  in
  find 0

(* Runs *)

(* A run's blocks and its state, C memory a custom block holds and frees once
   collected ([struct rig_run] in [rig_stubs.h]). *)
type run_c

external run_new : c -> run_c = "caml_rig_run_new"

(* Each setter is an external that takes the block as it is, untagged ints
   and unboxed floats, so no argument is boxed at a call, inlined or not. *)
module Run = struct
  type nonrec t = { r : run_c; sub : t }
  (* C reads the custom block from field 0. *)

  external take : t -> bool = "caml_rig_run_take" [@@noalloc]
  external release : t -> unit = "caml_rig_run_release" [@@noalloc]

  let make s = { r = run_new s.c; sub = s }
  let submission r = r.sub

  external groups :
    t ->
    block ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"

  external threads :
    t ->
    block ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"

  external shared : t -> block -> (int[@untagged]) -> unit
    = "caml_rig_run_shared_byte" "caml_rig_run_shared"

  external int32 : t -> block -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int32_byte" "caml_rig_run_int32"

  external int64 :
    t -> block -> (int[@untagged]) -> (int64[@unboxed]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"

  external float32 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"

  external float64 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
end

external call_collect : c -> call -> Run.t -> int array -> int
  = "caml_rig_call_collect"

external c_submit : c -> call -> Run.t -> int = "caml_rig_submit"

(* In-queue waits *)

let host_wait = -1

(* rig's array of timeline words, where every device's word lies, but for a
   word of a driver's own ([Def.live]'s [word] [0]). *)
external array_of_words : unit -> int * int = "caml_rig_edge_words"

let words_start, words_bytes = array_of_words ()

(* [d]'s mapping of the array, made at its first wait on a word in it: its
   first byte's address as [d]'s work addresses it, or [host_wait] if [d] maps
   none. *)
let map_words d =
  match Memory.map_host_range d words_start words_bytes with
  | None -> (host_wait, None)
  | Some m -> (
      match (Memory.locate m).address with
      | Some at -> (at, Some m)
      | None ->
          ignore (Memory.unmap d m : Dev.given);
          (host_wait, None))

(* Keeps [d]'s mapping of the array, unless another decision came first or
   [d]'s stop returned, as [set_pair] does for a pair. *)
let set_words (d, at, mapping) =
  let l = d.live in
  if l.words <> 0 then Some l.words
  else if Option.is_some mapping && Dev.stop_returned d then Some host_wait
  else begin
    Option.iter (fun m -> l.word_maps <- m :: l.word_maps) mapping;
    l.words <- at;
    None
  end

(* Where [d]'s work reads the array: one mapping for every producer, which [d]
   keeps until it is Stopped. Racy, as [pair]: the decision that comes second
   gives its mapping back. *)
let words_at d =
  let l = d.live in
  if l.words <> 0 then l.words
  else
    let at, mapping = map_words d in
    match Lock.protect l.lock set_words (d, at, mapping) with
    | None -> at
    | Some first ->
        Option.iter (fun m -> ignore (Memory.unmap d m : Dev.given)) mapping;
        first

(* How [d], a device of [p]'s driver, waits in its queue for [p]'s word where
   it does not reach rig's array: through its driver's mapping of [p]'s word
   region [r], which counts among [p]'s word unmaps. A [p] whose unmaps ended
   has stopped and its points are done: [d] waits on the host. *)
let map_word d p r =
  if not (Memory.hold_word p) then (host_wait, None)
  else
    match Memory.map_peer_region d r with
    | exception e ->
        Memory.word_unmapped p;
        reraise e
    | None ->
        Memory.word_unmapped p;
        (host_wait, None)
    | Some m -> (
        match (Memory.locate m).address with
        | Some at -> (at, Some (m, p))
        | None ->
            ignore (Memory.unmap_word d (m, p) : Dev.given);
            (host_wait, None))

(* How [d] waits for [p]'s values, decided once per pair: [host_wait], or for a
   queue that waits in it, the address of [p]'s word as [d] maps it: inside
   [d]'s one mapping of rig's array, or, for a device of [p]'s driver that maps
   no host memory, a mapping of [p]'s word region, which comes with the way. *)
let decide d p =
  let l = d.live and pl = p.live in
  let waits =
    match pl.completion with
    | Rig_edge.Store -> l.waits.stores
    | Host -> l.waits.hosts
  in
  if (not waits) || not (Dev.same_machine p d) then (host_wait, None)
  else if d.maps_host && pl.word <> 0 then
    let at = words_at d in
    ((if at = host_wait then at else at + (pl.word - words_start)), None)
  else
    match pl.word_region with
    | Some r when d.key = p.key -> map_word d p r
    | _ -> (host_wait, None)

let known_pair pairs i =
  if i < Array.length pairs then pairs.(i) else Memory.undecided

(* Grows [l.pairs] by replacement to hold the index [i]. *)
let grow l i =
  let n = Int.max (i + 1) (2 * Array.length l.pairs) in
  let a = Array.make n Memory.undecided in
  Array.blit l.pairs 0 a 0 (Array.length l.pairs);
  l.pairs <- a

type decided = Decided of pair | Replaced of pair | Taken of pair

(* Sets [d]'s pair of [p] to [way] with its mapping [map], unless a decision
   for [p] or a later device of its slot came first, or [d]'s stop returned,
   after which its drain owns its pairs: then the pair to use, and the caller
   gives [map] back. A pair decided for an earlier device of the slot is
   replaced, and its mapping goes among [d]'s retired ones at the value after
   every submit that may wait through it ({!Dev.next_value}), in the same
   section, so that a drain finds it in the cell or among them. Nothing
   allocates between the two stores. *)
let set_pair (d, (p : device), way, map) =
  let l = d.live and i = p.index in
  let known = known_pair l.pairs i in
  if known.occupant = p.opened then Taken known
  else if known.occupant > p.opened || Dev.stop_returned d then
    Taken Memory.undecided
  else begin
    let cell = { occupant = p.opened; way; word_map = map } in
    if i >= Array.length l.pairs then grow l i;
    (match known.word_map with
    | Some m -> Memory.retire_locked d m
    | None -> ());
    l.pairs.(i) <- cell;
    if known.occupant >= 0 then Replaced cell else Decided cell
  end

(* [l.pairs] is replaced whole under [d]'s lock and its cells are immutable
   pairs, so a read takes no lock. A pair decided for another device of [p]'s
   slot, or a stale undecided one, decides it again: the decision that comes
   second gives its mapping back. A decision for an earlier device of the
   slot comes from a submit whose point's device has since stopped: its wait
   on the host returns at once. A replacement gives back [d]'s retired
   mappings that are due, so a device that only submits gives back the ones
   it replaced. It frees no memory, since CUDA's frees of memory wait for
   every kernel running on the GPU: a word that loses its last mapping goes
   back at the next drain. A mapping's free waits for no work: CUDA's does
   nothing, and NV's and AMD's unmap at once. Is the pair to wait by. *)
let pair d (p : device) =
  let known = known_pair d.live.pairs p.index in
  if known.occupant = p.opened then known
  else
    let way, map = decide d p in
    match Lock.protect d.live.lock set_pair (d, p, way, map) with
    | Taken first ->
        Option.iter (fun m -> ignore (Memory.unmap_word d m : Dev.given)) map;
        first
    | Decided cell -> cell
    | Replaced cell ->
        Memory.give_retired d;
        cell

(* The way the submit of the call [c] on [d] waits for [p]'s values in its
   queue. A way through a mapping that [d] may give back makes the submit
   check its waits ({!Dev.next_value}). *)
let way_of d c (p : device) =
  let w = pair d p in
  if Option.is_some w.word_map then call_check c;
  w.way

(* Submit *)

let fn = "submit"

let counted n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")

let check_count s buffers =
  let n = Array.length buffers and slots = Array.length s.uses in
  if n <> slots then
    invalid_argf "Rig.%s: %s for a submission of %s" fn (counted n "buffer")
      (counted slots "buffer")

(* Raises the loss of a device other than [s]'s whose memory [b] is, or whose
   borrow it is. [s]'s device's own loss is the hand-over's. *)
let check_owner s b =
  let m = b.mem in
  if m.dev != s.dev && Dev.is_lost m.dev then Dev.raise_lost m.dev;
  let o = m.alloc.owner in
  if o != s.dev && o != m.dev && Dev.is_lost o then Dev.raise_lost o

(* Refuses [b], the run's buffer [k], used with [use], unless it is live, on
   [s]'s device or carried for copies alone, and, written, admits writes; and
   hands its stamps and address to the call's slot [k], and what a copy passes
   the driver to its side [k]. *)
let name_one s c use k b =
  if not (Buffer.is_live b) then
    invalid_argf "Rig.%s: buffers.(%d) is dead: %s" fn k b.mem.alloc.why;
  let naming = Array.unsafe_get s.naming k in
  let carried =
    match naming with Copies -> carried s.dev b | Refs | Both -> false
  in
  (* A queue reaches other memory only through a borrow on its device. *)
  if b.mem.dev != s.dev && not carried then
    invalid_argf "Rig.%s: buffers.(%d) is on %s, not on %s: borrow it" fn k
      b.mem.dev.name s.dev.name;
  check_owner s b;
  let a = b.mem.alloc in
  if a.entry == Memory.no_entry then Memory.ensure_entry a;
  let e = a.entry in
  if use <> Use_read && Buffer.access b = Def.Read then
    invalid_argf "Rig.%s: buffers.(%d) is written and admits only reads" fn k;
  (* -1 for memory with no address, which the collect refuses a ref to. *)
  let address = if b.mem.address < 0 then -1 else b.mem.address + b.offset in
  call_slot c k e.stamps address;
  match naming with
  | Refs -> ()
  | Copies | Both ->
      call_side c k b carried (owns_device_memory s.dev b)

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, or that its queue has no room for, and adds the others to the call's
   waits after committing their producers' work. *)
let rec wait_points s c n i count =
  if i = n then ()
  else
    let p = call_point c i in
    if Dev.is_done p then wait_points s c n (i + 1) count
    else
      let producer = Dev.of_index (Point.index p) and v = Point.value p in
      if Dev.is_lost producer || not (Dev.owns producer p) then begin
        Dev.check p;
        wait_points s c n (i + 1) count
      end
      else
        let way =
          if count >= s.dev.live.waits.most then host_wait
          else way_of s.dev c producer
        in
        if way = host_wait then begin
          (* The producer may have stopped since: a point it reached is
             done. *)
          Dev.wait_point 0 true p;
          wait_points s c n (i + 1) count
        end
        else begin
          Dev.commit producer v;
          call_wait c producer.index way v;
          wait_points s c n (i + 1) (count + 1)
        end

let rec hand_over s c run round lacked =
  let d = s.dev in
  let r = c_submit s.c c run in
  Dev.run_own d;
  if r = Dev.Answer.ok then Point.make d.index (call_value c)
  else if r = Dev.Answer.busy then hand_over s c run round lacked
  else if r = Dev.Answer.no_room then begin
    let at = call_no_room_at c in
    let w = Dev.word d in
    if w < at then Dev.wait d (w + 1);
    hand_over s c run round lacked
  end
  else if r = Dev.Answer.no_memory then raise Out_of_memory
  else if r = Dev.Answer.grow then
    let round, lacked = Memory.grow_queues d ~round ~lacked in
    hand_over s c run round lacked
  else if r = Dev.Answer.never then
    invalid_argf
      "Rig.%s: the parts never fit %s's queues, name work its driver does not \
       run, or hold a launch whose block it refuses"
      fn d.name
  else if r = Dev.Answer.producer_lost then
    Dev.raise_point_lost (call_producer c)
  else Dev.raise_lost d

let held_exclusive () = invalid_arg "Rig.submit: the memory is held exclusive"

(* Read claims on the [m] buffers of a frame, [m] from 1 to 4, taken in order:
   one that fails releases those taken before it. *)
let hold_frame m b0 b1 b2 b3 =
  if not (Memory.claim_add b0) then held_exclusive ();
  if m > 1 && not (Memory.claim_add b1) then begin
    Memory.claim_drop b0;
    held_exclusive ()
  end;
  if m > 2 && not (Memory.claim_add b2) then begin
    Memory.claim_drop b0;
    Memory.claim_drop b1;
    held_exclusive ()
  end;
  if m > 3 && not (Memory.claim_add b3) then begin
    Memory.claim_drop b0;
    Memory.claim_drop b1;
    Memory.claim_drop b2;
    held_exclusive ()
  end

let release_frame m b0 b1 b2 b3 =
  Memory.claim_drop b0;
  if m > 1 then Memory.claim_drop b1;
  if m > 2 then Memory.claim_drop b2;
  if m > 3 then Memory.claim_drop b3

(* Names the run's buffers from the [k]th on, then hands the work over. Each
   buffer is read once from its array and stays reachable in a frame until the
   hand-over returned: the C slots hold its stamps without a reference, and the
   caller's array may change meanwhile. The frame holds a read claim on each
   until then too, so that no donation writes one in place while the submit
   runs. A frame keeps four buffers, so the rooting costs a call per four. *)
let rec go s c run buffers waits k =
  let n = Array.length s.uses in
  if k >= n then begin
    wait_points s c (call_collect s.c c run waits) 0 0;
    hand_over s c run 1 0
  end
  else
    let m = min 4 (n - k) in
    let b0 = Array.unsafe_get buffers k in
    let b1 = if m > 1 then Array.unsafe_get buffers (k + 1) else b0 in
    let b2 = if m > 2 then Array.unsafe_get buffers (k + 2) else b0 in
    let b3 = if m > 3 then Array.unsafe_get buffers (k + 3) else b0 in
    hold_frame m b0 b1 b2 b3;
    match
      let a = s.uses in
      name_one s c (Array.unsafe_get a k) k b0;
      if m > 1 then name_one s c (Array.unsafe_get a (k + 1)) (k + 1) b1;
      if m > 2 then name_one s c (Array.unsafe_get a (k + 2)) (k + 2) b2;
      if m > 3 then name_one s c (Array.unsafe_get a (k + 3)) (k + 3) b3;
      go s c run buffers waits (k + 4)
    with
    | p ->
        release_frame m b0 b1 b2 b3;
        p
    | exception e ->
        release_frame m b0 b1 b2 b3;
        raise e

(* What each copy part of [s] moves, read at the submit from the run's
   [buffers]: the devices the memory its bytes leave and reach counts in, and
   their count. *)
let moves_of s buffers =
  let buffer = function Slot k -> buffers.(k) | Buffer b -> b in
  Array.map
    (fun (src, dst) ->
      let src = buffer src and dst = buffer dst in
      (Memory.account src.mem.alloc, Memory.account dst.mem.alloc, src.length))
    s.copies

(* What a submit reads for the profiles taken before it hands its work over: the
   profiles, each copy part's move, and when the copies were asked. *)
type taking = {
  ps : Profile.t list;
  moves : (device * device * int) array;
  asked : int;
}

(* The taking of a submit while no profile is taken. *)
let untaken = { ps = []; moves = [||]; asked = 0 }

(* Gives back [d]'s owed time pairs whose values [d] reached, one at a time,
   each once its driver settled it, after which the driver writes it no
   more. A pair taken here and not given back, as by an exception in the
   settle, stays with the device. *)
let rec give_owed d =
  let v = times_due d.live.c in
  if v > 0 then begin
    ignore (Dev.settle d v);
    times_give d.live.c v;
    give_owed d
  end

(* The holds of a time pair, one per profile's entry, and whether a read
   settled its value, after which its driver writes it no more. *)
type holds = { mutable left : int; [@atomic] mutable settled : bool [@atomic] }

(* Gives back the time pair [pair] of the value [v] of the device whose C
   record is [c], once [h]'s last hold ended: at once if a read settled [v],
   and otherwise owed, since the device may still write it, until
   [give_owed] finds [v] reached. Allocates nothing. *)
let give_pair c h v pair =
  if pair >= 0 then if h.settled then times_free c pair else times_owe c pair v

(* Leaves to each profile of [t.ps] an entry that reads into it the work of
   [s]'s value [p] and a copy per move of [t.moves] once [p] is reached, from
   the time pair [pair] its driver wrote, which the last entry to end gives back
   ([give_pair]). A driver that left the pair 0, or no pair, timed nothing: no
   work, and the copies run from [t.asked] to when the host saw [p] reached, now
   if the device ran the work in its submit, and otherwise at the read. A settle
   that failed or found the device lost records nothing and raises [Lost]. An
   asynchronous exception in the call may leave the pair held by profiles that
   never took the entry: the pair is then never given back, and never reused,
   as rig.mli allows of the memory of an interrupted call. *)
let record s (lane, name) { ps; moves; asked } p pair =
  let device = s.dev in
  let c = device.live.c and value = Point.value p in
  let seen = if Dev.is_done p then Profile.now () else -1 in
  let h = { left = List.length ps; settled = false } in
  let drop () =
    if Atomic.Loc.fetch_and_add [%atomic.loc h.left] (-1) = 1 then
      give_pair c h value pair
  in
  let read prof =
    let copies start stop =
      Array.iter
        (fun (src, dst, bytes) ->
          Profile.add [ prof ]
            (Copy { src; dst; bytes; point = Some p; start; stop }))
        moves
    in
    if not (pair < 0 || Dev.settle device value) then Dev.raise_lost device;
    h.settled <- true;
    match if pair < 0 then (0, 0) else times_read c pair with
    | 0, 0 -> copies asked (if seen >= 0 then seen else Profile.now ())
    | start, stop ->
        Profile.add [ prof ]
          (Work { point = p; device; lane; name; start; stop });
        copies start stop
  in
  let e = { Profile.device; value; read; drop } in
  List.iter (fun prof -> Profile.pend prof e) ps

(* The taking of a submit of [s] with [buffers], which first gives back
   the device's owed pairs it can. The last profile may stop between the two
   reads: the work is then untimed, since no entry would give its time pair
   back. *)
let taking s buffers =
  match s.span with
  | Some _ when Profile.enabled () -> (
      match Profile.active () with
      | [] -> untaken
      | ps ->
          give_owed s.dev;
          { ps; moves = moves_of s buffers; asked = Profile.now () })
  | _ -> untaken

external submit_begin : Run.t -> c -> int = "caml_rig_submit_begin"
external submit_end : Run.t -> call -> unit = "caml_rig_submit_end" [@@noalloc]

(* Submits [s] with its run [run]. [submit_begin] moves the run to
   submitting and takes a call, and is the last step before the trap that
   ends both, so in native code an exception anywhere after it, such as one
   where the taking allocates, leaves neither held. The C submit takes a time
   pair in the call that hands the work over, so a pair the trap finds was
   handed over with the work: the device may still write it, and it stays
   with the device. The C submit reads [s]'s C form, the run's blocks, and
   the stamps and templates they name without a reference, with the runtime
   released: their owners stay reachable until the submit ends. While a
   profile is taken, a submission with parts times its value. *)
let submit_run s run buffers waits =
  let c = submit_begin run s.c in
  if c = -1 then invalid_argf "Rig.%s: another thread took the run" fn;
  if c = -2 then invalid_argf "Rig.%s: another submit is using the run" fn;
  let t = ref untaken in
  match
    t := taking s buffers;
    call_timed c (!t != untaken);
    (* The submission holds its fixed and part-owned buffers, so they are live
       and no donation writes them: only their devices may be lost. *)
    for i = 0 to Array.length s.fixed - 1 do
      check_owner s (Array.unsafe_get s.fixed i)
    done;
    go s c run buffers waits 0
  with
  | p ->
      let pair = call_pair c in
      submit_end run c;
      (match s.span with
      | Some span when !t != untaken -> record s span !t p pair
      | _ -> ());
      ignore (Sys.opaque_identity s.c);
      ignore (Sys.opaque_identity s.fixed);
      ignore (Sys.opaque_identity s.images);
      ignore (Sys.opaque_identity s.kept);
      ignore (Sys.opaque_identity s.held);
      ignore (Sys.opaque_identity run.r);
      p
  | exception e ->
      submit_end run c;
      raise e

(* A submit that raises before [submit_begin], or anywhere, an asynchronous
   exception included, frees the run its thread took: [Run.release] does
   nothing to a run the thread does not hold, such as one a raising
   [submit_run] already freed. *)
let submit ?(waits = [||]) (run : Run.t) ~buffers =
  match
    Dev.check_held "submit";
    let s = run.sub in
    if Dev.is_lost s.dev then Dev.raise_lost s.dev;
    check_count s buffers;
    submit_run s run buffers waits
  with
  | p -> p
  | exception e ->
      Run.release run;
      raise e

(* Each device's run of one copy from slot 0 into slot 1, which [make]
   places on its copy engine. The submission has no launch or fill, so its
   run holds no bytes, and every domain submits it at once. *)
let copiers =
  Dev.new_key (fun d ->
      Run.make (make d [| copy ~src:(Slot 0) ~dst:(Slot 1) |]))

let run_copy d ~src ~dst =
  let run = Dev.local copiers d in
  submit_run run.sub run [| src; dst |] [||]

(* A fill's byte is part of its submission, so each makes its own: one
   part, no launch or fill, and a run that holds no bytes. *)
let run_set d ~dst c =
  let run = Run.make (make d [| set ~dst:(Slot 0) c |]) in
  submit_run run.sub run [| dst |] [||]
