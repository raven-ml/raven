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
  | Entry of { at : int; image : image; f : string }

let read slot ~at = Slot_ref { at; slot; use = Use_read }
let write slot ~at = Slot_ref { at; slot; use = Use_write }
let unordered slot ~at = Slot_ref { at; slot; use = Use_unordered }
let entry image f ~at = Entry { at; image; f }
let ref_at = function Slot_ref { at; _ } | Entry { at; _ } -> at

type side = Slot of int | Buffer of buffer

type part =
  | Words of buffer
  | Fill of { fill : nativeint; params : int; refs : ref array }
  | Copy of { src : side; dst : side }
  | Launch of { image : image; kernel : string; params : int; refs : ref array }

let copy ~src ~dst = Copy { src; dst }
let words b = Words b

(* The caller's array is copied: a part is fixed once made. *)
let fill ?(refs = [||]) fill ~params = Fill { fill; params; refs = Array.copy refs }

let launch ?(refs = [||]) image kernel ~params =
  Launch { image; kernel; params; refs = Array.copy refs }

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

external sub_side : c -> int -> bool -> buffer -> bool -> bool -> unit
  = "caml_rig_sub_side_byte" "caml_rig_sub_side"

external sub_seal : c -> unit = "caml_rig_sub_seal"

external reraise : exn -> 'a = "%reraise"

external sub_id : c -> int = "caml_rig_sub_id" [@@noalloc]

external sub_fixed : c -> int -> int -> use -> unit
  = "caml_rig_sub_fixed"

external ensure_record : Def.c -> int -> unit = "caml_rig_ensure_record"

(* Runs *)

(* A run's state, C memory a custom block holds and frees once collected. *)
type run

external run_new : unit -> run = "caml_rig_run_new"
external run_take : run -> c -> int = "caml_rig_run_take" [@@noalloc]
external run_fit : run -> c -> unit = "caml_rig_run_fit"
external run_give : run -> unit = "caml_rig_run_give" [@@noalloc]

(* Takes [run] for a submit of [c], fitting it to [c] unless it served [c] last:
   whether no other submit held it. *)
let take run c =
  match run_take run c with
  | 0 -> false
  | 1 -> true
  | 2 ->
      run_fit run c;
      true
  | 3 ->
      invalid_arg
        "Rig.submit: the run holds another submission's blocks"
  | _ -> invalid_arg "Rig.submit: a store is using the run"

external run_slot : run -> int -> int -> int -> unit
  = "caml_rig_run_slot"
[@@noalloc]

external run_side : run -> int -> buffer -> bool -> bool -> unit
  = "caml_rig_run_side"
[@@noalloc]

external run_collect : c -> run -> int array -> int
  = "caml_rig_run_collect"

external run_point : run -> int -> int = "caml_rig_run_point" [@@noalloc]

external run_wait : run -> int -> int -> int -> int -> unit
  = "caml_rig_run_wait"

external run_check : run -> unit = "caml_rig_run_check" [@@noalloc]

external run_value : run -> int = "caml_rig_run_value" [@@noalloc]
external run_no_room_at : run -> int = "caml_rig_run_no_room_at" [@@noalloc]
external run_producer : run -> int = "caml_rig_run_producer" [@@noalloc]
external c_submit : c -> run -> int = "caml_rig_submit"
external run_timed : run -> bool -> unit = "caml_rig_run_timed" [@@noalloc]
external run_pair : run -> int = "caml_rig_run_pair" [@@noalloc]
external times_read : Def.c -> int -> int * int = "caml_rig_times_read"
external times_free : Def.c -> int -> unit = "caml_rig_times_free" [@@noalloc]

(* A launch's or a fill's block, as the setters' C reads it: the id of its
   submission's C form, where it lies in a run, as [caml_rig_sub_launch]
   encodes it, and the most threads a group of its function has, [0] for a
   fill or where the driver states none, which [Run.threads] checks, with the
   function's name for its refusal. *)
type block = int * int * int * string

(* Each setter is an external that takes the block as it is, untagged ints
   and unboxed floats, so no argument is boxed at a call, inlined or not. *)
module Run = struct
  type t = run

  let make = run_new

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

  external int64 : t -> block -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"

  external float32 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"

  external float64 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
end

(* What names a run's slot: refs, copies or both. A slot only copies name may
   be this process's memory on a device of another machine, whose driver
   carries its bytes. *)
type naming = Refs | Copies | Both

type t = {
  dev : device;
  c : c;
  fixed : buffer array;
      (** The buffers [c] names, checked live at each submit and kept reachable:
          [c] holds their stamps without a reference. *)
  addressable : int;
      (** The first [addressable] buffers of [fixed] are the caller's fixed
          memory; the others belong to parts. *)
  images : image list;
      (** The images of [c]'s launches, kept loaded: [c] names their templates,
          which live while the image is loaded. *)
  uses : use array;  (** Each slot's use. *)
  naming : naming array;  (** What names each slot. *)
  hold : token option;
      (** The token of the value the submission keeps, kept reachable: its
          collection lets the value go once the work is done. *)
  blocks : block array;
      (** Each part's block, [no_block] for no launch or fill. *)
  span : (string * string) option;
      (** The first part's queue and the launches' functions, or the first
          part's kind. [None] for no parts. *)
  copies : (side * side) array;  (** The copy parts' sides, in order. *)
}

let no_block = (0, -1, 0, "")

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

(* Refuses part [i]'s copy on [d] unless each buffer end is live and [d]'s, or
   carried, the [dst] buffer admits writes, and two buffer ends have one
   length, share no byte and are not both carried. A slot end's buffer only a
   run gives, which the submit checks. *)
let check_ends fn d i src dst =
  let check = function
    | Slot _ -> ()
    | Buffer b ->
        Buffer.check_live fn b;
        if b.mem.dev != d && not (carried d b) then
          invalid_argf "Rig.%s: part %d copies a buffer on %s, not on %s: \
                        borrow it"
            fn i b.mem.dev.name d.name
  in
  check src;
  check dst;
  (match dst with
  | Buffer b when Buffer.access b = Def.Read ->
      invalid_argf "Rig.%s: part %d copies into memory that admits only reads"
        fn i
  | Slot _ | Buffer _ -> ());
  match (src, dst) with
  | Buffer a, Buffer b ->
      if Buffer.length a <> Buffer.length b then
        invalid_argf "Rig.%s: part %d's buffers differ in size" fn i;
      if Buffer.overlaps a b then
        invalid_argf "Rig.%s: part %d's buffers overlap" fn i;
      if carried d a && carried d b then
        invalid_argf "Rig.%s: neither of part %d's buffers is %s's memory" fn i
          d.name
  | _ -> ()

(* Refuses a device that runs no submitted work. *)
let check_device fn d =
  if Dev.is_lost d then Dev.raise_lost d;
  if Dev.is_io d then
    invalid_argf "Rig.%s: %s runs no submitted work" fn d.name

(* The most parameter bytes of a launch, [rig_edge.h]'s RIG_PARAMS. *)
let max_params = 4096

(* Refuses part [i]'s parameters unless they are at most [max_params] bytes and
   its refs lie among them and share no [at]. A ref's end is checked as [at >
   params - 8] once [params] is bounded: [at + 8] wraps for an [at] near
   [max_int]. *)
let check_args fn i params refs =
  if params < 0 || params > max_params then
    invalid_argf "Rig.%s: part %d has %d parameter bytes, not 0 to %d" fn i
      params max_params;
  Array.iter
    (fun r ->
      let at = ref_at r in
      if at < 0 || at mod 8 <> 0 || at > params - 8 then
        invalid_argf
          "Rig.%s: part %d's ref at %d is not 8 aligned bytes of its %d" fn i at
          params)
    refs;
  let ats = Array.map ref_at refs in
  Array.sort Int.compare ats;
  for k = 1 to Array.length ats - 1 do
    if ats.(k) = ats.(k - 1) then
      invalid_argf "Rig.%s: part %d has two refs at %d" fn i ats.(k)
  done

(* Refuses part [i]'s launch on [d] unless its image is [d]'s and has its
   function, and its parameters hold the bytes the function reads and each ref
   lies among them, where the driver states them: the entry. *)
let launch_entry fn d i image kernel params refs =
  if image.idev != d then
    invalid_argf "Rig.%s: part %d's image is not loaded on %s" fn i d.name;
  match Memory.kernel_entry image kernel with
  | None ->
      invalid_argf "Rig.%s: part %d's image has no function %S" fn i kernel
  | Some ({ params = None; _ } as e) -> e
  | Some ({ params = Some takes; _ } as e) ->
      if params < takes then
        invalid_argf
          "Rig.%s: %s takes %d bytes of parameters; part %d gives %d" fn kernel
          takes i params;
      Array.iter
        (fun r ->
          let at = ref_at r in
          if at > takes - 8 then
            invalid_argf
              "Rig.%s: part %d's ref at %d lies past the %d bytes of \
               parameters %s takes"
              fn i at takes kernel)
        refs;
      e

(* Refuses fixed memory that is dead, not [d]'s or, written, admits only
   reads. *)
let check_memory fn d (b, (access : Def.access)) =
  Buffer.check_live fn b;
  if b.mem.dev != d then
    invalid_argf "Rig.%s: a fixed buffer is on %s, not on %s: borrow it" fn
      b.mem.dev.name d.name;
  if access = Read_write && Buffer.access b = Def.Read then
    invalid_argf "Rig.%s: a fixed buffer written admits only reads" fn

(* The value of part [i]'s entry ref to [image]'s [f] on [d]. *)
let entry_code fn d i image f =
  if image.idev != d then
    invalid_argf "Rig.%s: part %d's entry ref names an image not loaded on %s"
      fn i d.name;
  match Memory.kernel_entry image f with
  | Some e -> e.code
  | None ->
      invalid_argf "Rig.%s: part %d's entry ref: its image has no function %S"
        fn i f

let refs_of = function
  | Fill { refs; _ } | Launch { refs; _ } -> refs
  | Words _ | Copy _ -> [||]

(* The buffer refs of [refs], which C places. *)
let slot_refs refs =
  Array.to_seq refs
  |> Seq.filter_map (function Slot_ref r -> Some r | Entry _ -> None)
  |> Array.of_seq

(* Slots *)

(* The slots part [i] names, how its work uses each, and by what: a copy reads
   its [src] and writes its [dst]. *)
let slots_of fn i = function
  | Copy { src = Slot k; dst = Slot k' } when k = k' ->
      invalid_argf "Rig.%s: part %d copies slot %d into itself" fn i k
  | Copy { src; dst } ->
      let named use = function
        | Slot k -> [ (k, use, Copies) ]
        | Buffer _ -> []
      in
      named Use_read src @ named Use_write dst
  | p ->
      Array.to_list
        (Array.map (fun r -> (r.slot, r.use, Refs)) (slot_refs (refs_of p)))

(* Each slot's use and what names it, from the refs and copies that name it:
   [Use_write] if one writes it. The slots are 0 to the highest one named,
   each named, and none used both unordered and ordered: a gap or a mix is a
   mistaken index. *)
let slot_uses fn parts =
  let named = Array.mapi (slots_of fn) parts in
  let n =
    Array.fold_left
      (List.fold_left (fun n (k, _, _) ->
           if k < 0 then invalid_argf "Rig.%s: a part names slot %d" fn k;
           max n (k + 1)))
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
  | Copy _ -> (
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
    | Fill _ | Copy _ | Launch _ -> ());
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
let hold d v =
  let t = Memory.token d.live.release (Release (Held v)) 0 max_int (-1) in
  if t == Memory.no_token then Dev.raise_lost d;
  t

let build keep ~fixed d parts =
  let parts = Array.copy parts in
  let fn = "Submission.make" in
  check_device fn d;
  let hold = Option.map (hold d) keep in
  let uses, naming = slot_uses fn parts in
  let check_buffer = Buffer.check_live fn in
  let nfixed = ref 0 and nrefs = ref 0 and ncodes = ref 0 in
  let ncopies = ref 0 in
  let entries = Array.make (Array.length parts) None in
  let slots = Array.map (fun p -> slot_refs (refs_of p)) parts in
  (* Each part's entry refs: where they lie in its parameters, their value
     and their image. *)
  let codes =
    Array.mapi
      (fun i p ->
        Array.to_list (refs_of p)
        |> List.filter_map (function
          | Slot_ref _ -> None
          | Entry { at; image; f } -> Some (at, entry_code fn d i image f, image)))
      parts
  in
  Array.iteri
    (fun i p ->
      match p with
      | Words w ->
          check_buffer w;
          ignore (host_address fn w);
          if Buffer.length w mod 4 <> 0 then
            invalid_argf
              "Rig.%s: part %d's words are of %d bytes, not a multiple of 4" fn
              i (Buffer.length w);
          incr nfixed
      | Fill f ->
          check_args fn i f.params f.refs;
          nrefs := !nrefs + Array.length slots.(i);
          ncodes := !ncodes + List.length codes.(i)
      | Copy { src; dst } ->
          check_ends fn d i src dst;
          let buffer = function Buffer _ -> 1 | Slot _ -> 0 in
          nfixed := !nfixed + buffer src + buffer dst;
          incr ncopies
      | Launch l ->
          check_args fn i l.params l.refs;
          entries.(i) <-
            Some (launch_entry fn d i l.image l.kernel l.params l.refs);
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
  let copies = Array.make !ncopies (Slot 0, Slot 0) in
  let at = ref 0 and k = ref 0 and r = ref 0 and j = ref 0 and m = ref 0 in
  let fixed = ref [] and images = ref [] in
  let fix_next b use =
    fix c !k b use;
    incr k;
    fixed := b :: !fixed
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
          blocks.(i) <- (sub_id c, place, 0, "");
          r := !r + Array.length slots.(i)
      | Copy { src; dst } ->
          let slot = function Slot k -> k | Buffer _ -> -1 in
          sub_copy c !m i (slot src) (slot dst);
          copies.(!m) <- (src, dst);
          let side ~dst use = function
            | Slot _ -> ()
            | Buffer b ->
                sub_side c !m dst b (carried d b) (owns_device_memory d b);
                fix_next b use
          in
          side ~dst:false Use_read src;
          side ~dst:true Use_write dst;
          incr m
      | Launch l ->
          let e = Option.get entries.(i) in
          let place = sub_launch c i e.code e.launch l.params slots.(i) !r in
          place_codes i place;
          let most = Option.value e.threads ~default:0 in
          blocks.(i) <- (sub_id c, place, most, l.kernel);
          r := !r + Array.length slots.(i);
          images := l.image :: !images)
    parts;
  Array.iter (fun (b, a) -> fix_next b (use_of a)) memory;
  sub_seal c;
  let kernels =
    Array.to_list parts
    |> List.filter_map (function Launch l -> Some l.kernel | _ -> None)
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
    hold;
    blocks;
    span;
    copies;
  }

let make ?keep ?(fixed = []) d parts = build keep ~fixed d parts

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

(* In-queue waits *)

let rig_word = 0
let rig_object = 1
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
          Memory.unmap d m;
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
        Option.iter (Memory.unmap d) mapping;
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
            Memory.unmap_word d (m, p);
            (host_wait, None))

(* How [d] waits for [p]'s values, decided once per pair: [host_wait], or for a
   queue that waits in it, the producer's object for an object completion, or
   the address of [p]'s word as [d] maps it: inside [d]'s one mapping of rig's
   array, or, for a device of [p]'s driver that maps no host memory, a mapping
   of [p]'s word region, which comes with the way. *)
let decide d p =
  let l = d.live and pl = p.live in
  let waits =
    match pl.completion with
    | Store -> l.waits.stores
    | Object _ -> l.waits.objects && d.key = p.key
    | Host_writes -> l.waits.hosts
  in
  if (not waits) || not (Dev.same_machine p d) then (host_wait, None)
  else
    match (pl.completion, pl.word_region) with
    | Object o, _ -> (o, None)
    | _, _ when d.maps_host && pl.word <> 0 ->
        let at = words_at d in
        ((if at = host_wait then at else at + (pl.word - words_start)), None)
    | _, Some r when d.key = p.key -> map_word d p r
    | _, _ -> (host_wait, None)

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
        Option.iter (Memory.unmap_word d) map;
        first
    | Decided cell -> cell
    | Replaced cell ->
        Memory.give_retired d;
        cell

(* The way the run [run] of [d] waits for [p]'s values in its queue. A way
   through a mapping that [d] may give back makes the submit check its waits
   ({!Dev.next_value}). *)
let way_of d run (p : device) =
  let c = pair d p in
  if Option.is_some c.word_map then run_check run;
  c.way

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
   [s]'s device or carried for copies alone, and, written, of memory that
   admits writes; and hands its stamps and address to the run's slot [k], and
   what a copy passes the driver to its side [k]. *)
let name_one s run use k b =
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
  if use <> Use_read && e.access = Def.Read then
    invalid_argf
      "Rig.%s: buffers.(%d) is written and its memory admits only reads" fn k;
  (* -1 for memory with no address, which the collect refuses a ref to. *)
  let address = if b.mem.address < 0 then -1 else b.mem.address + b.offset in
  run_slot run k e.stamps address;
  match naming with
  | Refs -> ()
  | Copies | Both ->
      run_side run k b carried (owns_device_memory s.dev b)

(* Waits on the host for the foreign points [s]'s device cannot wait for in its
   queue, or that its queue has no room for, adds the others to the run's waits
   after committing their producers' work, and is their number. *)
let rec wait_points s run n i count =
  if i = n then count
  else
    let p = run_point run i in
    if Dev.is_done p then wait_points s run n (i + 1) count
    else
      let producer = Dev.of_index (Point.index p) and v = Point.value p in
      if Dev.is_lost producer || not (Dev.owns producer p) then begin
        Dev.check p;
        wait_points s run n (i + 1) count
      end
      else
        let way =
          if count >= s.dev.live.waits.most then host_wait
          else way_of s.dev run producer
        in
        if way = host_wait then begin
          (* The producer may have stopped since: a point it reached is
             done. *)
          Dev.wait_point 0 true p;
          wait_points s run n (i + 1) count
        end
        else begin
          Dev.commit producer v;
          let kind =
            match producer.live.completion with
            | Object _ -> rig_object
            | _ -> rig_word
          in
          run_wait run producer.index way v kind;
          wait_points s run n (i + 1) (count + 1)
        end

let rec hand_over s run nwaits =
  let d = s.dev in
  let r = c_submit s.c run in
  Dev.run_own d;
  if r = Dev.Answer.ok then Point.make d.index (run_value run)
  else if r = Dev.Answer.busy then hand_over s run nwaits
  else if r = Dev.Answer.no_room then begin
    let at = run_no_room_at run in
    let w = Dev.word d in
    if w < at then Dev.wait d (w + 1);
    hand_over s run nwaits
  end
  else if r = Dev.Answer.need_record then begin
    ensure_record d.live.c nwaits;
    hand_over s run nwaits
  end
  else if r = Dev.Answer.never then
    invalid_argf
      "Rig.%s: the parts never fit %s's queues, name work its driver does not \
       run, or hold a launch whose block it refuses"
      fn d.name
  else if r = Dev.Answer.producer_lost then Dev.raise_point_lost (run_producer run)
  else Dev.raise_lost d

external claim_hold : buffer -> bool = "caml_rig_claim_hold" [@@noalloc]
external claim_drop : buffer -> unit = "caml_rig_claim_drop" [@@noalloc]

external release_fixed : buffer array -> int -> unit
  = "caml_rig_claim_drop_prefix"
[@@noalloc]

let held_exclusive () = invalid_arg "Rig.submit: the memory is held exclusive"

(* Read claims on the [m] buffers of a frame, [m] from 1 to 4, taken in order:
   one that fails releases those taken before it. *)
let hold_frame m b0 b1 b2 b3 =
  if not (claim_hold b0) then held_exclusive ();
  if m > 1 && not (claim_hold b1) then begin
    claim_drop b0;
    held_exclusive ()
  end;
  if m > 2 && not (claim_hold b2) then begin
    claim_drop b0;
    claim_drop b1;
    held_exclusive ()
  end;
  if m > 3 && not (claim_hold b3) then begin
    claim_drop b0;
    claim_drop b1;
    claim_drop b2;
    held_exclusive ()
  end

let release_frame m b0 b1 b2 b3 =
  claim_drop b0;
  if m > 1 then claim_drop b1;
  if m > 2 then claim_drop b2;
  if m > 3 then claim_drop b3

(* Names the run's buffers from the [k]th on, then hands the work over. Each
   buffer is read once from its array and stays reachable in a frame until the
   hand-over returned: the C slots hold its stamps without a reference, and the
   caller's array may change meanwhile. The frame holds a read claim on each
   until then too, so that no donation writes one in place while the submit
   runs. A frame keeps four buffers, so the rooting costs a call per four. *)
let rec go s run buffers waits k =
  let n = Array.length s.uses in
  if k >= n then
    hand_over s run (wait_points s run (run_collect s.c run waits) 0 0)
  else
    let m = min 4 (n - k) in
    let b0 = Array.unsafe_get buffers k in
    let b1 = if m > 1 then Array.unsafe_get buffers (k + 1) else b0 in
    let b2 = if m > 2 then Array.unsafe_get buffers (k + 2) else b0 in
    let b3 = if m > 3 then Array.unsafe_get buffers (k + 3) else b0 in
    hold_frame m b0 b1 b2 b3;
    match
      let a = s.uses in
      name_one s run (Array.unsafe_get a k) k b0;
      if m > 1 then name_one s run (Array.unsafe_get a (k + 1)) (k + 1) b1;
      if m > 2 then name_one s run (Array.unsafe_get a (k + 2)) (k + 2) b2;
      if m > 3 then name_one s run (Array.unsafe_get a (k + 3)) (k + 3) b3;
      go s run buffers waits (k + 4)
    with
    | p ->
        release_frame m b0 b1 b2 b3;
        p
    | exception e ->
        release_frame m b0 b1 b2 b3;
        raise e

(* What each copy part of [s] moves, read at the submit from the run's
   [buffers]: the devices of the allocations its bytes leave and reach, and
   their count. *)
let moves_of s buffers =
  let buffer = function Slot k -> buffers.(k) | Buffer b -> b in
  Array.map
    (fun (src, dst) ->
      let src = buffer src and dst = buffer dst in
      (src.mem.alloc.owner, dst.mem.alloc.owner, src.length))
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

(* Leaves to each profile of [t.ps] an entry that reads into it the span of
   [s]'s value [p] and a copy per move of [t.moves] once [p] is reached, from
   the time pair [pair] its driver wrote, which the last entry to end gives
   back. A driver that left the pair 0, or no pair, timed nothing: no span,
   and the copies run from [t.asked] to when the host saw [p] reached, now if
   the device ran the work in its submit, and otherwise at the read. A
   settle that failed or found the device lost records nothing. *)
let record s (lane, name) { ps; moves; asked } p pair =
  let device = s.dev and value = Point.value p in
  let c = device.live.c in
  let seen = if Dev.is_done p then Profile.now () else -1 in
  let left = Atomic.make (List.length ps) in
  let drop () =
    if Atomic.fetch_and_add left (-1) = 1 && pair >= 0 then times_free c pair
  in
  let read prof =
    let copies start stop =
      Array.iter
        (fun (src, dst, bytes) ->
          Profile.add [ prof ] (Copy { src; dst; bytes; start; stop }))
        moves
    in
    if pair < 0 || Dev.settle device value then
      match if pair < 0 then (0, 0) else times_read c pair with
      | 0, 0 -> copies asked (if seen >= 0 then seen else Profile.now ())
      | start, stop ->
          Profile.add [ prof ] (Span { device; lane; name; start; stop });
          copies start stop
  in
  List.iter (fun prof -> Profile.pend prof { device; value; read; drop }) ps

(* Submits [s] with [run], which the caller took. The C submit reads [s]'s C
   form, and the stamps and templates it names without a reference, with the
   runtime released: their owners stay reachable until the run is given back.
   While a profile is taken, a submission with parts times its value. *)
let taken s run buffers waits =
  (* The last profile may stop between the two reads: the work is then
     untimed, since no entry would give its time pair back. *)
  let t =
    match s.span with
    | Some _ when Profile.enabled () -> (
        match Profile.active () with
        | [] -> untaken
        | ps -> { ps; moves = moves_of s buffers; asked = Profile.now () })
    | _ -> untaken
  in
  run_timed run (t != untaken);
  let claimed = ref 0 in
  match
    (* Fixed and part-owned buffers need the same protection as run buffers:
       donation cannot overwrite them before hand-over publishes their use. *)
    while !claimed < Array.length s.fixed do
      let b = Array.unsafe_get s.fixed !claimed in
      if not (claim_hold b) then held_exclusive ();
      incr claimed;
      Buffer.check_live fn b;
      check_owner s b
    done;
    go s run buffers waits 0
  with
  | p ->
      if !claimed > 0 then release_fixed s.fixed !claimed;
      let pair = run_pair run in
      run_give run;
      (match s.span with
      | Some span when t != untaken -> record s span t p pair
      | _ -> ());
      ignore (Sys.opaque_identity s.c);
      ignore (Sys.opaque_identity s.fixed);
      ignore (Sys.opaque_identity s.images);
      ignore (Sys.opaque_identity s.hold);
      p
  | exception e ->
      if !claimed > 0 then release_fixed s.fixed !claimed;
      run_give run;
      raise e

let submit ?(waits = [||]) s ~run ~buffers =
  if Dev.is_lost s.dev then Dev.raise_lost s.dev;
  check_count s buffers;
  if not (take run s.c) then
    invalid_argf "Rig.%s: another submit is using the run" fn;
  taken s run buffers waits

(* Each domain's run for copies; one another thread of the domain holds is
   replaced by a fresh one for the copy. *)
let copy_runs = Domain.DLS.new_key run_new

(* Each device's submission of one copy from slot 0 into slot 1, which [make]
   places on its copy engine. *)
let copiers =
  Dev.new_key (fun d -> make d [| copy ~src:(Slot 0) ~dst:(Slot 1) |])

let run_copy d ~src ~dst =
  let s = Dev.local copiers d in
  let run = Domain.DLS.get copy_runs in
  let run =
    if take run s.c then run
    else
      let fresh = run_new () in
      ignore (take fresh s.c : bool);
      fresh
  in
  taken s run [| src; dst |] [||]
