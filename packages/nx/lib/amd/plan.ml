(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Sub = Rig.Submission

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type slot =
  | Array of Rig.Buffer.t
  | Workspace
  | Tickets
  | Fresh of int
  | Constant of { key : string; length : int; bytes : unit -> bigbytes }

let kept_workspace = 64 * 1024 * 1024
let kept_tickets = 4 * 1024 * 1024
let kept_constants = 64 * 1024 * 1024
let kept_constant_count = 16
let kept_count = 64

(* The bounds {!finish} applies, below the kept sizes in tests. *)
let workspace_bound = Atomic.make kept_workspace
let tickets_bound = Atomic.make kept_tickets

let bound_fresh ~workspace ~tickets =
  if workspace < 0 || workspace > kept_workspace then
    invalid_arg "Nx_amd.bound_fresh: workspace";
  if tickets < 0 || tickets > kept_tickets then
    invalid_arg "Nx_amd.bound_fresh: tickets";
  Atomic.set workspace_bound workspace;
  Atomic.set tickets_bound tickets

let kept_submissions = 64 * 1024 * 1024

(* The most bytes of submissions a device keeps, below {!kept_submissions}
   in tests. *)
let kept_bound = Atomic.make kept_submissions

let bound_kept n =
  if n < 0 || n > kept_submissions then invalid_arg "Nx_amd.bound_kept";
  Atomic.set kept_bound n

let kept_most () = Atomic.get kept_bound

type ref = { word : int; slot : int; write : bool }

(* A launch {!traced} saw. *)
type traced = {
  kernel : Kernels.t;
  groups : int * int * int;
  tickets : bool;
  shared : int;
}

type launch = {
  kernel : Kernels.t;
  groups : int * int * int;
  shared : int;
  words : Bytes.t;
  used : int;
  refs : ref array;
}

type t = {
  launches : launch array;
  slots : slot array;
  workspace : int;
  tickets : int;
  bytes : int;
  kept : bool;
}

(* Building

   Slots get indices in the order the plan first names them. The workspace
   is carved as a stack: a call's launches run in order, each after the
   last, so a piece is free once the launches that use it are planned. A
   ticket word is zero at each launch's start, since the launch before left
   it zero, so every launch takes its tickets from the slot's start. *)

type builder = {
  mutable launched : launch list;
  mutable named : slot list;
  mutable count : int;
  mutable workspace_slot : int;
  mutable tickets_slot : int;
  mutable top : int;
  mutable peak : int;
  mutable ticket_bytes : int;
}

let make () =
  {
    launched = [];
    named = [];
    count = 0;
    workspace_slot = -1;
    tickets_slot = -1;
    top = 0;
    peak = 0;
    ticket_bytes = 0;
  }

let add b s =
  b.named <- s :: b.named;
  b.count <- b.count + 1;
  b.count - 1

let rec find x i = function
  | [] -> -1
  | Array y :: _ when y == x -> i
  | _ :: rest -> find x (i - 1) rest

let slot b x =
  match find x (b.count - 1) b.named with -1 -> add b (Array x) | i -> i

let rec find_key key i = function
  | [] -> None
  | Constant c :: _ when String.equal c.key key -> Some (i, c.length)
  | _ :: rest -> find_key key (i - 1) rest

let constant b ~key ~length bytes =
  if length < 0 then invalid_arg "Plan.constant: length";
  match find_key key (b.count - 1) b.named with
  | None -> add b (Constant { key; length; bytes })
  | Some (i, l) when l = length -> i
  | Some _ -> invalid_arg (Printf.sprintf "Plan.constant: %S's length" key)

let workspace b =
  if b.workspace_slot < 0 then b.workspace_slot <- add b Workspace;
  b.workspace_slot

let take b n =
  let at = (b.top + 255) land lnot 255 in
  b.top <- at + n;
  b.peak <- Int.max b.peak b.top;
  at

let mark b = b.top
let release b m = b.top <- m

let tickets_slot b =
  if b.tickets_slot < 0 then b.tickets_slot <- add b Tickets;
  b.tickets_slot

let tickets b n =
  b.ticket_bytes <- Int.max b.ticket_bytes (4 * n);
  0

(* Division by an invariant integer (Granlund and Montgomery, 1994): for
   [s = ⌈log₂ d⌉], [m = ⌊2³² (2^s - d) / d⌋ + 1] fits 32 bits, and
   [⌊n m / 2³²⌋ + n], shifted right by [s], is [n / d] for every [n] below
   2³². *)
let divisor d =
  if d < 1 || d >= 1 lsl 32 then invalid_arg "Plan.divisor";
  let rec log2_up s = if 1 lsl s >= d then s else log2_up (s + 1) in
  let s = log2_up 0 in
  let num = Int64.shift_left (Int64.of_int ((1 lsl s) - d)) 32 in
  let m = Int64.add (Int64.unsigned_div num (Int64.of_int d)) 1L in
  Int64.logor m (Int64.shift_left (Int64.of_int s) 32)

let words k = Bytes.make (8 * Kernels.words k) '\000'
let set w i v = Bytes.set_int64_le w (8 * i) (Int64.of_int v)

let small n = n < 1 lsl 32

let axis w i n =
  if n < 0 then invalid_arg "Plan.axis: extent";
  set w (i + Kernels.axis_n) n;
  let d = if n >= 1 && small n then divisor n else 0L in
  Bytes.set_int64_le w (8 * (i + Kernels.axis_div)) d

let layout w i l =
  let module L = Nx_array.Layout in
  let rank = L.rank l in
  set w (i + Kernels.layout_rank) rank;
  set w (i + Kernels.layout_at) (L.offset l);
  set w (i + Kernels.layout_small) (Bool.to_int (small (L.numel l)));
  for j = 0 to rank - 1 do
    let a = i + Kernels.layout_axes + (Kernels.layout_axis * j) in
    axis w a (L.dim l j);
    set w (a + Kernels.axis_strides) (L.stride l j)
  done;
  i + Kernels.layout_axes + (Kernels.layout_axis * rank)

(* The LDS a group of RDNA4 holds. *)
let group_lds = 64 * 1024

let shared_most kernel = group_lds - Kernels.lds kernel

(* The launches the domain planned since {!traced} began, the latest
   first. *)
let trace : traced list option Domain.DLS.key =
  Domain.DLS.new_key (fun () -> None)

let traced f =
  Domain.DLS.set trace (Some []);
  let taken () =
    let l = Option.value (Domain.DLS.get trace) ~default:[] in
    Domain.DLS.set trace None;
    List.rev l
  in
  match f () with
  | x -> (x, taken ())
  | exception e ->
      ignore (taken ());
      raise e

let launch b kernel ~groups ?(shared = 0) ?used words refs =
  if shared < 0 || shared > shared_most kernel then
    invalid_arg "Plan.launch: shared";
  let all = Kernels.words kernel in
  let used = Option.value used ~default:all in
  if used < 0 || used > all then invalid_arg "Plan.launch: used";
  if Array.exists (fun r -> r.word < 0 || r.word >= used) refs then
    invalid_arg "Plan.launch: a ref past the words used";
  (match Domain.DLS.get trace with
  | None -> ()
  | Some l ->
      let ticket r = r.slot = b.tickets_slot in
      let tickets = b.tickets_slot >= 0 && Array.exists ticket refs in
      Domain.DLS.set trace (Some ({ kernel; groups; tickets; shared } :: l)));
  b.launched <- { kernel; groups; shared; words; used; refs } :: b.launched

(* Past the groups the GPU holds at once, a grid that strides gains no
   bandwidth, while each group reads its launch's words across PCIe (about
   13 ns a KiB a group) and the launch's own cost grows with its groups
   (3.3 us at 1024, 8 us at 8192). The R9700 runs 2048 waves at once, 16 a
   SIMD on 64 compute units, which 1024 groups of 64 work-items or more
   fill. *)
let stride_groups = 1024

(* The slow band. On the R9700, a launch of a kernel of at most 96 VGPRs,
   which holds 16 waves on each SIMD, takes about 10 us more at some
   counts of groups: with groups of 256 work-items (8 waves), 253-256 and
   509-512 (14.3 and 16.4 us against 3.4 at 252 and 4.5 at 508; HIP's
   runtime alike); with 128, 509-512 and 1021-1024 (13.9-16.5 us against
   3.3-5.2 around them); with 64, 1017-1024 (13.8-14.2 us against 3.06 at
   1016 and 3.5-3.9 at 1025-1028), and none at 505-516. A kernel of 181
   VGPRs, 8 waves a SIMD, shows none, and 768 and 1024 groups of 256 are
   not slow. The rule that fits every count measured, its mechanism
   unknown: the groups, dealt one to each of 4 shader engines of 512 wave
   slots, fill the busiest engine's slots, or twice that, to within 4
   waves. Its bands for other group sizes are extrapolated. A striding
   grid of such a kernel steps to the first count past the band, or,
   where that is past {!stride_groups}, back to the last count before
   it. *)
let wave = 32
let full_vgprs = 96
let engines = 4
let engine_slots = 512
let band_waves = 4

(* Whether [groups] of [waves] each fill the busiest engine's slots, or
   twice that, to within [band_waves]. *)
let fills_engines ~waves groups =
  let busiest = waves * ((groups + engines - 1) / engines) in
  let filled k =
    let slots = k * engine_slots in
    busiest > slots - band_waves && busiest <= slots
  in
  filled 1 || filled 2

let waves threads = (threads + wave - 1) / wave

let in_band kernel groups =
  Kernels.vgprs kernel <= full_vgprs
  && fills_engines ~waves:(waves (Kernels.threads kernel)) groups

let past_band ~threads groups =
  if threads <= 0 then invalid_arg "Plan.past_band: threads";
  let waves = waves threads in
  if not (fills_engines ~waves groups) then groups
  else
    (* The slots the busiest engine fills, and the counts either side of
       their band: of 2 waves a group, 1016 below 1017-1024, and 1025. *)
    let per_engine = (groups + engines - 1) / engines in
    let fills = ((waves * per_engine) + engine_slots - 1) / engine_slots in
    let slots = fills * engine_slots in
    let above = (engines * (slots / waves)) + 1 in
    if above <= stride_groups then above
    else engines * ((slots - band_waves) / waves)

(* The most groups {!grid} gives, above every count but in tests. *)
let grid_bound = Atomic.make max_int

let bound_grid ~groups =
  if groups <= 0 then invalid_arg "Plan.bound_grid: groups";
  Atomic.set grid_bound groups

let grid ?per kernel units =
  let per = match per with Some p -> p | None -> Kernels.threads kernel in
  if per <= 0 then invalid_arg "Plan.grid: per";
  let groups = Int.max 1 (Int.min stride_groups ((units + per - 1) / per)) in
  let groups =
    if Kernels.vgprs kernel > full_vgprs then groups
    else past_band ~threads:(Kernels.threads kernel) groups
  in
  Int.min (Atomic.get grid_bound) groups

(* The words and grid of a Zero launch over the [bytes] from byte [at]. *)
let zero_words ~at bytes =
  if bytes < 0 || bytes mod 4 <> 0 then invalid_arg "Plan.zero: bytes";
  let words = words Kernels.Zero and count = bytes / 4 in
  let groups = grid Kernels.Zero count in
  set words Kernels.zero_dst at;
  set words Kernels.zero_count count;
  set words Kernels.zero_groups groups;
  (words, (groups, 1, 1))

let zero b ~slot ~at bytes =
  let words, groups = zero_words ~at bytes in
  launch b Kernels.Zero ~groups words
    [| { word = Kernels.zero_dst; slot; write = true } |]

(* A launch, not counted, that zeroes the first [bytes] of [slot]. *)
let zeroing slot bytes =
  let words, groups = zero_words ~at:0 bytes in
  {
    kernel = Kernels.Zero;
    groups;
    shared = 0;
    words;
    used = Kernels.words Kernels.Zero;
    refs = [| { word = Kernels.zero_dst; slot; write = true } |];
  }

(* Refuses a ref that names no slot of [slots] and a ref that writes a
   constant; marks each slot a ref names. *)
let named_slots slots launched =
  let n = Array.length slots in
  let named = Array.make n false in
  let name r =
    if r.slot < 0 || r.slot >= n then
      invalid_arg (Printf.sprintf "Plan.finish: no slot %d" r.slot);
    (match slots.(r.slot) with
    | Constant _ when r.write ->
        invalid_arg (Printf.sprintf "Plan.finish: slot %d is written" r.slot)
    | Constant _ | Array _ | Workspace | Tickets | Fresh _ -> ());
    named.(r.slot) <- true
  in
  List.iter (fun l -> Array.iter name l.refs) launched;
  named

(* [launches] and [slots] with the slots no ref names left out, each ref
   renumbered to its slot's index among the [named] ones. A key ({!key})
   holds refs alone, so a call binds the slots its refs name. *)
let leave_out named slots launches =
  let index = Array.make (Array.length named) (-1) and next = ref 0 in
  Array.iteri
    (fun i named ->
      if named then begin
        index.(i) <- !next;
        incr next
      end)
    named;
  let renumber l =
    let refs = Array.map (fun r -> { r with slot = index.(r.slot) }) l.refs in
    { l with refs }
  in
  let kept = List.filteri (fun i _ -> named.(i)) (Array.to_list slots) in
  (Array.of_list kept, Array.map renumber launches)

(* The bytes a device counts for a call's submission (nx_amd.mli): 1 KiB,
   and for each launch of [r] refs 32 (8 + r) bytes for its part, refs and
   key and for rig's image, slot and name of it, a name of at most
   {!name_bytes}. The blocks that hold a submit's values are rig's, which
   keeps them whatever submissions a device keeps ({!Rig.submit}). *)
let call_bytes = 1024
let name_bytes = 30
let launch_bytes l = 32 * (8 + Array.length l.refs)

let finish b =
  if b.launched = [] then invalid_arg "Plan.finish: no launch";
  let slots = Array.of_list (List.rev b.named) in
  let named = named_slots slots b.launched in
  let held i = i >= 0 && named.(i) in
  let workspace =
    if not (held b.workspace_slot) then 0
    else if b.peak <= Atomic.get workspace_bound then b.peak
    else begin
      slots.(b.workspace_slot) <- Fresh b.peak;
      0
    end
  in
  let fresh_tickets =
    held b.tickets_slot && b.ticket_bytes > Atomic.get tickets_bound
  in
  let tickets =
    if (not (held b.tickets_slot)) || fresh_tickets then 0 else b.ticket_bytes
  in
  if fresh_tickets then slots.(b.tickets_slot) <- Fresh b.ticket_bytes;
  let launches = List.rev b.launched in
  let launches =
    if fresh_tickets then zeroing b.tickets_slot b.ticket_bytes :: launches
    else launches
  in
  let launches = Array.of_list launches in
  let slots, launches =
    if Array.for_all Fun.id named then (slots, launches)
    else leave_out named slots launches
  in
  let bytes =
    Array.fold_left (fun n l -> n + launch_bytes l) call_bytes launches
  in
  let kept = bytes <= kept_most () in
  { launches; slots; workspace; tickets; bytes; kept }

let finish_some b = if b.launched = [] then None else Some (finish b)

(* Submissions

   A key is the count of launches, then each one's kernel, count of words,
   count of refs and its refs, each a word, a slot and 1 if written. *)

let key c =
  let ls = c.launches in
  let n =
    Array.fold_left (fun n l -> n + 3 + (3 * Array.length l.refs)) 1 ls
  in
  let k = Array.make n (Array.length ls) in
  let at = ref 1 in
  Array.iter
    (fun l ->
      k.(!at) <- Kernels.index l.kernel;
      k.(!at + 1) <- Kernels.words l.kernel;
      k.(!at + 2) <- Array.length l.refs;
      at := !at + 3;
      Array.iter
        (fun r ->
          k.(!at) <- r.word;
          k.(!at + 1) <- r.slot;
          k.(!at + 2) <- Bool.to_int r.write;
          at := !at + 3)
        l.refs)
    ls;
  k

let rec same_refs k at refs j =
  j = Array.length refs
  ||
  let r = refs.(j) in
  k.(at) = r.word
  && k.(at + 1) = r.slot
  && k.(at + 2) = Bool.to_int r.write
  && same_refs k (at + 3) refs (j + 1)

(* Whether [ls] from launch [i] match [k] from [at]. *)
let rec same k at ls i =
  if i = Array.length ls then at = Array.length k
  else
    let l = ls.(i) in
    let m = Array.length l.refs in
    at + 3 + (3 * m) <= Array.length k
    && k.(at) = Kernels.index l.kernel
    && k.(at + 1) = Kernels.words l.kernel
    && k.(at + 2) = m
    && same_refs k (at + 3) l.refs 0
    && same k (at + 3 + (3 * m)) ls (i + 1)

let matches k c =
  Array.length k > 0 && k.(0) = Array.length c.launches && same k 1 c.launches 0

let parts fs c =
  Array.map
    (fun l ->
      let refs =
        Array.map
          (fun r ->
            if r.write then Sub.write r.slot ~at:(8 * r.word)
            else Sub.read r.slot ~at:(8 * r.word))
          l.refs
      in
      Sub.launch ~refs fs.(Kernels.index l.kernel)
        ~params:(8 * Kernels.words l.kernel))
    c.launches

(* Each kernel's threads a group, by {!Kernels.index}, made once so that a
   launch's values take none of their own. *)
let group_threads = Array.map (fun k -> (Kernels.threads k, 1, 1)) Kernels.all

(* A launch's values: its words up to the last one its kernel reads, which
   costs a copy of them where it reads fewer than all. *)
let launch_args l =
  let params =
    if l.used = Kernels.words l.kernel then l.words
    else Bytes.sub l.words 0 (8 * l.used)
  in
  {
    Sub.groups = l.groups;
    threads = group_threads.(Kernels.index l.kernel);
    shared = l.shared;
    params;
  }

let args c = Array.map launch_args c.launches
