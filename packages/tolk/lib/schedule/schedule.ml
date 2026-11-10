(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let rule = Pattern_matcher.rule
let ops = Op.Set.of_list
let var = Upat.var
let setting = Setting.value

(* Ordered tables: the keys in the order they were first added. *)
module Ordered = struct
  type 'a t = { tbl : 'a Tbl.t; mutable keys : Ops.t list }

  let create () = { tbl = Tbl.create 16; keys = [] }
  let find_opt t k = Tbl.find_opt t.tbl k

  let replace t k x =
    if not (Tbl.mem t.tbl k) then t.keys <- k :: t.keys;
    Tbl.replace t.tbl k x

  let bindings t = List.rev_map (fun k -> (k, Tbl.find t.tbl k)) t.keys
end

(* Linearizing *)

(* Whether the parameter [p] is a view of the storage {!Ops.whole} names. *)
let is_view p = whole p != p

(* The data source a kernel reads: a kernel's output or a buffer. *)
let rec unwrap_src s =
  match (op s, src s) with
  | (Op.After | Op.Buffer | Op.Alloc | Op.Param), _ | _, [] -> s
  | _, x :: _ -> unwrap_src x

(* A buffer's state is an after or the storage. A join is none: read through
   joins, a source reads the state under them, with the joins' dependencies,
   which its reader waits for. *)
(* The state [s] reads, as {!value_state} gives it without the joins'
   dependencies. *)
let rec state_of s =
  let s = whole (unwrap_src s) in
  if is_join s then state_of (nth s 0) else s

let value_state s =
  let rec go deps s =
    let s = whole (unwrap_src s) in
    if is_join s then go (List.rev_append (List.tl (src s)) deps) (nth s 0)
    else
      match op s with
      | Op.After | Op.Buffer | Op.Alloc | Op.Param -> (s, List.rev deps)
      | o ->
          invalid_arg
            (Format.asprintf "a kernel's input is a buffer state, not %a" Op.pp
               o)
  in
  go [] s

(* A scalar argument of a precompiled call is no state: a weak integer
   expression of the loops around the call, which its body's scalar parameter
   holds on each trip. *)
let scalar_argument k s =
  (match arg k with Call c -> c.precompile | _ -> false)
  && match addrspace s with Some Dtype.Alu -> true | _ -> false

(* The walk of a kernel graph enters no kernel, and enters a loop's linear of
   calls, which copies it makes as each trip starts precede: a linear of a
   precompiled call is the call's body, which the walk skips. *)
let in_graph u =
  match (op u, arg u) with Op.Sink, Kernel _ -> false | _ -> true

(* A loop around a call: a range's end, or a back edge. *)
let is_loop k = op k = Op.End || op k = Op.Backedge

(* Whether [r] is a serial loop's range, and whether [u] moves with one. *)
let loop_range r = Axis_type.equal (axis_type r) Axis_type.Loop
let moves_with_loop u = List.exists loop_range (Nodes.to_list (ranges u))

let split_after after =
  let effects = List.tl (src after) in
  let kernels, rest =
    List.partition (fun s -> op s = Op.Call || is_loop s) effects
  in
  let deps, rest = List.partition (fun s -> op s = Op.After) rest in
  (match List.find_opt (fun s -> op s <> Op.Store) rest with
  | Some s ->
      invalid_arg
        (Format.asprintf
           "an after orders a call, a loop, a store or an after, not %a" Op.pp
           (op s))
  | None -> ());
  (kernels, deps)

(* The calls a kernel runs: itself, or the calls a loop runs each trip, its
   call or a linear of calls. *)
let calls_of k =
  if not (is_loop k) then [ k ]
  else
    let b = nth k 0 in
    let calls = if op b = Op.Linear then src b else [ b ] in
    if List.exists (fun c -> op c <> Op.Call) calls then
      invalid_arg "a loop of a kernel loops over a call";
    calls

(* [k] running [calls] in place of its own. *)
let with_calls k calls =
  if not (is_loop k) then List.hd calls
  else
    let b = match calls with [ c ] -> c | cs -> v Op.Linear ~src:cs in
    replace k ~src:(b :: List.tl (src k))

(* The positions of the call [c]'s arguments its body writes, or [None] where
   the body does not show them ({!Shape.written_params}). *)
let written_positions c = Shape.written_params (body c)

(* The positions of [c]'s arguments that hold the state [s]. *)
let positions c s =
  List.concat
    (List.mapi
       (fun i a ->
         if (not (scalar_argument c a)) && fst (value_state a) == s then [ i ]
         else [])
       (List.tl (src c)))

(* Whether the call [c] reads the state [s] through an argument its body does
   not write while it writes [s] through another: as [memmove] does, it reads
   [s] as it was before the call. [None] where its body does not show its
   writes and it takes [s] twice: it has no such reading. *)
let reads_written c s =
  match positions c s with
  | [] | [ _ ] -> Some false
  | at ->
      Option.map
        (fun w ->
          List.exists (fun i -> List.mem i w) at
          && List.exists (fun i -> not (List.mem i w)) at)
        (written_positions c)

(* [reads_written c s] for a call [c] that writes [s]. *)
let aliases c s =
  match reads_written c s with
  | Some b -> b
  | None ->
      invalid_arg
        (Format.asprintf
           "a call whose writes do not show takes %a twice: pass it distinct \
            storage"
           pp (buf_uop s))

(* Copies before overwrites

   A kernel that reads a state and waits, through a value computed after it,
   for a kernel that overwrites it, as [b + (a * 3)] does where [b] read [a]
   before [a] was assigned, runs neither before nor after the overwrite. It
   reads instead a copy of the state into storage of its own, which a kernel
   makes before the overwrite:

     a --copy--> a'          a' = a as it was
     a --assign--> a         the overwrite, after the copy reads a
     a', a*3 --reader-->     the reader reads a' *)

(* The copies made, for tests. *)
let made = Atomic.make 0
let copies () = Atomic.get made

(* The next free slot of the graph's storage. *)
let next_slot sink =
  let slot u =
    match (op u, arg u) with
    | (Op.Alloc | Op.Buffer | Op.Param), Param p -> p.slot
    | _ -> -1
  in
  let slots = List.map slot (toposort ~calls:Skip sink) in
  1 + List.fold_left max (-1) slots

(* New storage of slot [slot], starting at [start], and the kernel that copies
   the state [s] into it. *)
let copy_call ~slot ~start:(align, phase) s =
  let buf = buf_uop s in
  (* Storage of no device, a body's parameter, is copied on the device its
     call's arguments are on, as call-local storage is made. *)
  let n = max_numel buf in
  let arg = param_arg ~slot ?device:(device buf) ~phase ~align (dtype buf) in
  let dst = v Op.Alloc ~src:[ size_src (Some n) ] ~arg:(Param arg) in
  let r = range ~axis_type:Weak (Int n) [ 0 ] in
  (* The kernel's parameters name no device: its call's arguments do. *)
  let param u slot =
    let align, phase = Shape.storage_phase u in
    Shape.param ?shape:(shape_opt u) ~phase ~align slot (dtype u)
  in
  let into = param dst 0 and from = param buf 1 in
  let body =
    sink ~kernel:(kernel_info ())
      [ end_ (store (index into [ r ]) (index from [ r ])) [ r ] ]
  in
  (dst, call body [ dst; s ])

(* Where a copy of [s] starts for the reader [c] at argument [i]: as [s]'s
   storage does, which the reader's kernel reads it at; or where new storage
   starts, for a parameter that [s], read off its start from storage [c]
   writes, can be given only through this copy. *)
let copy_start c i s =
  let start = Shape.storage_phase (buf_uop s) in
  match Shape.param_start (body c) i with
  | Some param
    when (not (Shape.fits_start ~param start))
         && Shape.reads_storage_written c i ->
      Shape.allocated_start
  | _ -> start

(* [d], a source of a kernel, reading [c] where it reads the state [s]: its
   path to the state is the one {!value_state} follows. *)
let rec swap s c d =
  if d == s then c
  else if is_join d then replace d ~src:(swap s c (nth d 0) :: List.tl (src d))
  else
    match (op d, src d) with
    | (Op.After | Op.Buffer | Op.Alloc | Op.Param), _ | _, [] -> d
    | _, x :: rest -> replace d ~src:(swap s c x :: rest)

(* [sink] with each kernel of [hazards], each a kernel, a state it reads and
   whether the kernel writes over that state, reading a copy through the
   arguments its calls do not write: of the state, or of its own run where it
   reads through a view. A loop whose call writes the state through one
   argument and reads it through another copies as each trip starts, in the
   loop; any other kernel reads one copy, which a kernel of its own makes
   before it. *)
let copy_before_overwrites sink hazards =
  let slot = ref (next_slot sink) and copies = ref [] in
  let fresh ~start s =
    let c = copy_call ~slot:!slot ~start s in
    incr slot;
    Atomic.incr made;
    c
  in
  (* Copies made so far, a state having one per start its readers need. *)
  let find found s start =
    List.find_map
      (fun (s', start', x) -> if s' == s && start' = start then Some x else None)
      found
  in
  let copy ~start s =
    match find !copies s start with
    | Some c -> c
    | None ->
        let dst, c = fresh ~start s in
        let c = after dst [ c ] in
        copies := (s, start, c) :: !copies;
        c
  in
  (* [c] reading [state ()] for the state [s], and a view of it reading
     [run] of the view: the body names no run of a copy of the whole state. *)
  let reread s state run c =
    let w = Option.value (written_positions c) ~default:[] in
    let read i a =
      let r = unwrap_src a in
      if is_view r && state_of r == s then
        swap r (run ~start:(copy_start c i r) r) a
      else if state_of a == s then swap s (state ~start:(copy_start c i s)) a
      else a
    in
    replace c
      ~src:
        (body c
        :: List.mapi (fun i a -> if List.mem i w then a else read i a)
             (List.tl (src c)))
  in
  let rebuilt =
    List.fold_left
      (fun acc (k, s, overwrites) ->
        let k' = Option.value (List.assq_opt k acc) ~default:k in
        let calls = calls_of k' in
        let calls =
          if
            is_loop k'
            && overwrites
            && List.exists (fun c -> reads_written c s = Some true) calls
          then
            (* Each is copied once a trip, and only if a call reads it. *)
            let trip = ref [] in
            let per_trip ~start u =
              match find !trip u start with
              | Some (d, _) -> d
              | None ->
                  let d, c = fresh ~start u in
                  trip := (u, start, (d, c)) :: !trip;
                  d
            in
            let state ~start = per_trip ~start s in
            let calls = List.map (reread s state per_trip) calls in
            List.rev_map (fun (_, _, (_, c)) -> c) !trip @ calls
          else
            let state ~start = copy ~start s in
            List.map (reread s state copy) calls
        in
        (k, with_calls k' calls) :: List.remove_assq k acc)
      [] hazards
  in
  (* A rebuilt kernel may read another, directly or through the copy of a
     state: each is replaced wherever it appears, in rebuilt kernels and
     copies too. *)
  substitute ~calls:Skip ~pass:Fixed_point sink rebuilt

(* A kernel as the schedule runs it: a call of its storage, in a loop where it
   runs in one. *)
let scheduled rk =
  (* A call's argument is the storage it reads, or a view of it that moves with
     the loops the call runs in. *)
  let rec argument s =
    if Op.Set.mem (op s) Op.Set.movement && moves_with_loop s then replace s ~src:(argument (nth s 0) :: List.tl (src s))
    else buf_uop (unwrap_src s)
  in
  let call k =
    if op k <> Op.Call then invalid_arg "a scheduled kernel is a call";
    let args =
      List.filter_map
        (fun s ->
          if is_bound_var s then None
          else if scalar_argument k s then Some s
          else Some (argument s))
        (List.tl (src k))
    in
    replace k ~src:(body k :: args)
  in
  let calls = List.map call (calls_of rk) in
  (* A loop around a call, or a linear of calls, stays, a back edge's
     condition read from the storage it names. *)
  match op rk with
  | Op.End when List.exists loop_range (List.tl (src rk)) -> with_calls rk calls
  | Op.Backedge ->
      let rk = with_calls rk calls in
      replace rk ~src:[ nth rk 0; nth rk 1; buf_uop (unwrap_src (nth rk 2)) ]
  | _ -> List.hd calls

module Iset = Set.Make (Int)

(* Raised by the orders where the kernels form a cycle. *)
exception Cycle

(* The kernels in an order of the graph [children], [in_degree] counting the
   kernels each waits for and numbering them in the order the walk of the sink
   meets them, the toposort that numbers a kernel after the kernels whose
   storage it reads.

   Of the ready kernels, the first by number that frees storage and makes none
   runs; else the first that frees storage and on no device makes more than it
   frees, bytes counted unrounded; else the first by number. The storage a
   kernel writes, [written], lives from the first kernel that takes it to the
   last ({!Memory.taken}); other storage, which only kernels read, lives
   throughout. So where the walk reaches a value's readers right after it, the
   value is made just before them, a closing that slices a partial for each
   device sums each slice before it makes the next, and the last reader of a
   gathered value runs before the kernels that make more. A pick walks the
   ready kernels: [O(k²)] for [k] kernels at worst. *)
let storage_order ~entry ~written ~children in_degree =
  let kernels = Array.of_list (List.map fst (Ordered.bindings in_degree)) in
  let count = Array.length kernels in
  let number = Tbl.create count in
  Array.iteri (fun i k -> Tbl.replace number k i) kernels;
  let entries = Array.map entry kernels in
  let made_here b = Tbl.mem written b && not (on_disk b) in
  let storage =
    Array.map (fun e -> List.filter made_here (Memory.taken e)) entries
  in
  (* The kernels left to take each storage, and the storage made so far. *)
  let users = Tbl.create count and made = Tbl.create count in
  Array.iter
    (List.iter (fun b ->
         Tbl.replace users b (1 + Option.value (Tbl.find_opt users b) ~default:0)))
    storage;
  let bytes b = max_numel b * element_size b in
  let net = Hashtbl.create 4 in
  (* Whether kernel [i] frees storage and, where [~makes] is false, makes
     none, else makes on no device more than it frees. *)
  let frees ~makes i =
    Hashtbl.reset net;
    let add b n =
      let d = Option.value (device b) ~default:"" in
      Hashtbl.replace net d (n + Option.value (Hashtbl.find_opt net d) ~default:0)
    in
    let freed = ref false and making = ref false in
    List.iter
      (fun b ->
        if not (Tbl.mem made b) then begin
          add b (bytes b);
          making := true
        end
        else if Tbl.find users b = 1 then begin
          add b (-bytes b);
          freed := true
        end)
      storage.(i);
    !freed
    && (if makes then Hashtbl.fold (fun _ n ok -> ok && n <= 0) net true
        else not !making)
  in
  let ready = ref Iset.empty in
  Array.iteri
    (fun i k ->
      if Ordered.find_opt in_degree k = Some 0 then ready := Iset.add i !ready)
    kernels;
  (* A kernel holds what it makes beside what it reads: one that only frees
     runs before one that makes storage too. *)
  let pick () =
    let seq = Iset.to_seq !ready in
    match Seq.find (frees ~makes:false) seq with
    | Some i -> i
    | None -> (
        match Seq.find (frees ~makes:true) seq with
        | Some i -> i
        | None -> Iset.min_elt !ready)
  in
  let linearized = ref [] in
  while not (Iset.is_empty !ready) do
    let i = pick () in
    ready := Iset.remove i !ready;
    List.iter
      (fun b ->
        Tbl.replace made b ();
        Tbl.replace users b (Tbl.find users b - 1))
      storage.(i);
    linearized := entries.(i) :: !linearized;
    List.iter
      (fun x ->
        let d = Option.get (Ordered.find_opt in_degree x) - 1 in
        Ordered.replace in_degree x d;
        if d = 0 then ready := Iset.add (Tbl.find number x) !ready)
      (List.rev (Option.value (Tbl.find_opt children kernels.(i)) ~default:[]))
  done;
  if List.exists (fun (_, d) -> d <> 0) (Ordered.bindings in_degree) then
    raise Cycle;
  v Op.Linear ~src:(List.rev !linearized)

let linearize ~entry ~children sched_sink in_degree =
  (* The storage the kernels write. *)
  let written = Tbl.create 16 in
  List.iter
    (fun u ->
      if op u = Op.After && not (is_join u) then
        Tbl.replace written (buf_uop u) ())
    (toposort ~calls:Skip ~gate:in_graph sched_sink);
  storage_order ~entry ~written ~children in_degree

(* A read that a write over its state must follow: the kernel [reader] reads
   [state]. *)
type war = { reader : t; state : t }

let same_war a b = a.reader == b.reader && a.state == b.state

(* What a copy of the state [s] costs: storage that cannot be copied last,
   then the bytes, none for a state copied already. *)
let copy_cost copied s =
  let b = buf_uop s in
  let copyable =
    match device b with Some _ -> not (on_disk b) | None -> false
  in
  ( (if copyable then 0 else 1),
    if List.memq s copied then 0 else max_numel b * element_size b )

(* The reads to copy so that the kernels [order], related by [edges], each an
   edge from a kernel to one that must run after it through the read it orders
   if any, run in some order. *)
let cuts ~copies ~order edges =
  let deg = Tbl.create 64 and finished = Tbl.create 64 in
  let children = Tbl.create 64 and parents = Tbl.create 64 in
  let push tbl key x =
    Tbl.replace tbl key (x :: Option.value (Tbl.find_opt tbl key) ~default:[])
  in
  List.iter (fun k -> Tbl.replace deg k 0) order;
  List.iter
    (fun (t, k, war) ->
      push children t (k, war);
      push parents k (t, war);
      Tbl.replace deg k (Tbl.find deg k + 1))
    edges;
  let queue = Queue.create () in
  List.iter (fun k -> if Tbl.find deg k = 0 then Queue.push k queue) order;
  let cut = ref [] in
  let is_cut = function
    | Some w -> List.exists (same_war w) !cut
    | None -> false
  in
  let release k =
    let d = Tbl.find deg k - 1 in
    Tbl.replace deg k d;
    if d = 0 then Queue.push k queue
  in
  let rec run () =
    while not (Queue.is_empty queue) do
      let k = Queue.pop queue in
      Tbl.replace finished k ();
      List.iter
        (fun (x, war) -> if not (is_cut war) then release x)
        (Option.value (Tbl.find_opt children k) ~default:[])
    done;
    match List.find_opt (fun k -> not (Tbl.mem finished k)) order with
    | None -> List.rev !cut
    | Some _ when not copies -> invalid_arg "cycle detected in assign graph"
    | Some k ->
        (* Walk back to a kernel met twice: the reads between are a cycle. *)
        let met = Tbl.create 16 in
        let rec back path k =
          match Tbl.find_opt met k with
          | Some i -> List.filteri (fun j _ -> j >= i) (List.rev path)
          | None ->
              Tbl.replace met k (List.length path);
              let p, war =
                List.find
                  (fun (p, war) -> (not (Tbl.mem finished p)) && not (is_cut war))
                  (Tbl.find parents k)
              in
              back (war :: path) p
        in
        let reads = List.filter_map Fun.id (back [] k) in
        if reads = [] then invalid_arg "cycle detected in assign graph";
        let copied = List.map (fun w -> w.state) !cut in
        let cheapest =
          List.fold_left
            (fun best w ->
              if Stdlib.compare (copy_cost copied w.state)
                   (copy_cost copied best.state) < 0
              then w
              else best)
            (List.hd reads) (List.tl reads)
        in
        cut := cheapest :: !cut;
        (* Its reader no longer waits for the overwrites it ran before. *)
        Tbl.iter
          (fun c ps ->
            if not (Tbl.mem finished c) then
              List.iter
                (fun (p, war) ->
                  match war with
                  | Some w when same_war w cheapest && not (Tbl.mem finished p) ->
                      release c
                  | _ -> ())
                ps)
          parents;
        run ()
  in
  run ()

(* [scheduled] tagging each call of a kernel of a write marked disjoint in
   [sink] with that write's number and the state's position: each position it
   writes of the state's storage, directly or through a view of it. A kernel
   that is a loop of calls takes no tag: its calls then join by their bytes
   alone. *)
let marked_entry nodes =
  let marks = Tbl.create 16 and count = ref 0 in
  let push tbl k x =
    Tbl.replace tbl k (x :: Option.value (Tbl.find_opt tbl k) ~default:[])
  in
  List.iter
    (fun u ->
      if op u = Op.After && not (is_join u) then begin
        let kernels = fst (split_after u) in
        if is_disjoint u then begin
          incr count;
          let number = !count and state = whole (storage_base u) in
          List.iter (fun k -> push marks k (number, state)) kernels
        end
      end)
    nodes;
  fun rk ->
    let c = scheduled rk in
    match (op c, arg c, Tbl.find_opt marks rk) with
    | Op.Call, Call info, Some marks ->
        let written = Option.value (written_positions c) ~default:[] in
        let args = src_without_body c in
        let positions state =
          List.filter
            (fun i -> whole (storage_base (List.nth args i)) == state)
            written
        in
        let write =
          List.concat_map
            (fun (number, state) ->
              List.map (fun i -> (number, i)) (positions state))
            marks
        in
        replace c ~arg:(Call { info with write })
    | _ -> c

let tagged nodes =
  if List.exists is_disjoint nodes then marked_entry nodes else scheduled

(* The Afters of [nodes] over a storage's state, with their kernels. *)
let writes_of nodes =
  List.filter_map
    (fun u ->
      if op u = Op.After && not (is_join u) then Some (u, fst (split_after u))
      else None)
    nodes

(* Whether storage's start is known to be on a 16-byte boundary: the start a
   parameter is most often given, found without allocating. *)
let rec on_boundary a =
  match op a with
  | Op.After -> on_boundary (nth a 0)
  | Op.Param | Op.Buffer | Op.Alloc ->
      let p = param_arg_of a in
      p.align >= 16 && p.phase = 0
  | _ -> false

(* [sink] with each read of storage its call writes, off its parameter's start,
   given a copy at new storage's start: no kernel can take the storage there,
   whether or not a cut would copy it. *)
let copy_off_start_reads nodes sink =
  let off_start c i a =
    (not (on_boundary a))
    && (not (scalar_argument c a))
    && (not (is_bound_var a))
    && (not
          (Shape.fits_start ~param:Shape.allocated_start (Shape.storage_phase a)))
    && Option.fold ~none:false
         ~some:(fun param ->
           (not (Shape.fits_start ~param (Shape.storage_phase a)))
           && Shape.fits_start ~param Shape.allocated_start)
         (Shape.param_start (body c) i)
    && Shape.reads_storage_written c i
  in
  let rec reads c i acc = function
    | [] -> acc
    | a :: rest ->
        reads c (i + 1) (if off_start c i a then (c, a) :: acc else acc) rest
  in
  let found acc n = if op n = Op.Call then reads n 0 acc (List.tl (src n)) else acc in
  match List.fold_left found [] nodes with
  | [] -> None
  | off ->
      let writes = writes_of nodes in
      let kernel_of c =
        let has k = List.memq c (calls_of k) in
        List.find_map (fun (_, ks) -> List.find_opt has ks) writes
      in
      let add hazards (c, a) =
        match kernel_of c with
        | Some k when not (List.mem (k, state_of a, true) hazards) ->
            (k, state_of a, true) :: hazards
        | _ -> hazards
      in
      match List.fold_left add [] off with
      | [] -> None
      | hazards -> Some (copy_before_overwrites sink hazards)

let rec schedule ~copies sched_sink =
  let nodes = toposort ~calls:Skip ~gate:in_graph sched_sink in
  match if copies then copy_off_start_reads nodes sched_sink else None with
  | Some sink -> schedule ~copies sink
  | None -> kernels ~copies nodes sched_sink

(* The relation over chains

   Each storage's states form a chain ({!sequence}): an After over a state
   lists the kernels that write the next. A kernel reads the states its
   arguments name, except where its call writes: there it reads the state its
   write is over. It runs after the kernels that write the states it reads,
   after those that wrote the state its write is over, and before the kernels
   of the write over each state it reads. A kernel reading a state through one
   argument while writing over it through another reads it as it was: a
   cycle of one kernel. Where these orders form a cycle, one read on it reads
   a copy. *)
and kernels ~copies nodes sched_sink =
  let children = Tbl.create 64 and in_degree = Ordered.create () in
  let edges = ref [] in
  let add_child ?war t k =
    Tbl.replace children t
      (k :: Option.value (Tbl.find_opt children t) ~default:[]);
    edges := (t, k, war) :: !edges;
    Ordered.replace in_degree k (Option.get (Ordered.find_opt in_degree k) + 1)
  in
  (* Each state with the kernels of the write over it, the states a marked
     write is over, and each read of a state, last first. *)
  let writes = Tbl.create 64 and marked = ref [] and reads = ref [] in
  let own = Tbl.create 8 and listed = Tbl.create 16 in
  (* The kernels a reader of [x] waits for: the last writers of its state,
     and what the joins on the way wait for. *)
  let waited = Tbl.create 16 in
  (* The kernels that share a write with others. *)
  let shared =
    List.concat_map
      (fun u ->
        if op u <> Op.After || is_join u then []
        else
          let kernel s = op s = Op.Call || is_loop s in
          let count n s = if kernel s then n + 1 else n in
          if List.fold_left count 0 (src u) < 2 then []
          else List.filter kernel (src u))
      nodes
  in
  let rec waits x =
    match Tbl.find_opt waited x with
    | Some w -> w
    | None ->
        let st, deps = value_state x in
        let mine = Option.value (Tbl.find_opt own st) ~default:[] in
        let mine =
          if mine = [] && op st = Op.After then waits (nth st 0) else mine
        in
        let w = dedup_nodes (mine @ List.concat_map waits deps) in
        Tbl.replace waited x w;
        w
  in
  List.iter
    (fun u ->
      if op u = Op.After && not (is_join u) then begin
        let kernels, after_deps = split_after u in
        let storage = storage_base u in
        let seen k = Option.value (Tbl.find_opt listed k) ~default:[] in
        let fresh = List.filter (fun k -> not (List.memq storage (seen k))) kernels in
        let prev = fst (value_state (nth u 0)) in
        if fresh <> [] then begin
          (* Two writes over one state: a tensor graph's are refused when it
             is sequenced ({!Prepare.sequence}), a kernel graph given to
             {!create_schedule} has its own. *)
          (match Tbl.find_opt writes prev with
          | Some ks when not (List.equal ( == ) ks fresh) ->
              invalid_arg
                (Format.asprintf
                   "two writes over one state of the storage in slot %d: \
                    list one over the After of the other, or list both in \
                    one After"
                   (param_arg_of (buf_uop prev)).slot)
          | _ -> ());
          Tbl.replace writes prev fresh;
          Tbl.replace own u fresh;
          if is_disjoint u then marked := prev :: !marked
        end;
        List.iter
          (fun k ->
            let first = not (Tbl.mem listed k) in
            Tbl.replace listed k (storage :: seen k);
            if Ordered.find_opt in_degree k = None then
              Ordered.replace in_degree k 0;
            let read_args =
              if not first then []
              else
                List.concat_map
                  (fun c ->
                    (* A kernel alone in its writes reads, through what it
                       writes, the state its write is over, which it already
                       runs after: only a kernel sharing a write needs its
                       written arguments told apart. A view it stores through
                       names its storage, no state it reads. *)
                    let w =
                      lazy (Option.value (written_positions c) ~default:[])
                    in
                    let alone = not (List.memq k shared) in
                    let written i s =
                      (if alone then is_view (unwrap_src s) else true)
                      && List.mem i (Lazy.force w)
                    in
                    List.filteri
                      (fun i s -> not (scalar_argument c s || written i s))
                      (List.tl (src c)))
                  (calls_of k)
            in
            reads :=
              List.rev_append
                (List.map (fun s -> (k, state_of s)) read_args)
                !reads;
            (* Read after write, and write after write. *)
            let before =
              List.concat_map waits read_args
              @ List.concat_map waits after_deps
              @ (if List.memq k fresh && op prev = Op.After then waits prev
                 else [])
            in
            List.iter (fun t -> if t != k then add_child t k) (dedup_nodes before))
          kernels
      end)
    nodes;
  (* Write after read. *)
  List.iter
    (fun (k, s) ->
      match Tbl.find_opt writes s with
      | None -> ()
      | Some ks ->
          let war = { reader = k; state = s } in
          if List.memq k ks && List.exists (fun c -> aliases c s) (calls_of k)
          then add_child ~war k k;
          let exempt = List.memq k ks && List.memq s !marked in
          if not exempt then
            List.iter (fun t -> if t != k then add_child ~war k t) ks)
    (List.rev !reads);
  let order = List.rev in_degree.Ordered.keys in
  let entry = tagged nodes in
  match linearize ~entry ~children sched_sink in_degree with
  | linear -> linear
  | exception Cycle -> (
      match cuts ~copies ~order (List.rev !edges) with
      | [] -> invalid_arg "cycle detected in assign graph"
      | cut ->
          let overwrites w =
            match Tbl.find_opt writes w.state with
            | Some ks -> List.memq w.reader ks
            | None -> false
          in
          schedule ~copies:false
            (copy_before_overwrites sched_sink
               (List.map (fun w -> (w.reader, w.state, overwrites w)) cut)))

let create_schedule sched_sink = schedule ~copies:true sched_sink

(* A linear in a linear is inlined into it. Disjoint writes are numbered per
   schedule ({!tagged}), so each inlined linear's calls move past the numbers
   of what comes before it: its writes are no others', even where one schedule
   is inlined twice. *)
let flatten_linear () =
  let top c =
    match arg c with
    | Call { write; _ } ->
        List.fold_left (fun m (number, _) -> max m number) 0 write
    | _ -> 0
  in
  let moved by c =
    match arg c with
    | Call ({ write = _ :: _; _ } as info) ->
        let write = List.map (fun (number, p) -> (number + by, p)) info.write in
        replace c ~arg:(Call { info with write })
    | _ -> c
  in
  let highest = List.fold_left (fun m c -> max m (top c)) 0 in
  rule (Upat.op Op.Linear ~name:"lin" ~early_reject:[ Op.Linear ]) (fun m ->
      let lin = m "lin" in
      let own = List.filter (fun c -> op c <> Op.Linear) (src lin) in
      let used = ref (highest own) in
      let inlined c =
        if op c <> Op.Linear then [ c ]
        else begin
          let by = !used in
          used := by + highest (src c);
          List.map (moved by) (src c)
        end
      in
      Some (replace lin ~src:(List.concat_map inlined (src lin))))

let pm_flatten_linear = Pattern_matcher.v (fun () -> [ flatten_linear () ])

(* Parameters and call-local storage *)

let create_new_buffer args b =
  let device =
    match device b with
    | Some d -> d
    | None -> (
        match List.find_map device args with
        | Some d -> d
        | None -> invalid_arg "call-local storage needs a device")
  in
  new_buffer device (max_numel b) (dtype b)

let pm_post_sched_cache =
  Pattern_matcher.v
    (fun () -> [
      (* Positional arguments are resolved outside kernel bodies; free variables
         have slot -1. *)
      rule_ctx (Upat.op Op.Param ~name:"x") (fun args m ->
          let slot = (param_arg_of (m "x")).slot in
          if slot >= 0 then Some (List.nth args slot) else None);
      (* Call-local storage is bound to new buffers for this invocation. *)
      rule_ctx (Upat.op Op.Alloc ~name:"b") (fun ctx m ->
          Some (create_new_buffer ctx (m "b")));
    ])

(* A kernel's scalar parameters that its scope binds to values, such as a loop's
   trip, stay parameters: each takes a slot of the kernel past its arguments,
   in the order the kernel's body reaches them, and the call passes the value
   there. The kernel's program then depends on neither the value, nor the loop
   it comes from, nor the order the scope met it, and a launch computes the
   value from the loop's ranges. *)
let pass_values values call =
  let args = src_without_body call in
  let held =
    List.filter
      (fun v ->
        op v = Op.Param
        && addrspace v = Some Dtype.Alu
        &&
        let p = param_arg_of v in
        p.name = None && List.mem_assoc p.slot values)
      (toposort ~calls:Skip (body call))
  in
  let n = List.length args in
  let slot j v = replace v ~arg:(Param { (param_arg_of v) with slot = n + j }) in
  let slotted = List.mapi (fun j v -> (v, slot j v)) held in
  replace call
    ~src:
      (substitute ~calls:Skip ~pass:Once (body call) slotted
      :: args
      @ List.map (fun v -> List.assoc (param_arg_of v).slot values) held)

(* Nested linear calls are lexical scopes: their positional parameters shadow
   the enclosing scope, while calls without scalar arguments inherit it. A
   scalar argument that is a variable is bound into each kernel that reads it in
   its slot, by its name, unbound: a run binds it. A variable bound to a value
   of a name in [several], which the schedule binds to several values, and any
   other scalar argument, is a value its kernels are passed ({!pass_values}):
   each call passes its own. A named variable keeps its slot: a custom kernel's
   takes its own. *)
let rec resolve_linear_call ~several ?(outer_binds = []) linear_call =
  let args = List.tl (src linear_call) in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx:args (body linear_call)
      (After_sources pm_post_sched_cache)
  in
  let binds =
    let local =
      List.concat
        (List.mapi
           (fun i x ->
             if addrspace x <> Some Dtype.Alu then []
             else if is_bound_var x && List.mem (expr x) several then
               [ (i, x) ]
             else if is_variable x then [ (i, unbound x) ]
             else [ (i, x) ])
           args)
    in
    local @ List.filter (fun (i, _) -> not (List.mem_assoc i local)) outer_binds
  in
  let params, values =
    List.partition (fun (_, x) -> op x = Op.Param && not (is_bound_var x)) binds
  in
  let apply_binds si =
    match (op si, src si) with
    | Op.Call, b :: _ when op b = Op.Linear ->
        resolve_linear_call ~several ~outer_binds:binds si
    (* Compiled parameters already have their slots in the program. *)
    | Op.Call, b :: _ when op b = Op.Program -> si
    | _ ->
        let subs =
          List.concat_map
            (fun s ->
              List.filter_map
                (fun v ->
                  let p = param_arg_of v in
                  if p.name <> None then None
                  else
                    Option.map (fun x -> (v, x)) (List.assoc_opt p.slot params))
                (variables s))
            (src si)
        in
        let si =
          replace si
            ~src:
              (List.map
                 (fun s -> substitute ~calls:Skip ~pass:Fixed_point s subs)
                 (src si))
        in
        if op si = Op.Call && op (body si) = Op.Sink then pass_values values si
        else si
  in
  replace linear ~src:(List.map apply_binds (src linear))

(* Its context is the names the schedule binds to several values. *)
let pm_resolve_linear_call =
  Pattern_matcher.v (fun () -> [
      rule_ctx
        (Upat.op Op.Call ~name:"linear_call" ~allow_any_len:true
           ~src:[ Upat.op Op.Linear ])
        (fun several m ->
          Some (resolve_linear_call ~several (m "linear_call")));
      flatten_linear ();
    ])

(* The names that the bound variables of [u] bind to several values. *)
let bound_several u =
  let values = Hashtbl.create 8 in
  List.iter
    (fun v ->
      if is_bound_var v then
        let x = Dtype.Value.to_int (Option.get (param_arg_of v).bound) in
        let name = expr v in
        let seen = Option.value (Hashtbl.find_opt values name) ~default:[] in
        if not (List.mem x seen) then Hashtbl.replace values name (x :: seen))
    (toposort ~calls:Enter u);
  Hashtbl.fold
    (fun name xs acc -> if List.length xs > 1 then name :: acc else acc)
    values []

(* Scheduling calls *)

let schedule_cache : t Helpers.Memo.t = Helpers.Memo.create ()

(* [in_order fn] is the pairs that take each range and each storage of [fn] to a
   placeholder numbered by its order in it. Whoever makes a loop numbers its
   range, and whoever makes storage, such as a staged loop's carry or
   call-local storage that inlining renames, takes its slot, from a counter
   whose value depends on what the process did before, such as making a
   schedule or reading one back: the key is [fn] with its placeholders, and so
   is the schedule kept, in memory and on disk, so that one schedule serves
   every numbering of a function, each given back its own. A placeholder is no
   node a schedule makes: ranges are numbered below [0], storage below every
   slot of [fn] and below [-1], the slot of a variable. *)
let in_order fn =
  let nodes = toposort ~calls:Enter fn in
  let slot u = match arg u with Param p -> Some p.slot | _ -> None in
  let lowest =
    List.fold_left
      (fun m u -> min m (Option.value ~default:0 (slot u)))
      (-1) nodes
  in
  let ranges = List.filter (fun u -> op u = Op.Range) nodes in
  let storage =
    List.filter
      (fun u ->
        (op u = Op.Buffer || op u = Op.Alloc)
        && match slot u with Some s -> s >= 0 | None -> false)
      nodes
  in
  List.mapi
    (fun k r ->
      ( r,
        replace r ~arg:(Range { axis_id = [ -1 - k ]; axis_type = axis_type r })
      ))
    ranges
  @ List.mapi
      (fun k u ->
        let p = param_arg_of u in
        (u, replace u ~arg:(Param { p with slot = lowest - 1 - k })))
      storage

type numbering = {
  key : string;
  place : t -> t;
  restore : t -> t;
  position : t -> int;
}

(* The key of a schedule, in memory and, with the setting scache at 2 or more,
   on disk: [fn_key], the key of the function with its placeholders
   ([in_order]), every setting and variable that shapes what compilation makes
   ([Setting.shaping]), and the digest of this library's sources, of which a
   schedule is a function. *)
let schedule_key fn_key =
  String.concat "\n"
    ([ Source_digest.digest; fn_key ]
    @ List.map (fun (k, v) -> k ^ "=" ^ v) (Setting.shaping ()))

(* [fn]'s numbering by its own order ({!numbering}): [fn] with its placeholders
   is keyed without being built, so that a hit builds no graph but what it
   reads. A placeholder keeps its node's sources, so where a placed node reaches
   another, such as a range whose bound reads a range, [key_with], which would
   place that one too, is not the key of the placed graph: it is built. *)
let numbered fn =
  let placed = in_order fn in
  let place g = substitute ~calls:Enter ~pass:Once g placed in
  let placeholder = Tbl.create 64 and rank = Tbl.create 64 in
  List.iteri
    (fun i (u, p) ->
      Tbl.replace placeholder u (arg p);
      Tbl.replace rank u i)
    placed;
  let reaches_placed (u, _) =
    List.exists
      (fun s -> s != u && Tbl.mem placeholder s)
      (toposort ~calls:Enter u)
  in
  let key =
    if List.exists reaches_placed placed then Ops.key (place fn)
    else key_with ~arg:(Tbl.find_opt placeholder) fn
  in
  {
    key;
    place;
    restore =
      (fun g ->
        substitute ~calls:Enter ~pass:Once g
          (List.map (fun (u, p) -> (p, u)) placed));
    position = (fun u -> Option.value ~default:(-1) (Tbl.find_opt rank u));
  }

let lower_sink_to_linear call =
  let fn = body call in
  let precompile = match arg call with Call c -> c.precompile | _ -> false in
  match (op fn, arg fn) with
  | Op.Sink, Kernel _ -> None
  | Op.Sink, _ when precompile ->
      let start = Unix.gettimeofday () in
      let numbering = numbered fn in
      let cache_key =
        Digest.BLAKE256.string (schedule_key numbering.key)
      in
      (* Whether neither memory nor disk had the schedule. *)
      let made = ref false in
      (* Its effects are sequenced before the check of its form, as at the
         top level, so that a mistake of its maker is refused as there. *)
      let make () =
        made := true;
        if setting Setting.spec <> 0 then (
          ignore (Prepare.sequence fn);
          Spec.type_verify ~calls:Skip Spec.tensor fn);
        numbering.place
          (schedule ~copies:true
             (Rangeify.get_kernel_graph (Prepare.prepare_rangeify fn)))
      in
      let read_or_make () =
        if setting Setting.scache < 2 then make ()
        else
          fst
            (Graph.cached ~table:"schedule_cache" ~key:cache_key
               ~valid:(fun l -> op l = Op.Linear)
               make)
      in
      let linear =
        if setting Setting.scache < 1 then read_or_make ()
        else Helpers.Memo.get schedule_cache cache_key read_or_make
      in
      let n = List.length (src linear) and debug = setting Setting.debug in
      if (debug >= 1 && n > 1) || debug >= 3 then
        Format.printf "scheduled %5d kernels in %8.2f ms | %s %s@." n
          ((Unix.gettimeofday () -. start) *. 1000.)
          (if !made then "CACHE MISS" else " cache hit")
          (String.sub (Digest.BLAKE256.to_hex cache_key) 0 8);
      let linear = numbering.restore linear in
      Some (replace call ~src:(linear :: List.tl (src call)))
  | _ -> None

let pm_schedule =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.op Op.Call ~name:"call") (fun () m ->
          lower_sink_to_linear (m "call"));
    ])

(* Copies *)

(* A kernel's parameters name no device: its call's arguments do. *)
let assert_all_same_devices call =
  let devices =
    List.fold_left
      (fun acc x ->
        match device x with
        | Some d when not (List.exists (String.equal d) acc) -> acc @ [ d ]
        | Some _ | None -> acc)
      [] (List.tl (src call))
  in
  if List.length devices >= 2 then
    invalid_arg
      (Format.asprintf "all buffers must be on the same device: %a"
         (Format.pp_print_list
            ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
            pp_device)
         devices)

(* A kernel whose arguments [dst] and [src] lie on two devices, or [dst] on a
   disk, is a copy. *)
let is_copy dst src =
  (not (Option.equal String.equal (device dst) (device src))) || on_disk dst

(* The element a copy kernel's index [i] of its range [r] starts at: [0] for
   [r], [c] for [r + c]. *)
let run_start r i =
  if i == r then Some 0
  else
    match (op i, Ops.src i) with
    | Op.Add, [ a; c ] when a == r -> (
        match arg c with Const (`Int z) -> Some (Bigint.to_int z) | _ -> None)
    | _ -> None

(* The copy kernel [call] of the [n] elements of [src] from [s] into those of
   [dst] from [d], a copy between the call's arguments where each run is its
   argument whole, and between views of them otherwise, as a copy into a run of
   storage is ({!Prepare}). *)
let copy_runs call dst src ~n ~d ~s =
  let args = Array.of_list (List.tl (Ops.src call)) in
  let flat p = ndim args.((param_arg_of p).slot) = 1 in
  let whole p start = start = 0 && max_numel p = n in
  let arg p = args.((param_arg_of p).slot) in
  if not (is_copy (arg dst) (arg src)) then None
  else if not ((whole dst d || flat dst) && (whole src s || flat src)) then
    None
  else
    let run p start =
      if whole p start then p
      else begin
        let k = (param_arg_of p).slot in
        args.(k) <- shrink args.(k) [ Some (Int start, Int (start + n)) ];
        replace p ~src:[ Ops.const ~dtype:(dtype (nth p 0)) (`Int (Bigint.of_int n)) ]
      end
    in
    let dst = run dst d and src = run src s in
    Some (replace call ~src:(store dst src :: Array.to_list args))

(* The copy kernels are code for copy engines. *)
(* A kernel of a variable is no copy, though the variable has no device. *)
(* Copy kernels become bulk stores. A copy of one element indexes each side at
   its element. *)
let pm_copy_kernel =
  let const name = Upat.op Op.Const ~name in
  let param name = Upat.op Op.Param ~name in
  let copy_call body =
    Upat.op Op.Call ~name:"call" ~allow_any_len:true ~src:[ Upat.sink [ body ] ]
  in
  Pattern_matcher.v
    (fun () -> [
      rule
        (copy_call
           (Upat.store
              (Upat.index (param "dst") [ const "di" ])
              [ Upat.index (param "src") [ const "si" ] ]))
        (fun m ->
          match (arg (m "di"), arg (m "si")) with
          | Const (`Int d), Const (`Int s) ->
              copy_runs (m "call") (m "dst") (m "src") ~n:1
                ~d:(Bigint.to_int d) ~s:(Bigint.to_int s)
          | _ -> None);
      rule
        (copy_call
           (Upat.end_
              (Upat.store
                 (Upat.index (param "dst") [ var "di" ])
                 [ Upat.index (param "src") [ var "si" ] ])
              [ Upat.op Op.Range ~name:"r" ]))
        (fun m ->
          let r = m "r" in
          match
            (run_start r (m "di"), run_start r (m "si"), arg (nth r 0))
          with
          | Some d, Some s, Const (`Int n) ->
              copy_runs (m "call") (m "dst") (m "src") ~n:(Bigint.to_int n) ~d
                ~s
          | _ -> None);
    ])

(* A kernel between devices or onto a disk is a copy when its simplified body
   is one; any other kernel goes to codegen as it is. *)
let simplify_copy_kernel call ast dst src =
  if is_virtual dst || is_virtual src || not (is_copy dst src) then None
  else
    let sink =
      graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Tbl.create 8) ast
        (After_sources
           (Pattern_matcher.concat
              [
                with_ctx Symbolic.sym;
                with_ctx Prepare.pm_mops;
                with_ctx Simplify.pm_flatten_range;
                Simplify.pm_simplify_ranges;
              ]))
    in
    Pattern_matcher.rewrite pm_copy_kernel ()
      (replace call ~src:(sink :: List.tl (Ops.src call)))

let pm_copy_from_store =
  Pattern_matcher.concat
    [
      Pattern_matcher.v (fun () -> [
          rule
            (Upat.op Op.Call ~name:"call"
               ~src:[ Upat.op Op.Sink ~name:"ast"; var "dst"; var "src" ])
            (fun m ->
              simplify_copy_kernel (m "call") (m "ast") (m "dst") (m "src"));
        ]);
      pm_copy_kernel;
      Pattern_matcher.v (fun () -> [
          (* Any other kernel stays on one device. *)
          rule
            (Upat.op Op.Call ~name:"call" ~allow_any_len:true
               ~src:[ Upat.op Op.Sink ])
            (fun m ->
              assert_all_same_devices (m "call");
              None);
        ]);
    ]

(* Callify: the tensor graph becomes a call, with its state scoped *)

type callify_ctx = {
  mutable replacements : t list; (* Last first. *)
  allocs : t Ordered.t;
  views : unit Tbl.t;
  dests : t Lazy.t Tbl.t;
      (* Each run of a state a store's destination views, with its
         parameter. *)
  mutable stores : t list; (* Last first. *)
  written : unit Tbl.t; (* The buffers the call's stores write. *)
  params : t Tbl.t; (* The parameter of each storage. *)
}

let callify_ctx () =
  {
    replacements = [];
    allocs = Ordered.create ();
    views = Tbl.create 8;
    dests = Tbl.create 8;
    stores = [];
    written = Tbl.create 8;
    params = Tbl.create 8;
  }

let all_int shape =
  List.for_all (function Int _ -> true | Sym _ -> false) shape

let replace_input_buffer ctx b =
  ctx.replacements <- b :: ctx.replacements;
  param_like b (List.length ctx.replacements - 1)

(* The parameter for the storage [b] of the call, one per storage. *)
let storage_param ctx b =
  match Tbl.find_opt ctx.params b with
  | Some p -> p
  | None ->
      let p = replace_input_buffer ctx b in
      Tbl.replace ctx.params b p;
      p

(* The parameter for the view [b] of storage: its own slot, for the kernels'
   programs to be one wherever the view starts, and a second source naming
   its storage's parameter, so that every use names one storage whichever view
   it goes through. *)
let view_param ctx b =
  let whole = storage_param ctx (storage_base b) in
  let p = replace_input_buffer ctx b in
  replace p ~src:(src p @ [ whole ])

(* The run of a buffer the movements and bitcasts [src] select, as [c] takes
   it: a shrink of the buffer, whatever state it was read at, or of [over], a
   state of the buffer. *)
let run_of ?over c src =
  if not (all_int (shape c)) then None
  else
    (* A store's run may start at bound variables, which its call knows: its
       kernel then knows the start modulo what they move it by. *)
    let at = function
      | Int _ -> true
      | Sym _ -> op c = Op.Store
    in
    match Prepare.contiguous_view src with
    | Some (buf, offset) when op buf = Op.Buffer && at offset ->
        let n = max_numel src * element_size src / element_size buf in
        Some
          (bitcast
             (shrink
                (Option.value over ~default:buf)
                [ Some (offset, Sint.(offset + Int n)) ])
             (dtype src))
    | _ -> None

(* What the movements and bitcasts [a] read: storage or a state of it. *)
let rec named a =
  let b = base a in
  if op b = Op.Bitcast then named (nth b 0) else b

(* [a], movements and bitcasts of storage or of states of it, with the states
   gone: the same elements, named by the storage alone. *)
let rec without_states a =
  if op a = Op.After then without_states (nth a 0)
  else if Op.Set.mem (op a) Op.Set.movement || op a = Op.Bitcast then
    replace a ~src:(without_states (nth a 0) :: List.tl (src a))
  else a

(* Movements and bitcasts of a buffer are a view of it when they select a
   contiguous range of its elements. *)
let view_of ctx c src =
  (* A store stores into its storage as the store runs, whichever state its
     destination names: the states order the store, not its elements. *)
  let state = named src in
  let src = if op c = Op.Store then without_states src else src in
  let buf = named src in
  (* A store's value that reads its buffer keeps the buffer's identity, by
     which the kernel split stages it before the store. Its kernel reads
     other storage at a stored state, an after, which ends the walk: each
     store walks its own kernel's values, and a chain of outputs, each
     reading the last, costs its length. *)
  let reads_target () =
    let walked u = op u <> Op.After || storage_base u == buf in
    op c = Op.Store && reaches ~gate:walked ~calls:Skip (nth c 1) buf
  in
  (* A copy or a stage of a buffer the call writes reads the state it was
     taken of, which a view, read when its reader runs, would not hold. *)
  let overwritten () =
    op c <> Op.Store
    && match ctx with Some ctx -> Tbl.mem ctx.written buf | None -> false
  in
  (* A store's view is a run of the state its destination names: stores of
     one value into one run over two states stay two stores, which the run of
     the storage alone would make one node. Its parameter stands for the run
     of the storage. A state of another shape than its storage's has no
     run. *)
  let over =
    if op c <> Op.Store || op state <> Op.After then Some buf
    else if
      List.length (shape state) = 1 && max_numel state = max_numel buf
    then Some state
    else None
  in
  if op buf <> Op.Buffer || reads_target () || overwritten () then None
  else
    let view = Option.bind over (fun over -> run_of ~over c src) in
    match (run_of c src, view) with
    | None, _ | _, None -> None
    | Some run, Some view ->
        (* A read of the same run of the state is no view of the storage:
           the destination alone takes the parameter. *)
        Option.iter
          (fun ctx ->
            if view == run then Tbl.replace ctx.views view ()
            else Tbl.replace ctx.dests view (lazy (view_param ctx run)))
          ctx;
        let view = reshape view (shape c) in
        if op c = Op.Copy || op c = Op.Store then
          Some (replace c ~src:(view :: List.tl (Ops.src c)))
        else Some view

let contiguous_mops_to_view c src = view_of None c src

let is_store_after u =
  op u = Op.After
  && (op (base (nth u 0)) <> Op.Alloc || op (nth u 1) = Op.Store)

(* Scheduling rewrites belong in Prepare; only storage and interface
   normalisation belongs here. *)
let pm_callify_ctx_collect =
  Pattern_matcher.v
    (fun () -> [
      (* Movements and bitcasts of a buffer that collapse to a contiguous range
         are a shrink of it. *)
      rule_ctx
        (Upat.v
           ~op:(ops Op.[ Copy; Stage ])
           ~name:"c" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(Op.Set.union Op.Set.movement (ops [ Op.Bitcast ]))
                 ~name:"src" ();
             ]
           ())
        (fun ctx m -> view_of (Some ctx) (m "c") (m "src"));
      (* So do those a store stores into: stores of one shape into runs of
         one buffer at other offsets are then one program. *)
      rule_ctx
        (Upat.op Op.Store ~name:"c" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(Op.Set.union Op.Set.movement (ops [ Op.Bitcast ]))
                 ~name:"src" ();
               Upat.wild;
             ])
        (fun ctx m -> view_of (Some ctx) (m "c") (m "src"));
      (* Effects are collected after their sources are rewritten, without
         entering call bodies. *)
      rule_ctx (Upat.op Op.After ~name:"u") (fun ctx m ->
          let u = m "u" in
          if is_store_after u then ctx.stores <- u :: ctx.stores;
          None);
    ])

(* Call-local storage gets slots of its own scope, so that equal calls hash
   alike for the schedule cache. Fresh slots count up from 0; negative slots are
   already the scope's own. *)
let canonicalize_alloc ctx b =
  let p = param_arg_of b in
  if p.slot >= 0 && Ordered.find_opt ctx.allocs b = None then
    Ordered.replace ctx.allocs b
      (replace b
         ~arg:(Param { p with slot = -1 - List.length ctx.allocs.keys }));
  Ordered.find_opt ctx.allocs b

(* Bound variables become parameters without their value, so that different
   values hit the same cache entry, and a name bound to several values takes a
   parameter for each. *)
let pm_pass_bound =
  Pattern_matcher.v (fun () -> [
      rule_ctx (Upat.op Op.Param ~name:"b") (fun ctx m ->
          let b = m "b" in
          if is_bound_var b then Some (replace_input_buffer ctx b) else None);
    ])

(* A call passes the variables its body binds, as the schedule passes its own
   ({!pm_replace_buf}): each takes a nameless parameter in a slot past the
   call's arguments, and the call passes the bound variable there. A kernel
   the body makes then reads each value from a slot of its own, and bodies
   that differ in their values alone are one. *)
let rec canonicalize_call_body c =
  let ctx = callify_ctx () in
  ctx.replacements <- List.rev (List.tl (src c));
  let rules =
    if op (body c) = Op.Sink then pm_canonicalize_body
    else pm_canonicalize_alloc
  in
  let body =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx (body c)
      (Before_sources (Lazy.force rules))
  in
  replace c ~src:(body :: List.rev ctx.replacements)

and pm_canonicalize_alloc =
  lazy
    (Pattern_matcher.v
       (fun () -> [
         rule (Upat.op Op.Call ~name:"c") (fun m ->
             Some (canonicalize_call_body (m "c")));
         rule_ctx (Upat.op Op.Alloc ~name:"b") (fun ctx m ->
             canonicalize_alloc ctx (m "b"));
       ]))

(* A kernel the body calls, such as a custom kernel, reads the variables its
   code binds as the kernels the schedule splits from the body do: from the
   body's slots, which its scope resolves ({!resolve_linear_call}). *)
and pm_pass_kernel_bound =
  lazy
    (Pattern_matcher.v (fun () -> [
         rule_ctx (Upat.op Op.Call ~name:"c") (fun ctx m ->
             let c = m "c" in
             let b = body c in
             match (op b, arg b) with
             | Op.Sink, Kernel _ ->
                 let b =
                   graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx b
                     (Before_sources pm_pass_bound)
                 in
                 Some
                   (canonicalize_call_body
                      (replace c ~src:(b :: List.tl (src c))))
             | _ -> None);
       ]))

and pm_canonicalize_body =
  lazy
    (Pattern_matcher.concat
       [
         Lazy.force pm_pass_kernel_bound;
         Lazy.force pm_canonicalize_alloc;
         pm_pass_bound;
       ])

(* Forced as the module initialises, on one domain: the compilers' domains
   would race to force it first, and a lazy value that two domains force at
   once raises. *)
let pm_canonicalize_body = Lazy.force pm_canonicalize_body

let pm_replace_buf =
  Pattern_matcher.v
    (fun () -> [
      (* Global buffers become parameters, which normalises the cache key;
         variables are scalar parameters and do not match. *)
      rule_ctx (Upat.op Op.Buffer ~name:"b") (fun ctx m ->
          let b = m "b" in
          if addrspace b = Some Dtype.Global then Some (storage_param ctx b)
          else None);
      (* So do the views of buffers found above. *)
      rule_ctx
        (Upat.v ~op:(ops Op.[ Shrink; Bitcast ]) ~name:"b" ())
        (fun ctx m ->
          let b = m "b" in
          if Tbl.mem ctx.views b then Some (view_param ctx b) else None);
      (* And the runs of states stores store into. *)
      rule_ctx
        (Upat.op Op.Store ~name:"c" ~allow_any_len:true)
        (fun ctx m ->
          let c = m "c" in
          let d = nth c 0 in
          let v = if op d = Op.Reshape then nth d 0 else d in
          match Tbl.find_opt ctx.dests v with
          | None -> None
          | Some p ->
              let p = Lazy.force p in
              let d =
                if d == v then p else replace d ~src:(p :: List.tl (src d))
              in
              Some (replace c ~src:(d :: List.tl (src c))));
    ])

(* Reads of a loop's call, each trip

   [trip_reads sink] is [sink], a sequenced graph ({!Prepare.sequence}), with
   each loop's call reading, each trip, the values it computes from the state
   the loop writes, in every scope: [sink] and each body it calls, at any
   depth, a body that changes being sequenced first. A loop's call, the call
   of an {!Op.End} over {!Ops.Axis_type.Loop} ranges or of a back edge
   ({!Ops.backedge}), reads the state its write is over, the state its
   written arguments name, as each trip starts. An argument that reads that
   state as the state itself, as a reshape of it, or as a row the loop moves
   along is read when the call runs. Any other argument reading it is a value
   the call computes: the call becomes one of a body that computes each such
   value from its parameters and calls the original body on them, taking the
   state whole, and taking as they are the storage, rows and values of other
   states the values read. A body that is no kernel is inlined there
   ({!Ops.is_inline_call}). It raises [Invalid_argument] if such a value reads
   a value that moves with the loop and is no row of storage, or if an
   argument stores a value that moves with the loop. *)

let loop_row u =
  op u = Op.Shrink
  && has_buffer_identity ~after_ok:true (nth u 0)
  && moves_with_loop u

let taken u = has_buffer_identity ~after_ok:true u || loop_row u
let storage_ops = Op.Set.union Op.Set.defines (ops [ Op.After ])

(* Whether [u] reads a state [trip] holds, through values and no other
   state. *)
let reads_trip ~trip =
  let seen = Tbl.create 8 in
  let rec go n =
    match Tbl.find_opt seen n with
    | Some r -> r
    | None ->
        let r =
          (Op.Set.mem (op n) storage_ops && trip n)
          || (op n <> Op.After && List.exists go (src_without_body n))
        in
        Tbl.replace seen n r;
        r
  in
  go

let wrap_trip_values c ~written ~trip =
  let args = src_without_body c in
  let reads_trip = reads_trip ~trip in
  let moved =
    List.filteri
      (fun i a -> (not (List.mem i written)) && (not (taken a)) && reads_trip a)
      args
  in
  if moved = [] then None
  else
    let is_moved a = List.memq a moved in
    let kept = List.filter (fun a -> not (is_moved a)) args in
    (* The wrapper's parameter for [n] at [slot]: a scalar of the loops,
       storage, a state or a row, or new storage holding a value. *)
    let param_of n slot =
      if shape n = [] && addrspace n = Some Dtype.Alu then
        Shape.param ~addrspace:Dtype.Alu slot (dtype n)
      else if taken n then Shape.param_like n slot
      else
        Shape.param ~shape:(shape n) ?device:(device (storage_base n)) slot
          (dtype n)
    in
    let next = ref (List.length kept) and reads = ref [] in
    let params = Tbl.create 8 in
    let leaf n =
      match Tbl.find_opt params n with
      | Some p -> p
      | None ->
          if moves_with_loop n && not (loop_row n) then
            invalid_arg
              "a call in a loop reads a value that moves with a loop and is no \
               row of storage it can take: store it into storage first";
          let p = param_of n !next in
          incr next;
          reads := n :: !reads;
          Tbl.replace params n p;
          p
    in
    let pure u =
      not
        (List.exists
           (fun n ->
             (Op.Set.mem (op n) storage_ops && not (is_bound_var n))
             || op n = Op.Range)
           (toposort ~calls:Skip u))
    in
    let rebuilt = Tbl.create 16 in
    let rec value n =
      match Tbl.find_opt rebuilt n with
      | Some v -> v
      | None ->
          let v =
            if pure n then n
            else if taken n || not (reads_trip n) then leaf n
            else
              let body = if op n = Op.Call then [ nth n 0 ] else [] in
              replace n ~src:(body @ List.map value (src_without_body n))
          in
          Tbl.replace rebuilt n v;
          v
    in
    (* A kept argument keeps the body's parameter, at the wrapper's slot. *)
    let params_of_body = toposort ~calls:Skip (body c) in
    let body_param i a slot =
      let at u = op u = Op.Param && (param_arg_of u).slot = i in
      match List.find_opt at params_of_body with
      | Some p -> replace p ~arg:(Param { (param_arg_of p) with slot })
      | None -> param_of a slot
    in
    let slot = ref 0 in
    let inner_args =
      List.mapi
        (fun i a ->
          if is_moved a then value a
          else begin
            let s = !slot in
            incr slot;
            body_param i a s
          end)
        args
    in
    let inner =
      match (op (body c), arg (body c)) with
      | Op.Sink, Kernel _ -> replace c ~src:(body c :: inner_args)
      | Op.Sink, _ -> call (body c) inner_args
      | Op.Store, _ -> call (sink [ body c ]) inner_args
      | _ -> replace c ~src:(body c :: inner_args)
    in
    let wrote =
      List.map (fun i -> after (List.nth inner_args i) [ inner ]) written
    in
    let wrapper = sink (if wrote = [] then [ inner ] else wrote) in
    Some (replace c ~src:(wrapper :: kept @ List.rev !reads))

(* A stage of a value a loop moves, left among a call's arguments, would be
   stored before the loop, over ranges no store before it holds. *)
let refuse_moving_stage c =
  let moving_stage a =
    (not (has_buffer_identity ~after_ok:true a))
    && List.exists
         (fun n -> op n = Op.Stage && moves_with_loop (nth n 0))
         (toposort ~calls:Skip a)
  in
  if List.exists moving_stage (src_without_body c) then
    invalid_arg
      "a call in a loop reads a stored value that moves with a loop: store it \
       in the call's body"

(* Whether [k] is a loop around a call: an end over loop ranges, or a back
   edge. *)
let loop_of_call k =
  (op k = Op.Backedge || (op k = Op.End && List.exists loop_range (List.tl (src k))))
  && op (nth k 0) = Op.Call

(* The loop [k] with its call reading its values each trip, if they need it. *)
let wrap_loop k =
  let c = nth k 0 in
  match Shape.written_params (body c) with
  | None -> None
  | Some written ->
      let priors =
        List.map named
          (List.filteri (fun i _ -> List.mem i written) (src_without_body c))
      in
      let c' = wrap_trip_values c ~written ~trip:(fun n -> List.memq n priors) in
      refuse_moving_stage (Option.value c' ~default:c);
      Option.map (fun c' -> replace k ~src:(c' :: List.tl (src k))) c'

(* The body a call runs that is no kernel. *)
let callee u =
  if op u <> Op.Call || op (body u) <> Op.Sink then None
  else match arg (body u) with Kernel _ -> None | _ -> Some (body u)

let trip_reads sink =
  let loops = Tbl.create 16 and bodies = Tbl.create 16 in
  (* Whether [g], or a body it calls at any depth, has a loop of a call. *)
  let rec has_loop g =
    match Tbl.find_opt loops g with
    | Some l -> l
    | None ->
        let l =
          List.exists
            (fun u ->
              loop_of_call u || Option.fold ~none:false ~some:has_loop (callee u))
            (toposort ~calls:Skip g)
        in
        Tbl.replace loops g l;
        l
  in
  (* [g], sequenced, with its bodies' loops wrapped, then its own: a value a
     loop moves into its wrapper holds the calls' wrapped bodies. *)
  let rec scope g =
    let with_body u =
      match callee u with
      | None -> None
      | Some b ->
          let b' = of_body b in
          if b' == b then None else Some (u, replace u ~src:(b' :: List.tl (src u)))
    in
    let substituted g f =
      match List.filter_map f (toposort ~calls:Skip g) with
      | [] -> g
      | changed -> substitute ~calls:Skip ~pass:Fixed_point g changed
    in
    let g = substituted g with_body in
    substituted g (fun u ->
        if loop_of_call u then Option.map (fun k -> (u, k)) (wrap_loop u)
        else None)
  (* A body keeps its form unless a loop in one of its scopes needs a wrap. *)
  and of_body b =
    match Tbl.find_opt bodies b with
    | Some b' -> b'
    | None ->
        let b' =
          if not (has_loop b) then b
          else
            let s = Prepare.sequence b in
            let s' = scope s in
            if s' == s then b else s'
        in
        Tbl.replace bodies b b';
        b'
  in
  if has_loop sink then scope sink else sink

let transform_to_call big_sink =
  if setting Setting.spec <> 0 then
    Spec.type_verify ~calls:Skip Spec.tensor big_sink;
  (* The stores are collected before these rewrites change node identities,
     and the buffers they write before any view is made. *)
  let ctx = callify_ctx () in
  let nodes = toposort ~calls:Skip big_sink in
  List.iter
    (fun u ->
      if is_store_after u && not (is_join u) then
        Tbl.replace ctx.written (storage_base u) ())
    nodes;
  let big_sink = trip_reads big_sink in
  ignore
    (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx big_sink
       (After_sources pm_callify_ctx_collect));
  let ret =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx
      (sink (List.rev ctx.stores))
      (Before_sources
         (Pattern_matcher.concat [ pm_canonicalize_body; pm_replace_buf ]))
  in
  call ~precompile:true ret (List.rev ctx.replacements)

(* Launches

   A launch runs calls of one device at once, one dispatch, so that no call
   pays a launch of its own. Its members keep no order, and a group is safe to
   run at once exactly when none of them accesses bytes another writes: two
   calls interfere when one writes a byte the other reads or writes. A write
   marked disjoint ({!Ops.after_disjoint}) is the only exception: its kernels
   write regions its maker proved apart, which their byte ranges cannot show
   when a region is no run of the storage (a column of a matrix, a piece
   starting at a variable), so they do not interfere on that state. The mark
   counts only on the state, for a pair that both write it there: a kernel of
   the write that also touches other storage interferes there as any call.

   One scan over each scope of the finished schedule decides: the top level,
   and each loop's body. Calls are weighed in order, and a call joins only a
   group of its device that starts after every position it interferes with,
   so it moves up only past calls it does not interfere with. A call moved
   up makes its storage earlier, and the two scopes weigh that apart.

   At the top level the schedule is the one each kernel is a call of, and it
   stays that one but for launches of the stores of one state, which are
   what a launch of many elements is for, and of calls of one element: a
   call of many elements joins only a group
   writing storage it writes, and no call moves past the first use of
   storage it takes, unless the storage is given to the schedule or holds
   one element.

   A loop's body runs its calls every trip, so each launch it saves is saved
   every trip, and the memory plan holds the body's own storage within a
   trip, the most a step holds for the whole loop ({!Memory}). There a call
   joins the earliest group at whose start no step up to the call would hold
   more of the body's own storage than the trip's peak: a launch costs the
   trip no byte. A buffer of one element counts as a block of the arena, so
   a body of small storage at its peak runs its calls apart. Storage the
   schedule gives, or that entries outside the body take, lives through
   every trip and does not count.

   Target limits are no part of this: every subset of a launch is a launch, and
   {!Realize.lower_and_compile} splits one its target cannot run. *)

(* The bytes of its storage the view [a] holds, from [(0, max_int)] for
   storage, or [None] where the bytes are not constants or the view is no run
   of them: such a view counts as the whole storage. *)
let rec view_range a =
  match op a with
  | Op.Buffer | Op.Alloc | Op.Param -> Some (0, max_int)
  | Op.Bitcast -> view_range (nth a 0)
  | Op.Shrink -> (
      match (view_range (nth a 0), marg a) with
      | Some (start, _), Shrink [ (Int lo, Int n) ] ->
          let size = element_size a in
          Some (start + (lo * size), start + ((lo + n) * size))
      | _ -> None)
  | _ -> None

(* An argument of a call: the bytes of [base] it takes, whether it writes
   them, and the disjoint writes it is in the state of. *)
type access = { base : t; lo : int; hi : int; written : bool; marks : int list }

let interferes a b =
  (a.written || b.written)
  && a.lo < b.hi && b.lo < a.hi
  && not (List.exists (fun m -> List.mem m b.marks) a.marks)

(* The storage an entry takes, without the values it is passed. *)
let storage_taken e =
  List.filter (fun b -> addrspace b <> Some Dtype.Alu) (Memory.taken e)

(* A kernel call's arguments as accesses; any other entry, a copy or a loop,
   writes its storage whole, a loop being one entry. *)
let accesses e =
  let whole base = { base; lo = 0; hi = max_int; written = true; marks = [] } in
  let is_kernel_call =
    op e = Op.Call && op (body e) = Op.Sink
    && match arg (body e) with Kernel _ -> true | _ -> false
  in
  let taken = storage_taken e in
  if not is_kernel_call then List.map whole taken
  else
    let writes = written_positions e in
    let marks_of i =
      match arg e with
      | Call c ->
          List.filter_map (fun (m, p) -> if p = i then Some m else None) c.write
      | _ -> []
    in
    let access i a =
      let base = storage_base a in
      if addrspace a = Some Dtype.Alu || not (List.memq base taken) then None
      else
        let lo, hi = Option.value (view_range a) ~default:(0, max_int) in
        let written =
          match writes with None -> true | Some w -> List.mem i w
        in
        Some { base; lo; hi; written; marks = marks_of i }
    in
    List.filter_map Fun.id (List.mapi access (src_without_body e))

(* A body's facts for {!launch_device}, which every call of it shares. *)
type facts = { constant_ranges : bool; one_element : bool }

let body_facts u =
  let ranges = List.filter (fun x -> op x = Op.Range) (toposort ~calls:Skip u) in
  {
    constant_ranges =
      List.for_all (fun r -> op (nth r 0) = Op.Const) ranges;
    one_element = ranges = [];
  }

(* The device of a call that can be a member of a launch, [taken] its
   storage: a call of a kernel, not compiled on its own, whose ranges have
   constant sizes, whose storage all lies on one device that is no disk. A
   launch lays its members' work end to end, which it must know when it is
   compiled. *)
let launch_device facts e taken =
  let of_kernel =
    op e = Op.Call && op (body e) = Op.Sink
    && (match arg (body e) with Kernel _ -> true | _ -> false)
    && match arg e with Call c -> not c.precompile | _ -> false
  in
  if not of_kernel then None
  else
    let f =
      match Tbl.find_opt facts (body e) with
      | Some f -> f
      | None ->
          let f = body_facts (body e) in
          Tbl.replace facts (body e) f;
          f
    in
    match List.sort_uniq String.compare (List.filter_map device taken) with
    | [ d ]
      when f.constant_ranges
           && List.for_all (fun b -> device b <> None) taken
           && not (is_disk_device d) ->
        Some (d, f)
    | _ -> None

(* A group: the position of its first call, its calls, last first. *)
type group = { first : int; mutable members : t list; mutable writes : t list }

(* The last index of [gs], ordered by position of their first calls, whose
   group starts at or after [bound], or their length. *)
let at_or_after gs bound =
  let rec go lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if (Dynarray.get gs mid).first < bound then go (mid + 1) hi else go lo mid
  in
  go 0 (Dynarray.length gs)

(* The bytes each step of a loop's body holds of the body's own storage on
   one device, from each buffer's first use to its last, and the most a step
   held before any call joined a launch. *)
type lane = { live : int array; peak : int }

(* The storage of a loop's body whose bytes grouping moves: a buffer the
   memory plan places, which the schedule is not given and no entry outside
   the body takes. Each one's uses, at the positions their entries run at,
   and each device's lane. *)
let own_storage ~fixed entries =
  let n = Array.length entries in
  let taken = Array.map Memory.buffers entries in
  let own b =
    (not (Tbl.mem fixed b))
    && match device b with Some d -> not (is_disk_device d) | None -> false
  in
  let uses = Tbl.create n in
  Array.iteri
    (fun i bs ->
      List.iter
        (fun b ->
          if own b then
            Tbl.replace uses b
              (i :: Option.value (Tbl.find_opt uses b) ~default:[]))
        bs)
    taken;
  let live = Hashtbl.create 4 in
  let lane b =
    let d = Option.get (device b) in
    match Hashtbl.find_opt live d with
    | Some l -> l
    | None ->
        let l = Array.make n 0 in
        Hashtbl.replace live d l;
        l
  in
  Tbl.iter
    (fun b us ->
      let l = lane b in
      for s = List.fold_left min n us to List.fold_left max (-1) us do
        l.(s) <- l.(s) + Memory.bytes b
      done)
    uses;
  let lanes = Hashtbl.create 4 in
  Hashtbl.iter
    (fun d l ->
      Hashtbl.replace lanes d { live = l; peak = Array.fold_left max 0 l })
    live;
  (taken, uses, lanes)

(* The entries [entries] with the calls that can run as one launched. [held]
   is the storage the schedule is given, which outlives it; [fixed], where
   the entries are a loop's body, is that storage and the storage entries
   outside the body take; [facts] holds the bodies met. *)
let launched_entries ~facts ~held ?fixed entries =
  let n = Array.length entries in
  let body = Option.map (fun fixed -> own_storage ~fixed entries) fixed in
  (* In a loop's body, the earliest of the device's groups [gs], back from its
     last, that call [i] joins: one that starts after [conflict], the last
     call [i] interferes with, at whose start every step up to [i] still
     holds no more than its lane's peak with the buffers [i] makes. Only the
     body's own storage counts: what the loop is given or shares outside its
     body lives through every trip. A call that joins frees its last reads
     earlier, which the check leaves out. *)
  let in_loop (taken, uses, lanes) gs i conflict =
    let made = Hashtbl.create 2 in
    List.iter
      (fun b ->
        match Tbl.find_opt uses b with
        | Some us when List.for_all (fun u -> u >= i) us ->
            let d = Option.get (device b) in
            let k = Option.value (Hashtbl.find_opt made d) ~default:0 in
            Hashtbl.replace made d (k + Memory.bytes b)
        | _ -> ())
      taken.(i);
    (* The most each lane holds from a step on, the scan's running maximum. *)
    let most = Hashtbl.create 2 in
    let holds s =
      Hashtbl.fold
        (fun d k ok ->
          let l = Hashtbl.find lanes d in
          let before = Option.value (Hashtbl.find_opt most d) ~default:0 in
          let m = max l.live.(s) before in
          Hashtbl.replace most d m;
          ok && m + k <= l.peak)
        made true
    in
    let rec back j from best =
      if j < 0 then best
      else
        let g = Dynarray.get gs j in
        let rec down s =
          s < g.first || Hashtbl.length made = 0 || (holds s && down (s - 1))
        in
        if g.first <= conflict || not (down from) then best
        else back (j - 1) (g.first - 1) (Some g)
    in
    back (Dynarray.length gs - 1) (i - 1) None
  in
  (* Call [i] runs at [p]: its own buffers' uses move with it. *)
  let move (taken, uses, lanes) i p =
    List.iter
      (fun b ->
        match Tbl.find_opt uses b with
        | None -> ()
        | Some us ->
            let l = (Hashtbl.find lanes (Option.get (device b))).live in
            let span us =
              (List.fold_left min n us, List.fold_left max (-1) us)
            in
            let at = List.map (fun u -> if u = i then p else u) us in
            let add (lo, hi) k =
              for s = lo to min hi (n - 1) do
                l.(s) <- l.(s) + k
              done
            in
            add (span us) (-Memory.bytes b);
            add (span at) (Memory.bytes b);
            Tbl.replace uses b at)
      taken.(i)
  in
  let reads = Tbl.create n and writes = Tbl.create n in
  let first_use = Tbl.create n in
  let log tbl b x =
    Tbl.replace tbl b (x :: Option.value (Tbl.find_opt tbl b) ~default:[])
  in
  (* The groups of each device, by their first call, and the groups writing
     each storage, last first by their first calls. *)
  let groups = Hashtbl.create 4 and writers = Tbl.create n in
  let opened = Array.make n None and joined = Array.make n false in
  let add_writer g b =
    let rec insert = function
      | x :: rest when x.first > g.first -> x :: insert rest
      | x :: _ as l when x == g -> l
      | l -> g :: l
    in
    Tbl.replace writers b
      (insert (Option.value (Tbl.find_opt writers b) ~default:[]))
  in
  let join g e written =
    g.members <- e :: g.members;
    List.iter
      (fun b ->
        if not (List.memq b g.writes) then begin
          g.writes <- b :: g.writes;
          add_writer g b
        end)
      written
  in
  (* The earliest group from [bound] on writing storage of [written], or
     none: [writers] are by first call, last first, so the walk of each
     storage's stops at the first earlier than [bound]. *)
  let writing written bound =
    let earlier g best =
      match best with Some b when b.first <= g.first -> best | _ -> Some g
    in
    let rec go best = function
      | g :: rest when g.first >= bound -> go (earlier g best) rest
      | _ -> best
    in
    List.fold_left
      (fun best b ->
        go best (Option.value (Tbl.find_opt writers b) ~default:[]))
      None written
  in
  for i = 0 to n - 1 do
    let e = entries.(i) in
    let acc = accesses e in
    let taken = storage_taken e in
    let conflict =
      List.fold_left
        (fun c a ->
          (* A read interferes only with writes; [reads] logs every access. *)
          let against tbl =
            List.fold_left
              (fun c (b, pos) -> if interferes a b then max c pos else c)
              c
              (Option.value (Tbl.find_opt tbl a.base) ~default:[])
          in
          if a.written then against reads else against writes)
        (-1) acc
    in
    let bound =
      List.fold_left
        (fun lo b ->
          if Tbl.mem held b || Shape.max_numel b = 1 then lo
          else max lo (Option.value (Tbl.find_opt first_use b) ~default:i))
        (conflict + 1) taken
    in
    let written =
      List.filter_map (fun a -> if a.written then Some a.base else None) acc
    in
    let pos =
      match launch_device facts e taken with
      | None -> i
      | Some (d, f) -> (
          let gs =
            match Hashtbl.find_opt groups d with
            | Some gs -> gs
            | None ->
                let gs = Dynarray.create () in
                Hashtbl.replace groups d gs;
                gs
          in
          let picked =
            match body with
            | Some b -> in_loop b gs i conflict
            | None when f.one_element ->
                let j = at_or_after gs bound in
                if j < Dynarray.length gs then Some (Dynarray.get gs j)
                else None
            | None ->
                (* Groups of other devices write other storage. *)
                writing written bound
          in
          match picked with
          | Some g ->
              Option.iter (fun b -> move b i g.first) body;
              join g e written;
              joined.(i) <- true;
              g.first
          | None ->
              let g = { first = i; members = []; writes = [] } in
              join g e written;
              Dynarray.add_last gs g;
              opened.(i) <- Some g;
              i)
    in
    List.iter
      (fun a ->
        log reads a.base (a, pos);
        if a.written then log writes a.base (a, pos);
        if not (Tbl.mem first_use a.base) then Tbl.replace first_use a.base pos)
      acc
  done;
  let out =
    List.filter_map
      (fun i ->
        if joined.(i) then None
        else
          match opened.(i) with
          | Some g when List.compare_length_with g.members 1 > 0 ->
              Some (Shape.launch (List.rev g.members))
          | _ -> Some entries.(i))
      (List.init n Fun.id)
  in
  out

(* [linear] with its calls launched, in its loops' bodies too. [fixed], in
   a loop's body, is the storage whose life the body does not decide. *)
let rec launch_linear ~facts ~held ?fixed linear =
  let es = Array.of_list (src linear) in
  let taken = Array.map Memory.buffers es in
  let count = Tbl.create 16 in
  Array.iter
    (List.iter (fun b ->
         let k = Option.value (Tbl.find_opt count b) ~default:0 in
         Tbl.replace count b (k + 1)))
    taken;
  let entries =
    Array.mapi
      (fun j e ->
        match op e with
        | (Op.End | Op.Backedge) when op (nth e 0) = Op.Linear ->
            (* What the scope gives the loop, and what its other entries
               take, lives through the loop's trips. *)
            let inner = Tbl.copy (Option.value fixed ~default:held) in
            Tbl.iter
              (fun b k ->
                if k > (if List.memq b taken.(j) then 1 else 0) then
                  Tbl.replace inner b ())
              count;
            let body = launch_linear ~facts ~held ~fixed:inner (nth e 0) in
            replace e ~src:(body :: List.tl (src e))
        | _ -> e)
      es
  in
  let src =
    if Array.length entries < 2 then Array.to_list entries
    else launched_entries ~facts ~held ?fixed entries
  in
  replace linear ~src

let launches ~held linear =
  let tbl = Tbl.create 16 in
  List.iter (fun b -> Tbl.replace tbl b ()) held;
  launch_linear ~facts:(Tbl.create 16) ~held:tbl linear

(* Schedules *)

let create_linear_with_vars ?(capturing = false) ?held big_sink =
  let several = bound_several big_sink in
  let big_sink = transform_to_call (Prepare.sequence big_sink) in
  (* The call's sources are the values to realize. *)
  let linear_call =
    graph_rewrite ~calls:Enter ~pass:Fixed_point ~ctx:() big_sink
      (After_sources pm_schedule)
  in
  (* The linear call is resolved recursively, and its storage allocated. *)
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:several linear_call
      (After_sources pm_resolve_linear_call)
  in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() linear
      (After_sources pm_copy_from_store)
  in
  (* The storage the call is given, passed or viewed, outlives it. *)
  let held_bufs =
    List.filter_map
      (fun a ->
        let b = storage_base a in
        if op b = Op.Buffer then Some b else None)
      (List.tl (src linear_call))
  in
  let linear =
    launches ~held:(Option.value held ~default:held_bufs) linear
  in
  (* The entries of [linear], those of its loops' bodies among them. *)
  let rec entries linear =
    List.concat_map
      (fun e ->
        if is_loop e && op (nth e 0) = Op.Linear then entries (nth e 0)
        else [ e ])
      (src linear)
  in
  let used_vars =
    List.concat_map
      (fun si ->
        (* A launch's members read the variables of their bodies, and a call
           passes a member's that it binds in its slot ({!Shape.launch}). *)
        let bodies =
          if op si = Op.Call && Shape.is_launch (nth si 0) then
            List.map body (src (nth si 0))
          else [ nth si 0 ]
        in
        let args = if op si = Op.Call then src_without_body si else [] in
        List.concat_map
          (fun b -> List.filter_map (fun v -> (param_arg_of v).name) (variables b))
          (bodies @ args))
      (entries linear)
  in
  let var_vals =
    List.fold_left
      (fun acc b ->
        if not (is_bound_var b) then acc
        else
          let nm = expr b in
          let value = Dtype.Value.to_int (Option.get (param_arg_of b).bound) in
          (* A name of several values is no run's: its calls pass them. *)
          if
            List.mem nm several
            || (not (List.mem nm used_vars))
            || List.mem_assoc nm acc
          then acc
          else acc @ [ (nm, value) ])
      []
      (List.tl (src big_sink))
  in
  if capturing then (linear, var_vals)
  else (Memory.memory_plan_rewrite ~held_bufs linear, var_vals)
