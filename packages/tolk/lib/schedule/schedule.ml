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
let is_view p = view_state p != p

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
  let s = view_state (unwrap_src s) in
  if is_join s then state_of (nth s 0) else s

let value_state s =
  let rec go deps s =
    let s = view_state (unwrap_src s) in
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
  let device =
    match device buf with
    | Some d -> d
    | None -> invalid_arg "a copied buffer state needs a device"
  in
  let n = max_numel buf in
  let arg = param_arg ~slot ~device ~phase ~align (dtype buf) in
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

(* Whether [o] is [s] or a state [s] is after. *)
let rec is_state_or_after o s =
  s == o || (op s = Op.After && is_state_or_after o (state_of (nth s 0)))

(* A kernel as the schedule runs it: a call of its storage, in a loop where it
   runs in one. *)
let scheduled rk =
  let loops r = Axis_type.equal (axis_type r) Axis_type.Loop in
  (* A call's argument is the storage it reads, or a view of it that moves with
     the loops the call runs in. *)
  let rec argument s =
    if
      Op.Set.mem (op s) Op.Set.movement
      && List.exists loops (Nodes.to_list (ranges s))
    then replace s ~src:(argument (nth s 0) :: List.tl (src s))
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
  | Op.End when List.exists loops (List.tl (src rk)) -> with_calls rk calls
  | Op.Backedge ->
      let rk = with_calls rk calls in
      replace rk ~src:[ nth rk 0; nth rk 1; buf_uop (unwrap_src (nth rk 2)) ]
  | _ -> List.hd calls

module Iset = Set.Make (Int)

(* Raised by the orders where the kernels form a cycle. *)
exception Cycle

(* The kernels in an order of the graph [children], [in_degree] counting the
   kernels each waits for: each as it becomes ready. *)
let ready_order ~entry ~children in_degree =
  let queue = Queue.create () in
  List.iter
    (fun (k, d) -> if d = 0 then Queue.push k queue)
    (Ordered.bindings in_degree);
  let linearized = ref [] in
  while not (Queue.is_empty queue) do
    let rk = Queue.pop queue in
    linearized := entry rk :: !linearized;
    List.iter
      (fun x ->
        let d = Option.get (Ordered.find_opt in_degree x) - 1 in
        Ordered.replace in_degree x d;
        if d = 0 then Queue.push x queue)
      (List.rev (Option.value (Tbl.find_opt children rk) ~default:[]))
  done;
  if List.exists (fun (_, d) -> d <> 0) (Ordered.bindings in_degree) then
    raise Cycle;
  v Op.Linear ~src:(List.rev !linearized)

(* The kernels in an order of the graph [children], [in_degree] counting the
   kernels each waits for and numbering them in the order the walk of the sink
   meets them.

   A ready kernel runs by its number, so that a value is consumed soon after it
   is made, as the program wrote it, and one that frees storage and on no device
   makes more than it frees runs before it. The storage a kernel writes,
   [written], lives from the first kernel that takes it to the last
   ({!Memory.taken}); other storage, which only kernels read, lives throughout.
   So a closing that slices a partial for each device sums each slice before it
   makes the next, and the last reader of a gathered value runs before the
   kernels that make more. A pick walks the ready kernels: [O(k²)] for [k]
   kernels at worst. *)
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

let linearize ~entry ~by_storage ~children sched_sink in_degree =
  if not by_storage then ready_order ~entry ~children in_degree
  else
    (* The storage the kernels write. *)
    let written = Tbl.create 16 in
    List.iter
      (fun u ->
        if op u = Op.After && not (is_join u) then
          Tbl.replace written (buf_uop u) ())
      (toposort ~calls:Skip ~gate:in_graph sched_sink);
    storage_order ~entry ~written ~children in_degree

(* The state the kernels [ks] of the after [u] write over: the latest state of
   [u]'s storage their calls are given, as a loop is given its carry's first
   state while its after names the storage alone; else the state [u] is
   after. *)
let written_over u ks =
  let declared = fst (value_state (nth u 0)) in
  let storage = buf_uop declared in
  let latest best c =
    List.fold_left
      (fun best a ->
        if scalar_argument c a || is_bound_var a || is_view (unwrap_src a) then
          best
        else
          let s = state_of a in
          if buf_uop s != storage then best
          else
            match best with
            | Some b when not (is_state_or_after b s) -> Some b
            | _ -> Some s)
      best (List.tl (src c))
  in
  let best =
    List.fold_left
      (fun best k -> List.fold_left latest best (calls_of k))
      None ks
  in
  Option.value best ~default:declared

(* The relation

   A kernel reads the states its arguments name, a view of storage the state
   its second source names ({!Ops.view_state}), and writes the states the Afters that list it are over. It
   runs after the kernels that write the states it reads, and before the
   kernels of every write over them of which it is no part, whichever elements
   each touches. Where these orders form a cycle, no order runs every kernel
   at its place: one read on the cycle reads instead a copy of its state, made
   before the state's overwrite, and the cycle is gone.

     a --write--> a'        b reads a, waits for c (through a'),
     b(a, c')               and c overwrites a: b -> c -> b
     c(a') --> a''          b reads a copy of a, made before c

   A cycle is found where Kahn's order stops: walking back from a kernel not
   run, through kernels not run, meets one of them again. The read cut is one
   whose state is copied already, else the one of fewest bytes, of storage that
   can be copied first. Finding the fewest copies is NP-hard; this is greedy.
   A graph whose kernels form no cycle gets no copy. *)

(* A read that a write over its state must follow: the kernel [reader] reads
   [state]. *)
type war = { reader : t; state : t }

let same_war a b = a.reader == b.reader && a.state == b.state

let after_over_view u = op u = Op.After && is_view (unwrap_src (nth u 0))

(* Whether the kernel [k] stores into the state [s]. *)
let writes_state k s =
  List.exists
    (fun c ->
      match written_positions c with
      | None -> true
      | Some w -> List.exists (fun i -> List.mem i w) (positions c s))
    (calls_of k)

(* Whether the argument [a] at position [i] of the call [c] is a view [c]
   stores through: it names its storage and no state [c] reads. *)
let stored_view c i a =
  is_view (unwrap_src a)
  &&
  match written_positions c with Some w -> List.mem i w | None -> false

(* Whether the kernel [k] stores into the storage [storage]. *)
let stores_into k storage =
  let into c =
    match written_positions c with
    | None -> false
    | Some w ->
        let args = src_without_body c in
        List.exists
          (fun i -> whole (storage_base (List.nth args i)) == storage)
          w
  in
  List.exists into (calls_of k)

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

(* Whether the kernel [k] is given the state [s] at an argument. *)
let given_state s k =
  List.exists
    (fun c ->
      List.exists
        (fun a -> (not (scalar_argument c a)) && state_of a == s)
        (List.tl (src c)))
    (calls_of k)

(* Whether a reader of the state [st], which the kernels [own] write, waits
   for the state under it beyond [own]: not where each is given it. *)
let waits_under st own =
  let prior = state_of (nth st 0) in
  op prior = Op.After && not (own <> [] && List.for_all (given_state prior) own)

(* The disjoint writes

   A write marked disjoint ({!Ops.after_disjoint}) lets a launch hold kernels
   its regions can't show apart. The kernels keep the fact, past the schedule
   and its cache, as a number of the write and the position of the state among
   their arguments ({!Ops.call_info}): numbers follow the walk of the graph,
   so that a schedule kept on disk reads back the same. *)

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

(* The kernels of a write in the order its After lists them, a kernel listed
   twice at its last place: a store landing twice leaves what its last landing
   left. *)
let rec landing = function
  | [] -> []
  | k :: rest -> if List.memq k rest then landing rest else k :: landing rest

(* Whether the kernel [k] reads the state [s] through an argument its calls do
   not write. *)
let reads_elsewhere k s =
  List.exists
    (fun c ->
      match written_positions c with
      | None -> false
      | Some w -> List.exists (fun i -> not (List.mem i w)) (positions c s))
    (calls_of k)

(* A state over the whole of its storage holds its own kernels' stores, those
   of the states under it, and each store those kernels run after: no other
   store over a view of a state on the way. So a reader of it runs before every
   kernel written over one of those states that the kernels of its chain do not
   follow. Each edge goes through the read it orders, which a copy can cut.
   [children] holds the edges of reads after writes and of listings. *)
let sibling_edges ~children ~writes ~add_child reads =
  (* The kernels of the state [s] and the state they are written over, for a
     state on a chain: a whole state with kernels. *)
  let links = Tbl.create 16 in
  let link s =
    match Tbl.find_opt links s with
    | Some l -> l
    | None ->
        let l =
          if
            op s <> Op.After || is_join s
            || is_view (unwrap_src (nth s 0))
          then None
          else
            match fst (split_after s) with
            | [] -> None
            | own -> Some (own, written_over s own)
        in
        Tbl.replace links s l;
        l
  in
  (* The kernels written over each state of [s]'s chain other than that state's
     own: almost always none. *)
  let others = Tbl.create 16 in
  let rec others_of s =
    match link s with
    | None -> []
    | Some (own, prev) -> (
        match Tbl.find_opt others s with
        | Some ks -> ks
        | None ->
            let here =
              List.filter
                (fun t -> not (List.memq t own))
                (Option.value (Tbl.find_opt writes prev) ~default:[])
            in
            let ks = here @ others_of prev in
            Tbl.replace others s ks;
            ks)
  in
  (* For each kernel the kernels it runs before, itself among them, along reads
     after writes and listings, and which states' chains hold one of them. *)
  let reach = Tbl.create 16 in
  let reached t =
    match Tbl.find_opt reach t with
    | Some r -> r
    | None ->
        let kernels = Tbl.create 16 in
        let rec go k =
          if not (Tbl.mem kernels k) then begin
            Tbl.replace kernels k ();
            List.iter go (Option.value (Tbl.find_opt children k) ~default:[])
          end
        in
        go t;
        let r = (kernels, Tbl.create 16) in
        Tbl.replace reach t r;
        r
  in
  let rec on_chain ((kernels, seen) as r) s =
    match link s with
    | None -> false
    | Some (own, prev) -> (
        match Tbl.find_opt seen s with
        | Some b -> b
        | None ->
            let b = List.exists (Tbl.mem kernels) own || on_chain r prev in
            Tbl.replace seen s b;
            b)
  in
  let memo = Tbl.create 16 in
  let siblings s =
    match Tbl.find_opt memo s with
    | Some ks -> ks
    | None ->
        let ks =
          match others_of s with
          | [] -> []
          | ks ->
              List.filter
                (fun t -> not (on_chain (reached t) s))
                (dedup_nodes ks)
        in
        Tbl.replace memo s ks;
        ks
  in
  (* The slots a body loads: in a store's value or index expressions, and at
     a nested call the slots it loads or does not write. A body that does not
     show its writes loads what it is given. *)
  let rec loaded_slots body =
    let slots = ref [] in
    let add u =
      if op u = Op.Param && addrspace u <> Some Dtype.Alu then
        slots := (param_arg_of u).slot :: !slots
    in
    let reads u = List.iter add (toposort ~calls:Skip u) in
    List.iter
      (fun u ->
        match op u with
        | Op.Store ->
            List.iter reads (List.tl (src u));
            List.iter reads (List.tl (src (nth u 0)))
        | Op.Call ->
            let args = src_without_body u in
            let wanted =
              match written_params (nth u 0) with
              | None -> fun _ -> true
              | Some w ->
                  let loaded = loaded_slots (nth u 0) in
                  fun i -> (not (List.mem i w)) || List.mem i loaded
            in
            List.iteri (fun i a -> if wanted i then reads a) args
        | _ -> ())
      (toposort ~calls:Skip body);
    !slots
  in
  (* The written positions of the kernel [k]'s calls that hold [s]: loaded, a
     read of [s] no copy can stand in for. *)
  let loaded_written k s =
    List.exists
      (fun c ->
        match written_positions c with
        | None -> false
        | Some w ->
            let loaded = loaded_slots (body c) in
            List.exists
              (fun i -> List.mem i w && List.mem i loaded)
              (positions c s))
      (calls_of k)
  in
  (* Whether the kernel [k] reads [s] other than where it writes it. *)
  let read_elsewhere k s =
    List.exists
      (fun c ->
        match written_positions c with
        | None -> positions c s <> []
        | Some w -> List.exists (fun i -> not (List.mem i w)) (positions c s))
      (calls_of k)
  in
  (* The edge from [k], reading [s], to the sibling [t], and the read it
     orders where a copy can stand in for it. A store through a state it does
     not load through there reads no element of it. A load through the
     argument [k] stores through reads the state the call writes, which holds
     each store the call runs after: it needs no edge to one it follows, and
     its edge to another orders no read a copy can take, as a listing's does
     not. *)
  let edge k s t =
    if t == k then None
    else if loaded_written k s then
      if Tbl.mem (fst (reached t)) k then None else Some None
    else if read_elsewhere k s then Some (Some { reader = k; state = s })
    else None
  in
  let edges =
    List.concat_map
      (fun (k, s) ->
        List.filter_map
          (fun t -> Option.map (fun war -> (k, t, war)) (edge k s t))
          (siblings s))
      (List.rev reads)
  in
  List.iter (fun (k, t, war) -> add_child ?war k t) edges

(* The Afters of [nodes] over a storage's state, with their kernels. *)
let writes_of nodes =
  List.filter_map
    (fun u ->
      if op u = Op.After && not (is_join u) then Some (u, fst (split_after u))
      else None)
    nodes

(* The kernels an {!Op.After} over a later state of a storage's lists, with
   each kernel's argument it may name that storage by: a store through an older
   state, listed over a later one, writes the later one. [sink] with such an
   argument naming the state the earliest such {!Op.After} is over, unless an
   {!Op.After} of the older state lists the kernel too. *)
let name_written_states nodes sink =
  let over u = fst (value_state (nth u 0)) in
  let older c a declared =
    let s = state_of a in
    (not (scalar_argument c a))
    && (not (is_view (unwrap_src a)))
    && s != declared && buf_uop s == buf_uop declared
    && is_state_or_after s declared
  in
  let rec any_older c declared = function
    | [] -> false
    | a :: rest -> older c a declared || any_older c declared rest
  in
  let call_older c declared = any_older c declared (List.tl (src c)) in
  let rec listed_older declared = function
    | [] -> false
    | d :: rest ->
        (match op d with
        | Op.Call -> call_older d declared
        | Op.End | Op.Backedge -> (
            let b = nth d 0 in
            match op b with
            | Op.Call -> call_older b declared
            | Op.Linear -> listed_older declared (src b)
            | _ -> false)
        | _ -> false)
        || listed_older declared rest
  in
  let later u =
    op u = Op.After
    && (not (is_join u))
    && op (nth u 0) = Op.After
    && listed_older (state_of (nth u 0)) (List.tl (src u))
  in
  match List.filter later nodes with
  | [] -> None
  | laters ->
      let rename k declared =
        let listed_over s =
          List.exists
            (fun (u, ks) -> over u == s && List.memq k ks)
            (writes_of nodes)
        in
        let call c =
          let args = src_without_body c in
          if not (List.exists (fun a -> older c a declared) args) then c
          else
            match written_positions c with
            | None -> c
            | Some w ->
                let named i a =
                  if
                    List.mem i w && older c a declared
                    && not (listed_over (state_of a))
                  then swap (state_of a) declared a
                  else a
                in
                let args' = List.mapi named args in
                if List.for_all2 ( == ) args args' then c
                else replace c ~src:(body c :: args')
        in
        let calls = calls_of k in
        let calls' = List.map call calls in
        if List.for_all2 ( == ) calls calls' then k else with_calls k calls'
      in
      let renamed =
        List.fold_left
          (fun acc u ->
            let declared = over u in
            List.fold_left
              (fun acc k ->
                let current = Option.value (List.assq_opt k acc) ~default:k in
                let k' = rename current declared in
                if k' == current then acc
                else (k, k') :: List.remove_assq k acc)
              acc
              (fst (split_after u)))
          [] laters
      in
      if renamed = [] then None
      else Some (substitute ~calls:Skip ~pass:Fixed_point sink renamed)

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

let rec schedule ~by_storage ~copies sched_sink =
  let nodes = toposort ~calls:Skip ~gate:in_graph sched_sink in
  let renormal =
    if not copies then None
    else
      match name_written_states nodes sched_sink with
      | Some _ as sink -> sink
      | None -> copy_off_start_reads nodes sched_sink
  in
  match renormal with
  | Some sink -> schedule ~by_storage ~copies sink
  | None -> kernels ~by_storage ~copies nodes sched_sink

and kernels ~by_storage ~copies nodes sched_sink =
  (* The graph of kernels: edges go from a kernel to those that must run after
     it, each through the read it orders, if any. *)
  let children = Tbl.create 64 and in_degree = Ordered.create () in
  let edges = ref [] in
  let add_child ?war t k =
    Tbl.replace children t
      (k :: Option.value (Tbl.find_opt children t) ~default:[]);
    edges := (t, k, war) :: !edges;
    Ordered.replace in_degree k (Option.get (Ordered.find_opt in_degree k) + 1)
  in
  (* Each written state with the kernels that write over it, and each read of
     a state by a kernel, last first. *)
  let writes = Tbl.create 64 and reads = ref [] and seen = Tbl.create 16 in
  (* The states all of whose writes so far are through views, and those a
     marked write is over. *)
  let viewed = ref [] and marked = ref [] in
  (* The kernels a reader of [x] waits for: the writers of the state it reads,
     what each join on the way to it waits for, and what the state under it
     waits for. A writer given that state waits for it already; one given
     another, as [first] in [x.after(second).after(first)] is given [x], does
     not, and the reader would read before [second]. *)
  let waited = Tbl.create 16 in
  let rec waits x =
    match Tbl.find_opt waited x with
    | Some w -> w
    | None ->
        let st, deps = value_state x in
        let own = if op st = Op.After then fst (split_after st) else [] in
        let under =
          if op st = Op.After && waits_under st own then waits (nth st 0)
          else []
        in
        let w = dedup_nodes (own @ under @ List.concat_map waits deps) in
        Tbl.replace waited x w;
        w
  in
  (* No storage holds two successors of one state. Writes of one state of
     which at most one is over the whole of it are one write, in any listing,
     each keeping to its run. *)
  let add_write ~whole prev fresh =
    match Tbl.find_opt writes prev with
    | Some ks when not (List.equal ( == ) ks fresh) ->
        if whole && not (List.memq prev !viewed) then
          invalid_arg
            (Format.asprintf "two writes over one state of %a" pp (buf_uop prev));
        if whole then viewed := List.filter (( != ) prev) !viewed;
        Tbl.replace writes prev (dedup_nodes (ks @ fresh))
    | Some _ -> ()
    | None ->
        if not whole then viewed := prev :: !viewed;
        Tbl.replace writes prev fresh
  in
  List.iter
    (fun u ->
      if op u = Op.After && not (is_join u) then begin
        let kernels, after_deps = split_after u in
        (* A kernel already on an earlier state of this storage orders the
           state, and writes nothing over it. *)
        let storage = storage_base u in
        let listed k = Option.value (Tbl.find_opt seen k) ~default:[] in
        let older k = List.memq storage (listed k) in
        let fresh =
          if List.exists older kernels then
            List.filter (fun k -> not (older k)) kernels
          else kernels
        in
        (* The state [fresh] write over. *)
        let prior = if fresh = [] then None else Some (written_over u fresh) in
        (match prior with
        | None -> ()
        | Some prev ->
            let whole = not (is_view (unwrap_src (nth u 0))) in
            add_write ~whole prev fresh;
            if is_disjoint u then marked := prev :: !marked);
        List.iter
          (fun k ->
            if Ordered.find_opt in_degree k = None then
              Ordered.replace in_degree k 0;
            let kernel_deps =
              List.concat_map
                (fun call ->
                  List.filteri
                    (fun i s ->
                      not (scalar_argument call s || stored_view call i s))
                    (List.tl (src call)))
                (calls_of k)
            in
            (* A kernel that several Afters list reads its arguments once. *)
            let read_deps = if Tbl.mem seen k then [] else kernel_deps in
            Tbl.replace seen k (storage :: listed k);
            reads :=
              List.rev_append
                (List.map (fun s -> (k, state_of s)) read_deps)
                !reads;
            (* Read after write: a kernel runs after the kernels that wrote
               the states it reads, and those its after waits for. *)
            List.iter (fun t -> add_child t k)
              (List.concat_map waits (read_deps @ after_deps));
            (* A kernel of a write that stores through a view of the storage
               and is not given the state written over still follows that
               state's writers. *)
            match prior with
            | Some prev
              when op prev = Op.After && List.memq k fresh
                   && stores_into k (whole storage)
                   && not (given_state prev k) ->
                List.iter (fun t -> if t != k then add_child t k) (waits prev)
            | _ -> ())
          kernels;
        (* An unmarked After lists its kernels in the order they land. *)
        match fresh with
        | _ :: _ :: _ when not (is_disjoint u) ->
            let rec order = function
              | a :: (b :: _ as rest) ->
                  add_child a b;
                  order rest
              | _ -> ()
            in
            order (landing fresh)
        | _ -> ()
      end)
    nodes;
  (* Write after read: a kernel reading a state runs before every write over
     it of which it is no part. Its part is a write of that one kernel, or, in
     a write of several, a kernel storing into the state that reads it only
     through the argument it stores through, or that is marked: a kernel
     storing into [p] and [q] reads [q] as the store it makes there. Any other
     kernel of a write reads the state as a reader beside it does, before the
     write's other kernels. A kernel writing the state it reads through
     another argument runs before itself, a cycle of one kernel.

     A state over the whole of its storage holds what its own kernels leave,
     no store of another After over a view of the state under it: a reader of
     it runs before those stores, as the reader of the state under them
     would. *)
  if List.exists after_over_view nodes then
    sibling_edges ~children ~writes
      ~add_child:(fun ?war k t -> add_child ?war k t)
      !reads;
  List.iter
    (fun (k, s) ->
      match Tbl.find_opt writes s with
      | None -> ()
      | Some write_kernels ->
          let war = { reader = k; state = s } in
          let alone = match write_kernels with [ _ ] -> true | _ -> false in
          let own =
            List.memq k write_kernels && (alone || writes_state k s)
          in
          if own && List.exists (fun c -> aliases c s) (calls_of k) then
            add_child ~war k k;
          let exempt =
            own && (alone || List.memq s !marked || not (reads_elsewhere k s))
          in
          if not exempt then
            List.iter
              (fun t -> if t != k then add_child ~war k t)
              write_kernels)
    (List.rev !reads);
  let order = List.rev in_degree.Ordered.keys in
  let entry = tagged nodes in
  match linearize ~entry ~by_storage ~children sched_sink in_degree with
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
          schedule ~by_storage ~copies:false
            (copy_before_overwrites sched_sink
               (List.map (fun w -> (w.reader, w.state, overwrites w)) cut)))

let create_schedule ?(by_storage = false) sched_sink =
  schedule ~by_storage ~copies:true sched_sink

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
let schedule_key ~by_storage fn_key =
  String.concat "\n"
    ([ Source_digest.digest; fn_key; string_of_bool by_storage ]
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

let lower_sink_to_linear ~by_storage call =
  let fn = body call in
  let precompile = match arg call with Call c -> c.precompile | _ -> false in
  match (op fn, arg fn) with
  | Op.Sink, Kernel _ -> None
  | Op.Sink, _ when precompile ->
      let start = Unix.gettimeofday () in
      let numbering = numbered fn in
      let cache_key =
        Digest.BLAKE256.string (schedule_key ~by_storage numbering.key)
      in
      (* Whether neither memory nor disk had the schedule. *)
      let made = ref false in
      let make () =
        made := true;
        if setting Setting.spec <> 0 then
          Spec.type_verify ~calls:Skip Spec.tensor fn;
        numbering.place
          (schedule ~by_storage ~copies:true
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
      rule_ctx (Upat.op Op.Call ~name:"call") (fun by_storage m ->
          lower_sink_to_linear ~by_storage (m "call"));
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
  mutable stores : t list; (* Last first. *)
  written : unit Tbl.t; (* The buffers the call's stores write. *)
  listed : t list Tbl.t;
      (* Each loop, and the states the {!Op.After}s listing it are over. *)
  params : t Tbl.t; (* The parameter of each storage. *)
}

let callify_ctx () =
  {
    replacements = [];
    allocs = Ordered.create ();
    views = Tbl.create 8;
    stores = [];
    written = Tbl.create 8;
    listed = Tbl.create 8;
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
   what it views: its storage's parameter, so that every use names one storage
   whichever view it goes through, or [state], the {!Op.After} of the storage
   whose elements a loop's read of [b] takes. *)
let view_param ?state ctx b =
  let whole = storage_param ctx (storage_base b) in
  let p = replace_input_buffer ctx b in
  (* The state a view reads, in the form the rewrite that moves a state's
     movements outside it leaves it: states over storage or over states, all
     the way down. *)
  let rec state_after a =
    let x = nth a 0 in
    if Op.Set.mem (op x) Op.Set.movement then
      state_after (replace a ~src:(nth x 0 :: List.tl (src a)))
    else if op x = Op.After then
      replace a ~src:(state_after x :: List.tl (src a))
    else a
  in
  replace p ~src:(src p @ [ Option.fold ~none:whole ~some:state_after state ])

(* The run of a buffer the movements and bitcasts [src] select, as [c] takes
   it: a shrink of the buffer, whatever state it was read at. *)
let run_of c src =
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
             (shrink buf [ Some (offset, Sint.(offset + Int n)) ])
             (dtype src))
    | _ -> None

(* What the movements and bitcasts [a] read: storage or a state of it. *)
let rec named a =
  let b = base a in
  if op b = Op.Bitcast then named (nth b 0) else b

(* Movements and bitcasts of a buffer are a view of it when they select a
   contiguous range of its elements. *)
let view_of ctx c src =
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
  if op buf <> Op.Buffer || reads_target () || overwritten () then None
  else
    match run_of c src with
    | None -> None
    | Some view ->
        Option.iter (fun ctx -> Tbl.replace ctx.views view ()) ctx;
        let view = reshape view (shape c) in
        if op c = Op.Copy || op c = Op.Store then
          Some (replace c ~src:(view :: List.tl (Ops.src c)))
        else Some view

let contiguous_mops_to_view c src = view_of None c src

(* [a], movements and bitcasts of storage or of states of it, with the states
   gone: the same elements, named by the storage alone. *)
let rec without_states a =
  if op a = Op.After then without_states (nth a 0)
  else if Op.Set.mem (op a) Op.Set.movement || op a = Op.Bitcast then
    replace a ~src:(without_states (nth a 0) :: List.tl (src a))
  else a

(* The view [a], a run of storage or a state of it, is read as: one parameter
   per argument node, naming the state its elements are read at, the storage
   where that is the storage's first. *)
let loop_view ctx a view =
  match Tbl.find_opt ctx.params a with
  | Some p -> p
  | None ->
      let state = named a in
      let state = if op state = Op.After then Some state else None in
      let p = view_param ?state ctx view in
      Tbl.replace ctx.params a p;
      p

(* Moving reads into the body

   An argument of a loop's call that reads a state the trips write, and that
   the call cannot take as it is, is a value of that state: each trip computes
   it from the state as the trip starts. The call becomes one of a wrapper body
   that computes the value and calls the original body on it, and the wrapper
   takes what the value reads as arguments of its own. Each is then read by the
   rule for its state: a run or a row read each trip, a state older than the
   loop's listing as it was. A plain body's inner call inlines, so the value
   fuses with the body's kernel; a body compiled on its own is called with a
   stored value each trip. [trip_read n] says whether a read of the state [n]
   takes it as the trip starts. *)
let storage_ops = Op.Set.union Op.Set.defines (ops [ Op.After ])

let reads_moved ~trip_read =
  let seen = Tbl.create 8 in
  let rec reaches n =
    match Tbl.find_opt seen n with
    | Some r -> r
    | None ->
        let r =
          (Op.Set.mem (op n) storage_ops && trip_read n)
          || (op n <> Op.After && List.exists reaches (src_without_body n))
        in
        Tbl.replace seen n r;
        r
  in
  reaches

(* Whether the state [a] names views its storage, through its states: a
   movement under them, whichever depth. Bitcasts are outside every state
   ({!pm_state_of_bitcast}). *)
let rec viewed_state a =
  match op a with
  | Op.After -> viewed_state (nth a 0)
  | o -> Op.Set.mem o Op.Set.movement

let absorb ctx c ~run_view ~trip_read =
  let args = src_without_body c in
  let loops n =
    List.exists
      (fun r -> Axis_type.equal (axis_type r) Axis_type.Loop)
      (Nodes.to_list (ranges n))
  in
  let loop_row n =
    op n = Op.Shrink && has_buffer_identity ~after_ok:true (nth n 0) && loops n
  in
  let is_run n = Option.is_some (run_view n) in
  (* A state over a view that is no run of its storage, at any depth of
     states. *)
  let over_view n = op n = Op.After && viewed_state n && not (is_run n) in
  let takeable n =
    (has_buffer_identity ~after_ok:true n && not (over_view n))
    || (op n = Op.After && is_run n)
    || loop_row n
    || (Op.Set.mem (op n) Op.Set.movement || op n = Op.Bitcast)
       && (not (loops n))
       && is_run n
  in
  let written = Option.value (written_positions c) ~default:[] in
  (* Most arguments are storage or a view of storage the trips do not write. *)
  let rec plain_view n =
    op n = Op.Buffer || op n = Op.Param || op n = Op.Alloc || op n = Op.After
    || (Op.Set.mem (op n) Op.Set.movement || op n = Op.Bitcast)
       && plain_view (nth n 0)
  in
  let settled a = plain_view a && not (trip_read (named a)) in
  let candidates =
    List.filteri
      (fun i a ->
        (not (List.mem i written)) && (not (settled a)) && not (takeable a))
      args
  in
  if candidates = [] then c
  else
  let reaches = reads_moved ~trip_read in
  if not (List.exists reaches candidates) then c
  else
    let moved a = List.memq a candidates && reaches a in
    let kept = List.filter (fun a -> not (moved a)) args in
    let fresh = ref (List.length kept) in
    let reads = ref [] in
    let next () =
      let s = !fresh in
      incr fresh;
      s
    in
    (* A read of a written state or a row keeps where it starts; a value the
       loop does not write is stored once, at new storage's start. *)
    let of_value n slot =
      if shape n = [] && addrspace n = Some Dtype.Alu then
        Shape.param ~addrspace:Dtype.Alu slot (dtype n)
      else if reaches n || loop_row n || has_buffer_identity ~after_ok:true n
      then Shape.param_like n slot
      else Shape.param ~shape:(shape n) ?device:(device (storage_base n)) slot (dtype n)
    in
    let params = Tbl.create 8 in
    let refuse_moving () =
      invalid_arg
        "a call in a loop reads a value that moves with a loop and is no row \
         of storage it can take: store it into storage first"
    in
    let leaf n =
      match Tbl.find_opt params n with
      | Some p -> p
      | None ->
          if loops n && not (loop_row n) then refuse_moving ();
          let slot = next () in
          reads := n :: !reads;
          let p = of_value n slot in
          Tbl.replace params n p;
          p
    in
    (* The whole of the storage of the state [n], for the views over it. *)
    let whole_state n =
      let buf = storage_base n in
      let view = shrink buf [ Some (Int 0, Int (max_numel buf)) ] in
      let slot = next () in
      reads := loop_view ctx n view :: !reads;
      Shape.param ~shape:(shape buf) ?device:(device buf) slot (dtype buf)
    in
    let rec over n m =
      if Op.Set.mem (op m) Op.Set.movement then
        replace m ~src:(over n (nth m 0) :: List.tl (src m))
      else if op m = Op.After then over n (nth m 0)
      else whole_state n
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
            else if not (reaches n) then leaf n
            else if over_view n then begin
              if loops n then
                invalid_arg
                  "a call in a loop reads a state over a view that moves with \
                   the loop: take that view of the state instead";
              over n (nth n 0)
            end
            else if takeable n || op n = Op.After then leaf n
            else
              let body = if op n = Op.Call then [ nth n 0 ] else [] in
              replace n ~src:(body @ List.map value (src_without_body n))
          in
          Tbl.replace rebuilt n v;
          v
    in
    let body_param i a slot =
      let at u = op u = Op.Param && (param_arg_of u).slot = i in
      match List.find_opt at (toposort ~calls:Skip (body c)) with
      | Some p -> replace p ~arg:(Param { (param_arg_of p) with slot })
      | None -> of_value a slot
    in
    let slot = ref 0 in
    let inner_args =
      List.mapi
        (fun i a ->
          if moved a then value a
          else begin
            let s = !slot in
            incr slot;
            body_param i a s
          end)
        args
    in
    let plain =
      match (op (body c), arg (body c)) with
      | Op.Sink, Kernel _ -> false
      | Op.Sink, _ -> true
      | _ -> false
    in
    let inner =
      if plain then call (body c) inner_args
      else if op (body c) = Op.Store then call (sink [ body c ]) inner_args
      else replace c ~src:(body c :: inner_args)
    in
    let wrote =
      List.map (fun i -> after (List.nth inner_args i) [ inner ]) written
    in
    let wrapper = sink (if wrote = [] then [ inner ] else wrote) in
    replace c ~src:(wrapper :: kept @ List.rev !reads)

(* A fixed run of storage that the call of the loop [k] only reads and writes
   through another argument becomes a view of its own, read each trip: stored
   into storage instead, the run would be made once, before the loop. A run of
   a state is such a view when the call writes that state or an earlier one; a
   run of an older state shows no trip's writes, and is stored once. A run of
   the storage's first state is a view whichever state the call writes: the
   read relation copies it, once, where it is older. *)
let loop_runs ctx k =
  let loops r = Axis_type.equal (axis_type r) Axis_type.Loop in
  let c = nth k 0 in
  let in_loop =
    op k = Op.Backedge || List.exists loops (List.tl (src k))
  in
  match if op c = Op.Call && in_loop then written_positions c else None with
  | None -> None
  | Some written ->
      let stale = Shape.reads_state_written c in
      let listed = Option.value ~default:[] (Tbl.find_opt ctx.listed k) in
      (* A read of a state older than the one the loop is listed over stays
         stored: {!name_written_states} moves the write up, and the read is
         of a state the trips do not write. *)
      let older s =
        let s = state_of s in
        List.exists (fun l -> l != s && is_state_or_after s l) listed
      in
      (* A join over writes of its storage other than its own state and the
         states under it: the loop runs after them, so storage holds them, not
         the join's value. *)
      let over_other_writes s =
        is_join s
        &&
        let own = state_of (nth s 0) in
        List.exists
          (fun d ->
            buf_uop d == buf_uop s && not (is_state_or_after (state_of d) own))
          (List.tl (src s))
      in
      let rec unlifted s =
        over_other_writes s
        || ((Op.Set.mem (op s) Op.Set.movement || op s = Op.After
           || op s = Op.Bitcast)
           && unlifted (nth s 0))
      in
      (* Whether the value [a] reads a join over other writes of its storage,
         a state the trips write or a later one. *)
      let reads_join a =
        let seen = Tbl.create 8 in
        let rec go n =
          (not (Tbl.mem seen n))
          && begin
               Tbl.replace seen n ();
               (over_other_writes n && stale n && not (older n))
               || (op n <> Op.After && List.exists go (src_without_body n))
               || (op n = Op.After && go (nth n 0))
             end
        in
        go a
      in
      let refuse_join i a =
        if
          (not (List.mem i written))
          && ((unlifted a && stale a && not (older a)) || reads_join a)
        then
          invalid_arg
            (Format.asprintf
               "a call in a loop reads its argument %d through a join over \
                other writes of the storage it writes: the loop runs after \
                those writes, so storage holds them, not the join's value; \
                read a state that holds those writes, or read a copy of the \
                state made before them and write the copy in the call"
               i)
      in
      List.iteri refuse_join (src_without_body c);
      let trip_read n = stale n && not (older n) in
      (* The run each argument views: computed once for both uses below. *)
      let runs = lazy (Tbl.create 8) in
      let run_view a =
        let runs = Lazy.force runs in
        match Tbl.find_opt runs a with
        | Some v -> v
        | None ->
            (* Under states of another shape than their storage's, as a
               movement over a state of a stage, the chain is no run. *)
            let v =
              match run_of a (without_states a) with
              | v -> v
              | exception Invalid_argument _ -> None
            in
            Tbl.replace runs a v;
            v
      in
      let c' = absorb ctx c ~run_view ~trip_read in
      let written, stale =
        if c' == c then (written, stale)
        else
          ( Option.value (written_positions c') ~default:[],
            Shape.reads_state_written c' )
      in
      (* A stage of a value a loop moves, left among the arguments, is stored
         before the loop, over ranges no store before it holds. *)
      let moving_stage a =
        (not (has_buffer_identity ~after_ok:true a))
        && List.exists
             (fun n ->
               op n = Op.Stage
               && List.exists loops (Nodes.to_list (ranges (nth n 0))))
             (toposort ~calls:Skip a)
      in
      List.iter
        (fun a ->
          if moving_stage a then
            invalid_arg
              "a call in a loop reads a stored value that moves with a loop: \
               store it in the call's body")
        (src_without_body c');
      let run i a =
        let viewing =
          Op.Set.mem (op a) Op.Set.movement || op a = Op.Bitcast
          || (op a = Op.After && viewed_state a)
        in
        let moves = List.exists loops (Nodes.to_list (ranges a)) in
        if
          List.mem i written || (not viewing) || moves
          || not (Shape.reads_storage_written c' i)
          || op (named a) = Op.After
             && ((not (stale a)) || older (named a))
        then a
        else
          match run_view a with
          | None -> a
          | Some view -> reshape (loop_view ctx a view) (shape a)
      in
      let args = src_without_body c' in
      let args' = List.mapi run args in
      if c' == c && List.for_all2 ( == ) args args' then None
      else
        let c'' = replace c' ~src:(nth c' 0 :: args') in
        Some (replace k ~src:(c'' :: List.tl (src k)))

(* An {!Op.After} of storage listing only stores into other storage leaves the
   state it is over as it is: a join after those stores' states. Stated here,
   where a store still names the storage it writes. *)
let join_foreign_stores u =
  let storage a = whole (buf_uop a) in
  let foreign d =
    op d = Op.Store
    && nth d 0 != nth u 0
    && storage (nth d 0) != storage (nth u 0)
  in
  let deps = List.tl (src u) in
  if
    deps <> [] && List.for_all foreign deps
    && has_buffer_identity ~after_ok:true (nth u 0)
  then
    (* The stores into each storage, in the order listed: one write of it. *)
    let add groups d =
      let b = buf_uop (nth d 0) in
      match List.assq_opt b groups with
      | Some ds -> (b, d :: ds) :: List.remove_assq b groups
      | None -> groups @ [ (b, [ d ]) ]
    in
    let write = if is_disjoint u then after_disjoint else after in
    let joins =
      List.map
        (fun (b, ds) -> write b (List.rev ds))
        (List.fold_left add [] deps)
    in
    Some (after (nth u 0) joins)
  else None

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
          match join_foreign_stores u with
          | Some _ as joined -> joined
          | None ->
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
    ])

(* A state of a bitcast of storage is that bitcast of the state of the storage:
   the stores land in the storage whichever type a reader views it at. The
   bitcast goes outside the {!Op.After}, so that a state's first source is
   storage or a view of it. A movement over a bitcast of elements of one size is
   the bitcast of the movement; one over a bitcast that resizes them views
   elements only the bitcast's type holds, and goes outside with it. A store
   through a movement of a bitcast of one size stores the value bitcast into the
   movement of the storage; a copy keeps its bitcast, which the copy's view
   holds. A store through a bitcast that resizes elements is
   {!Prepare.prepare_rangeify}'s, into the run of whole elements it covers. *)
let rec through_bitcast s =
  match op s with
  | Op.Bitcast when Dtype.itemsize (dtype s) = Dtype.itemsize (dtype (nth s 0)) ->
      Some (nth s 0, dtype s)
  | o when Op.Set.mem o Op.Set.movement ->
      Option.map
        (fun (x, dt) -> (replace s ~src:(x :: List.tl (src s)), dt))
        (through_bitcast (nth s 0))
  | _ -> None

let rec under_bitcast s =
  match op s with
  | Op.Bitcast -> Some (nth s 0, fun y -> bitcast y (dtype s))
  | o when Op.Set.mem o Op.Set.movement -> (
      match under_bitcast (nth s 0) with
      | Some (x, f) ->
          Some (x, fun y -> replace s ~src:(f y :: List.tl (src s)))
      | None -> None)
  | _ -> None

let stored_through_bitcast u =
  op u = Op.Store
  && op (nth u 1) <> Op.Copy
  && Op.Set.mem (op (nth u 0)) Op.Set.movement
  && Option.is_some (through_bitcast (nth u 0))

let pm_state_of_bitcast =
  Pattern_matcher.v (fun () ->
      [
        rule (Upat.op Op.Store ~name:"s") (fun m ->
            let s = m "s" in
            if not (stored_through_bitcast s) then None
            else
              match through_bitcast (nth s 0) with
              | Some (x, _) ->
                  (* A weak value takes the view's type first: a bitcast
                     reads its operand at a stated width. *)
                  let value =
                    let v = nth s 1 in
                    if Dtype.mem (dtype v) Dtype.weaks then
                      cast v (dtype (nth s 0))
                    else v
                  in
                  Some
                    (replace s
                       ~src:
                         (x :: bitcast value (dtype x)
                         :: List.tl (List.tl (src s))))
              | None -> None);
        rule (Upat.op Op.After ~name:"a") (fun m ->
            let a = m "a" in
            match through_bitcast (nth a 0) with
            | Some (x, dt) ->
                Some (bitcast (replace a ~src:(x :: List.tl (src a))) dt)
            | None ->
                Option.map
                  (fun (x, rebuild) ->
                    rebuild (replace a ~src:(x :: List.tl (src a))))
                  (under_bitcast (nth a 0)));
      ])

let transform_to_call big_sink =
  if setting Setting.spec <> 0 then
    Spec.type_verify ~calls:Skip Spec.tensor big_sink;
  (* The stores are collected before these rewrites change node identities,
     and the buffers they write before any view is made. *)
  let ctx = callify_ctx () in
  let nodes = toposort ~calls:Skip big_sink in
  let big_sink, nodes =
    if
      List.exists
        (fun u ->
          stored_through_bitcast u
          || (op u = Op.After && Option.is_some (under_bitcast (nth u 0))))
        nodes
    then
      let s =
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() big_sink
          (After_sources pm_state_of_bitcast)
      in
      (s, toposort ~calls:Skip s)
    else (big_sink, nodes)
  in
  List.iter
    (fun u ->
      if is_store_after u && not (is_join u) then
        Tbl.replace ctx.written (storage_base u) ();
      if op u = Op.After && not (is_join u) then
        List.iter
          (fun d ->
            if is_loop d then
              Tbl.replace ctx.listed d
                (state_of (nth u 0)
                :: Option.value ~default:[] (Tbl.find_opt ctx.listed d)))
          (List.tl (src u)))
    nodes;
  (* A fixed run a loop's call reads is a view, read each trip. Made here, over
     the graph the listing was taken of, so that a loop is the node the
     {!Op.After}s listing it name; each loop's, earlier ones under its
     arguments too. *)
  let lifted =
    List.filter_map
      (fun k ->
        if is_loop k then Option.map (fun r -> (k, r)) (loop_runs ctx k)
        else None)
      nodes
  in
  let big_sink =
    if lifted = [] then big_sink
    else substitute ~calls:Skip ~pass:Fixed_point big_sink lifted
  in
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

   One scan over the finished schedule decides. Calls are weighed in order; a
   call joins the earliest group of its device that starts after every
   position it interferes with, so it moves up only past calls it does not
   interfere with, and only where no storage's life grows: past the first use
   of each storage it takes, unless the storage is given to the schedule, holds
   one element, or the group is in a loop's body, whose storage lives for the
   whole loop. At the top level a call of many elements joins only a group
   writing storage it writes: the schedule's order there is chosen for the
   storage calls make, and the pieces of one state are what a launch of many
   elements is for.

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

(* The entries [entries] with the calls that can run as one launched. [held]
   is the storage the schedule is given, which outlives it; [free] is whether
   every storage lives for the whole scope, as in a loop's body; [facts] holds
   the bodies met. *)
let launched_entries ~facts ~held ~free entries =
  let n = Array.length entries in
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
          if free || Tbl.mem held b || Shape.max_numel b = 1 then lo
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
            if free || f.one_element then
              let j = at_or_after gs bound in
              if j < Dynarray.length gs then Some (Dynarray.get gs j) else None
            else
              (* Groups of other devices write other storage. *)
              writing written bound
          in
          match picked with
          | Some g ->
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

(* [linear] with its calls launched, in its loops' bodies too. *)
let rec launch_linear ~facts ~held ~free linear =
  let entries =
    Array.of_list
      (List.map
         (fun e ->
           match op e with
           | (Op.End | Op.Backedge) when op (nth e 0) = Op.Linear ->
               let inner = launch_linear ~facts ~held ~free:true (nth e 0) in
               replace e ~src:(inner :: List.tl (src e))
           | _ -> e)
         (src linear))
  in
  let src =
    if Array.length entries < 2 then Array.to_list entries
    else launched_entries ~facts ~held ~free entries
  in
  replace linear ~src

let launches ~held linear =
  let tbl = Tbl.create 16 in
  List.iter (fun b -> Tbl.replace tbl b ()) held;
  launch_linear ~facts:(Tbl.create 16) ~held:tbl ~free:false linear

(* Schedules *)

let create_linear_with_vars ?(capturing = false) ?(by_storage = false) big_sink
    =
  let several = bound_several big_sink in
  let big_sink = transform_to_call big_sink in
  (* The call's sources are the values to realize. *)
  let linear_call =
    graph_rewrite ~calls:Enter ~pass:Fixed_point ~ctx:by_storage big_sink
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
  let linear = launches ~held:held_bufs linear in
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
