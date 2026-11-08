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
let is_range_of_size u = op u = Op.Range && op (nth u 0) <> Op.Const

let opts u =
  match arg u with
  | Bufferize o -> o
  | _ -> invalid_arg "a stage needs its buffer's options"

let sorted_ranges rs = List.sort_uniq compare_range_arg rs

let rec zip l0 l1 =
  match (l0, l1) with x :: r0, y :: r1 -> (x, y) :: zip r0 r1 | _ -> []

(* Cleanups *)

let transcendental = ops Op.[ Exp2; Log2; Sin; Pow ]

(* Whether an axis dies is only known once an expand to its left is seen. *)
let cleanup_dead_axes b =
  let value = nth b 0 in
  (* An after is storage: its ranges say how consumers read it. *)
  if (opts b).keep = Whole || op value = Op.After then None
  else
    let axes = zip (shape b) (List.tl (src b)) in
    let dead rng =
      op rng = Op.Const
      || (op rng = Op.Range && not (Nodes.mem rng (ranges value)))
    in
    if List.exists (fun (_, r) -> is_range_of_size r) axes then None
    else if not (List.exists (fun (_, r) -> dead r) axes) then None
    else
      let live = List.filter (fun (_, r) -> not (dead r)) axes in
      let reshape = List.map (fun (s, r) -> if dead r then Int 1 else s) axes in
      Some
        (expand
           (Shape.reshape (replace b ~src:(value :: List.map snd live)) reshape)
           (shape b))

let pm_gate_substitute =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.v ~op:Op.Set.all ~name:"b" ()) (fun ctx m ->
          let rs = ranges (m "b") in
          if Tbl.to_seq_keys ctx |> Seq.exists (fun r -> Nodes.mem r rs) then
            None
          else raise Bottom_up_gate);
    ])

(* The value [src] of the stage [buf] at the index [idx], which replaces the
   stage's ranges. A constant range is not replaced. *)
let inlined src buf idx =
  let replaced =
    List.filter
      (fun (k, _) -> op k <> Op.Const)
      (zip (List.tl (Ops.src buf)) (List.tl (Ops.src idx)))
  in
  substitute ~calls:Skip ~pass:Fixed_point ~extra_pm:pm_gate_substitute src
    replaced

(* Whether [x] reads storage, searched until the first read: a stage's value
   is new to the graph a question is asked of, so a memo over its whole slice
   would be built for one question. *)
let storage x =
  let reads_storage n =
    match op n with Op.Param | Op.Stage | Op.After -> true | _ -> false
  in
  let gate n = if reads_storage n then raise_notrace Exit else true in
  match toposort ~calls:Skip ~gate x with _ -> false | exception Exit -> true

(* Whether the sum [red] over one range is computed at one trip
   ({!Simplify.load_collapse}) as its kernel holds it: with its reads of stages
   a later pass may inline whose values read no storage, as a range's positions
   do, computed in place and simplified as the cleanups simplify them, so that a
   selection by one of them compares with the range itself. Only the part of
   the sum that runs in its range is rebuilt: each value it reads from outside
   stands for a variable bounded as that value is. *)
let collapses red =
  match src red with
  | [ u; r ] when op r = Op.Range ->
      let inside x = Nodes.mem r (ranges x) in
      let included = toposort ~calls:Skip ~gate:inside u in
      let local = Tbl.create 64 in
      List.iter (fun x -> Tbl.replace local x ()) included;
      let index_function x =
        op x = Op.Index
        && op (nth x 0) = Op.Stage
        && (opts (nth x 0)).keep <> Whole
        && not (storage (nth (nth x 0) 0))
      in
      let subs = Tbl.create 16 in
      let stand_for s =
        if
          not
            (Tbl.mem local s || Tbl.mem subs s || op s = Op.Const
           || op s = Op.Range
            || Op.Set.mem (op s) Op.Set.defines)
        then
          let name = Printf.sprintf "in%d" (Tbl.length subs) in
          Tbl.replace subs s (variable ~dtype:(dtype s) name (vmin s) (vmax s))
      in
      List.iter
        (fun x ->
          if index_function x then
            Tbl.replace subs x (inlined (nth (nth x 0) 0) (nth x 0) x)
          else List.iter stand_for (src x))
        included;
      (* Over the variables, a reduction nested in the sum can rewrite without
         end, as an int64 fold of rune's over a padded window does; it raises
         [Invalid_argument], and the sum then counts its trips. *)
      (match
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
          (substitute ~calls:Skip ~pass:Fixed_point u
             (List.of_seq (Tbl.to_seq subs)))
          (After_sources
             (Pattern_matcher.concat
                [ Shape.symbolic; Simplify.pm_reduce_simplify ]))
      with
      | exception Invalid_argument _ -> false
      | body ->
          Option.is_some (Simplify.load_collapse (replace red ~src:[ body; r ])))
  | _ -> false

(* The product of the largest sizes of the ranges among [rs], a float since it
   can pass [max_int]. *)
let extent rs =
  let size n r =
    if op r <> Op.Range then n
    else n *. Dtype.Value.to_float (vmax (nth r 0))
  in
  List.fold_left size 1. rs

(* The ranges the indices [xs] run in, each once. *)
let read_ranges xs =
  let add r rs = if List.memq r rs then rs else r :: rs in
  List.fold_left (fun rs x -> Nodes.fold add (ranges x) rs) [] xs

(* Whether [x] reads [r] through floor divisions and remainders by positive
   constants alone, each dividing the size of what it is applied to:
   [(r / 16) mod 4] of a range of 1024. *)
let rec digit_of r x =
  x == r
  ||
  match op x with
  | (Op.Div | Op.Mod) when op (nth x 1) = Op.Const -> (
      let size = Bigint.succ (Dtype.Value.to_z (vmax (nth x 0))) in
      match Ops.value (nth x 1) with
      | `Int c when Bigint.sign c > 0 ->
          Bigint.divisible size c && digit_of r (nth x 0)
      | _ -> false)
  | _ -> false

(* The copies a read makes: [no] where its sum over the range [sum] is computed
   at every trip, [yes] where at one, [no] alone where it reads no such sum or
   the answer is known. *)
type share = { no : float; yes : float; sum : Ops.t option }

(* The copies of the reads over an axis: [lo] and [hi] their sums over the
   answers of the tests they leave pending. [shares] holds the reads latest
   first; every sum adds them earliest first, the order of the reads, so a
   sum with no test pending is the same float whatever is asked. *)
type axis_copies = { lo : float; hi : float; shares : share list }

(* The copies the read [idx] of the stage [buf] makes of the stage's value once
   inlined: one, or one per trip of the reductions its index runs in, as a
   running sum's windows of [w] compute each element they read [w] times. A
   trip is counted only where it computes an element again, which storing would
   save, so the count is at most the trips of the index per element: a sum that
   reads each element once computes each once. A read whose index runs in no
   reduction, as a broadcast through an expand, a reshape or padding, copies
   the value once. Trips in padding count, since only the inlined value's loads
   are gated there. A sum that selects one trip, where an index equals its
   range, is computed at that index alone ({!Simplify.load_collapse}), once for
   each point of the ranges the index runs in: its range counts as those
   ranges, the ranges its selecting conditions read besides it, as a read at
   the selected index would, and a factor over another range inside the sum
   broadcasts the selected value as it would that read. That holds but for a
   stage that reads no storage, which the sum's test sees as its
   value ({!collapses}), as a one-hot gather's positions. A coordinate the
   value does not vary along
   counts neither elements nor trips: substituting its index leaves the value as
   it is. An output range of constant size that each component reading it reads
   through one chain of floor divisions and remainders by constants dividing
   what they apply to ({!digit_of}), in components reading no other range,
   repeats the value across the outputs that differ only in the digits the
   chains drop, which read it at one index: [r / 16] of the rows of 16-row
   blocks, or [r mod n] of rows merged with [n] columns. The codegen splits a
   range at a remainder by a constant dividing it and shares a quotient's
   repeats in its tiles, as it shares a broadcast, so that reuse, the extent of
   such ranges over the extent of the axes their components index, is no copy,
   unless another component, or a sum computed at one trip, also runs in the
   range. Reduction ranges keep their count. A symbolic size counts at its
   largest.

   [selected r] is the ranges the sum over [r] is computed at when it selects
   one trip, if a reduction closes [r], and [collapses r] whether that sum is
   computed at one trip. A read of one
   such range leaves the test pending, with its copies either way, for the
   stage's decision to ask only where the answer changes it; a read of several
   asks it now. *)
let read_copies ~reduced ~selected ~collapses buf idx =
  let live =
    List.filter
      (fun (r, _) -> op r = Op.Range && Nodes.mem r (ranges (nth buf 0)))
      (zip (List.tl (src buf)) (List.tl (src idx)))
  in
  let rs = read_ranges (List.map snd live) in
  let elements = extent (List.map fst live) in
  let chained (_, x) =
    match Nodes.to_list (ranges x) with
    | [ r ] ->
        x != r && op (nth r 0) = Op.Const && (not (reduced r)) && digit_of r x
    | _ -> false
  in
  let pure, others = List.partition chained live in
  let digits = read_ranges (List.map snd pure) in
  let counted = read_ranges (List.map snd others) in
  let digit d = List.memq d digits in
  let reuse = Float.max 1. (extent digits /. extent (List.map fst pure)) in
  (* The copies when the sums over the ranges of [collapsing] are computed at
     one trip, each range then counting as its sum's [selected]. *)
  let count collapsing =
    let add rs r = if List.memq r rs then rs else r :: rs in
    let read =
      List.fold_left
        (fun rs r ->
          match List.assq_opt r collapsing with
          | None -> add rs r
          | Some selected -> List.fold_left add rs selected)
        [] rs
    in
    let reduce = List.filter reduced read in
    let shared =
      if List.exists digit counted then 1.
      else if List.exists (fun (_, o) -> List.exists digit o) collapsing then 1.
      else reuse
    in
    if elements <= 0. then 1.
    else
      Float.max 1.
        (Float.min (extent reduce) (extent read /. elements /. shared))
  in
  (* A value that reads no storage is seen as itself by the test of a sum
     that selects one trip, which then compares the index it selects with
     the range: its reads count the sum's own trips, as one that does not
     select, so that it is inlined as that test assumes. *)
  let sums =
    List.filter_map (fun r -> Option.map (fun o -> (r, o)) (selected r)) rs
  in
  let sums = if sums <> [] && not (storage (nth buf 0)) then [] else sums in
  let known c = { no = c; yes = c; sum = None } in
  match sums with
  | [] -> known (count [])
  | [ ((r, _) as sum) ] ->
      let no = count [] and yes = count [ sum ] in
      if no = yes then known no else { no; yes; sum = Some r }
  | _ -> known (count (List.filter (fun (r, _) -> collapses r) sums))

(* A range's axis: its type and identity, which name it while the cleanups
   rebuild the nodes around it. *)
let axis r = (axis_type r, axis_id r)

(* The copies the reads of the stages of a graph make of their values once
   inlined ({!read_copies}): [by_axis] those of the stages over each range
   axis. Reads at several distinct indices copy the value at each. The axes of
   a stage's ranges are its own and name it while the cleanups rebuild it on
   its rewritten value. [reduce_axes] holds the axes of the ranges a reduction
   closes, with the reduction, [selections] the ranges each reduction's sum
   is computed at when it selects one trip, and [collapsed] whether it does,
   each once asked. *)
type reads = {
  by_axis : (Axis_type.t * int list, axis_copies) Hashtbl.t;
  reduce_axes : (Axis_type.t * int list, Ops.t) Hashtbl.t;
  selections : (Axis_type.t * int list, Ops.t list) Hashtbl.t;
  collapsed : (Axis_type.t * int list, bool) Hashtbl.t;
}

(* Whether the range [r] is one a reduction of the graph [reads] was taken from
   closes. *)
let reduced reads r = op r = Op.Range && Hashtbl.mem reads.reduce_axes (axis r)

(* The ranges the sum over [r] is computed at when it selects one trip, if [r]
   is a range that a reduction of the graph [reads] was taken from closes:
   those its conditions that compare [r] with a value of other ranges read
   besides it, from which the index it selects is computed, or, with no such
   condition, every range it runs in. A condition of [r] alone, as a padding
   gate's, selects nothing.
   [where (ids (j) = e) (v (e, d) * y (m)) 0] summed over [e] is computed at
   [v (ids (j), d)], once a point of [j] and [d]: [y (m)] broadcasts it along
   [m], as it would a read at that index. *)
let selected reads r =
  if op r <> Op.Range then None
  else
    let k = axis r in
    match Hashtbl.find_opt reads.selections k with
    | Some s -> Some s
    | None ->
        Option.map
          (fun red ->
            let inside x = Nodes.mem r (ranges x) in
            let conditions =
              List.filter_map
                (fun x ->
                  let c = nth x 0 in
                  let others =
                    Nodes.fold (fun y found -> found || y != r) (ranges c) false
                  in
                  if op x = Op.Where && inside c && others then Some c
                  else None)
                (toposort ~calls:Skip ~gate:inside (nth red 0))
            in
            let s =
              if conditions = [] then
                Nodes.fold (fun x rs -> x :: rs) (ranges red) []
              else List.filter (fun x -> x != r) (read_ranges conditions)
            in
            Hashtbl.replace reads.selections k s;
            s)
          (Hashtbl.find_opt reads.reduce_axes k)

(* Whether the sum over [r], a range that a reduction of the graph [reads] was
   taken from closes, is computed at one trip ({!Simplify.load_collapse}). *)
let collapsed reads r =
  let k = axis r in
  match Hashtbl.find_opt reads.collapsed k with
  | Some c -> c
  | None ->
      let c = collapses (Hashtbl.find reads.reduce_axes k) in
      Hashtbl.replace reads.collapsed k c;
      c

(* The reads of the stages of [tsink]. A read through a stack is not counted: by
   now every stage has a device, so the cleanups never turn it into a read of
   the stage. *)
let stage_reads tsink =
  let by_axis = Hashtbl.create 16 and seen = Tbl.create 64 in
  let reduce_axes = Hashtbl.create 16 in
  let count c r =
    if op r = Op.Range then
      let k = axis r in
      let a =
        Option.value
          (Hashtbl.find_opt by_axis k)
          ~default:{ lo = 0.; hi = 0.; shares = [] }
      in
      Hashtbl.replace by_axis k
        {
          lo = a.lo +. Float.min c.no c.yes;
          hi = a.hi +. Float.max c.no c.yes;
          shares = c :: a.shares;
        }
  in
  (* Each node is pushed once, outside call bodies, in no order. *)
  let push stack u =
    if Tbl.mem seen u then stack
    else begin
      Tbl.replace seen u ();
      u :: stack
    end
  in
  let rec gather nodes = function
    | [] -> nodes
    | u :: rest ->
        if op u = Op.Reduce then
          List.iter
            (fun r ->
              if op r = Op.Range then Hashtbl.replace reduce_axes (axis r) u)
            (List.tl (src u));
        gather (u :: nodes) (List.fold_left push rest (src_without_body u))
  in
  let nodes = gather [] (push [] tsink) in
  let reads =
    {
      by_axis;
      reduce_axes;
      selections = Hashtbl.create 16;
      collapsed = Hashtbl.create 16;
    }
  in
  List.iter
    (fun u ->
      if op u = Op.Index && op (nth u 0) = Op.Stage then begin
        let buf = nth u 0 in
        let c =
          read_copies ~reduced:(reduced reads) ~selected:(selected reads)
            ~collapses:(collapsed reads) buf u
        in
        List.iter (count c) (List.tl (src buf))
      end)
    nodes;
  reads

(* The least over the axes of the stage [buf]'s ranges of the copies [f] takes
   of each axis's reads: each read of a stage counts at every axis of its
   ranges, and other stages can share an axis or one a symbolic size names, so
   the least count bounds the stage's copies from above. *)
let least_copies f reads buf =
  let least n r =
    if op r <> Op.Range then n
    else
      match Hashtbl.find_opt reads.by_axis (axis r) with
      | None -> Float.min n 0.
      | Some a -> Float.min n (f a)
  in
  List.fold_left least Float.infinity (List.tl (src buf))

(* The copies of a least count [n]: a stage of no ranges, whose least is
   infinite, is read at one index, once. *)
let copies_of n = if n = Float.infinity then 1. else Float.max n 1.

(* The copies of the stage [buf], asking the tests its reads leave pending on
   an axis only where the bounds of that axis differ. *)
let stage_copies reads buf =
  let share c =
    match c.sum with Some r when collapsed reads r -> c.yes | _ -> c.no
  in
  let exact a =
    if a.lo = a.hi then a.lo
    else List.fold_right (fun c n -> n +. share c) a.shares 0.
  in
  copies_of (least_copies exact reads buf)

(* The least and most copies of the stage [buf] over the answers of the tests
   its reads leave pending, if the copies are finite: a sum of reads and a least
   of sums grow with each read's copies. A read of an infinite extent over
   infinitely many elements counts NaN copies and makes both bounds NaN; a
   finite [hi] rules it out. *)
let stage_bounds reads buf =
  let hi = least_copies (fun a -> a.hi) reads buf in
  if not (Float.is_finite hi) then None
  else
    let lo = least_copies (fun a -> a.lo) reads buf in
    Some (copies_of lo, copies_of hi)

(* The nodes that inlining a stage may compute again: the nodes of its value
   that its ranges reach, which the inlining copies, times the copies its reads
   make beyond the one storing computes. *)
let inline_budget = 360.

(* The nodes of [computed] that inlining [buf] copies: those inside its ranges,
   which the substitution of the read's indices rebuilds. *)
let copied buf computed =
  let rs = List.filter (fun r -> op r = Op.Range) (List.tl (src buf)) in
  let inside x =
    let xs = ranges x in
    List.exists (fun r -> Nodes.mem r xs) rs
  in
  List.fold_left (fun n x -> if inside x then n + 1 else n) 0 computed

(* A buffer stored only to be read through movements is removed: the indices of
   the read are expressed in terms of the stored value's. *)
let remove_bufferize ~reads src buf idx =
  if List.compare_lengths (Ops.src buf) (Ops.src idx) <> 0 then
    invalid_arg "an index of a stage has one index per range";
  if
    not
      (List.for_all
         (fun x -> op x = Op.Range || op x = Op.Const)
         (List.tl (Ops.src buf)))
  then invalid_arg "a stage's ranges are ranges or constants";
  (* A user's materialisation is never removed. *)
  if (opts buf).keep = Whole then None
  else
    (* The cost: the buffers the value reads, and whether a reduction reads
       one. *)
    let accessed = Tbl.create 8 in
    let access u = Tbl.replace accessed u () in
    let red_gate x =
      match op x with
      | Op.After ->
          access (buf_uop x);
          false
      | Op.Stage when (opts x).addrspace = Dtype.Global ->
          access x;
          false
      (* Stores do not count as buffer accesses. *)
      | Op.Store -> false
      | Op.Param ->
          access x;
          true
      | _ -> true
    in
    let computed = toposort ~calls:Skip ~gate:red_gate src in
    let reduces = List.filter (fun x -> op x = Op.Reduce) computed in
    (* Inlined where it is broadcast, the value would be computed again for each
       element of the ranges it does not vary along: one whose computing runs a
       transcendental function stays stored. Its producers are already stored
       or inlined, so the walk sees what inlining would compute. *)
    let recomputes =
      (opts buf).keep = Broadcast
      && List.exists (fun x -> Op.Set.mem (op x) transcendental) computed
    in
    (* Read by every output of a reduction, as a matrix-vector product reads its
       vector, the value would be computed again for each output and each of its
       buffers loaded again: one that reads more than one buffer stays
       stored. *)
    let read_by_every_output =
      let read = read_ranges (List.tl (Ops.src idx)) in
      (opts buf).keep = Broadcast
      && read <> []
      && List.for_all (reduced reads) read
      && Tbl.length accessed > 1
    in
    (* Inlined, the value is copied to each position it is read at, and
       computed again at each trip of a reduction over a window. Its producers
       are inlined already, so the copies compound: [k] values each reading the
       one before at two positions are [2^k] copies of the first. A value whose
       copies beyond the first exceed the budget stays stored, so that an
       inlining adds at most the budget to the work of its readers, and to
       their code, which each copy at a distinct index adds to and a trip does
       not; the kernels of a program then grow at most linearly in it. A view
       of storage, a read whose index computes only positions, is loaded where
       it is read however many copies its reads make: stored, it would be a
       copy of storage its readers can load in place. Positions are weak
       integers and their comparisons on no device; a node on a device, as a
       position loaded from storage, or of another type computes. *)
    let computes x =
      let positional =
        Option.is_none (device x)
        && List.exists (Dtype.equal (dtype x)) Dtype.[ Weak_int; Bool ]
      in
      (Op.Set.mem (op x) Op.Set.elementwise
      || List.mem (op x) Op.[ Reduce; Call ])
      && not positional
    in
    let view = op src = Op.Index && not (List.exists computes computed) in
    let copies () =
      (* The copied nodes are among the computed ones, so most values are
         under the budget before their copies are counted. *)
      let n = float_of_int (List.length computed) in
      let copied = lazy (float_of_int (copied buf computed)) in
      let exceeds c =
        n *. (c -. 1.) > inline_budget
        && Lazy.force copied *. (c -. 1.) > inline_budget
      in
      (* [exceeds] grows with the copies: where the bounds agree, no pending
         test is asked. *)
      match stage_bounds reads buf with
      | Some (lo, hi) when exceeds lo = exceeds hi -> exceeds lo
      | _ -> exceeds (stage_copies reads buf)
    in
    if
      Tbl.length accessed > 3
      || recomputes || read_by_every_output
      || ((not view) && copies ())
    then None
    else
      (* A call returns no value here: a reduction reaches one only through the
         [After] it orders, so the operations kept outside call bodies answer as
         a walk into them does. *)
      let reads_buffer r =
        op_in_backward_slice_with_self ~calls:Skip (nth r 0)
          Op.[ Param; Stage; After ]
      in
      let dead =
        List.exists
          (fun v -> op v = Op.Const && is_invalid v)
          (List.tl (Ops.src idx))
      in
      if List.exists reads_buffer reduces then None
      else if dead then
        (* A read at an invalid index is dead, and reads 0 as a load at one
           does. The value inlined there would keep the stage's ranges, which
           no index replaces and no kernel ends. *)
        Some (const_like idx (`Int Bigint.zero))
      else Some (inlined src buf idx)

let remove_noop_bufferize idx b2 =
  if not (List.equal ( == ) (List.tl (src idx)) (List.tl (src b2))) then None
  else
    match shape b2 with
    | [] -> Some (nth idx 0)
    | s -> Some (shrink (nth idx 0) (List.map (fun s -> Some (Int 0, s)) s))

let after_all_invalid after =
  let buf = buf_uop (nth after 0) in
  let all_invalid s =
    op s = Op.End
    &&
    let st = nth s 0 in
    let ended = ended_ranges s in
    op st = Op.Store
    && is_invalid (base (nth st 1))
    && buf_uop (nth st 0) == buf
    && List.for_all (fun r -> Nodes.mem r (ranges (nth st 0))) ended
    && resolve ~default:false
         (eq
            (List.fold_left (fun p r -> mul p (nth r 0)) (int 1) ended)
            (sint_to_uop (numel buf)))
  in
  List.for_all all_invalid (List.tl (src after))

let pm_const_buffer_folding =
  Pattern_matcher.append (with_ctx Prepare.pm_mops)
    (Pattern_matcher.v
       (fun () -> [
         rule (Upat.op Op.Stage ~name:"b") (fun m -> cleanup_dead_axes (m "b"));
         (* A stage of an index by the stage's own ranges is the storage. *)
         rule
           (Upat.f
              (Upat.op Op.Index ~name:"idx")
              Op.Stage ~allow_any_len:true ~name:"b2")
           (fun m -> remove_noop_bufferize (m "idx") (m "b2"));
         (* A constant needs no buffer, in either spelling. *)
         rule
           (Upat.f
              (Upat.or_casted (Upat.cvar "c"))
              Op.Stage ~allow_any_len:true ~name:"b")
           (fun m -> Some (const_like (m "b") (value (m "c"))));
         (* An index of a constant is the constant. *)
         rule
           (Upat.op Op.Index ~src:[ Upat.or_casted ~name:"c" (Upat.cvar "k") ])
           (fun m -> Some (m "c"));
         (* An index of storage whose every store is invalid is invalid. *)
         rule
           (Upat.op Op.Index ~name:"idx" ~allow_any_len:true
              ~src:[ Upat.op Op.After ~name:"after" ])
           (fun m ->
             if after_all_invalid (m "after") then
               Some (const_like (m "idx") `Invalid)
             else None);
       ]))

(* The store of [value] at [idx] of the stage [buf] of [src], into [src]'s
   storage. A user's materialisation is written as asked. *)
let destination src buf idx value =
  let dead () =
    List.exists
      (fun v -> op v = Op.Const && is_invalid v)
      (List.tl (Ops.src idx))
  in
  if (opts buf).keep = Whole || dead () then None
  else Some (store (inlined src buf idx) value)

let pm_remove_bufferize =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx
        (Upat.f
           (Upat.f (var "src") Op.Stage ~allow_any_len:true ~name:"buf")
           Op.Index ~allow_any_len:true ~name:"idx")
        (fun reads m -> remove_bufferize ~reads (m "src") (m "buf") (m "idx"));
      (* A store writes its destination's storage. Indexing stages a view that
         several stores write through, as it stages any value its consumers
         index differently; kept, the stage would take the stores and the
         storage would hold what it held. So a stage a store writes through is
         inlined whatever its copies cost. *)
      rule
        (Upat.store
           (Upat.f
              (Upat.f (var "src") Op.Stage ~allow_any_len:true ~name:"buf")
              Op.Index ~allow_any_len:true ~name:"idx")
           [ var "v" ])
        (fun m -> destination (m "src") (m "buf") (m "idx") (m "v"));
      (* A store of a value into itself does nothing. *)
      rule (Upat.store (var "x") [ var "x" ]) (fun _ -> Some (v Op.Noop));
      rule
        (Upat.op Op.End ~allow_any_len:true ~src:[ Upat.op Op.Noop ~name:"x" ])
        (fun m -> Some (m "x"));
    ])

let strip_zero_offset_shrink x =
  match op x with
  | Op.Shrink -> (
      match marg x with
      | Shrink b when List.for_all (fun (s, _) -> Sint.equal s (Int 0)) b ->
          nth x 0
      | _ -> x)
  | _ -> x

(* A call's arguments that have consumers can be indexed; the call reads its
   storage. *)
let no_indexing_calls u =
  let arg x =
    match op x with
    | Op.Index -> nth x 0
    | Op.Shrink -> strip_zero_offset_shrink x
    | _ -> x
  in
  replace u ~src:(List.map arg (src u))

let pm_no_indexing_calls =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Call ~name:"u") (fun m ->
          Some (no_indexing_calls (m "u")));
    ])

let loop_range r = Axis_type.equal (axis_type r) Axis_type.Loop

(* The kernel graph is what runs: it has no views, and a value's storage is the
   storage. A view that moves with a range is what a call in that range reads on
   each trip, and stays. *)
let pm_no_views =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v
           ~op:(ops Op.[ Reshape; Shrink ])
           ~name:"v" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(Op.Set.union Op.Set.defines (ops [ Op.After ]))
                 ();
             ]
           ())
        (fun m ->
          let v = m "v" in
          if List.exists loop_range (Nodes.to_list (ranges v)) then None
          else Some (nth v 0));
    ])

module Bufs = Set.Make (struct
  type t = Ops.t

  let compare = Ops.compare
end)

type limit_bufs_ctx = { buf_cache : Bufs.t Tbl.t; mutable range_idx : int }

(* [s] stored first, over fresh weak ranges, and read back at its own. *)
let stage ctx s =
  let orig = Nodes.to_list (ranges s) in
  let fresh r =
    let n = ctx.range_idx in
    ctx.range_idx <- n + 1;
    replace r ~arg:(Range { axis_type = Axis_type.Weak; axis_id = [ n ] })
  in
  let ends = List.map fresh orig in
  let opts = { device = device s; addrspace = Dtype.Global; keep = Removable } in
  index
    (bufferize ~opts
       (substitute ~calls:Skip ~pass:Fixed_point s (List.combine orig ends))
       ends)
    orig

let limit_bufs ctx root =
  let max_bufs = Setting.value Setting.max_kernel_buffers in
  (* Without a device, the node computes indices. *)
  if Option.is_none (device root) || max_bufs = 0 then None
  else
    let visitor u =
      match (op u, src u) with
      | (Op.Stage | Op.After | Op.Param), _ ->
          Bufs.singleton u
      | _, [ s ] -> Tbl.find ctx.buf_cache s
      | _, srcs ->
          List.fold_left
            (fun acc s -> Bufs.union acc (Tbl.find ctx.buf_cache s))
            Bufs.empty srcs
    in
    (* One buffer is the output. *)
    if Bufs.cardinal (topovisit root visitor ctx.buf_cache) <= max_bufs - 1 then
      None
    else
      let each s =
        if Op.Set.mem (op s) Op.Set.elementwise && Option.is_some (device s)
        then stage ctx s
        else s
      in
      Some (replace root ~src:(List.map each (src root)))

(* A store whose value reads its destination at another index than the one it
   writes stores a stage of the value: the store would overwrite elements the
   value reads. A stage or a store behind an [After] is read whole already. *)
let reads_of dst value =
  let gate u = not (op u = Op.Stage || op u = Op.After) in
  List.filter
    (fun u -> op u = Op.Index && buf_uop u == buf_uop dst)
    (toposort ~calls:Skip ~gate value)

let fix_store_hazard ctx dst value =
  let at = get_idx (nth dst 1) in
  if List.exists (fun u -> get_idx (nth u 1) != at) (reads_of dst value) then
    Some (store dst (stage ctx value))
  else None

let pm_limit_bufs =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx
        (Upat.v ~op:(Op.Set.union Op.Set.binary Op.Set.ternary) ~name:"root" ())
        (fun ctx m -> limit_bufs ctx (m "root"));
      rule_ctx
        (Upat.op Op.Store ~src:[ Upat.op Op.Index ~name:"dst"; Upat.var "src" ])
        (fun ctx m -> fix_store_hazard ctx (m "dst") (m "src"));
    ])

(* Buffers *)

(* Whether [x] converts its source to a wider type that holds every value of
   it, as bfloat16 to float32. *)
let widens x =
  op x = Op.Cast
  &&
  let s = Ops.dtype (nth x 0) and d = Ops.dtype x in
  (not (Dtype.equal s Dtype.Bool))
  && (not (Dtype.mem s Dtype.weaks))
  && (not (Dtype.mem d Dtype.weaks))
  && Dtype.itemsize s < Dtype.itemsize d
  && Dtype.can_lossless_cast s d

let bufferize_to_store ctx x idx =
  let dtype = commit_dtype x in
  let rngs = sorted_ranges (Nodes.to_list (ranges idx)) in
  (match numel x with
  | Int n when (n > 0) [@mutate off "Prepare removes empty values"] -> ()
  | _ -> invalid_arg "a stage has no elements or a symbolic size");
  let value = nth x 0 in
  if op value = Op.After then
    (* The stores of the storage end over the stage's ranges, and the storage is
       read after them. *)
    let buf = base (buf_uop (nth value 0)) in
    let stores =
      List.filter
        (fun s -> op s = Op.Store && op (nth s 0) = Op.Index)
        (List.tl (src value))
    in
    if stores = [] then Some buf
    else
      let end_store st =
        let target = nth st 0 in
        (* A stage of an index stores through the underlying index. *)
        let target =
          if
            op (nth target 0) = Op.Stage && op (nth (nth target 0) 0) = Op.Index
          then nth (nth target 0) 0
          else target
        in
        if nth st 1 == target then None
        else
          let ends = sorted_ranges (Nodes.to_list (ranges target) @ rngs) in
          Some (end_ (store target (nth st 1)) ends)
      in
      Some (after buf (List.filter_map end_store stores))
  else
    match opts x with
    | { addrspace = Dtype.Global; device; keep } ->
        (* A value its kernels read widened from a narrower type is stored
           narrow, and each read widens it: the stores and reads move half the
           bytes of bfloat16 widened to float32, and a reader's float32 product
           of such values is one of bfloat16 values, which a tensor core
           multiplies. A value stored whole keeps its type. *)
        let value, dtype =
          if keep <> Whole && widens value then
            (nth value 0, commit_dtype (nth value 0))
          else (value, dtype)
        in
        let slot = !ctx in
        incr ctx;
        let vmin_vmax = stored_bounds value dtype in
        let buf = alloc ~slot ?vmin_vmax ?device [ Int (max_numel x) ] dtype in
        let do_store =
          end_ (store (index buf [ idx ]) (cast value dtype)) rngs
        in
        Some (cast (after buf [ do_store ]) (Ops.dtype x))
    | _ -> None

(* A stage over several ranges is a stage over their flat index, reshaped. *)
let flatten_bufferize x =
  match src x with
  | [ _; _ ] -> None
  | value :: rngs ->
      let flat =
        Helpers.get_single_element
          (Indexing.apply_movement_op [ numel x ] (Reshape (shape x)) rngs)
      in
      let ret = reshape (replace x ~src:[ value; flat ]) (shape x) in
      if List.exists is_range_of_size rngs then
        let size r = if op r = Op.Const then Int 1 else Sym (nth r 0) in
        Some (shrink ret (List.map (fun r -> Some (Int 0, size r)) rngs))
      else Some ret
  | [] -> None

(* Whether an effect an After waits on does nothing: a noop, the empty stack
   that a group of stores leaves once each store is dropped, as a store of a
   value into itself is, or an end of one or an After of one on ranges
   alone. *)
let rec is_noop_after_dep x =
  (op x = Op.Noop && src x = []) [@mutate off "a noop has no sources"]
  || (op x = Op.Stack && src x = [])
  || (op x = Op.End || op x = Op.After)
     && is_noop_after_dep (nth x 0)
     && List.for_all (fun s -> op s = Op.Range) (List.tl (src x))

let remove_noop_afters x =
  match src x with
  | first :: deps ->
      let kept = List.filter (fun s -> not (is_noop_after_dep s)) deps in
      if List.compare_lengths kept deps = 0 then None
      else if kept = [] then Some first
      else Some (replace x ~src:(first :: kept))
  | [] -> None

let pm_add_buffers =
  Pattern_matcher.concat
    [
      with_ctx Prepare.pm_mops;
      Pattern_matcher.v
        (fun () -> [
          rule (Upat.op Op.Stage ~name:"x") (fun m -> flatten_bufferize (m "x"));
        ]);
      Pattern_matcher.v
        (fun () -> [
          rule_ctx
            (Upat.op Op.Stage ~name:"x" ~src:[ Upat.wild; Upat.var "idx" ])
            (fun ctx m -> bufferize_to_store ctx (m "x") (m "idx"));
          (* An index of a buffer through the weak cast added above indexes the
             buffer and casts the value read. This must run in the rewrite that
             adds the cast, or the expander expands the whole cast buffer into
             one vector. *)
          rule
            (Upat.op Op.Index ~name:"u" ~allow_any_len:true
               ~src:[ Upat.op Op.Cast ~dtype:Dtype.weaks ~src:[ var "buf" ] ])
            (fun m ->
              let u = m "u" in
              Some
                (cast (replace u ~src:(m "buf" :: List.tl (src u))) (dtype u)));
          (* So does an index of a narrow store through its widening. *)
          rule
            (Upat.op Op.Index ~name:"u" ~allow_any_len:true
               ~src:[ Upat.op Op.Cast ~src:[ Upat.op Op.After ~name:"buf" ] ])
            (fun m ->
              let u = m "u" in
              Some
                (cast (replace u ~src:(m "buf" :: List.tl (src u))) (dtype u)));
          (* A kernel's arguments lose their reshapes. *)
          rule (Upat.op Op.Call ~name:"k") (fun m ->
              let k = m "k" in
              Some
                (replace k
                   ~src:
                     (List.map
                        (fun x -> if op x = Op.Reshape then nth x 0 else x)
                        (src k))));
          (* Invalid writes are dropped. *)
          rule
            (Upat.op Op.Store
               ~src:[ Upat.wild; Upat.op Op.Const ~arg:(Const `Invalid) ])
            (fun _ -> Some (v Op.Noop));
          rule (Upat.op Op.After ~name:"x") (fun m ->
              remove_noop_afters (m "x"));
        ]);
    ]

(* Kernels *)

(* A kernel's storage, each with the argument its call passes, and its ranges,
   in the order the kernel reaches them. *)
type local_ctx = {
  map : t Tbl.t;
  mutable order : t list; (* The keys of [map], last first. *)
  params : t Tbl.t; (* The key of [map] standing for each state. *)
  joined : t list Tbl.t;
      (* What the joins the kernel reads a state through wait for, by key. *)
  ranges : (Axis_type.t * int list, unit) Hashtbl.t;
  mutable range_order : (Axis_type.t * int list) list; (* last first *)
}

let add_arg ctx buf value =
  if not (Tbl.mem ctx.map buf) then begin
    Tbl.replace ctx.map buf value;
    ctx.order <- buf :: ctx.order
  end

(* A kernel takes a parameter per state of storage it reads or writes, the
   argument its call passes being that state. Storage stands for the state it
   is; any other state of it, an {!Op.After}, for a node of its own, storage
   in a slot of no storage's, which [debuf] makes a parameter as it makes
   storage one. A kernel taking two states of one storage reads one before an
   overwrite it waits for: the schedule cuts it, and passes it a copy of that
   state. *)
let stand_in ctx state =
  match Tbl.find_opt ctx.params state with
  | Some p -> p
  | None ->
      let buf = buf_uop state in
      let p =
        match arg buf with
        | Param a when state != buf ->
            replace buf
              ~arg:(Param { a with slot = min_int + Tbl.length ctx.params })
        | _ -> buf
      in
      Tbl.replace ctx.params state p;
      add_arg ctx p state;
      p

(* Storage is a parameter of the kernel: its slot is set once the kernel's
   storage is all known, and its sources are not the kernel's. *)
let add_buf ctx buf =
  ignore (stand_in ctx buf);
  raise Bottom_up_gate

let debuf slot buf =
  let align, phase = storage_phase buf in
  (* The kernel reads the storage's values within their bounds: an index
     clamped where it is stored is in range where a gather reads it. *)
  let vmin_vmax =
    match arg buf with Param p -> p.vmin_vmax | _ -> None
  in
  let param =
    Shape.param ~shape:[ Int (max_numel buf) ] ?vmin_vmax
      ?addrspace:(addrspace buf) ~phase ~align slot
      (dtype buf)
  in
  let ret = reshape param (List.map (fun n -> Int n) (max_shape buf)) in
  (* A buffer of symbolic shape is its greatest view, shrunk. *)
  if
    List.equal Sint.equal (List.map (fun n -> Int n) (max_shape buf)) (shape buf)
  then ret
  else shrink ret (List.map (fun s -> Some (Int 0, s)) (shape buf))

(* The state under the joins [u] reads through, and what they wait for. *)
let unjoin u =
  let rec go deps u =
    if is_join u then go (List.rev_append (List.tl (src u)) deps) (nth u 0)
    else (u, List.rev deps)
  in
  go [] u

(* A group of stores is no state of a buffer. A join is none either: the
   kernel reads the state under it, and its call's argument waits for what the
   join waits for. *)
let handle_after ctx after =
  if addrspace after = Some Dtype.Local || is_group after then None
  else if addrspace after = Some Dtype.Alu then
    invalid_arg "kernel boundary requires storage"
  else
    let state, deps = unjoin after in
    let p = stand_in ctx state in
    if deps <> [] then
      Tbl.replace ctx.joined p
        (Option.value (Tbl.find_opt ctx.joined p) ~default:[] @ deps);
    Some p

let add_range ctx r =
  let key = (axis_type r, axis_id r) in
  if not (Hashtbl.mem ctx.ranges key) then begin
    Hashtbl.replace ctx.ranges key ();
    ctx.range_order <- key :: ctx.range_order
  end

(* A kernel's ranges are renumbered from 0 in the order it reaches them, so
   equal kernels dedupe. *)
let pm_renumber_range =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.op Op.Range ~name:"r") (fun numbers m ->
          let r = m "r" in
          let axis_type = axis_type r in
          Option.map
            (fun n ->
              replace r
                ~arg:(Range { axis_type; axis_id = n :: List.tl (axis_id r) }))
            (Hashtbl.find_opt numbers (axis_type, axis_id r)));
    ])

let to_define_global =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx
        (Upat.v ~op:(ops Op.[ Buffer; Alloc ]) ~name:"buf" ())
        (fun ctx m -> add_buf ctx (m "buf"));
      (* Only storage parameters get kernel-local slots; scalar parameters keep
         the slots of their enclosing call. *)
      rule_ctx (Upat.op Op.Param ~name:"buf") (fun ctx m ->
          let buf = m "buf" in
          if addrspace buf = Some Dtype.Alu then None else add_buf ctx buf);
      (* Scalar parameters are values, not buffers. *)
      rule_ctx
        (Upat.op Op.Index ~src:[ Upat.op Op.Param ~name:"v" ])
        (fun _ m ->
          if addrspace (m "v") = Some Dtype.Alu then Some (m "v") else None);
      rule_ctx (Upat.op Op.After ~name:"after") (fun ctx m ->
          handle_after ctx (m "after"));
      rule_ctx (Upat.op Op.Range ~name:"r") (fun ctx m ->
          add_range ctx (m "r");
          None);
    ])

(* Whether the store or end [x] is a kernel of its own: its ranges are all
   closed, and it is no loop around a call, which runs that call. *)
let is_kernel x =
  Nodes.cardinal (ranges x) = 0
  && not
       (op x = Op.End
       && op (nth x 0) = Op.Call
       && List.exists loop_range (List.tl (src x)))

let split_store x =
  if not (is_kernel x) then None
  else
    let lctx =
      {
        map = Tbl.create 8;
        order = [];
        params = Tbl.create 8;
        joined = Tbl.create 8;
        ranges = Hashtbl.create 8;
        range_order = [];
      }
    in
    let ret =
      graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:lctx x
        (Before_sources
           (Pattern_matcher.append to_define_global
              (with_ctx Simplify.pm_flatten_range)))
    in
    let bufs = List.rev lctx.order in
    let ret =
      substitute ~calls:Skip ~pass:Once ret
        (List.mapi (fun slot b -> (b, debuf slot b)) bufs)
    in
    let numbers = Hashtbl.create 8 in
    List.iteri
      (fun n key -> Hashtbl.replace numbers key n)
      (List.rev lctx.range_order);
    let ret =
      graph_rewrite ~calls:Skip ~pass:Once ~ctx:numbers ret
        (After_sources pm_renumber_range)
    in
    (* Buffers can be on different devices here: the schedule compiles such
       kernels to copies. *)
    let args =
      List.map
        (fun b ->
          let a = Tbl.find lctx.map b in
          match Tbl.find_opt lctx.joined b with
          | Some deps -> after a (dedup_nodes deps)
          | None -> a)
        bufs
    in
    Some (call (sink ~kernel:(kernel_info ()) [ ret ]) args)

let split_kernels =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(ops Op.[ Store; End ]) ~name:"x" ())
        (fun m -> split_store (m "x"));
    ])

(* Cleanups for speed and runnability. *)
let pm_cleanups =
  Pattern_matcher.concat
    [
      with_ctx Shape.symbolic;
      with_ctx Simplify.pm_reduce_simplify;
      with_ctx pm_const_buffer_folding;
      pm_remove_bufferize;
    ]

(* Values that several kernels compute

   Each kernel computes the values it reads that are not stored, so a value of
   no range that [k] kernels read is computed [k] times. One is stored once, in
   one element, when the nodes computed again past the first copy, [(k - 1) n]
   for the [n] computed nodes of its value, exceed [inline_budget], the bound
   that stages within a kernel keep. A value of [n] nodes that 3 kernels read
   is stored past 180 nodes. A value with ranges is computed in several
   kernels only through a range they share, which no kernel graph makes yet,
   and storing it would hold its extent alive between its kernels. *)

(* Where a kernel's walk of its values stops: storage, a later state, a stage
   stored whole and a call. *)
let kernel_boundary u =
  match op u with
  | Op.Stage -> (opts u).addrspace = Dtype.Global
  | Op.After | Op.Call -> true
  | o -> Op.Set.mem o Op.Set.defines

(* A node that computes on a device: an elementwise operation or a
   reduction. *)
let computes u =
  (Op.Set.mem (op u) Op.Set.elementwise || op u = Op.Reduce)
  && Option.is_some (device u)

(* A weak value has no storage: it takes the width of the values it meets,
   which differ between its readers. *)
let is_weak u = List.exists (Dtype.equal (dtype u)) Dtype.weaks

(* The kernels that compute a node, as their numbers in increasing order and
   their count, the count kept exact up to [saturated]: past [saturated - 1]
   kernels any computed node passes the budget. *)
type kernels = { ids : int list; count : int }

let saturated = int_of_float inline_budget + 2
let no_kernels = { ids = []; count = 0 }

let union a b =
  if a == b || b.count = 0 then a
  else if a.count = 0 then b
  else
    let rec merge n acc a b =
      if n = saturated then (n, acc)
      else
        match (a, b) with
        | [], [] -> (n, acc)
        | x :: a', [] | [], x :: a' -> merge (n + 1) (x :: acc) a' []
        | x :: a', y :: b' ->
            if x = y then merge (n + 1) (x :: acc) a' b'
            else if x < y then merge (n + 1) (x :: acc) a' b
            else merge (n + 1) (y :: acc) a b'
    in
    let count, ids = merge 0 [] a.ids b.ids in
    { ids = List.rev ids; count }

(* The least [n] whose [(k - 1) n] passes the budget. *)
let past_budget k = (int_of_float inline_budget / (k - 1)) + 1

(* The computed nodes of [v]'s value, counted to [cap] at most, its walk
   stopping at kernel boundaries and at the nodes [stop] accepts, which it
   returns. *)
let computed_nodes ~stop v cap =
  let seen = Tbl.create 64 in
  let rec go n below = function
    | [] -> (n, below)
    | _ when n >= cap -> (n, below)
    | x :: rest when Tbl.mem seen x -> go n below rest
    | x :: rest ->
        Tbl.replace seen x ();
        if x != v && stop x then go n (x :: below) rest
        else if x != v && kernel_boundary x then go n below rest
        else
          let n = if computes x then n + 1 else n in
          go n below (List.rev_append (src x) rest)
  in
  go 0 [] [ v ]

(* The values to store, with the kernels each is computed in. One walk,
   consumers first: a node is computed in the kernels of its consumers, a
   kernel computing its root and a stored value being a kernel of its own, so
   its kernels are known when the walk reaches it. A candidate is a computed
   value of no range, not weak, that some consumer computes in fewer kernels
   than it: the values below it move with it. Under a node that is no
   candidate, as a weak value or a load several kernels share, the fewest
   kernels of its consumers pass down, so the value below it is weighed in its
   place. Its nodes are counted to storage, and those past the budget are
   stored. *)
let candidates nodes =
  let never _ = false in
  let kernels = Tbl.create 256 and least = Tbl.create 256 in
  let next = ref 0 in
  let fresh () =
    incr next;
    { ids = [ !next ]; count = 1 }
  in
  let push s ks k =
    if not (Op.Set.mem (op s) Op.Set.irreducible) then begin
      let cur = Option.value (Tbl.find_opt kernels s) ~default:no_kernels in
      Tbl.replace kernels s (union cur ks);
      match Tbl.find_opt least s with
      | Some m when m <= k -> ()
      | _ -> Tbl.replace least s k
    end
  in
  let chosen = ref [] in
  let visit x =
    if op x = Op.Stage && kernel_boundary x then push (nth x 0) (fresh ()) 1
    else if not (kernel_boundary x) then begin
      let ks =
        if (op x = Op.Store || op x = Op.End) && is_kernel x then fresh ()
        else Option.value (Tbl.find_opt kernels x) ~default:no_kernels
      in
      let k = ks.count in
      let fewest = Option.value (Tbl.find_opt least x) ~default:k in
      let eligible =
        computes x && (not (is_weak x)) && Nodes.cardinal (ranges x) = 0
      in
      let ks =
        if
          k >= 2 && fewest < k && eligible
          && (k - 1) * fst (computed_nodes ~stop:never x (past_budget k))
             > int_of_float inline_budget
        then begin
          chosen := (x, k) :: !chosen;
          fresh ()
        end
        else ks
      in
      let passed = if eligible then ks.count else fewest in
      if ks.count > 0 then List.iter (fun s -> push s ks passed) (src x)
    end
  in
  List.iter visit nodes;
  List.rev !chosen

(* The candidates [chosen], consumers first, without those whose nodes, counted
   to the candidates kept below them, are within the budget: each is computed
   again in its kernels, whose count the candidates below it then take on. *)
let kept chosen =
  let keep = Tbl.create 64 and count = Tbl.create 64 in
  List.iter
    (fun (x, k) ->
      Tbl.replace keep x ();
      Tbl.replace count x k)
    chosen;
  List.iter
    (fun (y, _) ->
      let k = Tbl.find count y in
      let n, below = computed_nodes ~stop:(Tbl.mem keep) y (past_budget k) in
      if (k - 1) * n <= int_of_float inline_budget then begin
        Tbl.remove keep y;
        List.iter
          (fun z -> Tbl.replace count z (Tbl.find count z + k - 1))
          below
      end)
    chosen;
  keep

(* [tsink] with each value that several kernels compute past the budget
   stored once, read by each. *)
let store_shared ctx tsink =
  let nodes = toposort ~calls:Skip tsink in
  match candidates (List.rev nodes) with
  | [] -> tsink
  | chosen ->
      let keep = kept chosen in
      let image = Tbl.create 64 in
      let rebuilt u =
        let srcs = src u in
        let srcs' =
          List.map
            (fun s -> Option.value (Tbl.find_opt image s) ~default:s)
            srcs
        in
        let u' =
          if List.for_all2 ( == ) srcs srcs' then u else replace u ~src:srcs'
        in
        let u' = if Tbl.mem keep u then stage ctx u' else u' in
        if u' != u then Tbl.replace image u u'
      in
      List.iter rebuilt nodes;
      Option.value (Tbl.find_opt image tsink) ~default:tsink

let get_kernel_graph tsink =
  let setting = Setting.value in
  let tsink = Indexing.run_rangeify tsink in
  (* An argument of a call that another kernel reads is indexed by that
     kernel's ranges, which would reach every value that reads the call's
     results: a stage of such a value would index its storage by them. *)
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
      (After_sources pm_no_indexing_calls)
  in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(stage_reads tsink) tsink
      (After_sources pm_cleanups)
  in
  let ranges =
    List.filter (fun x -> op x = Op.Range) (toposort ~calls:Skip tsink)
  in
  let next_range =
    1 + List.fold_left (fun m r -> max m (List.hd (axis_id r))) (-1) ranges
  in
  let ctx = { buf_cache = Tbl.create 64; range_idx = next_range } in
  let tsink = store_shared ctx tsink in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx tsink
      (After_sources pm_limit_bufs)
  in
  let slots =
    List.filter_map
      (fun x ->
        match (op x, arg x) with Op.Alloc, Param p -> Some p.slot | _ -> None)
      (toposort ~calls:Skip tsink)
  in
  let next_slot = ref (1 + List.fold_left max (-1) slots) in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:next_slot tsink
      (Before_sources pm_add_buffers)
  in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
      (After_sources pm_no_indexing_calls)
  in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
      (Before_sources split_kernels)
  in
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
      (After_sources pm_no_views)
  in
  if setting Setting.spec <> 0 then
    Spec.type_verify ~calls:Skip Spec.kernel_graph tsink;
  tsink
