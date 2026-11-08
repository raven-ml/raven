(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Why a value may differ between a workgroup's threads. [Lanes rs] varies only
   with the thread indices and vector lanes [rs], which a reduction over them
   removes: a maximum over a workgroup's threads is the same in each. [Any]
   varies for another reason, which nothing removes. *)
type cause = Lanes of t list | Any

let uniform = Lanes []
let is_uniform = function Lanes [] -> true | Lanes _ | Any -> false

let join c0 c1 =
  match (c0, c1) with
  | Any, _ | _, Any -> Any
  | Lanes [], c | c, Lanes [] -> c
  | Lanes r0, Lanes r1 ->
      Lanes (r0 @ List.filter (fun r -> not (List.memq r r0)) r1)

let is o u = Op.equal (op u) o
let serial r = is Op.Range r && List.mem (axis_type r) Axis_type.[ Weak; Loop ]
let unbounded r = Dtype.equal (dtype r) Dtype.Void

(* The storage an access through [p] reaches, if [p] names one. *)
let storage p =
  let b = buf_uop p in
  if Op.Set.mem (op b) Op.Set.defines then Some b else None

let space b = match arg b with Param p -> Some p.addrspace | _ -> None

(* The writes to each storage of a kernel: the stores that target it, and the
   calls it is a memory argument of. *)
let writers nodes =
  let tbl = Tbl.create 16 in
  let add w b =
    Tbl.replace tbl b (w :: Option.value ~default:[] (Tbl.find_opt tbl b))
  in
  let memory a =
    List.mem (addrspace a) Dtype.[ Some Global; Some Local; Some Reg ]
  in
  List.iter
    (fun u ->
      match op u with
      | Op.Store -> Option.iter (add u) (storage (nth u 0))
      | Op.Call ->
          List.iter
            (fun a -> if memory a then Option.iter (add u) (storage a))
            (src_without_body u)
      | _ -> ())
    nodes;
  tbl

(* Every thread reads the same workgroup memory at [load] when each of [writes]
   is ordered before it by a barrier: one of the barriers [load] is ordered
   after reaches the write. *)
let ordered_by_barriers writes load =
  let before =
    List.filter_map
      (fun u ->
        if is Op.Barrier u then Some (backward_slice_with_self ~calls:Skip u)
        else None)
      (Nodes.to_list (backward_slice ~calls:Skip (nth load 0)))
  in
  List.for_all (fun w -> List.exists (Nodes.mem w) before) writes

(* The cause of each value of a kernel whose writes are [writes], given the
   register buffers and the unbounded loops already found varying, [marked]. *)
let causes ~launched ~writes ~marked =
  let memo = Tbl.create 256 in
  let rec cause u =
    match Tbl.find_opt memo u with
    | Some c -> c
    | None ->
        let c = compute u in
        Tbl.replace memo u c;
        c
  and sources u = List.fold_left (fun c s -> join c (cause s)) uniform (src u)
  and compute u =
    match (op u, arg u) with
    | (Op.Const | Op.Binary | Op.Buffer | Op.Alloc), _ -> uniform
    | (Op.Param | Op.Refparam), _ ->
        (* a body's parameters and variables are what each thread's call passes
           it *)
        if launched then uniform else Any
    | Op.Special, String name ->
        if String.starts_with ~prefix:"gidx" name then uniform else Any
    | Op.Range, _ -> range u
    | Op.After, _ -> cause (nth u 0)
    | Op.Load, _ -> join (sources u) (memory u)
    | Op.Reduce, _ -> reduce u
    | (Op.Wmma | Op.Ins | Op.Call), _ -> Any
    | _ -> sources u
  and range r =
    match axis_type r with
    | Global -> uniform
    | Local | Warp | Upcast -> Lanes [ r ]
    | Placeholder -> Any
    | Weak | Loop ->
        let trips_vary =
          if unbounded r then Tbl.mem marked r
          else not (is_uniform (cause (nth r 0)))
        in
        if trips_vary then Any else uniform
  (* A reduction removes the thread indices and lanes it reduces over; over a
     loop whose trips vary, its result varies with them. *)
  and reduce u =
    let closed = ended_ranges u in
    if List.exists (fun r -> serial r && not (is_uniform (cause r))) closed then
      Any
    else
      match cause (nth u 0) with
      | Any -> Any
      | Lanes rs -> Lanes (List.filter (fun r -> not (List.memq r closed)) rs)
  and memory load =
    match storage (nth load 0) with
    | None -> Any
    | Some b -> (
        let ws = Option.value ~default:[] (Tbl.find_opt writes b) in
        match (op b, space b, arg b) with
        | Op.Binary, _, _ -> uniform
        | _, Some Global, Param p ->
            if p.volatile || ws <> [] then Any else uniform
        | _, Some Local, _ ->
            if ordered_by_barriers ws load then uniform else Any
        | _, Some Reg, _ -> if Tbl.mem marked b then Any else uniform
        | _ -> Any)
  in
  cause

(* A register buffer varies when one of its writes leaves it varying: a call or
   custom code, or a store of a varying value, index or gate, or in a loop whose
   trips vary. *)
let leaves_varying cause w =
  (not (is Op.Store w))
  || (not (is_uniform (cause w)))
  || Nodes.fold
       (fun r v -> v || (serial r && not (is_uniform (cause r))))
       (ranges w) false

let varying sink =
  let launched = match arg sink with Kernel k -> k.kernel | _ -> false in
  let nodes = toposort ~calls:Skip sink in
  let writes = writers nodes in
  let registers =
    Tbl.fold
      (fun b ws acc -> if space b = Some Dtype.Reg then (b, ws) :: acc else acc)
      writes []
  in
  let backedges = List.filter (is Op.Backedge) nodes in
  (* The least fixpoint: marks only grow, from none, until a round adds none, as
     a mark makes no value uniform. *)
  let rec solve marked =
    let cause = causes ~launched ~writes ~marked in
    let next = Tbl.create 8 in
    List.iter
      (fun e ->
        if not (is_uniform (cause (nth e 2))) then Tbl.replace next (nth e 1) ())
      backedges;
    List.iter
      (fun (b, ws) ->
        if List.exists (leaves_varying cause) ws then Tbl.replace next b ())
      registers;
    if Tbl.length next = Tbl.length marked then cause else solve next
  in
  let cause = solve (Tbl.create 8) in
  fun u -> not (is_uniform (cause u))

(* Check *)

let synchronises u = is Op.Barrier u || is Op.Wmma u

let what u =
  match op u with
  | Op.Barrier -> Some "a barrier"
  | Op.Wmma -> Some "a matrix multiply-accumulate"
  | Op.Call when List.exists synchronises (toposort ~calls:Enter (body u)) ->
      Some "a call holding a barrier or a matrix multiply-accumulate"
  | _ -> None

let constant_bound r = (not (unbounded r)) && is Op.Const (nth r 0)

let check sink =
  let name =
    match arg sink with Kernel k -> "kernel " ^ k.name | _ -> "a called body"
  in
  (* most kernels hold no barrier in a loop of a computed bound *)
  let varying = lazy (varying sink) in
  let check_loop w r =
    if serial r && (not (constant_bound r)) && Lazy.force varying r then
      invalid_arg
        (Printf.sprintf
           "%s: %s runs in loop %s, whose trips may differ between the threads \
            of a workgroup"
           name w (range_str r))
  in
  (* the operations a kernel reaches are known without a walk *)
  if op_in_backward_slice_with_self ~calls:Skip sink Op.[ Barrier; Wmma; Call ]
  then
    List.iter
      (fun u ->
        match what u with
        | None -> ()
        | Some w -> Nodes.fold (fun r () -> check_loop w r) (ranges u) ())
      (toposort ~calls:Skip sink)
