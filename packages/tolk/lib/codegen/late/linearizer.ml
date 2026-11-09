(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Linearizing *)

let priority u =
  let rs = ranges u in
  (* Smaller numbers are placed closer to the top. *)
  let priority, extra =
    match (op u, arg u) with
    | Op.Param, Param p -> (-20, Some p.slot)
    | (Op.Buffer | Op.Alloc), _ ->
        ((if addrspace u = Some Dtype.Local then -17 else -18), None)
    | Op.Load, _ -> (-1, None)
    | Op.Store, _ -> (1, None)
    | Op.Range, _ -> (5, None)
    | (Op.End | Op.Backedge), _ -> (-5, None)
    | _ -> (0, None)
  in
  (* A node in more loops is placed later. *)
  (Nodes.cardinal rs, priority, extra)

let pp_priority ppf (depth, priority, extra) =
  let pp_extra ppf = function
    | None -> Format.pp_print_string ppf "None"
    | Some slot -> Format.pp_print_int ppf slot
  in
  Format.fprintf ppf "(%d, %d, %a)" depth priority pp_extra extra

(* A heap of distinct ints in [a.(0)] to [a.(n - 1)], the greatest at the
   root. *)
let rec sift_up (a : int array) i =
  let parent = (i - 1) / 2 in
  if i > 0 && a.(parent) < a.(i) then begin
    let x = a.(i) in
    a.(i) <- a.(parent);
    a.(parent) <- x;
    sift_up a parent
  end

let rec sift_down (a : int array) n i =
  let l = (2 * i) + 1 in
  if l < n then begin
    let c = if l + 1 < n && a.(l + 1) > a.(l) then l + 1 else l in
    if a.(c) > a.(i) then begin
      let x = a.(i) in
      a.(i) <- a.(c);
      a.(c) <- x;
      sift_down a n c
    end
  end

let linearize sink =
  let lst = Array.of_list (toposort ~calls:Skip sink) in
  let n = Array.length lst in
  let index = Tbl.create n in
  Array.iteri (fun i u -> Tbl.replace index u i) lst;
  let out_degree = Array.make n 0 in
  let rec count = function
    | [] -> ()
    | s :: rest ->
        let i = Tbl.find index s in
        out_degree.(i) <- out_degree.(i) + 1;
        count rest
  in
  Array.iter (fun u -> count (src_without_body u)) lst;
  let priorities = Array.map priority lst in
  (* Number the nodes in the ideal order. *)
  let structure =
    if Setting.value Setting.tuple_order then
      Array.map (structure_ranks sink) lst
    else Array.make n 0
  in
  let ideal i0 i1 =
    let d0, p0, e0 = priorities.(i0) and d1, p1, e1 = priorities.(i1) in
    let c = Int.compare d0 d1 in
    let c = if c <> 0 then c else Int.compare p0 p1 in
    let c = if c <> 0 then c else Option.compare Int.compare e0 e1 in
    if c <> 0 then c else Int.compare structure.(i0) structure.(i1)
  in
  let by_nkey = Array.init n Fun.id in
  Array.stable_sort ideal by_nkey;
  let nkey = Array.make n 0 in
  Array.iteri (fun k i -> nkey.(i) <- k) by_nkey;
  (* Then place them in the topological order closest to it, from the end:
     [ready] is a heap of the numbers of the nodes whose consumers are all
     placed. *)
  let ready = Array.make n 0 and len = ref 1 and placed = ref [] in
  let rec release = function
    | [] -> ()
    | v :: rest ->
        let i = Tbl.find index v in
        out_degree.(i) <- out_degree.(i) - 1;
        if out_degree.(i) = 0 then begin
          ready.(!len) <- nkey.(i);
          incr len;
          sift_up ready (!len - 1)
        end;
        release rest
  in
  ready.(0) <- nkey.(n - 1);
  while !len > 0 do
    let u = lst.(by_nkey.(ready.(0))) in
    decr len;
    ready.(0) <- ready.(!len);
    sift_down ready !len 0;
    placed := u :: !placed;
    release (src_without_body u)
  done;
  let order = !placed in
  if Setting.value Setting.debug_linearize then
    List.iteri
      (fun i u ->
        Format.printf "%4d %-20s %s %a@." i
          (Format.asprintf "%a" Op.pp (op u))
          (multirange_str ~color:true ~pad:10 (Nodes.to_list (ranges u)))
          pp_priority
          priorities.(Tbl.find index u))
      order;
  order

(* Chaining loops *)

(* There are three relationships between the ranges x and y: nested, when ending
   y depends on ending x and x depends on ending y; dependent, when ending y
   depends on ending x and x does not depend on ending y; independent, when
   ending y does not depend on ending x. Everything is nested inside the
   sink. *)

type cfg_context = t Tbl.t

let is_loop u = match op u with Op.End | Op.Backedge -> true | _ -> false
let union d0 d1 = d0 @ List.filter (fun x -> not (List.memq x d0)) d1

let rec chain = function
  | x :: (y :: _ as rest) -> (x, y) :: chain rest
  | _ -> []

let cfg_context sink =
  let deps = Tbl.create 256 and nesting = ref [] in
  let nest u x =
    let inside = op u = Op.Sink || List.memq (nth u 1) (Tbl.find deps x) in
    if is_loop x && inside && not (List.mem_assq x !nesting) then
      nesting := (x, u) :: !nesting
  in
  List.iter
    (fun u ->
      let d =
        List.fold_left
          (fun d s -> union d (Tbl.find deps s))
          [] (src_without_body u)
      in
      if is_loop u || op u = Op.Sink then List.iter (nest u) d;
      let self = match op u with Op.Range -> true | _ -> is_loop u in
      Tbl.replace deps u (if self then d @ [ u ] else d))
    (toposort ~calls:Skip sink);
  let nesting = List.rev !nesting and edges = Tbl.create 16 in
  let add_edges k =
    let v =
      List.filter_map (fun (x, p) -> if p == k then Some x else None) nesting
    in
    (* Ranges that depend on other siblings are scheduled after them. *)
    let depended x =
      List.length (List.filter (fun u -> List.memq u (Tbl.find deps x)) v)
    in
    let order =
      List.stable_sort (fun x0 x1 -> Int.compare (depended x0) (depended x1)) v
    in
    let add (x, y) =
      let r = nth y 1 in
      if Nodes.mem r (backward_slice_with_self ~calls:Skip x) then
        invalid_arg
          (Printf.sprintf "range %s would run after a loop that depends on it"
             (range_str r));
      Tbl.replace edges r x
    in
    List.iter add (chain (if op k = Op.Sink then order else nth k 1 :: order))
  in
  List.iter add_edges (Helpers.dedup (module Ops) (List.map snd nesting));
  edges

let pm_add_control_flow =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.op Op.Range ~name:"x") (fun edges m ->
            let x = m "x" in
            Option.map
              (fun y -> replace ~src:[ after (nth x 0) [ y ] ] x)
              (Tbl.find_opt edges x));
      ]))

(* Splitting ends *)

let do_split_ends e =
  let ranges_of s =
    if op s = Op.Range then [ s ] else Nodes.to_list (ranges s)
  in
  let rngs =
    Helpers.dedup (module Ops) (List.concat_map ranges_of (List.tl (src e)))
  in
  let innermost_first =
    List.stable_sort (fun r0 r1 -> compare_range_arg r1 r0) rngs
  in
  Some (List.fold_left (fun ret r -> end_ ret [ r ]) (nth e 0) innermost_first)

let pm_split_ends =
  Pattern_matcher.(
    v (fun () -> [ rule (Upat.op Op.End ~name:"e") (fun m -> do_split_ends (m "e")) ]))
