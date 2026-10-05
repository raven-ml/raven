(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module Ready = Set.Make (Int)

(* Linearizing *)

let debug_linearize =
  Setting.bool ~reach:Process "DEBUG_LINEARIZE" false

let priority u =
  (* Nodes with higher run counts are placed later. *)
  let run_count =
    Nodes.fold (fun r n -> n * (Dtype.Value.to_int (vmax r) + 1)) (ranges u) 1
  in
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
  (run_count, priority, extra)

let pp_priority ppf (run_count, priority, extra) =
  let pp_extra ppf = function
    | None -> Format.pp_print_string ppf "None"
    | Some slot -> Format.pp_print_int ppf slot
  in
  Format.fprintf ppf "(%d, %d, %a)" run_count priority pp_extra extra

let linearize sink =
  let lst = toposort sink in
  let out_degree = Tbl.create 256 and priorities = Tbl.create 256 in
  let degree u = Option.value ~default:0 (Tbl.find_opt out_degree u) in
  List.iter
    (fun u ->
      List.iter (fun s -> Tbl.replace out_degree s (degree s + 1)) (src u);
      Tbl.replace priorities u (priority u))
    lst;
  (* Number the nodes in the ideal order. *)
  let tuple_order = Setting.value Setting.tuple_order in
  let ideal u0 u1 =
    match Stdlib.compare (Tbl.find priorities u0) (Tbl.find priorities u1) with
    | 0 when tuple_order -> compare_structure u0 u1
    | c -> c
  in
  let by_nkey = Array.of_list (List.stable_sort ideal lst) in
  let nkey = Tbl.create (Array.length by_nkey) in
  Array.iteri (fun i u -> Tbl.replace nkey u i) by_nkey;
  (* Then place them in the topological order closest to it, from the end. *)
  let rec place ready placed =
    match Ready.max_elt_opt ready with
    | None -> placed
    | Some k ->
        let u = by_nkey.(k) in
        let release ready v =
          let d = degree v - 1 in
          Tbl.replace out_degree v d;
          if d = 0 then Ready.add (Tbl.find nkey v) ready else ready
        in
        place
          (List.fold_left release (Ready.remove k ready) (src u))
          (u :: placed)
  in
  let lst = place (Ready.singleton (Tbl.find nkey sink)) [] in
  if Setting.value debug_linearize then
    List.iteri
      (fun i u ->
        Format.printf "%4d %-20s %s %a@." i
          (Format.asprintf "%a" Op.pp (op u))
          (multirange_str ~color:true ~pad:10 (Nodes.to_list (ranges u)))
          pp_priority (Tbl.find priorities u))
      lst;
  lst

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
        List.fold_left (fun d s -> union d (Tbl.find deps s)) [] (src u)
      in
      if is_loop u || op u = Op.Sink then List.iter (nest u) d;
      let self = match op u with Op.Range -> true | _ -> is_loop u in
      Tbl.replace deps u (if self then d @ [ u ] else d))
    (toposort sink);
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
      if Nodes.mem r (backward_slice_with_self x) then
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
              (fun y -> replace ~src:(src x @ [ y ]) x)
              (Tbl.find_opt edges x));
      ]))

(* Splitting ends *)

let compare_range_arg r0 r1 =
  match Stdlib.compare (axis_id r0) (axis_id r1) with
  | 0 -> Axis_type.compare (axis_type r0) (axis_type r1)
  | c -> c

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
