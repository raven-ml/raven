(* Recorded simplifications: each that a test of the reference makes, and [sym]
   on random expressions, as gen/uop/symbolic.py records them. A record holds
   the simplification's kind, the input, the result and, for some, the result's
   bounds. *)

open Windtrap
open Tolk
open Simplification

let matcher = function
  | "sym" -> Symbolic.sym
  | "symbolic" -> Shape.symbolic
  | "symbolic_simple" -> Shape.symbolic_simple
  | "commutative" -> Shape.commutative
  | "pm_simplify_valid" -> Symbolic.pm_simplify_valid
  | "pm_move_where_on_load" -> Symbolic.pm_move_where_on_load
  | "pm_drop_and_clauses" -> Symbolic.pm_drop_and_clauses
  | "pm_remove_invalid" -> Shape.pm_remove_invalid
  | "pm_clean_up_group_sink" -> Symbolic.pm_clean_up_group_sink
  | "sym+pm_move_where_on_load" ->
      Ops.Pattern_matcher.append Symbolic.sym Symbolic.pm_move_where_on_load
  | "symbolic_simple+pm_commit_weak" ->
      Ops.Pattern_matcher.append Shape.symbolic_simple Uop_weak.pm_commit_weak
  | name -> failf "no matcher %s" name

(* [simplification kind] is the simplification a record of [kind] makes. *)
let simplification kind =
  match String.split_on_char ' ' kind with
  | [ "simplify" ] -> fun u -> Some (Shape.simplify u)
  | [ "simplify_valid" ] -> Symbolic.simplify_valid
  | [ m ] -> fun u -> Some (rewrite (matcher m) u)
  | [ m; "bottom_up" ] -> fun u -> Some (rewrite ~order:before (matcher m) u)
  | _ -> failf "no simplification %s" kind

type record = {
  kind : string;
  bounds : (Dtype.value * Dtype.value) option;
  input : Ops.t;
  result : Ops.t option;
}

let bound c : Dtype.value =
  match Ops.value c with
  | #Dtype.value as v -> v
  | `Invalid -> failf "an invalid bound"

let record u =
  let kind, bounds, srcs =
    match Ops.src u with
    | k :: srcs when Ops.op k = Op.Custom_function -> (
        let kind =
          match Ops.arg k with
          | Custom_function f -> f.name
          | _ -> failf "a record's kind without its name"
        in
        match Ops.src k with
        | [] -> (kind, None, srcs)
        | [ lo; hi ] -> (kind, Some (bound lo, bound hi), srcs)
        | _ -> failf "a record's kind with %d bounds" (List.length (Ops.src k)))
    | _ -> failf "a record without its kind"
  in
  match srcs with
  | [ input; result ] -> { kind; bounds; input; result = Some result }
  | [ input ] -> { kind; bounds; input; result = None }
  | _ -> failf "a record of %d nodes" (List.length srcs)

(* The record's simplification gives its result, within its bounds, and the
   result keeps the input's value. A record of no result is a simplification
   that changes nothing. [replay r] is the number of bindings it checked the
   value at. *)
let replay r =
  equal ~msg:r.kind (option uop) r.result (simplification r.kind r.input);
  match r.result with
  | None -> 0
  | Some out ->
      Option.iter
        (fun (lo, hi) ->
          equal ~msg:"bounds"
            (pair Dtypes.value Dtypes.value)
            (lo, hi)
            (Ops.vmin out, Ops.vmax out))
        r.bounds;
      keeps_value ~name:r.kind r.input out

(* Whether [u] is integer arithmetic on a variable, whose values a binding
   draws: no float, and no read of memory or lane of a vector, which the value
   law leaves out. *)
let integer_arithmetic u =
  let nodes = Ops.toposort ~calls:Skip u in
  let variable n =
    match Ops.arg n with
    | Param { vmin_vmax = Some _; addrspace = Alu; dtype; _ } ->
        not (Dtype.is_float dtype)
    | _ -> false
  in
  let outside n =
    Dtype.is_float (Ops.dtype n)
    || List.mem (Ops.op n) Op.[ Index; Load ]
    || (Ops.op n = Op.Stack && not (Dtype.equal (Ops.dtype n) Void))
  in
  List.exists variable nodes && not (List.exists outside nodes)

(* Each record of the golden [name] as a subtest, so that one record's failure
   hides no other. A record of integer arithmetic on a variable has its value
   checked at some binding: the value law cannot pass for want of one. *)
let replay_all name =
  let records = List.map record (Ops.src (Golden.sink (name ^ ".golden"))) in
  List.iteri
    (fun k r ->
      subtest (Printf.sprintf "record %d (%s)" k r.kind) (fun () ->
          let checked = replay r in
          if Option.is_some r.result && integer_arithmetic r.input then
            greater ~msg:"bindings checked" int ~than:0 checked))
    records

let simplifications =
  group "makes each recorded simplification, within its recorded bounds"
    [
      Golden.cases "tests.golden" ~key:[ "class"; "test" ] (fun cell ->
          replay_all (cell "class" ^ "." ^ cell "test"));
    ]

let random_expressions =
  test "simplifies recorded random integer expressions as recorded" (fun () ->
      replay_all "random_expressions")
