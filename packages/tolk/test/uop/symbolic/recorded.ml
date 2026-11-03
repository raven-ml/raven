(* Replays of tinygrad: each simplification its tests make, and sym on random
   expressions, as gen/uop/symbolic.py records them, is the one Symbolic makes,
   with the bounds tinygrad gives its result, and keeps the value. *)

open Windtrap
open Tolk
open Dtypes
open Common

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
      Ops.Pattern_matcher.append Shape.symbolic_simple
        Uop_weak.pm_commit_weak
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
    | k :: srcs when Ops.op k = Op.Custom -> (
        let kind =
          match Ops.arg k with
          | Code c -> c.code
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

let replay test k r =
  let msg = Printf.sprintf "%s, record %d (%s)" test k r.kind in
  equal ~msg (option uop) r.result (simplification r.kind r.input);
  Option.iter
    (fun (lo, hi) ->
      let out = Option.get r.result in
      equal ~msg (pair value value) (lo, hi) (Ops.vmin out, Ops.vmax out))
    r.bounds;
  keeps_value ~name:msg r.input (Option.value r.result ~default:r.input)

let replay_all name =
  let records = Ops.src (Golden.sink (name ^ ".golden")) in
  List.iteri (fun k u -> replay name k (record u)) records

let tests =
  let replays cell =
    let name = cell "class" ^ "." ^ cell "test" in
    test name (fun () -> replay_all name)
  in
  group "tests.golden" (List.map replays (Golden.rows "tests.golden"))

let random_expressions =
  test "sym simplifies random integer expressions as tinygrad does" (fun () ->
      replay_all "random_expressions")
