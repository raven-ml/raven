(* Replays of tinygrad: each simplification its tests make, and sym on random
   expressions, as gen/uop/symbolic.py records them, is the one Symbolic makes,
   with the bounds tinygrad gives its result, and keeps the value. *)

open Windtrap
open Tolk
open Dtypes
open Common

let matcher = function
  | "sym" -> Symbolic.sym
  | "symbolic" -> Symbolic.symbolic
  | "symbolic_simple" -> Symbolic.symbolic_simple
  | "commutative" -> Symbolic.commutative
  | "pm_simplify_valid" -> Symbolic.pm_simplify_valid
  | "pm_move_where_on_load" -> Symbolic.pm_move_where_on_load
  | "pm_drop_and_clauses" -> Symbolic.pm_drop_and_clauses
  | "pm_remove_invalid" -> Symbolic.pm_remove_invalid
  | "pm_clean_up_group_sink" -> Symbolic.pm_clean_up_group_sink
  | "sym+pm_move_where_on_load" ->
      Ops.Pattern_matcher.append Symbolic.sym Symbolic.pm_move_where_on_load
  | "symbolic_simple+pm_commit_weak" ->
      Ops.Pattern_matcher.append Symbolic.symbolic_simple
        Uop_weak.pm_commit_weak
  | name -> failf "no matcher %s" name

(* [simplification kind] is the simplification a record of [kind] makes. *)
let simplification kind =
  match String.split_on_char ' ' kind with
  | [ "simplify" ] -> fun u -> Some (Ops.simplify u)
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

let bound : Ops.Tag.t -> Dtype.value = function
  | Int n -> `Int (Bigint.of_int n)
  | Bool b -> `Bool b
  | tag -> failf "%a is not a bound" Ops.Tag.pp tag

let record u =
  let kind, bounds =
    match Ops.tag u with
    | Some (String kind) -> (kind, None)
    | Some (Tuple [ String kind; lo; hi ]) -> (kind, Some (bound lo, bound hi))
    | _ -> failf "a record without its kind"
  in
  match Ops.src u with
  | [ input; result ] -> { kind; bounds; input; result = Some result }
  | [ input ] -> { kind; bounds; input; result = None }
  | _ -> failf "a record of %d nodes" (List.length (Ops.src u))

(* The host's libm folds these operations, and libms legitimately differ in the
   last place: macOS's sin 4.0 is one unit in the last place from glibc's, which
   is correctly rounded. A constant they fold agrees within one. *)
let host_libm = Op.[ Sin; Log2; Exp2; Pow ]

let folded_within_ulp =
  let near a b =
    Float.equal a b
    || Float.equal (Float.succ a) b
    || Float.equal (Float.pred a) b
  in
  Testable.make ~pp:(Testable.pp uop) ~equal:(fun u0 u1 ->
      match (Ops.arg u0, Ops.arg u1) with
      | Ops.Const (`Float a), Ops.Const (`Float b) ->
          Dtype.equal (Ops.dtype u0) (Ops.dtype u1) && near a b
      | _ -> Ops.equal u0 u1)

let replay test k r =
  let msg = Printf.sprintf "%s, record %d (%s)" test k r.kind in
  let w =
    if
      List.exists
        (fun u -> List.exists (Op.equal (Ops.op u)) host_libm)
        (Ops.toposort ~calls:Enter r.input)
    then folded_within_ulp
    else uop
  in
  equal ~msg (option w) r.result (simplification r.kind r.input);
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
