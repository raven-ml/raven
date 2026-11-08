(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* Sets

   A drawn case is an operation over sets [a] and [b], rows of a group [g] and
   a period [(s, e)] whose values come in either order, are equal or null, and
   draw from few values, so that ends meet often. Integer values are floats
   with no fraction. *)

type op = Merge | Intersect | Subtract
type row = { g : int option; s : float option; e : float option }
type case = { floats : bool; keyed : bool; op : op; a : row array; b : row array }

let ops = [ (Merge, "merge"); (Intersect, "intersect"); (Subtract, "subtract") ]

let values floats =
  if floats then [ neg_infinity; -1.; -0.; 0.; 1.5; 3.; infinity; nan ]
  else [ -2.; 0.; 1.; 3.; 5. ]

(* Every value, and a point between each two. *)
let grid floats =
  if floats then [ neg_infinity; -1.; -0.5; -0.; 0.; 0.75; 1.5; 2.; 3.; 4.; infinity; nan ]
  else List.init 10 (fun i -> Float.of_int (i - 3))

let case =
  let open Gen in
  let* floats = bool in
  let nullable g = frequency [ (1, constant None); (6, map Option.some g) ] in
  let value = nullable (of_list (values floats)) in
  let row = map (fun (g, (s, e)) -> { g; s; e }) (pair (nullable (int_range 0 1)) (pair value value)) in
  let set = array ~size:(frequency [ (1, int_range 0 1); (5, int_range 2 8) ]) row in
  let* a, b = pair set set in
  let+ op, keyed = pair (of_list (List.map fst ops)) bool in
  { floats; keyed; op; a; b }

let column floats v =
  if floats then Column.of_options Type.float64 v
  else Column.of_options Type.int64 (Array.map (Option.map Float.to_int) v)

let table floats rows =
  let col f = column floats (Array.map f rows) in
  Query.of_table
    (Talon.v ~rows:(Array.length rows)
       [ ("g", Column.of_options Type.int64 (Array.map (fun r -> r.g) rows));
         ("s", col (fun r -> r.s)); ("e", col (fun r -> r.e)) ])

let exec q = Error.get_ok (Query.run q)

let pp ppf c =
  Format.fprintf ppf "%s by=%b@.%a@.%a" (List.assoc c.op ops) c.keyed Talon.pp
    (exec (table c.floats c.a)) Talon.pp (exec (table c.floats c.b))

let apply c =
  let by = if c.keyed then [ "g" ] else [] and t = table c.floats in
  match c.op with
  | Merge -> Kit.merge_intervals ~by ("s", "e") (t c.a)
  | Intersect -> Kit.intersect_intervals ~by ("s", "e") (t c.b) (t c.a)
  | Subtract -> Kit.subtract_intervals ~by ("s", "e") (t c.b) (t c.a)

(* The reference *)

(* Talon's total order: NaN after every number, [-0.] equal to [0.]. *)
let order a b =
  match (Float.is_nan a, Float.is_nan b) with
  | false, false -> Float.compare (a +. 0.) (b +. 0.)
  | na, nb -> Bool.compare na nb

(* [holds c rows g x] is whether a row of [rows] in the group [g] holds [x]:
   its period runs from the lesser of its values, included, to the greater,
   excluded. *)
let holds c rows g x =
  let holds r =
    match (r.s, r.e) with
    | Some s, Some e ->
        let lo, hi = if order s e <= 0 then (s, e) else (e, s) in
        ((not c.keyed) || r.g = g) && order lo x <= 0 && order x hi < 0
    | _ -> false
  in
  Array.exists holds rows

let member c g x =
  let a = holds c c.a g x and b = holds c c.b g x in
  match c.op with Merge -> a | Intersect -> a && b | Subtract -> a && not b

let floats_of t name =
  let c = Talon.column t name in
  match Column.type_ c with
  | Any Int64 -> Array.map (Option.map Float.of_int) (Column.options Kind.int c)
  | _ -> Column.options Kind.float c

(* [normal t ~keyed] is whether [t]'s rows are sorted by group, nulls last,
   then by start, each holds a positive length, and each stops before the
   next of its group starts. *)
let normal t ~keyed =
  let s = Array.map Option.get (floats_of t "s") in
  let e = Array.map Option.get (floats_of t "e") in
  let g = if keyed then Column.options Kind.int (Talon.column t "g") else Array.map (fun _ -> None) s in
  let group i = match g.(i) with None -> (1, 0) | Some v -> (0, v) in
  let ok i =
    order s.(i) e.(i) < 0
    && (i = 0
       ||
       let c = compare (group (i - 1)) (group i) in
       c < 0 || (c = 0 && order e.(i - 1) s.(i) < 0))
  in
  List.for_all ok (List.init (Array.length s) Fun.id)

let set_law c =
  let got = exec (apply c) in
  let groups = if c.keyed then [ None; Some 0; Some 1 ] else [ None ] in
  let points = List.concat_map (fun g -> List.map (fun x -> (g, x)) (grid c.floats)) groups in
  (* A point [x] is in the result where the period [(x, x)] overlaps a row. *)
  let probes =
    let col f = Array.of_list (List.map f points) in
    Talon.v
      [ ("p", Column.v Type.int64 (Array.init (List.length points) Fun.id));
        ("pg", Column.of_options Type.int64 (col fst));
        ("x", column c.floats (col (fun (_, x) -> Some x))) ]
  in
  let on = Join.(overlaps ("x", "x") ("s", "e")) in
  let on = if c.keyed then Join.(eq "pg" "g" && on) else on in
  let hits = exec (Query.join ~kind:Semi ~on (Query.of_table got) (Query.of_table probes)) in
  let hit = Column.options Kind.int (Talon.column hits "p") in
  let rows = Array.append c.a c.b in
  let ends f r = match (r.s, r.e) with Some s, Some e -> f (order s e) | _ -> false in
  cover (List.assoc c.op ops) true;
  cover "floats" c.floats;
  cover "a reversed period" (Array.exists (ends (fun o -> o > 0)) rows);
  cover "a zero-length period" (Array.exists (ends (fun o -> o = 0)) rows);
  cover "a null value" (Array.exists (fun r -> r.s = None || r.e = None) rows);
  cover "an empty result" (Talon.rows got = 0);
  cover "two pieces or more" (Talon.rows got >= 2);
  equal ~msg:"normal form" bool true (normal got ~keyed:c.keyed);
  let pp ppf ((g, x), held) =
    Format.fprintf ppf "%a %h %b" (Format.pp_print_option Format.pp_print_int) g x held
  in
  let point = Testable.make ~pp ~equal:(fun a b -> compare a b = 0) in
  equal (list point)
    (List.map (fun (g, x) -> ((g, x), member c g x)) points)
    (List.mapi (fun i p -> (p, Array.mem (Some i) hit)) points)

(* Cases *)

let pieces q =
  let t = exec q in
  List.combine (Array.to_list (floats_of t "s")) (Array.to_list (floats_of t "e"))

let periods l = table true (Array.of_list (List.map (fun (s, e) -> { g = None; s; e }) l))

let cases () =
  let span = list (pair (option float_exact) (option float_exact)) in
  let some = List.map (fun (s, e) -> (Some s, Some e)) in
  let a = periods ((None, Some 2.) :: some [ (0., 3.); (3., 5.); (7., 7.); (9., 8.) ]) in
  let b = periods (some [ (1., 2.); (4.5, 8.5) ]) in
  equal ~msg:"merge" span (some [ (0., 5.); (8., 9.) ]) (pieces (Kit.merge_intervals ("s", "e") a));
  equal ~msg:"intersect" span (some [ (1., 2.); (4.5, 5.); (8., 8.5) ])
    (pieces (Kit.intersect_intervals ("s", "e") b a));
  equal ~msg:"subtract" span (some [ (0., 1.); (2., 4.5); (8.5, 9.) ])
    (pieces (Kit.subtract_intervals ("s", "e") b a))

let () =
  exit
    (run "intervals"
       [
         prop ~count:300 "a result is normal and holds the operation's points"
           (Gen.with_pp pp case) set_law;
         test "touching, zero-length, reversed and null periods" cases;
       ])
