(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let exec q = require_ok ~pp:Error.pp (Query.run q)
let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal

(* Sets

   A drawn case is an operation over sets [a] and [b]: rows of a key [g] and a
   period [(s, e)] whose values come in either order, are equal or null, and
   draw from few values, so that ends meet often. Integer values are floats with
   no fraction. Float keys draw [-0.], [0.] and NaN, which are two keys by
   identity. *)

type op = Merge | Intersect | Subtract
type keys = Unkeyed | Ints | Floats
type row = { g : float option; s : float option; e : float option }

type case = {
  floats : bool;
  keys : keys;
  op : op;
  a : row array;
  b : row array;
}

let ops = [ (Merge, "merge"); (Intersect, "intersect"); (Subtract, "subtract") ]

let values floats =
  if floats then [ neg_infinity; -1.; -0.; 0.; 1.5; 3.; infinity; nan ]
  else [ -2.; 0.; 1.; 3.; 5. ]

let key_values = function
  | Unkeyed -> []
  | Ints -> [ 0.; 1. ]
  | Floats -> [ -0.; 0.; nan ]

(* Every value, and a point between each two. *)
let grid floats =
  if floats then
    [ neg_infinity; -1.; -0.5; -0.; 0.; 0.75; 1.5; 2.; 3.; 4.; infinity; nan ]
  else List.init 10 (fun i -> Float.of_int (i - 3))

let case =
  let open Gen in
  let nullable g = frequency [ (1, constant None); (6, map Option.some g) ] in
  let* floats = bool in
  let* keys =
    frequency
      [ (2, constant Unkeyed); (1, constant Ints); (1, constant Floats) ]
  in
  let value = nullable (of_list (values floats)) in
  let key =
    if keys = Unkeyed then constant None
    else nullable (of_list (key_values keys))
  in
  let row =
    map (fun (g, (s, e)) -> { g; s; e }) (pair key (pair value value))
  in
  let set =
    array ~size:(frequency [ (1, int_range 0 1); (5, int_range 2 8) ]) row
  in
  let* a, b = pair set set in
  let+ op = of_list (List.map fst ops) in
  { floats; keys; op; a; b }

let column floats v =
  if floats then Column.of_options Type.float64 v
  else Column.of_options Type.int64 (Array.map (Option.map Float.to_int) v)

let table c rows =
  let col floats f = column floats (Array.map f rows) in
  Query.of_table
    (Talon.v ~rows:(Array.length rows)
       [
         ("g", col (c.keys = Floats) (fun r -> r.g));
         ("s", col c.floats (fun r -> r.s));
         ("e", col c.floats (fun r -> r.e));
       ])

let pp ppf c =
  Format.fprintf ppf "%s@.%a@.%a" (List.assoc c.op ops) Talon.pp
    (exec (table c c.a))
    Talon.pp
    (exec (table c c.b))

let drawn = Gen.with_pp pp case

let apply c =
  let by = if c.keys = Unkeyed then [] else [ "g" ] and t = table c in
  match c.op with
  | Merge -> Kit.merge_intervals ~by ("s", "e") (Query.append (t c.b) (t c.a))
  | Intersect -> Kit.intersect_intervals ~by ("s", "e") (t c.b) (t c.a)
  | Subtract -> Kit.subtract_intervals ~by ("s", "e") (t c.b) (t c.a)

(* The reference *)

(* Talon's total order: NaN after every number, [-0.] equal to [0.]. *)
let order a b =
  match (Float.is_nan a, Float.is_nan b) with
  | false, false -> Float.compare (a +. 0.) (b +. 0.)
  | na, nb -> Bool.compare na nb

let same_key = Option.equal (fun a b -> order a b = 0)

(* [holds g x (lo, hi, rg)] is whether the period [lo, hi) of key [rg] holds [x]
   of key [g]. *)
let holds g x (lo, hi, rg) = same_key rg g && order lo x <= 0 && order x hi < 0

(* A row's period runs from the lesser of its values to the greater. *)
let periods rows =
  Array.to_list rows
  |> List.filter_map (fun r ->
      match (r.s, r.e) with
      | Some s, Some e ->
          Some (if order s e <= 0 then (s, e, r.g) else (e, s, r.g))
      | _ -> None)

let member c g x =
  let a = List.exists (holds g x) (periods c.a) in
  let b = List.exists (holds g x) (periods c.b) in
  match c.op with
  | Merge -> a || b
  | Intersect -> a && b
  | Subtract -> a && not b

let floats_of t name =
  let c = Talon.column t name in
  match Column.type_ c with
  | Any Int64 -> Array.map (Option.map Float.of_int) (Column.options Kind.int c)
  | _ -> Column.options Kind.float c

(* The result's periods, [None] where a value is null. *)
let pieces t =
  let s = floats_of t "s" and e = floats_of t "e" in
  let keyed = List.mem_assoc "g" (Schema.columns (Talon.schema t)) in
  let g = if keyed then floats_of t "g" else Array.map (fun _ -> None) s in
  List.init (Talon.rows t) (fun i ->
      match (s.(i), e.(i)) with
      | Some s, Some e -> Some (s, e, g.(i))
      | _ -> None)

(* Keys ascending, nulls last, then periods by start, each starting before it
   stops and stopping before the next of its key starts. *)
let normal t =
  let key_order a b =
    match (a, b) with
    | None, None -> 0
    | None, Some _ -> 1
    | Some _, None -> -1
    | Some a, Some b -> order a b
  in
  let rec ordered = function
    | Some (s0, e0, g0) :: (Some (s1, _, g1) :: _ as rest) ->
        let k = key_order g0 g1 in
        order s0 e0 < 0 && (k < 0 || (k = 0 && order e0 s1 < 0)) && ordered rest
    | [ Some (s, e, _) ] -> order s e < 0
    | [] -> true
    | _ -> false
  in
  ordered (pieces t)

let probe_keys c =
  if c.keys = Unkeyed then [ None ]
  else None :: List.map Option.some (key_values c.keys)

let cover_case c got =
  let rows = Array.append c.a c.b in
  let ends f r =
    match (r.s, r.e) with Some s, Some e -> f (order s e) | _ -> false
  in
  let keyed p =
    Array.exists (fun r -> Option.fold ~none:false ~some:p r.g) rows
  in
  let zero neg g = g = 0. && Float.sign_bit g = neg in
  cover (List.assoc c.op ops) true;
  cover "floats" c.floats;
  cover "a reversed period" (Array.exists (ends (fun o -> o > 0)) rows);
  cover "a zero-length period" (Array.exists (ends (fun o -> o = 0)) rows);
  cover "a null value" (Array.exists (fun r -> r.s = None || r.e = None) rows);
  cover "a -0. and a 0. key" (keyed (zero true) && keyed (zero false));
  cover "a NaN key" (keyed Float.is_nan);
  cover "an empty result" (Talon.rows got = 0);
  cover "two pieces or more" (Talon.rows got >= 2)

let normal_law c =
  let got = exec (apply c) in
  cover_case c got;
  satisfies table_w ~claim:"in normal form" normal got

let points_law c =
  let result = exec (apply c) in
  cover_case c result;
  let got = List.filter_map Fun.id (pieces result) in
  let points =
    List.concat_map
      (fun g -> List.map (fun x -> (g, x)) (grid c.floats))
      (probe_keys c)
  in
  let point =
    Testable.make
      ~pp:(fun ppf ((g, x), held) ->
        Format.fprintf ppf "%a %h %b"
          (Format.pp_print_option (fun ppf -> Format.fprintf ppf "%h"))
          g x held)
      ~equal:(fun a b -> compare a b = 0)
  in
  equal (list point)
    (List.map (fun (g, x) -> ((g, x), member c g x)) points)
    (List.map (fun (g, x) -> ((g, x), List.exists (holds g x) got)) points)

(* Cases from the definition: a period holds its start, not its stop; a
   zero-length one adds nothing; a reversed one runs from its lesser value; one
   with a null holds none. *)

let periods_table l =
  Query.of_table
    (Talon.v
       [
         ("s", Column.of_options Type.float64 (Array.of_list (List.map fst l)));
         ("e", Column.of_options Type.float64 (Array.of_list (List.map snd l)));
       ])

let spans t =
  let t = exec t in
  List.combine
    (Array.to_list (floats_of t "s"))
    (Array.to_list (floats_of t "e"))

let some = List.map (fun (s, e) -> (Some s, Some e))

let a =
  periods_table
    ((None, Some 2.) :: some [ (0., 3.); (3., 5.); (7., 7.); (9., 8.) ])

let b = periods_table (some [ (1., 2.); (4.5, 8.5) ])

let worked =
  cases "touching, zero-length, reversed and null periods" ~id:fst
    [
      ("merge", (Kit.merge_intervals ("s", "e") a, some [ (0., 5.); (8., 9.) ]));
      ( "intersect",
        ( Kit.intersect_intervals ("s", "e") b a,
          some [ (1., 2.); (4.5, 5.); (8., 8.5) ] ) );
      ( "subtract",
        ( Kit.subtract_intervals ("s", "e") b a,
          some [ (0., 1.); (2., 4.5); (8.5, 9.) ] ) );
    ]
    (fun (_, (q, want)) ->
      equal
        (list (pair (option float_exact) (option float_exact)))
        want (spans q))

let ints ty xs = Column.v ty xs

let () =
  exit
    (run "Intervals"
       [
         group "merge_intervals, intersect_intervals, subtract_intervals"
           [
             prop ~count:300 "return a set in normal form" drawn normal_law;
             prop ~count:300 "hold the values the operation keeps" drawn
               points_law;
             worked;
             test "columns are by, s and e at the inputs' common type"
               (fun () ->
                 let set ty =
                   Query.of_table
                     (Talon.v
                        [
                          ("g", ints Type.int64 [| 1 |]);
                          ("s", ints ty [| 0 |]);
                          ("e", ints ty [| 5 |]);
                        ])
                 in
                 let got =
                   Kit.intersect_intervals ~by:[ "g" ] ("s", "e")
                     (set Type.int64) (set Type.int32)
                 in
                 equal
                   (Testable.make ~pp:Schema.pp ~equal:Schema.equal)
                   (Schema.v
                      [
                        ("g", Type.Any Type.int64);
                        ("s", Type.Any Type.int64);
                        ("e", Type.Any Type.int64);
                      ])
                   (Talon.schema (exec got)));
             test
               "an impulse matches the period that holds it, the Anti join the \
                others" (fun () ->
                 let set =
                   Kit.merge_intervals ("s", "e")
                     (Query.of_table
                        (Talon.v
                           [
                             ("s", ints Type.int64 [| 0 |]);
                             ("e", ints Type.int64 [| 5 |]);
                           ]))
                 in
                 let events =
                   Query.of_table
                     (Talon.v
                        [
                          ( "t",
                            Column.of_options Type.int64
                              [| Some (-1); Some 0; Some 3; Some 5; None |] );
                        ])
                 in
                 let on = Join.overlaps ("t", "t") ("s", "e") in
                 let ts kind =
                   Column.options Kind.int
                     (Talon.column (exec (Query.join ~kind ~on set events)) "t")
                 in
                 equal (array (option int)) [| Some 0; Some 3 |] (ts Semi);
                 equal
                   (array (option int))
                   [| Some (-1); Some 5; None |]
                   (ts Anti));
             (let q = periods_table [] in
              cases "raise Invalid_argument if by names a column twice" ~id:fst
                [
                  ( "Kit.merge_intervals",
                    fun () -> Kit.merge_intervals ~by:[ "s"; "s" ] ("s", "e") q
                  );
                  ( "Kit.intersect_intervals",
                    fun () ->
                      Kit.intersect_intervals ~by:[ "s"; "s" ] ("s", "e") q q );
                  ( "Kit.subtract_intervals",
                    fun () ->
                      Kit.subtract_intervals ~by:[ "s"; "s" ] ("s", "e") q q );
                ]
                (fun (sub, f) -> raises_match (Exn.invalid_arg ~sub) f));
           ];
       ])
