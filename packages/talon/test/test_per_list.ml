(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

(* Lists

   A list is [None] when null, its elements [None] when null. [column] lays rows
   out by hand, to reach what [Column.of_options] does not build: null elements,
   [lead] values before the first row, and a null row over a non-empty segment
   of [junk]. *)

type 'a lists = 'a option array option array

let column = G.list_column
let ints = Column.of_options Type.int64
let exec q = require_ok ~pp:Error.pp (Query.run q)
let out q = Talon.column (exec q) "v"

(* [same want got] compares two columns of one type, floats by their bits. *)
let same want got =
  let (Type.Any ty) = Column.type_ got in
  let k = Type.kind ty in
  equal
    (array (option (G.witness ty)))
    (Column.options k want) (Column.options k got)

(* Draws

   Lists are empty, null or of one element often enough that every run reaches
   each case. *)

let lists_of element =
  let elements n = Gen.array ~size:(Gen.constant n) element in
  Gen.frequency
    [
      (1, Gen.constant None);
      (1, Gen.constant (Some [||]));
      (1, Gen.map Option.some (elements 1));
      ( 4,
        Gen.bind (Gen.int_range 2 5) (fun n -> Gen.map Option.some (elements n))
      );
    ]

let list_of g = lists_of (Gen.option g)
let some_rows g = Gen.array ~size:(Gen.int_range 0 8) g
let small = Gen.int_range (-5) 5

(* Integers whose sums over a list, and products with [small], stay within
   OCaml's [int], which the comparisons decode to. *)
let edge = 1 lsl 57

let int64s =
  Gen.frequency
    [
      (3, small);
      (2, Gen.int_range (-edge) edge);
      (1, Gen.of_list [ -edge; edge; 0 ]);
    ]

let cover_lists (rows : _ lists) =
  let has f = Array.exists f rows in
  cover "a null list" (has Option.is_none);
  cover "an empty list" (has (fun r -> r = Some [||]));
  cover "a list of one element"
    (has (function Some [| _ |] -> true | _ -> false));
  cover "a null element"
    (has (function Some xs -> Array.exists Option.is_none xs | None -> false))

(* Bodies, drawn over an element [x] and the row's [w]. *)

type body = Body : string * ('b, Expr.agg) Expr.t -> body

let bodies : type a. a Type.t -> (a, Expr.row) Expr.t -> body list =
 fun ty x ->
  let w = Col.int "w" in
  let common =
    Expr.
      [
        Body ("count", count x);
        Body ("rows", rows);
        Body ("first", first x);
        Body ("last", last x);
        Body ("n_unique", n_unique x);
        Body ("min", min x);
        Body ("max", max x);
        Body ("arg_min", arg_min x);
        Body ("arg_max", arg_max x);
      ]
  in
  let k = Type.kind ty in
  match (Kind.provably_equal k Kind.int, Kind.provably_equal k Kind.float) with
  | Some Stdlib.Type.Equal, _ ->
      common
      @ Expr.
          [
            Body ("sum", sum x);
            Body ("mean", mean x);
            Body ("median", median x);
            Body ("var", var x);
            Body ("std", std x);
            Body ("sum of x * w", sum (x * w));
            Body ("cadence", sum (x - shift 1 x));
            Body ("rank", sum (rank x));
            Body ("over", sum (x - over (max x)));
            Body ("window", max (window (sum x)));
          ]
  | _, Some Stdlib.Type.Equal ->
      (* Float sums and means have a law of their own below. *)
      common
      @ Expr.
          [
            Body ("median", median x); Body ("var", var x); Body ("std", std x);
          ]
  | _ -> common

type case =
  | Case : {
      ty : 'a Type.t;
      rows : 'a lists;
      w : int option array;
      lead : int;
      junk : 'a option;
    }
      -> case

let case_of (type a) (ty : a Type.t) (value : a Gen.t) =
  Gen.(
    let* rows = some_rows (list_of value) in
    let* w = array ~size:(constant (Array.length rows)) (option small) in
    let* junk = option value in
    let+ lead = int_range 0 2 in
    Case { ty; rows; w; lead; junk })

let drawn =
  Gen.one_of
    [
      case_of Type.int64 int64s;
      case_of Type.float64 (Option.get (G.value Type.float64));
      case_of Type.string (Option.get (G.value Type.string));
      case_of Type.date (Option.get (G.value Type.date));
    ]

let table (Case c) =
  v [ ("xs", column c.ty ~lead:c.lead ?junk:c.junk c.rows); ("w", ints c.w) ]

let handle ty = Col.v (Kind.list (Type.kind ty)) "xs"

(* [per_lists c] is each body of [c]'s type as a per_list over [xs], under the
   body's name. *)
let per_lists (Case c) =
  List.map
    (fun (Body (name, e)) -> Expr.(name := per_list [ "x" := handle c.ty ] e))
    (bodies c.ty (Col.v (Type.kind c.ty) "x"))

(* [reference ty w elements] is every body over one row's list as a frame: its
   [elements] under the element's name, the row's [w] repeated. *)
let reference (type a) (ty : a Type.t) w (elements : a option array) =
  let n = Array.length elements in
  let t =
    v [ ("w", ints (Array.make n w)); ("x", Column.of_options ty elements) ]
  in
  let bodies = bodies ty (Col.v (Type.kind ty) "x") in
  exec
    (Query.aggregate ~by:[]
       (List.map (fun (Body (name, e)) -> Expr.(name := e)) bodies)
       (Query.of_table t))

(* [cell got i want name] states that [got]'s column [name] holds at row [i]
   what [want]'s holds at row 0, or null where [want] is [None]. *)
let cell got i want name =
  let c = Talon.column got name in
  let (Type.Any ty) = Column.type_ c in
  let k = Type.kind ty in
  let expected =
    Option.bind want (fun w -> (Column.options k (Talon.column w name)).(0))
  in
  equal
    ~msg:(Printf.sprintf "%s at row %d" name i)
    (option (G.witness ty))
    expected
    (Column.options k c).(i)

let definition =
  prop ~count:300 "is each body over each row's list" drawn
    (fun (Case c as case) ->
      cover_lists c.rows;
      cover "a null slot over a non-empty segment"
        (Option.is_some c.junk && Array.exists Option.is_none c.rows);
      cover "offsets that do not start at 0"
        (Option.is_some c.junk && c.lead > 0);
      let got =
        exec (Query.derive (per_lists case) (Query.of_table (table case)))
      in
      let names =
        List.map
          (fun (Body (n, _)) -> n)
          (bodies c.ty (Col.v (Type.kind c.ty) "x"))
      in
      Array.iteri
        (fun i r ->
          let want = Option.map (reference c.ty c.w.(i)) r in
          List.iter (cell got i want) names)
        c.rows)

let batching =
  prop "does not depend on batches"
    (Gen.bind drawn (fun case ->
         Gen.map (fun t -> (case, t)) (G.split (table case))))
    (fun (case, split) ->
      let q t = exec (Query.derive (per_lists case) (Query.of_table t)) in
      cover "several batches" (List.length (batches split) > 1);
      let whole = q (table case) and parts = q split in
      let (Case c) = case in
      List.iter
        (fun (Body (n, _)) ->
          same (Talon.column whole n) (Talon.column parts n))
        (bodies c.ty (Col.v (Type.kind c.ty) "x")))

(* Lists of lists *)

(* [nested rows] is a column of lists of lists of int64, its inner lists holding
   null elements. *)
let nested (rows : int option array option array option array) =
  let inner =
    Array.concat
      (Array.to_list (Array.map (function Some l -> l | None -> [||]) rows))
  in
  let offsets = Array.make (Array.length rows + 1) 0L in
  Array.iteri
    (fun i r ->
      let n = match r with Some l -> Array.length l | None -> 0 in
      offsets.(i + 1) <- Int64.add offsets.(i) (Int64.of_int n))
    rows;
  let valid = Array.map Option.is_some rows in
  let validity =
    if Array.for_all Fun.id valid then None
    else Some (Nx.create Nx.bit [| Array.length valid |] valid)
  in
  let offsets = Nx.create Nx.int64 [| Array.length offsets |] offsets in
  Result.get_ok
    (Column.of_layout
       (Any (Type.list (Type.list Type.int64)))
       (Varsize { validity; offsets; child = column Type.int64 inner }))

type nested_form = Sum | Min | Max | Sum_of_counts

let nesting =
  let ll = Col.v Kind.(list (list int)) "ll"
  and l = Col.v Kind.(list int) "l"
  and x = Col.int "x" in
  let inner r = Expr.(per_list [ "x" := l ] (r x)) in
  let expression =
    Expr.(
      function
      | Sum -> per_list [ "l" := ll ] (sum (inner sum))
      | Min -> per_list [ "l" := ll ] (min (inner min))
      | Max -> per_list [ "l" := ll ] (max (inner max))
      | Sum_of_counts -> per_list [ "l" := ll ] (sum (inner count)))
  in
  let extreme f = function
    | [] -> None
    | x :: xs -> Some (List.fold_left f x xs)
  in
  let expected = function
    | Sum -> fun xs -> Some (List.fold_left ( + ) 0 xs)
    | Min -> extreme Stdlib.min
    | Max -> extreme Stdlib.max
    | Sum_of_counts -> fun xs -> Some (List.length xs)
  in
  let forms =
    Gen.with_pp
      (fun ppf f ->
        Format.pp_print_string ppf
          (match f with
          | Sum -> "sum"
          | Min -> "min"
          | Max -> "max"
          | Sum_of_counts -> "sum of counts"))
      (Gen.of_list [ Sum; Min; Max; Sum_of_counts ])
  in
  prop "a frame of inner lists reduces their flattened values"
    (Gen.pair forms (some_rows (lists_of (list_of small))))
    (fun (form, rows) ->
      let flat l =
        List.concat_map
          (function
            | Some xs -> List.filter_map Fun.id (Array.to_list xs) | None -> [])
          (Array.to_list l)
      in
      let inners =
        Array.concat
          (Array.to_list
             (Array.map (function Some l -> l | None -> [||]) rows))
      in
      cover "an empty inner list" (Array.exists (fun r -> r = Some [||]) inners);
      cover "a null inner list" (Array.exists Option.is_none inners);
      let t = v [ ("ll", nested rows) ] in
      let got =
        Column.options Kind.int
          (out
             (Query.derive Expr.[ "v" := expression form ] (Query.of_table t)))
      in
      equal
        (array (option int))
        (Array.map
           (Option.fold ~none:None ~some:(fun l -> expected form (flat l)))
           rows)
        got)

(* Float sums and means

   A float list's sum adds its values in list order from 0, and its mean divides
   that sum, taken in float64, by their count. The values stay far from
   overflow; their var and std are any frame's, which the definition law states
   against an aggregate. *)

let float_values =
  Gen.frequency
    [
      (1, Gen.of_list [ -0.; 0.; infinity; neg_infinity; nan ]);
      (6, Gen.float_range (-1e6) 1e6);
    ]

let pp_float_lists ppf rows =
  let value = function Some x -> Printf.sprintf "%h" x | None -> "null" in
  let row = function
    | Some l ->
        "[" ^ String.concat "; " (Array.to_list (Array.map value l)) ^ "]"
    | None -> "null"
  in
  Format.pp_print_string ppf
    (String.concat " " (Array.to_list (Array.map row rows)))

let sum_in_order xs = List.fold_left ( +. ) 0. xs

let float_sums =
  let x = Col.float "x" and xs = Col.v Kind.(list float) "xs" in
  let means =
    Gen.with_pp
      (fun ppf m -> Format.pp_print_string ppf (if m then "mean" else "sum"))
      Gen.bool
  in
  prop "a float list's sum and mean add in list order"
    (Gen.pair means
       (Gen.with_pp pp_float_lists (some_rows (list_of float_values))))
    (fun (mean, rows) ->
      cover_lists rows;
      let want = function
        | None -> None
        | Some l -> (
            let vs = List.filter_map Fun.id (Array.to_list l) in
            match (mean, vs) with
            | false, _ -> Some (sum_in_order vs)
            | true, [] -> None
            | true, _ -> Some (sum_in_order vs /. Float.of_int (List.length vs))
            )
      in
      let e = if mean then Expr.mean x else Expr.sum x in
      let t = v [ ("xs", column Type.float64 rows) ] in
      equal
        (array (option float_exact))
        (Array.map want rows)
        (Column.options Kind.float
           (out
              (Query.derive
                 Expr.[ "v" := per_list [ "x" := xs ] e ]
                 (Query.of_table t)))))

(* Lockstep *)

let pairs =
  (* Two lists of one row, usually of one length. *)
  let row =
    Gen.(
      let* n = int_range 0 4 in
      let list n = array ~size:(constant n) (option small) in
      let* a = list n in
      let* m = frequency [ (6, constant n); (1, int_range 0 5) ] in
      let+ b = list m in
      (a, b))
  in
  let nullable g =
    Gen.frequency [ (1, Gen.constant None); (6, Gen.map Option.some g) ]
  in
  Gen.(
    triple
      (some_rows (pair (nullable row) bool))
      (int_range 0 2) (int_range 0 2))

(* A row whose two lists have one and two elements. *)
let unequal_lengths =
  ([| (Some ([| Some 1 |], [| Some 1; Some 2 |]), false) |], 0, 0)

let lockstep_columns (rows, _, _) =
  let a = Array.map (fun (r, _) -> Option.map fst r) rows
  and b =
    Array.map
      (fun (r, null_b) ->
        match r with Some (_, b) when not null_b -> Some b | _ -> None)
      rows
  in
  (a, b)

(* [lockstep_table (rows, la, lb)] lays the two lists over children of their
   own, with [la] and [lb] values before their first rows and null rows over
   non-empty segments, so that no list's offsets serve the other. *)
let lockstep_table ((_, la, lb) as draw) =
  let xs, ys = lockstep_columns draw in
  ( xs,
    ys,
    [
      ("as_", column Type.int64 ~lead:la ~junk:9 xs);
      ("bs", column Type.int64 ~lead:lb ~junk:9 ys);
    ] )

let as_ = Col.v Kind.(list int) "as_"
and bs = Col.v Kind.(list int) "bs"

let a = Col.int "a"
and b = Col.int "b"

let lockstep_body = Expr.(sum (a * b))

let lockstep =
  prop "lists of one row are read in lockstep or fail at the row"
    ~examples:[ unequal_lengths ] pairs (fun draw ->
      let xs, ys, cols = lockstep_table draw in
      let t = v cols in
      let _, la, lb = draw in
      cover "lists whose offsets differ" (la <> lb);
      let q =
        Query.derive
          Expr.[ "v" := per_list [ "a" := as_; "b" := bs ] lockstep_body ]
          (Query.of_table t)
      in
      let mismatch =
        Array.find_index Fun.id
          (Array.map2
             (fun x y ->
               match (x, y) with
               | Some x, Some y -> Array.length x <> Array.length y
               | _ -> false)
             xs ys)
      in
      cover "unequal lengths" (Option.is_some mismatch);
      cover "a null against a list"
        (Array.exists2 (fun x y -> Option.is_none x <> Option.is_none y) xs ys);
      cover "lists in lockstep"
        (Array.exists2
           (fun x y ->
             match (x, y) with
             | Some x, Some y -> Array.length x = Array.length y && x <> [||]
             | _ -> false)
           xs ys);
      match (mismatch, Query.run q) with
      | Some i, r ->
          let e = Format.asprintf "%a" Error.pp (require_error r) in
          contains ~sub:(Printf.sprintf ": row %d: " i) e;
          contains ~sub:{|"a"|} e;
          contains ~sub:{|"b"|} e
      | None, r ->
          let got = Talon.column (require_ok ~pp:Error.pp r) "v" in
          let want =
            Array.map2
              (fun x y ->
                match (x, y) with
                | Some x, Some y ->
                    let t = v [ ("a", ints x); ("b", ints y) ] in
                    (Column.options Kind.int
                       (out
                          (Query.aggregate ~by:[]
                             Expr.[ "v" := lockstep_body ]
                             (Query.of_table t)))).(0)
                | _ -> None)
              xs ys
          in
          equal (array (option int)) want (Column.options Kind.int got))

(* The optimizer

   The element's name is the input's column [t] under a select that renames it,
   so that a pushed predicate could read the one for the other. *)

(* A filter above a select that renames the input's column [t] to [y] and keeps
   the lists [ts]: a pushed predicate could read one for the other. *)
let renamed rows =
  let input =
    v
      [
        ("t", ints (Array.map fst rows));
        ("ts", column Type.int64 (Array.map snd rows));
      ]
  in
  Query.select
    Expr.[ "y" := Col.int "t"; "ts" := Col.v Kind.(list int) "ts" ]
    (Query.of_table input)

let renamed_rows = some_rows (Gen.pair (Gen.option small) (list_of small))

(* [kept_ys p rows] is the [y] of each row of [renamed rows] that [p] keeps. *)
let kept_ys p rows =
  cover_lists (Array.map snd rows);
  Column.options Kind.int
    (Talon.column (exec (Query.filter p (renamed rows))) "y")

let ys_where want rows =
  Array.of_list
    (List.filter_map
       (fun (y, l) -> if want y l then Some y else None)
       (Array.to_list rows))

let element_values l = List.filter_map Fun.id (Array.to_list l)

let ts = Col.v Kind.(list int) "ts"
and t = Col.int "t"
and y = Col.int "y"

let renamed_column =
  prop "a predicate that reads the renamed column stays above the select"
    renamed_rows (fun rows ->
      let want y l =
        match (y, l) with
        | Some y, Some l ->
            List.fold_left (fun s e -> s + (e * y)) 0 (element_values l) > 0
        | _ -> false
      in
      equal
        (array (option int))
        (ys_where want rows)
        (kept_ys Expr.(per_list [ "t" := ts ] (sum (t * y)) > int 0) rows))

let element_only =
  prop "a body that reads only its element moves below the select" renamed_rows
    (fun rows ->
      let want _ l =
        match l with
        | Some l -> List.fold_left ( + ) 0 (element_values l) > 0
        | None -> false
      in
      equal
        (array (option int))
        (ys_where want rows)
        (kept_ys Expr.(per_list [ "t" := ts ] (sum t) > int 0) rows))

let ordered_over =
  prop "an over in a body orders by the output of its name" renamed_rows
    (fun rows ->
      (* In ascending order, each value's predecessor less the value sums to the
         least value less the greatest; nulls sort last. *)
      let want _ l =
        match Option.map element_values l with
        | None | Some [] -> false
        | Some (v :: vs) ->
            List.fold_left Stdlib.min v vs < List.fold_left Stdlib.max v vs
      in
      let p =
        Expr.(
          per_list
            [ "t" := ts ]
            (sum (over ~order:[ Order.asc "t" ] (shift 1 t) - t))
          < int 0)
      in
      equal (array (option int)) (ys_where want rows) (kept_ys p rows))

let lockstep_filter =
  prop "a filter that drops unequal lengths guards the lockstep above it"
    ~examples:[ unequal_lengths ] pairs (fun ((rows, _, _) as draw) ->
      let xs, ys, cols = lockstep_table draw in
      let ids = Array.mapi (fun i _ -> Some i) rows in
      let t = v (("id", ints ids) :: cols) in
      let len name l = Expr.(per_list [ name := l ] rows) in
      let q =
        Query.of_table t
        |> Query.filter Expr.(len "u" as_ = len "v" bs)
        |> Query.filter
             Expr.(per_list [ "a" := as_; "b" := bs ] lockstep_body > int 0)
      in
      let want =
        List.filteri
          (fun i _ ->
            match (xs.(i), ys.(i)) with
            | Some x, Some y when Array.length x = Array.length y ->
                let product a b =
                  match (a, b) with Some a, Some b -> a * b | _ -> 0
                in
                Array.fold_left ( + ) 0 (Array.map2 product x y) > 0
            | _ -> false)
          (Array.to_list ids)
      in
      cover "a filtered mismatch"
        (Array.exists2
           (fun x y ->
             match (x, y) with
             | Some x, Some y -> Array.length x <> Array.length y
             | _ -> false)
           xs ys);
      equal
        (list (option int))
        want
        (Array.to_list (Column.options Kind.int (Talon.column (exec q) "id"))))

(* Cases the rules state *)

let xs_rows = [| Some [| 1; 2 |]; None; Some [||]; Some [| 10; 20 |] |]

let of_ints rows =
  column Type.int64 (Array.map (Option.map (Array.map Option.some)) rows)

let xs = Col.v Kind.(list int) "xs"
let x = Col.int "x"

let eval e =
  let t = v [ ("xs", of_ints xs_rows) ] in
  Column.options Kind.int
    (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))

let problem e =
  let t =
    v
      [
        ("xs", of_ints [| Some [| 1 |] |]);
        ("s", Column.v Type.string [| "a" |]);
        ("ll", nested [| Some [| Some [| Some 1 |] |] |]);
        ("w", Column.v Type.int64 [| 1 |]);
      ]
  in
  match Query.run (Query.derive Expr.[ "v" := e ] (Query.of_table t)) with
  | Ok _ -> "no problem"
  | Error e -> Format.asprintf "%a" Error.pp e
  | exception Invalid_argument m -> m

(* Outputs that would hide a column *)

let hiding =
  let null_ints = array (option int) in
  let t =
    v
      [
        ("ls", of_ints [| Some [| 1; 2 |] |]);
        ("s", Column.v Type.string [| "a,b" |]);
      ]
  in
  let derive e = Query.derive Expr.[ "v" := e ] (Query.of_table t) in
  let renamed = { Expr.column = (fun n c -> Expr.(n ^ "_e" := c)) } in
  let same_name = { Expr.column = (fun n c -> Expr.(n := c)) } in
  let ls = Col.int "ls" in
  [
    test "a renamed list output reads the element" (fun () ->
        equal null_ints [| Some 3 |]
          (Column.options Kind.int
             (out
                (derive
                   Expr.(
                     per_list
                       [ each Sel.(names [ "ls" ]) renamed ]
                       (sum (Col.int "ls_e")))))));
    test "keep outputs a list under its own name, read as its element"
      (fun () ->
        equal null_ints [| Some 3 |]
          (Column.options Kind.int
             (out
                (derive Expr.(per_list [ keep Sel.(names [ "ls" ]) ] (sum ls)))));
        equal null_ints [| Some 2 |]
          (Column.options Kind.int
             (out
                (derive
                   Expr.(
                     per_list [ each Sel.(names [ "ls" ]) same_name ] (count ls))))));
    test "a body that reads a kept list as a list is a problem" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"reads list") (fun () ->
            derive
              Expr.(
                per_list
                  [ keep Sel.(names [ "ls" ]) ]
                  (count (Col.v Kind.(list int) "ls")))));
    test "an output that hides another column is a problem" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"would hide") (fun () ->
            derive
              Expr.(
                per_list
                  [ "ls" := Str.split (Str.literal ",") (Col.string "s") ]
                  rows)));
  ]

let rules =
  let null_ints = array (option int) in
  [
    test "a null list has a null count and rows" (fun () ->
        equal null_ints
          [| Some 2; None; Some 0; Some 2 |]
          (eval Expr.(per_list [ "x" := xs ] (count x)));
        equal null_ints
          [| Some 2; None; Some 0; Some 2 |]
          (eval Expr.(per_list [ "x" := xs ] rows)));
    test "an empty list gives each reduction over no values" (fun () ->
        equal null_ints
          [| Some 3; None; Some 0; Some 30 |]
          (eval Expr.(per_list [ "x" := xs ] (sum x)));
        equal null_ints
          [| Some 1; None; None; Some 10 |]
          (eval Expr.(per_list [ "x" := xs ] (min x))));
    test "a computed list" (fun () ->
        let t =
          v
            [
              ( "s",
                Column.of_options Type.string [| Some "a_b"; None; Some "" |] );
            ]
        in
        let s = Col.string "s" in
        let e = Expr.(per_list [ "p" := Str.split (Str.literal "_") s ] rows) in
        equal null_ints [| Some 2; None; Some 1 |]
          (Column.options Kind.int
             (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))));
    (let failing =
       v
         [
           ("xs", of_ints [| Some [| 1; 1 |]; Some [||]; Some [| 300; 2 |] |]);
           ("w", ints [| Some 0; Some 0; Some 0 |]);
         ]
     in
     cases "a failure in r is at its list's row" ~id:fst
       Expr.
         [
           ("a cast", per_list [ "x" := xs ] (sum (cast Type.int8 x)));
           ("only", per_list [ "x" := xs ] (only x));
           ("over", per_list [ "x" := xs ] (sum (over (only x))));
           ( "over ~order",
             per_list
               [ "x" := xs ]
               (sum (over ~order:[ Order.asc "x" ] (only x))) );
           ("over ~by", per_list [ "x" := xs ] (sum (over ~by:[ "w" ] (only x))));
           ("window", per_list [ "x" := xs ] (sum (window (only x))));
         ]
       (fun (_, e) ->
         let q = Query.derive Expr.[ "v" := e ] (Query.of_table failing) in
         contains ~sub:": row 2: "
           (Format.asprintf "%a" Error.pp (require_error (Query.run q)))));
    test "a row without a list runs no body" (fun () ->
        let f (_ : int) : int = failwith "called" in
        let e =
          Expr.(per_list [ "x" := xs ] (store Type.int64 (const f $ count x)))
        in
        let t = v [ ("xs", of_ints [| None |]) ] in
        equal null_ints [| None |]
          (Column.options Kind.int
             (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))));
    cases "problems"
      ~id:(fun (n, _, _) -> n)
      Expr.
        [
          ( "an output named like an enclosing column",
            problem (per_list [ "w" := xs ] (count x)),
            "would hide" );
          ( "an inner output named like an outer output",
            problem
              (per_list
                 [ "l" := Col.v Kind.(list (list int)) "ll" ]
                 (sum
                    (per_list
                       [ "l" := Str.split (Str.literal ",") (Col.string "s") ]
                       rows))),
            "would hide" );
          ( "an output that is not a list",
            problem (per_list [ "x" := Col.int "w" ] (count x)),
            "is not a list" );
          ("no output", problem (per_list [] rows), "at least one output");
          ( "two outputs of one name",
            problem (per_list [ "x" := xs; "x" := xs ] rows),
            "named twice" );
          ( "a shift that reads no output",
            problem (per_list [ "x" := xs ] (sum (shift 1 (Col.int "w")))),
            "reads no output" );
          ( "an over that reads no output",
            problem
              (per_list [ "x" := xs ] (sum (x + over (max (Col.int "w"))))),
            "reads no output" );
          ( "a window of rows",
            problem (per_list [ "x" := xs ] (sum (window rows))),
            "reads no output" );
          ( "a rank that reads no output",
            problem (per_list [ "x" := xs ] (sum (rank (Col.int "w")))),
            "reads no output" );
        ]
      (fun (_, got, sub) -> contains ~sub got);
    test "a filter guards an only and a duration sum in a body" (fun () ->
        let dur = Type.duration Type.Ns in
        let span_of = Time.Span.of_ns in
        let table =
          v
            [
              ("id", ints [| Some 0; Some 1 |]);
              ("xs", of_ints [| Some [| 1; 1 |]; Some [| 1; 2 |] |]);
              ( "ds",
                column dur
                  [|
                    Some [| Some (span_of 1L) |];
                    Some
                      [|
                        Some (span_of Int64.max_int);
                        Some (span_of Int64.max_int);
                      |];
                  |] );
            ]
        in
        let keeps p =
          Query.of_table table
          |> Query.filter Expr.(Col.int "id" = int 0)
          |> Query.filter p |> exec
          |> fun t -> Column.options Kind.int (Talon.column t "id")
        in
        equal null_ints [| Some 0 |]
          (keeps Expr.(per_list [ "x" := xs ] (only x) > int 0));
        let ds = Col.v Kind.(list span) "ds" and d = Col.span "d" in
        equal null_ints [| Some 0 |]
          (keeps
             Expr.(per_list [ "d" := ds ] (sum d) > span (Time.Span.of_ns 0L))));
    test "it streams, a batch per batch" (fun () ->
        let batch = v [ ("xs", of_ints [| Some [| 1 |] |]) ] in
        let t = of_batches [ batch; batch ] in
        let q =
          Query.derive
            Expr.[ "v" := per_list [ "x" := xs ] rows ]
            (Query.of_table t)
        in
        equal int 2
          (require_ok ~pp:Error.pp
             (Query.fold q ~init:0 (fun n _ -> Stdlib.(n + 1)))));
    test "prints as written" (fun () ->
        equal string {|per_list ["x" := xs] (sum (x * x))|}
          (Format.asprintf "%a" Expr.pp
             Expr.(per_list [ "x" := xs ] (sum (x * x)))));
    (let t = v [ ("xs", of_ints [| Some [| 1 |] |]) ] in
     let q n =
       Query.derive Expr.[ "v" := per_list [ n := xs ] rows ] (Query.of_table t)
     in
     let query_w = Testable.make ~pp:Query.pp ~equal:Query.equal in
     test "plans compare by their outputs' names" (fun () ->
         equal query_w (q "a") (q "a");
         not_equal query_w (q "a") (q "b")));
    test "a body of a literal is the literal on each row with a list" (fun () ->
        let frame rows =
          let t = v [ ("xs", of_ints rows) ] in
          Column.options Kind.int
            (out
               (Query.derive
                  Expr.[ "v" := per_list [ "x" := xs ] (int 3) ]
                  (Query.of_table t)))
        in
        equal null_ints [| Some 3; Some 3; None |]
          (frame [| Some [| 1 |]; Some [||]; None |]);
        equal null_ints [| None |] (frame [| None |]));
    test "an aggregate reduces a per_list of its groups" (fun () ->
        let t =
          v
            [
              ("g", ints [| Some 0; Some 1; Some 0 |]);
              ("xs", of_ints [| Some [| 1; 2 |]; None; Some [| 10 |] |]);
            ]
        in
        let q =
          Query.aggregate ~by:[ "g" ]
            Expr.[ "v" := sum (per_list [ "x" := xs ] (sum x)) ]
            (Query.of_table t)
        in
        equal null_ints [| Some 13; Some 0 |] (Column.options Kind.int (out q)));
    test "a per_list stands in a window's operand" (fun () ->
        let w = Expr.(window (sum (per_list [ "x" := xs ] (count x)))) in
        equal null_ints [| Some 2; Some 2; Some 2; Some 4 |] (eval w));
    test "an order operation in an output is the window's" (fun () ->
        contains ~sub:"holds shift"
          (problem
             Expr.(window (sum (per_list [ "x" := shift 1 xs ] (count x))))));
    test "a body's over reaches by its output's name" (fun () ->
        equal null_ints
          [| Some 3; None; Some 0; Some 30 |]
          (eval Expr.(per_list [ "x" := xs ] (sum (over ~by:[ "x" ] x)))));
  ]

let () =
  exit
    (run "Per_list"
       [
         group "Expr.per_list"
           ([
              definition;
              batching;
              nesting;
              float_sums;
              lockstep;
              lockstep_filter;
              renamed_column;
              element_only;
              ordered_over;
            ]
           @ rules @ hiding);
       ])
