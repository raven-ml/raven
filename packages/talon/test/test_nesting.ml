(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Records and nesting: [Expr.record], [Expr.field], [Expr.within],
   [Expr.collect] and [Query.unnest]. The laws draw rows with null and empty
   lists, null elements and every type of [G.sample]; a list column's rows are
   laid out by hand to reach null lists over non-empty child segments. *)

open Talon
open Windtrap
module G = Talon_gen

let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal
let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let ints = Column.of_options Type.int64
let exec q = require_ok ~pp:Error.pp (Query.run q)
let problem f = match f () with _ -> "no problem" | exception Invalid_argument m -> m
let out q = Talon.column (exec q) "v"
let opt_ints = array (option int)
let ints_at t n = Column.options Kind.int (Talon.column t n)
let ints_of q = Column.options Kind.int (out q)
let names t = List.map fst (Schema.columns (Talon.schema t))

(* Lists *)

type 'a lists = 'a option array option array

let offsets_of ~lead lengths =
  let o = Array.make (Array.length lengths + 1) (Int64.of_int lead) in
  Array.iteri (fun i l -> o.(i + 1) <- Int64.add o.(i) (Int64.of_int l)) lengths;
  o

let laid_out ty valid offsets child =
  let validity =
    if Array.for_all Fun.id valid then None
    else Some (Nx.create Nx.bit [| Array.length valid |] valid)
  in
  let offsets = Nx.create Nx.int64 [| Array.length offsets |] offsets in
  Result.get_ok
    (Column.of_layout (Any (Type.list ty))
       (Varsize { validity; offsets; child }))

(* [column ty ?junk ?lead rows] is [rows] as a list column of element type [ty];
   a null row spans two [junk] elements, and [lead] of them precede the first
   row, when [junk] is given. *)
let column ty ?junk ?(lead = 0) (rows : 'a lists) =
  let segment = function
    | Some xs -> xs
    | None -> ( match junk with Some v -> [| Some v; Some v |] | None -> [||])
  in
  let segments = Array.map segment rows in
  let pad = match junk with Some v -> Array.make lead (Some v) | None -> [||] in
  let child =
    Column.of_options ty (Array.concat (pad :: Array.to_list segments))
  in
  laid_out ty
    (Array.map Option.is_some rows)
    (offsets_of ~lead:(Array.length pad) (Array.map Array.length segments))
    child

let of_ints rows = column Type.int64 rows
let xs = Col.v Kind.(list int) "xs"
let ys = Col.v Kind.(list int) "ys"
let id = Col.int "id"
let x = Col.int "x"

let positions = function Some l -> Array.length l | None -> 0
(* Draws *)

let small = Gen.int_range (-5) 5

let list_of g =
  let elements n = Gen.array ~size:(Gen.constant n) (Gen.option g) in
  Gen.frequency
    [
      (1, Gen.constant None);
      (1, Gen.constant (Some [||]));
      (1, Gen.map Option.some (elements 1));
      ( 4,
        Gen.bind (Gen.int_range 2 5) (fun n ->
            Gen.map Option.some (elements n)) );
    ]

let rows_of g = Gen.array ~size:(Gen.int_range 0 12) (list_of g)
let cover_lists rows =
  let has f = Array.exists f rows in
  cover "a null list" (has Option.is_none);
  cover "an empty list" (has (fun r -> r = Some [||]));
  cover "a one-element list" (has (function Some [| _ |] -> true | _ -> false));
  cover "a null element"
    (has (function Some xs -> Array.exists Option.is_none xs | None -> false));
  cover "no row" (Array.length rows = 0)

let table ?junk ?lead rows =
  v
    [
      ("id", ints (Array.init (Array.length rows) (fun i -> Some i)));
      ("xs", column Type.int64 ?junk ?lead rows);
    ]

(* Rows with null lists that span child rows and offsets from past 0. *)
let drawn = Gen.triple (rows_of small) (Gen.option small) (Gen.int_range 0 2)
let laid (rows, junk, lead) = table ?junk ~lead rows

let cover_laid (rows, junk, lead) =
  cover_lists rows;
  cover "a null list over child rows"
    (Option.is_some junk && Array.exists Option.is_none rows);
  cover "offsets past 0" (Option.is_some junk && lead > 0)

(* L1 and L2: records *)

let rec holds_ext : type a. a Type.t -> bool = function
  | Ext _ -> true
  | List t -> holds_ext t
  | Record fs -> List.exists (fun (_, Type.Any t) -> holds_ext t) fs
  | _ -> false

let l1 =
  prop ~count:300 "field k n (record [n := e]) is e, nulls included" G.sample
    (fun (G.Sample (ty, rows)) ->
      let k = Type.kind ty in
      match holds_ext ty with
      | true -> () (* A kind reads no extension; within does. *)
      | false ->
          let c = Column.of_options ty rows in
          let t = v [ ("a", c) ] in
          let e = Col.v k "a" in
          let q =
            Query.select
              Expr.[ "a" := field k "a" (record [ "a" := e ]) ]
              (Query.of_table t)
          in
          equal table_w t (exec q))

let l2 =
  prop "a record spread with within and rebuilt is itself, not null"
    (Gen.array ~size:(Gen.int_range 0 10)
       (Gen.pair (Gen.option small) (Gen.option small)))
    (fun rows ->
      cover "a null field" (Array.exists (fun (a, _) -> a = None) rows);
      let t =
        v
          [
            ("a", ints (Array.map fst rows));
            ("b", ints (Array.map snd rows));
          ]
      in
      let r = Expr.record Expr.[ keep Sel.all ] in
      let spread = Expr.(record [ within r (keep Sel.all) ]) in
      let q =
        Query.select Expr.[ "r" := r; "s" := spread ] (Query.of_table t)
      in
      let got = exec q in
      equal table_w
        (v [ ("v", Talon.column got "r") ])
        (v [ ("v", Talon.column got "s") ]))

let null_record =
  (* A null record whose child slots hold values: the field reads null. *)
  let fields =
    [ ("a", ints [| Some 1; Some 2 |]); ("b", ints [| None; Some 4 |]) ]
  in
  let ty = Type.record [ ("a", Any Type.int64); ("b", Any Type.int64) ] in
  let c =
    Result.get_ok
      (Column.of_layout (Any ty)
         (Children
            {
              validity = Some (Nx.create Nx.bit [| 2 |] [| false; true |]);
              length = 2;
              fields;
            }))
  in
  let t = v [ ("r", c) ] in
  let r = Col.v Record.kind "r" in
  [
    test "field is null where the record is, whatever the child holds"
      (fun () ->
        let q =
          Query.select
            Expr.[ "v" := field Kind.int "a" r ]
            (Query.of_table t)
        in
        equal opt_ints [| None; Some 2 |]
          (ints_of q));
    test "field ANDs the record's and the field's nulls" (fun () ->
        let q =
          Query.select
            Expr.[ "v" := field Kind.int "b" r ]
            (Query.of_table t)
        in
        equal opt_ints [| None; Some 4 |]
          (ints_of q));
    test "is_null reads the record, a record of nulls is not null" (fun () ->
        let q =
          Query.select
            Expr.
              [
                "v" := is_null r;
                "w" := is_null (record [ "a" := store Type.int64 null ]);
              ]
            (Query.of_table t)
        in
        let got = exec q in
        equal (array (option bool))
          [| Some true; Some false |]
          (Column.options Kind.bool (Talon.column got "v"));
        equal (array (option bool))
          [| Some false; Some false |]
          (Column.options Kind.bool (Talon.column got "w")));
    test "within spreads null where the record is" (fun () ->
        let q =
          Query.select Expr.[ within r (keep Sel.all) ] (Query.of_table t)
        in
        let got = exec q in
        equal opt_ints [| None; Some 2 |]
          (ints_at got "a");
        equal opt_ints [| None; Some 4 |]
          (ints_at got "b"));
    test "an expression over a field reads masked values" (fun () ->
        let q =
          Query.select
            Expr.[ within r ("v" := Col.int "a" + int 1) ]
            (Query.of_table t)
        in
        equal opt_ints [| None; Some 3 |]
          (ints_of q));
  ]

(* Spreading with within *)

let spread_t =
  v
    [
      ("id", ints [| Some 1; Some 2 |]);
      ("flux", Column.of_options Type.float64 [| Some 2.; None |]);
      ("err", Column.of_options Type.float64 [| Some 0.5; Some 1. |]);
    ]

let band =
  Expr.record
    Expr.[ "flux" := Col.float "flux"; "err" := Col.float "err" ]

let spreads =
  [
    test "keep narrows and spreads" (fun () ->
        let got =
          exec
            (Query.select
               Expr.[ within band (keep (Sel.names [ "err" ])) ]
               (Query.of_table spread_t))
        in
        equal schema_w
          (Schema.v [ ("err", Type.Any Type.float64) ])
          (Talon.schema got));
    test "each renames" (fun () ->
        let got =
          exec
            (Query.select
               Expr.
                 [
                   within band
                     (each Sel.all
                        { column = (fun n x -> ("band_" ^ n) := x) });
                 ]
               (Query.of_table spread_t))
        in
        equal (list string) [ "band_flux"; "band_err" ]
          (names got));
    test "across computes at a kind" (fun () ->
        let q =
          Query.select
            Expr.
              [
                within band
                  (across Kind.float
                     (Sel.names [ "flux" ])
                     (fun n x -> n := x *. float 2.));
              ]
            (Query.of_table spread_t)
        in
        equal (array (option float_exact)) [| Some 4.; None |]
          (Column.options Kind.float (Talon.column (exec q) "flux")));
    test "an aggregate reduces each field, as within's shape is the out's"
      (fun () ->
        let q =
          Query.aggregate ~by:[]
            Expr.
              [
                within band
                  (each Sel.all { column = (fun n x -> n := first x) });
              ]
            (Query.of_table spread_t)
        in
        equal (array (option float_exact)) [| Some 2. |]
          (Column.options Kind.float (Talon.column (exec q) "flux")));
    test "an over inside within reads fields as its keys" (fun () ->
        let q =
          Query.select
            Expr.[ within band ("v" := over ~by:[ "err" ] (count (Col.float "flux"))) ]
            (Query.of_table spread_t)
        in
        equal opt_ints [| Some 1; Some 0 |]
          (ints_of q));
  ]

(* L3, L4: nest and unnest *)

let l3 =
  prop ~count:200
    "nest then unnest is the rows in first-appearance group order"
    (Gen.bind G.sample (fun (G.Sample (_, rows) as s) ->
         Gen.map
           (fun ks -> (s, ks))
           (Gen.array ~size:(Gen.constant (Array.length rows))
              (Gen.int_range 0 3))))
    (fun (G.Sample (ty, rows), keys) ->
      let n = Array.length rows in
      let rank k =
        Option.get (Array.find_index (fun x -> x = k) keys)
      in
      let t =
        v
          [
            ("k", ints (Array.map Option.some keys));
            ("x", Column.of_options ty rows);
            ("rank", ints (Array.map (fun k -> Some (rank k)) keys));
            ("pos", ints (Array.init n (fun i -> Some i)));
          ]
      in
      let nested =
        Query.(
          of_table t
          |> aggregate ~by:[ "k" ]
               Expr.
                 [
                   each (Sel.names [ "x" ])
                     { column = (fun n c -> n := collect c) };
                 ]
          |> unnest Expr.[ keep (Sel.names [ "x" ]) ])
      in
      let want =
        Query.(
          of_table t
          |> sort [ Order.asc "rank"; Order.asc "pos" ]
          |> select Expr.[ keep (Sel.names [ "k"; "x" ]) ])
      in
      equal table_w (exec want) (exec nested))

let l4 =
  prop "unnest then nest is the rows with positions, for a unique id" drawn
    (fun d ->
      cover_laid d;
      let q = Query.of_table (laid d) in
      let back =
        q
        |> Query.unnest Expr.[ "x" := xs ]
        |> Query.aggregate ~by:[ "id" ] Expr.[ "xs" := collect x ]
      in
      let want =
        Query.filter Expr.(per_list [ "x" := xs ] rows > int 0) q
      in
      equal table_w (exec want) (exec back))

let non_inverse =
  [
    test "a repeated id merges its rows' lists" (fun () ->
        let t =
          v
            [
              ("id", ints [| Some 7; Some 7 |]);
              ("xs", of_ints [| Some [| Some 1 |]; Some [| Some 2 |] |]);
            ]
        in
        let back =
          Query.(
            of_table t
            |> unnest Expr.[ "x" := xs ]
            |> aggregate ~by:[ "id" ] Expr.[ "xs" := collect x ])
        in
        equal (array (option (array int)))
          [| Some [| 1; 2 |] |]
          (Column.options Kind.(list int) (Talon.column (exec back) "xs")));
    test "a row without positions vanishes, and a keep_empty row reads as null"
      (fun () ->
        let t =
          v
            [
              ("id", ints [| Some 1; Some 2; Some 3 |]);
              ("xs", of_ints [| Some [||]; None; Some [| Some 9 |] |]);
            ]
        in
        let ids q =
          ints_at (exec q) "id"
        in
        let q = Query.(of_table t) in
        equal opt_ints [| Some 3 |]
          (ids (Query.unnest Expr.[ "x" := xs ] q));
        let kept = Query.unnest ~keep_empty:true Expr.[ "x" := xs ] q in
        equal opt_ints [| Some 1; Some 2; Some 3 |] (ids kept);
        equal opt_ints [| None; None; Some 9 |]
          (ints_at (exec kept) "x"));
    test "a -0. key comes back as the group's first key, not its own"
      (fun () ->
        let t =
          v
            [
              ( "f",
                Column.of_options Type.float64
                  [| Some (-0.); Some 0.; Some Float.nan; Some Float.nan |] );
              ("x", ints [| Some 1; Some 2; Some 3; Some 4 |]);
            ]
        in
        let back =
          Query.(
            of_table t
            |> aggregate ~by:[ "f" ] Expr.[ "x" := collect x ]
            |> unnest Expr.[ keep (Sel.names [ "x" ]) ])
        in
        let keys = Column.values Kind.float (Talon.column (exec back) "f") in
        equal (array float_exact) [| -0.; -0.; Float.nan; Float.nan |] keys);
  ]

(* L5, L6, L7: positions, collect, reductions *)

let l5 =
  prop "unnest gives one row per position, keep_empty one or more per row"
    drawn (fun ((rows, _, _) as d) ->
      cover_laid d;
      let q = Query.of_table (laid d) in
      let count ?keep_empty () =
        Talon.rows (exec (Query.unnest ?keep_empty Expr.[ "x" := xs ] q))
      in
      let sum f = Array.fold_left (fun n r -> n + f (positions r)) 0 rows in
      equal int (sum Fun.id) (count ());
      equal int (sum (Int.max 1)) (count ~keep_empty:true ()))

let l6 =
  prop "a per_list of collect is the list, null and empty included"
    (Gen.triple (rows_of small) (Gen.option small) (Gen.int_range 0 2))
    (fun (rows, junk, lead) ->
      cover_lists rows;
      let c = column Type.int64 ?junk ~lead rows in
      let got =
        out
          (Query.select
             Expr.[ "v" := per_list [ "e" := xs ] (collect (Col.int "e")) ]
             (Query.of_table (v [ ("xs", c) ])))
      in
      equal table_w (v [ ("v", c) ]) (v [ ("v", got) ]))

let l7 =
  (* A row with no positions reduces no element: a null list has a null value,
     an empty one the value over none. *)
  let forms =
    Expr.
      [
        ("sum", (fun e -> sum e), `Sum);
        ("min", (fun e -> min e), `Min);
        ("count", (fun e -> count e), `Count);
      ]
  in
  prop "per_list agrees with unnest then a keyed reduction"
    (Gen.pair (Gen.int_range 0 2) (rows_of small))
    (fun (form, rows) ->
      cover_lists rows;
      let _, r, form = List.nth forms form in
      let t = table rows in
      let e = Col.int "e" in
      let per =
        out
          (Query.select
             Expr.[ "v" := per_list [ "e" := xs ] (r e) ]
             (Query.of_table t))
      in
      let grouped =
        exec
          Query.(
            of_table t
            |> unnest Expr.[ "e" := xs ]
            |> aggregate ~by:[ "id" ] Expr.[ "v" := r e ])
      in
      let by_id =
        Array.combine
          (Column.values Kind.int (Talon.column grouped "id"))
          (ints_at grouped "v")
      in
      let over_none = match form with `Sum -> Some 0 | `Count -> Some 0 | `Min -> None in
      let want i =
        match rows.(i) with
        | None -> None
        | Some [||] -> over_none
        | Some _ -> List.assoc i (Array.to_list by_id)
      in
      equal opt_ints
        (Array.init (Array.length rows) want)
        (Column.options Kind.int per))

(* Collect *)

let collecting =
  let t =
    v
      [
        ("g", ints [| Some 0; Some 1; Some 0 |]);
        ("x", ints [| Some 1; None; Some 3 |]);
      ]
  in
  [
    test "collect keeps nulls, one element per row" (fun () ->
        let got =
          exec
            Query.(
              of_table t
              |> aggregate ~by:[ "g" ]
                   Expr.[ "xs" := collect (Col.int "x") ]
              |> unnest ~keep_empty:false Expr.[ "e" := xs ])
        in
        equal opt_ints [| Some 1; Some 3; None |]
          (ints_at got "e"));
    test "collect over no rows is the empty list" (fun () ->
        let empty =
          Query.(of_table t |> slice ~offset:0 ~length:0)
          |> Query.aggregate ~by:[] Expr.[ "v" := collect (Col.int "x") ]
        in
        equal (array (option (array int))) [| Some [||] |]
          (Column.options Kind.(list int) (out empty)));
    test "collect under over holds the partition's list per row" (fun () ->
        let q =
          Query.(
            of_table t
            |> derive
                 Expr.
                   [
                     "v"
                     := per_list [ "e" := over ~by:[ "g" ] (collect (Col.int "x")) ] (count (Col.int "e"));
                   ])
        in
        equal opt_ints [| Some 2; Some 0; Some 2 |]
          (ints_of q));
    test "collect under window holds each window's list" (fun () ->
        let q =
          Query.(
            of_table t
            |> derive
                 Expr.[ "v" := per_list [ "e" := window (collect (Col.int "x")) ] rows ])
        in
        equal opt_ints [| Some 1; Some 2; Some 3 |]
          (ints_of q));
  ]

(* Compositions *)

let composed =
  let t =
    v
      [
        ("id", ints [| Some 0; Some 1 |]);
        ("l", of_ints [| Some [| Some 1; Some 2 |]; None |]);
      ]
  in
  let q = Query.of_table t in
  let l = Col.v Kind.(list int) "l" in
  let inner = Expr.(if_ (id > int 0) (record [ "a" := id ]) null) in
  [
    test "a null record rebuilt is a non-null record of nulls" (fun () ->
        let rebuilt = Expr.(record [ within inner (keep Sel.all) ]) in
        let got =
          exec
            (Query.select
               Expr.[ "n" := is_null rebuilt; "a" := field Kind.int "a" rebuilt ]
               q)
        in
        equal (array (option bool))
          [| Some false; Some false |]
          (Column.options Kind.bool (Talon.column got "n"));
        equal opt_ints [| None; Some 1 |] (ints_at got "a"));
    test "a field of a null inner record is null, and within nests" (fun () ->
        let outer = Expr.(record [ "b" := inner ]) in
        let got =
          exec
            (Query.select
               Expr.
                 [
                   "f" := field Kind.int "a" (field Record.kind "b" outer);
                   within outer
                     (within (Col.v Record.kind "b") (keep Sel.all));
                 ]
               q)
        in
        equal opt_ints [| None; Some 1 |] (ints_at got "f");
        equal opt_ints [| None; Some 1 |] (ints_at got "a"));
    test "a per_list stands in within, and within in a per_list" (fun () ->
        let r = Expr.(record [ "l" := l ]) in
        equal opt_ints [| Some 2; None |]
          (ints_of
             (Query.select
                Expr.[ within r ("v" := per_list [ "e" := l ] rows) ]
                q));
        let p = Col.v Record.kind "p" in
        let ps =
          Expr.(
            per_list [ "e" := l ] (collect (record [ "a" := Col.int "e" ])))
        in
        let back =
          Expr.(per_list [ "p" := ps ] (collect (record [ within p (keep Sel.all) ])))
        in
        let got = exec (Query.select Expr.[ "w" := ps; "v" := back ] q) in
        equal table_w
          (v [ ("v", Talon.column got "w") ])
          (v [ ("v", Talon.column got "v") ]));
    test "an output that reads other rows blocks over every batch" (fun () ->
        let got =
          exec
            (Query.unnest
               Expr.[ "e" := over (collect id) ]
               (Query.of_table (of_batches [ t; t ])))
        in
        let cycle = [| Some 0; Some 1; Some 0; Some 1 |] in
        equal opt_ints
          (Array.concat [ cycle; cycle; cycle; cycle ])
          (ints_at got "e");
        equal opt_ints
          (Array.concat
             (List.map
                (fun i -> Array.make 4 (Some i))
                [ 0; 1; 0; 1 ]))
          (ints_at got "id"));
    test "a filter moves through a select around a within's reads" (fun () ->
        let inner =
          Expr.(record [ "l" := Col.v Kind.(list int) "ls"; "z" := Col.int "b" ])
        in
        let total =
          Expr.(
            per_list
              [ "e" := Col.v Kind.(list int) "l" ]
              (sum (Col.int "e" + Col.int "z")))
        in
        let read =
          Expr.(field Kind.int "v" (record [ within inner ("v" := total) ]))
        in
        let renamed =
          v
            [
              ("a", ints [| Some 1; Some 2 |]);
              ("m", of_ints [| Some [| Some 1 |]; Some [| Some (-5) |] |]);
            ]
        in
        let got =
          exec
            Query.(
              of_table renamed
              |> select
                   Expr.
                     [
                       "b" := Col.int "a";
                       "ls" := Col.v Kind.(list int) "m";
                     ]
              |> filter Expr.(read >= int 0))
        in
        equal opt_ints [| Some 1 |] (ints_at got "b"));
    test "a body's order operation inside within reads an output, or is refused"
      (fun () ->
        let ps =
          Expr.(per_list [ "e" := l ] (collect (record [ "a" := Col.int "e" ])))
        in
        let p = Col.v Record.kind "p" in
        let shifted r =
          Expr.(
            per_list [ "p" := ps ]
              (sum
                 (field Kind.int "s"
                    (record [ within r ("s" := shift 1 (Col.int "a")) ]))))
        in
        equal opt_ints [| Some 1; None |]
          (ints_of (Query.select Expr.[ "v" := shifted p ] q));
        contains ~sub:"holds shift"
          (problem (fun () ->
               Query.select
                 Expr.[ "v" := shifted (record [ "a" := id ]) ]
                 q)));
  ]

(* The optimizer: a run equals each step run over the last's table, which no
   rewrite crosses. *)

let optimized =
  prop "optimizing keeps unnest's and within's rows" drawn (fun d ->
      cover_laid d;
      let r = Col.v Record.kind "r" and s = Col.v Record.kind "s" in
      let steps =
        Expr.
          [
            Query.derive [ "r" := record [ "id" := id + int 1; "xs" := xs ] ];
            Query.select [ "s" := r; "id" := id; "xs" := xs; "lc" := xs ];
            Query.filter (field Kind.int "id" s >= int 1);
            Query.unnest [ "e" := xs ];
            Query.filter (id >= int 1);
            Query.select [ within s (keep (Sel.names [ "id" ])); "e" := Col.int "e" ];
            Query.filter (Col.int "id" > int 1);
          ]
      in
      let t = laid d in
      let whole = List.fold_left (fun q f -> f q) (Query.of_table t) steps in
      let stepped =
        List.fold_left (fun t f -> exec (f (Query.of_table t))) t steps
      in
      equal table_w stepped (exec whole))

let sliced =
  prop "a slice's rows unnest as those rows do"
    (Gen.triple drawn (Gen.int_range 0 4) (Gen.int_range 0 6))
    (fun (((rows, _, _) as d), offset, length) ->
      cover_laid d;
      let unnest q =
        q
        |> Query.select Expr.[ keep (Sel.names [ "xs" ]) ]
        |> Query.unnest Expr.[ keep (Sel.names [ "xs" ]) ]
      in
      let n = Array.length rows in
      let first = Int.min offset n in
      let kept = Array.sub rows first (Int.min length (n - first)) in
      equal table_w
        (exec (unnest (Query.of_table (table kept))))
        (exec (unnest (Query.slice ~offset ~length (Query.of_table (laid d))))))

(* L8: batches *)

let l8 =
  prop "records, unnest and collect do not depend on batches"
    (Gen.bind drawn (fun d ->
         Gen.map (fun s -> (laid d, s)) (G.split (laid d))))
    (fun (whole, split) ->
      let same q =
        equal table_w (exec (q (Query.of_table whole)))
          (exec (q (Query.of_table split)))
      in
      same (Query.unnest Expr.[ "x" := xs ]);
      same (Query.unnest ~keep_empty:true Expr.[ "x" := xs ]);
      same (fun q ->
          Query.derive Expr.[ "r" := record [ "id" := id; "xs" := xs ] ] q
          |> Query.select
               Expr.[ within (Col.v Record.kind "r") (keep Sel.all) ]);
      same (fun q ->
          Query.aggregate ~by:[ "id" ]
            Expr.[ "c" := collect (record [ keep (Sel.names [ "id" ]) ]) ]
            q))

(* L9: zip by records *)

let zipped =
  let n_pair =
    Gen.frequency
      [
        (1, Gen.constant (None, None));
        (1, Gen.constant (None, Some [||]));
        (1, Gen.constant (Some [||], None));
        ( 4,
          Gen.bind (Gen.int_range 0 4) (fun n ->
              let a = Gen.array ~size:(Gen.constant n) (Gen.option small) in
              Gen.pair
                (Gen.map Option.some a)
                (Gen.map Option.some a)) );
      ]
  in
  Gen.array ~size:(Gen.int_range 0 10) n_pair

let l9 =
  prop "zipped lists unnest as a list of records spread" zipped (fun pairs ->
      cover "a null beside an empty list"
        (Array.exists (fun p -> p = (None, Some [||])) pairs);
      let t =
        v
          [
            ("xs", of_ints (Array.map fst pairs));
            ("ys", of_ints (Array.map snd pairs));
          ]
      in
      let by_zip =
        Query.(
          of_table t
          |> unnest Expr.[ "a" := xs; "b" := ys ]
          |> select Expr.[ keep (Sel.names [ "a"; "b" ]) ])
      in
      let zip =
        Expr.(
          per_list
            [ "a" := xs; "b" := ys ]
            (collect (record [ "a" := Col.int "a"; "b" := Col.int "b" ])))
      in
      let by_record =
        Query.(
          of_table t
          |> unnest Expr.[ "r" := zip ]
          |> select Expr.[ within (Col.v Record.kind "r") (keep Sel.all) ])
      in
      equal table_w (exec by_zip) (exec by_record))

(* Zip length errors *)

let zip_errors =
  let run a b =
    let t = v [ ("a", of_ints a); ("b", of_ints b) ] in
    Query.(of_table t |> unnest Expr.[ "x" := Col.v Kind.(list int) "a"; "y" := Col.v Kind.(list int) "b" ])
  in
  let message a b =
    Format.asprintf "%a" Error.pp (require_error (Query.run (run a b)))
  in
  [
    test "lists of different lengths are a data error naming the row"
      (fun () ->
        let m =
          message
            [| Some [| Some 1 |]; Some [| Some 1; Some 2 |] |]
            [| Some [| Some 1 |]; Some [| Some 1 |] |]
        in
        contains ~sub:"row 1" m;
        contains ~sub:"lengths differ" m;
        contains ~sub:{|"x" has 2|} m;
        contains ~sub:{|"y" has 1|} m);
    test "a null list beside a non-empty one is a data error naming the null"
      (fun () ->
        let m = message [| None |] [| Some [| Some 1 |] |] in
        contains ~sub:{|"x" is null|} m;
        contains ~sub:{|"y" has 1|} m);
    test "a null beside an empty list agrees, and gives no row" (fun () ->
        equal int 0
          (Talon.rows (exec (run [| None; Some [||] |] [| Some [||]; None |]))));
    test "a null over child rows beside an empty list gives no row" (fun () ->
        let a = column Type.int64 ~junk:9 [| None |] in
        let t = v [ ("a", a); ("b", of_ints [| Some [||] |]) ] in
        let l n = Col.v Kind.(list int) n in
        equal int 0
          (Talon.rows
             (exec
                (Query.unnest
                   Expr.[ "x" := l "a"; "y" := l "b" ]
                   (Query.of_table t)))));
    test "a keep_empty run stops at the failure" (fun () ->
        let q =
          Query.unnest ~keep_empty:true
            Expr.[ "x" := Col.v Kind.(list int) "a"; "y" := Col.v Kind.(list int) "b" ]
            (Query.of_table
               (v
                  [
                    ("a", of_ints [| None; None |]);
                    ("b", of_ints [| Some [||]; Some [| Some 1 |] |]);
                  ]))
        in
        let rows = ref 0 in
        ignore
          (require_error
             (Query.fold q ~init:() (fun () b ->
                  rows := !rows + Talon.rows b)));
        equal int 1 !rows);
    test "the rows before a failure stream out" (fun () ->
        let q =
          run
            [| Some [| Some 1; Some 2 |]; Some [| Some 1 |]; Some [||] |]
            [| Some [| Some 7; Some 8 |]; Some [| Some 9 |]; Some [| Some 0 |] |]
        in
        let seen = ref [] in
        let result =
          Query.fold q ~init:() (fun () b ->
              seen :=
                !seen
                @ Array.to_list
                    (ints_at b "y"))
        in
        ignore (require_error result);
        equal (list (option int)) [ Some 7; Some 8; Some 9 ] !seen);
  ]

(* keep_empty *)

let padding =
  let t =
    v
      [
        ("id", ints [| Some 1; Some 2; Some 3; Some 4 |]);
        ("xs", of_ints [| Some [| Some 1; None |]; Some [||]; None; Some [| Some 2 |] |]);
      ]
  in
  let kept = Query.(of_table t |> unnest ~keep_empty:true Expr.[ "x" := xs ]) in
  [
    test "a null or empty list keeps one row with a null element" (fun () ->
        let got = exec kept in
        equal opt_ints
          [| Some 1; Some 1; Some 2; Some 3; Some 4 |]
          (ints_at got "id");
        equal opt_ints
          [| Some 1; None; None; None; Some 2 |]
          (ints_at got "x"));
    test "rows counts a kept row as one" (fun () ->
        let got =
          exec (Query.aggregate ~by:[ "id" ] Expr.[ "n" := rows ] kept)
        in
        equal opt_ints [| Some 2; Some 1; Some 1; Some 1 |]
          (ints_at got "n"));
    test "zipped lists null at once keep their row under keep_empty" (fun () ->
        let z =
          v
            [
              ("a", of_ints [| None |]);
              ("b", of_ints [| Some [||] |]);
            ]
        in
        let got =
          exec
            (Query.unnest ~keep_empty:true
               Expr.[ "x" := Col.v Kind.(list int) "a"; "y" := Col.v Kind.(list int) "b" ]
               (Query.of_table z))
        in
        equal int 1 (Talon.rows got);
        equal opt_ints [| None |]
          (ints_at got "y"));
  ]

(* Outputs: placement and elements *)

let placement =
  let lists = Col.v Kind.(list int) "xs" in
  [
    test "an output replaces its name in place, the others follow" (fun () ->
        let t =
          v
            [
              ("a", ints [| Some 1 |]);
              ("xs", of_ints [| Some [| Some 7; Some 8 |] |]);
              ("b", ints [| Some 2 |]);
            ]
        in
        let got =
          exec
            (Query.unnest
               Expr.[ "xs" := lists; "y" := lists ]
               (Query.of_table t))
        in
        equal (list string) [ "a"; "xs"; "b"; "y" ]
          (names got);
        equal opt_ints [| Some 1; Some 1 |]
          (ints_at got "a");
        equal opt_ints [| Some 7; Some 8 |]
          (ints_at got "xs"));
    test "a list of lists unnests one level, nulls and all" (fun () ->
        let inner =
          of_ints [| Some [| Some 1 |]; None; Some [||] |]
        in
        let outer =
          laid_out (Type.list Type.int64) [| true |]
            (offsets_of ~lead:0 [| 3 |]) inner
        in
        let t = v [ ("ll", outer) ] in
        let got =
          exec
            (Query.unnest
               Expr.[ "l" := Col.v Kind.(list (list int)) "ll" ]
               (Query.of_table t))
        in
        equal (array (option (array int)))
          [| Some [| 1 |]; None; Some [||] |]
          (Column.options Kind.(list int) (Talon.column got "l")));
    test "a null element under a record column keeps the row" (fun () ->
        let t =
          v
            [
              ("id", ints [| Some 1; Some 2 |]);
              ("x", ints [| Some 5; None |]);
            ]
        in
        let got =
          exec
            Query.(
              of_table t
              |> aggregate ~by:[]
                   Expr.
                     [
                       "ps"
                       := collect (if_ (is_null x) null (record [ "x" := x ]));
                     ]
              |> unnest Expr.[ "p" := Col.v (Kind.list Record.kind) "ps" ]
              |> select
                   Expr.
                     [
                       within (Col.v Record.kind "p") (keep Sel.all);
                       "null" := is_null (Col.v Record.kind "p");
                     ])
        in
        equal opt_ints [| Some 5; None |]
          (ints_at got "x");
        equal (array (option bool)) [| Some false; Some true |]
          (Column.options Kind.bool (Talon.column got "null")));
  ]

(* Streaming *)

let streaming =
  [
    test "peak held rows stay one input batch's positions" (fun () ->
        let pulled = ref 0 in
        let n_batches = 50 and each = 40 in
        let batch =
          let xs = Array.make 2 (Some (Array.make each (Some 1))) in
          v [ ("xs", of_ints xs); ("id", ints [| Some 1; Some 2 |]) ]
        in
        let schema = Talon.schema batch in
        let reader () =
          let left = ref n_batches in
          let next () =
            if !left = 0 then Ok None
            else begin
              decr left;
              incr pulled;
              Ok (Some batch)
            end
          in
          Ok { Source.next; close = ignore }
        in
        let source =
          Source.v ~name:"batches" ~schema (fun _ ->
              Ok [ { Source.rows = None; open_ = reader } ])
        in
        let q =
          Query.unnest Expr.[ "x" := xs ] (Query.of_source source)
        in
        let folded = ref 0 and peak = ref 0 in
        require_ok ~pp:Error.pp
          (Query.fold q ~init:() (fun () b ->
               incr folded;
               at_most int ~than:(!folded + 1) !pulled;
               peak := Int.max !peak (Talon.rows b)));
        equal int n_batches !folded;
        at_most int ~than:(2 * each) !peak);
  ]

(* Names: every misuse is a problem naming the name *)

let misuse =
  let t =
    v
      [
        ("id", ints [| Some 1 |]);
        ("r", Column.of_options Type.int64 [| Some 1 |]);
      ]
  in
  let r = Expr.record Expr.[ "a" := id; "b" := id ] in
  let q = Query.of_table t in
  let rty = Type.record [ ("a", Any Type.int64) ] in
  let declared = Ext.v ~name:"r" ~ordered:false rty ~dec:Fun.id ~enc:Fun.id in
  let ext =
    Query.of_table
      (v [ ("c", Column.of_options (Type.ext ~name:"r" rty) [| None |]) ])
  in
  let misuses =
    [
      ( "a field its record lacks", "no field \"nope\"",
        fun () -> Query.select Expr.[ "v" := field Kind.int "nope" r ] q );
      ( "a near field name", "Did you mean",
        fun () -> Query.select Expr.[ "v" := field Kind.int "aa" r ] q );
      ( "a field its kind does not read", "does not read",
        fun () -> Query.select Expr.[ "v" := field Kind.string "a" r ] q );
      ( "a field of a null with no type", "has no type",
        fun () -> Query.select Expr.[ "v" := field Kind.int "a" null ] q );
      ( "within names the record's fields, not the columns", "no field \"id\"",
        fun () -> Query.select Expr.[ within r ("v" := id) ] q );
      ( "within's keep of a field the record lacks", "no field \"nope\"",
        fun () -> Query.select Expr.[ within r (keep (Sel.names [ "nope" ])) ] q );
      ( "within's each of a field the record lacks", "no field \"nope\"",
        fun () -> Query.select Expr.[ within r (each (Sel.names [ "nope" ]) { column = ( := ) }) ] q );
      ( "across a field of another kind", "does not read",
        fun () -> Query.select Expr.[ within r (across Kind.string Sel.all ( := )) ] q );
      ( "within a non-record column", "Col.v record reads",
        fun () -> Query.select Expr.[ within (Col.v Record.kind "id") (keep Sel.all) ] q );
      ( "a record output named twice", "\"a\" is named twice",
        fun () -> Query.select Expr.[ "v" := record [ "a" := id; "a" := id ] ] q );
      ( "a record field without a column type", "give it one with store",
        fun () -> Query.select Expr.[ "v" := record [ "a" := const 1 ] ] q );
      ( "a derive's spread over a column of the input", "of within has the name of a column",
        fun () -> Query.derive Expr.[ within r (keep (Sel.names [ "a" ])) ] (Query.select Expr.[ "a" := id; "id" := id; "r" := id ] q) );
      ( "field through a declaration of a record", "field reads a record column",
        fun () -> Query.select Expr.[ "v" := field Kind.int "a" (Ext.col declared "c") ] ext );
      ( "within through a declaration of a record", "within spreads a record column",
        fun () -> Query.select Expr.[ within (Ext.col declared "c") (keep Sel.all) ] ext );
      ( "unnest of a column that is no list", "which is not a list",
        fun () -> Query.unnest Expr.[ "v" := id ] q );
      ( "unnest of no output", "at least one output",
        fun () -> Query.unnest [] q );
      ( "unnest of one name twice", "appears twice",
        fun () -> Query.unnest Expr.[ "v" := xs; "v" := xs ] (Query.of_table (table [||])) );
      ( "unnest of a spread over a column", "of within has the name of a column",
        fun () -> Query.unnest Expr.[ within (record [ "xs" := xs; "id" := xs ]) (keep Sel.all) ] (Query.of_table (table [| Some [| Some 1 |] |])) );
      ( "unnest of an expression over a missing column", "no column \"nope\"",
        fun () -> Query.unnest Expr.[ "v" := Col.v Kind.(list int) "nope" ] q );
    ]
  in
  cases "a misuse names its name"
    ~id:(fun (title, _, _) -> title)
    misuses
    (fun (_, sub, f) -> contains ~sub (problem (fun () -> ignore (f ()))))

let collect_typed =
  [
    test "collect of an extension read through each has a type" (fun () ->
        let ext = Type.ext ~name:"money" Type.int64 in
        let t = v [ ("m", Column.of_options ext [| None |]) ] in
        let got =
          exec
            (Query.aggregate ~by:[]
               Expr.[ each Sel.all { column = (fun n x -> n := collect x) } ]
               (Query.of_table t))
        in
        equal schema_w
          (Schema.v [ ("m", Type.Any (Type.list ext)) ])
          (Talon.schema got));
    test "collect of an extension with its declaration is a problem"
      (fun () ->
        let d = Ext.v ~name:"money" ~ordered:false Type.int64 ~dec:Fun.id ~enc:Fun.id in
        let ty = Type.ext ~name:"money" Type.int64 in
        let t = v [ ("m", Column.of_options ty [| None |]) ] in
        contains ~sub:"collect has no type"
          (problem (fun () ->
               ignore
                 (Query.aggregate ~by:[]
                    Expr.[ "v" := collect (Ext.col d "m") ]
                    (Query.of_table t)))));
  ]

let () =
  exit
    (run "Nesting"
       [
         group "Records" ([ l1; l2 ] @ null_record @ spreads @ composed);
         group "Nest and unnest"
           ([ l3; l4; l8; l9; optimized; sliced ] @ non_inverse);
         group "Positions" ([ l5; l6; l7 ] @ collecting @ placement);
         group "Zip and padding" (zip_errors @ padding);
         group "Streaming" streaming;
         group "Types" collect_typed;
         group "Misuse" [ misuse ];
       ])
