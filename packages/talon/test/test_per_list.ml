(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

(* Lists

   A list is [None] when null, its elements [None] when null. [column] lays
   rows out by hand, to reach what [Column.of_options] does not build: null
   elements, [lead] values before the first row, and a null row over a
   non-empty segment of [junk]. *)

type 'a lists = 'a option array option array

let laid_out ty validity offsets child =
  let validity =
    if Array.for_all Fun.id validity then None
    else Some (Nx.create Nx.bit [| Array.length validity |] validity)
  in
  let offsets = Nx.create Nx.int64 [| Array.length offsets |] offsets in
  Result.get_ok
    (Column.of_layout (Any (Type.list ty))
       (Varsize { validity; offsets; child }))

let offsets_of ~lead lengths =
  let o = Array.make (Array.length lengths + 1) (Int64.of_int lead) in
  let add i l = o.(i + 1) <- Int64.add o.(i) (Int64.of_int l) in
  Array.iteri add lengths;
  o

let column ty ?(lead = 0) ?junk (rows : 'a lists) =
  let segment = function
    | Some xs -> xs
    | None -> ( match junk with Some v -> [| Some v; Some v |] | None -> [||])
  in
  let segments = Array.map segment rows in
  let pad = match junk with Some v -> Array.make lead (Some v) | None -> [||] in
  let lengths = Array.map Array.length segments in
  let child =
    Column.of_options ty (Array.concat (pad :: Array.to_list segments))
  in
  laid_out ty
    (Array.map Option.is_some rows)
    (offsets_of ~lead:(Array.length pad) lengths)
    child

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
        Gen.bind (Gen.int_range 2 5) (fun n ->
            Gen.map Option.some (elements n)) );
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
            Body ("sum of x * w", sum (x * w));
            Body ("cadence", sum (x - shift 1 x));
            Body ("rank", sum (rank x));
            Body ("over", sum (x - over (max x)));
            Body ("window", max (window (sum x)));
          ]
  | _, Some Stdlib.Type.Equal ->
      common
      @ Expr.
          [
            Body ("sum", sum x);
            Body ("mean", mean x);
            Body ("median", median x);
            Body ("sum of x * w", sum (x *. cast Type.float64 w));
          ]
  | _ -> common

(* [reference ty e w elements] is [e] over the frame of one row's list: its
   [elements] under the element's name, the row's [w] repeated. *)
let reference ty e w elements =
  let n = Array.length elements in
  let t =
    v
      [
        ("w", ints (Array.make n w)); ("x", Column.of_options ty elements);
      ]
  in
  out (Query.aggregate ~by:[] Expr.[ "v" := e ] (Query.of_table t))

(* [against ty rows w e got] states that [got] holds, on each row, [e] over its
   list, and null on a null list. *)
let against ty (rows : _ lists) w e got =
  let (Type.Any tv) = Column.type_ got in
  let k = Type.kind tv in
  let vs = Column.options k got in
  Array.iteri
    (fun i r ->
      let want =
        match r with
        | None -> None
        | Some xs -> (Column.options k (reference ty e w.(i) xs)).(0)
      in
      equal (option (G.witness tv)) want vs.(i))
    rows

type case =
  | Case : {
      ty : 'a Type.t;
      rows : 'a lists;
      w : int option array;
      lead : int;
      junk : 'a option;
      body : int;
    }
      -> case

let case_of (type a) (ty : a Type.t) (value : a Gen.t) =
  Gen.(
    let* rows = some_rows (list_of value) in
    let* w = array ~size:(constant (Array.length rows)) (option small) in
    let* junk = option value in
    let* lead = int_range 0 2 in
    let+ body = int_range 0 1000 in
    Case { ty; rows; w; lead; junk; body })

let drawn =
  Gen.one_of
    [
      case_of Type.int64 int64s;
      case_of Type.float64 (Option.get (G.value Type.float64));
      case_of Type.string (Option.get (G.value Type.string));
      case_of Type.date (Option.get (G.value Type.date));
    ]

let table (Case c) =
  v
    [
      ("xs", column c.ty ~lead:c.lead ?junk:c.junk c.rows);
      ("w", ints c.w);
    ]

let handle ty = Col.v (Kind.list (Type.kind ty)) "xs"

(* [body_of ty i] is the [i]th body of [ty], over the element [x]. *)
let body_of ty i =
  let bs = bodies ty (Col.v (Type.kind ty) "x") in
  List.nth bs (i mod List.length bs)

let definition =
  prop ~count:300 "per_list is its body over each row's list" drawn
    (fun (Case c as case) ->
      cover_lists c.rows;
      cover "a null slot over a non-empty segment"
        (Option.is_some c.junk && Array.exists Option.is_none c.rows);
      cover "offsets that do not start at 0"
        (Option.is_some c.junk && c.lead > 0);
      let (Body (_, e)) = body_of c.ty c.body in
      let q =
        Query.derive
          Expr.[ "v" := per_list [ "x" := handle c.ty ] e ]
          (Query.of_table (table case))
      in
      against c.ty c.rows c.w e (out q))

let batching =
  prop "per_list does not depend on batches"
    (Gen.bind drawn (fun case ->
         Gen.map (fun t -> (case, t)) (G.split (table case))))
    (fun ((Case c as case), split) ->
      let (Body (_, e)) = body_of c.ty c.body in
      let q t =
        Query.derive
          Expr.[ "v" := per_list [ "x" := handle c.ty ] e ]
          (Query.of_table t)
      in
      cover "several batches" (List.length (batches split) > 1);
      same (out (q (table case))) (out (q split)))

(* Lists of lists *)

let nested (rows : int option array option array option array) =
  let inner =
    Array.concat
      (Array.to_list
         (Array.map (function Some l -> l | None -> [||]) rows))
  in
  let lengths =
    Array.map (function Some l -> Array.length l | None -> 0) rows
  in
  laid_out (Type.list Type.int64)
    (Array.map Option.is_some rows)
    (offsets_of ~lead:0 lengths)
    (column Type.int64 inner)

let nesting =
  let ll = Col.v Kind.(list (list int)) "ll"
  and l = Col.v Kind.(list int) "l"
  and x = Col.int "x" in
  let inner r = Expr.(per_list [ "x" := l ] (r x)) in
  let total xs = Some (List.fold_left ( + ) 0 xs) in
  let extreme f = function
    | [] -> None
    | x :: xs -> Some (List.fold_left f x xs)
  in
  let forms =
    Expr.
      [
        ("sum", per_list [ "l" := ll ] (sum (inner sum)), total);
        ("min", per_list [ "l" := ll ] (min (inner min)), extreme Stdlib.min);
        ("max", per_list [ "l" := ll ] (max (inner max)), extreme Stdlib.max);
        ( "sum of counts",
          per_list [ "l" := ll ]
            (sum (per_list [ "x" := l ] (count x))),
          fun xs -> Some (List.length xs) );
      ]
  in
  prop "a frame of inner lists reduces their flattened values"
    (Gen.pair (Gen.int_range 0 3) (some_rows (lists_of (list_of small))))
    (fun (form, rows) ->
      let flat l =
        List.concat_map
          (function
            | Some xs -> List.filter_map Fun.id (Array.to_list xs)
            | None -> [])
          (Array.to_list l)
      in
      let inners =
        Array.concat
          (Array.to_list
             (Array.map (function Some l -> l | None -> [||]) rows))
      in
      cover "an empty inner list"
        (Array.exists (fun r -> r = Some [||]) inners);
      cover "a null inner list" (Array.exists Option.is_none inners);
      let _, e, want = List.nth forms form in
      let t = v [ ("ll", nested rows) ] in
      let got =
        Column.options Kind.int
          (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))
      in
      equal
        (array (option int))
        (Array.map (Option.fold ~none:None ~some:(fun l -> want (flat l))) rows)
        got)

let correlation =
  prop "an outer factor comes out of a sum"
    (some_rows (Gen.pair (list_of int64s) small))
    (fun rows ->
      let xs = Col.v Kind.(list int) "xs"
      and x = Col.int "x"
      and w = Col.int "w" in
      let t =
        v
          [
            ("xs", column Type.int64 (Array.map fst rows));
            ("w", ints (Array.map (fun (_, w) -> Some w) rows));
          ]
      in
      let eval e =
        out (Query.derive Expr.[ "v" := e ] (Query.of_table t))
      in
      same
        (eval Expr.(w * per_list [ "x" := xs ] (sum x)))
        (eval Expr.(per_list [ "x" := xs ] (sum (x * w)))))

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

let lockstep_columns (rows, _, _) =
  let a = Array.map (fun (r, _) -> Option.map fst r) rows
  and b =
    Array.map
      (fun (r, null_b) ->
        match r with
        | Some (_, b) when not null_b -> Some b
        | _ -> None)
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

let as_ = Col.v Kind.(list int) "as_" and bs = Col.v Kind.(list int) "bs"
let a = Col.int "a" and b = Col.int "b"
let lockstep_body = Expr.(sum (a * b))

let lockstep =
  prop "lists of one row are read in lockstep or fail at the row" pairs
    (fun draw ->
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
        (Array.exists2
           (fun x y -> Option.is_none x <> Option.is_none y)
           xs ys);
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
                    let t =
                      v
                        [
                          ("a", ints x);
                          ("b", ints y);
                        ]
                    in
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

   The element's name is the input's column [t] under a select that renames
   it, so that a pushed predicate could read the one for the other. *)

let renaming =
  let input rows =
    v
      [
        ("t", ints (Array.map fst rows));
        ("ts", column Type.int64 (Array.map snd rows));
      ]
  in
  let ts = Col.v Kind.(list int) "ts" and t = Col.int "t" in
  let y = Col.int "y" in
  let selected rows =
    Query.select Expr.[ "y" := t; "ts" := ts ] (Query.of_table (input rows))
  in
  let rows =
    some_rows (Gen.pair (Gen.option small) (list_of small))
  in
  let keeps name p want =
    prop name rows (fun rows ->
        let filtered = Query.filter (p ()) (selected rows) in
        let kept =
          Array.of_list
            (List.filter_map
               (fun (y, l) -> if want y l then Some y else None)
               (Array.to_list rows))
        in
        cover_lists (Array.map snd rows);
        equal
          (array (option int))
          kept
          (Column.options Kind.int (Talon.column (exec filtered) "y")))
  in
  let values l = List.filter_map Fun.id (Array.to_list l) in
  let sum_by f l = List.fold_left (fun s e -> s + f e) 0 (values l) in
  [
    keeps "a predicate that reads the renamed column stays above the select"
      (fun () -> Expr.(per_list [ "t" := ts ] (sum (t * y)) > int 0))
      (fun y l ->
        match (y, l) with
        | Some y, Some l -> sum_by (fun e -> e * y) l > 0
        | _ -> false);
    keeps "a body that reads only its element moves below the select"
      (fun () -> Expr.(per_list [ "t" := ts ] (sum t) > int 0))
      (fun _ l -> match l with Some l -> sum_by Fun.id l > 0 | None -> false);
    keeps "an over in a body orders by the output of its name"
      (fun () ->
        Expr.(
          per_list [ "t" := ts ]
            (sum (over ~order:[ Order.asc "t" ] (shift 1 t) - t))
          < int 0))
      (fun _ l ->
        (* In ascending order, each value's predecessor less the value sums
           to the least value less the greatest; nulls sort last. *)
        match Option.map values l with
        | None | Some [] -> false
        | Some (v :: vs) ->
            List.fold_left Stdlib.min v vs < List.fold_left Stdlib.max v vs);
  ]

let lockstep_filter =
  prop "a filter that drops unequal lengths guards the lockstep above it" pairs
    (fun ((rows, _, _) as draw) ->
      let xs, ys, cols = lockstep_table draw in
      let ids = Array.mapi (fun i _ -> Some i) rows in
      let t = v (("id", ints ids) :: cols) in
      let len name l =
        Expr.(per_list [ name := l ] rows)
      in
      let q =
        Query.of_table t
        |> Query.filter Expr.(len "u" as_ = len "v" bs)
        |> Query.filter
             Expr.(
               per_list [ "a" := as_; "b" := bs ] lockstep_body > int 0)
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
      equal (list (option int)) want
        (Array.to_list
           (Column.options Kind.int (Talon.column (exec q) "id"))))

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
    test "shift, over and window stay within a list" (fun () ->
        (* Rows 0 and 3 have lists: row 3's first element sees no previous. *)
        equal null_ints
          [| Some 1; None; Some 0; Some 10 |]
          (eval Expr.(per_list [ "x" := xs ] (sum (shift 1 x))));
        equal null_ints
          [| Some (-1); None; Some 0; Some (-10) |]
          (eval Expr.(per_list [ "x" := xs ] (sum (x - over (max x)))));
        equal null_ints
          [| Some 3; None; None; Some 30 |]
          (eval Expr.(per_list [ "x" := xs ] (max (window (sum x))))));
    test "a computed list" (fun () ->
        let t =
          v
            [
              ( "s",
                Column.of_options Type.string
                  [| Some "a_b"; None; Some "" |] );
            ]
        in
        let s = Col.string "s" in
        let e = Expr.(per_list [ "p" := Str.split (Str.literal "_") s ] rows) in
        equal null_ints
          [| Some 2; None; Some 1 |]
          (Column.options Kind.int
             (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))));
    test "an output that reads its own column hides only that list" (fun () ->
        let t =
          v
            [
              ("ls", of_ints [| Some [| 1; 2 |] |]);
              ("s", Column.v Type.string [| "a,b" |]);
            ]
        in
        let derive e = Query.derive Expr.[ "v" := e ] (Query.of_table t) in
        let e =
          Expr.(
            per_list
              [
                each
                  Sel.(names [ "ls" ])
                  { column = (fun n c -> (n ^ "_e") := c) };
              ]
              (sum (Col.int "ls_e")))
        in
        equal null_ints [| Some 3 |]
          (Column.options Kind.int (out (derive e)));
        let ls = Col.int "ls" in
        equal null_ints [| Some 3 |]
          (Column.options Kind.int
             (out
                (derive Expr.(per_list [ keep Sel.(names [ "ls" ]) ] (sum ls)))));
        equal null_ints [| Some 2 |]
          (Column.options Kind.int
             (out
                (derive
                   Expr.(
                     per_list
                       [
                         each
                           Sel.(names [ "ls" ])
                           { column = (fun n c -> n := c) };
                       ]
                       (count ls)))));
        raises_match (Exn.invalid_arg ~sub:"reads list")
          (fun () ->
            derive
              Expr.(
                per_list
                  [ keep Sel.(names [ "ls" ]) ]
                  (count (Col.v Kind.(list int) "ls"))));
        raises_match (Exn.invalid_arg ~sub:"would hide")
          (fun () ->
            derive
              Expr.(per_list [ "ls" := Str.split (Str.literal ",") (Col.string "s") ] rows)));
    test "a failure in a body is at its list's row" (fun () ->
        let xs_c = of_ints [| Some [| 1 |]; Some [||]; Some [| 300 |] |] in
        let t = v [ ("xs", xs_c) ] in
        let e = Expr.(per_list [ "x" := xs ] (sum (cast Type.int8 x))) in
        let r = Query.run (Query.derive Expr.[ "v" := e ] (Query.of_table t)) in
        contains ~sub:": row 2: "
          (Format.asprintf "%a" Error.pp (require_error r)));
    test "a reduction that fails fails at its list's row" (fun () ->
        let xs_c = of_ints [| Some [| 1; 1 |]; Some [||]; Some [| 1; 2 |] |] in
        let t = v [ ("xs", xs_c) ] in
        let e = Expr.(per_list [ "x" := xs ] (only x)) in
        let r = Query.run (Query.derive Expr.[ "v" := e ] (Query.of_table t)) in
        contains ~sub:": row 2: "
          (Format.asprintf "%a" Error.pp (require_error r)));
    test "a row without a list runs no body" (fun () ->
        let f (_ : int) : int = failwith "called" in
        let e =
          Expr.(
            per_list [ "x" := xs ]
              (store Type.int64 (const f $ count x)))
        in
        let t = v [ ("xs", of_ints [| None |]) ] in
        equal null_ints [| None |]
          (Column.options Kind.int
             (out (Query.derive Expr.[ "v" := e ] (Query.of_table t)))));
    cases "problems" ~id:(fun (n, _, _) -> n)
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
                    (per_list [ "l" := Str.split (Str.literal ",") (Col.string "s") ] rows))),
            "would hide" );
          ( "an output that is not a list",
            problem (per_list [ "x" := Col.int "w" ] (count x)),
            "is not a list" );
          ( "no output",
            problem (per_list [] rows),
            "at least one output" );
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
          |> Query.filter p
          |> exec
          |> fun t -> Column.options Kind.int (Talon.column t "id")
        in
        equal null_ints [| Some 0 |]
          (keeps Expr.(per_list [ "x" := xs ] (only x) > int 0));
        let ds = Col.v Kind.(list span) "ds" and d = Col.span "d" in
        equal null_ints [| Some 0 |]
          (keeps
             Expr.(
               per_list [ "d" := ds ] (sum d)
               > span (Time.Span.of_ns 0L))));
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
    test "it prints as written and compares by output names" (fun () ->
        let e = Expr.(per_list [ "x" := xs ] (sum (x * x))) in
        equal string {|per_list ["x" := xs] (sum (x * x))|}
          (Format.asprintf "%a" Expr.pp e);
        let t = v [ ("xs", of_ints [| Some [| 1 |] |]) ] in
        let q n r =
          Query.derive Expr.[ "v" := per_list [ n := xs ] r ] (Query.of_table t)
        in
        equal bool true
          (Query.equal (q "a" Expr.rows) (q "a" Expr.rows));
        equal bool false
          (Query.equal (q "a" Expr.rows) (q "b" Expr.rows)));
    (let failing =
       v
         [
           ("xs", of_ints [| Some [| 1 |]; Some [||]; Some [| 1; 2 |] |]);
           ("w", ints [| Some 0; Some 0; Some 0 |]);
         ]
     in
     cases "a failure in a frame operation is at its list's row"
       ~id:(fun (n, _) -> n)
       Expr.
         [
           ("over", per_list [ "x" := xs ] (sum (over (only x))));
           ("over ~order", per_list [ "x" := xs ]
              (sum (over ~order:[ Order.asc "x" ] (only x))));
           ( "over ~by",
             per_list [ "x" := xs ] (sum (over ~by:[ "w" ] (only x))) );
           ("window", per_list [ "x" := xs ] (sum (window (only x))));
         ]
       (fun (_, e) ->
         let q = Query.derive Expr.[ "v" := e ] (Query.of_table failing) in
         contains ~sub:": row 2: "
           (Format.asprintf "%a" Error.pp (require_error (Query.run q)))));
    test "a body of a literal is the literal on each row with a list"
      (fun () ->
        let frame rows =
          let t = v [ ("xs", of_ints rows) ] in
          Column.options Kind.int
            (out
               (Query.derive
                  Expr.[ "v" := per_list [ "x" := xs ] (int 3) ]
                  (Query.of_table t)))
        in
        equal null_ints
          [| Some 3; Some 3; None |]
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
        equal null_ints [| Some 13; Some 0 |]
          (Column.options Kind.int (out q)));
    test "a per_list stands in a window's operand" (fun () ->
        let w = Expr.(window (sum (per_list [ "x" := xs ] (count x)))) in
        equal null_ints
          [| Some 2; Some 2; Some 2; Some 4 |]
          (eval w));
    test "an order operation in an output is the window's" (fun () ->
        contains ~sub:"holds shift"
          (problem
             Expr.(window (sum (per_list [ "x" := shift 1 xs ] (count x))))));
    test "a body's over reaches by its output's name" (fun () ->
        equal null_ints
          [| Some 3; None; Some 0; Some 30 |]
          (eval
             Expr.(
               per_list [ "x" := xs ]
                 (sum (over ~by:[ "x" ] (x))))));
  ]

let () =
  exit
    (run "Per_list"
       [
         group "Laws"
           ([
              definition; batching; nesting; correlation; lockstep;
              lockstep_filter;
            ]
           @ renaming);
         group "Rules" rules;
       ])
