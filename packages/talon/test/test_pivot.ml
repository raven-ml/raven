(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module G = Talon_gen

(* Tables compare by key identity, so that -0. equals 0. and NaN equals NaN, as
   a key's value does. *)
let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal
let schema_w = Testable.make ~pp:Schema.pp ~equal:Schema.equal
let run_ok q = require_ok ~pp:Error.pp (Query.run q)
let of_ints = Column.of_options Type.int64
let of_floats = Column.of_options Type.float64
let of_keys d = Column.of_options (Type.categorical d)
let k = Col.string "k"
let x = Col.float "x"

(* [failure q] is why [q]'s run fails. *)
let failure q =
  match Query.run q with
  | Ok _ -> failwith "the run succeeded"
  | Error e -> Format.asprintf "%a" Error.pp e

let problem f =
  match f () with
  | _ -> failwith "no exception"
  | exception Invalid_argument m -> m

(* Longer, then wider

   [t] holds the id columns, unique over its rows, then the selected columns of
   one type, so that wider restores [t]. *)

type wide =
  | Wide : {
      ty : 'a Type.t;
      ids : string list;
      columns : (string * 'a option array) list;
      table : Talon.t;
    }
      -> wide

let pp_wide ppf (Wide { ty; columns; table; _ }) =
  Format.fprintf ppf "%d rows of %d columns %a@.%a" (Talon.rows table)
    (List.length columns) Type.pp ty Talon.pp table

(* A selected column needs a handle, and no handle reads an extension. *)
let has_handle (Type.Any ty) =
  let k = Type.kind ty in
  Option.is_some (Kind.provably_equal k k)

let wide =
  Gen.(
    with_pp pp_wide
      (let* (Type.Any ty) = such_that has_handle G.type_ in
       let* k = frequency [ (1, constant 1); (3, int_range 2 5) ] in
       let* drawn = list ~size:(constant k) (G.options ty) in
       let* blank = frequency [ (1, constant true); (3, constant false) ] in
       let n = Array.length (List.hd drawn) in
       let fit i xs =
         if blank && i = 0 then Array.make n None
         else
           let m = Array.length xs in
           Array.init n (fun i -> if m = 0 then None else xs.(i mod m))
       in
       let columns =
         List.mapi (fun i xs -> (Printf.sprintf "c%d" i, fit i xs)) drawn
       in
       let+ ids = permutation (List.init n Fun.id) in
       let table =
         Talon.v
           (("id", of_ints (Array.of_list (List.map Option.some ids)))
           :: ("g", of_ints (Array.of_list (List.map (fun i -> Some (i mod 3)) ids)))
           :: List.map (fun (c, xs) -> (c, Column.of_options ty xs)) columns)
       in
       Wide { ty; ids = [ "id"; "g" ]; columns; table }))

let l1 (Wide { ids; columns; table; ty = _ }) =
  let names = List.map fst columns in
  let n = Talon.rows table in
  cover "zero rows" (n = 0);
  cover "one column" (List.length names = 1);
  cover "several rows and columns" (n >= 2 && List.length names >= 2);
  cover "a column of nulls"
    (List.exists (fun (_, xs) -> Array.for_all Option.is_none xs) columns);
  let longer t =
    run_ok (Kit.longer ~name:"name" ~value:"value" (Sel.names names) (Query.of_table t))
  and wider l =
    run_ok (Kit.wider ~by:ids ~key:"name" ~value:"value" (Query.of_table l))
  in
  Law.round_trip table_w table_w longer wider table

(* Wider, then longer

   [L]'s columns are [by], then the key, then the value, in that order. The key
   is never null, and [(by, key)] is unique by key identity. Its float key
   holds both zeros and a NaN, one key each, so that a group's rows can differ
   in their bits. *)

type long =
  | Long : {
      ty : 'a Type.t;
      kind : 'a Kind.t;
      by : string list;
      dictionary : string array;
      table : Talon.t;
    }
      -> long

let pp_long ppf (Long { by; dictionary; table; ty; _ }) =
  Format.fprintf ppf "by [%s], %d categories, %a:@.%a" (String.concat ";" by)
    (Array.length dictionary) Type.pp ty Talon.pp table

let float_keys = [| Some 0.; Some (-0.); Some Float.nan; None |]

(* A float key's identity: [-0.] is [0.], and every NaN one key. *)
let float_key i = if i = 1 then 0 else i

let distinct key rows =
  let seen = Hashtbl.create 16 in
  List.filter
    (fun row ->
      let k = key row in
      if Hashtbl.mem seen k then false
      else begin
        Hashtbl.add seen k ();
        true
      end)
    rows

type value = Value : 'a Type.t -> value

let long =
  Gen.(
    with_pp pp_long
      (let* size = int_range 1 6 in
       let dictionary = Array.init size (Printf.sprintf "k%d") in
       let* by =
         of_list [ []; []; [ "g" ]; [ "f" ]; [ "g"; "f" ]; [ "f"; "g" ] ]
       in
       let* rows =
         list
           ~size:(frequency [ (1, constant 0); (2, int_range 1 25) ])
           (triple (int_range 0 2) (int_range 0 3) (int_range 0 (size - 1)))
       in
       (* Keep [(by, key)] unique by key identity. *)
       let identity (g, f, c) =
         ( (if List.mem "g" by then Some g else None),
           (if List.mem "f" by then Some (float_key f) else None),
           c )
       in
       let rows = Array.of_list (distinct identity rows) in
       let* (Value ty) =
         of_list [ Value Type.float64; Value Type.int64; Value Type.string ]
       in
       let+ drawn = G.options ty in
       let n = Array.length rows in
       let values =
         Array.init n (fun i ->
             let m = Array.length drawn in
             if m = 0 then None else drawn.(i mod m))
       in
       let keys =
         Array.map (fun (_, _, c) -> Some dictionary.(c)) rows
       in
       let by_columns =
         List.map
           (function
             | "g" ->
                 ("g", of_ints (Array.map (fun (g, _, _) -> Some g) rows))
             | _ ->
                 ( "f",
                   of_floats
                     (Array.map (fun (_, f, _) -> float_keys.(f)) rows) ))
           by
       in
       let table =
         Talon.v
           (by_columns
           @ [
               ("key", of_keys dictionary keys);
               ("value", Column.of_options ty values);
             ])
       in
       Long { ty; kind = Type.kind ty; by; dictionary; table }))

let wider ~by ?(key = "key") ?(value = "value") t =
  run_ok (Kit.wider ~by ~key ~value (Query.of_table t))

(* Rows in the order of their cells: the key, then [by]. A group's rows differ
   in their keys, so that one value of a float key can stand for another. *)
let in_cell_order by t =
  run_ok
    (Query.sort
       (Order.asc "key" :: List.map Order.asc by)
       (Query.of_table t))

let longer ~by ~dictionary kind t =
  let non_null = Expr.(not (is_null (Col.v kind "value"))) in
  run_ok
    (Kit.longer ~name:"key" ~value:"value"
       (Sel.names (Array.to_list dictionary))
       (Query.of_table t)
    |> Query.filter non_null
    |> Query.select Expr.[ keep (Sel.names (by @ [ "key"; "value" ])) ])

let l2 (Long { kind; by; dictionary; table; _ }) =
  let n = Talon.rows table in
  let held = Query.filter Expr.(not (is_null (Col.v kind "value"))) (Query.of_table table) in
  let long_w =
    Testable.make ~pp:Talon.pp ~equal:(fun a b ->
        Talon.equal (in_cell_order by a) (in_cell_order by b))
  in
  cover "no row" (n = 0);
  cover "no key" (by = []);
  cover "no key, no row" (by = [] && n = 0);
  cover "a null value" (Talon.rows (run_ok held) < n);
  cover "several groups" (n >= 2 && by <> []);
  cover "a float key" (List.mem "f" by && n >= 1);
  Law.round_trip long_w table_w (wider ~by)
    (longer ~by ~dictionary kind)
    (run_ok held);
  equal long_w
    ~msg:"nulls of the input drop"
    (run_ok held)
    (longer ~by ~dictionary kind (wider ~by table))

(* A reference on small inputs

   A row of [(g, k, x)] per input row; [g] groups by key identity, null one
   key. The wider table: one row per group in order of first appearance, one
   column per category. A cell two rows hold, or a value under a null key,
   fails at the first row of the first failing group. *)

let reference d rows =
  let groups = ref [] in
  Array.iteri
    (fun i (g, k, x) ->
      match List.assoc_opt g !groups with
      | Some cell -> cell := (i, k, x) :: !cell
      | None -> groups := !groups @ [ (g, ref [ (i, k, x) ]) ])
    rows;
  let cell_of rows c =
    match List.filter (fun (_, k, _) -> k = Some c) rows with
    | [] -> None
    | (_, _, x) :: _ -> x
  in
  let failed (_, cell) =
    let rows = List.rev !cell in
    let first_row =
      List.fold_left (fun m (i, _, _) -> min m i) max_int rows
    in
    let dup =
      List.find_opt
        (fun c ->
          List.length (List.filter (fun (_, k, _) -> k = Some c) rows) >= 2)
        (List.init (Array.length d) Fun.id)
    in
    match dup with
    | Some c -> Some (first_row, Printf.sprintf "two rows with key %S" d.(c))
    | None ->
        if List.exists (fun (_, k, x) -> k = None && Option.is_some x) rows
        then Some (first_row, "a value under a null key")
        else None
  in
  match List.find_map failed !groups with
  | Some (row, why) -> Error (row, why)
  | None ->
      let g = Array.of_list (List.map fst !groups) in
      let columns =
        Array.to_list d
        |> List.mapi (fun c name ->
            ( name,
              of_ints
                (Array.of_list
                   (List.map (fun (_, cell) -> cell_of !cell c) !groups)) ))
      in
      Ok (Talon.v (("g", of_ints g) :: columns))

let small =
  Gen.(
    with_pp
      (fun ppf (d, rows) ->
        Format.fprintf ppf "[%s]@ %a" (String.concat ";" (Array.to_list d))
          (Format.pp_print_seq (fun ppf (g, k, x) ->
               Format.fprintf ppf "(%a,%a,%a)@ "
                 Format.(pp_print_option pp_print_int) g
                 Format.(pp_print_option pp_print_int) k
                 Format.(pp_print_option pp_print_int) x))
          (Array.to_seq rows))
      (let* size = int_range 0 4 in
       let d = Array.init size (Printf.sprintf "k%d") in
       let group = option (int_range 0 2) in
       let key = if size = 0 then constant None else option (int_range 0 (size - 1)) in
       let+ rows = array ~size:(int_range 0 8) (triple group key (option (int_range 0 9))) in
       (d, rows)))

let agrees (d, rows) =
  let t =
    Talon.v
      [
        ("g", of_ints (Array.map (fun (g, _, _) -> g) rows));
        ( "k",
          of_keys d
            (Array.map (fun (_, k, _) -> Option.map (Array.get d) k) rows) );
        ("x", of_ints (Array.map (fun (_, _, x) -> x) rows));
      ]
  in
  let expected = reference d rows in
  cover "a missing cell"
    (Result.is_ok expected && Array.length rows >= 1);
  cover "a null key without value"
    (Array.exists (fun (_, k, x) -> k = None && x = None) rows
    && Array.length d >= 1);
  (match expected with
  | Error (_, why) ->
      cover "fails on a value under a null key"
        (why = "a value under a null key");
      cover "fails on two rows" (why <> "a value under a null key")
  | Ok _ -> cover "succeeds" true);
  let run = Query.run (Kit.wider ~by:[ "g" ] ~key:"k" ~value:"x" (Query.of_table t)) in
  match (expected, run) with
  | Ok want, Ok got -> equal table_w want got
  | Error (row, why), Error e ->
      let msg = Format.asprintf "%a" Error.pp e in
      contains ~sub:(Printf.sprintf "row %d: pivot finds %s" row why) msg
  | Ok _, Error e -> failf "ran to %a, expected success" Error.pp e
  | Error (row, why), Ok _ ->
      failf "succeeded, expected failure at row %d: %s" row why

(* A stream's batches change nothing, and a repeated key across two batches
   still fails. *)

let split_input =
  Gen.(
    let* (Long _ as l) = long in
    let (Long { table; _ }) = l in
    let+ stream = G.split table in
    (l, stream))

let streams (Long { by; table; _ }, stream) =
  cover "several batches" (List.length (batches stream) >= 2);
  let run t =
    Result.map_error
      (Format.asprintf "%a" Error.pp)
      (Query.run (Kit.wider ~by ~key:"key" ~value:"value" (Query.of_table t)))
  in
  let outcome_w =
    Testable.result table_w Testable.string
  in
  equal outcome_w (run table) (run stream)

(* Rules *)

let cat = [| "a"; "b"; "c" |]

let table ids keys xs =
  Query.of_table
    (Talon.v
       [
         ("id", of_ints ids);
         ("k", of_keys cat keys);
         ("x", of_floats xs);
       ])

let wide_q = Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x"

let rules =
  group "rules"
    [
      test "a missing cell and a null value are null" (fun () ->
          let got =
            run_ok
              (wide_q
                 (table
                    [| Some 1; Some 1; Some 2 |]
                    [| Some "a"; Some "b"; Some "a" |]
                    [| Some 1.; None; Some 3. |]))
          in
          equal table_w
            (Talon.v
               [
                 ("id", of_ints [| Some 1; Some 2 |]);
                 ("a", of_floats [| Some 1.; Some 3. |]);
                 ("b", of_floats [| None; None |]);
                 ("c", of_floats [| None; None |]);
               ])
            got);
      test "a left join's row with no match keeps its group, all null"
        (fun () ->
          let users = Query.of_table (Talon.v [ ("id", of_ints [| Some 1; Some 2 |]) ]) in
          let orders =
            Query.of_table
              (Talon.v
                 [
                   ("id", of_ints [| Some 1 |]);
                   ("k", of_keys cat [| Some "b" |]);
                   ("x", of_floats [| Some 5. |]);
                 ])
          in
          let got =
            users
            |> Query.join ~kind:Left ~on:(Join.keys [ "id" ]) orders
            |> wide_q |> run_ok
          in
          equal table_w
            (Talon.v
               [
                 ("id", of_ints [| Some 1; Some 2 |]);
                 ("a", of_floats [| None; None |]);
                 ("b", of_floats [| Some 5.; None |]);
                 ("c", of_floats [| None; None |]);
               ])
            got);
      test "a value under a null key is a data error at its frame's row, \
            naming the column"
        (fun () ->
          equal Testable.string
            {|aggregate ~by:["id"] ["x" := pivot k x]: row 2: pivot finds a value under a null key.|}
            (failure
               (wide_q
                  (table
                     [| Some 1; Some 1; Some 2 |]
                     [| Some "a"; Some "b"; None |]
                     [| Some 1.; Some 2.; Some 3. |]))));
      test "two rows in one cell are a data error at its frame's first row"
        (fun () ->
          (* The frame of id 2 starts at row 1, and its cell repeats at row 3. *)
          equal Testable.string
            {|aggregate ~by:["id"] ["x" := pivot k x]: row 1: pivot finds two rows with key "b".|}
            (failure
               (wide_q
                  (table
                     [| Some 1; Some 2; Some 2; Some 2 |]
                     [| Some "a"; Some "b"; Some "a"; Some "b" |]
                     [| Some 1.; Some 2.; Some 3.; Some 4. |]))));
      test "the first frame's failure comes first, whatever its row" (fun () ->
          (* Frames in order of first appearance: id 1 at row 0, id 2 at row 1. *)
          contains ~sub:"row 0"
            (failure
               (wide_q
                  (table
                     [| Some 1; Some 2; Some 2; Some 1 |]
                     [| Some "a"; Some "b"; Some "b"; Some "a" |]
                     [| Some 1.; Some 2.; Some 3.; Some 4. |]))));
      test "a value outside the dictionary is a data error at the cast"
        (fun () ->
          let q =
            Query.of_table
              (Talon.v
                 [
                   ("id", of_ints [| Some 1 |]);
                   ("s", Column.of_options Type.string [| Some "z" |]);
                   ("x", of_floats [| Some 1. |]);
                 ])
            |> Query.derive
                 Expr.
                   [
                     "k"
                     := cast (Type.categorical cat) (Col.string "s");
                   ]
          in
          contains ~sub:{|row 0: cannot cast "z"|}
            (failure (Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x" q)));
      test "a filter before wider removes the groups it empties" (fun () ->
          let got =
            run_ok
              (table
                 [| Some 1; Some 2 |]
                 [| Some "a"; Some "b" |]
                 [| Some 1.; Some 2. |]
              |> Query.filter Expr.(k = string "a")
              |> wide_q)
          in
          equal Testable.int 1 (Talon.rows got));
      test "an empty dictionary is an empty record" (fun () ->
          let q =
            Query.of_table
              (Talon.v
                 [
                   ("id", of_ints [| Some 1; Some 1; Some 2 |]);
                   ("k", of_keys [||] [| None; None; None |]);
                   ("x", of_floats [| None; None; None |]);
                 ])
          in
          equal table_w
            (Talon.v [ ("id", of_ints [| Some 1; Some 2 |]) ])
            (run_ok (Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x" q)));
      test "a value under an all-null key fails after categorize" (fun () ->
          (* categorize makes an empty dictionary, so no column has a field. *)
          let q =
            Query.of_table
              (Talon.v
                 [
                   ("id", of_ints [| Some 1; Some 2 |]);
                   ("k", Column.of_options Type.string [| None; None |]);
                   ("x", of_floats [| Some 3.; Some 4. |]);
                 ])
          in
          let q = require_ok ~pp:Error.pp (Kit.categorize [ "k" ] q) in
          contains ~sub:"row 0: pivot finds a value under a null key"
            (failure (Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x" q)));
      test "by = [] over no row is one row of nulls" (fun () ->
          let t =
            Talon.v
              [
                ("k", of_keys cat [||]);
                ("x", of_floats [||]);
              ]
          in
          equal table_w
            (Talon.v
               [
                 ("a", of_floats [| None |]);
                 ("b", of_floats [| None |]);
                 ("c", of_floats [| None |]);
               ])
            (run_ok (Kit.wider ~by:[] ~key:"k" ~value:"x" (Query.of_table t))));
      test "the schema needs no data" (fun () ->
          let raising =
            Source.v ~name:"nothing"
              ~schema:
                (Schema.v
                   [
                     ("id", Type.Any Type.int64);
                     ("k", Type.Any (Type.categorical cat));
                     ("x", Type.Any Type.float64);
                   ])
              (fun _ -> failwith "opened")
          in
          let q = Query.of_source raising in
          let wider = Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x" q in
          equal schema_w
            (Schema.v
               [
                 ("id", Type.Any Type.int64);
                 ("a", Type.Any Type.float64);
                 ("b", Type.Any Type.float64);
                 ("c", Type.Any Type.float64);
               ])
            (Query.schema wider);
          ignore (Format.asprintf "%a" Query.pp wider);
          let longer =
            Kit.longer ~name:"n" ~value:"v" Sel.(names [ "a"; "c" ]) wider
          in
          equal schema_w
            (Schema.v
               [
                 ("id", Type.Any Type.int64);
                 ("b", Type.Any Type.float64);
                 ("n", Type.Any (Type.categorical [| "a"; "c" |]));
                 ("v", Type.Any Type.float64);
               ])
            (Query.schema longer));
      test "a stream's cell repeated across batches still fails" (fun () ->
          let t i k' x' =
            Talon.v
              [
                ("id", of_ints [| Some i |]);
                ("k", of_keys cat [| Some k' |]);
                ("x", of_floats [| Some x' |]);
              ]
          in
          contains ~sub:{|two rows with key "a"|}
            (failure
               (Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x"
                  (Query.of_table
                     (of_batches [ t 1 "a" 1.; t 2 "a" 2.; t 1 "a" 3. ])))));
      test "a window's pivot reads each window's rows" (fun () ->
          let t =
            Talon.v
              [
                ("id", of_ints [| Some 1; Some 2 |]);
                ("k", of_keys cat [| Some "a"; Some "b" |]);
                ("x", of_floats [| Some 1.; Some 2. |]);
              ]
          in
          let got =
            run_ok
              (Query.derive
                 Expr.[ "r" := window (pivot k x) ]
                 (Query.of_table t))
          in
          let cell name =
            Expr.field Kind.float name (Col.v Record.kind "r")
          in
          equal table_w
            (Talon.v [ ("a", of_floats [| Some 1.; Some 1. |]);
                       ("b", of_floats [| None; Some 2. |]) ])
            (run_ok
               (Query.select
                  Expr.[ "a" := cell "a"; "b" := cell "b" ]
                  (Query.of_table got))));
      test "an aggregate's pivot and other reductions share frames"
        (fun () ->
          let got =
            run_ok
              (Query.aggregate ~by:[ "id" ]
                 Expr.
                   [
                     "r" := pivot k x;
                     "n" := rows;
                   ]
                 (table [| Some 1; Some 1 |]
                    [| Some "a"; Some "b" |]
                    [| Some 1.; Some 2. |]))
          in
          equal Testable.(array (option int)) [| Some 2 |]
            (Column.options Kind.int (Talon.column got "n"));
          equal Testable.(array (option float_exact)) [| Some 2. |]
            (Column.options Kind.float
               (Talon.column
                  (run_ok
                     (Query.select
                        Expr.[ within (Col.v Record.kind "r") (keep Sel.(names [ "b" ])) ]
                        (Query.of_table got)))
                  "b")));
      test "an extension value keeps its type and values" (fun () ->
          let ext = Type.ext ~name:"money" Type.int64 in
          let x =
            Result.get_ok
              (Column.of_layout (Any ext)
                 (Fixed
                    {
                      validity = None;
                      values = P (Nx.create Nx.int64 [| 2 |] [| 7L; 8L |]);
                    }))
          in
          let t =
            Talon.v
              [
                ("id", of_ints [| Some 1; Some 2 |]);
                ("k", of_keys cat [| Some "a"; Some "a" |]);
                ("x", x);
              ]
          in
          let got =
            run_ok (Kit.wider ~by:[ "id" ] ~key:"k" ~value:"x" (Query.of_table t))
          in
          equal schema_w
            (Schema.v
               [
                 ("id", Type.Any Type.int64);
                 ("a", Type.Any ext);
                 ("b", Type.Any ext);
                 ("c", Type.Any ext);
               ])
            (Talon.schema got);
          (* A declaration reads what the column holds. *)
          let money =
            Ext.v ~name:"money" ~ordered:false Type.int64 ~dec:Fun.id ~enc:Fun.id
          in
          equal Testable.(array int) [| 7; 8 |]
            (require_ok ~pp:Error.pp
               (Query.values (Ext.col money "a") (Query.of_table got))));
    ]

(* Longer and list *)

let lists =
  let of_list_of c = Talon.column (run_ok c) "v" in
  group "longer and list"
    [
      test "one row per cell, row after row, nulls included" (fun () ->
          let t =
            Talon.v
              [
                ("id", of_ints [| Some 1; Some 2 |]);
                ("a", of_floats [| Some 1.; None |]);
                ("b", of_floats [| None; Some 2. |]);
              ]
          in
          equal table_w
            (Talon.v
               [
                 ("id", of_ints [| Some 1; Some 1; Some 2; Some 2 |]);
                 ( "name",
                   of_keys [| "a"; "b" |]
                     [| Some "a"; Some "b"; Some "a"; Some "b" |] );
                 ("value", of_floats [| Some 1.; None; None; Some 2. |]);
               ])
            (run_ok
               (Kit.longer ~name:"name" ~value:"value"
                  Sel.(names [ "a"; "b" ])
                  (Query.of_table t))));
      test "a selected column may be named name or value" (fun () ->
          let t =
            Talon.v
              [
                ("id", of_ints [| Some 1 |]);
                ("value", of_floats [| Some 1. |]);
                ("x", of_floats [| Some 2. |]);
              ]
          in
          let got =
            run_ok
              (Kit.longer ~name:"name" ~value:"value"
                 Sel.(names [ "value"; "x" ])
                 (Query.of_table t))
          in
          equal schema_w
            (Schema.v
               [
                 ("id", Type.Any Type.int64);
                 ("name", Type.Any (Type.categorical [| "value"; "x" |]));
                 ("value", Type.Any Type.float64);
               ])
            (Talon.schema got));
      test "selected columns widen to their common type" (fun () ->
          let t =
            Talon.v
              [
                ("a", Column.of_options Type.int32 [| Some 1 |]);
                ("b", of_ints [| Some 2 |]);
              ]
          in
          let got =
            run_ok
              (Kit.longer ~name:"n" ~value:"v" Sel.all (Query.of_table t))
          in
          equal schema_w
            (Schema.v
               [
                 ("n", Type.Any (Type.categorical [| "a"; "b" |]));
                 ("v", Type.Any Type.int64);
               ])
            (Talon.schema got));
      cases "misuse raises"
        ~id:fst
        [
          ( "no column selected",
            fun q -> Kit.longer ~name:"n" ~value:"v" (Sel.names []) q );
          ( "name is an unselected column",
            fun q -> Kit.longer ~name:"a" ~value:"v" (Sel.names [ "b" ]) q );
          ( "value is an unselected column",
            fun q -> Kit.longer ~name:"n" ~value:"a" (Sel.names [ "b" ]) q );
        ]
        (fun (_, f) ->
          let q =
            Query.of_table
              (Talon.v
                 [ ("a", of_ints [| Some 1 |]); ("b", of_ints [| Some 2 |]) ])
          in
          contains ~sub:"Kit.longer" (problem (fun () -> ignore (f q))));
      test "an extension column raises" (fun () ->
          let ext = Type.ext ~name:"money" Type.int64 in
          let q =
            Query.of_table
              (Talon.v [ ("m", Column.of_options ext [| None |]) ])
          in
          contains ~sub:"extension"
            (problem (fun () ->
                 ignore (Kit.longer ~name:"n" ~value:"v" (Sel.names [ "m" ]) q))));
      test "columns of different kinds are the handle's problem" (fun () ->
          let q =
            Query.of_table
              (Talon.v
                 [ ("a", of_ints [| Some 1 |]); ("b", of_floats [| Some 2. |]) ])
          in
          contains ~sub:"Col.int"
            (problem (fun () ->
                 ignore (Kit.longer ~name:"n" ~value:"v" Sel.all q))));
      test "a list holds a row's values in order, a null element for a null"
        (fun () ->
          let t =
            Talon.v
              [
                ("a", of_ints [| Some 1; None |]);
                ("b", Column.of_options Type.int32 [| Some 2; Some 3 |]);
              ]
          in
          let v = Expr.(list [ Col.int "a"; Col.int "b"; int 9 ]) in
          let got =
            of_list_of (Query.unnest Expr.[ "v" := v ] (Query.of_table t))
          in
          equal Testable.(array (option int))
            [| Some 1; Some 2; Some 9; None; Some 3; Some 9 |]
            (Column.options Kind.int got));
      test "a list of lists nests" (fun () ->
          let t =
            Talon.v [ ("xs", Column.v Type.(list int64) [| [| 1; 2 |] |]) ]
          in
          let xs = Col.v Kind.(list int) "xs" in
          let got =
            of_list_of
              (Query.select Expr.[ "v" := list [ xs; xs ] ] (Query.of_table t))
          in
          equal Testable.(array (array (array int)))
            [| [| [| 1; 2 |]; [| 1; 2 |] |] |]
            (Column.values Kind.(list (list int)) got));
      test "a list of only literals takes store's type" (fun () ->
          let v = Expr.(store Type.(list int32) (list [ null; int 1 ])) in
          let got =
            of_list_of
              (Query.select Expr.[ "v" := v ]
                 (Query.of_table
                    (Talon.v [ ("i", of_ints [| Some 1 |]) ])))
          in
          let (Type.Any ty) = Column.type_ got in
          equal Testable.string "list[int32]"
            (Format.asprintf "%a" Type.pp ty));
      cases "list's problems" ~id:fst
        [
          ("no element", Expr.list []);
          ("no meeting type", Expr.list [ Col.int "i"; Col.int "u" ]);
        ]
        (fun (_, e) ->
          contains ~sub:"list"
            (problem (fun () ->
                 ignore
                   (Query.select
                      Expr.[ "v" := e ]
                      (Query.of_table
                         (Talon.v
                            [
                              ("i", Column.of_options Type.int8 [| Some 1 |]);
                              ("u", Column.of_options Type.uint64 [| Some 1 |]);
                            ]))))));
      test "per_list reads a row's list" (fun () ->
          let t =
            Talon.v
              [
                ("a", of_floats [| Some 1.; Some 3. |]);
                ("b", of_floats [| Some 3.; None |]);
              ]
          in
          let v =
            Expr.(
              per_list
                [ "e" := list [ Col.float "a"; Col.float "b" ] ]
                (mean (Col.float "e")))
          in
          equal Testable.(array (option float_exact))
            [| Some 2.; Some 3. |]
            (Column.options Kind.float
               (of_list_of (Query.select Expr.[ "v" := v ] (Query.of_table t)))));
      cases "printed plans show the form written" ~id:Fun.id
        [ "pivot k x"; "list [i]" ]
        (fun sub ->
          let t =
            Query.of_table
              (Talon.v
                 [
                   ("i", of_ints [| Some 1 |]);
                   ("k", of_keys cat [| Some "a" |]);
                   ("x", of_floats [| Some 1. |]);
                 ])
          in
          let q =
            if sub = "pivot k x" then
              Query.aggregate ~by:[] Expr.[ "r" := pivot k x ] t
            else Query.select Expr.[ "v" := list [ Col.int "i" ] ] t
          in
          contains ~sub (Format.asprintf "%a" Query.pp q));
      test "a key that is not categorical is a problem naming the type"
        (fun () ->
          let t =
            Query.of_table
              (Talon.v
                 [
                   ("id", of_ints [| Some 1 |]);
                   ("k", Column.of_options Type.string [| Some "a" |]);
                   ("x", of_floats [| Some 1. |]);
                 ])
          in
          contains ~sub:"pivot takes a categorical key, not string"
            (problem (fun () -> ignore (wide_q t))));
      test "a field named as a key column is a problem" (fun () ->
          let t =
            Query.of_table
              (Talon.v
                 [
                   ("a", of_ints [| Some 1 |]);
                   ("k", of_keys cat [| Some "a" |]);
                   ("x", of_floats [| Some 1. |]);
                 ])
          in
          contains ~sub:{|"a"|}
            (problem (fun () ->
                 ignore (Kit.wider ~by:[ "a" ] ~key:"k" ~value:"x" t))));
    ]

let () =
  exit
    (run "Pivot"
       [
         group "laws"
           [
             prop "wider restores what longer spread (L1)" wide l1;
             prop "longer restores what wider folded, nulls dropped (L2)"
               long l2;
             prop "wider agrees with a reference" small agrees;
             prop "a stream's batches change nothing" split_input streams;
           ];
         rules;
         lists;
       ])
