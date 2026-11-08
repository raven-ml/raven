(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let exec t = Error.get_ok (Query.run t)

(* [hash_q ?seed q] is [Expr.hash] of [q]'s one column, and [hash_of] that of a
   table of [c]. *)
let hash_q ?seed q =
  let hashed =
    Expr.[ each Sel.all { column = (fun _ x -> "h" := hash ?seed x) } ]
  in
  Column.options Kind.int (column (exec (Query.select hashed q)) "h")

let hash_of ?seed c = hash_q ?seed (Query.of_table (v [ ("x", c) ]))

(* Golden: golden/hash.txt holds the reference xxHash64 of each value, by
   kind, from the published algorithm, in the encoding Expr.hash states. *)

let golden =
  In_channel.with_open_text "golden/hash.txt" In_channel.input_lines
  |> List.map (String.split_on_char ' ')

let hex s =
  String.init (String.length s / 2) (fun i ->
      Char.chr (int_of_string ("0x" ^ String.sub s (2 * i) 2)))

let float_of_hex s = Int64.float_of_bits (Int64.of_string ("0x" ^ s))

let span_of u v =
  let v = int_of_string v in
  Time.Span.of_ns
    (Int64.mul (Int64.of_int v)
       (match u with
       | "ns" -> 1L
       | "us" -> 1000L
       | "ms" -> 1_000_000L
       | _ -> 1_000_000_000L))

let dt_unit u =
  match u with
  | "ns" -> Type.Ns
  | "us" -> Type.Us
  | "ms" -> Type.Ms
  | _ -> Type.S

let int64_column dt v =
  Column.of_tensor (Nx.create dt [| 1 |] [| Int64.of_string v |])

let split_fields s =
  (* Fields joined by ',', a nested record in parentheses. *)
  let n = String.length s in
  let rec go start depth i acc =
    if i = n then List.rev (String.sub s start (i - start) :: acc)
    else
      match s.[i] with
      | '(' -> go start (depth + 1) (i + 1) acc
      | ')' -> go start (depth - 1) (i + 1) acc
      | ',' when depth = 0 ->
          go (i + 1) depth (i + 1) (String.sub s start (i - start) :: acc)
      | _ -> go start depth (i + 1) acc
  in
  go 0 0 0 []

let ab_type = Type.(record [ ("a", Any int64); ("b", Any string) ])

let ab_cell s =
  match split_fields s with
  | [ a; b ] ->
      let opt f = function "null" -> None | x -> Some (f x) in
      Record.empty
      |> Record.add Kind.int "a" (opt int_of_string a)
      |> Record.add Kind.string "b" (opt hex b)
  | _ -> invalid_arg s

let nested_type = Type.(record [ ("r", Any ab_type); ("c", Any int32) ])

let nested_cell s =
  match split_fields s with
  | [ r; c ] ->
      let r = String.sub r 1 (String.length r - 2) in
      Record.empty
      |> Record.add Record.kind "r" (Some (ab_cell r))
      |> Record.add Kind.int "c"
           (match c with "null" -> None | c -> Some (int_of_string c))
  | _ -> invalid_arg s

let column_of kind value =
  let ints ty = Column.v ty [| int_of_string value |] in
  match kind with
  | "int8" -> ints Type.int8
  | "int16" -> ints Type.int16
  | "int32" -> ints Type.int32
  | "int64" -> int64_column Nx.int64 value
  | "uint8" -> ints Type.uint8
  | "uint16" -> ints Type.uint16
  | "uint32" -> ints Type.uint32
  | "uint64" ->
      Column.of_tensor
        (Nx.create Nx.uint64 [| 1 |] [| Int64.of_string ("0u" ^ value) |])
  | "bool" -> Column.v Type.bool [| value = "true" |]
  | "float64" -> Column.v Type.float64 [| float_of_hex value |]
  | "float32" -> Column.v Type.float32 [| float_of_hex value |]
  | "float16" -> Column.v Type.float16 [| float_of_hex value |]
  | "string" -> Column.v Type.string [| hex value |]
  | "binary" -> Column.v Type.binary [| Binary.of_string (hex value) |]
  | "categorical" ->
      let s = hex value in
      Column.v (Type.categorical [| "other"; s |]) [| s |]
  | "date" -> Column.v Type.date [| Option.get (Time.Date.of_days (int_of_string value)) |]
  | "null" -> Column.of_options Type.int64 [| None |]
  | "empty" -> Column.of_options (Type.record []) [| Some Record.empty |]
  | "wide" ->
      let vs = List.map int_of_string (String.split_on_char ',' value) in
      let name i = "f" ^ string_of_int i in
      let add (i, r) v = (i + 1, Record.add Kind.int (name i) (Some v) r) in
      Column.of_options
        (Type.record (List.mapi (fun i _ -> (name i, Type.Any Type.int64)) vs))
        [| Some (snd (List.fold_left add (0, Record.empty) vs)) |]
  | "wrap_s" ->
      let values = Nx.P (Nx.create Nx.int64 [| 1 |] [| Int64.of_string value |]) in
      Result.get_ok
        (Column.of_layout (Any (Type.instant Type.S))
           (Fixed { validity = None; values }))
  | "record" -> Column.of_options ab_type [| Some (ab_cell value) |]
  | "nested" -> Column.of_options nested_type [| Some (nested_cell value) |]
  | k -> (
      match String.split_on_char '_' k with
      | [ "datetime"; u ] ->
          Column.v (Type.instant (dt_unit u))
            [| Time.of_ns (Time.Span.to_ns (span_of u value)) |]
      | [ "duration"; u ] ->
          Column.v (Type.duration (dt_unit u)) [| span_of u value |]
      | [ "clock"; u ] -> Column.v (Type.clock (dt_unit u)) [| span_of u value |]
      | _ -> invalid_arg k)

let golden_cases =
  List.map
    (function
      | [ kind; value; seed; expected ] -> (kind, value, seed, expected)
      | _ -> failwith "golden/hash.txt")
    golden

let oracle (kind, value, seed, expected) =
  equal int (int_of_string expected)
    (Option.get
       (hash_of ~seed:(int_of_string seed) (column_of kind value)).(0))

(* Laws *)

let table = Testable.make ~pp:Talon.pp ~equal:Talon.equal

let widths = Gen.int_range (-128) 127

let widths_law v =
  let h ty = hash_of (Column.v ty [| v |]) in
  let same t0 t1 = equal (array (option int)) (h t0) (h t1) in
  List.iter (same Type.int64) Type.[ int8; int16; int32 ];
  if v >= 0 then List.iter (same Type.int64) Type.[ uint8; uint16; uint64 ]

let float_law x =
  let h ty = hash_of (Column.v ty [| x |]) in
  equal (array (option int)) (h Type.float64) (h Type.float32)

let floats = Gen.map (fun i -> float_of_int i *. 0.25) (Gen.int_range (-1000) 1000)

let zero_nan_law () =
  let f64 x = hash_of (Column.v Type.float64 [| x |]) in
  equal (array (option int)) (f64 0.) (f64 (-0.));
  equal (array (option int)) (f64 nan) (f64 Float.quiet_nan);
  equal (array (option int))
    (f64 nan)
    (f64 (Int64.float_of_bits 0xFFF8_0000_0000_0123L))

let null_law () =
  let null_of c = (hash_of ~seed:3 c).(0) in
  let int32 = null_of (Column.of_options Type.int32 [| None |]) in
  let text = null_of (Column.of_options Type.string [| None |]) in
  equal (option int) int32 text

let text_law s =
  let as_text = Column.v Type.string [| s |] in
  let dict = Column.v (Type.categorical [| s; s ^ "x" |]) [| s |] in
  equal (array (option int)) (hash_of as_text) (hash_of dict);
  equal (array (option int))
    (hash_of as_text)
    (hash_of (Column.v Type.binary [| Binary.of_string s |]))

let texts = Gen.string_of ~size:(Gen.int_range 0 40) (Gen.char_range 'a' 'z')

let record_law (a, s) =
  let at name = Type.(record [ (name, Any int64); ("b", Any string) ]) in
  let cell name =
    Record.empty
    |> Record.add Kind.int name (Some a)
    |> Record.add Kind.string "b" (Some s)
  in
  equal (array (option int))
    (hash_of (Column.of_options (at "a") [| Some (cell "a") |]))
    (hash_of (Column.of_options (at "z") [| Some (cell "z") |]))

(* [with_split xs] is a table of [xs] cut into batches, as drawn. *)
let with_split xs =
  let t = v [ ("x", Column.v Type.int64 (Array.of_list xs)) ] in
  Gen.map (fun split -> (t, split)) (Talon_gen.split t)

let batches =
  Gen.(
    let* xs = list ~size:(int_range 0 60) (int_range (-9) 9) in
    let* seed = int_range 0 1000 in
    let* ts = with_split xs in
    constant (ts, seed))

let batching_law ((t, split), seed) =
  let q t =
    exec
      (Query.derive
         Expr.[ "h" := hash ~seed (Col.int "x") ]
         (Query.of_table t))
  in
  equal table (q t) (q split)

let datetime_widths_law us =
  let t = Time.of_ns (Int64.mul (Int64.of_int us) 1000L) in
  let h u = hash_of (Column.v (Type.instant u) [| t |]) in
  equal (array (option int)) (h Type.Us) (h Type.Ns)

(* A column's rows each hash as they do in a column of their own: the rows,
   nulls included, that a multi-row kernel reads, views at an offset too. *)
let hashable (Talon_gen.Sample (ty, _)) =
  let rec ok : type a. a Type.t -> bool = function
    | List _ | Tensor _ | Ext _ -> false
    | Record fs -> List.for_all (fun (_, Type.Any t) -> ok t) fs
    | _ -> true
  in
  ok ty

let rows_law (Talon_gen.Sample (ty, rows) as s) =
  assume (hashable s && Array.length rows > 0);
  let c = Column.of_options ty rows in
  cover "a null" (Array.exists Option.is_none rows);
  let each = Array.map (fun r -> (hash_of (Column.of_options ty [| r |])).(0)) rows in
  equal (array (option int)) each (hash_of c);
  let n = Array.length rows - 1 in
  equal (array (option int))
    (Array.sub each 1 n)
    (hash_q (Query.slice ~offset:1 ~length:n (Query.of_table (v [ ("x", c) ]))))

(* Domains: the hash reads each row's value, so two domains agree. *)
let domains_law () =
  let t = v [ ("x", Column.v Type.int64 (Array.init 1000 Fun.id)) ] in
  let q () =
    exec
      (Query.derive
         Expr.[ "h" := hash ~seed:7 (Col.int "x") ]
         (Query.of_table t))
  in
  let d = Domain.spawn q in
  let here = q () in
  equal table here (Domain.join d)

let refused () =
  let plan ty x =
    let t = v [ ("c", Column.of_options ty [| Some x |]) ] in
    Query.derive
      Expr.
        [ each (Sel.names [ "c" ]) { column = (fun _ c -> "h" := hash c) } ]
      (Query.of_table t)
  in
  let refuses ty x = raises_match (Exn.invalid_arg ~sub:"is not one") (fun () -> plan ty x) in
  refuses (Type.list Type.int64) [| 1 |];
  let in_record = Type.record [ ("l", Type.Any (Type.list Type.int64)) ] in
  refuses in_record (Record.add (Kind.list Kind.int) "l" (Some [| 1 |]) Record.empty);
  refuses (Type.tensor Nx.int32 [| 2 |]) (Nx.create Nx.int32 [| 2 |] [| 1l; 2l |]);
  let values = Nx.P (Nx.create Nx.int64 [| 1 |] [| 1L |]) in
  let ext =
    Result.get_ok
      (Column.of_layout
         (Any (Type.ext ~name:"e" Type.int64))
         (Fixed { validity = None; values }))
  in
  let hash_ext () =
    Query.derive
      Expr.[ each (Sel.names [ "c" ]) { column = (fun _ c -> "h" := hash c) } ]
      (Query.of_table (v [ ("c", ext) ]))
  in
  raises_match (Exn.invalid_arg ~sub:"hash") hash_ext

(* Kit *)

let rows_of t name =
  Column.options Kind.int (Talon.column t name)
  |> Array.map (Option.value ~default:(-1))

let sorted a =
  let c = Array.copy a in
  Array.sort compare c;
  c

let table_int xs = v [ ("x", Column.v Type.int64 (Array.of_list xs)) ]
let ints = Gen.list ~size:(Gen.int_range 0 200) (Gen.int_range (-50) 50)

let permutation_law (xs, seed) =
  let t = table_int xs in
  let shuffled = exec (Kit.shuffle ~seed (Query.of_table t)) in
  equal (array int) (sorted (Array.of_list xs)) (sorted (rows_of shuffled "x"))

(* The order the .mli states: by the hash of each row's place. *)
let shuffle_definition_law (xs, seed) =
  let t = table_int xs in
  let by_hash =
    exec
      Query.(
        derive Expr.[ "key" := hash ~seed (window rows) ] (of_table t)
        |> sort [ Order.asc "key" ]
        |> select Expr.[ keep (Sel.names [ "x" ]) ])
  in
  equal table by_hash (exec (Kit.shuffle ~seed (Query.of_table t)))

let shuffle_batches_law ((t, split), seed) =
  let q t = exec (Kit.shuffle ~seed (Query.of_table t)) in
  equal table (q t) (q split)

let sample_rows_law (xs, n, seed) =
  let t = table_int xs in
  let n = n mod (List.length xs + 2) in
  let q = Query.of_table t in
  let s = exec (Kit.sample ~seed (`Rows n) q) in
  let head = exec (Kit.head n (Kit.shuffle ~seed q)) in
  equal table s head;
  equal int (min n (List.length xs)) (Talon.rows s)

let sample_batches_law ((t, split), seed) =
  let sampled s q = exec (Kit.sample ~seed s (Query.of_table q)) in
  List.iter
    (fun s -> equal table (sampled s t) (sampled s split))
    [ `Rows (Talon.rows t / 2); `Fraction 0.3 ]

let fraction_law (n, seed) =
  let xs = List.init n Fun.id in
  let t = table_int xs in
  let frac f =
    Array.to_list
      (rows_of (exec (Kit.sample ~seed (`Fraction f) (Query.of_table t))) "x")
  in
  let half = frac 0.5 and most = frac 0.8 in
  (* A row kept at a fraction is kept at a greater one, in input order. *)
  equal (list int) half (List.filter (fun x -> List.mem x half) most);
  equal (list int) [] (frac 0.);
  equal (list int) xs (frac 1.);
  (* The filter the .mli states. *)
  let cutoff = Int.of_float (0.5 *. 4611686018427387904.) in
  let filtered =
    exec
      (Query.filter
         Expr.(hash ~seed (window rows) < int cutoff)
         (Query.of_table t))
  in
  equal (list int) (Array.to_list (rows_of filtered "x")) half

(* Folds. *)

let groups_of g rows =
  let n = Array.length rows in
  Array.init n (fun i -> rows.(i) mod g)

let folds_table ~g rows =
  let rows = Array.of_list rows in
  v [ ("g", Column.v Type.int64 (groups_of g rows));
      ("x", Column.v Type.int64 rows) ]

let folds_gen =
  Gen.(
    let* g = int_range 1 40 in
    let* extra = list ~size:(int_range 0 120) (int_range 0 1000) in
    let* k = int_range 1 g in
    pair (constant (g, k)) (constant (List.init g Fun.id @ extra)))

let folds_law ((g, k), rows) =
  let t = folds_table ~g rows in
  let q = Query.of_table t in
  let r = exec (Kit.folds ~seed:9 ~by:[ "g" ] k q) in
  let gs = rows_of r "g" and fs = rows_of r "fold" in
  equal (array int) (rows_of t "x") (rows_of r "x");
  (* The fold the .mli states: the [r]th of the groups ranked by the hash of
     their one-key record takes [k (r - 1) / g]. *)
  let key_ty = Type.(record [ ("g", Any int64) ]) in
  let hash_group v =
    let cell = Record.add Kind.int "g" (Some v) Record.empty in
    Option.get (hash_of ~seed:9 (Column.of_options key_ty [| Some cell |])).(0)
  in
  let ranked =
    List.sort_uniq compare (Array.to_list gs)
    |> List.map (fun v -> (hash_group v, v))
    |> List.sort compare
  in
  let expected v =
    let r = Option.get (List.find_index (fun (_, x) -> x = v) ranked) in
    k * r / g
  in
  equal (array int) (Array.map expected gs) fs;
  let again = exec (Kit.folds ~seed:9 ~by:[ "g" ] k q) in
  equal table r again

let folds_nulls () =
  let g = Column.of_options Type.int64 [| Some 1; None; Some 1; None |] in
  let t = v [ ("g", g); ("x", Column.v Type.int64 [| 0; 1; 2; 3 |]) ] in
  let r = exec (Kit.folds ~seed:2 ~by:[ "g" ] 2 (Query.of_table t)) in
  let f = rows_of r "fold" in
  equal int f.(0) f.(2);
  equal int f.(1) f.(3);
  equal int 1 (abs (f.(0) - f.(1)))

let folds_too_few () =
  let t = v [ ("g", Column.v Type.int64 [| 1; 1; 2 |]) ] in
  raises_match
    (Exn.invalid_arg ~sub:"3 folds need at least as many groups, not 2")
    (fun () -> exec (Kit.folds ~seed:2 ~by:[ "g" ] 3 (Query.of_table t)));
  let empty = v [ ("g", Column.v Type.int64 [||]) ] in
  raises_match
    (Exn.invalid_arg ~sub:"1 folds need at least as many groups, not 0")
    (fun () -> exec (Kit.folds ~seed:2 ~by:[ "g" ] 1 (Query.of_table empty)))

(* Every NaN key is one group, and a float key hashes as a float. *)
let folds_nan_keys () =
  let g = Column.v Type.float64 [| nan; 1.; Float.quiet_nan; -0.; 0. |] in
  let t = v [ ("g", g); ("x", Column.v Type.int64 [| 0; 1; 2; 3; 4 |]) ] in
  let f = rows_of (exec (Kit.folds ~seed:2 ~by:[ "g" ] 3 (Query.of_table t))) "fold" in
  equal int f.(0) f.(2);
  equal int f.(3) f.(4)

let folds_named () =
  let t = table_int [ 1 ] in
  raises_match (Exn.invalid_arg ~sub:"column \"fold\"") (fun () ->
      Kit.folds ~seed:1 ~by:[ "fold" ] 1
        (Query.select Expr.[ "fold" := Col.int "x" ] (Query.of_table t)))

(* Several keys hash as a record of [by]'s columns, in [by]'s order. *)
let folds_key_order () =
  let k = 3 and seed = 5 in
  let a = [| 1; 1; 2; 2; 3; 3 |]
  and b = [| "x"; "y"; "x"; "y"; "x"; "y" |] in
  let t = v [ ("a", Column.v Type.int64 a); ("b", Column.v Type.string b) ] in
  let r = exec (Kit.folds ~seed ~by:[ "b"; "a" ] k (Query.of_table t)) in
  let ty = Type.(record [ ("b", Any string); ("a", Any int64) ]) in
  let hash_row i =
    let cell =
      Record.empty
      |> Record.add Kind.string "b" (Some b.(i))
      |> Record.add Kind.int "a" (Some a.(i))
    in
    Option.get (hash_of ~seed (Column.of_options ty [| Some cell |])).(0)
  in
  let hashes = Array.init 6 hash_row in
  (* The expression the .mli gives for a split that must hold. *)
  let by_record =
    Query.derive
      Expr.[ "h" := hash ~seed (record [ keep (Sel.names [ "b"; "a" ]) ]) ]
      (Query.of_table t)
  in
  equal (array int) hashes (rows_of (exec by_record) "h");
  let sorted = sorted hashes in
  let rank h = Option.get (Array.find_index (Int.equal h) sorted) in
  equal (array int)
    (Array.map (fun h -> k * rank h / 6) hashes)
    (rows_of r "fold")

let folds_multi_keys () =
  let g = Column.v Type.int64 [| 1; 1; 2; 2 |]
  and h = Column.v Type.string [| "a"; "b"; "a"; "b" |] in
  let t = v [ ("g", g); ("h", h) ] in
  let r = exec (Kit.folds ~seed:2 ~by:[ "g"; "h" ] 4 (Query.of_table t)) in
  equal (array int) [| 0; 1; 2; 3 |] (sorted (rows_of r "fold"))

let () =
  exit
    (run "hash"
       [
         cases "reference vectors"
           ~id:(fun (k, v, s, _) -> Printf.sprintf "%s %s %s" k v s)
           golden_cases oracle;
         group "laws"
           [
             prop "signed widths are one value" widths widths_law;
             prop "float32 and float64" floats float_law;
             test "-0. and NaN are canonical" zero_nan_law;
             test "a null is one word" null_law;
             prop "text and categorical and binary" texts text_law;
             prop "field names stay out"
               Gen.(pair (int_range (-1000) 1000) texts)
               record_law;
             prop "batching" batches batching_law;
             prop "datetime units" (Gen.int_range 0 1_000_000) datetime_widths_law;
             prop "rows hash as themselves" Talon_gen.sample rows_law;
             test "two domains agree" domains_law;
             test "a list is a problem" refused;
           ];
         group "shuffle"
           [
             prop "its order is the place's hash"
               (Gen.pair ints (Gen.int_range 0 1000))
               shuffle_definition_law;
             prop "is a permutation"
               (Gen.pair ints (Gen.int_range 0 1000))
               permutation_law;
             prop "batching" batches shuffle_batches_law;
           ];
         group "sample"
           [
             prop "rows are shuffle's first"
               (Gen.triple ints (Gen.int_range 0 400) (Gen.int_range 0 1000))
               sample_rows_law;
             prop "batching" batches sample_batches_law;
             prop "fractions nest"
               (Gen.pair (Gen.int_range 0 200) (Gen.int_range 0 1000))
               fraction_law;
           ];
         group "folds"
           [
             prop "balances groups, one fold each" folds_gen folds_law;
             test "a null key is one group" folds_nulls;
             test "NaN keys are one group" folds_nan_keys;
             test "a column named fold" folds_named;
             test "fewer groups than folds" folds_too_few;
             test "several keys" folds_multi_keys;
             test "the keys' order" folds_key_order;
           ];
       ])
