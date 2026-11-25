(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module P = Talon_parquet
module G = Talon_gen

let any t = Type.Any t
let error_text e = Format.asprintf "%a" Error.pp e

let encode_with f q =
  let b = Buffer.create 4096 in
  Error.get_ok (P.encode f q (Bytesrw.Bytes.Writer.of_buffer b));
  Buffer.contents b

let encode ?compression ?row_group t =
  let q = Query.of_table t in
  encode_with (P.format ?compression ?row_group (Query.schema q)) q

let read_with f s =
  let b = Rig.Buffer.of_string s in
  Error.get_ok (Query.run (Query.of_source (P.source (f b) b)))

let decode s = read_with (fun b -> Error.get_ok (P.sniff b)) s

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

let laid_out ty layout =
  match Column.of_layout ty layout with
  | Ok c -> c
  | Error (_, why) -> failwith why

(* Writing *)

(* A list adds two definition levels and an optional leaf one: 32,767 lists of
   [int32] reach 65,535, the most a 16-bit level holds, and a record around them
   one more. *)
let level_bound =
  test "format refuses a definition level past 16 bits, naming the column"
  @@ fun () ->
  let rec lists k (Type.Any t as ty) =
    if k = 0 then ty else lists (k - 1) (any (Type.list t))
  in
  let deepest = lists 32_767 (any Type.int32) in
  ignore (P.format (Schema.v [ ("a", deepest) ]));
  let past = any (Type.record [ ("x", deepest) ]) in
  raises_match (Exn.invalid_arg ~sub:"\"a\"") (fun () ->
      P.format (Schema.v [ ("a", past) ]))

(* Written types *)

(* The Parquet type TP lists for each talon type, and the type it reads back as,
   as pp_format prints them. *)
let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let written_types =
  cases "format gives each type the Parquet type it writes as"
    ~id:(fun (ty, _, _) -> type_name ty)
    [
      (any Type.bool, "bool", "boolean");
      (any Type.int8, "int8", "int32 (INTEGER(8,true))");
      (any Type.int16, "int16", "int32 (INTEGER(16,true))");
      (any Type.int32, "int32", "int32");
      (any Type.int64, "int64", "int64");
      (any Type.uint8, "uint8", "int32 (INTEGER(8,false))");
      (any Type.uint16, "uint16", "int32 (INTEGER(16,false))");
      (any Type.uint32, "uint32", "int32 (INTEGER(32,false))");
      (any Type.uint64, "uint64", "int64 (INTEGER(64,false))");
      (any Type.float16, "float16", "fixed_len_byte_array(2) (FLOAT16)");
      (any Type.float32, "float32", "float");
      (any Type.float64, "float64", "double");
      (any Type.string, "string", "binary (STRING)");
      (any Type.binary, "binary", "binary");
      (any (Type.categorical [| "a" |]), "string", "binary (STRING)");
      (any Type.date, "date", "int32 (DATE)");
      (any (Type.clock S), "clock[ms]", "int32 (TIME(MILLIS,false))");
      (any (Type.clock Ms), "clock[ms]", "int32 (TIME(MILLIS,false))");
      (any (Type.clock Us), "clock[us]", "int64 (TIME(MICROS,false))");
      (any (Type.clock Ns), "clock[ns]", "int64 (TIME(NANOS,false))");
      (any (Type.wall S), "datetime[ms]", "int64 (TIMESTAMP(MILLIS,false))");
      (any (Type.wall Us), "datetime[us]", "int64 (TIMESTAMP(MICROS,false))");
      ( any (Type.instant S),
        "datetime[ms, UTC]",
        "int64 (TIMESTAMP(MILLIS,true))" );
      ( any (Type.instant Ns),
        "datetime[ns, UTC]",
        "int64 (TIMESTAMP(NANOS,true))" );
      (any (Type.ext ~name:"m" Type.int32), "int32", "int32");
    ]
    (fun (ty, reads, parquet) ->
      equal text
        (Printf.sprintf "parquet (1 column)\n  a %s ← optional %s" reads parquet)
        (Format.asprintf "%a" P.pp_format (P.format (Schema.v [ ("a", ty) ]))))

(* Nested types, whose Parquet groups print whole. *)
let written_groups () =
  List.iter
    (fun ty ->
      print_endline
        (Format.asprintf "%a" P.pp_format (P.format (Schema.v [ ("a", ty) ]))))
    [
      any (Type.list Type.string);
      any
        (Type.record
           [ ("x", any Type.int32); ("y", any (Type.list Type.bool)) ]);
      any (Type.tensor Nx.float32 [| 2; 3 |]);
    ];
  expect (output ())
  @@ __POS_OF__
       {|
    parquet (1 column)
      a list[string] ← optional group (LIST) { repeated group list { optional binary element (STRING); } }
    parquet (1 column)
      a record[x int32, y list[bool]] ← optional group { optional int32 x; optional group y (LIST) { repeated group list { optional boolean element; } } }
    parquet (1 column)
      a list[list[float32]] ← optional group (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional float element; } } } }
    |}

(* A tensor column as the nested lists it reads back as, row-major: each row's
   outermost list null where the tensor is, from the innermost dimension out. *)
let tensor_lists shape validity (Nx.P x) =
  let shape = Iarray.to_array shape in
  let n = Nx.dim 0 x in
  let elements = Column.of_tensor (Nx.reshape [| -1 |] x) in
  let wrap c d =
    let rows = Array.fold_left ( * ) n (Array.sub shape 0 d) in
    let offsets =
      Nx.mul_s (Nx.arange Nx.int64 0 (rows + 1) 1) (Int64.of_int shape.(d))
    in
    let (Type.Any e) = Column.type_ c in
    let validity = if d = 0 then validity else None in
    laid_out (any (Type.list e)) (Varsize { validity; offsets; child = c })
  in
  let depths = List.rev (List.init (Array.length shape) Fun.id) in
  List.fold_left wrap elements depths

(* [normal c] is [c] with the type talon reads it back as. *)
let rec normal c =
  let (Type.Any t) = Column.type_ c in
  let options = lazy (Column.options (Type.kind t) c) in
  match (t, Column.layout c) with
  | Categorical _, _ -> Column.of_options Type.string (Lazy.force options)
  | Clock S, _ -> Column.of_options (Type.clock Ms) (Lazy.force options)
  | Datetime { unit_ = S; clock = Time.Utc }, _ ->
      Column.of_options (Type.instant Ms) (Lazy.force options)
  | Datetime { unit_ = S; clock = Time.Local }, _ ->
      Column.of_options (Type.wall Ms) (Lazy.force options)
  | Ext { storage; _ }, l -> normal (laid_out (any storage) l)
  | Tensor (_, shape), Fixed { validity; values } ->
      normal (tensor_lists shape validity values)
  | List _, Varsize { validity; offsets; child } ->
      let child = normal child in
      let (Type.Any e) = Column.type_ child in
      laid_out (any (Type.list e)) (Varsize { validity; offsets; child })
  | Record _, Children { validity; length; fields } ->
      let fields = List.map (fun (f, c) -> (f, normal c)) fields in
      let types = List.map (fun (f, c) -> (f, Column.type_ c)) fields in
      laid_out (any (Type.record types)) (Children { validity; length; fields })
  | _ -> c

(* [writable t] is [true] iff [format] states a column of type [t]. *)
let rec writable (Type.Any t) =
  match t with
  | Duration _ -> false
  | List e -> writable (any e)
  | Record [] -> false
  | Record fields -> List.for_all (fun (_, t) -> writable t) fields
  | Ext { storage; _ } -> writable (any storage)
  | Tensor (dt, _) -> (
      match Column.of_tensor (Nx.zeros dt [| 0 |]) with
      | _ -> true
      | exception Invalid_argument _ -> false)
  | _ -> true

let rec nested (Type.Any t) =
  match t with
  | List _ | Record _ | Tensor _ -> true
  | Ext { storage; _ } -> nested (any storage)
  | _ -> false

let normalized t =
  let columns = Schema.columns (Talon.schema t) in
  Talon.v ~rows:(Talon.rows t)
    (List.map (fun (name, _) -> (name, normal (Talon.column t name))) columns)

let compressions : P.compression array =
  [| Uncompressed; Snappy; Gzip; Zstd; Lz4_raw |]

(* A table of a sample, a compression and a row group size. *)
let case =
  let open Gen in
  let* (G.Sample (ty, vs) as s) = G.sample in
  let t = Talon.v [ ("a", Column.of_options ty vs) ] in
  let* t' = G.split t in
  let* row_group = int_range 1 45 in
  let+ codec = int_range 0 (Array.length compressions - 1) in
  (s, t, t', row_group, compressions.(codec))

(* A tensor column of two rows, one null. *)
let tensor_example =
  let ty = Type.tensor Nx.float32 [| 2 |] in
  let vs = [| Some (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]); None |] in
  let t = Talon.v [ ("a", Column.of_options ty vs) ] in
  (G.Sample (ty, vs), t, t, 1, P.Snappy)

let round_trip =
  prop ~count:300 "a table of written types reads back"
    ~examples:[ tensor_example ] case
    (fun (G.Sample (ty, _), t, _, row_group, compression) ->
      assume (writable (any ty));
      let n = Talon.rows t in
      cover "nested" (nested (any ty));
      cover "a tensor" (match ty with Tensor _ -> true | _ -> false);
      cover "no row" (n = 0);
      cover "one row" (n = 1);
      cover "a row group boundary" (row_group <= n);
      let written = normalized t in
      Law.round_trip table_w string
        (encode ~compression ~row_group)
        decode written;
      let stated _ = P.format ~compression ~row_group (Talon.schema t) in
      equal table_w written
        (read_with stated (encode ~compression ~row_group t)))

let types_written =
  prop ~count:300 "other types write as their written type, byte for byte" case
    (fun (G.Sample (ty, _), t, _, row_group, compression) ->
      assume (writable (any ty));
      equal string
        (encode ~compression ~row_group (normalized t))
        (encode ~compression ~row_group t))

(* Nested tables

   The nulls a sample cannot hold: a list with a null element, and nulls at each
   level of lists and records. *)

let leaves =
  [
    any Type.int32;
    any Type.int64;
    any Type.float64;
    any Type.string;
    any Type.bool;
    any Type.date;
  ]

let rec nested_type depth =
  let open Gen in
  let leaf = of_list ~pp:(fun ppf (Type.Any t) -> Type.pp ppf t) leaves in
  if depth = 0 then leaf
  else
    let inner = nested_type (depth - 1) in
    frequency
      [
        (1, leaf);
        (3, map (fun (Type.Any t) -> any (Type.list t)) inner);
        ( 3,
          let+ ts = list ~size:(int_range 1 3) inner in
          any
            (Type.record
               (List.mapi (fun i t -> (Printf.sprintf "f%d" i, t)) ts)) );
      ]

let validity n =
  let open Gen in
  let bit = frequency [ (1, constant false); (3, constant true) ] in
  let+ v = array ~size:(constant n) bit in
  Nx.cast Nx.bit (Nx.create Nx.bool [| n |] v)

(* [column t n] draws [n] rows of [t], with nulls at every level. *)
let rec column (Type.Any t as ty) n =
  let open Gen in
  match t with
  | List e ->
      let* lens = array ~size:(constant n) (int_range 0 3) in
      let total = Array.fold_left ( + ) 0 lens in
      let at = ref 0L in
      let offsets =
        Array.init (n + 1) (fun i ->
            let o = !at in
            if i < n then at := Int64.add !at (Int64.of_int lens.(i));
            o)
      in
      let offsets = Nx.create Nx.int64 [| n + 1 |] offsets in
      let* child = column (any e) total in
      let+ v = validity n in
      laid_out ty (Varsize { validity = Some v; offsets; child })
  | Record fields ->
      let rec all = function
        | [] -> constant []
        | (f, ft) :: rest ->
            let* c = column ft n in
            let+ cs = all rest in
            (f, c) :: cs
      in
      let* cs = all fields in
      let+ v = validity n in
      laid_out ty (Children { validity = Some v; length = n; fields = cs })
  | _ ->
      let value = Option.get (G.value t) in
      let cell = frequency [ (1, constant None); (3, map Option.some value) ] in
      let+ vs = array ~size:(constant n) cell in
      Column.of_options t vs

(* The nulls and the empty lists a column holds, by the level they are at. *)
let rec shapes c =
  let at layout_validity =
    match layout_validity with
    | Some v -> Array.exists not (Nx.to_array (Nx.cast Nx.bool v))
    | None -> false
  in
  match Column.layout c with
  | Varsize { validity; offsets; child } when nested (Column.type_ c) ->
      let o = Nx.to_array offsets in
      let empty = ref false in
      Array.iteri (fun i a -> if i > 0 && a = o.(i - 1) then empty := true) o;
      (if at validity then [ "null list" ] else [])
      @ (if !empty then [ "empty list" ] else [])
      @ (if Column.null_count child > 0 then [ "null element" ] else [])
      @ shapes child
  | Children { validity; fields; _ } ->
      (if at validity then [ "null record" ] else [])
      @ List.concat_map (fun (_, c) -> shapes c) fields
  | _ -> []

let nested_case =
  let open Gen in
  let* ty = nested_type 3 in
  let* n = int_range 0 12 in
  let* c = column ty n in
  let* row_group = int_range 1 15 in
  let+ codec = int_range 0 (Array.length compressions - 1) in
  (Talon.v ~rows:n [ ("a", c) ], row_group, compressions.(codec))

let nested_round_trip =
  prop ~count:300 "nested tables read back, nulls at every level" nested_case
    (fun (t, row_group, compression) ->
      let labels = shapes (Talon.column t "a") in
      List.iter
        (fun l -> cover l (List.mem l labels))
        [ "null list"; "empty list"; "null element"; "null record" ];
      cover "a row group boundary" (row_group <= rows t);
      Law.round_trip table_w string (encode ~compression ~row_group) decode t)

(* Tensors: a tensor writes as nested lists, which a declaration reads back as
   the same tensor, alone or as a list's element. *)

let tensor_makers =
  let maker dt =
    (Format.asprintf "%a" Nx.Dtype.pp dt, fun s -> any (Type.tensor dt s))
  in
  [|
    maker Nx.float16;
    maker Nx.float32;
    maker Nx.float64;
    maker Nx.int8;
    maker Nx.int16;
    maker Nx.int32;
    maker Nx.int64;
    maker Nx.uint8;
    maker Nx.uint16;
    maker Nx.uint32;
    maker Nx.uint64;
    maker Nx.bool;
    maker Nx.bit;
  |]

let tensor_case =
  let open Gen in
  let* maker = int_range 0 (Array.length tensor_makers - 1) in
  let* rank = int_range 1 3 in
  let* shape = array ~size:(int_range rank rank) (int_range 0 3) in
  let* listed = bool in
  let name, make = tensor_makers.(maker) in
  let (Type.Any t) = make shape in
  let (Type.Any t) = if listed then any (Type.list t) else any t in
  let* vs = G.options t in
  let+ row_group = int_range 1 45 in
  (name, shape, G.Sample (t, vs), row_group)

let tensor_round_trip =
  prop ~count:500 "a tensor reads back with its type declared" tensor_case
    (fun (dtype, shape, G.Sample (ty, vs), row_group) ->
      Array.iter (fun (d, _) -> cover d (d = dtype)) tensor_makers;
      cover "a zero extent" (Array.mem 0 shape);
      cover "a cell of one element" (Array.for_all (( = ) 1) shape);
      cover "three dimensions" (Array.length shape = 3);
      cover "a null row" (Array.mem None vs);
      cover "a list of tensors" (match ty with List _ -> true | _ -> false);
      cover "several row groups" (row_group < Array.length vs);
      let t = Talon.v [ ("a", Column.of_options ty vs) ] in
      let declared b = P.with_type "a" (any ty) (Error.get_ok (P.sniff b)) in
      equal table_w t (read_with declared (encode ~row_group t)))

(* A dictionary keeps [-0.], [0.] and NaN apart, each repeated so that the
   column is dictionary-encoded. *)
let float_bits () =
  let values =
    Array.concat
      (List.init 20 (fun _ ->
           [| 1.5; -0.; 0.; Float.nan; Float.infinity; Float.neg_infinity |]))
  in
  let column = Column.of_options Type.float64 (Array.map Option.some values) in
  let back =
    Column.to_tensor Nx.float64
      (Talon.column (decode (encode (Talon.v [ ("x", column) ]))) "x")
  in
  equal (array int64)
    (Array.map Int64.bits_of_float values)
    (Array.map Int64.bits_of_float (Nx.to_array back))

(* Durations: the file records no unit, so format refuses one and names what
   writes its ticks. *)
let duration_refused () =
  let refuses ty sub =
    raises_match (Exn.invalid_arg ~sub) (fun () ->
        P.format (Schema.v [ ("d", ty) ]))
  in
  refuses (any (Type.duration Ms)) "Expr.cast";
  refuses (any (Type.list (Type.duration Us))) "Expr.cast";
  refuses
    (any (Type.ext ~name:"raven.span" (Type.duration S)))
    "Column.of_layout"

(* The expression the refusal names writes the ticks exactly, past 2^53. *)
let ticks_written () =
  let xs = [| Int64.min_int; Int64.max_int; 0x20_0000_0000_0001L; -1L |] in
  let values = Nx.create Nx.int64 [| Array.length xs |] xs in
  let c =
    laid_out
      (any (Type.duration Ms))
      (Fixed { validity = None; values = P values })
  in
  let q =
    Query.select
      Expr.[ "d" := cast Type.int64 (Col.span "d") ]
      (Query.of_table (Talon.v [ ("d", c) ]))
  in
  let t = decode (encode_with (P.format (Query.schema q)) q) in
  equal (array int64) xs
    (Nx.to_array (Column.to_tensor Nx.int64 (Talon.column t "d")))

(* Statistics *)

let ints = [ -5; -1; 0; 1; 5 ]
let floats = [ -0.; 0.; Float.nan; 1.5; -2.; Float.infinity ]
let long c n = String.make n c

(* Texts around the 64-byte bound, past it and ending in a multibyte character
   at the cut, and bytes of 0xFF runs, which no value lies past; their truncated
   bounds are the edges a filter draws. *)
let bounds = [ long 'a' 63; long 'q' 62; long 'q' 64; long 'z' 64 ]

let texts =
  [
    "";
    "a";
    "\xc3\xa9";
    long 'a' 63 ^ "\xc3\xa9";
    long 'q' 62 ^ "\xf0\x9f\xa6\x8a";
  ]
  @ [ long 'q' 64 ^ "\xe6\x97\xa5"; long 'z' 70 ]

let byte_bounds = [ long '\xff' 64; long 'a' 64 ]

let bytes_ =
  [ ""; "\xff"; long '\xff' 70; long '\xff' 64 ^ "\x00"; long 'a' 65 ]

let instants =
  List.map (fun k -> Option.get (Time.of_us (Int64.of_int k))) [ -3; 0; 3 ]

(* A filter on one column, and whether it compares the column with an edge of
   its statistics. *)
type filter = {
  column : string;
  pred : (bool, Expr.row) Expr.t;
  at_edge : bool;
}

(* [compared column x pool edges] draws a comparison or [In] of [x], the column
   [column], against values of [pool], half of them from [edges]. *)
let compared column (type a) (x : (a, Expr.row) Expr.t) (pool : a list) edges =
  let open Gen in
  let value =
    if edges = [] then of_list pool
    else frequency [ (1, of_list pool); (1, of_list edges) ]
  in
  let edge vs = List.exists (fun v -> List.mem v edges) vs in
  let ops = Expr.[ ( = ); ( <> ); ( < ); ( <= ); ( > ); ( >= ) ] in
  one_of
    [
      (let* op = of_list ops in
       let+ v = value in
       { column; pred = op x (Expr.const v); at_edge = edge [ v ] });
      (let+ vs = list ~size:(int_range 1 3) value in
       { column; pred = Expr.is_in vs x; at_edge = edge vs });
    ]

(* Rows of a column of each class of statistics, and a filter on each: integers,
   uint64 (which no bound prunes), floats with signed zeros and NaN, datetimes,
   and texts and bytes around the 64-byte bound, the filters' edges being the
   zeros and the truncated bounds. *)
let stats_case =
  let open Gen in
  let* n = int_range 1 30 in
  let column g = array ~size:(constant n) (option g) in
  let* i = column (of_list ints) in
  let* u = column (of_list [ 0; 1; max_int ]) in
  let* x =
    column (frequency [ (1, constant Float.nan); (2, of_list floats) ])
  in
  let* d = column (of_list instants) in
  let* s = column (of_list texts) in
  let* b = column (of_list bytes_) in
  let t =
    Talon.v
      [
        ("id", Column.v Type.int64 (Array.init n Fun.id));
        ("i", Column.of_options Type.int32 i);
        ("u", Column.of_options Type.uint64 u);
        ("x", Column.of_options Type.float64 x);
        ("d", Column.of_options (Type.wall Us) d);
        ("s", Column.of_options Type.string s);
        ( "b",
          Column.of_options Type.binary
            (Array.map (Option.map Binary.of_string) b) );
      ]
  in
  let binary = List.map Binary.of_string in
  let* pi = compared "i" (Col.int "i") ints [] in
  let* pu = compared "u" (Col.int "u") [ 0; 1; max_int ] [] in
  let* px = compared "x" (Col.float "x") floats [ 0.; -0. ] in
  let* pd = compared "d" (Col.wall "d") instants [] in
  let* ps = compared "s" (Col.string "s") texts bounds in
  let* pb =
    compared "b" (Col.binary "b") (binary bytes_) (binary byte_bounds)
  in
  let+ row_group = frequency [ (1, constant 1); (2, int_range 2 10) ] in
  (t, row_group, [ pi; pu; px; pd; ps; pb ])

let pp_stats ppf (t, row_group, filters) =
  Format.fprintf ppf "@[<v>%a@,row groups of %d, filters %a@]" Talon.pp t
    row_group
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf f ->
         Expr.pp ppf f.pred))
    filters

let statistics =
  prop ~count:40
    "a filter over a written file keeps the rows it keeps over the table"
    (Gen.with_pp pp_stats stats_case) (fun (t, row_group, filters) ->
      let n = Talon.rows t in
      let groups =
        List.init
          (((n - 1) / row_group) + 1)
          (fun k -> (k * row_group, min n ((k + 1) * row_group)))
      in
      let kept f =
        Error.get_ok (Query.run (Query.filter f.pred (Query.of_table t)))
      in
      (* [excludes f g] is [true] iff no row of the row group [g] passes [f]. *)
      let excludes f (a, b) =
        Array.for_all
          (function Some i -> i < a || i >= b | None -> true)
          (Column.options Kind.int (Talon.column (kept f) "id"))
      in
      let xs = Column.options Kind.float (Talon.column t "x") in
      cover "a row group no row of which passes"
        (List.exists (fun f -> List.exists (excludes f) groups) filters);
      cover "a constant equal to a truncated bound"
        (List.exists
           (fun f -> f.at_edge && (f.column = "s" || f.column = "b"))
           filters);
      cover "a zero against signed zeros"
        (List.exists (fun f -> f.at_edge && f.column = "x") filters);
      cover "an all-NaN group"
        (List.exists
           (fun (a, b) ->
             let vs =
               List.filter_map Fun.id (Array.to_list (Array.sub xs a (b - a)))
             in
             vs <> [] && List.for_all Float.is_nan vs)
           groups);
      let b = Rig.Buffer.of_string (encode ~row_group t) in
      let read = Error.get_ok (P.sniff b) in
      List.iter
        (fun f ->
          equal table_w ~msg:f.column (kept f)
            (Error.get_ok
               (Query.run
                  (Query.filter f.pred (Query.of_source (P.source read b))))))
        filters)

(* The corpus: tables whose files pyarrow read, golden/written.txt holding each
   file's MD5 and what pyarrow read *)

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

(* [render c] is the rows of [c], flat or nested, as golden/written.txt writes
   pyarrow's. *)
let rec render c =
  let (Type.Any t) = Column.type_ c in
  let n = Column.length c in
  let valid =
    match Column.validity c with
    | None -> Array.make n true
    | Some v -> Nx.to_array (Nx.cast Nx.bool v)
  in
  let guard f = Array.init n (fun i -> if valid.(i) then f i else "null") in
  let ints ?(scale = 1L) x =
    let vs = Nx.to_array (Nx.cast Nx.int64 x) in
    guard (fun i -> Int64.to_string (Int64.mul vs.(i) scale))
  in
  let bits_text v =
    if Float.is_nan v then "nan"
    else Printf.sprintf "%016Lx" (Int64.bits_of_float v)
  in
  match (t, Column.layout c) with
  | Bool, Fixed { values = Nx.P x; _ } ->
      let vs = Nx.to_array (Nx.cast Nx.bool x) in
      guard (fun i -> string_of_bool vs.(i))
  | (Float16 | Float32 | Float64), Fixed { values = Nx.P x; _ } ->
      let vs = Nx.to_array (Nx.cast Nx.float64 x) in
      guard (fun i -> bits_text vs.(i))
  | (Clock S | Datetime { unit_ = S; _ }), Fixed { values = Nx.P x; _ } ->
      ints ~scale:1000L x
  | Uint64, Fixed { values = Nx.P x; _ } ->
      let vs = Nx.to_array (Nx.bitcast Nx.int64 x) in
      guard (fun i -> Printf.sprintf "%Lu" vs.(i))
  | Categorical d, Fixed { values = Nx.P x; _ } ->
      let vs = Nx.to_array (Nx.cast Nx.int32 x) in
      let d = Iarray.to_array d in
      guard (fun i -> hex d.(Int32.to_int vs.(i)))
  | _, Fixed { values = Nx.P x; _ } when Nx.ndim x = 1 -> ints x
  | (String | Binary), Varsize { offsets; child; _ } ->
      let o = Nx.to_array offsets in
      let bytes = Nx.to_array (Column.to_tensor Nx.uint8 child) in
      guard (fun i ->
          let a = Int64.to_int o.(i) and e = Int64.to_int o.(i + 1) in
          hex (String.init (e - a) (fun j -> Char.chr bytes.(a + j))))
  | List _, Varsize { offsets; child; _ } ->
      let o = Nx.to_array offsets in
      let items = render child in
      guard (fun i ->
          let a = Int64.to_int o.(i) and e = Int64.to_int o.(i + 1) in
          "["
          ^ String.concat " " (Array.to_list (Array.sub items a (e - a)))
          ^ "]")
  | Record _, Children { fields; _ } ->
      let parts = List.map (fun (f, c) -> (f, render c)) fields in
      guard (fun i ->
          "{"
          ^ String.concat "," (List.map (fun (f, r) -> f ^ "=" ^ r.(i)) parts)
          ^ "}")
  | Tensor (_, shape), Fixed { values = Nx.P x; _ } ->
      let leaves = Nx.to_array (Nx.reshape [| -1 |] (Nx.cast Nx.int64 x)) in
      let shape = Iarray.to_array shape in
      let k = ref 0 in
      let rec cell d =
        if d = Array.length shape then begin
          let v = leaves.(!k) in
          incr k;
          Int64.to_string v
        end
        else
          "["
          ^ String.concat " " (List.init shape.(d) (fun _ -> cell (d + 1)))
          ^ "]"
      in
      let cells = Array.init n (fun _ -> cell 0) in
      guard (fun i -> cells.(i))
  | Ext { storage; _ }, l -> render (laid_out (any storage) l)
  | _ -> failwith (Format.asprintf "render: %a" Type.pp t)

let rendered c =
  let text = String.concat " " (Array.to_list (render c)) in
  if String.length text > 4096 then "md5:" ^ Digest.to_hex (Digest.string text)
  else text

let golden =
  In_channel.with_open_bin "golden/written.txt" In_channel.input_lines

(* The corpus's file names, from the golden's [name rows …] lines. *)
let corpus_names =
  List.filter_map
    (fun l ->
      match String.split_on_char ' ' l with
      | [ name; "rows"; _; "groups"; _ ] -> Some name
      | _ -> None)
    golden

let item name =
  List.find
    (fun (i : Written_corpus.item) -> i.name = name)
    (Lazy.force Written_corpus.items)

let golden_lines prefix =
  List.filter_map
    (fun l ->
      if String.starts_with ~prefix l then
        Some
          (String.sub l (String.length prefix)
             (String.length l - String.length prefix))
      else None)
    golden

let encoded (i : Written_corpus.item) =
  encode ~compression:i.compression ~row_group:i.row_group i.table

(* What pyarrow read of a file this writer writes today: the file's MD5 ties the
   golden to it. *)
let pyarrow_read name =
  let i = item name in
  equal (list string)
    (golden_lines (name ^ " file "))
    [ "md5:" ^ Digest.to_hex (Digest.string (encoded i)) ];
  equal (list string)
    (golden_lines (name ^ " values "))
    (List.map
       (fun (c, _) -> c ^ " " ^ rendered (Talon.column i.table c))
       (Schema.columns (Talon.schema i.table)))

let file name =
  In_channel.with_open_bin ("support/" ^ name) In_channel.input_all

(* format *)

let one_int = Talon.v [ ("a", Column.of_options Type.int32 [| Some 1 |]) ]

let format_raises =
  group "format"
    [
      test "writes with Snappy and row groups of 1_048_576 rows by default"
        (fun () ->
          equal string
            (encode ~compression:Snappy ~row_group:1_048_576 one_int)
            (encode one_int));
      test "raises for a schema of no column" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.format") (fun () ->
              P.format (Schema.v [])));
      cases "raises for a row_group of 0 or less" ~id:string_of_int [ 0; -1 ]
        (fun row_group ->
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.format") (fun () ->
              P.format ~row_group (Talon.schema one_int)));
      cases "raises for a record of no field and a tensor no scalar stores"
        ~id:fst
        [
          ("record[]", any (Type.record []));
          ("tensor[complex64]", any (Type.tensor Nx.complex64 [| 2 |]));
        ]
        (fun (_, ty) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.format") (fun () ->
              P.format (Schema.v [ ("a", ty) ])));
      test "refuses a duration, naming what writes its ticks" duration_refused;
    ]

(* encode *)

(* [failing_after t] is a query of the rows of [t], then of a part whose first
   batch fails. *)
let failing_after t =
  let ok = ref false in
  let part next =
    {
      Source.rows = None;
      open_ = (fun () -> Ok { Source.next; close = ignore });
    }
  in
  let first () =
    if !ok then Ok None
    else (
      ok := true;
      Ok (Some t))
  in
  let failed () = Error (Error.v "the second part fails") in
  Query.of_source
    (Source.v ~name:"parts" ~schema:(Talon.schema t) (fun _ ->
         Ok [ part first; part failed ]))

(* [sink_failure raise_] is [encode]'s result into a writer whose writes call
   [raise_], and the text of what [raise_] raises. *)
let sink_failure raise_ =
  let why =
    match raise_ () with
    | () -> "no exception"
    | exception Sys_error m -> m
    | exception Bytesrw.Bytes.Stream.Error e ->
        Bytesrw.Bytes.Stream.error_message e
  in
  let w = Bytesrw.Bytes.Writer.make (fun _ -> raise_ ()) in
  (P.encode (P.format (Talon.schema one_int)) (Query.of_table one_int) w, why)

(* A datetime of seconds [s] as the one row of a wall-clock column. *)
let seconds s =
  let values = Nx.create Nx.int64 [| 1 |] [| s |] in
  let c =
    laid_out (any (Type.wall S)) (Fixed { validity = None; values = P values })
  in
  let q = Query.of_table (Talon.v [ ("x", c) ]) in
  let w = Bytesrw.Bytes.Writer.of_buffer (Buffer.create 64) in
  Result.map_error error_text (P.encode (P.format (Query.schema q)) q w)

let encoding =
  group "encode"
    [
      prop ~count:300 "the bytes depend on the rows and the format only" case
        (fun (G.Sample (ty, _), t, t', row_group, compression) ->
          assume (writable (any ty));
          cover "nested" (nested (any ty));
          cover "several batches" (List.length (batches t') > 1);
          cover "a row group cuts a batch" (row_group < rows t);
          equal string
            (encode ~compression ~row_group t)
            (encode ~compression ~row_group t'));
      (* A value far larger than the page target takes a page of its own, past
         small values that do not push it over the limit. *)
      test "a value past the page target takes a page whole" (fun () ->
          let big = String.make (2 * 1024 * 1024) 'x' in
          let rows =
            Array.init 400 (fun i ->
                Some (if i mod 100 = 7 then big ^ string_of_int i else "s"))
          in
          let t = Talon.v [ ("t", Column.of_options Type.string rows) ] in
          equal table_w t (decode (encode t)));
      (* The corpus's tables past a data page's size. *)
      cases "a table past a page reads back" ~id:Fun.id
        [ "pages_int"; "pages_text"; "pages_nested" ] (fun name ->
          equal table_w
            (normalized (item name).table)
            (decode (encoded (item name))));
      cases "pyarrow reads each corpus file as its table" ~id:Fun.id
        corpus_names pyarrow_read;
      test "ends the writer with eod, after its bytes" (fun () ->
          let eods = ref [] in
          let w =
            Bytesrw.Bytes.Writer.make (fun s ->
                eods := Bytesrw.Bytes.Slice.is_eod s :: !eods)
          in
          Error.get_ok
            (P.encode
               (P.format (Talon.schema one_int))
               (Query.of_table one_int) w);
          (* Whether each slice is eod, the last first. *)
          equal (list bool) (List.mapi (fun i _ -> i = 0) !eods) !eods;
          greater int ~than:1 (List.length !eods));
      test "a run that fails keeps the row groups written before it" (fun () ->
          let t = Talon.v [ ("x", Column.v Type.int64 [| 1; 2; 3 |]) ] in
          let f = P.format ~row_group:3 (Talon.schema t) in
          let b = Buffer.create 256 in
          (match
             P.encode f (failing_after t) (Bytesrw.Bytes.Writer.of_buffer b)
           with
          | Ok () -> fail "the run did not fail"
          | Error _ -> ());
          let whole =
            encode_with f (Query.of_table (Talon.of_batches [ t; t ]))
          in
          let kept = Buffer.contents b in
          greater int ~than:4 (String.length kept);
          starts_with ~prefix:kept whole);
      cases "a sink that raises is an Error" ~id:fst
        [
          ("Sys_error", fun () -> raise (Sys_error "disk full"));
          ( "Stream.Error",
            fun () ->
              Bytesrw.Bytes.Stream.error Bytesrw.Bytes.Stream.limit_error 10 );
        ]
        (fun (_, raise_) ->
          match sink_failure raise_ with
          | Ok (), _ -> fail "the run did not fail"
          | Error e, why -> contains ~sub:why (error_text e));
      (* Milliseconds in int64 hold the seconds of [Int64.max_int / 1000] and
         [Int64.min_int / 1000], and no further. *)
      cases "a datetime of seconds writes while its milliseconds fit int64"
        ~id:Int64.to_string
        [ Int64.div Int64.max_int 1000L; Int64.div Int64.min_int 1000L ]
        (fun s -> equal (result unit string) (Ok ()) (seconds s));
      cases "a datetime of seconds past milliseconds in int64 is an Error"
        ~id:Int64.to_string
        [
          Int64.succ (Int64.div Int64.max_int 1000L);
          Int64.pred (Int64.div Int64.min_int 1000L);
        ]
        (fun s -> is_error (seconds s));
      test "raises if the query's columns are not the format's" (fun () ->
          let other = P.format (Schema.v [ ("a", any Type.int64) ]) in
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.encode") (fun () ->
              encode_with other (Query.of_table one_int));
          let d =
            Talon.v
              [
                ( "r",
                  Column.of_options (Type.duration Ns)
                    [| Some (Time.Span.ns 1) |] );
              ]
          in
          raises_match (Exn.invalid_arg ~sub:"Expr.cast Type.int64") (fun () ->
              encode_with
                (P.format (Schema.v [ ("r", any Type.int64) ]))
                (Query.of_table d)));
      test "raises for a format that leaves a column undeclared" (fun () ->
          let b = Rig.Buffer.of_string (file "decimals_int.parquet") in
          let f = Error.get_ok (P.sniff b) in
          let names = [ "d9_0"; "d9_2"; "d18_0"; "d18_6"; "d38_0"; "d38_10" ] in
          let t =
            Talon.v
              (List.map (fun c -> (c, Column.v Type.float64 [| 1. |])) names)
          in
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.encode") (fun () ->
              encode_with f (Query.of_table t)));
      test "raises for a required column of a sniffed format" (fun () ->
          let b =
            Rig.Buffer.of_string (file "delta_encoding_required_column.parquet")
          in
          let f = Error.get_ok (P.sniff b) in
          raises_match (Exn.invalid_arg ~sub:"Talon_parquet.encode: column")
            (fun () -> encode_with f (Query.of_source (P.source f b))));
      test "a sniffed format of optional columns writes the file again"
        (fun () ->
          let s = encode one_int in
          let b = Rig.Buffer.of_string s in
          let f = Error.get_ok (P.sniff b) in
          equal string s (encode_with f (Query.of_source (P.source f b))));
    ]

let () =
  exit
  @@ run "talon.parquet.writer"
       [
         group "Writing" [ level_bound ];
         group "Written types"
           [
             written_types;
             test "format writes lists, records and tensors as groups"
               written_groups;
             round_trip;
             nested_round_trip;
             tensor_round_trip;
             types_written;
             test "a float's bits read back, -0. and NaN included" float_bits;
             test "a duration's int64 ticks read back exactly" ticks_written;
           ];
         group "Statistics" [ statistics ];
         format_raises;
         encoding;
       ]
