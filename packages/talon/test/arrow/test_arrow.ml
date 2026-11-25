(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module A = Talon_arrow
module G = Talon_gen
module Writer = Bytesrw.Bytes.Writer

let error_text e = Format.asprintf "%a" Error.pp e
let support = Filename.concat "support"

(* The buffer of the file at [path]. *)
let opened path = Rig_disk.buffer (Result.get_ok (Rig_disk.of_file path))
let sniffed path = Error.get_ok (A.sniff (opened path))
let run_source s = Query.run (Query.of_source s)
let read path = Error.get_ok (run_source (Error.get_ok (A.file path)))

(* [read_as f path] reads the file at [path] as the format [f]. *)
let read_as f path = run_source (A.source f (opened path))

let type_text (Type.Any t) =
  match t with
  | Categorical _ -> "categorical"
  | _ -> Format.asprintf "%a" Type.pp t

(* [raised f] is the message of the [Invalid_argument] that [f ()] raises. *)
let raised f =
  match f () with
  | _ -> fail "the call did not raise"
  | exception Invalid_argument why -> why

let table_w =
  let pp ppf t =
    Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t)
  in
  Testable.make ~pp ~equal:Talon.equal

let encoded f q =
  let b = Buffer.create 4096 in
  Error.get_ok (A.encode f q (Writer.of_buffer b));
  Buffer.contents b

(* [batch_rows s] is the rows of each batch that a read of [s] yields. *)
let batch_rows s =
  List.rev
    (Error.get_ok
       (Query.fold (Query.of_source s) ~init:[] (fun acc b -> rows b :: acc)))

let read_bytes s =
  let b = Rig.Buffer.of_string s in
  run_source (A.source (Error.get_ok (A.sniff b)) b)

(* Rendering, as gen/arrow.py renders pyarrow's values: integers and ticks in
   decimal, floats as C99's %.17g, text and bytes in hexadecimal, null as n. *)

let is_float (type a) (t : a Type.t) =
  match t with Float16 | Float32 | Float64 -> true | _ -> false

let hex s =
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq s)))

let or_e s = if s = "" then "e" else s

(* [g17 x] is [x] as C99's [%.17g] writes it, as the generator's Python does,
   with an exponent of at least two digits. OCaml's [%.17g] is the C library's,
   which on Windows writes three ([1e-018]). *)
let g17 x =
  let s = Printf.sprintf "%.17g" x in
  match String.index_opt s 'e' with
  | None -> s
  | Some i ->
      let e = int_of_string (String.sub s (i + 1) (String.length s - i - 1)) in
      Printf.sprintf "%se%c%02d" (String.sub s 0 i)
        (if e < 0 then '-' else '+')
        (abs e)

let validity_at v n =
  match v with
  | None -> Array.make n true
  | Some v -> Array.init n (fun i -> Nx.item [ i ] (Nx.cast Nx.bool v))

let rec tokens (Type.Any t) c : string array =
  let n = Column.length c in
  match (t, Column.layout c) with
  | Ext { storage; _ }, _ -> tokens (Any storage) c
  | Tensor _, Fixed { validity; values = P x } ->
      let valid = validity_at validity n in
      let cells = if n = 0 then 0 else Nx.numel x / n in
      let flat = Nx.reshape [| -1 |] (Nx.cast Nx.float64 x) in
      let is_float =
        List.exists
          (fun (Nx.Dtype.Any d) -> Nx.Dtype.equal (Nx.dtype x) d)
          Nx.Dtype.[ Any Nx.float16; Any Nx.float32; Any Nx.float64 ]
      in
      Array.init n (fun i ->
          if not valid.(i) then "n"
          else
            "["
            ^ String.concat " "
                (List.init cells (fun j ->
                     let v = Nx.item [ (i * cells) + j ] flat in
                     if is_float then g17 v else Printf.sprintf "%.0f" v))
            ^ "]")
  | (String | Categorical _), _ ->
      Array.map
        (function None -> "n" | Some s -> or_e (hex s))
        (Column.options Kind.string c)
  | Binary, _ ->
      Array.map
        (function None -> "n" | Some b -> or_e (hex (b : Binary.t :> string)))
        (Column.options Kind.binary c)
  | _, Fixed { validity; values = P x } ->
      let valid = validity_at validity n in
      let as_float = Nx.cast Nx.float64 x and as_int = Nx.cast Nx.int64 x in
      Array.init n (fun i ->
          if not valid.(i) then "n"
          else if is_float t then g17 (Nx.item [ i ] as_float)
          else Int64.to_string (Nx.item [ i ] as_int))
  | List e, Varsize { validity; offsets; child } ->
      let valid = validity_at validity n in
      let items = tokens (Any e) child in
      Array.init n (fun i ->
          if not valid.(i) then "n"
          else
            let a = Int64.to_int (Nx.item [ i ] offsets)
            and b = Int64.to_int (Nx.item [ i + 1 ] offsets) in
            "["
            ^ String.concat " " (Array.to_list (Array.sub items a (b - a)))
            ^ "]")
  | Record fs, Children { validity; fields; _ } ->
      let valid = validity_at validity n in
      let cols = List.map2 (fun (_, ty) (_, c) -> tokens ty c) fs fields in
      Array.init n (fun i ->
          if not valid.(i) then "n"
          else "{" ^ String.concat " " (List.map (fun t -> t.(i)) cols) ^ "}")
  | _ -> invalid_arg "tokens"

let rendered c = String.concat " " (Array.to_list (tokens (Column.type_ c) c))

(* Goldens: one line per file and column, its fields split at '|'. *)
let golden path =
  List.map (String.split_on_char '|')
    (In_channel.with_open_bin path In_channel.input_lines)

let values_golden = lazy (golden "golden/values.txt")

(* [reads_as_pyarrow file] checks every column of [file] against pyarrow's
   reading of it: the talon type its Arrow type reads as, and its values. *)
let reads_as_pyarrow file =
  let t = read (support file) in
  let lines =
    List.filter_map
      (function
        | [ f; name; ty; values ] when f = file -> Some (name, ty, values)
        | _ -> None)
      (Lazy.force values_golden)
  in
  equal (list string)
    (List.map (fun (name, _, _) -> name) lines)
    (List.map fst (Schema.columns (schema t)));
  List.iter
    (fun (name, ty, values) ->
      let c = column t name in
      equal ~msg:name string ty (type_text (Column.type_ c));
      equal ~msg:name string values (rendered c))
    lines

(* Types *)

let types =
  group "types"
    [
      cases "scalars and times read as pyarrow reads them" ~id:Fun.id
        [
          "flat.arrow";
          "flat.arrows";
          "flat_lz4.arrow";
          "flat_zstd.arrow";
          "empty.arrow";
          "one_row.arrows";
          "views.arrows";
          "metadata.arrow";
        ]
        reads_as_pyarrow;
      test "a date64 that is not a midnight reads as the day that holds it"
        (fun () ->
          equal string "-1 0 -1 -2 n"
            (rendered (column (read (support "date64_floor.arrow")) "d")));
    ]

(* Undeclared decimals *)

(* The type each decimal column of the golden is declared as. *)
let declaration name ty : Type.any =
  match (name, ty) with
  | _, "int64" -> Any Type.int64
  | ("l" | "fl"), _ -> Any (Type.list Type.float64)
  | "s", _ ->
      Any (Type.record [ ("a", Any Type.float64); ("i", Any Type.int32) ])
  | "m", _ ->
      Any
        (Type.list
           (Type.record
              [ ("key", Any Type.string); ("value", Any Type.float64) ]))
  | "e", _ -> Any (Type.ext ~name:"raven.money" Type.float64)
  | _ -> Any Type.float64

let decimal_lines = golden "golden/decimals.txt"

(* [declared file] is the format of [file] with every decimal column declared as
   its first golden line says. *)
let declared file =
  let firsts =
    List.fold_left
      (fun acc line ->
        match line with
        | [ g; name; ty; _ ] when g = file && not (List.mem_assoc name acc) ->
            acc @ [ (name, ty) ]
        | _ -> acc)
      [] decimal_lines
  in
  List.fold_left
    (fun f (name, ty) -> A.with_type name (declaration name ty) f)
    (sniffed (support file))
    firsts

let undeclared_decimals =
  group "undeclared decimals"
    [
      test "sniff leaves a decimal undeclared, and pp_format says so" (fun () ->
          let text path =
            Format.asprintf "%a" A.pp_format (sniffed (support path))
          in
          contains ~sub:"d32_1_0 undeclared ← decimal32(1, 0)"
            (text "decimals.arrow");
          contains ~sub:"under_null undeclared ← decimal128(18, 0)"
            (text "decimals.arrow");
          contains ~sub:"l undeclared ← list<item: decimal128(9, 2)>"
            (text "decimals_nested.arrow"));
      cases "a declared decimal reads as pyarrow reads it"
        ~id:(fun line ->
          String.concat " " (List.filteri (fun i _ -> i < 3) line))
        decimal_lines
        (function
          | [ file; name; ty; values ] ->
              let f = A.with_type name (declaration name ty) (declared file) in
              let c = column (Error.get_ok (read_as f (support file))) name in
              equal string ty (type_text (Column.type_ c));
              equal string values (rendered c)
          | line -> failf "a golden line of %d fields" (List.length line));
      cases
        "source, file and encode raise for an undeclared column, naming it and \
         its types"
        ~id:fst
        [
          ( "source",
            fun () ->
              ignore
                (A.source
                   (sniffed (support "decimals.arrow"))
                   (opened (support "decimals.arrow"))) );
          ("file", fun () -> ignore (A.file (support "decimals.arrow")));
          ( "encode",
            fun () ->
              let q =
                Query.of_table (Talon.v [ ("x", Column.v Type.int32 [| 1 |]) ])
              in
              ignore
                (A.encode
                   (sniffed (support "decimals.arrow"))
                   q
                   (Writer.of_buffer (Buffer.create 16))) );
        ]
        (fun (fn, call) ->
          let why = raised call in
          starts_with ~prefix:("Talon_arrow." ^ fn) why;
          contains ~sub:{|"d32_1_0"|} why;
          contains ~sub:"float64" why;
          contains ~sub:"int64" why);
      test "an unscaled value past int64 fails its row, read as int64"
        (fun () ->
          let path = support "refused_decimal_range.arrow" in
          let f = A.with_type "d" (Any Type.int64) (sniffed path) in
          let e = error_text (require_error (read_as f path)) in
          contains ~sub:{|column "d": row 0: |} e);
      test "an int64 declaration reads only decimals of its format's scale"
        (fun () ->
          let scale_0 = support "refused_decimal_range.arrow"
          and scale_4 = support "decimals_other_scale.arrow"
          and plain = support "decimals_plain.arrow" in
          let as_ ty path = A.with_type "d" ty (sniffed path) in
          let refused f path =
            contains ~sub:"another scale"
              (error_text (require_error (read_as f path)))
          in
          refused (as_ (Any Type.int64) scale_0) scale_4;
          refused (as_ (Any Type.int64) scale_0) plain;
          refused (sniffed plain) scale_0;
          equal
            (array (option float_exact))
            [| Some 1.2345 |]
            (Column.options Kind.float
               (column
                  (Error.get_ok
                     (read_as (as_ (Any Type.float64) scale_0) scale_4))
                  "d")));
    ]

(* Dictionary-encoded columns *)

let dictionaries =
  group "dictionaries"
    [
      cases "dictionary-encoded columns read as pyarrow reads them" ~id:Fun.id
        [
          "dicts.arrow";
          "delta.arrows";
          "dict_replaced.arrows";
          "dict_extended.arrows";
          "dict_delta_dup.arrows";
          "dict_int_delta.arrows";
          "dict_binary.arrow";
          "dict_binary0.arrow";
          "dict_dup.arrow";
          "dict_float.arrow";
          "dict_int.arrow";
          "dict_list.arrow";
          "dict_nested.arrow";
          "dict_null.arrow";
          "dict_null_index.arrow";
          "dict_struct.arrow";
        ]
        reads_as_pyarrow;
      test "a dictionary after a batch of nulls reads its later rows" (fun () ->
          let c = column (read (support "late_dictionary.arrows")) "c" in
          equal string "int64" (type_text (Column.type_ c));
          equal string "n n 10 20" (rendered c));
      test "a fixed list of dictionary-encoded elements reads as their values"
        (fun () ->
          equal string "[0 1]"
            (rendered (column (read (support "tensor_dict.arrow")) "c")));
    ]

(* Lists, records and extensions *)

let nested =
  group "lists and records"
    [
      cases "lists and records read as pyarrow reads them" ~id:Fun.id
        [ "nested.arrow"; "nested_zstd.arrows" ]
        reads_as_pyarrow;
      test "a null element of a tensor fails at its row, and reads as a list"
        (fun () ->
          let path = support "refused_tensor_null.arrow" in
          let e = error_text (require_error (read_as (sniffed path) path)) in
          contains ~sub:{|column "c": row 0|} e;
          let f = A.with_type "c" (Any (Type.list Type.int32)) (sniffed path) in
          equal string "[1 n] [3 4]"
            (rendered (column (Error.get_ok (read_as f path)) "c")));
    ]

let extensions =
  group "extensions"
    [
      cases "extensions read as pyarrow reads them" ~id:Fun.id
        [ "extension.arrow" ] reads_as_pyarrow;
      cases "a tensor extension talon cannot read as a tensor reads as itself"
        ~id:fst
        [
          ( "short",
            {|ext[arrow.fixed_shape_tensor "{\"shape\":[3]}", tensor[int32, 2]]|}
          );
          ( "text",
            {|ext[arrow.fixed_shape_tensor "{\"shape\":[2]}", list[string]]|} );
        ]
        (fun (name, ty) ->
          equal string ty
            (type_text
               (Column.type_ (column (read (support "ext_shape.arrow")) name))));
    ]

(* Columns of no bytes *)

let no_bytes =
  group "columns of no bytes"
    [
      cases "zero-width columns read as pyarrow reads them" ~id:Fun.id
        [ "zero_width.arrow" ] reads_as_pyarrow;
      cases "more rows than the file has bytes take no memory" ~id:fst
        [
          ("zero_many.arrows", ("z", "tensor[uint8, 0]"));
          ("zero_ext.arrows", ("c", "tensor[uint8, 1×0]"));
        ]
        (fun (file, (name, ty)) ->
          let t = read (support file) in
          let c = column t name in
          equal int 16_777_216 (rows t);
          equal string ty (type_text (Column.type_ c));
          match Column.layout c with
          | Fixed { values = P x; _ } -> equal int 0 (Nx.numel x)
          | _ -> fail "a zero-width column is not laid out as a tensor");
    ]

(* Refusals *)

let refusals =
  group "refusals"
    [
      cases "fields talon cannot hold are refused, naming the column" ~id:fst
        [
          ("refused_null.arrow", "bad");
          ("refused_interval.arrow", "bad");
          ("refused_union.arrow", "bad");
          ("refused_ree.arrow", "bad");
          ("refused_list_view.arrow", "bad");
          ("refused_large_list_view.arrow", "bad");
          ("refused_nested.arrow", "outer.bad");
          ("refused_dict_decimal.arrow", "bad");
          ("refused_decimal_precision.arrows", "d");
          ("refused_dict_replaced.arrow", "c");
          ("refused_shared_id.arrows", "x");
          ("refused_names.arrow", "x");
          ("refused_fields.arrow", "s");
        ]
        (fun (file, column) ->
          match A.file (support file) with
          | Ok _ -> failf "%s was not refused" file
          | Error e -> contains ~sub:(Printf.sprintf "%S" column) (error_text e));
      cases
        "bytes no Arrow 0.15 or later little-endian V4 writer makes are refused"
        ~id:Fun.id
        [
          "refused_text.arrow";
          "refused_legacy.arrows";
          "refused_v3.arrows";
          "refused_big_endian.arrows";
        ] (fun file -> is_error (A.sniff (opened (support file))));
      test "a schema of repeated children is refused, in bounded work"
        (fun () -> is_error (A.file (support "refused_bomb.arrows")));
    ]

(* Formats *)

let with_type =
  group "with_type"
    [
      test "reads a numeric fixed-size list as a list" (fun () ->
          let path = support "nested.arrow" in
          let f =
            A.with_type "fixed_list" (Any (Type.list Type.int32)) (sniffed path)
          in
          let c = column (Error.get_ok (read_as f path)) "fixed_list" in
          equal string "list[int32]" (type_text (Column.type_ c)));
      cases "raises for an unknown column or a type it does not read as"
        ~id:(fun (file, name, _) -> file ^ " " ^ name)
        [
          ("nested.arrow", "nope", Type.Any Type.string);
          ("nested.arrow", "list", Any Type.string);
          ("zero_width.arrow", "binary0_lists", Any (Type.list Type.binary));
          ("zero_width.arrow", "ints0", Any (Type.list Type.int32));
        ]
        (fun (file, name, ty) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_arrow.with_type") (fun () ->
              A.with_type name ty (sniffed (support file))));
      cases "names what a decimal reads as"
        ~id:(fun (_, name, _, _) -> name)
        [
          ("decimals.arrow", "d128_38_0", Type.Any Type.int64, "float64");
          ("decimals.arrow", "d32_9_0", Any Type.int32, "float64 or int64");
          ( "decimals_nested.arrow",
            "s",
            Any (Type.record [ ("a", Any Type.float64); ("i", Any Type.int64) ]),
            "record[a float64 or int64, i int32]" );
        ]
        (fun (file, name, ty, reads) ->
          let why =
            raised (fun () -> A.with_type name ty (sniffed (support file)))
          in
          contains ~sub:(Printf.sprintf "%S" name) why;
          contains ~sub:("reads as " ^ reads ^ ",") why);
    ]

let pairs = list (pair string string)

let metadata =
  group "metadata"
    [
      test "is the file's custom metadata, in file order" (fun () ->
          let f = sniffed (support "metadata.arrow") in
          equal pairs ~msg:"schema"
            [ ("schema key", "schema value"); ("ARROW:x", "reserved") ]
            (A.metadata f);
          equal pairs ~msg:"field"
            [ ("unit", "kg"); ("origin", "lab:1") ]
            (A.metadata ~column:"x" f));
      test "a field's metadata holds no extension key" (fun () ->
          equal pairs []
            (A.metadata ~column:"kg" (sniffed (support "extension.arrow"))));
      test "with_metadata sets the metadata encode writes" (fun () ->
          let s = Schema.v [ ("x", Any Type.int32) ] in
          let f =
            A.with_metadata ~column:"x"
              [ ("k", "v") ]
              (A.with_metadata
                 [ ("s", "t") ]
                 (A.format ~metadata:[ ("old", "o") ] s))
          in
          let bytes =
            encoded f
              (Query.of_table (Talon.v [ ("x", Column.v Type.int32 [| 1 |]) ]))
          in
          let back = Error.get_ok (A.sniff (Rig.Buffer.of_string bytes)) in
          equal pairs [ ("s", "t") ] (A.metadata back);
          equal pairs [ ("k", "v") ] (A.metadata ~column:"x" back));
      test
        "metadata and with_metadata raise for an unknown column or a reserved \
         key" (fun () ->
          let f = sniffed (support "metadata.arrow") in
          let raises call =
            raises_match (Exn.invalid_arg ~sub:"Talon_arrow.") (fun () ->
                call f)
          in
          raises (A.metadata ~column:"nope");
          raises (A.with_metadata ~column:"nope" []);
          raises (A.with_metadata [ ("ARROW:extension:name", "x") ]);
          raises
            (A.with_metadata ~column:"x" [ ("ARROW:extension:metadata", "") ]));
    ]

let pp_format =
  group "pp_format"
    [
      cases "names the framing and each column's types"
        ~id:(fun (id, _, _) -> id)
        [
          ("file", A.File, "arrow file (2 columns)");
          ("stream", A.Stream, "arrow stream (2 columns)");
        ]
        (fun (_, framing, header) ->
          let q =
            Query.of_table
              (Talon.v
                 [
                   ("a", Column.of_options Type.int32 [| Some 1 |]);
                   ("b", Column.of_options Type.string [| None |]);
                 ])
          in
          let f = A.format ~framing (Query.schema q) in
          equal text
            (header ^ "\n  a int32 ← int32\n  b string ← large_utf8")
            (Format.asprintf "%a" A.pp_format
               (Error.get_ok (A.sniff (Rig.Buffer.of_string (encoded f q))))));
    ]

(* Reading *)

let one_column ty vs = Talon.v [ ("x", Column.of_options ty vs) ]
let bytes_of t = encoded (A.format (schema t)) (Query.of_table t)

(* The fixtures the corruption law edits: every file of support/. *)
let fixtures =
  lazy
    (Sys.readdir "support" |> Array.to_list |> List.sort compare
    |> List.filter (fun n -> not (Sys.is_directory (support n)))
    |> List.map (fun n ->
        In_channel.with_open_bin (support n) In_channel.input_all)
    |> Array.of_list)

let reading =
  group "reading"
    [
      test "a source reads one part per record batch" (fun () ->
          equal (list int) [ 10; 10; 10; 7 ]
            (batch_rows (Error.get_ok (A.file (support "written/file.arrow")))));
      cases "a read fails on a missing column or one of another type" ~id:fst
        [
          ("missing", Talon.v [ ("y", Column.v Type.int32 [| 1 |]) ]);
          ("another type", one_column Type.string [| Some "a" |]);
        ]
        (fun (_, t) ->
          let f = A.format (Schema.v [ ("x", Any Type.int32) ]) in
          is_error (run_source (A.source f (Rig.Buffer.of_string (bytes_of t)))));
      cases "a malformed record batch names its column and row" ~id:fst
        [
          ("refused_dict_index.arrow", {|column "c": row 1|});
          ("refused_date_range.arrow", {|column "d": row 0|});
          ("refused_zero_null.arrows", {|column "c": row 0|});
        ]
        (fun (file, place) ->
          let path = support file in
          contains ~sub:place
            (error_text (require_error (read_as (sniffed path) path))));
      test "a file is named arrow \"path\" and states its rows" (fun () ->
          let path = support "metadata.arrow" in
          contains
            ~sub:(Printf.sprintf "arrow %S (2 columns, 3 rows)" path)
            (Format.asprintf "%a" Query.pp
               (Query.of_source (Error.get_ok (A.file path)))));
      test "a file's errors name the file" (fun () ->
          let path = support "refused_dict_index.arrow" in
          starts_with ~prefix:path
            (error_text
               (require_error (run_source (Error.get_ok (A.file path))))));
      test "a missing file is an Error" (fun () ->
          is_error (A.file (Filename.concat (temp_dir ()) "missing.arrow")));
      (* Talon_held.settled counts the host's bytes after collection. *)
      test "a column shares the file's bytes unless its buffer is compressed"
        (fun () ->
          let n = 1 lsl 20 in
          let t =
            Talon.v [ ("x", Column.v Type.int64 (Array.init n Fun.id)) ]
          in
          (* Each compression writes a file of its own, since Windows refuses to
             truncate a file whose pages are mapped. *)
          let held compression =
            let path = temp_file ~suffix:".arrow" () in
            Out_channel.with_open_bin path (fun oc ->
                Error.get_ok
                  (A.encode
                     (A.format ~compression ~batch:n (schema t))
                     (Query.of_table t) (Writer.of_out_channel oc)));
            let column () = column (read path) "x" in
            ignore (column ());
            let before = Talon_held.settled () in
            let c = column () in
            let grown = Talon_held.settled () - before in
            equal int n (Column.length c);
            grown
          in
          less ~than:(8 * n) int (held A.Uncompressed);
          at_least ~than:(8 * n) int (held A.Zstd));
      prop ~count:1500 "corrupted bytes are an Error, never an exception"
        Gen.(
          pair nat
            (list
               (pair nat
                  (one_of
                     [
                       map (fun b -> `Byte b) (int_range 0 255);
                       map
                         (fun v -> `Long v)
                         (of_list
                            [
                              0L;
                              -1L;
                              Int64.max_int;
                              Int64.min_int;
                              0x3FFF_FFFF_FFFF_FFFFL;
                            ]);
                     ]))))
        (fun (i, edits) ->
          let files = Lazy.force fixtures in
          let original = files.(i mod Array.length files) in
          let len = String.length original in
          let b = Bytes.of_string original in
          let edit (at, e) =
            let at = at mod len in
            match e with
            | `Byte v -> Bytes.set b at (Char.chr v)
            | `Long v -> if at + 8 <= len then Bytes.set_int64_le b at v
          in
          List.iter edit edits;
          let buffer = Rig.Buffer.of_string (Bytes.to_string b) in
          let outcome =
            match A.sniff buffer with
            | Error _ -> `Refused
            | Ok f -> (
                match A.source f buffer with
                | exception Invalid_argument why ->
                    contains ~sub:"with_type" why;
                    `Undeclared
                | s -> (
                    match run_source s with Ok _ -> `Read | Error _ -> `Failed))
          in
          cover "sniff accepts the bytes" (outcome <> `Refused);
          cover "the read fails" (outcome = `Failed));
    ]

(* Writing *)

let written_golden = lazy (golden "golden/written.txt")

(* The Arrow type each column of Arrow_corpus writes as, by the interface's
   "Written types", as pyarrow prints it. *)
let written_types =
  [
    ("bool", "bool");
    ("int8", "int8");
    ("int16", "int16");
    ("int32", "int32");
    ("int64", "int64");
    ("uint8", "uint8");
    ("uint16", "uint16");
    ("uint32", "uint32");
    ("uint64", "uint64");
    ("float16", "halffloat");
    ("float32", "float");
    ("float64", "double");
    ("string", "large_string");
    ("binary", "large_binary");
    ("category", "dictionary<values=large_string, indices=int32, ordered=0>");
    ("date", "date32[day]");
    ("clock_s", "time32[s]");
    ("clock_ms", "time32[ms]");
    ("clock_us", "time64[us]");
    ("clock_ns", "time64[ns]");
    ("duration_ms", "duration[ms]");
    ("duration_ns", "duration[ns]");
    ("instant_us", "timestamp[us, tz=UTC]");
    ("wall_ms", "timestamp[ms]");
    ("list", "large_list<item: int32>");
    ("record", "struct<a: int64, b: large_string>");
    ( "tensor",
      "extension<arrow.fixed_shape_tensor[value_type=int32, shape=[3]]>" );
    ("kg", "extension<raven.mass<Kg>>");
  ]

let writing =
  group "writing"
    [
      cases "encode writes the committed bytes"
        ~id:(fun it -> it.Arrow_corpus.name)
        Arrow_corpus.items
        (fun it ->
          let committed =
            In_channel.with_open_bin
              (support ("written/" ^ it.name))
              In_channel.input_all
          in
          equal string committed (Arrow_corpus.bytes it));
      cases "each type writes as the Arrow type the interface states, nullable"
        ~id:fst written_types (fun (name, arrow) ->
          match
            List.find_opt
              (function
                | [ "file.arrow"; n; _; _; _; _ ] -> n = name | _ -> false)
              (Lazy.force written_golden)
          with
          | Some [ _; _; _; written; nullable; _ ] ->
              equal string arrow written;
              equal string "nullable" nullable
          | _ -> failf "no golden line for %s" name);
      cases "pyarrow reads encode's files as the table written"
        ~id:(fun it -> it.Arrow_corpus.name)
        Arrow_corpus.items
        (fun it ->
          let lines =
            List.filter_map
              (function
                | [ f; name; ty; _; _; values ] when f = it.name ->
                    Some (name, (ty, values))
                | _ -> None)
              (Lazy.force written_golden)
          in
          let ours =
            List.map
              (fun (name, _) ->
                let c = column it.table name in
                (name, (type_text (Column.type_ c), rendered c)))
              (Schema.columns (schema it.table))
          in
          equal (list (pair string (pair string string))) ours lines);
    ]

(* Format *)

let format =
  group "format"
    [
      test
        "raises for a batch below one, a reserved key or a tensor Arrow lacks"
        (fun () ->
          let x ty = Schema.v [ ("x", ty) ] in
          let raises f =
            raises_match (Exn.invalid_arg ~sub:"Talon_arrow.format") f
          in
          raises (fun () -> A.format ~batch:0 (x (Any Type.int32)));
          raises (fun () -> A.format ~batch:(-1) (x (Any Type.int32)));
          raises (fun () ->
              A.format
                ~metadata:[ ("ARROW:extension:name", "x") ]
                (x (Any Type.int32)));
          raises (fun () -> A.format (x (Any (Type.tensor Nx.bool [| 2 |]))));
          raises (fun () ->
              A.format
                (x (Any (Type.ext ~name:"m" (Type.tensor Nx.float32 [| 2 |]))))));
      test "a record batch holds 65,536 rows by default" (fun () ->
          let t = one_column Type.int8 (Array.make 65_537 (Some 1)) in
          let b = Rig.Buffer.of_string (bytes_of t) in
          equal (list int) [ 65_536; 1 ]
            (batch_rows (A.source (Error.get_ok (A.sniff b)) b)));
    ]

(* Encode *)

let compressions = [| A.Uncompressed; Lz4_frame; Zstd |]
let framings = [| A.File; Stream |]

let arrow_number dt =
  List.exists
    (fun (Nx.Dtype.Any d) -> Nx.Dtype.equal d dt)
    Nx.Dtype.
      [
        Any Nx.int8;
        Any Nx.int16;
        Any Nx.int32;
        Any Nx.int64;
        Any Nx.uint8;
        Any Nx.uint16;
        Any Nx.uint32;
        Any Nx.uint64;
        Any Nx.float16;
        Any Nx.float32;
        Any Nx.float64;
      ]

(* The types [format] writes: every one but a tensor whose dtype is not an Arrow
   integer or float, and a tensor under an extension. *)
let rec writable (Type.Any t) =
  match t with
  | Tensor (dt, _) -> arrow_number dt
  | Ext { storage = Tensor _; _ } -> false
  | Ext { storage; _ } -> writable (Any storage)
  | List e -> writable (Any e)
  | Record fields -> List.for_all (fun (_, t) -> writable t) fields
  | _ -> true

let case =
  let open Gen in
  let* (G.Sample (ty, vs) as s) = G.sample in
  let t = Talon.v [ ("a", Column.of_options ty vs) ] in
  let* t' = G.split t in
  let* batch = int_range 1 50 in
  let* framing = int_range 0 1 in
  let+ codec = int_range 0 2 in
  (s, t, t', batch, framings.(framing), compressions.(codec))

let nested_type (G.Sample (t, _)) =
  match t with List _ | Record _ -> true | _ -> false

let to_bytes ~batch ~framing ~compression t =
  encoded (A.format ~batch ~framing ~compression (schema t)) (Query.of_table t)

let encoding =
  group "encode"
    [
      prop ~count:300 "a table reads back, nulls at every level" case
        (fun ((G.Sample (ty, _) as s), t, _, batch, framing, compression) ->
          assume (writable (Any ty));
          let n = rows t in
          cover "nested" (nested_type s);
          cover "no row" (n = 0);
          cover "one row" (n = 1);
          cover "a batch boundary" (batch <= n);
          cover "a null" (Column.null_count (column t "a") > 0);
          cover "a stream" (framing = A.Stream);
          cover "lz4" (compression = A.Lz4_frame);
          cover "zstd" (compression = A.Zstd);
          Law.round_trip table_w string
            (to_bytes ~batch ~framing ~compression)
            (fun s -> Error.get_ok (read_bytes s))
            t);
      prop ~count:300 "bytes depend only on rows and format, never on batches"
        case (fun (G.Sample (ty, _), t, t', batch, framing, compression) ->
          assume (writable (Any ty));
          cover "several batches" (List.length (batches t') > 1);
          equal string
            (to_bytes ~batch ~framing ~compression t)
            (to_bytes ~batch ~framing ~compression t'));
      test "a tensor writes as Arrow's canonical extension and reads back"
        (fun () ->
          let x =
            Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let t = Talon.v [ ("t", Column.of_tensor x) ] in
          equal table_w t
            (Error.get_ok
               (read_bytes
                  (to_bytes ~batch:1 ~framing:File ~compression:A.Uncompressed t))));
      test "encode ends the writer with eod" (fun () ->
          let ended = ref "no eod" in
          let w =
            Writer.make (fun s ->
                if Bytesrw.Bytes.Slice.is_eod s then ended := "eod")
          in
          let t = one_column Type.int32 [| Some 1 |] in
          Error.get_ok (A.encode (A.format (schema t)) (Query.of_table t) w);
          equal string "eod" !ended);
      test "a failing run, or a writer that raises, is an Error" (fun () ->
          let path = support "refused_dict_index.arrow" in
          let failing =
            Query.of_source (A.source (sniffed path) (opened path))
          in
          let f = A.format (Query.schema failing) in
          is_error ~msg:"a failing run"
            (A.encode f failing (Writer.of_buffer (Buffer.create 16)));
          let t = one_column Type.int32 [| Some 1 |] in
          is_error ~msg:"a writer that raises"
            (A.encode
               (A.format (schema t))
               (Query.of_table t)
               (Writer.make (fun _ -> raise (Sys_error "disk full")))));
      cases "raises when the query does not write as the format" ~id:fst
        [
          ( "another column",
            ( A.format (Schema.v [ ("x", Any Type.int32) ]),
              Query.of_table (Talon.v [ ("y", Column.v Type.int32 [| 1 |]) ]) )
          );
          ( "a utf8 column of a sniffed file",
            let path = support "metadata.arrow" in
            (sniffed path, Query.of_table (read path)) );
          ( "a declared decimal",
            let path = support "refused_decimal_range.arrow" in
            ( A.with_type "d" (Any Type.int64) (sniffed path),
              Query.of_table (Talon.v [ ("d", Column.v Type.int64 [| 1 |]) ]) )
          );
        ]
        (fun (_, (f, q)) ->
          raises_match (Exn.invalid_arg ~sub:"Talon_arrow.encode") (fun () ->
              A.encode f q (Writer.of_buffer (Buffer.create 16))));
    ]

let () =
  exit
  @@ run "talon.arrow"
       [
         types;
         undeclared_decimals;
         dictionaries;
         nested;
         extensions;
         no_bytes;
         refusals;
         with_type;
         metadata;
         pp_format;
         reading;
         writing;
         format;
         encoding;
       ]
