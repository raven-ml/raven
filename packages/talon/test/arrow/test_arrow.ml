(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module A = Talon_arrow
module G = Talon_gen

(* Rendering, as the generator renders pyarrow's values. *)

let is_float (type a) (t : a Type.t) =
  match t with Float16 | Float32 | Float64 -> true | _ -> false

let hex s = String.concat "" (List.map (fun c -> Printf.sprintf "%02x" (Char.code c)) (List.of_seq (String.to_seq s)))
let or_e s = if s = "" then "e" else s

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
      let is_float = Nx.Dtype.equal (Nx.dtype x) Nx.float32 || Nx.Dtype.equal (Nx.dtype x) Nx.float64 || Nx.Dtype.equal (Nx.dtype x) Nx.float16 in
      Array.init n (fun i ->
          if not valid.(i) then "n"
          else
            "["
            ^ String.concat " "
                (List.init cells (fun j ->
                     let v = Nx.item [ (i * cells) + j ] flat in
                     if is_float then Printf.sprintf "%.17g" v
                     else Printf.sprintf "%.0f" v))
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
          else if is_float t then Printf.sprintf "%.17g" (Nx.item [ i ] as_float)
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

let read path =
  Error.get_ok (Query.run (Query.of_source (Error.get_ok (A.file path))))

let lines path = In_channel.with_open_bin path In_channel.input_lines

let type_text (Type.Any t) =
  match t with Categorical _ -> "categorical" | _ -> Format.asprintf "%a" Type.pp t

let check_file file columns =
  let path = Filename.concat "support" file in
  test (file ^ " reads as pyarrow does") @@ fun () ->
  let table = read path in
  let line (name, ty, values) =
    let c = column table name in
    equal ~msg:(file ^ " " ^ name) string ty (type_text (Column.type_ c));
    let got =
      String.concat " " (Array.to_list (tokens (Column.type_ c) c))
    in
    equal ~msg:(file ^ " " ^ name) string values got
  in
  List.iter line columns

let golden path =
  let by_file = Hashtbl.create 16 in
  List.iter
    (fun l ->
      match String.split_on_char '|' l with
      | [ file; name; ty; values ] ->
          let cur =
            Option.value (Hashtbl.find_opt by_file file) ~default:[]
          in
          Hashtbl.replace by_file file ((name, ty, values) :: cur)
      | _ -> failwith l)
    (lines path);
  Hashtbl.fold (fun f cols acc -> (f, List.rev cols) :: acc) by_file []
  |> List.sort compare

let corpus =
  List.map (fun (f, cols) -> check_file f cols) (golden "golden/values.txt")

(* talon's own files, as pyarrow reads them and as they stay. *)
let written =
  let read_back (f, cols) = check_file ("written/" ^ f) cols in
  List.map read_back (golden "golden/written.txt")

let bytes_stay =
  cases "encode writes the committed bytes" ~id:(fun it -> it.Arrow_corpus.name)
    Arrow_corpus.items (fun it ->
      let path = Filename.concat "support/written" it.name in
      let committed = In_channel.with_open_bin path In_channel.input_all in
      equal string committed (Arrow_corpus.bytes it))

(* Refusals *)

let refused =
  cases "fields talon cannot hold are refused, naming the column" ~id:fst
    [
      ("refused_null.arrow", "bad");
      ("refused_decimal.arrow", "bad");
      ("refused_interval.arrow", "bad");
      ("refused_union.arrow", "bad");
      ("refused_ree.arrow", "bad");
      ("refused_list_view.arrow", "bad");
      ("refused_nested.arrow", "outer.bad");
      ("refused_dict_replaced.arrow", "c");
      ("refused_shared_id.arrows", "x");
    ]
    (fun (name, column) ->
      match A.file (Filename.concat "support" name) with
      | Ok _ -> failf "%s was not refused" name
      | Error e ->
          contains ~sub:(Printf.sprintf "%S" column)
            (Format.asprintf "%a" Error.pp e))

let corrupted =
  let fixture name =
    In_channel.with_open_bin (Filename.concat "support" name) In_channel.input_all
  in
  let small n = Filename.extension n <> ".txt" && Sys.is_directory ("support/" ^ n) |> not in
  let files =
    Array.of_list
      (List.filter_map
         (fun n -> if small n then Some (fixture n) else None)
         (Sys.readdir "support" |> Array.to_list |> List.sort compare))
  in
  let extreme = [ 0L; -1L; Int64.max_int; Int64.min_int; 0x3FFF_FFFF_FFFF_FFFFL ] in
  let edit =
    Gen.(pair nat (one_of [ map (fun b -> `Byte b) (int_range 0 255);
                           map (fun i -> `Long (List.nth extreme i)) (int_range 0 4) ]))
  in
  let case = Gen.(pair (int_range 0 (Array.length files - 1)) (list edit)) in
  prop ~count:1500 "corrupted bytes are an error, never a crash" case
    (fun (i, edits) ->
      let original = files.(i) in
      let len = String.length original in
      let b = Bytes.of_string original in
      let apply (at, e) =
        let at = at mod len in
        match e with
        | `Byte v -> Bytes.set b at (Char.chr v)
        | `Long v -> if at + 8 <= len then Bytes.set_int64_le b at v
      in
      List.iter apply edits;
      let buffer = Rig.Buffer.of_string (Bytes.to_string b) in
      match A.sniff buffer with
      | Error _ -> ()
      | Ok f -> ignore (Query.run (Query.of_source (A.source f buffer))))

let bomb =
  test "a schema of repeated children is refused, in bounded work" @@ fun () ->
  is_error ~msg:"the bomb" (A.file "support/refused_bomb.arrows")

(* Zero-width and by-value dictionary columns *)

let zero_many =
  test "a zero-width column of more rows than bytes takes no memory" @@ fun () ->
  let t = read "support/zero_many.arrows" in
  equal int 16_777_216 (rows t);
  let c = column t "z" in
  equal string "tensor[uint8, 0]" (type_text (Column.type_ c));
  match Column.layout c with
  | Fixed { values = P x; _ } -> equal int 0 (Nx.numel x)
  | _ -> fail "a zero-width column is not laid out as a tensor"

let tensor_dict =
  test "a fixed list of dictionary-encoded elements reads as their values"
  @@ fun () ->
  let c = column (read "support/tensor_dict.arrow") "c" in
  equal string "[0 1]" (String.concat " " (Array.to_list (tokens (Column.type_ c) c)))

let zero_width_type =
  test "a zero-width column reads as a tensor, and not as a list" @@ fun () ->
  let f =
    Error.get_ok
      (A.sniff (Result.get_ok (Rig_disk.of_file "support/zero_width.arrow")))
  in
  raises_match ~msg:"elements that take no bytes"
    (Exn.invalid_arg ~sub:"not as")
    (fun () -> A.with_type "binary0_lists" (Any (Type.list Type.binary)) f);
  raises_match ~msg:"no element"
    (Exn.invalid_arg ~sub:"not as")
    (fun () -> A.with_type "ints0" (Any (Type.list Type.int32)) f)

let zero_ext =
  test "zero-width extension elements of many rows take no memory" @@ fun () ->
  let t = read "support/zero_ext.arrows" in
  equal int 16_777_216 (rows t);
  let c = column t "c" in
  equal string "tensor[uint8, 1\195\1510]" (type_text (Column.type_ c));
  match Column.layout c with
  | Fixed { values = P x; _ } -> equal int 0 (Nx.numel x)
  | _ -> fail "a zero-width column is not laid out as a tensor"

let zero_null =
  test "a null element of a zero-width cell fails with its row" @@ fun () ->
  let b = Result.get_ok (Rig_disk.of_file "support/refused_zero_null.arrows") in
  let f = Error.get_ok (A.sniff b) in
  let e = Result.get_error (Query.run (Query.of_source (A.source f b))) in
  contains ~sub:"row 0" (Format.asprintf "%a" Error.pp e)

let late_dictionary =
  test "a dictionary after a batch of nulls reads its later rows" @@ fun () ->
  let c = column (read "support/late_dictionary.arrows") "c" in
  equal string "int64" (type_text (Column.type_ c));
  equal (array string) [| "n"; "n"; "10"; "20" |] (tokens (Column.type_ c) c)

let dict_index =
  test "an index past its dictionary fails with its row" @@ fun () ->
  let b = Result.get_ok (Rig_disk.of_file "support/refused_dict_index.arrow") in
  let f = Error.get_ok (A.sniff b) in
  let e = Result.get_error (Query.run (Query.of_source (A.source f b))) in
  contains ~sub:"row 1" (Format.asprintf "%a" Error.pp e)

let date64_floor =
  test "a date64 reads as the day that holds it" @@ fun () ->
  let c = column (read "support/date64_floor.arrow") "d" in
  let days = tokens (Column.type_ c) c in
  equal (array string) [| "-1"; "0"; "-1"; "-2"; "n" |] days

let date_range =
  test "a date past int32 days fails with its row" @@ fun () ->
  let b = Result.get_ok (Rig_disk.of_file "support/refused_date_range.arrow") in
  let f = Error.get_ok (A.sniff b) in
  let e = Result.get_error (Query.run (Query.of_source (A.source f b))) in
  contains ~sub:"row 0" (Format.asprintf "%a" Error.pp e)

let not_arrow =
  test "bytes that are neither file nor stream are an error" @@ fun () ->
  is_error ~msg:"garbage"
    (A.sniff (Rig.Buffer.of_string "not an arrow file, not at all"))

(* Formats *)

let table_w =
  let pp ppf t = Format.fprintf ppf "%a: %d rows" Schema.pp (schema t) (rows t) in
  Testable.make ~pp ~equal:Talon.equal

let encode_with f q =
  let b = Buffer.create 4096 in
  Error.get_ok (A.encode f q (Bytesrw.Bytes.Writer.of_buffer b));
  Buffer.contents b

let run_bytes s =
  let b = Rig.Buffer.of_string s in
  Error.get_ok (Query.run (Query.of_source (A.source (Error.get_ok (A.sniff b)) b)))

let flat_format =
  test "pp_format states each column's Arrow type" @@ fun () ->
  let f = Error.get_ok (A.sniff (Rig.Buffer.of_string (encode_with
      (A.format (Schema.v [ ("a", Any Type.int32); ("b", Any Type.string) ]))
      (Query.of_table (Talon.v [ ("a", Column.of_options Type.int32 [| Some 1 |]);
                                 ("b", Column.of_options Type.string [| None |]) ]))))) in
  equal text "arrow file (2 columns)\n  a int32 \226\134\144 int32\n  b string \226\134\144 large_utf8"
    (Format.asprintf "%a" A.pp_format f)

let metadata_kept =
  test "custom metadata reads and writes" @@ fun () ->
  let f = Error.get_ok (Result.bind (Rig_disk.of_file "support/metadata.arrow" |> Result.map_error Error.v) A.sniff) in
  equal (list (pair string string)) ~msg:"schema"
    [ ("schema key", "schema value"); ("ARROW:x", "reserved") ]
    (A.metadata f);
  equal (list (pair string string)) ~msg:"field"
    [ ("unit", "kg"); ("origin", "lab:1") ]
    (A.metadata ~column:"x" f);
  let s = Schema.v [ ("x", Any Type.int32) ] in
  let f' =
    A.with_metadata ~column:"x" [ ("k", "v") ]
      (A.with_metadata [ ("s", "t") ] (A.format s))
  in
  let bytes =
    encode_with f' (Query.of_table (Talon.v [ ("x", Column.v Type.int32 [| 1 |]) ]))
  in
  let back = Error.get_ok (A.sniff (Rig.Buffer.of_string bytes)) in
  equal (list (pair string string)) [ ("s", "t") ] (A.metadata back);
  equal (list (pair string string)) [ ("k", "v") ] (A.metadata ~column:"x" back);
  raises_match ~msg:"reserved key"
    (Exn.invalid_arg ~sub:"ARROW:extension")
    (fun () -> A.with_metadata [ ("ARROW:extension:name", "x") ] f')

let tensor_as_list =
  test "with_type reads a fixed-size list as a list, and nothing else" @@ fun () ->
  let b = Result.get_ok (Rig_disk.of_file "support/nested.arrow") in
  let f = Error.get_ok (A.sniff b) in
  let f = A.with_type "fixed_list" (Any (Type.list Type.int32)) f in
  let t = Error.get_ok (Query.run (Query.of_source (A.source f b))) in
  equal string "list[int32]" (type_text (Column.type_ (column t "fixed_list")));
  raises_match ~msg:"an int does not read as text"
    (Exn.invalid_arg ~sub:"not as")
    (fun () -> A.with_type "list" (Any Type.string) f);
  raises_match ~msg:"unknown column" (Exn.invalid_arg ~sub:"no column")
    (fun () -> A.with_type "nope" (Any Type.string) f)

(* Round trips *)

let compressions = [| A.Uncompressed; Lz4_frame; Zstd |]
let framings = [| A.File; Stream |]

let writable (G.Sample (ty, _)) =
  match A.format (Schema.v [ ("a", Any ty) ]) with
  | _ -> true
  | exception Invalid_argument _ -> false

let case =
  let open Gen in
  let* (G.Sample (ty, vs) as s) = G.sample in
  let t = Talon.v [ ("a", Column.of_options ty vs) ] in
  let* t' = G.split t in
  let* batch = int_range 1 50 in
  let* framing = int_range 0 1 in
  let+ codec = int_range 0 2 in
  (s, t, t', batch, framings.(framing), compressions.(codec))

let nested (G.Sample (t, _)) =
  match t with List _ | Record _ -> true | _ -> false

let to_bytes ~batch ~framing ~compression t =
  let q = Query.of_table t in
  encode_with (A.format ~batch ~framing ~compression (Query.schema q)) q

let round_trip =
  prop ~count:300 "a table reads back, nulls at every level" case
    (fun ((G.Sample (ty, _) as s), t, _, batch, framing, compression) ->
      assume (writable s);
      let n = Talon.rows t in
      cover "nested" (nested s);
      cover "no row" (n = 0);
      cover "one row" (n = 1);
      cover "a batch boundary" (batch <= n);
      cover "a null" (Column.null_count (column t "a") > 0);
      let f = A.format ~batch ~framing ~compression (Schema.v [ ("a", Any ty) ]) in
      equal table_w t (run_bytes (encode_with f (Query.of_table t)));
      Law.round_trip table_w string
        (fun t -> to_bytes ~batch ~framing ~compression t)
        run_bytes t)

let deterministic =
  prop ~count:300 "bytes depend only on rows and format, never on batches"
    case (fun ((G.Sample _ as s), t, t', batch, framing, compression) ->
      assume (writable s);
      cover "several batches" (List.length (batches t') > 1);
      equal string
        (to_bytes ~batch ~framing ~compression t)
        (to_bytes ~batch ~framing ~compression t'))

let tensor_round_trip =
  test "a tensor round-trips over Arrow's canonical extension" @@ fun () ->
  let x = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t = Talon.v [ ("t", Column.of_tensor x) ] in
  let back = run_bytes (to_bytes ~batch:1 ~framing:File ~compression:A.Uncompressed t) in
  equal table_w t back

let () =
  exit @@ run "talon.arrow"
    [
      group "corpus" corpus;
      group "written" (bytes_stay :: written);
      group "refusals" [ refused; not_arrow; corrupted; bomb; date_range; date64_floor ];
      group "by value"
        [ zero_many; zero_ext; zero_null; zero_width_type; tensor_dict; dict_index;
          late_dictionary ];
      group "formats" [ flat_format; metadata_kept; tensor_as_list ];
      group "round trips" [ round_trip; deterministic; tensor_round_trip ];
    ]
