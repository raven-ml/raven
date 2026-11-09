(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module P = Talon_parquet
module G = Talon_gen

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

(* The written types *)

let any t = Type.Any t

let laid_out ty layout =
  match Column.of_layout ty layout with
  | Ok c -> c
  | Error (_, why) -> failwith why

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
    (List.map
       (fun (name, _) -> (name, normal (Talon.column t name)))
       columns)

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

let round_trip =
  prop ~count:300 "a table of written types reads back" case
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
  prop ~count:300 "other types read as their written type, in the same file"
    case
    (fun (G.Sample (ty, _), t, _, row_group, compression) ->
      assume (writable (any ty));
      equal string
        (encode ~compression ~row_group (normalized t))
        (encode ~compression ~row_group t))

let deterministic =
  prop ~count:300 "a file's bytes depend on its rows and settings only" case
    (fun (G.Sample (ty, _), t, t', row_group, compression) ->
      assume (writable (any ty));
      cover "nested" (nested (any ty));
      cover "several batches" (List.length (batches t') > 1);
      cover "row group cuts a batch" (row_group < rows t);
      equal string (encode ~compression ~row_group t)
        (encode ~compression ~row_group t'))

(* Nested tables

   The nulls a sample cannot hold: a list with a null element, and nulls at
   each level of lists and records. *)

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
      Law.round_trip table_w string
        (encode ~compression ~row_group)
        decode t)

(* Float bits *)

let float_bits () =
  let values =
    Array.concat
      (List.init 20 (fun _ ->
           [| 1.5; -0.; 0.; Float.nan; Float.infinity; Float.neg_infinity |]))
  in
  let column = Column.of_options Type.float64 (Array.map Option.some values) in
  let t = Talon.v [ ("x", column) ] in
  let back =
    Column.to_tensor Nx.float64 (Talon.column (decode (encode t)) "x")
  in
  let expected = Nx.create Nx.float64 [| 120 |] values in
  equal
    (array int64)
    (Array.map Int64.bits_of_float (Nx.to_array expected))
    (Array.map Int64.bits_of_float (Nx.to_array back))

let filtered s p =
  let b = Rig.Buffer.of_string s in
  let f = Error.get_ok (P.sniff b) in
  Error.get_ok (Query.run (Query.filter p (Query.of_source (P.source f b))))

(* Pruning by statistics keeps every row a filter passes. *)
let pruning () =
  let n = 100 in
  let ints = Array.init n (fun i -> Some (if i < 50 then i else 1000 + i)) in
  let long i = String.make 80 'x' ^ string_of_int i in
  let texts =
    Array.init n (fun i -> if i mod 10 = 0 then None else Some (long i))
  in
  let floats =
    Array.init n (fun i -> Some (if i = 90 then Float.nan else float_of_int i))
  in
  let t =
    Talon.v
      [
        ("i", Column.of_options Type.int32 ints);
        ("t", Column.of_options Type.string texts);
        ("x", Column.of_options Type.float64 floats);
      ]
  in
  let file = encode ~row_group:7 t in
  let same p =
    equal table_w (Error.get_ok (Query.run (Query.filter p (Query.of_table t))))
      (filtered file p)
  in
  let open Expr in
  same (Col.int "i" > int 1040);
  same (Col.int "i" = int 3);
  same (Col.string "t" = string (long 3));
  same (Col.string "t" > string (long 96));
  same (Col.float "x" > float 80.);
  same (Col.float "x" <> float 80.)

let invalid () =
  let t = Talon.v [ ("a", Column.of_options Type.int32 [| Some 1 |]) ] in
  let s = Query.schema (Query.of_table t) in
  raises (Invalid_argument "Talon_parquet.format: the schema has no columns")
    (fun () -> P.format (Schema.v []));
  raises (Invalid_argument "Talon_parquet.format: row_group must be positive")
    (fun () -> P.format ~row_group:0 s);
  let other = P.format (Schema.v [ ("a", any Type.int64) ]) in
  raises_match
    (function
      | Invalid_argument m ->
          String.starts_with ~prefix:"Talon_parquet.encode" m
      | _ -> false)
    (fun () ->
      ignore
        (P.encode other (Query.of_table t)
           (Bytesrw.Bytes.Writer.of_buffer (Buffer.create 16))))

(* A datetime of seconds that does not fit milliseconds in [int64]. *)
let overflow () =
  let huge = Nx.create Nx.int64 [| 1 |] [| Int64.max_int |] in
  let layout : Column.layout = Fixed { validity = None; values = P huge } in
  let c = Result.get_ok (Column.of_layout (any (Type.wall S)) layout) in
  let q = Query.of_table (Talon.v [ ("x", c) ]) in
  let w = Bytesrw.Bytes.Writer.of_buffer (Buffer.create 16) in
  equal (result unit string)
    (Error "a datetime in seconds does not fit Parquet's milliseconds")
    (P.encode (P.format (Query.schema q)) q w
    |> Result.map_error (Format.asprintf "%a" Error.pp))

let format_text () =
  let schema =
    Schema.v
      [
        ("a", any Type.int8);
        ("when", any (Type.instant S));
        ("l", any (Type.list Type.string));
      ]
  in
  equal string
    "parquet (3 columns)\n\
    \  a int8                 ← optional int32 (INTEGER(8,true))\n\
    \  when datetime[ms, UTC] ← optional int64 (TIMESTAMP(MILLIS,true))\n\
    \  l list[string]         ← optional group (LIST) { repeated group list { \
     optional binary element (STRING); } }"
    (Format.asprintf "%a" P.pp_format (P.format schema))

(* A sink that fails is an error of the call. *)
let sink_failure () =
  let t = Talon.v [ ("a", Column.of_options Type.int32 [| Some 1 |]) ] in
  let q = Query.of_table t in
  let w = Bytesrw.Bytes.Writer.make (fun _ -> raise (Sys_error "disk full")) in
  equal (result unit string) (Error "disk full")
    (P.encode (P.format (Query.schema q)) q w
    |> Result.map_error (Format.asprintf "%a" Error.pp))

let file name =
  In_channel.with_open_bin ("support/" ^ name) In_channel.input_all

(* A format from a file writes only what it prints. *)
let sniffed_format () =
  let t = Talon.v [ ("a", Column.of_options Type.int32 [| Some 1 |]) ] in
  let s = encode t in
  let again b =
    let f = Error.get_ok (P.sniff b) in
    let q = Query.of_source (P.source f b) in
    encode_with f q
  in
  equal string s (again (Rig.Buffer.of_string s));
  raises_match
    (function
      | Invalid_argument m ->
          String.starts_with ~prefix:"Talon_parquet.encode: column" m
      | _ -> false)
    (fun () ->
      again
        (Rig.Buffer.of_string (file "delta_encoding_required_column.parquet")))

(* A value far larger than the page target takes a page of its own, past
   small values that do not push it over the limit. *)
let skewed_pages () =
  let long = String.make (2 * 1024 * 1024) 'x' in
  let rows =
    Array.init 400 (fun i ->
        Some (if i mod 100 = 7 then long ^ string_of_int i else "s"))
  in
  let t = Talon.v [ ("t", Column.of_options Type.string rows) ] in
  equal table_w t (decode (encode t))

let large_pages (item : Written_corpus.item) =
  test (item.name ^ ": reads back") (fun () ->
      let back =
        decode
          (encode ~compression:item.compression ~row_group:item.row_group
             item.table)
      in
      equal table_w item.table back)

(* The corpus *)

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

(* [render c] are the rows of [c], flat or nested, as golden/written.txt writes
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
      let bytes =
        match Column.layout child with
        | Fixed { values = Nx.P x; _ } -> Nx.to_array (Nx.cast Nx.uint8 x)
        | _ -> assert false
      in
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
          ^ String.concat ","
              (List.map (fun (f, r) -> f ^ "=" ^ r.(i)) parts)
          ^ "}")
  | Tensor (_, shape), Fixed { values = Nx.P x; _ } ->
      let row = Nx.reshape [| -1 |] (Nx.cast Nx.int64 x) in
      let leaves = Nx.to_array row in
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
  | Ext { storage; _ }, l -> (
      match Column.of_layout (Any storage) l with
      | Ok c -> render c
      | Error (_, why) -> failwith why)
  | _ -> failwith (Format.asprintf "render: %a" Type.pp t)

let support name = "support/written/" ^ name ^ ".parquet"
let golden =
  In_channel.with_open_bin "golden/written.txt" In_channel.input_lines

let rendered c =
  let text = String.concat " " (Array.to_list (render c)) in
  if String.length text > 4096 then "md5:" ^ Digest.to_hex (Digest.string text)
  else text

let corpus (item : Written_corpus.item) =
  (if item.large then []
   else
     [
       test (item.name ^ ": the file is this writer's") (fun () ->
           let file =
             In_channel.with_open_bin (support item.name) In_channel.input_all
           in
           equal string file
             (encode ~compression:item.compression ~row_group:item.row_group
                item.table));
     ])
  @ [
    test (item.name ^ ": pyarrow read its values") (fun () ->
        let t = item.table in
        let prefix = item.name ^ " values " in
        let golden =
          List.filter (String.starts_with ~prefix) golden
          |> List.map (fun l ->
                 String.sub l (String.length prefix)
                   (String.length l - String.length prefix))
        in
        let lines =
          List.map
            (fun (name, _) ->
              name ^ " " ^ rendered (Talon.column t name))
            (Schema.columns (Talon.schema t))
        in
        equal (list string) golden lines);
  ]

(* A list adds two definition levels and an optional leaf one: 32,767 lists of
   [int32] reach 65,535, the most a 16-bit level holds, and a record around them
   one more. *)
let level_bound =
  test "format refuses a definition level past 16 bits, naming the column"
  @@ fun () ->
  let rec lists k (Type.Any t as ty) =
    if k = 0 then ty else lists (k - 1) (Type.Any (Type.list t))
  in
  let deepest = lists 32_767 (Type.Any Type.int32) in
  ignore (P.format (Schema.v [ ("a", deepest) ]));
  let past = Type.Any (Type.record [ ("x", deepest) ]) in
  raises_match
    (Exn.invalid_arg ~sub:"\"a\"")
    (fun () -> P.format (Schema.v [ ("a", past) ]))

(* Durations: the file records no unit, so format refuses one and names the
   cast that writes its ticks. *)
let duration_refused =
  test "format refuses a duration, naming the expression that writes its ticks"
  @@ fun () ->
  raises
    (Invalid_argument
       {|Talon_parquet.format: column "d" is duration[ms], which Parquet records with no unit, so its ticks would lose their meaning. Write Expr.cast Type.int64 (Col.span "d") first, its ticks of its own unit; a cast to another duration before it chooses theirs.|})
    (fun () -> ignore (P.format (Schema.v [ ("d", any (Type.duration Ms)) ])));
  raises
    (Invalid_argument
       {|Talon_parquet.format: column "l" is list[duration[us]], which Parquet records with no unit, so its ticks would lose their meaning. Cast its durations to int64 with Expr.cast first.|})
    (fun () ->
      ignore
        (P.format (Schema.v [ ("l", any (Type.list (Type.duration Us))) ])));
  raises
    (Invalid_argument
       {|Talon_parquet.format: column "e" is ext[raven.span, duration[s]], which Parquet records with no unit, so its ticks would lose their meaning. An extension does not cast: rebuild the column over its storage with Column.of_layout, then cast that.|})
    (fun () ->
      ignore
        (P.format
           (Schema.v
              [ ("e", any (Type.ext ~name:"raven.span" (Type.duration S))) ])));
  let t =
    Talon.v [ ("r", Column.of_options (Type.duration Ns) [| Some (Time.Span.ns 1) |]) ]
  in
  let q = Query.of_table t in
  raises_match
    (Exn.invalid_arg ~sub:"Expr.cast Type.int64")
    (fun () -> ignore (encode_with (P.format (Schema.v [ ("r", any Type.int64) ])) q))

(* The expression the refusal names writes the ticks exactly, past 2^53. *)
let ticks_written =
  test "the ticks a duration casts to write and read back exactly" @@ fun () ->
  let xs = [| Int64.min_int; Int64.max_int; 0x20_0000_0000_0001L; -1L |] in
  let values = Nx.create Nx.int64 [| Array.length xs |] xs in
  let c =
    Result.get_ok
      (Column.of_layout (Any (Type.duration Ms))
         (Fixed { validity = None; values = P values }))
  in
  let q =
    Query.select
      Expr.[ "d" := cast Type.int64 (Col.span "d") ]
      (Query.of_table (Talon.v [ ("d", c) ]))
  in
  let t = decode (encode_with (P.format (Query.schema q)) q) in
  let got = Column.to_tensor Nx.int64 (Talon.column t "d") in
  equal (array int64) xs (Nx.to_array got)

let () =
  exit
  @@ run "talon.parquet.writer"
    [
      group "durations" [ duration_refused; ticks_written ];
      group "round trips"
        [ round_trip; types_written; deterministic; nested_round_trip ];
      group "values"
        [
          test "float bits survive" float_bits;
          test "a filter over statistics keeps its rows" pruning;
          test "invalid arguments" invalid;
          test "a datetime that does not fit is an error" overflow;
          test "the format states the file's types" format_text;
          test "a failing sink is an error" sink_failure;
          test "a sniffed format writes what it prints" sniffed_format;
          test "a page takes a value past the target whole" skewed_pages;
          level_bound;
        ];
      group "pages"
        (List.map large_pages
           (List.filter
              (fun (i : Written_corpus.item) ->
                i.large && not (String.equal i.name "pages_nested"))
              Written_corpus.items));
      group "corpus" (List.concat_map corpus Written_corpus.items);
    ]
