(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap
module P = Talon_parquet

(* Fixtures *)

(* [path name] is the fixture [name], written with [/], which every system
   opens, so that messages name it alike. *)
let path name = "support/" ^ name

let buffer name =
  match Rig_disk.of_file (path name) with
  | Ok f -> Rig_disk.buffer f
  | Error why -> failwith why

let file_bytes name = In_channel.with_open_bin (path name) In_channel.input_all
let sniff name = Error.get_ok (P.sniff (buffer name))
let format_text f = Format.asprintf "%a" P.pp_format f
let error_text e = Format.asprintf "%a" Error.pp e
let outcome = function Ok _ -> "ok" | Error e -> error_text e
let any t = Type.Any t
let table_w = Testable.make ~pp:Talon.pp ~equal:Talon.equal
let table q = Error.get_ok (Query.run q)

(* [read_from f b column] is the column [column] of the file [b], read as [f],
   through a query of its source. *)
let read_from f b column =
  Query.of_source (P.source f b)
  |> Query.select Expr.[ keep (Sel.names [ column ]) ]
  |> Query.run
  |> Result.map (fun t -> Talon.column t column)

let read f name column = read_from f (buffer name) column

(* [format_type f column] is the type [f] reads [column] as, from its printed
   form, since a format has no accessor. *)
let format_type f column =
  let s =
    Format.asprintf "%a" Schema.pp (Schema.v [ (column, any Type.bool) ])
  in
  let prefix = "  " ^ String.sub s 0 (String.length s - 4) in
  match
    List.find_opt
      (String.starts_with ~prefix)
      (String.split_on_char '\n' (format_text f))
  with
  | None -> failf "no column %S in the format" column
  | Some l ->
      let rest =
        String.sub l (String.length prefix)
          (String.length l - String.length prefix)
      in
      String.trim (String.sub rest 0 (String.index rest '\xe2'))

(* Rendering, as gen/fixtures.py and gen/nested.py render pyarrow's values *)

let quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      if c >= ' ' && c <= '~' && c <> '"' && c <> '\\' then Buffer.add_char b c
      else Printf.bprintf b "\\x%02x" (Char.code c))
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let masked validity cells =
  match validity with
  | None -> cells
  | Some v ->
      Array.map2 (fun ok s -> if ok then s else "null") (Nx.to_array v) cells

let byte_cells offsets child =
  let o = Nx.to_array offsets in
  let d = Nx.to_array (Column.to_tensor Nx.uint8 child) in
  let cell i =
    let first = Int64.to_int o.(i) and last = Int64.to_int o.(i + 1) in
    quote (String.init (last - first) (fun k -> Char.chr d.(first + k)))
  in
  Array.init (Array.length o - 1) cell

(* The nx dtype that stores a talon type, by the type's printed form. *)
let storage ty =
  let starts p = String.starts_with ~prefix:p ty in
  if starts "clock" || starts "datetime" then "int64"
  else if ty = "date" then "int32"
  else if ty = "bool" then "bit"
  else ty

(* [flat_cells ty c] is the rows of the flat column [c] of type [ty], as
   golden/values.txt holds them: floats by their bits. *)
let flat_cells ty c =
  let map f a = Array.map f (Nx.to_array a) in
  match Column.layout c with
  | Fixed { validity; values = Nx.P x } ->
      let dtype = Format.asprintf "%a" Nx.pp_dtype (Nx.dtype x) in
      equal string ~msg:"storage dtype" (storage ty) dtype;
      masked validity
        (match ty with
        | "bool" ->
            map
              (fun b -> if b = 1 then "true" else "false")
              (Nx.cast Nx.uint8 x)
        | "uint64" -> map (Printf.sprintf "%Lu") (Nx.bitcast Nx.int64 x)
        | "float16" -> map (Printf.sprintf "0x%04x") (Nx.bitcast Nx.uint16 x)
        | "float32" -> map (Printf.sprintf "0x%08lx") (Nx.bitcast Nx.int32 x)
        | "float64" -> map (Printf.sprintf "0x%016Lx") (Nx.bitcast Nx.int64 x)
        | _ -> map Int64.to_string (Nx.cast Nx.int64 x))
  | Varsize { validity; offsets; child } ->
      masked validity (byte_cells offsets child)
  | Children _ -> fail "a record column"

let scalars (Nx.P x) =
  let map f a = Array.map f (Nx.to_array a) in
  match Format.asprintf "%a" Nx.pp_dtype (Nx.dtype x) with
  | "bit" ->
      map (fun b -> if b = 1 then "true" else "false") (Nx.cast Nx.uint8 x)
  | "float64" -> map (Printf.sprintf "0x%016Lx") (Nx.bitcast Nx.int64 x)
  | _ -> map Int64.to_string (Nx.cast Nx.int64 x)

(* [nested_lists shape xs] is the cell [xs] of [shape], in row-major order, as
   the nested lists it reads from. *)
let rec nested_lists shape xs =
  match shape with
  | [] -> xs.(0)
  | d :: rest ->
      let n = if d = 0 then 0 else Array.length xs / d in
      let item k = nested_lists rest (Array.sub xs (k * n) n) in
      "[" ^ String.concat ", " (List.init d item) ^ "]"

(* [cells c] is the rows of [c], flat or nested, as golden/nested.txt holds
   them, a tensor as the nested lists it reads from. *)
let rec cells c =
  match (Column.type_ c, Column.layout c) with
  | Type.Any (Tensor (_, shape)), Fixed { validity; values } ->
      let xs = scalars values in
      let size = Iarray.fold_left ( * ) 1 shape in
      let row i =
        nested_lists (Iarray.to_list shape) (Array.sub xs (i * size) size)
      in
      masked validity (Array.init (Column.length c) row)
  | _, Fixed { validity; values } -> masked validity (scalars values)
  | Type.Any (String | Binary), Varsize { validity; offsets; child } ->
      masked validity (byte_cells offsets child)
  | _, Varsize { validity; offsets; child } ->
      let o = Array.map Int64.to_int (Nx.to_array offsets) in
      let items = cells child in
      let row i =
        let its = Array.sub items o.(i) (o.(i + 1) - o.(i)) in
        "[" ^ String.concat ", " (Array.to_list its) ^ "]"
      in
      masked validity (Array.init (Array.length o - 1) row)
  | _, Children { validity; length; fields } ->
      let fs = List.map (fun (n, c) -> (n, cells c)) fields in
      let row i =
        let field (n, cs) = n ^ ": " ^ cs.(i) in
        "{" ^ String.concat ", " (List.map field fs) ^ "}"
      in
      masked validity (Array.init length row)

let cells_text c = String.concat " " (Array.to_list (cells c))

(* Agreement with pyarrow

   A line of golden/values.txt or golden/overrides.txt is [file group "column"
   "type" rows md5 values], or [file group "column" error]: one row group of a
   column as pyarrow reads it, its type the talon type of pyarrow's, float64 for
   a decimal. *)

let golden name =
  In_channel.with_open_text name In_channel.input_lines
  |> List.map (fun l -> (List.hd (String.split_on_char ' ' l), l))

let values_lines = golden "golden/values.txt"
let override_lines = golden "golden/overrides.txt"
let files lines = List.sort_uniq String.compare (List.map fst lines)

(* [columns lines file] is each column of [file] with its lines [(group, line,
   rest)], [rest] being the line past the column's name. *)
let columns lines file =
  let parse l =
    Scanf.sscanf l "%_s %d %S %[^\n]" (fun g c rest -> (c, (g, l, rest)))
  in
  let ls =
    List.filter_map
      (fun (f, l) -> if f = file then Some (parse l) else None)
      lines
  in
  List.sort_uniq String.compare (List.map fst ls)
  |> List.map (fun c ->
      (c, List.filter_map (fun (c', g) -> if c = c' then Some g else None) ls))

let line file g column ty values =
  let values = Array.to_list values in
  let md5 = Digest.to_hex (Digest.string (String.concat " " values)) in
  let first = List.filteri (fun i _ -> i < 8) values in
  String.trim
    (Printf.sprintf "%s %d %s %s %d %s %s" file g (quote column) (quote ty)
       (List.length values) md5 (String.concat " " first))

let golden_type (_, _, rest) = Scanf.sscanf rest "%S" Fun.id

(* [agrees file f (column, groups)] reads [column] of [file] as [f], and checks
   each row group's line, or that the read fails at the row group whose line is
   [error]. *)
let agrees file f (column, groups) =
  match List.find_opt (fun (_, _, rest) -> rest = "error") groups with
  | Some (g, _, _) -> (
      match read f file column with
      | Ok _ -> failf "%s: read a column that fails" column
      | Error e ->
          let place = List.hd (String.split_on_char ':' (error_text e)) in
          equal string ~msg:column (Printf.sprintf "row group %d" g) place)
  | None -> (
      let ty = golden_type (List.hd groups) in
      match read f file column with
      | Error e -> failf "%s: %s" column (error_text e)
      | Ok c ->
          let cs = flat_cells ty c in
          let cut first (g, l, rest) =
            let rows = Scanf.sscanf rest "%_S %d" Fun.id in
            equal string l (line file g column ty (Array.sub cs first rows));
            first + rows
          in
          equal ~msg:column int (Array.length cs) (List.fold_left cut 0 groups))

(* [declared file] is [file]'s format, its decimals declared as float64. *)
let declared file =
  let declare f (c, _) =
    if format_type f c = "undeclared" then P.with_type c (any Type.float64) f
    else f
  in
  List.fold_left declare (sniff file) (columns values_lines file)

(* Each column reads as its sniffed type, a decimal as float64 once declared. *)
let reads_as_pyarrow file =
  let sniffed = sniff file and f = declared file in
  let check ((column, groups) as c) =
    let _, _, rest = List.hd groups in
    if rest <> "error" then
      equal string ~msg:column
        (golden_type (List.hd groups))
        (match format_type sniffed column with
        | "undeclared" -> "float64"
        | ty -> ty);
    agrees file f c
  in
  List.iter check (columns values_lines file)

let alternative_reads_as_pyarrow file =
  let ty = function
    | "string" -> any Type.string
    | "datetime[us]" -> any (Type.wall Us)
    | "int64" -> any Type.int64
    | ty -> failf "no talon type for %s" ty
  in
  let f = declared file in
  let check ((column, groups) as c) =
    agrees file (P.with_type column (ty (golden_type (List.hd groups))) f) c
  in
  List.iter check (columns override_lines file)

(* Synthesized files

   Files written with a Thrift compact encoder, so that their fields can hold
   any value, including those no writer produces: sizes at the limits of their
   types, and integers of another wire type than their field's. *)

type thrift =
  | I32 of int
  | I64 of int
  | Bin of string
  | List of thrift list
  | Struct of (int * thrift) list

let wire = function
  | I32 _ -> 5
  | I64 _ -> 6
  | Bin _ -> 8
  | List _ -> 9
  | Struct _ -> 12

let rec varint b n =
  if Int64.logand n (-128L) = 0L then Buffer.add_uint8 b (Int64.to_int n)
  else begin
    Buffer.add_uint8 b (Int64.to_int (Int64.logand n 0x7FL) lor 0x80);
    varint b (Int64.shift_right_logical n 7)
  end

let zigzag n =
  let n = Int64.of_int n in
  Int64.logxor (Int64.shift_left n 1) (Int64.shift_right n 63)

let rec put b = function
  | I32 n | I64 n -> varint b (zigzag n)
  | Bin s ->
      varint b (Int64.of_int (String.length s));
      Buffer.add_string b s
  | List vs ->
      let n = List.length vs in
      let ty = match vs with [] -> 12 | v :: _ -> wire v in
      if n < 15 then Buffer.add_uint8 b ((n lsl 4) lor ty)
      else begin
        Buffer.add_uint8 b (0xF0 lor ty);
        varint b (Int64.of_int n)
      end;
      List.iter (put b) vs
  | Struct fields ->
      (* A field's id is its delta from the last in a byte's high bits, or, past
         15, a zigzag varint after the type. *)
      let field last (id, v) =
        let delta = id - last in
        if delta >= 1 && delta <= 15 then
          Buffer.add_uint8 b ((delta lsl 4) lor wire v)
        else begin
          Buffer.add_uint8 b (wire v);
          varint b (zigzag id)
        end;
        put b v;
        id
      in
      ignore (List.fold_left field 0 (List.sort compare fields));
      Buffer.add_uint8 b 0

let thrift v =
  let b = Buffer.create 64 in
  put b v;
  Buffer.contents b

(* [set fields changes] is [fields] with the values of [changes]. *)
let set fields changes =
  List.map
    (fun (id, v) -> (id, Option.value ~default:v (List.assoc_opt id changes)))
    fields
  @ List.filter (fun (id, _) -> not (List.mem_assoc id fields)) changes

let int32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.of_int v)) vs;
  Bytes.to_string b

let prefixed s = int32s [ String.length s ] ^ s

(* [page ?header ?data_page values data] is a [DATA_PAGE] of [values] values
   whose bytes are [data], PLAIN with RLE levels, with the changes [header] and
   [data_page] to its header's fields. *)
let page ?(header = []) ?(data_page = []) values data =
  let size = I32 (String.length data) in
  let data_page = set [ (1, I32 values); (2, I32 0); (3, I32 3) ] data_page in
  thrift
    (Struct
       (set [ (1, I32 0); (2, size); (3, size); (5, Struct data_page) ] header))
  ^ data

(* [dictionary_page words] is a [DICTIONARY_PAGE] of the PLAIN byte arrays
   [words]. *)
let dictionary_page words =
  let plain = String.concat "" (List.map prefixed words) in
  let size = I32 (String.length plain) in
  thrift
    (Struct
       [
         (1, I32 2);
         (2, size);
         (3, size);
         (7, Struct [ (1, I32 (List.length words)); (2, I32 0) ]);
       ])
  ^ plain

(* [leaf ?ty ?dictionary ?changes path ~values pages] is a column chunk of
   [values] values of the physical type [ty], INT32 unless given, at [path],
   whose bytes are [pages], [dictionary] bytes of them a dictionary page, with
   the [changes] to its ColumnMetaData. It is the chunk's fields as a function
   of the position of its first byte, and its pages. *)
let leaf ?(ty = 1) ?dictionary ?(changes = []) path ~values pages =
  let fields pos =
    let size = I64 (String.length pages) in
    let offsets =
      match dictionary with
      | None -> [ (9, I64 pos) ]
      | Some d -> [ (9, I64 (pos + d)); (11, I64 pos) ]
    in
    let meta =
      [
        (1, I32 ty);
        (3, List (List.map (fun s -> Bin s) path));
        (4, I32 0);
        (5, I64 values);
        (6, size);
        (7, size);
      ]
      @ offsets
    in
    [ (3, Struct (set meta changes)) ]
  in
  (fields, pages)

(* [file ?columns ~elements groups] is the Parquet file of the schema
   [elements], with [columns] fields below its root, and of a row group [(rows,
   chunks)] for each of [groups]. *)
let file ?(columns = 1) ~elements groups =
  let pos = ref 4 and pages = Buffer.create 256 in
  let chunk (fields, p) =
    let c = Struct (fields !pos) in
    pos := !pos + String.length p;
    Buffer.add_string pages p;
    c
  in
  let group (rows, chunks) =
    let cs = List.map chunk chunks in
    Struct [ (1, List cs); (3, I64 rows) ]
  in
  let gs = List.map group groups in
  let rows = List.fold_left (fun n (r, _) -> n + r) 0 groups in
  let root = Struct [ (4, Bin "schema"); (5, I32 columns) ] in
  let footer =
    thrift
      (Struct [ (2, List (root :: elements)); (3, I64 rows); (4, List gs) ])
  in
  Rig.Buffer.of_string
    ("PAR1" ^ Buffer.contents pages ^ footer
    ^ int32s [ String.length footer ]
    ^ "PAR1")

let required = 0
let optional = 1
let repeated = 2

let field ?(ty = 1) ?converted repetition name =
  Struct
    ([ (1, I32 ty); (3, I32 repetition); (4, Bin name) ]
    @ Option.fold ~none:[] ~some:(fun c -> [ (6, I32 c) ]) converted)

let node ?converted repetition name children =
  Struct
    ((3, I32 repetition) :: (4, Bin name) :: (5, I32 children)
    :: Option.fold ~none:[] ~some:(fun c -> [ (6, I32 c) ]) converted)

let list_group = node ~converted:3

(* [parquet ?column ?chunk ?dictionary ~rows pages] is a file of one row group
   of [rows] rows of the column [x], required INT32 unless [column] changes its
   schema element, whose chunk is [pages], with the changes [chunk] to its
   metadata. *)
let parquet ?(column = []) ?(chunk = []) ?dictionary ~rows pages =
  let x = set [ (1, I32 1); (3, I32 required); (4, Bin "x") ] column in
  let ty = match List.assoc 1 x with I32 t -> t | _ -> 1 in
  file ~elements:[ Struct x ]
    [
      (rows, [ leaf ~ty ?dictionary ~changes:chunk [ "x" ] ~values:rows pages ]);
    ]

let read_x b =
  match P.sniff b with Error e -> Error e | Ok f -> read_from f b "x"

let i32_max = 0x7FFF_FFFF

(* The RLE and bit-packing hybrid of Parquet's Encodings.md: [Repeated (k, v)]
   is [v] [k] times, [Packed vs] the values [vs] bit-packed in groups of eight.
   [hybrid w runs] is their bytes at [w] bits: a run's header is a ULEB128
   varint, [k lsl 1] before a repeated value in its [⌈w / 8⌉] little-endian
   bytes ("rle-header := varint-encode( (rle-run-len) << 1)"), [g lsl 1 lor 1]
   before [g] groups of eight values packed from each byte's least significant
   bit, the last group padded with zeros. *)
type run = Repeated of int * int | Packed of int list

let hybrid w runs =
  let b = Buffer.create 64 in
  let rec varint v =
    if v < 0x80 then Buffer.add_char b (Char.chr v)
    else (
      Buffer.add_char b (Char.chr (v land 0x7f lor 0x80));
      varint (v lsr 7))
  in
  let run = function
    | Repeated (k, v) ->
        varint (k lsl 1);
        for i = 0 to ((w + 7) / 8) - 1 do
          Buffer.add_char b (Char.chr ((v lsr (8 * i)) land 0xff))
        done
    | Packed vs ->
        let groups = (List.length vs + 7) / 8 in
        varint ((groups lsl 1) lor 1);
        let bits = Bytes.make (groups * w) '\000' in
        List.iteri
          (fun i v ->
            for j = 0 to w - 1 do
              if (v lsr j) land 1 = 1 then begin
                let at = (i * w) + j in
                let c = Char.code (Bytes.get bits (at / 8)) in
                Bytes.set bits (at / 8) (Char.chr (c lor (1 lsl (at mod 8))))
              end
            done)
          vs;
        Buffer.add_bytes b bits
  in
  List.iter run runs;
  Buffer.contents b

(* [rle runs] is the repeated runs [(k, v)] of values of at most 8 bits. *)
let rle runs = hybrid 8 (List.map (fun (k, v) -> Repeated (k, v)) runs)

(* [runs_of choices vs] is [vs] cut into runs, each choice picking the next: an
   even one, the run of the equal values that start it; an odd one, the next 8,
   16 or 24 values packed, or the rest, which the last group pads. *)
let rec runs_of choices vs =
  match (vs, choices) with
  | [], _ -> []
  | _ :: _, c :: cs when c mod 2 = 1 ->
      let k = 8 * (1 + (c / 2 mod 3)) in
      Packed (List.filteri (fun i _ -> i < k) vs)
      :: runs_of cs (List.filteri (fun i _ -> i >= k) vs)
  | v :: _, cs ->
      let rec stretch n = function
        | x :: xs when x = v -> stretch (n + 1) xs
        | xs -> (n, xs)
      in
      let n, rest = stretch 0 vs in
      Repeated (n, v) :: runs_of (match cs with [] -> [] | _ :: cs -> cs) rest

let packed_bits w = function Packed vs -> List.length vs * w | Repeated _ -> 0

(* [unpacks_eight w runs] is [true] iff a packed run of eight values or more
   leaves eight bytes from its eighth value's first byte inside [runs]' bytes,
   where the decoder loads eight values at once. *)
let unpacks_eight w runs =
  let len = String.length (hybrid w runs) in
  let rec go before = function
    | [] -> false
    | r :: rest ->
        let upto = before @ [ r ] in
        let fast =
          match r with
          | Packed vs when List.compare_length_with vs 8 >= 0 ->
              let payload = (List.length vs + 7) / 8 * w in
              String.length (hybrid w upto) - payload + (7 * w / 8) + 8 <= len
          | _ -> false
        in
        fast || go upto rest
  in
  w > 0 && go [] runs

(* [cover_runs w runs] covers the packed runs the decoder takes apart. *)
let cover_runs w runs =
  cover "a packed run of more than eight bytes"
    (List.exists (fun r -> packed_bits w r > 64) runs);
  cover "a packed run of under eight bytes"
    (List.exists (fun r -> packed_bits w r > 0 && packed_bits w r < 64) runs)

(* [level_page ?data_page ~rep ~def values data] is a page of [values] values
   whose repetition and definition levels are the RLE hybrid bytes [rep] and
   [def]. *)
let level_page ?(data_page = [ (4, I32 3) ]) ~rep ~def values data =
  page ~data_page values (prefixed rep ^ prefixed def ^ data)

(* The nested column [x]: an optional list of optional int32s. *)
let x_elements =
  [
    list_group optional "x" 1; node repeated "list" 1; field optional "element";
  ]

let x_path = [ "x"; "list"; "element" ]

let x_file ?(rows = 2) ?changes ~values pages =
  file ~elements:x_elements [ (rows, [ leaf ?changes x_path ~values pages ]) ]

(* [cuts n ks] is the pages [(a, b)] of [n] entries cut at [ks mod (n + 1)]. *)
let cuts n ks =
  let ks = List.sort_uniq compare (List.map (fun k -> k mod (n + 1)) ks) in
  let bounds = List.sort_uniq compare ((0 :: ks) @ [ n ]) in
  let rec pages = function
    | a :: (b :: _ as rest) -> (a, b) :: pages rest
    | _ -> []
  in
  pages bounds

let sub l (a, b) = List.filteri (fun i _ -> a <= i && i < b) l

(* Types *)

let reads_as_pyarrow_test =
  cases "each column reads as pyarrow reads it" ~id:Fun.id (files values_lines)
    reads_as_pyarrow

(* The types TP lists for each Parquet type, as pp_format prints them. *)
let sniffed_types =
  cases "sniff gives each Parquet type the talon type it reads as" ~id:fst
    [
      ( "types.parquet",
        {|parquet (30 columns)
  bool bool                     ← optional boolean
  i8 int8                       ← optional int32 (INTEGER(8,true))
  i16 int16                     ← optional int32 (INTEGER(16,true))
  i32 int32                     ← optional int32
  i64 int64                     ← optional int64
  u8 uint8                      ← optional int32 (INTEGER(8,false))
  u16 uint16                    ← optional int32 (INTEGER(16,false))
  u32 uint32                    ← optional int32 (INTEGER(32,false))
  u64 uint64                    ← optional int64 (INTEGER(64,false))
  f16 float16                   ← optional fixed_len_byte_array(2) (FLOAT16)
  f32 float32                   ← optional float
  f64 float64                   ← optional double
  d9 undeclared                 ← optional fixed_len_byte_array(4) (DECIMAL(9,2))
  d18 undeclared                ← optional fixed_len_byte_array(8) (DECIMAL(18,6))
  date date                     ← optional int32 (DATE)
  clock_ms clock[ms]            ← optional int32 (TIME(MILLIS,false))
  clock_us clock[us]            ← optional int64 (TIME(MICROS,false))
  clock_ns clock[ns]            ← optional int64 (TIME(NANOS,false))
  ts_ms datetime[ms]            ← optional int64 (TIMESTAMP(MILLIS,false))
  ts_us_utc datetime[us, UTC]   ← optional int64 (TIMESTAMP(MICROS,true))
  ts_us_paris datetime[us, UTC] ← optional int64 (TIMESTAMP(MICROS,true))
  ts_ns datetime[ns]            ← optional int64 (TIMESTAMP(NANOS,false))
  string string                 ← optional binary (STRING)
  binary binary                 ← optional binary
  fsb binary                    ← optional fixed_len_byte_array(5)
  uuid binary                   ← optional fixed_len_byte_array(16) (UUID)
  all_null int32                ← optional int32
  no_null int64                 ← optional int64
  req_i32 int32                 ← required int32
  req_string string             ← required binary (STRING)|}
      );
      ( "alltypes_plain.parquet",
        {|parquet (11 columns)
  id int32                   ← optional int32
  bool_col bool              ← optional boolean
  tinyint_col int32          ← optional int32
  smallint_col int32         ← optional int32
  int_col int32              ← optional int32
  bigint_col int64           ← optional int64
  float_col float32          ← optional float
  double_col float64         ← optional double
  date_string_col binary     ← optional binary
  string_col binary          ← optional binary
  timestamp_col datetime[ns] ← optional int96|}
      );
      ( "byte_stream_split_extended.gzip.parquet",
        {|parquet (14 columns)
  float16_plain float16                ← optional fixed_len_byte_array(2) (FLOAT16)
  float16_byte_stream_split float16    ← optional fixed_len_byte_array(2) (FLOAT16)
  float_plain float32                  ← optional float
  float_byte_stream_split float32      ← optional float
  double_plain float64                 ← optional double
  double_byte_stream_split float64     ← optional double
  int32_plain int32                    ← optional int32
  int32_byte_stream_split int32        ← optional int32
  int64_plain int64                    ← optional int64
  int64_byte_stream_split int64        ← optional int64
  flba5_plain binary                   ← optional fixed_len_byte_array(5)
  flba5_byte_stream_split binary       ← optional fixed_len_byte_array(5)
  decimal_plain undeclared             ← optional fixed_len_byte_array(4) (DECIMAL(7,3))
  decimal_byte_stream_split undeclared ← optional fixed_len_byte_array(4) (DECIMAL(7,3))|}
      );
      ( "decimals_int.parquet",
        {|parquet (6 columns)
  d9_0 undeclared   ← optional int32 (DECIMAL(9,0))
  d9_2 undeclared   ← optional int32 (DECIMAL(9,2))
  d18_0 undeclared  ← optional int64 (DECIMAL(18,0))
  d18_6 undeclared  ← optional int64 (DECIMAL(18,6))
  d38_0 undeclared  ← optional fixed_len_byte_array(16) (DECIMAL(38,0))
  d38_10 undeclared ← optional fixed_len_byte_array(16) (DECIMAL(38,10))|}
      );
      ( "decimals_bytes.parquet",
        {|parquet (6 columns)
  d9_0 undeclared   ← optional fixed_len_byte_array(4) (DECIMAL(9,0))
  d9_2 undeclared   ← optional fixed_len_byte_array(4) (DECIMAL(9,2))
  d18_0 undeclared  ← optional fixed_len_byte_array(8) (DECIMAL(18,0))
  d18_6 undeclared  ← optional fixed_len_byte_array(8) (DECIMAL(18,6))
  d38_0 undeclared  ← optional fixed_len_byte_array(16) (DECIMAL(38,0))
  d38_10 undeclared ← optional fixed_len_byte_array(16) (DECIMAL(38,10))|}
      );
    ]
    (fun (name, want) -> equal text want (format_text (sniff name)))

(* LogicalTypes.md: for an annotation that does not apply, "readers should
   ignore both the logical type annotation and column order for that column". *)
let ignored_annotations =
  cases "an annotation that does not apply or is unknown is ignored" ~id:fst
    [
      ( "unknown-logical-type.parquet",
        {|parquet (2 columns)
  "column with known type" string   ← optional binary (STRING)
  "column with unknown type" binary ← optional binary|}
      );
      ( "int32_with_uuid_logical_type.parquet",
        {|parquet (1 column)
  int32_uuid int32 ← required int32|} );
      ( "flba12_timestamp.parquet",
        {|parquet (3 columns)
  timestamp_millis binary ← optional fixed_len_byte_array(12)
  timestamp_micros binary ← optional fixed_len_byte_array(12)
  timestamp_nanos binary  ← optional fixed_len_byte_array(12)|}
      );
    ]
    (fun (name, want) -> equal text want (format_text (sniff name)))

(* [decimal scale] is a file of the int32s 7 and -8 annotated DECIMAL(5,
   [scale]). *)
let decimal scale =
  parquet ~rows:2
    ~column:[ (6, I32 5); (7, I32 scale); (8, I32 5) ]
    (page 2 (int32s [ 7; -8 ]))

let types =
  group "Types"
    [
      reads_as_pyarrow_test;
      sniffed_types;
      ignored_annotations;
      (* LogicalTypes.md: "Scale must be zero or a positive integer less than or
         equal to the precision." *)
      test "a DECIMAL of a negative scale is ignored" (fun () ->
          equal text "parquet (1 column)\n  x int32 ← required int32"
            (format_text (Error.get_ok (P.sniff (decimal (-2))))));
      test "a declared decimal reads another scale as float64, not as int64"
        (fun () ->
          let declared ty =
            P.with_type "x" ty (Error.get_ok (P.sniff (decimal 2)))
          in
          (match read_from (declared (any Type.int64)) (decimal 4) "x" with
          | Ok _ -> fail "read decimals of another scale as int64"
          | Error e -> contains ~sub:"another scale" (error_text e));
          let got =
            Error.get_ok
              (read_from (declared (any Type.float64)) (decimal 4) "x")
          in
          equal
            (array (option float_exact))
            [| Some 0.0007; Some (-0.0008) |]
            (Column.options Kind.float got));
    ]

(* Nested columns *)

(* The law: rows of nullable lists of nullable lists of nullable integers,
   flattened by hand to Dremel's stripes, read back as themselves. Definition
   levels: 0 null row, 1 empty, 2 null inner list, 3 empty, 4 null, 5 a
   value. *)

let stripe_elements =
  [
    list_group optional "x" 1;
    node repeated "list" 1;
    list_group optional "element" 1;
    node repeated "list" 1;
    field optional "element";
  ]

let shred rows =
  let entries = ref [] and values = ref [] in
  let entry r d = entries := (r, d) :: !entries in
  let inner r = function
    | None -> entry r 2
    | Some [||] -> entry r 3
    | Some xs ->
        Array.iteri
          (fun i x ->
            let r = if i = 0 then r else 2 in
            match x with
            | None -> entry r 4
            | Some v ->
                entry r 5;
                values := v :: !values)
          xs
  in
  let row = function
    | None -> entry 0 0
    | Some [||] -> entry 0 1
    | Some ls -> Array.iteri (fun i l -> inner (min i 1) l) ls
  in
  Array.iter row rows;
  (List.rev !entries, List.rev !values)

let rendered rows =
  let bracket f xs =
    "[" ^ String.concat ", " (Array.to_list (Array.map f xs)) ^ "]"
  in
  let opt f = function None -> "null" | Some x -> f x in
  Array.map (opt (bracket (opt (bracket (opt string_of_int))))) rows

let stripes =
  let open Gen in
  let small g = array ~size:(int_range 0 3) g in
  let choices = list ~size:(int_range 0 12) (int_range 0 5) in
  quad
    (array ~size:(int_range 1 12)
       (option (small (option (small (option (int_range (-9) 9)))))))
    nat choices choices

(* [striped rows k rep def] is the file of [rows], its pages cut at the entry
   [k], its levels cut into runs by the choices [rep] and [def]. *)
let striped rows k rc dc =
  let entries, values = shred rows in
  let n = List.length entries in
  let present (a, b) =
    List.length (List.filter (fun (_, d) -> d = 5) (sub entries (a, b)))
  in
  let page (a, b) =
    let es = sub entries (a, b) in
    let reps = runs_of rc (List.map fst es)
    and defs = runs_of dc (List.map snd es) in
    ( reps,
      defs,
      level_page ~rep:(hybrid 2 reps) ~def:(hybrid 3 defs) (b - a)
        (int32s (sub values (present (0, a), present (0, b)))) )
  in
  let pages = List.map page (cuts n [ k ]) in
  let b =
    file ~elements:stripe_elements
      [
        ( Array.length rows,
          [
            leaf
              [ "x"; "list"; "element"; "list"; "element" ]
              ~values:n
              (String.concat "" (List.map (fun (_, _, p) -> p) pages));
          ] );
      ]
  in
  (b, entries, pages)

(* A map of the pairs (k, 1) and (k, 2) in one row, its outer group annotated
   [converted] (MAP 1, MAP_KEY_VALUE 2). *)
let map_file converted =
  let elements =
    [
      node ~converted optional "m" 1;
      node repeated "key_value" 2;
      field ~ty:6 required "key";
      field optional "value";
    ]
  in
  let pairs ~def data =
    level_page ~rep:(rle [ (1, 0); (1, 1) ]) ~def:(rle [ (2, def) ]) 2 data
  in
  file ~elements
    [
      ( 1,
        [
          leaf ~ty:6
            [ "m"; "key_value"; "key" ]
            ~values:2
            (pairs ~def:2 (prefixed "k" ^ prefixed "k"));
          leaf
            [ "m"; "key_value"; "value" ]
            ~values:2
            (pairs ~def:3 (int32s [ 1; 2 ]));
        ] );
    ]

let nested =
  group "Nested columns"
    [
      (* The example's 24 definition levels packed at 3 bits take 9 bytes. *)
      prop "a table read from its Dremel stripes equals the table they flatten"
        ~examples:
          [
            ( Array.make 12
                (Some [| Some [| Some 1; Some 2 |]; Some [| Some 3 |] |]),
              0,
              [],
              [ 5 ] );
          ]
        stripes
        (fun (rows, k, rc, dc) ->
          let has f = Array.exists f rows in
          let inner f =
            has (function Some ls -> Array.exists f ls | _ -> false)
          in
          let b, entries, pages = striped rows k rc dc in
          cover "a null row" (has Option.is_none);
          cover "an empty row" (has (( = ) (Some [||])));
          cover "a null inner list" (inner Option.is_none);
          cover "an empty inner list" (inner (( = ) (Some [||])));
          cover "a null integer"
            (inner (function Some xs -> Array.mem None xs | _ -> false));
          let cut = k mod (List.length entries + 1) in
          cover "a page that starts inside a row"
            (0 < cut
            && cut < List.length entries
            && fst (List.nth entries cut) > 0);
          cover_runs 3 (List.concat_map (fun (_, d, _) -> d) pages);
          match read_from (Error.get_ok (P.sniff b)) b "x" with
          | Error e -> failf "%s" (error_text e)
          | Ok c -> equal (array string) (rendered rows) (cells c));
      cases "each nested column reads as pyarrow reads it"
        ~id:(fun l -> Scanf.sscanf l "%s %S" (fun f c -> f ^ " " ^ c))
        (In_channel.with_open_text "golden/nested.txt" In_channel.input_lines)
        (fun l ->
          let f, c, want =
            Scanf.sscanf l "%s %S %[^\n]" (fun f c w -> (f, c, w))
          in
          match read (sniff f) f c with
          | Ok col -> equal string want (cells_text col)
          | Error e -> failf "%s %s: %s" f c (error_text e));
      (* LogicalTypes.md, lists: rule 2, "If the repeated field is a group with
         multiple fields, then its type is the element type"; rule 4, a repeated
         field "named either array or uses the LIST-annotated group's name with
         _tuple appended"; and a repeated group outside a LIST. *)
      cases "a LIST's element follows Parquet's backward-compatibility rules"
        ~id:(fun (rule, _, _, _, _) -> rule)
        (let l = list_group optional "x" 1 in
         let a = field required "a" and b = field required "b" in
         [
           ( "a repeated group of two fields",
             "x list[record[a int32, b int32]]",
             [ l; node repeated "pair" 2; a; b ],
             [ [ "x"; "pair"; "a" ]; [ "x"; "pair"; "b" ] ],
             2 );
           ( "a repeated group named array",
             "x list[record[a int32]]",
             [ l; node repeated "array" 1; a ],
             [ [ "x"; "array"; "a" ] ],
             2 );
           ( "a repeated group outside a LIST",
             "x list[record[a int32, b int32]]",
             [ node repeated "x" 2; a; b ],
             [ [ "x"; "a" ]; [ "x"; "b" ] ],
             1 );
         ])
        (fun (_, want, elements, paths, def) ->
          let chunk path =
            leaf path ~values:2
              (level_page ~rep:(rle [ (2, 0) ]) ~def:(rle [ (2, def) ]) 2 "")
          in
          let b = file ~elements [ (2, List.map chunk paths) ] in
          let line =
            List.nth
              (String.split_on_char '\n'
                 (format_text (Error.get_ok (P.sniff b))))
              1
          in
          equal string want
            (String.trim (List.hd (String.split_on_char '\xe2' line))));
      cases "a map reads its pairs in the file's order, equal keys included"
        ~id:fst
        [ ("MAP", 1); ("MAP_KEY_VALUE", 2) ]
        (fun (_, converted) ->
          let b = map_file converted in
          equal (array string)
            [| {|[{key: "k", value: 1}, {key: "k", value: 2}]|} |]
            (cells (Error.get_ok (read_from (Error.get_ok (P.sniff b)) b "m"))));
      test "a definition level past 255 reads" (fun () ->
          (* Definition levels 300 and 0, as RLE runs of 2-byte values. *)
          let depth = 300 in
          let elements =
            List.init depth (fun i ->
                node optional (if i = 0 then "x" else "g") 1)
            @ [ field required "v" ]
          in
          let path = ("x" :: List.init (depth - 1) (fun _ -> "g")) @ [ "v" ] in
          let p =
            page ~data_page:[] 2
              (prefixed (hybrid 9 [ Repeated (1, 300); Repeated (1, 0) ])
              ^ int32s [ 42 ])
          in
          let b = file ~elements [ (2, [ leaf path ~values:2 p ]) ] in
          let c = Error.get_ok (read_from (Error.get_ok (P.sniff b)) b "x") in
          equal int 1 (Column.null_count c));
    ]

(* Tensors, in the file gen/tensors.py writes *)

let tensors = "tensors.parquet"

(* [cell i b y x] is the value gen/tensors.py writes at cell (b, y, x) of row
   [i]. *)
let cell i b y x = Float.of_int ((i * 100) + (b * 12) + (y * 4) + x)

(* [image_cells i (bands, height, width)] is the cells of row [i] of that shape,
   in row-major order. *)
let image_cells i (bands, height, width) =
  Array.init
    (bands * height * width)
    (fun k ->
      cell i (k / (height * width)) (k / width mod height) (k mod width))

let image = Type.tensor Nx.float32 [| 2; 3; 4 |]

let read_as column ty =
  read (P.with_type column (any ty) (sniff tensors)) tensors column

(* [fixed c] is the validity and the values of the column [c] of one value per
   row, flattened. *)
let fixed c =
  match Column.layout c with
  | Fixed { validity; values = Nx.P x } ->
      let valid =
        match validity with
        | None -> Array.make (Column.length c) true
        | Some v -> Nx.to_array (Nx.cast Nx.bool v)
      in
      (valid, Nx.to_array (Nx.cast Nx.float64 (Nx.reshape [| -1 |] x)))
  | _ -> failwith "not one value per row"

let tensor_columns =
  group "Tensors"
    [
      test "three list levels read as a tensor of the stated shape" (fun () ->
          let c = Error.get_ok (read_as "image" image) in
          let valid, values = fixed c in
          equal (array bool) (Array.init 8 (fun i -> i <> 2)) valid;
          let size = 2 * 3 * 4 in
          List.iter
            (fun i ->
              equal
                ~msg:(Printf.sprintf "row %d" i)
                (array float_exact)
                (image_cells i (2, 3, 4))
                (Array.sub values (i * size) size))
            [ 0; 1; 3; 4; 5; 6; 7 ]);
      test "four list levels read as a list of tensors" (fun () ->
          let ty = Type.list (Type.tensor Nx.float32 [| 3; 4 |]) in
          let c = Error.get_ok (read_as "bands" ty) in
          match Column.layout c with
          | Varsize { validity; offsets; child } ->
              equal (array bool)
                (Array.init 8 (fun i -> i <> 4))
                (Nx.to_array (Nx.cast Nx.bool (Option.get validity)));
              let bands i = if i = 4 then 0 else i mod 3 in
              let starts = Array.make 9 0 in
              for i = 0 to 7 do
                starts.(i + 1) <- starts.(i) + bands i
              done;
              equal (array int) starts
                (Array.map Int64.to_int (Nx.to_array offsets));
              let want =
                Array.concat
                  (List.init 8 (fun i -> image_cells i (bands i, 3, 4)))
              in
              equal (array float_exact) want (snd (fixed child))
          | _ -> fail "not a list");
      test "lists of two levels read as tensors" (fun () ->
          let ints ty file column =
            let f = P.with_type column (any ty) (sniff file) in
            Result.map cells (Result.map_error error_text (read f file column))
          in
          let cs = result (array string) string in
          equal cs (Ok [| "[[1, 2], [3, 4]]" |])
            (ints
               (Type.tensor Nx.int32 [| 2; 2 |])
               "old_list_structure.parquet" "a");
          equal cs
            (Error
               "row group 0: column \"Int32_list\": row 1: a list of level 0 \
                holds 0 elements, not the shape's 4")
            (ints
               (Type.tensor Nx.int32 [| 4 |])
               "repeated_primitive_no_list.parquet" "Int32_list"));
    ]

(* Refusals *)

(* [interval ~listed] is a file of an INTERVAL (converted 21) column of
   fixed_len_byte_array(12), alone or as a list's element. *)
let interval ~listed =
  let v =
    Struct
      [
        (1, I32 7);
        (2, I32 12);
        (3, I32 optional);
        (4, Bin "element");
        (6, I32 21);
      ]
  in
  if listed then
    file
      ~elements:[ list_group optional "x" 1; node repeated "list" 1; v ]
      [ (0, [ leaf ~ty:7 x_path ~values:0 "" ]) ]
  else file ~elements:[ v ] [ (0, [ leaf ~ty:7 [ "element" ] ~values:0 "" ]) ]

(* [variant ~listed] is a file of a group annotated VARIANT (LogicalType field
   16), alone or as a list's element. *)
let variant ~listed =
  let v =
    Struct
      [
        (3, I32 optional);
        (4, Bin "element");
        (5, I32 1);
        (10, Struct [ (16, Struct []) ]);
      ]
  in
  let m = Struct [ (1, I32 6); (3, I32 required); (4, Bin "metadata") ] in
  if listed then
    file
      ~elements:[ list_group optional "x" 1; node repeated "list" 1; v; m ]
      [
        (0, [ leaf ~ty:6 [ "x"; "list"; "element"; "metadata" ] ~values:0 "" ]);
      ]
  else
    file ~elements:[ v; m ]
      [ (0, [ leaf ~ty:6 [ "element"; "metadata" ] ~values:0 "" ]) ]

(* [codec n] is a file whose chunk states the codec [n]. *)
let codec n = parquet ~rows:0 ~chunk:[ (4, I32 n) ] ""
let refused elements = file ~elements [ (0, [ leaf x_path ~values:0 "" ]) ]

let refusals =
  test "sniff refuses a file talon cannot read in full" (fun () ->
      let show name b = Printf.printf "%s: %s\n" name (outcome (P.sniff b)) in
      List.iter
        (fun name -> show name (buffer name))
        [
          "uniform_encryption.parquet.encrypted";
          "encrypt_columns_plaintext_footer.parquet.encrypted";
          "hadoop_lz4_compressed.parquet";
          "non_hadoop_lz4_compressed.parquet";
          "brotli.parquet";
          "large_string_map.brotli.parquet";
        ];
      show "LZO" (codec 3);
      show "codec 99" (codec 99);
      (let fields, p = leaf [ "x" ] ~values:0 "" in
       show "a chunk in another file"
         (file
            ~elements:[ field required "x" ]
            [
              (0, [ ((fun pos -> (1, Bin "other.parquet") :: fields pos), p) ]);
            ]));
      show "an interval" (interval ~listed:false);
      show "an interval in a list" (interval ~listed:true);
      show "a variant" (variant ~listed:false);
      show "a variant in a list" (variant ~listed:true);
      let a = field repeated "a" and r = field required "a" in
      show "a LIST of two fields" (refused [ list_group optional "x" 2; a; a ]);
      show "a LIST of no repeated field"
        (refused [ list_group optional "x" 1; field optional "a" ]);
      show "a repeated LIST outside a list"
        (refused [ list_group repeated "x" 1; a ]);
      show "a MAP whose entries are not a group"
        (refused [ node ~converted:1 optional "x" 1; a ]);
      show "a group of no fields" (refused [ node optional "x" 0 ]);
      show "two fields of one name" (refused [ node optional "x" 2; r; r ]);
      expect (output ())
      @@ __POS_OF__
           {|
        uniform_encryption.parquet.encrypted: the file is encrypted, which talon does not read
        encrypt_columns_plaintext_footer.parquet.encrypted: the file is encrypted, which talon does not read
        hadoop_lz4_compressed.parquet: row group 0: column "c0" is compressed with LZ4 in Hadoop's framing, which talon does not read; LZ4_RAW is read
        non_hadoop_lz4_compressed.parquet: row group 0: column "c0" is compressed with LZ4 in Hadoop's framing, which talon does not read; LZ4_RAW is read
        brotli.parquet: row group 0: column "i32" is compressed with Brotli, which talon does not read
        large_string_map.brotli.parquet: row group 0: column "arr.key_value.key" is compressed with Brotli, which talon does not read
        LZO: row group 0: column "x" is compressed with LZO, which talon does not read
        codec 99: row group 0: column "x" is compressed with codec 99, which Parquet does not define
        a chunk in another file: row group 0: column "x" is stored in "other.parquet", which talon does not read
        an interval: column "element" is an interval, which talon does not read
        an interval in a list: column "x.list.element" is an interval, which talon does not read
        a variant: column "element" is a variant, which talon does not read
        a variant in a list: column "x.list.element" is a variant, which talon does not read
        a LIST of two fields: column "x" is a list of 2 fields, not of one repeated field
        a LIST of no repeated field: column "x" is a list of 1 fields, not of one repeated field
        a repeated LIST outside a list: column "x" is a repeated list or map, which Parquet allows only as the element of another list
        a MAP whose entries are not a group: column "x" is a map whose entries are not a group
        a group of no fields: the schema is malformed: column "x" is a group of no fields
        two fields of one name: column "x" has two fields named "a"
        |})

(* sniff *)

(* [footer s] is the position of the footer of the Parquet file [s]. *)
let footer s =
  let n = String.length s in
  n - 8 - Int32.to_int (String.get_int32_le s (n - 8))

(* [cut_footer s k] is the Parquet file [s] with the last [k] bytes of its
   footer cut, and the footer's length rewritten. *)
let cut_footer s k =
  let first = footer s in
  let len = String.length s - 8 - first - k in
  String.sub s 0 (first + len) ^ int32s [ len ] ^ "PAR1"

let sniffing =
  group "sniff"
    [
      cases "a disk buffer and a host buffer of a file give one format"
        ~id:Fun.id
        [
          "alltypes_plain.parquet";
          "types.parquet";
          "names.parquet";
          "empty.parquet";
        ] (fun name ->
          let bytes =
            Error.get_ok (P.sniff (Rig.Buffer.of_string (file_bytes name)))
          in
          equal text (format_text (sniff name)) (format_text bytes));
      test "is an Error naming the bytes of a file that is not Parquet"
        (fun () ->
          let file = file_bytes "alltypes_plain.parquet" in
          let n = String.length file in
          let show what b =
            Printf.printf "%s: %s\n" what (outcome (P.sniff b))
          in
          List.iter
            (fun (what, s) -> show what (Rig.Buffer.of_string s))
            [
              ("no bytes", "");
              ("11 bytes", "PAR1PAR1PAR");
              ("PAR1 at the start only", String.sub file 0 (n - 1) ^ "X");
              ("PAR1 at the end only", "X" ^ String.sub file 1 (n - 1));
              ("a footer longer than the file", "PAR1" ^ "\xff\xff\x00\x00PAR1");
              ( "a footer of 5 bytes in 4",
                "PAR1" ^ "\x00\x00\x00\x00" ^ "\x05\x00\x00\x00PAR1" );
              ( "a footer of 4 bytes in 4",
                "PAR1" ^ "\x00\x00\x00\x00" ^ "\x04\x00\x00\x00PAR1" );
              ("a footer cut short", cut_footer file 1);
            ];
          List.iter
            (fun name -> show name (buffer name))
            [
              "bad_data/PARQUET-1481.parquet";
              "bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet";
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            no bytes: not a Parquet file: it has 0 bytes, fewer than 12
            11 bytes: not a Parquet file: it has 11 bytes, fewer than 12
            PAR1 at the start only: bytes 1847-1850: not a Parquet file: it does not end with "PAR1"
            PAR1 at the end only: bytes 0-3: not a Parquet file: it does not start with "PAR1"
            a footer longer than the file: bytes 4-7: the footer's length, 65535 bytes, does not fit the file
            a footer of 5 bytes in 4: bytes 8-11: the footer's length, 5 bytes, does not fit the file
            a footer of 4 bytes in 4: byte 4: the footer is malformed: FileMetaData has no field 2 (schema)
            a footer cut short: byte 1842: the footer is malformed: unexpected end of the data
            bad_data/PARQUET-1481.parquet: byte 307: the footer is malformed: an undefined physical type, -7
            bad_data/ARROW-RS-GH-6229-DICTHEADER.parquet: row group 0: bytes 129-450: column "name": its pages overrun the file, whose footer starts at byte 291
            |});
    ]

(* with_type *)

(* fallback.parquet's string column, read as a categorical of its sorted values
   but one, which then fails, and of all of them, whose codes then read back as
   the strings. *)
let categorical () =
  let name = "fallback.parquet" in
  let f = sniff name in
  let strings = flat_cells "string" (Error.get_ok (read f name "string")) in
  let dict =
    Array.to_list strings
    |> List.filter (( <> ) "null")
    |> List.sort_uniq compare
    |> List.map (fun s -> Scanf.sscanf s "%S" Fun.id)
    |> Array.of_list
  in
  let read_as d =
    read (P.with_type "string" (any (Type.categorical d)) f) name "string"
  in
  (match read_as (Array.sub dict 1 (Array.length dict - 1)) with
  | Ok _ -> fail "read a value outside the dictionary"
  | Error e ->
      expect (error_text e)
      @@ __POS_OF__
           {| row group 0: "v0é\x0a\"": column "string": the value is not in the categorical's dictionary |});
  let codes = flat_cells "int32" (Error.get_ok (read_as dict)) in
  let back =
    Array.map
      (fun s -> if s = "null" then s else quote dict.(int_of_string s))
      codes
  in
  equal (array string) strings back

let retype f column t =
  match P.with_type column t f with
  | f -> Printf.sprintf "%s reads as %s" column (format_type f column)
  | exception Invalid_argument msg -> msg

let refusal f =
  match f () with _ -> "no refusal" | exception Invalid_argument msg -> msg

let declare_decimals f =
  List.fold_left
    (fun f c ->
      match P.with_type c (any Type.float64) f with
      | f -> f
      | exception Invalid_argument _ -> f)
    f [ "d9"; "d18" ]

let with_type =
  group "with_type"
    [
      cases "is the identity at a column's own type" ~id:Fun.id
        [
          "bool";
          "i8";
          "i16";
          "i32";
          "i64";
          "u8";
          "u16";
          "u32";
          "u64";
          "f16";
          "f32";
          "f64";
          "date";
          "clock_ms";
          "clock_us";
          "clock_ns";
          "ts_ms";
          "ts_us_utc";
          "ts_us_paris";
          "ts_ns";
          "string";
          "binary";
          "fsb";
          "uuid";
          "all_null";
          "no_null";
          "req_i32";
          "req_string";
        ] (fun column ->
          let f = sniff "types.parquet" in
          let s = P.source (declare_decimals f) (buffer "types.parquet") in
          let t =
            List.assoc column
              (Schema.columns (Query.schema (Query.of_source s)))
          in
          equal text (format_text f) (format_text (P.with_type column t f)));
      cases "a column read as an alternative reads as pyarrow reads it"
        ~id:Fun.id (files override_lines) alternative_reads_as_pyarrow;
      test
        "reads byte strings as string, binary or a categorical, and int96 as \
         any unit" (fun () ->
          let f = sniff "types.parquet"
          and plain = sniff "alltypes_plain.parquet" in
          List.iter print_endline
            [
              retype f "string" (any Type.binary);
              retype f "binary" (any Type.string);
              retype f "fsb" (any (Type.categorical [| "a" |]));
              retype f "uuid" (any Type.string);
              retype plain "timestamp_col" (any (Type.wall S));
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            string reads as binary
            binary reads as string
            fsb reads as categorical["a"]
            uuid reads as string
            timestamp_col reads as datetime[s]
            |});
      test "a categorical reads codes into its dictionary" categorical;
      test "declares a decimal as float64, or as int64 to 18 digits" (fun () ->
          let f = sniff "decimals_int.parquet" in
          List.iter print_endline
            [
              retype f "d9_2" (any Type.float64);
              retype f "d18_6" (any Type.int64);
              retype f "d38_10" (any Type.float64);
              retype f "d38_0" (any Type.int64);
              retype f "d9_0" (any Type.int32);
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            d9_2 reads as float64
            d18_6 reads as int64
            d38_10 reads as float64
            Talon_parquet.with_type: column "d38_0" (optional fixed_len_byte_array(16) (DECIMAL(38,0))) reads as float64, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "d9_0" (optional int32 (DECIMAL(9,0))) reads as float64 or int64, not int32. Cast it once read (Expr.cast).
            |});
      cases "retypes a nested column's leaves, keeping its shape"
        ~id:(fun (c, _, _) -> c)
        [
          ("texts", "list[binary]", Type.list Type.binary |> any);
          ( "people",
            "list[record[age int32, name binary]]",
            any
              (Type.list
                 (Type.record
                    [ ("age", any Type.int32); ("name", any Type.binary) ])) );
          ( "record",
            "record[x int32, tags list[binary], sub record[y float64]]",
            any
              (Type.record
                 [
                   ("x", any Type.int32);
                   ("tags", any (Type.list Type.binary));
                   ("sub", any (Type.record [ ("y", any Type.float64) ]));
                 ]) );
        ]
        (fun (column, want, ty) ->
          let name = "nested_v1_plain.parquet" in
          let c =
            Error.get_ok (read (P.with_type column ty (sniff name)) name column)
          in
          let (Type.Any t) = Column.type_ c in
          equal string want (Format.asprintf "%a" Type.pp t));
      test "raises naming the types a column reads as" (fun () ->
          let types = sniff "types.parquet"
          and plain = sniff "alltypes_plain.parquet"
          and nested = sniff "nested_v1_plain.parquet"
          and tensor = sniff tensors in
          List.iter print_endline
            [
              retype types "carier" (any Type.string);
              retype types "i32" (any Type.int64);
              retype types "f64" (any Type.float32);
              retype types "string" (any Type.int64);
              retype types "d9" (any Type.binary);
              retype plain "timestamp_col" (any Type.date);
              retype plain "timestamp_col" (any (Type.instant Us));
              retype types "ts_us_utc" (any (Type.wall Us));
              retype types "ts_ms" (any (Type.instant Ms));
              retype nested "lists" (any Type.int64);
              retype tensor "image" (any (Type.tensor Nx.float32 [| 6; 4 |]));
              retype tensor "image"
                (any (Type.tensor Nx.float32 [| 2; 3; 4; 1 |]));
              retype tensor "image" (any (Type.tensor Nx.float64 [| 2; 3; 4 |]));
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_parquet.with_type: no column "carier" in the format.
            Talon_parquet.with_type: column "i32" (optional int32) reads as int32, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "f64" (optional double) reads as float64, not float32. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "string" (optional binary (STRING)) reads as string, binary or a categorical, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "d9" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) reads as float64 or int64, not binary. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "timestamp_col" (optional int96) reads as wall-clock datetime of any unit, not date. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "timestamp_col" (optional int96) reads as wall-clock datetime of any unit, not datetime[us, UTC]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "ts_us_utc" (optional int64 (TIMESTAMP(MICROS,true))) reads as datetime[us, UTC], not datetime[us]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "ts_ms" (optional int64 (TIMESTAMP(MILLIS,false))) reads as datetime[ms], not datetime[ms, UTC]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "lists" (optional group (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional int64 element; } } } }) reads as list[list[int64]], or a tensor over int64 in place of its 1 to 2 innermost lists, not int64. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "image" (optional group (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional float element; } } } } } }) reads as list[list[list[float32]]], or a tensor over float32 in place of its 1 to 3 innermost lists, not tensor[float32, 6×4]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "image" (optional group (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional float element; } } } } } }) reads as list[list[list[float32]]], or a tensor over float32 in place of its 1 to 3 innermost lists, not tensor[float32, 2×3×4×1]. Cast it once read (Expr.cast).
            Talon_parquet.with_type: column "image" (optional group (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional group element (LIST) { repeated group list { optional float element; } } } } } }) reads as list[list[list[float32]]], or a tensor over float32 in place of its 1 to 3 innermost lists, not tensor[float64, 2×3×4]. Cast it once read (Expr.cast).
            |});
    ]

(* pp_format *)

let pp_format =
  test "pp_format prints a count line and aligned columns" (fun () ->
      equal text
        {|parquet (4 columns)
  "a b" int64   ← optional int64
  "\"q\"" int64 ← optional int64
  é int64       ← optional int64
  "" int64      ← optional int64|}
        (format_text (sniff "names.parquet")))

(* Reading *)

(* [first_batch f b column] is the first batch a query of [column] of the file
   [b], read as [f], folds over: its first row group, as decoded. *)
let first_batch f b column =
  let q =
    Query.of_source (P.source f b)
    |> Query.select Expr.[ keep (Sel.names [ column ]) ]
  in
  let first acc t = match acc with None -> Some t | some -> some in
  Option.get (Error.get_ok (Query.fold q ~init:None first))

let written ?(compression = P.Uncompressed) ?row_group t =
  let q = Query.of_table t in
  let w = Buffer.create 1024 in
  Error.get_ok
    (P.encode
       (P.format ~compression ?row_group (Query.schema q))
       q
       (Bytesrw.Bytes.Writer.of_buffer w));
  Buffer.contents w

(* A file of a list, a record and a tensor column, three rows, no null. *)
let nested_written =
  lazy
    (let record =
       Column.of_layout
         (any (Type.record [ ("a", any Type.int32) ]))
         (Children
            {
              validity = None;
              length = 3;
              fields = [ ("a", Column.v Type.int32 [| 1; 2; 3 |]) ];
            })
       |> Result.get_ok
     in
     Rig.Buffer.of_string
       (written
          (Talon.v
             [
               ( "list",
                 Column.v (Type.list Type.int32) [| [| 1; 2 |]; [||]; [| 3 |] |]
               );
               ("record", record);
               ( "tensor",
                 Column.of_tensor
                   (Nx.create Nx.float32 [| 3; 2 |]
                      [| 1.; 2.; 3.; 4.; 5.; 6. |]) );
             ])))

(* G1: an optional boolean column, its definition levels and its values (RLE)
   cut into hybrid runs by drawn choices, in pages cut at drawn rows, so that a
   page starts at any bit of the validity and of the values. *)
let optional_bools =
  let open Gen in
  let choices = list ~size:(int_range 0 30) (int_range 0 5) in
  quad
    (array ~size:(int_range 1 200) (option bool))
    (list ~size:(int_range 0 4) nat)
    choices choices

let bools_file rows ks dc vc =
  let n = Array.length rows in
  let pages = cuts n ks in
  let page (a, b) =
    let rs = Array.to_list (Array.sub rows a (b - a)) in
    let defs = List.map (fun r -> if Option.is_some r then 1 else 0) rs in
    let values = List.filter_map (Option.map Bool.to_int) rs in
    let rle_bools = hybrid 1 (runs_of vc values) in
    page
      ~data_page:[ (2, I32 3) ]
      (b - a)
      (prefixed (hybrid 1 (runs_of dc defs)) ^ prefixed rle_bools)
  in
  let elements = [ field ~ty:0 optional "x" ] in
  ( file ~elements
      [
        ( n,
          [
            leaf ~ty:0 [ "x" ] ~values:n
              (String.concat "" (List.map page pages));
          ] );
      ],
    pages )

(* G2: dictionary-encoded words, their indices at any bit width from the
   smallest to 32 ("max bit width = 32", Encodings.md), cut into hybrid runs by
   drawn choices, in several data pages. The words run from empty to past 32
   bytes through 4, 8 and 16, where the gather changes how it copies a row, in
   ASCII, two- and four-byte UTF-8 and U+0000. *)
let word =
  let open Gen in
  let scalar = of_list [ "a"; "b"; "\xc3\xa9"; "\000"; "\xf0\x9d\x84\x9e" ] in
  (* [fill n] is scalars of [n] bytes in all, padded with ["a"]. *)
  let rec fill n =
    if n = 0 then constant ""
    else
      let* s = scalar in
      let s = if String.length s <= n then s else "a" in
      let+ rest = fill (n - String.length s) in
      s ^ rest
  in
  bind
    (one_of
       [
         int_range 0 3;
         int_range 4 7;
         int_range 8 15;
         int_range 16 32;
         int_range 33 40;
       ])
    fill

let rec smallest_width n w = if 1 lsl w >= n then w else smallest_width n (w + 1)

let dictionary_case =
  let open Gen in
  let* words = array ~size:(int_range 1 12) word in
  let lo = smallest_width (Array.length words) 0 in
  let* w =
    frequency
      [ (1, constant lo); (1, int_range lo (max lo 8)); (2, int_range lo 32) ]
  in
  let* indices =
    list ~size:(int_range 0 60) (int_range 0 (Array.length words - 1))
  in
  let* ks = list ~size:(int_range 0 3) nat in
  let+ choices = list ~size:(int_range 0 12) (int_range 0 5) in
  (words, indices, w, ks, choices)

(* [dictionary_file words w pages] is a file of the required text column [x]
   (BYTE_ARRAY, UTF8): a dictionary page of [words], then a data page of each of
   [pages]' indices at [w] bits, cut into runs by [choices]. *)
let dictionary_file ?(choices = []) words w pages =
  let dict = dictionary_page words in
  let data indices =
    page
      ~data_page:[ (2, I32 8) ]
      (List.length indices)
      (String.make 1 (Char.chr w) ^ hybrid w (runs_of choices indices))
  in
  parquet
    ~rows:(List.length (List.concat pages))
    ~column:[ (1, I32 6); (6, I32 0) ]
    ~dictionary:(String.length dict)
    (dict ^ String.concat "" (List.map data pages))

let words_at words indices =
  Array.of_list (List.map (fun i -> Some words.(i)) indices)

let dictionary_words =
  prop "dictionary indices at any bit width read as their words"
    ~examples:[ ([| "a" |], [ 0; 0; 0 ], 0, [], []) ]
    dictionary_case
    (fun (words, indices, w, ks, choices) ->
      let pages = List.map (sub indices) (cuts (List.length indices) ks) in
      let runs = List.map (runs_of choices) pages in
      let used lo hi =
        List.exists
          (fun i ->
            let l = String.length words.(i) in
            lo <= l && l <= hi)
          indices
      in
      cover "a row of 1 to 3 bytes" (used 1 3);
      cover "a row of 4 to 7 bytes" (used 4 7);
      cover "a row of 8 to 15 bytes" (used 8 15);
      cover "a row of 16 to 32 bytes" (used 16 32);
      cover "a row past 32 bytes" (used 33 max_int);
      cover "a width above the smallest"
        (w > smallest_width (Array.length words) 0);
      cover "several data pages"
        (List.length (List.filter (( <> ) []) pages) > 1);
      cover_runs w (List.concat runs);
      cover "eight values unpacked at once" (List.exists (unpacks_eight w) runs);
      let b = dictionary_file ~choices (Array.to_list words) w pages in
      equal
        (array (option string))
        (words_at words indices)
        (Column.options Kind.string (Error.get_ok (read_x b)));
      (* A dictionary one word short of the largest index. *)
      if indices <> [] then begin
        let largest = List.fold_left max 0 indices in
        let short =
          List.filteri (fun i _ -> i < largest) (Array.to_list words)
        in
        is_error ~msg:"an index past the dictionary"
          (read_x (dictionary_file ~choices short w pages))
      end)

(* Words of one length read through a gather of fixed rows: the empty word, the
   widths with a one-element gather and others. *)
let one_length_case =
  let open Gen in
  let* l = one_of [ int_range 0 4; int_range 5 12; constant 8 ] in
  let* words =
    array ~size:(int_range 1 12)
      (string_of ~size:(constant l) (map Char.chr (int_range 32 126)))
  in
  let+ indices =
    list ~size:(int_range 0 40) (int_range 0 (Array.length words - 1))
  in
  (l, words, indices)

let one_length =
  prop "dictionary words of one length read as their words"
    ~examples:[ (0, [| "" |], [ 0; 0 ]); (1, [| "a"; "b" |], [ 1; 0 ]) ]
    one_length_case
    (fun (l, words, indices) ->
      cover "empty words" (l = 0 && indices <> []);
      cover "one-byte words" (l = 1 && indices <> []);
      cover "wider words" (l > 1 && indices <> []);
      let w = smallest_width (Array.length words) 0 in
      let b = dictionary_file (Array.to_list words) w [ indices ] in
      equal
        (array (option string))
        (words_at words indices)
        (Column.options Kind.string (Error.get_ok (read_x b))))

let pages =
  group "pages"
    [
      prop "an optional column reads its nulls from pages cut at any row"
        optional_bools (fun (rows, ks, dc, vc) ->
          let b, pages = bools_file rows ks dc vc in
          let long (a, b) = b - a >= 64 in
          cover "a page of 64 rows or more from a row not a multiple of 8"
            (List.exists (fun (a, b) -> long (a, b) && a mod 8 <> 0) pages);
          cover "a page of 64 rows or more from a multiple of 8"
            (List.exists (fun (a, b) -> long (a, b) && a mod 8 = 0) pages);
          equal
            (array (option bool))
            rows
            (Column.options Kind.bool (Error.get_ok (read_x b))));
      dictionary_words;
      one_length;
      (* A page of a type talon does not read carries nothing: the dictionary
         page after it still opens the chunk. *)
      test "a page of a type talon skips may precede the dictionary page"
        (fun () ->
          let words = [ "a"; "bb" ] in
          let other = thrift (Struct [ (1, I32 1); (2, I32 0); (3, I32 0) ]) in
          let dict = dictionary_page words in
          let data = String.make 1 '\001' ^ hybrid 1 [ Packed [ 1; 0; 1 ] ] in
          let b =
            parquet ~rows:3
              ~column:[ (1, I32 6); (6, I32 0) ]
              ~dictionary:(String.length other + String.length dict)
              (other ^ dict ^ page ~data_page:[ (2, I32 8) ] 3 data)
          in
          equal
            (array (option string))
            [| Some "bb"; Some "a"; Some "bb" |]
            (Column.options Kind.string (Error.get_ok (read_x b))));
      (* parquet.thrift: BIT_PACKED, deprecated, is "usable for
         definition/repetition levels encoding" alone. *)
      test "levels in BIT_PACKED fail the read" (fun () ->
          let b =
            parquet ~rows:2
              ~column:[ (3, I32 optional) ]
              (page ~data_page:[ (3, I32 4) ] 2 ("\x01" ^ int32s [ 1; 2 ]))
          in
          expect (outcome (read_x b))
          @@ __POS_OF__
               {| row group 0: bytes 4-27: column "x": definition levels in BIT_PACKED are not read |});
    ]

let reading =
  group "Reading"
    [
      test "a read decodes only the request's columns" (fun () ->
          let name = "nation.dict-malformed.parquet" in
          let f = sniff name in
          List.iter
            (fun c ->
              Printf.printf "%s: %s\n" c
                (outcome (Result.map Column.length (read f name c))))
            [ "nation_key"; "name" ];
          expect (output ())
          @@ __POS_OF__
               {|
            nation_key: ok
            name: row group 0: bytes 421-450: column "name": the page overruns its column chunk
            |});
      test "a read of no column has the rows of the file" (fun () ->
          let name = "names.parquet" in
          let q =
            Query.select []
              (Query.of_source (P.source (sniff name) (buffer name)))
          in
          equal int 2 (Talon.rows (table q)));
      test "a file's row groups are the source's batches" (fun () ->
          let t =
            Talon.v [ ("x", Column.v Type.int64 (Array.init 7 Fun.id)) ]
          in
          let b = Rig.Buffer.of_string (written ~row_group:3 t) in
          let q = Query.of_source (P.source (Error.get_ok (P.sniff b)) b) in
          equal (list int) [ 3; 3; 1 ]
            (List.rev
               (Error.get_ok
                  (Query.fold q ~init:[] (fun acc t -> Talon.rows t :: acc)))));
      (* Talon.column gives a batch's own column only when it is canonical, else
         a canonical copy each call. The files' 731-row groups end inside a
         byte, and their pages of 13 rows start inside one. *)
      cases "a batch's columns are canonical"
        ~id:(fun (f, c) -> f ^ " " ^ c)
        (List.concat_map
           (fun f ->
             List.map
               (fun c -> (f, c))
               [ "b"; "b_full"; "b_none"; "b_req"; "f"; "s" ])
           [ "bits_v1.parquet"; "bits_v2_rle.parquet"; "bits_dict.parquet" ])
        (fun (file, column) ->
          let t = first_batch (sniff file) (buffer file) column in
          satisfies ~claim:"Talon.column takes the batch's own column" pass
            (fun c -> c == Talon.column t column)
            (Talon.column t column));
      cases "a batch's nested columns with no null are canonical"
        ~id:(fun (c, _) -> c)
        [
          ("list", None);
          ("record", None);
          ("tensor", Some (any (Type.tensor Nx.float32 [| 2 |])));
        ]
        (fun (column, ty) ->
          let b = Lazy.force nested_written in
          let f = Error.get_ok (P.sniff b) in
          let f =
            match ty with Some ty -> P.with_type column ty f | None -> f
          in
          let t = first_batch f b column in
          satisfies ~claim:"Talon.column takes the batch's own column" pass
            (fun c -> c == Talon.column t column)
            (Talon.column t column));
      pages;
    ]

(* Text conjuncts *)

(* The files whose text columns the source filters in its scan: dictionary
   pages, dictionary pages that fall back to PLAIN ones, PLAIN pages only, and
   no rows; each column with nulls but [req_string]; [fsb] of fixed length. *)
let scanned =
  [
    ("types.parquet", [ "string"; "binary"; "req_string"; "fsb" ]);
    ("fallback.parquet", [ "string"; "binary" ]);
    ("types_plain.parquet", [ "string"; "binary"; "req_string"; "fsb" ]);
    ("empty.parquet", [ "string"; "binary" ]);
  ]

let scan_source file = P.source (declare_decimals (sniff file)) (buffer file)

(* Each file of [scanned], read whole once. *)
let tables =
  lazy
    (List.map
       (fun (file, _) -> (file, table (Query.of_source (scan_source file))))
       scanned)

let is_binary c = c = "binary" || c = "fsb"

let texts file c =
  let col = Talon.column (List.assoc file (Lazy.force tables)) c in
  let ts =
    if is_binary c then
      Array.map
        (Option.map (fun b -> (b : Binary.t :> string)))
        (Column.options Kind.binary col)
    else Column.options Kind.string col
  in
  List.filter_map Fun.id (Array.to_list ts)

(* [pred x v texts] draws a predicate on [x], whose values [v] makes of texts,
   over [texts] and texts no column holds. *)
let pred (type a) (x : (a, Expr.row) Expr.t) (v : string -> a) texts =
  let open Gen in
  let absent = [ ""; "v"; "zz"; "\x00"; "v1\xc3\xa9" ] in
  let text =
    map v
      (if texts = [] then of_list absent
       else frequency [ (3, of_list texts); (1, of_list absent) ])
  in
  let cmp =
    let* op = of_list Expr.[ ( = ); ( <> ); ( < ); ( <= ); ( > ); ( >= ) ] in
    let+ s = text in
    op x (Expr.const s)
  in
  let atom =
    frequency
      [
        (4, cmp);
        (2, map (fun ss -> Expr.is_in ss x) (list ~size:(int_range 0 9) text));
        (3, constant (Expr.is_null x));
      ]
  in
  let rec p depth =
    if depth = 0 then atom
    else
      let sub = p (depth - 1) in
      frequency
        [
          (3, atom);
          (1, map Expr.not sub);
          (1, map (fun (a, b) -> Expr.(a && b)) (pair sub sub));
          (1, map (fun (a, b) -> Expr.(a || b)) (pair sub sub));
        ]
  in
  p 2

let text_case =
  let open Gen in
  let* file, cs = of_list scanned in
  let* c = of_list cs in
  let* p =
    if is_binary c then pred (Col.binary c) Binary.of_string (texts file c)
    else pred (Col.string c) Fun.id (texts file c)
  in
  let+ positive = bool and+ shown = int_range 0 2 in
  let p = if positive then Expr.(p && Col.float "f64" > float 0.) else p in
  (file, c, p, List.nth [ [ c; "f64" ]; [ "f64" ]; [] ] shown)

let text_conjuncts =
  group "Text conjuncts"
    [
      test "a text conjunct is Exact and an integer one Inexact" (fun () ->
          let q p =
            Query.filter p (Query.of_source (scan_source "types.parquet"))
          in
          (* The lines of the plan that hold the conjunct. *)
          let plan p =
            Format.asprintf "%a" Query.pp (Query.optimize (q p))
            |> String.split_on_char '\n'
            |> List.filter (fun l -> not (String.starts_with ~prefix:"query" l))
            |> String.concat "\n"
          in
          print_endline (plan Expr.(Col.string "string" = string "a"));
          print_endline (plan Expr.(Col.int "i32" > int 0));
          expect (output ())
          @@ __POS_OF__
               {|
            parquet (30 columns) ~filters:["string" = "a"]
            filter (i32 > 0)
            └ parquet (30 columns) ~filters:[i32 > 0]
            |});
      prop "a source keeps the rows the filter keeps over the read table"
        ~examples:
          [
            ("empty.parquet", "string", Expr.is_null (Col.string "string"), []);
            ( "types.parquet",
              "string",
              Expr.is_null (Col.string "string"),
              [ "string"; "f64" ] );
          ]
        (Gen.with_pp
           (fun ppf (file, c, p, names) ->
             Format.fprintf ppf "%s, %s: %a, reading [%s]" file c Expr.pp p
               (String.concat "; " names))
           text_case)
        (fun (file, c, p, names) ->
          let t = List.assoc file (Lazy.force tables) in
          let kept = table (Query.filter p (Query.of_table t)) in
          cover "dictionary pages" (file = "types.parquet");
          cover "dictionary pages falling back to PLAIN"
            (file = "fallback.parquet");
          cover "PLAIN pages" (file = "types_plain.parquet");
          cover "no rows" (Talon.rows t = 0);
          cover "some rows pass, others not"
            (Talon.rows kept > 0 && Talon.rows kept < Talon.rows t);
          cover "a null row passes" (Column.null_count (Talon.column kept c) > 0);
          cover "the filtered column not read back" (not (List.mem c names));
          cover "no column read back" (names = []);
          let shown q =
            table
              (Query.select Expr.[ keep (Sel.names names) ] (Query.filter p q))
          in
          equal table_w
            (shown (Query.of_table t))
            (shown (Query.of_source (scan_source file))));
      (* Columns of a valid and an invalid UTF-8 word, read as text: a filter in
         the scan fails where a read fails, at the row, and not on a word no row
         reads, on the column it filters or another, dictionary-encoded or
         PLAIN. *)
      test "a filtered read fails only at a row of invalid UTF-8" (fun () ->
          let rows q =
            match Query.run q with
            | Ok t -> string_of_int (Talon.rows t) ^ " rows"
            | Error e -> error_text e
          in
          let as_text b names =
            let f = Error.get_ok (P.sniff b) in
            List.fold_left
              (fun f c -> P.with_type c (any Type.string) f)
              f names
          in
          List.iter
            (fun indices ->
              let b = dictionary_file [ "ok"; "\xff" ] 1 [ indices ] in
              let q = Query.of_source (P.source (as_text b [ "x" ]) b) in
              Printf.printf "x at %s: read %s, x = \"ok\" %s\n"
                (String.concat " " (List.map string_of_int indices))
                (rows q)
                (rows (Query.filter Expr.(Col.string "x" = string "ok") q)))
            [ [ 0; 0; 0 ]; [ 0; 1; 0 ] ];
          let text_column (name, c) =
            match c with
            | `Dict (words, indices) ->
                let dict = dictionary_page words in
                let data =
                  page
                    ~data_page:[ (2, I32 8) ]
                    4
                    ("\001" ^ hybrid 1 [ Packed indices ])
                in
                leaf ~ty:6 ~dictionary:(String.length dict) [ name ] ~values:4
                  (dict ^ data)
            | `Plain vs ->
                leaf ~ty:6 [ name ] ~values:4
                  (page 4 (String.concat "" (List.map prefixed vs)))
          in
          List.iter
            (fun (name, y) ->
              let b =
                file ~columns:2
                  ~elements:
                    [ field ~ty:6 required "s"; field ~ty:6 required "y" ]
                  [
                    ( 4,
                      List.map text_column
                        [
                          ("s", `Dict ([ "a"; "b" ], [ 0; 0; 1; 0 ])); ("y", y);
                        ] );
                  ]
              in
              let q = Query.of_source (P.source (as_text b [ "s"; "y" ]) b) in
              let s_is v = Query.filter Expr.(Col.string "s" = string v) q in
              Printf.printf "%s y: read %s, s = \"a\" %s, s = \"b\" %s\n" name
                (rows q)
                (rows (s_is "a"))
                (rows (s_is "b")))
            [
              ("dictionary", `Dict ([ "ok"; "\xff" ], [ 0; 0; 1; 0 ]));
              ("PLAIN", `Plain [ "ok"; "ok"; "\xff"; "ok" ]);
            ];
          expect (output ())
          @@ __POS_OF__
               {|
        x at 0 0 0: read 3 rows, x = "ok" 3 rows
        x at 0 1 0: read row group 0: column "x": row 1: invalid UTF-8 at byte 0, x = "ok" row group 0: column "x": row 1: invalid UTF-8 at byte 0
        dictionary y: read row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "a" row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "b" row group 0: column "y": row 2: invalid UTF-8 at byte 0
        PLAIN y: read row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "a" row group 0: column "y": row 2: invalid UTF-8 at byte 0, s = "b" row group 0: column "y": row 2: invalid UTF-8 at byte 0
        |});
      test "a filter on a flat column takes the rows of a nested one" (fun () ->
          let name = "nested_v1_dict.parquet" in
          let t =
            Query.of_source (P.source (sniff name) (buffer name))
            |> Query.filter Expr.(Col.string "tag" = string "t1")
            |> Query.select Expr.[ keep (Sel.names [ "people" ]) ]
            |> table
          in
          let all = cells (Error.get_ok (read (sniff name) name "people")) in
          equal (list string)
            (List.filteri (fun i _ -> i mod 3 = 1) (Array.to_list all))
            (Array.to_list (cells (Talon.column t "people"))));
    ]

(* Pruning *)

let float32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    vs;
  Bytes.to_string b

(* [stats ?nulls ?nans bytes] is a chunk's [Statistics] of minimum and maximum
   [bytes] ([min ^ max]), split in two halves. *)
let stats ?nulls ?nans bytes =
  let half = String.length bytes / 2 in
  let opt id = Option.map (fun n -> (id, I64 n)) in
  ( 12,
    Struct
      (List.filter_map Fun.id
         [
           opt 3 nulls;
           Some (5, Bin (String.sub bytes half half));
           Some (6, Bin (String.sub bytes 0 half));
           opt 9 nans;
         ]) )

(* A file of two row groups of the optional column [x]: the first holds 1 and 2
   with their statistics, and the second has the statistics [second] but a page
   that overruns its chunk, so a read fails if and only if it reads the second.
   [ty] is [x]'s physical type, [values] the first group's values, [bytes] the
   encoding of a pair of statistics. *)
let pruned ?(ty = 1) ~values ~bytes second =
  let levels = "\x02\x00\x00\x00\x04\x01" in
  let first = page 2 (levels ^ values) in
  let bad = page ~header:[ (3, I32 1000) ] 2 (levels ^ values) in
  file
    ~elements:[ field ~ty optional "x" ]
    [
      (2, [ leaf ~ty ~changes:[ stats (bytes 1 2) ] [ "x" ] ~values:2 first ]);
      (2, [ leaf ~ty ~changes:[ second ] [ "x" ] ~values:2 bad ]);
    ]

let filtered show b x ps =
  let f = Error.get_ok (P.sniff b) in
  List.iter
    (fun (name, p) ->
      let q = Query.filter p (Query.of_source (P.source f b)) in
      Printf.printf "%s: %s\n" name
        (match Query.values x q with
        | Ok vs -> String.concat " " (Array.to_list (Array.map show vs))
        | Error e -> error_text e))
    ps

let ints a b = int32s [ a; b ]

let pruning =
  group "Pruning"
    [
      test "statistics skip the row groups no row of which passes" (fun () ->
          let b =
            pruned ~values:(ints 1 2) ~bytes:ints (stats ~nulls:0 (ints 10 20))
          in
          let x = Col.int "x" in
          filtered string_of_int b x
            Expr.
              [
                ("x < 10", x < int 10);
                ("x <= 10", x <= int 10);
                ("x > 20", x > int 20);
                ("x >= 20", x >= int 20);
                ("x = 5", x = int 5);
                ("x = 15", x = int 15);
                ("x <> 10", x <> int 10);
                ("x in [3; 25]", is_in [ 3; 25 ] x);
                ("x in []", is_in [] x);
                ("not (x >= 10)", not (x >= int 10));
                ("x < 10 || x > 20", x < int 10 || x > int 20);
                ("x > 1 && x < 10", x > int 1 && x < int 10);
                ("x > 1 && not (x >= 10)", x > int 1 && not (x >= int 10));
                ("x is null", is_null x);
                ("x is not null", not (is_null x));
              ];
          expect (output ())
          @@ __POS_OF__
               {|
            x < 10: 1 2
            x <= 10: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x > 20:
            x >= 20: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x = 5:
            x = 15: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x <> 10: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x in [3; 25]:
            x in []:
            not (x >= 10): row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x < 10 || x > 20: 1 2
            x > 1 && x < 10: 2
            x > 1 && not (x >= 10): row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x is null:
            x is not null: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            |});
      test "a row group of nulls passes no comparison" (fun () ->
          let nulls = (12, Struct [ (3, I64 2) ]) in
          let b = pruned ~values:(ints 1 2) ~bytes:ints nulls in
          let x = Col.int "x" in
          filtered string_of_int b x
            Expr.
              [
                ("x > 0", x > int 0);
                ("x <> 0", x <> int 0);
                ("x is not null", not (is_null x));
                ("x is null", is_null x);
              ];
          expect (output ())
          @@ __POS_OF__
               {|
            x > 0: 1 2
            x <> 0: 1 2
            x is not null: 1 2
            x is null: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            |});
      test "float statistics skip a row group only without NaN" (fun () ->
          let floats a b = float32s [ float_of_int a; float_of_int b ] in
          let x = Col.float "x" in
          List.iter
            (fun (name, nans) ->
              let second = stats ~nulls:0 ?nans (floats 10 20) in
              let b = pruned ~ty:4 ~values:(floats 1 2) ~bytes:floats second in
              print_endline name;
              filtered string_of_float b x
                Expr.[ ("x < 5.", x < float 5.); ("x > 15.", x > float 15.) ])
            [ ("no NaN count", None); ("no NaN", Some 0); ("a NaN", Some 1) ];
          expect (output ())
          @@ __POS_OF__
               {|
            no NaN count
            x < 5.: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x > 15.: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            no NaN
            x < 5.: 1. 2.
            x > 15.: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            a NaN
            x < 5.: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            x > 15.: row group 1: bytes 33-62: column "x": the page overruns its column chunk
            |});
      test "a conjunct on a nested column prunes no row group" (fun () ->
          (* The leaf's statistics say no null, and the first row is null. *)
          let b =
            x_file ~values:2
              ~changes:[ (12, Struct [ (3, I64 0) ]) ]
              (level_page
                 ~rep:(rle [ (2, 0) ])
                 ~def:(rle [ (1, 0); (1, 3) ])
                 2 (int32s [ 7 ]))
          in
          Query.of_source (P.source (Error.get_ok (P.sniff b)) b)
          |> Query.filter Expr.(is_null (Col.v (Kind.list Kind.int) "x"))
          |> table |> Talon.rows |> equal int 1);
    ]

(* Prefixes: a slice from the start reads the pages its rows need *)

(* [head f b column k] is the first [k] rows of the column [column] of [b], read
   as [f], through a slice of its source. *)
let head f b column k =
  Query.of_source (P.source f b)
  |> Query.select Expr.[ keep (Sel.names [ column ]) ]
  |> Query.slice ~offset:0 ~length:k
  |> Query.run
  |> Result.map (fun t -> Talon.column t column)

let head_text b column k =
  match head (Error.get_ok (P.sniff b)) b column k with
  | Ok c -> cells_text c
  | Error e -> error_text e

(* [crc_failure ~values data] is a page of [values] values whose bytes [data]
   fail the CRC its header states. *)
let crc_failure ?data_page values data =
  page ~header:[ (4, I32 0) ] ?data_page values data

(* A file of one row group of 2^15 rows of tensors of 64 float64s, 16 MiB that
   encode writes in pages of about 1 MiB, uncompressed, and its format with the
   tensor declared. *)
let wide_file =
  lazy
    (let rows = 1 lsl 15 and width = 64 in
     let x =
       Nx.reshape [| rows; width |]
         (Nx.cast Nx.float64 (Nx.arange Nx.int32 0 (rows * width) 1))
     in
     let b =
       Rig.Buffer.of_string
         (written ~row_group:rows (Talon.v [ ("x", Column.of_tensor x) ]))
     in
     let ty = any (Type.tensor Nx.float64 [| width |]) in
     (P.with_type "x" ty (Error.get_ok (P.sniff b)), b))

let prefix_files =
  [
    "nested_v1_plain.parquet";
    "nested_v1_dict.parquet";
    "nested_v2_plain.parquet";
    "nested_v2_dict.parquet";
  ]

let prefixes =
  group "Prefixes"
    [
      test "the pages past a flat column's first rows are not decoded"
        (fun () ->
          let b =
            parquet ~rows:4
              (page 2 (int32s [ 1; 2 ]) ^ crc_failure 2 (int32s [ 3; 4 ]))
          in
          List.iter
            (fun k -> Printf.printf "%d rows: %s\n" k (head_text b "x" k))
            [ 2; 3 ];
          expect (output ())
          @@ __POS_OF__
               {|
            2 rows: 1 2
            3 rows: row group 0: bytes 27-51: column "x": the page fails its CRC-32 checksum
            |});
      test "a row split across pages is read whole, and no page past it"
        (fun () ->
          (* Rows [1; 2], [3; 4] and [5; 6]: the second begins in the first page
             and ends in the second, and the third ends in a page that fails its
             CRC. *)
          let pages =
            level_page
              ~rep:(rle [ (1, 0); (1, 1); (1, 0) ])
              ~def:(rle [ (3, 3) ])
              3
              (int32s [ 1; 2; 3 ])
            ^ level_page
                ~rep:(rle [ (1, 1); (1, 0) ])
                ~def:(rle [ (2, 3) ])
                2
                (int32s [ 4; 5 ])
            ^ crc_failure
                ~data_page:[ (4, I32 3) ]
                1
                (prefixed (rle [ (1, 1) ])
                ^ prefixed (rle [ (1, 3) ])
                ^ int32s [ 6 ])
          in
          let b = x_file ~rows:3 ~values:6 pages in
          List.iter
            (fun k -> Printf.printf "%d rows: %s\n" k (head_text b "x" k))
            [ 1; 2; 3 ];
          expect (output ())
          @@ __POS_OF__
               {|
            1 rows: [1, 2]
            2 rows: [1, 2] [3, 4]
            3 rows: row group 0: bytes 88-122: column "x.list.element": the page fails its CRC-32 checksum
            |});
      test "a header past a nested column's first rows is not read" (fun () ->
          let pages =
            level_page
              ~rep:(rle [ (1, 0); (1, 1); (1, 0) ])
              ~def:(rle [ (3, 3) ])
              3
              (int32s [ 1; 2; 3 ])
            ^ level_page
                ~rep:(rle [ (1, 1); (1, 0) ])
                ~def:(rle [ (2, 3) ])
                2
                (int32s [ 4; 5 ])
            ^ "\x15\xff\xff\xff\xff\xff\xff\xff\xff\xff"
          in
          equal string "[1, 2] [3, 4]"
            (head_text (x_file ~rows:3 ~values:5 pages) "x" 2));
      test "a tensor's first rows read when a later row breaks its shape"
        (fun () ->
          (* Rows 0 to 4 of [ragged] are [image]'s, and row 5 breaks. *)
          let f = P.with_type "ragged" (any image) (sniff tensors) in
          let first k =
            Result.map_error error_text
              (Result.map cells (head f (buffer tensors) "ragged" k))
          in
          let image_rows = cells (Error.get_ok (read_as "image" image)) in
          let cs = result (array string) string in
          equal cs (Ok (Array.sub image_rows 0 5)) (first 5);
          equal cs
            (Error
               "row group 1: column \"ragged\": row 1: a list of level 2 holds \
                3 elements, not the shape's 4")
            (first 6));
      test "the first rows hold the memory of the pages they read" (fun () ->
          (* Twice the writer's page of about 1 MiB, and one more page: the row
             group's 16 MiB stays unread. *)
          let f, b = Lazy.force wide_file in
          let read () = Error.get_ok (head f b "x" 8) in
          ignore (read ());
          let before = Talon_held.settled () in
          let c = read () in
          let held = Talon_held.settled () - before in
          equal int 8 (Column.length c);
          less int ~than:(3 lsl 20) held);
      (* The fixtures hold 40 rows in row groups of 13. *)
      prop "a slice from the start reads a file's first rows"
        ~examples:
          [
            ("nested_v1_plain.parquet", 0, 0);
            ("nested_v1_dict.parquet", 1, 13);
            ("nested_v2_plain.parquet", 2, 45);
          ]
        Gen.(
          triple (of_list prefix_files) (int_range 0 9)
            (frequency
               [
                 (1, constant 0);
                 (1, of_list [ 13; 26; 39 ]);
                 (1, int_range 41 45);
                 (4, int_range 1 40);
               ]))
        (fun (file, column, k) ->
          cover "no row" (k = 0);
          cover "inside a row group" (k mod 13 <> 0 && k < 40);
          cover "a row group's end" (k > 0 && k mod 13 = 0);
          cover "past the last row" (k > 40);
          let f = sniff file in
          let source = P.source f (buffer file) in
          let columns =
            Schema.columns (Query.schema (Query.of_source source))
          in
          let c = fst (List.nth columns (column mod List.length columns)) in
          let all = cells (Error.get_ok (read f file c)) in
          let want = Array.sub all 0 (min k (Array.length all)) in
          match head f (buffer file) c k with
          | Ok got -> equal (array string) want (cells got)
          | Error e -> failf "%s" (error_text e));
    ]

(* Failure *)

let failure =
  group "Failure"
    [
      test "a column chunk that fails to decode fails the read" (fun () ->
          let show name ?(f = sniff name) column =
            Printf.printf "%s %s: %s\n" name column
              (outcome (read f name column))
          in
          show "datapage_v1-corrupt-checksum.parquet" "a";
          show "rle-dict-uncompressed-corrupt-checksum.parquet" "binary_field";
          show "bad_data/ARROW-GH-47662.parquet" "flba_field";
          show "int96_from_spark.parquet" "a";
          show "int96_from_spark.parquet" "a"
            ~f:
              (P.with_type "a"
                 (any (Type.wall Us))
                 (sniff "int96_from_spark.parquet"));
          show "int96.parquet" "ts_ns"
            ~f:
              (P.with_type "ts_ns" (any (Type.wall Us)) (sniff "int96.parquet"));
          show "bad_encoding.parquet" "i32";
          expect (output ())
          @@ __POS_OF__
               {|
        datapage_v1-corrupt-checksum.parquet a: row group 0: bytes 4-10271: column "a": the page fails its CRC-32 checksum
        rle-dict-uncompressed-corrupt-checksum.parquet binary_field: row group 0: bytes 57-115: column "binary_field": the page fails its CRC-32 checksum
        bad_data/ARROW-GH-47662.parquet flba_field: row group 0: bytes 4-393: column "flba_field": the PLAIN values take 400 bytes of the page's 364
        int96_from_spark.parquet a: row group 0: column "a": an int96 timestamp is outside the range of datetime[ns]. Read the column as a datetime of a coarser unit (with_type).
        int96_from_spark.parquet a: row group 0: column "a": an int96 timestamp is outside the range of datetime[us]. Read the column as a datetime of a coarser unit (with_type).
        int96.parquet ts_ns: row group 0: column "ts_ns": an int96 timestamp is not a whole number of microseconds. Read the column as a datetime of a finer unit (with_type).
        bad_encoding.parquet i32: row group 0: bytes 4-360: column "i32": the values are in encoding 15, which Parquet does not define
        |});
      cases "a flat column's failure names its column"
        ~id:(fun (f, c, _) -> f ^ " " ^ c)
        [
          ("fallback.parquet", "string", any (Type.categorical [| "x" |]));
          ("datapage_v1-corrupt-checksum.parquet", "a", any Type.int32);
          ("int96_from_spark.parquet", "a", any (Type.wall Ns));
        ]
        (fun (file, column, ty) ->
          match read (P.with_type column ty (sniff file)) file column with
          | Ok _ -> failf "%s read" column
          | Error e ->
              contains ~sub:(Printf.sprintf "column %S" column) (error_text e));
      test "a read of a column the file lacks fails" (fun () ->
          equal string {|the file has no column "a b"|}
            (outcome (read (sniff "names.parquet") "types.parquet" "a b")));
      test "a dictionary index past the dictionary fails" (fun () ->
          (* The words "a" and "b", then two indices at bit width 3: an RLE run
             of 2 of index 5. *)
          let dict = dictionary_page [ "a"; "b" ] in
          let data = page ~data_page:[ (2, I32 8) ] 2 "\x03\x04\x05" in
          let b =
            parquet ~rows:2
              ~column:[ (1, I32 6) ]
              ~dictionary:(String.length dict) (dict ^ data)
          in
          expect (outcome (read_x b))
          @@ __POS_OF__
               {| row group 0: bytes 27-44: column "x": the dictionary indices: a value is out of its bounds |});
      test "an RLE boolean run of value 2 fails" (fun () ->
          (* Four booleans: a 4-byte length, then an RLE run of 4 of value 2. *)
          let b =
            parquet ~rows:4
              ~column:[ (1, I32 0) ]
              (page ~data_page:[ (2, I32 3) ] 4 "\x02\x00\x00\x00\x08\x02")
          in
          expect (outcome (read_x b))
          @@ __POS_OF__
               {| row group 0: bytes 4-24: column "x": the RLE booleans: a value is out of its bounds |});
      test "an int8 that does not fit names its row" (fun () ->
          (* Rows 0, 2 and 3 hold 1, 2 and 300: the levels are a bit-packed run
             of 1, 0, 1, 1. *)
          let levels = "\x02\x00\x00\x00\x03\x0d" in
          let b =
            parquet ~rows:4
              ~column:[ (3, I32 optional); (6, I32 15) ]
              (page 4 (levels ^ int32s [ 1; 2; 300 ]))
          in
          expect (outcome (read_x b))
          @@ __POS_OF__
               {| row group 0: column "x": the value 300 of row 3 does not fit int8 |});
      test "sizes at the limits of their types fail" (fun () ->
          let show what b =
            Printf.printf "%s: %s\n" what (outcome (read_x b))
          in
          let two = int32s [ 7; 8 ] in
          let header changes = parquet ~rows:2 (page ~header:changes 2 two) in
          show "a page of i32_max bytes" (header [ (3, I32 i32_max) ]);
          show "a page of 2^31 bytes" (header [ (3, I32 (i32_max + 1)) ]);
          show "a page size of wire type i64" (header [ (3, I64 8) ]);
          show "levels of twice i32_max bytes"
            (header
               [
                 (1, I32 3);
                 ( 8,
                   Struct
                     [
                       (1, I32 2);
                       (2, I32 0);
                       (3, I32 2);
                       (4, I32 0);
                       (5, I32 i32_max);
                       (6, I32 i32_max);
                     ] );
               ]);
          show "a chunk of max_int bytes"
            (parquet ~rows:2 ~chunk:[ (7, I64 max_int) ] (page 2 two));
          show "a chunk at max_int"
            (parquet ~rows:2
               ~chunk:[ (7, I64 max_int); (9, I64 max_int) ]
               (page 2 two));
          show "2^40 rows of i32_max bytes"
            (parquet ~rows:(1 lsl 40)
               ~column:[ (1, I32 7); (2, I32 i32_max) ]
               (page 2 two));
          expect (output ())
          @@ __POS_OF__
               {|
            a page of i32_max bytes: row group 0: bytes 4-30: column "x": the page overruns its column chunk
            a page of 2^31 bytes: row group 0: byte 9: column "x": a page header is malformed: an i32 outside its range, 2147483648
            a page size of wire type i64: row group 0: byte 9: column "x": a page header is malformed: expected an i32, found wire type 6
            levels of twice i32_max bytes: row group 0: bytes 4-48: column "x": the levels overrun the page
            a chunk of max_int bytes: row group 0: bytes 4-4611686018427387902: column "x": its pages overrun the file, whose footer starts at byte 27
            a chunk at max_int: row group 0: byte 4611686018427387903: column "x": its pages overrun the file, whose footer starts at byte 27
            2^40 rows of i32_max bytes: row group 0: bytes 4-26: column "x": the chunk's 1099511627776 values do not fit in memory
            |});
      (* Eight int32s compressed with a codec of parquet.thrift's
         CompressionCodec, in a page whose header states their size. *)
      cases "pages of each codec decode" ~id:fst
        (let raw = int32s [ 1; 2; 3; 4; 5; 6; 7; 8 ] in
         [
           ("snappy", (1, Lz.Snappy.encode_string raw));
           ( "gzip of two members",
             ( 2,
               Lz.Gzip.encode_string (String.sub raw 0 12)
               ^ Lz.Gzip.encode_string (String.sub raw 12 20) ) );
           ("zstd", (6, Lz.Zstd.encode_string raw));
           ("lz4_raw", (7, Lz.Lz4.Block.encode_string raw));
         ])
        (fun (_, (codec, data)) ->
          let b =
            parquet ~rows:8
              ~chunk:[ (4, I32 codec); (6, I64 1000) ]
              (page ~header:[ (2, I32 32) ] 8 data)
          in
          equal
            (array (option int))
            (Array.init 8 (fun i -> Some (i + 1)))
            (Column.options Kind.int (Error.get_ok (read_x b))));
      test "a page that does not decompress fails, giving its bytes" (fun () ->
          let raw = int32s [ 1; 2; 3; 4; 5; 6; 7; 8 ] in
          let show ?(size = String.length raw) what codec data =
            let b =
              parquet ~rows:8
                ~chunk:[ (4, I32 codec); (6, I64 1000) ]
                (page ~header:[ (2, I32 size) ] 8 data)
            in
            Printf.printf "%s (%d bytes): %s\n" what (String.length data)
              (outcome (read_x b))
          in
          let snappy = Lz.Snappy.encode_string raw
          and zstd = Lz.Zstd.encode_string raw
          and gzip = Lz.Gzip.encode_string raw
          and lz4 = Lz.Lz4.Block.encode_string raw in
          let cut s = String.sub s 0 (String.length s - 1) in
          let flip_last s =
            let b = Bytes.of_string s in
            let i = Bytes.length b - 1 in
            Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor 1);
            Bytes.to_string b
          in
          show "zstd with a wrong checksum" 6 (flip_last zstd);
          show "gzip cut short" 2 (cut gzip);
          show "lz4_raw cut short" 7 (cut lz4);
          show "snappy of fewer bytes than its header" 1
            (Lz.Snappy.encode_string (String.sub raw 0 16));
          show "snappy of more bytes than its header" ~size:16 1 snappy;
          show "snappy followed by a byte" 1 (snappy ^ "\x00");
          show "lz4_raw followed by a byte" 7 (lz4 ^ "\x00");
          expect (output ())
          @@ __POS_OF__
               {|
            zstd with a wrong checksum (30 bytes): row group 0: bytes 4-48: column "x": the page does not decompress: malformed at byte 26: Content_Checksum mismatch
            gzip cut short (41 bytes): row group 0: bytes 4-59: column "x": the page does not decompress: truncated at byte 41
            lz4_raw cut short (33 bytes): row group 0: bytes 4-51: column "x": the page does not decompress: truncated at byte 33
            snappy of fewer bytes than its header (18 bytes): row group 0: bytes 4-36: column "x": the page decompresses to 16 bytes, not its header's 32
            snappy of more bytes than its header (34 bytes): row group 0: bytes 4-52: column "x": the page does not decompress: output too long at byte 0
            snappy followed by a byte (35 bytes): row group 0: bytes 4-53: column "x": the page does not decompress: malformed at byte 34: output longer than the preamble
            lz4_raw followed by a byte (35 bytes): row group 0: bytes 4-53: column "x": the page does not decompress: truncated at byte 35
            |});
      test "a nested column's malformed levels fail, naming its leaf" (fun () ->
          let x ?rows ?claim ~rep ~def values data =
            let claim = Option.value claim ~default:values in
            x_file ?rows ~values:claim
              (level_page ~rep:(rle rep) ~def:(rle def) values data)
          in
          let show name b =
            Printf.printf "%s: %s\n" name (outcome (read_x b))
          in
          show "a level past the maximum"
            (x ~rep:[ (2, 2) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "a first entry that does not start a row"
            (x ~rep:[ (2, 1) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "an entry repeating an undefined list"
            (x ~rep:[ (1, 0); (1, 1) ] ~def:[ (1, 3); (1, 1) ] 2 (int32s [ 1 ]));
          show "pages that do not add up to the chunk"
            (x ~claim:5 ~rep:[ (2, 0) ] ~def:[ (2, 3) ] 2 (int32s [ 1; 2 ]));
          show "fewer rows than the row group"
            (x ~rows:3
               ~rep:[ (1, 0); (2, 1) ]
               ~def:[ (3, 3) ]
               3
               (int32s [ 1; 2; 3 ]));
          show "two fields of a record that disagree"
            (let chunk name rep =
               leaf [ "x"; name ] ~values:2
                 (level_page ~rep:(rle rep)
                    ~def:(rle [ (2, 1) ])
                    2
                    (int32s [ 1; 2 ]))
             in
             let f = field repeated in
             file
               ~elements:[ node required "x" 2; f "a"; f "b" ]
               [ (2, [ chunk "a" [ (2, 0) ]; chunk "b" [ (1, 0); (1, 1) ] ]) ]);
          expect (output ())
          @@ __POS_OF__
               {|
            a level past the maximum: row group 0: bytes 4-40: column "x.list.element": the repetition levels: a value is out of its bounds
            a first entry that does not start a row: row group 0: column "x.list.element": the levels: the first entry does not start a row
            an entry repeating an undefined list: row group 0: column "x": the levels: an entry repeats a field that is not defined
            pages that do not add up to the chunk: row group 0: bytes 4-40: column "x.list.element": the pages hold 2 values, and the column chunk 5
            fewer rows than the row group: row group 0: column "x" holds 1 rows in a row group of 3
            two fields of a record that disagree: row group 0: column "x": field "b" holds 1 values for the record's 2
            |});
      cases "a row that breaks a tensor's shape fails, naming it"
        ~id:(fun (column, _) -> column)
        [
          ( "ragged",
            "row group 1: column \"ragged\": row 1: a list of level 2 holds 3 \
             elements, not the shape's 4" );
          ( "wide",
            "row group 1: column \"wide\": row 2: a list of level 2 holds 5 \
             elements, not the shape's 4" );
          ( "null_list",
            "row group 0: column \"null_list\": row 1: a list of level 1 is \
             null" );
          ( "null_value",
            "row group 1: column \"null_value\": row 3: a value of the tensor \
             is null" );
          ( "bands",
            "row group 0: column \"bands\": row 0: a list of level 0 holds 0 \
             elements, not the shape's 2" );
          (* Row 1 fails at level 2 and row 3 at level 0: the earlier row is
             named. *)
          ( "order",
            "row group 0: column \"order\": row 1: a list of level 2 holds 3 \
             elements, not the shape's 4" );
        ]
        (fun (column, want) ->
          match read_as column image with
          | Ok _ -> failf "column %S read" column
          | Error e -> equal string want (error_text e));
    ]

(* Memory *)

(* nulls_dict.parquet's column [x]: 2^16 int64s in one row group, a tenth of
   them null, in a dictionary of 97 values. *)
let nulls_rows = 1 lsl 16

(* Reads of one source, which keeps the buffer its pages decompress into. *)
let nulls_source =
  lazy (P.source (sniff "nulls_dict.parquet") (buffer "nulls_dict.parquet"))

let read_nulls () =
  Talon.column (table (Query.of_source (Lazy.force nulls_source))) "x"

(* [with_ratio r f] is [f ()] with the collector's [custom_major_ratio] at
   [r]. *)
let with_ratio r f =
  let was = (Gc.get ()).custom_major_ratio in
  Gc.set { (Gc.get ()) with custom_major_ratio = r };
  Fun.protect f ~finally:(fun () ->
      Gc.set { (Gc.get ()) with custom_major_ratio = was })

(* The decoder's shapes: every column of the types fixture in one row group,
   with dictionary pages and with plain pages, its decimals read both ways, its
   text read as categoricals, int96 timestamps, and a file of no row. *)
let shapes =
  let columns =
    [
      "bool";
      "i8";
      "i16";
      "i32";
      "i64";
      "u8";
      "u16";
      "u32";
      "u64";
      "f16";
      "f32";
      "f64";
      "date";
      "clock_ms";
      "clock_us";
      "clock_ns";
      "ts_ms";
      "ts_us_utc";
      "ts_us_paris";
      "ts_ns";
      "string";
      "binary";
      "fsb";
      "uuid";
      "all_null";
      "no_null";
      "req_i32";
      "req_string";
    ]
  in
  let texts = Array.init 40 (fun i -> Printf.sprintf "v%d\xc3\xa9\n\"" i) in
  let typed =
    [
      ("d9", Some (any Type.float64));
      ("d9", Some (any Type.int64));
      ("d18", Some (any Type.float64));
      ("d18", Some (any Type.int64));
      ("string", Some (any (Type.categorical texts)));
      ( "req_string",
        Some (any (Type.categorical (Array.init 7 (Printf.sprintf "r%d")))) );
    ]
  in
  let file name =
    List.map (fun c -> (name, c, None)) columns
    @ List.map (fun (c, ty) -> (name, c, ty)) typed
  in
  file "one_group.parquet"
  @ file "one_group_plain.parquet"
  @ [
      ("one_group_int96.parquet", "ts_ns", None);
      ("empty.parquet", "i64", None);
      ("empty.parquet", "string", None);
    ]

let memory =
  group "Memory"
    [
      (* A column is host memory: the host holds its arrays' bytes while it is
         live, and has them back once it is collected. Each case is a shape of
         the decoder, in one row group so that the column is that chunk's
         arrays: validity bits and the spread of values and offsets under nulls,
         dictionary indices gathered as values and as byte strings, plain byte
         strings, booleans, int96 timestamps, decimals as unscaled integers and
         as floats, categoricals, and no row. *)
      cases "a column is the host's memory while it is live" shapes
        ~id:(fun (file, column, ty) ->
          Printf.sprintf "%s %s%s" file column
            (match ty with
            | None -> ""
            | Some (Type.Any t) -> Format.asprintf " as %a" Type.pp t))
        (fun (file, column, ty) ->
          let f =
            if file = "one_group_int96.parquet" then sniff file
            else declare_decimals (sniff file)
          in
          let f =
            match ty with Some ty -> P.with_type column ty f | None -> f
          in
          let b = buffer file in
          Talon_held.check (fun () -> Error.get_ok (read_from f b column)));
      (* Dropped columns pace the collector as rig's host buffers do, whatever
         [custom_major_ratio] says of the heap: the bytes of host buffers that
         reach the major heap bring a cycle's work per bound of at least 4 MiB,
         and a buffer that starts young with more than [custom_minor_max_size]
         bytes brings an eighth of its bytes at birth as well. A read of the
         column of nulls and a dictionary makes four arrays of about its 2^16
         rows of 8 bytes, 2 MiB at most together, the table of values, the
         indices, the gathered values and the spread ones, each starting young,
         and its validity, of 8 KiB; it decompresses into a buffer its source
         keeps. Held across a minor collection, the column's spread values and
         validity reach the major heap, and the other arrays die young. So 64
         reads bring at most a cycle's work per 4 MiB of the column's arrays and
         of an eighth of the four. The count starts from a compaction, which
         leaves the heap about its live words: a cycle's work, which the runtime
         sizes as sweeping the heap and marking part of it, then completes at
         most a cycle, and two more are under way at either end. At a ratio of
         1, a bigarray paced by the heap brings a cycle's work each. *)
      test "dropped columns pace the collector by the program's memory"
        (fun () ->
          let reads = 64 in
          let c =
            with_ratio 1 @@ fun () ->
            ignore (read_nulls ());
            Talon_held.cycles (fun () ->
                for _ = 1 to reads do
                  let held = read_nulls () in
                  Gc.minor ();
                  ignore (Sys.opaque_identity held)
                done)
          in
          let column = (8 * nulls_rows) + (nulls_rows / 8) in
          let born = 4 * 8 * nulls_rows in
          let work = reads * (column + (born / 8)) in
          at_most
            ~msg:("major cycles at a ratio of 1: " ^ Talon_held.pp_cycles c)
            int
            ~than:((work / (4 lsl 20)) + 2)
            c.majors);
    ]

(* source *)

let source =
  group "source"
    [
      test "source and file raise naming a decimal left undeclared" (fun () ->
          let name = "decimals_bytes.parquet" in
          let f = P.with_type "d9_0" (any Type.float64) (sniff name) in
          print_endline (refusal (fun () -> P.source f (buffer name)));
          print_endline (refusal (fun () -> P.file (path name)));
          expect (output ())
          @@ __POS_OF__
               {|
            Talon_parquet.source: column "d9_2" (optional fixed_len_byte_array(4) (DECIMAL(9,2))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            Talon_parquet.file: column "d9_0" (optional fixed_len_byte_array(4) (DECIMAL(9,0))) is a decimal, which reads as float64, or as its unscaled int64 integers up to 18 digits. Declare one with with_type.
            |});
      (* One file, its column named [a] then [b], which are one length: a source
         of [a]'s format fails once the bytes are [b]'s. *)
      test "each read reads the footer again" (fun () ->
          let bytes name =
            written (Talon.v [ (name, Column.v Type.int64 [| 1; 2 |]) ])
          in
          let a = bytes "a" and b = bytes "b" in
          let ba =
            Bigarray.Array1.create Bigarray.char Bigarray.c_layout
              (String.length a)
          in
          String.iteri (fun i c -> ba.{i} <- c) a;
          let buf = Rig.Buffer.of_bigarray ba in
          let s = Query.of_source (P.source (Error.get_ok (P.sniff buf)) buf) in
          equal int 2 (Talon.rows (table s));
          String.iteri (fun i c -> ba.{i} <- c) b;
          equal string {|the file has no column "a"|} (outcome (Query.run s)));
    ]

(* file *)

let rec remove path =
  if Sys.is_directory path then begin
    Array.iter (fun name -> remove (path ^ "/" ^ name)) (Sys.readdir path);
    Sys.rmdir path
  end
  else Sys.remove path

let write_part path rows =
  Sys.mkdir (Filename.dirname path) 0o755;
  Out_channel.with_open_bin path @@ fun oc ->
  let w = Bytesrw.Bytes.Writer.of_out_channel oc in
  let t = Talon.v [ ("x", Column.v Type.int64 rows) ] in
  Result.get_ok (P.encode (P.format (Talon.schema t)) (Query.of_table t) w)

(* [occurrences w s] is the number of times [w] occurs in [s]. *)
let occurrences w s =
  let n = String.length w and m = String.length s in
  let rec at i acc =
    if i + n > m then acc
    else if String.sub s i n = w then at (i + 1) (acc + 1)
    else at (i + 1) acc
  in
  at 0 0

let many_files () =
  let root = Filename.temp_dir "talon-many" "" in
  Fun.protect ~finally:(fun () -> remove root) @@ fun () ->
  let n = 1000 in
  let rows_of i = [| i * 10; (i * 10) + 1 |] in
  let paths =
    List.init n (fun i ->
        let path = Printf.sprintf "%s/part=%04d/file.parquet" root i in
        write_part path (rows_of i);
        path)
  in
  let manifest = Error.get_ok (Source.hive paths) in
  let branch path part =
    Query.of_source (Error.get_ok (P.file path))
    |> Query.derive Expr.[ "part" := int part ]
  in
  let parts = Column.options Kind.string (column manifest "part") in
  let queries =
    List.mapi
      (fun i path -> branch path (int_of_string (Option.get parts.(i))))
      paths
  in
  let reference =
    let xs = Array.concat (List.init n rows_of) in
    let part = Array.concat (List.init n (fun i -> [| i; i |])) in
    Talon.v
      [ ("x", Column.v Type.int64 xs); ("part", Column.v Type.int64 part) ]
  in
  let all = Kit.union_all queries in
  equal ~msg:"every file's rows, in order" table_w reference (table all);
  let by_partition = Expr.(Col.int "part" = int 7 || Col.int "x" > int 9990) in
  equal ~msg:"a filter mixing a partition and a file column" table_w
    (table (Query.filter by_partition (Query.of_table reference)))
    (table (Query.filter by_partition all));
  (* A partition filter empties every branch but the one it keeps. *)
  let kept = Query.filter Expr.(Col.int "part" = int 7) all in
  equal ~msg:"a filter on a partition column" table_w
    (table
       (Query.filter Expr.(Col.int "part" = int 7) (Query.of_table reference)))
    (table kept);
  equal ~msg:"branches the filter excludes are never read" int (n - 1)
    (occurrences "~limit:0"
       (Format.asprintf "%a" Query.pp (Query.optimize kept)))

let files_group =
  group "file"
    [
      test "is named after its path and states its rows" (fun () ->
          let plan s = Format.asprintf "%a" Query.pp (Query.of_source s) in
          print_endline (plan (Error.get_ok (P.file (path "names.parquet"))));
          print_endline
            (plan (P.source (sniff "names.parquet") (buffer "names.parquet")));
          expect (output ())
          @@ __POS_OF__
               {|
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet "support/names.parquet" (4 columns, 2 rows)
            query → "a b" int64, "\"q\"" int64, é int64, "" int64
            parquet (4 columns)
            |});
      test "refuses what sniff refuses, naming the file" (fun () ->
          List.iter
            (fun name -> print_endline (outcome (P.file (path name))))
            [ "brotli.parquet"; "missing.parquet" ];
          expect (output ())
          @@ __POS_OF__
               {|
            support/brotli.parquet: row group 0: column "i32" is compressed with Brotli, which talon does not read
            support/missing.parquet: No such file or directory
            |});
      test "a union of a thousand files reads every file's rows" many_files;
    ]

let () =
  exit
    (run "talon.parquet"
       [
         types;
         nested;
         tensor_columns;
         group "Refusals" [ refusals ];
         sniffing;
         with_type;
         pp_format;
         reading;
         text_conjuncts;
         pruning;
         prefixes;
         failure;
         memory;
         source;
         files_group;
       ])
