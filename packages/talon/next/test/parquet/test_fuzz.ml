(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Malformed Parquet files: each decode is a table or an error, never an
   exception, a crash or a hang. CI runs this suite with the C built under the
   sanitizers, so that a decoder that reads or writes out of bounds fails
   there. *)

open Talon_next
open Windtrap
module P = Talon_next_parquet

(* Corpus *)

(* One file per page version, codec and encoding, with the columns each file
   holds as decimals, which reading needs declared. The uncompressed files reach
   the page decoders with every changed byte; the compressed ones reach the
   decompressors. *)
let corpus =
  [
    ("types_plain.parquet", [ "d9"; "d18" ]);
    ("encodings_uncompressed.parquet", [ "d9" ]);
    ("alltypes_dictionary.parquet", []);
    ("data_index_bloom_encoding_with_length.parquet", []);
    ("delta_binary_packed.parquet", []);
    ("delta_byte_array.parquet", []);
    ("delta_encoding_optional_column.parquet", []);
    ("int32_with_null_pages.parquet", []);
    ("fixed_length_byte_array.parquet", []);
    ("float16_nonzeros_and_nans.parquet", []);
    ("nan_in_stats.parquet", []);
    ("types.parquet", [ "d9"; "d18" ]);
    ("types_v2_zstd.parquet", [ "d9"; "d18" ]);
    ("types_gzip.parquet", [ "d18" ]);
    ("types_lz4.parquet", [ "d18" ]);
  ]

let file_bytes =
  let files =
    List.map
      (fun (name, _) ->
        let path = Filename.concat "fixtures" name in
        (name, In_channel.with_open_bin path In_channel.input_all))
      corpus
  in
  fun name -> List.assoc name files

let names = List.map fst corpus
let pick = Gen.of_list ~pp:Format.pp_print_string names

let of_string s =
  Nx_device.Buffer.of_bigarray
    (Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
       (String.length s) (fun i -> Char.code s.[i]))

(* [footer s] is the position of the footer of the Parquet file [s]. *)
let footer s =
  let n = String.length s in
  n - 8 - Int32.to_int (String.get_int32_le s (n - 8))

let le32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.to_string b

(* Decoding *)

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t
let format_text f = Format.asprintf "%a" P.pp_format f

(* [undeclared f] is [true] iff [f] has a decimal column that no [with_type]
   declared, which [P.source] refuses. *)
let undeclared f =
  List.exists
    (fun l -> List.mem "undeclared" (String.split_on_char ' ' l))
    (String.split_on_char '\n' (format_text f))

(* [declare decimals f] declares the columns [decimals] of [f] as [float64],
   when the file still has them as decimals. *)
let declare decimals f =
  List.fold_left
    (fun f c ->
      match P.with_type c (Type.Any Type.float64) f with
      | f -> f
      | exception Invalid_argument _ -> f)
    f decimals

(* [conjuncts schema] compares each numeric and string column of [schema] with a
   constant, so that row group statistics decide each conjunct. *)
let conjuncts schema =
  List.filter_map
    (fun (name, t) ->
      match type_name t with
      | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
      | "uint64" ->
          Some Expr.(Col.int name > int 0)
      | "float16" | "float32" | "float64" ->
          Some Expr.(Col.float name < float 0.)
      | "string" -> Some Expr.(Col.string name >= string "m")
      | _ -> None)
    (Schema.columns schema)

(* [decode decimals s] reads every column of the file [s], then the rows that
   pass a conjunct on each column, as sniffing [s] says, and is how far the read
   went. A read is a table or an error; [P.source] raises only on an undeclared
   decimal, as it documents. *)
let decode decimals s =
  let b = of_string s in
  match P.sniff b with
  | Error _ -> "sniff fails"
  | Ok f -> (
      let f = declare decimals f in
      match P.source f b with
      | exception Invalid_argument _ when undeclared f ->
          "an undeclared decimal"
      | src ->
          let q = Query.of_source src in
          let read = Query.run q in
          (match conjuncts (Query.schema q) with
          | [] -> ()
          | p :: ps ->
              let p = List.fold_left Expr.( && ) p ps in
              ignore (Query.run (Query.filter p q)));
          if Result.is_ok read then "the file reads" else "the read fails")

(* [decode_mutated name mutate] decodes the file [name] mutated by [mutate]. The
   case must reach the page decoders: [cover] fails the property when no case
   sniffs and then fails to read. *)
let decode_mutated name mutate =
  let outcome = decode (List.assoc name corpus) (mutate (file_bytes name)) in
  collect outcome;
  cover "the read fails" (outcome = "the read fails")

(* Mutations *)

(* Where a mutation lands: in the footer, which includes the eight bytes that
   end the file, or in the pages before it. *)
type region = Footer | Pages

let pp_region ppf r =
  Format.pp_print_string ppf
    (match r with Footer -> "footer" | Pages -> "pages")

let region = Gen.of_list ~pp:pp_region [ Footer; Pages ]

(* [locate r s k] is a position of the region [r] of the file [s], for any [k >=
   0]. *)
let locate r s k =
  let first = footer s and n = String.length s in
  match r with
  | Footer -> first + (k mod (n - first))
  | Pages -> 4 + (k mod (first - 4))

let flip r bits s =
  let b = Bytes.of_string s in
  List.iter
    (fun (k, bit) ->
      let at = locate r s k in
      Bytes.set_uint8 b at (Bytes.get_uint8 b at lxor (1 lsl bit)))
    bits;
  Bytes.to_string b

let excise r k len s =
  let at = locate r s k in
  let len = min len (String.length s - at) in
  String.sub s 0 at ^ String.sub s (at + len) (String.length s - at - len)

(* [splice a b] is the pages of the file [a] followed by the footer of the file
   [b], its length and its magic. *)
let splice a b =
  String.sub a 0 (footer a)
  ^ String.sub b (footer b) (String.length b - footer b)

(* [cut_footer s k] is the file [s] with the last [k] bytes of its footer cut,
   and the footer's length rewritten. *)
let cut_footer s k =
  let first = footer s in
  let len = String.length s - 8 - first - k in
  String.sub s 0 (first + len) ^ le32 len ^ "PAR1"

(* Lengths, sizes and counts, Thrift's and the pages', written over the bytes of
   a file in each of the forms that store them: a Thrift varint, unsigned
   (lengths of binaries and lists) or zigzag (i32 and i64 fields), and the
   little-endian 4 bytes that prefix levels and PLAIN byte arrays. *)
type form = Varint | Zigzag | Le32

let pp_form ppf f =
  Format.pp_print_string ppf
    (match f with Varint -> "varint" | Zigzag -> "zigzag" | Le32 -> "le32")

let form = Gen.of_list ~pp:pp_form [ Varint; Zigzag; Le32 ]

let oversized =
  Gen.of_list ~pp:Format.pp_print_int
    [ 0x7FFF_FFFF; 0x8000_0000; 0xFFFF_FFFF; 1 lsl 40; max_int; -1; min_int ]

let rec varint b n =
  if Int64.logand n (-128L) = 0L then Buffer.add_uint8 b (Int64.to_int n)
  else begin
    Buffer.add_uint8 b (Int64.to_int (Int64.logand n 0x7FL) lor 0x80);
    varint b (Int64.shift_right_logical n 7)
  end

let encode form n =
  let n64 = Int64.of_int n in
  match form with
  | Le32 -> le32 n
  | Varint | Zigzag ->
      let n64 =
        if form = Varint then n64
        else Int64.logxor (Int64.shift_left n64 1) (Int64.shift_right n64 63)
      in
      let b = Buffer.create 10 in
      varint b n64;
      Buffer.contents b

let overwrite r k bytes s =
  let at = locate r s k in
  let len = min (String.length bytes) (String.length s - at) in
  let b = Bytes.of_string s in
  Bytes.blit_string bytes 0 b at len;
  Bytes.to_string b

(* Properties *)

let bits = Gen.(list ~size:(int_range 1 8) (pair nat (int_range 0 7)))

let mutated =
  group ~timeout:600. "Mutated files decode to a table or an error"
    [
      prop "with bits flipped"
        Gen.(triple pick region bits)
        (fun (name, r, bits) -> decode_mutated name (flip r bits));
      prop "cut short"
        Gen.(pair pick nat)
        (fun (name, k) ->
          let s = file_bytes name in
          collect
            (decode (List.assoc name corpus)
               (String.sub s 0 (k mod String.length s))));
      prop "with bytes removed"
        Gen.(quad pick region nat (int_range 1 64))
        (fun (name, r, k, len) -> decode_mutated name (excise r k len));
      prop "with another file's footer"
        Gen.(pair pick pick)
        (fun (a, b) -> decode_mutated a (fun s -> splice s (file_bytes b)));
      prop "with an oversized length"
        Gen.(quad pick region nat (pair form oversized))
        (fun (name, r, k, (form, n)) ->
          decode_mutated name (overwrite r k (encode form n)));
    ]

let footers =
  prop "A footer cut short fails sniff"
    Gen.(pair pick nat)
    (fun (name, k) ->
      let s = file_bytes name in
      let len = String.length s - 8 - footer s in
      is_error (P.sniff (of_string (cut_footer s (1 + (k mod len))))))

let () = exit (run "talon.next.parquet fuzz" [ mutated; footers ])
