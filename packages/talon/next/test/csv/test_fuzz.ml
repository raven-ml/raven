(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Malformed CSV files: each sniff and decode is a format, a table or an error,
   never an exception or a hang, however the reader slices the input. *)

open Talon_next
open Windtrap
module Csv = Talon_next_csv
module Reader = Bytesrw.Bytes.Reader

(* Corpus *)

(* Files of each dialect and of every type sniffing infers, with the syntax's
   corners: quoted line breaks, doubled quotes, carriage returns, a byte order
   mark, empty lines, null tokens and fields of every width. *)
let corpus =
  [
    ( "types",
      "id,flag,score,ratio,day,at,at_utc,name\n\
       1,true,3,0.5,2024-02-29,2024-02-29T12:00:00,2024-02-29T12:00:00+01:00,\"a\"\n\
       -9223372036854775808,false,-7,-0.0,1970-01-01,1970-01-01T00:00:00.000000001,1970-01-01T00:00:00Z,\"b, \
       \"\"c\"\"\"\n\
       9223372036854775807,,1e308,nan,9999-12-31,,2000-01-01T00:00:00-23:59,\"\n\
       multi\n\
       line\"\n\
       ,true,,inf,,2024-01-01T00:00:00.5,,plain text é\n" );
    ( "crlf",
      "\xef\xbb\xbfa;b;c\r\n1.5;x;2\r\n\r\n-2.25e-3;\"y\r\nz\";3\r\n;\"\";\r\n"
    );
    ( "tabs",
      "station\tdate\tvalue\tflag\n\
       NA\t2020-01-01\t12\tNA\n\
       US1\tNA\t-0\tQ\n\
       US2\t2020-01-03\tNA\t\n\n\
       US3\t2020-01-04\t007\tNA" );
    ("one column", "x\n1\n\n2\nNA\n\"3\"\n");
    ("no header", "1|2|3\n4|5|6\n7|8|9\n");
    ("quotes", "\"\"\"\",\"a\"\"b\",\"\n\"\n\"\",\"\",\"\"\n\"x\",\"y\",\"z\"\n");
  ]

let pick = Gen.of_list ~pp:Format.pp_print_string (List.map fst corpus)
let text name = List.assoc name corpus
let nulls = [ "NA" ]

(* The format sniffing gives each corpus file. *)
let formats =
  List.map
    (fun (name, s) ->
      match Csv.sniff ~nulls (Reader.of_string s) with
      | Ok f -> (name, f)
      | Error e -> failwith (Format.asprintf "%s: %a" name Error.pp e))
    corpus

(* Decoding *)

(* [decode ?rows ?slice f s] decodes [s] in slices of [slice] bytes, sniffed
   from its first [rows] records, then as [f], and labels the case with how far
   each read went. A property fails when no case of it fails to read as [f]. *)
let decode ?rows ?slice f s =
  let reader () = Reader.of_string ?slice_length:slice s in
  let outcome = function Ok _ -> "reads" | Error _ -> "fails" in
  (match Csv.sniff ?rows ~nulls (reader ()) with
  | Error _ -> collect "sniff fails"
  | Ok sniffed ->
      collect ("sniffed, " ^ outcome (Csv.decode sniffed (reader ()))));
  let original = outcome (Csv.decode f (reader ())) in
  collect ("as the original, " ^ original);
  cover "as the original, fails" (original = "fails")

let rows = Gen.(option (int_range 1 4))
let slice = Gen.(option (int_range 1 16))

(* Mutations *)

let at s k = k mod (String.length s + 1)

let flip bits s =
  if s = "" then s
  else begin
    let b = Bytes.of_string s in
    List.iter
      (fun (k, bit) ->
        let i = k mod String.length s in
        Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor (1 lsl bit)))
      bits;
    Bytes.to_string b
  end

let insert k piece s =
  let i = at s k in
  String.sub s 0 i ^ piece ^ String.sub s i (String.length s - i)

let remove k len s =
  let i = at s k in
  let len = min len (String.length s - i) in
  String.sub s 0 i ^ String.sub s (i + len) (String.length s - i - len)

(* [repeat k len n s] is [s] with its [len] bytes at [k] written [n] times. *)
let repeat k len n s =
  let i = at s k in
  let piece = String.sub s i (min len (String.length s - i)) in
  insert i (String.concat "" (List.init n (Fun.const piece))) s

(* The bytes that the syntax gives a meaning to, and bytes that are not
   UTF-8. *)
let syntax =
  Gen.of_list
    ~pp:(fun ppf s -> Format.fprintf ppf "%S" s)
    [
      "\"";
      "\"\"";
      ",";
      "\t";
      ";";
      "|";
      "\n";
      "\r";
      "\r\n";
      "\n\n";
      "\000";
      "\xef\xbb\xbf";
      "\xff";
      "\xc3";
      "NA";
      "-";
      ".";
      "e";
      "T";
      ":";
      "+";
    ]

let mutated name mutate (rows, slice) =
  let s = text name in
  decode ?rows ?slice (List.assoc name formats) (mutate s)

let bits = Gen.(list ~size:(int_range 1 8) (pair nat (int_range 0 7)))

let mutations =
  group ~timeout:600. "Mutated files decode to a table or an error"
    [
      prop "with bits flipped"
        Gen.(triple pick bits (pair rows slice))
        (fun (name, bits, r) -> mutated name (flip bits) r);
      prop "cut short"
        Gen.(triple pick nat (pair rows slice))
        (fun (name, k, r) -> mutated name (fun s -> String.sub s 0 (at s k)) r);
      prop "with syntax inserted"
        Gen.(
          triple pick
            (list ~size:(int_range 1 4) (pair nat syntax))
            (pair rows slice))
        (fun (name, pieces, r) ->
          mutated name
            (fun s -> List.fold_left (fun s (k, p) -> insert k p s) s pieces)
            r);
      prop "with bytes removed"
        Gen.(quad pick nat (int_range 1 16) (pair rows slice))
        (fun (name, k, len, r) -> mutated name (remove k len) r);
      prop "with bytes repeated"
        Gen.(
          quad pick
            (pair nat (int_range 1 32))
            (int_range 2 64) (pair rows slice))
        (fun (name, (k, len), n, r) -> mutated name (repeat k len n) r);
      (* The example is a file of one line feed, a batch of no record. *)
      prop "spliced with another file"
        ~examples:[ ("types", "one column", (0, 403), (None, None)) ]
        Gen.(quad pick pick (pair nat nat) (pair rows slice))
        (fun (a, b, (i, j), r) ->
          let b = text b in
          mutated a
            (fun s ->
              String.sub s 0 (at s i)
              ^ String.sub b (at b j) (String.length b - at b j))
            r);
    ]

(* Batches *)

(* A scanner finds a batch's end by counting quotes past 1 MiB of input, so a
   quote or a line break near that mark moves the boundary. The file has records
   of quoted line breaks, which put a line feed inside quotes on every other
   line. *)
let mib = 1 lsl 20

let big =
  let b = Buffer.create (mib + 4096) in
  Buffer.add_string b "a,b\n";
  let i = ref 0 in
  while Buffer.length b < mib + 2048 do
    Printf.bprintf b "\"r\n%d\",%d\n" !i !i;
    incr i
  done;
  Buffer.contents b

let big_format =
  Csv.format [ ("a", Type.Any Type.string); ("b", Type.Any Type.int64) ]

let boundary =
  prop ~count:50 ~timeout:600. "Mutations near a batch boundary"
    Gen.(triple (int_range (-64) 64) syntax (option (int_range 1 4096)))
    (fun (k, piece, slice) ->
      let s = insert (mib + k) piece big in
      decode ?slice big_format s)

let () = exit (run "talon.next.csv fuzz" [ mutations; boundary ])
