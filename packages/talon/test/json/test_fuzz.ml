(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Malformed JSON text: sniff gives a format or an Error, and decode a table or
   an Error, never an exception or a hang, however the reader slices it. *)

open Windtrap
module J = Talon_json
module Reader = Bytesrw.Bytes.Reader

let flat =
  String.concat "\n"
    [
      {|{"id":1,"price":1.5,"name":"alpha","active":true}|};
      {|{"id":2,"price":null,"name":"bé\"ta\n","active":false}|};
      {|{"id":-3,"name":"é😀","tags":[1,2,[3]],"o":{"a":1}}|};
      "";
    ]

let array_doc =
  {|[
  {"id": 1, "x": [1, 2, 3], "o": {"k": "v", "n": null}},
  {"id": 2.5e0, "x": [], "o": {"k": "w", "n": 7}}
]|}

let nested =
  {|{"user":{"name":"ana","loc":{"lat":1.5}},"tags":["a","b"],"m":[[1],[2,3]]}
{"user":null,"tags":[],"m":null}
|}

let documents = [| flat; array_doc; nested |]

(* Each document's format, as which decode reads its mutations. *)
let formats =
  lazy
    (Array.map
       (fun s -> Result.get_ok (J.sniff (Reader.of_string s)))
       documents)

let tokens =
  [
    "\"";
    "\\";
    ",";
    ":";
    "{";
    "}";
    "[";
    "]";
    "\n";
    "\r\n";
    "\000";
    "\xff";
    "\xc3";
    "\xed\xa0\x80";
    "null";
    "true";
    "1e999";
    "-";
    "\\ud800";
    "\\uZZZZ";
    "\xef\xbb\xbf";
    "01";
    "1.";
    ".5";
    "NaN";
  ]

(* Mutations of a document, at a position [k] that wraps past its end. *)

let at s k = k mod (String.length s + 1)

let flip bits s =
  let b = Bytes.of_string s in
  List.iter
    (fun (k, bit) ->
      let i = k mod String.length s in
      Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor (1 lsl bit)))
    bits;
  Bytes.to_string b

let insert k piece s =
  let i = at s k in
  String.sub s 0 i ^ piece ^ String.sub s i (String.length s - i)

let remove k len s =
  let i = at s k in
  let len = min len (String.length s - i) in
  String.sub s 0 i ^ String.sub s (i + len) (String.length s - i - len)

(* A case: the document whose format decode reads, the malformed text, how it
   was made, and sniff's rows and the reader's slices. *)
type case = {
  doc : int;
  mutation : string;
  text : string;
  rows : int option;
  slice : int option;
}

let pp_case ppf c =
  let pp_int = Format.(pp_print_option pp_print_int) in
  Format.fprintf ppf "%s of document %d: %S, rows %a, slices %a" c.mutation
    c.doc c.text pp_int c.rows pp_int c.slice

let case =
  let open Gen in
  let mutated doc =
    let s = documents.(doc) in
    one_of
      [
        map
          (fun bits -> ("flipped bits", flip bits s))
          (list ~size:(int_range 1 8) (pair nat (int_range 0 7)));
        map
          (fun (k, piece) -> ("inserted syntax", insert k piece s))
          (pair nat (of_list tokens));
        map
          (fun (k, len) -> ("removed bytes", remove k len s))
          (pair nat (int_range 1 16));
        map (fun s -> ("random bytes", s)) string;
      ]
  in
  with_pp pp_case
    (let* doc = int_range 0 (Array.length documents - 1) in
     let* mutation, text = mutated doc in
     let* rows = option (int_range 1 4) in
     let+ slice = option (int_range 1 16) in
     { doc; mutation; text; rows; slice })

let reader c = Reader.of_string ?slice_length:c.slice c.text

let cover_mutations c =
  List.iter
    (fun m -> cover m (c.mutation = m))
    [ "flipped bits"; "inserted syntax"; "removed bytes"; "random bytes" ]

let sniff =
  prop ~count:1000 "sniff › malformed text is a format or an Error" case
    (fun c ->
      let r = J.sniff ?rows:c.rows (reader c) in
      cover_mutations c;
      cover "sniffed" (Result.is_ok r);
      cover "refused" (Result.is_error r))

(* Decode reads the text as the document's format, and as the format sniff gives
   for the text itself, as Talon_json.file does without one. *)
let decode =
  prop ~count:1000 "decode › malformed text is a table or an Error" case
    (fun c ->
      let r = J.decode (Lazy.force formats).(c.doc) (reader c) in
      let sniffed = J.sniff ?rows:c.rows (reader c) in
      let r' = Result.map (fun f -> J.decode f (reader c)) sniffed in
      cover_mutations c;
      cover "read" (Result.is_ok r);
      cover "refused" (Result.is_error r);
      cover "read as sniffed" (match r' with Ok (Ok _) -> true | _ -> false);
      cover "refused as sniffed"
        (match r' with Ok (Error _) -> true | _ -> false))

let () = exit (run "talon.json fuzz" [ sniff; decode ])
