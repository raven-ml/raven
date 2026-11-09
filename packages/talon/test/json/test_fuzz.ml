(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Malformed JSON files: each sniff and decode is a format, a table or an error,
   never an exception or a hang, however the reader slices the input. *)

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

let corpus =
  [ ("flat", flat); ("array", array_doc); ("nested", nested); ("empty", "") ]

let pick = Gen.of_list ~pp:Format.pp_print_string (List.map fst corpus)
let text name = List.assoc name corpus

let formats =
  List.filter_map
    (fun (name, s) ->
      match J.sniff (Reader.of_string s) with
      | Ok f -> Some (name, f)
      | Error _ -> None)
    corpus

(* [decode ?rows ?slice f s] decodes [s] in slices of [slice] bytes, sniffed
   from its first [rows] rows, then as [f]. Any failure is an [Error]. *)
let decode ?rows ?slice f s =
  let reader () = Reader.of_string ?slice_length:slice s in
  let outcome = function Ok _ -> "reads" | Error _ -> "fails" in
  (match J.sniff ?rows (reader ()) with
  | Error _ -> collect "sniff fails"
  | Ok sniffed -> collect ("sniffed, " ^ outcome (J.decode sniffed (reader ()))));
  match f with
  | None -> ()
  | Some f ->
      let original = outcome (J.decode f (reader ())) in
      collect ("as the original, " ^ original);
      cover "as the original, fails" (original = "fails")

let rows = Gen.(option (int_range 1 4))
let slice = Gen.(option (int_range 1 16))
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

let tokens =
  Gen.of_list
    ~pp:(fun ppf s -> Format.fprintf ppf "%S" s)
    [
      "\""; "\\"; ","; ":"; "{"; "}"; "["; "]"; "\n"; "\r\n"; "\000"; "\xff";
      "\xc3"; "\xed\xa0\x80"; "null"; "true"; "1e999"; "-"; "\\ud800";
      "\\uZZZZ"; "\xef\xbb\xbf"; "01"; "1."; ".5"; "NaN";
    ]

let mutated name mutate (rows, slice) =
  decode ?rows ?slice (List.assoc_opt name formats) (mutate (text name))

let bits = Gen.(list ~size:(int_range 1 8) (pair nat (int_range 0 7)))

let flipped =
  prop ~count:500 "flipped bits fail cleanly"
    Gen.(triple pick bits (pair rows slice))
    (fun (name, bits, opts) -> mutated name (flip bits) opts)

let inserted =
  prop ~count:500 "inserted syntax fails cleanly"
    Gen.(quad pick nat tokens (pair rows slice))
    (fun (name, k, piece, opts) -> mutated name (insert k piece) opts)

let removed =
  prop ~count:500 "removed bytes fail cleanly"
    Gen.(quad pick nat (int_range 1 16) (pair rows slice))
    (fun (name, k, len, opts) -> mutated name (remove k len) opts)

let garbage =
  prop ~count:300 "random bytes fail cleanly"
    Gen.(pair string (pair rows slice))
    (fun (s, (rows, slice)) -> decode ?rows ?slice None s)

let () =
  exit (run "talon.json fuzz" [ flipped; inserted; removed; garbage ])
