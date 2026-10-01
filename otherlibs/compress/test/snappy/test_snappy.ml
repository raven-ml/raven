(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Snappy blocks. Decompression inverts compression and refuses malformed blocks
   with an error. Both directions agree with python-snappy byte for byte:
   compress writes the blocks it writes. *)

open Windtrap
module S = Compress_snappy
module F = Compress_fixtures

let compress s =
  let src = F.bigbytes_of_string s in
  let dst = F.zeros (S.max_compressed_length (String.length s)) in
  let n = S.compress src dst in
  F.string_of_bigbytes (Bigarray.Array1.sub dst 0 n)

let decompress block n =
  let dst = F.zeros n in
  Result.map
    (fun () -> F.string_of_bigbytes dst)
    (S.decompress (F.bigbytes_of_string block) dst)

let strings =
  Gen.frequency
    [
      (8, Gen.string);
      (2, Gen.string_of (Gen.char_range 'a' 'c'));
      ( 1,
        Gen.string_of
          ~size:(Gen.int_range 60_000 200_000)
          (Gen.char_range 'a' 'd') );
      (1, Gen.string_of ~size:(Gen.int_range 60_000 200_000) Gen.char);
    ]

let round_trips =
  group "round trips"
    [
      prop ~tags:[ "slow" ]
        "decompress inverts compress, across 64 KiB fragments" strings (fun s ->
          equal string s
            (require_ok (decompress (compress s) (String.length s))));
      prop ~tags:[ "slow" ] "a block is no longer than max_compressed_length"
        strings (fun s ->
          at_most int
            (String.length (compress s))
            ~than:(S.max_compressed_length (String.length s)));
      cases ~name:string_of_int
        "a block of that many random bytes is no longer than \
         max_compressed_length"
        [ 0; 1; 65535; 65536; 65537; 1 lsl 20 ]
        (fun n ->
          let state = Random.State.make [| n |] in
          let s =
            String.init n (fun _ -> Char.chr (Random.State.int state 256))
          in
          at_most int
            (String.length (compress s))
            ~than:(S.max_compressed_length n));
      prop "a block depends only on the bytes it compresses"
        (Gen.pair Gen.string (Gen.int_range 0 64))
        (fun (s, at) ->
          (* [s] at offset [at] of a larger array, after other bytes. *)
          let n = String.length s in
          let b = F.bigbytes_of_string (String.make at 'x' ^ s ^ "tail") in
          let dst = F.zeros (S.max_compressed_length n + 8) in
          let length =
            S.compress
              (Bigarray.Array1.sub b at n)
              (Bigarray.Array1.sub dst 8 (S.max_compressed_length n))
          in
          equal string (compress s)
            (F.string_of_bigbytes (Bigarray.Array1.sub dst 8 length)));
    ]

let corpora =
  [
    ("text", F.text);
    ("columns", F.columns);
    ("runs", F.runs);
    ("random", F.random);
    ("empty", "");
  ]

let fixtures =
  group "python-snappy's blocks"
    [
      cases ~name:fst "decompress" corpora (fun (name, data) ->
          let block = F.read ("snappy_" ^ name ^ ".snappy") in
          equal string data (require_ok (decompress block (String.length data))));
      cases ~name:fst "are what compress writes" corpora (fun (name, data) ->
          equal string (F.read ("snappy_" ^ name ^ ".snappy")) (compress data));
    ]

let hello = compress "hello, hello, hello, snappy!"

let malformed_cases =
  [
    ( "a length of more than 5 bytes",
      "\x80\x80\x80\x80\x80\x01",
      0,
      "invalid length" );
    ("a length beyond 32 bits", "\xff\xff\xff\xff\x1f", 0, "invalid length");
    ("a literal past the end", "\x05\x10ab", 5, "literal past the end");
    ("a copy at offset 0", "\x08\x08abc\x01\x00", 8, "copy at offset 0");
    ("a copy before the data", "\x08\x08abc\x01\x09", 8, "copy before the data");
    ( "a copy past the destination",
      "\x05\x08abc\x01\x03",
      5,
      "data longer than the destination" );
    ("a copy whose offset is cut", "\x08\x08abc\x02\x01", 8, "truncated data");
    ( "data shorter than declared",
      "\x08\x08abc",
      8,
      "data shorter than the destination" );
    ( "a declared length unlike the destination's",
      hello,
      7,
      "declared length differs" );
    ("no data", "", 0, "truncated data");
  ]

let flip s at bit =
  String.mapi
    (fun i c -> if i = at then Char.chr (Char.code c lxor (1 lsl bit)) else c)
    s

let malformed =
  group "malformed blocks"
    [
      cases
        ~name:(fun (name, _, _, _) -> name)
        "are refused" malformed_cases
        (fun (_, block, n, error) ->
          contains ~sub:error (require_error (decompress block n)));
      test "a block cut short at any byte is refused" (fun () ->
          let data = F.lines 300 in
          let block = compress data in
          for n = 0 to String.length block - 1 do
            ignore
              (require_error ~msg:(string_of_int n)
                 (decompress (String.sub block 0 n) (String.length data)))
          done);
      prop "a block with a bit flipped decompresses or is refused"
        (let open Gen in
         let* s = string_of ~size:(int_range 1 3000) (char_range 'a' 'f') in
         let block = compress s in
         let+ at = int_range 0 (String.length block - 1)
         and+ bit = int_range 0 7 in
         (flip block at bit, String.length s))
        (fun (block, n) -> ignore (decompress block n));
      test "an error names the byte where decoding failed" (fun () ->
          contains ~sub:"at byte 5"
            (require_error (decompress "\x08\x08abc\x01\x00" 8)));
    ]

let arguments =
  group "arguments"
    [
      cases ~name:string_of_int
        "max_compressed_length refuses a length outside 32 bits"
        [ -1; 0x1_0000_0000 ] (fun n ->
          raises_match Exn.invalid_arg (fun () -> S.max_compressed_length n));
      test "compress refuses a destination shorter than max_compressed_length"
        (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              S.compress
                (F.bigbytes_of_string "abc")
                (F.zeros (S.max_compressed_length 3 - 1))));
      test "compress and decompress refuse overlapping arrays" (fun () ->
          let b = F.zeros 200 in
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              S.compress
                (Bigarray.Array1.sub b 0 50)
                (Bigarray.Array1.sub b 40 160));
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              S.decompress
                (Bigarray.Array1.sub b 0 50)
                (Bigarray.Array1.sub b 40 160)));
    ]

let () =
  exit (run "compress.snappy" [ round_trips; fixtures; malformed; arguments ])
