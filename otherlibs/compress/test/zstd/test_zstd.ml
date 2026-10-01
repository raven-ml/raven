(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Zstandard frames. The decoder reads what Python's zstandard wrote, with every
   block type, literals type and sequence mode, and refuses, with an error, the
   frames and blocks built here to break each rule of the format. *)

open Windtrap
module Z = Compress_zstd
module F = Compress_fixtures

let decompress src n =
  let dst = F.zeros n in
  Result.map
    (fun () -> F.string_of_bigbytes dst)
    (Z.decompress_into (F.bigbytes_of_string src) dst)

let le n bytes =
  String.init bytes (fun i -> Char.chr ((n lsr (8 * i)) land 0xFF))

(* A single-segment frame of [blocks], [(type, content)] each, declaring [size]
   bytes. *)
let make_frame ?(fhd = 0x20) ~size blocks =
  let rec encode = function
    | [] -> ""
    | [ (t, s, n) ] -> le ((n lsl 3) lor (t lsl 1) lor 1) 3 ^ s
    | (t, s, n) :: rest -> le ((n lsl 3) lor (t lsl 1)) 3 ^ s ^ encode rest
  in
  le 0xFD2FB528 4 ^ le fhd 1 ^ le size 1 ^ encode blocks

(* A compressed block of [content]. *)
let compressed content = (2, content, String.length content)

(* Literals "abc", raw, then one sequence whose three codes are repeated (RLE)
   and whose extra bits are the last byte. *)
let sequence ~ll ~ofc ~ml ~bits =
  "\x18abc\x01\x54"
  ^ String.init 3 (fun i -> Char.chr [| ll; ofc; ml |].(i))
  ^ bits

let abcabc = sequence ~ll:3 ~ofc:2 ~ml:0 ~bits:"\x06"

(* Reference vectors *)

let reference =
  cases ~name:fst "frames Python's zstandard wrote decode"
    [
      ("zstd_level-5.zst", F.text);
      ("zstd_level1.zst", F.text);
      ("zstd_level3.zst", F.text);
      ("zstd_level19.zst", F.text);
      ("zstd_level22.zst", F.text);
      ("zstd_columns.zst", F.columns);
      ("zstd_random.zst", F.random);
      ("zstd_rle.zst", String.make 300000 '\x07');
      ("zstd_runs.zst", F.runs);
      ( "zstd_small.zst",
        String.concat "" (List.init 4 (fun _ -> "hello, hello, zstandard!\n"))
      );
      ("zstd_blocks.zst", F.lines 40000);
      ("zstd_window28.zst", F.text);
      ("zstd_frames.zst", "first frame\nsecond frame\n");
      ("zstd_skippable.zst", "after\n");
      ("zstd_empty.zst", "");
    ]
    (fun (file, data) ->
      equal string data
        (require_ok ~pp:Format.pp_print_string
           (decompress (F.read file) (String.length data))))

let built =
  cases
    ~name:(fun (name, _, _) -> name)
    "frames built here decode"
    [
      ( "raw literals and a match",
        make_frame ~size:6 [ compressed abcabc ],
        "abcabc" );
      ("a raw block", make_frame ~size:3 [ (0, "xyz", 3) ], "xyz");
      ("an RLE block", make_frame ~size:5 [ (1, "q", 5) ], "qqqqq");
      ( "blocks of every type",
        make_frame ~size:14 [ (0, "xyz", 3); (1, "q", 5); compressed abcabc ],
        "xyzqqqqqabcabc" );
    ]
    (fun (_, src, data) ->
      equal string data
        (require_ok ~pp:Format.pp_print_string
           (decompress src (String.length data))))

(* Malformed frames *)

let hello_frame = make_frame ~size:3 [ (0, "xyz", 3) ]

let flip s at bit =
  String.mapi
    (fun i c -> if i = at then Char.chr (Char.code c lxor (1 lsl bit)) else c)
    s

let huffman_literals =
  (* Compressed literals, one stream, 4 bytes regenerated from 3: a direct tree
     of weights 3 and 1, which no last weight completes, and a byte of
     stream. *)
  let header = 2 lor (4 lsl 4) lor (3 lsl 14) in
  le header 3 ^ "\x81\x31\x80" ^ "\x00"

let malformed_frames =
  [
    ("no data", "", 3, "no frame");
    ("a bad magic number", flip hello_frame 0 0, 3, "not a Zstandard frame");
    ( "a reserved frame header bit",
      make_frame ~fhd:0x28 ~size:3 [ (0, "xyz", 3) ],
      3,
      "reserved bit" );
    ("a dictionary", F.read "zstd_dictionary.zst", 32, "dictionary needed");
    ( "a content size that does not match",
      make_frame ~size:4 [ (0, "xyz", 3) ],
      3,
      "content size mismatch" );
    ( "a content checksum that does not match",
      (let z = F.read "zstd_level3.zst" in
       flip z (String.length z - 1) 0),
      String.length F.text,
      "content checksum mismatch" );
    ( "a reserved block type",
      make_frame ~size:3 [ (3, "xyz", 3) ],
      3,
      "reserved block type" );
    ( "a block larger than 128 KiB",
      make_frame ~size:3 [ (1, "q", 131073) ],
      3,
      "larger than 128 KiB" );
    ("data after the frame", hello_frame ^ "!", 3, "truncated data at byte 12");
    ( "a match before the data",
      make_frame ~size:6
        [ compressed (sequence ~ll:3 ~ofc:3 ~ml:0 ~bits:"\x08") ],
      6,
      "match before the data" );
    ( "a match at offset 0",
      make_frame ~size:3
        [ compressed (sequence ~ll:0 ~ofc:1 ~ml:0 ~bits:"\x03") ],
      3,
      "match at offset 0" );
    ( "more literals than the section has",
      make_frame ~size:8
        [ compressed (sequence ~ll:5 ~ofc:2 ~ml:0 ~bits:"\x06") ],
      8,
      "invalid sequences section" );
    ( "an FSE table of accuracy log 20",
      make_frame ~size:6 [ compressed "\x18abc\x01\x94\x0f\x02\x00\x06" ],
      6,
      "invalid sequences section" );
    ( "Huffman weights no last weight completes",
      make_frame ~size:4 [ compressed (huffman_literals ^ "\x00") ],
      4,
      "invalid Huffman table" );
    ( "a sequence stream not read to its end",
      make_frame ~size:6 [ compressed (abcabc ^ "\x80") ],
      6,
      "invalid sequences section" );
  ]

let malformed =
  group "malformed frames"
    [
      cases
        ~name:(fun (name, _, _, _) -> name)
        "are refused" malformed_frames
        (fun (_, src, n, error) ->
          contains ~sub:error (require_error (decompress src n)));
      cases ~name:fst "a frame cut short at any byte is refused"
        [ ("zstd_small.zst", 100); ("zstd_frames.zst", 25); ("built", 14) ]
        (fun (file, n) ->
          let src =
            if file = "built" then
              make_frame ~size:14
                [ (0, "xyz", 3); (1, "q", 5); compressed abcabc ]
            else F.read file
          in
          for k = 0 to String.length src - 1 do
            ignore
              (require_error ~msg:(string_of_int k)
                 (decompress (String.sub src 0 k) n))
          done);
      prop "a frame with a bit flipped decompresses or is refused"
        (let open Gen in
         let* file =
           of_list
             [
               "zstd_level3.zst";
               "zstd_level19.zst";
               "zstd_blocks.zst";
               "zstd_columns.zst";
             ]
         in
         let src = F.read file in
         let+ at = int_range 0 (String.length src - 1)
         and+ bit = int_range 0 7 in
         (file, flip src at bit))
        (fun (file, src) ->
          let n =
            String.length
              (if file = "zstd_blocks.zst" then F.lines 40000 else F.text)
          in
          ignore (decompress src n));
      cases ~name:fst "a destination one byte short or long is refused"
        [
          ("zstd_level3.zst", F.text);
          ("zstd_rle.zst", String.make 300000 '\x07');
        ]
        (fun (file, data) ->
          let n = String.length data in
          ignore (require_error (decompress (F.read file) (n - 1)));
          ignore (require_error (decompress (F.read file) (n + 1))));
      test "overlapping arrays are refused" (fun () ->
          let b = F.zeros 64 in
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              Z.decompress_into
                (Bigarray.Array1.sub b 0 40)
                (Bigarray.Array1.sub b 30 34)));
    ]

let () =
  exit
    (run "compress.zstd" [ group "reference" [ reference; built ]; malformed ])
