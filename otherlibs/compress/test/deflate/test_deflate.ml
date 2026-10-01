(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Deflate, zlib and gzip. Each format reads back what it writes through both
   decoders, which agree on the streams Python's zlib and gzip wrote, whatever
   the slicing of their input. Malformed streams are refused with an error,
   never another exception. *)

open Windtrap
open Bytesrw
module C = Compress_deflate
module F = Compress_fixtures

(* Formats *)

type format = {
  name : string;
  reads : unit -> Bytes.Reader.filter;
  writes : ?level:int -> unit -> Bytes.Writer.filter;
  decompress : F.bigbytes -> F.bigbytes -> (unit, string) result;
}

let deflate =
  {
    name = "deflate";
    reads = C.Deflate.decompress_reads;
    writes = C.Deflate.compress_writes;
    decompress = C.Deflate.decompress;
  }

let zlib =
  {
    name = "zlib";
    reads = C.Zlib.decompress_reads;
    writes = C.Zlib.compress_writes;
    decompress = C.Zlib.decompress;
  }

let gzip =
  {
    name = "gzip";
    reads = C.Gzip.decompress_reads;
    writes = C.Gzip.compress_writes;
    decompress = C.Gzip.decompress;
  }

let formats = [ deflate; zlib; gzip ]
let compress ?level f s = Bytes.Writer.filter_string [ f.writes ?level () ] s

(* [compress_sliced f k s] writes [s] in slices of [k] bytes. *)
let compress_sliced ?level f k s =
  let b = Buffer.create 256 in
  let w = f.writes ?level () ~eod:true (Bytes.Writer.of_buffer b) in
  let rec loop first =
    if first < String.length s then begin
      let last = Int.min (String.length s) (first + k) - 1 in
      Bytes.Writer.write w (Bytes.Slice.of_string ~first ~last s);
      loop (last + 1)
    end
  in
  loop 0;
  Bytes.Writer.write_eod w;
  Buffer.contents b

let reads ?(slice_length = Bytes.Slice.io_buffer_size) f s =
  Bytes.Reader.to_string (f.reads () (Bytes.Reader.of_string ~slice_length s))

(* [decompress f s n] decompresses [s] in memory into [n] bytes. *)
let decompress f s n =
  let dst = F.zeros n in
  Result.map
    (fun () -> F.string_of_bigbytes dst)
    (f.decompress (F.bigbytes_of_string s) dst)

(* [refused f s error] asserts that both decoders refuse [s] with a message that
   contains [error]. *)
let refused ?(length = 64) f s error =
  contains ~msg:"decompress" ~sub:error
    (require_error ~msg:"decompress" (decompress f s length));
  match reads f s with
  | _ -> fail "decompress_reads: no error"
  | exception Bytes.Stream.Error e ->
      contains ~msg:"decompress_reads" ~sub:error (Bytes.Stream.error_message e)

(* [returns_or_errors f s] asserts that both decoders return or error on [s], as
   they promise for any bytes: never another exception or a crash. *)
let returns_or_errors f s n =
  ignore (decompress f s n);
  match reads f s with _ -> () | exception Bytes.Stream.Error _ -> ()

let strings =
  let long = Gen.int_range 60_000 200_000 in
  Gen.frequency
    [
      (18, Gen.string);
      (1, Gen.string_of ~size:long (Gen.char_range 'a' 'd'));
      (1, Gen.string_of ~size:long Gen.char);
    ]

(* Round trips *)

let inverts f =
  prop ~tags:[ "slow" ]
    (f.name
   ^ " decompression inverts compress_writes on short strings and on long ones \
      past the window and block sizes") strings (fun s ->
      let z = compress f s in
      equal string s (reads f z);
      equal string s (require_ok (decompress f z (String.length s))))

let levels =
  prop ~tags:[ "slow" ]
    "every level round trips and writes the same stream however it is fed"
    (Gen.triple (Gen.int_range 0 9)
       (Gen.frequency
          [
            (8, Gen.string);
            ( 1,
              Gen.string_of
                ~size:(Gen.int_range 60_000 140_000)
                (Gen.char_range 'a' 'd') );
          ])
       (Gen.int_range 1 70_000))
    (fun (level, s, k) ->
      let z = compress ~level zlib s in
      equal string s (reads zlib z);
      equal string z (compress_sliced ~level zlib k s))

(* [letters seed] is 200 to 400 thousand pseudo-random letters from [a] to [p]:
   about 4 bits each once compressed, so the stream outgrows a reader's input
   buffer and the reader resumes inside dynamic Huffman blocks. *)
let letters seed =
  let r = Random.State.make [| seed |] in
  String.init (Random.State.int_in_range r ~min:200_000 ~max:400_000) (fun _ ->
      Char.chr (Char.code 'a' + Random.State.int r 16))

let resumes =
  prop ~tags:[ "slow" ] ~count:20
    "a reader resumes decoding wherever its input buffer ends in a dynamic \
     block"
    (Gen.triple
       (Gen.of_list
          ~pp:(fun ppf f -> Format.pp_print_string ppf f.name)
          formats)
       (Gen.int_range 1 9) Gen.int)
    (fun (f, level, seed) ->
      let s = letters seed in
      let z = compress ~level f s in
      greater int (String.length z) ~than:Bytes.Slice.io_buffer_size;
      equal string s (reads f z);
      equal string s (require_ok (decompress f z (String.length s))))

let round_trips =
  group "round trips"
    (List.map inverts formats
    @ [
        levels;
        resumes;
        cases ~name:Fun.id "the empty string round trips"
          [ "deflate"; "zlib"; "gzip" ] (fun name ->
            let f = List.find (fun f -> f.name = name) formats in
            equal string "" (reads f (compress f ""));
            equal string "" (require_ok (decompress f (compress f "") 0)));
      ])

(* Streams other tools wrote *)

let python_streams =
  [
    ("zlib_empty.z", zlib, "");
    ("zlib_fixed.z", zlib, "hello, nx zlib!\n");
    ("zlib_stored.z", zlib, "stored block\n");
    ("zlib_dynamic.z", zlib, F.lines 200);
    ("zlib_long.z", zlib, F.lines 60_000);
    ("zlib_level0.z", zlib, F.text);
    ("zlib_level1.z", zlib, F.text);
    ("zlib_level6.z", zlib, F.text);
    ("zlib_level9.z", zlib, F.text);
    ("zlib_window512.z", zlib, F.text);
    ("deflate_text.deflate", deflate, F.text);
    ("gzip_python.gz", gzip, F.text);
    ("gzip_members.gz", gzip, "first member\nsecond member\n");
    ("gzip_flags.gz", gzip, "flags\n");
  ]

let slice_lengths = List.init 17 succ @ [ 1000; 65536 ]

let fixtures =
  group "streams Python wrote"
    [
      cases
        ~name:(fun (file, _, _) -> file)
        "decode in memory and through readers sliced every way" python_streams
        (fun (file, f, data) ->
          let z = F.read file in
          equal string data (require_ok (decompress f z (String.length data)));
          List.iter
            (fun slice_length ->
              equal
                ~msg:(Printf.sprintf "slice length %d" slice_length)
                string data (reads ~slice_length f z))
            slice_lengths);
      prop "a reader's output does not depend on how its input is sliced"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (file, _, _) -> Format.pp_print_string ppf file)
              python_streams)
           (Gen.int_range 1 4096))
        (fun ((file, f, data), slice_length) ->
          equal string data (reads ~slice_length f (F.read file)));
    ]

(* Golden streams: what the encoder writes, pinned. generate.py checks that
   Python's zlib and gzip decode each to its input. *)

let golden =
  cases
    ~name:(fun (file, _, _, _) -> file)
    "the encoder writes the pinned streams"
    [
      ("golden_text_level6.z", zlib, 6, F.text);
      ("golden_text_level1.gz", gzip, 1, F.text);
      ("golden_text_level9.deflate", deflate, 9, F.text);
      ("golden_hello_level0.z", zlib, 0, "hello, compress!\n");
      ("golden_empty_level6.gz", gzip, 6, "");
      ("golden_columns_level6.deflate", deflate, 6, F.columns);
      ("golden_random_level6.z", zlib, 6, F.random);
    ]
    (fun (file, f, level, data) ->
      equal string (F.read file) (compress ~level f data))

(* Malformed streams *)

(* [pack fields] is the bytes of the bit [fields], [(value, width)] each, packed
   from the least significant bit as deflate packs them. [code v w] is the
   Huffman code [v] of [w] bits, which deflate packs from its most significant
   bit. *)
let pack fields =
  let b = Buffer.create 16 in
  let acc = ref 0 and n = ref 0 in
  List.iter
    (fun (v, w) ->
      acc := !acc lor (v lsl !n);
      n := !n + w;
      while !n >= 8 do
        Buffer.add_char b (Char.chr (!acc land 0xFF));
        acc := !acc lsr 8;
        n := !n - 8
      done)
    fields;
  if !n > 0 then Buffer.add_char b (Char.chr !acc);
  Buffer.contents b

let code v w =
  let r = ref 0 in
  for i = 0 to w - 1 do
    if v land (1 lsl i) <> 0 then r := !r lor (1 lsl (w - 1 - i))
  done;
  (!r, w)

let fixed = [ (1, 1); (1, 2) ]
let dynamic = [ (1, 1); (2, 2) ]

(* Code length code lengths in the order deflate sends them. *)
let codelen_lengths l = List.map (fun n -> (n, 3)) l

let flip s at bit =
  String.mapi
    (fun i c -> if i = at then Char.chr (Char.code c lxor (1 lsl bit)) else c)
    s

let hello = "hello, compress!\n"

let malformed_cases =
  [
    ( "deflate: block type 3",
      deflate,
      pack [ (1, 1); (3, 2) ],
      "invalid block type" );
    ( "deflate: stored length and complement differ",
      deflate,
      "\x01\x05\x00\x00\x00",
      "does not match its complement" );
    ( "deflate: a distance before the data",
      deflate,
      pack (fixed @ [ code 1 7; (0, 5); code 0 7 ]),
      "distance beyond the data" );
    ( "deflate: an invalid literal/length symbol",
      deflate,
      pack (fixed @ [ code 0xC6 8 ]),
      "invalid symbol" );
    ( "deflate: an invalid distance symbol",
      deflate,
      pack (fixed @ [ code 0x30 8; code 1 7; code 30 5 ]),
      "invalid symbol" );
    ( "deflate: 287 literal/length codes",
      deflate,
      pack (dynamic @ [ (30, 5); (0, 5); (0, 4) ]),
      "too many length or distance codes" );
    ( "deflate: 31 distance codes",
      deflate,
      pack (dynamic @ [ (0, 5); (30, 5); (0, 4) ]),
      "too many length or distance codes" );
    ( "deflate: an over-subscribed code length code",
      deflate,
      pack
        (dynamic @ [ (0, 5); (0, 5); (0, 4) ] @ codelen_lengths [ 1; 1; 1; 0 ]),
      "invalid Huffman code lengths" );
    ( "deflate: a repeat with no length before it",
      deflate,
      pack
        (dynamic
        @ [ (0, 5); (0, 5); (0, 4) ]
        @ codelen_lengths [ 1; 0; 1; 0 ]
        @ [ (0, 1) ]),
      "repeat out of range" );
    ( "deflate: no end-of-block code",
      deflate,
      pack
        (dynamic
        @ [ (0, 5); (0, 5); (14, 4) ]
        @ codelen_lengths
            [ 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1 ]
        @ [ (0, 1); (0, 1); (1, 1); (127, 7); (1, 1); (107, 7) ]),
      "no end-of-block code" );
    ( "deflate: data after the stream",
      deflate,
      compress deflate hello ^ "!",
      "data after the stream" );
    ("deflate: no data", deflate, "", "truncated data");
    ( "zlib: a preset dictionary",
      zlib,
      "\x78\x20\x00\x00\x00\x01\x03\x00",
      "preset dictionary needed" );
    ( "zlib: a header check that fails",
      zlib,
      "\x78\x9d\x03\x00\x00\x00\x00\x01",
      "invalid zlib header" );
    ( "zlib: a method other than deflate",
      zlib,
      "\x77\x09\x03\x00\x00\x00\x00\x01",
      "invalid zlib header" );
    ( "zlib: an Adler-32 that does not match",
      zlib,
      (let z = compress zlib hello in
       flip z (String.length z - 1) 0),
      "Adler-32 mismatch" );
    ( "zlib: data after the stream",
      zlib,
      compress zlib hello ^ "!",
      "data after the stream" );
    ("zlib: no data", zlib, "", "truncated data");
    ( "gzip: a bad magic number",
      gzip,
      flip (compress gzip hello) 1 0,
      "not a gzip member" );
    ( "gzip: reserved flags",
      gzip,
      flip (compress gzip hello) 3 5,
      "reserved flags set" );
    ( "gzip: a header checksum that does not match",
      gzip,
      flip (F.read "gzip_flags.gz") 37 0,
      "header checksum mismatch" );
    ( "gzip: a CRC-32 that does not match",
      gzip,
      (let z = compress gzip hello in
       flip z (String.length z - 8) 0),
      "CRC-32 mismatch" );
    ( "gzip: a length that does not match",
      gzip,
      (let z = compress gzip hello in
       flip z (String.length z - 1) 0),
      "length mismatch" );
    ( "gzip: garbage after a member",
      gzip,
      compress gzip hello ^ "garbage",
      "not a gzip member" );
    ("gzip: no data", gzip, "", "truncated data");
  ]

let cut s n = String.sub s 0 n

let truncations =
  cases ~name:fst "a stream cut short at any byte is refused"
    [
      ("deflate", (deflate, F.lines 300));
      ("zlib", (zlib, F.lines 300));
      ("gzip", (gzip, hello));
    ]
    (fun (_, (f, data)) ->
      let z = compress f data in
      for n = 0 to String.length z - 1 do
        refused ~length:(String.length data) f (cut z n) "truncated data"
      done)

let damage =
  let open Gen in
  let* f =
    of_list ~pp:(fun ppf f -> Format.pp_print_string ppf f.name) formats
  in
  let* s = string_of ~size:(int_range 1 3000) (char_range 'a' 'f') in
  let z = compress f s in
  let+ at = int_range 0 (String.length z - 1) and+ bit = int_range 0 7 in
  (f, String.length s, flip z at bit)

let malformed =
  group "malformed streams"
    [
      cases
        ~name:(fun (name, _, _, _) -> name)
        "are refused" malformed_cases
        (fun (_, f, s, error) -> refused f s error);
      truncations;
      prop "a stream with a bit flipped decodes or is refused" damage
        (fun (f, n, z) -> returns_or_errors f z n);
      prop "a header and any bytes decode or are refused"
        (Gen.pair (Gen.of_list [ 0; 1; 2 ]) Gen.string)
        (fun (i, b) ->
          let f = List.nth formats i in
          let header =
            String.sub (compress f "") 0 (if i = 2 then 10 else i * 2)
          in
          returns_or_errors f (header ^ b) 64);
      cases
        ~name:(fun f -> f.name)
        "a destination one byte short or long is refused" formats
        (fun f ->
          let z = compress f hello in
          let n = String.length hello in
          ignore (require_error (decompress f z (n - 1)));
          ignore (require_error (decompress f z (n + 1))));
      cases
        ~name:(fun f -> f.name)
        "overlapping arrays are refused" formats
        (fun f ->
          let b = F.zeros 64 in
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              f.decompress
                (Bigarray.Array1.sub b 0 40)
                (Bigarray.Array1.sub b 30 34)));
    ]

(* Gzip members *)

let members =
  group "gzip members"
    [
      prop "the data of concatenated members is their data, concatenated"
        (Gen.list ~size:(Gen.int_range 1 4) Gen.string)
        (fun l ->
          let z = String.concat "" (List.map (compress gzip) l) in
          let data = String.concat "" l in
          equal string data (reads gzip z);
          equal string data
            (require_ok (decompress gzip z (String.length data))));
      test "a member's header is fixed" (fun () ->
          equal string "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
            (String.sub (compress gzip F.text) 0 10));
    ]

(* Writers *)

let writers =
  group "compress_writes"
    [
      cases ~name:string_of_int "refuses a level outside 0 to 9" [ -1; 10 ]
        (fun level ->
          raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
              C.Zlib.compress_writes ~level ()));
      test "defaults to level 6" (fun () ->
          equal string (compress ~level:6 zlib F.text) (compress zlib F.text));
      test "without eod leaves the writer open for other writes" (fun () ->
          let b = Buffer.create 64 in
          let w = Bytes.Writer.of_buffer b in
          let z = C.Gzip.compress_writes () ~eod:false w in
          Bytes.Writer.write_string z hello;
          Bytes.Writer.write_eod z;
          Bytes.Writer.write_string w "tail";
          Bytes.Writer.write_eod w;
          equal string (compress gzip hello ^ "tail") (Buffer.contents b));
      test "writes slices no longer than the writer's slice length" (fun () ->
          let longest = ref 0 in
          let w =
            Bytes.Writer.make ~slice_length:7 (fun s ->
                longest := Int.max !longest (Bytes.Slice.length s))
          in
          let z = C.Zlib.compress_writes () ~eod:true w in
          Bytes.Writer.write_string z F.random;
          Bytes.Writer.write_eod z;
          equal int 7 !longest);
      prop ~tags:[ "slow" ]
        "a deflate stream of n bytes is at most n + 5 * (n / 65535 + 1) bytes \
         long"
        (Gen.pair (Gen.int_range 0 9)
           (Gen.frequency
              [
                (3, Gen.string);
                (1, Gen.string_of ~size:(Gen.int_range 60_000 140_000) Gen.char);
              ]))
        (fun (level, s) ->
          let n = String.length s in
          at_most int
            (String.length (compress ~level deflate s))
            ~than:(n + (5 * ((n / 65535) + 1))));
    ]

(* Positions and errors *)

let positions =
  group "positions"
    [
      cases ~name:string_of_int
        "a reader's slices are no longer than its slice length"
        [ 1; 7; 1000; 65536 ] (fun slice_length ->
          let r = Bytes.Reader.of_string (compress zlib F.text) in
          let r = C.Zlib.decompress_reads () ~slice_length r in
          let rec longest acc =
            let s = Bytes.Reader.read r in
            if Bytes.Slice.is_eod s then acc
            else longest (Int.max acc (Bytes.Slice.length s))
          in
          equal int slice_length (longest 0));
      test "a reader starts at position 0 or the given one" (fun () ->
          let r = Bytes.Reader.of_string (compress zlib hello) in
          equal int 0 (Bytes.Reader.pos (C.Zlib.decompress_reads () r));
          equal int 42 (Bytes.Reader.pos (C.Zlib.decompress_reads () ~pos:42 r)));
      cases ~name:fst "an error names the byte where decoding failed"
        [ ("in memory", `Memory); ("on a reader", `Reader) ]
        (fun (_, path) ->
          let z = compress zlib hello in
          let at = String.length z in
          let msg =
            match path with
            | `Memory -> require_error (decompress zlib (z ^ "!") 17)
            | `Reader -> (
                match reads zlib (z ^ "!") with
                | _ -> fail "no error"
                | exception Bytes.Stream.Error e -> Bytes.Stream.error_message e
                )
          in
          contains ~sub:(string_of_int at) msg);
      test "a reader's error is a zlib error" (fun () ->
          match reads zlib "" with
          | _ -> fail "no error"
          | exception Bytes.Stream.Error (C.Error _, _) -> ()
          | exception Bytes.Stream.Error e ->
              failf "%s" (Bytes.Stream.error_message e));
    ]

(* CRC-32 *)

(* CRC-32 one bit at a time, from its definition. *)
let crc32_spec s =
  let crc = ref 0xFFFFFFFF in
  String.iter
    (fun c ->
      crc := !crc lxor Char.code c;
      for _ = 1 to 8 do
        crc := (!crc lsr 1) lxor (!crc land 1 * 0xEDB88320)
      done)
    s;
  !crc lxor 0xFFFFFFFF

let crc32 =
  group "Crc32"
    [
      test "the checksum of 123456789 is 0xCBF43926" (fun () ->
          equal int 0xCBF43926
            (C.Crc32.bigbytes (F.bigbytes_of_string "123456789")));
      prop "agrees with the bit-by-bit definition" Gen.string (fun s ->
          equal int (crc32_spec s) (C.Crc32.bigbytes (F.bigbytes_of_string s));
          equal int (crc32_spec s)
            (C.Crc32.slice (Bytes.Slice.of_string_or_eod s)));
      prop "continues a checksum with ~crc" (Gen.pair Gen.string Gen.string)
        (fun (a, b) ->
          let crc = C.Crc32.slice (Bytes.Slice.of_string_or_eod a) in
          equal int
            (crc32_spec (a ^ b))
            (C.Crc32.bigbytes ~crc (F.bigbytes_of_string b)));
    ]

let () =
  exit
    (run "compress.deflate"
       [
         round_trips;
         fixtures;
         golden;
         malformed;
         members;
         writers;
         positions;
         crc32;
       ])
