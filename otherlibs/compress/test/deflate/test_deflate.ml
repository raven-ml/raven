(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Deflate, zlib and gzip. Each format reads back what it writes through its
   three decoders, which agree on the streams Python's zlib and gzip wrote,
   whatever the slicing of their input. Malformed streams are refused with an
   error, never an exception. *)

open Windtrap
module C = Compress_deflate
module F = Compress_fixtures

(* Formats *)

type format = {
  name : string;
  compress : ?level:int -> string -> string;
  decompress : string -> (string, string) result;
  encoder : ?level:int -> unit -> C.Encoder.t;
  decoder : unit -> C.Decoder.t;
  decompress_into : F.bigbytes -> F.bigbytes -> (unit, string) result;
}

let deflate =
  {
    name = "deflate";
    compress = C.Deflate.compress;
    decompress = C.Deflate.decompress;
    encoder = C.Deflate.encoder;
    decoder = C.Deflate.decoder;
    decompress_into = C.Deflate.decompress_into;
  }

let zlib =
  {
    name = "zlib";
    compress = C.Zlib.compress;
    decompress = C.Zlib.decompress;
    encoder = C.Zlib.encoder;
    decoder = C.Zlib.decoder;
    decompress_into = C.Zlib.decompress_into;
  }

let gzip =
  {
    name = "gzip";
    compress = C.Gzip.compress;
    decompress = C.Gzip.decompress;
    encoder = C.Gzip.encoder;
    decoder = C.Gzip.decoder;
    decompress_into = C.Gzip.decompress_into;
  }

let formats = [ deflate; zlib; gzip ]
let compress ?level f s = f.compress ?level s

(* [compress_sliced f k s] gives [s] to an encoder [k] bytes at a time. *)
let compress_sliced ?level f k s =
  let e = f.encoder ?level () in
  let src = Bytes.of_string s in
  let b = Buffer.create 256 in
  let rec loop first =
    match C.Encoder.encode e with
    | `Data (bytes, j, l) ->
        Buffer.add_subbytes b bytes j l;
        loop first
    | `Await ->
        let n = Int.min k (String.length s - first) in
        C.Encoder.src e src first n;
        loop (first + n)
    | `End -> Buffer.contents b
  in
  loop 0

(* [decode ~k f s] gives [s] to a decoder [k] bytes at a time. *)
let decode ?(k = 65536) f s =
  let d = f.decoder () in
  let src = Bytes.of_string s in
  let b = Buffer.create 256 in
  let rec loop first =
    match C.Decoder.decode d with
    | `Data (bytes, j, l) ->
        Buffer.add_subbytes b bytes j l;
        loop first
    | `Await ->
        let n = Int.min k (String.length s - first) in
        C.Decoder.src d src first n;
        loop (first + n)
    | `End -> Ok (Buffer.contents b)
    | `Error e -> Error e
  in
  loop 0

(* [decompress_into f s n] decompresses [s] between byte arrays into [n]
   bytes. *)
let decompress_into f s n =
  let dst = F.zeros n in
  Result.map
    (fun () -> F.string_of_bigbytes dst)
    (f.decompress_into (F.bigbytes_of_string s) dst)

(* [decodes f data z] asserts that the three decoders decode [z] to [data]. *)
let decodes f data z =
  equal ~msg:"decompress" string data (require_ok (f.decompress z));
  equal ~msg:"decoder" string data (require_ok (decode f z));
  equal ~msg:"decompress_into" string data
    (require_ok (decompress_into f z (String.length data)))

(* [refused f s error] asserts that the three decoders refuse [s] with a message
   that contains [error]. *)
let refused ?(length = 64) f s error =
  contains ~msg:"decompress" ~sub:error
    (require_error ~msg:"decompress" (f.decompress s));
  contains ~msg:"decoder" ~sub:error (require_error ~msg:"decoder" (decode f s));
  contains ~msg:"decompress_into" ~sub:error
    (require_error ~msg:"decompress_into" (decompress_into f s length))

(* [returns_or_errors f s] asserts that the decoders return or error on [s], as
   they promise for any bytes: never an exception or a crash. *)
let returns_or_errors f s n =
  ignore (f.decompress s);
  ignore (decode ~k:7 f s);
  ignore (decompress_into f s n)

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
   ^ " decompression inverts compress on short strings and on long ones past \
      the window and block sizes") strings (fun s -> decodes f s (compress f s))

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
      equal string s (require_ok (decode zlib z));
      equal string z (compress_sliced ~level zlib k s))

(* [letters seed] is 200 to 400 thousand pseudo-random letters from [a] to [p]:
   about 4 bits each once compressed, so the stream outgrows a decoder's input
   buffer and the decoder resumes inside dynamic Huffman blocks. *)
let letters seed =
  let r = Random.State.make [| seed |] in
  String.init (Random.State.int_in_range r ~min:200_000 ~max:400_000) (fun _ ->
      Char.chr (Char.code 'a' + Random.State.int r 16))

let resumes =
  prop ~tags:[ "slow" ] ~count:20
    "a decoder resumes decoding wherever its input buffer ends in a dynamic \
     block"
    (Gen.triple
       (Gen.of_list
          ~pp:(fun ppf f -> Format.pp_print_string ppf f.name)
          formats)
       (Gen.int_range 1 9) Gen.int)
    (fun (f, level, seed) ->
      let s = letters seed in
      let z = compress ~level f s in
      greater int (String.length z) ~than:65536;
      decodes f s z)

let round_trips =
  group "round trips"
    (List.map inverts formats
    @ [
        levels;
        resumes;
        cases ~name:Fun.id "the empty string round trips"
          [ "deflate"; "zlib"; "gzip" ] (fun name ->
            let f = List.find (fun f -> f.name = name) formats in
            decodes f "" (compress f ""));
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
        "decode whole and through decoders given every slicing" python_streams
        (fun (file, f, data) ->
          let z = F.read file in
          decodes f data z;
          List.iter
            (fun k ->
              equal
                ~msg:(Printf.sprintf "slices of %d bytes" k)
                string data
                (require_ok (decode ~k f z)))
            slice_lengths);
      prop "a decoder's output does not depend on how its input is sliced"
        (Gen.pair
           (Gen.of_list
              ~pp:(fun ppf (file, _, _) -> Format.pp_print_string ppf file)
              python_streams)
           (Gen.int_range 1 4096))
        (fun ((file, f, data), k) ->
          equal string data (require_ok (decode ~k f (F.read file))));
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
          ignore (require_error (decompress_into f z (n - 1)));
          ignore (require_error (decompress_into f z (n + 1))));
      cases
        ~name:(fun f -> f.name)
        "overlapping arrays are refused" formats
        (fun f ->
          let b = F.zeros 64 in
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              f.decompress_into
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
          decodes gzip (String.concat "" l) z);
      test "a member's header is fixed" (fun () ->
          equal string "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
            (String.sub (compress gzip F.text) 0 10));
    ]

(* Encoders *)

(* [outputs step] is the lengths of the slices [step] returns before it awaits
   input or ends, and whether it ended. *)
let outputs step =
  let rec loop acc =
    match step () with
    | `Data (_, _, l) -> loop (l :: acc)
    | `Await -> (List.rev acc, false)
    | `End -> (List.rev acc, true)
    | `Error e -> failf "%s" e
  in
  loop []

let encode e () =
  (C.Encoder.encode e
    :> [ `Await | `Data of bytes * int * int | `End | `Error of string ])

let not_awaiting f = raises_match (Exn.invalid_arg ~substring:"await") f
let out_of_range f = raises_match (Exn.invalid_arg ~substring:"range") f

let encoders =
  group "encoders"
    [
      cases ~name:string_of_int "refuse a level outside 0 to 9" [ -1; 10 ]
        (fun level ->
          raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
              C.Zlib.encoder ~level ());
          raises_match (Exn.invalid_arg ~substring:"level") (fun () ->
              C.Zlib.compress ~level ""));
      test "default to level 6" (fun () ->
          equal string (compress ~level:6 zlib F.text) (compress zlib F.text));
      prop "compress writes a fresh encoder's bytes after any other calls"
        Gen.(
          list ~size:(int_range 1 4)
            (pair (int_range 0 9)
               (frequency
                  [
                    (4, string);
                    ( 1,
                      string_of ~size:(int_range 32_800 36_000)
                        (char_range 'a' 'e') );
                  ])))
        (fun calls ->
          (* each call reuses the state the one before it left *)
          List.iter
            (fun (level, s) ->
              cover "a long input" (String.length s > 32768);
              equal string
                (compress_sliced ~level gzip (String.length s + 1) s)
                (compress ~level gzip s))
            calls);
      test "await input until it ends, then stay ended" (fun () ->
          let e = C.Zlib.encoder () in
          let _, ended = outputs (encode e) in
          is_false ended;
          C.Encoder.src e (Bytes.of_string hello) 0 (String.length hello);
          let _, ended = outputs (encode e) in
          is_false ended;
          C.Encoder.src e Bytes.empty 0 0;
          let _, ended = outputs (encode e) in
          is_true ended;
          let _, ended = outputs (encode e) in
          is_true ended);
      test "refuse input when they do not await it" (fun () ->
          let e = C.Zlib.encoder () in
          let b = Bytes.of_string hello in
          C.Encoder.src e b 0 5;
          not_awaiting (fun () -> C.Encoder.src e b 5 5));
      cases
        ~name:(fun (j, l) -> Printf.sprintf "%d+%d" j l)
        "refuse a range outside the bytes"
        [ (-1, 1); (0, -1); (3, 3); (6, 0) ]
        (fun (j, l) ->
          out_of_range (fun () ->
              C.Encoder.src (C.Zlib.encoder ()) (Bytes.create 5) j l));
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

(* Decoders and errors *)

let decode_step d () = C.Decoder.decode d

let decoders =
  group "decoders"
    [
      test "return slices of at most 64 KiB" (fun () ->
          let d = C.Zlib.decoder () in
          let z = Bytes.of_string (compress zlib F.runs) in
          C.Decoder.src d z 0 (Bytes.length z);
          let lengths, _ = outputs (decode_step d) in
          equal int (String.length F.runs) (List.fold_left ( + ) 0 lengths);
          at_most int (List.fold_left Int.max 0 lengths) ~than:65536);
      test "await the end of the input after a whole stream, then stay ended"
        (fun () ->
          let d = C.Zlib.decoder () in
          let z = Bytes.of_string (compress zlib hello) in
          C.Decoder.src d z 0 (Bytes.length z);
          let _, ended = outputs (decode_step d) in
          is_false ended;
          C.Decoder.src d z 0 0;
          let _, ended = outputs (decode_step d) in
          is_true ended;
          let _, ended = outputs (decode_step d) in
          is_true ended);
      test "stay failed after an error" (fun () ->
          let d = C.Zlib.decoder () in
          C.Decoder.src d (Bytes.of_string "\x00\x00") 0 2;
          let first = C.Decoder.decode d in
          let again = C.Decoder.decode d in
          match (first, again) with
          | `Error a, `Error b -> equal string a b
          | _ -> fail "no error");
      test "refuse input when they do not await it" (fun () ->
          let d = C.Gzip.decoder () in
          let b = Bytes.of_string hello in
          C.Decoder.src d b 0 5;
          not_awaiting (fun () -> C.Decoder.src d b 5 5));
      cases
        ~name:(fun (j, l) -> Printf.sprintf "%d+%d" j l)
        "refuse a range outside the bytes"
        [ (-1, 1); (0, -1); (3, 3); (6, 0) ]
        (fun (j, l) ->
          out_of_range (fun () ->
              C.Decoder.src (C.Zlib.decoder ()) (Bytes.create 5) j l));
      test "an error names the byte where decoding failed" (fun () ->
          let z = compress zlib hello in
          let at = string_of_int (String.length z) in
          let z = z ^ "!" in
          contains ~sub:at (require_error (zlib.decompress z));
          contains ~sub:at (require_error (decode ~k:3 zlib z));
          contains ~sub:at (require_error (decompress_into zlib z 17)));
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
          equal int (crc32_spec s) (C.Crc32.string s));
      prop "checks the range of a string it is given"
        (Gen.triple Gen.string Gen.nat Gen.nat) (fun (s, a, b) ->
          let n = String.length s in
          let first = if n = 0 then 0 else a mod (n + 1) in
          let length = if n - first = 0 then 0 else b mod (n - first + 1) in
          equal int
            (crc32_spec (String.sub s first length))
            (C.Crc32.string ~first ~length s);
          equal int
            (crc32_spec (String.sub s first (n - first)))
            (C.Crc32.string ~first s));
      cases
        ~name:(fun (j, l) -> Printf.sprintf "%d+%d" j l)
        "refuses a range outside the string"
        [ (-1, 1); (0, -1); (3, 3); (6, 0) ]
        (fun (first, length) ->
          out_of_range (fun () -> C.Crc32.string ~first ~length "12345"));
      prop "continues a checksum with ~crc" (Gen.pair Gen.string Gen.string)
        (fun (a, b) ->
          let crc = C.Crc32.string a in
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
         encoders;
         decoders;
         crc32;
       ])
