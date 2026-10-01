(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* LZ4 blocks and frames. The decoders read what Python's lz4 wrote, and refuse,
   with an error, the frames and blocks built here to break each rule of the
   formats. *)

open Windtrap
module L = Compress_lz4
module F = Compress_fixtures

let decompress_with f src n =
  let dst = F.zeros n in
  Result.map
    (fun () -> F.string_of_bigbytes dst)
    (f (F.bigbytes_of_string src) dst)

let block = decompress_with L.Block.decompress
let frame = decompress_with L.Frame.decompress

(* XXH32 with seed 0, from its specification. *)
let xxh32 s =
  let open Int32 in
  let p1 = 0x9E3779B1l and p2 = 0x85EBCA77l and p3 = 0xC2B2AE3Dl in
  let p4 = 0x27D4EB2Fl and p5 = 0x165667B1l in
  let rotl x r = logor (shift_left x r) (shift_right_logical x (32 - r)) in
  let lane i = String.get_int32_le s i in
  let round acc v = mul (rotl (add acc (mul v p2)) 13) p1 in
  let n = String.length s in
  let i = ref 0 in
  let h =
    if n >= 16 then begin
      let v = [| add p1 p2; p2; 0l; neg p1 |] in
      while !i + 16 <= n do
        for k = 0 to 3 do
          v.(k) <- round v.(k) (lane (!i + (4 * k)))
        done;
        i := !i + 16
      done;
      add
        (add (rotl v.(0) 1) (rotl v.(1) 7))
        (add (rotl v.(2) 12) (rotl v.(3) 18))
    end
    else p5
  in
  let h = ref (add h (of_int n)) in
  while !i + 4 <= n do
    h := mul (rotl (add !h (mul (lane !i) p3)) 17) p4;
    i := !i + 4
  done;
  while !i < n do
    h := mul (rotl (add !h (mul (of_int (Char.code s.[!i])) p5)) 11) p1;
    incr i
  done;
  let h = !h in
  let h = mul (logxor h (shift_right_logical h 15)) p2 in
  let h = mul (logxor h (shift_right_logical h 13)) p3 in
  to_int (logxor h (shift_right_logical h 16)) land 0xFFFFFFFF

let le32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xFF))
let le64 n = le32 (n land 0xFFFFFFFF) ^ le32 (n lsr 32)

type block = Raw of string | Lz4 of string

(* A frame of [blocks], whose header and checksums are right unless an argument
   breaks them. *)
let make_frame ?(magic = 0x184D2204) ?(version = 1) ?(linked = false)
    ?(block_checksum = false) ?size ?(content_checksum = false) ?(dict = false)
    ?(reserved = false) ?(bd = 0x40) ?(hc_delta = 0) ?(block_delta = 0)
    ?(content_delta = 0) ~content blocks =
  let flag b v = if b then v else 0 in
  let flg =
    (version lsl 6) lor flag (not linked) 0x20 lor flag block_checksum 0x10
    lor flag (size <> None) 0x08
    lor flag content_checksum 0x04 lor flag reserved 0x02 lor flag dict 0x01
  in
  let descriptor =
    String.make 1 (Char.chr flg)
    ^ String.make 1 (Char.chr bd)
    ^ (match size with Some n -> le64 n | None -> "")
    ^ if dict then le32 7 else ""
  in
  let hc = ((xxh32 descriptor lsr 8) + hc_delta) land 0xFF in
  let block = function
    | Raw s -> (le32 (String.length s lor 0x80000000), s)
    | Lz4 s -> (le32 (String.length s), s)
  in
  le32 magic ^ descriptor
  ^ String.make 1 (Char.chr hc)
  ^ String.concat ""
      (List.map
         (fun b ->
           let header, s = block b in
           header ^ s
           ^
           if block_checksum then le32 ((xxh32 s + block_delta) land 0xFFFFFFFF)
           else "")
         blocks)
  ^ le32 0
  ^
  if content_checksum then le32 ((xxh32 content + content_delta) land 0xFFFFFFFF)
  else ""

let hello = "hello, lz4!\n"

(* An LZ4 block of literals [s]: one sequence and no match. *)
let literals s =
  let n = String.length s in
  if n < 15 then String.make 1 (Char.chr (n lsl 4)) ^ s
  else
    let rec extra n =
      if n >= 255 then "\xff" ^ extra (n - 255) else String.make 1 (Char.chr n)
    in
    "\xf0" ^ extra (n - 15) ^ s

(* Reference vectors *)

let blocks =
  cases ~name:fst "blocks Python's lz4 wrote decode"
    [
      ("text", F.text);
      ("columns", F.columns);
      ("runs", F.runs);
      ("random", F.random);
    ]
    (fun (name, data) ->
      equal string data
        (require_ok
           (block (F.read ("lz4_" ^ name ^ ".lz4b")) (String.length data))))

let frames =
  cases ~name:fst "frames Python's lz4 wrote decode"
    [
      ("lz4_64k_linked.lz4", F.text);
      ("lz4_256k_independent.lz4", F.text);
      ("lz4_1m.lz4", F.text);
      ("lz4_4m.lz4", F.text);
      ("lz4_random.lz4", F.random);
      ("lz4_frames.lz4", "first frame\nsecond frame\n");
      ("lz4_skippable.lz4", "after\n");
      ("lz4_empty.lz4", "");
    ]
    (fun (file, data) ->
      equal string data (require_ok (frame (F.read file) (String.length data))))

let built =
  cases
    ~name:(fun (name, _, _) -> name)
    "frames built here decode"
    [
      ("a raw block", make_frame ~content:hello [ Raw hello ], hello);
      ( "a compressed block",
        make_frame ~content:hello [ Lz4 (literals hello) ],
        hello );
      ( "every checksum and the size",
        make_frame ~block_checksum:true ~content_checksum:true
          ~size:(String.length hello) ~content:hello [ Raw hello ],
        hello );
      ( "a linked block whose match reaches into the block before",
        make_frame ~linked:true ~content:(hello ^ hello)
          [ Lz4 (literals hello); Lz4 "\x08\x0c\x00\x00" ],
        hello ^ hello );
      ( "a long literal run",
        make_frame ~content:F.text [ Lz4 (literals (String.sub F.text 0 600)) ],
        String.sub F.text 0 600 );
    ]
    (fun (_, f, data) ->
      equal string data (require_ok (frame f (String.length data))))

let malformed_frames =
  [
    ("no data", "", "no frame");
    ( "a bad magic number",
      make_frame ~magic:0x184D2205 ~content:hello [ Raw hello ],
      "not an LZ4 frame" );
    ( "the legacy format",
      le32 0x184C2102 ^ le32 5 ^ "hello",
      "legacy frame format" );
    ( "version 2",
      make_frame ~version:2 ~content:hello [ Raw hello ],
      "unknown frame version" );
    ( "a reserved flag",
      make_frame ~reserved:true ~content:hello [ Raw hello ],
      "reserved bits set" );
    ( "a reserved block size bit",
      make_frame ~bd:0x41 ~content:hello [ Raw hello ],
      "reserved bits set" );
    ( "a block maximum size below 64 KiB",
      make_frame ~bd:0x30 ~content:hello [ Raw hello ],
      "invalid block maximum size" );
    ( "a header checksum that does not match",
      make_frame ~hc_delta:1 ~content:hello [ Raw hello ],
      "header checksum mismatch" );
    ( "a dictionary",
      make_frame ~dict:true ~content:hello [ Raw hello ],
      "dictionary needed" );
    ( "a block larger than the maximum",
      make_frame ~content:hello [ Raw (String.make 65537 'x') ],
      "larger than the frame's maximum" );
    ( "a block checksum that does not match",
      make_frame ~block_checksum:true ~block_delta:1 ~content:hello
        [ Raw hello ],
      "block checksum mismatch" );
    ( "a content checksum that does not match",
      make_frame ~content_checksum:true ~content_delta:1 ~content:hello
        [ Raw hello ],
      "content checksum mismatch" );
    ( "a content size that does not match",
      make_frame ~size:(String.length hello + 1) ~content:hello [ Raw hello ],
      "content size mismatch" );
    ( "data after a frame",
      make_frame ~content:hello [ Raw hello ] ^ "!",
      "truncated data" );
    ( "a block of a match at offset 0",
      make_frame ~content:hello [ Lz4 "\x10a\x00\x00" ],
      "match at offset 0" );
    ( "a block of a match before the data",
      make_frame ~content:hello [ Lz4 "\x10a\x02\x00" ],
      "match before the data" );
    ( "a block that ends inside a sequence",
      make_frame ~content:hello [ Lz4 "\x10a\x01" ],
      "truncated data" );
    ( "a block that ends with a match",
      make_frame ~content:hello [ Lz4 "\x10a\x01\x00" ],
      "truncated data" );
    ( "an independent block whose match reaches into the block before",
      make_frame ~content:hello [ Lz4 (literals hello); Lz4 "\x08\x0c\x00\x00" ],
      "match before the data" );
  ]

let malformed_blocks =
  [
    ("no data", "", "truncated data");
    ("literals past the end", "\x50abc", "truncated data");
    ("a literal length past the end", "\xf0\xff", "truncated data");
    ("a match at offset 0", "\x10a\x00\x00", "match at offset 0");
    ("a match before the data", "\x10a\x02\x00", "match before the data");
    ("an offset cut short", "\x10a\x01", "truncated data");
    ("a match length past the end", "\x1fa\x01\x00\xff", "truncated data");
    ("a match last", "\x10a\x01\x00", "truncated data");
  ]

let flip s at bit =
  String.mapi
    (fun i c -> if i = at then Char.chr (Char.code c lxor (1 lsl bit)) else c)
    s

let malformed =
  group "malformed data"
    [
      cases
        ~name:(fun (name, _, _) -> name)
        "frames are refused" malformed_frames
        (fun (_, f, error) ->
          contains ~sub:error (require_error (frame f (String.length hello))));
      cases
        ~name:(fun (name, _, _) -> name)
        "blocks are refused" malformed_blocks
        (fun (_, b, error) -> contains ~sub:error (require_error (block b 64)));
      cases ~name:fst "data cut short at any byte is refused"
        [
          ( "a frame after a skippable frame",
            (`Frame, F.read "lz4_skippable.lz4", "after\n") );
          ("an empty frame", (`Frame, F.read "lz4_empty.lz4", ""));
          ("a block", (`Block, F.read "lz4_runs.lz4b", F.runs));
        ]
        (fun (_, (kind, src, data)) ->
          let f = match kind with `Frame -> frame | `Block -> block in
          for n = 0 to String.length src - 1 do
            ignore
              (require_error ~msg:(string_of_int n)
                 (f (String.sub src 0 n) (String.length data)))
          done);
      prop "data with a bit flipped decompresses or is refused"
        (let open Gen in
         let* file =
           of_list [ "lz4_64k_linked.lz4"; "lz4_frames.lz4"; "lz4_text.lz4b" ]
         in
         let src = F.read file in
         let+ at = int_range 0 (String.length src - 1)
         and+ bit = int_range 0 7 in
         (file, flip src at bit))
        (fun (file, src) ->
          let f = if Filename.extension file = ".lz4b" then block else frame in
          ignore (f src (String.length F.text)));
      cases ~name:fst "a destination one byte short or long is refused"
        [
          ("a block", (block, F.read "lz4_text.lz4b"));
          ("a frame", (frame, F.read "lz4_1m.lz4"));
        ]
        (fun (_, (f, src)) ->
          let n = String.length F.text in
          ignore (require_error (f src (n - 1)));
          ignore (require_error (f src (n + 1))));
      test "an error names the byte where decoding failed" (fun () ->
          contains ~sub:"at byte 2" (require_error (block "\x10a\x00\x00" 2)));
      test "overlapping arrays are refused" (fun () ->
          let b = F.zeros 64 in
          let src = Bigarray.Array1.sub b 0 40
          and dst = Bigarray.Array1.sub b 30 34 in
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              L.Block.decompress src dst);
          raises_match (Exn.invalid_arg ~substring:"overlap") (fun () ->
              L.Frame.decompress src dst));
    ]

let checksums =
  test "the header checksums built here are those Python's lz4 writes"
    (fun () ->
      let f = F.read "lz4_frames.lz4" in
      let descriptor = String.sub f 4 10 in
      equal int (Char.code f.[14]) ((xxh32 descriptor lsr 8) land 0xFF))

let () =
  exit
    (run "compress.lz4"
       [ group "reference" [ blocks; frames; built; checksums ]; malformed ])
