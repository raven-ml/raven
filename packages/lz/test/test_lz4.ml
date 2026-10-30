(* LZ4 blocks and frames. The hand-built inputs follow lz4's "LZ4 Block
   Format Description" and "LZ4 Frame Format Description" 1.6.4. An
   error's expected offset is the first byte of what is wrong, as lz.mli
   states it: of a sequence, the token for its literals and the offset
   field for its match; of a frame, the first byte of the field. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib

(* Buffers *)

let guard = 64

(* A destination of [n] bytes at an odd offset, followed by [guard] bytes
   that a codec must not write. *)
let guarded n =
  let whole = Lz_support.create (3 + n + guard) in
  (whole, Bigarray.Array1.sub whole 3 n)

let untouched (whole, view) =
  let n = Bigarray.Array1.dim view in
  equal string ~msg:"the bytes past the destination"
    (String.make guard '\xa5')
    (String.init guard (fun i -> Char.chr (Bigarray.Array1.get whole (3 + n + i))))

(* A result as one line: its progress, or the error's kind and offset. *)
let seen = function
  | Ok { Lz.consumed; produced } ->
      Printf.sprintf "consumed %d, produced %d" consumed produced
  | Error { Lz.at; why } ->
      let kind =
        match why with
        | Lz.Truncated -> "truncated"
        | Malformed _ -> "malformed"
        | Too_long -> "too long"
        | Unsupported _ -> "unsupported"
      in
      Printf.sprintf "%s at %d" kind at

let malformed_at at = Printf.sprintf "malformed at %d" at

(* Codecs *)

type codec = {
  name : string;
  encode : Lz.bigbytes -> Lz.bigbytes -> int option;
  encode_string : string -> string;
  decode : Lz.bigbytes -> Lz.bigbytes -> (Lz.progress, Lz.error) result;
  bound : int -> int;
}

let block = Lz.Lz4.Block.{ name = "block"; encode; encode_string; decode; bound }
let frame = Lz.Lz4.Frame.{ name = "frame"; encode; encode_string; decode; bound }

let encode c s =
  let dst = Lz_support.create (c.bound (String.length s)) in
  let k = require_some ~msg:"fits its bound" (c.encode (Lz_support.of_string s) dst) in
  Lz_support.to_string ~length:k dst

let default_cap s = (4 * String.length s) + 64

let decode c ?cap s =
  let cap = Option.value cap ~default:(default_cap s) in
  let dst = Lz_support.create cap in
  match c.decode (Lz_support.of_string s) dst with
  | Ok { Lz.consumed = _; produced } ->
      Ok (Lz_support.to_string ~length:produced dst)
  | Error e -> Error e

let decode_seen c ?cap s =
  let cap = Option.value cap ~default:(default_cap s) in
  seen (c.decode (Lz_support.of_string s) (Lz_support.create cap))

(* Drawn inputs, each at an offset of 0 to 7 bytes into its buffer. *)
let placed = Gen.pair Lz_support.recipe (Gen.int_range 0 7)
let large = List.map (fun r -> (r, 1)) Lz_support.large

let cover_recipe r s =
  let open Lz_support in
  cover "empty" (s = "");
  cover "one byte" (String.length s = 1);
  cover "a match" (match r with Match _ -> true | _ -> false);
  cover "noise" (match r with Noise _ -> true | _ -> false);
  cover "a run" (match r with Run _ -> true | _ -> false);
  cover "a concatenation" (match r with Concat _ -> true | _ -> false)

(* Laws *)

let round_trip c =
  prop ~examples:large "decode (encode x) = x" placed (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let n = String.length s in
      let src = Lz_support.placed ~offset s in
      let dst = Lz_support.room ~offset (c.bound n) in
      let k = require_some (c.encode src dst) in
      let out = Lz_support.room ~offset:(offset + 1) n in
      match c.decode (Bigarray.Array1.sub dst 0 k) out with
      | Error e -> failf "%a" (Testable.pp Lz_support.error) e
      | Ok p ->
          equal Lz_support.progress { Lz.consumed = k; produced = n } p;
          equal Lz_support.decoded (Ok s) (Ok (Lz_support.to_string out)))

let bound c =
  prop ~examples:large "a destination of bound n holds the encoding" placed
    (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let g = guarded (c.bound (String.length s)) in
      ignore (require_some (c.encode (Lz_support.placed ~offset s) (snd g)));
      untouched g)

let short c =
  prop "a destination one byte short of the encoding is refused" placed
    (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let k = String.length (encode c s) in
      let src = Lz_support.placed ~offset s in
      let g = guarded (k - 1) in
      is_none ~msg:"one byte short" (c.encode src (snd g));
      untouched g;
      equal (option int) ~msg:"exactly its length" (Some k)
        (c.encode src (Lz_support.create k)))

let strings c =
  prop "decode (encode_string x) = x" Lz_support.recipe (fun r ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let z = c.encode_string s in
      let dst = Lz_support.create (String.length s) in
      match c.decode (Lz_support.of_string z) dst with
      | Error e -> failf "%a" (Testable.pp Lz_support.error) e
      | Ok p ->
          equal Lz_support.progress
            { Lz.consumed = String.length z; produced = String.length s }
            p;
          equal string s (Lz_support.to_string dst))

let laws c = group c.name [ round_trip c; bound c; short c; strings c ]

(* Frames and blocks by hand *)

let le16 n = String.init 2 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
let le32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
let le64 n = le32 (n land 0xffffffff) ^ le32 (n lsr 32)

let byte n = String.make 1 (Char.chr n)
let xxh32 s = Int32.to_int (Lz.Xxh32.string s) land 0xffffffff

(* FLG: the version 01 in bits 7-6, then B.Indep, B.Checksum, C.Size,
   C.Checksum, a reserved bit and DictID. BD: the block maximum size in
   bits 6-4, 4 for 64 KiB to 7 for 4 MiB. *)
let version = 0x40
let independent = 0x20
let block_checksum = 0x10
let content_size = 0x08
let content_checksum = 0x04
let dict_id = 0x01
let max64k = 0x40
let max4m = 0x70

(* A frame header: magic, FLG, BD, the optional fields, and HC, the second
   byte of the descriptor's XXH32. Without options it is 7 bytes. *)
let header ?(flg = version) ?(bd = max64k) ?size ?dict () =
  let d =
    byte flg ^ byte bd
    ^ (match size with Some n -> le64 n | None -> "")
    ^ match dict with Some n -> le32 n | None -> ""
  in
  le32 0x184D2204 ^ d ^ byte ((xxh32 d lsr 8) land 0xff)

let stored s = le32 (String.length s lor 0x80000000) ^ s
let compressed b = le32 (String.length b) ^ b
let end_mark = le32 0

let skippable ?(nibble = 0) s =
  le32 (0x184D2A50 + nibble) ^ le32 (String.length s) ^ s

(* A sequence of under 15 literals and a match of under 19 bytes, and a
   last sequence of under 15 literals. *)
let sequence lits ~offset ~length =
  byte ((String.length lits lsl 4) lor (length - 4)) ^ lits ^ le16 offset

let last lits = byte (String.length lits lsl 4) ^ lits

(* A length past a token's 15: 255s, then the rest. *)
let length_bytes n = String.make ((n - 15) / 255) '\xff' ^ byte ((n - 15) mod 255)

(* "a", a match of [length] bytes from 1 back, and 5 literals: the
   match's last byte is the seventh from the end. *)
let run_block length =
  "\x1f" ^ "a" ^ le16 1 ^ length_bytes (length - 4) ^ last "bcdef"

(* Blocks drawn as sequences *)

(* A sequence of the block format: literals, then a match of [length]
   bytes, 4 or more, from [offset] back. *)
type sequence = { lits : string; offset : int; length : int }

let lengthed n = if n < 15 then "" else length_bytes n

let sequence_bytes { lits; offset; length } =
  let l = String.length lits and m = length - 4 in
  byte ((min l 15 lsl 4) lor min m 15) ^ lengthed l ^ lits ^ le16 offset ^ lengthed m

let last_bytes lits =
  let l = String.length lits in
  byte (min l 15 lsl 4) ^ lengthed l ^ lits

(* What the sequences and the last literals decode to. A match reads byte
   by byte, so one longer than its offset repeats what it has just
   written. *)
let meaning seqs final =
  let b = Buffer.create 256 in
  List.iter
    (fun { lits; offset; length } ->
      Buffer.add_string b lits;
      for _ = 1 to length do
        Buffer.add_char b (Buffer.nth b (Buffer.length b - offset))
      done)
    seqs;
  Buffer.add_string b final;
  Buffer.contents b

let output seqs =
  List.fold_left (fun n s -> n + String.length s.lits + s.length) 0 seqs

(* Up to 30 sequences of every literal and match length class, half of
   them a match of 9 to 40 bytes from under 16 back, which the fast loop
   copies in overlapping stores; then last literals that keep the
   end-of-block rules. *)
let sequences =
  let open Gen in
  let lits n = string_of ~size:(constant n) char in
  let one produced =
    let* n = frequency [ (6, int_range 0 14); (1, int_range 15 300) ] in
    let n = if produced = 0 then max n 1 else n in
    let* lits = lits n in
    let reach = min 65535 (produced + n) in
    let+ offset, length =
      frequency
        [
          (3, pair (int_range 1 (min 15 reach)) (int_range 9 40));
          (2, pair (int_range 1 reach) (int_range 4 30));
          (1, pair (int_range 1 reach) (int_range 19 600));
        ]
    in
    { lits; offset; length }
  in
  let rec go k produced acc =
    if k = 0 then constant (List.rev acc)
    else
      let* s = one produced in
      go (k - 1) (produced + String.length s.lits + s.length) (s :: acc)
  in
  let pp ppf (seqs, final) =
    List.iter
      (fun s ->
        Format.fprintf ppf "(%d literals, offset %d, length %d) "
          (String.length s.lits) s.offset s.length)
      seqs;
    Format.fprintf ppf "%d last literals" (String.length final)
  in
  with_pp pp
    (let* k = int_range 1 30 in
     let* seqs = go k 0 [] in
     let last_match = (List.nth seqs (k - 1)).length in
     let* n = int_range (max 5 (12 - last_match)) 40 in
     let+ final = lits n in
     (seqs, final))

(* The fast loop decodes every sequence after which 32 bytes of the block
   remain, given room in the destination. *)
let in_fast_loop seqs total =
  let rec go pos acc = function
    | [] -> List.rev acc
    | s :: rest ->
        let pos = pos + String.length (sequence_bytes s) in
        go pos ((s, total - pos >= 32) :: acc) rest
  in
  go 0 [] seqs

let drawn =
  prop "a block decodes to what its sequences mean" sequences (fun (seqs, final) ->
      let b = String.concat "" (List.map sequence_bytes seqs) ^ last_bytes final in
      let raw = meaning seqs final in
      let marked = in_fast_loop seqs (String.length b) in
      let fast p = List.exists (fun (s, fast) -> fast && p s) marked in
      for d = 1 to 7 do
        cover
          (Printf.sprintf "offset %d over 8 bytes in the fast loop" d)
          (fast (fun s -> s.offset = d && s.length > 8))
      done;
      cover "offset 8 to 15 over 8 bytes in the fast loop"
        (fast (fun s -> s.offset >= 8 && s.offset < 16 && s.length > 8));
      cover "a long literal run in the fast loop"
        (fast (fun s -> String.length s.lits >= 15));
      cover "a long match in the fast loop" (fast (fun s -> s.length >= 19));
      let offset = String.length raw mod 8 in
      let dst = Lz_support.room ~offset (String.length raw + 64) in
      let r = Lz.Lz4.Block.decode (Lz_support.placed ~offset b) dst in
      equal (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed = String.length b; produced = String.length raw })
        r;
      equal string raw (Lz_support.to_string ~length:(String.length raw) dst))

let bad_offsets =
  (* Valid sequences, then one whose offset is 0 or before the output,
     then up to 63 bytes: malformed at the offset field, in either loop,
     whatever follows it. *)
  let gen =
    Gen.(
      pair sequences
        (triple bool (string_of ~size:(int_range 0 14) char) (int_range 0 63)))
  in
  prop "a bad offset is malformed at its offset field" gen
    (fun ((seqs, _), (zero, lits, filler)) ->
      let prefix = String.concat "" (List.map sequence_bytes seqs) in
      let reach = output seqs + String.length lits in
      let offset = if zero || reach >= 65535 then 0 else reach + 1 in
      let bad = sequence_bytes { lits; offset; length = 8 } in
      let b = prefix ^ bad ^ String.make filler 'x' in
      let tail = String.length b - String.length prefix in
      cover "a zero offset" (offset = 0);
      cover "an offset before the output" (offset > 0);
      cover "in the fast loop" (tail >= 32);
      cover "in the exact loop" (tail < 32);
      let cap = reach + 64 + filler in
      equal string
        (malformed_at (String.length prefix + 1 + String.length lits))
        (decode_seen block ~cap b))

(* Expected values from the format documents *)

(* A block has no end mark: bytes after it are read as more of the block.
   After the last literals of "hello"'s block, one byte is a cut offset
   field; an offset of 1, the token's match of 4 and a sequence of 8
   literals decode to more, 12 bytes from the match to the end. *)
let after_block =
  test "bytes after a block are read as more of it" (fun () ->
      let b = encode block "hello" in
      let n = String.length b in
      equal string
        (Printf.sprintf "truncated at %d" (n + 1))
        (decode_seen block (b ^ "\x00"));
      let more = b ^ "\x01\x00\x80abcdefgh" in
      equal string
        (Printf.sprintf "consumed %d, produced 17" (String.length more))
        (decode_seen block more);
      equal Lz_support.decoded
        (Ok ("hello" ^ "oooo" ^ "abcdefgh"))
        (decode block more))

let formats =
  group "the format documents"
    [
      after_block;
      test "an empty block is a zero token" (fun () ->
          equal string "\x00" (encode block "");
          equal string "\x00" (Lz.Lz4.Block.encode_string "");
          equal Lz_support.decoded (Ok "") (decode block "\x00"));
      test "15 literals take a zero byte after the token" (fun () ->
          let s = "abcdefghijklmno" in
          equal Lz_support.decoded (Ok s) (decode block ("\xf0\x00" ^ s)));
      test "280 literals take 15, then 255 and 10" (fun () ->
          let s = String.init 280 (fun i -> Char.chr (i land 0xff)) in
          equal Lz_support.decoded (Ok s) (decode block ("\xf0\xff\x0a" ^ s)));
      test "an offset of 1 repeats the last byte" (fun () ->
          equal Lz_support.decoded
            (Ok ("a" ^ String.make 8 'a' ^ "bcdef"))
            (decode block (sequence "a" ~offset:1 ~length:8 ^ last "bcdef")));
      test "an offset reaches 65535 bytes back" (fun () ->
          let lits = String.init 65535 (fun i -> Char.chr ((i * 7) land 0xff)) in
          let b = "\xf4" ^ length_bytes 65535 ^ lits ^ le16 65535 ^ last "vwxyz" in
          equal Lz_support.decoded
            (Ok (lits ^ String.sub lits 0 8 ^ "vwxyz"))
            (decode block b));
      test "a match of 4 MiB" (fun () ->
          equal Lz_support.decoded
            (Ok ("a" ^ String.make (4 * mib) 'a' ^ "bcdef"))
            (decode block ~cap:((4 * mib) + 6) (run_block (4 * mib))));
      test "a frame is magic, descriptor, blocks and end mark" (fun () ->
          let f = header () ^ stored "hello" ^ end_mark in
          equal string "consumed 20, produced 5" (decode_seen frame f);
          equal Lz_support.decoded (Ok "hello") (decode frame f));
      test "a frame's block holds up to 4 MiB" (fun () ->
          let s = Lz_support.corpus Random (4 * mib) in
          let b = run_block ((4 * mib) - 6) in
          equal Lz_support.decoded
            (Ok (s ^ "a" ^ String.make ((4 * mib) - 6) 'a' ^ "bcdef"))
            (decode frame ~cap:(8 * mib)
               (header ~bd:max4m () ^ stored s ^ compressed b ^ end_mark)));
      test "a stored block of 0 bytes is a block" (fun () ->
          equal Lz_support.decoded (Ok "ab")
            (decode frame
               (header () ^ stored "a" ^ stored "" ^ stored "b" ^ end_mark)));
      test "the content checksum is the content's XXH32" (fun () ->
          let f =
            header ~flg:(version lor content_checksum) ()
            ^ stored "hello" ^ end_mark ^ le32 (xxh32 "hello")
          in
          equal Lz_support.decoded (Ok "hello") (decode frame f));
      test "a block checksum is the XXH32 of the block's bytes" (fun () ->
          let b = last "hello" in
          let f =
            header ~flg:(version lor block_checksum) ()
            ^ compressed b ^ le32 (xxh32 b) ^ end_mark
          in
          equal Lz_support.decoded (Ok "hello") (decode frame f));
      test "a frame of no blocks is empty" (fun () ->
          equal string "consumed 11, produced 0"
            (decode_seen frame (header () ^ end_mark)));
      test "the empty string is a frame of no blocks" (fun () ->
          (* The header's 15 bytes, then the end mark and the content's
             checksum. *)
          let f = Lz.Lz4.Frame.encode_string "" in
          equal int ~msg:"length" 23 (String.length f);
          equal string ~msg:"after the header" (end_mark ^ le32 (xxh32 ""))
            (String.sub f 15 8);
          equal string "consumed 23, produced 0" (decode_seen frame f));
      test "skippable frames produce nothing" (fun () ->
          let f =
            skippable "xyz" ^ header () ^ stored "ab" ^ end_mark
            ^ skippable ~nibble:15 ""
          in
          equal string
            (Printf.sprintf "consumed %d, produced 2" (String.length f))
            (decode_seen frame f);
          equal string "consumed 8, produced 0" (decode_seen frame (skippable "")));
      test "frames are read to the input's end" (fun () ->
          let f = header () ^ stored "ab" ^ end_mark in
          equal string
            (Printf.sprintf "consumed %d, produced 4" (2 * String.length f))
            (decode_seen frame (f ^ f));
          equal string (malformed_at (String.length f))
            (decode_seen frame (f ^ "trailing")));
    ]

(* The frame encoder's promises, as Lz.Lz4.Frame states them. *)
let choices =
  let flg f = Char.code f.[4] in
  group "the frame encoder"
    [
      cases ~name:string_of_int
        "writes independent blocks, the content size and the content checksum"
        [ 0; 100; 5 * mib ]
        (fun n ->
          let s = Lz_support.corpus Columns n in
          let f = encode frame s in
          let len = String.length f in
          equal int ~msg:"FLG"
            (version lor independent lor content_size lor content_checksum)
            (flg f);
          equal string ~msg:"content size" (le64 n) (String.sub f 6 8);
          equal string ~msg:"end mark" end_mark (String.sub f (len - 8) 4);
          equal string ~msg:"content checksum" (le32 (xxh32 s))
            (String.sub f (len - 4) 4));
    ]

(* Vendor goldens *)

let golden file =
  List.find (fun g -> g.Lz_support.file = file) (Lz_support.goldens_of "lz4f")

let decodes g =
  let raw = Lazy.force g.Lz_support.raw and data = Lazy.force g.data in
  equal string
    (Printf.sprintf "consumed %d, produced %d" (String.length data)
       (String.length raw))
    (decode_seen frame ~cap:(String.length raw) data);
  equal Lz_support.decoded (Ok raw) (decode frame ~cap:(String.length raw) data)

let vendor =
  group "vendor"
    [
      group "blocks" [ Lz_support.agreement "lz4b" ];
      group "frames" [ Lz_support.agreement "lz4f" ];
      test "frames.lz4: two frames around a skippable frame" (fun () ->
          let g = golden "frames.lz4" in
          equal int ~msg:"members" 3 g.members;
          decodes g);
      test "linked64k.lz4: linked blocks of 64 KiB" (fun () ->
          let g = golden "linked64k.lz4" in
          let data = Lazy.force g.data in
          equal int ~msg:"B.Indep" 0 (Char.code data.[4] land independent);
          equal int ~msg:"BD" max64k (Char.code data.[5]);
          decodes g);
      test "block4m.lz4: block checksums" (fun () ->
          (* Asked for blocks of 4 MiB, lz4 declares the smallest size
             that holds the content. *)
          let g = golden "block4m.lz4" in
          let data = Lazy.force g.data in
          equal int ~msg:"B.Checksum" block_checksum
            (Char.code data.[4] land block_checksum);
          decodes g);
    ]

(* Several frames *)

let frames =
  prop "decode (encode a ^ skippable ^ encode b) = a ^ b"
    (Gen.pair Lz_support.recipe Lz_support.recipe) (fun (a, b) ->
      let a = Lz_support.build a and b = Lz_support.build b in
      cover "an empty first frame" (a = "");
      let f = encode frame a ^ skippable "between" ^ encode frame b in
      let n = String.length a + String.length b in
      equal string
        (Printf.sprintf "consumed %d, produced %d" (String.length f) n)
        (decode_seen frame ~cap:n f);
      equal Lz_support.decoded (Ok (a ^ b)) (decode frame ~cap:n f))

(* Refusals *)

let hostile_blocks =
  group "hostile blocks"
    [
      test "an empty block is truncated" (fun () ->
          equal string "truncated at 0" (decode_seen block ""));
      test "a block ending after a match is truncated" (fun () ->
          let b = sequence "abcd" ~offset:4 ~length:8 in
          equal string
            (Printf.sprintf "truncated at %d" (String.length b))
            (decode_seen block b));
      test "a block cut inside its last literals is truncated" (fun () ->
          let b = sequence "abcd" ~offset:4 ~length:8 ^ "\x50" in
          equal string
            (Printf.sprintf "truncated at %d" (String.length b))
            (decode_seen block b));
      test "an offset of 0 is malformed at the offset" (fun () ->
          equal string (malformed_at 2)
            (decode_seen block (sequence "a" ~offset:0 ~length:8 ^ last "bcdefgh")));
      test "an offset before the output is malformed at the offset" (fun () ->
          equal string (malformed_at 2)
            (decode_seen block (sequence "a" ~offset:2 ~length:8 ^ last "bcdefgh")));
      test "a bad offset is malformed before its match length ends" (fun () ->
          (* The match length's 255s go on past the input's end, but no
             byte after them makes the offset valid. *)
          equal string (malformed_at 2) (decode_seen block "\x1fa\x00\x00\xff\xff");
          equal string (malformed_at 2) (decode_seen block "\x1fa\x05\x00\xff\xff"));
      test "the last match starts at least 12 bytes before the end" (fun () ->
          (* 1 literal, a match of 7, 5 literals: the match starts 12
             bytes before the end; with a match of 6, 11 bytes, and the
             last sequence, at 4, breaks the rule. *)
          equal Lz_support.decoded
            (Ok ("a" ^ String.make 7 'a' ^ "bcdef"))
            (decode block (sequence "a" ~offset:1 ~length:7 ^ last "bcdef"));
          equal string (malformed_at 4)
            (decode_seen block (sequence "a" ~offset:1 ~length:6 ^ last "bcdef")));
      test "the last 5 bytes are literals" (fun () ->
          equal string (malformed_at 4)
            (decode_seen block (sequence "a" ~offset:1 ~length:18 ^ last "bcde")));
      test "literals past the destination are too long at their token" (fun () ->
          equal string "too long at 0" (decode_seen block ~cap:3 (last "abcdefg"));
          equal string "too long at 4"
            (decode_seen block ~cap:10
               (sequence "a" ~offset:1 ~length:5 ^ last "bcdefgh")));
      test "literals past the destination are too long before the input ends"
        (fun () ->
          (* 100 literals stated, 50 present, room for 10. *)
          equal string "too long at 0"
            (decode_seen block ~cap:10 ("\xf0\x55" ^ String.make 50 'x')));
      test "a match past the destination is too long at its offset" (fun () ->
          equal string "too long at 2"
            (decode_seen block ~cap:8 (sequence "a" ~offset:1 ~length:8 ^ last "bcdef")));
      test "a zip bomb ends at the destination" (fun () ->
          (* 10 MiB of length bytes state a match of 2.5 GiB. *)
          let b = run_block ((10 * mib * 255) + 19) in
          let g = guarded mib in
          equal string "too long at 2"
            (seen (Lz.Lz4.Block.decode (Lz_support.of_string b) (snd g)));
          untouched g);
    ]

let hostile_frames =
  group "hostile frames"
    [
      test "an empty input is truncated" (fun () ->
          equal string "truncated at 0" (decode_seen frame ""));
      test "a byte that begins no frame is malformed at its offset" (fun () ->
          equal string (malformed_at 0) (decode_seen frame "\x04\x22\x4d\x19abc");
          equal string (malformed_at 0) (decode_seen frame "\x00\x22\x4d\x18abc");
          equal string (malformed_at 0) (decode_seen frame "\x5f\x2b");
          equal string (malformed_at 11) (decode_seen frame (header () ^ end_mark ^ "\x00"));
          equal string (malformed_at 19)
            (decode_seen frame (skippable "" ^ header () ^ end_mark ^ "\x04\x22\x4d\x19")));
      test "a legacy frame is unsupported at its magic number" (fun () ->
          equal string "unsupported at 0"
            (decode_seen frame (le32 0x184C2102 ^ le32 0));
          equal string "unsupported at 11"
            (decode_seen frame (header () ^ end_mark ^ le32 0x184C2102 ^ le32 0)));
      test "a cut magic number is truncated" (fun () ->
          equal string "truncated at 2" (decode_seen frame "\x04\x22");
          equal string "truncated at 1" (decode_seen frame "\x5a");
          equal string "truncated at 13"
            (decode_seen frame (header () ^ end_mark ^ "\x04\x22")));
      cases ~name:fst "the version is 01"
        [ ("00", 0x00); ("10", 0x80); ("11", 0xc0) ]
        (fun (_, flg) ->
          equal string (malformed_at 4) (decode_seen frame (header ~flg () ^ end_mark)));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "reserved bits are 0"
        [
          ("FLG bit 1", version lor 0x02, max64k, 4);
          ("BD bit 7", version, max64k lor 0x80, 5);
          ("BD bit 3", version, max64k lor 0x08, 5);
          ("BD bit 0", version, max64k lor 0x01, 5);
        ]
        (fun (_, flg, bd, at) ->
          equal string (malformed_at at)
            (decode_seen frame (header ~flg ~bd () ^ end_mark)));
      cases ~name:string_of_int "block maximum sizes 0 to 3 are undefined"
        [ 0; 1; 2; 3 ]
        (fun id ->
          equal string (malformed_at 5)
            (decode_seen frame (header ~bd:(id lsl 4) () ^ end_mark)));
      test "a wrong header checksum is malformed at HC" (fun () ->
          let h = header () in
          let h = String.sub h 0 6 ^ byte (Char.code h.[6] lxor 1) in
          equal string (malformed_at 6) (decode_seen frame (h ^ end_mark)));
      test "a dictionary is unsupported at its identifier" (fun () ->
          equal string "unsupported at 6"
            (decode_seen frame
               (header ~flg:(version lor dict_id) ~dict:7 () ^ end_mark)));
      test "a block above the maximum size is malformed at its size" (fun () ->
          equal string (malformed_at 7)
            (decode_seen frame
               (header () ^ stored (String.make ((64 * kib) + 1) 'x') ^ end_mark));
          equal Lz_support.decoded
            (Ok (String.make (64 * kib) 'x'))
            (decode frame (header () ^ stored (String.make (64 * kib) 'x') ^ end_mark)));
      test "a block decoding past the maximum size is malformed at the code"
        (fun () ->
          (* A match, whose offset is at 13, of 64 KiB after a literal;
             and literals at 11 that state 64 KiB + 1. *)
          equal string (malformed_at 13)
            (decode_seen frame ~cap:mib
               (header () ^ compressed (run_block (64 * kib)) ^ end_mark));
          equal string (malformed_at 11)
            (decode_seen frame ~cap:mib
               (header () ^ compressed ("\xf0" ^ length_bytes ((64 * kib) + 1) ^ "xyz")
               ^ end_mark)));
      cases
        ~name:(fun (n, _, _) -> n)
        "a content size that differs is malformed at the size field"
        [
          ("above, in a stored block", 6, stored "hello");
          ("below, in a stored block", 4, stored "hello");
          ("below, in literals", 3, compressed (last "hello"));
          ("below, in a match", 12, compressed (run_block 20));
          ("above, by a block", 5, stored "he" ^ compressed (last "llo") ^ stored "ab");
        ]
        (fun (_, size, blocks) ->
          let h = header ~flg:(version lor content_size) ~size () in
          equal string (malformed_at 6) (decode_seen frame ~cap:64 (h ^ blocks ^ end_mark)));
      test "a wrong block checksum is malformed at the checksum" (fun () ->
          let b = last "hello" in
          let h = header ~flg:(version lor block_checksum) () in
          equal string
            (malformed_at (String.length h + 4 + String.length b))
            (decode_seen frame (h ^ compressed b ^ le32 (xxh32 b lxor 1) ^ end_mark)));
      test "a wrong content checksum is malformed at the checksum" (fun () ->
          let f =
            header ~flg:(version lor content_checksum) () ^ stored "hello" ^ end_mark
          in
          equal string (malformed_at (String.length f))
            (decode_seen frame (f ^ le32 (xxh32 "hello" lxor 1))));
      test "a block ending inside a sequence is malformed at its token" (fun () ->
          (* The block's token is at 11. *)
          let ends b = decode_seen frame (header () ^ compressed b ^ end_mark) in
          equal string (malformed_at 11) (ends "\x50ab");
          equal string (malformed_at 11) (ends "\x40abcd\x04");
          equal string (malformed_at 11) (ends "\x4fabcd\x04\x00\xff");
          equal string (malformed_at 18) (ends (sequence "abcd" ~offset:4 ~length:8 ^ "\x50a")));
      test "a block ending after a match is malformed at its last token" (fun () ->
          let ends ?cap b = decode_seen frame ?cap (header () ^ compressed b ^ end_mark) in
          equal string (malformed_at 11) (ends (sequence "abcd" ~offset:4 ~length:8));
          (* 32 bytes whose match length runs to the block's end: the fast
             loop decodes it. *)
          let long = "\xef" ^ String.make 14 'a' ^ le16 1 ^ length_bytes (15 + (14 * 255)) in
          equal int ~msg:"the long sequence's length" 32 (String.length long);
          equal string (malformed_at 11) (ends ~cap:mib long);
          equal string (malformed_at 18)
            (ends ~cap:mib (sequence "abcd" ~offset:4 ~length:8 ^ long)));
      test "a bad offset is malformed at the offset in a frame" (fun () ->
          equal string (malformed_at 13)
            (decode_seen frame (header () ^ compressed "\x1fa\x00\x00\xff\xff" ^ end_mark)));
      test "a cut block is decoded as far as the input goes" (fun () ->
          (* A block of 100 bytes cut after 6: malformed if a byte before
             the cut proves it, truncated otherwise. *)
          let cut b = decode_seen frame (header () ^ le32 100 ^ b) in
          equal string (malformed_at 13) (cut "\x1fa\x00\x00\xff\xff");
          equal string "too long at 11" (decode_seen frame ~cap:3 (header () ^ le32 100 ^ "\x50ab"));
          equal string "truncated at 17" (cut "\x1fa\x01\x00\xff\xff");
          equal string "truncated at 13" (cut "\x50a");
          equal string "truncated at 14"
            (decode_seen frame (header () ^ le32 (5 lor 0x80000000) ^ "hel")));
      test "only linked blocks reach into earlier blocks" (fun () ->
          let b = sequence "" ~offset:8 ~length:8 ^ last "ijklm" in
          let f flg = header ~flg () ^ stored "abcdefgh" ^ compressed b ^ end_mark in
          equal Lz_support.decoded (Ok "abcdefghabcdefghijklm")
            (decode frame (f version));
          equal string
            (malformed_at (7 + 12 + 4 + 1))
            (decode_seen frame (f (version lor independent))));
      test "a frame's matches never reach into the frame before" (fun () ->
          let first = header () ^ stored "abcdefgh" ^ end_mark in
          let b = sequence "" ~offset:8 ~length:8 ^ last "ijklm" in
          equal string
            (malformed_at (String.length first + 7 + 4 + 1))
            (decode_seen frame (first ^ header () ^ compressed b ^ end_mark)));
      test "a stored block past the destination is too long at its size" (fun () ->
          equal string "too long at 7"
            (decode_seen frame ~cap:3 (header () ^ stored "hello" ^ end_mark)));
      test "a zip bomb ends at the destination" (fun () ->
          (* Blocks of 4 MiB from 16 KiB each: 10 MiB of input state
             2.5 GiB. *)
          let b = compressed (run_block ((4 * mib) - 6)) in
          let f =
            header ~bd:max4m ()
            ^ String.concat "" (List.init (10 * mib / String.length b) (fun _ -> b))
            ^ end_mark
          in
          let g = guarded mib in
          equal string "too long at 13"
            (seen (Lz.Lz4.Frame.decode (Lz_support.of_string f) (snd g)));
          untouched g);
    ]

(* Allocation *)

let allocation =
  let words f =
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    int_of_float (Gc.minor_words () -. before)
  in
  cases
    ~name:(fun c -> c.name)
    "decode allocates only its result" [ block; frame ]
    (fun c ->
      let s = Lz_support.corpus Text (256 * kib) in
      let src = Lz_support.of_string (encode c s) in
      let dst = Lz_support.create (String.length s) in
      let baseline = words (fun () -> ()) in
      (* Ok { consumed; produced }: 2 words for Ok, 3 for the record. *)
      at_most int ~than:5 (words (fun () -> c.decode src dst) - baseline))

let () =
  exit
    (run "Lz4"
       [
         group "arguments"
           [
             test "a negative size has no bound" (fun () ->
                 raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
                     Lz.Lz4.Block.bound (-1));
                 raises_match (Exn.invalid_arg ~substring:"negative") (fun () ->
                     Lz.Lz4.Frame.bound (-1)));
             group "block" [ Lz_support.largest_bound Lz.Lz4.Block.bound ];
             group "frame" [ Lz_support.largest_bound Lz.Lz4.Frame.bound ];
           ];
         laws block;
         laws frame;
         group "frames" [ frames ];
         group "drawn blocks" [ drawn; bad_offsets ];
         formats;
         choices;
         vendor;
         hostile_blocks;
         hostile_frames;
         allocation;
       ])
