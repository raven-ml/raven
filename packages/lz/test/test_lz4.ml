(* LZ4 blocks and frames. The hand-built inputs follow lz4's "LZ4 Block Format
   Description" and "LZ4 Frame Format Description" 1.6.4. An error's expected
   offset is the first byte of what is wrong, as lz.mli states it: of a
   sequence, the token for its literals and the offset field for its match; of a
   frame, the first byte of the field. Errors are compared on their offset and
   kind: lz's messages are a baseline, which each module's pp_error test
   prints. *)

open Windtrap

let kib = 1024
let mib = 1024 * kib

module type Codec = module type of Lz.Lz4.Block

let block = (module Lz.Lz4.Block : Codec)
let frame = (module Lz.Lz4.Frame : Codec)

(* Buffers *)

let guard = 64

(* A destination of [n] bytes at an odd offset, followed by [guard] bytes that a
   codec must not write. *)
let guarded n =
  let whole = Lz_support.create (3 + n + guard) in
  (whole, Bigarray.Array1.sub whole 3 n)

let untouched (whole, view) =
  let n = Bigarray.Array1.dim view in
  equal string ~msg:"the bytes past the destination" (String.make guard '\xa5')
    (String.init guard (fun i ->
         Char.chr (Bigarray.Array1.get whole (3 + n + i))))

(* Calls *)

let encode (module C : Codec) s =
  let dst = Lz_support.create (C.bound (String.length s)) in
  let k =
    require_some ~msg:"fits its bound" (C.encode (Lz_support.of_string s) dst)
  in
  Lz_support.to_string ~length:k dst

(* [decode c ~cap s] is [s] decoded into a destination of [cap] bytes (default
   four times [s]'s length and 64). A decode that succeeds has read all of [s],
   and none writes past the destination. *)
let decode (module C : Codec) ?cap s =
  let cap = Option.value cap ~default:((4 * String.length s) + 64) in
  let g = guarded cap in
  let r = C.decode (Lz_support.of_string s) (snd g) in
  untouched g;
  match r with
  | Error e -> Error e
  | Ok { Lz.consumed; produced } ->
      equal int ~msg:"consumed" (String.length s) consumed;
      Ok (Lz_support.to_string ~length:produced (snd g))

(* A decode's output, or its error compared on its offset and kind. *)
let outcome =
  Testable.make ~pp:(Testable.pp Lz_support.decoded) ~equal:(fun a b ->
      match (a, b) with
      | Ok a, Ok b -> String.equal a b
      | Error a, Error b -> Testable.equal Lz_support.refusal a b
      | Ok _, Error _ | Error _, Ok _ -> false)

(* Expected errors, whose message is not compared. *)
let truncated at = Error { Lz.at; why = Truncated }
let malformed at = Error { Lz.at; why = Malformed "" }
let too_long at = Error { Lz.at; why = Too_long }
let unsupported at = Error { Lz.at; why = Unsupported "" }

(* Drawn inputs, each at an offset of 0 to 7 bytes into its buffer. *)
let placed = Gen.pair Lz_support.recipe (Gen.int_range 0 7)
let large = List.map (fun r -> (r, 1)) Lz_support.large

(* One input of each kind [cover_recipe] covers, so that each holds on every
   run. *)
let kinds = List.map (fun r -> (r, 0)) Lz_support.every_kind

let cover_recipe r s =
  let open Lz_support in
  cover "empty" (s = "");
  cover "one byte" (String.length s = 1);
  cover "a match" (match r with Match _ -> true | _ -> false);
  cover "noise" (match r with Noise _ -> true | _ -> false);
  cover "a run" (match r with Run _ -> true | _ -> false);
  cover "a concatenation" (match r with Concat _ -> true | _ -> false)

(* Frames and blocks by hand *)

let le16 n = String.init 2 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
let le32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff))
let le64 n = le32 (n land 0xffffffff) ^ le32 (n lsr 32)
let byte n = String.make 1 (Char.chr n)
let xxh32 s = Int32.to_int (Lz.Xxh32.string s) land 0xffffffff

(* FLG: the version 01 in bits 7-6, then B.Indep, B.Checksum, C.Size,
   C.Checksum, a reserved bit and DictID. BD: the block maximum size in bits
   6-4, 4 for 64 KiB to 7 for 4 MiB. *)
let version = 0x40
let independent = 0x20
let block_checksum = 0x10
let content_size = 0x08
let content_checksum = 0x04
let dict_id = 0x01
let max64k = 0x40
let max4m = 0x70

(* A frame header: magic, FLG, BD, the optional fields, and HC, the second byte
   of the descriptor's XXH32. Without options it is 7 bytes. *)
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

(* A length past a token's 15: 255s, then the rest. *)
let length_bytes n =
  String.make ((n - 15) / 255) '\xff' ^ byte ((n - 15) mod 255)

let rest n = if n < 15 then "" else length_bytes n

(* A sequence: a token holding the count of literals and the match's length less
   4, each up to 15 with the rest in bytes after it; the literals; the match's
   offset; the rest of its length. *)
let sequence lits ~offset ~length =
  let l = String.length lits and m = length - 4 in
  byte ((min l 15 lsl 4) lor min m 15) ^ rest l ^ lits ^ le16 offset ^ rest m

(* A block's last sequence, which holds literals only. *)
let last lits =
  let l = String.length lits in
  byte (min l 15 lsl 4) ^ rest l ^ lits

(* "a", a match of [length] bytes from 1 back, and 5 literals: the match's last
   byte is the seventh from the end. *)
let run_block length = sequence "a" ~offset:1 ~length ^ last "bcdef"

(* Blocks drawn as sequences *)

(* The end-of-block rules: at least 5 last literals, and the last match starting
   at least 12 bytes before the block's end. *)
let last_literals = 5
let match_margin = 12

type sequence = { lits : string; offset : int; length : int }

let bytes_of s = sequence s.lits ~offset:s.offset ~length:s.length
let block_of seqs final = String.concat "" (List.map bytes_of seqs) ^ last final

(* What the sequences and the last literals decode to. A match reads byte by
   byte, so one longer than its offset repeats what it has just written. *)
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

let produced seqs =
  List.fold_left (fun n s -> n + String.length s.lits + s.length) 0 seqs

let pp_sequences ppf (seqs, final) =
  List.iter
    (fun s ->
      Format.fprintf ppf "(%d literals, offset %d, length %d) "
        (String.length s.lits) s.offset s.length)
    seqs;
  Format.fprintf ppf "%d last literals" (String.length final)

(* Up to 30 sequences of every literal and match length class, half of them a
   match of 9 to 40 bytes from under 16 back, which the fast loop copies in
   overlapping stores; then last literals that keep the end-of-block rules. *)
let sequences =
  let open Gen in
  (* Literals from a drawn seed: their bytes change no law, and drawing each
     byte as an element costs more than the decode. *)
  let lits n =
    map (fun seed -> Lz_support.(build (Noise (seed, n)))) small_int
  in
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
  with_pp pp_sequences
    (let* k = int_range 1 30 in
     let* seqs = go k 0 [] in
     let last_match = (List.nth seqs (k - 1)).length in
     let* n = int_range (max last_literals (match_margin - last_match)) 40 in
     let+ final = lits n in
     (seqs, final))

(* The loop that decodes each sequence into a destination with [slack] bytes to
   spare: the fast loop while 32 bytes of the block and of the destination
   remain after the sequence; the exact loop from the first sequence whose token
   is under 32 bytes from the block's end or whose output starts under 32 bytes
   from the destination's. *)
let loops ~slack seqs final =
  let input = String.length (block_of seqs final) in
  let room = String.length (meaning seqs final) + slack in
  let rec go pos out acc = function
    | [] -> List.rev acc
    | s :: rest ->
        let next = pos + String.length (bytes_of s) in
        let made = out + String.length s.lits + s.length in
        let fast = input - next >= 32 && room - made >= 32 in
        let exact = input - pos < 32 || room - out < 32 in
        go next made ((s, fast, exact) :: acc) rest
  in
  go 0 0 [] seqs

(* A match from each distance under 8, past its distance, in the exact loop: the
   first sequence, 11 bytes, leaves 27 of the block. *)
let exact_periods =
  ( ( { lits = "abcdefgh"; offset = 8; length = 9 }
      :: List.init 7 (fun i -> { lits = ""; offset = i + 1; length = 9 }),
      "vwxyz" ),
    0 )

let drawn =
  prop "a block decodes to what its sequences mean" ~examples:[ exact_periods ]
    (Gen.pair sequences (Gen.of_list ~pp:Format.pp_print_int [ 0; 64 ]))
    (fun ((seqs, final), slack) ->
      let b = block_of seqs final and raw = meaning seqs final in
      let marked = loops ~slack seqs final in
      let fast p = List.exists (fun (s, fast, _) -> fast && p s) marked in
      let exact p = List.exists (fun (s, _, exact) -> exact && p s) marked in
      for d = 1 to 7 do
        cover
          (Printf.sprintf "offset %d over 8 bytes in the fast loop" d)
          (fast (fun s -> s.offset = d && s.length > 8));
        cover
          (Printf.sprintf "offset %d over 8 bytes in the exact loop" d)
          (exact (fun s -> s.offset = d && s.length > 8))
      done;
      cover "offset 8 to 15 over 8 bytes in the fast loop"
        (fast (fun s -> s.offset >= 8 && s.offset < 16 && s.length > 8));
      cover "a long literal run in the fast loop"
        (fast (fun s -> String.length s.lits >= 15));
      cover "a long match in the fast loop" (fast (fun s -> s.length >= 19));
      cover "a destination of the output's length" (slack = 0);
      let offset = String.length raw mod 8 in
      let dst = Lz_support.room ~offset (String.length raw + slack) in
      let r = Lz.Lz4.Block.decode (Lz_support.placed ~offset b) dst in
      equal
        (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed = String.length b; produced = String.length raw })
        r;
      equal string raw (Lz_support.to_string ~length:(String.length raw) dst))

(* Expected values from the format documents *)

(* A block has no end mark: bytes after it are read as more of the block. After
   the last literals of "hello"'s block, one byte is a cut offset field; an
   offset of 1, the token's match of 4 and a sequence of 8 literals decode to
   more, 12 bytes from the match to the end. *)
let after_block =
  test "bytes after a block are read as more of it" (fun () ->
      let b = encode block "hello" in
      equal outcome
        (truncated (String.length b + 1))
        (decode block (b ^ "\x00"));
      equal outcome
        (Ok ("hello" ^ "oooo" ^ "abcdefgh"))
        (decode block (b ^ "\x01\x00\x80abcdefgh")))

let block_sequences =
  group "sequences"
    [
      after_block;
      test "a zero token is the empty block" (fun () ->
          equal outcome (Ok "") (decode block "\x00"));
      test "15 literals take a zero byte after the token" (fun () ->
          let s = "abcdefghijklmno" in
          equal outcome (Ok s) (decode block ("\xf0\x00" ^ s)));
      test "280 literals take 15, then 255 and 10" (fun () ->
          let s = String.init 280 (fun i -> Char.chr (i land 0xff)) in
          equal outcome (Ok s) (decode block ("\xf0\xff\x0a" ^ s)));
      test "an offset of 1 repeats the last byte" (fun () ->
          equal outcome
            (Ok ("a" ^ String.make 8 'a' ^ "bcdef"))
            (decode block (sequence "a" ~offset:1 ~length:8 ^ last "bcdef")));
      test "an offset reaches 65535 bytes back" (fun () ->
          let lits = String.init 65535 (fun i -> Char.chr (i * 7 land 0xff)) in
          equal outcome
            (Ok (lits ^ String.sub lits 0 8 ^ "vwxyz"))
            (decode block
               (sequence lits ~offset:65535 ~length:8 ^ last "vwxyz")));
      test "a match of 4 MiB decodes" (fun () ->
          equal outcome
            (Ok ("a" ^ String.make (4 * mib) 'a' ^ "bcdef"))
            (decode block ~cap:((4 * mib) + 6) (run_block (4 * mib))));
      (* 1 literal, a match of 7, 5 literals: the match starts 12 bytes before
         the end. *)
      test "a last match 12 bytes before the end decodes" (fun () ->
          equal outcome
            (Ok ("a" ^ String.make 7 'a' ^ "bcdef"))
            (decode block (sequence "a" ~offset:1 ~length:7 ^ last "bcdef")));
      drawn;
    ]

(* Refusals *)

(* [reference ~cap b] is the block [b] decoded byte by byte into [cap] bytes as
   lz.mli's Lz4 states it: the output, or the rule [b] breaks first and its
   error. A length only grows with its bytes, so it asks for output past the
   destination once its bytes so far do, whatever follows. Last literals that
   break the end-of-block rules are Malformed whatever the room: a decode is
   Too_long only when a larger destination would let it go on. *)
let reference ~cap b =
  let input = String.length b in
  let out = Buffer.create 256 in
  let exception Stop of string * Lz.error in
  let stop rule at why = raise (Stop (rule, { Lz.at; why })) in
  let truncated () =
    stop "an input that ends inside a sequence or after a match" input Truncated
  in
  let get i = if i < input then Char.code b.[i] else truncated () in
  (* The bytes of a length after a token's 15, from [n]: 255s, then the last. *)
  let rec length n i ~room ~rule ~at =
    let v = get i in
    let n = n + v in
    if n > room then stop rule at Too_long;
    if v = 255 then length n (i + 1) ~room ~rule ~at else (n, i + 1)
  in
  let rec sequence i last_match =
    let token = get i in
    let room = cap - Buffer.length out in
    let past = "literals past the destination" in
    let lits, j =
      if token lsr 4 < 15 then (token lsr 4, i + 1)
      else length 15 (i + 1) ~room ~rule:past ~at:i
    in
    let ends = j + lits = input in
    let breaks = lits < last_literals || lits + last_match < match_margin in
    if ends && last_match > 0 && breaks then
      stop
        (if lits > room then
           "last literals past the destination that break the end-of-block \
            rules"
         else "last literals that break the end-of-block rules")
        i (Malformed "");
    if lits > room then stop past i Too_long;
    if j + lits > input then truncated ();
    Buffer.add_substring out b j lits;
    if ends then Buffer.contents out
    else
      let at = j + lits in
      let offset = get at lor (get (at + 1) lsl 8) in
      if offset = 0 || offset > Buffer.length out then
        stop "a match offset of 0 or before the block's start" at (Malformed "");
      let room = cap - Buffer.length out in
      let past = "a match past the destination" in
      let length, k =
        if token land 15 < 15 then ((token land 15) + 4, at + 2)
        else length 19 (at + 2) ~room ~rule:past ~at
      in
      if length > room then stop past at Too_long;
      for _ = 1 to length do
        Buffer.add_char out (Buffer.nth out (Buffer.length out - offset))
      done;
      sequence k length
  in
  match sequence 0 0 with
  | s -> Ok s
  | exception Stop (rule, e) -> Error (rule, e)

type change = Same | Set of int * int | Cut of int

let pp_change ppf = function
  | Same -> Format.pp_print_string ppf "unchanged"
  | Set (p, v) -> Format.fprintf ppf "byte %d set to %d" p v
  | Cut p -> Format.fprintf ppf "cut to %d bytes" p

let apply change s =
  match change with
  | Same -> s
  | Set (p, v) -> String.mapi (fun i c -> if i = p then Char.chr v else c) s
  | Cut p -> String.sub s 0 p

(* The positions of a block's tokens, lengths and offsets, and of the first byte
   after each sequence's literals. *)
let fields seqs final =
  let count n = String.length (rest n) in
  let span pos n = List.init n (fun k -> pos + k) in
  let rec go pos acc ends = function
    | [] ->
        let tail = span pos (1 + count (String.length final)) in
        (List.concat (List.rev (tail :: acc)), ends)
    | s :: seqs ->
        let l = String.length s.lits and m = 2 + count (s.length - 4) in
        let after = pos + 1 + count l + l in
        let here = span pos (1 + count l) @ span after m in
        go (after + m) (here :: acc) (after :: ends) seqs
  in
  go 0 [] [] seqs

(* Drawn blocks, then the same with a token, length or offset byte changed, any
   byte changed, or cut, anywhere or after a sequence's literals; into a
   destination of the output's length, short of it, or with room. *)
let changed =
  let open Gen in
  let* seqs, final = sequences in
  let b = block_of seqs final in
  let n = String.length b and out = String.length (meaning seqs final) in
  let headers, ends = fields seqs final in
  let* change =
    frequency
      [
        (1, constant Same);
        ( 4,
          let+ p = of_list headers and+ v = int_range 0 255 in
          Set (p, v) );
        ( 1,
          let+ p = int_range 0 (n - 1) and+ v = int_range 0 255 in
          Set (p, v) );
        (1, map (fun p -> Cut p) (int_range 0 n));
        (1, map (fun p -> Cut p) (of_list ends));
      ]
  in
  let+ cap =
    frequency
      [
        (3, constant out);
        (1, map (fun d -> max 0 (out - d)) (int_range 1 8));
        (1, int_range 0 out);
        (1, constant (out + 64));
      ]
  in
  (seqs, final, change, cap)

let pp_changed ppf (seqs, final, change, cap) =
  Format.fprintf ppf "%a, %a, into %d bytes" pp_sequences (seqs, final)
    pp_change change cap

(* One block for each rule a drawn block reaches less often than every run: "a",
   a match of 6 from 1 back, then "bc", a match of 4 and "defghijk". The second
   token is at 4, its offset at 7; the output passes 8 bytes in "bc" and 10 in
   its match. *)
let breaking =
  let seqs =
    [
      { lits = "a"; offset = 1; length = 6 };
      { lits = "bc"; offset = 1; length = 4 };
    ]
  in
  List.map
    (fun (change, cap) -> (seqs, "defghijk", change, cap))
    [
      (Same, 8);
      (Same, 10);
      (Set (7, 0), 64);
      (Set (7, 20), 64);
      (Cut 7, 64);
      (Cut 7, 8);
      (Cut 6, 64);
      (Set (5, 0x78), 64);
    ]

let law_reference =
  let rules =
    [
      "an input that ends inside a sequence or after a match";
      "a match offset of 0 or before the block's start";
      "last literals that break the end-of-block rules";
      "last literals past the destination that break the end-of-block rules";
      "literals past the destination";
      "a match past the destination";
    ]
  in
  prop "any bytes decode as the reference decodes them" ~examples:breaking
    (Gen.with_pp pp_changed changed) (fun (seqs, final, change, cap) ->
      let b = apply change (block_of seqs final) in
      let want = reference ~cap b in
      List.iter
        (fun rule ->
          cover rule
            (match want with Error (r, _) -> r = rule | Ok _ -> false))
        rules;
      cover "a block decoded after a change"
        (change <> Same && Result.is_ok want);
      equal outcome (Result.map_error snd want) (decode block ~cap b))

(* Valid sequences, then one whose offset is 0 or before the output, then up to
   63 bytes: Malformed at the offset field, in either loop, whatever follows
   it. *)
let bad_offsets =
  let gen =
    Gen.(
      pair sequences
        (triple bool (string_of ~size:(int_range 0 14) char) (int_range 0 63)))
  in
  (* A block whose first match has an offset of 0, or of 2 after one literal. *)
  let first zero = (([], ""), (zero, "a", 7)) in
  prop "a bad offset is Malformed at its offset field"
    ~examples:[ first true; first false ]
    gen
    (fun ((seqs, _), (zero, lits, filler)) ->
      let prefix = String.concat "" (List.map bytes_of seqs) in
      let reach = produced seqs + String.length lits in
      let offset = if zero || reach >= 65535 then 0 else reach + 1 in
      let bad = sequence lits ~offset ~length:8 in
      let b = prefix ^ bad ^ String.make filler 'x' in
      let tail = String.length b - String.length prefix in
      cover "a zero offset" (offset = 0);
      cover "an offset before the output" (offset > 0);
      cover "in the fast loop" (tail >= 32);
      cover "in the exact loop" (tail < 32);
      let cap = reach + 64 + filler in
      equal outcome
        (malformed (String.length prefix + 1 + String.length lits))
        (decode block ~cap b))

let block_refusals =
  group "refusals"
    [
      test "an empty block is Truncated" (fun () ->
          equal outcome (truncated 0) (decode block ""));
      test "a block ending after a match is Truncated" (fun () ->
          let b = sequence "abcd" ~offset:4 ~length:8 in
          equal outcome (truncated (String.length b)) (decode block b));
      test "a block cut inside its last literals is Truncated" (fun () ->
          let b = sequence "abcd" ~offset:4 ~length:8 ^ "\x50" in
          equal outcome (truncated (String.length b)) (decode block b));
      test "a block cut inside a match's length is Truncated given room"
        (fun () ->
          equal outcome (truncated 5)
            (decode block ~cap:kib "\x1fa\x01\x00\xff"));
      test "a bad offset is Malformed before its match length ends" (fun () ->
          (* The match length's 255s go on past the input's end, but no byte
             after them makes the offset valid. *)
          equal outcome (malformed 2) (decode block "\x1fa\x00\x00\xff\xff");
          equal outcome (malformed 2) (decode block "\x1fa\x05\x00\xff\xff"));
      bad_offsets;
      (* 1 literal, a match of 6, 5 literals: the match starts 11 bytes before
         the end, and the last sequence, at 4, breaks the rule. The output
         passes 8 bytes in the last literals. *)
      cases
        "a last match 11 bytes before the end is Malformed at the last token"
        ~id:fst
        [ ("given room", 64); ("past the destination", 8) ]
        (fun (_, cap) ->
          equal outcome (malformed 4)
            (decode block ~cap
               (sequence "a" ~offset:1 ~length:6 ^ last "bcdef")));
      (* The output passes 20 bytes in the last literals. *)
      cases "last literals fewer than 5 are Malformed at their token" ~id:fst
        [ ("given room", 64); ("past the destination", 20) ]
        (fun (_, cap) ->
          equal outcome (malformed 4)
            (decode block ~cap
               (sequence "a" ~offset:1 ~length:18 ^ last "bcde")));
      test "literals past the destination are Too_long at their token"
        (fun () ->
          equal outcome (too_long 0) (decode block ~cap:3 (last "abcdefg"));
          equal outcome (too_long 4)
            (decode block ~cap:10
               (sequence "a" ~offset:1 ~length:5 ^ last "bcdefgh")));
      test "literals past the destination are Too_long before the input ends"
        (fun () ->
          (* 100 literals stated, 50 present, room for 10. *)
          equal outcome (too_long 0)
            (decode block ~cap:10 ("\xf0\x55" ^ String.make 50 'x')));
      test "a match past the destination is Too_long at its offset" (fun () ->
          equal outcome (too_long 2)
            (decode block ~cap:8
               (sequence "a" ~offset:1 ~length:8 ^ last "bcdef")));
      law_reference;
    ]

(* [messages c rows] prints the error of each row [(name, cap, input)] decoded
   by [c]. *)
let messages c rows =
  List.iter
    (fun (name, cap, s) ->
      match decode c ~cap s with
      | Ok _ -> Format.printf "%s: decodes@." name
      | Error e -> Format.printf "%s: %a@." name Lz.pp_error e)
    rows

let block_messages =
  group "pp_error"
    [
      test "prints each refusal's kind, offset and message" (fun () ->
          let b = sequence "a" ~offset:1 ~length:6 in
          messages block
            [
              ("empty", 64, "");
              ("a zero offset", 64, "\x1fa\x00\x00");
              ("an offset before the start", 64, "\x1fa\x02\x00");
              ("a last match 11 bytes before the end", 64, b ^ last "bcdef");
              ("literals past the destination", 3, last "abcdefg");
              ("a match past the destination", 4, b ^ last "bcdef");
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            empty: truncated at byte 0
            a zero offset: malformed at byte 2: invalid zero offset
            an offset before the start: malformed at byte 2: invalid offset too far back
            a last match 11 bytes before the end: malformed at byte 4: invalid end of block
            literals past the destination: output too long at byte 0
            a match past the destination: output too long at byte 2
          |});
    ]

(* Frames by hand *)

let frames =
  prop "decode (encode a ^ skippable ^ encode b) = a ^ b"
    ~examples:[ (Lz_support.Empty, Lz_support.Byte 'a') ]
    (Gen.pair Lz_support.recipe Lz_support.recipe)
    (fun (a, b) ->
      let a = Lz_support.build a and b = Lz_support.build b in
      cover "an empty first frame" (a = "");
      let f = encode frame a ^ skippable "between" ^ encode frame b in
      let n = String.length a + String.length b in
      equal outcome (Ok (a ^ b)) (decode frame ~cap:n f))

(* The encoder's frame, as Lz.Lz4.Frame states it. *)
let encoder_frame =
  let flg f = Char.code f.[4] in
  cases
    "the encoder writes independent blocks without block checksums, with the \
     content size and the content checksum"
    ~id:string_of_int
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
      equal string ~msg:"content checksum"
        (le32 (xxh32 s))
        (String.sub f (len - 4) 4))

let frame_frames =
  group "frames"
    [
      test "a frame is magic, descriptor, blocks and end mark" (fun () ->
          equal outcome (Ok "hello")
            (decode frame (header () ^ stored "hello" ^ end_mark)));
      cases "a block of the declared maximum size decodes" ~id:fst
        [
          ( "64 KiB, stored",
            fun () ->
              let raw = String.make (64 * kib) 'x' in
              (header (), stored raw, raw) );
          ( "4 MiB, compressed",
            fun () ->
              let n = (4 * mib) - 6 in
              ( header ~bd:max4m (),
                compressed (run_block n),
                "a" ^ String.make n 'a' ^ "bcdef" ) );
        ]
        (fun (_, frame_of) ->
          let h, b, raw = frame_of () in
          equal outcome (Ok raw)
            (decode frame ~cap:(String.length raw) (h ^ b ^ end_mark)));
      test "a stored block of 0 bytes does not end the frame" (fun () ->
          equal outcome (Ok "ab")
            (decode frame
               (header () ^ stored "a" ^ stored "" ^ stored "b" ^ end_mark)));
      test "the content checksum is the content's XXH32" (fun () ->
          let f =
            header ~flg:(version lor content_checksum) ()
            ^ stored "hello" ^ end_mark
            ^ le32 (xxh32 "hello")
          in
          equal outcome (Ok "hello") (decode frame f));
      test "a block checksum is the XXH32 of the block's bytes" (fun () ->
          let b = last "hello" in
          let f =
            header ~flg:(version lor block_checksum) ()
            ^ compressed b
            ^ le32 (xxh32 b)
            ^ end_mark
          in
          equal outcome (Ok "hello") (decode frame f));
      test "a frame of no blocks is empty" (fun () ->
          equal outcome (Ok "") (decode frame (header () ^ end_mark)));
      test "skippable frames produce nothing" (fun () ->
          let f =
            skippable "xyz" ^ header () ^ stored "ab" ^ end_mark
            ^ skippable ~nibble:15 ""
          in
          equal outcome (Ok "ab") (decode frame f);
          equal outcome (Ok "") (decode frame (skippable "")));
      test "frames are read to the input's end" (fun () ->
          let f = header () ^ stored "ab" ^ end_mark in
          equal outcome (Ok "abab") (decode frame (f ^ f)));
      frames;
      encoder_frame;
    ]

let frame_refusals =
  group "refusals"
    [
      test "an empty input is Truncated" (fun () ->
          equal outcome (truncated 0) (decode frame ""));
      cases "a byte that begins no frame is Malformed at its offset" ~id:fst
        [
          ("a frame's magic, its first byte changed", ("\x04\x22\x4d\x19abc", 0));
          ("a frame's magic, its last byte changed", ("\x00\x22\x4d\x18abc", 0));
          ("two bytes of no magic", ("\x5f\x2b", 0));
          ("a byte after a frame", (header () ^ end_mark ^ "\x00", 11));
          ( "a changed magic after a skippable frame",
            (skippable "" ^ header () ^ end_mark ^ "\x04\x22\x4d\x19", 19) );
        ]
        (fun (_, (f, at)) -> equal outcome (malformed at) (decode frame f));
      test "a legacy frame is Unsupported at its magic number" (fun () ->
          equal outcome (unsupported 0)
            (decode frame (le32 0x184C2102 ^ le32 0));
          equal outcome (unsupported 11)
            (decode frame (header () ^ end_mark ^ le32 0x184C2102 ^ le32 0)));
      test "a cut magic number is Truncated" (fun () ->
          equal outcome (truncated 2) (decode frame "\x04\x22");
          equal outcome (truncated 1) (decode frame "\x5a");
          equal outcome (truncated 13)
            (decode frame (header () ^ end_mark ^ "\x04\x22")));
      cases "a cut skippable frame is Truncated" ~id:fst
        [ ("inside its size", 6); ("inside its bytes", 9) ]
        (fun (_, n) ->
          equal outcome (truncated n)
            (decode frame (String.sub (skippable "xyz") 0 n)));
      cases "a version other than 01 is Malformed at FLG" ~id:fst
        [ ("00", 0x00); ("10", 0x80); ("11", 0xc0) ]
        (fun (_, flg) ->
          equal outcome (malformed 4) (decode frame (header ~flg () ^ end_mark)));
      cases "a set reserved bit is Malformed at its byte"
        ~id:(fun (n, _, _, _) -> n)
        [
          ("FLG bit 1", version lor 0x02, max64k, 4);
          ("BD bit 7", version, max64k lor 0x80, 5);
          ("BD bit 3", version, max64k lor 0x08, 5);
          ("BD bit 2", version, max64k lor 0x04, 5);
          ("BD bit 1", version, max64k lor 0x02, 5);
          ("BD bit 0", version, max64k lor 0x01, 5);
        ]
        (fun (_, flg, bd, at) ->
          equal outcome (malformed at)
            (decode frame (header ~flg ~bd () ^ end_mark)));
      cases "a block maximum size below 4 is Malformed at BD" ~id:string_of_int
        [ 0; 1; 2; 3 ] (fun id ->
          equal outcome (malformed 5)
            (decode frame (header ~bd:(id lsl 4) () ^ end_mark)));
      test "a wrong header checksum is Malformed at HC" (fun () ->
          let h = header () in
          let h = String.sub h 0 6 ^ byte (Char.code h.[6] lxor 1) in
          equal outcome (malformed 6) (decode frame (h ^ end_mark)));
      test "a dictionary is Unsupported at its identifier" (fun () ->
          equal outcome (unsupported 6)
            (decode frame
               (header ~flg:(version lor dict_id) ~dict:7 () ^ end_mark)));
      test "a block above the maximum size is Malformed at its size" (fun () ->
          equal outcome (malformed 7)
            (decode frame
               (header () ^ stored (String.make ((64 * kib) + 1) 'x') ^ end_mark)));
      test "a block decoding past the maximum size is Malformed at the code"
        (fun () ->
          (* A match, whose offset is at 13, of 64 KiB after a literal; and
             literals at 11 that state 64 KiB + 1. *)
          equal outcome (malformed 13)
            (decode frame ~cap:mib
               (header () ^ compressed (run_block (64 * kib)) ^ end_mark));
          equal outcome (malformed 11)
            (decode frame ~cap:mib
               (header ()
               ^ compressed ("\xf0" ^ length_bytes ((64 * kib) + 1) ^ "xyz")
               ^ end_mark)));
      cases "a content size that differs is Malformed at the size field"
        ~id:(fun (n, _, _) -> n)
        [
          ("above, in a stored block", 6, stored "hello");
          ("below, in a stored block", 4, stored "hello");
          ("below, in literals", 3, compressed (last "hello"));
          ("below, in a match", 12, compressed (run_block 20));
          ( "above, by a block",
            5,
            stored "he" ^ compressed (last "llo") ^ stored "ab" );
        ]
        (fun (_, size, blocks) ->
          let h = header ~flg:(version lor content_size) ~size () in
          equal outcome (malformed 6)
            (decode frame ~cap:64 (h ^ blocks ^ end_mark)));
      test "a wrong block checksum is Malformed at the checksum" (fun () ->
          let b = last "hello" in
          let h = header ~flg:(version lor block_checksum) () in
          equal outcome
            (malformed (String.length h + 4 + String.length b))
            (decode frame (h ^ compressed b ^ le32 (xxh32 b lxor 1) ^ end_mark)));
      test "a wrong content checksum is Malformed at the checksum" (fun () ->
          let f =
            header ~flg:(version lor content_checksum) ()
            ^ stored "hello" ^ end_mark
          in
          equal outcome
            (malformed (String.length f))
            (decode frame (f ^ le32 (xxh32 "hello" lxor 1))));
      test "a block ending inside a sequence is Malformed at its token"
        (fun () ->
          (* The block's token is at 11. *)
          let ends ?cap b =
            decode frame ?cap (header () ^ compressed b ^ end_mark)
          in
          equal outcome (malformed 11) (ends "\x50ab");
          equal outcome (malformed 11) (ends "\x40abcd\x04");
          (* A match of 274 bytes or more, given room for it. *)
          equal outcome (malformed 11) (ends ~cap:kib "\x4fabcd\x04\x00\xff");
          equal outcome (malformed 18)
            (ends (sequence "abcd" ~offset:4 ~length:8 ^ "\x50a")));
      test "a block ending after a match is Malformed at its last token"
        (fun () ->
          let ends ?cap b =
            decode frame ?cap (header () ^ compressed b ^ end_mark)
          in
          equal outcome (malformed 11)
            (ends (sequence "abcd" ~offset:4 ~length:8));
          (* 32 bytes, 1 + 14 + 2 + 15, whose match length runs to the block's
             end: the fast loop decodes it. *)
          let long =
            "\xef" ^ String.make 14 'a' ^ le16 1 ^ length_bytes (15 + (14 * 255))
          in
          equal outcome (malformed 11) (ends ~cap:mib long);
          equal outcome (malformed 18)
            (ends ~cap:mib (sequence "abcd" ~offset:4 ~length:8 ^ long)));
      test "a bad offset is Malformed at the offset in a frame" (fun () ->
          equal outcome (malformed 13)
            (decode frame
               (header () ^ compressed "\x1fa\x00\x00\xff\xff" ^ end_mark)));
      (* A block of 100 bytes cut short: Malformed if a byte before the cut
         proves it, Too_long if a byte before it asks for output past the
         destination, Truncated otherwise. A match of 529 bytes or more is
         Truncated given room for it, Too_long at its offset without. *)
      cases "a cut block is the first error its bytes prove, else Truncated"
        ~id:(fun (name, _, _, _) -> name)
        [
          ( "a bad offset",
            malformed 13,
            4096,
            le32 100 ^ "\x1fa\x00\x00\xff\xff" );
          ("literals past the destination", too_long 11, 3, le32 100 ^ "\x50ab");
          ( "a long match, given room",
            truncated 17,
            kib,
            le32 100 ^ "\x1fa\x01\x00\xff\xff" );
          ( "a long match past the destination",
            too_long 13,
            100,
            le32 100 ^ "\x1fa\x01\x00\xff\xff" );
          ("literals", truncated 13, 4096, le32 100 ^ "\x50a");
          ("a stored block", truncated 14, 4096, le32 (5 lor 0x80000000) ^ "hel");
        ]
        (fun (_, expected, cap, b) ->
          equal outcome expected (decode frame ~cap (header () ^ b)));
      test "only linked blocks reach into earlier blocks" (fun () ->
          let b = sequence "" ~offset:8 ~length:8 ^ last "ijklm" in
          let f flg =
            header ~flg () ^ stored "abcdefgh" ^ compressed b ^ end_mark
          in
          equal outcome (Ok "abcdefghabcdefghijklm") (decode frame (f version));
          equal outcome
            (malformed (7 + 12 + 4 + 1))
            (decode frame (f (version lor independent))));
      test "a frame's matches never reach into the frame before" (fun () ->
          let first = header () ^ stored "abcdefgh" ^ end_mark in
          let b = sequence "" ~offset:8 ~length:8 ^ last "ijklm" in
          equal outcome
            (malformed (String.length first + 7 + 4 + 1))
            (decode frame (first ^ header () ^ compressed b ^ end_mark)));
      test "a stored block past the destination is Too_long at its size"
        (fun () ->
          equal outcome (too_long 7)
            (decode frame ~cap:3 (header () ^ stored "hello" ^ end_mark)));
      test "a stored block past the content size first is Malformed at it"
        (fun () ->
          (* The bytes pass a content size of 3 before 4 bytes of room. *)
          let h = header ~flg:(version lor content_size) ~size:3 () in
          equal outcome (malformed 6)
            (decode frame ~cap:4 (h ^ stored "hello" ^ end_mark)));
      test
        "literals past the destination are Too_long at their token in a frame"
        (fun () ->
          equal outcome (too_long 11)
            (decode frame ~cap:3
               (header () ^ compressed (last "abcdefgh") ^ end_mark)));
      (* 65534 bytes from "a" and a match of 65533, whose last length byte ends
         at 271, then a match of 8 or 5 last literals: each passes the block
         maximum of 64 KiB, and a destination of 65535 bytes. 12 literals, or a
         stored block of 12, pass a content size of 10 and a destination of 8
         bytes. *)
      cases
        "a code or stored block past the block maximum or the content size is \
         Malformed whatever the destination"
        ~id:(fun (name, _, _, _) -> name)
        (let long = sequence "a" ~offset:1 ~length:65533 in
         let sized = header ~flg:(version lor content_size) ~size:10 () in
         [
           ( "a match past the maximum and the destination",
             65535,
             header ()
             ^ compressed (long ^ sequence "" ~offset:1 ~length:8 ^ last "bcdef"),
             malformed 273 );
           ( "a match past the maximum",
             mib,
             header ()
             ^ compressed (long ^ sequence "" ~offset:1 ~length:8 ^ last "bcdef"),
             malformed 273 );
           ( "literals past the maximum and the destination",
             65535,
             header () ^ compressed (long ^ last "bcdef"),
             malformed 272 );
           ( "literals past the maximum",
             mib,
             header () ^ compressed (long ^ last "bcdef"),
             malformed 272 );
           ( "literals past the content size and the destination",
             8,
             sized ^ compressed (last "abcdefghijkl"),
             malformed 6 );
           ( "literals past the content size",
             64,
             sized ^ compressed (last "abcdefghijkl"),
             malformed 6 );
           ( "a stored block past the content size and the destination",
             8,
             sized ^ stored "abcdefghijkl",
             malformed 6 );
         ])
        (fun (_, cap, f, expected) ->
          equal outcome expected (decode frame ~cap (f ^ end_mark)));
      (* A zero offset at 13, under a checksum of 0. *)
      cases "a malformed block is refused before a wrong checksum" ~id:fst
        (let b = sequence "a" ~offset:0 ~length:8 ^ last "bcdefgh" in
         [
           ( "the block's",
             header ~flg:(version lor block_checksum) ()
             ^ compressed b ^ le32 0 ^ end_mark );
           ( "the content's",
             header ~flg:(version lor content_checksum) ()
             ^ compressed b ^ end_mark ^ le32 0 );
         ])
        (fun (_, f) -> equal outcome (malformed 13) (decode frame f));
      test "a match past the destination is Too_long at its offset in a frame"
        (fun () ->
          (* A block of 4 MiB from 16 KiB, into 1 MiB. *)
          equal outcome (too_long 13)
            (decode frame ~cap:mib
               (header ~bd:max4m ()
               ^ compressed (run_block ((4 * mib) - 6))
               ^ end_mark)));
    ]

let frame_messages =
  group "pp_error"
    [
      test "prints each refusal's kind, offset and message" (fun () ->
          let h = header () in
          let sized = header ~flg:(version lor content_size) ~size:3 () in
          let summed flg = header ~flg:(version lor flg) () in
          let wrong_hc = String.sub h 0 6 ^ byte (Char.code h.[6] lxor 1) in
          messages frame
            [
              ("empty", 64, "");
              ("no magic", 64, "\x00\x22\x4d\x18abc");
              ("a legacy frame", 64, le32 0x184C2102 ^ le32 0);
              ("a version of 00", 64, header ~flg:0 () ^ end_mark);
              ( "a reserved bit",
                64,
                header ~flg:(version lor 0x02) () ^ end_mark );
              ("a block maximum size of 0", 64, header ~bd:0 () ^ end_mark);
              ("a wrong header checksum", 64, wrong_hc ^ end_mark);
              ( "a dictionary",
                64,
                header ~flg:(version lor dict_id) ~dict:7 () ^ end_mark );
              ( "a block above the maximum size",
                64,
                h ^ stored (String.make ((64 * kib) + 1) 'x') ^ end_mark );
              ( "a block decoding past the maximum size",
                mib,
                h ^ compressed (run_block (64 * kib)) ^ end_mark );
              ( "a content size that differs",
                64,
                sized ^ stored "hello" ^ end_mark );
              ( "a wrong block checksum",
                64,
                summed block_checksum
                ^ compressed (last "hello")
                ^ le32 0 ^ end_mark );
              ( "a wrong content checksum",
                64,
                summed content_checksum ^ stored "hello" ^ end_mark ^ le32 0 );
              ( "a block ending inside a sequence",
                64,
                h ^ compressed "\x50ab" ^ end_mark );
              ( "a block ending after a match",
                64,
                h ^ compressed (sequence "abcd" ~offset:4 ~length:8) ^ end_mark
              );
              ( "a zero offset",
                64,
                h ^ compressed "\x1fa\x00\x00\xff\xff" ^ end_mark );
              ( "a stored block past the destination",
                3,
                h ^ stored "hello" ^ end_mark );
            ];
          expect (output ())
          @@ __POS_OF__
               {|
            empty: truncated at byte 0
            no magic: malformed at byte 0: unknown frame magic
            a legacy frame: unsupported at byte 0: legacy frame
            a version of 00: malformed at byte 4: invalid version
            a reserved bit: malformed at byte 4: reserved bit set
            a block maximum size of 0: malformed at byte 5: invalid block maximum size
            a wrong header checksum: malformed at byte 6: incorrect header checksum
            a dictionary: unsupported at byte 6: dictionary
            a block above the maximum size: malformed at byte 7: invalid block size
            a block decoding past the maximum size: malformed at byte 13: block too large
            a content size that differs: malformed at byte 6: incorrect content size
            a wrong block checksum: malformed at byte 17: incorrect block checksum
            a wrong content checksum: malformed at byte 20: incorrect content checksum
            a block ending inside a sequence: malformed at byte 11: block ends inside a sequence
            a block ending after a match: malformed at byte 11: invalid end of block
            a zero offset: malformed at byte 13: invalid zero offset
            a stored block past the destination: output too long at byte 7
          |});
    ]

(* Decode, bound, encode and encode_string, as each module states them *)

let every_golden (module C : Codec) codec =
  cases "every golden decodes to its input, all of it consumed"
    ~id:(fun g -> g.Lz_support.file)
    (Lz_support.goldens_of codec)
    (fun g ->
      let raw = Lazy.force g.raw and data = Lazy.force g.data in
      let dst = Lz_support.create (String.length raw) in
      equal
        (result Lz_support.progress Lz_support.error)
        (Ok { Lz.consumed = String.length data; produced = String.length raw })
        (C.decode (Lz_support.of_string data) dst);
      equal Lz_support.decoded (Ok raw) (Ok (Lz_support.to_string dst)))

(* The words [f ()] allocates on the OCaml heap, less what measuring an empty
   call allocates, and the words its result holds. *)
let allocated f =
  let measure g =
    let before = Gc.allocated_bytes () in
    let r = Sys.opaque_identity (g ()) in
    (Gc.allocated_bytes () -. before, r)
  in
  let empty, () = measure (fun () -> ()) in
  let bytes, r = measure f in
  ( int_of_float ((bytes -. empty) /. float_of_int (Sys.word_size / 8)),
    Obj.reachable_words (Obj.repr r) )

let text = lazy (Lz_support.corpus Text (256 * kib))

let decode_allocates (module C : Codec) =
  test "decode allocates no memory but its result" (fun () ->
      let s = Lazy.force text in
      let src = Lz_support.of_string (encode (module C) s) in
      let dst = Lz_support.create (String.length s) in
      let words, result = allocated (fun () -> C.decode src dst) in
      equal int result words)

let encode_allocates (module C : Codec) =
  test "encode allocates its tables outside the OCaml heap" (fun () ->
      let src = Lz_support.of_string (Lazy.force text) in
      let dst = Lz_support.create (C.bound (Bigarray.Array1.dim src)) in
      let words, result = allocated (fun () -> C.encode src dst) in
      equal int result words)

let round_trip (module C : Codec) =
  prop ~examples:(kinds @ large) "decode (encode x) = x" placed
    (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let n = String.length s in
      let src = Lz_support.placed ~offset s in
      let dst = Lz_support.room ~offset (C.bound n) in
      let k = require_some (C.encode src dst) in
      let out = Lz_support.room ~offset:(offset + 1) n in
      match C.decode (Bigarray.Array1.sub dst 0 k) out with
      | Error e -> failf "%a" (Testable.pp Lz_support.error) e
      | Ok p ->
          equal Lz_support.progress { Lz.consumed = k; produced = n } p;
          equal Lz_support.decoded (Ok s) (Ok (Lz_support.to_string out)))

let exact (module C : Codec) =
  prop ~examples:kinds "a destination of the stream's length holds it" placed
    (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let z = encode (module C) s in
      let k = String.length z in
      let dst = Lz_support.create k in
      equal (option int) (Some k) (C.encode (Lz_support.placed ~offset s) dst);
      equal string z (Lz_support.to_string dst))

let short (module C : Codec) =
  prop ~examples:kinds "a destination one byte short of the stream is refused"
    placed (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let k = String.length (encode (module C) s) in
      let dst = Lz_support.create (k - 1) in
      is_none (C.encode (Lz_support.placed ~offset s) dst))

let within (module C : Codec) =
  prop ~examples:kinds "an encode writes nothing past its destination" placed
    (fun (r, offset) ->
      let s = Lz_support.build r in
      cover_recipe r s;
      let src = Lz_support.placed ~offset s in
      let k = String.length (encode (module C) s) in
      List.iter
        (fun n ->
          let g = guarded n in
          ignore (C.encode src (snd g));
          untouched g)
        [ C.bound (String.length s); k - 1 ])

let encode_string (module C : Codec) =
  prop ~examples:Lz_support.every_kind "encode_string writes encode's bytes"
    Lz_support.recipe (fun r ->
      let s = Lz_support.build r in
      cover_recipe r s;
      equal string (encode (module C) s) (C.encode_string s))

let bound (module C : Codec) =
  group "bound"
    [
      test "a negative size has no bound" (fun () ->
          raises_match Exn.invalid_arg (fun () -> C.bound (-1)));
      test "a bound past max_int raises" (fun () ->
          Lz_support.largest_bound C.bound);
    ]

let encoding c =
  group "encode"
    [ round_trip c; exact c; short c; within c; encode_allocates c ]

let () =
  exit
    (run "Lz4"
       [
         group "Block"
           [
             block_sequences;
             block_refusals;
             block_messages;
             group "decode"
               [ every_golden block "lz4b"; decode_allocates block ];
             bound block;
             encoding block;
             group "encode_string"
               [
                 encode_string block;
                 test "the empty string is a zero token" (fun () ->
                     equal string "\x00" (Lz.Lz4.Block.encode_string ""));
               ];
           ];
         group "Frame"
           [
             frame_frames;
             frame_refusals;
             frame_messages;
             group "decode"
               [ every_golden frame "lz4f"; decode_allocates frame ];
             bound frame;
             encoding frame;
             group "encode_string"
               [
                 encode_string frame;
                 test "the empty string is a frame of no blocks" (fun () ->
                     (* The header's 15 bytes, then the end mark and the
                        content's checksum. *)
                     let f = Lz.Lz4.Frame.encode_string "" in
                     equal int ~msg:"length" 23 (String.length f);
                     equal string ~msg:"after the header"
                       (end_mark ^ le32 (xxh32 ""))
                       (String.sub f 15 8));
               ];
           ];
       ])
