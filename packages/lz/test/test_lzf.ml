(* LZF blocks through Lz.Lzf. LZF has no specification: expected values come
   from the elements lz.mli states after liblzf's decoder, and from a block
   liblzf wrote. Blocks come from Lz_support's encoder, written from those
   elements, or are built element by element. The decoder is held to a model
   written from lz.mli's statement, on those blocks and on blocks with bytes
   changed. *)

open Windtrap
open Lz_support

(* The model *)

(* lz.mli's decoder over strings: the output of [s], or the error lz.mli states,
   into a destination of [cap] bytes. A match too far back is Malformed whatever
   [cap]: a decode is Too_long only when a larger destination would let it go
   on. *)
let model ~cap s =
  let n = String.length s in
  let out = Buffer.create 64 in
  let fail at why = Error { Lz.at; why } in
  let rec loop i =
    if i = n then Ok (Buffer.contents out, n)
    else
      let c = Char.code s.[i] in
      if c < 32 then
        if Buffer.length out + c + 1 > cap then fail i Lz.Too_long
        else if i + 1 + c + 1 > n then fail n Lz.Truncated
        else begin
          Buffer.add_string out (String.sub s (i + 1) (c + 1));
          loop (i + 2 + c)
        end
      else
        let l = c lsr 5 in
        let l, j =
          if l = 7 && i + 1 < n then (l + Char.code s.[i + 1], i + 2)
          else (l, i + 1)
        in
        if j >= n then fail n Lz.Truncated
        else
          let distance = (((c land 31) lsl 8) lor Char.code s.[j]) + 1 in
          let length = l + 2 in
          if distance > Buffer.length out then
            fail i (Lz.Malformed "invalid offset too far back")
          else if Buffer.length out + length > cap then fail i Lz.Too_long
          else begin
            for _ = 1 to length do
              Buffer.add_char out
                (Buffer.nth out (Buffer.length out - distance))
            done;
            loop (j + 1)
          end
  in
  if n = 0 then fail 0 Lz.Truncated else loop 0

(* Calls *)

(* Decoded bytes and the bytes consumed, or the error, compared on its offset
   and kind: lz's messages are a baseline, which the pp_error test prints. *)
let outcome =
  let pp ppf (s, consumed) =
    Format.fprintf ppf "%d bytes %s, consumed %d" (String.length s)
      (Digest.to_hex (Digest.string s))
      consumed
  in
  result (Testable.make ~pp ~equal:( = )) refusal

(* [decode ~offset ~cap s] decodes [s], placed at [offset], into [cap] bytes at
   [offset]: the output and the bytes consumed, or the error. *)
let decode ?(offset = 0) ~cap s =
  let dst = room ~offset cap in
  match Lz.Lzf.decode (placed ~offset s) dst with
  | Ok { consumed; produced } -> Ok (to_string ~length:produced dst, consumed)
  | Error e -> Error e

let decodes ?(cap = 0) s expected =
  let cap = Int.max cap (String.length expected) in
  equal outcome (Ok (expected, String.length s)) (decode ~cap s)

let fails ?(cap = 1 lsl 16) ~at why s =
  equal outcome (Error { Lz.at; why }) (decode ~cap s)

(* Refused blocks: a literal run of "ab", then a match from 3 back, one byte
   before the block's start; a match at the block's first byte. *)
let before_start = "\x01ab\x20\x02"
let at_start = "\x20\x00"

(* Elements *)

let elements =
  group "elements"
    [
      test "liblzf's block of \"hello hello hello hello\" decodes to its input"
        (fun () ->
          (* liblzf's, through python-lzf 0.2.6: a literal run of 7, a match of
             7 + 5 + 2 bytes from 6 back, which overlaps its output, and a
             literal run of 2. *)
          decodes "\x06hello h\xe0\x05\x05\x01lo" "hello hello hello hello");
      cases "a literal run holds 1 to 32 bytes" ~id:string_of_int
        [ 1; 2; 31; 32 ] (fun n ->
          let l = String.init n (fun i -> Char.chr (65 + i)) in
          decodes (lzf_pack [ Literal l ]) l);
      cases
        "a match is (c lsr 5) + 2 bytes, plus the next byte when c lsr 5 is 7"
        ~id:(fun (field, extra) -> Printf.sprintf "%d, then %d" field extra)
        [ (1, 0); (6, 0); (7, 0); (7, 1); (7, 255) ]
        (fun (field, extra) ->
          (* Byte 0x41, then a match from 1 back of the field's length, plus the
             next byte when the field is 7. *)
          let length = field + 2 + if field = 7 then extra else 0 in
          let block =
            "\x00A"
            ^ (if field = 7 then
                 String.init 2 (function 0 -> '\xe0' | _ -> Char.chr extra)
               else String.make 1 (Char.chr (field lsl 5)))
            ^ "\x00"
          in
          decodes block (String.make (length + 1) 'A'));
      test "a match reaches 8192 bytes back" (fun () ->
          let s = String.init 8192 (fun i -> Char.chr (i * 7 land 0xff)) in
          let lits =
            List.init 256 (fun k -> Literal (String.sub s (32 * k) 32))
          in
          let z = lzf_pack (lits @ [ Copy { distance = 8192; length = 3 } ]) in
          decodes z (s ^ String.sub s 0 3));
      test "a match reaches the block's first byte" (fun () ->
          decodes
            (lzf_pack [ Literal "abc"; Copy { distance = 3; length = 264 } ])
            ("abc" ^ String.concat "" (List.init 88 (fun _ -> "abc"))));
    ]

(* Refusals *)

let malformed =
  group "Malformed"
    [
      test
        "a match from one byte before the block's start is Malformed at its \
         control byte" (fun () -> fails ~at:3 (Malformed "") before_start);
      test "a match at the block's first byte is Malformed at it" (fun () ->
          fails ~at:0 (Malformed "") at_start);
    ]

let truncated =
  group "Truncated"
    [
      test "an empty input is Truncated at 0" (fun () ->
          fails ~at:0 Truncated "");
      test "a literal run cut short is Truncated at the input's end" (fun () ->
          fails ~at:5 Truncated "\x00a\x05bc");
      test "a match without its distance byte is Truncated" (fun () ->
          fails ~at:3 Truncated "\x00a\x20");
      test "a long match without its length byte is Truncated" (fun () ->
          fails ~at:3 Truncated "\x00a\xe0");
      test "a long match without its distance byte is Truncated" (fun () ->
          fails ~at:4 Truncated "\x00a\xe0\x00");
      test "every prefix decodes as the model decodes it" (fun () ->
          let z = lzf (build (Corpus (Text, 2000))) in
          for n = 0 to String.length z do
            let p = String.sub z 0 n in
            equal outcome ~msg:(string_of_int n) (model ~cap:4000 p)
              (decode ~cap:4000 p)
          done);
    ]

let too_long =
  group "Too_long"
    [
      test "a literal run past the destination is Too_long before Truncated"
        (fun () -> fails ~cap:2 ~at:3 Too_long "\x01ab\x05c");
      test "a match past the destination is Too_long at its control byte"
        (fun () -> fails ~cap:4 ~at:2 Too_long "\x00a\x40\x00");
      prop "a destination too short is Too_long, as the model says"
        ~examples:(List.map (fun sh -> (Corpus (Text, 1000), sh)) short_edges)
        (Gen.pair recipe short)
        (fun (r, sh) ->
          let s = build r in
          assume (s <> "");
          let z = lzf s in
          let cap = capacity (String.length s) sh in
          cover "an empty destination" (cap = 0);
          match decode ~cap z with
          | Error { why = Too_long; _ } as r -> equal outcome (model ~cap z) r
          | r -> failf "%a, not Too_long" (Testable.pp outcome) r);
      (* No destination lets the decode go on past a match before the block's
         start. *)
      test "a match past dst's end and before the block's start is Malformed"
        (fun () -> fails ~cap:2 ~at:3 (Malformed "") before_start);
    ]

let messages =
  group "pp_error"
    [
      test "prints each refusal's kind, offset and message" (fun () ->
          let refused =
            [
              ("empty", 16, "");
              ("before the start", 16, before_start);
              ("at the start", 16, at_start);
              ("cut literal run", 16, "\x00a\x05bc");
              ("past dst", 4, "\x00a\x40\x00");
            ]
          in
          List.iter
            (fun (name, cap, s) ->
              match decode ~cap s with
              | Ok _ -> Format.printf "%s: decodes@." name
              | Error e -> Format.printf "%s: %a@." name Lz.pp_error e)
            refused;
          expect (output ())
          @@ __POS_OF__
               {|
            empty: truncated at byte 0
            before the start: malformed at byte 3: invalid offset too far back
            at the start: malformed at byte 0: invalid offset too far back
            cut literal run: truncated at byte 5
            past dst: output too long at byte 2
          |});
    ]

(* Whole-buffer decoding *)

(* The first and last bytes of a block of 1000 bytes of text, changed, and the
   literal of a one-byte block, changed into another that decodes. *)
let edges =
  let r = Corpus (Text, 1000) in
  [ (r, 0, 0x80); (r, -1, 1); (Byte 'a', 1, 1) ]

let decoding =
  group "decode"
    [
      (* A run of one byte is a literal run and matches from 1 back, each 264
         bytes long, overlapping its output. *)
      prop "decodes the encoder's block to its input, all of it consumed"
        ~examples:[ Run ('r', 1000) ]
        recipe
        (fun r ->
          let s = build r in
          assume (s <> "");
          let es = lzf_elements s in
          cover "an overlapping match"
            (List.exists
               (function
                 | Copy { distance; length } -> length > distance
                 | Literal _ -> false)
               es);
          cover "a match of 10 bytes or more"
            (List.exists
               (function
                 | Copy { length; _ } -> length >= 10 | Literal _ -> false)
               es);
          let z = lzf_pack es in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) z);
          equal outcome
            (Ok (s, String.length z))
            (decode ~offset:3 ~cap:(String.length s + 5) z));
      prop "decodes a block with a byte changed as the model decodes it"
        ~examples:edges
        (Gen.triple recipe anywhere (Gen.int_range 1 255))
        (fun (r, i, mask) ->
          let s = build r in
          assume (s <> "");
          let z = Bytes.of_string (lzf s) in
          let i = position i (Bytes.length z) in
          Bytes.set z i (Char.chr (Char.code (Bytes.get z i) lxor mask));
          let z = Bytes.to_string z in
          let cap = (2 * String.length s) + 16 in
          let m = model ~cap z in
          cover "decodes" (Result.is_ok m);
          cover "too far back"
            (match m with Error { why = Malformed _; _ } -> true | _ -> false);
          equal outcome m (decode ~cap z));
      prop "decodes any bytes as the model decodes them"
        (Gen.pair Gen.string (Gen.int_range 0 64))
        (fun (z, cap) -> equal outcome (model ~cap z) (decode ~cap z));
      test "allocates nothing but its result" (fun () ->
          let s = build (Corpus (Text, 64 * 1024)) in
          let src = of_string (lzf s) and dst = create (String.length s) in
          let allocated f =
            let before = Gc.allocated_bytes () in
            let r = Sys.opaque_identity (f ()) in
            (r, int_of_float (Gc.allocated_bytes () -. before))
          in
          let (), baseline = allocated (fun () -> ()) in
          let r, bytes = allocated (fun () -> Lz.Lzf.decode src dst) in
          let words = Obj.reachable_words (Obj.repr r) in
          equal int (baseline + (words * (Sys.word_size / 8))) bytes);
    ]

let () =
  exit
    (run "Lzf" [ elements; malformed; truncated; too_long; messages; decoding ])
