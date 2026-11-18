(* LZF blocks through Lz.Lzf. LZF has no specification: expected values
   come from the elements lz.mli states after liblzf's decoder, and from a
   block liblzf wrote. Blocks come from Lz_support's encoder, written from
   those elements, or are built element by element. The decoder is held to
   a model written from lz.mli's statement, on those blocks and on blocks
   with bytes changed. *)

open Windtrap
open Lz_support

(* The model *)

(* lz.mli's decoder over strings: the output of [s], or the error lz.mli
   states, into a destination of [cap] bytes. *)
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

let outcome =
  let pp ppf = function
    | Ok (s, consumed) ->
        Format.fprintf ppf "Ok (%d bytes %s, consumed %d)" (String.length s)
          (Digest.to_hex (Digest.string s))
          consumed
    | Error e -> Format.fprintf ppf "Error (%a)" Lz.pp_error e
  in
  Testable.make ~pp ~equal:( = )

(* [decode ~offset ~cap s] decodes [s], placed at [offset], into [cap]
   bytes at [offset]: the output and the bytes consumed, or the error. *)
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

(* Elements *)

let elements =
  group "elements"
    [
      test "liblzf's block of \"hello hello hello hello\"" (fun () ->
          (* liblzf's, through python-lzf 0.2.6: a literal run of 7, a
             match of 7 + 5 + 2 bytes from 6 back, which overlaps its
             output, and a literal run of 2. *)
          decodes "\x06hello h\xe0\x05\x05\x01lo" "hello hello hello hello");
      cases "a literal run holds 1 to 32 bytes" ~id:string_of_int
        [ 1; 2; 31; 32 ]
        (fun n ->
          let l = String.init n (fun i -> Char.chr (65 + i)) in
          decodes (lzf_pack [ Literal l ]) l);
      cases "a match's length field"
        ~id:(fun (field, extra) -> Printf.sprintf "%d, then %d" field extra)
        [ (1, 0); (6, 0); (7, 0); (7, 1); (7, 255) ]
        (fun (field, extra) ->
          (* Byte 0x41, then a match from 1 back of the field's length,
             plus the next byte when the field is 7. *)
          let length = field + 2 + if field = 7 then extra else 0 in
          let block =
            "\x00A"
            ^ (if field = 7 then
                 String.init 2 (function 0 -> '\xe0' | _ -> Char.chr extra)
               else String.make 1 (Char.chr (field lsl 5)))
            ^ "\x00"
          in
          decodes block (String.make (length + 1) 'A'));
      test "a match from 8192 back, the farthest" (fun () ->
          let s = String.init 8192 (fun i -> Char.chr (i * 7 land 0xff)) in
          let lits =
            List.init 256 (fun k -> Literal (String.sub s (32 * k) 32))
          in
          let z = lzf_pack (lits @ [ Copy { distance = 8192; length = 3 } ]) in
          decodes z (s ^ String.sub s 0 3));
      test "a match from as far back as the output reaches" (fun () ->
          decodes
            (lzf_pack [ Literal "abc"; Copy { distance = 3; length = 264 } ])
            ("abc" ^ String.concat "" (List.init 88 (fun _ -> "abc"))));
    ]

(* Round trips *)

let round_trips =
  group "round trips"
    [
      prop "the encoder's block decodes to its input" recipe (fun r ->
          let s = build r in
          assume (s <> "");
          let es = lzf_elements s in
          cover "an overlapping match"
            (List.exists
               (function
                 | Copy { distance; length } -> length > distance
                 | _ -> false)
               es);
          cover "a match of 10 bytes or more"
            (List.exists
               (function Copy { length; _ } -> length >= 10 | _ -> false)
               es);
          let z = lzf_pack es in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) z);
          equal outcome
            (Ok (s, String.length z))
            (decode ~offset:3 ~cap:(String.length s + 5) z));
      prop "a destination too short is Too_long, as the model says"
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
      test "every prefix decodes as the model decodes it" (fun () ->
          let z = lzf (build (Corpus (Text, 2000))) in
          for n = 0 to String.length z do
            let p = String.sub z 0 n in
            equal outcome ~msg:(string_of_int n) (model ~cap:4000 p)
              (decode ~cap:4000 p)
          done);
    ]

(* Refusals *)

let refusals =
  group "refusals"
    [
      test "an empty input is Truncated at 0" (fun () ->
          fails ~at:0 Truncated "");
      test "a literal run cut short is Truncated at the input's end" (fun () ->
          fails ~at:5 Truncated "\x00a\x05bc");
      test "a match without its distance byte" (fun () ->
          fails ~at:3 Truncated "\x00a\x20");
      test "a long match without its length byte" (fun () ->
          fails ~at:3 Truncated "\x00a\xe0");
      test "a long match without its distance byte" (fun () ->
          fails ~at:4 Truncated "\x00a\xe0\x00");
      test "a match from one byte before the block's start" (fun () ->
          fails ~at:3
            (Malformed "invalid offset too far back")
            "\x01ab\x20\x02");
      test "a match at the block's first byte" (fun () ->
          fails ~at:0 (Malformed "invalid offset too far back") "\x20\x00");
      test "a literal run past the destination is Too_long before Truncated"
        (fun () -> fails ~cap:2 ~at:3 Too_long "\x01ab\x05c");
      test "a match past the destination is Too_long at its control byte"
        (fun () -> fails ~cap:4 ~at:2 Too_long "\x00a\x40\x00");
      test "a source and destination that share a byte" (fun () ->
          let b = create 64 in
          raises_match (Exn.invalid_arg ~sub:"overlap") (fun () ->
              Lz.Lzf.decode (Bigarray.Array1.sub b 0 33)
                (Bigarray.Array1.sub b 32 32)));
    ]

(* Changed bytes *)

let changed =
  group "changed bytes"
    [
      prop "a block with a byte changed decodes as the model decodes it"
        (Gen.triple recipe Gen.small_int (Gen.int_range 1 255))
        (fun (r, i, mask) ->
          let s = build r in
          assume (s <> "");
          let z = Bytes.of_string (lzf s) in
          let i = (i land max_int) mod Bytes.length z in
          Bytes.set z i (Char.chr (Char.code (Bytes.get z i) lxor mask));
          let z = Bytes.to_string z in
          let cap = 2 * String.length s + 16 in
          let m = model ~cap z in
          cover "decodes" (Result.is_ok m);
          cover "too far back"
            (match m with
            | Error { why = Malformed _; _ } -> true
            | _ -> false);
          equal outcome m (decode ~cap z));
      prop "any bytes decode as the model decodes them"
        (Gen.pair Gen.string (Gen.int_range 0 64))
        (fun (z, cap) -> equal outcome (model ~cap z) (decode ~cap z));
    ]

let () =
  exit (run "lz.lzf" [ elements; round_trips; refusals; changed ])
