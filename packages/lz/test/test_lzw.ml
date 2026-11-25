(* TIFF's LZW through Lz.Lzw. Expected values come from TIFF 6.0's section 13:
   its worked example, its widths, its table of at most 4096 entries and its
   ClearCode and EndOfInformation. Streams come from Lz_support's encoder,
   written from the section's encoder, or are built code by code. The decoder is
   held to a model written from the section's decoding pseudocode, on those
   streams and on streams with bytes changed. An error's expected offset is the
   byte that holds the first bit of the code at fault, as lz.mli states it. *)

open Windtrap
open Lz_support

let clear = 256
let eoi = 257

(* Codes, each with the width the section gives it in a stream of [codes] read
   in order: 9 bits until string 510 is stored, 10 until 1022, 11 until 2046, 12
   after. A ClearCode empties the table; each code but the first after it stores
   a string. *)
let widths codes =
  let next = ref 258 and first = ref true in
  List.map
    (fun c ->
      let w =
        if !next < 511 then 9
        else if !next < 1023 then 10
        else if !next < 2047 then 11
        else 12
      in
      if c = clear then (
        next := 258;
        first := true)
      else if c <> eoi then (
        if not !first then incr next;
        first := false);
      (c, w))
    codes

let stream codes = lzw_pack (widths codes)

(* The byte that holds the first bit of the [i]th code of [coded]. *)
let byte_of coded i =
  List.fold_left ( + ) 0 (List.filteri (fun j _ -> j < i) (List.map snd coded))
  / 8

(* ClearCode, then [k] codes of single bytes, then EndOfInformation: the [k - 1]
   strings it stores are pairs. *)
let literals k = (clear :: List.init k (fun j -> j land 0xff)) @ [ eoi ]
let literal_bytes k = String.init k (fun j -> Char.chr (j land 0xff))

(* The model *)

(* The section's decoder over strings, reading [s] most significant bit first:
   the output and the bytes up to EndOfInformation's last, or the error lz.mli
   states, into a destination of [cap] bytes. A code that would add entry 4096
   is Malformed whatever [cap]: a decode is Too_long only when a larger
   destination would let it go on. *)
let model ~cap s =
  let n = String.length s in
  let bit = ref 0 in
  let code w =
    if !bit + w > 8 * n then None
    else begin
      let v = ref 0 in
      for k = 0 to w - 1 do
        let b = !bit + k in
        let set = Char.code s.[b / 8] land (0x80 lsr (b mod 8)) <> 0 in
        v := (!v lsl 1) lor if set then 1 else 0
      done;
      bit := !bit + w;
      Some !v
    end
  in
  let out = Buffer.create 64 in
  let table = Array.make 4096 "" in
  let next = ref 258 and old = ref None in
  let fail at why = Error { Lz.at; why } in
  let width () =
    if !next < 511 then 9
    else if !next < 1023 then 10
    else if !next < 2047 then 11
    else 12
  in
  let string_of c =
    if c < 256 then Some (String.make 1 (Char.chr c))
    else if c >= 258 && c < !next then Some table.(c)
    else None
  in
  let rec loop ~started =
    let at = !bit / 8 in
    match code (width ()) with
    | None -> fail n Lz.Truncated
    | Some c when (not started) && c <> clear ->
        fail at (Lz.Malformed "the first code is not ClearCode")
    | Some c when c = clear ->
        next := 258;
        old := None;
        loop ~started:true
    | Some c when c = eoi -> Ok (Buffer.contents out, (!bit + 7) / 8)
    | Some c -> (
        let str =
          match (string_of c, !old) with
          | Some str, _ -> Some str
          | None, Some o when c = !next -> Some (o ^ String.sub o 0 1)
          | None, _ -> None
        in
        match (str, !old) with
        | None, _ -> fail at (Lz.Malformed "invalid code")
        | Some _, Some _ when !next = 4096 ->
            fail at (Lz.Malformed "the table is full")
        | Some str, _ when Buffer.length out + String.length str > cap ->
            fail at Lz.Too_long
        | Some str, o ->
            Buffer.add_string out str;
            Option.iter
              (fun o ->
                table.(!next) <- o ^ String.sub str 0 1;
                incr next)
              o;
            old := Some str;
            loop ~started)
  in
  if n >= 2 && s.[0] = '\x00' && Char.code s.[1] land 1 = 1 then
    fail 0 (Lz.Unsupported "codes least significant bit first")
  else loop ~started:false

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

(* A refusal's kind, whose message [outcome] does not compare. *)
let malformed = Lz.Malformed ""

(* [decode ~offset ~cap s] decodes [s], placed at [offset], into [cap] bytes at
   [offset]: the output and the bytes consumed, or the error. *)
let decode ?(offset = 0) ~cap s =
  let dst = room ~offset cap in
  match Lz.Lzw.decode (placed ~offset s) dst with
  | Ok { consumed; produced } -> Ok (to_string ~length:produced dst, consumed)
  | Error e -> Error e

let decodes ?(cap = 0) s expected =
  let cap = Int.max cap (String.length expected) in
  equal outcome (Ok (expected, String.length s)) (decode ~cap s)

let fails ?(cap = 1 lsl 16) ~at why s =
  equal outcome (Error { Lz.at; why }) (decode ~cap s)

(* Refused streams *)

let not_clear = lzw_pack [ (65, 9); (eoi, 9) ]
let above = widths [ clear; 65; 259; eoi ]
let after_clear = widths [ clear; 65; clear; 258; eoi ]

(* [full last] is ClearCode, 3839 single bytes, which fill the table, then
   [last], which would add entry 4096, then EndOfInformation. Its code [last] is
   its 3840th. *)
let full last =
  widths ((clear :: List.init 3839 (fun j -> j land 0xff)) @ [ last; eoi ])

(* [coded] packed least significant bit first, as no TIFF reader reads. *)
let lsb_first coded =
  let b = Buffer.create 16 in
  let acc = ref 0 and bits = ref 0 in
  List.iter
    (fun (c, n) ->
      acc := !acc lor (c lsl !bits);
      bits := !bits + n;
      while !bits >= 8 do
        Buffer.add_char b (Char.chr (!acc land 0xff));
        acc := !acc lsr 8;
        bits := !bits - 8
      done)
    coded;
  if !bits > 0 then Buffer.add_char b (Char.chr !acc);
  Buffer.contents b

let lsb = lsb_first (widths [ clear; 65; eoi ])

(* Codes *)

let codes =
  group "codes"
    [
      test "the section's code stream decodes to its pixels" (fun () ->
          (* "the output code stream, <7><258><8><8><258><6>", then the code of
             the string held at the strip's end, <6>, between ClearCode and
             EndOfInformation, all of them 9 bits wide. *)
          decodes
            (stream [ clear; 7; 258; 8; 8; 258; 6; 6; eoi ])
            "\007\007\007\008\008\007\007\006\006");
      (* Code [j] after ClearCode adds entry [256 + j], so code 255 is the first
         read once the table holds entry 510. A decoder that widens early or
         late misreads the codes at the switch. *)
      cases
        "a code is 10, 11 and 12 bits wide once the table holds entry 510, \
         1022 and 2046"
        ~id:(fun (e, w) -> Printf.sprintf "entry %d, %d bits" e w)
        [ (510, 10); (1022, 11); (2046, 12) ]
        (fun (entry, _) ->
          let k = entry - 256 + 1 in
          decodes (stream (literals k)) (literal_bytes k));
      test "the table holds entry 4095, which the 3839th single byte adds"
        (fun () ->
          let k = 4095 - 256 in
          decodes (stream (literals k)) (literal_bytes k));
      test "ClearCode after a full table empties it and sets the width to 9"
        (fun () ->
          (* Entry 258 is "AB" again, read 9 bits wide. *)
          let k = 4095 - 256 in
          let codes =
            (clear :: List.init k (fun j -> j land 0xff))
            @ [ clear; 65; 66; 258; eoi ]
          in
          decodes (stream codes) (literal_bytes k ^ "ABAB"));
      test "ClearCode may follow ClearCode" (fun () ->
          decodes (stream [ clear; 65; clear; clear; 66; eoi ]) "AB");
      test "EndOfInformation right after ClearCode ends an empty stream"
        (fun () -> decodes (stream [ clear; eoi ]) "");
      test
        "a code equal to the next entry is the string before it and its first \
         byte" (fun () ->
          decodes (stream [ clear; 65; 258; 259; eoi ]) "AAAAAA");
    ]

(* Refusals *)

let malformed_streams =
  group "Malformed"
    [
      test "a first code other than ClearCode is Malformed at byte 0" (fun () ->
          fails ~at:0 malformed not_clear);
      test "a code above the next entry is Malformed at its first byte"
        (fun () -> fails ~at:(byte_of above 2) malformed (lzw_pack above));
      test
        "the next entry's code right after ClearCode is Malformed at its first \
         byte" (fun () ->
          fails ~at:(byte_of after_clear 3) malformed (lzw_pack after_clear));
      test "a code that would add entry 4096 is Malformed at its first byte"
        (fun () ->
          let coded = full 65 in
          fails ~at:(byte_of coded 3840) malformed (lzw_pack coded));
    ]

let unsupported =
  group "Unsupported"
    [
      test "ClearCode least significant bit first is Unsupported at byte 0"
        (fun () -> fails ~at:0 (Unsupported "") lsb);
    ]

let too_long =
  group "Too_long"
    [
      prop "a destination too short is Too_long at the code that passes it"
        ~examples:(List.map (fun sh -> (Corpus (Text, 1000), sh)) short_edges)
        (Gen.pair recipe short)
        (fun (r, sh) ->
          let s = build r in
          assume (s <> "");
          let z = lzw s in
          let cap = capacity (String.length s) sh in
          cover "an empty destination" (cap = 0);
          match decode ~cap z with
          | Error { why = Too_long; _ } as r -> equal outcome (model ~cap z) r
          | r -> failf "%a, not Too_long" (Testable.pp outcome) r);
      (* No destination lets the decode go on past a full table. The 3839 bytes
         before the code fill dst; the code is a byte, written in place, or an
         entry, copied. *)
      cases "a code past dst's end that would add entry 4096 is Malformed"
        ~id:fst
        [ ("a byte", 65); ("an entry", 258) ]
        (fun (_, last) ->
          let coded = full last in
          fails ~cap:3839 ~at:(byte_of coded 3840) malformed (lzw_pack coded));
    ]

let truncated =
  group "Truncated"
    [
      test "an empty input is Truncated at 0" (fun () ->
          fails ~at:0 Truncated "");
      test "every proper prefix is Truncated at its length" (fun () ->
          let z = lzw (build (Corpus (Text, 2000))) in
          for n = 0 to String.length z - 1 do
            fails ~at:n Truncated (String.sub z 0 n)
          done);
    ]

let messages =
  group "pp_error"
    [
      test "prints each refusal's kind, offset and message" (fun () ->
          let refused =
            [
              ("empty", 16, "");
              ("first code", 16, not_clear);
              ("least significant bit first", 16, lsb);
              ("above the next entry", 16, lzw_pack above);
              ("after ClearCode", 16, lzw_pack after_clear);
              ("entry 4096", 1 lsl 16, lzw_pack (full 65));
              ("entry 4096 past dst", 3839, lzw_pack (full 65));
              ("past dst", 1, stream [ clear; 65; 66; eoi ]);
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
            first code: malformed at byte 0: the first code is not ClearCode
            least significant bit first: unsupported at byte 0: codes least significant bit first
            above the next entry: malformed at byte 2: invalid code
            after ClearCode: malformed at byte 3: invalid code
            entry 4096: malformed at byte 5408: the table is full
            entry 4096 past dst: malformed at byte 5408: the table is full
            past dst: output too long at byte 2
          |});
    ]

(* Whole-buffer decoding *)

let encodings =
  (* The section's two ways to keep the byte held at a full table, a table
     filled to its last entry, and tables cleared early. *)
  [
    ("ClearCode after entry 4094", 4094, false);
    ("the byte held written before ClearCode, after 4093", 4093, true);
    ("ClearCode after entry 4095", 4095, false);
    ("the byte held written after entry 4095", 4095, true);
    ("ClearCode after every entry", 258, false);
    ("ClearCode after entry 300, the byte held before it", 300, true);
  ]

let encoding =
  Gen.of_list
    ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
    encodings

(* Whether the codes [iter] gives, read in order, hold a code 12 bits wide and a
   code equal to the next entry. *)
let traits iter =
  let next = ref 258 and first = ref true in
  let twelve = ref false and found = ref false in
  iter (fun c w ->
      if w = 12 then twelve := true;
      if c = clear then (
        next := 258;
        first := true)
      else if c <> eoi then begin
        if (not !first) && c = !next then found := true;
        if not !first then incr next;
        first := false
      end);
  (!twelve, !found)

(* Every kind of input with the section's encoding, and an input that fills the
   table many times with each encoding. *)
let round_trips =
  List.map (fun r -> (r, List.hd encodings)) every_kind
  @ List.map (fun e -> (Corpus (Columns, (128 * 1024) + 1), e)) encodings

(* The stream of "a", ClearCode, 97 and EndOfInformation 9 bits wide, with byte
   1 xor 0x01, which reads 101 for 97, and with byte 3 xor 0x40, which reads 259
   for EndOfInformation; the first and last bytes of a stream of 1000 bytes of
   text, changed. *)
let changes =
  let e = List.hd encodings and r = Corpus (Text, 1000) in
  [
    (Byte 'a', e, 1, 0x01);
    (Byte 'a', e, 3, 0x40);
    (r, e, 0, 0x80);
    (r, e, -1, 1);
  ]

let decoding =
  group "decode"
    [
      prop
        "decodes the section's encoder's stream to its input, all of it \
         consumed"
        ~examples:round_trips (Gen.pair recipe encoding)
        (fun (r, (_, last, flush)) ->
          let s = build r in
          let twelve, next_code = traits (lzw_iter ~last ~flush s) in
          cover "empty" (s = "");
          cover "12-bit codes" twelve;
          cover "a code equal to the next entry" next_code;
          let z = lzw ~last ~flush s in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) z);
          equal outcome
            (Ok (s, String.length z))
            (decode ~offset:3 ~cap:(String.length s + 5) z));
      prop "does not read the bytes after EndOfInformation"
        (Gen.pair recipe Gen.string) (fun (r, tail) ->
          let s = build r in
          let z = lzw s in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) (z ^ tail)));
      prop "decodes a stream with a byte changed as the model decodes it"
        ~examples:changes
        (Gen.quad recipe encoding anywhere (Gen.int_range 1 255))
        (fun (r, (_, last, flush), i, mask) ->
          let s = build r in
          let z = Bytes.of_string (lzw ~last ~flush s) in
          let i = position i (Bytes.length z) in
          Bytes.set z i (Char.chr (Char.code (Bytes.get z i) lxor mask));
          let z = Bytes.to_string z in
          let cap = (2 * String.length s) + 16 in
          let m = model ~cap z in
          cover "decodes" (Result.is_ok m);
          cover "an invalid code"
            (match m with
            | Error { why = Malformed "invalid code"; _ } -> true
            | _ -> false);
          equal outcome m (decode ~cap z));
      (* 0x80, then a byte whose top bit is clear, begins with ClearCode. *)
      prop "decodes any codes after ClearCode as the model decodes them"
        (Gen.pair Gen.string (Gen.int_range 0 64))
        (fun (tail, cap) ->
          let tail =
            String.mapi
              (fun i c -> if i = 0 then Char.chr (Char.code c land 0x7f) else c)
              tail
          in
          let z = "\x80" ^ tail in
          equal outcome (model ~cap z) (decode ~cap z));
      test "allocates nothing but its result" (fun () ->
          let s = build (Corpus (Text, 64 * 1024)) in
          let src = of_string (lzw s) and dst = create (String.length s) in
          let allocated f =
            let before = Gc.allocated_bytes () in
            let r = Sys.opaque_identity (f ()) in
            (r, int_of_float (Gc.allocated_bytes () -. before))
          in
          let (), baseline = allocated (fun () -> ()) in
          let r, bytes = allocated (fun () -> Lz.Lzw.decode src dst) in
          let words = Obj.reachable_words (Obj.repr r) in
          equal int (baseline + (words * (Sys.word_size / 8))) bytes);
    ]

let () =
  exit
    (run "Lzw"
       [
         codes;
         malformed_streams;
         unsupported;
         too_long;
         truncated;
         messages;
         decoding;
       ])
