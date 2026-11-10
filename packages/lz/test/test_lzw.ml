(* TIFF's LZW through Lz.Lzw. Expected values come from TIFF 6.0's section
   13: its worked example, its widths, its table of at most 4096 entries
   and its ClearCode and EndOfInformation. Streams come from Lz_support's
   encoder, written from the section's encoder, or are built code by code.
   The decoder is held to a model written from the section's decoding
   pseudocode, on those streams and on streams with bytes changed. An
   error's expected offset is the byte that holds the first bit of the
   code at fault, as lz.mli states it. *)

open Windtrap
open Lz_support

let clear = 256
let eoi = 257

(* Codes, each with the width the section gives it in a stream of [codes]
   read in order: 9 bits until string 510 is stored, 10 until 1022, 11
   until 2046, 12 after. A ClearCode empties the table; each code but the
   first after it stores a string. *)
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

(* The model *)

(* The section's decoder over strings, reading [s] most significant bit
   first: the output and the bytes up to EndOfInformation's last, or the
   error lz.mli states, into a destination of [cap] bytes. *)
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
  let table = Hashtbl.create 4096 in
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
    else Hashtbl.find_opt table c
  in
  let rec loop ~started =
    let at = !bit / 8 in
    match code (width ()) with
    | None -> fail n Lz.Truncated
    | Some c when (not started) && c <> clear ->
        fail at (Lz.Malformed "the first code is not ClearCode")
    | Some c when c = clear ->
        Hashtbl.reset table;
        next := 258;
        old := None;
        loop ~started:true
    | Some c when c = eoi -> Ok (Buffer.contents out, (!bit + 7) / 8)
    | Some c -> (
        let str =
          match (string_of c, !old) with
          | Some str, _ when c <> clear && c <> eoi -> Some str
          | None, Some o when c = !next -> Some (o ^ String.sub o 0 1)
          | _ -> None
        in
        match str with
        | None -> fail at (Lz.Malformed "invalid code")
        | Some str ->
            if Buffer.length out + String.length str > cap then
              fail at Lz.Too_long
            else begin
              Buffer.add_string out str;
              match !old with
              | Some _ when !next = 4096 ->
                  fail at (Lz.Malformed "the table is full")
              | Some o ->
                  Hashtbl.replace table !next (o ^ String.sub str 0 1);
                  incr next;
                  old := Some str;
                  loop ~started
              | None ->
                  old := Some str;
                  loop ~started
            end)
  in
  if n >= 2 && s.[0] = '\x00' && Char.code s.[1] land 1 = 1 then
    fail 0 (Lz.Unsupported "codes least significant bit first")
  else loop ~started:false

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
  match Lz.Lzw.decode (placed ~offset s) dst with
  | Ok { consumed; produced } -> Ok (to_string ~length:produced dst, consumed)
  | Error e -> Error e

let decodes ?(cap = 0) s expected =
  let cap = Int.max cap (String.length expected) in
  equal outcome (Ok (expected, String.length s)) (decode ~cap s)

let fails ?(cap = 1 lsl 16) ~at why s =
  equal outcome (Error { Lz.at; why }) (decode ~cap s)

(* The section's example *)

let example =
  test "the section's example: pixels 7 7 7 8 8 7 7 6 6" (fun () ->
      (* "the output code stream, <7><258><8><8><258><6>", then the code of
         the string held at the strip's end, <6>, between ClearCode and
         EndOfInformation, all of them 9 bits wide. *)
      let pixels = "\007\007\007\008\008\007\007\006\006" in
      let codes = [ clear; 7; 258; 8; 8; 258; 6; 6; eoi ] in
      equal (list (pair int int))
        (List.map (fun c -> (c, 9)) codes)
        (lzw_codes pixels);
      decodes (stream codes) pixels)

(* Widths *)

(* ClearCode, then [k] codes of single bytes, then EndOfInformation: the
   [k - 1] strings it stores are pairs. *)
let literals k = (clear :: List.init k (fun j -> j land 0xff)) @ [ eoi ]
let literal_bytes k = String.init k (fun j -> Char.chr (j land 0xff))

let widen =
  (* Code [j] after ClearCode stores string [256 + j], so code 255 is the
     first read once string 510 is stored. *)
  cases "the width grows once strings 510, 1022 and 2046 are stored"
    ~id:(fun (s, w) -> Printf.sprintf "string %d, %d bits" s w)
    [ (510, 10); (1022, 11); (2046, 12) ]
    (fun (string, w) ->
      let k = string - 256 + 1 in
      let coded = widths (literals k) in
      equal int ~msg:"the last code's width" w (snd (List.nth coded k));
      equal int ~msg:"the code before" (w - 1) (snd (List.nth coded (k - 1)));
      decodes (lzw_pack coded) (literal_bytes k);
      (* A stream that widens one code later reads other bytes. *)
      let late =
        List.mapi (fun i (c, w') -> if i = k then (c, w - 1) else (c, w')) coded
      in
      let z = lzw_pack late in
      not_equal outcome ~msg:"one code late reads otherwise"
        (Ok (literal_bytes k, String.length z))
        (decode ~cap:(2 * k) z))

let table =
  group "the table"
    [
      test "holds 4096 entries: the 3839th single byte stores entry 4095"
        (fun () ->
          let k = 4095 - 256 in
          decodes (stream (literals k)) (literal_bytes k));
      test "a code that would store entry 4096 is refused" (fun () ->
          let k = 4096 - 256 in
          let coded = widths (literals k) in
          fails ~at:(byte_of coded k) (Lz.Malformed "the table is full")
            (lzw_pack coded));
      test "ClearCode after a full table starts again at 9 bits" (fun () ->
          let k = 4095 - 256 in
          let codes =
            (clear :: List.init k (fun j -> j land 0xff))
            @ [ clear; 65; 66; 258; eoi ]
          in
          let coded = widths codes in
          equal int ~msg:"after ClearCode" 9 (snd (List.nth coded (k + 2)));
          decodes (lzw_pack coded) (literal_bytes k ^ "ABAB"));
      test "a code equal to the next entry is the string before it and its \
            first byte" (fun () ->
          decodes (stream [ clear; 65; 258; 259; eoi ]) "AAAAAA");
    ]

(* Round trips *)

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
  Gen.of_list ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
    encodings

(* Whether reading [codes] meets a code equal to the next entry. *)
let has_next_code codes =
  let next = ref 258 and first = ref true and found = ref false in
  List.iter
    (fun (c, _) ->
      if c = clear then (
        next := 258;
        first := true)
      else if c <> eoi then begin
        if (not !first) && c = !next then found := true;
        if not !first then incr next;
        first := false
      end)
    codes;
  !found

let round_trips =
  group "round trips"
    [
      prop "the section's encoder's stream decodes to its input"
        (Gen.pair recipe encoding)
        (fun (r, (_, last, flush)) ->
          let s = build r in
          let coded = lzw_codes ~last ~flush s in
          cover "empty" (s = "");
          cover "12-bit codes" (List.exists (fun (_, w) -> w = 12) coded);
          cover "a code equal to the next entry" (has_next_code coded);
          let z = lzw_pack coded in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) z);
          equal outcome
            (Ok (s, String.length z))
            (decode ~offset:3 ~cap:(String.length s + 5) z));
      prop "bytes after EndOfInformation are not read"
        (Gen.pair recipe Gen.string)
        (fun (r, tail) ->
          let s = build r in
          let z = lzw s in
          equal outcome
            (Ok (s, String.length z))
            (decode ~cap:(String.length s) (z ^ tail)));
      prop "a destination too short is Too_long at the code that passes it"
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
      test "every proper prefix is Truncated at its length" (fun () ->
          let z = lzw (build (Corpus (Text, 2000))) in
          for n = 0 to String.length z - 1 do
            fails ~at:n Lz.Truncated (String.sub z 0 n)
          done);
    ]

(* Refusals *)

(* [codes] packed least significant bit first, as no TIFF reader reads. *)
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

let refusals =
  group "refusals"
    [
      test "an empty input is Truncated at 0" (fun () ->
          fails ~at:0 Truncated "");
      test "a stream that does not begin with ClearCode" (fun () ->
          fails ~at:0
            (Malformed "the first code is not ClearCode")
            (lzw_pack [ (65, 9); (eoi, 9) ]));
      test "ClearCode least significant bit first is Unsupported" (fun () ->
          fails ~at:0
            (Unsupported "codes least significant bit first")
            (lsb_first (widths [ clear; 65; eoi ])));
      test "a code above the next entry" (fun () ->
          let coded = widths [ clear; 65; 259; eoi ] in
          fails ~at:(byte_of coded 2) (Malformed "invalid code")
            (lzw_pack coded));
      test "the next entry's code right after ClearCode" (fun () ->
          let coded = widths [ clear; 65; clear; 258; eoi ] in
          fails ~at:(byte_of coded 3) (Malformed "invalid code")
            (lzw_pack coded));
      test "a source and destination that share a byte" (fun () ->
          let b = create 64 in
          raises_match (Exn.invalid_arg ~sub:"overlap") (fun () ->
              Lz.Lzw.decode (Bigarray.Array1.sub b 0 33)
                (Bigarray.Array1.sub b 32 32)));
      test "ClearCode twice, and EndOfInformation right after ClearCode"
        (fun () ->
          decodes (stream [ clear; 65; clear; clear; 66; eoi ]) "AB";
          decodes (stream [ clear; eoi ]) "");
    ]

(* Changed bytes *)

let changed =
  group "changed bytes"
    [
      prop "a stream with a byte changed decodes as the model decodes it"
        (Gen.quad recipe encoding Gen.small_int (Gen.int_range 1 255))
        (fun (r, (_, last, flush), i, mask) ->
          let z = lzw_pack (lzw_codes ~last ~flush (build r)) in
          let z = Bytes.of_string z in
          let i = (i land max_int) mod Bytes.length z in
          Bytes.set z i (Char.chr (Char.code (Bytes.get z i) lxor mask));
          let z = Bytes.to_string z in
          let cap = 2 * String.length (build r) + 16 in
          let m = model ~cap z in
          cover "decodes" (Result.is_ok m);
          cover "an invalid code"
            (match m with
            | Error { why = Malformed "invalid code"; _ } -> true
            | _ -> false);
          equal outcome m (decode ~cap z));
      prop "any bytes after ClearCode decode as the model decodes them"
        (Gen.pair Gen.string (Gen.int_range 0 64))
        (fun (tail, cap) ->
          let z = "\x80" ^ tail in
          equal outcome (model ~cap z) (decode ~cap z));
    ]

let () =
  exit
    (run "lz.lzw"
       [ example; widen; table; round_trips; refusals; changed ])
