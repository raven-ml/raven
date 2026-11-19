(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* whisper/utils.py's writers: WriteTXT, WriteVTT, WriteSRT, WriteTSV and
   WriteJSON, over the result transcribe.py returns. *)

let segments (r : Transcript.t) =
  List.concat_map (fun (w : Transcript.window) -> w.segments) r.windows

(* Python's [s.replace (sub, by)]: each occurrence of [sub], from the left. *)
let replace ~sub ~by s =
  let n = String.length sub and len = String.length s in
  let b = Buffer.create len in
  let rec go i =
    if i > len - n then Buffer.add_substring b s i (len - i)
    else if String.sub s i n = sub then begin
      Buffer.add_string b by;
      go (i + n)
    end
    else begin
      Buffer.add_char b s.[i];
      go (i + 1)
    end
  in
  go 0;
  Buffer.contents b

(* utils.py, format_timestamp, from a frame of 10 ms. *)
let timestamp ~hours ~marker f =
  let ms = 10 * f in
  let h = ms / 3_600_000 and ms = ms mod 3_600_000 in
  let m = ms / 60_000 and ms = ms mod 60_000 in
  let s = ms / 1000 and ms = ms mod 1000 in
  let h = if hours || h > 0 then Printf.sprintf "%02d:" h else "" in
  Printf.sprintf "%s%02d:%02d%s%03d" h m s marker ms

let lines r f =
  let b = Buffer.create 1024 in
  List.iteri (fun i s -> Buffer.add_string b (f i s)) (segments r);
  Buffer.contents b

let txt r =
  lines r (fun _ (s : Transcript.segment) -> Space.strip s.text ^ "\n")

let cue (s : Transcript.segment) =
  replace ~sub:"-->" ~by:"->" (Space.strip s.text)

(* Subtitles' cues. When every segment is cleared upstream reads their empty
   word timings (transcribe.py gives a cleared segment [words = []]) and writes
   no cue. *)
let cues r f =
  let cleared (s : Transcript.segment) = s.tokens = [||] in
  if List.for_all cleared (segments r) then "" else lines r f

let vtt r =
  let ts = timestamp ~hours:false ~marker:"." in
  "WEBVTT\n\n"
  ^ cues r (fun _ (s : Transcript.segment) ->
      Printf.sprintf "%s --> %s\n%s\n\n" (ts s.start) (ts s.stop) (cue s))

let srt r =
  let ts = timestamp ~hours:true ~marker:"," in
  cues r (fun i (s : Transcript.segment) ->
      Printf.sprintf "%d\n%s --> %s\n%s\n\n" (i + 1) (ts s.start) (ts s.stop)
        (cue s))

let tsv r =
  "start\tend\ttext\n"
  ^ lines r (fun _ (s : Transcript.segment) ->
      Printf.sprintf "%d\t%d\t%s\n" (10 * s.start) (10 * s.stop)
        (replace ~sub:"\t" ~by:" " (Space.strip s.text)))

(* JSON as json.dump writes it *)

(* Python's float repr: the shortest decimal that reads back to [x], the nearest
   of those (Float_text.shortest), positional for a decimal exponent from -4 to
   15 with at least one digit after the point, scientific otherwise with a
   signed exponent of at least two digits. *)
let float_repr x =
  if Float.is_nan x then "NaN"
  else if x = Float.infinity then "Infinity"
  else if x = Float.neg_infinity then "-Infinity"
  else
    let sign = if Float.sign_bit x then "-" else "" in
    if x = 0. then sign ^ "0.0"
    else
      let digits, e = Float_text.shortest Float_text.binary64 x in
      let n = String.length digits in
      (* The exponent of the first digit. *)
      let exp = e + n - 1 in
      let body =
        if exp >= 16 || exp < -4 then
          let mant =
            if n = 1 then digits
            else String.sub digits 0 1 ^ "." ^ String.sub digits 1 (n - 1)
          in
          Printf.sprintf "%se%c%02d" mant
            (if exp < 0 then '-' else '+')
            (abs exp)
        else if exp < 0 then "0." ^ String.make (-exp - 1) '0' ^ digits
        else if n <= exp + 1 then digits ^ String.make (exp + 1 - n) '0' ^ ".0"
        else
          String.sub digits 0 (exp + 1)
          ^ "."
          ^ String.sub digits (exp + 1) (n - exp - 1)
      in
      sign ^ body

(* json's ensure_ascii: '"', '\\' and \b \f \n \r \t by name, every other
   character outside ' ' to '~' as \uXXXX, beyond U+FFFF as surrogates. *)
let string_repr s =
  let b = Buffer.create (String.length s + 2) in
  let u c = Buffer.add_string b (Printf.sprintf "\\u%04x" c) in
  Buffer.add_char b '"';
  let rec go i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let c = Uchar.to_int (Uchar.utf_decode_uchar d) in
      (match c with
      | 0x22 -> Buffer.add_string b "\\\""
      | 0x5C -> Buffer.add_string b "\\\\"
      | 0x08 -> Buffer.add_string b "\\b"
      | 0x0C -> Buffer.add_string b "\\f"
      | 0x0A -> Buffer.add_string b "\\n"
      | 0x0D -> Buffer.add_string b "\\r"
      | 0x09 -> Buffer.add_string b "\\t"
      | c when c >= 0x20 && c <= 0x7E -> Buffer.add_char b (Char.chr c)
      | c when c > 0xFFFF ->
          let c = c - 0x10000 in
          u (0xD800 lor (c lsr 10));
          u (0xDC00 lor (c land 0x3FF))
      | c -> u c);
      go (i + Uchar.utf_decode_length d)
    end
  in
  go 0;
  Buffer.add_char b '"';
  Buffer.contents b

type value =
  | Int of int
  | Float of float
  | String of string
  | List of value list
  | Object of (string * value) list

let rec add b = function
  | Int i -> Buffer.add_string b (string_of_int i)
  | Float x -> Buffer.add_string b (float_repr x)
  | String s -> Buffer.add_string b (string_repr s)
  | List l ->
      Buffer.add_char b '[';
      List.iteri
        (fun i v ->
          if i > 0 then Buffer.add_string b ", ";
          add b v)
        l;
      Buffer.add_char b ']'
  | Object mems ->
      Buffer.add_char b '{';
      List.iteri
        (fun i (k, v) ->
          if i > 0 then Buffer.add_string b ", ";
          Buffer.add_string b (string_repr k);
          Buffer.add_string b ": ";
          add b v)
        mems;
      Buffer.add_char b '}'

let temperature : Transcript.rung -> float = function
  | Greedy | Beam _ -> 0.
  | Sample { temperature; _ } -> temperature

let json (r : Transcript.t) =
  let seconds f = Float (float f /. 100.) in
  let id = ref (-1) in
  let segment (w : Transcript.window) (s : Transcript.segment) =
    incr id;
    let c = w.choice in
    (* transcribe.py gives a cleared segment empty words. *)
    let words = if s.tokens = [||] then [ ("words", List []) ] else [] in
    Object
      ([
         ("id", Int !id);
         ("seek", Int w.seek);
         ("start", seconds s.start);
         ("end", seconds s.stop);
         ("text", String s.text);
         ("tokens", List (List.map (fun t -> Int t) (Array.to_list s.tokens)));
         ("temperature", Float (temperature c.rung));
         ("avg_logprob", Float c.result.avg_logprob);
         ("compression_ratio", Float c.compression_ratio);
         ("no_speech_prob", Float c.result.no_speech_prob);
       ]
      @ words)
  in
  let segments =
    List.concat_map
      (fun (w : Transcript.window) -> List.map (segment w) w.segments)
      r.windows
  in
  let b = Buffer.create 4096 in
  add b
    (Object
       [
         ("text", String r.text);
         ("segments", List segments);
         ("language", String r.language);
       ]);
  Buffer.contents b
