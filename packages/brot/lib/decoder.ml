(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | BPE of { suffix : string }
  | Byte_level
  | Byte_fallback
  | Word_piece of { prefix : string; cleanup : bool }
  | Metaspace of {
      replacement : string;
      prepend_scheme : Pre_tokenizer.prepend_scheme;
    }
  | CTC of { pad_token : string; word_delimiter_token : string; cleanup : bool }
  | Sequence of t list
  | Replace of { pattern : string; replacement : string }
  | Strip of { content : string; start : int; stop : int }
  | Fuse

(* Errors *)

let strf = Printf.sprintf
let err_replace_missing_pattern = "missing pattern in Replace decoder"

let err_replace_regex_pattern =
  "Replace decoder: a regular expression pattern is not supported, only a \
   literal one"

let err_seq_missing_decoders =
  "invalid Sequence decoder: missing decoders array"

let err_unknown_type typ = strf "unknown decoder type: %s" typ
let err_expected_object = "invalid decoder JSON: expected object"
let err_unknown_scheme s = strf "unknown prepend_scheme '%s'" s

(* Decoding *)

(* Literal string replacement without regex overhead. Returns [s] unchanged when
   [pattern] does not occur—no allocation on the fast path. *)
let replace_all ~pattern ~by s =
  let plen = String.length pattern in
  let slen = String.length s in
  if plen = 0 || plen > slen then s
  else
    let match_at i =
      let rec check j =
        j >= plen
        || String.unsafe_get s (i + j) = String.unsafe_get pattern j
           && check (j + 1)
      in
      check 0
    in
    let rec find_first i =
      if i > slen - plen then -1
      else if match_at i then i
      else find_first (i + 1)
    in
    let pos = find_first 0 in
    if pos < 0 then s
    else
      let buf = Buffer.create slen in
      Buffer.add_substring buf s 0 pos;
      Buffer.add_string buf by;
      let i = ref (pos + plen) in
      while !i <= slen - plen do
        if match_at !i then (
          Buffer.add_string buf by;
          i := !i + plen)
        else (
          Buffer.add_char buf (String.unsafe_get s !i);
          incr i)
      done;
      if !i < slen then Buffer.add_substring buf s !i (slen - !i);
      Buffer.contents buf

(* The suffix marks the end of a word, so it stands for the space that follows
   it—except on the last token, where no space follows. *)
let decode_bpe ~suffix tokens =
  match tokens with
  | [] -> []
  | _ ->
      let last = List.length tokens - 1 in
      List.mapi
        (fun i token ->
          replace_all ~pattern:suffix ~by:(if i = last then "" else " ") token)
        tokens

let replacement_char = "\xef\xbf\xbd"

(* Bytes as text, the way HuggingFace reads them back: each maximal ill-formed
   subpart becomes one replacement character, which is what a decode that fails
   skips over. A lone [\xE9] and a four-byte sequence cut short cost one each,
   while [\xC3\x28] costs one and keeps the ['(']. *)
let utf8_lossy s =
  if String.is_valid_utf_8 s then s
  else begin
    let len = String.length s in
    let buf = Buffer.create (len + 8) in
    let i = ref 0 in
    while !i < len do
      let decode = String.get_utf_8_uchar s !i in
      let step = Uchar.utf_decode_length decode in
      if Uchar.utf_decode_is_valid decode then
        Buffer.add_substring buf s !i step
      else Buffer.add_string buf replacement_char;
      i := !i + step
    done;
    Buffer.contents buf
  end

(* Each token maps to bytes on its own, but the text they spell is read whole,
   so a character split across two tokens decodes and only what the bytes of no
   run of tokens spell is replaced. *)
let decode_byte_level tokens =
  let buf = Buffer.create 128 in
  List.iter
    (fun token -> Buffer.add_string buf (Pre_tokenizer.byte_level_decode token))
    tokens;
  utf8_lossy (Buffer.contents buf)

let hex_digit c =
  match c with
  | '0' .. '9' -> Char.code c - Char.code '0'
  | 'a' .. 'f' -> Char.code c - Char.code 'a' + 10
  | 'A' .. 'F' -> Char.code c - Char.code 'A' + 10
  | _ -> -1

let byte_token_value token =
  if
    String.length token = 6
    && String.starts_with ~prefix:"<0x" token
    && String.ends_with ~suffix:">" token
  then
    let hi = hex_digit (String.unsafe_get token 3) in
    let lo = hex_digit (String.unsafe_get token 4) in
    if hi >= 0 && lo >= 0 then Some (Char.unsafe_chr ((hi * 16) + lo)) else None
  else None

(* A run of byte tokens spells one UTF-8 sequence. A run that does not is
   unrecoverable, so each of its bytes becomes a replacement character. *)
let decode_byte_fallback tokens =
  let run = Buffer.create 16 in
  let flush acc =
    if Buffer.length run = 0 then acc
    else
      let bytes = Buffer.contents run in
      Buffer.clear run;
      if String.is_valid_utf_8 bytes then bytes :: acc
      else begin
        let acc = ref acc in
        for _ = 1 to String.length bytes do
          acc := replacement_char :: !acc
        done;
        !acc
      end
  in
  let rec loop acc = function
    | [] -> List.rev (flush acc)
    | token :: rest -> (
        match byte_token_value token with
        | Some b ->
            Buffer.add_char run b;
            loop acc rest
        | None -> loop (token :: flush acc) rest)
  in
  loop [] tokens

let detokenization_cleanups =
  [
    (" .", ".");
    (" ?", "?");
    (" !", "!");
    (" ,", ",");
    (" ' ", "'");
    (" n't", "n't");
    (" 'm", "'m");
    (" do not", " don't");
    (" 's", "'s");
    (" 've", "'ve");
    (" 're", "'re");
  ]

(* Undo the spacing around punctuation and English contractions; the last rule
   rewrites rather than unspaces. It sees one piece at a time, so only a space
   already inside that piece is taken back: a full stop that was a token of its
   own keeps the space that follows it. *)
let cleanup_piece piece =
  List.fold_left
    (fun s (pattern, by) -> replace_all ~pattern ~by s)
    piece detokenization_cleanups

let decode_wordpiece ~prefix ~cleanup tokens =
  let plen = String.length prefix in
  List.mapi
    (fun i token ->
      let piece =
        if i = 0 then token
        else if String.starts_with ~prefix token then
          String.sub token plen (String.length token - plen)
        else " " ^ token
      in
      if cleanup then cleanup_piece piece else piece)
    tokens

(* The replacement stands for a space, except on the first token when it was
   prepended: there it stands for nothing and every occurrence goes. *)
let decode_metaspace ~replacement ~prepend_scheme tokens =
  let prepended = prepend_scheme <> `Never in
  List.mapi
    (fun i token ->
      replace_all ~pattern:replacement
        ~by:(if prepended && i = 0 then "" else " ")
        token)
    tokens

let decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens =
  let rec dedup acc = function
    | [] -> List.rev acc
    | [ x ] -> List.rev (x :: acc)
    | x :: (y :: _ as rest) ->
        if String.equal x y then dedup acc rest else dedup (x :: acc) rest
  in
  dedup [] tokens
  |> List.filter_map (fun token ->
      let token = replace_all ~pattern:pad_token ~by:"" token in
      let token =
        if cleanup then
          replace_all ~pattern:word_delimiter_token ~by:" "
            (cleanup_piece token)
        else token
      in
      if String.equal token "" then None else Some token)

let strip_token ~content ~start ~stop token =
  let clen = String.length content in
  let len = String.length token in
  if clen = 0 then token
  else
    let match_at i =
      let rec check j =
        j >= clen
        || String.unsafe_get token (i + j) = String.unsafe_get content j
           && check (j + 1)
      in
      i + clen <= len && check 0
    in
    let first = ref 0 in
    let stripped = ref 0 in
    while !stripped < start && match_at !first do
      first := !first + clen;
      incr stripped
    done;
    let last = ref len in
    stripped := 0;
    while
      !stripped < stop && !last - clen >= !first && match_at (!last - clen)
    do
      last := !last - clen;
      incr stripped
    done;
    if !first >= !last then "" else String.sub token !first (!last - !first)

let rec decode_chain decoder tokens =
  match decoder with
  | BPE { suffix } -> decode_bpe ~suffix tokens
  | Byte_level -> [ decode_byte_level tokens ]
  | Byte_fallback -> decode_byte_fallback tokens
  | Word_piece { prefix; cleanup } -> decode_wordpiece ~prefix ~cleanup tokens
  | Metaspace { replacement; prepend_scheme } ->
      decode_metaspace ~replacement ~prepend_scheme tokens
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens
  | Replace { pattern; replacement } ->
      List.map (replace_all ~pattern ~by:replacement) tokens
  | Strip { content; start; stop } ->
      List.map (strip_token ~content ~start ~stop) tokens
  | Fuse -> [ String.concat "" tokens ]
  | Sequence decoders ->
      List.fold_left (fun toks dec -> decode_chain dec toks) tokens decoders

let decode decoder tokens = String.concat "" (decode_chain decoder tokens)

(* Constructors *)

let bpe ?(suffix = "</w>") () = BPE { suffix }
let byte_level () = Byte_level
let byte_fallback () = Byte_fallback

let wordpiece ?(prefix = "##") ?(cleanup = true) () =
  Word_piece { prefix; cleanup }

let metaspace ?(replacement = "\xe2\x96\x81") ?(prepend_scheme = `Always) () =
  Metaspace { replacement; prepend_scheme }

let ctc ?(pad_token = "<pad>") ?(word_delimiter_token = "|") ?(cleanup = true)
    () =
  CTC { pad_token; word_delimiter_token; cleanup }

let sequence decoders = Sequence decoders
let replace ~pattern ~by () = Replace { pattern; replacement = by }

let strip ?(content = " ") ?(start = 0) ?(stop = 0) () =
  Strip { content; start; stop }

let fuse () = Fuse

(* Formatting *)

let scheme_to_string = function
  | `First -> "first"
  | `Never -> "never"
  | `Always -> "always"

let rec pp ppf = function
  | BPE { suffix } -> Format.fprintf ppf "bpe ~suffix:%S" suffix
  | Byte_level -> Format.fprintf ppf "byte_level"
  | Byte_fallback -> Format.fprintf ppf "byte_fallback"
  | Word_piece { prefix; cleanup } ->
      Format.fprintf ppf "wordpiece ~prefix:%S ~cleanup:%b" prefix cleanup
  | Metaspace { replacement; prepend_scheme } ->
      Format.fprintf ppf "metaspace ~replacement:%S ~prepend_scheme:%s"
        replacement
        (scheme_to_string prepend_scheme)
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      Format.fprintf ppf
        "ctc ~pad_token:%S ~word_delimiter_token:%S ~cleanup:%b" pad_token
        word_delimiter_token cleanup
  | Replace { pattern; replacement } ->
      Format.fprintf ppf "replace ~pattern:%S ~by:%S" pattern replacement
  | Strip { content; start; stop } ->
      Format.fprintf ppf "strip ~content:%S ~start:%d ~stop:%d" content start
        stop
  | Fuse -> Format.fprintf ppf "fuse"
  | Sequence decoders ->
      Format.fprintf ppf "@[<hv 2>sequence [%a]@]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           pp)
        decoders

(* Serialization *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let rec to_json = function
  | BPE { suffix } ->
      json_obj
        [
          ("type", Jsont.Json.string "BPEDecoder");
          ("suffix", Jsont.Json.string suffix);
        ]
  (* The byte-level decoder reads neither member, but HuggingFace requires both
     to be present. *)
  | Byte_level ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("add_prefix_space", Jsont.Json.bool true);
          ("trim_offsets", Jsont.Json.bool true);
        ]
  | Byte_fallback -> json_obj [ ("type", Jsont.Json.string "ByteFallback") ]
  | Word_piece { prefix; cleanup } ->
      json_obj
        [
          ("type", Jsont.Json.string "WordPiece");
          ("prefix", Jsont.Json.string prefix);
          ("cleanup", Jsont.Json.bool cleanup);
        ]
  | Metaspace { replacement; prepend_scheme } ->
      json_obj
        [
          ("type", Jsont.Json.string "Metaspace");
          ("replacement", Jsont.Json.string replacement);
          ("prepend_scheme", Jsont.Json.string (scheme_to_string prepend_scheme));
        ]
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      json_obj
        [
          ("type", Jsont.Json.string "CTC");
          ("pad_token", Jsont.Json.string pad_token);
          ("word_delimiter_token", Jsont.Json.string word_delimiter_token);
          ("cleanup", Jsont.Json.bool cleanup);
        ]
  | Replace { pattern; replacement } ->
      json_obj
        [
          ("type", Jsont.Json.string "Replace");
          ("pattern", json_obj [ ("String", Jsont.Json.string pattern) ]);
          ("content", Jsont.Json.string replacement);
        ]
  | Strip { content; start; stop } ->
      json_obj
        [
          ("type", Jsont.Json.string "Strip");
          ("content", Jsont.Json.string content);
          ("start", Jsont.Json.int start);
          ("stop", Jsont.Json.int stop);
        ]
  | Fuse -> json_obj [ ("type", Jsont.Json.string "Fuse") ]
  | Sequence decoders ->
      json_obj
        [
          ("type", Jsont.Json.string "Sequence");
          ("decoders", Jsont.Json.list (List.map to_json decoders));
        ]

let find_field fields name = Option.map snd (Jsont.Json.find_mem name fields)

let string_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.String (s, _)) -> s
  | _ -> default

let bool_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.Bool (b, _)) -> b
  | _ -> default

let int_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.Number (f, _)) -> int_of_float f
  | _ -> default

let scheme_of_string = function
  | "first" -> Ok `First
  | "never" -> Ok `Never
  | "always" -> Ok `Always
  | s -> Error (err_unknown_scheme s)

let rec of_json = function
  | Jsont.Object (fields, _) -> (
      let ( let* ) = Result.bind in
      match find_field fields "type" with
      | Some (Jsont.String ("BPEDecoder", _)) ->
          Ok (BPE { suffix = string_field fields "suffix" ~default:"</w>" })
      | Some (Jsont.String ("ByteLevel", _)) -> Ok Byte_level
      | Some (Jsont.String ("ByteFallback", _)) -> Ok Byte_fallback
      | Some (Jsont.String ("WordPiece", _)) ->
          Ok
            (Word_piece
               {
                 prefix = string_field fields "prefix" ~default:"##";
                 cleanup = bool_field fields "cleanup" ~default:true;
               })
      | Some (Jsont.String ("Metaspace", _)) ->
          let* prepend_scheme =
            scheme_of_string
              (string_field fields "prepend_scheme" ~default:"always")
          in
          Ok
            (Metaspace
               {
                 replacement =
                   string_field fields "replacement" ~default:"\xe2\x96\x81";
                 prepend_scheme;
               })
      | Some (Jsont.String ("CTC", _)) ->
          Ok
            (CTC
               {
                 pad_token = string_field fields "pad_token" ~default:"<pad>";
                 word_delimiter_token =
                   string_field fields "word_delimiter_token" ~default:"|";
                 cleanup = bool_field fields "cleanup" ~default:true;
               })
      | Some (Jsont.String ("Replace", _)) ->
          let* pattern =
            match find_field fields "pattern" with
            | Some (Jsont.String (s, _)) -> Ok s
            | Some (Jsont.Object (pattern_fields, _)) -> (
                match Jsont.Json.find_mem "String" pattern_fields with
                | Some (_, Jsont.String (p, _)) -> Ok p
                | _ ->
                    if
                      Option.is_some
                        (Jsont.Json.find_mem "Regex" pattern_fields)
                    then Error err_replace_regex_pattern
                    else Error err_replace_missing_pattern)
            | _ -> Error err_replace_missing_pattern
          in
          Ok
            (Replace
               {
                 pattern;
                 replacement = string_field fields "content" ~default:"";
               })
      | Some (Jsont.String ("Strip", _)) ->
          Ok
            (Strip
               {
                 content = string_field fields "content" ~default:" ";
                 start = int_field fields "start" ~default:0;
                 stop = int_field fields "stop" ~default:0;
               })
      | Some (Jsont.String ("Fuse", _)) -> Ok Fuse
      | Some (Jsont.String ("Sequence", _)) -> (
          match find_field fields "decoders" with
          | Some (Jsont.Array (decs, _)) ->
              let* decoders =
                List.fold_left
                  (fun acc j ->
                    let* acc = acc in
                    let* d = of_json j in
                    Ok (d :: acc))
                  (Ok []) decs
              in
              Ok (Sequence (List.rev decoders))
          | _ -> Error err_seq_missing_decoders)
      | Some (Jsont.String (typ, _)) -> Error (err_unknown_type typ)
      | _ -> Error "missing or invalid decoder type field")
  | _ -> Error err_expected_object
