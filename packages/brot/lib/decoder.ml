(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | BPE of { suffix : string }
  | Byte_level
  | Byte_fallback
  | Word_piece of { prefix : string; cleanup : bool }
  | Metaspace of { replacement : char; add_prefix_space : bool }
  | CTC of { pad_token : string; word_delimiter_token : string; cleanup : bool }
  | Sequence of t list
  | Replace of { pattern : string; replacement : string }
  | Strip of { left : bool; right : bool; content : char }
  | Fuse

(* Errors *)

let strf = Printf.sprintf
let err_replace_missing_pattern = "missing pattern in Replace decoder"

let err_seq_missing_decoders =
  "invalid Sequence decoder: missing decoders array"

let err_unknown_type typ = strf "unknown decoder type: %s" typ
let err_expected_object = "invalid decoder JSON: expected object"

(* Decoding *)

let whitespace_re = Re.compile (Re.rep1 (Re.char ' '))

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

let decode_bpe ~suffix tokens =
  let suffix_len = String.length suffix in
  let strip token =
    if suffix_len > 0 && String.ends_with ~suffix token then
      String.sub token 0 (String.length token - suffix_len)
    else token
  in
  let rec loop acc = function
    | [] -> List.rev acc
    | [ token ] -> List.rev (strip token :: acc)
    | token :: rest -> loop (" " :: strip token :: acc) rest
  in
  loop [] tokens

let decode_byte_level tokens =
  let buf = Buffer.create 128 in
  List.iter
    (fun token -> Buffer.add_string buf (Pre_tokenizer.byte_level_decode token))
    tokens;
  Buffer.contents buf

let decode_byte_fallback tokens =
  let flush acc = function
    | [] -> acc
    | byte_acc ->
        let bytes = List.rev byte_acc in
        let s = Bytes.create (List.length bytes) in
        List.iteri (fun i b -> Bytes.unsafe_set s i (Char.chr b)) bytes;
        Bytes.unsafe_to_string s :: acc
  in
  let is_byte_token token =
    String.length token = 6
    && String.starts_with ~prefix:"<0x" token
    && String.ends_with ~suffix:">" token
  in
  let rec loop acc byte_acc = function
    | [] -> List.rev (flush acc byte_acc)
    | token :: rest when is_byte_token token -> (
        let hex = String.sub token 3 2 in
        match int_of_string_opt ("0x" ^ hex) with
        | Some b when b >= 0 && b <= 255 -> loop acc (b :: byte_acc) rest
        | _ -> loop (token :: flush acc byte_acc) [] rest)
    | token :: rest -> loop (token :: flush acc byte_acc) [] rest
  in
  loop [] [] tokens

let decode_wordpiece ~prefix ~cleanup tokens =
  let plen = String.length prefix in
  let buf = Buffer.create 128 in
  List.iteri
    (fun i token ->
      if i > 0 && String.starts_with ~prefix token then
        Buffer.add_substring buf token plen (String.length token - plen)
      else begin
        if i > 0 then Buffer.add_char buf ' ';
        Buffer.add_string buf token
      end)
    tokens;
  let s = Buffer.contents buf in
  if cleanup then String.trim (Re.replace_string whitespace_re ~by:" " s) else s

let decode_metaspace ~replacement ~add_prefix_space tokens =
  List.mapi
    (fun i token ->
      let s = String.map (fun c -> if c = replacement then ' ' else c) token in
      if add_prefix_space && i = 0 && String.length s > 0 && s.[0] = ' ' then
        String.sub s 1 (String.length s - 1)
      else s)
    tokens

let decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens =
  let rec dedup acc = function
    | [] -> List.rev acc
    | [ x ] -> List.rev (x :: acc)
    | x :: (y :: _ as rest) ->
        if String.equal x y then dedup acc rest else dedup (x :: acc) rest
  in
  let re =
    if cleanup then Some (Re.compile (Re.str word_delimiter_token)) else None
  in
  dedup [] tokens
  |> List.filter_map (fun token ->
      if String.equal token pad_token then None
      else
        let s =
          match re with
          | Some re -> Re.replace_string re ~by:" " token
          | None -> token
        in
        if String.equal s "" then None else Some s)

let decode_replace ~pattern ~replacement tokens =
  [ replace_all ~pattern ~by:replacement (String.concat "" tokens) ]

let strip_token ~left ~right content token =
  let len = String.length token in
  let start =
    if left then
      let rec find i =
        if i < len && Char.equal token.[i] content then find (i + 1) else i
      in
      find 0
    else 0
  in
  let stop =
    if right then
      let rec find i =
        if i >= 0 && Char.equal token.[i] content then find (i - 1) else i + 1
      in
      find (len - 1)
    else len
  in
  if start < stop then String.sub token start (stop - start) else ""

let rec decode_chain decoder tokens =
  match decoder with
  | BPE { suffix } -> decode_bpe ~suffix tokens
  | Byte_level -> [ decode_byte_level tokens ]
  | Byte_fallback -> decode_byte_fallback tokens
  | Word_piece { prefix; cleanup } ->
      [ decode_wordpiece ~prefix ~cleanup tokens ]
  | Metaspace { replacement; add_prefix_space } ->
      decode_metaspace ~replacement ~add_prefix_space tokens
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens
  | Replace { pattern; replacement } ->
      decode_replace ~pattern ~replacement tokens
  | Strip { left; right; content } ->
      [ strip_token ~left ~right content (String.concat "" tokens) ]
  | Fuse -> [ String.concat "" tokens ]
  | Sequence decoders ->
      List.fold_left (fun toks dec -> decode_chain dec toks) tokens decoders

let decode decoder tokens = String.concat "" (decode_chain decoder tokens)

(* Constructors *)

let bpe ?(suffix = "") () = BPE { suffix }
let byte_level () = Byte_level
let byte_fallback () = Byte_fallback

let wordpiece ?(prefix = "##") ?(cleanup = true) () =
  Word_piece { prefix; cleanup }

let metaspace ?(replacement = '_') ?(add_prefix_space = true) () =
  Metaspace { replacement; add_prefix_space }

let ctc ?(pad_token = "<pad>") ?(word_delimiter_token = "|") ?(cleanup = true)
    () =
  CTC { pad_token; word_delimiter_token; cleanup }

let sequence decoders = Sequence decoders
let replace ~pattern ~by () = Replace { pattern; replacement = by }

let strip ?(left = false) ?(right = false) ?(content = ' ') () =
  Strip { left; right; content }

let fuse () = Fuse

(* Formatting *)

let rec pp ppf = function
  | BPE { suffix } ->
      if suffix <> "" then Format.fprintf ppf "bpe ~suffix:%S" suffix
      else Format.fprintf ppf "bpe"
  | Byte_level -> Format.fprintf ppf "byte_level"
  | Byte_fallback -> Format.fprintf ppf "byte_fallback"
  | Word_piece { prefix; cleanup } ->
      Format.fprintf ppf "wordpiece ~prefix:%S ~cleanup:%b" prefix cleanup
  | Metaspace { replacement; add_prefix_space } ->
      Format.fprintf ppf "metaspace ~replacement:%C ~add_prefix_space:%b"
        replacement add_prefix_space
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      Format.fprintf ppf
        "ctc ~pad_token:%S ~word_delimiter_token:%S ~cleanup:%b" pad_token
        word_delimiter_token cleanup
  | Replace { pattern; replacement } ->
      Format.fprintf ppf "replace ~pattern:%S ~by:%S" pattern replacement
  | Strip { left; right; content } ->
      Format.fprintf ppf "strip ~left:%b ~right:%b ~content:%C" left right
        content
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
  | Byte_level -> json_obj [ ("type", Jsont.Json.string "Byte_level") ]
  | Byte_fallback -> json_obj [ ("type", Jsont.Json.string "Byte_fallback") ]
  | Word_piece { prefix; cleanup } ->
      json_obj
        [
          ("type", Jsont.Json.string "Word_piece");
          ("prefix", Jsont.Json.string prefix);
          ("cleanup", Jsont.Json.bool cleanup);
        ]
  | Metaspace { replacement; add_prefix_space } ->
      json_obj
        [
          ("type", Jsont.Json.string "Metaspace");
          ("replacement", Jsont.Json.string (String.make 1 replacement));
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
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
          ("pattern", Jsont.Json.string pattern);
          ("content", Jsont.Json.string replacement);
        ]
  | Strip { left; right; content } ->
      json_obj
        [
          ("type", Jsont.Json.string "Strip");
          ("strip_left", Jsont.Json.bool left);
          ("strip_right", Jsont.Json.bool right);
          ("content", Jsont.Json.string (String.make 1 content));
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

let char_field fields name ~default =
  match find_field fields name with
  | Some (Jsont.String (s, _)) when String.length s > 0 -> s.[0]
  | _ -> default

let rec of_json = function
  | Jsont.Object (fields, _) -> (
      let ( let* ) = Result.bind in
      match find_field fields "type" with
      | Some (Jsont.String ("BPEDecoder", _)) ->
          Ok (BPE { suffix = string_field fields "suffix" ~default:"" })
      | Some (Jsont.String (("Byte_level" | "ByteLevel"), _)) -> Ok Byte_level
      | Some (Jsont.String (("Byte_fallback" | "ByteFallback"), _)) ->
          Ok Byte_fallback
      | Some (Jsont.String (("Word_piece" | "WordPiece"), _)) ->
          Ok
            (Word_piece
               {
                 prefix = string_field fields "prefix" ~default:"##";
                 cleanup = bool_field fields "cleanup" ~default:true;
               })
      | Some (Jsont.String ("Metaspace", _)) ->
          Ok
            (Metaspace
               {
                 replacement = char_field fields "replacement" ~default:'_';
                 add_prefix_space =
                   bool_field fields "add_prefix_space" ~default:true;
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
                | _ -> Error err_replace_missing_pattern)
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
                 left = bool_field fields "strip_left" ~default:false;
                 right = bool_field fields "strip_right" ~default:false;
                 content = char_field fields "content" ~default:' ';
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
