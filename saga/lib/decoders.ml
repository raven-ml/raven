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

(* ───── Decoder Implementations ───── *)

let decode_bpe ~suffix tokens =
  let n = List.length tokens - 1 in
  List.mapi
    (fun i token ->
      let replacement = if i = n then "" else " " in
      if suffix <> "" then
        Re.replace_string (Re.compile (Re.str suffix)) ~by:replacement token
      else token)
    tokens

let decode_byte_level tokens =
  let buffer = Buffer.create 128 in
  List.iter
    (fun token ->
      let decoded = Pre_tokenizers.byte_level_decode token in
      Buffer.add_string buffer decoded)
    tokens;
  Buffer.contents buffer

let decode_byte_fallback tokens =
  (* Convert tokens like <0x61> to bytes and attempt UTF-8 decoding *)
  let rec process_tokens tokens acc byte_acc =
    match tokens with
    | [] ->
        (* Process any remaining bytes *)
        let final_tokens =
          if byte_acc = [] then acc
          else
            let bytes = Array.of_list (List.rev byte_acc) in
            let str =
              try
                (* Ensure bytes are in valid range for Char.chr *)
                let safe_chr b =
                  if b >= 0 && b <= 255 then Char.chr b else '?'
                in
                Bytes.to_string
                  (Bytes.init (Array.length bytes) (fun i -> safe_chr bytes.(i)))
              with _ ->
                (* Invalid UTF-8, use replacement character for each byte *)
                String.concat "" (List.map (fun _ -> "�") byte_acc)
            in
            acc @ [ str ]
        in
        List.rev final_tokens
    | token :: rest ->
        (* Check if token is a byte token like <0x61> *)
        if
          String.length token = 6
          && String.starts_with ~prefix:"<0x" token
          && String.ends_with ~suffix:">" token
        then
          (* Try to parse the hex value *)
          let hex_str = String.sub token 3 2 in
          match int_of_string_opt ("0x" ^ hex_str) with
          | Some byte when byte >= 0 && byte <= 255 ->
              (* Accumulate byte *)
              process_tokens rest acc (byte :: byte_acc)
          | _ ->
              (* Invalid hex, treat as regular token *)
              let new_acc =
                if byte_acc = [] then acc
                else
                  let bytes = Array.of_list (List.rev byte_acc) in
                  let str =
                    try
                      (* Ensure bytes are in valid range for Char.chr *)
                      let safe_chr b =
                        if b >= 0 && b <= 255 then Char.chr b else '?'
                      in
                      Bytes.to_string
                        (Bytes.init (Array.length bytes) (fun i ->
                             safe_chr bytes.(i)))
                    with _ ->
                      String.concat "" (List.map (fun _ -> "�") byte_acc)
                  in
                  acc @ [ str ]
              in
              process_tokens rest (token :: new_acc) []
        else
          (* Regular token - flush any accumulated bytes first *)
          let new_acc =
            if byte_acc = [] then acc
            else
              let bytes = Array.of_list (List.rev byte_acc) in
              let str =
                try
                  (* Ensure bytes are in valid range for Char.chr *)
                  let safe_chr b =
                    if b >= 0 && b <= 255 then Char.chr b else '?'
                  in
                  Bytes.to_string
                    (Bytes.init (Array.length bytes) (fun i ->
                         safe_chr bytes.(i)))
                with _ -> String.concat "" (List.map (fun _ -> "�") byte_acc)
              in
              acc @ [ str ]
          in
          process_tokens rest (token :: new_acc) []
  in
  process_tokens tokens [] []

let decode_wordpiece ~prefix ~cleanup tokens =
  let decoded =
    List.mapi
      (fun i token ->
        if i > 0 && String.starts_with ~prefix token then
          String.sub token (String.length prefix)
            (String.length token - String.length prefix)
        else if i > 0 then " " ^ token
        else token)
      tokens
    |> String.concat ""
  in
  if cleanup then
    (* Clean up tokenization artifacts *)
    decoded |> Re.replace_string (Re.compile (Re.rep1 (Re.char ' '))) ~by:" " |> String.trim
  else decoded

let decode_metaspace ~replacement ~add_prefix_space:_ tokens =
  String.concat "" tokens
  |> String.map (fun c -> if c = replacement then ' ' else c)

let decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens =
  (* Remove consecutive duplicates first *)
  let rec dedup = function
    | [] -> []
    | [ x ] -> [ x ]
    | x :: y :: rest ->
        if x = y then dedup (y :: rest) else x :: dedup (y :: rest)
  in
  let deduped = dedup tokens in
  (* Filter out pad tokens and replace word delimiter *)
  List.filter_map
    (fun token ->
      if token = pad_token then None
      else
        let replaced =
          if cleanup then
            Re.replace_string
              (Re.compile (Re.str word_delimiter_token))
              ~by:" " token
          else token
        in
        if replaced = "" then None else Some replaced)
    deduped

let decode_replace ~pattern ~replacement tokens =
  let text = String.concat "" tokens in
  Re.replace_string (Re.compile (Re.Pcre.re pattern)) ~by:replacement text

let decode_strip ~left ~right ~content tokens =
  let text = String.concat "" tokens in
  let strip_char s c =
    let len = String.length s in
    let start =
      if left then
        let rec find i = if i < len && s.[i] = c then find (i + 1) else i in
        find 0
      else 0
    in
    let stop =
      if right then
        let rec find i = if i >= 0 && s.[i] = c then find (i - 1) else i + 1 in
        find (len - 1)
      else len
    in
    if start < stop then String.sub s start (stop - start) else ""
  in
  strip_char text content

let decode_fuse tokens = String.concat "" tokens

(* ───── Main Decode Function ───── *)

let rec decode_chain decoder tokens =
  match decoder with
  | BPE { suffix } -> decode_bpe ~suffix tokens
  | Byte_level -> [ decode_byte_level tokens ]
  | Byte_fallback -> decode_byte_fallback tokens
  | Word_piece { prefix; cleanup } ->
      [ decode_wordpiece ~prefix ~cleanup tokens ]
  | Metaspace { replacement; add_prefix_space } ->
      [ decode_metaspace ~replacement ~add_prefix_space tokens ]
  | CTC { pad_token; word_delimiter_token; cleanup } ->
      decode_ctc ~pad_token ~word_delimiter_token ~cleanup tokens
  | Replace { pattern; replacement } ->
      [ decode_replace ~pattern ~replacement tokens ]
  | Strip { left; right; content } ->
      [ decode_strip ~left ~right ~content tokens ]
  | Fuse -> [ decode_fuse tokens ]
  | Sequence decoders ->
      (* Chain decoders on list of tokens *)
      List.fold_left (fun toks dec -> decode_chain dec toks) tokens decoders

let decode decoder tokens = String.concat "" (decode_chain decoder tokens)

(* ───── Constructors ───── *)

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
let replace ~pattern ~content () = Replace { pattern; replacement = content }

let strip ?(left = false) ?(right = false) ?(content = ' ') () =
  Strip { left; right; content }

let fuse () = Fuse

(* ───── Serialization ───── *)

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

let rec of_json json =
  let find fields name =
    match Jsont.Json.find_mem name fields with
    | Some (_, v) -> Some v
    | None -> None
  in
  match json with
  | Jsont.Object (fields, _) -> (
      match find fields "type" with
      | Some (Jsont.String ("BPEDecoder", _)) ->
          let suffix =
            match find fields "suffix" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> ""
          in
          BPE { suffix }
      | Some (Jsont.String (("Byte_level" | "ByteLevel"), _)) -> Byte_level
      | Some (Jsont.String (("Byte_fallback" | "ByteFallback"), _)) ->
          Byte_fallback
      | Some (Jsont.String (("Word_piece" | "WordPiece"), _)) ->
          let prefix =
            match find fields "prefix" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> "##"
          in
          let cleanup =
            match find fields "cleanup" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> true
          in
          Word_piece { prefix; cleanup }
      | Some (Jsont.String ("Metaspace", _)) ->
          let replacement =
            match find fields "replacement" with
            | Some (Jsont.String (s, _)) when String.length s > 0 -> s.[0]
            | _ -> '_'
          in
          let add_prefix_space =
            match find fields "add_prefix_space" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> true
          in
          Metaspace { replacement; add_prefix_space }
      | Some (Jsont.String ("CTC", _)) ->
          let pad_token =
            match find fields "pad_token" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> "<pad>"
          in
          let word_delimiter_token =
            match find fields "word_delimiter_token" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> "|"
          in
          let cleanup =
            match find fields "cleanup" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> true
          in
          CTC { pad_token; word_delimiter_token; cleanup }
      | Some (Jsont.String ("Replace", _)) ->
          let pattern =
            match find fields "pattern" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> failwith "Missing pattern in Replace decoder"
          in
          let replacement =
            match find fields "content" with
            | Some (Jsont.String (s, _)) -> s
            | _ -> ""
          in
          Replace { pattern; replacement }
      | Some (Jsont.String ("Strip", _)) ->
          let left =
            match find fields "strip_left" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> false
          in
          let right =
            match find fields "strip_right" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> false
          in
          let content =
            match find fields "content" with
            | Some (Jsont.String (s, _)) when String.length s > 0 -> s.[0]
            | _ -> ' '
          in
          Strip { left; right; content }
      | Some (Jsont.String ("Fuse", _)) -> Fuse
      | Some (Jsont.String ("Sequence", _)) -> (
          match find fields "decoders" with
          | Some (Jsont.Array (decs, _)) -> Sequence (List.map of_json decs)
          | _ -> failwith "Invalid Sequence decoder")
      | _ -> failwith "Unknown decoder type")
  | _ -> failwith "Invalid decoder JSON"
