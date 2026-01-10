(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Text normalization module matching HuggingFace tokenizers. *)

type normalized_string = {
  normalized : string;
  original : string;
  alignments : (int * int) array;
}
(** Type representing a normalized string with alignment information *)

(** Main normalizer type - abstract *)
type t =
  | Bert of {
      clean_text : bool;
      handle_chinese_chars : bool;
      strip_accents : bool option;
      lowercase : bool;
    }
  | Strip of { left : bool; right : bool }
  | StripAccents
  | NFC
  | NFD
  | NFKC
  | NFKD
  | Lowercase
  | Replace of { pattern : string; replacement : string }
  | Prepend of { prepend : string }
  | ByteLevel of { add_prefix_space : bool; use_regex : bool }
  | Sequence of t list

(** Helper functions for character checking *)
let is_whitespace c =
  match c with
  | '\t' | '\n' | '\r' -> true
  | c -> Uucp.White.is_white_space (Uchar.of_char c)

let is_control c =
  match c with
  | '\t' | '\n' | '\r' -> false
  | c ->
      let u = Uchar.of_char c in
      Uucp.Gc.general_category u = `Cc
      || Uucp.Gc.general_category u = `Cf
      || Uucp.Gc.general_category u = `Cn
      || Uucp.Gc.general_category u = `Co

let is_chinese_char c =
  let code = Char.code c in
  (code >= 0x4E00 && code <= 0x9FFF)
  || (code >= 0x3400 && code <= 0x4DBF)
  || (code >= 0x20000 && code <= 0x2A6DF)
  || (code >= 0x2A700 && code <= 0x2B73F)
  || (code >= 0x2B740 && code <= 0x2B81F)
  || (code >= 0x2B920 && code <= 0x2CEAF)
  || (code >= 0xF900 && code <= 0xFAFF)
  || (code >= 0x2F800 && code <= 0x2FA1F)

let is_combining_mark c =
  let u = Uchar.of_char c in
  Uucp.Gc.general_category u = `Mn

(** Transform operations on normalized strings *)
let transform_string str transformations =
  let buf = Buffer.create (String.length str * 2) in
  let alignments = ref [] in
  let current_pos = ref 0 in

  List.iter
    (fun (c, offset) ->
      Buffer.add_char buf c;
      alignments := (!current_pos, !current_pos + offset) :: !alignments;
      current_pos := !current_pos + 1)
    transformations;

  {
    normalized = Buffer.contents buf;
    original = str;
    alignments = Array.of_list (List.rev !alignments);
  }

(** Filter characters from a string *)
let filter_string str pred =
  let buf = Buffer.create (String.length str) in
  let alignments = ref [] in
  let orig_pos = ref 0 in

  String.iter
    (fun c ->
      if pred c then (
        Buffer.add_char buf c;
        alignments := (!orig_pos, !orig_pos + 1) :: !alignments);
      incr orig_pos)
    str;

  {
    normalized = Buffer.contents buf;
    original = str;
    alignments = Array.of_list (List.rev !alignments);
  }

(** Map characters in a string *)
let map_string str f =
  let buf = Buffer.create (String.length str) in
  let alignments = ref [] in
  let pos = ref 0 in

  String.iter
    (fun c ->
      let c' = f c in
      Buffer.add_char buf c';
      alignments := (!pos, !pos + 1) :: !alignments;
      incr pos)
    str;

  {
    normalized = Buffer.contents buf;
    original = str;
    alignments = Array.of_list (List.rev !alignments);
  }

(** Apply Unicode normalization *)
let apply_unicode_normalization form str =
  let normalized = Unicode.normalize form str in
  (* Simple alignment - can be improved *)
  let len = String.length normalized in
  let alignments = Array.init len (fun i -> (i, i + 1)) in
  { normalized; original = str; alignments }

(** BERT text cleaning *)
let do_clean_text ns =
  let ns' =
    filter_string ns.normalized (fun c ->
        let code = Char.code c in
        not (code = 0 || code = 0xfffd || is_control c))
  in
  map_string ns'.normalized (fun c -> if is_whitespace c then ' ' else c)

(** Handle Chinese characters *)
let do_handle_chinese_chars ns =
  let transformations = ref [] in
  String.iter
    (fun c ->
      if is_chinese_char c then
        transformations := (' ', 0) :: (c, 1) :: (' ', 1) :: !transformations
      else transformations := (c, 0) :: !transformations)
    ns.normalized;
  transform_string ns.original (List.rev !transformations)

(** Strip accents *)
let do_strip_accents ns =
  let ns = apply_unicode_normalization Unicode.NFD ns.normalized in
  filter_string ns.normalized (fun c -> not (is_combining_mark c))

(** Lowercase *)
let do_lowercase ns = map_string ns.normalized Char.lowercase_ascii

(** Strip whitespace *)
let strip_whitespace ns ~left ~right =
  let s = ns.normalized in
  let len = String.length s in
  let start =
    if left then
      let rec find_start i =
        if i >= len then len
        else if is_whitespace s.[i] then find_start (i + 1)
        else i
      in
      find_start 0
    else 0
  in
  let stop =
    if right then
      let rec find_stop i =
        if i < 0 then 0
        else if is_whitespace s.[i] then find_stop (i - 1)
        else i + 1
      in
      find_stop (len - 1)
    else len
  in
  let normalized = String.sub s start (stop - start) in
  let alignments = Array.sub ns.alignments start (stop - start) in
  { normalized; original = ns.original; alignments }

(** Apply regex replacement *)
let apply_replace ns ~pattern ~replacement =
  let regex = Str.regexp pattern in
  let normalized = Str.global_replace regex replacement ns.normalized in
  (* Simple alignment - can be improved *)
  let len = String.length normalized in
  let alignments =
    Array.init len (fun i -> (i, min i (String.length ns.original - 1)))
  in
  { normalized; original = ns.original; alignments }

(** Prepend string *)
let prepend_string ns ~prepend =
  if String.length ns.normalized = 0 then ns
  else
    let prepend_len = String.length prepend in
    let normalized = prepend ^ ns.normalized in
    let alignments =
      Array.concat
        [
          Array.make prepend_len (0, 1);
          Array.map
            (fun (a, b) -> (a + prepend_len, b + prepend_len))
            ns.alignments;
        ]
    in
    { normalized; original = ns.original; alignments }

(** Byte-level encoding *)
let bytes_char =
  let chars = ref [] in
  (* Add base characters *)
  for i = 0 to 255 do
    let c =
      if i < 0x21 || (i >= 0x7F && i <= 0xA0) || i = 0xAD then
        (* Map to printable Unicode characters in private use area *)
        (* We can't use Char.chr with values > 255, so use substitute chars *)
        Char.chr
          (if i < 0x21 then i + 0x41
           else if i >= 0x7F && i <= 0x9F then i - 0x7F + 0x41
           else 0x41)
      else Char.chr i
    in
    chars := (i, c) :: !chars
  done;
  List.rev !chars

let apply_byte_level ns ~add_prefix_space ~use_regex:_ =
  let bytes_map = bytes_char in
  let find_char b = List.assoc b bytes_map in

  let s =
    if
      add_prefix_space
      && String.length ns.normalized > 0
      && ns.normalized.[0] <> ' '
    then " " ^ ns.normalized
    else ns.normalized
  in

  let transformations = ref [] in
  String.iter
    (fun c ->
      let byte = Char.code c in
      transformations := (find_char byte, 0) :: !transformations)
    s;

  transform_string ns.original (List.rev !transformations)

(** Main normalization function *)
let rec normalize_impl normalizer ns =
  match normalizer with
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      let ns = if clean_text then do_clean_text ns else ns in
      let ns =
        if handle_chinese_chars then do_handle_chinese_chars ns else ns
      in
      let strip_accents_val =
        match strip_accents with Some v -> v | None -> lowercase
      in
      let ns = if strip_accents_val then do_strip_accents ns else ns in
      let ns = if lowercase then do_lowercase ns else ns in
      ns
  | Strip { left; right } -> strip_whitespace ns ~left ~right
  | StripAccents -> do_strip_accents ns
  | NFC -> apply_unicode_normalization Unicode.NFC ns.normalized
  | NFD -> apply_unicode_normalization Unicode.NFD ns.normalized
  | NFKC -> apply_unicode_normalization Unicode.NFKC ns.normalized
  | NFKD -> apply_unicode_normalization Unicode.NFKD ns.normalized
  | Lowercase -> do_lowercase ns
  | Replace { pattern; replacement } -> apply_replace ns ~pattern ~replacement
  | Prepend { prepend } -> prepend_string ns ~prepend
  | ByteLevel { add_prefix_space; use_regex } ->
      apply_byte_level ns ~add_prefix_space ~use_regex
  | Sequence normalizers ->
      List.fold_left (fun ns n -> normalize_impl n ns) ns normalizers

and of_json = function
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String ("Bert" | "BertNormalizer")) ->
          let get_bool name default =
            match List.assoc_opt name fields with
            | Some (`Bool b) -> b
            | _ -> default
          in
          let strip_accents =
            match List.assoc_opt "strip_accents" fields with
            | Some (`Bool b) -> Some b
            | Some `Null | None -> None
            | _ -> None
          in
          Bert
            {
              clean_text = get_bool "clean_text" true;
              handle_chinese_chars = get_bool "handle_chinese_chars" true;
              strip_accents;
              lowercase = get_bool "lowercase" true;
            }
      | Some (`String "Strip") ->
          let get_bool name default =
            match List.assoc_opt name fields with
            | Some (`Bool b) -> b
            | _ -> default
          in
          Strip
            {
              left = get_bool "strip_left" false;
              right = get_bool "strip_right" true;
            }
      | Some (`String "StripAccents") -> StripAccents
      | Some (`String "NFC") -> NFC
      | Some (`String "NFD") -> NFD
      | Some (`String "NFKC") -> NFKC
      | Some (`String "NFKD") -> NFKD
      | Some (`String "Lowercase") -> Lowercase
      | Some (`String "Replace") ->
          let pattern =
            match List.assoc_opt "pattern" fields with
            | Some (`Assoc [ ("String", `String p) ]) -> p
            | _ -> failwith "Invalid Replace normalizer pattern"
          in
          let replacement =
            match List.assoc_opt "content" fields with
            | Some (`String r) -> r
            | _ -> failwith "Invalid Replace normalizer content"
          in
          Replace { pattern; replacement }
      | Some (`String "Prepend") -> (
          match List.assoc_opt "prepend" fields with
          | Some (`String p) -> Prepend { prepend = p }
          | _ -> failwith "Invalid Prepend normalizer")
      | Some (`String "ByteLevel") ->
          let add_prefix_space =
            match List.assoc_opt "add_prefix_space" fields with
            | Some (`Bool b) -> b
            | _ -> false
          in
          let use_regex =
            match List.assoc_opt "use_regex" fields with
            | Some (`Bool b) -> b
            | _ -> false
          in
          ByteLevel { add_prefix_space; use_regex }
      | Some (`String "Sequence") -> (
          match List.assoc_opt "normalizers" fields with
          | Some (`List l) -> Sequence (List.map of_json l)
          | _ -> failwith "Invalid Sequence normalizer")
      | Some (`String other) ->
          failwith (Printf.sprintf "Unknown normalizer type: %s" other)
      | _ -> failwith "Invalid normalizer JSON")
  | _ -> failwith "Invalid normalizer JSON"

(** Constructors *)
let bert ?(clean_text = true) ?(handle_chinese_chars = true)
    ?(strip_accents = None) ?(lowercase = true) () =
  Bert { clean_text; handle_chinese_chars; strip_accents; lowercase }

let strip ?(left = false) ?(right = true) () = Strip { left; right }
let strip_accents () = StripAccents
let nfc () = NFC
let nfd () = NFD
let nfkc () = NFKC
let nfkd () = NFKD
let lowercase () = Lowercase
let replace ~pattern ~replacement () = Replace { pattern; replacement }
let prepend ~prepend = Prepend { prepend }

let byte_level ?(add_prefix_space = false) ?(use_regex = false) () =
  ByteLevel { add_prefix_space; use_regex }

let sequence normalizers = Sequence normalizers

(** Operations *)
let normalize t str =
  let initial =
    {
      normalized = str;
      original = str;
      alignments = Array.init (String.length str) (fun i -> (i, i + 1));
    }
  in
  normalize_impl t initial

let normalize_str t str =
  let ns = normalize t str in
  ns.normalized

(** Serialization *)
let rec to_json = function
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      `Assoc
        [
          ("type", `String "Bert");
          ("clean_text", `Bool clean_text);
          ("handle_chinese_chars", `Bool handle_chinese_chars);
          ( "strip_accents",
            match strip_accents with None -> `Null | Some b -> `Bool b );
          ("lowercase", `Bool lowercase);
        ]
  | Strip { left; right } ->
      `Assoc
        [
          ("type", `String "Strip");
          ("strip_left", `Bool left);
          ("strip_right", `Bool right);
        ]
  | StripAccents -> `Assoc [ ("type", `String "StripAccents") ]
  | NFC -> `Assoc [ ("type", `String "NFC") ]
  | NFD -> `Assoc [ ("type", `String "NFD") ]
  | NFKC -> `Assoc [ ("type", `String "NFKC") ]
  | NFKD -> `Assoc [ ("type", `String "NFKD") ]
  | Lowercase -> `Assoc [ ("type", `String "Lowercase") ]
  | Replace { pattern; replacement } ->
      `Assoc
        [
          ("type", `String "Replace");
          ("pattern", `Assoc [ ("String", `String pattern) ]);
          ("content", `String replacement);
        ]
  | Prepend { prepend } ->
      `Assoc [ ("type", `String "Prepend"); ("prepend", `String prepend) ]
  | ByteLevel { add_prefix_space; use_regex } ->
      `Assoc
        [
          ("type", `String "ByteLevel");
          ("add_prefix_space", `Bool add_prefix_space);
          ("use_regex", `Bool use_regex);
        ]
  | Sequence normalizers ->
      `Assoc
        [
          ("type", `String "Sequence");
          ("normalizers", `List (List.map to_json normalizers));
        ]
