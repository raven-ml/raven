(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type normalized_string = {
  normalized : string;
  original : string;
  alignments : (int * int) array;
}

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

(* ───── Character Checking ───── *)

let is_whitespace c =
  match c with
  | '\t' | '\n' | '\r' -> true
  | c -> Unicode_data.is_white_space (Char.code c)

let is_control c =
  match c with
  | '\t' | '\n' | '\r' -> false
  | c -> (
      let u = Uchar.of_char c in
      match Unicode_data.general_category (Uchar.to_int u) with
      | `Cc | `Cf | `Cn | `Co -> true
      | _ -> false)

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
  Unicode_data.general_category (Uchar.to_int u) = `Mn

(* ───── Transform Operations ───── *)

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

let apply_unicode_normalization form str =
  let normalized = Unicode.normalize form str in
  (* Simple alignment - can be improved *)
  let len = String.length normalized in
  let alignments = Array.init len (fun i -> (i, i + 1)) in
  { normalized; original = str; alignments }

(* ───── BERT Normalization ───── *)

let do_clean_text ns =
  let ns' =
    filter_string ns.normalized (fun c ->
        let code = Char.code c in
        not (code = 0 || code = 0xfffd || is_control c))
  in
  map_string ns'.normalized (fun c -> if is_whitespace c then ' ' else c)

let do_handle_chinese_chars ns =
  let transformations = ref [] in
  String.iter
    (fun c ->
      if is_chinese_char c then
        transformations := (' ', 0) :: (c, 1) :: (' ', 1) :: !transformations
      else transformations := (c, 0) :: !transformations)
    ns.normalized;
  transform_string ns.original (List.rev !transformations)

let do_strip_accents ns =
  let ns = apply_unicode_normalization Unicode.NFD ns.normalized in
  filter_string ns.normalized (fun c -> not (is_combining_mark c))

let do_lowercase ns = map_string ns.normalized Char.lowercase_ascii

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

let apply_replace ns ~pattern ~replacement =
  let regex = Re.compile (Re.Pcre.re pattern) in
  let normalized = Re.replace_string regex ~by:replacement ns.normalized in
  (* Simple alignment - can be improved *)
  let len = String.length normalized in
  let alignments =
    Array.init len (fun i -> (i, min i (String.length ns.original - 1)))
  in
  { normalized; original = ns.original; alignments }

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

(* ───── Byte-Level Encoding ───── *)

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

(* ───── Normalization Implementation ───── *)

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

and of_json json =
  let json_mem name = function
    | Jsont.Object (mems, _) -> (
        match Jsont.Json.find_mem name mems with
        | Some (_, v) -> v
        | None -> Jsont.Null ((), Jsont.Meta.none))
    | _ -> Jsont.Null ((), Jsont.Meta.none)
  in
  match json with
  | Jsont.Object (fields, _) -> (
      let find name =
        match Jsont.Json.find_mem name fields with
        | Some (_, v) -> Some v
        | None -> None
      in
      match find "type" with
      | Some (Jsont.String (("Bert" | "BertNormalizer"), _)) ->
          let get_bool name default =
            match find name with Some (Jsont.Bool (b, _)) -> b | _ -> default
          in
          let strip_accents =
            match find "strip_accents" with
            | Some (Jsont.Bool (b, _)) -> Some b
            | Some (Jsont.Null _) | None -> None
            | _ -> None
          in
          Bert
            {
              clean_text = get_bool "clean_text" true;
              handle_chinese_chars = get_bool "handle_chinese_chars" true;
              strip_accents;
              lowercase = get_bool "lowercase" true;
            }
      | Some (Jsont.String ("Strip", _)) ->
          let get_bool name default =
            match find name with Some (Jsont.Bool (b, _)) -> b | _ -> default
          in
          Strip
            {
              left = get_bool "strip_left" false;
              right = get_bool "strip_right" true;
            }
      | Some (Jsont.String ("StripAccents", _)) -> StripAccents
      | Some (Jsont.String ("NFC", _)) -> NFC
      | Some (Jsont.String ("NFD", _)) -> NFD
      | Some (Jsont.String ("NFKC", _)) -> NFKC
      | Some (Jsont.String ("NFKD", _)) -> NFKD
      | Some (Jsont.String ("Lowercase", _)) -> Lowercase
      | Some (Jsont.String ("Replace", _)) ->
          let pattern =
            match json_mem "pattern" json with
            | Jsont.Object (pattern_fields, _) -> (
                match Jsont.Json.find_mem "String" pattern_fields with
                | Some (_, Jsont.String (p, _)) -> p
                | _ -> failwith "Invalid Replace normalizer pattern")
            | _ -> failwith "Invalid Replace normalizer pattern"
          in
          let replacement =
            match json_mem "content" json with
            | Jsont.String (r, _) -> r
            | _ -> failwith "Invalid Replace normalizer content"
          in
          Replace { pattern; replacement }
      | Some (Jsont.String ("Prepend", _)) -> (
          match find "prepend" with
          | Some (Jsont.String (p, _)) -> Prepend { prepend = p }
          | _ -> failwith "Invalid Prepend normalizer")
      | Some (Jsont.String ("ByteLevel", _)) ->
          let add_prefix_space =
            match find "add_prefix_space" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> false
          in
          let use_regex =
            match find "use_regex" with
            | Some (Jsont.Bool (b, _)) -> b
            | _ -> false
          in
          ByteLevel { add_prefix_space; use_regex }
      | Some (Jsont.String ("Sequence", _)) -> (
          match find "normalizers" with
          | Some (Jsont.Array (l, _)) -> Sequence (List.map of_json l)
          | _ -> failwith "Invalid Sequence normalizer")
      | Some (Jsont.String (other, _)) ->
          failwith (Printf.sprintf "Unknown normalizer type: %s" other)
      | _ -> failwith "Invalid normalizer JSON")
  | _ -> failwith "Invalid normalizer JSON"

(* ───── Constructors ───── *)

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

(* ───── Operations ───── *)

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

(* ───── Serialization ───── *)

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let rec to_json = function
  | Bert { clean_text; handle_chinese_chars; strip_accents; lowercase } ->
      json_obj
        [
          ("type", Jsont.Json.string "Bert");
          ("clean_text", Jsont.Json.bool clean_text);
          ("handle_chinese_chars", Jsont.Json.bool handle_chinese_chars);
          ( "strip_accents",
            match strip_accents with
            | None -> Jsont.Json.null ()
            | Some b -> Jsont.Json.bool b );
          ("lowercase", Jsont.Json.bool lowercase);
        ]
  | Strip { left; right } ->
      json_obj
        [
          ("type", Jsont.Json.string "Strip");
          ("strip_left", Jsont.Json.bool left);
          ("strip_right", Jsont.Json.bool right);
        ]
  | StripAccents -> json_obj [ ("type", Jsont.Json.string "StripAccents") ]
  | NFC -> json_obj [ ("type", Jsont.Json.string "NFC") ]
  | NFD -> json_obj [ ("type", Jsont.Json.string "NFD") ]
  | NFKC -> json_obj [ ("type", Jsont.Json.string "NFKC") ]
  | NFKD -> json_obj [ ("type", Jsont.Json.string "NFKD") ]
  | Lowercase -> json_obj [ ("type", Jsont.Json.string "Lowercase") ]
  | Replace { pattern; replacement } ->
      json_obj
        [
          ("type", Jsont.Json.string "Replace");
          ("pattern", json_obj [ ("String", Jsont.Json.string pattern) ]);
          ("content", Jsont.Json.string replacement);
        ]
  | Prepend { prepend } ->
      json_obj
        [
          ("type", Jsont.Json.string "Prepend");
          ("prepend", Jsont.Json.string prepend);
        ]
  | ByteLevel { add_prefix_space; use_regex } ->
      json_obj
        [
          ("type", Jsont.Json.string "ByteLevel");
          ("add_prefix_space", Jsont.Json.bool add_prefix_space);
          ("use_regex", Jsont.Json.bool use_regex);
        ]
  | Sequence normalizers ->
      json_obj
        [
          ("type", Jsont.Json.string "Sequence");
          ("normalizers", Jsont.Json.list (List.map to_json normalizers));
        ]
